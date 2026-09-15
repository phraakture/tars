use std::sync::Arc;

use tars_base::{CancelToken, Context, Message, StreamOptions, UserMessage};
use tars_engine::ProviderRegistry;
use tars_engine::agent::{AgentConfig, AgentResult, needs_continuation, repair_messages};
use tars_plugin_worker::InProcessWorker;

use crate::server::SharedState;
use tars_base::protocol::Response;

pub async fn run_session_turn(
    state: Arc<SharedState>,
    registry: &ProviderRegistry,
    session_id: &str,
    user_text: &str,
    cancel: &CancelToken,
) -> tars_base::Result<AgentResult> {
    // Load session and existing messages
    let (model, system_prompt, cwd, mut messages) = {
        let db = state.db.lock().await;
        let session = db.get_session(session_id)?.ok_or_else(|| {
            tars_base::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("session {} not found", session_id),
            ))
        })?;
        let msgs = db.get_messages(session_id)?;
        (
            session.model.clone(),
            session.system_prompt.clone(),
            session.cwd.clone(),
            msgs,
        )
    };

    // Repair any orphan tool_use from a prior crash before appending the new turn
    let stubs = repair_messages(&messages);
    if !stubs.is_empty() {
        let db = state.db.lock().await;
        for stub in &stubs {
            db.append_message(session_id, stub)?;
        }
        messages.extend(stubs.clone());
    }

    // Append user message to DB and to context
    let user_msg = Message::User(UserMessage::text(user_text.to_string()));
    {
        let db = state.db.lock().await;
        db.append_message(session_id, &user_msg)?;
    }
    messages.push(user_msg.clone());

    let mut context = Context {
        system_prompt,
        messages,
        tools: tars_plugin_worker::tool_schemas(&tars_plugin_worker::default_tools()),
    };

    let cwd_str = cwd.unwrap_or_else(|| "/tmp".to_string());
    let mut worker = InProcessWorker::new(cwd_str);

    let options = StreamOptions::default();
    let config = AgentConfig::default();

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
    let state_clone = state.clone();
    let forwarder = tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            let _ = state_clone.broadcast.send(Response::Stream {
                event: Box::new(ev),
            });
        }
    });

    let mut persisted_messages: Vec<Message> = Vec::new();
    let result = tars_engine::agent::run(
        &model,
        &mut context,
        registry,
        &mut worker,
        &options,
        &config,
        cancel,
        &event_tx,
        &mut |msg| {
            persisted_messages.push(msg);
        },
    )
    .await;

    drop(event_tx);
    let _ = forwarder.await;

    match result {
        Ok(agent_result) => {
            // Persist new messages (assistant + tool results) to DB
            // Note: user message already persisted, so persist only agent_result
            let db = state.db.lock().await;
            for msg in &agent_result.new_messages {
                db.append_message(session_id, msg)?;
            }
            // Broadcast terminal
            let _ = state.broadcast.send(Response::AgentDone);
            Ok(agent_result)
        }
        Err(e) => {
            let _ = state.broadcast.send(Response::Error {
                kind: tars_base::protocol::ErrorKind::Internal,
                message: e.to_string(),
            });
            Err(e)
        }
    }
}

pub async fn repair_session_if_needed(
    state: &Arc<SharedState>,
    session_id: &str,
) -> tars_base::Result<bool> {
    let messages = {
        let db = state.db.lock().await;
        db.get_messages(session_id)?
    };
    let stubs = repair_messages(&messages);
    if !stubs.is_empty() {
        let db = state.db.lock().await;
        for stub in &stubs {
            db.append_message(session_id, stub)?;
        }
        return Ok(true);
    }
    if needs_continuation(&messages) {
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use tars_base::ToolCall;
    use tars_base::{Model, ModelCost, ThinkingStyle};
    use tars_engine::providers::{MockProvider, MockResponse};

    fn mock_model() -> Model {
        Model {
            id: "mock-model".into(),
            name: "Mock".into(),
            api: "mock".into(),
            provider: "mock".into(),
            base_url: "http://mock".into(),
            thinking: ThinkingStyle::None,
            cost: ModelCost::default(),
            context_window: 100_000,
            max_tokens: 4096,
            headers: Default::default(),
        }
    }

    #[tokio::test]
    async fn chat_end_to_end_via_runner() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        // Create session
        let session_id = "s1";
        {
            let db = state.db.lock().await;
            db.create_session(&crate::db::StoredSession {
                id: session_id.into(),
                model: mock_model(),
                system_prompt: Some("you are helpful".into()),
                cwd: Some("/tmp".into()),
                created_at: tars_base::timestamp_ms() as i64,
            })
            .unwrap();
        }

        let mut registry = ProviderRegistry::new();
        registry.register(MockProvider::new(vec![MockResponse::Text(
            "hello from mock".into(),
        )]));

        let cancel = CancelToken::new();
        let mut broadcast_rx = state.broadcast.subscribe();

        let result = run_session_turn(state.clone(), &registry, session_id, "hi", &cancel)
            .await
            .unwrap();

        assert_eq!(result.reason, tars_base::StopReason::Stop);
        assert!(!result.new_messages.is_empty());

        // DB should have user + assistant
        let msgs = state.db.lock().await.get_messages(session_id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], Message::User(_)));
        assert!(matches!(msgs[1], Message::Assistant(_)));

        // Broadcast should have received Stream events and AgentDone
        let mut saw_stream = false;
        let mut saw_done = false;
        while let Ok(resp) = broadcast_rx.try_recv() {
            match resp {
                Response::Stream { .. } => saw_stream = true,
                Response::AgentDone => saw_done = true,
                _ => {}
            }
        }
        assert!(saw_stream, "should have seen Stream events");
        assert!(saw_done, "should have seen AgentDone");
    }

    #[tokio::test]
    async fn chat_with_tool_use() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let session_id = "s2";
        {
            let db = state.db.lock().await;
            db.create_session(&crate::db::StoredSession {
                id: session_id.into(),
                model: mock_model(),
                system_prompt: None,
                cwd: Some("/tmp".into()),
                created_at: tars_base::timestamp_ms() as i64,
            })
            .unwrap();
        }

        let mut registry = ProviderRegistry::new();
        registry.register(MockProvider::new(vec![
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "echo hi"}),
            }]),
            MockResponse::Text("done".into()),
        ]));
        let cancel = CancelToken::new();

        let result = run_session_turn(state.clone(), &registry, session_id, "run bash", &cancel)
            .await
            .unwrap();

        assert_eq!(result.new_messages.len(), 3); // assistant tool, tool_result, assistant text
        let msgs = state.db.lock().await.get_messages(session_id).unwrap();
        assert_eq!(msgs.len(), 4); // user + 3
    }

    #[tokio::test]
    async fn repair_killed_mid_tool_turn() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let session_id = "s3";
        {
            let db = state.db.lock().await;
            db.create_session(&crate::db::StoredSession {
                id: session_id.into(),
                model: mock_model(),
                system_prompt: None,
                cwd: Some("/tmp".into()),
                created_at: tars_base::timestamp_ms() as i64,
            })
            .unwrap();
            // Simulate a crash: assistant with ToolUse but no ToolResult
            let mut assistant = tars_base::AssistantMessage::empty("mock", "mock", "mock-model");
            assistant.stop_reason = tars_base::StopReason::ToolUse;
            assistant
                .content
                .push(tars_base::AssistantContent::ToolCall(ToolCall {
                    id: "tc_orphan".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": "sleep 10"}),
                }));
            db.append_message(session_id, &Message::Assistant(assistant))
                .unwrap();
        }

        // Repair should synthesize a stub
        let repaired = repair_session_if_needed(&state, session_id).await.unwrap();
        assert!(repaired);
        let msgs = state.db.lock().await.get_messages(session_id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(
            matches!(&msgs[1], Message::ToolResult(tr) if tr.tool_call_id == "tc_orphan" && tr.is_error)
        );
        // No dangling tool_use after repair
        let stubs = tars_engine::agent::repair_messages(&msgs);
        assert!(stubs.is_empty());

        // Now a normal Chat should see a clean history (orphan stub + new user)
        let mut registry = ProviderRegistry::new();
        registry.register(MockProvider::new(vec![MockResponse::Text(
            "recovered".into(),
        )]));
        let cancel = CancelToken::new();
        let result = run_session_turn(state.clone(), &registry, session_id, "continue", &cancel)
            .await
            .unwrap();
        assert_eq!(result.reason, tars_base::StopReason::Stop);
        let final_msgs = state.db.lock().await.get_messages(session_id).unwrap();
        // Should be: assistant orphan + stub + user continue + assistant recovered
        assert_eq!(final_msgs.len(), 4);
    }
}
