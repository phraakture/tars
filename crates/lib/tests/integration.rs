//! Integration tests — full-turn e2e, subprocess plugin, crash recovery.

use std::sync::Arc;

use tars_base::protocol::Response;
use tars_base::{CancelToken, Message, Model, ModelCost, ThinkingStyle};
use tars_engine::ProviderRegistry;
use tars_engine::providers::{MockProvider, MockResponse};
use tars_lib::db::{Db, StoredSession};
use tars_lib::server::SharedState;

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

fn mock_state(db: Db) -> Arc<SharedState> {
    Arc::new(SharedState::new(db))
}

fn mock_state_with_pm(db: Db, pm: tars_lib::plugin_manager::PluginManager) -> Arc<SharedState> {
    let mut state = SharedState::new(db);
    state.plugin_manager = Some(Arc::new(tokio::sync::Mutex::new(pm)));
    Arc::new(state)
}

/// Full-turn e2e: user sends message, mock returns text, verify DB + events.
#[tokio::test]
async fn e2e_full_turn_mock() {
    let db = Db::open_memory().unwrap();
    let state = mock_state(db);
    let session_id = "e2e_mock";

    {
        let db = state.db.lock().await;
        db.create_session(&StoredSession {
            id: session_id.into(),
            model: mock_model(),
            system_prompt: Some("you are a test helper".into()),
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
    let result = tars_lib::agent_runner::run_session_turn(
        state.clone(),
        &registry,
        session_id,
        "test message",
        &cancel,
    )
    .await
    .unwrap();

    assert_eq!(result.reason, tars_base::StopReason::Stop);
    assert!(!result.new_messages.is_empty());

    // DB: user + assistant
    let msgs = state.db.lock().await.get_messages(session_id).unwrap();
    assert_eq!(msgs.len(), 2);
    assert!(matches!(msgs[0], Message::User(_)));
    assert!(matches!(msgs[1], Message::Assistant(_)));

    // Session buffer: Stream events + AgentDone
    let buffered = state.drain_session_events(session_id).await;
    let has_stream = buffered
        .iter()
        .any(|r| matches!(r, Response::Stream { .. }));
    let has_done = buffered.iter().any(|r| matches!(r, Response::AgentDone));
    assert!(has_stream, "should have Stream events");
    assert!(has_done, "should have AgentDone");
}

/// Full-turn e2e with tool use: mock returns ToolCall, then text.
#[tokio::test]
async fn e2e_tool_use_mock() {
    let db = Db::open_memory().unwrap();
    let state = mock_state(db);
    let session_id = "e2e_tool";

    {
        let db = state.db.lock().await;
        db.create_session(&StoredSession {
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
        MockResponse::ToolCalls(vec![tars_base::ToolCall {
            id: "tc_e2e".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "echo e2e_ok"}),
        }]),
        MockResponse::Text("tool done".into()),
    ]));

    let cancel = CancelToken::new();
    let result = tars_lib::agent_runner::run_session_turn(
        state.clone(),
        &registry,
        session_id,
        "run bash",
        &cancel,
    )
    .await
    .unwrap();

    // Should have: assistant tool_use, tool_result, assistant text
    assert_eq!(result.new_messages.len(), 3);

    let msgs = state.db.lock().await.get_messages(session_id).unwrap();
    assert_eq!(msgs.len(), 4); // user + 3
}

/// Subprocess plugin e2e: tool call routes through tars-worker binary.
#[tokio::test]
async fn e2e_subprocess_plugin() {
    let exe = {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.push("target/debug/tars-worker");
        path.to_string_lossy().to_string()
    };
    if !std::path::Path::new(&exe).exists() {
        eprintln!("skipping e2e_subprocess_plugin: tars-worker not built");
        return;
    }

    let mut pm = tars_lib::plugin_manager::PluginManager::new();
    pm.spawn_plugin(&[exe], "/tmp").unwrap();

    let db = Db::open_memory().unwrap();
    let state = mock_state_with_pm(db, pm);

    let session_id = "e2e_sub";
    {
        let db = state.db.lock().await;
        db.create_session(&StoredSession {
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
        MockResponse::ToolCalls(vec![tars_base::ToolCall {
            id: "tc_sub_e2e".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "echo subprocess_e2e_ok"}),
        }]),
        MockResponse::Text("subprocess done".into()),
    ]));

    let cancel = CancelToken::new();
    let result = tars_lib::agent_runner::run_session_turn(
        state.clone(),
        &registry,
        session_id,
        "run bash",
        &cancel,
    )
    .await
    .unwrap();

    assert_eq!(result.new_messages.len(), 3);

    // Verify tool result came from subprocess
    let tool_result = &result.new_messages[1];
    match tool_result {
        Message::ToolResult(tr) => {
            let text: String = tr.content.iter().map(|c| c.text().to_string()).collect();
            assert!(
                text.contains("subprocess_e2e_ok"),
                "expected subprocess output, got: {text}"
            );
        }
        other => panic!("expected ToolResult, got: {:?}", other),
    }
}

/// Crash recovery: orphan tool_use gets a stub on repair.
#[tokio::test]
async fn e2e_crash_recovery() {
    let db = Db::open_memory().unwrap();
    let state = mock_state(db);
    let session_id = "e2e_crash";

    {
        let db = state.db.lock().await;
        db.create_session(&StoredSession {
            id: session_id.into(),
            model: mock_model(),
            system_prompt: None,
            cwd: Some("/tmp".into()),
            created_at: tars_base::timestamp_ms() as i64,
        })
        .unwrap();

        // Simulate crash: assistant with ToolUse but no ToolResult
        let mut assistant = tars_base::AssistantMessage::empty("mock", "mock", "mock-model");
        assistant.stop_reason = tars_base::StopReason::ToolUse;
        assistant
            .content
            .push(tars_base::AssistantContent::ToolCall(tars_base::ToolCall {
                id: "tc_orphan_e2e".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "sleep 100"}),
            }));
        db.append_message(session_id, &Message::Assistant(assistant))
            .unwrap();
    }

    // Repair should synthesize a stub
    let repaired = tars_lib::agent_runner::repair_session_if_needed(&state, session_id)
        .await
        .unwrap();
    assert!(repaired);

    let msgs = state.db.lock().await.get_messages(session_id).unwrap();
    assert_eq!(msgs.len(), 2);
    assert!(
        matches!(&msgs[1], Message::ToolResult(tr) if tr.tool_call_id == "tc_orphan_e2e" && tr.is_error)
    );

    // No dangling tool_use after repair
    let stubs = tars_engine::agent::repair_messages(&msgs);
    assert!(stubs.is_empty());

    // Normal chat should see clean history
    let mut registry = ProviderRegistry::new();
    registry.register(MockProvider::new(vec![MockResponse::Text(
        "recovered".into(),
    )]));
    let cancel = CancelToken::new();
    let result = tars_lib::agent_runner::run_session_turn(
        state.clone(),
        &registry,
        session_id,
        "continue",
        &cancel,
    )
    .await
    .unwrap();
    assert_eq!(result.reason, tars_base::StopReason::Stop);

    let final_msgs = state.db.lock().await.get_messages(session_id).unwrap();
    assert_eq!(final_msgs.len(), 4);
}

/// Server round-trip: create session, chat, verify persistence.
#[tokio::test]
async fn e2e_server_roundtrip() {
    let db = Db::open_memory().unwrap();
    let state = Arc::new(SharedState::new(db));
    let (a, b) = tokio::net::UnixStream::pair().unwrap();
    let state_clone = state.clone();
    tokio::spawn(async move {
        tars_lib::server::handle_connection(state_clone, a).await;
    });

    let (read_half, mut write_half) = b.into_split();
    let mut reader = tokio::io::BufReader::new(read_half);

    // CreateSession
    let req = tars_base::protocol::Request::CreateSession {
        model: None,
        provider: None,
        system_prompt: None,
        cwd: None,
        parent_id: None,
        tagline: None,
    };
    tars_base::write_json_line_async(&mut write_half, &req)
        .await
        .unwrap();
    let resp: Response = tars_base::read_json_line_async(&mut reader)
        .await
        .unwrap()
        .unwrap();
    let session_id = match resp {
        Response::SessionCreated { session_id } => session_id,
        other => panic!("expected SessionCreated, got {:?}", other),
    };

    // Chat
    let req = tars_base::protocol::Request::Chat {
        session_id: session_id.clone(),
        text: "hello server".into(),
        attachments: vec![],
    };
    tars_base::write_json_line_async(&mut write_half, &req)
        .await
        .unwrap();
    let resp: Response = tars_base::read_json_line_async(&mut reader)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(resp, Response::Ok);

    // Verify DB has user + agent response
    let msgs = state.db.lock().await.get_messages(&session_id).unwrap();
    assert!(!msgs.is_empty());
    assert!(matches!(msgs[0], Message::User(_)));
}
