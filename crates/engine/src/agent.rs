use tars_base::{
    AssistantContent, CancelToken, Context, Message, Model, StopReason, StreamEvent, StreamOptions,
    ToolCall, ToolResultMessage,
};
use tars_plugin::ToolExecutor;

use crate::ProviderRegistry;
use tars_base::AgentPhase;

// ---------------------------------------------------------------------------
// Config / Result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Hard turn limit — loop exits with `max_turns_reached = true` when hit.
    pub turn_limit: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { turn_limit: 32 }
    }
}

#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Messages produced during this run (assistant + tool results, in order).
    pub new_messages: Vec<Message>,
    /// Why the loop stopped.
    pub reason: StopReason,
    /// True if the turn limit was hit.
    pub max_turns_reached: bool,
}

// ---------------------------------------------------------------------------
// Crash recovery helpers
// ---------------------------------------------------------------------------

/// Returns true if the last message is a `ToolResult` — meaning the session
/// was interrupted after tool execution but before the LLM responded.
pub fn needs_continuation(messages: &[Message]) -> bool {
    matches!(messages.last(), Some(Message::ToolResult(_)))
}

/// Repair a history corrupted by a crash. See `tau`'s `repair_messages` for
/// rationale — we synthesize error `ToolResult` stubs for orphaned `tool_use`s.
pub fn repair_messages(messages: &[Message]) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut last_assistant_idx = None;
    for (i, msg) in messages.iter().enumerate().rev() {
        match msg {
            Message::ToolResult(_) | Message::Info(_) => continue,
            Message::Assistant(a) if a.stop_reason == StopReason::Error => continue,
            Message::Assistant(_) => {
                last_assistant_idx = Some(i);
                break;
            }
            _ => break,
        }
    }

    let Some(assistant_idx) = last_assistant_idx else {
        return Vec::new();
    };

    let assistant = match &messages[assistant_idx] {
        Message::Assistant(a) if a.stop_reason == StopReason::ToolUse => a,
        _ => return Vec::new(),
    };

    let tool_call_ids: Vec<(&str, &str)> = assistant
        .content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some((tc.id.as_str(), tc.name.as_str())),
            _ => None,
        })
        .collect();

    if tool_call_ids.is_empty() {
        return Vec::new();
    }

    let existing_result_ids: std::collections::HashSet<&str> = messages[assistant_idx + 1..]
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult(tr) => Some(tr.tool_call_id.as_str()),
            _ => None,
        })
        .collect();

    let mut stubs = Vec::new();
    for (id, name) in &tool_call_ids {
        if !existing_result_ids.contains(id) {
            stubs.push(Message::ToolResult(ToolResultMessage::error(
                *id,
                *name,
                "error: session interrupted before execution",
            )));
        }
    }

    stubs
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

/// Run the agent turn loop.
///
/// `context` is mutated in place (new assistant + tool result messages are
/// appended). Every produced `Message` is also passed to `on_message` (for
/// persistence) and every `StreamEvent` to `event_tx` (for streaming to the
/// caller). Tools are executed sequentially to keep result ordering trivial.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model: &Model,
    context: &mut Context,
    registry: &ProviderRegistry,
    executor: &mut dyn ToolExecutor,
    options: &StreamOptions,
    config: &AgentConfig,
    cancel: &CancelToken,
    event_tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    on_message: &mut dyn FnMut(Message),
) -> tars_base::Result<AgentResult> {
    let mut new_messages = Vec::new();

    for _turn in 0..config.turn_limit {
        if cancel.is_cancelled() {
            return Ok(AgentResult {
                new_messages,
                reason: StopReason::Aborted,
                max_turns_reached: false,
            });
        }

        let _ = event_tx
            .send(StreamEvent::Phase {
                phase: AgentPhase::Responding,
                turn_started_at_ms: None,
                phase_started_at_ms: None,
            })
            .await;

        let mut rx = registry.stream(model, context, options).await?;

        // Forward provider events, capture terminal message
        let mut terminal: Option<(StopReason, tars_base::AssistantMessage)> = None;
        while let Some(ev) = rx.recv().await {
            let is_done = matches!(ev, StreamEvent::Done { .. });
            let is_error = matches!(ev, StreamEvent::Error { .. });
            // forward
            let _ = event_tx.send(ev.clone()).await;
            match ev {
                StreamEvent::Done { reason, message } => {
                    terminal = Some((reason, message));
                    break;
                }
                StreamEvent::Error { reason, error } => {
                    terminal = Some((reason, error));
                    break;
                }
                _ => {}
            }
            if cancel.is_cancelled() {
                // drain quickly and abort
                // drop rx to cancel provider task
                break;
            }
            if is_done || is_error {
                break;
            }
        }

        if cancel.is_cancelled() {
            // synthesize stubs for any orphan tool_calls? For now just abort
            return Ok(AgentResult {
                new_messages,
                reason: StopReason::Aborted,
                max_turns_reached: false,
            });
        }

        let Some((stop_reason, assistant_msg)) = terminal else {
            return Err(tars_base::Error::Internal(
                "no completion from provider".into(),
            ));
        };

        let msg = Message::Assistant(assistant_msg.clone());
        on_message(msg.clone());
        new_messages.push(msg.clone());
        context.messages.push(msg);

        if stop_reason == StopReason::Error || stop_reason == StopReason::Aborted {
            return Ok(AgentResult {
                new_messages,
                reason: stop_reason,
                max_turns_reached: false,
            });
        }

        if stop_reason != StopReason::ToolUse {
            return Ok(AgentResult {
                new_messages,
                reason: stop_reason,
                max_turns_reached: false,
            });
        }

        let tool_calls: Vec<ToolCall> = assistant_msg
            .content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();

        if tool_calls.is_empty() {
            return Ok(AgentResult {
                new_messages,
                reason: stop_reason,
                max_turns_reached: false,
            });
        }

        let _ = event_tx
            .send(StreamEvent::Phase {
                phase: AgentPhase::ToolExec,
                turn_started_at_ms: None,
                phase_started_at_ms: None,
            })
            .await;

        for (idx, tc) in tool_calls.iter().enumerate() {
            if cancel.is_cancelled() {
                for remaining in &tool_calls[idx..] {
                    let stub = ToolResultMessage::error(
                        remaining.id.clone(),
                        remaining.name.clone(),
                        "error: cancelled before execution",
                    );
                    let _ = event_tx
                        .send(StreamEvent::ToolResult {
                            tool_call_id: remaining.id.clone(),
                            tool_name: remaining.name.clone(),
                            is_error: true,
                            content: stub.text(),
                            summary: None,
                        })
                        .await;
                    let m = Message::ToolResult(stub.clone());
                    on_message(m.clone());
                    new_messages.push(m.clone());
                    context.messages.push(m);
                }
                return Ok(AgentResult {
                    new_messages,
                    reason: StopReason::Aborted,
                    max_turns_reached: false,
                });
            }

            let (output_tx, mut output_rx) = tokio::sync::mpsc::channel::<String>(32);
            let event_tx_clone = event_tx.clone();
            let tc_id = tc.id.clone();
            let fwd = tokio::spawn(async move {
                while let Some(delta) = output_rx.recv().await {
                    let _ = event_tx_clone
                        .send(StreamEvent::ToolOutputDelta {
                            tool_call_id: tc_id.clone(),
                            delta,
                        })
                        .await;
                }
            });

            let result = executor.execute(tc, &output_tx, cancel).await;
            drop(output_tx);
            let _ = fwd.await;

            let tool_result = match result {
                Ok(r) => r,
                Err(e) => ToolResultMessage::error(tc.id.clone(), tc.name.clone(), e.to_string()),
            };

            let _ = event_tx
                .send(StreamEvent::ToolResult {
                    tool_call_id: tool_result.tool_call_id.clone(),
                    tool_name: tool_result.tool_name.clone(),
                    is_error: tool_result.is_error,
                    content: tool_result.text(),
                    summary: tool_result.summary.clone(),
                })
                .await;

            let m = Message::ToolResult(tool_result.clone());
            on_message(m.clone());
            new_messages.push(m.clone());
            context.messages.push(m);
        }
    }

    Ok(AgentResult {
        new_messages,
        reason: StopReason::Stop,
        max_turns_reached: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::{Context, Model, ModelCost, ThinkingStyle, Tool, UserMessage};
    use tars_plugin::ToolExecutor;

    use crate::ProviderRegistry;
    use crate::providers::{MockProvider, MockResponse};

    struct EchoExecutor;

    #[async_trait::async_trait]
    impl ToolExecutor for EchoExecutor {
        async fn execute(
            &mut self,
            tool_call: &ToolCall,
            _output_tx: &tokio::sync::mpsc::Sender<String>,
            _cancel: &CancelToken,
        ) -> tars_base::Result<ToolResultMessage> {
            Ok(ToolResultMessage::success(
                tool_call.id.clone(),
                tool_call.name.clone(),
                format!("echo: {}", tool_call.arguments),
            ))
        }
    }

    struct FailExecutor;

    #[async_trait::async_trait]
    impl ToolExecutor for FailExecutor {
        async fn execute(
            &mut self,
            tool_call: &ToolCall,
            _output_tx: &tokio::sync::mpsc::Sender<String>,
            _cancel: &CancelToken,
        ) -> tars_base::Result<ToolResultMessage> {
            Err(tars_base::Error::Internal(format!(
                "fail tool {}",
                tool_call.name
            )))
        }
    }

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
    async fn loop_tool_then_stop() {
        let mock = MockProvider::new(vec![
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"cmd":"ls"}),
            }]),
            MockResponse::Text("done".into()),
        ]);
        let handle = mock.handle();
        let mut registry = ProviderRegistry::new();
        registry.register(mock);

        let mut ctx = Context {
            system_prompt: Some("you are helpful".into()),
            messages: vec![Message::User(UserMessage::text("hi"))],
            tools: vec![Tool {
                name: "bash".into(),
                description: "run bash".into(),
                parameters: serde_json::json!({"type":"object"}),
            }],
        };

        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
        let mut persisted = Vec::new();

        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig { turn_limit: 5 },
            &cancel,
            &event_tx,
            &mut |m| persisted.push(m),
        )
        .await
        .unwrap();

        assert_eq!(result.new_messages.len(), 3); // assistant tool, tool_result, assistant text
        assert_eq!(result.reason, StopReason::Stop);
        assert!(!result.max_turns_reached);
        assert_eq!(ctx.messages.len(), 4); // user + 3 new
        // persisted matches new_messages
        assert_eq!(persisted, result.new_messages);

        // Provider saw correct contexts: first turn had 1 user msg, second had user+assistant+tool_result
        let caps = handle.captures();
        assert_eq!(caps.len(), 2);
        assert_eq!(caps[0].context.messages.len(), 1);
        assert_eq!(caps[1].context.messages.len(), 3);

        // events include tool result
        let mut saw_tool_result = false;
        let mut events = Vec::new();
        while let Ok(ev) = event_rx.try_recv() {
            events.push(ev);
        }
        for ev in &events {
            if matches!(ev, StreamEvent::ToolResult { .. }) {
                saw_tool_result = true;
            }
        }
        assert!(saw_tool_result);
    }

    #[tokio::test]
    async fn loop_stops_at_turn_limit() {
        let mock = MockProvider::new(vec![
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            }]),
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc2".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            }]),
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc3".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            }]),
        ]);
        let mut registry = ProviderRegistry::new();
        registry.register(mock);

        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("loop"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        let (event_tx, _) = tokio::sync::mpsc::channel(64);

        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig { turn_limit: 2 },
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap();

        assert!(result.max_turns_reached);
        assert_eq!(result.new_messages.len(), 4); // 2 turns * (assistant + tool_result)
    }

    #[tokio::test]
    async fn executor_error_becomes_tool_error() {
        let mock = MockProvider::new(vec![
            MockResponse::ToolCalls(vec![ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            }]),
            MockResponse::Text("recovered".into()),
        ]);
        let mut registry = ProviderRegistry::new();
        registry.register(mock);

        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = FailExecutor;
        let cancel = CancelToken::new();
        let (event_tx, _) = tokio::sync::mpsc::channel(64);

        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig::default(),
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap();

        // First tool result should be error
        assert!(matches!(&result.new_messages[1], Message::ToolResult(tr) if tr.is_error));
        assert_eq!(result.reason, StopReason::Stop);
    }

    #[test]
    fn needs_continuation_and_repair() {
        assert!(!needs_continuation(&[]));
        let msgs = vec![Message::User(UserMessage::text("hi"))];
        assert!(!needs_continuation(&msgs));
        let msgs2 = vec![
            Message::User(UserMessage::text("hi")),
            Message::ToolResult(ToolResultMessage::success("tc1", "bash", "ok")),
        ];
        assert!(needs_continuation(&msgs2));

        // repair: orphan tool_use
        let mut a = tars_base::AssistantMessage::empty("mock", "mock", "mock");
        a.stop_reason = StopReason::ToolUse;
        a.content
            .push(tars_base::AssistantContent::ToolCall(ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            }));
        let msgs = vec![Message::Assistant(a)];
        let stubs = repair_messages(&msgs);
        assert_eq!(stubs.len(), 1);
        assert!(matches!(&stubs[0], Message::ToolResult(tr) if tr.tool_call_id == "tc1"));
    }

    #[tokio::test]
    async fn cancel_mid_loop_aborts() {
        let mock = MockProvider::new(vec![MockResponse::Text("hi".into())]);
        let mut registry = ProviderRegistry::new();
        registry.register(mock);
        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        cancel.cancel();
        let (event_tx, _) = tokio::sync::mpsc::channel(64);
        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig::default(),
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap();
        assert_eq!(result.reason, StopReason::Aborted);
        assert!(result.new_messages.is_empty());
    }
}
