use tars_base::{
    AssistantContent, CancelToken, Context, Message, Model, StopReason, StreamEvent, StreamOptions,
    ToolCall, ToolResultMessage,
};
use tars_plugin::ToolExecutor;

use crate::ProviderRegistry;
use crate::retry::{classify_error, retry_countdown};
use tars_base::AgentPhase;

// ---------------------------------------------------------------------------
// Config / Result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Hard turn limit — loop exits with `max_turns_reached = true` when hit.
    pub turn_limit: usize,
    /// Maximum retries for transient provider errors.
    pub max_retries: usize,
    /// Base delay for exponential backoff in milliseconds.
    pub retry_base_ms: u64,
    /// Maximum retry delay cap in milliseconds.
    pub max_retry_delay_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            turn_limit: 32,
            max_retries: 3,
            retry_base_ms: 500,
            max_retry_delay_ms: 32_000,
        }
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

        // Stream with retry on typed transient errors
        #[allow(unused_assignments)]
        let mut terminal: Option<(StopReason, tars_base::AssistantMessage)> = None;
        let mut attempt: usize = 0;
        loop {
            let stream_res = registry.stream(model, context, options).await;
            match stream_res {
                Err(err) => {
                    if let Some(delay) =
                        classify_error(&err, attempt, config.max_retries, config.retry_base_ms)
                    {
                        let kind = match &err {
                            tars_base::Error::RateLimited { .. } => "rate limited",
                            tars_base::Error::Timeout(_) => "timeout",
                            _ => "retryable error",
                        };
                        let _ = event_tx
                            .send(StreamEvent::Phase {
                                phase: AgentPhase::RateLimited,
                                turn_started_at_ms: None,
                                phase_started_at_ms: None,
                            })
                            .await;
                        match retry_countdown(
                            delay,
                            attempt + 1,
                            config.max_retries + 1,
                            kind,
                            &err.to_string(),
                            event_tx,
                            cancel,
                        )
                        .await
                        {
                            Ok(()) => {
                                attempt += 1;
                                continue;
                            }
                            Err(tars_base::Error::Cancelled) => {
                                return Ok(AgentResult {
                                    new_messages,
                                    reason: StopReason::Aborted,
                                    max_turns_reached: false,
                                });
                            }
                            Err(e) => return Err(e),
                        }
                    } else {
                        return Err(err);
                    }
                }
                Ok(mut rx) => {
                    // Forward provider events, capture terminal message
                    let mut inner_terminal: Option<(StopReason, tars_base::AssistantMessage)> =
                        None;
                    while let Some(ev) = rx.recv().await {
                        let is_done = matches!(ev, StreamEvent::Done { .. });
                        let is_error = matches!(ev, StreamEvent::Error { .. });
                        let _ = event_tx.send(ev.clone()).await;
                        match ev {
                            StreamEvent::Done { reason, message } => {
                                inner_terminal = Some((reason, message));
                                break;
                            }
                            StreamEvent::Error { reason, error } => {
                                inner_terminal = Some((reason, error));
                                break;
                            }
                            _ => {}
                        }
                        if cancel.is_cancelled() {
                            break;
                        }
                        if is_done || is_error {
                            break;
                        }
                    }
                    terminal = inner_terminal;
                    break;
                }
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
            &AgentConfig {
                turn_limit: 5,
                ..Default::default()
            },
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
            &AgentConfig {
                turn_limit: 2,
                ..Default::default()
            },
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

    // -- retry injection --

    struct FlakyProvider {
        fails_remaining: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        err: tars_base::Error,
        success: MockResponse,
    }

    #[async_trait::async_trait]
    impl crate::Provider for FlakyProvider {
        fn api_id(&self) -> &str {
            "mock"
        }
        async fn stream(
            &self,
            model: &Model,
            context: &Context,
            options: &StreamOptions,
        ) -> tars_base::Result<crate::EventReceiver> {
            let prev = self.fails_remaining.fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |n| if n > 0 { Some(n - 1) } else { None },
            );
            if prev.is_ok() {
                return Err(match &self.err {
                    tars_base::Error::RateLimited {
                        provider,
                        retry_after,
                    } => tars_base::Error::RateLimited {
                        provider: provider.clone(),
                        retry_after: *retry_after,
                    },
                    other => tars_base::Error::Internal(other.to_string()),
                });
            }
            // success path: delegate to a one-shot MockProvider
            let p = MockProvider::new(vec![self.success.clone()]);
            p.stream(model, context, options).await
        }
    }

    #[tokio::test]
    async fn retry_recovers_from_transient_rate_limited() {
        let fails = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(2));
        let flaky = FlakyProvider {
            fails_remaining: fails.clone(),
            err: tars_base::Error::RateLimited {
                provider: "mock".into(),
                retry_after: Some(0),
            },
            success: MockResponse::Text("recovered".into()),
        };
        let mut registry = ProviderRegistry::new();
        registry.register(flaky);
        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig {
                max_retries: 3,
                retry_base_ms: 10,
                ..Default::default()
            },
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap();
        assert_eq!(result.reason, StopReason::Stop);
        assert_eq!(fails.load(std::sync::atomic::Ordering::SeqCst), 0);
        // should have emitted at least one RateLimited phase and status
        let mut saw_rate_limited = false;
        let mut saw_status = false;
        while let Ok(ev) = event_rx.try_recv() {
            if matches!(
                ev,
                StreamEvent::Phase {
                    phase: AgentPhase::RateLimited,
                    ..
                }
            ) {
                saw_rate_limited = true;
            }
            if matches!(ev, StreamEvent::Status { .. }) {
                saw_status = true;
            }
        }
        assert!(saw_rate_limited);
        assert!(saw_status);
    }

    #[tokio::test]
    async fn retry_exhausted_returns_error() {
        let fails = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(5));
        let flaky = FlakyProvider {
            fails_remaining: fails.clone(),
            err: tars_base::Error::Internal("boom".into()),
            success: MockResponse::Text("never".into()),
        };
        let mut registry = ProviderRegistry::new();
        registry.register(flaky);
        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        let (event_tx, _) = tokio::sync::mpsc::channel(64);
        let err = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig {
                max_retries: 2,
                retry_base_ms: 10,
                ..Default::default()
            },
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(err, tars_base::Error::Internal(_)));
    }

    #[tokio::test]
    async fn retry_cancel_aborts_backoff() {
        let fails = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(10));
        let flaky = FlakyProvider {
            fails_remaining: fails,
            err: tars_base::Error::RateLimited {
                provider: "mock".into(),
                retry_after: Some(5),
            },
            success: MockResponse::Text("never".into()),
        };
        let mut registry = ProviderRegistry::new();
        registry.register(flaky);
        let mut ctx = Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        };
        let model = mock_model();
        let mut executor = EchoExecutor;
        let cancel = CancelToken::new();
        let cancel_clone = cancel.clone();
        let (event_tx, _) = tokio::sync::mpsc::channel(64);
        // cancel after 50ms
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });
        let result = run(
            &model,
            &mut ctx,
            &registry,
            &mut executor,
            &StreamOptions::default(),
            &AgentConfig {
                max_retries: 5,
                retry_base_ms: 500,
                ..Default::default()
            },
            &cancel,
            &event_tx,
            &mut |_| {},
        )
        .await
        .unwrap();
        assert_eq!(result.reason, StopReason::Aborted);
    }
}
