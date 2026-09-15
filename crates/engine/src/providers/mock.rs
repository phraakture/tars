//! Mock LLM provider for deterministic testing.
//!
//! Returns pre-configured responses from a queue. Each call to `stream()`
//! pops the next response and delivers it as proper stream events through
//! a tokio mpsc channel.
//!
//! ## Handle pattern
//!
//! `MockProvider` uses `Arc` internally and exposes a cheap-clone
//! [`MockProviderHandle`]. Move the provider into the agent loop, keep the
//! handle to assert captured contexts and turn count after the loop exits.
//!
//! ## Omissions vs the reference implementation
//!
//! `MockToolExecutor` and `MockToolResponse` are deferred to Phase 3
//! (the `ToolExecutor` trait milestone). MockResponse covers the
//! provider-level variants only.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::{EventReceiver, Provider, STREAM_CAPACITY};
use tars_base::{
    AssistantContent, AssistantMessage, Context, Model, ModelCost, StopReason, StreamEvent,
    StreamOptions, TextContent, ThinkingStyle, ToolCall,
};

const API_ID: &str = "mock";

// ---------------------------------------------------------------------------
// MockResponse
// ---------------------------------------------------------------------------

/// A pre-configured response for the mock provider.
#[derive(Debug, Clone)]
pub enum MockResponse {
    /// Return assistant text, then `StopReason::Stop`.
    Text(String),
    /// Return tool calls, then `StopReason::ToolUse`.
    ToolCalls(Vec<ToolCall>),
    /// Return an `Error` stream event with the given message.
    Error(String),
    /// Return partial text then error (no TextEnd/Done — simulates a
    /// mid-stream failure).
    PartialText { text: String, error: String },
    /// Wait `delay_ms` then deliver the inner response. The outer `Start`
    /// event is sent immediately; the inner response sends its own `Start`
    /// (callers should tolerate the duplicate).
    Delayed {
        delay_ms: u64,
        response: Box<MockResponse>,
    },
    /// Send one event then hang forever (for idle timeout testing).
    Hang,
}

// ---------------------------------------------------------------------------
// MockCapture
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MockCapture {
    /// Zero-based turn index (order of `stream()` calls).
    pub index: usize,
    /// The context that was passed to `stream()`.
    pub context: Context,
    /// When this turn started.
    pub timestamp: std::time::Instant,
}

// ---------------------------------------------------------------------------
// MockProvider + Handle
// ---------------------------------------------------------------------------

struct MockProviderInner {
    responses: Mutex<VecDeque<MockResponse>>,
    captures: Mutex<Vec<MockCapture>>,
}

/// Mock provider that returns pre-configured responses.
pub struct MockProvider {
    inner: Arc<MockProviderInner>,
}

/// Cheap cloneable handle to inspect mock state after the test.
#[derive(Clone)]
pub struct MockProviderHandle {
    inner: Arc<MockProviderInner>,
}

impl MockProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            inner: Arc::new(MockProviderInner {
                responses: Mutex::new(responses.into()),
                captures: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Obtain a handle for inspecting captures after the provider has been
    /// moved into the agent.
    pub fn handle(&self) -> MockProviderHandle {
        MockProviderHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl MockProviderHandle {
    /// Snapshot of all captured contexts (one per `stream()` call).
    pub fn captures(&self) -> Vec<MockCapture> {
        self.inner
            .captures
            .lock()
            .expect("captures mutex poisoned")
            .clone()
    }

    /// Duration between consecutive `stream()` calls.
    pub fn turn_durations(&self) -> Vec<std::time::Duration> {
        let caps = self.inner.captures.lock().expect("captures mutex poisoned");
        caps.windows(2)
            .map(|w| w[1].timestamp.duration_since(w[0].timestamp))
            .collect()
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn api_id(&self) -> &str {
        API_ID
    }

    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
    ) -> tars_base::Result<EventReceiver> {
        // Capture context before popping.
        {
            let mut caps = self.inner.captures.lock().expect("captures mutex poisoned");
            let index = caps.len();
            caps.push(MockCapture {
                index,
                context: context.clone(),
                timestamp: std::time::Instant::now(),
            });
        }

        let response = self
            .inner
            .responses
            .lock()
            .expect("responses mutex poisoned")
            .pop_front()
            .unwrap_or(MockResponse::Error("no more mock responses".into()));

        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CAPACITY);
        tokio::spawn(send_mock_response(tx, response));
        Ok(rx)
    }
}

// ---------------------------------------------------------------------------
// Stream event generation
// ---------------------------------------------------------------------------

fn send_mock_response(
    tx: crate::EventSender,
    response: MockResponse,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async move {
        let mut output = AssistantMessage::empty(API_ID, "mock", "mock-model");

        let _ = tx
            .send(StreamEvent::Start {
                partial: output.clone(),
            })
            .await;

        match response {
            MockResponse::Text(text) => {
                output.content.push(AssistantContent::Text(TextContent {
                    text: text.clone(),
                    text_signature: None,
                }));
                output.usage.input = 100;
                output.usage.output = text.len() as u64 / 4;
                output.stop_reason = StopReason::Stop;

                let _ = tx
                    .send(StreamEvent::TextStart {
                        content_index: 0,
                        partial: output.clone(),
                    })
                    .await;
                let _ = tx
                    .send(StreamEvent::TextDelta {
                        content_index: 0,
                        delta: text.clone(),
                        partial: output.clone(),
                    })
                    .await;
                let _ = tx
                    .send(StreamEvent::TextEnd {
                        content_index: 0,
                        content: text,
                        partial: output.clone(),
                    })
                    .await;
                let _ = tx
                    .send(StreamEvent::Done {
                        reason: StopReason::Stop,
                        message: output,
                    })
                    .await;
            }
            MockResponse::ToolCalls(calls) => {
                for (i, tc) in calls.iter().enumerate() {
                    output.content.push(AssistantContent::ToolCall(tc.clone()));
                    let _ = tx
                        .send(StreamEvent::ToolcallStart {
                            content_index: i,
                            partial: output.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(StreamEvent::ToolcallEnd {
                            content_index: i,
                            tool_call: tc.clone(),
                            partial: output.clone(),
                        })
                        .await;
                }
                output.usage.input = 100;
                output.usage.output = 50;
                output.stop_reason = StopReason::ToolUse;

                let _ = tx
                    .send(StreamEvent::Done {
                        reason: StopReason::ToolUse,
                        message: output,
                    })
                    .await;
            }
            MockResponse::Error(msg) => {
                output.stop_reason = StopReason::Error;
                output.error_message = Some(msg);
                let _ = tx
                    .send(StreamEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    })
                    .await;
            }
            MockResponse::PartialText { text, error } => {
                output.content.push(AssistantContent::Text(TextContent {
                    text: text.clone(),
                    text_signature: None,
                }));
                let _ = tx
                    .send(StreamEvent::TextStart {
                        content_index: 0,
                        partial: output.clone(),
                    })
                    .await;
                let _ = tx
                    .send(StreamEvent::TextDelta {
                        content_index: 0,
                        delta: text,
                        partial: output.clone(),
                    })
                    .await;

                output.stop_reason = StopReason::Error;
                output.error_message = Some(error);
                let _ = tx
                    .send(StreamEvent::Error {
                        reason: StopReason::Error,
                        error: output,
                    })
                    .await;
            }
            MockResponse::Delayed { delay_ms, response } => {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                // The outer Start was sent above; the inner response sends its
                // own Start — callers should tolerate the duplicate.
                send_mock_response(tx.clone(), *response).await;
            }
            MockResponse::Hang => {
                output.content.push(AssistantContent::Text(TextContent {
                    text: "hanging...".into(),
                    text_signature: None,
                }));
                let _ = tx
                    .send(StreamEvent::TextStart {
                        content_index: 0,
                        partial: output,
                    })
                    .await;
                // Park forever — the channel is dropped when the task is aborted.
                std::future::pending::<()>().await;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a minimal [`Tool`] schema for use in tests.
pub fn mock_tool(name: &str, description: &str) -> tars_base::Tool {
    tars_base::Tool {
        name: name.into(),
        description: description.into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
    }
}

/// Create a mock model for testing (api id `"mock"`).
pub fn mock_model() -> Model {
    Model {
        id: "mock-model".into(),
        name: "Mock Model".into(),
        api: API_ID.into(),
        provider: "mock".into(),
        base_url: "http://mock".into(),
        thinking: ThinkingStyle::None,
        cost: ModelCost::default(),
        context_window: 100_000,
        max_tokens: 4_096,
        headers: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::log::{LogProvider, log_model};

    #[tokio::test]
    async fn text_response_stream_events() {
        let provider = MockProvider::new(vec![MockResponse::Text("hello".into())]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        // Expect: Start, TextStart, TextDelta, TextEnd, Done
        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let text_start = rx.recv().await.unwrap();
        assert!(matches!(
            text_start,
            StreamEvent::TextStart {
                content_index: 0,
                ..
            }
        ));
        let text_delta = rx.recv().await.unwrap();
        assert!(matches!(text_delta, StreamEvent::TextDelta { delta: ref d, .. } if d == "hello"));
        let text_end = rx.recv().await.unwrap();
        assert!(matches!(text_end, StreamEvent::TextEnd { content: ref c, .. } if c == "hello"));
        let done = rx.recv().await.unwrap();
        assert!(matches!(
            done,
            StreamEvent::Done {
                reason: StopReason::Stop,
                ..
            }
        ));
        assert!(rx.recv().await.is_none(), "channel closes after Done");
    }

    #[tokio::test]
    async fn tool_calls_stream_events() {
        let tc = ToolCall {
            id: "tc_1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "Cargo.toml"}),
        };
        let provider = MockProvider::new(vec![MockResponse::ToolCalls(vec![tc.clone()])]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let tc_start = rx.recv().await.unwrap();
        assert!(matches!(
            tc_start,
            StreamEvent::ToolcallStart {
                content_index: 0,
                ..
            }
        ));
        let tc_end = rx.recv().await.unwrap();
        assert!(
            matches!(tc_end, StreamEvent::ToolcallEnd { content_index: 0, tool_call: ref t, .. } if *t == tc)
        );
        let done = rx.recv().await.unwrap();
        assert!(matches!(
            done,
            StreamEvent::Done {
                reason: StopReason::ToolUse,
                ..
            }
        ));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn error_response() {
        let provider = MockProvider::new(vec![MockResponse::Error("boom".into())]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let err = rx.recv().await.unwrap();
        assert!(
            matches!(err, StreamEvent::Error { reason: StopReason::Error, error: ref e } if e.error_message.as_deref() == Some("boom"))
        );
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn partial_text_then_error() {
        let provider = MockProvider::new(vec![MockResponse::PartialText {
            text: "partial".into(),
            error: "connection dropped".into(),
        }]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let ts = rx.recv().await.unwrap();
        assert!(matches!(ts, StreamEvent::TextStart { .. }));
        let td = rx.recv().await.unwrap();
        assert!(matches!(td, StreamEvent::TextDelta { delta: ref d, .. } if d == "partial"));
        // Then an Error (no TextEnd)
        let err = rx.recv().await.unwrap();
        assert!(
            matches!(err, StreamEvent::Error { error: ref e, .. } if e.error_message.as_deref() == Some("connection dropped"))
        );
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn delayed_response() {
        let provider = MockProvider::new(vec![MockResponse::Delayed {
            delay_ms: 50,
            response: Box::new(MockResponse::Text("late".into())),
        }]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        // Inner response sends its own Start after the delay.
        let second = rx.recv().await.unwrap();
        assert!(matches!(second, StreamEvent::Start { .. }));
        let ts = rx.recv().await.unwrap();
        assert!(matches!(ts, StreamEvent::TextStart { .. }));
        let td = rx.recv().await.unwrap();
        assert!(matches!(td, StreamEvent::TextDelta { delta: ref d, .. } if d == "late"));
        // Drain Done
        while let Some(ev) = rx.recv().await {
            if matches!(ev, StreamEvent::Done { .. }) {
                break;
            }
        }
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn hang_never_finishes() {
        let provider = MockProvider::new(vec![MockResponse::Hang]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        // Receives Start and a TextStart, then the mock parks forever.
        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let ts = rx.recv().await.unwrap();
        assert!(matches!(ts, StreamEvent::TextStart { .. }));

        // A short timeout should fire — recv() must not resolve.
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await;
        assert!(result.is_err(), "mock should block past the timeout");

        // Dropping the receiver releases the parked sender task.
        drop(rx);
    }

    #[tokio::test]
    async fn captures_recorded() {
        let provider = MockProvider::new(vec![
            MockResponse::Text("a".into()),
            MockResponse::Text("b".into()),
        ]);
        let handle = provider.handle();

        let ctx1 = Context {
            system_prompt: Some("first".into()),
            ..Context::default()
        };
        let ctx2 = Context {
            system_prompt: Some("second".into()),
            ..Context::default()
        };

        let mut rx1 = provider
            .stream(&mock_model(), &ctx1, &StreamOptions::default())
            .await
            .unwrap();
        // Drain first stream
        while rx1.recv().await.is_some() {}

        let mut rx2 = provider
            .stream(&mock_model(), &ctx2, &StreamOptions::default())
            .await
            .unwrap();
        while rx2.recv().await.is_some() {}

        let caps = handle.captures();
        assert_eq!(caps.len(), 2);
        assert_eq!(caps[0].context.system_prompt.as_deref(), Some("first"));
        assert_eq!(caps[1].context.system_prompt.as_deref(), Some("second"));
    }

    #[tokio::test]
    async fn exhausted_queue_emits_error() {
        let provider = MockProvider::new(vec![]);
        let mut rx = provider
            .stream(
                &mock_model(),
                &Context::default(),
                &StreamOptions::default(),
            )
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let err = rx.recv().await.unwrap();
        assert!(
            matches!(err, StreamEvent::Error { ref error, .. } if error.error_message.as_deref() == Some("no more mock responses"))
        );
        assert!(rx.recv().await.is_none());
    }

    #[test]
    fn log_provider_needs_no_api_key() {
        assert!(!LogProvider.needs_api_key());
        assert_eq!(LogProvider.api_id(), "log");
    }

    #[tokio::test]
    async fn log_provider_streams_start_then_done() {
        let model = log_model();
        let mut rx = LogProvider
            .stream(&model, &Context::default(), &StreamOptions::default())
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let done = rx.recv().await.unwrap();
        assert!(matches!(
            done,
            StreamEvent::Done {
                reason: StopReason::Stop,
                ..
            }
        ));
        assert!(rx.recv().await.is_none());
    }

    #[test]
    fn mock_model_and_tool_helpers() {
        let m = mock_model();
        assert_eq!(m.api, "mock");
        assert_eq!(m.context_window, 100_000);

        let t = mock_tool("bash", "run a command");
        assert_eq!(t.name, "bash");
        assert_eq!(t.description, "run a command");
        assert_eq!(
            t.parameters,
            serde_json::json!({"type":"object","properties":{},"required":[]})
        );
    }

    #[tokio::test]
    async fn registry_routes_mock_through_stream_and_captures_context() {
        use crate::ProviderRegistry;

        let mock = MockProvider::new(vec![MockResponse::Text("via registry".into())]);
        let handle = mock.handle();
        let mut registry = ProviderRegistry::new();
        registry.register(mock);
        assert!(registry.needs_api_key("mock"));

        let ctx = Context {
            system_prompt: Some("registry test".into()),
            ..Context::default()
        };
        let mut rx = registry
            .stream(&mock_model(), &ctx, &StreamOptions::default())
            .await
            .unwrap();

        // Consume the scripted stream end-to-end (Start .. Done).
        let mut saw_done = false;
        while let Some(ev) = rx.recv().await {
            if matches!(ev, StreamEvent::Done { .. }) {
                saw_done = true;
                break;
            }
        }
        assert!(saw_done);
        assert!(rx.recv().await.is_none());

        // Capture was recorded through the registry indirection.
        let caps = handle.captures();
        assert_eq!(caps.len(), 1);
        assert_eq!(
            caps[0].context.system_prompt.as_deref(),
            Some("registry test")
        );
    }

    #[tokio::test]
    async fn registry_routes_log_provider() {
        use crate::ProviderRegistry;
        use crate::providers::log::LogProvider;

        let mut registry = ProviderRegistry::new();
        registry.register(LogProvider);
        assert!(!registry.needs_api_key("log"));

        let mut rx = registry
            .stream(&log_model(), &Context::default(), &StreamOptions::default())
            .await
            .unwrap();

        let first = rx.recv().await.unwrap();
        assert!(matches!(first, StreamEvent::Start { .. }));
        let done = rx.recv().await.unwrap();
        assert!(matches!(
            done,
            StreamEvent::Done {
                reason: StopReason::Stop,
                ..
            }
        ));
        assert!(rx.recv().await.is_none());
    }
}
