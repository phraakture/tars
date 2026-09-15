//! Log provider — a no-op LLM that returns an immediate end-turn response.
//!
//! Sessions using this provider can only be driven via `ExecuteTool`.
//! The agent loop exits cleanly after any tool results because the provider
//! returns a minimal `AssistantMessage` with `StopReason::Stop` (no text,
//! no tool calls).

use async_trait::async_trait;

use crate::{EventReceiver, Provider, STREAM_CAPACITY};
use tars_base::{
    AssistantMessage, Context, Model, ModelCost, StopReason, StreamEvent, StreamOptions,
    ThinkingStyle,
};

const API_ID: &str = "log";

/// No-op provider that returns an immediate end-turn with empty content.
pub struct LogProvider;

#[async_trait]
impl Provider for LogProvider {
    fn api_id(&self) -> &str {
        API_ID
    }

    fn needs_api_key(&self) -> bool {
        false
    }

    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> tars_base::Result<EventReceiver> {
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CAPACITY);
        let output = AssistantMessage::empty(API_ID, "log", "log");
        let start = StreamEvent::Start {
            partial: output.clone(),
        };
        let done = StreamEvent::Done {
            reason: StopReason::Stop,
            message: output,
        };
        tokio::spawn(async move {
            let _ = tx.send(start).await;
            let _ = tx.send(done).await;
        });
        Ok(rx)
    }
}

/// Create the built-in `log` model definition.
pub fn log_model() -> Model {
    Model {
        id: "log".into(),
        name: "Log (no LLM)".into(),
        api: API_ID.into(),
        provider: "log".into(),
        base_url: String::new(),
        thinking: ThinkingStyle::None,
        cost: ModelCost::default(),
        context_window: 1_000_000,
        max_tokens: 0,
        headers: std::collections::HashMap::new(),
    }
}
