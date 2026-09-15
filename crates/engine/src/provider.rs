//! Provider: the seam through which model inference enters the system.
//!
//! Each LLM API (Anthropic, OpenAI-compatible, mock, log) implements
//! [`Provider`] and is registered in a [`ProviderRegistry`] keyed by its
//! [`api_id`](Provider::api_id) — for example `"anthropic-messages"`,
//! `"openai-completions"`, `"mock"`, `"log"`.
//!
//! ## Why an async-boundary + channel receiver?
//!
//! [`Provider::stream`] is genuinely async and returns an
//! [`EventReceiver`] (`tokio::sync::mpsc::Receiver<StreamEvent>`) instead of
//! streaming inline. This is a deliberate divergence from the reference
//! implementation, which marks a *synchronous* trait `#[async_trait]` and
//! streams on child threads:
//!
//! - the agent loop can `await` the returned future normally and pull events
//!   with `recv().await`;
//! - providers spawn a tokio task to fetch + parse, so dropping the receiver
//!   (agent cancelled / turn aborted) cancels the whole request cleanly with
//!   no thread leak;
//! - providers stay completely decoupled from the loop — they just emit
//!   [`StreamEvent`]s into the channel, including error events.
//!
//! `#[async_trait]` is required here (not native `async fn`) because the
//! registry must store `Arc<dyn Provider>` for dynamic dispatch, and
//! dyn-compatible RPITIT (async fn in traits) is not yet stable. The macro
//! boxes the returned future once at the trait boundary; the *body* is still
//! genuinely asynchronous.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use tars_base::{Context, Error, Model, Result, StreamEvent, StreamOptions};

/// Receiver end of a stream of events from an LLM provider.
pub type EventReceiver = tokio::sync::mpsc::Receiver<StreamEvent>;

/// Sender end — used by provider implementations to emit events.
pub type EventSender = tokio::sync::mpsc::Sender<StreamEvent>;

/// Default channel capacity for provider event streams.
pub const STREAM_CAPACITY: usize = 64;

/// Trait implemented by each LLM API provider (Anthropic, OpenAI, mock, …).
#[async_trait]
pub trait Provider: Send + Sync {
    /// Identifier for this API, e.g. `"anthropic-messages"`.
    fn api_id(&self) -> &str;

    /// Whether this provider requires an API key to run a turn.
    ///
    /// Real LLM providers return `true` (the default). No-op providers such
    /// as the log provider — which never makes an outbound call — override
    /// this to `false` so the agent runner skips the key preflight.
    fn needs_api_key(&self) -> bool {
        true
    }

    /// Start streaming a completion for `context` using `model`.
    ///
    /// Returns immediately with a channel receiver; all output — text
    /// deltas, tool calls, thinking, usage, terminal `Done`/`Error` —
    /// arrives through the returned channel as it is produced by the
    /// provider's own async task.
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventReceiver>;
}

/// Simple registry mapping API id → provider.
#[derive(Default, Clone)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider under its `api_id()`, replacing any previous
    /// provider with the same id.
    pub fn register(&mut self, provider: impl Provider + 'static) {
        let api_id = provider.api_id().to_string();
        self.providers.insert(api_id, Arc::new(provider));
    }

    /// Look up the provider registered for `api`.
    pub fn get(&self, api: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(api).cloned()
    }

    /// Convenience: whether the provider registered for `api` needs an API
    /// key. Returns `true` when `api` isn't registered (conservative
    /// default — the caller fails loudly downstream with
    /// [`Error::NoProvider`]).
    pub fn needs_api_key(&self, api: &str) -> bool {
        self.get(api).map(|p| p.needs_api_key()).unwrap_or(true)
    }

    /// Stream a completion using the provider registered for `model.api`.
    pub async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventReceiver> {
        let provider = self
            .providers
            .get(&model.api)
            .cloned()
            .ok_or_else(|| Error::NoProvider(model.api.clone()))?;
        provider.stream(model, context, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tars_base::{AssistantMessage, ModelCost, StopReason, ThinkingStyle};

    struct TestProvider {
        api_id: &'static str,
        needs_key: bool,
        /// Records the last `(model, context, options)` seen by `stream`.
        seen: Arc<Mutex<Option<(Model, Context, StreamOptions)>>>,
    }

    impl TestProvider {
        fn new(api_id: &'static str, needs_key: bool) -> Self {
            Self {
                api_id,
                needs_key,
                seen: Arc::new(Mutex::new(None)),
            }
        }
    }

    #[async_trait]
    impl Provider for TestProvider {
        fn api_id(&self) -> &str {
            self.api_id
        }

        fn needs_api_key(&self) -> bool {
            self.needs_key
        }

        async fn stream(
            &self,
            model: &Model,
            context: &Context,
            options: &StreamOptions,
        ) -> Result<EventReceiver> {
            *self.seen.lock().unwrap() = Some((model.clone(), context.clone(), options.clone()));

            let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CAPACITY);
            let start = StreamEvent::Start {
                partial: AssistantMessage::empty("mock", "mock", &model.id),
            };
            let done = StreamEvent::Done {
                reason: StopReason::Stop,
                message: AssistantMessage::empty("mock", "mock", &model.id),
            };
            tx.send(start).await.map_err(|_| Error::ChannelClosed)?;
            tx.send(done).await.map_err(|_| Error::ChannelClosed)?;
            Ok(rx)
        }
    }

    fn sample_model(api: &str) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test".into(),
            api: api.into(),
            provider: "test".into(),
            base_url: "http://localhost".into(),
            thinking: ThinkingStyle::None,
            cost: ModelCost::default(),
            context_window: 100_000,
            max_tokens: 4_096,
            headers: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn registry_dispatches_by_api_id() {
        let provider = TestProvider::new("mock", false);
        let seen = provider.seen.clone();
        let mut reg = ProviderRegistry::new();
        reg.register(provider);

        let model = sample_model("mock");
        let context = Context {
            system_prompt: Some("You are a test".into()),
            ..Context::default()
        };

        let mut rx = reg
            .stream(&model, &context, &StreamOptions::default())
            .await
            .unwrap();

        // The registered provider saw our exact context.
        {
            let binding = seen.lock().unwrap();
            let (seen_model, seen_ctx, _) = binding.as_ref().expect("stream ran");
            assert_eq!(seen_model.id, "test-model");
            assert_eq!(seen_ctx.system_prompt.as_deref(), Some("You are a test"));
        }

        // And its events flow through the channel untouched.
        let start = rx.recv().await.unwrap();
        assert!(matches!(start, StreamEvent::Start { .. }));
        let done = rx.recv().await.unwrap();
        assert!(matches!(done, StreamEvent::Done { .. }));
        assert!(rx.recv().await.is_none(), "channel must close after done");
    }

    #[tokio::test]
    async fn registry_unknown_api_returns_no_provider() {
        let reg = ProviderRegistry::new();
        let model = sample_model("not-registered");
        let err = reg
            .stream(&model, &Context::default(), &StreamOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NoProvider(a) if a == "not-registered"));
    }

    #[test]
    fn needs_api_key_defaults_true_for_unknown_api() {
        let reg = ProviderRegistry::new();
        assert!(reg.needs_api_key("definitely-not-registered"));
    }

    #[test]
    fn needs_api_key_reports_registered_value() {
        let mut reg = ProviderRegistry::new();
        reg.register(TestProvider::new("log", false));
        reg.register(TestProvider::new("real", true));
        assert!(!reg.needs_api_key("log"));
        assert!(reg.needs_api_key("real"));
    }

    #[test]
    fn register_replaces_same_api_id() {
        let mut reg = ProviderRegistry::new();
        reg.register(TestProvider::new("mock", false));
        reg.register(TestProvider::new("mock", true));
        // the second registration replaces the first
        assert!(reg.needs_api_key("mock"));
    }
}
