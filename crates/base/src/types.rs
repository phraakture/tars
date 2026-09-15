//! Core data types shared across the tars workspace.
//!
//! This is the type vocabulary of the system: what a conversation looks like
//! as data (`Message` + content blocks), what tools are (`Tool`, `ToolCall`),
//! what the agent sends to the model (`Context`), and how model output is
//! streamed back (`StreamEvent`). Everything is `serde`-serializable so these
//! types can cross the wire unchanged.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::timestamp_ms;

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

/// Shared cancellation flag for tool execution.
///
/// A thin wrapper around `Arc<AtomicBool>`. Long-running tools (bash) poll
/// [`is_cancelled`](Self::is_cancelled) at intervals and abort when it becomes
/// true; the server flips the flag on Ctrl-C / cancel RPC.
///
/// Fast tools (read, write, edit) may ignore the token entirely or check it
/// once at the top to return a `cancelled` error if the user cancelled before
/// execution began. Clones share the same underlying atomic.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// Create a new, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap an existing shared flag. Useful when the server already owns an
    /// `Arc<AtomicBool>` (per-session cancel flag) and wants to expose it to
    /// tool-execution paths without re-wrapping.
    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self { flag }
    }

    /// Return the underlying shared flag (same `Arc`).
    pub fn flag(&self) -> Arc<AtomicBool> {
        self.flag.clone()
    }

    /// True if the token has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Set the cancel flag. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    /// base64-encoded bytes.
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON arguments matching the tool's `parameters` schema.
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    Text(TextContent),
    Image(ImageContent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text(TextContent),
    Image(ImageContent),
}

impl ToolResultContent {
    /// Return the text if this is a Text variant, empty string otherwise.
    pub fn text(&self) -> &str {
        match self {
            Self::Text(t) => &t.text,
            Self::Image(_) => "",
        }
    }
}

// ---------------------------------------------------------------------------
// Usage & cost
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// USD.
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Token counts as reported by the provider.
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: Cost,
}

impl Usage {
    /// Recompute `total_tokens` as the sum of the four token counters.
    pub fn recompute_total(&mut self) {
        self.total_tokens = self.input + self.output + self.cache_read + self.cache_write;
    }
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model finished normally (end_turn / stop_sequence).
    Stop,
    /// Model hit the max-tokens limit.
    Length,
    /// Model emitted tool calls; the loop should dispatch them.
    ToolUse,
    /// The stream failed; `AssistantMessage::error_message` carries details.
    Error,
    /// The run was aborted (cancelled).
    Aborted,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: Vec<UserContent>,
    pub timestamp: u64,
}

impl UserMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![UserContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            timestamp: timestamp_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    /// Provider api id this message came from (e.g. "anthropic-messages").
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub timestamp: u64,
}

impl AssistantMessage {
    pub fn empty(api: &str, provider: &str, model: &str) -> Self {
        Self {
            content: Vec::new(),
            api: api.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: timestamp_ms(),
        }
    }

    /// Concatenate all text content blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage {
    /// The id of the `ToolCall` this result answers.
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// True when the tool failed; the model sees the error and can retry.
    pub is_error: bool,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Short human description of what the tool did, shown to the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Tier-2 actions to run after this tool result is persisted to the
    /// caller's session history, still inside the caller's turn.
    ///
    /// Not serialised as part of the permanent message history — these are
    /// transient side effects attached to the returned tool result and
    /// dropped once drained by the agent loop.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub post_persist_actions: Vec<PostPersistAction>,
}

/// Tier-2 actions the server performs after persisting a tool result, still
/// inside the calling session's agent loop (lock held).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PostPersistAction {
    /// Append an info message to any session's history. Use for side effects
    /// that must render after the tool result.
    EmitInfoMessage {
        target_session_id: String,
        text: String,
    },
    /// End the current agent turn cleanly (`AgentDone`, not `Cancelled`)
    /// after this tool result is persisted. Intended for tools that retire
    /// the calling session (e.g. `session_succeed`); other tools should let
    /// the agent decide when to stop on its own.
    StopAgentLoop {
        /// Free-form reason recorded in logs. Not surfaced to the model or
        /// client — the tool itself is responsible for a human-meaningful
        /// result.
        reason: String,
    },
}

/// Tier-3 actions the server performs after the calling session's lock is
/// released. Used for side effects that need exclusive access after the loop
/// has exited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PostIdleAction {
    /// Archive all archivable sessions for a task.
    ArchiveTaskSessions { task_id: i64 },
}

impl ToolResultMessage {
    pub fn success(
        id: impl Into<String>,
        name: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: id.into(),
            tool_name: name.into(),
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: timestamp_ms(),
            duration_ms: None,
            summary: None,
            post_persist_actions: Vec::new(),
        }
    }

    pub fn error(id: impl Into<String>, name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tool_call_id: id.into(),
            tool_name: name.into(),
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            is_error: true,
            timestamp: timestamp_ms(),
            duration_ms: None,
            summary: None,
            post_persist_actions: Vec::new(),
        }
    }

    /// Concatenate all text content blocks.
    pub fn text(&self) -> String {
        self.content.iter().map(|c| c.text()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionSummaryMessage {
    pub summary: String,
    /// How many tokens the context had before compaction.
    pub tokens_before: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfoMessage {
    pub text: String,
    pub timestamp: u64,
}

impl InfoMessage {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            timestamp: timestamp_ms(),
        }
    }
}

/// A single message in the conversation history, tagged by `role`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    CompactionSummary(CompactionSummaryMessage),
    Info(InfoMessage),
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelCost {
    /// USD per million tokens.
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// How a model supports extended thinking/reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingStyle {
    /// No thinking support.
    #[default]
    None,
    /// Anthropic: budget_tokens or adaptive thinking.
    Anthropic,
    /// OpenAI: `reasoning_effort` parameter.
    #[serde(alias = "openai")]
    OpenAi,
    /// Qwen (OpenAI-compat): `enable_thinking: bool`.
    Qwen,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    /// Provider api id this model runs on (e.g. "openai-completions").
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub thinking: ThinkingStyle,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

impl Model {
    /// Compute the per-token dollar cost of `usage` and store it in place.
    pub fn calculate_cost(&self, usage: &mut Usage) {
        usage.cost.input = (self.cost.input / 1_000_000.0) * usage.input as f64;
        usage.cost.output = (self.cost.output / 1_000_000.0) * usage.output as f64;
        usage.cost.cache_read = (self.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
        usage.cost.cache_write = (self.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
        usage.cost.total =
            usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    }
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

/// The LLM-facing description of a callable tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema describing `ToolCall::arguments`.
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Context (what gets sent to the LLM)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Context {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

/// Effort level for adaptive thinking (Anthropic Opus 4.6+, Sonnet 4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// How thinking content is returned in the response (Anthropic-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    /// Thinking blocks contain summarized thinking text.
    Summarized,
    /// Thinking blocks return an empty `thinking` field; the encrypted
    /// signature still travels back for multi-turn continuity.
    Omitted,
}

/// Prompt-cache retention hint.
///
/// `Short` is the default and matches OpenAI's ~5-minute prefix cache TTL.
/// `Long` requests the 24h retention tier. `None` opts out of provider-side
/// prompt caching where applicable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

impl CacheRetention {
    /// Resolve an optional retention hint into a concrete value.
    /// `None` falls back to `Short`.
    pub fn resolve(opt: Option<Self>) -> Self {
        opt.unwrap_or_default()
    }
}

/// Per-call options forwarded from the caller to the provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    /// Extended-thinking budget for the non-adaptive Anthropic path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u64>,
    /// Explicit on/off for extended thinking. `None` lets the provider decide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_enabled: Option<bool>,
    /// Effort level for adaptive thinking (Anthropic-specific for now).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,
    /// How thinking content is returned (Anthropic-specific for now).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_display: Option<ThinkingDisplay>,
    /// Opaque session identifier for provider-side prompt-cache affinity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Prompt-cache retention hint; `None` defers to the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
}

// ---------------------------------------------------------------------------
// Agent phase
// ---------------------------------------------------------------------------

/// Current phase of the agent loop, broadcast to subscribers for UI display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPhase {
    /// No agent turn running.
    #[default]
    Idle,
    /// Blocked waiting for the session lock (another turn in progress).
    Waiting,
    /// Loading the session, spawning plugins, running hooks.
    Preparing,
    /// HTTP request sent, waiting for the first SSE byte from the provider.
    Connecting,
    /// Receiving thinking tokens from the LLM.
    Thinking,
    /// Streaming text/tool-call tokens from the LLM.
    Responding,
    /// Executing tool calls.
    ToolExec,
    /// Running context compaction.
    Compacting,
    /// Waiting for rate limit / retry backoff.
    RateLimited,
}

impl AgentPhase {
    /// Human-readable label for a status line.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Waiting => "waiting...",
            Self::Preparing => "preparing...",
            Self::Connecting => "sending request...",
            Self::Thinking => "thinking...",
            Self::Responding => "working...",
            Self::ToolExec => "running tools...",
            Self::Compacting => "compacting...",
            Self::RateLimited => "rate limited...",
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// Events emitted while the agent streams a model response.
///
/// Interactive variants carry a `partial` snapshot of the
/// [`AssistantMessage`] being assembled, so clients can render accumulated
/// state at any step. The `type` tag mirrors the pi-ai project's event
/// vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolcallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    /// One fragment of JSON arguments for the tool call at `content_index`.
    ToolcallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolcallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    /// Incremental tool output line (e.g. live bash output).
    ToolOutputDelta {
        tool_call_id: String,
        delta: String,
    },
    /// Tool execution completed.
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        is_error: bool,
        /// Full text output.
        content: String,
        summary: Option<String>,
    },
    /// Stream finished naturally.
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    /// Stream failed.
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
    /// A steering message was injected mid-loop.
    SteerMessage {
        message: UserMessage,
    },
    /// Agent phase transition. `turn_started_at_ms` lets clients anchor a
    /// "Working... Xs" counter across phases; `phase_started_at_ms` anchors a
    /// per-phase elapsed counter. Both are cleared on going idle.
    Phase {
        phase: AgentPhase,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_started_at_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase_started_at_ms: Option<u64>,
    },
    /// Informational status message (e.g. retry notices).
    Status {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_content(s: &str) -> TextContent {
        TextContent {
            text: s.to_string(),
            text_signature: None,
        }
    }

    // -- usage / cost --

    #[test]
    fn usage_recompute_total_sums_fields() {
        let mut u = Usage {
            input: 10,
            output: 20,
            cache_read: 3,
            cache_write: 4,
            total_tokens: 0,
            cost: Cost::default(),
        };
        u.recompute_total();
        assert_eq!(u.total_tokens, 37);
        // idempotent when fields unchanged
        u.recompute_total();
        assert_eq!(u.total_tokens, 37);
        // overwrites stale values
        u.total_tokens = 999;
        u.recompute_total();
        assert_eq!(u.total_tokens, 37);
    }

    #[test]
    fn model_calculate_cost() {
        let model = Model {
            id: "test".into(),
            name: "Test".into(),
            api: "test-api".into(),
            provider: "test".into(),
            base_url: "http://localhost".into(),
            thinking: ThinkingStyle::None,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            context_window: 200_000,
            max_tokens: 4096,
            headers: HashMap::new(),
        };
        let mut usage = Usage {
            input: 1_000_000,
            output: 100_000,
            cache_read: 2_000_000,
            cache_write: 10_000,
            total_tokens: 0,
            cost: Cost::default(),
        };
        model.calculate_cost(&mut usage);
        assert_eq!(usage.cost.input, 3.0);
        assert_eq!(usage.cost.output, 1.5);
        assert_eq!(usage.cost.cache_read, 0.6);
        assert_eq!(usage.cost.cache_write, 0.0375);
        assert!((usage.cost.total - 5.1375).abs() < 1e-9);
    }

    // -- cancel token --

    #[test]
    fn cancel_token_defaults_uncancelled_and_cancels() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        t.cancel();
        assert!(t.is_cancelled());
        // idempotent
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn cancel_token_clones_share_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        b.cancel();
        assert!(a.is_cancelled());
    }

    #[test]
    fn cancel_token_from_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let t = CancelToken::from_flag(flag.clone());
        flag.store(true, Ordering::Relaxed);
        assert!(t.is_cancelled());
        // flag() returns the same Arc
        assert!(Arc::ptr_eq(&flag, &t.flag()));
    }

    // -- message serde round-trips --

    #[test]
    fn user_message_roundtrip() {
        let msg = Message::User(UserMessage {
            content: vec![UserContent::Text(text_content("hello"))],
            timestamp: 1234,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""type":"text""#));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
        assert!(matches!(back, Message::User(u) if u.timestamp == 1234));
    }

    #[test]
    fn assistant_message_roundtrip_with_tool_call() {
        let msg = AssistantMessage {
            content: vec![
                AssistantContent::Text(text_content("let me check")),
                AssistantContent::ToolCall(ToolCall {
                    id: "tc1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}),
                }),
            ],
            api: "test-api".into(),
            provider: "test".into(),
            model: "test-model".into(),
            response_id: Some("resp_1".into()),
            usage: Usage {
                input: 5,
                output: 7,
                ..Usage::default()
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 42,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""stop_reason":"tool_use""#));
        let back: AssistantMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
        assert_eq!(back.text(), "let me check");
    }

    #[test]
    fn tool_result_message_roundtrip() {
        let msg = ToolResultMessage::success("tc1", "bash", "ok");
        let wrapped = Message::ToolResult(msg.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        assert!(json.contains(r#""role":"tool_result""#));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(wrapped, back);
        // and standalone without the role tag
        let back: ToolResultMessage =
            serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn tool_result_duration_ms_backward_compat() {
        // Old messages without duration_ms should deserialize to None
        let json = r#"{"tool_call_id":"tc1","tool_name":"bash","content":[{"type":"text","text":"ok"}],"is_error":false,"timestamp":1000}"#;
        let msg: ToolResultMessage = serde_json::from_str(json).expect("deserialize");
        assert_eq!(msg.duration_ms, None);
        assert_eq!(msg.summary, None);
    }

    #[test]
    fn tool_result_none_fields_not_serialized() {
        let msg = ToolResultMessage::success("tc1", "bash", "ok");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("duration_ms"));
        assert!(!json.contains("summary"));
    }

    #[test]
    fn tool_result_post_persist_actions_not_serialized() {
        let mut msg = ToolResultMessage::success("tc1", "bash", "ok");
        msg.post_persist_actions = vec![PostPersistAction::StopAgentLoop {
            reason: "retire".into(),
        }];
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("post_persist_actions"));
        assert!(!json.contains("StopAgentLoop"));
        // and it survives a round-trip as empty
        let back: ToolResultMessage = serde_json::from_str(&json).unwrap();
        assert!(back.post_persist_actions.is_empty());
    }

    #[test]
    fn all_message_roles_roundtrip() {
        let messages = vec![
            Message::User(UserMessage::text("hi")),
            Message::Assistant(AssistantMessage::empty("api", "prov", "model")),
            Message::ToolResult(ToolResultMessage::success("tc", "t", "ok")),
            Message::CompactionSummary(CompactionSummaryMessage {
                summary: "old context".into(),
                tokens_before: 100,
                timestamp: 1,
            }),
            Message::Info(InfoMessage::new("an info")),
        ];
        for m in &messages {
            let json = serde_json::to_string(m).unwrap();
            let back: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(m, &back, "round-trip failed for {json}");
        }
    }

    // -- stream events --

    #[test]
    fn stream_event_start_done_roundtrip() {
        let dup = |m: &AssistantMessage| m.clone();
        let base = AssistantMessage::empty("api", "prov", "model");

        let start = StreamEvent::Start {
            partial: dup(&base),
        };
        let done = StreamEvent::Done {
            reason: StopReason::Stop,
            message: dup(&base),
        };
        for ev in [&start, &done] {
            let json = serde_json::to_string(ev).unwrap();
            let back: StreamEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(ev, &back, "round-trip failed for {json}");
        }
        assert!(
            serde_json::to_string(&start)
                .unwrap()
                .starts_with(r#"{"type":"start""#)
        );
        assert!(
            serde_json::to_string(&done)
                .unwrap()
                .contains(r#""type":"done""#)
        );
    }

    #[test]
    fn stream_event_text_delta_roundtrip() {
        let ev = StreamEvent::TextDelta {
            content_index: 0,
            delta: "Hel".into(),
            partial: AssistantMessage::empty("api", "prov", "model"),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""type":"text_delta""#));
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn stream_event_toolcall_end_roundtrip() {
        let tc = ToolCall {
            id: "tc_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"cmd": "ls"}),
        };
        let ev = StreamEvent::ToolcallEnd {
            content_index: 1,
            tool_call: tc.clone(),
            partial: AssistantMessage::empty("api", "prov", "model"),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""tool_call""#));
        assert!(json.contains(r#""name":"bash""#));
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        match back {
            StreamEvent::ToolcallEnd {
                content_index: 1,
                tool_call,
                ..
            } => assert_eq!(tc, tool_call),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn stream_event_tool_output_and_result_roundtrip() {
        let delta = StreamEvent::ToolOutputDelta {
            tool_call_id: "tc_1".into(),
            delta: "building...".into(),
        };
        let result = StreamEvent::ToolResult {
            tool_call_id: "tc_1".into(),
            tool_name: "bash".into(),
            is_error: false,
            content: "done".into(),
            summary: Some("built in 3s".into()),
        };
        for ev in [&delta, &result] {
            let json = serde_json::to_string(ev).unwrap();
            let back: StreamEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(ev, &back);
        }
    }

    #[test]
    fn stream_event_phase_and_status_roundtrip() {
        let phase = StreamEvent::Phase {
            phase: AgentPhase::Connecting,
            turn_started_at_ms: Some(1000),
            phase_started_at_ms: Some(1001),
        };
        let status = StreamEvent::Status {
            message: "retrying in 2s".into(),
        };
        for ev in [&phase, &status] {
            let json = serde_json::to_string(ev).unwrap();
            let back: StreamEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(ev, &back);
        }
        // missing timestamps default to None (forward-compat)
        let json = r#"{"type":"phase","phase":"idle"}"#;
        let back: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            back,
            StreamEvent::Phase {
                phase: AgentPhase::Idle,
                turn_started_at_ms: None,
                phase_started_at_ms: None
            }
        ));
    }

    // -- tools & context --

    #[test]
    fn tool_and_context_roundtrip() {
        let ctx = Context {
            system_prompt: Some("You are an expert".into()),
            messages: vec![Message::User(UserMessage::text("do the thing"))],
            tools: vec![Tool {
                name: "bash".into(),
                description: "run a command".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains(r#""system_prompt""#));
        assert!(json.contains(r#""name":"bash""#));
        let back: Context = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx, back);
        assert_eq!(back.tools[0].name, "bash");
    }

    #[test]
    fn context_empty_tools_not_serialized() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![],
            tools: vec![],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("system_prompt"));
    }

    // -- model --

    #[test]
    fn model_roundtrip_skips_empty_headers() {
        let model = Model {
            id: "test".into(),
            name: "Test".into(),
            api: "test-api".into(),
            provider: "test".into(),
            base_url: "http://localhost/v1".into(),
            thinking: ThinkingStyle::None,
            cost: ModelCost::default(),
            context_window: 1000,
            max_tokens: 500,
            headers: HashMap::new(),
        };
        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.contains("headers"));
        let back: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(model, back);

        let mut with_headers = model;
        with_headers.headers.insert("x-test".into(), "1".into());
        let json = serde_json::to_string(&with_headers).unwrap();
        assert!(json.contains(r#""headers""#));
    }

    // -- misc --

    #[test]
    fn cache_retention_resolve_defaults_to_short() {
        assert_eq!(CacheRetention::resolve(None), CacheRetention::Short);
        assert_eq!(
            CacheRetention::resolve(Some(CacheRetention::Long)),
            CacheRetention::Long
        );
        assert_eq!(
            CacheRetention::resolve(Some(CacheRetention::None)),
            CacheRetention::None
        );
    }

    #[test]
    fn agent_phase_labels() {
        assert_eq!(AgentPhase::Idle.label(), "idle");
        assert_eq!(AgentPhase::ToolExec.label(), "running tools...");
        assert_eq!(AgentPhase::default(), AgentPhase::Idle);
    }

    #[test]
    fn tool_result_text_concatenates() {
        let msg = ToolResultMessage::success("tc", "t", "hello world");
        assert_eq!(msg.text(), "hello world");
        let err = ToolResultMessage::error("tc", "t", "nope");
        assert!(err.is_error);
        assert_eq!(err.text(), "nope");
    }
}
