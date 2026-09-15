#![allow(clippy::manual_range_contains)]
use tars_base::{
    AssistantContent, Message, StopReason, StreamEvent, ToolResultContent, UserContent, UserMessage,
};
use tars_base::{Context, truncate_str, truncate_str_end};

use crate::provider::EventReceiver;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

pub fn estimate_tokens(message: &Message) -> u64 {
    let chars = match message {
        Message::User(u) => u
            .content
            .iter()
            .map(|c| match c {
                UserContent::Text(t) => t.text.len(),
                UserContent::Image(_) => 4800,
            })
            .sum(),
        Message::Assistant(a) => a
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => t.text.len(),
                AssistantContent::Thinking(t) => t.thinking.len(),
                AssistantContent::ToolCall(tc) => {
                    tc.name.len()
                        + serde_json::to_string(&tc.arguments)
                            .unwrap_or_default()
                            .len()
                }
            })
            .sum(),
        Message::ToolResult(tr) => tr
            .content
            .iter()
            .map(|c| match c {
                ToolResultContent::Text(t) => t.text.len(),
                ToolResultContent::Image(_) => 4800,
            })
            .sum(),
        Message::CompactionSummary(cs) => cs.summary.len(),
        Message::Info(i) => i.text.len(),
    };
    (chars as u64).div_ceil(4)
}

pub fn estimate_context_tokens(messages: &[Message]) -> u64 {
    let mut last_usage_idx = None;
    for (i, msg) in messages.iter().enumerate().rev() {
        if let Message::Assistant(a) = msg
            && a.stop_reason != StopReason::Error
            && a.stop_reason != StopReason::Aborted
        {
            let total = a.usage.input + a.usage.cache_read + a.usage.cache_write;
            if total > 0 {
                last_usage_idx = Some((i, total + a.usage.output));
                break;
            }
        }
    }

    match last_usage_idx {
        Some((idx, usage_tokens)) => {
            let trailing: u64 = messages[idx + 1..].iter().map(estimate_tokens).sum();
            usage_tokens + trailing
        }
        None => messages.iter().map(estimate_tokens).sum(),
    }
}

// ---------------------------------------------------------------------------
// Compaction decision
// ---------------------------------------------------------------------------

pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled || context_window == 0 {
        return false;
    }
    context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

// ---------------------------------------------------------------------------
// Cut point
// ---------------------------------------------------------------------------

pub fn find_cut_point(messages: &[Message], keep_recent_tokens: u64) -> usize {
    let mut accumulated: u64 = 0;

    for i in (0..messages.len()).rev() {
        accumulated += estimate_tokens(&messages[i]);
        if accumulated >= keep_recent_tokens {
            for (j, msg) in messages.iter().enumerate().skip(i) {
                match msg {
                    Message::User(_) | Message::CompactionSummary(_) | Message::Info(_) => {
                        return j;
                    }
                    _ => continue,
                }
            }
            return 0;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Summary generation
// ---------------------------------------------------------------------------

const SUMMARIZATION_SYSTEM_PROMPT: &str =
    "You are a precise summarizer. Create structured summaries of coding conversations.";

const SUMMARIZATION_PROMPT: &str = r#"The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, file paths, or references needed to continue]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

fn serialize_messages(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg {
            Message::User(u) => {
                out.push_str("## User\n");
                for c in &u.content {
                    if let UserContent::Text(t) = c {
                        out.push_str(&t.text);
                        out.push('\n');
                    }
                }
            }
            Message::Assistant(a) => {
                out.push_str("## Assistant\n");
                for c in &a.content {
                    match c {
                        AssistantContent::Text(t) => {
                            out.push_str(&t.text);
                            out.push('\n');
                        }
                        AssistantContent::Thinking(t) => {
                            out.push_str("<thinking>\n");
                            out.push_str(&t.thinking);
                            out.push_str("\n</thinking>\n");
                        }
                        AssistantContent::ToolCall(tc) => {
                            out.push_str(&format!("[tool_call: {}({})]\n", tc.name, tc.arguments));
                        }
                    }
                }
            }
            Message::ToolResult(tr) => {
                out.push_str(&format!("## Tool Result ({})\n", tr.tool_name));
                for c in &tr.content {
                    if let ToolResultContent::Text(t) = c {
                        if t.text.len() > 2000 {
                            out.push_str(truncate_str(&t.text, 1000));
                            out.push_str("\n... [truncated] ...\n");
                            out.push_str(truncate_str_end(&t.text, 1000));
                        } else {
                            out.push_str(&t.text);
                        }
                        out.push('\n');
                    }
                }
            }
            Message::CompactionSummary(cs) => {
                out.push_str("## Previous Summary\n");
                out.push_str(&cs.summary);
                out.push('\n');
            }
            Message::Info(_) => {}
        }
        out.push('\n');
    }
    out
}

pub fn build_summarization_context(
    messages_to_summarize: &[Message],
    keep_hint: Option<&str>,
) -> Context {
    let conversation = serialize_messages(messages_to_summarize);
    let mut prompt = format!(
        "<conversation>\n{}</conversation>\n\n{}",
        conversation, SUMMARIZATION_PROMPT
    );

    if let Some(hint) = keep_hint {
        let trimmed = hint.trim();
        if !trimmed.is_empty() {
            prompt.push_str(
                "\n\n## User-specified focus\n\
                 The user explicitly asked you to preserve the following in the summary, \
                 in addition to (not instead of) the standard sections above. \
                 Treat it as advisory guidance, not a hard filter:\n\n\
                 <keep_hint>\n",
            );
            prompt.push_str(trimmed);
            prompt.push_str("\n</keep_hint>");
        }
    }

    Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
        messages: vec![Message::User(UserMessage::text(&prompt))],
        tools: Vec::new(),
    }
}

pub async fn extract_summary(mut rx: EventReceiver) -> tars_base::Result<String> {
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Done { message, .. } => {
                return Ok(message.text());
            }
            StreamEvent::Error { error, .. } => {
                return Err(tars_base::Error::Http {
                    status: 0,
                    message: error
                        .error_message
                        .unwrap_or_else(|| "summarization failed".into()),
                });
            }
            _ => continue,
        }
    }
    Err(tars_base::Error::ChannelClosed)
}

/// Async summarization via the provider — uses the same streaming path as the
/// agent loop (no blocking `recv_blocking`).
pub async fn summarize(
    model: &tars_base::Model,
    messages_to_summarize: &[Message],
    registry: &crate::ProviderRegistry,
    options: &tars_base::StreamOptions,
    keep_hint: Option<&str>,
) -> tars_base::Result<String> {
    let ctx = build_summarization_context(messages_to_summarize, keep_hint);
    let mut rx = registry.stream(model, &ctx, options).await?;
    // Reuse extract_summary helper for streaming
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Done { message, .. } => return Ok(message.text()),
            StreamEvent::Error { error, .. } => {
                return Err(tars_base::Error::Http {
                    status: 0,
                    message: error
                        .error_message
                        .unwrap_or_else(|| "summarization failed".into()),
                });
            }
            _ => continue,
        }
    }
    Err(tars_base::Error::ChannelClosed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::{
        AssistantMessage, CompactionSummaryMessage, InfoMessage, TextContent, ToolResultMessage,
    };

    fn user(text: &str) -> Message {
        Message::User(UserMessage::text(text))
    }

    fn assistant(text: &str, input_tokens: u64) -> Message {
        let mut a = AssistantMessage::empty("test", "test", "test");
        a.content.push(AssistantContent::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        }));
        a.usage.input = input_tokens;
        a.usage.output = 100;
        Message::Assistant(a)
    }

    fn compaction_summary(text: &str) -> Message {
        Message::CompactionSummary(CompactionSummaryMessage {
            summary: text.to_string(),
            tokens_before: 50000,
            timestamp: 0,
        })
    }

    #[test]
    fn estimate_tokens_basic() {
        let msg = user("hello");
        assert_eq!(estimate_tokens(&msg), 2);
        let msg = user(&"x".repeat(400));
        assert_eq!(estimate_tokens(&msg), 100);
    }

    #[test]
    fn should_compact_thresholds() {
        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_000,
            keep_recent_tokens: 20_000,
        };
        assert!(should_compact(190_000, 200_000, &settings));
        assert!(!should_compact(180_000, 200_000, &settings));
        let disabled = CompactionSettings {
            enabled: false,
            ..settings
        };
        assert!(!should_compact(190_000, 200_000, &disabled));
    }

    #[test]
    fn find_cut_point_keeps_recent() {
        let big = "x".repeat(400);
        let messages = vec![
            user(&big),
            assistant(&big, 500),
            user(&big),
            assistant(&big, 800),
            user(&big),
            assistant(&big, 1000),
        ];
        let cut = find_cut_point(&messages, 250);
        assert_eq!(cut, 4);
    }

    #[test]
    fn find_cut_point_never_cuts_at_tool_result() {
        let big = "x".repeat(400);
        let messages = vec![
            user(&big),
            assistant(&big, 500),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tc1".into(),
                tool_name: "bash".into(),
                content: vec![ToolResultContent::Text(TextContent {
                    text: big.clone(),
                    text_signature: None,
                })],
                details: None,
                is_error: false,
                timestamp: 0,
                duration_ms: None,
                summary: None,
                post_persist_actions: Vec::new(),
            }),
            user(&big),
            assistant(&big, 1000),
        ];
        let cut = find_cut_point(&messages, 250);
        assert_eq!(cut, 3);
    }

    #[test]
    fn find_cut_point_with_compaction_summary() {
        let big = "x".repeat(400);
        let messages = vec![
            compaction_summary("previous summary"),
            user(&big),
            assistant(&big, 500),
            user(&big),
            assistant(&big, 1000),
        ];
        let cut = find_cut_point(&messages, 250);
        assert_eq!(cut, 3);
    }

    #[test]
    fn estimate_context_tokens_uses_usage() {
        let messages = vec![user("hello"), assistant("world", 5000), user("followup")];
        let est = estimate_context_tokens(&messages);
        assert!(est > 5000);
        assert!(est < 5200);
    }

    #[test]
    fn estimate_context_tokens_no_usage() {
        let messages = vec![user("hello"), user("world")];
        let est = estimate_context_tokens(&messages);
        assert_eq!(est, 4);
    }

    #[test]
    fn serialize_messages_roundtrip() {
        let messages = vec![user("write a test"), assistant("here's the code", 100)];
        let text = serialize_messages(&messages);
        assert!(text.contains("## User"));
        assert!(text.contains("write a test"));
        assert!(text.contains("## Assistant"));
        assert!(text.contains("here's the code"));
    }

    #[test]
    fn repeated_compaction_preserves_recent() {
        let big = "x".repeat(400);
        let messages = vec![
            compaction_summary("previous work summary"),
            user(&big),
            assistant(&big, 500),
            user(&big),
            assistant(&big, 800),
            user(&big),
            assistant(&big, 1000),
        ];
        let cut = find_cut_point(&messages, 250);
        assert!(cut >= 3, "cut={cut} should be >= 3 (keep recent)");
        assert!(cut <= 5, "cut={cut} should be <= 5");
        assert!(
            matches!(
                &messages[cut],
                Message::User(_) | Message::CompactionSummary(_)
            ),
            "cut point must be at a turn boundary"
        );
    }

    #[test]
    fn should_compact_after_previous_compaction() {
        let messages = vec![
            compaction_summary("summary of earlier work"),
            user("continue working"),
            assistant("ok", 190_000),
        ];
        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_000,
            keep_recent_tokens: 20_000,
        };
        let ctx_tokens = estimate_context_tokens(&messages);
        assert!(ctx_tokens > 180_000);
        assert!(should_compact(ctx_tokens, 200_000, &settings));
        assert!(!should_compact(ctx_tokens, 1_000_000, &settings));
    }

    #[test]
    fn estimate_tokens_info() {
        let msg = Message::Info(InfoMessage {
            text: "hello".into(),
            timestamp: 0,
        });
        assert_eq!(estimate_tokens(&msg), 2);
    }

    #[test]
    fn find_cut_point_info_is_valid_boundary() {
        let big = "x".repeat(400);
        let messages = vec![
            user(&big),
            assistant(&big, 500),
            Message::Info(InfoMessage {
                text: "notification".into(),
                timestamp: 0,
            }),
            user(&big),
            assistant(&big, 1000),
        ];
        let cut = find_cut_point(&messages, 250);
        assert!(cut >= 2 && cut <= 3, "cut={cut} should be 2 or 3");
        assert!(
            matches!(&messages[cut], Message::User(_) | Message::Info(_)),
            "cut point must be at a valid boundary"
        );
    }

    #[test]
    fn build_summarization_context_without_hint_has_no_keep_hint_block() {
        let messages = vec![user("hi"), assistant("hello", 5)];
        let ctx = build_summarization_context(&messages, None);
        let prompt = match &ctx.messages[0] {
            Message::User(u) => match &u.content[0] {
                UserContent::Text(t) => t.text.clone(),
                _ => panic!("expected text content"),
            },
            _ => panic!("expected user message"),
        };
        assert!(prompt.contains("## Goal"), "standard sections present");
        assert!(
            !prompt.contains("<keep_hint>"),
            "no keep_hint block when hint=None"
        );
        assert!(
            !prompt.contains("User-specified focus"),
            "no focus header when hint=None"
        );
    }

    #[test]
    fn build_summarization_context_with_hint_includes_block() {
        let messages = vec![user("hi"), assistant("hello", 5)];
        let ctx = build_summarization_context(&messages, Some("keep file paths verbatim"));
        let prompt = match &ctx.messages[0] {
            Message::User(u) => match &u.content[0] {
                UserContent::Text(t) => t.text.clone(),
                _ => panic!("expected text content"),
            },
            _ => panic!("expected user message"),
        };
        assert!(
            prompt.contains("## Goal"),
            "standard sections must still be present"
        );
        assert!(
            prompt.contains("<keep_hint>\nkeep file paths verbatim\n</keep_hint>"),
            "hint must appear inside <keep_hint> tags, got: {prompt}"
        );
        assert!(
            prompt.contains("advisory"),
            "hint block must mark itself as advisory"
        );
    }

    #[test]
    fn build_summarization_context_with_blank_hint_omits_block() {
        let messages = vec![user("hi"), assistant("hello", 5)];
        let ctx = build_summarization_context(&messages, Some("   \n  \t"));
        let prompt = match &ctx.messages[0] {
            Message::User(u) => match &u.content[0] {
                UserContent::Text(t) => t.text.clone(),
                _ => panic!("expected text content"),
            },
            _ => panic!("expected user message"),
        };
        assert!(
            !prompt.contains("<keep_hint>"),
            "whitespace-only hint must be treated as None"
        );
    }

    #[tokio::test]
    async fn extract_summary_from_stream() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut msg = tars_base::AssistantMessage::empty("test", "test", "test");
        msg.content.push(AssistantContent::Text(TextContent {
            text: "summary text here".into(),
            text_signature: None,
        }));
        tx.send(StreamEvent::Done {
            reason: tars_base::StopReason::Stop,
            message: msg.clone(),
        })
        .await
        .unwrap();
        drop(tx);
        let summary = extract_summary(rx).await.unwrap();
        assert_eq!(summary, "summary text here");
    }

    #[tokio::test]
    async fn summarize_via_mock_provider() {
        use crate::ProviderRegistry;
        use crate::providers::{MockProvider, MockResponse};
        use tars_base::{Model, ModelCost, ThinkingStyle};

        let mock = MockProvider::new(vec![MockResponse::Text("mock summary".into())]);
        let mut registry = ProviderRegistry::new();
        registry.register(mock);
        let model = Model {
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
        };
        let messages = vec![user("hello"), assistant("world", 100)];
        let summary = summarize(
            &model,
            &messages,
            &registry,
            &tars_base::StreamOptions::default(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(summary, "mock summary");
    }
}
