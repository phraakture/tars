use async_trait::async_trait;
use futures::StreamExt;

use super::anthropic_types::*;
use crate::{EventReceiver, EventSender, Provider, STREAM_CAPACITY};
use tars_base::{
    AssistantContent, AssistantMessage, Context, Message, Model, ModelCost, StopReason,
    StreamEvent, StreamOptions, TextContent, ThinkingContent, ThinkingDisplay, ThinkingEffort,
    ThinkingStyle, ToolCall, ToolResultContent, UserContent,
};

const API_ID: &str = "anthropic-messages";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

pub struct Anthropic;

#[async_trait]
impl Provider for Anthropic {
    fn api_id(&self) -> &str {
        API_ID
    }

    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> tars_base::Result<EventReceiver> {
        let body = build_request_body(model, context, options)?;
        let base_url = model.base_url.clone();
        let api_key = options
            .api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| tars_base::Error::NoApiKey("anthropic".into()))?;
        let api_id = model.api.clone();
        let provider_name = model.provider.clone();
        let model_id = model.id.clone();
        let model_clone = model.clone();
        let extra_headers = model.headers.clone();
        let options_clone = options.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CAPACITY);

        tokio::spawn(async move {
            let result = run_anthropic_stream(
                &base_url,
                &api_key,
                &api_id,
                &provider_name,
                &model_id,
                &model_clone,
                &body,
                extra_headers,
                &options_clone,
                &tx,
            )
            .await;
            if let Err(e) = result {
                let _ = emit_error(&tx, &api_id, &provider_name, &model_id, e).await;
            }
        });

        Ok(rx)
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_anthropic_stream(
    base_url: &str,
    api_key: &str,
    api_id: &str,
    provider_name: &str,
    model_id: &str,
    model: &Model,
    body: &MessagesRequest,
    extra_model_headers: std::collections::HashMap<String, String>,
    options: &StreamOptions,
    tx: &EventSender,
) -> tars_base::Result<()> {
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| tars_base::Error::Internal(e.to_string()))?;

    let mut req = client
        .post(&url)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("accept", "application/json")
        .header("x-api-key", api_key)
        .json(body);

    for (k, v) in &extra_model_headers {
        req = req.header(k.as_str(), v.as_str());
    }
    for (k, v) in &options.headers {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = req
        .send()
        .await
        .map_err(|e| map_reqwest_error(e, provider_name))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let body_text = resp.text().await.unwrap_or_default();
        return Err(map_http_status(
            status,
            body_text,
            retry_after,
            provider_name,
        ));
    }

    let mut output = AssistantMessage::empty(api_id, provider_name, model_id);
    send_event(
        tx,
        StreamEvent::Start {
            partial: output.clone(),
        },
    )
    .await?;

    let mut block_index_map: Vec<(u64, usize)> = Vec::new();
    let mut tool_json_accum: std::collections::HashMap<u64, String> =
        std::collections::HashMap::new();
    let mut current_event_type = String::new();

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| tars_base::Error::Http {
            status: 0,
            message: e.to_string(),
        })?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].to_string();
            buf = buf[nl + 1..].to_string();
            let line = line.trim_end_matches('\r');

            if let Some(event_type) = line.strip_prefix("event: ") {
                current_event_type = event_type.to_string();
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];

            match current_event_type.as_str() {
                "message_start" => {
                    let ev: MessageStartEvent = serde_json::from_str(data)
                        .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
                    output.response_id = Some(ev.message.id);
                    if let Some(usage) = ev.message.usage {
                        usage.apply_to(&mut output.usage);
                        model.calculate_cost(&mut output.usage);
                    }
                }
                "content_block_start" => {
                    let ev: ContentBlockStartEvent = serde_json::from_str(data)
                        .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
                    match ev.content_block {
                        ContentBlock::Text { .. } => {
                            output.content.push(AssistantContent::Text(TextContent {
                                text: String::new(),
                                text_signature: None,
                            }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            send_event(
                                tx,
                                StreamEvent::TextStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        ContentBlock::Thinking { .. } => {
                            output
                                .content
                                .push(AssistantContent::Thinking(ThinkingContent {
                                    thinking: String::new(),
                                    thinking_signature: None,
                                    redacted: false,
                                }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            send_event(
                                tx,
                                StreamEvent::ThinkingStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        ContentBlock::RedactedThinking { data: sig } => {
                            output
                                .content
                                .push(AssistantContent::Thinking(ThinkingContent {
                                    thinking: "[Reasoning redacted]".into(),
                                    thinking_signature: Some(sig),
                                    redacted: true,
                                }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            send_event(
                                tx,
                                StreamEvent::ThinkingStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        ContentBlock::ToolUse { id, name, .. } => {
                            output.content.push(AssistantContent::ToolCall(ToolCall {
                                id,
                                name,
                                arguments: serde_json::Value::Object(Default::default()),
                            }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            send_event(
                                tx,
                                StreamEvent::ToolcallStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                    }
                }
                "content_block_delta" => {
                    let ev: ContentBlockDeltaEvent = serde_json::from_str(data)
                        .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
                    let ci = block_index_map
                        .iter()
                        .find(|(bi, _)| *bi == ev.index)
                        .map(|(_, ci)| *ci);
                    let Some(ci) = ci else { continue };
                    match ev.delta {
                        Delta::TextDelta { text } => {
                            if let Some(AssistantContent::Text(t)) = output.content.get_mut(ci) {
                                t.text.push_str(&text);
                            }
                            send_event(
                                tx,
                                StreamEvent::TextDelta {
                                    content_index: ci,
                                    delta: text,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        Delta::ThinkingDelta { thinking } => {
                            if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci)
                            {
                                t.thinking.push_str(&thinking);
                            }
                            send_event(
                                tx,
                                StreamEvent::ThinkingDelta {
                                    content_index: ci,
                                    delta: thinking,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        Delta::InputJsonDelta { partial_json } => {
                            tool_json_accum
                                .entry(ev.index)
                                .or_default()
                                .push_str(&partial_json);
                            send_event(
                                tx,
                                StreamEvent::ToolcallDelta {
                                    content_index: ci,
                                    delta: partial_json,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        Delta::SignatureDelta { signature } => {
                            if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci)
                            {
                                let s = t.thinking_signature.get_or_insert_with(String::new);
                                s.push_str(&signature);
                            }
                        }
                    }
                }
                "content_block_stop" => {
                    let ev: ContentBlockStopEvent = serde_json::from_str(data)
                        .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
                    let ci = block_index_map
                        .iter()
                        .find(|(bi, _)| *bi == ev.index)
                        .map(|(_, ci)| *ci);
                    let Some(ci) = ci else { continue };
                    match output.content.get(ci) {
                        Some(AssistantContent::Text(t)) => {
                            send_event(
                                tx,
                                StreamEvent::TextEnd {
                                    content_index: ci,
                                    content: t.text.clone(),
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        Some(AssistantContent::Thinking(t)) => {
                            send_event(
                                tx,
                                StreamEvent::ThinkingEnd {
                                    content_index: ci,
                                    content: t.thinking.clone(),
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        Some(AssistantContent::ToolCall(_)) => {
                            if let Some(json_str) = tool_json_accum.remove(&ev.index)
                                && let Ok(args) = serde_json::from_str(&json_str)
                                && let Some(AssistantContent::ToolCall(tc)) =
                                    output.content.get_mut(ci)
                            {
                                tc.arguments = args;
                            }
                            let tc = match output.content.get(ci) {
                                Some(AssistantContent::ToolCall(tc)) => tc.clone(),
                                _ => continue,
                            };
                            send_event(
                                tx,
                                StreamEvent::ToolcallEnd {
                                    content_index: ci,
                                    tool_call: tc,
                                    partial: output.clone(),
                                },
                            )
                            .await?;
                        }
                        None => {}
                    }
                }
                "message_delta" => {
                    let ev: MessageDeltaEvent = serde_json::from_str(data)
                        .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
                    if let Some(delta) = ev.delta
                        && let Some(reason) = delta.stop_reason
                    {
                        output.stop_reason = map_stop_reason(&reason);
                    }
                    if let Some(usage) = ev.usage {
                        usage.apply_to(&mut output.usage);
                        model.calculate_cost(&mut output.usage);
                    }
                }
                "message_stop" => {
                    send_event(
                        tx,
                        StreamEvent::Done {
                            reason: output.stop_reason,
                            message: output,
                        },
                    )
                    .await?;
                    return Ok(());
                }
                "error" => {
                    let error_msg = serde_json::from_str::<serde_json::Value>(data)
                        .ok()
                        .and_then(|v| {
                            v.get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| format!("SSE error: {data}"));
                    strip_partial_tool_state(&mut output, &mut tool_json_accum, &block_index_map);
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(error_msg);
                    send_event(
                        tx,
                        StreamEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        },
                    )
                    .await?;
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    strip_partial_tool_state(&mut output, &mut tool_json_accum, &block_index_map);
    output.stop_reason = StopReason::Error;
    output.error_message = Some("Stream ended unexpectedly".into());
    send_event(
        tx,
        StreamEvent::Error {
            reason: StopReason::Error,
            error: output,
        },
    )
    .await?;
    Ok(())
}

fn strip_partial_tool_state(
    output: &mut AssistantMessage,
    tool_json_accum: &mut std::collections::HashMap<u64, String>,
    block_index_map: &[(u64, usize)],
) {
    for (block_index, _) in tool_json_accum.drain() {
        if let Some((_, ci)) = block_index_map.iter().find(|(bi, _)| *bi == block_index)
            && let Some(AssistantContent::ToolCall(tc)) = output.content.get_mut(*ci)
        {
            tc.arguments = serde_json::Value::Object(Default::default());
        }
    }
}

async fn send_event(tx: &EventSender, event: StreamEvent) -> tars_base::Result<()> {
    tx.send(event)
        .await
        .map_err(|_| tars_base::Error::ChannelClosed)
}

async fn emit_error(
    tx: &EventSender,
    api_id: &str,
    provider_name: &str,
    model_id: &str,
    err: tars_base::Error,
) -> tars_base::Result<()> {
    let mut msg = AssistantMessage::empty(api_id, provider_name, model_id);
    msg.stop_reason = StopReason::Error;
    msg.error_message = Some(err.to_string());
    let _ = tx
        .send(StreamEvent::Error {
            reason: StopReason::Error,
            error: msg,
        })
        .await;
    Ok(())
}

fn map_reqwest_error(err: reqwest::Error, provider: &str) -> tars_base::Error {
    if err.is_timeout() {
        return tars_base::Error::Timeout(provider.to_string());
    }
    tars_base::Error::Http {
        status: 0,
        message: err.to_string(),
    }
}

fn map_http_status(
    status: u16,
    body: String,
    retry_after: Option<u64>,
    provider: &str,
) -> tars_base::Error {
    match status {
        401 | 403 => tars_base::Error::Auth(provider.to_string()),
        429 => tars_base::Error::RateLimited {
            provider: provider.to_string(),
            retry_after,
        },
        408 => tars_base::Error::Timeout(provider.to_string()),
        413 | 422 => tars_base::Error::ContextOverflow,
        _ => tars_base::Error::Http {
            status,
            message: body,
        },
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

fn build_request_body(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> tars_base::Result<MessagesRequest> {
    let mut messages = Vec::new();

    for msg in &context.messages {
        match msg {
            Message::User(u) => {
                messages.push(ApiMessage {
                    role: "user",
                    content: convert_user_content(&u.content),
                });
            }
            Message::Assistant(a) => {
                let content = convert_assistant_content(&a.content);
                if !content.is_empty() {
                    messages.push(ApiMessage {
                        role: "assistant",
                        content: serde_json::Value::Array(content),
                    });
                }
            }
            Message::ToolResult(tr) => {
                let content = convert_tool_result_content(&tr.content);
                messages.push(ApiMessage {
                    role: "user",
                    content: serde_json::json!([{
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": content,
                        "is_error": tr.is_error,
                    }]),
                });
            }
            Message::CompactionSummary(cs) => {
                let text = format!(
                    "[Context compacted — {} tokens before compaction]\n\n{}",
                    cs.tokens_before, cs.summary
                );
                messages.push(ApiMessage {
                    role: "user",
                    content: serde_json::Value::String(text),
                });
            }
            Message::Info(_) => {}
        }
    }

    add_cache_breakpoint_to_last_user_message(&mut messages);

    let max_tokens = options
        .max_tokens
        .unwrap_or((model.max_tokens / 3).max(1024));

    let cc = Some(CacheControl::ephemeral());

    let system = context.system_prompt.as_ref().map(|prompt| {
        vec![SystemBlock {
            block_type: "text",
            text: prompt.clone(),
            cache_control: cc.clone(),
        }]
    });

    let tools = if context.tools.is_empty() {
        None
    } else {
        let mut defs: Vec<ToolDef> = context
            .tools
            .iter()
            .map(|t| ToolDef {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
                eager_input_streaming: true,
                cache_control: None,
            })
            .collect();
        if let Some(last) = defs.last_mut() {
            last.cache_control = cc.clone();
        }
        Some(defs)
    };

    let thinking_enabled = thinking_requested(model, options);
    let (thinking, output_config) = if thinking_enabled {
        build_thinking_config(model, options)
    } else {
        (None, None)
    };

    Ok(MessagesRequest {
        model: model.id.clone(),
        messages,
        max_tokens,
        stream: true,
        system,
        temperature: options.temperature,
        tools,
        thinking,
        output_config,
    })
}

// ---------------------------------------------------------------------------
// Thinking helpers
// ---------------------------------------------------------------------------

pub(crate) fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6")
        || model_id.contains("opus-4.6")
        || model_id.contains("opus-4-7")
        || model_id.contains("opus-4.7")
        || model_id.contains("sonnet-4-6")
        || model_id.contains("sonnet-4.6")
}

fn map_effort(effort: ThinkingEffort, model_id: &str) -> &'static str {
    match effort {
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::Max => "max",
        ThinkingEffort::XHigh => {
            if model_id.contains("opus-4-6") || model_id.contains("opus-4.6") {
                "max"
            } else if model_id.contains("opus-4-7") || model_id.contains("opus-4.7") {
                "xhigh"
            } else {
                "high"
            }
        }
    }
}

fn thinking_requested(model: &Model, options: &StreamOptions) -> bool {
    if model.thinking != ThinkingStyle::Anthropic {
        return false;
    }
    match options.thinking_enabled {
        Some(v) => v,
        None => {
            options.thinking_budget.is_some()
                || (supports_adaptive_thinking(&model.id) && options.thinking_effort.is_some())
        }
    }
}

fn build_thinking_config(
    model: &Model,
    options: &StreamOptions,
) -> (Option<ThinkingConfig>, Option<OutputConfig>) {
    let display = match options
        .thinking_display
        .unwrap_or(ThinkingDisplay::Summarized)
    {
        ThinkingDisplay::Summarized => "summarized",
        ThinkingDisplay::Omitted => "omitted",
    };

    if supports_adaptive_thinking(&model.id) {
        let output_config = options.thinking_effort.map(|e| OutputConfig {
            effort: map_effort(e, &model.id),
        });
        (
            Some(ThinkingConfig {
                thinking_type: "adaptive",
                budget_tokens: None,
                display: Some(display),
            }),
            output_config,
        )
    } else {
        let budget = options.thinking_budget.unwrap_or(1024);
        (
            Some(ThinkingConfig {
                thinking_type: "enabled",
                budget_tokens: Some(budget),
                display: Some(display),
            }),
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Content conversion
// ---------------------------------------------------------------------------

fn convert_user_content(content: &[UserContent]) -> serde_json::Value {
    if content.len() == 1
        && let UserContent::Text(t) = &content[0]
    {
        return serde_json::Value::String(t.text.clone());
    }
    let blocks: Vec<serde_json::Value> = content
        .iter()
        .map(|c| match c {
            UserContent::Text(t) => serde_json::json!({"type": "text", "text": t.text}),
            UserContent::Image(img) => serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.data,
                }
            }),
        })
        .collect();
    serde_json::Value::Array(blocks)
}

fn convert_assistant_content(content: &[AssistantContent]) -> Vec<serde_json::Value> {
    content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) if !t.text.is_empty() => {
                Some(serde_json::json!({"type": "text", "text": t.text}))
            }
            AssistantContent::Thinking(t) if t.redacted => Some(serde_json::json!({
                "type": "redacted_thinking",
                "data": t.thinking_signature.as_deref().unwrap_or(""),
            })),
            AssistantContent::Thinking(t) if !t.thinking.is_empty() => {
                if let Some(ref sig) = t.thinking_signature
                    && !sig.is_empty()
                {
                    return Some(serde_json::json!({
                        "type": "thinking",
                        "thinking": t.thinking,
                        "signature": sig,
                    }));
                }
                Some(serde_json::json!({"type": "text", "text": t.thinking}))
            }
            AssistantContent::ToolCall(tc) => Some(serde_json::json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })),
            _ => None,
        })
        .collect()
}

fn convert_tool_result_content(content: &[ToolResultContent]) -> serde_json::Value {
    if content.len() == 1
        && let ToolResultContent::Text(t) = &content[0]
    {
        return serde_json::Value::String(t.text.clone());
    }
    let blocks: Vec<serde_json::Value> = content
        .iter()
        .map(|c| match c {
            ToolResultContent::Text(t) => serde_json::json!({"type": "text", "text": t.text}),
            ToolResultContent::Image(img) => serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.data,
                }
            }),
        })
        .collect();
    serde_json::Value::Array(blocks)
}

fn add_cache_breakpoint_to_last_user_message(messages: &mut [ApiMessage]) {
    let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") else {
        return;
    };

    match &mut last_user.content {
        serde_json::Value::String(text) => {
            let text = text.clone();
            last_user.content = serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }]);
        }
        serde_json::Value::Array(blocks) => {
            if let Some(last_block) = blocks.last_mut()
                && let Some(obj) = last_block.as_object_mut()
            {
                obj.insert(
                    "cache_control".into(),
                    serde_json::json!({"type": "ephemeral"}),
                );
            }
        }
        _ => {}
    }
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

pub fn models() -> Vec<Model> {
    vec![
        Model {
            id: "claude-opus-4-7".into(),
            name: "Claude Opus 4.7".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
            context_window: 1_000_000,
            max_tokens: 128_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-sonnet-4-6".into(),
            name: "Claude Sonnet 4.6".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            context_window: 1_000_000,
            max_tokens: 64_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-opus-4-6".into(),
            name: "Claude Opus 4.6".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
            context_window: 1_000_000,
            max_tokens: 128_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-sonnet-4-5".into(),
            name: "Claude Sonnet 4.5".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            context_window: 200_000,
            max_tokens: 64_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-opus-4-5".into(),
            name: "Claude Opus 4.5".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
            context_window: 200_000,
            max_tokens: 64_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-sonnet-4-20250514".into(),
            name: "Claude Sonnet 4".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            context_window: 200_000,
            max_tokens: 64_000,
            headers: Default::default(),
        },
        Model {
            id: "claude-haiku-4-5".into(),
            name: "Claude Haiku 4.5".into(),
            api: API_ID.into(),
            provider: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            thinking: ThinkingStyle::Anthropic,
            cost: ModelCost {
                input: 1.0,
                output: 5.0,
                cache_read: 0.1,
                cache_write: 1.25,
            },
            context_window: 200_000,
            max_tokens: 64_000,
            headers: Default::default(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Fixture helper
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) async fn collect_events_from_sse_fixture(
    api_id: &str,
    provider_name: &str,
    model_id: &str,
    model: &Model,
    raw: &str,
) -> Vec<StreamEvent> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(128);
    let raw = raw.to_string();
    let model_clone = model.clone();
    let api_id = api_id.to_string();
    let provider_name = provider_name.to_string();
    let model_id = model_id.to_string();

    let producer = tokio::spawn(async move {
        let mut output = AssistantMessage::empty(&api_id, &provider_name, &model_id);
        let _ = tx
            .send(StreamEvent::Start {
                partial: output.clone(),
            })
            .await;

        let mut block_index_map: Vec<(u64, usize)> = Vec::new();
        let mut tool_json_accum: std::collections::HashMap<u64, String> =
            std::collections::HashMap::new();
        let mut current_event_type = String::new();

        for line in raw.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(event_type) = line.strip_prefix("event: ") {
                current_event_type = event_type.to_string();
                continue;
            }
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            match current_event_type.as_str() {
                "message_start" => {
                    let ev: MessageStartEvent = serde_json::from_str(data).unwrap();
                    output.response_id = Some(ev.message.id);
                    if let Some(usage) = ev.message.usage {
                        usage.apply_to(&mut output.usage);
                        model_clone.calculate_cost(&mut output.usage);
                    }
                }
                "content_block_start" => {
                    let ev: ContentBlockStartEvent = serde_json::from_str(data).unwrap();
                    match ev.content_block {
                        ContentBlock::Text { .. } => {
                            output.content.push(AssistantContent::Text(TextContent {
                                text: String::new(),
                                text_signature: None,
                            }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            let _ = tx
                                .send(StreamEvent::TextStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        ContentBlock::Thinking { .. } => {
                            output
                                .content
                                .push(AssistantContent::Thinking(ThinkingContent {
                                    thinking: String::new(),
                                    thinking_signature: None,
                                    redacted: false,
                                }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            let _ = tx
                                .send(StreamEvent::ThinkingStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        ContentBlock::RedactedThinking { data: sig } => {
                            output
                                .content
                                .push(AssistantContent::Thinking(ThinkingContent {
                                    thinking: "[Reasoning redacted]".into(),
                                    thinking_signature: Some(sig),
                                    redacted: true,
                                }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            let _ = tx
                                .send(StreamEvent::ThinkingStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        ContentBlock::ToolUse { id, name, .. } => {
                            output.content.push(AssistantContent::ToolCall(ToolCall {
                                id,
                                name,
                                arguments: serde_json::Value::Object(Default::default()),
                            }));
                            let ci = output.content.len() - 1;
                            block_index_map.push((ev.index, ci));
                            let _ = tx
                                .send(StreamEvent::ToolcallStart {
                                    content_index: ci,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                    }
                }
                "content_block_delta" => {
                    let ev: ContentBlockDeltaEvent = serde_json::from_str(data).unwrap();
                    let ci = block_index_map
                        .iter()
                        .find(|(bi, _)| *bi == ev.index)
                        .map(|(_, ci)| *ci)
                        .unwrap();
                    match ev.delta {
                        Delta::TextDelta { text } => {
                            if let Some(AssistantContent::Text(t)) = output.content.get_mut(ci) {
                                t.text.push_str(&text);
                            }
                            let _ = tx
                                .send(StreamEvent::TextDelta {
                                    content_index: ci,
                                    delta: text,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        Delta::ThinkingDelta { thinking } => {
                            if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci)
                            {
                                t.thinking.push_str(&thinking);
                            }
                            let _ = tx
                                .send(StreamEvent::ThinkingDelta {
                                    content_index: ci,
                                    delta: thinking,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        Delta::InputJsonDelta { partial_json } => {
                            tool_json_accum
                                .entry(ev.index)
                                .or_default()
                                .push_str(&partial_json);
                            let _ = tx
                                .send(StreamEvent::ToolcallDelta {
                                    content_index: ci,
                                    delta: partial_json,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        Delta::SignatureDelta { signature } => {
                            if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci)
                            {
                                let s = t.thinking_signature.get_or_insert_with(String::new);
                                s.push_str(&signature);
                            }
                        }
                    }
                }
                "content_block_stop" => {
                    let ev: ContentBlockStopEvent = serde_json::from_str(data).unwrap();
                    let ci = block_index_map
                        .iter()
                        .find(|(bi, _)| *bi == ev.index)
                        .map(|(_, ci)| *ci)
                        .unwrap();
                    match output.content.get(ci) {
                        Some(AssistantContent::Text(t)) => {
                            let _ = tx
                                .send(StreamEvent::TextEnd {
                                    content_index: ci,
                                    content: t.text.clone(),
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        Some(AssistantContent::Thinking(t)) => {
                            let _ = tx
                                .send(StreamEvent::ThinkingEnd {
                                    content_index: ci,
                                    content: t.thinking.clone(),
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        Some(AssistantContent::ToolCall(_)) => {
                            if let Some(json_str) = tool_json_accum.remove(&ev.index)
                                && let Ok(args) = serde_json::from_str(&json_str)
                                && let Some(AssistantContent::ToolCall(tc)) =
                                    output.content.get_mut(ci)
                            {
                                tc.arguments = args;
                            }
                            let tc = match output.content.get(ci) {
                                Some(AssistantContent::ToolCall(tc)) => tc.clone(),
                                _ => continue,
                            };
                            let _ = tx
                                .send(StreamEvent::ToolcallEnd {
                                    content_index: ci,
                                    tool_call: tc,
                                    partial: output.clone(),
                                })
                                .await;
                        }
                        None => {}
                    }
                }
                "message_delta" => {
                    let ev: MessageDeltaEvent = serde_json::from_str(data).unwrap();
                    if let Some(delta) = ev.delta
                        && let Some(reason) = delta.stop_reason
                    {
                        output.stop_reason = map_stop_reason(&reason);
                    }
                    if let Some(usage) = ev.usage {
                        usage.apply_to(&mut output.usage);
                        model_clone.calculate_cost(&mut output.usage);
                    }
                }
                "message_stop" => {
                    let _ = tx
                        .send(StreamEvent::Done {
                            reason: output.stop_reason,
                            message: output,
                        })
                        .await;
                    return;
                }
                "error" => {
                    let error_msg = serde_json::from_str::<serde_json::Value>(data)
                        .ok()
                        .and_then(|v| {
                            v.get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| format!("SSE error: {data}"));
                    strip_partial_tool_state(&mut output, &mut tool_json_accum, &block_index_map);
                    output.stop_reason = StopReason::Error;
                    output.error_message = Some(error_msg);
                    let _ = tx
                        .send(StreamEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        })
                        .await;
                    return;
                }
                _ => {}
            }
        }
        strip_partial_tool_state(&mut output, &mut tool_json_accum, &block_index_map);
        output.stop_reason = StopReason::Error;
        output.error_message = Some("Stream ended unexpectedly".into());
        let _ = tx
            .send(StreamEvent::Error {
                reason: StopReason::Error,
                error: output,
            })
            .await;
    });

    let _ = producer.await;
    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        events.push(ev);
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::{Tool as BaseTool, UserMessage};

    fn simple_context(system: Option<&str>, user_text: &str) -> Context {
        Context {
            system_prompt: system.map(String::from),
            messages: vec![Message::User(UserMessage::text(user_text))],
            tools: Vec::new(),
        }
    }

    fn build(context: &Context, options: &StreamOptions) -> serde_json::Value {
        let model = models().into_iter().next().unwrap();
        let req = build_request_body(&model, context, options).unwrap();
        serde_json::to_value(req).unwrap()
    }

    // -- cache breakpoints & system --

    #[test]
    fn system_prompt_has_cache_control() {
        let body = build(
            &simple_context(Some("Be helpful."), "hi"),
            &StreamOptions::default(),
        );
        let blocks = body["system"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["text"], "Be helpful.");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn last_user_message_gets_cache_breakpoint() {
        let body = build(
            &simple_context(None, "hello world"),
            &StreamOptions::default(),
        );
        let messages = body["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        let content = last["content"].as_array().unwrap();
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn last_tool_has_cache_control_only() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage::text("hi"))],
            tools: vec![
                BaseTool {
                    name: "first".into(),
                    description: "first".into(),
                    parameters: serde_json::json!({"type":"object"}),
                },
                BaseTool {
                    name: "second".into(),
                    description: "second".into(),
                    parameters: serde_json::json!({"type":"object"}),
                },
            ],
        };
        let body = build(&ctx, &StreamOptions::default());
        let tools = body["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn strip_partial_tool_state_resets() {
        let mut output = AssistantMessage::empty(API_ID, "anthropic", "claude");
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: "tc_complete".into(),
            name: "done".into(),
            arguments: serde_json::json!({"ok": true}),
        }));
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: "tc_in_flight".into(),
            name: "in_flight".into(),
            arguments: serde_json::Value::Object(Default::default()),
        }));
        let block_index_map: Vec<(u64, usize)> = vec![(0, 0), (1, 1)];
        let mut accum: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
        accum.insert(1, "{\"partial\": \"val".to_string());
        strip_partial_tool_state(&mut output, &mut accum, &block_index_map);
        assert!(accum.is_empty());
        assert_eq!(
            output.content[0],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_complete".into(),
                name: "done".into(),
                arguments: serde_json::json!({"ok": true})
            })
        );
        assert_eq!(
            output.content[1],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_in_flight".into(),
                name: "in_flight".into(),
                arguments: serde_json::Value::Object(Default::default())
            })
        );
    }

    #[test]
    fn map_stop_reason_exhaustive() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::Stop);
        assert_eq!(map_stop_reason("pause_turn"), StopReason::Stop);
        assert_eq!(map_stop_reason("stop_sequence"), StopReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("unknown"), StopReason::Error);
        assert_eq!(map_stop_reason(""), StopReason::Error);
    }

    #[test]
    fn http_status_mapping() {
        assert!(matches!(
            map_http_status(401, "".into(), None, "a"),
            tars_base::Error::Auth(_)
        ));
        assert!(matches!(
            map_http_status(429, "".into(), Some(7), "a"),
            tars_base::Error::RateLimited {
                retry_after: Some(7),
                ..
            }
        ));
        assert!(matches!(
            map_http_status(408, "".into(), None, "a"),
            tars_base::Error::Timeout(_)
        ));
        assert!(matches!(
            map_http_status(413, "".into(), None, "a"),
            tars_base::Error::ContextOverflow
        ));
        assert!(matches!(
            map_http_status(500, "oops".into(), None, "a"),
            tars_base::Error::Http { status: 500, .. }
        ));
    }

    // -- SSE fixture tests --

    #[tokio::test]
    async fn sse_text_stream() {
        let model = models()
            .into_iter()
            .find(|m| m.id == "claude-sonnet-4-6")
            .unwrap();
        let raw = [
            "event: message_start",
            r#"data: {"message":{"id":"msg_1","usage":{"input_tokens":10}}}"#,
            "event: content_block_start",
            r#"data: {"index":0,"content_block":{"type":"text","text":""}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            "event: content_block_stop",
            r#"data: {"index":0}"#,
            "event: message_delta",
            r#"data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            "event: message_stop",
            r#"data: {}"#,
        ]
        .join("\n");
        let events =
            collect_events_from_sse_fixture(API_ID, "anthropic", "claude-sonnet-4-6", &model, &raw)
                .await;
        assert!(matches!(events[0], StreamEvent::Start { .. }));
        assert!(matches!(
            events[1],
            StreamEvent::TextStart {
                content_index: 0,
                ..
            }
        ));
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["Hello", " world"]);
        let done = events.last().unwrap();
        match done {
            StreamEvent::Done { reason, message } => {
                assert_eq!(*reason, StopReason::Stop);
                assert_eq!(message.usage.input, 10);
                assert_eq!(message.usage.output, 5);
                assert_eq!(message.text(), "Hello world");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sse_tool_use_stream() {
        let model = models()
            .into_iter()
            .find(|m| m.id == "claude-sonnet-4-6")
            .unwrap();
        let raw = [
            "event: message_start",
            r#"data: {"message":{"id":"msg_2"}}"#,
            "event: content_block_start",
            r#"data: {"index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":"}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}"#,
            "event: content_block_stop",
            r#"data: {"index":0}"#,
            "event: message_delta",
            r#"data: {"delta":{"stop_reason":"tool_use"}}"#,
            "event: message_stop",
            r#"data: {}"#,
        ]
        .join("\n");
        let events =
            collect_events_from_sse_fixture(API_ID, "anthropic", "claude-sonnet-4-6", &model, &raw)
                .await;
        assert!(matches!(
            events[1],
            StreamEvent::ToolcallStart {
                content_index: 0,
                ..
            }
        ));
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolcallDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["{\"cmd\":", "\"ls\"}"]);
        let end = events
            .iter()
            .find(|e| matches!(e, StreamEvent::ToolcallEnd { .. }))
            .unwrap();
        match end {
            StreamEvent::ToolcallEnd { tool_call, .. } => {
                assert_eq!(tool_call.id, "toolu_1");
                assert_eq!(tool_call.name, "bash");
                assert_eq!(tool_call.arguments, serde_json::json!({"cmd":"ls"}));
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            events.last().unwrap(),
            StreamEvent::Done {
                reason: StopReason::ToolUse,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn sse_thinking_stream() {
        let model = models()
            .into_iter()
            .find(|m| m.id == "claude-sonnet-4-6")
            .unwrap();
        let raw = [
            "event: message_start",
            r#"data: {"message":{"id":"msg_3"}}"#,
            "event: content_block_start",
            r#"data: {"index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"thinking_delta","thinking":"Let me"}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"thinking_delta","thinking":" think"}}"#,
            "event: content_block_stop",
            r#"data: {"index":0}"#,
            "event: content_block_start",
            r#"data: {"index":1,"content_block":{"type":"text","text":""}}"#,
            "event: content_block_delta",
            r#"data: {"index":1,"delta":{"type":"text_delta","text":"Answer"}}"#,
            "event: content_block_stop",
            r#"data: {"index":1}"#,
            "event: message_delta",
            r#"data: {"delta":{"stop_reason":"end_turn"}}"#,
            "event: message_stop",
            r#"data: {}"#,
        ]
        .join("\n");
        let events =
            collect_events_from_sse_fixture(API_ID, "anthropic", "claude-sonnet-4-6", &model, &raw)
                .await;
        assert!(matches!(events[1], StreamEvent::ThinkingStart { .. }));
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkingDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["Let me", " think"]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ThinkingEnd { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextStart { .. }))
        );
        assert!(matches!(
            events.last().unwrap(),
            StreamEvent::Done {
                reason: StopReason::Stop,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn sse_cache_usage() {
        let model = models()
            .into_iter()
            .find(|m| m.id == "claude-sonnet-4-6")
            .unwrap();
        let raw = [
            "event: message_start",
            r#"data: {"message":{"id":"msg_4","usage":{"input_tokens":100,"cache_read_input_tokens":40,"cache_creation_input_tokens":10}}}"#,
            "event: content_block_start",
            r#"data: {"index":0,"content_block":{"type":"text","text":""}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            "event: content_block_stop",
            r#"data: {"index":0}"#,
            "event: message_delta",
            r#"data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            "event: message_stop",
            r#"data: {}"#,
        ]
        .join("\n");
        let events =
            collect_events_from_sse_fixture(API_ID, "anthropic", "claude-sonnet-4-6", &model, &raw)
                .await;
        match events.last().unwrap() {
            StreamEvent::Done { message, .. } => {
                assert_eq!(message.usage.input, 100);
                assert_eq!(message.usage.cache_read, 40);
                assert_eq!(message.usage.cache_write, 10);
                assert_eq!(message.usage.output, 5);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sse_error_scrubs_partial_tool() {
        let model = models()
            .into_iter()
            .find(|m| m.id == "claude-sonnet-4-6")
            .unwrap();
        let raw = [
            "event: message_start",
            r#"data: {"message":{"id":"msg_5"}}"#,
            "event: content_block_start",
            r#"data: {"index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}"#,
            "event: content_block_delta",
            r#"data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"partial\":"}}"#,
            "event: error",
            r#"data: {"error":{"type":"overloaded_error","message":"overloaded"}}"#,
        ]
        .join("\n");
        let events =
            collect_events_from_sse_fixture(API_ID, "anthropic", "claude-sonnet-4-6", &model, &raw)
                .await;
        let err = events
            .iter()
            .find(|e| matches!(e, StreamEvent::Error { .. }))
            .expect("error event");
        match err {
            StreamEvent::Error { error, .. } => {
                // partial JSON should have been scrubbed
                let tc = match &error.content[0] {
                    AssistantContent::ToolCall(tc) => tc,
                    other => panic!("expected ToolCall, got {other:?}"),
                };
                assert_eq!(tc.arguments, serde_json::Value::Object(Default::default()));
                assert_eq!(error.error_message.as_deref(), Some("overloaded"));
            }
            _ => unreachable!(),
        }
    }
}
