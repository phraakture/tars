//! OpenAI Chat Completions API provider.
//!
//! Also used for OpenAI-compatible APIs (OpenRouter, Groq, Qwen, LiteLLM, …)
//! via different `base_url` and model settings.

use async_trait::async_trait;
use futures::StreamExt;

use crate::{EventReceiver, EventSender, Provider, STREAM_CAPACITY};
use tars_base::{
    AssistantContent, AssistantMessage, CacheRetention, Context, Message, Model, StopReason,
    StreamEvent, StreamOptions, TextContent, ThinkingContent, ThinkingStyle, ToolCall,
    ToolResultContent, UserContent,
};

use super::openai_types;

const API_ID: &str = "openai-completions";

pub struct OpenAi;

#[async_trait]
impl Provider for OpenAi {
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
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let api_id = model.api.clone();
        let provider_name = model.provider.clone();
        let model_id = model.id.clone();
        let model_clone = model.clone();
        let extra_headers = model.headers.clone();
        let options_clone = options.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CAPACITY);

        tokio::spawn(async move {
            let result = run_openai_stream(
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
                // If the channel is already closed, ignore.
                let _ = emit_error(&tx, &api_id, &provider_name, &model_id, e).await;
            }
        });

        Ok(rx)
    }
}

// ---------------------------------------------------------------------------
// HTTP streaming
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_openai_stream(
    base_url: &str,
    api_key: &str,
    api_id: &str,
    provider_name: &str,
    model_id: &str,
    model: &Model,
    body: &openai_types::ChatCompletionRequest,
    extra_model_headers: std::collections::HashMap<String, String>,
    options: &StreamOptions,
    tx: &EventSender,
) -> tars_base::Result<()> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(|e| tars_base::Error::Internal(e.to_string()))?;

    let mut req = client
        .post(&url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .json(body);

    if !api_key.is_empty() {
        req = req.header("authorization", format!("Bearer {api_key}"));
    }
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

    // SSE parsing via bytes_stream
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    struct ToolAccum {
        id: String,
        name: String,
        arguments: String,
        content_index: usize,
    }
    let mut tool_accums: Vec<ToolAccum> = Vec::new();
    let mut text_started = false;
    let mut thinking_started = false;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| tars_base::Error::Http {
            status: 0,
            message: e.to_string(),
        })?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete lines
        while let Some(nl_pos) = buf.find('\n') {
            let line = buf[..nl_pos].to_string();
            buf = buf[nl_pos + 1..].to_string();
            let line = line.trim_end_matches('\r');

            if line.trim().is_empty() {
                continue;
            }
            let Some(data) = line
                .strip_prefix("data:")
                .map(|s| s.trim_start_matches(' '))
            else {
                continue;
            };

            if data == "[DONE]" {
                buf.clear();
                break;
            }

            let chunk: openai_types::ChatCompletionChunk =
                serde_json::from_str(data).map_err(|e| tars_base::Error::Parse(e.to_string()))?;

            if output.response_id.is_none() {
                if let Some(id) = chunk.id.clone() {
                    output.response_id = Some(id);
                }
            }

            if let Some(usage) = chunk.usage {
                usage.apply_to(&mut output.usage);
                model.calculate_cost(&mut output.usage);
            }

            let Some(choice) = chunk.choices.first() else {
                continue;
            };

            if let Some(ref reason) = choice.finish_reason {
                output.stop_reason = map_finish_reason(reason);
            }

            let delta = &choice.delta;

            // Reasoning (thinking)
            if let Some(ref thinking) = delta.reasoning_content {
                if !thinking_started {
                    output
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: false,
                        }));
                    thinking_started = true;
                    send_event(
                        tx,
                        StreamEvent::ThinkingStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        },
                    )
                    .await?;
                }
                let ci = output.content.len() - 1;
                if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci) {
                    t.thinking.push_str(thinking);
                }
                send_event(
                    tx,
                    StreamEvent::ThinkingDelta {
                        content_index: ci,
                        delta: thinking.clone(),
                        partial: output.clone(),
                    },
                )
                .await?;
            }

            // Text content
            if let Some(ref text) = delta.content {
                if thinking_started {
                    let ci = output.content.len() - 1;
                    if let Some(AssistantContent::Thinking(t)) = output.content.get(ci) {
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
                    thinking_started = false;
                }

                if !text_started {
                    output.content.push(AssistantContent::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }));
                    text_started = true;
                    send_event(
                        tx,
                        StreamEvent::TextStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        },
                    )
                    .await?;
                }
                let ci = output.content.len() - 1;
                if let Some(AssistantContent::Text(t)) = output.content.get_mut(ci) {
                    t.text.push_str(text);
                }
                send_event(
                    tx,
                    StreamEvent::TextDelta {
                        content_index: ci,
                        delta: text.clone(),
                        partial: output.clone(),
                    },
                )
                .await?;
            }

            // Tool calls
            if let Some(ref tool_calls) = delta.tool_calls {
                if text_started {
                    let ci = output.content.len() - 1;
                    if let Some(AssistantContent::Text(t)) = output.content.get(ci) {
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
                    text_started = false;
                }

                for tc in tool_calls {
                    while tool_accums.len() <= tc.index {
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::Value::Object(Default::default()),
                        }));
                        let ci = output.content.len() - 1;
                        tool_accums.push(ToolAccum {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            content_index: ci,
                        });
                        send_event(
                            tx,
                            StreamEvent::ToolcallStart {
                                content_index: ci,
                                partial: output.clone(),
                            },
                        )
                        .await?;
                    }

                    let accum = &mut tool_accums[tc.index];
                    if let Some(ref id) = tc.id {
                        accum.id = id.clone();
                    }
                    if let Some(ref func) = tc.function {
                        if let Some(ref name) = func.name {
                            accum.name.push_str(name);
                        }
                        if let Some(ref args) = func.arguments {
                            if !args.is_empty() {
                                accum.arguments.push_str(args);
                                send_event(
                                    tx,
                                    StreamEvent::ToolcallDelta {
                                        content_index: accum.content_index,
                                        delta: args.clone(),
                                        partial: output.clone(),
                                    },
                                )
                                .await?;
                            } else {
                                accum.arguments.push_str(args);
                            }
                        }
                    }
                }
            }
        }
    }

    // Handle any trailing buffered data without final newline
    if !buf.trim().is_empty() {
        let line = buf.trim();
        if let Some(data) = line.strip_prefix("data:").map(|s| s.trim()) {
            if data != "[DONE]" && !data.is_empty() {
                if let Ok(chunk) = serde_json::from_str::<openai_types::ChatCompletionChunk>(data) {
                    let _ = chunk;
                }
            }
        }
    }

    if thinking_started {
        let ci = output.content.len() - 1;
        if let Some(AssistantContent::Thinking(t)) = output.content.get(ci) {
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
    }
    if text_started {
        let ci = output.content.len() - 1;
        if let Some(AssistantContent::Text(t)) = output.content.get(ci) {
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
    }

    for accum in &tool_accums {
        let args: serde_json::Value = serde_json::from_str(&accum.arguments)
            .unwrap_or(serde_json::Value::Object(Default::default()));
        if let Some(AssistantContent::ToolCall(tc)) = output.content.get_mut(accum.content_index) {
            tc.id = accum.id.clone();
            tc.name = accum.name.clone();
            tc.arguments = args;
        }
        send_event(
            tx,
            StreamEvent::ToolcallEnd {
                content_index: accum.content_index,
                tool_call: ToolCall {
                    id: accum.id.clone(),
                    name: accum.name.clone(),
                    arguments: serde_json::from_str(&accum.arguments)
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                },
                partial: output.clone(),
            },
        )
        .await?;
    }

    send_event(
        tx,
        StreamEvent::Done {
            reason: output.stop_reason,
            message: output,
        },
    )
    .await?;

    Ok(())
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
    if err.is_connect() {
        return tars_base::Error::Http {
            status: 0,
            message: err.to_string(),
        };
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

pub(crate) fn build_request_body(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> tars_base::Result<openai_types::ChatCompletionRequest> {
    let mut messages = Vec::new();

    if let Some(ref prompt) = context.system_prompt {
        messages.push(openai_types::ChatMessage {
            role: "system".into(),
            content: Some(serde_json::Value::String(prompt.clone())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    for msg in &context.messages {
        match msg {
            Message::User(u) => {
                let content = convert_user_content(&u.content);
                messages.push(openai_types::ChatMessage {
                    role: "user".into(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            Message::Assistant(a) => {
                let (content, tool_calls) = convert_assistant_to_openai(a);
                messages.push(openai_types::ChatMessage {
                    role: "assistant".into(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                    name: None,
                });
            }
            Message::ToolResult(tr) => {
                let text = tr
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ToolResultContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(openai_types::ChatMessage {
                    role: "tool".into(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: None,
                    tool_call_id: Some(tr.tool_call_id.clone()),
                    name: Some(tr.tool_name.clone()),
                });
            }
            Message::CompactionSummary(cs) => {
                let text = format!(
                    "[Context compacted — {} tokens before compaction]\n\n{}",
                    cs.tokens_before, cs.summary
                );
                messages.push(openai_types::ChatMessage {
                    role: "user".into(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            Message::Info(_) => {}
        }
    }

    let max_tokens = options
        .max_tokens
        .unwrap_or((model.max_tokens / 3).max(1024));

    let tools = if context.tools.is_empty() {
        None
    } else {
        Some(
            context
                .tools
                .iter()
                .map(|t| openai_types::ToolDef {
                    tool_type: "function",
                    function: openai_types::ToolDefFunction {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    };

    let (reasoning_effort, enable_thinking) = match model.thinking {
        ThinkingStyle::OpenAi => (Some("medium".to_string()), None),
        ThinkingStyle::Qwen => (None, Some(true)),
        _ => (None, None),
    };

    let is_openai = model.base_url.contains("api.openai.com");
    let retention = CacheRetention::resolve(options.cache_retention);
    let prompt_cache_key = if is_openai && retention != CacheRetention::None {
        options.session_id.clone()
    } else {
        None
    };
    let prompt_cache_retention = if is_openai && retention == CacheRetention::Long {
        Some("24h")
    } else {
        None
    };

    Ok(openai_types::ChatCompletionRequest {
        model: model.id.clone(),
        messages,
        max_completion_tokens: Some(max_tokens),
        temperature: options.temperature,
        tools,
        stream: true,
        stream_options: Some(openai_types::StreamOptions {
            include_usage: true,
        }),
        reasoning_effort,
        enable_thinking,
        prompt_cache_key,
        prompt_cache_retention,
    })
}

fn convert_user_content(content: &[UserContent]) -> serde_json::Value {
    if content.len() == 1
        && let UserContent::Text(t) = &content[0]
    {
        return serde_json::Value::String(t.text.clone());
    }
    let parts: Vec<serde_json::Value> = content
        .iter()
        .map(|c| match c {
            UserContent::Text(t) => serde_json::json!({"type": "text", "text": t.text}),
            UserContent::Image(img) => serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", img.mime_type, img.data),
                }
            }),
        })
        .collect();
    serde_json::Value::Array(parts)
}

fn convert_assistant_to_openai(
    a: &AssistantMessage,
) -> (
    Option<serde_json::Value>,
    Option<Vec<openai_types::ToolCallMessage>>,
) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for c in &a.content {
        match c {
            AssistantContent::Text(t) if !t.text.is_empty() => {
                text_parts.push(t.text.as_str());
            }
            AssistantContent::Thinking(t) if !t.thinking.is_empty() => {
                text_parts.push(t.thinking.as_str());
            }
            AssistantContent::ToolCall(tc) => {
                tool_calls.push(openai_types::ToolCallMessage {
                    id: tc.id.clone(),
                    call_type: "function".into(),
                    function: openai_types::ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(text_parts.join("")))
    };
    let tcs = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };
    (content, tcs)
}

// ---------------------------------------------------------------------------
// Finish reason
// ---------------------------------------------------------------------------

pub fn map_finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        "content_filter" => StopReason::Error,
        _ => StopReason::Stop,
    }
}

// ---------------------------------------------------------------------------
// Test helpers: SSE fixture parsing without network
// ---------------------------------------------------------------------------

/// Parse a sequence of raw SSE lines (as they appear on the wire, including
/// the `data:` prefix and `[DONE]`) into a vector of `StreamEvent`s using
/// the same state machine as the live provider. Used for fixture tests.
#[cfg(test)]
pub(crate) async fn collect_events_from_sse_fixture(
    api_id: &str,
    provider_name: &str,
    model_id: &str,
    model: &Model,
    sse_lines: &[&str],
) -> Vec<StreamEvent> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(128);
    let sse_owned: Vec<String> = sse_lines.iter().map(|s| s.to_string()).collect();
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

        struct ToolAccum {
            id: String,
            name: String,
            arguments: String,
            content_index: usize,
        }
        let mut tool_accums: Vec<ToolAccum> = Vec::new();
        let mut text_started = false;
        let mut thinking_started = false;

        for line in sse_owned {
            if line.trim().is_empty() {
                continue;
            }
            let Some(data) = line
                .strip_prefix("data:")
                .map(|s| s.trim_start_matches(' '))
            else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let chunk: openai_types::ChatCompletionChunk =
                serde_json::from_str(data).expect("valid fixture json");

            if output.response_id.is_none() {
                if let Some(id) = chunk.id.clone() {
                    output.response_id = Some(id);
                }
            }
            if let Some(usage) = chunk.usage {
                usage.apply_to(&mut output.usage);
                model_clone.calculate_cost(&mut output.usage);
            }
            let Some(choice) = chunk.choices.first() else {
                continue;
            };
            if let Some(ref reason) = choice.finish_reason {
                output.stop_reason = map_finish_reason(reason);
            }
            let delta = &choice.delta;
            if let Some(ref thinking) = delta.reasoning_content {
                if !thinking_started {
                    output
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: false,
                        }));
                    thinking_started = true;
                    let _ = tx
                        .send(StreamEvent::ThinkingStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        })
                        .await;
                }
                let ci = output.content.len() - 1;
                if let Some(AssistantContent::Thinking(t)) = output.content.get_mut(ci) {
                    t.thinking.push_str(thinking);
                }
                let _ = tx
                    .send(StreamEvent::ThinkingDelta {
                        content_index: ci,
                        delta: thinking.clone(),
                        partial: output.clone(),
                    })
                    .await;
            }
            if let Some(ref text) = delta.content {
                if thinking_started {
                    let ci = output.content.len() - 1;
                    if let Some(AssistantContent::Thinking(t)) = output.content.get(ci) {
                        let _ = tx
                            .send(StreamEvent::ThinkingEnd {
                                content_index: ci,
                                content: t.thinking.clone(),
                                partial: output.clone(),
                            })
                            .await;
                    }
                    thinking_started = false;
                }
                if !text_started {
                    output.content.push(AssistantContent::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }));
                    text_started = true;
                    let _ = tx
                        .send(StreamEvent::TextStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        })
                        .await;
                }
                let ci = output.content.len() - 1;
                if let Some(AssistantContent::Text(t)) = output.content.get_mut(ci) {
                    t.text.push_str(text);
                }
                let _ = tx
                    .send(StreamEvent::TextDelta {
                        content_index: ci,
                        delta: text.clone(),
                        partial: output.clone(),
                    })
                    .await;
            }
            if let Some(ref tool_calls) = delta.tool_calls {
                if text_started {
                    let ci = output.content.len() - 1;
                    if let Some(AssistantContent::Text(t)) = output.content.get(ci) {
                        let _ = tx
                            .send(StreamEvent::TextEnd {
                                content_index: ci,
                                content: t.text.clone(),
                                partial: output.clone(),
                            })
                            .await;
                    }
                    text_started = false;
                }
                for tc in tool_calls {
                    while tool_accums.len() <= tc.index {
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::Value::Object(Default::default()),
                        }));
                        let ci = output.content.len() - 1;
                        tool_accums.push(ToolAccum {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            content_index: ci,
                        });
                        let _ = tx
                            .send(StreamEvent::ToolcallStart {
                                content_index: ci,
                                partial: output.clone(),
                            })
                            .await;
                    }
                    let accum = &mut tool_accums[tc.index];
                    if let Some(ref id) = tc.id {
                        accum.id = id.clone();
                    }
                    if let Some(ref func) = tc.function {
                        if let Some(ref name) = func.name {
                            accum.name.push_str(name);
                        }
                        if let Some(ref args) = func.arguments {
                            if !args.is_empty() {
                                accum.arguments.push_str(args);
                                let _ = tx
                                    .send(StreamEvent::ToolcallDelta {
                                        content_index: accum.content_index,
                                        delta: args.clone(),
                                        partial: output.clone(),
                                    })
                                    .await;
                            } else {
                                accum.arguments.push_str(args);
                            }
                        }
                    }
                }
            }
        }

        if thinking_started {
            let ci = output.content.len() - 1;
            if let Some(AssistantContent::Thinking(t)) = output.content.get(ci) {
                let _ = tx
                    .send(StreamEvent::ThinkingEnd {
                        content_index: ci,
                        content: t.thinking.clone(),
                        partial: output.clone(),
                    })
                    .await;
            }
        }
        if text_started {
            let ci = output.content.len() - 1;
            if let Some(AssistantContent::Text(t)) = output.content.get(ci) {
                let _ = tx
                    .send(StreamEvent::TextEnd {
                        content_index: ci,
                        content: t.text.clone(),
                        partial: output.clone(),
                    })
                    .await;
            }
        }
        for accum in &tool_accums {
            let args: serde_json::Value = serde_json::from_str(&accum.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            if let Some(AssistantContent::ToolCall(tc)) =
                output.content.get_mut(accum.content_index)
            {
                tc.id = accum.id.clone();
                tc.name = accum.name.clone();
                tc.arguments = args;
            }
            let _ = tx
                .send(StreamEvent::ToolcallEnd {
                    content_index: accum.content_index,
                    tool_call: ToolCall {
                        id: accum.id.clone(),
                        name: accum.name.clone(),
                        arguments: serde_json::from_str(&accum.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default())),
                    },
                    partial: output.clone(),
                })
                .await;
        }
        let _ = tx
            .send(StreamEvent::Done {
                reason: output.stop_reason,
                message: output,
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
    use tars_base::{CacheRetention, ModelCost, ThinkingStyle, UserMessage};

    fn test_model(base_url: &str) -> Model {
        Model {
            id: "gpt-test".into(),
            name: "GPT Test".into(),
            api: API_ID.into(),
            provider: "openai".into(),
            base_url: base_url.into(),
            thinking: ThinkingStyle::None,
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_000,
            headers: Default::default(),
        }
    }

    fn ctx() -> Context {
        Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage::text("hi"))],
            tools: Vec::new(),
        }
    }

    fn build(base_url: &str, options: &StreamOptions) -> serde_json::Value {
        let model = test_model(base_url);
        let req = build_request_body(&model, &ctx(), options).expect("build_request_body");
        serde_json::to_value(req).expect("serialize")
    }

    #[test]
    fn prompt_cache_key_set_for_openai_with_session_id() {
        let opts = StreamOptions {
            session_id: Some("sess-abc".into()),
            ..Default::default()
        };
        let body = build("https://api.openai.com/v1", &opts);
        assert_eq!(body["prompt_cache_key"], "sess-abc");
    }

    #[test]
    fn prompt_cache_key_omitted_for_compatible_backend() {
        let opts = StreamOptions {
            session_id: Some("sess-abc".into()),
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        };
        let body = build("https://api.litellm.example/v1", &opts);
        assert!(
            body.get("prompt_cache_key").is_none(),
            "non-OpenAI backends must not receive prompt_cache_key, got {body}"
        );
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "non-OpenAI backends must not receive prompt_cache_retention, got {body}"
        );
    }

    #[test]
    fn prompt_cache_retention_long_emits_24h() {
        let opts = StreamOptions {
            session_id: Some("sess-abc".into()),
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        };
        let body = build("https://api.openai.com/v1", &opts);
        assert_eq!(body["prompt_cache_retention"], "24h");
        assert_eq!(body["prompt_cache_key"], "sess-abc");
    }

    #[test]
    fn prompt_cache_retention_short_omits_field() {
        let opts = StreamOptions {
            session_id: Some("sess-abc".into()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        };
        let body = build("https://api.openai.com/v1", &opts);
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "Short retention should omit prompt_cache_retention, got {body}"
        );
        assert_eq!(body["prompt_cache_key"], "sess-abc");
    }

    #[test]
    fn cache_retention_none_disables_key() {
        let opts = StreamOptions {
            session_id: Some("sess-abc".into()),
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        let body = build("https://api.openai.com/v1", &opts);
        assert!(
            body.get("prompt_cache_key").is_none(),
            "CacheRetention::None must suppress prompt_cache_key, got {body}"
        );
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "CacheRetention::None must suppress prompt_cache_retention, got {body}"
        );
    }

    #[test]
    fn prompt_cache_key_omitted_without_session_id() {
        let body = build("https://api.openai.com/v1", &StreamOptions::default());
        assert!(
            body.get("prompt_cache_key").is_none(),
            "no session_id means no prompt_cache_key, got {body}"
        );
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "default retention is Short -> no prompt_cache_retention, got {body}"
        );
    }

    #[test]
    fn map_finish_reason_exhaustive() {
        assert_eq!(map_finish_reason("stop"), StopReason::Stop);
        assert_eq!(map_finish_reason("length"), StopReason::Length);
        assert_eq!(map_finish_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(map_finish_reason("content_filter"), StopReason::Error);
        assert_eq!(map_finish_reason("unknown"), StopReason::Stop);
        assert_eq!(map_finish_reason(""), StopReason::Stop);
    }

    #[tokio::test]
    async fn sse_text_stream() {
        let model = test_model("https://api.openai.com/v1");
        let lines = [
            r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            "data: [DONE]",
        ];
        let events =
            collect_events_from_sse_fixture(API_ID, "openai", "gpt-test", &model, &lines).await;
        // Expect: Start, TextStart, TextDelta*2, TextEnd, Done
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
        assert!(
            matches!(events[events.len() - 2], StreamEvent::TextEnd { ref content, .. } if content == "Hello world")
        );
        match &events[events.len() - 1] {
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
    async fn sse_tool_call_accumulation() {
        let model = test_model("https://api.openai.com/v1");
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ];
        let events =
            collect_events_from_sse_fixture(API_ID, "openai", "gpt-test", &model, &lines).await;

        assert!(matches!(events[0], StreamEvent::Start { .. }));
        assert!(matches!(
            events[1],
            StreamEvent::ToolcallStart {
                content_index: 0,
                ..
            }
        ));
        let tool_deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolcallDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_deltas, vec!["{\"cmd\":", "\"ls\"}"]);

        match events
            .iter()
            .find(|e| matches!(e, StreamEvent::ToolcallEnd { .. }))
            .unwrap()
        {
            StreamEvent::ToolcallEnd { tool_call, .. } => {
                assert_eq!(tool_call.id, "call_1");
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
    async fn sse_reasoning_content_as_thinking() {
        let model = test_model("https://api.openai.com/v1");
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":" more"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "data: [DONE]",
        ];
        let events =
            collect_events_from_sse_fixture(API_ID, "openai", "gpt-test", &model, &lines).await;

        assert!(matches!(events[1], StreamEvent::ThinkingStart { .. }));
        let thinking_deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkingDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking_deltas, vec!["Let me think", " more"]);
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
    async fn sse_usage_with_cache_and_reasoning() {
        let model = test_model("https://api.openai.com/v1");
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":40},"completion_tokens_details":{"reasoning_tokens":5}}}"#,
            "data: [DONE]",
        ];
        let events =
            collect_events_from_sse_fixture(API_ID, "openai", "gpt-test", &model, &lines).await;
        match events.last().unwrap() {
            StreamEvent::Done { message, .. } => {
                assert_eq!(message.usage.input, 60);
                assert_eq!(message.usage.output, 15);
                assert_eq!(message.usage.cache_read, 40);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn http_status_mapping() {
        assert!(matches!(
            map_http_status(401, "unauthorized".into(), None, "openai"),
            tars_base::Error::Auth(_)
        ));
        assert!(matches!(
            map_http_status(429, "slow".into(), Some(7), "openai"),
            tars_base::Error::RateLimited {
                retry_after: Some(7),
                ..
            }
        ));
        assert!(matches!(
            map_http_status(408, "".into(), None, "openai"),
            tars_base::Error::Timeout(_)
        ));
        assert!(matches!(
            map_http_status(413, "".into(), None, "openai"),
            tars_base::Error::ContextOverflow
        ));
        assert!(matches!(
            map_http_status(500, "oops".into(), None, "openai"),
            tars_base::Error::Http { status: 500, .. }
        ));
    }
}
