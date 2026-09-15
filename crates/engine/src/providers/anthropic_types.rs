//! Typed structs for the Anthropic Messages wire format.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    pub max_tokens: u64,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

#[derive(Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Serialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: &'static str,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral",
        }
    }
}

#[derive(Serialize)]
pub struct ApiMessage {
    pub role: &'static str,
    pub content: serde_json::Value,
}

#[derive(Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub eager_input_streaming: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<&'static str>,
}

#[derive(Serialize)]
pub struct OutputConfig {
    pub effort: &'static str,
}

// SSE

#[derive(Deserialize, Debug)]
pub struct ApiUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

impl ApiUsage {
    pub fn apply_to(&self, usage: &mut tars_base::Usage) {
        if let Some(n) = self.input_tokens {
            usage.input = n;
        }
        if let Some(n) = self.output_tokens {
            usage.output = n;
        }
        if let Some(n) = self.cache_read_input_tokens {
            usage.cache_read = n;
        }
        if let Some(n) = self.cache_creation_input_tokens {
            usage.cache_write = n;
        }
        usage.recompute_total();
    }
}

#[derive(Deserialize, Debug)]
pub struct MessageStartEvent {
    pub message: MessageStartMessage,
}

#[derive(Deserialize, Debug)]
pub struct MessageStartMessage {
    pub id: String,
    #[serde(default)]
    pub usage: Option<ApiUsage>,
}

#[derive(Deserialize, Debug)]
pub struct ContentBlockStartEvent {
    pub index: u64,
    pub content_block: ContentBlock,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
pub struct ContentBlockDeltaEvent {
    pub index: u64,
    pub delta: Delta,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    InputJsonDelta { partial_json: String },
    SignatureDelta { signature: String },
}

#[derive(Deserialize, Debug)]
pub struct ContentBlockStopEvent {
    pub index: u64,
}

#[derive(Deserialize, Debug)]
pub struct MessageDeltaEvent {
    #[serde(default)]
    pub delta: Option<MessageDelta>,
    #[serde(default)]
    pub usage: Option<ApiUsage>,
}

#[derive(Deserialize, Debug)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}
