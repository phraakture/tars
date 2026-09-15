//! Plugin wire protocol — JSON-lines over stdin/stdout between server and plugin
//! subprocesses. Simplified from tau: no hooks, no server tunnel, no session
//! orchestration for MVP.

use serde::{Deserialize, Serialize};

// ── Server → Plugin ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginRequest {
    /// Execute a tool call.
    ToolCall {
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
    /// Cancel an in-flight tool call.
    CancelToolCall { tool_call_id: String },
    /// Shut down the plugin.
    Shutdown,
}

// ── Plugin → Server ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginMessage {
    /// Register the plugin's tools (sent once on startup).
    Register(PluginRegistration),
    /// Final tool execution result.
    ToolResult(PluginToolResult),
    /// Streaming output delta while a tool runs.
    OutputDelta { tool_call_id: String, text: String },
}

/// Sent once at plugin startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistration {
    pub name: String,
    #[serde(default)]
    pub tools: Vec<PluginToolDef>,
}

/// One tool definition exposed by the plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Final result of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginToolResult {
    pub tool_call_id: String,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

// ── Conversions ─────────────────────────────────────────────────────────

impl From<&PluginToolDef> for tars_base::Tool {
    fn from(def: &PluginToolDef) -> Self {
        tars_base::Tool {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
        }
    }
}
