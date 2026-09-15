pub mod bash;
pub mod diagnostics_scan;
pub mod edit;
pub mod get_function;
pub mod line_hash;
pub mod read;
pub mod skeleton;
pub mod tree_sitter_support;
pub mod write;

use std::path::{Path, PathBuf};

use tars_base::{CancelToken, Tool, ToolCall, ToolResultMessage, timestamp_ms};
use tars_base::{ImageContent, TextContent, ToolResultContent};

#[allow(dead_code)]
pub(crate) fn resolve_path(cwd: &str, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(cwd).join(p)
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
    pub summary: Option<String>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            is_error: false,
            summary: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            is_error: true,
            summary: None,
        }
    }

    pub fn image(data: String, mime_type: String) -> Self {
        Self {
            content: vec![ToolResultContent::Image(ImageContent { data, mime_type })],
            is_error: false,
            summary: None,
        }
    }

    pub fn with_image(mut self, data: String, mime_type: String) -> Self {
        self.content
            .push(ToolResultContent::Image(ImageContent { data, mime_type }));
        self
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }
}

pub struct ToolDef {
    pub tool: Tool,
    #[allow(clippy::type_complexity)]
    pub execute: Box<dyn Fn(serde_json::Value, &str, &CancelToken) -> ToolOutput + Send + Sync>,
    #[allow(clippy::type_complexity)]
    pub prepare_arguments:
        Option<Box<dyn Fn(serde_json::Value) -> serde_json::Value + Send + Sync>>,
}

pub fn execute_tool(
    tools: &[ToolDef],
    tool_call: &ToolCall,
    cwd: &str,
    cancel: &CancelToken,
) -> ToolResultMessage {
    if cancel.is_cancelled() {
        return ToolResultMessage {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            content: vec![ToolResultContent::Text(TextContent {
                text: "error: cancelled before execution".into(),
                text_signature: None,
            })],
            details: None,
            is_error: true,
            timestamp: timestamp_ms(),
            duration_ms: None,
            summary: None,
            post_persist_actions: Vec::new(),
        };
    }

    let result = match tools.iter().find(|t| t.tool.name == tool_call.name) {
        Some(def) => {
            let args = match &def.prepare_arguments {
                Some(prepare) => prepare(tool_call.arguments.clone()),
                None => tool_call.arguments.clone(),
            };
            (def.execute)(args, cwd, cancel)
        }
        None => ToolOutput::error(format!("unknown tool: {}", tool_call.name)),
    };

    ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: result.content,
        details: None,
        is_error: result.is_error,
        timestamp: timestamp_ms(),
        duration_ms: None,
        summary: result.summary,
        post_persist_actions: Vec::new(),
    }
}

pub fn default_tools() -> Vec<ToolDef> {
    vec![
        bash::tool_def(),
        read::tool_def(),
        write::tool_def(),
        edit::tool_def(),
        skeleton::tool_def(),
        get_function::tool_def(),
        diagnostics_scan::tool_def(),
    ]
}

pub fn tool_schemas(tools: &[ToolDef]) -> Vec<Tool> {
    tools.iter().map(|t| t.tool.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dummy_tool(name: &str) -> ToolDef {
        ToolDef {
            tool: Tool {
                name: name.into(),
                description: format!("dummy {name}"),
                parameters: json!({"type":"object"}),
            },
            execute: Box::new(|args, _cwd, _cancel| {
                ToolOutput::text(format!("executed with {}", args))
            }),
            prepare_arguments: None,
        }
    }

    #[test]
    fn dispatch_known_tool() {
        let tools = vec![dummy_tool("echo")];
        let tc = ToolCall {
            id: "tc1".into(),
            name: "echo".into(),
            arguments: json!({"msg":"hi"}),
        };
        let res = execute_tool(&tools, &tc, "/tmp", &CancelToken::new());
        assert!(!res.is_error);
        assert_eq!(res.tool_call_id, "tc1");
        assert!(res.text().contains("hi"));
    }

    #[test]
    fn dispatch_unknown_tool() {
        let tools = vec![dummy_tool("echo")];
        let tc = ToolCall {
            id: "tc2".into(),
            name: "unknown".into(),
            arguments: json!({}),
        };
        let res = execute_tool(&tools, &tc, "/tmp", &CancelToken::new());
        assert!(res.is_error);
        assert!(res.text().contains("unknown tool"));
    }

    #[test]
    fn dispatch_pre_cancelled() {
        let tools = vec![dummy_tool("echo")];
        let tc = ToolCall {
            id: "tc3".into(),
            name: "echo".into(),
            arguments: json!({}),
        };
        let cancel = CancelToken::new();
        cancel.cancel();
        let res = execute_tool(&tools, &tc, "/tmp", &cancel);
        assert!(res.is_error);
        assert_eq!(res.text(), "error: cancelled before execution");
        assert_eq!(res.tool_name, "echo");
    }

    #[test]
    fn prepare_arguments_hook() {
        let tool = ToolDef {
            tool: Tool {
                name: "legacy".into(),
                description: "legacy".into(),
                parameters: json!({"type":"object"}),
            },
            execute: Box::new(|args, _, _| {
                // expects normalized field "path"
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                ToolOutput::text(path.to_string())
            }),
            prepare_arguments: Some(Box::new(|mut v| {
                // legacy used "file" instead of "path"
                if let Some(obj) = v.as_object_mut()
                    && let Some(file) = obj.remove("file")
                {
                    obj.insert("path".to_string(), file);
                }
                v
            })),
        };
        let tc = ToolCall {
            id: "tc4".into(),
            name: "legacy".into(),
            arguments: json!({"file":"old.txt"}),
        };
        let res = execute_tool(&[tool], &tc, "/tmp", &CancelToken::new());
        assert!(!res.is_error);
        assert_eq!(res.text(), "old.txt");
    }

    #[test]
    fn default_tools_contains_bash() {
        let tools = default_tools();
        assert!(tools.iter().any(|t| t.tool.name == "bash"));
    }

    #[test]
    fn tool_schemas_extracts_tools() {
        let tools = vec![dummy_tool("a"), dummy_tool("b")];
        let schemas = tool_schemas(&tools);
        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].name, "a");
        assert_eq!(schemas[1].name, "b");
    }
}
