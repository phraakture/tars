use super::{ToolDef, ToolOutput};
use tars_base::Tool;

pub fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "write".into(),
            description:
                "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        },
        execute: Box::new(execute),
        prepare_arguments: None,
    }
}

fn execute(args: serde_json::Value, cwd: &str, _cancel: &tars_base::CancelToken) -> ToolOutput {
    let Some(path_str) = args.get("path").and_then(|p| p.as_str()) else {
        return ToolOutput::error("missing 'path' argument");
    };
    let Some(content) = args.get("content").and_then(|c| c.as_str()) else {
        return ToolOutput::error("missing 'content' argument");
    };

    let path = super::resolve_path(cwd, path_str);

    if let Some(parent) = path.parent()
        && !parent.exists()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return ToolOutput::error(format!("failed to create directory: {}", e));
    }

    match std::fs::write(&path, content) {
        Ok(()) => {
            let line_count = content.lines().count();
            let summary = format!(
                "write: {} ({} lines, {} bytes)",
                path_str,
                line_count,
                content.len()
            );
            ToolOutput::text(format!(
                "Successfully wrote {} bytes to {}",
                content.len(),
                path.display()
            ))
            .with_summary(summary)
        }
        Err(e) => ToolOutput::error(format!("failed to write {}: {}", path.display(), e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tars_base::CancelToken;

    #[test]
    fn write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("file.txt");
        let path_str = path.to_str().unwrap();
        let out = execute(
            json!({"path": path_str, "content": "hello\nworld\n"}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld\n");
        assert!(out.summary.unwrap().contains("write:"));
    }

    #[test]
    fn write_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "old").unwrap();
        let path_str = path.to_str().unwrap();
        let out = execute(
            json!({"path": path_str, "content": "new"}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    #[test]
    fn write_missing_args() {
        let out = execute(json!({"path": "a.txt"}), "/tmp", &CancelToken::new());
        assert!(out.is_error);
        assert!(out.content[0].text().contains("content"));
    }
}
