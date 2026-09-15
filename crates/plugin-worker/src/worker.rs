use async_trait::async_trait;
use tars_base::{CancelToken, ToolCall, ToolResultMessage};
use tars_plugin::ToolExecutor;

use crate::tools::{self, ToolDef};

pub struct InProcessWorker {
    cwd: String,
    tools: Vec<ToolDef>,
}

impl InProcessWorker {
    pub fn new(cwd: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            tools: tools::default_tools(),
        }
    }

    pub fn with_tools(cwd: impl Into<String>, tools: Vec<ToolDef>) -> Self {
        Self {
            cwd: cwd.into(),
            tools,
        }
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    pub fn set_cwd(&mut self, cwd: impl Into<String>) {
        self.cwd = cwd.into();
    }
}

impl Default for InProcessWorker {
    fn default() -> Self {
        Self::new("/tmp")
    }
}

#[async_trait]
impl ToolExecutor for InProcessWorker {
    async fn execute(
        &mut self,
        tool_call: &ToolCall,
        _output_tx: &tokio::sync::mpsc::Sender<String>,
        cancel: &CancelToken,
    ) -> tars_base::Result<ToolResultMessage> {
        let result = tools::execute_tool(&self.tools, tool_call, &self.cwd, cancel);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tars_base::ToolCall;

    #[tokio::test]
    async fn bash_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut worker = InProcessWorker::new(dir.path().to_str().unwrap());
        let tc = ToolCall {
            id: "tc1".into(),
            name: "bash".into(),
            arguments: json!({"command": "echo hello"}),
        };
        let (tx, _) = tokio::sync::mpsc::channel(8);
        let res = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(!res.is_error);
        assert!(res.text().contains("hello"));
    }

    #[tokio::test]
    async fn read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "content here\n").unwrap();
        let mut worker = InProcessWorker::new(dir.path().to_str().unwrap());
        let tc = ToolCall {
            id: "tc2".into(),
            name: "read".into(),
            arguments: json!({"paths": [path.to_str().unwrap()]}),
        };
        let (tx, _) = tokio::sync::mpsc::channel(8);
        let res = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(!res.is_error);
        assert!(res.text().contains("content here"));
    }

    #[tokio::test]
    async fn write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        let mut worker = InProcessWorker::new(dir.path().to_str().unwrap());
        let tc = ToolCall {
            id: "tc3".into(),
            name: "write".into(),
            arguments: json!({"path": path.to_str().unwrap(), "content": "hello write"}),
        };
        let (tx, _) = tokio::sync::mpsc::channel(8);
        let res = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(!res.is_error);
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("hello write")
        );
    }

    #[tokio::test]
    async fn edit_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        std::fs::write(&path, "hello world").unwrap();
        let mut worker = InProcessWorker::new(dir.path().to_str().unwrap());
        let tc = ToolCall {
            id: "tc4".into(),
            name: "edit".into(),
            arguments: json!({
                "files": [{
                    "path": path.to_str().unwrap(),
                    "edits": [{"old_text": "world", "new_text": "there"}]
                }]
            }),
        };
        let (tx, _) = tokio::sync::mpsc::channel(8);
        let res = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(!res.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello there");
    }

    #[tokio::test]
    async fn worker_respects_cwd() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        std::fs::write(dir1.path().join("file.txt"), "from dir1").unwrap();
        let mut worker = InProcessWorker::new(dir1.path().to_str().unwrap());
        let tc = ToolCall {
            id: "tc5".into(),
            name: "read".into(),
            arguments: json!({"paths": ["file.txt"]}),
        };
        let (tx, _) = tokio::sync::mpsc::channel(8);
        let res = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(!res.is_error);
        assert!(res.text().contains("from dir1"));

        worker.set_cwd(dir2.path().to_str().unwrap());
        let res2 = worker.execute(&tc, &tx, &CancelToken::new()).await.unwrap();
        assert!(res2.is_error);
        assert!(res2.text().contains("failed to read"));
    }
}
