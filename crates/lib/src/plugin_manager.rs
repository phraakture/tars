//! Subprocess plugin manager — spawns plugin binaries, reads their
//! registration, routes tool calls over JSON-lines stdin/stdout, and
//! implements `ToolExecutor` so the agent loop can dispatch through it.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use tars_base::{CancelToken, Tool, ToolCall, ToolResultMessage};
use tars_plugin::protocol::{PluginMessage, PluginRegistration, PluginRequest, PluginToolResult};
use tokio::sync::Mutex;

/// A spawned plugin subprocess and its registration metadata.
pub struct PluginHandle {
    pub name: String,
    pub registration: PluginRegistration,
    child: Option<Child>,
    stdin: Option<BufWriter<std::process::ChildStdin>>,
    stdout: Option<BufReader<std::process::ChildStdout>>,
}

impl PluginHandle {
    /// Spawn a plugin process and read its initial `Register` message.
    pub fn spawn(command: &[String], cwd: &str) -> anyhow::Result<Self> {
        let mut cmd = Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn plugin {:?}: {}", command, e))?;

        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("plugin stdin not available"))?,
        );
        let mut stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("plugin stdout not available"))?,
        );

        // Read the first line: must be PluginMessage::Register
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        let msg: PluginMessage = serde_json::from_str(line.trim())
            .map_err(|e| anyhow::anyhow!("failed to parse plugin registration: {}", e))?;

        let registration = match msg {
            PluginMessage::Register(reg) => reg,
            other => return Err(anyhow::anyhow!("expected Register, got {:?}", other)),
        };

        let name = registration.name.clone();

        Ok(Self {
            name,
            registration,
            child: Some(child),
            stdin: Some(stdin),
            stdout: Some(stdout),
        })
    }

    /// Send a `PluginRequest` to the plugin's stdin.
    pub fn send(&mut self, req: &PluginRequest) -> anyhow::Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("stdin taken"))?;
        let mut line = serde_json::to_string(req)?;
        line.push('\n');
        stdin.write_all(line.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }

    /// Read the next `PluginMessage` from the plugin's stdout.
    pub fn read_message(&mut self) -> anyhow::Result<PluginMessage> {
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("stdout taken"))?;
        let mut line = String::new();
        let n = stdout.read_line(&mut line)?;
        if n == 0 {
            return Err(anyhow::anyhow!("plugin stdout closed"));
        }
        serde_json::from_str(line.trim()).map_err(|e| anyhow::anyhow!("parse error: {}", e))
    }

    /// Send a ToolCall and wait for the corresponding ToolResult.
    pub fn execute_tool_call(
        &mut self,
        tool_call: &ToolCall,
        cwd: &str,
    ) -> anyhow::Result<PluginToolResult> {
        let req = PluginRequest::ToolCall {
            tool_call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
            cwd: Some(cwd.to_string()),
        };
        self.send(&req)?;

        // Read messages until we get a ToolResult for this tool_call_id
        loop {
            match self.read_message()? {
                PluginMessage::ToolResult(result) if result.tool_call_id == tool_call.id => {
                    return Ok(result);
                }
                PluginMessage::ToolResult(_) => continue, // different tool call
                PluginMessage::OutputDelta { .. } => continue,
                PluginMessage::Register(_) => {
                    return Err(anyhow::anyhow!("unexpected Register during tool call"));
                }
            }
        }
    }

    /// Kill the plugin subprocess.
    pub fn kill(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for PluginHandle {
    fn drop(&mut self) {
        // Send shutdown, best-effort
        if self.stdin.is_some() {
            let _ = self.send(&PluginRequest::Shutdown);
        }
        self.kill();
    }
}

/// Manages multiple plugin subprocesses and routes tool calls to them.
pub struct PluginManager {
    plugins: Vec<PluginHandle>,
    /// Tool name → plugin index
    tool_map: HashMap<String, usize>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            tool_map: HashMap::new(),
        }
    }

    /// Spawn a plugin and register its tools.
    pub fn spawn_plugin(&mut self, command: &[String], cwd: &str) -> anyhow::Result<()> {
        let handle = PluginHandle::spawn(command, cwd)?;
        let idx = self.plugins.len();
        for tool in &handle.registration.tools {
            self.tool_map.insert(tool.name.clone(), idx);
        }
        tracing::info!(
            plugin = %handle.name,
            tools = handle.registration.tools.len(),
            "plugin ready"
        );
        self.plugins.push(handle);
        Ok(())
    }

    /// Get all tool schemas from all plugins.
    pub fn tool_schemas(&self) -> Vec<Tool> {
        self.plugins
            .iter()
            .flat_map(|p| p.registration.tools.iter().map(Tool::from))
            .collect()
    }

    /// Execute a tool call by routing to the appropriate plugin.
    pub fn execute_tool_call(
        &mut self,
        tool_call: &ToolCall,
        cwd: &str,
    ) -> anyhow::Result<PluginToolResult> {
        let &idx = self
            .tool_map
            .get(&tool_call.name)
            .ok_or_else(|| anyhow::anyhow!("no plugin provides tool '{}'", tool_call.name))?;
        self.plugins[idx].execute_tool_call(tool_call, cwd)
    }

    /// Kill all plugins.
    pub fn shutdown(&mut self) {
        for p in &mut self.plugins {
            p.kill();
        }
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Wrapper that implements `ToolExecutor` using the subprocess `PluginManager`.
pub struct SubprocessExecutor {
    pub plugin_manager: Arc<Mutex<PluginManager>>,
    pub cwd: String,
}

#[async_trait::async_trait]
impl tars_plugin::ToolExecutor for SubprocessExecutor {
    async fn execute(
        &mut self,
        tool_call: &ToolCall,
        _output_tx: &tokio::sync::mpsc::Sender<String>,
        _cancel: &CancelToken,
    ) -> tars_base::Result<ToolResultMessage> {
        let cwd = self.cwd.clone();
        let pm = self.plugin_manager.clone();
        let tc = tool_call.clone();

        tokio::task::spawn_blocking(move || {
            let mut pm = pm.blocking_lock();
            match pm.execute_tool_call(&tc, &cwd) {
                Ok(result) => {
                    let content: Vec<tars_base::ToolResultContent> =
                        vec![tars_base::ToolResultContent::Text(tars_base::TextContent {
                            text: result.content,
                            text_signature: None,
                        })];
                    Ok(ToolResultMessage {
                        tool_call_id: tc.id,
                        tool_name: tc.name,
                        content,
                        details: None,
                        is_error: result.is_error,
                        timestamp: tars_base::timestamp_ms(),
                        duration_ms: None,
                        summary: result.summary,
                        post_persist_actions: Vec::new(),
                    })
                }
                Err(e) => Err(tars_base::Error::Internal(e.to_string())),
            }
        })
        .await
        .map_err(|e| tars_base::Error::Internal(format!("spawn_blocking failed: {}", e)))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_binary() -> String {
        // CARGO_BIN_EXE_tars-worker is set when running tars-worker's own tests,
        // but not when running tars-lib tests. Fall back to target/debug path.
        std::env::var("CARGO_BIN_EXE_tars-worker").unwrap_or_else(|_| {
            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop(); // crates/lib → crates
            path.pop(); // crates → workspace root
            path.push("target/debug/tars-worker");
            path.to_string_lossy().to_string()
        })
    }

    #[test]
    fn spawn_worker_and_round_trip() {
        let exe = worker_binary();
        assert!(
            std::path::Path::new(&exe).exists(),
            "binary not found at {exe} — run `cargo build -p tars-worker` first"
        );
        let mut pm = PluginManager::new();
        pm.spawn_plugin(&[exe], "/tmp").unwrap();

        // Should have registered tools
        let schemas = pm.tool_schemas();
        assert!(schemas.iter().any(|t| t.name == "bash"));
        assert!(schemas.iter().any(|t| t.name == "read"));

        // Execute a tool call
        let tc = ToolCall {
            id: "test-1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "echo hello"}),
        };
        let result = pm.execute_tool_call(&tc, "/tmp").unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn unknown_tool_returns_error() {
        let mut pm = PluginManager::new();
        let tc = ToolCall {
            id: "test-2".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        };
        let result = pm.execute_tool_call(&tc, "/tmp");
        assert!(result.is_err());
    }
}
