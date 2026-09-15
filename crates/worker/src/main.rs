//! Subprocess worker binary for tars.
//!
//! Reads PluginRequests from stdin (JSON lines), executes tool calls, and writes
//! PluginMessages to stdout. Implements the plugin wire protocol for subprocess
//! plugin transport.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

use tars_base::{CancelToken, ToolCall};
use tars_plugin::{
    PluginMessage, PluginRegistration, PluginRequest, PluginToolDef, PluginToolResult,
};
use tars_plugin_worker::{default_tools, execute_tool, tool_schemas};

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create tokio runtime");
    rt.block_on(async_main());
}

async fn async_main() {
    // Share tools via Arc (ToolDef is !Clone because of Box<dyn Fn>)
    let tools: Arc<Vec<tars_plugin_worker::ToolDef>> = Arc::new(default_tools());
    let schemas = tool_schemas(&tools);
    let tool_defs: Vec<PluginToolDef> = schemas
        .iter()
        .map(|t| PluginToolDef {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect();

    let register = PluginMessage::Register(PluginRegistration {
        name: "worker".into(),
        tools: tool_defs,
    });
    write_message(&register);

    // Track in-flight tool calls for cancellation
    let in_flight: Arc<Mutex<HashMap<String, CancelToken>>> = Arc::new(Mutex::new(HashMap::new()));

    // Set up outbound channel and writer task
    let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<PluginMessage>();
    let writer_handle = tokio::spawn(async move {
        let stdout = tokio::io::stdout();
        let mut writer = tokio::io::BufWriter::new(stdout);
        while let Some(msg) = msg_rx.recv().await {
            let mut line = serde_json::to_string(&msg).expect("serialize PluginMessage");
            line.push('\n');
            writer.write_all(line.as_bytes()).await.ok();
            writer.flush().await.ok();
        }
    });

    // Read PluginRequests from stdin
    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let req: PluginRequest = match serde_json::from_str(line.trim()) {
            Ok(r) => r,
            Err(_) => continue,
        };

        match req {
            PluginRequest::ToolCall {
                tool_call_id,
                name,
                arguments,
                cwd,
            } => {
                let cancel = CancelToken::new();
                in_flight
                    .lock()
                    .await
                    .insert(tool_call_id.clone(), cancel.clone());

                let msg_tx = msg_tx.clone();
                let tools = tools.clone();
                let in_flight = in_flight.clone();

                tokio::task::spawn_blocking(move || {
                    let cwd_str = cwd.unwrap_or_else(|| "/tmp".into());
                    let tool_call = ToolCall {
                        id: tool_call_id.clone(),
                        name,
                        arguments,
                    };
                    let result = execute_tool(&tools, &tool_call, &cwd_str, &cancel);

                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        in_flight.lock().await.remove(&tool_call_id);
                    });

                    let content = result
                        .content
                        .iter()
                        .map(|c| c.text().to_string())
                        .collect::<Vec<_>>()
                        .join("");

                    let _ = msg_tx.send(PluginMessage::ToolResult(PluginToolResult {
                        tool_call_id: result.tool_call_id,
                        content,
                        is_error: result.is_error,
                        summary: result.summary,
                    }));
                });
            }
            PluginRequest::CancelToolCall { tool_call_id } => {
                if let Some(cancel) = in_flight.lock().await.remove(&tool_call_id) {
                    cancel.cancel();
                }
            }
            PluginRequest::Shutdown => break,
        }
    }

    drop(msg_tx);
    let _ = writer_handle.await;
}

fn write_message(msg: &PluginMessage) {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let mut line = serde_json::to_string(msg).expect("serialize PluginMessage");
    line.push('\n');
    handle.write_all(line.as_bytes()).ok();
    handle.flush().ok();
}
