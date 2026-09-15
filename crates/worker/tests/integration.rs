use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use tars_plugin::{PluginMessage, PluginRequest};

#[test]
fn worker_round_trip() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tars-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn tars-worker");

    let mut stdin = child.stdin.take().expect("no stdin");
    let stdout = child.stdout.take().expect("no stdout");
    let mut reader = BufReader::new(stdout);

    // Read the Register message
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("failed to read register");
    let register: PluginMessage =
        serde_json::from_str(line.trim()).expect("failed to parse register");
    match &register {
        PluginMessage::Register(reg) => {
            assert_eq!(reg.name, "worker");
            assert!(!reg.tools.is_empty());
            assert!(reg.tools.iter().any(|t| t.name == "bash"));
            assert!(reg.tools.iter().any(|t| t.name == "read"));
            assert!(reg.tools.iter().any(|t| t.name == "write"));
            assert!(reg.tools.iter().any(|t| t.name == "edit"));
        }
        _ => panic!("expected Register, got: {:?}", register),
    }

    // Send a ToolCall for bash: echo hello
    let call = PluginRequest::ToolCall {
        tool_call_id: "test-1".into(),
        name: "bash".into(),
        arguments: serde_json::json!({"command": "echo hello"}),
        cwd: Some("/tmp".into()),
    };
    let mut req_line = serde_json::to_string(&call).unwrap();
    req_line.push('\n');
    stdin.write_all(req_line.as_bytes()).unwrap();
    stdin.flush().unwrap();

    // Read the ToolResult
    line.clear();
    reader.read_line(&mut line).expect("failed to read result");
    let result: PluginMessage = serde_json::from_str(line.trim()).expect("failed to parse result");
    match &result {
        PluginMessage::ToolResult(tr) => {
            assert_eq!(tr.tool_call_id, "test-1");
            assert!(!tr.is_error);
            assert!(tr.content.contains("hello"));
        }
        _ => panic!("expected ToolResult, got: {:?}", result),
    }

    // Send Shutdown
    let shutdown = PluginRequest::Shutdown;
    let mut shut_line = serde_json::to_string(&shutdown).unwrap();
    shut_line.push('\n');
    stdin.write_all(shut_line.as_bytes()).unwrap();
    stdin.flush().unwrap();

    child.wait().expect("worker should exit cleanly");
}

#[test]
fn worker_cancel_tool() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tars-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn tars-worker");

    let mut stdin = child.stdin.take().expect("no stdin");
    let stdout = child.stdout.take().expect("no stdout");
    let mut reader = BufReader::new(stdout);

    // Read Register
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("failed to read register");

    // Send a long-running bash command
    let call = PluginRequest::ToolCall {
        tool_call_id: "cancel-test".into(),
        name: "bash".into(),
        arguments: serde_json::json!({"command": "sleep 30"}),
        cwd: Some("/tmp".into()),
    };
    let mut req_line = serde_json::to_string(&call).unwrap();
    req_line.push('\n');
    stdin.write_all(req_line.as_bytes()).unwrap();
    stdin.flush().unwrap();

    // Cancel it
    let cancel = PluginRequest::CancelToolCall {
        tool_call_id: "cancel-test".into(),
    };
    let mut cancel_line = serde_json::to_string(&cancel).unwrap();
    cancel_line.push('\n');
    stdin.write_all(cancel_line.as_bytes()).unwrap();
    stdin.flush().unwrap();

    // Should get a result back (error or cancelled)
    line.clear();
    reader
        .read_line(&mut line)
        .expect("failed to read cancelled result");
    let result: PluginMessage = serde_json::from_str(line.trim()).expect("failed to parse result");
    match &result {
        PluginMessage::ToolResult(tr) => {
            assert_eq!(tr.tool_call_id, "cancel-test");
        }
        _ => panic!("expected ToolResult, got: {:?}", result),
    }

    // Shutdown
    let shutdown = PluginRequest::Shutdown;
    let mut shut_line = serde_json::to_string(&shutdown).unwrap();
    shut_line.push('\n');
    stdin.write_all(shut_line.as_bytes()).unwrap();
    stdin.flush().unwrap();

    child.wait().expect("worker should exit cleanly");
}
