use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::{ToolDef, ToolOutput};
use tars_base::{CancelToken, Tool};

pub fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "bash".into(),
            description: "Execute a bash command and return its output. Use for running shell commands, scripts, build tools, git operations, etc.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)"
                    }
                },
                "required": ["command"]
            }),
        },
        execute: Box::new(execute),
        prepare_arguments: None,
    }
}

pub fn check_cwd(cwd: &str) -> Option<String> {
    let path = std::path::Path::new(cwd);
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => None,
        Ok(_) => Some(format!(
            "session cwd is not a directory: {} (use /cd <path> to switch to a valid directory)",
            cwd,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(format!(
            "session cwd no longer exists: {} (was the worktree removed? use /cd <path> to switch to a valid directory)",
            cwd,
        )),
        Err(e) => Some(format!("cannot access session cwd {}: {}", cwd, e,)),
    }
}

fn execute(args: Value, cwd: &str, cancel: &CancelToken) -> ToolOutput {
    let Some(command) = args.get("command").and_then(|c| c.as_str()) else {
        return ToolOutput::error("missing 'command' argument");
    };
    let timeout_secs = args.get("timeout").and_then(|t| t.as_u64()).unwrap_or(120);

    if let Some(err) = check_cwd(cwd) {
        return ToolOutput::error(err);
    }

    if cancel.is_cancelled() {
        return ToolOutput::error("error: cancelled before execution");
    }

    let mut child = match Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return ToolOutput::error(format!("failed to execute command: {}", e)),
    };

    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    // Poll for exit, cancellation, or timeout
    let mut timed_out = false;
    let mut cancelled = false;

    loop {
        if cancel.is_cancelled() {
            cancelled = true;
            let _ = child.kill();
            break;
        }
        if start.elapsed() >= timeout {
            timed_out = true;
            let _ = child.kill();
            break;
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => break,
        }
    }

    // Ensure child is reaped; if we killed it, wait will collect it
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return ToolOutput::error(format!("failed to wait for command: {}", e)),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exit_code = output.status.code().unwrap_or(-1);

    format_output(stdout, stderr, exit_code, timed_out, cancelled, command)
}

fn format_output(
    stdout: String,
    stderr: String,
    exit_code: i32,
    timed_out: bool,
    cancelled: bool,
    command: &str,
) -> ToolOutput {
    let mut text = stdout;
    if !stderr.is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("STDERR:\n");
        text.push_str(&stderr);
    }

    if text.len() > 100_000 {
        let head = &text[..50_000];
        let tail = &text[text.len() - 50_000..];
        text = format!(
            "{}\n\n... [truncated {} bytes] ...\n\n{}",
            head,
            text.len() - 100_000,
            tail
        );
    }

    if cancelled {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("(cancelled)");
        let mut out = ToolOutput::error(text.trim_end().to_string());
        maybe_add_summary(&mut out, command, exit_code);
        return out;
    }

    if timed_out {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("(timed out)");
        let mut out = ToolOutput::error(text.trim_end().to_string());
        maybe_add_summary(&mut out, command, exit_code);
        return out;
    }

    let success = exit_code == 0;
    if text.is_empty() {
        text = format!("(exit code: {})", exit_code);
    } else if !success {
        text.push_str(&format!("\n(exit code: {})", exit_code));
    }

    let text = text.trim_end().to_string();
    let mut out = if success {
        ToolOutput::text(text)
    } else {
        ToolOutput::error(text)
    };
    maybe_add_summary(&mut out, command, exit_code);
    out
}

fn maybe_add_summary(output: &mut ToolOutput, command: &str, exit_code: i32) {
    let text_content = output.content.first().map(|c| c.text()).unwrap_or("");
    let line_count = text_content.lines().count();
    if line_count > 20 {
        let cmd_preview = if command.chars().count() > 60 {
            let truncated: String = command.chars().take(57).collect();
            format!("{}...", truncated)
        } else {
            command.to_string()
        };
        let exit_suffix = if exit_code != 0 {
            format!(", exit {}", exit_code)
        } else {
            String::new()
        };
        output.summary = Some(format!(
            "bash: $ {} → {} lines{}",
            cmd_preview, line_count, exit_suffix
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;
    use tars_base::CancelToken;

    #[test]
    fn bash_simple_command() {
        let out = execute(
            json!({"command": "echo hello"}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(out.content[0].text().contains("hello"));
    }

    #[test]
    fn bash_pipe_command() {
        let out = execute(
            json!({"command": "echo hello | tr a-z A-Z"}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(out.content[0].text().contains("HELLO"));
    }

    #[test]
    fn bash_timeout_kills_process() {
        let start = Instant::now();
        let out = execute(
            json!({"command": "sleep 60", "timeout": 1}),
            "/tmp",
            &CancelToken::new(),
        );
        let elapsed = start.elapsed();
        assert!(out.is_error);
        assert!(out.content[0].text().contains("timed out"));
        assert!(elapsed.as_secs() < 5);
    }

    #[test]
    fn bash_cancel_flag_kills_quickly() {
        let cancel = CancelToken::new();
        let cancel_clone = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            cancel_clone.cancel();
        });
        let start = Instant::now();
        let out = execute(json!({"command": "sleep 10"}), "/tmp", &cancel);
        let elapsed = start.elapsed();
        assert!(out.is_error);
        assert!(out.content[0].text().contains("cancelled"));
        assert!(elapsed < Duration::from_secs(2));
    }

    #[test]
    fn bash_prefire_cancel_returns_immediately() {
        let cancel = CancelToken::new();
        cancel.cancel();
        let start = Instant::now();
        let out = execute(json!({"command": "sleep 10"}), "/tmp", &cancel);
        let elapsed = start.elapsed();
        assert!(out.is_error);
        assert!(out.content[0].text().contains("cancelled"));
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn bash_missing_cwd_error() {
        let bogus = format!("/tmp/tars-bash-bogus-{}", std::process::id());
        let _ = std::fs::remove_dir_all(&bogus);
        let out = execute(json!({"command": "echo hi"}), &bogus, &CancelToken::new());
        assert!(out.is_error);
        assert!(out.content[0].text().contains(&bogus));
        assert!(out.content[0].text().contains("no longer exists"));
    }

    #[test]
    fn bash_formatting_exit_code() {
        let out = execute(json!({"command": "exit 2"}), "/tmp", &CancelToken::new());
        assert!(out.is_error);
        assert!(out.content[0].text().contains("(exit code: 2)"));
    }

    #[test]
    fn bash_stdout_stderr_combined() {
        let out = execute(
            json!({"command": "echo out; echo err >&2"}),
            "/tmp",
            &CancelToken::new(),
        );
        let text = out.content[0].text();
        assert!(text.contains("out"));
        assert!(text.contains("STDERR:"));
        assert!(text.contains("err"));
    }
}
