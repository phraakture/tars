# Plugin System

Tool execution runs through a plugin transport. The default transport executes tools in-process inside the server. The subprocess transport isolates tool execution in worker processes that talk to the server over stdin/stdout, so a crashing tool cannot take the server down.

## Architecture

```
┌──────────┐     JSON lines     ┌──────────────┐
│  Server  │◄──────────────────►│ tars-worker  │
│(tars-lib)│    stdin/stdout    │ (subprocess) │
└──────────┘                    └──────┬───────┘
                                      │
                              ┌───────┴───────┐
                              │  Tool Router  │
                              │ (bash, read,  │
                              │  write, edit) │
                              └───────────────┘
```

The server side is implemented by `tars_lib::plugin_manager::PluginManager`. It spawns worker binaries, reads their tool registrations, and routes tool calls by name. `SubprocessExecutor` adapts the manager to the `ToolExecutor` trait the agent loop consumes.

## Wire Protocol

Communication is JSON lines: one JSON object per line, newline terminated. Both directions use tagged enums with a `"type"` discriminator.

### Server to Worker (`PluginRequest`)

| Type | Fields | Purpose |
|------|--------|---------|
| `tool_call` | `tool_call_id`, `name`, `arguments`, `cwd` (optional) | Execute a tool |
| `cancel_tool_call` | `tool_call_id` | Abort an in-flight call |
| `shutdown` | | Stop the worker |

```json
{"type": "tool_call", "tool_call_id": "tc-1", "name": "bash", "arguments": {"command": "echo hello"}, "cwd": "/tmp"}
{"type": "cancel_tool_call", "tool_call_id": "tc-1"}
{"type": "shutdown"}
```

### Worker to Server (`PluginMessage`)

| Type | Fields | Purpose |
|------|--------|---------|
| `register` | `name`, `tools` | First message at startup; advertises the tool set |
| `tool_result` | `tool_call_id`, `content`, `is_error`, `summary` (optional) | Final result for one call |
| `output_delta` | `tool_call_id`, `text` | Streaming progress for one call |

```json
{"type": "register", "name": "worker", "tools": [{"name": "bash", "description": "Execute bash commands", "parameters": {}}]}
{"type": "tool_result", "tool_call_id": "tc-1", "content": "hello\n", "is_error": false}
{"type": "output_delta", "tool_call_id": "tc-1", "text": "chunk"}
```

Each registered tool is a `PluginToolDef` with a `name`, a `description`, and a JSON Schema `parameters` object. The server converts these into provider tool definitions for the model context.

## Message Flow

1. The worker starts and immediately sends one `register` message.
2. The server spawns the binary and blocks until `register` arrives. Any other first message is a registration failure.
3. The server sends `tool_call` requests as the agent loop emits tool use.
4. The worker executes each call and answers with exactly one `tool_result` carrying the same `tool_call_id`.
5. A `cancel_tool_call` may arrive while a call is running; the worker sets a cancel token that the executing tool polls.
6. On `shutdown` (or when stdin closes), the worker drains pending sends and exits.

## Built-in Tools

Shipped by the `tars-worker` binary:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands in the session working directory |
| `read` | Read file contents; each line is prefixed with a stable per-line hash usable by `edit` |
| `write` | Create or fully overwrite files |
| `edit` | Precise edits by exact text match or by line anchor, batched across files |
| `get_file_skeleton` | Outline source files (signatures only, no bodies) via tree-sitter; Rust, Python, JS, TS, TSX |
| `get_function` | Extract complete function/method bodies by qualified name (`Foo.bar`, `::` accepted for Rust) |
| `diagnostics_scan` | Per-file lint/diagnostics: Rust via `cargo check`; other extensions via `.tars/diagnostics.toml` |

## Writing a Custom Plugin

A plugin is any binary that implements the protocol. In Rust, depend on `tars-plugin` and reuse the wire types:

```rust
use tars_plugin::{PluginMessage, PluginRegistration, PluginToolDef, PluginRequest, PluginToolResult};

fn main() {
    // 1. Register first, before reading anything.
    let registration = PluginMessage::Register(PluginRegistration {
        name: "my-plugin".into(),
        tools: vec![PluginToolDef {
            name: "my_tool".into(),
            description: "Does a thing".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }],
    });
    println!("{}", serde_json::to_string(&registration).unwrap());

    // 2. Serve requests from stdin.
    for line in std::io::stdin().lock().lines() {
        let Ok(req) = serde_json::from_str::<PluginRequest>(&line.unwrap()) else {
            continue;
        };
        match req {
            PluginRequest::ToolCall { tool_call_id, name, arguments, cwd } => {
                // Execute, then report exactly one result.
                let result = PluginToolResult {
                    tool_call_id,
                    content: format!("ran {} with {:?}", name, arguments),
                    is_error: false,
                    summary: Some("my_tool ok".into()),
                };
                println!("{}", serde_json::to_string(&PluginMessage::ToolResult(result)).unwrap());
            }
            PluginRequest::CancelToolCall { .. } => {}
            PluginRequest::Shutdown => break,
        }
    }
}
```

Honor `cwd` for path arguments, keep stdout reserved exclusively for protocol messages, and never block the main loop for long work without honoring cancellation.

## Spawning and Registration

Plugins are spawned programmatically today:

```rust
use tars_lib::plugin_manager::PluginManager;

let mut pm = PluginManager::new();
pm.spawn_plugin(&["/path/to/tars-worker".to_string()], "/tmp")?;

// Advertised schemas feed the model context.
let schemas = pm.tool_schemas();

// Route a call.
let result = pm.execute_tool_call(&tool_call, "/tmp")?;
```

Config-driven registration (a `plugins.toml` naming commands to spawn at server start) is planned but not implemented.

## Testing a Plugin

The round-trip pattern used throughout the workspace:

```rust
#[test]
fn my_plugin_round_trip() {
    let exe = build_or_locate_my_plugin();
    let mut pm = PluginManager::new();
    pm.spawn_plugin(&[exe], "/tmp").unwrap();

    assert!(pm.tool_schemas().iter().any(|t| t.name == "my_tool"));

    let tc = ToolCall {
        id: "test-1".into(),
        name: "my_tool".into(),
        arguments: serde_json::json!({"path": "x"}),
    };
    let result = pm.execute_tool_call(&tc, "/tmp").unwrap();
    assert!(!result.is_error);
}
```

Spawn the binary under test with `CARGO_BIN_EXE_<name>` when the plugin lives in the same crate, or build it first and point at `target/debug/`.

## Cancellation

1. The server sends `cancel_tool_call` with the tool call ID.
2. The worker removes the call from its in-flight table and signals the matching `CancelToken`.
3. The executing tool observes the token at its next check point and aborts.
4. The worker still returns a `tool_result` for the call, marked as an error.

Cancellation is cooperative. Tools that never check the token run to completion.

## Current Limitations

- `output_delta` is part of the protocol but the server does not yet forward deltas to clients; long tools appear as a single result.
- The server serializes tool calls per manager (one blocking lock), so a single worker handles calls one at a time from the server side.
- Worker stderr is not forwarded to server logs yet.
- If a worker dies mid-call, the server surfaces an IO error for that call; there is no automatic respawn yet.
- `get_file_skeleton` drops the `export` keyword on exported JS/TS declarations (the declaration itself is always included).

## Configuring Extra Diagnostics

`.tars/diagnostics.toml` in a project root adds tools for extensions that have no built-in scanner:

```toml
[[tool]]
extensions = ["py"]
command = "ruff check {file}"
```

Each configured command runs once per file; combined stdout/stderr becomes one `info`-severity diagnostic for that file.
