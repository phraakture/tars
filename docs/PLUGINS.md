# Plugin System

tars uses a subprocess plugin transport for tool execution. Tools run in isolated worker processes that communicate with the server via JSON-lines over stdin/stdout.

## Architecture

```
┌──────────┐    JSON-lines     ┌──────────────┐
│  Server  │◄─────────────────►│  tars-worker │
│ (tars-lib)│   stdin/stdout   │  (subprocess)│
└──────────┘                   └──────┬───────┘
                                      │
                              ┌───────┴───────┐
                              │  Tool Router   │
                              │  (bash, read,  │
                              │   write, edit) │
                              └───────────────┘
```

## Wire Protocol

Communication uses JSON-lines (one JSON object per line) over stdin/stdout.

### Server → Worker (PluginRequest)

```json
{"type": "tool_call", "tool_call_id": "tc-1", "name": "bash", "arguments": {"command": "echo hello"}, "cwd": "/tmp"}
{"type": "cancel_tool_call", "tool_call_id": "tc-1"}
{"type": "shutdown"}
```

### Worker → Server (PluginMessage)

```json
{"type": "register", "name": "worker", "tools": [{"name": "bash", "description": "...", "parameters": {...}}]}
{"type": "tool_result", "tool_call_id": "tc-1", "content": "hello\n", "is_error": false}
{"type": "output_delta", "tool_call_id": "tc-1", "text": "chunk"}
```

### Flow

1. Worker starts, sends `register` with tool definitions
2. Server sends `tool_call` requests
3. Worker executes and sends `tool_result` back
4. Server can send `cancel_tool_call` to abort
5. Server sends `shutdown` to exit

## Built-in Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `read` | Read file contents (with line hashes) |
| `write` | Create or overwrite files |
| `edit` | Precise file edits with text/anchor replacement |

## Adding Custom Plugins

### 1. Implement the Protocol

Your plugin binary must:
- Read `PluginRequest` from stdin (JSON lines)
- Write `PluginMessage` to stdout (JSON lines)
- Send `register` as the first message
- Handle `tool_call` and `cancel_tool_call`

### 2. Register in Config

Add your plugin to the server configuration (future — currently auto-discovered).

### 3. Build and Deploy

```bash
# Build your plugin
cargo build --release -p my-plugin

# The server will spawn it as a subprocess
```

## Testing Plugins

Use the integration test pattern:

```rust
#[test]
fn my_plugin_round_trip() {
    let mut pm = PluginManager::new();
    pm.spawn_plugin(&["/path/to/my-plugin"], "/tmp").unwrap();

    let schemas = pm.tool_schemas();
    assert!(schemas.iter().any(|t| t.name == "my_tool"));

    let tc = ToolCall {
        id: "test-1".into(),
        name: "my_tool".into(),
        arguments: serde_json::json!({"arg": "value"}),
    };
    let result = pm.execute_tool_call(&tc, "/tmp").unwrap();
    assert!(!result.is_error);
}
```

## Cancellation

Tool calls can be cancelled mid-execution:
1. Server sends `cancel_tool_call` with the tool call ID
2. Worker sets a cancel token
3. Tool checks the token and aborts if set
4. Worker sends `tool_result` with error message

## Error Handling

- Unknown tool name → error response
- Tool execution failure → `is_error: true` in result
- Worker crash → server receives IO error
- Parse error → skipped (worker continues)
