<p align="center">
  <img src="assets/banner.jpg?v=1" alt="tars" width="100%">
</p>

A Rust agent harness for LLM-powered coding assistance, with an interactive terminal UI.

## Features

- **Multiple providers**: Anthropic, OpenAI, and a mock provider for testing
- **Tool execution**: bash, read, write, and edit tools with in-process or subprocess worker transport
- **Session persistence**: SQLite-backed storage with automatic crash recovery
- **Client/server design**: Unix socket transport with a typed JSON-line protocol
- **Context compaction**: automatic summarization when the context window fills
- **Terminal UI**: session picker, live streaming chat, and scrolling transcript
- **Graceful shutdown**: in-flight turns are drained before process exit

## Quickstart

Build the workspace:

```bash
cargo build --release
```

Start the server in the foreground:

```bash
cargo run -- server
```

Send a one-shot message (starts the server automatically if it is not running):

```bash
cargo run -- chat -m "hello"
```

Start an interactive REPL:

```bash
cargo run -- chat
```

Or open the full terminal UI with a session picker and live streaming:

```bash
cargo run -- tui
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `tars-base` | Types, wire protocol, error handling, configuration |
| `tars-engine` | Provider abstraction, agent loop, compaction |
| `tars-plugin` | ToolExecutor trait, plugin wire protocol |
| `tars-plugin-worker` | In-process tool implementations |
| `tars-worker` | Subprocess worker binary |
| `tars-lib` | Server daemon, database, agent runner, plugin manager |
| `tars-client` | Client library for Unix socket communication |
| `tars-tui` | Ratatui terminal interface |
| `tars-agent` | CLI binary (`tars` command) |

## Configuration

| Path | Purpose |
|------|---------|
| `~/.config/tars/providers.toml` | Provider definitions and API keys |
| `~/.config/tars/models.toml` | Model aliases |
| `~/.local/share/tars/tars.db` | Session database |
| `~/.tars/tars.sock` | Unix socket |

See [docs/CONFIG.md](docs/CONFIG.md) for the full configuration reference, including XDG overrides and reload semantics.

## CLI Reference

| Command | Description |
|---------|-------------|
| `tars server` | Run the server in the foreground |
| `tars chat -m "text"` | One-shot message; auto-starts the server |
| `tars chat -M <model>` | Use a specific model |
| `tars chat` | Interactive REPL |
| `tars tui` | Terminal UI: session picker and live streaming chat |
| `tars sessions` | List sessions |
| `tars models` | List registered models |
| `tars config reload` | Re-read and summarize provider configuration |

## Plugin System

Tool execution runs through a plugin transport. The default transport executes tools in-process. The subprocess transport spawns worker processes, reads their tool registrations, and routes tool calls over the JSON-line wire protocol, isolating tool failures from the server. Custom plugins can be written against the same protocol.

See [docs/PLUGINS.md](docs/PLUGINS.md) for the protocol specification and a guide to writing custom plugins.

## Development

Run the full test suite:

```bash
cargo test
```

Run tests with logging enabled:

```bash
RUST_LOG=info cargo test
```

Check formatting and lints:

```bash
cargo fmt --check
cargo clippy
```

Build the worker binary (required by the subprocess integration tests):

```bash
cargo build -p tars-worker
```

Common development tasks are available through [just](https://github.com/casey/just):

```bash
just ci        # fmt check, clippy, tests
just test      # run all tests
just summary   # crate list and test count
```

## License

MIT
