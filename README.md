# tars

A Rust agent harness — LLM-powered coding assistant with tool execution, session persistence, and subprocess plugin transport.

## What

tars is an agent harness that:
- Connects to LLM providers (Anthropic, OpenAI) for conversational coding assistance
- Executes tools (bash, read, write, edit) via subprocess workers
- Persists sessions to SQLite with crash recovery
- Communicates over Unix sockets with a JSON-line protocol
- Supports compaction (context window management) and graceful shutdown

## Quickstart

```bash
# Build
cargo build --release

# Start the server (foreground)
cargo run -- server start --foreground

# Chat (auto-starts server if not running)
cargo run -- chat -m "hello"

# Interactive REPL
cargo run -- chat

# Check config
cargo run -- config reload
```

## Architecture

```
┌─────────┐     Unix Socket      ┌──────────┐
│  CLI    │◄────────────────────►│  Server  │
│ (agent) │   JSON-line protocol │  (lib)   │
└─────────┘                      └────┬─────┘
                                      │
                              ┌───────┴───────┐
                              │  Agent Runner  │
                              │  (engine loop) │
                              └───────┬───────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                    ┌────▼────┐  ┌────▼────┐  ┌───▼────┐
                    │  Mock   │  │Anthropic│  │ OpenAI │
                    │ Provider│  │ Provider│  │Provider│
                    └─────────┘  └─────────┘  └────────┘
                         │
                    ┌────▼────────────────┐
                    │   Tool Executor     │
                    │  (subprocess plugin)│
                    └────┬───────────┬────┘
                         │           │
                    ┌────▼────┐ ┌────▼────┐
                    │  bash   │ │  read   │ ...
                    │  tool   │ │  tool   │
                    └─────────┘ └─────────┘
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `tars-base` | Types, protocol, error handling, config |
| `tars-engine` | Provider abstraction, agent loop, compaction |
| `tars-plugin` | ToolExecutor trait, plugin wire protocol |
| `tars-plugin-worker` | In-process tool implementations |
| `tars-worker` | Subprocess worker binary |
| `tars-lib` | Server daemon, DB, agent runner, plugin manager |
| `tars-client` | Client library for Unix socket communication |
| `tars-agent` | CLI binary (`tars` command) |

## Configuration

- **`~/.config/tars/providers.toml`** — provider definitions and API keys
- **`~/.config/tars/models.toml`** — model aliases
- **`~/.local/share/tars/tars.db`** — session database
- **`~/.runtime/tars/tars.sock`** — Unix socket

See [docs/CONFIG.md](docs/CONFIG.md) for details.

## Plugin System

tars uses a subprocess plugin transport for tool execution. The `tars-worker` binary communicates with the server via JSON-lines over stdin/stdout.

See [docs/PLUGINS.md](docs/PLUGINS.md) for details.

## Development

```bash
# Run all tests
cargo test

# Run with logging
RUST_LOG=info cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy

# Build worker binary (needed for integration tests)
cargo build -p tars-worker
```

## License

MIT
