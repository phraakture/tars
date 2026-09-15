# Configuration

tars resolves all paths through XDG conventions. Every location respects an environment override, so tests and sandboxed runs can redirect the whole tree.

## Directory Layout

| Purpose | Default | Override |
|---------|---------|----------|
| User-editable config | `~/.config/tars/` | `XDG_CONFIG_HOME` |
| Session data | `~/.local/share/tars/` | `XDG_DATA_HOME` |
| Runtime files (socket, PID) | `~/.tars/` | `XDG_RUNTIME_DIR` |
| Machine state | `~/.local/state/tars/` | `XDG_STATE_HOME` |

When `HOME` is unset (minimal containers), paths fall back to namespaced `/tmp` directories so nothing collides with unrelated files.

## providers.toml

Location: `$XDG_CONFIG_HOME/tars/providers.toml` (or `~/.config/tars/providers.toml`). Defines LLM providers and the models each exposes.

```toml
[providers.anthropic]
api = "anthropic"
base_url = "https://api.anthropic.com/v1"
api_key = "$ANTHROPIC_API_KEY"

[[providers.anthropic.models]]
id = "claude-sonnet-4-6"
name = "Claude Sonnet 4.6"
context_window = 200000
max_tokens = 32000
thinking = "anthropic"

[providers.openai]
api = "openai"
base_url = "https://api.openai.com/v1"
api_key = "$OPENAI_API_KEY"

[[providers.openai.models]]
id = "gpt-4.1"
context_window = 1048576
max_tokens = 32768
```

A missing file is not an error: tars starts with an empty provider set and the built-in providers remain available.

### Provider Fields

| Field | Required | Description |
|-------|----------|-------------|
| `api` | yes | API family: `"anthropic"` or `"openai"` (mapped internally to `anthropic-messages` and `openai-completions`) |
| `base_url` | yes | API endpoint URL |
| `api_key` | no | Authentication key, see resolution rules below |
| `models` | no | Array of model definitions |

### API Key Resolution

| Value | Behavior |
|-------|----------|
| `"sk-..."` | Used literally |
| `"$ENV_VAR"` | Read from the named environment variable; unset or empty resolves to no key |
| `"none"` or `""` | No authentication |
| omitted | No authentication |

### Model Fields

| Field | Default | Description |
|-------|---------|-------------|
| `id` | required | Model identifier sent to the API and used in session records |
| `name` | the `id` | Display name |
| `context_window` | `128000` | Maximum input context in tokens; drives compaction |
| `max_tokens` | `16384` | Maximum output tokens per response |
| `thinking` | `"none"` | Extended thinking style, see below |
| `cost` | `0` for all | Usage pricing, see below |

### Thinking Styles

| Value | Behavior |
|-------|----------|
| `none` | No extended thinking |
| `anthropic` | Anthropic budget or adaptive thinking |
| `open_ai` (alias `openai`) | OpenAI `reasoning_effort` parameter |
| `qwen` | Qwen-style `enable_thinking` flag |

### Cost Fields

All values are USD per million tokens.

| Field | Description |
|-------|-------------|
| `input` | Input token price |
| `output` | Output token price |
| `cache_read` | Cache read price |
| `cache_write` | Cache write price |

## models.toml

Location: `~/.config/tars/models.toml`. Defines short aliases for model IDs.

```toml
[aliases]
smart = "claude-sonnet-4-6"
cheap = "openai/gpt-4.1-mini"
```

A per-project override file is supported at `config_dir()/projects/{name}/models.toml`. Operator entries replace global entries with the same name; both maps are merged with operator priority.

## Runtime Files

| Path | Purpose |
|------|---------|
| `~/.tars/tars.sock` | Unix socket for the server |
| `~/.tars/tars.pid` | PID file, removed on clean shutdown |
| `~/.local/share/tars/tars.db` | SQLite session database |
| `~/.local/state/tars/logs/` | Log file directory (rotation helper exists; daemon wiring is not complete yet) |

On server start, tars refuses to run if the PID file points to a live process and removes a stale PID file. A stale socket is removed before binding.

## CLI Reference

| Command | Description |
|---------|-------------|
| `tars server` | Run the server in the foreground; Ctrl+C stops it |
| `tars chat -m "text"` | One-shot message; auto-starts the server if none is running |
| `tars chat -M <model>` | Use a specific model (usable with `-m` or standalone) |
| `tars chat` | Interactive REPL |
| `tars sessions` | List sessions |
| `tars models` | List registered models |
| `tars config reload` | Re-read `providers.toml` and print the loaded providers and models |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Tracing filter (for example `info` or `tars_lib=debug`) |
| `ANTHROPIC_API_KEY` | Referenced from `providers.toml` as `$ANTHROPIC_API_KEY` |
| `OPENAI_API_KEY` | Referenced from `providers.toml` as `$OPENAI_API_KEY` |
| `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_RUNTIME_DIR`, `XDG_STATE_HOME` | Directory overrides |

## Reload Semantics

Sessions pin the `Model` they were created with. Reloading configuration rebuilds the provider and model tables for new sessions without touching live ones. The registry swap currently requires a server restart to take full effect; `tars config reload` validates and summarizes what would be loaded.
