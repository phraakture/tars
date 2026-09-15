# Configuration

tars configuration lives in `~/.config/tars/` (XDG standard).

## providers.toml

Defines LLM providers and their models.

```toml
[providers.anthropic]
api = "anthropic"
base_url = "https://api.anthropic.com/v1"
api_key = "$ANTHROPIC_API_KEY"  # or inline, or "none"

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

### API Key Resolution

API keys support three formats:
- **Inline**: `api_key = "sk-..."`
- **Environment**: `api_key = "$ENV_VAR"` — expanded from environment
- **None**: `api_key = "none"` or omitted — no authentication

### Provider Fields

| Field | Description |
|-------|-------------|
| `api` | API type: `"anthropic"`, `"openai"`, or custom |
| `base_url` | API endpoint URL |
| `api_key` | Authentication key (see above) |
| `models` | Array of model definitions |

### Model Fields

| Field | Default | Description |
|-------|---------|-------------|
| `id` | (required) | Model identifier |
| `name` | `id` | Display name |
| `context_window` | `128000` | Maximum context tokens |
| `max_tokens` | `16384` | Maximum output tokens |
| `thinking` | `"none"` | Thinking style: `"none"`, `"anthropic"`, `"qwen"` |
| `cost` | default | Cost per token (for usage tracking) |

## models.toml

Model aliases for convenient referencing.

```toml
[aliases]
smart = "claude-sonnet-4-6"
cheap = "openai/gpt-4.1-mini"
```

Aliases can be used in place of model IDs when creating sessions.

## Runtime Paths

| Path | Purpose |
|------|---------|
| `~/.config/tars/` | Configuration files |
| `~/.local/share/tars/tars.db` | Session database |
| `~/.runtime/tars/tars.sock` | Unix socket |
| `~/.runtime/tars/tars.pid` | PID file |
| `~/.local/share/tars/logs/` | Log files |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level filter (e.g., `info`, `tars_lib=debug`) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
