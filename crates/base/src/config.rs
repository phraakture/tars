//! Provider and model configuration from TOML files.
//!
//! Two files, both rooted in [`crate::Paths::config_dir`]:
//!
//! - **`providers.toml`** — provider definitions and their models
//!   (`Config`). API keys may be inline, `"none"`, or `"$ENV_VAR"` for
//!   environment expansion (see [`resolve_provider_api_key`]).
//! - **`models.toml`** — the global model-alias map (`[aliases]`, one hop).
//!   A per-project operator overrides file lives at
//!   `config_dir()/projects/{name}/models.toml`.
//!
//! Loading is a dependency-injected concern: every loader takes a `&Paths`
//! so callers decide where config lives (tests use
//! [`crate::Paths::from_home`], sandboxed servers point elsewhere), and
//! nothing reads process-global env vars except the `$VAR` key expansion
//! done by [`resolve_provider_api_key`] itself.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::Paths;
use crate::types::{Model, ModelCost, ThinkingStyle};

// ---------------------------------------------------------------------------
// Config file types
// ---------------------------------------------------------------------------

/// Root structure of `providers.toml`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API type: `"anthropic"` or `"openai"` (or a custom API name).
    pub api: String,
    pub base_url: String,
    /// Inline API key, `"none"`/empty (no key), or `"$ENV_VAR"` naming the
    /// environment variable holding the key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default = "default_context_window")]
    pub context_window: u64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
    #[serde(default)]
    pub thinking: ThinkingStyle,
    #[serde(default)]
    pub cost: ModelCost,
}

fn default_context_window() -> u64 {
    128_000
}

fn default_max_tokens() -> u64 {
    16_384
}

// ---------------------------------------------------------------------------
// Loading & saving
// ---------------------------------------------------------------------------

/// Load `providers.toml`, returning an empty `Config` when the file is absent.
pub fn load_config(paths: &Paths) -> crate::Result<Config> {
    let path = paths.providers_path();
    if !path.exists() {
        return Ok(Config::default());
    }
    let content = std::fs::read_to_string(&path)?;
    toml::from_str(&content)
        .map_err(|e| crate::Error::Parse(format!("providers.toml ({}): {e}", path.display())))
}

/// Write `providers.toml`, creating parent directories as needed.
pub fn save_config(paths: &Paths, config: &Config) -> crate::Result<()> {
    let path = paths.providers_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(config)
        .map_err(|e| crate::Error::Parse(format!("serialize providers.toml: {e}")))?;
    std::fs::write(&path, content)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// API key resolution
// ---------------------------------------------------------------------------

/// Resolve a provider's API key.
///
/// Returns `None` when the key is absent, empty, or literally `"none"`.
/// A key of the form `"$ENV_VAR"` is expanded from the environment (unset
/// or empty var → `None`); anything else is treated as the literal key.
pub fn resolve_provider_api_key(provider_config: &ProviderConfig) -> Option<String> {
    let key = provider_config.api_key.as_deref()?;
    if key == "none" || key.is_empty() {
        return None;
    }
    if let Some(var) = key.strip_prefix('$') {
        return std::env::var(var).ok().filter(|v| !v.is_empty());
    }
    Some(key.to_string())
}

// ---------------------------------------------------------------------------
// Model conversion
// ---------------------------------------------------------------------------

/// Map a provider's `api` string to the engine api-id used by providers
/// (`"anthropic"` → `"anthropic-messages"`, `"openai"` → `"openai-completions"`).
pub fn api_id_from_api(api: &str) -> String {
    match api {
        "anthropic" => "anthropic-messages".to_string(),
        "openai" => "openai-completions".to_string(),
        other => other.to_string(),
    }
}

/// Convert a `ModelConfig` (plus its enclosing provider) into a full
/// `types::Model` ready for the registry.
pub fn model_from_config(provider_name: &str, provider: &ProviderConfig, m: &ModelConfig) -> Model {
    Model {
        id: m.id.clone(),
        name: m.name.clone().unwrap_or_else(|| m.id.clone()),
        api: api_id_from_api(&provider.api),
        provider: provider_name.to_string(),
        base_url: provider.base_url.clone(),
        thinking: m.thinking,
        cost: m.cost.clone(),
        context_window: m.context_window,
        max_tokens: m.max_tokens,
        headers: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Model aliases (models.toml)
// ---------------------------------------------------------------------------

/// Root structure of `models.toml` (global and operator scopes).
#[derive(Debug, Default, Deserialize)]
struct AliasesConfig {
    #[serde(default)]
    aliases: HashMap<String, String>,
}

/// Read the `[aliases]` map from a `models.toml`-shaped file.
///
/// Never fails the caller: a missing/unreadable file returns an empty map,
/// a malformed file prints a warning and returns an empty map — a broken
/// aliases file must not take down the server.
fn read_aliases(path: &Path) -> HashMap<String, String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    match toml::from_str::<AliasesConfig>(&content) {
        Ok(c) => c.aliases,
        Err(e) => {
            eprintln!("config: failed to parse {}: {e}", path.display());
            HashMap::new()
        }
    }
}

/// Global alias map from `config_dir()/models.toml`.
pub fn load_global_aliases(paths: &Paths) -> HashMap<String, String> {
    read_aliases(&paths.models_path())
}

/// Operator alias map from `config_dir()/projects/{project_name}/models.toml`.
pub fn load_operator_aliases(paths: &Paths, project_name: &str) -> HashMap<String, String> {
    read_aliases(&paths.project_config_dir(project_name).join("models.toml"))
}

/// Merge alias maps in priority order (operator > global).
///
/// Operator entries override global entries of the same name.
pub fn merge_alias_maps(
    operator: HashMap<String, String>,
    global: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = global;
    merged.extend(operator);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_paths() -> (tempfile::TempDir, Paths) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let paths = Paths::from_home(dir.path());
        (dir, paths)
    }

    fn write_config(paths: &Paths, filename: &str, content: &str) {
        let dir = paths.config_dir();
        fs::create_dir_all(dir).expect("mkdir config dir");
        fs::write(dir.join(filename), content).expect("write config file");
    }

    // -- providers.toml --

    #[test]
    fn missing_config_file_returns_default() {
        let (_dir, paths) = tmp_paths();
        let config = load_config(&paths).unwrap();
        assert!(config.providers.is_empty());
    }

    #[test]
    fn parse_providers_toml_array_schema() {
        let (_dir, paths) = tmp_paths();
        write_config(
            &paths,
            "providers.toml",
            r#"
[providers.anthropic]
api = "anthropic"
base_url = "https://api.anthropic.com/v1"

[[providers.anthropic.models]]
id = "claude-sonnet-4-6"
name = "Claude Sonnet 4.6"
context_window = 200000
max_tokens = 32000
thinking = "anthropic"

[[providers.anthropic.models]]
id = "claude-haiku-4"
"#,
        );
        let config = load_config(&paths).unwrap();
        let anthropic = config.providers.get("anthropic").expect("provider");
        assert_eq!(anthropic.api, "anthropic");
        assert_eq!(anthropic.base_url, "https://api.anthropic.com/v1");
        assert_eq!(anthropic.models.len(), 2);
        let sonnet = &anthropic.models[0];
        assert_eq!(sonnet.id, "claude-sonnet-4-6");
        assert_eq!(sonnet.context_window, 200_000);
        assert_eq!(sonnet.thinking, ThinkingStyle::Anthropic);
        let haiku = &anthropic.models[1];
        // defaults apply
        assert_eq!(haiku.context_window, 128_000);
        assert_eq!(haiku.max_tokens, 16_384);
        assert_eq!(haiku.thinking, ThinkingStyle::None);
    }

    #[test]
    fn parse_malformed_providers_toml_is_error() {
        let (_dir, paths) = tmp_paths();
        write_config(&paths, "providers.toml", "not [[ valid toml }{");
        let err = load_config(&paths).unwrap_err();
        assert!(matches!(err, crate::Error::Parse(_)));
    }

    #[test]
    fn save_config_roundtrip() {
        let (_dir, paths) = tmp_paths();
        let mut config = Config::default();
        config.providers.insert(
            "local".into(),
            ProviderConfig {
                api: "openai".into(),
                base_url: "http://localhost:8080/v1".into(),
                api_key: Some("$LOCAL_KEY".into()),
                models: vec![ModelConfig {
                    id: "qwen3.5-72b".into(),
                    name: None,
                    context_window: 131_072,
                    max_tokens: 32_768,
                    thinking: ThinkingStyle::Qwen,
                    cost: ModelCost::default(),
                }],
            },
        );
        save_config(&paths, &config).unwrap();
        let loaded = load_config(&paths).unwrap();
        assert_eq!(loaded, config);
        assert_eq!(loaded.providers["local"].models[0].id, "qwen3.5-72b");
    }

    // -- API key resolution --

    #[test]
    fn api_key_expansion() {
        let home = std::env::var("HOME").unwrap();
        let expanded = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: Some("$HOME".into()),
            models: vec![],
        };
        assert_eq!(
            resolve_provider_api_key(&expanded).as_deref(),
            Some(home.as_str())
        );
    }

    #[test]
    fn api_key_literal_and_none() {
        let literal = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: Some("sk-literal".into()),
            models: vec![],
        };
        assert_eq!(
            resolve_provider_api_key(&literal).as_deref(),
            Some("sk-literal")
        );

        let none = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: Some("none".into()),
            models: vec![],
        };
        assert_eq!(resolve_provider_api_key(&none), None);

        let empty = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: Some("".into()),
            models: vec![],
        };
        assert_eq!(resolve_provider_api_key(&empty), None);

        let absent = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: None,
            models: vec![],
        };
        assert_eq!(resolve_provider_api_key(&absent), None);
    }

    #[test]
    fn api_key_unset_var_returns_none() {
        let pc = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: Some("$__TARS_TEST_NO_SUCH_VAR__".into()),
            models: vec![],
        };
        assert_eq!(resolve_provider_api_key(&pc), None);
    }

    // -- model conversion --

    #[test]
    fn model_from_config_maps_fields() {
        let provider = ProviderConfig {
            api: "openai".into(),
            base_url: "http://localhost:8080/v1".into(),
            api_key: None,
            models: vec![],
        };
        let mc = ModelConfig {
            id: "qwen3.5-72b".into(),
            name: None,
            context_window: 131_072,
            max_tokens: 32_768,
            thinking: ThinkingStyle::Qwen,
            cost: ModelCost::default(),
        };
        let model = model_from_config("my-qwen", &provider, &mc);
        assert_eq!(model.id, "qwen3.5-72b");
        assert_eq!(model.name, "qwen3.5-72b"); // name falls back to id
        assert_eq!(model.api, "openai-completions");
        assert_eq!(model.provider, "my-qwen");
        assert_eq!(model.base_url, "http://localhost:8080/v1");
        assert_eq!(model.thinking, ThinkingStyle::Qwen);
    }

    #[test]
    fn api_id_mapping() {
        assert_eq!(api_id_from_api("anthropic"), "anthropic-messages");
        assert_eq!(api_id_from_api("openai"), "openai-completions");
        assert_eq!(api_id_from_api("custom"), "custom");
    }

    // -- aliases --

    #[test]
    fn aliases_missing_file_returns_empty() {
        let (_dir, paths) = tmp_paths();
        assert!(load_global_aliases(&paths).is_empty());
        assert!(load_operator_aliases(&paths, "myproj").is_empty());
    }

    #[test]
    fn aliases_loaded_from_global_models_toml() {
        let (_dir, paths) = tmp_paths();
        write_config(
            &paths,
            "models.toml",
            r#"
[aliases]
smart = "claude-sonnet-4-6"
cheap = "openai/gpt-4.1-mini"
"#,
        );
        let aliases = load_global_aliases(&paths);
        assert_eq!(aliases.len(), 2);
        assert_eq!(
            aliases.get("smart").map(String::as_str),
            Some("claude-sonnet-4-6")
        );
        assert_eq!(
            aliases.get("cheap").map(String::as_str),
            Some("openai/gpt-4.1-mini")
        );
    }

    #[test]
    fn aliases_loaded_from_operator_dir() {
        let (_dir, paths) = tmp_paths();
        let op_dir = paths.project_config_dir("myproj");
        fs::create_dir_all(&op_dir).expect("mkdir operator dir");
        fs::write(
            op_dir.join("models.toml"),
            "[aliases]\nsmart = \"operator-model\"\n",
        )
        .expect("write operator models.toml");

        let aliases = load_operator_aliases(&paths, "myproj");
        assert_eq!(
            aliases.get("smart").map(String::as_str),
            Some("operator-model")
        );
        // other projects unaffected
        assert!(load_operator_aliases(&paths, "other").is_empty());
    }

    #[test]
    fn aliases_malformed_file_returns_empty() {
        let (_dir, paths) = tmp_paths();
        write_config(&paths, "models.toml", "not [[ valid toml }{");
        assert!(load_global_aliases(&paths).is_empty());
    }

    #[test]
    fn merge_operator_overrides_global() {
        let global = HashMap::from([
            ("smart".to_string(), "global-model".to_string()),
            ("global-only".to_string(), "g".to_string()),
        ]);
        let operator = HashMap::from([
            ("smart".to_string(), "operator-model".to_string()),
            ("op-only".to_string(), "o".to_string()),
        ]);
        let merged = merge_alias_maps(operator.clone(), global.clone());
        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged.get("smart").map(String::as_str),
            Some("operator-model")
        );
        assert_eq!(merged.get("global-only").map(String::as_str), Some("g"));
        assert_eq!(merged.get("op-only").map(String::as_str), Some("o"));
    }
}
