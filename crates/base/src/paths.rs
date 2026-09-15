//! XDG-style path resolution for tars directories.
//!
//! Resolution is dependency-injected: call [`Paths::detect`] once to snapshot
//! the environment, or [`Paths::from_home`] to point the whole tree at a
//! specific `HOME` (used in tests and sandboxed runs). All file lookups
//! (config, models, socket) go through a `&Paths` so callers never re-read
//! process-global env vars and tests never mutate them.

use std::path::{Path, PathBuf};

/// Resolved tars directory layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    config_dir: PathBuf,
    data_dir: PathBuf,
    runtime_dir: PathBuf,
    state_dir: PathBuf,
}

impl Paths {
    /// Detect the directory layout from the environment.
    ///
    /// Honour `$XDG_CONFIG_HOME` / `$XDG_DATA_HOME` / `$XDG_RUNTIME_DIR` /
    /// `$XDG_STATE_HOME` when set, fall back to `$HOME`-relative paths, and
    /// when even `$HOME` is unset (containers, minimal environments) fall
    /// back to namespaced `/tmp` directories so `create_dir_all` works
    /// uniformly and files never collide with unrelated `/tmp` entries.
    pub fn detect() -> Self {
        let config_dir = if let Ok(c) = std::env::var("XDG_CONFIG_HOME") {
            PathBuf::from(c).join("tars")
        } else if let Ok(h) = std::env::var("HOME") {
            PathBuf::from(h).join(".config").join("tars")
        } else {
            PathBuf::from("/tmp").join("tars-config")
        };

        let data_dir = if let Ok(d) = std::env::var("XDG_DATA_HOME") {
            PathBuf::from(d).join("tars")
        } else if let Ok(h) = std::env::var("HOME") {
            PathBuf::from(h).join(".local").join("share").join("tars")
        } else {
            PathBuf::from("/tmp").join("tars-data")
        };

        let runtime_dir = if let Ok(r) = std::env::var("XDG_RUNTIME_DIR") {
            PathBuf::from(r).join("tars")
        } else if let Ok(h) = std::env::var("HOME") {
            PathBuf::from(h).join(".tars")
        } else {
            PathBuf::from("/tmp").join(format!("tars-{}", std::process::id()))
        };

        let state_dir = if let Ok(s) = std::env::var("XDG_STATE_HOME") {
            PathBuf::from(s).join("tars")
        } else if let Ok(h) = std::env::var("HOME") {
            PathBuf::from(h).join(".local").join("state").join("tars")
        } else {
            PathBuf::from("/tmp").join("tars-state")
        };

        Self {
            config_dir,
            data_dir,
            runtime_dir,
            state_dir,
        }
    }

    /// Build the layout from a specific `HOME`, ignoring `XDG_*` overrides.
    ///
    /// Useful for tests (nothing touches the real environment) and for
    /// sandboxing the server under an isolated `HOME`.
    pub fn from_home(home: &Path) -> Self {
        Self {
            config_dir: home.join(".config").join("tars"),
            data_dir: home.join(".local").join("share").join("tars"),
            runtime_dir: home.join(".tars"),
            state_dir: home.join(".local").join("state").join("tars"),
        }
    }

    /// `$XDG_CONFIG_HOME/tars` (or `~/.config/tars`). User-editable config.
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// `$XDG_DATA_HOME/tars` (or `~/.local/share/tars`). User data.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// `$XDG_RUNTIME_DIR/tars` (or `~/.tars`). Short-lived runtime files.
    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    /// `$XDG_STATE_HOME/tars` (or `~/.local/state/tars`). Durable
    /// machine state that is not user-editable config: logs, crash dumps.
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// `state_dir()/logs`.
    pub fn logs_dir(&self) -> PathBuf {
        self.state_dir.join("logs")
    }

    /// Default unix-socket path for the tars server.
    pub fn socket_path(&self) -> PathBuf {
        self.runtime_dir.join("tars.sock")
    }

    /// PID file next to the socket.
    pub fn pid_path(&self) -> PathBuf {
        self.runtime_dir.join("tars.pid")
    }

    /// `config_dir()/providers.toml`.
    pub fn providers_path(&self) -> PathBuf {
        self.config_dir.join("providers.toml")
    }

    /// `config_dir()/models.toml` (model aliases).
    pub fn models_path(&self) -> PathBuf {
        self.config_dir.join("models.toml")
    }

    /// Operator config directory for a project: `config_dir()/projects/{name}`.
    pub fn project_config_dir(&self, name: &str) -> PathBuf {
        self.config_dir.join("projects").join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_home_joins_standard_dirs() {
        let home = Path::new("/home/alice");
        let paths = Paths::from_home(home);
        assert_eq!(paths.config_dir(), Path::new("/home/alice/.config/tars"));
        assert_eq!(paths.data_dir(), Path::new("/home/alice/.local/share/tars"));
        assert_eq!(paths.runtime_dir(), Path::new("/home/alice/.tars"));
        assert_eq!(
            paths.state_dir(),
            Path::new("/home/alice/.local/state/tars")
        );
    }

    #[test]
    fn derived_paths_follow_dirs() {
        let home = Path::new("/home/alice");
        let paths = Paths::from_home(home);
        assert_eq!(
            paths.logs_dir(),
            Path::new("/home/alice/.local/state/tars/logs")
        );
        assert_eq!(
            paths.socket_path(),
            Path::new("/home/alice/.tars/tars.sock")
        );
        assert_eq!(paths.pid_path(), Path::new("/home/alice/.tars/tars.pid"));
        assert_eq!(
            paths.providers_path(),
            Path::new("/home/alice/.config/tars/providers.toml")
        );
        assert_eq!(
            paths.models_path(),
            Path::new("/home/alice/.config/tars/models.toml")
        );
        assert_eq!(
            paths.project_config_dir("myproj"),
            Path::new("/home/alice/.config/tars/projects/myproj")
        );
    }

    #[test]
    fn detect_config_data_state_end_with_tars() {
        // Loose smoke test: whatever the environment says, the resolved
        // config/data/state dirs are named "tars". (runtime_dir is "~/.tars"
        // on the HOME fallback, so it is not asserted here.)
        let paths = Paths::detect();
        assert_eq!(
            paths.config_dir().file_name().and_then(|s| s.to_str()),
            Some("tars")
        );
        assert_eq!(
            paths.data_dir().file_name().and_then(|s| s.to_str()),
            Some("tars")
        );
        assert_eq!(
            paths.state_dir().file_name().and_then(|s| s.to_str()),
            Some("tars")
        );
        // socket lives in the runtime dir
        assert!(paths.socket_path().to_string_lossy().contains("tars"));
    }
}
