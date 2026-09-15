//! Project discovery: detect the project a working directory belongs to.
//!
//! A project is the nearest ancestor directory (of a cwd) containing either
//! a `.git` entry (directory, or file in the case of worktrees/submodules)
//! or a `.tars` directory. The project name is the discovered root's
//! directory name.
//!
//! Discovery is a pure filesystem walk — no process-global state. Tests
//! build synthetic trees in tempdirs.

use std::path::{Path, PathBuf};

/// A discovered project: its root directory and derived name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Project {
    /// Directory holding the project marker (`.git` / `.tars`).
    pub root: PathBuf,
    /// The root directory's file name; `None` when the root is `/`.
    pub name: Option<String>,
}

/// Discover the project containing `cwd`, walking up through ancestors.
///
/// Returns `None` when no ancestor holds a `.git` or `.tars` marker.
pub fn discover(cwd: &Path) -> Option<Project> {
    let start = if cwd.is_dir() { cwd } else { cwd.parent()? };
    let mut dir = Some(start);
    while let Some(dir_ref) = dir {
        if is_project_root(dir_ref) {
            return Some(Project {
                root: dir_ref.to_path_buf(),
                name: dir_ref.file_name().map(|n| n.to_string_lossy().to_string()),
            });
        }
        dir = dir_ref.parent();
    }
    None
}

fn is_project_root(dir: &Path) -> bool {
    dir.join(".git").exists() || dir.join(".tars").is_dir()
}

/// Merge alias maps in priority order (operator > global) — thin wrapper
/// kept here so callers of the project module do not reach into config.
///
/// Returns `None` when `project` is `None` (no project ⇒ no operator tier).
pub fn operator_alias_tier(
    project: Option<&Project>,
    paths: &crate::Paths,
) -> Option<std::collections::HashMap<String, String>> {
    let name = project?.name.as_deref()?;
    Some(crate::config::load_operator_aliases(paths, name))
}

/// Resolve `raw` through the project-scoped config tier: operator aliases
/// for the project containing `cwd` take priority over global aliases.
///
/// Returns the alias target when `raw` is a known alias, `None` otherwise
/// (the caller keeps `raw` as a literal model id).
pub fn alias_target_for_cwd(raw: &str, cwd: Option<&Path>, paths: &crate::Paths) -> Option<String> {
    let project = cwd.and_then(discover);
    let operator = operator_alias_tier(project.as_ref(), paths).unwrap_or_default();
    let global = crate::config::load_global_aliases(paths);
    let merged = crate::config::merge_alias_maps(operator, global);
    merged.get(raw).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_dir_discovered() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("myproj");
        std::fs::create_dir_all(root.join("src/nested")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();

        let project = discover(&root.join("src/nested")).expect("discovered");
        assert_eq!(project.root, root);
        assert_eq!(project.name.as_deref(), Some("myproj"));
    }

    #[test]
    fn git_file_counts_as_root() {
        // Worktrees and submodules use a `.git` file, not a directory.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("worktree");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".git"), "gitdir: /elsewhere").unwrap();

        let project = discover(&root).unwrap();
        assert_eq!(project.root, root);
    }

    #[test]
    fn tars_dir_discovered() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("tarred");
        std::fs::create_dir_all(root.join(".tars")).unwrap();

        let project = discover(&root).unwrap();
        assert_eq!(project.root, root);
        assert_eq!(project.name.as_deref(), Some("tarred"));
    }

    #[test]
    fn nearest_marker_wins() {
        let dir = tempfile::tempdir().unwrap();
        let outer = dir.path().join("outer");
        let inner = outer.join("inner");
        std::fs::create_dir_all(inner.join(".git")).unwrap();
        std::fs::create_dir_all(outer.join(".tars")).unwrap();

        let project = discover(&inner).unwrap();
        assert_eq!(project.root, inner, "closest marker wins over parent's");
    }

    #[test]
    fn no_marker_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert!(discover(&plain).is_none());
    }

    #[test]
    fn missing_path_uses_parent() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        // A file path that does not exist yet inside the project.
        let ghost = root.join("ghost.txt");
        let project = discover(&ghost).unwrap();
        assert_eq!(project.root, root);
    }

    #[test]
    fn operator_alias_tier_loads_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("aliased");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let paths = crate::Paths::from_home(dir.path());
        let op_dir = paths.project_config_dir("aliased");
        std::fs::create_dir_all(&op_dir).unwrap();
        std::fs::write(
            op_dir.join("models.toml"),
            "[aliases]\nsmart = \"opus-4\"\n",
        )
        .unwrap();

        let project = discover(&root).unwrap();
        let tier = operator_alias_tier(Some(&project), &paths).unwrap();
        assert_eq!(tier.get("smart").map(String::as_str), Some("opus-4"));

        // No project ⇒ no operator tier.
        assert!(operator_alias_tier(None, &paths).is_none());
    }

    #[test]
    fn alias_target_for_cwd_uses_operator_over_global() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("aliased");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let paths = crate::Paths::from_home(dir.path());
        // Global alias: smart → global-model
        let cfg_dir = paths.config_dir();
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(
            cfg_dir.join("models.toml"),
            "[aliases]\nsmart = \"global-model\"\n",
        )
        .unwrap();
        // Operator alias: smart → operator-model
        let op_dir = paths.project_config_dir("aliased");
        std::fs::create_dir_all(&op_dir).unwrap();
        std::fs::write(
            op_dir.join("models.toml"),
            "[aliases]\nsmart = \"operator-model\"\n",
        )
        .unwrap();

        assert_eq!(
            alias_target_for_cwd("smart", Some(&root), &paths),
            Some("operator-model".into())
        );
        // Outside the project, the global tier applies.
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(
            alias_target_for_cwd("smart", Some(&plain), &paths),
            Some("global-model".into())
        );
        // Non-alias input maps to None.
        assert_eq!(alias_target_for_cwd("opus-4", Some(&root), &paths), None);
    }
}
