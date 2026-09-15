//! `diagnostics_scan` tool — run lint/syntax diagnostics on specific files
//! and return structured per-file feedback.
//!
//! Built-in: Rust files are grouped by their enclosing cargo project and
//! checked with `cargo check --message-format=short` (one invocation per
//! project, output filtered to the requested files). Other extensions are
//! skipped unless a `.tars/diagnostics.toml` file found in a nearest
//! ancestor directory defines a command for them:
//!
//! ```toml
//! [[tool]]
//! extensions = ["py"]
//! command = "ruff check {file}"
//! ```
//!
//! `{file}` is substituted with the requested path. Configured tools run
//! via `sh -c`; any output becomes one `info`-severity diagnostic for the
//! file.
//!
//! Output is structured JSON:
//!
//! ```json
//! {
//!   "summary": {"files_scanned": 2, "errors": 1, "warnings": 3, "files_skipped": 0},
//!   "diagnostics": [{"file": "...", "line": 2, "column": 9, "severity": "error", "message": "..."}],
//!   "skipped": [{"path": "...", "reason": "..."}]
//! }
//! ```
//!
//! `is_error` stays false even when diagnostics are present — the model
//! reads the JSON and counts errors itself. The call only fails when no
//! file at all could be scanned.

use super::{ToolDef, ToolOutput};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use tars_base::{CancelToken, Tool};

pub(crate) const MAX_PATHS: usize = 20;

pub(crate) fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "diagnostics_scan".into(),
            description: "Run lint/syntax diagnostics on specific files and get structured per-file feedback (built-in: Rust via cargo check; configurable via .tars/diagnostics.toml). Pass only the files you actually changed; the tool resolves project context automatically. Output is structured JSON ({summary, diagnostics[], skipped[]}); is_error is false even when diagnostics are present."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_PATHS,
                        "description": "Files to diagnose (1-20)."
                    }
                },
                "required": ["paths"]
            }),
        },
        execute: Box::new(execute),
        prepare_arguments: None,
    }
}

// ---------------------------------------------------------------------------
// Config (.tars/diagnostics.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct ConfiguredTool {
    pub extensions: Vec<String>,
    pub command: String,
}

fn load_config(dir: &Path) -> Vec<ConfiguredTool> {
    let path = dir.join(".tars").join("diagnostics.toml");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };

    #[derive(serde::Deserialize)]
    struct ToolEntry {
        extensions: Vec<String>,
        command: String,
    }

    #[derive(serde::Deserialize)]
    struct Root {
        #[serde(default)]
        tool: Vec<ToolEntry>,
    }

    match toml::from_str::<Root>(&content) {
        Ok(root) => root
            .tool
            .into_iter()
            .map(|t| ConfiguredTool {
                extensions: t
                    .extensions
                    .iter()
                    .map(|s| s.trim_start_matches('.').to_string())
                    .collect(),
                command: t.command,
            })
            .collect(),
        Err(e) => {
            tracing::warn!("failed to parse {}: {e}", path.display());
            Vec::new()
        }
    }
}

/// Nearest ancestor of `path` that contains a marker (e.g. `Cargo.toml`),
/// mirroring how build tools resolve the project root.
fn find_project_root(path: &Path, marker: &str) -> Option<PathBuf> {
    let start = if path.is_dir() { path } else { path.parent()? };
    let mut dir = Some(start);
    while let Some(dir_ref) = dir {
        if dir_ref.join(marker).is_file() {
            return Some(dir_ref.to_path_buf());
        }
        dir = dir_ref.parent();
    }
    None
}

// ---------------------------------------------------------------------------
// cargo check
// ---------------------------------------------------------------------------

/// One cargo-check (or configured-tool) finding.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct Diagnostic {
    pub file: String,
    pub line: u64,
    pub column: u64,
    pub severity: String,
    pub message: String,
}

/// Locate `path:line:col: severity: message` in one cargo short-format
/// line. Handles bracketed error codes (`error[E0308]:`) and bare
/// cargo-level lines (`error: could not compile ...`, no location).
fn split_short_line(line: &str) -> Option<(String, String, String)> {
    const SEVS: [&str; 4] = ["error", "warning", "help", "note"];
    for sev in SEVS {
        let plain = format!(": {sev}: ");
        let bracket = format!(": {sev}[");
        if let Some(idx) = line.find(&plain) {
            let loc = &line[..idx];
            if loc.matches(':').count() >= 2
                && loc
                    .split(':')
                    .nth(1)
                    .map(|s| s.parse::<u64>().is_ok())
                    .unwrap_or(false)
            {
                return Some((
                    loc.to_string(),
                    sev.to_string(),
                    line[idx + plain.len()..].trim().to_string(),
                ));
            }
        } else if let Some(idx) = line.find(&bracket) {
            let loc = &line[..idx];
            if loc.matches(':').count() >= 2 {
                // Skip past the code: `E0308]: message`
                let rest = &line[idx + bracket.len()..];
                if let Some(close) = rest.find("]: ") {
                    return Some((
                        loc.to_string(),
                        sev.to_string(),
                        rest[close + 3..].to_string(),
                    ));
                }
            }
        }
    }
    if let Some(rest) = line.strip_prefix("error: ") {
        return Some((String::new(), "error".into(), rest.to_string()));
    }
    if let Some(rest) = line.strip_prefix("warning: ") {
        return Some((String::new(), "warning".into(), rest.to_string()));
    }
    None
}

fn parse_short_format(stderr: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((loc, severity, message)) = split_short_line(line) {
            let mut parts = loc.split(':');
            let file = parts.next().unwrap_or("").to_string();
            let (l, c) = match (parts.next(), parts.next()) {
                (Some(l), Some(c)) if l.parse::<u64>().is_ok() => (
                    l.parse::<u64>().unwrap_or(0),
                    c.parse::<u64>().ok().unwrap_or(0),
                ),
                _ => (0, 0),
            };
            out.push(Diagnostic {
                file,
                line: l,
                column: c,
                severity,
                message,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize, Clone)]
pub(crate) struct SkippedEntry {
    pub path: String,
    pub reason: String,
}

fn execute(args: serde_json::Value, cwd: &str, _cancel: &CancelToken) -> ToolOutput {
    let Some(paths) = args.get("paths").and_then(|p| p.as_array()) else {
        return ToolOutput::error("missing required argument: paths");
    };
    if paths.is_empty() || paths.len() > MAX_PATHS {
        return ToolOutput::error(format!("paths must contain 1..{MAX_PATHS} entries"));
    }

    let cwd = PathBuf::from(cwd);

    // Resolve requested files (absolute, existing).
    let mut files: Vec<PathBuf> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for p in paths {
        let Some(p) = p.as_str() else { continue };
        let full = super::resolve_path(cwd.to_str().unwrap_or(""), p);
        if full.is_file() {
            files.push(full);
        } else {
            missing.push(p.to_string());
        }
    }

    // Group: rs → cargo projects; others → configured tools by extension.
    let mut rust_by_root: BTreeMap<PathBuf, BTreeSet<PathBuf>> = BTreeMap::new();
    let mut skipped: Vec<SkippedEntry> = missing
        .into_iter()
        .map(|path| SkippedEntry {
            path,
            reason: "file does not exist".into(),
        })
        .collect();
    let mut configured_by_root: BTreeMap<PathBuf, Vec<(PathBuf, String)>> = BTreeMap::new();

    for file in &files {
        if file.extension().and_then(|e| e.to_str()) == Some("rs") {
            let root = find_project_root(file, "Cargo.toml").unwrap_or_else(|| cwd.clone());
            rust_by_root.entry(root).or_default().insert(file.clone());
        } else {
            let ext = file
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let root =
                find_project_root(file, ".tars/diagnostics.toml").unwrap_or_else(|| cwd.clone());
            let Some(tool) = load_config(&root)
                .into_iter()
                .find(|t| t.extensions.iter().any(|e| e.eq_ignore_ascii_case(&ext)))
            else {
                skipped.push(SkippedEntry {
                    path: file.display().to_string(),
                    reason: format!(
                        "no built-in diagnostics for .{ext}; configure a tool in .tars/diagnostics.toml"
                    ),
                });
                continue;
            };
            configured_by_root
                .entry(root)
                .or_default()
                .push((file.clone(), tool.command.clone()));
        }
    }

    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut files_scanned = 0usize;

    // cargo check per project root.
    for (root, project_files) in &rust_by_root {
        let output = std::process::Command::new("cargo")
            .args(["check", "--message-format=short", "-q"])
            .current_dir(root)
            .output();
        let Ok(output) = output else {
            for f in project_files {
                skipped.push(SkippedEntry {
                    path: f.display().to_string(),
                    reason: "failed to run cargo check".into(),
                });
            }
            continue;
        };
        files_scanned += project_files.len();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let wanted: BTreeSet<String> = project_files
            .iter()
            .map(|f| normalize_for_compare(f))
            .collect();
        for d in parse_cargo_check(&stderr, root) {
            let dfile = normalize_for_compare(Path::new(&d.file));
            if wanted.contains(&dfile) {
                diagnostics.push(d);
            }
        }
    }

    // Configured tools.
    for (root, entries) in &configured_by_root {
        for (file, command) in entries {
            let substituted = command.replace("{file}", &file.display().to_string());
            let result = std::process::Command::new("sh")
                .args(["-c", &substituted])
                .current_dir(root)
                .output();
            files_scanned += 1;
            match result {
                Ok(out) => {
                    let text = String::from_utf8_lossy(&out.stdout);
                    let text = if text.trim().is_empty() {
                        String::from_utf8_lossy(&out.stderr).to_string()
                    } else {
                        text.to_string()
                    };
                    diagnostics.push(Diagnostic {
                        file: file.display().to_string(),
                        line: 0,
                        column: 0,
                        severity: "info".into(),
                        message: text.trim().to_string(),
                    });
                }
                Err(e) => skipped.push(SkippedEntry {
                    path: file.display().to_string(),
                    reason: format!("failed to run configured tool: {e}"),
                }),
            }
        }
    }

    let errors = diagnostics.iter().filter(|d| d.severity == "error").count();
    let warnings = diagnostics
        .iter()
        .filter(|d| d.severity == "warning")
        .count();

    #[derive(serde::Serialize)]
    struct Summary {
        files_scanned: usize,
        errors: usize,
        warnings: usize,
        files_skipped: usize,
    }
    #[derive(serde::Serialize)]
    struct Report {
        summary: Summary,
        diagnostics: Vec<Diagnostic>,
        skipped: Vec<SkippedEntry>,
    }

    let report = Report {
        summary: Summary {
            files_scanned,
            errors,
            warnings,
            files_skipped: skipped.len(),
        },
        diagnostics,
        skipped,
    };

    let json = serde_json::to_string_pretty(&report)
        .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize report: {e}\"}}"));

    if files_scanned == 0 {
        return ToolOutput::error(json).with_summary("no file could be scanned");
    }
    ToolOutput::text(json).with_summary(format!(
        "{files_scanned} file(s) scanned: {errors} errors, {warnings} warnings"
    ))
}

fn normalize_for_compare(p: &Path) -> String {
    match p.canonicalize() {
        Ok(c) => c.display().to_string(),
        Err(_) => p.display().to_string(),
    }
}

/// Parse cargo short-format output, resolving relative file paths against
/// the project root.
fn parse_cargo_check(stderr: &str, root: &Path) -> Vec<Diagnostic> {
    parse_short_format(stderr)
        .into_iter()
        .map(|mut d| {
            if !d.file.is_empty() && !Path::new(&d.file).is_absolute() {
                d.file = root.join(&d.file).display().to_string();
            }
            d
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ToolOutput, execute_tool};
    use tars_base::{CancelToken, ToolCall};

    fn run_tool(args: serde_json::Value, cwd: &str) -> ToolOutput {
        let result = execute_tool(
            &[tool_def()],
            &ToolCall {
                id: "tc_d".into(),
                name: "diagnostics_scan".into(),
                arguments: args,
            },
            cwd,
            &CancelToken::new(),
        );
        ToolOutput {
            content: result.content,
            is_error: result.is_error,
            summary: result.summary,
        }
    }

    #[test]
    fn parse_short_format_basics() {
        let stderr = "\
src/main.rs:2:9: warning: unused variable: `x`
src/lib.rs:10:5: error[E0308]: mismatched types
error: could not compile `proj` (bin \"proj\") due to 1 previous error
";
        let diags = parse_cargo_check(stderr, Path::new("/proj"));
        assert_eq!(diags.len(), 3);
        assert_eq!(diags[0].file, "/proj/src/main.rs");
        assert_eq!(diags[0].line, 2);
        assert_eq!(diags[0].column, 9);
        assert_eq!(diags[0].severity, "warning");
        assert!(diags[1].message.contains("mismatched types"));
        assert_eq!(diags[1].severity, "error");
        // Location-less cargo error keeps its text.
        assert!(diags[2].message.contains("could not compile"));
        assert_eq!(diags[2].line, 0);
    }

    #[test]
    fn cargo_project_scan() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"bad\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        // Deliberate error + warning
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    let unused = 5;\n    a + \"x\"\n}\n",
        )
        .unwrap();

        let out = run_tool(
            serde_json::json!({"paths": [root.join("src/lib.rs").display().to_string()]}),
            "/tmp",
        );
        assert!(
            !out.is_error,
            "diagnostics present but call succeeds: {:?}",
            out.content
        );
        let json: serde_json::Value =
            serde_json::from_str(out.content[0].text()).expect("valid JSON");
        assert_eq!(json["summary"]["errors"], 1, "{json}");
        let diag = &json["diagnostics"][0];
        assert_eq!(diag["severity"], "error");
        assert!(
            diag["file"].as_str().unwrap().ends_with("src/lib.rs"),
            "{json}"
        );
    }

    #[test]
    fn clean_project_zero_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"good\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();

        let out = run_tool(
            serde_json::json!({"paths": [root.join("src/lib.rs").display().to_string()]}),
            "/tmp",
        );
        assert!(!out.is_error);
        let json: serde_json::Value = serde_json::from_str(out.content[0].text()).unwrap();
        assert_eq!(json["summary"]["files_scanned"], 1);
        assert_eq!(json["summary"]["errors"], 0);
        assert_eq!(json["diagnostics"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn unconfigured_extension_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let py = dir.path().join("script.py");
        std::fs::write(&py, "print('hi')\n").unwrap();

        let out = run_tool(serde_json::json!({"paths": [py.to_str().unwrap()]}), "/tmp");
        assert!(out.is_error, "nothing scanned: {:?}", out.content);
        let json: serde_json::Value = serde_json::from_str(out.content[0].text()).unwrap();
        assert_eq!(json["summary"]["files_scanned"], 0);
        let skipped = json["skipped"].as_array().unwrap();
        assert!(
            skipped[0]["reason"]
                .as_str()
                .unwrap()
                .contains("configure a tool"),
            "{json}"
        );
    }

    #[test]
    fn configured_tool_runs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".tars")).unwrap();
        std::fs::write(
            root.join(".tars/diagnostics.toml"),
            "[[tool]]\nextensions = [\"py\"]\ncommand = \"echo checked {file}\"\n",
        )
        .unwrap();
        let py = root.join("script.py");
        std::fs::write(&py, "print('hi')\n").unwrap();

        let out = run_tool(serde_json::json!({"paths": [py.to_str().unwrap()]}), "/tmp");
        assert!(!out.is_error, "{:?}", out.content);
        let json: serde_json::Value = serde_json::from_str(out.content[0].text()).unwrap();
        assert_eq!(json["summary"]["files_scanned"], 1);
        let diag = &json["diagnostics"][0];
        assert!(
            diag["message"].as_str().unwrap().contains("checked"),
            "{json}"
        );
    }

    #[test]
    fn missing_file_skipped() {
        let out = run_tool(serde_json::json!({"paths": ["/nope/missing.rs"]}), "/tmp");
        assert!(out.is_error);
        let json: serde_json::Value = serde_json::from_str(out.content[0].text()).unwrap();
        assert_eq!(json["summary"]["files_skipped"], 1);
    }

    #[test]
    fn args_validation() {
        let out = run_tool(serde_json::json!({}), "/tmp");
        assert!(out.is_error);
    }
}
