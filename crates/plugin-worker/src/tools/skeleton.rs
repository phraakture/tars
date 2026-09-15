//! `get_file_skeleton` tool — structural outline (classes, functions,
//! methods, including nested) of one or more source files via tree-sitter.
//!
//! One header line per definition, in source order, deduped by start row.
//! The header is the definition's signature: the source slice from the
//! definition's start through its body opener (`{`, or `:` for Python).
//! Multi-line signatures are collapsed onto one virtual line; single-line
//! signatures are emitted verbatim.
//!
//! Per-file failures render inline as `error: ...` blocks. The call is
//! flagged `is_error` only when every requested path failed.

use super::tree_sitter_support::{self, Lang};
use super::{ToolDef, ToolOutput};
use std::collections::BTreeMap;

use tars_base::{CancelToken, Tool};
use tree_sitter::{Node, QueryCursor, StreamingIterator};

/// Max paths per call (mirrors `read` so budgeting is consistent).
pub(crate) const MAX_PATHS: usize = 20;
/// Total output byte cap, defensive.
pub(crate) const MAX_TOTAL_BYTES: usize = 256 * 1024;

pub(crate) fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "get_file_skeleton".into(),
            description: "Quickly outline source files (functions, methods, classes, types; no bodies) using tree-sitter. Supports Rust, Python, JavaScript, TypeScript, and TSX. Other extensions return a per-file error suggesting `read` instead."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_PATHS,
                        "description": "Paths to source files to outline (1-20)."
                    }
                },
                "required": ["paths"]
            }),
        },
        execute: Box::new(execute),
        prepare_arguments: None,
    }
}

fn execute(args: serde_json::Value, cwd: &str, _cancel: &CancelToken) -> ToolOutput {
    let Some(paths) = args.get("paths").and_then(|p| p.as_array()) else {
        return ToolOutput::error("missing required argument: paths");
    };
    if paths.is_empty() || paths.len() > MAX_PATHS {
        return ToolOutput::error(format!("paths must contain 1..{MAX_PATHS} entries"));
    }

    let mut out = String::new();
    let mut ok = 0usize;
    for p in paths {
        let Some(path) = p.as_str() else {
            continue;
        };
        let rendered = outline_path(cwd, path);
        if !rendered.starts_with("error:") {
            ok += 1;
        }
        out.push_str(&rendered);
        if out.len() > MAX_TOTAL_BYTES {
            out.push_str("\n... output truncated ...\n");
            break;
        }
    }

    if ok == 0 {
        return ToolOutput::error(out);
    }
    ToolOutput::text(out).with_summary(format!("outlined {ok} file(s)"))
}

fn outline_path(cwd: &str, path: &str) -> String {
    let full = super::resolve_path(cwd, path);
    let header = format!("===== {} =====\n", full.display());

    let ext = full
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let Some(lang) = Lang::from_extension(&ext) else {
        return format!("error: {header}no skeleton support for .{ext}; use `read` instead\n");
    };

    let source = match std::fs::read_to_string(&full) {
        Ok(s) => s,
        Err(e) => return format!("error: {header}{e}"),
    };

    match extract(source.as_str(), lang) {
        Ok(headers) if headers.is_empty() => format!("{header}(no definitions found)\n"),
        Ok(headers) => {
            let mut out = header;
            for h in headers {
                out.push_str(&h);
                out.push('\n');
            }
            out
        }
        Err(e) => format!("error: {header}{e}"),
    }
}

/// Extract signature header lines: source-ordered, deduped by name row.
pub(crate) fn extract(source: &str, lang: Lang) -> Result<Vec<String>, String> {
    let tree = tree_sitter_support::parse(lang, source)?;
    let query = tree_sitter_support::query_for(lang);
    let capture_names = query.capture_names();

    let mut cursor = QueryCursor::new();
    let mut headers: BTreeMap<usize, String> = BTreeMap::new();
    let mut matches = cursor.matches(query, tree.root_node(), source.as_bytes());

    while let Some(m) = matches.next() {
        let mut name_node: Option<Node<'_>> = None;
        let mut def_node: Option<Node<'_>> = None;
        for cap in m.captures {
            let Some(name) = capture_names.get(cap.index as usize) else {
                continue;
            };
            if name.starts_with("name.def") {
                name_node.get_or_insert(cap.node);
            } else if name.starts_with("definition") {
                def_node = Some(cap.node);
            }
        }
        let Some(name_node) = name_node else { continue };
        let row = name_node.start_position().row;
        if headers.contains_key(&row) {
            continue;
        }
        let def_node = def_node.unwrap_or(name_node);
        let line = render_signature(source, def_node);
        headers.insert(row, line);
    }

    Ok(headers.into_values().collect())
}

/// Render one signature line for a definition.
///
/// Slice from the definition start byte through its body opener (`{` for
/// braced languages, the `block` child's start for Python), trimmed and
/// collapsed onto a single line. Definitions without bodies (`pub type X
/// = i32;`, trait method signatures) emit the full node text with any
/// trailing `;` trimmed.
fn render_signature(source: &str, def: Node<'_>) -> String {
    let text = source.as_bytes();

    // Locate the body within the node, if any.
    let mut body_start = def.end_byte();
    let mut cursor = def.walk();
    for child in def.children(&mut cursor) {
        if matches!(
            child.kind(),
            "body"
                | "block"
                | "statement_block"
                | "field_declaration_list"
                | "declaration_list"
                | "enum_body"
                | "enum_variant_list"
                | "class_body"
                | "interface_body"
        ) {
            body_start = child.start_byte();
            break;
        }
    }

    let has_body = body_start != def.end_byte();
    let slice_end = if has_body { body_start } else { def.end_byte() };

    let mut sig = String::from_utf8_lossy(&text[def.start_byte()..slice_end])
        .trim()
        .to_string();
    // Strip leftover body-openers / separators from the slice. Python's
    // colon belongs to the header (the block child starts after it), so
    // it is kept.
    while sig.ends_with('{') || sig.ends_with(';') {
        sig.pop();
    }
    let mut sig = sig.trim().to_string();

    // Definitions without a body that are Python headers keep their colon.
    if !has_body && matches!(def.kind(), "function_definition" | "class_definition") {
        sig = sig.trim_end_matches(':').trim().to_string();
    }

    if !sig.contains('\n') {
        return sig;
    }

    // Multi-line signature: preserve first-line indent, collapse the rest.
    let first_indent = sig
        .lines()
        .next()
        .map(|l| l.len() - l.trim_start().len())
        .unwrap_or(0);
    let mut collapsed = String::new();
    for l in sig.lines() {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        if !collapsed.is_empty() {
            collapsed.push(' ');
        }
        collapsed.push_str(l);
    }
    // Tighten spacing the line-join introduced around brackets and commas.
    let collapsed = collapsed
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" ,", ",")
        .replace(",)", ")")
        .replace("[ ", "[")
        .replace(" ]", "]");
    format!("{}{collapsed}", " ".repeat(first_indent.min(8)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::CancelToken;

    fn run(args: serde_json::Value, cwd: &str) -> ToolOutput {
        execute(args, cwd, &CancelToken::new())
    }

    #[test]
    fn rust_skeleton_outline() {
        let dir = tempfile::tempdir().unwrap();
        let src = r#"
use std::fmt;

/// Docs on the struct
#[derive(Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub enum Shape {
    Circle,
    Square,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

pub trait Drawable {
    fn draw(&self);
}

pub type Meters = f64;

mod internal {
    pub fn helper() {}
}
"#;
        let path = dir.path().join("lib.rs");
        std::fs::write(&path, src).unwrap();

        let out = run(
            serde_json::json!({"paths": [path.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error, "expected success, got {:?}", out.content);
        let text = out.content[0].text();

        assert!(text.contains("===== "), "header present");
        assert!(text.contains("pub struct Point"), "struct: {text}");
        assert!(text.contains("pub enum Shape"), "enum: {text}");
        assert!(text.contains("impl Point"), "impl: {text}");
        assert!(
            text.contains("pub fn new(x: f64, y: f64) -> Self"),
            "method: {text}"
        );
        assert!(
            text.contains("fn distance(&self) -> f64"),
            "private method: {text}"
        );
        assert!(text.contains("pub trait Drawable"), "trait: {text}");
        assert!(text.contains("fn draw(&self)"), "trait method: {text}");
        assert!(text.contains("pub type Meters"), "type alias: {text}");
        assert!(text.contains("mod internal"), "module: {text}");
        assert!(text.contains("pub fn helper()"), "nested fn: {text}");

        // Bodies must not leak
        assert!(!text.contains("sqrt"), "no impl bodies: {text}");
        assert!(!text.contains("Point { x, y }"), "no method bodies: {text}");
    }

    #[test]
    fn rust_multiline_signature_collapses() {
        let dir = tempfile::tempdir().unwrap();
        let src = r#"
pub fn long_function(
    first: String,
    second: u64,
) -> Result<String, Error> {
    Ok(first)
}
"#;
        let path = dir.path().join("a.rs");
        std::fs::write(&path, src).unwrap();
        let out = run(
            serde_json::json!({"paths": [path.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error);
        let text = out.content[0].text();
        assert!(
            text.contains(
                "pub fn long_function(first: String, second: u64) -> Result<String, Error>"
            ),
            "collapsed: {text}"
        );
    }

    #[test]
    fn python_skeleton_outline() {
        let dir = tempfile::tempdir().unwrap();
        let src = "import os\n\n@decorator\nclass Widget:\n    def __init__(self, name):\n        self.name = name\n\n    def render(self):\n        return self.name\n\ndef helper(x):\n    return x\n";
        let path = dir.path().join("w.py");
        std::fs::write(&path, src).unwrap();
        let out = run(
            serde_json::json!({"paths": [path.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error, "{:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("class Widget:"), "class: {text}");
        assert!(text.contains("def __init__(self, name):"), "init: {text}");
        assert!(text.contains("def render(self):"), "method: {text}");
        assert!(text.contains("def helper(x):"), "fn: {text}");
        // Bodies must not leak
        assert!(!text.contains("return self.name"), "no bodies: {text}");
        assert!(!text.contains("@decorator"), "decorators excluded: {text}");
    }

    #[test]
    fn typescript_skeleton_outline() {
        let dir = tempfile::tempdir().unwrap();
        let src = r#"
export interface Shape {
    area(): number;
}

export class Circle implements Shape {
    constructor(private r: number) {}

    area(): number {
        return Math.PI * this.r * this.r;
    }
}

export function areaOf(s: Shape): number {
    return s.area();
}

const shrink = (f: number): number => f * 0.5;
"#;
        let path = dir.path().join("a.ts");
        std::fs::write(&path, src).unwrap();
        let out = run(
            serde_json::json!({"paths": [path.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error, "{:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("interface Shape"), "interface: {text}");
        assert!(text.contains("class Circle"), "class: {text}");
        assert!(
            text.contains("constructor(private r: number)"),
            "ctor: {text}"
        );
        assert!(text.contains("area(): number"), "method: {text}");
        assert!(text.contains("function areaOf"), "fn: {text}");
        assert!(text.contains("const shrink"), "arrow fn: {text}");
        assert!(!text.contains("Math.PI"), "no bodies: {text}");
    }

    #[test]
    fn unsupported_extension_errors_per_file() {
        let dir = tempfile::tempdir().unwrap();
        let md = dir.path().join("notes.md");
        std::fs::write(&md, "# hi").unwrap();
        let out = run(serde_json::json!({"paths": [md.to_str().unwrap()]}), "/tmp");
        assert!(out.is_error, "all-fail call is an error");
        let text = out.content[0].text();
        assert!(text.contains("no skeleton support for .md"), "{text}");
        assert!(text.contains("use `read` instead"), "{text}");
    }

    #[test]
    fn partial_success_is_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let md = dir.path().join("notes.md");
        std::fs::write(&md, "# hi").unwrap();
        let rs = dir.path().join("ok.rs");
        std::fs::write(&rs, "fn main() {}").unwrap();
        let out = run(
            serde_json::json!({"paths": [md.to_str().unwrap(), rs.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error, "one success keeps the call successful");
        let text = out.content[0].text();
        assert!(text.contains("no skeleton support"), "{text}");
        assert!(text.contains("fn main()"), "{text}");
    }

    #[test]
    fn missing_file_errors_per_file() {
        let out = run(
            serde_json::json!({"paths": ["/nonexistent/file.rs"]}),
            "/tmp",
        );
        assert!(out.is_error);
        let text = out.content[0].text();
        assert!(text.contains("error:"), "{text}");
    }

    #[test]
    fn paths_validation() {
        let out = run(serde_json::json!({}), "/tmp");
        assert!(out.is_error);
        let out = run(serde_json::json!({"paths": []}), "/tmp");
        assert!(out.is_error);
    }

    #[test]
    fn empty_definitions_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.rs");
        std::fs::write(&path, "// just a comment").unwrap();
        let out = run(
            serde_json::json!({"paths": [path.to_str().unwrap()]}),
            "/tmp",
        );
        assert!(!out.is_error);
        assert!(
            out.content[0].text().contains("no definitions found"),
            "{:?}",
            out.content
        );
    }
}
