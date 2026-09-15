//! `get_function` tool — extract the complete source bodies of named
//! functions / methods from one or more files using tree-sitter.
//!
//! Pairs with `get_file_skeleton`: skeleton to skim, `get_function` to
//! drill into the bodies you actually want.
//!
//! ## Name matching
//!
//! Requests use dot-paths (`Foo.bar`, `Outer.Inner.method`); Rust's `::`
//! is accepted as an alias and normalised to `.`. A request `bar` matches
//! any definition whose normalised full name ends with `.bar` (or equals
//! `bar`); `Foo.bar` requires the suffix `Foo.bar`. All matches are
//! returned — ambiguity is resolved by the model reading the output.
//!
//! ## Extended range
//!
//! Each emitted body extends over preceding attributes / doc comments
//! (`#[derive]`, `///`, `@decorator`) so the slice starts at the first
//! relevant annotation, and covers the whole definition node verbatim.
//!
//! The call is flagged `is_error` only when zero names were extracted
//! from all files; misses are reported in a footer otherwise.

use super::tree_sitter_support::{self, Lang};
use super::{ToolDef, ToolOutput};
use tars_base::{CancelToken, Tool};
use tree_sitter::{Node, QueryCursor, StreamingIterator};

pub(crate) const MAX_PATHS: usize = 20;
pub(crate) const MAX_NAMES: usize = 32;
pub(crate) const MAX_TOTAL_BYTES: usize = 256 * 1024;

pub(crate) fn tool_def() -> ToolDef {
    ToolDef {
        tool: Tool {
            name: "get_function".into(),
            description: "Extract complete bodies of named functions/methods from one or more files (tree-sitter). Pairs with get_file_skeleton: skim the skeleton first, then pull bodies you actually need instead of reading whole files. Function names support dot-paths for methods (`Foo.bar`, `ClassName.methodName`); `::` is also accepted for Rust. Bare names (`bar`) match any definition whose qualified name ends in `.bar` — multiple matches are returned together."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_PATHS,
                        "description": "Paths to search (1-20)."
                    },
                    "function_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_NAMES,
                        "description": "Qualified names to extract (dot-paths; `::` accepted)."
                    }
                },
                "required": ["paths", "function_names"]
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
    let Some(names) = args.get("function_names").and_then(|p| p.as_array()) else {
        return ToolOutput::error("missing required argument: function_names");
    };
    if paths.is_empty() || paths.len() > MAX_PATHS {
        return ToolOutput::error(format!("paths must contain 1..{MAX_PATHS} entries"));
    }
    if names.is_empty() || names.len() > MAX_NAMES {
        return ToolOutput::error(format!(
            "function_names must contain 1..{MAX_NAMES} entries"
        ));
    }
    let names: Vec<String> = names
        .iter()
        .filter_map(|n| n.as_str().map(normalize))
        .collect();

    let mut out = String::new();
    let mut found = 0usize;
    let mut misses: Vec<String> = Vec::new();

    'files: for p in paths {
        let Some(path) = p.as_str() else { continue };
        let full = super::resolve_path(cwd, path);
        let header = format!("===== {} =====\n", full.display());

        let ext = full
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let Some(lang) = Lang::from_extension(&ext) else {
            out.push_str(&format!(
                "{header}error: no skeleton support for .{ext}; use `read` instead\n"
            ));
            continue;
        };
        let source = match std::fs::read_to_string(&full) {
            Ok(s) => s,
            Err(e) => {
                out.push_str(&format!("{header}error: {e}\n"));
                continue;
            }
        };

        let tree = match parse_tree(lang, &source) {
            Ok(tree) => tree,
            Err(e) => {
                out.push_str(&format!("{header}error: {e}\n"));
                continue;
            }
        };
        let defs = collect_defs(lang, &tree, &source);

        let mut per_file: Vec<String> = Vec::new();
        for name in &names {
            let matches: Vec<(String, Node<'_>)> = defs
                .iter()
                .filter(|(_, qualified, _)| qualified_ends_with(qualified, name))
                .map(|(_, q, node)| (q.clone(), *node))
                .collect();
            if matches.is_empty() {
                misses.push(format!("{name} not found in {}", full.display()));
                continue;
            }
            for (qualified, node) in matches {
                let start_node = extended_start_node(node);
                let body = &source[start_node.start_byte()..node.end_byte()];
                let start_row = start_node.start_position().row + 1;
                let end_row = node.end_position().row + 1;
                per_file.push(format!(
                    "// {} (lines {}-{})\n{}\n",
                    qualified, start_row, end_row, body
                ));
                found += 1;
            }
        }

        if !per_file.is_empty() {
            out.push_str(&header);
            for block in per_file {
                out.push_str(&block);
            }
        }

        if out.len() > MAX_TOTAL_BYTES {
            out.push_str("\n... output truncated ...\n");
            break 'files;
        }
    }

    let footer = if misses.is_empty() {
        String::new()
    } else {
        format!(
            "\nnot found:\n{}",
            misses
                .iter()
                .map(|m| format!("- {m}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    if found == 0 {
        return ToolOutput::error(format!("{out}{footer}"));
    }
    ToolOutput::text(format!("{out}{footer}"))
        .with_summary(format!("extracted {found} function(s)"))
}

/// `::` normalises to `.` for Rust ergonomics.
fn normalize(name: &str) -> String {
    name.replace("::", ".")
}

/// `request` matches `qualified` when qualified == request or
/// qualified ends with ".request".
fn qualified_ends_with(qualified: &str, request: &str) -> bool {
    qualified == request || qualified.ends_with(&format!(".{request}"))
}

/// Collect (bare, qualified, node) triples for every definition in `tree`.
/// Node lifetimes tie to `tree`, which the caller keeps alive.
fn collect_defs<'a>(
    lang: Lang,
    tree: &'a tree_sitter::Tree,
    source: &'a str,
) -> Vec<(String, String, Node<'a>)> {
    let query = tree_sitter_support::query_for(lang);
    let capture_names = query.capture_names();

    let mut cursor = QueryCursor::new();
    let mut out: Vec<(String, String, Node<'a>)> = Vec::new();
    let mut matches = cursor.matches(query, tree.root_node(), source.as_bytes());

    while let Some(m) = matches.next() {
        let mut name_node: Option<Node<'_>> = None;
        let mut def_node: Option<Node<'_>> = None;
        for cap in m.captures {
            let Some(cname) = capture_names.get(cap.index as usize) else {
                continue;
            };
            if cname.starts_with("name.def") {
                name_node.get_or_insert(cap.node);
            } else if cname.starts_with("definition") {
                def_node.get_or_insert(cap.node);
            }
        }
        let (Some(name), Some(def)) = (name_node, def_node) else {
            continue;
        };
        let bare = source[name.byte_range()].to_string();
        let qualified = qualified_name(source, name, def);
        out.push((bare, qualified, def));
    }
    out
}

fn parse_tree(lang: Lang, source: &str) -> Result<tree_sitter::Tree, String> {
    tree_sitter_support::parse(lang, source)
}

/// Build the dot-qualified name for a definition by walking ancestors.
///
/// Rust: `mod Outer { impl Foo { fn bar } }` → `Outer.Foo.bar`; a method
/// directly in a trait → `Trait.name`. Python/TS: nested classes and
/// methods chain through their class containers.
fn qualified_name(source: &str, name: Node<'_>, def: Node<'_>) -> String {
    let mut segments: Vec<String> = vec![source[name.byte_range()].to_string()];

    // (kind, container name), innermost first.
    let mut chain: Vec<String> = Vec::new();
    let mut ancestor = def.parent();
    while let Some(a) = ancestor {
        let container = match a.kind() {
            "impl_item" => impl_type_name(source, a),
            "trait_item"
            | "mod_item"
            | "class_definition"
            | "class_declaration"
            | "abstract_class_declaration" => a
                .child_by_field_name("name")
                .map(|n| source[n.byte_range()].to_string()),
            _ => None,
        };
        if let Some(c) = container {
            chain.push(c);
        }
        ancestor = a.parent();
    }

    // Insert outermost-first, each before the last segment's predecessor...
    // Container names slot before the innermost segment, outermost at the
    // front: insert at position len-1 in reverse-collection order works
    // because chain is innermost-first.
    for container in chain.into_iter().rev() {
        let at = segments.len().saturating_sub(1);
        segments.insert(at, container);
    }

    segments.join(".")
}

/// The implemented type name of an `impl` block (generics resolve to
/// their inner type identifier).
fn impl_type_name(source: &str, impl_node: Node<'_>) -> Option<String> {
    let ty = impl_node.child_by_field_name("type")?;
    if ty.kind() == "type_identifier" {
        return Some(source[ty.byte_range()].to_string());
    }
    if ty.kind() == "generic_type" {
        let mut cursor = ty.walk();
        for child in ty.children(&mut cursor) {
            if child.kind() == "type_identifier" {
                return Some(source[child.byte_range()].to_string());
            }
        }
    }
    None
}

/// Walk backwards over adjacent attributes / doc comments / decorators
/// attached to `node`, returning the earliest attached sibling.
fn extended_start_node(node: Node<'_>) -> Node<'_> {
    let mut node = node;
    while let Some(prev) = node.prev_sibling() {
        if matches!(
            prev.kind(),
            "attribute_item" | "line_comment" | "block_comment" | "decorator" | "comment"
        ) {
            node = prev;
        } else {
            break;
        }
    }
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::CancelToken;

    fn run(args: serde_json::Value, cwd: &str) -> ToolOutput {
        execute(args, cwd, &CancelToken::new())
    }

    #[test]
    fn rust_bare_and_qualified() {
        let dir = tempfile::tempdir().unwrap();
        let src = r#"
struct Point { x: f64, y: f64 }

impl Point {
    /// Docs for new
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

mod outer {
    pub fn helper() -> u8 {
        42
    }
}

fn free(x: u8) -> u8 { x + 1 }
"#;
        let path = dir.path().join("lib.rs");
        std::fs::write(&path, src).unwrap();
        let p = path.to_str().unwrap();

        // Bare method name matches any `.new`
        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["new"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "{:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("Point.new"), "qualified name: {text}");
        assert!(text.contains("Point { x, y }"), "body: {text}");
        assert!(text.contains("/// Docs for new"), "docs included: {text}");
        assert!(text.contains("(lines 5-8)"), "line numbers: {text}");

        // Qualified dotted path
        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Point.distance"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(out.content[0].text().contains("sqrt"));

        // Rust :: alias
        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Point::distance"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(out.content[0].text().contains("sqrt"), "{:?}", out.content);

        // Mod-prefixed
        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["outer.helper"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        assert!(out.content[0].text().contains("42"), "{:?}", out.content);

        // Bare free function
        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["free_nothing"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(out.is_error);
        let _ = &src;
    }

    #[test]
    fn rust_free_function_by_bare_name() {
        let dir = tempfile::tempdir().unwrap();
        let src = "fn double(x: u8) -> u8 {\n    x * 2\n}\n";
        let path = dir.path().join("a.rs");
        std::fs::write(&path, src).unwrap();
        let out = execute(
            serde_json::json!({"paths": [path.to_str().unwrap()], "function_names": ["double"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "{:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("x * 2"), "{text}");
        assert!(text.contains("double"), "{text}");
    }

    #[test]
    fn python_class_method() {
        let dir = tempfile::tempdir().unwrap();
        let src = "class Widget:\n    def render(self):\n        return self.name\n\ndef top_level():\n    pass\n";
        let path = dir.path().join("w.py");
        std::fs::write(&path, src).unwrap();
        let p = path.to_str().unwrap();

        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Widget.render"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "{:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("return self.name"), "{text}");

        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["top_level"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);

        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Widget.nope"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(out.is_error, "all-miss is error: {:?}", out.content);
        assert!(
            out.content[0].text().contains("not found"),
            "{:?}",
            out.content
        );
    }

    #[test]
    fn typescript_class_method() {
        let dir = tempfile::tempdir().unwrap();
        let src = r#"
export class Circle {
    area(): number {
        return 1;
    }

    private scale(f: number): number {
        return f;
    }
}
"#;
        let path = dir.path().join("a.ts");
        std::fs::write(&path, src).unwrap();
        let p = path.to_str().unwrap();

        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Circle.area"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "{:?}", out.content);
        assert!(out.content[0].text().contains("return 1"));

        let out = execute(
            serde_json::json!({"paths": [p], "function_names": ["Circle.scale"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "{:?}", out.content);
    }

    #[test]
    fn ambiguous_bare_name_returns_all_matches() {
        let dir = tempfile::tempdir().unwrap();
        let src = "class A:\n    def run(self):\n        pass\n\nclass B:\n    def run(self):\n        return 2\n";
        let path = dir.path().join("m.py");
        std::fs::write(&path, src).unwrap();
        let out = execute(
            serde_json::json!({"paths": [path.to_str().unwrap()], "function_names": ["run"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error);
        let text = out.content[0].text();
        assert!(text.contains("A.run"), "{text}");
        assert!(text.contains("B.run"), "{text}");
        assert!(text.contains("return 2"), "{text}");
    }

    #[test]
    fn miss_reported_in_footer_not_error_when_partial() {
        let dir = tempfile::tempdir().unwrap();
        let src = "def exists():\n    pass\n";
        let path = dir.path().join("a.py");
        std::fs::write(&path, src).unwrap();
        let out = execute(
            serde_json::json!({"paths": [path.to_str().unwrap()], "function_names": ["exists", "missing"]}),
            "/tmp",
            &CancelToken::new(),
        );
        assert!(!out.is_error, "partial success: {:?}", out.content);
        let text = out.content[0].text();
        assert!(text.contains("exists"), "{text}");
        assert!(text.contains("not found"), "{text}");
        assert!(text.contains("missing"), "{text}");
    }

    #[test]
    fn args_validation() {
        let cancel = CancelToken::new();
        let out = execute(serde_json::json!({}), "/tmp", &cancel);
        assert!(out.is_error);
        let out = execute(serde_json::json!({"paths": ["a.rs"]}), "/tmp", &cancel);
        assert!(out.is_error);
    }
}
