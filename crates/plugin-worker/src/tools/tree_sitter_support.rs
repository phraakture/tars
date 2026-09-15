//! Shared tree-sitter plumbing for the source-analysis tools
//! (`get_file_skeleton`, `get_function`).
//!
//! Every language gets one tag-style query that emits two capture kinds:
//!
//! - `name.def.<kind>` — the identifier node naming the definition. Its
//!   row is the dedupe key for skeleton output; its text is the bare name.
//! - `definition.<kind>` — the whole definition node (function, class,
//!   impl, …) whose text covers the body; ancestor walks and slicing key
//!   off it.
//!
//! Queries are hand-written per language (not ports of any external
//! `.scm` set) and kept deliberately small: declarations and methods only,
//! no reference captures. Compiled once per language per process via
//! `OnceLock`.

use std::cell::RefCell;
use std::sync::OnceLock;

use tree_sitter::{Language, Parser, Query, Tree};

/// Languages with skeleton / function-extraction support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
}

impl Lang {
    /// Extension dispatch. `None` means "no grammar for this extension".
    pub(crate) fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "js" | "mjs" | "cjs" | "jsx" => Some(Self::JavaScript),
            "ts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            _ => None,
        }
    }

    pub(crate) fn ts_language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        }
    }
}

fn compile(source: &str, lang: Lang) -> Query {
    Query::new(&lang.ts_language(), source)
        .unwrap_or_else(|e| panic!("tree-sitter query failed to compile: {e}"))
}

// -- queries -------------------------------------------------------------

const RUST_QUERY_SRC: &str = r#"
(function_item name: (identifier) @name.def.function) @definition.function
(function_signature_item name: (identifier) @name.def.function) @definition.function
(struct_item name: (type_identifier) @name.def.class) @definition.class
(enum_item name: (type_identifier) @name.def.class) @definition.class
(union_item name: (type_identifier) @name.def.class) @definition.class
(trait_item name: (type_identifier) @name.def.interface) @definition.interface
(type_item name: (type_identifier) @name.def.type) @definition.type
(mod_item name: (identifier) @name.def.module) @definition.module
(impl_item type: (type_identifier) @name.def.class) @definition.impl
"#;

const PYTHON_QUERY_SRC: &str = r#"
(function_definition name: (identifier) @name.def.function) @definition.function
(class_definition name: (identifier) @name.def.class) @definition.class
"#;

const JAVASCRIPT_QUERY_SRC: &str = r#"
(function_declaration name: (identifier) @name.def.function) @definition.function
(class_declaration name: (identifier) @name.def.class) @definition.class
(method_definition name: (property_identifier) @name.def.function) @definition.function
(lexical_declaration
  (variable_declarator
    name: (identifier) @name.def.function
    value: [(arrow_function) (function_expression)])) @definition.function
(generator_function_declaration name: (identifier) @name.def.function) @definition.function
"#;

const TYPESCRIPT_QUERY_SRC: &str = r#"
(function_declaration name: (identifier) @name.def.function) @definition.function
(class_declaration name: (type_identifier) @name.def.class) @definition.class
(abstract_class_declaration name: (type_identifier) @name.def.class) @definition.class
(interface_declaration name: (type_identifier) @name.def.interface) @definition.interface
(method_definition name: (property_identifier) @name.def.function) @definition.function
(method_signature name: (property_identifier) @name.def.function) @definition.function
(lexical_declaration
  (variable_declarator
    name: (identifier) @name.def.function
    value: [(arrow_function) (function_expression)])) @definition.function
(function_signature name: (identifier) @name.def.function) @definition.function
(generator_function_declaration name: (identifier) @name.def.function) @definition.function
"#;

static RUST_QUERY: OnceLock<Query> = OnceLock::new();
static PYTHON_QUERY: OnceLock<Query> = OnceLock::new();
static JS_QUERY: OnceLock<Query> = OnceLock::new();
static TS_QUERY: OnceLock<Query> = OnceLock::new();

fn compiled(slot: &'static OnceLock<Query>, source: &'static str, lang: Lang) -> &'static Query {
    slot.get_or_init(|| compile(source, lang))
}

pub(crate) fn query_for(lang: Lang) -> &'static Query {
    match lang {
        Lang::Rust => compiled(&RUST_QUERY, RUST_QUERY_SRC, lang),
        Lang::Python => compiled(&PYTHON_QUERY, PYTHON_QUERY_SRC, lang),
        Lang::JavaScript => compiled(&JS_QUERY, JAVASCRIPT_QUERY_SRC, lang),
        Lang::TypeScript | Lang::Tsx => compiled(&TS_QUERY, TYPESCRIPT_QUERY_SRC, lang),
    }
}

// -- parsing -------------------------------------------------------------

thread_local! {
    static PARSER: RefCell<Option<(Lang, Parser)>> = const { RefCell::new(None) };
}

/// Parse `source` with `lang`'s grammar. Reuses a thread-local `Parser`
/// when the language matches (`Parser` is `!Sync`; thread-local reuse is
/// the documented happy path), otherwise rebuilds for the new grammar.
pub(crate) fn parse(lang: Lang, source: &str) -> Result<Tree, String> {
    PARSER.with_borrow_mut(|slot| {
        if slot.as_ref().map(|(l, _)| *l) != Some(lang) {
            let mut p = Parser::new();
            p.set_language(&lang.ts_language())
                .map_err(|e| format!("failed to load grammar: {e}"))?;
            *slot = Some((lang, p));
        }
        slot.as_mut()
            .and_then(|(_, p)| p.parse(source, None))
            .ok_or_else(|| "parse failed".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_dispatch() {
        assert_eq!(Lang::from_extension("rs"), Some(Lang::Rust));
        assert_eq!(Lang::from_extension("py"), Some(Lang::Python));
        assert_eq!(Lang::from_extension("pyi"), Some(Lang::Python));
        assert_eq!(Lang::from_extension("tsx"), Some(Lang::Tsx));
        assert_eq!(Lang::from_extension("md"), None);
    }

    #[test]
    fn rust_parses() {
        let tree = parse(Lang::Rust, "fn main() {}").unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn python_parses() {
        let tree = parse(Lang::Python, "def f():\n    pass\n").unwrap();
        assert!(!tree.root_node().has_error());
    }
}
