//! Markdown rendering for the TUI transcript.
//!
//! A compact pulldown-cmark → ratatui renderer covering what LLM output
//! actually contains: headings, paragraphs, bold/italic/strikethrough,
//! inline code, fenced code blocks, lists, block quotes, and horizontal
//! rules. Unknown constructs degrade to plain text.
//!
//! Output is a flat `Vec<Line<'static>>`; the caller wraps them in a
//! `Paragraph` widget with `Wrap`.

use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render markdown `text` into styled lines. Blocks are separated by one
/// blank line; a trailing blank line is trimmed.
pub fn to_lines(text: &str) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut style_stack: Vec<Style> = Vec::new();
    let mut in_code_block = false;

    let current_style = |stack: &Vec<Style>| stack.last().copied().unwrap_or_default();
    let flush = |spans: &mut Vec<Span<'static>>, lines: &mut Vec<Line<'static>>| {
        if !spans.is_empty() {
            lines.push(Line::from(std::mem::take(spans)));
        }
    };

    for event in Parser::new(text) {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph if !lines.is_empty() && !lines.last().unwrap().spans.is_empty() => {
                    // Blank separator between blocks (not before the first).
                    lines.push(Line::from(Vec::new()));
                }
                Tag::Paragraph => {}
                Tag::Heading { level, .. } => {
                    // Headings render bold; h1/h2 additionally cyan.
                    let mut base = Style::default().add_modifier(Modifier::BOLD);
                    if level as usize <= 2 {
                        base = base.fg(Color::Cyan);
                    }
                    style_stack.push(base);
                }
                Tag::Strong => {
                    style_stack.push(current_style(&style_stack).add_modifier(Modifier::BOLD))
                }
                Tag::Emphasis => {
                    style_stack.push(current_style(&style_stack).add_modifier(Modifier::ITALIC))
                }
                Tag::Strikethrough => style_stack
                    .push(current_style(&style_stack).add_modifier(Modifier::CROSSED_OUT)),
                Tag::CodeBlock(_) => {
                    in_code_block = true;
                    style_stack.push(Style::default().fg(Color::Green));
                }
                Tag::BlockQuote(_) => spans.push(Span::styled(
                    "> ",
                    Style::default().add_modifier(Modifier::DIM),
                )),
                Tag::Item => {
                    flush(&mut spans, &mut lines);
                    spans.push(Span::styled(
                        "• ",
                        Style::default().add_modifier(Modifier::DIM),
                    ));
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Heading(_) => {
                    style_stack.pop();
                    flush(&mut spans, &mut lines);
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    style_stack.pop();
                    flush(&mut spans, &mut lines);
                }
                TagEnd::Paragraph => flush(&mut spans, &mut lines),
                TagEnd::Item => flush(&mut spans, &mut lines),
                TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {
                    style_stack.pop();
                }
                _ => {}
            },
            Event::Text(t) => {
                let style = current_style(&style_stack);
                // Inside a code block the text carries raw newlines; emit
                // one line per source line so the block stays contiguous.
                if in_code_block {
                    let mut first = true;
                    for seg in t.split('\n') {
                        if !first {
                            flush(&mut spans, &mut lines);
                        }
                        if !seg.is_empty() {
                            spans.push(Span::styled(seg.to_string(), style));
                        }
                        first = false;
                    }
                } else {
                    spans.push(Span::styled(t.to_string(), style));
                }
            }
            Event::Code(c) => spans.push(Span::styled(
                c.to_string(),
                current_style(&style_stack).fg(Color::Yellow),
            )),
            Event::SoftBreak | Event::HardBreak => flush(&mut spans, &mut lines),
            Event::Rule => {
                flush(&mut spans, &mut lines);
                lines.push(Line::styled(
                    "────────────────────────────────────────",
                    Style::default().add_modifier(Modifier::DIM),
                ));
            }
            _ => {}
        }
    }
    flush(&mut spans, &mut lines);

    // Trim trailing blank lines.
    while lines.last().map(|l| l.spans.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.clone()).collect())
            .collect()
    }

    #[test]
    fn paragraphs_and_separators() {
        let lines = to_lines("first\nparagraph\n\nsecond");
        let texts = plain(&lines);
        // Soft breaks become line breaks; paragraphs separated by one blank.
        assert_eq!(texts, vec!["first", "paragraph", "", "second"], "{texts:?}");
    }

    #[test]
    fn heading_text_is_bold() {
        let lines = to_lines("# Hello");
        assert_eq!(plain(&lines)[0], "Hello");
        assert!(
            lines[0].spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }

    #[test]
    fn inline_code_styled() {
        let lines = to_lines("run `cargo test` now");
        let spans = &lines[0].spans;
        assert!(spans.iter().any(|s| s.content == "cargo test"), "{spans:?}");
    }

    #[test]
    fn code_block_lines() {
        let text = "```rust\nfn main() {}\nlet x = 1;\n```";
        let lines = to_lines(text);
        let texts = plain(&lines);
        assert!(
            texts.iter().any(|l| l.contains("fn main() {}")),
            "{texts:?}"
        );
        assert!(texts.iter().any(|l| l.contains("let x = 1;")), "{texts:?}");
        // Code lines must not carry a paragraph separator between them.
        let main_idx = texts.iter().position(|l| l.contains("fn main")).unwrap();
        let let_idx = texts.iter().position(|l| l.contains("let x")).unwrap();
        assert_eq!(let_idx - main_idx, 1, "code block stays contiguous");
    }

    #[test]
    fn bold_and_italic_no_literal_markers() {
        let all: String = plain(&to_lines("this is **bold** and *italic*")).join("");
        assert!(all.contains("bold"), "{all}");
        assert!(all.contains("italic"), "{all}");
        assert!(!all.contains("**"), "no literal markers: {all}");
    }

    #[test]
    fn list_items_bulletted() {
        let lines = to_lines("- one\n- two\n");
        let texts = plain(&lines);
        assert_eq!(texts, vec!["• one", "• two"], "{texts:?}");
    }

    #[test]
    fn blockquote_prefix() {
        let lines = to_lines("> quoted");
        let texts = plain(&lines);
        assert_eq!(texts[0], "> quoted", "{texts:?}");
    }

    #[test]
    fn rule_renders() {
        let lines = to_lines("---");
        assert!(
            lines
                .iter()
                .any(|l| l.spans.iter().any(|s| s.content.contains("────"))),
            "{lines:?}"
        );
    }

    #[test]
    fn plain_text_is_just_text() {
        let lines = to_lines("plain line");
        assert_eq!(plain(&lines), vec!["plain line"]);
    }

    #[test]
    fn heading_does_not_leak_bold() {
        // After the heading, following text must not be bold.
        let lines = to_lines("# Head\n\nafter");
        let after = &lines[2].spans[0];
        assert!(
            !after.style.add_modifier.contains(Modifier::BOLD),
            "style leaked: {:?}",
            after.style
        );
    }
}
