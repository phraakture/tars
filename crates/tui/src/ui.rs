//! Ratatui rendering for the TUI screens.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{App, ChatState, Screen};

const HINT_PICKER: &str = "j/k navigate  enter open  n new  r refresh  q quit";

pub fn draw(app: &App, frame: &mut Frame) {
    match &app.screen {
        Screen::Picker => draw_picker(app, frame),
        Screen::Chat(chat) => draw_chat(chat, frame),
    }
}

fn draw_picker(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" tars sessions ");

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows: Vec<ListItem> = if app.picker.sessions.is_empty() {
        let hint = if app.picker.loading {
            "loading sessions..."
        } else {
            "no sessions yet, press n to create one"
        };
        vec![ListItem::new(Line::styled(
            hint,
            ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::DIM),
        ))]
    } else {
        app.picker
            .sessions
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let label = crate::app::PickerState::row_label(s);
                if i == app.picker.selected {
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            "> ",
                            ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
                        ),
                        Span::raw(label),
                    ]))
                } else {
                    ListItem::new(Line::from(format!("  {label}")))
                }
            })
            .collect()
    };

    let list = List::new(rows).highlight_style(ratatui::style::Style::default());
    let mut state = ListState::default().with_selected(Some(app.picker.selected));
    frame.render_stateful_widget(list, inner, &mut state);

    // Error banner and hints at the bottom
    let footer_area = ratatui::layout::Rect {
        y: area.height.saturating_sub(1),
        height: 1,
        ..area
    };
    let footer = match &app.picker.error {
        Some(e) => Line::styled(
            e.clone(),
            ratatui::style::Style::default().fg(ratatui::style::Color::Red),
        ),
        None => Line::raw(HINT_PICKER).dim(),
    };
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn draw_chat(chat: &ChatState, frame: &mut Frame) {
    let area = frame.area();
    let [transcript_area, input_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_transcript(chat, frame, transcript_area);
    draw_input(chat, frame, input_area);
    draw_status(chat, frame, status_area);
}

fn draw_transcript(chat: &ChatState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let title = format!(" tars chat  [{}] ", chat.model);
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    for line in &chat.lines {
        match line {
            crate::app::TranscriptLine::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "you  ",
                        ratatui::style::Style::default()
                            .fg(ratatui::style::Color::Cyan)
                            .bold(),
                    ),
                    Span::raw(text),
                ]));
            }
            crate::app::TranscriptLine::Assistant(text) => {
                lines.push(Line::from(text.clone()));
            }
            crate::app::TranscriptLine::Tool { name, is_error } => {
                let (mark, color) = if *is_error {
                    ("[tool failed] ", ratatui::style::Color::Red)
                } else {
                    ("[tool] ", ratatui::style::Color::Green)
                };
                lines.push(Line::from(vec![Span::styled(
                    format!("{mark}{name}"),
                    ratatui::style::Style::default().fg(color),
                )]));
            }
            crate::app::TranscriptLine::Summary(_) => {
                lines.push(Line::styled(
                    "-- context compacted, earlier messages summarized --",
                    ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::DIM),
                ));
            }
            crate::app::TranscriptLine::Info(text) => {
                lines.push(Line::styled(
                    text.clone(),
                    ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::DIM),
                ));
            }
        }
    }

    if let Some(streaming) = &chat.streaming {
        lines.push(Line::from(streaming.clone()));
        lines.push(Line::from(Span::styled(
            "▌",
            ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
        )));
    }

    if let Some(err) = &chat.error {
        lines.push(Line::styled(
            format!("error: {err}"),
            ratatui::style::Style::default().fg(ratatui::style::Color::Red),
        ));
    }

    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((chat.scroll.max(total_lines_overflow(chat, inner.height)), 0));
    frame.render_widget(para, inner);
}

/// Compute the scroll offset clamped so we never scroll past the content.
fn total_lines_overflow(chat: &ChatState, visible: u16) -> u16 {
    // Approximate: one render line per transcript entry plus streaming lines.
    let mut count = chat.lines.len() as u16;
    if chat.streaming.is_some() {
        count += 2;
    }
    if chat.error.is_some() {
        count += 1;
    }
    count.saturating_sub(visible)
}

fn draw_input(chat: &ChatState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let (title, style) = if chat.busy {
        (
            " working... (esc disabled while busy) ",
            ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
        )
    } else {
        (
            " message  (enter send  esc back) ",
            ratatui::style::Style::default(),
        )
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, style));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let para = Paragraph::new(Line::from(chat.input.as_str()));
    frame.render_widget(para, inner);
}

fn draw_status(chat: &ChatState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let scroll_note = if chat.scroll > 0 {
        format!("  (scrolled {} lines, pgdn to follow)", chat.scroll)
    } else {
        String::new()
    };
    let line = Line::from(vec![
        Span::raw("pgup/pgdn scroll  "),
        Span::raw(format!("session {}", chat.session_id)),
        Span::raw(scroll_note),
    ]);
    frame.render_widget(Paragraph::new(line.dim()), area);
}
