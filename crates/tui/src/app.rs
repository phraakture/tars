//! TUI state machine.
//!
//! Pure state and transitions, no terminal I/O, so everything here is
//! unit-testable. Rendering lives in `ui`, the event loop in `lib`.

use tars_base::protocol::SessionInfo;
use tars_base::{Message, StreamEvent, ToolResultContent, UserContent};

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

/// One renderable line group in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptLine {
    User(String),
    Assistant(String),
    Tool { name: String, is_error: bool },
    Summary(String),
    Info(String),
}

/// Flatten persisted messages into transcript lines.
pub fn build_transcript(messages: &[Message]) -> Vec<TranscriptLine> {
    let mut lines = Vec::new();
    for msg in messages {
        match msg {
            Message::User(u) => {
                let text: String = u
                    .content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text(t) => t.text.as_str(),
                        UserContent::Image(_) => "[image]",
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.trim().is_empty() {
                    lines.push(TranscriptLine::User(text));
                }
            }
            Message::Assistant(a) => {
                let text = a.text();
                if !text.trim().is_empty() {
                    lines.push(TranscriptLine::Assistant(text));
                }
            }
            Message::ToolResult(tr) => {
                let text: String = tr
                    .content
                    .iter()
                    .map(|c| match c {
                        ToolResultContent::Text(t) => t.text.clone(),
                        ToolResultContent::Image(_) => "[image]".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let _ = text;
                lines.push(TranscriptLine::Tool {
                    name: tr.tool_name.clone(),
                    is_error: tr.is_error,
                });
            }
            Message::CompactionSummary(cs) => {
                lines.push(TranscriptLine::Summary(cs.summary.clone()));
            }
            Message::Info(i) => {
                lines.push(TranscriptLine::Info(i.text.clone()));
            }
        }
    }
    lines
}

// ---------------------------------------------------------------------------
// Chat screen state
// ---------------------------------------------------------------------------

/// Join a snapshot's thinking blocks (the current thinking text).
fn thinking_text(msg: &tars_base::AssistantMessage) -> String {
    msg.content
        .iter()
        .filter_map(|c| match c {
            tars_base::AssistantContent::Thinking(t) => Some(t.thinking.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[derive(Debug, Default, Clone)]
pub struct ChatState {
    pub session_id: String,
    pub model: String,
    pub lines: Vec<TranscriptLine>,
    /// Current input buffer.
    pub input: String,
    /// Text of the assistant message currently streaming in.
    pub streaming: Option<String>,
    /// Extended-thinking text currently streaming in (rendered dim).
    pub thinking: Option<String>,
    /// True while an agent turn is running.
    pub busy: bool,
    /// Lines scrolled up from the bottom (0 = pinned to bottom).
    pub scroll: u16,
    /// Last error surfaced from the server.
    pub error: Option<String>,
}

impl ChatState {
    pub fn new(session_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            model: model.into(),
            lines: Vec::new(),
            input: String::new(),
            streaming: None,
            thinking: None,
            busy: false,
            scroll: 0,
            error: None,
        }
    }

    /// Replace the transcript from a fresh `GetMessages` result.
    pub fn set_history(&mut self, messages: &[Message]) {
        self.lines = build_transcript(messages);
        self.scroll = 0;
    }

    /// Apply a streaming event. Snapshots (`partial`) are authoritative, so
    /// this is a simple assignment per event rather than incremental bookkeeping.
    pub fn apply_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Start { partial }
            | StreamEvent::TextStart { partial, .. }
            | StreamEvent::TextDelta { partial, .. }
            | StreamEvent::TextEnd { partial, .. } => {
                self.streaming = Some(partial.text());
            }
            StreamEvent::ThinkingStart { partial, .. }
            | StreamEvent::ThinkingDelta { partial, .. } => {
                self.streaming = None;
                self.thinking = Some(thinking_text(partial));
            }
            StreamEvent::ThinkingEnd { .. } => {
                self.thinking = None;
            }
            StreamEvent::ToolcallStart { .. } | StreamEvent::ToolcallDelta { .. } => {}
            StreamEvent::ToolcallEnd { tool_call, .. } => {
                self.flush_streaming();
                self.lines.push(TranscriptLine::Tool {
                    name: tool_call.name.clone(),
                    is_error: false,
                });
            }
            StreamEvent::ToolOutputDelta { .. } => {}
            StreamEvent::ToolResult {
                tool_name,
                is_error,
                ..
            } => {
                self.lines.push(TranscriptLine::Tool {
                    name: tool_name.clone(),
                    is_error: *is_error,
                });
            }
            StreamEvent::Done { message, .. } => {
                self.flush_streaming();
                let text = message.text();
                if !text.trim().is_empty() {
                    self.lines.push(TranscriptLine::Assistant(text));
                }
            }
            StreamEvent::Error { error, .. } => {
                self.flush_streaming();
                self.error = Some(error.text());
                if self.error.as_deref().map(str::is_empty).unwrap_or(true) {
                    self.error = Some("agent turn failed".into());
                }
            }
            StreamEvent::Status { message } => {
                self.lines.push(TranscriptLine::Info(message.clone()));
            }
            StreamEvent::SteerMessage { .. } | StreamEvent::Phase { .. } => {}
        }
    }

    /// Move any in-flight streaming text into a transcript line.
    pub fn flush_streaming(&mut self) {
        if let Some(text) = self.streaming.take() {
            if !text.trim().is_empty() {
                self.lines.push(TranscriptLine::Assistant(text));
            }
        }
    }

    /// Agent turn finished. Streaming text is finalized by a subsequent
    /// history reload; clear the live view state.
    pub fn finish_turn(&mut self) {
        self.streaming = None;
        self.thinking = None;
        self.busy = false;
        self.scroll = 0;
    }

    /// Scroll by `delta` lines (positive = scroll up).
    pub fn scroll_by(&mut self, delta: u16) {
        self.scroll = self.scroll.saturating_add(delta);
    }
}

// ---------------------------------------------------------------------------
// Session picker state
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct PickerState {
    pub sessions: Vec<SessionInfo>,
    pub selected: usize,
    pub loading: bool,
    pub error: Option<String>,
}

impl PickerState {
    pub fn move_selection(&mut self, delta: i32) {
        if self.sessions.is_empty() {
            return;
        }
        let len = self.sessions.len() as i32;
        let next = (self.selected as i32 + delta).rem_euclid(len);
        self.selected = next as usize;
    }

    /// Short summary line for one session row.
    pub fn row_label(session: &SessionInfo) -> String {
        let tagline = session.tagline.as_deref().unwrap_or("");
        format!(
            "{}  {}  {} msgs  {}",
            session.id, session.model, session.message_count, tagline
        )
    }
}

// ---------------------------------------------------------------------------
// App + actions
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Screen {
    Picker,
    Chat(Box<ChatState>),
}

#[derive(Debug)]
pub struct App {
    pub running: bool,
    pub screen: Screen,
    pub picker: PickerState,
}

/// Outcome of a key on the picker screen (drives async work in the event loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerAction {
    None,
    Quit,
    Refresh,
    NewSession,
    Open(usize),
}

/// Outcome of a key on the chat screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatAction {
    None,
    Quit,
    Back,
    Cancel,
    Send(String),
}

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            running: true,
            screen: Screen::Picker,
            picker: PickerState::default(),
        }
    }

    pub fn handle_picker_key(&mut self, key: KeyEvent) -> PickerAction {
        if key.kind != crossterm::event::KeyEventKind::Press {
            return PickerAction::None;
        }
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => PickerAction::Quit,
            (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => PickerAction::Quit,
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
                self.picker.move_selection(-1);
                PickerAction::None
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
                self.picker.move_selection(1);
                PickerAction::None
            }
            (KeyCode::Char('r'), _) => PickerAction::Refresh,
            (KeyCode::Char('n'), _) => PickerAction::NewSession,
            (KeyCode::Enter, _) => {
                if self.picker.sessions.is_empty() {
                    PickerAction::None
                } else {
                    PickerAction::Open(self.picker.selected)
                }
            }
            _ => PickerAction::None,
        }
    }

    pub fn handle_chat_key(&mut self, key: KeyEvent) -> ChatAction {
        if key.kind != crossterm::event::KeyEventKind::Press {
            return ChatAction::None;
        }
        let Screen::Chat(chat) = &mut self.screen else {
            return ChatAction::None;
        };
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => ChatAction::Quit,
            (KeyCode::Esc, _) => {
                if chat.busy {
                    ChatAction::Cancel
                } else {
                    ChatAction::Back
                }
            }
            (KeyCode::Enter, _) => {
                let text = chat.input.trim().to_string();
                if !text.is_empty() && !chat.busy {
                    chat.input.clear();
                    ChatAction::Send(text)
                } else {
                    ChatAction::None
                }
            }
            (KeyCode::Backspace, _) => {
                chat.input.pop();
                ChatAction::None
            }
            (KeyCode::PageUp, _) => {
                chat.scroll_by(10);
                ChatAction::None
            }
            (KeyCode::PageDown, _) => {
                chat.scroll = chat.scroll.saturating_sub(10);
                ChatAction::None
            }
            (KeyCode::Up, _) => {
                chat.scroll_by(3);
                ChatAction::None
            }
            (KeyCode::Down, _) => {
                chat.scroll = chat.scroll.saturating_sub(3);
                ChatAction::None
            }
            (KeyCode::Char(c), mods) if mods.is_empty() || mods == KeyModifiers::SHIFT => {
                chat.input.push(c);
                ChatAction::None
            }
            _ => ChatAction::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use serde_json::json;
    use tars_base::{AssistantContent, AssistantMessage, UserMessage};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    fn user_msg(text: &str) -> Message {
        Message::User(UserMessage::text(text))
    }

    fn assistant_msg(text: &str) -> Message {
        let mut a = AssistantMessage::empty("test", "test", "test-model");
        a.content
            .push(AssistantContent::Text(tars_base::TextContent {
                text: text.into(),
                text_signature: None,
            }));
        Message::Assistant(a)
    }

    // -- transcript building --

    #[test]
    fn transcript_user_and_assistant() {
        let lines = build_transcript(&[user_msg("hi"), assistant_msg("hello")]);
        assert_eq!(lines[0], TranscriptLine::User("hi".into()));
        assert_eq!(lines[1], TranscriptLine::Assistant("hello".into()));
    }

    #[test]
    fn transcript_tool_result_carries_name_and_error() {
        let tr = tool_result("tc1", "bash", "ok");
        let lines = build_transcript(&[Message::ToolResult(tr)]);
        match &lines[0] {
            TranscriptLine::Tool { name, is_error } => {
                assert_eq!(name, "bash");
                assert!(!is_error);
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn transcript_skips_empty_assistant() {
        let a = AssistantMessage::empty("t", "t", "t");
        let lines = build_transcript(&[Message::Assistant(a)]);
        assert!(lines.is_empty());
    }

    #[test]
    fn transcript_summary_and_info() {
        let msgs = vec![
            Message::CompactionSummary(tars_base::CompactionSummaryMessage {
                summary: "earlier work".into(),
                tokens_before: 100,
                timestamp: 0,
            }),
            Message::Info(tars_base::InfoMessage {
                text: "note".into(),
                timestamp: 0,
            }),
        ];
        let lines = build_transcript(&msgs);
        assert_eq!(lines[0], TranscriptLine::Summary("earlier work".into()));
        assert_eq!(lines[1], TranscriptLine::Info("note".into()));
    }

    // -- chat state: streaming events --

    #[test]
    fn text_delta_accumulates_via_snapshot() {
        let mut chat = ChatState::new("s1", "m1");

        let mut partial = AssistantMessage::empty("t", "t", "t");
        partial
            .content
            .push(AssistantContent::Text(tars_base::TextContent {
                text: "hel".into(),
                text_signature: None,
            }));
        chat.apply_event(&StreamEvent::Start {
            partial: partial.clone(),
        });
        assert_eq!(chat.streaming.as_deref(), Some("hel"));

        partial.content[0] = AssistantContent::Text(tars_base::TextContent {
            text: "hello".into(),
            text_signature: None,
        });
        chat.apply_event(&StreamEvent::TextDelta {
            content_index: 0,
            delta: "lo".into(),
            partial: partial.clone(),
        });
        assert_eq!(chat.streaming.as_deref(), Some("hello"));
    }

    #[test]
    fn done_flushes_assistant_text() {
        let mut chat = ChatState::new("s1", "m1");
        let mut msg = AssistantMessage::empty("t", "t", "t");
        msg.content
            .push(AssistantContent::Text(tars_base::TextContent {
                text: "final answer".into(),
                text_signature: None,
            }));
        chat.busy = true;
        chat.apply_event(&StreamEvent::Done {
            reason: tars_base::StopReason::Stop,
            message: msg,
        });
        chat.finish_turn();
        assert!(chat.streaming.is_none());
        assert!(!chat.busy);
        assert_eq!(
            chat.lines.last(),
            Some(&TranscriptLine::Assistant("final answer".into()))
        );
    }

    #[test]
    fn toolcall_end_flushes_streaming_then_tool() {
        let mut chat = ChatState::new("s1", "m1");
        let mut partial = AssistantMessage::empty("t", "t", "t");
        partial
            .content
            .push(AssistantContent::Text(tars_base::TextContent {
                text: "running tool".into(),
                text_signature: None,
            }));
        chat.apply_event(&StreamEvent::TextDelta {
            content_index: 0,
            delta: String::new(),
            partial: partial.clone(),
        });

        chat.apply_event(&StreamEvent::ToolcallEnd {
            content_index: 1,
            tool_call: tars_base::ToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: json!({}),
            },
            partial,
        });

        assert!(chat.streaming.is_none());
        assert_eq!(
            chat.lines[0],
            TranscriptLine::Assistant("running tool".into())
        );
        assert_eq!(
            chat.lines[1],
            TranscriptLine::Tool {
                name: "bash".into(),
                is_error: false
            }
        );
    }

    #[test]
    fn error_event_records_message() {
        let mut chat = ChatState::new("s1", "m1");
        let mut err = AssistantMessage::empty("t", "t", "t");
        err.content
            .push(AssistantContent::Text(tars_base::TextContent {
                text: "boom".into(),
                text_signature: None,
            }));
        chat.apply_event(&StreamEvent::Error {
            reason: tars_base::StopReason::Error,
            error: err,
        });
        assert_eq!(chat.error.as_deref(), Some("boom"));
    }

    // -- picker navigation --

    #[test]
    fn picker_selection_wraps() {
        let mut app = App::new();
        app.picker.sessions = vec![session_info("a"), session_info("b"), session_info("c")];
        app.handle_picker_key(key(KeyCode::Char('j')));
        assert_eq!(app.picker.selected, 1);
        app.handle_picker_key(key(KeyCode::Up));
        assert_eq!(app.picker.selected, 0);
        app.handle_picker_key(key(KeyCode::Up));
        assert_eq!(app.picker.selected, 2, "wraps to end");
        app.handle_picker_key(key(KeyCode::Char('j')));
        assert_eq!(app.picker.selected, 0, "wraps to start");
        app.handle_picker_key(key(KeyCode::Down));
        assert_eq!(app.picker.selected, 1);
    }

    #[test]
    fn picker_open_requires_sessions() {
        let mut app = App::new();
        assert_eq!(
            app.handle_picker_key(key(KeyCode::Enter)),
            PickerAction::None
        );
        app.picker.sessions = vec![session_info("a")];
        assert_eq!(
            app.handle_picker_key(key(KeyCode::Enter)),
            PickerAction::Open(0)
        );
    }

    #[test]
    fn picker_quit_keys() {
        let mut app = App::new();
        assert_eq!(
            app.handle_picker_key(key(KeyCode::Char('q'))),
            PickerAction::Quit
        );
        assert_eq!(app.handle_picker_key(key(KeyCode::Esc)), PickerAction::Quit);
        assert_eq!(app.handle_picker_key(ctrl_c()), PickerAction::Quit);
    }

    // -- chat keys --

    fn app_in_chat() -> (App, &'static str) {
        let mut app = App::new();
        app.screen = Screen::Chat(Box::new(ChatState::new("s1", "m1")));
        (app, "s1")
    }

    #[test]
    fn chat_input_and_send() {
        let (mut app, _) = app_in_chat();
        for c in "hi there".chars() {
            app.handle_chat_key(key(KeyCode::Char(c)));
        }
        let Screen::Chat(chat) = &app.screen else {
            panic!("expected chat screen")
        };
        assert_eq!(chat.input, "hi there");

        let action = app.handle_chat_key(key(KeyCode::Enter));
        assert_eq!(action, ChatAction::Send("hi there".into()));
        let Screen::Chat(chat) = &app.screen else {
            panic!("expected chat screen")
        };
        assert!(chat.input.is_empty());
    }

    #[test]
    fn chat_send_blocked_while_busy() {
        let (mut app, _) = app_in_chat();
        if let Screen::Chat(chat) = &mut app.screen {
            chat.input = "blocked".into();
            chat.busy = true;
        }
        assert_eq!(app.handle_chat_key(key(KeyCode::Enter)), ChatAction::None);
    }

    #[test]
    fn chat_backspace_and_back() {
        let (mut app, _) = app_in_chat();
        app.handle_chat_key(key(KeyCode::Char('x')));
        app.handle_chat_key(key(KeyCode::Backspace));
        let Screen::Chat(chat) = &app.screen else {
            panic!("expected chat screen")
        };
        assert!(chat.input.is_empty());

        // Back returns to picker when idle
        app.handle_chat_key(key(KeyCode::Char('y')));
        assert_eq!(app.handle_chat_key(key(KeyCode::Esc)), ChatAction::Back);

        // While busy, Esc cancels the turn instead.
        if let Screen::Chat(chat) = &mut app.screen {
            chat.busy = true;
        }
        assert_eq!(app.handle_chat_key(key(KeyCode::Esc)), ChatAction::Cancel);
    }

    #[test]
    fn chat_scrolling() {
        let (mut app, _) = app_in_chat();
        app.handle_chat_key(key(KeyCode::PageUp));
        app.handle_chat_key(key(KeyCode::Up));
        let Screen::Chat(chat) = &app.screen else {
            panic!("expected chat screen")
        };
        assert_eq!(chat.scroll, 13);

        app.handle_chat_key(key(KeyCode::PageDown));
        app.handle_chat_key(key(KeyCode::PageDown));
        let Screen::Chat(chat) = &app.screen else {
            panic!("expected chat screen")
        };
        assert_eq!(chat.scroll, 0, "clamped at bottom");
    }

    // -- picker row label --

    fn session_info(id: &str) -> SessionInfo {
        SessionInfo {
            id: id.into(),
            model: "test-model".into(),
            provider: "mock".into(),
            cwd: Some("/tmp".into()),
            message_count: 3,
            stats: tars_base::protocol::SessionStats {
                user_messages: 1,
                assistant_messages: 1,
                tool_calls: 0,
                tool_results: 0,
                tokens: tars_base::protocol::TokenStats::default(),
                cost: 0.0,
                context_window: 100_000,
                context_tokens: None,
            },
            state: tars_base::AgentPhase::Idle,
            created_at: 0,
            last_activity: 0,
            parent_id: None,
            context_pct: None,
            project_name: Some("tars".into()),
            tagline: Some("working on tests".into()),
            archived: false,
        }
    }

    #[test]
    fn row_label_includes_fields() {
        let label = PickerState::row_label(&session_info("abc"));
        assert!(label.contains("abc"));
        assert!(label.contains("test-model"));
        assert!(label.contains("3 msgs"));
        assert!(label.contains("working on tests"));
    }

    // -- misc test helpers --

    fn tool_result(id: &str, name: &str, text: &str) -> tars_base::ToolResultMessage {
        tars_base::ToolResultMessage {
            tool_call_id: id.into(),
            tool_name: name.into(),
            content: vec![tars_base::ToolResultContent::Text(tars_base::TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
            duration_ms: None,
            summary: None,
            post_persist_actions: Vec::new(),
        }
    }

    #[test]
    fn user_message_renders() {
        let u = UserMessage::text("hello world");
        let lines = build_transcript(&[Message::User(u)]);
        assert_eq!(lines, vec![TranscriptLine::User("hello world".into())]);
    }
}
