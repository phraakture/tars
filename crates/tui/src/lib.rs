//! Terminal UI for tars.
//!
//! Connects to the server via [`tars_lib::daemon::connect_or_start`], shows a
//! session picker, then a chat view with live streaming. The state machine in
//! `app` is pure and unit-tested; this module owns the terminal and the async
//! event loop only.

mod app;
mod ui;

use std::io::stdout;

use crossterm::event::{Event, EventStream, KeyEvent};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use tokio::sync::mpsc::Receiver;

use tars_base::protocol::{Request, Response};
use tars_client::Client;

use app::{App, ChatAction, ChatState, PickerAction, Screen};

/// Run the TUI against the standard tars paths (starts the server if needed).
pub async fn run() -> anyhow::Result<()> {
    let paths = tars_base::Paths::detect();
    let client = tars_lib::daemon::connect_or_start(&paths).await?;
    let socket_path = paths.socket_path();
    run_with(client, &socket_path).await
}

/// Run the TUI with an already-connected client (used by tests and the CLI).
pub async fn run_with(client: Client, socket_path: &std::path::Path) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(client, socket_path, &mut terminal).await;

    // Always restore the terminal, even on error.
    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Outcome of handling one streamed response.
enum StreamOutcome {
    Continue,
    TurnFinished,
}

async fn event_loop(
    mut control: Client,
    socket_path: &std::path::Path,
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
) -> anyhow::Result<()> {
    let mut app = App::new();
    let mut events = EventStream::new();
    let mut stream_rx: Option<Receiver<Response>> = None;

    // Initial session load
    refresh_sessions(&mut control, &mut app).await;

    loop {
        terminal.draw(|f| ui::draw(&app, f))?;

        tokio::select! {
            maybe_event = events.next() => {
                let Some(event) = maybe_event else { break };
                match event? {
                    Event::Key(key) => {
                        match handle_key(&mut app, key) {
                            LoopAction::None => {}
                            LoopAction::Quit => {
                                app.running = false;
                                break;
                            }
                            LoopAction::RefreshSessions => {
                                refresh_sessions(&mut control, &mut app).await;
                            }
                            LoopAction::NewSession => {
                                match create_session(&mut control).await {
                                    Ok(info) => {
                                        let chat = ChatState::new(&info.id, info.model.clone());
                                        app.screen = Screen::Chat(Box::new(chat));
                                        load_history(&mut control, &mut app).await;
                                    }
                                    Err(e) => app.picker.error = Some(e.to_string()),
                                }
                            }
                            LoopAction::Open(index) => {
                                if let Some(info) = app.picker.sessions.get(index).cloned() {
                                    let chat = ChatState::new(&info.id, info.model.clone());
                                    app.screen = Screen::Chat(Box::new(chat));
                                    load_history(&mut control, &mut app).await;
                                }
                            }
                            LoopAction::Back => {
                                app.screen = Screen::Picker;
                                refresh_sessions(&mut control, &mut app).await;
                            }
                            LoopAction::Send(text) => {
                                let Screen::Chat(chat) = &mut app.screen else { unreachable!() };
                                let session_id = chat.session_id.clone();
                                chat.busy = true;
                                chat.error = None;
                                // New connection per turn: Client::chat consumes it.
                                match Client::connect(socket_path).await {
                                    Ok(c) => match c.chat(&session_id, &text).await {
                                        Ok(rx) => stream_rx = Some(rx),
                                        Err(e) => {
                                            chat.busy = false;
                                            chat.error = Some(e.to_string());
                                        }
                                    },
                                    Err(e) => {
                                        chat.busy = false;
                                        chat.error = Some(e.to_string());
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some(resp) = async {
                match stream_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            }, if stream_rx.is_some() => {
                let outcome = apply_response(&mut app, resp);
                match outcome {
                    StreamOutcome::Continue => {}
                    StreamOutcome::TurnFinished => {
                        stream_rx = None;
                        // Reload authoritative history from the DB.
                        load_history(&mut control, &mut app).await;
                    }
                }
            }
        }
    }
    Ok(())
}

enum LoopAction {
    None,
    Quit,
    RefreshSessions,
    NewSession,
    Open(usize),
    Back,
    Send(String),
}

fn handle_key(app: &mut App, key: KeyEvent) -> LoopAction {
    match &mut app.screen {
        Screen::Picker => match app.handle_picker_key(key) {
            PickerAction::None => LoopAction::None,
            PickerAction::Quit => LoopAction::Quit,
            PickerAction::Refresh => LoopAction::RefreshSessions,
            PickerAction::NewSession => LoopAction::NewSession,
            PickerAction::Open(index) => LoopAction::Open(index),
        },
        Screen::Chat(_) => match app.handle_chat_key(key) {
            ChatAction::None => LoopAction::None,
            ChatAction::Quit => LoopAction::Quit,
            ChatAction::Back => LoopAction::Back,
            ChatAction::Send(text) => LoopAction::Send(text),
        },
    }
}

fn apply_response(app: &mut App, resp: Response) -> StreamOutcome {
    let Screen::Chat(chat) = &mut app.screen else {
        return StreamOutcome::Continue;
    };
    match resp {
        Response::Stream { event } => {
            chat.apply_event(&event);
            StreamOutcome::Continue
        }
        Response::AgentDone | Response::Cancelled => {
            chat.finish_turn();
            StreamOutcome::TurnFinished
        }
        Response::Error { message, .. } => {
            chat.busy = false;
            chat.streaming = None;
            chat.error = Some(message);
            StreamOutcome::TurnFinished
        }
        Response::UserMessage { .. } | Response::Ok | Response::SessionCreated { .. } => {
            StreamOutcome::Continue
        }
        _ => StreamOutcome::Continue,
    }
}

async fn refresh_sessions(control: &mut Client, app: &mut App) {
    app.picker.loading = true;
    app.picker.error = None;
    match control.list_sessions().await {
        Ok(sessions) => {
            app.picker.sessions = sessions;
            if app.picker.selected >= app.picker.sessions.len() {
                app.picker.selected = app.picker.sessions.len().saturating_sub(1);
            }
        }
        Err(e) => app.picker.error = Some(e.to_string()),
    }
    app.picker.loading = false;
}

async fn create_session(control: &mut Client) -> anyhow::Result<tars_base::protocol::SessionInfo> {
    let session_id = control.create_session(None, None).await?;
    Ok(tars_base::protocol::SessionInfo {
        id: session_id,
        model: String::new(),
        provider: String::new(),
        cwd: None,
        message_count: 0,
        stats: tars_base::protocol::SessionStats {
            user_messages: 0,
            assistant_messages: 0,
            tool_calls: 0,
            tool_results: 0,
            tokens: tars_base::protocol::TokenStats::default(),
            cost: 0.0,
            context_window: 0,
            context_tokens: None,
        },
        state: tars_base::AgentPhase::Idle,
        created_at: 0,
        last_activity: 0,
        parent_id: None,
        context_pct: None,
        tagline: None,
        archived: false,
    })
}

async fn load_history(control: &mut Client, app: &mut App) {
    let Screen::Chat(chat) = &mut app.screen else {
        return;
    };
    let req = Request::GetMessages {
        session_id: chat.session_id.clone(),
    };
    match control.send_and_recv(&req).await {
        Ok(Response::Messages { messages }) => chat.set_history(&messages),
        Ok(Response::Error { message, .. }) => chat.error = Some(message),
        Ok(_) => {}
        Err(e) => chat.error = Some(e.to_string()),
    }
}

/// Force a flush of module boundaries before tests.
#[cfg(test)]
mod stream_tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn quit_on_ctrl_c_from_picker() {
        let mut app = App::new();
        let action =
            app.handle_picker_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert_eq!(action, PickerAction::Quit);
    }
}
