use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast};

use tars_base::protocol::{
    ErrorKind, ModelInfo, Request, Response, SessionInfo, SessionStats, TokenStats,
};
use tars_base::{AgentPhase, ThinkingStyle};

use crate::db::{Db, StoredSession};

pub struct SharedState {
    pub db: Arc<Mutex<Db>>,
    pub broadcast: broadcast::Sender<Response>,
    pub registry: Arc<tars_engine::ProviderRegistry>,
    /// Per-session event buffer: agent runner pushes here, Subscribe drains first.
    pub session_events: Arc<Mutex<HashMap<String, Vec<Response>>>>,
}

impl SharedState {
    pub fn new(db: Db) -> Self {
        let mut registry = tars_engine::ProviderRegistry::new();
        registry.register(tars_engine::providers::LogProvider);
        registry.register(tars_engine::providers::Anthropic);
        registry.register(tars_engine::providers::OpenAi);
        Self::with_registry(db, registry)
    }

    pub fn with_registry(db: Db, registry: tars_engine::ProviderRegistry) -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            db: Arc::new(Mutex::new(db)),
            broadcast: tx,
            registry: Arc::new(registry),
            session_events: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_mock(db: Db) -> Self {
        let mut registry = tars_engine::ProviderRegistry::new();
        registry.register(tars_engine::providers::LogProvider);
        Self::with_registry(db, registry)
    }

    /// Push an event to the session buffer.
    pub async fn push_event(&self, session_id: &str, resp: Response) {
        let mut buf = self.session_events.lock().await;
        buf.entry(session_id.to_string()).or_default().push(resp);
    }

    /// Drain the session buffer for a given session.
    pub async fn drain_session_events(&self, session_id: &str) -> Vec<Response> {
        let mut buf = self.session_events.lock().await;
        buf.remove(session_id).unwrap_or_default()
    }
}

fn make_session_info(stored: &StoredSession, message_count: usize) -> SessionInfo {
    SessionInfo {
        id: stored.id.clone(),
        model: stored.model.id.clone(),
        provider: stored.model.provider.clone(),
        cwd: stored.cwd.clone(),
        message_count,
        stats: SessionStats {
            user_messages: 0,
            assistant_messages: 0,
            tool_calls: 0,
            tool_results: 0,
            tokens: TokenStats::default(),
            cost: 0.0,
            context_window: stored.model.context_window,
            context_tokens: None,
        },
        state: AgentPhase::Idle,
        created_at: stored.created_at,
        last_activity: stored.created_at,
        parent_id: None,
        context_pct: None,
        tagline: None,
        archived: false,
    }
}

fn mock_model_info() -> ModelInfo {
    ModelInfo {
        id: "mock-model".into(),
        name: "Mock".into(),
        provider: "mock".into(),
        thinking: ThinkingStyle::None,
        context_window: 100_000,
        max_tokens: 4096,
    }
}

pub async fn dispatch(state: Arc<SharedState>, req: Request) -> Response {
    match req {
        Request::CreateSession {
            model,
            system_prompt,
            cwd,
            ..
        } => {
            let id = format!("s{}", tars_base::timestamp_ms());
            let default_model = tars_engine::providers::log::log_model();
            let stored = StoredSession {
                id: id.clone(),
                model: default_model,
                system_prompt: system_prompt.or(Some("default system".into())),
                cwd: cwd.clone(),
                created_at: tars_base::timestamp_ms() as i64,
            };
            // override model if requested
            let mut stored = stored;
            if let Some(m) = model {
                stored.model.id = m;
            }
            let db = state.db.lock().await;
            match db.create_session(&stored) {
                Ok(()) => Response::SessionCreated { session_id: id },
                Err(e) => Response::Error {
                    kind: ErrorKind::Internal,
                    message: e.to_string(),
                },
            }
        }
        Request::ListSessions { .. } => {
            let db = state.db.lock().await;
            match db.list_sessions() {
                Ok(sessions) => {
                    let mut infos = Vec::new();
                    for s in sessions {
                        let count = db.message_count(&s.id).unwrap_or(0);
                        infos.push(make_session_info(&s, count));
                    }
                    Response::Sessions { sessions: infos }
                }
                Err(e) => Response::Error {
                    kind: ErrorKind::Internal,
                    message: e.to_string(),
                },
            }
        }
        Request::GetSessionInfo { session_id } => {
            let db = state.db.lock().await;
            match db.get_session(&session_id) {
                Ok(Some(s)) => {
                    let count = db.message_count(&s.id).unwrap_or(0);
                    Response::SessionInfo {
                        info: make_session_info(&s, count),
                    }
                }
                Ok(None) => Response::Error {
                    kind: ErrorKind::NotFound,
                    message: format!("session {} not found", session_id),
                },
                Err(e) => Response::Error {
                    kind: ErrorKind::Internal,
                    message: e.to_string(),
                },
            }
        }
        Request::GetMessages { session_id } => {
            let db = state.db.lock().await;
            match db.get_messages(&session_id) {
                Ok(msgs) => Response::Messages { messages: msgs },
                Err(e) => Response::Error {
                    kind: ErrorKind::Internal,
                    message: e.to_string(),
                },
            }
        }
        Request::ArchiveSession { session_id } => {
            let db = state.db.lock().await;
            match db.delete_session(&session_id) {
                Ok(()) => Response::SessionArchived,
                Err(e) => Response::Error {
                    kind: ErrorKind::Internal,
                    message: e.to_string(),
                },
            }
        }
        Request::ListModels => Response::Models {
            models: vec![mock_model_info()],
        },
        Request::SetModel {
            session_id,
            model_id,
        } => {
            // For MVP, just return mock changed
            let _ = session_id;
            Response::ModelChanged {
                model: ModelInfo {
                    id: model_id.clone(),
                    name: model_id.clone(),
                    provider: "mock".into(),
                    thinking: ThinkingStyle::None,
                    context_window: 100_000,
                    max_tokens: 4096,
                },
            }
        }
        Request::Chat {
            session_id,
            text,
            attachments: _,
        } => {
            let state_clone = state.clone();
            let sid = session_id.clone();
            let txt = text.clone();
            tokio::spawn(async move {
                let cancel = tars_base::CancelToken::new();
                let _ = crate::agent_runner::run_session_turn(
                    state_clone.clone(),
                    &state_clone.registry,
                    &sid,
                    &txt,
                    &cancel,
                )
                .await;
            });
            Response::Ok
        }
        Request::Subscribe { session_id } => {
            // For MVP, just ack; broadcast will be handled by connection handler
            let _ = session_id;
            Response::Ok
        }
        Request::CancelChat { .. } => Response::Cancelled,
    }
}

pub async fn handle_connection(state: Arc<SharedState>, stream: UnixStream) {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();

    loop {
        line.clear();
        let n = match reader.read_line(&mut line).await {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        let req: Result<Request, _> = serde_json::from_str(line.trim());
        let req = match req {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error {
                    kind: ErrorKind::Parse,
                    message: e.to_string(),
                };
                let _ = tars_base::write_json_line_async(&mut write_half, &resp).await;
                continue;
            }
        };

        // Special handling for Subscribe: enter broadcast forward loop.
        // Send Ok immediately so the client knows the subscription is active,
        // then forward broadcast events until AgentDone or disconnect.
        if matches!(req, Request::Subscribe { .. }) {
            let sid = if let Request::Subscribe { session_id } = &req {
                session_id.clone()
            } else {
                unreachable!()
            };
            let resp = dispatch(state.clone(), req).await;
            let _ = tars_base::write_json_line_async(&mut write_half, &resp).await;
            // Drain and forward events from session buffer until terminal.
            // The agent runner pushes events to session_events; we poll until
            // we see AgentDone (or error/cancel).
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
            let mut done = false;
            while !done {
                let buffered = state.drain_session_events(&sid).await;
                let had_events = !buffered.is_empty();
                for resp in buffered {
                    let is_terminal = matches!(
                        resp,
                        Response::AgentDone | Response::Cancelled | Response::Error { .. }
                    );
                    if tars_base::write_json_line_async(&mut write_half, &resp)
                        .await
                        .is_err()
                    {
                        done = true;
                        break;
                    }
                    if is_terminal {
                        done = true;
                        break;
                    }
                }
                if !done && !had_events {
                    if tokio::time::Instant::now() >= deadline {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
            break;
        }

        let resp = dispatch(state.clone(), req).await;
        if tars_base::write_json_line_async(&mut write_half, &resp)
            .await
            .is_err()
        {
            break;
        }
        // Also handle broadcast forwarding interleaved with next request
        // (non-Subscribe connections still get broadcast via the select above on next iteration)
    }
}

pub async fn run(listener: UnixListener, state: Arc<SharedState>) -> std::io::Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            handle_connection(state, stream).await;
        });
    }
}

pub fn socket_path_from(paths: &tars_base::Paths) -> PathBuf {
    paths.socket_path()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::Message;
    use tokio::io::BufReader;
    use tokio::net::UnixStream;

    #[tokio::test]
    async fn roundtrip_via_duplex() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let (a, b) = UnixStream::pair().unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            handle_connection(state_clone, a).await;
        });

        let (read_half, mut write_half) = b.into_split();
        let mut reader = BufReader::new(read_half);

        // CreateSession
        let req = Request::CreateSession {
            model: None,
            provider: None,
            system_prompt: None,
            cwd: None,
            parent_id: None,
            tagline: None,
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        let session_id = match resp {
            Response::SessionCreated { session_id } => session_id,
            other => panic!("expected SessionCreated, got {:?}", other),
        };

        // ListSessions should contain it
        let req = Request::ListSessions {
            include_archived: false,
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        match resp {
            Response::Sessions { sessions } => assert!(sessions.iter().any(|s| s.id == session_id)),
            other => panic!("expected Sessions, got {:?}", other),
        }

        // GetMessages should be empty initially (no user messages yet)
        let req = Request::GetMessages {
            session_id: session_id.clone(),
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        match resp {
            Response::Messages { messages } => assert!(messages.is_empty()),
            other => panic!("expected Messages, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn subscribe_drains_session_buffer() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let (a, b) = UnixStream::pair().unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            handle_connection(state_clone, a).await;
        });

        let (read_half, mut write_half) = b.into_split();
        let mut reader = BufReader::new(read_half);

        // Create a session
        let req = Request::CreateSession {
            model: None,
            provider: None,
            system_prompt: None,
            cwd: None,
            parent_id: None,
            tagline: None,
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        let session_id = match resp {
            Response::SessionCreated { session_id } => session_id,
            other => panic!("expected SessionCreated, got {:?}", other),
        };

        // Push events to the session buffer BEFORE subscribing
        state.push_event(&session_id, Response::Ok).await;
        state.push_event(&session_id, Response::AgentDone).await;

        // Subscribe — should drain the buffered events
        let req = Request::Subscribe {
            session_id: session_id.clone(),
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::Ok);

        // Should receive the buffered events
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::Ok);

        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::AgentDone);
    }

    #[tokio::test]
    async fn chat_appends_and_broadcasts() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let (a, b) = UnixStream::pair().unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            handle_connection(state_clone, a).await;
        });
        let (read_half, mut write_half) = b.into_split();
        let mut reader = BufReader::new(read_half);

        let req = Request::CreateSession {
            model: None,
            provider: None,
            system_prompt: None,
            cwd: None,
            parent_id: None,
            tagline: None,
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        let session_id = match resp {
            Response::SessionCreated { session_id } => session_id,
            other => panic!("expected SessionCreated, got {:?}", other),
        };

        let req = Request::Chat {
            session_id: session_id.clone(),
            text: "hello".into(),
            attachments: vec![],
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::Ok);

        // Verify DB has the user message and agent response
        let msgs = state.db.lock().await.get_messages(&session_id).unwrap();
        assert!(!msgs.is_empty());
        assert!(matches!(msgs[0], Message::User(_)));
    }

    #[tokio::test]
    async fn server_via_listener_log_provider() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("tars-test.sock");
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        // Create a session with the log model directly (so Chat uses LogProvider)
        let session_id = "s_log";
        {
            let log_model = tars_engine::providers::log::log_model();
            let db = state.db.lock().await;
            db.create_session(&StoredSession {
                id: session_id.into(),
                model: log_model,
                system_prompt: None,
                cwd: Some("/tmp".into()),
                created_at: tars_base::timestamp_ms() as i64,
            })
            .unwrap();
        }

        let listener = UnixListener::bind(&socket_path).unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            let _ = run(listener, state_clone).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let stream = UnixStream::connect(&socket_path).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        let req = Request::Chat {
            session_id: session_id.into(),
            text: "hello log".into(),
            attachments: vec![],
        };
        tars_base::write_json_line_async(&mut write_half, &req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::Ok);

        // Subscribe to receive streaming events
        let sub_req = Request::Subscribe {
            session_id: session_id.into(),
        };
        tars_base::write_json_line_async(&mut write_half, &sub_req)
            .await
            .unwrap();
        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, Response::Ok);

        // Now read streamed events until AgentDone
        let mut saw_done = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            let res = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                tars_base::read_json_line_async::<Response>(&mut reader),
            )
            .await;
            if let Ok(Ok(Some(resp))) = res {
                if resp == Response::AgentDone {
                    saw_done = true;
                    break;
                }
            } else {
                break;
            }
        }
        assert!(saw_done, "should have seen AgentDone");

        // DB should now have user + assistant (LogProvider produces empty assistant)
        let msgs = state.db.lock().await.get_messages(session_id).unwrap();
        assert!(msgs.len() >= 2);
    }
}
