use std::path::PathBuf;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast};

use tars_base::protocol::{
    ErrorKind, ModelInfo, Request, Response, SessionInfo, SessionStats, TokenStats,
};
use tars_base::{AgentPhase, Message, Model, ModelCost, ThinkingStyle};

use crate::db::{Db, StoredSession};

pub struct SharedState {
    pub db: Arc<Mutex<Db>>,
    pub broadcast: broadcast::Sender<Response>,
}

impl SharedState {
    pub fn new(db: Db) -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            db: Arc::new(Mutex::new(db)),
            broadcast: tx,
        }
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

fn test_model() -> Model {
    Model {
        id: "mock-model".into(),
        name: "Mock".into(),
        api: "mock".into(),
        provider: "mock".into(),
        base_url: "http://mock".into(),
        thinking: ThinkingStyle::None,
        cost: ModelCost::default(),
        context_window: 100_000,
        max_tokens: 4096,
        headers: Default::default(),
    }
}

pub async fn dispatch(state: &SharedState, req: Request) -> Response {
    match req {
        Request::CreateSession {
            model,
            system_prompt,
            cwd,
            ..
        } => {
            let id = format!("s{}", tars_base::timestamp_ms());
            let stored = StoredSession {
                id: id.clone(),
                model: test_model(),
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
            attachments,
        } => {
            // MVP: append user message to DB and broadcast, return Ok
            let mut content = vec![tars_base::UserContent::Text(tars_base::TextContent {
                text: text.clone(),
                text_signature: None,
            })];
            for att in attachments {
                content.push(att.to_user_content());
            }
            let msg = Message::User(tars_base::UserMessage {
                content,
                timestamp: tars_base::timestamp_ms(),
            });
            let db = state.db.lock().await;
            let _ = db.append_message(&session_id, &msg);
            let _ = state
                .broadcast
                .send(Response::UserMessage { text: text.clone() });
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
    let mut broadcast_rx = state.broadcast.subscribe();

    loop {
        line.clear();
        let n = tokio::select! {
            res = reader.read_line(&mut line) => {
                match res {
                    Ok(n) => n,
                    Err(_) => break,
                }
            }
            res = broadcast_rx.recv() => {
                if let Ok(resp) = res {
                    let _ = tars_base::write_json_line_async(&mut write_half, &resp).await;
                }
                continue;
            }
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

        // Special handling for Subscribe: enter broadcast forward loop
        if matches!(req, Request::Subscribe { .. }) {
            let resp = dispatch(&state, req).await;
            let _ = tars_base::write_json_line_async(&mut write_half, &resp).await;
            // Now forward broadcast events until client disconnects
            loop {
                tokio::select! {
                    res = reader.read_line(&mut line) => {
                        match res {
                            Ok(0) => break,
                            Ok(_) => {
                                // For MVP, ignore further requests while subscribed
                                line.clear();
                                continue;
                            }
                            Err(_) => break,
                        }
                    }
                    res = broadcast_rx.recv() => {
                        if let Ok(resp) = res {
                            if tars_base::write_json_line_async(&mut write_half, &resp).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            break;
        }

        let resp = dispatch(&state, req).await;
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
    async fn broadcast_via_subscribe() {
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let (a, b) = UnixStream::pair().unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            handle_connection(state_clone, a).await;
        });

        let (read_half, mut write_half) = b.into_split();
        let mut reader = BufReader::new(read_half);

        // First create a session to subscribe to
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

        // Subscribe
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

        // Broadcast a Stream event via SharedState
        let event = Response::Stream {
            event: Box::new(tars_base::StreamEvent::Status {
                message: "hello broadcast".into(),
            }),
        };
        let _ = state.broadcast.send(event.clone());

        let resp: Response = tars_base::read_json_line_async(&mut reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resp, event);
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

        // Verify DB has the user message
        let msgs = state.db.lock().await.get_messages(&session_id).unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], Message::User(_)));
    }
}
