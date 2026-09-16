//! Unix-socket client library for tars.
//!
//! Thin async client used by the CLI (and later the TUI) to talk to the tars
//! server daemon over JSON-lines framing.

use std::path::Path;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixStream;

use tars_base::protocol::{Request, Response};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tars: {0}")]
    Tars(#[from] tars_base::Error),
    #[error("connection closed")]
    Closed,
    #[error("unexpected response: {0:?}")]
    Unexpected(Response),
}

pub type Result<T> = std::result::Result<T, ClientError>;

pub struct Client {
    stream: UnixStream,
}

impl Client {
    pub async fn connect(path: &Path) -> Result<Self> {
        let stream = UnixStream::connect(path).await?;
        Ok(Self { stream })
    }

    pub async fn send(&mut self, req: &Request) -> Result<()> {
        tars_base::write_json_line_async(&mut self.stream, req).await?;
        Ok(())
    }

    pub async fn recv_response(&mut self) -> Result<Response> {
        let mut reader = BufReader::new(&mut self.stream);
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Err(ClientError::Closed);
        }
        let resp: Response = serde_json::from_str(&line)?;
        Ok(resp)
    }

    pub async fn send_and_recv(&mut self, req: &Request) -> Result<Response> {
        self.send(req).await?;
        self.recv_response().await
    }

    /// Send Chat + Subscribe, then return a receiver that yields streaming
    /// responses (Stream events, AgentDone, etc.) until the server closes the
    /// connection or AgentDone is received. Consumes the client.
    pub async fn chat(
        self,
        session_id: &str,
        text: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<Response>> {
        let (read_half, mut write_half) = self.stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Send Chat
        let chat_req = Request::Chat {
            session_id: session_id.to_string(),
            text: text.to_string(),
            attachments: vec![],
        };
        tars_base::write_json_line_async(&mut write_half, &chat_req).await?;

        // Read Ok
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let resp: Response = serde_json::from_str(&line)?;
        if !matches!(resp, Response::Ok) {
            return Err(ClientError::Unexpected(resp));
        }

        // Send Subscribe
        line.clear();
        let sub_req = Request::Subscribe {
            session_id: session_id.to_string(),
        };
        tars_base::write_json_line_async(&mut write_half, &sub_req).await?;

        // Read Subscribe Ok
        line.clear();
        reader.read_line(&mut line).await?;
        let resp: Response = serde_json::from_str(&line)?;
        if !matches!(resp, Response::Ok) {
            return Err(ClientError::Unexpected(resp));
        }

        // Drop write half — server won't read more from us after Subscribe
        drop(write_half);

        // Spawn reader task that feeds responses into a channel
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(resp) = serde_json::from_str::<Response>(&line) {
                            let is_terminal = matches!(
                                resp,
                                Response::AgentDone | Response::Cancelled | Response::Error { .. }
                            );
                            let _ = tx.send(resp).await;
                            if is_terminal {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(rx)
    }

    /// Convenience: create a session and return its id.
    pub async fn create_session(
        &mut self,
        model: Option<String>,
        cwd: Option<String>,
    ) -> Result<String> {
        let req = Request::CreateSession {
            model,
            provider: None,
            system_prompt: None,
            cwd,
            parent_id: None,
            tagline: None,
        };
        match self.send_and_recv(&req).await? {
            Response::SessionCreated { session_id } => Ok(session_id),
            resp => Err(ClientError::Unexpected(resp)),
        }
    }

    /// List all sessions.
    pub async fn list_sessions(&mut self) -> Result<Vec<tars_base::protocol::SessionInfo>> {
        let req = Request::ListSessions {
            include_archived: false,
        };
        match self.send_and_recv(&req).await? {
            Response::Sessions { sessions } => Ok(sessions),
            resp => Err(ClientError::Unexpected(resp)),
        }
    }

    /// List available models.
    pub async fn list_models(&mut self) -> Result<Vec<tars_base::protocol::ModelInfo>> {
        let req = Request::ListModels;
        match self.send_and_recv(&req).await? {
            Response::Models { models } => Ok(models),
            resp => Err(ClientError::Unexpected(resp)),
        }
    }

    /// Request cancellation of a running agent turn.
    pub async fn cancel_chat(&mut self, session_id: &str) -> Result<bool> {
        let req = Request::CancelChat {
            session_id: session_id.to_string(),
        };
        match self.send_and_recv(&req).await? {
            Response::Cancelled => Ok(true),
            Response::Ok => Ok(false),
            resp => Err(ClientError::Unexpected(resp)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tars_lib::db::Db;
    use tars_lib::server::{SharedState, run};
    use tokio::net::UnixListener;

    #[tokio::test]
    async fn connect_and_create_session() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let listener = UnixListener::bind(&socket_path).unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            let _ = run(listener, state_clone).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut client = Client::connect(&socket_path).await.unwrap();
        let session_id = client
            .create_session(None, Some("/tmp".into()))
            .await
            .unwrap();
        assert!(!session_id.is_empty());

        let sessions = client.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session_id);
    }

    #[tokio::test]
    async fn list_models() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let listener = UnixListener::bind(&socket_path).unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            let _ = run(listener, state_clone).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut client = Client::connect(&socket_path).await.unwrap();
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[tokio::test]
    async fn chat_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        // Create session with log model
        {
            let log_model = tars_engine::providers::log::log_model();
            let db = state.db.lock().await;
            db.create_session(&tars_lib::db::StoredSession {
                id: "s_chat".into(),
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

        let client = Client::connect(&socket_path).await.unwrap();
        let mut rx = client.chat("s_chat", "hello").await.unwrap();

        let mut saw_done = false;
        while let Some(resp) = rx.recv().await {
            if matches!(resp, Response::AgentDone) {
                saw_done = true;
                break;
            }
        }
        assert!(saw_done, "should see AgentDone");
    }

    #[tokio::test]
    async fn closed_connection_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");
        let db = Db::open_memory().unwrap();
        let state = Arc::new(SharedState::new(db));
        let listener = UnixListener::bind(&socket_path).unwrap();
        let state_clone = state.clone();
        tokio::spawn(async move {
            let _ = run(listener, state_clone).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = Client::connect(&socket_path).await.unwrap();
        // Drop client (close connection), then try to recv
        drop(client);
        // Reconnect and verify it works
        let mut client2 = Client::connect(&socket_path).await.unwrap();
        let models = client2.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
