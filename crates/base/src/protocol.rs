//! JSON-lines wire protocol between client and server (unix socket).
//!
//! Every line on the socket is one serialized `Request` (client → server) or
//! one `Response` (server → client). Tagged enums keep the framing trivial:
//!
//! ```
//! use tars_base::protocol::{Request, Response};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Client → server: one line
//! let request = Request::Chat {
//!     session_id: "abc123".into(),
//!     text: "hello".into(),
//!     attachments: Vec::new(),
//! };
//! let mut line = serde_json::to_string(&request)?;
//! line.push('\n');
//! assert!(line.starts_with(r#"{"type":"chat""#));
//!
//! // Server → client: one line
//! let response = Response::Ok;
//! let line = serde_json::to_string(&response)?;
//! assert_eq!(line, r#"{"type":"ok"}"#);
//!
//! // Round-trip: a response parsed back from the wire
//! let back: Response = serde_json::from_str(&line)?;
//! assert_eq!(response, back);
//! # Ok(())
//! # }
//! ```
//!
//! Versioning is done the pragmatic way: optional fields are
//! `#[serde(default, skip_serializing_if = "Option::is_none")]` and new enum
//! variants are appended, so old clients ignore unknown fields and new clients
//! handle unknown variants through an exhaustive `_` arm rather than a wire
//! version number.

use serde::{Deserialize, Serialize};

use crate::types::{AgentPhase, Message, StreamEvent, ThinkingStyle, UserContent};

// ---------------------------------------------------------------------------
// Client → Server
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Send a chat message in a session (starts or resumes the agent loop).
    Chat {
        session_id: String,
        text: String,
        /// Optional attachments (images for now).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<ChatAttachment>,
    },
    /// Create a new session. All fields optional; the server fills model and
    /// cwd from defaults when omitted.
    CreateSession {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        system_prompt: Option<String>,
        /// Working directory for tool execution.
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        /// Parent session ID (for child sessions).
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        /// Short description of the session's task.
        #[serde(skip_serializing_if = "Option::is_none")]
        tagline: Option<String>,
    },
    /// Info about a single session.
    GetSessionInfo { session_id: String },
    /// List sessions.
    ListSessions {
        /// Include archived sessions in the listing.
        #[serde(default)]
        include_archived: bool,
    },
    /// Archive a session.
    ArchiveSession { session_id: String },
    /// Message history for a session.
    GetMessages { session_id: String },
    /// Subscribe to live events on a session (for multi-client). The
    /// connection stays open and receives `Stream` / `AgentDone` /
    /// `Cancelled` / `UserMessage` responses.
    Subscribe { session_id: String },
    /// Cancel an in-progress chat (agent loop) for a session.
    CancelChat { session_id: String },
    /// List available models.
    ListModels,
    /// Change model for a session.
    SetModel {
        session_id: String,
        model_id: String,
    },
}

/// Attachments to a `Request::Chat` message.
///
/// Today only images are supported; the open enum lets us add more kinds
/// without bumping the protocol shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatAttachment {
    /// An image. `data` is base64-encoded bytes; `mime_type` is a MIME type
    /// accepted by the server validator (`image/png`, `image/jpeg`,
    /// `image/gif`, `image/webp`).
    Image { data: String, mime_type: String },
}

impl ChatAttachment {
    /// Convert this attachment into an engine `UserContent` block.
    ///
    /// Pure structural mapping; validation (decoded byte length, allowed
    /// MIME) belongs to the caller.
    pub fn to_user_content(&self) -> UserContent {
        match self {
            ChatAttachment::Image { data, mime_type } => {
                UserContent::Image(crate::types::ImageContent {
                    data: data.clone(),
                    mime_type: mime_type.clone(),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Server → Client
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// The session was created.
    SessionCreated { session_id: String },
    /// Info about a single session.
    SessionInfo { info: SessionInfo },
    /// List of sessions.
    Sessions { sessions: Vec<SessionInfo> },
    /// The session was archived.
    SessionArchived,
    /// Available models.
    Models { models: Vec<ModelInfo> },
    /// Model changed for a session.
    ModelChanged { model: ModelInfo },
    /// Streaming event from the LLM. Boxed to keep `Response` small on the
    /// hot path where `Stream` dominates the socket traffic.
    Stream { event: Box<StreamEvent> },
    /// Message history for a session.
    Messages { messages: Vec<Message> },
    /// A user message was sent (broadcast to subscribers).
    UserMessage { text: String },
    /// Agent loop completed (all turns done).
    AgentDone,
    /// Agent loop was cancelled by the user.
    Cancelled,
    /// Generic success ack.
    Ok,
    /// Structured failure. `kind` lets clients branch on the error class
    /// without string matching; `message` is for humans and logs.
    Error { kind: ErrorKind, message: String },
}

/// Wire-level classification of a failed request.
///
/// Mirrors the discriminant of the server's internal [`crate::Error`] so
/// clients can programmatically retry (rate limits, timeouts), surface auth
/// problems, and distinguish "not found" from "invalid input" — without
/// parsing the message text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ErrorKind {
    NoProvider,
    NoApiKey,
    Http {
        status: u16,
    },
    RateLimited {
        retry_after: Option<u64>,
    },
    Timeout,
    Auth,
    ContextOverflow,
    Cancelled,
    /// The addressed session/model/object does not exist.
    NotFound,
    /// The request was malformed or supplied invalid arguments.
    Invalid,
    Parse,
    ChannelClosed,
    Internal,
}

impl ErrorKind {
    /// Best-effort classification of an internal [`crate::Error`] into a wire
    /// class. Unrecognised variants classify as [`Self::Internal`]; callers
    /// wanting a richer kind should match on the concrete `Error` first.
    pub fn from_error(err: &crate::Error) -> Self {
        match err {
            crate::Error::NoProvider(_) => Self::NoProvider,
            crate::Error::NoApiKey(_) => Self::NoApiKey,
            crate::Error::Http { status, .. } => Self::Http { status: *status },
            crate::Error::RateLimited { retry_after, .. } => Self::RateLimited {
                retry_after: *retry_after,
            },
            crate::Error::Timeout(_) => Self::Timeout,
            crate::Error::Auth(_) => Self::Auth,
            crate::Error::ContextOverflow => Self::ContextOverflow,
            crate::Error::Cancelled => Self::Cancelled,
            crate::Error::Parse(_) => Self::Parse,
            crate::Error::ChannelClosed => Self::ChannelClosed,
            crate::Error::Io(_) | crate::Error::Json(_) | crate::Error::Internal(_) => {
                Self::Internal
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Session metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub model: String,
    pub provider: String,
    pub cwd: Option<String>,
    pub message_count: usize,
    pub stats: SessionStats,
    /// Current agent phase. `Idle` when no turn is running.
    #[serde(default)]
    pub state: AgentPhase,
    /// Unix timestamp (seconds) of session creation.
    pub created_at: i64,
    /// Unix timestamp (seconds) of last activity (last message or creation).
    pub last_activity: i64,
    /// Parent session ID (None for root sessions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Context window usage as a percentage (0–100), if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_pct: Option<f64>,
    /// Short description of what this session is working on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tagline: Option<String>,
    #[serde(default)]
    pub archived: bool,
}

/// Cumulative per-session usage statistics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionStats {
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub tool_calls: usize,
    pub tool_results: usize,
    pub tokens: TokenStats,
    /// Cost in USD.
    pub cost: f64,
    /// Context window size from the model.
    pub context_window: u64,
    /// Estimated context usage from the last assistant response (input tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenStats {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

impl TokenStats {
    pub fn total(&self) -> u64 {
        self.input + self.output + self.cache_read + self.cache_write
    }
}

/// Model metadata surfaced in `ListModels` / `SetModel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub thinking: ThinkingStyle,
    pub context_window: u64,
    pub max_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_session_info() -> SessionInfo {
        SessionInfo {
            id: "s1".into(),
            model: "test-model".into(),
            provider: "test".into(),
            cwd: Some("/home/user/project".into()),
            message_count: 3,
            stats: SessionStats::default(),
            state: AgentPhase::Idle,
            created_at: 1000,
            last_activity: 2000,
            parent_id: None,
            context_pct: None,
            tagline: None,
            archived: false,
        }
    }

    #[test]
    fn request_chat_roundtrip() {
        let req = Request::Chat {
            session_id: "s1".into(),
            text: "hello".into(),
            attachments: vec![ChatAttachment::Image {
                data: "AAAA".into(),
                mime_type: "image/png".into(),
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.starts_with(r#"{"type":"chat""#));
        assert!(json.contains(r#""type":"image""#));
        let back: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn request_chat_empty_attachments_not_serialized() {
        let req = Request::Chat {
            session_id: "s1".into(),
            text: "hello".into(),
            attachments: Vec::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("attachments"));
        // ...and round-trips as empty
        let back: Request = serde_json::from_str(&json).unwrap();
        match back {
            Request::Chat { attachments, .. } => assert!(attachments.is_empty()),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn request_create_session_defaults() {
        // A minimal CreateSession must still parse (only known non-default fields)
        let json = r#"{"type":"create_session","model":"m"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::CreateSession {
                model,
                system_prompt,
                ..
            } => {
                assert_eq!(model.as_deref(), Some("m"));
                assert_eq!(system_prompt, None);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn request_list_sessions_defaults_include_archived() {
        let req: Request = serde_json::from_str(r#"{"type":"list_sessions"}"#).unwrap();
        assert!(matches!(
            req,
            Request::ListSessions {
                include_archived: false
            }
        ));
        let req: Request =
            serde_json::from_str(r#"{"type":"list_sessions","include_archived":true}"#).unwrap();
        assert!(matches!(
            req,
            Request::ListSessions {
                include_archived: true
            }
        ));
    }

    #[test]
    fn request_scalar_variants_roundtrip() {
        let requests = vec![
            Request::GetSessionInfo {
                session_id: "s1".into(),
            },
            Request::ArchiveSession {
                session_id: "s1".into(),
            },
            Request::GetMessages {
                session_id: "s1".into(),
            },
            Request::Subscribe {
                session_id: "s1".into(),
            },
            Request::CancelChat {
                session_id: "s1".into(),
            },
            Request::ListModels,
            Request::SetModel {
                session_id: "s1".into(),
                model_id: "m2".into(),
            },
        ];
        for req in &requests {
            let json = serde_json::to_string(req).unwrap();
            let back: Request = serde_json::from_str(&json).unwrap();
            assert_eq!(req, &back, "round-trip failed for {json}");
        }
    }

    #[test]
    fn chat_attachment_to_user_content() {
        let att = ChatAttachment::Image {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        };
        let uc = att.to_user_content();
        assert!(matches!(uc, UserContent::Image(i) if i.mime_type == "image/png"));
    }

    #[test]
    fn response_stream_and_done_roundtrip() {
        let stream = Response::Stream {
            event: Box::new(StreamEvent::TextDelta {
                content_index: 0,
                delta: "hi".into(),
                partial: crate::types::AssistantMessage::empty("api", "prov", "m"),
            }),
        };
        let json = serde_json::to_string(&stream).unwrap();
        assert!(json.starts_with(r#"{"type":"stream""#));
        let back: Response = serde_json::from_str(&json).unwrap();
        assert_eq!(stream, back);
    }

    #[test]
    fn response_broadcast_variants_roundtrip() {
        assert_eq!(
            serde_json::from_str::<Response>(r#"{"type":"agent_done"}"#).unwrap(),
            Response::AgentDone
        );
        assert_eq!(
            serde_json::from_str::<Response>(r#"{"type":"cancelled"}"#).unwrap(),
            Response::Cancelled
        );
        assert_eq!(
            serde_json::from_str::<Response>(r#"{"type":"ok"}"#).unwrap(),
            Response::Ok
        );
        assert_eq!(
            serde_json::from_str::<Response>(r#"{"type":"session_archived"}"#).unwrap(),
            Response::SessionArchived
        );
    }

    #[test]
    fn response_error_kind_roundtrip_and_mapping() {
        let err = Response::Error {
            kind: ErrorKind::RateLimited {
                retry_after: Some(5),
            },
            message: "rate limited".into(),
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.starts_with(r#"{"type":"error","kind":{"kind":"rate_limited""#));
        let back: Response = serde_json::from_str(&json).unwrap();
        assert_eq!(err, back);

        // Internal-error mapping
        assert_eq!(
            ErrorKind::from_error(&crate::Error::Auth("x".into())),
            ErrorKind::Auth
        );
        assert_eq!(
            ErrorKind::from_error(&crate::Error::RateLimited {
                provider: "x".into(),
                retry_after: Some(3),
            }),
            ErrorKind::RateLimited {
                retry_after: Some(3)
            }
        );
        assert_eq!(
            ErrorKind::from_error(&crate::Error::Internal("boom".into())),
            ErrorKind::Internal
        );
        assert_eq!(
            ErrorKind::from_error(&crate::Error::NoApiKey("x".into())),
            ErrorKind::NoApiKey
        );
    }

    #[test]
    fn session_info_missing_optional_fields_default() {
        let json = r#"{"id":"s1","model":"m","provider":"p","cwd":"/cwd","message_count":0,"stats":{"user_messages":0,"assistant_messages":0,"tool_calls":0,"tool_results":0,"tokens":{"input":0,"output":0,"cache_read":0,"cache_write":0},"cost":0.0,"context_window":0},"created_at":1,"last_activity":2}"#;
        let info: SessionInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.state, AgentPhase::Idle);
        assert_eq!(info.parent_id, None);
        assert!(!info.archived);
    }

    #[test]
    fn session_info_roundtrip() {
        let info = sample_session_info();
        let json = serde_json::to_string(&info).unwrap();
        let back: SessionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn token_stats_total() {
        let t = TokenStats {
            input: 1,
            output: 2,
            cache_read: 3,
            cache_write: 4,
        };
        assert_eq!(t.total(), 10);
        assert_eq!(TokenStats::default().total(), 0);
    }

    #[test]
    fn all_requests_have_snake_case_type_tags() {
        let requests = vec![
            Request::Chat {
                session_id: "s".into(),
                text: "t".into(),
                attachments: vec![],
            },
            Request::CreateSession {
                model: None,
                provider: None,
                system_prompt: None,
                cwd: None,
                parent_id: None,
                tagline: None,
            },
            Request::GetSessionInfo {
                session_id: "s".into(),
            },
            Request::ListSessions {
                include_archived: false,
            },
            Request::ArchiveSession {
                session_id: "s".into(),
            },
            Request::GetMessages {
                session_id: "s".into(),
            },
            Request::Subscribe {
                session_id: "s".into(),
            },
            Request::CancelChat {
                session_id: "s".into(),
            },
            Request::ListModels,
            Request::SetModel {
                session_id: "s".into(),
                model_id: "m".into(),
            },
        ];
        // Rust enum names → snake_case wire tags, sanity-checked via serde.
        for req in requests {
            let json = serde_json::to_string(&req).unwrap();
            assert!(json.starts_with("{\"type\":"), "no type tag: {json}");
        }
    }

    #[test]
    fn chat_attachment_wire_shape() {
        let json = serde_json::json!({
            "type": "image",
            "data": "AAAA",
            "mime_type": "image/png"
        });
        let att: ChatAttachment = serde_json::from_value(json).unwrap();
        assert!(matches!(att, ChatAttachment::Image { mime_type, .. } if mime_type == "image/png"));
    }
}
