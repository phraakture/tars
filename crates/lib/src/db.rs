use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};

use tars_base::{Message, Model, StopReason};

pub fn message_indicates_incomplete_turn(msg: &Message) -> bool {
    match msg {
        Message::User(_) | Message::ToolResult(_) => true,
        Message::Assistant(a) => matches!(a.stop_reason, StopReason::ToolUse | StopReason::Error),
        _ => false,
    }
}

fn db_err(ctx: &str) -> impl FnOnce(rusqlite::Error) -> tars_base::Error + '_ {
    move |e| tars_base::Error::Io(std::io::Error::other(format!("{}: {}", ctx, e)))
}

#[derive(Debug, Clone)]
pub struct StoredSession {
    pub id: String,
    pub model: Model,
    pub system_prompt: Option<String>,
    pub cwd: Option<String>,
    pub created_at: i64,
}

const SESSION_COLUMNS: &str = "id, model_json, system_prompt, cwd, created_at";

fn row_to_session(row: &rusqlite::Row) -> rusqlite::Result<StoredSession> {
    let model_json: String = row.get(1)?;
    let model: Model = serde_json::from_str(&model_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(StoredSession {
        id: row.get(0)?,
        model,
        system_prompt: row.get(2)?,
        cwd: row.get(3)?,
        created_at: row.get(4)?,
    })
}

pub struct Db {
    conn: Connection,
}

unsafe impl Send for Db {}
unsafe impl Sync for Db {}

impl Db {
    pub fn open(path: &Path) -> tars_base::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                tars_base::Error::Io(std::io::Error::other(format!(
                    "mkdir {}: {}",
                    parent.display(),
                    e
                )))
            })?;
        }
        let conn =
            Connection::open(path).map_err(db_err(&format!("open db {}", path.display())))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(db_err("pragma"))?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    pub fn open_memory() -> tars_base::Result<Self> {
        let conn = Connection::open_in_memory().map_err(db_err("open in-memory db"))?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .map_err(db_err("pragma"))?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    fn init_schema(conn: &Connection) -> tars_base::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id             TEXT PRIMARY KEY,
                model_json     TEXT NOT NULL,
                system_prompt  TEXT,
                cwd            TEXT,
                created_at     INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_json TEXT NOT NULL,
                created_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            ",
        )
        .map_err(db_err("create tables"))?;
        Ok(())
    }

    pub fn create_session(&self, session: &StoredSession) -> tars_base::Result<()> {
        let model_json = serde_json::to_string(&session.model)
            .map_err(|e| tars_base::Error::Parse(e.to_string()))?;
        self.conn
            .execute(
                "INSERT INTO sessions (id, model_json, system_prompt, cwd, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    session.id,
                    model_json,
                    session.system_prompt,
                    session.cwd,
                    session.created_at,
                ],
            )
            .map_err(db_err("insert session"))?;
        Ok(())
    }

    pub fn get_session(&self, id: &str) -> tars_base::Result<Option<StoredSession>> {
        let sql = format!("SELECT {} FROM sessions WHERE id = ?1", SESSION_COLUMNS);
        self.conn
            .query_row(&sql, params![id], row_to_session)
            .optional()
            .map_err(db_err("get session"))
    }

    pub fn list_sessions(&self) -> tars_base::Result<Vec<StoredSession>> {
        let sql = format!(
            "SELECT {} FROM sessions ORDER BY created_at",
            SESSION_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql).map_err(db_err("prepare list"))?;
        let rows = stmt
            .query_map([], row_to_session)
            .map_err(db_err("list sessions"))?;
        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.map_err(db_err("read session row"))?);
        }
        Ok(sessions)
    }

    pub fn delete_session(&self, id: &str) -> tars_base::Result<()> {
        self.conn
            .execute("DELETE FROM sessions WHERE id = ?1", params![id])
            .map_err(db_err("delete session"))?;
        Ok(())
    }

    pub fn append_message(&self, session_id: &str, message: &Message) -> tars_base::Result<()> {
        let json =
            serde_json::to_string(message).map_err(|e| tars_base::Error::Parse(e.to_string()))?;
        let now = tars_base::timestamp_ms() as i64;
        self.conn
            .execute(
                "INSERT INTO messages (session_id, message_json, created_at) VALUES (?1, ?2, ?3)",
                params![session_id, json, now],
            )
            .map_err(db_err("insert message"))?;
        Ok(())
    }

    pub fn get_messages(&self, session_id: &str) -> tars_base::Result<Vec<Message>> {
        let mut stmt = self
            .conn
            .prepare("SELECT message_json FROM messages WHERE session_id = ?1 ORDER BY id")
            .map_err(db_err("prepare messages"))?;
        let rows = stmt
            .query_map(params![session_id], |row| {
                let json: String = row.get(0)?;
                let msg: Message = serde_json::from_str(&json).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
                Ok(msg)
            })
            .map_err(db_err("query messages"))?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row.map_err(db_err("read message row"))?);
        }
        Ok(messages)
    }

    pub fn message_count(&self, session_id: &str) -> tars_base::Result<usize> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(db_err("count messages"))?;
        Ok(count as usize)
    }

    /// Replace messages from index 0..cut with new replacement messages.
    /// Used for compaction: removes old messages and prepends a summary.
    pub fn replace_messages(
        &self,
        session_id: &str,
        cut: usize,
        replacements: &[Message],
    ) -> tars_base::Result<()> {
        // Get message IDs to delete (first `cut` messages by id order)
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM messages WHERE session_id = ?1 ORDER BY id LIMIT ?2")
            .map_err(db_err("prepare replace_messages"))?;

        let ids_to_delete: Vec<i64> = stmt
            .query_map(params![session_id, cut as i64], |row| row.get(0))
            .map_err(db_err("query ids"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(db_err("collect ids"))?;

        drop(stmt);

        if ids_to_delete.is_empty() {
            return Ok(());
        }

        // Delete old messages
        let placeholders: Vec<String> = ids_to_delete.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "DELETE FROM messages WHERE id IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<&dyn rusqlite::types::ToSql> = ids_to_delete
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        self.conn
            .execute(&sql, params.as_slice())
            .map_err(db_err("delete old messages"))?;

        // Insert replacement messages
        let now = tars_base::timestamp_ms() as i64;
        for msg in replacements {
            let json =
                serde_json::to_string(msg).map_err(|e| tars_base::Error::Parse(e.to_string()))?;
            self.conn
                .execute(
                    "INSERT INTO messages (session_id, message_json, created_at) VALUES (?1, ?2, ?3)",
                    params![session_id, json, now],
                )
                .map_err(db_err("insert replacement"))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tars_base::{Model, ModelCost, ThinkingStyle, UserMessage};

    fn test_model() -> Model {
        Model {
            id: "test-model".into(),
            name: "Test".into(),
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

    fn session(id: &str) -> StoredSession {
        StoredSession {
            id: id.into(),
            model: test_model(),
            system_prompt: Some("you are helpful".into()),
            cwd: Some("/tmp".into()),
            created_at: tars_base::timestamp_ms() as i64,
        }
    }

    #[test]
    fn persistence_round_trip() {
        let db = Db::open_memory().unwrap();
        let s = session("s1");
        db.create_session(&s).unwrap();
        db.append_message("s1", &Message::User(UserMessage::text("hello")))
            .unwrap();
        let mut assistant = tars_base::AssistantMessage::empty("mock", "mock", "test-model");
        assistant
            .content
            .push(tars_base::AssistantContent::Text(tars_base::TextContent {
                text: "hi there".into(),
                text_signature: None,
            }));
        db.append_message("s1", &Message::Assistant(assistant))
            .unwrap();

        let sessions = db.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "s1");

        let msgs = db.get_messages("s1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], Message::User(_)));
        assert!(matches!(msgs[1], Message::Assistant(_)));
    }

    #[test]
    fn reopen_sees_prior_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = Db::open(&path).unwrap();
            db.create_session(&session("s1")).unwrap();
            db.append_message("s1", &Message::User(UserMessage::text("hello")))
                .unwrap();
        }
        {
            let db = Db::open(&path).unwrap();
            let sessions = db.list_sessions().unwrap();
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].id, "s1");
            let msgs = db.get_messages("s1").unwrap();
            assert_eq!(msgs.len(), 1);
        }
    }

    #[test]
    fn delete_cascades_messages() {
        let db = Db::open_memory().unwrap();
        db.create_session(&session("s1")).unwrap();
        db.append_message("s1", &Message::User(UserMessage::text("hi")))
            .unwrap();
        db.delete_session("s1").unwrap();
        assert!(db.get_session("s1").unwrap().is_none());
        assert_eq!(db.get_messages("s1").unwrap().len(), 0);
    }

    #[test]
    fn message_indicates_incomplete() {
        let user = Message::User(UserMessage::text("hi"));
        assert!(message_indicates_incomplete_turn(&user));
        let tr = Message::ToolResult(tars_base::ToolResultMessage::success("tc1", "bash", "ok"));
        assert!(message_indicates_incomplete_turn(&tr));
        let mut a = tars_base::AssistantMessage::empty("m", "m", "m");
        a.stop_reason = StopReason::ToolUse;
        assert!(message_indicates_incomplete_turn(&Message::Assistant(
            a.clone()
        )));
        a.stop_reason = StopReason::Error;
        assert!(message_indicates_incomplete_turn(&Message::Assistant(
            a.clone()
        )));
        a.stop_reason = StopReason::Stop;
        assert!(!message_indicates_incomplete_turn(&Message::Assistant(a)));
        let info = Message::Info(tars_base::InfoMessage::new("info"));
        assert!(!message_indicates_incomplete_turn(&info));
    }
}
