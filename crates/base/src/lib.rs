//! Shared types, wire protocol, config, and utilities for the tars workspace.
//!
//! This is the leaf crate that every other tars workspace crate depends on.
//! Dependencies are kept minimal: serde, serde_json, toml, thiserror, and
//! tokio (for async JSON-line I/O helpers used by the client and server
//! crates).

mod error;
pub use error::{Error, Result};

mod types;
pub use types::*;

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

/// Current wall-clock time in milliseconds since the Unix epoch.
pub fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// JSON-line I/O helpers (sync)
// ---------------------------------------------------------------------------

/// Write a single JSON object as one line to `writer`, followed by a newline,
/// and flush.
pub fn write_json_line<T: serde::Serialize>(
    writer: &mut impl std::io::Write,
    val: &T,
) -> Result<()> {
    let mut line = serde_json::to_string(val)?;
    line.push('\n');
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    Ok(())
}

/// Read a single JSON line from `reader`.  Returns `Ok(None)` on EOF.
pub fn read_json_line<T: serde::de::DeserializeOwned>(
    reader: &mut impl std::io::BufRead,
) -> Result<Option<T>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&line)?))
}

// ---------------------------------------------------------------------------
// Async JSON-line I/O helpers (tokio)
// ---------------------------------------------------------------------------

/// Async version of [`write_json_line`].
pub async fn write_json_line_async<T: serde::Serialize>(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    val: &T,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut line = serde_json::to_string(val)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

/// Async version of [`read_json_line`].  Returns `Ok(None)` on EOF.
pub async fn read_json_line_async<T: serde::de::DeserializeOwned>(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
) -> Result<Option<T>> {
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&line)?))
}

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

/// Truncate `s` to at most `max_bytes` bytes, rounding down to a char
/// boundary.  Keeps the **start** of the string.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Truncate `s` to at most `max_bytes` bytes from the *end*, rounding up to a
/// char boundary.  Keeps the **end** of the string.
pub fn truncate_str_end(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len() - max_bytes;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- timestamp_ms --

    #[test]
    fn timestamp_ms_is_reasonable() {
        let ts = timestamp_ms();
        // After 2020-01-01 and before 2100-01-01 (in ms)
        assert!(ts > 1_577_836_800_000);
        assert!(ts < 4_102_444_800_000);
    }

    // -- truncate_str (keeps the start) --

    #[test]
    fn truncate_str_short_unchanged() {
        let s = "hello";
        assert_eq!(truncate_str(s, 10), "hello");
    }

    #[test]
    fn truncate_str_exact() {
        let s = "abcde";
        assert_eq!(truncate_str(s, 5), "abcde");
    }

    #[test]
    fn truncate_str_long() {
        let s = "abcdefghij";
        assert_eq!(truncate_str(s, 6), "abcdef");
    }

    #[test]
    fn truncate_str_char_boundary_ascii() {
        // ASCII: every byte is a char boundary
        let s = "abcdef";
        assert_eq!(truncate_str(s, 4), "abcd");
    }

    #[test]
    fn truncate_str_char_boundary_multibyte() {
        // "café" = [63, 61, 66, c3, a9] = 5 bytes; é is 2 bytes at indices 3-4
        // max_bytes=4 would land in the middle of é, so rounds down to 3
        let s = "café";
        assert_eq!(truncate_str(s, 4), "caf");
        assert_eq!(truncate_str(s, 3), "caf");
        assert_eq!(truncate_str(s, 5), "café");
        // max_bytes=2 stays at "ca"
        assert_eq!(truncate_str(s, 2), "ca");
    }

    #[test]
    fn truncate_str_emoji() {
        // "hello👋" = "hello" (5 bytes) + 👋 (4 bytes) = 9 bytes
        let s = "hello👋";
        assert_eq!(truncate_str(s, 8), "hello");
        assert_eq!(truncate_str(s, 9), "hello👋");
    }

    // -- truncate_str_end (keeps the end) --

    #[test]
    fn truncate_str_end_short_unchanged() {
        let s = "hello";
        assert_eq!(truncate_str_end(s, 10), "hello");
    }

    #[test]
    fn truncate_str_end_long() {
        let s = "abcdefghij";
        assert_eq!(truncate_str_end(s, 6), "efghij");
    }

    #[test]
    fn truncate_str_end_char_boundary() {
        // "café" keeps the end when truncating from the start
        let s = "café";
        assert_eq!(truncate_str_end(s, 4), "afé");
        assert_eq!(truncate_str_end(s, 5), "café");
        assert_eq!(truncate_str_end(s, 2), "é");
        // max_bytes=1 lands mid-é and rounds up past it to the end
        assert_eq!(truncate_str_end(s, 1), "");
    }

    #[test]
    fn truncate_str_end_emoji() {
        let s = "hello👋";
        // last 4 bytes = 👋
        assert_eq!(truncate_str_end(s, 4), "👋");
        assert_eq!(truncate_str_end(s, 5), "o👋");
    }

    // -- JSON-line helpers --

    #[test]
    fn json_line_round_trip_sync() {
        use std::io::Cursor;

        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct Msg {
            kind: String,
            count: u32,
        }

        let msg = Msg {
            kind: "tool_result".into(),
            count: 42,
        };

        let mut buf = Vec::new();
        write_json_line(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let read_back: Msg = read_json_line(&mut cursor).unwrap().unwrap();
        assert_eq!(msg, read_back);
    }

    #[test]
    fn json_line_read_eof() {
        use std::io::Cursor;
        let mut cursor = Cursor::new(b"");
        let result: Option<serde_json::Value> = read_json_line(&mut cursor).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn json_line_read_trailing_newline() {
        use std::io::Cursor;
        let data = b"{\"x\":1}\n";
        let mut cursor = Cursor::new(data);
        let val: serde_json::Value = read_json_line(&mut cursor).unwrap().unwrap();
        assert_eq!(val, serde_json::json!({"x": 1}));
    }

    #[test]
    fn json_line_read_invalid_json() {
        use std::io::Cursor;
        let mut cursor = Cursor::new(b"not json\n");
        let err = read_json_line::<serde_json::Value>(&mut cursor).unwrap_err();
        assert!(matches!(err, Error::Json(_)));
    }

    #[tokio::test]
    async fn json_line_round_trip_async() {
        use tokio::io::BufReader;

        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct Msg {
            text: String,
        }

        let msg = Msg {
            text: "hello world".into(),
        };

        // Use an in-process duplex: write to a Vec, then read it back
        let mut writer: Vec<u8> = Vec::new();
        write_json_line_async(&mut writer, &msg).await.unwrap();

        let mut reader = BufReader::new(&writer[..]);
        let read_back: Msg = read_json_line_async(&mut reader).await.unwrap().unwrap();
        assert_eq!(msg, read_back);
    }

    #[tokio::test]
    async fn json_line_async_eof() {
        use tokio::io::BufReader;
        let empty: Vec<u8> = Vec::new();
        let mut reader = BufReader::new(&empty[..]);
        let result: Option<serde_json::Value> = read_json_line_async(&mut reader).await.unwrap();
        assert!(result.is_none());
    }
}
