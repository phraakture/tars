/// The error type propagated throughout tars.
///
/// Designed as a public API contract: variants carry structured data (not
/// opaque strings) so callers can match on variant kind and extract fields
/// like `status`, `provider`, or `retry_after` without string parsing.
///
/// [`io::Error`] and [`serde_json::Error`] are preserved via `#[from] #[source]`
/// so the `?` operator works directly and the original cause chain is intact.
///
/// Note: thiserror 2.x requires explicit `#[source]`; `#[from]` alone does NOT
/// imply it (unlike thiserror 1.x), and `#[error(transparent)]` delegates to
/// the inner error's *own* `source()`, not the inner error itself.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no provider registered for '{0}'")]
    NoProvider(String),

    #[error("no API key for provider '{0}'")]
    NoApiKey(String),

    #[error("HTTP error: {status} {message}")]
    Http { status: u16, message: String },

    #[error("rate limited by {provider}")]
    RateLimited {
        provider: String,
        retry_after: Option<u64>,
    },

    #[error("request to '{0}' timed out")]
    Timeout(String),

    #[error("authentication error for provider '{0}'")]
    Auth(String),

    #[error("context window overflow")]
    ContextOverflow,

    #[error("cancelled")]
    Cancelled,

    #[error("io: {0}")]
    Io(
        #[from]
        #[source]
        std::io::Error,
    ),

    #[error("json: {0}")]
    Json(
        #[from]
        #[source]
        serde_json::Error,
    ),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("channel closed")]
    ChannelClosed,

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn error_display_no_provider() {
        let e = Error::NoProvider("anthropic-messages".into());
        assert_eq!(
            e.to_string(),
            "no provider registered for 'anthropic-messages'"
        );
    }

    #[test]
    fn error_display_http() {
        let e = Error::Http {
            status: 429,
            message: "too many requests".into(),
        };
        assert_eq!(e.to_string(), "HTTP error: 429 too many requests");
    }

    #[test]
    fn error_display_rate_limited_with_retry_after() {
        let e = Error::RateLimited {
            provider: "openai".into(),
            retry_after: Some(30),
        };
        assert_eq!(e.to_string(), "rate limited by openai");
    }

    #[test]
    fn error_display_rate_limited_without_retry_after() {
        let e = Error::RateLimited {
            provider: "openai".into(),
            retry_after: None,
        };
        assert_eq!(e.to_string(), "rate limited by openai");
    }

    #[test]
    fn error_display_simple_variants() {
        assert_eq!(
            Error::Timeout("anthropic".into()).to_string(),
            "request to 'anthropic' timed out"
        );
        assert_eq!(
            Error::Auth("openai".into()).to_string(),
            "authentication error for provider 'openai'"
        );
        assert_eq!(
            Error::ContextOverflow.to_string(),
            "context window overflow"
        );
        assert_eq!(Error::Cancelled.to_string(), "cancelled");
        assert_eq!(Error::ChannelClosed.to_string(), "channel closed");
        assert_eq!(
            Error::Internal("something broke".into()).to_string(),
            "internal error: something broke"
        );
        assert_eq!(
            Error::Parse("bad json".into()).to_string(),
            "parse error: bad json"
        );
    }

    #[test]
    fn error_source_chain_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err: Error = io_err.into();
        assert!(err.source().is_some());
        // source() is the io::Error itself (not its inner source)
        assert_eq!(err.source().unwrap().to_string(), "pipe broke");
        // display includes the variant context
        assert_eq!(err.to_string(), "io: pipe broke");
    }

    #[test]
    fn error_source_chain_json() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let err: Error = json_err.into();
        assert!(err.source().is_some());
    }

    #[test]
    fn error_source_chain_no_provider() {
        let err = Error::NoProvider("test".into());
        // NoProvider has no #[source], so source() returns None
        assert!(err.source().is_none());
    }
}
