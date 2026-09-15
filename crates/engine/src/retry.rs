#![allow(clippy::manual_range_contains)]
use std::time::Duration;
use tars_base::{CancelToken, Error, StreamEvent};

const MAX_RETRY_DELAY_MS: u64 = 32_000;

/// Classify an error and decide retry delay.
///
/// Returns `Some(delay)` if the error is retryable at this attempt,
/// `None` if the error should be returned immediately.
pub fn classify_error(
    err: &Error,
    attempt: usize,
    max_retries: usize,
    retry_base_ms: u64,
) -> Option<Duration> {
    if attempt >= max_retries {
        return None;
    }
    match err {
        Error::RateLimited { retry_after, .. } => {
            if let Some(secs) = retry_after {
                Some(Duration::from_secs(*secs))
            } else {
                Some(backoff(attempt, retry_base_ms))
            }
        }
        Error::Timeout(_) => Some(backoff(attempt, retry_base_ms)),
        Error::Http { status, .. } if (500..600).contains(status) => {
            Some(backoff(attempt, retry_base_ms))
        }
        Error::Internal(_) => Some(backoff(attempt, retry_base_ms)),
        Error::Auth(_)
        | Error::ContextOverflow
        | Error::NoProvider(_)
        | Error::NoApiKey(_)
        | Error::Cancelled
        | Error::ChannelClosed
        | Error::Parse(_)
        | Error::Io(_)
        | Error::Json(_) => None,
        // Http 4xx other than 429/401 are not retryable
        Error::Http { .. } => None,
    }
}

fn backoff(attempt: usize, base_ms: u64) -> Duration {
    let raw = base_ms
        .saturating_mul(1u64 << attempt)
        .min(MAX_RETRY_DELAY_MS);
    // subtractive jitter 0-25% using nanos as pseudo-random
    let jitter = (raw as f64
        * 0.25
        * (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as f64
            / 1_000_000_000.0)) as u64;
    Duration::from_millis(raw.saturating_sub(jitter))
}

pub fn backoff_for_test(attempt: usize, base_ms: u64) -> Duration {
    let raw = base_ms
        .saturating_mul(1u64 << attempt)
        .min(MAX_RETRY_DELAY_MS);
    Duration::from_millis(raw)
}

/// Cancellable countdown that emits a `Status` event each second.
pub async fn retry_countdown(
    delay: Duration,
    attempt: usize,
    max_attempts: usize,
    kind: &str,
    err_msg: &str,
    event_tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    cancel: &CancelToken,
) -> Result<(), Error> {
    if delay.is_zero() {
        let _ = event_tx
            .send(StreamEvent::Status {
                message: format!(
                    "Retrying (attempt {}/{}) immediately: {}",
                    attempt, max_attempts, err_msg
                ),
            })
            .await;
        return Ok(());
    }

    let deadline = tokio::time::Instant::now() + delay;
    let mut last_shown: Option<u64> = None;
    loop {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        let remaining_secs = remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0);
        if last_shown != Some(remaining_secs) {
            let _ = event_tx
                .send(StreamEvent::Status {
                    message: format!(
                        "Retrying (attempt {}/{}) in {}s... ({}: {})",
                        attempt, max_attempts, remaining_secs, kind, err_msg
                    ),
                })
                .await;
            last_shown = Some(remaining_secs);
        }
        // sleep min(remaining, 1s) but interruptible via cancel check loop
        let sleep = remaining.min(Duration::from_secs(1));
        tokio::time::sleep(sleep).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limited_with_retry_after() {
        let err = Error::RateLimited {
            provider: "openai".into(),
            retry_after: Some(7),
        };
        let d = classify_error(&err, 0, 3, 500).unwrap();
        assert_eq!(d, Duration::from_secs(7));
    }

    #[test]
    fn rate_limited_without_retry_after_uses_backoff() {
        let err = Error::RateLimited {
            provider: "openai".into(),
            retry_after: None,
        };
        let d = classify_error(&err, 0, 3, 500).unwrap();
        // backoff attempt 0 = 500ms (jitter may reduce slightly, but at least > 300)
        assert!(d.as_millis() >= 300 && d.as_millis() <= 500);
    }

    #[test]
    fn internal_is_retryable() {
        let err = Error::Internal("boom".into());
        assert!(classify_error(&err, 0, 3, 500).is_some());
    }

    #[test]
    fn auth_not_retryable() {
        let err = Error::Auth("anthropic".into());
        assert!(classify_error(&err, 0, 3, 500).is_none());
    }

    #[test]
    fn max_retries_exhausted() {
        let err = Error::RateLimited {
            provider: "a".into(),
            retry_after: Some(1),
        };
        assert!(classify_error(&err, 3, 3, 500).is_none());
    }

    #[tokio::test]
    async fn countdown_emits_status_and_respects_cancel() {
        let cancel = CancelToken::new();
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let delay = Duration::from_millis(150);
        let fut = retry_countdown(delay, 1, 3, "retryable", "boom", &tx, &cancel);
        // should complete after ~150ms with at least one status
        let res = tokio::time::timeout(Duration::from_secs(1), fut)
            .await
            .unwrap();
        assert!(res.is_ok());
        let mut saw = false;
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Status { message } = ev {
                if message.starts_with("Retrying ") {
                    saw = true;
                }
            }
        }
        assert!(saw);
    }

    #[tokio::test]
    async fn countdown_aborts_on_cancel() {
        let cancel = CancelToken::new();
        let (tx, _) = tokio::sync::mpsc::channel(16);
        let delay = Duration::from_secs(5);
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });
        let res = retry_countdown(delay, 1, 3, "retryable", "boom", &tx, &cancel).await;
        assert!(matches!(res, Err(Error::Cancelled)));
    }
}
