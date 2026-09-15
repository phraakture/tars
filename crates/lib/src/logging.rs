//! Logging setup — tracing-subscriber with EnvFilter and optional file output.

use std::path::PathBuf;

use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initialize logging. Call once at process start.
///
/// - `log_dir`: directory for log files (None = stderr only)
/// - `level`: RUST_LOG-style filter (e.g. "info,tars_lib=debug")
pub fn init(log_dir: Option<&PathBuf>, level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    let stderr_layer = fmt::layer()
        .with_target(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_writer(std::io::stderr);

    match log_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir).ok();
            let file_appender = tracing_appender::rolling::daily(dir, "server.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

            // Leak the guard so the file appender stays alive for the process lifetime.
            // This is fine — it's a single static allocation for the process.
            Box::leak(Box::new(_guard));

            let file_layer = fmt::layer()
                .with_target(true)
                .with_span_events(fmt::format::FmtSpan::CLOSE)
                .with_writer(non_blocking)
                .with_ansi(false);

            tracing_subscriber::registry()
                .with(filter)
                .with(stderr_layer)
                .with(file_layer)
                .init();
        }
        None => {
            tracing_subscriber::registry()
                .with(filter)
                .with(stderr_layer)
                .init();
        }
    }
}

/// Prune log files older than `max_days` in the given directory.
pub fn prune_logs(dir: &PathBuf, max_days: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(max_days * 86400);

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        if let Ok(meta) = path.metadata() {
            if let Ok(modified) = meta.modified() {
                if modified < cutoff {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_handles_missing_dir() {
        // Should not panic on missing directory
        prune_logs(&PathBuf::from("/nonexistent/path"), 30);
    }

    #[test]
    fn prune_skips_non_log_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("not_a_log.txt"), "keep me").unwrap();
        prune_logs(&dir.path().to_path_buf(), 30);
        assert!(dir.path().join("not_a_log.txt").exists());
    }
}
