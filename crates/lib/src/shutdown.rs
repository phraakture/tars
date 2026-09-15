//! Graceful shutdown coordination.
//!
//! Tracks in-flight agent turns, handles SIGTERM/SIGINT, and ensures clean
//! shutdown of plugins, socket, and DB.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Shared shutdown state across all tasks.
#[derive(Clone)]
pub struct ShutdownHandle {
    /// Set to true when shutdown is requested.
    flag: Arc<AtomicBool>,
    /// Number of in-flight agent loops.
    in_flight: Arc<AtomicUsize>,
}

impl ShutdownHandle {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Check if shutdown has been requested.
    pub fn is_shutting_down(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Request shutdown.
    pub fn request_shutdown(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Increment in-flight counter. Returns a guard that decrements on drop.
    pub fn enter(&self) -> ShutdownGuard {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        ShutdownGuard {
            handle: self.clone(),
        }
    }

    /// Number of active in-flight turns.
    pub fn active_count(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Returns a future that resolves when shutdown is requested.
    pub async fn cancelled(&self) {
        while !self.is_shutting_down() {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// Wait for all in-flight turns to complete, up to `timeout`.
    pub async fn drain(&self, timeout: std::time::Duration) {
        let deadline = tokio::time::Instant::now() + timeout;
        while self.active_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}

impl Default for ShutdownHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that decrements in-flight count on drop.
pub struct ShutdownGuard {
    handle: ShutdownHandle,
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        self.handle.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// PID file management.
pub struct PidFile {
    path: PathBuf,
}

impl PidFile {
    /// Create a new PID file. Returns error if another instance is running.
    pub fn create(path: PathBuf) -> std::io::Result<Self> {
        // Check if existing PID file points to a running process
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(pid) = contents.trim().parse::<u32>() {
                    if is_process_running(pid) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::AlreadyExists,
                            format!("server already running (pid {})", pid),
                        ));
                    }
                }
            }
        }

        // Write our PID
        std::fs::write(&path, std::process::id().to_string())?;
        Ok(Self { path })
    }

    /// Remove the PID file.
    pub fn remove(&self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        self.remove();
    }
}

/// Check if a process with the given PID is running.
fn is_process_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Spawn a tokio signal handler for SIGTERM and SIGINT.
/// Returns a ShutdownHandle that is triggered on signal.
pub fn install_signal_handler() -> ShutdownHandle {
    let handle = ShutdownHandle::new();
    let h = handle.clone();

    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();

        tokio::select! {
            _ = ctrl_c => {
                tracing::info!("received SIGINT, shutting down");
                h.request_shutdown();
            }
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM, shutting down");
                h.request_shutdown();
            }
        }
    });

    handle
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn pid_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pid");
        let pf = PidFile::create(path.clone()).unwrap();
        assert!(path.exists());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.trim().parse::<u32>().unwrap(), std::process::id());
        drop(pf);
        assert!(!path.exists());
    }

    #[test]
    fn pid_file_detects_running() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pid");
        // First create succeeds — writes our PID
        let _pf = PidFile::create(path.clone()).unwrap();
        // Second create should fail — same PID is still running
        let result = PidFile::create(path);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn shutdown_handle_drain() {
        let handle = ShutdownHandle::new();
        assert_eq!(handle.active_count(), 0);

        let guard = handle.enter();
        assert_eq!(handle.active_count(), 1);

        let guard2 = handle.enter();
        assert_eq!(handle.active_count(), 2);

        drop(guard);
        assert_eq!(handle.active_count(), 1);

        drop(guard2);
        assert_eq!(handle.active_count(), 0);
    }

    #[tokio::test]
    async fn shutdown_handle_drain_waits() {
        let handle = ShutdownHandle::new();
        let h = handle.clone();

        // Spawn a task that holds the guard for 100ms
        tokio::spawn(async move {
            let _guard = h.enter();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        // Give it time to start
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(handle.active_count() > 0);

        // Drain should wait
        handle.drain(Duration::from_secs(1)).await;
        assert_eq!(handle.active_count(), 0);
    }

    #[tokio::test]
    async fn shutdown_handle_drain_timeout() {
        let handle = ShutdownHandle::new();
        let h = hold_forever(handle.clone());

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(handle.active_count() > 0);

        handle.drain(Duration::from_millis(50)).await;
        // Should have timed out, still active
        assert!(handle.active_count() > 0);

        drop(h);
    }

    fn hold_forever(h: ShutdownHandle) -> impl Drop {
        h.enter()
    }
}
