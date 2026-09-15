//! Server daemon helpers shared by the CLI and the TUI.
//!
//! `connect_or_start` connects to the unix socket if a server is already
//! listening; otherwise it binds the socket in-process (background task) so
//! the calling process effectively becomes the daemon, then waits for the
//! socket to accept connections.

use tokio::net::UnixListener;

/// Connect to the server socket, starting the daemon if none is running.
///
/// Returns a connected client once the socket accepts. Fails after ~5s if
/// the server never comes up.
pub async fn connect_or_start(paths: &tars_base::Paths) -> anyhow::Result<tars_client::Client> {
    let socket_path = paths.socket_path();
    if let Ok(client) = tars_client::Client::connect(&socket_path).await {
        return Ok(client);
    }

    tracing::info!("server not running, starting daemon");
    start_server_daemon(paths).await?;

    for _ in 0..50 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if let Ok(client) = tars_client::Client::connect(&socket_path).await {
            return Ok(client);
        }
    }
    Err(anyhow::anyhow!("server did not start within 5s"))
}

/// Start the server listener in this process (background task).
///
/// Performs single-instance checks via the PID file, removes stale socket
/// files, binds the socket, writes the PID file, and detaches the accept
/// loop into a tokio task. The calling process must stay alive.
pub async fn start_server_daemon(paths: &tars_base::Paths) -> anyhow::Result<()> {
    let socket_path = paths.socket_path();
    let db_path = paths.data_dir().join("tars.db");

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Single-instance check via PID file
    let pid_path = paths.pid_path();
    if pid_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&pid_path) {
            if let Ok(pid) = contents.trim().parse::<u32>() {
                if is_process_running(pid) {
                    return Err(anyhow::anyhow!("server already running (pid {pid})"));
                }
            }
        }
        // Stale PID file
        let _ = std::fs::remove_file(&pid_path);
    }

    // Stale socket
    let _ = std::fs::remove_file(&socket_path);

    let db = crate::db::Db::open(&db_path)?;
    let state = std::sync::Arc::new(crate::server::SharedState::new(db));
    let listener = UnixListener::bind(&socket_path)?;

    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&pid_path, std::process::id().to_string())?;

    let sock = socket_path.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::server::run(listener, state).await {
            tracing::error!("server error: {e}");
        }
        let _ = std::fs::remove_file(&sock);
        let _ = std::fs::remove_file(&pid_path);
    });

    Ok(())
}

/// Check whether a process with the given PID exists (`kill -0`).
pub fn is_process_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
