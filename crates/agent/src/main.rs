use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tars", version, about = "tars agent harness")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Server(ServerArgs),
}

#[derive(Parser)]
struct ServerArgs {
    #[command(subcommand)]
    cmd: Option<ServerCmd>,

    /// Run in foreground (don't daemonize)
    #[arg(long, default_value_t = false)]
    foreground: bool,
}

#[derive(Subcommand)]
enum ServerCmd {
    Start {
        #[arg(long, default_value_t = false)]
        foreground: bool,
    },
    Status,
    Stop,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Server(args)) => {
            let foreground =
                args.foreground || matches!(args.cmd, Some(ServerCmd::Start { foreground: true }));
            run_server(foreground).await?;
        }
        None => {
            println!("tars: use 'tars server start --foreground' to run the daemon");
        }
    }

    Ok(())
}

async fn run_server(foreground: bool) -> anyhow::Result<()> {
    let _ = foreground;
    let paths = tars_base::Paths::detect();
    let socket_path = paths.socket_path();
    let db_path = paths.data_dir().join("tars.db");

    // Ensure directories exist
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Remove stale socket
    let _ = std::fs::remove_file(&socket_path);

    let db = tars_lib::db::Db::open(&db_path)?;
    let state = std::sync::Arc::new(tars_lib::server::SharedState::new(db));

    let listener = tokio::net::UnixListener::bind(&socket_path)?;
    println!("tars server listening on {}", socket_path.display());

    // Write pid file
    let pid_path = paths.pid_path();
    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&pid_path, std::process::id().to_string())?;

    let result = tars_lib::server::run(listener, state).await;
    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_file(&pid_path);
    result?;
    Ok(())
}
