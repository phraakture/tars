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
    Sessions,
    Models,
    Chat(ChatArgs),
    Config(ConfigArgs),
    /// Interactive terminal UI
    Tui,
}

#[derive(Parser)]
struct ConfigArgs {
    #[command(subcommand)]
    cmd: ConfigCmd,
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Reload provider/model configuration from disk
    Reload,
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

#[derive(Parser)]
struct ChatArgs {
    /// One-shot message to send
    #[arg(short = 'm', long)]
    message: Option<String>,

    /// Model to use (default: first available)
    #[arg(short = 'M', long)]
    model: Option<String>,
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
        Some(Commands::Sessions) => {
            let mut client = connect_or_start().await?;
            let sessions = client.list_sessions().await?;
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                for s in &sessions {
                    let tagline = s.tagline.as_deref().unwrap_or("");
                    let project = s
                        .project_name
                        .as_deref()
                        .map(|p| format!("  project={p}"))
                        .unwrap_or_default();
                    println!(
                        "{}  model={}  msgs={}{}  {}",
                        s.id, s.model, s.message_count, project, tagline
                    );
                }
            }
        }
        Some(Commands::Models) => {
            let mut client = connect_or_start().await?;
            let models = client.list_models().await?;
            for m in &models {
                println!(
                    "{:<20} provider={:<12} ctx={}",
                    m.id, m.provider, m.context_window
                );
            }
        }
        Some(Commands::Chat(args)) => {
            let paths = tars_base::Paths::detect();
            let socket_path = paths.socket_path();
            let mut client = connect_or_start().await?;
            let session_id = client.create_session(args.model, None).await?;

            if let Some(text) = args.message {
                let mut rx = client.chat(&session_id, &text).await?;
                while let Some(resp) = rx.recv().await {
                    match resp {
                        tars_base::protocol::Response::Stream { event } => {
                            print_stream_text(&event);
                        }
                        tars_base::protocol::Response::AgentDone => break,
                        tars_base::protocol::Response::Error { message, .. } => {
                            eprintln!("error: {message}");
                            std::process::exit(1);
                        }
                        _ => {}
                    }
                }
                println!();
            } else {
                // Interactive REPL
                use std::io::{self, BufRead, Write};
                let stdin = io::stdin();
                let mut stdout = io::stdout();
                loop {
                    print!("> ");
                    stdout.flush()?;
                    let mut line = String::new();
                    match stdin.lock().read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {
                            let text = line.trim().to_string();
                            if text.is_empty() {
                                continue;
                            }
                            if text == "/quit" || text == "/exit" {
                                break;
                            }
                            let chat_client = tars_client::Client::connect(&socket_path).await?;
                            let mut rx = chat_client.chat(&session_id, &text).await?;
                            while let Some(resp) = rx.recv().await {
                                match resp {
                                    tars_base::protocol::Response::Stream { event } => {
                                        print_stream_text(&event);
                                    }
                                    tars_base::protocol::Response::AgentDone => break,
                                    tars_base::protocol::Response::Error { message, .. } => {
                                        eprintln!("\nerror: {message}");
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            println!();
                        }
                        Err(e) => {
                            eprintln!("read error: {e}");
                            break;
                        }
                    }
                }
            }
        }
        Some(Commands::Tui) => {
            tars_tui::run().await?;
        }
        Some(Commands::Config(args)) => match args.cmd {
            ConfigCmd::Reload => {
                let paths = tars_base::Paths::detect();
                let config = tars_base::config::load_config(&paths)?;
                println!(
                    "loaded {} provider(s) from {}",
                    config.providers.len(),
                    paths.providers_path().display()
                );
                for (name, p) in &config.providers {
                    let models: Vec<&str> = p.models.iter().map(|m| m.id.as_str()).collect();
                    println!("  {} ({}): {}", name, p.api, models.join(", "));
                }
                println!("config reload complete (registry swap requires restart)");
            }
        },
        None => {
            println!(
                "tars: use 'tars chat -m \"hello\"' to talk, or 'tars server start --foreground' to run the daemon"
            );
        }
    }

    Ok(())
}
fn print_stream_text(event: &tars_base::StreamEvent) {
    match event {
        tars_base::StreamEvent::TextDelta { delta, .. } => {
            print!("{delta}");
        }
        tars_base::StreamEvent::ThinkingDelta { delta, .. } => {
            eprint!("[thinking] {delta}");
        }
        _ => {}
    }
}
async fn connect_or_start() -> anyhow::Result<tars_client::Client> {
    let paths = tars_base::Paths::detect();
    tars_lib::daemon::connect_or_start(&paths).await
}

async fn run_server(foreground: bool) -> anyhow::Result<()> {
    let _ = foreground;
    let paths = tars_base::Paths::detect();
    let socket_path = paths.socket_path();
    let db_path = paths.data_dir().join("tars.db");

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(&socket_path);

    let db = tars_lib::db::Db::open(&db_path)?;
    let state = std::sync::Arc::new(tars_lib::server::SharedState::new(db));

    let listener = tokio::net::UnixListener::bind(&socket_path)?;
    println!("tars server listening on {}", socket_path.display());

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
