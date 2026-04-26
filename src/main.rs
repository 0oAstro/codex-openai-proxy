use clap::{Parser, Subcommand};
use tracing::info;

mod auth;
mod chat;
mod config;
mod models;
mod proxy;

#[derive(Parser)]
#[command(name = "codex-openai-proxy", version, about = "Proxy ChatGPT/Codex subscription as an OpenAI-compatible API")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the proxy server
    Serve {
        /// Port to listen on (overrides PORT env var)
        #[arg(long, short = 'p', default_value_t = default_port())]
        port: u16,
        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Pin a specific Codex client version (otherwise fetched from npm)
        #[arg(long)]
        codex_version: Option<String>,
    },
    /// Log in via OAuth PKCE browser flow
    Login,
    /// Remove stored credentials
    Logout,
    /// Authentication-related commands
    Auth {
        #[command(subcommand)]
        sub: AuthCommands,
    },
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Show current authentication status
    Status,
}

fn default_port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "codex_openai_proxy=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            port,
            host,
            codex_version,
        } => run_server(port, &host, codex_version).await,
        Commands::Login => run_login().await,
        Commands::Logout => run_logout().await,
        Commands::Auth { sub } => match sub {
            AuthCommands::Status => run_auth_status().await,
        },
    }
}

async fn run_server(port: u16, host: &str, codex_version: Option<String>) -> anyhow::Result<()> {
    let state = config::make_state(port, codex_version).await;
    config::spawn_version_refresher(state.clone());

    let app = axum::Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route(
            "/v1/models",
            axum::routing::get(models::handle_models),
        )
        .route(
            "/v1/responses",
            axum::routing::post(proxy::handle_responses),
        )
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat::handle_chat_completions),
        )
        .with_state(state);

    let addr = format!("{host}:{port}");
    info!("codex-openai-proxy listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

async fn run_login() -> anyhow::Result<()> {
    let tokens = auth::login_flow().await?;
    println!("Logged in successfully.");
    if let Some(ref aid) = tokens.account_id {
        println!("Account ID: {aid}");
    }
    Ok(())
}

async fn run_logout() -> anyhow::Result<()> {
    if let Some(tokens) = auth::AuthTokens::load() {
        if let Err(e) = auth::revoke_token(&tokens.access_token).await {
            info!("Token revocation returned an error (may be expected): {e}");
        }
    }
    auth::AuthTokens::delete()?;
    println!("Logged out successfully.");
    Ok(())
}

async fn run_auth_status() -> anyhow::Result<()> {
    match auth::AuthTokens::load() {
        Some(tokens) => {
            println!("Authenticated: yes");
            if let Some(ref aid) = tokens.account_id {
                println!("Account ID: {aid}");
            }
            let expired = tokens.is_expired();
            println!("Token expired: {expired}");
            if let Some(ref obtained) = tokens.obtained_at {
                println!("Obtained at: {obtained}");
            }
            if expired {
                if tokens.refresh_token.is_some() {
                    println!("Refresh token: present (will auto-refresh)");
                } else {
                    println!("Refresh token: none (please re-login)");
                }
            }
        }
        None => {
            println!("Authenticated: no");
            println!("Run `codex-openai-proxy login` to authenticate.");
        }
    }
    Ok(())
}
