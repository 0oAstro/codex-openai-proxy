use axum::response::IntoResponse;
use clap::{Parser, Subcommand};
use tracing::info;

mod auth;
mod chat;
mod config;
mod images;
mod models;
mod proxy;
mod usage;

#[derive(Parser)]
#[command(
    name = "codex-openai-proxy",
    version,
    about = "Proxy ChatGPT/Codex subscription as an OpenAI-compatible API"
)]
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
        /// Pin a specific Codex client version (otherwise fetched from npm/CODEX_CLIENT_VERSION)
        #[arg(long)]
        codex_version: Option<String>,
    },
    /// Log in via OAuth PKCE browser flow
    Login,
    /// Log in via device code (for headless/SSH)
    LoginDevice,
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
        Commands::LoginDevice => run_login_device().await,
        Commands::Logout => run_logout().await,
        Commands::Auth { sub } => match sub {
            AuthCommands::Status => run_auth_status().await,
        },
    }
}

/// API key auth middleware. If PROXY_API_KEY is set, validate Bearer token.
async fn auth_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    if let Some(expected) = config::proxy_api_key() {
        let provided = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match provided {
            Some(key) if key == expected => next.run(req).await,
            _ => axum::http::StatusCode::UNAUTHORIZED.into_response(),
        }
    } else {
        next.run(req).await
    }
}

/// CORS middleware. Handles OPTIONS preflight and adds CORS headers to all responses.
async fn cors_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::header;

    if req.method() == axum::http::Method::OPTIONS {
        return axum::http::StatusCode::NO_CONTENT.into_response();
    }

    let mut resp = next.run(req).await;
    let headers = resp.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    resp
}

async fn run_server(port: u16, host: &str, codex_version: Option<String>) -> anyhow::Result<()> {
    let state = config::make_state(port, codex_version).await;
    config::spawn_version_refresher(state.clone());

    if config::proxy_api_key().is_some() {
        info!("Proxy API key auth enabled");
    } else {
        info!("Proxy API key auth disabled (set PROXY_API_KEY to enable)");
    }

    let app = axum::Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route("/usage", axum::routing::get(usage::handle_usage))
        .route("/v1/models", axum::routing::get(models::handle_models))
        .route(
            "/v1/responses",
            axum::routing::post(proxy::handle_responses),
        )
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat::handle_chat_completions),
        )
        .route(
            "/v1/images/generations",
            axum::routing::post(images::handle_images_generations),
        )
        .route(
            "/v1/images/edits",
            axum::routing::post(images::handle_images_edits),
        )
        .layer(axum::middleware::from_fn(cors_middleware))
        .layer(axum::middleware::from_fn(auth_middleware))
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
    tokens.save_primary()?;
    println!("Logged in successfully.");
    if let Some(ref aid) = tokens.account_id {
        println!("Account ID: {aid}");
    }
    Ok(())
}

async fn run_login_device() -> anyhow::Result<()> {
    let tokens = auth::device_login_flow().await?;
    tokens.save_primary()?;
    println!("Logged in successfully via device code.");
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
    match auth::AuthTokens::load_all().first().cloned() {
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
