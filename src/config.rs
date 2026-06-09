use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::http::HeaderValue;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// OAuth / API constants derived from the official codex-rs source.
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const AUTH_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const REVOKE_URL: &str = "https://auth.openai.com/oauth/revoke";
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const UPSTREAM_BASE: &str = "https://chatgpt.com/backend-api/codex";
pub const SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
/// Auth file lives directly at `~/auth.json`.
pub const AUTH_FILE: &str = "auth.json";
pub const AUTH_FILE_ENV: &str = "CODEX_AUTH_FILE";

/// Fallback version if dynamic fetch fails.
pub const FALLBACK_CLIENT_VERSION: &str = "0.125.0";

/// How often to refresh the version from npm in the background.
const VERSION_REFRESH_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60); // 6 hours

/// Seconds before token expiry at which we trigger a refresh.
pub const REFRESH_MARGIN_SECS: i64 = 300; // 5 minutes

/// How long the models cache is valid (seconds).
pub const MODELS_CACHE_TTL_SECS: u64 = 300; // 5 minutes

/// Reasoning effort levels recognised in model-name suffixes.
pub const REASONING_EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh"];

/// Built-in models to force-inject into the models list (not returned by upstream).
pub const BUILTIN_MODELS: &[&str] = &[
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
];

/// Read the proxy API key from the `PROXY_API_KEY` env var.
/// If set, all requests (except /health) require `Authorization: Bearer <key>`.
pub fn proxy_api_key() -> Option<String> {
    std::env::var("PROXY_API_KEY").ok()
}

/// Resolve the ChatGPT OAuth token file.
///
/// `CODEX_AUTH_FILE` is useful in containers where credentials should live in
/// a mounted application directory instead of whatever `$HOME` happens to be.
pub fn auth_file_path() -> PathBuf {
    if let Ok(path) = std::env::var(AUTH_FILE_ENV) {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            return path;
        }
        return dirs_home().join(path);
    }

    dirs_home().join(AUTH_FILE)
}

/// Originator used by the official Codex CLI for ChatGPT backend calls.
pub const ORIGINATOR: &str = "codex_cli_rs";

/// Build a Codex-compatible User-Agent. Matches the current Rust Codex client
/// shape closely enough for ChatGPT backend feature gates while remaining
/// deterministic in containers.
pub fn codex_user_agent_for_version(version: &str) -> String {
    let os = os_info::get();
    let os_type = os.os_type().to_string();
    let os_version = os.version().to_string();
    let arch = os.architecture().unwrap_or("unknown");
    let terminal = terminal_user_agent_token();
    sanitize_user_agent(format!(
        "{ORIGINATOR}/{version} ({os_type} {os_version}; {arch}) {terminal}"
    ))
}

fn terminal_user_agent_token() -> String {
    if let Ok(term_program) = std::env::var("TERM_PROGRAM") {
        if !term_program.trim().is_empty() {
            if let Ok(version) = std::env::var("TERM_PROGRAM_VERSION") {
                if !version.trim().is_empty() {
                    return sanitize_header_token(format!("{term_program}/{version}"));
                }
            }
            return sanitize_header_token(term_program);
        }
    }

    if let Ok(version) = std::env::var("WEZTERM_VERSION") {
        if !version.trim().is_empty() {
            return sanitize_header_token(format!("WezTerm/{version}"));
        }
    }

    if let Ok(term) = std::env::var("TERM") {
        if !term.trim().is_empty() {
            return sanitize_header_token(term);
        }
    }

    "unknown".to_string()
}

fn sanitize_header_token(value: String) -> String {
    value
        .chars()
        .map(|ch| if matches!(ch, ' '..='~') { ch } else { '_' })
        .collect()
}

fn sanitize_user_agent(candidate: String) -> String {
    if HeaderValue::from_str(&candidate).is_ok() {
        return candidate;
    }
    let sanitized = sanitize_header_token(candidate);
    if HeaderValue::from_str(&sanitized).is_ok() {
        sanitized
    } else {
        ORIGINATOR.to_string()
    }
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("~"))
}

// ── Shared application state ──────────────────────────────────────────────

#[derive(Debug)]
pub struct AppState {
    pub http: reqwest::Client,
    pub port: u16,
    pub models_cache: Arc<crate::models::ModelsCache>,
    /// Dynamically updated Codex client version.
    client_version: Arc<RwLock<String>>,
    /// Dynamically updated User-Agent string.
    codex_user_agent: Arc<RwLock<String>>,
}

impl AppState {
    pub fn new(port: u16, version: String) -> Self {
        let user_agent = codex_user_agent_for_version(&version);
        let http = reqwest::Client::builder()
            .user_agent(&user_agent)
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            port,
            models_cache: Arc::new(crate::models::ModelsCache::new()),
            client_version: Arc::new(RwLock::new(version)),
            codex_user_agent: Arc::new(RwLock::new(user_agent)),
        }
    }

    pub async fn client_version(&self) -> String {
        self.client_version.read().await.clone()
    }

    pub async fn codex_user_agent(&self) -> String {
        self.codex_user_agent.read().await.clone()
    }
}

/// Fetch the latest Codex CLI version from npm registry.
/// Falls back to FALLBACK_CLIENT_VERSION on any error.
pub async fn fetch_latest_codex_version() -> String {
    let client = reqwest::Client::new();
    match client
        .get("https://registry.npmjs.org/@openai/codex/latest")
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(version) = json.get("version").and_then(|v| v.as_str()) {
                    return version.to_string();
                }
            }
        }
        Ok(resp) => {
            warn!("npm registry returned status {}", resp.status());
        }
        Err(e) => {
            warn!("Failed to fetch Codex version from npm: {e}");
        }
    }
    FALLBACK_CLIENT_VERSION.to_string()
}

/// Update the version stored in AppState (called by background task).
async fn update_version(state: &Arc<AppState>) {
    let version = fetch_latest_codex_version().await;
    let user_agent = codex_user_agent_for_version(&version);
    {
        let mut cv = state.client_version.write().await;
        *cv = version.clone();
    }
    {
        let mut ua = state.codex_user_agent.write().await;
        *ua = user_agent;
    }
    info!("Updated Codex client version to {version}");
}

/// Spawn a background task that periodically refreshes the version.
pub fn spawn_version_refresher(state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(VERSION_REFRESH_INTERVAL).await;
            update_version(&state).await;
        }
    });
}

/// Build AppState with dynamically fetched client version.
pub async fn make_state(port: u16, version_override: Option<String>) -> Arc<AppState> {
    let version = match version_override {
        Some(v) => {
            info!("Using pinned Codex client version: {v}");
            v
        }
        None => {
            if let Ok(v) = std::env::var("CODEX_CLIENT_VERSION") {
                info!("Using pinned Codex client version from CODEX_CLIENT_VERSION: {v}");
                v
            } else {
                let v = fetch_latest_codex_version().await;
                info!("Using Codex client version: {v}");
                v
            }
        }
    };
    Arc::new(AppState::new(port, version))
}
