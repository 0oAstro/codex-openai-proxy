use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{TimeDelta, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{error, info};

use crate::config::{
    auth_file_path, AUTH_URL, CLIENT_ID, REDIRECT_URI, REFRESH_MARGIN_SECS, REVOKE_URL, SCOPES,
    TOKEN_URL,
};

// ── Stored token data ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub token_type: Option<String>,
    pub expires_in: Option<i64>,
    /// ISO-8601 timestamp when the token was obtained / refreshed.
    pub obtained_at: Option<String>,
    /// Account id extracted from the JWT (sub claim).
    pub account_id: Option<String>,
}

impl AuthTokens {
    /// Return the auth file path.
    pub fn path() -> PathBuf {
        auth_file_path()
    }

    /// Load tokens from disk. Returns `None` if the file does not exist.
    pub fn load() -> Option<Self> {
        let p = Self::path();
        if !p.exists() {
            return None;
        }
        match fs::read_to_string(&p) {
            Ok(data) => parse_auth_tokens(&data).ok(),
            Err(e) => {
                error!("Failed to read auth file: {e}");
                None
            }
        }
    }

    /// Persist tokens to disk, creating parent directories as needed.
    pub fn save(&self) -> anyhow::Result<()> {
        let p = Self::path();
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }

        if let Ok(existing) = fs::read_to_string(&p) {
            if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&existing) {
                if value.get("tokens").is_some() {
                    value["tokens"]["access_token"] =
                        serde_json::Value::String(self.access_token.clone());
                    value["tokens"]["refresh_token"] =
                        serde_json::Value::String(self.refresh_token.clone().unwrap_or_default());
                    if let Some(id_token) = self.id_token.as_ref() {
                        value["tokens"]["id_token"] = serde_json::json!({"raw_jwt": id_token});
                    }
                    if let Some(account_id) = self.account_id.as_ref() {
                        value["tokens"]["account_id"] =
                            serde_json::Value::String(account_id.clone());
                    }
                    value["last_refresh"] = serde_json::Value::String(Utc::now().to_rfc3339());
                    fs::write(&p, serde_json::to_string_pretty(&value)?)?;
                    return Ok(());
                }
            }
        }

        fs::write(&p, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Remove the auth file.
    pub fn delete() -> anyhow::Result<()> {
        let p = Self::path();
        if p.exists() {
            fs::remove_file(&p)?;
        }
        Ok(())
    }

    /// Check whether the access token is expired (or will expire within
    /// `REFRESH_MARGIN_SECS`).
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = token_expiration(&self.access_token) {
            return Utc::now() + TimeDelta::seconds(REFRESH_MARGIN_SECS) >= exp;
        }

        let obtained = self
            .obtained_at
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok());
        let ttl = self.expires_in.unwrap_or(3600);
        match obtained {
            Some(t) => {
                let expires_at = t + TimeDelta::seconds(ttl);
                Utc::now() + TimeDelta::seconds(REFRESH_MARGIN_SECS) >= expires_at
            }
            None => true,
        }
    }
}

// ── JWT helpers ───────────────────────────────────────────────────────────

/// Decode the JWT payload **without** verifying the signature (we are only
/// interested in reading the `sub` claim to obtain the ChatGPT account id).
fn decode_jwt_payload(token: &str) -> Option<serde_json::Value> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = URL_SAFE_NO_PAD.decode(parts[1]).ok()?;
    serde_json::from_slice(&payload).ok()
}

/// Extract the account id from the JWT. Prefers the `sub` claim but also
/// checks `https://api.openai.com/auth`.
pub fn extract_account_id(token: &str) -> Option<String> {
    let claims = decode_jwt_payload(token)?;
    // Prefer the dedicated claim.
    if let Some(v) = claims
        .get("https://api.openai.com/auth")
        .and_then(|v| v.as_str())
    {
        return Some(v.to_string());
    }
    claims
        .get("sub")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn token_expiration(token: &str) -> Option<chrono::DateTime<Utc>> {
    let claims = decode_jwt_payload(token)?;
    let exp = claims.get("exp")?.as_i64()?;
    chrono::DateTime::from_timestamp(exp, 0)
}

fn parse_auth_tokens(data: &str) -> anyhow::Result<AuthTokens> {
    if let Ok(tokens) = serde_json::from_str::<AuthTokens>(data) {
        return Ok(tokens);
    }

    let value: serde_json::Value = serde_json::from_str(data)?;
    let tokens = value
        .get("tokens")
        .ok_or_else(|| anyhow::anyhow!("Missing tokens in auth file"))?;

    let access_token = tokens
        .get("access_token")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing access_token in auth file"))?
        .to_string();
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    let id_token = tokens
        .get("id_token")
        .and_then(|v| {
            v.as_str()
                .or_else(|| v.get("raw_jwt").and_then(|raw| raw.as_str()))
        })
        .map(ToString::to_string);
    let account_id = tokens
        .get("account_id")
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
        .or_else(|| extract_account_id(&access_token));

    Ok(AuthTokens {
        access_token,
        refresh_token,
        id_token,
        token_type: Some("Bearer".to_string()),
        expires_in: None,
        obtained_at: value
            .get("last_refresh")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
        account_id,
    })
}

// ── PKCE ──────────────────────────────────────────────────────────────────

pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
}

impl PkceChallenge {
    pub fn generate() -> Self {
        let mut bytes = [0u8; 64];
        rand::thread_rng().fill_bytes(&mut bytes);
        let verifier = URL_SAFE_NO_PAD.encode(bytes);

        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let hash = hasher.finalize();
        let challenge = URL_SAFE_NO_PAD.encode(hash);

        Self {
            verifier,
            challenge,
        }
    }
}

// ── OAuth flows ───────────────────────────────────────────────────────────

/// Perform the full PKCE browser-login flow:
/// 1. Generate PKCE challenge
/// 2. Open the browser
/// 3. Listen on localhost:1455 for the callback
/// 4. Exchange the code for tokens
/// 5. Persist tokens
pub async fn login_flow() -> anyhow::Result<AuthTokens> {
    let pkce = PkceChallenge::generate();
    let state = random_state();

    let auth_url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}&codex_cli_simplified_flow=true",
        AUTH_URL, CLIENT_ID, urlencoding::encode(REDIRECT_URI), urlencoding::encode(SCOPES), pkce.challenge, state,
    );

    println!("Opening browser for login…");
    if let Err(e) = open::that(&auth_url) {
        info!("Could not open browser: {e}. Visit:\n{auth_url}");
    }

    // Start a tiny HTTP server to catch the callback.
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    let tx = Arc::new(tokio::sync::Mutex::new(Some(tx)));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:1455").await?;
    info!("Listening for OAuth callback on http://localhost:1455");

    // Accept a single connection.
    let handle = tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 4096];
            let (mut read, mut write) = stream.into_split();
            let n = read.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);

            // Parse the first line: GET /auth/callback?code=xxx&state=yyy HTTP/1.1
            if let Some(line) = req.lines().next() {
                if let Some(query) = line.split('?').nth(1) {
                    let query = query.split(' ').next().unwrap_or("");
                    let mut code: Option<String> = None;
                    for pair in query.split('&') {
                        if let Some(v) = pair.strip_prefix("code=") {
                            code = Some(urlencoding::decode(v).unwrap_or_else(|_| v.into()).into());
                        }
                    }
                    if let Some(code) = code {
                        let tx = tx.lock().await.take();
                        if let Some(tx) = tx {
                            let _ = tx.send(code);
                        }
                    }
                }
            }

            let body = "<html><body><h2>Login successful!</h2><p>You can close this tab.</p><script>window.close()</script></body></html>";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = write.write_all(resp.as_bytes()).await;
        }
    });

    let code = rx
        .await
        .map_err(|_| anyhow::anyhow!("Callback server closed without receiving code"))?;
    handle.abort();

    info!("Received authorization code, exchanging for tokens…");
    let tokens = exchange_code(&code, &pkce.verifier).await?;
    info!("Login successful!");
    tokens.save()?;
    Ok(tokens)
}

/// Exchange an authorization code for tokens.
async fn exchange_code(code: &str, verifier: &str) -> anyhow::Result<AuthTokens> {
    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "grant_type=authorization_code&client_id={}&code={}&redirect_uri={}&code_verifier={}",
            CLIENT_ID,
            urlencoding::encode(code),
            urlencoding::encode(REDIRECT_URI),
            urlencoding::encode(verifier),
        ))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Token exchange failed ({status}): {body}");
    }

    let raw: serde_json::Value = resp.json().await?;
    parse_token_response(raw)
}

/// Refresh an access token using a refresh token.
pub async fn refresh_token(refresh: &str) -> anyhow::Result<AuthTokens> {
    info!("Refreshing access token…");
    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "grant_type=refresh_token&client_id={}&refresh_token={}",
            CLIENT_ID,
            urlencoding::encode(refresh),
        ))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Token refresh failed ({status}): {body}");
    }

    let raw: serde_json::Value = resp.json().await?;
    let tokens = parse_token_response(raw)?;
    tokens.save()?;
    Ok(tokens)
}

/// Revoke the current token.
pub async fn revoke_token(token: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .post(REVOKE_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "client_id={}&token={}",
            CLIENT_ID,
            urlencoding::encode(token),
        ))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Token revocation failed ({status}): {body}");
    }
    Ok(())
}

/// Ensure we have a valid (non-expired) token, refreshing if needed.
/// Returns the refreshed-or-existing `AuthTokens`.
pub async fn ensure_valid_token(tokens: AuthTokens) -> anyhow::Result<AuthTokens> {
    if !tokens.is_expired() {
        return Ok(tokens);
    }
    match tokens.refresh_token.as_deref() {
        Some(rt) => refresh_token(rt).await,
        None => anyhow::bail!(
            "Token expired and no refresh token available. Please run `codex-openai-proxy login`."
        ),
    }
}

// ── internals ─────────────────────────────────────────────────────────────

// ── Device code login (for headless/SSH) ──────────────────────────────────

const DEVICE_AUTH_BASE: &str = "https://auth.openai.com/api/accounts";
const DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";

/// Perform device code login (for headless environments).
/// 1. Request a user code from the device auth endpoint
/// 2. Print the URL and code for the user to visit
/// 3. Poll until the user authorizes
/// 4. Exchange the resulting code for tokens
pub async fn device_login_flow() -> anyhow::Result<AuthTokens> {
    let client = reqwest::Client::new();

    // Step 1: Request user code
    info!("Requesting device code…");
    let resp = client
        .post(format!("{DEVICE_AUTH_BASE}/deviceauth/usercode"))
        .json(&serde_json::json!({"client_id": CLIENT_ID}))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Device code request failed ({status}): {body}");
    }

    let dc: DeviceCodeResp = resp.json().await?;
    println!(
        "\nOpen this URL in any browser:\n  https://auth.openai.com/codex/device\n\nEnter this code:\n  {}\n\n(expires in 15 minutes)",
        dc.user_code
    );

    // Step 2: Poll for authorization
    let poll_interval = std::time::Duration::from_secs(dc.interval.max(5));
    let max_wait = std::time::Duration::from_secs(15 * 60);
    let start = std::time::Instant::now();

    let code_resp = loop {
        if start.elapsed() >= max_wait {
            anyhow::bail!("Device code timed out after 15 minutes");
        }

        tokio::time::sleep(poll_interval).await;

        let poll_resp = client
            .post(format!("{DEVICE_AUTH_BASE}/deviceauth/token"))
            .json(&serde_json::json!({
                "device_auth_id": dc.device_auth_id,
                "user_code": dc.user_code,
            }))
            .send()
            .await?;

        if poll_resp.status().is_success() {
            break poll_resp.json::<DeviceCodeSuccessResp>().await?;
        }

        // 403/404 = still pending, keep polling
        let status = poll_resp.status();
        if status != reqwest::StatusCode::FORBIDDEN && status != reqwest::StatusCode::NOT_FOUND {
            let body = poll_resp.text().await.unwrap_or_default();
            anyhow::bail!("Device auth failed ({status}): {body}");
        }
    };

    info!("Device code authorized, exchanging for tokens…");

    // Step 3: Exchange authorization_code via standard PKCE token exchange
    // The device auth flow returns authorization_code + PKCE codes server-side
    let redirect_uri = format!("{DEVICE_AUTH_BASE}/deviceauth/callback");
    let token_resp = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "grant_type=authorization_code&client_id={}&code={}&redirect_uri={}&code_verifier={}",
            CLIENT_ID,
            urlencoding::encode(&code_resp.authorization_code),
            urlencoding::encode(&redirect_uri),
            urlencoding::encode(&code_resp.code_verifier),
        ))
        .send()
        .await?;

    if !token_resp.status().is_success() {
        let status = token_resp.status();
        let body = token_resp.text().await.unwrap_or_default();
        anyhow::bail!("Token exchange failed ({status}): {body}");
    }

    let raw: serde_json::Value = token_resp.json().await?;
    let tokens = parse_token_response(raw)?;
    tokens.save()?;
    info!("Device code login successful!");
    Ok(tokens)
}

#[derive(Deserialize)]
struct DeviceCodeResp {
    user_code: String,
    device_auth_id: String,
    #[serde(
        default = "default_interval",
        deserialize_with = "deserialize_interval"
    )]
    interval: u64,
}

fn default_interval() -> u64 {
    5
}

fn deserialize_interval<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct IntervalVisitor;

    impl serde::de::Visitor<'_> for IntervalVisitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a u64 or a string containing a u64")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            u64::try_from(value).map_err(|_| E::custom("interval cannot be negative"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            value
                .parse::<u64>()
                .map_err(|_| E::custom("interval string must contain a u64"))
        }
    }

    deserializer.deserialize_any(IntervalVisitor)
}

#[derive(Deserialize)]
struct DeviceCodeSuccessResp {
    authorization_code: String,
    code_challenge: String,
    code_verifier: String,
}

fn parse_token_response(raw: serde_json::Value) -> anyhow::Result<AuthTokens> {
    let access_token = raw["access_token"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing access_token in token response"))?
        .to_string();
    let account_id = extract_account_id(&access_token);

    Ok(AuthTokens {
        access_token,
        refresh_token: raw["refresh_token"].as_str().map(|s| s.to_string()),
        id_token: raw["id_token"].as_str().map(|s| s.to_string()),
        token_type: raw["token_type"].as_str().map(|s| s.to_string()),
        expires_in: raw["expires_in"].as_i64(),
        obtained_at: Some(Utc::now().to_rfc3339()),
        account_id,
    })
}

fn random_state() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}
