use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::config::{AppState, UPSTREAM_BASE};

// ── Response types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelList {
    pub object: &'static str,
    pub data: Vec<OpenAIModel>,
}

// ── Upstream types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct UpstreamModel {
    slug: String,
    #[allow(dead_code)]
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpstreamModelResponse {
    models: Vec<UpstreamModel>,
}

// ── Cache ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct CacheEntry {
    data: Vec<OpenAIModel>,
    fetched_at: Instant,
}

#[derive(Debug)]
pub struct ModelsCache {
    entry: RwLock<Option<CacheEntry>>,
    ttl: Duration,
}

impl ModelsCache {
    pub fn new() -> Self {
        Self {
            entry: RwLock::new(None),
            ttl: Duration::from_secs(crate::config::MODELS_CACHE_TTL_SECS),
        }
    }

    pub async fn get_or_fetch(
        &self,
        state: &Arc<AppState>,
        auth_headers: &HeaderMap,
    ) -> Result<Vec<OpenAIModel>, String> {
        {
            let guard = self.entry.read().await;
            if let Some(ref cached) = *guard {
                if cached.fetched_at.elapsed() < self.ttl {
                    return Ok(cached.data.clone());
                }
            }
        }

        // Cache miss or expired – fetch from upstream.
        let version = state.client_version().await;
        let url = format!("{}/models?client_version={version}", UPSTREAM_BASE);
        debug!("Fetching models from upstream: {url}");

        let resp = state
            .http
            .get(&url)
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| format!("Upstream request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!("Upstream models request failed ({status}): {body}");
            return Err(format!("Upstream returned {status}"));
        }

        let upstream: UpstreamModelResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse upstream models response: {e}"))?;

        let now = Instant::now();
        let mut data: Vec<OpenAIModel> = upstream
            .models
            .into_iter()
            .map(|m| OpenAIModel {
                id: m.slug,
                object: "model",
                created: 1_700_000_000,
                owned_by: "openai",
            })
            .collect();

        // Force-inject built-in models (e.g. gpt-image-2) that upstream may not list.
        let existing_ids: std::collections::HashSet<String> =
            data.iter().map(|m| m.id.clone()).collect();
        for id in crate::config::BUILTIN_MODELS {
            if !existing_ids.contains(*id) {
                data.push(OpenAIModel {
                    id: id.to_string(),
                    object: "model",
                    created: 1_700_000_000,
                    owned_by: "openai",
                });
            }
        }

        let result = data.clone();
        {
            let mut guard = self.entry.write().await;
            *guard = Some(CacheEntry {
                data,
                fetched_at: now,
            });
        }

        Ok(result)
    }
}

// ── Handler ───────────────────────────────────────────────────────────────

pub async fn handle_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<OpenAIModelList>, (axum::http::StatusCode, String)> {
    let auth = crate::auth::AuthTokens::load().ok_or_else(|| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            "Not authenticated. Run `codex-openai-proxy login`.".into(),
        )
    })?;
    let auth: crate::auth::AuthTokens = crate::auth::ensure_valid_token(auth)
        .await
        .map_err(|e: anyhow::Error| (axum::http::StatusCode::UNAUTHORIZED, e.to_string()))?;

    let user_agent = state.codex_user_agent().await;
    let auth_headers = build_auth_headers(&auth, &user_agent);

    let models = state
        .models_cache
        .get_or_fetch(&state, &auth_headers)
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e))?;

    Ok(Json(OpenAIModelList {
        object: "list",
        data: models,
    }))
}

// ── helpers ───────────────────────────────────────────────────────────────

pub fn build_auth_headers(tokens: &crate::auth::AuthTokens, user_agent: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::AUTHORIZATION,
        format!("Bearer {}", tokens.access_token)
            .parse()
            .expect("valid header value"),
    );
    if let Some(ref aid) = tokens.account_id {
        headers.insert(
            "chatgpt-account-id",
            aid.parse().expect("valid header value"),
        );
    }
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        "application/json".parse().expect("valid header value"),
    );
    headers.insert(
        axum::http::header::USER_AGENT,
        user_agent.parse().expect("valid header value"),
    );
    headers.insert(
        "originator",
        "codex-tui".parse().expect("valid header value"),
    );
    headers
}
