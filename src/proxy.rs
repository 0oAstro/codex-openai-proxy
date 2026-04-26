use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::stream::StreamExt;
use tracing::{debug, error, info};

use crate::config::AppState;
use crate::models::build_auth_headers;

/// Passthrough handler for `/v1/responses`.
///
/// Forwards the request body verbatim to the upstream Codex `/responses`
/// endpoint and streams the SSE response back to the client.
pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let auth = match load_and_refresh_auth().await {
        Ok(a) => a,
        Err(e) => {
            return (
                StatusCode::UNAUTHORIZED,
                format!("{{\"error\":{{\"message\":\"{e}\"}}}}"),
            )
                .into_response();
        }
    };

    let mut auth_headers = build_auth_headers(&auth, &state.codex_user_agent().await);
    // Copy selected headers from the incoming request (e.g. OpenAI-Beta).
    for key in &["openai-beta", "openai-organization"] {
        if let Some(val) = headers.get(*key) {
            auth_headers.insert(*key, val.clone());
        }
    }

    let url = format!("{}/responses", crate::config::UPSTREAM_BASE);
    debug!("Proxying to upstream: {url}");

    let resp = match state
        .http
        .post(&url)
        .headers(auth_headers.clone())
        .body(body.to_vec())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("Upstream request error: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                format!("{{\"error\":{{\"message\":\"Upstream request failed: {e}\"}}}}"),
            )
                .into_response();
        }
    };

    let status = resp.status();
    debug!("Upstream responded with {status}");

    // If we got a 401, attempt one token refresh then retry.
    if status == StatusCode::UNAUTHORIZED {
        info!("Received 401, attempting token refresh…");
        if let Some(tokens) = crate::auth::AuthTokens::load() {
            if let Some(rt) = tokens.refresh_token.clone() {
                match crate::auth::refresh_token(&rt).await {
                    Ok(refreshed) => {
                        let ua = state.codex_user_agent().await;
                        let mut retry_headers = build_auth_headers(&refreshed, &ua);
                        for key in &["openai-beta", "openai-organization"] {
                            if let Some(val) = headers.get(*key) {
                                retry_headers.insert(*key, val.clone());
                            }
                        }

                        let retry_resp = match state
                            .http
                            .post(&url)
                            .headers(retry_headers)
                            .body(body.to_vec())
                            .send()
                            .await
                        {
                            Ok(r) => r,
                            Err(e) => {
                                return (
                                    StatusCode::BAD_GATEWAY,
                                    format!("{{\"error\":{{\"message\":\"Retry upstream request failed: {e}\"}}}}"),
                                )
                                    .into_response();
                            }
                        };

                        return stream_response(retry_resp).await;
                    }
                    Err(e) => {
                        error!("Token refresh failed: {e}");
                    }
                }
            }
        }
    }

    stream_response(resp).await
}

/// Stream an upstream response back to the client, proxying the status code
/// and content-type, and forwarding the body as-is (SSE byte stream).
async fn stream_response(upstream: reqwest::Response) -> Response {
    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .cloned();

    let stream = upstream.bytes_stream().map(|result| {
        result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    });
    let body = Body::from_stream(stream);

    let mut builder = Response::builder().status(status);
    if let Some(ct) = content_type {
        builder = builder.header(reqwest::header::CONTENT_TYPE, ct);
    }
    match builder.body(body) {
        Ok(resp) => resp,
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to build streaming response: {e}"),
        )
            .into_response(),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

async fn load_and_refresh_auth() -> Result<crate::auth::AuthTokens, String> {
    let tokens = crate::auth::AuthTokens::load().ok_or_else(|| {
        "Not authenticated. Run `codex-openai-proxy login`.".to_string()
    })?;
    crate::auth::ensure_valid_token(tokens)
        .await
        .map_err(|e: anyhow::Error| e.to_string())
}
