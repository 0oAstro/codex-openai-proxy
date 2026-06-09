use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream, StreamExt};
use serde_json::Value;
use tracing::{debug, error, info};

use crate::config::AppState;
use crate::models::{build_auth_headers, copy_codex_passthrough_headers};

/// Passthrough handler for `/v1/responses`.
///
/// Forwards the request body verbatim to the upstream Codex `/responses`
/// endpoint and streams the SSE response back to the client.
pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let (upstream_body, stream_requested) = sanitize_responses_body(&body);

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

    let user_agent = state.codex_user_agent().await;
    let version = state.client_version().await;
    let mut auth_headers = build_auth_headers(&auth, &user_agent, &version);
    // Always send OpenAI-Beta header for the Responses API.
    auth_headers.insert(
        "openai-beta",
        "responses=experimental"
            .parse()
            .expect("valid header value"),
    );
    copy_codex_passthrough_headers(&headers, &mut auth_headers);

    let url = format!("{}/responses", crate::config::UPSTREAM_BASE);
    debug!("Proxying to upstream: {url}");

    let resp = match state
        .http
        .post(&url)
        .headers(auth_headers.clone())
        .body(upstream_body.clone())
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
                        let version = state.client_version().await;
                        let mut retry_headers = build_auth_headers(&refreshed, &ua, &version);
                        retry_headers.insert(
                            "openai-beta",
                            "responses=experimental"
                                .parse()
                                .expect("valid header value"),
                        );
                        copy_codex_passthrough_headers(&headers, &mut retry_headers);

                        let retry_resp = match state
                            .http
                            .post(&url)
                            .headers(retry_headers)
                            .body(upstream_body.clone())
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

                        return stream_response(retry_resp, stream_requested).await;
                    }
                    Err(e) => {
                        error!("Token refresh failed: {e}");
                    }
                }
            }
        }
    }

    stream_response(resp, stream_requested).await
}

/// Stream an upstream response back to the client, proxying the status code
/// and content-type, and forwarding the body as-is (SSE byte stream).
async fn stream_response(upstream: reqwest::Response, stream_requested: bool) -> Response {
    let status = upstream.status();

    if !stream_requested {
        // Client wants a non-streaming response, but upstream always streams.
        // Collect the SSE into a single JSON response.
        return collect_sse_to_json(upstream, status).await;
    }

    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .cloned();

    let stream = transform_response_sse_stream(upstream.bytes_stream());
    let body = Body::from_stream(stream);

    let mut builder = Response::builder().status(status);
    if let Some(ct) = content_type {
        builder = builder.header(reqwest::header::CONTENT_TYPE, ct);
    } else {
        builder = builder.header(reqwest::header::CONTENT_TYPE, "text/event-stream");
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

struct ResponseSseTransformState {
    upstream: Pin<Box<dyn Stream<Item = reqwest::Result<axum::body::Bytes>> + Send>>,
    buffer: String,
    output_items: Vec<Value>,
    pending: VecDeque<Result<axum::body::Bytes, std::io::Error>>,
    upstream_done: bool,
}

fn transform_response_sse_stream<S>(
    upstream: S,
) -> impl Stream<Item = Result<axum::body::Bytes, std::io::Error>>
where
    S: Stream<Item = reqwest::Result<axum::body::Bytes>> + Send + 'static,
{
    let state = ResponseSseTransformState {
        upstream: Box::pin(upstream),
        buffer: String::new(),
        output_items: Vec::new(),
        pending: VecDeque::new(),
        upstream_done: false,
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(item) = state.pending.pop_front() {
                return Some((item, state));
            }

            if state.upstream_done {
                if !state.buffer.is_empty() {
                    let frame = std::mem::take(&mut state.buffer);
                    let transformed = transform_response_sse_frame(&frame, &mut state.output_items);
                    return Some((Ok(axum::body::Bytes::from(transformed)), state));
                }
                return None;
            }

            match state.upstream.next().await {
                Some(Ok(bytes)) => {
                    state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                    drain_complete_sse_frames(&mut state);
                }
                Some(Err(e)) => {
                    return Some((
                        Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
                        state,
                    ));
                }
                None => {
                    state.upstream_done = true;
                }
            }
        }
    })
}

fn drain_complete_sse_frames(state: &mut ResponseSseTransformState) {
    while let Some((frame_len, sep_len)) = next_sse_frame(&state.buffer) {
        let rest = state.buffer.split_off(frame_len + sep_len);
        state.buffer.truncate(frame_len);
        let frame = std::mem::replace(&mut state.buffer, rest);
        let transformed = transform_response_sse_frame(&frame, &mut state.output_items);
        state
            .pending
            .push_back(Ok(axum::body::Bytes::from(transformed)));
    }
}

fn next_sse_frame(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\n\n"), buffer.find("\r\n\r\n")) {
        (Some(lf), Some(crlf)) if crlf < lf => Some((crlf, 4)),
        (Some(lf), _) => Some((lf, 2)),
        (None, Some(crlf)) => Some((crlf, 4)),
        (None, None) => None,
    }
}

fn transform_response_sse_frame(frame: &str, output_items: &mut Vec<Value>) -> String {
    let mut transformed_lines = Vec::new();

    for raw_line in frame.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some(data) = line.strip_prefix("data: ") else {
            transformed_lines.push(line.to_string());
            continue;
        };

        if data.trim() == "[DONE]" {
            transformed_lines.push(line.to_string());
            continue;
        }

        let mut event: Value = match serde_json::from_str(data.trim()) {
            Ok(v) => v,
            Err(_) => {
                transformed_lines.push(line.to_string());
                continue;
            }
        };

        if event.get("type").and_then(Value::as_str) == Some("response.output_item.done") {
            if let Some(item) = event.get("item").cloned() {
                output_items.push(item);
            }
        }

        if event.get("type").and_then(Value::as_str) == Some("response.completed") {
            if let Some(resp) = event.get_mut("response") {
                let needs_output = resp
                    .get("output")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty);
                if needs_output && !output_items.is_empty() {
                    if let Some(obj) = resp.as_object_mut() {
                        obj.insert("output".to_string(), Value::Array(output_items.clone()));
                    }
                }
            }
        }

        transformed_lines.push(format!(
            "data: {}",
            serde_json::to_string(&event).unwrap_or_else(|_| data.to_string())
        ));
    }

    transformed_lines.join("\n") + "\n\n"
}

/// Collect an SSE stream from upstream into a single JSON response.
/// This is used when the client requested a non-streaming response but
/// upstream always streams.
async fn collect_sse_to_json(upstream: reqwest::Response, status: StatusCode) -> Response {
    let raw = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read upstream SSE body: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"message": format!("Failed to read upstream: {e}")}})),
            )
                .into_response();
        }
    };

    let text = String::from_utf8_lossy(&raw);
    let mut last_response: Option<Value> = None;
    let mut output_items: Vec<Value> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let data = match line.strip_prefix("data: ") {
            Some(d) => d.trim(),
            None => continue,
        };
        if data == "[DONE]" {
            continue;
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // ChatGPT Codex streaming responses no longer include the accumulated
        // `output` array on the final `response.completed.response` object.
        // OpenAI-compatible non-streaming Responses clients still expect it,
        // so reconstruct it from completed output-item events while collecting
        // the stream.
        if event.get("type").and_then(Value::as_str) == Some("response.output_item.done") {
            if let Some(item) = event.get("item").cloned() {
                output_items.push(item);
            }
        }

        // Track the latest `response` object from any event that carries it.
        if let Some(resp) = event.get("response").cloned() {
            last_response = Some(resp);
        }
    }

    match last_response {
        Some(mut resp) => {
            let needs_output = resp
                .get("output")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty);
            if needs_output && !output_items.is_empty() {
                if let Some(obj) = resp.as_object_mut() {
                    obj.insert("output".to_string(), Value::Array(output_items));
                }
            }

            let mut builder = Response::builder().status(status);
            builder = builder.header(reqwest::header::CONTENT_TYPE, "application/json");
            match builder.body(axum::body::Body::from(
                serde_json::to_vec(&resp).unwrap_or_default(),
            )) {
                Ok(r) => r,
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to build response: {e}"),
                )
                    .into_response(),
            }
        }
        None => (
            status,
            Json(serde_json::json!({"error": {"message": "Upstream did not return a response"}})),
        )
            .into_response(),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

/// Codex upstream does not support OpenAI `safety_identifier`; drop it while
/// preserving all other request fields verbatim. Also ensures `instructions`
/// and `store` fields have sane defaults, and always forces `stream: true`
/// since the Codex API requires streaming.
fn sanitize_responses_body(body: &[u8]) -> (Vec<u8>, bool) {
    let mut parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return (body.to_vec(), false),
    };

    let stream_requested = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if let Some(obj) = parsed.as_object_mut() {
        obj.remove("safety_identifier");
        obj.entry("instructions")
            .or_insert_with(|| Value::String(String::new()));
        if !obj.contains_key("store") {
            obj.insert("store".to_string(), Value::Bool(false));
        }
        // Always force stream on for upstream (Codex requires it).
        // For non-streaming clients, the SSE is collected.
        obj.insert("stream".to_string(), Value::Bool(true));
    }

    (
        serde_json::to_vec(&parsed).unwrap_or_else(|_| body.to_vec()),
        stream_requested,
    )
}

async fn load_and_refresh_auth() -> Result<crate::auth::AuthTokens, String> {
    let tokens = crate::auth::AuthTokens::load()
        .ok_or_else(|| "Not authenticated. Run `codex-openai-proxy login`.".to_string())?;
    crate::auth::ensure_valid_token(tokens)
        .await
        .map_err(|e: anyhow::Error| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn collect_sse_to_json_reconstructs_missing_completed_output() {
        let app = axum::Router::new().route(
            "/sse",
            axum::routing::get(|| async {
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    concat!(
                        "event: response.output_item.done\n",
                        r#"data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}}"#,
                        "\n\n",
                        "event: response.completed\n",
                        r#"data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output":[]}}"#,
                        "\n\n",
                    ),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let upstream = reqwest::get(format!("http://{addr}/sse")).await.unwrap();
        let response = collect_sse_to_json(upstream, StatusCode::OK).await;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["id"], "resp_1");
        assert_eq!(parsed["output"][0]["id"], "msg_1");
        assert_eq!(parsed["output"][0]["content"][0]["text"], "ok");
    }

    #[test]
    fn streaming_transform_injects_output_into_completed_event() {
        let mut output_items = Vec::new();

        let item_frame = concat!(
            "event: response.output_item.done\n",
            r#"data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}}"#,
        );
        let completed_frame = concat!(
            "event: response.completed\n",
            r#"data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed"}}"#,
        );

        let item_out = transform_response_sse_frame(item_frame, &mut output_items);
        let completed_out = transform_response_sse_frame(completed_frame, &mut output_items);

        let completed_data = completed_out
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("completed data line");
        let completed: Value = serde_json::from_str(completed_data).unwrap();

        assert!(item_out.contains("response.output_item.done"));
        assert_eq!(completed["response"]["output"][0]["id"], "msg_1");
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "ok"
        );
    }

    #[test]
    fn sanitize_responses_body_adds_required_instructions_and_store() {
        let (body, stream_requested) = sanitize_responses_body(
            br#"{"model":"gpt-5.5","input":[],"stream":true,"safety_identifier":"sid"}"#,
        );
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert!(stream_requested);
        assert_eq!(parsed["instructions"], Value::String(String::new()));
        assert!(parsed.get("safety_identifier").is_none());
        assert_eq!(parsed["store"], false);
    }

    #[test]
    fn sanitize_responses_body_preserves_existing_instructions() {
        let (body, _) =
            sanitize_responses_body(br#"{"model":"gpt-5.5","instructions":"be direct"}"#);
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["instructions"], "be direct");
    }

    #[test]
    fn sanitize_responses_body_preserves_existing_store() {
        let (body, _) = sanitize_responses_body(br#"{"model":"gpt-5.5","store":true}"#);
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["store"], true);
    }
}
