use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use base64::Engine;
use serde_json::Value;
use tracing::{debug, error, info};

use crate::config::AppState;
use crate::models::build_auth_headers;

const DEFAULT_IMAGES_MAIN_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_IMAGES_TOOL_MODEL: &str = "gpt-image-2";

// ── Request types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ImagesGenerationsRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub n: Option<u64>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default)]
    pub output_compression: Option<u64>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub partial_images: Option<u64>,
    #[serde(default)]
    pub moderation: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ImagesEditsJsonRequest {
    pub prompt: String,
    pub images: Vec<InputImage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default)]
    pub output_compression: Option<u64>,
    #[serde(default)]
    pub input_fidelity: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub partial_images: Option<u64>,
    #[serde(default)]
    pub moderation: Option<String>,
    #[serde(default)]
    pub mask: Option<MaskInput>,
}

#[derive(Debug, Deserialize)]
pub struct InputImage {
    pub image_url: String,
}

#[derive(Debug, Deserialize)]
pub struct MaskInput {
    pub image_url: Option<String>,
}

// ── Response types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct ImagesApiResponse {
    created: u64,
    data: Vec<ImageData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
}

// ── Image generation call result (from SSE) ──────────────────────────────

#[derive(Debug)]
struct ImageCallResult {
    result: String,
    revised_prompt: String,
    output_format: String,
    size: String,
    background: String,
    quality: String,
}

// ── Shared logic ─────────────────────────────────────────────────────────

fn resolve_image_model(requested: Option<&str>) -> String {
    match requested {
        Some(m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => DEFAULT_IMAGES_TOOL_MODEL.to_string(),
    }
}

fn resolve_main_model(image_model: &str) -> String {
    if let Some(idx) = image_model.rfind('/') {
        let prefix = image_model[..idx].trim();
        if !prefix.is_empty() {
            return format!("{}/{}", prefix, DEFAULT_IMAGES_MAIN_MODEL);
        }
    }
    DEFAULT_IMAGES_MAIN_MODEL.to_string()
}

fn resolve_response_format(fmt: Option<&str>) -> String {
    match fmt {
        Some(f) if !f.trim().is_empty() => f.trim().to_lowercase(),
        _ => "b64_json".to_string(),
    }
}

fn build_tool_json(action: &str, image_model: &str, params: &ImageToolParams) -> Value {
    let mut tool = serde_json::json!({
        "type": "image_generation",
        "action": action,
        "model": image_model,
    });

    if let Some(ref v) = params.size {
        tool["size"] = Value::String(v.clone());
    }
    if let Some(ref v) = params.quality {
        tool["quality"] = Value::String(v.clone());
    }
    if let Some(ref v) = params.background {
        tool["background"] = Value::String(v.clone());
    }
    if let Some(ref v) = params.output_format {
        tool["output_format"] = Value::String(v.clone());
    }
    if let Some(v) = params.output_compression {
        tool["output_compression"] = Value::Number(v.into());
    }
    if let Some(v) = params.partial_images {
        tool["partial_images"] = Value::Number(v.into());
    }
    if let Some(ref v) = params.input_fidelity {
        tool["input_fidelity"] = Value::String(v.clone());
    }
    if let Some(ref v) = params.moderation {
        tool["moderation"] = Value::String(v.clone());
    }
    if let Some(ref mask_url) = params.mask_image_url {
        tool["input_image_mask"] = serde_json::json!({"image_url": mask_url});
    }

    tool
}

struct ImageToolParams {
    size: Option<String>,
    quality: Option<String>,
    background: Option<String>,
    output_format: Option<String>,
    output_compression: Option<u64>,
    partial_images: Option<u64>,
    input_fidelity: Option<String>,
    moderation: Option<String>,
    mask_image_url: Option<String>,
}

fn build_responses_request(
    prompt: &str,
    images: &[String],
    tool_json: &Value,
    main_model: &str,
) -> Value {
    let mut content: Vec<Value> = vec![serde_json::json!({
        "type": "input_text",
        "text": prompt,
    })];

    for img_url in images {
        if !img_url.trim().is_empty() {
            content.push(serde_json::json!({
                "type": "input_image",
                "image_url": img_url,
            }));
        }
    }

    serde_json::json!({
        "model": main_model,
        "instructions": "",
        "input": [{
            "type": "message",
            "role": "user",
            "content": content,
        }],
        "tools": [tool_json],
        "tool_choice": {"type": "image_generation"},
        "parallel_tool_calls": true,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "stream": true,
        "store": false,
    })
}

fn extract_images_from_completed(payload: &Value) -> Vec<ImageCallResult> {
    let output = match payload.get("response").and_then(|r| r.get("output")) {
        Some(o) => o,
        None => return vec![],
    };

    let arr = match output.as_array() {
        Some(a) => a,
        None => return vec![],
    };

    arr.iter()
        .filter(|item| item.get("type").and_then(|t| t.as_str()) == Some("image_generation_call"))
        .filter_map(|item| {
            let result = item.get("result")?.as_str()?.to_string();
            if result.trim().is_empty() {
                return None;
            }
            Some(ImageCallResult {
                result,
                revised_prompt: item
                    .get("revised_prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                output_format: item
                    .get("output_format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                size: item
                    .get("size")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                background: item
                    .get("background")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                quality: item
                    .get("quality")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            })
        })
        .collect()
}

fn extract_usage(payload: &Value) -> Option<Value> {
    payload
        .get("response")
        .and_then(|r| r.get("tool_usage"))
        .and_then(|t| t.get("image_gen"))
        .cloned()
}

fn mime_type_from_output_format(fmt: &str) -> &'static str {
    match fmt.to_lowercase().trim() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

fn build_images_response(
    results: &[ImageCallResult],
    response_format: &str,
    created: u64,
    usage: Option<Value>,
) -> Value {
    let first = match results.first() {
        Some(f) => f,
        None => {
            return serde_json::json!({
                "created": created,
                "data": [],
            });
        }
    };

    let data: Vec<Value> = results
        .iter()
        .map(|img| {
            let mut item = serde_json::json!({});
            if response_format == "url" {
                let mt = mime_type_from_output_format(&img.output_format);
                item["url"] = Value::String(format!("data:{};base64,{}", mt, img.result));
            } else {
                item["b64_json"] = Value::String(img.result.clone());
            }
            if !img.revised_prompt.is_empty() {
                item["revised_prompt"] = Value::String(img.revised_prompt.clone());
            }
            item
        })
        .collect();

    let mut resp = serde_json::json!({
        "created": created,
        "data": data,
    });

    if !first.background.is_empty() {
        resp["background"] = Value::String(first.background.clone());
    }
    if !first.output_format.is_empty() {
        resp["output_format"] = Value::String(first.output_format.clone());
    }
    if !first.quality.is_empty() {
        resp["quality"] = Value::String(first.quality.clone());
    }
    if !first.size.is_empty() {
        resp["size"] = Value::String(first.size.clone());
    }
    if let Some(u) = usage {
        resp["usage"] = u;
    }

    resp
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(ErrorBody {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
        .into_response()
}

// ── Upstream call helpers ─────────────────────────────────────────────────

async fn send_responses_request(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    body: &Value,
) -> Result<reqwest::Response, Response> {
    let auth = match load_and_refresh_auth().await {
        Ok(a) => a,
        Err(e) => {
            return Err(error_response(StatusCode::UNAUTHORIZED, &e));
        }
    };

    let mut auth_headers = build_auth_headers(&auth, &state.codex_user_agent().await);
    for key in &["openai-beta", "openai-organization"] {
        if let Some(val) = headers.get(*key) {
            auth_headers.insert(*key, val.clone());
        }
    }

    let url = format!("{}/responses", crate::config::UPSTREAM_BASE);
    debug!("Sending images request to upstream: {url}");

    match state
        .http
        .post(&url)
        .headers(auth_headers)
        .json(body)
        .send()
        .await
    {
        Ok(r) => Ok(r),
        Err(e) => {
            error!("Upstream request error: {e}");
            Err(error_response(
                StatusCode::BAD_GATEWAY,
                &format!("Upstream request failed: {e}"),
            ))
        }
    }
}

/// Handle a 401 by refreshing tokens and retrying once.
async fn retry_on_401(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    body: &Value,
    original: reqwest::Response,
) -> Result<reqwest::Response, Response> {
    if original.status() != StatusCode::UNAUTHORIZED {
        return Ok(original);
    }

    info!("Received 401 from upstream, attempting token refresh...");
    if let Some(tokens) = crate::auth::AuthTokens::load() {
        if let Some(rt) = tokens.refresh_token.clone() {
            match crate::auth::refresh_token(&rt).await {
                Ok(refreshed) => {
                    let mut auth_headers = build_auth_headers(&refreshed, &state.codex_user_agent().await);
                    for key in &["openai-beta", "openai-organization"] {
                        if let Some(val) = headers.get(*key) {
                            auth_headers.insert(*key, val.clone());
                        }
                    }
                    let url = format!("{}/responses", crate::config::UPSTREAM_BASE);
                    match state
                        .http
                        .post(&url)
                        .headers(auth_headers)
                        .json(body)
                        .send()
                        .await
                    {
                        Ok(r) => return Ok(r),
                        Err(e) => {
                            return Err(error_response(
                                StatusCode::BAD_GATEWAY,
                                &format!("Retry upstream request failed: {e}"),
                            ));
                        }
                    }
                }
                Err(e) => {
                    error!("Token refresh failed: {e}");
                }
            }
        }
    }

    Err(error_response(StatusCode::UNAUTHORIZED, "Token refresh failed"))
}

// ── Non-streaming collector ───────────────────────────────────────────────

async fn collect_images(
    upstream: reqwest::Response,
    response_format: &str,
) -> Response {
    let status = upstream.status();
    if !status.is_success() {
        let body = upstream.text().await.unwrap_or_default();
        error!("Upstream error ({status}): {body}");
        return error_response(status, &body);
    }

    let raw = upstream.bytes().await.unwrap_or_default();
    let text = String::from_utf8_lossy(&raw);

    let mut results: Vec<ImageCallResult> = Vec::new();
    let mut created = now_epoch();
    let mut usage: Option<Value> = None;

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

        if event["type"].as_str() == Some("response.completed") {
            if let Some(ts) = event.get("response").and_then(|r| r.get("created_at")).and_then(|v| v.as_u64()) {
                created = ts;
            }
            results = extract_images_from_completed(&event);
            usage = extract_usage(&event);
        }
    }

    if results.is_empty() {
        return error_response(
            StatusCode::BAD_GATEWAY,
            "Upstream did not return image output",
        );
    }

    let resp = build_images_response(&results, response_format, created, usage);
    (StatusCode::OK, axum::Json(resp)).into_response()
}

// ── Streaming forwarder ──────────────────────────────────────────────────

async fn stream_images(
    upstream: reqwest::Response,
    response_format: &str,
    stream_prefix: &str,
) -> Response {
    let response_format = response_format.to_string();
    let stream_prefix = stream_prefix.to_string();

    let stream = upstream.bytes_stream().map(move |result| {
        let chunk = match result {
            Ok(b) => b,
            Err(e) => {
                error!("Stream error: {e}");
                return Ok::<_, std::io::Error>(axum::body::Bytes::from(format!(
                    "event: error\ndata: {{\"error\":{{\"message\":\"Stream error: {e}\"}}}}\n\n"
                )));
            }
        };

        let raw = String::from_utf8_lossy(&chunk);
        let mut output = String::new();

        for line in raw.lines() {
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

            let event_type = event["type"].as_str().unwrap_or("");

            match event_type {
                "response.image_generation_call.partial_image" => {
                    let b64 = match event.get("partial_image_b64").and_then(|v| v.as_str()) {
                        Some(b) => b,
                        None => continue,
                    };
                    if b64.trim().is_empty() {
                        continue;
                    }
                    let output_format = event
                        .get("output_format")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let index = event
                        .get("partial_image_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);

                    let event_name = format!("{stream_prefix}.partial_image");
                    let mut payload = serde_json::json!({
                        "type": event_name,
                        "partial_image_index": index,
                    });

                    if response_format == "url" {
                        let mt = mime_type_from_output_format(output_format);
                        payload["url"] = Value::String(format!("data:{};base64,{}", mt, b64));
                    } else {
                        payload["b64_json"] = Value::String(b64.to_string());
                    }

                    output.push_str(&format!("event: {event_name}\ndata: {payload}\n\n"));
                }
                "response.completed" => {
                    let results = extract_images_from_completed(&event);
                    let usage = extract_usage(&event);

                    if results.is_empty() {
                        let err_payload =
                            serde_json::json!({"error": {"message": "Upstream did not return image output"}});
                        output.push_str(&format!(
                            "event: error\ndata: {err_payload}\n\n"
                        ));
                        continue;
                    }

                    let event_name = format!("{stream_prefix}.completed");
                    for img in &results {
                        let mut payload = serde_json::json!({
                            "type": event_name,
                        });
                        if response_format == "url" {
                            let mt = mime_type_from_output_format(&img.output_format);
                            payload["url"] =
                                Value::String(format!("data:{};base64,{}", mt, img.result));
                        } else {
                            payload["b64_json"] = Value::String(img.result.clone());
                        }
                        if !img.revised_prompt.is_empty() {
                            payload["revised_prompt"] =
                                Value::String(img.revised_prompt.clone());
                        }
                        if let Some(ref u) = usage {
                            payload["usage"] = u.clone();
                        }
                        output.push_str(&format!("event: {event_name}\ndata: {payload}\n\n"));
                    }
                }
                _ => {
                    debug!("Unhandled SSE event type in images stream: {event_type}");
                }
            }
        }

        Ok(axum::body::Bytes::from(output))
    });

    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body)
        .unwrap()
}

// ── Handlers ──────────────────────────────────────────────────────────────

pub async fn handle_images_generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ImagesGenerationsRequest>,
) -> Response {
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "prompt is required");
    }

    let image_model = resolve_image_model(req.model.as_deref());
    let main_model = resolve_main_model(&image_model);
    let response_format = resolve_response_format(req.response_format.as_deref());
    let stream = req.stream.unwrap_or(false);

    let tool_params = ImageToolParams {
        size: req.size.clone(),
        quality: req.quality.clone(),
        background: req.background.clone(),
        output_format: req.output_format.clone(),
        output_compression: req.output_compression,
        partial_images: req.partial_images,
        input_fidelity: None,
        moderation: req.moderation.clone(),
        mask_image_url: None,
    };

    let tool_json = build_tool_json("generate", &image_model, &tool_params);
    let responses_req = build_responses_request(&prompt, &[], &tool_json, &main_model);

    let upstream = match send_responses_request(&state, &headers, &responses_req).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    let upstream = match retry_on_401(&state, &headers, &responses_req, upstream).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    if stream {
        stream_images(upstream, &response_format, "image_generation").await
    } else {
        collect_images(upstream, &response_format).await
    }
}

pub async fn handle_images_edits(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw_request: axum::extract::Request,
) -> Response {
    let ct = raw_request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let body = match axum::body::to_bytes(raw_request.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("Failed to read body: {e}")),
    };

    if ct.contains("application/json") {
        match serde_json::from_slice::<ImagesEditsJsonRequest>(&body) {
            Ok(req) => handle_images_edits_json_inner(state, headers, req).await,
            Err(e) => error_response(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {e}")),
        }
    } else {
        // multipart/form-data or empty content-type: parse manually
        handle_images_edits_multipart_inner(state, headers, &body, &ct).await
    }
}

async fn handle_images_edits_json_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    req: ImagesEditsJsonRequest,
) -> Response {
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "prompt is required");
    }

    let images: Vec<String> = req
        .images
        .iter()
        .map(|img| img.image_url.trim().to_string())
        .filter(|u| !u.is_empty())
        .collect();

    if images.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "images[].image_url is required",
        );
    }

    let image_model = resolve_image_model(req.model.as_deref());
    let main_model = resolve_main_model(&image_model);
    let response_format = resolve_response_format(req.response_format.as_deref());
    let stream = req.stream.unwrap_or(false);

    let mask_url = req
        .mask
        .as_ref()
        .and_then(|m| m.image_url.as_ref())
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty());

    let tool_params = ImageToolParams {
        size: req.size.clone(),
        quality: req.quality.clone(),
        background: req.background.clone(),
        output_format: req.output_format.clone(),
        output_compression: req.output_compression,
        partial_images: req.partial_images,
        input_fidelity: req.input_fidelity.clone(),
        moderation: req.moderation.clone(),
        mask_image_url: mask_url,
    };

    let tool_json = build_tool_json("edit", &image_model, &tool_params);
    let responses_req = build_responses_request(&prompt, &images, &tool_json, &main_model);

    let upstream = match send_responses_request(&state, &headers, &responses_req).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    let upstream = match retry_on_401(&state, &headers, &responses_req, upstream).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    if stream {
        stream_images(upstream, &response_format, "image_edit").await
    } else {
        collect_images(upstream, &response_format).await
    }
}

async fn handle_images_edits_multipart_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: &[u8],
    content_type: &str,
) -> Response {
    let boundary = match extract_boundary(content_type) {
        Some(b) => b,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Invalid Content-Type: missing multipart boundary",
            );
        }
    };

    let fields = match parse_multipart(body, &boundary) {
        Ok(f) => f,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to parse multipart: {e}"),
            );
        }
    };

    let prompt = fields
        .get("prompt")
        .map(|v| v.trim().to_string())
        .unwrap_or_default();
    if prompt.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "prompt is required");
    }

    let mut images: Vec<String> = Vec::new();

    // image[] or image field
    if let Some(raw) = fields.get("image[]") {
        images.push(raw.clone());
    }
    if let Some(raw) = fields.get("image") {
        images.push(raw.clone());
    }

    if images.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "image is required");
    }

    let mask_data_url = fields.get("mask").cloned();

    let image_model = fields
        .get("model")
        .map(|v| resolve_image_model(Some(v)))
        .unwrap_or_else(|| DEFAULT_IMAGES_TOOL_MODEL.to_string());
    let main_model = resolve_main_model(&image_model);
    let response_format = resolve_response_format(fields.get("response_format").map(|s| s.as_str()));
    let stream = fields
        .get("stream")
        .and_then(|v| v.trim().parse::<bool>().ok())
        .unwrap_or(false);

    let tool_params = ImageToolParams {
        size: fields.get("size").cloned(),
        quality: fields.get("quality").cloned(),
        background: fields.get("background").cloned(),
        output_format: fields.get("output_format").cloned(),
        output_compression: fields
            .get("output_compression")
            .and_then(|v| v.trim().parse().ok()),
        partial_images: fields
            .get("partial_images")
            .and_then(|v| v.trim().parse().ok()),
        input_fidelity: fields.get("input_fidelity").cloned(),
        moderation: fields.get("moderation").cloned(),
        mask_image_url: mask_data_url,
    };

    let tool_json = build_tool_json("edit", &image_model, &tool_params);
    let responses_req = build_responses_request(&prompt, &images, &tool_json, &main_model);

    let upstream = match send_responses_request(&state, &headers, &responses_req).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    let upstream = match retry_on_401(&state, &headers, &responses_req, upstream).await {
        Ok(r) => r,
        Err(e) => return e,
    };

    if stream {
        stream_images(upstream, &response_format, "image_edit").await
    } else {
        collect_images(upstream, &response_format).await
    }
}

// ── Multipart parsing helpers ─────────────────────────────────────────────

fn extract_boundary(content_type: &str) -> Option<String> {
    for part in content_type.split(';') {
        let part = part.trim();
        if part.starts_with("boundary=") {
            let boundary = part.strip_prefix("boundary=").unwrap().trim_matches('"');
            return Some(boundary.to_string());
        }
    }
    None
}

/// Minimal multipart parser. Extracts text fields and converts file uploads
/// to data URLs (base64-encoded).
fn parse_multipart(body: &[u8], boundary: &str) -> Result<std::collections::HashMap<String, String>, String> {
    let body_str = String::from_utf8_lossy(body);
    let delimiter = format!("--{boundary}");
    let mut fields = std::collections::HashMap::new();

    for part in body_str.split(&delimiter) {
        let part = part.trim();
        if part.is_empty() || part == "--" {
            continue;
        }

        // Split headers from body
        let (headers_section, body_section) = match part.find("\r\n\r\n") {
            Some(idx) => (&part[..idx], &part[idx + 4..]),
            None => match part.find("\n\n") {
                Some(idx) => (&part[..idx], &part[idx + 2..]),
                None => continue,
            },
        };

        // Extract field name from Content-Disposition
        let mut field_name = None;
        let mut content_type = None;
        let mut is_file = false;

        for header_line in headers_section.lines() {
            let header_line = header_line.trim();
            if let Some(cd) = header_line.strip_prefix("Content-Disposition:") {
                let cd = cd.trim();
                if let Some(name) = extract_attr(cd, "name") {
                    field_name = Some(name);
                }
                is_file = cd.contains("filename=");
            } else if let Some(ct) = header_line.strip_prefix("Content-Type:") {
                content_type = Some(ct.trim().to_string());
            }
        }

        let name = match field_name {
            Some(n) => n,
            None => continue,
        };

        // Strip trailing \r\n-- (boundary closing)
        let body_text = body_section.trim_end_matches("\r\n").trim_end_matches("--").trim_end_matches("\r\n");

        if is_file {
            let ct = content_type.unwrap_or_else(|| "application/octet-stream".to_string());
            let b64 = base64::engine::general_purpose::STANDARD.encode(body_text.as_bytes());
            let data_url = format!("data:{ct};base64,{b64}");
            fields.insert(name, data_url);
        } else {
            fields.insert(name, body_text.to_string());
        }
    }

    Ok(fields)
}

fn extract_attr(header: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = header.find(&pattern)?;
    let value_start = start + pattern.len();
    let value_end = header[value_start..].find('"')?;
    Some(header[value_start..value_start + value_end].to_string())
}

// ── Auth helper ───────────────────────────────────────────────────────────

async fn load_and_refresh_auth() -> Result<crate::auth::AuthTokens, String> {
    let tokens = crate::auth::AuthTokens::load().ok_or_else(|| {
        "Not authenticated. Run `codex-openai-proxy login`.".to_string()
    })?;
    crate::auth::ensure_valid_token(tokens)
        .await
        .map_err(|e: anyhow::Error| e.to_string())
}
