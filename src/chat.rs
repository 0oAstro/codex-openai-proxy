use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::config::{AppState, REASONING_EFFORTS, UPSTREAM_BASE};
use crate::models::{build_auth_headers, copy_codex_passthrough_headers};

// ── Request types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub n: Option<u64>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub response_format: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub seed: Option<i64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub service_tier: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub tool_type: String,
    pub function: Option<FunctionDef>,
}

#[derive(Debug, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

// ── Response types ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Serialize)]
struct DeltaToolCall {
    index: u32,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<DeltaFunction>,
}

#[derive(Debug, Serialize)]
struct DeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: u32,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Debug, Serialize)]
struct ToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionResponse,
}

#[derive(Debug, Serialize)]
struct FunctionResponse {
    name: String,
    arguments: String,
}

// ── Reasoning suffix parser ──────────────────────────────────────────────

/// Parse model name for reasoning effort suffix.
/// E.g. "gpt-5.5-xhigh" -> ("gpt-5.5", Some("xhigh"))
fn parse_reasoning_suffix(model: &str) -> (String, Option<String>) {
    for effort in REASONING_EFFORTS {
        let suffix = format!("-{}", effort);
        if model.ends_with(&suffix) {
            return (
                model[..model.len() - suffix.len()].to_string(),
                Some(effort.to_string()),
            );
        }
    }
    (model.to_string(), None)
}

// ── Translation: Chat -> Responses ───────────────────────────────────────

fn translate_messages(messages: &[ChatMessage]) -> (String, Vec<Value>) {
    let mut instruction_parts: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" | "developer" => {
                instruction_parts.push(extract_text(&msg.content));
            }
            "user" => {
                input.push(serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": translate_content(&msg.content, "input_text"),
                }));
            }
            "assistant" => {
                let mut content_parts: Vec<Value> = Vec::new();
                if let Some(ref c) = msg.content {
                    let text = extract_text_val(c);
                    if !text.is_empty() {
                        content_parts.push(serde_json::json!({
                            "type": "output_text",
                            "text": text,
                        }));
                    }
                }
                if !content_parts.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": content_parts,
                    }));
                }
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        input.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }));
                    }
                }
            }
            "tool" => {
                input.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": extract_text(&msg.content),
                }));
            }
            other => {
                warn!("Unknown message role: {other}, treating as user");
                input.push(serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": translate_content(&msg.content, "input_text"),
                }));
            }
        }
    }

    let instructions = instruction_parts.join("\n\n");
    (instructions, input)
}

fn extract_text(content: &Option<Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Some(v) => v.to_string(),
    }
}

fn extract_text_val(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn translate_content(content: &Option<Value>, text_type: &str) -> Value {
    match content {
        None => Value::Array(vec![serde_json::json!({"type": text_type, "text": ""})]),
        Some(Value::String(s)) => {
            Value::Array(vec![serde_json::json!({"type": text_type, "text": s})])
        }
        Some(Value::Array(arr)) => {
            let parts: Vec<Value> = arr
                .iter()
                .map(|v| {
                    let part_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match part_type {
                        "text" => {
                            let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
                            serde_json::json!({"type": text_type, "text": text})
                        }
                        "image_url" => {
                            let url = v
                                .get("image_url")
                                .and_then(|iu| iu.get("url"))
                                .and_then(|u| u.as_str())
                                .unwrap_or("");
                            serde_json::json!({"type": "input_image", "image_url": url})
                        }
                        _ => v.clone(),
                    }
                })
                .collect();
            Value::Array(parts)
        }
        Some(v) => Value::Array(vec![
            serde_json::json!({"type": text_type, "text": v.to_string()}),
        ]),
    }
}

fn build_responses_body(req: &ChatRequest) -> Value {
    let (model_clean, reasoning_from_suffix) = parse_reasoning_suffix(&req.model);
    let (instructions, input) = translate_messages(&req.messages);

    let mut body = serde_json::json!({
        "model": model_clean,
        "instructions": instructions,
        "input": input,
        // Always stream from upstream (Codex requires it). For non-streaming
        // client requests, we collect the SSE into a single response.
        "stream": true,
        "store": false,
    });

    // Reasoning effort: explicit request param takes priority over model suffix.
    let effort = req
        .reasoning_effort
        .as_deref()
        .or(reasoning_from_suffix.as_deref());
    if let Some(effort) = effort {
        body["reasoning"] = serde_json::json!({"effort": effort});
    }

    // Sampling controls are intentionally accepted for OpenAI compatibility
    // but not forwarded. Codex Responses rejects these fields for GPT-5/Codex
    // models, and callers frequently send low-temperature chat-completion
    // defaults that would otherwise turn into 400s before Bifrost can fall
    // back or retry cleanly.

    // Codex rejects the Responses API token cap, so chat token limits are
    // intentionally accepted for OpenAI compatibility but not forwarded.

    // Stop sequences
    if let Some(ref stop) = req.stop {
        match stop {
            Value::String(s) => {
                body["stop"] = serde_json::json!([s]);
            }
            Value::Array(arr) => {
                body["stop"] = Value::Array(arr.clone());
            }
            _ => {}
        }
    }

    // Pass through tools if present.
    if let Some(ref tools) = req.tools {
        let codex_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                t.function.as_ref().map(|f| {
                    serde_json::json!({
                        "type": "function",
                        "name": f.name,
                        "description": f.description,
                        "parameters": f.parameters,
                    })
                })
            })
            .collect();
        if !codex_tools.is_empty() {
            body["tools"] = Value::Array(codex_tools);
        }
    }

    // Tool choice - only pass through explicit function selection.
    // The Codex API does not accept {"type":"auto"} etc.
    if let Some(ref tc) = req.tool_choice {
        match tc {
            Value::String(s) => {
                // "auto", "none", "required" are OpenAI spec values but Codex
                // doesn't support them as objects. Omit for auto/required, send
                // empty tools array for none.
                if s == "none" {
                    body["tools"] = Value::Array(vec![]);
                }
                // For "auto" / "required" - just omit tool_choice to let Codex decide.
            }
            Value::Object(obj) => {
                if obj.get("type").and_then(|v| v.as_str()) == Some("function") {
                    if let Some(fname) = obj
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                    {
                        body["tool_choice"] =
                            serde_json::json!({"type": "function", "name": fname});
                    }
                }
            }
            _ => {}
        }
    }

    // Parallel tool calls
    if let Some(p) = req.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(p);
    }

    body
}

// ── Handler ───────────────────────────────────────────────────────────────

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    let auth_candidates = match load_and_refresh_auth().await {
        Ok(a) => a,
        Err(e) => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"message": e}})),
            )
                .into_response();
        }
    };

    let stream = req.stream;
    let stream_include_usage = req
        .stream_options
        .as_ref()
        .and_then(|so| so.include_usage)
        .unwrap_or(false);
    let resp_id = format!("chatcmpl-{}", uuid_short());
    let model = req.model.clone();

    let upstream_body = build_responses_body(&req);
    debug!(
        "Translated request body: {}",
        serde_json::to_string_pretty(&upstream_body).unwrap_or_default()
    );

    let url = format!("{}/responses", UPSTREAM_BASE);
    let mut last_error: Option<(StatusCode, String)> = None;

    for (idx, auth) in auth_candidates.iter().enumerate() {
        let has_next = idx + 1 < auth_candidates.len();
        let account = auth.account_label();
        let user_agent = state.codex_user_agent().await;
        let version = state.client_version().await;
        let mut auth_headers = build_auth_headers(auth, &user_agent, &version);
        // Always send OpenAI-Beta header for the Responses API.
        auth_headers.insert(
            "openai-beta",
            "responses=experimental"
                .parse()
                .expect("valid header value"),
        );
        copy_codex_passthrough_headers(&headers, &mut auth_headers);

        let mut upstream_resp = match state
            .http
            .post(&url)
            .headers(auth_headers.clone())
            .json(&upstream_body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!("Upstream request error for {account}: {e}");
                if has_next {
                    last_error = Some((
                        StatusCode::BAD_GATEWAY,
                        format!("Upstream request failed: {e}"),
                    ));
                    continue;
                }
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": {"message": format!("Upstream request failed: {e}")}})),
                )
                    .into_response();
            }
        };

        if upstream_resp.status() == StatusCode::UNAUTHORIZED {
            info!("Received 401 for {account}, attempting token refresh...");
            match crate::auth::refresh_existing_token(auth).await {
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
                    match state
                        .http
                        .post(&url)
                        .headers(retry_headers)
                        .json(&upstream_body)
                        .send()
                        .await
                    {
                        Ok(r) => upstream_resp = r,
                        Err(e) => {
                            error!("Retry failed for {account}: {e}");
                            if has_next {
                                last_error =
                                    Some((StatusCode::BAD_GATEWAY, format!("Retry failed: {e}")));
                                continue;
                            }
                            return (
                                StatusCode::BAD_GATEWAY,
                                Json(serde_json::json!({"error": {"message": format!("Retry failed: {e}")}})),
                            )
                                .into_response();
                        }
                    }
                }
                Err(e) => {
                    error!("Token refresh failed for {account}: {e}");
                }
            }
        }

        let status = upstream_resp.status();
        if status.is_success() {
            return if stream {
                handle_streaming(upstream_resp, &resp_id, &model, stream_include_usage).await
            } else {
                handle_non_streaming(upstream_resp, &resp_id, &model).await
            };
        }

        let body = upstream_resp.text().await.unwrap_or_default();
        error!("Upstream error for {account} ({status}): {body}");
        if has_next && crate::auth::should_fallback_status(status) {
            info!("Falling back from {account} after upstream status {status}");
            last_error = Some((status, body));
            continue;
        }
        return (
            status,
            Json(serde_json::json!({"error": {"message": body}})),
        )
            .into_response();
    }

    let (status, body) = last_error.unwrap_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            "No configured auth accounts are usable".to_string(),
        )
    });
    (
        status,
        Json(serde_json::json!({"error": {"message": body}})),
    )
        .into_response()
}

// ── Streaming translation ─────────────────────────────────────────────────

async fn handle_streaming(
    upstream: reqwest::Response,
    resp_id: &str,
    model: &str,
    stream_include_usage: bool,
) -> Response {
    let resp_id = resp_id.to_string();
    let model = model.to_string();

    let stream = futures::stream::unfold(
        (
            upstream.bytes_stream(),
            resp_id,
            model,
            false,
            false,
            false,
            false,
            stream_include_usage,
        ),
        |(
            mut upstream_stream,
            resp_id,
            model,
            mut saw_completed,
            mut sent_eof_done,
            mut saw_tool_call,
            mut saw_function_arg_delta,
            stream_include_usage,
        )| async move {
            loop {
                let Some(result) = upstream_stream.next().await else {
                    if saw_completed || sent_eof_done {
                        return None;
                    }

                    sent_eof_done = true;
                    let finish_reason = if saw_tool_call { "tool_calls" } else { "stop" };
                    let output = final_stream_chunk(
                        &resp_id,
                        &model,
                        None,
                        finish_reason,
                        stream_include_usage,
                    );
                    return Some((
                        Ok::<Bytes, std::io::Error>(Bytes::from(output)),
                        (
                            upstream_stream,
                            resp_id,
                            model,
                            saw_completed,
                            sent_eof_done,
                            saw_tool_call,
                            saw_function_arg_delta,
                            stream_include_usage,
                        ),
                    ));
                };

                let chunk = match result {
                    Ok(b) => b,
                    Err(e) => {
                        error!("Stream error: {e}");
                        return Some((
                            Ok::<Bytes, std::io::Error>(Bytes::from(format!(
                                "data: {{\"error\":{{\"message\":\"Stream error: {e}\"}}}}\n\n"
                            ))),
                            (
                                upstream_stream,
                                resp_id,
                                model,
                                saw_completed,
                                true,
                                saw_tool_call,
                                saw_function_arg_delta,
                                stream_include_usage,
                            ),
                        ));
                    }
                };

                let translation = translate_stream_chunk(
                    &chunk,
                    &resp_id,
                    &model,
                    saw_tool_call,
                    saw_function_arg_delta,
                    stream_include_usage,
                );
                saw_completed |= translation.completed;
                saw_tool_call |= translation.saw_tool_call;
                saw_function_arg_delta |= translation.saw_function_arg_delta;
                let output = translation.output;
                if output.is_empty() {
                    continue;
                }

                return Some((
                    Ok::<Bytes, std::io::Error>(Bytes::from(output)),
                    (
                        upstream_stream,
                        resp_id,
                        model,
                        saw_completed,
                        sent_eof_done,
                        saw_tool_call,
                        saw_function_arg_delta,
                        stream_include_usage,
                    ),
                ));
            }
        },
    );

    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body)
        .unwrap()
}

struct StreamTranslation {
    output: String,
    completed: bool,
    saw_tool_call: bool,
    saw_function_arg_delta: bool,
}

fn translate_stream_chunk(
    chunk: &[u8],
    resp_id: &str,
    model: &str,
    saw_tool_call_before: bool,
    saw_function_arg_delta_before: bool,
    stream_include_usage: bool,
) -> StreamTranslation {
    let raw = String::from_utf8_lossy(chunk);
    let mut output = String::new();
    let mut completed = false;
    let mut saw_tool_call = false;
    let mut saw_function_arg_delta = false;
    let mut tool_call_seen = saw_tool_call_before;
    let mut arg_delta_seen = saw_function_arg_delta_before;

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                completed = true;
                output.push_str("data: [DONE]\n\n");
                continue;
            }

            let event: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    debug!("Failed to parse SSE event: {e}");
                    continue;
                }
            };

            let event_type = event["type"].as_str().unwrap_or("");

            match event_type {
                "response.created" => {
                    let chunk = ChatCompletionChunk {
                        id: resp_id.to_string(),
                        object: "chat.completion.chunk",
                        created: now_epoch(),
                        model: model.to_string(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: Some("assistant".into()),
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    output.push_str(&format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    ));
                }
                "response.output_item.added" => {
                    let item = &event["item"];
                    if item["type"].as_str() == Some("function_call") {
                        saw_tool_call = true;
                        tool_call_seen = true;
                        let fname = item["name"].as_str().unwrap_or("");
                        let fid = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item["id"].as_str())
                            .unwrap_or("");
                        let chunk = ChatCompletionChunk {
                            id: resp_id.to_string(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.to_string(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    tool_calls: Some(vec![DeltaToolCall {
                                        index: 0,
                                        id: Some(fid.to_string()),
                                        call_type: Some("function".into()),
                                        function: Some(DeltaFunction {
                                            name: Some(fname.to_string()),
                                            arguments: Some(String::new()),
                                        }),
                                    }]),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        output.push_str(&format!(
                            "data: {}\n\n",
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                    }
                }
                "response.output_text.delta" => {
                    let text = event["delta"].as_str().unwrap_or("");
                    let chunk = ChatCompletionChunk {
                        id: resp_id.to_string(),
                        object: "chat.completion.chunk",
                        created: now_epoch(),
                        model: model.to_string(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text.to_string()),
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    output.push_str(&format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    ));
                }
                "response.function_call_arguments.delta" => {
                    saw_tool_call = true;
                    tool_call_seen = true;
                    saw_function_arg_delta = true;
                    arg_delta_seen = true;
                    let delta_args = event["delta"].as_str().unwrap_or("");
                    let chunk = ChatCompletionChunk {
                        id: resp_id.to_string(),
                        object: "chat.completion.chunk",
                        created: now_epoch(),
                        model: model.to_string(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                                tool_calls: Some(vec![DeltaToolCall {
                                    index: 0,
                                    id: None,
                                    call_type: Some("function".into()),
                                    function: Some(DeltaFunction {
                                        name: None,
                                        arguments: Some(delta_args.to_string()),
                                    }),
                                }]),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    output.push_str(&format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    ));
                }
                "response.output_item.done" => {
                    // A function call item finished — check if it's a function_call
                    let item = &event["item"];
                    if item["type"].as_str() == Some("function_call") {
                        saw_tool_call = true;
                        tool_call_seen = true;
                        if arg_delta_seen {
                            continue;
                        }
                        let fname = item["name"].as_str().unwrap_or("");
                        let fid = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item["id"].as_str())
                            .unwrap_or("");
                        let fargs = item["arguments"].as_str().unwrap_or("");
                        let chunk = ChatCompletionChunk {
                            id: resp_id.to_string(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.to_string(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    tool_calls: Some(vec![DeltaToolCall {
                                        index: 0,
                                        id: Some(fid.to_string()),
                                        call_type: Some("function".into()),
                                        function: Some(DeltaFunction {
                                            name: Some(fname.to_string()),
                                            arguments: Some(fargs.to_string()),
                                        }),
                                    }]),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        output.push_str(&format!(
                            "data: {}\n\n",
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                    }
                }
                "response.completed" => {
                    completed = true;
                    let usage = event.get("response").and_then(|r| r.get("usage"));
                    let finish_reason = if tool_call_seen { "tool_calls" } else { "stop" };
                    output.push_str(&final_stream_chunk(
                        resp_id,
                        model,
                        usage,
                        finish_reason,
                        stream_include_usage,
                    ));
                }
                "response.incomplete" => {
                    completed = true;
                    let usage = event.get("response").and_then(|r| r.get("usage"));
                    output.push_str(&final_stream_chunk(
                        resp_id,
                        model,
                        usage,
                        "length",
                        stream_include_usage,
                    ));
                }
                _ => {
                    // Forward unknown events as-is (helps debugging).
                    debug!("Unhandled SSE event type: {event_type}");
                }
            }
        }
    }

    StreamTranslation {
        output,
        completed,
        saw_tool_call,
        saw_function_arg_delta,
    }
}

fn final_stream_chunk(
    resp_id: &str,
    model: &str,
    usage: Option<&Value>,
    finish_reason: &str,
    stream_include_usage: bool,
) -> String {
    let usage_obj = usage.map(|u| {
        let prompt_tokens = u["input_tokens"].as_u64().unwrap_or(0);
        let completion_tokens = u["output_tokens"].as_u64().unwrap_or(0);
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    });

    let finish_chunk = ChatCompletionChunk {
        id: resp_id.to_string(),
        object: "chat.completion.chunk",
        created: now_epoch(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(finish_reason.into()),
        }],
        usage: None,
    };

    // If stream_options.include_usage is true, emit a separate final chunk with usage.
    if stream_include_usage {
        let usage_chunk = ChatCompletionChunk {
            id: resp_id.to_string(),
            object: "chat.completion.chunk",
            created: now_epoch(),
            model: model.to_string(),
            choices: vec![],
            usage: usage_obj,
        };
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&finish_chunk).unwrap_or_default(),
            serde_json::to_string(&usage_chunk).unwrap_or_default(),
        )
    } else {
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&finish_chunk).unwrap_or_default(),
        )
    }
}

// ── Non-streaming translation ─────────────────────────────────────────────

/// Collect the SSE stream from upstream into a single chat completion response.
async fn handle_non_streaming(upstream: reqwest::Response, resp_id: &str, model: &str) -> Response {
    let status = upstream.status();
    if !status.is_success() {
        let body = upstream.text().await.unwrap_or_default();
        error!("Upstream error ({status}): {body}");
        return (
            status,
            Json(serde_json::json!({"error": {"message": body}})),
        )
            .into_response();
    }

    let raw = upstream.bytes().await.unwrap_or_default();
    let text = String::from_utf8_lossy(&raw);

    let mut content_text = String::new();
    let mut tool_calls_resp: Vec<ToolCallResponse> = Vec::new();
    let mut usage: Option<Usage> = None;
    let mut finish_reason = "stop".to_string();

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

        let event_type = event["type"].as_str().unwrap_or("");

        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = event["delta"].as_str() {
                    content_text.push_str(delta);
                }
            }
            "response.output_item.added" => {
                let item = &event["item"];
                if item["type"].as_str() == Some("function_call") {
                    let id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item["id"].as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    tool_calls_resp.push(ToolCallResponse {
                        id,
                        call_type: "function".into(),
                        function: FunctionResponse {
                            name,
                            arguments: String::new(),
                        },
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(tc) = tool_calls_resp.last_mut() {
                    if let Some(delta) = event["delta"].as_str() {
                        tc.function.arguments.push_str(delta);
                    }
                }
            }
            "response.output_item.done" => {
                let item = &event["item"];
                if item["type"].as_str() == Some("function_call") {
                    let id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item["id"].as_str())
                        .unwrap_or("");
                    // If we already have partial args from deltas, keep them.
                    // Otherwise use the full args from this done event.
                    if let Some(tc) = tool_calls_resp.iter_mut().find(|t| t.id == id) {
                        if tc.function.arguments.is_empty() {
                            tc.function.arguments =
                                item["arguments"].as_str().unwrap_or("").to_string();
                        }
                    }
                }
            }
            "response.completed" => {
                if let Some(resp) = event.get("response") {
                    if let Some(u) = resp.get("usage") {
                        let prompt_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                        let completion_tokens = u["output_tokens"].as_u64().unwrap_or(0);
                        usage = Some(Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        });
                    }
                }
                finish_reason = if !tool_calls_resp.is_empty() {
                    "tool_calls".to_string()
                } else {
                    "stop".to_string()
                };
            }
            "response.incomplete" => {
                finish_reason = "length".to_string();
            }
            _ => {}
        }
    }

    let response = ChatCompletionResponse {
        id: resp_id.to_string(),
        object: "chat.completion",
        created: now_epoch(),
        model: model.to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content: if content_text.is_empty() {
                    None
                } else {
                    Some(content_text)
                },
                tool_calls: if tool_calls_resp.is_empty() {
                    None
                } else {
                    Some(tool_calls_resp)
                },
            },
            finish_reason,
        }],
        usage: usage.unwrap_or(Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }),
    };

    Json(response).into_response()
}

// ── helpers ───────────────────────────────────────────────────────────────

async fn load_and_refresh_auth() -> Result<Vec<crate::auth::AuthTokens>, String> {
    crate::auth::load_and_refresh_auth_candidates().await
}

fn uuid_short() -> String {
    let mut bytes = [0u8; 8];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: "gpt-5.5".to_string(),
            messages,
            stream: true,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            stop: None,
            n: None,
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            frequency_penalty: None,
            presence_penalty: None,
            logprobs: None,
            response_format: None,
            seed: None,
            user: None,
            service_tier: None,
        }
    }

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(Value::String(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_tool_call_message(content: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: content.map(|s| Value::String(s.to_string())),
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                call_type: Some("function".to_string()),
                function: FunctionCall {
                    name: "lookup".to_string(),
                    arguments: r#"{"query":"status"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn chat_translation_always_includes_instructions() {
        let body = build_responses_body(&request(vec![message("user", "hello")]));

        assert_eq!(body["instructions"], Value::String(String::new()));
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn chat_translation_uses_system_message_as_instructions() {
        let body = build_responses_body(&request(vec![
            message("system", "be direct"),
            message("user", "hello"),
        ]));

        assert_eq!(body["instructions"], "be direct");
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn chat_translation_uses_developer_message_as_instructions() {
        let body = build_responses_body(&request(vec![
            message("developer", "follow the repo instructions"),
            message("user", "hello"),
        ]));

        assert_eq!(body["instructions"], "follow the repo instructions");
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["input"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn chat_translation_omits_unsupported_token_caps() {
        let mut req = request(vec![message("user", "title this")]);
        req.max_tokens = Some(12);
        req.max_completion_tokens = Some(8);

        let body = build_responses_body(&req);

        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn chat_translation_omits_unsupported_sampling_params() {
        let mut req = request(vec![message("user", "title this")]);
        req.temperature = Some(0.1);
        req.top_p = Some(0.2);

        let body = build_responses_body(&req);

        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn chat_translation_puts_assistant_tool_calls_at_top_level() {
        let body = build_responses_body(&request(vec![
            message("user", "check status"),
            assistant_tool_call_message(Some("I will check.")),
            ChatMessage {
                role: "tool".to_string(),
                content: Some(Value::String(r#"{"status":"ok"}"#.to_string())),
                tool_calls: None,
                tool_call_id: Some("call_123".to_string()),
                name: None,
            },
        ]));

        assert_eq!(body["input"][1]["type"], "message");
        assert_eq!(body["input"][1]["content"][0]["type"], "output_text");
        assert_eq!(body["input"][2]["type"], "function_call");
        assert_eq!(body["input"][2]["call_id"], "call_123");
        assert!(body["input"][2].get("content").is_none());
        assert_eq!(body["input"][3]["type"], "function_call_output");
    }

    #[test]
    fn chat_translation_omits_empty_assistant_message_for_tool_only_turn() {
        let body = build_responses_body(&request(vec![
            message("user", "check status"),
            assistant_tool_call_message(None),
        ]));

        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(body["input"][1]["type"], "function_call");
    }
}
