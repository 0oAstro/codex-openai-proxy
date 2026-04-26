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
use crate::models::build_auth_headers;

// ── Request types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub temperature: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
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

fn translate_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let mut instructions: Option<String> = None;
    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                instructions = Some(extract_text(&msg.content));
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
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        content_parts.push(serde_json::json!({
                            "type": "function_call",
                            "id": tc.id,
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }));
                    }
                }
                input.push(serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": content_parts,
                }));
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
                    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                        serde_json::json!({"type": text_type, "text": t})
                    } else {
                        v.clone()
                    }
                })
                .collect();
            Value::Array(parts)
        }
        Some(v) => Value::Array(vec![serde_json::json!({"type": text_type, "text": v.to_string()})]),
    }
}

fn build_responses_body(req: &ChatRequest) -> Value {
    let (model_clean, reasoning) = parse_reasoning_suffix(&req.model);
    let (instructions, input) = translate_messages(&req.messages);

    let mut body = serde_json::json!({
        "model": model_clean,
        "input": input,
        "stream": req.stream,
        "store": false,
    });

    if let Some(instr) = instructions {
        body["instructions"] = Value::String(instr);
    }
    if let Some(effort) = reasoning {
        body["reasoning"] = serde_json::json!({"effort": effort});
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

    body
}

// ── Handler ───────────────────────────────────────────────────────────────

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    let auth = match load_and_refresh_auth().await {
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
    let resp_id = format!("chatcmpl-{}", uuid_short());
    let model = req.model.clone();

    let upstream_body = build_responses_body(&req);
    debug!("Translated request body: {}", serde_json::to_string_pretty(&upstream_body).unwrap_or_default());

    let mut auth_headers = build_auth_headers(&auth, &state.codex_user_agent().await);
    for key in &["openai-beta", "openai-organization"] {
        if let Some(val) = headers.get(*key) {
            auth_headers.insert(*key, val.clone());
        }
    }

    let url = format!("{}/responses", UPSTREAM_BASE);
    let upstream_resp = match state
        .http
        .post(&url)
        .headers(auth_headers.clone())
        .json(&upstream_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("Upstream request error: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"message": format!("Upstream request failed: {e}")}})),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
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
                        let retry_resp = state
                            .http
                            .post(&url)
                            .headers(retry_headers)
                            .json(&upstream_body)
                            .send()
                            .await;

                        match retry_resp {
                            Ok(r) => {
                                return if stream {
                                    handle_streaming(r, &resp_id, &model).await
                                } else {
                                    handle_non_streaming(r, &resp_id, &model).await
                                };
                            }
                            Err(e) => {
                                return (
                                    StatusCode::BAD_GATEWAY,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {e}")}})),
                                )
                                    .into_response();
                            }
                        }
                    }
                    Err(e) => {
                        error!("Token refresh failed: {e}");
                    }
                }
            }
        }
        return (StatusCode::UNAUTHORIZED, "Token refresh failed").into_response();
    }

    if !status.is_success() {
        let body = upstream_resp.text().await.unwrap_or_default();
        error!("Upstream error ({status}): {body}");
        return (
            status,
            Json(serde_json::json!({"error": {"message": body}})),
        )
            .into_response();
    }

    if stream {
        handle_streaming(upstream_resp, &resp_id, &model).await
    } else {
        handle_non_streaming(upstream_resp, &resp_id, &model).await
    }
}

// ── Streaming translation ─────────────────────────────────────────────────

async fn handle_streaming(
    upstream: reqwest::Response,
    resp_id: &str,
    model: &str,
) -> Response {
    let resp_id = resp_id.to_string();
    let model = model.to_string();

    let stream = upstream.bytes_stream().map(move |result| {
        let chunk = match result {
            Ok(b) => b,
            Err(e) => {
                error!("Stream error: {e}");
                return Ok::<_, std::io::Error>(Bytes::from(format!(
                    "data: {{\"error\":{{\"message\":\"Stream error: {e}\"}}}}\n\n"
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
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
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
                            id: resp_id.clone(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.clone(),
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
                        // Could signal start of a new content block.
                        // Nothing needed in OpenAI format — text follows via deltas.
                    }
                    "response.output_text.delta" => {
                        let text = event["delta"].as_str().unwrap_or("");
                        let chunk = ChatCompletionChunk {
                            id: resp_id.clone(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.clone(),
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
                        let delta_args = event["delta"].as_str().unwrap_or("");
                        let item_id = event.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                        let chunk = ChatCompletionChunk {
                            id: resp_id.clone(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    tool_calls: Some(vec![DeltaToolCall {
                                        index: 0,
                                        id: if delta_args.is_empty() { Some(item_id.to_string()) } else { None },
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
                            let fname = item["name"].as_str().unwrap_or("");
                            let fid = item["id"].as_str().unwrap_or("");
                            let fargs = item["arguments"].as_str().unwrap_or("");
                            let chunk = ChatCompletionChunk {
                                id: resp_id.clone(),
                                object: "chat.completion.chunk",
                                created: now_epoch(),
                                model: model.clone(),
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
                        let mut usage = Usage {
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            total_tokens: 0,
                        };
                        if let Some(u) = event.get("response").and_then(|r| r.get("usage")) {
                            usage.prompt_tokens =
                                u["input_tokens"].as_u64().unwrap_or(0);
                            usage.completion_tokens =
                                u["output_tokens"].as_u64().unwrap_or(0);
                            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
                        }
                        let chunk = ChatCompletionChunk {
                            id: resp_id.clone(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta::default(),
                                finish_reason: Some("stop".into()),
                            }],
                            usage: Some(usage),
                        };
                        output.push_str(&format!(
                            "data: {}\n\n",
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                        output.push_str("data: [DONE]\n\n");
                    }
                    _ => {
                        // Forward unknown events as-is (helps debugging).
                        debug!("Unhandled SSE event type: {event_type}");
                    }
                }
            }
        }

        Ok(Bytes::from(output))
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

// ── Non-streaming translation ─────────────────────────────────────────────

async fn handle_non_streaming(
    upstream: reqwest::Response,
    resp_id: &str,
    model: &str,
) -> Response {
    let body: Value = match upstream.json().await {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to parse upstream response: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"message": format!("Bad upstream response: {e}")}})),
            )
                .into_response();
        }
    };

    debug!("Non-streaming upstream response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    let mut content_text = String::new();
    let mut tool_calls_resp: Vec<ToolCallResponse> = Vec::new();

    // Extract output items.
    let output = body.get("output").and_then(|o| o.as_array()).cloned().unwrap_or_default();
    for item in &output {
        let item_type = item["type"].as_str().unwrap_or("");
        match item_type {
            "message" => {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for part in parts {
                        if part["type"].as_str() == Some("output_text") {
                            if let Some(text) = part["text"].as_str() {
                                content_text.push_str(text);
                            }
                        }
                    }
                }
            }
            "function_call" => {
                tool_calls_resp.push(ToolCallResponse {
                    id: item["id"].as_str().unwrap_or("").to_string(),
                    call_type: "function".into(),
                    function: FunctionResponse {
                        name: item["name"].as_str().unwrap_or("").to_string(),
                        arguments: item["arguments"].as_str().unwrap_or("{}").to_string(),
                    },
                });
            }
            _ => {}
        }
    }

    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    if let Some(u) = body.get("usage") {
        usage.prompt_tokens = u["input_tokens"].as_u64().unwrap_or(0);
        usage.completion_tokens = u["output_tokens"].as_u64().unwrap_or(0);
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    }

    let finish_reason = if !tool_calls_resp.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };

    let response = ChatCompletionResponse {
        id: resp_id.to_string(),
        object: "chat.completion",
        created: now_epoch(),
        model: model.to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content: if content_text.is_empty() && !tool_calls_resp.is_empty() {
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
            finish_reason: finish_reason.to_string(),
        }],
        usage,
    };

    Json(response).into_response()
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
