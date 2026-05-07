use std::collections::BTreeSet;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Json;
use serde::Serialize;
use serde_json::Value;

use crate::config::{AppState, UPSTREAM_BASE};
use crate::models::build_auth_headers;

#[derive(Debug, Serialize, PartialEq)]
pub struct UsageResponse {
    object: &'static str,
    rate_limits: Vec<RateLimitSnapshot>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct RateLimitSnapshot {
    limit_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    primary: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    secondary: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    credits: Option<CreditsSnapshot>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct RateLimitWindow {
    used_percent: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    window_minutes: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resets_at: Option<i64>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct CreditsSnapshot {
    has_credits: bool,
    unlimited: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    balance: Option<String>,
}

pub async fn handle_usage(
    State(state): State<Arc<AppState>>,
) -> Result<Json<UsageResponse>, (StatusCode, String)> {
    let auth = crate::auth::AuthTokens::load().ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            "Not authenticated. Run `codex-openai-proxy login`.".into(),
        )
    })?;
    let auth = crate::auth::ensure_valid_token(auth)
        .await
        .map_err(|e| (StatusCode::UNAUTHORIZED, e.to_string()))?;

    let user_agent = state.codex_user_agent().await;
    let mut auth_headers = build_auth_headers(&auth, &user_agent);
    auth_headers.insert(
        "openai-beta",
        "responses=experimental"
            .parse()
            .expect("valid header value"),
    );

    let version = state.client_version().await;
    let url = format!("{}/responses?client_version={version}", UPSTREAM_BASE);
    let body = serde_json::json!({
        "model": "gpt-5.5",
        "instructions": "",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}]
        }],
        "store": false,
        "stream": true
    });

    let resp = state
        .http
        .post(url)
        .headers(auth_headers)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Upstream request failed: {e}"),
            )
        })?;

    let status = resp.status();
    let header_limits = parse_all_rate_limits(resp.headers());
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err((status, body));
    }

    let mut rate_limits = header_limits;
    let raw = resp.text().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to read upstream: {e}"),
        )
    })?;
    rate_limits.extend(parse_rate_limit_events(&raw));
    dedupe_rate_limits(&mut rate_limits);

    Ok(Json(UsageResponse {
        object: "codex.usage",
        rate_limits,
    }))
}

fn parse_all_rate_limits(headers: &HeaderMap) -> Vec<RateLimitSnapshot> {
    let mut snapshots = Vec::new();
    if let Some(snapshot) = parse_rate_limit_for_limit(headers, None) {
        snapshots.push(snapshot);
    }

    let mut limit_ids = BTreeSet::new();
    for name in headers.keys() {
        let header_name = name.as_str().to_ascii_lowercase();
        if let Some(limit_id) = header_name_to_limit_id(&header_name) {
            if limit_id != "codex" {
                limit_ids.insert(limit_id);
            }
        }
    }

    snapshots.extend(limit_ids.into_iter().filter_map(|limit_id| {
        let snapshot = parse_rate_limit_for_limit(headers, Some(&limit_id))?;
        has_rate_limit_data(&snapshot).then_some(snapshot)
    }));
    snapshots
}

fn parse_rate_limit_for_limit(
    headers: &HeaderMap,
    limit_id: Option<&str>,
) -> Option<RateLimitSnapshot> {
    let normalized_limit = limit_id
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("codex")
        .to_ascii_lowercase()
        .replace('_', "-");
    let prefix = format!("x-{normalized_limit}");
    let primary = parse_rate_limit_window(
        headers,
        &format!("{prefix}-primary-used-percent"),
        &format!("{prefix}-primary-window-minutes"),
        &format!("{prefix}-primary-reset-at"),
    );
    let secondary = parse_rate_limit_window(
        headers,
        &format!("{prefix}-secondary-used-percent"),
        &format!("{prefix}-secondary-window-minutes"),
        &format!("{prefix}-secondary-reset-at"),
    );
    let credits = parse_credits_snapshot(headers);
    let limit_name = parse_header_str(headers, &format!("{prefix}-limit-name"))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string);

    Some(RateLimitSnapshot {
        limit_id: normalize_limit_id(normalized_limit),
        limit_name,
        primary,
        secondary,
        credits,
    })
}

fn parse_rate_limit_window(
    headers: &HeaderMap,
    used_percent_header: &str,
    window_minutes_header: &str,
    resets_at_header: &str,
) -> Option<RateLimitWindow> {
    let used_percent = parse_header_f64(headers, used_percent_header)?;
    let window_minutes = parse_header_i64(headers, window_minutes_header);
    let resets_at = parse_header_i64(headers, resets_at_header);

    let has_data = used_percent != 0.0
        || window_minutes.is_some_and(|minutes| minutes != 0)
        || resets_at.is_some();

    has_data.then_some(RateLimitWindow {
        used_percent,
        window_minutes,
        resets_at,
    })
}

fn parse_credits_snapshot(headers: &HeaderMap) -> Option<CreditsSnapshot> {
    let has_credits = parse_header_bool(headers, "x-codex-credits-has-credits")?;
    let unlimited = parse_header_bool(headers, "x-codex-credits-unlimited")?;
    let balance = parse_header_str(headers, "x-codex-credits-balance")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);

    Some(CreditsSnapshot {
        has_credits,
        unlimited,
        balance,
    })
}

fn parse_rate_limit_events(raw: &str) -> Vec<RateLimitSnapshot> {
    raw.lines()
        .filter_map(|line| line.trim().strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(parse_rate_limit_event)
        .collect()
}

fn parse_rate_limit_event(data: &str) -> Option<RateLimitSnapshot> {
    let event: Value = serde_json::from_str(data).ok()?;
    if event.get("type").and_then(Value::as_str) != Some("codex.rate_limits") {
        return None;
    }

    let details = event.get("rate_limits");
    let primary = details.and_then(|details| map_event_window(details.get("primary")));
    let secondary = details.and_then(|details| map_event_window(details.get("secondary")));
    let credits = event.get("credits").and_then(|credits| {
        Some(CreditsSnapshot {
            has_credits: credits.get("has_credits")?.as_bool()?,
            unlimited: credits.get("unlimited")?.as_bool()?,
            balance: credits
                .get("balance")
                .and_then(Value::as_str)
                .map(ToString::to_string),
        })
    });

    let limit_id = event
        .get("metered_limit_name")
        .or_else(|| event.get("limit_name"))
        .and_then(Value::as_str)
        .map(normalize_limit_id)
        .unwrap_or_else(|| "codex".to_string());

    Some(RateLimitSnapshot {
        limit_id,
        limit_name: None,
        primary,
        secondary,
        credits,
    })
}

fn map_event_window(window: Option<&Value>) -> Option<RateLimitWindow> {
    let window = window?;
    Some(RateLimitWindow {
        used_percent: window.get("used_percent")?.as_f64()?,
        window_minutes: window.get("window_minutes").and_then(Value::as_i64),
        resets_at: window.get("reset_at").and_then(Value::as_i64),
    })
}

fn dedupe_rate_limits(rate_limits: &mut Vec<RateLimitSnapshot>) {
    let mut seen = BTreeSet::new();
    rate_limits.retain(|snapshot| seen.insert(snapshot.limit_id.clone()));
}

fn parse_header_f64(headers: &HeaderMap, name: &str) -> Option<f64> {
    parse_header_str(headers, name)?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

fn parse_header_i64(headers: &HeaderMap, name: &str) -> Option<i64> {
    parse_header_str(headers, name)?.parse::<i64>().ok()
}

fn parse_header_bool(headers: &HeaderMap, name: &str) -> Option<bool> {
    let raw = parse_header_str(headers, name)?;
    if raw.eq_ignore_ascii_case("true") || raw == "1" {
        Some(true)
    } else if raw.eq_ignore_ascii_case("false") || raw == "0" {
        Some(false)
    } else {
        None
    }
}

fn parse_header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

fn has_rate_limit_data(snapshot: &RateLimitSnapshot) -> bool {
    snapshot.primary.is_some() || snapshot.secondary.is_some() || snapshot.credits.is_some()
}

fn header_name_to_limit_id(header_name: &str) -> Option<String> {
    let prefix = header_name.strip_suffix("-primary-used-percent")?;
    let limit = prefix.strip_prefix("x-")?;
    Some(normalize_limit_id(limit))
}

fn normalize_limit_id(name: impl AsRef<str>) -> String {
    name.as_ref().trim().to_ascii_lowercase().replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn parses_header_rate_limits() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-codex-primary-used-percent",
            HeaderValue::from_static("42"),
        );
        headers.insert(
            "x-codex-primary-window-minutes",
            HeaderValue::from_static("300"),
        );
        headers.insert(
            "x-codex-primary-reset-at",
            HeaderValue::from_static("1700000000"),
        );
        headers.insert(
            "x-codex-credits-has-credits",
            HeaderValue::from_static("true"),
        );
        headers.insert(
            "x-codex-credits-unlimited",
            HeaderValue::from_static("false"),
        );

        let snapshots = parse_all_rate_limits(&headers);

        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].limit_id, "codex");
        assert_eq!(snapshots[0].primary.as_ref().unwrap().used_percent, 42.0);
        assert!(snapshots[0].credits.as_ref().unwrap().has_credits);
    }

    #[test]
    fn parses_sse_rate_limit_events() {
        let raw = r#"event: codex.rate_limits
data: {"type":"codex.rate_limits","rate_limits":{"primary":{"used_percent":12,"window_minutes":60,"reset_at":1700000000},"secondary":null},"credits":{"has_credits":true,"unlimited":false,"balance":"123"}}

data: [DONE]
"#;

        let snapshots = parse_rate_limit_events(raw);

        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].limit_id, "codex");
        assert_eq!(
            snapshots[0].primary.as_ref().unwrap().resets_at,
            Some(1700000000)
        );
        assert_eq!(
            snapshots[0].credits.as_ref().unwrap().balance.as_deref(),
            Some("123")
        );
    }
}
