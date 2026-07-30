//! Narrow `wasm-bindgen` boundary for the native rules core.
//!
//! JSON strings are intentional: each call performs strict DTO decoding and
//! contract validation before reaching game logic, and the resulting wire
//! shapes are identical in native tests and WebAssembly.

#![forbid(unsafe_code)]

use serde_json::Value;
use wasm_bindgen::prelude::*;
use wrongway_normal_duel::{
    action_from_json, apply_action, create_initial_state, game_state_from_json, legal_action_codes,
    position_from_json, position_key, validate_config_json, validate_state, Config,
    NormalDuelError, RULESET,
};

type BoundaryResult<T> = std::result::Result<T, String>;

fn parse_value(input: &str, error: &str) -> BoundaryResult<Value> {
    serde_json::from_str(input).map_err(|_| error.to_owned())
}

fn parse_config(input: &str) -> BoundaryResult<Config> {
    let value = parse_value(input, "invalid_config")?;
    validate_config_json(&value).map_err(|error| error.code().to_owned())
}

fn serialize<T: serde::Serialize>(value: &T) -> BoundaryResult<String> {
    serde_json::to_string(value).map_err(|_| "serialization_error".to_owned())
}

fn initial_state_impl(config_json: &str) -> BoundaryResult<String> {
    let config = parse_config(config_json)?;
    serialize(&create_initial_state(&config).map_err(code)?)
}

fn legal_action_codes_impl(config_json: &str, state_json: &str) -> BoundaryResult<String> {
    let config = parse_config(config_json)?;
    let state_value = parse_value(state_json, "invalid_state")?;
    let state = game_state_from_json(&state_value).map_err(code)?;
    let state = validate_state(&config, &state).map_err(code)?;
    serialize(&legal_action_codes(&config, &state).map_err(code)?)
}

fn apply_action_impl(
    config_json: &str,
    state_json: &str,
    action_json: &str,
) -> BoundaryResult<String> {
    let config = parse_config(config_json)?;
    let state_value = parse_value(state_json, "invalid_state")?;
    if state_value
        .get("outcome")
        .and_then(Value::as_object)
        .is_some_and(|outcome| outcome.get("kind").and_then(Value::as_str) != Some("ongoing"))
    {
        return Err("terminal_state".into());
    }
    let state = game_state_from_json(&state_value).map_err(code)?;
    let state = validate_state(&config, &state).map_err(code)?;
    let action_value = parse_value(action_json, "invalid_action")?;
    let action = action_from_json(&action_value).map_err(code)?;
    serialize(&apply_action(&config, &state, &action).map_err(code)?)
}

fn position_key_impl(config_json: &str, position_json: &str) -> BoundaryResult<String> {
    let config = parse_config(config_json)?;
    let position_value = parse_value(position_json, "invalid_position")?;
    let position = position_from_json(&position_value).map_err(code)?;
    position_key(&config, &position).map_err(code)
}

fn code(error: NormalDuelError) -> String {
    error.code().to_owned()
}

fn js_error(code: String) -> JsValue {
    JsValue::from_str(&code)
}

#[wasm_bindgen(js_name = normalDuelVersion)]
#[must_use]
pub fn normal_duel_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen(js_name = normalDuelRuleset)]
#[must_use]
pub fn normal_duel_ruleset() -> String {
    RULESET.to_owned()
}

#[wasm_bindgen(js_name = normalDuelInitialState)]
pub fn normal_duel_initial_state(config_json: &str) -> std::result::Result<String, JsValue> {
    initial_state_impl(config_json).map_err(js_error)
}

#[wasm_bindgen(js_name = normalDuelLegalActionCodes)]
pub fn normal_duel_legal_action_codes(
    config_json: &str,
    state_json: &str,
) -> std::result::Result<String, JsValue> {
    legal_action_codes_impl(config_json, state_json).map_err(js_error)
}

#[wasm_bindgen(js_name = normalDuelApplyAction)]
pub fn normal_duel_apply_action(
    config_json: &str,
    state_json: &str,
    action_json: &str,
) -> std::result::Result<String, JsValue> {
    apply_action_impl(config_json, state_json, action_json).map_err(js_error)
}

#[wasm_bindgen(js_name = normalDuelPositionKey)]
pub fn normal_duel_position_key(
    config_json: &str,
    position_json: &str,
) -> std::result::Result<String, JsValue> {
    position_key_impl(config_json, position_json).map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PERFT: &str = include_str!("../../../tests/fixtures/normal-duel-perft-v1.json");

    fn config() -> Value {
        json!({
            "ruleset": "normal-duel-v1",
            "rows": 9,
            "columns": 9,
            "start": {"A": {"r": 8, "c": 4}, "B": {"r": 0, "c": 4}},
            "goalRows": {"A": 0, "B": 8},
            "initialStock": {"A": 10, "B": 10},
            "jumpRule": "permissive-adjacent-exit-v1",
            "repetitionThreshold": 3,
            "plyCap": 200,
            "firstPlayer": "A"
        })
    }

    #[test]
    fn native_json_boundary_round_trips_initial_legal_apply_and_key() {
        let config = config().to_string();
        let initial_json = initial_state_impl(&config).unwrap();
        let initial: Value = serde_json::from_str(&initial_json).unwrap();
        let codes: Vec<usize> =
            serde_json::from_str(&legal_action_codes_impl(&config, &initial_json).unwrap())
                .unwrap();
        assert_eq!(codes.len(), 131);
        let next_json = apply_action_impl(
            &config,
            &initial_json,
            r#"{"kind":"pawn","to":{"r":7,"c":4}}"#,
        )
        .unwrap();
        let next: Value = serde_json::from_str(&next_json).unwrap();
        assert_eq!(next["ply"], 1);
        assert_eq!(
            position_key_impl(&config, &initial["position"].to_string()).unwrap(),
            initial["positionKey"].as_str().unwrap()
        );
    }

    #[test]
    fn native_json_boundary_is_strict_and_accepts_js_integer_spelling() {
        let exponent_config = config()
            .to_string()
            .replace(r#""plyCap":200"#, r#""plyCap":2e2"#);
        assert!(initial_state_impl(&exponent_config).is_ok());
        let mut unsupported = config();
        unsupported["features"] = json!({"hammer": true});
        assert_eq!(
            initial_state_impl(&unsupported.to_string()),
            Err("unsupported_feature".into())
        );
        assert_eq!(
            apply_action_impl(
                &config().to_string(),
                &initial_state_impl(&config().to_string()).unwrap(),
                r#"{"kind":"wall","wall":"H-9-0"}"#
            ),
            Err("invalid_action".into())
        );
    }

    #[test]
    fn terminal_fixture_precedes_malformed_action_shape() {
        let fixture: Value = serde_json::from_str(PERFT).unwrap();
        let terminal = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "terminal-goal-win-7x7-a")
            .unwrap();
        let config_id = terminal["configId"].as_str().unwrap();
        assert_eq!(
            apply_action_impl(
                &fixture["configs"][config_id].to_string(),
                &terminal["state"].to_string(),
                "{}"
            ),
            Err("terminal_state".into())
        );
    }
}
