//! Narrow `wasm-bindgen` boundary for the native rules core.
//!
//! JSON strings are intentional: each call performs strict DTO decoding and
//! contract validation before reaching game logic, and the resulting wire
//! shapes are identical in native tests and WebAssembly.

#![forbid(unsafe_code)]

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasm_bindgen::prelude::*;
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::puct::{PuctParams, PuctTreeSearch};
use wrongway_normal_duel::selfplay::{
    Exploration, GameOutcome, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_FLOATS,
    RECORD_META_FIELDS, RECORD_POLICY, RECORD_STATE_FIELDS, RECORD_VERSION, RECORD_WINDOW_FIELDS,
};
use wrongway_normal_duel::{
    action_from_json, apply_action, create_initial_state, decode_action, game_state_from_json,
    legal_action_codes, position_from_json, position_key, search_for, search_nodes,
    validate_config_json, validate_state, Action, Config, NormalDuelError, SearchDiagnostics,
    SearchOptions, SearchReport, MAX_JS_SAFE_INTEGER, NN_INPUT_PLANES, RULESET,
};

type BoundaryResult<T> = std::result::Result<T, String>;

fn parse_value(input: &str, error: &str) -> BoundaryResult<Value> {
    serde_json::from_str(input).map_err(|_| error.to_owned())
}

fn parse_config(input: &str) -> BoundaryResult<Config> {
    let value = parse_value(input, "invalid_config")?;
    config_from_value(&value)
}

fn config_from_value(value: &Value) -> BoundaryResult<Config> {
    validate_config_json(value).map_err(code)
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

/// Search options are optional as a whole. When omitted, the core defaults are
/// used; when provided, all three fields are required so the wire contract
/// cannot silently accept a misspelled or incomplete tuning request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SearchOptionsRequest {
    max_depth: Value,
    transposition_capacity: Value,
    aspiration_window: Value,
}

impl Default for SearchOptionsRequest {
    fn default() -> Self {
        let options = SearchOptions::default();
        Self {
            max_depth: Value::from(options.max_depth),
            transposition_capacity: Value::from(options.transposition_capacity),
            aspiration_window: Value::from(options.aspiration_window),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SearchNodesRequest {
    config: Value,
    state: Value,
    node_budget: Value,
    #[serde(default)]
    options: SearchOptionsRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SearchForRequest {
    config: Value,
    state: Value,
    time_budget_ms: Value,
    #[serde(default)]
    options: SearchOptionsRequest,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchDiagnosticsWire {
    root_action_count: u64,
    static_leaf_count: u64,
    immediate_goal_horizon_hits: u64,
    zero_wall_oracle_queries: u64,
    zero_wall_oracle_solutions: u64,
    zero_wall_oracle_quota_backoffs: u64,
    zero_wall_oracle_post_backoff_memo_hits: u64,
    zero_wall_oracle_post_backoff_memo_misses: u64,
    zero_wall_oracle_post_backoff_parent_exhaustions: u64,
    tt_probes: u64,
    tt_hits: u64,
    tt_bound_cutoffs: u64,
    beta_cutoffs: u64,
    pvs_researches: u64,
    aspiration_researches: u64,
    repetition_hint_only_probes: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchReportWire {
    /// Canonical action JSON decoded by the rules core. `actionCode` remains
    /// available as a deterministic policy/search diagnostic.
    action: Option<Action>,
    action_code: Option<u64>,
    score: Option<i32>,
    completed_depth: u8,
    nodes: u64,
    stopped: bool,
    principal_variation: Vec<u64>,
    committed_iteration_nodes: Vec<u64>,
    committed_iteration_scores: Vec<i32>,
    diagnostics: SearchDiagnosticsWire,
}

fn parse_search_request<T: for<'de> Deserialize<'de>>(input: &str) -> BoundaryResult<T> {
    serde_json::from_str(input).map_err(|_| "invalid_search_request".to_owned())
}

fn safe_nonnegative_integer(value: &Value) -> Option<u64> {
    let Value::Number(number) = value else {
        return None;
    };
    if let Some(integer) = number.as_u64() {
        return (integer <= MAX_JS_SAFE_INTEGER).then_some(integer);
    }
    let float = number.as_f64()?;
    (float.is_finite()
        && float >= 0.0
        && float <= MAX_JS_SAFE_INTEGER as f64
        && float.fract() == 0.0)
        .then_some(float as u64)
}

fn search_budget(value: &Value) -> BoundaryResult<u64> {
    safe_nonnegative_integer(value).ok_or_else(|| "invalid_search_budget".to_owned())
}

fn search_options(request: SearchOptionsRequest) -> BoundaryResult<SearchOptions> {
    let max_depth = safe_nonnegative_integer(&request.max_depth)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| "invalid_search_options".to_owned())?;
    let transposition_capacity = safe_nonnegative_integer(&request.transposition_capacity)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "invalid_search_options".to_owned())?;
    let aspiration_window = match &request.aspiration_window {
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                i32::try_from(integer).ok()
            } else {
                number.as_f64().and_then(|float| {
                    (float.is_finite()
                        && float >= i32::MIN as f64
                        && float <= i32::MAX as f64
                        && float.fract() == 0.0)
                        .then_some(float as i32)
                })
            }
        }
        _ => None,
    }
    .ok_or_else(|| "invalid_search_options".to_owned())?;

    Ok(SearchOptions {
        max_depth,
        transposition_capacity,
        aspiration_window,
    })
}

fn validated_search_state(
    config: &Config,
    value: &Value,
) -> BoundaryResult<wrongway_normal_duel::GameState> {
    let state = game_state_from_json(value).map_err(code)?;
    validate_state(config, &state).map_err(code)
}

fn js_safe_counter(value: u64) -> BoundaryResult<u64> {
    (value <= MAX_JS_SAFE_INTEGER)
        .then_some(value)
        .ok_or_else(|| "serialization_error".to_owned())
}

fn js_safe_usize(value: usize) -> BoundaryResult<u64> {
    js_safe_counter(u64::try_from(value).map_err(|_| "serialization_error".to_owned())?)
}

fn diagnostics_wire(diagnostics: &SearchDiagnostics) -> BoundaryResult<SearchDiagnosticsWire> {
    Ok(SearchDiagnosticsWire {
        root_action_count: js_safe_usize(diagnostics.root_action_count)?,
        static_leaf_count: js_safe_counter(diagnostics.static_leaf_count)?,
        immediate_goal_horizon_hits: js_safe_counter(diagnostics.immediate_goal_horizon_hits)?,
        zero_wall_oracle_queries: js_safe_counter(diagnostics.zero_wall_oracle_queries)?,
        zero_wall_oracle_solutions: js_safe_counter(diagnostics.zero_wall_oracle_solutions)?,
        zero_wall_oracle_quota_backoffs: js_safe_counter(
            diagnostics.zero_wall_oracle_quota_backoffs,
        )?,
        zero_wall_oracle_post_backoff_memo_hits: js_safe_counter(
            diagnostics.zero_wall_oracle_post_backoff_memo_hits,
        )?,
        zero_wall_oracle_post_backoff_memo_misses: js_safe_counter(
            diagnostics.zero_wall_oracle_post_backoff_memo_misses,
        )?,
        zero_wall_oracle_post_backoff_parent_exhaustions: js_safe_counter(
            diagnostics.zero_wall_oracle_post_backoff_parent_exhaustions,
        )?,
        tt_probes: js_safe_counter(diagnostics.tt_probes)?,
        tt_hits: js_safe_counter(diagnostics.tt_hits)?,
        tt_bound_cutoffs: js_safe_counter(diagnostics.tt_bound_cutoffs)?,
        beta_cutoffs: js_safe_counter(diagnostics.beta_cutoffs)?,
        pvs_researches: js_safe_counter(diagnostics.pvs_researches)?,
        aspiration_researches: js_safe_counter(diagnostics.aspiration_researches)?,
        repetition_hint_only_probes: js_safe_counter(diagnostics.repetition_hint_only_probes)?,
    })
}

fn search_report_wire(config: &Config, report: &SearchReport) -> BoundaryResult<SearchReportWire> {
    Ok(SearchReportWire {
        action: report
            .action_code
            .map(|action_code| decode_action(config, action_code).map_err(code))
            .transpose()?,
        action_code: report.action_code.map(js_safe_usize).transpose()?,
        score: report.score,
        completed_depth: report.completed_depth,
        nodes: js_safe_counter(report.nodes)?,
        stopped: report.stopped,
        principal_variation: report
            .principal_variation
            .iter()
            .copied()
            .map(js_safe_usize)
            .collect::<BoundaryResult<Vec<_>>>()?,
        committed_iteration_nodes: report
            .committed_iteration_nodes
            .iter()
            .copied()
            .map(js_safe_counter)
            .collect::<BoundaryResult<Vec<_>>>()?,
        committed_iteration_scores: report.committed_iteration_scores.clone(),
        diagnostics: diagnostics_wire(&report.diagnostics)?,
    })
}

fn search_nodes_impl(request_json: &str) -> BoundaryResult<String> {
    let request: SearchNodesRequest = parse_search_request(request_json)?;
    let config = config_from_value(&request.config)?;
    let state = validated_search_state(&config, &request.state)?;
    let budget = search_budget(&request.node_budget)?;
    let options = search_options(request.options)?;
    let report = search_nodes(&config, &state, budget, options).map_err(code)?;
    serialize(&search_report_wire(&config, &report)?)
}

fn search_for_impl(request_json: &str) -> BoundaryResult<String> {
    let request: SearchForRequest = parse_search_request(request_json)?;
    let config = config_from_value(&request.config)?;
    let state = validated_search_state(&config, &request.state)?;
    let budget_ms = search_budget(&request.time_budget_ms)?;
    let options = search_options(request.options)?;
    let report =
        search_for(&config, &state, Duration::from_millis(budget_ms), options).map_err(code)?;
    serialize(&search_report_wire(&config, &report)?)
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

#[wasm_bindgen(js_name = normalDuelSearchNodes)]
pub fn normal_duel_search_nodes(request_json: &str) -> std::result::Result<String, JsValue> {
    search_nodes_impl(request_json).map_err(js_error)
}

#[wasm_bindgen(js_name = normalDuelSearchFor)]
pub fn normal_duel_search_for(request_json: &str) -> std::result::Result<String, JsValue> {
    search_for_impl(request_json).map_err(js_error)
}

/// Options DTO for [`NormalDuelSelfPlayBatch`]. JSON here is fine: it is read
/// once at construction, off the hot path.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelfPlayOptionsDto {
    games: usize,
    simulations: u32,
    max_considered: u32,
    #[serde(default = "default_c_puct")]
    c_puct: f64,
    /// `"visitTemperature"` (default) or `"uniformEpsilon"`. Spelled out rather
    /// than a bool so a third recipe does not have to break the wire format.
    #[serde(default)]
    exploration: ExplorationDto,
    #[serde(default)]
    epsilon: f64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    temperature_moves: u64,
    #[serde(default = "default_ply_cap")]
    ply_cap: u64,
    #[serde(default)]
    seed_base: u32,
    #[serde(default)]
    openings: Vec<Vec<u16>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ExplorationDto {
    #[default]
    VisitTemperature,
    UniformEpsilon,
}

impl From<ExplorationDto> for Exploration {
    fn from(dto: ExplorationDto) -> Self {
        match dto {
            ExplorationDto::VisitTemperature => Exploration::VisitTemperature,
            ExplorationDto::UniformEpsilon => Exploration::UniformEpsilon,
        }
    }
}

fn default_temperature() -> f64 {
    1.0
}

fn default_c_puct() -> f64 {
    wrongway_normal_duel::puct::DEFAULT_C_PUCT
}

fn default_ply_cap() -> u64 {
    200
}

/// Batched self-play across the wasm boundary, **without JSON on the hot path**.
///
/// The per-step protocol is
///
/// ```text
/// const n = batch.collect();      // features[0..n) are filled
/// // read features through a Float32Array over wasm memory, run the network,
/// // write logits into policy[0..n) and values into value[0..n)
/// batch.submit(n);
/// ```
///
/// # The wasm-memory-growth hazard
///
/// A JS `Float32Array` built over `WebAssembly.Memory.buffer` **detaches** the
/// moment the wasm heap grows: the old `ArrayBuffer` is replaced, and the stale
/// view reads as length 0 or throws. Detached-view reads do not surface as a
/// crash in this pipeline — they surface as all-zero features and all-zero
/// policy targets, i.e. silently corrupt training data. Two defences, both
/// required:
///
/// 1. **Here:** every buffer the hot path touches is allocated at its final
///    size in [`NormalDuelSelfPlayBatch::new`] and never resized —
///    `features` (`games * 810` f32), `policy` (`games * 209` f32), `value`
///    (`games` f32), plus the record and metadata sinks reserved to their
///    worst case (`games * plyCap` records). Nothing on the steady-state path
///    asks the allocator for more pages.
/// 2. **On the JS side:** `memory.buffer` identity is compared before every
///    access and the views are rebuilt when it changes. The tree arenas still
///    allocate as a search deepens, so defence 1 shrinks the window rather than
///    closing it; defence 2 is what actually closes it.
#[wasm_bindgen(js_name = NormalDuelSelfPlayBatch)]
pub struct NormalDuelSelfPlayBatch {
    inner: SelfPlayBatch,
}

#[wasm_bindgen(js_class = NormalDuelSelfPlayBatch)]
impl NormalDuelSelfPlayBatch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        options_json: &str,
    ) -> std::result::Result<NormalDuelSelfPlayBatch, JsValue> {
        let config = parse_config(config_json).map_err(js_error)?;
        let dto: SelfPlayOptionsDto = serde_json::from_str(options_json)
            .map_err(|_| js_error("invalid_options".to_owned()))?;
        let options = SelfPlayOptions {
            games: dto.games,
            simulations: dto.simulations,
            max_considered: dto.max_considered,
            c_puct: dto.c_puct,
            exploration: dto.exploration.into(),
            epsilon: dto.epsilon,
            temperature: dto.temperature,
            temperature_moves: dto.temperature_moves,
            ply_cap: dto.ply_cap,
            seed_base: dto.seed_base,
            openings: dto.openings,
        };
        let inner =
            SelfPlayBatch::new(&config, options).map_err(|error| js_error(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Advance every unfinished game to its next leaf. Returns the number of
    /// filled feature slots; `0` means the batch is finished.
    pub fn collect(&mut self) -> std::result::Result<usize, JsValue> {
        self.inner
            .collect()
            .map_err(|error| js_error(error.to_string()))
    }

    /// Feed back `n` evaluations written into the policy and value buffers.
    pub fn submit(&mut self, n: usize) -> std::result::Result<(), JsValue> {
        self.inner
            .submit(n)
            .map_err(|error| js_error(error.to_string()))
    }

    #[wasm_bindgen(js_name = isDone)]
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.inner.done()
    }

    /// Drain finished games into the record sink; returns the record count.
    #[wasm_bindgen(js_name = takeRecords)]
    pub fn take_records(&mut self) -> usize {
        self.inner.take_records()
    }

    // --- Buffer addresses. Byte offsets into `WebAssembly.Memory.buffer`. ---
    //
    // Re-read these after *every* call into wasm: a moved buffer and a grown
    // heap are indistinguishable from JS otherwise.

    #[wasm_bindgen(js_name = featuresPtr)]
    #[must_use]
    pub fn features_ptr(&self) -> u32 {
        self.inner.features().as_ptr() as u32
    }

    #[wasm_bindgen(js_name = legalMaskPtr)]
    #[must_use]
    pub fn legal_mask_ptr(&self) -> u32 {
        self.inner.legal_mask().as_ptr() as u32
    }

    #[wasm_bindgen(js_name = policyPtr)]
    #[must_use]
    pub fn policy_ptr(&mut self) -> u32 {
        self.inner.policy_mut().as_ptr() as u32
    }

    #[wasm_bindgen(js_name = valuePtr)]
    #[must_use]
    pub fn value_ptr(&mut self) -> u32 {
        self.inner.value_mut().as_ptr() as u32
    }

    #[wasm_bindgen(js_name = recordsPtr)]
    #[must_use]
    pub fn records_ptr(&self) -> u32 {
        self.inner.records().as_ptr() as u32
    }

    #[wasm_bindgen(js_name = recordMetaPtr)]
    #[must_use]
    pub fn record_meta_ptr(&self) -> u32 {
        self.inner.record_meta().as_ptr() as u32
    }

    /// The repetition windows, as a `Uint32Array` — NOT a `Float32Array`.
    ///
    /// Read AFTER `takeRecords()` and copy out before the next call into wasm.
    /// Unlike the record sink this buffer is not reserved to its worst case, so
    /// `takeRecords` may grow the heap and detach anything built before it; there
    /// is nothing to detach, because there is nothing to read before then. See
    /// [`SelfPlayBatch::record_window`].
    #[wasm_bindgen(js_name = recordWindowPtr)]
    #[must_use]
    pub fn record_window_ptr(&self) -> u32 {
        self.inner.record_window().as_ptr() as u32
    }

    // --- Lengths, in elements. ---

    #[wasm_bindgen(js_name = featuresLen)]
    #[must_use]
    pub fn features_len(&self) -> usize {
        self.inner.features().len()
    }

    #[wasm_bindgen(js_name = policyLen)]
    #[must_use]
    pub fn policy_len(&mut self) -> usize {
        self.inner.policy_mut().len()
    }

    #[wasm_bindgen(js_name = valueLen)]
    #[must_use]
    pub fn value_len(&mut self) -> usize {
        self.inner.value_mut().len()
    }

    #[wasm_bindgen(js_name = recordsLen)]
    #[must_use]
    pub fn records_len(&self) -> usize {
        self.inner.records().len()
    }

    #[wasm_bindgen(js_name = recordMetaLen)]
    #[must_use]
    pub fn record_meta_len(&self) -> usize {
        self.inner.record_meta().len()
    }

    /// `RECORD_WINDOW_FIELDS` u32 per window ENTRY, and the entry count varies
    /// per record — so this is not `recordCount * k`. Slice it with the
    /// `windowLen` column of each record, in record order.
    #[wasm_bindgen(js_name = recordWindowLen)]
    #[must_use]
    pub fn record_window_len(&self) -> usize {
        self.inner.record_window().len()
    }

    // --- Run summary. Cold path; JSON is fine. ---

    /// `-1` B win, `0` draw, `1` A win, `2` ongoing, one per game.
    ///
    /// Ongoing is its own code rather than being folded into draw. A game that
    /// stops because it hit the shard's ply cap is *unfinished*, which is what
    /// the incumbent shard worker reports it as (`outcomes: {ongoing: 1}`);
    /// mapping it to `0` made the two workers' shard stats disagree about what
    /// had happened — one calling a truncated game a draw, the other calling it
    /// ongoing — even when the records themselves were identical. `z` is 0
    /// either way, so this never affected training, only the reporting that
    /// would have to be trusted to notice if it had.
    #[must_use]
    pub fn outcomes(&self) -> Vec<i32> {
        self.inner
            .outcomes()
            .into_iter()
            .map(|outcome| match outcome {
                GameOutcome::Win(wrongway_normal_duel::Player::A) => 1,
                GameOutcome::Win(wrongway_normal_duel::Player::B) => -1,
                GameOutcome::Draw => 0,
                GameOutcome::Ongoing => 2,
            })
            .collect()
    }

    #[wasm_bindgen(js_name = pliesPlayed)]
    #[must_use]
    pub fn plies_played(&self) -> Vec<u32> {
        self.inner
            .plies_played()
            .into_iter()
            .map(|plies| u32::try_from(plies).unwrap_or(u32::MAX))
            .collect()
    }
}

/// One search-to-move over the native PUCT tree, with the network evaluated by
/// the caller.
///
/// `NormalDuelSelfPlayBatch` cannot serve matchplay: it owns the game loop and
/// picks its own moves, so it has no way to play an externally chosen opponent
/// reply. This wrapper exposes the single-search coroutine instead — the caller
/// supplies the root state, pumps one leaf at a time, and reads the chosen
/// action. That is one boundary crossing per NODE, but the crossing itself is
/// nanoseconds; what it removes is the JS tree's ~4 ms/simulation of move
/// generation. The network round trip is paid identically by both paths.
///
/// The per-move protocol is
///
/// ```text
/// const s = new NormalDuelSearch(configJson, stateJson, optionsJson);
/// while (s.nextLeaf()) {                // features[0..featuresLen) filled
///   s.pendingLeafMask();                // mask[0..policyLen)  filled
///   // run the network, write probabilities into policy[0..policyLen)
///   s.submit(value);
/// }
/// const chosen = s.actionCode();
/// ```
///
/// The same wasm-memory-growth hazard documented on `NormalDuelSelfPlayBatch`
/// applies: the tree arenas grow as a search deepens, so the JS side MUST
/// compare `memory.buffer` identity before every access and rebuild its views.
#[wasm_bindgen(js_name = NormalDuelSearch)]
pub struct NormalDuelSearch {
    config: Config,
    inner: PuctTreeSearch,
    features: Vec<f32>,
    policy: Vec<f32>,
    mask: Vec<f32>,
    /// One value per leaf in the current batch; unused by the one-at-a-time API.
    values: Vec<f64>,
    /// How many leaves the feature and policy buffers are sized for.
    batch_width: usize,
}

/// Options DTO for [`NormalDuelSearch`]. Read once at construction.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchOptionsDto {
    simulations: u32,
    max_considered: u32,
    #[serde(default = "default_c_puct")]
    c_puct: f64,
    /// The JS reference drives one search from `createLcg32(seed)`; the same
    /// seed here is what makes the two trees choose the same move.
    seed: u32,
}

/// The improved policy over a FINISHED search, or the refusal that keeps a
/// partial tree from being recorded as a training target.
///
/// **Guarded where its four siblings are not, and that asymmetry is the point.**
/// `actionCode`, `rootValue`, `simulationsUsed` and `visitCounts` are read in
/// flows where doneness is structural — the driver reads them after its own
/// `while (nextLeaf())` has already exited — and what they feed is move
/// selection and telemetry, where a partial read shows up as a bad move or an
/// odd number. This one labels TRAINING CORPORA, and a partial read of it shows
/// up as nothing at all: at 10 of 48 simulations it is still a strictly
/// ascending, all-positive distribution over every legal action summing to 1.0,
/// structurally indistinguishable from the finished answer while differing from
/// it by a large fraction of some action's mass. A consumer that broke its leaf
/// loop early — a deadline, a swallowed exception, a resumed shard — would
/// mislabel every position it touched and pass every downstream audit. So the
/// guard sits on the accessor whose wrong answer nobody can see, and the other
/// four are left alone rather than uniformly wrapped for symmetry's sake.
///
/// Split out of the `#[wasm_bindgen]` method for the reason every other error
/// path in this file is: `JsValue::from_str` aborts the process off the wasm
/// target, so a refusal is only reachable by a native test as a
/// [`BoundaryResult`].
fn improved_policy_impl(search: &PuctTreeSearch) -> BoundaryResult<Vec<f64>> {
    if !search.is_done() {
        return Err("search_not_done".to_owned());
    }
    Ok(search
        .result()
        .improved_policy
        .into_iter()
        .flat_map(|(code, probability)| [f64::from(code), probability])
        .collect())
}

#[wasm_bindgen(js_class = NormalDuelSearch)]
impl NormalDuelSearch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        state_json: &str,
        options_json: &str,
    ) -> std::result::Result<NormalDuelSearch, JsValue> {
        let config = parse_config(config_json).map_err(js_error)?;
        let state_value = parse_value(state_json, "invalid_state").map_err(js_error)?;
        // Same validation the other search entry points use: decode, then check
        // the state against this config, so a state from a different board
        // cannot reach the tree.
        let state = validated_search_state(&config, &state_value).map_err(js_error)?;
        let dto: SearchOptionsDto = serde_json::from_str(options_json)
            .map_err(|_| js_error("invalid_options".to_owned()))?;
        let params = PuctParams {
            simulations: dto.simulations,
            max_considered: dto.max_considered,
            c_puct: dto.c_puct,
        };
        let inner = PuctTreeSearch::from_state(&config, &state, params, Lcg32::new(dto.seed))
            .map_err(|error| js_error(error.reason().to_owned()))?;
        let features = vec![0.0; NN_INPUT_PLANES * config.cells()];
        let policy = vec![0.0; config.policy_size()];
        let mask = vec![0.0; config.policy_size()];
        Ok(Self {
            config,
            inner,
            features,
            policy,
            mask,
            values: vec![0.0],
            batch_width: 1,
        })
    }

    /// One leaf's policy width, independent of the batch buffers.
    ///
    /// `policyLen()` is the BUFFER length and `configureBatch` multiplies it, so on a
    /// REUSED search it reads back `batch` times too large -- and a caller that sizes
    /// its mask view from it runs thousands of floats past a buffer that never grew,
    /// which corrupts wasm memory rather than throwing. Sibling of `featureStride`.
    #[wasm_bindgen(js_name = policyWidth)]
    #[must_use]
    pub fn policy_width(&self) -> usize {
        self.config.policy_size()
    }

    /// One leaf's feature width, independent of the batch buffers.
    ///
    /// `featuresLen()` is the BUFFER length, which `configureBatch` multiplies. A
    /// caller that derives the stride from it is right exactly once -- on a search
    /// whose buffers have never been resized -- and wrong on every reused search,
    /// where it reads back a stride `batch` times too large.
    #[wasm_bindgen(js_name = featureStride)]
    #[must_use]
    pub fn feature_stride(&self) -> usize {
        NN_INPUT_PLANES * self.config.cells()
    }

    /// Re-root this search at the node reached by `codes`, keeping the statistics
    /// underneath, and reset the schedule from `options`.
    ///
    /// Returns `false` when that path was never expanded -- the caller then builds a
    /// fresh search as before. A miss is ordinary: the opponent's reply is only in
    /// the tree if the previous search actually visited it.
    ///
    /// This is tree reuse across moves. What is inherited is the VALUE estimates;
    /// the Gumbel plan is re-run from scratch, so a resumed search still spends its
    /// whole budget. It makes each simulation better informed, it does not make
    /// simulations unnecessary.
    #[wasm_bindgen(js_name = reroot)]
    pub fn reroot(&mut self, codes: Vec<u16>, options: &str) -> std::result::Result<bool, JsValue> {
        let dto: SearchOptionsDto =
            serde_json::from_str(options).map_err(|error| js_error(error.to_string()))?;
        if !self.inner.reroot(&self.config, &codes) {
            return Ok(false);
        }
        self.inner
            .restart(
                PuctParams {
                    simulations: dto.simulations,
                    max_considered: dto.max_considered,
                    c_puct: dto.c_puct,
                },
                Lcg32::new(dto.seed),
            )
            .map_err(|error| js_error(error.reason().to_owned()))?;
        Ok(true)
    }

    /// Nodes currently held. A reroot that inherits nothing is not worth having, and
    /// this is how a caller can tell without guessing.
    #[wasm_bindgen(js_name = nodeCount)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Size the buffers for batched collection, and return the width in use.
    ///
    /// `featuresPtr` and `policyPtr` are invalidated by this call: it reallocates.
    #[wasm_bindgen(js_name = configureBatch)]
    pub fn configure_batch(&mut self, width: usize) -> std::result::Result<usize, JsValue> {
        if width == 0 {
            return Err(js_error("invalid_batch_width".to_owned()));
        }
        let stride = NN_INPUT_PLANES * self.config.cells();
        self.features.resize(stride * width, 0.0);
        self.policy.resize(self.config.policy_size() * width, 0.0);
        self.values.resize(width, 0.0);
        self.batch_width = width;
        Ok(width)
    }

    /// The in-flight penalty. `0` keeps the search bit-identical to the
    /// one-at-a-time API; anything else trades exactness for larger batches.
    #[wasm_bindgen(js_name = setVirtualLoss)]
    pub fn set_virtual_loss(&mut self, virtual_loss: f64) -> std::result::Result<(), JsValue> {
        self.inner
            .set_virtual_loss(virtual_loss)
            .map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Fill up to `configureBatch(width)` leaves' features at once and return how
    /// many were written; `0` means the search is finished. Evaluate all of them,
    /// write the policies into `policy` slot by slot and the values into `values`,
    /// then call `submitBatch(n)`.
    #[wasm_bindgen(js_name = collectLeaves)]
    pub fn collect_leaves(&mut self) -> std::result::Result<usize, JsValue> {
        let width = self.batch_width;
        self.inner
            .collect_leaves(&self.config, &mut self.features, width)
            .map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Hand back `n` evaluations for the leaves `collectLeaves` produced.
    #[wasm_bindgen(js_name = submitBatch)]
    pub fn submit_batch(&mut self, n: usize) -> std::result::Result<(), JsValue> {
        let policy = std::mem::take(&mut self.policy);
        let values = std::mem::take(&mut self.values);
        let outcome = self.inner.submit_batch(&self.config, &policy, &values, n);
        self.policy = policy;
        self.values = values;
        outcome.map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Fill `mask` with the `index`-th batched leaf's legal-action mask.
    #[wasm_bindgen(js_name = batchLeafMask)]
    pub fn batch_leaf_mask(&mut self, index: usize) -> std::result::Result<(), JsValue> {
        let mut mask = std::mem::take(&mut self.mask);
        let outcome = self.inner.batch_leaf_mask(&self.config, index, &mut mask);
        self.mask = mask;
        outcome.map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Base of the per-leaf value buffer; `n` f64s, one per collected leaf.
    #[wasm_bindgen(js_name = valuesPtr)]
    #[must_use]
    pub fn values_ptr(&self) -> u32 {
        self.values.as_ptr() as u32
    }

    /// Advance to the next leaf awaiting evaluation, filling `features`.
    /// Returns `false` when the search is finished.
    #[wasm_bindgen(js_name = nextLeaf)]
    pub fn next_leaf(&mut self) -> std::result::Result<bool, JsValue> {
        self.inner
            .next_leaf(&self.config, &mut self.features)
            .map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Fill `mask` with the pending leaf's legal-action mask.
    #[wasm_bindgen(js_name = pendingLeafMask)]
    pub fn pending_leaf_mask(&mut self) -> std::result::Result<(), JsValue> {
        self.inner
            .pending_leaf_mask(&self.config, &mut self.mask)
            .map_err(|error| js_error(error.reason().to_owned()))
    }

    /// Hand the pending leaf's evaluation back to the tree. The caller has
    /// already written probabilities into `policy[0..policyLen)`.
    pub fn submit(&mut self, value: f64) -> std::result::Result<(), JsValue> {
        let policy = std::mem::take(&mut self.policy);
        let outcome = self.inner.submit(&self.config, &policy, value);
        self.policy = policy;
        outcome.map_err(|error| js_error(error.reason().to_owned()))
    }

    #[wasm_bindgen(js_name = isDone)]
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// The chosen action code. Only meaningful once the search is done.
    #[wasm_bindgen(js_name = actionCode)]
    #[must_use]
    pub fn action_code(&self) -> u16 {
        self.inner.result().action_code
    }

    /// `simulationsUsed`, for the harness's node-budget check.
    #[wasm_bindgen(js_name = simulationsUsed)]
    #[must_use]
    pub fn simulations_used(&self) -> u32 {
        self.inner.result().simulations_used
    }

    #[wasm_bindgen(js_name = rootValue)]
    #[must_use]
    pub fn root_value(&self) -> f64 {
        self.inner.result().root_value
    }

    /// `(code, visits)` pairs flattened, ascending by code — the parity surface.
    #[wasm_bindgen(js_name = visitCounts)]
    #[must_use]
    pub fn visit_counts(&self) -> Vec<u32> {
        self.inner
            .result()
            .visit_counts
            .into_iter()
            .flat_map(|(code, visits)| [u32::from(code), visits])
            .collect()
    }

    /// `(code, pi')` pairs flattened, ascending by code — the policy target.
    ///
    /// Read only once `nextLeaf()` has returned false; before that the tree is
    /// partial and the distribution is plausible but wrong. Unlike its
    /// siblings this one does not take the caller's word for that — it throws
    /// `search_not_done` instead. [`improved_policy_impl`] carries the argument
    /// for why this accessor and not the other four.
    ///
    /// This is `PuctResult::improved_policy` handed across unchanged: the
    /// value the core already produced and the one `SelfPlayBatch`'s record
    /// writer narrows to `f32` and stores as `policyTarget`. Deliberately not
    /// recomputed here from visits, priors or the root value. A second
    /// derivation would be a second definition of the training target, and the
    /// two could drift while every test of each one separately stayed green.
    /// `improved_policy_is_exactly_what_self_play_records_as_the_policy_target`
    /// pins this against what a real self-play game writes for the same root.
    ///
    /// **Identical to the record only after the recorder's own narrowing.**
    /// These are `f64`; `SelfPlayBatch` stores `probability as f32`. A consumer
    /// that keeps the `f64` and expects it to match a re-derived self-play label
    /// is out by about `1e-8` relative, and by a whole order of magnitude in the
    /// deep tail, where an `f64` `1e-60` narrows to a hard `0.0`. Store these
    /// into the `f32` record format and the two are bit-identical — which is
    /// what the parity test asserts, and what a consumer holding wider labels
    /// does not get.
    ///
    /// **Not `visitCounts` normalised**, which is the mistake this accessor
    /// exists to prevent. Under sequential halving the counts are the search
    /// *schedule*: both finalists take the same visits whichever one won, and
    /// every legal action outside the considered set takes zero. The improved
    /// policy reads each action's completed Q instead, so it separates the
    /// finalists and it covers the actions the Gumbel draw never considered —
    /// which is also why this list is longer than `visitCounts`'s whenever
    /// `maxConsidered` is below the legal count.
    ///
    /// Beware the name in the frozen JavaScript reference: `puctSearch` returns
    /// an `improvedPolicy` field too, and it is `encodePolicyTarget` over the
    /// visit counts — the pre-`puct-az-tree-v2` target, and so exactly the
    /// distribution this one is not. They agree on neither support nor values.
    ///
    /// Pairs rather than a dense `policyLen` vector, matching `visitCounts`:
    /// the support is exactly the root's legal actions, so a consumer that
    /// scatters these into its own zeroed buffer cannot confuse "illegal here"
    /// with "targeted at zero". Codes are `u16` and so exact in `f64`, which
    /// lets one array carry both halves of the pair.
    #[wasm_bindgen(js_name = improvedPolicy)]
    pub fn improved_policy(&self) -> std::result::Result<Vec<f64>, JsValue> {
        improved_policy_impl(&self.inner).map_err(js_error)
    }

    /// Visits under the node reached by playing `codes` from the root; `-1` when that
    /// path was never expanded.
    ///
    /// READ-ONLY DIAGNOSTIC for the tree-reuse question: with `codes = [our move,
    /// their reply]`, this is exactly what the next search would inherit if it reused
    /// this tree rather than rebuilding. It changes no search state.
    ///
    /// Returns i32 rather than Option<u32> because `-1` survives the JS boundary as a
    /// plain number, and "never expanded" (-1) must stay distinguishable from
    /// "expanded but never visited" (0) -- collapsing those would flatter reuse.
    #[wasm_bindgen(js_name = subtreeVisitsAfter)]
    #[must_use]
    pub fn subtree_visits_after(&self, codes: &[u16]) -> i32 {
        self.inner
            .subtree_visits_after(codes)
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(-1)
    }

    #[wasm_bindgen(js_name = featuresPtr)]
    #[must_use]
    pub fn features_ptr(&self) -> u32 {
        self.features.as_ptr() as u32
    }

    #[wasm_bindgen(js_name = featuresLen)]
    #[must_use]
    pub fn features_len(&self) -> usize {
        self.features.len()
    }

    #[wasm_bindgen(js_name = policyPtr)]
    #[must_use]
    pub fn policy_ptr(&self) -> u32 {
        self.policy.as_ptr() as u32
    }

    #[wasm_bindgen(js_name = policyLen)]
    #[must_use]
    pub fn policy_len(&self) -> usize {
        self.policy.len()
    }

    #[wasm_bindgen(js_name = maskPtr)]
    #[must_use]
    pub fn mask_ptr(&self) -> u32 {
        self.mask.as_ptr() as u32
    }
}

/// Record layout constants, so the JS driver never hard-codes a stride.
///
/// This is also the ONLY thing that tells a driver which record shape it is
/// looking at, and that is deliberate. The obvious alternative — a
/// `recordVersion` on [`SelfPlayOptionsDto`] — would have to be sent to get v2
/// records, and a key sent unconditionally to an engine that predates it is the
/// failure this project has already paid for once: the DTO refuses unknown
/// fields, so every shard dies at construction with `invalid_options` before the
/// first game. So nothing new crosses the boundary INWARD. The engine emits the
/// widest record it knows, says so here, and the driver dispatches on the
/// answer. New driver against an old engine reads `stateFields: 0` (absent) and
/// writes a v1 shard; old driver against a new engine trips its own
/// `recordFloats === features + 2*policy + 1` assertion and stops. Neither can
/// write a mis-shaped shard.
///
/// `stateFields` and `windowFields` are new keys in the OUTPUT, which is safe:
/// this is parsed with `JSON.parse`, and no consumer asserts an exact key set.
#[wasm_bindgen(js_name = normalDuelSelfPlayLayout)]
#[must_use]
pub fn normal_duel_self_play_layout() -> String {
    format!(
        r#"{{"features":{RECORD_FEATURES},"policy":{RECORD_POLICY},"recordFloats":{RECORD_FLOATS},"metaFields":{RECORD_META_FIELDS},"stateFields":{RECORD_STATE_FIELDS},"windowFields":{RECORD_WINDOW_FIELDS},"recordVersion":{RECORD_VERSION}}}"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wrongway_normal_duel::mock_evaluator;

    const PERFT: &str = include_str!("../../../tests/fixtures/normal-duel-perft-v1.json");
    const SEARCH_PARITY: &str =
        include_str!("../../../tests/fixtures/normal-duel-wasm-search-nodes-v1.json");

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

    fn initial_search_request(node_budget: u64) -> Value {
        let config = config();
        let state: Value =
            serde_json::from_str(&initial_state_impl(&config.to_string()).unwrap()).unwrap();
        json!({
            "config": config,
            "state": state,
            "nodeBudget": node_budget,
            "options": {
                "maxDepth": 6,
                "transpositionCapacity": 1024,
                "aspirationWindow": 64
            }
        })
    }

    fn assert_legal_search_report(report: &Value, config: &Value, state: &Value) {
        let legal: Vec<usize> = serde_json::from_str(
            &legal_action_codes_impl(&config.to_string(), &state.to_string()).unwrap(),
        )
        .unwrap();
        let action = report["actionCode"].as_u64().unwrap() as usize;
        assert!(legal.contains(&action));
        let expected_action = serde_json::to_value(
            decode_action(&config_from_value(config).unwrap(), action).unwrap(),
        )
        .unwrap();
        assert_eq!(report["action"], expected_action);
        assert_eq!(
            report["principalVariation"][0].as_u64(),
            Some(action as u64)
        );
        assert_eq!(
            report["diagnostics"]["rootActionCount"].as_u64(),
            Some(legal.len() as u64)
        );
        assert!(report["nodes"].is_u64());
        assert!(report["diagnostics"]["ttProbes"].is_u64());
        assert!(report["diagnostics"]["staticLeafCount"].is_u64());
        assert!(report["diagnostics"]["immediateGoalHorizonHits"].is_u64());
        assert!(report["diagnostics"]["zeroWallOracleQueries"].is_u64());
        assert!(report["diagnostics"]["zeroWallOracleSolutions"].is_u64());
        assert!(report["diagnostics"]["zeroWallOracleQuotaBackoffs"].is_u64());
        assert!(report["diagnostics"]["zeroWallOraclePostBackoffMemoHits"].is_u64());
        assert!(report["diagnostics"]["zeroWallOraclePostBackoffMemoMisses"].is_u64());
        assert!(report["diagnostics"]["zeroWallOraclePostBackoffParentExhaustions"].is_u64());
        assert!(report["committedIterationNodes"].is_array());
        assert!(report["committedIterationScores"].is_array());
        let completed_depth = report["completedDepth"].as_u64().unwrap() as usize;
        assert_eq!(
            report["committedIterationNodes"].as_array().unwrap().len(),
            completed_depth
        );
        assert_eq!(
            report["committedIterationScores"].as_array().unwrap().len(),
            completed_depth
        );
        assert!(report.get("action").is_some());
        assert!(report.get("action_code").is_none());
        assert!(report.get("principal_variation").is_none());
    }

    #[test]
    fn native_search_nodes_is_deterministic_and_emits_a_legal_camel_case_report() {
        let request = initial_search_request(500);
        let first = search_nodes_impl(&request.to_string()).unwrap();
        let second = search_nodes_impl(&request.to_string()).unwrap();
        assert_eq!(first, second);

        let report: Value = serde_json::from_str(&first).unwrap();
        assert!(report["nodes"].as_u64().unwrap() <= 500);
        assert_legal_search_report(&report, &request["config"], &request["state"]);
    }

    #[test]
    fn native_canonical_oracle_quota_diagnostics_are_serialized() {
        let mut request = initial_search_request(5_000);
        request["config"]["initialStock"] = json!({"A": 0, "B": 0});
        request["state"] =
            serde_json::from_str(&initial_state_impl(&request["config"].to_string()).unwrap())
                .unwrap();
        request["options"] = json!({
            "maxDepth": 5,
            "transpositionCapacity": 1024,
            "aspirationWindow": 32
        });

        let first: Value =
            serde_json::from_str(&search_nodes_impl(&request.to_string()).unwrap()).unwrap();
        let second: Value =
            serde_json::from_str(&search_nodes_impl(&request.to_string()).unwrap()).unwrap();
        assert_eq!(first, second);
        assert!(first["completedDepth"].as_u64().unwrap() > 0);
        assert_eq!(first["diagnostics"]["zeroWallOracleQueries"], 1);
        assert_eq!(first["diagnostics"]["zeroWallOracleSolutions"], 0);
        assert_eq!(first["diagnostics"]["zeroWallOracleQuotaBackoffs"], 1);
        assert!(
            first["diagnostics"]["zeroWallOraclePostBackoffMemoMisses"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(first["diagnostics"]["zeroWallOraclePostBackoffMemoHits"], 0);
        assert_eq!(
            first["diagnostics"]["zeroWallOraclePostBackoffParentExhaustions"],
            0
        );
        assert_legal_search_report(&first, &request["config"], &request["state"]);
    }

    #[test]
    fn native_search_nodes_matches_the_committed_wasm_parity_fixture() {
        let fixture: Value = serde_json::from_str(SEARCH_PARITY).unwrap();
        let request = &fixture["request"];
        let expected = &fixture["report"];
        let actual: Value =
            serde_json::from_str(&search_nodes_impl(&request.to_string()).unwrap()).unwrap();

        assert_eq!(actual, *expected);

        let default_horizon = &fixture["defaultHorizon"];
        let default_request = &default_horizon["request"];
        let default_expected = &default_horizon["report"];
        let default_actual: Value =
            serde_json::from_str(&search_nodes_impl(&default_request.to_string()).unwrap())
                .unwrap();

        assert_eq!(default_actual, *default_expected);
        assert!(
            default_actual["diagnostics"]["immediateGoalHorizonHits"]
                .as_u64()
                .unwrap()
                > 0
        );
    }

    #[test]
    fn native_search_request_is_strict_and_uses_defaults_only_when_options_are_omitted() {
        let mut omitted = initial_search_request(4096);
        omitted.as_object_mut().unwrap().remove("options");
        let mut explicit_defaults = omitted.clone();
        explicit_defaults["options"] = json!({
            "maxDepth": 64,
            "transpositionCapacity": 262_144,
            "aspirationWindow": 256,
        });
        let omitted_report = search_nodes_impl(&omitted.to_string()).unwrap();
        let explicit_defaults_report = search_nodes_impl(&explicit_defaults.to_string()).unwrap();
        assert_eq!(omitted_report, explicit_defaults_report);

        let report: Value = serde_json::from_str(&omitted_report).unwrap();
        assert_legal_search_report(&report, &omitted["config"], &omitted["state"]);

        let mut unknown_top_level = omitted.clone();
        unknown_top_level["future"] = json!(true);
        assert_eq!(
            search_nodes_impl(&unknown_top_level.to_string()),
            Err("invalid_search_request".into())
        );
        let mut partial_options = initial_search_request(1);
        partial_options["options"] = json!({"maxDepth": 1});
        assert_eq!(
            search_nodes_impl(&partial_options.to_string()),
            Err("invalid_search_request".into())
        );
        let mut unknown_option = initial_search_request(1);
        unknown_option["options"]["future"] = json!(true);
        assert_eq!(
            search_nodes_impl(&unknown_option.to_string()),
            Err("invalid_search_request".into())
        );
        assert_eq!(
            search_nodes_impl("[]"),
            Err("invalid_search_request".into())
        );
    }

    #[test]
    fn native_search_preserves_core_error_codes_for_budget_options_config_and_state() {
        let mut zero_budget = initial_search_request(0);
        assert_eq!(
            search_nodes_impl(&zero_budget.to_string()),
            Err("invalid_search_budget".into())
        );
        zero_budget["nodeBudget"] = json!(-1);
        assert_eq!(
            search_nodes_impl(&zero_budget.to_string()),
            Err("invalid_search_budget".into())
        );
        zero_budget["nodeBudget"] = json!(0.5);
        assert_eq!(
            search_nodes_impl(&zero_budget.to_string()),
            Err("invalid_search_budget".into())
        );

        let mut invalid_options = initial_search_request(1);
        invalid_options["options"]["maxDepth"] = json!(0);
        assert_eq!(
            search_nodes_impl(&invalid_options.to_string()),
            Err("invalid_search_options".into())
        );
        let mut invalid_config = initial_search_request(1);
        invalid_config["config"]["rows"] = json!(8);
        assert_eq!(
            search_nodes_impl(&invalid_config.to_string()),
            Err("invalid_config".into())
        );
        let mut invalid_state = initial_search_request(1);
        invalid_state["state"] = json!({});
        assert_eq!(
            search_nodes_impl(&invalid_state.to_string()),
            Err("invalid_state".into())
        );
    }

    #[test]
    fn native_search_reports_terminal_roots_and_deadline_fallbacks() {
        let fixture: Value = serde_json::from_str(PERFT).unwrap();
        let terminal = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "terminal-goal-win-7x7-a")
            .unwrap();
        let terminal_request = json!({
            "config": fixture["configs"][terminal["configId"].as_str().unwrap()],
            "state": terminal["state"],
            "nodeBudget": 1
        });
        let terminal_report: Value =
            serde_json::from_str(&search_nodes_impl(&terminal_request.to_string()).unwrap())
                .unwrap();
        assert!(terminal_report["actionCode"].is_null());
        assert!(terminal_report["action"].is_null());
        assert_eq!(terminal_report["principalVariation"], json!([]));
        assert_eq!(terminal_report["completedDepth"], 0);
        assert_eq!(terminal_report["committedIterationNodes"], json!([]));
        assert_eq!(terminal_report["committedIterationScores"], json!([]));

        let request = initial_search_request(1);
        let deadline_request = json!({
            "config": request["config"],
            "state": request["state"],
            "timeBudgetMs": 1,
            "options": request["options"]
        });
        let deadline_report: Value =
            serde_json::from_str(&search_for_impl(&deadline_request.to_string()).unwrap()).unwrap();
        assert_legal_search_report(
            &deadline_report,
            &deadline_request["config"],
            &deadline_request["state"],
        );

        let mut invalid_deadline = deadline_request;
        invalid_deadline["timeBudgetMs"] = json!(0);
        assert_eq!(
            search_for_impl(&invalid_deadline.to_string()),
            Err("invalid_search_budget".into())
        );
    }

    /* -------------------------------------------------------------- *
     * `improvedPolicy`: the recorded policy target, across the boundary
     * -------------------------------------------------------------- */

    /// Budget for the parity searches below: small enough to keep the suite
    /// quick, large enough that sequential halving runs several rounds and the
    /// improved policy stops being the renormalised prior.
    const PARITY_SIMS: u32 = 48;
    /// The production shape — far fewer candidates than the 131 legal actions
    /// at a fresh 9x9 root, so the considered set is a strict subset of the
    /// legal set and the two accessors cannot have the same support.
    const PARITY_CONSIDERED: u32 = 8;
    const PARITY_SEED: u32 = 20_260_818;
    /// Legal actions at a fresh 9x9 root, pinned so that a config change cannot
    /// quietly turn the subset case below into the equality case.
    const PARITY_LEGAL: usize = 131;

    /// The shared mock evaluator, with the value already round-tripped through
    /// `f32`.
    ///
    /// `SelfPlayBatch` carries leaf values in an `f32` slot and widens them
    /// again at `submit`, while `NormalDuelSearch::submit` takes the `f64`
    /// straight from JS. Narrowing here is what makes the two searches receive
    /// the same number rather than two a ULP apart, and a ULP is enough to flip
    /// one halving comparison and with it the whole tree.
    fn parity_leaf(features: &[f32], policy_out: &mut [f32]) -> f64 {
        f64::from(mock_evaluator::evaluate(features, policy_out) as f32)
    }

    /// Play one self-play game through the real record writer and return the
    /// `policyTarget` slice of its first recorded ply — the bytes that go to
    /// disk, read out of the record buffer rather than recomputed.
    fn recorded_policy_target(config: &Config, options: SelfPlayOptions) -> Vec<f32> {
        let mut batch = SelfPlayBatch::new(config, options).expect("self-play batch");
        let mut policy = vec![0.0_f32; RECORD_POLICY];
        loop {
            let leaves = batch.collect().expect("collect");
            if leaves == 0 {
                break;
            }
            for slot in 0..leaves {
                let features =
                    batch.features()[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES].to_vec();
                let value = parity_leaf(&features, &mut policy);
                batch.policy_mut()[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY]
                    .copy_from_slice(&policy);
                batch.value_mut()[slot] = value as f32;
            }
            batch.submit(leaves).expect("submit");
        }
        assert_eq!(batch.take_records(), 1, "ply_cap 1 records exactly one ply");
        batch.records()[RECORD_FEATURES..RECORD_FEATURES + RECORD_POLICY].to_vec()
    }

    /// Drive a `NormalDuelSearch` to completion over the same mock, through the
    /// same three calls the JS protocol makes. The fields stand in for the
    /// pointers: a native test cannot take a `u32` offset into wasm memory, and
    /// `submit` reads the same buffer either way.
    ///
    /// Returns the ROOT's features, which `next_leaf` hands out first, before
    /// any descent.
    fn drive_search(search: &mut NormalDuelSearch) -> Vec<f32> {
        let mut policy = vec![0.0_f32; RECORD_POLICY];
        let mut root_features: Vec<f32> = Vec::new();
        while search.next_leaf().expect("next_leaf") {
            search.pending_leaf_mask().expect("pending_leaf_mask");
            if root_features.is_empty() {
                root_features = search.features.clone();
            }
            let value = parity_leaf(&search.features, &mut policy);
            search.policy.copy_from_slice(&policy);
            search.submit(value).expect("submit");
        }
        root_features
    }

    /// The boundary search at the root of a fresh 9x9 game, built from the JSON
    /// the JS driver sends.
    fn parity_search(considered: u32, c_puct: f64) -> (String, String, NormalDuelSearch) {
        let config_json = config().to_string();
        let state_json = initial_state_impl(&config_json).expect("initial state");
        let options_json = json!({
            "simulations": PARITY_SIMS,
            "maxConsidered": considered,
            "cPuct": c_puct,
            "seed": PARITY_SEED,
        })
        .to_string();
        let search =
            NormalDuelSearch::new(&config_json, &state_json, &options_json).expect("search");
        (config_json, state_json, search)
    }

    /// Legal action codes at a root, as the boundary reports them.
    fn legal_codes(config_json: &str, state_json: &str) -> Vec<u16> {
        serde_json::from_str(&legal_action_codes_impl(config_json, state_json).expect("legal"))
            .expect("legal codes")
    }

    /// Unflatten the accessor's output, checking the pairing convention itself:
    /// an even length, and codes that survive `u16 -> f64 -> u16` exactly.
    fn improved_pairs(flat: &[f64]) -> Vec<(u16, f64)> {
        assert_eq!(flat.len() % 2, 0, "flattened pairs have an even length");
        flat.chunks_exact(2)
            .map(|pair| {
                let code = pair[0] as u16;
                assert_eq!(
                    f64::from(code),
                    pair[0],
                    "code {} is not an exact u16",
                    pair[0]
                );
                (code, pair[1])
            })
            .collect()
    }

    /// Codes only, out of a flattened `visitCounts`.
    fn visit_codes(flat: &[u32]) -> Vec<u16> {
        flat.chunks_exact(2)
            .map(|pair| u16::try_from(pair[0]).expect("visit code fits a u16"))
            .collect()
    }

    /// `improvedPolicy()` is the record writer's `policyTarget`, not a lookalike.
    ///
    /// The accessor's whole claim is that it hands back the value production
    /// self-play records, so the right-hand side here is not a second
    /// derivation of the improved policy: it is a real `SelfPlayBatch` game,
    /// pumped through `collect`/`submit` the way the shard driver pumps it,
    /// whose first ply's `policyTarget` is read straight out of the record
    /// buffer. An accessor that returned anything the recorder does not write
    /// fails here.
    ///
    /// The two searches are *made* the same search rather than assumed to be:
    ///
    /// * same root — game 0's ply 0 is the initial position, which is where
    ///   `NormalDuelSearch` is constructed;
    /// * same params — `simulations`, `maxConsidered` and `cPuct` are handed to
    ///   both from one struct, so a changed default cannot leave them two
    ///   plausible-looking different searches;
    /// * same stream — self-play runs game `i` off `Lcg32::new(seed_base + i)`
    ///   and has drawn nothing from it before the first search, which is
    ///   exactly the state `seed` builds at the boundary;
    /// * same evaluations — the shared mock is a pure function of the feature
    ///   vector, and `parity_leaf` narrows the value identically for both.
    ///
    /// `ply_cap: 1` stops the game after that ply, and not for speed: from ply
    /// 1 on, self-play folds the finished search's RNG back into the game
    /// stream, so no `seed` reaches those roots and ply 0 is the only one this
    /// entry point can reproduce.
    ///
    /// WHAT THIS CANNOT CATCH is reordering. The comparison scatters the pairs
    /// into a dense vector by code, so it is order-blind by construction, and a
    /// mutant that reversed them passes here. The ascending contract is carried
    /// by `improved_policy_spans_the_legal_set_and_visit_counts_the_considered_set`
    /// and by the JS boundary test, not by this one.
    #[test]
    fn improved_policy_is_exactly_what_self_play_records_as_the_policy_target() {
        let options = SelfPlayOptions {
            games: 1,
            simulations: PARITY_SIMS,
            max_considered: PARITY_CONSIDERED,
            ply_cap: 1,
            seed_base: PARITY_SEED,
            ..SelfPlayOptions::default()
        };
        let core_config = config_from_value(&config()).expect("config");
        let recorded = recorded_policy_target(&core_config, options.clone());

        let (config_json, state_json, mut search) =
            parity_search(PARITY_CONSIDERED, options.c_puct);
        let root_features = drive_search(&mut search);
        let improved = improved_pairs(&search.improved_policy().expect("a finished search"));

        // The recorder's own narrowing: scatter the pairs into a zeroed
        // `RECORD_POLICY` vector as `f32`. Compared bit for bit rather than
        // within a tolerance — these are the same `f64`s through the same cast,
        // so anything short of equality means they are not the same numbers.
        let mut scattered = vec![0.0_f32; RECORD_POLICY];
        for (code, probability) in &improved {
            scattered[usize::from(*code)] = *probability as f32;
        }
        for (code, (accessor, record)) in scattered.iter().zip(&recorded).enumerate() {
            assert_eq!(
                accessor.to_bits(),
                record.to_bits(),
                "policyTarget[{code}]: accessor {accessor} vs recorded {record}"
            );
        }

        // What follows keeps that equality from passing for an uninteresting
        // reason. A target that were empty, one-hot, the visit distribution or
        // the network's own policy would match a recorder producing the same
        // wrong thing; these say it is none of those.
        let legal = legal_codes(&config_json, &state_json);
        assert_eq!(legal.len(), PARITY_LEGAL);
        assert_eq!(
            recorded.iter().filter(|mass| **mass > 0.0).count(),
            legal.len(),
            "every legal action carries mass; the prior floor is what puts it there"
        );

        // The substitution this accessor exists to prevent, run as an
        // assertion: normalised visit counts are a different distribution at
        // this root, so an accessor returning them could not have passed above.
        let visits = search.visit_counts();
        let total: f64 = visits.chunks_exact(2).map(|pair| f64::from(pair[1])).sum();
        assert!(total > 0.0, "the search spent its budget");
        let mut visit_normalised = vec![0.0_f32; RECORD_POLICY];
        for pair in visits.chunks_exact(2) {
            visit_normalised[pair[0] as usize] = (f64::from(pair[1]) / total) as f32;
        }
        assert_ne!(
            visit_normalised, recorded,
            "the visit distribution must differ from the improved policy here, or this \
             comparison could not tell the two apart"
        );

        // Nor is it merely the renormalised prior, which is what the improved
        // policy collapses to when the search has separated nothing. So the
        // recorded numbers carry search information, and a boundary that handed
        // back the network's own policy would fail too.
        let mut prior = vec![0.0_f32; RECORD_POLICY];
        mock_evaluator::evaluate(&root_features, &mut prior);
        let prior_mass: f64 = legal
            .iter()
            .map(|code| f64::from(prior[usize::from(*code)]))
            .sum();
        let moved = improved
            .iter()
            .map(|(code, probability)| {
                (probability - f64::from(prior[usize::from(*code)]) / prior_mass).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-3,
            "pi' is within {moved} of the renormalised prior"
        );
    }

    /// The accessor's shape contract, and where it parts company with
    /// `visitCounts`.
    ///
    /// `improvedPolicy` spans every LEGAL root action; `visitCounts` spans the
    /// CONSIDERED ones, which under Gumbel is `min(maxConsidered, legal)` of
    /// them. At the production shape those are 131 and 8, so the code lists
    /// agree only in the sense that one contains the other — a consumer that
    /// treated them as interchangeable supports would be reading a 131-action
    /// target off an 8-action schedule. They coincide exactly when the
    /// considered set is the whole legal set, which is the second half here.
    #[test]
    fn improved_policy_spans_the_legal_set_and_visit_counts_the_considered_set() {
        let c_puct = SelfPlayOptions::default().c_puct;
        let (config_json, state_json, mut narrow) = parity_search(PARITY_CONSIDERED, c_puct);
        drive_search(&mut narrow);
        let legal = legal_codes(&config_json, &state_json);
        assert_eq!(legal.len(), PARITY_LEGAL);

        let improved = improved_pairs(&narrow.improved_policy().expect("a finished search"));
        let codes: Vec<u16> = improved.iter().map(|(code, _)| *code).collect();
        assert!(
            codes.windows(2).all(|pair| pair[0] < pair[1]),
            "codes ascend, strictly: no repeats and no reordering"
        );
        assert_eq!(codes, legal, "support is exactly the legal root actions");
        let total: f64 = improved.iter().map(|(_, mass)| *mass).sum();
        assert!((total - 1.0).abs() < 1e-12, "pi' sums to {total}");
        assert!(
            improved.iter().all(|(_, mass)| *mass > 0.0 && *mass < 1.0),
            "every legal action gets positive mass, and none takes all of it"
        );

        let considered = visit_codes(&narrow.visit_counts());
        assert!(
            considered.windows(2).all(|pair| pair[0] < pair[1]),
            "visitCounts ascends by code too"
        );
        assert_eq!(considered.len(), PARITY_CONSIDERED as usize);
        assert!(
            considered.iter().all(|code| codes.contains(code)),
            "the considered set is drawn from the legal set"
        );
        assert_ne!(
            considered, codes,
            "at maxConsidered {PARITY_CONSIDERED} over {PARITY_LEGAL} legal actions it is a \
             STRICT subset"
        );

        // Considered set at least as large as the legal set: now, and only now,
        // the two code lists are the same list.
        let wide = u32::try_from(legal.len()).expect("legal count fits a u32");
        let (_, _, mut all_considered) = parity_search(wide, c_puct);
        drive_search(&mut all_considered);
        let wide_codes: Vec<u16> =
            improved_pairs(&all_considered.improved_policy().expect("a finished search"))
                .iter()
                .map(|(code, _)| *code)
                .collect();
        assert_eq!(wide_codes, legal);
        assert_eq!(visit_codes(&all_considered.visit_counts()), legal);
    }

    /// The accessor refuses a search that has not finished.
    ///
    /// A doc sentence would not have been enough, and the middle of this test
    /// is the evidence: reaching past the guard at 10 of 48 simulations yields
    /// something nothing structural can fault — 131 codes, strictly ascending,
    /// every mass positive, the total 1.0 to the last bit — that is simply a
    /// different distribution from the one the same search finishes with. There
    /// is no shape check, no sum check and no support check that separates the
    /// two. The only thing that can is the tree's own doneness, which is why it
    /// is asked here rather than assumed.
    ///
    /// Asserted against [`improved_policy_impl`] rather than the boundary
    /// method: the refusal crosses as a `JsValue`, and constructing one off the
    /// wasm target aborts the process instead of returning an error a test
    /// could read. The happy path is taken through the boundary method, which
    /// is the same code either way.
    #[test]
    fn improved_policy_refuses_a_search_that_is_not_finished() {
        let c_puct = SelfPlayOptions::default().c_puct;
        let (_, _, mut search) = parity_search(PARITY_CONSIDERED, c_puct);
        assert_eq!(
            improved_policy_impl(&search.inner),
            Err("search_not_done".to_owned()),
            "a search that has not started has no target to hand out"
        );

        let mut policy = vec![0.0_f32; RECORD_POLICY];
        let mut submit_one = |search: &mut NormalDuelSearch| -> bool {
            if !search.next_leaf().expect("next_leaf") {
                return false;
            }
            search.pending_leaf_mask().expect("pending_leaf_mask");
            let value = parity_leaf(&search.features, &mut policy);
            search.policy.copy_from_slice(&policy);
            search.submit(value).expect("submit");
            true
        };

        for leaf in 0..10 {
            assert!(
                submit_one(&mut search),
                "leaf {leaf} of a 48-simulation search"
            );
            assert_eq!(
                improved_policy_impl(&search.inner),
                Err("search_not_done".to_owned()),
                "still refused after {} leaves",
                leaf + 1
            );
        }
        assert!(!search.inner.is_done());

        // What the guard is standing in front of, read directly off the core.
        let partial = search.inner.result().improved_policy;
        assert_eq!(partial.len(), PARITY_LEGAL, "full support already");
        assert!(partial.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(partial.iter().all(|(_, mass)| *mass > 0.0));
        let partial_total: f64 = partial.iter().map(|(_, mass)| *mass).sum();
        assert!(
            (partial_total - 1.0).abs() < 1e-12,
            "a partial read even sums to 1: {partial_total}"
        );

        while submit_one(&mut search) {}
        let finished = improved_pairs(&search.improved_policy().expect("a finished search"));
        assert_eq!(finished.len(), PARITY_LEGAL);

        // ... and yet it is a different answer, and not subtly: at this seed
        // the largest per-action difference measures 0.99999998, i.e. virtually
        // the whole of one action's probability moves between the partial read
        // and the finished one. The threshold below sits two orders under that,
        // so what is asserted is the premise and not a knife edge.
        let drift = finished
            .iter()
            .zip(&partial)
            .map(|((done_code, done), (partial_code, part))| {
                assert_eq!(done_code, partial_code);
                (done - part).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            drift > 0.01,
            "the partial and finished targets differ by only {drift}, which would make \
             this test's premise false rather than its guard unnecessary"
        );
    }
}
