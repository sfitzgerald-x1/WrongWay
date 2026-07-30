//! Contract parity tests for the frozen JavaScript normal-duel-v1 artifacts.
//!
//! These tests deliberately decode the repository fixtures instead of keeping
//! a second set of hand-written Rust expectations.  A fixture change is thus
//! a visible cross-engine contract change, not an accidental Rust oracle
//! update.

use std::collections::{BTreeSet, HashMap};

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Map, Value};
use wrongway_normal_duel::{
    adjudicate, apply_action, create_initial_state, decode_action, encode_action,
    fast_perft_with_options, fast_throughput_probe, fast_trace, has_path, is_legal_wall,
    legal_action_codes, legal_pawn_destinations, legal_position_actions, normalize_position,
    perft_report_with_options, perft_with_options, position_key, seeded_wall_state,
    state_from_action_codes, validate_config_json, validate_state, Action, Config, Coord,
    GameState, NormalDuelError, Outcome, PerftOptions, Player, Position, PreparedGameState,
};

const CASES: &str = include_str!("../../../tests/fixtures/normal-duel-v1-cases.json");
const TRAJECTORIES: &str =
    include_str!("../../../tests/fixtures/normal-duel-v1-trajectories.jsonl");
const PERFT: &str = include_str!("../../../tests/fixtures/normal-duel-perft-v1.json");

fn json(text: &str, label: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|error| panic!("{label} must be valid JSON: {error}"))
}

fn object<'a>(value: &'a Value, label: &str) -> &'a Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be a JSON object"))
}

fn field<'a>(object: &'a Map<String, Value>, name: &str, label: &str) -> &'a Value {
    object
        .get(name)
        .unwrap_or_else(|| panic!("{label} must include `{name}`"))
}

fn decode<T: DeserializeOwned>(value: &Value, label: &str) -> T {
    serde_json::from_value(value.clone())
        .unwrap_or_else(|error| panic!("{label} must deserialize exactly: {error}"))
}

fn assert_exact_keys(value: &Value, expected: &[&str], label: &str) {
    let actual = object(value, label)
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "{label} keys");
}

fn assert_json<T: Serialize>(actual: &T, expected: &Value, label: &str) {
    assert_eq!(
        serde_json::to_value(actual).expect("test values serialize"),
        *expected,
        "{label}"
    );
}

fn required_string<'a>(object: &'a Map<String, Value>, name: &str, label: &str) -> &'a str {
    field(object, name, label)
        .as_str()
        .unwrap_or_else(|| panic!("{label}.{name} must be a string"))
}

fn required_u64(object: &Map<String, Value>, name: &str, label: &str) -> u64 {
    field(object, name, label)
        .as_u64()
        .unwrap_or_else(|| panic!("{label}.{name} must be an unsigned integer"))
}

fn fixture_config(fixture: &Value, id: &str) -> Config {
    let root = object(fixture, "fixture");
    let configs = object(field(root, "configs", "fixture"), "fixture.configs");
    let config: Config = decode(
        configs
            .get(id)
            .unwrap_or_else(|| panic!("fixture config `{id}` must exist")),
        id,
    );
    config.validate().unwrap_or_else(|error| {
        panic!("fixture config `{id}` must validate: {error}");
    });
    config
}

fn case_config(fixture: &Value, case: &Map<String, Value>, label: &str) -> Config {
    fixture_config(fixture, required_string(case, "configId", label))
}

#[test]
fn normal_duel_v1_case_fixture_is_a_typed_rust_contract() {
    let fixture = json(CASES, "normal-duel-v1 cases fixture");
    assert_exact_keys(
        &fixture,
        &["cases", "configs", "fixtureFormat", "ruleset"],
        "fixture",
    );
    let root = object(&fixture, "fixture");
    assert_eq!(
        required_string(root, "ruleset", "fixture"),
        "normal-duel-v1"
    );
    let cases = field(root, "cases", "fixture")
        .as_array()
        .expect("fixture.cases must be an array");
    assert_eq!(cases.len(), 22, "all frozen v1 cases are exercised");

    let mut covered = BTreeSet::new();
    for raw_case in cases {
        let case = object(raw_case, "fixture case");
        let id = required_string(case, "id", "fixture case");
        let kind = required_string(case, "kind", id);
        covered.insert(id);
        match kind {
            "query" => {
                let config = case_config(&fixture, case, id);
                let position: Position = decode(field(case, "position", id), id);
                assert_json(
                    &normalize_position(&config, &position)
                        .unwrap_or_else(|error| panic!("{id} position must normalize: {error}")),
                    field(case, "position", id),
                    &format!("{id} canonical position"),
                );
                let expect = object(field(case, "expect", id), id);
                if let Some(destinations) = expect.get("legalPawnDestinations") {
                    let expected: Vec<Coord> =
                        decode(destinations, &format!("{id} legalPawnDestinations"));
                    assert_eq!(
                        legal_pawn_destinations(&config, &position).unwrap(),
                        expected,
                        "{id} permissive pawn exits"
                    );
                }
                if let Some(count) = expect.get("legalActionCount") {
                    assert_eq!(
                        legal_position_actions(&config, &position).unwrap().len(),
                        count.as_u64().expect("legalActionCount integer") as usize,
                        "{id} legal action count"
                    );
                }
                if let Some(raw_action) = case.get("action") {
                    let action: Action = decode(raw_action, &format!("{id} action"));
                    if let Some(code) = expect.get("actionCode") {
                        assert_eq!(
                            encode_action(&config, &action).unwrap(),
                            code.as_u64().expect("actionCode integer") as usize,
                            "{id} action code"
                        );
                    }
                    if let Some(legal) = expect.get("legal") {
                        let Action::Wall { wall } = &action else {
                            panic!("{id} fixture's legal flag currently applies to walls");
                        };
                        assert_eq!(
                            is_legal_wall(&config, &position, wall).unwrap(),
                            legal.as_bool().expect("legal boolean"),
                            "{id} wall legality"
                        );
                    }
                    if let Some(paths_remain) = expect.get("pathsRemain") {
                        let Action::Wall { wall } = action else {
                            panic!("{id} fixture's path fact currently applies to walls");
                        };
                        let mut walls = position.walls.clone();
                        walls.push(wall);
                        assert_eq!(
                            has_path(&config, position.pawns.a, config.goal_rows.a, &walls)
                                .unwrap()
                                && has_path(&config, position.pawns.b, config.goal_rows.b, &walls)
                                    .unwrap(),
                            paths_remain.as_bool().expect("pathsRemain boolean"),
                            "{id} candidate path fact"
                        );
                    }
                }
            }
            "transition" => {
                let config = case_config(&fixture, case, id);
                let state: GameState = decode(field(case, "state", id), &format!("{id} state"));
                let action: Action = decode(field(case, "action", id), &format!("{id} action"));
                let expected: GameState =
                    decode(field(case, "nextState", id), &format!("{id} nextState"));
                assert_eq!(
                    validate_state(&config, &state).unwrap(),
                    state,
                    "{id} input state"
                );
                assert_eq!(
                    encode_action(&config, &action).unwrap(),
                    required_u64(object(field(case, "expect", id), id), "actionCode", id) as usize,
                    "{id} action code"
                );
                assert_eq!(
                    apply_action(&config, &state, &action).unwrap(),
                    expected,
                    "{id} exact transition"
                );
            }
            "adjudication" => {
                let config = case_config(&fixture, case, id);
                let facts = object(field(case, "facts", id), id);
                let goal_winner = match field(facts, "goalWinner", id) {
                    Value::Null => None,
                    value => Some(decode(value, &format!("{id} goalWinner"))),
                };
                let expected: Outcome = decode(
                    field(object(field(case, "expect", id), id), "outcome", id),
                    &format!("{id} outcome"),
                );
                assert_eq!(
                    adjudicate(
                        &config,
                        goal_winner,
                        required_u64(facts, "resultingPositionCount", id),
                        required_u64(facts, "resultingPly", id),
                    )
                    .unwrap(),
                    expected,
                    "{id} adjudication priority"
                );
            }
            "rejection" => match id {
                "terminal-action-rejected" => {
                    let config = fixture_config(&fixture, "standard-a");
                    let mut terminal = create_initial_state(&config).unwrap();
                    terminal.outcome = decode(
                        field(object(field(case, "input", id), id), "outcome", id),
                        "terminal fixture outcome",
                    );
                    let action: Action = decode(
                        field(object(field(case, "input", id), id), "action", id),
                        "terminal fixture action",
                    );
                    assert_eq!(
                        apply_action(&config, &terminal, &action),
                        Err(NormalDuelError::TerminalState),
                        "terminal actions are rejected before a state transition"
                    );
                }
                "hammer-feature-rejected" => {
                    let config_input = field(case, "input", id);
                    assert_eq!(
                        validate_config_json(config_input),
                        Err(NormalDuelError::UnsupportedFeature),
                        "normal-duel boundary must not reinterpret Hammer features"
                    );
                }
                other => panic!("unhandled rejection fixture `{other}`"),
            },
            other => panic!("unhandled fixture kind `{other}` for {id}"),
        }
    }
    assert_eq!(covered.len(), 22, "fixture case ids are unique");
    assert!(
        covered.contains("initial-9x9-b"),
        "first-player B query is covered"
    );
    assert!(
        covered.contains("permissive-jump-open"),
        "permissive exits are covered"
    );
    assert!(
        covered.contains("terminal-action-rejected"),
        "terminal rejection is covered"
    );
}

#[test]
fn frozen_trajectory_jsonl_replays_exactly_in_rust() {
    assert!(TRAJECTORIES.ends_with('\n'), "JSONL has one final LF");
    assert!(!TRAJECTORIES.contains('\r'), "JSONL is LF-only");
    let lines = TRAJECTORIES
        .strip_suffix('\n')
        .unwrap()
        .split('\n')
        .collect::<Vec<_>>();
    assert_eq!(
        lines.len(),
        33,
        "all frozen trajectory records are replayed"
    );

    let mut previous = HashMap::<String, GameState>::new();
    let mut trajectories = BTreeSet::new();
    let mut first_players = BTreeSet::new();
    for (line_index, line) in lines.iter().enumerate() {
        assert!(
            !line.is_empty(),
            "JSONL line {} is non-empty",
            line_index + 1
        );
        let record = json(line, "trajectory record");
        assert_exact_keys(
            &record,
            &[
                "configuration",
                "corpusFormat",
                "endReason",
                "generatorVersion",
                "legalActionCodes",
                "nextState",
                "outcome",
                "prng",
                "seed",
                "selectedAction",
                "selectedActionCode",
                "selectionMode",
                "state",
                "step",
                "trajectoryId",
            ],
            &format!("trajectory record {}", line_index + 1),
        );
        let raw = object(&record, "trajectory record");
        let id = required_string(raw, "trajectoryId", "trajectory record");
        let step = required_u64(raw, "step", id) as usize;
        let config: Config = decode(
            field(raw, "configuration", id),
            &format!("{id} configuration"),
        );
        config.validate().unwrap();
        first_players.insert(config.first_player.as_str());
        let state: GameState = decode(field(raw, "state", id), &format!("{id} state"));
        let next_state: GameState = decode(field(raw, "nextState", id), &format!("{id} nextState"));
        let action: Action = decode(field(raw, "selectedAction", id), &format!("{id} action"));
        let selected_code = required_u64(raw, "selectedActionCode", id) as usize;
        let legal_codes: Vec<usize> = decode(
            field(raw, "legalActionCodes", id),
            &format!("{id} legal codes"),
        );

        assert_eq!(
            validate_state(&config, &state).unwrap(),
            state,
            "{id}/{step} state"
        );
        assert_eq!(
            legal_action_codes(&config, &state).unwrap(),
            legal_codes,
            "{id}/{step} canonical legal action codes"
        );
        assert_eq!(
            decode_action(&config, selected_code).unwrap(),
            action,
            "{id}/{step} decode"
        );
        assert_eq!(
            encode_action(&config, &action).unwrap(),
            selected_code,
            "{id}/{step} encode"
        );
        assert!(
            legal_codes.contains(&selected_code),
            "{id}/{step} action is legal"
        );
        assert_eq!(
            position_key(&config, &state.position).unwrap(),
            state.position_key,
            "{id}/{step} canonical state key"
        );
        let actual_next = apply_action(&config, &state, &action).unwrap();
        assert_eq!(actual_next, next_state, "{id}/{step} exact next state");
        assert_eq!(
            position_key(&config, &next_state.position).unwrap(),
            next_state.position_key,
            "{id}/{step} canonical next-state key"
        );
        assert_eq!(
            serde_json::to_value(actual_next.outcome).unwrap(),
            *field(raw, "outcome", id),
            "{id}/{step} outcome"
        );
        match previous.get(id) {
            Some(prior) => assert_eq!(state, *prior, "{id}/{step} is linked to prior record"),
            None => {
                assert_eq!(step, 0, "{id} starts at step zero");
                assert_eq!(
                    state,
                    create_initial_state(&config).unwrap(),
                    "{id} starts at initial state"
                );
            }
        }
        previous.insert(id.to_owned(), next_state);
        trajectories.insert(id.to_owned());
    }
    assert_eq!(
        trajectories.len(),
        5,
        "all five frozen trajectories are present"
    );
    assert_eq!(first_players, BTreeSet::from(["A", "B"]));
}

#[test]
fn frozen_perft_roots_match_exact_depth_and_divide_vectors() {
    let fixture = json(PERFT, "normal-duel perft fixture");
    assert_exact_keys(
        &fixture,
        &[
            "cases",
            "configs",
            "fixtureFormat",
            "generator",
            "ruleset",
            "semantics",
            "source",
        ],
        "perft fixture",
    );
    let root = object(&fixture, "perft fixture");
    let cases = field(root, "cases", "perft fixture")
        .as_array()
        .expect("perft fixture.cases must be an array");
    assert_eq!(cases.len(), 9, "all frozen perft roots are replayed");

    for raw_case in cases {
        let case = object(raw_case, "perft case");
        let id = required_string(case, "id", "perft case");
        let config = case_config(&fixture, case, id);
        let expected_state: GameState = decode(field(case, "state", id), &format!("{id} state"));
        let replayed = match required_string(case, "kind", id) {
            "initial" => create_initial_state(&config).unwrap(),
            "action-codes" => {
                let codes: Vec<usize> =
                    decode(field(case, "actionCodes", id), &format!("{id} actionCodes"));
                for &code in &codes {
                    assert!(
                        decode_action(&config, code).is_ok(),
                        "{id} action code {code} is structurally decodable"
                    );
                }
                state_from_action_codes(&config, &codes).unwrap()
            }
            "seeded-walls" => {
                let generator = object(field(case, "generator", id), id);
                assert_eq!(required_string(generator, "algorithm", id), "lcg32-v1");
                seeded_wall_state(
                    &config,
                    required_u64(generator, "seed", id) as u32,
                    required_u64(generator, "plies", id),
                )
                .unwrap()
            }
            other => panic!("{id} has unsupported perft provenance `{other}`"),
        };
        assert_eq!(replayed, expected_state, "{id} root state replay");
        assert_eq!(
            validate_state(&config, &expected_state).unwrap(),
            expected_state
        );

        let expect = object(field(case, "expect", id), id);
        let depth = required_u64(expect, "depth", id) as u8;
        let leaves: Vec<u64> = decode(field(expect, "leavesByDepth", id), &format!("{id} leaves"));
        let root_codes: Vec<usize> = decode(
            field(expect, "rootActionCodes", id),
            &format!("{id} root codes"),
        );
        let expected_divide: Vec<(usize, Vec<u64>)> =
            decode(field(expect, "divide", id), &format!("{id} divide"));
        let options = case
            .get("perftOptions")
            .filter(|value| !value.is_null())
            .map(|value| PerftOptions {
                max_nodes: required_u64(object(value, id), "maxNodes", id),
            })
            .unwrap_or_default();
        let report = perft_report_with_options(&config, &expected_state, depth, options).unwrap();
        assert_eq!(report.leaves_by_depth, leaves, "{id} exact-depth leaves");
        assert_eq!(
            report.node_visits,
            required_u64(expect, "nodeVisits", id),
            "{id} JS-compatible scalar-evaluation accounting"
        );
        assert_eq!(
            report
                .divide
                .iter()
                .map(|entry| entry.action_code)
                .collect::<Vec<_>>(),
            root_codes,
            "{id} root action codes"
        );
        assert_eq!(
            report
                .divide
                .iter()
                .map(|entry| (entry.action_code, entry.child_leaves_by_depth.clone()))
                .collect::<Vec<_>>(),
            expected_divide,
            "{id} root divide vectors"
        );
        let prepared = PreparedGameState::from_game_state(&config, &expected_state).unwrap();
        assert_eq!(
            prepared.legal_action_codes(&config).unwrap(),
            root_codes,
            "{id} compact root actions"
        );
        let probe = fast_throughput_probe(&config, &expected_state, depth, options).unwrap();
        assert_eq!(
            probe.root_action_codes, root_codes,
            "{id} compact probe roots"
        );
        assert_eq!(
            probe.perft_leaves,
            *leaves.last().unwrap(),
            "{id} compact probe perft"
        );
        let compact_trace = fast_trace(&config, &expected_state).unwrap();
        assert_eq!(
            compact_trace.root_action_codes, root_codes,
            "{id} compact trace root action codes"
        );
        assert_eq!(
            compact_trace.children.len(),
            root_codes.len(),
            "{id} compact trace child count"
        );
        for (&code, compact_child) in root_codes.iter().zip(&compact_trace.children) {
            let immutable_child = apply_action(
                &config,
                &expected_state,
                &decode_action(&config, code).unwrap(),
            )
            .expect("frozen root code applies");
            assert_eq!(
                compact_child.action_code, code,
                "{id} compact trace action code"
            );
            assert_eq!(
                compact_child.position_key, immutable_child.position_key,
                "{id} compact child position key at action {code}"
            );
            assert_eq!(
                compact_child.legal_action_codes,
                legal_action_codes(&config, &immutable_child).unwrap(),
                "{id} compact child action codes at action {code}"
            );
        }
        let immutable_child_count = root_codes
            .iter()
            .try_fold(0_u64, |total, &code| {
                let child = apply_action(&config, &expected_state, &decode_action(&config, code)?)
                    .expect("frozen root code applies");
                Ok::<_, NormalDuelError>(total + legal_action_codes(&config, &child)?.len() as u64)
            })
            .unwrap();
        assert_eq!(
            probe.child_action_count, immutable_child_count,
            "{id} compact child legal counts"
        );
        for (current_depth, expected) in leaves.iter().copied().enumerate() {
            assert_eq!(
                perft_with_options(&config, &expected_state, current_depth as u8, options,)
                    .unwrap(),
                expected,
                "{id} scalar exact-depth perft at {current_depth}"
            );
            assert_eq!(
                fast_perft_with_options(&config, &expected_state, current_depth as u8, options,)
                    .unwrap(),
                expected,
                "{id} compact exact-depth perft at {current_depth}"
            );
        }
    }
}

#[test]
fn exact_depth_terminal_behavior_and_first_player_b_are_pinned() {
    let fixture = json(PERFT, "normal-duel perft fixture");
    let perft_cases = field(object(&fixture, "perft fixture"), "cases", "perft fixture")
        .as_array()
        .unwrap();
    let terminal = perft_cases
        .iter()
        .find(|entry| {
            object(entry, "perft case").get("id")
                == Some(&Value::String("terminal-goal-win-7x7-a".into()))
        })
        .expect("terminal perft root exists");
    let terminal_case = object(terminal, "terminal perft case");
    let terminal_config = case_config(&fixture, terminal_case, "terminal-goal-win-7x7-a");
    let terminal_state: GameState = decode(
        field(terminal_case, "state", "terminal perft case"),
        "terminal perft state",
    );
    let terminal_report = perft_report_with_options(
        &terminal_config,
        &terminal_state,
        4,
        PerftOptions::default(),
    )
    .unwrap();
    assert_eq!(terminal_report.leaves_by_depth, vec![1, 0, 0, 0, 0]);
    assert!(terminal_report.divide.is_empty());

    let config = fixture_config(&fixture, "standardB");
    let initial = create_initial_state(&config).unwrap();
    assert_eq!(config.first_player, Player::B);
    assert_eq!(initial.position.turn, Player::B);
    assert_eq!(
        legal_action_codes(&config, &initial).unwrap()[0],
        3,
        "first-player B begins with B's north-edge legal pawn ordering"
    );
}
