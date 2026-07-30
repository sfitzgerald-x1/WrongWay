use std::time::Duration;

use wrongway_normal_duel::{
    create_initial_state, legal_action_codes, search_for, search_nodes, Config, Coord,
    NormalDuelError, Player, Players, SearchOptions, SearchReport, JUMP_RULE, RULESET,
};

fn standard_config() -> Config {
    Config {
        ruleset: RULESET.into(),
        rows: 7,
        columns: 7,
        start: Players {
            a: Coord { r: 6, c: 3 },
            b: Coord { r: 0, c: 3 },
        },
        goal_rows: Players { a: 0, b: 6 },
        initial_stock: Players { a: 8, b: 8 },
        jump_rule: JUMP_RULE.into(),
        repetition_threshold: 3,
        ply_cap: 200,
        first_player: Player::A,
    }
}

fn assert_legal_root_action(report: &SearchReport, legal: &[usize]) {
    let action = report
        .action_code
        .expect("an ongoing root must return a deterministic legal fallback");
    assert!(legal.contains(&action));
    assert_eq!(report.principal_variation.first(), Some(&action));
    assert_eq!(report.diagnostics.root_action_count, legal.len());
}

#[test]
fn public_fixed_node_search_is_deterministic_and_legal() {
    let config = standard_config();
    let state = create_initial_state(&config).unwrap();
    let legal = legal_action_codes(&config, &state).unwrap();
    let options = SearchOptions {
        max_depth: 6,
        transposition_capacity: 1 << 14,
        aspiration_window: 64,
    };

    let first: SearchReport = search_nodes(&config, &state, 5_000, options).unwrap();
    let second = search_nodes(&config, &state, 5_000, options).unwrap();

    assert_eq!(first, second);
    assert_legal_root_action(&first, &legal);
}

#[test]
fn public_deadline_search_returns_a_legal_committed_result() {
    let config = standard_config();
    let state = create_initial_state(&config).unwrap();
    let legal = legal_action_codes(&config, &state).unwrap();

    let report = search_for(
        &config,
        &state,
        Duration::from_millis(1),
        SearchOptions::default(),
    )
    .unwrap();

    assert_legal_root_action(&report, &legal);
}

#[test]
fn public_search_rejects_invalid_budgets_and_options() {
    let config = standard_config();
    let state = create_initial_state(&config).unwrap();

    assert_eq!(
        search_nodes(&config, &state, 0, SearchOptions::default()),
        Err(NormalDuelError::InvalidSearchBudget)
    );
    assert_eq!(
        search_for(&config, &state, Duration::ZERO, SearchOptions::default()),
        Err(NormalDuelError::InvalidSearchBudget)
    );
    assert_eq!(
        search_nodes(
            &config,
            &state,
            1,
            SearchOptions {
                max_depth: 0,
                ..SearchOptions::default()
            }
        ),
        Err(NormalDuelError::InvalidSearchOptions)
    );
}
