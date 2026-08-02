//! Cross-engine parity for the allocation-free hot path.
//!
//! The reference answers are produced by shelling out to node against the real
//! `js/` modules (`scripts/dump-normal-duel-nn-parity.mjs`), not by a Rust
//! reimplementation of the JavaScript. A fast function that disagrees with JS
//! is worse than no fast function, so `encode_state_into` is compared as f32
//! bit patterns rather than approximately.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use wrongway_normal_duel::{
    encode_state_into, legal_action_codes_fast, GameState, MAX_POLICY_CODES, NN_INPUT_PLANES,
};

#[path = "common/state_pool.rs"]
mod state_pool;

const STATES: usize = 600;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate sits two directories below the repository root")
}

/// Reference output for `states`: legal action codes and raw encoder floats.
fn javascript_reference(config: &Value, states: &[GameState]) -> (Vec<Vec<usize>>, Vec<f32>) {
    let root = repository_root();
    let directory = std::env::temp_dir().join(format!("normal-duel-parity-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("temp directory is creatable");
    let input = directory.join("states.json");
    let codes_path = directory.join("codes.json");
    let planes_path = directory.join("planes.bin");

    std::fs::write(
        &input,
        serde_json::to_vec(&json!({ "config": config, "states": states }))
            .expect("states serialize"),
    )
    .expect("temp input is writable");

    let output = Command::new("node")
        .current_dir(&root)
        .arg("scripts/dump-normal-duel-nn-parity.mjs")
        .arg(&input)
        .arg(&codes_path)
        .arg(&planes_path)
        .output()
        .expect("node must be on PATH to run the cross-engine parity harness");
    assert!(
        output.status.success(),
        "JS reference harness failed ({}):\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let codes: Value =
        serde_json::from_slice(&std::fs::read(&codes_path).expect("harness wrote the codes file"))
            .expect("codes file is JSON");
    let codes = codes["codes"]
        .as_array()
        .expect("codes is an array")
        .iter()
        .map(|entry| {
            entry
                .as_array()
                .expect("each entry is an array")
                .iter()
                .map(|code| code.as_u64().expect("codes are unsigned") as usize)
                .collect()
        })
        .collect();

    let bytes = std::fs::read(&planes_path).expect("harness wrote the planes file");
    let planes = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks_exact yields 4 bytes")))
        .collect();

    std::fs::remove_dir_all(&directory).ok();
    (codes, planes)
}

#[test]
fn hot_path_matches_the_javascript_reference_bit_for_bit() {
    let config = state_pool::canonical_config();
    let states = state_pool::state_pool(&config, STATES);
    assert_eq!(states.len(), STATES, "the pool must be full");

    let config_json = serde_json::to_value(&config).expect("config serializes");
    let (expected_codes, expected_planes) = javascript_reference(&config_json, &states);
    assert_eq!(expected_codes.len(), STATES);
    let plane_floats = NN_INPUT_PLANES * config.cells();
    assert_eq!(expected_planes.len(), STATES * plane_floats);

    let mut encoded = vec![0.0_f32; plane_floats];
    let mut codes = [0_u16; MAX_POLICY_CODES];
    let mut encoding_mismatches = Vec::new();
    let mut code_mismatches = Vec::new();

    for (index, state) in states.iter().enumerate() {
        // A caller buffer is reused across states, so a stale value left
        // behind by a previous state would surface here.
        encoded.fill(f32::NAN);
        encode_state_into(&config, state, &mut encoded).expect("canonical state encodes");
        let reference = &expected_planes[index * plane_floats..(index + 1) * plane_floats];
        for (slot, (actual, expect)) in encoded.iter().zip(reference).enumerate() {
            if actual.to_bits() != expect.to_bits() {
                encoding_mismatches.push(format!(
                    "state {index} float {slot} (plane {}): rust {:#010x} vs js {:#010x}",
                    slot / config.cells(),
                    actual.to_bits(),
                    expect.to_bits()
                ));
            }
        }

        let count = legal_action_codes_fast(&config, &state.position, &mut codes)
            .expect("canonical position generates");
        let actual: Vec<usize> = codes[..count].iter().map(|code| *code as usize).collect();
        if actual != expected_codes[index] {
            code_mismatches.push(format!(
                "state {index} (ply {}): rust {:?} vs js {:?}",
                state.ply, actual, expected_codes[index]
            ));
        }
    }

    assert!(
        encoding_mismatches.is_empty(),
        "encode_state_into diverged from JS encodeState in {} of {} states:\n{}",
        encoding_mismatches.len(),
        STATES,
        encoding_mismatches
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        code_mismatches.is_empty(),
        "legal_action_codes_fast diverged from JS legalActionCodes in {} of {} states:\n{}",
        code_mismatches.len(),
        STATES,
        code_mismatches
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
