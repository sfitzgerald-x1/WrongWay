//! `crate::js_math` against real V8, on bit patterns.
//!
//! The PUCT parity test would catch a bad `js_log` only if the error happened
//! to flip a ranking, which is exactly the kind of "passes until it doesn't"
//! coverage worth replacing with a direct check. `Math.log` is the only
//! non-IEEE operation in the search; this pins it over ~380,000 inputs drawn
//! from the ranges the search really uses.

use std::path::{Path, PathBuf};
use std::process::Command;

use wrongway_normal_duel::js_math::{js_log, Lcg32};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate sits two directories below the repository root")
}

fn javascript_reference() -> (Vec<(f64, f64)>, Vec<u32>) {
    let root = repository_root();
    let directory =
        std::env::temp_dir().join(format!("normal-duel-js-math-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("temp directory is creatable");
    let logs_path = directory.join("logs.bin");
    let stream_path = directory.join("stream.bin");

    let run = Command::new("node")
        .current_dir(&root)
        .arg("scripts/dump-normal-duel-js-math.mjs")
        .arg(&logs_path)
        .arg(&stream_path)
        .output()
        .expect("node must be on PATH to run the cross-engine parity harness");
    assert!(
        run.status.success(),
        "JS reference harness failed ({}):\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );

    let logs = std::fs::read(&logs_path).expect("harness wrote the logs file");
    let logs = logs
        .chunks_exact(16)
        .map(|chunk| {
            (
                f64::from_le_bytes(chunk[..8].try_into().expect("8 bytes")),
                f64::from_le_bytes(chunk[8..].try_into().expect("8 bytes")),
            )
        })
        .collect();

    let stream = std::fs::read(&stream_path).expect("harness wrote the stream file");
    let stream = stream
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect();

    (logs, stream)
}

#[test]
fn js_log_is_bit_identical_to_v8_math_log() {
    let (samples, stream) = javascript_reference();
    assert!(
        samples.len() > 300_000,
        "the harness produced only {} samples",
        samples.len()
    );

    let mut mismatches = 0_usize;
    let mut first: Option<(f64, f64, f64)> = None;
    for (x, expected) in &samples {
        let actual = js_log(*x);
        if actual.to_bits() != expected.to_bits() {
            mismatches += 1;
            first.get_or_insert((*x, *expected, actual));
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{} log values disagree with V8; first at x = {:?}",
        samples.len(),
        first
    );
    println!("js_log: {} samples, all bit-identical to Math.log", samples.len());

    let mut rng = Lcg32::new(1234);
    for (index, expected) in stream.iter().enumerate() {
        assert_eq!(rng.next_u32(), *expected, "lcg32 draw {index}");
    }
    println!("lcg32: {} draws identical to js/lcg32.mjs", stream.len());
}
