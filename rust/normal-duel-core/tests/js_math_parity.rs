//! `crate::js_math` against a committed V8 reference, on bit patterns.
//!
//! The PUCT parity test would catch a bad `js_log` only if the error happened
//! to flip a ranking, which is exactly the kind of "passes until it doesn't"
//! coverage worth replacing with a direct check. `Math.log` is the only
//! non-IEEE operation in the search; this pins it over search-representative
//! inputs.
//!
//! The reference is a checked-in fixture rather than a live `node` invocation,
//! because V8's `Math.log` is *not* bit-portable: it is C++ (`base/ieee754.cc`)
//! compiled by the host toolchain, and whether `a*b+c` contracts to an FMA
//! depends on the target. Comparing against the local engine therefore compared
//! against a different oracle on every machine — green on arm64, 4064 1-ULP
//! mismatches on x86-64. `js_log` itself is bit-portable (explicit
//! `f64::mul_add`), so the fixture pins it exactly, everywhere. See the module
//! comment of `src/js_math.rs`.
//!
//! Regenerate the fixture with `node scripts/dump-normal-duel-js-math.mjs`, on
//! arm64 — that is what production runs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use wrongway_normal_duel::js_math::{js_log, Lcg32};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate sits two directories below the repository root")
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn parse_bits(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("bad hex field {hex:?}")))
}

/// `(x, Math.log(x))` pairs from the committed fixture, as bit patterns.
fn log_reference() -> Vec<(f64, f64)> {
    let path = fixture_path("js_log_reference.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("fixture {} is readable: {error}", path.display()));
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split(' ');
            let x = parse_bits(fields.next().expect("every row has an x column"));
            let expected = parse_bits(fields.next().expect("every row has a log column"));
            (x, expected)
        })
        .collect()
}

fn lcg32_reference() -> Vec<u32> {
    let path = fixture_path("lcg32_stream.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("fixture {} is readable: {error}", path.display()));
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.trim().parse().expect("every row is a u32"))
        .collect()
}

/// The exact guarantee: `js_log` reproduces the committed V8 output bit for bit.
///
/// Deterministic on every target — no engine, no toolchain, no host `node`.
#[test]
fn js_log_is_bit_identical_to_the_reference_fixture() {
    let samples = log_reference();
    assert!(
        samples.len() > 4_000,
        "the fixture holds only {} samples; was it truncated?",
        samples.len()
    );
    // The input that first diverged between arm64 and x86-64 CI. If a future
    // regeneration drops it, this test stops covering the case it exists for.
    let divergent = 0.681_908_422_033_302_5_f64;
    assert!(
        samples
            .iter()
            .any(|(x, _)| x.to_bits() == divergent.to_bits()),
        "the fixture must retain the known architecture-divergent input {divergent:?}"
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
        "{mismatches}/{} log values disagree with the fixture; first at {:?} \
         (x, fixture, js_log)",
        samples.len(),
        first
    );
    println!(
        "js_log: {} samples, all bit-identical to the fixture",
        samples.len()
    );
}

#[test]
fn lcg32_is_identical_to_the_reference_stream() {
    let stream = lcg32_reference();
    assert!(
        stream.len() > 4_000,
        "the fixture holds only {} draws; was it truncated?",
        stream.len()
    );
    let mut rng = Lcg32::new(1234);
    for (index, expected) in stream.iter().enumerate() {
        assert_eq!(rng.next_u32(), *expected, "lcg32 draw {index}");
    }
    println!("lcg32: {} draws identical to js/lcg32.mjs", stream.len());
}

/// Diagnostic, not an assertion: how far the *host's* live `Math.log` is from
/// the fixture.
///
/// This never fails. A divergence here is a property of the local V8 build —
/// arm64 clang contracts `a*b+c` in `base/ieee754.cc` to an FMA and baseline
/// x86-64 cannot — not a defect in `js_log`, which is pinned exactly by
/// `js_log_is_bit_identical_to_the_reference_fixture`. Making it fail would put
/// the host toolchain in the test's trust boundary, which is what made the old
/// test unportable. Making it silent would let the architecture dependence go
/// latent, which is worse: it is real, and it is the reason a JS/Rust mixed
/// deployment must not straddle architectures.
#[test]
fn host_math_log_divergence_from_the_fixture_is_reported() {
    let root = repository_root();
    let script = root.join("scripts/check-normal-duel-js-math-host.mjs");
    let run = Command::new("node")
        .current_dir(&root)
        .arg(&script)
        .arg(fixture_path("js_log_reference.txt"))
        .output();

    let Ok(run) = run else {
        report("node is not on PATH; skipped the host Math.log comparison. The exact js_log guarantee is unaffected: it is pinned by the committed fixture.");
        return;
    };
    if !run.status.success() {
        report(&format!(
            "the host Math.log comparison could not run ({}):\n{}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        ));
        return;
    }

    let stdout = String::from_utf8_lossy(&run.stdout);
    let field = |name: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .unwrap_or("?")
            .to_owned()
    };
    let mismatches: usize = field("mismatches").parse().unwrap_or(usize::MAX);
    let total = field("total");

    if mismatches == 0 {
        println!(
            "host Math.log ({} on {}) agrees with the fixture on all {total} samples",
            field("host-node"),
            field("host-arch")
        );
        return;
    }

    report(&format!(
        "host V8 Math.log disagrees with the committed fixture on {mismatches}/{total} samples.\n\
         \x20 host:    node {} on {}-{}\n\
         \x20 fixture: node {} on {}\n\
         \x20 first:   x = {}, fixture 0x{}, host 0x{}\n\
         This is EXPECTED and is not a failure. V8's Math.log is C++ compiled by the host\n\
         toolchain: arm64 clang contracts a*b+c in base/ieee754.cc into an FMA, baseline\n\
         x86-64 has no FMA instruction and does not, so the two differ by 1 ULP. Production\n\
         self-play runs on arm64, which is what the fixture records, and Rust js_log\n\
         reproduces those bits on every target via explicit f64::mul_add.\n\
         The consequence to keep in mind: a JS/Rust mixed deployment on x86_64 can pick a\n\
         different move where the two disagree by 1 ULP in a logit, because that can reorder\n\
         a Gumbel draw. Do not mix engines across architectures in one search.",
        field("host-node"),
        field("host-arch"),
        field("host-platform"),
        field("fixture-node"),
        field("fixture-arch"),
        field("first-x"),
        field("first-fixture"),
        field("first-host"),
    ));
}

/// Write straight to fd 2 rather than through `eprintln!`.
///
/// libtest captures the print macros and only replays them for *failing* tests,
/// so a diagnostic that (correctly) passes would be invisible in CI. A direct
/// `Stderr` write bypasses that capture and lands in the log where it is useful.
fn report(message: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "\n[js_math parity diagnostic] {message}\n");
}
