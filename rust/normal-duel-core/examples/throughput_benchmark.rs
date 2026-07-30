//! Native half of the normal-duel JS-versus-Rust throughput benchmark.
//!
//! See `BENCHMARK.md` for the fixed workload and why fixture construction is
//! excluded from the timed region.

use std::collections::BTreeMap;
use std::env;
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;
use wrongway_normal_duel::{
    apply_legal_action, create_initial_state, fast_throughput_probe, fast_trace,
    legal_action_codes, legal_actions, perft_with_options, seeded_wall_state,
    state_from_action_codes, validate_state, Action, Config, GameState, NormalDuelError,
    PerftOptions,
};

const FIXTURE: &str = include_str!("../../../tests/fixtures/normal-duel-perft-v1.json");
const WORKLOAD_VERSION: &str = "normal-duel-throughput-v1";
const DEFAULT_WARMUP: usize = 2;
const DEFAULT_SAMPLES: usize = 9;

type BenchResult<T> = Result<T, String>;

#[derive(Debug, Deserialize)]
struct Fixture {
    configs: BTreeMap<String, Config>,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    id: String,
    kind: String,
    #[serde(rename = "configId")]
    config_id: String,
    #[serde(rename = "actionCodes")]
    action_codes: Option<Vec<usize>>,
    generator: Option<Generator>,
    #[serde(rename = "perftOptions")]
    perft_options: Option<FixturePerftOptions>,
    state: GameState,
    expect: Expectation,
}

#[derive(Debug, Deserialize)]
struct Generator {
    seed: u32,
    plies: u64,
}

#[derive(Debug, Deserialize)]
struct FixturePerftOptions {
    #[serde(rename = "maxNodes")]
    max_nodes: u64,
}

#[derive(Debug, Deserialize)]
struct Expectation {
    depth: u8,
    #[serde(rename = "leavesByDepth")]
    leaves_by_depth: Vec<u64>,
    #[serde(rename = "rootActionCodes")]
    root_action_codes: Vec<usize>,
}

#[derive(Debug, Clone)]
struct BenchCase {
    id: String,
    config: Config,
    state: GameState,
    depth: u8,
    expected_leaves: u64,
    expected_root_codes: Vec<usize>,
    perft_options: PerftOptions,
}

#[derive(Debug)]
struct Plan {
    fixture_sha256: String,
    cases: Vec<BenchCase>,
}

#[derive(Debug)]
struct Integrity {
    action_checksum: String,
    perft_checksum: String,
    root_action_count: u64,
    child_action_count: u64,
    expected_leaf_total: u64,
    compact_trace_verified: bool,
}

/// FNV-1a-64 with the exact length-prefixed UTF-8 framing used by the Node
/// benchmark. It is a portable operation checksum, not a cryptographic hash.
#[derive(Debug)]
struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn add_text(&mut self, text: &str) {
        self.update(text.len().to_string().as_bytes());
        self.update(b":");
        self.update(text.as_bytes());
        self.update(b"|");
    }

    fn add_u64(&mut self, value: u64) {
        self.add_text(&value.to_string());
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    warmup: usize,
    samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Verify,
    Measure,
    Profile,
}

fn parse_count(raw: Option<String>, name: &str, minimum: usize) -> BenchResult<usize> {
    let raw = raw.ok_or_else(|| format!("{name} needs a value"))?;
    raw.parse::<usize>()
        .map_err(|_| format!("{name} must be an integer >= {minimum}"))
        .and_then(|value| {
            if value < minimum {
                Err(format!("{name} must be an integer >= {minimum}"))
            } else {
                Ok(value)
            }
        })
}

fn parse_arguments() -> BenchResult<Arguments> {
    let mut result = Arguments {
        mode: Mode::Measure,
        warmup: DEFAULT_WARMUP,
        samples: DEFAULT_SAMPLES,
    };
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--verify" => result.mode = Mode::Verify,
            "--measure" => result.mode = Mode::Measure,
            "--profile" => result.mode = Mode::Profile,
            "--smoke" => {
                result.mode = Mode::Measure;
                result.warmup = 0;
                result.samples = 1;
            }
            "--warmup" => result.warmup = parse_count(arguments.next(), "warmup", 0)?,
            "--samples" => result.samples = parse_count(arguments.next(), "samples", 1)?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(result)
}

fn normal_error(error: NormalDuelError) -> String {
    error.code().to_owned()
}

fn fixture_sha256_from_manifest() -> BenchResult<String> {
    #[derive(Deserialize)]
    struct Manifest {
        sha256: String,
    }
    const MANIFEST: &str =
        include_str!("../../../tests/fixtures/normal-duel-perft-v1.manifest.json");
    serde_json::from_str::<Manifest>(MANIFEST)
        .map(|manifest| manifest.sha256)
        .map_err(|error| format!("invalid perft manifest: {error}"))
}

fn make_plan() -> BenchResult<Plan> {
    let fixture: Fixture =
        serde_json::from_str(FIXTURE).map_err(|error| format!("invalid perft fixture: {error}"))?;
    if fixture.cases.is_empty() {
        return Err("fixture has no cases".to_owned());
    }
    let mut cases = Vec::with_capacity(fixture.cases.len());
    for entry in fixture.cases {
        let config = fixture
            .configs
            .get(&entry.config_id)
            .ok_or_else(|| format!("{} refers to unknown config {}", entry.id, entry.config_id))?
            .clone();
        config.validate().map_err(normal_error)?;
        let replayed = match entry.kind.as_str() {
            "initial" => create_initial_state(&config).map_err(normal_error)?,
            "action-codes" => state_from_action_codes(
                &config,
                entry
                    .action_codes
                    .as_deref()
                    .ok_or_else(|| format!("{} is missing actionCodes", entry.id))?,
            )
            .map_err(normal_error)?,
            "seeded-walls" => {
                let generator = entry
                    .generator
                    .as_ref()
                    .ok_or_else(|| format!("{} is missing generator", entry.id))?;
                seeded_wall_state(&config, generator.seed, generator.plies).map_err(normal_error)?
            }
            other => return Err(format!("{} has unsupported fixture kind {other}", entry.id)),
        };
        if replayed != entry.state {
            return Err(format!("{} replay differs from its frozen state", entry.id));
        }
        let state = validate_state(&config, &replayed).map_err(normal_error)?;
        let expected_leaves = *entry
            .expect
            .leaves_by_depth
            .last()
            .ok_or_else(|| format!("{} has no perft leaves", entry.id))?;
        let perft_options = entry
            .perft_options
            .map(|options| PerftOptions {
                max_nodes: options.max_nodes,
            })
            .unwrap_or_default();
        cases.push(BenchCase {
            id: entry.id,
            config,
            state,
            depth: entry.expect.depth,
            expected_leaves,
            expected_root_codes: entry.expect.root_action_codes,
            perft_options,
        });
    }
    Ok(Plan {
        fixture_sha256: fixture_sha256_from_manifest()?,
        cases,
    })
}

/// Run the immutable correctness checks that make a cached perft total or a
/// differently-shaped workload detectable before timing starts.
fn verify_plan(plan: &Plan) -> BenchResult<Integrity> {
    let mut actions_digest = Fnv1a64::new();
    let mut compact_actions_digest = Fnv1a64::new();
    let mut perft_digest = Fnv1a64::new();
    actions_digest.add_text(WORKLOAD_VERSION);
    compact_actions_digest.add_text(WORKLOAD_VERSION);
    perft_digest.add_text(WORKLOAD_VERSION);
    let mut root_action_count = 0_u64;
    let mut child_action_count = 0_u64;
    let mut expected_leaf_total = 0_u64;

    for entry in &plan.cases {
        let actions = legal_actions(&entry.config, &entry.state).map_err(normal_error)?;
        let codes = actions
            .iter()
            .map(|action| action.policy_code(&entry.config).map_err(normal_error))
            .collect::<BenchResult<Vec<_>>>()?;
        if codes != entry.expected_root_codes {
            return Err(format!(
                "{} root legal action codes differ from fixture",
                entry.id
            ));
        }
        let compact = fast_trace(&entry.config, &entry.state).map_err(normal_error)?;
        if compact.root_action_codes != codes {
            return Err(format!(
                "{} compact root legal action codes differ from immutable Rust",
                entry.id
            ));
        }

        actions_digest.add_text(&entry.id);
        actions_digest.add_text(&entry.state.position_key);
        actions_digest.add_u64(u64::from(entry.depth));
        actions_digest.add_u64(codes.len() as u64);
        compact_actions_digest.add_text(&entry.id);
        compact_actions_digest.add_text(&entry.state.position_key);
        compact_actions_digest.add_u64(u64::from(entry.depth));
        compact_actions_digest.add_u64(compact.root_action_codes.len() as u64);
        root_action_count = root_action_count
            .checked_add(codes.len() as u64)
            .ok_or_else(|| "root action count overflow".to_owned())?;
        for ((action, code), compact_child) in actions.iter().zip(codes).zip(&compact.children) {
            let child =
                apply_legal_action(&entry.config, &entry.state, action).map_err(normal_error)?;
            let child_codes = legal_action_codes(&entry.config, &child).map_err(normal_error)?;
            if compact_child.action_code != code
                || compact_child.position_key != child.position_key
                || compact_child.legal_action_codes != child_codes
            {
                return Err(format!(
                    "{} compact child trace differs at action code {code}",
                    entry.id
                ));
            }
            actions_digest.add_u64(code as u64);
            actions_digest.add_text(&child.position_key);
            actions_digest.add_u64(child_codes.len() as u64);
            for child_code in &child_codes {
                actions_digest.add_u64(*child_code as u64);
            }
            compact_actions_digest.add_u64(compact_child.action_code as u64);
            compact_actions_digest.add_text(&compact_child.position_key);
            compact_actions_digest.add_u64(compact_child.legal_action_codes.len() as u64);
            for child_code in &compact_child.legal_action_codes {
                compact_actions_digest.add_u64(*child_code as u64);
            }
            child_action_count = child_action_count
                .checked_add(child_codes.len() as u64)
                .ok_or_else(|| "child action count overflow".to_owned())?;
        }

        let leaves = perft_with_options(
            &entry.config,
            &entry.state,
            entry.depth,
            entry.perft_options,
        )
        .map_err(normal_error)?;
        if leaves != entry.expected_leaves {
            return Err(format!("{} perft differs from fixture", entry.id));
        }
        perft_digest.add_text(&entry.id);
        perft_digest.add_u64(u64::from(entry.depth));
        perft_digest.add_u64(leaves);
        expected_leaf_total = expected_leaf_total
            .checked_add(leaves)
            .ok_or_else(|| "perft leaf total overflow".to_owned())?;
    }

    if compact_actions_digest.hex() != actions_digest.hex() {
        return Err("compact trace checksum differs from immutable Rust".to_owned());
    }
    Ok(Integrity {
        action_checksum: actions_digest.hex(),
        perft_checksum: perft_digest.hex(),
        root_action_count,
        child_action_count,
        expected_leaf_total,
        compact_trace_verified: true,
    })
}

/// The timed unit: exactly the legal/apply/child-legal/perft pass hashed by
/// `verify_plan`. State creation, JSON parsing, checksum construction, and
/// JSON output are all outside the interval.
fn run_pass(plan: &Plan) -> BenchResult<u64> {
    let mut result = 0_u64;
    for entry in &plan.cases {
        let probe = fast_throughput_probe(
            &entry.config,
            &entry.state,
            entry.depth,
            entry.perft_options,
        )
        .map_err(normal_error)?;
        result = result
            .checked_add(probe.root_action_codes.len() as u64)
            .ok_or_else(|| "work result overflow".to_owned())?;
        for code in probe.root_action_codes {
            result = result
                .checked_add(code as u64)
                .ok_or_else(|| "work result overflow".to_owned())?;
        }
        result = result
            .checked_add(probe.child_action_count)
            .ok_or_else(|| "work result overflow".to_owned())?;
        result = result
            .checked_add(probe.perft_leaves)
            .ok_or_else(|| "work result overflow".to_owned())?;
    }
    Ok(result)
}

fn measure(plan: &Plan, warmup: usize, samples: usize) -> BenchResult<(u64, Vec<f64>)> {
    for _ in 0..warmup {
        std::hint::black_box(run_pass(plan)?);
    }
    let mut work_result = None;
    let mut sample_milliseconds = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let result = std::hint::black_box(run_pass(plan)?);
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        if let Some(previous) = work_result {
            if previous != result {
                return Err("timed workload is not deterministic".to_owned());
            }
        }
        work_result = Some(result);
        sample_milliseconds.push(elapsed);
    }
    Ok((
        work_result.ok_or_else(|| "no timed samples".to_owned())?,
        sample_milliseconds,
    ))
}

fn median(samples: &[f64]) -> BenchResult<f64> {
    if samples.is_empty() {
        return Err("no profile samples".to_owned());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

fn profile_operation<F>(
    mut work: F,
    warmup: usize,
    samples: usize,
) -> BenchResult<serde_json::Value>
where
    F: FnMut() -> BenchResult<u64>,
{
    for _ in 0..warmup {
        std::hint::black_box(work()?);
    }
    let mut result = None;
    let mut sample_milliseconds = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let value = std::hint::black_box(work()?);
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        if let Some(previous) = result {
            if previous != value {
                return Err("profiled operation is not deterministic".to_owned());
            }
        }
        result = Some(value);
        sample_milliseconds.push(elapsed);
    }
    Ok(json!({
        "result": result.ok_or_else(|| "no profile samples".to_owned())?.to_string(),
        "sampleMilliseconds": sample_milliseconds,
        "medianMilliseconds": median(&sample_milliseconds)?,
    }))
}

fn profile(plan: &Plan, warmup: usize, samples: usize) -> BenchResult<serde_json::Value> {
    // These setup arrays are deliberately outside every profile interval. Each
    // row below isolates one public-core operation; `run_pass` is still the
    // authoritative benchmark because it includes their normal composition.
    let root_actions = plan
        .cases
        .iter()
        .map(|entry| legal_actions(&entry.config, &entry.state).map_err(normal_error))
        .collect::<BenchResult<Vec<Vec<Action>>>>()?;
    let children = root_actions
        .iter()
        .enumerate()
        .map(|(index, actions)| {
            actions
                .iter()
                .map(|action| {
                    apply_legal_action(&plan.cases[index].config, &plan.cases[index].state, action)
                        .map_err(normal_error)
                })
                .collect::<BenchResult<Vec<GameState>>>()
        })
        .collect::<BenchResult<Vec<Vec<GameState>>>>()?;

    let root_legal = profile_operation(
        || {
            plan.cases.iter().try_fold(0_u64, |total, entry| {
                total
                    .checked_add(
                        legal_actions(&entry.config, &entry.state)
                            .map_err(normal_error)?
                            .len() as u64,
                    )
                    .ok_or_else(|| "profile result overflow".to_owned())
            })
        },
        warmup,
        samples,
    )?;
    let checked_apply = profile_operation(
        || {
            root_actions
                .iter()
                .enumerate()
                .try_fold(0_u64, |total, (index, actions)| {
                    actions.iter().try_fold(total, |case_total, action| {
                        let child = apply_legal_action(
                            &plan.cases[index].config,
                            &plan.cases[index].state,
                            action,
                        )
                        .map_err(normal_error)?;
                        case_total
                            .checked_add(child.position_key.len() as u64)
                            .ok_or_else(|| "profile result overflow".to_owned())
                    })
                })
        },
        warmup,
        samples,
    )?;
    let child_legal = profile_operation(
        || {
            children
                .iter()
                .enumerate()
                .try_fold(0_u64, |total, (index, states)| {
                    states.iter().try_fold(total, |case_total, state| {
                        case_total
                            .checked_add(
                                legal_action_codes(&plan.cases[index].config, state)
                                    .map_err(normal_error)?
                                    .len() as u64,
                            )
                            .ok_or_else(|| "profile result overflow".to_owned())
                    })
                })
        },
        warmup,
        samples,
    )?;
    let scalar_perft = profile_operation(
        || {
            plan.cases.iter().try_fold(0_u64, |total, entry| {
                total
                    .checked_add(
                        perft_with_options(
                            &entry.config,
                            &entry.state,
                            entry.depth,
                            entry.perft_options,
                        )
                        .map_err(normal_error)?,
                    )
                    .ok_or_else(|| "profile result overflow".to_owned())
            })
        },
        warmup,
        samples,
    )?;
    Ok(json!({
        "rootLegal": root_legal,
        "checkedApply": checked_apply,
        "childLegal": child_legal,
        "scalarPerft": scalar_perft,
        "note": "Component setup is outside timings; use the normal full-pass benchmark for the speedup gate."
    }))
}

fn execute() -> BenchResult<()> {
    let arguments = parse_arguments()?;
    let plan = make_plan()?;
    let integrity = verify_plan(&plan)?;
    // This runs after every correctness check and before any warmup or sample.
    let baseline_work_result = run_pass(&plan)?;
    let mut result = json!({
        "benchmarkFormat": WORKLOAD_VERSION,
        "engine": "rust-native-core",
        "fixture": {
            "path": "tests/fixtures/normal-duel-perft-v1.json",
            "sha256": plan.fixture_sha256,
            "caseCount": plan.cases.len(),
        },
        "integrity": {
            "actionChecksum": integrity.action_checksum,
            "perftChecksum": integrity.perft_checksum,
            "rootActionCount": integrity.root_action_count,
            "childActionCount": integrity.child_action_count,
            "expectedLeafTotal": integrity.expected_leaf_total,
            "compactTraceVerified": integrity.compact_trace_verified,
            "workResult": baseline_work_result.to_string(),
        },
        "environment": {
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
        },
        "verifiedBeforeTiming": true,
    });
    if arguments.mode == Mode::Measure {
        let (work_result, samples) = measure(&plan, arguments.warmup, arguments.samples)?;
        if work_result != baseline_work_result {
            return Err("timed work result differs from verified work result".to_owned());
        }
        result["measurement"] = json!({
            "warmupPasses": arguments.warmup,
            "sampleCount": arguments.samples,
            "sampleMilliseconds": samples,
            "workResult": work_result.to_string(),
        });
    } else if arguments.mode == Mode::Profile {
        result["profile"] = profile(&plan, arguments.warmup, arguments.samples)?;
    }
    println!("{result}");
    Ok(())
}

fn entrypoint() {
    if let Err(error) = execute() {
        eprintln!("normal-duel-throughput-benchmark: {error}");
        std::process::exit(1);
    }
}

fn main() {
    entrypoint();
}
