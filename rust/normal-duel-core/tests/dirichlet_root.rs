//! D3 — the Dirichlet root floor — at the search and self-play levels.
//!
//! `P'(a) = (1 - eps) * P(a) + eps * eta_a`, `eta ~ Dir(alpha)`, root only.
//! Three claims are made here and they are tested in three different
//! directions, because none of them implies the others.
//!
//! 1. **`eps = 0` is byte-for-byte absent.** A negative claim, so it is pinned
//!    to digests computed on the build *before* D3 existed — the same two
//!    constants `tests/root_mode_classic.rs` carries, re-asserted here across a
//!    sweep of `alpha` values that must all be unread. This file's [`digest`] is
//!    a copy of that file's, which is self-checking: a copy that had drifted
//!    could not reproduce the constants.
//! 2. **The floor never touches the game's own draw stream.** A rejection
//!    sampler consumes a variable number of words, so the noise gets its own
//!    [`Lcg32`] keyed on `(game_seed, ply)`. The test counts the main stream's
//!    words exactly, at several `eps`, under both root modes.
//! 3. **The floor moves the search and not the recorded target.** The half of
//!    that which is visible from outside the crate is here — the search really
//!    does decide differently, under BOTH root modes. The other half, that the
//!    target is a function of the network's prior alone, needs the root's edge
//!    list and lives in `puct.rs`'s own test module.

use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{
    dirichlet_stream_seed, root_dirichlet, PuctParams, PuctResult, PuctTreeSearch, RootMode,
    DEFAULT_DIRICHLET_ALPHA, DEFAULT_DIRICHLET_EPSILON,
};
use wrongway_normal_duel::selfplay::{
    Exploration, GameOutcome, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_POLICY,
};
use wrongway_normal_duel::{
    create_initial_state, legal_action_codes, Config, Coord, Player, Players, JUMP_RULE, RULESET,
};

fn config() -> Config {
    Config {
        ruleset: RULESET.into(),
        rows: 9,
        columns: 9,
        start: Players {
            a: Coord { r: 8, c: 4 },
            b: Coord { r: 0, c: 4 },
        },
        goal_rows: Players { a: 0, b: 8 },
        initial_stock: Players { a: 10, b: 10 },
        jump_rule: JUMP_RULE.into(),
        repetition_threshold: 3,
        ply_cap: 200,
        first_player: Player::A,
    }
}

/* ------------------------------------------------------------------ *
 * Self-play harness, copied from tests/root_mode_classic.rs
 * ------------------------------------------------------------------ */

struct Run {
    records: Vec<f32>,
    meta: Vec<i32>,
    count: usize,
    outcomes: Vec<GameOutcome>,
    plies: Vec<u64>,
}

fn run(options: SelfPlayOptions) -> Run {
    let config = config();
    let mut batch = SelfPlayBatch::new(&config, options).expect("valid options");
    let mut scratch = vec![0.0_f32; RECORD_POLICY];
    loop {
        let n = batch.collect().expect("collect");
        if n == 0 {
            break;
        }
        for slot in 0..n {
            let features =
                batch.features()[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES].to_vec();
            let value = mock_evaluator::evaluate(&features, &mut scratch);
            batch.policy_mut()[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY]
                .copy_from_slice(&scratch);
            batch.value_mut()[slot] = value as f32;
        }
        batch.submit(n).expect("submit");
    }
    let count = batch.take_records();
    Run {
        records: batch.records().to_vec(),
        meta: batch.record_meta().to_vec(),
        count,
        outcomes: batch.outcomes(),
        plies: batch.plies_played(),
    }
}

/// FNV-1a over every byte a self-play run produces. Identical to
/// `tests/root_mode_classic.rs`'s, which is what makes the constants below
/// meaningful across the two files.
fn digest(run: &Run) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    };
    eat(&(run.count as u64).to_le_bytes());
    for value in &run.records {
        eat(&value.to_bits().to_le_bytes());
    }
    for field in &run.meta {
        eat(&field.to_le_bytes());
    }
    for outcome in &run.outcomes {
        eat(&[match outcome {
            GameOutcome::Ongoing => 0,
            GameOutcome::Win(Player::A) => 1,
            GameOutcome::Win(Player::B) => 2,
            GameOutcome::Draw => 3,
        }]);
    }
    for ply in &run.plies {
        eat(&ply.to_le_bytes());
    }
    hash
}

fn gumbel_argmax_options() -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 64,
        max_considered: 8,
        ply_cap: 24,
        seed_base: 20_260_809,
        ..SelfPlayOptions::default()
    }
}

fn gumbel_temperature_options() -> SelfPlayOptions {
    SelfPlayOptions {
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 16,
        ..gumbel_argmax_options()
    }
}

/// **The pre-D3 baseline.** These are `GUMBEL_ARGMAX_DIGEST` and
/// `GUMBEL_TEMPERATURE_DIGEST` from `tests/root_mode_classic.rs`, over the same
/// two option sets — computed on `scott/completed-q-targets` at 043b28c, a
/// commit on which none of this file's code existed.
///
/// Copied rather than shared because integration tests are separate binaries.
/// The copy cannot rot silently: if this file's `digest`, `run` or option
/// builders differed from that file's by so much as a field, the assertions
/// below would not reproduce these numbers.
const PRE_D3_ARGMAX_DIGEST: u64 = 0xf4b3_b3a9_c83b_5534;
const PRE_D3_TEMPERATURE_DIGEST: u64 = 0x8eb8_8c62_f02a_f7c5;

/// Claim 1: `eps = 0` is not "a mixture with weight zero", it is *absent*.
///
/// Every `alpha` in the sweep is a value that would produce wildly different
/// noise if it were read at all — 0.02 is near-degenerate, 0.99 is nearly
/// uniform — and a NaN alpha would poison every prior it reached. All of them
/// must reproduce a shard byte-for-byte identical to the pre-D3 build's.
#[test]
fn zero_epsilon_reproduces_the_pre_d3_build_byte_for_byte() {
    for alpha in [DEFAULT_DIRICHLET_ALPHA, 0.02, 0.5, 0.99, f64::NAN] {
        let argmax = digest(&run(SelfPlayOptions {
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: alpha,
            ..gumbel_argmax_options()
        }));
        let temperature = digest(&run(SelfPlayOptions {
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: alpha,
            ..gumbel_temperature_options()
        }));
        assert_eq!(
            (argmax, temperature),
            (PRE_D3_ARGMAX_DIGEST, PRE_D3_TEMPERATURE_DIGEST),
            "alpha {alpha} leaked into a run with eps = 0: argmax {argmax:#018x}, \
             temperature {temperature:#018x}"
        );
    }
}

/// The other side of the same coin, and the anti-vacuity guard for the test
/// above: switching the floor ON must move the shard, and how far it moves must
/// depend on `alpha`. Without this, an `apply_root_dirichlet` that returned
/// early always would pass everything above.
#[test]
fn a_positive_epsilon_moves_the_shard_and_alpha_moves_it_differently() {
    let with = |epsilon: f64, alpha: f64| -> u64 {
        digest(&run(SelfPlayOptions {
            dirichlet_epsilon: epsilon,
            dirichlet_alpha: alpha,
            ..gumbel_temperature_options()
        }))
    };
    let plain = with(0.0, DEFAULT_DIRICHLET_ALPHA);
    assert_eq!(plain, PRE_D3_TEMPERATURE_DIGEST);

    let floored = with(DEFAULT_DIRICHLET_EPSILON, DEFAULT_DIRICHLET_ALPHA);
    assert_ne!(
        floored, plain,
        "eps = {DEFAULT_DIRICHLET_EPSILON} produced the same shard as no floor at all"
    );

    // `alpha` is read, and read as a shape rather than as a flag.
    let sparser = with(DEFAULT_DIRICHLET_EPSILON, 0.02);
    let flatter = with(DEFAULT_DIRICHLET_EPSILON, 0.9);
    assert_ne!(floored, sparser);
    assert_ne!(floored, flatter);
    assert_ne!(sparser, flatter);

    // And so is `eps` itself, beyond being positive.
    assert_ne!(with(0.5, DEFAULT_DIRICHLET_ALPHA), floored);
}

/// **A regression pin on the floored path, and the only test here that can see
/// the self-play plumbing.**
///
/// Unlike [`PRE_D3_ARGMAX_DIGEST`] this number was computed from this
/// implementation, so it attests that the floored path does not MOVE, not that
/// it is right; the claims that it is right are the tests around it. Its
/// specific job is what those cannot reach: that `Game::advance` hands the
/// search `seed_base + index` as the game seed and the search keys on
/// `(that, ply)`. A `game_seed` wired to `0`, to the game index, or to the live
/// stream state would still produce a self-consistent, reproducible,
/// eps-sensitive shard — every other assertion in this file would pass — and it
/// would be a shard in which every game shared one noise sequence.
///
/// So if this moves, name the change. Do not recompute it to make a build green.
const FLOORED_TEMPERATURE_DIGEST: u64 = 0xc239_e45d_00ec_102a;

/// The floor is keyed on the game seed, so two shards that differ only in
/// `seed_base` must differ in their noise as well as in their Gumbel draws —
/// and the same shard must reproduce exactly.
#[test]
fn a_floored_shard_is_reproducible_and_seed_dependent() {
    let options = SelfPlayOptions {
        dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
        ..gumbel_temperature_options()
    };
    let first = run(options.clone());
    let second = run(options.clone());
    let pinned = digest(&first);
    assert_eq!(
        pinned, FLOORED_TEMPERATURE_DIGEST,
        "the floored path produced {pinned:#018x}"
    );
    assert_eq!(first.count, second.count);
    assert_eq!(first.meta, second.meta);
    assert!(
        first.records == second.records,
        "two floored runs from the same seed produced different records"
    );

    let moved = run(SelfPlayOptions {
        seed_base: options.seed_base + 1,
        ..options
    });
    assert_ne!(digest(&first), digest(&moved));
}

/// The floor applies to the classic arm too — D4's design says so, and its
/// Dirichlet has nowhere else to enter than `select_edge`'s `P`.
#[test]
fn the_floor_moves_the_classic_arm_as_well() {
    let classic = |epsilon: f64| -> u64 {
        digest(&run(SelfPlayOptions {
            games: 3,
            simulations: 128,
            max_considered: 8,
            root_mode: RootMode::Classic,
            dirichlet_epsilon: epsilon,
            ply_cap: 12,
            seed_base: 20_260_809,
            temperature: 1.0,
            temperature_moves: 6,
            ..SelfPlayOptions::default()
        }))
    };
    assert_ne!(
        classic(DEFAULT_DIRICHLET_EPSILON),
        classic(0.0),
        "the classic root ignored the Dirichlet floor"
    );
}

/* ------------------------------------------------------------------ *
 * Claim 2: the main stream is untouched
 * ------------------------------------------------------------------ */

/// Drive one search to completion against the mock network and return what it
/// decided, plus the game's draw stream as it left it.
fn search(root_mode: RootMode, epsilon: f64, alpha: f64, seed: u32) -> (PuctResult, Lcg32) {
    search_keyed(root_mode, epsilon, alpha, seed, seed)
}

/// The same, with the game seed the noise is keyed on and the seed the main
/// stream runs off given separately — which is the only way to tell the two
/// apart, since in production they are the same number.
fn search_keyed(
    root_mode: RootMode,
    epsilon: f64,
    alpha: f64,
    game_seed: u32,
    stream_seed: u32,
) -> (PuctResult, Lcg32) {
    let config = config();
    let state = create_initial_state(&config).expect("initial state");
    let mut tree = PuctTreeSearch::from_state(
        &config,
        &state,
        PuctParams {
            simulations: 96,
            max_considered: 8,
            c_puct: 1.25,
            root_mode,
            game_seed,
            dirichlet_epsilon: epsilon,
            dirichlet_alpha: alpha,
        },
        Lcg32::new(stream_seed),
    )
    .expect("an ongoing 9x9 state starts a search");

    let mut features = vec![0.0_f32; RECORD_FEATURES];
    let mut policy = vec![0.0_f32; RECORD_POLICY];
    while tree.next_leaf(&config, &mut features).expect("next_leaf") {
        let value = mock_evaluator::evaluate(&features, &mut policy);
        tree.submit(&config, &policy, value).expect("submit");
    }
    (tree.result(), tree.rng())
}

fn legal_root_actions() -> usize {
    let config = config();
    let state = create_initial_state(&config).expect("initial state");
    legal_action_codes(&config, &state)
        .expect("legal action codes")
        .len()
}

/// **Claim 2, exactly.** The Gumbel root takes one word per legal root action
/// and the classic root takes none, at every `eps`. Not "the same number of
/// words as at `eps = 0`" — the count is computed here from the legal action
/// count, so a floor that quietly moved the main stream by one word would fail
/// even if it moved it identically at both settings.
///
/// This is the whole reason the noise has a stream of its own. The gamma
/// sampler's rejection loop consumes a variable number of words; the main
/// stream's offset is contractually a pure function of the ply index, which is
/// what lets `cluster-selfplay.mjs` reproduce a game and what
/// `zero_temperature_moves_consumes_no_draws` in `tests/selfplay_exploration.rs`
/// holds the temperature draw to.
#[test]
fn the_floor_takes_no_word_from_the_games_own_stream() {
    const SEED: u32 = 31_337;
    let legal = legal_root_actions();
    assert!(legal > 100, "the opening has {legal} legal actions");

    let mut expected = Lcg32::new(SEED);
    for _ in 0..legal {
        expected.gumbel();
    }
    let untouched = Lcg32::new(SEED);
    assert_ne!(
        expected, untouched,
        "the Gumbel root is supposed to consume words; this test would be vacuous otherwise"
    );

    for epsilon in [0.0, 0.001, DEFAULT_DIRICHLET_EPSILON, 1.0] {
        let (_, after) = search(RootMode::Gumbel, epsilon, DEFAULT_DIRICHLET_ALPHA, SEED);
        assert_eq!(
            after, expected,
            "at eps = {epsilon} the Gumbel root left the main stream somewhere other than \
             {legal} words in"
        );
        let (_, after) = search(RootMode::Classic, epsilon, DEFAULT_DIRICHLET_ALPHA, SEED);
        assert_eq!(
            after, untouched,
            "at eps = {epsilon} the classic root moved the main stream at all"
        );
    }

    // The rejection sampler's draw count varies with alpha, so a floor that
    // drew from the main stream would show a DIFFERENT offset per alpha rather
    // than merely a wrong one.
    for alpha in [0.02, DEFAULT_DIRICHLET_ALPHA, 0.9] {
        let (_, after) = search(RootMode::Gumbel, DEFAULT_DIRICHLET_EPSILON, alpha, SEED);
        assert_eq!(after, expected, "alpha {alpha} moved the main stream");
    }
}

/// The floor changes what each root mode does with its budget. Together with
/// the stream test above this is "perturbs the search, not the bookkeeping".
#[test]
fn the_floor_changes_the_search_under_both_root_modes() {
    const SEED: u32 = 4242;
    let (plain, _) = search(RootMode::Gumbel, 0.0, DEFAULT_DIRICHLET_ALPHA, SEED);
    let (floored, _) = search(
        RootMode::Gumbel,
        DEFAULT_DIRICHLET_EPSILON,
        DEFAULT_DIRICHLET_ALPHA,
        SEED,
    );
    assert_ne!(
        plain.considered, floored.considered,
        "the Gumbel considered set is drawn from `g + log P'`, so the floor must reach it"
    );

    let (plain, _) = search(RootMode::Classic, 0.0, DEFAULT_DIRICHLET_ALPHA, SEED);
    let (floored, _) = search(
        RootMode::Classic,
        DEFAULT_DIRICHLET_EPSILON,
        DEFAULT_DIRICHLET_ALPHA,
        SEED,
    );
    assert_ne!(
        plain.visit_counts, floored.visit_counts,
        "the classic root's only prior read is `select_edge`'s P, so the floor must reach it"
    );
    // Same tree, same budget: it is where the budget went that moved.
    assert_eq!(plain.simulations_used, floored.simulations_used);
}

/// The noise is keyed on the GAME seed, not on the stream the search happens to
/// be handed. Two searches with the same draw stream and different game seeds
/// must explore differently; two with the same game seed and the same stream
/// must not.
///
/// In production those two numbers are equal — `Game::advance` passes the game's
/// own seed — so nothing else in this file can separate them, and a
/// `game_seed` that silently arrived as `0` for every game would look exactly
/// like a correct one.
#[test]
fn the_floor_is_keyed_on_the_game_seed_rather_than_the_stream_it_is_handed() {
    const STREAM: u32 = 90_210;
    let of = |game_seed: u32| {
        search_keyed(
            RootMode::Gumbel,
            DEFAULT_DIRICHLET_EPSILON,
            DEFAULT_DIRICHLET_ALPHA,
            game_seed,
            STREAM,
        )
        .0
    };
    let first = of(1);
    assert_eq!(first, of(1));
    assert_ne!(
        first,
        of(2),
        "two games sharing a draw stream got the same root noise"
    );
    // And the game seed is not read when the floor is off, so a caller that has
    // no game to name is not forced to invent one.
    assert_eq!(
        search_keyed(RootMode::Gumbel, 0.0, DEFAULT_DIRICHLET_ALPHA, 1, STREAM).0,
        search_keyed(RootMode::Gumbel, 0.0, DEFAULT_DIRICHLET_ALPHA, 999, STREAM).0
    );
}

/// Same seed, same everything.
#[test]
fn a_floored_search_is_reproducible() {
    let (first, _) = search(
        RootMode::Gumbel,
        DEFAULT_DIRICHLET_EPSILON,
        DEFAULT_DIRICHLET_ALPHA,
        11,
    );
    let (second, _) = search(
        RootMode::Gumbel,
        DEFAULT_DIRICHLET_EPSILON,
        DEFAULT_DIRICHLET_ALPHA,
        11,
    );
    assert_eq!(first, second);
}

/* ------------------------------------------------------------------ *
 * The noise itself
 * ------------------------------------------------------------------ */

/// `eta` is a point on the simplex, whatever the key and whatever the alpha.
#[test]
fn the_noise_is_a_distribution() {
    for (seed, ply, alpha, count) in [
        (0_u32, 0_u64, DEFAULT_DIRICHLET_ALPHA, 1_usize),
        (1, 0, DEFAULT_DIRICHLET_ALPHA, 2),
        (20_260_809, 7, 0.02, 131),
        (20_260_809, 7, 0.9, 44),
        (u32::MAX, u64::MAX, 0.999, 3),
    ] {
        let noise = root_dirichlet(seed, ply, alpha, count);
        assert_eq!(noise.len(), count);
        let total: f64 = noise.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "Dir({alpha}) over {count} sums to {total}"
        );
        for eta in &noise {
            assert!(
                eta.is_finite() && *eta >= 0.0 && *eta <= 1.0,
                "component {eta} is not a probability"
            );
        }
    }
    // One component is the degenerate case, and it must be exactly 1.0 rather
    // than 1.0 - 1e-17: it multiplies a prior that is itself exactly 1.0.
    assert_eq!(root_dirichlet(9, 9, 0.15, 1), vec![1.0]);
    assert!(root_dirichlet(9, 9, 0.15, 0).is_empty());
}

/// The key is `(game_seed, ply)` and both halves of it matter. Neighbouring
/// games and neighbouring plies are the cases that actually occur — game `i`
/// seeds off `seed_base + i` — and an LCG seeded with adjacent values is where
/// a missing mix step would show.
#[test]
fn the_noise_is_keyed_on_the_game_and_the_ply() {
    let base = root_dirichlet(1000, 5, DEFAULT_DIRICHLET_ALPHA, 40);
    assert_eq!(base, root_dirichlet(1000, 5, DEFAULT_DIRICHLET_ALPHA, 40));
    assert_ne!(base, root_dirichlet(1001, 5, DEFAULT_DIRICHLET_ALPHA, 40));
    assert_ne!(base, root_dirichlet(1000, 6, DEFAULT_DIRICHLET_ALPHA, 40));

    // No collisions across a whole shard's worth of keys: 64 games x 200 plies.
    let mut seeds: Vec<u32> = Vec::with_capacity(64 * 200);
    for game in 0..64_u32 {
        for ply in 0..200_u64 {
            seeds.push(dirichlet_stream_seed(
                20_260_809_u32.wrapping_add(game),
                ply,
            ));
        }
    }
    let total = seeds.len();
    seeds.sort_unstable();
    seeds.dedup();
    assert_eq!(
        seeds.len(),
        total,
        "{} of {total} Dirichlet stream keys collided",
        total - seeds.len()
    );

    // And the stream is not the game's own: keying it off the game seed
    // unmixed would have made the noise's first word the game's first word.
    for game_seed in [0_u32, 1, 20_260_809] {
        assert_ne!(dirichlet_stream_seed(game_seed, 0), game_seed);
    }
}

/// `alpha` is a concentration, not a decoration: a small one puts nearly all
/// the mass on one component, a large one spreads it. Measured over many keys
/// so the claim is about the distribution rather than about one draw.
#[test]
fn alpha_controls_how_concentrated_the_noise_is() {
    const COUNT: usize = 40;
    const KEYS: u32 = 400;
    let mean_top = |alpha: f64| -> f64 {
        let mut total = 0.0_f64;
        for key in 0..KEYS {
            let noise = root_dirichlet(key, u64::from(key % 17), alpha, COUNT);
            total += noise.iter().copied().fold(0.0_f64, f64::max);
        }
        total / f64::from(KEYS)
    };
    let sparse = mean_top(0.02);
    let planned = mean_top(DEFAULT_DIRICHLET_ALPHA);
    let flat = mean_top(0.9);
    assert!(
        sparse > planned && planned > flat,
        "mean top component: alpha 0.02 -> {sparse}, {DEFAULT_DIRICHLET_ALPHA} -> {planned}, \
         0.9 -> {flat}"
    );
    // An ordering alone would be satisfied by any monotone function of alpha,
    // including a wrong one, so the three values are also checked against an
    // independent implementation: Python's `random.gammavariate` normalised the
    // same way gives mean top components of 0.6815, 0.2900 and 0.1125 over 4000
    // draws of Dir(alpha) on 40 components (its own sampling error is about
    // 0.005 at this count). A sampler that produced the right ORDER from the
    // wrong distribution -- the failure a threshold-free test cannot see -- does
    // not land inside these bands.
    for (alpha, measured, reference) in [
        (0.02, sparse, 0.6815),
        (DEFAULT_DIRICHLET_ALPHA, planned, 0.2900),
        (0.9, flat, 0.1125),
    ] {
        assert!(
            (measured - reference).abs() < 0.02,
            "Dir({alpha}) over {COUNT} has mean top component {measured}; \
             an independent sampler gives {reference}"
        );
    }

    // Every marginal has mean 1/n, so no index is privileged -- the sampler
    // draws each component from the same distribution rather than, say,
    // consuming the stream in a way that biases the first.
    let mut means = vec![0.0_f64; COUNT];
    for key in 0..KEYS {
        for (slot, eta) in root_dirichlet(key, 3, DEFAULT_DIRICHLET_ALPHA, COUNT)
            .iter()
            .enumerate()
        {
            means[slot] += eta / f64::from(KEYS);
        }
    }
    let expected = 1.0 / COUNT as f64;
    for (slot, mean) in means.iter().enumerate() {
        assert!(
            (mean - expected).abs() < 0.5 * expected,
            "component {slot} averaged {mean}, not about {expected}"
        );
    }
}

/* ------------------------------------------------------------------ *
 * Options validation
 * ------------------------------------------------------------------ */

/// A malformed floor is rejected at construction, by the batch and by the
/// search, rather than producing NaN priors somewhere inside a shard.
#[test]
fn a_malformed_floor_is_rejected() {
    let config = config();
    let state = create_initial_state(&config).expect("initial state");
    let batch = |epsilon: f64, alpha: f64| {
        SelfPlayBatch::new(
            &config,
            SelfPlayOptions {
                dirichlet_epsilon: epsilon,
                dirichlet_alpha: alpha,
                ..gumbel_argmax_options()
            },
        )
        .map(|_| ())
    };
    let tree = |epsilon: f64, alpha: f64| {
        PuctTreeSearch::from_state(
            &config,
            &state,
            PuctParams {
                dirichlet_epsilon: epsilon,
                dirichlet_alpha: alpha,
                ..PuctParams::default()
            },
            Lcg32::new(1),
        )
        .map(|_| ())
    };

    for (epsilon, alpha) in [
        (-0.1, DEFAULT_DIRICHLET_ALPHA),
        (1.1, DEFAULT_DIRICHLET_ALPHA),
        (f64::NAN, DEFAULT_DIRICHLET_ALPHA),
        (f64::INFINITY, DEFAULT_DIRICHLET_ALPHA),
        // alpha is only read when the floor is on, and then it must be a
        // concentration the one sampler this crate has can produce.
        (0.25, 0.0),
        (0.25, -0.5),
        (0.25, 1.0),
        (0.25, 4.0),
        (0.25, f64::NAN),
    ] {
        assert_eq!(
            batch(epsilon, alpha).map_err(|error| error.reason()),
            Err("invalid_dirichlet"),
            "SelfPlayBatch accepted eps = {epsilon}, alpha = {alpha}"
        );
        assert_eq!(
            tree(epsilon, alpha).map_err(|error| error.reason()),
            Err("invalid_dirichlet"),
            "PuctTreeSearch accepted eps = {epsilon}, alpha = {alpha}"
        );
    }

    // The boundary values are legal, and an unread alpha is not policed.
    for (epsilon, alpha) in [
        (0.0, f64::NAN),
        (0.0, 0.0),
        (1.0, DEFAULT_DIRICHLET_ALPHA),
        (DEFAULT_DIRICHLET_EPSILON, 0.999),
    ] {
        assert!(
            batch(epsilon, alpha).is_ok() && tree(epsilon, alpha).is_ok(),
            "eps = {epsilon}, alpha = {alpha} should be accepted"
        );
    }
}
