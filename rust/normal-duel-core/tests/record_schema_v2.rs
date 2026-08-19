//! Guards on shard record schema v2: the three carried state columns and the
//! repetition window that travels beside them.
//!
//! The load-bearing claim is that widening the record did not change the GAME.
//! `tests/fixtures/normal-duel-selfplay-record-v1.json` was minted by the build
//! that predates this schema, driving the same options against the same
//! deterministic mock evaluator; the first test here replays those options and
//! requires the first 1229 floats of every record — and the whole meta buffer,
//! which is a second, independent description of the same game — to hash to the
//! digests in that file. A v2 record is a v1 record with three floats after it,
//! and this is the only thing that can say so about bytes rather than about
//! intent.
//!
//! The rest guard the new columns against the ONE failure they exist to
//! prevent: a carried state that looks plausible and is not the state the game
//! was in. Every check here is a relationship between two things the writer
//! produced separately — the record's `ply` against the meta buffer's, the
//! record's `windowLen` against the window buffer's length, `historyStartPly`
//! against the window's own sum — so agreement is evidence and not tautology.

use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::compact_key;
use wrongway_normal_duel::selfplay::{
    Exploration, GameOutcome, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_FLOATS,
    RECORD_META_FIELDS, RECORD_POLICY, RECORD_PREFIX, RECORD_STATE_FIELDS, RECORD_VERSION,
    RECORD_WINDOW_FIELDS,
};
use wrongway_normal_duel::{Config, Coord, Player, Players, JUMP_RULE, RULESET};

const GOLDEN: &str = include_str!("../../../tests/fixtures/normal-duel-selfplay-record-v1.json");

/// Column offsets of the three v2 fields, from the start of a record.
const PLY: usize = RECORD_PREFIX;
const HISTORY_START_PLY: usize = RECORD_PREFIX + 1;
const WINDOW_LEN: usize = RECORD_PREFIX + 2;

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

/// The golden's options, spelled out once so the two tests that use them cannot
/// drift apart.
fn golden_options() -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 16,
        max_considered: 4,
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 8,
        ply_cap: 60,
        seed_base: 7,
        ..SelfPlayOptions::default()
    }
}

/// FNV-1a 64 over little-endian bytes.
///
/// Written out rather than pulled in: the generator that minted the golden ran
/// against a checkout of the previous engine, which has no hashing dependency
/// either, and a digest whose definition lives in a crate version is a digest
/// that can change without anyone editing this file.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn digest_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fnv1a(&bytes)
}

fn digest_i32(values: &[i32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fnv1a(&bytes)
}

/// One field out of the golden. Deliberately a dumb scan rather than a JSON
/// parser: the fixture is a hand-maintained note as much as a data file, and a
/// missing key must be a panic naming the key rather than a `None` that some
/// later `unwrap_or` turns into a passing test.
fn golden_str(key: &str) -> String {
    let needle = format!("\"{key}\":");
    let start = GOLDEN
        .find(&needle)
        .unwrap_or_else(|| panic!("golden fixture has no {key}"))
        + needle.len();
    let rest = GOLDEN[start..].trim_start();
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    let end = rest
        .find(['"', ',', '\n', ']', '}'])
        .unwrap_or_else(|| panic!("golden fixture value for {key} is unterminated"));
    rest[..end].trim().to_owned()
}

fn golden_u64(key: &str) -> u64 {
    golden_str(key)
        .parse()
        .unwrap_or_else(|_| panic!("golden fixture {key} is not a number"))
}

fn golden_digest(key: &str) -> u64 {
    u64::from_str_radix(&golden_str(key), 16)
        .unwrap_or_else(|_| panic!("golden fixture {key} is not hex"))
}

struct Run {
    records: Vec<f32>,
    meta: Vec<i32>,
    window: Vec<u32>,
    count: usize,
    outcomes: Vec<GameOutcome>,
    plies: Vec<u64>,
}

impl Run {
    fn record(&self, index: usize) -> &[f32] {
        &self.records[index * RECORD_FLOATS..(index + 1) * RECORD_FLOATS]
    }

    /// `(key, count)` pairs for record `index`, found by walking the `windowLen`
    /// column — which is the only way a consumer can find them, so walking it is
    /// part of what is under test.
    fn windows(&self) -> Vec<Vec<(u32, u32)>> {
        let mut out = Vec::with_capacity(self.count);
        let mut offset = 0_usize;
        for index in 0..self.count {
            let len = self.record(index)[WINDOW_LEN] as usize;
            let mut entries = Vec::with_capacity(len);
            for entry in 0..len {
                let base = (offset + entry) * RECORD_WINDOW_FIELDS;
                entries.push((self.window[base], self.window[base + 1]));
            }
            offset += len;
            out.push(entries);
        }
        assert_eq!(
            offset * RECORD_WINDOW_FIELDS,
            self.window.len(),
            "the windowLen columns must account for the window buffer exactly: \
             {} entries claimed, {} values present",
            offset * RECORD_WINDOW_FIELDS,
            self.window.len()
        );
        out
    }
}

fn run(config: &Config, options: SelfPlayOptions) -> Run {
    run_taking(config, options, 1)
}

/// As [`run`], but calling `take_records` `takes` times and reading the buffers
/// after the last one.
fn run_taking(config: &Config, options: SelfPlayOptions, takes: usize) -> Run {
    let mut batch = SelfPlayBatch::new(config, options).expect("valid options");
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
    let mut count = 0;
    for _ in 0..takes {
        count = batch.take_records();
    }
    Run {
        records: batch.records().to_vec(),
        meta: batch.record_meta().to_vec(),
        window: batch.record_window().to_vec(),
        count,
        outcomes: batch.outcomes(),
        plies: batch.plies_played(),
    }
}

/// THE golden. Same options, same evaluator, one build apart.
#[test]
fn the_v1_prefix_of_a_v2_record_is_the_v1_record() {
    assert_eq!(
        RECORD_PREFIX,
        golden_u64("prefixFloats") as usize,
        "the fixture describes a different v1 width than this build's prefix"
    );
    assert_eq!(RECORD_FLOATS, RECORD_PREFIX + RECORD_STATE_FIELDS);
    assert_eq!(RECORD_VERSION, 2);

    let run = run(&config(), golden_options());
    assert_eq!(run.count, golden_u64("recordCount") as usize);
    assert_eq!(run.records.len(), run.count * RECORD_FLOATS);
    assert_eq!(run.meta.len(), run.count * RECORD_META_FIELDS);

    // Concatenate only the prefix of every record: that run of bytes is what a
    // v1 build produced end to end, and the three new columns are exactly what
    // is left out.
    let mut prefix = Vec::with_capacity(run.count * RECORD_PREFIX);
    for index in 0..run.count {
        prefix.extend_from_slice(&run.record(index)[..RECORD_PREFIX]);
    }
    assert_eq!(
        format!("{:016x}", digest_f32(&prefix)),
        format!("{:016x}", golden_digest("prefixDigest")),
        "the v1 prefix changed: widening the record perturbed the game or its \
         targets, which is the one thing this schema change may not do"
    );

    // The meta buffer is an independent description of the same game — which
    // game, which ply, whose turn, which move was played — and it did not gain
    // a field. If the prefix matched but this did not, the records would be the
    // old bytes attached to a different game.
    assert_eq!(
        format!("{:016x}", digest_i32(&run.meta)),
        format!("{:016x}", golden_digest("metaDigest")),
        "the same records came out of a different sequence of moves"
    );

    let outcomes: Vec<i32> = run
        .outcomes
        .iter()
        .map(|outcome| match outcome {
            GameOutcome::Win(Player::A) => 1,
            GameOutcome::Win(Player::B) => -1,
            GameOutcome::Draw => 0,
            GameOutcome::Ongoing => 2,
        })
        .collect();
    assert_eq!(outcomes, vec![0, 0, 0, 0], "the golden's games all drew");
    assert_eq!(run.plies, vec![48, 35, 29, 56]);
}

/// `take_records` DRAINS the games into the batch's sinks, so a second call
/// finds nothing left and empties all of them. The window is a third sink added
/// beside the two that already behaved this way, and it has to behave the same.
///
/// This is not idempotence and the test does not ask for it. What it asks is
/// that the three sinks agree about being empty: a missing `clear` on the window
/// alone leaves a second call reporting zero records while the window buffer
/// still holds the first call's entries, which is the shape a consumer cannot
/// detect — the record count is right, the record buffer is right, and every
/// record's window slice comes from a buffer that should not be there.
#[test]
fn a_second_take_records_empties_every_sink_together() {
    let once = run_taking(&config(), golden_options(), 1);
    assert!(once.count > 0 && !once.window.is_empty());

    let twice = run_taking(&config(), golden_options(), 2);
    assert_eq!(twice.count, 0, "the games were already drained");
    assert!(twice.records.is_empty());
    assert!(twice.meta.is_empty());
    assert!(
        twice.window.is_empty(),
        "records and meta emptied but the window kept {} values from the \
         previous take",
        twice.window.len()
    );
}

/// `ply` has a second source: the meta buffer has carried it all along, and
/// nothing in `finish` derives one from the other.
#[test]
fn the_carried_ply_agrees_with_the_meta_buffer() {
    let run = run(&config(), golden_options());
    assert!(run.count > 0);
    for index in 0..run.count {
        let carried = run.record(index)[PLY];
        let meta = run.meta[index * RECORD_META_FIELDS + 1];
        assert_eq!(
            carried, meta as f32,
            "record {index}: carried ply {carried} against meta ply {meta}"
        );
        // The cast is only sound while these stay small integers, so say so
        // where it would break rather than only in the comment that claims it.
        assert!(carried.fract() == 0.0 && (0.0..16_777_216.0).contains(&carried));
    }
}

/// The equation `validateState` will impose from the other side. A reader that
/// rebuilds a state hands it `ply`, `historyStartPly` and the window, and the
/// engine refuses unless the counts sum to `ply - historyStartPly + 1`. Checking
/// it at the writer means a shard cannot be produced that the reader must reject.
#[test]
fn the_window_sums_to_the_span_it_covers() {
    let run = run(&config(), golden_options());
    let windows = run.windows();
    for (index, window) in windows.iter().enumerate() {
        let record = run.record(index);
        let ply = record[PLY] as u64;
        let history_start_ply = record[HISTORY_START_PLY] as u64;
        assert!(
            history_start_ply <= ply,
            "record {index}: window starts at {history_start_ply}, after ply {ply}"
        );
        assert_eq!(
            window.len(),
            record[WINDOW_LEN] as usize,
            "record {index}: windowLen disagrees with the entries found for it"
        );
        assert!(!window.is_empty(), "record {index}: empty window");
        let total: u64 = window.iter().map(|(_, count)| u64::from(*count)).sum();
        assert_eq!(
            total,
            ply - history_start_ply + 1,
            "record {index}: counts sum to {total} over plies {history_start_ply}..={ply}"
        );
        // Keys are unique and ascending. Unique because a count, not a repeat
        // entry, is how a repetition is recorded; ascending because the bytes on
        // disk must be a function of the game and not of insertion order.
        for pair in window.windows(2) {
            assert!(
                pair[0].0 < pair[1].0,
                "record {index}: window keys {:?} are not strictly ascending",
                window.iter().map(|(key, _)| *key).collect::<Vec<_>>()
            );
        }
        assert!(window.iter().all(|(_, count)| *count >= 1));
    }
}

/// A wall placement is the only thing that resets the window, and when it does
/// the window is a single entry at the ply it happened. Asserted on a run that
/// actually contains both cases rather than on a constructed one.
#[test]
fn a_wall_placement_restarts_the_window_and_nothing_else_does() {
    let run = run(&config(), golden_options());
    let windows = run.windows();
    let mut fresh_after_wall = 0;
    let mut extended = 0;
    for (index, window) in windows.iter().enumerate() {
        let record = run.record(index);
        let ply = record[PLY] as u64;
        let history_start_ply = record[HISTORY_START_PLY] as u64;
        if history_start_ply == ply {
            assert_eq!(
                window.len(),
                1,
                "record {index}: window opened at this ply, so it holds one position"
            );
            assert_eq!(window[0].1, 1);
            fresh_after_wall += 1;
        } else {
            extended += 1;
            // An extended window spans more than one ply, so it must account for
            // every one of them — either as distinct positions or as counts.
            let total: u64 = window.iter().map(|(_, count)| u64::from(*count)).sum();
            assert!(
                total > 1,
                "record {index}: extended window covering one ply"
            );
        }
    }
    assert!(
        fresh_after_wall > 0 && extended > 0,
        "the golden run must exercise both a just-reset window ({fresh_after_wall}) \
         and an extended one ({extended}), or this test proves nothing"
    );
}

/// The point of the whole schema: some records carry a position that has ALREADY
/// occurred. Those are the roots a synthesised window gets wrong, and if the
/// golden run had none of them the guards above would be checking arithmetic on
/// a corpus that never exercises the case.
#[test]
fn the_golden_run_carries_repeated_positions() {
    let run = run(&config(), golden_options());
    let windows = run.windows();
    let repeated = windows
        .iter()
        .filter(|window| window.iter().any(|(_, count)| *count >= 2))
        .count();
    assert!(
        repeated > 0,
        "no record in the golden run has a repetition count above 1, so nothing \
         here tests the case the schema exists for"
    );
    // Never at or past the threshold: the game ends when a position occurs a
    // third time, so a RECORDED root cannot already be there.
    for (index, window) in windows.iter().enumerate() {
        for (key, count) in window {
            assert!(
                *count < 3,
                "record {index}: key {key} already at count {count}, which is a \
                 finished game rather than a position the search was asked about"
            );
        }
    }
}

/// The root's own position must be in its own window, or `validateState` refuses
/// the rebuilt state outright. The key is recomputed here from the initial
/// position rather than read back from the record, so this is a check on the
/// writer and not a restatement of it.
#[test]
fn the_first_record_of_a_game_is_the_opening_position() {
    let config = config();
    let run = run(&config, golden_options());
    let windows = run.windows();
    let opening = compact_key(&config, config.start.a, config.start.b, config.first_player);
    // Record 0 is game 0's first recorded ply. With no openings configured that
    // is ply 0: nothing has moved, the window holds the initial position once.
    assert_eq!(run.meta[1], 0, "record 0 is not at ply 0");
    assert_eq!(run.record(0)[PLY], 0.0);
    assert_eq!(run.record(0)[HISTORY_START_PLY], 0.0);
    assert_eq!(windows[0], vec![(opening, 1)]);
}

/// A batch whose worst-case window would blow the budget is refused at
/// construction with a readable error, not by aborting inside the allocator.
#[test]
fn an_unbuildable_window_budget_is_refused_at_construction() {
    let mut deep = config();
    // The window bound only bites above ply_cap ~1231, where the window finally
    // outgrows the records; below that the record bound is the binding one. Move
    // the config cap up so the option cap is reachable and the arithmetic is the
    // window's rather than the records'.
    deep.ply_cap = 2048;
    // 16 games x 2048 plies reserves 161 MB of records, which the record bound
    // passes; its worst-case window is 16 x 2,098,176 entries = 268.6 MB, which
    // the window bound does not. So this case is refused by the NEW bound, which
    // is the one under test — at 15 games both pass and nothing is proved.
    let options = SelfPlayOptions {
        games: 16,
        ply_cap: 2048,
        ..SelfPlayOptions::default()
    };
    let error = SelfPlayBatch::new(&deep, options).expect_err("must refuse");
    assert_eq!(error.to_string(), "invalid_buffer_length");
    SelfPlayBatch::new(
        &deep,
        SelfPlayOptions {
            games: 15,
            ply_cap: 2048,
            ..SelfPlayOptions::default()
        },
    )
    .expect("one game under the bound must still build");

    // And the production shape is nowhere near it.
    let production = SelfPlayOptions {
        games: 32,
        ply_cap: 200,
        ..SelfPlayOptions::default()
    };
    SelfPlayBatch::new(&config(), production).expect("32 games at the production cap must build");
}
