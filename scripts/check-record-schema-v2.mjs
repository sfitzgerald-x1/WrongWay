#!/usr/bin/env node
/**
 * Self-checks for the carried state in shard record schema v2.
 *
 *     node scripts/build-normal-duel-wasm-candidate.mjs
 *     node scripts/check-record-schema-v2.mjs
 *
 * WHAT IS ACTUALLY UNDER TEST, and why the loop has to be closed here rather
 * than on either side of it. The Rust writer puts `ply`, `historyStartPly` and a
 * repetition window into every record; `stateFromCarried` turns them back into a
 * `GameState`. A Rust-side test can show the writer is self-consistent and a
 * JS-side test can show the reader accepts what it is handed, and BOTH can pass
 * while the two disagree about what a window entry means -- which packing, which
 * player the turn bit names, whether the window is the state before the played
 * move or after it. Every one of those produces a state `validateState` accepts,
 * because they are all legal states; they are simply not the state the game was
 * in, which is the entire failure this schema exists to end.
 *
 * So the ground truth here is neither side's bookkeeping. It is the JS reference
 * engine REPLAYING the moves: the meta buffer says which move was played at
 * every recorded ply, `createInitialState` plus `applyLegalAction` is the
 * engine's own definition of the resulting state, and the carried state must
 * equal it field for field -- ply, historyStartPly, and the repetition window
 * position key by position key. Nothing in that chain is shared with the writer.
 *
 * No network, no cluster, no shard, no node_modules. The wasm candidate is
 * required (it IS the writer) and the check skips cleanly when it is not built,
 * because `rust/target` is gitignored.
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  applyLegalAction, createInitialState, decodeAction, policySize
} from '../js/normal-duel-engine.mjs';
import { encodeState, legalMaskFloat } from '../js/normal-duel-nn-encoding.mjs';
import { hashFeatures, mix32 } from '../js/normal-duel-mock-evaluator.mjs';
import {
  FeatureInversionError, SELFPLAY_CONFIG as CONFIG, featureLength, stateFromCarried,
  verifyCarriedRecord
} from './feature-inversion.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RELEASE = path.join(ROOT, 'rust/target/wasm-candidate/release');
const FEATURE_LEN = featureLength(CONFIG);
const POLICY_SIZE = policySize(CONFIG);
const FAILURES = [];

function report(ok, name, detail) {
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name.padEnd(40)} -> ${detail}`);
  if (!ok) FAILURES.push(`${name}: ${detail}`);
}

/** A 24-bit fraction in [0, 1), exact in f32 and f64 alike. */
function unit(word) { return (word >>> 8) / 16777216; }

/**
 * Drive one self-play batch through the wasm boundary and hand back everything
 * it recorded, including the window.
 *
 * The view discipline is the one the boundary documents: rebuild after every
 * call into wasm, and read the window only AFTER `takeRecords` -- that buffer is
 * not reserved to its worst case, so `takeRecords` is exactly the call that may
 * grow the heap under it.
 */
async function runBatch(options) {
  const wasm = await import(path.join(RELEASE, 'normal-duel-wasm.mjs'));
  const instance = await wasm.default({
    module_or_path: readFileSync(path.join(RELEASE, 'normal-duel-wasm_bg.wasm'))
  });
  const memory = instance.memory ?? wasm.__wasm?.memory ?? wasm.memory;
  const layout = JSON.parse(wasm.normalDuelSelfPlayLayout());
  const batch = new wasm.NormalDuelSelfPlayBatch(
    JSON.stringify(CONFIG), JSON.stringify(options));
  const view = (ptr, len) => new Float32Array(memory.buffer, ptr, len);

  for (;;) {
    const n = batch.collect();
    if (n === 0) break;
    const features = view(batch.featuresPtr(), batch.featuresLen());
    const policy = new Float32Array(n * POLICY_SIZE);
    const values = new Float32Array(n);
    for (let slot = 0; slot < n; slot += 1) {
      const hash = hashFeatures(
        features.subarray(slot * FEATURE_LEN, (slot + 1) * FEATURE_LEN));
      for (let code = 0; code < POLICY_SIZE; code += 1) {
        policy[slot * POLICY_SIZE + code] =
          unit(mix32((hash ^ Math.imul(code, 0x9e3779b1)) >>> 0));
      }
      values[slot] = unit(mix32((hash ^ 0xdeadbeef) >>> 0)) * 2 - 1;
    }
    view(batch.policyPtr(), batch.policyLen()).set(policy);
    view(batch.valuePtr(), batch.valueLen()).set(values);
    batch.submit(n);
  }

  const count = batch.takeRecords();
  const records = view(batch.recordsPtr(), batch.recordsLen()).slice();
  const meta = new Int32Array(
    memory.buffer, batch.recordMetaPtr(), batch.recordMetaLen()).slice();
  const window = new Uint32Array(
    memory.buffer, batch.recordWindowPtr(), batch.recordWindowLen()).slice();
  return { layout, count, records, meta, window };
}

/**
 * Cut the flat buffers into one object per record, walking `windowLen` to find
 * each record's slice of the window. That walk is the consumer's only route to
 * the window, so doing it here is part of what is being checked rather than
 * setup for it.
 */
function cutRecords({ layout, count, records, meta, window }) {
  const { features: F, policy: P, recordFloats: R, stateFields, windowFields } = layout;
  const out = [];
  let entry = 0;
  for (let i = 0; i < count; i += 1) {
    const base = i * R;
    const windowLen = records[base + F + 2 * P + 3];
    const carried = [];
    for (let k = 0; k < windowLen; k += 1) {
      carried.push([window[(entry + k) * windowFields], window[(entry + k) * windowFields + 1]]);
    }
    entry += windowLen;
    out.push({
      features: records.subarray(base, base + F),
      legalMask: records.subarray(base + F + P, base + F + 2 * P),
      z: records[base + F + 2 * P],
      ply: records[base + F + 2 * P + 1],
      historyStartPly: records[base + F + 2 * P + 2],
      windowLen,
      window: carried,
      game: meta[i * 4],
      metaPly: meta[i * 4 + 1],
      playedCode: meta[i * 4 + 3]
    });
  }
  return { cut: out, entriesConsumed: entry, stateFields };
}

const options = {
  games: 4,
  simulations: 16,
  maxConsidered: 4,
  temperature: 1.0,
  temperatureMoves: 8,
  plyCap: 60,
  seedBase: 7
};

if (!existsSync(path.join(RELEASE, 'normal-duel-wasm_bg.wasm'))) {
  // Locally this is a skip: `rust/target` is gitignored and a contributor who
  // has not built the candidate should not get a red suite for it.
  //
  // In CI it is a FAILURE, and the difference matters more than it looks. This
  // is the ONLY check anywhere that drives the Rust writer and the JS reader
  // against each other; every other suite tests one side. If the build step
  // ahead of it is renamed, moved to another job, or reordered, the skip makes
  // the one end-to-end fidelity check go silently green -- and a green suite
  // saying nothing is worse than a red one saying something.
  const required = process.env.WW_REQUIRE_WASM === '1' || process.env.CI === 'true';
  const message = 'wasm candidate not built at ' + RELEASE
    + ' (node scripts/build-normal-duel-wasm-candidate.mjs)';
  if (required) {
    console.error('FAIL ' + message
      + '\n     This check needs it: it is the only place the writer and the reader'
      + '\n     are compared. Skipping it in CI would leave that comparison unrun.');
    process.exit(1);
  }
  console.log('SKIP: ' + message);
  process.exit(0);
}

const raw = await runBatch(options);
report(raw.layout.recordVersion === 2, 'the engine emits record v2',
  `recordVersion ${raw.layout.recordVersion}, stateFields ${raw.layout.stateFields}, `
  + `recordFloats ${raw.layout.recordFloats}`);
report(raw.layout.recordFloats === raw.layout.features + 2 * raw.layout.policy + 1
  + raw.layout.stateFields, 'the layout adds up',
  `${raw.layout.features} + 2*${raw.layout.policy} + 1 + ${raw.layout.stateFields}`);

const { cut, entriesConsumed } = cutRecords(raw);
report(entriesConsumed * raw.layout.windowFields === raw.window.length,
  'windowLen accounts for the buffer',
  `${entriesConsumed} entries claimed, ${raw.window.length / raw.layout.windowFields} present`);

// ------------------------------------------------------------- the real check
//
// Replay each game through the reference engine and require the carried state to
// BE the replayed state. `applyLegalAction` is the engine's own transition, so
// nothing the writer computed is reused here -- not the ply, not the window
// start, not a single count.

const byGame = new Map();
for (const record of cut) {
  if (!byGame.has(record.game)) byGame.set(record.game, []);
  byGame.get(record.game).push(record);
}

/** The packed key of a state's own position, in the writer's packing. */
function packKey(position) {
  const square = ({ r, c }) => r * CONFIG.columns + c;
  return square(position.pawns.A)
    | (square(position.pawns.B) << 8)
    | ((position.turn === 'B' ? 1 : 0) << 16);
}

let checked = 0;
let repeatedPositions = 0;
let multiEntryWindows = 0;
let windowsAfterAWall = 0;
let firstMismatch = null;
/** Terminal states reached after the last recorded ply, for the outcome case. */
const terminals = [];

for (const [game, plies] of [...byGame.entries()].sort(([a], [b]) => a - b)) {
  let state = createInitialState(CONFIG);
  // The window carried forward independently of the writer, updated by the SAME
  // two rules the engine uses: a wall restarts it, a pawn move extends it. Each
  // record checks it against the writer's, so by the end of a game it is a
  // validated window -- which is what lets the terminal case below exist, since
  // the terminal state itself was never recorded.
  let tracked = new Map();
  let trackedStart = 0;
  for (const record of plies) {
    const carried = {
      ply: record.ply,
      historyStartPly: record.historyStartPly,
      window: record.window
    };
    let rebuilt;
    try {
      rebuilt = verifyCarriedRecord(CONFIG, record.features, record.legalMask, carried);
    } catch (error) {
      firstMismatch ??= `game ${game} ply ${record.ply}: ${error.reason ?? error.code ?? error.message}`;
      break;
    }

    const truth = JSON.stringify({
      ply: state.ply,
      historyStartPly: state.historyStartPly,
      repetitionCounts: state.repetitionCounts,
      positionKey: state.positionKey
    });
    const claim = JSON.stringify({
      ply: rebuilt.ply,
      historyStartPly: rebuilt.historyStartPly,
      repetitionCounts: rebuilt.repetitionCounts,
      positionKey: rebuilt.positionKey
    });
    if (truth !== claim) {
      firstMismatch ??= `game ${game} ply ${record.ply}:\n    replayed ${truth}\n    carried  ${claim}`;
      break;
    }

    checked += 1;
    if (state.repetitionCounts.some(({ count }) => count >= 2)) repeatedPositions += 1;
    if (record.windowLen > 1) multiEntryWindows += 1;
    if (record.historyStartPly > 0 && record.historyStartPly === record.ply) windowsAfterAWall += 1;

    tracked = new Map(record.window);
    trackedStart = record.historyStartPly;
    const action = decodeAction(CONFIG, record.playedCode);
    state = applyLegalAction(CONFIG, state, action);
    const key = packKey(state.position);
    if (action.kind === 'wall') {
      tracked = new Map([[key, 1]]);
      trackedStart = state.ply;
    } else {
      tracked.set(key, (tracked.get(key) ?? 0) + 1);
    }
  }
  if (state.outcome.kind !== 'ongoing') {
    terminals.push({ game, state, window: [...tracked.entries()], historyStartPly: trackedStart });
  }
}

report(firstMismatch === null && checked === cut.length,
  'every carried state is the replayed state',
  firstMismatch ?? `${checked} records across ${byGame.size} games`);

// A run in which nothing ever repeats and no wall is ever placed would pass the
// comparison above while testing none of the arithmetic that makes the schema
// necessary. Assert the corpus is not that run.
report(repeatedPositions > 0, 'the corpus contains repeated positions',
  `${repeatedPositions} of ${checked} records sit on a position already seen`);
report(multiEntryWindows > 0, 'the corpus contains multi-entry windows',
  `${multiEntryWindows} of ${checked} records carry more than one position`);
report(windowsAfterAWall > 0, 'the corpus contains just-reset windows',
  `${windowsAfterAWall} of ${checked} records open at their own ply`);

// -------------------------------------------------- the outcome is DERIVED
//
// Every RECORDED position is ongoing -- a game ends when a position occurs a
// third time, so a root the search was asked about never is one -- which means
// the recorded corpus cannot tell a derived outcome from a hardcoded
// `{ kind: 'ongoing' }`. The states one move PAST the last record can: they are
// the threefold draws the games ended on, and their window says so. The state is
// built from the tracked window, which every record above has already checked
// against the writer's.
const draws = terminals.filter(({ state }) => state.outcome.reason === 'threefold_repetition');
report(draws.length > 0, 'the games reach a real threefold',
  `${draws.length} of ${terminals.length} terminal states`);
if (draws.length > 0) {
  const { state, window, historyStartPly } = draws[0];
  // Caught, not allowed to propagate: a builder that hands `validateState` an
  // outcome it did not derive is refused as `invalid_state`, and an uncaught
  // throw would exit this script with no failure marker at all -- which reads as
  // a crashed harness rather than as the finding it is.
  let rebuilt = null;
  try {
    rebuilt = stateFromCarried(CONFIG, encodeState(CONFIG, state),
      { ply: state.ply, historyStartPly, window });
  } catch (error) {
    report(false, 'a threefold window rebuilds as a draw',
      `refused: ${error.reason ?? error.code ?? error.message}`);
  }
  if (rebuilt) {
    report(rebuilt.outcome.kind === 'draw' && rebuilt.outcome.reason === 'threefold_repetition',
      'a threefold window rebuilds as a draw',
      `${rebuilt.outcome.kind}/${rebuilt.outcome.reason ?? '-'} at ply ${state.ply}`);
    // And the mask agrees: a terminal state has no legal action, so the audit
    // would refuse a rebuild that read it as ongoing -- which is what makes this
    // the same check the corpus records get, not a weaker one.
    const mask = legalMaskFloat(CONFIG, rebuilt);
    report(mask.every((value) => value === 0), 'the rebuilt draw has no legal action',
      `${mask.reduce((sum, value) => sum + value, 0)} legal codes`);
  }
}

// ------------------------------------------------------------- the refusals
//
// A carried state that has been corrupted must be REFUSED with a named reason,
// not repaired and not quietly accepted. Each case perturbs exactly one field of
// a record that passes untouched, so a refusal here is attributable.

const sample = cut.find((record) => record.windowLen > 1) ?? cut[0];
const base = {
  ply: sample.ply,
  historyStartPly: sample.historyStartPly,
  window: sample.window.map((pair) => [...pair])
};

function refuses(name, mutate, wanted) {
  const carried = {
    ply: base.ply,
    historyStartPly: base.historyStartPly,
    window: base.window.map((pair) => [...pair])
  };
  mutate(carried);
  try {
    verifyCarriedRecord(CONFIG, sample.features, sample.legalMask, carried);
    report(false, name, 'accepted a corrupted carried state');
  } catch (error) {
    const reason = error instanceof FeatureInversionError
      ? error.reason : (error.code ?? error.message);
    report(reason === wanted, name, `${reason}${reason === wanted ? '' : ` (wanted ${wanted})`}`);
  }
}

// The untouched sample must pass, or every refusal below is vacuous.
try {
  verifyCarriedRecord(CONFIG, sample.features, sample.legalMask, base);
  report(true, 'the untouched sample passes', `ply ${base.ply}, ${base.window.length} entries`);
} catch (error) {
  report(false, 'the untouched sample passes', error.message);
}

refuses('a ply that is not the true ply', (c) => { c.ply += 2; },
  'carried_window_sum_mismatch');
refuses('a window start past the ply', (c) => { c.historyStartPly = c.ply + 1; },
  'carried_history_start_ply_invalid');
refuses('a count that does not belong', (c) => { c.window[0][1] += 1; },
  'carried_window_sum_mismatch');
refuses('an empty window', (c) => { c.window = []; },
  'carried_window_empty');
refuses('a key with bits above the packing', (c) => { c.window[0][0] += 1 << 20; },
  'window_key_out_of_range');
refuses('a key naming a square off the board',
  (c) => { c.window[0][0] = (c.window[0][0] & ~0xff) | 0xfe; },
  'window_key_square_out_of_board');
refuses('a duplicated window entry',
  (c) => { c.window[1] = [c.window[0][0], c.window[1][1]]; },
  'carried_window_duplicate_key');

// A window that sums correctly and does not contain the root is the subtle one:
// every arithmetic identity still holds, and the state is simply about a
// different position. Built by moving the root's count onto another entry.
refuses('a window that has lost its own root', (c) => {
  const rootIndex = c.window.findIndex(([key]) => key === rootKeyOf(sample));
  const other = rootIndex === 0 ? 1 : 0;
  c.window[other][1] += c.window[rootIndex][1];
  c.window.splice(rootIndex, 1);
}, 'carried_window_missing_root');

/**
 * The packed key of a record's own position, derived from the record rather than
 * read out of its window -- otherwise the case above could not tell a window
 * that lost its root from one that never had it.
 */
function rootKeyOf(record) {
  // `historyStartPly === ply` means the window holds exactly the root, which is
  // the cheapest place to read it; otherwise find the entry whose position key
  // matches the rebuilt state's.
  const rebuilt = verifyCarriedRecord(CONFIG, record.features, record.legalMask, {
    ply: record.ply, historyStartPly: record.historyStartPly, window: record.window
  });
  const square = ({ r, c }) => r * CONFIG.columns + c;
  return square(rebuilt.position.pawns.A)
    | (square(rebuilt.position.pawns.B) << 8)
    | ((rebuilt.position.turn === 'B' ? 1 : 0) << 16);
}

console.log('');
if (FAILURES.length) {
  console.error(`FAIL (${FAILURES.length})`);
  for (const failure of FAILURES) console.error(`  ${failure}`);
  process.exit(1);
}
console.log(`record schema v2 verified: ${checked} records, ${byGame.size} games, `
  + `${repeatedPositions} on repeated positions`);
