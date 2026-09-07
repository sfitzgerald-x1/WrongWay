// Local reproduction of the cluster shard-parity failure, without the cluster.
//
// Runs BOTH self-play drivers over the same seeds, openings and options against
// the deterministic mock evaluator (so no ONNX runtime can confound the
// comparison), writes the shard lines each would emit, and reports the first
// ply at which the two RNG streams part.
//
//   A: the `cluster-selfplay.mjs` game loop, copied verbatim.
//   B: the Rust `SelfPlayBatch` over wasm, driven like `batched-selfplay.mjs`.
//
// Per ply it records the action chosen, the action played, the legal-move count
// and how many `rng()` draws that ply consumed, split into search draws and
// epsilon draws. A divergence in draw count is the signature of an RNG-stream
// fork; a divergence in ply count with matching draws is a termination-rule
// difference.
//
// SINCE `puct-az-tree-v2` THE TWO DRIVERS RECORD DIFFERENT POLICY TARGETS BY
// DESIGN: driver B records the Gumbel improved policy, driver A is the frozen
// JS reference and still records normalised visit counts. That is a deliberate
// format divergence, not the fork this script hunts. So `policyTarget` is
// compared SEPARATELY and does not affect the verdict.
//
// SINCE `puct-az-tree-v3` THE TWO DRIVERS ALSO PLAY DIFFERENT MOVES BY DESIGN.
// Driver B's sequential halving ranks survivors with the mctx qtransform;
// driver A is still frozen at `v1` and ranks them with the old raw-Q `sigma`.
// The two therefore visit different children and can choose differently at any
// ply -- and once they do, the games are at DIFFERENT POSITIONS and nothing
// after that point is comparable at all.
//
// Comparing whole games would therefore have pinned the exit code at 1 forever,
// which destroys the tool: its own target condition -- an RNG-stream fork --
// would no longer be distinguishable from its steady state, and a permanently
// red check is one nobody reads. Nor is the answer to make it green by
// asserting nothing.
//
// So the verdict is computed PER GAME OVER THE SHARED PREFIX: every ply up to
// and including the first one where the two drivers played different moves.
// Inside that prefix both drivers are at the same position with the same draw
// stream, so identical features, identical legal masks and an identical
// absolute-ply trace still mean exactly what they always meant, and a
// termination-rule difference or a stream fork still fails. The fork ply itself
// is reported per game, informational: under `v3` it is expected to be finite.
//
// `z` is compared only for games that never forked, because a game that forked
// can legitimately end differently. Where the two drivers agree on every played
// move, the comparison is the whole game exactly as before.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const eng = await import(`${ROOT}/js/normal-duel-engine.mjs`);
const { createInitialState, applyAction, decodeAction, legalActionCodes, encodeAction } = eng;
const { encodeState, legalMaskFloat, encodeLegalPolicyTarget } =
  await import(`${ROOT}/js/normal-duel-nn-encoding.mjs`);
const { createLcg32 } = await import(`${ROOT}/js/lcg32.mjs`);
const { puctSearch, effectiveVisitCounts } =
  await import(`${ROOT}/js/normal-duel-puct-search.mjs`);
const { mockEvaluator } = await import(`${ROOT}/js/normal-duel-mock-evaluator.mjs`);

const CONFIG = {
  ruleset: 'normal-duel-v1', rows: 9, columns: 9,
  start: { A: { r: 8, c: 4 }, B: { r: 0, c: 4 } },
  goalRows: { A: 0, B: 8 },
  initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3,
  plyCap: 200, firstPlayer: 'A'
};

const opts = JSON.parse(process.argv[2] ?? '{}');
const GAMES = opts.games ?? 2;
const SIMS = opts.simulations ?? 32;
const CONSIDERED = opts.maxConsidered ?? 8;
const C_PUCT = opts.cPuct ?? 1.4;
const EPSILON = opts.epsilon ?? 0;
const PLY_CAP = opts.plyCap ?? 60;
const SEED_BASE = opts.seedBase ?? 777;

// Stand-in for book200.json: deterministic opening lines of differing length,
// which is what exposes an absolute-ply vs recorded-ply cap disagreement.
function makeOpenings(count) {
  const openings = [];
  for (let g = 0; g < count; g += 1) {
    let state = createInitialState(CONFIG);
    const codes = [];
    const length = 4 + (g % 3);
    let pick = (g + 1) * 0x9e3779b1 >>> 0;
    for (let i = 0; i < length; i += 1) {
      const legal = legalActionCodes(CONFIG, state);
      pick = (Math.imul(pick, 1664525) + 1013904223) >>> 0;
      const code = legal[pick % legal.length];
      codes.push(code);
      state = applyAction(CONFIG, state, decodeAction(CONFIG, code));
    }
    openings.push(codes);
  }
  return openings;
}

const OPENINGS = opts.openings ?? makeOpenings(GAMES);
const evaluate = (config, state) => mockEvaluator(config, state);

/* ---------------------------------------------------------------- *
 * Driver A: the incumbent cluster-selfplay.mjs loop, verbatim.
 * ---------------------------------------------------------------- */
function runJs() {
  const lines = [];
  const positions = [];
  const targets = [];
  const trace = [];
  const outcomes = {};
  let plyTotal = 0;

  for (let g = 0; g < GAMES; g += 1) {
    const gameSeed = (SEED_BASE + g) >>> 0;
    const base = createLcg32(gameSeed);
    let draws = 0;
    const rng = () => { draws += 1; return base(); };

    let state = createInitialState(CONFIG);
    for (const code of OPENINGS[g]) state = applyAction(CONFIG, state, decodeAction(CONFIG, code));

    const pending = [];
    while (state.outcome.kind === 'ongoing' && state.ply < PLY_CAP) {
      const before = draws;
      const result = puctSearch({
        config: CONFIG, state, evaluate, simulations: SIMS,
        maxConsidered: CONSIDERED, cPuct: C_PUCT, random: rng
      });
      const expert = decodeAction(CONFIG, result.actionCode);
      const visitCounts = effectiveVisitCounts(result);
      const searchDraws = draws - before;

      const expertCode = encodeAction(CONFIG, expert);
      const legal = legalActionCodes(CONFIG, state);
      if (!legal.includes(expertCode)) throw new Error(`illegal expert move at ply ${state.ply}`);

      pending.push({
        features: Array.from(encodeState(CONFIG, state)),
        policyTarget: Array.from(encodeLegalPolicyTarget(
          CONFIG, state, visitCounts ?? new Map([[expertCode, 1]]))),
        legalMask: Array.from(legalMaskFloat(CONFIG, state)),
        turn: state.position.turn
      });

      const epsBefore = draws;
      let played = expert;
      if (EPSILON > 0 && (rng() % 10000) / 10000 < EPSILON) {
        played = decodeAction(CONFIG, legal[rng() % legal.length]);
      }
      trace.push({
        game: g, absPly: state.ply, recPly: pending.length - 1,
        legalCount: legal.length, searchDraws, epsilonDraws: draws - epsBefore,
        expertCode, playedCode: encodeAction(CONFIG, played)
      });
      state = applyAction(CONFIG, state, played);
      plyTotal += 1;
    }

    const o = state.outcome;
    outcomes[o.kind] = (outcomes[o.kind] ?? 0) + 1;
    for (const rec of pending) {
      const z = o.kind === 'win' ? (rec.turn === o.winner ? 1 : -1) : 0;
      lines.push(JSON.stringify({
        features: rec.features, legalMask: rec.legalMask, z
      }));
      // The position alone, without `z`: the outcome of a game that forked is
      // not a fact about either driver's correctness.
      positions.push(JSON.stringify({ features: rec.features, legalMask: rec.legalMask }));
      targets.push(JSON.stringify(rec.policyTarget));
    }
  }
  return { lines, positions, targets, trace, outcomes, plyTotal };
}

/* ---------------------------------------------------------------- *
 * Driver B: Rust SelfPlayBatch over wasm, mock evaluator.
 * ---------------------------------------------------------------- */
async function runRust() {
  const rel = `${ROOT}/rust/target/wasm-candidate/release`;
  const wasmMod = await import(`${rel}/normal-duel-wasm.mjs`);
  const instance = await wasmMod.default({
    module_or_path: readFileSync(`${rel}/normal-duel-wasm_bg.wasm`)
  });
  const memory = instance.memory ?? wasmMod.__wasm?.memory ?? wasmMod.memory;

  const { hashFeatures, mix32 } = await import(`${ROOT}/js/normal-duel-mock-evaluator.mjs`);
  const unit = (word) => (word >>> 8) / 16777216;

  // `exploration` must be explicit. It defaults to visit-temperature with
  // temperature_moves 0, so sending epsilon without it used to run pure argmax
  // with no exploration at all -- and this script would then have compared a JS
  // side that applies epsilon against a Rust side that silently does not, i.e.
  // two different algorithms, while reporting on their "parity". The Rust side
  // now rejects that combination outright; being explicit here keeps the two
  // arms comparing the same thing.
  const options = {
    games: GAMES, simulations: SIMS, maxConsidered: CONSIDERED, cPuct: C_PUCT,
    exploration: 'uniformEpsilon',
    epsilon: EPSILON, plyCap: PLY_CAP, seedBase: SEED_BASE >>> 0, openings: OPENINGS
  };
  const batch = new wasmMod.NormalDuelSelfPlayBatch(
    JSON.stringify(CONFIG), JSON.stringify(options)
  );
  const layout = JSON.parse(wasmMod.normalDuelSelfPlayLayout());
  const P = layout.policy;

  const view = (ptr, len) => new Float32Array(memory.buffer, ptr, len);
  for (;;) {
    const n = batch.collect();
    if (n === 0) break;
    const features = view(batch.featuresPtr(), batch.featuresLen());
    const scratch = new Float32Array(n * P);
    const values = new Float32Array(n);
    for (let slot = 0; slot < n; slot += 1) {
      const f = features.subarray(slot * layout.features, (slot + 1) * layout.features);
      const hash = hashFeatures(f);
      for (let code = 0; code < P; code += 1) {
        scratch[slot * P + code] = unit(mix32((hash ^ Math.imul(code, 0x9e3779b1)) >>> 0));
      }
      values[slot] = unit(mix32((hash ^ 0xdeadbeef) >>> 0)) * 2 - 1;
    }
    view(batch.policyPtr(), batch.policyLen()).set(scratch);
    view(batch.valuePtr(), batch.valueLen()).set(values);
    batch.submit(n);
  }

  const count = batch.takeRecords();
  const records = view(batch.recordsPtr(), batch.recordsLen()).slice();
  const meta = new Int32Array(memory.buffer, batch.recordMetaPtr(), batch.recordMetaLen()).slice();
  const outcomes = Array.from(batch.outcomes());
  const plies = Array.from(batch.pliesPlayed());

  const { features: F, recordFloats: R } = layout;
  const lines = [];
  const positions = [];
  const targets = [];
  const trace = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * R;
    const features = Array.from(records.subarray(base, base + F));
    const legalMask = Array.from(records.subarray(base + F + P, base + F + 2 * P));
    lines.push(JSON.stringify({ features, legalMask, z: records[base + F + 2 * P] }));
    positions.push(JSON.stringify({ features, legalMask }));
    targets.push(JSON.stringify(Array.from(records.subarray(base + F, base + F + P))));
    trace.push({
      game: meta[i * 4], absPly: meta[i * 4 + 1],
      turn: meta[i * 4 + 2], playedCode: meta[i * 4 + 3]
    });
  }
  return { lines, positions, targets, trace, outcomes, plies, count };
}

const js = runJs();
const rust = await runRust();

// Group each driver's records by game. Both drivers emit one record per ply in
// game order, so the record index and the trace index are the same index.
function byGame(trace) {
  const games = new Map();
  trace.forEach((entry, index) => {
    if (!games.has(entry.game)) games.set(entry.game, []);
    games.get(entry.game).push(index);
  });
  return games;
}

const jsGames = byGame(js.trace);
const rustGames = byGame(rust.trace);

// Walk each game's shared prefix: every ply up to and including the first one
// where the drivers played different moves. Inside it they are at the same
// position, so every disagreement is a real fault.
const faults = [];
const perGame = [];
for (const game of [...new Set([...jsGames.keys(), ...rustGames.keys()])].sort((l, r) => l - r)) {
  const a = jsGames.get(game) ?? [];
  const b = rustGames.get(game) ?? [];
  let forkPly = -1;
  const shared = Math.min(a.length, b.length);
  for (let i = 0; i < shared; i += 1) {
    const left = js.trace[a[i]];
    const right = rust.trace[b[i]];
    if (left.absPly !== right.absPly) {
      faults.push({ game, recPly: i, why: 'absPly', js: left, rust: right });
      break;
    }
    if (js.positions[a[i]] !== rust.positions[b[i]]) {
      faults.push({ game, recPly: i, why: 'position', js: left, rust: right });
      break;
    }
    if (left.playedCode !== right.playedCode) { forkPly = i; break; }
  }
  if (forkPly === -1) {
    // Never forked: the whole game must match, `z` and ply count included.
    if (a.length !== b.length) {
      faults.push({ game, recPly: shared, why: 'plyCount', js: a.length, rust: b.length });
    } else {
      for (let i = 0; i < a.length; i += 1) {
        if (js.lines[a[i]] !== rust.lines[b[i]]) {
          faults.push({ game, recPly: i, why: 'z', js: js.trace[a[i]], rust: rust.trace[b[i]] });
          break;
        }
      }
    }
  }
  perGame.push({ game, jsPlies: a.length, rustPlies: b.length, forkPly });
}

const perGameJs = {};
for (const t of js.trace) perGameJs[t.game] = (perGameJs[t.game] ?? 0) + 1;
const perGameRust = {};
for (const t of rust.trace) perGameRust[t.game] = (perGameRust[t.game] ?? 0) + 1;

const report = {
  options: { games: GAMES, sims: SIMS, epsilon: EPSILON, plyCap: PLY_CAP, seedBase: SEED_BASE },
  openingLengths: OPENINGS.map((o) => o.length),
  js: { records: js.lines.length, outcomes: js.outcomes, perGame: perGameJs },
  rust: { records: rust.lines.length, outcomes: rust.outcomes, plies: rust.plies, perGame: perGameRust },
  // The verdict deliberately excludes `policyTarget` and everything after a
  // game's fork ply (see the header), and deliberately includes the absolute
  // ply of every shared position: a stream fork that produced the same number
  // of identical-looking records but a different ply trace is exactly the
  // failure this script exists to catch.
  identical: faults.length === 0,
  faults,
  // Per game: how far the two drivers agreed on the played move. `forkPly: -1`
  // means they agreed for the whole game. Expected to be finite under
  // `puct-az-tree-v3` and NOT a fault -- see the header.
  perGameFork: perGame,
  forkedGames: perGame.filter((entry) => entry.forkPly >= 0).length,
  // Informational: expected to be 0 since `puct-az-tree-v2`, because the two
  // drivers record different targets on purpose. A -1 here would mean the JS
  // reference had been changed to match, which is a decision, not an accident.
  firstDifferingTarget: (() => {
    for (let i = 0; i < Math.max(js.targets.length, rust.targets.length); i += 1) {
      if (js.targets[i] !== rust.targets[i]) return i;
    }
    return -1;
  })(),
  jsTailTrace: faults.length ? js.trace.slice(0, 8) : [],
  rustTailTrace: faults.length ? rust.trace.slice(0, 8) : []
};
console.log(JSON.stringify(report, null, 2));

// Exit non-zero on divergence. A parity check that only prints is not a check:
// nothing in CI or a script wrapper notices, which is how this file could have
// been comparing two different algorithms without anyone being told.
if (!report.identical) process.exitCode = 1;
