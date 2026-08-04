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
        features: rec.features, policyTarget: rec.policyTarget, legalMask: rec.legalMask, z
      }));
    }
  }
  return { lines, trace, outcomes, plyTotal };
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
  const trace = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * R;
    lines.push(JSON.stringify({
      features: Array.from(records.subarray(base, base + F)),
      policyTarget: Array.from(records.subarray(base + F, base + F + P)),
      legalMask: Array.from(records.subarray(base + F + P, base + F + 2 * P)),
      z: records[base + F + 2 * P]
    }));
    trace.push({
      game: meta[i * 4], absPly: meta[i * 4 + 1],
      turn: meta[i * 4 + 2], playedCode: meta[i * 4 + 3]
    });
  }
  return { lines, trace, outcomes, plies, count };
}

const js = runJs();
const rust = await runRust();

// Locate the first differing shard line, and the first differing ply.
let firstLine = -1;
for (let i = 0; i < Math.max(js.lines.length, rust.lines.length); i += 1) {
  if (js.lines[i] !== rust.lines[i]) { firstLine = i; break; }
}
let firstPly = -1;
for (let i = 0; i < Math.max(js.trace.length, rust.trace.length); i += 1) {
  const a = js.trace[i];
  const b = rust.trace[i];
  if (!a || !b || a.game !== b.game || a.absPly !== b.absPly || a.playedCode !== b.playedCode) {
    firstPly = i; break;
  }
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
  identical: firstLine === -1 && js.lines.length === rust.lines.length,
  firstDifferingLine: firstLine,
  firstDifferingPly: firstPly,
  jsPlyAt: firstPly >= 0 ? js.trace[firstPly] : null,
  rustPlyAt: firstPly >= 0 ? rust.trace[firstPly] : null,
  jsTailTrace: js.trace.slice(Math.max(0, firstPly - 2), firstPly + 3),
  rustTailTrace: rust.trace.slice(Math.max(0, firstPly - 2), firstPly + 3)
};
console.log(JSON.stringify(report, null, 2));

// Exit non-zero on divergence. A parity check that only prints is not a check:
// nothing in CI or a script wrapper notices, which is how this file could have
// been comparing two different algorithms without anyone being told.
if (!report.identical) process.exitCode = 1;
