/**
 * The `NormalDuelSearch` wrapper returns the move `puctSearch` returns.
 *
 * Scope, deliberately narrow
 * --------------------------
 * `rust/normal-duel-core/tests/js_puct_parity.rs` already proves the two TREES
 * are the same search: 220 states x 6 configurations, identical visit counts,
 * identical action code, bit-identical root value. That is not re-derived here.
 *
 * What is unproven until this file runs is the PASSTHROUGH: the wasm wrapper
 * parses the config/state/options JSON, hands features out through a pointer,
 * takes probabilities back through another, and reports an action code. A
 * transposed buffer, an options field silently defaulted, or a mask read at the
 * wrong length would all leave the proven tree intact and still play a
 * different move. So this test drives the wrapper end to end and compares the
 * move — plus the visit counts and root value, which localise a failure to the
 * tree rather than the plumbing.
 *
 * The evaluator is the shared mock (`js/normal-duel-mock-evaluator.mjs` /
 * `rust/normal-duel-core/src/mock_evaluator.rs`), so no network, GPU or ONNX
 * runtime is involved and every number is exact in f32 and f64 alike. It is
 * applied to the features the WASM SIDE produced, which means an encoder
 * disagreement also fails here instead of being averaged away.
 */
import { strict as assert } from 'node:assert';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import {
  applyLegalAction, createInitialState, decodeAction, legalActionCodes
} from '../js/normal-duel-engine.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import { hashFeatures, mix32, mockEvaluator } from '../js/normal-duel-mock-evaluator.mjs';
import { puctSearch } from '../js/normal-duel-puct-search.mjs';
import { loadWasmPuct, wasmPuctSearch } from '../js/normal-duel-wasm-puct-search.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RELEASE = join(ROOT, 'rust', 'target', 'wasm-candidate', 'release');

// The candidate is a BUILD ARTIFACT under rust/target, so it is absent on a
// clean checkout and in the CI job that only runs `npm test`. Skipping is right
// here rather than failing: the wasm is exercised by its own workflow job, which
// builds it first. The skip is explicit and names the command, so a local run
// that meant to cover this cannot mistake it for a pass.
const HAVE_CANDIDATE = existsSync(join(RELEASE, 'manifest.json'))
  && existsSync(join(RELEASE, 'normal-duel-wasm_bg.wasm'));
const SKIP = HAVE_CANDIDATE
  ? false
  : 'wasm candidate not built (node scripts/build-normal-duel-wasm-candidate.mjs)';

const CONFIG = Object.freeze({
  ruleset: 'normal-duel-v1',
  rows: 9,
  columns: 9,
  start: { A: { r: 8, c: 4 }, B: { r: 0, c: 4 } },
  goalRows: { A: 0, B: 8 },
  initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1',
  repetitionThreshold: 3,
  plyCap: 200,
  firstPlayer: 'A'
});

/** The mock evaluator, applied to the features the wasm tree handed out. */
const unit = (word) => (word >>> 8) / 16777216;
function mockEvaluateFeatures(features, _mask, policyOut) {
  const hash = hashFeatures(features);
  for (let code = 0; code < policyOut.length; code += 1) {
    policyOut[code] = unit(mix32((hash ^ Math.imul(code, 0x9e3779b1)) >>> 0));
  }
  return unit(mix32((hash ^ 0xdeadbeef) >>> 0)) * 2 - 1;
}

/** A spread of reachable states: walk one pseudo-random legal line. */
function statesAlongALine(count, seed) {
  const random = createLcg32(seed);
  const states = [];
  let state = createInitialState(CONFIG);
  while (states.length < count) {
    if (state.outcome.kind !== 'ongoing') break;
    states.push(state);
    const legal = legalActionCodes(CONFIG, state);
    const code = legal[random() % legal.length];
    state = applyLegalAction(CONFIG, state, decodeAction(CONFIG, code));
  }
  return states;
}

// Small, mid and lopsided budgets: sequential halving takes different paths
// through the rounds loop at each, and `maxConsidered` above the legal count is
// the clamp the wrapper could get wrong on its own.
const BUDGETS = Object.freeze([
  { simulations: 16, maxConsidered: 4 },
  { simulations: 48, maxConsidered: 8 },
  { simulations: 7, maxConsidered: 16 },
  { simulations: 64, maxConsidered: 1 }
]);

test('the NormalDuelSearch wrapper plays puctSearch\'s move', { skip: SKIP }, async () => {
  assert.ok(
    readFileSync(join(RELEASE, 'manifest.json'), 'utf8').length > 0,
    'build the candidate first: node scripts/build-normal-duel-wasm-candidate.mjs'
  );
  const handle = await loadWasmPuct(RELEASE);
  const states = statesAlongALine(12, 20260803);
  assert.equal(states.length, 12);

  let compared = 0;
  for (const [index, state] of states.entries()) {
    for (const budget of BUDGETS) {
      // One seed, two searches. `puctSearch` draws its Gumbels from a fresh
      // `createLcg32(seed)`; the wrapper is constructed with the same seed and
      // seeds `Lcg32::new` from it, so the root draw order is the same stream.
      const seed = (1_000_003 + index * 7919 + budget.simulations * 104_729) >>> 0;
      const js = puctSearch({
        config: CONFIG,
        state,
        evaluate: mockEvaluator,
        simulations: budget.simulations,
        maxConsidered: budget.maxConsidered,
        random: createLcg32(seed)
      });
      const rust = wasmPuctSearch({
        handle,
        config: CONFIG,
        state,
        evaluateFeatures: mockEvaluateFeatures,
        simulations: budget.simulations,
        maxConsidered: budget.maxConsidered,
        seed
      });

      const where = `state ${index} at ${budget.simulations}/${budget.maxConsidered}`;
      // The move is what the match actually plays, so it is asserted first.
      assert.equal(rust.actionCode, js.actionCode, `actionCode: ${where}`);
      assert.ok(
        legalActionCodes(CONFIG, state).includes(rust.actionCode),
        `wrapper returned an illegal action: ${where}`
      );
      assert.equal(rust.simulationsUsed, js.simulationsUsed, `simulationsUsed: ${where}`);
      // Bit-identical, not approximate: the mock's values are exact in f32.
      assert.equal(rust.rootValue, js.rootValue, `rootValue: ${where}`);
      assert.deepEqual(
        [...rust.visitCounts.entries()].sort((l, r) => l[0] - r[0]),
        [...js.visitCounts.entries()].sort((l, r) => l[0] - r[0]),
        `visitCounts: ${where}`
      );
      compared += 1;
    }
  }
  assert.equal(compared, 48, 'every state x budget pair must have been compared');
});
