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

/**
 * `improvedPolicy()` crosses the boundary, under that name, in that shape.
 *
 * What it IS is proved natively, in `rust/normal-duel-wasm/src/lib.rs`: the
 * accessor hands back exactly the numbers a real self-play game writes as
 * `policyTarget`. What a native test cannot see is the crossing — the
 * `js_name`, and a `Vec<f64>` arriving as one array whose even slots are codes
 * and odd slots are probabilities. That flat pairing is the whole read
 * protocol: a consumer walks it `i += 2` and scatters `flat[i + 1]` into its
 * own zeroed policy vector at `flat[i]`. So this asserts the three things that
 * loop rests on, against the built wasm.
 *
 * The search is driven here rather than through `wasmPuctSearch` on purpose,
 * and the reason is a name collision worth stating. That driver's return shape
 * is the parity surface compared against the frozen JavaScript reference, and
 * `puctSearch` already returns a field called `improvedPolicy` — but it is
 * `encodePolicyTarget(config, visitCounts)`, normalised visits over the
 * considered set, which is the pre-`puct-az-tree-v2` target and a different
 * distribution from this one on a different support. Putting the Rust accessor
 * into the parity surface next to it would set up a comparison that must fail,
 * and the tempting repair for that failure is to make one of the two agree with
 * the other.
 */
test('NormalDuelSearch hands improvedPolicy across as ascending (code, probability) pairs',
  { skip: SKIP }, async () => {
    const { mod, memory } = await loadWasmPuct(RELEASE);
    assert.equal(
      typeof mod.NormalDuelSearch.prototype.improvedPolicy, 'function',
      'the accessor must be on the prototype: a consumer checks for it there before it runs'
    );
    // The Rust side returns `Result<Vec<f64>, JsValue>`, which wasm-bindgen
    // surfaces as a throw and NOT as a second return value or an out-parameter.
    // Pinned because the guard was added after the accessor shipped: a consumer
    // that probes for the method and then calls it with no arguments must see
    // exactly what it saw before.
    assert.equal(mod.NormalDuelSearch.prototype.improvedPolicy.length, 0, 'still nullary');

    const state = createInitialState(CONFIG);
    const legal = legalActionCodes(CONFIG, state);
    const maxConsidered = 8;
    const search = new mod.NormalDuelSearch(
      JSON.stringify(CONFIG),
      JSON.stringify(state),
      JSON.stringify({ simulations: 48, maxConsidered, seed: 20260818 })
    );
    // Before a single leaf: the tree has nothing to say and says so, rather
    // than handing back the empty array that would scatter to an all-zero row.
    assert.throws(() => search.improvedPolicy(), /search_not_done/,
      'an unstarted search must refuse');

    // Views rebuilt every iteration, as everywhere else: `nextLeaf` grows the
    // arenas and a growth detaches them.
    const f32 = (ptr, len) => new Float32Array(memory.buffer, ptr, len);
    let leaves = 0;
    while (search.nextLeaf()) {
      search.pendingLeafMask();
      const policyLen = search.policyLen();
      search.submit(mockEvaluateFeatures(
        f32(search.featuresPtr(), search.featuresLen()),
        f32(search.maskPtr(), policyLen),
        f32(search.policyPtr(), policyLen)
      ));
      leaves += 1;
      // Mid-search, where the unguarded read is a full, plausible, WRONG
      // distribution. The refusal has to cross the boundary as a throw, or the
      // guard only exists in Rust.
      if (leaves === 10) {
        assert.throws(() => search.improvedPolicy(), /search_not_done/,
          'a partial tree must refuse across the boundary too');
      }
    }

    const flat = search.improvedPolicy();
    assert.ok(flat instanceof Float64Array, 'the happy path is still one Float64Array');
    assert.equal(flat.length % 2, 0, 'flattened pairs have an even length');
    const codes = [];
    let total = 0;
    for (let index = 0; index < flat.length; index += 2) {
      codes.push(flat[index]);
      total += flat[index + 1];
      assert.ok(flat[index + 1] > 0, `code ${flat[index]} came across with no mass`);
    }
    assert.deepEqual(codes, legal, 'support is exactly the legal root actions, ascending by code');
    assert.ok(Math.abs(total - 1) < 1e-12, `pi' must sum to 1, got ${total}`);

    // And the reason the accessor exists rather than a normalisation of the
    // other one: `visitCounts` spans the CONSIDERED set, 8 of the root's 131
    // legal actions, and its codes are a strict subset of these.
    const visits = search.visitCounts();
    assert.equal(visits.length / 2, maxConsidered);
    for (let index = 0; index < visits.length; index += 2) {
      assert.ok(codes.includes(visits[index]), `considered code ${visits[index]} is not legal`);
    }
    assert.ok(codes.length > visits.length / 2, 'pi\' covers more than the considered set');
  });
