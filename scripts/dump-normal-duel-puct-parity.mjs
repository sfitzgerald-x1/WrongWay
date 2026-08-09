#!/usr/bin/env node
/**
 * Reference side of the Rust PUCT parity harness.
 *
 * Reads `{ config, states, cases }` and runs the shipped
 * `js/normal-duel-puct-search.mjs` over every (state, case) pair with the
 * shared deterministic mock evaluator and a fresh `createLcg32(case.seed)`,
 * writing what the Rust test compares against:
 *
 *   { "version": PUCT_SEARCH_VERSION,
 *     "results": [ { actionCode, visitCounts, rootValueBits, simulationsUsed,
 *                    maxDepthReached, considered, allScoreBits } ] }
 *
 * `version` is the reference's own `PUCT_SEARCH_VERSION`. The Rust side has
 * moved to `puct-az-tree-v3` (the mctx qtransform, in the halving ranking as
 * well as the recorded target) while this reference stays frozen at `v1`; the
 * Rust test asserts this string against `JS_REFERENCE_SEARCH_VERSION` so that
 * freeze is a checked fact rather than a comment. Which fields the two engines
 * are still required to agree on exactly, and which now diverge by design, is
 * decided in `rust/normal-duel-core/tests/js_puct_parity.rs`, not here: this
 * script dumps what the reference did and states nothing about what should
 * match.
 *
 * Doubles cross as their IEEE-754 bit patterns in hex, never as JSON numbers.
 * Decimal is a lossy channel for this comparison in practice: serde_json 1.0.151
 * parses `0.48560023307800293` one ULP high, so a decimal round trip would fail
 * a search that is in fact exact — and, worse, could mask one that is not.
 *
 * `allScoreBits` is `g + logit` for EVERY legal root action, in ascending code
 * order. Nothing in the contract depends on it; it is dumped so the Rust test
 * can measure how much slack the Gumbel ranking has, which is the one place a
 * 1-ULP `Math.log` disagreement could ever matter. See
 * `rust/normal-duel-core/src/js_math.rs`.
 *
 * It covers every action rather than only the considered ones because the
 * margin that actually protects the search is the gap AT THE CUT — between the
 * worst kept candidate and the best discarded one — and a dump restricted to
 * the considered set cannot see it. At `maxConsidered = 1` it could not see any
 * gap at all, and reported an empty ranking as infinite slack.
 *
 * Nothing here reimplements a rule, an encoding or a search.
 */

import { readFileSync, writeFileSync } from 'node:fs';

import { createLcg32 } from '../js/lcg32.mjs';
import { PUCT_SEARCH_VERSION, puctSearch } from '../js/normal-duel-puct-search.mjs';
import { mockEvaluator } from '../js/normal-duel-mock-evaluator.mjs';

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('usage: dump-normal-duel-puct-parity.mjs <input.json> <results.json>');
  process.exit(2);
}

const { config, states, cases } = JSON.parse(readFileSync(inputPath, 'utf8'));

const doubleView = new DataView(new ArrayBuffer(8));

/** A double's IEEE-754 bit pattern as 16 hex digits. */
function doubleBits(value) {
  doubleView.setFloat64(0, value);
  return doubleView.getBigUint64(0).toString(16).padStart(16, '0');
}

/**
 * Re-derive every legal root action's Gumbel score by replaying the draw stream
 * the search consumed. `puctSearch` does not expose the scores, and
 * instrumenting it would mean editing the module under test.
 */
function candidateScores(seed, priorsByCode) {
  const random = createLcg32(seed);
  const draws = new Map();
  for (const code of priorsByCode.keys()) {
    const raw = random();
    let u = (raw + 0.5) / 4294967296;
    if (!(u > 0)) u = Number.MIN_VALUE;
    if (!(u < 1)) u = 1 - Number.EPSILON / 2;
    draws.set(code, -Math.log(-Math.log(u)));
  }
  return [...priorsByCode.keys()].map(
    (code) => draws.get(code) + Math.log(Math.max(priorsByCode.get(code), 1e-9))
  );
}

/** The root priors, recomputed the way `expand` does: masked, renormalised. */
function rootPriors(state, legal) {
  const { policy } = mockEvaluator(config, state);
  let mass = 0;
  for (const code of legal) mass += policy[code];
  const priors = new Map();
  for (const code of legal) priors.set(code, mass > 0 ? policy[code] / mass : 1 / legal.length);
  return priors;
}

const { legalActionCodes } = await import('../js/normal-duel-engine.mjs');

const results = [];
for (const state of states) {
  for (const testCase of cases) {
    const result = puctSearch({
      config,
      state,
      evaluate: mockEvaluator,
      simulations: testCase.simulations,
      cPuct: testCase.cPuct,
      maxConsidered: testCase.maxConsidered,
      random: createLcg32(testCase.seed)
    });
    const priors = rootPriors(state, legalActionCodes(config, state));
    results.push({
      actionCode: result.actionCode,
      visitCounts: [...result.visitCounts.entries()].sort((left, right) => left[0] - right[0]),
      rootValueBits: doubleBits(result.rootValue),
      simulationsUsed: result.simulationsUsed,
      maxDepthReached: result.maxDepthReached,
      considered: [...result.considered],
      allScoreBits: candidateScores(testCase.seed, priors).map(doubleBits)
    });
  }
}

writeFileSync(outputPath, JSON.stringify({ version: PUCT_SEARCH_VERSION, results }));
