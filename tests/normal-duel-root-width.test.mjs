/**
 * The root width the search should use for a given simulation budget.
 *
 * These are not arbitrary constants: each of the three pinned below is a MEASURED
 * optimum from a seat-swapped paired match against the incumbent width 6 at that
 * budget. If someone retunes the ratio, these break, and the message says what
 * evidence they would be contradicting.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MIN_ROOT_WIDTH,
  SIMS_PER_CONSIDERED_CHILD,
  rootWidth
} from '../js/normal-duel-root-width.mjs';

test('reproduces the three measured optima', () => {
  assert.equal(rootWidth(128), 6,  'width 6 won at 128 sims; 24 was -129 Elo [-180,-82] there');
  assert.equal(rootWidth(256), 12, 'width 12 won at 256 sims: +58 Elo [8,112] over 6, and 24 was -59');
  assert.equal(rootWidth(512), 24, 'width 24 won at 512 sims: +154 Elo [76,250] over 6');
});

test('the optima all sit at the same simulations-per-child ratio', () => {
  // This is the whole content of the rule -- if these three stop agreeing, the
  // rule is no longer what the measurements support.
  for (const [sims, width] of [[128, 6], [256, 12], [512, 24]]) {
    assert.ok(Math.abs(sims / width - SIMS_PER_CONSIDERED_CHILD) < 0.1,
      `${sims}/${width} = ${(sims / width).toFixed(2)}, not ${SIMS_PER_CONSIDERED_CHILD}`);
  }
});

test('never asks for fewer than two candidates', () => {
  // The slider's floor is 16 sims, which the bare ratio would turn into 1 -- and
  // a search that "considers" one move is not choosing, it is echoing the prior.
  // The engine also rejects max_considered < 1 outright (puct.rs).
  assert.equal(rootWidth(16), MIN_ROOT_WIDTH);
  assert.equal(rootWidth(1), MIN_ROOT_WIDTH);
  assert.ok(MIN_ROOT_WIDTH >= 2);
});

test('grows monotonically with the budget across the slider range', () => {
  let prev = 0;
  for (let sims = 16; sims <= 4096; sims += 16) {
    const w = rootWidth(sims);
    assert.ok(w >= prev, `width fell from ${prev} to ${w} at ${sims} sims`);
    prev = w;
  }
});

test('rejects a budget that is not a positive count', () => {
  for (const bad of [0, -1, NaN, Infinity, undefined, null, '256']) {
    assert.throws(() => rootWidth(bad), TypeError, `should reject ${String(bad)}`);
  }
});
