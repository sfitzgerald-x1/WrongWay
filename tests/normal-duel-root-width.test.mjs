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

import { MEASURED_WIDTHS, MIN_ROOT_WIDTH, rootWidth } from '../js/normal-duel-root-width.mjs';

test('reproduces the three measured optima', () => {
  assert.equal(rootWidth(128), 6,  'width 6 won at 128 sims; 24 was -129 Elo [-180,-82] there');
  assert.equal(rootWidth(256), 12, 'width 12 won at 256 sims: +58 Elo [8,112] over 6, and 24 was -59');
  assert.equal(rootWidth(512), 24, 'width 24 won at 512 sims: +154 Elo [76,250] over 6');
});

test('4096 takes the width measured best THERE, not one extrapolated from below', () => {
  // The ratio rule that fitted 128/256/512 predicted 192 here. Measured against
  // width 24 at 4096: width 12 is +88.7 Elo [28, 155], width 48 is -106.3, and
  // all-legal is -60.3. Extrapolating the rule would have cost roughly 150 Elo.
  assert.equal(rootWidth(4096), 12);
});

test('the optimum is NOT monotonic in the budget', () => {
  // It rises to 24 at 512 and comes back to 12 by 4096. Nothing here explains
  // why -- 1024 and 2048 have never been measured -- but a formula that assumed
  // monotonicity is exactly what produced the 192 above.
  assert.ok(rootWidth(512) > rootWidth(256), 'rises to 512');
  assert.ok(rootWidth(4096) < rootWidth(512), 'and falls again by 4096');
});

test('every measured budget returns its own measured width', () => {
  for (const { sims, width } of MEASURED_WIDTHS) {
    assert.equal(rootWidth(sims), width, `${sims} sims`);
  }
});

test('never asks for fewer than two candidates', () => {
  // The floor still exists but is no longer reached: the narrowest measured width
  // is 6, so a tiny budget now takes 128's width rather than being clamped. A
  // search that "considers" one move is not choosing, it is echoing the prior, and
  // the engine rejects max_considered < 1 outright.
  for (const sims of [1, 16, 32, 64, 128, 512, 4096, 100_000]) {
    assert.ok(rootWidth(sims) >= MIN_ROOT_WIDTH, `${sims} sims returned too few`);
  }
  assert.ok(MIN_ROOT_WIDTH >= 2);
});

test('unmeasured budgets take the nearer measured one on a log scale', () => {
  // Budgets compare by ratio, not difference: 1024 is nearer 512 than 4096, and
  // 2048 is nearer 4096 than 512. Both are interpolations and neither has been
  // measured -- this pins the choice so it is visible rather than incidental.
  assert.equal(rootWidth(1024), rootWidth(512));
  assert.equal(rootWidth(2048), rootWidth(4096));
});

test('rejects a budget that is not a positive count', () => {
  for (const bad of [0, -1, NaN, Infinity, undefined, null, '256']) {
    assert.throws(() => rootWidth(bad), TypeError, `should reject ${String(bad)}`);
  }
});
