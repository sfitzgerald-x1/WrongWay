/**
 * How many root moves the search should consider, given its simulation budget.
 *
 * Gumbel AlphaZero spends its budget across `maxConsidered` root candidates by
 * sequential halving, so width and depth trade directly against each other: every
 * extra candidate is visits taken from the candidates already there. The right
 * width is therefore a function of the budget, and a constant is wrong everywhere
 * except the one budget it happened to be picked for.
 *
 * Measured on pool-iter-085 with the native tree, seat-swapped paired matches,
 * each width played against the incumbent width 6 at its own budget:
 *
 *   sims   width   sims/child   result vs width 6
 *    128     24        5.3      -129 Elo  [-180, -82]
 *    256     12       21.3       +58 Elo  [   8,  112]
 *    256     24       10.7       -59 Elo  [-147,   23]
 *    512     24       21.3      +154 Elo  [  76,  250]
 *    512     68        7.5      +104 Elo  [  34,  184]
 *
 * Three budgets, three different winners -- 6 at 128, 12 at 256, 24 at 512 -- and
 * all three sit at ~21 simulations per considered child. Every measured loser is
 * off that ratio, on one side or the other: 512 at width 6 is 85 sims/child and
 * too narrow, 512 at all-legal is 7.5 and too wide. The peak is interior, and it
 * moves with the budget.
 *
 * CALIBRATION RANGE: 128 to 512 simulations. Below and above that this is
 * extrapolation from three points. It is a reasonable extrapolation -- the ratio
 * is the thing the halving schedule actually cares about -- but it has not been
 * measured there, and the slider reaches 4096.
 */

// Simulations per considered root child, at the three measured optima.
// 128/6 = 21.3, 256/12 = 21.3, 512/24 = 21.3.
export const SIMS_PER_CONSIDERED_CHILD = 21.3;

// Fewer than two candidates is not a choice, so the smallest budgets on the
// slider (16 sims would ask for 1) still compare a pair.
export const MIN_ROOT_WIDTH = 2;

export function rootWidth(simulations) {
  if (!Number.isFinite(simulations) || simulations <= 0) {
    throw new TypeError(`rootWidth needs a positive simulation count, got ${simulations}`);
  }
  return Math.max(MIN_ROOT_WIDTH, Math.round(simulations / SIMS_PER_CONSIDERED_CHILD));
}
