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
 * THE RATIO RULE IS REFUTED ABOVE 512. Measured at 4096 simulations, against
 * width 24: width 12 is +88.7 Elo [28, 155], width 48 is -106.3 [-151, -65], and
 * all-legal is -60.3 [-115, -8]. The ratio predicted 192. So the optimum does not
 * keep growing with the budget -- it peaks and comes back down, and no
 * explanation for that is offered here because none has been measured.
 *
 * What is left is a table of what was actually observed, and nearest-measured
 * selection between the entries. That is uglier than a formula and honest about
 * where the evidence stops: 1024 and 2048 have never been measured, and they take
 * whichever neighbouring budget is nearer on a log scale.
 */

// Widths measured as best at each budget, each against the next-best width tried
// there. 128 -> 6 (24 was -129), 256 -> 12 (+58.5 over 6), 512 -> 24 (+154 over
// 6), 4096 -> 12 (+88.7 over 24).
export const MEASURED_WIDTHS = Object.freeze([
  Object.freeze({ sims: 128, width: 6 }),
  Object.freeze({ sims: 256, width: 12 }),
  Object.freeze({ sims: 512, width: 24 }),
  Object.freeze({ sims: 4096, width: 12 }),
]);

// Fewer than two candidates is not a choice, so the smallest budgets still
// compare a pair.
export const MIN_ROOT_WIDTH = 2;

export function rootWidth(simulations) {
  if (!Number.isFinite(simulations) || simulations <= 0) {
    throw new TypeError(`rootWidth needs a positive simulation count, got ${simulations}`);
  }
  // Nearest measured budget on a LOG scale: budgets are compared by ratio, not by
  // difference, so 1024 sits nearer 512 than 4096 and 2048 nearer 4096 than 512.
  let best = MEASURED_WIDTHS[0];
  let bestDistance = Infinity;
  for (const entry of MEASURED_WIDTHS) {
    const distance = Math.abs(Math.log(simulations) - Math.log(entry.sims));
    if (distance < bestDistance) { bestDistance = distance; best = entry; }
  }
  return Math.max(MIN_ROOT_WIDTH, best.width);
}
