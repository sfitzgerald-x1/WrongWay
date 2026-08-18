/**
 * Head-to-head between TWO NETWORKS at the same search budget.
 *
 * WHY THIS EXISTS RATHER THAN `sims-match.mjs`. That script pits two BUDGETS of one
 * network and takes no per-side checkpoint, so it cannot express "same search, two
 * students" -- which is exactly the Bet 2 screen: two nets trained from one
 * checkpoint on one corpus, differing only in whether their policy targets came
 * from a 512-sim or a 4096-sim search.
 *
 * WHY IT DOES NOT REUSE `runLockStep`. That function gets its throughput by putting
 * every live search's pending leaf into ONE inference request. Leaves belonging to
 * DIFFERENT checkpoints cannot share a forward -- the server picks one set of
 * weights per request -- so a two-net match needs the batch split by side. Rather
 * than thread a checkpoint selector through the shared function that
 * `distill-collect` and `distill-relabel` both depend on, the loop is duplicated
 * here with that one difference. The corpus pipeline is working and is not worth
 * risking for code reuse.
 *
 * The split costs nothing real: at N slots each step issues two requests of about
 * N/2 leaves instead of one of N. At 64 slots that is ~32 per request, still far
 * above the batch-1 floor of 343 positions/s and near the 8,826/s measured at 32.
 *
 * COLOURS ALTERNATE and the seed varies per game AND per ply. Both matter: with two
 * deterministic players a fixed seed replays one game N times, and a colour bias
 * would measure who moves first. `sims-match.mjs` learned both the hard way.
 *
 *   node scripts/distill-compare.mjs --a shallow-512 --b deep-4096 --games 300 \
 *     --sims 512 --considered 6 --slots 64 --infer http://127.0.0.1:8304
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import { applyAction, createInitialState, decodeAction } from '../js/normal-duel-engine.mjs';
import {
  CONFIG, CONFIG_JSON, FEATURE_LEN, POLICY_LEN,
  createInferClient, fmtDuration, historyEntry, loadWasm, parseArgs
} from './distill-lib.mjs';
import { createNetGuard } from './net-guard.mjs';

const args = parseArgs(process.argv.slice(2), {
  a: 'shallow-512',
  b: 'deep-4096',
  games: 300,
  sims: 512,
  considered: 6,
  cpuct: 1.25,
  slots: 64,
  seed: 1,
  infer: 'http://127.0.0.1:8304',
  // The parent checkpoint every student must be measured against. A run where `b`
  // (or `a`) IS this id is an ANCHOR run and is recorded; any other pairing is a
  // student-vs-student screen and is gated on both sides having an anchor.
  base: 'base',
  // One-time assertion that `--base` really is a parent checkpoint. Parenthood cannot
  // be inferred: in a lineage every net is both a student of the previous one and the
  // parent of the next, so ledger membership answers the wrong question.
  'declare-parent': false,
  ledger: path.join(process.env.HOME || '/tmp', 'wwplay/distill/anchors.json')
});

// Reserved top-level ledger key holding declared parents. Student entries live at the
// top level beside it, so an id colliding with this key is refused rather than silently
// merged into the declarations.
const PARENTS_KEY = '__parents__';
for (const [flag, id] of [['--a', args.a], ['--b', args.b], ['--base', args.base]]) {
  if (id === PARENTS_KEY) {
    console.error(`REFUSED: ${flag}=${PARENTS_KEY} collides with the ledger's reserved key.`);
    process.exit(2);
  }
}

// ------------------------------------------------------- R0 guards
/**
 * The pool's historical Elo spread, used by the effect-size tripwire.
 *
 * FULL is every member ever fitted, from the classical `hard` anchor at -362 to the
 * best net at +239: ~600 points covering the entire history of the project. HEALTHY is
 * the band the trained nets actually occupy (~145).
 *
 * Why this exists: on 2026-08-16 a screen reported +551 Elo from one fine-tuning step.
 * That is larger than the whole pool range, i.e. it claimed a single label change was
 * worth more than every arm ever run put together. It was believed for an hour. The
 * number was in the same session's Elo table and did not register, so the check is
 * automatic here rather than left to judgement.
 */
/**
 * Read a positive finite number from the environment, or refuse.
 *
 * `Number(process.env.X ?? d)` was the first cut and it is a hole in the tripwire: the
 * docstring below says "2x the healthy spread", so `WW_BREAKAGE_ELO=2x145` is a
 * plausible typo, and it yields NaN. Every `mag > NaN` is false, so the SUSPECTED
 * BREAKAGE branch becomes unreachable and the 552-Elo incident falls through to the
 * mild "verify the loser" note -- the exact outcome this threshold was chosen to
 * avoid. An empty value yields 0 and fires on every run, including a dead-even 50%.
 * A guard whose threshold can be silently turned into NaN is not a guard.
 */
function envNumber(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    console.error(`REFUSED: ${name}=${JSON.stringify(raw)} is not a positive finite number.`);
    process.exit(2);
  }
  return value;
}

const POOL_SPREAD_FULL = envNumber('WW_POOL_SPREAD_FULL', 600);
const POOL_SPREAD_HEALTHY = envNumber('WW_POOL_SPREAD_HEALTHY', 145);
/**
 * The breakage threshold, and why it is NOT `POOL_SPREAD_FULL`.
 *
 * The incident this guard exists for measured 96.0%, which is 552 Elo -- BELOW the ~600
 * full-pool spread. Checking against 600 alone would have let the exact case that
 * motivated the guard through with only a mild warning. Caught while testing the guard
 * against its own design case, which is the reason to test guards against real numbers
 * rather than plausible ones.
 *
 * 2x the healthy spread is the line instead. For calibration, the largest legitimate
 * single-intervention gain this project has recorded is g2's ~+76 over its seed, and the
 * healthy nets span ~145 end to end. A single label or target change producing 290+ is
 * outside anything the loop has ever done honestly.
 */
const BREAKAGE_ELO = envNumber('WW_BREAKAGE_ELO', 2 * POOL_SPREAD_HEALTHY);
// The EFFECTIVE thresholds, once, in the run's own output. A valid override is
// otherwise invisible: a screen reported months later cannot be checked against the
// threshold it actually ran under, which is the same "reads on while configured
// differently" gap the env validation above exists to close.
console.log(`tripwire: breakage ${BREAKAGE_ELO}, healthy ${POOL_SPREAD_HEALTHY},`
  + ` full ${POOL_SPREAD_FULL}`);

/**
 * A missing ledger is an empty ledger; a MALFORMED one is not.
 *
 * `JSON.parse` of `null` succeeds and returns null, and `{"id": null}` parses fine too,
 * so the first cut turned a corrupt ledger into a TypeError deep inside the anchor gate
 * -- fail-closed by accident, with no REFUSED line saying why.
 */
function readLedger(file) {
  let text;
  try { text = readFileSync(file, 'utf8'); } catch { return {}; }
  let parsed;
  try { parsed = JSON.parse(text); } catch {
    console.error(`REFUSED: ledger ${file} is not valid JSON. Fix or delete it.`);
    process.exit(2);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    console.error(`REFUSED: ledger ${file} is not a JSON object.`);
    process.exit(2);
  }
  return parsed;
}

/**
 * Is this a usable anchor record, or merely a key that exists?
 *
 * The gate used to test `!ledger[id]?.vsBase`, which any truthy value satisfies:
 * `{"s1":{"vsBase":true}}` passed, and the run then printed "scored undefined% vs P
 * (undefined Elo), undefined games". The ledger is a plain JSON file an operator can
 * hand-edit, so the record has to be checked, not merely present.
 */
function anchorProblem(rec, wantBase) {
  if (rec === null || typeof rec !== 'object' || Array.isArray(rec)) return 'is not an object';
  if (rec.base !== wantBase) return `was measured against '${rec.base}', not '${wantBase}'`;
  if (!Number.isFinite(rec.games) || rec.games <= 0) return `has no game count (${rec.games})`;
  if (!Number.isFinite(rec.scorePct)) return `has no score (${rec.scorePct})`;
  if (rec.elo !== null && !Number.isFinite(rec.elo)) return `has a non-numeric Elo (${rec.elo})`;
  return null;
}

/**
 * Refuse a student-vs-student screen until BOTH sides have been measured against the
 * parent checkpoint.
 *
 * A two-arm comparison measures the DIFFERENCE between the arms and is silent about
 * the LEVEL. On 2026-08-16 both arms had been destroyed by over-training; the screen
 * still returned 96.0% with a 93.7..98.2% interval and "significant", and 860 games
 * went into comparisons that could not have detected the collapse. Only an anchor can.
 *
 * The escape hatch is deliberately narrow: `--base <id>` names the parent, and a run
 * involving it is an anchor run that gets recorded. There is no flag to skip the check,
 * because the whole failure was believing a difference without knowing the level.
 */
function requireAnchors(ledger, a, b, base) {
  // Comparing a checkpoint with itself is never a screen. It also passes every gate
  // below, and reports "significant: X is stronger" than itself whenever noise pushes
  // the interval off 50%.
  if (a === b) {
    console.error(`REFUSED: --a and --b are both '${a}'. That is not a comparison.`);
    process.exit(3);
  }

  // PARENTHOOD IS DECLARED, NOT INFERRED.
  //
  // The first cut inferred it from absence -- anything not already recorded as a
  // student was allowed to be a base. That is simultaneously too strict and too weak.
  //
  // Too strict: in a lineage every checkpoint is a student of the previous one AND the
  // parent of the next, so screening a second generation against `g2-030` was refused
  // for the crime of `g2-030` having its own anchor against `g2-020`. With no override,
  // the only escape was deleting a real anchor record -- pushing operators to corrupt
  // the file the whole gate depends on.
  //
  // Too weak: on an EMPTY ledger nothing is a student, so `--a s1 --b s2 --base s2`
  // sailed through -- and because the anchor-run branch returns before every check
  // below, it skipped the collapse gate too, repeatably and forever. The guard was
  // weakest exactly when it matters most: a fresh experiment with an empty ledger,
  // which is when the incident happened.
  //
  // A positive assertion fixes both. `--declare-parent` is a one-time operator act
  // recorded in the ledger; after that the id is usable as a base and its own
  // studenthood is irrelevant.
  const parents = ledger[PARENTS_KEY] || {};
  if (args['declare-parent'] && !parents[base]) {
    parents[base] = { ts: new Date().toISOString() };
    ledger[PARENTS_KEY] = parents;
    mkdirSync(path.dirname(args.ledger), { recursive: true });
    writeFileSync(args.ledger, `${JSON.stringify(ledger, null, 2)}\n`);
    console.log(`declared '${base}' as a parent checkpoint -> ${args.ledger}`);
  }
  if (!parents[base]) {
    console.error(`REFUSED: '${base}' has not been declared as a parent checkpoint.`);
    console.error(`  Naming a sibling as '--base' is how an unanchored screen gets recorded`);
    console.error(`  as an anchored one, so the base has to be asserted rather than assumed.`);
    console.error(`  If '${base}' really is the parent these arms were trained from:`);
    console.error(`    node scripts/distill-compare.mjs --a ${a} --b ${base} --base ${base} --declare-parent ...`);
    console.error(`  Ledger: ${args.ledger}`);
    process.exit(3);
  }
  if (a === base || b === base) return { anchorRun: true, other: a === base ? b : a };

  // The anchor must EXIST, be a real record, and be against THIS base. The gate used to
  // test only that some truthy `vsBase` key was present and then print the base this run
  // asked for -- so a ledger anchored against an older parent passed, and the run's own
  // output asserted a measurement that had never been made.
  const problems = [];
  for (const id of [a, b]) {
    const rec = ledger[id]?.vsBase;
    if (rec === undefined) { problems.push([id, `has no anchor against '${base}'`]); continue; }
    const why = anchorProblem(rec, base);
    if (why) problems.push([id, why]);
  }
  if (problems.length) {
    console.error(`REFUSED: student-vs-student screen without a usable anchor.`);
    for (const [id, why] of problems) console.error(`    ${id} ${why}`);
    console.error(`  A two-arm screen measures the difference and says nothing about the level.`);
    console.error(`  Both arms can be far below '${base}' and still produce a clean, significant`);
    console.error(`  result -- that is exactly what happened on 2026-08-16 (860 games wasted).`);
    console.error(`  Run each student against the parent first:`);
    for (const [id] of problems) {
      console.error(`    node scripts/distill-compare.mjs --a ${id} --b ${base} --games 200 ...`);
    }
    console.error(`  Ledger: ${args.ledger}`);
    process.exit(3);
  }

  // AN ANCHOR THAT SHOWS COLLAPSE IS NOT PERMISSION TO PROCEED. Requiring the anchor to
  // exist, but not to be acceptable, would have passed the incident verbatim: the ledger
  // would have held 0-0-200 and 9-1-190 against the parent and the screen would still
  // have run its 300 games. Comparing two damaged nets measures degrees of damage.
  const unfit = [];
  for (const id of [a, b]) {
    const rec = ledger[id].vsBase;
    const saturated = rec.elo === null;
    if (saturated) {
      unfit.push([id, `scored ${rec.scorePct}% vs '${base}' (saturated: one side won every`
        + ` decided game)`]);
    } else if (Math.abs(rec.elo) > BREAKAGE_ELO) {
      // ONE threshold, both directions.
      //
      // This was two branches, a floor at -POOL_SPREAD_HEALTHY and a magnitude check at
      // BREAKAGE_ELO. The floor was wrong twice over. At -145 it refused a legitimately
      // weak early-generation arm (30.3% against its parent) with no override, and it
      // was stricter than the tripwire's own definition of "beyond anything this loop
      // has produced honestly". Raising it to -BREAKAGE_ELO fixed that but made the
      // branch DEAD: anything below -290 already fails `abs > 290`. A guard clause that
      // cannot fire is the bug this file exists to prevent, so the two are one check.
      //
      // Both incident students are still refused: 0.0% is saturated above, 4.5% is -530.
      unfit.push([id, rec.elo < 0
        ? `is ${rec.elo} Elo BELOW '${base}', past the ${BREAKAGE_ELO} breakage threshold`
        : `is +${rec.elo} Elo against '${base}', beyond the ${BREAKAGE_ELO} breakage`
          + ` threshold -- a student does not beat its own parent by that much honestly`]);
    }
  }
  if (unfit.length) {
    console.error(`REFUSED: an arm's own anchor says it is not a healthy net.`);
    for (const [id, why] of unfit) console.error(`    ${id} ${why}`);
    console.error(`  A screen between these arms measures degrees of damage, not teaching`);
    console.error(`  quality. On 2026-08-16 both arms were in exactly this state (0-0-200 and`);
    console.error(`  9-1-190 against their own parent) and the screen still returned a clean`);
    console.error(`  96.0% with a "significant" verdict.`);
    console.error(`  Fix the recipe and retrain; do not screen damaged nets against each other.`);
    console.error(`  Ledger: ${args.ledger}`);
    process.exit(3);
  }

  for (const id of [a, b]) {
    const rec = ledger[id].vsBase;
    // `rec.base`, never `base`: printing the requested base would restate the argument
    // as if it were a measurement.
    console.log(`anchor on record: ${id} scored ${rec.scorePct}% vs ${rec.base}`
      + ` (${rec.elo === null ? 'saturated' : `${rec.elo > 0 ? '+' : ''}${rec.elo} Elo`})`
      + `, ${rec.games} games (${rec.decided ?? '?'} decided), ${rec.ts}`);
  }
  return { anchorRun: false, other: null };
}

const GAMES = Number(args.games);
if (GAMES % 2 !== 0) {
  console.error(`--games must be even so each net plays each colour equally, got ${GAMES}`);
  process.exit(2);
}
const SIMS = Number(args.sims);
const CONS = Number(args.considered);
const SLOTS = Number(args.slots);

// Guard 2 runs BEFORE the engine loads and before a single game is played: a refusal
// should cost nothing, and a screen that cannot be reported should not be run.
const ledger = readLedger(args.ledger);
const { anchorRun, other: anchoredSubject } = requireAnchors(ledger, args.a, args.b, args.base);

const { wasm, memory, build } = await loadWasm();
// One client per checkpoint. Same server, different `x-checkpoint` header, so the
// two nets are served by one process and cannot drift in device, dtype or fold.
const clientA = createInferClient({ url: args.infer, checkpoint: args.a, maxSockets: 4 });
const clientB = createInferClient({ url: args.infer, checkpoint: args.b, maxSockets: 4 });

console.log(`distill-compare: A=${args.a} vs B=${args.b} at ${SIMS} sims / ${CONS} considered`);
console.log(`${GAMES} games, ${SLOTS} slots, wasm ${build}, server ${args.infer}\n`);

/** One game in progress. `aSide` is the colour net A plays. */
class Game {
  constructor(index) {
    this.index = index;
    this.aSide = index % 2 === 0 ? 'A' : 'B';
    this.state = createInitialState(CONFIG);
    this.history = [];
    this.plies = 0;
    this.done = false;
    this.result = null;
  }

  /** Which checkpoint owns the move about to be searched. */
  moverCheckpoint() {
    return this.state.position.turn === this.aSide ? 'a' : 'b';
  }

  nextSpec() {
    if (this.done || this.state.outcome.kind !== 'ongoing') return null;
    return {
      state: this.state,
      sims: SIMS,
      considered: CONS,
      // Vary with game AND ply: a constant seed makes two deterministic players
      // replay one game, which would report 60-0 off a sample size of one.
      seed: (Number(args.seed) + this.index * 100003 + this.plies + 1) >>> 0
    };
  }

  play(actionCode) {
    const mover = this.state.position.turn;
    const action = decodeAction(CONFIG, actionCode);
    this.history.push(historyEntry(mover, action));
    this.state = applyAction(CONFIG, this.state, action);
    this.plies += 1;
    if (this.state.outcome.kind !== 'ongoing') {
      this.done = true;
      const o = this.state.outcome;
      this.result = o.kind === 'draw' ? 'draw' : (o.winner === this.aSide ? 'a' : 'b');
    }
  }
}

const games = Array.from({ length: GAMES }, (_, i) => new Game(i));
let nextGame = 0;

const slots = Array.from({ length: SLOTS }, () => ({
  game: null, search: null, guard: null, side: null,
  scratchMask: new Float32Array(POLICY_LEN)
}));
const batch = new Float32Array(SLOTS * FEATURE_LEN);

function fill(slot) {
  for (;;) {
    if (!slot.game || slot.game.done) {
      if (nextGame >= games.length) return false;
      slot.game = games[nextGame++];
    }
    const spec = slot.game.nextSpec();
    if (!spec) { slot.game = null; continue; }
    slot.side = slot.game.moverCheckpoint();
    slot.search = new wasm.NormalDuelSearch(CONFIG_JSON, JSON.stringify(spec.state),
      JSON.stringify({
        simulations: spec.sims, maxConsidered: spec.considered,
        cPuct: Number(args.cpuct), seed: spec.seed
      }));
    slot.guard = createNetGuard(slot.search.policyLen());
    return true;
  }
}

function finishSearch(slot) {
  const code = slot.search.actionCode();
  // The narrow-flat endgame refusal is expected in real games and must not abort a
  // 300-game screen: 1-2 legal moves make the logits necessarily flat, so the guard
  // reports no informed leaf. See distill-lib for the full diagnosis. Here the move
  // is simply played -- nothing is being recorded, so there is nothing to skip.
  try {
    slot.guard.finish();
  } catch (err) {
    const benign = err.code === 'net_never_consulted'
      && slot.guard.leaves > 0 && slot.guard.leaves === slot.guard.narrowFlat;
    if (!benign) throw err;
  }
  slot.search.free();
  slot.search = null;
  slot.game.play(code);
}

const t0 = performance.now();
let steps = 0;
let leaves = 0;

for (;;) {
  const pending = [];
  for (const slot of slots) {
    if (!slot.search && !fill(slot)) continue;
    for (;;) {
      slot.search.nextLeaf();
      if (!slot.search.isDone()) break;
      finishSearch(slot);
      if (!fill(slot)) break;
    }
    if (!slot.search) continue;
    const view = new Float32Array(memory.buffer, slot.search.featuresPtr(), slot.search.featuresLen());
    batch.set(view, pending.length * FEATURE_LEN);
    slot.search.pendingLeafMask();
    slot.scratchMask.set(new Float32Array(memory.buffer, slot.search.maskPtr(), POLICY_LEN));
    pending.push(slot);
  }
  if (pending.length === 0) break;

  // THE ONE DIFFERENCE FROM runLockStep: dispatch per checkpoint. Features are
  // re-packed into a contiguous sub-batch per side, because the wire format is a
  // flat run of n * 810 floats with no per-row routing.
  for (const side of ['a', 'b']) {
    const group = pending.filter((s) => s.side === side);
    if (group.length === 0) continue;
    const sub = new Float32Array(group.length * FEATURE_LEN);
    for (let i = 0; i < group.length; i += 1) {
      const at = pending.indexOf(group[i]) * FEATURE_LEN;
      sub.set(batch.subarray(at, at + FEATURE_LEN), i * FEATURE_LEN);
    }
    const client = side === 'a' ? clientA : clientB;
    const out = await client.infer(sub, group.length);
    for (let i = 0; i < group.length; i += 1) {
      const slot = group[i];
      const row = out.subarray(i * (POLICY_LEN + 1), (i + 1) * (POLICY_LEN + 1));
      const { probs, value } = slot.guard.classify(row, slot.scratchMask);
      new Float32Array(memory.buffer, slot.search.policyPtr(), POLICY_LEN).set(probs);
      slot.search.submit(value);
    }
  }
  steps += 1;
  leaves += pending.length;
  if (steps % 500 === 0) {
    const done = games.filter((g) => g.done).length;
    const sec = (performance.now() - t0) / 1000;
    console.log(`  ${done}/${GAMES} games, ${leaves} leaves, ${(leaves / sec).toFixed(0)} leaves/s`);
  }
}

const finished = games.filter((g) => g.done);
const aWins = finished.filter((g) => g.result === 'a').length;
const bWins = finished.filter((g) => g.result === 'b').length;
const draws = finished.filter((g) => g.result === 'draw').length;
const decided = aWins + bWins;
const rate = decided > 0 ? aWins / decided : 0;

console.log(`\n${args.a} (A) vs ${args.b} (B), ${finished.length} games in ${fmtDuration(performance.now() - t0)}`);
console.log(`  A ${aWins} - ${draws} - ${bWins} B   (A scores ${(100 * rate).toFixed(1)}% of decided)`);
// A 95% normal-approximation interval on the DECIDED games. Reported as an interval
// because a point estimate off a few hundred games invites reading noise as signal:
// at 300 games the interval is about +/-6 points of win rate, i.e. tens of Elo.
if (decided > 0) {
  const se = Math.sqrt(rate * (1 - rate) / decided);
  const lo = Math.max(0, rate - 1.96 * se);
  const hi = Math.min(1, rate + 1.96 * se);
  console.log(`  95% CI on A's score: ${(100 * lo).toFixed(1)}% .. ${(100 * hi).toFixed(1)}%`);
  const elo = (p) => (p <= 0 || p >= 1) ? null : -400 * Math.log10(1 / p - 1);
  const e = elo(rate); const el = elo(lo); const eh = elo(hi);
  console.log(`  implied Elo for A: ${e === null ? 'undefined (saturated)' : e.toFixed(0)}`
    + `${el !== null && eh !== null ? ` (${el.toFixed(0)} .. ${eh.toFixed(0)})` : ''}`);
  if (lo < 0.5 && hi > 0.5) {
    console.log('  -> interval spans 50%: NO significant difference at this sample size.');
  } else {
    console.log(`  -> significant: ${rate > 0.5 ? args.a : args.b} is stronger.`);
  }

  // GUARD 3 -- effect-size tripwire. Automatic, because judgement already failed once.
  const mag = e === null ? Infinity : Math.abs(e);
  if (mag > BREAKAGE_ELO) {
    const shown = e === null ? 'saturated (one side won every decided game)' : `${e.toFixed(0)} Elo`;
    console.log(`  !! SUSPECTED BREAKAGE: ${shown} is beyond anything this loop has produced`
      + ` honestly (threshold ${BREAKAGE_ELO}; healthy nets span ~${POOL_SPREAD_HEALTHY};`
      + ` largest legitimate single-intervention gain on record ~+76).`);
    console.log('  !! Most likely one side is DAMAGED rather than the other being good. Do not'
      + ' report this as a result until both sides are anchored to the parent checkpoint.');
    if (mag > POOL_SPREAD_FULL) {
      console.log(`  !! It also exceeds the pool's entire historical spread (~${POOL_SPREAD_FULL},`
        + ' classical anchor to best net) -- more than every arm ever run, combined.');
    }
  } else if (mag > POOL_SPREAD_HEALTHY) {
    console.log(`  !  ${e.toFixed(0)} Elo is larger than the spread among healthy nets`
      + ` (~${POOL_SPREAD_HEALTHY}). Not impossible, but verify the loser against the parent.`);
  }

  // An anchor run is the thing student-vs-student screens are gated on, so record it.
  if (anchorRun && anchoredSubject) {
    // `rate` is A's score. Store it from the SUBJECT's point of view regardless of
    // which side the base was given on, so the ledger cannot be read backwards.
    const subjectRate = args.a === anchoredSubject ? rate : 1 - rate;
    const subjectElo = elo(subjectRate);
    ledger[anchoredSubject] = {
      ...(ledger[anchoredSubject] || {}),
      vsBase: {
        base: args.base,
        scorePct: +(100 * subjectRate).toFixed(1),
        elo: subjectElo === null ? null : +subjectElo.toFixed(0),
        // BOTH counts. `scorePct` is a rate over DECIDED games, while `games` is every
        // finished game, so a draw-heavy anchor recorded "300 games" for a rate resting
        // on ten of them -- and the gate later printed that 300 as its justification.
        games: finished.length,
        decided,
        sims: SIMS,
        considered: CONS,
        seed: Number(args.seed),
        wasm: build,
        ts: new Date().toISOString()
      }
    };
    mkdirSync(path.dirname(args.ledger), { recursive: true });
    writeFileSync(args.ledger, `${JSON.stringify(ledger, null, 2)}\n`);
    console.log(`  anchor recorded for '${anchoredSubject}' -> ${args.ledger}`);
    // `=== null` FIRST. `elo()` returns null at a 0% or 100% score, so the original
    // `subjectElo !== null && ...` skipped this warning at total annihilation: the
    // incident's 0-0-200 student printed nothing while its 9-1-190 sibling printed the
    // warning. The worse result was the silent one. Line 325 already uses the right
    // idiom (`null -> Infinity`); this is the same rule spelled the same way.
    if (subjectElo === null || subjectElo < -POOL_SPREAD_HEALTHY) {
      const how = subjectElo === null
        ? `scored ${(100 * subjectRate).toFixed(1)}% against its parent (saturated: one side`
          + ` won every decided game, so the gap has no finite estimate)`
        : `is ${subjectElo.toFixed(0)} Elo BELOW its parent`;
      console.log(`  !! '${anchoredSubject}' ${how}.`
        + ' Any screen involving it measures degrees of damage, not teaching quality.');
    }
  }
}
console.log(`  plies: mean ${(finished.reduce((s, g) => s + g.plies, 0) / Math.max(1, finished.length)).toFixed(1)}`
  + `, min ${Math.min(...finished.map((g) => g.plies))}, max ${Math.max(...finished.map((g) => g.plies))}`);
console.log(`  infer: A ${clientA.stats.calls} calls / ${clientA.stats.positions} positions`
  + `, B ${clientB.stats.calls} calls / ${clientB.stats.positions} positions`);
