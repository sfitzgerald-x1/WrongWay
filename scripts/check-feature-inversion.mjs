#!/usr/bin/env node
/**
 * Self-checks for `scripts/feature-inversion.mjs`.
 *
 *     node scripts/check-feature-inversion.mjs [shard.bin]
 *
 * The hard part is that a broken inversion returns a PLAUSIBLE BOARD. Drop one
 * wall and the result is still a legal position, still encodes to a well-formed
 * feature vector, still produces a sane legal mask, and still searches -- it is
 * simply not the position the shard recorded, and Reanalyse would write a target
 * for a game that was never played. Nothing downstream can see that. So every
 * case here is built so that a wrong answer FAILS rather than looks fine:
 *
 *   - the wall inversion is proved bijective by exhaustion, not sampled;
 *   - corrupted planes must THROW, one named reason each, never return a board;
 *   - generated positions are checked against their own ground truth AND
 *     re-encoded bit-for-bit, with class counts asserted so no class is vacuous;
 *   - perturbed positions must be DETECTED, so the comparison can fail at all;
 *   - the two outcomes the features cannot carry must be REFUSED, not guessed.
 *
 * No network, no cluster, no wasm, no node_modules, and no shard: positions come
 * from the engine itself. A shard path may be passed for a spot check and is
 * skipped cleanly when absent, because the file this was developed against is
 * scratch and will not exist in CI.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  applyAction, createInitialState, decodeAction, edgeBlocked, legalActionCodes,
  legalPositionActionCodes, normalizePosition, policySize, validateConfig
} from '../js/normal-duel-engine.mjs';
import {
  NN_UNREACHABLE_DISTANCE, encodeState, legalMaskFloat, planeIndex
} from '../js/normal-duel-nn-encoding.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import {
  FeatureInversionError, SELFPLAY_CONFIG as CONFIG, featureLength,
  positionFromFeatures, stateFromFeatures, synthesizeState, verifyRecord,
  wallAnchorsFromPlaneLine
} from './feature-inversion.mjs';

const ROWS = CONFIG.rows;
const COLUMNS = CONFIG.columns;
const CELLS = ROWS * COLUMNS;
const FEATURE_LEN = featureLength(CONFIG);
const FAILURES = [];

function report(ok, name, detail) {
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name.padEnd(34)} -> ${detail}`);
  if (!ok) FAILURES.push(`${name}: ${detail}`);
}

const other = (player) => (player === 'A' ? 'B' : 'A');
const manhattan = (left, right) => Math.abs(left.r - right.r) + Math.abs(left.c - right.c);
const samePosition = (left, right) => left.turn === right.turn
  && left.stock.A === right.stock.A && left.stock.B === right.stock.B
  && manhattan(left.pawns.A, right.pawns.A) === 0 && manhattan(left.pawns.B, right.pawns.B) === 0
  && left.walls.join('|') === right.walls.join('|');

// ------------------------------------------------------- wall-line bijection
//
// EXHAUSTIVE, not sampled, and it can be: `edgeBlockedSet` consults only
// `H-${min(r)}-${c}` and `H-${min(r)}-${c-1}` for a vertical step, so row r of the
// horizontal plane is a function of the H anchors in row r ALONE -- and likewise a
// column of the vertical plane. Enumerating every anchor subset of ONE line
// therefore covers the entire space of wall configurations as far as this plane
// line is concerned. Legality is decided by the ENGINE (`normalizePosition`
// raising `invalid_wall_geometry`) and the bits are rendered by the ENGINE
// (`edgeBlocked`), so neither the conflict rule nor the domino claim is assumed.
//
// `path_blocked` counts as geometry-legal: the wall set is well formed, it just
// seals a pawn in, which is a different rule and not what this case is about.

function geometryLegal(walls) {
  try {
    normalizePosition(CONFIG, {
      pawns: { A: { ...CONFIG.start.A }, B: { ...CONFIG.start.B } },
      walls, stock: { ...CONFIG.initialStock }, turn: CONFIG.firstPlayer
    });
    return true;
  } catch (error) {
    return error.code === 'path_blocked';
  }
}

function planeLine(orientation, line, walls) {
  const bits = [];
  const span = orientation === 'H' ? COLUMNS : ROWS;
  for (let i = 0; i < span; i += 1) {
    const from = orientation === 'H' ? { r: line, c: i } : { r: i, c: line };
    const to = orientation === 'H' ? { r: line + 1, c: i } : { r: i, c: line + 1 };
    bits.push(edgeBlocked(CONFIG, from, to, walls) ? 1 : 0);
  }
  return bits;
}

{
  const anchorSlots = COLUMNS - 1;
  const subsets = 1 << anchorSlots;
  let lines = 0;
  let legalSets = 0;
  let inverted = 0;
  let collisions = 0;
  let crossTalk = 0;
  const perLineLegal = new Set();
  for (const orientation of ['H', 'V']) {
    for (let line = 0; line < ROWS - 1; line += 1) {
      lines += 1;
      let legalHere = 0;
      const seen = new Map();
      for (let subset = 0; subset < subsets; subset += 1) {
        const anchors = [];
        for (let slot = 0; slot < anchorSlots; slot += 1) if (subset & (1 << slot)) anchors.push(slot);
        const walls = anchors.map((slot) => (orientation === 'H' ? `H-${line}-${slot}` : `V-${slot}-${line}`));
        if (!geometryLegal(walls)) continue;
        legalSets += 1;
        legalHere += 1;
        const bits = planeLine(orientation, line, walls);
        const key = bits.join('');
        if (seen.has(key)) collisions += 1;
        else seen.set(key, anchors);
        let recovered;
        try { recovered = wallAnchorsFromPlaneLine(bits); } catch { recovered = null; }
        if (recovered !== null && recovered.join(',') === anchors.join(',')) inverted += 1;
        // Walls elsewhere on the board must not change THIS line's bits, or the
        // line-at-a-time argument above is not sound and the enumeration is not
        // exhaustive after all.
        const noise = (orientation === 'H'
          ? [`H-${(line + 2) % (ROWS - 1)}-0`, `V-0-${(line + 3) % (COLUMNS - 1)}`]
          : [`V-0-${(line + 2) % (COLUMNS - 1)}`, `H-${(line + 3) % (ROWS - 1)}-0`])
          .filter((wall) => !walls.includes(wall));
        const noisy = [...walls, ...noise];
        if (geometryLegal(noisy)) {
          const noisyBits = planeLine(orientation, line, noisy);
          if (noisyBits.join('') !== key) crossTalk += 1;
        }
      }
      perLineLegal.add(legalHere);
    }
  }
  // The engine's conflict rule restricted to one line is exactly "no two adjacent
  // anchors", and the number of subsets of n slots with no two adjacent is
  // Fibonacci(n + 2). Deriving the expected count that way rather than pasting
  // the observed 880 means the case still means something if the rule changes:
  // it would then disagree with the combinatorics instead of with a stale number.
  const fibonacci = (n) => { let a = 1; let b = 1; for (let i = 2; i < n; i += 1) [a, b] = [b, a + b]; return b; };
  const wantPerLine = fibonacci(anchorSlots + 2);
  const wantTotal = wantPerLine * lines;
  report(legalSets === wantTotal && perLineLegal.size === 1 && [...perLineLegal][0] === wantPerLine,
    'wall lines: legal set count',
    `${legalSets} legal of ${lines * subsets} subsets, want ${wantTotal} (${wantPerLine} per line`
    + `, non-adjacent subsets of ${anchorSlots} slots)`);
  report(inverted === legalSets, 'wall lines: every set inverts',
    `${inverted}/${legalSets} recovered exactly`);
  report(collisions === 0, 'wall lines: no two sets collide',
    `${collisions} distinct anchor sets sharing a plane line`);
  report(crossTalk === 0, 'wall lines: independent of others',
    `${crossTalk} lines changed by unrelated walls`);
}

// --------------------------------------------------- corrupted planes refuse
//
// Each case takes a VALID vector and breaks exactly one thing. The reason code is
// asserted, not merely that something threw: "it threw" is also what a typo in the
// module produces, and these have to distinguish a diagnosis from a crash.

const OPENING = [
  { kind: 'pawn', to: { r: 7, c: 4 } }, { kind: 'pawn', to: { r: 1, c: 4 } },
  { kind: 'wall', wall: 'H-3-3' }, { kind: 'pawn', to: { r: 2, c: 4 } },
  { kind: 'pawn', to: { r: 6, c: 4 } }, { kind: 'wall', wall: 'V-5-5' }
];
const SAMPLE_STATE = OPENING.reduce((state, action) => applyAction(CONFIG, state, action),
  createInitialState(CONFIG));
const SAMPLE_FEATURES = encodeState(CONFIG, SAMPLE_STATE);
const PLANE = {
  moverPawn: planeIndex('mover_pawn'),
  opponentPawn: planeIndex('opponent_pawn'),
  wallHorizontal: planeIndex('wall_horizontal'),
  wallVertical: planeIndex('wall_vertical'),
  moverStock: planeIndex('mover_stock'),
  opponentStock: planeIndex('opponent_stock'),
  sideToMove: planeIndex('side_to_move'),
  moverPathDistance: planeIndex('mover_path_distance'),
  opponentPathDistance: planeIndex('opponent_path_distance')
};
const at = (plane, r, c) => plane * CELLS + r * COLUMNS + c;
const corrupt = (mutate) => { const copy = SAMPLE_FEATURES.slice(); mutate(copy); return copy; };

function refusalCase(name, want, features) {
  let got = 'RETURNED A POSITION';
  try {
    positionFromFeatures(CONFIG, features);
  } catch (error) {
    got = error instanceof FeatureInversionError ? error.reason : `${error.constructor.name}: ${error.message}`;
  }
  report(got === want, name, got === want ? got : `${got}, want ${want}`);
}

// The unmodified vector must invert, or every refusal below could be refusing for
// a reason that has nothing to do with the corruption. Caught rather than left to
// throw: an inversion broken badly enough to reject its own valid input would
// otherwise take the whole file down with a stack trace at this line, reporting
// nothing about the twenty cases underneath it.
{
  let detail;
  try {
    detail = samePosition(positionFromFeatures(CONFIG, SAMPLE_FEATURES), SAMPLE_STATE.position)
      ? 'recovered the sample position' : 'recovered a DIFFERENT position';
  } catch (error) {
    detail = `threw ${error instanceof FeatureInversionError ? error.reason : error.message}`;
  }
  report(detail === 'recovered the sample position', 'control: the intact vector inverts', detail);
}

refusalCase('not a Float32Array', 'features_not_float32', Array.from(SAMPLE_FEATURES));
refusalCase('truncated vector', 'feature_length', SAMPLE_FEATURES.slice(0, FEATURE_LEN - 1));
refusalCase('side_to_move not constant', 'side_to_move_not_constant',
  corrupt((f) => { f[at(PLANE.sideToMove, 4, 4)] = 0; }));
refusalCase('side_to_move not a bit', 'side_to_move_not_binary',
  corrupt((f) => { f.fill(0.5, PLANE.sideToMove * CELLS, (PLANE.sideToMove + 1) * CELLS); }));
refusalCase('pawn plane empty', 'pawn_plane_empty',
  corrupt((f) => { f.fill(0, PLANE.moverPawn * CELLS, (PLANE.moverPawn + 1) * CELLS); }));
refusalCase('pawn plane multi-hot', 'pawn_plane_multi_hot',
  corrupt((f) => { f[at(PLANE.moverPawn, 0, 0)] = 1; }));
refusalCase('pawn plane fractional', 'pawn_plane_not_binary',
  corrupt((f) => { f[at(PLANE.opponentPawn, 5, 5)] = 0.5; }));
refusalCase('stock plane not constant', 'stock_plane_not_constant',
  corrupt((f) => { f[at(PLANE.moverStock, 2, 2)] = 0.5; }));
refusalCase('stock plane not integral', 'stock_not_integral',
  corrupt((f) => { f.fill(0.55, PLANE.moverStock * CELLS, (PLANE.moverStock + 1) * CELLS); }));
// The case above sits at 0.5, the FURTHEST a scaled stock can be from an integer,
// so it refuses under any tolerance short of 0.5 and pins nothing about the one
// actually chosen. This one is 0.02 off and rounds back to the true stock, so no
// later check can catch it either -- only the tolerance can, and only if it stays
// near the float32 slack it was sized for.
refusalCase('stock a hair off an integer', 'stock_not_integral',
  corrupt((f) => {
    const stock = SAMPLE_STATE.position.stock[SAMPLE_STATE.position.turn];
    f.fill((stock + 0.02) / CONFIG.initialStock[SAMPLE_STATE.position.turn],
      PLANE.moverStock * CELLS, (PLANE.moverStock + 1) * CELLS);
  }));
refusalCase('stock above capacity', 'stock_out_of_range',
  corrupt((f) => { f.fill(1.1, PLANE.moverStock * CELLS, (PLANE.moverStock + 1) * CELLS); }));
refusalCase('wall plane fractional', 'wall_plane_not_binary',
  corrupt((f) => { f[at(PLANE.wallHorizontal, 0, 0)] = 0.5; }));
// One lone wall cell is the case that matters most: a tiler that rounded down
// would silently drop it and hand back a board with one fewer wall.
refusalCase('odd run of wall cells', 'odd_wall_run',
  corrupt((f) => { f[at(PLANE.wallHorizontal, 0, 0)] = 1; }));
refusalCase('H wall past the last anchor row', 'wall_beyond_last_anchor_row',
  corrupt((f) => { f[at(PLANE.wallHorizontal, ROWS - 1, 0)] = 1; f[at(PLANE.wallHorizontal, ROWS - 1, 1)] = 1; }));
refusalCase('V wall past the last anchor column', 'wall_beyond_last_anchor_column',
  corrupt((f) => {
    f[at(PLANE.wallVertical, 0, COLUMNS - 1)] = 1;
    f[at(PLANE.wallVertical, 1, COLUMNS - 1)] = 1;
  }));
// Flipping the side-to-move bit reassigns the mover-relative planes AND leaves the
// goal-proximity gradient pointing at the other player's goal row. Nothing about
// the board is malformed; only the redundancy between planes 6 and 7 catches it.
refusalCase('turn flipped, gradient not', 'goal_proximity_disagrees_with_turn',
  corrupt((f) => { f.fill(SAMPLE_FEATURES[PLANE.sideToMove * CELLS] === 1 ? 0 : 1,
    PLANE.sideToMove * CELLS, (PLANE.sideToMove + 1) * CELLS); }));
refusalCase('stock disagrees with wall count', 'stock_wall_count_disagree',
  corrupt((f) => { f.fill(8 / CONFIG.initialStock.A, PLANE.moverStock * CELLS, (PLANE.moverStock + 1) * CELLS); }));
refusalCase('both pawns on one square', 'engine_rejected_position',
  corrupt((f) => {
    f.fill(0, PLANE.opponentPawn * CELLS, (PLANE.opponentPawn + 1) * CELLS);
    for (let i = 0; i < CELLS; i += 1) {
      if (SAMPLE_FEATURES[PLANE.moverPawn * CELLS + i] === 1) f[PLANE.opponentPawn * CELLS + i] = 1;
    }
  }));

// A position can be perfectly legal to `normalizePosition` and still belong to no
// game, so the inversion's refusals do not cover synthesis's. These are the two
// corrupt-shard shapes that reach it: a stock column that does not pay for the
// walls on the board, and a side to move that has already won.

function synthesisCase(name, want, position) {
  let got = 'RETURNED A STATE';
  try {
    synthesizeState(CONFIG, position);
  } catch (error) {
    got = error instanceof FeatureInversionError ? error.reason : `${error.constructor.name}: ${error.message}`;
  }
  report(got === want, name, got === want ? got : `${got}, want ${want}`);
}

synthesisCase('stock does not pay for the walls', 'stock_wall_count_disagree',
  { ...SAMPLE_STATE.position, stock: { ...CONFIG.initialStock } });
synthesisCase('mover already stands on its goal', 'mover_on_own_goal_row', {
  pawns: { A: { r: CONFIG.goalRows.A, c: 4 }, B: { r: 4, c: 4 } },
  walls: [], stock: { ...CONFIG.initialStock }, turn: 'A'
});
synthesisCase('both pawns home at once', 'both_pawns_on_goal_rows', {
  pawns: { A: { r: CONFIG.goalRows.A, c: 4 }, B: { r: CONFIG.goalRows.B, c: 5 } },
  walls: [], stock: { ...CONFIG.initialStock }, turn: 'A'
});

// `verifyRecord` asserts two things and neither one is redundant, so each is given
// a record only IT can catch. The path-distance planes are the planes the
// inversion never reads: a record whose stored plane 8 disagrees with its own
// walls reconstructs to the right board and the right mask, and only the
// re-encode comparison sees it -- which is also what backs the claim that planes
// 8 and 9 are audited at all. A corrupt stored MASK is the mirror image: the
// features are perfect and only the mask comparison sees it. Delete either
// assertion and every other case in this file still passes.

const SAMPLE_MASK = legalMaskFloat(CONFIG, SAMPLE_STATE);

function auditCase(name, want, features, mask) {
  let got = 'ACCEPTED';
  try {
    verifyRecord(CONFIG, features, mask);
  } catch (error) {
    got = error instanceof FeatureInversionError ? error.reason : `${error.constructor.name}: ${error.message}`;
  }
  report(got === want, name, got === want ? got : `${got}, want ${want}`);
}

auditCase('control: the intact record verifies', 'ACCEPTED', SAMPLE_FEATURES, SAMPLE_MASK);
auditCase('stored mover path plane is wrong', 'feature_mismatch',
  corrupt((f) => { f[at(PLANE.moverPathDistance, 4, 0)] = 0.5; }), SAMPLE_MASK);
auditCase('stored opponent path plane is wrong', 'feature_mismatch',
  corrupt((f) => { f[at(PLANE.opponentPathDistance, 0, 0)] = 0.25; }), SAMPLE_MASK);
auditCase('stored mask legalises an illegal move', 'legal_mask_mismatch', SAMPLE_FEATURES,
  (() => { const copy = SAMPLE_MASK.slice(); copy[copy.indexOf(0)] = 1; return copy; })());
auditCase('stored mask drops a legal move', 'legal_mask_mismatch', SAMPLE_FEATURES,
  (() => { const copy = SAMPLE_MASK.slice(); copy[copy.indexOf(1)] = 0; return copy; })());
auditCase('mask of the wrong length', 'legal_mask_length', SAMPLE_FEATURES, SAMPLE_MASK.slice(1));
// The record's bytes are not the bytes the engine produces, and every value-level
// test in this file calls them equal: a stored `-0` in a cleared wall cell reads
// as clear, rebuilds the right board and re-encodes to `+0`. Only a comparison of
// BITS refuses it, which is the whole reason the audit compares uint32 views.
auditCase('stored wall cell holds negative zero', 'feature_mismatch',
  corrupt((f) => { f[at(PLANE.wallHorizontal, 0, 0)] = -0; }), SAMPLE_MASK);

// Surviving `validateState` once is not the bar. Reanalyse SEARCHES the rebuilt
// state, and a search descends by applying actions -- each of which revalidates
// the state it was handed and rebuilds the history that comes out. So every legal
// action must actually apply to a state that was synthesised from features alone.
{
  let detail;
  try {
    const state = stateFromFeatures(CONFIG, SAMPLE_FEATURES);
    const codes = legalActionCodes(CONFIG, state);
    let played = 0;
    for (const code of codes) {
      applyAction(CONFIG, state, decodeAction(CONFIG, code));
      played += 1;
    }
    detail = `${played}/${codes.length} legal actions applied at ply ${state.ply}`;
    report(codes.length > 0 && played === codes.length, 'rebuilt state is playable', detail);
  } catch (error) {
    report(false, 'rebuilt state is playable', `threw ${error.code ?? error.reason ?? error.message}`);
  }
}

// ------------------------------------------------------------ position classes
//
// Ground truth comes from the engine: play games, keep every state, and check the
// reconstruction against the position that actually produced the features. The
// arms exist to reach classes a single policy does not -- wall-free boards, boards
// with no stock left, pawns adjacent (the jump rule), and walls that seal cells off
// from a goal row entirely. Class counts are ASSERTED below, because a class with
// zero members is a case that silently tested nothing.

const ARMS = [
  { label: 'pawn-only', games: 6, pick: (codes) => codes.filter((code) => code < CELLS) },
  { label: 'uniform', games: 6, pick: (codes) => codes },
  {
    label: 'wall-saturating',
    games: 6,
    pick: (codes, random) => {
      const walls = codes.filter((code) => code >= CELLS);
      return (walls.length > 0 && random() % 100 < 92) ? walls : codes;
    }
  },
  {
    label: 'contact-seeking',
    games: 6,
    pick: (codes, random, state) => {
      const pawnCodes = codes.filter((code) => code < CELLS);
      if (pawnCodes.length === 0 || random() % 100 < 15) return codes;
      const rival = state.position.pawns[other(state.position.turn)];
      const distance = (code) => manhattan({ r: Math.floor(code / COLUMNS), c: code % COLUMNS }, rival);
      const best = Math.min(...pawnCodes.map(distance));
      return pawnCodes.filter((code) => distance(code) === best);
    }
  }
];

function playGame(seed, pick) {
  const random = createLcg32(seed);
  const seen = [];
  let state = createInitialState(CONFIG);
  for (let step = 0; step <= CONFIG.plyCap; step += 1) {
    seen.push(state);
    if (state.outcome.kind !== 'ongoing') break;
    const pool = pick(legalActionCodes(CONFIG, state), random, state);
    state = applyAction(CONFIG, state, decodeAction(CONFIG, pool[random() % pool.length]));
  }
  return seen;
}

/** A pawn race to the goal row, so the corpus is guaranteed a won terminal state. */
function goalSeekingGame() {
  const seen = [];
  let state = createInitialState(CONFIG);
  for (let step = 0; step <= CONFIG.plyCap; step += 1) {
    seen.push(state);
    if (state.outcome.kind !== 'ongoing') break;
    const mover = state.position.turn;
    const goalRow = CONFIG.goalRows[mover];
    const pawnCodes = legalActionCodes(CONFIG, state).filter((code) => code < CELLS);
    const toGoal = (code) => Math.abs(Math.floor(code / COLUMNS) - goalRow);
    const best = pawnCodes.reduce((left, right) => (toGoal(right) < toGoal(left) ? right : left));
    state = applyAction(CONFIG, state, decodeAction(CONFIG, best));
  }
  return seen;
}

/** Two pawns shuffling, so the corpus is guaranteed a threefold-repetition draw. */
function repetitionGame() {
  const cycle = [
    { kind: 'pawn', to: { r: 7, c: 4 } }, { kind: 'pawn', to: { r: 1, c: 4 } },
    { kind: 'pawn', to: { r: 8, c: 4 } }, { kind: 'pawn', to: { r: 0, c: 4 } }
  ];
  const seen = [];
  let state = createInitialState(CONFIG);
  for (let round = 0; round < 3; round += 1) {
    for (const action of cycle) {
      seen.push(state);
      if (state.outcome.kind !== 'ongoing') return seen;
      state = applyAction(CONFIG, state, action);
    }
  }
  seen.push(state);
  return seen;
}

const CORPUS = [];
for (const arm of ARMS) {
  for (let game = 0; game < arm.games; game += 1) {
    CORPUS.push(...playGame((0x51ed2701 + game * 2654435 + arm.label.length * 7919) >>> 0, arm.pick));
  }
}
const GOAL_SEEKING = goalSeekingGame();
const REPETITION = repetitionGame();
CORPUS.push(...GOAL_SEEKING, ...REPETITION);

// The two scripted games exist to guarantee a class the random arms may not reach.
// A generator that quietly stopped reaching it -- a rule change, a different tie
// break -- would leave its class assertion below satisfied by nothing at all.
{
  const won = GOAL_SEEKING.at(-1).outcome;
  const drawn = REPETITION.at(-1).outcome;
  report(won.kind === 'win' && won.reason === 'goal', 'scripted: pawn race reaches a goal',
    `${GOAL_SEEKING.length} states, ${JSON.stringify(won)}`);
  report(drawn.kind === 'draw' && drawn.reason === 'threefold_repetition',
    'scripted: shuffle repeats threefold', `${REPETITION.length} states, ${JSON.stringify(drawn)}`);
}

{
  const classes = {
    wallFree: 0, walled: 0, zeroStock: 0, pawnsAdjacent: 0, sealedCells: 0,
    ongoing: 0, wonTerminal: 0, drawTerminal: 0
  };
  let positionExact = 0;
  let verified = 0;
  let checked = 0;
  const drawReasons = new Map();
  const refusedDraws = new Map();
  const problems = [];
  const seenFeatures = new Map();
  let injectivityViolations = 0;

  for (const state of CORPUS) {
    checked += 1;
    const truth = state.position;
    const features = encodeState(CONFIG, state);
    const mask = legalMaskFloat(CONFIG, state);
    if (truth.walls.length === 0) classes.wallFree += 1; else classes.walled += 1;
    if (truth.stock[truth.turn] === 0) classes.zeroStock += 1;
    if (manhattan(truth.pawns.A, truth.pawns.B) === 1) classes.pawnsAdjacent += 1;
    for (let i = 0; i < CELLS; i += 1) {
      if (features[PLANE.moverPathDistance * CELLS + i] !== NN_UNREACHABLE_DISTANCE) continue;
      classes.sealedCells += 1;
      break;
    }
    if (state.outcome.kind === 'ongoing') classes.ongoing += 1;
    else if (state.outcome.kind === 'win') classes.wonTerminal += 1;
    else {
      classes.drawTerminal += 1;
      drawReasons.set(state.outcome.reason, (drawReasons.get(state.outcome.reason) ?? 0) + 1);
    }

    try {
      if (samePosition(positionFromFeatures(CONFIG, features), truth)) positionExact += 1;
      else if (problems.length < 6) problems.push(`position mismatch at ply ${state.ply}`);
    } catch (error) {
      if (problems.length < 6) problems.push(`invert threw ${error.reason ?? error.message} at ply ${state.ply}`);
    }

    // Distinct positions must not share a feature vector, or "the features
    // determine the board" is false however well the inversion is written.
    const featureKey = Buffer.from(features.buffer, features.byteOffset, features.byteLength).toString('base64');
    const positionText = JSON.stringify(truth);
    if ((seenFeatures.get(featureKey) ?? positionText) !== positionText) injectivityViolations += 1;
    seenFeatures.set(featureKey, positionText);

    // A ply-cap or repetition draw is the one thing the planes cannot express, so
    // the audit has to REFUSE it rather than hand back the same board relabelled
    // as ongoing. That refusal is what makes the stored-mask comparison
    // load-bearing: the features of a drawn position re-encode perfectly.
    const drawn = state.outcome.kind === 'draw';
    try {
      verifyRecord(CONFIG, features, mask);
      if (drawn) problems.push(`a ${state.outcome.reason} draw at ply ${state.ply} was ACCEPTED`);
      else verified += 1;
    } catch (error) {
      const reason = error instanceof FeatureInversionError ? error.reason : `${error.constructor.name}`;
      if (drawn) refusedDraws.set(reason, (refusedDraws.get(reason) ?? 0) + 1);
      else if (problems.length < 6) {
        problems.push(`verifyRecord threw ${reason} at ply ${state.ply}: ${error.message.slice(0, 120)}`);
      }
    }
  }

  const auditable = checked - classes.drawTerminal;
  report(positionExact === checked, 'generated: position recovered',
    `${positionExact}/${checked} recovered exactly`);
  report(verified === auditable, 'generated: record verified',
    `${verified}/${auditable} re-encoded and masked bit-exact`);
  report(problems.length === 0, 'generated: no anomalies',
    problems.length === 0 ? 'none' : problems.join(' | '));
  report(injectivityViolations === 0, 'generated: features determine board',
    `${seenFeatures.size} distinct vectors, ${injectivityViolations} collision(s)`);
  report(classes.drawTerminal > 0 && refusedDraws.size > 0
    && [...refusedDraws.values()].reduce((sum, n) => sum + n, 0) === classes.drawTerminal,
    'draws are refused, not relabelled',
    `${classes.drawTerminal} drawn state(s), refused as ${JSON.stringify([...refusedDraws])}`);
  report(drawReasons.has('threefold_repetition'), 'repetition draws are in the corpus',
    JSON.stringify([...drawReasons]));
  for (const [name, minimum] of [['wallFree', 1], ['walled', 1], ['zeroStock', 1], ['pawnsAdjacent', 1],
    ['sealedCells', 1], ['wonTerminal', 1], ['drawTerminal', 1]]) {
    report(classes[name] >= minimum, `class covered: ${name}`, `${classes[name]} position(s)`);
  }
  console.log(`       corpus ${checked} positions ${JSON.stringify(classes)}`);
}

// ------------------------------------------------- the other draw, and a config
//
// Random play draws by repetition long before ply 200, so the corpus above holds
// no ply-cap draw -- and manufacturing a 200-ply game without a repetition is a
// lot of machinery for one class. Lowering the CAP produces the same terminal
// kind in twelve plies. It also runs the module against a configuration that is
// not the self-play default, which is where anything hardcoded to `plyCap` 200
// or to the shipped config object rather than derived from the argument would
// show itself.

{
  const shortCap = validateConfig({ ...CONFIG, plyCap: 12 });
  // Both pawns advance without ever repeating a position, meeting the cap with the
  // repetition rule untouched. A stays in column 4 and B leaves it immediately, so
  // the two never contest a square and the script needs no legality search.
  const walk = [
    { kind: 'pawn', to: { r: 7, c: 4 } }, { kind: 'pawn', to: { r: 0, c: 5 } },
    { kind: 'pawn', to: { r: 6, c: 4 } }, { kind: 'pawn', to: { r: 0, c: 6 } },
    { kind: 'pawn', to: { r: 5, c: 4 } }, { kind: 'pawn', to: { r: 0, c: 7 } },
    { kind: 'pawn', to: { r: 4, c: 4 } }, { kind: 'pawn', to: { r: 0, c: 8 } },
    { kind: 'pawn', to: { r: 3, c: 4 } }, { kind: 'pawn', to: { r: 1, c: 8 } },
    { kind: 'pawn', to: { r: 2, c: 4 } }, { kind: 'pawn', to: { r: 2, c: 8 } }
  ];
  const states = [];
  let state = createInitialState(shortCap);
  for (const action of walk) {
    states.push(state);
    state = applyAction(shortCap, state, action);
  }
  const outcome = state.outcome;
  report(outcome.kind === 'draw' && outcome.reason === 'ply_cap', 'scripted: the walk hits the ply cap',
    `ply ${state.ply}/${shortCap.plyCap}, ${JSON.stringify(outcome)}`);

  let ongoingVerified = 0;
  for (const earlier of states) {
    try {
      verifyRecord(shortCap, encodeState(shortCap, earlier), legalMaskFloat(shortCap, earlier));
      ongoingVerified += 1;
    } catch { /* counted by the assertion below */ }
  }
  report(ongoingVerified === states.length, 'non-default config still verifies',
    `${ongoingVerified}/${states.length} ongoing states under plyCap ${shortCap.plyCap}`);

  let refusal = 'ACCEPTED';
  try {
    verifyRecord(shortCap, encodeState(shortCap, state), legalMaskFloat(shortCap, state));
  } catch (error) {
    refusal = error instanceof FeatureInversionError ? error.reason : error.constructor.name;
  }
  report(refusal === 'legal_mask_mismatch', 'a ply-cap draw is refused', refusal);
}

// The scan bound is `ply < plyCap`, and `<=` is NOT the same function. They differ
// on exactly one shape: a position whose EARLIEST feasible ply is the cap itself.
// Shipped refuses it; `<=` hands back a state labelled a ply-cap draw for a board
// it has no ply evidence about whatsoever, which is the fabrication the earliest-ply
// rule exists to avoid. The block above cannot reach that boundary -- its position
// synthesises at ply 6 under a cap of 12 -- so it gets its own case.

{
  const capAtEarliest = validateConfig({ ...CONFIG, plyCap: 2 });
  // One pawn move each. Turn A forces an even ply, and B's single step needs
  // floor(ply / 2) >= 1, so ply 2 is the first that fits -- and it is the cap.
  const position = {
    pawns: { A: { r: 7, c: 4 }, B: { r: 1, c: 4 } },
    walls: [], stock: { ...CONFIG.initialStock }, turn: 'A'
  };
  let atCap;
  try {
    const built = synthesizeState(capAtEarliest, position);
    atCap = `RETURNED ply ${built.ply} labelled ${JSON.stringify(built.outcome)}`;
  } catch (error) {
    atCap = error instanceof FeatureInversionError ? error.reason : error.constructor.name;
  }
  report(atCap === 'state_synthesis_failed', 'earliest ply AT the cap refuses', atCap);
  // ...and the identical position is ordinary under the production cap, so the
  // refusal above is about the boundary rather than about the board.
  let belowCap;
  try {
    belowCap = `ply ${synthesizeState(CONFIG, position).ply}`;
  } catch (error) {
    belowCap = error instanceof FeatureInversionError ? error.reason : error.constructor.name;
  }
  report(belowCap === 'ply 2', 'the same position is fine below the cap', belowCap);
}

// ---------------------------------------------------------- negative controls
//
// A comparison that cannot fail proves nothing. Every dimension the features are
// supposed to carry is perturbed by the smallest legal amount, and the perturbed
// board must re-encode to something DIFFERENT from the original bytes. Anything
// that survives undetected is a collision in the ENCODING, which would sink
// Reanalyse no matter how correct this module is.

function perturbations(config, position) {
  const out = [];
  const mover = position.turn;
  const rival = other(mover);
  const clone = () => ({
    pawns: { A: { ...position.pawns.A }, B: { ...position.pawns.B } },
    walls: [...position.walls], stock: { ...position.stock }, turn: position.turn
  });
  for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    const square = { r: position.pawns[mover].r + dr, c: position.pawns[mover].c + dc };
    if (square.r < 0 || square.r >= config.rows || square.c < 0 || square.c >= config.columns) continue;
    if (manhattan(square, position.pawns[rival]) === 0) continue;
    const moved = clone();
    moved.pawns[mover] = square;
    out.push({ name: 'mover_pawn_step', position: moved });
    break;
  }
  const swapped = clone();
  swapped.pawns[mover] = { ...position.pawns[rival] };
  swapped.pawns[rival] = { ...position.pawns[mover] };
  out.push({ name: 'swap_pawns', position: swapped });
  const flipped = clone();
  flipped.turn = rival;
  out.push({ name: 'flip_turn', position: flipped });
  if (position.stock.A !== position.stock.B) {
    const swappedStock = clone();
    swappedStock.stock = { A: position.stock.B, B: position.stock.A };
    out.push({ name: 'swap_stock', position: swappedStock });
  }
  if (position.walls.length > 0) {
    // The refund keeps `spent === walls.length`, so this stays a REACHABLE
    // position rather than being rejected for arithmetic before it is compared.
    const refunded = position.stock.A < config.initialStock.A ? 'A' : 'B';
    const dropped = clone();
    dropped.walls = position.walls.slice(1);
    dropped.stock[refunded] += 1;
    out.push({ name: 'remove_one_wall', position: dropped });
  }
  const spender = position.stock[mover] > 0 ? mover : (position.stock[rival] > 0 ? rival : null);
  if (spender) {
    const addable = legalPositionActionCodes(config, position).find((code) => code >= CELLS);
    if (addable !== undefined) {
      const added = clone();
      added.walls = [...position.walls, decodeAction(config, addable).wall];
      added.stock[spender] -= 1;
      out.push({ name: 'add_one_wall', position: added });
    }
  }
  return out;
}

{
  const byKind = new Map();
  let attempted = 0;
  let detected = 0;
  let unbuildable = 0;
  const undetected = [];
  for (let index = 0; index < CORPUS.length; index += 7) {
    const state = CORPUS[index];
    if (state.outcome.kind !== 'ongoing') continue;
    const features = encodeState(CONFIG, state);
    for (const perturbation of perturbations(CONFIG, state.position)) {
      let rebuilt;
      // An illegal or unreachable perturbation is not a counter-example: the
      // engine already refuses it, so it never reaches a comparison.
      try { rebuilt = synthesizeState(CONFIG, perturbation.position); }
      catch { unbuildable += 1; continue; }
      attempted += 1;
      const entry = byKind.get(perturbation.name) ?? { attempted: 0, detected: 0 };
      entry.attempted += 1;
      const reencoded = encodeState(CONFIG, rebuilt);
      let differs = false;
      for (let i = 0; i < FEATURE_LEN; i += 1) if (reencoded[i] !== features[i]) { differs = true; break; }
      if (differs) { detected += 1; entry.detected += 1; }
      else if (undetected.length < 6) undetected.push(`${perturbation.name} at corpus ${index}`);
      byKind.set(perturbation.name, entry);
    }
  }
  report(attempted > 0 && detected === attempted, 'perturbations are detected',
    `${detected}/${attempted} detected (${unbuildable} refused by the engine)`
    + `${undetected.length ? ` UNDETECTED: ${undetected.join(', ')}` : ''}`);
  const thin = [...byKind].filter(([, entry]) => entry.attempted === 0 || entry.detected !== entry.attempted);
  report(byKind.size >= 6 && thin.length === 0, 'every perturbation kind fired',
    JSON.stringify([...byKind].map(([name, entry]) => `${name} ${entry.detected}/${entry.attempted}`)));
}

// ------------------------------------------------------- optional shard check
//
// The shard format's own reader is imported rather than reimplemented -- a second
// 64-byte header parser is a truncation bug waiting to happen -- and it lives in
// the training repo, so this skips rather than fails when it is not reachable.

const shardPath = process.argv.slice(2).find((arg) => !arg.startsWith('-'));
if (!shardPath) {
  console.log('  skip  shard spot check                 -> no shard path given (optional)');
} else {
  const formatPath = process.env.WW_SHARD_FORMAT
    || path.resolve(path.dirname(fileURLToPath(import.meta.url)),
      '../../wrongway-training/cluster/shard-format.mjs');
  let format = null;
  try { format = await import(formatPath); }
  catch { format = null; }
  if (format === null) {
    console.log(`  skip  shard spot check                 -> ${formatPath} not importable; set WW_SHARD_FORMAT`);
  } else {
    const buffer = readFileSync(shardPath);
    const header = format.readShardHeader(buffer, buffer.length, path.basename(shardPath));
    const body = new Float32Array(
      buffer.buffer.slice(buffer.byteOffset + format.SHARD_HEADER_BYTES, buffer.byteOffset + buffer.length)
    );
    const shapeOk = header.featureLen === FEATURE_LEN && header.maskLen === policySize(CONFIG);
    report(shapeOk, 'shard: shape matches this config',
      `featureLen ${header.featureLen}/${FEATURE_LEN}, maskLen ${header.maskLen}/${policySize(CONFIG)}`);
    if (shapeOk) {
      let ok = 0;
      const errors = new Map();
      const started = Date.now();
      for (let index = 0; index < header.recordCount; index += 1) {
        const base = index * header.recordFloats;
        const features = body.subarray(base, base + header.featureLen);
        const maskAt = base + header.featureLen + header.policyLen;
        const mask = body.subarray(maskAt, maskAt + header.maskLen);
        try { verifyRecord(CONFIG, features, mask); ok += 1; }
        catch (error) {
          const reason = error instanceof FeatureInversionError ? error.reason : error.constructor.name;
          errors.set(reason, (errors.get(reason) ?? 0) + 1);
        }
      }
      const elapsed = Date.now() - started;
      report(ok === header.recordCount, 'shard: every record verified',
        `${ok}/${header.recordCount} in ${(elapsed / 1000).toFixed(1)}s `
        + `(${(elapsed / header.recordCount).toFixed(2)} ms/record)`
        + `${errors.size ? ` ${JSON.stringify([...errors])}` : ''}`);
    }
  }
}

// -------------------------------------------------------------------------------

if (FAILURES.length) {
  console.error(`\nFAIL (${FAILURES.length})`);
  for (const failure of FAILURES) console.error(`  ${failure}`);
  process.exit(1);
}
console.log('\nall feature-inversion checks pass: the wall inversion is bijective over every'
  + '\nplane line, corrupted planes refuse with a named reason instead of returning a'
  + '\nplausible board, every generated position round-trips bit-exact against its own'
  + '\nground truth, and perturbations are all detected. The two draws the planes cannot'
  + '\ncarry are refused by the stored-mask comparison, not guessed.');
