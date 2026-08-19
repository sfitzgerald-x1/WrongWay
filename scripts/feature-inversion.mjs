/**
 * Rebuild a playable game state from the stored feature planes alone.
 *
 * WHY THIS IS A MODULE. A self-play shard record is `features | policyTarget |
 * legalMask | z` and nothing else: no move list, no ply, no repetition history.
 * Every other position-addressing path in this repo replays a MOVE HISTORY from
 * `createInitialState` (`distill-lib.mjs` says so in its own header, and it is
 * right -- `validateState` ties ply, historyStartPly and each side's wall spend
 * together, so a hand-assembled board is rejected as `invalid_state`).
 * Reanalyse -- re-searching stored positions at a deeper budget -- has to start
 * from positions that were written months ago by a fleet that kept no histories.
 * Either the planes invert, or Reanalyse cannot run on the corpus that exists.
 *
 * WHAT IS RECOVERABLE: `{ pawns, walls, stock, turn }`, exactly. The pawn planes
 * are one-hot, the stock planes are `k / initialStock`, the side-to-move plane is
 * a bit, and the wall planes are an exact domino tiling per line (see
 * `wallAnchorsFromPlaneLine`). Nothing is guessed and nothing is fitted.
 *
 * WHAT IS NOT: `ply` and the repetition window. The features do not carry them,
 * so `synthesizeState` INVENTS both -- the earliest ply consistent with the
 * position, and a window padded with positions picked to satisfy `validateState`
 * rather than because they ever occurred. Measured over 2503 engine-generated
 * ongoing states: the rebuilt ply differs from the true one in 93.0% of them
 * (mean gap 24.3 plies, max 82), the current position's repetition count is wrong
 * in 7.9%, and 160 states carried a window entry for a position the game never
 * reached.
 *
 * WHAT THAT COSTS THE RE-SEARCH, which is this module's whole purpose. A search
 * launched from a rebuilt state adjudicates against that synthetic history, so it
 * MISSES draws that are really there -- a genuine threefold sits one identical
 * legal move from a root whose true ply was 69 and whose rebuilt ply was 21 -- and
 * it can COUNT repetitions that never happened: of the fabricated window entries,
 * 30 are reachable from the rebuilt position within 6 plies and 10 after a single
 * pawn move. Ply-cap draws go the same way, the rebuilt ply being almost always
 * short. How far that moves a reanalysed target has NOT been measured. It is a
 * hazard of unknown size, and nothing downstream reports it.
 *
 * `verifyRecord` CANNOT SEE ANY OF THAT, and is not evidence against it. It pins
 * the POSITION, not the state: `encodeState` reads only `state.position`, so two
 * states that share a position and differ in ply or repetition window encode to
 * identical bytes and both pass. What the audit does protect is the record's own
 * stored LABEL -- a stored terminal record has an empty legal mask and a
 * rebuilt-as-ongoing one does not, so the mask comparison refuses it instead of
 * relabelling it. Production shards hold no such record (the audited one had
 * `minLegal` 1 and no empty mask), which is why that refusal is free in practice.
 *
 * AND IT IS NOT A RECOVERY PROBLEM FOR FUTURE DATA. That schema change has now
 * been made: the self-play record carries `ply`, `historyStartPly` and its
 * repetition window (`selfplay.rs`), so anything written by a build from
 * `RECORD_VERSION` 2 onward goes through `stateFromCarried` below and the
 * synthesis is not involved at all. `synthesizeState` remains for the positions
 * already on disk, which have no such recourse, and for the counterfactual a v2
 * consumer can now run: rebuild the state the old way beside the true one and
 * report how far apart they are.
 *
 * WHAT THIS DOES NOT REIMPLEMENT: any rule. Geometry, legality, wall conflicts,
 * path checks, the encoding and the plane order all come from
 * `js/normal-duel-engine.mjs` and `js/normal-duel-nn-encoding.mjs`. Plane indices
 * are looked up by NAME through `planeIndex`, not written as literals -- the
 * nn-runtime fixture records what a hardcoded plane count cost when the encoder
 * moved from 8 planes to 10, and a hardcoded plane INDEX fails the same way with
 * no golden to catch it.
 *
 * THE EVIDENCE THIS WAS PORTED FROM: a production shard of 1941 records and 6190
 * engine-generated positions, on which the feature re-encode and the stored legal
 * mask came back bit-exact for every single one, the wall inversion was
 * exhaustively bijective over all 2^8 anchor subsets of all 16 plane lines, and
 * 2013 engine-legal perturbations were detected without exception.
 * `check-feature-inversion.mjs` rebuilds that whole argument from the engine
 * alone -- exhaustive proof included, generated corpus an order of magnitude
 * smaller so it runs in seconds -- and spot-checks a shard when handed one. That
 * shard re-verified at 1941/1941 against this module.
 */

import {
  adjudicate, comparePositionKeys, createInitialState, decodeAction,
  normalizePosition, policySize, positionKey, validateConfig, validateState
} from '../js/normal-duel-engine.mjs';
import {
  NN_INPUT_PLANES, NN_PLANE_LAYOUT, encodeState, legalMaskFloat, planeIndex
} from '../js/normal-duel-nn-encoding.mjs';
import { CONFIG_9X9 } from '../tests/support/nn-runtime-fixture.mjs';

/**
 * The configuration self-play shards were written under, validated by the engine.
 *
 * Taken from the fixture the play server, `sims-match.mjs` and `distill-lib.mjs`
 * already share rather than declared again here, and passed EXPLICITLY by every
 * caller: `initialStock` is the divisor that turns the stock planes back into
 * wall counts, so a wrong config does not fail, it returns a plausible position
 * with the wrong stock. A default parameter would make that the quiet path.
 */
export const SELFPLAY_CONFIG = validateConfig(CONFIG_9X9);

/** Refused because the planes are not an encoding of any legal position. */
export class FeatureInversionError extends Error {
  constructor(reason, detail) {
    super(detail === undefined ? reason : `${reason}: ${JSON.stringify(detail)}`);
    this.name = 'FeatureInversionError';
    this.reason = reason;
    this.detail = detail;
  }
}

function fail(reason, detail) { throw new FeatureInversionError(reason, detail); }

/** Plane positions by name, resolved once. `planeIndex` throws on a typo. */
const PLANE = Object.freeze({
  moverPawn: planeIndex('mover_pawn'),
  opponentPawn: planeIndex('opponent_pawn'),
  wallHorizontal: planeIndex('wall_horizontal'),
  wallVertical: planeIndex('wall_vertical'),
  moverStock: planeIndex('mover_stock'),
  opponentStock: planeIndex('opponent_stock'),
  goalProximity: planeIndex('goal_proximity'),
  sideToMove: planeIndex('side_to_move')
});

/**
 * How far a scaled stock value may sit from an integer before it is corruption.
 *
 * The stored value is `fround(k / S)`, so scaling by S reintroduces at most
 * `S * 2^-24` of float32 slack -- 6e-7 at S = 10, and the largest deviation seen
 * over a whole production shard was 2.4e-7. The bound has to sit above that and
 * far below 0.5, where it would start rounding to the WRONG wall count; 1e-3
 * leaves three orders of magnitude of headroom on both sides.
 */
const STOCK_ROUNDING_TOLERANCE = 1e-3;

const PLAYERS = Object.freeze(['A', 'B']);
function other(player) { return player === 'A' ? 'B' : 'A'; }
function manhattan(left, right) { return Math.abs(left.r - right.r) + Math.abs(left.c - right.c); }
function sameSquare(left, right) { return left.r === right.r && left.c === right.c; }
function expectedTurn(config, ply) { return ply % 2 === 0 ? config.firstPlayer : other(config.firstPlayer); }
function completedActions(config, ply) {
  return {
    [config.firstPlayer]: Math.ceil(ply / 2),
    [other(config.firstPlayer)]: Math.floor(ply / 2)
  };
}

/** Feature vector length for a config, from the encoder's own plane count. */
export function featureLength(config) {
  const checked = validateConfig(config);
  return NN_INPUT_PLANES * checked.rows * checked.columns;
}

/**
 * Invert one line of a wall plane -- one row of the horizontal plane, one column
 * of the vertical plane -- back into the anchor offsets that produced it.
 *
 * `H-r-c` blocks the edges below (r,c) AND below (r,c+1), so it lights a DOMINO
 * of two adjacent cells in row r of the horizontal plane. The engine refuses
 * `H-r-(c-1)` and `H-r-(c+1)` next to an existing `H-r-c` (`invalid_wall_geometry`
 * in `normalizePosition`, and the same test in `isLegalWallNormalized`), so
 * anchors within a line are never adjacent and their dominoes never overlap.
 * A maximal run of set cells is therefore an exact domino tiling of a 1xL strip:
 * L must be even, and the tiling is unique because the leftmost cell of the run
 * can only be covered by the domino that starts on it -- then induct. So the
 * inversion is a function, not a search, and there is nothing to disambiguate.
 *
 * This is the one place where a silent wrong answer was cheapest to write, so it
 * is the one place with the exhaustive proof: `check-feature-inversion.mjs` runs
 * every anchor subset of every line through the ENGINE's `edgeBlocked` and back
 * through this function, and requires 880 legal sets to produce 880 distinct
 * lines and invert to themselves.
 *
 * Values must be exactly 0 or 1. Treating "not 1" as clear would read a corrupted
 * 0.5 as an empty edge and hand back a position that is merely plausible.
 */
export function wallAnchorsFromPlaneLine(bits) {
  const anchors = [];
  let runStart = -1;
  for (let i = 0; i <= bits.length; i += 1) {
    if (i < bits.length && bits[i] !== 0 && bits[i] !== 1) {
      fail('wall_plane_not_binary', { index: i, value: bits[i] });
    }
    const set = i < bits.length && bits[i] === 1;
    if (set) {
      if (runStart < 0) runStart = i;
      continue;
    }
    if (runStart < 0) continue;
    const runLength = i - runStart;
    if (runLength % 2 !== 0) fail('odd_wall_run', { start: runStart, length: runLength });
    for (let anchor = runStart; anchor < i; anchor += 2) anchors.push(anchor);
    runStart = -1;
  }
  return anchors;
}

/**
 * Reconstruct `{ pawns, walls, stock, turn }` from the feature planes alone.
 *
 * Returns the ENGINE's normalized position (canonically ordered walls, geometry
 * and both path requirements checked by `normalizePosition`), and throws
 * `FeatureInversionError` on anything the planes cannot mean. Refusing matters
 * more than it looks: every failure mode here has a plausible-looking wrong
 * answer sitting next to it -- an odd run of wall cells could be tiled by
 * dropping a cell, a non-integral stock could be floored -- and Reanalyse would
 * then search a board that never occurred and write a target for it.
 *
 * Planes 8 and 9 (the path-distance fields) are not read: they are a pure
 * function of the walls and the goal rows, so they carry nothing the wall planes
 * do not. They are still CHECKED, by `verifyRecord`'s full re-encode, which is
 * where a disagreement between the two representations shows up.
 */
export function positionFromFeatures(config, features) {
  const checked = validateConfig(config);
  const { rows, columns } = checked;
  const cells = rows * columns;
  // Float32Array specifically, not "array-like": the audit's claim is bit
  // equality against what is on disk, and the on-disk dtype is float32. A
  // Float64Array holding the same numbers scales differently in `stockOf` and
  // would compare unequal for reasons that have nothing to do with the position.
  if (!(features instanceof Float32Array)) fail('features_not_float32', { type: typeof features });
  if (features.length !== NN_INPUT_PLANES * cells) {
    fail('feature_length', { got: features.length, want: NN_INPUT_PLANES * cells });
  }
  const plane = (index) => features.subarray(index * cells, (index + 1) * cells);

  // Whose turn it is comes first because planes 0/1 and 4/5 are mover-relative by
  // channel selection: until the mover is known, "plane 0" names no player.
  const sideToMove = plane(PLANE.sideToMove);
  const flag = sideToMove[0];
  for (let i = 1; i < cells; i += 1) {
    if (sideToMove[i] !== flag) fail('side_to_move_not_constant', { cell: i, value: sideToMove[i] });
  }
  if (flag !== 0 && flag !== 1) fail('side_to_move_not_binary', { value: flag });
  const turn = flag === 1 ? 'A' : 'B';

  const oneHotSquare = (index) => {
    const values = plane(index);
    let found = -1;
    for (let i = 0; i < cells; i += 1) {
      if (values[i] === 0) continue;
      if (values[i] !== 1) fail('pawn_plane_not_binary', { plane: index, cell: i, value: values[i] });
      if (found >= 0) fail('pawn_plane_multi_hot', { plane: index, cells: [found, i] });
      found = i;
    }
    if (found < 0) fail('pawn_plane_empty', { plane: index });
    return { r: Math.floor(found / columns), c: found % columns };
  };
  const pawns = { [turn]: oneHotSquare(PLANE.moverPawn), [other(turn)]: oneHotSquare(PLANE.opponentPawn) };

  const stockOf = (index, player) => {
    const values = plane(index);
    const value = values[0];
    for (let i = 1; i < cells; i += 1) {
      if (values[i] !== value) fail('stock_plane_not_constant', { plane: index, cell: i });
    }
    const capacity = checked.initialStock[player];
    // `encodeState` guards the division, so a side that started with no walls
    // encodes as a constant 0 whatever it holds. Scaling by 0 would accept ANY
    // constant plane as "0 walls left"; the only consistent reading is that the
    // plane is 0 too, and anything else is corruption.
    if (capacity === 0) {
      if (value !== 0) fail('stock_plane_not_zero', { plane: index, value });
      return 0;
    }
    const scaled = value * capacity;
    const rounded = Math.round(scaled);
    if (Math.abs(scaled - rounded) > STOCK_ROUNDING_TOLERANCE) {
      fail('stock_not_integral', { plane: index, value, scaled });
    }
    if (rounded < 0 || rounded > capacity) fail('stock_out_of_range', { plane: index, stock: rounded, capacity });
    return rounded;
  };
  const stock = {
    [turn]: stockOf(PLANE.moverStock, turn),
    [other(turn)]: stockOf(PLANE.opponentStock, other(turn))
  };

  const horizontal = plane(PLANE.wallHorizontal);
  const vertical = plane(PLANE.wallVertical);
  const walls = [];
  for (let r = 0; r < rows; r += 1) {
    const line = [];
    for (let c = 0; c < columns; c += 1) line.push(horizontal[r * columns + c]);
    const anchors = wallAnchorsFromPlaneLine(line);
    // An `H` anchor exists only for r < rows - 1, so the last row of the
    // horizontal plane is unreachable and must be clear. The engine would refuse
    // `H-8-c` as `invalid_wall` anyway; naming the row says which plane is wrong.
    if (r === rows - 1 && anchors.length > 0) fail('wall_beyond_last_anchor_row', { row: r, anchors });
    for (const c of anchors) walls.push(`H-${r}-${c}`);
  }
  for (let c = 0; c < columns; c += 1) {
    const line = [];
    for (let r = 0; r < rows; r += 1) line.push(vertical[r * columns + c]);
    const anchors = wallAnchorsFromPlaneLine(line);
    if (c === columns - 1 && anchors.length > 0) fail('wall_beyond_last_anchor_column', { column: c, anchors });
    for (const r of anchors) walls.push(`V-${r}-${c}`);
  }

  // Redundancy the encoding already contains, spent on cross-checking rather than
  // left unread: the goal-proximity plane is a function of the MOVER's goal row
  // alone, so it independently names the side to move that plane 7 just claimed.
  // `fround` because the encoder computes the gradient in float64 and stores it
  // into a Float32Array; comparing against the float64 value fails on every row
  // whose value is not representable.
  const proximity = plane(PLANE.goalProximity);
  const goalRow = checked.goalRows[turn];
  for (let r = 0; r < rows; r += 1) {
    const want = Math.fround((rows - 1 - Math.abs(r - goalRow)) / (rows - 1));
    for (let c = 0; c < columns; c += 1) {
      if (proximity[r * columns + c] !== want) {
        fail('goal_proximity_disagrees_with_turn',
          { turn, row: r, column: c, got: proximity[r * columns + c], want });
      }
    }
  }

  // Walls are counted from planes 2/3 and stock from planes 4/5, so this is a
  // free consistency test between two independently decoded parts of the vector.
  // It is also `validateState`'s own invariant (`spent !== walls.length` fails
  // there), so a vector that breaks it encodes no reachable position -- failing
  // here names the disagreement instead of surfacing it later as `invalid_state`.
  const spent = (checked.initialStock.A - stock.A) + (checked.initialStock.B - stock.B);
  if (spent !== walls.length) fail('stock_wall_count_disagree', { spent, walls: walls.length });

  try {
    return normalizePosition(checked, { pawns, walls, stock, turn });
  } catch (error) {
    fail('engine_rejected_position', { code: error.code ?? error.message, pawns, walls, stock, turn });
  }
}

/**
 * History-window entries for a position that has at least one wall on the board.
 *
 * `validateState` requires every entry in `repetitionCounts` to carry the SAME
 * walls and the SAME stock as the current position, so a window can only differ
 * in pawns and turn -- which is exactly right, because a window that starts after
 * the last wall placement spans pawn moves only. The window is two plies, so the
 * budget is one rival move, and `validateState` bounds a move at Manhattan 2 (a
 * jump over the other pawn). Hence: the rival's own square, plus every square
 * within 2 of it.
 *
 * Nearest first, so the ordinary case is found immediately. The turn-flipped copy
 * comes first and is usually accepted; it is skipped when a pawn stands on its
 * goal row, because `validateState` allows a goal pawn only in the CURRENT entry
 * -- a won position's predecessor really did have the winner one step back.
 */
function* walledWindowCandidates(config, position) {
  const rival = other(position.turn);
  const goalFree = (candidate) => candidate.pawns.A.r !== config.goalRows.A
    && candidate.pawns.B.r !== config.goalRows.B;
  const twin = (square) => ({
    pawns: { ...position.pawns, [rival]: square },
    walls: [...position.walls],
    stock: { ...position.stock },
    turn: rival
  });
  const flipped = twin({ ...position.pawns[rival] });
  if (goalFree(flipped)) yield flipped;
  const moved = [];
  for (let r = 0; r < config.rows; r += 1) {
    for (let c = 0; c < config.columns; c += 1) {
      const square = { r, c };
      const step = manhattan(square, position.pawns[rival]);
      if (step === 0 || step > 2) continue;
      if (sameSquare(square, position.pawns[position.turn])) continue;
      const candidate = twin(square);
      if (goalFree(candidate)) moved.push(candidate);
    }
  }
  moved.sort((left, right) => manhattan(left.pawns[rival], position.pawns[rival])
    - manhattan(right.pawns[rival], position.pawns[rival]));
  yield* moved;
}

/**
 * History-window entries for a position with no walls, where the window has to
 * reach all the way back to ply 0.
 *
 * `validateState` couples the two: `(walls.length === 0) !== (historyStartPly === 0)`
 * fails, so a wall-free position cannot use a short window and a walled one
 * cannot use a long one. The initial position is yielded first because the same
 * function REQUIRES it in the window whenever `historyStartPly` is 0.
 *
 * Candidates are pawn placements within each side's move budget, excluding goal
 * rows (only the current entry may hold a goal pawn) and nearest first, both
 * turns offered because the window's turn counts have to come out ceil/floor.
 */
function* wallFreeWindowCandidates(config, position, bounds) {
  yield createInitialState(config).position;
  const squares = [];
  for (let r = 0; r < config.rows; r += 1) {
    for (let c = 0; c < config.columns; c += 1) squares.push({ r, c });
  }
  const near = (player) => squares
    .filter((square) => manhattan(square, position.pawns[player]) <= bounds[player]
      && square.r !== config.goalRows[player])
    .sort((left, right) => manhattan(left, position.pawns[player]) - manhattan(right, position.pawns[player]));
  for (const a of near('A')) {
    for (const b of near('B')) {
      if (sameSquare(a, b)) continue;
      for (const turn of PLAYERS) {
        yield {
          pawns: { A: a, B: b },
          walls: [...position.walls],
          stock: { ...position.stock },
          turn
        };
      }
    }
  }
}

/**
 * A `Map` of position key to count, in the order `validateState` demands.
 *
 * Shared by the two state builders rather than written out twice, and the reason
 * is a mutation that survived: with the sort spelled out at both call sites,
 * breaking one of them left the other correct and the suite green, so the check
 * was pinning one path and reporting on both. One implementation means one thing
 * to test.
 *
 * The order is by the position key's UTF-8 BYTES, which is not the numeric order
 * of the squares inside it -- `"9"` sorts after `"10"` -- so this cannot be
 * approximated by sorting the packed keys instead.
 */
function canonicalRepetitionCounts(counts) {
  return [...counts.entries()]
    .sort(([left], [right]) => comparePositionKeys(left, right))
    .map(([positionKeyText, count]) => ({ positionKey: positionKeyText, count }));
}

/** Build and VALIDATE one candidate state; the engine decides whether it stands. */
function stateAtPly(config, position, ply, historyStartPly) {
  const key = positionKey(config, position);
  const historyLength = ply - historyStartPly + 1;
  const counts = new Map([[key, 1]]);
  const historyFirstPlayer = expectedTurn(config, historyStartPly);
  const need = {
    [historyFirstPlayer]: Math.ceil(historyLength / 2),
    [other(historyFirstPlayer)]: Math.floor(historyLength / 2)
  };
  need[position.turn] -= 1;

  if (historyLength > 1) {
    const atPly = completedActions(config, ply);
    const atStart = completedActions(config, historyStartPly);
    const bounds = { A: 2 * (atPly.A - atStart.A), B: 2 * (atPly.B - atStart.B) };
    const candidates = historyStartPly === 0
      ? wallFreeWindowCandidates(config, position, bounds)
      : walledWindowCandidates(config, position);
    for (const candidate of candidates) {
      if (counts.size >= historyLength) break;
      if (need[candidate.turn] <= 0) continue;
      let candidateKey;
      // A candidate can be geometrically impossible -- a pawn pushed into a
      // pocket the existing walls seal off -- which is a skip, not a failure.
      try { candidateKey = positionKey(config, candidate); } catch { continue; }
      if (counts.has(candidateKey)) continue;
      counts.set(candidateKey, 1);
      need[candidate.turn] -= 1;
    }
    if (counts.size !== historyLength) {
      fail('history_window_unfilled', { ply, historyStartPly, filled: counts.size, want: historyLength });
    }
  }

  const atGoal = {
    A: position.pawns.A.r === config.goalRows.A,
    B: position.pawns.B.r === config.goalRows.B
  };
  const goalWinner = atGoal.A ? 'A' : atGoal.B ? 'B' : null;
  const outcome = ply === 0
    ? { kind: 'ongoing' }
    : adjudicate(config, { goalWinner, resultingPositionCount: counts.get(key), resultingPly: ply });
  return validateState(config, {
    position,
    positionKey: key,
    ply,
    historyStartPly,
    repetitionCounts: canonicalRepetitionCounts(counts),
    outcome
  });
}

/**
 * Wrap a position in the cheapest history the engine will accept.
 *
 * `encodeState` and `legalMaskFloat` read only `state.position`, but both run the
 * caller's state through `validateState` first, so Reanalyse cannot hand them a
 * bare position -- and the features carry no history to hand them instead. This
 * synthesises one and lets `validateState` be the judge of it; nothing here
 * decides that a state is legal.
 *
 * EARLIEST ply, deliberately. The scan stops one short of `plyCap` and the window
 * is filled with distinct positions, so the state handed back is labelled `win`
 * (a pawn is standing on its goal row, which the features DO carry) or `ongoing`,
 * and never a draw this function invented. `ply < plyCap` is load-bearing, not
 * cosmetic: at `<=`, a position whose earliest feasible ply IS the cap comes back
 * labelled a ply-cap draw on no evidence at all, where refusing is the honest
 * answer.
 *
 * That is a claim about THE LABEL ON THIS STATE and nothing wider. The ply is
 * synthetic and the window is padded with positions that did not occur, and a
 * search descending from here adjudicates against both -- missing real threefold
 * and ply-cap draws, and able to count repetitions that never happened. The
 * module header carries the measurements. `verifyRecord` cannot detect any of it,
 * because the features do not depend on either field.
 *
 * The ply scan is cheap because the two conditions that eliminate most plies --
 * turn parity, and `validateState`'s bound of two squares of pawn travel per
 * completed action -- are arithmetic, and are tested before a state is built.
 * On a production shard the accepted ply was 2..11 for the sampled records and
 * the window was a single position for 1883 of 1941.
 */
export function synthesizeState(config, positionInput) {
  const checked = validateConfig(config);
  const position = normalizePosition(checked, positionInput);
  const wallCount = position.walls.length;
  const spent = {
    A: checked.initialStock.A - position.stock.A,
    B: checked.initialStock.B - position.stock.B
  };
  if (spent.A + spent.B !== wallCount) {
    fail('stock_wall_count_disagree', { spent, walls: wallCount });
  }
  // Two boards `validateState` refuses at every ply: a pawn on its own goal row
  // with that side still to move (the game ended when it arrived), and both pawns
  // home at once. No history can rescue either, so they are named here rather
  // than discovered by scanning to `plyCap` -- which on a wall-free board means
  // rebuilding a hundred-entry repetition window a hundred times to learn nothing.
  const atGoal = {
    A: position.pawns.A.r === checked.goalRows.A,
    B: position.pawns.B.r === checked.goalRows.B
  };
  if (atGoal.A && atGoal.B) fail('both_pawns_on_goal_rows', { pawns: position.pawns });
  if (atGoal[position.turn]) {
    fail('mover_on_own_goal_row', { turn: position.turn, pawn: position.pawns[position.turn] });
  }
  // `historyStartPly` is forced to 0 with no walls on the board and forced above
  // 0 with walls, so the window length is not a free choice. With walls, the ply
  // before the window must have been a wall placement by its mover, so the
  // one-ply window needs the RIVAL to have spent; otherwise two plies, which
  // needs the mover to have spent -- and one of the two holds whenever a wall
  // exists at all.
  const windows = wallCount === 0 ? [null] : (spent[other(position.turn)] > 0 ? [1, 2] : [2]);
  const attempts = [];
  for (let ply = wallCount; ply < checked.plyCap; ply += 1) {
    if (expectedTurn(checked, ply) !== position.turn) continue;
    const completed = completedActions(checked, ply);
    const outOfReach = (player) =>
      manhattan(checked.start[player], position.pawns[player]) > 2 * completed[player];
    if (PLAYERS.some(outOfReach)) continue;
    for (const window of windows) {
      const historyStartPly = window === null ? 0 : ply - (window - 1);
      if (historyStartPly < wallCount) continue;
      const atStart = completedActions(checked, historyStartPly);
      if (spent.A > atStart.A || spent.B > atStart.B) continue;
      try {
        return stateAtPly(checked, position, ply, historyStartPly);
      } catch (error) {
        if (attempts.length < 4) {
          attempts.push(`ply ${ply} window ${historyStartPly}..${ply}: `
            + `${error.code ?? error.reason ?? error.message}`);
        }
      }
    }
  }
  fail('state_synthesis_failed', { position, attempts });
}

/** The whole path: stored planes in, a state the engine accepts out. */
export function stateFromFeatures(config, features) {
  const checked = validateConfig(config);
  return synthesizeState(checked, positionFromFeatures(checked, features));
}

/**
 * Decode one packed window key into the pawns and turn it names.
 *
 * The writer's `compact_key` is `squareA | squareB << 8 | turnBit << 16`, and
 * that is the whole of a window entry because a window is delimited by wall
 * placements: `validateState` requires every entry to share the current walls
 * and stock, so nothing else inside one can differ. Bits above the turn flag
 * would mean the reader and the writer disagree about the packing, which is a
 * refusal and not something to mask off.
 */
function pawnsFromWindowKey(config, key) {
  if (!Number.isInteger(key) || key < 0 || key > 0x1ffff) {
    fail('window_key_out_of_range', { key });
  }
  const cells = config.rows * config.columns;
  const squareA = key & 0xff;
  const squareB = (key >>> 8) & 0xff;
  if (squareA >= cells || squareB >= cells) fail('window_key_square_out_of_board', { key, squareA, squareB });
  return {
    pawns: {
      A: { r: Math.floor(squareA / config.columns), c: squareA % config.columns },
      B: { r: Math.floor(squareB / config.columns), c: squareB % config.columns }
    },
    turn: (key >>> 16) & 1 ? 'B' : 'A'
  };
}

/**
 * Build the state the game was ACTUALLY in, from the record that carries it.
 *
 * This is `synthesizeState`'s replacement and its opposite. Nothing here is
 * chosen: the position comes from the planes, `ply` and `historyStartPly` come
 * off the record, and the repetition window comes out of the sidecar the writer
 * filled from the window it was playing with. The only work is turning packed
 * keys back into the position keys the engine speaks, which is substitution
 * rather than search -- every window entry shares this position's walls and
 * stock by construction.
 *
 * `validateState` is still the judge, and it is a real one here: it re-derives
 * the outcome, re-checks the turn parity of the whole window, bounds each
 * historical pawn's distance from the current one by the moves available inside
 * the window, and requires the counts to sum to `ply - historyStartPly + 1`. A
 * true state passes all of that because the engine's own transition function
 * produced it. A state assembled from a corrupted or mismatched sidecar does
 * not.
 *
 * The two cheap identities are checked HERE rather than left to the engine so
 * the failure names the field: `validateState` would reject a wrong window sum
 * as `invalid_state`, which is true and tells a reader nothing about which of
 * the six fields was wrong.
 *
 * `window` is `[[key, count], ...]` in any order; the canonical ordering the
 * engine demands is applied below, so a caller need not know what it is.
 */
export function stateFromCarried(config, features, carried) {
  const checked = validateConfig(config);
  const position = positionFromFeatures(checked, features);
  const { ply, historyStartPly, window } = carried ?? {};
  if (!Number.isInteger(ply) || ply < 0) fail('carried_ply_invalid', { ply });
  if (!Number.isInteger(historyStartPly) || historyStartPly < 0 || historyStartPly > ply) {
    fail('carried_history_start_ply_invalid', { ply, historyStartPly });
  }
  if (!Array.isArray(window) || window.length === 0) {
    fail('carried_window_empty', { entries: Array.isArray(window) ? window.length : null });
  }

  const counts = new Map();
  let total = 0;
  for (const entry of window) {
    if (!Array.isArray(entry) || entry.length !== 2) fail('carried_window_entry_shape', { entry });
    const [key, count] = entry;
    if (!Number.isInteger(count) || count < 1) fail('carried_window_count_invalid', { key, count });
    const { pawns, turn } = pawnsFromWindowKey(checked, key);
    // Same walls, same stock -- that is what a window MEANS -- so the whole of
    // the historical position is this one with its pawns and turn swapped.
    let historicalKey;
    try {
      historicalKey = positionKey(checked, {
        pawns, walls: position.walls, stock: position.stock, turn
      });
    } catch (error) {
      fail('carried_window_entry_not_a_position', { key, reason: error.code ?? error.message });
    }
    if (counts.has(historicalKey)) fail('carried_window_duplicate_key', { key });
    counts.set(historicalKey, count);
    total += count;
  }
  if (total !== ply - historyStartPly + 1) {
    fail('carried_window_sum_mismatch', { total, ply, historyStartPly, want: ply - historyStartPly + 1 });
  }

  const key = positionKey(checked, position);
  if (!counts.has(key)) fail('carried_window_missing_root', { ply, historyStartPly });

  const atGoal = {
    A: position.pawns.A.r === checked.goalRows.A,
    B: position.pawns.B.r === checked.goalRows.B
  };
  const goalWinner = atGoal.A ? 'A' : atGoal.B ? 'B' : null;
  // Derived, never asserted: `validateState` recomputes it and refuses a
  // disagreement, so writing `{ kind: 'ongoing' }` here would be a claim rather
  // than a reading, and it would be the wrong claim at exactly the boundaries
  // this schema exists to get right.
  const outcome = ply === 0
    ? { kind: 'ongoing' }
    : adjudicate(checked, { goalWinner, resultingPositionCount: counts.get(key), resultingPly: ply });

  return validateState(checked, {
    position,
    positionKey: key,
    ply,
    historyStartPly,
    repetitionCounts: canonicalRepetitionCounts(counts),
    outcome
  });
}

/**
 * Index of the first element whose BITS differ, or -1.
 *
 * Bits rather than `!==` because the claim being made is bit equality: `===`
 * calls +0 and -0 equal and calls a NaN pair unequal, so it answers a slightly
 * different question than the one the audit is for. Not hypothetical -- a stored
 * `-0` in a cleared wall cell reads as clear, reconstructs the right board and
 * re-encodes to `+0`, so the float comparison passes a record whose bytes on disk
 * are not the bytes the engine produces. A Float32Array's byteOffset is always a
 * multiple of 4, so the uint32 view is always constructible.
 */
function firstDifferingElement(actual, expected) {
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return i;
  return -1;
}

/**
 * Audit one stored record: reconstruct it, then re-assert its own bytes.
 *
 * SCOPE, because the summary line reads as more than it is: this establishes that
 * the reconstructed POSITION is the one the record encodes. It says nothing about
 * ply or the repetition window -- `encodeState` never reads them -- so it is not
 * evidence that a search from the returned state will adjudicate the way the real
 * game did. The module header has the measurements and what they cost.
 *
 * Within that scope it is strong, and stronger than the tiling argument it rests
 * on. The features are re-encoded from the reconstruction and compared bit for
 * bit, and as far as a whole shard corpus and the perturbation controls can show,
 * `encodeState` sends distinct positions to distinct vectors -- so a vector this
 * accepts is the encoding of exactly one position, and a wrong tiling would be
 * caught by the re-encode even if the uniqueness proof in
 * `wallAnchorsFromPlaneLine` were wrong.
 *
 * The mask is compared too, not instead: the features pin the board and the mask
 * pins what the board can DO, and they fail in different places. The mask is the
 * only check that sees an outcome the features cannot carry -- a stored terminal
 * record has an empty mask, and a rebuild that reads it as ongoing does not.
 *
 * Returns the validated state, which is what a caller wants next anyway (it is
 * ready for `encodeState`, `legalMaskFloat` and search). THROWS on any mismatch
 * rather than returning a verdict: a returned boolean is a boolean someone
 * forgets to read, and the entire point is that a record which does not
 * reconstruct must not reach the searcher.
 *
 * Cost is dominated by the engine work the audit is made of -- one re-encode
 * (two BFS fields plus every edge) and one legal-mask build (a path test per
 * candidate wall). Measured over a whole production shard at 3.0 ms per record
 * warm and 5.1 ms cold, so a 1.1M-position corpus is roughly an hour on one core
 * and minutes across a shard set, which is a pass to run over a corpus rather
 * than something to put inline in a training step.
 */
export function verifyRecord(config, features, legalMask) {
  const checked = validateConfig(config);
  assertMaskShape(checked, legalMask);
  return auditRecordAgainst(checked, features, legalMask, stateFromFeatures(checked, features));
}

/**
 * The same audit against a state the CALLER built, for records that carry one.
 *
 * Under the v2 schema the state is not reconstructed, so the audit stops being a
 * check on the reconstruction and becomes a check on the WRITER: the carried ply
 * and window came out of one place in the engine and the features out of
 * another, and this is what says they describe the same position. A shard whose
 * carried state does not encode to its own features is refused rather than
 * relabelled -- the same refusal, aimed at the half that can now be wrong.
 *
 * It is free either way. The re-encode and the mask build are work the audit was
 * already doing; all that changes is which state goes in.
 *
 * Note what it still does NOT establish, for the same reason as before:
 * `encodeState` reads only `state.position`, so this cannot tell a right window
 * from a wrong one. Nothing can, from the features alone -- that is why the
 * window is carried. What pins the window is `validateState` inside
 * `stateFromCarried`, which re-derives the outcome and the turn parity of the
 * whole window from the ply, plus the sum identity checked there explicitly.
 */
export function verifyCarriedRecord(config, features, legalMask, carried) {
  const checked = validateConfig(config);
  assertMaskShape(checked, legalMask);
  return auditRecordAgainst(checked, features, legalMask, stateFromCarried(checked, features, carried));
}

/**
 * Checked BEFORE any state is built, in both entry points. A caller who passed
 * the wrong array should hear about the array, not about whichever plane the
 * inversion happened to trip over first.
 */
function assertMaskShape(checked, legalMask) {
  if (!(legalMask instanceof Float32Array)) fail('legal_mask_not_float32', { type: typeof legalMask });
  if (legalMask.length !== policySize(checked)) {
    fail('legal_mask_length', { got: legalMask.length, want: policySize(checked) });
  }
}

function auditRecordAgainst(checked, features, legalMask, state) {
  const reencoded = encodeState(checked, state);
  const featureIndex = firstDifferingElement(reencoded, features);
  if (featureIndex >= 0) {
    const cells = checked.rows * checked.columns;
    fail('feature_mismatch', {
      index: featureIndex,
      plane: NN_PLANE_LAYOUT[Math.floor(featureIndex / cells)].name,
      cell: featureIndex % cells,
      stored: features[featureIndex],
      reencoded: reencoded[featureIndex]
    });
  }

  const mask = legalMaskFloat(checked, state);
  const maskIndex = firstDifferingElement(mask, legalMask);
  if (maskIndex >= 0) {
    fail('legal_mask_mismatch', {
      code: maskIndex,
      action: decodeAction(checked, maskIndex),
      stored: legalMask[maskIndex],
      engine: mask[maskIndex],
      outcome: state.outcome.kind
    });
  }
  return state;
}
