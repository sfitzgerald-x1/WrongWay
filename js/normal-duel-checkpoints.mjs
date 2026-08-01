/**
 * Stage 4 checkpoint identity, checkpoint evaluation and promotion for the
 * canonical 9x9 normal duel (`docs/ai-engine-plan.md:440-444`).
 *
 * The plan requires Gumbel simulation counts to be fixed only once deterministic
 * replay (PR #19), checkpoint promotion and checkpoint evaluation are reliable,
 * and it requires checkpoints to be gated "through paired-opening matches
 * against alpha-beta rather than self-play Elo alone". This module supplies the
 * last two halves of that: a content hash that names a checkpoint, an engine
 * adapter that lets a checkpoint be driven by the *existing* release-gate match
 * harness, and a promotion rule.
 *
 * Reuse, not reinvention
 * ----------------------
 * Matches are played by `scripts/evaluation/normal-duel-strength.mjs`
 * (`runMatch`) — the same protocol, the same rules authority, the same
 * adjudication and the same forfeit semantics as the release gate. Search is
 * `js/normal-duel-gumbel-search.mjs`. Rules are `js/normal-duel-engine.mjs`.
 * Randomness is `js/lcg32.mjs`. Nothing here reimplements a rule, a match loop,
 * or a PRNG.
 *
 * Determinism
 * -----------
 * No `Math.random`, no `Date`, no `performance.now` participates in any decision
 * or in any returned value. Every per-session RNG stream is derived by SHA-256
 * from the agent seed and the session context, so a match is reproducible from
 * (agent seed, opening, side) alone, and the two sides of a paired opening never
 * share a stream. Evaluation runs in the harness's `fixed-node-budget-v1`
 * regression mode by default, whose clock is the harness's deterministic logical
 * clock — the monotonic-deadline mode reads a real clock and would make results
 * machine-dependent.
 *
 * Absolute action frame
 * ---------------------
 * Action codes and input planes stay engine-absolute (PR #18). No mirroring and
 * no coordinate transform is applied anywhere in this file.
 *
 * Scope
 * -----
 * No I/O. A checkpoint here is a plain object carrying its own metadata; the
 * evaluator function is passed in separately. Loading weights from disk is out
 * of scope. Nothing here has learned knowledge, so no strength claim can be made
 * from any number it produces.
 */

import { createHash } from 'node:crypto';

import {
  decodeAction, legalActionCodes, validateConfig, validateState
} from './normal-duel-engine.mjs';
import { gumbelRootSearch } from './normal-duel-gumbel-search.mjs';
import { createLcg32 } from './lcg32.mjs';
import {
  REGRESSION_MODE,
  STRENGTH_MODE,
  runMatch,
  summarizeEvaluation,
  validateOpeningBook
} from '../scripts/evaluation/normal-duel-strength.mjs';

/** Frozen identifier for this checkpoint record format. */
export const CHECKPOINT_FORMAT = 'normal-duel-checkpoint-v1';

export class CheckpointError extends Error {
  constructor(reason) { super(reason); this.name = 'CheckpointError'; this.reason = reason; }
}

function fail(reason) { throw new CheckpointError(reason); }

function canonical9x9(config) {
  const checked = validateConfig(config);
  if (checked.rows !== 9 || checked.columns !== 9) fail('unsupported_board');
  return checked;
}

function positiveInteger(value, reason) {
  if (!Number.isSafeInteger(value) || value < 1) fail(reason);
  return value;
}

function uint32(value, reason) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) fail(reason);
  return value;
}

function sha256Hex(text) {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

/** Deterministic uint32 derived from a string, for seeding `createLcg32`. */
function seedFromDigest(text) {
  return Number.parseInt(sha256Hex(text).slice(0, 8), 16) >>> 0;
}

/* ------------------------------------------------------------------ *
 * Checkpoint identity
 * ------------------------------------------------------------------ */

/**
 * Canonical serialization. Object keys are emitted in explicit sorted order —
 * `JSON.stringify` would otherwise let insertion order leak into the hash, so
 * two structurally identical checkpoints built in different orders would get
 * different ids. Typed arrays (weight blobs) serialize as their element list, so
 * a `Float32Array` and the equivalent plain array agree.
 *
 * Anything that cannot be canonically represented — `undefined`, a function, a
 * symbol, `NaN`, an infinity — is rejected rather than silently dropped, because
 * a silently dropped field is a checkpoint whose id does not cover its content.
 */
function canonicalize(value, path) {
  if (value === null) return 'null';
  const type = typeof value;
  if (type === 'boolean') return value ? 'true' : 'false';
  if (type === 'string') return JSON.stringify(value);
  if (type === 'number') {
    if (!Number.isFinite(value)) fail(`non_canonical_number:${path}`);
    // Number#toString is the shortest round-tripping form (ECMA-262), so it is
    // stable across runs and distinguishes every distinct double except -0/0,
    // which is handled explicitly.
    return Object.is(value, -0) ? '-0' : value.toString();
  }
  if (type === 'bigint') return `${value.toString()}n`;
  if (type !== 'object') fail(`non_canonical_value:${path}`);
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const parts = [];
    for (let index = 0; index < value.length; index += 1) {
      parts.push(canonicalize(value[index], `${path}[${index}]`));
    }
    return `${value.constructor.name}[${parts.join(',')}]`;
  }
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => canonicalize(item, `${path}[${index}]`)).join(',')}]`;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) fail(`non_canonical_object:${path}`);
  const keys = Object.keys(value).sort();
  const parts = keys.map((key) => `${JSON.stringify(key)}:${canonicalize(value[key], `${path}.${key}`)}`);
  return `{${parts.join(',')}}`;
}

/**
 * Deterministic content hash of a checkpoint: SHA-256 over the format tag plus
 * the canonical serialization of every own field. Identical content gives an
 * identical id; any change to weights, to search settings, or to any other
 * declared field changes it.
 */
export function checkpointId(checkpoint) {
  if (checkpoint === null || typeof checkpoint !== 'object' || Array.isArray(checkpoint)) {
    fail('invalid_checkpoint');
  }
  return sha256Hex(`${CHECKPOINT_FORMAT}\n${canonicalize(checkpoint, 'checkpoint')}`);
}

/* ------------------------------------------------------------------ *
 * Checkpoint engine adapter
 * ------------------------------------------------------------------ */

function cloneAction(action) {
  return action.kind === 'pawn'
    ? Object.freeze({ kind: 'pawn', to: Object.freeze({ r: action.to.r, c: action.to.c }) })
    : Object.freeze({ kind: 'wall', wall: action.wall });
}

/**
 * Wrap a checkpoint as a strength-harness engine descriptor
 * (`{ id, version, capabilities, createSession(context) }`), so checkpoint
 * evaluation runs the identical protocol as the release gate.
 *
 * Budget mapping (harness `nodeBudget` vs. Gumbel `simulations`)
 * -------------------------------------------------------------
 * The harness's regression mode speaks in *nodes per move*; this agent spends
 * exactly one root-child expansion-and-evaluation per simulation and never more
 * than `simulations` of them per move, so one simulation is charged as one node.
 * `capabilities.simulationsPerMove` publishes the agent's own count.
 * `capabilities.nodeBudget` is `false`: the harness only enforces a reported
 * `stats.nodes` for engines that advertise `true`, and this adapter returns a
 * bare `Action` (which the harness supports) rather than a stats envelope. The
 * budget is therefore enforced here instead — `selectAction` throws if a search
 * ever spends more simulations than `request.limits.nodeBudget` allows, which
 * the harness scores as a forfeit exactly as it would a node-budget violation.
 * That check is live, not decorative: `evaluateCheckpoint` takes the shared
 * budget as an explicit argument and never widens it to fit this agent, so a
 * budget below this agent's own `simulations` forfeits its games here.
 */
export function createGumbelAgent({ checkpoint, evaluate, simulations, maxConsidered, seed }) {
  const digest = checkpointId(checkpoint);
  if (typeof evaluate !== 'function') fail('invalid_evaluator');
  positiveInteger(simulations, 'invalid_simulations');
  positiveInteger(maxConsidered, 'invalid_max_considered');
  uint32(seed, 'invalid_seed');

  // The id covers the checkpoint content *and* the runtime search settings, so
  // two agents over one checkpoint at different budgets are distinguishable to
  // the harness (which keys per-engine telemetry by id).
  const runtime = sha256Hex(`${digest}|${simulations}|${maxConsidered}|${seed}`).slice(0, 16);
  const id = `${CHECKPOINT_FORMAT}:${digest.slice(0, 16)}:${runtime}`;

  const descriptor = {
    id,
    version: CHECKPOINT_FORMAT,
    checkpointId: digest,
    capabilities: Object.freeze({
      nodeBudget: false,
      simulationsPerMove: simulations,
      maxConsidered,
      deterministic: true,
      hardDeadlineIsolation: false
    }),

    createSession(context) {
      if (context === null || typeof context !== 'object') fail('invalid_session_context');
      const side = context.side;
      if (side !== 'A' && side !== 'B') fail('invalid_session_side');
      const config = canonical9x9(context.config);

      // Per-session stream: agent seed + checkpoint + game + side + the seed the
      // harness derived for this session. Same match, same stream; the two sides
      // of a paired opening get different streams because `side` is mixed in.
      const random = createLcg32(seedFromDigest(
        `${CHECKPOINT_FORMAT}|${digest}|${seed}|${simulations}|${maxConsidered}`
        + `|${String(context.gameId)}|${side}|${String(context.seed)}`
      ));

      return Object.freeze({
        selectAction(request) {
          if (request === null || typeof request !== 'object') fail('invalid_request');
          // The request and everything in it belong to the harness: read only.
          const requestConfig = canonical9x9(request.config ?? config);
          const state = validateState(requestConfig, request.state);

          const result = gumbelRootSearch({
            config: requestConfig, state, evaluate, simulations, maxConsidered, random
          });

          // 100% legality is a Stage 4 exit gate: checked against the engine for
          // the exact state played from, never assumed.
          const legal = legalActionCodes(requestConfig, state);
          if (!legal.includes(result.actionCode)) fail('illegal_agent_action');

          const budget = request.limits?.nodeBudget;
          if (typeof budget === 'number' && result.simulationsUsed > budget) {
            fail('node_budget_exceeded');
          }

          return cloneAction(decodeAction(requestConfig, result.actionCode));
        },
        observe() {},
        close() {}
      });
    }
  };

  return Object.freeze(descriptor);
}

/* ------------------------------------------------------------------ *
 * Paired-opening checkpoint evaluation
 * ------------------------------------------------------------------ */

function assertDescriptor(engine, name) {
  if (!engine || typeof engine !== 'object'
    || typeof engine.id !== 'string' || engine.id.length === 0
    || typeof engine.version !== 'string' || engine.version.length === 0
    || typeof engine.createSession !== 'function') {
    fail(`invalid_engine:${name}`);
  }
  return engine;
}

/**
 * The shared per-move node budget for a regression-mode evaluation.
 *
 * There is no default. A budget that is not published is not guessed: a missing
 * count used to collapse to `1`, which handed the *baseline* (the side that
 * advertises `capabilities.nodeBudget: true` and is therefore policed by
 * `runMatch`) an unmeetable budget while the candidate, which polices itself
 * against the same number, was left with its own full simulation count. Every
 * baseline move became a `node_budget` forfeit and every checkpoint promoted.
 *
 * So: an explicit `nodeBudget` wins, otherwise the baseline must publish a
 * usable `capabilities.simulationsPerMove` and the budget is the larger of the
 * two published counts. Anything else throws `baseline_node_budget_unknown` —
 * the candidate's own number is never adopted as the shared budget.
 */
function resolveNodeBudget({ candidate, baseline, nodeBudget }) {
  if (nodeBudget !== undefined) return positiveInteger(nodeBudget, 'invalid_node_budget');
  const published = baseline.capabilities?.simulationsPerMove;
  if (!Number.isSafeInteger(published) || published < 1) fail('baseline_node_budget_unknown');
  const own = candidate.capabilities?.simulationsPerMove;
  if (!Number.isSafeInteger(own) || own < 1) fail('candidate_node_budget_unknown');
  return Math.max(own, published);
}

/**
 * Play `openingLimit` book openings as *pairs*: for every opening the candidate
 * plays once as A and once as B against the same baseline, so a side bias in the
 * opening cannot be read as strength. Both games of a pair go through the
 * harness's `runMatch`, so adjudication, forfeits (illegal action, crash, null
 * action) and the ply cap are the release gate's, not ours.
 *
 * `winRate` is strict wins over games (draws are not half-credit); the reused
 * `pairedClusterConfidenceIntervals` reports both a strict-win and a
 * draw-as-half score interval under its own `paired-opening-cluster-normal-95-v1`
 * method — no new statistical method is invented here.
 *
 * A forfeit is counted as an ordinary loss for the forfeiting side, which is the
 * harness's own semantics — but it is also *reported*: `failures` (from the
 * harness's own `summarizeEvaluation`) counts every forfeit by reason and
 * `opponentFailures` counts the ones the baseline committed, so a record built
 * out of the opponent's forfeits cannot read as a record built out of play.
 * `promoteCheckpoint` refuses such a record.
 *
 * `nodeBudget` is the shared per-move budget in regression mode; see
 * `resolveNodeBudget`. Strength mode is deadline-scored and passes no node
 * budget to the harness, so none is resolved there.
 */
export async function evaluateCheckpoint({
  config, book, candidate, baseline, openingLimit, seed, nodeBudget, mode = REGRESSION_MODE
}) {
  const checkedConfig = canonical9x9(config);
  assertDescriptor(candidate, 'candidate');
  assertDescriptor(baseline, 'baseline');
  if (candidate.id === baseline.id) fail('candidate_and_baseline_share_an_id');
  uint32(seed, 'invalid_seed');
  if (mode !== REGRESSION_MODE && mode !== STRENGTH_MODE) fail('unsupported_mode');

  const validated = validateOpeningBook(book);
  if (validated.config.rows !== 9 || validated.config.columns !== 9) fail('unsupported_board');
  if (validated.config.rows !== checkedConfig.rows
    || validated.config.columns !== checkedConfig.columns
    || validated.config.firstPlayer !== checkedConfig.firstPlayer) {
    fail('book_config_mismatch');
  }

  const available = validated.book.openings;
  const limit = openingLimit === undefined
    ? available.length
    : positiveInteger(openingLimit, 'invalid_opening_limit');
  const openings = available.slice(0, Math.min(limit, available.length));
  if (openings.length === 0) fail('empty_opening_book');

  // One simulation = one node. The budget is explicit or derived from published
  // counts only — never defaulted (see `resolveNodeBudget`).
  const resolvedNodeBudget = mode === REGRESSION_MODE
    ? resolveNodeBudget({ candidate, baseline, nodeBudget })
    : undefined;

  const results = [];

  for (const opening of openings) {
    for (const contenderSide of ['A', 'B']) {
      const gameInPair = contenderSide === 'A' ? 0 : 1;
      const engines = contenderSide === 'A'
        ? { A: candidate, B: baseline }
        : { A: baseline, B: candidate };
      // Match seed is a pure function of (evaluation seed, opening, side): no
      // counter, no clock, so a re-run of one opening reproduces exactly.
      const matchSeed = seedFromDigest(
        `${CHECKPOINT_FORMAT}|match|${seed}|${opening.id}|${contenderSide}|${gameInPair}`
      );
      // eslint-disable-next-line no-await-in-loop -- games are sequential by design.
      const result = await runMatch({
        config: checkedConfig,
        opening,
        engines,
        contenderSide,
        gameInPair,
        seed: matchSeed,
        mode,
        ...(mode === REGRESSION_MODE ? { nodeBudget: resolvedNodeBudget } : {})
      });
      results.push(result);
    }
  }

  // Wins, losses, draws, side splits, the paired-cluster interval and the
  // per-reason forfeit map all come from the harness's own `summarizeEvaluation`
  // — the module counts nothing itself.
  const harness = summarizeEvaluation(results, {
    contender: candidate, baseline, minimumOpeningPairs: openings.length
  });

  // Which side forfeited is on the harness result (`failedPlayer` plus the
  // engine ids it assigned); `failures` alone does not say. A win handed over by
  // the opponent is not a win, so it has to be countable at promotion time.
  const opponentFailures = results.filter((result) =>
    result.resultKind === 'forfeit'
    && result.engines[result.failedPlayer] !== candidate.id).length;

  return Object.freeze({
    format: CHECKPOINT_FORMAT,
    checkpointId: candidate.checkpointId ?? candidate.id,
    nodeBudget: resolvedNodeBudget ?? null,
    games: harness.games,
    openingPairs: harness.openingPairs,
    wins: harness.wins,
    losses: harness.losses,
    draws: harness.draws,
    winRate: harness.winRate,
    sideSplits: harness.sideSplits,
    failures: harness.failures,
    opponentFailures,
    confidence: harness.confidenceIntervals
  });
}

/* ------------------------------------------------------------------ *
 * Promotion
 * ------------------------------------------------------------------ */

function tallyIsWellFormed(tally) {
  return tally !== null && typeof tally === 'object'
    && Number.isSafeInteger(tally.games) && tally.games >= 0
    && Number.isSafeInteger(tally.wins) && tally.wins >= 0
    && Number.isSafeInteger(tally.losses) && tally.losses >= 0
    && Number.isSafeInteger(tally.draws) && tally.draws >= 0;
}

function summaryIsWellFormed(summary) {
  return summary !== null && typeof summary === 'object'
    && Number.isSafeInteger(summary.games) && summary.games > 0
    && Number.isSafeInteger(summary.openingPairs) && summary.openingPairs >= 0
    && Number.isSafeInteger(summary.wins) && summary.wins >= 0
    && Number.isSafeInteger(summary.losses) && summary.losses >= 0
    && Number.isSafeInteger(summary.draws) && summary.draws >= 0
    && Number.isFinite(summary.winRate)
    // A summary that does not say how many games the opponent forfeited cannot
    // be checked for forfeit-dependence, so it is malformed, not merely clean.
    && Number.isSafeInteger(summary.opponentFailures) && summary.opponentFailures >= 0
    && summary.opponentFailures <= summary.games
    && summary.wins + summary.losses + summary.draws === summary.games
    && summary.openingPairs * 2 === summary.games
    && tallyIsWellFormed(summary.sideSplits?.A)
    && tallyIsWellFormed(summary.sideSplits?.B);
}

/**
 * The promotion rule. Every criterion must hold; `reasons` lists *all* failures,
 * not the first, so one run tells the whole story.
 *
 * Criteria: enough opening pairs evaluated, `winRate >= minimumWinRate`, at
 * least one win on each side — a checkpoint that only ever wins as A has shown a
 * side artefact, not strength — and *no opponent forfeit at all*.
 *
 * Forfeit threshold
 * -----------------
 * Zero. Not "wins minus forfeits still clears the bar": any opponent forfeit
 * (`summary.opponentFailures > 0`) refuses the checkpoint outright, with reason
 * `wins_depend_on_opponent_failures`. A baseline that crashed, ran out of node
 * budget, returned a null action or played an illegal one was not measured
 * against, and the run cannot say which of the *remaining* games were affected
 * by whatever made it fail. Zero is also the threshold that would have caught
 * the budget-collapse bug this rule exists for: there, every win was a
 * `node_budget` forfeit. A run with a genuinely flaky baseline is re-run with a
 * fixed baseline, not promoted on the difference.
 *
 * Fails closed: a missing, malformed or self-inconsistent summary, a bad
 * threshold, or a malformed incumbent is not promoted, with a reason. `incumbent`
 * is optional; when supplied it must be a well-formed summary, and the candidate
 * must not have a lower win rate than it.
 */
export function promoteCheckpoint({ summary, incumbent, minimumWinRate, minimumOpeningPairs }) {
  const reasons = [];

  const validSummary = summaryIsWellFormed(summary);
  if (!validSummary) reasons.push('summary_missing_or_malformed');

  const validRate = Number.isFinite(minimumWinRate) && minimumWinRate >= 0 && minimumWinRate <= 1;
  if (!validRate) reasons.push('minimum_win_rate_missing_or_malformed');

  const validPairs = Number.isSafeInteger(minimumOpeningPairs) && minimumOpeningPairs >= 1;
  if (!validPairs) reasons.push('minimum_opening_pairs_missing_or_malformed');

  const incumbentSupplied = incumbent !== undefined && incumbent !== null;
  const validIncumbent = !incumbentSupplied || summaryIsWellFormed(incumbent);
  if (!validIncumbent) reasons.push('incumbent_malformed');

  if (validSummary && validPairs && summary.openingPairs < minimumOpeningPairs) {
    reasons.push('insufficient_opening_pairs');
  }
  if (validSummary && validRate && !(summary.winRate >= minimumWinRate)) {
    reasons.push('win_rate_below_minimum');
  }
  if (validSummary && summary.opponentFailures > 0) {
    reasons.push('wins_depend_on_opponent_failures');
  }
  if (validSummary && summary.sideSplits.A.wins < 1) reasons.push('no_win_as_a');
  if (validSummary && summary.sideSplits.B.wins < 1) reasons.push('no_win_as_b');
  if (validSummary && incumbentSupplied && validIncumbent && summary.winRate < incumbent.winRate) {
    reasons.push('win_rate_below_incumbent');
  }

  return Object.freeze({ promoted: reasons.length === 0, reasons: Object.freeze(reasons) });
}
