/**
 * Stage 4 checkpoint identity, evaluation and promotion tests.
 *
 * Both agents in the match tests use `uniformStubEvaluator`, which carries no
 * learned knowledge. A stub-vs-stub match is a PLUMBING CHECK — determinism,
 * legality, pairing and bookkeeping. It is not a strength measurement, and no
 * number produced here says anything about how strong anything is.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CHECKPOINT_FORMAT,
  CheckpointError,
  checkpointId,
  createGumbelAgent,
  evaluateCheckpoint,
  promoteCheckpoint
} from '../js/normal-duel-checkpoints.mjs';
import { uniformStubEvaluator } from '../js/normal-duel-gumbel-search.mjs';
import {
  applyAction,
  createInitialState,
  decodeAction,
  encodeAction,
  legalActionCodes,
  validateConfig
} from '../js/normal-duel-engine.mjs';
import { assertEngineDescriptor } from '../scripts/evaluation/normal-duel-strength.mjs';
import { generateBalancedOpeningBook } from '../scripts/generate-normal-duel-balanced-openings.mjs';

const BOOK = generateBalancedOpeningBook();
const CONFIG = validateConfig(BOOK.config);

const CHECKPOINT = Object.freeze({
  name: 'stage4-stub',
  step: 1000,
  weights: [0.25, -0.5, 0.125],
  search: Object.freeze({ simulations: 8, maxConsidered: 4 })
});

function agent(seed, overrides = {}) {
  return createGumbelAgent({
    checkpoint: CHECKPOINT,
    evaluate: uniformStubEvaluator,
    simulations: 8,
    maxConsidered: 4,
    seed,
    ...overrides
  });
}

/**
 * Wrap a descriptor so every (state, action) decision is recorded, letting a
 * test replay the whole game through the engine instead of trusting a record.
 */
function recording(engine, log, role = 'engine') {
  return {
    ...engine,
    createSession(context) {
      const session = engine.createSession(context);
      return {
        selectAction(request) {
          const action = session.selectAction(request);
          log.push({
            role,
            gameId: context.gameId,
            side: context.side,
            positionKey: request.state.positionKey,
            ply: request.state.ply,
            action
          });
          return action;
        },
        observe(transition) { return session.observe(transition); },
        close() { return session.close(); }
      };
    }
  };
}

test('checkpointId is stable, content-sensitive, and key-order insensitive', () => {
  const id = checkpointId(CHECKPOINT);
  assert.match(id, /^[0-9a-f]{64}$/);
  assert.equal(id, checkpointId(structuredClone(CHECKPOINT)));

  // Key insertion order must not matter.
  const reordered = {
    search: { maxConsidered: 4, simulations: 8 },
    weights: [0.25, -0.5, 0.125],
    step: 1000,
    name: 'stage4-stub'
  };
  assert.equal(checkpointId(reordered), id);

  // A Float32Array weight blob agrees with the equivalent plain array.
  assert.equal(
    checkpointId({ ...CHECKPOINT, weights: [1, 2] }),
    checkpointId({ ...CHECKPOINT, weights: [1, 2] })
  );

  // Any change to weights or to search settings changes the id.
  assert.notEqual(checkpointId({ ...CHECKPOINT, weights: [0.25, -0.5, 0.126] }), id);
  assert.notEqual(checkpointId({ ...CHECKPOINT, search: { simulations: 9, maxConsidered: 4 } }), id);
  assert.notEqual(checkpointId({ ...CHECKPOINT, search: { simulations: 8, maxConsidered: 5 } }), id);
  assert.notEqual(checkpointId({ ...CHECKPOINT, step: 1001 }), id);

  // Fails closed on values it cannot canonically represent.
  assert.throws(() => checkpointId(null), CheckpointError);
  assert.throws(() => checkpointId([1, 2]), CheckpointError);
  assert.throws(() => checkpointId({ ...CHECKPOINT, hook: () => 1 }), CheckpointError);
  assert.throws(() => checkpointId({ ...CHECKPOINT, loss: Number.NaN }), CheckpointError);
});

test('the agent is a valid engine descriptor and rejects a 7x7 board', async () => {
  const candidate = agent(1);
  assert.equal(assertEngineDescriptor(candidate, 'candidate'), candidate);
  assert.equal(candidate.version, CHECKPOINT_FORMAT);
  assert.ok(candidate.id.includes(checkpointId(CHECKPOINT).slice(0, 16)));
  assert.equal(candidate.checkpointId, checkpointId(CHECKPOINT));
  assert.equal(candidate.capabilities.simulationsPerMove, 8);

  const session = candidate.createSession({
    gameId: 'x/0', side: 'A', config: CONFIG, seed: 7
  });
  assert.equal(typeof session.selectAction, 'function');
  assert.equal(session.observe({}), undefined);
  assert.equal(session.close(), undefined);

  const small = validateConfig({
    ...BOOK.config,
    rows: 7,
    columns: 7,
    start: { A: { r: 6, c: 3 }, B: { r: 0, c: 3 } },
    goalRows: { A: 0, B: 6 }
  });
  assert.throws(
    () => candidate.createSession({ gameId: 'x/0', side: 'A', config: small, seed: 7 }),
    /unsupported_board/
  );
  await assert.rejects(
    () => evaluateCheckpoint({
      config: small, book: BOOK, candidate, baseline: agent(2), openingLimit: 1, seed: 5
    }),
    /unsupported_board/
  );

  assert.throws(() => createGumbelAgent({
    checkpoint: CHECKPOINT, evaluate: uniformStubEvaluator, simulations: 0, maxConsidered: 4, seed: 1
  }), /invalid_simulations/);
  assert.throws(() => createGumbelAgent({
    checkpoint: CHECKPOINT, evaluate: null, simulations: 8, maxConsidered: 4, seed: 1
  }), /invalid_evaluator/);
});

test('paired evaluation plays legal, deterministic, well-reconciled games', async () => {
  const log = [];
  const candidate = recording(agent(11), log);
  const baseline = recording(agent(22), log);

  const before = structuredClone({ config: CONFIG, book: BOOK, checkpoint: CHECKPOINT });
  const summary = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate, baseline, openingLimit: 3, seed: 4242
  });

  // Purity: nothing the caller handed in was mutated.
  assert.deepEqual(structuredClone({ config: CONFIG, book: BOOK, checkpoint: CHECKPOINT }), before);

  assert.equal(Object.isFrozen(summary), true);
  assert.equal(summary.openingPairs, 3);
  assert.equal(summary.games, 6);
  assert.equal(summary.openingPairs * 2, summary.games);
  assert.equal(summary.wins + summary.losses + summary.draws, summary.games);
  assert.equal(summary.winRate, summary.wins / summary.games);
  assert.equal(summary.checkpointId, checkpointId(CHECKPOINT));

  const { A, B } = summary.sideSplits;
  assert.equal(A.games, 3);
  assert.equal(B.games, 3);
  assert.equal(A.games + B.games, summary.games);
  assert.equal(A.wins + B.wins, summary.wins);
  assert.equal(A.losses + B.losses, summary.losses);
  assert.equal(A.draws + B.draws, summary.draws);
  assert.equal(A.wins + A.losses + A.draws, A.games);
  assert.equal(B.wins + B.losses + B.draws, B.games);
  // The interval is the strength module's own paired-cluster method, reused.
  assert.equal(summary.confidence.method, 'paired-opening-cluster-normal-95-v1');

  // Cross-check the hand-kept totals against the reused interval's *own*
  // clustering of the same games: each pair contributes the mean of its two
  // contender points, so mean x clusters x 2 must reproduce the win total (and
  // the draw-as-half score total). A miscount on either side breaks this.
  assert.equal(summary.confidence.winRate.clusters, summary.openingPairs);
  const intervalWins = summary.confidence.winRate.mean * summary.confidence.winRate.clusters * 2;
  assert.ok(
    Math.abs(intervalWins - summary.wins) < 1e-9,
    `interval implies ${intervalWins} wins, summary says ${summary.wins}`
  );
  const intervalScore = summary.confidence.score.mean * summary.confidence.score.clusters * 2;
  assert.ok(Math.abs(intervalScore - (summary.wins + 0.5 * summary.draws)) < 1e-9);

  // Every action both engines played is replayed through the engine from the
  // opening and checked against the engine's own legal set — the record is
  // re-derived, not trusted.
  assert.ok(log.length > 0);
  const byGame = new Map();
  for (const entry of log) {
    if (!byGame.has(entry.gameId)) byGame.set(entry.gameId, []);
    byGame.get(entry.gameId).push(entry);
  }
  assert.equal(byGame.size, 6);
  let replayedPlies = 0;
  for (const [gameId, entries] of byGame) {
    const opening = BOOK.openings.find((item) => gameId.startsWith(`${item.id}/`));
    assert.ok(opening, `unknown opening for ${gameId}`);
    let state = createInitialState(CONFIG);
    for (const code of opening.actionCodes) {
      state = applyAction(CONFIG, state, decodeAction(CONFIG, code));
    }
    for (const entry of entries) {
      assert.equal(state.outcome.kind, 'ongoing', `${gameId} continued past a terminal state`);
      assert.equal(state.positionKey, entry.positionKey, `${gameId} diverged at ply ${entry.ply}`);
      assert.equal(state.ply, entry.ply);
      assert.equal(state.position.turn, entry.side);
      const code = encodeAction(CONFIG, entry.action);
      assert.ok(
        legalActionCodes(CONFIG, state).includes(code),
        `${gameId} played an illegal action at ply ${entry.ply}`
      );
      state = applyAction(CONFIG, state, entry.action);
      replayedPlies += 1;
    }
  }
  assert.ok(replayedPlies >= 6);

  // Determinism: identical inputs and seed give an identical summary.
  const repeat = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate: agent(11), baseline: agent(22), openingLimit: 3, seed: 4242
  });
  assert.deepEqual(repeat, summary);

  // A different seed is allowed to differ; it must still reconcile.
  const other = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate: agent(11), baseline: agent(22), openingLimit: 3, seed: 99
  });
  assert.equal(other.games, 6);
  assert.equal(other.wins + other.losses + other.draws, other.games);
});

/**
 * Replay every logged game through the engine and derive, independently of the
 * summary, which seat the candidate held and how that game ended. The engine
 * terminates every line itself (goal, threefold repetition or the ply cap), so
 * a completed forfeit-free game always replays to a terminal outcome.
 */
function traceSideRecord(log, config) {
  const byGame = new Map();
  for (const entry of log) {
    if (!byGame.has(entry.gameId)) byGame.set(entry.gameId, []);
    byGame.get(entry.gameId).push(entry);
  }
  const record = {
    A: { games: 0, wins: 0, losses: 0, draws: 0 },
    B: { games: 0, wins: 0, losses: 0, draws: 0 }
  };
  for (const [gameId, entries] of byGame) {
    const candidateSide = entries.find((entry) => entry.role === 'candidate')?.side;
    assert.ok(candidateSide, `no candidate move recorded for ${gameId}`);
    const opening = BOOK.openings.find((item) => gameId.startsWith(`${item.id}/`));
    let state = createInitialState(config);
    for (const code of opening.actionCodes) {
      state = applyAction(config, state, decodeAction(config, code));
    }
    for (const entry of entries) state = applyAction(config, state, entry.action);
    assert.notEqual(state.outcome.kind, 'ongoing', `${gameId} did not reach a terminal outcome`);
    const tally = record[candidateSide];
    tally.games += 1;
    if (state.outcome.kind === 'draw') tally.draws += 1;
    else if (state.outcome.winner === candidateSide) tally.wins += 1;
    else tally.losses += 1;
  }
  return record;
}

test('sideSplits report the seat the candidate actually held', async () => {
  const log = [];
  const candidate = recording(agent(11), log, 'candidate');
  const baseline = recording(agent(22), log, 'baseline');

  const summary = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate, baseline, openingLimit: 4, seed: 4242
  });

  // The trace below is only valid for games decided by play, not by forfeit.
  assert.equal(summary.opponentFailures, 0);
  assert.deepEqual(summary.failures, {});

  const traced = traceSideRecord(log, CONFIG);

  // Guard the guard: if the two seats produced the same record, swapping the
  // index would be invisible and the assertions below would prove nothing.
  assert.notDeepEqual(
    traced.A, traced.B,
    'seats produced identical records; this fixture cannot detect a swapped index'
  );

  for (const side of ['A', 'B']) {
    const split = summary.sideSplits[side];
    assert.equal(split.games, traced[side].games, `${side}.games`);
    assert.equal(split.wins, traced[side].wins, `${side}.wins`);
    assert.equal(split.losses, traced[side].losses, `${side}.losses`);
    assert.equal(split.draws, traced[side].draws, `${side}.draws`);
  }
  assert.equal(summary.wins, traced.A.wins + traced.B.wins);
  assert.equal(summary.losses, traced.A.losses + traced.B.losses);
});

test('evaluation fails closed on a shared engine id and an empty limit', async () => {
  const one = agent(11);
  await assert.rejects(
    () => evaluateCheckpoint({
      config: CONFIG, book: BOOK, candidate: one, baseline: one, openingLimit: 1, seed: 1
    }),
    /candidate_and_baseline_share_an_id/
  );
  await assert.rejects(
    () => evaluateCheckpoint({
      config: CONFIG, book: BOOK, candidate: agent(1), baseline: agent(2), openingLimit: 0, seed: 1
    }),
    /invalid_opening_limit/
  );
});

/**
 * Regression test for the promotion-gate escape: the shipped baseline
 * (`scripts/evaluation/normal-duel-wasm-candidate-adapter.mjs`) advertises
 * `capabilities.nodeBudget: true` and publishes no `simulationsPerMove`. A
 * missing count used to default to 1, collapsing the shared budget to the
 * candidate's own number and forfeiting every baseline move on `node_budget` —
 * a clean sweep and an automatic promotion for any checkpoint at all. An
 * unpublished budget must now stop the evaluation, not produce a record.
 */
test('an unpublished baseline node budget stops the evaluation', async () => {
  const opaqueBaseline = {
    ...agent(22),
    id: 'opaque-baseline',
    capabilities: Object.freeze({ nodeBudget: true, deadline: true })
  };

  await assert.rejects(
    () => evaluateCheckpoint({
      config: CONFIG, book: BOOK, candidate: agent(11), baseline: opaqueBaseline,
      openingLimit: 2, seed: 4242
    }),
    (error) => {
      assert.ok(error instanceof CheckpointError);
      assert.equal(error.reason, 'baseline_node_budget_unknown');
      return true;
    }
  );

  // An explicit budget is the honest way to evaluate such a baseline, and it is
  // not allowed to be nonsense either.
  await assert.rejects(
    () => evaluateCheckpoint({
      config: CONFIG, book: BOOK, candidate: agent(11), baseline: opaqueBaseline,
      openingLimit: 2, seed: 4242, nodeBudget: 0
    }),
    /invalid_node_budget/
  );

  // Supplied explicitly, the same pairing runs and is scored normally.
  const summary = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate: agent(11), baseline: opaqueBaseline,
    openingLimit: 2, seed: 4242, nodeBudget: 8
  });
  assert.equal(summary.nodeBudget, 8);
  assert.equal(summary.games, 4);
});

test('a budget below the candidate\'s own simulation count forfeits its games', async () => {
  // The overspend check inside the adapter is reachable precisely because the
  // shared budget is no longer widened to fit the candidate.
  const session = agent(11).createSession({
    gameId: 'x/0', side: 'A', config: CONFIG, seed: 7
  });
  const state = createInitialState(CONFIG);
  assert.throws(
    () => session.selectAction({ config: CONFIG, state, limits: { nodeBudget: 1 } }),
    /node_budget_exceeded/
  );
  // The same request without the crushing limit is fine.
  assert.ok(session.selectAction({ config: CONFIG, state, limits: { nodeBudget: 8 } }));

  // End to end: the harness scores those throws as forfeits, and they show up
  // on the summary rather than disappearing into the win column.
  const summary = await evaluateCheckpoint({
    config: CONFIG, book: BOOK, candidate: agent(11), baseline: agent(22),
    openingLimit: 2, seed: 4242, nodeBudget: 1
  });
  assert.equal(summary.games, 4);
  assert.equal(summary.wins + summary.losses + summary.draws, 4);
  assert.equal(
    Object.values(summary.failures).reduce((sum, count) => sum + count, 0), 4,
    'every game should have ended in a forfeit'
  );
  assert.equal(
    promoteCheckpoint({ summary, minimumWinRate: 0, minimumOpeningPairs: 1 }).promoted,
    false
  );
});

function summaryOf({ aWins = 1, bWins = 1, pairs = 4, opponentFailures = 0 } = {}) {
  const games = pairs * 2;
  const wins = aWins + bWins;
  return {
    games,
    openingPairs: pairs,
    wins,
    losses: games - wins,
    draws: 0,
    winRate: wins / games,
    opponentFailures,
    sideSplits: {
      A: { games: pairs, wins: aWins, losses: pairs - aWins, draws: 0 },
      B: { games: pairs, wins: bWins, losses: pairs - bWins, draws: 0 }
    }
  };
}

test('promotion requires pairs, win rate, and a win on both sides', () => {
  const good = summaryOf({ aWins: 3, bWins: 3, pairs: 4 }); // 6/8 = 0.75
  const promoted = promoteCheckpoint({
    summary: good, minimumWinRate: 0.55, minimumOpeningPairs: 4
  });
  assert.deepEqual(promoted, { promoted: true, reasons: [] });
  assert.equal(Object.isFrozen(promoted), true);
  assert.equal(Object.isFrozen(promoted.reasons), true);

  // Exactly at the threshold promotes.
  assert.equal(promoteCheckpoint({
    summary: good, minimumWinRate: 0.75, minimumOpeningPairs: 4
  }).promoted, true);

  // Too few pairs.
  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 2, bWins: 2, pairs: 2 }), minimumWinRate: 0.5, minimumOpeningPairs: 8
  }), { promoted: false, reasons: ['insufficient_opening_pairs'] });

  // Low win rate.
  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 1, bWins: 1, pairs: 4 }), minimumWinRate: 0.6, minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['win_rate_below_minimum'] });

  // One-sided record: wins only as A.
  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 4, bWins: 0, pairs: 4 }), minimumWinRate: 0.4, minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['no_win_as_b'] });

  // Every failing criterion is reported at once.
  const many = promoteCheckpoint({
    summary: summaryOf({ aWins: 0, bWins: 0, pairs: 2 }), minimumWinRate: 0.6, minimumOpeningPairs: 8
  });
  assert.equal(many.promoted, false);
  assert.deepEqual(
    many.reasons,
    ['insufficient_opening_pairs', 'win_rate_below_minimum', 'no_win_as_a', 'no_win_as_b']
  );

  // A record that leans on the opponent's forfeits is refused outright — the
  // threshold is zero, so even one forfeit in an otherwise dominant run blocks
  // promotion.
  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 4, bWins: 4, pairs: 4, opponentFailures: 8 }),
    minimumWinRate: 0.5,
    minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['wins_depend_on_opponent_failures'] });
  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 4, bWins: 3, pairs: 4, opponentFailures: 1 }),
    minimumWinRate: 0.5,
    minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['wins_depend_on_opponent_failures'] });

  // A summary that cannot report opponent forfeits cannot be cleared of them.
  const { opponentFailures, ...silent } = summaryOf({ aWins: 3, bWins: 3 });
  assert.equal(opponentFailures, 0);
  assert.deepEqual(promoteCheckpoint({
    summary: silent, minimumWinRate: 0.5, minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['summary_missing_or_malformed'] });

  // A weaker candidate does not displace a recorded incumbent.
  assert.deepEqual(promoteCheckpoint({
    summary: good,
    incumbent: summaryOf({ aWins: 4, bWins: 4, pairs: 4 }),
    minimumWinRate: 0.5,
    minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['win_rate_below_incumbent'] });
});

test('promotion fails closed on malformed input', () => {
  assert.deepEqual(promoteCheckpoint({ minimumWinRate: 0.5, minimumOpeningPairs: 4 }), {
    promoted: false, reasons: ['summary_missing_or_malformed']
  });
  assert.equal(promoteCheckpoint({}).promoted, false);
  assert.deepEqual(promoteCheckpoint({}).reasons, [
    'summary_missing_or_malformed',
    'minimum_win_rate_missing_or_malformed',
    'minimum_opening_pairs_missing_or_malformed'
  ]);

  // Internally inconsistent summaries are malformed, not merely unlucky.
  const inconsistent = { ...summaryOf(), games: 9 };
  assert.deepEqual(promoteCheckpoint({
    summary: inconsistent, minimumWinRate: 0.5, minimumOpeningPairs: 1
  }), { promoted: false, reasons: ['summary_missing_or_malformed'] });

  assert.deepEqual(promoteCheckpoint({
    summary: summaryOf({ aWins: 3, bWins: 3 }),
    incumbent: { games: 2 },
    minimumWinRate: 0.5,
    minimumOpeningPairs: 4
  }), { promoted: false, reasons: ['incumbent_malformed'] });

  assert.equal(promoteCheckpoint({
    summary: summaryOf({ aWins: 3, bWins: 3 }), minimumWinRate: 2, minimumOpeningPairs: 4
  }).promoted, false);
});
