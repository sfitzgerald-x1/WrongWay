# WrongWay AI Engine Plan

## Status

- **Scope:** Two-player 2D WrongWay, beginning with 9×9 Duel and keeping
  board size, starting positions, goal rows, and barricade stock configurable.
- **Decision:** Build a rules-faithful classical engine first, then use it as
  the foundation and benchmark for learned search.
- **Recommended deployment path:** JavaScript reference engine → native/WASM
  alpha-beta engine → optional Gumbel AlphaZero training → measured hybrid.
- **Non-goal:** Claiming that full 9×9, ten-barricade WrongWay is solved.

## Executive decision

The project should follow a staged hybrid strategy:

1. Extract one authoritative, parameterized rules engine and prove that the
   client, AI, and headless simulations agree with it.
2. Make games finite with an explicit threefold-repetition draw and a defensive
   ply cap.
3. Build a fast bitboard alpha-beta engine with a transposition table and an
   exact solver for positions in which both players have exhausted their
   barricades.
4. Ship that engine behind a new difficulty level once it decisively beats the
   current Hard bot.
5. Only then train a policy-and-value network with Gumbel AlphaZero-style
   self-play, first on 7×7 to validate the pipeline and then on 9×9.
6. Choose the final browser search architecture from measured
   strength-per-millisecond, potentially distilling the learned model into the
   alpha-beta engine.

This path produces a useful result even if machine learning never ships. It
also prevents a learned agent from silently training against rules that differ
from the product.

## Why this order is necessary

### The current rules are duplicated

The 1v1 jump rule appears independently in the human move validator in
[`index.html`](../index.html#L6382) and the AI move generator in
[`js/ai.js`](../js/ai.js#L140). Both implement WrongWay's permissive jump:
when the opponent is adjacent through an open edge, the mover may use any
unblocked exit from the opponent's cell other than the square it came from.
Side exits therefore remain legal even when the straight exit is open.

The 2v2 implementation in
[`js/game-logic.js`](../js/game-logic.js#L78) instead follows conventional
Quoridor behavior: take the straight jump when available and allow side exits
only when it is blocked. The 1v1 engine must preserve the former behavior; 2v2
should remain outside the first extraction unless its rule is deliberately
changed.

Wall/path rules are also coupled to browser globals and spread across
[`js/game-logic.js`](../js/game-logic.js#L14),
[`js/ai.js`](../js/ai.js), and inline client code. An AI port made before this
duplication is removed could be fast, deterministic, and wrong.

### The current Hard bot omits strategic wall classes

The Hard search generates walls from candidates near the opponent's path in
the two `genActions` implementations in
[`js/ai.js`](../js/ai.js#L449) and
[`js/ai.js`](../js/ai.js#L606). Defensive walls near the bot's own path,
prophylactic walls, and zero-immediate-gain tempo walls can be absent from the
search. This is a larger limitation than the nominal search depth.

The replacement engine should use wall scoring for **ordering**, not permanent
deletion. Proof and correctness modes must always generate the complete legal
action set.

### Draw behavior is undefined

The client currently models a winner but no draw, and uses an AI anti-stall
override in [`index.html`](../index.html#L4526) and
[`index.html`](../index.html#L5635). Search values and self-play targets are not
well-defined until cycling has a rules-level outcome.

## Rules contract

Stage 0 should publish a versioned rules contract and fixtures before any fast
engine is written.

### Configuration

The pure engine accepts an explicit configuration rather than browser globals:

```text
rows, columns
starting position for A and B
goal row for A and B
initial barricade stock for A and B
jump rule identifier
repetition threshold
ply cap
```

The first supported configurations are 9×9 Duel and 7×7 Blitz. Existing 1v1
variants may use the same core when their configuration and special item rules
are explicitly represented. The 2v2 rules are not implicitly folded into this
contract.

### Position and game state

The position key contains:

```text
pawn A square
pawn B square
horizontal wall anchors
vertical wall anchors
remaining stock for A
remaining stock for B
side to move
rules/configuration identifier
```

Stock cannot be inferred from placed walls because wall ownership is not
recorded. History is kept outside the position key for repetition adjudication.
Wall placement is irreversible, so repetition history may be reset after every
wall placement.

### Actions

For 9×9, use the fixed 209-action encoding:

- `0..80`: pawn destination square
- `81..144`: horizontal wall anchor
- `145..208`: vertical wall anchor

A legal-action mask distinguishes legal actions from the full policy space.
Destination encoding is unambiguous because a pawn move has exactly one
resulting square. Smaller boards use the same three contiguous action classes
with configuration-derived offsets.

### Wall legality

A wall is legal only when:

1. its anchor and orientation are in bounds;
2. it does not overlap or cross an existing wall under current geometry rules;
3. both players retain a wall-only path to their goal rows.

Pawn locations do not participate in the connectivity check. This matches the
existing 1v1 behavior in
[`js/game-logic.js`](../js/game-logic.js#L167).

### Terminal outcomes

The engine returns an outcome and an explicit reason:

```text
ongoing
A win: goal | timeout | forfeit
B win: goal | timeout | forfeit
draw: threefold repetition | ply cap
```

Adopt threefold repetition of the full position key. The search checks both
the actual game history and the current search path. Add a 200-ply cap for 9×9
as a defensive backstop, configurable for other variants.

Superko is not recommended: making repeated positions illegal adds
history-dependent move legality, complicates transposition tables, and gives a
policy/value model a less natural state representation.

## Target architecture

```text
                      ┌─────────────────────┐
                      │ Versioned rules spec │
                      └──────────┬──────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Pure JS reference engine │
                    └──────┬───────────┬──────┘
                           │           │
                    browser client  golden JSONL
                                       │
                            ┌──────────▼──────────┐
                            │ Rust native/WASM core│
                            └──────┬────────┬─────┘
                                   │        │
                              alpha-beta  self-play
                                   │        │
                              Hard+ WASM  policy/value net
                                   └────┬───┘
                                        │
                              measured hybrid target
```

The JavaScript engine is the behavioral reference, not the performance target.
It should expose pure functions for initial state creation, legal actions,
apply/undo or immutable transition, path queries, terminal evaluation, position
serialization, and repetition tracking. A thin browser adapter may expose
these functions to the current Babel client, but must contain no game rules.

The Rust crate should compile to:

- a native library/binary for search, solving, self-play, and tournaments;
- WASM for the shipped client;
- deterministic corpus/perft tools used in CI.

## Delivery stages

### Stage 0 — Rules formalization and a single source of truth

**Work**

- Document the 1v1 permissive jump rule with positive and negative examples.
- Add the threefold-repetition and ply-cap outcomes to the contract.
- Extract a pure, parameterized JavaScript engine.
- Replace the inline human validator and AI move generator with calls into it.
- Add a headless Node harness.
- Generate deterministic JSONL trajectories containing configuration, state,
  legal action set, selected action, next state, and outcome.
- Add perft-style legal-node counts to depth 3–4 across curated and randomized
  positions.
- Keep 2v2 on its existing rules path until separately migrated.

**Exit gate**

- No independent 1v1 move or wall legality implementation remains.
- The existing client produces the same legal moves and wall decisions for the
  pre-extraction golden corpus.
- Tests lock in permissive side exits, edge/corner jumps, wall intersection,
  path preservation, stock accounting, terminal goals, repetition, and ply cap.

### Stage 1 — Rust core and parity

**Work**

- Encode horizontal and vertical 8×8 wall-anchor grids in two `u64` values for
  9×9.
- Precompute the board edges blocked by every wall anchor.
- Implement bit-parallel flood fill and shortest-distance maps.
- Implement fast apply/undo and full legal move generation.
- Use Zobrist hashing over pawns, walls, stock, side, and rules configuration.
- Canonicalize the left/right mirror symmetry where applicable; separately
  evaluate row-flip plus player-swap normalization.
- Add a wall-cut fast path so full connectivity BFS runs only when necessary.
- Replay every JavaScript golden trajectory and compare every legal action and
  transition.

**Exit gate**

- Bit-exact parity with the JavaScript corpus and identical perft counts.
- Sanitizer/property tests pass for randomized apply/undo sequences.
- Benchmarks demonstrate at least a 100× throughput improvement over the
  JavaScript reference on the agreed machine.

### Stage 2 — Strong classical search

**Work**

- Iterative deepening with a hard wall-clock deadline.
- Principal variation search and aspiration windows.
- A bounded transposition table with bound type, depth, score, and best action.
- TT, killer, counter-move, and history ordering.
- Staged generation: pawn moves and highest-scored walls first, then the
  complete remaining wall set if no cutoff occurs.
- Evaluation v1:
  - shortest-path distance difference;
  - remaining stock difference;
  - side-to-move tempo/parity;
  - alternate-path robustness;
  - wall-chain and mobility features only when validated by ablation.
- Repetition-aware search.
- Exact evaluation when both players have zero barricades.

Wall ordering must include:

- walls that delay the opponent;
- walls that defend the engine's route;
- extensions of existing chains;
- prophylactic walls near high-value opponent placements;
- legal zero-immediate-gain walls.

**Zero-wall oracle**

For a fixed wall layout with no stock remaining, solve the pawn-position graph
exactly, including the permissive jump and repetition-draw rules. Cache the
result by wall-layout hash. This makes the most common endgame stratum exact
and provides ground truth for tests and future learning.

**Exit gate**

- At least a 90% score against the current Hard bot under paired openings and
  the same move-time budget.
- 100% agreement with exact reduced-position suites.
- No deadline overrun beyond a defined small tolerance in native or WASM builds.
- Full-width verification of all headline match results.

### Stage 3 — Product integration

**Work**

- Compile the search engine to WASM.
- Add it behind a `Hard+` feature flag at the existing client think-time budget.
- Add draw state, messaging, replay storage, online synchronization, and result
  handling.
- Remove the anti-stall override once all active agents honor formal draw rules.
- Add telemetry that records engine version, budget, completed depth, nodes,
  and outcome without storing private player data.

**Exit gate**

- Desktop and representative mobile browser budgets are respected.
- Draws round-trip correctly through local, replay, and supported online modes.
- A rollback to the previous bot requires only disabling the feature flag.

### Stage 4 — Gumbel AlphaZero self-play

This stage starts only after Stage 3 has shipped or met its release gate.

**Model**

- 209-way masked policy head on 9×9.
- Scalar value head with win/draw/loss targets.
- Input planes for both pawns, horizontal walls, vertical walls, remaining
  stock, and side to move.
- Optional shortest-path/distance planes may be tested as a declared inductive
  bias.
- Initial size target: 6–10 residual blocks with 64–96 channels.

**Training**

- Validate the complete loop on 7×7 with 16–64 Gumbel search simulations per
  move.
- Move to 9×9 after deterministic replay, checkpoint promotion, and evaluation
  are reliable.
- Use the full legal-action mask from the authoritative engine.
- Gate checkpoints through paired-opening matches against alpha-beta rather
  than self-play Elo alone.

7×7 is a pipeline test, not a solved stepping stone or a promise of direct
strategy transfer. Ten walls on 7×7 create a different wall density from 9×9.

**Exit gate**

- The learned agent beats Stage 2 alpha-beta at equal wall-clock time with a
  statistically meaningful confidence interval.
- It retains 100% legality and passes the exact reduced-position suite.
- Adversarial wall-defense, corridor, tempo, and repetition probes show no
  regression against the classical baseline.

### Stage 5 — Hybrid deployment and truth track

Choose the final browser design through benchmarks:

- PUCT with the policy/value network;
- alpha-beta with a distilled small value network and policy-based ordering;
- an NNUE-style incrementally updated evaluation;
- classical alpha-beta on constrained devices and learned search elsewhere.

In parallel, expand exact truth data:

- all zero-wall positions encountered in evaluation;
- complete 3×3 and reduced 5×5 variants;
- low-wall 7×7 strata where feasible;
- proof-number search from selected tactical positions.

These are validation assets and exact engine components, not evidence that the
full game is solved.

## Evaluation protocol

### Agent ladder

1. Random legal action
2. Greedy shortest-path mover with no walls
3. Existing Normal bot
4. Existing Hard bot
5. Full-width alpha-beta v1
6. Tuned alpha-beta plus zero-wall oracle
7. Raw learned policy
8. Learned policy/value plus search

### Match design

A deterministic game can reduce a naïve head-to-head comparison to one bit.
Use a shared book of balanced positions after 4–6 opening plies. For each
opening, play a pair with the agents on opposite sides. Report:

- wins, losses, and draws;
- paired score difference;
- first-player and second-player splits;
- Elo estimate with confidence interval;
- average and percentile move time;
- nodes, completed depth, and wall/pawn action mix.

Use SPRT or an equivalent sequential gate for engine promotions. Keep two
non-binding diagnostic suites:

- exact solved positions, which require 100% value/action agreement;
- adversarial probes targeting wall defense, wall chains, corridor traps,
  tempo races, jump edge cases, and repetition.

## Testing and reproducibility

Every engine version and training run should record:

- rules-contract version;
- engine/model commit;
- configuration and action encoding version;
- random seed;
- time or simulation budget;
- opening-book version;
- hardware and relevant compiler/runtime versions.

CI should run, in increasing cost:

1. unit tests for move, wall, hashing, and terminal rules;
2. golden JSONL parity;
3. curated perft positions;
4. randomized apply/undo and symmetry properties;
5. exact-position agreement;
6. a short deterministic ladder smoke test;
7. native and WASM deadline/throughput benchmarks on scheduled runners.

## Risks and controls

| Risk | Control |
| --- | --- |
| Rules drift between UI, AI, native engine, and trainer | Single JS reference, versioned contract, golden trajectories, perft |
| New draw rule changes product behavior | Treat it as a visible rules change; update client, sync, replay, and agents together |
| Wall pruning hides strategically necessary moves | Order all walls; do not delete them in proof mode; re-run headline matches full-width |
| Deterministic matches exaggerate conclusions | Paired opening book, side swaps, confidence intervals, SPRT |
| Hand evaluation plateaus in wall-war positions | Feature ablation, adversarial suites, optional learned policy/value stage |
| ML effort expands without a product result | Make Stage 3 independently shippable; require a measured win over alpha-beta to proceed |
| 7×7 behavior fails to transfer | Use 7×7 only to validate infrastructure; retrain and retune on 9×9 |
| Search and UI disagree under deadlines | Engine-owned deadlines, immutable last-completed iteration, WASM integration tests |

## Practical strength versus proof

The intended claims must remain distinct:

- **Strong practical play:** Stage 2–3 should decisively exceed the current Hard
  bot, play zero-wall races exactly, and fit the browser budget.
- **Near-optimal practical play:** Stage 4–5 may produce a superhuman agent with
  no known exploit under the evaluation suite, but without a proof of optimality.
- **Provably optimal play:** Limit claims to exactly enumerated reduced
  configurations and zero-wall strata. Do not describe full 9×9 or full-stock
  7×7 as solved.

Generalized Quoridor was proved PSPACE-complete in 2026
([Drop, Rin, and van der Velde](https://arxiv.org/abs/2605.22747);
[Carboni and Muscillo](https://arxiv.org/abs/2606.28931)). A reduced 5×5,
one-wall-per-player game has been solved by retrograde analysis
([Iwanaga et al.](https://www.alife-robotics.co.jp/members2022/icarob/data/html/data/OS/OS13/OS13-3.pdf)).
These results support using exact small configurations as test truth while
treating full-game strength as an empirical engineering objective.

## Immediate next PRs

Keep changes reviewable and independently reversible:

1. **Rules contract and fixtures:** examples, state/action serialization,
   repetition semantics, and golden positions without client rewiring.
2. **JavaScript engine extraction:** pure core plus Node tests.
3. **Client/AI migration:** remove duplicated 1v1 validators and demonstrate
   parity.
4. **Draw plumbing:** local client, replay, online protocol, and presentation.
5. **Rust parity core:** native/WASM crate, corpus replay, perft, and benchmarks.
6. **Alpha-beta v1:** search, TT, evaluation, zero-wall oracle, and ladder.

No Rust or ML work should merge before the Stage 0 parity gate is green.
