# WrongWay AI Engine Plan

## Status

- **Scope:** The normal 1v1 Duel ruleset, beginning with 9×9 Duel and 7×7
  Blitz, with Chaos, Hammer, random drops, Classic, 2v2, clocks, and online
  authority treated as explicit follow-up integrations.
- **Decision:** Build a rules-faithful classical engine first, then use it as
  the foundation and benchmark for learned search.
- **Recommended deployment path:** JavaScript reference engine → native/WASM
  alpha-beta engine → optional Gumbel AlphaZero training → measured hybrid.
- **Non-goal:** Claiming that full 9×9, ten-barricade WrongWay is solved.

## Executive decision

The project should follow a staged hybrid strategy:

1. Extract one authoritative, parameterized `normal-duel-v1` rules engine and
   prove that every in-repository consumer migrated to it agrees.
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

There is also a third, partial implementation in `moveTowardGoal` in
[`js/ai.js`](../js/ai.js#L205). It independently chooses the first available
exit when the next path square is occupied and is used by normal AI advancement
and the client anti-stall override. All three paths must be migrated together.

The 2v2 implementation in
[`js/game-logic.js`](../js/game-logic.js#L78) instead follows conventional
Quoridor behavior: take the straight jump when available and allow side exits
only when it is blocked. The 1v1 engine must preserve the former behavior; 2v2
should remain outside the first extraction unless its rule is deliberately
changed.

Core 1v1 wall legality is already shared through `tryWall`, but it is coupled
to mutable browser globals such as `ROWS`, `COLS`, `CUR_MAP`, and `_hamCtx` in
[`js/game-logic.js`](../js/game-logic.js#L159). AI candidate generation and
special-mode policy remain spread through [`js/ai.js`](../js/ai.js) and inline
client code. A port made before these boundaries are explicit could be fast,
deterministic, and wrong.

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
state ref in [`index.html`](../index.html#L4527) with enforcement in
[`index.html`](../index.html#L5639). Search values and self-play targets are not
well-defined until cycling has a rules-level outcome.

### Online authority is outside this repository

The production WebSocket service is authoritative for online geometry, moves,
walls, turns, winners, and ranked results. Its implementation is not present in
this repository. A client-only rules extraction therefore cannot claim global
authority for online matches or safely introduce an online draw result.

Stage 0 establishes one source of truth for the local client, headless harness,
and bots in this repository. Online draw/rules parity is a separate integration
gate requiring the server repository, a versioned protocol, coordinated
deployment, and rollback. Until that gate passes, the new engine must not
silently adjudicate an online result that the server does not recognize.

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
ruleset version
```

The first ruleset is `normal-duel-v1` with special modes and external clocks
disabled. It supports 9×9 Duel and 7×7 Blitz. Classic is not just another Duel
size: it is 9×13, starts both pawns on the bottom row, and gives both players
goal row 0. Chaos, Hammer, random drops, Classic, and 2v2 require separately
versioned contracts or adapters rather than implicit reuse.

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
part of the board position, and special modes can grant stock or inject/remove
walls. The full `GameState` additionally contains the repetition history/counts,
ply count, and terminal outcome/reason. These fields are serialized in corpora
even when they are excluded from the transposition position key.

Within `normal-duel-v1`, wall placement is irreversible, so repetition history
may be reset after every placement. This optimization is invalid for Hammer or
any future ruleset that can remove walls.

### Actions

For 9×9 `normal-duel-v1`, use the fixed 209-action encoding:

- `0..80`: pawn destination square
- `81..144`: horizontal wall anchor
- `145..208`: vertical wall anchor

A legal-action mask distinguishes legal actions from the full policy space.
Destination encoding is unambiguous because a pawn move has exactly one
resulting square. Smaller boards use the same three contiguous action classes
with configuration-derived offsets. Every ongoing normal-duel state must have
at least one legal pawn action, so an all-zero mask is an invariant violation.

The 209 actions intentionally do not encode Hammer wall destruction, stochastic
system drops, item pickup as a separate action, or 2v2 turns. Those rulesets
need distinct versioned action schemas.

### Wall legality

A wall is legal in `normal-duel-v1` only when:

1. its anchor and orientation are in bounds;
2. it does not overlap or cross an existing wall under current geometry rules;
3. both players retain a wall-only path to their goal rows.

Pawn locations do not participate in the normal-duel connectivity check. This
matches the base path check in
[`js/game-logic.js`](../js/game-logic.js#L167).

Hammer is explicitly different: `_hamCtx` can reject a placement based on pawn
reachability to remaining hammers, and steel walls affect wall destruction.
Those mechanics remain on a legacy/special-mode path until their own contract
is implemented.

### Terminal outcomes

The engine returns an outcome and an explicit reason:

```text
ongoing
A win: goal
B win: goal
draw: threefold repetition | ply cap
```

Adopt threefold repetition of the full position key. The search checks both
the actual game history and the current search path. Add a 200-ply cap for 9×9
as a defensive backstop, configurable for other variants.

Timeout, skipped-turn, disconnect, and forfeit results are typed external
adjudication events owned by the clock/client/server layer, not board-rule
outcomes inferred by the pure engine.

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
             local client + bots   golden JSONL
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

```text
authoritative online server (separate repository)
          ⇅ versioned rules/protocol conformance
local client adapter (this repository)
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
- Publish the `normal-duel-v1` scope and explicitly reject special-mode states.
- Add the threefold-repetition and ply-cap outcomes to the contract.
- Extract a pure, parameterized JavaScript engine.
- Replace the inline human validator, `getMovesFrom`, and `moveTowardGoal` with
  calls into it for normal Duel.
- Add a headless Node harness.
- Generate deterministic JSONL trajectories containing configuration, state,
  legal action set, selected action, next state, and outcome.
- Add perft-style legal-node counts to depth 3–4 across curated and randomized
  positions.
- Keep special modes, Classic, 2v2, clocks, and online server authority on
  explicit legacy/integration paths until separately migrated.

**Exit gate**

- No independent `normal-duel-v1` move or wall-legality implementation remains
  among migrated in-repository consumers.
- The local/PVC client produces the same legal moves and wall decisions for the
  pre-extraction golden corpus.
- Tests lock in permissive side exits, edge/corner jumps, wall intersection,
  malformed wall rejection, path preservation, stock accounting, non-empty
  action masks, terminal goals, repetition, and ply cap.
- Unsupported rulesets fail closed rather than silently using normal-duel
  semantics.
- Online behavior remains unchanged until server conformance is independently
  available and verified.

### Stage 1 — Rust core and parity

**Work**

- Encode horizontal and vertical 8×8 wall-anchor grids in two `u64` values for
  9×9.
- Precompute the board edges blocked by every wall anchor.
- Implement bit-parallel flood fill and shortest-distance maps.
- Implement fast apply/undo and full legal move generation.
- Use Zobrist hashing over pawns, walls, stock, side, and rules configuration.
- Canonicalize the left/right mirror symmetry for Duel; separately evaluate
  row-flip plus player-swap normalization only for configurations where starts,
  goals, and rules prove it valid. Do not apply it to Classic.
- Evaluate a wall-cut fast path against the existing disjoint-path references
  in `disjointPaths2`, `aiDisjoint`, and `worstWallDelay`; retain full BFS as the
  correctness reference.
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
- A repetition-safe TT policy: include the necessary repetition context in the
  key or suppress reusable bounds at path-dependent nodes. Exact suites must
  run with a correctness mode that cannot reuse a history-invalid score.
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
result by wall-layout hash. This makes zero-wall endgames exact and provides
ground truth for tests and future learning. Measure their frequency from real
replays before making product-performance claims about the oracle's hit rate.

**Exit gate**

- At least a 90% score against a pinned current-Hard baseline under paired
  openings, injected random seeds, and the same move-time budget on recorded
  hardware.
- 100% agreement with exact reduced-position suites.
- No deadline overrun beyond a defined small tolerance in native or WASM builds.
- Full-width verification of all headline match results.
- Deterministic fixed-node/fixed-depth suites reproduce search regressions
  independently of wall-clock variance.

### Stage 3 — Product integration

**Work**

- Compile the search engine to WASM.
- Add it behind a `Hard+` feature flag at the existing client think-time budget.
- Add draw state, messaging, replay storage, and local result handling.
- Add online draw/rules synchronization only in a coordinated change with the
  authoritative server and a versioned protocol compatibility check.
- Remove the anti-stall override once all active agents honor formal draw rules.
- Add telemetry that records engine version, budget, completed depth, nodes,
  and outcome without storing private player data.

**Exit gate**

- Desktop and representative mobile browser budgets are respected.
- Draws round-trip correctly through local and replay modes.
- Any enabled online mode has passed server/client conformance for the exact
  rules-contract version; otherwise online draw adjudication remains disabled.
- A rollback to the previous bot requires only disabling the feature flag.

### Stage 4 — Gumbel AlphaZero self-play

This stage starts only after Stage 3 has shipped or met its release gate.

**Model**

- 209-way masked policy head on 9×9 `normal-duel-v1`.
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

Run the legacy bots with injected deterministic random seeds. Use fixed
node/depth budgets for reproducible regression tests and pinned-hardware
wall-clock budgets for product-strength claims. Use SPRT or an equivalent
sequential gate for engine promotions. Keep two non-binding diagnostic suites:

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
| Rules drift between UI, AI, native engine, trainer, and online server | Single JS reference for in-repo consumers, versioned server protocol, golden trajectories, perft, cross-service conformance |
| Authoritative server code is unavailable | Keep online adjudication unchanged; do not claim global single-sourcing or enable online draws until the server repository is in scope |
| Special modes invalidate normal-duel assumptions | Version rulesets/action schemas; reject unsupported states; migrate Hammer, Chaos, drops, Classic, and 2v2 separately |
| New draw rule changes product behavior | Treat it as a visible rules change; update local client, replay, agents, and later the server in coordinated gates |
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

1. **Normal-duel rules contract and fixtures:** scope/exclusions, examples,
   state/action serialization, repetition semantics, and golden positions
   without client rewiring.
2. **JavaScript reference core:** pure 9×9/7×7 normal-Duel functions, Node
   tests, and shadow comparisons against existing local Duel.
3. **Consumer migration:** replace `getValidMoves`, `getMovesFrom`,
   `moveTowardGoal`, and normal-Duel `tryWall` call paths; keep special modes on
   explicit legacy dispatch.
4. **Corpus/perft CI:** freeze deterministic trajectories and complete legal
   node counts; add seeded transition and serialization properties.
5. **Local draw plumbing:** engine history, UI, replay, and local agents.
   Online draws require a separate coordinated server/protocol PR in the
   authoritative server repository.
6. **Rust parity core:** native/WASM crate, corpus replay, perft, and benchmarks.
7. **Alpha-beta v1:** search, repetition-safe TT, evaluation, zero-wall oracle,
   deterministic baseline, and ladder.
8. **Separately versioned adapters:** Classic, Chaos, Hammer, random drops, and
   2v2, each with its own contract and corpus.

No Rust or ML work should merge before the Stage 0 parity gate is green.
