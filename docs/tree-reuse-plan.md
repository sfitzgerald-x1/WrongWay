# Tree reuse across moves

## What this is

Today every turn builds a search tree from scratch. `NormalDuelSearch::new` takes a
position and allocates a fresh arena; when the move is played the tree is dropped.
But after the opponent replies, the position we now face was already a node in the
tree we just threw away — a grandchild of its root — and the visits underneath it are
still valid statistics about the same game.

Every serious engine inherits them. We don't. This plan is how we would.

## Read this part before writing any code

The premise is that reuse buys a 2–4× effective increase in visits. **That figure
comes from engines whose search is deep and whose trees hold millions of nodes, and
there is a specific reason to doubt it transfers here.**

Our root does not spend its budget the way a PUCT root does. Gumbel sequential
halving picks `m = min(maxConsidered, legal)` candidates and divides the budget among
them in `ceil(log2(m))` rounds. At the deployed setting — 512 simulations, 12
considered — that is 4 rounds and about 10 first-round visits per candidate, with the
budget concentrating onto survivors. Two things follow:

1. The inherited subtree is rooted at a **grandchild**: our move, then their reply.
   Its visit count is bounded by the visits our search gave that specific
   continuation, and the halving schedule spends almost everything on a handful of
   root candidates. If the opponent replies with a move our search had already
   discarded, the inheritance is close to zero.
2. A human opponent on the play site is *especially* likely to leave the inherited
   set, because they are not choosing from our candidate list at all.

So the honest expectation is a saving well under 2×, possibly small enough not to
justify a wasm API change. That is a claim we can settle for a few hours of work
instead of arguing about, which is why step 0 exists and why nothing else starts
until it reports.

## Step 0 RESULT — measured, and the verdict is "self-play only"

Done. `scripts/tree-reuse-probe.mjs`, driving the real `d3-iter-150` network, with
`PuctTreeSearch::subtree_visits_after` behind `NormalDuelSearch.subtreeVisitsAfter`.

| budget | transitions | p10 | p25 | **p50** | p75 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 sims / 12 considered | 125 | 0.2% | 8.8% | **23.4%** | 28.9% | 34.8% | 49.8% |
| 128 sims / 12 considered | 126 | 0.0% | 7.0% | **20.3%** | 32.8% | 34.4% | 49.2% |

**Verdict by the rule below: 10–25% → build it for self-play, not for the play site.**
23.4% sits just under the boundary, and it is not rounded up: the rule was written down
in advance precisely so a number this close to the line would not be argued across it.

Two things the measurement corrected in this plan's own reasoning:

- **The predicted failure did not happen.** The plan expected the reply to often be
  absent from the tree entirely. It never was — 0% of transitions in both runs. The
  reason is a mistake in the original argument: the opponent's reply is a child of *the
  move we searched hardest*, not a root candidate we might have discarded, so the
  relevant subtree is the best-explored part of the tree rather than the worst.
- **The distribution is wide, not bimodal.** The plan predicted bimodality and asked for
  percentiles on that basis. Percentiles were still the right call, but for a different
  reason: a quarter of transitions inherit under 9% while a tenth inherit over a third.

Not measured: the play-site regime (a human opponent, who is not choosing from any
candidate set). The rule already excludes the play site, so that gap does not change the
decision — but it means there is no number for it, and none should be invented.

## Step 0 — the design, as it was pre-registered (half a day)

Add a diagnostic that answers one question: **at the moment a search begins, how many
visits would it have inherited from the previous search?**

This needs no API change and no reuse logic. In `PuctTreeSearch`, add behind a
non-default feature or a debug-only method:

```rust
/// Visits sitting under `codes` from the root, if that path exists in this tree.
pub fn subtree_visits_after(&self, codes: &[u16]) -> Option<u32>
```

It walks `edges` from node 0 following `code`, returning the child's `visits`. Then in
a harness (extend `scripts/sims-match.mjs`, which already plays full games and holds
one search per move) keep the previous move's search alive for one extra move and log
`subtree_visits_after(&[our_move, their_reply])` against the budget.

Report the distribution, not the mean — a mean hides the bimodality that decides this.
Run it in both regimes, because they will differ:

- **self-play / engine-vs-engine** (opponent picks from a similar candidate set)
- **vs. the ladder or recorded human games** (opponent does not)

Decision rule, written down now so it is not negotiated afterwards:

| Median inherited fraction of budget | Then |
| --- | --- |
| ≥ 25 % | Build it. The rest of this plan applies. |
| 10–25 % | Build it only for self-play (training throughput), not the play site. |
| < 10 % | Stop. Record the number in `FOOTPRINT.md` as a settled question and spend the effort on depth, which we already know pays to ~2048 sims. |

## Current architecture, as it actually is

`rust/normal-duel-core/src/puct.rs`:

- `PuctTreeSearch { nodes: Vec<Node>, edges: Vec<Edge>, .. }` — an arena. `Node` holds
  `position: SearchPosition`, `ply`, `rep_count`, `resets_window`, `terminal`,
  `expanded`, `edges_start`, `edges_len`, `visits`, `value_sum`, `value`. `Edge` holds
  `code`, `prior`, `visits`, `value_sum`, `child: u32`. Node 0 is the root.
- `RootContext { position, ply, window: RepetitionWindow }` is the whole of what a root
  needs beyond the board, and `PuctTreeSearch::new(config, root, params, rng)` is the
  one constructor everything funnels through (`from_state` wraps it).
- The Gumbel logit comes from the **stored edge prior**:
  `let logit = js_log(edge.prior.max(POLICY_FLOOR));`
- Draw stream: `rng: Lcg32` is handed in at construction and taken back with `rng()`,
  because one self-play game runs off one `createLcg32(seed)`.

`rust/normal-duel-wasm/src/lib.rs`: `NormalDuelSearch::new(config_json, state_json,
options_json)` validates the state through `validated_search_state` and builds a fresh
`PuctTreeSearch` with `Lcg32::new(dto.seed)`.

Two consequences worth naming, because they make this cheaper than it looks:

- **A promoted node's edges already carry priors**, so the new root needs no network
  evaluation to seed Gumbel. Reuse saves the root eval outright — one call per move,
  ~0.2 % of a 512-sim budget, but it is free and it is exact.
- **`resets_window` and `rep_count` are already per-node**, so the repetition
  bookkeeping needed for a rebased root is largely present rather than absent.

## Step 1 — subtree extraction in core (2–3 days)

Add to `PuctTreeSearch`:

```rust
/// Consume this search and return a tree rooted at the node reached by `codes`,
/// or `None` when that path was never expanded.
pub fn into_subtree(self, config: &Config, codes: &[u16]) -> Option<PuctTreeSearch>;
```

and a constructor that accepts one:

```rust
pub fn resume(
    config: &Config,
    inherited: PuctTreeSearch,   // already rooted at the new position
    params: PuctParams,
    rng: Lcg32,
) -> Result<Self>;
```

Extraction has to do four things, and three of them are where the bugs will be:

1. **Reachability copy.** Walk from the new root, copying reachable `Node`s and
   `Edge`s into fresh vectors and remapping `child` and `edges_start`. Do not try to
   compact in place; the arena has no free list and a partial remap is unreviewable.
2. **Rebase the repetition window.** The new root's `RepetitionWindow` is *not* the old
   root's. Rebuild it by replaying `codes` from the old root's window: `push` each
   intermediate key, and `reset` at any step whose node has `resets_window` (a wall
   placement starts a fresh window). Then recompute `rep_count` for every copied node,
   because a node's count is relative to its window and the window just moved. Getting
   this wrong silently changes threefold-repetition adjudication — the search would see
   draws that are not there, or miss ones that are.
3. **Rebase ply.** `Node.ply` is absolute, so it survives extraction unchanged. Check
   it against the tree ply cap (`rust/normal-duel-core/tests/tree_ply_cap.rs`) — an
   inherited deep node must not smuggle a node past a cap the fresh path enforces.
4. **Reset the schedule, keep the statistics.** This is the load-bearing decision:

   > The Gumbel sequential-halving schedule is **re-run from scratch** at the new root
   > — fresh Gumbel draws, fresh `candidates`/`survivors`/`rounds`/`per_candidate`,
   > `used = 0`, `budget` reset. Only `visits`/`value_sum`/`value`/`prior` on inherited
   > nodes and edges are kept.

   Why: Gumbel's guarantee (its policy-improvement property) rests on the schedule
   allocating a *planned* number of visits to each candidate. Counting inherited visits
   toward that plan breaks the guarantee, and it also biases the target: a candidate
   with 40 inherited visits and 10 planned ones would dominate the visit-count policy
   target for reasons that have nothing to do with this search. Inherited statistics
   improve the *value estimates* the schedule reads; they must not replace the
   schedule. Note the consequence honestly — with the schedule re-run, reuse makes each
   simulation *better informed* rather than making simulations *unnecessary*, so the
   payoff shows up as strength at equal sims, not as fewer sims for equal strength.

Everything else stays: `phase` starts at `RootPending` unless the root is already
`expanded` (then `Ready`, and the root eval is skipped), `path`/`descent_keys` clear.

## Step 2 — wasm API (1 day)

The stepped API is the constraint the user correctly identified: `NormalDuelSearch`
takes a fresh position. Extend rather than replace, so every existing caller keeps
working:

```rust
/// Play `action_code` and `reply_code` on this search's root and reuse what is
/// underneath. Returns false when the path was not in the tree, in which case the
/// search has been rebuilt fresh and the caller need not care.
#[wasm_bindgen(js_name = advanceRoot)]
pub fn advance_root(&mut self, action_code: u16, reply_code: u16)
    -> std::result::Result<bool, JsValue>;
```

`advanceRoot` returning `false` rather than erroring on a miss is deliberate: a miss is
the *common* case, not an exception, and a caller that has to branch on an exception
will get it wrong.

The state still has to be validated. Keep `validated_search_state` in the fresh path,
and have `advance_root` verify the promoted node's `position` against the caller's
expected position key — the play server already sends `expect` for exactly this reason
and `stateFromHistory` already replays history to build a trustworthy state. **On any
mismatch, discard the tree and build fresh.** Silently searching a position the player
is not looking at is the failure this repo has already had twice, in
`pendingLeafMask()` and in `Buffer.from(typedArray)`; both times the move was legal and
nothing looked wrong.

## Step 3 — callers

- `scripts/play-server.mjs`: `aiMove` currently constructs and frees a search per call.
  Hold one search per game in the existing `games` map, call `advanceRoot`, fall back to
  fresh on `false`. The per-game search must be freed when the game ends or is
  abandoned, or the server leaks wasm memory across sessions — and note that wasm heap
  growth detaches views, which is the other bug this repo has had.
- `rust/normal-duel-core/src/selfplay.rs` / `SelfPlayBatch`: the bigger prize if step 0
  shows self-play inherits well, because a training iteration is thousands of games.
  Also the riskier one: it changes the visit-count **policy targets** that training
  consumes, and this project has already lost 114 iterations to a bad policy target
  (see `wrongway-alphazero-target-bug`). Gate it behind a flag, default off, and do not
  turn it on in the same run as any other change.

## Verification

The repo's parity tests are the asset here, and they are also the obstacle: `tests/
js_puct_parity.rs`, `js_hot_path_parity.rs` and `fixture_parity.rs` pin this tree
move-for-move against the JS reference, which has no reuse. Reuse *will* change chosen
moves — that is the point — so:

1. **Fresh path unchanged.** With reuse never invoked, every parity test must pass
   bit-identically. This is the gate that says the refactor was a refactor.
2. **Extraction invariants** (new unit tests): a subtree copy has the same visit and
   value sums under the promoted node as the parent tree did; every `child` and
   `edges_start` index is in bounds; `rep_count` after extraction equals what a fresh
   search at that position with the replayed window would compute; a `resets_window`
   step produces a one-entry window.
3. **Equivalence at equal information.** A search resumed from a tree with all
   inherited visits *zeroed* must choose exactly what a fresh search chooses given the
   same seed. This isolates "extraction is correct" from "inherited statistics help".
4. **Strength, measured, not assumed.** `scripts/sims-match.mjs` with reuse on one side
   at equal sims. Use its exact binomial and expect to need ≥ 100 games: the effects
   this repo has been chasing are ±30 Elo and 10 games cannot see them. If reuse is
   within noise at equal sims, it has not earned its complexity, whatever step 0 said.

## Risks

- **Repetition rebasing is the real bug surface.** It is invisible in ordinary play and
  changes adjudication in drawn-ish lines. Test 2 above is not optional.
- **Policy-target contamination in self-play.** Covered by the flag and by never
  co-launching it with another change.
- **Memory.** One retained search per active game instead of one transient search per
  move. Bounded by concurrent games (small on the play site, large in self-play).
- **It may simply not pay here.** Step 0 exists so that this costs half a day to find
  out rather than a week.

## Cost

Step 0: half a day. Steps 1–3 with verification: about a week. The order matters more
than the total — step 0 is the only part that should be scheduled unconditionally.
