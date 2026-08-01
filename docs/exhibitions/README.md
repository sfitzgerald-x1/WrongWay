# Normal-duel exhibitions

Isolated tooling for recording and inspecting a single **non-enforced** canonical
9x9 exhibition game. Nothing here participates in the enforced strength gate
(`docs/normal-duel-strength-evaluation.md`) and nothing here is loaded by the
production UI.

The exhibition is the authenticated Hard+ WASM candidate (side **A**) against the
privately branded pinned Hard subprocess baseline (side **B**), from the committed
balanced opening `balanced-9x9-007`, with an outer per-turn deadline of
**15000 ms**. The shared `runMatch` protocol plays the game, so deadlines,
subprocess isolation, and adjudication are the same code paths the strength
harness uses; this tooling only records what was played.

## Record the exhibition

```sh
npm run exhibition:hardplus-9x9
```

That builds the hermetic candidate release (`npm run build:normal-duel-wasm-candidate`)
and then runs `scripts/run-hardplus-exhibition.mjs`, which writes

```
docs/exhibitions/hardplus-exhibition-balanced-9x9-007.json
```

Requires the WASM toolchain used by the existing build (`cargo`, the
`wasm32-unknown-unknown` target, and `wasm-bindgen` 0.2.126). Expect several
minutes of wall clock: every candidate turn spends close to its full 15 s budget.

Options (all optional): `--opening <id>`, `--deadline-ms <ms>`,
`--candidate-requested-budget-ms <ms>`, `--seed <int>`, `--out <file>`,
`--candidate <adapter.mjs>`, `--candidate-manifest <manifest.json>`,
`--allow-commit-drift`, `--recycle-normal-duel-wasm-per-decision`.

### Per-decision recycling is required at 15 s

A 15 s exhibition run with one persistent candidate subprocess per game will
**fail** near the end of a long game. Dropped Rust allocations are never returned
to WASM linear memory, so the candidate's high-water mark grows monotonically
until the 512 MiB canonical profile kills it. The recorded failure
(`hardplus-exhibition-balanced-9x9-007.json`) forfeited with `memory_limit` at ply
74 while the rules verdict was still `ongoing`.

Pass `--recycle-normal-duel-wasm-per-decision` to give the candidate a fresh
subprocess per move, which contains the growth:

```sh
npm run build:normal-duel-wasm-candidate
node scripts/run-hardplus-exhibition.mjs \
  --opening balanced-9x9-007 \
  --deadline-ms 15000 \
  --recycle-normal-duel-wasm-per-decision \
  --out docs/exhibitions/hardplus-exhibition-balanced-9x9-007-recycled.json
```

Only the candidate is recycled; the pinned Hard baseline keeps its per-game
subprocess. The recorded recycled run
(`hardplus-exhibition-balanced-9x9-007-recycled.json`) completes as a genuine
board win by goal at ply 75, doing comparable work (589,062,513 nodes over 36
decisions versus 592,025,766 over 35 before the kill) with a per-move cold-start
cost of 95.6 ms mean / 285.5 ms maximum.

The run is pinned to repository commit `bb35d8c` and refuses to record from a
different `HEAD` unless `--allow-commit-drift` is passed. A dirty worktree is
recorded, not rejected.

## Inspect the replay

```sh
npm run serve:exhibitions      # python3 -m http.server 8080, from the repo root
```

Then open
<http://localhost:8080/docs/exhibitions/replay.html>.

`replay.html` is a self-contained static page (no dependencies, no imports). It
loads `?artifact=<relative-url>` and defaults to
`./hardplus-exhibition-balanced-9x9-007.json`; the artifact must be same-origin.
`?ply=<n>` deep-links to a ply. Controls: first / back / play / step / last, a
scrub slider, and ← → / space keys. The board draws cells, both goal rows, pawns,
wall segments, and highlights the ply's own action; the panels show the actor and
its role, the action, the outcome, and the per-ply candidate telemetry.

## Load a replay into the app

`hardplus-exhibition-replay.html` converts an artifact into the same record shape
the app writes when a local game ends and stores it in the app's replay history.
Serve the repository root over http first (the app is a different origin over
`file:`), then open
<http://localhost:8080/docs/exhibitions/hardplus-exhibition-replay.html>.

**Forfeits are not board wins.** The app's replay record has no field for "won
because the opponent failed". A result counts as a board win only when the
exhibition result kind is `win`, no side failed, and the rules verdict at the
final recorded ply independently names the same winner.

A non-board result is stored with `winner: null`, with the side names annotated
(`Hard+ (forfeited: memory_limit)` and `NO BOARD WIN — Pinned Hard`), and with
machine-readable detail in
exhibition-only fields the app ignores (`exhibitionResultKind`,
`exhibitionReason`, `exhibitionFailedPlayer`, `exhibitionBoardOutcome`). The page
also warns before storing.

The name annotation is not cosmetic. Two app screens read the record differently:
the playback screen keys off `winner` and correctly presents no victory for
`null`, but the replay **list** renders `entry.winner === 'A' ? nameA : nameB`,
so a null winner falls through to side B and would still read as "Pinned Hard
wins". The app never writes a null winner itself, so that screen was never built
for this state, and this tooling cannot change `index.html`. The names are the
only channel that reaches the list, and the disclaimer leads because the list
appends its own "wins" — the entry reads
`NO BOARD WIN — Pinned Hard wins`.

## Artifact shape

`format: normal-duel-hardplus-exhibition-replay-v1`

- `exhibition` — non-enforced marker, mode, per-turn deadline, seed, clock
  profile, opening id, side roles, engine ids and versions.
- `repository`, `config`, `book`, `corpusProvenance` — pinned commit and worktree
  state, canonical 9x9 config, verified opening-book provenance.
- `candidate`, `baseline` — content-addressed candidate artifact provenance and
  subprocess isolation provenance; pinned baseline identity and trust root.
- `outcome` — result kind, winner and winning role, reason, final ply, plus the
  forfeit fields (`failedPlayer`, `error`) when a side fails.
- `initialPosition` and `plies[]` — every ply from ply 1, `phase` marking the four
  committed opening plies versus engine-played plies, with `actor`, `action`
  (`actionCode` for opening plies), resulting `position` (pawns, walls, stock,
  turn) and `outcome`.
- `plies[].telemetry` — per-ply engine telemetry: `nodes`, `depth`, trusted
  charged timings (`chargedActiveMs`, `chargedSelectMs`, `chargedSetupMs`,
  `chargedObserverMs`), the engine's self-reported `reportedElapsedMs` when it
  reports one, and the timing source / clock profile. Opening plies have `null`.
- `telemetry` — the raw per-side arrays from `runMatch`, index alignment counts,
  and caveats.

### Caveats recorded in the artifact

- Transitions are captured by wrapping the **baseline** descriptor so it records
  every applied transition. Its trailing setup/observer active-time flush is
  therefore not charged. The candidate side is unwrapped, so Hard+ telemetry uses
  the trusted subprocess timing path unchanged.
- The pinned Hard baseline reports search depth but not node counts, and the WASM
  candidate reports nodes and depth but no self-reported elapsed time; the trusted
  charged timings cover both sides.
- Per-ply telemetry is index-aligned per side in decision order. The candidate's
  `decisionSamples` is normally one greater than its ply count: the harness
  appends the end-of-game active-time flush as a trailing sample.
