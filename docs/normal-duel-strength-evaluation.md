# Normal-duel Stage 2 strength evaluation

The Stage 2 product bar is:

- at least 200 deterministic opening pairs / 400 games;
- the candidate wins at least 66% of games against the pinned Hard product
  behavior; and
- the paired-opening cluster 95% score interval has a lower bound above 50%.

This is the canonical 9x9 AI acceptance protocol. Existing 7x7 rules,
fixtures, and harness unit-test configurations remain historical compatibility;
they are not targets for new AI implementation, tuning, benchmarking,
self-play, training, truth-track, and release work, nor for future evaluation.

An enforced claim is fail-closed: it must use strength mode, at least 200
evaluated opening pairs, exact verified corpus provenance, and subprocess-isolated
descriptors for both engines. Lower sample sizes and in-process engines are
available only for non-enforced smoke or regression runs.

The enforced environment is canonical rather than caller-selectable:

- 9x9 normal Duel, player A first, 200-ply cap, and configuration SHA-256
  `1869dc2a71b4043aebee681e64283f42b93d13665f7ead9f1eca9ccb1a0b1da0`;
- opening/session seed `1831565813` (`0x6d2b79f5`);
- generator `normal-duel-balanced-opening-generator-1.0.0` with `lcg32-v1`;
- the complete verified corpus, containing at least 200 openings; and
- exactly 900 ms active-time budget per move, a 900 ms session-initialization
  cap, and a 900 ms cap on each observer call;
- a 512 MiB process memory ceiling plus a 128 MiB V8 old-space ceiling; and
- a content-addressed, hermetic candidate release with filesystem-content,
  module-load, environment, and network isolation.

These values are exported from
`scripts/evaluation/normal-duel-strength-constants.mjs`. Enforced evaluation
rejects alternate seeds, board sizes, starting players, configurations,
generator metadata, deadlines, and corpus subsets.

The 900 ms protocol is the enforced deadline for automated and pinned-Hard
acceptance. It is distinct from the feature-flagged, human-facing `Hard+`
15,000 ms per-turn ceiling. Routine CI, smoke, and regression tests may use
shorter deterministic logical-clock deadlines or fixed-node/fixed-depth limits;
they must never use the 15,000 ms human-facing budget.

Fixed-node regression mode is auxiliary and deterministic for both sides. By
default every game receives a fresh logical clock starting at 0 ms and advancing
exactly 4 ms per `now()` poll. The `normal-duel-regression-logical-clock-v1`
profile was calibrated from 12 canonical 9x9 pinned-Hard opening roots: the
median observed real work per clock poll was 4.189 ms, and the nine roots near
Hard's 700 ms cutoff used 165–186 polls. This was a single macOS calibration
under Node 22.22.2, not a portable throughput claim. The rounded 4 ms tick
approximates that median work while making ablation comparisons repeatable. It
is not a product timing model and cannot produce release or strength-gate
evidence. Regression mode is in-process only: the harness rejects any worker
adapter before starting a game because worker IPC cannot share the logical
clock synchronously.

The built-in regression profile also gives pinned Hard a separate 10,000 ms
real-time VM safety ceiling. This is only a runaway guard; it does not replace
or shorten Hard's logical 700 ms decision cutoff. A safety-ceiling overrun is
reported as `clock_profile_exceeded`, not as a generic crash. Strength mode
retains its existing real deadline and VM-timeout behavior.

Every game records its clock profile in `settings.clockProfile`, and the
top-level report records the selected `clockProfile`. Explicit `clockFactory`
callers may provide their own `clockProfile`; an override without one is
reported as `caller-supplied-clock-unprofiled-v1`. Strength mode continues to
use the real monotonic `performance.now` profile. Per-engine telemetry records
the `clockProfileId` and timing source; deterministic regression measurements
use `deterministic-logical-clock-time`, distinct from strength mode's
`trusted-harness-active-time`.

The repository's checked-in 12-opening book is a smoke and regression artifact,
not enough to claim the gate. No 400-game run is part of this harness change.

## Pinned Hard baseline

`scripts/evaluation/hard-baseline-46a871c7.mjs` runs unmodified snapshots of
`js/game-logic.js` and `js/ai.js` from commit
`46a871c7b061a33922bdb9c6d78355e2e9b6b607`. Baseline verification does not let
the checked-in manifest authenticate itself. The adapter contains this
hard-coded trust root:

- manifest:
  `55591ccaeb32cdfff4c30f23723b5e98d13ef304b5fd6eaa1d7ca761148f0df6`;
- game logic:
  `db8dc06882f015d78460882e0ab629f660d0a4c1c7600d986a0e0a979139e75a`;
- AI:
  `07ec05bb99602e36a879095ec159728bbf0ba520338d0a20a904888f2b57a777`;
- orchestration snapshot:
  `8dd628e49934e3bb401833d6b69a8854ed6f1dffaee945fbc399bc2a4d2e72bc`;
- original index:
  `87468038df741428fe25f8532f8871459f181554a6dbadf4e151b6bf44621f4c`.

This is the explicitly frozen pre-AI-development product baseline, not a
moving alias for the latest `main`. [The plan](./ai-engine-plan.md)'s Stage 2 exit gate and its
**Match design** baseline protocol freeze the direct Hard call and
orchestration at this commit. In that plan, “current-Hard” means the product
opponent current at the moment the development baseline was frozen. Later
rules-routing implementation PRs therefore do not move the opponent or change
the 66% denominator.

The exact identity, source commit, source records, and snapshot hashes must all
match. Rewriting both a snapshot and its manifest record therefore still fails.
The same complete trust root is carried into enforced-run branding and report
provenance.

The source commit has no standalone license file. The snapshots are retained
under the team's permission solely as an evaluation oracle; they are not wired
into or intended to replace product code.

The adapter also reproduces the product orchestration from that commit:

- Duel Hard calls `aiDuelHard` directly;
- each game resets `aiProgRef` to `{best: Infinity, stall: 0}`;
- `recentAi` is the current square followed by up to five prior same-side pawn
  moves, newest first; walls are ignored;
- only prior positions at indices 1 through 4 receive Hard's cycle penalty;
- stall changes only for proposed pawn moves and persists across wall moves;
- the third non-improving pawn proposal is replaced with `moveTowardGoal` on
  the pre-decision BFS path, then stall resets.

The pinned product assigned Hard only to player B. When a paired game assigns
Hard to A, the adapter rotates coordinates, walls, and same-side history 180
degrees, swaps player roles, runs the original B oracle and B anti-stall
orchestration, then inverse-rotates its action. It never invokes
`aiDuelHard` under invented A-side semantics. Seeded equivalence tests compare
this path directly with the corresponding rotated B position.

Enforcement accepts only the privately branded pinned subprocess adapter. Its
identity is fixed to `hard-product-46a871c7`, version
`wrongway-hard-product-baseline-v1`, and source commit
`46a871c7b061a33922bdb9c6d78355e2e9b6b607`. A generic adapter that copies those
public fields is not accepted.

The VM exposes board dimensions and a minimal `Date.now` capability. `Date.now`
delegates to the injected monotonic clock, so the pinned baseline retains its
original relative 700 ms cutoff without wall-clock jumps and self-limits at
roughly 700 ms. The outer acceptance protocol gives both engines the canonical
900 ms active-time allowance and passes the absolute `deadlineAtMs`; clock
rollback fails closed.

## Engine adapter contract

A candidate module exports either a descriptor as `default` or
`createEngineAdapter()`:

```js
export default {
  id: 'rust-hard-plus',
  version: '0.1.0',
  capabilities: { nodeBudget: true, deadline: true },
  createSession(game) {
    return {
      async selectAction(request) {
        // request.state/config are immutable snapshots of authoritative data.
        // limits has nodeBudget or deadlineAtMs plus wallClockBudgetMs.
        return {
          action: { kind: 'pawn', to: { r: 7, c: 4 } },
          stats: { nodes: 1234, depth: 5, elapsedMs: 12.5 }
        };
      },
      observe(transition) {},
      close() {}
    };
  }
};
```

This boundary is suitable for a future Rust/WASM or subprocess adapter. The
adapter never adjudicates or mutates the game. The JavaScript reference engine
validates/applies every action and owns goal, threefold-repetition, and ply-cap
outcomes.

The authoritative state never crosses this boundary. Session context, request
state, opening history, and observer transitions are structured-cloned and
deep-frozen. Returned actions are cloned before validation and application, so
an engine cannot retain a reference that changes turn order or outcomes.

Every strength-mode CLI run hosts each engine/game in a fresh Node subprocess;
enforced validation also requires the proxy's private canonical brand. The
official WASM candidate can opt into one subprocess per candidate decision
instead — see [Per-decision WASM candidate recycling](#per-decision-wasm-candidate-recycling)
— but per game remains the default for every engine. A
host-side timer sends `SIGKILL` to a process that exceeds its allowance,
including code blocked synchronously, and waits for process exit before cleanup
completes. IPC uses a random private channel. Candidate attempts to use
`process.kill`, the named `node:process` kill export, `process._kill`,
`process.abort`, `process.send`, or `process.disconnect` are denied. Invalid,
forged, duplicate, late, or closed-channel IPC fails the engine closed.

Candidate stdin, stdout, and stderr are connected to the null device. This
contains console calls, stream writes, direct `fs.writeSync(1/2)`, and
`_rawDebug` without creating parent-side pipe pressure; only the parent CLI
writes the JSON report. The privately branded adapter exposes this as
`stdioIsolation: "null-device-v1"`, and canonical enforcement requires that
exact profile.

Canonical memory isolation on macOS uses `/usr/sbin/taskpolicy -m 512` around
each engine process and `--max-old-space-size=128` inside Node. Before an
adapter is accepted, a cached preflight proves that a touched 112 MiB
`Buffer` is killed under a 96 MiB taskpolicy ceiling. This covers native and
external allocations, not just the V8 heap. A limit kill is reported as
`memory_limit`; only the taskpolicy `SIGKILL` exit shape receives that private
classification. A candidate-selected numeric exit status such as 137 remains
an ordinary crash. An IPC disconnect is held briefly for the authoritative
exit status so a real limit kill is not mislabeled as a generic crash.

Enforced runs fail closed if that exact taskpolicy profile and preflight are
unavailable. Other platforms may run non-enforced evaluations with the
explicitly reported `v8-old-space-only-ineligible-v1` fallback, but that
fallback cannot produce a gate-eligible report. A future Linux implementation
must provide and verify an equivalent cgroup-v2 process ceiling before it can
be made eligible.

Before candidate import, the runtime captures the pristine object-inspection,
numeric-validation, string-slicing, JSON-size, clock, collection, and send
intrinsics used by the IPC boundary. Candidate monkeypatches therefore cannot
disable the boundary. No candidate descriptor, decision, observer result, or
error object is sent directly:

- descriptors become bounded ID/version/source-commit strings, the exact
  eight-string Hard trust-root schema when present, and the boolean
  `nodeBudget`, `deadline`, and `deterministicClock` capabilities;
- decisions become only `null`, a canonical pawn action, a canonical
  `H-r-c`/`V-r-c` wall action, or an `{ action, stats }` envelope whose stats
  are limited to finite `elapsedMs`, `nodes`, `depth`, and
  `antiStallReplaced` scalars;
- unknown descriptor, capability, and stats fields are ignored without
  traversing them, preserving compatibility with extra score/PV telemetry,
  while unknown action/envelope fields and accessors on inspected fields fail
  closed;
- observer return values are discarded, and thrown values are converted to
  capped own-data name/message/code/stack strings.

Every resulting null-prototype frame must also fit a 16 KiB JSON byte-size
check before private IPC. Synchronous serialization/send failure exits the
child and is adjudicated as a bounded crash rather than recursively reporting
candidate data.

Subprocesses run under the Node 22 permission model without filesystem writes,
child processes, nested workers, or native addons. An enforced candidate gets
read permission to one hermetic release directory. A canonical manifest names
the entry and every JavaScript, WASM, and model file by relative path and
SHA-256. The directory may contain only that manifest, the declared regular
files, and necessary directories—no undeclared files, symlinks, or special
files. The parent verifies this before the descriptor probe and every game
session; the child re-hashes the manifest and every declared file immediately
before import.

Before candidate import, a synchronous module hook admits only declared file
URLs and this explicit Node builtin set: `assert`, `buffer`, `console`,
`crypto`, `events`, `fs`, `fs/promises`, `path`, `perf_hooks`, `process`,
`stream`, `string_decoder`, `timers`, `url`, `util`, and `zlib`, including the
documented safe submodules used by those families. The check uses the
resolver's canonical `node:*` URL, so bare internal names and `node:` names
cannot take different paths. Every other builtin fails before module load,
including internal HTTP/TLS transports, `inspector/promises`, `wasi`,
`sqlite`, child processes, workers, `module`, `vm`, and `v8`.

Exact-path wrappers restrict candidate content opens to declared files,
`createRequire` and loader registration are disabled, and raw file-descriptor
writes are denied. Node's loader retains raw descriptor reads, but it can only
open files under the permission root; the candidate cannot use
content-opening APIs to obtain an undeclared descriptor. Pure WebAssembly
compilation remains available, and declared WASM/model bytes remain readable
through the restricted filesystem surface.

The child is spawned with a new minimal environment containing only `LANG=C`,
`LC_ALL=C`, and `TZ=UTC`; it never inherits `NODE_OPTIONS`, `NODE_PATH`, preload
hooks, or other parent variables. The runtime clears and reinstalls that same
allowlist before candidate import as defense in depth. Network builtins and
their public connection APIs, global `fetch`, `WebSocket`, and `EventSource`,
plus low-level binding access are denied before candidate code loads.
Candidate-visible `process.exit` and `process.reallyExit` are also denied; the
runtime uses a captured private exit function for orderly parent disconnect.
These controls establish the enforced artifact execution identity; they are
not a general-purpose secrecy sandbox for arbitrary host metadata APIs.

The descriptor probe is excluded from game timing because it occurs before a
game exists. For each game, subprocess spawn, module load, and `createSession`
time become setup debt. Successful observer calls accrue observer debt. At the
next `selectAction`, both debts are subtracted from the 900 ms allowance and
then reset; only the remainder is available for selection. The parent checks
elapsed time again after IPC returns so a timer race cannot admit an
over-budget action. A 900 ms initialization or observer cap limits a single
operation but never grants extra move time. Pending setup/observer debt at a
terminal result is harvested before sessions close, so trusted primary
telemetry includes all setup, observe, and select active time.

## Match failures

The following are immediate losses for the offending engine:

- crash, invalid return, or 900 ms cap overrun during `createSession`;
- crash in `selectAction` or `observe`;
- null/undefined action;
- illegal action;
- active-time deadline overrun or subprocess timeout;
- process-memory ceiling overrun;
- in logical-clock regression mode, a pinned-Hard real safety-ceiling overrun,
  reported as `clock_profile_exceeded`;
- in fixed-node regression mode, a node-budget-capable engine omitting node
  telemetry or reporting more than the requested budget.

Cleanup failures after adjudication do not rewrite a result.

## Openings and pairing

`scripts/generate-normal-duel-balanced-openings.mjs` uses `lcg32-v1` and
canonical legal-action order. Every opening has 4, 5, or 6 plies and remains
ongoing. After each selected ply, both players' wall-only shortest-path
distances must differ by at most one.

The opening itself is replayed by the reference engine. Each opening is played
twice: candidate as A / baseline as B, then candidate as B / baseline as A.
This is the statistical cluster; games from the same opening are not treated as
independent samples.

Regenerate or check the smoke book:

```sh
node scripts/generate-normal-duel-balanced-openings.mjs --write
node scripts/generate-normal-duel-balanced-openings.mjs --check
```

Generate a larger book and matching manifest without overwriting the smoke
artifact:

```sh
node scripts/generate-normal-duel-balanced-openings.mjs \
  --count 200 \
  --seed 1831565813 \
  --book-path /tmp/wrongway-openings-200.json \
  --manifest-path /tmp/wrongway-openings-200.manifest.json \
  --write
```

Before an enforced run, the harness checks the exact book and manifest bytes,
their SHA-256 values, generator version and algorithm, seed, configuration
hash, count, and byte-for-byte seeded regeneration. A custom or edited book
without that matching manifest cannot produce a passing enforced report. The
report records both hashes and the full generator provenance.

Verification returns deeply frozen book, manifest, and provenance objects with
a private in-process association. Enforcement requires those exact objects;
copying their public fields or mutating/slicing the verified book invalidates
the association. `--opening-limit` is therefore disallowed with
`--enforce-gate`.

## Running an evaluation

Fast deterministic regression mode uses a fixed candidate node budget and the
auxiliary deterministic logical clock described above:

```sh
node scripts/run-normal-duel-strength.mjs \
  --candidate ./path/to/adapter.mjs \
  --mode regression \
  --node-budget 50000 \
  --opening-limit 2
```

Product-strength mode uses subprocess isolation and monotonic active-time
deadlines:

```sh
node scripts/run-normal-duel-strength.mjs \
  --candidate ./path/to/release/adapter.mjs \
  --candidate-manifest ./path/to/release/manifest.json \
  --book /tmp/wrongway-openings-200.json \
  --manifest /tmp/wrongway-openings-200.manifest.json \
  --mode strength \
  --deadline-ms 900 \
  --enforce-gate
```

`--enforce-gate` is opt-in and fixes the minimum at 200 or more opening pairs.
It rejects regression mode, any `--opening-limit`, noncanonical seeds or
deadlines, an absent or invalid `--candidate-manifest`, unverified corpora,
noncanonical memory isolation, non-subprocess candidates, and non-pinned
baselines before play.

The logical regression result is suitable for paired candidate A/B comparisons
only. The official 900 ms gate remains the real-monotonic, subprocess-isolated
strength protocol and is unchanged by the regression clock profile.

### Per-decision WASM candidate recycling

`--recycle-normal-duel-wasm-per-decision` gives the official normal-duel WASM
candidate a fresh subprocess for every one of its own moves instead of one child
per game. It is off by default: without the flag every engine keeps the existing
per-game lifecycle. The flag is valid only in strength mode and requires
`--candidate-manifest`; regression mode and a missing manifest are rejected while
arguments are parsed, before any engine is loaded.

```sh
node scripts/run-normal-duel-strength.mjs \
  --candidate ./path/to/release/adapter.mjs \
  --candidate-manifest ./path/to/release/manifest.json \
  --mode strength \
  --deadline-ms 15000 \
  --recycle-normal-duel-wasm-per-decision
```

The CLI always loads the ordinary subprocess-isolated candidate first and then
wraps only that candidate. Wrapping is admitted only for the authenticated
official release: the content-addressed manifest must verify, the entry must be
`adapter.mjs`, and its bytes must hash equal to the checked-in
`scripts/evaluation/normal-duel-wasm-candidate-adapter.mjs`. The pinned Hard
baseline is never recycled and keeps its per-game subprocess in every run, as
does any other engine.

What this costs and guarantees:

- every candidate move pays a cold child: process spawn, module load, manifest
  and file re-verification, WASM instantiation, and `createSession`, all charged
  as setup debt against that move's active-time allowance;
- every candidate move also pays teardown, charged into the same move's
  selection time;
- teardown completes before the move is accepted. A decision whose child cannot
  be reaped is not played at all: it becomes an engine crash carrying
  `code: "subprocess_recycle_failed"`, and any earlier deadline or memory verdict
  survives only as context, because an unreaped child may still be running. No
  move is ever accepted from a possibly live child;
- `observe` is an exact no-op. The official WASM adapter derives each decision
  from the full request state and is stateless per request, so there is no
  per-game observer state to carry and no child to feed between decisions;
- the candidate's WASM linear-memory high water cannot accumulate across a game,
  which is the containment this flag exists for.

This is intended for 15,000 ms human-strength and exhibition runs, where a
per-move cold start is a small fraction of the budget. The 900 ms canonical gate
may use either lifecycle: both are explicitly reported and enforced-eligible, so
recycling is neither required nor silently assumed. At 900 ms the per-move setup
and teardown are subtracted from the same 900 ms allowance, so a recycled gate
run gives the candidate materially less search time than a per-game run.

Recycled and non-recycled reports are provenance-distinct. The candidate's
isolation provenance names both its exact subprocess isolation
(`node-subprocess-per-game-v1` or `node-subprocess-per-decision-v1`) and its
exact session lifecycle (`stateful-session-per-game-v1` or
`stateless-wasm-per-decision-v1`), and an enforced token binds the same two
values, so no report is ambiguous about which lifecycle produced it.

The candidate manifest is canonical JSON with keys in this exact order,
strictly path-sorted file records, and a trailing newline:

```json
{
  "format": "wrongway-candidate-artifact-manifest-v1",
  "entry": "adapter.mjs",
  "files": [
    {
      "path": "adapter.mjs",
      "sha256": "<lowercase SHA-256 of exact file bytes>"
    }
  ]
}
```

Paths are relative POSIX paths without empty, `.` or `..` components. The
entry must be listed. Put the manifest and only its declared release contents
in a dedicated directory; a working tree containing unrelated files is
intentionally rejected.

## Report

The report includes:

- wins, losses, draws, observed win rate, and draw-as-half score;
- candidate side splits;
- parent-measured per-engine active-time distribution, split into setup,
  observer, and selection charges;
- separately labeled, untrusted engine self-reported timing;
- reported nodes and depth;
- pawn/wall action mix;
- failure counts;
- paired-cluster 95% intervals for raw win rate and draw-as-half score.

An enforced report also binds the exact book and manifest SHA-256 values, all
evaluated opening IDs and their count, the canonical configuration/seed/900 ms
deadline, candidate artifact manifest and file hashes, the exact isolation
profiles (including 512 MiB process and 128 MiB old-space limits), and the
pinned baseline ID/version/source commit. Candidate and baseline isolation
provenance are also reported from private adapter metadata, including each
engine's exact subprocess isolation and session lifecycle. Enforced validation
reads that private pair rather than public capability labels and fails closed
unless it is one of the two supported pairs, per-game/stateful or
per-decision/stateless. Gate eligibility
is controlled by a private enforcement token associated with the actual result
array; fabricated public `{ enforced: true, eligible: true }` objects cannot
mark a report as passing. Candidate-thrown `code: "deadline_exceeded"` values
remain ordinary crashes; only a privately branded parent timeout becomes a
deadline forfeit.

The interval is the large-sample normal interval over opening-pair cluster
means, clipped to `[0, 1]`. The gate uses the score interval because draws carry
half a point; the 66% bar remains the stricter raw observed win rate. The pinned
Hard implementation does not expose a node counter, so its node coverage is
reported honestly as unavailable.
