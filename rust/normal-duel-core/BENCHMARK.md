# Normal-duel throughput benchmark

Run the portable correctness smoke check with:

```sh
npm run check:normal-duel-throughput-benchmark
```

Run the full benchmark (two warmup passes and nine timing samples per engine)
with:

```sh
npm run benchmark:normal-duel-throughput
```

For a coarse diagnostic split of the JavaScript or Rust public-core work (not
the speedup gate), run either matching executable with `--profile --warmup 1
--samples 3`:

```sh
node scripts/normal-duel-throughput-benchmark.mjs --profile --warmup 1 --samples 3
cargo run --quiet --release --manifest-path rust/Cargo.toml --example throughput_benchmark -- --profile --warmup 1 --samples 3
```

The result is JSON. It records the SHA-256 of
`tests/fixtures/normal-duel-perft-v1.json`, Node and `rustc` versions, host OS
and architecture, every sample, both medians, and Rust's observed speedup over
the JavaScript reference.

## Comparable work

The driver first runs both engines in verification mode. Each reconstructs all
nine perft roots from their frozen provenance, checks the frozen root action
codes and exact-depth leaf count, and produces a shared FNV-1a-64 checksum over
the root action codes, checked child states, and child legal-action code lists.
The driver compares that fixture SHA and both checksums *before* asking either
engine to time anything.

One timed pass then performs the same deterministic game-tree work in each
engine for each root: generate legal actions; for every root action, apply it
and generate the child's legal action codes; then run scalar perft at that
root's frozen depth. Rust validates and converts each frozen root once, then
uses the compact generated-code apply/undo path; the JavaScript side uses its
reference transition objects. Before timing, Rust materializes a compact trace
and asserts that every root action code, child position key, and complete child
legal-action list matches its immutable engine, then asserts both Rust paths
produce the same checksum which the driver compares with JavaScript.
Differential tests repeat that exact trace comparison over every frozen root
and pin every perft depth. Fixture parsing, state construction, checksum work,
process startup, and JSON formatting are outside the timing interval. The
timed pass returns a checksum-independent numeric result that must equal the
pre-timing pass.

This is deliberately a current-core baseline, not a benchmark of parsing,
FFI, WASM, network transport, or a precomputed perft table.

## Performance policy

The `Pinned 100x throughput gate` workflow runs for PRs, `main` pushes, manual
dispatches, and the Monday schedule. Its documented runner class and toolchain
are `ubuntu-24.04`, Node `22.22.2`, and Rust `1.94.1`; it runs one warmup and
seven measured samples and enforces:

```sh
npm run benchmark:normal-duel-throughput -- --min-speedup 100
```

If a benchmark is below 100x, treat it as an optimization backlog, not as an
excuse to compare a different workload. The compact path must preserve
canonical policy-code order and reproduce this benchmark's action and perft
checksums; it must not skip the wall-path legality rule.

The implementation cleared the gate locally on 2026-07-30 (Apple arm64,
Node 22.22.2, rustc 1.94.1): seven measured samples after one warmup produced
JavaScript median 2853.649 ms and Rust median 4.227 ms, an observed 675.114x
speedup. This is verification evidence for that host, not a portable promise.
