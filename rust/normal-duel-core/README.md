# wrongway-normal-duel

`wrongway-normal-duel` is the native, dependency-light implementation of the
`normal-duel-v1` contract. Its public structures serialize with the same field
names and action/outcome shapes as the JavaScript reference engine. It covers
only normal 1v1 7x7 and 9x9 duel; callers must reject or adapt every other game
mode before entering this crate. New AI implementation, tuning, benchmarking,
self-play, training, truth-track, and release work targets canonical 9x9 only;
the existing 7x7 support is legacy compatibility.

The public API is deliberately plain Rust and contains no platform bindings, so
the same core can later sit behind a command-line evaluator, an FFI layer, or a
WASM wrapper without duplicating rules.
