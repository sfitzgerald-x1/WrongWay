# Browser WebAssembly package

The wrapper uses `wasm-bindgen` crate and CLI version `0.2.126` exactly. Install
the matching CLI once:

```sh
cargo install wasm-bindgen-cli --version 0.2.126 --locked
```

Build standards-based browser ESM glue and its `.wasm` module under the ignored
`rust/target/wasm-bindgen/web` directory:

```sh
npm run build:normal-duel-wasm
```

Run the boundary gate, which instantiates that browser-target package in Node
and compares initial state, legal codes, transitions, position keys, strict
errors, and JavaScript-safe integer parsing with the JavaScript reference:

```sh
npm run check:normal-duel-wasm
```

Neither generated JavaScript nor generated WebAssembly is committed. CI
rebuilds both from the locked Rust dependency graph and the exact matching CLI.
If the CLI is installed elsewhere, set `WASM_BINDGEN_BIN` to its executable;
the build otherwise checks `CARGO_HOME/bin`, then `~/.cargo/bin`, then `PATH`
(including the `.exe` suffix on Windows). Every route must still report the
exact pinned version.
