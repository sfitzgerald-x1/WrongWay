//! Deterministic full-width search for `normal-duel-v1`.
//!
//! The public boundary accepts the strict [`GameState`](crate::GameState).
//! Search traversal stays inside this crate so only actions emitted by the
//! compact legal generator reach its allocation-light apply/undo path.

mod budget;
mod engine;
mod eval;
mod move_picker;
mod oracle;
mod tt;

use std::time::Duration;

use crate::{Config, GameState, Result};

pub use engine::{SearchDiagnostics, SearchOptions, SearchReport};

/// Search until the monotonic wall-clock deadline expires. The returned move
/// always comes from the last fully completed iteration (or the deterministic
/// legal fallback when depth one did not complete).
pub fn search_for(
    config: &Config,
    state: &GameState,
    duration: Duration,
    options: SearchOptions,
) -> Result<SearchReport> {
    engine::search_for(config, state, duration, options)
}

/// Deterministic search mode for tests, tuning and reproducible evaluation.
/// Exactly the same engine is used as [`search_for`], with node count replacing
/// the clock as its stopping condition.
pub fn search_nodes(
    config: &Config,
    state: &GameState,
    node_budget: u64,
    options: SearchOptions,
) -> Result<SearchReport> {
    engine::search_nodes(config, state, node_budget, options)
}
