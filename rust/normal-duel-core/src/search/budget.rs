use std::time::Duration;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    /// Browser and Node JavaScript hosts expose the high-resolution
    /// `performance` clock globally. Unlike `Date.now()`, it is monotonic.
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now_millis() -> f64;
}

pub(crate) trait SearchBudget {
    fn exhausted(&mut self, visited_nodes: u64) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeBudget {
    limit: u64,
}

impl NodeBudget {
    pub(crate) const fn new(limit: u64) -> Self {
        Self { limit }
    }
}

impl SearchBudget for NodeBudget {
    fn exhausted(&mut self, visited_nodes: u64) -> bool {
        visited_nodes >= self.limit
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeadlineBudget {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    deadline: Instant,
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    started_millis: f64,
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    duration_millis: f64,
}

impl DeadlineBudget {
    pub(crate) fn new(duration: Duration) -> Option<Self> {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            Instant::now()
                .checked_add(duration)
                .map(|deadline| Self { deadline })
        }

        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            const MAX_SAFE_MILLIS: f64 = 9_007_199_254_740_991.0;
            let started_millis = performance_now_millis();
            let duration_millis = duration.as_secs_f64() * 1_000.0;
            (started_millis.is_finite()
                && duration_millis.is_finite()
                && duration_millis <= MAX_SAFE_MILLIS)
                .then_some(Self {
                    started_millis,
                    duration_millis,
                })
        }
    }
}

impl SearchBudget for DeadlineBudget {
    fn exhausted(&mut self, _visited_nodes: u64) -> bool {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            Instant::now() >= self.deadline
        }

        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            let elapsed_millis = performance_now_millis() - self.started_millis;
            !elapsed_millis.is_finite() || elapsed_millis >= self.duration_millis
        }
    }
}

#[cfg(test)]
pub(crate) struct CheckBudget {
    checks_remaining: u64,
}

#[cfg(test)]
impl CheckBudget {
    pub(crate) const fn new(checks_remaining: u64) -> Self {
        Self { checks_remaining }
    }
}

#[cfg(test)]
impl SearchBudget for CheckBudget {
    fn exhausted(&mut self, _visited_nodes: u64) -> bool {
        if self.checks_remaining == 0 {
            true
        } else {
            self.checks_remaining -= 1;
            false
        }
    }
}

/// Test budget that starts counting polls once an exact node total is reached.
/// It lets transaction tests cut off after search, during PV construction, or
/// at the final pre-commit check without relying on wall-clock timing.
#[cfg(test)]
pub(crate) struct ExactCutoffBudget {
    target_nodes: u64,
    allowed_polls_at_target: u64,
}

#[cfg(test)]
impl ExactCutoffBudget {
    pub(crate) const fn new(target_nodes: u64, allowed_polls_at_target: u64) -> Self {
        Self {
            target_nodes,
            allowed_polls_at_target,
        }
    }
}

#[cfg(test)]
impl SearchBudget for ExactCutoffBudget {
    fn exhausted(&mut self, visited_nodes: u64) -> bool {
        if visited_nodes < self.target_nodes {
            return false;
        }
        if self.allowed_polls_at_target == 0 {
            true
        } else {
            self.allowed_polls_at_target -= 1;
            false
        }
    }
}
