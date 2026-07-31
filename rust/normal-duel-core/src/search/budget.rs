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

/// Fixed-node regression searches reserve 4,096 nodes for a canonical 9x9
/// exact-oracle attempt (8.192% of the standard 50,000-node evaluation run).
pub(crate) const FIXED_ORACLE_NODE_QUOTA: u64 = 4_096;

// Deadline searches retain a duration-scaled node cap as a secondary hard
// bound on an unexpectedly large exact-oracle tree. Their primary bound is a
// local wall-clock slice equal to 10% of the requested duration, clamped to
// 1–1,500 ms. This quota deliberately is not justified by, or coupled to,
// ordinary negamax throughput.
const DEADLINE_ORACLE_NODES_PER_MILLISECOND: u128 = 50;
const DEADLINE_ORACLE_MAX_NODES: u128 = 750_000;
const DEADLINE_ORACLE_SLICE_DIVISOR: u32 = 10;
const DEADLINE_ORACLE_MIN_SLICE: Duration = Duration::from_millis(1);
const DEADLINE_ORACLE_MAX_SLICE: Duration = Duration::from_millis(1_500);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CanonicalOracleLimits {
    pub(crate) node_quota: u64,
    /// `None` keeps deterministic fixed-node searches independent of time.
    pub(crate) wall_clock_slice: Option<Duration>,
}

pub(crate) fn deadline_oracle_node_quota(duration: Duration) -> u64 {
    let quota = duration
        .as_millis()
        .saturating_mul(DEADLINE_ORACLE_NODES_PER_MILLISECOND)
        .clamp(
            u128::from(FIXED_ORACLE_NODE_QUOTA),
            DEADLINE_ORACLE_MAX_NODES,
        );
    u64::try_from(quota).unwrap_or(u64::MAX)
}

pub(crate) fn deadline_oracle_wall_clock_slice(duration: Duration) -> Duration {
    (duration / DEADLINE_ORACLE_SLICE_DIVISOR)
        .clamp(DEADLINE_ORACLE_MIN_SLICE, DEADLINE_ORACLE_MAX_SLICE)
}

pub(crate) trait SearchBudget {
    fn exhausted(&mut self, visited_nodes: u64) -> bool;

    fn canonical_oracle_limits(&self) -> CanonicalOracleLimits {
        CanonicalOracleLimits {
            node_quota: FIXED_ORACLE_NODE_QUOTA,
            wall_clock_slice: None,
        }
    }
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
    duration: Duration,
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
                .map(|deadline| Self { duration, deadline })
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
                    duration,
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

    fn canonical_oracle_limits(&self) -> CanonicalOracleLimits {
        CanonicalOracleLimits {
            node_quota: deadline_oracle_node_quota(self.duration),
            wall_clock_slice: Some(deadline_oracle_wall_clock_slice(self.duration)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_oracle_quota_scales_and_caps_without_overflow() {
        assert_eq!(
            deadline_oracle_node_quota(Duration::ZERO),
            FIXED_ORACLE_NODE_QUOTA
        );
        assert_eq!(
            deadline_oracle_node_quota(Duration::from_millis(1)),
            FIXED_ORACLE_NODE_QUOTA
        );
        assert_eq!(
            deadline_oracle_node_quota(Duration::from_millis(800)),
            40_000
        );
        assert_eq!(deadline_oracle_node_quota(Duration::from_secs(15)), 750_000);
        assert_eq!(deadline_oracle_node_quota(Duration::from_secs(16)), 750_000);
        assert_eq!(deadline_oracle_node_quota(Duration::MAX), 750_000);
    }

    #[test]
    fn deadline_oracle_slice_scales_without_timing_or_overflow() {
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::ZERO),
            Duration::from_millis(1)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::from_millis(800)),
            Duration::from_millis(80)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::from_millis(1_900)),
            Duration::from_millis(190)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::from_millis(4_900)),
            Duration::from_millis(490)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::from_millis(9_900)),
            Duration::from_millis(990)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::from_millis(14_900)),
            Duration::from_millis(1_490)
        );
        assert_eq!(
            deadline_oracle_wall_clock_slice(Duration::MAX),
            Duration::from_millis(1_500)
        );
    }

    #[test]
    fn budget_modes_report_distinct_canonical_limits_without_timing() {
        let fixed = NodeBudget::new(50_000);
        assert_eq!(
            fixed.canonical_oracle_limits(),
            CanonicalOracleLimits {
                node_quota: FIXED_ORACLE_NODE_QUOTA,
                wall_clock_slice: None,
            }
        );

        let deadline = DeadlineBudget::new(Duration::from_millis(800)).unwrap();
        assert_eq!(
            deadline.canonical_oracle_limits(),
            CanonicalOracleLimits {
                node_quota: 40_000,
                wall_clock_slice: Some(Duration::from_millis(80)),
            }
        );
    }
}
