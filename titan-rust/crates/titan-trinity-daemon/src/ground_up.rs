//! ground_up — Per-daemon GROUND_UP grounding nudge (Rust 1:1 port of
//! `titan_hcl/logic/ground_up.py`).
//!
//! # SPEC ground truths
//!
//! - **G5** — FILTER_DOWN + GROUND_UP both push toward 0.5 — productive
//!   tension, not cancellation.
//! - **G10** — GROUND_UP modifies Body **all 5D** + Mind **willing[10:15] only**.
//!   Body[0:5] full grounding; Mind[0:5] thinking + Mind[5:10] feeling are
//!   NOT grounded; Spirit[0:45] is NOT grounded.
//! - **G4** — `topology_30d[10:20]` is the inner_lower slice; daemons read
//!   this and split it into body[0:5] + mind[5:10] halves.
//!
//! # Per-daemon split (Rust port)
//!
//! Python `GroundUpEnricher` had a single class instance applying both body
//! AND mind nudges. The Rust port splits this into two independent enrichers
//! since each daemon is its own process:
//!
//! - **inner-body** instantiates a [`GroundUpEnricher`] in [`Side::Body`] mode;
//!   reads `topology_lower[0:5]` (the body half) and applies to body[0:5].
//! - **inner-mind** instantiates a [`GroundUpEnricher`] in [`Side::MindWilling`]
//!   mode; reads `topology_lower[5:10]` (the mind half) and applies to
//!   mind[10:15].
//!
//! The damped `prev_nudge` state is per-daemon (mirrors Python's
//! `_prev_nudge_body` / `_prev_nudge_mind` separation — they never
//! cross-couple in the Python compute path, so byte-identical parity holds).
//!
//! # Numerical parity guarantee
//!
//! Damping formula (per `ground_up.py:78-83`):
//! `nudge[i] = damping * prev[i] + (1.0 - damping) * raw[i]`
//!
//! Clamp (per `ground_up.py:87-88`): `nudge[i] := clamp(nudge[i], -MAX, +MAX)`
//!
//! Apply (per `ground_up.py:127-130`):
//! `delta = nudge[i] * strength * min(dt, 30.0)`
//! `value[i] := clamp(value[i] + delta, 0.0, 1.0)`
//!
//! All operations use `f32` (matches `state_registry.py` slot float32 layout
//! per SPEC §7.1; Python uses double internally but slot serialization
//! truncates to float32 anyway).

use crate::error::DaemonResult;

// SPEC v0.1.3 (C-S3) shipped these constants in titan-core::constants
// (f64). Daemon-facing f32 aliases below — slot payloads are float32 per
// SPEC §7.1, so daemons read/write f32; the cast at compile time is
// loss-less for the canonical decimal values 0.1 / 0.95 / 0.05.

/// Default grounding strength — conservative, same order as FILTER_DOWN.
/// Sourced from `titan_core::constants::GROUND_UP_DEFAULT_STRENGTH`
/// (matches `ground_up.py:21 DEFAULT_STRENGTH=0.1`).
pub const GROUND_UP_DEFAULT_STRENGTH: f32 =
    titan_core::constants::GROUND_UP_DEFAULT_STRENGTH as f32;

/// Damping factor to prevent oscillation overshoot.
/// Sourced from `titan_core::constants::GROUND_UP_DEFAULT_DAMPING`
/// (matches `ground_up.py:24 DEFAULT_DAMPING=0.95`).
pub const GROUND_UP_DEFAULT_DAMPING: f32 = titan_core::constants::GROUND_UP_DEFAULT_DAMPING as f32;

/// Maximum nudge per dimension per epoch (safety clamp).
/// Sourced from `titan_core::constants::GROUND_UP_MAX_NUDGE`
/// (matches `ground_up.py:27 MAX_NUDGE=0.05`).
pub const GROUND_UP_MAX_NUDGE: f32 = titan_core::constants::GROUND_UP_MAX_NUDGE as f32;

/// dt clamp upper bound per `ground_up.py:127 min(dt, 30.0)`.
/// Prevents runaway nudges after long pauses.
pub const GROUND_UP_DT_CLAMP_S: f32 = 30.0;

/// Which half of the 10D inner_lower topology this enricher reads + which
/// daemon dimensions it applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// `topology_lower[0:5]` → body[0:5] (all 5 dims grounded). G10.
    Body,
    /// `topology_lower[5:10]` → mind[10:15] (willing dims only). G10.
    MindWilling,
}

/// Result of one nudge computation. Useful for tests + parity vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundUpNudge {
    /// 5-element nudge vector (clamped to ±[`GROUND_UP_MAX_NUDGE`]).
    pub nudge: [f32; 5],
    /// L2 magnitude of the nudge (for stats / observability).
    pub magnitude: f32,
}

/// Stateful ground_up enricher for one side of the daemon split. Holds the
/// damped `prev_nudge` vector that smooths nudges between consecutive ticks.
///
/// Stateless after `new(side, ...)` until `compute_nudge` or `apply` is
/// called — first call's damping uses `prev_nudge = [0.0; 5]` (matches
/// Python `__init__` initialization at `ground_up.py:45`).
#[derive(Debug, Clone)]
pub struct GroundUpEnricher {
    /// Which slice of topology_lower this enricher reads.
    pub side: Side,
    /// Strength multiplier (0.0 = off, 1.0 = max). Default
    /// [`GROUND_UP_DEFAULT_STRENGTH`].
    pub strength: f32,
    /// Damping smoothing factor. Default [`GROUND_UP_DEFAULT_DAMPING`].
    pub damping: f32,
    /// Persisted damped nudge from the previous tick. Initialized to zeros.
    prev_nudge: [f32; 5],
    /// Cumulative apply count (for stats; not part of slot output).
    apply_count: u64,
    /// Cumulative |nudge| sum across all ticks (for stats).
    cumulative_delta: f32,
}

impl GroundUpEnricher {
    /// New enricher with default strength + damping.
    pub fn new(side: Side) -> Self {
        Self {
            side,
            strength: GROUND_UP_DEFAULT_STRENGTH,
            damping: GROUND_UP_DEFAULT_DAMPING,
            prev_nudge: [0.0; 5],
            apply_count: 0,
            cumulative_delta: 0.0,
        }
    }

    /// New enricher with explicit strength + damping (for tuning + tests).
    pub fn with_params(side: Side, strength: f32, damping: f32) -> Self {
        Self {
            side,
            strength,
            damping,
            prev_nudge: [0.0; 5],
            apply_count: 0,
            cumulative_delta: 0.0,
        }
    }

    /// Total number of `apply()` calls so far. Used by daemon stats endpoints.
    pub fn apply_count(&self) -> u64 {
        self.apply_count
    }

    /// Cumulative |nudge| sum across all ticks. Used by daemon stats.
    pub fn cumulative_delta(&self) -> f32 {
        self.cumulative_delta
    }

    /// The damped previous-nudge vector. Used for hot-reload state
    /// preservation (mirrors Python `get_state()`).
    pub fn prev_nudge(&self) -> [f32; 5] {
        self.prev_nudge
    }

    /// Restore from a snapshot (mirrors Python `restore_state()`).
    pub fn restore_prev_nudge(&mut self, prev: [f32; 5]) {
        self.prev_nudge = prev;
    }

    /// Compute the damped + clamped 5D nudge for this side from the FULL
    /// 10D topology_lower vector.
    ///
    /// Updates `self.prev_nudge` (damping continuity) but does NOT modify
    /// the enricher's stats counters — those update only on `apply`.
    pub fn compute_nudge(&mut self, topology_lower: &[f32; 10]) -> GroundUpNudge {
        // Slice per Side per ground_up.py:73-74:
        //     raw_body = signal[0:5]
        //     raw_mind = signal[5:10]
        let raw: [f32; 5] = match self.side {
            Side::Body => [
                topology_lower[0],
                topology_lower[1],
                topology_lower[2],
                topology_lower[3],
                topology_lower[4],
            ],
            Side::MindWilling => [
                topology_lower[5],
                topology_lower[6],
                topology_lower[7],
                topology_lower[8],
                topology_lower[9],
            ],
        };

        // Damping (ground_up.py:77-83):
        //     nudge[i] = damping * prev[i] + (1.0 - damping) * raw[i]
        let inv_damp = 1.0 - self.damping;
        let mut nudge = [0.0_f32; 5];
        for i in 0..5 {
            nudge[i] = self.damping * self.prev_nudge[i] + inv_damp * raw[i];
        }

        // Clamp (ground_up.py:87-88):
        for n in nudge.iter_mut() {
            *n = n.clamp(-GROUND_UP_MAX_NUDGE, GROUND_UP_MAX_NUDGE);
        }

        // Persist prev_nudge for next tick (ground_up.py:90-91)
        self.prev_nudge = nudge;

        // Magnitude (ground_up.py:93-95) — sqrt(sum(n*n))
        let mag_sq: f32 = nudge.iter().map(|n| n * n).sum();
        let magnitude = mag_sq.sqrt();

        GroundUpNudge { nudge, magnitude }
    }

    /// Apply the currently-**held** nudge (`self.prev_nudge`, last set by
    /// [`compute_nudge`]) to a body[0:5] vector in place — WITHOUT recomputing.
    /// Returns Err if `self.side != Side::Body`.
    ///
    /// SPEC §G5.1 / Phase 0 chunk 0E (D-SPEC-97 refinement, v1.36.2): the
    /// nudge is *recomputed* only on `KERNEL_EPOCH_TICK` (via
    /// [`compute_nudge`], which evolves the per-epoch 0.95 damping EMA at
    /// epoch cadence); this method *applies* that held value every Schumann
    /// tick so the grounding offset manifests continuously in the per-tick-
    /// recomputed tensor (symmetric with the held filter_down multiplier).
    /// `dt_s` is the per-epoch unit (1.0) — applied to each tick's fresh body
    /// it holds a steady offset (no compounding: body is rebuilt each tick).
    ///
    /// Implements the apply half of `ground_up.py:124-130`:
    ///     `delta = held_nudge[i] * strength * min(dt, 30.0); body[i] = clamp(body[i] + delta, 0.0, 1.0)`
    pub fn apply_held_to_body(&mut self, body: &mut [f32; 5], dt_s: f32) -> DaemonResult<()> {
        if self.side != Side::Body {
            // Programming error — daemon constructed wrong-Side enricher.
            // Use a structured error so the binary's main.rs can log + exit.
            return Err(crate::error::DaemonError::DimMismatch {
                expected: 5,
                expected_bytes: 20,
                actual_bytes: 0,
            });
        }
        let dt_clamped = dt_s.clamp(0.0, GROUND_UP_DT_CLAMP_S);
        for (b, n) in body.iter_mut().zip(self.prev_nudge.iter()) {
            let delta = n * self.strength * dt_clamped;
            *b = (*b + delta).clamp(0.0, 1.0);
        }
        self.apply_count = self.apply_count.saturating_add(1);
        let abs_sum: f32 = self.prev_nudge.iter().map(|n| n.abs()).sum();
        self.cumulative_delta += abs_sum;
        Ok(())
    }

    /// Apply the currently-**held** nudge to a mind[0:15] vector — modifies
    /// ONLY `mind[10:15]` (willing) per G10, WITHOUT recomputing. Returns Err
    /// if `self.side != Side::MindWilling`. See [`apply_held_to_body`] for the
    /// 0E held-nudge contract.
    ///
    /// Implements the apply half of `ground_up.py:133-140`:
    ///     `delta = held_nudge[i] * strength * min(dt, 30.0); mind[10+i] = clamp(mind[10+i] + delta, 0.0, 1.0)`
    pub fn apply_held_to_mind(&mut self, mind: &mut [f32; 15], dt_s: f32) -> DaemonResult<()> {
        if self.side != Side::MindWilling {
            return Err(crate::error::DaemonError::DimMismatch {
                expected: 15,
                expected_bytes: 60,
                actual_bytes: 0,
            });
        }
        let dt_clamped = dt_s.clamp(0.0, GROUND_UP_DT_CLAMP_S);
        for i in 0..5 {
            let delta = self.prev_nudge[i] * self.strength * dt_clamped;
            mind[10 + i] = (mind[10 + i] + delta).clamp(0.0, 1.0);
        }
        self.apply_count = self.apply_count.saturating_add(1);
        let abs_sum: f32 = self.prev_nudge.iter().map(|n| n.abs()).sum();
        self.cumulative_delta += abs_sum;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn body_side_reads_topology_lower_first_half() {
        let mut e = GroundUpEnricher::new(Side::Body);
        // Topology lower = [1, 2, 3, 4, 5, 90, 91, 92, 93, 94] —
        // body should read [1, 2, 3, 4, 5] (clamped to MAX_NUDGE=0.05).
        let topo: [f32; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 90.0, 91.0, 92.0, 93.0, 94.0];
        let nudge = e.compute_nudge(&topo);
        // damping=0.95, prev=0 → nudge = 0.05 * raw, then clamp ±0.05 → all 0.05
        for n in nudge.nudge.iter() {
            assert!(approx_eq(*n, GROUND_UP_MAX_NUDGE, 1e-6));
        }
    }

    #[test]
    fn mind_side_reads_topology_lower_second_half() {
        let mut e = GroundUpEnricher::new(Side::MindWilling);
        let topo: [f32; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 90.0, 91.0, 92.0, 93.0, 94.0];
        let nudge = e.compute_nudge(&topo);
        // mind reads [90, 91, ...] — all clamp to +0.05
        for n in nudge.nudge.iter() {
            assert!(approx_eq(*n, GROUND_UP_MAX_NUDGE, 1e-6));
        }
    }

    #[test]
    fn damping_smooths_across_calls() {
        // damping=0.95 → smoothing across multiple compute_nudge calls.
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo_first: [f32; 10] = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let n1 = e.compute_nudge(&topo_first);
        // First call: 0.95*0 + 0.05*0.04 = 0.002
        assert!(approx_eq(n1.nudge[0], 0.002, 1e-6));

        let topo_second: [f32; 10] = [0.0; 10];
        let n2 = e.compute_nudge(&topo_second);
        // Second: 0.95*0.002 + 0.05*0.0 = 0.0019
        assert!(approx_eq(n2.nudge[0], 0.0019, 1e-6));
    }

    #[test]
    fn nudge_clamped_to_max() {
        // Big topology values → clamped to ±MAX_NUDGE
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [100.0, -100.0, 100.0, -100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let nudge = e.compute_nudge(&topo);
        assert!(approx_eq(nudge.nudge[0], GROUND_UP_MAX_NUDGE, 1e-6));
        assert!(approx_eq(nudge.nudge[1], -GROUND_UP_MAX_NUDGE, 1e-6));
    }

    #[test]
    fn apply_to_body_updates_all_5_dims() {
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [0.04, 0.04, 0.04, 0.04, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut body: [f32; 5] = [0.5; 5];
        // 0E: recompute the held nudge on the (simulated) epoch, then apply it.
        e.compute_nudge(&topo);
        e.apply_held_to_body(&mut body, 1.0).unwrap();
        // delta = 0.002 * 0.1 * 1.0 = 0.0002 per dim
        for v in body.iter() {
            assert!(approx_eq(*v, 0.5002, 1e-6));
        }
    }

    #[test]
    fn apply_to_mind_only_touches_willing_10_15() {
        let mut e = GroundUpEnricher::new(Side::MindWilling);
        // Strong topo on the mind half [5:10]
        let topo: [f32; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04, 0.04];
        let mut mind: [f32; 15] = [0.5; 15];
        e.compute_nudge(&topo);
        e.apply_held_to_mind(&mut mind, 1.0).unwrap();
        // [0:5] thinking + [5:10] feeling unchanged
        for i in 0..10 {
            assert!(
                approx_eq(mind[i], 0.5, 1e-6),
                "mind[{i}] should be untouched"
            );
        }
        // [10:15] willing nudged by delta = 0.002 * 0.1 * 1.0 = 0.0002
        for i in 10..15 {
            assert!(
                approx_eq(mind[i], 0.5002, 1e-6),
                "mind[{i}] should be grounded"
            );
        }
    }

    #[test]
    fn apply_clamps_to_0_1_range() {
        // Body already at 1.0 + positive nudge → clamps to 1.0
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [100.0; 10];
        let mut body: [f32; 5] = [1.0; 5];
        e.compute_nudge(&topo);
        e.apply_held_to_body(&mut body, 30.0).unwrap();
        for v in body.iter() {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
        // Body at 0.0 + negative nudge → clamps to 0.0
        let mut e2 = GroundUpEnricher::new(Side::Body);
        let neg_topo: [f32; 10] = [-100.0; 10];
        let mut body2: [f32; 5] = [0.0; 5];
        e2.compute_nudge(&neg_topo);
        e2.apply_held_to_body(&mut body2, 30.0).unwrap();
        for v in body2.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6));
        }
    }

    #[test]
    fn apply_to_body_rejects_mind_side_enricher() {
        let mut e = GroundUpEnricher::new(Side::MindWilling);
        let mut body: [f32; 5] = [0.5; 5];
        let r = e.apply_held_to_body(&mut body, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn apply_to_mind_rejects_body_side_enricher() {
        let mut e = GroundUpEnricher::new(Side::Body);
        let mut mind: [f32; 15] = [0.5; 15];
        let r = e.apply_held_to_mind(&mut mind, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn dt_clamped_to_30s_max() {
        // dt=1000 should be clamped to 30, NOT cause runaway nudge.
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [0.04; 10];
        let mut body: [f32; 5] = [0.5; 5];
        e.compute_nudge(&topo);
        e.apply_held_to_body(&mut body, 1000.0).unwrap();
        // Expected: delta = 0.002 * 0.1 * 30 = 0.006 per dim
        for v in body.iter() {
            assert!(approx_eq(*v, 0.506, 1e-5));
        }
    }

    #[test]
    fn negative_dt_clamped_to_zero() {
        // dt < 0 (clock skew): clamp to 0, no nudge applied.
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [0.04; 10];
        let mut body: [f32; 5] = [0.5; 5];
        e.compute_nudge(&topo);
        e.apply_held_to_body(&mut body, -5.0).unwrap();
        for v in body.iter() {
            assert!(approx_eq(*v, 0.5, 1e-6));
        }
    }

    #[test]
    fn restore_prev_nudge_resumes_damping() {
        let mut e = GroundUpEnricher::new(Side::Body);
        let saved: [f32; 5] = [0.04, 0.04, 0.04, 0.04, 0.04];
        e.restore_prev_nudge(saved);
        let topo: [f32; 10] = [0.0; 10];
        let n = e.compute_nudge(&topo);
        // 0.95 * 0.04 + 0.05 * 0 = 0.038
        for v in n.nudge.iter() {
            assert!(approx_eq(*v, 0.038, 1e-5));
        }
    }

    #[test]
    fn apply_count_increments_on_each_apply() {
        let mut e = GroundUpEnricher::new(Side::Body);
        let topo: [f32; 10] = [0.0; 10];
        let mut body: [f32; 5] = [0.5; 5];
        // compute_nudge alone does NOT bump apply_count (0E: compute is the
        // epoch event; apply is per-tick) — only apply_held_to_body does.
        e.compute_nudge(&topo);
        assert_eq!(e.apply_count(), 0);
        e.apply_held_to_body(&mut body, 1.0).unwrap();
        assert_eq!(e.apply_count(), 1);
        e.apply_held_to_body(&mut body, 1.0).unwrap();
        assert_eq!(e.apply_count(), 2);
    }

    #[test]
    fn nudge_magnitude_matches_l2_norm() {
        let mut e = GroundUpEnricher::new(Side::Body);
        // Pre-load prev_nudge so first compute returns it (damping≈0.95 dominates)
        e.restore_prev_nudge([0.05, 0.0, 0.0, 0.0, 0.0]);
        let topo: [f32; 10] = [0.0; 10];
        let nudge = e.compute_nudge(&topo);
        // Expected: nudge[0] = 0.95 * 0.05 = 0.0475; magnitude = 0.0475
        assert!(approx_eq(nudge.magnitude, 0.0475, 1e-5));
    }
}
