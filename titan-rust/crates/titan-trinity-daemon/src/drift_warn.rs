//! `DriftAggregator` — rate-limited summary emitter for tick-drift warnings.
//!
//! Inner-trinity daemons (body 7.83 Hz, mind 23.49 Hz, spirit 70.47 Hz) check
//! per-tick drift against the SPEC §16 OBS gate (0.5%). On a multi-tenant
//! cloud VM the gate fires routinely on benign sub-10% jitter, producing
//! ~115 WARN/sec across the three crates and several GB/day of duplicated
//! syslog/journal noise (see SESSION_20260508 — 40 GB syslog rooted in this
//! emitter).
//!
//! This aggregator preserves the SPEC threshold (every drift event is still
//! counted) while bounding emission to ≤1 WARN/sec/role, with a summary
//! payload (`drifts_in_window`, `max_drift_pct`, `avg_drift_pct`,
//! `max_jitter_ns`, `last_epoch`, `window_s`) so the OBS signal is preserved
//! at orders-of-magnitude lower volume.

use std::time::{Duration, Instant};

use tracing::warn;

/// Minimum interval between `DAEMON_TICK_DRIFT_HIGH` emissions per daemon.
pub const DRIFT_WARN_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Per-daemon drift accumulator. One instance per tick_loop.
#[derive(Debug)]
pub struct DriftAggregator {
    role: &'static str,
    threshold_pct: f64,
    last_emit: Instant,
    count: u64,
    max_drift_pct: f64,
    sum_drift_pct: f64,
    max_jitter_ns: u64,
    last_epoch: u64,
}

impl DriftAggregator {
    /// Create an aggregator for `role` (e.g. "body", "mind", "spirit") with
    /// the SPEC §16 OBS threshold in fractional form (0.005 = 0.5%).
    pub fn new(role: &'static str, threshold_pct: f64) -> Self {
        debug_assert!(
            threshold_pct > 0.0 && threshold_pct < 1.0,
            "DRIFT_THRESHOLD_PCT must be a fraction in (0,1), got {threshold_pct}"
        );
        Self {
            role,
            threshold_pct,
            last_emit: Instant::now(),
            count: 0,
            max_drift_pct: 0.0,
            sum_drift_pct: 0.0,
            max_jitter_ns: 0,
            last_epoch: 0,
        }
    }

    /// Record a tick's measured drift. If `drift_pct` exceeds the threshold
    /// the event is added to the window. When the window has been open for
    /// at least `DRIFT_WARN_MIN_INTERVAL` AND at least one over-threshold
    /// event has been observed, a single summary WARN is emitted and the
    /// window resets.
    pub fn observe(&mut self, drift_pct: f64, jitter_ns: u64, epoch: u64) {
        if drift_pct > self.threshold_pct {
            self.count += 1;
            if drift_pct > self.max_drift_pct {
                self.max_drift_pct = drift_pct;
            }
            self.sum_drift_pct += drift_pct;
            if jitter_ns > self.max_jitter_ns {
                self.max_jitter_ns = jitter_ns;
            }
            self.last_epoch = epoch;
        }

        let elapsed = self.last_emit.elapsed();
        if self.count > 0 && elapsed >= DRIFT_WARN_MIN_INTERVAL {
            let avg = self.sum_drift_pct / self.count as f64;
            warn!(
                event = "DAEMON_TICK_DRIFT_HIGH",
                role = self.role,
                drifts_in_window = self.count,
                max_drift_pct = self.max_drift_pct,
                avg_drift_pct = avg,
                max_jitter_ns = self.max_jitter_ns,
                last_epoch = self.last_epoch,
                window_s = elapsed.as_secs_f64(),
            );
            self.last_emit = Instant::now();
            self.count = 0;
            self.max_drift_pct = 0.0;
            self.sum_drift_pct = 0.0;
            self.max_jitter_ns = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn under_threshold_never_emits() {
        let mut agg = DriftAggregator::new("test", 0.005);
        for epoch in 0..1_000 {
            agg.observe(0.001, 10_000, epoch);
        }
        assert_eq!(agg.count, 0, "under-threshold ticks must not be counted");
    }

    #[test]
    fn over_threshold_within_window_aggregates_no_emit() {
        let mut agg = DriftAggregator::new("test", 0.005);
        for epoch in 0..50 {
            agg.observe(0.01, 100_000, epoch);
        }
        assert_eq!(
            agg.count, 50,
            "over-threshold within window must accumulate"
        );
        assert!(agg.max_drift_pct >= 0.01);
        assert_eq!(agg.last_epoch, 49);
    }

    #[test]
    fn emit_after_min_interval_resets_window() {
        let mut agg = DriftAggregator::new("test", 0.005);
        agg.observe(0.02, 100_000, 1);
        sleep(Duration::from_millis(1100));
        agg.observe(0.03, 200_000, 2);
        // After the second observe (>1s elapsed), window emitted + reset.
        assert_eq!(agg.count, 0, "emit must reset count");
        assert_eq!(agg.max_drift_pct, 0.0, "emit must reset max");
        assert_eq!(agg.sum_drift_pct, 0.0, "emit must reset sum");
    }

    #[test]
    fn no_over_threshold_no_emit_no_reset_state() {
        let mut agg = DriftAggregator::new("test", 0.005);
        sleep(Duration::from_millis(1100));
        agg.observe(0.001, 1_000, 1);
        // Time elapsed but no over-threshold events: nothing to emit, nothing to reset.
        assert_eq!(agg.count, 0);
    }
}
