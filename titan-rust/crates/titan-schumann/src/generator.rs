//! generator — `SchumannGenerator` + `TickEvent` + role-specific aliases.
//!
//! Per PLAN §8 implementation contract.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::sync::{mpsc, Notify};
use tokio::time::{interval_at, Instant, MissedTickBehavior};
use tracing::{debug, warn};

use titan_core::constants::{SCHUMANN_BODY_HZ, SCHUMANN_MIND_HZ, SCHUMANN_SPIRIT_HZ};

/// Schumann role — biological constant, NOT a runtime parameter.
///
/// Per Preamble G13: 7.83 Hz body / 23.49 Hz mind (×3) / 70.47 Hz spirit (×9).
/// Phase relationships are exact small-integer ratios; per-tick frequency
/// values are the locked LOCKED-BY-BIOLOGY constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchumannRole {
    /// Body cadence — 7.83 Hz fundamental.
    Body,
    /// Mind cadence — 23.49 Hz (= body × 3).
    Mind,
    /// Spirit cadence — 70.47 Hz (= body × 9).
    Spirit,
}

impl SchumannRole {
    /// Canonical role name for logging / observability.
    pub fn as_str(&self) -> &'static str {
        match self {
            SchumannRole::Body => "body",
            SchumannRole::Mind => "mind",
            SchumannRole::Spirit => "spirit",
        }
    }

    /// Frequency in Hz — read from `titan_core::constants` (auto-generated
    /// from SPEC TOML). NOT a parameter.
    pub fn hz(&self) -> f64 {
        match self {
            SchumannRole::Body => SCHUMANN_BODY_HZ,
            SchumannRole::Mind => SCHUMANN_MIND_HZ,
            SchumannRole::Spirit => SCHUMANN_SPIRIT_HZ,
        }
    }
}

/// Period in nanoseconds for a given role. `period_ns = floor(1e9 / hz)` —
/// truncation toward integer ns. Verified by parity vectors:
///
/// - body   ≈ 127_713_921 ns
/// - mind   ≈  42_571_307 ns
/// - spirit ≈  14_190_435 ns
pub const fn period_ns_for_role(role: SchumannRole) -> u64 {
    let hz = match role {
        SchumannRole::Body => SCHUMANN_BODY_HZ,
        SchumannRole::Mind => SCHUMANN_MIND_HZ,
        SchumannRole::Spirit => SCHUMANN_SPIRIT_HZ,
    };
    (1_000_000_000.0_f64 / hz) as u64
}

/// One emitted tick event from a [`SchumannGenerator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickEvent {
    /// Monotonic tick counter; resets on substrate boot. Survives within a
    /// single substrate boot generation.
    pub epoch: u64,
    /// Wall-clock nanoseconds since the shared `epoch_t0` when this tick was
    /// SCHEDULED to fire (`epoch * period_ns`).
    pub scheduled_ns: u64,
    /// Wall-clock nanoseconds since `epoch_t0` when this tick ACTUALLY fired.
    /// Jitter = `observed_ns - scheduled_ns` (always ≥ 0 by tokio guarantee
    /// when the timer fires after its deadline).
    pub observed_ns: u64,
    /// Period of this generator in nanoseconds.
    pub period_ns: u64,
    /// Generator role.
    pub role: SchumannRole,
}

impl TickEvent {
    /// Per-tick jitter (observed − scheduled) in nanoseconds. Always ≥ 0.
    pub fn jitter_ns(&self) -> u64 {
        self.observed_ns.saturating_sub(self.scheduled_ns)
    }
}

/// Errors during generator setup / spawn.
#[derive(Debug, Error)]
pub enum SchumannError {
    /// Caller attempted to override the locked Schumann frequency. Per G13
    /// LOCKED: frequencies are biological, NOT tunable.
    #[error("Schumann frequency is LOCKED (Preamble G13) — overriding is forbidden")]
    FrequencyOverrideForbidden,
}

/// A Schumann tick generator — wraps a tokio interval pinned to a shared
/// `epoch_t0` for cross-generator phase alignment.
///
/// Use [`SchumannGenerator::new`] to construct, then [`SchumannGenerator::spawn`]
/// to start emitting ticks on a returned mpsc receiver.
#[derive(Debug)]
pub struct SchumannGenerator {
    role: SchumannRole,
    period_ns: u64,
    epoch_t0: Instant,
    epoch: Arc<AtomicU64>,
}

impl SchumannGenerator {
    /// Create a generator for a given role pinned to `epoch_t0` (typically the
    /// substrate's boot instant — shared across all 3 role generators so
    /// phases are aligned).
    pub fn new(role: SchumannRole, epoch_t0: Instant) -> Self {
        Self {
            role,
            period_ns: period_ns_for_role(role),
            epoch_t0,
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Read-only access to the role — useful for diagnostics + tests.
    pub fn role(&self) -> SchumannRole {
        self.role
    }

    /// Period in nanoseconds (locked per role).
    pub fn period_ns(&self) -> u64 {
        self.period_ns
    }

    /// Reference to the shared epoch counter. Useful for cross-generator
    /// inspection (e.g., asserting phase alignment in tests).
    pub fn epoch_counter(&self) -> Arc<AtomicU64> {
        self.epoch.clone()
    }

    /// Spawn the generator as a tokio task. Returns the receiver for
    /// [`TickEvent`]s. The task exits when `shutdown` is notified.
    ///
    /// Per PLAN §8.3:
    /// - `MissedTickBehavior::Skip` — under runtime stress, advance to next
    ///   scheduled tick rather than burst-catch-up. Bounds peak load.
    /// - bounded `mpsc::channel(64)` — backpressure if consumer falls behind.
    pub fn spawn(self, shutdown: Arc<Notify>) -> mpsc::Receiver<TickEvent> {
        let (tx, rx) = mpsc::channel::<TickEvent>(64);
        let role = self.role;
        let period_ns = self.period_ns;
        let epoch_t0 = self.epoch_t0;
        let epoch = self.epoch;

        tokio::spawn(async move {
            let mut interval = interval_at(epoch_t0, Duration::from_nanos(period_ns));
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                let shutdown_fut = shutdown.notified();
                tokio::select! {
                    instant = interval.tick() => {
                        let tick_idx = epoch.fetch_add(1, Ordering::Release);
                        let scheduled_ns = (instant - epoch_t0).as_nanos() as u64;
                        let observed_ns = (Instant::now() - epoch_t0).as_nanos() as u64;
                        let event = TickEvent {
                            epoch: tick_idx,
                            scheduled_ns,
                            observed_ns,
                            period_ns,
                            role,
                        };
                        // Try-send: if consumer is too slow, drop the tick + warn (rate-limited)
                        // rather than block the timer wheel. This honors the bounded-mpsc contract.
                        if let Err(e) = tx.try_send(event) {
                            match e {
                                mpsc::error::TrySendError::Full(_) => {
                                    if tick_idx.is_multiple_of(100) {
                                        warn!(
                                            role = role.as_str(),
                                            epoch = tick_idx,
                                            "schumann: consumer slow — dropped tick (queue full)"
                                        );
                                    }
                                }
                                mpsc::error::TrySendError::Closed(_) => {
                                    debug!(
                                        role = role.as_str(),
                                        "schumann: consumer dropped — generator exiting"
                                    );
                                    break;
                                }
                            }
                        }
                    }
                    _ = shutdown_fut => {
                        debug!(
                            role = role.as_str(),
                            "schumann: shutdown signal received — generator exiting"
                        );
                        break;
                    }
                }
            }
        });

        rx
    }
}

/// Convenience alias — body 7.83 Hz Schumann generator.
pub type BodySchumann = SchumannGenerator;
/// Convenience alias — mind 23.49 Hz (×3) Schumann generator.
pub type MindSchumann = SchumannGenerator;
/// Convenience alias — spirit 70.47 Hz (×9) Schumann generator.
pub type SpiritSchumann = SchumannGenerator;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;
    use titan_core::constants::SCHUMANN_DRIFT_TARGET_PCT;

    /// Default test runtime: paused-time mode for deterministic assertions.
    fn paused_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap()
    }

    #[test]
    fn period_ns_for_each_role_matches_spec_vectors() {
        // Per tests/parity/vectors.json::schumann_periods_ns ±1 ns tolerance.
        let body = period_ns_for_role(SchumannRole::Body);
        let mind = period_ns_for_role(SchumannRole::Mind);
        let spirit = period_ns_for_role(SchumannRole::Spirit);
        // Allow ±1 ns rounding tolerance from f64 division
        assert!(
            (body as i64 - 127_713_921).abs() <= 1,
            "body period_ns = {body}, expected 127713921 ± 1"
        );
        assert!(
            (mind as i64 - 42_571_307).abs() <= 1,
            "mind period_ns = {mind}, expected 42571307 ± 1"
        );
        assert!(
            (spirit as i64 - 14_190_435).abs() <= 1,
            "spirit period_ns = {spirit}, expected 14190435 ± 1"
        );
    }

    #[test]
    fn role_hz_returns_locked_constants() {
        assert_eq!(SchumannRole::Body.hz(), SCHUMANN_BODY_HZ);
        assert_eq!(SchumannRole::Mind.hz(), SCHUMANN_MIND_HZ);
        assert_eq!(SchumannRole::Spirit.hz(), SCHUMANN_SPIRIT_HZ);
        // Locked frequencies — NOT tunable
        assert_eq!(SchumannRole::Body.hz(), 7.83);
        assert_eq!(SchumannRole::Mind.hz(), 23.49);
        assert_eq!(SchumannRole::Spirit.hz(), 70.47);
    }

    #[test]
    fn role_as_str_canonical_names() {
        assert_eq!(SchumannRole::Body.as_str(), "body");
        assert_eq!(SchumannRole::Mind.as_str(), "mind");
        assert_eq!(SchumannRole::Spirit.as_str(), "spirit");
    }

    #[test]
    fn tick_event_jitter_is_observed_minus_scheduled() {
        let ev = TickEvent {
            epoch: 5,
            scheduled_ns: 1_000,
            observed_ns: 1_500,
            period_ns: period_ns_for_role(SchumannRole::Body),
            role: SchumannRole::Body,
        };
        assert_eq!(ev.jitter_ns(), 500);
    }

    #[test]
    fn tick_event_jitter_saturates_when_observed_before_scheduled() {
        // Should never happen from real generator (observed >= scheduled by tokio
        // contract — interval.tick() returns the deadline that already passed),
        // but saturate to zero on synthetic inputs to avoid underflow.
        let ev = TickEvent {
            epoch: 0,
            scheduled_ns: 1_000,
            observed_ns: 500,
            period_ns: period_ns_for_role(SchumannRole::Body),
            role: SchumannRole::Body,
        };
        assert_eq!(ev.jitter_ns(), 0);
    }

    #[test]
    fn generator_records_role_and_period() {
        let rt = paused_runtime();
        let _g = rt.enter();
        let epoch_t0 = Instant::now();
        let g = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
        assert_eq!(g.role(), SchumannRole::Spirit);
        assert_eq!(g.period_ns(), period_ns_for_role(SchumannRole::Spirit));
    }

    #[test]
    fn generator_emits_ticks_with_correct_role_and_period() {
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
            let period_ns = g.period_ns();
            let mut rx = g.spawn(shutdown.clone());

            // With MissedTickBehavior::Skip, advancing N periods at once
            // collapses to a single tick. Advance one period at a time so we
            // actually observe each tick. Yield between to let the spawned
            // generator task get scheduled.
            let mut events = Vec::with_capacity(5);
            for _ in 0..5 {
                tokio::time::advance(StdDuration::from_nanos(period_ns)).await;
                let ev = rx
                    .recv()
                    .await
                    .expect("generator should emit per-period tick");
                events.push(ev);
            }

            assert_eq!(events.len(), 5);
            for (i, ev) in events.iter().enumerate() {
                assert_eq!(ev.role, SchumannRole::Body);
                assert_eq!(ev.period_ns, period_ns);
                assert_eq!(ev.epoch, i as u64);
            }
            shutdown.notify_waiters();
        });
    }

    #[test]
    fn epoch_counter_increments_monotonically() {
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
            let counter = g.epoch_counter();
            let mut rx = g.spawn(shutdown.clone());

            let period_ns = period_ns_for_role(SchumannRole::Spirit);
            // Drive 10 ticks one at a time + receive each, so the spawned task
            // actually advances the counter.
            for expected in 0..10u64 {
                tokio::time::advance(StdDuration::from_nanos(period_ns)).await;
                let ev = rx.recv().await.expect("tick");
                assert_eq!(ev.epoch, expected);
            }
            let n = counter.load(Ordering::Acquire);
            assert_eq!(n, 10, "counter must equal observed tick count");
            shutdown.notify_waiters();
        });
    }

    #[test]
    fn shutdown_signal_stops_generator() {
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
            let mut rx = g.spawn(shutdown.clone());

            // Advance one period so generator emits something
            tokio::time::advance(StdDuration::from_nanos(period_ns_for_role(
                SchumannRole::Body,
            )))
            .await;
            tokio::task::yield_now().await;
            let _first = rx.try_recv();

            // Notify shutdown
            shutdown.notify_waiters();
            tokio::task::yield_now().await;
            tokio::task::yield_now().await;

            // After shutdown, sender side is dropped → recv returns None eventually
            // when the generator's tx goes out of scope. With paused time + yields
            // we expect this within a couple of yields.
            // Drain any pending events first
            while rx.try_recv().is_ok() {}
            // Advance further; shutdown should keep generator from emitting more
            tokio::time::advance(StdDuration::from_secs(10)).await;
            tokio::task::yield_now().await;
            // After generator task exits, sender drops → recv returns None
            let final_recv = rx.recv().await;
            assert!(final_recv.is_none(), "generator should have stopped");
        });
    }

    #[test]
    fn three_role_generators_share_epoch_t0_for_phase_alignment() {
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let body_g = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
            let mind_g = SchumannGenerator::new(SchumannRole::Mind, epoch_t0);
            let spirit_g = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
            let body_period = body_g.period_ns();
            let mind_period = mind_g.period_ns();
            let spirit_period = spirit_g.period_ns();

            // Phase ratio invariants — body:mind:spirit = 1:3:9 by construction
            // (these are the SPEC-locked frequencies; ratio rounding tolerance ~1ns/period)
            let mind_to_body_ratio = mind_period as f64 / body_period as f64;
            let spirit_to_body_ratio = spirit_period as f64 / body_period as f64;
            // mind 23.49/7.83 ≈ 0.333... (= 1/3)
            assert!(
                (mind_to_body_ratio - 1.0 / 3.0).abs() < 0.0001,
                "mind/body period ratio {mind_to_body_ratio} ≠ 1/3 (within 0.01%)"
            );
            // spirit 70.47/7.83 ≈ 0.111... (= 1/9)
            assert!(
                (spirit_to_body_ratio - 1.0 / 9.0).abs() < 0.0001,
                "spirit/body period ratio {spirit_to_body_ratio} ≠ 1/9 (within 0.01%)"
            );

            let _body_rx = body_g.spawn(shutdown.clone());
            let _mind_rx = mind_g.spawn(shutdown.clone());
            let _spirit_rx = spirit_g.spawn(shutdown.clone());
            shutdown.notify_waiters();
        });
    }

    #[test]
    fn scheduled_ns_is_monotonic_and_period_aligned() {
        // In paused-time + MissedTickBehavior::Skip mode, `scheduled_ns` is
        // NOT necessarily `epoch * period_ns` (ticks may be skipped under
        // delay), but it MUST be monotonically increasing AND a multiple of
        // period_ns (Schumann timer wheel never fires off-grid).
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
            let period_ns = g.period_ns();
            let mut rx = g.spawn(shutdown.clone());

            let mut events = Vec::with_capacity(20);
            for _ in 0..20 {
                tokio::time::advance(StdDuration::from_nanos(period_ns)).await;
                events.push(rx.recv().await.expect("tick"));
            }
            assert_eq!(events.len(), 20);

            // Invariant 1: epoch counter is contiguous (one fetch_add per emitted event)
            for (i, ev) in events.iter().enumerate() {
                assert_eq!(ev.epoch, i as u64);
            }
            // Invariant 2: scheduled_ns is a multiple of period_ns
            for ev in &events {
                assert_eq!(
                    ev.scheduled_ns % period_ns,
                    0,
                    "scheduled_ns {} not multiple of period {}",
                    ev.scheduled_ns,
                    period_ns
                );
            }
            // Invariant 3: scheduled_ns is monotonically non-decreasing
            for w in events.windows(2) {
                assert!(
                    w[1].scheduled_ns >= w[0].scheduled_ns,
                    "scheduled_ns regressed: {} → {}",
                    w[0].scheduled_ns,
                    w[1].scheduled_ns
                );
            }
            shutdown.notify_waiters();
        });
    }

    #[test]
    fn observed_ns_never_precedes_scheduled_ns() {
        // tokio guarantees a tick fires AT or AFTER its deadline. In paused-time
        // mode, the runtime may add a small bookkeeping epsilon to observed_ns
        // (e.g., 1ms wheel granularity) — but observed_ns >= scheduled_ns must
        // always hold. This is the precondition for `jitter_ns()` to be
        // non-negative without saturation.
        let rt = paused_runtime();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
            let period_ns = g.period_ns();
            let mut rx = g.spawn(shutdown.clone());

            for _ in 0..50 {
                tokio::time::advance(StdDuration::from_nanos(period_ns)).await;
                let ev = rx.recv().await.expect("tick");
                assert!(
                    ev.observed_ns >= ev.scheduled_ns,
                    "observed_ns {} < scheduled_ns {} at epoch {}",
                    ev.observed_ns,
                    ev.scheduled_ns,
                    ev.epoch
                );
            }
            shutdown.notify_waiters();
        });
    }

    /// Real wall-clock drift soak — IGNORED by default. Runs ~5 s of real
    /// time at body cadence (≈40 ticks). Enforces drift target ratio per
    /// SCHUMANN_DRIFT_TARGET_PCT.
    ///
    /// Run with: `cargo test -p titan-schumann -- --ignored drift_soak`
    #[test]
    #[ignore]
    fn real_wall_clock_drift_soak_5s() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            let epoch_t0 = Instant::now();
            let shutdown = Arc::new(Notify::new());
            let g = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
            let period_ns = g.period_ns();
            let mut rx = g.spawn(shutdown.clone());

            let start = Instant::now();
            let mut events = Vec::new();
            while start.elapsed() < StdDuration::from_secs(5) {
                if let Some(ev) = rx.recv().await {
                    events.push(ev);
                } else {
                    break;
                }
            }

            let n_ticks = events.len() as u64;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            let expected_ns = n_ticks.saturating_mul(period_ns);
            let drift_ns = elapsed_ns.saturating_sub(expected_ns);
            let drift_pct = (drift_ns as f64) / (expected_ns as f64) * 100.0;
            assert!(
                drift_pct < SCHUMANN_DRIFT_TARGET_PCT,
                "drift {drift_pct}% > target {} % over 5s soak ({} ticks)",
                SCHUMANN_DRIFT_TARGET_PCT,
                n_ticks
            );
            shutdown.notify_waiters();
        });
    }
}
