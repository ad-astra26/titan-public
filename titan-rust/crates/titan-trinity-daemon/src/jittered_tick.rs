//! jittered_tick — Variable-cadence ticker for outer trinity daemons.
//!
//! Phase C C-S6 — complement to the Schumann-rate inner ticker. Outer
//! daemons (titan-outer-{body,mind,spirit}-rs) are NOT Schumann-locked
//! per SPEC §18.1; they run at jittered seconds-scale cadences:
//!
//!   - outer-body  : 10.0 s base ± 20% jitter (per OUTER_BODY_TICK_BASE_S
//!     / OUTER_BODY_TICK_JITTER_PCT)
//!   - outer-mind  :  5.0 s base ± 20% jitter (OUTER_MIND_*)
//!   - outer-spirit: 30.0 s base ± 10% jitter (OUTER_SPIRIT_*)
//!
//! # Why jitter
//!
//! Hard-locked cadences across 3 daemons would all fire at the same
//! wall-clock instants, creating bursty load on the bus broker, the
//! Python sensor sidecars, and the kernel's persistence layer. Jitter
//! decorrelates the firings — same average cadence, smoother system load.
//!
//! # Implementation
//!
//! Single-task: each daemon's tick loop calls
//! [`JitteredTicker::next_tick`] in a loop until `KERNEL_SHUTDOWN_ANNOUNCE`.
//! Sleeps for `base × (1 + jitter_signed)` where `jitter_signed ∈
//! [-jitter_pct/100, +jitter_pct/100]`.
//!
//! # Determinism
//!
//! For parity tests, the ticker can be constructed with a deterministic
//! RNG seed via [`JitteredTicker::with_seed`]. Production code uses
//! [`JitteredTicker::new`] which seeds from system entropy.
//!
//! # SPEC ground truths
//!
//! - SPEC §18.1: outer cadences locked at 10/5/30 s ± 20/20/10 % jitter.
//! - PLAN §1.1 item 8: cadence + jitter constants live in
//!   `OUTER_*_TICK_BASE_S` + `OUTER_*_TICK_JITTER_PCT` per the generated
//!   `titan-core::constants`.
//! - SPEC §3 D24 / G15: NO local `Duration` magic; all values come
//!   from constants.

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// Per-tick observation reported by [`JitteredTicker::next_tick`].
#[derive(Debug, Clone, Copy)]
pub struct JitteredTickEvent {
    /// Monotonic tick number (1-based).
    pub n: u64,
    /// Wall-clock instant when the tick was OBSERVED.
    pub now: Instant,
    /// The randomized period used for THIS tick (= sleep duration).
    pub period: Duration,
    /// The signed jitter fraction applied (`actual / base − 1`).
    /// Range: `[-jitter_pct/100, +jitter_pct/100]`.
    pub jitter_signed: f32,
}

/// Outer-trinity daemon ticker with bounded-uniform jitter.
///
/// Each call to [`JitteredTicker::next_tick`] sleeps for
/// `base_period × (1 + uniform(-jitter, +jitter))` and returns a
/// [`JitteredTickEvent`] describing the actual period used.
pub struct JitteredTicker {
    base_period: Duration,
    jitter_pct: f32,
    rng: StdRng,
    n: u64,
}

impl JitteredTicker {
    /// Construct a ticker seeded from system entropy.
    ///
    /// `base_s` — base cadence (e.g. `OUTER_BODY_TICK_BASE_S` = 10.0)
    /// `jitter_pct` — max fractional jitter as percent (e.g. `20` for ±20%)
    pub fn new(base_s: f64, jitter_pct: u32) -> Self {
        Self::with_seed(base_s, jitter_pct, rand::random())
    }

    /// Construct a ticker with a deterministic seed (parity tests).
    pub fn with_seed(base_s: f64, jitter_pct: u32, seed: u64) -> Self {
        Self {
            base_period: Duration::from_secs_f64(base_s),
            jitter_pct: jitter_pct as f32,
            rng: StdRng::seed_from_u64(seed),
            n: 0,
        }
    }

    /// Configured base period.
    pub fn base_period(&self) -> Duration {
        self.base_period
    }

    /// Configured jitter percent (e.g. 20 for ±20%).
    pub fn jitter_pct(&self) -> f32 {
        self.jitter_pct
    }

    /// Total ticks observed so far.
    pub fn total_ticks(&self) -> u64 {
        self.n
    }

    /// Sample the NEXT period (without sleeping). Public for parity tests
    /// + offline analysis. Each call advances the internal RNG.
    pub fn sample_next_period(&mut self) -> Duration {
        let signed = self.sample_jitter_signed();
        scale_period(self.base_period, signed)
    }

    /// Sample the next signed jitter fraction. Uniform over
    /// `[-jitter_pct/100, +jitter_pct/100]`.
    pub fn sample_jitter_signed(&mut self) -> f32 {
        let max_frac = self.jitter_pct / 100.0;
        if max_frac <= 0.0 {
            return 0.0;
        }
        // Use the rng's underlying integer source for portable + fast
        // uniform sampling.
        let raw = self.rng.next_u32();
        // Map u32 → [0, 1)
        let unit = (raw as f64) / (u32::MAX as f64 + 1.0);
        // Map [0, 1) → [-max_frac, +max_frac]
        ((unit * 2.0 - 1.0) * max_frac as f64) as f32
    }

    /// Wait for the next tick. Returns a [`JitteredTickEvent`].
    ///
    /// Cancellation safety: built on `tokio::time::sleep` which is
    /// cancel-safe. If the future is dropped before completion, no tick
    /// is recorded; next call samples fresh jitter + sleeps fully.
    pub async fn next_tick(&mut self) -> JitteredTickEvent {
        let signed = self.sample_jitter_signed();
        let period = scale_period(self.base_period, signed);
        tokio::time::sleep(period).await;
        let now = Instant::now();
        self.n = self.n.saturating_add(1);
        JitteredTickEvent {
            n: self.n,
            now,
            period,
            jitter_signed: signed,
        }
    }
}

/// Scale a base period by a signed jitter fraction. Free function for
/// unit tests.
///
/// `scaled = base × (1 + signed)`. Clamped to `≥ 1ms` to avoid degenerate
/// hot loops if a future caller passes `signed ≤ -1`.
pub fn scale_period(base: Duration, signed: f32) -> Duration {
    let factor = (1.0_f32 + signed).max(0.001);
    let scaled_s = base.as_secs_f64() * factor as f64;
    Duration::from_secs_f64(scaled_s.max(0.001))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_period_zero_jitter_is_identity() {
        let base = Duration::from_secs_f64(10.0);
        let scaled = scale_period(base, 0.0);
        assert_eq!(scaled, base);
    }

    #[test]
    fn scale_period_positive_jitter_extends() {
        let base = Duration::from_secs_f64(10.0);
        let scaled = scale_period(base, 0.20); // +20%
                                               // 1e-6 tolerance: 10s × f32(1.2) → f64 round-trip has ~1e-7 error.
                                               // Jitter is for cadence decorrelation, not nanosecond precision.
        assert!((scaled.as_secs_f64() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn scale_period_negative_jitter_shortens() {
        let base = Duration::from_secs_f64(10.0);
        let scaled = scale_period(base, -0.20); // -20%
        assert!((scaled.as_secs_f64() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn scale_period_extreme_negative_clamps() {
        let base = Duration::from_secs_f64(10.0);
        // signed = -2 would give negative scaled; clamp to ≥ 1ms via factor floor.
        let scaled = scale_period(base, -2.0);
        assert!(scaled >= Duration::from_millis(1));
    }

    #[test]
    fn jitter_signed_within_bounds_for_outer_body() {
        // SPEC §18.1: outer-body uses 20% jitter
        let mut ticker = JitteredTicker::with_seed(10.0, 20, 0xC6FAB1E_u64);
        for _ in 0..1000 {
            let signed = ticker.sample_jitter_signed();
            assert!(
                signed.abs() <= 0.20 + 1e-6,
                "jitter {} out of [-0.20, +0.20]",
                signed,
            );
        }
    }

    #[test]
    fn jitter_signed_within_bounds_for_outer_spirit() {
        // SPEC §18.1: outer-spirit uses 10% jitter (tighter)
        let mut ticker = JitteredTicker::with_seed(30.0, 10, 0xC6FAB1E_u64);
        for _ in 0..1000 {
            let signed = ticker.sample_jitter_signed();
            assert!(
                signed.abs() <= 0.10 + 1e-6,
                "jitter {} out of [-0.10, +0.10]",
                signed,
            );
        }
    }

    #[test]
    fn deterministic_seed_reproduces_sequence() {
        let mut a = JitteredTicker::with_seed(10.0, 20, 42);
        let mut b = JitteredTicker::with_seed(10.0, 20, 42);
        for _ in 0..50 {
            assert_eq!(a.sample_jitter_signed(), b.sample_jitter_signed());
        }
    }

    #[test]
    fn distinct_seeds_produce_different_sequences() {
        let mut a = JitteredTicker::with_seed(10.0, 20, 1);
        let mut b = JitteredTicker::with_seed(10.0, 20, 2);
        let mut diffs = 0;
        for _ in 0..50 {
            if a.sample_jitter_signed() != b.sample_jitter_signed() {
                diffs += 1;
            }
        }
        // Two seeds should diverge on virtually every draw.
        assert!(diffs > 40, "expected >40 differing draws, got {}", diffs);
    }

    #[test]
    fn zero_jitter_pct_yields_constant_period() {
        let mut ticker = JitteredTicker::with_seed(10.0, 0, 7);
        for _ in 0..50 {
            let p = ticker.sample_next_period();
            assert!((p.as_secs_f64() - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn total_ticks_starts_at_zero() {
        let ticker = JitteredTicker::with_seed(10.0, 20, 0);
        assert_eq!(ticker.total_ticks(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn next_tick_advances_counter_and_wall_clock() {
        // Real sleep at 5ms base × 20% jitter — total test runtime ≤ ~15ms.
        // Tokio's `start_paused` would let us go even faster but requires
        // the `test-util` feature which isn't in the workspace tokio dep.
        let mut ticker = JitteredTicker::with_seed(0.005, 20, 0xC6FAB1E_u64);
        let ev1 = ticker.next_tick().await;
        let ev2 = ticker.next_tick().await;
        assert_eq!(ev1.n, 1);
        assert_eq!(ev2.n, 2);
        assert!(ev2.now >= ev1.now);
        // Each period should be within ±20% of base 5ms
        for ev in [ev1, ev2] {
            let pms = ev.period.as_secs_f64() * 1000.0;
            assert!(
                (4.0..=6.0).contains(&pms),
                "period {}ms out of ±20% of 5ms base",
                pms,
            );
        }
    }
}
