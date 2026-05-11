//! backoff — Jittered exponential restart backoff per SPEC §11.B step 5.
//!
//! Sequence: `100 ms → 200 → 400 → 800 → 1600 → 2000` (capped) with ±25%
//! jitter applied per attempt. Jitter prevents thundering-herd when many
//! children crash simultaneously (e.g. shared dependency failure).
//!
//! Constants from `titan-core::constants` (auto-generated from SPEC TOML).

use std::time::Duration;

use rand::Rng;

use crate::constants::{
    SUPERVISION_RESTART_BACKOFF_INITIAL_MS, SUPERVISION_RESTART_BACKOFF_MAX_S,
    SUPERVISION_RESTART_JITTER_PCT,
};

/// Compute the un-jittered base backoff for a given restart count.
///
/// `100ms × 2^(restart_count - 1)`, capped at
/// `SUPERVISION_RESTART_BACKOFF_MAX_S`. Used by tests + as the base for
/// [`compute_backoff`].
pub fn backoff_base_unjittered(restart_count: u32) -> Duration {
    let exp_ms = SUPERVISION_RESTART_BACKOFF_INITIAL_MS
        .saturating_mul(2u64.pow(restart_count.saturating_sub(1).min(31)));
    let max_ms = (SUPERVISION_RESTART_BACKOFF_MAX_S * 1000.0) as u64;
    Duration::from_millis(exp_ms.min(max_ms))
}

/// Compute the next backoff with ±`SUPERVISION_RESTART_JITTER_PCT` jitter.
///
/// Returns a duration in the range
/// `[base × (1 - jitter_pct/100), base × (1 + jitter_pct/100)]`.
///
/// Jitter is applied multiplicatively + symmetrically. RNG is `thread_rng()`
/// — for deterministic tests, use [`compute_backoff_with_rng`].
pub fn compute_backoff(restart_count: u32) -> Duration {
    let mut rng = rand::thread_rng();
    compute_backoff_with_rng(restart_count, &mut rng)
}

/// Compute backoff with an injected RNG (used by tests for determinism).
pub fn compute_backoff_with_rng(restart_count: u32, rng: &mut impl Rng) -> Duration {
    let base = backoff_base_unjittered(restart_count);
    let jitter_pct = SUPERVISION_RESTART_JITTER_PCT as f64 / 100.0;
    // Generate jitter factor in [-jitter_pct, +jitter_pct]
    let jitter: f64 = rng.gen_range(-jitter_pct..=jitter_pct);
    let factor = 1.0 + jitter;
    Duration::from_secs_f64(base.as_secs_f64() * factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn backoff_initial_is_100ms() {
        assert_eq!(backoff_base_unjittered(1), Duration::from_millis(100));
    }

    #[test]
    fn backoff_doubles_each_step() {
        assert_eq!(backoff_base_unjittered(1), Duration::from_millis(100));
        assert_eq!(backoff_base_unjittered(2), Duration::from_millis(200));
        assert_eq!(backoff_base_unjittered(3), Duration::from_millis(400));
        assert_eq!(backoff_base_unjittered(4), Duration::from_millis(800));
        assert_eq!(backoff_base_unjittered(5), Duration::from_millis(1600));
    }

    #[test]
    fn backoff_caps_at_max() {
        // restart_count=6: 100×32 = 3200ms, but max=2000ms → capped
        assert_eq!(backoff_base_unjittered(6), Duration::from_millis(2000));
        // 100 → still 2000ms cap
        assert_eq!(backoff_base_unjittered(100), Duration::from_millis(2000));
    }

    #[test]
    fn backoff_handles_zero_restart_count() {
        // restart_count=0 → saturating_sub(1) = 0 → 100×1 = 100ms (sane fallback)
        let d = backoff_base_unjittered(0);
        assert_eq!(d, Duration::from_millis(100));
    }

    #[test]
    fn jittered_backoff_within_25_percent_band() {
        // Run many trials; verify all fall in [75ms, 125ms] for restart_count=1
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let d = compute_backoff_with_rng(1, &mut rng);
            let ms = d.as_secs_f64() * 1000.0;
            assert!(
                (75.0..=125.0).contains(&ms),
                "backoff {ms}ms outside [75, 125] band for restart_count=1"
            );
        }
    }

    #[test]
    fn jittered_backoff_distribution_spreads() {
        // With 1000 samples, we should see values across the full band
        // (proves jitter is actually random, not a constant).
        let mut rng = StdRng::seed_from_u64(7);
        let mut min_ms = f64::MAX;
        let mut max_ms = f64::MIN;
        for _ in 0..1000 {
            let d = compute_backoff_with_rng(2, &mut rng);
            let ms = d.as_secs_f64() * 1000.0;
            min_ms = min_ms.min(ms);
            max_ms = max_ms.max(ms);
        }
        // base=200ms; jitter ±25% → [150, 250]
        assert!(min_ms < 175.0, "min jitter sample {min_ms}ms too high");
        assert!(max_ms > 225.0, "max jitter sample {max_ms}ms too low");
    }

    #[test]
    fn jittered_backoff_at_cap_still_jittered() {
        // restart_count=100 → base=2000ms (cap); jitter ±25% → [1500, 2500]
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..50 {
            let d = compute_backoff_with_rng(100, &mut rng);
            let ms = d.as_secs_f64() * 1000.0;
            assert!(
                (1500.0..=2500.0).contains(&ms),
                "capped backoff {ms}ms outside [1500, 2500]"
            );
        }
    }

    #[test]
    fn compute_backoff_thread_rng_in_band() {
        // No determinism needed — just verify the public API doesn't panic
        // and values fall in the expected band.
        for _ in 0..10 {
            let d = compute_backoff(3);
            let ms = d.as_secs_f64() * 1000.0;
            // base=400ms; ±25% → [300, 500]
            assert!((300.0..=500.0).contains(&ms));
        }
    }
}
