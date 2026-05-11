//! `PublishThrottle` — time-based gate for outer-trinity bus publishes.
//!
//! Outer daemons (post-A.S8 D2 cadence migration) tick at Schumann frequency
//! (7.83 / 23.49 / 70.47 Hz for body / mind / spirit) but throttle bus
//! publishes (`STATE` events + `OUTER_*_FILTER_DOWN`) to a slower interval
//! (45s / 15s / 5s respectively) so consumers see manageable update rates
//! while the daemon's slot writes remain content-hash gated at full Schumann
//! cadence.
//!
//! Body-slowest G13 invariant at the bus-publish layer:
//!   `OUTER_BODY_BUS_PUBLISH_INTERVAL_S` (45) > `OUTER_MIND_BUS_PUBLISH_INTERVAL_S` (15) >
//!   `OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S` (5)
//!
//! Per SPEC §13 G13 LOCKED + post-A.S8 D2 (rFP_phase_c_definitive_runtime_closure §4.2).

use std::time::{Duration, Instant};

/// Time-based throttle for outer-trinity bus publish rate-limiting.
///
/// First call to [`PublishThrottle::should_publish`] returns `true` (immediate
/// initial publish at startup). Subsequent calls return `true` only after
/// `interval` has elapsed since the last `true` return.
#[derive(Debug)]
pub struct PublishThrottle {
    interval: Duration,
    last_publish: Option<Instant>,
}

impl PublishThrottle {
    /// Create a throttle with a fixed interval (in seconds, from SPEC TOML
    /// `OUTER_*_BUS_PUBLISH_INTERVAL_S` constant).
    pub fn new(interval_s: f64) -> Self {
        debug_assert!(
            interval_s > 0.0,
            "PublishThrottle interval must be positive, got {interval_s}"
        );
        Self {
            interval: Duration::from_secs_f64(interval_s),
            last_publish: None,
        }
    }

    /// Returns `true` if a publish should fire NOW. Consumes the slot — calling
    /// twice in a row without `interval` elapsing between calls will return
    /// `true, false`.
    pub fn should_publish(&mut self) -> bool {
        let now = Instant::now();
        match self.last_publish {
            None => {
                self.last_publish = Some(now);
                true
            }
            Some(prev) if now.duration_since(prev) >= self.interval => {
                self.last_publish = Some(now);
                true
            }
            _ => false,
        }
    }

    /// Configured interval (for diagnostics + tests).
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Time since last publish (`None` if never published).
    pub fn since_last(&self) -> Option<Duration> {
        self.last_publish.map(|t| t.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_returns_true() {
        let mut t = PublishThrottle::new(5.0);
        assert!(t.should_publish());
    }

    #[test]
    fn second_immediate_call_returns_false() {
        let mut t = PublishThrottle::new(5.0);
        assert!(t.should_publish());
        assert!(!t.should_publish());
    }

    #[test]
    fn returns_true_after_interval() {
        let mut t = PublishThrottle::new(0.001); // 1ms for fast test
        assert!(t.should_publish());
        std::thread::sleep(Duration::from_millis(2));
        assert!(t.should_publish());
    }

    #[test]
    fn interval_accessor_matches_constructor() {
        let t = PublishThrottle::new(45.0);
        assert_eq!(t.interval(), Duration::from_secs(45));
    }

    #[test]
    fn since_last_is_none_before_first_publish() {
        let t = PublishThrottle::new(5.0);
        assert!(t.since_last().is_none());
    }

    #[test]
    fn since_last_is_some_after_publish() {
        let mut t = PublishThrottle::new(5.0);
        t.should_publish();
        let elapsed = t.since_last().expect("populated after publish");
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn body_slowest_invariant_holds_for_canonical_intervals() {
        use titan_core::constants::{
            OUTER_BODY_BUS_PUBLISH_INTERVAL_S, OUTER_MIND_BUS_PUBLISH_INTERVAL_S,
            OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S,
        };
        // SPEC §13 G13 — body slowest at bus publish layer
        assert!(OUTER_BODY_BUS_PUBLISH_INTERVAL_S > OUTER_MIND_BUS_PUBLISH_INTERVAL_S);
        assert!(OUTER_MIND_BUS_PUBLISH_INTERVAL_S > OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S);
        // Locked values per SPEC TOML
        assert_eq!(OUTER_BODY_BUS_PUBLISH_INTERVAL_S, 45.0);
        assert_eq!(OUTER_MIND_BUS_PUBLISH_INTERVAL_S, 15.0);
        assert_eq!(OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S, 5.0);
    }
}
