//! slow_consumer — Per-subscriber drop-rate detection + rate-limited
//! `BUS_SLOW_CONSUMER` emission.
//!
//! Per SPEC §3.1 D22 + §8.2:
//! - Threshold: `BUS_SLOW_CONSUMER_DROP_RATE_RATIO = 0.05` (5%)
//! - Warn rate-limit: `BUS_SLOW_CONSUMER_WARN_INTERVAL_S = 60.0`
//! - Window reset: `60s` (counters cleared each minute)
//!
//! The drop-rate computation lives on
//! [`crate::subscriber::BrokerSubscriber::is_slow_consumer`]. This module
//! provides the periodic sweep + emission loop.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::Mutex;
use tokio::time::interval;
use tracing::{debug, warn};

use titan_core::constants::BUS_SLOW_CONSUMER_WARN_INTERVAL_S;

use crate::subscriber::SubscriberMap;

/// Standard 60s sliding window for drop-rate accounting.
pub const DROP_RATE_WINDOW: Duration = Duration::from_secs(60);

/// One iteration of the slow-consumer sweep. Exposed for unit testing.
///
/// Returns the names of subscribers that crossed the threshold + had not
/// been warned within `BUS_SLOW_CONSUMER_WARN_INTERVAL_S`. Caller emits
/// `BUS_SLOW_CONSUMER` to all subscribers (or just the broker's
/// observability channel).
pub async fn slow_consumer_sweep(subs: &Arc<Mutex<SubscriberMap>>) -> Vec<String> {
    let now = SystemTime::now();
    let warn_interval = Duration::from_secs_f64(BUS_SLOW_CONSUMER_WARN_INTERVAL_S);
    let mut to_warn = Vec::new();
    let mut subs_guard = subs.lock().await;
    for (name, sub) in subs_guard.iter_mut() {
        // Reset window if elapsed
        sub.maybe_reset_window(now, DROP_RATE_WINDOW);
        if !sub.is_slow_consumer() {
            continue;
        }
        let should_warn = match sub.last_slow_consumer_warn_ts {
            None => true,
            Some(prev) => now
                .duration_since(prev)
                .map(|elapsed| elapsed >= warn_interval)
                .unwrap_or(true),
        };
        if should_warn {
            sub.last_slow_consumer_warn_ts = Some(now);
            to_warn.push(name.clone());
            warn!(
                name = %name,
                drop_rate = sub.drop_rate_60s(),
                drops = sub.drop_count_60s,
                recvs = sub.recv_count_60s,
                "slow consumer detected"
            );
        }
    }
    to_warn
}

/// Run the slow-consumer detection loop forever.
///
/// Cadence: every 10s walk the subscriber map; emit `BUS_SLOW_CONSUMER` for
/// any subscriber whose drop-rate crossed the threshold AND hasn't been
/// warned within the last `BUS_SLOW_CONSUMER_WARN_INTERVAL_S=60` seconds.
pub async fn run_slow_consumer_loop(
    subs: Arc<Mutex<SubscriberMap>>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    /// Sweep interval (faster than warn-rate-limit so we catch crossings).
    const SWEEP_INTERVAL: Duration = Duration::from_secs(10);
    let mut interval = interval(SWEEP_INTERVAL);
    interval.tick().await; // first immediate tick

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let warned = slow_consumer_sweep(&subs).await;
                if !warned.is_empty() {
                    debug!(count = warned.len(), "slow_consumer_loop: warned subs");
                }
            }
            _ = shutdown.notified() => {
                debug!("slow_consumer loop: shutdown signal received");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{BoundedRing, Envelope};
    use crate::subscriber::BrokerSubscriber;
    use std::collections::HashMap;
    use titan_core::bus_specs::Priority;

    fn env(msg_type: &str, src: &str, priority: Priority) -> Envelope {
        Envelope {
            msg_type: msg_type.into(),
            src: src.into(),
            priority,
            payload: vec![0u8; 8],
        }
    }

    #[tokio::test]
    async fn sweep_warns_slow_sub_first_time() {
        let mut sub = BrokerSubscriber::with_ring("slow", BoundedRing::new(5, 2).unwrap());
        // Force >5% drop rate
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2))
                .unwrap();
        }
        assert!(sub.is_slow_consumer());

        let mut map = HashMap::new();
        map.insert("slow".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));

        let warned = slow_consumer_sweep(&subs).await;
        assert_eq!(warned, vec!["slow".to_string()]);
    }

    #[tokio::test]
    async fn sweep_skips_recently_warned_sub() {
        let mut sub = BrokerSubscriber::with_ring("slow", BoundedRing::new(5, 2).unwrap());
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2))
                .unwrap();
        }
        // Mark as warned 1 second ago — within the 60s rate-limit
        sub.last_slow_consumer_warn_ts = Some(SystemTime::now() - Duration::from_secs(1));

        let mut map = HashMap::new();
        map.insert("slow".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));

        let warned = slow_consumer_sweep(&subs).await;
        assert!(warned.is_empty(), "should not re-warn within rate-limit");
    }

    #[tokio::test]
    async fn sweep_re_warns_after_rate_limit_elapses() {
        let mut sub = BrokerSubscriber::with_ring("slow", BoundedRing::new(5, 2).unwrap());
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2))
                .unwrap();
        }
        // Marked as warned long ago — past rate-limit
        sub.last_slow_consumer_warn_ts = Some(
            SystemTime::now() - Duration::from_secs_f64(BUS_SLOW_CONSUMER_WARN_INTERVAL_S + 5.0),
        );

        let mut map = HashMap::new();
        map.insert("slow".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));

        let warned = slow_consumer_sweep(&subs).await;
        assert_eq!(warned, vec!["slow".to_string()]);
    }

    #[tokio::test]
    async fn sweep_ignores_healthy_sub() {
        let mut sub = BrokerSubscriber::with_ring("healthy", BoundedRing::new(105, 2).unwrap());
        // No drops — capacity is sufficient
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2))
                .unwrap();
        }
        assert!(!sub.is_slow_consumer());

        let mut map = HashMap::new();
        map.insert("healthy".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));

        let warned = slow_consumer_sweep(&subs).await;
        assert!(warned.is_empty());
    }
}
