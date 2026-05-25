//! heartbeat — Broker-side BUS_PING loop + 3-missed-pings disconnect.
//!
//! Per SPEC §3.1 D10 + §10.B 3-layer heartbeat hierarchy:
//! - `BUS_PING_INTERVAL_S = 5.0` — broker sends ping to every connected client.
//! - `BUS_PING_TIMEOUT_S = 15.0` — if `now - last_pong_ts > 15s`, broker drops
//!   the connection (3 missed pings = dead client).
//!
//! Per-client state lives on [`crate::subscriber::BrokerSubscriber::last_pong_ts`].
//! The recv loop updates this on every `BUS_PONG` reception.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::Mutex;
use tokio::time::interval;
use tracing::{debug, error};

use titan_core::constants::{BUS_PING_INTERVAL_S, BUS_PING_TIMEOUT_S};

use crate::message::encode_simple;
use crate::ring::{EnqueueOutcome, Envelope};
use crate::subscriber::SubscriberMap;
use titan_core::bus_specs::Priority;

/// Heartbeat tick result for one subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatResult {
    /// Subscriber is alive — ping enqueued.
    Pinged,
    /// Subscriber missed `> BUS_PING_TIMEOUT_S` of pongs — broker should
    /// purge.
    TimedOut,
    /// Subscriber already marked closed — broker will purge separately.
    Closed,
}

/// Returns `true` if the elapsed-since-last-pong exceeds the timeout
/// threshold. Used by the heartbeat loop to detect dead connections.
pub fn is_pong_timeout(last_pong_ts: SystemTime, now: SystemTime) -> bool {
    let timeout = Duration::from_secs_f64(BUS_PING_TIMEOUT_S);
    match now.duration_since(last_pong_ts) {
        Ok(elapsed) => elapsed > timeout,
        Err(_) => false, // clock went backwards — be lenient
    }
}

/// Encode a `BUS_PING` envelope. Used by the heartbeat loop to enqueue into
/// every subscriber's ring.
pub fn encode_ping_envelope() -> Envelope {
    let payload = encode_simple("BUS_PING", Some("broker"), None, None)
        .expect("encode BUS_PING never fails for this static input");
    Envelope {
        msg_type: "BUS_PING".into(),
        src: "broker".into(),
        priority: Priority::P0, // BUS_PING is P0 per SPEC §8.2
        payload,
    }
}

/// Boot-grace window: heartbeat-timeout enforcement is SUSPENDED for the
/// first `BOOT_GRACE_S` seconds after broker start.
///
/// **Why** (BUG-AGNO-SILENT-HANG follow-up, 2026-05-25): under heavy VPS
/// load (observed: load avg 17/4 cores during fleet restart), workers
/// can't always send BUS_PONG within the 15s `BUS_PING_TIMEOUT_S` ceiling
/// — scheduling latency alone can exceed it during the boot storm where
/// 30+ workers attach simultaneously. Pre-grace, this caused a cascading
/// close→reconnect cycle (39 closes in 30s observed) that prevented
/// agno_worker chat from stabilizing.
///
/// 120s is sized to cover:
///   - the worst-observed boot cascade (~60s of churn)
///   - 2x margin for slower VPS conditions
///   - well within human-perceptible liveness expectations
///
/// After grace, normal 15s timeout enforcement resumes. Steady-state
/// behavior unchanged — a TRULY dead worker still gets force-closed
/// within 15s of going silent.
///
/// SPEC error-cascade discipline (Maker 2026-05-25): grace expiry +
/// resumed enforcement is logged at INFO so operators see the
/// transition.
const BOOT_GRACE_S: f64 = 120.0;

/// Run the heartbeat loop forever. Caller wraps in `tokio::spawn` + holds a
/// shutdown signal externally to stop it.
///
/// Per `BUS_PING_INTERVAL_S=5.0` cadence:
/// 1. Walk the subscriber map.
/// 2. For each non-closed subscriber:
///    - If `last_pong_ts` exceeds `BUS_PING_TIMEOUT_S` AND boot-grace has
///      elapsed → mark closed via `signal_close()`; recv_loop + send_loop
///      exit promptly; connection_handler purges.
///    - Otherwise → enqueue a `BUS_PING` envelope into their ring.
pub async fn run_heartbeat_loop(
    subs: Arc<Mutex<SubscriberMap>>,
    notify_per_sub: Arc<Mutex<std::collections::HashMap<String, Arc<tokio::sync::Notify>>>>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    let mut interval = interval(Duration::from_secs_f64(BUS_PING_INTERVAL_S));
    interval.tick().await; // first tick fires immediately; consume it

    let broker_start = SystemTime::now();
    let mut boot_grace_logged = false;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let in_boot_grace = SystemTime::now()
                    .duration_since(broker_start)
                    .map(|d| d.as_secs_f64() < BOOT_GRACE_S)
                    .unwrap_or(false);
                if !in_boot_grace && !boot_grace_logged {
                    // SPEC error-cascade discipline: surface the
                    // transition so operators see when full enforcement
                    // resumed (helps correlate post-grace timeouts to
                    // true dead workers vs boot-storm artifacts).
                    tracing::info!(
                        grace_s = BOOT_GRACE_S,
                        timeout_s = BUS_PING_TIMEOUT_S,
                        "heartbeat boot-grace window expired; \
                         full timeout enforcement now active"
                    );
                    boot_grace_logged = true;
                }
                heartbeat_tick(&subs, &notify_per_sub, in_boot_grace).await;
            }
            _ = shutdown.notified() => {
                debug!("heartbeat loop: shutdown signal received");
                return;
            }
        }
    }
}

/// One iteration of the heartbeat loop. Exposed for unit testing.
///
/// `in_boot_grace=true` suspends timeout enforcement (still sends
/// pings; just doesn't close on timeout). See [`BOOT_GRACE_S`] docstring.
pub async fn heartbeat_tick(
    subs: &Arc<Mutex<SubscriberMap>>,
    notify_per_sub: &Arc<Mutex<std::collections::HashMap<String, Arc<tokio::sync::Notify>>>>,
    in_boot_grace: bool,
) {
    let now = SystemTime::now();
    let names: Vec<String> = subs.lock().await.keys().cloned().collect();

    for name in names {
        let result = {
            let mut subs_guard = subs.lock().await;
            let sub = match subs_guard.get_mut(&name) {
                Some(s) => s,
                None => continue, // gone since we listed
            };
            if sub.closed {
                HeartbeatResult::Closed
            } else if is_pong_timeout(sub.last_pong_ts, now) && !in_boot_grace {
                // Promoted WARN→ERROR with structured fields per SPEC
                // error-cascade discipline (Maker 2026-05-25). This is
                // ONE OF TWO paths that previously produced silent-zombie
                // subscribers (BUG-AGNO-SILENT-HANG):
                //   - Pre-fix the WARN was logged with the MAP KEY only
                //     (e.g. "anon-44") so operators couldn't tell WHICH
                //     worker was disconnected. The `sub_name` field now
                //     surfaces the logical worker name (e.g. "agno_worker")
                //     so kernel logs become diagnostically complete.
                //   - Pre-fix `sub.closed = true` left recv_loop blocked
                //     in `read_half.read_exact` forever (half-open
                //     connection), the subscriber map never purged, and
                //     all fanout to `dst=<sub.name>` silent-skipped at
                //     broker.rs:663 → permanent silent zombie.
                //
                // `signal_close()` fixes BOTH: structured ERROR log
                // AND fires the per-sub notify, waking recv_loop's
                // tokio::select! so it exits, allowing the
                // connection_handler to purge the map entry. The
                // worker's bus_socket recv_loop sees EOF → triggers
                // reconnect → fresh subscription → routing restored.
                error!(
                    name = %name,
                    sub_name = %sub.name,
                    timeout_s = BUS_PING_TIMEOUT_S,
                    reason = "heartbeat_pong_timeout",
                    "ERROR: subscriber heartbeat timeout — closing connection \
                     (BUG-AGNO-SILENT-HANG defense; worker should reconnect \
                     within seconds and re-subscribe)"
                );
                sub.signal_close();
                HeartbeatResult::TimedOut
            } else {
                let _ = sub.publish(encode_ping_envelope());
                HeartbeatResult::Pinged
            }
        };

        // Wake the send task so it picks up the BUS_PING
        if matches!(result, HeartbeatResult::Pinged) {
            if let Some(notify) = notify_per_sub.lock().await.get(&name) {
                notify.notify_one();
            }
        }
    }
}

// Suppress unused-import warning in non-test builds for `EnqueueOutcome`.
#[allow(dead_code)]
const _: Option<EnqueueOutcome> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscriber::BrokerSubscriber;
    use std::collections::HashMap;
    use tokio::sync::Notify;

    #[test]
    fn pong_timeout_threshold_uses_spec_constant() {
        let now = SystemTime::now();
        // Exactly at threshold: NOT timed out
        let last_pong = now - Duration::from_secs_f64(BUS_PING_TIMEOUT_S - 0.5);
        assert!(!is_pong_timeout(last_pong, now));

        // Past threshold
        let last_pong = now - Duration::from_secs_f64(BUS_PING_TIMEOUT_S + 0.5);
        assert!(is_pong_timeout(last_pong, now));
    }

    #[test]
    fn ping_envelope_has_correct_priority_and_type() {
        let env = encode_ping_envelope();
        assert_eq!(env.msg_type, "BUS_PING");
        assert_eq!(env.src, "broker");
        assert_eq!(env.priority, Priority::P0);
        // payload is non-empty msgpack
        assert!(!env.payload.is_empty());
    }

    #[tokio::test]
    async fn tick_pings_alive_sub() {
        let mut map = HashMap::new();
        map.insert("alive".to_string(), BrokerSubscriber::new("alive"));
        let subs = Arc::new(Mutex::new(map));
        let notify_map = Arc::new(Mutex::new(HashMap::new()));
        let notify = Arc::new(Notify::new());
        notify_map
            .lock()
            .await
            .insert("alive".to_string(), notify.clone());

        heartbeat_tick(&subs, &notify_map, false).await;

        // Sub should have one BUS_PING enqueued
        let s = subs.lock().await;
        let alive = s.get("alive").unwrap();
        assert_eq!(alive.ring.len(), 1);
    }

    #[tokio::test]
    async fn tick_marks_dead_sub_closed() {
        let mut sub = BrokerSubscriber::new("dead");
        sub.last_pong_ts = SystemTime::now() - Duration::from_secs_f64(BUS_PING_TIMEOUT_S + 1.0);
        let mut map = HashMap::new();
        map.insert("dead".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));
        let notify_map = Arc::new(Mutex::new(HashMap::new()));

        heartbeat_tick(&subs, &notify_map, false).await;

        let s = subs.lock().await;
        assert!(s.get("dead").unwrap().closed);
    }

    #[tokio::test]
    async fn tick_skips_already_closed_sub() {
        let mut sub = BrokerSubscriber::new("closed");
        sub.closed = true;
        let mut map = HashMap::new();
        map.insert("closed".to_string(), sub);
        let subs = Arc::new(Mutex::new(map));
        let notify_map = Arc::new(Mutex::new(HashMap::new()));

        heartbeat_tick(&subs, &notify_map, false).await;

        // No ping enqueued; sub still closed
        let s = subs.lock().await;
        let c = s.get("closed").unwrap();
        assert!(c.closed);
        assert_eq!(c.ring.len(), 0);
    }
}
