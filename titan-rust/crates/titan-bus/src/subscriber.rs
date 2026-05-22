//! subscriber — Per-connection state held by the broker.
//!
//! Byte-identical port of Python `BrokerSubscriber` (bus_socket.py).
//! Each connected client (Rust daemon or Python module) gets one of these
//! held in the broker's `subscribers: HashMap<String, BrokerSubscriber>`.
//!
//! # State
//!
//! - `name` — module/worker name; used as bus `dst` field
//! - `ring` — bounded P0+main queue ([`crate::ring::BoundedRing`])
//! - `coalesce_index` — `(src, type)` → ring slot pointer for O(1)
//!   coalesce-replace lookups
//! - `subscribed_topics` — set of message types the client subscribed to
//! - `last_pong_ts` — heartbeat tracking; broker drops connections after
//!   3 missed pings
//! - `drop_count_60s` / `recv_count_60s` — sliding-window counters for
//!   `BUS_SLOW_CONSUMER` drop-rate detection (threshold
//!   `BUS_SLOW_CONSUMER_DROP_RATE_RATIO=0.05`)
//! - `closed` — flag; broker purges closed subscribers from the map

use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};
use titan_core::bus_specs::Priority;
use titan_core::constants::BUS_SLOW_CONSUMER_DROP_RATE_RATIO;

use crate::ring::{BoundedRing, EnqueueOutcome, Envelope, RingError};

/// Coalesce-key tuple — `(src, msg_type)` pair.
///
/// Used by [`BrokerSubscriber::publish`] to look up an existing P1 message
/// in the ring with the same key + overwrite in place (instead of consuming
/// a new slot). Per SPEC §8.5 — applies to BODY/MIND/SPIRIT_STATE +
/// OUTER_OBSERVATION.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoalesceKey {
    /// Source identifier (`"inner-body"`, `"outer-spirit"`, etc.).
    pub src: String,
    /// Canonical message type.
    pub msg_type: String,
}

impl CoalesceKey {
    /// Build a coalesce key from the envelope.
    pub fn of(envelope: &Envelope) -> Self {
        Self {
            src: envelope.src.clone(),
            msg_type: envelope.msg_type.clone(),
        }
    }
}

/// All state the broker holds about one connected client.
///
/// **Concurrency**: in C-S2 chunk C2-2.b (server.rs), each subscriber will
/// be wrapped in `Arc<Mutex<BrokerSubscriber>>` and accessed by both the
/// broker's accept loop (publish path) + per-subscriber send task. The
/// struct itself stays sync to keep the data model simple — locking is at
/// the broker level.
pub struct BrokerSubscriber {
    /// Module/worker name (e.g. `"inner-body"`, `"reasoning"`); used as
    /// bus `dst` field for fanout. Primary name — first BUS_SUBSCRIBE
    /// from this connection sets it (replacing the initial "anon-N").
    pub name: String,

    /// Additional names this connection is subscribed under. Set via
    /// subsequent BUS_SUBSCRIBE frames AFTER the primary name is set;
    /// fanout matches `dst` against both `name` and `aliases`. Closes
    /// BUG-PHASE-C-BUS-FANOUT-MULTI-NAME-20260512: kernel-side proxy
    /// reply queues (output_verifier_proxy, agency_proxy, …) need
    /// RESPONSE messages routed to the parent's titan_HCL bus client
    /// without spawning a separate connection per proxy. The Python
    /// titan_HCL client now fires N BUS_SUBSCRIBE frames over its
    /// single connection — the broker treats subscribes 2..N as alias
    /// additions instead of replacing the primary name.
    pub aliases: HashSet<String>,

    /// Per-connection bounded queue (P0 reserve + main region).
    pub ring: BoundedRing,

    /// (src, type) → presence flag for O(1) coalesce-replace decisions.
    /// Entry exists iff a matching P1 message is currently in the ring.
    /// Removed on `pop_for_send()` consumption.
    pub coalesce_index: HashSet<CoalesceKey>,

    /// Topics this client subscribed to (`BUS_SUBSCRIBE` payload).
    pub subscribed_topics: HashSet<String>,

    /// D-SPEC-42 (SPEC v1.4.0, 2026-05-12) — subscriber intent flag.
    ///
    /// When `true`, this subscriber receives ONLY targeted `dst=<name>`
    /// (or `dst=<alias>`) messages — never `dst="all"` broadcasts.
    /// Broker `fanout` silently skips reply_only subscribers from the
    /// broadcast fan-out (no enqueue, no warn, no drop counter — they
    /// are not in the broadcast contract by design).
    ///
    /// Mirrors Python `BusSocketServer.BrokerSubscriber.reply_only`.
    /// Set from the BUS_SUBSCRIBE payload's `reply_only` field by
    /// `decode_bus_subscribe_payload`. Connection-level property:
    /// last BUS_SUBSCRIBE value sent on a multi-name subscribe wins.
    pub reply_only: bool,

    /// Last `BUS_PONG` reception time. Broker checks against
    /// `BUS_PING_TIMEOUT_S` to detect dead connections.
    pub last_pong_ts: SystemTime,

    /// Drop count in the current 60s window.
    pub drop_count_60s: u32,

    /// Receive (publish-attempt) count in the current 60s window.
    pub recv_count_60s: u32,

    /// Last `BUS_SLOW_CONSUMER` warn emission (rate-limited per
    /// `BUS_SLOW_CONSUMER_WARN_INTERVAL_S=60`).
    pub last_slow_consumer_warn_ts: Option<SystemTime>,

    /// Sliding-window reset boundary; reset every 60s.
    pub last_window_reset_ts: SystemTime,

    /// Connection closed (broker purges on next sweep).
    pub closed: bool,
}

impl BrokerSubscriber {
    /// New subscriber with default ring sizes.
    pub fn new(name: impl Into<String>) -> Self {
        let now = SystemTime::now();
        Self {
            name: name.into(),
            aliases: HashSet::new(),
            ring: BoundedRing::with_defaults(),
            coalesce_index: HashSet::new(),
            subscribed_topics: HashSet::new(),
            reply_only: false,
            last_pong_ts: now,
            drop_count_60s: 0,
            recv_count_60s: 0,
            last_slow_consumer_warn_ts: None,
            last_window_reset_ts: now,
            closed: false,
        }
    }

    /// New subscriber with custom ring sizes (used by tests).
    pub fn with_ring(name: impl Into<String>, ring: BoundedRing) -> Self {
        let now = SystemTime::now();
        Self {
            name: name.into(),
            aliases: HashSet::new(),
            ring,
            coalesce_index: HashSet::new(),
            subscribed_topics: HashSet::new(),
            reply_only: false,
            last_pong_ts: now,
            drop_count_60s: 0,
            recv_count_60s: 0,
            last_slow_consumer_warn_ts: None,
            last_window_reset_ts: now,
            closed: false,
        }
    }

    /// Enqueue a message, applying P1 coalesce-by-(src,type) when applicable.
    ///
    /// Workflow:
    /// 1. Increment `recv_count_60s`
    /// 2. If priority is P1 AND coalesce_index has the key →
    ///    `ring.coalesce_replace()` (no new slot consumed)
    /// 3. Otherwise enqueue normally; on Accepted, insert key into
    ///    coalesce_index for future P1 matches; on Eviction, increment
    ///    `drop_count_60s`; on Rejected, increment `drop_count_60s`
    pub fn publish(&mut self, msg: Envelope) -> Result<EnqueueOutcome, RingError> {
        self.recv_count_60s = self.recv_count_60s.saturating_add(1);

        // Coalesce-by-(src,type) only for P1 messages with a coalesce-by spec
        // (caller passes priority directly — caller looked it up in SPECS).
        let key = CoalesceKey::of(&msg);
        if msg.priority == Priority::P1
            && self.coalesce_index.contains(&key)
            && self.ring.coalesce_replace(&msg)
        {
            return Ok(EnqueueOutcome::Accepted);
        }

        let outcome = self.ring.enqueue(msg)?;
        match outcome {
            EnqueueOutcome::Accepted => {
                // Track for future coalesce on P1 messages
                self.coalesce_index.insert(key);
            }
            EnqueueOutcome::AcceptedWithEviction => {
                self.coalesce_index.insert(key);
                self.drop_count_60s = self.drop_count_60s.saturating_add(1);
            }
            EnqueueOutcome::Rejected => {
                self.drop_count_60s = self.drop_count_60s.saturating_add(1);
            }
        }
        Ok(outcome)
    }

    /// Pop up to `max_msgs` from the ring + maintain `coalesce_index`.
    ///
    /// Each popped message's `(src, type)` key is removed from the index
    /// (a fresh publish on the same key starts a new ring slot).
    pub fn pop_for_send(&mut self, max_msgs: usize) -> Vec<Envelope> {
        let popped = self.ring.pop_for_send(max_msgs);
        for env in &popped {
            self.coalesce_index.remove(&CoalesceKey::of(env));
        }
        popped
    }

    /// Record a `BUS_PONG` reception → reset heartbeat timer.
    pub fn note_pong(&mut self) {
        self.last_pong_ts = SystemTime::now();
    }

    /// Returns the current 60s drop-rate ratio. Used by slow-consumer
    /// detection (threshold `BUS_SLOW_CONSUMER_DROP_RATE_RATIO=0.05`).
    pub fn drop_rate_60s(&self) -> f64 {
        if self.recv_count_60s == 0 {
            return 0.0;
        }
        self.drop_count_60s as f64 / self.recv_count_60s as f64
    }

    /// Returns `true` if drop-rate ratio exceeds the SPEC threshold
    /// (`BUS_SLOW_CONSUMER_DROP_RATE_RATIO=0.05`). Caller emits
    /// `BUS_SLOW_CONSUMER` (rate-limited per 60s).
    pub fn is_slow_consumer(&self) -> bool {
        self.drop_rate_60s() > BUS_SLOW_CONSUMER_DROP_RATE_RATIO
    }

    /// Reset 60s sliding window if elapsed > 60s. Called by broker's
    /// per-tick maintenance loop.
    pub fn maybe_reset_window(&mut self, now: SystemTime, window: Duration) {
        if let Ok(elapsed) = now.duration_since(self.last_window_reset_ts) {
            if elapsed >= window {
                self.drop_count_60s = 0;
                self.recv_count_60s = 0;
                self.last_window_reset_ts = now;
            }
        }
    }
}

/// Internal map from subscriber name → state. Used by broker's accept loop.
pub type SubscriberMap = HashMap<String, BrokerSubscriber>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::BoundedRing;

    fn env(msg_type: &str, src: &str, priority: Priority, payload: &[u8]) -> Envelope {
        Envelope {
            msg_type: msg_type.to_string(),
            src: src.to_string(),
            priority,
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn publish_p1_first_time_takes_slot() {
        let mut sub = BrokerSubscriber::new("test");
        let outcome = sub
            .publish(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        assert_eq!(outcome, EnqueueOutcome::Accepted);
        assert_eq!(sub.ring.len(), 1);
        assert_eq!(sub.recv_count_60s, 1);
    }

    #[test]
    fn publish_p1_same_src_type_coalesces_in_place() {
        let mut sub = BrokerSubscriber::new("test");
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v2"))
            .unwrap();
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v3"))
            .unwrap();
        // 3 publishes but only 1 ring slot consumed
        assert_eq!(sub.ring.len(), 1);
        assert_eq!(sub.recv_count_60s, 3);
        // Latest payload is v3
        let popped = sub.pop_for_send(10);
        assert_eq!(popped[0].payload, b"v3");
    }

    #[test]
    fn publish_p1_different_src_type_no_coalesce() {
        let mut sub = BrokerSubscriber::new("test");
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        sub.publish(env("MIND_STATE", "inner-mind", Priority::P1, b"v1"))
            .unwrap();
        sub.publish(env("BODY_STATE", "outer-body", Priority::P1, b"v1"))
            .unwrap();
        // 3 distinct (src, type) → 3 ring slots
        assert_eq!(sub.ring.len(), 3);
    }

    #[test]
    fn publish_p2_no_coalesce_even_same_key() {
        let mut sub = BrokerSubscriber::new("test");
        sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2, b"v1"))
            .unwrap();
        sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2, b"v2"))
            .unwrap();
        // P2 does NOT coalesce — both slots exist
        assert_eq!(sub.ring.len(), 2);
    }

    #[test]
    fn pop_clears_coalesce_index_entry() {
        let mut sub = BrokerSubscriber::new("test");
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        assert!(sub.coalesce_index.contains(&CoalesceKey {
            src: "inner-body".into(),
            msg_type: "BODY_STATE".into()
        }));
        sub.pop_for_send(10);
        assert!(!sub.coalesce_index.contains(&CoalesceKey {
            src: "inner-body".into(),
            msg_type: "BODY_STATE".into()
        }));
        // After pop, a fresh publish takes a new slot
        sub.publish(env("BODY_STATE", "inner-body", Priority::P1, b"v2"))
            .unwrap();
        assert_eq!(sub.ring.len(), 1);
    }

    #[test]
    fn p2_drop_increments_drop_count() {
        let mut sub = BrokerSubscriber::with_ring("test", BoundedRing::new(5, 2).unwrap());
        // main_capacity = 3
        for i in 0..3 {
            sub.publish(env(
                "REASONING_CHAIN",
                "reasoning",
                Priority::P2,
                &[i as u8],
            ))
            .unwrap();
        }
        assert_eq!(sub.drop_count_60s, 0);
        // Next P2 evicts oldest → drop_count++
        sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2, &[3]))
            .unwrap();
        assert_eq!(sub.drop_count_60s, 1);
    }

    #[test]
    fn p3_reject_increments_drop_count() {
        let mut sub = BrokerSubscriber::with_ring("test", BoundedRing::new(5, 2).unwrap());
        for _ in 0..3 {
            sub.publish(env("OBSERVATORY_EVENT", "obs", Priority::P3, b"x"))
                .unwrap();
        }
        // Main full now. P3 rejects.
        sub.publish(env("OBSERVATORY_EVENT", "obs", Priority::P3, b"new"))
            .unwrap();
        assert_eq!(sub.drop_count_60s, 1);
    }

    #[test]
    fn drop_rate_zero_with_no_traffic() {
        let sub = BrokerSubscriber::new("test");
        assert_eq!(sub.drop_rate_60s(), 0.0);
        assert!(!sub.is_slow_consumer());
    }

    #[test]
    fn drop_rate_above_threshold_marks_slow() {
        let mut sub = BrokerSubscriber::with_ring("test", BoundedRing::new(5, 2).unwrap());
        // 100 publishes, 50 evictions → 50% drop rate >> 5% threshold
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2, b"x"))
                .unwrap();
        }
        // Exactly 100 - main_capacity(3) = 97 evictions
        assert_eq!(sub.drop_count_60s, 97);
        assert_eq!(sub.recv_count_60s, 100);
        assert!(sub.drop_rate_60s() > 0.05);
        assert!(sub.is_slow_consumer());
    }

    #[test]
    fn drop_rate_below_threshold_not_slow() {
        let mut sub = BrokerSubscriber::with_ring("test", BoundedRing::new(105, 2).unwrap());
        // 100 publishes into capacity 103 → 0 drops
        for _ in 0..100 {
            sub.publish(env("REASONING_CHAIN", "reasoning", Priority::P2, b"x"))
                .unwrap();
        }
        assert_eq!(sub.drop_count_60s, 0);
        assert_eq!(sub.drop_rate_60s(), 0.0);
        assert!(!sub.is_slow_consumer());
    }

    #[test]
    fn reset_window_zeros_counters() {
        let mut sub = BrokerSubscriber::new("test");
        sub.recv_count_60s = 100;
        sub.drop_count_60s = 10;
        let later = SystemTime::now() + Duration::from_secs(120);
        sub.maybe_reset_window(later, Duration::from_secs(60));
        assert_eq!(sub.recv_count_60s, 0);
        assert_eq!(sub.drop_count_60s, 0);
    }

    #[test]
    fn reset_window_no_op_within_window() {
        let mut sub = BrokerSubscriber::new("test");
        sub.recv_count_60s = 100;
        sub.drop_count_60s = 10;
        let earlier = sub.last_window_reset_ts + Duration::from_secs(30);
        sub.maybe_reset_window(earlier, Duration::from_secs(60));
        // Counters NOT reset
        assert_eq!(sub.recv_count_60s, 100);
        assert_eq!(sub.drop_count_60s, 10);
    }

    #[test]
    fn note_pong_updates_timestamp() {
        let mut sub = BrokerSubscriber::new("test");
        let initial = sub.last_pong_ts;
        std::thread::sleep(Duration::from_millis(10));
        sub.note_pong();
        assert!(sub.last_pong_ts > initial);
    }
}
