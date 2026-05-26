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
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use titan_core::bus_specs::Priority;
use titan_core::constants::BUS_SLOW_CONSUMER_DROP_RATE_RATIO;
use tokio::sync::{watch, Notify};

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
    ///
    /// **Forbidden direct assignment** — all callers MUST use
    /// [`BrokerSubscriber::signal_close`] (see method docstring for the
    /// BUG-AGNO-SILENT-HANG context — direct `closed = true` without
    /// firing the per-sub notify leaves recv_loop blocked in `socket.read()`
    /// indefinitely, producing the silent-zombie subscriber state observed
    /// 2026-05-25 fleet-wide).
    pub closed: bool,

    /// Per-subscriber **data-wake** notify (SPEC §8.0.quat data-wake primitive).
    /// Edge-triggered wake signal, fired ONLY when new bytes need to leave
    /// this subscriber's ring:
    ///
    /// 1. fanout (`broker.rs`) — after enqueueing a message into the sub's
    ///    ring, calls `notify.notify_one()` to wake `run_send_loop_via_map`
    ///    so it drains.
    /// 2. heartbeat (`heartbeat.rs`) — after enqueueing a `BUS_PING`, same
    ///    `notify.notify_one()`.
    /// 3. [`signal_close`](Self::signal_close) — fires `notify.notify_waiters()`
    ///    as defense-in-depth wake (so a send_loop sleeping in
    ///    `notify.notified().await` wakes up, observes `sub.closed`, and
    ///    returns). This is **belt-and-suspenders** — the authoritative
    ///    close signal is [`close_tx`](Self::close_tx); the notify wake just
    ///    avoids a wait until the next legitimate message.
    ///
    /// **recv_loop MUST NOT listen to this primitive** (SPEC §8.0.quat
    /// invariant 2). Cloned into the broker's `notify_per_sub` map at
    /// subscriber-creation time (handle_connection) so fanout reaches it
    /// without re-locking the subs map.
    ///
    /// **Why this primitive is edge-triggered (and safe)**: legitimate
    /// data wakes are best-effort — losing one wake is harmless because
    /// the next published message re-wakes. The bug that motivated
    /// SPEC §8.0.quat / D-SPEC-131 was the OLD design's reuse of this
    /// notify for shutdown signaling, where its edge-triggered wake +
    /// undefined-choice-of-waiter `notify_one` semantics caused recv_loop
    /// to be probabilistically woken by every fanout tick → recv_loop
    /// exited thinking it was a shutdown → cascade. The fix is the
    /// `close_tx` watch channel below.
    pub notify: Arc<Notify>,

    /// Per-subscriber **close-state** channel (SPEC §8.0.quat close-state
    /// primitive). **Level-triggered** state broadcast: once
    /// [`signal_close`](Self::signal_close) calls `close_tx.send(true)`,
    /// every present and future `watch::Receiver` derived from this
    /// sender observes the `true` value via `close_rx.wait_for(|v| *v)`.
    ///
    /// Subscribers (one each):
    ///
    /// 1. `run_recv_loop` (`server.rs`) — `tokio::select!` arm
    ///    `close_rx.wait_for(|v| *v)` cancels the in-progress
    ///    `read_half.read_exact(...)` and returns.
    /// 2. `run_send_loop_via_map` (`broker.rs`) — same select arm
    ///    cancels the in-progress `notify.notified().await` and returns.
    ///
    /// **Level-triggered correctness**: if `signal_close()` fires BEFORE
    /// a task subscribes a receiver, the receiver's first `.wait_for(...)`
    /// call observes `true` immediately. This eliminates the H1
    /// miss-the-wake race documented in
    /// `titan-docs/HANDOFF_broker_silent_hang_continuation_20260526.md §7`.
    ///
    /// **Why a separate primitive from `notify`** — the two carry
    /// unrelated semantics ("there's new data to flush" vs "you should
    /// exit"). Reusing one edge-triggered primitive for both with
    /// `notify_one` semantics (undefined choice-of-waiter) caused
    /// BUG-AGNO-SILENT-HANG cascade. See SPEC §8.0.quat for the
    /// invariant text + audit boundaries.
    pub close_tx: watch::Sender<bool>,

    /// `true` once this connection has sent at least one `BUS_SUBSCRIBE`
    /// frame. Distinguishes a **pre-subscribe** anon (just connected, not
    /// yet declared intent — a normal connect→subscribe-race transient) from
    /// a subscriber that explicitly subscribed with an EMPTY topic set +
    /// `reply_only=false` (the SPEC §8.2 v1.4.0 / D-SPEC-42 forbidden
    /// regression). Broadcast fanout silently skips the former (no WARN, no
    /// drop counter — mirrors the D-SPEC-45 closed-subscriber skip) and only
    /// loud-drops the latter. Without this the connect→subscribe window
    /// generated WARN+drop spam for every broadcast fired before a
    /// freshly-connected worker finished subscribing (observed at boot:
    /// SPHERE_PULSE/SPIRIT_STATE/MIND_STATE → anon-N).
    pub has_subscribed: bool,
}

impl BrokerSubscriber {
    /// New subscriber with default ring sizes.
    pub fn new(name: impl Into<String>) -> Self {
        let now = SystemTime::now();
        let (close_tx, _) = watch::channel(false);
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
            notify: Arc::new(Notify::new()),
            close_tx,
            has_subscribed: false,
        }
    }

    /// New subscriber with custom ring sizes (used by tests).
    pub fn with_ring(name: impl Into<String>, ring: BoundedRing) -> Self {
        let now = SystemTime::now();
        let (close_tx, _) = watch::channel(false);
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
            notify: Arc::new(Notify::new()),
            close_tx,
            has_subscribed: false,
        }
    }

    /// Mark this subscriber closed AND wake both `run_recv_loop` and
    /// `run_send_loop_via_map` so they exit promptly via the two-primitive
    /// signal split mandated by SPEC §8.0.quat / D-SPEC-131.
    ///
    /// **Effect:**
    ///
    /// 1. Sets `self.closed = true` (in-memory authoritative flag).
    /// 2. `self.close_tx.send(true)` — **level-triggered** state broadcast.
    ///    Every present + future receiver derived from `close_tx` observes
    ///    `true` via `wait_for(|v| *v)`. This is the authoritative shutdown
    ///    signal for recv_loop + send_loop. Survives the H1 miss-the-wake
    ///    race (late awaiter sees the current value).
    /// 3. `self.notify.notify_waiters()` — defense-in-depth edge wake so
    ///    any send_loop currently sleeping in `notify.notified().await`
    ///    returns immediately (instead of waiting for the next legitimate
    ///    data publication). recv_loop is NOT registered on this primitive
    ///    so the wake cannot falsely terminate it.
    ///
    /// **Why both primitives are fired** — per SPEC §8.0.quat invariant 4:
    /// the close_tx is authoritative (level-triggered, never lost), the
    /// notify_waiters is a wake-up optimization. Either primitive alone
    /// would not satisfy: close_tx alone leaves send_loop sleeping until
    /// its next select tick (which is fine — wait_for(true) wakes it via
    /// the watch channel's internal notify); notify alone has the
    /// edge-triggered + dual-purpose problems that caused
    /// BUG-AGNO-SILENT-HANG (D-SPEC-131 root cause).
    ///
    /// **History** — D-SPEC-130 (2026-05-25) introduced `signal_close()` to
    /// fix the original silent-zombie bug (direct `closed = true` left
    /// `run_recv_loop` blocked in `read_half.read_exact` forever — see
    /// `BrokerSubscriber.closed` field docstring). The D-SPEC-130 version
    /// used `notify.notify_waiters()` to wake BOTH recv_loop (via select)
    /// and send_loop. That re-purpose of the data-wake `notify` for
    /// shutdown signaling was the D-SPEC-131 root cause: `notify.notify_one()`
    /// from fanout/heartbeat could probabilistically wake recv_loop instead
    /// of send_loop, killing the connection during normal traffic. The
    /// D-SPEC-131 fix splits the primitives: `notify` reverts to its
    /// pure data-wake role (send_loop only listens), `close_tx` is the
    /// authoritative close-state channel (both loops listen).
    ///
    /// **All call sites must use this method.** Direct `closed = true`
    /// is forbidden going forward (the field is `pub` only because
    /// existing tests construct it directly; production code must
    /// route through `signal_close`).
    ///
    /// Idempotent: safe to call multiple times (`watch::Sender::send` is
    /// a no-op when the value is unchanged; `notify_waiters` on no waiters
    /// is also a no-op).
    pub fn signal_close(&mut self) {
        self.closed = true;
        // Authoritative close signal — level-triggered. `send_replace`
        // updates the watch's stored value AND notifies receivers
        // **regardless of whether any receivers currently exist** (unlike
        // `send` which returns Err and DOES NOT UPDATE the value when
        // receivers=0). Using `send_replace` is load-bearing for the
        // late-subscriber semantics required by SPEC §8.0.quat: a
        // receiver created AFTER `signal_close()` fires (e.g. during
        // reconnect race) MUST still observe `true` via `wait_for(|v| *v)`.
        // With `send`, the value would silently stay `false` if no
        // receivers existed at fire time → late awaiter blocks forever
        // → H1 race re-introduced.
        let _ = self.close_tx.send_replace(true);
        // Defense-in-depth wake for any send_loop currently sleeping
        // in notify.notified().await. recv_loop does NOT listen to this
        // primitive (SPEC §8.0.quat invariant 2) so this cannot
        // falsely terminate recv_loop.
        self.notify.notify_waiters();
    }

    /// Subscribe a fresh `watch::Receiver<bool>` to observe the close
    /// state. Subscribers MUST use `close_rx.wait_for(|v| *v)` to wait
    /// for shutdown (level-triggered — resolves immediately if already
    /// closed at subscribe time).
    ///
    /// Used by `handle_connection` to give recv_task + send_task each
    /// their own receiver.
    pub fn subscribe_close(&self) -> watch::Receiver<bool> {
        self.close_tx.subscribe()
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

    #[test]
    fn new_subscriber_has_not_subscribed_yet() {
        // A freshly-accepted connection (anon-N) has NOT declared intent.
        // Broadcast fanout silent-skips this pre-subscribe transient.
        let sub = BrokerSubscriber::new("anon-1");
        assert!(!sub.has_subscribed);
        assert!(sub.subscribed_topics.is_empty());
        assert!(!sub.reply_only);
    }

    #[test]
    fn pre_subscribe_anon_distinguishable_from_empty_topics_regression() {
        // The §8.2 v1.4.0 / D-SPEC-42 forbidden regression (empty topics +
        // reply_only=false) is loud-dropped by fanout; a pre-subscribe anon
        // in the connect→subscribe race is silently skipped. The two states
        // are identical EXCEPT for `has_subscribed` — the discriminator the
        // fanout guard uses (added with the boot-time WARN-spam fix).
        let pre = BrokerSubscriber::new("anon-2");
        let mut subscribed_empty = BrokerSubscriber::new("worker");
        subscribed_empty.has_subscribed = true; // broker sets this on BUS_SUBSCRIBE
        assert!(pre.subscribed_topics.is_empty() && !pre.reply_only);
        assert!(subscribed_empty.subscribed_topics.is_empty() && !subscribed_empty.reply_only);
        assert_ne!(pre.has_subscribed, subscribed_empty.has_subscribed);
    }

    // ── BUG-AGNO-SILENT-HANG fix tests ───────────────────────────
    // D-SPEC-130 (2026-05-25): signal_close API.
    // D-SPEC-131 (2026-05-26): two-primitive split (close_tx + notify).

    #[tokio::test]
    async fn signal_close_marks_closed_and_wakes_send_loop_notify() {
        // signal_close MUST mark closed AND fire the defense-in-depth
        // notify wake so a send_loop sleeping in notify.notified()
        // returns immediately. recv_loop is NOT a notify listener
        // per SPEC §8.0.quat invariant 2.
        let mut sub = BrokerSubscriber::new("agno_worker");
        assert!(!sub.closed);
        let notify_ref = sub.notify.clone();

        // Spawn a "fake send_loop" that awaits notify.
        let woken = std::sync::Arc::new(tokio::sync::Mutex::new(false));
        let woken_clone = woken.clone();
        let task = tokio::spawn(async move {
            notify_ref.notified().await;
            *woken_clone.lock().await = true;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        sub.signal_close();
        assert!(sub.closed);

        tokio::time::timeout(Duration::from_millis(200), task)
            .await
            .expect(
                "send_loop equivalent did not wake within 200ms — \
                 signal_close not firing defense-in-depth notify",
            )
            .unwrap();
        assert!(*woken.lock().await);
    }

    #[tokio::test]
    async fn signal_close_is_idempotent() {
        // Repeated signal_close must be safe (heartbeat may fire it
        // again on subsequent ticks while purge is in flight).
        // watch::Sender::send is a no-op when the value is unchanged;
        // notify.notify_waiters() on no listeners is also a no-op.
        let mut sub = BrokerSubscriber::new("worker");
        sub.signal_close();
        sub.signal_close();
        sub.signal_close();
        assert!(sub.closed);
        assert!(*sub.close_tx.borrow());
    }

    #[tokio::test]
    async fn signal_close_sets_close_tx_to_true() {
        // SPEC §8.0.quat: signal_close MUST set close_tx to true so
        // both recv_loop and send_loop's wait_for(|v| *v) resolve.
        let mut sub = BrokerSubscriber::new("worker");
        let mut rx = sub.subscribe_close();
        assert!(!*rx.borrow_and_update());

        sub.signal_close();

        // Level-triggered: rx should now see true.
        let closed = tokio::time::timeout(Duration::from_millis(200), rx.wait_for(|v| *v))
            .await
            .expect("close_rx.wait_for(true) did not resolve within 200ms")
            .unwrap();
        assert!(*closed);
    }

    #[tokio::test]
    async fn close_tx_is_level_triggered_late_subscriber_sees_closed() {
        // D-SPEC-131 H1 fix: a receiver subscribed AFTER signal_close()
        // fires MUST still observe closed=true via wait_for(|v| *v).
        // Edge-triggered primitives (Notify) would lose this signal —
        // hence the watch::Sender level-triggered design.
        let mut sub = BrokerSubscriber::new("worker");
        sub.signal_close();

        // Subscribe AFTER the close — late awaiter must see the state.
        let mut rx = sub.subscribe_close();

        let closed = tokio::time::timeout(Duration::from_millis(100), rx.wait_for(|v| *v))
            .await
            .expect(
                "late close_rx subscriber did not see closed=true — \
             SPEC §8.0.quat H1 race regression",
            )
            .unwrap();
        assert!(*closed);
    }

    #[tokio::test]
    async fn data_wake_notify_does_not_trigger_close_rx() {
        // SPEC §8.0.quat invariant 2: data-wake notify and close-state
        // watch are DISTINCT primitives. Firing the data-wake notify
        // (as fanout + heartbeat do via notify.notify_one() / .notify_waiters())
        // MUST NOT cause a close-state receiver's wait_for(|v| *v) to
        // resolve. This is the regression test for D-SPEC-131 (recv_loop
        // was being killed by stray notify wakes under the old design).
        let sub = BrokerSubscriber::new("worker");
        let notify = sub.notify.clone();
        let mut close_rx = sub.subscribe_close();

        // Spawn a task that races a data-wake fire vs the close_rx.
        let task = tokio::spawn(async move {
            tokio::select! {
                _ = close_rx.wait_for(|v| *v) => "closed_unexpectedly",
                _ = tokio::time::sleep(Duration::from_millis(150)) => "timed_out_correctly",
            }
        });

        // Fire data-wake notify several times — should NOT influence close_rx.
        for _ in 0..10 {
            notify.notify_one();
            notify.notify_waiters();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let outcome = task.await.unwrap();
        assert_eq!(
            outcome, "timed_out_correctly",
            "data-wake notify leaked into close_rx — SPEC §8.0.quat regression"
        );
    }

    #[tokio::test]
    async fn signal_close_wakes_multiple_send_loop_waiters() {
        // notify.notify_waiters() in signal_close wakes ALL pending
        // notify listeners (multiple send_loop instances during
        // hypothetical reconnect / race). Cross-check that the
        // defense-in-depth wake still functions.
        let mut sub = BrokerSubscriber::new("worker");
        let n1 = sub.notify.clone();
        let n2 = sub.notify.clone();

        let counter = std::sync::Arc::new(tokio::sync::Mutex::new(0u32));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let t1 = tokio::spawn(async move {
            n1.notified().await;
            *c1.lock().await += 1;
        });
        let t2 = tokio::spawn(async move {
            n2.notified().await;
            *c2.lock().await += 1;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        sub.signal_close();

        tokio::time::timeout(Duration::from_millis(200), async {
            t1.await.unwrap();
            t2.await.unwrap();
        })
        .await
        .expect("notify_waiters() did not wake both listeners");

        assert_eq!(*counter.lock().await, 2);
    }

    #[test]
    fn new_subscriber_has_fresh_notify_and_close_tx() {
        // Each subscriber gets its own Notify AND its own watch::Sender.
        let sub_a = BrokerSubscriber::new("a");
        let sub_b = BrokerSubscriber::new("b");
        assert!(!Arc::ptr_eq(&sub_a.notify, &sub_b.notify));
        // close_tx is its own Sender; subscribing each gives independent
        // receivers (no cross-talk).
        let mut rx_a = sub_a.subscribe_close();
        let mut rx_b = sub_b.subscribe_close();
        assert!(!*rx_a.borrow_and_update());
        assert!(!*rx_b.borrow_and_update());
    }
}
