//! Boot-window buffer for late-subscriber recovery (SPEC §8.0.bis).
//!
//! When a targeted P0 message arrives at the broker but the destination
//! subscriber has not yet attached (boot-race window), the buffer holds
//! the frame until the subscriber registers — then drains in arrival
//! order. Bounded by TTL + per-destination frame cap.
//!
//! Closes the bootstrap-race class:
//!   1. Initial boot — subscriber emits MODULE_READY before Guardian's
//!      `guardian` alias is active on parent's connection
//!   2. Meditation cold-start race — first meditation msg emitted before
//!      memory_worker attaches to broker (per
//!      `feedback_meditation_cold_start_race.md`)
//!   3. Module hot-reload window — old subprocess exits, new subscribes
//!      ~seconds later (Phase B of `rFP_phase_c_bus_delivery_continuity_and_hot_reload`)
//!   4. Network reconnect — transient socket reset
//!   5. Kernel shadow swap (within same broker, post-reconnect)
//!
//! Buffered types are restricted to `BOOT_BUFFERED_TYPES` (lifecycle +
//! supervision only). State transport (G18) is untouched. Broadcasts
//! (dst="all") are untouched. Swap-protocol messages (SWAP_HANDOFF etc.)
//! are NOT buffered — they must deliver immediately or fail loudly.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Maximum frames buffered per destination subscriber name. Bounded to
/// absorb a worker's boot-window emissions (MODULE_READY + AGENCY_READY
/// + first heartbeats) without unbounded growth.
///
/// SPEC §8.0.bis constant. Matches Python `_phase_c_constants.BOOT_BUFFER_MAX_FRAMES_PER_DST`.
pub const BOOT_BUFFER_MAX_FRAMES_PER_DST: usize = 32;

/// TTL for buffered frames. Aligns with the SPEC §11.B module boot timeout
/// (60s) so real boot failures still surface as BOOT_FAILURE; frames
/// older than this are silently dropped on next drain or GC pass.
///
/// SPEC §8.0.bis constant. Matches Python `_phase_c_constants.BOOT_BUFFER_TTL_S`.
pub const BOOT_BUFFER_TTL_S: f64 = 60.0;

/// Targeted message types eligible for boot-window buffering. Restricted
/// to lifecycle + supervision frames where late-delivery is preferable
/// to drop. Application RPCs, broadcasts, swap protocol messages are
/// explicitly NOT buffered.
///
/// SPEC §8.0.bis. Mirrored in Python `bus_constants.BOOT_BUFFERED_TYPES`.
pub const BOOT_BUFFERED_TYPES: &[&str] = &[
    // Module lifecycle (§8.1)
    "MODULE_READY",
    "MODULE_HEARTBEAT",
    "MODULE_SHUTDOWN",
    "MODULE_CRASHED",
    // Supervision lifecycle (§8.1)
    "SUPERVISION_CHILD_DOWN",
    "SUPERVISION_CHILD_RESTARTED",
    // Service-ready broadcasts when targeted
    "AGENCY_READY",
    "NS_READY",
    "MEMORY_READY",
    // Phase B reload ACK (Phase B of this rFP; reserved here per §8.3)
    "MODULE_RELOAD_ACK",
];

/// True if `msg_type` is eligible for boot-window buffering per SPEC §8.0.bis.
#[inline]
pub fn is_boot_buffered_type(msg_type: &str) -> bool {
    BOOT_BUFFERED_TYPES.contains(&msg_type)
}

/// One buffered frame: when it arrived + the raw bytes to deliver
/// once the destination subscriber attaches.
#[derive(Debug)]
pub struct BufferedFrame {
    /// Monotonic arrival timestamp for TTL eviction.
    pub arrived_at: Instant,
    /// Encoded msgpack frame bytes (matches subscriber's send-queue format).
    pub raw_bytes: Vec<u8>,
    /// `msg_type` extracted at insert time — for observability + GC logging.
    pub msg_type: String,
}

/// Identity-keyed buffer: `dst_name → queue of (arrived_at, frame_bytes)`.
///
/// Memory bound: `BOOT_BUFFER_MAX_FRAMES_PER_DST` × N destinations
/// × avg frame size. Per-destination eviction on overflow; TTL eviction
/// on drain or lazy GC. Total memory is observably bounded at all times.
#[derive(Default, Debug)]
pub struct BootBuffer {
    inner: HashMap<String, VecDeque<BufferedFrame>>,
    /// Per-destination overflow log rate-limiter — last overflow log time.
    /// Rate-limit: 1 log per (dst, 60s) — keeps log volume bounded under
    /// pathological producers.
    last_overflow_log: HashMap<String, Instant>,
}

impl BootBuffer {
    /// New empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a frame for destination `dst`. Returns `true` if buffered,
    /// `false` if rejected (msg_type not in `BOOT_BUFFERED_TYPES`).
    ///
    /// Overflow handling: if the destination already has
    /// `BOOT_BUFFER_MAX_FRAMES_PER_DST` frames, the oldest is evicted
    /// to make room (FIFO drop-oldest semantics). Overflow logged WARN
    /// rate-limited to 1/60s per destination.
    pub fn push(
        &mut self,
        dst: &str,
        msg_type: String,
        raw_bytes: Vec<u8>,
        now: Instant,
    ) -> BootBufferPushOutcome {
        if !is_boot_buffered_type(&msg_type) {
            return BootBufferPushOutcome::TypeNotBuffered;
        }
        let queue = self.inner.entry(dst.to_string()).or_default();
        let mut overflowed = false;
        while queue.len() >= BOOT_BUFFER_MAX_FRAMES_PER_DST {
            queue.pop_front();
            overflowed = true;
        }
        queue.push_back(BufferedFrame {
            arrived_at: now,
            raw_bytes,
            msg_type,
        });
        if overflowed {
            // Rate-limit overflow log: 1 per (dst, 60s)
            let should_log = match self.last_overflow_log.get(dst) {
                Some(prev) => now.duration_since(*prev) > Duration::from_secs_f64(60.0),
                None => true,
            };
            if should_log {
                self.last_overflow_log.insert(dst.to_string(), now);
            }
            return if should_log {
                BootBufferPushOutcome::BufferedOverflowLogged
            } else {
                BootBufferPushOutcome::BufferedOverflowSilent
            };
        }
        BootBufferPushOutcome::Buffered
    }

    /// Drain all buffered frames for destination `dst`, evicting any
    /// frames older than `BOOT_BUFFER_TTL_S`. Returns frames in arrival
    /// order (FIFO). Drained entries are removed from the buffer.
    ///
    /// Called when a `BUS_SUBSCRIBE` registers `dst` as primary name or
    /// alias.
    pub fn drain(&mut self, dst: &str, now: Instant) -> Vec<BufferedFrame> {
        let ttl = Duration::from_secs_f64(BOOT_BUFFER_TTL_S);
        let queue = match self.inner.remove(dst) {
            Some(q) => q,
            None => return Vec::new(),
        };
        let drained: Vec<BufferedFrame> = queue
            .into_iter()
            .filter(|frame| now.duration_since(frame.arrived_at) <= ttl)
            .collect();
        drained
    }

    /// Lazy GC pass: evict TTL-expired entries across all destinations.
    /// Designed to be called from within `fanout` so no extra timer
    /// thread is needed. O(N) where N = total buffered frames; typical
    /// N is in single digits during boot, zero in steady state.
    ///
    /// Returns the count of evicted frames.
    pub fn gc(&mut self, now: Instant) -> usize {
        let ttl = Duration::from_secs_f64(BOOT_BUFFER_TTL_S);
        let mut evicted_total: usize = 0;
        let mut empty_keys: Vec<String> = Vec::new();
        for (dst, queue) in self.inner.iter_mut() {
            let before = queue.len();
            queue.retain(|frame| now.duration_since(frame.arrived_at) <= ttl);
            evicted_total += before - queue.len();
            if queue.is_empty() {
                empty_keys.push(dst.clone());
            }
        }
        for key in empty_keys {
            self.inner.remove(&key);
        }
        evicted_total
    }

    /// Total buffered frame count across all destinations (observability).
    pub fn total_frames(&self) -> usize {
        self.inner.values().map(|q| q.len()).sum()
    }

    /// Number of distinct destinations currently buffered (observability).
    pub fn destination_count(&self) -> usize {
        self.inner.len()
    }

    /// Frame count for a specific destination (test/observability).
    #[cfg(test)]
    pub fn dst_frame_count(&self, dst: &str) -> usize {
        self.inner.get(dst).map(|q| q.len()).unwrap_or(0)
    }
}

/// Result of a `push` call — used by the broker for logging decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootBufferPushOutcome {
    /// Frame buffered successfully, no overflow.
    Buffered,
    /// Frame buffered, but oldest was evicted to make room. Overflow
    /// WARN log SHOULD be emitted by the caller.
    BufferedOverflowLogged,
    /// Same as above but rate-limiter suppressed the log.
    BufferedOverflowSilent,
    /// `msg_type` not in `BOOT_BUFFERED_TYPES` — frame NOT buffered.
    /// Caller falls through to the existing fanout WARN+drop path
    /// (SPEC §8.2 v1.4.0 D-SPEC-42 forbidden-regression handling, etc.).
    TypeNotBuffered,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ts(after_s: f64) -> Instant {
        Instant::now() + Duration::from_secs_f64(after_s)
    }

    #[test]
    fn buffers_only_listed_types() {
        let mut bb = BootBuffer::new();
        let now = ts(0.0);
        assert_eq!(
            bb.push("agency_worker", "MODULE_READY".into(), vec![1, 2, 3], now),
            BootBufferPushOutcome::Buffered
        );
        assert_eq!(
            bb.push("agency_worker", "RANDOM_APP_MSG".into(), vec![4, 5, 6], now),
            BootBufferPushOutcome::TypeNotBuffered
        );
        assert_eq!(bb.dst_frame_count("agency_worker"), 1);
    }

    #[test]
    fn fifo_drain_order_preserved() {
        let mut bb = BootBuffer::new();
        let now = ts(0.0);
        bb.push("worker", "MODULE_READY".into(), vec![1], now);
        bb.push("worker", "MODULE_HEARTBEAT".into(), vec![2], now);
        bb.push("worker", "MODULE_HEARTBEAT".into(), vec![3], now);

        let drained = bb.drain("worker", now);
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].raw_bytes, vec![1]);
        assert_eq!(drained[1].raw_bytes, vec![2]);
        assert_eq!(drained[2].raw_bytes, vec![3]);
        // Buffer empty after drain
        assert_eq!(bb.dst_frame_count("worker"), 0);
    }

    #[test]
    fn ttl_evicts_on_drain() {
        let mut bb = BootBuffer::new();
        let t0 = Instant::now();
        bb.push("worker", "MODULE_READY".into(), vec![1], t0);
        // Drain 61s later — past TTL
        let drained = bb.drain("worker", t0 + Duration::from_secs(61));
        assert_eq!(drained.len(), 0, "TTL-expired frames dropped on drain");
    }

    #[test]
    fn overflow_evicts_oldest() {
        let mut bb = BootBuffer::new();
        let now = ts(0.0);
        for i in 0..BOOT_BUFFER_MAX_FRAMES_PER_DST {
            bb.push("w", "MODULE_HEARTBEAT".into(), vec![i as u8], now);
        }
        assert_eq!(bb.dst_frame_count("w"), BOOT_BUFFER_MAX_FRAMES_PER_DST);
        // 33rd frame — first overflow, logged
        let outcome = bb.push("w", "MODULE_HEARTBEAT".into(), vec![99], now);
        assert_eq!(outcome, BootBufferPushOutcome::BufferedOverflowLogged);
        assert_eq!(bb.dst_frame_count("w"), BOOT_BUFFER_MAX_FRAMES_PER_DST);
        // Drain: oldest gone, newest present
        let drained = bb.drain("w", now);
        assert_eq!(drained[0].raw_bytes, vec![1]); // 0 was evicted
        assert_eq!(drained.last().unwrap().raw_bytes, vec![99]);
    }

    #[test]
    fn overflow_log_rate_limited() {
        let mut bb = BootBuffer::new();
        let now = ts(0.0);
        for _ in 0..BOOT_BUFFER_MAX_FRAMES_PER_DST {
            bb.push("w", "MODULE_HEARTBEAT".into(), vec![0], now);
        }
        // 1st overflow logs, 2nd overflow within 60s silent
        let o1 = bb.push("w", "MODULE_HEARTBEAT".into(), vec![1], now);
        let o2 = bb.push("w", "MODULE_HEARTBEAT".into(), vec![2], now);
        assert_eq!(o1, BootBufferPushOutcome::BufferedOverflowLogged);
        assert_eq!(o2, BootBufferPushOutcome::BufferedOverflowSilent);
        // After 61s, log eligible again
        let later = now + Duration::from_secs(61);
        let o3 = bb.push("w", "MODULE_HEARTBEAT".into(), vec![3], later);
        assert_eq!(o3, BootBufferPushOutcome::BufferedOverflowLogged);
    }

    #[test]
    fn lazy_gc_evicts_expired_and_drops_empty_queues() {
        let mut bb = BootBuffer::new();
        let t0 = Instant::now();
        bb.push("w1", "MODULE_READY".into(), vec![1], t0);
        bb.push("w2", "MODULE_HEARTBEAT".into(), vec![2], t0);
        bb.push(
            "w2",
            "MODULE_HEARTBEAT".into(),
            vec![3],
            t0 + Duration::from_secs(30),
        );

        // GC at t+61 — both w1's and w2's first frame expired; w2's second alive
        let evicted = bb.gc(t0 + Duration::from_secs(61));
        assert_eq!(evicted, 2);
        assert_eq!(bb.destination_count(), 1, "w1 fully evicted, queue dropped");
        assert_eq!(bb.dst_frame_count("w2"), 1);
    }

    #[test]
    fn drain_nonexistent_dst_returns_empty() {
        let mut bb = BootBuffer::new();
        let drained = bb.drain("never_registered", Instant::now());
        assert!(drained.is_empty());
    }
}
