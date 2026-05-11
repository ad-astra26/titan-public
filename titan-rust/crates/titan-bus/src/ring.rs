//! ring — Bounded per-subscriber queue with P0 reserve region.
//!
//! Byte-identical port of Python `BoundedRing` in `bus_socket.py`.
//!
//! # Capacity layout
//!
//! Per SPEC §3.1 D09 + canonical TOML:
//! - Total capacity: `BUS_RING_CAPACITY_SLOTS=1024`
//! - P0 reserve: `BUS_P0_RESERVE_SLOTS=64`
//! - P1/P2/P3 main region: `1024 - 64 = 960` slots
//!
//! # Drop policy per priority (SPEC §8.0)
//!
//! - **P0** never drop — broker reserves 64 ring slots exclusively for P0.
//! - **P1** drop OLDEST non-P0 of same `(src, type)` + coalesce-replace.
//! - **P2** drop OLDEST under pressure (no coalesce; default lane).
//! - **P3** drop NEWEST under pressure (existing work gets priority to land).
//!
//! # Coalesce semantics
//!
//! When a P1 message with the same `(src, type)` is already in the ring,
//! the broker overwrites the pending message in place — ring slot is NOT
//! consumed twice. The `coalesce_index` lives in [`crate::subscriber`].
//! `BoundedRing` itself just stores messages; coalesce logic is at the
//! subscriber layer.

use std::collections::VecDeque;
use titan_core::bus_specs::Priority;
use titan_core::constants::{BUS_P0_RESERVE_SLOTS, BUS_RING_CAPACITY_SLOTS};

/// Errors during ring operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RingError {
    /// Configuration error: P0 reserve >= total capacity.
    #[error("p0_reserve {reserve} >= capacity {capacity}")]
    InvalidConfig {
        /// Configured P0 reserve.
        reserve: usize,
        /// Configured total capacity.
        capacity: usize,
    },
    /// P0 reserve full (extreme overflow — broker logs as critical event).
    #[error("P0 reserve full ({size} slots) — extreme overflow")]
    P0ReserveFull {
        /// Configured P0 reserve size.
        size: usize,
    },
}

/// Outcome of an enqueue operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// Message accepted into the ring with no eviction.
    Accepted,
    /// Message accepted but oldest non-P0 was evicted to make room
    /// (drop event — counts toward `BUS_SLOW_CONSUMER` drop-rate ratio).
    AcceptedWithEviction,
    /// Message rejected (P3 drop-newest under pressure).
    Rejected,
}

/// One generic message envelope stored in the ring. The broker treats it
/// as opaque bytes once enqueued — wire framing already happened.
///
/// Coalesce hot-path uses `(src, msg_type)` as a key; both fields are kept
/// alongside the payload so the subscriber's coalesce_index can look up
/// in-place without re-parsing the payload.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Canonical message type (e.g. `"BODY_STATE"`).
    pub msg_type: String,
    /// Source identifier (e.g. `"inner-body"`).
    pub src: String,
    /// Priority lane (cached from spec lookup at enqueue time).
    pub priority: Priority,
    /// Msgpack-encoded payload (or other opaque wire bytes).
    pub payload: Vec<u8>,
}

/// Bounded queue with P0 reserve region.
///
/// Designed to be small + easy to port to `crossbeam::ArrayQueue` semantics
/// in Phase D if hot-path latency requires lock-free.
pub struct BoundedRing {
    main: VecDeque<Envelope>, // P1/P2/P3 — main region
    p0: VecDeque<Envelope>,   // P0 reserve — never overwritten by non-P0
    capacity: usize,
    p0_reserve: usize,
}

impl BoundedRing {
    /// Construct with default sizes (`BUS_RING_CAPACITY_SLOTS=1024` total +
    /// `BUS_P0_RESERVE_SLOTS=64` reserved for P0).
    pub fn with_defaults() -> Self {
        Self::new(
            BUS_RING_CAPACITY_SLOTS as usize,
            BUS_P0_RESERVE_SLOTS as usize,
        )
        .expect("default ring config must be valid")
    }

    /// Construct with custom sizes (used by tests + adaptive sizing).
    /// Returns `RingError::InvalidConfig` if `p0_reserve >= capacity`.
    pub fn new(capacity: usize, p0_reserve: usize) -> Result<Self, RingError> {
        if p0_reserve >= capacity {
            return Err(RingError::InvalidConfig {
                reserve: p0_reserve,
                capacity,
            });
        }
        let main_capacity = capacity - p0_reserve;
        Ok(Self {
            main: VecDeque::with_capacity(main_capacity),
            p0: VecDeque::with_capacity(p0_reserve),
            capacity,
            p0_reserve,
        })
    }

    /// Total slots configured (`capacity = main_capacity + p0_reserve`).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// P0 reserve size.
    pub fn p0_reserve(&self) -> usize {
        self.p0_reserve
    }

    /// Main (non-P0) region capacity.
    pub fn main_capacity(&self) -> usize {
        self.capacity - self.p0_reserve
    }

    /// Total messages currently in the ring (main + p0).
    pub fn len(&self) -> usize {
        self.main.len() + self.p0.len()
    }

    /// Returns `true` if both regions are empty.
    pub fn is_empty(&self) -> bool {
        self.main.is_empty() && self.p0.is_empty()
    }

    /// Returns `true` if main region is at capacity.
    pub fn main_is_full(&self) -> bool {
        self.main.len() == self.main_capacity()
    }

    /// Returns `true` if P0 reserve is at capacity (extreme overflow).
    pub fn p0_is_full(&self) -> bool {
        self.p0.len() == self.p0_reserve
    }

    /// Enqueue a message according to its priority lane (SPEC §8.0).
    ///
    /// - **P0**: appended to reserve. If reserve full → `Err(P0ReserveFull)`
    ///   (broker logs as critical event; should never happen in normal ops).
    /// - **P1, P2**: appended to main. If main full → drops OLDEST main slot
    ///   to make room → returns `AcceptedWithEviction`.
    /// - **P3**: if main full → returns `Rejected` (drop-newest semantics).
    ///
    /// Coalesce-replace for P1 same-`(src,type)` happens at the subscriber
    /// layer BEFORE this function is called. By the time `enqueue()` runs,
    /// the message is known to be a fresh slot.
    pub fn enqueue(&mut self, msg: Envelope) -> Result<EnqueueOutcome, RingError> {
        match msg.priority {
            Priority::P0 => {
                if self.p0_is_full() {
                    Err(RingError::P0ReserveFull {
                        size: self.p0_reserve,
                    })
                } else {
                    self.p0.push_back(msg);
                    Ok(EnqueueOutcome::Accepted)
                }
            }
            Priority::P1 | Priority::P2 => {
                if self.main_is_full() {
                    self.main.pop_front();
                    self.main.push_back(msg);
                    Ok(EnqueueOutcome::AcceptedWithEviction)
                } else {
                    self.main.push_back(msg);
                    Ok(EnqueueOutcome::Accepted)
                }
            }
            Priority::P3 => {
                if self.main_is_full() {
                    Ok(EnqueueOutcome::Rejected)
                } else {
                    self.main.push_back(msg);
                    Ok(EnqueueOutcome::Accepted)
                }
            }
        }
    }

    /// Pop up to `max_msgs` messages, P0 first then main, FIFO within each.
    /// Used by the broker's send loop (one per subscriber).
    pub fn pop_for_send(&mut self, max_msgs: usize) -> Vec<Envelope> {
        let mut out = Vec::with_capacity(max_msgs.min(self.len()));
        while !self.p0.is_empty() && out.len() < max_msgs {
            if let Some(e) = self.p0.pop_front() {
                out.push(e);
            }
        }
        while !self.main.is_empty() && out.len() < max_msgs {
            if let Some(e) = self.main.pop_front() {
                out.push(e);
            }
        }
        out
    }

    /// Coalesce-replace: find the slot in the main region with the same
    /// `(src, msg_type)` and overwrite its payload. Returns `true` if a
    /// match was found and replaced; `false` if none found.
    ///
    /// Caller (subscriber.rs) maintains a coalesce_index so this function
    /// only fires after an O(1) hash lookup. Linear scan over main is the
    /// fallback for cache-miss / index-rebuild scenarios.
    pub fn coalesce_replace(&mut self, msg: &Envelope) -> bool {
        for slot in self.main.iter_mut() {
            if slot.msg_type == msg.msg_type && slot.src == msg.src {
                slot.payload = msg.payload.clone();
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(msg_type: &str, src: &str, priority: Priority, payload: &[u8]) -> Envelope {
        Envelope {
            msg_type: msg_type.to_string(),
            src: src.to_string(),
            priority,
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn defaults_match_spec_constants() {
        let r = BoundedRing::with_defaults();
        assert_eq!(r.capacity(), 1024);
        assert_eq!(r.p0_reserve(), 64);
        assert_eq!(r.main_capacity(), 960);
    }

    #[test]
    fn invalid_config_rejected() {
        let result = BoundedRing::new(64, 64);
        assert_eq!(
            result.err(),
            Some(RingError::InvalidConfig {
                reserve: 64,
                capacity: 64
            })
        );
    }

    #[test]
    fn enqueue_p0_into_reserve() {
        let mut r = BoundedRing::new(10, 4).unwrap();
        let result = r
            .enqueue(env("KERNEL_EPOCH_TICK", "kernel", Priority::P0, b"x"))
            .unwrap();
        assert_eq!(result, EnqueueOutcome::Accepted);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn enqueue_p1_into_main() {
        let mut r = BoundedRing::new(10, 4).unwrap();
        let result = r
            .enqueue(env("BODY_STATE", "inner-body", Priority::P1, b"x"))
            .unwrap();
        assert_eq!(result, EnqueueOutcome::Accepted);
    }

    #[test]
    fn p0_reserve_full_returns_error() {
        let mut r = BoundedRing::new(10, 2).unwrap();
        r.enqueue(env("BUS_PING", "broker", Priority::P0, b"x"))
            .unwrap();
        r.enqueue(env("BUS_PING", "broker", Priority::P0, b"x"))
            .unwrap();
        let result = r.enqueue(env("BUS_PING", "broker", Priority::P0, b"x"));
        assert_eq!(result.err(), Some(RingError::P0ReserveFull { size: 2 }));
    }

    #[test]
    fn p2_full_evicts_oldest_main() {
        let mut r = BoundedRing::new(5, 2).unwrap(); // main_capacity = 3
        for i in 0..3 {
            let res = r.enqueue(env(
                "REASONING_CHAIN",
                "reasoning",
                Priority::P2,
                &[i as u8],
            ));
            assert_eq!(res.unwrap(), EnqueueOutcome::Accepted);
        }
        // Main full now (3 of 3). Next enqueue evicts oldest.
        let res = r
            .enqueue(env("REASONING_CHAIN", "reasoning", Priority::P2, &[3]))
            .unwrap();
        assert_eq!(res, EnqueueOutcome::AcceptedWithEviction);
        // Main still has 3 items; oldest (payload [0]) is gone.
        assert_eq!(r.main.len(), 3);
        assert_eq!(r.main.front().unwrap().payload, vec![1]);
        assert_eq!(r.main.back().unwrap().payload, vec![3]);
    }

    #[test]
    fn p3_full_rejects_newest() {
        let mut r = BoundedRing::new(5, 2).unwrap(); // main_capacity = 3
        for _ in 0..3 {
            r.enqueue(env("OBSERVATORY_EVENT", "obs", Priority::P3, b"x"))
                .unwrap();
        }
        // Main full. Next P3 enqueue is rejected.
        let res = r
            .enqueue(env("OBSERVATORY_EVENT", "obs", Priority::P3, b"new"))
            .unwrap();
        assert_eq!(res, EnqueueOutcome::Rejected);
        // Main UNCHANGED — drop-newest preserves existing work.
        assert_eq!(r.main.len(), 3);
    }

    #[test]
    fn pop_p0_first_then_main() {
        let mut r = BoundedRing::new(10, 4).unwrap();
        r.enqueue(env("MIND_STATE", "inner-mind", Priority::P1, b"m1"))
            .unwrap();
        r.enqueue(env("KERNEL_EPOCH_TICK", "kernel", Priority::P0, b"k1"))
            .unwrap();
        r.enqueue(env("BODY_STATE", "inner-body", Priority::P1, b"b1"))
            .unwrap();
        let popped = r.pop_for_send(10);
        // P0 first
        assert_eq!(popped[0].msg_type, "KERNEL_EPOCH_TICK");
        // Main FIFO
        assert_eq!(popped[1].msg_type, "MIND_STATE");
        assert_eq!(popped[2].msg_type, "BODY_STATE");
    }

    #[test]
    fn pop_respects_max_msgs_limit() {
        let mut r = BoundedRing::new(10, 4).unwrap();
        for i in 0..5 {
            r.enqueue(env("BODY_STATE", "inner-body", Priority::P1, &[i as u8]))
                .unwrap();
        }
        let popped = r.pop_for_send(3);
        assert_eq!(popped.len(), 3);
        assert_eq!(r.len(), 2); // 2 remaining
    }

    #[test]
    fn coalesce_replace_overwrites_in_place() {
        let mut r = BoundedRing::new(10, 2).unwrap();
        r.enqueue(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        r.enqueue(env("MIND_STATE", "inner-mind", Priority::P1, b"v1"))
            .unwrap();
        let replacement = env("BODY_STATE", "inner-body", Priority::P1, b"v2");
        let did_replace = r.coalesce_replace(&replacement);
        assert!(did_replace);
        assert_eq!(r.len(), 2); // still 2 messages
                                // Verify the BODY_STATE payload is now v2
        let popped = r.pop_for_send(10);
        let body = popped.iter().find(|e| e.msg_type == "BODY_STATE").unwrap();
        assert_eq!(body.payload, b"v2");
    }

    #[test]
    fn coalesce_replace_returns_false_if_no_match() {
        let mut r = BoundedRing::new(10, 2).unwrap();
        r.enqueue(env("BODY_STATE", "inner-body", Priority::P1, b"v1"))
            .unwrap();
        let replacement = env("MIND_STATE", "inner-mind", Priority::P1, b"v1");
        let did_replace = r.coalesce_replace(&replacement);
        assert!(!did_replace);
    }

    #[test]
    fn empty_ring_pops_nothing() {
        let mut r = BoundedRing::new(10, 2).unwrap();
        let popped = r.pop_for_send(10);
        assert!(popped.is_empty());
    }
}
