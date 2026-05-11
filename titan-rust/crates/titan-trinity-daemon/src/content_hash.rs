//! content_hash — Per-slot content-hash gating per SPEC §7.1.
//!
//! Slots marked "Content-hash gated: Yes" in SPEC §7.1 are written ONLY when
//! the payload bytes differ from the previous write. Saves ~99% of slot
//! writes when state is steady — matches `state_registry.py:103-115`'s
//! gating logic byte-identically.
//!
//! # Why xxh3_64
//!
//! - Non-cryptographic — we only need collision avoidance for "did the
//!   payload change?", not security.
//! - ~1.7 GB/s on x86_64 — negligible cost at 70.47 Hz × 180 bytes
//!   (worst-case: spirit at ~12 MB/s peak demand).
//! - Same hash family as Python `xxhash.xxh3_64_intdigest()` used in
//!   `state_registry.py` content-gate (byte-identical hash output for
//!   identical payload bytes — verified in tests/parity_xxhash.rs of C-S2).
//!
//! # Usage
//!
//! ```ignore
//! let mut gate = ContentGate::new();
//! let payload = [0u8; 20];
//! if gate.should_write(&payload) {
//!     slot.write(&payload)?;
//! }
//! // Subsequent calls with same payload return false.
//! assert!(!gate.should_write(&payload));
//! ```

use xxhash_rust::xxh3::xxh3_64;

/// Per-slot content-hash gate. Holds the last-written hash and returns
/// false on subsequent calls with identical payload.
///
/// The gate is single-writer (one daemon per slot per SPEC §7.1 writer
/// column) so no synchronization is needed.
#[derive(Debug, Clone, Default)]
pub struct ContentGate {
    /// Hash of the last payload that returned `should_write(true)`.
    /// `None` until the first call.
    last_hash: Option<u64>,
    /// Number of times `should_write` was called and returned true (i.e.
    /// the slot would be written). Used by daemon stats endpoints.
    write_count: u64,
    /// Number of times `should_write` was called and returned false (i.e.
    /// the slot write was suppressed because content unchanged).
    suppress_count: u64,
}

impl ContentGate {
    /// New gate with no prior payload — first `should_write` call always
    /// returns true.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if `payload` differs from the previously gated
    /// payload (or if this is the first call). On `true`, updates the
    /// internal hash.
    ///
    /// On `false`, the gate state is unchanged but `suppress_count`
    /// increments.
    pub fn should_write(&mut self, payload: &[u8]) -> bool {
        let hash = xxh3_64(payload);
        let changed = self.last_hash != Some(hash);
        if changed {
            self.last_hash = Some(hash);
            self.write_count = self.write_count.saturating_add(1);
        } else {
            self.suppress_count = self.suppress_count.saturating_add(1);
        }
        changed
    }

    /// Force the gate to forget its last hash — next call always returns
    /// true. Used during graceful resume from a snapshot where we want
    /// the first post-resume tick to write regardless.
    pub fn reset(&mut self) {
        self.last_hash = None;
    }

    /// Number of writes that PASSED the gate.
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Number of writes that were SUPPRESSED by the gate (content
    /// unchanged).
    pub fn suppress_count(&self) -> u64 {
        self.suppress_count
    }

    /// Total `should_write()` calls = `write_count + suppress_count`.
    pub fn total_calls(&self) -> u64 {
        self.write_count.saturating_add(self.suppress_count)
    }

    /// Suppress ratio: fraction of calls that returned false. Returns 0.0
    /// if no calls yet. Useful for stats: a healthy steady-state daemon
    /// shows >90% suppression.
    pub fn suppress_ratio(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.suppress_count as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_always_writes() {
        let mut g = ContentGate::new();
        assert!(g.should_write(b"hello"));
        assert_eq!(g.write_count(), 1);
        assert_eq!(g.suppress_count(), 0);
    }

    #[test]
    fn identical_payload_suppressed() {
        let mut g = ContentGate::new();
        let payload = b"steady state";
        assert!(g.should_write(payload));
        assert!(!g.should_write(payload));
        assert!(!g.should_write(payload));
        assert!(!g.should_write(payload));
        assert_eq!(g.write_count(), 1);
        assert_eq!(g.suppress_count(), 3);
    }

    #[test]
    fn changed_payload_writes() {
        let mut g = ContentGate::new();
        assert!(g.should_write(b"v1"));
        assert!(g.should_write(b"v2"));
        assert!(g.should_write(b"v3"));
        assert_eq!(g.write_count(), 3);
        assert_eq!(g.suppress_count(), 0);
    }

    #[test]
    fn return_to_prior_payload_writes_again() {
        // Once a different payload comes in, returning to the original
        // counts as a CHANGE (the gate only tracks the LAST hash).
        let mut g = ContentGate::new();
        g.should_write(b"v1");
        g.should_write(b"v2");
        assert!(g.should_write(b"v1"));
        assert_eq!(g.write_count(), 3);
    }

    #[test]
    fn empty_payload_handled() {
        let mut g = ContentGate::new();
        assert!(g.should_write(&[]));
        assert!(!g.should_write(&[]));
    }

    #[test]
    fn reset_forces_next_write() {
        let mut g = ContentGate::new();
        g.should_write(b"steady");
        assert!(!g.should_write(b"steady"));
        g.reset();
        assert!(g.should_write(b"steady"));
    }

    #[test]
    fn float_payload_byte_identical_suppressed() {
        // 5 × float32 LE = 20 bytes; identical floats → identical bytes →
        // suppressed.
        let mut g = ContentGate::new();
        let payload: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes: [u8; 20] = unsafe { std::mem::transmute(payload) };
        assert!(g.should_write(&bytes));
        assert!(!g.should_write(&bytes));
    }

    #[test]
    fn near_identical_floats_not_suppressed() {
        // Even tiny float differences change the byte representation →
        // gate sees them as changed (correct behavior for SeqLock slots).
        let mut g = ContentGate::new();
        let p1: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        let p2: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.50001];
        let b1: [u8; 20] = unsafe { std::mem::transmute(p1) };
        let b2: [u8; 20] = unsafe { std::mem::transmute(p2) };
        assert!(g.should_write(&b1));
        assert!(g.should_write(&b2));
    }

    #[test]
    fn suppress_ratio_initially_zero() {
        let g = ContentGate::new();
        assert_eq!(g.suppress_ratio(), 0.0);
    }

    #[test]
    fn suppress_ratio_after_steady_state() {
        let mut g = ContentGate::new();
        g.should_write(b"steady");
        for _ in 0..9 {
            g.should_write(b"steady");
        }
        // 1 write, 9 suppresses → 90% suppression
        assert!((g.suppress_ratio() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn total_calls_sums_writes_and_suppresses() {
        let mut g = ContentGate::new();
        g.should_write(b"a"); // write
        g.should_write(b"a"); // suppress
        g.should_write(b"b"); // write
        g.should_write(b"b"); // suppress
        g.should_write(b"b"); // suppress
        assert_eq!(g.total_calls(), 5);
        assert_eq!(g.write_count(), 2);
        assert_eq!(g.suppress_count(), 3);
    }
}
