//! handle — Typed [`Producer`] / [`Consumer`] handles over the SPSC ring.
//!
//! Per PLAN §9.5 memory ordering rules. Handles borrow from the [`Ring`] and
//! are tied to its lifetime; SPSC discipline (one writer, one reader per
//! direction) is the caller's responsibility.

use std::sync::atomic::Ordering;

use crate::ring::{FastbusError, RingHeader};
use crate::FASTBUS_SLOT_BYTES;

/// Single-producer side of the ring. Owns `write_idx` and slot writes.
///
/// Per PLAN §9.5: producer does `Acquire` on `read_idx` (sees consumer
/// commits) and `Release` on `write_idx` (publishes payload before the
/// consumer can `Acquire` it).
pub struct Producer<'a> {
    pub(crate) header: &'a RingHeader,
    pub(crate) slots: *mut u8,
    pub(crate) slots_len: usize,
}

// SAFETY: slots pointer points into mmap'd memory whose lifetime exceeds 'a.
// We discipline access via SPSC ordering rules; no aliasing within a single
// process side. The producer is the only writer to the slot it claims via
// `(write_idx & mask)`.
unsafe impl Send for Producer<'_> {}

impl<'a> Producer<'a> {
    /// Publish a 256-byte slot. Returns the index that was written (the
    /// producer's pre-bump `write_idx`).
    ///
    /// Errors:
    /// - [`FastbusError::QueueFull`] if `write_idx - read_idx >= capacity` —
    ///   producer must back off / drop / retry.
    pub fn publish(
        &mut self,
        payload: &[u8; FASTBUS_SLOT_BYTES as usize],
    ) -> Result<u64, FastbusError> {
        let mask = self.header.mask.load(Ordering::Relaxed);
        let capacity = (mask as u64) + 1;
        let write_idx = self.header.write_idx.load(Ordering::Relaxed);
        // Acquire on read_idx — see consumer's commits before deciding whether
        // we have a free slot.
        let read_idx = self.header.read_idx.load(Ordering::Acquire);
        if write_idx.wrapping_sub(read_idx) >= capacity {
            return Err(FastbusError::QueueFull {
                capacity: capacity as u32,
            });
        }
        let slot_idx = (write_idx & mask as u64) as usize;
        let offset = slot_idx * (FASTBUS_SLOT_BYTES as usize);
        debug_assert!(
            offset + (FASTBUS_SLOT_BYTES as usize) <= self.slots_len,
            "fastbus slot offset out of bounds"
        );
        // SAFETY: offset + 256 <= slots_len (debug-asserted); only the producer
        // writes this slot per SPSC discipline; bytes are POD; well-aligned for u8.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                self.slots.add(offset),
                FASTBUS_SLOT_BYTES as usize,
            );
        }
        // Release on write_idx — payload writes happens-before any consumer
        // Acquire-load of write_idx that observes this index.
        self.header
            .write_idx
            .store(write_idx + 1, Ordering::Release);
        Ok(write_idx)
    }

    /// Returns the producer's pre-published `write_idx` and the consumer's
    /// `read_idx` (Acquire) — useful for diagnostics + tests.
    pub fn indices(&self) -> (u64, u64) {
        let r = self.header.read_idx.load(Ordering::Acquire);
        let w = self.header.write_idx.load(Ordering::Relaxed);
        (r, w)
    }

    /// Free slot count from the producer's view (Acquire on consumer's
    /// `read_idx`). Useful for slow-consumer-detection logic on producer side.
    pub fn free_slots(&self) -> u64 {
        let mask = self.header.mask.load(Ordering::Relaxed);
        let capacity = (mask as u64) + 1;
        let (r, w) = self.indices();
        capacity - w.wrapping_sub(r)
    }
}

/// Single-consumer side of the ring. Owns `read_idx` and slot reads.
///
/// Per PLAN §9.5: consumer does `Acquire` on `write_idx` (sees producer's
/// payload before reading the slot) and `Release` on `read_idx` (frees the
/// slot — happens-before producer's next `Acquire` on `read_idx`).
pub struct Consumer<'a> {
    pub(crate) header: &'a RingHeader,
    pub(crate) slots: *const u8,
    pub(crate) slots_len: usize,
}

// SAFETY: same rationale as Producer.
unsafe impl Send for Consumer<'_> {}

impl<'a> Consumer<'a> {
    /// Try to receive one slot. Returns `Some(payload)` if the queue has data;
    /// `None` if empty.
    ///
    /// Caller must call [`Consumer::commit`] AFTER reading the payload to free
    /// the slot. Forgetting to commit means subsequent `try_recv` returns the
    /// same slot — bounded but undesirable.
    pub fn try_recv(&mut self) -> Option<[u8; FASTBUS_SLOT_BYTES as usize]> {
        let mask = self.header.mask.load(Ordering::Relaxed);
        let read_idx = self.header.read_idx.load(Ordering::Relaxed);
        // Acquire on write_idx — synchronizes with producer's Release; payload
        // is fully written before we observe write_idx > read_idx.
        let write_idx = self.header.write_idx.load(Ordering::Acquire);
        if read_idx == write_idx {
            return None;
        }
        let slot_idx = (read_idx & mask as u64) as usize;
        let offset = slot_idx * (FASTBUS_SLOT_BYTES as usize);
        debug_assert!(
            offset + (FASTBUS_SLOT_BYTES as usize) <= self.slots_len,
            "fastbus slot offset out of bounds"
        );
        let mut payload = [0u8; FASTBUS_SLOT_BYTES as usize];
        // SAFETY: offset + 256 <= slots_len (debug-asserted); the slot is fully
        // written per Acquire ordering above; bytes are POD.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.slots.add(offset),
                payload.as_mut_ptr(),
                FASTBUS_SLOT_BYTES as usize,
            );
        }
        Some(payload)
    }

    /// Commit a single read by bumping `read_idx` (Release barrier — payload
    /// read happens-before producer sees the freed slot).
    ///
    /// Call after each successful `try_recv` that returned `Some`.
    pub fn commit(&mut self) {
        let read_idx = self.header.read_idx.load(Ordering::Relaxed);
        self.header.read_idx.store(read_idx + 1, Ordering::Release);
    }

    /// Try-recv-then-commit convenience: returns the payload AND advances
    /// `read_idx`. Caller doesn't need to call `commit` separately.
    pub fn recv_and_commit(&mut self) -> Option<[u8; FASTBUS_SLOT_BYTES as usize]> {
        let payload = self.try_recv()?;
        self.commit();
        Some(payload)
    }

    /// Used slot count from the consumer's view (Acquire on producer's
    /// `write_idx`).
    pub fn used_slots(&self) -> u64 {
        let r = self.header.read_idx.load(Ordering::Relaxed);
        let w = self.header.write_idx.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Returns the (read_idx, write_idx) pair — useful for diagnostics.
    pub fn indices(&self) -> (u64, u64) {
        let r = self.header.read_idx.load(Ordering::Relaxed);
        let w = self.header.write_idx.load(Ordering::Acquire);
        (r, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Ring;
    use crate::{FASTBUS_FILE_TOTAL_BYTES, FASTBUS_RING_CAPACITY_SLOTS};
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn make_ring() -> (tempfile::TempDir, Ring) {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("fastbus.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0u8; FASTBUS_FILE_TOTAL_BYTES]).unwrap();
        let ring = Ring::attach(&path).unwrap();
        (tmp, ring)
    }

    #[test]
    fn empty_ring_try_recv_returns_none() {
        let (_tmp, mut ring) = make_ring();
        let (_p, mut c) = ring.split();
        assert_eq!(c.try_recv(), None);
        assert_eq!(c.used_slots(), 0);
    }

    #[test]
    fn publish_then_recv_round_trip() {
        let (_tmp, mut ring) = make_ring();
        let (mut p, mut c) = ring.split();
        let payload = {
            let mut buf = [0u8; 256];
            buf[0] = 0x42;
            buf[7] = 0x99;
            buf[255] = 0xFF;
            buf
        };
        let idx = p.publish(&payload).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(c.used_slots(), 1);
        let recv = c.try_recv().unwrap();
        assert_eq!(&recv[..], &payload[..]);
        c.commit();
        assert_eq!(c.used_slots(), 0);
    }

    #[test]
    fn fifo_ordering_preserved_across_many_publishes() {
        let (_tmp, mut ring) = make_ring();
        let (mut p, mut c) = ring.split();
        for i in 0..100u64 {
            let mut payload = [0u8; 256];
            payload[..8].copy_from_slice(&i.to_le_bytes());
            p.publish(&payload).unwrap();
        }
        for i in 0..100u64 {
            let recv = c.recv_and_commit().unwrap();
            let got = u64::from_le_bytes(recv[..8].try_into().unwrap());
            assert_eq!(got, i, "FIFO violated at index {i}");
        }
        assert_eq!(c.used_slots(), 0);
    }

    #[test]
    fn queue_full_returns_error() {
        let (_tmp, mut ring) = make_ring();
        let (mut p, _c) = ring.split();
        let payload = [0u8; 256];
        // Fill exactly capacity slots
        for _ in 0..FASTBUS_RING_CAPACITY_SLOTS {
            p.publish(&payload).unwrap();
        }
        // Next publish must error
        let err = p.publish(&payload).unwrap_err();
        assert!(
            matches!(err, FastbusError::QueueFull { capacity: 1024 }),
            "got {err:?}"
        );
    }

    #[test]
    fn ring_drains_after_consume_then_accepts_new_publishes() {
        let (_tmp, mut ring) = make_ring();
        let (mut p, mut c) = ring.split();
        let payload = [0u8; 256];
        // Fill exactly capacity slots
        for _ in 0..FASTBUS_RING_CAPACITY_SLOTS {
            p.publish(&payload).unwrap();
        }
        assert!(p.publish(&payload).is_err());
        // Drain
        for _ in 0..FASTBUS_RING_CAPACITY_SLOTS {
            c.recv_and_commit().unwrap();
        }
        // Should accept again
        let idx = p.publish(&payload).unwrap();
        assert_eq!(idx, FASTBUS_RING_CAPACITY_SLOTS);
    }

    #[test]
    fn free_slots_and_used_slots_complement() {
        let (_tmp, mut ring) = make_ring();
        {
            let (mut p, c) = ring.split();
            assert_eq!(p.free_slots(), FASTBUS_RING_CAPACITY_SLOTS);
            assert_eq!(c.used_slots(), 0);
            let payload = [0u8; 256];
            p.publish(&payload).unwrap();
        } // borrows released
        let (p2, c2) = ring.split();
        assert_eq!(p2.free_slots(), FASTBUS_RING_CAPACITY_SLOTS - 1);
        assert_eq!(c2.used_slots(), 1);
    }

    #[test]
    fn cross_thread_spsc_round_trip() {
        // SPSC across thread boundary — proves Acquire/Release barriers work.
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::sync::Arc;
        use std::thread;

        let (_tmp, ring) = make_ring();
        // We'll use a separate ring instance per thread by re-mmap'ing the same file.
        let path = {
            let path = _tmp.path().join("fastbus.bin");
            assert!(path.exists());
            path
        };

        // Drop the owning ring so the second attach can succeed cleanly
        drop(ring);

        let path_clone = path.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();

        let producer_handle = thread::spawn(move || {
            let mut ring = Ring::attach(&path_clone).unwrap();
            let mut p = ring.producer_only();
            let mut sent = 0u64;
            while sent < 5000 {
                let mut payload = [0u8; 256];
                payload[..8].copy_from_slice(&sent.to_le_bytes());
                match p.publish(&payload) {
                    Ok(_) => sent += 1,
                    Err(_) => std::thread::yield_now(),
                }
            }
            stop_clone.store(true, O::Release);
            sent
        });

        let mut ring2 = Ring::attach(&path).unwrap();
        let mut c = ring2.consumer_only();
        let mut received = 0u64;
        while received < 5000 {
            match c.recv_and_commit() {
                Some(payload) => {
                    let idx = u64::from_le_bytes(payload[..8].try_into().unwrap());
                    assert_eq!(idx, received, "FIFO across threads violated");
                    received += 1;
                }
                None => std::thread::yield_now(),
            }
        }

        let sent = producer_handle.join().unwrap();
        assert_eq!(sent, 5000);
        assert_eq!(received, 5000);
        assert!(stop.load(O::Acquire));
    }

    #[test]
    fn try_recv_without_commit_re_reads_same_slot() {
        let (_tmp, mut ring) = make_ring();
        let (mut p, mut c) = ring.split();
        let payload = {
            let mut buf = [0u8; 256];
            buf[0] = 0xAB;
            buf
        };
        p.publish(&payload).unwrap();
        let r1 = c.try_recv().unwrap();
        let r2 = c.try_recv().unwrap();
        assert_eq!(
            r1, r2,
            "without commit, second try_recv must re-read same slot"
        );
        c.commit();
        assert_eq!(c.try_recv(), None);
    }
}
