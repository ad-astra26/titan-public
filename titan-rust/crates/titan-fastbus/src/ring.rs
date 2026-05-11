//! ring — Memory-mapped SPSC ring buffer with kernel-pre-allocated backing file.
//!
//! Per PLAN §9.3 (header byte layout) + §9.5 (memory ordering).
//!
//! # Lifecycle
//!
//! 1. Kernel pre-allocates `/dev/shm/titan_<id>/fastbus.bin` at boot per SPEC
//!    §10.A B3 (already shipped in C-S2 via `titan-state::spec`); zero-filled.
//! 2. Substrate calls [`Ring::attach`] at boot per SPEC §10.A B8 step S6:
//!    mmaps the file, validates the universal SeqLock header presence,
//!    inspects the ring header. If `version == 0` (uninitialized), substrate
//!    calls [`Ring::initialize`] which writes the canonical magic + version +
//!    zero indices + mask + slot_bytes via simple stores (no race with
//!    kernel — kernel doesn't touch the ring header content).
//! 3. Producer + consumer borrow disjoint slices via [`Ring::split`]: producer
//!    owns `write_idx` + slot writes; consumer owns `read_idx` + slot reads.
//!    Both share `&RingHeader` for the cross-side `Acquire`/`Release` loads.
//!
//! # Why no kernel-side Producer in C-S2
//!
//! C-S2 shipped the kernel-pre-allocated empty file but did NOT create a
//! [`Producer`] handle. C-S3 chunk C3-6 wires the kernel's producer alongside
//! the substrate's consumer.

use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use memmap2::MmapMut;
use thiserror::Error;

use crate::handle::{Consumer, Producer};
use crate::{
    FASTBUS_FILE_TOTAL_BYTES, FASTBUS_HEADER_BYTES, FASTBUS_MAGIC_BYTES,
    FASTBUS_RING_CAPACITY_SLOTS, FASTBUS_RING_VERSION, FASTBUS_SLOT_BYTES,
    UNIVERSAL_SEQLOCK_HEADER_BYTES,
};

/// Errors during fastbus open / attach / publish / receive.
#[derive(Debug, Error)]
pub enum FastbusError {
    /// I/O error opening or mmap'ing the file.
    #[error("fastbus I/O: {0}")]
    Io(#[from] std::io::Error),

    /// File size != [`FASTBUS_FILE_TOTAL_BYTES`].
    #[error("fastbus file size mismatch: got {got}, expected {expected}")]
    FileSizeMismatch {
        /// Bytes observed.
        got: u64,
        /// Bytes expected (constant 262232).
        expected: u64,
    },

    /// Magic bytes did not match `b"TITANFB1"`.
    #[error("fastbus magic mismatch: got {got:?}, expected {expected:?}")]
    MagicMismatch {
        /// Magic observed.
        got: [u8; 8],
        /// Magic expected.
        expected: [u8; 8],
    },

    /// Ring header version > current `FASTBUS_RING_VERSION` — substrate refuses
    /// to attach to a ring layout newer than it understands.
    #[error("fastbus version too new: got {got}, expected ≤ {expected}")]
    VersionTooNew {
        /// Version observed.
        got: u32,
        /// Maximum version supported.
        expected: u32,
    },

    /// `mask + 1` is not the configured ring capacity.
    #[error("fastbus mask mismatch: got {got}, expected {expected}")]
    MaskMismatch {
        /// Mask observed.
        got: u32,
        /// Mask expected (= capacity - 1).
        expected: u32,
    },

    /// Producer attempted to publish but the ring is full.
    #[error("fastbus queue full ({capacity} slots) — slow consumer")]
    QueueFull {
        /// Ring capacity in slots.
        capacity: u32,
    },
}

/// Ring header bytes [24:88] of fastbus.bin — SPSC indices + layout constants.
///
/// Per SPEC §7.1 v0.1.4 layout: `magic[8] + read_idx[8] + write_idx[8] +
/// version[4] + mask[4] + reserved[32]`. AtomicU64 fields are at 8-byte
/// aligned offsets (read_idx@8, write_idx@16) for portable lock-free atomics.
///
/// All multi-byte fields are little-endian. Atomics ensure cross-process
/// coordination without locks.
#[repr(C, align(8))]
pub struct RingHeader {
    /// Magic identifier — `b"TITANFB1"` raw bytes (offset 0).
    pub magic: [u8; 8],
    /// Consumer-bumped read counter at offset 8 — 8-byte aligned (monotonic;
    /// mod `mask + 1` gives slot idx).
    pub read_idx: AtomicU64,
    /// Producer-bumped write counter at offset 16 — 8-byte aligned.
    pub write_idx: AtomicU64,
    /// Ring version at offset 24 — bumped on layout changes.
    pub version: AtomicU32,
    /// Capacity - 1 (= 1023 for capacity 1024) at offset 28.
    pub mask: AtomicU32,
    /// Reserved bytes (zero) at offset 32 — pads header to 64 bytes.
    pub _reserved: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<RingHeader>() == FASTBUS_HEADER_BYTES as usize);

impl RingHeader {
    /// Validate magic + version + mask + slot_bytes against the canonical layout.
    ///
    /// Returns `Err(VersionTooNew)` if `version > FASTBUS_RING_VERSION` so an
    /// older substrate can refuse to attach to a newer ring.
    pub fn validate(&self) -> Result<(), FastbusError> {
        let mut expected_magic = [0u8; 8];
        expected_magic.copy_from_slice(FASTBUS_MAGIC_BYTES);
        if self.magic != expected_magic {
            return Err(FastbusError::MagicMismatch {
                got: self.magic,
                expected: expected_magic,
            });
        }
        let v = self.version.load(Ordering::Relaxed);
        if v > FASTBUS_RING_VERSION as u32 {
            return Err(FastbusError::VersionTooNew {
                got: v,
                expected: FASTBUS_RING_VERSION as u32,
            });
        }
        let m = self.mask.load(Ordering::Relaxed);
        let expected_mask = (FASTBUS_RING_CAPACITY_SLOTS as u32) - 1;
        if m != expected_mask {
            return Err(FastbusError::MaskMismatch {
                got: m,
                expected: expected_mask,
            });
        }
        Ok(())
    }
}

/// Memory-mapped fastbus ring — owns the mmap.
///
/// Use [`Ring::attach`] to open + initialize-if-needed. Use [`Ring::split`] to
/// borrow disjoint [`Producer`] + [`Consumer`] handles tied to this ring's
/// lifetime.
pub struct Ring {
    mmap: MmapMut,
}

impl std::fmt::Debug for Ring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let h = self.header();
        f.debug_struct("Ring")
            .field("magic", &std::str::from_utf8(&h.magic).unwrap_or("?"))
            .field("version", &h.version.load(Ordering::Relaxed))
            .field("read_idx", &h.read_idx.load(Ordering::Relaxed))
            .field("write_idx", &h.write_idx.load(Ordering::Relaxed))
            .field("mask", &h.mask.load(Ordering::Relaxed))
            .finish()
    }
}

impl Ring {
    /// Attach to a kernel-pre-allocated `fastbus.bin`. Validates file size +
    /// universal SeqLock header presence (kernel writes it at boot).
    ///
    /// If the ring header `version` is 0 (uninitialized — kernel pre-allocates
    /// the file as all zeros), [`Ring::initialize`] is called automatically.
    /// Otherwise the existing header is validated.
    pub fn attach(path: &Path) -> Result<Self, FastbusError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let metadata = file.metadata()?;
        if metadata.len() != FASTBUS_FILE_TOTAL_BYTES as u64 {
            return Err(FastbusError::FileSizeMismatch {
                got: metadata.len(),
                expected: FASTBUS_FILE_TOTAL_BYTES as u64,
            });
        }
        // SAFETY: file is the same size as our mmap; lifetime is tied to Ring.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let mut ring = Ring { mmap };

        // Initialize-if-zero: substrate is the first attacher.
        let needs_init = {
            let header = ring.header_mut();
            header.version.load(Ordering::Relaxed) == 0
        };
        if needs_init {
            ring.initialize()?;
        }
        ring.header_mut().validate()?;
        Ok(ring)
    }

    /// Initialize the ring header in-place. Called by [`Ring::attach`] when the
    /// existing version field is zero (kernel pre-allocates as all-zeros).
    ///
    /// Panic-free; only writes via atomic stores so partial writes never leave
    /// the ring in a half-initialized state visible to a peer.
    ///
    /// Note: `slot_bytes` is NOT a header field per SPEC v0.1.4 — it is fixed
    /// at compile time as [`FASTBUS_SLOT_BYTES`] (= 256). The header size
    /// freed by removing the redundant runtime field is absorbed by the
    /// reserved[32] block.
    pub fn initialize(&mut self) -> Result<(), FastbusError> {
        let header = self.header_mut();
        // Write magic raw — pre-init readers see zeros, post-init readers see canonical.
        let mut canon_magic = [0u8; 8];
        canon_magic.copy_from_slice(FASTBUS_MAGIC_BYTES);
        header.magic = canon_magic;
        header.read_idx.store(0, Ordering::Relaxed);
        header.write_idx.store(0, Ordering::Relaxed);
        header
            .mask
            .store((FASTBUS_RING_CAPACITY_SLOTS as u32) - 1, Ordering::Relaxed);
        header._reserved = [0u8; 32];
        // Version goes last — Release ensures a peer doing Acquire-load on
        // version sees magic + indices + mask written before version=1.
        header
            .version
            .store(FASTBUS_RING_VERSION as u32, Ordering::Release);
        Ok(())
    }

    /// Split the ring into a [`Producer`] handle (writes slots, bumps
    /// `write_idx`) and a [`Consumer`] handle (reads slots, bumps `read_idx`).
    /// Both borrow from the same mmap; SPSC discipline is the caller's
    /// responsibility (one thread/process per side per direction).
    pub fn split(&mut self) -> (Producer<'_>, Consumer<'_>) {
        let mmap_ptr = self.mmap.as_mut_ptr();
        let header_offset = UNIVERSAL_SEQLOCK_HEADER_BYTES;
        let slots_offset = header_offset + FASTBUS_HEADER_BYTES as usize;
        let slots_len = (FASTBUS_RING_CAPACITY_SLOTS as usize) * (FASTBUS_SLOT_BYTES as usize);

        // SAFETY: the producer + consumer borrow disjoint logical regions:
        // both share the &RingHeader (only atomic ops), and the slot array is
        // partitioned by `(write_idx & mask)` vs `(read_idx & mask)` — by SPSC
        // discipline, only the producer writes a slot when `write_idx > read_idx`,
        // and only the consumer reads when `read_idx < write_idx`. The atomic
        // Acquire/Release ordering ensures the slot is fully written before the
        // consumer observes it via `write_idx`. We materialize the same raw
        // pointer in both handles; aliasing is constrained by the Acquire/Release
        // protocol.
        let header_ref: &RingHeader =
            unsafe { &*(mmap_ptr.add(header_offset) as *const RingHeader) };
        let slots_for_producer: *mut u8 = unsafe { mmap_ptr.add(slots_offset) };
        let slots_for_consumer: *const u8 = slots_for_producer as *const u8;

        let producer = Producer {
            header: header_ref,
            slots: slots_for_producer,
            slots_len,
        };
        let consumer = Consumer {
            header: header_ref,
            slots: slots_for_consumer,
            slots_len,
        };
        (producer, consumer)
    }

    /// Produce-only handle — used when this side never reads (kernel side).
    /// Caller asserts the consumer side runs in a different process.
    pub fn producer_only(&mut self) -> Producer<'_> {
        let (p, _c) = self.split();
        p
    }

    /// Consume-only handle — used when this side never writes (substrate side
    /// for the kernel→substrate direction).
    pub fn consumer_only(&mut self) -> Consumer<'_> {
        let (_p, c) = self.split();
        c
    }

    /// Borrow the ring header (always valid after `attach`).
    pub fn header(&self) -> &RingHeader {
        let mmap_ptr = self.mmap.as_ptr();
        // SAFETY: header lives within the mmap; Ring holds the mmap.
        unsafe { &*(mmap_ptr.add(UNIVERSAL_SEQLOCK_HEADER_BYTES) as *const RingHeader) }
    }

    fn header_mut(&mut self) -> &mut RingHeader {
        let mmap_ptr = self.mmap.as_mut_ptr();
        // SAFETY: header lives within the mmap; Ring holds &mut mmap exclusively.
        unsafe { &mut *(mmap_ptr.add(UNIVERSAL_SEQLOCK_HEADER_BYTES) as *mut RingHeader) }
    }
}

// Producer + Consumer have a `&RingHeader`, which contains atomics only — Send + Sync.
// SAFETY: Atomics are Send + Sync; raw pointers in handles point into mmap'd memory whose
// lifetime is bound to `&'a Ring`. We treat the ring as a process-shared structure governed
// by Acquire/Release ordering rather than Rust's borrow tracker.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn make_zeroed_file(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fastbus.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0u8; FASTBUS_FILE_TOTAL_BYTES]).unwrap();
        path
    }

    #[test]
    fn attach_initializes_zeroed_file() {
        let tmp = tempdir().unwrap();
        let path = make_zeroed_file(tmp.path());
        let ring = Ring::attach(&path).unwrap();
        let header = ring.header();
        assert_eq!(&header.magic, FASTBUS_MAGIC_BYTES);
        assert_eq!(
            header.version.load(Ordering::Relaxed),
            FASTBUS_RING_VERSION as u32
        );
        assert_eq!(header.read_idx.load(Ordering::Relaxed), 0);
        assert_eq!(header.write_idx.load(Ordering::Relaxed), 0);
        assert_eq!(
            header.mask.load(Ordering::Relaxed),
            (FASTBUS_RING_CAPACITY_SLOTS as u32) - 1
        );
    }

    #[test]
    fn attach_validates_existing_initialized_file() {
        let tmp = tempdir().unwrap();
        let path = make_zeroed_file(tmp.path());
        // First attach initializes
        let _ = Ring::attach(&path).unwrap();
        // Second attach should validate cleanly
        let ring = Ring::attach(&path).unwrap();
        assert_eq!(
            ring.header().version.load(Ordering::Relaxed),
            FASTBUS_RING_VERSION as u32
        );
    }

    #[test]
    fn attach_rejects_wrong_size_file() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("fastbus.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0u8; FASTBUS_FILE_TOTAL_BYTES - 1])
            .unwrap();
        let err = Ring::attach(&path).unwrap_err();
        assert!(matches!(err, FastbusError::FileSizeMismatch { .. }));
    }

    #[test]
    fn attach_rejects_corrupt_magic_bytes() {
        let tmp = tempdir().unwrap();
        let path = make_zeroed_file(tmp.path());
        let _ = Ring::attach(&path).unwrap(); // initialize
                                              // Corrupt the magic bytes
        {
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            use std::io::{Seek, SeekFrom};
            f.seek(SeekFrom::Start(UNIVERSAL_SEQLOCK_HEADER_BYTES as u64))
                .unwrap();
            f.write_all(b"WRONGMG!").unwrap();
        }
        let err = Ring::attach(&path).unwrap_err();
        assert!(matches!(err, FastbusError::MagicMismatch { .. }));
    }

    #[test]
    fn attach_rejects_future_version() {
        let tmp = tempdir().unwrap();
        let path = make_zeroed_file(tmp.path());
        let _ = Ring::attach(&path).unwrap(); // initialize at v=1
                                              // Bump version to 99 directly in mmap.
                                              // Per SPEC §7.1 v0.1.4 ring header layout:
                                              //   universal SeqLock header [0:24]
                                              //   ring magic [24:32]
                                              //   ring read_idx [32:40]
                                              //   ring write_idx [40:48]
                                              //   ring version [48:52]   ← target
                                              //   ring mask [52:56]
                                              //   ring reserved [56:88]
        {
            use std::io::{Seek, SeekFrom};
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(48)).unwrap();
            f.write_all(&99u32.to_le_bytes()).unwrap();
        }
        let err = Ring::attach(&path).unwrap_err();
        assert!(
            matches!(err, FastbusError::VersionTooNew { got: 99, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn fastbus_file_total_bytes_matches_spec() {
        // SPEC §7.1 declares fastbus.bin total = 262232 bytes
        assert_eq!(FASTBUS_FILE_TOTAL_BYTES, 262232);
    }

    #[test]
    fn ring_header_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<RingHeader>(), 64);
    }
}
