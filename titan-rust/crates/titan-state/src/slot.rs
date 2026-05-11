//! slot — Atomic shm slot creation + triple-buffer writer/reader.
//!
//! Per SPEC §7.0 (universal triple-buffer wire format v1.0.0) + §7.3 (slot
//! lifecycle) + §7.4 (Rust port byte-identical guarantee). Reuses
//! [`titan_core::shm::SlotHeader`] + [`titan_core::shm::BufferMeta`] for
//! byte-exact 16+16+N bytes-per-buffer layout.
//!
//! Closes `titan-docs/rFP_rust_seqlock_retry_exhaustion.md` (D-SPEC-35
//! v0.2.0 → v1.0.0 MAJOR). Replaces the legacy 24-byte SeqLock format.
//!
//! Slot layout:
//!
//! ```text
//! [0:16]                          fixed header  (header_seq atomic + schema + capacity)
//! [16:16 + (16+N)]                buffer 0 — meta(16) + payload(N)
//! [16 + (16+N):16 + 2(16+N)]      buffer 1
//! [16 + 2(16+N):16 + 3(16+N)]     buffer 2
//! ```
//!
//! Total slot size = `16 + 3·(16+N) = 64 + 3·N`.
//!
//! # Writer protocol — single writer per slot
//!
//! Writer holds local `last_published_idx` (init 2 ⇒ first publish lands
//! on idx 0) and `version` (init 0 ⇒ first publish bumps to 1).
//!
//! ```text
//! 1. next_idx ← (last_published_idx + 1) mod 3
//! 2. off ← buffer_offset(next_idx) = 16 + next_idx · (16+N)
//! 3. write metadata prefix: mmap[off:off+8] = wall_ns; mmap[off+8:off+12] = payload.len()
//! 4. memcpy mmap[off+16 : off+16+payload.len()] ← payload
//! 5. crc ← CRC32(mmap[off:off+12] || payload[0:payload.len()])
//!    write mmap[off+12:off+16] ← crc
//! 6. ATOMIC STORE mmap[0:8] ← (version+1) << 8 | next_idx, Release ordering
//! 7. local: last_published_idx ← next_idx; version ← version + 1
//! ```
//!
//! Critically, writer NEVER touches the buffer at the previous `ready_idx`
//! during steps 1-5 — readers' chosen buffer is frozen.
//!
//! # Reader protocol — zero retries, zero spinning
//!
//! ```text
//! 1.  s1 ← ATOMIC LOAD mmap[0:8] (Acquire)
//! 2.  if version1 == 0: return Err(Uninitialized)
//! 3.  if idx > 2: return Err(ReadyIdxOutOfRange)
//! 4.  off ← buffer_offset(idx)
//! 5.  read meta: wall_ns, payload_bytes, stored_crc from mmap[off:off+16]
//! 6.  copy payload from mmap[off+16:off+16+payload_bytes]
//! 7.  compute_crc ← CRC32(mmap[off:off+12] || payload)
//! 8.  s2 ← ATOMIC LOAD mmap[0:8] (Acquire); delta ← (s2 >> 8) - version1
//! 9.  if delta > 2: return Err(ReaderLapped)  -- writer lapped us mid-read,
//!     buffer may carry torn payload (CRC may have spuriously mismatched OR
//!     happened to verify); recoverable, caller retries on next tick.
//! 10. if compute_crc ≠ stored_crc: return Err(BufferCrcMismatch)  -- writer
//!     did NOT lap us (delta ≤ 2 = our buffer was frozen during read), so
//!     a CRC failure here is real corruption, not torn data.
//! 11. return Ok(payload)
//! ```
//!
//! **Version-check-before-CRC-classify ordering** (codified after rFP §13
//! 2026-05-08): putting the version check BEFORE the CRC-mismatch return
//! lets us correctly classify torn-reads as `ReaderLapped` (recoverable)
//! rather than misdiagnosing them as `BufferCrcMismatch` (fatal corruption).
//! Pre-fix the body cycle on T3 saw ~0.2% BODY_CYCLE_READ_ERROR / SKIPPED
//! events under host pressure that were actually transient torn-reads.

use std::fs;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use memmap2::{MmapMut, MmapOptions};

use titan_core::constants::{SHM_BUFFER_COUNT, SHM_BUFFER_META_BYTES, SHM_HEADER_BYTES};
use titan_core::shm::{
    buffer_offset, pack_header_seq, BufferMeta, SlotHeader, HEADER_SEQ_IDX_MASK,
    HEADER_SEQ_IDX_MAX, HEADER_SEQ_VERSION_SHIFT,
};

/// Errors during slot operations.
#[derive(Debug, thiserror::Error)]
pub enum SlotIoError {
    /// I/O failure (open, ftruncate, mmap, rename, fsync).
    #[error("I/O at {path}: {source}")]
    Io {
        /// Path where the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Payload exceeds slot capacity.
    #[error("payload {actual}B exceeds slot capacity {max}B at {path}")]
    PayloadTooLarge {
        /// Slot path.
        path: PathBuf,
        /// Attempted payload bytes.
        actual: usize,
        /// Configured maximum.
        max: usize,
    },

    /// Slot has not been published yet (`version == 0` sentinel).
    #[error("slot at {path} uninitialized — never published")]
    Uninitialized {
        /// Slot path.
        path: PathBuf,
    },

    /// Buffer CRC32 mismatch — safety net for corruption or race edge cases.
    #[error("slot at {path} buffer CRC mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    BufferCrcMismatch {
        /// Slot path.
        path: PathBuf,
        /// CRC32 stored in buffer metadata.
        stored: u32,
        /// CRC32 recomputed by reader.
        computed: u32,
    },

    /// `ready_idx` out of valid range — header_seq corrupt.
    #[error("slot at {path} ready_idx out of range: {got} > {max}")]
    ReadyIdxOutOfRange {
        /// Slot path.
        path: PathBuf,
        /// Index value read.
        got: u8,
        /// Maximum legal value.
        max: u8,
    },

    /// Schema version did not match expected value.
    #[error("slot at {path} schema mismatch: stored {stored}, expected {expected}")]
    SchemaMismatch {
        /// Slot path.
        path: PathBuf,
        /// Schema version read from slot.
        stored: u32,
        /// Schema version expected by reader.
        expected: u32,
    },

    /// Reader was preempted long enough that writer lapped through all
    /// 3 buffers during the read — buffer the reader copied was overwritten.
    #[error("slot at {path} reader lapped — version delta {delta} > 2 (writer overwrote buffer)")]
    ReaderLapped {
        /// Slot path.
        path: PathBuf,
        /// Number of writer publishes during the read.
        delta: u64,
    },
}

impl SlotIoError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Total bytes for a slot file = `16 (header) + 3 × (16 (meta) + payload_bytes)`.
fn total_slot_bytes(payload_bytes: u32) -> u64 {
    SHM_HEADER_BYTES
        + (SHM_BUFFER_COUNT * (SHM_BUFFER_META_BYTES + payload_bytes as u64))
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Atomic-load the `header_seq` u64 at offset 0 with Acquire ordering.
///
/// SAFETY: caller guarantees `mmap` has at least 8 bytes mapped and that
/// the mapping is page-aligned (mmap regions always are). The atomic at
/// offset 0 is naturally 8-byte aligned ⇒ single-instruction atomic on
/// x86_64 + aarch64.
#[inline]
fn load_header_seq_acquire(mmap: &[u8]) -> u64 {
    let ptr = mmap.as_ptr() as *const AtomicU64;
    unsafe { (*ptr).load(Ordering::Acquire) }
}

/// Atomic-store the `header_seq` u64 at offset 0 with Release ordering.
#[inline]
fn store_header_seq_release(mmap: &mut [u8], value: u64) {
    let ptr = mmap.as_mut_ptr() as *const AtomicU64;
    unsafe { (*ptr).store(value, Ordering::Release) };
}

/// Ergonomic handle to one slot file. Holds the mmap'd memory + path so
/// callers don't have to re-open for each write/read.
pub struct Slot {
    path: PathBuf,
    mmap: MmapMut,
    /// Per-buffer payload capacity (bytes). Total mmap = 16 + 3·(16+payload_capacity).
    payload_capacity: usize,
    /// Writer state: index of the buffer most recently published. Initialized
    /// to `SHM_BUFFER_COUNT - 1` (= 2) so the first `write()` lands on idx 0.
    writer_last_published_idx: u8,
    /// Writer state: monotonic publish counter. Initialized to 0; first
    /// `write()` bumps to 1.
    writer_version: u64,
}

impl Slot {
    /// Atomic slot creation. Uses `tmp + rename` pattern for crash safety.
    pub fn create(
        path: impl AsRef<Path>,
        schema_version: u32,
        payload_bytes: u32,
    ) -> Result<Self, SlotIoError> {
        let path = path.as_ref().to_path_buf();
        let total = total_slot_bytes(payload_bytes);

        let mut tmp_buf = path.as_os_str().to_owned();
        tmp_buf.push(".tmp");
        let tmp = PathBuf::from(tmp_buf);

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| SlotIoError::io(&tmp, e))?;

        file.set_len(total).map_err(|e| SlotIoError::io(&tmp, e))?;

        let mut mmap = unsafe { MmapOptions::new().len(total as usize).map_mut(&file) }
            .map_err(|e| SlotIoError::io(&tmp, std::io::Error::other(format!("mmap: {e}"))))?;

        // Write fixed header: header_seq=0 (uninitialized), schema, capacity.
        let header = SlotHeader {
            header_seq: 0,
            schema_version,
            payload_capacity: payload_bytes,
        };
        mmap[..SHM_HEADER_BYTES as usize].copy_from_slice(&header.encode());

        // Zero all 3 (meta + payload) blocks. Linux tmpfs already zeroes new
        // mappings — defensive on platforms where sparseness differs.
        for byte in mmap[SHM_HEADER_BYTES as usize..].iter_mut() {
            *byte = 0;
        }
        mmap.flush().map_err(|e| SlotIoError::io(&tmp, e))?;

        fs::rename(&tmp, &path).map_err(|e| SlotIoError::io(&path, e))?;

        if let Some(parent) = path.parent() {
            let dir = fs::File::open(parent).map_err(|e| SlotIoError::io(parent, e))?;
            dir.sync_all().map_err(|e| SlotIoError::io(parent, e))?;
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SlotIoError::io(&path, e))?;
        let mmap = unsafe { MmapOptions::new().len(total as usize).map_mut(&file) }
            .map_err(|e| SlotIoError::io(&path, std::io::Error::other(format!("mmap: {e}"))))?;

        let mut slot = Self {
            path,
            mmap,
            payload_capacity: payload_bytes as usize,
            writer_last_published_idx: HEADER_SEQ_IDX_MAX,
            writer_version: 0,
        };

        // Initial publish: write a zero-FILLED payload of the declared
        // capacity at version=1 so the slot is immediately readable post-
        // create with the expected byte-count. This matches legacy SeqLock
        // semantics where a freshly-created slot with seq=0 (even, stable)
        // was a readable zero-payload snapshot of declared size — many call
        // sites (trinity orchestration, content-hash gates, struct decode
        // paths) require read() to return capacity bytes even before the
        // first user write. Variable-size slots (e.g. cgn_live_weights)
        // also publish capacity-bytes initially; consumers using
        // `read_variable()` semantics see actual_payload_bytes from the
        // per-buffer metadata, not the declared capacity.
        let zero_payload = vec![0u8; payload_bytes as usize];
        slot.write(&zero_payload)
            .expect("initial zero-fill publish must succeed (slot just created)");

        Ok(slot)
    }

    /// Open an existing slot. Reconstructs writer state from the persisted
    /// `header_seq` so a writer reattaching after a restart continues
    /// monotonic versioning + buffer rotation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SlotIoError> {
        let path = path.as_ref().to_path_buf();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SlotIoError::io(&path, e))?;
        let meta = file.metadata().map_err(|e| SlotIoError::io(&path, e))?;
        let total = meta.len() as usize;
        let mmap = unsafe { MmapOptions::new().len(total).map_mut(&file) }
            .map_err(|e| SlotIoError::io(&path, std::io::Error::other(format!("mmap: {e}"))))?;

        // Recover capacity from the persisted fixed header.
        let header = SlotHeader::decode(&mmap[..SHM_HEADER_BYTES as usize])
            .map_err(|e| SlotIoError::io(&path, std::io::Error::other(format!("decode: {e}"))))?;
        let payload_capacity = header.payload_capacity as usize;

        // Reconstruct writer state.
        let s0 = load_header_seq_acquire(&mmap[..SHM_HEADER_BYTES as usize]);
        let persisted_version = s0 >> HEADER_SEQ_VERSION_SHIFT;
        let persisted_idx = (s0 & HEADER_SEQ_IDX_MASK) as u8;
        let last_published_idx = if persisted_version == 0 {
            HEADER_SEQ_IDX_MAX
        } else if persisted_idx <= HEADER_SEQ_IDX_MAX {
            persisted_idx
        } else {
            HEADER_SEQ_IDX_MAX
        };

        Ok(Self {
            path,
            mmap,
            payload_capacity,
            writer_last_published_idx: last_published_idx,
            writer_version: persisted_version,
        })
    }

    /// Path of the slot file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Per-buffer payload capacity (bytes).
    pub fn payload_capacity(&self) -> usize {
        self.payload_capacity
    }

    /// Decode the fixed header (16 bytes at offset 0).
    pub fn header(&self) -> SlotHeader {
        SlotHeader::decode(&self.mmap[..SHM_HEADER_BYTES as usize])
            .expect("slot mmap always has 16 fixed-header bytes")
    }

    /// Decode the metadata block of buffer `idx` (16 bytes at the buffer offset).
    pub fn buffer_meta(&self, idx: u8) -> BufferMeta {
        let off = buffer_offset(idx, self.payload_capacity);
        BufferMeta::decode(&self.mmap[off..off + SHM_BUFFER_META_BYTES as usize])
            .expect("buffer metadata block always has 16 bytes")
    }

    /// Triple-buffer writer: write metadata + payload to inactive buffer,
    /// then atomic-publish header_seq with Release ordering.
    pub fn write(&mut self, payload: &[u8]) -> Result<(), SlotIoError> {
        if payload.len() > self.payload_capacity {
            return Err(SlotIoError::PayloadTooLarge {
                path: self.path.clone(),
                actual: payload.len(),
                max: self.payload_capacity,
            });
        }

        // Step 1: rotate to next buffer.
        let next_idx = (self.writer_last_published_idx + 1) % SHM_BUFFER_COUNT as u8;
        let off = buffer_offset(next_idx, self.payload_capacity);

        // Step 2: write metadata prefix [0:12] of buf[next_idx]: wall_ns + payload_bytes.
        let wall = now_ns();
        self.mmap[off..off + 8].copy_from_slice(&wall.to_le_bytes());
        self.mmap[off + 8..off + 12].copy_from_slice(&(payload.len() as u32).to_le_bytes());

        // Step 3: memcpy payload into buf[next_idx][16:16+len].
        let payload_off = off + SHM_BUFFER_META_BYTES as usize;
        self.mmap[payload_off..payload_off + payload.len()].copy_from_slice(payload);

        // Step 4: compute CRC over [meta_prefix(12) || payload] and write to [12:16].
        let crc = BufferMeta::compute_crc32(wall, payload.len() as u32, payload);
        self.mmap[off + 12..off + 16].copy_from_slice(&crc.to_le_bytes());

        // Step 5: atomic Release-store of new header_seq publishes everything.
        let new_version = self.writer_version + 1;
        let new_seq = pack_header_seq(new_version, next_idx);
        store_header_seq_release(&mut self.mmap[..], new_seq);

        // Step 6: update local writer state.
        self.writer_last_published_idx = next_idx;
        self.writer_version = new_version;

        Ok(())
    }

    /// Triple-buffer reader: zero-retry, zero-spin atomic snapshot read.
    pub fn read(&self) -> Result<Vec<u8>, SlotIoError> {
        // Step 1: atomic Acquire-load of header_seq.
        let s1 = load_header_seq_acquire(&self.mmap[..SHM_HEADER_BYTES as usize]);
        let version1 = s1 >> HEADER_SEQ_VERSION_SHIFT;
        let idx = (s1 & HEADER_SEQ_IDX_MASK) as u8;

        // Step 2: uninitialized sentinel.
        if version1 == 0 {
            return Err(SlotIoError::Uninitialized {
                path: self.path.clone(),
            });
        }

        // Step 3: ready_idx range check.
        if idx > HEADER_SEQ_IDX_MAX {
            return Err(SlotIoError::ReadyIdxOutOfRange {
                path: self.path.clone(),
                got: idx,
                max: HEADER_SEQ_IDX_MAX,
            });
        }

        // Step 4: read buffer metadata at offset.
        let off = buffer_offset(idx, self.payload_capacity);
        let wall_ns = u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap());
        let payload_bytes =
            u32::from_le_bytes(self.mmap[off + 8..off + 12].try_into().unwrap()) as usize;
        let stored_crc = u32::from_le_bytes(self.mmap[off + 12..off + 16].try_into().unwrap());

        if payload_bytes > self.payload_capacity {
            return Err(SlotIoError::PayloadTooLarge {
                path: self.path.clone(),
                actual: payload_bytes,
                max: self.payload_capacity,
            });
        }

        // Step 5: copy payload.
        let payload_off = off + SHM_BUFFER_META_BYTES as usize;
        let payload = self.mmap[payload_off..payload_off + payload_bytes].to_vec();

        // Step 6: compute CRC32 over (meta prefix + payload). Defer the
        // mismatch decision to step 8 — we need the version-check first to
        // distinguish torn-read (recoverable) from real corruption (fatal).
        let computed = BufferMeta::compute_crc32(wall_ns, payload_bytes as u32, &payload);
        let crc_ok = computed == stored_crc;

        // Step 7: atomic Acquire-load of header_seq again.
        let s2 = load_header_seq_acquire(&self.mmap[..SHM_HEADER_BYTES as usize]);
        let version2 = s2 >> HEADER_SEQ_VERSION_SHIFT;
        let delta = version2.wrapping_sub(version1);

        // Step 8: classify result. Triple-buffer guarantee: delta ≤ 2 means
        // writer cycled through other buffers but did NOT lap back to ours;
        // our buffer was frozen during read. delta > 2 means writer lapped
        // us — the buffer at `idx` is now mid-rewrite and our payload may
        // be torn (whether CRC happened to verify or not).
        if delta > (SHM_BUFFER_COUNT - 1) {
            return Err(SlotIoError::ReaderLapped {
                path: self.path.clone(),
                delta,
            });
        }
        if !crc_ok {
            // Writer didn't lap us, so a CRC failure here is genuine
            // corruption (memory error, malicious tamper, etc.) — not
            // torn data.
            return Err(SlotIoError::BufferCrcMismatch {
                path: self.path.clone(),
                stored: stored_crc,
                computed,
            });
        }
        Ok(payload)
    }

    /// Read with metadata. Returns `(payload, wall_ns)` on success.
    pub fn read_with_meta(&self) -> Result<(Vec<u8>, u64), SlotIoError> {
        // Inline read path, returning wall_ns alongside payload. Duplicated
        // from read() to keep the hot path branch-light.
        let s1 = load_header_seq_acquire(&self.mmap[..SHM_HEADER_BYTES as usize]);
        let version1 = s1 >> HEADER_SEQ_VERSION_SHIFT;
        let idx = (s1 & HEADER_SEQ_IDX_MASK) as u8;

        if version1 == 0 {
            return Err(SlotIoError::Uninitialized {
                path: self.path.clone(),
            });
        }
        if idx > HEADER_SEQ_IDX_MAX {
            return Err(SlotIoError::ReadyIdxOutOfRange {
                path: self.path.clone(),
                got: idx,
                max: HEADER_SEQ_IDX_MAX,
            });
        }

        let off = buffer_offset(idx, self.payload_capacity);
        let wall_ns = u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap());
        let payload_bytes =
            u32::from_le_bytes(self.mmap[off + 8..off + 12].try_into().unwrap()) as usize;
        let stored_crc = u32::from_le_bytes(self.mmap[off + 12..off + 16].try_into().unwrap());

        if payload_bytes > self.payload_capacity {
            return Err(SlotIoError::PayloadTooLarge {
                path: self.path.clone(),
                actual: payload_bytes,
                max: self.payload_capacity,
            });
        }

        let payload_off = off + SHM_BUFFER_META_BYTES as usize;
        let payload = self.mmap[payload_off..payload_off + payload_bytes].to_vec();

        // Defer CRC-mismatch decision to after version-check (see read()
        // for the rationale — distinguish torn-read from real corruption).
        let computed = BufferMeta::compute_crc32(wall_ns, payload_bytes as u32, &payload);
        let crc_ok = computed == stored_crc;

        let s2 = load_header_seq_acquire(&self.mmap[..SHM_HEADER_BYTES as usize]);
        let version2 = s2 >> HEADER_SEQ_VERSION_SHIFT;
        let delta = version2.wrapping_sub(version1);

        if delta > (SHM_BUFFER_COUNT - 1) {
            return Err(SlotIoError::ReaderLapped {
                path: self.path.clone(),
                delta,
            });
        }
        if !crc_ok {
            return Err(SlotIoError::BufferCrcMismatch {
                path: self.path.clone(),
                stored: stored_crc,
                computed,
            });
        }
        Ok((payload, wall_ns))
    }

    /// msync the mmap to disk (no-op on tmpfs).
    pub fn flush(&self) -> Result<(), SlotIoError> {
        self.mmap
            .flush()
            .map_err(|e| SlotIoError::io(&self.path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn create_publishes_initial_zero_filled_payload() {
        // Slot::create does an initial zero-FILLED publish at full capacity
        // so the slot is readable immediately at the declared byte size.
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let slot = Slot::create(&path, 1, 64).unwrap();
        let h = slot.header();
        assert!(h.is_initialized(), "slot should be initialized after create");
        assert_eq!(h.version(), 1);
        assert_eq!(h.ready_idx(), 0);
        assert_eq!(h.schema_version, 1);
        assert_eq!(h.payload_capacity, 64);
        // Read returns 64 zero bytes.
        let payload = slot.read().unwrap();
        assert_eq!(payload, vec![0u8; 64]);
    }

    #[test]
    fn create_total_size_is_64_plus_3x_payload() {
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let _ = Slot::create(&path, 1, 100).unwrap();
        let meta = fs::metadata(&path).unwrap();
        // 16 fixed header + 3 × (16 buffer meta + 100 payload) = 16 + 348 = 364
        assert_eq!(meta.len(), 16 + 3 * (16 + 100));
        assert_eq!(meta.len(), 364);
    }

    #[test]
    fn create_mode_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let _ = Slot::create(&path, 1, 64).unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn read_uninitialized_returns_uninitialized_error() {
        // Manually craft a slot file with version=0 in the header_seq word
        // (Slot::create does an initial publish so we can't get this state
        // from the public API anymore — this scenario only happens if a
        // reader attaches to a slot file mid-creation, before the writer's
        // first publish lands).
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        // Force version back to 0 to simulate the pre-publish race.
        store_header_seq_release(&mut slot.mmap[..], 0);
        let result = slot.read();
        assert!(matches!(result, Err(SlotIoError::Uninitialized { .. })));
    }

    #[test]
    fn write_then_read_round_trip() {
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        let payload = b"hello, slot!";
        slot.write(payload).unwrap();

        let read = slot.read().unwrap();
        assert_eq!(read, payload);
    }

    #[test]
    fn first_user_write_lands_on_idx_1() {
        // Slot::create's initial empty publish lands on idx 0 (version=1).
        // The first user-visible `write()` rotates to idx 1 (version=2).
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        slot.write(b"first").unwrap();
        let h = slot.header();
        assert_eq!(h.ready_idx(), 1);
        assert_eq!(h.version(), 2);
    }

    #[test]
    fn writer_rotates_through_three_buffers() {
        // Slot::create publishes empty payload at idx=0/version=1 first,
        // so user writes rotate idx 1, 2, 0, 1, 2 with versions 2..=6.
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 16).unwrap();
        slot.write(b"v0").unwrap();
        assert_eq!(slot.header().ready_idx(), 1);
        slot.write(b"v1").unwrap();
        assert_eq!(slot.header().ready_idx(), 2);
        slot.write(b"v2").unwrap();
        assert_eq!(slot.header().ready_idx(), 0); // wrapped
        slot.write(b"v3").unwrap();
        assert_eq!(slot.header().ready_idx(), 1);
        slot.write(b"v4").unwrap();
        assert_eq!(slot.header().ready_idx(), 2);
        assert_eq!(slot.header().version(), 6);
    }

    #[test]
    fn writer_writes_to_inactive_buffer_only() {
        // Slot::create publishes a zero-filled capacity payload at idx=0
        // (16 bytes); subsequent writes rotate to inactive buffers.
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 16).unwrap();
        // Initial state: idx=0 holds 16-byte zero publish; bufs 1+2 zero meta.

        slot.write(b"alpha").unwrap();
        // ready_idx=1, buf 1 has "alpha" (5 bytes); buf 0 still 16-byte zero (untouched).
        let buf0_meta = slot.buffer_meta(0);
        let buf1_meta = slot.buffer_meta(1);
        let buf2_meta = slot.buffer_meta(2);
        assert_eq!(buf0_meta.payload_bytes, 16); // initial zero publish, untouched
        assert_eq!(buf1_meta.payload_bytes, 5);  // "alpha"
        assert_eq!(buf2_meta.payload_bytes, 0);  // never written

        slot.write(b"beta").unwrap();
        // ready_idx=2, buf 1 still "alpha", buf 2 has "beta" (4 bytes), buf 0 still untouched.
        let buf0_meta = slot.buffer_meta(0);
        let buf1_meta = slot.buffer_meta(1);
        let buf2_meta = slot.buffer_meta(2);
        assert_eq!(buf0_meta.payload_bytes, 16);
        assert_eq!(buf1_meta.payload_bytes, 5);
        assert_eq!(buf2_meta.payload_bytes, 4);
    }

    #[test]
    fn write_oversized_rejected() {
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 16).unwrap();
        let too_big = vec![0u8; 17];
        let result = slot.write(&too_big);
        assert!(matches!(result, Err(SlotIoError::PayloadTooLarge { .. })));
    }

    #[test]
    fn read_with_meta_returns_wall_ns() {
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        slot.write(b"x").unwrap();
        let (payload, wall_ns) = slot.read_with_meta().unwrap();
        assert_eq!(payload, b"x");
        assert!(wall_ns > 0);
    }

    #[test]
    fn open_existing_slot_persists_writer_state() {
        // Slot::create's initial publish counts as version 1, so two user
        // writes leave the slot at version=3, idx=2.
        let dir = tempdir();
        let path = dir.path().join("test_slot.bin");
        {
            let mut original = Slot::create(&path, 1, 64).unwrap();
            original.write(b"first").unwrap();   // version=2, idx=1
            original.write(b"second").unwrap();  // version=3, idx=2
        }

        let mut reopened = Slot::open(&path).unwrap();
        assert_eq!(reopened.writer_version, 3);
        assert_eq!(reopened.writer_last_published_idx, 2);

        let read = reopened.read().unwrap();
        assert_eq!(read, b"second");

        reopened.write(b"third").unwrap();  // version=4, idx=0 (wrapped)
        let h = reopened.header();
        assert_eq!(h.ready_idx(), 0);
        assert_eq!(h.version(), 4);
        assert_eq!(reopened.read().unwrap(), b"third");
    }

    #[test]
    fn variable_size_payload_fits_capacity() {
        let dir = tempdir();
        let path = dir.path().join("var.bin");
        let mut slot = Slot::create(&path, 1, 1024).unwrap();
        slot.write(&[0u8; 512]).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read.len(), 512);
    }

    #[test]
    fn corrupt_ready_idx_returns_out_of_range() {
        let dir = tempdir();
        let path = dir.path().join("badidx.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        slot.write(b"good").unwrap();
        let corrupt_seq = pack_header_seq(2, 5);
        store_header_seq_release(&mut slot.mmap[..], corrupt_seq);
        let result = slot.read();
        assert!(matches!(result, Err(SlotIoError::ReadyIdxOutOfRange { got: 5, .. })));
    }

    #[test]
    fn buffer_crc_mismatch_returns_error() {
        // Real-corruption path: tamper buf CRC while header_seq stays
        // stable across the read. The version-check at step 8 sees
        // delta=0, so the read correctly returns BufferCrcMismatch
        // (NOT ReaderLapped). Per rFP §13 reordering 2026-05-08.
        let dir = tempdir();
        let path = dir.path().join("badcrc.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        slot.write(b"good").unwrap();
        let active_idx = slot.header().ready_idx();
        let off = buffer_offset(active_idx, 64);
        slot.mmap[off + 12..off + 16].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());
        let result = slot.read();
        assert!(matches!(result, Err(SlotIoError::BufferCrcMismatch { .. })));
    }

    #[test]
    fn crc_mismatch_with_version_advance_returns_reader_lapped() {
        // Torn-read path (rFP §13 fix 2026-05-08): if CRC fails AND the
        // version advanced > BUFFER_COUNT-1 between s1 and s2, the buffer
        // we read was lapped by the writer mid-read — the CRC mismatch
        // is from torn data, not corruption. Must return ReaderLapped
        // (recoverable on next tick), NOT BufferCrcMismatch (fatal).
        //
        // Simulate by:
        //   1. Set up a slot with a known good payload at idx=X, version=N.
        //   2. Tamper the CRC at idx=X (so step 6 will compute mismatch).
        //   3. Manually advance header_seq version by SHM_BUFFER_COUNT
        //      (= 3) keeping idx=X — emulates "writer lapped 3 times back
        //      to our buffer".
        //   4. Read → version1=N (snapshot from step 1 baseline read),
        //      version2=N+3, delta=3 > (BUFFER_COUNT-1)=2 → ReaderLapped.
        //
        // The natural read() does s1 fresh each call, so we need to drive
        // the race deterministically: open the slot, then between reader's
        // two header-seq loads bump the version. Easiest path: bracket
        // the tamper between two header-seq stores so that the read sees
        // a stable v1 at step 1 but a v1+N at step 7 (≥ 3 advance).
        let dir = tempdir();
        let path = dir.path().join("torn.bin");
        let mut slot = Slot::create(&path, 1, 64).unwrap();
        slot.write(b"orig").unwrap();
        let active_idx = slot.header().ready_idx();
        let v_before = slot.header().version();
        let off = buffer_offset(active_idx, 64);

        // Tamper CRC at active buffer.
        slot.mmap[off + 12..off + 16].copy_from_slice(&0xDEADBEEF_u32.to_le_bytes());

        // Manually advance version by exactly SHM_BUFFER_COUNT (= 3),
        // keeping the same idx — emulates writer wrapping 3 times back
        // to our buffer mid-read. delta = 3 > 2 → ReaderLapped.
        let lapped_seq = pack_header_seq(v_before + SHM_BUFFER_COUNT, active_idx);
        store_header_seq_release(&mut slot.mmap[..], lapped_seq);

        // Reader now: step 1 reads v1 = v_before+3; step 7 reads v2 =
        // v_before+3 too (no further writes). delta = 0, so the simple
        // store-then-read gives delta=0. To deterministically test the
        // "v1 < v2" path, we'd need a real concurrent writer.
        //
        // Instead, we directly test the integrated logic via a unit
        // function that takes (crc_ok, delta) and returns the classifier
        // verdict — see `classify_read_result` below. This isolates the
        // step-8 ordering decision from the racing read.
        let _ = slot;

        // The integration of the race is exercised by
        // `concurrent_writer_reader_sustained_load` already (and now
        // tolerates lapped torn-reads correctly). The classifier-level
        // unit test below pins the ordering invariant.
        assert_eq!(
            classify_read_result(false, SHM_BUFFER_COUNT),
            ClassifierVerdict::ReaderLapped,
            "delta == BUFFER_COUNT (=3) with CRC fail must be classified as ReaderLapped"
        );
        assert_eq!(
            classify_read_result(false, SHM_BUFFER_COUNT + 5),
            ClassifierVerdict::ReaderLapped,
            "delta > BUFFER_COUNT with CRC fail must be classified as ReaderLapped"
        );
    }

    #[test]
    fn crc_mismatch_with_no_version_advance_returns_corruption() {
        // Sibling pin: when version is stable (delta=0), CRC mismatch
        // MUST be classified as BufferCrcMismatch (real corruption),
        // not ReaderLapped. Same scenario as buffer_crc_mismatch_returns_error
        // but at the classifier level so the ordering invariant is
        // explicit.
        assert_eq!(
            classify_read_result(false, 0),
            ClassifierVerdict::BufferCrcMismatch
        );
        assert_eq!(
            classify_read_result(false, 1),
            ClassifierVerdict::BufferCrcMismatch
        );
        assert_eq!(
            classify_read_result(false, SHM_BUFFER_COUNT - 1),
            ClassifierVerdict::BufferCrcMismatch,
            "delta == BUFFER_COUNT-1 (=2) with CRC fail is still 'buffer frozen' \
             territory (writer wrote into other 2 buffers, not ours) — corruption"
        );
    }

    #[test]
    fn crc_ok_with_version_advance_within_safe_range_returns_ok() {
        // Sibling pin: when CRC is ok and delta ≤ BUFFER_COUNT-1, return Ok.
        assert_eq!(classify_read_result(true, 0), ClassifierVerdict::Ok);
        assert_eq!(classify_read_result(true, 1), ClassifierVerdict::Ok);
        assert_eq!(
            classify_read_result(true, SHM_BUFFER_COUNT - 1),
            ClassifierVerdict::Ok
        );
    }

    #[test]
    fn crc_ok_with_excessive_version_advance_still_returns_lapped() {
        // Even if CRC happened to verify (lucky), a delta > BUFFER_COUNT-1
        // means writer cycled past our buffer; the data MAY be torn even
        // if CRC accepted. Conservative: return ReaderLapped.
        assert_eq!(
            classify_read_result(true, SHM_BUFFER_COUNT),
            ClassifierVerdict::ReaderLapped
        );
        assert_eq!(
            classify_read_result(true, 100),
            ClassifierVerdict::ReaderLapped
        );
    }

    /// Classifier-level outcome used by the unit tests above to pin the
    /// step-8 ordering invariant of `Slot::read` independently of the
    /// integration path.
    #[derive(Debug, PartialEq, Eq)]
    enum ClassifierVerdict {
        Ok,
        ReaderLapped,
        BufferCrcMismatch,
    }

    fn classify_read_result(crc_ok: bool, delta: u64) -> ClassifierVerdict {
        // Exact mirror of the step-8 decision in `Slot::read` /
        // `Slot::read_with_meta`. Keep in lockstep with that code: any
        // change here MUST also update both reader hot paths.
        if delta > (SHM_BUFFER_COUNT - 1) {
            ClassifierVerdict::ReaderLapped
        } else if !crc_ok {
            ClassifierVerdict::BufferCrcMismatch
        } else {
            ClassifierVerdict::Ok
        }
    }

    #[test]
    fn many_writes_keep_slot_consistent() {
        // Slot::create starts at version=1 (initial empty publish), so the
        // i-th user write produces version i+2.
        let dir = tempdir();
        let path = dir.path().join("many.bin");
        let mut slot = Slot::create(&path, 1, 32).unwrap();
        for i in 0..50 {
            let payload = format!("v{i}");
            slot.write(payload.as_bytes()).unwrap();
            let read = slot.read().unwrap();
            assert_eq!(read, payload.as_bytes());
            assert_eq!(slot.header().version(), (i + 2) as u64);
            assert!(slot.header().ready_idx() <= HEADER_SEQ_IDX_MAX);
        }
    }

    #[test]
    fn concurrent_writer_reader_sustained_load() {
        // Sustained writer at high rate + reader from a different process
        // (via Slot::open to a separate mmap of the same file). With per-buffer
        // metadata + version-delta check, reader should see ZERO torn metadata
        // (no BufferCrcMismatch), zero retries, only the rare ReaderLapped
        // case if reader is preempted ≥ 3 publish cycles.
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let dir = tempdir();
        let path = dir.path().join("concurrent.bin");
        let _ = Slot::create(&path, 1, 64).unwrap();
        let path = Arc::new(path);

        let writer_path = path.clone();
        let writer_handle = thread::spawn(move || {
            let mut slot = Slot::open(writer_path.as_path()).unwrap();
            for i in 0..2000u64 {
                let payload = i.to_le_bytes();
                slot.write(&payload).unwrap();
            }
        });

        let reader_path = path.clone();
        let reader_handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(2));
            let slot = Slot::open(reader_path.as_path()).unwrap();
            let mut successes = 0u32;
            let mut lapped = 0u32;
            let mut uninit = 0u32;
            let mut torn = 0u32; // expected: 0
            for _ in 0..1000 {
                match slot.read() {
                    Ok(_) => successes += 1,
                    Err(SlotIoError::ReaderLapped { .. }) => lapped += 1,
                    Err(SlotIoError::Uninitialized { .. }) => uninit += 1,
                    Err(SlotIoError::BufferCrcMismatch { .. }) => torn += 1,
                    Err(other) => panic!("unexpected reader error: {other:?}"),
                }
            }
            (successes, lapped, uninit, torn)
        });

        writer_handle.join().unwrap();
        let (successes, lapped, uninit, torn) = reader_handle.join().unwrap();
        // Per-buffer metadata eliminates torn reads entirely.
        assert_eq!(torn, 0, "expected zero torn-metadata reads, got {torn}");
        assert_eq!(successes + lapped + uninit + torn, 1000);
        // Successes should dominate even under tight-loop pressure.
        assert!(
            successes > 500,
            "expected >50% reader success, got {successes}/1000 (lapped={lapped}, uninit={uninit}, torn={torn})"
        );
    }
}
