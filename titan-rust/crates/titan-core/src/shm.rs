//! shm — Universal triple-buffer slot wire format (§7.0 v1.0.0).
//!
//! Per SPEC §7.0 (D-SPEC-35 v0.2.0 → v1.0.0 MAJOR). Replaces the legacy
//! 24-byte SeqLock header. Closes
//! `titan-docs/rFP_rust_seqlock_retry_exhaustion.md`.
//!
//! # Wire format
//!
//! Slot = 16-byte fixed header + 3 × (16-byte per-buffer metadata + N-byte payload).
//!
//! Total slot size = `SHM_HEADER_BYTES + SHM_BUFFER_COUNT × (SHM_BUFFER_META_BYTES + N) = 16 + 3·(16+N) = 64 + 3·N`.
//!
//! ## Fixed header (16 bytes, `SHM_HEADER_STRUCT="<QII"`)
//!
//! ```text
//! [0:8]    uint64 LE   header_seq        — atomic publish word: bits[63:8]=version (monotonic), bits[7:0]=ready_idx ∈ {0,1,2}
//! [8:12]   uint32 LE   schema_version    — per-slot version (SPEC §3.1 D05); CONSTANT post-create
//! [12:16]  uint32 LE   payload_capacity  — per-buffer max payload bytes (= N); CONSTANT post-create
//! ```
//!
//! ## Per-buffer block (16 + N bytes, `SHM_BUFFER_META_STRUCT="<QII"` + payload)
//!
//! ```text
//! [0:8]    uint64 LE   wall_ns           — time.time_ns() at this buffer's publish completion
//! [8:12]   uint32 LE   payload_bytes     — actual bytes used in this buffer's payload (≤ payload_capacity)
//! [12:16]  uint32 LE   buffer_crc32      — CRC32 over buffer's [0:12] metadata + payload[0:payload_bytes]
//! [16:16+N]            payload           — up to N bytes
//! ```
//!
//! Buffer offset within slot mmap: `16 + idx × (16+N)` for `idx ∈ {0,1,2}`.
//!
//! # Why per-buffer metadata
//!
//! Earlier draft v1.0.0 placed metadata in the fixed header [8:32] and
//! atomically published only `header_seq`. Under sustained writer
//! contention, readers Acquire-loading the OLD header_seq could observe
//! the writer's mid-update of metadata fields, producing torn metadata
//! reads (CRC mismatch). Co-locating metadata with payload makes the
//! entire buffer state atomic with the publish: writer fills
//! `buf[next_idx]` ENTIRELY (metadata + payload + CRC) BEFORE
//! atomic-Release-storing `header_seq`; readers Acquire-load and read
//! from the published buffer with zero torn reads.
//!
//! # Sentinel values
//!
//! - `version == 0` ⇒ slot uninitialized (never published). Reader returns
//!   `Uninitialized`. First successful writer publish sets `version = 1`.
//! - `ready_idx > 2` ⇒ corrupt header_seq. Reader returns
//!   `ReadyIdxOutOfRange`.
//!
//! # Race-elimination proof
//!
//! Writer rotates 0→1→2→0. After publishing idx=K (version V), the next
//! 2 publishes target (K+1)%3 and (K+2)%3 — neither is K. The 3rd publish
//! reuses K with version V+3. Reader who copied buf[K] at version V is
//! safe iff version2 < V+3 ⇔ delta ≤ 2.
//!
//! # Endianness
//!
//! All multi-byte fields little-endian. Rust uses explicit `to_le_bytes()`
//! / `from_le_bytes()` for portability.

pub use crate::constants::{SHM_BUFFER_COUNT, SHM_BUFFER_META_BYTES, SHM_HEADER_BYTES};

/// Bit shift to extract `version` from `header_seq`.
pub const HEADER_SEQ_VERSION_SHIFT: u32 = 8;
/// Mask to extract `ready_idx` from `header_seq`.
pub const HEADER_SEQ_IDX_MASK: u64 = 0xFF;
/// Maximum legal value for `ready_idx` (= SHM_BUFFER_COUNT - 1).
pub const HEADER_SEQ_IDX_MAX: u8 = 2;

/// Compose a `header_seq` u64 value from version + ready_idx.
#[inline]
pub fn pack_header_seq(version: u64, ready_idx: u8) -> u64 {
    (version << HEADER_SEQ_VERSION_SHIFT) | (ready_idx as u64 & HEADER_SEQ_IDX_MASK)
}

/// Compute the byte offset of buffer `idx` within a slot's mmap, given
/// per-buffer payload capacity `payload_capacity` (= N).
#[inline]
pub fn buffer_offset(idx: u8, payload_capacity: usize) -> usize {
    SHM_HEADER_BYTES as usize + (idx as usize) * (SHM_BUFFER_META_BYTES as usize + payload_capacity)
}

/// 16-byte fixed slot header (§7.0 v1.0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotHeader {
    /// Atomic publish word — `(version << 8) | ready_idx`.
    pub header_seq: u64,
    /// Per-slot schema version (set at Slot::create, constant post-create).
    pub schema_version: u32,
    /// Per-buffer max payload bytes (set at Slot::create, constant post-create).
    pub payload_capacity: u32,
}

impl SlotHeader {
    /// Extract version (high 56 bits) from `header_seq`.
    #[inline]
    pub fn version(&self) -> u64 {
        self.header_seq >> HEADER_SEQ_VERSION_SHIFT
    }

    /// Extract ready_idx (low 8 bits) from `header_seq`.
    #[inline]
    pub fn ready_idx(&self) -> u8 {
        (self.header_seq & HEADER_SEQ_IDX_MASK) as u8
    }

    /// `true` if the slot has been published at least once.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.version() > 0
    }

    /// `true` if `ready_idx` is in the valid range `[0, SHM_BUFFER_COUNT)`.
    #[inline]
    pub fn ready_idx_valid(&self) -> bool {
        self.ready_idx() <= HEADER_SEQ_IDX_MAX
    }

    /// Decode a 16-byte fixed header from raw bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, SlotError> {
        if bytes.len() < SHM_HEADER_BYTES as usize {
            return Err(SlotError::ShortHeader {
                got: bytes.len(),
                expected: SHM_HEADER_BYTES as usize,
            });
        }
        Ok(SlotHeader {
            header_seq: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            schema_version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            payload_capacity: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }

    /// Encode the fixed header as exactly 16 bytes.
    pub fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.header_seq.to_le_bytes());
        buf[8..12].copy_from_slice(&self.schema_version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.payload_capacity.to_le_bytes());
        buf
    }
}

/// Per-buffer metadata block (16 bytes preceding the payload of one buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferMeta {
    /// `time.time_ns()` at this buffer's publish completion.
    pub wall_ns: u64,
    /// Actual payload size in this buffer (≤ `payload_capacity`).
    pub payload_bytes: u32,
    /// CRC32 over buffer's [0:12] metadata + payload[0:payload_bytes].
    pub buffer_crc32: u32,
}

impl BufferMeta {
    /// Decode 16-byte metadata from raw bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, SlotError> {
        if bytes.len() < SHM_BUFFER_META_BYTES as usize {
            return Err(SlotError::ShortBufferMeta {
                got: bytes.len(),
                expected: SHM_BUFFER_META_BYTES as usize,
            });
        }
        Ok(BufferMeta {
            wall_ns: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            payload_bytes: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            buffer_crc32: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }

    /// Encode the 16-byte metadata.
    pub fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.wall_ns.to_le_bytes());
        buf[8..12].copy_from_slice(&self.payload_bytes.to_le_bytes());
        buf[12..16].copy_from_slice(&self.buffer_crc32.to_le_bytes());
        buf
    }

    /// Compute the CRC32 that should be stored in `buffer_crc32`, given
    /// the buffer's wall_ns + payload_bytes (the first 12 bytes of the
    /// 16-byte metadata block) and the actual payload bytes.
    pub fn compute_crc32(wall_ns: u64, payload_bytes: u32, payload: &[u8]) -> u32 {
        // CRC over [0:12] metadata prefix + payload[0:payload_bytes].
        let mut hdr = [0u8; 12];
        hdr[0..8].copy_from_slice(&wall_ns.to_le_bytes());
        hdr[8..12].copy_from_slice(&payload_bytes.to_le_bytes());
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in &hdr {
            let idx = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[idx];
        }
        for &byte in payload {
            let idx = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[idx];
        }
        crc ^ 0xFFFF_FFFF
    }
}

/// Errors during slot read/write at the wire-format layer.
#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    /// Header bytes too short.
    #[error("slot header: got {got} bytes, expected {expected}")]
    ShortHeader {
        /// Bytes received.
        got: usize,
        /// Bytes expected.
        expected: usize,
    },

    /// Per-buffer metadata bytes too short.
    #[error("slot buffer metadata: got {got} bytes, expected {expected}")]
    ShortBufferMeta {
        /// Bytes received.
        got: usize,
        /// Bytes expected.
        expected: usize,
    },

    /// Buffer CRC32 mismatch (corruption or torn read).
    #[error("buffer CRC32 mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    BufferCrcMismatch {
        /// CRC32 stored in buffer metadata.
        stored: u32,
        /// CRC32 computed from buffer content.
        computed: u32,
    },

    /// Schema version did not match expected value.
    #[error("slot schema mismatch: stored {stored}, expected {expected}")]
    SchemaMismatch {
        /// Schema version read from slot.
        stored: u32,
        /// Schema version expected by reader.
        expected: u32,
    },

    /// Slot has not been published yet (`version == 0` sentinel).
    #[error("slot uninitialized — never published")]
    Uninitialized,

    /// Reader was preempted long enough that writer lapped through all
    /// 3 buffers during the read — buffer the reader copied was overwritten.
    /// Extraordinarily rare in practice.
    #[error("slot reader lapped — version delta {delta} > 2 (writer overwrote buffer during read)")]
    ReaderLapped {
        /// Number of writer publishes that completed during the read.
        delta: u64,
    },

    /// `ready_idx` field out of valid range `[0, SHM_BUFFER_COUNT)`.
    #[error("slot ready_idx out of range: {got} > {max}")]
    ReadyIdxOutOfRange {
        /// Index value read.
        got: u8,
        /// Maximum legal value.
        max: u8,
    },

    /// I/O error.
    #[error("slot I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Standard CRC32 (IEEE 802.3 / Ethernet polynomial), table-driven.
///
/// Matches Python `binascii.crc32()` and `zlib.crc32()` — both use the same
/// polynomial. Verified by parity vectors at
/// `tests/parity/vectors.json::shm_layout`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut j = 0;
        while j < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_header_size_is_16_bytes() {
        let h = SlotHeader {
            header_seq: 0,
            schema_version: 1,
            payload_capacity: 0,
        };
        assert_eq!(h.encode().len(), SHM_HEADER_BYTES as usize);
        assert_eq!(SHM_HEADER_BYTES, 16);
    }

    #[test]
    fn buffer_meta_size_is_16_bytes() {
        let m = BufferMeta {
            wall_ns: 0,
            payload_bytes: 0,
            buffer_crc32: 0,
        };
        assert_eq!(m.encode().len(), SHM_BUFFER_META_BYTES as usize);
        assert_eq!(SHM_BUFFER_META_BYTES, 16);
    }

    #[test]
    fn buffer_count_is_3() {
        assert_eq!(SHM_BUFFER_COUNT, 3);
    }

    #[test]
    fn fixed_header_round_trip() {
        let h = SlotHeader {
            header_seq: pack_header_seq(42, 1),
            schema_version: 7,
            payload_capacity: 648,
        };
        let bytes = h.encode();
        let decoded = SlotHeader::decode(&bytes).unwrap();
        assert_eq!(decoded, h);
    }

    #[test]
    fn buffer_meta_round_trip() {
        let m = BufferMeta {
            wall_ns: 1_700_000_000_000_000_000,
            payload_bytes: 100,
            buffer_crc32: 0xDEAD_BEEF,
        };
        let bytes = m.encode();
        let decoded = BufferMeta::decode(&bytes).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn header_seq_pack_unpack() {
        let h = SlotHeader {
            header_seq: pack_header_seq(123, 2),
            schema_version: 1,
            payload_capacity: 0,
        };
        assert_eq!(h.version(), 123);
        assert_eq!(h.ready_idx(), 2);
    }

    #[test]
    fn header_seq_idx_range() {
        for idx in 0..=2u8 {
            let h = SlotHeader {
                header_seq: pack_header_seq(1, idx),
                schema_version: 1,
                payload_capacity: 0,
            };
            assert!(h.ready_idx_valid());
            assert_eq!(h.ready_idx(), idx);
        }
        // 3 is out of range
        let h = SlotHeader {
            header_seq: pack_header_seq(1, 3),
            schema_version: 1,
            payload_capacity: 0,
        };
        assert!(!h.ready_idx_valid());
    }

    #[test]
    fn version_zero_sentinel() {
        let h = SlotHeader {
            header_seq: 0,
            schema_version: 1,
            payload_capacity: 0,
        };
        assert!(!h.is_initialized());
        let h = SlotHeader {
            header_seq: pack_header_seq(1, 0),
            schema_version: 1,
            payload_capacity: 0,
        };
        assert!(h.is_initialized());
    }

    #[test]
    fn version_truncation_top_bit() {
        let big_version = 1u64 << 55;
        let h = SlotHeader {
            header_seq: pack_header_seq(big_version, 0),
            schema_version: 1,
            payload_capacity: 0,
        };
        assert_eq!(h.version(), big_version);
    }

    #[test]
    fn buffer_offset_formula() {
        // For N=100: buf 0 at 16, buf 1 at 16+(16+100)=132, buf 2 at 16+2*(16+100)=248
        assert_eq!(buffer_offset(0, 100), 16);
        assert_eq!(buffer_offset(1, 100), 132);
        assert_eq!(buffer_offset(2, 100), 248);
    }

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_matches_python_zlib() {
        assert_eq!(crc32(b"hello"), 0x3610_A686);
    }

    #[test]
    fn buffer_meta_compute_crc32_matches_concatenated() {
        // BufferMeta::compute_crc32 should equal crc32 over the same bytes.
        let wall_ns: u64 = 1_700_000_000_000_000_000;
        let payload = b"hello world";
        let payload_bytes = payload.len() as u32;

        let mut concat = Vec::new();
        concat.extend_from_slice(&wall_ns.to_le_bytes());
        concat.extend_from_slice(&payload_bytes.to_le_bytes());
        concat.extend_from_slice(payload);

        let expected = crc32(&concat);
        let computed = BufferMeta::compute_crc32(wall_ns, payload_bytes, payload);
        assert_eq!(computed, expected);
    }
}
