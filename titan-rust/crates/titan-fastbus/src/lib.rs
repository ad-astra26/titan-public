//! titan-fastbus — Lock-free SPSC shm ring for kernel ↔ trinity-substrate IPC.
//!
//! Per SPEC §7.1 fastbus.bin layout + PLAN_microkernel_phase_c_s3_substrate.md §9.
//! Sub-100µs latency (OBS-c-s3-fastbus-latency: p50 < 100µs, p99 < 500µs).
//!
//! # Roles
//!
//! - **Producer (kernel)**: writes `circadian_tick` / `pi_heartbeat_tick` events
//!   for substrate to consume + modulate Schumann phase.
//! - **Consumer (substrate)**: reads producer events; also operates a
//!   producer-side handle to publish `schumann_epoch` ticks back to kernel for
//!   clock validation. SPSC: each direction is single-producer, single-consumer.
//!
//! # File layout (SPEC §7.1 + §9.1)
//!
//! ```text
//! [0:24]      Universal SeqLock header (24 bytes — written by kernel at boot per SPEC §7.0).
//!             Substrate does NOT modify; lock-free ring uses atomic indices instead.
//! [24:88]     Ring header (64 bytes):
//!     [24:32]   uint8[8]   magic         = b"TITANFB1"
//!     [32:36]   uint32 LE  version       = 1
//!     [36:44]   uint64 LE  read_idx      (consumer-modified, Acquire/Release)
//!     [44:52]   uint64 LE  write_idx     (producer-modified, Release/Acquire)
//!     [52:56]   uint32 LE  mask          = 1023
//!     [56:60]   uint32 LE  slot_bytes    = 256
//!     [60:88]   uint8[28]  reserved      = zero
//! [88:262232] Slot array: 1024 × 256 bytes = 262144 bytes.
//! ```
//!
//! Total file = **262232 bytes** (matches kernel's `titan-state` slot spec).
//!
//! # Memory ordering
//!
//! - Producer: `Acquire` load on `read_idx` (sees consumer commits); `Release`
//!   store on `write_idx` (releases payload writes happens-before consumer
//!   `Acquire` on `write_idx`).
//! - Consumer: `Acquire` load on `write_idx` (acquires producer payload);
//!   `Release` store on `read_idx`.
//!
//! See PLAN §9.5 for Linux Kernel Memory Model rationale.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod handle;
pub mod payload;
pub mod ring;

pub use crate::handle::{Consumer, Producer};
pub use crate::payload::{Message, MsgType, PayloadError};
pub use crate::ring::{FastbusError, Ring, RingHeader};

// Re-export the canonical slot byte size for caller convenience.
pub use titan_core::constants::{
    FASTBUS_HEADER_BYTES, FASTBUS_MAGIC_BYTES, FASTBUS_RING_CAPACITY_SLOTS, FASTBUS_RING_VERSION,
    FASTBUS_SLOT_BYTES,
};

/// Total file size for the fastbus.bin layout: 24 (universal SeqLock header) +
/// 64 (ring header) + 1024 × 256 (slot array) = **262232 bytes**.
///
/// This MUST equal the kernel-pre-allocated size (per `titan-state::spec` /
/// SPEC §7.1). Mismatch is a SPEC violation.
pub const FASTBUS_FILE_TOTAL_BYTES: usize = 24
    + FASTBUS_HEADER_BYTES as usize
    + (FASTBUS_RING_CAPACITY_SLOTS as usize) * (FASTBUS_SLOT_BYTES as usize);

/// Universal SeqLock header byte size, mirrored from `titan-core::constants`
/// for caller convenience. Per SPEC §7.0.
pub const UNIVERSAL_SEQLOCK_HEADER_BYTES: usize = 24;

const _: () = assert!(FASTBUS_FILE_TOTAL_BYTES == 262232);
