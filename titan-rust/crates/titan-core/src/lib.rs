//! titan-core — Shared primitives for Titan microkernel v2 Phase C.
//!
//! This crate is consumed by every Phase C Rust binary (kernel, substrate,
//! unified-spirit, 6 trinity daemons) and provides byte-identical contracts
//! to today's Python `titan_hcl/core/` modules.
//!
//! # Modules
//!
//! - [`constants`] — AUTO-GENERATED from
//!   `titan-docs/SPEC_titan_architecture_constants.toml` via
//!   `arch_map phase-c regen`. Hand-editing is a SPEC violation per §19.
//! - [`frame`] — length-prefix + HMAC-SHA256 challenge-response framing
//!   (B.2 protocol; SPEC §8.10 byte-identical guarantee; RFC 4231 vectors).
//! - [`authkey`] — HKDF-SHA256 bus authkey derivation from Ed25519 identity
//!   (B.2 §D1; RFC 5869 vectors; SPEC §3.1 D06; SPEC §11.6 parity).
//! - [`identity`] — Ed25519 keypair load + holder with `Zeroize`-on-drop
//!   (SPEC G16(8) SACRED file class).
//! - [`atomic_write`] — Atomic-write helper with N-generation backup
//!   rotation (SPEC §11.H.2; G16 data integrity).
//! - [`shm`] — SeqLock primitives + 24-byte universal slot header
//!   (SPEC §7.0). Full slot lifecycle ships in `titan-state` (C2-3).
//! - [`supervisor`] — Supervisor primitives skeleton (one_for_one,
//!   `SupervisionReason` enum). Full impl ships in C2-5
//!   (escalation handshake §11.B.1, dependency-aware respawn §11.G).
//! - [`bus_specs`] — `phf::Map<&str, BusMsgSpec>` skeleton (SPEC §8 catalog).
//!   Full 47-message catalog ships in C2-2.
//!
//! # SPEC discipline
//!
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every value comes from
//! `constants` (auto-generated from TOML). No inline magic numbers.
//! Pre-commit hook (`arch_map phase-c verify --strict`) blocks drift.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod atomic_write;
pub mod authkey;
pub mod bus_specs;
#[allow(missing_docs)]
// Auto-generated from SPEC TOML; constants are self-describing per `// `-comment.
pub mod constants;
pub mod frame;
pub mod identity;
pub mod middle_path;
pub mod shm;
pub mod small_filter_down;
pub mod supervisor;
pub mod transition_buffer;
pub mod trinity_value_net;

// Convenience re-exports for the most-used types
pub use crate::authkey::{derive_bus_authkey, AUTHKEY_BYTES, AUTHKEY_HKDF_INFO, AUTHKEY_HKDF_SALT};
pub use crate::frame::{
    compute_hmac, constant_time_eq, FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES,
    FRAME_LENGTH_PREFIX_BYTES, FRAME_MAX_FRAME_BYTES,
};
pub use crate::identity::{Identity, TitanId};
pub use crate::supervisor::SupervisionReason;
