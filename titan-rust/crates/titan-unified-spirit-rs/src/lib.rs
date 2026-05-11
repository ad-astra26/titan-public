//! titan-unified-spirit-rs library surface.
//!
//! Phase C C-S4 — L1b SELF orchestrator. Owns 162D TITAN_SELF tensor
//! assembly, ResonanceDetector + UnifiedSpirit advance() crystallization,
//! and FilterDownV5Engine (V5 ReLU MLP + TD(0) trainer + multiplier
//! publisher) per SPEC §9.A + §10.F.
//!
//! Per SPEC §3.0 Running-Titans Safety Rule: behind
//! `microkernel.l0_rust_enabled = false` flag default. C-S7 first flag-flip.
//!
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every constant comes
//! from `titan-core::constants` (auto-generated from SPEC TOML); no hand
//! magic numbers. Rule 2: `arch_map phase-c verify --strict` runs clean
//! against this crate at session-close. Rule 3: this crate's design is
//! 100% compatible with master plan §10.4 + SPEC v0.1.3+.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod boot;
pub mod child_specs;
pub mod cli;
pub mod exit;
pub mod filter_down;
pub mod logging;
pub mod middle_path;
pub mod orchestration;
pub mod resonance;
pub mod runtime;
pub mod self_assembly;
pub mod slot_handles;
pub mod supervise;
pub mod unified_spirit;
pub mod version;
