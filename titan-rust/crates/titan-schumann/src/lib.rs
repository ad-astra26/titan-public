//! titan-schumann — Schumann frequency timer-wheel generators.
//!
//! Per Preamble G13 LOCKED + SPEC §10.H consolidated cadence view + PLAN
//! §8 (this PLAN). Frequencies are biological constants tied to Earth's
//! electromagnetic field — **NOT tunable**:
//!
//! - Body  @ `SCHUMANN_BODY_HZ`  = 7.83 Hz  (period 127.71 ms)
//! - Mind  @ `SCHUMANN_MIND_HZ`  = 23.49 Hz (period  42.57 ms — = 7.83 × 3)
//! - Spirit@ `SCHUMANN_SPIRIT_HZ`= 70.47 Hz (period  14.19 ms — = 7.83 × 9)
//!
//! All 3 generators share a common `epoch_t0` so their phases stay aligned by
//! construction: every 9th spirit tick coincides with a body tick; every 3rd
//! mind tick coincides with a body tick.
//!
//! # Targets (substrate-level)
//!
//! - Drift over 24h < `SCHUMANN_DRIFT_TARGET_PCT` = 0.1%  (OBS-c-s3-schumann-precision)
//! - Per-tick jitter p99 < `SCHUMANN_JITTER_P99_MS` = 1.0 ms (OBS-c-s3-schumann-precision)
//!
//! # Why tokio interval is sufficient
//!
//! tokio::time on Linux x86_64 uses a high-precision timer wheel + clock_nanosleep
//! at the floor; resolution ~100µs. The 70.47 Hz target gives 14.2 ms periods —
//! comfortably above the floor. Soak-data-driven escape hatch: switch to direct
//! `nix::time::clock_nanosleep` if Phase D measurements show we exceed targets.
//!
//! # Use from substrate (C3-3+)
//!
//! ```no_run
//! use std::sync::Arc;
//! use tokio::sync::Notify;
//! use titan_schumann::{SchumannGenerator, SchumannRole};
//!
//! # async fn substrate_boot() {
//! let shutdown = Arc::new(Notify::new());
//! let epoch_t0 = tokio::time::Instant::now();
//! let mut spirit_rx = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0)
//!     .spawn(shutdown.clone());
//!
//! while let Some(tick) = spirit_rx.recv().await {
//!     // every 9th tick is a body cycle boundary per Preamble G13 ratios
//!     if tick.epoch % 9 == 0 {
//!         // run substrate body cycle (read 6 daemon slots → topology → publish)
//!     }
//! }
//! # }
//! ```
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod generator;

pub use crate::generator::{
    period_ns_for_role, BodySchumann, MindSchumann, SchumannError, SchumannGenerator, SchumannRole,
    SpiritSchumann, TickEvent,
};

// Re-export the LOCKED Schumann constants for caller convenience
pub use titan_core::constants::{
    SCHUMANN_BODY_HZ, SCHUMANN_DRIFT_TARGET_PCT, SCHUMANN_JITTER_P99_MS, SCHUMANN_MIND_HZ,
    SCHUMANN_SPIRIT_HZ,
};
