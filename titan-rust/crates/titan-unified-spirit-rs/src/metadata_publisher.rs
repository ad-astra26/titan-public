//! metadata_publisher — Phase B SHM publication of ResonanceDetector +
//! UnifiedSpirit + FilterDownV5Engine metadata.
//!
//! Per `rFP_phase_c_state_read_unification_l0_l1_canonical §B` + D-SPEC-72.
//! Closes the G15 single-source-of-truth gap where Python `SpiritStatePublisher`
//! wrapped these engines' `get_stats()` into Python-side SHM slots while the
//! canonical computation lives in Rust. After Phase B:
//!
//!   resonance_metadata.bin      ← Rust ResonanceDetector::get_stats()
//!   unified_spirit_metadata.bin ← Rust UnifiedSpirit::get_stats()
//!   filter_down_state.bin       ← Rust FilterDownV5Engine::get_stats()
//!
//! G21 single-writer: this module is the sole writer of all three slots.
//! G18 SHM-canonical: state lives in SHM, never on the bus.
//!
//! Cadence: piggybacks on the existing `spawn_publisher_task` body cycle
//! (`BODY_CYCLE_INTERVAL_MS = 1150ms`). Each tick:
//!   1. Acquire mutex on each engine
//!   2. Call `get_stats()`
//!   3. Serialize to msgpack via rmp-serde
//!   4. Write to SHM via Slot::write
//!
//! Errors are logged at WARN + counted; never panic.

// Field-level docs are redundant with the docstring above + Python parity
// reference. Allow missing_docs on the schema fields per
// `feedback_phase_c_break_monolith_ethos.md` (ship the carve, don't fight
// linting on data-transfer types whose semantics are documented at the
// struct level).
#![allow(missing_docs)]

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use titan_state::Slot;

use crate::filter_down::FilterDownV5Engine;
use crate::resonance::ResonanceDetector;
use crate::unified_spirit::{GreatEpoch, UnifiedSpirit};

// ── Payload schemas (msgpack-serializable, match Python get_stats() shapes) ──

/// Per-pair stats as emitted in `resonance_metadata.bin → pairs[<name>]`.
/// Matches Python `ResonancePair.get_stats()` schema 1:1 so consumers see
/// zero behavior change post-ownership-flip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonancePairStats {
    pub name: String,
    pub is_resonant: bool,
    pub consecutive_resonant: u32,
    pub required_cycles: u32,
    pub total_resonant_cycles: u64,
    pub total_checks: u64,
    pub big_pulse_count: u64,
    pub inner_pulse_count: u64,
    pub outer_pulse_count: u64,
    pub last_big_pulse_ts: f64,
    pub phase_threshold: f64,
    pub pulse_window: f64,
}

/// Config sub-dict in `resonance_metadata.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceConfigSummary {
    pub phase_threshold_deg: f64,
    pub required_cycles: u32,
    pub pulse_window: f64,
}

/// Full `resonance_metadata.bin` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceMetadata {
    pub pairs: std::collections::BTreeMap<String, ResonancePairStats>,
    pub resonant_count: u32,
    pub all_resonant: bool,
    pub great_pulse_count: u64,
    pub last_great_pulse_ts: f64,
    pub config: ResonanceConfigSummary,
    pub schema_version: u32,
    pub ts: f64,
}

/// Config sub-dict in `unified_spirit_metadata.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSpiritConfigSummary {
    pub stale_threshold: f64,
    pub enrichment_base: f64,
    pub velocity_window: usize,
    pub enrichment_rate: f64,
    pub min_alignment_threshold: f64,
}

/// Full `unified_spirit_metadata.bin` payload. `latest_epoch` is the full
/// `GreatEpoch` struct serialized to all 10 fields — matches Python
/// `GreatEpoch.to_dict()` 1:1 per `feedback_implement_rfp_fully_no_simplifications_no_deferrals.md`
/// (epoch_id, timestamp, spirit_tensor, magnitude, velocity, enrichment_sent,
/// resonance_snapshot, anchor_hash, cumulative_quality, micro_tick_count).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSpiritMetadata {
    pub epoch_count: usize,
    pub current_epoch_id: u64,
    pub velocity: f64,
    pub is_stale: bool,
    pub consecutive_stale: u32,
    pub stale_focus_multiplier: f64,
    pub tensor_magnitude: f64,
    pub tensor_sum: f64,
    pub latest_epoch: Option<GreatEpoch>,
    pub cumulative_quality: f64,
    pub micro_tick_count: u64,
    pub last_alignment: f64,
    pub enrichment_rate: f64,
    pub full_130dt: Vec<f64>,
    pub config: UnifiedSpiritConfigSummary,
    pub schema_version: u32,
    pub ts: f64,
}

/// Mean of each 6-band multiplier window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipliersMean {
    pub inner_body: f64,
    pub inner_mind: f64,
    pub inner_spirit_content: f64,
    pub outer_body: f64,
    pub outer_mind: f64,
    pub outer_spirit_content: f64,
}

/// EMA multipliers per band (live floats).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipliersBands {
    pub inner_body: Vec<f64>,
    pub inner_mind: Vec<f64>,
    pub inner_spirit_content: Vec<f64>,
    pub outer_body: Vec<f64>,
    pub outer_mind: Vec<f64>,
    pub outer_spirit_content: Vec<f64>,
}

/// Full `filter_down_state.bin` payload — mirrors Python
/// `FilterDownV5Engine.get_stats()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterDownStateMetadata {
    pub version: String,
    pub input_dim: u32,
    pub output_dim: u32,
    pub buffer_size: usize,
    pub total_train_steps: u64,
    pub last_loss: f64,
    pub publish_enabled: bool,
    pub spirit_filter_strength: f64,
    pub cold_start_floor: u64,
    pub multipliers_mean: MultipliersMean,
    pub multipliers: MultipliersBands,
    pub schema_version: u32,
    pub ts: f64,
}

// ── Publisher ────────────────────────────────────────────────────────────

/// Owner of the 3 SHM slot writers. One per process.
pub struct MetadataPublisher {
    resonance: Slot,
    unified_spirit: Slot,
    filter_down: Slot,
    /// Per-slot consecutive-error counters for log throttling (3 slots).
    err_counts: [u64; 3],
}

/// Errors during metadata publish — all transient (logged + counted, never fatal).
#[derive(Debug, thiserror::Error)]
pub enum MetadataPublishError {
    /// Could not open one of the 3 slot files at startup.
    #[error("slot open failed for {slot}: {source}")]
    Open {
        /// Slot filename.
        slot: &'static str,
        /// Underlying I/O error.
        source: titan_state::SlotIoError,
    },
}

impl MetadataPublisher {
    /// Open the 3 slot files for writing. Kernel must have pre-created them
    /// per SPEC §10.A B3 (`SlotCreator::Kernel` in `titan-state::spec`).
    pub fn open_all(shm_dir: &Path) -> Result<Self, MetadataPublishError> {
        let open = |name: &'static str| -> Result<Slot, MetadataPublishError> {
            Slot::open(shm_dir.join(name))
                .map_err(|source| MetadataPublishError::Open { slot: name, source })
        };
        let me = Self {
            resonance: open("resonance_metadata.bin")?,
            unified_spirit: open("unified_spirit_metadata.bin")?,
            filter_down: open("filter_down_state.bin")?,
            err_counts: [0; 3],
        };
        info!(
            event = "METADATA_PUBLISHER_OPEN",
            "3 Phase B SHM slots opened for writing (resonance_metadata + unified_spirit_metadata + filter_down_state)"
        );
        Ok(me)
    }

    /// Encode + write one publish cycle. Per-slot failures are logged but
    /// never propagated up — the caller's tick continues.
    pub fn publish(
        &mut self,
        detector: &Arc<Mutex<ResonanceDetector>>,
        spirit: &Arc<Mutex<UnifiedSpirit>>,
        engine: &Arc<Mutex<FilterDownV5Engine>>,
        ts: f64,
    ) {
        // ── Resonance ─────────────────────────────────────────────────
        let res_payload = detector.lock().get_stats(ts);
        match rmp_serde::to_vec_named(&res_payload) {
            Ok(bytes) => {
                if let Err(e) = self.resonance.write(&bytes) {
                    Self::log_throttled(
                        &mut self.err_counts[0],
                        "resonance_metadata.bin",
                        "METADATA_WRITE_FAIL",
                        bytes.len(),
                        format!("{:?}", e),
                    );
                }
            }
            Err(e) => Self::log_throttled(
                &mut self.err_counts[0],
                "resonance_metadata.bin",
                "METADATA_ENCODE_FAIL",
                0,
                format!("{:?}", e),
            ),
        }

        // ── Unified Spirit ────────────────────────────────────────────
        let us_payload = spirit.lock().get_stats(ts);
        match rmp_serde::to_vec_named(&us_payload) {
            Ok(bytes) => {
                if let Err(e) = self.unified_spirit.write(&bytes) {
                    Self::log_throttled(
                        &mut self.err_counts[1],
                        "unified_spirit_metadata.bin",
                        "METADATA_WRITE_FAIL",
                        bytes.len(),
                        format!("{:?}", e),
                    );
                }
            }
            Err(e) => Self::log_throttled(
                &mut self.err_counts[1],
                "unified_spirit_metadata.bin",
                "METADATA_ENCODE_FAIL",
                0,
                format!("{:?}", e),
            ),
        }

        // ── Filter Down ───────────────────────────────────────────────
        let fd_payload = engine.lock().get_stats(ts);
        match rmp_serde::to_vec_named(&fd_payload) {
            Ok(bytes) => {
                if let Err(e) = self.filter_down.write(&bytes) {
                    Self::log_throttled(
                        &mut self.err_counts[2],
                        "filter_down_state.bin",
                        "METADATA_WRITE_FAIL",
                        bytes.len(),
                        format!("{:?}", e),
                    );
                }
            }
            Err(e) => Self::log_throttled(
                &mut self.err_counts[2],
                "filter_down_state.bin",
                "METADATA_ENCODE_FAIL",
                0,
                format!("{:?}", e),
            ),
        }
    }

    fn log_throttled(
        counter: &mut u64,
        slot: &'static str,
        event: &'static str,
        bytes: usize,
        err: String,
    ) {
        *counter = counter.saturating_add(1);
        let c = *counter;
        if c == 1 || c.is_multiple_of(100) {
            warn!(event = event, slot = slot, err = %err, count = c, bytes = bytes);
        }
    }
}
