//! unified_spirit — Full Rust port of `titan_hcl/logic/unified_spirit.py`.
//!
//! Per SPEC §10.F G12 (SPIRIT cannot move backward) + §11.H.1 critical-data
//! row + Decision Log D-SPEC-26 (lives inside unified-spirit-rs).
//!
//! 130D SPIRIT tensor layout (matches Python `_tensor`):
//! - `[0:5]`     Inner Body
//! - `[5:20]`    Inner Mind (15D)
//! - `[20:65]`   Inner Spirit (45D — `[20:25]` observer dims, NEVER modulated)
//! - `[65:70]`   Outer Body
//! - `[70:85]`   Outer Mind (15D)
//! - `[85:130]`  Outer Spirit (45D — `[85:90]` observer dims, NEVER modulated)
//!
//! Lifecycle:
//! 1. `update_subconscious` + `update_conscious` per body cycle (130D refresh)
//! 2. `micro_enrich(realtime_state)` accumulates `cumulative_quality`
//! 3. On GREAT PULSE (caller — `boot::run_bus_dispatch_loop`'s on_big_pulse
//!    callback when `great_pulse_ready=true`): `advance(resonance_snapshot)`
//!    crystallizes a GreatEpoch + persists state via atomic_write.
//! 4. `compute_enrichment()` returns per-component rewards (consumed by
//!    body/mind/spirit daemons in C-S5/C-S6 — wire-published in C4-3c).
//! 5. `is_stale` + `stale_focus_multiplier` drive FOCUS cascade
//!    (consumed in C4-3c V5 publish).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Total tensor width — Inner 65D + Outer 65D.
pub const SPIRIT_DIMS: usize = 130;
/// Inner half width.
pub const INNER_DIMS: usize = 65;
/// Outer half width.
pub const OUTER_DIMS: usize = 65;
/// Default cell value at init / migration pad — matches Python `[0.5] * 130`.
pub const TENSOR_INIT_VALUE: f64 = 0.5;

/// Default base enrichment reward per GREAT PULSE — matches Python
/// `DEFAULT_ENRICHMENT_BASE`.
pub const DEFAULT_ENRICHMENT_BASE: f64 = 0.02;
/// Default enrichment rate for `micro_enrich` quality factor — matches
/// Python default.
pub const DEFAULT_ENRICHMENT_RATE: f64 = 0.02;
/// Default minimum alignment for enrichment to trigger — matches Python
/// default.
pub const DEFAULT_MIN_ALIGNMENT_THRESHOLD: f64 = 0.1;
/// Default cap on stale-focus escalation — Python `min(3.0, ...)`.
pub const STALE_FOCUS_MULTIPLIER_CAP: f64 = 3.0;
/// Per-stale-cycle escalation increment (Python `0.2 * consecutive_stale`).
pub const STALE_FOCUS_ESCALATION_PER_CYCLE: f64 = 0.2;
/// Schema version for `data/unified_spirit_state.json` per SPEC §11.H.4.
pub const UNIFIED_SPIRIT_STATE_SCHEMA_VERSION: u32 = 1;

/// Errors during persistence + restore.
#[derive(Debug, thiserror::Error)]
pub enum UnifiedSpiritError {
    /// `data/unified_spirit_state.json` write failed.
    #[error("unified_spirit_state write failed: {0}")]
    Write(#[from] titan_core::atomic_write::AtomicWriteError),
    /// JSON encode/decode failed.
    #[error("unified_spirit_state json: {0}")]
    Json(#[from] serde_json::Error),
    /// io error reading state file.
    #[error("unified_spirit_state io: {0}")]
    Io(#[from] std::io::Error),
    /// G12 monotonic guarantee violated — loaded state has lower
    /// epoch_id than known prior in-memory state. SPEC §10.F G12: SPIRIT
    /// cannot move backward. Caller halts module + escalates.
    #[error("G12 monotonic violation: loaded epoch_id={loaded}, in-memory={in_memory}")]
    G12Violation {
        /// epoch_id read from the state file.
        loaded: u64,
        /// epoch_id held in memory at load attempt.
        in_memory: u64,
    },
    /// Both canonical + 2 backups failed to load. SPEC §11.H.4 → emit
    /// `SUPERVISION_DATA_LOST` and halt module.
    #[error("all backups corrupt: canonical, .bak, .bak.prev")]
    AllBackupsCorrupt,
}

/// V5 multiplier dict — matches SPEC §10.F V5 cascade payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct V5Multipliers {
    /// Inner body multipliers — 5 floats.
    pub inner_body: Option<Vec<f64>>,
    /// Inner mind multipliers — 15 floats.
    pub inner_mind: Option<Vec<f64>>,
    /// Inner spirit content multipliers — 40 floats (observer dims [0:5]
    /// of inner_spirit_45d, absolute `[20:25]` of `_tensor`, NEVER modulated).
    pub inner_spirit_content: Option<Vec<f64>>,
    /// Outer body multipliers — 5 floats.
    pub outer_body: Option<Vec<f64>>,
    /// Outer mind multipliers — 15 floats.
    pub outer_mind: Option<Vec<f64>>,
    /// Outer spirit content multipliers — 40 floats (observer dims [0:5]
    /// of outer_spirit_45d, absolute `[85:90]` of `_tensor`, NEVER modulated).
    pub outer_spirit_content: Option<Vec<f64>>,
}

/// Resonance snapshot passed to `advance()` per Python signature. Stored
/// inside `GreatEpoch` for on-chain anchoring + audit.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResonanceSnapshot {
    /// Total GREAT PULSE counter from `ResonanceDetector` (1-indexed).
    pub great_pulse_count: u64,
    /// Per-pair big_pulse_count at the moment of GREAT PULSE.
    pub pair_big_pulse_counts: std::collections::BTreeMap<String, u64>,
    /// Wall clock seconds at GREAT PULSE.
    pub ts: f64,
}

/// One GreatEpoch — immutable record of a single forward step in Titan's
/// subjective time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GreatEpoch {
    /// Monotonic epoch counter (1-indexed).
    pub epoch_id: u64,
    /// Wall clock seconds at creation.
    pub timestamp: f64,
    /// 130D snapshot of the SPIRIT tensor at this moment.
    pub spirit_tensor: Vec<f64>,
    /// L2 magnitude of `spirit_tensor` — pre-computed for velocity calc.
    pub magnitude: f64,
    /// Velocity at the moment of crystallization.
    pub velocity: f64,
    /// True after `compute_enrichment` has been called.
    pub enrichment_sent: bool,
    /// Resonance snapshot passed to `advance()`.
    pub resonance_snapshot: ResonanceSnapshot,
    /// On-chain anchor hash — populated post-anchoring (empty for now).
    pub anchor_hash: String,
    /// Crystallized cumulative quality at GREAT PULSE.
    pub cumulative_quality: f64,
    /// Crystallized micro_tick_count at GREAT PULSE.
    pub micro_tick_count: u64,
}

impl GreatEpoch {
    /// Construct from current tensor + velocity + resonance snapshot.
    pub fn new(
        epoch_id: u64,
        spirit_tensor: &[f64; SPIRIT_DIMS],
        velocity: f64,
        resonance_snapshot: ResonanceSnapshot,
    ) -> Self {
        let tensor_vec = spirit_tensor.to_vec();
        let magnitude = tensor_magnitude(&tensor_vec);
        Self {
            epoch_id,
            timestamp: wall_seconds(),
            spirit_tensor: tensor_vec,
            magnitude,
            velocity,
            enrichment_sent: false,
            resonance_snapshot,
            anchor_hash: String::new(),
            cumulative_quality: 0.0,
            micro_tick_count: 0,
        }
    }
}

/// Per-component enrichment payload — output of `compute_enrichment`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnrichmentReward {
    /// Reward = base × balance × velocity × quality.
    pub reward: f64,
    /// 1.0 = perfect center, 0.0 = max imbalance.
    pub balance_score: f64,
    /// `clamp(velocity, [0.5, 2.0])`.
    pub velocity_bonus: f64,
    /// `1.0 + cumulative_quality / 100`, capped at 2.0.
    pub quality_bonus: f64,
}

/// Persistent state schema for `data/unified_spirit_state.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedSpiritStateFile {
    /// Schema version — `UNIFIED_SPIRIT_STATE_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// Live 130D tensor.
    pub tensor: Vec<f64>,
    /// Latest epoch_id (== number of advances).
    pub current_epoch_id: u64,
    /// Velocity computed at last advance.
    pub current_velocity: f64,
    /// Whether SPIRIT was stale at last advance.
    pub is_stale: bool,
    /// Consecutive stale advances.
    pub consecutive_stale: u32,
    /// Cumulative quality since last reset.
    pub cumulative_quality: f64,
    /// Micro-tick count since last reset.
    pub micro_tick_count: u64,
    /// Last alignment scalar from `micro_enrich`.
    pub last_alignment: f64,
    /// Full epoch history (capped to `UNIFIED_SPIRIT_EPOCHS_HISTORY_CAP`).
    pub epochs: Vec<GreatEpoch>,
}

/// Configuration mirroring Python `cfg` dict.
#[derive(Debug, Clone)]
pub struct UnifiedSpiritConfig {
    /// Velocity below this = STALE (Python `stale_threshold`).
    pub stale_threshold: f64,
    /// Base reward per GREAT PULSE component.
    pub enrichment_base: f64,
    /// Base FOCUS multiplier when STALE.
    pub stale_focus_multiplier: f64,
    /// Velocity rolling window size.
    pub velocity_window: usize,
    /// micro_enrich quality_factor coefficient.
    pub enrichment_rate: f64,
    /// Min alignment for enrichment.
    pub min_alignment_threshold: f64,
    /// In-memory epoch history cap (rotates oldest into archive when reached).
    pub epochs_history_cap: usize,
}

impl Default for UnifiedSpiritConfig {
    fn default() -> Self {
        use titan_core::constants::{
            UNIFIED_SPIRIT_EPOCHS_HISTORY_CAP, UNIFIED_SPIRIT_STALE_FOCUS_MULTIPLIER,
            UNIFIED_SPIRIT_STALE_THRESHOLD, UNIFIED_SPIRIT_VELOCITY_WINDOW,
        };
        Self {
            stale_threshold: UNIFIED_SPIRIT_STALE_THRESHOLD,
            enrichment_base: DEFAULT_ENRICHMENT_BASE,
            stale_focus_multiplier: UNIFIED_SPIRIT_STALE_FOCUS_MULTIPLIER,
            velocity_window: UNIFIED_SPIRIT_VELOCITY_WINDOW as usize,
            enrichment_rate: DEFAULT_ENRICHMENT_RATE,
            min_alignment_threshold: DEFAULT_MIN_ALIGNMENT_THRESHOLD,
            epochs_history_cap: UNIFIED_SPIRIT_EPOCHS_HISTORY_CAP as usize,
        }
    }
}

/// The Unified SPIRIT — "I AM" / "EXIST". 130D tensor + GreatEpoch
/// history + velocity tracking + enrichment lifecycle.
pub struct UnifiedSpirit {
    config: UnifiedSpiritConfig,
    state_path: PathBuf,
    tensor: [f64; SPIRIT_DIMS],
    epochs: Vec<GreatEpoch>,
    current_epoch_id: u64,
    current_velocity: f64,
    is_stale: bool,
    consecutive_stale: u32,
    cumulative_quality: f64,
    micro_tick_count: u64,
    last_alignment: f64,
}

impl UnifiedSpirit {
    /// Construct with SPEC defaults; loads state from
    /// `<data_dir>/unified_spirit_state.json` (with `.bak` / `.bak.prev`
    /// fallback per SPEC §11.H.4).
    pub fn with_defaults(data_dir: &Path) -> Result<Self, UnifiedSpiritError> {
        Self::new(UnifiedSpiritConfig::default(), data_dir)
    }

    /// Construct with explicit config (test path).
    pub fn new(config: UnifiedSpiritConfig, data_dir: &Path) -> Result<Self, UnifiedSpiritError> {
        let state_path = data_dir.join("unified_spirit_state.json");
        let mut spirit = Self {
            config,
            state_path,
            tensor: [TENSOR_INIT_VALUE; SPIRIT_DIMS],
            epochs: Vec::new(),
            current_epoch_id: 0,
            current_velocity: 1.0,
            is_stale: false,
            consecutive_stale: 0,
            cumulative_quality: 0.0,
            micro_tick_count: 0,
            last_alignment: 0.0,
        };
        // Best-effort load — propagate Err so caller can decide on halt.
        match spirit.load_state() {
            Ok(_) | Err(UnifiedSpiritError::Io(_)) => {} // Ok or "no file" → fresh start
            Err(e) => return Err(e),
        }
        info!(
            event = "UNIFIED_SPIRIT_INIT",
            epoch_count = spirit.epochs.len(),
            current_velocity = spirit.current_velocity,
            is_stale = spirit.is_stale,
            "unified_spirit initialized"
        );
        Ok(spirit)
    }

    // ── Tensor updates ─────────────────────────────────────────────────

    /// Update Inner Trinity feed. Optional `mults` applies V5 multipliers
    /// to body[0:5] + mind[5:20] + spirit_content[25:65]; observer
    /// dims [20:25] are NEVER modulated.
    pub fn update_subconscious(
        &mut self,
        inner_body: &[f64],
        inner_mind: &[f64],
        inner_spirit: &[f64],
        mults: Option<&V5Multipliers>,
    ) {
        copy_padded(&mut self.tensor[0..5], inner_body, TENSOR_INIT_VALUE);
        copy_padded(&mut self.tensor[5..20], inner_mind, TENSOR_INIT_VALUE);
        copy_padded(&mut self.tensor[20..65], inner_spirit, TENSOR_INIT_VALUE);
        if let Some(m) = mults {
            self.apply_filter_down_v5_inner(m);
        }
    }

    /// Update Outer Trinity feed. Optional `mults` applies V5 multipliers
    /// to body[65:70] + mind[70:85] + spirit_content[90:130]; observer
    /// dims [85:90] are NEVER modulated.
    pub fn update_conscious(
        &mut self,
        outer_body: &[f64],
        outer_mind: &[f64],
        outer_spirit: &[f64],
        mults: Option<&V5Multipliers>,
    ) {
        copy_padded(&mut self.tensor[65..70], outer_body, TENSOR_INIT_VALUE);
        copy_padded(&mut self.tensor[70..85], outer_mind, TENSOR_INIT_VALUE);
        copy_padded(&mut self.tensor[85..130], outer_spirit, TENSOR_INIT_VALUE);
        if let Some(m) = mults {
            self.apply_filter_down_v5_outer(m);
        }
    }

    fn apply_filter_down_v5_inner(&mut self, mults: &V5Multipliers) {
        if let Some(ib) = &mults.inner_body {
            if ib.len() == 5 {
                for (i, m) in ib.iter().enumerate() {
                    self.tensor[i] = clamp01(self.tensor[i] * m);
                }
            }
        }
        if let Some(im) = &mults.inner_mind {
            if im.len() == 15 {
                for (i, m) in im.iter().enumerate() {
                    self.tensor[5 + i] = clamp01(self.tensor[5 + i] * m);
                }
            }
        }
        if let Some(isc) = &mults.inner_spirit_content {
            if isc.len() == 40 {
                for (i, m) in isc.iter().enumerate() {
                    self.tensor[25 + i] = clamp01(self.tensor[25 + i] * m);
                }
            }
        }
        // Observer dims [20:25] are NEVER touched.
    }

    fn apply_filter_down_v5_outer(&mut self, mults: &V5Multipliers) {
        if let Some(ob) = &mults.outer_body {
            if ob.len() == 5 {
                for (i, m) in ob.iter().enumerate() {
                    self.tensor[65 + i] = clamp01(self.tensor[65 + i] * m);
                }
            }
        }
        if let Some(om) = &mults.outer_mind {
            if om.len() == 15 {
                for (i, m) in om.iter().enumerate() {
                    self.tensor[70 + i] = clamp01(self.tensor[70 + i] * m);
                }
            }
        }
        if let Some(osc) = &mults.outer_spirit_content {
            if osc.len() == 40 {
                for (i, m) in osc.iter().enumerate() {
                    self.tensor[90 + i] = clamp01(self.tensor[90 + i] * m);
                }
            }
        }
        // Observer dims [85:90] are NEVER touched.
    }

    // ── Enrichment ─────────────────────────────────────────────────────

    /// Continuous enrichment. Returns alignment scalar. Geometric blend
    /// of `tensor` toward `realtime_state` weighted by quality_factor =
    /// `max(0, alignment) * enrichment_rate`.
    ///
    /// Accepts state widths 30 (legacy), 65 (inner-only), or 130 (full).
    /// Anything else returns 0.0 (no-op).
    pub fn micro_enrich(&mut self, realtime_state: &[f64]) -> f64 {
        let n = match realtime_state.len() {
            SPIRIT_DIMS => SPIRIT_DIMS,
            INNER_DIMS => INNER_DIMS,
            30 => 30, // Legacy 30D
            _ => return 0.0,
        };

        let spirit_slice = &self.tensor[0..n];
        let mut dot = 0.0;
        let mut mag_s = 0.0;
        let mut mag_r = 0.0;
        for i in 0..n {
            dot += spirit_slice[i] * realtime_state[i];
            mag_s += spirit_slice[i] * spirit_slice[i];
            mag_r += realtime_state[i] * realtime_state[i];
        }
        let mag_s = mag_s.sqrt();
        let mag_r = mag_r.sqrt();

        if mag_s < 1e-8 || mag_r < 1e-8 {
            return 0.0;
        }

        let alignment = dot / (mag_s * mag_r);
        self.last_alignment = alignment;
        self.micro_tick_count += 1;

        if alignment < self.config.min_alignment_threshold {
            return alignment;
        }

        let quality_factor = alignment.max(0.0) * self.config.enrichment_rate;

        // Geometric blend: spirit^(1-qf) * state^qf
        for (i, r_val) in realtime_state.iter().take(n).enumerate() {
            let s = self.tensor[i].max(0.001);
            let r = r_val.max(0.001);
            self.tensor[i] = s.powf(1.0 - quality_factor) * r.powf(quality_factor);
        }

        self.cumulative_quality += alignment.max(0.0);
        alignment
    }

    /// Reset cumulative quality (called at GREAT PULSE — also done internally
    /// in `advance`). Returns quality before reset.
    pub fn reset_quality(&mut self) -> f64 {
        let q = self.cumulative_quality;
        self.cumulative_quality = 0.0;
        q
    }

    // ── GREAT PULSE → advance ──────────────────────────────────────────

    /// Crystallize a GreatEpoch. Auto-persists state to disk per SPEC
    /// §11.H ("these are precious"). Returns the new epoch.
    ///
    /// Caller (the bus dispatch loop in `boot::run_bus_dispatch_loop`)
    /// invokes this when `BigPulse::great_pulse_ready == true`.
    pub fn advance(
        &mut self,
        resonance_snapshot: ResonanceSnapshot,
    ) -> Result<GreatEpoch, UnifiedSpiritError> {
        self.current_epoch_id += 1;

        let velocity = self.compute_velocity_internal();
        self.current_velocity = velocity;

        let crystallized_quality = self.cumulative_quality;
        let crystallized_ticks = self.micro_tick_count;

        let mut epoch = GreatEpoch::new(
            self.current_epoch_id,
            &self.tensor,
            velocity,
            resonance_snapshot,
        );
        epoch.cumulative_quality = crystallized_quality;
        epoch.micro_tick_count = crystallized_ticks;
        self.epochs.push(epoch.clone());

        // Cap history per SPEC v0.1.3 UNIFIED_SPIRIT_EPOCHS_HISTORY_CAP
        if self.epochs.len() > self.config.epochs_history_cap {
            let drop_count = self.epochs.len() - self.config.epochs_history_cap;
            self.epochs.drain(0..drop_count);
        }

        // Reset quality for new cycle
        self.cumulative_quality = 0.0;
        self.micro_tick_count = 0;

        // Stale tracking
        if velocity < self.config.stale_threshold && self.current_epoch_id > 1 {
            self.is_stale = true;
            self.consecutive_stale += 1;
        } else {
            self.is_stale = false;
            self.consecutive_stale = 0;
        }

        info!(
            event = "GREAT_EPOCH",
            epoch_id = self.current_epoch_id,
            magnitude = epoch.magnitude,
            velocity = velocity,
            is_stale = self.is_stale,
            "GREAT EPOCH crystallized"
        );

        // Auto-persist — these are precious
        self.save_state()?;

        Ok(epoch)
    }

    fn compute_velocity_internal(&self) -> f64 {
        let current_mag = tensor_magnitude_arr(&self.tensor);
        if self.epochs.is_empty() {
            return 1.0;
        }
        let window_size = self.config.velocity_window.min(self.epochs.len());
        let start = self.epochs.len() - window_size;
        let avg_mag: f64 = self.epochs[start..]
            .iter()
            .map(|e| e.magnitude)
            .sum::<f64>()
            / window_size as f64;
        if avg_mag < 1e-8 {
            return 1.0;
        }
        current_mag / avg_mag
    }

    /// Current growth velocity.
    pub fn velocity(&self) -> f64 {
        self.current_velocity
    }

    /// Whether SPIRIT is currently stale.
    pub fn is_stale(&self) -> bool {
        self.is_stale
    }

    /// FOCUS cascade multiplier — escalates with consecutive stale cycles,
    /// capped at 3.0. Returns 1.0 when not stale.
    pub fn stale_focus_multiplier(&self) -> f64 {
        if !self.is_stale {
            return 1.0;
        }
        let escalation = 1.0 + STALE_FOCUS_ESCALATION_PER_CYCLE * self.consecutive_stale as f64;
        STALE_FOCUS_MULTIPLIER_CAP.min(self.config.stale_focus_multiplier * escalation)
    }

    // ── Enrichment computation ─────────────────────────────────────────

    /// Compute per-component enrichment rewards from the latest GreatEpoch.
    /// Sets `enrichment_sent = true` on the epoch. Returns empty map if
    /// no epochs.
    pub fn compute_enrichment(
        &mut self,
    ) -> std::collections::BTreeMap<&'static str, EnrichmentReward> {
        let mut out = std::collections::BTreeMap::new();
        let latest = match self.epochs.last_mut() {
            Some(e) => e,
            None => return out,
        };

        let tensor = &latest.spirit_tensor;
        // 130D layout
        if tensor.len() < SPIRIT_DIMS {
            return out;
        }

        let components: [(&'static str, &[f64]); 6] = [
            ("inner_body", &tensor[0..5]),
            ("inner_mind", &tensor[5..20]),
            ("inner_spirit", &tensor[20..65]),
            ("outer_body", &tensor[65..70]),
            ("outer_mind", &tensor[70..85]),
            ("outer_spirit", &tensor[85..130]),
        ];

        let velocity_bonus = latest.velocity.clamp(0.5, 2.0);
        let quality_bonus = (1.0_f64 + latest.cumulative_quality / 100.0).min(2.0);

        for (name, slice) in components {
            let avg_delta: f64 =
                slice.iter().map(|v| (v - 0.5).abs()).sum::<f64>() / slice.len() as f64;
            let balance_score = (1.0 - avg_delta * 2.0).max(0.0);
            let reward = DEFAULT_ENRICHMENT_BASE * balance_score * velocity_bonus * quality_bonus;
            out.insert(
                name,
                EnrichmentReward {
                    reward,
                    balance_score,
                    velocity_bonus,
                    quality_bonus,
                },
            );
        }

        latest.enrichment_sent = true;
        out
    }

    // ── Getters ────────────────────────────────────────────────────────

    /// Current 130D tensor as a Vec.
    pub fn tensor(&self) -> Vec<f64> {
        self.tensor.to_vec()
    }

    /// Inner Trinity 65D slice.
    pub fn inner_tensor(&self) -> Vec<f64> {
        self.tensor[..INNER_DIMS].to_vec()
    }

    /// Outer Trinity 65D slice.
    pub fn outer_tensor(&self) -> Vec<f64> {
        self.tensor[INNER_DIMS..].to_vec()
    }

    /// Total GREAT EPOCHs (Titan's subjective age).
    pub fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    /// Latest GREAT EPOCH (None on fresh boot).
    pub fn latest_epoch(&self) -> Option<&GreatEpoch> {
        self.epochs.last()
    }

    /// Get a specific GREAT EPOCH by ID. O(N) scan — N capped at 4096.
    pub fn get_epoch(&self, epoch_id: u64) -> Option<&GreatEpoch> {
        self.epochs.iter().find(|e| e.epoch_id == epoch_id)
    }

    /// Read the live cumulative_quality scalar.
    pub fn cumulative_quality(&self) -> f64 {
        self.cumulative_quality
    }

    /// Read the live micro_tick_count.
    pub fn micro_tick_count(&self) -> u64 {
        self.micro_tick_count
    }

    /// Read the last alignment from `micro_enrich`.
    pub fn last_alignment(&self) -> f64 {
        self.last_alignment
    }

    /// Phase B: msgpack-serializable snapshot for `unified_spirit_metadata.bin`
    /// SHM slot (rFP_phase_c_state_read_unification §B / D-SPEC-72). Mirrors
    /// Python `UnifiedSpirit.get_stats()` 1:1 — pre-Phase-B the Python wrapper
    /// computed this against an in-Python instance of UnifiedSpirit; post
    /// Phase B the Rust instance is the canonical source.
    pub fn get_stats(&self, ts: f64) -> crate::metadata_publisher::UnifiedSpiritMetadata {
        use crate::metadata_publisher::{UnifiedSpiritConfigSummary, UnifiedSpiritMetadata};
        use titan_core::constants::UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION;

        // Python parity rounds — match precision exactly so consumers
        // post-ownership-flip see byte-identical msgpack output.
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let r6 = |v: f64| (v * 1e6).round() / 1e6;

        let tensor_mag = r4(tensor_magnitude_arr(&self.tensor));
        let tensor_sum: f64 = r4(self.tensor.iter().sum());

        // Clone full GreatEpoch (all 10 fields) — matches Python
        // `GreatEpoch.to_dict()` 1:1 (no simplification).
        let latest_epoch = self.epochs.last().cloned();

        UnifiedSpiritMetadata {
            epoch_count: self.epochs.len(),
            current_epoch_id: self.current_epoch_id,
            velocity: r4(self.current_velocity),
            is_stale: self.is_stale,
            consecutive_stale: self.consecutive_stale,
            stale_focus_multiplier: r4(self.stale_focus_multiplier()),
            tensor_magnitude: tensor_mag,
            tensor_sum,
            latest_epoch,
            cumulative_quality: r4(self.cumulative_quality),
            micro_tick_count: self.micro_tick_count,
            last_alignment: r4(self.last_alignment),
            enrichment_rate: self.config.enrichment_rate,
            full_130dt: self.tensor.iter().map(|v| r6(*v)).collect(),
            config: UnifiedSpiritConfigSummary {
                stale_threshold: self.config.stale_threshold,
                enrichment_base: self.config.enrichment_base,
                velocity_window: self.config.velocity_window,
                enrichment_rate: self.config.enrichment_rate,
                min_alignment_threshold: self.config.min_alignment_threshold,
            },
            schema_version: UNIFIED_SPIRIT_METADATA_SCHEMA_VERSION as u32,
            ts,
        }
    }

    // ── Persistence ────────────────────────────────────────────────────

    /// Persist state to `data/unified_spirit_state.json` via atomic_write
    /// + 2-backup retention per SPEC §11.H.1.
    pub fn save_state(&self) -> Result<(), UnifiedSpiritError> {
        let state = UnifiedSpiritStateFile {
            schema_version: UNIFIED_SPIRIT_STATE_SCHEMA_VERSION,
            tensor: self.tensor.to_vec(),
            current_epoch_id: self.current_epoch_id,
            current_velocity: self.current_velocity,
            is_stale: self.is_stale,
            consecutive_stale: self.consecutive_stale,
            cumulative_quality: self.cumulative_quality,
            micro_tick_count: self.micro_tick_count,
            last_alignment: self.last_alignment,
            epochs: self.epochs.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&state)?;
        titan_core::atomic_write::atomic_write(
            &self.state_path,
            &bytes,
            titan_core::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load state, falling back to `.bak` then `.bak.prev` per SPEC §11.H.4.
    /// G12 monotonic check: refuse-load if loaded `current_epoch_id` is less
    /// than current in-memory state.
    pub fn load_state(&mut self) -> Result<(), UnifiedSpiritError> {
        let candidates = [
            self.state_path.clone(),
            self.state_path.with_extension("json.bak"),
            self.state_path.with_extension("json.bak.prev"),
        ];

        let mut found_io_or_decode = false;
        for candidate in &candidates {
            if !candidate.exists() {
                continue;
            }
            match std::fs::read(candidate) {
                Ok(bytes) => match serde_json::from_slice::<UnifiedSpiritStateFile>(&bytes) {
                    Ok(state) => {
                        if state.schema_version != UNIFIED_SPIRIT_STATE_SCHEMA_VERSION {
                            warn!(
                                event = "UNIFIED_SPIRIT_SCHEMA_MISMATCH",
                                loaded = state.schema_version,
                                expected = UNIFIED_SPIRIT_STATE_SCHEMA_VERSION,
                                ?candidate,
                                "schema mismatch; trying next backup"
                            );
                            found_io_or_decode = true;
                            continue;
                        }
                        // G12 monotonic guard: loaded epoch_id must be ≥
                        // any prior in-memory epoch_id (rare — only
                        // matters if load_state is called twice).
                        if state.current_epoch_id < self.current_epoch_id {
                            return Err(UnifiedSpiritError::G12Violation {
                                loaded: state.current_epoch_id,
                                in_memory: self.current_epoch_id,
                            });
                        }
                        // Apply state
                        if state.tensor.len() == SPIRIT_DIMS {
                            for (i, v) in state.tensor.iter().enumerate() {
                                self.tensor[i] = *v;
                            }
                        }
                        self.current_epoch_id = state.current_epoch_id;
                        self.current_velocity = state.current_velocity;
                        self.is_stale = state.is_stale;
                        self.consecutive_stale = state.consecutive_stale;
                        self.cumulative_quality = state.cumulative_quality;
                        self.micro_tick_count = state.micro_tick_count;
                        self.last_alignment = state.last_alignment;
                        self.epochs = state.epochs;
                        return Ok(());
                    }
                    Err(e) => {
                        warn!(
                            event = "UNIFIED_SPIRIT_DECODE_FAIL",
                            ?candidate,
                            err = ?e,
                            "decode failed; trying next backup"
                        );
                        found_io_or_decode = true;
                    }
                },
                Err(e) => {
                    warn!(
                        event = "UNIFIED_SPIRIT_IO_FAIL",
                        ?candidate,
                        err = ?e,
                        "io failed; trying next backup"
                    );
                    found_io_or_decode = true;
                }
            }
        }

        if found_io_or_decode {
            // We saw at least one candidate but all failed.
            return Err(UnifiedSpiritError::AllBackupsCorrupt);
        }
        // No file found — fresh start.
        Err(UnifiedSpiritError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no state file present (clean start)",
        )))
    }
}

// ── Utility ───────────────────────────────────────────────────────────

fn copy_padded(dst: &mut [f64], src: &[f64], pad: f64) {
    let n = dst.len();
    if src.len() >= n {
        dst.copy_from_slice(&src[..n]);
    } else {
        dst[..src.len()].copy_from_slice(src);
        for v in &mut dst[src.len()..] {
            *v = pad;
        }
    }
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

fn tensor_magnitude(tensor: &[f64]) -> f64 {
    tensor.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn tensor_magnitude_arr<const N: usize>(tensor: &[f64; N]) -> f64 {
    tensor.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn wall_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fresh(dir: &Path) -> UnifiedSpirit {
        UnifiedSpirit::with_defaults(dir).unwrap_or_else(|e| match e {
            UnifiedSpiritError::Io(_) => UnifiedSpirit::new(UnifiedSpiritConfig::default(), dir)
                .unwrap_or_else(|_| panic!("fresh init")),
            other => panic!("fresh failed: {other:?}"),
        })
    }

    #[test]
    fn great_epoch_serializes_round_trip() {
        // C4-2b2 test 1: GreatEpoch JSON round-trip preserves all fields
        let snapshot = ResonanceSnapshot {
            great_pulse_count: 5,
            pair_big_pulse_counts: [
                ("body".to_string(), 3),
                ("mind".to_string(), 4),
                ("spirit".to_string(), 5),
            ]
            .into_iter()
            .collect(),
            ts: 1234567890.5,
        };
        let tensor = [0.7_f64; SPIRIT_DIMS];
        let mut epoch = GreatEpoch::new(7, &tensor, 1.2, snapshot);
        epoch.cumulative_quality = 4.2;
        epoch.micro_tick_count = 100;
        let json = serde_json::to_string(&epoch).unwrap();
        let decoded: GreatEpoch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, epoch);
    }

    #[test]
    fn magnitude_correct() {
        // C4-2b2 test 2: L2 magnitude formula
        let v = vec![3.0_f64, 4.0]; // 3-4-5 triangle
        assert!((tensor_magnitude(&v) - 5.0).abs() < 1e-10);
        let zeros = vec![0.0_f64; 130];
        assert_eq!(tensor_magnitude(&zeros), 0.0);
    }

    #[test]
    fn advance_increments_epoch_id_and_crystallizes_quality() {
        // C4-2b2 test 3: advance increments + resets quality + saves
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        spirit.cumulative_quality = 4.2;
        spirit.micro_tick_count = 99;
        let snapshot = ResonanceSnapshot::default();
        let epoch = spirit.advance(snapshot).unwrap();
        assert_eq!(epoch.epoch_id, 1);
        assert!((epoch.cumulative_quality - 4.2).abs() < 1e-10);
        assert_eq!(epoch.micro_tick_count, 99);
        // Reset post-advance
        assert_eq!(spirit.cumulative_quality, 0.0);
        assert_eq!(spirit.micro_tick_count, 0);
    }

    #[test]
    fn velocity_uses_rolling_window() {
        // C4-2b2 test 4: velocity = current_mag / avg(last N magnitudes)
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        // Fake 3 prior epochs by directly inserting (test path)
        for i in 1..=3 {
            let snapshot = ResonanceSnapshot::default();
            let _ = spirit.advance(snapshot).unwrap();
            spirit.tensor[0] = 0.5 + (i as f64) * 0.01;
        }
        // After 3 advances, velocity should be > 0
        assert!(spirit.velocity() > 0.0);
        assert!(spirit.epochs.len() == 3);
    }

    #[test]
    fn first_epoch_velocity_is_one() {
        // C4-2b2 test 5: no history → velocity = 1.0 (neutral)
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        let epoch = spirit.advance(ResonanceSnapshot::default()).unwrap();
        assert_eq!(epoch.velocity, 1.0);
    }

    #[test]
    fn stale_tracking_transitions_and_escalation() {
        // C4-2b2 test 6: stale flips on velocity below threshold +
        // escalation cap at 3.0
        let dir = tempdir().unwrap();
        let cfg = UnifiedSpiritConfig {
            stale_threshold: 0.95, // High threshold = easy to be stale
            stale_focus_multiplier: 1.5,
            velocity_window: 10,
            ..UnifiedSpiritConfig::default()
        };
        let mut spirit = UnifiedSpirit::new(cfg, dir.path()).unwrap();

        // First advance — no stale (epoch_id == 1, gate disables stale)
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        assert!(!spirit.is_stale());
        assert_eq!(spirit.stale_focus_multiplier(), 1.0);

        // Subsequent advances with constant tensor → velocity = 1.0
        // not < 0.95 threshold → not stale.
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        // Still not stale
        assert!(!spirit.is_stale());

        // Force stale by raising threshold to 1.5 (always triggers)
        spirit.config.stale_threshold = 1.5;
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        assert!(spirit.is_stale());
        assert!(spirit.stale_focus_multiplier() >= 1.5);

        // Multiple stale cycles → escalation
        for _ in 0..20 {
            let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        }
        // Cap at 3.0
        assert!(spirit.stale_focus_multiplier() <= STALE_FOCUS_MULTIPLIER_CAP + 1e-10);
    }

    #[test]
    fn advance_invoked_on_great_pulse_via_callback_pattern() {
        // C4-2b2 test 7: integration pattern — when
        // ResonanceDetector.record_pulse_with_phase returns BigPulse with
        // great_pulse_ready=true, caller invokes spirit.advance(...).
        // Verifies the contract: spirit's epoch_count grows accordingly.
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        // Simulate 3 GREAT PULSEs
        for _ in 0..3 {
            let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        }
        assert_eq!(spirit.epoch_count(), 3);
        assert_eq!(spirit.current_epoch_id, 3);
    }

    #[test]
    fn g12_monotonic_violation_detected_on_load() {
        // C4-2b2 test 8: loaded epoch_id < in-memory epoch_id → G12Violation
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap(); // epoch_id = 1
        spirit.save_state().unwrap();
        // Force in-memory epoch_id = 5 (higher than what disk has)
        spirit.current_epoch_id = 5;
        let result = spirit.load_state();
        // Loading should refuse since on-disk (1) < in-memory (5).
        assert!(matches!(
            result,
            Err(UnifiedSpiritError::G12Violation { .. })
        ));
    }

    #[test]
    fn auto_persist_on_advance() {
        // C4-2b2 test 9: advance() writes state file
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        let path = dir.path().join("unified_spirit_state.json");
        assert!(!path.exists());
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap();
        assert!(path.exists(), "state file persisted after advance");
        // Reload and verify epoch_id round-trip
        let mut spirit2 = fresh(dir.path());
        // (with_defaults already loaded)
        assert_eq!(spirit2.epoch_count(), 1);
        assert_eq!(spirit2.current_epoch_id, 1);
        // No further side effects
        let _ = spirit2.advance(ResonanceSnapshot::default()).unwrap();
        assert_eq!(spirit2.current_epoch_id, 2);
    }

    #[test]
    fn restore_from_bak_when_canonical_corrupt() {
        // C4-2b2 test 10: corrupt canonical → load from .bak
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap(); // 1
        spirit.save_state().unwrap();
        let _ = spirit.advance(ResonanceSnapshot::default()).unwrap(); // 2
        spirit.save_state().unwrap();

        let canonical = dir.path().join("unified_spirit_state.json");
        let bak = dir.path().join("unified_spirit_state.json.bak");
        assert!(canonical.exists());
        assert!(bak.exists());
        // Corrupt canonical
        std::fs::write(&canonical, b"not json").unwrap();

        // Fresh detector — should fall through to .bak
        let spirit2 = fresh(dir.path());
        // Recovered some prior state
        assert!(spirit2.epoch_count() >= 1);
    }

    #[test]
    fn all_backups_corrupt_returns_error() {
        // C4-2b2 test 11: all 3 candidates corrupt → AllBackupsCorrupt
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("unified_spirit_state.json"), b"not json").unwrap();
        std::fs::write(
            dir.path().join("unified_spirit_state.json.bak"),
            b"also bad",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("unified_spirit_state.json.bak.prev"),
            b"also worse",
        )
        .unwrap();

        let cfg = UnifiedSpiritConfig::default();
        let result = UnifiedSpirit::new(cfg, dir.path());
        assert!(matches!(result, Err(UnifiedSpiritError::AllBackupsCorrupt)));
    }

    #[test]
    fn compute_enrichment_output_schema() {
        // C4-2b2 test 12: per-component reward map + balance/velocity/quality
        // bonus computed correctly
        let dir = tempdir().unwrap();
        let mut spirit = fresh(dir.path());
        // Set tensor to all-0.5 (perfectly balanced)
        for v in &mut spirit.tensor {
            *v = 0.5;
        }
        spirit.cumulative_quality = 50.0; // → quality_bonus = 1.5
        let epoch = spirit.advance(ResonanceSnapshot::default()).unwrap();
        assert!((epoch.magnitude - tensor_magnitude(&[0.5_f64; SPIRIT_DIMS])).abs() < 1e-10);

        let enrichment = spirit.compute_enrichment();
        assert_eq!(enrichment.len(), 6);
        for &name in &[
            "inner_body",
            "inner_mind",
            "inner_spirit",
            "outer_body",
            "outer_mind",
            "outer_spirit",
        ] {
            let r = enrichment.get(name).expect("component present");
            // Perfect balance → balance_score = 1.0
            assert!((r.balance_score - 1.0).abs() < 1e-10, "{name}");
            // Velocity bonus = clamp(velocity, 0.5..=2.0). First epoch
            // velocity = 1.0 → velocity_bonus = 1.0.
            assert_eq!(r.velocity_bonus, 1.0);
            // quality_bonus = min(2.0, 1.0 + 50.0/100.0) = 1.5
            assert!((r.quality_bonus - 1.5).abs() < 1e-10);
            // reward = 0.02 * 1.0 * 1.0 * 1.5 = 0.03
            assert!((r.reward - 0.03).abs() < 1e-10);
        }
        // Latest epoch enrichment_sent flag flipped
        assert!(spirit.latest_epoch().unwrap().enrichment_sent);
    }
}
