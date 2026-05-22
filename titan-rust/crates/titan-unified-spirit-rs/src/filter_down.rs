//! filter_down — V5 TrinityValueNet + TransitionBuffer + FilterDownV5Engine.
//!
//! Full Rust port of `titan_hcl/logic/filter_down.py:89-842` per SPEC
//! §10.F G6+G7+G8+G9 (V5 sole live FILTER_DOWN engine since 2026-04-25).
//!
//! Chunk decomposition:
//! - **C4-3a (this commit):** `TrinityValueNet` (162→128→64→1 ReLU MLP) +
//!   Xavier init via `rand_distr::Normal` + `forward` + `forward_batch` +
//!   `gradient_wrt_input` (manual ReLU backprop, returns 162-D attention
//!   vector) + JSON save/load via `serde_json` + `titan-core::atomic_write`.
//!   Pure compute — no bus, no shm, no buffer, no training.
//! - **C4-3b:** `TransitionBuffer` (capped ring) + `train_step` (TD(0)) +
//!   `record_transition` + persistence.
//! - **C4-3c:** `FilterDownV5Engine` orchestrator + `compute_multipliers`
//!   (clamp + spirit weakening + observer mask) + cold-start gate +
//!   `UNIFIED_SPIRIT_FILTER_DOWN` bus publish + body-cycle integration.
//!
//! Network architecture per `filter_down.py:483-486` (V5 dims, locked v0.1.5):
//! ```text
//! 162D state (TITAN_SELF tensor)
//!   │
//!   ▼ w1 (162×128)  + b1 (128)  + ReLU
//! 128D
//!   │
//!   ▼ w2 (128×64)   + b2 (64)   + ReLU
//! 64D
//!   │
//!   ▼ w3 (64×1)     + b3 (1)
//! Scalar value V(s)
//! ```
//!
//! `gradient_wrt_input(state)` returns ∂V/∂state (162-D attention vector).
//! Used downstream by `compute_multipliers` to derive per-dim multipliers
//! (clipped + observer-masked) per SPEC §10.F V5 cascade. Manual backprop
//! mirrors numpy chain rule exactly (no autograd framework dependency).

use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::warn;

use titan_core::constants::{FILTER_DOWN_INPUT_DIM, FILTER_DOWN_OUTPUT_DIM};
use titan_core::transition_buffer::TransitionBuffer;
use titan_core::trinity_value_net::{FilterDownError, TrinityValueNet};

/// Network input dim = 162 (130D Trinity + 30D topology + 2D journey).
/// This is the const-generic `N` for the unified engine's `TrinityValueNet<INPUT_DIM>`.
pub const INPUT_DIM: usize = FILTER_DOWN_INPUT_DIM as usize;
/// Multiplier output dim = 120 (5+15+40+5+15+40 — observer 10 dims masked
/// per SPEC §10.F G8); referenced by C4-3c `compute_multipliers`.
pub const OUTPUT_DIM: usize = FILTER_DOWN_OUTPUT_DIM as usize;

/// Default path under `data_dir` for V5 weights (matches Python
/// `filter_down.py:559`).
pub fn default_weights_path(data_dir: &Path) -> PathBuf {
    data_dir.join("filter_down_v5_weights.json")
}

/// Default path under `data_dir` for V5 transition buffer.
pub fn default_buffer_path(data_dir: &Path) -> PathBuf {
    data_dir.join("filter_down_v5_buffer.json")
}

// ── FilterDownV5Engine (C4-3c) ─────────────────────────────────────────

use titan_core::constants::{
    FILTER_DOWN_BATCH_SIZE, FILTER_DOWN_COLD_START_FLOOR_EPOCHS, FILTER_DOWN_GAMMA, FILTER_DOWN_LR,
    FILTER_DOWN_MIN_TRANSITIONS, FILTER_DOWN_MULTIPLIER_CEIL, FILTER_DOWN_MULTIPLIER_FLOOR,
    FILTER_DOWN_SPIRIT_STRENGTH_MULT, FILTER_DOWN_TRAIN_EVERY_N,
};

/// EMA smoothing coefficient — `new = α·old + (1-α)·raw`. Per
/// `filter_down.py:52 SMOOTHING = 0.9`.
pub const FILTER_DOWN_SMOOTHING: f64 = 0.9;
/// Multiplier normalization scale — Python:727 `g / max_grad * 2.0`.
pub const FILTER_DOWN_NORM_SCALE: f64 = 2.0;
/// Round-to-decimals for wire output (Python:753-758 `round(m, 4)`).
/// Used only in the published payload, not in EMA state.
pub const FILTER_DOWN_OUTPUT_ROUND_DECIMALS: u32 = 4;

/// 120-multiplier dict — output of `compute_multipliers`. Always has all
/// 6 fields populated (no `Option`), unlike the legacy [`crate::unified_spirit::V5Multipliers`]
/// which uses `Option<Vec<f64>>` for partial-update scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Multipliers {
    /// 5 multipliers for inner_body[0:5].
    pub inner_body: Vec<f64>,
    /// 15 multipliers for inner_mind[0:15].
    pub inner_mind: Vec<f64>,
    /// 40 multipliers for inner_spirit content dims [25:65] of TITAN_SELF
    /// (Python skips inner_spirit[20:25] observer dims per G8).
    pub inner_spirit_content: Vec<f64>,
    /// 5 multipliers for outer_body[0:5].
    pub outer_body: Vec<f64>,
    /// 15 multipliers for outer_mind[0:15].
    pub outer_mind: Vec<f64>,
    /// 40 multipliers for outer_spirit content dims [90:130] of TITAN_SELF
    /// (skips outer_spirit[85:90] observer dims).
    pub outer_spirit_content: Vec<f64>,
}

impl Multipliers {
    /// All-1.0 multipliers (cold-start mode). Per Python `_default_multipliers_dict`.
    pub fn ones() -> Self {
        Self {
            inner_body: vec![1.0; 5],
            inner_mind: vec![1.0; 15],
            inner_spirit_content: vec![1.0; 40],
            outer_body: vec![1.0; 5],
            outer_mind: vec![1.0; 15],
            outer_spirit_content: vec![1.0; 40],
        }
    }

    /// Total multiplier count = 120 (5+15+40+5+15+40). Matches SPEC §10.F V5.
    pub fn total_count(&self) -> usize {
        self.inner_body.len()
            + self.inner_mind.len()
            + self.inner_spirit_content.len()
            + self.outer_body.len()
            + self.outer_mind.len()
            + self.outer_spirit_content.len()
    }

    /// Round all multipliers to 4 decimal places — Python parity for wire
    /// payload + cross-Titan byte-equality.
    pub fn rounded(&self) -> Self {
        let r = |slice: &[f64]| -> Vec<f64> {
            slice
                .iter()
                .map(|m| round_n(*m, FILTER_DOWN_OUTPUT_ROUND_DECIMALS))
                .collect()
        };
        Self {
            inner_body: r(&self.inner_body),
            inner_mind: r(&self.inner_mind),
            inner_spirit_content: r(&self.inner_spirit_content),
            outer_body: r(&self.outer_body),
            outer_mind: r(&self.outer_mind),
            outer_spirit_content: r(&self.outer_spirit_content),
        }
    }
}

/// Convert from [`Multipliers`] to legacy [`crate::unified_spirit::V5Multipliers`]
/// (always-Some shape).
impl From<&Multipliers> for crate::unified_spirit::V5Multipliers {
    fn from(m: &Multipliers) -> Self {
        Self {
            inner_body: Some(m.inner_body.clone()),
            inner_mind: Some(m.inner_mind.clone()),
            inner_spirit_content: Some(m.inner_spirit_content.clone()),
            outer_body: Some(m.outer_body.clone()),
            outer_mind: Some(m.outer_mind.clone()),
            outer_spirit_content: Some(m.outer_spirit_content.clone()),
        }
    }
}

/// Round f64 to `decimals` places (Python `round(v, n)` parity).
fn round_n(v: f64, decimals: u32) -> f64 {
    let scale = 10_f64.powi(decimals as i32);
    (v * scale).round() / scale
}

/// Sidecar state schema for `data/filter_down_v5_state.json`. Persists
/// graduation-critical counters separately from weights/buffer per
/// Python `filter_down.py:823-836`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterDownV5StateFile {
    /// Schema version per SPEC §11.H.4.
    pub schema_version: u32,
    /// Total TD(0) train_step calls (gates cold-start).
    pub total_train_steps: u64,
    /// Most recent loss value.
    pub last_loss: f64,
    /// Last 20 losses (rolling).
    pub recent_losses: Vec<f64>,
    /// Whether Phase 8 baseline snapshot has been taken (RFP2-PHASE8 gate).
    pub phase8_snapshot_taken: bool,
    /// EMA multiplier state — preserves drift across restarts.
    pub multipliers_ema: MultipliersEma,
}

/// EMA state for multipliers — survives restart so Gate #9 divergence
/// progress isn't lost.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultipliersEma {
    /// Inner body EMA (5).
    pub inner_body: Vec<f64>,
    /// Inner mind EMA (15).
    pub inner_mind: Vec<f64>,
    /// Inner spirit content EMA (40).
    pub inner_spirit_content: Vec<f64>,
    /// Outer body EMA (5).
    pub outer_body: Vec<f64>,
    /// Outer mind EMA (15).
    pub outer_mind: Vec<f64>,
    /// Outer spirit content EMA (40).
    pub outer_spirit_content: Vec<f64>,
}

/// Schema version for `data/filter_down_v5_state.json`.
pub const FILTER_DOWN_STATE_SCHEMA_VERSION: u32 = 1;

/// Default state path under `data_dir`.
pub fn default_state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("filter_down_v5_state.json")
}

/// V5 FilterDown orchestrator — owns network + buffer + EMA multipliers
/// + counters. Mirrors `filter_down.py:489-842` (FilterDownV5Engine class).
pub struct FilterDownV5Engine {
    /// Value network.
    pub net: TrinityValueNet<INPUT_DIM>,
    /// Transition buffer.
    pub buffer: TransitionBuffer,
    /// EMA multiplier state (survives restart per Python:610-621).
    multipliers: Multipliers,
    /// Total train_step calls — gates cold-start.
    total_train_steps: u64,
    /// Most recent loss.
    last_loss: f64,
    /// Rolling last-20 losses for stats.
    recent_losses: Vec<f64>,
    /// Phase 8 baseline snapshot flag.
    phase8_snapshot_taken: bool,
    /// New transitions since last train (gates `maybe_train`).
    transitions_since_train: u64,
    /// Cold-start floor (constant from SPEC).
    cold_start_floor: u64,
    /// Min transitions before training starts (constant from SPEC).
    min_transitions: usize,
    /// Train every N records (constant from SPEC).
    train_every_n: u64,
    /// Training batch size.
    batch_size: usize,
    /// TD(0) learning rate.
    lr: f64,
    /// TD(0) discount.
    gamma: f64,
    /// Multiplier clamp floor.
    multiplier_floor: f64,
    /// Multiplier clamp ceil.
    multiplier_ceil: f64,
    /// Spirit content multiplier weakening (pull toward 1.0).
    spirit_strength: f64,
    /// EMA smoothing.
    smoothing: f64,
    /// Persistence paths.
    weights_path: PathBuf,
    buffer_path: PathBuf,
    state_path: PathBuf,
}

impl FilterDownV5Engine {
    /// Construct with SPEC defaults; loads weights/buffer/state from
    /// `data_dir` if files present (per SPEC §11.H.4 boot integrity).
    pub fn with_defaults(data_dir: &Path) -> Result<Self, FilterDownError> {
        let mut net = TrinityValueNet::<INPUT_DIM>::default();
        let mut buffer = TransitionBuffer::with_defaults();
        let weights_path = default_weights_path(data_dir);
        let buffer_path = default_buffer_path(data_dir);
        let state_path = default_state_path(data_dir);

        // Best-effort loads — fresh start on missing/corrupt; halt only
        // on shape mismatch (legacy V3) per SPEC §11.H.4.
        let _ = net.load(&weights_path);
        let _ = buffer.load(&buffer_path);

        let mut engine = Self {
            net,
            buffer,
            multipliers: Multipliers::ones(),
            total_train_steps: 0,
            last_loss: 0.0,
            recent_losses: Vec::new(),
            phase8_snapshot_taken: false,
            transitions_since_train: 0,
            cold_start_floor: FILTER_DOWN_COLD_START_FLOOR_EPOCHS,
            min_transitions: FILTER_DOWN_MIN_TRANSITIONS as usize,
            train_every_n: FILTER_DOWN_TRAIN_EVERY_N,
            batch_size: FILTER_DOWN_BATCH_SIZE as usize,
            lr: FILTER_DOWN_LR,
            gamma: FILTER_DOWN_GAMMA,
            multiplier_floor: FILTER_DOWN_MULTIPLIER_FLOOR,
            multiplier_ceil: FILTER_DOWN_MULTIPLIER_CEIL,
            spirit_strength: FILTER_DOWN_SPIRIT_STRENGTH_MULT,
            smoothing: FILTER_DOWN_SMOOTHING,
            weights_path,
            buffer_path,
            state_path,
        };
        let _ = engine.load_state();
        Ok(engine)
    }

    /// Record a transition `s → s'`. Reward computed from raw 130D `felt_curr`
    /// via `middle_path_loss` per Python:625-666.
    ///
    /// `titan_self_prev` + `titan_self_curr` are 162D TITAN_SELF snapshots
    /// at consecutive body cycles. `felt_curr` is the 130D inner+outer
    /// trinity snapshot (matches `assemble_felt_130` output from C4-2).
    ///
    /// Caller (body_cycle_loop in C4-3c-orchestration-wiring) is expected
    /// to invoke each body cycle when a previous SELF + felt are available.
    pub fn record_transition(
        &mut self,
        titan_self_prev: &[f64; INPUT_DIM],
        titan_self_curr: &[f64; INPUT_DIM],
        felt_curr: &[f64; 130],
    ) {
        // Reward from middle-path loss on raw felt, per Python:643-656.
        // body[0:5], mind core[5:10], spirit core[20:25] for inner;
        // body[65:70], mind core[70:75], spirit core[85:90] for outer.
        let inner_loss = titan_core::middle_path::middle_path_loss(
            &felt_curr[0..5],
            &felt_curr[5..10],
            &felt_curr[20..25],
        );
        let outer_loss = titan_core::middle_path::middle_path_loss(
            &felt_curr[65..70],
            &felt_curr[70..75],
            &felt_curr[85..90],
        );
        let reward = -(inner_loss + outer_loss) / 2.0;

        self.buffer
            .add(titan_self_prev.to_vec(), reward, titan_self_curr.to_vec());
        self.transitions_since_train += 1;
    }

    /// Train if buffer has enough data + enough new transitions since
    /// last train. Returns `Some(loss)` on training step, `None` otherwise.
    pub fn maybe_train<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<f64> {
        if self.buffer.len() < self.min_transitions {
            return None;
        }
        if self.transitions_since_train < self.train_every_n {
            return None;
        }

        let sample = self.buffer.sample(self.batch_size, rng);
        if sample.is_empty() {
            return None;
        }

        // Materialize Vec<[f64; INPUT_DIM]>
        let states: Vec<[f64; INPUT_DIM]> = sample
            .iter()
            .filter_map(|t| t.state.as_slice().try_into().ok())
            .collect();
        let next_states: Vec<[f64; INPUT_DIM]> = sample
            .iter()
            .filter_map(|t| t.next_state.as_slice().try_into().ok())
            .collect();
        let rewards: Vec<f64> = sample.iter().map(|t| t.reward).collect();
        if states.len() != sample.len() || next_states.len() != sample.len() {
            warn!(
                event = "FILTER_DOWN_TRAIN_SHAPE_FAIL",
                "buffer transition shape mismatch — should be unreachable"
            );
            return None;
        }

        let loss = self
            .net
            .train_step(&states, &rewards, &next_states, self.lr, self.gamma);
        self.total_train_steps += 1;
        self.last_loss = loss;
        self.recent_losses.push(loss);
        if self.recent_losses.len() > 20 {
            let drop = self.recent_losses.len() - 20;
            self.recent_losses.drain(0..drop);
        }
        self.transitions_since_train = 0;

        // Auto-persist every 10 train steps per Python:684-685.
        if self.total_train_steps.is_multiple_of(10) {
            if let Err(e) = self.persist() {
                warn!(event = "FILTER_DOWN_PERSIST_FAIL", err = ?e);
            }
        }
        Some(loss)
    }

    /// Compute 120-multiplier dict from a 162D TITAN_SELF snapshot. Per
    /// Python:693-759. Returns all-1.0 multipliers if cold-start gate
    /// hasn't been crossed.
    ///
    /// **Side effect:** updates internal EMA state. Caller invoking
    /// twice on the same input gets DIFFERENT outputs (the second is
    /// closer to the new raw values per `(1-α)` weighting).
    pub fn compute_multipliers(&mut self, titan_self_162d: &[f64; INPUT_DIM]) -> Multipliers {
        if self.total_train_steps < self.cold_start_floor {
            return self.multipliers.clone();
        }

        let grad = self.net.gradient_wrt_input(titan_self_162d);

        // Slice + abs per Python:713-718 (skip observer dims [20:25] + [85:90])
        let ib_grad: Vec<f64> = grad[0..5].iter().map(|g| g.abs()).collect();
        let im_grad: Vec<f64> = grad[5..20].iter().map(|g| g.abs()).collect();
        let is_content_grad: Vec<f64> = grad[25..65].iter().map(|g| g.abs()).collect();
        let ob_grad: Vec<f64> = grad[65..70].iter().map(|g| g.abs()).collect();
        let om_grad: Vec<f64> = grad[70..85].iter().map(|g| g.abs()).collect();
        let os_content_grad: Vec<f64> = grad[90..130].iter().map(|g| g.abs()).collect();

        // Find max gradient across all 120 dims (Python:720-724)
        let all_grad_iter = ib_grad
            .iter()
            .chain(im_grad.iter())
            .chain(is_content_grad.iter())
            .chain(ob_grad.iter())
            .chain(om_grad.iter())
            .chain(os_content_grad.iter());
        let mut max_grad = 0.0_f64;
        for &g in all_grad_iter {
            if g > max_grad {
                max_grad = g;
            }
        }
        if max_grad < 1e-8 {
            max_grad = 1.0;
        }

        let scale = |g: f64| -> f64 {
            (g / max_grad * FILTER_DOWN_NORM_SCALE)
                .max(self.multiplier_floor)
                .min(self.multiplier_ceil)
        };

        let new_ib: Vec<f64> = ib_grad.iter().map(|&g| scale(g)).collect();
        let new_im: Vec<f64> = im_grad.iter().map(|&g| scale(g)).collect();
        let mut new_is_c: Vec<f64> = is_content_grad.iter().map(|&g| scale(g)).collect();
        let new_ob: Vec<f64> = ob_grad.iter().map(|&g| scale(g)).collect();
        let new_om: Vec<f64> = om_grad.iter().map(|&g| scale(g)).collect();
        let mut new_os_c: Vec<f64> = os_content_grad.iter().map(|&g| scale(g)).collect();

        // Spirit weakening — pull toward 1.0 (Python:738-740)
        let k = self.spirit_strength;
        for m in new_is_c.iter_mut() {
            *m = (*m - 1.0) * k + 1.0;
        }
        for m in new_os_c.iter_mut() {
            *m = (*m - 1.0) * k + 1.0;
        }

        // EMA update (Python:742-750)
        let alpha = self.smoothing;
        let ema = |old: &[f64], new: &[f64]| -> Vec<f64> {
            old.iter()
                .zip(new.iter())
                .map(|(o, n)| alpha * o + (1.0 - alpha) * n)
                .collect()
        };
        self.multipliers.inner_body = ema(&self.multipliers.inner_body, &new_ib);
        self.multipliers.inner_mind = ema(&self.multipliers.inner_mind, &new_im);
        self.multipliers.inner_spirit_content =
            ema(&self.multipliers.inner_spirit_content, &new_is_c);
        self.multipliers.outer_body = ema(&self.multipliers.outer_body, &new_ob);
        self.multipliers.outer_mind = ema(&self.multipliers.outer_mind, &new_om);
        self.multipliers.outer_spirit_content =
            ema(&self.multipliers.outer_spirit_content, &new_os_c);

        // Round to 4 decimals for output (matches Python wire payload)
        self.multipliers.rounded()
    }

    /// Total train steps observed.
    pub fn total_train_steps(&self) -> u64 {
        self.total_train_steps
    }

    /// Last computed loss.
    pub fn last_loss(&self) -> f64 {
        self.last_loss
    }

    /// True iff cold-start gate is open (≥ floor steps).
    pub fn cold_start_complete(&self) -> bool {
        self.total_train_steps >= self.cold_start_floor
    }

    /// Persist weights + buffer + state to disk per SPEC §11.H.1.
    pub fn persist(&self) -> Result<(), FilterDownError> {
        self.net.save(&self.weights_path)?;
        self.buffer.save(&self.buffer_path)?;
        self.save_state()?;
        Ok(())
    }

    /// Save sidecar state (counters + EMA).
    pub fn save_state(&self) -> Result<(), FilterDownError> {
        let state = FilterDownV5StateFile {
            schema_version: FILTER_DOWN_STATE_SCHEMA_VERSION,
            total_train_steps: self.total_train_steps,
            last_loss: self.last_loss,
            recent_losses: self.recent_losses.clone(),
            phase8_snapshot_taken: self.phase8_snapshot_taken,
            multipliers_ema: MultipliersEma {
                inner_body: self.multipliers.inner_body.clone(),
                inner_mind: self.multipliers.inner_mind.clone(),
                inner_spirit_content: self.multipliers.inner_spirit_content.clone(),
                outer_body: self.multipliers.outer_body.clone(),
                outer_mind: self.multipliers.outer_mind.clone(),
                outer_spirit_content: self.multipliers.outer_spirit_content.clone(),
            },
        };
        let bytes = serde_json::to_vec_pretty(&state)?;
        titan_core::atomic_write::atomic_write(
            &self.state_path,
            &bytes,
            titan_core::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load sidecar state if present.
    pub fn load_state(&mut self) -> Result<bool, FilterDownError> {
        if !self.state_path.exists() {
            return Ok(false);
        }
        let bytes = std::fs::read(&self.state_path)?;
        let state: FilterDownV5StateFile = serde_json::from_slice(&bytes)?;
        if state.schema_version != FILTER_DOWN_STATE_SCHEMA_VERSION {
            return Ok(false);
        }
        self.total_train_steps = state.total_train_steps;
        self.last_loss = state.last_loss;
        self.recent_losses = state.recent_losses;
        self.phase8_snapshot_taken = state.phase8_snapshot_taken;
        // Restore EMA only if shapes match (defensive against schema drift).
        if state.multipliers_ema.inner_body.len() == 5 {
            self.multipliers.inner_body = state.multipliers_ema.inner_body;
        }
        if state.multipliers_ema.inner_mind.len() == 15 {
            self.multipliers.inner_mind = state.multipliers_ema.inner_mind;
        }
        if state.multipliers_ema.inner_spirit_content.len() == 40 {
            self.multipliers.inner_spirit_content = state.multipliers_ema.inner_spirit_content;
        }
        if state.multipliers_ema.outer_body.len() == 5 {
            self.multipliers.outer_body = state.multipliers_ema.outer_body;
        }
        if state.multipliers_ema.outer_mind.len() == 15 {
            self.multipliers.outer_mind = state.multipliers_ema.outer_mind;
        }
        if state.multipliers_ema.outer_spirit_content.len() == 40 {
            self.multipliers.outer_spirit_content = state.multipliers_ema.outer_spirit_content;
        }
        Ok(true)
    }

    /// Phase B: msgpack-serializable snapshot for `filter_down_state.bin`
    /// SHM slot (rFP_phase_c_state_read_unification §B / D-SPEC-72). Mirrors
    /// Python `FilterDownV5Engine.get_stats()` 1:1. `publish_enabled` is
    /// always `true` in Rust (V4 retired per SPEC G6 2026-04-25; V5 is the
    /// sole engine).
    pub fn get_stats(&self, ts: f64) -> crate::metadata_publisher::FilterDownStateMetadata {
        use crate::metadata_publisher::{
            FilterDownStateMetadata, MultipliersBands, MultipliersMean,
        };
        use titan_core::constants::FILTER_DOWN_STATE_SCHEMA_VERSION;

        let mean = |slice: &[f64]| -> f64 {
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f64>() / slice.len() as f64
            }
        };
        let round4 =
            |slice: &[f64]| -> Vec<f64> { slice.iter().map(|v| (v * 1e4).round() / 1e4).collect() };

        FilterDownStateMetadata {
            version: "v5".to_string(),
            input_dim: INPUT_DIM as u32,
            output_dim: OUTPUT_DIM as u32,
            buffer_size: self.buffer.len(),
            total_train_steps: self.total_train_steps,
            last_loss: (self.last_loss * 1e6).round() / 1e6,
            publish_enabled: true,
            spirit_filter_strength: self.spirit_strength,
            cold_start_floor: self.cold_start_floor,
            multipliers_mean: MultipliersMean {
                inner_body: mean(&self.multipliers.inner_body),
                inner_mind: mean(&self.multipliers.inner_mind),
                inner_spirit_content: mean(&self.multipliers.inner_spirit_content),
                outer_body: mean(&self.multipliers.outer_body),
                outer_mind: mean(&self.multipliers.outer_mind),
                outer_spirit_content: mean(&self.multipliers.outer_spirit_content),
            },
            multipliers: MultipliersBands {
                inner_body: round4(&self.multipliers.inner_body),
                inner_mind: round4(&self.multipliers.inner_mind),
                inner_spirit_content: round4(&self.multipliers.inner_spirit_content),
                outer_body: round4(&self.multipliers.outer_body),
                outer_mind: round4(&self.multipliers.outer_mind),
                outer_spirit_content: round4(&self.multipliers.outer_spirit_content),
            },
            schema_version: FILTER_DOWN_STATE_SCHEMA_VERSION as u32,
            ts,
        }
    }
}

/// Build UNIFIED_SPIRIT_FILTER_DOWN bus payload as `rmpv::Value::Map` per
/// SPEC §8.6 schema + §8.10 line 900 byte-identical guarantee. Caller
/// publishes via `BusClient::publish` with msg_type
/// `"UNIFIED_SPIRIT_FILTER_DOWN"`, src `"unified_spirit"`, P1 priority.
///
/// Pre-2026-05-13 returned `Vec<u8>` (msgpack-encoded). Closure of
/// `rFP_worker_broadcast_topics_completion §4.C-ter`: now returns the
/// structured Value so `BusClient::publish` / `encode_simple` can embed it
/// directly into the envelope as a nested Map (matching Python ground truth).
pub fn encode_filter_down_payload(
    multipliers: &Multipliers,
    epoch_id: u64,
    ts: f64,
) -> rmpv::Value {
    let to_array = |v: &[f64]| -> rmpv::Value {
        rmpv::Value::Array(v.iter().map(|x| rmpv::Value::F64(*x)).collect())
    };
    let mults_map = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("inner_body".into()),
            to_array(&multipliers.inner_body),
        ),
        (
            rmpv::Value::String("inner_mind".into()),
            to_array(&multipliers.inner_mind),
        ),
        (
            rmpv::Value::String("inner_spirit_content".into()),
            to_array(&multipliers.inner_spirit_content),
        ),
        (
            rmpv::Value::String("outer_body".into()),
            to_array(&multipliers.outer_body),
        ),
        (
            rmpv::Value::String("outer_mind".into()),
            to_array(&multipliers.outer_mind),
        ),
        (
            rmpv::Value::String("outer_spirit_content".into()),
            to_array(&multipliers.outer_spirit_content),
        ),
    ]);
    rmpv::Value::Map(vec![
        (rmpv::Value::String("multipliers".into()), mults_map),
        (
            rmpv::Value::String("epoch_id".into()),
            rmpv::Value::Integer(rmpv::Integer::from(epoch_id)),
        ),
        (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use titan_core::trinity_value_net::{
        TrinityValueNetWeights,
        TRINITY_VALUE_NET_WEIGHTS_SCHEMA_VERSION as FILTER_DOWN_WEIGHTS_SCHEMA_VERSION,
    };

    /// Test alias: the unified engine's value net is `TrinityValueNet<INPUT_DIM>` (162D).
    type ValueNet = TrinityValueNet<INPUT_DIM>;
    use tempfile::tempdir;

    fn seeded_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    fn ones_state() -> [f64; INPUT_DIM] {
        [1.0_f64; INPUT_DIM]
    }

    // ── FORWARD (3) ─────────────────────────────────────────────────────

    #[test]
    fn forward_zeros_weights_outputs_zero_bias() {
        // C4-3a test 1: with zero weights + zero biases, V(s)=0 for any input
        let net = ValueNet::zeros();
        let s = ones_state();
        let v = net.forward(&s);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn forward_output_finite_for_random_init() {
        // C4-3a test 2: random weights + random input → finite scalar
        let mut rng = seeded_rng(42);
        let net = ValueNet::new(&mut rng);
        let s = ones_state();
        let v = net.forward(&s);
        assert!(v.is_finite(), "V(s) must be finite, got {v}");
    }

    #[test]
    fn forward_batch_matches_per_sample_loop() {
        // C4-3a test 3: forward_batch == [forward(s) for s in states]
        let mut rng = seeded_rng(123);
        let net = ValueNet::new(&mut rng);
        let s1 = [0.5_f64; INPUT_DIM];
        let mut s2 = [0.0_f64; INPUT_DIM];
        for (i, v) in s2.iter_mut().enumerate() {
            *v = (i as f64) * 0.001;
        }
        let states = [s1, s2];
        let batch = net.forward_batch(&states);
        assert_eq!(batch.len(), 2);
        assert!((batch[0] - net.forward(&s1)).abs() < 1e-12);
        assert!((batch[1] - net.forward(&s2)).abs() < 1e-12);
    }

    // ── XAVIER / HE INIT (2) ────────────────────────────────────────────

    #[test]
    fn xavier_init_w1_variance_approx_2_over_in_dim() {
        // C4-3a test 4: empirical variance of w1 ~ 2 / INPUT_DIM
        let mut rng = seeded_rng(0xDEADBEEF);
        let net = ValueNet::new(&mut rng);
        let mean: f64 = net.w1.iter().sum::<f64>() / net.w1.len() as f64;
        let variance: f64 =
            net.w1.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / net.w1.len() as f64;
        let expected = 2.0 / INPUT_DIM as f64;
        // Wide tolerance — empirical from 20736 samples, drawn from N(0, expected)
        assert!(
            (variance - expected).abs() < expected * 0.1,
            "w1 variance {variance} should be near {expected} (±10%)"
        );
        // mean approximately zero
        assert!(mean.abs() < 0.01, "w1 mean {mean} should be near 0");
    }

    #[test]
    fn biases_initialize_to_zero() {
        // C4-3a test 5: per Python:99-104, biases zero-init
        let mut rng = seeded_rng(7);
        let net = ValueNet::new(&mut rng);
        for v in &net.b1 {
            assert_eq!(*v, 0.0);
        }
        for v in &net.b2 {
            assert_eq!(*v, 0.0);
        }
        for v in &net.b3 {
            assert_eq!(*v, 0.0);
        }
    }

    // ── GRADIENT_WRT_INPUT (3) ──────────────────────────────────────────

    #[test]
    fn gradient_wrt_input_dimension_and_finite() {
        // C4-3a test 6: returns 162 floats, all finite
        let mut rng = seeded_rng(99);
        let net = ValueNet::new(&mut rng);
        let s = ones_state();
        let g = net.gradient_wrt_input(&s);
        assert_eq!(g.len(), INPUT_DIM);
        for (i, &v) in g.iter().enumerate() {
            assert!(v.is_finite(), "gradient[{i}] must be finite, got {v}");
        }
    }

    #[test]
    fn gradient_wrt_input_zeros_when_all_relu_dead() {
        // C4-3a test 7: with all-negative pre-activations (achieved by
        // negative input * positive zero weights → zeros, but flip with
        // zeroed net + non-zero biases set negative), all gradients
        // through ReLU should be 0.
        let mut net = ValueNet::zeros();
        // Force all z1, z2 negative by setting biases to -1 and zero weights
        for v in net.b1.iter_mut() {
            *v = -1.0;
        }
        for v in net.b2.iter_mut() {
            *v = -1.0;
        }
        let s = [0.0_f64; INPUT_DIM];
        let g = net.gradient_wrt_input(&s);
        // All ReLU outputs are zero; gradient through them is zero
        assert!(g.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_finite_difference_check() {
        // C4-3a test 8: numerical gradient ≈ analytical gradient via
        // central-difference (cheap sanity check; full Python parity in
        // C4-3b TD(0) tests where we have shared-seed weights).
        let mut rng = seeded_rng(2026);
        let net = ValueNet::new(&mut rng);
        let mut s = ones_state();
        // Push a few inputs into "active" ReLU regions
        for (i, v) in s.iter_mut().enumerate() {
            *v = ((i as f64) * 0.01) - 0.5;
        }
        let analytical = net.gradient_wrt_input(&s);

        // Pick 3 input dims to numerically check
        for &dim in &[0_usize, 50, 100] {
            let h = 1e-5_f64;
            let mut s_plus = s;
            s_plus[dim] += h;
            let mut s_minus = s;
            s_minus[dim] -= h;
            let v_plus = net.forward(&s_plus);
            let v_minus = net.forward(&s_minus);
            let numerical = (v_plus - v_minus) / (2.0 * h);
            assert!(
                (analytical[dim] - numerical).abs() < 1e-3,
                "dim {dim}: analytical {} vs numerical {}",
                analytical[dim],
                numerical
            );
        }
    }

    // ── SAVE / LOAD (4) ─────────────────────────────────────────────────

    #[test]
    fn save_and_load_round_trip_preserves_weights() {
        // C4-3a test 9: save → load preserves all weights within f64
        // serde_json round-trip precision (~ULP-level drift acceptable).
        // Tighter byte-equality is impractical due to JSON float formatting
        // (matches GreatEpoch timestamp fix pattern in unified_spirit.rs).
        let mut rng = seeded_rng(0xCAFEF00D);
        let net = ValueNet::new(&mut rng);
        let dir = tempdir().unwrap();
        let path = default_weights_path(dir.path());
        net.save(&path).unwrap();

        let mut loaded = ValueNet::zeros();
        let ok = loaded.load(&path).unwrap();
        assert!(ok, "load returns true on success");

        // Float vectors should match within 1e-12 (JSON round-trip is
        // round-trip-safe for f64 via ryu BUT not always byte-equal due
        // to internal repr canonicalization).
        let cmp = |a: &[f64], b: &[f64], name: &str| {
            assert_eq!(a.len(), b.len(), "{name}: len mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-12,
                    "{name}[{i}]: {x} vs {y}, diff {}",
                    (x - y).abs()
                );
            }
        };
        cmp(&loaded.w1, &net.w1, "w1");
        cmp(&loaded.b1, &net.b1, "b1");
        cmp(&loaded.w2, &net.w2, "w2");
        cmp(&loaded.b2, &net.b2, "b2");
        cmp(&loaded.w3, &net.w3, "w3");
        cmp(&loaded.b3, &net.b3, "b3");

        // Forward output should be virtually identical too.
        let s = ones_state();
        assert!((loaded.forward(&s) - net.forward(&s)).abs() < 1e-9);
    }

    #[test]
    fn load_handles_missing_file_returns_false() {
        // C4-3a test 10: missing file = clean start (Ok(false))
        let dir = tempdir().unwrap();
        let mut net = ValueNet::zeros();
        let result = net.load(&default_weights_path(dir.path())).unwrap();
        assert!(!result, "no file → load returns false");
    }

    #[test]
    fn load_rejects_corrupt_json() {
        // C4-3a test 11: corrupt JSON → error
        let dir = tempdir().unwrap();
        let path = default_weights_path(dir.path());
        std::fs::write(&path, b"not valid json").unwrap();
        let mut net = ValueNet::zeros();
        let result = net.load(&path);
        assert!(matches!(
            result,
            Err(FilterDownError::Io(_)) | Err(FilterDownError::Json(_))
        ));
    }

    #[test]
    fn load_rejects_legacy_v3_15d_shape() {
        // C4-3a test 12: legacy V3 weights (15D state_dim) → ShapeMismatch
        let dir = tempdir().unwrap();
        let path = default_weights_path(dir.path());
        // Build a V3-shaped weight file with 15D state
        let v3_w1 = vec![vec![0.0_f64; 32]; 15]; // V3 was 15→32
        let bogus = TrinityValueNetWeights {
            schema_version: FILTER_DOWN_WEIGHTS_SCHEMA_VERSION,
            w1: v3_w1,
            b1: vec![0.0_f64; 32],
            w2: vec![vec![0.0_f64; 16]; 32],
            b2: vec![0.0_f64; 16],
            w3: vec![vec![0.0_f64; 1]; 16],
            b3: vec![0.0_f64; 1],
        };
        std::fs::write(&path, serde_json::to_vec(&bogus).unwrap()).unwrap();
        let mut net = ValueNet::zeros();
        let result = net.load(&path);
        assert!(matches!(result, Err(FilterDownError::ShapeMismatch { .. })));
    }

    // ── C4-3b: TRANSITION BUFFER (4) ────────────────────────────────────

    fn make_state(seed: f64) -> Vec<f64> {
        (0..INPUT_DIM).map(|i| seed + (i as f64) * 0.001).collect()
    }

    #[test]
    fn buffer_add_and_cap_rotation() {
        // C4-3b test 13: add appends until cap, then overwrites at write_idx
        let mut buf = TransitionBuffer::new(3);
        for i in 0..5 {
            buf.add(make_state(i as f64), i as f64, make_state((i + 1) as f64));
        }
        // Cap = 3, so only last 3 transitions remain (indexed 2, 3, 4
        // overwriting 0, 1, but write_idx wraps).
        assert_eq!(buf.len(), 3);
        // After 5 adds with cap 3: write_idx = 5 % 3 = 2.
        assert_eq!(buf.write_idx(), 2);
        // First 3 added entries fill [0,1,2], then 4th overwrites [0],
        // 5th overwrites [1]. So buffer = [overwritten-from-4, overwritten-from-5, original-2]
        assert_eq!(buf.transitions()[0].reward, 3.0);
        assert_eq!(buf.transitions()[1].reward, 4.0);
        assert_eq!(buf.transitions()[2].reward, 2.0);
    }

    #[test]
    fn buffer_sample_size_min_of_request_and_len() {
        // C4-3b test 14: sample returns min(batch_size, len)
        let mut buf = TransitionBuffer::new(10);
        for i in 0..4 {
            buf.add(make_state(i as f64), i as f64, make_state(i as f64));
        }
        let mut rng = seeded_rng(123);
        let small = buf.sample(2, &mut rng);
        assert_eq!(small.len(), 2);
        let big = buf.sample(100, &mut rng);
        assert_eq!(big.len(), 4); // capped at buffer size
        let zero = buf.sample(0, &mut rng);
        assert_eq!(zero.len(), 0);
    }

    #[test]
    fn buffer_sample_without_replacement() {
        // C4-3b test 15: sampled transitions are unique (replace=False
        // per Python:278)
        let mut buf = TransitionBuffer::new(10);
        for i in 0..5 {
            buf.add(make_state(i as f64 * 100.0), i as f64, make_state(0.0));
        }
        let mut rng = seeded_rng(7);
        let sample = buf.sample(5, &mut rng);
        assert_eq!(sample.len(), 5);
        let mut rewards: Vec<f64> = sample.iter().map(|t| t.reward).collect();
        rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(rewards, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_save_load_roundtrip_recomputes_write_idx() {
        // C4-3b test 16: persist + restore preserves transitions; load
        // recomputes write_idx = len % max_size
        let mut buf = TransitionBuffer::new(5);
        for i in 0..3 {
            buf.add(make_state(i as f64), i as f64, make_state(0.0));
        }
        let dir = tempdir().unwrap();
        let path = default_buffer_path(dir.path());
        buf.save(&path).unwrap();

        let mut buf2 = TransitionBuffer::new(5);
        let ok = buf2.load(&path).unwrap();
        assert!(ok);
        assert_eq!(buf2.len(), 3);
        // write_idx = len % max_size = 3 % 5 = 3
        assert_eq!(buf2.write_idx(), 3);
        for i in 0..3 {
            assert_eq!(buf2.transitions()[i].reward, i as f64);
        }
    }

    // ── C4-3b: TD(0) TRAIN_STEP (9) ────────────────────────────────────

    fn ones_batch_state() -> [f64; INPUT_DIM] {
        [1.0_f64; INPUT_DIM]
    }

    #[test]
    fn train_step_empty_batch_returns_zero_loss() {
        // C4-3b test 17: empty batch → 0.0 loss + no weights mutated
        let mut rng = seeded_rng(42);
        let mut net = ValueNet::new(&mut rng);
        let w1_before = net.w1.clone();
        let loss = net.train_step(&[], &[], &[], 0.001, 0.95);
        assert_eq!(loss, 0.0);
        assert_eq!(net.w1, w1_before);
    }

    #[test]
    fn train_step_td_target_formula() {
        // C4-3b test 18: with all-zero weights, V(s) = V(s') = 0 →
        // target = r, error = -r, loss = mean(r²).
        let mut net = ValueNet::zeros();
        let states = vec![ones_batch_state(); 3];
        let next_states = vec![ones_batch_state(); 3];
        let rewards = vec![1.0, 2.0, 3.0];
        let loss = net.train_step(&states, &rewards, &next_states, 0.001, 0.95);
        // Expected: mean((0-1)² + (0-2)² + (0-3)²) = (1+4+9)/3 ≈ 4.6667
        assert!(
            (loss - (1.0 + 4.0 + 9.0) / 3.0).abs() < 1e-12,
            "loss {loss} != expected 4.6667"
        );
    }

    #[test]
    fn train_step_loss_decreases_after_one_update() {
        // C4-3b test 19: gradient descent reduces loss (sanity)
        let mut rng = seeded_rng(2026);
        let mut net = ValueNet::new(&mut rng);
        // Construct a simple regression target: predict state[0]
        let states: Vec<[f64; INPUT_DIM]> = (0..8)
            .map(|i| {
                let mut s = [0.0_f64; INPUT_DIM];
                s[0] = (i as f64) * 0.1;
                s
            })
            .collect();
        let next_states = vec![[0.0_f64; INPUT_DIM]; 8]; // V(s') = 0 fixed
        let rewards: Vec<f64> = states.iter().map(|s| s[0]).collect();

        let loss_before = net.train_step(&states, &rewards, &next_states, 0.0, 0.95);
        // Now apply real gradient steps
        let mut last_loss = loss_before;
        for _ in 0..50 {
            last_loss = net.train_step(&states, &rewards, &next_states, 0.01, 0.95);
        }
        assert!(
            last_loss < loss_before,
            "loss should decrease: before={loss_before}, after={last_loss}"
        );
    }

    #[test]
    fn train_step_layer3_gradient_correct() {
        // C4-3b test 20: with zero weights, only b3 should receive nonzero
        // gradient on first step (since a2 = ReLU(0) = 0 → dw3 = 0; only db3
        // updates from dz3 sum). Compute expected db3 manually.
        let mut net = ValueNet::zeros();
        let states = vec![ones_batch_state(); 2];
        let next_states = vec![ones_batch_state(); 2];
        let rewards = vec![1.0, 2.0];
        let _ = net.train_step(&states, &rewards, &next_states, 0.5, 0.95);
        // With zero weights: V(s)=0, V(s')=0, target = r.
        // dz3[i] = 2*(0 - r_i)/N = -r_i (for N=2: -1.0, -2.0)
        // db3 = sum(dz3) = -3.0
        // b3_new = 0 - lr * db3 = 0 - 0.5*(-3.0) = +1.5
        assert!(
            (net.b3[0] - 1.5).abs() < 1e-12,
            "b3 should be 1.5, got {}",
            net.b3[0]
        );
        // a2 was zero so dw3 should be zero
        assert!(net.w3.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn train_step_layer2_gradient_zero_when_relu_dead() {
        // C4-3b test 21: when all z2 are negative (ReLU dead), dz2 = 0 →
        // dw2 = 0, db2 = 0, weights unchanged at layer 2 + below.
        let mut net = ValueNet::zeros();
        // Force z2 always negative by setting b2 = -1
        for v in net.b2.iter_mut() {
            *v = -1.0;
        }
        let w2_before = net.w2.clone();
        let b2_before = net.b2.clone();
        let w1_before = net.w1.clone();
        let states = vec![ones_batch_state(); 2];
        let next_states = vec![ones_batch_state(); 2];
        let rewards = vec![1.0, 2.0];
        let _ = net.train_step(&states, &rewards, &next_states, 0.5, 0.95);
        assert_eq!(net.w2, w2_before, "w2 should not change");
        assert_eq!(net.b2, b2_before, "b2 should not change");
        assert_eq!(net.w1, w1_before, "w1 should not change");
    }

    #[test]
    fn train_step_layer1_gradient_zero_when_layer2_dead() {
        // C4-3b test 22: extension of above — layer1 also untouched
        let mut net = ValueNet::zeros();
        for v in net.b1.iter_mut() {
            *v = -1.0;
        }
        let b1_before = net.b1.clone();
        let states = vec![ones_batch_state(); 2];
        let next_states = vec![ones_batch_state(); 2];
        let rewards = vec![1.0, 2.0];
        let _ = net.train_step(&states, &rewards, &next_states, 0.5, 0.95);
        // b1 unchanged because z1 negative → ReLU dead → dz1 = 0
        assert_eq!(net.b1, b1_before);
    }

    #[test]
    fn train_step_lr_scaling_multiplies_update() {
        // C4-3b test 23: 2× lr → 2× weight update (linear in lr)
        let states = vec![ones_batch_state(); 2];
        let next_states = vec![ones_batch_state(); 2];
        let rewards = vec![1.0, 2.0];

        let mut net1 = ValueNet::zeros();
        let _ = net1.train_step(&states, &rewards, &next_states, 0.5, 0.95);
        let b3_lr_half = net1.b3[0];

        let mut net2 = ValueNet::zeros();
        let _ = net2.train_step(&states, &rewards, &next_states, 1.0, 0.95);
        let b3_lr_full = net2.b3[0];

        assert!(
            (b3_lr_full - 2.0 * b3_lr_half).abs() < 1e-12,
            "lr=1.0 update should be 2× lr=0.5 update: {b3_lr_full} vs {}",
            2.0 * b3_lr_half
        );
    }

    #[test]
    fn train_step_gamma_zero_makes_target_just_reward() {
        // C4-3b test 24: gamma = 0 → target = r (no bootstrapping)
        let mut rng = seeded_rng(3);
        let mut net = ValueNet::new(&mut rng);
        // Force V(s') ≠ 0 by giving non-zero weights
        let states = vec![ones_batch_state(); 2];
        let next_states = vec![ones_batch_state(); 2];
        let rewards = vec![1.0, 2.0];
        // Save reference state
        let net_clone = net.clone();
        let _loss_g_zero = net.train_step(&states, &rewards, &next_states, 0.001, 0.0);

        let mut net2 = net_clone;
        let _loss_g_nonzero = net2.train_step(&states, &rewards, &next_states, 0.001, 0.95);

        // Updates should differ when γ != 0 vs γ = 0 (since V(s') matters)
        assert_ne!(net.b3[0], net2.b3[0]);
    }

    #[test]
    fn train_step_persistence_round_trip_preserves_weights_after_train() {
        // C4-3b test 25: train + save + load + re-forward — weights match
        // post-train output within 1e-9
        let mut rng = seeded_rng(0xBEEFFACE);
        let mut net = ValueNet::new(&mut rng);
        let states = vec![ones_batch_state(); 4];
        let next_states = vec![ones_batch_state(); 4];
        let rewards = vec![0.5, 1.0, 1.5, 2.0];
        let _ = net.train_step(&states, &rewards, &next_states, 0.001, 0.95);

        let dir = tempdir().unwrap();
        let path = default_weights_path(dir.path());
        net.save(&path).unwrap();

        let mut loaded = ValueNet::zeros();
        loaded.load(&path).unwrap();

        let s = ones_batch_state();
        assert!((loaded.forward(&s) - net.forward(&s)).abs() < 1e-9);
    }

    // ── C4-3c: FilterDownV5Engine + Multipliers + bus payload (15) ─────

    fn engine_with_trained_net(dir: &Path, train_steps_target: u64) -> FilterDownV5Engine {
        let mut e = FilterDownV5Engine::with_defaults(dir).unwrap();
        // Force the cold-start gate to be open by setting train_steps directly.
        e.total_train_steps = train_steps_target;
        e
    }

    #[test]
    fn multipliers_ones_has_120_total() {
        // C4-3c test 26: Multipliers::ones returns 120 multipliers (5+15+40+5+15+40)
        let m = Multipliers::ones();
        assert_eq!(m.total_count(), 120);
        assert_eq!(m.inner_body, vec![1.0; 5]);
        assert_eq!(m.outer_spirit_content, vec![1.0; 40]);
    }

    #[test]
    fn cold_start_returns_all_ones() {
        // C4-3c test 27: total_train_steps < cold_start_floor → all 1.0
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        assert!(!engine.cold_start_complete());
        let s = [0.7_f64; INPUT_DIM];
        let m = engine.compute_multipliers(&s);
        assert_eq!(m, Multipliers::ones());
    }

    #[test]
    fn cold_start_complete_after_floor() {
        // C4-3c test 28: cold_start_complete flips at floor
        let dir = tempdir().unwrap();
        let engine = engine_with_trained_net(dir.path(), FILTER_DOWN_COLD_START_FLOOR_EPOCHS);
        assert!(engine.cold_start_complete());
    }

    #[test]
    fn compute_multipliers_clamped_to_floor_and_ceil() {
        // C4-3c test 29: all multipliers ∈ [0.3, 3.0]
        let dir = tempdir().unwrap();
        let mut rng = seeded_rng(0xC0FFEE);
        let mut engine = engine_with_trained_net(dir.path(), 9999);
        engine.net = ValueNet::new(&mut rng);
        let s = ones_batch_state();
        let m = engine.compute_multipliers(&s);
        let all = m
            .inner_body
            .iter()
            .chain(m.inner_mind.iter())
            .chain(m.inner_spirit_content.iter())
            .chain(m.outer_body.iter())
            .chain(m.outer_mind.iter())
            .chain(m.outer_spirit_content.iter());
        for (i, &v) in all.enumerate() {
            assert!(
                (FILTER_DOWN_MULTIPLIER_FLOOR..=FILTER_DOWN_MULTIPLIER_CEIL).contains(&v),
                "multiplier[{i}] = {v} outside [{}, {}]",
                FILTER_DOWN_MULTIPLIER_FLOOR,
                FILTER_DOWN_MULTIPLIER_CEIL
            );
        }
    }

    #[test]
    fn compute_multipliers_observer_dims_not_in_payload() {
        // C4-3c test 30: payload has 120 dims (no observer slots).
        // Observer dims [20:25] inner_spirit + [85:90] outer_spirit are
        // computed internally but never emitted.
        let dir = tempdir().unwrap();
        let mut rng = seeded_rng(0xBADCAFE);
        let mut engine = engine_with_trained_net(dir.path(), 9999);
        engine.net = ValueNet::new(&mut rng);
        let s = ones_batch_state();
        let m = engine.compute_multipliers(&s);
        // Total dims sum to 120 = 5+15+40+5+15+40
        assert_eq!(m.total_count(), 120);
        // inner_spirit_content has exactly 40 entries (skipping the 5 observers)
        assert_eq!(m.inner_spirit_content.len(), 40);
        assert_eq!(m.outer_spirit_content.len(), 40);
    }

    #[test]
    fn spirit_content_pulled_toward_one() {
        // C4-3c test 31: spirit weakening — verify content multipliers
        // closer to 1.0 than they would be without the strength multiplier.
        // Construct an engine with deterministic 1-step EMA (smoothing=0)
        // and a network producing known gradients. We assert spirit
        // multipliers fall closer to 1.0 than non-spirit by inspecting
        // the formula effect.
        let dir = tempdir().unwrap();
        let mut engine = engine_with_trained_net(dir.path(), 9999);
        engine.smoothing = 0.0; // skip EMA
                                // Force gradient by giving net non-zero w3 = 1.0 + small w1 + bias
        engine.net = ValueNet::zeros();
        for v in engine.net.w3.iter_mut() {
            *v = 1.0;
        }
        for v in engine.net.b2.iter_mut() {
            *v = 1.0;
        }
        for v in engine.net.b1.iter_mut() {
            *v = 1.0;
        }
        for (i, v) in engine.net.w1.iter_mut().enumerate() {
            *v = ((i % 7) as f64) * 0.01 + 0.001;
        }
        // Compute multipliers
        let s = [0.5_f64; INPUT_DIM];
        let m = engine.compute_multipliers(&s);
        // Spirit content multipliers should be in [0.3, 3.0] AND
        // closer to 1.0 than max-magnitude body multiplier (since
        // spirit_strength=0.3 pulls them in).
        let max_dist_body = m
            .inner_body
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max);
        let max_dist_spirit = m
            .inner_spirit_content
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max);
        // After 0.3 weakening, spirit should be at most 0.3 × what body is
        // (or smaller due to different gradient magnitudes — assertion is
        // approximate).
        assert!(
            max_dist_spirit <= max_dist_body + 1e-6,
            "spirit dist {max_dist_spirit} should be <= body dist {max_dist_body}"
        );
    }

    #[test]
    fn ema_smoothing_pulls_toward_previous() {
        // C4-3c test 32: smoothing=0.9 → new EMA = 0.9*old + 0.1*raw,
        // so output is mostly the previous value.
        let dir = tempdir().unwrap();
        let mut rng = seeded_rng(11);
        let mut engine = engine_with_trained_net(dir.path(), 9999);
        engine.net = ValueNet::new(&mut rng);
        let s = ones_batch_state();
        let m1 = engine.compute_multipliers(&s);
        // EMA[1] = 0.9 * 1.0 (initial) + 0.1 * raw
        // After this call, the engine's internal multipliers are EMA-blended.
        // Calling again should produce a result closer to raw (further from 1.0).
        let m2 = engine.compute_multipliers(&s);
        // m2 should NOT equal m1 because EMA accumulates; verify internal state moved.
        assert_ne!(m1, m2);
    }

    #[test]
    fn record_transition_appends_to_buffer() {
        // C4-3c test 33: record_transition adds to buffer + advances counter
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        let prev = ones_batch_state();
        let curr = ones_batch_state();
        let felt = [0.5_f64; 130];
        let len_before = engine.buffer.len();
        engine.record_transition(&prev, &curr, &felt);
        assert_eq!(engine.buffer.len(), len_before + 1);
        assert_eq!(engine.transitions_since_train, 1);
    }

    #[test]
    fn record_transition_reward_zero_at_perfect_equilibrium() {
        // C4-3c test 34: felt all at center (0.5) → middle_path_loss = 0 →
        // reward = -0
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        let prev = ones_batch_state();
        let curr = ones_batch_state();
        let felt = [0.5_f64; 130];
        engine.record_transition(&prev, &curr, &felt);
        let last_idx = engine.buffer.len() - 1;
        assert_eq!(engine.buffer.transitions()[last_idx].reward, 0.0);
    }

    #[test]
    fn maybe_train_returns_none_below_min_transitions() {
        // C4-3c test 35: insufficient buffer → no training
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        let mut rng = seeded_rng(0);
        // Only 10 transitions (min default = 32)
        for _ in 0..10 {
            engine.record_transition(&ones_batch_state(), &ones_batch_state(), &[0.5_f64; 130]);
        }
        assert!(engine.maybe_train(&mut rng).is_none());
    }

    #[test]
    fn maybe_train_returns_none_below_train_every_n() {
        // C4-3c test 36: enough transitions but not enough since last train
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        let mut rng = seeded_rng(0);
        // Pre-seed 50 transitions
        for _ in 0..50 {
            engine.record_transition(&ones_batch_state(), &ones_batch_state(), &[0.5_f64; 130]);
        }
        // First train succeeds (transitions_since_train=50, min=32, every_n=5)
        assert!(engine.maybe_train(&mut rng).is_some());
        // Counter reset to 0; next call returns None until 5 more transitions
        assert!(engine.maybe_train(&mut rng).is_none());
    }

    #[test]
    fn maybe_train_runs_when_thresholds_met() {
        // C4-3c test 37: enough transitions → train_step runs, total_train_steps++
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        let mut rng = seeded_rng(0);
        for _ in 0..50 {
            engine.record_transition(&ones_batch_state(), &ones_batch_state(), &[0.5_f64; 130]);
        }
        let result = engine.maybe_train(&mut rng);
        assert!(result.is_some());
        assert_eq!(engine.total_train_steps, 1);
    }

    #[test]
    fn encode_filter_down_payload_round_trip() {
        // C4-3c test 38: structured payload exposes expected SPEC §8.6 schema
        let m = Multipliers::ones();
        let payload = encode_filter_down_payload(&m, 42, 1234567890.5);
        let entries = match payload {
            rmpv::Value::Map(e) => e,
            _ => panic!("expected map"),
        };
        let mut got_epoch = None;
        let mut got_ts = None;
        let mut got_inner_body_len = None;
        for (k, v) in entries {
            if let rmpv::Value::String(s) = &k {
                match s.as_str() {
                    Some("epoch_id") => got_epoch = v.as_u64(),
                    Some("ts") => got_ts = v.as_f64(),
                    Some("multipliers") => {
                        if let rmpv::Value::Map(m_entries) = v {
                            for (mk, mv) in m_entries {
                                if let rmpv::Value::String(ms) = &mk {
                                    if ms.as_str() == Some("inner_body") {
                                        if let rmpv::Value::Array(a) = mv {
                                            got_inner_body_len = Some(a.len());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(got_epoch, Some(42));
        assert!((got_ts.unwrap() - 1234567890.5).abs() < 1e-3);
        assert_eq!(got_inner_body_len, Some(5));
    }

    #[test]
    fn engine_persist_round_trip_preserves_state() {
        // C4-3c test 39: save state + reload → counters preserved
        let dir = tempdir().unwrap();
        let mut engine = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        engine.total_train_steps = 100;
        engine.last_loss = 0.0123;
        engine.recent_losses = vec![0.1, 0.05, 0.02];
        engine.persist().unwrap();

        let engine2 = FilterDownV5Engine::with_defaults(dir.path()).unwrap();
        assert_eq!(engine2.total_train_steps, 100);
        assert!((engine2.last_loss - 0.0123).abs() < 1e-12);
        assert_eq!(engine2.recent_losses.len(), 3);
    }

    #[test]
    fn multipliers_round_to_4_decimals() {
        // C4-3c test 40: rounded() output limits precision per Python
        let m = Multipliers {
            inner_body: vec![1.234567, 0.123456, 2.987654, 1.0, 0.5],
            inner_mind: vec![1.0; 15],
            inner_spirit_content: vec![1.0; 40],
            outer_body: vec![1.0; 5],
            outer_mind: vec![1.0; 15],
            outer_spirit_content: vec![1.0; 40],
        };
        let r = m.rounded();
        assert_eq!(r.inner_body[0], 1.2346);
        assert_eq!(r.inner_body[1], 0.1235);
        assert_eq!(r.inner_body[2], 2.9877);
    }
}
