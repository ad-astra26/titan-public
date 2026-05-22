//! small_filter_down — per-trinity-half learned filter_down engine
//! (SPEC §G5.1 small filter_down tier, Phase 0 chunk 0C).
//!
//! The "small" filter_down enriches one trinity half (inner OR outer):
//! `spirit → body + mind`. It is the consciousness-bootstrap tier — fired
//! once per `KERNEL_EPOCH_TICK` (D-SPEC-97), it reshapes + smoothens the
//! body/mind dims via learned gradient-attention, pulling them toward the
//! 0.5 Divine Center so per-layer coherence (§G4) can cross the balance
//! threshold (§G11) and the sphere clocks can resonate.
//!
//! Mirrors the unified `FilterDownV5Engine` formula (`g/max_grad×2.0`
//! clamp [0.3,3.0] + EMA 0.9 + spirit-content weakening) but over a single
//! half's 65D state (body[5] + mind[15] + spirit[45]) using a shared
//! `TrinityValueNet<65>`. Replaces the `1.0+(spirit−0.5)×0.1` MVP
//! placeholder (D2 drift) that the Phase C Rust port shipped.
//!
//! Half-state layout (65D): `[0:5]` body, `[5:20]` mind (15), `[20:65]`
//! spirit (45) — within spirit: observer `[20:25]`, content `[25:65]` (40).

use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::constants::{
    FILTER_DOWN_BATCH_SIZE, FILTER_DOWN_COLD_START_FLOOR_EPOCHS, FILTER_DOWN_GAMMA, FILTER_DOWN_LR,
    FILTER_DOWN_MIN_TRANSITIONS, FILTER_DOWN_MULTIPLIER_CEIL, FILTER_DOWN_MULTIPLIER_FLOOR,
    FILTER_DOWN_SPIRIT_STRENGTH_MULT, FILTER_DOWN_TRAIN_EVERY_N,
};
use crate::middle_path::middle_path_loss;
use crate::transition_buffer::TransitionBuffer;
use crate::trinity_value_net::{FilterDownError, TrinityValueNet};

/// EMA smoothing coefficient (`new = α·old + (1−α)·raw`) — mirrors the
/// unified engine's `filter_down.py:52 SMOOTHING = 0.9` tuning constant.
const FILTER_DOWN_SMOOTHING: f64 = 0.9;
/// Gradient-attention output scale (`g/max_grad × NORM_SCALE`) — mirrors
/// the unified engine's value.
const FILTER_DOWN_NORM_SCALE: f64 = 2.0;

/// Per-half input dim: body(5) + mind(15) + spirit(45).
pub const HALF_DIM: usize = 65;
/// Body multiplier count.
pub const BODY_DIMS: usize = 5;
/// Mind multiplier count.
pub const MIND_DIMS: usize = 15;
/// Spirit-content multiplier count (45 spirit − 5 observer).
pub const CONTENT_DIMS: usize = 40;
/// Schema version for the per-half state file.
pub const SMALL_FILTER_DOWN_STATE_SCHEMA_VERSION: u32 = 1;

/// 60 multipliers for one half: body[5] + mind[15] + spirit_content[40].
/// Observer dims [20:25] are NEVER emitted (G8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmallMultipliers {
    /// Body multipliers (5).
    pub body: Vec<f64>,
    /// Mind multipliers (15).
    pub mind: Vec<f64>,
    /// Spirit-content multipliers (40, observer-masked, spirit-weakened).
    pub spirit_content: Vec<f64>,
}

impl SmallMultipliers {
    /// Neutral (all-1.0) multipliers — cold-start default + EMA seed.
    pub fn ones() -> Self {
        Self {
            body: vec![1.0; BODY_DIMS],
            mind: vec![1.0; MIND_DIMS],
            spirit_content: vec![1.0; CONTENT_DIMS],
        }
    }

    fn rounded(&self) -> Self {
        let r = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| (x * 1.0e4).round() / 1.0e4).collect() };
        Self {
            body: r(&self.body),
            mind: r(&self.mind),
            spirit_content: r(&self.spirit_content),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SmallFilterDownStateFile {
    #[serde(default)]
    schema_version: u32,
    total_train_steps: u64,
    last_loss: f64,
    recent_losses: Vec<f64>,
    multipliers_ema: SmallMultipliers,
}

/// Per-half learned filter_down engine. One instance per trinity half
/// (inner-spirit-rs + outer-spirit-rs each own one).
pub struct SmallFilterDownEngine {
    net: TrinityValueNet<HALF_DIM>,
    buffer: TransitionBuffer,
    multipliers: SmallMultipliers,
    total_train_steps: u64,
    last_loss: f64,
    recent_losses: Vec<f64>,
    transitions_since_train: u64,
    cold_start_floor: u64,
    min_transitions: usize,
    train_every_n: u64,
    batch_size: usize,
    lr: f64,
    gamma: f64,
    multiplier_floor: f64,
    multiplier_ceil: f64,
    spirit_strength: f64,
    smoothing: f64,
    weights_path: PathBuf,
    buffer_path: PathBuf,
    state_path: PathBuf,
}

impl SmallFilterDownEngine {
    /// Construct with SPEC defaults for the given half (`"inner"` / `"outer"`),
    /// loading weights/buffer/state from `data_dir` if present.
    pub fn with_defaults(data_dir: &Path, half: &str) -> Result<Self, FilterDownError> {
        let weights_path = data_dir.join(format!("filter_down_local_{half}_weights.json"));
        let buffer_path = data_dir.join(format!("filter_down_local_{half}_buffer.json"));
        let state_path = data_dir.join(format!("filter_down_local_{half}_state.json"));

        let mut net = TrinityValueNet::<HALF_DIM>::default();
        let _ = net.load(&weights_path);
        let mut buffer = TransitionBuffer::with_defaults();
        let _ = buffer.load(&buffer_path);

        let mut engine = Self {
            net,
            buffer,
            multipliers: SmallMultipliers::ones(),
            total_train_steps: 0,
            last_loss: 0.0,
            recent_losses: Vec::new(),
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

    /// Record a transition `s → s'` for one half. Reward = −middle_path_loss
    /// over the half's body[0:5] + mind-core[5:10] + spirit-observer[20:25],
    /// mirroring the unified engine's inner/outer-loss reward.
    pub fn record_transition(&mut self, prev: &[f64; HALF_DIM], curr: &[f64; HALF_DIM]) {
        let loss = middle_path_loss(&curr[0..5], &curr[5..10], &curr[20..25]);
        let reward = -loss;
        self.buffer.add(prev.to_vec(), reward, curr.to_vec());
        self.transitions_since_train += 1;
    }

    /// Train if buffer has enough data + enough new transitions since last
    /// train. Returns `Some(loss)` on a training step.
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
        let states: Vec<[f64; HALF_DIM]> = sample
            .iter()
            .filter_map(|t| t.state.as_slice().try_into().ok())
            .collect();
        let next_states: Vec<[f64; HALF_DIM]> = sample
            .iter()
            .filter_map(|t| t.next_state.as_slice().try_into().ok())
            .collect();
        let rewards: Vec<f64> = sample.iter().map(|t| t.reward).collect();
        if states.len() != sample.len() || next_states.len() != sample.len() {
            warn!(
                event = "SMALL_FILTER_DOWN_TRAIN_SHAPE_FAIL",
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
        if self.total_train_steps.is_multiple_of(10) {
            if let Err(e) = self.persist() {
                warn!(event = "SMALL_FILTER_DOWN_PERSIST_FAIL", err = ?e);
            }
        }
        Some(loss)
    }

    /// Compute the 60 multipliers (body[5] + mind[15] + spirit_content[40])
    /// from the half's 65D state via learned gradient-attention. Returns
    /// all-1.0 until the cold-start floor is cleared. Updates internal EMA.
    pub fn compute_multipliers(&mut self, state: &[f64; HALF_DIM]) -> SmallMultipliers {
        if self.total_train_steps < self.cold_start_floor {
            return self.multipliers.clone();
        }

        let grad = self.net.gradient_wrt_input(state);
        let body_grad: Vec<f64> = grad[0..5].iter().map(|g| g.abs()).collect();
        let mind_grad: Vec<f64> = grad[5..20].iter().map(|g| g.abs()).collect();
        let content_grad: Vec<f64> = grad[25..65].iter().map(|g| g.abs()).collect();

        let mut max_grad = 0.0_f64;
        for &g in body_grad
            .iter()
            .chain(mind_grad.iter())
            .chain(content_grad.iter())
        {
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
        let new_body: Vec<f64> = body_grad.iter().map(|&g| scale(g)).collect();
        let new_mind: Vec<f64> = mind_grad.iter().map(|&g| scale(g)).collect();
        let mut new_content: Vec<f64> = content_grad.iter().map(|&g| scale(g)).collect();

        // Spirit weakening — pull content toward 1.0 (G9).
        let k = self.spirit_strength;
        for m in new_content.iter_mut() {
            *m = (*m - 1.0) * k + 1.0;
        }

        // EMA update.
        let alpha = self.smoothing;
        let ema = |old: &[f64], new: &[f64]| -> Vec<f64> {
            old.iter()
                .zip(new.iter())
                .map(|(o, n)| alpha * o + (1.0 - alpha) * n)
                .collect()
        };
        self.multipliers.body = ema(&self.multipliers.body, &new_body);
        self.multipliers.mind = ema(&self.multipliers.mind, &new_mind);
        self.multipliers.spirit_content = ema(&self.multipliers.spirit_content, &new_content);

        self.multipliers.rounded()
    }

    /// Total train steps observed.
    pub fn total_train_steps(&self) -> u64 {
        self.total_train_steps
    }

    /// Most recent training loss.
    pub fn last_loss(&self) -> f64 {
        self.last_loss
    }

    /// True once the cold-start floor is cleared (multipliers go live).
    pub fn cold_start_complete(&self) -> bool {
        self.total_train_steps >= self.cold_start_floor
    }

    /// Persist weights + buffer + state.
    pub fn persist(&self) -> Result<(), FilterDownError> {
        self.net.save(&self.weights_path)?;
        self.buffer.save(&self.buffer_path)?;
        self.save_state()?;
        Ok(())
    }

    /// Persist the state file (counters + EMA multipliers).
    pub fn save_state(&self) -> Result<(), FilterDownError> {
        let state = SmallFilterDownStateFile {
            schema_version: SMALL_FILTER_DOWN_STATE_SCHEMA_VERSION,
            total_train_steps: self.total_train_steps,
            last_loss: self.last_loss,
            recent_losses: self.recent_losses.clone(),
            multipliers_ema: self.multipliers.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&state)?;
        crate::atomic_write::atomic_write(
            &self.state_path,
            &bytes,
            crate::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load the state file if present + schema-compatible.
    pub fn load_state(&mut self) -> Result<bool, FilterDownError> {
        if !self.state_path.exists() {
            return Ok(false);
        }
        let bytes = std::fs::read(&self.state_path)?;
        let state: SmallFilterDownStateFile = serde_json::from_slice(&bytes)?;
        if state.schema_version != SMALL_FILTER_DOWN_STATE_SCHEMA_VERSION {
            return Ok(false);
        }
        self.total_train_steps = state.total_train_steps;
        self.last_loss = state.last_loss;
        self.recent_losses = state.recent_losses;
        if state.multipliers_ema.body.len() == BODY_DIMS {
            self.multipliers.body = state.multipliers_ema.body;
        }
        if state.multipliers_ema.mind.len() == MIND_DIMS {
            self.multipliers.mind = state.multipliers_ema.mind;
        }
        if state.multipliers_ema.spirit_content.len() == CONTENT_DIMS {
            self.multipliers.spirit_content = state.multipliers_ema.spirit_content;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use tempfile::tempdir;

    fn centered_state() -> [f64; HALF_DIM] {
        [0.5_f64; HALF_DIM]
    }

    #[test]
    fn cold_start_returns_ones() {
        let dir = tempdir().unwrap();
        let mut e = SmallFilterDownEngine::with_defaults(dir.path(), "inner").unwrap();
        let m = e.compute_multipliers(&centered_state());
        assert_eq!(m.body, vec![1.0; BODY_DIMS]);
        assert_eq!(m.mind, vec![1.0; MIND_DIMS]);
        assert_eq!(m.spirit_content, vec![1.0; CONTENT_DIMS]);
    }

    #[test]
    fn record_transition_reward_zero_at_center() {
        let dir = tempdir().unwrap();
        let mut e = SmallFilterDownEngine::with_defaults(dir.path(), "inner").unwrap();
        let s = centered_state();
        e.record_transition(&s, &s);
        // center → middle_path_loss 0 → reward 0
        assert_eq!(e.buffer.transitions()[0].reward, 0.0);
    }

    #[test]
    fn training_increments_steps_and_eventually_clears_cold_start() {
        let dir = tempdir().unwrap();
        let mut e = SmallFilterDownEngine::with_defaults(dir.path(), "outer").unwrap();
        e.cold_start_floor = 3; // shrink for the test
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut s = [0.2_f64; HALF_DIM];
        for i in 0..200 {
            let mut next = s;
            next[i % HALF_DIM] = (next[i % HALF_DIM] + 0.01).min(1.0);
            e.record_transition(&s, &next);
            e.maybe_train(&mut rng);
            s = next;
        }
        assert!(e.total_train_steps() > 3, "should have trained past floor");
        assert!(e.cold_start_complete());
        // Post-cold-start multipliers should deviate from all-1.0 for a
        // non-degenerate gradient.
        let m = e.compute_multipliers(&s);
        let any_non_unit = m
            .body
            .iter()
            .chain(m.mind.iter())
            .any(|&v| (v - 1.0).abs() > 1e-6);
        assert!(
            any_non_unit,
            "trained engine should produce non-1.0 multipliers"
        );
    }

    #[test]
    fn persist_then_reload_restores_steps() {
        let dir = tempdir().unwrap();
        {
            let mut e = SmallFilterDownEngine::with_defaults(dir.path(), "inner").unwrap();
            e.total_train_steps = 1234;
            e.persist().unwrap();
        }
        let e2 = SmallFilterDownEngine::with_defaults(dir.path(), "inner").unwrap();
        assert_eq!(e2.total_train_steps(), 1234);
    }
}
