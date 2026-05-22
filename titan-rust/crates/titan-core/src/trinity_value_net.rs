//! trinity_value_net — shared const-generic TD(0) value network for the
//! filter_down family (SPEC §G5.1 + §G7).
//!
//! `TrinityValueNet<N>` is an `N → HIDDEN_1 → HIDDEN_2 → 1` ReLU MLP that
//! predicts expected future middle-path-loss from an `N`-dim state. It is
//! the SOLE learned-attention engine for both filter_down tiers:
//! - **Unified GREAT filter_down** (`titan-unified-spirit-rs`) uses
//!   `TrinityValueNet<162>` over the 162D TITAN_SELF.
//! - **Small per-trinity-half filter_down** (`titan-{inner,outer}-spirit-rs`,
//!   Phase 0 chunk 0C) uses `TrinityValueNet<65>` over the half's
//!   body(5) + mind(15) + spirit(45) state.
//!
//! Made const-generic in Phase 0 (D-SPEC-97) so the proven learned formula
//! is shared, not duplicated. Pure Rust — no PyTorch / autograd / ndarray.
//! Weight matrices stored row-major as flat `Vec<f64>` for cache friendliness.
//! Hidden widths are fixed across both instantiations (`HIDDEN_1`/`HIDDEN_2`
//! from `crate::constants`); only the input dim `N` is parameterized.

use std::path::Path;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};

use crate::constants::{FILTER_DOWN_HIDDEN_1, FILTER_DOWN_HIDDEN_2};

/// Hidden layer 1 width (shared by all instantiations).
pub const HIDDEN_1: usize = FILTER_DOWN_HIDDEN_1 as usize;
/// Hidden layer 2 width (shared by all instantiations).
pub const HIDDEN_2: usize = FILTER_DOWN_HIDDEN_2 as usize;
/// Schema version for persisted weights JSON per SPEC §11.H.4.
pub const TRINITY_VALUE_NET_WEIGHTS_SCHEMA_VERSION: u32 = 1;

/// Errors during value-net persistence + restore (also reused by the
/// filter_down engines for buffer/state persistence — same Write/Json/Io
/// surface).
#[derive(Debug, Error)]
pub enum FilterDownError {
    /// weights/state/buffer write failed.
    #[error("filter_down write failed: {0}")]
    Write(#[from] crate::atomic_write::AtomicWriteError),
    /// JSON encode/decode failed.
    #[error("filter_down json: {0}")]
    Json(#[from] serde_json::Error),
    /// io error reading a state file.
    #[error("filter_down io: {0}")]
    Io(#[from] std::io::Error),
    /// Loaded weights have wrong shape — refuse-load (legacy 15D / 162D vs
    /// 65D, etc.). SPEC §11.H.4 boot integrity check.
    #[error(
        "filter_down weights shape mismatch: layer={layer} expected={expected} actual={actual}"
    )]
    ShapeMismatch {
        /// Which layer (`w1` / `b1` / etc.).
        layer: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count from JSON.
        actual: usize,
    },
}

/// Persistent weights schema (nested-list matrices, Python-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityValueNetWeights {
    /// Schema version (rejected on mismatch at load).
    #[serde(default)]
    pub schema_version: u32,
    /// Layer 1 weights — nested list `[N][HIDDEN_1]`.
    pub w1: Vec<Vec<f64>>,
    /// Layer 1 bias — `[HIDDEN_1]`.
    pub b1: Vec<f64>,
    /// Layer 2 weights — nested list `[HIDDEN_1][HIDDEN_2]`.
    pub w2: Vec<Vec<f64>>,
    /// Layer 2 bias — `[HIDDEN_2]`.
    pub b2: Vec<f64>,
    /// Layer 3 weights — nested list `[HIDDEN_2][1]`.
    pub w3: Vec<Vec<f64>>,
    /// Layer 3 bias — `[1]`.
    pub b3: Vec<f64>,
}

/// `N → HIDDEN_1 → HIDDEN_2 → 1` ReLU value network.
#[derive(Debug, Clone)]
pub struct TrinityValueNet<const N: usize> {
    /// Layer 1 weights (`N*HIDDEN_1`, row-major: `w1[in*HIDDEN_1 + h1]`).
    pub w1: Vec<f64>,
    /// Layer 1 biases (`HIDDEN_1`).
    pub b1: Vec<f64>,
    /// Layer 2 weights (`HIDDEN_1*HIDDEN_2`, row-major).
    pub w2: Vec<f64>,
    /// Layer 2 biases (`HIDDEN_2`).
    pub b2: Vec<f64>,
    /// Layer 3 weights (`HIDDEN_2`, column vector).
    pub w3: Vec<f64>,
    /// Layer 3 bias (1 element).
    pub b3: Vec<f64>,
}

impl<const N: usize> TrinityValueNet<N> {
    /// Construct with He-init weights: `w_ij ~ N(0, sqrt(2/fan_in))`.
    pub fn new<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let std1 = (2.0 / N as f64).sqrt();
        let std2 = (2.0 / HIDDEN_1 as f64).sqrt();
        let std3 = (2.0 / HIDDEN_2 as f64).sqrt();
        let dist1 = Normal::new(0.0, std1).expect("std1 > 0");
        let dist2 = Normal::new(0.0, std2).expect("std2 > 0");
        let dist3 = Normal::new(0.0, std3).expect("std3 > 0");

        let mut w1 = vec![0.0_f64; N * HIDDEN_1];
        for v in w1.iter_mut() {
            *v = dist1.sample(rng);
        }
        let mut w2 = vec![0.0_f64; HIDDEN_1 * HIDDEN_2];
        for v in w2.iter_mut() {
            *v = dist2.sample(rng);
        }
        let mut w3 = vec![0.0_f64; HIDDEN_2];
        for v in w3.iter_mut() {
            *v = dist3.sample(rng);
        }

        Self {
            w1,
            b1: vec![0.0_f64; HIDDEN_1],
            w2,
            b2: vec![0.0_f64; HIDDEN_2],
            w3,
            b3: vec![0.0_f64; 1],
        }
    }

    /// Construct with all-zero weights (deterministic-output / fallback path).
    pub fn zeros() -> Self {
        Self {
            w1: vec![0.0_f64; N * HIDDEN_1],
            b1: vec![0.0_f64; HIDDEN_1],
            w2: vec![0.0_f64; HIDDEN_1 * HIDDEN_2],
            b2: vec![0.0_f64; HIDDEN_2],
            w3: vec![0.0_f64; HIDDEN_2],
            b3: vec![0.0_f64; 1],
        }
    }

    /// Forward pass: `V(state)`.
    pub fn forward(&self, state: &[f64; N]) -> f64 {
        let z1 = self.layer1_z(state);
        let a1 = relu_inplace(z1);
        let z2 = self.layer2_z(&a1);
        let a2 = relu_inplace(z2);
        self.layer3_z(&a2)
    }

    /// Forward pass for a batch.
    pub fn forward_batch(&self, states: &[[f64; N]]) -> Vec<f64> {
        states.iter().map(|s| self.forward(s)).collect()
    }

    /// Compute ∂V/∂state via manual backprop through ReLU layers. Returns
    /// the `N`-dim attention vector (per-dim importance for downstream
    /// multiplier derivation).
    pub fn gradient_wrt_input(&self, state: &[f64; N]) -> [f64; N] {
        let z1 = self.layer1_z(state);
        let a1 = relu_inplace(z1.clone());
        let z2 = self.layer2_z(&a1);

        let mut da2 = vec![0.0_f64; HIDDEN_2];
        for (h2, slot) in da2.iter_mut().enumerate() {
            *slot = self.w3[h2];
        }

        let mut dz2 = vec![0.0_f64; HIDDEN_2];
        for (slot, (&zv, &dv)) in dz2.iter_mut().zip(z2.iter().zip(da2.iter())) {
            *slot = if zv > 0.0 { dv } else { 0.0 };
        }

        let mut da1 = vec![0.0_f64; HIDDEN_1];
        for (h1, slot) in da1.iter_mut().enumerate() {
            let row_off = h1 * HIDDEN_2;
            let mut sum = 0.0;
            for (h2, &d) in dz2.iter().enumerate() {
                sum += d * self.w2[row_off + h2];
            }
            *slot = sum;
        }

        let mut dz1 = vec![0.0_f64; HIDDEN_1];
        for (slot, (&zv, &dv)) in dz1.iter_mut().zip(z1.iter().zip(da1.iter())) {
            *slot = if zv > 0.0 { dv } else { 0.0 };
        }

        let mut ds = [0.0_f64; N];
        for (i, slot) in ds.iter_mut().enumerate() {
            let row = &self.w1[i * HIDDEN_1..(i + 1) * HIDDEN_1];
            let mut sum = 0.0;
            for (h1, &w) in row.iter().enumerate() {
                sum += dz1[h1] * w;
            }
            *slot = sum;
        }
        ds
    }

    /// Layer 1 pre-activation: `z1[h1] = sum_i(state[i] * w1[i*HIDDEN_1 + h1]) + b1[h1]`.
    fn layer1_z(&self, state: &[f64; N]) -> Vec<f64> {
        let mut z1 = self.b1.clone();
        for (i, &s) in state.iter().enumerate() {
            let row = &self.w1[i * HIDDEN_1..(i + 1) * HIDDEN_1];
            for (slot, &w) in z1.iter_mut().zip(row.iter()) {
                *slot += s * w;
            }
        }
        z1
    }

    /// Layer 2 pre-activation.
    fn layer2_z(&self, a1: &[f64]) -> Vec<f64> {
        let mut z2 = self.b2.clone();
        for (h1, &a) in a1.iter().enumerate() {
            let row = &self.w2[h1 * HIDDEN_2..(h1 + 1) * HIDDEN_2];
            for (slot, &w) in z2.iter_mut().zip(row.iter()) {
                *slot += a * w;
            }
        }
        z2
    }

    /// Layer 3: scalar output.
    fn layer3_z(&self, a2: &[f64]) -> f64 {
        let mut z3 = self.b3[0];
        for (h2, &a) in a2.iter().enumerate() {
            z3 += a * self.w3[h2];
        }
        z3
    }

    /// Convert weights to nested-list schema (Python-compatible).
    pub fn to_weights(&self) -> TrinityValueNetWeights {
        TrinityValueNetWeights {
            schema_version: TRINITY_VALUE_NET_WEIGHTS_SCHEMA_VERSION,
            w1: matrix_to_nested(&self.w1, N, HIDDEN_1),
            b1: self.b1.clone(),
            w2: matrix_to_nested(&self.w2, HIDDEN_1, HIDDEN_2),
            b2: self.b2.clone(),
            w3: matrix_to_nested(&self.w3, HIDDEN_2, 1),
            b3: self.b3.clone(),
        }
    }

    /// Apply weights from nested-list schema. Validates dims; refuses on
    /// mismatch per SPEC §11.H.4.
    pub fn apply_weights(&mut self, w: &TrinityValueNetWeights) -> Result<(), FilterDownError> {
        check_matrix_shape(&w.w1, N, HIDDEN_1, "w1")?;
        check_vec_shape(&w.b1, HIDDEN_1, "b1")?;
        check_matrix_shape(&w.w2, HIDDEN_1, HIDDEN_2, "w2")?;
        check_vec_shape(&w.b2, HIDDEN_2, "b2")?;
        check_matrix_shape(&w.w3, HIDDEN_2, 1, "w3")?;
        check_vec_shape(&w.b3, 1, "b3")?;

        self.w1 = nested_to_matrix(&w.w1);
        self.b1 = w.b1.clone();
        self.w2 = nested_to_matrix(&w.w2);
        self.b2 = w.b2.clone();
        self.w3 = nested_to_matrix(&w.w3);
        self.b3 = w.b3.clone();
        Ok(())
    }

    /// Save weights to JSON via `atomic_write` + 2-backup retention.
    pub fn save(&self, path: &Path) -> Result<(), FilterDownError> {
        let w = self.to_weights();
        let bytes = serde_json::to_vec_pretty(&w)?;
        crate::atomic_write::atomic_write(
            path,
            &bytes,
            crate::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load weights from JSON, falling back to `.bak` then `.bak.prev`.
    /// Returns `Ok(false)` when no file found (caller keeps fresh-init).
    pub fn load(&mut self, path: &Path) -> Result<bool, FilterDownError> {
        let candidates = [
            path.to_path_buf(),
            path.with_extension("json.bak"),
            path.with_extension("json.bak.prev"),
        ];

        let mut last_err: Option<FilterDownError> = None;
        let mut found_any = false;
        for candidate in &candidates {
            if !candidate.exists() {
                continue;
            }
            found_any = true;
            match std::fs::read(candidate) {
                Ok(bytes) => match serde_json::from_slice::<TrinityValueNetWeights>(&bytes) {
                    Ok(w) => {
                        if w.schema_version != TRINITY_VALUE_NET_WEIGHTS_SCHEMA_VERSION {
                            warn!(
                                event = "FILTER_DOWN_SCHEMA_MISMATCH",
                                loaded = w.schema_version,
                                expected = TRINITY_VALUE_NET_WEIGHTS_SCHEMA_VERSION,
                                ?candidate,
                                "schema mismatch; trying next backup"
                            );
                            continue;
                        }
                        self.apply_weights(&w)?;
                        info!(event = "FILTER_DOWN_LOADED", ?candidate, "weights loaded");
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!(event = "FILTER_DOWN_DECODE_FAIL", ?candidate, err = ?e, "decode failed; trying next backup");
                        last_err = Some(e.into());
                    }
                },
                Err(e) => {
                    warn!(event = "FILTER_DOWN_IO_FAIL", ?candidate, err = ?e, "io failed; trying next backup");
                    last_err = Some(e.into());
                }
            }
        }

        if !found_any {
            return Ok(false);
        }
        Err(last_err.unwrap_or(FilterDownError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "all backups failed",
        ))))
    }

    /// One TD(0) batch update. Returns mean MSE loss BEFORE the gradient
    /// descent step. TD target: `target_i = reward_i + γ × V(next_state_i)`
    /// (target net = self). Empty batch is a no-op (returns 0.0).
    pub fn train_step(
        &mut self,
        states: &[[f64; N]],
        rewards: &[f64],
        next_states: &[[f64; N]],
        lr: f64,
        gamma: f64,
    ) -> f64 {
        debug_assert_eq!(states.len(), rewards.len());
        debug_assert_eq!(states.len(), next_states.len());
        let n = states.len();
        if n == 0 {
            return 0.0;
        }
        let n_f = n as f64;

        let mut z1_batch = vec![vec![0.0_f64; HIDDEN_1]; n];
        let mut a1_batch = vec![vec![0.0_f64; HIDDEN_1]; n];
        let mut z2_batch = vec![vec![0.0_f64; HIDDEN_2]; n];
        let mut a2_batch = vec![vec![0.0_f64; HIDDEN_2]; n];
        let mut values = vec![0.0_f64; n];
        for i in 0..n {
            z1_batch[i] = self.layer1_z(&states[i]);
            a1_batch[i] = relu_inplace(z1_batch[i].clone());
            z2_batch[i] = self.layer2_z(&a1_batch[i]);
            a2_batch[i] = relu_inplace(z2_batch[i].clone());
            values[i] = self.layer3_z(&a2_batch[i]);
        }

        let mut next_values = vec![0.0_f64; n];
        for i in 0..n {
            let z1n = self.layer1_z(&next_states[i]);
            let a1n = relu_inplace(z1n);
            let z2n = self.layer2_z(&a1n);
            let a2n = relu_inplace(z2n);
            next_values[i] = self.layer3_z(&a2n);
        }

        let mut targets = vec![0.0_f64; n];
        let mut errors = vec![0.0_f64; n];
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            targets[i] = rewards[i] + gamma * next_values[i];
            errors[i] = values[i] - targets[i];
            sum_sq += errors[i] * errors[i];
        }
        let loss = sum_sq / n_f;

        let mut dv = vec![0.0_f64; n];
        for i in 0..n {
            dv[i] = 2.0 * errors[i] / n_f;
        }

        let mut dw3 = vec![0.0_f64; HIDDEN_2];
        let mut db3 = 0.0_f64;
        for i in 0..n {
            let dz3_i = dv[i];
            for (slot, &a) in dw3.iter_mut().zip(a2_batch[i].iter()) {
                *slot += a * dz3_i;
            }
            db3 += dz3_i;
        }

        let mut dw2 = vec![0.0_f64; HIDDEN_1 * HIDDEN_2];
        let mut db2 = vec![0.0_f64; HIDDEN_2];
        let mut dz2_batch = vec![vec![0.0_f64; HIDDEN_2]; n];
        for i in 0..n {
            let dz3_i = dv[i];
            for h2 in 0..HIDDEN_2 {
                let da2_i = dz3_i * self.w3[h2];
                let g = if z2_batch[i][h2] > 0.0 { da2_i } else { 0.0 };
                dz2_batch[i][h2] = g;
                db2[h2] += g;
            }
            for (h1, &a) in a1_batch[i].iter().enumerate() {
                let row_off = h1 * HIDDEN_2;
                for (h2, &dz) in dz2_batch[i].iter().enumerate() {
                    dw2[row_off + h2] += a * dz;
                }
            }
        }

        let mut dw1 = vec![0.0_f64; N * HIDDEN_1];
        let mut db1 = vec![0.0_f64; HIDDEN_1];
        for i in 0..n {
            let mut da1 = vec![0.0_f64; HIDDEN_1];
            for (h1, slot) in da1.iter_mut().enumerate() {
                let row_off = h1 * HIDDEN_2;
                let mut sum = 0.0;
                for (h2, &dz) in dz2_batch[i].iter().enumerate() {
                    sum += dz * self.w2[row_off + h2];
                }
                *slot = sum;
            }
            let mut dz1 = vec![0.0_f64; HIDDEN_1];
            for (h1, slot) in dz1.iter_mut().enumerate() {
                *slot = if z1_batch[i][h1] > 0.0 { da1[h1] } else { 0.0 };
                db1[h1] += *slot;
            }
            for (k, &s) in states[i].iter().enumerate() {
                let row_off = k * HIDDEN_1;
                for (h1, &dz) in dz1.iter().enumerate() {
                    dw1[row_off + h1] += s * dz;
                }
            }
        }

        for (w, dw) in self.w3.iter_mut().zip(dw3.iter()) {
            *w -= lr * dw;
        }
        self.b3[0] -= lr * db3;
        for (w, dw) in self.w2.iter_mut().zip(dw2.iter()) {
            *w -= lr * dw;
        }
        for (b, db) in self.b2.iter_mut().zip(db2.iter()) {
            *b -= lr * db;
        }
        for (w, dw) in self.w1.iter_mut().zip(dw1.iter()) {
            *w -= lr * dw;
        }
        for (b, db) in self.b1.iter_mut().zip(db1.iter()) {
            *b -= lr * db;
        }

        loss
    }
}

impl<const N: usize> Default for TrinityValueNet<N> {
    fn default() -> Self {
        Self::new(&mut rand::thread_rng())
    }
}

/// ReLU activation in place: `z[i] = max(0, z[i])`.
fn relu_inplace(mut z: Vec<f64>) -> Vec<f64> {
    for v in z.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    z
}

fn matrix_to_nested(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    debug_assert_eq!(flat.len(), rows * cols);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        out.push(flat[r * cols..(r + 1) * cols].to_vec());
    }
    out
}

fn nested_to_matrix(nested: &[Vec<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(nested.len() * nested.first().map(|r| r.len()).unwrap_or(0));
    for row in nested {
        out.extend_from_slice(row);
    }
    out
}

fn check_matrix_shape(
    nested: &[Vec<f64>],
    rows: usize,
    cols: usize,
    layer: &'static str,
) -> Result<(), FilterDownError> {
    if nested.len() != rows {
        return Err(FilterDownError::ShapeMismatch {
            layer,
            expected: rows,
            actual: nested.len(),
        });
    }
    for row in nested {
        if row.len() != cols {
            return Err(FilterDownError::ShapeMismatch {
                layer,
                expected: cols,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

fn check_vec_shape(v: &[f64], n: usize, layer: &'static str) -> Result<(), FilterDownError> {
    if v.len() != n {
        return Err(FilterDownError::ShapeMismatch {
            layer,
            expected: n,
            actual: v.len(),
        });
    }
    Ok(())
}
