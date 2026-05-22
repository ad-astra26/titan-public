//! transition_buffer — shared capped replay buffer for the filter_down
//! TD(0) learners (unified `FilterDownV5Engine` + per-half
//! `SmallFilterDownEngine`).
//!
//! Dim-agnostic: a `Transition` stores `state` / `next_state` as `Vec<f64>`
//! of whatever dim the owning engine uses (162 for unified, 65 per half).
//! Per `filter_down.py:259-303`: FIFO eviction when full; `sample` returns
//! `n = min(batch_size, len)` transitions WITHOUT replacement.

use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::trinity_value_net::FilterDownError;

/// Single transition record. Matches Python `(state, reward, next_state)`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transition {
    /// State `s` at time t.
    pub state: Vec<f64>,
    /// Reward `r` observed in transition s → s'.
    pub reward: f64,
    /// Next state `s'` at time t+1.
    pub next_state: Vec<f64>,
}

/// Capped ring buffer of `(state, reward, next_state)` transitions.
#[derive(Debug, Clone)]
pub struct TransitionBuffer {
    buffer: Vec<Transition>,
    max_size: usize,
    write_idx: usize,
}

impl TransitionBuffer {
    /// Construct an empty buffer with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            write_idx: 0,
        }
    }

    /// Construct with SPEC default capacity (`FILTER_DOWN_BUFFER_MAX`).
    pub fn with_defaults() -> Self {
        Self::new(crate::constants::FILTER_DOWN_BUFFER_MAX as usize)
    }

    /// Add one transition. Append until cap, then overwrite at write_idx
    /// (FIFO eviction). Advances write_idx mod max_size.
    pub fn add(&mut self, state: Vec<f64>, reward: f64, next_state: Vec<f64>) {
        debug_assert_eq!(
            state.len(),
            next_state.len(),
            "state / next_state dim mismatch"
        );
        let t = Transition {
            state,
            reward,
            next_state,
        };
        if self.buffer.len() < self.max_size {
            self.buffer.push(t);
        } else {
            self.buffer[self.write_idx] = t;
        }
        self.write_idx = (self.write_idx + 1) % self.max_size;
    }

    /// Random mini-batch sampler — `n = min(batch_size, len)` transitions
    /// without replacement.
    pub fn sample<R: Rng + ?Sized>(&self, batch_size: usize, rng: &mut R) -> Vec<Transition> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let n = batch_size.min(self.buffer.len());
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(rng);
        indices.truncate(n);
        indices.iter().map(|&i| self.buffer[i].clone()).collect()
    }

    /// Number of transitions stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True if no transitions stored.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Capacity (max_size).
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Current FIFO write index (for introspection / white-box tests).
    pub fn write_idx(&self) -> usize {
        self.write_idx
    }

    /// Read-only view of stored transitions (for introspection / white-box tests).
    pub fn transitions(&self) -> &[Transition] {
        &self.buffer
    }

    /// Save buffer to JSON via `atomic_write` + retention.
    pub fn save(&self, path: &Path) -> Result<(), FilterDownError> {
        let bytes = serde_json::to_vec_pretty(&self.buffer)?;
        crate::atomic_write::atomic_write(
            path,
            &bytes,
            crate::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load buffer from JSON (`.bak` / `.bak.prev` fallback). Recomputes
    /// `write_idx = len % max_size`. Returns `Ok(false)` when no file present.
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
                Ok(bytes) => match serde_json::from_slice::<Vec<Transition>>(&bytes) {
                    Ok(buffer) => {
                        let buffer = if buffer.len() > self.max_size {
                            buffer.into_iter().take(self.max_size).collect()
                        } else {
                            buffer
                        };
                        let len = buffer.len();
                        self.buffer = buffer;
                        self.write_idx = len % self.max_size;
                        info!(
                            event = "TRANSITION_BUFFER_LOADED",
                            ?candidate,
                            len = len,
                            "transition buffer restored"
                        );
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!(event = "TRANSITION_BUFFER_DECODE_FAIL", ?candidate, err = ?e, "decode failed; trying next backup");
                        last_err = Some(e.into());
                    }
                },
                Err(e) => {
                    warn!(event = "TRANSITION_BUFFER_IO_FAIL", ?candidate, err = ?e, "io failed; trying next backup");
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
}

impl Default for TransitionBuffer {
    fn default() -> Self {
        Self::with_defaults()
    }
}
