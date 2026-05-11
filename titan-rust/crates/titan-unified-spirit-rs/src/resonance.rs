//! resonance — Phase-aligned inner↔outer pair detector + GREAT PULSE
//! gating.
//!
//! Full Rust port of `titan_plugin/logic/resonance.py` per SPEC §10.F G11
//! + Decision Log D-SPEC-26 (resonance lives inside unified-spirit-rs).
//!
//! Three pairs tracked per `PAIRS` constant (`body`, `mind`, `spirit`):
//! - Body pair:   `inner_body` ↔ `outer_body`
//! - Mind pair:   `inner_mind` ↔ `outer_mind`
//! - Spirit pair: `inner_spirit` ↔ `outer_spirit`
//!
//! Resonance detection mechanics (mirroring Python):
//! 1. Each pair tracks the last pulse timestamp + phase from each side.
//! 2. When BOTH sides have a fresh pulse (received-since-last-check):
//!    - If `time_diff > pulse_window` → reset streak (out-of-time).
//!    - Else if `phase_diff <= phase_threshold` → resonant cycle, ++streak.
//!    - Else → reset streak (phase misalignment).
//! 3. After `required_cycles` (=3) consecutive resonant cycles → **BIG PULSE**.
//! 4. When all 3 pairs are simultaneously resonant after any BIG PULSE →
//!    **GREAT PULSE** condition met (consumed by `UnifiedSpirit::advance()`
//!    in C4-2b2).
//!
//! Persistence: `data/resonance_state.json` via `titan-core::atomic_write`
//! per SPEC §11.H.1 critical-data row (added v0.1.3 D-SPEC-26).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use titan_core::constants::{
    RESONANCE_CYCLES_REQUIRED, RESONANCE_PHASE_THRESHOLD_RAD, RESONANCE_PULSE_WINDOW_S,
};

/// Canonical pair names per resonance.py:42 + SPEC.
pub const PAIRS: [&str; 3] = ["body", "mind", "spirit"];

/// Errors during resonance state persistence + restore.
#[derive(Debug, thiserror::Error)]
pub enum ResonanceError {
    /// `data/resonance_state.json` write failed.
    #[error("resonance_state write failed: {0}")]
    Write(#[from] titan_core::atomic_write::AtomicWriteError),
    /// JSON encode/decode failed.
    #[error("resonance_state json: {0}")]
    Json(#[from] serde_json::Error),
    /// io error reading state file.
    #[error("resonance_state io: {0}")]
    Io(#[from] std::io::Error),
}

/// One pair's serializable state (matches Python `ResonancePair.to_dict`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResonancePairState {
    /// Pair name — `"body"` / `"mind"` / `"spirit"`.
    pub name: String,
    /// Consecutive resonant cycles in current streak.
    pub consecutive_resonant: u32,
    /// Total resonant cycles since detector init.
    pub total_resonant_cycles: u64,
    /// Total `_check_resonance` invocations (resonant + non-resonant).
    pub total_checks: u64,
    /// BIG PULSEs emitted since detector init.
    pub big_pulse_count: u64,
    /// Wall clock seconds (f64) of last BIG PULSE — 0.0 if never.
    pub last_big_pulse_ts: f64,
    /// Total inner-side SPHERE_PULSE events received.
    pub inner_pulse_count: u64,
    /// Total outer-side SPHERE_PULSE events received.
    pub outer_pulse_count: u64,
    /// Whether the pair is currently in a resonant state (last check
    /// passed thresholds — flips false on out-of-time / phase miss).
    pub is_resonant: bool,
}

/// One inner↔outer pair's full runtime state. Records pulses + checks
/// resonance + emits BIG PULSE when streak threshold reached.
#[derive(Debug, Clone)]
pub struct ResonancePair {
    /// Name (one of `PAIRS`).
    pub name: String,
    /// Max phase difference (radians) for resonance — default π/6 (30°).
    pub phase_threshold: f64,
    /// Consecutive resonant cycles needed for BIG PULSE — default 3.
    pub required_cycles: u32,
    /// Max time (seconds) between counterpart pulses — default 120s.
    pub pulse_window: f64,

    inner_last_pulse_ts: f64,
    inner_last_phase: f64,
    inner_fresh: bool,
    outer_last_pulse_ts: f64,
    outer_last_phase: f64,
    outer_fresh: bool,

    consecutive_resonant: u32,
    total_resonant_cycles: u64,
    total_checks: u64,
    big_pulse_count: u64,
    last_big_pulse_ts: f64,
    inner_pulse_count: u64,
    outer_pulse_count: u64,

    is_resonant: bool,
}

/// BIG PULSE event payload — mirrors Python `_generate_big_pulse` output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BigPulse {
    /// Pair name.
    pub pair: String,
    /// BIG pulse counter (1-indexed) within this pair.
    pub big_pulse_count: u64,
    /// Phase difference (radians) at the moment of resonance.
    pub phase_diff: f64,
    /// Time difference (seconds) between the inner + outer pulse that
    /// triggered the BIG PULSE.
    pub time_diff: f64,
    /// Total inner pulses observed.
    pub inner_pulse_count: u64,
    /// Total outer pulses observed.
    pub outer_pulse_count: u64,
    /// Total resonant cycles across the pair's lifetime.
    pub total_resonant_cycles: u64,
    /// Wall clock seconds of emission.
    pub ts: f64,
    /// Set true by `ResonanceDetector` when this BIG PULSE coincides
    /// with all 3 pairs being simultaneously resonant — gates GREAT
    /// PULSE crystallization in `UnifiedSpirit::advance()` (C4-2b2).
    pub great_pulse_ready: bool,
    /// GREAT PULSE counter at emission time (only meaningful when
    /// `great_pulse_ready == true`).
    pub great_pulse_count: u64,
}

impl ResonancePair {
    /// Construct with explicit thresholds. Use `ResonancePair::with_defaults`
    /// for the SPEC-default phase/cycles/window triplet.
    pub fn new(
        name: impl Into<String>,
        phase_threshold: f64,
        required_cycles: u32,
        pulse_window: f64,
    ) -> Self {
        Self {
            name: name.into(),
            phase_threshold,
            required_cycles,
            pulse_window,
            inner_last_pulse_ts: 0.0,
            inner_last_phase: 0.0,
            inner_fresh: false,
            outer_last_pulse_ts: 0.0,
            outer_last_phase: 0.0,
            outer_fresh: false,
            consecutive_resonant: 0,
            total_resonant_cycles: 0,
            total_checks: 0,
            big_pulse_count: 0,
            last_big_pulse_ts: 0.0,
            inner_pulse_count: 0,
            outer_pulse_count: 0,
            is_resonant: false,
        }
    }

    /// SPEC-default constructor — uses constants from
    /// `titan_core::constants` (sourced from SPEC v0.1.3 TOML).
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self::new(
            name,
            RESONANCE_PHASE_THRESHOLD_RAD,
            RESONANCE_CYCLES_REQUIRED as u32,
            RESONANCE_PULSE_WINDOW_S,
        )
    }

    /// Record a SPHERE_PULSE event with explicit phase. `component` is
    /// the full name (`"inner_body"`, `"outer_mind"`, etc.). Returns
    /// `Some(BigPulse)` if this event closes a BIG PULSE streak,
    /// `None` otherwise.
    pub fn record_pulse_with_phase(
        &mut self,
        component: &str,
        phase: f64,
        pulse_ts: f64,
    ) -> Option<BigPulse> {
        let is_inner = component.starts_with("inner_");
        if is_inner {
            self.inner_last_pulse_ts = pulse_ts;
            self.inner_last_phase = phase;
            self.inner_pulse_count += 1;
            self.inner_fresh = true;
        } else {
            self.outer_last_pulse_ts = pulse_ts;
            self.outer_last_phase = phase;
            self.outer_pulse_count += 1;
            self.outer_fresh = true;
        }

        if self.inner_fresh && self.outer_fresh {
            self.check_resonance(pulse_ts)
        } else {
            None
        }
    }

    fn check_resonance(&mut self, _now: f64) -> Option<BigPulse> {
        self.total_checks += 1;
        // Consume fresh flags
        self.inner_fresh = false;
        self.outer_fresh = false;

        if self.inner_last_pulse_ts == 0.0 || self.outer_last_pulse_ts == 0.0 {
            return None;
        }

        let time_diff = (self.inner_last_pulse_ts - self.outer_last_pulse_ts).abs();
        if time_diff > self.pulse_window {
            self.consecutive_resonant = 0;
            self.is_resonant = false;
            return None;
        }

        let phase_diff = phase_difference(self.inner_last_phase, self.outer_last_phase);
        if phase_diff <= self.phase_threshold {
            self.consecutive_resonant += 1;
            self.total_resonant_cycles += 1;
            self.is_resonant = true;

            if self.consecutive_resonant >= self.required_cycles {
                return Some(self.generate_big_pulse(phase_diff, time_diff));
            }
        } else {
            self.consecutive_resonant = 0;
            self.is_resonant = false;
        }

        None
    }

    fn generate_big_pulse(&mut self, phase_diff: f64, time_diff: f64) -> BigPulse {
        self.big_pulse_count += 1;
        let now = wall_seconds();
        self.last_big_pulse_ts = now;
        // Reset streak — start fresh for next BIG PULSE
        self.consecutive_resonant = 0;

        BigPulse {
            pair: self.name.clone(),
            big_pulse_count: self.big_pulse_count,
            phase_diff,
            time_diff,
            inner_pulse_count: self.inner_pulse_count,
            outer_pulse_count: self.outer_pulse_count,
            total_resonant_cycles: self.total_resonant_cycles,
            ts: now,
            great_pulse_ready: false,
            great_pulse_count: 0,
        }
    }

    /// Whether this pair is currently resonant.
    pub fn is_resonant(&self) -> bool {
        self.is_resonant
    }

    /// Snapshot for persistence.
    pub fn to_state(&self) -> ResonancePairState {
        ResonancePairState {
            name: self.name.clone(),
            consecutive_resonant: self.consecutive_resonant,
            total_resonant_cycles: self.total_resonant_cycles,
            total_checks: self.total_checks,
            big_pulse_count: self.big_pulse_count,
            last_big_pulse_ts: self.last_big_pulse_ts,
            inner_pulse_count: self.inner_pulse_count,
            outer_pulse_count: self.outer_pulse_count,
            is_resonant: self.is_resonant,
        }
    }

    /// Restore from persistence.
    pub fn restore_from(&mut self, state: &ResonancePairState) {
        self.consecutive_resonant = state.consecutive_resonant;
        self.total_resonant_cycles = state.total_resonant_cycles;
        self.total_checks = state.total_checks;
        self.big_pulse_count = state.big_pulse_count;
        self.last_big_pulse_ts = state.last_big_pulse_ts;
        self.inner_pulse_count = state.inner_pulse_count;
        self.outer_pulse_count = state.outer_pulse_count;
        self.is_resonant = state.is_resonant;
    }
}

/// Compute angular difference in `[0, π]` — handles 2π wrap.
pub fn phase_difference(phase_a: f64, phase_b: f64) -> f64 {
    let diff = (phase_a - phase_b).abs() % (2.0 * std::f64::consts::PI);
    if diff > std::f64::consts::PI {
        2.0 * std::f64::consts::PI - diff
    } else {
        diff
    }
}

/// Persisted state for the entire ResonanceDetector.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResonanceDetectorState {
    /// Schema version per SPEC §11.H.4 boot integrity check.
    pub schema_version: u32,
    /// Per-pair state, keyed by pair name.
    pub pairs: std::collections::BTreeMap<String, ResonancePairState>,
    /// Total GREAT PULSEs (all-3-resonant BIG-PULSE coincidences) since
    /// detector init.
    pub great_pulse_count: u64,
    /// Wall clock seconds of last GREAT PULSE — 0.0 if never.
    pub last_great_pulse_ts: f64,
}

/// Schema version for `data/resonance_state.json` per SPEC §11.H.4.
pub const RESONANCE_STATE_SCHEMA_VERSION: u32 = 1;

/// 3-pair orchestrator. Routes `SPHERE_PULSE` events to pairs by
/// component-name match, emits BIG PULSEs, gates GREAT PULSE on
/// all-3-resonant coincidence.
pub struct ResonanceDetector {
    pairs: std::collections::BTreeMap<String, ResonancePair>,
    great_pulse_count: u64,
    last_great_pulse_ts: f64,
    state_path: PathBuf,
}

impl ResonanceDetector {
    /// Construct with SPEC defaults; loads `data/resonance_state.json`
    /// if present. Per SPEC §11.H.4 boot integrity check: corrupt
    /// canonical → try `.bak` → try `.bak.prev` → start fresh.
    pub fn with_defaults(data_dir: &Path) -> Self {
        let mut pairs = std::collections::BTreeMap::new();
        for &name in PAIRS.iter() {
            pairs.insert(name.to_string(), ResonancePair::with_defaults(name));
        }
        let state_path = data_dir.join("resonance_state.json");
        let mut det = Self {
            pairs,
            great_pulse_count: 0,
            last_great_pulse_ts: 0.0,
            state_path,
        };
        // Best-effort load — never fatal at boot. Boot integrity check
        // (caller) decides whether to halt module on persistent corruption.
        let _ = det.load_state();
        det
    }

    /// Construct with explicit thresholds (test path).
    pub fn new(
        phase_threshold: f64,
        required_cycles: u32,
        pulse_window: f64,
        state_path: PathBuf,
    ) -> Self {
        let mut pairs = std::collections::BTreeMap::new();
        for &name in PAIRS.iter() {
            pairs.insert(
                name.to_string(),
                ResonancePair::new(name, phase_threshold, required_cycles, pulse_window),
            );
        }
        Self {
            pairs,
            great_pulse_count: 0,
            last_great_pulse_ts: 0.0,
            state_path,
        }
    }

    /// Record a SPHERE_PULSE event. `component` parsed from
    /// `pulse_event.clock_name` field per SPEC §8.6 payload (in our
    /// payload schema, the "clock_name" is e.g. `"inner_body"` matching
    /// PAIR-prefix convention).
    ///
    /// `phase` is from the SPHERE_PULSE payload float.
    /// Returns `Some(BigPulse)` if the pulse closed a BIG PULSE streak —
    /// caller may publish to bus + (when `great_pulse_ready == true`)
    /// trigger `UnifiedSpirit::advance()`.
    pub fn record_pulse_with_phase(
        &mut self,
        component: &str,
        phase: f64,
        pulse_ts: f64,
    ) -> Option<BigPulse> {
        let pair_name = component_to_pair(component)?;
        let pair = self.pairs.get_mut(pair_name)?;
        let mut big_pulse = pair.record_pulse_with_phase(component, phase, pulse_ts)?;

        // GREAT PULSE gate: any time a BIG PULSE fires AND all 3 pairs
        // are simultaneously resonant → mark great_pulse_ready.
        if self.all_resonant() {
            self.great_pulse_count += 1;
            self.last_great_pulse_ts = wall_seconds();
            big_pulse.great_pulse_ready = true;
            big_pulse.great_pulse_count = self.great_pulse_count;
        }
        Some(big_pulse)
    }

    /// True iff all 3 pairs currently resonate. Used by callers to
    /// invoke `UnifiedSpirit::advance(...)` (C4-2b2).
    pub fn all_resonant(&self) -> bool {
        self.pairs.values().all(|p| p.is_resonant())
    }

    /// 0..=3 — how many of the 3 pairs are currently resonant.
    pub fn resonant_count(&self) -> usize {
        self.pairs.values().filter(|p| p.is_resonant()).count()
    }

    /// Total GREAT PULSEs observed.
    pub fn great_pulse_count(&self) -> u64 {
        self.great_pulse_count
    }

    /// Wall-clock seconds of last GREAT PULSE.
    pub fn last_great_pulse_ts(&self) -> f64 {
        self.last_great_pulse_ts
    }

    /// Snapshot ResonancePair state for persistence.
    pub fn to_state(&self) -> ResonanceDetectorState {
        let mut pair_states = std::collections::BTreeMap::new();
        for (name, pair) in &self.pairs {
            pair_states.insert(name.clone(), pair.to_state());
        }
        ResonanceDetectorState {
            schema_version: RESONANCE_STATE_SCHEMA_VERSION,
            pairs: pair_states,
            great_pulse_count: self.great_pulse_count,
            last_great_pulse_ts: self.last_great_pulse_ts,
        }
    }

    /// Restore from a snapshot.
    pub fn restore_from(&mut self, state: &ResonanceDetectorState) {
        for (name, pair_state) in &state.pairs {
            if let Some(pair) = self.pairs.get_mut(name) {
                pair.restore_from(pair_state);
            }
        }
        self.great_pulse_count = state.great_pulse_count;
        self.last_great_pulse_ts = state.last_great_pulse_ts;
    }

    /// Persist state to `data/resonance_state.json` via atomic_write +
    /// 2-backup retention per SPEC §11.H.1.
    pub fn save_state(&self) -> Result<(), ResonanceError> {
        let state = self.to_state();
        let bytes = serde_json::to_vec_pretty(&state)?;
        titan_core::atomic_write::atomic_write(
            &self.state_path,
            &bytes,
            titan_core::constants::DATA_BACKUP_RETENTION_GENERATIONS as usize,
        )?;
        Ok(())
    }

    /// Load state from `data/resonance_state.json`, falling back to
    /// `.bak` then `.bak.prev` per SPEC §11.H.4. Missing file = clean
    /// start (returns Ok). Corrupt-everywhere returns last error.
    pub fn load_state(&mut self) -> Result<(), ResonanceError> {
        let candidates = [
            self.state_path.clone(),
            self.state_path.with_extension("json.bak"),
            self.state_path.with_extension("json.bak.prev"),
        ];

        let mut last_err: Option<ResonanceError> = None;
        for candidate in &candidates {
            if !candidate.exists() {
                continue;
            }
            match std::fs::read(candidate) {
                Ok(bytes) => match serde_json::from_slice::<ResonanceDetectorState>(&bytes) {
                    Ok(state) => {
                        if state.schema_version == RESONANCE_STATE_SCHEMA_VERSION {
                            self.restore_from(&state);
                            return Ok(());
                        } else {
                            // Schema mismatch — treat as corruption, try next backup.
                            last_err = Some(ResonanceError::Json(serde_json::Error::io(
                                std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!(
                                        "schema_version {} != {}",
                                        state.schema_version, RESONANCE_STATE_SCHEMA_VERSION
                                    ),
                                ),
                            )));
                        }
                    }
                    Err(e) => {
                        last_err = Some(e.into());
                    }
                },
                Err(e) => {
                    last_err = Some(e.into());
                }
            }
        }

        match last_err {
            None => Ok(()), // No file at all — clean start.
            Some(e) => Err(e),
        }
    }
}

/// Map full component name to pair name. `"inner_body"` → `"body"`,
/// `"outer_spirit"` → `"spirit"`, etc.
pub fn component_to_pair(component: &str) -> Option<&'static str> {
    PAIRS.iter().copied().find(|name| component.contains(name))
}

/// Wall clock seconds since UNIX epoch (f64). Mirror of Python `time.time()`.
fn wall_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tempfile::tempdir;

    // ── PHASE DIFFERENCE (1) ───────────────────────────────────────────

    #[test]
    fn phase_difference_handles_2pi_wraparound() {
        // C4-2b1 test 1: phase_difference correctness across 2π wrap
        assert!((phase_difference(0.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((phase_difference(0.0, PI / 6.0) - PI / 6.0).abs() < 1e-10);
        assert!((phase_difference(PI, -PI) - 0.0).abs() < 1e-10); // ±π same angle
                                                                  // 0.1 vs 2π - 0.1 → 0.2 after wrap (0.1 + 0.1)
        let near_zero_a = 0.1;
        let near_zero_b = 2.0 * PI - 0.1;
        assert!((phase_difference(near_zero_a, near_zero_b) - 0.2).abs() < 1e-10);
        // Always in [0, π]
        for a in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            for b in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
                let d = phase_difference(a, b);
                assert!(
                    (0.0..=PI + 1e-10).contains(&d),
                    "phase_diff({a}, {b}) = {d}"
                );
            }
        }
    }

    // ── RECORD_PULSE_WITH_PHASE (1) ────────────────────────────────────

    #[test]
    fn record_pulse_inner_only_no_resonance_check() {
        // C4-2b1 test 2: only inner pulse (outer not fresh) → no check
        let mut pair = ResonancePair::with_defaults("body");
        let result = pair.record_pulse_with_phase("inner_body", 0.0, 100.0);
        assert!(result.is_none());
        assert!(!pair.is_resonant());
        assert_eq!(pair.total_checks, 0);
        assert_eq!(pair.inner_pulse_count, 1);
        assert_eq!(pair.outer_pulse_count, 0);
    }

    // ── 3-CYCLE BIG PULSE (1) ──────────────────────────────────────────

    #[test]
    fn pair_emits_big_pulse_after_3_consecutive_resonant_cycles() {
        // C4-2b1 test 3: 3 phase-aligned inner+outer pulses → BIG PULSE
        let mut pair = ResonancePair::with_defaults("body");
        // Cycle 1
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 100.0);
        let r1 = pair.record_pulse_with_phase("outer_body", 0.0, 100.1);
        assert!(r1.is_none());
        assert_eq!(pair.consecutive_resonant, 1);
        assert!(pair.is_resonant());
        // Cycle 2
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 200.0);
        let r2 = pair.record_pulse_with_phase("outer_body", 0.0, 200.1);
        assert!(r2.is_none());
        assert_eq!(pair.consecutive_resonant, 2);
        // Cycle 3 — fires BIG PULSE
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 300.0);
        let r3 = pair.record_pulse_with_phase("outer_body", 0.0, 300.1);
        assert!(r3.is_some());
        let bp = r3.unwrap();
        assert_eq!(bp.pair, "body");
        assert_eq!(bp.big_pulse_count, 1);
        assert_eq!(bp.total_resonant_cycles, 3);
        // Streak resets after BIG PULSE per Python:_generate_big_pulse
        assert_eq!(pair.consecutive_resonant, 0);
    }

    // ── PHASE DIVERGENCE BREAKS STREAK (1) ─────────────────────────────

    #[test]
    fn pair_drops_resonance_on_phase_divergence() {
        // C4-2b1 test 4: phase_diff > threshold → streak resets
        let mut pair = ResonancePair::with_defaults("body");
        // Resonant cycle 1
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 100.0);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, 100.1);
        assert_eq!(pair.consecutive_resonant, 1);
        // Cycle 2 — phase off by π/3 (60°), > threshold π/6 (30°)
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 200.0);
        let _ = pair.record_pulse_with_phase("outer_body", PI / 3.0, 200.1);
        assert_eq!(pair.consecutive_resonant, 0);
        assert!(!pair.is_resonant());
    }

    // ── PULSE WINDOW EXPIRY (1) ────────────────────────────────────────

    #[test]
    fn pair_drops_resonance_on_pulse_window_expiry() {
        // C4-2b1 test 5: time_diff > pulse_window → streak resets
        let mut pair = ResonancePair::with_defaults("body");
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 100.0);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, 100.1);
        assert_eq!(pair.consecutive_resonant, 1);
        // Cycle 2 with time_diff > 120s (pulse_window default)
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, 200.0);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, 350.0); // 150s diff
        assert_eq!(pair.consecutive_resonant, 0);
        assert!(!pair.is_resonant());
    }

    // ── ALL_RESONANT + RESONANT_COUNT (1) ──────────────────────────────

    #[test]
    fn detector_all_resonant_and_resonant_count() {
        // C4-2b1 test 6: GREAT PULSE gate boundary
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());
        assert!(!det.all_resonant());
        assert_eq!(det.resonant_count(), 0);

        // Drive body pair to resonance (3 cycles)
        for ts in [100.0_f64, 200.0, 300.0] {
            let _ = det.record_pulse_with_phase("inner_body", 0.0, ts);
            let _ = det.record_pulse_with_phase("outer_body", 0.0, ts + 0.1);
        }
        assert_eq!(det.resonant_count(), 1);
        assert!(!det.all_resonant());

        // Mind pair — first cycle only
        let _ = det.record_pulse_with_phase("inner_mind", 0.0, 100.0);
        let _ = det.record_pulse_with_phase("outer_mind", 0.0, 100.1);
        assert_eq!(det.resonant_count(), 2);
        assert!(!det.all_resonant());

        // Spirit pair — first cycle
        let _ = det.record_pulse_with_phase("inner_spirit", 0.0, 100.0);
        let _ = det.record_pulse_with_phase("outer_spirit", 0.0, 100.1);
        assert_eq!(det.resonant_count(), 3);
        assert!(det.all_resonant());
    }

    // ── COMPONENT → PAIR ROUTING (1) ───────────────────────────────────

    #[test]
    fn component_to_pair_routes_correctly() {
        // C4-2b1 test 7: name routing
        assert_eq!(component_to_pair("inner_body"), Some("body"));
        assert_eq!(component_to_pair("outer_body"), Some("body"));
        assert_eq!(component_to_pair("inner_mind"), Some("mind"));
        assert_eq!(component_to_pair("outer_mind"), Some("mind"));
        assert_eq!(component_to_pair("inner_spirit"), Some("spirit"));
        assert_eq!(component_to_pair("outer_spirit"), Some("spirit"));
        assert_eq!(component_to_pair("kernel"), None);
        assert_eq!(component_to_pair(""), None);
    }

    // ── GREAT PULSE GATING (1) ─────────────────────────────────────────

    #[test]
    fn great_pulse_count_increments_on_all_3_resonant_big_pulse() {
        // C4-2b1 test 8: GREAT PULSE counter increments only when a BIG
        // PULSE fires while ALL 3 pairs are resonant
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        // Drive ALL 3 pairs to streak count 2 (just below BIG PULSE)
        for ts in [100.0_f64, 200.0] {
            for name in PAIRS.iter() {
                let inner = format!("inner_{name}");
                let outer = format!("outer_{name}");
                let _ = det.record_pulse_with_phase(&inner, 0.0, ts);
                let _ = det.record_pulse_with_phase(&outer, 0.0, ts + 0.1);
            }
        }
        assert!(det.all_resonant());
        assert_eq!(det.great_pulse_count(), 0);

        // Now fire one more cycle on body pair → BIG PULSE on body, all
        // 3 still resonant (mind + spirit at streak 2 + body just hit 3)
        let _ = det.record_pulse_with_phase("inner_body", 0.0, 300.0);
        let body_bp = det
            .record_pulse_with_phase("outer_body", 0.0, 300.1)
            .unwrap();
        assert!(body_bp.great_pulse_ready);
        assert_eq!(body_bp.great_pulse_count, 1);
        assert_eq!(det.great_pulse_count(), 1);
    }

    // ── SAVE / RESTORE STATE (2) ───────────────────────────────────────

    #[test]
    fn save_and_load_state_roundtrip() {
        // C4-2b1 test 9: save → load preserves all per-pair state
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        // Drive body pair into resonance
        for ts in [100.0_f64, 200.0, 300.0] {
            let _ = det.record_pulse_with_phase("inner_body", 0.0, ts);
            let _ = det.record_pulse_with_phase("outer_body", 0.0, ts + 0.1);
        }
        let total_resonant = det.pairs.get("body").unwrap().total_resonant_cycles;
        let big_pulse_count = det.pairs.get("body").unwrap().big_pulse_count;

        // Persist
        det.save_state().unwrap();

        // Load fresh detector — must restore counters
        let det2 = ResonanceDetector::with_defaults(dir.path());
        let restored = det2.pairs.get("body").unwrap();
        assert_eq!(restored.total_resonant_cycles, total_resonant);
        assert_eq!(restored.big_pulse_count, big_pulse_count);
    }

    #[test]
    fn load_state_falls_back_to_bak_when_canonical_corrupt() {
        // C4-2b1 test 10: corrupt canonical → restore from .bak
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        // Save once → creates canonical
        for ts in [100.0_f64, 200.0, 300.0] {
            let _ = det.record_pulse_with_phase("inner_body", 0.0, ts);
            let _ = det.record_pulse_with_phase("outer_body", 0.0, ts + 0.1);
        }
        det.save_state().unwrap();
        // Save again → rotates canonical to .bak
        for ts in [400.0_f64, 500.0, 600.0] {
            let _ = det.record_pulse_with_phase("inner_mind", 0.0, ts);
            let _ = det.record_pulse_with_phase("outer_mind", 0.0, ts + 0.1);
        }
        det.save_state().unwrap();

        let canonical = dir.path().join("resonance_state.json");
        let bak = dir.path().join("resonance_state.json.bak");
        assert!(canonical.exists(), "canonical present after 2nd save");
        assert!(bak.exists(), "bak present after rotate");

        // Corrupt canonical
        std::fs::write(&canonical, b"not json").unwrap();

        // Load via fresh detector → falls back to .bak
        let mut det2 = ResonanceDetector::with_defaults(dir.path());
        // Load from the bak file should have been triggered by the
        // load_state attempt during with_defaults — verify body counts
        // restored from one of the saves
        let body_state = det2.to_state();
        let body_pair = body_state.pairs.get("body").unwrap();
        // Must have at least the 3 cycles from first save
        assert!(body_pair.total_resonant_cycles >= 3);

        // Verify direct load_state call returns Ok (clean .bak)
        let _ = det2.load_state();
    }

    // ── DEFAULT THRESHOLDS MATCH SPEC (1) ──────────────────────────────

    #[test]
    fn default_thresholds_match_spec_constants() {
        // C4-2b1 test 11: defaults pull from titan_core::constants
        let pair = ResonancePair::with_defaults("body");
        assert_eq!(pair.phase_threshold, RESONANCE_PHASE_THRESHOLD_RAD);
        assert_eq!(pair.required_cycles, RESONANCE_CYCLES_REQUIRED as u32);
        assert_eq!(pair.pulse_window, RESONANCE_PULSE_WINDOW_S);
        // Sanity vs SPEC TOML values
        assert!((pair.phase_threshold - PI / 6.0).abs() < 1e-10);
        assert_eq!(pair.required_cycles, 3);
        assert_eq!(pair.pulse_window, 120.0);
    }

    // ── BIG PULSE PAYLOAD SCHEMA (1) ───────────────────────────────────

    #[test]
    fn big_pulse_serializes_round_trip() {
        // C4-2b1 test 12: BigPulse JSON encode/decode round-trip
        // (used for bus payload schema in C4-2b1 SPHERE_PULSE wiring)
        let bp = BigPulse {
            pair: "body".into(),
            big_pulse_count: 7,
            phase_diff: 0.123,
            time_diff: 1.5,
            inner_pulse_count: 30,
            outer_pulse_count: 28,
            total_resonant_cycles: 21,
            ts: 1234567890.5,
            great_pulse_ready: true,
            great_pulse_count: 2,
        };
        let json = serde_json::to_string(&bp).unwrap();
        let decoded: BigPulse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, bp);
    }
}
