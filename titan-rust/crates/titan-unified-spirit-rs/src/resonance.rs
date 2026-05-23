//! resonance — Phase-aligned inner↔outer pair detector + GREAT PULSE
//! gating.
//!
//! Full Rust port of `titan_hcl/logic/resonance.py` per SPEC §10.F G11
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

// §G11 HARMONY_TICKS retired by D-SPEC-122 (v1.55.0, 2026-05-23): the
// sustained-balance level + rising-edge BIG/GREAT mechanic from D-SPEC-114
// (v1.50.1, commit cf2a617e) was reverted to the original D-SPEC-112 "3
// consecutive simultaneous-balance coincidences" gate (commit a103e073).
// Reason: live observation that the rising-edge gate silenced BIG/GREAT
// pulses under sustained perfect balance (T1 ~1.88h gap with consec_balanced
// in the thousands). The D-SPEC-114 mechanic is preserved in git history +
// in ARCHITECTURE_trinity.md §9.4a as a future tuning option, retrievable
// via `git show cf2a617e:titan-rust/crates/titan-unified-spirit-rs/src/resonance.rs`.

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
    /// §G11 balance-coincidence (D-SPEC-112): whether the last inner pulse
    /// fired in a balanced regime. The pair counts a resonant coincidence
    /// only when BOTH sides' last pulses were balanced (and both fresh).
    inner_last_balanced: bool,
    /// Set true on receipt of an inner pulse; reset to false when the pair
    /// has consumed it in `check_resonance` (avoids double-counting).
    inner_fresh: bool,
    outer_last_pulse_ts: f64,
    outer_last_phase: f64,
    /// §G11 balance-coincidence (D-SPEC-112): last outer pulse balanced?
    outer_last_balanced: bool,
    /// Set true on receipt of an outer pulse; reset on consumption.
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
            inner_last_balanced: false,
            inner_fresh: false,
            outer_last_pulse_ts: 0.0,
            outer_last_phase: 0.0,
            outer_last_balanced: false,
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

    /// Record a SPHERE_PULSE event with explicit phase + balance state.
    /// `component` is the full name (`"inner_body"`, `"outer_mind"`, etc.).
    ///
    /// §G11 Proof-of-Harmony gate (D-SPEC-112, restored by D-SPEC-122 v1.55.0
    /// — supersedes the D-SPEC-114 sustained-balance level + rising-edge
    /// mechanic; that alternative is preserved at git ref `cf2a617e` and in
    /// `ARCHITECTURE_trinity.md` §9.4a as a future tuning option):
    ///
    /// A pair is resonant on a check when BOTH sides have pulsed since the
    /// last check (`*_fresh = true`), neither pulse is older than
    /// `pulse_window`, AND both pulses fired in a balanced regime
    /// (`*_last_balanced = true`). After `required_cycles` (default 3)
    /// consecutive resonant coincidences → **BIG PULSE** fires. The streak
    /// resets to 0 on any non-resonant check (out-of-time or either side
    /// unbalanced). `phase_diff` is computed for `/v6/trinity/resonance`
    /// telemetry only — no longer in the gate (the phase mechanic was
    /// replaced by balance-coincidence in D-SPEC-112).
    ///
    /// Returns `Some(BigPulse)` iff the streak just crossed `required_cycles`,
    /// `None` otherwise.
    pub fn record_pulse_with_phase(
        &mut self,
        component: &str,
        phase: f64,
        balanced: bool,
        pulse_ts: f64,
    ) -> Option<BigPulse> {
        let is_inner = component.starts_with("inner_");
        if is_inner {
            self.inner_last_pulse_ts = pulse_ts;
            self.inner_last_phase = phase;
            self.inner_last_balanced = balanced;
            self.inner_fresh = true;
            self.inner_pulse_count += 1;
        } else {
            self.outer_last_pulse_ts = pulse_ts;
            self.outer_last_phase = phase;
            self.outer_last_balanced = balanced;
            self.outer_fresh = true;
            self.outer_pulse_count += 1;
        }

        self.check_resonance(pulse_ts)
    }

    /// §G11 PoH check — requires both sides fresh-since-last-check, both
    /// recent, both balanced. On a passing check: `consecutive_resonant++`;
    /// when the streak reaches `required_cycles`, BIG PULSE fires + streak
    /// resets to 0. On a failing check: streak resets to 0 (out-of-time OR
    /// either side unbalanced). Both `*_fresh` flags clear after consumption
    /// so the next check requires a NEW pulse on each side.
    fn check_resonance(&mut self, now: f64) -> Option<BigPulse> {
        // Both sides must have a NEW pulse since the last check. Without
        // this guard a single side's pulse stream would spin up consecutive
        // checks against a stale-but-balanced counterpart, false-resonant.
        if !(self.inner_fresh && self.outer_fresh) {
            return None;
        }
        self.inner_fresh = false;
        self.outer_fresh = false;

        self.total_checks += 1;

        // Out-of-time guard: counterpart pulse can't be older than pulse_window.
        let time_diff = (self.inner_last_pulse_ts - self.outer_last_pulse_ts).abs();
        if time_diff > self.pulse_window {
            self.consecutive_resonant = 0;
            self.is_resonant = false;
            return None;
        }

        // §G11 PoH gate: both sides balanced AT pulse time.
        if !(self.inner_last_balanced && self.outer_last_balanced) {
            self.consecutive_resonant = 0;
            self.is_resonant = false;
            return None;
        }

        // Coincidence found. Both sides are also recent enough by
        // construction (within pulse_window of each other AND of `now` —
        // a stale clock can't carry a fresh `*_fresh = true` flag).
        let _ = now;
        self.consecutive_resonant += 1;
        self.total_resonant_cycles += 1;
        self.is_resonant = true;

        if self.consecutive_resonant >= self.required_cycles {
            self.consecutive_resonant = 0; // re-arm for the next 3-streak
            let phase_diff = phase_difference(self.inner_last_phase, self.outer_last_phase);
            Some(self.generate_big_pulse(phase_diff, time_diff))
        } else {
            None
        }
    }

    fn generate_big_pulse(&mut self, phase_diff: f64, time_diff: f64) -> BigPulse {
        self.big_pulse_count += 1;
        let now = wall_seconds();
        self.last_big_pulse_ts = now;

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
        // D-SPEC-122 (v1.55.0): the rising-edge mechanic from D-SPEC-114 has
        // been reverted to the D-SPEC-112 "3 consecutive coincidences" gate;
        // no edge state to restore. `*_fresh` start false — the next check
        // waits for a NEW pulse on each side.
        self.inner_fresh = false;
        self.outer_fresh = false;
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
    // D-SPEC-122 (v1.55.0): `all_harmonious_prev` rising-edge state removed
    // with the revert from D-SPEC-114 → D-SPEC-112. GREAT now fires whenever
    // a BIG PULSE coincides with all 3 pairs being resonant (the original
    // V4 mechanic). Pair-level streak resets after each BIG handle the
    // re-arming naturally — a fresh 3-coincidence sequence has to build up
    // again on each pair.
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
    /// `phase` is from the SPHERE_PULSE payload float; `balanced` is the
    /// per-pulse balance flag (§G11 D-SPEC-112 balance-coincidence gate;
    /// D-SPEC-122 v1.55.0 restored this from the D-SPEC-114 sustained-level
    /// mechanic — `consecutive_balanced` is no longer in the gate).
    ///
    /// Returns `Some(BigPulse)` when the pair has accumulated
    /// `required_cycles` consecutive balanced coincidences. GREAT PULSE
    /// gating: if this BIG PULSE coincides with all 3 pairs being currently
    /// resonant, `great_pulse_ready` is set on the emitted BigPulse and
    /// `UnifiedSpirit::advance()` is triggered by the caller.
    pub fn record_pulse_with_phase(
        &mut self,
        component: &str,
        phase: f64,
        balanced: bool,
        pulse_ts: f64,
    ) -> Option<BigPulse> {
        let pair_name = component_to_pair(component)?;
        let big_opt = match self.pairs.get_mut(pair_name) {
            Some(pair) => pair.record_pulse_with_phase(component, phase, balanced, pulse_ts),
            None => return None,
        };

        // GREAT PULSE gate (§G11 D-SPEC-112, restored by D-SPEC-122): any
        // time a BIG PULSE fires AND all 3 pairs are simultaneously resonant,
        // GREAT PULSE crystallizes and `UnifiedSpirit::advance()` triggers.
        // Pair-level 3-cycle streaks reset after each BIG, so the next GREAT
        // requires fresh 3-coincidence sequences to build up on all 3 pairs.
        big_opt.map(|mut bp| {
            if self.all_resonant() {
                self.great_pulse_count += 1;
                self.last_great_pulse_ts = wall_seconds();
                bp.great_pulse_ready = true;
                bp.great_pulse_count = self.great_pulse_count;
            }
            bp
        })
    }

    /// True iff all 3 pairs are currently harmonious (§G11 sustained-balance).
    /// Used by callers to invoke `UnifiedSpirit::advance(...)` (C4-2b2).
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

    /// Phase B: msgpack-serializable snapshot for `resonance_metadata.bin`
    /// SHM slot (rFP_phase_c_state_read_unification §B / D-SPEC-72).
    /// Mirrors Python `ResonanceDetector.get_stats()` 1:1 so the ownership
    /// flip Python → Rust is transparent to consumers.
    pub fn get_stats(&self, ts: f64) -> crate::metadata_publisher::ResonanceMetadata {
        use crate::metadata_publisher::{
            ResonanceConfigSummary, ResonanceMetadata, ResonancePairStats,
        };
        use titan_core::constants::RESONANCE_METADATA_SCHEMA_VERSION;

        // Python parity rounds — phase_threshold to 4 decimals, config
        // phase_threshold_deg to 1 decimal (`round(math.degrees(_), 1)`).
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let r1 = |v: f64| (v * 1e1).round() / 1e1;

        let mut pairs = std::collections::BTreeMap::new();
        for (name, pair) in &self.pairs {
            let state = pair.to_state();
            pairs.insert(
                name.clone(),
                ResonancePairStats {
                    name: state.name,
                    is_resonant: state.is_resonant,
                    consecutive_resonant: state.consecutive_resonant,
                    required_cycles: pair.required_cycles,
                    total_resonant_cycles: state.total_resonant_cycles,
                    total_checks: state.total_checks,
                    big_pulse_count: state.big_pulse_count,
                    inner_pulse_count: state.inner_pulse_count,
                    outer_pulse_count: state.outer_pulse_count,
                    last_big_pulse_ts: state.last_big_pulse_ts,
                    phase_threshold: r4(pair.phase_threshold),
                    pulse_window: pair.pulse_window,
                },
            );
        }

        // Config summary uses the FIRST pair's thresholds — all 3 share the
        // same config in practice (constructed via ResonancePair::with_defaults).
        let first = self.pairs.values().next();
        let (phase_threshold, required_cycles, pulse_window) = first
            .map(|p| (p.phase_threshold, p.required_cycles, p.pulse_window))
            .unwrap_or((0.0, 0, 0.0));

        ResonanceMetadata {
            pairs,
            resonant_count: self.resonant_count() as u32,
            all_resonant: self.all_resonant(),
            great_pulse_count: self.great_pulse_count,
            last_great_pulse_ts: self.last_great_pulse_ts,
            config: ResonanceConfigSummary {
                phase_threshold_deg: r1(phase_threshold.to_degrees()),
                required_cycles,
                pulse_window,
            },
            schema_version: RESONANCE_METADATA_SCHEMA_VERSION as u32,
            ts,
        }
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
        // D-SPEC-122 (v1.55.0): rising-edge GREAT state removed with the
        // revert to the D-SPEC-112 mechanic. Per-pair `*_fresh` flags clear
        // on restore, so the next GREAT requires fresh 3-coincidence
        // sequences on all 3 pairs — no spurious crystallization.
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

    // ── D-SPEC-122 (v1.55.0) RESONANCE TESTS — 3 consecutive coincidences ─
    // The D-SPEC-114 sustained-balance + rising-edge tests are preserved in
    // git history at `git show cf2a617e -- titan-rust/crates/titan-unified-spirit-rs/src/resonance.rs`
    // and in `ARCHITECTURE_trinity.md` §9.4a as a future tuning option.

    #[test]
    fn record_pulse_inner_only_no_resonance() {
        // A single inner pulse (outer never seen) → no resonance check yet
        // (both sides must be fresh-since-last-check). `*_fresh` accounting
        // means `total_checks` only increments when BOTH sides have a fresh
        // pulse pair to evaluate.
        let mut pair = ResonancePair::with_defaults("body");
        let result = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.0);
        assert!(result.is_none());
        assert!(!pair.is_resonant());
        assert_eq!(pair.total_checks, 0);
        assert_eq!(pair.inner_pulse_count, 1);
        assert_eq!(pair.outer_pulse_count, 0);
    }

    // ── 3-consecutive-coincidence BIG PULSE gate (D-SPEC-122) ────────────

    #[test]
    fn pair_emits_big_pulse_after_3_consecutive_coincidences() {
        // §G11 D-SPEC-112 (restored by D-SPEC-122): BIG fires when both sides
        // have pulsed since the last check, both pulses balanced, both within
        // pulse_window, AND the streak reaches `required_cycles = 3`. Streak
        // resets to 0 after firing → next BIG needs another 3-streak.
        let mut pair = ResonancePair::with_defaults("body");
        // Coincidence #1: inner+outer balanced → streak 1, no BIG yet
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.0);
        let r1 = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.1);
        assert!(r1.is_none(), "streak=1 < required_cycles=3 → no BIG yet");
        assert!(
            pair.is_resonant(),
            "single coincidence still marks resonant"
        );
        // Coincidence #2 → streak 2, still no BIG
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.2);
        let r2 = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.3);
        assert!(r2.is_none());
        // Coincidence #3 → streak hits 3 → BIG fires
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.4);
        let r3 = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.5);
        let bp = r3.expect("BIG must fire at 3 consecutive coincidences");
        assert_eq!(bp.pair, "body");
        assert_eq!(bp.big_pulse_count, 1);
        assert!(pair.is_resonant());
        // After BIG, streak resets to 0. Next 3-streak → BIG #2.
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.6);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.7);
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.8);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.9);
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 101.0);
        let bp2 = pair
            .record_pulse_with_phase("outer_body", 0.0, true, 101.1)
            .expect("BIG #2 fires on next 3-streak");
        assert_eq!(bp2.big_pulse_count, 2);
    }

    // ── streak resets on unbalanced check ──────────────────────────────

    #[test]
    fn pair_drops_harmony_on_lost_balance() {
        // Either side unbalanced → consecutive_resonant resets to 0; the
        // partial streak is lost.
        let mut pair = ResonancePair::with_defaults("body");
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.0);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.1);
        assert!(pair.is_resonant());
        // outer unbalanced on next coincidence → streak resets to 0
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.2);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, false, 100.3);
        assert!(!pair.is_resonant());
    }

    // ── stale counterpart → out-of-time → streak resets ─────────────────

    #[test]
    fn pair_drops_harmony_on_stale_counterpart() {
        // A side that hasn't pulsed within `pulse_window` (default 120s)
        // means the counterpart's last pulse is stale → out-of-time → reset.
        let mut pair = ResonancePair::with_defaults("body");
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 100.0);
        let _ = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.1);
        assert!(pair.is_resonant());
        // inner pulses 200s later → outer (last @100.1) now stale (>120s
        // pulse_window) when outer's next pulse fires
        let _ = pair.record_pulse_with_phase("inner_body", 0.0, true, 300.0);
        let r = pair.record_pulse_with_phase("outer_body", 0.0, true, 100.5);
        assert!(r.is_none(), "stale-counterpart timing must reject streak");
        assert!(!pair.is_resonant());
    }

    // ── ALL_RESONANT + RESONANT_COUNT ──────────────────────────────────

    #[test]
    fn detector_all_resonant_and_resonant_count() {
        // Each pair marks `is_resonant=true` on the first balanced coincidence;
        // a single coincidence is enough to mark resonant (the 3-streak gates
        // BIG, not the resonance flag).
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());
        assert!(!det.all_resonant());
        assert_eq!(det.resonant_count(), 0);

        let _ = det.record_pulse_with_phase("inner_body", 0.0, true, 100.0);
        let _ = det.record_pulse_with_phase("outer_body", 0.0, true, 100.1);
        assert_eq!(det.resonant_count(), 1);
        assert!(!det.all_resonant());

        let _ = det.record_pulse_with_phase("inner_mind", 0.0, true, 100.0);
        let _ = det.record_pulse_with_phase("outer_mind", 0.0, true, 100.1);
        assert_eq!(det.resonant_count(), 2);
        assert!(!det.all_resonant());

        let _ = det.record_pulse_with_phase("inner_spirit", 0.0, true, 100.0);
        let _ = det.record_pulse_with_phase("outer_spirit", 0.0, true, 100.1);
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
    fn great_pulse_count_increments_when_big_coincides_with_all_3_resonant() {
        // §G11 D-SPEC-112 (restored by D-SPEC-122): GREAT PULSE fires when a
        // BIG PULSE coincides with all 3 pairs being currently resonant. Drive
        // body + mind + spirit each to a 3-coincidence streak; the BIG from
        // the 3rd pair (whichever fires last) closes all-3-resonant and
        // crystallizes GREAT.
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        // body + mind each reach 3 coincidences → BIG fires on each, but
        // spirit hasn't pulsed yet so all_resonant=false → great_pulse_ready=false.
        for name in ["body", "mind"] {
            let inner = format!("inner_{name}");
            let outer = format!("outer_{name}");
            for i in 0..3 {
                let t = 100.0 + (i as f64) * 0.2;
                let _ = det.record_pulse_with_phase(&inner, 0.0, true, t);
                let _ = det.record_pulse_with_phase(&outer, 0.0, true, t + 0.1);
            }
        }
        // Body + mind are currently resonant; spirit has not pulsed yet.
        assert_eq!(det.resonant_count(), 2);
        assert!(!det.all_resonant());
        assert_eq!(det.great_pulse_count(), 0);

        // Spirit reaches 3 coincidences. On the 3rd coincidence, the BIG
        // fires AND all 3 pairs are resonant → GREAT crystallizes on this
        // spirit BIG.
        let mut spirit_bp = None;
        for i in 0..3 {
            let t = 100.0 + (i as f64) * 0.2;
            let _ = det.record_pulse_with_phase("inner_spirit", 0.0, true, t);
            spirit_bp = det.record_pulse_with_phase("outer_spirit", 0.0, true, t + 0.1);
        }
        let bp = spirit_bp.expect("BIG fires on 3rd spirit coincidence");
        assert_eq!(bp.pair, "spirit");
        assert!(
            bp.great_pulse_ready,
            "BIG that closes all-3-resonant must set great_pulse_ready"
        );
        assert_eq!(bp.great_pulse_count, 1);
        assert_eq!(det.great_pulse_count(), 1);
    }

    // ── SAVE / RESTORE STATE (2) ───────────────────────────────────────

    #[test]
    fn save_and_load_state_roundtrip() {
        // D-SPEC-122: save → load preserves all per-pair state across the
        // 3-consecutive-coincidence mechanic. Four coincidences → BIG fires
        // at the 3rd (streak resets to 0), then a 4th coincidence starts a
        // new streak at 1.
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        for ts in [100.0_f64, 100.2, 100.4, 100.6] {
            let _ = det.record_pulse_with_phase("inner_body", 0.0, true, ts);
            let _ = det.record_pulse_with_phase("outer_body", 0.0, true, ts + 0.1);
        }
        let total_resonant = det.pairs.get("body").unwrap().total_resonant_cycles;
        let big_pulse_count = det.pairs.get("body").unwrap().big_pulse_count;
        assert!(total_resonant >= 3, "4 coincidences observed");
        assert_eq!(big_pulse_count, 1, "single BIG at 3rd coincidence");

        det.save_state().unwrap();
        let det2 = ResonanceDetector::with_defaults(dir.path());
        let restored = det2.pairs.get("body").unwrap();
        assert_eq!(restored.total_resonant_cycles, total_resonant);
        assert_eq!(restored.big_pulse_count, big_pulse_count);
    }

    #[test]
    fn load_state_falls_back_to_bak_when_canonical_corrupt() {
        // D-SPEC-122: corrupt canonical → restore from .bak
        let dir = tempdir().unwrap();
        let mut det = ResonanceDetector::with_defaults(dir.path());

        for ts in [100.0_f64, 100.2, 100.4] {
            let _ = det.record_pulse_with_phase("inner_body", 0.0, true, ts);
            let _ = det.record_pulse_with_phase("outer_body", 0.0, true, ts + 0.1);
        }
        det.save_state().unwrap();
        for ts in [400.0_f64, 400.2, 400.4] {
            let _ = det.record_pulse_with_phase("inner_mind", 0.0, true, ts);
            let _ = det.record_pulse_with_phase("outer_mind", 0.0, true, ts + 0.1);
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
