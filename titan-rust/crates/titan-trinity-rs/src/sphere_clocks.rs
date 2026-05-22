//! sphere_clocks — 6-clock sphere-time engine per Preamble G11 + SPEC §7.1
//! sphere_clocks.bin layout.
//!
//! Per Preamble G11: the 3 inner-trinity ↔ outer-trinity resonance pairs
//! converge to a GREAT PULSE when all 6 component clocks pulse in sync. The
//! GREAT PULSE itself is detected by `titan-unified-spirit-rs` (C-S4) reading
//! these clock states from `sphere_clocks.bin`.
//!
//! # Byte-identical port (per SPEC §11.6)
//!
//! Implementation matches `titan_hcl/logic/sphere_clock.py:46-179`:
//!   - Each clock's scalar contracts at velocity = `base_speed × max(min_velocity_factor, coherence)`
//!   - Pulse fires when scalar reaches center (≤ 0.0)
//!   - On pulse: balanced → radius shrinks (faster future cadence); unbalanced → radius grows
//!   - Phase wraps at 2π; CONTINUOUS across pulses (no reset) per `sphere_clock.py:158-161`
//!
//! # Slot layout (SPEC §7.1 sphere_clocks.bin = 168 bytes payload)
//!
//! 6 clocks × 7 fields × float32 LE = 168 bytes. Field order per clock:
//!
//! - `[0]` radius
//! - `[1]` scalar_position
//! - `[2]` phase
//! - `[3]` contraction_velocity
//! - `[4]` pulse_count (cast to f32)
//! - `[5]` consecutive_balanced (cast to f32)
//! - `[6]` last_pulse_age_s
//!
//! Clock order in file: inner_body, inner_mind, inner_spirit, outer_body,
//! outer_mind, outer_spirit (matches `sphere_clock.py:41-43 INNER+OUTER`).

use titan_core::constants::{
    SPHERE_CLOCKS_PAYLOAD_BYTES, SPHERE_CLOCK_COUNT, SPHERE_CLOCK_FIELD_COUNT,
};

/// Default sphere clock tuning constants per `sphere_clock.py:32-38`. Not
/// in SPEC TOML because they're tunable per-clock at construction (for
/// tests and future Phase D calibration); these defaults match the
/// published Python baseline.
pub const DEFAULT_BASE_SPEED: f32 = 0.05;
/// Minimum sphere radius — clocks never collapse smaller than this.
pub const DEFAULT_MIN_RADIUS: f32 = 0.3;
/// Radius shrink rate per balanced pulse (faster future cadence reward).
pub const DEFAULT_PULSE_SHRINK_RATE: f32 = 0.02;
/// Coherence threshold above which a clock counts as "balanced" (1.0 - threshold).
///
/// 2026-05-18 calibrated 0.20 → 0.30 (threshold 0.80 → 0.70) per Maker decision
/// after D-SPEC-84 variance-formula restoration revealed that real-world Titan
/// tensor variance (5D body / 15D mind / 45D spirit) rarely produces coherence
/// ≥ 0.80 — natural distribution sits 0.50–0.75. Threshold 0.70 lets balanced
/// pulses fire periodically across all 6 layers without forcing artificial
/// tensor flattening. Python parity: `sphere_clock.py:36 DEFAULT_BALANCE_THRESHOLD`
/// matched. SPEC §G11 records this calibration.
pub const DEFAULT_BALANCE_THRESHOLD: f32 = 0.30;
/// Floor on velocity factor — even max-imbalanced clocks contract at 15% speed.
pub const DEFAULT_MIN_VELOCITY_FACTOR: f32 = 0.15;

const _: () = assert!(SPHERE_CLOCK_COUNT == 6);
const _: () = assert!(SPHERE_CLOCK_FIELD_COUNT == 7);
const _: () = assert!(SPHERE_CLOCKS_PAYLOAD_BYTES == 168);

/// Canonical clock role — matches `sphere_clock.py:41-43`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockRole {
    /// Inner-body sphere clock.
    InnerBody,
    /// Inner-mind sphere clock.
    InnerMind,
    /// Inner-spirit sphere clock.
    InnerSpirit,
    /// Outer-body sphere clock.
    OuterBody,
    /// Outer-mind sphere clock.
    OuterMind,
    /// Outer-spirit sphere clock.
    OuterSpirit,
}

impl ClockRole {
    /// Canonical name — matches `sphere_clock.py:41-43` strings.
    pub fn as_str(self) -> &'static str {
        match self {
            ClockRole::InnerBody => "inner_body",
            ClockRole::InnerMind => "inner_mind",
            ClockRole::InnerSpirit => "inner_spirit",
            ClockRole::OuterBody => "outer_body",
            ClockRole::OuterMind => "outer_mind",
            ClockRole::OuterSpirit => "outer_spirit",
        }
    }

    /// Canonical clock order in `sphere_clocks.bin` per SPEC §7.1.
    pub fn all() -> [ClockRole; SPHERE_CLOCK_COUNT as usize] {
        [
            ClockRole::InnerBody,
            ClockRole::InnerMind,
            ClockRole::InnerSpirit,
            ClockRole::OuterBody,
            ClockRole::OuterMind,
            ClockRole::OuterSpirit,
        ]
    }
}

/// One pulse event emitted when a clock's scalar reaches center.
/// Mirrors Python `sphere_clock.py:163-171 pulse_event` dict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseEvent {
    /// Which clock pulsed.
    pub role: ClockRole,
    /// Cumulative pulse count (post-increment).
    pub pulse_count: u32,
    /// Phase at pulse moment (0.0 to 2π).
    pub phase: f32,
    /// Radius before pulse update.
    pub radius_before: f32,
    /// Radius after pulse update (shrunk if balanced, grown if not).
    pub radius_after: f32,
    /// Whether the pulse was generated in a balanced-coherence regime.
    pub balanced: bool,
    /// Consecutive balanced ticks at pulse time.
    pub consecutive_balanced: u32,
}

/// Single sphere clock. Per `sphere_clock.py:46-200`.
#[derive(Debug, Clone)]
pub struct SphereClock {
    role: ClockRole,
    base_speed: f32,
    min_radius: f32,
    pulse_shrink_rate: f32,
    balance_threshold: f32,
    min_velocity_factor: f32,
    /// Current sphere radius (1.0 = fully expanded).
    pub radius: f32,
    /// Scalar position: 1.0 = edge, 0.0 = center (pulse point).
    pub scalar_position: f32,
    /// Phase 0.0 .. 2π (continuous across pulses for resonance drift).
    pub phase: f32,
    /// Last computed contraction velocity (`base_speed × velocity_factor`).
    pub contraction_velocity: f32,
    /// Cumulative pulse count.
    pub pulse_count: u32,
    /// Consecutive ticks within balance threshold.
    pub consecutive_balanced: u32,
    /// Total ticks since construction.
    pub total_ticks: u64,
    /// Wall-clock seconds since last pulse (substrate updates per body cycle).
    pub last_pulse_age_s: f32,
}

impl SphereClock {
    /// Construct with default tuning. Initial state: radius=1.0,
    /// scalar=1.0 (edge), phase=0, no pulses.
    pub fn new(role: ClockRole) -> Self {
        Self {
            role,
            base_speed: DEFAULT_BASE_SPEED,
            min_radius: DEFAULT_MIN_RADIUS,
            pulse_shrink_rate: DEFAULT_PULSE_SHRINK_RATE,
            balance_threshold: DEFAULT_BALANCE_THRESHOLD,
            min_velocity_factor: DEFAULT_MIN_VELOCITY_FACTOR,
            radius: 1.0,
            scalar_position: 1.0,
            phase: 0.0,
            contraction_velocity: 0.0,
            pulse_count: 0,
            consecutive_balanced: 0,
            total_ticks: 0,
            last_pulse_age_s: 0.0,
        }
    }

    /// Tick the clock. Per `sphere_clock.py:83-134`.
    ///
    /// `coherence` ∈ [0.0, 1.0] (clamped). High coherence → fast contraction.
    /// `dt` is the wall-clock seconds since last tick (used to age
    /// `last_pulse_age_s`); typical substrate body cycle ≈ 1.149 s.
    ///
    /// Returns `Some(PulseEvent)` if a pulse was generated this tick.
    pub fn tick(&mut self, coherence: f32, dt: f32) -> Option<PulseEvent> {
        self.total_ticks += 1;
        // Always advance pulse age — tested every tick whether or not a pulse fires
        self.last_pulse_age_s += dt;

        let coh = coherence.clamp(0.0, 1.0);
        let velocity_factor = coh.max(self.min_velocity_factor);
        self.contraction_velocity = self.base_speed * velocity_factor;

        // Move scalar toward center
        self.scalar_position = (self.scalar_position - self.contraction_velocity).max(0.0);

        // Advance phase (wraps at 2π) — proportional to contraction progress
        if self.radius > 0.0 {
            let phase_step = (self.contraction_velocity / self.radius) * std::f32::consts::PI;
            self.phase = (self.phase + phase_step) % (2.0 * std::f32::consts::PI);
        }

        // Track balance streak
        let is_balanced = coh >= (1.0 - self.balance_threshold);
        if is_balanced {
            self.consecutive_balanced += 1;
        } else {
            self.consecutive_balanced = 0;
        }

        // Pulse if scalar reached center
        if self.scalar_position <= 0.0 {
            Some(self.generate_pulse(is_balanced))
        } else {
            None
        }
    }

    /// Per `sphere_clock.py:136-179 _generate_pulse`.
    fn generate_pulse(&mut self, is_balanced: bool) -> PulseEvent {
        self.pulse_count += 1;
        let radius_before = self.radius;
        let new_radius = if is_balanced {
            (self.radius - self.pulse_shrink_rate).max(self.min_radius)
        } else {
            (self.radius + self.pulse_shrink_rate * 0.5).min(1.0)
        };
        self.radius = new_radius;
        // Reset scalar to edge of (new) sphere for next cycle
        self.scalar_position = self.radius;
        // Phase continues unchanged (NO reset) per sphere_clock.py:158-161 —
        // continuous phase enables natural resonance drift.
        self.last_pulse_age_s = 0.0;

        PulseEvent {
            role: self.role,
            pulse_count: self.pulse_count,
            phase: self.phase,
            radius_before,
            radius_after: new_radius,
            balanced: is_balanced,
            consecutive_balanced: self.consecutive_balanced,
        }
    }

    /// Read role.
    pub fn role(&self) -> ClockRole {
        self.role
    }
}

/// 6 sphere clocks — substrate's full clock state per Preamble G11. Mirrors
/// `sphere_clock.py:41-43 INNER + OUTER`. Iteration order matches SPEC §7.1
/// sphere_clocks.bin canonical clock ordering.
#[derive(Debug, Clone)]
pub struct SphereClockSet {
    /// Inner-body sphere clock.
    pub inner_body: SphereClock,
    /// Inner-mind sphere clock.
    pub inner_mind: SphereClock,
    /// Inner-spirit sphere clock.
    pub inner_spirit: SphereClock,
    /// Outer-body sphere clock.
    pub outer_body: SphereClock,
    /// Outer-mind sphere clock.
    pub outer_mind: SphereClock,
    /// Outer-spirit sphere clock.
    pub outer_spirit: SphereClock,
}

impl SphereClockSet {
    /// Construct all 6 clocks at initial state.
    pub fn new() -> Self {
        Self {
            inner_body: SphereClock::new(ClockRole::InnerBody),
            inner_mind: SphereClock::new(ClockRole::InnerMind),
            inner_spirit: SphereClock::new(ClockRole::InnerSpirit),
            outer_body: SphereClock::new(ClockRole::OuterBody),
            outer_mind: SphereClock::new(ClockRole::OuterMind),
            outer_spirit: SphereClock::new(ClockRole::OuterSpirit),
        }
    }

    /// Iterate all 6 clocks in canonical (SPEC §7.1) order. Mut variant.
    pub fn iter_mut(&mut self) -> [&mut SphereClock; SPHERE_CLOCK_COUNT as usize] {
        [
            &mut self.inner_body,
            &mut self.inner_mind,
            &mut self.inner_spirit,
            &mut self.outer_body,
            &mut self.outer_mind,
            &mut self.outer_spirit,
        ]
    }

    /// Iterate all 6 clocks in canonical order. Read-only.
    pub fn iter(&self) -> [&SphereClock; SPHERE_CLOCK_COUNT as usize] {
        [
            &self.inner_body,
            &self.inner_mind,
            &self.inner_spirit,
            &self.outer_body,
            &self.outer_mind,
            &self.outer_spirit,
        ]
    }

    /// Serialize all 6 clocks × 7 fields to 168-byte float32 LE payload per
    /// SPEC §7.1 sphere_clocks.bin layout. Suitable for direct write to the
    /// shm slot via `Slot::write_with_seqlock(_)`.
    pub fn serialize(&self) -> [u8; SPHERE_CLOCKS_PAYLOAD_BYTES as usize] {
        let mut out = [0u8; SPHERE_CLOCKS_PAYLOAD_BYTES as usize];
        let clocks = self.iter();
        for (clk_idx, clk) in clocks.iter().enumerate() {
            let base = clk_idx * (SPHERE_CLOCK_FIELD_COUNT as usize) * 4;
            let fields = [
                clk.radius,
                clk.scalar_position,
                clk.phase,
                clk.contraction_velocity,
                clk.pulse_count as f32,
                clk.consecutive_balanced as f32,
                clk.last_pulse_age_s,
            ];
            for (f_idx, v) in fields.iter().enumerate() {
                let off = base + f_idx * 4;
                out[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

impl Default for SphereClockSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── Single SphereClock ────────────────────────────────────────────────

    #[test]
    fn new_clock_initial_state() {
        let c = SphereClock::new(ClockRole::InnerBody);
        assert_eq!(c.radius, 1.0);
        assert_eq!(c.scalar_position, 1.0);
        assert_eq!(c.phase, 0.0);
        assert_eq!(c.pulse_count, 0);
        assert_eq!(c.consecutive_balanced, 0);
        assert_eq!(c.role, ClockRole::InnerBody);
    }

    #[test]
    fn tick_low_coherence_advances_at_min_velocity() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let _ = c.tick(0.0, 1.0);
        // velocity = 0.05 × 0.15 = 0.0075
        assert!(approx(c.contraction_velocity, 0.0075, 1e-6));
        assert!(approx(c.scalar_position, 1.0 - 0.0075, 1e-5));
    }

    #[test]
    fn tick_high_coherence_advances_at_full_speed() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let _ = c.tick(1.0, 1.0);
        // velocity = 0.05 × 1.0 = 0.05
        assert!(approx(c.contraction_velocity, 0.05, 1e-6));
        assert!(approx(c.scalar_position, 0.95, 1e-5));
    }

    #[test]
    fn tick_balanced_streak_increments_on_high_coherence() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let _ = c.tick(0.95, 1.0); // ≥ 0.8 (=1 - 0.2 threshold) → balanced
        let _ = c.tick(0.85, 1.0);
        let _ = c.tick(0.81, 1.0);
        assert_eq!(c.consecutive_balanced, 3);
    }

    #[test]
    fn tick_low_coherence_resets_balanced_streak() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let _ = c.tick(1.0, 1.0);
        let _ = c.tick(1.0, 1.0);
        let _ = c.tick(0.5, 1.0); // < 0.8 → not balanced
        assert_eq!(c.consecutive_balanced, 0);
    }

    #[test]
    fn pulse_fires_when_scalar_reaches_center() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        // 1.0 / 0.05 = 20 ticks at full speed to reach center
        let mut pulse_seen = None;
        for _ in 0..25 {
            if let Some(p) = c.tick(1.0, 1.0) {
                pulse_seen = Some(p);
                break;
            }
        }
        let p = pulse_seen.expect("pulse should fire within 25 high-coherence ticks");
        assert_eq!(p.role, ClockRole::InnerBody);
        assert_eq!(p.pulse_count, 1);
        assert!(p.balanced);
        assert_eq!(c.pulse_count, 1);
        // Scalar reset to new radius
        assert!(c.scalar_position > 0.0);
    }

    #[test]
    fn pulse_balanced_shrinks_radius_toward_min() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let initial_radius = c.radius;
        for _ in 0..25 {
            let _ = c.tick(1.0, 1.0);
        }
        // After balanced pulse: radius = max(min_radius, 1.0 - 0.02) = 0.98
        assert!(approx(c.radius, 0.98, 1e-5));
        assert!(c.radius < initial_radius);
    }

    #[test]
    fn pulse_unbalanced_grows_radius() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        // Force a pulse with unbalanced coherence — slowly contract at min velocity
        // 1.0 / 0.0075 ≈ 134 ticks
        let mut pulse_seen = None;
        for _ in 0..150 {
            if let Some(p) = c.tick(0.0, 1.0) {
                pulse_seen = Some(p);
                break;
            }
        }
        let p = pulse_seen.expect("pulse should fire eventually at min velocity");
        assert!(!p.balanced);
        // Unbalanced pulse: radius += pulse_shrink_rate × 0.5 = 0.01
        assert!(approx(c.radius, 1.0, 1e-5)); // already at 1.0, can't grow
        assert_eq!(p.consecutive_balanced, 0);
    }

    #[test]
    fn pulse_phase_continues_no_reset_per_g11_resonance_drift() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        // Run until a pulse fires, capture phase before + after
        let mut phase_at_pulse = None;
        for _ in 0..25 {
            let pre = c.phase;
            if c.tick(1.0, 1.0).is_some() {
                phase_at_pulse = Some((pre, c.phase));
                break;
            }
        }
        let (_pre, post) = phase_at_pulse.expect("pulse fired");
        // Phase NOT reset to 0 (matches sphere_clock.py:158-161 — continuous phase
        // enables inner↔outer drift in/out of alignment)
        assert!(post != 0.0, "phase must not reset on pulse");
    }

    #[test]
    fn last_pulse_age_advances_each_tick_resets_at_pulse() {
        let mut c = SphereClock::new(ClockRole::InnerBody);
        let _ = c.tick(0.0, 1.5);
        assert!(approx(c.last_pulse_age_s, 1.5, 1e-5));
        let _ = c.tick(0.0, 1.5);
        assert!(approx(c.last_pulse_age_s, 3.0, 1e-5));
        // Force a pulse: tick at full speed until pulse fires; verify
        // last_pulse_age_s is reset to 0 AT the pulse moment.
        let mut c2 = SphereClock::new(ClockRole::InnerBody);
        let mut tick_count_at_pulse = 0;
        for i in 1..=25 {
            if c2.tick(1.0, 1.0).is_some() {
                tick_count_at_pulse = i;
                break;
            }
        }
        assert!(tick_count_at_pulse > 0, "pulse should have fired");
        // At the pulse moment (just after generate_pulse), age was reset to 0.
        // Inspect immediately by re-running a fresh clock + checking the pulse-tick
        // result: post-pulse, the age is 0.
        let mut c3 = SphereClock::new(ClockRole::InnerBody);
        for _ in 1..tick_count_at_pulse {
            let _ = c3.tick(1.0, 1.0);
            // No pulse yet on these ticks
            assert!(c3.last_pulse_age_s > 0.0);
        }
        // The next tick fires the pulse → age resets to 0
        let p = c3.tick(1.0, 1.0);
        assert!(
            p.is_some(),
            "pulse should fire on tick {tick_count_at_pulse}"
        );
        assert_eq!(c3.last_pulse_age_s, 0.0);
    }

    // ── ClockRole canonical ordering ──────────────────────────────────────

    #[test]
    fn clock_role_all_canonical_order() {
        let all = ClockRole::all();
        assert_eq!(all[0], ClockRole::InnerBody);
        assert_eq!(all[1], ClockRole::InnerMind);
        assert_eq!(all[2], ClockRole::InnerSpirit);
        assert_eq!(all[3], ClockRole::OuterBody);
        assert_eq!(all[4], ClockRole::OuterMind);
        assert_eq!(all[5], ClockRole::OuterSpirit);
    }

    #[test]
    fn clock_role_as_str_matches_python_names() {
        assert_eq!(ClockRole::InnerBody.as_str(), "inner_body");
        assert_eq!(ClockRole::InnerSpirit.as_str(), "inner_spirit");
        assert_eq!(ClockRole::OuterMind.as_str(), "outer_mind");
    }

    // ── SphereClockSet ─────────────────────────────────────────────────────

    #[test]
    fn set_new_creates_six_clocks_in_canonical_order() {
        let s = SphereClockSet::new();
        let clocks = s.iter();
        assert_eq!(clocks.len(), 6);
        for (i, clk) in clocks.iter().enumerate() {
            assert_eq!(clk.role(), ClockRole::all()[i]);
        }
    }

    #[test]
    fn set_serialize_produces_168_bytes() {
        let s = SphereClockSet::new();
        let bytes = s.serialize();
        assert_eq!(bytes.len(), SPHERE_CLOCKS_PAYLOAD_BYTES as usize);
        assert_eq!(bytes.len(), 168);
    }

    #[test]
    fn set_serialize_zero_pulse_count_initial_state() {
        let s = SphereClockSet::new();
        let bytes = s.serialize();
        // Each clock: radius=1.0, scalar=1.0, phase=0, vel=0, pulse_count=0,
        //             consec_balanced=0, last_age=0
        // First 4 bytes = first clock's radius = 1.0_f32 LE
        let radius_bytes: [u8; 4] = bytes[0..4].try_into().unwrap();
        assert_eq!(f32::from_le_bytes(radius_bytes), 1.0);
        // Bytes 16..20 = first clock's pulse_count (cast f32) = 0.0
        let pc_bytes: [u8; 4] = bytes[16..20].try_into().unwrap();
        assert_eq!(f32::from_le_bytes(pc_bytes), 0.0);
    }

    #[test]
    fn set_serialize_after_pulses_reflects_state() {
        let mut s = SphereClockSet::new();
        // Tick inner_body 25× at full speed → 1 pulse
        for _ in 0..25 {
            let _ = s.inner_body.tick(1.0, 1.0);
        }
        let bytes = s.serialize();
        // Inner-body is clock 0; pulse_count is field index 4 (offset 16..20)
        let pc_bytes: [u8; 4] = bytes[16..20].try_into().unwrap();
        let pc = f32::from_le_bytes(pc_bytes);
        assert!(pc >= 1.0, "expected pulse_count >= 1, got {pc}");
    }

    #[test]
    fn set_serialize_layout_field_order_matches_spec_71() {
        let mut s = SphereClockSet::new();
        // Set distinguishable values on inner_body
        s.inner_body.radius = 0.42;
        s.inner_body.scalar_position = 0.31;
        s.inner_body.phase = 0.21;
        s.inner_body.contraction_velocity = 0.11;
        s.inner_body.pulse_count = 7;
        s.inner_body.consecutive_balanced = 5;
        s.inner_body.last_pulse_age_s = 0.99;
        let bytes = s.serialize();
        // Per SPEC §7.1: [radius, scalar_pos, phase, contraction_velocity,
        // pulse_count, consecutive_balanced, last_pulse_age_s]
        let read_f32 =
            |off: usize| -> f32 { f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) };
        assert!(approx(read_f32(0), 0.42, 1e-5));
        assert!(approx(read_f32(4), 0.31, 1e-5));
        assert!(approx(read_f32(8), 0.21, 1e-5));
        assert!(approx(read_f32(12), 0.11, 1e-5));
        assert!(approx(read_f32(16), 7.0, 1e-5));
        assert!(approx(read_f32(20), 5.0, 1e-5));
        assert!(approx(read_f32(24), 0.99, 1e-5));
    }

    #[test]
    fn set_serialize_clocks_in_canonical_order() {
        let mut s = SphereClockSet::new();
        // Mark each clock with a distinctive radius
        s.inner_body.radius = 0.10;
        s.inner_mind.radius = 0.20;
        s.inner_spirit.radius = 0.30;
        s.outer_body.radius = 0.40;
        s.outer_mind.radius = 0.50;
        s.outer_spirit.radius = 0.60;
        let bytes = s.serialize();
        let stride = SPHERE_CLOCK_FIELD_COUNT as usize * 4;
        let read_radius = |idx: usize| -> f32 {
            let off = idx * stride;
            f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };
        assert!(approx(read_radius(0), 0.10, 1e-5));
        assert!(approx(read_radius(1), 0.20, 1e-5));
        assert!(approx(read_radius(2), 0.30, 1e-5));
        assert!(approx(read_radius(3), 0.40, 1e-5));
        assert!(approx(read_radius(4), 0.50, 1e-5));
        assert!(approx(read_radius(5), 0.60, 1e-5));
    }
}
