//! journey — Per-cycle JourneyAccumulator + gift digest formulas for the
//! §G5.1 UP-leg "balance gift" (P0.5 / PLAN §6.5).
//!
//! Each body or mind daemon maintains one [`JourneyAccumulator`] across the
//! current cycle — defined as "ticks since the last `balanced` PulseEvent
//! for this clock". On the next balanced pulse, the daemon calls
//! [`JourneyAccumulator::finalize_body_gift`] (for body daemons) or
//! [`JourneyAccumulator::finalize_mind_gift`] (for mind daemons), publishes
//! the digest as `BODY_BALANCE_GIFT` / `MIND_BALANCE_GIFT`, then [`reset`]s.
//!
//! The trajectory tracked is the SAME tick stream feeding §G5.2's
//! [`crate::homeostasis::stateful_update`] — same `x[t]` per tick, same
//! `obs(prev, prev2)` 5-observable signature. The accumulator is a thin
//! O(1)-per-tick aggregator on top.
//!
//! # Sovereign half
//!
//! PLAN §6.5.1: inner gifts only enrich inner_spirit; outer gifts only
//! enrich outer_spirit. The accumulator is side-agnostic (it never names
//! `Inner` / `Outer`); the daemon that owns the accumulator names its
//! side in the [`TrinitySide`] field of the emitted gift digest.
//!
//! # First-cycle suppression
//!
//! Per PLAN §6.5.2: the very first balanced pulse after daemon boot only
//! RESETS the accumulator — no gift emitted. This avoids gifting a
//! cold-start partial cycle whose excursion_integral / path_length etc.
//! are unrepresentative.

use std::convert::TryInto;

use crate::homeostasis::LayerObs;

/// Schumann sample stride — accumulator captures one snapshot every N ticks
/// into [`JourneyAccumulator::snapshot_ring`] (PLAN §6.5.2 Maker call
/// 2026-05-24 "sample every 3rd tick"). Body 7.83 Hz → 1 snapshot / ~0.38 s;
/// Mind 23.49 Hz → 1 snapshot / ~0.13 s.
pub const JOURNEY_SAMPLE_STRIDE_TICKS: u8 = 3;

/// Snapshot ring depth — 32 slots × N floats per slot (PLAN §6.5.2 storage
/// budget body ~640 B / mind ~1.92 KB). When the ring fills, oldest slot is
/// overwritten (newest 32 snapshots retained — sufficient to reconstruct the
/// shape of a typical cycle since most balanced cycles end well before the
/// 96-tick window).
pub const JOURNEY_SNAPSHOT_RING_LEN: usize = 32;

/// PLAN §6.5.3 BODY gift weights (quantitative essence of journey).
#[derive(Debug, Clone, Copy)]
pub struct BodyGiftWeights {
    /// Total path-length weight (Σ_i path_length[i]).
    pub w_m: f32,
    /// Peak excursion weight (max_i peak_excursion[i]).
    pub w_v: f32,
    /// Excursion integral weight (Σ_i excursion_integral[i]).
    pub w_e: f32,
    /// Duration-normalised weight (hard journey vs easy).
    pub w_t: f32,
}

/// PLAN §6.5.3 BODY default weights (start equal-ish, tune in §G5.2 verify gate).
pub const BODY_GIFT_WEIGHTS: BodyGiftWeights = BodyGiftWeights {
    w_m: 0.30,
    w_v: 0.25,
    w_e: 0.25,
    w_t: 0.20,
};

/// PLAN §6.5.3 MIND gift weights (qualitative essence of journey).
#[derive(Debug, Clone, Copy)]
pub struct MindGiftWeights {
    /// Coherence-climb weight.
    pub w_c: f32,
    /// Polarity-resolution weight (polarity_max − |polarity_at_balance|).
    pub w_p: f32,
    /// Direction-stability weight (1 − Σ_i direction_flips[i] / max_flips).
    pub w_d: f32,
    /// Duration-normalised weight.
    pub w_t: f32,
}

/// PLAN §6.5.3 MIND default weights.
pub const MIND_GIFT_WEIGHTS: MindGiftWeights = MindGiftWeights {
    w_c: 0.40,
    w_p: 0.30,
    w_d: 0.20,
    w_t: 0.10,
};

/// Which half of the trinity a journey/gift belongs to. Sovereign-half is
/// preserved by routing: an [`TrinitySide::Inner`] gift only enriches
/// `inner_spirit_45d`; an [`TrinitySide::Outer`] gift only enriches
/// `outer_spirit_45d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrinitySide {
    /// Inner trinity — inner_body / inner_mind / inner_spirit.
    Inner,
    /// Outer trinity — outer_body / outer_mind / outer_spirit.
    Outer,
}

impl TrinitySide {
    /// Wire encoding for the gift payload `side` field.
    pub fn as_str(self) -> &'static str {
        match self {
            TrinitySide::Inner => "inner",
            TrinitySide::Outer => "outer",
        }
    }
}

/// Per-tick inputs to [`JourneyAccumulator::tick`]. Body daemons supply N=5,
/// mind daemons supply N=15. `obs` is the same 5-observable signature
/// [`crate::homeostasis::observe`] already computes for the §G5.2 spring.
#[derive(Debug, Clone, Copy)]
pub struct JourneyTickInputs<'a, const N: usize> {
    /// The traveling tensor at this Schumann tick (post-§G5.2 integrator).
    pub x: &'a [f32; N],
    /// The 5-observable signature of (prev, prev2) — passed in so the
    /// daemon's existing per-tick `observe()` call is reused (no double-compute).
    pub obs: LayerObs,
    /// Monotonic seconds since UNIX epoch at tick time. Source: daemon's
    /// existing `now_secs()` helper (matches BODY_STATE / MIND_STATE `ts`).
    pub now_secs: f32,
}

/// Per-dim + summary aggregates accumulated across one body/mind cycle.
///
/// All updates are O(N) per tick (a handful of scalar additions/comparisons
/// per dim plus one snapshot copy every [`JOURNEY_SAMPLE_STRIDE_TICKS`]
/// ticks) — well within the µs-scale per-tick budget [PLAN §6.5.2].
#[derive(Debug, Clone)]
pub struct JourneyAccumulator<const N: usize> {
    // Per-dim quantitative aggregates.
    peak_excursion: [f32; N],
    excursion_integral: [f32; N],
    path_length: [f32; N],
    direction_flips: [u16; N],
    /// Sign of the last x[t]−x[t-1] delta per dim; `0.0` = no prior tick.
    last_delta_sign: [f32; N],

    // Per-tick qualitative aggregates (one scalar each, layer-summary).
    coherence_climb_max: f32,
    /// Running min(coherence) — used to compute climb from prior low.
    min_coh_since_start: f32,
    polarity_max: f32,
    polarity_at_balance: f32,

    // Cycle bookkeeping.
    cycle_start_ts: f32,
    last_tick_ts: f32,
    cycle_tick_count: u32,
    last_x: [f32; N],
    has_prior_tick: bool,

    // Snapshot ring (oldest-overwrite when full).
    snapshot_ring: [[f32; N]; JOURNEY_SNAPSHOT_RING_LEN],
    snapshot_count: u8,
    snapshot_head: u8,
    tick_since_last_sample: u8,

    // Cycle-duration normalisation (median EMA per PLAN §6.5.3).
    median_cycle_ema: f32,
    has_completed_cycle: bool,

    // First-cycle suppression (PLAN §6.5.2).
    first_balanced_pulse_seen: bool,
}

impl<const N: usize> JourneyAccumulator<N> {
    /// Fresh accumulator at daemon boot — first balanced pulse will be
    /// consumed without emitting (PLAN §6.5.2 "first cycle suppressed").
    pub fn new() -> Self {
        Self {
            peak_excursion: [0.0; N],
            excursion_integral: [0.0; N],
            path_length: [0.0; N],
            direction_flips: [0; N],
            last_delta_sign: [0.0; N],
            coherence_climb_max: 0.0,
            min_coh_since_start: 1.0,
            polarity_max: 0.0,
            polarity_at_balance: 0.0,
            cycle_start_ts: 0.0,
            last_tick_ts: 0.0,
            cycle_tick_count: 0,
            last_x: [0.0; N],
            has_prior_tick: false,
            snapshot_ring: [[0.0; N]; JOURNEY_SNAPSHOT_RING_LEN],
            snapshot_count: 0,
            snapshot_head: 0,
            tick_since_last_sample: 0,
            median_cycle_ema: 0.0,
            has_completed_cycle: false,
            first_balanced_pulse_seen: false,
        }
    }

    /// Per-Schumann-tick update — O(N). Daemons call this every tick AFTER
    /// the §G5.2 integrator settles the new x[t] (PLAN §6.5.2 "running per-dim
    /// statistics over the cycle").
    pub fn tick(&mut self, inputs: JourneyTickInputs<'_, N>) {
        if self.cycle_tick_count == 0 && !self.has_prior_tick {
            self.cycle_start_ts = inputs.now_secs;
            self.min_coh_since_start = inputs.obs.coherence;
        }
        // ── per-dim aggregates ──────────────────────────────────────────
        let dt = (inputs.now_secs - self.last_tick_ts).max(0.0);
        for i in 0..N {
            let cur = inputs.x[i];
            // 1. peak_excursion = max |x − 0.5|.
            let excursion = (cur - 0.5).abs();
            if excursion > self.peak_excursion[i] {
                self.peak_excursion[i] = excursion;
            }
            // 2. excursion_integral = ∫|x − 0.5|·dt (Riemann sum).
            if self.has_prior_tick {
                self.excursion_integral[i] += excursion * dt;
            }
            // 3. path_length = Σ |x[t] − x[t-1]|.
            if self.has_prior_tick {
                let delta = cur - self.last_x[i];
                self.path_length[i] += delta.abs();
                // 4. direction_flips: count sign(dx/dt) sign changes.
                let sign = if delta > 0.0 {
                    1.0
                } else if delta < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                if sign != 0.0 && self.last_delta_sign[i] != 0.0 && sign != self.last_delta_sign[i]
                {
                    self.direction_flips[i] = self.direction_flips[i].saturating_add(1);
                }
                if sign != 0.0 {
                    self.last_delta_sign[i] = sign;
                }
            }
            self.last_x[i] = cur;
        }

        // ── qualitative aggregates ──────────────────────────────────────
        let coh = inputs.obs.coherence;
        if coh < self.min_coh_since_start {
            self.min_coh_since_start = coh;
        }
        let climb = coh - self.min_coh_since_start;
        if climb > self.coherence_climb_max {
            self.coherence_climb_max = climb;
        }
        let pol_abs = inputs.obs.polarity.abs();
        if pol_abs > self.polarity_max {
            self.polarity_max = pol_abs;
        }

        // ── snapshot ring (stride-sampled) ──────────────────────────────
        if self.tick_since_last_sample == 0 {
            self.snapshot_ring[self.snapshot_head as usize] = *inputs.x;
            self.snapshot_head =
                ((self.snapshot_head as usize + 1) % JOURNEY_SNAPSHOT_RING_LEN) as u8;
            if (self.snapshot_count as usize) < JOURNEY_SNAPSHOT_RING_LEN {
                self.snapshot_count += 1;
            }
        }
        self.tick_since_last_sample =
            (self.tick_since_last_sample + 1) % JOURNEY_SAMPLE_STRIDE_TICKS;

        self.last_tick_ts = inputs.now_secs;
        self.cycle_tick_count = self.cycle_tick_count.saturating_add(1);
        self.has_prior_tick = true;
    }

    /// Mark current tick's `polarity_at_balance` — daemons call this once
    /// at the same moment the balanced pulse rising-edge is detected
    /// (before `finalize_*` + `reset_for_next_cycle`), so the digest
    /// captures the polarity AT the moment of balance (not the cycle-peak).
    pub fn mark_balanced(&mut self, obs: LayerObs) {
        self.polarity_at_balance = obs.polarity.abs();
    }

    /// Finalize the BODY gift digest at the current balanced moment.
    /// Returns `None` for the very first balanced pulse after boot
    /// (PLAN §6.5.2 first-cycle suppression); caller still calls
    /// [`reset_for_next_cycle`] in both branches.
    pub fn finalize_body_gift(&self, weights: &BodyGiftWeights) -> Option<BodyJourneyDigest<N>> {
        if !self.first_balanced_pulse_seen {
            return None;
        }
        if self.cycle_tick_count == 0 {
            return None;
        }
        let cycle_seconds = (self.last_tick_ts - self.cycle_start_ts).max(0.0);
        let median = if self.median_cycle_ema > 0.0 {
            self.median_cycle_ema
        } else {
            cycle_seconds.max(1e-3)
        };
        let duration_norm = (cycle_seconds / median).clamp(0.5, 2.0);

        let sum_path: f32 = self.path_length.iter().sum();
        let max_peak: f32 = self.peak_excursion.iter().copied().fold(0.0_f32, f32::max);
        let sum_integral: f32 = self.excursion_integral.iter().sum();

        let amplitude = weights.w_m * sum_path
            + weights.w_v * max_peak
            + weights.w_e * sum_integral
            + weights.w_t * duration_norm;

        let per_dim_contribution = normalise_to_unit_sum(&self.path_length);

        let snapshots = self.snapshots_oldest_first();

        Some(BodyJourneyDigest {
            gift_amplitude: amplitude.max(0.0),
            cycle_duration_s: cycle_seconds,
            cycle_tick_count: self.cycle_tick_count,
            peak_excursion: self.peak_excursion,
            path_length: self.path_length,
            excursion_integral: self.excursion_integral,
            direction_flips: self.direction_flips,
            polarity_max: self.polarity_max,
            polarity_at_balance: self.polarity_at_balance,
            per_dim_contribution,
            snapshots,
        })
    }

    /// Finalize the MIND gift digest at the current balanced moment.
    /// Returns `None` on first cycle / empty cycle (see body variant).
    pub fn finalize_mind_gift(&self, weights: &MindGiftWeights) -> Option<MindJourneyDigest<N>> {
        if !self.first_balanced_pulse_seen {
            return None;
        }
        if self.cycle_tick_count == 0 {
            return None;
        }
        let cycle_seconds = (self.last_tick_ts - self.cycle_start_ts).max(0.0);
        let median = if self.median_cycle_ema > 0.0 {
            self.median_cycle_ema
        } else {
            cycle_seconds.max(1e-3)
        };
        let duration_norm = (cycle_seconds / median).clamp(0.5, 2.0);

        let polarity_resolution = (self.polarity_max - self.polarity_at_balance).clamp(0.0, 1.0);
        let max_flips = (self.cycle_tick_count.saturating_sub(1) * N as u32) as f32;
        let total_flips: u32 = self.direction_flips.iter().map(|v| *v as u32).sum();
        let purpose_held = if max_flips > 0.0 {
            (1.0 - total_flips as f32 / max_flips).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let amplitude = weights.w_c * self.coherence_climb_max
            + weights.w_p * polarity_resolution
            + weights.w_d * purpose_held
            + weights.w_t * duration_norm;

        let per_dim_contribution = normalise_to_unit_sum(&self.excursion_integral);
        let snapshots = self.snapshots_oldest_first();

        Some(MindJourneyDigest {
            gift_amplitude: amplitude.max(0.0),
            cycle_duration_s: cycle_seconds,
            cycle_tick_count: self.cycle_tick_count,
            peak_excursion: self.peak_excursion,
            path_length: self.path_length,
            excursion_integral: self.excursion_integral,
            direction_flips: self.direction_flips,
            coherence_climb_max: self.coherence_climb_max,
            polarity_max: self.polarity_max,
            polarity_at_balance: self.polarity_at_balance,
            per_dim_contribution,
            snapshots,
        })
    }

    /// Reset state for the next cycle. The median-cycle EMA absorbs the
    /// just-finalised cycle's seconds (only after first complete cycle).
    /// Always mark `first_balanced_pulse_seen = true` after first reset
    /// (so the very first call returns None gifts but subsequent ones emit).
    pub fn reset_for_next_cycle(&mut self) {
        if self.first_balanced_pulse_seen && self.cycle_tick_count > 0 {
            let cycle_seconds = (self.last_tick_ts - self.cycle_start_ts).max(0.0);
            if cycle_seconds > 0.0 {
                if !self.has_completed_cycle {
                    self.median_cycle_ema = cycle_seconds;
                    self.has_completed_cycle = true;
                } else {
                    // Light EMA (α=0.1) — slow drift towards typical cycle length.
                    self.median_cycle_ema = 0.9 * self.median_cycle_ema + 0.1 * cycle_seconds;
                }
            }
        }
        self.first_balanced_pulse_seen = true;
        self.peak_excursion = [0.0; N];
        self.excursion_integral = [0.0; N];
        self.path_length = [0.0; N];
        self.direction_flips = [0; N];
        self.last_delta_sign = [0.0; N];
        self.coherence_climb_max = 0.0;
        self.min_coh_since_start = 1.0;
        self.polarity_max = 0.0;
        self.polarity_at_balance = 0.0;
        self.cycle_start_ts = self.last_tick_ts;
        self.cycle_tick_count = 0;
        // KEEP last_tick_ts + last_x + has_prior_tick — next tick's delta
        // measurement bridges cleanly across the cycle boundary.
        self.snapshot_count = 0;
        self.snapshot_head = 0;
        self.tick_since_last_sample = 0;
    }

    /// Live median-cycle EMA (seconds). Exposed for telemetry / Observatory.
    pub fn median_cycle_seconds(&self) -> f32 {
        self.median_cycle_ema
    }

    /// True iff this accumulator has already received one balanced pulse and
    /// is now tracking its second-or-later cycle (i.e. the next finalize_*
    /// call will return Some(_), not None).
    pub fn is_armed(&self) -> bool {
        self.first_balanced_pulse_seen
    }

    /// Current cycle's tick count — diagnostic.
    pub fn cycle_tick_count(&self) -> u32 {
        self.cycle_tick_count
    }

    /// Return the live snapshot ring in oldest-first order, padded with
    /// the LAST snapshot (or `[0.0; N]` if ring empty) so the returned
    /// `Vec` is always [`JOURNEY_SNAPSHOT_RING_LEN`] long for stable
    /// downstream consumers.
    fn snapshots_oldest_first(&self) -> [[f32; N]; JOURNEY_SNAPSHOT_RING_LEN] {
        let mut out = [[0.0_f32; N]; JOURNEY_SNAPSHOT_RING_LEN];
        let count = self.snapshot_count as usize;
        if count == 0 {
            return out;
        }
        // snapshot_head points at the NEXT-to-write slot. If ring full,
        // oldest is at head; if not full yet, oldest is at slot 0.
        let oldest_idx = if (count) == JOURNEY_SNAPSHOT_RING_LEN {
            self.snapshot_head as usize
        } else {
            0
        };
        for k in 0..count {
            let src = (oldest_idx + k) % JOURNEY_SNAPSHOT_RING_LEN;
            out[k] = self.snapshot_ring[src];
        }
        // Pad tail with the last real sample so downstream u8-quantize is
        // stable (no [0,0,0,..] tail confused with valid 0.0 readings).
        if count < JOURNEY_SNAPSHOT_RING_LEN {
            let last = out[count - 1];
            for k in count..JOURNEY_SNAPSHOT_RING_LEN {
                out[k] = last;
            }
        }
        out
    }
}

impl<const N: usize> Default for JourneyAccumulator<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// PLAN §6.5.1 body-balance gift payload. Body daemons publish this on the
/// `BODY_BALANCE_GIFT` bus event when their sphere clock pulses balanced.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyJourneyDigest<const N: usize> {
    /// Aggregate amplitude per the §6.5.3 body formula; non-negative.
    pub gift_amplitude: f32,
    /// Wall-clock seconds the cycle spanned.
    pub cycle_duration_s: f32,
    /// Schumann ticks the cycle spanned.
    pub cycle_tick_count: u32,
    /// Per-dim peak |x − 0.5|.
    pub peak_excursion: [f32; N],
    /// Per-dim Σ|Δx|.
    pub path_length: [f32; N],
    /// Per-dim ∫|x − 0.5|·dt.
    pub excursion_integral: [f32; N],
    /// Per-dim direction-flip count.
    pub direction_flips: [u16; N],
    /// Layer-max |polarity| during cycle.
    pub polarity_max: f32,
    /// Layer |polarity| at the balanced pulse moment.
    pub polarity_at_balance: f32,
    /// Per-dim contribution to body amplitude, normalised Σ = 1.0
    /// (path_length share).
    pub per_dim_contribution: [f32; N],
    /// Stride-sampled tensor snapshots (oldest-first; padded with last
    /// real snapshot when ring not yet full).
    pub snapshots: [[f32; N]; JOURNEY_SNAPSHOT_RING_LEN],
}

/// PLAN §6.5.1 mind-balance gift payload. Mind daemons publish this on the
/// `MIND_BALANCE_GIFT` bus event when their sphere clock pulses balanced.
#[derive(Debug, Clone, PartialEq)]
pub struct MindJourneyDigest<const N: usize> {
    /// Aggregate amplitude per the §6.5.3 mind formula; non-negative.
    pub gift_amplitude: f32,
    /// Wall-clock seconds the cycle spanned.
    pub cycle_duration_s: f32,
    /// Schumann ticks the cycle spanned.
    pub cycle_tick_count: u32,
    /// Per-dim peak |x − 0.5|.
    pub peak_excursion: [f32; N],
    /// Per-dim Σ|Δx|.
    pub path_length: [f32; N],
    /// Per-dim ∫|x − 0.5|·dt.
    pub excursion_integral: [f32; N],
    /// Per-dim direction-flip count.
    pub direction_flips: [u16; N],
    /// Layer running-max climb in coherence (best (cur − min_seen)).
    pub coherence_climb_max: f32,
    /// Layer-max |polarity| during cycle.
    pub polarity_max: f32,
    /// Layer |polarity| at balanced pulse moment.
    pub polarity_at_balance: f32,
    /// Per-dim contribution to mind amplitude (excursion_integral share).
    pub per_dim_contribution: [f32; N],
    /// Stride-sampled tensor snapshots.
    pub snapshots: [[f32; N]; JOURNEY_SNAPSHOT_RING_LEN],
}

/// Normalise an array to sum=1.0. If the sum is ≤ 0 (zero/empty cycle), all
/// outputs default to `1/N` (uniform — implicit "no dim was the protagonist").
fn normalise_to_unit_sum<const N: usize>(v: &[f32; N]) -> [f32; N] {
    let s: f32 = v.iter().sum();
    let mut out = [0.0_f32; N];
    if s > 0.0 {
        for i in 0..N {
            out[i] = v[i] / s;
        }
    } else {
        let uniform = 1.0 / N as f32;
        out.fill(uniform);
    }
    out
}

/// Pack a `[f32; N]` with values in [0,1] into a `[u8; N]` for SQL storage
/// (PLAN §6.5.6 u8-quantisation; 1/256 ≈ 0.004 precision, well below dim
/// noise floor). Values outside [0,1] are saturated.
pub fn u8_quantise<const N: usize>(v: &[f32; N]) -> [u8; N] {
    let mut out = [0_u8; N];
    for i in 0..N {
        let clipped = v[i].clamp(0.0, 1.0);
        out[i] = (clipped * 255.0).round() as u8;
    }
    out
}

/// Inverse of [`u8_quantise`] — for round-trip tests.
pub fn u8_dequantise<const N: usize>(v: &[u8; N]) -> [f32; N] {
    let mut out = [0.0_f32; N];
    for i in 0..N {
        out[i] = v[i] as f32 / 255.0;
    }
    out
}

/// Pack the whole snapshot ring (32 × N floats) to bytes (32·N bytes).
pub fn u8_quantise_ring<const N: usize>(ring: &[[f32; N]; JOURNEY_SNAPSHOT_RING_LEN]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N * JOURNEY_SNAPSHOT_RING_LEN);
    for snap in ring.iter() {
        let q = u8_quantise(snap);
        out.extend_from_slice(&q);
    }
    out
}

/// Reverse of [`u8_quantise_ring`].
pub fn u8_dequantise_ring<const N: usize>(
    bytes: &[u8],
) -> Option<[[f32; N]; JOURNEY_SNAPSHOT_RING_LEN]> {
    if bytes.len() != N * JOURNEY_SNAPSHOT_RING_LEN {
        return None;
    }
    let mut out = [[0.0_f32; N]; JOURNEY_SNAPSHOT_RING_LEN];
    for k in 0..JOURNEY_SNAPSHOT_RING_LEN {
        let slice = &bytes[k * N..(k + 1) * N];
        let arr: [u8; N] = slice.try_into().ok()?;
        out[k] = u8_dequantise(&arr);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_inputs<const N: usize>(
        x: &'static [f32; N],
        coh: f32,
        pol: f32,
        t: f32,
    ) -> JourneyTickInputs<'static, N> {
        // Caller is responsible for storage backing the &'static slice; tests
        // construct ad-hoc arrays elsewhere.
        let obs = LayerObs {
            coherence: coh,
            magnitude: 0.5,
            velocity: 0.0,
            direction: 0.0,
            polarity: pol,
        };
        JourneyTickInputs {
            x,
            obs,
            now_secs: t,
        }
    }

    #[test]
    fn first_balanced_pulse_suppresses_gift() {
        let mut acc = JourneyAccumulator::<5>::new();
        let x = [0.5; 5];
        acc.tick(JourneyTickInputs {
            x: &x,
            obs: LayerObs::default(),
            now_secs: 0.0,
        });
        acc.tick(JourneyTickInputs {
            x: &x,
            obs: LayerObs::default(),
            now_secs: 0.1,
        });
        acc.mark_balanced(LayerObs::default());
        // Pre-reset: first_balanced_pulse_seen is still false → None.
        assert!(acc.finalize_body_gift(&BODY_GIFT_WEIGHTS).is_none());
        assert!(acc.finalize_mind_gift(&MIND_GIFT_WEIGHTS).is_none());
        acc.reset_for_next_cycle();
        // Subsequent cycle produces a gift (assuming has ticks).
        acc.tick(JourneyTickInputs {
            x: &x,
            obs: LayerObs::default(),
            now_secs: 0.2,
        });
        acc.tick(JourneyTickInputs {
            x: &x,
            obs: LayerObs::default(),
            now_secs: 0.3,
        });
        acc.mark_balanced(LayerObs::default());
        assert!(acc.finalize_body_gift(&BODY_GIFT_WEIGHTS).is_some());
    }

    #[test]
    fn path_length_accumulates_absolute_deltas() {
        let mut acc = JourneyAccumulator::<3>::new();
        let xs: [[f32; 3]; 4] = [
            [0.5, 0.5, 0.5],
            [0.6, 0.4, 0.5],
            [0.4, 0.6, 0.5],
            [0.5, 0.5, 0.5],
        ];
        for (i, x) in xs.iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x,
                obs: LayerObs::default(),
                now_secs: i as f32 * 0.1,
            });
        }
        // Deltas tick-to-tick: |0.6-0.5|+|0.4-0.6|+|0.5-0.4| = 0.1+0.2+0.1 = 0.4 per dim 0.
        // dim 0: 0.1 + 0.2 + 0.1 = 0.4
        // dim 1: 0.1 + 0.2 + 0.1 = 0.4
        // dim 2: 0
        // Internally first tick has no prior, so 3 deltas added.
        assert!((acc.path_length[0] - 0.4).abs() < 1e-5);
        assert!((acc.path_length[1] - 0.4).abs() < 1e-5);
        assert!((acc.path_length[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn peak_excursion_tracks_max_distance_from_centre() {
        let mut acc = JourneyAccumulator::<2>::new();
        let xs: [[f32; 2]; 3] = [[0.5, 0.5], [0.8, 0.4], [0.6, 0.2]];
        for (i, x) in xs.iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x,
                obs: LayerObs::default(),
                now_secs: i as f32 * 0.1,
            });
        }
        // dim0: max(|0.5-0.5|, |0.8-0.5|, |0.6-0.5|) = 0.3
        // dim1: max(0.0, 0.1, 0.3) = 0.3
        assert!((acc.peak_excursion[0] - 0.3).abs() < 1e-5);
        assert!((acc.peak_excursion[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn direction_flips_count_sign_changes() {
        let mut acc = JourneyAccumulator::<1>::new();
        let xs: [[f32; 1]; 5] = [[0.5], [0.6], [0.7], [0.65], [0.7]];
        // deltas: +, +, -, + ⇒ 2 sign changes.
        for (i, x) in xs.iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x,
                obs: LayerObs::default(),
                now_secs: i as f32 * 0.1,
            });
        }
        assert_eq!(acc.direction_flips[0], 2);
    }

    #[test]
    fn coherence_climb_max_captures_best_low_to_high_rise() {
        let mut acc = JourneyAccumulator::<1>::new();
        let cohs = [0.5, 0.3, 0.4, 0.8, 0.6];
        // min_seen: 0.5, 0.3, 0.3, 0.3, 0.3
        // climb: 0.0, 0.0, 0.1, 0.5, 0.3
        // max: 0.5
        let x = [0.5; 1];
        for (i, &c) in cohs.iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x: &x,
                obs: LayerObs {
                    coherence: c,
                    ..LayerObs::default()
                },
                now_secs: i as f32 * 0.1,
            });
        }
        assert!((acc.coherence_climb_max - 0.5).abs() < 1e-5);
    }

    #[test]
    fn body_gift_amplitude_is_weighted_sum() {
        let mut acc = JourneyAccumulator::<2>::new();
        // Arm first cycle.
        let x0 = [0.5_f32, 0.5];
        acc.tick(JourneyTickInputs {
            x: &x0,
            obs: LayerObs::default(),
            now_secs: 0.0,
        });
        acc.mark_balanced(LayerObs::default());
        acc.reset_for_next_cycle();

        // Real cycle: deltas accumulate.
        let xs: [[f32; 2]; 3] = [[0.5, 0.5], [0.7, 0.3], [0.5, 0.5]];
        for (i, x) in xs.iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x,
                obs: LayerObs::default(),
                now_secs: 0.1 + (i as f32 * 0.1),
            });
        }
        acc.mark_balanced(LayerObs::default());
        let g = acc
            .finalize_body_gift(&BODY_GIFT_WEIGHTS)
            .expect("armed cycle");
        // Σpath = (0.2+0.2) + (0.2+0.2) = 0.8 (dim0 + dim1 each ≈0.4 path)
        // max_peak = max(0.2, 0.2) = 0.2
        // Σintegral = roughly 0.2·dt over middle window
        assert!(g.gift_amplitude > 0.0);
        assert_eq!(g.cycle_tick_count, 3);
        // per_dim sums to ~1.0.
        let s: f32 = g.per_dim_contribution.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mind_gift_amplitude_uses_qualitative_terms() {
        let mut acc = JourneyAccumulator::<3>::new();
        let x0 = [0.5_f32; 3];
        acc.tick(JourneyTickInputs {
            x: &x0,
            obs: LayerObs::default(),
            now_secs: 0.0,
        });
        acc.mark_balanced(LayerObs::default());
        acc.reset_for_next_cycle();

        for (i, &c) in [0.5_f32, 0.3, 0.5, 0.8].iter().enumerate() {
            acc.tick(JourneyTickInputs {
                x: &x0,
                obs: LayerObs {
                    coherence: c,
                    polarity: 0.4,
                    ..LayerObs::default()
                },
                now_secs: 0.1 + i as f32 * 0.1,
            });
        }
        acc.mark_balanced(LayerObs {
            polarity: 0.05,
            ..LayerObs::default()
        });
        let g = acc.finalize_mind_gift(&MIND_GIFT_WEIGHTS).unwrap();
        // climb 0.5 -> 0.8 from min 0.3 → climb_max ≥ 0.5
        assert!(g.coherence_climb_max >= 0.5 - 1e-5);
        assert!((g.polarity_max - 0.4).abs() < 1e-5);
        assert!((g.polarity_at_balance - 0.05).abs() < 1e-5);
        assert!(g.gift_amplitude > 0.0);
    }

    #[test]
    fn reset_zeros_aggregates() {
        let mut acc = JourneyAccumulator::<2>::new();
        acc.tick(JourneyTickInputs {
            x: &[0.7, 0.3],
            obs: LayerObs::default(),
            now_secs: 0.0,
        });
        acc.tick(JourneyTickInputs {
            x: &[0.6, 0.4],
            obs: LayerObs::default(),
            now_secs: 0.1,
        });
        acc.mark_balanced(LayerObs::default());
        acc.reset_for_next_cycle();
        // All per-cycle aggregates zeroed.
        for &v in &acc.peak_excursion {
            assert_eq!(v, 0.0);
        }
        for &v in &acc.path_length {
            assert_eq!(v, 0.0);
        }
        for &v in &acc.excursion_integral {
            assert_eq!(v, 0.0);
        }
        assert_eq!(acc.cycle_tick_count, 0);
        assert!(acc.first_balanced_pulse_seen);
    }

    #[test]
    fn snapshot_ring_strides_and_wraps() {
        let mut acc = JourneyAccumulator::<2>::new();
        // Drive enough ticks that the ring fills + wraps.
        let ticks = 100;
        for i in 0..ticks {
            let v = 0.5 + 0.001 * i as f32;
            acc.tick(JourneyTickInputs {
                x: &[v, v + 0.01],
                obs: LayerObs::default(),
                now_secs: i as f32 * 0.1,
            });
        }
        // snapshot every 3 ticks → 100/3 ≈ 33 samples, capped at ring len 32.
        assert_eq!(acc.snapshot_count as usize, JOURNEY_SNAPSHOT_RING_LEN);
    }

    #[test]
    fn u8_quantise_round_trip_within_tolerance() {
        let v: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        let q = u8_quantise(&v);
        let r = u8_dequantise(&q);
        for i in 0..5 {
            assert!((r[i] - v[i]).abs() < 1.0 / 255.0 + 1e-6);
        }
    }

    #[test]
    fn quantise_ring_round_trip() {
        let mut ring = [[0.0_f32; 3]; JOURNEY_SNAPSHOT_RING_LEN];
        for k in 0..JOURNEY_SNAPSHOT_RING_LEN {
            ring[k] = [0.1 * k as f32 % 1.0, 0.5, 0.7];
        }
        let bytes = u8_quantise_ring(&ring);
        assert_eq!(bytes.len(), 3 * JOURNEY_SNAPSHOT_RING_LEN);
        let restored = u8_dequantise_ring::<3>(&bytes).unwrap();
        for k in 0..JOURNEY_SNAPSHOT_RING_LEN {
            for i in 0..3 {
                assert!((restored[k][i] - ring[k][i]).abs() < 1.0 / 255.0 + 1e-6);
            }
        }
    }

    // Suppress unused-warning on the helper used only inside this module.
    #[test]
    fn mk_inputs_compiles() {
        static X: [f32; 2] = [0.5, 0.5];
        let _ = mk_inputs::<2>(&X, 1.0, 0.0, 0.0);
    }
}
