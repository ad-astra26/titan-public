//! corrective_events — `EXTREME_IMBALANCE_DETECTED` + `CORRECTIVE_NUDGE`
//! wire encoders + decoders for the P0.6-C polarity-correction event chain
//! (PLAN §6.6.3-§6.6.5 / D-SPEC-132).
//!
//! Event flow per the sovereign-half lock:
//!
//! 1. Body / mind daemon's [`crate::PolarityHomeostat::tick`] fires →
//!    publishes `EXTREME_IMBALANCE_DETECTED` (this module's
//!    [`encode_extreme_imbalance`]). Payload carries `src` (body/mind) +
//!    `side` (inner/outer) + dominant_dim_idx + duration + telemetry.
//! 2. Spirit daemon of the SAME side subscribes → on receipt computes the
//!    nudge amplitude via the §6.6.4 emergent formula:
//!    `nudge_amp = base_gain × excess × (1/max(0.1, chi)) × (1 + atan(rate/upper))`
//!    → publishes `CORRECTIVE_NUDGE` (this module's [`encode_corrective_nudge`])
//!    targeting the originating body/mind daemon's `target_dim_idx`.
//! 3. Body / mind daemon subscribes to CORRECTIVE_NUDGE → on receipt stores
//!    `(target_dim_idx, signed_nudge)` for one-shot application on the next
//!    tick (composes into enrichment_force toward 0.5 on that dim only).
//!
//! Cross-half filtering: target_src + target_side fields let each daemon
//! reject events not addressed to it (e.g. inner-body discards CORRECTIVE_NUDGE
//! addressed to outer-mind).

use rmpv::Value;

use crate::error::DaemonError;
use crate::journey::TrinitySide;
use crate::polarity_homeostat::ExtremeImbalanceEvent;

/// Bus message type for body/mind→spirit imbalance detection.
pub const EXTREME_IMBALANCE_DETECTED_TOPIC: &str = "EXTREME_IMBALANCE_DETECTED";

/// Bus message type for spirit→body/mind corrective intervention.
pub const CORRECTIVE_NUDGE_TOPIC: &str = "CORRECTIVE_NUDGE";

/// Encode an `EXTREME_IMBALANCE_DETECTED` payload. `src` is "body" or "mind"
/// — the canonical name of the publishing daemon's tensor.
pub fn encode_extreme_imbalance(
    src: &str,
    side: TrinitySide,
    ev: &ExtremeImbalanceEvent,
    ts: f64,
) -> Value {
    Value::Map(vec![
        (Value::String("src".into()), Value::String(src.into())),
        (
            Value::String("side".into()),
            Value::String(side.as_str().into()),
        ),
        (
            Value::String("dominant_dim_idx".into()),
            Value::Integer((ev.dominant_dim_idx as u64).into()),
        ),
        (
            Value::String("dominant_dim_value".into()),
            Value::F64(ev.dominant_dim_value as f64),
        ),
        (
            Value::String("polarity_at_fire".into()),
            Value::F64(ev.polarity_at_fire as f64),
        ),
        (
            Value::String("polarity_sign".into()),
            Value::F64(ev.polarity_sign as f64),
        ),
        (
            Value::String("duration_ticks".into()),
            Value::Integer((ev.duration_ticks as u64).into()),
        ),
        (
            Value::String("sigma_multiplier".into()),
            Value::F64(ev.sigma_multiplier as f64),
        ),
        (
            Value::String("extreme_event_count_lifetime".into()),
            Value::Integer(ev.extreme_event_count_lifetime.into()),
        ),
        (Value::String("ts".into()), Value::F64(ts)),
    ])
}

/// Decoded form of `EXTREME_IMBALANCE_DETECTED` carrying only the fields
/// the receiving spirit daemon needs to compute the corrective amplitude
/// + target the back-reply.
#[derive(Debug, Clone)]
pub struct ExtremeImbalanceIn {
    /// Origin daemon's tensor: "body" or "mind".
    pub src: String,
    /// Sovereign half — inner / outer.
    pub side: TrinitySide,
    /// Origin dim index that was the protagonist.
    pub dominant_dim_idx: u32,
    /// Polarity-sign of the imbalance — drives the nudge direction.
    pub polarity_sign: f32,
    /// Excess fraction (|polarity| − baseline) at fire — feeds the amplitude.
    pub polarity_at_fire: f32,
    /// Live σ multiplier at fire (carried for telemetry / chronicity calc).
    pub sigma_multiplier: f32,
    /// Wall-clock timestamp.
    pub ts: f64,
}

/// Decode an `EXTREME_IMBALANCE_DETECTED` payload at the spirit daemon.
pub fn decode_extreme_imbalance(payload: &Value) -> Result<ExtremeImbalanceIn, DaemonError> {
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("payload not a map".into())),
    };
    let mut src = String::new();
    let mut side: Option<TrinitySide> = None;
    let mut dom_idx: u32 = 0;
    let mut pol_sign: f32 = 0.0;
    let mut pol_at_fire: f32 = 0.0;
    let mut sigma: f32 = 0.0;
    let mut ts: f64 = 0.0;
    for (k, v) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "src" => {
                if let Value::String(s) = v {
                    src = s.as_str().unwrap_or("").to_string();
                }
            }
            "side" => {
                let s = match v {
                    Value::String(s) => s.as_str().unwrap_or(""),
                    _ => "",
                };
                side = match s {
                    "inner" => Some(TrinitySide::Inner),
                    "outer" => Some(TrinitySide::Outer),
                    _ => None,
                };
            }
            "dominant_dim_idx" => dom_idx = v.as_u64().unwrap_or(0) as u32,
            "polarity_sign" => pol_sign = v.as_f64().unwrap_or(0.0) as f32,
            "polarity_at_fire" => pol_at_fire = v.as_f64().unwrap_or(0.0) as f32,
            "sigma_multiplier" => sigma = v.as_f64().unwrap_or(0.0) as f32,
            "ts" => ts = v.as_f64().unwrap_or(0.0),
            _ => {}
        }
    }
    let side = side.ok_or_else(|| DaemonError::MsgpackDecode("missing/invalid side".into()))?;
    if src.is_empty() {
        return Err(DaemonError::MsgpackDecode("missing src".into()));
    }
    Ok(ExtremeImbalanceIn {
        src,
        side,
        dominant_dim_idx: dom_idx,
        polarity_sign: pol_sign,
        polarity_at_fire: pol_at_fire,
        sigma_multiplier: sigma,
        ts,
    })
}

/// Encode a `CORRECTIVE_NUDGE` payload published by spirit. `target_src` is
/// "body" or "mind" of the receiving daemon.
pub fn encode_corrective_nudge(
    target_src: &str,
    target_side: TrinitySide,
    target_dim_idx: u32,
    nudge_value: f32,
    intensity: f32,
    ts: f64,
) -> Value {
    Value::Map(vec![
        (
            Value::String("target_src".into()),
            Value::String(target_src.into()),
        ),
        (
            Value::String("target_side".into()),
            Value::String(target_side.as_str().into()),
        ),
        (
            Value::String("target_dim_idx".into()),
            Value::Integer((target_dim_idx as u64).into()),
        ),
        (
            Value::String("nudge_value".into()),
            Value::F64(nudge_value as f64),
        ),
        (
            Value::String("intensity".into()),
            Value::F64(intensity as f64),
        ),
        (Value::String("ts".into()), Value::F64(ts)),
    ])
}

/// Decoded form of `CORRECTIVE_NUDGE` carrying only the fields the receiving
/// body / mind daemon needs to apply the one-shot intervention.
#[derive(Debug, Clone)]
pub struct CorrectiveNudgeIn {
    /// Target daemon's tensor: "body" or "mind".
    pub target_src: String,
    /// Target sovereign half.
    pub target_side: TrinitySide,
    /// Target dim index (0..N).
    pub target_dim_idx: u32,
    /// Signed nudge magnitude (caller adds to enrichment[dim_idx]).
    pub nudge_value: f32,
    /// Telemetry intensity (= |nudge_value| before sign).
    pub intensity: f32,
    /// Wall-clock timestamp.
    pub ts: f64,
}

/// Decode a `CORRECTIVE_NUDGE` payload at the body / mind daemon.
pub fn decode_corrective_nudge(payload: &Value) -> Result<CorrectiveNudgeIn, DaemonError> {
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("payload not a map".into())),
    };
    let mut target_src = String::new();
    let mut target_side: Option<TrinitySide> = None;
    let mut target_dim_idx: u32 = 0;
    let mut nudge_value: f32 = 0.0;
    let mut intensity: f32 = 0.0;
    let mut ts: f64 = 0.0;
    for (k, v) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "target_src" => {
                if let Value::String(s) = v {
                    target_src = s.as_str().unwrap_or("").to_string();
                }
            }
            "target_side" => {
                let s = match v {
                    Value::String(s) => s.as_str().unwrap_or(""),
                    _ => "",
                };
                target_side = match s {
                    "inner" => Some(TrinitySide::Inner),
                    "outer" => Some(TrinitySide::Outer),
                    _ => None,
                };
            }
            "target_dim_idx" => target_dim_idx = v.as_u64().unwrap_or(0) as u32,
            "nudge_value" => nudge_value = v.as_f64().unwrap_or(0.0) as f32,
            "intensity" => intensity = v.as_f64().unwrap_or(0.0) as f32,
            "ts" => ts = v.as_f64().unwrap_or(0.0),
            _ => {}
        }
    }
    let target_side =
        target_side.ok_or_else(|| DaemonError::MsgpackDecode("missing target_side".into()))?;
    if target_src.is_empty() {
        return Err(DaemonError::MsgpackDecode("missing target_src".into()));
    }
    Ok(CorrectiveNudgeIn {
        target_src,
        target_side,
        target_dim_idx,
        nudge_value,
        intensity,
        ts,
    })
}

/// PLAN §6.6.4 emergent nudge amplitude formula. Computed by the spirit
/// daemon on receipt of `EXTREME_IMBALANCE_DETECTED`.
///
/// ```text
/// excess          = (|polarity| − baseline − threshold)
///                   / (1.0 − baseline − threshold)   // 0..1
/// metabolic       = 1.0 / max(0.1, chi_health)
/// chronicity      = 1.0 + atan(rate_24h_ema / rate_target_upper)
/// nudge_amp       = base_gain × excess × metabolic × chronicity
/// ```
///
/// `chi_health` ∈ [0, 1] is read from `chi_state.bin` SHM (or 1.0 if
/// unavailable — graceful default; low-chi amplification still works for
/// the steady-state Titan). `base_gain` is the only tunable amplitude knob
/// per PLAN §6.6.3.
pub fn compute_nudge_amplitude(
    polarity_at_fire: f32,
    baseline: f32,
    threshold: f32,
    chi_health: f32,
    rate_24h_ema: f32,
    rate_target_upper: f32,
    base_gain: f32,
) -> f32 {
    let head_room = (1.0 - baseline - threshold).max(1.0e-6);
    let excess = ((polarity_at_fire - baseline - threshold) / head_room).clamp(0.0, 1.0);
    let chi_clamped = chi_health.max(0.1);
    let metabolic = 1.0 / chi_clamped;
    let chronicity = 1.0 + (rate_24h_ema / rate_target_upper.max(1.0e-6)).atan();
    let amp = base_gain * excess * metabolic * chronicity;
    amp.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event() -> ExtremeImbalanceEvent {
        ExtremeImbalanceEvent {
            dominant_dim_idx: 3,
            dominant_dim_value: 0.92,
            polarity_at_fire: 0.74,
            polarity_sign: 1.0,
            duration_ticks: 47,
            sigma_multiplier: 2.7,
            extreme_event_count_lifetime: 12,
        }
    }

    #[test]
    fn extreme_imbalance_round_trip_inner_body() {
        let ev = make_event();
        let payload = encode_extreme_imbalance("body", TrinitySide::Inner, &ev, 1234.5);
        let dec = decode_extreme_imbalance(&payload).unwrap();
        assert_eq!(dec.src, "body");
        assert_eq!(dec.side, TrinitySide::Inner);
        assert_eq!(dec.dominant_dim_idx, 3);
        assert!((dec.polarity_sign - 1.0).abs() < 1e-6);
        assert!((dec.sigma_multiplier - 2.7).abs() < 1e-5);
        assert!((dec.ts - 1234.5).abs() < 1e-3);
    }

    #[test]
    fn extreme_imbalance_round_trip_outer_mind() {
        let ev = make_event();
        let payload = encode_extreme_imbalance("mind", TrinitySide::Outer, &ev, 99.0);
        let dec = decode_extreme_imbalance(&payload).unwrap();
        assert_eq!(dec.src, "mind");
        assert_eq!(dec.side, TrinitySide::Outer);
    }

    #[test]
    fn corrective_nudge_round_trip() {
        let payload = encode_corrective_nudge("body", TrinitySide::Inner, 2, -0.05, 0.05, 10.0);
        let dec = decode_corrective_nudge(&payload).unwrap();
        assert_eq!(dec.target_src, "body");
        assert_eq!(dec.target_side, TrinitySide::Inner);
        assert_eq!(dec.target_dim_idx, 2);
        assert!((dec.nudge_value - (-0.05)).abs() < 1e-6);
        assert!((dec.intensity - 0.05).abs() < 1e-6);
    }

    #[test]
    fn corrective_nudge_decode_rejects_missing_side() {
        let v = Value::Map(vec![(
            Value::String("target_src".into()),
            Value::String("body".into()),
        )]);
        assert!(decode_corrective_nudge(&v).is_err());
    }

    #[test]
    fn nudge_amplitude_proportional_to_excess() {
        // Holding everything else fixed, doubling the excess doubles the amp.
        let amp_small = compute_nudge_amplitude(0.55, 0.1, 0.2, 0.5, 5.0, 50.0, 0.1);
        let amp_big = compute_nudge_amplitude(0.9, 0.1, 0.2, 0.5, 5.0, 50.0, 0.1);
        assert!(amp_big > amp_small, "bigger excess → bigger amp");
    }

    #[test]
    fn nudge_amplitude_inversely_proportional_to_chi() {
        let amp_healthy = compute_nudge_amplitude(0.6, 0.1, 0.2, 1.0, 5.0, 50.0, 0.1);
        let amp_failing = compute_nudge_amplitude(0.6, 0.1, 0.2, 0.1, 5.0, 50.0, 0.1);
        assert!(
            amp_failing > amp_healthy,
            "low chi → stronger correction (metabolic factor 1/chi)",
        );
    }

    #[test]
    fn nudge_amplitude_chronicity_grows_with_rate() {
        let amp_fresh = compute_nudge_amplitude(0.6, 0.1, 0.2, 0.5, 0.0, 50.0, 0.1);
        let amp_chronic = compute_nudge_amplitude(0.6, 0.1, 0.2, 0.5, 100.0, 50.0, 0.1);
        assert!(
            amp_chronic > amp_fresh,
            "chronic offender → escalating amplitude (chronicity factor)",
        );
    }

    #[test]
    fn nudge_amplitude_floors_at_zero() {
        // Polarity just at threshold = 0 excess → amplitude = 0.
        let amp = compute_nudge_amplitude(0.3, 0.1, 0.2, 1.0, 5.0, 50.0, 0.1);
        assert!(amp.abs() < 1e-6);
    }

    #[test]
    fn topic_strings_are_canonical() {
        assert_eq!(
            EXTREME_IMBALANCE_DETECTED_TOPIC,
            "EXTREME_IMBALANCE_DETECTED"
        );
        assert_eq!(CORRECTIVE_NUDGE_TOPIC, "CORRECTIVE_NUDGE");
    }
}
