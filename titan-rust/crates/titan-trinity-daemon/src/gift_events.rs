//! gift_events — `BODY_BALANCE_GIFT` + `MIND_BALANCE_GIFT` wire encoders
//! and decoders for the §G5.1 UP-leg balance-gift event family (P0.5 /
//! PLAN §6.5 / D-SPEC-131).
//!
//! Wire shape per `feedback_bus_dst_must_have_subscriber` + SPEC §8.6
//! convention: structured `rmpv::Value::Map` payload — NEVER an opaque
//! binary blob — so consumers can introspect via `rmpv::decode` without
//! a custom binary protocol. Encoders return `Value::Map` (delivered
//! directly to `BusClient::publish` per SPEC §8.2 line 789 + §8.10 line 900,
//! same pattern used by [`crate::subscriptions::encode_filter_down_payload`]).
//!
//! ## Payload schema
//!
//! ```text
//! {
//!   "src":                 "body" | "mind",
//!   "side":                "inner" | "outer",
//!   "gift_amplitude":      f64,
//!   "cycle_duration_s":    f64,
//!   "cycle_tick_count":    u64,
//!   "per_dim_contribution": [f64; N],      // N=5 body, 15 mind
//!   "peak_excursion":      [f64; N],
//!   "path_length":         [f64; N],
//!   "excursion_integral":  [f64; N],
//!   "direction_flips":     [u64; N],
//!   "polarity_max":        f64,
//!   "polarity_at_balance": f64,
//!   "coherence_climb_max": f64,            // present in mind gift only; absent / 0.0 in body
//!   "snapshots":           binary u8-quantised ring (32·N bytes)
//!                          [see `journey::u8_quantise_ring`]
//!   "ts":                  f64
//! }
//! ```
//!
//! Sovereign-half preserved: a payload's `side` is set by the publishing
//! daemon (`Inner` for `titan-{inner-body,inner-mind}-rs`; `Outer` for
//! `titan-{outer-body,outer-mind}-rs`). The receiving spirit daemon
//! (`inner-spirit` / `outer-spirit`) discards gifts of the opposite side.

use rmpv::Value;

use crate::error::DaemonError;
use crate::journey::{
    u8_dequantise_ring, u8_quantise_ring, BodyJourneyDigest, MindJourneyDigest, TrinitySide,
    JOURNEY_SNAPSHOT_RING_LEN,
};

/// Bus message type for the §G5.1 UP-leg body-balance gift.
pub const BODY_BALANCE_GIFT_TOPIC: &str = "BODY_BALANCE_GIFT";

/// Bus message type for the §G5.1 UP-leg mind-balance gift.
pub const MIND_BALANCE_GIFT_TOPIC: &str = "MIND_BALANCE_GIFT";

fn arr_f32<const N: usize>(v: &[f32; N]) -> Value {
    Value::Array(v.iter().map(|f| Value::F64(*f as f64)).collect())
}

fn arr_u16<const N: usize>(v: &[u16; N]) -> Value {
    Value::Array(
        v.iter()
            .map(|n| Value::Integer((*n as u64).into()))
            .collect(),
    )
}

/// Encode a `BODY_BALANCE_GIFT` payload as a structured rmpv Value.
/// `N` is the body tensor dim (5 for both inner_body + outer_body).
pub fn encode_body_balance_gift<const N: usize>(
    side: TrinitySide,
    digest: &BodyJourneyDigest<N>,
    ts: f64,
) -> Value {
    let snapshots_bytes = u8_quantise_ring(&digest.snapshots);
    Value::Map(vec![
        (Value::String("src".into()), Value::String("body".into())),
        (
            Value::String("side".into()),
            Value::String(side.as_str().into()),
        ),
        (
            Value::String("gift_amplitude".into()),
            Value::F64(digest.gift_amplitude as f64),
        ),
        (
            Value::String("cycle_duration_s".into()),
            Value::F64(digest.cycle_duration_s as f64),
        ),
        (
            Value::String("cycle_tick_count".into()),
            Value::Integer((digest.cycle_tick_count as u64).into()),
        ),
        (
            Value::String("per_dim_contribution".into()),
            arr_f32(&digest.per_dim_contribution),
        ),
        (
            Value::String("peak_excursion".into()),
            arr_f32(&digest.peak_excursion),
        ),
        (
            Value::String("path_length".into()),
            arr_f32(&digest.path_length),
        ),
        (
            Value::String("excursion_integral".into()),
            arr_f32(&digest.excursion_integral),
        ),
        (
            Value::String("direction_flips".into()),
            arr_u16(&digest.direction_flips),
        ),
        (
            Value::String("polarity_max".into()),
            Value::F64(digest.polarity_max as f64),
        ),
        (
            Value::String("polarity_at_balance".into()),
            Value::F64(digest.polarity_at_balance as f64),
        ),
        (
            Value::String("snapshots".into()),
            Value::Binary(snapshots_bytes),
        ),
        (Value::String("ts".into()), Value::F64(ts)),
    ])
}

/// Encode a `MIND_BALANCE_GIFT` payload. `N` = 15 for both inner_mind + outer_mind.
pub fn encode_mind_balance_gift<const N: usize>(
    side: TrinitySide,
    digest: &MindJourneyDigest<N>,
    ts: f64,
) -> Value {
    let snapshots_bytes = u8_quantise_ring(&digest.snapshots);
    Value::Map(vec![
        (Value::String("src".into()), Value::String("mind".into())),
        (
            Value::String("side".into()),
            Value::String(side.as_str().into()),
        ),
        (
            Value::String("gift_amplitude".into()),
            Value::F64(digest.gift_amplitude as f64),
        ),
        (
            Value::String("cycle_duration_s".into()),
            Value::F64(digest.cycle_duration_s as f64),
        ),
        (
            Value::String("cycle_tick_count".into()),
            Value::Integer((digest.cycle_tick_count as u64).into()),
        ),
        (
            Value::String("per_dim_contribution".into()),
            arr_f32(&digest.per_dim_contribution),
        ),
        (
            Value::String("peak_excursion".into()),
            arr_f32(&digest.peak_excursion),
        ),
        (
            Value::String("path_length".into()),
            arr_f32(&digest.path_length),
        ),
        (
            Value::String("excursion_integral".into()),
            arr_f32(&digest.excursion_integral),
        ),
        (
            Value::String("direction_flips".into()),
            arr_u16(&digest.direction_flips),
        ),
        (
            Value::String("coherence_climb_max".into()),
            Value::F64(digest.coherence_climb_max as f64),
        ),
        (
            Value::String("polarity_max".into()),
            Value::F64(digest.polarity_max as f64),
        ),
        (
            Value::String("polarity_at_balance".into()),
            Value::F64(digest.polarity_at_balance as f64),
        ),
        (
            Value::String("snapshots".into()),
            Value::Binary(snapshots_bytes),
        ),
        (Value::String("ts".into()), Value::F64(ts)),
    ])
}

/// Subset of [`BodyJourneyDigest`] / [`MindJourneyDigest`] that the spirit
/// daemon actually needs from a decoded payload — `gift_amplitude` and
/// (optionally) `side` for sovereign-half routing. The full per-dim
/// arrays are decoded to a Vec because the spirit daemon only multiplies
/// the amplitude by the const Q/L/D mask; it doesn't need the original
/// tensor digest. (The persistence worker on Python L2 reads the full
/// payload and writes the SQL row independently — that's a separate path.)
#[derive(Debug, Clone)]
pub struct GiftAtSpiritIn {
    /// Sovereign half — Inner / Outer.
    pub side: TrinitySide,
    /// Aggregate gift amplitude.
    pub gift_amplitude: f32,
    /// Wall-clock seconds the cycle spanned.
    pub cycle_duration_s: f32,
    /// Schumann ticks the cycle spanned.
    pub cycle_tick_count: u32,
    /// Publisher timestamp (passes through).
    pub ts: f64,
}

/// Decode a `BODY_BALANCE_GIFT` or `MIND_BALANCE_GIFT` payload to the
/// fields the spirit daemon needs. Returns `Err` on any schema drift —
/// caller should `warn!` and skip the gift (defensive design; one bad
/// gift never blocks the spirit tick).
pub fn decode_gift_at_spirit(payload: &Value) -> Result<GiftAtSpiritIn, DaemonError> {
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("payload not a map".into())),
    };
    let mut side: Option<TrinitySide> = None;
    let mut amp: Option<f32> = None;
    let mut dur: Option<f32> = None;
    let mut ticks: Option<u32> = None;
    let mut ts: f64 = 0.0;
    for (k, v) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
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
            "gift_amplitude" => amp = v.as_f64().map(|f| f as f32),
            "cycle_duration_s" => dur = v.as_f64().map(|f| f as f32),
            "cycle_tick_count" => ticks = v.as_u64().map(|n| n as u32),
            "ts" => ts = v.as_f64().unwrap_or(0.0),
            _ => {}
        }
    }
    let side =
        side.ok_or_else(|| DaemonError::MsgpackDecode("missing/invalid side field".into()))?;
    let amp =
        amp.ok_or_else(|| DaemonError::MsgpackDecode("missing gift_amplitude field".into()))?;
    let dur = dur.unwrap_or(0.0);
    let ticks = ticks.unwrap_or(0);
    Ok(GiftAtSpiritIn {
        side,
        gift_amplitude: amp,
        cycle_duration_s: dur,
        cycle_tick_count: ticks,
        ts,
    })
}

/// Test-helper: decode the snapshots Binary back to a ring (returns `None`
/// when payload has no snapshots field or bytes length mismatch). Not used
/// by the spirit daemon (it ignores snapshots — that's the persistence
/// worker's concern) but exposed for the journey_persistence_worker test
/// harness on the Python L2 side.
pub fn decode_snapshots<const N: usize>(
    payload: &Value,
) -> Option<[[f32; N]; JOURNEY_SNAPSHOT_RING_LEN]> {
    let map = match payload {
        Value::Map(items) => items,
        _ => return None,
    };
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("snapshots") {
                if let Value::Binary(bytes) = v {
                    return u8_dequantise_ring::<N>(bytes);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journey::{BODY_GIFT_WEIGHTS, JOURNEY_SNAPSHOT_RING_LEN};

    fn synthetic_body_digest() -> BodyJourneyDigest<5> {
        BodyJourneyDigest {
            gift_amplitude: 0.42,
            cycle_duration_s: 3.5,
            cycle_tick_count: 27,
            peak_excursion: [0.1, 0.2, 0.3, 0.05, 0.15],
            path_length: [0.5, 0.6, 0.7, 0.4, 0.55],
            excursion_integral: [0.1; 5],
            direction_flips: [2, 1, 3, 0, 1],
            polarity_max: 0.25,
            polarity_at_balance: 0.05,
            per_dim_contribution: [0.2; 5],
            snapshots: [[0.5_f32; 5]; JOURNEY_SNAPSHOT_RING_LEN],
        }
    }

    fn synthetic_mind_digest() -> MindJourneyDigest<15> {
        MindJourneyDigest {
            gift_amplitude: 0.33,
            cycle_duration_s: 1.2,
            cycle_tick_count: 28,
            peak_excursion: [0.2; 15],
            path_length: [0.5; 15],
            excursion_integral: [0.1; 15],
            direction_flips: [1; 15],
            coherence_climb_max: 0.45,
            polarity_max: 0.6,
            polarity_at_balance: 0.1,
            per_dim_contribution: [1.0 / 15.0; 15],
            snapshots: [[0.5_f32; 15]; JOURNEY_SNAPSHOT_RING_LEN],
        }
    }

    #[test]
    fn body_gift_encode_decode_round_trip() {
        let d = synthetic_body_digest();
        let payload = encode_body_balance_gift::<5>(TrinitySide::Inner, &d, 1234.5);
        let dec = decode_gift_at_spirit(&payload).unwrap();
        assert_eq!(dec.side, TrinitySide::Inner);
        assert!((dec.gift_amplitude - 0.42).abs() < 1e-5);
        assert!((dec.cycle_duration_s - 3.5).abs() < 1e-5);
        assert_eq!(dec.cycle_tick_count, 27);
        assert!((dec.ts - 1234.5).abs() < 1e-3);
    }

    #[test]
    fn mind_gift_encode_decode_round_trip() {
        let d = synthetic_mind_digest();
        let payload = encode_mind_balance_gift::<15>(TrinitySide::Outer, &d, 1.0);
        let dec = decode_gift_at_spirit(&payload).unwrap();
        assert_eq!(dec.side, TrinitySide::Outer);
        assert!((dec.gift_amplitude - 0.33).abs() < 1e-5);
    }

    #[test]
    fn decode_rejects_non_map_payload() {
        let v = Value::String("not a map".into());
        assert!(decode_gift_at_spirit(&v).is_err());
    }

    #[test]
    fn decode_rejects_missing_side() {
        let mut m = vec![(Value::String("gift_amplitude".into()), Value::F64(0.5))];
        m.push((Value::String("src".into()), Value::String("body".into())));
        let v = Value::Map(m);
        assert!(decode_gift_at_spirit(&v).is_err());
    }

    #[test]
    fn snapshots_round_trip_via_binary() {
        let d = synthetic_body_digest();
        let payload = encode_body_balance_gift::<5>(TrinitySide::Inner, &d, 0.0);
        let restored = decode_snapshots::<5>(&payload).unwrap();
        for k in 0..JOURNEY_SNAPSHOT_RING_LEN {
            for i in 0..5 {
                assert!((restored[k][i] - 0.5).abs() < 1.0 / 255.0 + 1e-6);
            }
        }
    }

    #[test]
    fn topic_strings_are_canonical() {
        assert_eq!(BODY_BALANCE_GIFT_TOPIC, "BODY_BALANCE_GIFT");
        assert_eq!(MIND_BALANCE_GIFT_TOPIC, "MIND_BALANCE_GIFT");
    }

    #[test]
    fn weight_constants_loadable_at_use_site() {
        // sanity: BODY_GIFT_WEIGHTS still has w_m field (covered indirectly by
        // journey tests, but checked here so subscriber rewires can rely on it).
        let _ = BODY_GIFT_WEIGHTS.w_m;
    }
}
