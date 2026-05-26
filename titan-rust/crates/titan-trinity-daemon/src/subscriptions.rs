//! subscriptions — Daemon-side bus subscription helpers.
//!
//! Wraps `titan_bus::BusClient` (imported from C-S4 — see `[C5-bridge]`
//! commit) with conveniences for the 6 trinity daemons (C-S5 inner +
//! C-S6 outer):
//!
//! - `connect_daemon` — single call that resolves env vars, connects to
//!   the main bus socket, performs HMAC handshake, and subscribes to a
//!   topic list.
//! - `decode_filter_down_payload` — typed decoder for
//!   UNIFIED_SPIRIT_FILTER_DOWN (per SPEC §8.6) that returns the inner
//!   slice (5 + 15 + 40 = 60 multipliers) for inner daemons.
//! - `decode_local_filter_down_payload` — typed decoder for
//!   INNER_SPIRIT_FILTER_DOWN (Phase C addition per SPEC §8.6).
//!
//! # SPEC discipline
//!
//! Daemon subscription lists are SPEC §9.A REQUIRED contracts:
//!   inner-body / inner-mind / outer-body / outer-mind:
//!     KERNEL_SHUTDOWN_ANNOUNCE, UNIFIED_SPIRIT_FILTER_DOWN,
//!     {INNER,OUTER}_SPIRIT_FILTER_DOWN (per side),
//!     TRINITY_SUBSTRATE_TOPOLOGY_UPDATED.
//!   inner-spirit / outer-spirit (no topology — no ground_up):
//!     KERNEL_SHUTDOWN_ANNOUNCE, UNIFIED_SPIRIT_FILTER_DOWN.
//! Drift = `arch_map phase-c verify` failure.

use std::path::Path;

use titan_bus::BusClient;

use crate::error::DaemonError;

/// SPEC §9.A REQUIRED bus subscriptions for `titan-inner-body-rs`.
pub const INNER_BODY_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    "INNER_SPIRIT_FILTER_DOWN",
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
    // Phase 0 / 0E (D-SPEC-97 refinement): ground_up nudge is RECOMPUTED
    // once per kernel epoch (held + applied per Schumann tick), NOT per tick.
    "KERNEL_EPOCH_TICK",
];

/// SPEC §9.A REQUIRED bus subscriptions for `titan-inner-mind-rs`.
pub const INNER_MIND_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    "INNER_SPIRIT_FILTER_DOWN",
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
    // Phase 0 / 0E: ground_up nudge recomputed per kernel epoch.
    "KERNEL_EPOCH_TICK",
];

/// SPEC §9.A REQUIRED bus subscriptions for `titan-inner-spirit-rs`.
/// Inner-spirit does NOT subscribe to topology (it's the Observer of
/// inner_body / inner_mind sibling slots; no ground_up applied to spirit).
pub const INNER_SPIRIT_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    // Phase 0 / D-SPEC-97: small filter_down fires once per kernel epoch
    // (Rust-native), NOT per Schumann tick.
    "KERNEL_EPOCH_TICK",
    // P0.5 / D-SPEC-131 §G5.1 UP-leg gift: inner_body + inner_mind publish
    // a meaning-mapped journey digest on each balanced PulseEvent (sub-1%
    // of Schumann ticks). Inner-spirit applies it via the per-dim Q/L/D
    // mask in titan-trinity-daemon::up_leg_masks. Outer gifts filtered by
    // payload `side` at decode time (sovereign-half lock per PLAN §6.5.1).
    "BODY_BALANCE_GIFT",
    "MIND_BALANCE_GIFT",
];

/// SPEC §9.A REQUIRED bus subscriptions for `titan-outer-body-rs`.
/// Provided here for C-S6 reuse (the inner trinity daemons consume the
/// inner trio above).
pub const OUTER_BODY_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    "OUTER_SPIRIT_FILTER_DOWN",
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
    // Phase 0 / 0E: ground_up nudge recomputed per kernel epoch.
    "KERNEL_EPOCH_TICK",
];

/// SPEC §9.A REQUIRED bus subscriptions for `titan-outer-mind-rs`.
pub const OUTER_MIND_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    "OUTER_SPIRIT_FILTER_DOWN",
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
    // Phase 0 / 0E: ground_up nudge recomputed per kernel epoch.
    "KERNEL_EPOCH_TICK",
];

/// SPEC §9.A REQUIRED bus subscriptions for `titan-outer-spirit-rs`.
pub const OUTER_SPIRIT_TOPICS: &[&str] = &[
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "UNIFIED_SPIRIT_FILTER_DOWN",
    // Phase 0 / D-SPEC-97: small filter_down fires once per kernel epoch
    // (Rust-native), NOT per Schumann tick.
    "KERNEL_EPOCH_TICK",
    // P0.5 / D-SPEC-131 §G5.1 UP-leg gift — outer mirror (see inner above).
    "BODY_BALANCE_GIFT",
    "MIND_BALANCE_GIFT",
];

/// Connect a daemon to the main bus + subscribe to its REQUIRED topics
/// in one call.
pub async fn connect_daemon(
    socket_path: &Path,
    authkey: &[u8],
    client_name: &str,
    topics: &[&str],
) -> Result<BusClient, DaemonError> {
    let client = BusClient::connect(socket_path, authkey, client_name)
        .await
        .map_err(|e| DaemonError::MsgpackEncode(format!("bus connect: {e}")))?;
    client
        .subscribe(topics)
        .await
        .map_err(|e| DaemonError::MsgpackEncode(format!("bus subscribe: {e}")))?;
    Ok(client)
}

/// UNIFIED_SPIRIT_FILTER_DOWN payload per SPEC §8.6 (inner slice only —
/// outer fields are decoded but discarded since inner daemons never use
/// them).
#[derive(Debug, Clone)]
pub struct InnerFilterDownPayload {
    /// 5 multipliers for inner_body (positional).
    pub inner_body: [f32; 5],
    /// 15 multipliers for inner_mind (positional).
    pub inner_mind: [f32; 15],
    /// 40 multipliers for inner_spirit_content (G8 observer dims [0:5]
    /// already masked at publish side).
    pub inner_spirit_content: [f32; 40],
    /// Epoch counter from the publisher (passes through for trace).
    pub epoch_id: i64,
    /// Wall-clock timestamp from the publisher.
    pub ts: f64,
}

/// Decode an UNIFIED_SPIRIT_FILTER_DOWN structured payload to typed inner
/// multipliers. Outer fields + event_id are ignored at the inner daemon
/// boundary.
///
/// Per SPEC §8.2 line 789 + §8.10 line 900, the envelope `payload` field is a
/// structured Value (Map per SPEC §8.6 schema) — NOT an opaque Binary blob.
/// Callers use `titan_bus::client::extract_payload` to obtain the Value from
/// the envelope, then pass it here. Closure of the pre-2026-05-13 parity gap:
/// `rFP_worker_broadcast_topics_completion §4.C-ter` (2026-05-13).
///
/// Returns Err on schema drift (missing keys, wrong array lengths,
/// non-numeric multipliers). The inner daemon falls back to neutral
/// multipliers (all 1.0) on Err per defensive design.
pub fn decode_filter_down_payload(
    payload: &rmpv::Value,
) -> Result<InnerFilterDownPayload, DaemonError> {
    use rmpv::Value;
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("payload not a map".into())),
    };
    let mut multipliers: Option<&Value> = None;
    let mut epoch_id: i64 = 0;
    let mut ts: f64 = 0.0;
    for (k, val) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "multipliers" => multipliers = Some(val),
            "epoch_id" => {
                if let Some(n) = val.as_i64() {
                    epoch_id = n;
                }
            }
            "ts" => {
                if let Some(f) = val.as_f64() {
                    ts = f;
                }
            }
            _ => {}
        }
    }
    let mults_map = match multipliers {
        Some(Value::Map(items)) => items,
        _ => {
            return Err(DaemonError::MsgpackDecode(
                "multipliers field missing or not a map".into(),
            ))
        }
    };
    let mut inner_body = [1.0_f32; 5];
    let mut inner_mind = [1.0_f32; 15];
    let mut inner_spirit_content = [1.0_f32; 40];
    for (k, val) in mults_map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "inner_body" => decode_float_array_into(val, &mut inner_body)?,
            "inner_mind" => decode_float_array_into(val, &mut inner_mind)?,
            "inner_spirit_content" => decode_float_array_into(val, &mut inner_spirit_content)?,
            _ => {}
        }
    }
    Ok(InnerFilterDownPayload {
        inner_body,
        inner_mind,
        inner_spirit_content,
        epoch_id,
        ts,
    })
}

/// INNER_SPIRIT_FILTER_DOWN payload per SPEC §8.6 (Phase C addition).
/// Same shape as the inner slice of UNIFIED_SPIRIT_FILTER_DOWN, minus
/// epoch_id (LOCAL filter has no epoch counter).
#[derive(Debug, Clone)]
pub struct LocalFilterDownPayload {
    /// 5 multipliers for inner_body / outer_body.
    pub body: [f32; 5],
    /// 15 multipliers for inner_mind / outer_mind.
    pub mind: [f32; 15],
    /// 40 multipliers for inner_spirit_content / outer_spirit_content.
    pub spirit_content: [f32; 40],
    /// Wall-clock timestamp.
    pub ts: f64,
}

/// Decode an INNER_SPIRIT_FILTER_DOWN (or OUTER_SPIRIT_FILTER_DOWN) payload.
/// Used by inner-body + inner-mind daemons (and outer counterparts in C-S6).
///
/// Per SPEC §8.2 line 789 + §8.10 line 900: structured `rmpv::Value` payload
/// (Map per SPEC §8.6 schema). Closure of the pre-2026-05-13 parity gap:
/// `rFP_worker_broadcast_topics_completion §4.C-ter`.
pub fn decode_local_filter_down_payload(
    payload: &rmpv::Value,
) -> Result<LocalFilterDownPayload, DaemonError> {
    use rmpv::Value;
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("payload not a map".into())),
    };
    let mut multipliers: Option<&Value> = None;
    let mut ts: f64 = 0.0;
    for (k, val) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "multipliers" => multipliers = Some(val),
            "ts" => ts = val.as_f64().unwrap_or(0.0),
            _ => {}
        }
    }
    let mults_map = match multipliers {
        Some(Value::Map(items)) => items,
        _ => {
            return Err(DaemonError::MsgpackDecode(
                "multipliers field missing or not a map".into(),
            ))
        }
    };
    let mut body = [1.0_f32; 5];
    let mut mind = [1.0_f32; 15];
    let mut spirit_content = [1.0_f32; 40];
    for (k, val) in mults_map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "inner_body" | "body" => decode_float_array_into(val, &mut body)?,
            "inner_mind" | "mind" => decode_float_array_into(val, &mut mind)?,
            "inner_spirit_content" | "spirit_content" => {
                decode_float_array_into(val, &mut spirit_content)?
            }
            _ => {}
        }
    }
    Ok(LocalFilterDownPayload {
        body,
        mind,
        spirit_content,
        ts,
    })
}

fn decode_float_array_into<const N: usize>(
    v: &rmpv::Value,
    out: &mut [f32; N],
) -> Result<(), DaemonError> {
    let arr = match v {
        rmpv::Value::Array(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("not an array".into())),
    };
    if arr.len() != N {
        return Err(DaemonError::MsgpackDecode(format!(
            "array length mismatch: expected {N}, got {}",
            arr.len()
        )));
    }
    for (i, item) in arr.iter().enumerate() {
        out[i] = item
            .as_f64()
            .ok_or_else(|| DaemonError::MsgpackDecode(format!("array[{i}] not a float")))?
            as f32;
    }
    Ok(())
}

/// Build a UNIFIED_SPIRIT_FILTER_DOWN payload as `rmpv::Value::Map` for use
/// in tests + the e2e harness's stub publisher. Matches Python
/// `filter_down.py` publish shape per SPEC §8.6 + §8.10 byte-identical
/// guarantee.
///
/// Refactoring this signature would break the SPEC §8.6 wire-shape
/// contract that the test harness relies on. The 8 args mirror the
/// 6 tensor channels (inner+outer × body/mind/spirit) plus epoch_id +
/// ts metadata — they are NOT incidental complexity.
///
/// Pre-2026-05-13 this returned `Vec<u8>` (msgpack-encoded). Closure of
/// `rFP_worker_broadcast_topics_completion §4.C-ter`: now returns the
/// structured `Value::Map` so callers can pass it directly to
/// `BusClient::publish` or `encode_simple` per SPEC §8.2 line 789 +
/// §8.10 line 900.
#[allow(clippy::too_many_arguments)]
pub fn encode_filter_down_payload(
    inner_body: &[f32; 5],
    inner_mind: &[f32; 15],
    inner_spirit_content: &[f32; 40],
    outer_body: &[f32; 5],
    outer_mind: &[f32; 15],
    outer_spirit_content: &[f32; 40],
    epoch_id: i64,
    ts: f64,
) -> rmpv::Value {
    use rmpv::Value;

    fn arr<const N: usize>(v: &[f32; N]) -> Value {
        Value::Array(v.iter().map(|f| Value::F64(*f as f64)).collect())
    }

    let multipliers = Value::Map(vec![
        (Value::String("inner_body".into()), arr(inner_body)),
        (Value::String("inner_mind".into()), arr(inner_mind)),
        (
            Value::String("inner_spirit_content".into()),
            arr(inner_spirit_content),
        ),
        (Value::String("outer_body".into()), arr(outer_body)),
        (Value::String("outer_mind".into()), arr(outer_mind)),
        (
            Value::String("outer_spirit_content".into()),
            arr(outer_spirit_content),
        ),
    ]);

    Value::Map(vec![
        (Value::String("multipliers".into()), multipliers),
        (
            Value::String("epoch_id".into()),
            Value::Integer(epoch_id.into()),
        ),
        (Value::String("ts".into()), Value::F64(ts)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_lists_match_spec_9a() {
        // SPEC §9.A inner daemons: body / mind subscribe to 5; spirit to 5
        // (the original 3 + BODY_BALANCE_GIFT + MIND_BALANCE_GIFT per
        // P0.5 / D-SPEC-131 §G5.1 UP-leg gift).
        assert_eq!(INNER_BODY_TOPICS.len(), 5);
        assert_eq!(INNER_MIND_TOPICS.len(), 5);
        assert_eq!(INNER_SPIRIT_TOPICS.len(), 5);
        assert_eq!(OUTER_BODY_TOPICS.len(), 5);
        assert_eq!(OUTER_MIND_TOPICS.len(), 5);
        assert_eq!(OUTER_SPIRIT_TOPICS.len(), 5);
    }

    #[test]
    fn inner_spirit_does_not_subscribe_to_topology() {
        // G8 / Observer Principle: inner-spirit reads sibling slots
        // directly, not topology. SPEC §9.A explicitly omits
        // TRINITY_SUBSTRATE_TOPOLOGY_UPDATED from inner-spirit's REQUIRED.
        assert!(!INNER_SPIRIT_TOPICS.contains(&"TRINITY_SUBSTRATE_TOPOLOGY_UPDATED"));
        assert!(!INNER_SPIRIT_TOPICS.contains(&"INNER_SPIRIT_FILTER_DOWN"));
    }

    #[test]
    fn spirit_daemons_subscribe_to_balance_gifts() {
        // P0.5 / D-SPEC-131: spirit daemons receive both body + mind gifts
        // (sovereign-half filtered by payload `side` at decode time).
        assert!(INNER_SPIRIT_TOPICS.contains(&"BODY_BALANCE_GIFT"));
        assert!(INNER_SPIRIT_TOPICS.contains(&"MIND_BALANCE_GIFT"));
        assert!(OUTER_SPIRIT_TOPICS.contains(&"BODY_BALANCE_GIFT"));
        assert!(OUTER_SPIRIT_TOPICS.contains(&"MIND_BALANCE_GIFT"));
        // body/mind daemons do NOT subscribe to the gifts (they're publishers).
        assert!(!INNER_BODY_TOPICS.contains(&"BODY_BALANCE_GIFT"));
        assert!(!INNER_MIND_TOPICS.contains(&"MIND_BALANCE_GIFT"));
    }

    #[test]
    fn body_and_mind_subscribe_to_local_spirit_filter() {
        // INNER_SPIRIT_FILTER_DOWN cascades to inner_body + inner_mind.
        assert!(INNER_BODY_TOPICS.contains(&"INNER_SPIRIT_FILTER_DOWN"));
        assert!(INNER_MIND_TOPICS.contains(&"INNER_SPIRIT_FILTER_DOWN"));
        assert!(OUTER_BODY_TOPICS.contains(&"OUTER_SPIRIT_FILTER_DOWN"));
        assert!(OUTER_MIND_TOPICS.contains(&"OUTER_SPIRIT_FILTER_DOWN"));
    }

    #[test]
    fn shutdown_topic_present_in_all_daemons() {
        for topics in [
            INNER_BODY_TOPICS,
            INNER_MIND_TOPICS,
            INNER_SPIRIT_TOPICS,
            OUTER_BODY_TOPICS,
            OUTER_MIND_TOPICS,
            OUTER_SPIRIT_TOPICS,
        ] {
            assert!(
                topics.contains(&"KERNEL_SHUTDOWN_ANNOUNCE"),
                "every daemon must subscribe to KERNEL_SHUTDOWN_ANNOUNCE"
            );
        }
    }

    #[test]
    fn unified_filter_down_topic_present_in_all_daemons() {
        for topics in [
            INNER_BODY_TOPICS,
            INNER_MIND_TOPICS,
            INNER_SPIRIT_TOPICS,
            OUTER_BODY_TOPICS,
            OUTER_MIND_TOPICS,
            OUTER_SPIRIT_TOPICS,
        ] {
            assert!(topics.contains(&"UNIFIED_SPIRIT_FILTER_DOWN"));
        }
    }

    #[test]
    fn encode_decode_round_trip_inner_slice() {
        let inner_body = [0.5, 1.0, 1.5, 2.0, 2.5];
        let inner_mind: [f32; 15] = std::array::from_fn(|i| (i as f32) * 0.1 + 0.5);
        let inner_spirit_content: [f32; 40] = std::array::from_fn(|i| (i as f32) * 0.05 + 0.7);
        let outer_body = [1.0; 5];
        let outer_mind = [1.0; 15];
        let outer_spirit_content = [1.0; 40];

        let payload = encode_filter_down_payload(
            &inner_body,
            &inner_mind,
            &inner_spirit_content,
            &outer_body,
            &outer_mind,
            &outer_spirit_content,
            42,
            1234567890.5,
        );

        let decoded = decode_filter_down_payload(&payload).unwrap();
        for i in 0..5 {
            assert!((decoded.inner_body[i] - inner_body[i]).abs() < 1e-5);
        }
        for i in 0..15 {
            assert!((decoded.inner_mind[i] - inner_mind[i]).abs() < 1e-5);
        }
        for i in 0..40 {
            assert!((decoded.inner_spirit_content[i] - inner_spirit_content[i]).abs() < 1e-5);
        }
        assert_eq!(decoded.epoch_id, 42);
        assert!((decoded.ts - 1234567890.5).abs() < 1e-3);
    }

    #[test]
    fn decode_filter_down_rejects_non_map() {
        let v = rmpv::Value::String("not a map".into());
        let r = decode_filter_down_payload(&v);
        assert!(r.is_err());
    }

    #[test]
    fn decode_filter_down_rejects_wrong_array_length() {
        // multipliers.inner_body = [3 floats] instead of 5 → DimMismatch
        use rmpv::Value;
        let bad_inner_body = Value::Array(vec![Value::F64(0.5), Value::F64(0.5), Value::F64(0.5)]);
        let mults = Value::Map(vec![(Value::String("inner_body".into()), bad_inner_body)]);
        let payload = Value::Map(vec![(Value::String("multipliers".into()), mults)]);
        let r = decode_filter_down_payload(&payload);
        assert!(r.is_err());
    }

    #[test]
    fn decode_local_filter_down_round_trip() {
        // Encode an INNER_SPIRIT_FILTER_DOWN-shape payload (no epoch_id).
        use rmpv::Value;
        fn arr<const N: usize>(v: &[f32; N]) -> Value {
            Value::Array(v.iter().map(|f| Value::F64(*f as f64)).collect())
        }
        let body = [0.7_f32; 5];
        let mind: [f32; 15] = std::array::from_fn(|i| (i as f32) * 0.05);
        let spirit_content: [f32; 40] = std::array::from_fn(|i| (i as f32) * 0.02 + 0.5);
        let mults = Value::Map(vec![
            (Value::String("inner_body".into()), arr(&body)),
            (Value::String("inner_mind".into()), arr(&mind)),
            (
                Value::String("inner_spirit_content".into()),
                arr(&spirit_content),
            ),
        ]);
        let payload = Value::Map(vec![
            (Value::String("multipliers".into()), mults),
            (Value::String("ts".into()), Value::F64(99.0)),
        ]);

        let decoded = decode_local_filter_down_payload(&payload).unwrap();
        for i in 0..5 {
            assert!((decoded.body[i] - body[i]).abs() < 1e-5);
        }
        for i in 0..15 {
            assert!((decoded.mind[i] - mind[i]).abs() < 1e-5);
        }
        assert!((decoded.ts - 99.0).abs() < 1e-3);
    }
}
