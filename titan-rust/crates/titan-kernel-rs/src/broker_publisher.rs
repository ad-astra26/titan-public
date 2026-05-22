//! broker_publisher — `BrokerEpochPublisher` impl that publishes
//! `KERNEL_EPOCH_TICK` via the in-process `BusBroker::publish_local` path.
//!
//! Per SPEC §8.1 (`KERNEL_EPOCH_TICK` is P0, never drop) + §10.A B6 step
//! (kernel publishes ~3 Hz). Used by `titan-clocks::run_pi_heartbeat_loop`
//! via the `EpochTickPublisher` trait.

use std::sync::Arc;

use titan_bus::{message, BusBroker};
use titan_clocks::{EpochTickPublisher, PiTickEvent};
use tokio::runtime::Handle;

/// Publishes `KERNEL_EPOCH_TICK` to the kernel-owned broker.
pub struct BrokerEpochPublisher {
    broker: Arc<BusBroker>,
    runtime: Handle,
}

impl BrokerEpochPublisher {
    /// Construct against a shared broker handle. `runtime` is the tokio
    /// `Handle` used to bridge the sync `EpochTickPublisher::publish()` call
    /// to the broker's async `publish_local()`.
    pub fn new(broker: Arc<BusBroker>, runtime: Handle) -> Self {
        Self { broker, runtime }
    }
}

impl EpochTickPublisher for BrokerEpochPublisher {
    fn publish(&self, event: &PiTickEvent) -> Result<(), String> {
        // KERNEL_EPOCH_TICK payload per SPEC §8.1:
        // `{epoch_id: u64, ts: f64, dt_s: f64}`.
        // We additionally include phase + pulse_count for observability;
        // older subscribers ignore the extra fields (msgpack is forward-compat).
        let payload = encode_epoch_tick_payload(event).map_err(|e| format!("encode: {e}"))?;
        let broker = self.broker.clone();

        // Dispatch on the kernel's tokio runtime. The clocks loop calls
        // `publish()` from inside an async context, so we use `Handle::block_on`
        // only when we're NOT already on a runtime thread (best-effort).
        let runtime = self.runtime.clone();
        runtime.spawn(async move {
            broker
                .publish_local("KERNEL_EPOCH_TICK", "kernel", payload)
                .await;
        });

        Ok(())
    }
}

/// Build the KERNEL_EPOCH_TICK msgpack envelope per SPEC §8.1 + §8.10.
///
/// Envelope structure (SPEC §8.10 line 900): `{type, src?, dst?, payload?}`
/// where `payload` is a nested msgpack Map per the per-message schema.
/// For KERNEL_EPOCH_TICK (SPEC §8.1 line 1057) the payload schema is
/// `{epoch_id: int, ts: float, dt_s: float}` plus the extra observability
/// fields `pulse_count` + `phase` (forward-compatible; subscribers that
/// don't know these fields ignore them).
///
/// Pre-2026-05-18 the publisher placed all event fields at the top level
/// alongside `type`/`src`/`dst` (no nested `payload` map). Python
/// subscribers reading `msg.get("payload", {}).get("epoch_id", 0)` saw
/// `0` on every tick because the `payload` key was absent. Closes
/// BUG-MEDITATION-NEVER-TRIGGERS-SINCE-PHASE-C-MIGRATION-20260518:
/// `meditation_worker._emergent_check` computed `epoch_gap = 0`, so the
/// first-boot fallback `count == 0 and epoch_gap > _med_min_epochs`
/// never fired → fleet-wide ZERO MEDITATION_REQUEST emissions since
/// 2026-05-14 Phase C migration. Same envelope bug also affected
/// timechain_worker / language_worker / cognitive_worker `payload.phase`
/// reads. Aligns publisher with the canonical `encode_simple` encoder
/// that already produces SPEC-correct envelopes for every other Rust
/// bus emit site (fixed for `BUS_SUBSCRIBE` in 2026-05-13 per
/// `rFP_worker_broadcast_topics_completion §4.C-ter`; this is the
/// parallel fix for KERNEL_EPOCH_TICK).
fn encode_epoch_tick_payload(event: &PiTickEvent) -> Result<Vec<u8>, String> {
    use rmpv::Value as MpValue;

    let payload_map = MpValue::Map(vec![
        (
            MpValue::String("epoch_id".into()),
            MpValue::Integer(event.epoch_id.into()),
        ),
        (
            MpValue::String("pulse_count".into()),
            MpValue::Integer(event.pulse_count.into()),
        ),
        (MpValue::String("phase".into()), MpValue::F32(event.phase)),
        (MpValue::String("ts".into()), MpValue::F64(event.ts)),
        (MpValue::String("dt_s".into()), MpValue::F64(event.dt_s)),
    ]);
    message::encode_simple(
        "KERNEL_EPOCH_TICK",
        Some("kernel"),
        Some("all"),
        Some(payload_map),
    )
    .map_err(|e| format!("encode_simple: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmpv::Value as MpValue;

    #[test]
    fn encode_payload_round_trips_via_decode_header() {
        let event = PiTickEvent {
            epoch_id: 42,
            pulse_count: 100,
            phase: 0.5,
            ts: 1714408800.123,
            dt_s: 0.333,
        };
        let bytes = encode_epoch_tick_payload(&event).unwrap();
        let hdr = message::decode_header(&bytes).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("KERNEL_EPOCH_TICK"));
        assert_eq!(hdr.src.as_deref(), Some("kernel"));
        assert_eq!(hdr.dst.as_deref(), Some("all"));
    }

    /// Regression for BUG-MEDITATION-NEVER-TRIGGERS-SINCE-PHASE-C-MIGRATION-20260518.
    /// Envelope MUST nest `epoch_id` / `pulse_count` / `phase` / `ts` / `dt_s` inside
    /// `payload` per SPEC §8.1 + §8.10. Python subscribers read these via
    /// `msg.get("payload", {}).get("epoch_id", 0)`; a top-level placement returns 0.
    #[test]
    fn envelope_nests_event_fields_inside_payload_map_per_spec_8_1() {
        let event = PiTickEvent {
            epoch_id: 12345,
            pulse_count: 67890,
            phase: 0.75,
            ts: 1779000000.5,
            dt_s: 0.333,
        };
        let bytes = encode_epoch_tick_payload(&event).unwrap();
        let decoded: MpValue =
            rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes[..])).unwrap();
        let top_map = match decoded {
            MpValue::Map(m) => m,
            other => panic!("envelope must be a Map, got {other:?}"),
        };

        let top_keys: Vec<&str> = top_map
            .iter()
            .filter_map(|(k, _)| {
                if let MpValue::String(s) = k {
                    s.as_str()
                } else {
                    None
                }
            })
            .collect();
        for envelope_key in ["type", "src", "dst", "payload"] {
            assert!(
                top_keys.contains(&envelope_key),
                "envelope missing required key `{envelope_key}` (have {top_keys:?})"
            );
        }
        for event_field in ["epoch_id", "pulse_count", "phase", "ts", "dt_s"] {
            assert!(
                !top_keys.contains(&event_field),
                "event field `{event_field}` must NOT be at envelope top level — \
                 belongs inside `payload` per SPEC §8.1 + §8.10. Top-level placement \
                 reintroduces the BUG-MEDITATION-NEVER-TRIGGERS-SINCE-PHASE-C-MIGRATION \
                 envelope-shape regression."
            );
        }

        let payload_map = top_map
            .into_iter()
            .find_map(|(k, v)| {
                if matches!(k, MpValue::String(ref s) if s.as_str() == Some("payload")) {
                    Some(v)
                } else {
                    None
                }
            })
            .expect("payload key checked above");
        let payload_entries = match payload_map {
            MpValue::Map(m) => m,
            other => panic!("payload must be a Map per §8.10, got {other:?}"),
        };

        let mut got_epoch_id: Option<u64> = None;
        let mut got_pulse_count: Option<u64> = None;
        let mut got_phase: Option<f32> = None;
        let mut got_ts: Option<f64> = None;
        let mut got_dt_s: Option<f64> = None;
        for (k, v) in payload_entries {
            if let MpValue::String(key_s) = k {
                match key_s.as_str() {
                    Some("epoch_id") => {
                        if let MpValue::Integer(i) = v {
                            got_epoch_id = i.as_u64();
                        }
                    }
                    Some("pulse_count") => {
                        if let MpValue::Integer(i) = v {
                            got_pulse_count = i.as_u64();
                        }
                    }
                    Some("phase") => match v {
                        MpValue::F32(f) => got_phase = Some(f),
                        MpValue::F64(f) => got_phase = Some(f as f32),
                        _ => {}
                    },
                    Some("ts") => match v {
                        MpValue::F64(f) => got_ts = Some(f),
                        MpValue::F32(f) => got_ts = Some(f as f64),
                        _ => {}
                    },
                    Some("dt_s") => match v {
                        MpValue::F64(f) => got_dt_s = Some(f),
                        MpValue::F32(f) => got_dt_s = Some(f as f64),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        assert_eq!(got_epoch_id, Some(event.epoch_id), "payload.epoch_id");
        assert_eq!(
            got_pulse_count,
            Some(event.pulse_count),
            "payload.pulse_count"
        );
        assert_eq!(got_phase, Some(event.phase), "payload.phase");
        assert_eq!(got_ts, Some(event.ts), "payload.ts");
        assert_eq!(got_dt_s, Some(event.dt_s), "payload.dt_s");
    }
}
