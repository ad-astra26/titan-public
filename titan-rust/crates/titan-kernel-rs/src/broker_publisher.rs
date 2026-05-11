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

/// Build the KERNEL_EPOCH_TICK msgpack payload using the same encoder
/// `titan_bus::message::encode_simple` uses, plus extra event fields.
fn encode_epoch_tick_payload(event: &PiTickEvent) -> Result<Vec<u8>, String> {
    use rmpv::Value as MpValue;

    let entries = vec![
        (
            MpValue::String("type".into()),
            MpValue::String("KERNEL_EPOCH_TICK".into()),
        ),
        (
            MpValue::String("src".into()),
            MpValue::String("kernel".into()),
        ),
        (MpValue::String("dst".into()), MpValue::String("all".into())),
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
    ];
    let mut out = Vec::with_capacity(96);
    rmpv::encode::write_value(&mut out, &MpValue::Map(entries))
        .map_err(|e| format!("rmpv: {e:?}"))?;
    Ok(out)
}

// `message::decode_header` referenced for documentation/test cross-check;
// silence unused-import warning when not invoked outside tests.
#[allow(dead_code)]
fn _silence_unused_message_import() {
    let _ = message::decode_header(&[]);
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
