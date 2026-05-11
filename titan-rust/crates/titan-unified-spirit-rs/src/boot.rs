//! boot — Connect to bus, subscribe to required topics, wire SPHERE_PULSE
//! → ResonanceDetector dispatch loop.
//!
//! Per SPEC §10.A boot ordering for unified-spirit:
//! - T+60ms: connect to main bus + subscribe (REQUIRED:
//!   `TRINITY_SUBSTRATE_TOPOLOGY_UPDATED`, `SPHERE_PULSE`,
//!   `KERNEL_SHUTDOWN_ANNOUNCE`, `SUPERVISION_CHILD_DOWN`).
//! - T+60ms onward: read shm slots in body_cycle_loop (C4-2 already
//!   shipped); spawn 6 trinity daemons (C4-4); first SELF assembly +
//!   write within 200ms.
//!
//! C-S4 chunks deliver this sequence in pieces. C4-2b1 (this file)
//! wires bus_client connection + SPHERE_PULSE → ResonanceDetector. The
//! body_cycle_loop already runs (per C4-2). Daemon supervision lands
//! in C4-4; substrate handshake hardening lands in C4-5.
//!
//! Key dispatch logic:
//! - Receive SPHERE_PULSE envelope → extract `clock_name` + `phase` +
//!   `ts` from msgpack payload → call
//!   `ResonanceDetector::record_pulse_with_phase(component, phase, ts)`.
//! - On BIG PULSE return value: log + (when great_pulse_ready)
//!   trigger `UnifiedSpirit::advance()` via callback (wired in C4-2b2).
//! - Receive KERNEL_SHUTDOWN_ANNOUNCE → set shutdown flag → graceful
//!   exit propagates per SPEC §10.A + §15.

use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use crate::resonance::{BigPulse, ResonanceDetector};
use crate::unified_spirit::{ResonanceSnapshot, UnifiedSpirit};
use titan_bus::client::{extract_payload, BusClient, InboundEvent};

/// Topics the unified-spirit-rs binary subscribes to per SPEC §9.A
/// REQUIRED set + Decision Log D-SPEC-26 (SPHERE_PULSE added in
/// SPEC v0.1.3).
pub const REQUIRED_SUBSCRIPTIONS: [&str; 4] = [
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
    "SPHERE_PULSE",
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "SUPERVISION_CHILD_DOWN",
];

/// Optional subscriptions per SPEC §9.A (don't fail boot if absent).
pub const OPTIONAL_SUBSCRIPTIONS: [&str; 1] = ["SWAP_SUBTREE_REQUEST"];

/// Decode a SPHERE_PULSE payload msgpack into `(clock_name, phase, ts)`.
///
/// Per SPEC §8.6 row: `{clock_name: str, pulse_count: int, phase: float, ts: float}`.
/// We only need `clock_name`, `phase`, `ts` for ResonanceDetector.
///
/// Returns `None` on schema mismatch — caller logs + drops.
pub fn decode_sphere_pulse(envelope_bytes: &[u8]) -> Option<SpherePulseFields> {
    // The payload is a msgpack-binary blob nested inside the outer envelope.
    // Try direct envelope decode first — if the broker forwards the inner
    // map directly (no nested binary), we get the fields off the outer map.
    let raw_map = decode_top_level_map(envelope_bytes)?;
    if let Some(fields) = pluck_pulse_fields(&raw_map) {
        return Some(fields);
    }
    // Fallback: payload is a msgpack-binary inside the outer map.
    let inner_bytes = extract_payload(envelope_bytes)?;
    let inner_map = decode_top_level_map(&inner_bytes)?;
    pluck_pulse_fields(&inner_map)
}

/// Decoded SPHERE_PULSE fields handed to ResonanceDetector.
#[derive(Debug, Clone, PartialEq)]
pub struct SpherePulseFields {
    /// Clock canonical name — `"inner_body"`, `"outer_mind"`, etc. Maps
    /// to a pair via `resonance::component_to_pair`.
    pub clock_name: String,
    /// Phase (radians).
    pub phase: f64,
    /// Wall clock seconds (f64) of the pulse.
    pub ts: f64,
}

fn decode_top_level_map(bytes: &[u8]) -> Option<Vec<(rmpv::Value, rmpv::Value)>> {
    let value: rmpv::Value = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes)).ok()?;
    if let rmpv::Value::Map(entries) = value {
        Some(entries)
    } else {
        None
    }
}

fn pluck_pulse_fields(entries: &[(rmpv::Value, rmpv::Value)]) -> Option<SpherePulseFields> {
    let mut clock_name: Option<String> = None;
    let mut phase: Option<f64> = None;
    let mut ts: Option<f64> = None;
    for (k, v) in entries {
        let key = match k {
            rmpv::Value::String(s) => s.as_str()?,
            _ => continue,
        };
        match key {
            "clock_name" => {
                if let rmpv::Value::String(s) = v {
                    clock_name = s.as_str().map(|x| x.to_string());
                }
            }
            "phase" => {
                phase = v.as_f64();
            }
            "ts" => {
                ts = v.as_f64();
            }
            _ => {}
        }
    }
    Some(SpherePulseFields {
        clock_name: clock_name?,
        phase: phase?,
        ts: ts?,
    })
}

/// Build the canonical `on_big_pulse` callback that bridges
/// `ResonanceDetector::record_pulse_with_phase` BIG PULSE emissions to
/// `UnifiedSpirit::advance()` per SPEC §10.F G11–G12 lifecycle.
///
/// When `BigPulse::great_pulse_ready == true` (all 3 pairs simultaneously
/// resonant), invokes `UnifiedSpirit::advance(ResonanceSnapshot)` —
/// which crystallizes a GreatEpoch + auto-persists state.
///
/// Used by main.rs to wire the dispatch loop:
/// ```ignore
/// let detector = Arc::new(Mutex::new(ResonanceDetector::with_defaults(&data_dir)));
/// let spirit = Arc::new(Mutex::new(UnifiedSpirit::with_defaults(&data_dir)?));
/// let on_big_pulse = build_advance_callback(spirit.clone(), detector.clone());
/// run_bus_dispatch_loop(client, detector, on_big_pulse, shutdown).await;
/// ```
pub fn build_advance_callback(
    spirit: Arc<Mutex<UnifiedSpirit>>,
    detector: Arc<Mutex<ResonanceDetector>>,
) -> impl Fn(BigPulse) + Send + Sync + 'static {
    move |big_pulse: BigPulse| {
        if !big_pulse.great_pulse_ready {
            // BIG PULSE on a single pair — only log, don't advance.
            debug!(
                event = "BIG_PULSE",
                pair = %big_pulse.pair,
                count = big_pulse.big_pulse_count,
                "BIG PULSE on single pair — not yet GREAT"
            );
            return;
        }

        let snapshot = build_resonance_snapshot(&detector, &big_pulse);
        let mut spirit_guard = spirit.lock();
        match spirit_guard.advance(snapshot) {
            Ok(epoch) => {
                info!(
                    event = "GREAT_PULSE_ADVANCED",
                    epoch_id = epoch.epoch_id,
                    velocity = epoch.velocity,
                    magnitude = epoch.magnitude,
                    great_pulse_count = big_pulse.great_pulse_count,
                    "GREAT PULSE → UnifiedSpirit advanced + persisted"
                );
            }
            Err(e) => {
                warn!(
                    event = "GREAT_PULSE_ADVANCE_FAIL",
                    err = ?e,
                    "advance() failed — escalation pending"
                );
            }
        }
    }
}

/// Build a `ResonanceSnapshot` from the detector + the triggering BigPulse.
fn build_resonance_snapshot(
    detector: &Arc<Mutex<ResonanceDetector>>,
    big_pulse: &BigPulse,
) -> ResonanceSnapshot {
    let det = detector.lock();
    let mut pair_big_pulse_counts = std::collections::BTreeMap::new();
    for &name in crate::resonance::PAIRS.iter() {
        if let Some(state) = det.to_state().pairs.get(name) {
            pair_big_pulse_counts.insert(name.to_string(), state.big_pulse_count);
        }
    }
    ResonanceSnapshot {
        great_pulse_count: big_pulse.great_pulse_count,
        pair_big_pulse_counts,
        ts: big_pulse.ts,
    }
}

/// Run the bus dispatch loop. Returns when the recv loop ends (broker
/// closed) OR `KERNEL_SHUTDOWN_ANNOUNCE` is received OR the caller's
/// shutdown flag is set externally.
///
/// `on_big_pulse` is invoked synchronously when ResonanceDetector emits
/// a BIG PULSE. Caller wires this via [`build_advance_callback`] to
/// `UnifiedSpirit::advance()`.
pub async fn run_bus_dispatch_loop(
    client: Arc<BusClient>,
    detector: Arc<Mutex<ResonanceDetector>>,
    on_big_pulse: impl Fn(BigPulse) + Send + Sync + 'static,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
) {
    info!(
        event = "BUS_DISPATCH_LOOP_START",
        "bus dispatch loop running"
    );

    loop {
        if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
            info!(
                event = "BUS_DISPATCH_LOOP_STOP",
                reason = "shutdown_flag",
                "stopping"
            );
            break;
        }

        let event = match client.recv().await {
            Some(e) => e,
            None => {
                info!(
                    event = "BUS_DISPATCH_LOOP_STOP",
                    reason = "channel_closed",
                    "stopping"
                );
                break;
            }
        };

        match event {
            InboundEvent::Message {
                msg_type,
                raw_bytes,
                ..
            } => match msg_type.as_str() {
                "SPHERE_PULSE" => match decode_sphere_pulse(&raw_bytes) {
                    Some(fields) => {
                        let mut det = detector.lock();
                        if let Some(big_pulse) =
                            det.record_pulse_with_phase(&fields.clock_name, fields.phase, fields.ts)
                        {
                            // BIG PULSE fired. If great_pulse_ready, the
                            // detector already incremented its counter.
                            on_big_pulse(big_pulse);
                        }
                    }
                    None => {
                        warn!(
                            event = "SPHERE_PULSE_DECODE_FAIL",
                            "could not decode SPHERE_PULSE payload"
                        );
                    }
                },
                "KERNEL_SHUTDOWN_ANNOUNCE" => {
                    info!(
                        event = "KERNEL_SHUTDOWN_RECEIVED",
                        "received KERNEL_SHUTDOWN_ANNOUNCE; setting shutdown flag"
                    );
                    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED" => {
                    debug!(
                        event = "TOPOLOGY_UPDATED",
                        "topology_30d.bin freshness signal received"
                    );
                    // Body cycle loop reads topology slot directly per C4-2;
                    // optional early-wake landing in C4-3c (debounce per
                    // BODY_CYCLE_DEBOUNCE_MS).
                }
                "SUPERVISION_CHILD_DOWN" => {
                    info!(
                        event = "CHILD_DOWN_RECEIVED",
                        "supervisor will respawn child (C4-4 wires this)"
                    );
                }
                other => {
                    debug!(
                        event = "BUS_MSG_UNHANDLED",
                        msg_type = other,
                        "received message — no handler in C4-2b1"
                    );
                }
            },
            InboundEvent::Disconnected { reason } => {
                warn!(
                    event = "BUS_DISCONNECTED",
                    reason = %reason,
                    "broker connection closed; exiting dispatch loop"
                );
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_bus::message::encode_simple;

    fn encode_sphere_pulse(clock_name: &str, phase: f64, ts: f64) -> Vec<u8> {
        let entries = vec![
            (
                rmpv::Value::String("clock_name".into()),
                rmpv::Value::String(clock_name.into()),
            ),
            (rmpv::Value::String("phase".into()), rmpv::Value::F64(phase)),
            (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
        ];
        let payload_bytes = {
            let mut buf = Vec::new();
            rmpv::encode::write_value(&mut buf, &rmpv::Value::Map(entries)).unwrap();
            buf
        };
        encode_simple(
            "SPHERE_PULSE",
            Some("titan-trinity-rs"),
            Some("all"),
            Some(&payload_bytes),
        )
        .unwrap()
    }

    #[test]
    fn decode_sphere_pulse_extracts_fields_from_inner_map() {
        // C4-2b1 boot test 1: round-trip SPHERE_PULSE encode → decode
        let envelope = encode_sphere_pulse("inner_body", 0.5, 1234567890.5);
        let fields = decode_sphere_pulse(&envelope).unwrap();
        assert_eq!(fields.clock_name, "inner_body");
        assert!((fields.phase - 0.5).abs() < 1e-10);
        assert!((fields.ts - 1234567890.5).abs() < 1e-3);
    }

    #[test]
    fn decode_sphere_pulse_handles_outer_map_directly() {
        // C4-2b1 boot test 2: when broker forwards the pulse fields
        // at the OUTER map level (no nested payload binary), decode works.
        let entries = vec![
            (
                rmpv::Value::String("type".into()),
                rmpv::Value::String("SPHERE_PULSE".into()),
            ),
            (
                rmpv::Value::String("clock_name".into()),
                rmpv::Value::String("outer_spirit".into()),
            ),
            (rmpv::Value::String("phase".into()), rmpv::Value::F64(1.5)),
            (rmpv::Value::String("ts".into()), rmpv::Value::F64(99.0)),
        ];
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &rmpv::Value::Map(entries)).unwrap();
        let fields = decode_sphere_pulse(&buf).unwrap();
        assert_eq!(fields.clock_name, "outer_spirit");
        assert!((fields.phase - 1.5).abs() < 1e-10);
    }

    #[test]
    fn decode_sphere_pulse_returns_none_on_missing_fields() {
        // C4-2b1 boot test 3: malformed payload (missing required field)
        let entries = vec![(
            rmpv::Value::String("clock_name".into()),
            rmpv::Value::String("inner_body".into()),
        )];
        // missing "phase" + "ts"
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &rmpv::Value::Map(entries)).unwrap();
        let envelope = encode_simple("SPHERE_PULSE", None, None, Some(&buf)).unwrap();
        assert!(decode_sphere_pulse(&envelope).is_none());
    }

    #[test]
    fn required_subscriptions_includes_sphere_pulse() {
        // C4-2b1 boot test 4: SPEC §9.A unified-spirit-rs row + D-SPEC-26
        // mandates SPHERE_PULSE REQUIRED.
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"SPHERE_PULSE"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"TRINITY_SUBSTRATE_TOPOLOGY_UPDATED"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"KERNEL_SHUTDOWN_ANNOUNCE"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"SUPERVISION_CHILD_DOWN"));
        assert_eq!(REQUIRED_SUBSCRIPTIONS.len(), 4);
    }

    #[test]
    fn build_advance_callback_invokes_advance_on_great_pulse() {
        // C4-2b2 integration test: when BigPulse.great_pulse_ready=true,
        // build_advance_callback's closure invokes UnifiedSpirit::advance.
        use crate::unified_spirit::UnifiedSpirit;
        let dir = tempfile::tempdir().unwrap();

        let detector = Arc::new(Mutex::new(ResonanceDetector::with_defaults(dir.path())));
        let spirit = Arc::new(Mutex::new(
            UnifiedSpirit::with_defaults(dir.path()).unwrap_or_else(|_| {
                UnifiedSpirit::new(
                    crate::unified_spirit::UnifiedSpiritConfig::default(),
                    dir.path(),
                )
                .unwrap()
            }),
        ));
        let cb = build_advance_callback(spirit.clone(), detector.clone());

        // BIG PULSE on a single pair — should NOT advance
        let bp_single = BigPulse {
            pair: "body".into(),
            big_pulse_count: 1,
            phase_diff: 0.1,
            time_diff: 1.0,
            inner_pulse_count: 3,
            outer_pulse_count: 3,
            total_resonant_cycles: 3,
            ts: 100.0,
            great_pulse_ready: false,
            great_pulse_count: 0,
        };
        cb(bp_single);
        assert_eq!(
            spirit.lock().epoch_count(),
            0,
            "single-pair BIG PULSE should NOT advance"
        );

        // BIG PULSE with great_pulse_ready=true — SHOULD advance
        let bp_great = BigPulse {
            pair: "spirit".into(),
            big_pulse_count: 1,
            phase_diff: 0.1,
            time_diff: 1.0,
            inner_pulse_count: 3,
            outer_pulse_count: 3,
            total_resonant_cycles: 3,
            ts: 200.0,
            great_pulse_ready: true,
            great_pulse_count: 1,
        };
        cb(bp_great);
        assert_eq!(
            spirit.lock().epoch_count(),
            1,
            "great_pulse_ready=true should invoke advance"
        );
        // Verify ResonanceSnapshot fields propagated correctly
        let latest = spirit.lock();
        let epoch = latest.latest_epoch().unwrap();
        assert_eq!(epoch.resonance_snapshot.great_pulse_count, 1);
        assert!((epoch.resonance_snapshot.ts - 200.0).abs() < 1e-6);
    }
}
