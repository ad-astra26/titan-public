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
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use crate::resonance::{BigPulse, ResonanceDetector};
use crate::unified_spirit::{ResonanceSnapshot, UnifiedSpirit};
use titan_bus::client::{extract_payload, BusClient, InboundEvent};

/// Topics the unified-spirit-rs binary subscribes to per SPEC §9.A.
///
/// `SPHERE_PULSE` is NO LONGER a bus subscription (D-SPEC-117): the
/// resonance detector now reads `sphere_clocks.bin` SHM-direct per
/// Preamble G18 (state via SHM, not bus). The bus `SPHERE_PULSE` event is
/// retained for observers (observatory_worker → dashboard WS) but the
/// detector derives pulse + sustained-balance from the SHM slot the
/// substrate already writes — eliminating the fragile broker-delivery path
/// that left the detector starved fleet-wide (see
/// `project_sphere_pulse_not_reaching_broker_freeze_20260522`).
pub const REQUIRED_SUBSCRIPTIONS: [&str; 3] = [
    "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
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
    // SPEC §8.6 publisher emits SPHERE_PULSE as `{clock_name, phase, ts}`
    // wrapped in the envelope's `payload` field (nested Map per SPEC §8.2 +
    // §8.10 byte-identical wire-format guarantee). Try direct envelope decode
    // first — if a publisher happens to flatten the fields at the outer-map
    // level (legacy / test path), we can still extract them. Otherwise pluck
    // them from the nested payload Map.
    let raw_map = decode_top_level_map(envelope_bytes)?;
    if let Some(fields) = pluck_pulse_fields(&raw_map) {
        return Some(fields);
    }
    // Canonical path: payload is a nested Map per §4.C-ter wire-format closure.
    let payload = extract_payload(envelope_bytes)?;
    let inner_map = match payload {
        rmpv::Value::Map(entries) => entries,
        _ => return None,
    };
    pluck_pulse_fields(&inner_map)
}

/// Decoded SPHERE_PULSE fields handed to ResonanceDetector.
#[derive(Debug, Clone, PartialEq)]
pub struct SpherePulseFields {
    /// Clock canonical name — `"inner_body"`, `"outer_mind"`, etc. Maps
    /// to a pair via `resonance::component_to_pair`.
    pub clock_name: String,
    /// Phase (radians). Retained for `/v4/resonance` telemetry; the §G11
    /// gate now uses `balanced` (D-SPEC-112), not phase.
    pub phase: f64,
    /// Wall clock seconds (f64) of the pulse.
    pub ts: f64,
    /// Whether this pulse fired in a balanced-coherence regime (§G11
    /// balance-coincidence gate, D-SPEC-112).
    pub balanced: bool,
    /// Consecutive balanced ticks at pulse time (sphere_clock counter).
    /// Feeds the §G11 sustained-balance harmony gate (D-SPEC-113): a side
    /// counts as harmonious only after `HARMONY_TICKS` sustained balanced
    /// ticks, debouncing coherence flicker around the 0.70 threshold.
    /// Absent on the wire → 0 (safe-fail: not-yet-sustained).
    pub consecutive_balanced: u32,
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
    let mut balanced: bool = false; // §G11 balance-coincidence (D-SPEC-112); absent → false (safe-fail)
    let mut consecutive_balanced: u32 = 0; // §G11 sustained-balance (D-SPEC-113); absent → 0 (safe-fail)
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
            "balanced" => {
                balanced = v.as_bool().unwrap_or(false);
            }
            "consecutive_balanced" => {
                consecutive_balanced = v.as_u64().unwrap_or(0) as u32;
            }
            _ => {}
        }
    }
    Some(SpherePulseFields {
        clock_name: clock_name?,
        phase: phase?,
        ts: ts?,
        balanced,
        consecutive_balanced,
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
/// On a successful advance the callback also sends the `great_pulse_count`
/// over `great_pulse_tx`. The publisher task watches this channel and
/// computes + publishes `UNIFIED_SPIRIT_FILTER_DOWN` ONCE per GREAT pulse
/// — the unified filter_down is a GREAT-gated EVENT, never per-tick state
/// (SPEC §G5.1 / D-SPEC-96 D4 closure).
///
/// Used by main.rs to wire the dispatch loop:
/// ```ignore
/// let detector = Arc::new(Mutex::new(ResonanceDetector::with_defaults(&data_dir)));
/// let spirit = Arc::new(Mutex::new(UnifiedSpirit::with_defaults(&data_dir)?));
/// let (great_tx, great_rx) = tokio::sync::watch::channel(0u64);
/// let on_big_pulse = build_advance_callback(spirit.clone(), detector.clone(), great_tx);
/// run_bus_dispatch_loop(client, detector, on_big_pulse, shutdown).await;
/// ```
pub fn build_advance_callback(
    spirit: Arc<Mutex<UnifiedSpirit>>,
    detector: Arc<Mutex<ResonanceDetector>>,
    great_pulse_tx: watch::Sender<u64>,
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
                // §G5.1 / D4: signal the publisher to compute + publish the
                // GREAT-gated unified filter_down for this pulse. send()
                // only errs if every receiver dropped (publisher gone) —
                // benign during shutdown.
                let _ = great_pulse_tx.send(big_pulse.great_pulse_count);
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
            InboundEvent::Message { msg_type, .. } => match msg_type.as_str() {
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

/// Canonical sphere-clock order in `sphere_clocks.bin` (SPEC §7.1) — must
/// match `titan-trinity-rs::sphere_clocks::ClockRole::all()`.
const SPHERE_CLOCK_NAMES: [&str; 6] = [
    "inner_body",
    "inner_mind",
    "inner_spirit",
    "outer_body",
    "outer_mind",
    "outer_spirit",
];

/// Per-clock field count + byte layout in `sphere_clocks.bin` (SPEC §7.1):
/// 6 clocks × 7 f32 LE. Field indices: [2]=phase, [4]=pulse_count,
/// [5]=consecutive_balanced, [6]=last_pulse_age_s.
const SPHERE_CLOCK_FIELDS: usize = 7;
const SPHERE_CLOCKS_MIN_BYTES: usize = 6 * SPHERE_CLOCK_FIELDS * 4;

/// SHM-direct resonance feed (D-SPEC-117, Preamble G18). Polls
/// `sphere_clocks.bin` at `cadence_ms` and feeds the `ResonanceDetector`
/// from the substrate's canonical clock state — replacing the retired
/// `SPHERE_PULSE` bus subscription whose broker delivery left the detector
/// starved fleet-wide.
///
/// A pulse is detected when a clock's `pulse_count` increments since the
/// last poll (monotonic counter — no missed pulses). `consecutive_balanced`
/// feeds the §G11 sustained-balance harmony gate (D-SPEC-114). `phase` is
/// passed through for `/v4/resonance` telemetry only.
pub async fn run_sphere_clock_poll_loop(
    shm_dir: std::path::PathBuf,
    cadence_ms: u64,
    detector: Arc<Mutex<ResonanceDetector>>,
    on_big_pulse: impl Fn(BigPulse) + Send + Sync + 'static,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
) {
    let path = shm_dir.join("sphere_clocks.bin");
    let slot = match titan_state::Slot::open(&path) {
        Ok(s) => s,
        Err(e) => {
            error!(
                event = "SPHERE_POLL_OPEN_FAIL",
                err = ?e,
                path = ?path,
                "could not open sphere_clocks.bin; resonance detector will NOT be fed"
            );
            return;
        }
    };

    // u32::MAX sentinel = uninitialised; the first poll seeds last-seen
    // counts WITHOUT firing pulses (avoids a spurious burst at boot).
    let mut last_pulse_counts = [u32::MAX; 6];
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(cadence_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    info!(
        event = "SPHERE_POLL_LOOP_START",
        cadence_ms, "SHM-direct resonance feed running (G18 / D-SPEC-117)"
    );

    loop {
        if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
            info!(
                event = "SPHERE_POLL_LOOP_STOP",
                reason = "shutdown_flag",
                "stopping"
            );
            break;
        }
        interval.tick().await;

        let payload = match slot.read() {
            Ok(p) => p,
            Err(_) => continue, // transient SeqLock contention — retry next tick
        };
        if payload.len() < SPHERE_CLOCKS_MIN_BYTES {
            continue; // slot not yet written by the substrate
        }

        let now = systime_unix_secs();
        for (i, name) in SPHERE_CLOCK_NAMES.iter().enumerate() {
            let base = i * SPHERE_CLOCK_FIELDS * 4;
            let read_f32 = |fi: usize| -> f32 {
                let off = base + fi * 4;
                f32::from_le_bytes([
                    payload[off],
                    payload[off + 1],
                    payload[off + 2],
                    payload[off + 3],
                ])
            };
            let phase = read_f32(2) as f64;
            let pulse_count = read_f32(4) as u32;
            let consecutive_balanced = read_f32(5) as u32;
            let last_pulse_age_s = read_f32(6) as f64;

            let prev = last_pulse_counts[i];
            last_pulse_counts[i] = pulse_count;
            if prev == u32::MAX || pulse_count <= prev {
                continue; // first poll (seed) or no new pulse on this clock
            }

            // A pulse fired since the last poll. Per §G11 D-SPEC-122 (v1.55.0,
            // reverts D-SPEC-114), the per-pulse balanced flag is what gates
            // the 3-consecutive-coincidence streak; `consecutive_balanced` is
            // no longer in the gate (preserved as a per-clock telemetry field).
            let balanced = consecutive_balanced > 0;
            let pulse_ts = now - last_pulse_age_s;
            let maybe_big = {
                let mut det = detector.lock();
                det.record_pulse_with_phase(name, phase, balanced, pulse_ts)
            };
            if let Some(big_pulse) = maybe_big {
                on_big_pulse(big_pulse);
            }
        }
    }
}

/// Wall-clock seconds since UNIX epoch (mirror of `time.time()`).
fn systime_unix_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_bus::message::encode_simple;

    fn encode_sphere_pulse(clock_name: &str, phase: f64, ts: f64) -> Vec<u8> {
        let payload = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("clock_name".into()),
                rmpv::Value::String(clock_name.into()),
            ),
            (rmpv::Value::String("phase".into()), rmpv::Value::F64(phase)),
            (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
        ]);
        encode_simple(
            "SPHERE_PULSE",
            Some("titan-trinity-rs"),
            Some("all"),
            Some(payload),
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
        // Decode the buf back to Value so we can pass as structured payload
        let payload: rmpv::Value =
            rmpv::decode::read_value(&mut std::io::Cursor::new(&buf[..])).unwrap();
        let envelope = encode_simple("SPHERE_PULSE", None, None, Some(payload)).unwrap();
        assert!(decode_sphere_pulse(&envelope).is_none());
    }

    #[test]
    fn required_subscriptions_excludes_sphere_pulse_d_spec_117() {
        // D-SPEC-117: SPHERE_PULSE is NO LONGER a bus subscription — the
        // detector reads sphere_clocks.bin SHM-direct (G18). The 3 control
        // topics remain.
        assert!(!REQUIRED_SUBSCRIPTIONS.contains(&"SPHERE_PULSE"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"TRINITY_SUBSTRATE_TOPOLOGY_UPDATED"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"KERNEL_SHUTDOWN_ANNOUNCE"));
        assert!(REQUIRED_SUBSCRIPTIONS.contains(&"SUPERVISION_CHILD_DOWN"));
        assert_eq!(REQUIRED_SUBSCRIPTIONS.len(), 3);
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
        let (great_tx, great_rx) = watch::channel(0u64);
        let cb = build_advance_callback(spirit.clone(), detector.clone(), great_tx);

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
        assert_eq!(
            *great_rx.borrow(),
            0,
            "single-pair BIG PULSE should NOT signal the filter_down publisher"
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
        // §G5.1 / D4: GREAT advance must signal the publisher with the count
        assert_eq!(
            *great_rx.borrow(),
            1,
            "great_pulse_ready=true should signal the filter_down publisher with the GREAT count"
        );
        // Verify ResonanceSnapshot fields propagated correctly
        let latest = spirit.lock();
        let epoch = latest.latest_epoch().unwrap();
        assert_eq!(epoch.resonance_snapshot.great_pulse_count, 1);
        assert!((epoch.resonance_snapshot.ts - 200.0).abs() < 1e-6);
    }
}
