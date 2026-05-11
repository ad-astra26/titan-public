//! tick_loop — inner-mind 23.49 Hz Schumann tick + state machine.
//!
//! Per SPEC §9.A `titan-inner-mind-rs` row + master plan §10.5 chunk C5-3.
//! G13 frequency: 7.83 Hz × 3 = 23.49 Hz (period 42.57 ms).
//! G10 ground_up scope: mind[10:15] WILLING ONLY — thinking[0:5] +
//! feeling[5:10] are NEVER grounded.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{INNER_MIND_FIRING_MAX_BYTES, INNER_MIND_FIRING_SCHEMA_VERSION};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_multipliers_default, decode_filter_down_payload,
    decode_local_filter_down_payload, encode_floats, ContentGate, DriftAggregator, FiringSlotWriter,
    GroundUpEnricher, Side, INNER_MIND_TOPICS,
};

const MIND_DIMS: usize = 15;
const SENSOR_CACHE_DIMS: usize = 15;
const DRIFT_THRESHOLD_PCT: f64 = 0.005;

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "inner-mind")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(INNER_MIND_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?INNER_MIND_TOPICS);

    let inner_mind_slot = open_slot(shm_dir, "inner_mind_15d.bin")?;
    let topology_slot = open_slot(shm_dir, "topology_30d.bin")?;
    // sensor_cache_inner_mind.bin: created by Python sidecar
    // (inner_mind_sensor_refresh in mind_worker), spawns LATER than this
    // Rust daemon — Option<Slot> at boot, retry-open in tick loop.
    // C-S5 closure 2026-05-08.
    let sensor_cache_path = shm_dir.join("sensor_cache_inner_mind.bin");
    let sensor_cache = Slot::open(&sensor_cache_path).ok();
    info!(event = "SHM_OPENED", sensor_cache_present = sensor_cache.is_some());

    // Phase C 130D dim-live tracker bridge (rFP §4.7).
    let firing_writer = FiringSlotWriter::new(
        "inner_mind",
        shm_dir,
        INNER_MIND_FIRING_SCHEMA_VERSION as u32,
        INNER_MIND_FIRING_MAX_BYTES as u32,
    );

    client
        .publish("MODULE_READY", Some("guardian"), None)
        .await
        .context("publish MODULE_READY")?;
    info!(event = "MODULE_READY_SENT");

    let state = Arc::new(Mutex::new(DaemonState::default()));
    let shutdown = Arc::new(Notify::new());
    let bus = Arc::new(client);

    let dispatcher_state = state.clone();
    let dispatcher_shutdown = shutdown.clone();
    let bus_for_dispatcher = bus.clone();
    let dispatcher = tokio::spawn(async move {
        run_event_dispatcher(bus_for_dispatcher, dispatcher_state, dispatcher_shutdown).await;
    });

    let tick_result = run_tick_loop(
        bus.clone(),
        state.clone(),
        shutdown.clone(),
        inner_mind_slot,
        topology_slot,
        sensor_cache,
        sensor_cache_path,
        firing_writer,
    )
    .await;

    shutdown.notify_waiters();
    let _ = dispatcher.await;
    bus.shutdown().await;

    tick_result
}

#[derive(Debug, Default)]
struct DaemonState {
    /// UNIFIED inner_mind multipliers (15D).
    unified: Option<[f32; 15]>,
    /// LOCAL inner_mind multipliers (15D, from INNER_SPIRIT_FILTER_DOWN).
    local: Option<[f32; 15]>,
    /// Topology fresh signal.
    topology_signaled: bool,
    /// Shutdown via KERNEL_SHUTDOWN_ANNOUNCE.
    shutdown_requested: bool,
}

async fn run_event_dispatcher(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
) {
    loop {
        tokio::select! {
            _ = shutdown.notified() => { debug!(event = "DISPATCHER_SHUTDOWN_NOTIFIED"); break; }
            event = bus.recv() => match event {
                Some(InboundEvent::Message { msg_type, raw_bytes, .. }) => {
                    handle_bus_message(&msg_type, &raw_bytes, &state);
                    if msg_type == "KERNEL_SHUTDOWN_ANNOUNCE" {
                        if let Ok(mut s) = state.lock() { s.shutdown_requested = true; }
                        shutdown.notify_waiters();
                        break;
                    }
                }
                Some(InboundEvent::Disconnected { reason }) => {
                    warn!(event = "BUS_DISCONNECTED", reason = %reason);
                    break;
                }
                None => {
                    warn!(event = "BUS_EVENT_CHANNEL_CLOSED");
                    break;
                }
            }
        }
    }
}

fn handle_bus_message(msg_type: &str, raw_bytes: &[u8], state: &Arc<Mutex<DaemonState>>) {
    match msg_type {
        "UNIFIED_SPIRIT_FILTER_DOWN" => {
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_filter_down_payload(&payload) {
                Ok(p) => {
                    if let Ok(mut s) = state.lock() {
                        s.unified = Some(p.inner_mind);
                    }
                }
                Err(e) => warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN failed"),
            }
        }
        "INNER_SPIRIT_FILTER_DOWN" => {
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_local_filter_down_payload(&payload) {
                Ok(p) => {
                    if let Ok(mut s) = state.lock() {
                        s.local = Some(p.mind);
                    }
                }
                Err(e) => warn!(err = ?e, "decode INNER_SPIRIT_FILTER_DOWN failed"),
            }
        }
        "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED" => {
            if let Ok(mut s) = state.lock() {
                s.topology_signaled = true;
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_tick_loop(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
    mut inner_mind_slot: Slot,
    topology_slot: Slot,
    mut sensor_cache: Option<Slot>,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
) -> Result<()> {
    let mut content_gate = ContentGate::new();
    let mut ground_up = GroundUpEnricher::new(Side::MindWilling);
    let mut drift_agg = DriftAggregator::new("mind", DRIFT_THRESHOLD_PCT);

    // Per master plan §7 + C-S3 PLAN §1.1 #2: Schumann timer wheels live in
    // titan-schumann (canonical shared library for trinity daemons).
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Mind, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    info!(event = "TICK_LOOP_START", role = "mind", period_ns);

    // Retry-open sensor cache every ~1s when None — Python sidecar
    // starts AFTER this Rust daemon. C-S5 closure 2026-05-08.
    let mut tick_count: u64 = 0;
    let retry_every_n: u64 = (1.0_f64 / 0.04258).ceil() as u64; // ~24 ticks ≈ 1s at 23.49 Hz

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                info!(event = "TICK_LOOP_SHUTDOWN_REQUESTED");
                break;
            }
            maybe_tick = tick_rx.recv() => match maybe_tick {
                Some(tick_event) => {
                    if sensor_cache.is_none() && tick_count.is_multiple_of(retry_every_n) {
                        if let Ok(slot) = Slot::open(&sensor_cache_path) {
                            info!(
                                event = "SENSOR_CACHE_OPENED_LATE",
                                path = %sensor_cache_path.display(),
                                tick = tick_count,
                            );
                            sensor_cache = Some(slot);
                        }
                    }
                    let drift_pct = tick_event.jitter_ns() as f64 / tick_event.period_ns as f64;
                    drift_agg.observe(drift_pct, tick_event.jitter_ns(), tick_event.epoch);
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut ground_up,
                        &mut inner_mind_slot, &topology_slot, &sensor_cache,
                        &mut firing_writer,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
                    }
                    tick_count = tick_count.wrapping_add(1);
                    if let Ok(s) = state.lock() {
                        if s.shutdown_requested {
                            info!(event = "SHUTDOWN_REQUESTED_VIA_BUS");
                            break;
                        }
                    }
                }
                None => {
                    info!(event = "TICK_CHANNEL_CLOSED");
                    break;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_one_tick(
    bus: &Arc<BusClient>,
    state: &Arc<Mutex<DaemonState>>,
    content_gate: &mut ContentGate,
    ground_up: &mut GroundUpEnricher,
    inner_mind_slot: &mut Slot,
    topology_slot: &Slot,
    sensor_cache: &Option<Slot>,
    firing_writer: &mut FiringSlotWriter,
) -> Result<()> {
    let mut mind = read_sensor_cache(sensor_cache)?;

    let (unified, local, topology_fresh) = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        (s.unified, s.local, s.topology_signaled)
    };

    // Compose UNIFIED ⊗ LOCAL → 15D mult vec.
    let composed: Vec<f32> = match (unified, local) {
        (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
        (Some(u), None) => u.to_vec(),
        (None, Some(l)) => l.to_vec(),
        (None, None) => vec![1.0_f32; MIND_DIMS],
    };

    apply_multipliers(&mut mind, &composed);

    // G10: ground_up applied to WILLING ONLY (mind[10:15]) — NOT thinking
    // (mind[0:5]) NOT feeling (mind[5:10]). Enforced by Side::MindWilling
    // which only modifies mind[10:15] in `apply_to_mind`.
    if topology_fresh {
        if let Ok(topology_lower) = titan_trinity_daemon::read_topology_inner_lower(topology_slot) {
            ground_up.apply_to_mind(&mut mind, &topology_lower, 1.0)?;
        }
    }

    let bytes = encode_floats::<MIND_DIMS>(&mind);

    if content_gate.should_write(&bytes) {
        inner_mind_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // Phase C dim-live tracker bridge (rFP §4.7) — always record per
    // tick (matches MIND_STATE bus publish cadence). inputs_state = &[]
    // until inner sensor_cache schema migrates to source-dict in P5.
    firing_writer.record_tick(&mind, &[], now_secs());

    let payload = encode_mind_state_payload(&mind);
    bus.publish("MIND_STATE", Some("all"), Some(&payload))
        .await
        .map_err(|e| anyhow!("publish MIND_STATE: {e}"))?;

    Ok(())
}

fn read_sensor_cache(sensor_cache: &Option<Slot>) -> Result<[f32; MIND_DIMS]> {
    match sensor_cache {
        Some(slot) => {
            let raw = slot.read().map_err(|e| anyhow!("sensor_cache read: {e}"))?;
            // Step 8 §4.5 schema migration v1→v2: msgpack source-dict
            // {"tensor": [v0..v14]} (Phase C — Python computes the 15D
            // via collect_mind_15d, publishes via msgpack; full Rust
            // per-dim formula port deferred to follow-up). Falls back to
            // legacy 15-float32-LE for backward-compat with pre-deploy.
            let is_msgpack = !raw.is_empty()
                && matches!(raw[0], 0x80..=0x8f | 0xde | 0xdf);
            if is_msgpack {
                decode_mind_source_dict(&raw).or_else(|_| Ok([0.5_f32; MIND_DIMS]))
            } else {
                let mut out = [0.0_f32; MIND_DIMS];
                for i in 0..SENSOR_CACHE_DIMS.min(raw.len() / 4) {
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&raw[i * 4..i * 4 + 4]);
                    out[i] = f32::from_le_bytes(buf);
                }
                Ok(out)
            }
        }
        None => Ok([0.0; MIND_DIMS]),
    }
}

/// Decode msgpack source dict {"tensor": [15 floats]} produced by
/// `mind_worker._provide_mind_source_dict` (Step 8 §4.5 schema v2).
fn decode_mind_source_dict(payload: &[u8]) -> Result<[f32; MIND_DIMS]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Err(anyhow!("source dict not a map")),
    };
    for (k, val) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("tensor") {
                if let Value::Array(items) = val {
                    let mut out = [0.5_f32; MIND_DIMS];
                    for (i, item) in items.iter().take(MIND_DIMS).enumerate() {
                        out[i] = item.as_f64().unwrap_or(0.5) as f32;
                    }
                    return Ok(out);
                }
            }
        }
    }
    Err(anyhow!("tensor key missing from source dict"))
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

fn encode_mind_state_payload(mind: &[f32; MIND_DIMS]) -> Vec<u8> {
    use rmpv::Value;
    let values = Value::Array(mind.iter().map(|f| Value::F64(*f as f64)).collect());
    let map = Value::Map(vec![
        (Value::String("src".into()), Value::String("inner".into())),
        (
            Value::String("type".into()),
            Value::String("MIND_STATE".into()),
        ),
        (Value::String("values".into()), values),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ]);
    let mut out = Vec::with_capacity(192);
    rmpv::encode::write_value(&mut out, &map)
        .expect("rmpv encode never fails on well-formed Value");
    out
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use titan_core::constants::INNER_MIND_15D_SCHEMA_VERSION;

    fn pure_compute(
        raw_mind: [f32; MIND_DIMS],
        unified: Option<[f32; MIND_DIMS]>,
        local: Option<[f32; MIND_DIMS]>,
        topology_lower: Option<[f32; 10]>,
        ground_up: &mut GroundUpEnricher,
    ) -> [f32; MIND_DIMS] {
        let mut mind = raw_mind;
        let composed: Vec<f32> = match (unified, local) {
            (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
            (Some(u), None) => u.to_vec(),
            (None, Some(l)) => l.to_vec(),
            (None, None) => vec![1.0_f32; MIND_DIMS],
        };
        apply_multipliers(&mut mind, &composed);
        if let Some(topo) = topology_lower {
            ground_up.apply_to_mind(&mut mind, &topo, 1.0).unwrap();
        }
        mind
    }

    #[test]
    fn schumann_role_mind_hz_is_locked() {
        // G13 LOCKED: body × 3 = 23.49 Hz. Sourced from titan-schumann
        // (canonical Schumann library per master plan §7).
        assert_eq!(SchumannRole::Mind.hz(), 23.49);
    }

    #[test]
    fn neutral_inputs_preserve_mind() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5; MIND_DIMS];
        let out = pure_compute(raw, None, None, None, &mut g);
        for i in 0..MIND_DIMS {
            assert!((out[i] - raw[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ground_up_only_touches_willing_10_15() {
        // G10 enforcement: thinking[0:5] + feeling[5:10] MUST be untouched
        // by ground_up; only willing[10:15] gets the nudge.
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5; MIND_DIMS];
        // Strong topology on the mind half (signal[5:10]).
        let topo: [f32; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04, 0.04];
        let out = pure_compute(raw, None, None, Some(topo), &mut g);
        // Thinking + feeling unchanged
        for i in 0..10 {
            assert!(
                (out[i] - 0.5).abs() < 1e-6,
                "mind[{i}] should be untouched, got {}",
                out[i]
            );
        }
        // Willing nudged: delta = 0.002 * 0.1 * 1.0 = 0.0002
        for i in 10..15 {
            assert!(
                (out[i] - 0.5002).abs() < 1e-5,
                "mind[{i}] should be grounded, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn unified_filter_applies_to_all_15_dims() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5; MIND_DIMS];
        let mut unified = [1.0_f32; MIND_DIMS];
        unified[0] = 2.0;
        unified[7] = 0.5;
        unified[14] = 1.5;
        let out = pure_compute(raw, Some(unified), None, None, &mut g);
        // 0.5 * 2.0 = 1.0 (clipped at 1.0); 0.5 * 0.5 = 0.25; 0.5 * 1.5 = 0.75
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[7] - 0.25).abs() < 1e-6);
        assert!((out[14] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn unified_local_compose_clamps_at_ceil() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5; MIND_DIMS];
        let mut unified = [1.0_f32; MIND_DIMS];
        let mut local = [1.0_f32; MIND_DIMS];
        unified[3] = 2.5;
        local[3] = 2.0; // 2.5 * 2.0 = 5.0 → clamp to MULTIPLIER_CEIL=3.0
        let out = pure_compute(raw, Some(unified), Some(local), None, &mut g);
        // body[3] = 0.5 * 3.0 = 1.5 → clamp to TENSOR_MAX=1.0
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mind_payload_msgpack_shape() {
        let mind = [0.0_f32; MIND_DIMS];
        let bytes = encode_mind_state_payload(&mind);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        let map = match v {
            Value::Map(items) => items,
            _ => panic!("not a map"),
        };
        let keys: Vec<String> = map
            .iter()
            .filter_map(|(k, _)| {
                if let Value::String(s) = k {
                    s.as_str().map(String::from)
                } else {
                    None
                }
            })
            .collect();
        assert!(keys.contains(&"src".to_string()));
        assert!(keys.contains(&"type".to_string()));
        assert!(keys.contains(&"values".to_string()));
        assert!(keys.contains(&"ts".to_string()));
    }

    #[test]
    fn mind_payload_src_is_inner_and_type_mind_state() {
        let mind = [0.0_f32; MIND_DIMS];
        let bytes = encode_mind_state_payload(&mind);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        let mut found_src = false;
        let mut found_type = false;
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("src") {
                        assert_eq!(val, Value::String("inner".into()));
                        found_src = true;
                    } else if s.as_str() == Some("type") {
                        assert_eq!(val, Value::String("MIND_STATE".into()));
                        found_type = true;
                    }
                }
            }
        }
        assert!(found_src && found_type);
    }

    #[test]
    fn mind_payload_values_array_15d() {
        let mind: [f32; MIND_DIMS] = std::array::from_fn(|i| (i as f32) * 0.05);
        let bytes = encode_mind_state_payload(&mind);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("values") {
                        if let Value::Array(arr) = val {
                            assert_eq!(arr.len(), MIND_DIMS);
                            return;
                        }
                    }
                }
            }
        }
        panic!("values array of length 15 not found");
    }

    #[test]
    fn slot_write_15d_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inner_mind_15d.bin");
        let mut slot = Slot::create(&path, INNER_MIND_15D_SCHEMA_VERSION as u32, 60).unwrap();
        let mind: [f32; MIND_DIMS] = std::array::from_fn(|i| (i as f32) * 0.05);
        let bytes = encode_floats::<MIND_DIMS>(&mind);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn read_sensor_cache_handles_none() {
        let r = read_sensor_cache(&None).unwrap();
        assert_eq!(r, [0.0; MIND_DIMS]);
    }

    #[test]
    fn read_sensor_cache_decodes_15_floats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sensor_cache_inner_mind.bin");
        let mut slot = Slot::create(&path, 1, 60).unwrap();
        let raw: [f32; MIND_DIMS] = std::array::from_fn(|i| (i as f32) * 0.07);
        let bytes = encode_floats::<MIND_DIMS>(&raw);
        slot.write(&bytes).unwrap();
        let r = read_sensor_cache(&Some(slot)).unwrap();
        for i in 0..MIND_DIMS {
            assert!((r[i] - raw[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn content_hash_gates_redundant_writes() {
        let mut gate = ContentGate::new();
        let mind: [f32; MIND_DIMS] = std::array::from_fn(|i| (i as f32) * 0.05);
        let bytes = encode_floats::<MIND_DIMS>(&mind);
        assert!(gate.should_write(&bytes));
        assert!(!gate.should_write(&bytes));
        assert_eq!(gate.write_count(), 1);
        assert_eq!(gate.suppress_count(), 1);
    }

    #[test]
    fn payload_size_15d_under_msgpack_cap() {
        // 15 floats msgpack encoded should be well under FRAME_MAX_FRAME_BYTES.
        let mind: [f32; MIND_DIMS] = std::array::from_fn(|i| (i as f32) * 0.07);
        let bytes = encode_mind_state_payload(&mind);
        assert!(
            bytes.len() < 1024,
            "MIND_STATE payload bloated: {}B",
            bytes.len()
        );
    }
}
