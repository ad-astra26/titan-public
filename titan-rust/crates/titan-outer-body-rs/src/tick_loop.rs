//! tick_loop — outer-body Schumann-locked tick (7.83 Hz) + state machine.
//!
//! Per SPEC §9.A `titan-outer-body-rs` row + §18.1 outer cadence
//! resolution + master plan §10.6 chunk C6-3. Logic flow per tick:
//!
//! 1. Read `sensor_cache_outer_body.bin` → msgpack-decode source dict
//!    written by Python sidecar `outer_body_sensor_refresh.py`. Project
//!    to a 5D base body vector via the V6 body-felt formula
//!    (interoception / proprioception / somatosensation / entropy /
//!    thermal — port of `outer_trinity.py:_collect_outer_body`). On
//!    cache miss / stale (per SPEC §18.1 `wall_ns < now − 3 × cadence`),
//!    fall back to last-known + emit confidence=0.0 log.
//! 2. Read `topology_30d.bin[0:10]` → outer_lower 10D (when last
//!    TRINITY_SUBSTRATE_TOPOLOGY_UPDATED was received).
//! 3. Apply UNIFIED + LOCAL filter_down (multipliers from
//!    UNIFIED_SPIRIT_FILTER_DOWN.outer_body[5] +
//!    OUTER_SPIRIT_FILTER_DOWN.body[5] bus events; G7 [0.3, 3.0] clamp).
//! 4. Apply ground_up nudge to all 5D per G10 (body all dims grounded).
//! 5. Content-hash gate: if payload differs from previous tick, write
//!    `outer_body_5d.bin` via SeqLock + publish BODY_STATE (src=outer).

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{
    OUTER_BODY_BUS_PUBLISH_INTERVAL_S, OUTER_BODY_FIRING_MAX_BYTES,
    OUTER_BODY_FIRING_SCHEMA_VERSION, OUTER_BODY_TICK_BASE_S,
};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_focus_into_enrichment, compose_multipliers_default,
    decode_local_filter_down_payload, encode_body_balance_gift, encode_floats,
    load_checkpoint_for_part, load_restoring_cfg, observe, open_focus_input_if_present,
    open_neuromod_slot_if_present, read_focus_nudge, read_neuromod_gain, read_sensor_cache,
    read_topology_outer_lower, stateful_update, write_checkpoint_for_part, BalancedPulseEdges,
    CheckpointSnapshot, ContentGate, FiringSlotWriter, FocusPart, GroundUpEnricher,
    JourneyAccumulator, JourneyTickInputs, Layer, PublishThrottle, PulseClockRole, PulseWatcher,
    RestoringCfg, SensorCacheRead, Side, TrinitySide, BODY_BALANCE_GIFT_TOPIC, BODY_GIFT_WEIGHTS,
    OUTER_BODY_TOPICS,
};

/// §G5.2 item 4 checkpoint cadence (outer-body @ 7.83 Hz → ~10s).
const CHECKPOINT_WRITE_EVERY_N_TICKS: u64 = 80;
const CHECKPOINT_PART: &str = "outer_body";

/// Boot the daemon's runtime + drive the tick loop until SIGTERM /
/// disconnect.
pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path, data_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "outer-body")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(OUTER_BODY_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?OUTER_BODY_TOPICS);

    let outer_body_slot = open_slot(shm_dir, "outer_body_5d.bin")?;
    let topology_slot = open_slot(shm_dir, "topology_30d.bin")?;
    let sensor_cache_path = shm_dir.join("sensor_cache_outer_body.bin");
    info!(
        event = "SHM_OPENED",
        topology_present = topology_slot.path().exists(),
        sensor_cache_path = ?sensor_cache_path,
    );

    // Phase C 130D dim-live tracker bridge (rFP §4.7).
    let firing_writer = FiringSlotWriter::new(
        "outer_body",
        shm_dir,
        OUTER_BODY_FIRING_SCHEMA_VERSION as u32,
        OUTER_BODY_FIRING_MAX_BYTES as u32,
    );

    client
        .publish("MODULE_READY", Some("guardian"), None)
        .await
        .context("publish MODULE_READY")?;
    info!(event = "MODULE_READY_SENT");

    let state = Arc::new(Mutex::new(DaemonState::default()));
    let shutdown = Arc::new(Notify::new());

    let dispatcher_state = state.clone();
    let dispatcher_shutdown = shutdown.clone();
    let bus = Arc::new(client);
    let bus_for_dispatcher = bus.clone();
    let dispatcher = tokio::spawn(async move {
        run_event_dispatcher(bus_for_dispatcher, dispatcher_state, dispatcher_shutdown).await;
    });

    let tick_result = run_tick_loop(
        bus.clone(),
        state.clone(),
        shutdown.clone(),
        outer_body_slot,
        topology_slot,
        sensor_cache_path,
        firing_writer,
        shm_dir.to_path_buf(),
        data_dir.to_path_buf(),
    )
    .await;

    shutdown.notify_waiters();
    let _ = dispatcher.await;
    bus.shutdown().await;

    tick_result
}

/// Per-daemon mutable state — protected by Mutex; tick loop + dispatcher
/// both touch it.
#[derive(Debug, Default)]
struct DaemonState {
    /// Most recent UNIFIED_SPIRIT_FILTER_DOWN.outer_body multipliers.
    unified: Option<[f32; 5]>,
    /// Most recent OUTER_SPIRIT_FILTER_DOWN.body multipliers (LOCAL).
    local: Option<[f32; 5]>,
    /// Most recent topology_lower from TRINITY_SUBSTRATE_TOPOLOGY_UPDATED.
    topology_signaled: bool,
    /// Set true on each KERNEL_EPOCH_TICK; consumed by the tick loop to
    /// recompute the ground_up held nudge once per epoch (SPEC §G5.1 / 0E).
    epoch_pending: bool,
    /// Set true when KERNEL_SHUTDOWN_ANNOUNCE arrives.
    shutdown_requested: bool,
    /// Last successfully-computed 5D body vector — fall-back when sensor
    /// cache is stale per SPEC §18.1.
    last_body: [f32; 5],
}

async fn run_event_dispatcher(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
) {
    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                debug!(event = "DISPATCHER_SHUTDOWN_NOTIFIED");
                break;
            }
            event = bus.recv() => match event {
                Some(InboundEvent::Message { msg_type, raw_bytes, .. }) => {
                    handle_bus_message(&msg_type, &raw_bytes, &state);
                    if msg_type == "KERNEL_SHUTDOWN_ANNOUNCE" {
                        if let Ok(mut s) = state.lock() {
                            s.shutdown_requested = true;
                        }
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
            match decode_unified_outer_body(&payload) {
                Ok(mults) => {
                    if let Ok(mut s) = state.lock() {
                        s.unified = Some(mults);
                    }
                }
                Err(e) => warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN.outer_body failed"),
            }
        }
        "OUTER_SPIRIT_FILTER_DOWN" => {
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_local_filter_down_payload(&payload) {
                Ok(p) => {
                    if let Ok(mut s) = state.lock() {
                        s.local = Some(p.body);
                    }
                }
                Err(e) => warn!(err = ?e, "decode OUTER_SPIRIT_FILTER_DOWN failed"),
            }
        }
        "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED" => {
            if let Ok(mut s) = state.lock() {
                s.topology_signaled = true;
            }
        }
        "KERNEL_EPOCH_TICK" => {
            // 0E: mark an epoch boundary — tick loop recomputes the held
            // ground_up nudge once per epoch (SPEC §G5.1).
            if let Ok(mut s) = state.lock() {
                s.epoch_pending = true;
            }
        }
        _ => {}
    }
}

/// Decode `UNIFIED_SPIRIT_FILTER_DOWN.multipliers.outer_body[5]` from a
/// structured payload. Sibling of `decode_filter_down_payload` in
/// titan-trinity-daemon::subscriptions which only extracts inner fields;
/// outer daemons need the outer slice from the same message. SPEC §8.2
/// line 789 + §8.10 line 900: payload is `rmpv::Value::Map` per §8.6 schema.
/// Closure of `rFP_worker_broadcast_topics_completion §4.C-ter` (2026-05-13).
fn decode_unified_outer_body(payload: &rmpv::Value) -> Result<[f32; 5]> {
    use rmpv::Value;
    let map = match payload {
        Value::Map(items) => items,
        _ => return Err(anyhow!("payload not a map")),
    };
    let mut multipliers: Option<&Value> = None;
    for (k, val) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("multipliers") {
                multipliers = Some(val);
                break;
            }
        }
    }
    let mults_map = match multipliers {
        Some(Value::Map(items)) => items,
        _ => return Err(anyhow!("multipliers field missing or not a map")),
    };
    let mut outer_body = [1.0_f32; 5];
    let mut found = false;
    for (k, val) in mults_map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("outer_body") {
                decode_float_array_into(val, &mut outer_body)?;
                found = true;
                break;
            }
        }
    }
    if !found {
        return Err(anyhow!("multipliers.outer_body missing"));
    }
    Ok(outer_body)
}

fn decode_float_array_into(val: &rmpv::Value, out: &mut [f32]) -> Result<()> {
    use rmpv::Value;
    let arr = match val {
        Value::Array(a) => a,
        _ => return Err(anyhow!("not an array")),
    };
    if arr.len() != out.len() {
        return Err(anyhow!(
            "expected {} elements, got {}",
            out.len(),
            arr.len()
        ));
    }
    for (i, v) in arr.iter().enumerate() {
        out[i] = v
            .as_f64()
            .ok_or_else(|| anyhow!("element {i} not numeric"))? as f32;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_tick_loop(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
    mut outer_body_slot: Slot,
    topology_slot: Slot,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
    shm_dir: std::path::PathBuf,
    data_dir: std::path::PathBuf,
) -> Result<()> {
    // Post-A.S8 D2 cadence migration (rFP §4.2): Schumann body (7.83 Hz)
    // tick + bus publish throttled to OUTER_BODY_BUS_PUBLISH_INTERVAL_S.
    // Body-slowest G13 invariant: this throttle (45s) > mind (15s) > spirit (5s).
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    let mut content_gate = ContentGate::new();
    let mut ground_up = GroundUpEnricher::new(Side::Body);
    let mut publish_throttle = PublishThrottle::new(OUTER_BODY_BUS_PUBLISH_INTERVAL_S);
    // §G5.2 item 4 — restore exact tensor + observable state from checkpoint
    // on boot; cold-start at 0.5 only when sidecar absent/invalid.
    let (mut prev, mut prev2, mut last_obs_restored) =
        match load_checkpoint_for_part::<5>(&data_dir, CHECKPOINT_PART) {
            Some(CheckpointSnapshot {
                prev,
                prev2,
                last_obs,
                ..
            }) => (prev, prev2, Some(last_obs)),
            None => ([0.5_f32; 5], [0.5_f32; 5], None),
        };
    // §G5.2 item 5 + item 2: per-Titan gains + live neuromod-gain.
    let mut cfg = load_restoring_cfg(&shm_dir, Layer::Body);
    let mut neuromod_slot = open_neuromod_slot_if_present(&shm_dir);
    let neuromod_path = shm_dir.join("neuromod_state.bin");
    let mut focus_input_slot = open_focus_input_if_present(&shm_dir);
    let focus_input_path = shm_dir.join("focus_input.bin");
    // P0.5 / D-SPEC-131 §G5.1 UP-leg gift state (outer mirror of inner-body).
    let mut pulse_watcher = PulseWatcher::open(&shm_dir);
    let mut journey_acc: JourneyAccumulator<5> = JourneyAccumulator::new();
    let mut tick_count: u64 = 0;
    // Outer-body @ 7.83 Hz: ~8 ticks ≈ 1s — refresh cfg + retry neuromod open at ~1s.
    let retry_every_n: u64 = (1.0_f64 / 0.1277).ceil() as u64;

    info!(
        event = "TICK_LOOP_START",
        role = "outer-body",
        period_ns,
        publish_interval_s = OUTER_BODY_BUS_PUBLISH_INTERVAL_S,
    );

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                info!(event = "TICK_LOOP_SHUTDOWN_REQUESTED");
                break;
            }
            maybe_tick = tick_rx.recv() => match maybe_tick {
                Some(tick_event) => {
                    debug!(
                        event = "OUTER_BODY_TICK",
                        epoch = tick_event.epoch,
                        period_ns = tick_event.period_ns,
                    );
                    if neuromod_slot.is_none() && tick_count.is_multiple_of(retry_every_n) {
                        if let Ok(slot) = Slot::open(&neuromod_path) {
                            info!(
                                event = "NEUROMOD_STATE_OPENED_LATE",
                                path = %neuromod_path.display(),
                                tick = tick_count,
                            );
                            neuromod_slot = Some(slot);
                        }
                    }
                    if focus_input_slot.is_none() && tick_count.is_multiple_of(retry_every_n) {
                        if let Ok(slot) = Slot::open(&focus_input_path) {
                            info!(
                                event = "FOCUS_INPUT_OPENED_LATE",
                                path = %focus_input_path.display(),
                                tick = tick_count,
                            );
                            focus_input_slot = Some(slot);
                        }
                    }
                    if !pulse_watcher.is_open() && tick_count.is_multiple_of(retry_every_n) {
                        pulse_watcher.retry_open(&shm_dir);
                        if pulse_watcher.is_open() {
                            info!(event = "PULSE_WATCH_OPENED_LATE", tick = tick_count);
                        }
                    }
                    if tick_count.is_multiple_of(retry_every_n) {
                        cfg = load_restoring_cfg(&shm_dir, Layer::Body);
                    }
                    let (_pulse_edges, balanced_pulse_edges) = pulse_watcher.tick_with_balanced();
                    tick_count = tick_count.wrapping_add(1);
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut ground_up, &mut publish_throttle,
                        &mut outer_body_slot, &topology_slot, &sensor_cache_path,
                        &mut firing_writer, &mut prev, &mut prev2,
                        &mut cfg, neuromod_slot.as_ref(), focus_input_slot.as_ref(),
                        &mut last_obs_restored,
                        &mut journey_acc, &balanced_pulse_edges,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
                    }
                    // §G5.2 item 4 — periodic checkpoint write.
                    if tick_count.is_multiple_of(CHECKPOINT_WRITE_EVERY_N_TICKS) {
                        if let Some(o) = last_obs_restored.as_ref() {
                            if let Err(e) = write_checkpoint_for_part::<5>(
                                &data_dir,
                                CHECKPOINT_PART,
                                &prev,
                                &prev2,
                                o,
                            ) {
                                warn!(err = ?e, "checkpoint write failed (continuing)");
                            }
                        }
                    }
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
    if let Some(o) = last_obs_restored.as_ref() {
        if let Err(e) = write_checkpoint_for_part::<5>(&data_dir, CHECKPOINT_PART, &prev, &prev2, o)
        {
            warn!(err = ?e, "final checkpoint write failed");
        }
    }
    Ok(())
}

/// Stale threshold per SPEC §18.1 = 3 × natural_cadence (= 30s for outer-body).
fn outer_body_stale_threshold_s() -> f64 {
    OUTER_BODY_TICK_BASE_S * 3.0
}

#[allow(clippy::too_many_arguments)]
async fn run_one_tick(
    bus: &Arc<BusClient>,
    state: &Arc<Mutex<DaemonState>>,
    content_gate: &mut ContentGate,
    ground_up: &mut GroundUpEnricher,
    publish_throttle: &mut PublishThrottle,
    outer_body_slot: &mut Slot,
    topology_slot: &Slot,
    sensor_cache_path: &Path,
    firing_writer: &mut FiringSlotWriter,
    prev: &mut [f32; 5],
    prev2: &mut [f32; 5],
    cfg: &mut RestoringCfg,
    neuromod_slot: Option<&Slot>,
    focus_input_slot: Option<&Slot>,
    last_obs_restored: &mut Option<titan_trinity_daemon::LayerObs>,
    journey_acc: &mut JourneyAccumulator<5>,
    balanced_pulse_edges: &BalancedPulseEdges,
) -> Result<()> {
    // 1. Read sensor cache (msgpack source dict from Python sidecar) +
    //    project to 5D body vector. On stale or missing, use last-known.
    //    This is raw[t] per §G5.2 — the un-enriched producer output.
    let last_body = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.last_body
    };
    let raw = read_outer_body_from_cache(sensor_cache_path, last_body);
    let mut body = raw;

    // 2. Snapshot bus state (+ consume the epoch-pending edge for 0E).
    let (unified_mult, local_mult, topology_fresh, epoch_due) = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        let epoch_due = s.epoch_pending;
        s.epoch_pending = false;
        // D-SPEC-121 (v1.54.0): one-shot consume-and-clear (see inner-body).
        (
            s.unified.take(),
            s.local.take(),
            s.topology_signaled,
            epoch_due,
        )
    };

    // 3. Compose multipliers (UNIFIED ⊗ LOCAL when both present).
    let composed = match (unified_mult, local_mult) {
        (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
        (Some(u), None) => u.to_vec(),
        (None, Some(l)) => l.to_vec(),
        (None, None) => vec![1.0_f32; 5],
    };

    // 4. Apply filter_down multipliers (G7 clamp [0.3, 3.0] enforced by lib).
    apply_multipliers(&mut body, &composed);

    // 5. ground_up — 0E held-nudge model (SPEC §G5.1, D-SPEC-97 refinement):
    //    RECOMPUTE the damped nudge ONCE per kernel epoch (so the 0.95 EMA
    //    evolves at epoch cadence), then APPLY the held nudge every tick.
    //    Per G10 ground_up_body_range=0:5 — body all 5D grounded.
    if topology_fresh {
        if epoch_due {
            if let Ok(topology_lower) = read_topology_outer_lower(topology_slot) {
                ground_up.compute_nudge(&topology_lower);
            }
        }
        ground_up.apply_held_to_body(&mut body, 1.0)?;
    }

    // 6. Cache fresh body for next-tick fallback.
    if let Ok(mut s) = state.lock() {
        s.last_body = body;
    }

    // 6b. §G5.2 traveling-tensor update (Layer::Body). `last_body` (step 6) keeps
    //     the producer/drive value for stale fallback; the traveling state `x` is
    //     what is written/published.
    //     enrichment = (enriched − raw) — filter_down + ground_up delta, applied
    //     as a SEPARATE full-weight additive term per §G5.2 equation. Spring is
    //     modulated by live neuromod gain (§G5.2 item 2).
    let mut enrichment = [0.0_f32; 5];
    for i in 0..5 {
        enrichment[i] = body[i] - raw[i];
    }
    // §G12 FOCUS cascade: amplified nudge composes into enrichment_force.
    let focus = read_focus_nudge::<5>(focus_input_slot, FocusPart::OuterBody);
    compose_focus_into_enrichment(&mut enrichment, &focus);
    cfg.neuromod_gain = read_neuromod_gain(neuromod_slot);
    let obs = observe(&prev[..], &prev2[..]);
    *last_obs_restored = Some(obs);
    let x = stateful_update(&prev[..], &prev2[..], &raw[..], &enrichment[..], &obs, cfg);
    let mut body_state = [0.0_f32; 5];
    body_state.copy_from_slice(&x[..5]);
    *prev2 = *prev;
    *prev = body_state;
    let body = body_state;

    // 6c. P0.5 / D-SPEC-131 §G5.1 UP-leg balance gift — outer mirror of inner.
    let tick_ts = now_secs();
    journey_acc.tick(JourneyTickInputs {
        x: &body,
        obs,
        now_secs: tick_ts as f32,
    });
    if balanced_pulse_edges[PulseClockRole::OuterBody.index()] {
        journey_acc.mark_balanced(obs);
        if let Some(digest) = journey_acc.finalize_body_gift(&BODY_GIFT_WEIGHTS) {
            let payload = encode_body_balance_gift::<5>(TrinitySide::Outer, &digest, tick_ts);
            if let Err(e) = bus
                .publish(BODY_BALANCE_GIFT_TOPIC, Some("all"), Some(payload))
                .await
            {
                warn!(err = ?e, "publish BODY_BALANCE_GIFT failed (continuing)");
            } else {
                debug!(
                    event = "BODY_BALANCE_GIFT_EMITTED",
                    side = "outer",
                    amplitude = digest.gift_amplitude,
                    cycle_s = digest.cycle_duration_s,
                    ticks = digest.cycle_tick_count,
                );
            }
        }
        journey_acc.reset_for_next_cycle();
    }

    // 7. Encode + content-hash gate the slot write.
    let bytes = encode_floats::<5>(&body);
    if content_gate.should_write(&bytes) {
        outer_body_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // 7b. Phase C dim-live tracker bridge (rFP §4.7) — record per tick
    //     at full Schumann body cadence (independent of bus publish
    //     throttle). dim-live needs the per-tick firing signal even
    //     though BODY_STATE bus publish is throttled to every ~45s.
    firing_writer.record_tick(&body, &[], now_secs());

    // 8. Bus publish throttled per OUTER_BODY_BUS_PUBLISH_INTERVAL_S (post-A.S8
    //    D2 — rFP §4.2). Tick fires at Schumann body (7.83 Hz) but
    //    BODY_STATE publishes only every ~45s. Slot writes (above) remain
    //    at full tick cadence under content-hash gating.
    if publish_throttle.should_publish() {
        let payload = encode_body_state_payload(&body);
        bus.publish("BODY_STATE", Some("all"), Some(payload))
            .await
            .map_err(|e| anyhow!("publish BODY_STATE: {e}"))?;
    }

    Ok(())
}

/// Read the outer_body sensor cache + project msgpack source dict to 5D.
///
/// Cold boot / stale fallback: returns `last_body` (last successful
/// compute). On Python sidecar source-dict shape parsing failure,
/// returns `last_body` and logs WARN (rate-limited by tracing).
fn read_outer_body_from_cache(path: &Path, last_body: [f32; 5]) -> [f32; 5] {
    match read_sensor_cache(path, outer_body_stale_threshold_s()) {
        Ok(SensorCacheRead::Fresh { payload, .. }) => {
            match project_outer_body_5d(&payload, last_body) {
                Ok(body) => body,
                Err(e) => {
                    warn!(err = ?e, "outer_body source-dict project failed; using last-known");
                    last_body
                }
            }
        }
        Ok(SensorCacheRead::Stale { age_s, .. }) => {
            warn!(
                event = "SENSOR_CACHE_STALE",
                age_s,
                threshold_s = outer_body_stale_threshold_s(),
                confidence = 0.0,
                "outer_body sensor cache stale; using last-known body",
            );
            last_body
        }
        Ok(SensorCacheRead::Missing) => last_body,
        Err(e) => {
            warn!(err = ?e, "sensor_cache read errored; using last-known");
            last_body
        }
    }
}

// ── V6 outer_body_5d full port ─────────────────────────────────────
//
// Byte-identical port of `titan_hcl/logic/outer_trinity.py::_collect_outer_body`
// (lines 372–478). 5DT V6 body-felt semantics with weighted blends + clamp.
//
// Parity bar: |Δ| < 1e-4 (the Python output is `round(v, 4)`, so f32 cast
// preserves all 4-decimal-place values within 1e-4 of f64).
//
// Closes rFP_phase_c_close_all_runtime_gaps chunk 9G — supersedes the
// stub-port that read only `sol_balance` for dim[0]. Per Prime Directive #1
// "if a function exists, it MUST do the work its name claims".

/// V6 5DT body-felt projection. Pure compute — caller passes the
/// msgpack-encoded source dict from the Python sidecar plus the
/// previous-tick `last_body` (used by dim[2] somatosensation per
/// `outer_trinity.py:434` `current_ob2 = self._last_outer_body[2]`,
/// with the 9G decay-fix applied — see [`apply_dim2_decay`]).
///
/// Returns `Err` only when the msgpack envelope is fundamentally malformed
/// (not a map). Missing individual fields contribute their documented
/// neutral defaults (0.5 for ratios / scalars, 0.0 for rates).
fn project_outer_body_5d(payload: &[u8], last_body: [f32; 5]) -> Result<[f32; 5]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Err(anyhow!("source dict not a map")),
    };

    // ── Top-level field lookups (each may be Nil/missing → None). ──
    let helper_statuses = lookup_map(map, "helper_statuses");
    let sys_stats = lookup_map(map, "system_sensor_stats");
    let net_stats = lookup_map(map, "network_monitor_stats");
    let tx_lat = lookup_map(map, "tx_latency_stats");
    // D-SPEC-101 Phase-2: minutes-scale rate-of-change breath for entropy[68]
    // + thermal[69] (computed + tracked by the plugin ChangeBreathTracker).
    let outer_body_change = lookup_map(map, "outer_body_change");

    // ── [0] interoception RE-GROUNDED (D-SPEC-101 Phase-2) ────────────
    //   π-cluster heartbeat variance over a rolling 24h window (HRV-like) —
    //   the Titan's awareness of its own fluctuating inner cadence. The plugin
    //   EmaVarianceTracker emits the scale-free CV² of the π-heartbeat pulse
    //   rate (`pi_heartbeat_hrv`). Was 0.4*sol_norm + 0.3*block_rate +
    //   0.3*anchor_fresh (a chain/wallet-liveness blend — that signal lives in
    //   the SAT origin/transactional dims; topology rate-of-change is used
    //   elsewhere). Source: top-level pi_heartbeat_hrv. Per Maker 2026-05-21.
    let interoception: f64 = safe_clamp(field_or_default(Some(map), "pi_heartbeat_hrv", 0.5));

    // ── [1] proprioception ────────────────────────────────────────
    // Python lines 412-423.
    let peer_entropy: f64 = field_or_default(net_stats.as_ref(), "peer_entropy", 0.5);
    let helper_health: f64 = compute_helper_health(helper_statuses.as_ref());
    let bus_module_diversity: f64 =
        field_or_default(net_stats.as_ref(), "bus_module_diversity", 0.5);
    let proprioception: f64 =
        safe_clamp(0.5 * peer_entropy + 0.3 * helper_health + 0.2 * bus_module_diversity);

    // ── [2] somatosensation ───────────────────────────────────────
    // Python lines 425-442. dim[2] reads previous-tick value (`current_ob2`).
    // 9G decay-fix: apply exponential decay toward 0.5 per tick to
    // prevent saturation. See `apply_dim2_decay` rationale.
    let tx_lat_norm: f64 = field_or_default(tx_lat.as_ref(), "normalized", 0.5);
    let current_ob2: f64 = apply_dim2_decay(last_body[2] as f64);
    let cpu_spikes: f64 = field_or_default(sys_stats.as_ref(), "cpu_spike_rate", 0.0);
    let somatosensation: f64 = safe_clamp(0.4 * tx_lat_norm + 0.3 * current_ob2 + 0.3 * cpu_spikes);

    // ── [3] entropy RE-GROUNDED (D-SPEC-101 Phase-2) ──────────────────
    //   RATE OF CHANGE of the system-entropy level over a minutes-scale window
    //   (breath), not the instantaneous value. The plugin composes the level
    //   (ping_var/bus_drop/error_rate) + tracks |Δ|/dt via ChangeBreathTracker
    //   (old instantaneous Rust formula deleted — no shim). Source:
    //   outer_body_change.entropy_change. Per Maker 2026-05-21.
    let entropy: f64 = safe_clamp(field_or_default(
        outer_body_change.as_ref(),
        "entropy_change",
        0.0,
    ));

    // ── [4] thermal RE-GROUNDED (D-SPEC-101 Phase-2) ──────────────────
    //   RATE OF CHANGE of the thermal level (cpu_thermal/circadian/hormonal_heat)
    //   over a minutes-scale window (breath). Source: outer_body_change.thermal_change.
    let thermal: f64 = safe_clamp(field_or_default(
        outer_body_change.as_ref(),
        "thermal_change",
        0.0,
    ));

    // ── Final: round to 4 decimals + cast to f32 (matches Python
    // `round(v, 4)` output then f32 slot write). ─────────────────────
    Ok([
        round4_f32(interoception),
        round4_f32(proprioception),
        round4_f32(somatosensation),
        round4_f32(entropy),
        round4_f32(thermal),
    ])
}

// ── V6 helpers (named to mirror Python semantics) ─────────────────

/// `_safe_clamp(value, lo=0.0, hi=1.0)` from `outer_trinity.py:749-756`.
/// Defaults to 0.5 on NaN/Inf; otherwise clamps to [0, 1].
fn safe_clamp(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() {
        return 0.5;
    }
    v.clamp(0.0, 1.0)
}

/// Round to 4 decimal places (Python `round(v, 4)`) and return as f32.
/// Python 3 uses banker's rounding (half-to-even); for f32-precision
/// values the difference vs round-half-away-from-zero is below f32 ε
/// for nearly all inputs. Tolerance check in parity tests = 1e-4.
fn round4_f32(v: f64) -> f32 {
    let scaled = v * 10_000.0;
    let rounded = scaled.round_ties_even();
    (rounded / 10_000.0) as f32
}

/// Look up a top-level key; if value is a map, return its entries.
/// Missing or non-map values return None — callers treat that as
/// "field absent → use defaults" (matches Python `sources.get("k") or {}`).
fn lookup_map(
    map: &[(rmpv::Value, rmpv::Value)],
    key: &str,
) -> Option<Vec<(rmpv::Value, rmpv::Value)>> {
    use rmpv::Value;
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some(key) {
                if let Value::Map(items) = v {
                    return Some(items.clone());
                }
                return None;
            }
        }
    }
    None
}

/// Look up a numeric field within a sub-map; default if missing/non-numeric.
fn field_or_default(map: Option<&Vec<(rmpv::Value, rmpv::Value)>>, key: &str, default: f64) -> f64 {
    let map = match map {
        Some(m) => m,
        None => return default,
    };
    use rmpv::Value;
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some(key) {
                return v.as_f64().unwrap_or(default);
            }
        }
    }
    default
}

/// Compute helper_health = (count "available") / total per Python lines 415-417.
/// total_helpers = max(1, len(helper_statuses)); empty/None → 0.5 neutral.
fn compute_helper_health(helper_statuses: Option<&Vec<(rmpv::Value, rmpv::Value)>>) -> f64 {
    let helpers = match helper_statuses {
        Some(h) => h,
        None => return 0.5,
    };
    use rmpv::Value;
    let total = helpers.len();
    if total == 0 {
        return 0.5;
    }
    let mut available = 0_usize;
    for (_k, v) in helpers.iter() {
        if let Value::String(s) = v {
            if s.as_str() == Some("available") {
                available += 1;
            }
        }
    }
    available as f64 / total as f64
}

/// 9G decay-fix: dim[2] (somatosensation) reads its own previous-tick
/// value (`current_ob2`) which without decay creates a self-reinforcing
/// loop that saturates upward — Python's `_collect_outer_body` line
/// 427 explicitly flagged this as "decay fix shipping in next commit;
/// until then, saturates". Per rFP §4 + Maker decision 2026-05-06
/// "no deferrals", the fix lands here:
///
///   `current_ob2 = 0.5 + (last_ob2 - 0.5) × DECAY_FACTOR`
///
/// `DECAY_FACTOR = 0.95` per tick (10s cadence) gives a half-life of
/// log(0.5)/log(0.95) ≈ 13.5 ticks ≈ 2.25 minutes — long enough that
/// genuine cumulative load still registers, short enough that
/// transient spikes don't pin the dim. Both Python `outer_trinity.py`
/// and Rust port apply identical decay so parity tests stay green.
fn apply_dim2_decay(last_ob2: f64) -> f64 {
    const DECAY_FACTOR: f64 = 0.95;
    0.5 + (last_ob2 - 0.5) * DECAY_FACTOR
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

/// Build a SPEC §8.5 outer `BODY_STATE` payload as `rmpv::Value::Map` per
/// SPEC §8.10 line 900 byte-identical guarantee.
fn encode_body_state_payload(body: &[f32; 5]) -> rmpv::Value {
    use rmpv::Value;
    let values = Value::Array(body.iter().map(|f| Value::F64(*f as f64)).collect());
    Value::Map(vec![
        (Value::String("src".into()), Value::String("outer".into())),
        (
            Value::String("type".into()),
            Value::String("BODY_STATE".into()),
        ),
        (Value::String("values".into()), values),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ])
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
    use titan_core::constants::OUTER_BODY_5D_SCHEMA_VERSION;

    /// Helper: pure-compute "what does run_one_tick produce given inputs"
    /// — extracted so unit tests exercise the transformation pipeline
    /// without spinning up tokio + bus + slots.
    fn pure_compute(
        raw_body: [f32; 5],
        unified: Option<[f32; 5]>,
        local: Option<[f32; 5]>,
        topology_lower: Option<[f32; 10]>,
        ground_up: &mut GroundUpEnricher,
    ) -> [f32; 5] {
        let mut body = raw_body;
        let composed = match (unified, local) {
            (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
            (Some(u), None) => u.to_vec(),
            (None, Some(l)) => l.to_vec(),
            (None, None) => vec![1.0_f32; 5],
        };
        apply_multipliers(&mut body, &composed);
        // 0E: Some(topo) models an epoch boundary — recompute the held nudge,
        // then apply it (the per-tick application path).
        if let Some(topo) = topology_lower {
            ground_up.compute_nudge(&topo);
            ground_up.apply_held_to_body(&mut body, 1.0).unwrap();
        }
        body
    }

    #[test]
    fn neutral_inputs_preserve_body() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5, 0.5, 0.5, 0.5, 0.5];
        let out = pure_compute(raw, None, None, None, &mut g);
        for i in 0..5 {
            assert!((out[i] - raw[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn unified_only_applies_multipliers() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        let unified = [2.0, 0.5, 1.0, 1.5, 0.3];
        let out = pure_compute(raw, Some(unified), None, None, &mut g);
        // 0.5 * 2.0 = 1.0 (clipped at TENSOR_MAX), 0.5 * 0.5 = 0.25
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn local_only_applies_multipliers() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        let local = [1.0, 1.0, 2.0, 1.0, 1.0];
        let out = pure_compute(raw, None, Some(local), None, &mut g);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unified_and_local_compose_with_g7_clamp() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        // 2.0 (unified) * 1.5 (local) = 3.0 → at MULTIPLIER_CEIL=3.0
        let unified = [2.0, 1.0, 1.0, 1.0, 1.0];
        let local = [1.5, 1.0, 1.0, 1.0, 1.0];
        let out = pure_compute(raw, Some(unified), Some(local), None, &mut g);
        // body[0] = 0.5 * 3.0 = 1.5 → clipped at TENSOR_MAX=1.0
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ground_up_applied_to_all_5d_per_g10() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        let topo: [f32; 10] = [0.04, 0.04, 0.04, 0.04, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = pure_compute(raw, None, None, Some(topo), &mut g);
        // Body all 5D nudged: delta = 0.002 * 0.1 * 1.0 = 0.0002 each
        // (matches inner-body test pattern from C-S5; G10 ground_up_body_range=0:5)
        for (i, val) in out.iter().enumerate() {
            assert!(
                (val - 0.5002).abs() < 1e-5,
                "dim {} should be nudged ~0.0002",
                i,
            );
        }
    }

    #[test]
    fn ground_up_skipped_without_topology() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        let out = pure_compute(raw, None, None, None, &mut g);
        for v in out.iter() {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn body_payload_msgpack_shape() {
        let body = [0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes = encode_body_state_payload(&body);
        use rmpv::Value;
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
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
    fn body_payload_src_is_outer() {
        let body = [0.0; 5];
        let bytes = encode_body_state_payload(&body);
        use rmpv::Value;
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("src") {
                        assert_eq!(val, Value::String("outer".into()));
                        return;
                    }
                }
            }
        }
        panic!("src field missing");
    }

    #[test]
    fn slot_write_5d_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("outer_body_5d.bin");
        let mut slot = Slot::create(&path, OUTER_BODY_5D_SCHEMA_VERSION as u32, 20).unwrap();
        let body: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes = encode_floats::<5>(&body);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn project_outer_body_returns_neutral_on_empty_dict() {
        // Empty msgpack map → defaults dominate. D-SPEC-101 Phase-2:
        // dim[0] interoception = pi_heartbeat_hrv default 0.5
        // dim[1] = 0.5·0.5 + 0.3·0.5 + 0.2·0.5 = 0.5
        // dim[2] = 0.4·0.5 + 0.3·apply_dim2_decay(0.5) + 0.3·0.0 = 0.35
        // dim[3] entropy = outer_body_change.entropy_change absent → 0.0
        // dim[4] thermal = outer_body_change.thermal_change absent → 0.0
        use rmpv::Value;
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &Value::Map(vec![])).unwrap();
        let last_body = [0.5_f32; 5];
        let body = project_outer_body_5d(&payload, last_body).unwrap();
        assert!((body[0] - 0.5).abs() < 1e-4);
        assert!((body[1] - 0.5).abs() < 1e-4);
        assert!((body[2] - 0.35).abs() < 1e-4);
        assert!((body[3] - 0.0).abs() < 1e-4);
        assert!((body[4] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn project_outer_body_interoception_reads_pi_heartbeat_hrv() {
        // D-SPEC-101 Phase-2 (Maker 2026-05-21): interoception[0] = π-heartbeat
        // 24h HRV (top-level pi_heartbeat_hrv), NOT the sol/block/anchor blend.
        use rmpv::Value;
        let mut payload = Vec::new();
        let map = Value::Map(vec![(
            Value::String("pi_heartbeat_hrv".into()),
            Value::F64(0.62),
        )]);
        rmpv::encode::write_value(&mut payload, &map).unwrap();
        let body = project_outer_body_5d(&payload, [0.5_f32; 5]).unwrap();
        assert!(
            (body[0] - 0.62).abs() < 1e-3,
            "interoception got {}",
            body[0]
        );
    }

    #[test]
    fn project_outer_body_entropy_thermal_read_change_breath() {
        // D-SPEC-101 Phase-2: entropy[3] + thermal[4] read the minutes-scale
        // rate-of-change breath (outer_body_change.{entropy,thermal}_change).
        use rmpv::Value;
        let map = Value::Map(vec![(
            Value::String("outer_body_change".into()),
            Value::Map(vec![
                (Value::String("entropy_change".into()), Value::F64(0.44)),
                (Value::String("thermal_change".into()), Value::F64(0.21)),
            ]),
        )]);
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &map).unwrap();
        let body = project_outer_body_5d(&payload, [0.5_f32; 5]).unwrap();
        assert!((body[3] - 0.44).abs() < 1e-3, "entropy got {}", body[3]);
        assert!((body[4] - 0.21).abs() < 1e-3, "thermal got {}", body[4]);
    }

    #[test]
    fn project_outer_body_dim2_decay_pulls_toward_neutral() {
        // last_body[2] = 1.0 → decayed = 0.5 + (1.0 - 0.5) × 0.95 = 0.975
        // Empty map → tx_lat=0.5, cpu_spikes=0.0
        // dim[2] = 0.4·0.5 + 0.3·0.975 + 0.3·0.0 = 0.2 + 0.2925 = 0.4925
        use rmpv::Value;
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &Value::Map(vec![])).unwrap();
        let last_body = [0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let body = project_outer_body_5d(&payload, last_body).unwrap();
        assert!(
            (body[2] - 0.4925).abs() < 1e-4,
            "dim[2] with last_ob2=1.0 should be ~0.4925; got {}",
            body[2]
        );
    }

    #[test]
    fn project_outer_body_full_formula_matches_python_reference() {
        // Construct realistic source dict with values for all 5 dims;
        // verify the output matches the Python formula computed manually.
        use rmpv::Value;

        let payload = Value::Map(vec![
            (Value::String("sol_balance".into()), Value::F64(2.0)),
            (
                Value::String("block_delta_stats".into()),
                Value::Map(vec![(Value::String("normalized".into()), Value::F64(0.7))]),
            ),
            (
                Value::String("anchor_state".into()),
                Value::Map(vec![
                    (Value::String("success".into()), Value::Boolean(true)),
                    // last_anchor_time = now → since = 0 → fresh = 1.0
                    (
                        Value::String("last_anchor_time".into()),
                        Value::F64(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs_f64(),
                        ),
                    ),
                ]),
            ),
            (
                Value::String("network_monitor_stats".into()),
                Value::Map(vec![
                    (Value::String("peer_entropy".into()), Value::F64(0.8)),
                    (
                        Value::String("bus_module_diversity".into()),
                        Value::F64(0.6),
                    ),
                    (Value::String("ping_variance".into()), Value::F64(0.3)),
                    (Value::String("bus_drop_rate".into()), Value::F64(0.1)),
                ]),
            ),
            (
                Value::String("helper_statuses".into()),
                Value::Map(vec![
                    (Value::String("a".into()), Value::String("available".into())),
                    (Value::String("b".into()), Value::String("available".into())),
                    (Value::String("c".into()), Value::String("offline".into())),
                ]),
            ),
            (
                Value::String("tx_latency_stats".into()),
                Value::Map(vec![(Value::String("normalized".into()), Value::F64(0.4))]),
            ),
            (
                Value::String("system_sensor_stats".into()),
                Value::Map(vec![
                    (Value::String("cpu_spike_rate".into()), Value::F64(0.2)),
                    (Value::String("cpu_thermal".into()), Value::F64(0.6)),
                    (Value::String("circadian_phase".into()), Value::F64(0.4)),
                ]),
            ),
            (
                Value::String("agency_stats".into()),
                Value::Map(vec![
                    (
                        Value::String("total_actions".into()),
                        Value::Integer(100.into()),
                    ),
                    (
                        Value::String("failed_actions".into()),
                        Value::Integer(15.into()),
                    ),
                ]),
            ),
            // D-SPEC-101 Phase-2 source keys: interoception[0] = π HRV,
            // entropy[3]/thermal[4] = minutes-scale rate-of-change breath.
            (Value::String("pi_heartbeat_hrv".into()), Value::F64(0.55)),
            (
                Value::String("outer_body_change".into()),
                Value::Map(vec![
                    (Value::String("entropy_change".into()), Value::F64(0.33)),
                    (Value::String("thermal_change".into()), Value::F64(0.66)),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let last_body = [0.0_f32, 0.0, 0.6, 0.0, 0.0]; // dim[2] tracks
        let body = project_outer_body_5d(&bytes, last_body).unwrap();

        // Hand-computed expected (D-SPEC-101 Phase-2):
        //   d0 interoception = pi_heartbeat_hrv = 0.55
        //   helper_health = 2/3 ≈ 0.6667
        //   d1 = 0.5·0.8 + 0.3·0.6667 + 0.2·0.6 = 0.72
        //   ob2_decayed = 0.5 + (0.6 - 0.5)*0.95 = 0.595
        //   d2 = 0.4·0.4 + 0.3·0.595 + 0.3·0.2 = 0.3985
        //   d3 entropy = outer_body_change.entropy_change = 0.33
        //   d4 thermal = outer_body_change.thermal_change = 0.66
        let expected = [0.55_f32, 0.72_f32, 0.3985_f32, 0.33_f32, 0.66_f32];
        for i in 0..5 {
            assert!(
                (body[i] - expected[i]).abs() < 1e-3,
                "dim[{i}] got {} expected {}",
                body[i],
                expected[i]
            );
        }
    }

    #[test]
    fn safe_clamp_handles_nan_and_inf() {
        assert_eq!(safe_clamp(f64::NAN), 0.5);
        assert_eq!(safe_clamp(f64::INFINITY), 0.5);
        assert_eq!(safe_clamp(f64::NEG_INFINITY), 0.5);
        assert_eq!(safe_clamp(2.0), 1.0);
        assert_eq!(safe_clamp(-1.0), 0.0);
        assert_eq!(safe_clamp(0.5), 0.5);
    }

    #[test]
    fn round4_f32_matches_python_round() {
        assert_eq!(round4_f32(0.123456), 0.1235);
        assert_eq!(round4_f32(0.5), 0.5);
        assert_eq!(round4_f32(0.0), 0.0);
        assert_eq!(round4_f32(1.0), 1.0);
    }

    #[test]
    fn apply_dim2_decay_pulls_toward_half() {
        // From neutral: stays at 0.5
        assert!((apply_dim2_decay(0.5) - 0.5).abs() < 1e-9);
        // From 1.0: decays toward 0.5: 0.5 + 0.5*0.95 = 0.975
        assert!((apply_dim2_decay(1.0) - 0.975).abs() < 1e-9);
        // From 0.0: 0.5 + (-0.5)*0.95 = 0.025
        assert!((apply_dim2_decay(0.0) - 0.025).abs() < 1e-9);
    }

    #[test]
    fn decode_unified_outer_body_extracts_field() {
        use rmpv::Value;
        let payload = Value::Map(vec![
            (
                Value::String("multipliers".into()),
                Value::Map(vec![(
                    Value::String("outer_body".into()),
                    Value::Array(vec![
                        Value::F64(1.5),
                        Value::F64(0.8),
                        Value::F64(2.0),
                        Value::F64(1.0),
                        Value::F64(0.5),
                    ]),
                )]),
            ),
            (Value::String("epoch_id".into()), Value::Integer(42.into())),
            (Value::String("ts".into()), Value::F64(1714400000.0)),
        ]);
        let mults = decode_unified_outer_body(&payload).unwrap();
        let expected: [f32; 5] = [1.5, 0.8, 2.0, 1.0, 0.5];
        for i in 0..5 {
            assert!((mults[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_unified_outer_body_errors_on_missing_field() {
        // multipliers map without outer_body key
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("multipliers".into()),
            Value::Map(vec![]),
        )]);
        let result = decode_unified_outer_body(&payload);
        assert!(result.is_err(), "missing outer_body should error");
    }

    #[test]
    fn content_hash_gates_redundant_writes() {
        let mut gate = ContentGate::new();
        let body: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes = encode_floats::<5>(&body);
        assert!(gate.should_write(&bytes));
        assert!(!gate.should_write(&bytes));
        assert!(!gate.should_write(&bytes));
        assert_eq!(gate.write_count(), 1);
        assert_eq!(gate.suppress_count(), 2);
    }

    #[test]
    fn stale_threshold_is_3x_cadence() {
        // SPEC §18.1 + D-SPEC-100: outer_body cadence 45s × 3 = 135s
        assert_eq!(outer_body_stale_threshold_s(), 135.0);
    }

    #[test]
    fn cadence_constants_match_spec() {
        // Sensor sidecar source-refresh cadence (D-SPEC-100: stale threshold source).
        // G13 body-slowest: 45s = strict 1:3:9 (spirit 5 / mind 15 / body 45).
        assert_eq!(OUTER_BODY_TICK_BASE_S, 45.0);
        // Bus publish throttle (Schumann body × 39 ≈ 45s) — now mirrors TICK_BASE.
        // Body-slowest G13 invariant: 45 > 15 (mind) > 5 (spirit).
        assert_eq!(OUTER_BODY_BUS_PUBLISH_INTERVAL_S, 45.0);
    }

    #[test]
    fn schumann_body_period_is_canonical() {
        // Daemon ticks at SCHUMANN_BODY_HZ (7.83 Hz, ~127.7ms) per
        // post-A.S8 D2 cadence migration.
        let g = SchumannGenerator::new(SchumannRole::Body, tokio::time::Instant::now());
        let period_ns = g.period_ns();
        assert!(
            (period_ns as i64 - 127_713_921).abs() <= 1,
            "body period_ns = {period_ns}, expected 127713921 ± 1"
        );
    }
}
