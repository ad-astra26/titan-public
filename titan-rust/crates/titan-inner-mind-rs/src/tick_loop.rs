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
    apply_multipliers, compose_focus_into_enrichment, compose_multipliers_default,
    decode_corrective_nudge, decode_filter_down_payload, decode_local_filter_down_payload,
    encode_extreme_imbalance, encode_floats, encode_mind_balance_gift, load_checkpoint_for_part,
    load_restoring_cfg, observe, open_focus_input_if_present, open_neuromod_slot_if_present,
    read_focus_nudge, read_neuromod_gain, stateful_update, write_checkpoint_for_part,
    BalancedPulseEdges, CheckpointSnapshot, ContentGate, DriftAggregator, FiringSlotWriter,
    FocusPart, GroundUpEnricher, JourneyAccumulator, JourneyTickInputs, Layer, PolarityHomeostat,
    PolarityHomeostatCfg, PulseClockRole, PulseWatcher, RestoringCfg, Side, TrinitySide,
    EXTREME_IMBALANCE_DETECTED_TOPIC, INNER_MIND_TOPICS, MIND_BALANCE_GIFT_TOPIC,
    MIND_GIFT_WEIGHTS,
};

/// §G5.2 item 4 checkpoint write cadence. Inner-mind ticks @ 23.49 Hz so
/// 240 ticks ≈ 10 s between snapshots.
const CHECKPOINT_WRITE_EVERY_N_TICKS: u64 = 240;
const CHECKPOINT_PART: &str = "inner_mind";

const MIND_DIMS: usize = 15;
const SENSOR_CACHE_DIMS: usize = 15;
const DRIFT_THRESHOLD_PCT: f64 = 0.005;

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path, data_dir: &Path) -> Result<()> {
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
    info!(
        event = "SHM_OPENED",
        sensor_cache_present = sensor_cache.is_some()
    );

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
        shm_dir.to_path_buf(),
        data_dir.to_path_buf(),
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
    /// Set true on each KERNEL_EPOCH_TICK; consumed by the tick loop to
    /// recompute the ground_up held nudge once per epoch (SPEC §G5.1 / 0E).
    epoch_pending: bool,
    /// Shutdown via KERNEL_SHUTDOWN_ANNOUNCE.
    shutdown_requested: bool,
    /// P0.6-C / D-SPEC-132 one-shot CORRECTIVE_NUDGE from inner-spirit-rs.
    pending_nudge: Option<(usize, f32)>,
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
        "KERNEL_EPOCH_TICK" => {
            // 0E: mark an epoch boundary — tick loop recomputes the held
            // ground_up nudge once per epoch (SPEC §G5.1).
            if let Ok(mut s) = state.lock() {
                s.epoch_pending = true;
            }
        }
        "CORRECTIVE_NUDGE" => {
            // P0.6-C / D-SPEC-132: accept only target_src=mind + target_side=Inner.
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_corrective_nudge(&payload) {
                Ok(n) => {
                    if n.target_src != "mind" || n.target_side != TrinitySide::Inner {
                        return;
                    }
                    let idx = n.target_dim_idx as usize;
                    if idx >= 15 {
                        warn!(target_dim_idx = idx, "CORRECTIVE_NUDGE idx out of range");
                        return;
                    }
                    if let Ok(mut s) = state.lock() {
                        s.pending_nudge = Some((idx, n.nudge_value));
                    }
                }
                Err(e) => warn!(err = ?e, "decode CORRECTIVE_NUDGE failed"),
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
    shm_dir: std::path::PathBuf,
    data_dir: std::path::PathBuf,
) -> Result<()> {
    let mut content_gate = ContentGate::new();
    let mut ground_up = GroundUpEnricher::new(Side::MindWilling);
    let mut drift_agg = DriftAggregator::new("mind", DRIFT_THRESHOLD_PCT);
    // §G5.2 item 4 — restore exact tensor state from checkpoint on boot;
    // cold-start at 0.5 only when sidecar absent/invalid.
    let (mut prev, mut prev2, mut last_obs_restored) =
        match load_checkpoint_for_part::<MIND_DIMS>(&data_dir, CHECKPOINT_PART) {
            Some(CheckpointSnapshot {
                prev,
                prev2,
                last_obs,
                ..
            }) => (prev, prev2, Some(last_obs)),
            None => ([0.5_f32; MIND_DIMS], [0.5_f32; MIND_DIMS], None),
        };
    // §G5.2 item 5 — load gains from titan_params.toml [trinity_restoring]
    // via the Python L2-published `trinity_restoring.bin` sidecar.
    let mut cfg = load_restoring_cfg(&shm_dir, Layer::Mind);
    // §G5.2 item 2 — live neuromod-gain read per tick.
    let mut neuromod_slot = open_neuromod_slot_if_present(&shm_dir);
    let neuromod_path = shm_dir.join("neuromod_state.bin");
    // §G5.2 item 2 + §G12 — FOCUS cascade nudge slot.
    let mut focus_input_slot = open_focus_input_if_present(&shm_dir);
    let focus_input_path = shm_dir.join("focus_input.bin");

    // P0.5 / D-SPEC-131 §G5.1 UP-leg balance-gift state (mirror of inner-body
    // wiring): SHM-direct PulseWatcher detects the inner_mind clock balanced
    // rising-edge; the JourneyAccumulator<15> tracks per-cycle qualitative
    // aggregates (coherence climb, polarity dynamics, direction stability)
    // that feed the MIND_BALANCE_GIFT digest.
    let mut pulse_watcher = PulseWatcher::open(&shm_dir);
    let mut journey_acc: JourneyAccumulator<MIND_DIMS> = JourneyAccumulator::new();
    // P0.6-C / D-SPEC-132 PolarityHomeostat for inner_mind.
    let mut polarity_homeostat: PolarityHomeostat<MIND_DIMS> =
        PolarityHomeostat::new(PolarityHomeostatCfg::for_mind());

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
                        cfg = load_restoring_cfg(&shm_dir, Layer::Mind);
                    }
                    let (_pulse_edges, balanced_pulse_edges) = pulse_watcher.tick_with_balanced();
                    let drift_pct = tick_event.jitter_ns() as f64 / tick_event.period_ns as f64;
                    drift_agg.observe(drift_pct, tick_event.jitter_ns(), tick_event.epoch);
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut ground_up,
                        &mut inner_mind_slot, &topology_slot, &sensor_cache,
                        &mut firing_writer, &mut prev, &mut prev2,
                        &mut cfg, neuromod_slot.as_ref(), focus_input_slot.as_ref(),
                        &mut last_obs_restored,
                        &mut journey_acc, &balanced_pulse_edges,
                        &mut polarity_homeostat,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
                    }
                    tick_count = tick_count.wrapping_add(1);
                    // §G5.2 item 4 — periodic checkpoint write.
                    if tick_count.is_multiple_of(CHECKPOINT_WRITE_EVERY_N_TICKS) {
                        if let Some(o) = last_obs_restored.as_ref() {
                            if let Err(e) = write_checkpoint_for_part::<MIND_DIMS>(
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
        if let Err(e) =
            write_checkpoint_for_part::<MIND_DIMS>(&data_dir, CHECKPOINT_PART, &prev, &prev2, o)
        {
            warn!(err = ?e, "final checkpoint write failed");
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
    prev: &mut [f32; MIND_DIMS],
    prev2: &mut [f32; MIND_DIMS],
    cfg: &mut RestoringCfg,
    neuromod_slot: Option<&Slot>,
    focus_input_slot: Option<&Slot>,
    last_obs_restored: &mut Option<titan_trinity_daemon::LayerObs>,
    journey_acc: &mut JourneyAccumulator<MIND_DIMS>,
    balanced_pulse_edges: &BalancedPulseEdges,
    polarity_homeostat: &mut PolarityHomeostat<MIND_DIMS>,
) -> Result<()> {
    // raw[t] = un-enriched 15D sensor reading (§G5.2 drive target).
    let raw = read_sensor_cache(sensor_cache)?;
    let mut mind = raw;

    let (unified, local, topology_fresh, epoch_due) = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        let epoch_due = s.epoch_pending;
        s.epoch_pending = false;
        // D-SPEC-121 (v1.54.0): one-shot consume-and-clear of filter_down
        // multipliers (see inner-body for rationale).
        (
            s.unified.take(),
            s.local.take(),
            s.topology_signaled,
            epoch_due,
        )
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
    // which only modifies mind[10:15] in `apply_held_to_mind`.
    // 0E held-nudge model (SPEC §G5.1): RECOMPUTE the damped nudge ONCE per
    // kernel epoch (so the 0.95 EMA evolves at epoch cadence, not 23.49 Hz),
    // then APPLY the held nudge every tick (continuous manifestation, no
    // compounding — mind is rebuilt from raw each tick).
    if topology_fresh {
        if epoch_due {
            if let Ok(topology_lower) =
                titan_trinity_daemon::read_topology_inner_lower(topology_slot)
            {
                ground_up.compute_nudge(&topology_lower);
            }
        }
        ground_up.apply_held_to_mind(&mut mind, 1.0)?;
    }

    // §G5.2 traveling-tensor update (Layer::Mind — gradient .5/.5).
    // enrichment = (enriched − raw) — the filter_down + (willing-only) ground_up
    // delta applied above. Passed as a SEPARATE full-weight additive term per
    // §G5.2 (G5.2-enrichment-separate). Spring is modulated by the live
    // neuromod gain (G5.2-neuromod-gain).
    let mut enrichment = [0.0_f32; MIND_DIMS];
    for i in 0..MIND_DIMS {
        enrichment[i] = mind[i] - raw[i];
    }
    // §G12 FOCUS cascade: amplified nudge composes into enrichment_force.
    let focus = read_focus_nudge::<MIND_DIMS>(focus_input_slot, FocusPart::InnerMind);
    compose_focus_into_enrichment(&mut enrichment, &focus);
    cfg.neuromod_gain = read_neuromod_gain(neuromod_slot);
    let obs = observe(&prev[..], &prev2[..]);
    *last_obs_restored = Some(obs);
    // P0.6-C / D-SPEC-132: apply pending CORRECTIVE_NUDGE before stateful_update.
    let pending_nudge = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.pending_nudge.take()
    };
    if let Some((dim_idx, signed_nudge)) = pending_nudge {
        if dim_idx < MIND_DIMS {
            enrichment[dim_idx] += signed_nudge;
            debug!(event = "CORRECTIVE_NUDGE_APPLIED", dim_idx, signed_nudge);
        }
    }

    let x = stateful_update(&prev[..], &prev2[..], &raw[..], &enrichment[..], &obs, cfg);
    let mut mind_state = [0.0_f32; MIND_DIMS];
    mind_state.copy_from_slice(&x[..MIND_DIMS]);
    *prev2 = *prev;
    *prev = mind_state;
    let mind = mind_state;

    // P0.5 / D-SPEC-131 §G5.1 UP-leg: track this tick's journey + emit a
    // MIND_BALANCE_GIFT on this daemon's own clock balanced rising-edge.
    let tick_ts = now_secs();
    journey_acc.tick(JourneyTickInputs {
        x: &mind,
        obs,
        now_secs: tick_ts as f32,
    });
    // P0.6-C / D-SPEC-132: PolarityHomeostat tick on the post-§G5.2 mind state.
    if let Some(ev) = polarity_homeostat.tick(obs.polarity, &mind) {
        let payload = encode_extreme_imbalance("mind", TrinitySide::Inner, &ev, tick_ts);
        if let Err(e) = bus
            .publish(EXTREME_IMBALANCE_DETECTED_TOPIC, Some("all"), Some(payload))
            .await
        {
            warn!(err = ?e, "publish EXTREME_IMBALANCE_DETECTED failed (continuing)");
        } else {
            debug!(
                event = "EXTREME_IMBALANCE_DETECTED",
                side = "inner",
                src = "mind",
                dim = ev.dominant_dim_idx,
                pol = ev.polarity_at_fire,
                duration_ticks = ev.duration_ticks,
                sigma = ev.sigma_multiplier,
            );
        }
    }

    if balanced_pulse_edges[PulseClockRole::InnerMind.index()] {
        journey_acc.mark_balanced(obs);
        if let Some(digest) = journey_acc.finalize_mind_gift(&MIND_GIFT_WEIGHTS) {
            let payload =
                encode_mind_balance_gift::<MIND_DIMS>(TrinitySide::Inner, &digest, tick_ts);
            if let Err(e) = bus
                .publish(MIND_BALANCE_GIFT_TOPIC, Some("all"), Some(payload))
                .await
            {
                warn!(err = ?e, "publish MIND_BALANCE_GIFT failed (continuing)");
            } else {
                debug!(
                    event = "MIND_BALANCE_GIFT_EMITTED",
                    side = "inner",
                    amplitude = digest.gift_amplitude,
                    cycle_s = digest.cycle_duration_s,
                    ticks = digest.cycle_tick_count,
                );
            }
        }
        journey_acc.reset_for_next_cycle();
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
    bus.publish("MIND_STATE", Some("all"), Some(payload))
        .await
        .map_err(|e| anyhow!("publish MIND_STATE: {e}"))?;

    Ok(())
}

fn read_sensor_cache(sensor_cache: &Option<Slot>) -> Result<[f32; MIND_DIMS]> {
    match sensor_cache {
        Some(slot) => {
            let raw = slot.read().map_err(|e| anyhow!("sensor_cache read: {e}"))?;
            // §4.5 schema v2 msgpack source-dict. Two payload shapes are
            // accepted during the Rust-port cutover (Sprint 6 of
            // rFP_phase_c_130d_rust_l1_port):
            //   (a) NEW raw-inputs shape — has `thinking_5d` key. Rust
            //       computes the 15D per SPEC §23.5 collect_mind_15d.
            //   (b) LEGACY pre-computed shape — has `tensor` key. Rust
            //       passes the values through unchanged. Retained for
            //       backward-compat during the Python+Rust atomic deploy.
            // Pre-msgpack-era 15-float32-LE bytes also still decode.
            let is_msgpack = !raw.is_empty() && matches!(raw[0], 0x80..=0x8f | 0xde | 0xdf);
            if is_msgpack {
                project_inner_mind_15d(&raw, [0.5_f32; MIND_DIMS])
                    .or_else(|_| Ok([0.5_f32; MIND_DIMS]))
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

/// V6 15D inner-mind projection. Pure compute over the msgpack source
/// dict from `mind_worker._provide_mind_source_dict`. Implements SPEC
/// §23.5 `collect_mind_15d` (Thinking[0:5] + Feeling[5:10] + Willing[10:15]).
///
/// Source dict (NEW schema, Sprint 6):
///   - `thinking_5d`: [5 × f32] — passthrough from `_collect_mind_tensor`
///   - `audio_state`: {`creates_recent`: u, `ambient`: f}
///   - `interaction_quality`: f (defaults 0.5)
///   - `visual_state`: {`creates_recent`: u, `ambient`: f}
///   - `assessment_quality`: f (defaults 0.5)
///   - `ambient_change`: f (defaults 0.0)
///   - `hormone_levels`: {IMPULSE, EMPATHY, CREATIVITY, VIGILANCE, CURIOSITY}
///
/// Source dict (LEGACY schema, pre-Sprint-6 atomic-deploy fallback):
///   - `tensor`: [15 × f32] — already-computed 15D, return as-is.
///
/// `fallback` is returned only when the msgpack envelope is fundamentally
/// malformed (not a map). Per Prime Directive: no silent default-fill on
/// recoverable per-field misses — those use SPEC-locked field defaults.
pub fn project_inner_mind_15d(
    payload: &[u8],
    fallback: [f32; MIND_DIMS],
) -> Result<[f32; MIND_DIMS]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Ok(fallback),
    };

    // Legacy schema: caller already computed the 15D — pass through.
    if let Some(tensor) = lookup_array(map, "tensor") {
        let mut out = [0.5_f32; MIND_DIMS];
        for (i, item) in tensor.iter().take(MIND_DIMS).enumerate() {
            out[i] = item.as_f64().unwrap_or(0.5) as f32;
        }
        return Ok(out);
    }

    // NEW schema: compute per SPEC §23.5.
    let thinking_5d = lookup_array(map, "thinking_5d")
        .map(|arr| {
            let mut t = [0.5_f32; 5];
            for (i, item) in arr.iter().take(5).enumerate() {
                t[i] = item.as_f64().unwrap_or(0.5) as f32;
            }
            t
        })
        .unwrap_or([0.5_f32; 5]);

    let audio_state = lookup_map(map, "audio_state");
    let visual_state = lookup_map(map, "visual_state");
    let hormone_levels = lookup_map(map, "hormone_levels");
    let interaction_quality = top_level_f64_or(map, "interaction_quality", 0.5);
    let assessment_quality = top_level_f64_or(map, "assessment_quality", 0.5);
    let ambient_change = top_level_f64_or(map, "ambient_change", 0.0);

    let mut mind = [0.5_f32; MIND_DIMS];

    // ── THINKING (0..4): passthrough from current_5d. ──
    mind[..5].copy_from_slice(&thinking_5d);

    // ── FEELING (5..9) ──
    // [5] inner_hearing — 0.5*min(1, audio.creates/5) + 0.5*audio.ambient;
    //     fallback 0.4 when audio_state absent entirely (matches Python).
    mind[5] = match audio_state.as_ref() {
        Some(a) => {
            let creates = field_or_default(Some(a), "creates_recent", 0.0);
            let ambient = field_or_default(Some(a), "ambient", 0.5);
            safe_clamp(0.5 * (creates / 5.0).min(1.0) + 0.5 * ambient) as f32
        }
        None => 0.4,
    };
    // [6] inner_touch — clamp(interaction_quality).
    mind[6] = safe_clamp(interaction_quality) as f32;
    // [7] inner_sight — same shape as [5] but visual.
    mind[7] = match visual_state.as_ref() {
        Some(v) => {
            let creates = field_or_default(Some(v), "creates_recent", 0.0);
            let ambient = field_or_default(Some(v), "ambient", 0.5);
            safe_clamp(0.5 * (creates / 5.0).min(1.0) + 0.5 * ambient) as f32
        }
        None => 0.4,
    };
    // [8] inner_taste — clamp(assessment_quality).
    mind[8] = safe_clamp(assessment_quality) as f32;
    // [9] inner_smell — clamp(ambient_change) (already-clipped stddev).
    mind[9] = safe_clamp(ambient_change) as f32;

    // ── WILLING (10..14): hormone-level pressures.
    //     Python defaults to 0.5 when hormone_levels is None (the
    //     `willing = [0.5] * 5` initializer), else clamps each hormone
    //     field with 0.0 default. Match exactly.
    if hormone_levels.is_some() {
        let h = hormone_levels.as_ref();
        mind[10] = safe_clamp(field_or_default(h, "IMPULSE", 0.0)) as f32;
        mind[11] = safe_clamp(field_or_default(h, "EMPATHY", 0.0)) as f32;
        mind[12] = safe_clamp(field_or_default(h, "CREATIVITY", 0.0)) as f32;
        mind[13] = safe_clamp(field_or_default(h, "VIGILANCE", 0.0)) as f32;
        mind[14] = safe_clamp(field_or_default(h, "CURIOSITY", 0.0)) as f32;
    }

    Ok(mind)
}

// ── Helpers (inlined per outer-spirit-rs / outer-mind-rs convention;
//    extraction to a shared crate is a follow-up D8 cleanup item) ──

fn safe_clamp(v: f64) -> f64 {
    if v.is_nan() {
        return 0.5;
    }
    v.clamp(0.0, 1.0)
}

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

fn lookup_array(map: &[(rmpv::Value, rmpv::Value)], key: &str) -> Option<Vec<rmpv::Value>> {
    use rmpv::Value;
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some(key) {
                if let Value::Array(items) = v {
                    return Some(items.clone());
                }
                return None;
            }
        }
    }
    None
}

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

fn top_level_f64_or(map: &[(rmpv::Value, rmpv::Value)], key: &str, default: f64) -> f64 {
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

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

/// Build a SPEC §8.5 `MIND_STATE` payload as `rmpv::Value::Map`. Embedded
/// directly into the envelope by `encode_simple` per SPEC §8.10 line 900
/// byte-identical guarantee — NOT pre-encoded to opaque bytes.
fn encode_mind_state_payload(mind: &[f32; MIND_DIMS]) -> rmpv::Value {
    use rmpv::Value;
    let values = Value::Array(mind.iter().map(|f| Value::F64(*f as f64)).collect());
    Value::Map(vec![
        (Value::String("src".into()), Value::String("inner".into())),
        (
            Value::String("type".into()),
            Value::String("MIND_STATE".into()),
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
        // 0E: Some(topo) models an epoch boundary — recompute the held nudge,
        // then apply it (the per-tick application path).
        if let Some(topo) = topology_lower {
            ground_up.compute_nudge(&topo);
            ground_up.apply_held_to_mind(&mut mind, 1.0).unwrap();
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
    fn mind_payload_src_is_inner_and_type_mind_state() {
        let mind = [0.0_f32; MIND_DIMS];
        let bytes = encode_mind_state_payload(&mind);
        use rmpv::Value;
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
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
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
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

    fn encode_msgpack_map(pairs: Vec<(&str, rmpv::Value)>) -> Vec<u8> {
        use rmpv::Value;
        let items: Vec<(Value, Value)> = pairs
            .into_iter()
            .map(|(k, v)| (Value::String(k.into()), v))
            .collect();
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &Value::Map(items)).unwrap();
        out
    }

    #[test]
    fn project_inner_mind_15d_neutral_inputs_match_python_defaults() {
        // Python collect_mind_15d with: current_5d=[0.5;5], audio_state=None,
        // interaction_quality=0.5, visual_state=None, assessment_quality=0.5,
        // ambient_change=0.0, hormone_levels=None
        // → thinking=[0.5;5], feeling=[0.4, 0.5, 0.4, 0.5, 0.0], willing=[0.5;5]
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("interaction_quality", rmpv::Value::F64(0.5)),
            ("assessment_quality", rmpv::Value::F64(0.5)),
            ("ambient_change", rmpv::Value::F64(0.0)),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        for i in 0..5 {
            assert!((out[i] - 0.5).abs() < 1e-6, "thinking[{i}]={}", out[i]);
        }
        assert!((out[5] - 0.4).abs() < 1e-6, "inner_hearing={}", out[5]);
        assert!((out[6] - 0.5).abs() < 1e-6, "inner_touch={}", out[6]);
        assert!((out[7] - 0.4).abs() < 1e-6, "inner_sight={}", out[7]);
        assert!((out[8] - 0.5).abs() < 1e-6, "inner_taste={}", out[8]);
        assert!((out[9] - 0.0).abs() < 1e-6, "inner_smell={}", out[9]);
        for i in 10..15 {
            assert!((out[i] - 0.5).abs() < 1e-6, "willing[{}]={}", i, out[i]);
        }
    }

    #[test]
    fn project_inner_mind_15d_audio_state_drives_inner_hearing() {
        // creates_recent=3, ambient=0.6 → 0.5*min(1,3/5) + 0.5*0.6 = 0.3 + 0.3 = 0.6
        let audio = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("creates_recent".into()),
                rmpv::Value::F64(3.0),
            ),
            (rmpv::Value::String("ambient".into()), rmpv::Value::F64(0.6)),
        ]);
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("audio_state", audio),
            ("interaction_quality", rmpv::Value::F64(0.5)),
            ("assessment_quality", rmpv::Value::F64(0.5)),
            ("ambient_change", rmpv::Value::F64(0.0)),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        assert!((out[5] - 0.6).abs() < 1e-5, "inner_hearing={}", out[5]);
    }

    #[test]
    fn project_inner_mind_15d_audio_creates_saturates_at_5() {
        // creates_recent=10, ambient=0.0 → 0.5*min(1,10/5) + 0.5*0.0 = 0.5 (saturated)
        let audio = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("creates_recent".into()),
                rmpv::Value::F64(10.0),
            ),
            (rmpv::Value::String("ambient".into()), rmpv::Value::F64(0.0)),
        ]);
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("audio_state", audio),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        assert!((out[5] - 0.5).abs() < 1e-5, "inner_hearing={}", out[5]);
    }

    #[test]
    fn project_inner_mind_15d_visual_state_drives_inner_sight() {
        let visual = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("creates_recent".into()),
                rmpv::Value::F64(2.0),
            ),
            (rmpv::Value::String("ambient".into()), rmpv::Value::F64(0.8)),
        ]);
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("visual_state", visual),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        // 0.5*min(1, 2/5) + 0.5*0.8 = 0.2 + 0.4 = 0.6
        assert!((out[7] - 0.6).abs() < 1e-5, "inner_sight={}", out[7]);
    }

    #[test]
    fn project_inner_mind_15d_hormone_levels_drive_willing() {
        let hormones = rmpv::Value::Map(vec![
            (rmpv::Value::String("IMPULSE".into()), rmpv::Value::F64(0.3)),
            (rmpv::Value::String("EMPATHY".into()), rmpv::Value::F64(0.7)),
            (
                rmpv::Value::String("CREATIVITY".into()),
                rmpv::Value::F64(0.55),
            ),
            (
                rmpv::Value::String("VIGILANCE".into()),
                rmpv::Value::F64(0.4),
            ),
            (
                rmpv::Value::String("CURIOSITY".into()),
                rmpv::Value::F64(0.9),
            ),
        ]);
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("hormone_levels", hormones),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        assert!((out[10] - 0.3).abs() < 1e-5);
        assert!((out[11] - 0.7).abs() < 1e-5);
        assert!((out[12] - 0.55).abs() < 1e-5);
        assert!((out[13] - 0.4).abs() < 1e-5);
        assert!((out[14] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn project_inner_mind_15d_clamps_out_of_range() {
        let hormones = rmpv::Value::Map(vec![
            (rmpv::Value::String("IMPULSE".into()), rmpv::Value::F64(1.5)),
            (
                rmpv::Value::String("EMPATHY".into()),
                rmpv::Value::F64(-0.3),
            ),
        ]);
        let payload = encode_msgpack_map(vec![
            (
                "thinking_5d",
                rmpv::Value::Array(vec![rmpv::Value::F32(0.5); 5]),
            ),
            ("interaction_quality", rmpv::Value::F64(2.0)),
            ("assessment_quality", rmpv::Value::F64(-0.5)),
            ("ambient_change", rmpv::Value::F64(1.7)),
            ("hormone_levels", hormones),
        ]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        assert_eq!(out[6], 1.0, "interaction_quality clamped");
        assert_eq!(out[8], 0.0, "assessment_quality clamped");
        assert_eq!(out[9], 1.0, "ambient_change clamped");
        assert_eq!(out[10], 1.0, "IMPULSE clamped to 1");
        assert_eq!(out[11], 0.0, "EMPATHY clamped to 0");
    }

    #[test]
    fn project_inner_mind_15d_legacy_tensor_schema_passes_through() {
        // Atomic-deploy fallback: Python may briefly publish {tensor:[15D]}
        // (legacy v2 shape) during the cutover. Rust must passthrough.
        let tensor: Vec<rmpv::Value> = (0..15).map(|i| rmpv::Value::F32(i as f32 * 0.05)).collect();
        let payload = encode_msgpack_map(vec![("tensor", rmpv::Value::Array(tensor))]);
        let out = project_inner_mind_15d(&payload, [0.0; MIND_DIMS]).unwrap();
        for i in 0..15 {
            assert!(
                (out[i] - (i as f32 * 0.05)).abs() < 1e-5,
                "tensor[{}]={}",
                i,
                out[i]
            );
        }
    }

    #[test]
    fn project_inner_mind_15d_malformed_envelope_returns_fallback() {
        // Top-level Array (not Map) → fallback.
        let payload = {
            let v = rmpv::Value::Array(vec![rmpv::Value::F64(1.0)]);
            let mut out = Vec::new();
            rmpv::encode::write_value(&mut out, &v).unwrap();
            out
        };
        let fallback = [0.123_f32; MIND_DIMS];
        let out = project_inner_mind_15d(&payload, fallback).unwrap();
        for i in 0..15 {
            assert!((out[i] - 0.123).abs() < 1e-6);
        }
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
        let value = encode_mind_state_payload(&mind);
        // §4.C-ter: encoder now returns Value; size-check encodes to bytes inline.
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();
        assert!(
            bytes.len() < 1024,
            "MIND_STATE payload bloated: {}B",
            bytes.len()
        );
    }
}
