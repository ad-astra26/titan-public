//! tick_loop — inner-body 7.83 Hz Schumann tick + state machine.
//!
//! Per SPEC §9.A `titan-inner-body-rs` row + master plan §10.5 chunk C5-2.
//! Logic flow per tick:
//!
//! 1. Read `sensor_cache_inner_body.bin` → 5D raw input (interoception,
//!    proprioception, somatosensation, entropy, thermal).
//! 2. Read `topology_30d.bin[10:20]` → inner_lower 10D (when last
//!    TRINITY_SUBSTRATE_TOPOLOGY_UPDATED was received).
//! 3. Apply UNIFIED + LOCAL filter_down (multipliers from
//!    UNIFIED_SPIRIT_FILTER_DOWN + INNER_SPIRIT_FILTER_DOWN bus events).
//! 4. Apply ground_up nudge to all 5D per G10 (body all dims grounded).
//! 5. Content-hash gate: if payload differs from previous tick, write
//!    `inner_body_5d.bin` via SeqLock + publish BODY_STATE.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{INNER_BODY_FIRING_MAX_BYTES, INNER_BODY_FIRING_SCHEMA_VERSION};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_focus_into_enrichment, compose_multipliers_default,
    decode_filter_down_payload, decode_local_filter_down_payload, encode_body_balance_gift,
    encode_floats, load_checkpoint_for_part, load_restoring_cfg, observe,
    open_focus_input_if_present, open_neuromod_slot_if_present, read_focus_nudge,
    read_neuromod_gain, stateful_update, write_checkpoint_for_part, BalancedPulseEdges,
    CheckpointSnapshot, ContentGate, DriftAggregator, FiringSlotWriter, FocusPart,
    GroundUpEnricher, JourneyAccumulator, JourneyTickInputs, Layer, PulseClockRole, PulseWatcher,
    RestoringCfg, Side, TrinitySide, BODY_BALANCE_GIFT_TOPIC, BODY_GIFT_WEIGHTS, INNER_BODY_TOPICS,
};

/// §G5.2 item 4 checkpoint write cadence. Inner-body ticks @ 7.83 Hz so 80 ticks
/// ≈ 10 s of wall-clock between snapshots — gives the integrator a fresh
/// snapshot at human-reaction cadence without bottlenecking the tick on
/// a same-FS rename(2).
const CHECKPOINT_WRITE_EVERY_N_TICKS: u64 = 80;
/// §G5.2 item 4 checkpoint sidecar basename (matches the part's daemon role).
const CHECKPOINT_PART: &str = "inner_body";

/// Drift threshold as fraction of period — 0.5% per master plan §16 OBS gate.
const DRIFT_THRESHOLD_PCT: f64 = 0.005;

/// Read length for sensor_cache_inner_body — matches `state_registry.py`
/// expected dim count for body input vectors.
const SENSOR_CACHE_DIMS: usize = 5;

/// Boot the daemon's runtime + drive the tick loop until SIGTERM /
/// disconnect.
pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path, data_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "inner-body")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(INNER_BODY_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?INNER_BODY_TOPICS);

    // Open shm slots
    let inner_body_slot = open_slot(shm_dir, "inner_body_5d.bin")?;
    let topology_slot = open_slot(shm_dir, "topology_30d.bin")?;
    // sensor_cache_inner_body.bin is created by Python sidecar
    // (titan_hcl/logic/inner_body_sensor_refresh.py inside body_worker)
    // which spawns LATER than the Rust daemon — hence Option<Slot> at
    // boot, with retry-open in the tick loop. C-S5 closure 2026-05-08.
    let sensor_cache_path = shm_dir.join("sensor_cache_inner_body.bin");
    let sensor_cache = Slot::open(&sensor_cache_path).ok();
    info!(
        event = "SHM_OPENED",
        topology_present = topology_slot.path().exists(),
        sensor_cache_present = sensor_cache.is_some(),
    );

    // Phase C 130D dim-live tracker bridge (rFP §4.7). Lazy-opens
    // inner_body_firing.bin on first tick — slot file is kernel-created.
    // Single-writer per G21: this Rust daemon owns the firing slot post-port;
    // Python DimFiringTracker no-ops under l0_rust_enabled=true.
    let firing_writer = FiringSlotWriter::new(
        "inner_body",
        shm_dir,
        INNER_BODY_FIRING_SCHEMA_VERSION as u32,
        INNER_BODY_FIRING_MAX_BYTES as u32,
    );

    // Send MODULE_READY (P0)
    client
        .publish("MODULE_READY", Some("guardian"), None)
        .await
        .context("publish MODULE_READY")?;
    info!(event = "MODULE_READY_SENT");

    // Shared state — last-received UNIFIED + LOCAL filter_down + topology.
    let state = Arc::new(Mutex::new(DaemonState::default()));
    let shutdown = Arc::new(Notify::new());

    // Spawn bus event dispatcher (runs concurrent with tick loop).
    let dispatcher_state = state.clone();
    let dispatcher_shutdown = shutdown.clone();
    let bus = Arc::new(client);
    let bus_for_dispatcher = bus.clone();
    let dispatcher = tokio::spawn(async move {
        run_event_dispatcher(bus_for_dispatcher, dispatcher_state, dispatcher_shutdown).await;
    });

    // Tick loop drives the 7.83 Hz cadence.
    let tick_result = run_tick_loop(
        bus.clone(),
        state.clone(),
        shutdown.clone(),
        inner_body_slot,
        topology_slot,
        sensor_cache,
        sensor_cache_path,
        firing_writer,
        shm_dir.to_path_buf(),
        data_dir.to_path_buf(),
    )
    .await;

    // Shut down dispatcher cleanly.
    shutdown.notify_waiters();
    let _ = dispatcher.await;
    bus.shutdown().await;

    tick_result
}

/// Per-daemon mutable state — protected by Mutex; tick loop + dispatcher
/// both touch it.
#[derive(Debug, Default)]
struct DaemonState {
    /// Most recent UNIFIED_SPIRIT_FILTER_DOWN.inner_body multipliers.
    /// `None` until first message — daemon uses neutral [1.0; 5] until then.
    unified: Option<[f32; 5]>,
    /// Most recent INNER_SPIRIT_FILTER_DOWN.body multipliers (Phase C
    /// LOCAL cascade — published by inner-spirit-rs after it computes
    /// each tick).
    local: Option<[f32; 5]>,
    /// Most recent topology_lower from TRINITY_SUBSTRATE_TOPOLOGY_UPDATED.
    /// Cached + slot read happens on tick when topology_signaled=true.
    topology_signaled: bool,
    /// Set true on each KERNEL_EPOCH_TICK; consumed (swapped false) by the
    /// tick loop to recompute the ground_up held nudge once per epoch
    /// (SPEC §G5.1 / 0E — the 0.95 damping EMA must evolve at epoch cadence,
    /// NOT per Schumann tick).
    epoch_pending: bool,
    /// Set true when KERNEL_SHUTDOWN_ANNOUNCE arrives.
    shutdown_requested: bool,
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
            // Extract payload from envelope (raw_bytes is the full envelope per BusClient).
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_filter_down_payload(&payload) {
                Ok(p) => {
                    if let Ok(mut s) = state.lock() {
                        s.unified = Some(p.inner_body);
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
                        s.local = Some(p.body);
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
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_tick_loop(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
    mut inner_body_slot: Slot,
    topology_slot: Slot,
    mut sensor_cache: Option<Slot>,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
    shm_dir: std::path::PathBuf,
    data_dir: std::path::PathBuf,
) -> Result<()> {
    let mut content_gate = ContentGate::new();
    let mut ground_up = GroundUpEnricher::new(Side::Body);
    let mut drift_agg = DriftAggregator::new("body", DRIFT_THRESHOLD_PCT);
    // §G5.2 item 4 — restore the traveling tensor's exact position from
    // the per-part checkpoint on boot (Maker override 2026-05-23: full P0
    // closure, no deferral). Cold-start at the 0.5 Divine Centre only when
    // the sidecar is absent / invalid / version-mismatched. The last
    // observable signature is also restored so the very first stateful_update
    // tick sees the same gradient-weighted gains it had pre-restart.
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
    // §G5.2 item 5 — gains sourced from titan_params.toml [trinity_restoring]
    // via the Python L2-published sidecar `trinity_restoring.bin`. Falls back
    // to crate DEFAULT_* constants if sidecar absent. Gradient is fixed per
    // INV-9 (Body 0.7/0.3 quant/qual).
    let mut cfg = load_restoring_cfg(&shm_dir, Layer::Body);
    // §G5.2 item 2 — `k_restore` is scaled by the live neuromod gain. Open
    // the slot now (Python L2 may have already populated it) and retry
    // periodically below if absent (sidecar arrives after daemon boot).
    let mut neuromod_slot = open_neuromod_slot_if_present(&shm_dir);
    let neuromod_path = shm_dir.join("neuromod_state.bin");
    // §G5.2 item 2 + §G12 — FOCUS cascade nudge slot. Read each tick + compose
    // into enrichment_force; daemon NEVER writes (G21/INV-4 — sole writer is
    // Python L2's FocusPIDPublisher).
    let mut focus_input_slot = open_focus_input_if_present(&shm_dir);
    let focus_input_path = shm_dir.join("focus_input.bin");

    // P0.5 / D-SPEC-131 §G5.1 UP-leg balance-gift state. SHM-direct
    // PulseWatcher reads `sphere_clocks.bin` per tick to detect the inner_body
    // clock's balanced rising-edge — that's when the JourneyAccumulator
    // finalises one cycle's BODY_BALANCE_GIFT digest, publishes it to spirit,
    // and resets for the next cycle. First cycle after boot is suppressed
    // by the accumulator (PLAN §6.5.2).
    let mut pulse_watcher = PulseWatcher::open(&shm_dir);
    let mut journey_acc: JourneyAccumulator<5> = JourneyAccumulator::new();

    // Per master plan §7 + C-S3 PLAN §1.1 #2: Schumann timer wheels live in
    // titan-schumann (substrate-owned shared library). Pinning epoch_t0 to
    // tokio::time::Instant::now() at daemon boot — phase relations across
    // body/mind/spirit are coordinated at the substrate level when all 6
    // daemons share a parent unified-spirit-rs that propagates a common t0
    // (C-S4 deliverable). For now each daemon uses its own t0 — drift
    // measurement remains per-daemon.
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Body, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    info!(event = "TICK_LOOP_START", role = "body", period_ns);

    // Retry-open sensor cache every ~1s when None — Python sidecar
    // (inner_body_sensor_refresh in body_worker) starts AFTER the Rust
    // daemon, so the slot file may not exist at boot. Once open, stays
    // open. C-S5 closure 2026-05-08.
    let mut tick_count: u64 = 0;
    let retry_every_n: u64 = (1.0_f64 / 0.1277).ceil() as u64; // ~8 ticks ≈ 1s at 7.83 Hz

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                info!(event = "TICK_LOOP_SHUTDOWN_REQUESTED");
                break;
            }
            maybe_tick = tick_rx.recv() => match maybe_tick {
                Some(tick_event) => {
                    // Retry sensor_cache open if still None.
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
                    // Retry neuromod_state.bin open if Python L2 hadn't created it at boot.
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
                    // Refresh cfg from sidecar at the same retry cadence so a Python
                    // L2 re-publish (config reload) becomes visible without restart.
                    if tick_count.is_multiple_of(retry_every_n) {
                        cfg = load_restoring_cfg(&shm_dir, Layer::Body);
                    }
                    // P0.5 / D-SPEC-131 SHM-direct pulse-edge read. The balanced
                    // edge on the inner_body clock is what arms this daemon's
                    // BODY_BALANCE_GIFT emission (PLAN §6.5).
                    let (_pulse_edges, balanced_pulse_edges) = pulse_watcher.tick_with_balanced();
                    let drift_pct = tick_event.jitter_ns() as f64 / tick_event.period_ns as f64;
                    drift_agg.observe(drift_pct, tick_event.jitter_ns(), tick_event.epoch);
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut ground_up,
                        &mut inner_body_slot, &topology_slot, &sensor_cache,
                        &mut firing_writer, &mut prev, &mut prev2,
                        &mut cfg, neuromod_slot.as_ref(), focus_input_slot.as_ref(),
                        &mut last_obs_restored,
                        &mut journey_acc, &balanced_pulse_edges,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
                    }
                    tick_count = tick_count.wrapping_add(1);
                    // §G5.2 item 4 — periodic checkpoint write. Use the
                    // per-tick `last_obs_restored` (now populated from the
                    // most-recent `observe(prev, prev2)`) to preserve the
                    // 5D signature across restart.
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
    // Final checkpoint on clean shutdown so a planned restart resumes from
    // the latest tick (not from the periodic snapshot up to ~10 s ago).
    if let Some(o) = last_obs_restored.as_ref() {
        if let Err(e) = write_checkpoint_for_part::<5>(&data_dir, CHECKPOINT_PART, &prev, &prev2, o)
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
    inner_body_slot: &mut Slot,
    topology_slot: &Slot,
    sensor_cache: &Option<Slot>,
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
    // 1. Read sensor cache (raw 5D body input). This is `raw[t]` per §G5.2 —
    //    the un-enriched producer output that the spring/drive integrate.
    let raw = read_sensor_cache(sensor_cache)?;
    let mut body = raw;

    // 2. Snapshot bus state (filter_down multipliers + topology flag +
    //    consume the epoch-pending edge for 0E ground_up recompute).
    let (unified_mult, local_mult, topology_fresh, epoch_due) = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        let epoch_due = s.epoch_pending;
        s.epoch_pending = false;
        // D-SPEC-121 (v1.54.0): one-shot consume-and-clear of filter_down
        // multipliers — applied once on this tick, then held value returns to
        // None until the next *_SPIRIT_FILTER_DOWN event arrives. Supersedes
        // the v1.36.2 R1 "held + applied per tick" clause that, combined with
        // the §G5.2 stateful integrator, was saturating high-raw dims at 0.
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
        (None, None) => vec![1.0_f32; 5], // neutral until first publish arrives
    };

    // 4. Apply filter_down multipliers to body[0:5]
    apply_multipliers(&mut body, &composed);

    // 5. ground_up — 0E held-nudge model (SPEC §G5.1, D-SPEC-97 refinement):
    //    RECOMPUTE the damped nudge ONCE per kernel epoch (so the 0.95 EMA +
    //    ±0.05/epoch clamp evolve at epoch cadence, not 7.83 Hz), then APPLY
    //    the held nudge every Schumann tick so the grounding offset manifests
    //    continuously in the per-tick-recomputed body tensor (no compounding —
    //    body is rebuilt from raw each tick).
    if topology_fresh {
        if epoch_due {
            if let Ok(topology_lower) =
                titan_trinity_daemon::read_topology_inner_lower(topology_slot)
            {
                ground_up.compute_nudge(&topology_lower);
            }
        }
        ground_up.apply_held_to_body(&mut body, 1.0)?;
    }

    // 5b. §G5.2 traveling-tensor update.
    //   - raw       = the un-enriched sensor output (already snapshotted as `raw`).
    //   - enrichment = the full enrichment delta the producer pipeline added:
    //                  filter_down (UNIFIED ⊗ LOCAL) + ground_up. We compute it
    //                  element-wise as (enriched − raw) so the kernel can apply
    //                  it as a SEPARATE full-weight additive term per the
    //                  §G5.2 equation literal (G5.2-enrichment-separate clause).
    //   - obs(prev, prev2) feeds all 5 observables into the kernel; the per-tick
    //                  neuromod gain reads from `neuromod_state.bin` and
    //                  multiplies the spring per §G5.2 item 2 (G5.2-neuromod-gain).
    let mut enrichment = [0.0_f32; 5];
    for i in 0..5 {
        enrichment[i] = body[i] - raw[i];
    }
    // §G12 FOCUS cascade: amplified nudge from the FocusPIDPublisher composes
    // into enrichment_force as another full-weight additive (still SEPARATE
    // from drive per G5.2-enrichment-separate). When the layer is STALE the
    // stale_focus_multiplier amplifies SPIRIT→Lower-Spirit→Mind→Body.
    let focus = read_focus_nudge::<5>(focus_input_slot, FocusPart::InnerBody);
    compose_focus_into_enrichment(&mut enrichment, &focus);
    cfg.neuromod_gain = read_neuromod_gain(neuromod_slot);
    let obs = observe(&prev[..], &prev2[..]);
    // Cache the latest observable signature so the checkpoint writer can
    // persist it (§G5.2 item 4: tensor + observables MUST checkpoint together).
    *last_obs_restored = Some(obs);
    let x = stateful_update(&prev[..], &prev2[..], &raw[..], &enrichment[..], &obs, cfg);
    let mut body_state = [0.0_f32; 5];
    body_state.copy_from_slice(&x[..5]);
    *prev2 = *prev;
    *prev = body_state;
    let body = body_state; // the traveling state is what we publish

    // 5c. P0.5 / D-SPEC-131 §G5.1 UP-leg: track this tick's journey + emit a
    //     BODY_BALANCE_GIFT on this daemon's own clock balanced rising-edge.
    //     PulseWatcher already read (edges, balanced_edges) before this tick;
    //     the journey accumulator tracks the post-§G5.2 traveling tensor.
    let tick_ts = now_secs();
    journey_acc.tick(JourneyTickInputs {
        x: &body,
        obs,
        now_secs: tick_ts as f32,
    });
    if balanced_pulse_edges[PulseClockRole::InnerBody.index()] {
        journey_acc.mark_balanced(obs);
        if let Some(digest) = journey_acc.finalize_body_gift(&BODY_GIFT_WEIGHTS) {
            let payload = encode_body_balance_gift::<5>(TrinitySide::Inner, &digest, tick_ts);
            if let Err(e) = bus
                .publish(BODY_BALANCE_GIFT_TOPIC, Some("all"), Some(payload))
                .await
            {
                warn!(err = ?e, "publish BODY_BALANCE_GIFT failed (continuing)");
            } else {
                debug!(
                    event = "BODY_BALANCE_GIFT_EMITTED",
                    side = "inner",
                    amplitude = digest.gift_amplitude,
                    cycle_s = digest.cycle_duration_s,
                    ticks = digest.cycle_tick_count,
                );
            }
        }
        journey_acc.reset_for_next_cycle();
    }

    // 6. Encode payload bytes.
    let bytes = encode_floats::<5>(&body);

    // 7. Content-hash gate the slot write.
    if content_gate.should_write(&bytes) {
        inner_body_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // 8. Update inner_body_firing.bin diagnostic slot (rFP §4.7).
    //    Always record (matches BODY_STATE bus publish — diagnostic that
    //    the daemon ticked, regardless of content-gate suppression).
    //    inputs_state = &[] until inner sensor_cache schema migrates to
    //    source-dict in P4 (rFP §4.4) — until then dim-live can classify
    //    ALIVE/SILENT but not PARTIAL/ALIVE_AT_DEFAULT for inner_body.
    firing_writer.record_tick(&body, &[], now_secs());

    // 9. Publish BODY_STATE (P1) with values + ts. Always publish (even
    //    when slot write was suppressed) so consumers see the cadence.
    let payload = encode_body_state_payload(&body);
    bus.publish("BODY_STATE", Some("all"), Some(payload))
        .await
        .map_err(|e| anyhow!("publish BODY_STATE: {e}"))?;

    Ok(())
}

fn read_sensor_cache(sensor_cache: &Option<Slot>) -> Result<[f32; 5]> {
    match sensor_cache {
        Some(slot) => {
            let raw = slot.read().map_err(|e| anyhow!("sensor_cache read: {e}"))?;
            // Step 7 §4.4 schema migration v1→v2 (rFP_phase_c_130d_rust_l1_port):
            // sensor_cache_inner_body.bin now carries msgpack source-dict
            // {senses: {<name>: {value, severity, velocity}, ...}, critical_threshold}
            // for Rust-native formula execution per SPEC §23.4. Falls back to
            // legacy 5-float32-LE layout if msgpack decode fails (e.g. Python
            // sensor_refresh deployed pre-bump).
            //
            // Format detection: msgpack source dict starts with map-marker byte
            // (0x80-0x8f for fixmap, 0xde for map16, 0xdf for map32). 5*float32
            // = 20 bytes raw with no map prefix.
            let is_msgpack = !raw.is_empty() && matches!(raw[0], 0x80..=0x8f | 0xde | 0xdf);
            if is_msgpack {
                project_inner_body_5d(&raw).or_else(|_| Ok([0.5_f32; 5]))
            } else {
                // Legacy float32 LE — Phase A+B compatibility / pre-migration
                // T3 deployments. Decode 5×f32 from first 20 bytes.
                let mut out = [0.0_f32; 5];
                for i in 0..SENSOR_CACHE_DIMS.min(raw.len() / 4) {
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&raw[i * 4..i * 4 + 4]);
                    out[i] = f32::from_le_bytes(buf);
                }
                Ok(out)
            }
        }
        None => Ok([0.0; 5]), // sensor cache not yet populated by Python sensor refresh
    }
}

/// SPEC §23.4 inner_body 5D Rust-native formula (Step 7 §4.4 P4):
/// decodes msgpack source dict produced by `body_worker._provide_body_source_dict`
/// (per-sense {value, severity, velocity}) and applies the urgency-weighted
/// health-score formula:
///
///   urgency = min(1, raw * sev / CRITICAL + |vel| * 0.3)
///   health = max(0, 1 - urgency)
///
/// FILTER_DOWN multipliers + GROUND_UP nudges + FOCUS bias are applied in
/// the substrate processing pipeline downstream (run_one_tick steps 4-5),
/// not here. Sense interpretation (the OS reads inside `_sense_*`) stays
/// in Python because /proc/meminfo etc. are platform-dependent.
///
/// Per-sense order: [interoception, proprioception, somatosensation,
/// entropy, thermal] per SPEC §23.4 + body_worker.py:_collect_body_tensor.
/// Missing senses → neutral default 0.5.
fn project_inner_body_5d(payload: &[u8]) -> Result<[f32; 5]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Err(anyhow!("source dict not a map")),
    };
    // Lookup `critical_threshold` (defaults to 10.0 = Severity.CRITICAL.value).
    let critical_threshold: f64 = lookup_f64_or(map, "critical_threshold", 10.0);
    // Lookup `senses` sub-map.
    let senses_map = match lookup_value(map, "senses") {
        Some(Value::Map(items)) => items,
        _ => return Ok([0.5_f32; 5]), // no senses → neutral
    };
    let sense_names = [
        "interoception",
        "proprioception",
        "somatosensation",
        "entropy",
        "thermal",
    ];
    let mut out = [0.5_f32; 5];
    for (i, name) in sense_names.iter().enumerate() {
        let sense = match lookup_value(senses_map, name) {
            Some(Value::Map(items)) => items,
            _ => continue, // missing sense → keep neutral default
        };
        let raw_value = lookup_f64_or(sense, "value", 0.5);
        let severity = lookup_f64_or(sense, "severity", 1.0); // Severity.INFO.value
        let velocity = lookup_f64_or(sense, "velocity", 0.0);
        let urgency = (raw_value * severity / critical_threshold + velocity.abs() * 0.3).min(1.0);
        let health = (1.0 - urgency).max(0.0);
        out[i] = health as f32;
    }
    Ok(out)
}

// Helper: lookup a key in a msgpack map, return its Value or None.
fn lookup_value<'a>(map: &'a [(rmpv::Value, rmpv::Value)], key: &str) -> Option<&'a rmpv::Value> {
    for (k, v) in map.iter() {
        if let rmpv::Value::String(s) = k {
            if s.as_str() == Some(key) {
                return Some(v);
            }
        }
    }
    None
}

// Helper: lookup a key whose value is numeric, return f64 or default.
fn lookup_f64_or(map: &[(rmpv::Value, rmpv::Value)], key: &str, default: f64) -> f64 {
    match lookup_value(map, key) {
        Some(v) => v.as_f64().unwrap_or(default),
        None => default,
    }
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

/// Build a SPEC §8.5 `BODY_STATE` payload as `rmpv::Value::Map`. Embedded
/// directly into the envelope by `encode_simple` per SPEC §8.10 line 900
/// byte-identical guarantee — NOT pre-encoded to opaque bytes.
fn encode_body_state_payload(body: &[f32; 5]) -> rmpv::Value {
    use rmpv::Value;
    let values = Value::Array(body.iter().map(|f| Value::F64(*f as f64)).collect());
    Value::Map(vec![
        (Value::String("src".into()), Value::String("inner".into())),
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

#[allow(dead_code)]
const TICK_PERIOD_S: Duration = Duration::from_micros(127_714); // 1/7.83 ≈ 127.7 ms

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use titan_core::constants::INNER_BODY_5D_SCHEMA_VERSION;

    /// Helper: pure-compute "what does run_one_tick produce given inputs"
    /// — extracted so unit tests can exercise the transformation pipeline
    /// without spinning up tokio + bus.
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
        // 0.5 * 2.0 = 1.0 (clipped), 0.5 * 0.5 = 0.25, etc.
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
    fn unified_and_local_compose() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        // 2.0 (unified) * 1.5 (local) = 3.0 → clipped at MULTIPLIER_CEIL=3.0
        let unified = [2.0, 1.0, 1.0, 1.0, 1.0];
        let local = [1.5, 1.0, 1.0, 1.0, 1.0];
        let out = pure_compute(raw, Some(unified), Some(local), None, &mut g);
        // body[0] = 0.5 * 3.0 = 1.5 → clipped at TENSOR_MAX=1.0
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ground_up_applied_when_topology_fresh() {
        let mut g = GroundUpEnricher::new(Side::Body);
        let raw = [0.5; 5];
        let topo: [f32; 10] = [0.04, 0.04, 0.04, 0.04, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = pure_compute(raw, None, None, Some(topo), &mut g);
        // Body all 5D nudged: delta = 0.002 * 0.1 * 1.0 = 0.0002 each
        for &v in &out[..5] {
            assert!((v - 0.5002).abs() < 1e-5);
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
        // Must decode successfully + contain expected keys.
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
    fn body_payload_src_is_inner() {
        let body = [0.0; 5];
        let bytes = encode_body_state_payload(&body);
        use rmpv::Value;
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("src") {
                        assert_eq!(val, Value::String("inner".into()));
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
        let path = dir.path().join("inner_body_5d.bin");
        let mut slot = Slot::create(&path, INNER_BODY_5D_SCHEMA_VERSION as u32, 20).unwrap();
        let body: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes = encode_floats::<5>(&body);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn read_sensor_cache_handles_none() {
        let r = read_sensor_cache(&None).unwrap();
        assert_eq!(r, [0.0; 5]);
    }

    #[test]
    fn read_sensor_cache_decodes_5_floats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sensor_cache_inner_body.bin");
        let mut slot = Slot::create(&path, 1, 20).unwrap();
        let raw: [f32; 5] = [0.11, 0.22, 0.33, 0.44, 0.55];
        let bytes = encode_floats::<5>(&raw);
        slot.write(&bytes).unwrap();
        let r = read_sensor_cache(&Some(slot)).unwrap();
        for i in 0..5 {
            assert!((r[i] - raw[i]).abs() < 1e-6);
        }
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
    fn schumann_role_body_hz_is_locked() {
        // G13 LOCKED: 7.83 Hz biological constant. Sourced from
        // titan-schumann (canonical Schumann timer-wheel library per
        // master plan §7 + C-S3 PLAN §1.1 #2). NOT a daemon-local constant.
        assert_eq!(SchumannRole::Body.hz(), 7.83);
    }
}
