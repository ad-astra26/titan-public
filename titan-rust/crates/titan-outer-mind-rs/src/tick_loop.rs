//! tick_loop — outer-mind Schumann-locked tick (23.49 Hz) + state machine.
//!
//! Per SPEC §9.A `titan-outer-mind-rs` row + §18.1 + master plan §10.6 C6-4.
//! Logic flow per tick:
//!
//! 1. Read `sensor_cache_outer_mind.bin` → msgpack-decode → project to 15D.
//! 2. Read `topology_30d.bin[0:10]` → outer_lower 10D (when topology fresh).
//! 3. Apply UNIFIED + LOCAL filter_down to all 15D (G7 [0.3, 3.0] clamp).
//! 4. Apply ground_up to **willing[10:15] ONLY** per SPEC G10
//!    (`ground_up_skip_mind_thinking=0:5`, `ground_up_skip_mind_feeling=5:10`).
//! 5. Content-hash gate; write outer_mind_15d.bin + publish MIND_STATE.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{
    OUTER_MIND_BUS_PUBLISH_INTERVAL_S, OUTER_MIND_FIRING_MAX_BYTES,
    OUTER_MIND_FIRING_SCHEMA_VERSION, OUTER_MIND_TICK_BASE_S,
};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_multipliers_default, decode_local_filter_down_payload,
    encode_floats, observe, read_dim_slice, read_sensor_cache, read_topology_outer_lower,
    stateful_update, ContentGate, FiringSlotWriter, GroundUpEnricher, Layer, PublishThrottle,
    RestoringCfg, SensorCacheRead, Side, OUTER_MIND_TOPICS,
};

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "outer-mind")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(OUTER_MIND_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?OUTER_MIND_TOPICS);

    let outer_mind_slot = open_slot(shm_dir, "outer_mind_15d.bin")?;
    let topology_slot = open_slot(shm_dir, "topology_30d.bin")?;
    let outer_body_slot = open_slot(shm_dir, "outer_body_5d.bin")?;
    let sensor_cache_path = shm_dir.join("sensor_cache_outer_mind.bin");
    info!(
        event = "SHM_OPENED",
        topology_present = topology_slot.path().exists(),
        outer_body_present = outer_body_slot.path().exists(),
    );

    // Phase C 130D dim-live tracker bridge (rFP §4.7).
    let firing_writer = FiringSlotWriter::new(
        "outer_mind",
        shm_dir,
        OUTER_MIND_FIRING_SCHEMA_VERSION as u32,
        OUTER_MIND_FIRING_MAX_BYTES as u32,
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
        outer_mind_slot,
        topology_slot,
        outer_body_slot,
        sensor_cache_path,
        firing_writer,
    )
    .await;

    shutdown.notify_waiters();
    let _ = dispatcher.await;
    bus.shutdown().await;

    tick_result
}

#[derive(Debug)]
struct DaemonState {
    unified: Option<[f32; 15]>,
    local: Option<[f32; 15]>,
    topology_signaled: bool,
    /// Set true on each KERNEL_EPOCH_TICK; consumed by the tick loop to
    /// recompute the ground_up held nudge once per epoch (SPEC §G5.1 / 0E).
    epoch_pending: bool,
    shutdown_requested: bool,
    last_mind: [f32; 15],
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            unified: None,
            local: None,
            topology_signaled: false,
            epoch_pending: false,
            shutdown_requested: false,
            last_mind: [0.5; 15],
        }
    }
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
            match decode_unified_outer_mind(&payload) {
                Ok(mults) => {
                    if let Ok(mut s) = state.lock() {
                        s.unified = Some(mults);
                    }
                }
                Err(e) => warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN.outer_mind failed"),
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
                        s.local = Some(p.mind);
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

/// Decode the `outer_mind` slice (length 15) from a structured
/// UNIFIED_SPIRIT_FILTER_DOWN payload. SPEC §8.2 line 789 + §8.10 line 900:
/// payload is `rmpv::Value::Map` per the §8.6 schema. Closure of
/// `rFP_worker_broadcast_topics_completion §4.C-ter` (2026-05-13).
fn decode_unified_outer_mind(payload: &rmpv::Value) -> Result<[f32; 15]> {
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
    let mut outer_mind = [1.0_f32; 15];
    let mut found = false;
    for (k, val) in mults_map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("outer_mind") {
                decode_float_array_into(val, &mut outer_mind)?;
                found = true;
                break;
            }
        }
    }
    if !found {
        return Err(anyhow!("multipliers.outer_mind missing"));
    }
    Ok(outer_mind)
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
    mut outer_mind_slot: Slot,
    topology_slot: Slot,
    outer_body_slot: Slot,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
) -> Result<()> {
    // Post-A.S8 D2 cadence migration (rFP §4.2): Schumann mind (23.49 Hz)
    // tick + bus publish throttled to OUTER_MIND_BUS_PUBLISH_INTERVAL_S.
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Mind, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    let mut content_gate = ContentGate::new();
    let mut ground_up = GroundUpEnricher::new(Side::MindWilling);
    let mut publish_throttle = PublishThrottle::new(OUTER_MIND_BUS_PUBLISH_INTERVAL_S);
    // §G5.2 traveling-tensor state (x[t-1], x[t-2]); cold-start at 0.5 centre.
    let mut prev: [f32; 15] = [0.5; 15];
    let mut prev2: [f32; 15] = [0.5; 15];

    info!(
        event = "TICK_LOOP_START",
        role = "outer-mind",
        period_ns,
        publish_interval_s = OUTER_MIND_BUS_PUBLISH_INTERVAL_S,
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
                        event = "OUTER_MIND_TICK",
                        epoch = tick_event.epoch,
                        period_ns = tick_event.period_ns,
                    );
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut ground_up, &mut publish_throttle,
                        &mut outer_mind_slot, &topology_slot, &outer_body_slot,
                        &sensor_cache_path, &mut firing_writer, &mut prev, &mut prev2,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
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
    Ok(())
}

fn outer_mind_stale_threshold_s() -> f64 {
    OUTER_MIND_TICK_BASE_S * 3.0
}

#[allow(clippy::too_many_arguments)]
async fn run_one_tick(
    bus: &Arc<BusClient>,
    state: &Arc<Mutex<DaemonState>>,
    content_gate: &mut ContentGate,
    ground_up: &mut GroundUpEnricher,
    publish_throttle: &mut PublishThrottle,
    outer_mind_slot: &mut Slot,
    topology_slot: &Slot,
    outer_body_slot: &Slot,
    sensor_cache_path: &Path,
    firing_writer: &mut FiringSlotWriter,
    prev: &mut [f32; 15],
    prev2: &mut [f32; 15],
) -> Result<()> {
    let last_mind = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.last_mind
    };
    // Read outer_body_5d.bin (cross-daemon dependency for feeling[2,3]).
    // On miss, default to neutral [0.5; 5] — daemon still produces sane mind.
    let outer_body_5d = read_dim_slice::<5>(outer_body_slot).unwrap_or([0.5_f32; 5]);
    let mut mind = read_outer_mind_from_cache(sensor_cache_path, last_mind, outer_body_5d);

    let (unified_mult, local_mult, topology_fresh, epoch_due) = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        let epoch_due = s.epoch_pending;
        s.epoch_pending = false;
        (s.unified, s.local, s.topology_signaled, epoch_due)
    };

    let composed: Vec<f32> = match (unified_mult, local_mult) {
        (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
        (Some(u), None) => u.to_vec(),
        (None, Some(l)) => l.to_vec(),
        (None, None) => vec![1.0_f32; 15],
    };

    apply_multipliers(&mut mind, &composed);

    // Per SPEC G10: ground_up applies to willing[10:15] ONLY.
    // Thinking[0:5] + Feeling[5:10] are NEVER grounded.
    // 0E held-nudge model (SPEC §G5.1): RECOMPUTE the damped nudge ONCE per
    // kernel epoch, then APPLY the held nudge every tick (continuous,
    // no compounding — mind is rebuilt from raw each tick).
    if topology_fresh {
        if epoch_due {
            if let Ok(topology_lower) = read_topology_outer_lower(topology_slot) {
                ground_up.compute_nudge(&topology_lower);
            }
        }
        ground_up.apply_held_to_mind(&mut mind, 1.0)?;
    }

    if let Ok(mut s) = state.lock() {
        s.last_mind = mind;
    }

    // §G5.2 traveling-tensor update (Layer::Mind). `last_mind` keeps the
    // producer/drive value for stale fallback; `x` (traveling state) is written.
    let cfg = RestoringCfg::for_layer(Layer::Mind);
    // P0-0b kernel signature (§G5.2 ratified): see inner-body for design notes.
    let obs = observe(&prev[..], &prev2[..]);
    let enrichment_zero = [0.0_f32; 15];
    let x = stateful_update(
        &prev[..],
        &prev2[..],
        &mind[..],
        &enrichment_zero[..],
        &obs,
        &cfg,
    );
    let mut mind_state = [0.0_f32; 15];
    mind_state.copy_from_slice(&x[..15]);
    *prev2 = *prev;
    *prev = mind_state;
    let mind = mind_state;

    let bytes = encode_floats::<15>(&mind);
    if content_gate.should_write(&bytes) {
        outer_mind_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // Phase C dim-live tracker bridge (rFP §4.7) — record per tick at
    // full Schumann mind cadence (independent of bus publish throttle).
    firing_writer.record_tick(&mind, &[], now_secs());

    // Bus publish throttled per OUTER_MIND_BUS_PUBLISH_INTERVAL_S (post-A.S8
    // D2 — rFP §4.2). Tick fires at Schumann mind (23.49 Hz) but
    // MIND_STATE publishes only every ~15s. Slot writes (above) remain at
    // full tick cadence under content-hash gating.
    if publish_throttle.should_publish() {
        let payload = encode_mind_state_payload(&mind);
        bus.publish("MIND_STATE", Some("all"), Some(payload))
            .await
            .map_err(|e| anyhow!("publish MIND_STATE: {e}"))?;
    }

    Ok(())
}

fn read_outer_mind_from_cache(
    path: &Path,
    last_mind: [f32; 15],
    outer_body: [f32; 5],
) -> [f32; 15] {
    match read_sensor_cache(path, outer_mind_stale_threshold_s()) {
        Ok(SensorCacheRead::Fresh { payload, .. }) => {
            match project_outer_mind_15d(&payload, outer_body) {
                Ok(mind) => mind,
                Err(e) => {
                    warn!(err = ?e, "outer_mind project failed; using last-known");
                    last_mind
                }
            }
        }
        Ok(SensorCacheRead::Stale { age_s, .. }) => {
            warn!(
                event = "SENSOR_CACHE_STALE",
                age_s,
                threshold_s = outer_mind_stale_threshold_s(),
                confidence = 0.0,
            );
            last_mind
        }
        Ok(SensorCacheRead::Missing) => last_mind,
        Err(e) => {
            warn!(err = ?e, "sensor_cache read errored");
            last_mind
        }
    }
}

// ── V6 outer_mind_15d full port ────────────────────────────────────
//
// Byte-faithful port of `titan_hcl/logic/outer_mind_tensor.py::collect_outer_mind_15d`
// (lines 35-173) wrapping `titan_hcl/logic/outer_trinity.py::_collect_extended`
// preprocessing (lines 592-712 — the subset that feeds the 15D mind formula).
//
// Closes rFP_phase_c_close_all_runtime_gaps chunk 9I — supersedes the
// stub-port that read only `soul_health` (dim[0]) and `uptime_seconds`
// (dim[1]). Per Prime Directive #1 "if a function exists, it MUST do
// the work its name claims".
//
// **Sidecar contract**: `outer_mind_sensor_refresh.py:SOURCE_KEYS` writes
// the raw upstream source dict (16 keys including `agency_stats`,
// `assessment_stats`, `memory_status`, `social_perception_stats`,
// `twin_state`, `anchor_state`, `bus_stats`, ...). Per Q3 of the Day 1
// session-handoff, `_collect_extended` preprocessing is inlined here
// (vs extracted to a shared crate) — kept per-daemon until 9J reveals
// what's actually shared.
//
// Parity bar: |Δ| < 1e-3 (matches 9G outer_body convention; Python's
// `_clamp` does NOT round to 4 decimals, so `round4_f32` is NOT used
// here — output is the natural f32 cast of the f64 formula).

/// V6 15D outer-mind projection. Pure compute over the msgpack source
/// dict from the Python sidecar plus the freshly-read `outer_body_5d`
/// (consumed by feeling[2] / feeling[3] per `outer_mind_tensor.py:124,134,136`).
///
/// Returns `Err` only when the msgpack envelope is fundamentally malformed
/// (not a map). Missing individual fields contribute their documented
/// neutral defaults (0.5 / 0.0 per Python's `.get(key, default)` calls).
pub fn project_outer_mind_15d(payload: &[u8], outer_body: [f32; 5]) -> Result<[f32; 15]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Err(anyhow!("source dict not a map")),
    };

    // ── Step 1: Read raw upstream stats via SOURCE_KEYS lookups. ──
    let agency = lookup_map(map, "agency_stats");
    let assessment = lookup_map(map, "assessment_stats");
    let memory_status = lookup_map(map, "memory_status");
    let social_perception = lookup_map(map, "social_perception_stats");
    let twin_state = lookup_map(map, "twin_state");
    let anchor_state = lookup_map(map, "anchor_state");
    let bus_stats = lookup_map(map, "bus_stats");
    // Step 5 §4.2 P2 additions for redesigned thinking + willing dims:
    let meta_cgn = lookup_map(map, "meta_cgn_stats");
    let cgn = lookup_map(map, "cgn_stats");
    let memory_stats = lookup_map(map, "memory_stats");
    let events_teacher = lookup_map(map, "events_teacher_stats");
    let social_x_gateway = lookup_map(map, "social_x_gateway_stats");

    // ── Step 2: Preprocess into derived stats ─
    // Mirrors `_collect_extended` lines 607-694 — only the subset
    // consumed by the 15D mind formula.

    // action_stats — derived from agency_stats (lines 608-622)
    let total_actions: f64 = field_or_default(agency.as_ref(), "total_actions", 0.0);
    let failed_actions: f64 = field_or_default(agency.as_ref(), "failed_actions", 0.0);
    // Python: success_rate = (total - failed) / max(1, total)
    let action_success_rate: f64 = (total_actions - failed_actions) / total_actions.max(1.0);
    let action_per_window: f64 = field_or_default(agency.as_ref(), "actions_this_hour", 0.0);
    // SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a (2026-05-18) — 24h smoothing
    // variant feeds willing[10] action_throughput with divisor /240 (was
    // per_hour/10 → per_day/(10*24)). Eliminates bursty-window misses for
    // active Titans whose actions cluster around quiet hours.
    let action_per_day: f64 = field_or_default(agency.as_ref(), "actions_this_day", 0.0);

    // creative_stats — derived (lines 624-653); only `per_window` consumed
    let creative_per_window: f64 = field_or_default(agency.as_ref(), "creative_this_hour", 0.0);
    // D-SPEC-87 24h smoothing for willing[12] creative_output (divisor /120).
    let creative_per_day: f64 = field_or_default(agency.as_ref(), "creative_this_day", 0.0);

    // guardian_stats — derived (lines 656-661); only `rejections_per_window`
    let rejections_per_window: f64 = field_or_default(agency.as_ref(), "rejections_this_hour", 0.0);

    // social_stats — derived (lines 664-672)
    let interactions_per_window: f64 =
        field_or_default(memory_status.as_ref(), "unique_interactors", 0.0);
    let sentiment_avg: f64 = field_or_default(social_perception.as_ref(), "sentiment_ema", 0.5);
    // soc.outputs_per_window is NOT set by _collect_extended → defaults to 0
    // (faithful to current Python runtime — willing[1] always 0 today).
    let social_outputs_per_window: f64 = 0.0;

    // research_stats — fixed defaults per _collect_extended lines 675-680
    let research_queries: f64 = field_or_default(memory_status.as_ref(), "research_nodes", 0.0);
    let research_usage_rate: f64 = 0.5;
    let research_seconds_since_last: f64 = 300.0;
    let research_queries_per_window: f64 = 0.0;

    // assessment_stats — derived (lines 683-688)
    let assessment_mean: f64 = field_or_default(assessment.as_ref(), "average_score", 0.5);

    // ── Step 3: Compute 15D tensor ─
    // Python `_clamp` defaults NaN→0.5 then clamps to [0,1]. `safe_clamp`
    // here matches that.

    // ── THINKING (5D) — SPEC §23.8 thinking[0,1,2] REDESIGNED 2026-05-07 ──
    let mut thinking = [0.5_f64; 5];

    // [0] research_effectiveness — SPEC §23.8 REDESIGNED:
    // 0.4*meta_cgn.knowledge_helpful_ratio + 0.3*cgn.avg_reward_norm
    // + 0.3*memory.directive_alignment
    let mc_knowledge_helpful: f64 =
        field_or_default(meta_cgn.as_ref(), "knowledge_helpful_ratio", 0.5);
    let cgn_avg_reward_norm: f64 = field_or_default(cgn.as_ref(), "avg_reward_norm", 0.5);
    let memory_directive_alignment: f64 =
        field_or_default(memory_stats.as_ref(), "directive_alignment", 0.5);
    thinking[0] = safe_clamp(
        0.4 * mc_knowledge_helpful + 0.3 * cgn_avg_reward_norm + 0.3 * memory_directive_alignment,
    );

    // [1] knowledge_retrieval — SPEC §23.8 REDESIGNED:
    // 0.35*memory.directive_alignment + 0.25*meta_cgn.knowledge_helpful_ratio
    // + 0.20*vocab.avg_confidence + 0.20*(1 - meta_cgn.usage_gini)
    // vocab_stats not yet in outer_mind SOURCE_KEYS → defaults 0.5.
    let mc_usage_gini: f64 = field_or_default(meta_cgn.as_ref(), "usage_gini", 0.5);
    let vocab_avg_confidence: f64 = 0.5; // vocab_stats not yet plumbed
    thinking[1] = safe_clamp(
        0.35 * memory_directive_alignment
            + 0.25 * mc_knowledge_helpful
            + 0.20 * vocab_avg_confidence
            + 0.20 * (1.0 - mc_usage_gini),
    );

    // [2] situational_awareness — SPEC §23.8 REDESIGNED:
    // 0.5*(1/(1+t_since_last_event/1800)) + 0.3*min(1, events_teacher.felt_experiences_24h/20)
    // + 0.2*memory.learning_velocity
    let t_since_last_event: f64 =
        field_or_default(social_x_gateway.as_ref(), "t_since_last_event", 1800.0);
    let felt_experiences_24h: f64 =
        field_or_default(events_teacher.as_ref(), "felt_experiences_24h", 0.0);
    let memory_learning_velocity: f64 =
        field_or_default(memory_stats.as_ref(), "learning_velocity", 0.5);
    thinking[2] = safe_clamp(
        0.5 * (1.0 / (1.0 + t_since_last_event / 1800.0))
            + 0.3 * (felt_experiences_24h / 20.0).min(1.0)
            + 0.2 * memory_learning_velocity,
    );

    // [3] problem_solving — unchanged
    thinking[3] = safe_clamp(action_success_rate);
    // [4] communication_clarity — unchanged
    thinking[4] = safe_clamp(assessment_mean);
    // Suppress unused-var warnings for legacy research_* now that
    // thinking[0,2] use the redesigned formulas.
    let _ = (
        research_queries,
        research_usage_rate,
        research_seconds_since_last,
    );

    // ── FEELING (5D) — outer_mind_tensor.py:96-148 ──
    let mut feeling = [0.5_f64; 5];

    // Shared social_activity score (lines 100-102)
    let social_activity =
        safe_clamp((interactions_per_window / 5.0).min(1.0) * 0.5 + sentiment_avg * 0.5);

    // [5] social_temperature (lines 104-107)
    let interaction_rate = (interactions_per_window / 8.0).min(1.0);
    feeling[0] = safe_clamp(0.5 * sentiment_avg + 0.3 * interaction_rate + 0.2 * social_activity);

    // [6] social_connection — twin resonance + general activity (lines 109-119)
    let twin_reachable = lookup_bool(twin_state.as_ref(), "reachable");
    if twin_reachable {
        let twin_da = field_or_default(twin_state.as_ref(), "DA", 0.5);
        let twin_ne = field_or_default(twin_state.as_ref(), "NE", 0.5);
        let twin_gaba = field_or_default(twin_state.as_ref(), "GABA", 0.5);
        let twin_sim =
            1.0 - ((twin_da - 0.5).abs() + (twin_ne - 0.5).abs() + (twin_gaba - 0.5).abs()) / 3.0;
        feeling[1] = safe_clamp(0.6 * (0.3 + 0.5 * twin_sim) + 0.4 * social_activity);
    } else {
        feeling[1] = safe_clamp(social_activity);
    }

    // [7] network_weather — inverted body entropy (lines 121-124)
    let body_entropy = outer_body[3] as f64;
    feeling[2] = safe_clamp(1.0 - body_entropy);

    // [8] environmental_rhythm — blockchain + circadian + net oscillation (lines 126-137)
    let blockchain_active = compute_blockchain_active(anchor_state.as_ref());
    let circadian = outer_body[4] as f64;
    let net_oscillation = outer_body[3] as f64;
    feeling[3] = safe_clamp(0.35 * blockchain_active + 0.35 * circadian + 0.30 * net_oscillation);

    // [9] external_information_flow (lines 139-148)
    let bus_published: f64 = field_or_default(bus_stats.as_ref(), "published", 0.0);
    let bus_diversity = if bus_published > 0.0 {
        (bus_published / 1000.0).min(1.0)
    } else {
        0.1
    };
    let social_input = (interactions_per_window / 10.0).min(1.0);
    let bus_types = lookup_array_len(bus_stats.as_ref(), "modules") as f64;
    let perturbation_richness = (bus_types / 8.0).min(1.0);
    feeling[4] = safe_clamp(0.4 * social_input + 0.3 * bus_diversity + 0.3 * perturbation_richness);

    // ── WILLING (5D) — outer_mind_tensor.py:150-171 ──
    // SPEC §23.8 willing[1,3,4] updated 2026-05-07 (Phase 1 redesign).
    let mut willing = [0.5_f64; 5];
    // willing[10..14] RE-GROUNDED (D-SPEC-101 Phase-2, Maker 2026-05-21):
    //   ~90s fast-EMA breath at the 23.49 Hz mind cadence, replacing the
    //   D-SPEC-87 24h-rolling-count normalizations (action_per_day/240,
    //   (posts+replies)_day/120, …) that barely moved. The plugin willing-
    //   window tracker emits a dual-EMA breath over cumulative volitional
    //   counters — action (total_actions) / social (SOCIAL composite fires) /
    //   creative (ART+MUSIC fires) / protective (verifier rejections) /
    //   exploration (vocab + grounded primitives). Old per-day path deleted
    //   (no shim) per SPEC §23.8.
    let ww = lookup_map(map, "willing_window");
    willing[0] = safe_clamp(field_or_default(ww.as_ref(), "action_rate", 0.0)); // [10] action_throughput
    willing[1] = safe_clamp(field_or_default(ww.as_ref(), "social_rate", 0.0)); // [11] social_initiative
    willing[2] = safe_clamp(field_or_default(ww.as_ref(), "creative_rate", 0.0)); // [12] creative_output
    willing[3] = safe_clamp(field_or_default(ww.as_ref(), "protective_rate", 0.0)); // [13] protective_response
    willing[4] = safe_clamp(field_or_default(ww.as_ref(), "exploration_rate", 0.0)); // [14] exploration_drive
                                                                                     // Preprocessing vars superseded by the willing-window breath (kept as
                                                                                     // upstream lookups for the thinking/feeling dims; the willing-only ones
                                                                                     // are explicitly discarded here).
    let _ = (
        action_per_window,
        action_per_day,
        social_outputs_per_window,
        creative_per_window,
        creative_per_day,
        rejections_per_window,
        research_queries_per_window,
    );

    // ── Final: cast f64 → f32. Python does NOT `round(v, 4)` here
    // (unlike `_collect_outer_body`); preserve full f32 precision. ─
    let mut out = [0.0_f32; 15];
    for i in 0..5 {
        out[i] = thinking[i] as f32;
        out[i + 5] = feeling[i] as f32;
        out[i + 10] = willing[i] as f32;
    }
    Ok(out)
}

// ── V6 helpers (named to mirror Python semantics) ─────────────────

/// `_clamp(v, lo=0.0, hi=1.0)` from `outer_mind_tensor.py:224-226`:
/// `max(lo, min(hi, float(v) if v == v else 0.5))` — NaN → 0.5, then clamp.
/// Inf is not explicitly handled in Python but `min/max` propagate it
/// safely; we treat Inf as the clamp boundary (matches outer_body 9G).
fn safe_clamp(v: f64) -> f64 {
    if v.is_nan() {
        return 0.5;
    }
    v.clamp(0.0, 1.0)
}

/// Look up a top-level key; if value is a map, return its entries.
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

/// Look up a boolean field within a sub-map; false if missing/non-bool.
/// Used for `twin_state.reachable` per `outer_mind_tensor.py:112`.
fn lookup_bool(map: Option<&Vec<(rmpv::Value, rmpv::Value)>>, key: &str) -> bool {
    let map = match map {
        Some(m) => m,
        None => return false,
    };
    use rmpv::Value;
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some(key) {
                return matches!(v, Value::Boolean(true));
            }
        }
    }
    false
}

/// Look up an array-or-set length within a sub-map; 0 if missing/non-array.
/// Mirrors Python `len(_bus.get("modules", set())) if isinstance(_bus.get("modules"),
/// (set, list)) else 0` from `outer_mind_tensor.py:146`. msgpack-encoded
/// Python sets serialize as arrays.
fn lookup_array_len(map: Option<&Vec<(rmpv::Value, rmpv::Value)>>, key: &str) -> usize {
    let map = match map {
        Some(m) => m,
        None => return 0,
    };
    use rmpv::Value;
    for (k, v) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some(key) {
                if let Value::Array(arr) = v {
                    return arr.len();
                }
                return 0;
            }
        }
    }
    0
}

/// Compute blockchain-active proxy per `outer_mind_tensor.py:128-133`:
///   if anchor.success and anchor.last_anchor_time:
///     since = time.time() - last_anchor_time
///     return max(0.1, 1.0 / (1.0 + since / 300.0))
///   else: return 0.5
fn compute_blockchain_active(anchor: Option<&Vec<(rmpv::Value, rmpv::Value)>>) -> f64 {
    let anchor = match anchor {
        Some(a) => a,
        None => return 0.5,
    };
    use rmpv::Value;
    let mut success = false;
    let mut last_ts: Option<f64> = None;
    for (k, v) in anchor.iter() {
        if let Value::String(s) = k {
            match s.as_str() {
                Some("success") => {
                    success = matches!(v, Value::Boolean(true));
                }
                Some("last_anchor_time") => {
                    last_ts = v.as_f64();
                }
                _ => {}
            }
        }
    }
    if !success {
        return 0.5;
    }
    let last_ts = match last_ts {
        Some(t) if t > 0.0 => t,
        _ => return 0.5,
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let since = (now - last_ts).max(0.0);
    let raw = 1.0 / (1.0 + since / 300.0);
    raw.max(0.1)
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

/// Build a SPEC §8.5 outer `MIND_STATE` payload as `rmpv::Value::Map` per
/// SPEC §8.10 line 900 byte-identical guarantee.
fn encode_mind_state_payload(mind: &[f32; 15]) -> rmpv::Value {
    use rmpv::Value;
    let values = Value::Array(mind.iter().map(|f| Value::F64(*f as f64)).collect());
    Value::Map(vec![
        (Value::String("src".into()), Value::String("outer".into())),
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
    use titan_core::constants::OUTER_MIND_15D_SCHEMA_VERSION;

    fn pure_compute(
        raw_mind: [f32; 15],
        unified: Option<[f32; 15]>,
        local: Option<[f32; 15]>,
        topology_lower: Option<[f32; 10]>,
        ground_up: &mut GroundUpEnricher,
    ) -> [f32; 15] {
        let mut mind = raw_mind;
        let composed: Vec<f32> = match (unified, local) {
            (Some(u), Some(l)) => compose_multipliers_default(&u, &l),
            (Some(u), None) => u.to_vec(),
            (None, Some(l)) => l.to_vec(),
            (None, None) => vec![1.0_f32; 15],
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
    fn neutral_inputs_preserve_mind() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5_f32; 15];
        let out = pure_compute(raw, None, None, None, &mut g);
        for i in 0..15 {
            assert!((out[i] - raw[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ground_up_applies_to_willing_only_per_g10() {
        // SPEC G10: ground_up_skip_mind_thinking=0:5 +
        //          ground_up_skip_mind_feeling=5:10 +
        //          ground_up_mind_range=10:15
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5_f32; 15];
        // apply_to_mind reads topology[5:10] (the latter half of the 10D
        // outer_lower slice) for the willing nudge — see C-S5
        // titan-inner-mind-rs::ground_up_only_touches_willing_10_15.
        let topo: [f32; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04, 0.04];
        let out = pure_compute(raw, None, None, Some(topo), &mut g);
        // Thinking [0:5] UNTOUCHED
        for i in 0..5 {
            assert!(
                (out[i] - 0.5).abs() < 1e-6,
                "Thinking dim {} should be unchanged (G10 skip), got {}",
                i,
                out[i],
            );
        }
        // Feeling [5:10] UNTOUCHED
        for i in 5..10 {
            assert!(
                (out[i] - 0.5).abs() < 1e-6,
                "Feeling dim {} should be unchanged (G10 skip), got {}",
                i,
                out[i],
            );
        }
        // Willing [10:15] NUDGED — delta = 0.002 * 0.1 * 1.0 = 0.0002 each
        for i in 10..15 {
            assert!(
                (out[i] - 0.5002).abs() < 1e-5,
                "Willing dim {} should be nudged ~0.0002, got {}",
                i,
                out[i],
            );
        }
    }

    #[test]
    fn ground_up_skipped_without_topology() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5_f32; 15];
        let out = pure_compute(raw, None, None, None, &mut g);
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn unified_only_applies_multipliers() {
        let mut g = GroundUpEnricher::new(Side::MindWilling);
        let raw = [0.5_f32; 15];
        let mut unified = [1.0_f32; 15];
        unified[0] = 2.0;
        let out = pure_compute(raw, Some(unified), None, None, &mut g);
        assert!((out[0] - 1.0).abs() < 1e-6); // 0.5 * 2.0 = 1.0 (TENSOR_MAX)
    }

    #[test]
    fn mind_payload_src_is_outer() {
        let mind = [0.0_f32; 15];
        let bytes = encode_mind_state_payload(&mind);
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
    fn mind_payload_type_is_mind_state() {
        let mind = [0.0_f32; 15];
        let bytes = encode_mind_state_payload(&mind);
        use rmpv::Value;
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("type") {
                        assert_eq!(val, Value::String("MIND_STATE".into()));
                        return;
                    }
                }
            }
        }
        panic!("type field missing");
    }

    #[test]
    fn slot_write_15d_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("outer_mind_15d.bin");
        let mut slot = Slot::create(&path, OUTER_MIND_15D_SCHEMA_VERSION as u32, 60).unwrap();
        let mind: [f32; 15] = [0.1; 15];
        let bytes = encode_floats::<15>(&mind);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn project_outer_mind_empty_dict_matches_spec_defaults() {
        // Empty msgpack map → all derived stats default. Hand-computed
        // expected values per SPEC §23.8 (REDESIGNED 2026-05-07 thinking[0,1,2]
        // + willing[1,3,4]) with neutral inputs (all source-dict fields None):
        //
        //   thinking[0] = 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
        //                 (meta_cgn.knowledge_helpful + cgn.avg_reward_norm
        //                  + memory.directive_alignment all default 0.5)
        //   thinking[1] = 0.35*0.5 + 0.25*0.5 + 0.20*0.5 + 0.20*(1-0.5) = 0.5
        //   thinking[2] = 0.5*(1/(1+1800/1800)) + 0.3*0 + 0.2*0.5 = 0.35
        //                 (t_since_last_event default 1800; felt_experiences 0;
        //                  memory.learning_velocity default 0.5)
        //   thinking[3] = (0-0)/max(1,0) = 0 → action_success_rate
        //   thinking[4] = 0.5 (assessment_mean default)
        //   social_activity = clamp(0*0.5 + 0.5*0.5) = 0.25
        //   feeling[0] = 0.5*0.5 + 0.3*0 + 0.2*0.25 = 0.30
        //   feeling[1] = social_activity = 0.25 (no twin)
        //   feeling[2] = 1.0 - 0.5 = 0.5 (outer_body[3] default 0.5)
        //   feeling[3] = 0.35*0.5 + 0.35*0.5 + 0.30*0.5 = 0.5 (no anchor)
        //   feeling[4] = 0.4*0 + 0.3*0.1 + 0.3*0 = 0.03
        //   willing[*] = 0 (per-hour rates / counts all 0)
        use rmpv::Value;
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &Value::Map(vec![])).unwrap();
        let outer_body = [0.5_f32; 5];
        let mind = project_outer_mind_15d(&payload, outer_body).unwrap();
        assert!((mind[0] - 0.5).abs() < 1e-3, "thinking[0] = {}", mind[0]);
        assert!((mind[1] - 0.5).abs() < 1e-3, "thinking[1] = {}", mind[1]);
        assert!((mind[2] - 0.35).abs() < 1e-3, "thinking[2] = {}", mind[2]);
        assert!((mind[3] - 0.0).abs() < 1e-3);
        assert!((mind[4] - 0.5).abs() < 1e-3);
        assert!((mind[5] - 0.30).abs() < 1e-3);
        assert!((mind[6] - 0.25).abs() < 1e-3);
        assert!((mind[7] - 0.5).abs() < 1e-3);
        assert!((mind[8] - 0.5).abs() < 1e-3);
        assert!((mind[9] - 0.03).abs() < 1e-3);
        for i in 10..15 {
            assert!(
                (mind[i] - 0.0).abs() < 1e-3,
                "willing[{}] = {}",
                i - 10,
                mind[i]
            );
        }
    }

    #[test]
    fn project_outer_mind_willing_window_drives_willing_dims() {
        // D-SPEC-101 Phase-2 (Maker 2026-05-21): willing[10-14] read the ~90s
        // breath from the plugin willing-window tracker (action/social/creative/
        // protective/exploration rates), replacing the D-SPEC-87 24h per-day
        // normalizations.
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("willing_window".into()),
            Value::Map(vec![
                (Value::String("action_rate".into()), Value::F64(0.61)),
                (Value::String("social_rate".into()), Value::F64(0.22)),
                (Value::String("creative_rate".into()), Value::F64(0.48)),
                (Value::String("protective_rate".into()), Value::F64(0.13)),
                (Value::String("exploration_rate".into()), Value::F64(0.37)),
            ]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        assert!(
            (mind[10] - 0.61).abs() < 1e-4,
            "action_throughput={}",
            mind[10]
        );
        assert!(
            (mind[11] - 0.22).abs() < 1e-4,
            "social_initiative={}",
            mind[11]
        );
        assert!(
            (mind[12] - 0.48).abs() < 1e-4,
            "creative_output={}",
            mind[12]
        );
        assert!(
            (mind[13] - 0.13).abs() < 1e-4,
            "protective_response={}",
            mind[13]
        );
        assert!(
            (mind[14] - 0.37).abs() < 1e-4,
            "exploration_drive={}",
            mind[14]
        );
    }

    #[test]
    fn project_outer_mind_assessment_drives_thinking_4_only() {
        // SPEC §23.8 REDESIGNED 2026-05-07: thinking[1] knowledge_retrieval
        // no longer depends on assessment_mean (now memory.directive_alignment
        // + meta_cgn.knowledge_helpful_ratio + vocab.avg_confidence + meta_cgn
        // usage_gini). Only thinking[4] communication_clarity still maps to
        // assessment_mean.
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("assessment_stats".into()),
            Value::Map(vec![(
                Value::String("average_score".into()),
                Value::F64(0.8),
            )]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        // thinking[1] now ignores assessment — should equal default-input result (0.5).
        assert!((mind[1] - 0.5).abs() < 1e-3, "thinking[1] = {}", mind[1]);
        assert!((mind[4] - 0.8).abs() < 1e-3, "thinking[4] = {}", mind[4]);
    }

    #[test]
    fn project_outer_mind_action_success_drives_thinking_3() {
        // total=100, failed=20 → success_rate = 80/100 = 0.8
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("agency_stats".into()),
            Value::Map(vec![
                (
                    Value::String("total_actions".into()),
                    Value::Integer(100.into()),
                ),
                (
                    Value::String("failed_actions".into()),
                    Value::Integer(20.into()),
                ),
            ]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        assert!((mind[3] - 0.8).abs() < 1e-3, "thinking[3] = {}", mind[3]);
    }

    #[test]
    fn project_outer_mind_outer_body_drives_feeling_2_and_3() {
        // outer_body[3] = 0.7 (entropy) → feeling[2] = 1.0 - 0.7 = 0.3
        // outer_body[4] = 0.6 (thermal/circadian) → contributes to feeling[3]
        use rmpv::Value;
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &Value::Map(vec![])).unwrap();
        let outer_body = [0.5_f32, 0.5, 0.5, 0.7, 0.6];
        let mind = project_outer_mind_15d(&payload, outer_body).unwrap();
        assert!((mind[7] - 0.3).abs() < 1e-3, "feeling[2] = {}", mind[7]);
        // feeling[3] = 0.35*0.5 (no anchor) + 0.35*0.6 (circadian) + 0.30*0.7 (osc)
        //            = 0.175 + 0.21 + 0.21 = 0.595
        assert!((mind[8] - 0.595).abs() < 1e-3, "feeling[3] = {}", mind[8]);
    }

    #[test]
    fn project_outer_mind_twin_state_lifts_feeling_1() {
        // twin reachable, all neutral DA/NE/GABA → twin_sim = 1.0
        // feeling[1] = 0.6*(0.3 + 0.5*1.0) + 0.4*social_activity
        //            = 0.6*0.8 + 0.4*0.25 = 0.48 + 0.10 = 0.58
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("twin_state".into()),
            Value::Map(vec![
                (Value::String("reachable".into()), Value::Boolean(true)),
                (Value::String("DA".into()), Value::F64(0.5)),
                (Value::String("NE".into()), Value::F64(0.5)),
                (Value::String("GABA".into()), Value::F64(0.5)),
            ]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        assert!((mind[6] - 0.58).abs() < 1e-3, "feeling[1] = {}", mind[6]);
    }

    #[test]
    fn project_outer_mind_willing_ignores_legacy_per_day_counts() {
        // D-SPEC-101 Phase-2 (Maker 2026-05-21): willing[10-14] no longer read
        // the D-SPEC-87 24h per-day counts (actions_this_day/240,
        // creative_this_day/120, …) — they read the ~90s willing-window breath.
        // Feeding agency per-day counts alone leaves willing at the absent-
        // window default 0.0 (the breath comes from willing_window — see
        // project_outer_mind_willing_window_drives_willing_dims).
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("agency_stats".into()),
            Value::Map(vec![
                (
                    Value::String("actions_this_day".into()),
                    Value::Integer(120.into()),
                ),
                (
                    Value::String("creative_this_day".into()),
                    Value::Integer(72.into()),
                ),
            ]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        assert_eq!(mind[10], 0.0, "willing[0] should ignore actions_this_day");
        assert_eq!(mind[12], 0.0, "willing[2] should ignore creative_this_day");
    }

    #[test]
    fn project_outer_mind_research_queries_change_thinking_0() {
        // research_nodes > 0 → thinking[0] = research_usage_rate (0.5 fixed)
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("memory_status".into()),
            Value::Map(vec![(
                Value::String("research_nodes".into()),
                Value::Integer(10.into()),
            )]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mind = project_outer_mind_15d(&bytes, [0.5_f32; 5]).unwrap();
        assert!((mind[0] - 0.5).abs() < 1e-3, "thinking[0] = {}", mind[0]);
    }

    #[test]
    fn safe_clamp_handles_nan() {
        assert_eq!(safe_clamp(f64::NAN), 0.5);
        assert_eq!(safe_clamp(2.0), 1.0);
        assert_eq!(safe_clamp(-1.0), 0.0);
        assert_eq!(safe_clamp(0.5), 0.5);
    }

    #[test]
    fn lookup_bool_returns_true_only_for_explicit_true() {
        use rmpv::Value;
        let map = vec![
            (Value::String("a".into()), Value::Boolean(true)),
            (Value::String("b".into()), Value::Boolean(false)),
            (Value::String("c".into()), Value::F64(1.0)),
        ];
        assert!(lookup_bool(Some(&map), "a"));
        assert!(!lookup_bool(Some(&map), "b"));
        assert!(!lookup_bool(Some(&map), "c"));
        assert!(!lookup_bool(Some(&map), "missing"));
        assert!(!lookup_bool(None, "a"));
    }

    #[test]
    fn lookup_array_len_returns_array_size() {
        use rmpv::Value;
        let map = vec![(
            Value::String("modules".into()),
            Value::Array(vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
            ]),
        )];
        assert_eq!(lookup_array_len(Some(&map), "modules"), 3);
        assert_eq!(lookup_array_len(Some(&map), "missing"), 0);
        assert_eq!(lookup_array_len(None, "modules"), 0);
    }

    #[test]
    fn decode_unified_outer_mind_extracts_field() {
        use rmpv::Value;
        let mults_arr: Vec<Value> = (0..15).map(|i| Value::F64(0.5 + i as f64 * 0.1)).collect();
        let payload = Value::Map(vec![(
            Value::String("multipliers".into()),
            Value::Map(vec![(
                Value::String("outer_mind".into()),
                Value::Array(mults_arr),
            )]),
        )]);
        let mults = decode_unified_outer_mind(&payload).unwrap();
        for i in 0..15 {
            let expected = 0.5_f32 + i as f32 * 0.1;
            assert!((mults[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn decode_unified_outer_mind_errors_on_missing() {
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("multipliers".into()),
            Value::Map(vec![]),
        )]);
        assert!(decode_unified_outer_mind(&payload).is_err());
    }

    #[test]
    fn content_hash_gates_redundant_writes() {
        let mut gate = ContentGate::new();
        let mind: [f32; 15] = [0.1; 15];
        let bytes = encode_floats::<15>(&mind);
        assert!(gate.should_write(&bytes));
        assert!(!gate.should_write(&bytes));
        assert_eq!(gate.write_count(), 1);
    }

    #[test]
    fn stale_threshold_is_3x_cadence() {
        // SPEC §18.1 + D-SPEC-100: outer_mind cadence 15s × 3 = 45s
        assert_eq!(outer_mind_stale_threshold_s(), 45.0);
    }

    #[test]
    fn cadence_constants_match_spec() {
        // Sensor sidecar source-refresh cadence (D-SPEC-100: stale threshold source).
        // G13: 15s = strict 1:3:9 (spirit 5 / mind 15 / body 45).
        assert_eq!(OUTER_MIND_TICK_BASE_S, 15.0);
        // Bus publish throttle (Schumann mind ≈ 15s) — now mirrors TICK_BASE.
        assert_eq!(OUTER_MIND_BUS_PUBLISH_INTERVAL_S, 15.0);
    }

    #[test]
    fn schumann_mind_period_is_canonical() {
        // Daemon ticks at SCHUMANN_MIND_HZ (23.49 Hz, ~42.6ms) per
        // post-A.S8 D2 cadence migration.
        let g = SchumannGenerator::new(SchumannRole::Mind, tokio::time::Instant::now());
        let period_ns = g.period_ns();
        assert!(
            (period_ns as i64 - 42_571_307).abs() <= 1,
            "mind period_ns = {period_ns}, expected 42571307 ± 1"
        );
    }
}
