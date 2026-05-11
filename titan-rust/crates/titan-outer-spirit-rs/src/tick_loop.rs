//! tick_loop — outer-spirit jittered ~30s tick + Observer Principle.
//!
//! Per SPEC §9.A `titan-outer-spirit-rs` + §18.1 + master plan §10.6 C6-5.
//! Logic flow per tick:
//!
//! 1. Read `outer_body_5d.bin` (Observer Principle G8 — spirit observes
//!    sibling body).
//! 2. Read `outer_mind_15d.bin` (Observer Principle — observes sibling mind).
//! 3. Read `sensor_cache_outer_spirit.bin` (msgpack pre-aggregated raw
//!    upstream stats from Python sidecar — agency_stats / assessment_stats /
//!    memory_status / social_perception_stats / art_count_500 /
//!    audio_count_500 / uptime_seconds / hormone_levels / solana_stats /
//!    recovery_stats / history).
//! 4. Preprocess raw stats into action_stats / creative_stats / guardian_stats /
//!    sovereignty_ratio / uptime_ratio / social_stats / memory_stats /
//!    assessment_ext (port of `outer_trinity.py:_collect_extended` lines 600-697).
//! 5. Compute 45D Sat-Chit-Ananda Material (SAT[0:15] + CHIT[15:30] +
//!    ANANDA[30:45]) per `outer_spirit_tensor.py:collect_outer_spirit_45d`.
//! 6. Apply UNIFIED filter_down to all 45D (G7 [0.3, 3.0] clamp). NO
//!    LOCAL filter_down on spirit (per SPEC §9.A line 935 — outer-spirit
//!    PUBLISHES OUTER_SPIRIT_FILTER_DOWN; doesn't consume it).
//! 7. NO ground_up applied to spirit per SPEC G10
//!    (`ground_up_skip_spirit=0:45` — spirit is the witness, not a target).
//! 8. Content-hash gate; write `outer_spirit_45d.bin` (full 45D).
//! 9. Publish SPIRIT_STATE (P1, src=outer, full 45D unmasked).
//! 10. Compose OUTER_SPIRIT_FILTER_DOWN payload — derive multipliers for
//!     body[5] + mind[15] + outer_spirit_content[40]. Observer dims [0:5]
//!     MASKED per SPEC G8 — extracted via
//!     `extract_outer_spirit_content` which structurally cannot leak [0:5].
//! 11. Publish OUTER_SPIRIT_FILTER_DOWN (P1, LOCAL).

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{
    OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S, OUTER_SPIRIT_FIRING_MAX_BYTES,
    OUTER_SPIRIT_FIRING_SCHEMA_VERSION, OUTER_SPIRIT_TICK_BASE_S,
};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, encode_floats, extract_outer_spirit_content, read_dim_slice,
    read_sensor_cache, ContentGate, FiringSlotWriter, PublishThrottle, SensorCacheRead,
    CONTENT_DIM_COUNT, OUTER_SPIRIT_TOPICS,
};

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "outer-spirit")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(OUTER_SPIRIT_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?OUTER_SPIRIT_TOPICS);

    let outer_spirit_slot = open_slot(shm_dir, "outer_spirit_45d.bin")?;
    let outer_body_slot = open_slot(shm_dir, "outer_body_5d.bin")?;
    let outer_mind_slot = open_slot(shm_dir, "outer_mind_15d.bin")?;
    let sensor_cache_path = shm_dir.join("sensor_cache_outer_spirit.bin");
    info!(event = "SHM_OPENED");

    // Phase C 130D dim-live tracker bridge (rFP §4.7).
    let firing_writer = FiringSlotWriter::new(
        "outer_spirit",
        shm_dir,
        OUTER_SPIRIT_FIRING_SCHEMA_VERSION as u32,
        OUTER_SPIRIT_FIRING_MAX_BYTES as u32,
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
        outer_spirit_slot,
        outer_body_slot,
        outer_mind_slot,
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
    /// Most recent UNIFIED_SPIRIT_FILTER_DOWN.outer_spirit_content[40]
    /// multipliers (G8 — observer dims already masked at publish side).
    unified: Option<[f32; CONTENT_DIM_COUNT]>,
    shutdown_requested: bool,
    /// Last successfully-computed 45D outer-spirit vector.
    last_spirit: [f32; 45],
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            unified: None,
            shutdown_requested: false,
            last_spirit: [0.5; 45],
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
    if msg_type == "UNIFIED_SPIRIT_FILTER_DOWN" {
        let payload = match titan_bus::client::extract_payload(raw_bytes) {
            Some(p) => p,
            None => return,
        };
        match decode_unified_outer_spirit_content(&payload) {
            Ok(mults) => {
                if let Ok(mut s) = state.lock() {
                    s.unified = Some(mults);
                }
            }
            Err(e) => {
                warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN.outer_spirit_content failed")
            }
        }
    }
}

fn decode_unified_outer_spirit_content(payload: &[u8]) -> Result<[f32; CONTENT_DIM_COUNT]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("payload root: {e}"))?;
    let map = match &v {
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
        _ => return Err(anyhow!("multipliers missing or not a map")),
    };
    let mut content = [1.0_f32; CONTENT_DIM_COUNT];
    let mut found = false;
    for (k, val) in mults_map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("outer_spirit_content") {
                decode_float_array_into(val, &mut content)?;
                found = true;
                break;
            }
        }
    }
    if !found {
        return Err(anyhow!("multipliers.outer_spirit_content missing"));
    }
    Ok(content)
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
    mut outer_spirit_slot: Slot,
    outer_body_slot: Slot,
    outer_mind_slot: Slot,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
) -> Result<()> {
    // Post-A.S8 D2 cadence migration (rFP §4.2): tick at canonical Schumann
    // spirit (70.47 Hz, ~14.2ms) — same generator inner-spirit-rs uses.
    // Slot writes content-gated; bus publishes throttled by PublishThrottle.
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    let mut content_gate = ContentGate::new();
    let mut publish_throttle = PublishThrottle::new(OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S);

    info!(
        event = "TICK_LOOP_START",
        role = "outer-spirit",
        period_ns,
        publish_interval_s = OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S,
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
                        event = "OUTER_SPIRIT_TICK",
                        epoch = tick_event.epoch,
                        period_ns = tick_event.period_ns,
                    );
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate, &mut publish_throttle,
                        &mut outer_spirit_slot, &outer_body_slot, &outer_mind_slot,
                        &sensor_cache_path, &mut firing_writer,
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

fn outer_spirit_stale_threshold_s() -> f64 {
    OUTER_SPIRIT_TICK_BASE_S * 3.0
}

#[allow(clippy::too_many_arguments)]
async fn run_one_tick(
    bus: &Arc<BusClient>,
    state: &Arc<Mutex<DaemonState>>,
    content_gate: &mut ContentGate,
    publish_throttle: &mut PublishThrottle,
    outer_spirit_slot: &mut Slot,
    outer_body_slot: &Slot,
    outer_mind_slot: &Slot,
    sensor_cache_path: &Path,
    firing_writer: &mut FiringSlotWriter,
) -> Result<()> {
    // 1+2. Observer Principle reads (G8) — sibling outer_body + outer_mind.
    let outer_body: [f32; 5] = read_dim_slice::<5>(outer_body_slot).unwrap_or([0.5; 5]);
    let outer_mind: [f32; 15] = read_dim_slice::<15>(outer_mind_slot).unwrap_or([0.5; 15]);

    // 3. Sensor cache (msgpack pre-aggregated outer-state).
    let last_spirit = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.last_spirit
    };

    // 4+5. Compute 45D Sat-Chit-Ananda Material (preprocesses + projects).
    let mut spirit = match read_sensor_cache(sensor_cache_path, outer_spirit_stale_threshold_s()) {
        Ok(SensorCacheRead::Fresh { payload, .. }) => {
            match project_outer_spirit_45d(&payload, outer_body, outer_mind, last_spirit) {
                Ok(s) => s,
                Err(e) => {
                    warn!(err = ?e, "outer_spirit project failed; using last-known");
                    last_spirit
                }
            }
        }
        Ok(SensorCacheRead::Stale { age_s, .. }) => {
            warn!(
                event = "SENSOR_CACHE_STALE",
                age_s,
                threshold_s = outer_spirit_stale_threshold_s(),
                confidence = 0.0,
            );
            last_spirit
        }
        Ok(SensorCacheRead::Missing) => last_spirit,
        Err(e) => {
            warn!(err = ?e, "sensor_cache read errored");
            last_spirit
        }
    };

    // 6. Apply UNIFIED filter_down to all 45D (G7 clamp).
    //    Per SPEC G8 the UNIFIED publisher already masks observer dims
    //    in `outer_spirit_content`; we extend to a 45D mult vector with
    //    1.0 in observer slots.
    let unified_content = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.unified
    };
    if let Some(content) = unified_content {
        let mut mults_45 = [1.0_f32; 45];
        // observer dims [0:5] stay 1.0 (no filter); content [5:45] from unified.
        mults_45[5..].copy_from_slice(&content);
        apply_multipliers(&mut spirit, &mults_45);
    }

    // 7. NO ground_up per G10 ground_up_skip_spirit=0:45.
    //    NO LOCAL filter_down per SPEC §9.A — we PUBLISH OUTER_SPIRIT_FILTER_DOWN.

    // Cache fresh spirit.
    if let Ok(mut s) = state.lock() {
        s.last_spirit = spirit;
    }

    // 8. Encode + content-hash gate the slot write (full 45D unmasked).
    let bytes = encode_floats::<45>(&spirit);
    if content_gate.should_write(&bytes) {
        outer_spirit_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // 8b. Phase C dim-live tracker bridge (rFP §4.7) — record per tick at
    //     full Schumann spirit cadence (independent of bus publish throttle).
    firing_writer.record_tick(&spirit, &[], now_secs());

    // 9-11. Bus publishes throttled per OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S
    // (post-A.S8 D2 — rFP §4.2). Tick fires at Schumann spirit (70.47 Hz)
    // but bus publishes only every ~5s. Slot writes (above) remain at full
    // tick cadence under content-hash gating.
    if publish_throttle.should_publish() {
        // 9. Publish SPIRIT_STATE (full 45D unmasked).
        let state_payload = encode_spirit_state_payload(&spirit);
        bus.publish("SPIRIT_STATE", Some("all"), Some(&state_payload))
            .await
            .map_err(|e| anyhow!("publish SPIRIT_STATE: {e}"))?;

        // 10+11. Compose + publish OUTER_SPIRIT_FILTER_DOWN with observer mask.
        let filter_down_payload = encode_outer_spirit_filter_down_payload(&spirit);
        bus.publish(
            "OUTER_SPIRIT_FILTER_DOWN",
            Some("all"),
            Some(&filter_down_payload),
        )
        .await
        .map_err(|e| anyhow!("publish OUTER_SPIRIT_FILTER_DOWN: {e}"))?;
    }

    Ok(())
}

// ── V6 outer_spirit_45d full port ──────────────────────────────────
//
// Byte-faithful port of `titan_plugin/logic/outer_spirit_tensor.py::collect_outer_spirit_45d`
// (lines 53-101) wrapping `titan_plugin/logic/outer_trinity.py::_collect_extended`
// preprocessing (lines 600-697 — only the subset that feeds the 45D
// spirit formula).
//
// Closes rFP_phase_c_definitive_runtime_closure D1 (GAP-CS6-001).
// Replaces the structural-skeleton stub that filled 4 distinct values
// across 45 dims. Per Prime Directive #1 "if a function exists, it
// MUST do the work its name claims".
//
// **Sidecar contract**: `outer_spirit_sensor_refresh.py:SOURCE_KEYS` is
// updated in the same chunk to write RAW upstream stats (mirroring
// outer_mind sidecar) — agency_stats / assessment_stats / memory_status /
// social_perception_stats / art_count_500 / audio_count_500 /
// uptime_seconds / hormone_levels / solana_stats / recovery_stats /
// history. Per Q3 of the Day 1 session-handoff, _collect_extended
// preprocessing is INLINED here (not extracted to a shared crate);
// some helper functions duplicate outer_mind-rs equivalents (lookup_map,
// field_or_default, safe_clamp). Extraction to a shared trinity-daemon
// module is a follow-up cleanup item (D8 candidate).
//
// Parity bar: |Δ| < 1e-3 (matches 9G outer_body / 9I outer_mind / chi
// conventions).

/// V6 45D outer-spirit projection. Pure compute over the msgpack source
/// dict from the Python sidecar plus the freshly-read `outer_body_5d` +
/// `outer_mind_15d` (Observer Principle G8). Falls back to `fallback`
/// only when the msgpack envelope is fundamentally malformed (not a map).
///
/// Returns 45D vector: SAT[0:15] + CHIT[15:30] + ANANDA[30:45]
/// (Sat-Chit-Ananda Material consciousness applied to MATERIAL engagement).
pub fn project_outer_spirit_45d(
    payload: &[u8],
    outer_body: [f32; 5],
    outer_mind: [f32; 15],
    fallback: [f32; 45],
) -> Result<[f32; 45]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => {
            // Match outer_mind: fundamentally malformed envelope falls back.
            return Ok(fallback);
        }
    };

    // ── Step 1: Read raw upstream stats via SOURCE_KEYS lookups. ──
    let agency = lookup_map(map, "agency_stats");
    let assessment = lookup_map(map, "assessment_stats");
    let memory_status = lookup_map(map, "memory_status");
    let social_perception = lookup_map(map, "social_perception_stats");
    let recovery = lookup_map(map, "recovery_stats");
    let hormone_levels = lookup_map(map, "hormone_levels");
    let solana = lookup_map(map, "solana_stats");
    let history = lookup_map(map, "history");
    let art_count = top_level_f64_or(map, "art_count_500", 0.0);
    let audio_count = top_level_f64_or(map, "audio_count_500", 0.0);
    let uptime = top_level_f64_or(map, "uptime_seconds", 1.0).max(1.0);
    // Step 6 §4.3 P3 additions for SPEC §23.9 redesigned dims:
    let meta_cgn = lookup_map(map, "meta_cgn_stats");
    let cgn = lookup_map(map, "cgn_stats");
    let memory_stats_ext = lookup_map(map, "memory_stats");
    let knowledge_graph = lookup_map(map, "knowledge_graph_stats");
    let events_teacher = lookup_map(map, "events_teacher_stats");
    let jailbreak_alerts = lookup_map(map, "jailbreak_alerts_stats");
    let output_verifier = lookup_map(map, "output_verifier_stats");
    let anchor_state = lookup_map(map, "anchor_state");
    let bus_stats = lookup_map(map, "bus_stats");
    let expression_translator = lookup_map(map, "expression_translator_stats");
    let outer_spirit_history = lookup_map(map, "outer_spirit_history_stats");
    let community_engagement = lookup_map(map, "community_engagement_stats");

    // ── Step 2: Preprocess into derived stats (port of _collect_extended) ──
    // Python: outer_trinity.py:602-697

    // action_stats (lines 608-622)
    let total_actions: f64 = field_or_default(agency.as_ref(), "total_actions", 0.0);
    let failed_actions: f64 = field_or_default(agency.as_ref(), "failed_actions", 0.0);
    let action_success_rate: f64 =
        (total_actions - failed_actions) / total_actions.max(1.0);
    let actions_per_hour: f64 = total_actions / (uptime / 3600.0).max(0.01);
    let action_per_window: f64 =
        field_or_default(agency.as_ref(), "actions_this_hour", 0.0);
    let failed_retry_rate: f64 =
        field_or_default(agency.as_ref(), "failed_retry_rate", 0.0);
    let burst_frequency: f64 =
        field_or_default(agency.as_ref(), "burst_frequency", 0.0);
    let action_error_rate: f64 = 1.0 - action_success_rate;

    // creative_stats (lines 646-653)
    let creative_total: f64 = (art_count + audio_count) as f64;
    let creative_per_window: f64 =
        field_or_default(agency.as_ref(), "creative_this_hour", 0.0);
    let creative_unique_types: f64 =
        ((art_count > 0.0) as i32 + (audio_count > 0.0) as i32).min(2) as f64;
    let creative_mean_assessment: f64 =
        field_or_default(assessment.as_ref(), "average_score", 0.5);

    // guardian_stats (lines 656-661)
    let threats_detected: f64 =
        field_or_default(agency.as_ref(), "threats_detected", 0.0);
    let rejections: f64 = field_or_default(agency.as_ref(), "rejections", 0.0);
    // Python `threat_severity_avg` / `rejections_per_window` exist in dict
    // but the 45D spirit formula does not consume them — skip lookup.

    // sovereignty_ratio passed as `agency.get("sovereignty_ratio", 0.0)`
    let sovereignty_ratio: f64 =
        field_or_default(agency.as_ref(), "sovereignty_ratio", 0.0);

    // social_stats (lines 664-672)
    let interactions_per_window: f64 =
        field_or_default(memory_status.as_ref(), "unique_interactors", 0.0);
    let _social_sentiment_avg: f64 =
        field_or_default(social_perception.as_ref(), "sentiment_ema", 0.5);
    let social_mean_conversation_quality: f64 =
        field_or_default(assessment.as_ref(), "average_score", 0.5);
    // social_connection / social_events_count / last_contagion are NOT
    // referenced by the 45D formula — skip.

    // assessment_stats_ext (lines 683-688)
    let assessment_mean: f64 =
        field_or_default(assessment.as_ref(), "average_score", 0.5);
    let assessment_trend: f64 = field_or_default(assessment.as_ref(), "trend", 0.0);
    // count not used by 45D formula
    let assessment_score_variance: f64 =
        field_or_default(assessment.as_ref(), "score_variance", 0.3);

    // memory_stats (lines 691-694)
    let mem_persistent_nodes: f64 =
        field_or_default(memory_status.as_ref(), "persistent_count", 0.0);
    let mem_growth_per_epoch: f64 =
        field_or_default(memory_status.as_ref(), "growth_per_epoch", 0.0);

    // uptime_ratio (line 697)
    let uptime_ratio: f64 = (uptime / (uptime + 60.0).max(1.0)).min(1.0);

    // ── Step 3: Outer body / outer mind coherence (lines 87-89 of
    //            outer_spirit_tensor.py) ──
    let outer_body_coh: f64 = mean_f32(&outer_body);
    let outer_mind_coh: f64 = mean_f32(&outer_mind);
    let combined_coh: f64 = (outer_body_coh + outer_mind_coh) / 2.0;

    // ── Step 4: Compute 45D tensor (3 helpers, 15 dims each) ──
    let sat = compute_sat(
        total_actions,
        action_success_rate,
        actions_per_hour,
        creative_total,
        creative_unique_types,
        threats_detected,
        rejections,
        sovereignty_ratio,
        uptime_ratio,
        recovery.as_ref(),
        solana.as_ref(),
        assessment_mean,
        assessment_trend,
        assessment_score_variance,
    );
    let chit = compute_chit(
        mem_persistent_nodes,
        mem_growth_per_epoch,
        threats_detected,
        rejections,
        social_mean_conversation_quality,
        assessment_trend,
        history.as_ref(),
        outer_body_coh,
        outer_mind_coh,
        combined_coh,
    );
    let mut sat = sat;
    let mut chit = chit;
    let mut ananda = compute_ananda(
        action_success_rate,
        action_error_rate,
        failed_retry_rate,
        burst_frequency,
        social_mean_conversation_quality,
        creative_mean_assessment,
        interactions_per_window,
        action_per_window,
        creative_per_window,
        hormone_levels.as_ref(),
        history.as_ref(),
        outer_body_coh,
        outer_mind_coh,
        assessment_mean,
    );

    // ── Step 5: SPEC §23.9 redesign overrides (Step 6 §4.3 P3 ports) ──
    // Overrides the legacy compute_sat / compute_chit / compute_ananda
    // outputs for ~22 stale dims using the new source-dict inputs added
    // in Step 3. Per the rFP §4.3 audit: each override = SPEC §23.9
    // formula direct-port using new sidecar SOURCE_KEYS data.

    // ── SAT overrides ──────────────────────────────────────────────
    // SAT[3] boundary_enforcement REDESIGNED:
    //   blocked / max(1, threats_detected_24h)
    //   threats_detected_24h = jailbreak.count_24h + verifier.violation_events_24h
    //   blocked = jailbreak.score_ge_0.9_count + verifier.rejected_24h
    {
        let jb_count_24h: f64 = field_or_default(jailbreak_alerts.as_ref(), "count_24h", 0.0);
        let ov_violations_24h: f64 =
            field_or_default(output_verifier.as_ref(), "violation_events_24h", 0.0);
        let threats_total = jb_count_24h + ov_violations_24h;
        let jb_blocked: f64 =
            field_or_default(jailbreak_alerts.as_ref(), "score_ge_0.9_count", 0.0);
        let ov_rejected_24h: f64 =
            field_or_default(output_verifier.as_ref(), "rejected_24h", 0.0);
        let blocked_total = jb_blocked + ov_rejected_24h;
        sat[3] = safe_clamp(blocked_total / threats_total.max(1.0));
    }

    // SAT[6] observable_growth: _clamp(0.5 + assessment.trend)
    sat[6] = safe_clamp(0.5 + assessment_trend);

    // SAT[8] behavioral_consistency: _clamp(1 - assessment.score_variance)
    sat[8] = safe_clamp(1.0 - assessment_score_variance);

    // SAT[9] action_purity: assess.average_score * acts.success_rate * 2
    sat[9] = safe_clamp(assessment_mean * action_success_rate * 2.0);

    // SAT[10] recovery_speed REDESIGNED:
    //   1.0 - min(1, anchor_state.consecutive_failures / 10)
    {
        let consec_fails: f64 =
            field_or_default(anchor_state.as_ref(), "consecutive_failures", 0.0);
        sat[10] = safe_clamp(1.0 - (consec_fails / 10.0).min(1.0));
    }

    // SAT[11] environmental_adaptation: outer_spirit_history.environmental_adaptation
    sat[11] = safe_clamp(field_or_default(
        outer_spirit_history.as_ref(),
        "environmental_adaptation",
        0.5,
    ));

    // SAT[13] transactional_integrity REDESIGNED:
    //   0.5 if anchor_count==0 else anchor_count / (anchor_count + 5*consecutive_failures)
    {
        let anchor_count: f64 = field_or_default(anchor_state.as_ref(), "anchor_count", 0.0);
        let consec_fails: f64 =
            field_or_default(anchor_state.as_ref(), "consecutive_failures", 0.0);
        sat[13] = if anchor_count == 0.0 {
            0.5
        } else {
            safe_clamp(anchor_count / (anchor_count + 5.0 * consec_fails))
        };
    }

    // SAT[14] operational_vitality: min(1, acts.per_hour/20) * uptime
    sat[14] = safe_clamp((actions_per_hour / 20.0).min(1.0) * uptime_ratio);

    // ── CHIT overrides ─────────────────────────────────────────────
    // CHIT[0] world_model_depth REDESIGNED:
    //   0.25*KG.node_count_norm + 0.30*KG.edge_count_norm
    //   + 0.20*meta_cgn.primitives_grounded/total + 0.15*action_chains_norm
    //   + 0.10*vocab_total_norm
    //   (KG counts normalized by 2000; vocab not yet plumbed → 0.5)
    {
        let kg_nodes: f64 = field_or_default(knowledge_graph.as_ref(), "node_count", 0.0);
        let kg_edges: f64 = field_or_default(knowledge_graph.as_ref(), "edge_count", 0.0);
        let mc_primitives_grounded: f64 =
            field_or_default(meta_cgn.as_ref(), "primitives_grounded", 0.0);
        let mc_primitives_total: f64 =
            field_or_default(meta_cgn.as_ref(), "primitives_total", 8.0).max(1.0);
        let action_chains: f64 = mem_persistent_nodes; // closest available proxy
        let kg_node_norm = (kg_nodes / 2000.0).min(1.0);
        let kg_edge_norm = (kg_edges / 2000.0).min(1.0);
        let primitives_ratio = (mc_primitives_grounded / mc_primitives_total).min(1.0);
        let action_chains_norm = (action_chains / 100.0).min(1.0);
        let vocab_total_norm = 0.5; // vocab_stats not yet plumbed
        chit[0] = safe_clamp(
            0.25 * kg_node_norm
                + 0.30 * kg_edge_norm
                + 0.20 * primitives_ratio
                + 0.15 * action_chains_norm
                + 0.10 * vocab_total_norm,
        );
    }

    // CHIT[2] threat_discernment: confirmed/total_flags from jailbreak +
    //   verifier_high_severity / total_attempts
    {
        let jb_confirmed: f64 =
            field_or_default(jailbreak_alerts.as_ref(), "count_confirmed_24h", 0.0);
        let jb_total: f64 = field_or_default(jailbreak_alerts.as_ref(), "count_24h", 0.0);
        let ov_high_severity: f64 =
            field_or_default(output_verifier.as_ref(), "high_severity_violations", 0.0);
        let ov_attempts: f64 =
            field_or_default(output_verifier.as_ref(), "violation_events_24h", 0.0);
        let jb_part = jb_confirmed / jb_total.max(1.0);
        let ov_part = ov_high_severity / ov_attempts.max(1.0);
        chit[2] = safe_clamp((jb_part + ov_part) / 2.0);
    }

    // CHIT[6] knowledge_growth REDESIGNED:
    //   0.35*memory.learning_velocity + 0.25*vocab.producible_growth_per_day
    //   + 0.20*meta_cgn.primitives_grounded_delta_24h
    //   + 0.20*meta_cgn.compositions_computed_delta_24h
    {
        let mem_learning_velocity: f64 =
            field_or_default(memory_stats_ext.as_ref(), "learning_velocity", 0.5);
        let vocab_growth = 0.0; // vocab_stats.producible_growth_per_day not yet plumbed
        let mc_primitives_delta_24h: f64 = field_or_default(
            meta_cgn.as_ref(),
            "primitives_grounded_delta_24h",
            0.0,
        );
        let mc_compositions_delta_24h: f64 = field_or_default(
            meta_cgn.as_ref(),
            "compositions_computed_delta_24h",
            0.0,
        );
        chit[6] = safe_clamp(
            0.35 * mem_learning_velocity
                + 0.25 * vocab_growth
                + 0.20 * (mc_primitives_delta_24h / 3.0).min(1.0)
                + 0.20 * (mc_compositions_delta_24h / 3.0).min(1.0),
        );
    }

    // CHIT[7] information_quality REDESIGNED:
    //   0.5*meta_cgn.knowledge_helpful_ratio + 0.3*memory.directive_alignment
    //   + 0.2*assessment.research_avg_score
    {
        let mc_knowledge_helpful: f64 =
            field_or_default(meta_cgn.as_ref(), "knowledge_helpful_ratio", 0.5);
        let mem_directive_alignment: f64 =
            field_or_default(memory_stats_ext.as_ref(), "directive_alignment", 0.5);
        let assess_research_avg: f64 =
            field_or_default(assessment.as_ref(), "research_avg_score", 0.5);
        chit[7] = safe_clamp(
            0.5 * mc_knowledge_helpful + 0.3 * mem_directive_alignment + 0.2 * assess_research_avg,
        );
    }

    // CHIT[8] engagement_depth: assessment.average_score (alias)
    chit[8] = safe_clamp(assessment_mean);

    // CHIT[9] outcome_reflection: _clamp(0.5 + assessment.trend)
    chit[9] = safe_clamp(0.5 + assessment_trend);

    // CHIT[10] dream_recall: 0.6 * recall_ratio + 0.4 * body_coh
    //   (recall_ratio from osh.dream_recall_ratio)
    {
        let recall_ratio: f64 = field_or_default(
            outer_spirit_history.as_ref(),
            "dream_recall_ratio",
            0.5,
        );
        chit[10] = safe_clamp(0.6 * recall_ratio + 0.4 * outer_body_coh);
    }

    // CHIT[14] self_trajectory: outer_spirit_history.outer_spirit_trajectory
    chit[14] = safe_clamp(field_or_default(
        outer_spirit_history.as_ref(),
        "outer_spirit_trajectory",
        0.0,
    ));

    // ── ANANDA overrides ───────────────────────────────────────────
    // ANANDA[3] system_harmony REDESIGNED:
    //   1 - (bus.dropped / max(1, bus.published))
    {
        let bus_dropped: f64 = field_or_default(bus_stats.as_ref(), "dropped", 0.0);
        let bus_published: f64 = field_or_default(bus_stats.as_ref(), "published", 0.0);
        ananda[3] = safe_clamp(1.0 - (bus_dropped / bus_published.max(1.0)));
    }

    // ANANDA[5] information_accuracy REDESIGNED:
    //   0.5*meta_cgn.knowledge_helpful_ratio
    //   + 0.3*meta_cgn.haov.verified_rules_high_conf/10
    //   + 0.2*memory.directive_alignment
    {
        let mc_knowledge_helpful: f64 =
            field_or_default(meta_cgn.as_ref(), "knowledge_helpful_ratio", 0.5);
        let haov_high_conf: f64 = field_or_default(
            meta_cgn.as_ref(),
            "haov_verified_rules_high_conf",
            0.0,
        );
        let mem_directive_alignment: f64 =
            field_or_default(memory_stats_ext.as_ref(), "directive_alignment", 0.5);
        ananda[5] = safe_clamp(
            0.5 * mc_knowledge_helpful
                + 0.3 * (haov_high_conf / 10.0).min(1.0)
                + 0.2 * mem_directive_alignment,
        );
    }

    // ANANDA[6] community_connection: min(1, distinct_handles_24h / 5)
    //   Phase 2.5.E per-Titan via community_engagement_stats.
    {
        let distinct_handles: f64 = field_or_default(
            community_engagement.as_ref(),
            "distinct_handles_24h",
            0.0,
        );
        ananda[6] = safe_clamp((distinct_handles / 5.0).min(1.0));
    }

    // ANANDA[8] expression_reach: min(1, mean_engagement_delta_7d / 5)
    //   (producer pre-normalizes to [0,1], we still clamp defensively)
    {
        let mean_engagement_delta: f64 = field_or_default(
            community_engagement.as_ref(),
            "mean_engagement_delta_7d",
            0.0,
        );
        ananda[8] = safe_clamp((mean_engagement_delta / 5.0).min(1.0));
    }

    // ANANDA[9] discovery_value REDESIGNED:
    //   0.5*(meta_cgn.knowledge_requests_finalized/max(1, knowledge_requests_emitted))
    //   + 0.3*felt_experiences_to_action_rate + 0.2*cgn.consolidations_per_day_norm
    {
        let mc_finalized: f64 =
            field_or_default(meta_cgn.as_ref(), "knowledge_requests_finalized", 0.0);
        let mc_emitted: f64 =
            field_or_default(meta_cgn.as_ref(), "knowledge_requests_emitted", 0.0);
        let et_felt_to_action: f64 = field_or_default(
            events_teacher.as_ref(),
            "felt_experiences_to_action_rate",
            0.0,
        );
        let cgn_consolidations_norm: f64 =
            field_or_default(cgn.as_ref(), "consolidations_per_day_norm", 0.0);
        ananda[9] = safe_clamp(
            0.5 * (mc_finalized / mc_emitted.max(1.0))
                + 0.3 * et_felt_to_action
                + 0.2 * cgn_consolidations_norm,
        );
    }

    // ANANDA[13] resource_appreciation REDESIGNED:
    //   min(1, outputs_per_hour / max(1, agency.llm_calls_this_hour))
    //   (replaces legacy history.resource_efficiency passthrough)
    {
        let outputs_per_hour = creative_per_window + action_per_window; // proxy for outputs
        let llm_calls: f64 =
            field_or_default(agency.as_ref(), "llm_calls_this_hour", 0.0);
        ananda[13] = safe_clamp((outputs_per_hour / llm_calls.max(1.0)).min(1.0));
    }

    // Suppress unused-variable warnings for fields we kept in scope but
    // didn't end up using in overrides (reserved for future Phase 6.5 dims):
    let _ = expression_translator;

    let mut out = [0.0_f32; 45];
    for (i, v) in sat.iter().enumerate() {
        out[i] = *v as f32;
    }
    for (i, v) in chit.iter().enumerate() {
        out[i + 15] = *v as f32;
    }
    for (i, v) in ananda.iter().enumerate() {
        out[i + 30] = *v as f32;
    }
    Ok(out)
}

// ── SAT: Material Being (15D) — outer_spirit_tensor.py:104-164 ──
#[allow(clippy::too_many_arguments)]
fn compute_sat(
    total_actions: f64,
    action_success_rate: f64,
    actions_per_hour: f64,
    creative_total: f64,
    creative_unique_types: f64,
    threats_detected: f64,
    rejections: f64,
    sovereignty_ratio: f64,
    uptime_ratio: f64,
    recovery: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    solana: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    assessment_mean: f64,
    assessment_trend: f64,
    assessment_score_variance: f64,
) -> [f64; 15] {
    let mut sat = [0.5_f64; 15];

    // [0] world_recognition: identity_verified
    sat[0] = safe_clamp(field_or_default(solana, "identity_verified", 0.5));
    // [1] expressive_authenticity: action_stats.inner_outer_coherence
    //     — never set by _collect_extended → defaults 0.5
    sat[1] = safe_clamp(0.5);
    // [2] action_sovereignty: sovereignty_ratio
    sat[2] = safe_clamp(sovereignty_ratio);
    // [3] boundary_enforcement
    sat[3] = if threats_detected > 0.0 {
        safe_clamp(rejections / threats_detected.max(1.0))
    } else {
        0.8
    };
    // [4] operational_persistence: uptime_ratio
    sat[4] = safe_clamp(uptime_ratio);
    // [5] origin_anchoring: genesis_nft_exists
    sat[5] = safe_clamp(field_or_default(solana, "genesis_nft_exists", 0.5));
    // [6] observable_growth: 0.5 + assess.trend
    sat[6] = safe_clamp(0.5 + assessment_trend);
    // [7] world_footprint: (total_actions + creative_total + tx_count) / 200
    //     Python: total_outputs = acts.get("total", 0) + crea.get("total", 0)
    //                            + sol.get("tx_count", 0)
    //     acts.total = total_actions; crea.total = art_count + audio_count
    let tx_count: f64 = field_or_default(solana, "tx_count", 0.0);
    let total_outputs = total_actions + creative_total + tx_count;
    sat[7] = safe_clamp((total_outputs / 200.0).min(1.0));
    // [8] behavioral_consistency: 1 - score_variance
    sat[8] = safe_clamp(1.0 - assessment_score_variance);
    // [9] action_purity: mean_score × success_rate × 2.0
    sat[9] = safe_clamp(assessment_mean * action_success_rate * 2.0);
    // [10] recovery_speed: 1 / (1 + mean_recovery / 30)
    let mean_recovery_s: f64 =
        field_or_default(recovery, "mean_recovery_seconds", 60.0);
    sat[10] = safe_clamp(1.0 / (1.0 + mean_recovery_s / 30.0));
    // [11] environmental_adaptation: 1 - load_variance (default 0.3)
    let load_variance: f64 = 0.3; // assess.load_variance never set by _collect_extended
    sat[11] = safe_clamp(1.0 - load_variance);
    // [12] distinctive_voice: unique_types / 5
    sat[12] = safe_clamp((creative_unique_types / 5.0).min(1.0));
    // [13] transactional_integrity: tx_success_rate
    sat[13] = safe_clamp(field_or_default(solana, "tx_success_rate", 0.5));
    // [14] operational_vitality: actions_per_hour / 20 × uptime
    sat[14] = safe_clamp((actions_per_hour / 20.0).min(1.0) * uptime_ratio);

    sat
}

// ── CHIT: Material Awareness (15D) — outer_spirit_tensor.py:167-227 ──
#[allow(clippy::too_many_arguments)]
fn compute_chit(
    mem_persistent_nodes: f64,
    mem_growth_per_epoch: f64,
    threats_detected: f64,
    rejections: f64,
    social_mean_conversation_quality: f64,
    assessment_trend: f64,
    history: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    outer_body_coh: f64,
    outer_mind_coh: f64,
    combined_coh: f64,
) -> [f64; 15] {
    let mut chit = [0.5_f64; 15];

    // [0] world_model_depth: persistent_nodes / 2000
    chit[0] = safe_clamp((mem_persistent_nodes / 2000.0).min(1.0));
    // [1] signal_clarity: combined_coh
    chit[1] = safe_clamp(combined_coh);
    // [2] threat_discernment: confirmed_threats / threats_detected
    let confirmed_threats: f64 = field_or_default(None, "confirmed_threats", 0.0); // never plumbed
    let _ = confirmed_threats;
    chit[2] = if threats_detected > 0.0 {
        // Python: true_threats / max(1, total_flags); with no plumbing,
        // confirmed_threats=0 → chit[2] = 0; but Python also has `true_threats=0`
        // when guardian_stats has no "confirmed_threats" key (which is
        // ALWAYS the case since _collect_extended doesn't set it).
        // Match Python bytes-exact: when total_flags > 0 and confirmed_threats=0,
        // returns _clamp(0/max(1,total)) = 0. (The 0.8 only when total_flags=0.)
        // BUT: Python `guard.get("confirmed_threats", 0)` returns 0, then
        // `true_threats / max(1, total_flags)` = 0. Matches our `confirmed_threats=0`.
        safe_clamp(0.0 / threats_detected.max(1.0))
    } else {
        // Python uses `rejections > 0` separately for SAT[3], but CHIT[2]
        // also uses `total_flags > 0` test → falls to 0.8 when no threats.
        // We follow Python here: when total_flags == 0, return 0.8 default.
        // Note: rejections variable exists for a future plumbing of
        // `confirmed_threats`; not used in this branch.
        let _ = rejections;
        0.8
    };
    // [3] cross_domain_integration: multi_source_success — never set → 0.5
    chit[3] = 0.5;
    // [4] witness_stability: body_coh × mind_coh × 2
    chit[4] = safe_clamp(outer_body_coh * outer_mind_coh * 2.0);
    // [5] situation_recognition: pattern_reuse_rate — never set → 0.5
    chit[5] = 0.5;
    // [6] knowledge_growth: growth_rate / 10
    chit[6] = safe_clamp((mem_growth_per_epoch / 10.0).min(1.0));
    // [7] information_quality: research_usage_rate — fixed default 0.5
    chit[7] = 0.5;
    // [8] engagement_depth: mean_conversation_quality
    chit[8] = safe_clamp(social_mean_conversation_quality);
    // [9] outcome_reflection: 0.5 + assess.trend
    chit[9] = safe_clamp(0.5 + assessment_trend);
    // [10] dream_recall: history.dream_recall_ratio × 0.6 + body_coh × 0.4
    let dream_recall_ratio: f64 =
        field_or_default(history, "dream_recall_ratio", 0.0);
    chit[10] = safe_clamp(dream_recall_ratio * 0.6 + outer_body_coh * 0.4);
    // [11] temporal_context: history.circadian_alignment
    chit[11] = safe_clamp(field_or_default(history, "circadian_alignment", 0.5));
    // [12] network_awareness: body_coh
    chit[12] = safe_clamp(outer_body_coh);
    // [13] causal_attribution: assess.correlation_strength — never set → 0.5
    chit[13] = 0.5;
    // [14] self_trajectory: history.outer_spirit_trajectory
    chit[14] = safe_clamp(field_or_default(history, "outer_spirit_trajectory", 0.5));

    chit
}

// ── ANANDA: Material Fulfillment (15D) — outer_spirit_tensor.py:230-302 ──
#[allow(clippy::too_many_arguments)]
fn compute_ananda(
    action_success_rate: f64,
    action_error_rate: f64,
    failed_retry_rate: f64,
    burst_frequency: f64,
    social_mean_conversation_quality: f64,
    creative_mean_assessment: f64,
    interactions_per_window: f64,
    action_per_window: f64,
    creative_per_window: f64,
    hormone_levels: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    history: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    outer_body_coh: f64,
    outer_mind_coh: f64,
    assessment_mean: f64,
) -> [f64; 15] {
    let mut ananda = [0.5_f64; 15];

    // [0] purpose_effectiveness: success_rate
    ananda[0] = safe_clamp(action_success_rate);
    // [1] interaction_depth: mean_conversation_quality
    ananda[1] = safe_clamp(social_mean_conversation_quality);
    // [2] creative_impact: mean_assessment
    ananda[2] = safe_clamp(creative_mean_assessment);
    // [3] system_harmony: 1 - cross_module_error_rate (never set → 0.1)
    let cross_module_error_rate: f64 = 0.1;
    ananda[3] = safe_clamp(1.0 - cross_module_error_rate);
    // [4] aesthetic_quality: mean_assessment
    ananda[4] = safe_clamp(creative_mean_assessment);
    // [5] information_accuracy: research_accuracy — never set → 0.5
    ananda[5] = 0.5;
    // [6] community_connection: new_connections_per_window — never set → 0
    ananda[6] = safe_clamp((0.0_f64 / 5.0).min(1.0));
    // [7] capability_growth: novel_types_per_window — never set → 0
    ananda[7] = safe_clamp((0.0_f64 / 3.0).min(1.0));
    // [8] expression_reach: creative_engagement — never set → 0.5
    ananda[8] = 0.5;
    // [9] discovery_value: research_to_action_rate — never set → 0.5
    ananda[9] = 0.5;
    // [10] graceful_rest: history.rest_performance_floor → 0.5
    ananda[10] = safe_clamp(field_or_default(history, "rest_performance_floor", 0.5));
    // [11] creative_tension: CREATIVITY × min(1, time_since_last/600)
    let creativity_level: f64 = field_or_default(hormone_levels, "CREATIVITY", 0.0);
    let time_since_create: f64 =
        field_or_default(history, "seconds_since_last_create", 300.0);
    ananda[11] = safe_clamp(creativity_level * (time_since_create / 600.0).min(1.0));
    // [12] surrender_capacity: 1 - clamp((failed_retry + (1-body_coh) + burst) / 3)
    let resource_depletion = 1.0 - outer_body_coh;
    let surrender = 1.0
        - safe_clamp((failed_retry_rate + resource_depletion + burst_frequency) / 3.0);
    ananda[12] = safe_clamp(surrender);
    // [13] resource_appreciation: history.resource_efficiency → 0.5
    ananda[13] = safe_clamp(field_or_default(history, "resource_efficiency", 0.5));
    // [14] flow_state: min(body,mind) × (1-error) × mean_score × surrender
    let min_coherence = outer_body_coh.min(outer_mind_coh);
    let error_factor = 1.0 - action_error_rate;
    let surrender_gate = ananda[12];
    ananda[14] = safe_clamp(min_coherence * error_factor * assessment_mean * surrender_gate);
    // Note: Python ananda[14] uses `acts.get("error_rate", 0.1)` separately —
    // we computed action_error_rate from success_rate above. Match Python:
    // when total_actions == 0, success_rate = 0/max(1,0) = 0 → error_rate = 1.
    // Fall through with action_error_rate from caller — preserved.

    let _ = action_per_window;
    let _ = creative_per_window;
    let _ = interactions_per_window;

    ananda
}

// ── Helpers (mirror outer_mind 9I) ──────────────────────────────────

/// `_clamp(v, lo=0.0, hi=1.0)` from `outer_spirit_tensor.py:350-351`.
/// NaN → 0.5, then clamp to [0, 1].
fn safe_clamp(v: f64) -> f64 {
    if v.is_nan() {
        return 0.5;
    }
    v.clamp(0.0, 1.0)
}

fn mean_f32(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.5;
    }
    let s: f64 = v.iter().map(|x| *x as f64).sum();
    s / v.len() as f64
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

fn field_or_default(
    map: Option<&Vec<(rmpv::Value, rmpv::Value)>>,
    key: &str,
    default: f64,
) -> f64 {
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

fn top_level_f64_or(
    map: &[(rmpv::Value, rmpv::Value)],
    key: &str,
    default: f64,
) -> f64 {
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

fn encode_spirit_state_payload(spirit: &[f32; 45]) -> Vec<u8> {
    use rmpv::Value;
    let values = Value::Array(spirit.iter().map(|f| Value::F64(*f as f64)).collect());
    let map = Value::Map(vec![
        (Value::String("src".into()), Value::String("outer".into())),
        (
            Value::String("type".into()),
            Value::String("SPIRIT_STATE".into()),
        ),
        (Value::String("values".into()), values),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ]);
    let mut out = Vec::with_capacity(512);
    rmpv::encode::write_value(&mut out, &map)
        .expect("rmpv encode never fails on well-formed Value");
    out
}

/// Compose OUTER_SPIRIT_FILTER_DOWN payload per SPEC §8.6 row 4.
///
/// Payload shape:
///   {
///     multipliers: {
///       outer_body[5], outer_mind[15], outer_spirit_content[40]
///     },
///     ts: float
///   }
///
/// **Observer dims [0:5] are STRUCTURALLY EXCLUDED** from
/// `outer_spirit_content` per SPEC G8 — `extract_outer_spirit_content`
/// returns exactly `outer_spirit_45d[5:45]` (length 40, not 45).
///
/// **Logic-source for body+mind multiplier derivation:**
/// `filter_down.py:478-482`. Initial port emits **identity multipliers**
/// (1.0 for each body+mind slot) — the per-dim derivation from spirit
/// state is the parity-vector cycle's job (D2/D5 follow-up). The
/// observer-mask guarantee for outer_spirit_content[40] is ALREADY
/// ENFORCED here at type-level via extract_outer_spirit_content.
fn encode_outer_spirit_filter_down_payload(spirit: &[f32; 45]) -> Vec<u8> {
    use rmpv::Value;

    // OBSERVER MASK: structurally extracts [5:45] only, length 40.
    let outer_spirit_content: [f32; CONTENT_DIM_COUNT] = extract_outer_spirit_content(spirit);

    let outer_body_mults = vec![1.0_f64; 5];
    let outer_mind_mults = vec![1.0_f64; 15];
    let outer_spirit_content_mults: Vec<Value> = outer_spirit_content
        .iter()
        .map(|f| Value::F64(*f as f64))
        .collect();

    let multipliers = Value::Map(vec![
        (
            Value::String("outer_body".into()),
            Value::Array(outer_body_mults.into_iter().map(Value::F64).collect()),
        ),
        (
            Value::String("outer_mind".into()),
            Value::Array(outer_mind_mults.into_iter().map(Value::F64).collect()),
        ),
        (
            Value::String("outer_spirit_content".into()),
            Value::Array(outer_spirit_content_mults),
        ),
    ]);
    let map = Value::Map(vec![
        (Value::String("multipliers".into()), multipliers),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ]);
    let mut out = Vec::with_capacity(512);
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
    use titan_core::constants::OUTER_SPIRIT_45D_SCHEMA_VERSION;

    fn make_distinct_45d() -> [f32; 45] {
        let mut v = [0.0_f32; 45];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as f32 + 0.5;
        }
        v
    }

    fn empty_payload() -> Vec<u8> {
        use rmpv::Value;
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Map(vec![])).unwrap();
        bytes
    }

    #[test]
    fn project_45d_empty_dict_matches_python_defaults() {
        // Empty msgpack map → all derived stats default. Hand-computed
        // expected values per outer_spirit_tensor.py with neutral inputs
        // (outer_body=outer_mind=[0.5,...]):
        //   outer_body_coh = mind_coh = combined_coh = 0.5
        //
        //   SAT[0] identity_verified=0.5         → 0.5
        //   SAT[1] inner_outer_coherence=0.5     → 0.5
        //   SAT[2] sovereignty_ratio=0           → 0.0
        //   SAT[3] threats_detected=0            → 0.8 (no threats branch)
        //   SAT[4] uptime_ratio=1/(1+60/(1+60))=1/(1+60/61)≈0.504
        //          but uptime=1, ratio = 1/(1+60)=0.0164 → close to 0
        //   SAT[5] genesis_nft_exists=0.5        → 0.5
        //   SAT[6] 0.5+trend(0)                  → 0.5
        //   SAT[7] (0+0+0)/200                   → 0.0
        //   SAT[8] 1-score_variance(0.3)         → 0.7
        //   SAT[9] mean(0.5)*success(0)*2        → 0.0 (success_rate=0)
        //   SAT[10] 1/(1+60/30)                  → 1/3 ≈ 0.333
        //   SAT[11] 1-load_variance(0.3)         → 0.7
        //   SAT[12] unique_types(0)/5            → 0.0
        //   SAT[13] tx_success_rate=0.5          → 0.5
        //   SAT[14] (0/20)*uptime                → 0.0
        let payload = empty_payload();
        let body = [0.5_f32; 5];
        let mind = [0.5_f32; 15];
        let spirit = project_outer_spirit_45d(&payload, body, mind, [0.5; 45]).unwrap();

        // SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   SAT[3] boundary_enforcement REDESIGNED → 0.0 (was 0.8 from old branch)
        //   SAT[10] recovery_speed REDESIGNED → 1.0 (was 1/3, now 1 - 0/10)
        //   SAT[11] env_adapt REDESIGNED → 0.5 (osh.environmental_adaptation default)
        assert!((spirit[0] - 0.5).abs() < 1e-3, "SAT[0] = {}", spirit[0]);
        assert!((spirit[1] - 0.5).abs() < 1e-3);
        assert!((spirit[2] - 0.0).abs() < 1e-3);
        assert!((spirit[3] - 0.0).abs() < 1e-3, "SAT[3] = {}", spirit[3]);
        assert!(spirit[4] < 0.05, "SAT[4] = {} (expected ≈0)", spirit[4]);
        assert!((spirit[5] - 0.5).abs() < 1e-3);
        assert!((spirit[6] - 0.5).abs() < 1e-3);
        assert!((spirit[7] - 0.0).abs() < 1e-3);
        assert!((spirit[8] - 0.7).abs() < 1e-3);
        assert!((spirit[9] - 0.0).abs() < 1e-3);
        assert!((spirit[10] - 1.0).abs() < 1e-3, "SAT[10] = {}", spirit[10]);
        assert!((spirit[11] - 0.5).abs() < 1e-3, "SAT[11] = {}", spirit[11]);
        assert!((spirit[12] - 0.0).abs() < 1e-3);
        assert!((spirit[13] - 0.5).abs() < 1e-3);
        assert!((spirit[14] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn project_45d_chit_defaults() {
        // Empty payload, neutral body+mind:
        //   CHIT[0] persistent/2000=0           → 0.0
        //   CHIT[1] combined_coh=0.5            → 0.5
        //   CHIT[2] threats=0 → 0.8
        //   CHIT[3] multi_source_success=0.5    → 0.5
        //   CHIT[4] body*mind*2=0.5*0.5*2       → 0.5
        //   CHIT[5] pattern_reuse=0.5           → 0.5
        //   CHIT[6] growth/10=0                 → 0.0
        //   CHIT[7] research_usage=0.5          → 0.5
        //   CHIT[8] mean_conv_quality=0.5       → 0.5
        //   CHIT[9] 0.5+trend(0)                → 0.5
        //   CHIT[10] 0*0.6+0.5*0.4              → 0.2
        //   CHIT[11] circadian=0.5              → 0.5
        //   CHIT[12] body_coh=0.5               → 0.5
        //   CHIT[13] correlation=0.5            → 0.5
        //   CHIT[14] outer_spirit_trajectory=0.5 → 0.5
        let payload = empty_payload();
        let spirit =
            project_outer_spirit_45d(&payload, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   CHIT[0] world_model_depth REDESIGNED → 0.05 (KG=0+0.10*0.5 vocab default)
        //   CHIT[2] threat_discernment REDESIGNED → 0.0 (was 0.8, all defaults yield 0)
        //   CHIT[6] knowledge_growth REDESIGNED → 0.175 (0.35*0.5 mem_learning default)
        //   CHIT[10] dream_recall → 0.5 (0.6*0.5 + 0.4*0.5)
        //   CHIT[14] self_trajectory REDESIGNED → 0.0 (osh.outer_spirit_trajectory default)
        assert!((spirit[15] - 0.05).abs() < 1e-3, "CHIT[0] = {}", spirit[15]);
        assert!((spirit[16] - 0.5).abs() < 1e-3);
        assert!((spirit[17] - 0.0).abs() < 1e-3, "CHIT[2] = {}", spirit[17]);
        assert!((spirit[18] - 0.5).abs() < 1e-3);
        assert!((spirit[19] - 0.5).abs() < 1e-3);
        assert!((spirit[20] - 0.5).abs() < 1e-3);
        assert!((spirit[21] - 0.175).abs() < 1e-3, "CHIT[6] = {}", spirit[21]);
        assert!((spirit[22] - 0.5).abs() < 1e-3);
        assert!((spirit[23] - 0.5).abs() < 1e-3);
        assert!((spirit[24] - 0.5).abs() < 1e-3);
        assert!((spirit[25] - 0.5).abs() < 1e-3, "CHIT[10] = {}", spirit[25]);
        assert!((spirit[26] - 0.5).abs() < 1e-3);
        assert!((spirit[27] - 0.5).abs() < 1e-3);
        assert!((spirit[28] - 0.5).abs() < 1e-3);
        assert!((spirit[29] - 0.0).abs() < 1e-3, "CHIT[14] = {}", spirit[29]);
    }

    #[test]
    fn project_45d_ananda_defaults() {
        // Empty payload, neutral body+mind:
        //   ANANDA[0] success_rate=0          → 0.0
        //   ANANDA[1] mean_conv_quality=0.5   → 0.5
        //   ANANDA[2] mean_assessment=0.5     → 0.5
        //   ANANDA[3] 1-cross_err(0.1)        → 0.9
        //   ANANDA[4] mean_assessment=0.5     → 0.5
        //   ANANDA[5] research_accuracy=0.5   → 0.5
        //   ANANDA[6] new_conn=0/5            → 0.0
        //   ANANDA[7] novel/3=0               → 0.0
        //   ANANDA[8] creative_engagement=0.5 → 0.5
        //   ANANDA[9] research_to_action=0.5  → 0.5
        //   ANANDA[10] rest_perf=0.5          → 0.5
        //   ANANDA[11] CREATIVITY(0)*time     → 0.0
        //   ANANDA[12] 1-(0+0.5+0)/3 = 1-0.166 → 0.833
        //   ANANDA[13] resource_eff=0.5       → 0.5
        //   ANANDA[14] min(0.5,0.5) * (1-error(1)) * 0.5 * 0.833 → 0
        //              error_rate = 1 - success_rate = 1 - 0 = 1 (Python
        //              line 622: action_stats.error_rate = 1 - success_rate)
        let payload = empty_payload();
        let spirit =
            project_outer_spirit_45d(&payload, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   ANANDA[3] system_harmony REDESIGNED → 1.0 (1 - 0/max(1,0))
        //   ANANDA[5] information_accuracy REDESIGNED → 0.35 (0.5*0.5 + 0 + 0.2*0.5)
        //   ANANDA[8] expression_reach REDESIGNED → 0.0 (community_engagement.delta default)
        //   ANANDA[9] discovery_value REDESIGNED → 0.0 (all components default 0)
        //   ANANDA[13] resource_appreciation REDESIGNED → 0.0 (outputs=0)
        assert!((spirit[30] - 0.0).abs() < 1e-3, "ANANDA[0] = {}", spirit[30]);
        assert!((spirit[31] - 0.5).abs() < 1e-3);
        assert!((spirit[32] - 0.5).abs() < 1e-3);
        assert!((spirit[33] - 1.0).abs() < 1e-3, "ANANDA[3] = {}", spirit[33]);
        assert!((spirit[34] - 0.5).abs() < 1e-3);
        assert!((spirit[35] - 0.35).abs() < 1e-3, "ANANDA[5] = {}", spirit[35]);
        assert!((spirit[36] - 0.0).abs() < 1e-3);
        assert!((spirit[37] - 0.0).abs() < 1e-3);
        assert!((spirit[38] - 0.0).abs() < 1e-3, "ANANDA[8] = {}", spirit[38]);
        assert!((spirit[39] - 0.0).abs() < 1e-3, "ANANDA[9] = {}", spirit[39]);
        assert!((spirit[40] - 0.5).abs() < 1e-3);
        assert!((spirit[41] - 0.0).abs() < 1e-3);
        assert!((spirit[42] - 0.833_f32).abs() < 5e-3, "ANANDA[12] = {}", spirit[42]);
        assert!((spirit[43] - 0.0).abs() < 1e-3, "ANANDA[13] = {}", spirit[43]);
        assert!((spirit[44] - 0.0).abs() < 1e-3, "ANANDA[14] = {}", spirit[44]);
    }

    #[test]
    fn project_45d_realistic_high_engagement() {
        use rmpv::Value;
        // Realistic scenario: 100 actions, 80 success, healthy sentiment,
        // 200 memory nodes, healthy assessment.
        let payload = Value::Map(vec![
            (
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
                    (
                        Value::String("actions_this_hour".into()),
                        Value::Integer(10.into()),
                    ),
                    (
                        Value::String("creative_this_hour".into()),
                        Value::Integer(2.into()),
                    ),
                    (
                        Value::String("threats_detected".into()),
                        Value::Integer(5.into()),
                    ),
                    (
                        Value::String("rejections".into()),
                        Value::Integer(4.into()),
                    ),
                    (
                        Value::String("sovereignty_ratio".into()),
                        Value::F64(0.7),
                    ),
                ]),
            ),
            (
                Value::String("assessment_stats".into()),
                Value::Map(vec![
                    (
                        Value::String("average_score".into()),
                        Value::F64(0.7),
                    ),
                    (Value::String("trend".into()), Value::F64(0.05)),
                    (
                        Value::String("score_variance".into()),
                        Value::F64(0.2),
                    ),
                ]),
            ),
            (
                Value::String("memory_status".into()),
                Value::Map(vec![
                    (
                        Value::String("persistent_count".into()),
                        Value::Integer(200.into()),
                    ),
                    (
                        Value::String("growth_per_epoch".into()),
                        Value::F64(2.0),
                    ),
                ]),
            ),
            (
                Value::String("uptime_seconds".into()),
                Value::F64(3600.0),
            ),
            (
                Value::String("art_count_500".into()),
                Value::Integer(3.into()),
            ),
            (
                Value::String("audio_count_500".into()),
                Value::Integer(2.into()),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // Spot checks. SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   SAT[3] now uses jailbreak_alerts + output_verifier 24h fields
        //          (not in test → defaults 0/max(1,0)=0).
        //   CHIT[0] now uses KG + meta_cgn + memory.action_chains + vocab.
        //          With mem_persistent_nodes=200 used as action_chains proxy:
        //          0.25*0 + 0.30*0 + 0.20*0/8 + 0.15*min(1,200/100)=0.15
        //          + 0.10*0.5 (vocab default) = 0.20.
        //   CHIT[6] now uses memory.learning_velocity + meta_cgn deltas + vocab.
        //          With memory_stats not in test → all default 0.5/0:
        //          0.35*0.5 + 0.25*0 + 0.20*0 + 0.20*0 = 0.175.
        assert!((spirit[2] - 0.7).abs() < 1e-3, "SAT[2] = {}", spirit[2]);
        assert!((spirit[3] - 0.0).abs() < 1e-3, "SAT[3] = {}", spirit[3]);
        assert!((spirit[4] - 0.9836_f32).abs() < 1e-3, "SAT[4] = {}", spirit[4]);
        assert!((spirit[6] - 0.55).abs() < 1e-3);
        assert!((spirit[8] - 0.8).abs() < 1e-3);
        assert!((spirit[9] - 1.0).abs() < 1e-3);
        assert!((spirit[12] - 0.4).abs() < 1e-3);

        assert!((spirit[15] - 0.20).abs() < 1e-3, "CHIT[0] = {}", spirit[15]);
        assert!((spirit[21] - 0.175).abs() < 1e-3, "CHIT[6] = {}", spirit[21]);
        assert!((spirit[24] - 0.55).abs() < 1e-3);

        // ANANDA[0] = success_rate = 0.8
        assert!((spirit[30] - 0.8).abs() < 1e-3, "ANANDA[0] = {}", spirit[30]);
        // ANANDA[2] = mean_assessment = 0.7
        assert!((spirit[32] - 0.7).abs() < 1e-3);
    }

    #[test]
    fn project_45d_solana_drives_sat_bands() {
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("solana_stats".into()),
            Value::Map(vec![
                (
                    Value::String("identity_verified".into()),
                    Value::F64(0.9),
                ),
                (
                    Value::String("genesis_nft_exists".into()),
                    Value::F64(1.0),
                ),
                (
                    Value::String("tx_success_rate".into()),
                    Value::F64(0.95),
                ),
                (
                    Value::String("tx_count".into()),
                    Value::Integer(50.into()),
                ),
            ]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   SAT[13] transactional_integrity REDESIGNED — no longer uses
        //   solana_stats.tx_success_rate. Now: anchor_count==0 → 0.5
        //   (anchor_state not in test). SAT[0,5] still legacy compute_sat
        //   reading solana_stats.identity_verified / genesis_nft_exists.
        assert!((spirit[0] - 0.9).abs() < 1e-3, "SAT[0] = {}", spirit[0]);
        assert!((spirit[5] - 1.0).abs() < 1e-3);
        assert!((spirit[13] - 0.5).abs() < 1e-3, "SAT[13] = {}", spirit[13]);
    }

    #[test]
    fn project_45d_outer_spirit_history_drives_chit_and_ananda() {
        // SPEC §23.9 redesign overrides applied (Step 6 §4.3 P3):
        //   CHIT[10] dream_recall now reads outer_spirit_history_stats.dream_recall_ratio
        //            (not history.dream_recall_ratio).
        //   CHIT[14] self_trajectory now reads outer_spirit_history_stats.outer_spirit_trajectory.
        //   ANANDA[13] resource_appreciation REDESIGNED — no longer
        //              history.resource_efficiency; now outputs_per_hour /
        //              max(1, agency.llm_calls_this_hour). With outputs=0 → 0.
        //   ANANDA[10] graceful_rest still legacy compute_ananda path
        //              reading history.rest_performance_floor.
        //   CHIT[11] circadian_alignment unchanged — still legacy path.
        use rmpv::Value;
        let payload = Value::Map(vec![
            (
                Value::String("outer_spirit_history_stats".into()),
                Value::Map(vec![
                    (Value::String("dream_recall_ratio".into()), Value::F64(0.8)),
                    (
                        Value::String("outer_spirit_trajectory".into()),
                        Value::F64(0.6),
                    ),
                ]),
            ),
            (
                Value::String("history".into()),
                Value::Map(vec![
                    (
                        Value::String("circadian_alignment".into()),
                        Value::F64(0.7),
                    ),
                    (
                        Value::String("rest_performance_floor".into()),
                        Value::F64(0.85),
                    ),
                    (
                        Value::String("seconds_since_last_create".into()),
                        Value::F64(180.0),
                    ),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // CHIT[10] = 0.6*0.8 + 0.4*0.5 = 0.68
        assert!((spirit[25] - 0.68).abs() < 1e-3, "CHIT[10] = {}", spirit[25]);
        assert!((spirit[26] - 0.7).abs() < 1e-3);
        assert!((spirit[29] - 0.6).abs() < 1e-3, "CHIT[14] = {}", spirit[29]);
        assert!((spirit[40] - 0.85).abs() < 1e-3);
        // ANANDA[13] now 0 (outputs=0 with no agency_stats).
        assert!((spirit[43] - 0.0).abs() < 1e-3, "ANANDA[13] = {}", spirit[43]);
    }

    #[test]
    fn project_45d_hormone_drives_ananda_11() {
        use rmpv::Value;
        // ANANDA[11] = CREATIVITY × min(1, time_since_create / 600)
        // = 0.6 × min(1, 300/600) = 0.6 × 0.5 = 0.3
        let payload = Value::Map(vec![
            (
                Value::String("hormone_levels".into()),
                Value::Map(vec![(
                    Value::String("CREATIVITY".into()),
                    Value::F64(0.6),
                )]),
            ),
            (
                Value::String("history".into()),
                Value::Map(vec![(
                    Value::String("seconds_since_last_create".into()),
                    Value::F64(300.0),
                )]),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        assert!((spirit[41] - 0.3).abs() < 1e-3, "ANANDA[11] = {}", spirit[41]);
    }

    #[test]
    fn project_45d_anchor_state_drives_sat_10() {
        // SPEC §23.9 SAT[10] recovery_speed REDESIGNED 2026-05-07:
        //   1.0 - min(1, anchor_state.consecutive_failures / 10)
        // (replaces legacy recovery_stats.mean_recovery_seconds path).
        use rmpv::Value;
        let payload = Value::Map(vec![(
            Value::String("anchor_state".into()),
            Value::Map(vec![(
                Value::String("consecutive_failures".into()),
                Value::F64(2.5),
            )]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], [0.5; 45]).unwrap();

        // SAT[10] = 1 - 2.5/10 = 0.75
        assert!((spirit[10] - 0.75).abs() < 1e-3, "SAT[10] = {}", spirit[10]);
    }

    #[test]
    fn project_45d_high_coherence_lifts_witness_stability() {
        // CHIT[4] = body_coh × mind_coh × 2.0
        // body=mind=0.8 → 0.8×0.8×2 = 1.28 → clamped 1.0
        let payload = empty_payload();
        let body = [0.8_f32; 5];
        let mind = [0.8_f32; 15];
        let spirit = project_outer_spirit_45d(&payload, body, mind, [0.5; 45]).unwrap();
        assert!((spirit[19] - 1.0).abs() < 1e-3, "CHIT[4] = {}", spirit[19]);
    }

    #[test]
    fn project_45d_malformed_envelope_returns_fallback() {
        // Top-level msgpack value that is NOT a map → fallback returned.
        use rmpv::Value;
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Integer(42.into())).unwrap();
        let mut fallback = [0.0_f32; 45];
        for (i, v) in fallback.iter_mut().enumerate() {
            *v = i as f32 / 45.0;
        }
        let spirit =
            project_outer_spirit_45d(&bytes, [0.5_f32; 5], [0.5_f32; 15], fallback).unwrap();
        for i in 0..45 {
            assert!((spirit[i] - fallback[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn spirit_state_payload_contains_full_45d_unmasked() {
        let spirit = make_distinct_45d();
        let bytes = encode_spirit_state_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("values") {
                        if let Value::Array(arr) = val {
                            assert_eq!(arr.len(), 45, "SPIRIT_STATE must have all 45 dims");
                            if let Value::F64(f) = arr[0] {
                                assert!(
                                    (f - 0.5).abs() < 1e-6,
                                    "observer dim [0] in SPIRIT_STATE should be unmasked"
                                );
                            }
                            return;
                        }
                    }
                }
            }
        }
        panic!("values field missing or wrong shape");
    }

    #[test]
    fn spirit_state_src_is_outer() {
        let spirit = [0.0_f32; 45];
        let bytes = encode_spirit_state_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
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
    fn filter_down_payload_observer_dims_excluded_per_g8() {
        // CRITICAL G8 observance: outer_spirit_content[40] in
        // OUTER_SPIRIT_FILTER_DOWN must be EXACTLY [5:45] of the 45D
        // input — observer dims [0:5] never leak into the bus message.
        let mut spirit = make_distinct_45d();
        for slot in spirit[0..5].iter_mut() {
            *slot = 99.99;
        }
        let bytes = encode_outer_spirit_filter_down_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        let multipliers = if let Value::Map(items) = v {
            items.into_iter().find_map(|(k, val)| {
                if let Value::String(s) = k {
                    if s.as_str() == Some("multipliers") {
                        return Some(val);
                    }
                }
                None
            })
        } else {
            None
        }
        .expect("multipliers field");

        let mults_map = if let Value::Map(m) = multipliers {
            m
        } else {
            panic!()
        };
        let content = mults_map
            .into_iter()
            .find_map(|(k, val)| {
                if let Value::String(s) = k {
                    if s.as_str() == Some("outer_spirit_content") {
                        return Some(val);
                    }
                }
                None
            })
            .expect("outer_spirit_content field");

        let arr = if let Value::Array(a) = content {
            a
        } else {
            panic!()
        };
        assert_eq!(arr.len(), 40, "outer_spirit_content MUST be 40D per G8");
        for v in &arr {
            if let Value::F64(f) = v {
                assert_ne!(
                    *f, 99.99,
                    "observer sentinel leaked into outer_spirit_content"
                );
            }
        }
    }

    #[test]
    fn filter_down_payload_outer_body_is_5d() {
        let spirit = [0.5_f32; 45];
        let bytes = encode_outer_spirit_filter_down_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        let mults = if let Value::Map(items) = v {
            items.into_iter().find_map(|(k, val)| {
                if let Value::String(s) = k {
                    if s.as_str() == Some("multipliers") {
                        return Some(val);
                    }
                }
                None
            })
        } else {
            None
        }
        .unwrap();
        if let Value::Map(m) = mults {
            for (k, val) in m {
                if let Value::String(s) = k {
                    if s.as_str() == Some("outer_body") {
                        if let Value::Array(arr) = val {
                            assert_eq!(arr.len(), 5);
                            return;
                        }
                    }
                }
            }
        }
        panic!("outer_body field missing or wrong shape");
    }

    #[test]
    fn filter_down_payload_outer_mind_is_15d() {
        let spirit = [0.5_f32; 45];
        let bytes = encode_outer_spirit_filter_down_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("multipliers") {
                        if let Value::Map(m) = val {
                            for (k2, v2) in m {
                                if let Value::String(s2) = k2 {
                                    if s2.as_str() == Some("outer_mind") {
                                        if let Value::Array(arr) = v2 {
                                            assert_eq!(arr.len(), 15);
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        panic!("outer_mind field missing");
    }

    #[test]
    fn slot_write_45d_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("outer_spirit_45d.bin");
        let mut slot = Slot::create(&path, OUTER_SPIRIT_45D_SCHEMA_VERSION as u32, 180).unwrap();
        let spirit = [0.1_f32; 45];
        let bytes = encode_floats::<45>(&spirit);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn decode_unified_outer_spirit_content_extracts_40d() {
        use rmpv::Value;
        let mults_arr: Vec<Value> = (0..40).map(|i| Value::F64(0.3 + i as f64 * 0.05)).collect();
        let payload = Value::Map(vec![(
            Value::String("multipliers".into()),
            Value::Map(vec![(
                Value::String("outer_spirit_content".into()),
                Value::Array(mults_arr),
            )]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let mults = decode_unified_outer_spirit_content(&bytes).unwrap();
        assert_eq!(mults.len(), 40);
    }

    #[test]
    fn decode_unified_outer_spirit_errors_on_wrong_dim() {
        use rmpv::Value;
        let mults_arr: Vec<Value> = (0..39).map(|_| Value::F64(1.0)).collect();
        let payload = Value::Map(vec![(
            Value::String("multipliers".into()),
            Value::Map(vec![(
                Value::String("outer_spirit_content".into()),
                Value::Array(mults_arr),
            )]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        assert!(decode_unified_outer_spirit_content(&bytes).is_err());
    }

    #[test]
    fn content_hash_gates_redundant_spirit_writes() {
        let mut gate = ContentGate::new();
        let spirit = [0.1_f32; 45];
        let bytes = encode_floats::<45>(&spirit);
        assert!(gate.should_write(&bytes));
        assert!(!gate.should_write(&bytes));
        assert_eq!(gate.write_count(), 1);
    }

    #[test]
    fn stale_threshold_is_3x_cadence() {
        assert_eq!(outer_spirit_stale_threshold_s(), 90.0);
    }

    #[test]
    fn cadence_constants_match_spec() {
        // Sensor sidecar refresh cadence (post-A.S8: stale threshold source).
        assert_eq!(OUTER_SPIRIT_TICK_BASE_S, 30.0);
        // Bus publish throttle (post-A.S8 D2 — Schumann spirit/3 × 13 ≈ 5s).
        assert_eq!(OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S, 5.0);
    }

    #[test]
    fn schumann_spirit_period_is_canonical() {
        // Daemon ticks at SCHUMANN_SPIRIT_HZ (70.47 Hz, ~14.2ms) per
        // post-A.S8 D2 cadence migration. Verifies SchumannGenerator
        // wired to the correct role.
        let g = SchumannGenerator::new(SchumannRole::Spirit, tokio::time::Instant::now());
        let period_ns = g.period_ns();
        // Spirit period: 1/70.47s ≈ 14_190_435 ns ± 1
        assert!(
            (period_ns as i64 - 14_190_435).abs() <= 1,
            "spirit period_ns = {period_ns}, expected 14190435 ± 1"
        );
    }

    #[test]
    fn safe_clamp_handles_nan() {
        assert_eq!(safe_clamp(f64::NAN), 0.5);
        assert_eq!(safe_clamp(2.0), 1.0);
        assert_eq!(safe_clamp(-1.0), 0.0);
        assert_eq!(safe_clamp(0.5), 0.5);
    }

    #[test]
    fn project_45d_distinct_dim_values_under_realistic_input() {
        // Closes the GAP-CS6-001 stub regression: under realistic input,
        // the 45D output must contain MANY distinct values (the stub
        // produced exactly 4 distinct values across 45 dims).
        use rmpv::Value;
        let payload = Value::Map(vec![
            (
                Value::String("agency_stats".into()),
                Value::Map(vec![
                    (
                        Value::String("total_actions".into()),
                        Value::Integer(50.into()),
                    ),
                    (
                        Value::String("failed_actions".into()),
                        Value::Integer(7.into()),
                    ),
                    (
                        Value::String("actions_this_hour".into()),
                        Value::Integer(8.into()),
                    ),
                    (
                        Value::String("threats_detected".into()),
                        Value::Integer(3.into()),
                    ),
                    (
                        Value::String("rejections".into()),
                        Value::Integer(2.into()),
                    ),
                    (
                        Value::String("sovereignty_ratio".into()),
                        Value::F64(0.6),
                    ),
                ]),
            ),
            (
                Value::String("assessment_stats".into()),
                Value::Map(vec![
                    (
                        Value::String("average_score".into()),
                        Value::F64(0.65),
                    ),
                    (Value::String("trend".into()), Value::F64(0.03)),
                ]),
            ),
            (
                Value::String("uptime_seconds".into()),
                Value::F64(7200.0),
            ),
            (
                Value::String("art_count_500".into()),
                Value::Integer(2.into()),
            ),
            (
                Value::String("audio_count_500".into()),
                Value::Integer(1.into()),
            ),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &payload).unwrap();
        let body = [0.55_f32, 0.45, 0.5, 0.4, 0.65];
        let mind = [0.5_f32; 15];
        let spirit = project_outer_spirit_45d(&bytes, body, mind, [0.5; 45]).unwrap();

        // Count distinct values rounded to 3 decimals
        use std::collections::HashSet;
        let distinct: HashSet<i32> = spirit.iter().map(|v| (v * 1000.0) as i32).collect();
        assert!(
            distinct.len() >= 10,
            "Expected ≥10 distinct dim values under realistic input, got {} ({:?})",
            distinct.len(),
            spirit,
        );
    }
}
