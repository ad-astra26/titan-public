//! tick_loop — inner-spirit 70.47 Hz Schumann tick + Observer Principle.
//!
//! Per SPEC §9.A `titan-inner-spirit-rs` + master plan §10.5 chunk C5-4 +
//! Preamble G8 (Observer Principle) + G9 (content/observer split) + G13.
//!
//! # Spirit tensor composition (Observer Principle G8)
//!
//! Inner-spirit READS its sibling slots (`inner_body_5d.bin` +
//! `inner_mind_15d.bin`) every tick — it is NOT pure compute. The 45D
//! spirit tensor's three aspects (SAT / CHIT / ANANDA) are derived from:
//! - body input (interoception, proprioception, somatosensation, entropy,
//!   thermal) — provides "embodiment" component of presence
//! - mind input (thinking + feeling + willing) — provides "knowing" +
//!   "willing-bliss" components
//! - pre-aggregated experiential signal from `sensor_cache_inner_spirit`
//!   (Python sidecar-written: consciousness depth, hormone levels,
//!   memory stats — NOT real-time sensor input but smoothed L2 state)
//!
//! Daemon-side composition is DELIBERATELY simple: dim N is a stable
//! linear combination of inputs. The deeper learned spirit tensor lives
//! in `titan-unified-spirit-rs` (C-S4 V5 engine); the daemon's job is to
//! produce a 45D output every 14.2 ms that reflects current Trinity state.
//!
//! # G9 content/observer split
//!
//! The 45D tensor splits into:
//! - observer dims `[0:5]` (= absolute SELF[20:25]): WITNESS — never
//!   participate in filter_down output (G8). Computed but masked.
//! - content dims `[5:45]` (40D): SAT[5:15] + CHIT[15:30] + ANANDA[30:45]
//!   excluding the first 5D observer slice.
//!
//! Some practitioners read this as "SAT[0:15] / CHIT[15:30] / ANANDA[30:45]
//! with `[0:5]` of SAT being the Observer slice". Either framing is
//! consistent with G8 + G9 (the 5 observer dims live within SAT;
//! filter_down masks them at publish time).
//!
//! # Filter_down LOCAL publish (Phase C addition per SPEC §10.F)
//!
//! After computing the 45D tensor, inner-spirit publishes
//! `INNER_SPIRIT_FILTER_DOWN` carrying 60 multipliers
//! (inner_body[5] + inner_mind[15] + inner_spirit_content[40]).
//! These cascade DOWN to inner_body + inner_mind as a LOCAL bias on top
//! of the GLOBAL UNIFIED_SPIRIT_FILTER_DOWN.
//!
//! Observer dims [0:5] are NEVER published (G8). The
//! `inner_spirit_content` field carries dims [5:45] (40D).
//!
//! For C-S5 MVP the LOCAL multipliers are derived as `1.0 + (spirit -
//! 0.5) * 0.1` per dim (gentle bias proportional to spirit deviation
//! from neutral 0.5). Spirit-strength scaling (0.3 toward 1.0) attenuates
//! per `apply_spirit_strength`. EMA smoothing across consecutive
//! publishes prevents jerky bias changes (mirrors V5 publish-side
//! smoothing per filter_down.py:742-743).

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{INNER_SPIRIT_FIRING_MAX_BYTES, INNER_SPIRIT_FIRING_SCHEMA_VERSION};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, apply_spirit_strength, decode_filter_down_payload, encode_floats,
    read_dim_slice, ContentGate, DriftAggregator, EmaSmoother, FiringSlotWriter,
    INNER_SPIRIT_TOPICS, MULTIPLIER_CEIL, MULTIPLIER_FLOOR, SPIRIT_FILTER_STRENGTH_MULT,
};

const SPIRIT_DIMS: usize = 45;
const DRIFT_THRESHOLD_PCT: f64 = 0.005;
/// Content slice (everything except observer) — used for filter_down output.
/// Per G8 + G9: observer dims `[0:5]` are NEVER published as multipliers
/// (mask happens by construction here — `inner_spirit_content` field carries
/// only `[5:45]` = 40D).
const CONTENT_RANGE: std::ops::Range<usize> = 5..45;
const CONTENT_DIMS: usize = 40;

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "inner-spirit")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(INNER_SPIRIT_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?INNER_SPIRIT_TOPICS);

    let inner_spirit_slot = open_slot(shm_dir, "inner_spirit_45d.bin")?;
    let body_slot = open_slot(shm_dir, "inner_body_5d.bin")?;
    let mind_slot = open_slot(shm_dir, "inner_mind_15d.bin")?;
    // sensor_cache_inner_spirit.bin: created by Python sidecar
    // (inner_spirit_sensor_refresh in spirit_worker heartbeat-stub mode),
    // spawns LATER than this Rust daemon — Option<Slot> at boot, retry
    // open in tick loop. C-S5 closure 2026-05-08.
    let sensor_cache_path = shm_dir.join("sensor_cache_inner_spirit.bin");
    let sensor_cache = Slot::open(&sensor_cache_path).ok();
    info!(event = "SHM_OPENED", sensor_cache_present = sensor_cache.is_some());

    // Phase C 130D dim-live tracker bridge (rFP §4.7).
    let firing_writer = FiringSlotWriter::new(
        "inner_spirit",
        shm_dir,
        INNER_SPIRIT_FIRING_SCHEMA_VERSION as u32,
        INNER_SPIRIT_FIRING_MAX_BYTES as u32,
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
        inner_spirit_slot,
        body_slot,
        mind_slot,
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
    /// UNIFIED inner_spirit_content multipliers (40D, observer already
    /// masked at publish side).
    unified_spirit_content: Option<[f32; CONTENT_DIMS]>,
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
    if msg_type == "UNIFIED_SPIRIT_FILTER_DOWN" {
        let payload = match titan_bus::client::extract_payload(raw_bytes) {
            Some(p) => p,
            None => return,
        };
        match decode_filter_down_payload(&payload) {
            Ok(p) => {
                if let Ok(mut s) = state.lock() {
                    s.unified_spirit_content = Some(p.inner_spirit_content);
                }
            }
            Err(e) => warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN failed"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_tick_loop(
    bus: Arc<BusClient>,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<Notify>,
    mut inner_spirit_slot: Slot,
    body_slot: Slot,
    mind_slot: Slot,
    mut sensor_cache: Option<Slot>,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
) -> Result<()> {
    let mut content_gate = ContentGate::new();
    // EMA smoothers for the 3 LOCAL filter_down output fields.
    let mut ema_body = EmaSmoother::new(5);
    let mut ema_mind = EmaSmoother::new(15);
    let mut ema_spirit_content = EmaSmoother::new(CONTENT_DIMS);
    let mut drift_agg = DriftAggregator::new("spirit", DRIFT_THRESHOLD_PCT);

    // Per master plan §7 + C-S3 PLAN §1.1 #2: Schumann timer wheels live in
    // titan-schumann (canonical shared library for trinity daemons).
    let epoch_t0 = tokio::time::Instant::now();
    let generator = SchumannGenerator::new(SchumannRole::Spirit, epoch_t0);
    let period_ns = generator.period_ns();
    let mut tick_rx = generator.spawn(shutdown.clone());

    info!(event = "TICK_LOOP_START", role = "spirit", period_ns);

    // Retry-open sensor cache every ~1s when None — Python sidecar
    // starts AFTER this Rust daemon. C-S5 closure 2026-05-08.
    let mut tick_count: u64 = 0;
    let retry_every_n: u64 = (1.0_f64 / 0.01419).ceil() as u64; // ~71 ticks ≈ 1s at 70.47 Hz

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
                        &bus, &state, &mut content_gate,
                        &mut ema_body, &mut ema_mind, &mut ema_spirit_content,
                        &mut inner_spirit_slot, &body_slot, &mind_slot, &sensor_cache,
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
    ema_body: &mut EmaSmoother,
    ema_mind: &mut EmaSmoother,
    ema_spirit_content: &mut EmaSmoother,
    inner_spirit_slot: &mut Slot,
    body_slot: &Slot,
    mind_slot: &Slot,
    sensor_cache: &Option<Slot>,
    firing_writer: &mut FiringSlotWriter,
) -> Result<()> {
    // 1. Observer Principle G8: read sibling body + mind slots.
    let body = read_dim_slice::<5>(body_slot).map_err(|e| anyhow!("read body: {e}"))?;
    let mind = read_dim_slice::<15>(mind_slot).map_err(|e| anyhow!("read mind: {e}"))?;

    // 2. Read sensor_cache_inner_spirit (45D Python-computed canonical
    //    spirit_tensor.collect_spirit_45d output per SPEC §G1 + §23.6).
    //    When the Python sidecar (logic/inner_spirit_sensor_refresh.py)
    //    is running, this path returns the canonical 45D and we skip the
    //    Rust-side `compose_spirit_tensor` MVP entirely — produces
    //    byte-identical Python ↔ Rust spirit values.
    //
    //    When sensor cache is absent (cold boot, sidecar starting, Python
    //    writer crashed), fall back to the local MVP `compose_spirit_tensor`
    //    that derives 45D from body + mind alone. This keeps the daemon
    //    resilient — degraded but not stuck.
    let mut spirit: [f32; SPIRIT_DIMS] = match read_spirit_cache(sensor_cache) {
        Some(cache_45d) => cache_45d, // canonical Python compute (preferred)
        None => compose_spirit_tensor(&body, &mind, &[0.0_f32; 5]),
    };

    // 4. Apply UNIFIED filter_down to spirit content [5:45] (NOT observer).
    let unified_content = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.unified_spirit_content
    };
    if let Some(u) = unified_content {
        // Observer dims [0:5] are NEVER filtered (G8). Content dims [5:45]
        // get the unified multiplier applied.
        apply_multipliers(&mut spirit[CONTENT_RANGE], &u);
    }

    // 5. Encode + content-hash gate slot write.
    let bytes = encode_floats::<SPIRIT_DIMS>(&spirit);
    if content_gate.should_write(&bytes) {
        inner_spirit_slot
            .write(&bytes)
            .map_err(|e| anyhow!("slot write: {e}"))?;
    }

    // 5b. Phase C dim-live tracker bridge (rFP §4.7) — record per tick
    //     (matches SPIRIT_STATE bus publish cadence). inputs_state = &[]
    //     until inner sensor_cache schema migrates to source-dict in P6.
    firing_writer.record_tick(&spirit, &[], now_secs());

    // 6. Publish SPIRIT_STATE (P1 coalesce, src=inner) every tick.
    let payload = encode_spirit_state_payload(&spirit);
    bus.publish("SPIRIT_STATE", Some("all"), Some(&payload))
        .await
        .map_err(|e| anyhow!("publish SPIRIT_STATE: {e}"))?;

    // 7. Compute LOCAL filter_down multipliers for inner_body + inner_mind +
    //    inner_spirit_content. Per SPEC §10.F + filter_down.py:737-740 +
    //    apply_spirit_strength: gentle bias proportional to spirit deviation
    //    from neutral 0.5; observer dims masked at publish.
    let (body_mults, mind_mults, content_mults) =
        compute_local_filter_down(&spirit, ema_body, ema_mind, ema_spirit_content);

    // 8. Publish INNER_SPIRIT_FILTER_DOWN (P1, LOCAL bias) — observer dims
    //    NEVER appear in inner_spirit_content (G8: it's already only [5:45]
    //    = 40D content).
    let local_payload = encode_local_filter_down_payload(&body_mults, &mind_mults, &content_mults);
    bus.publish(
        "INNER_SPIRIT_FILTER_DOWN",
        Some("all"),
        Some(&local_payload),
    )
    .await
    .map_err(|e| anyhow!("publish INNER_SPIRIT_FILTER_DOWN: {e}"))?;

    Ok(())
}

/// Pure-compute spirit tensor from body + mind + (optional) cache. 45D
/// output per SPEC §7.1 + Preamble G3 / G8.
///
/// Composition is INTENTIONALLY simple for the daemon MVP:
///
/// - SAT[0:15]: presence/being. dims [0:5] = body itself (Observer slice
///   per G8 — preserves embodiment witness); dims [5:10] = mean of body +
///   thinking[0:5]; dims [10:15] = mean of body + feeling[5:10].
/// - CHIT[15:30]: knowing/awareness. dims [15:25] = mean of thinking+feeling
///   pairwise (10D); dims [25:30] = mean of mind willing[10:15] doubled.
/// - ANANDA[30:45]: bliss/resonance. dims [30:40] = mind itself (10 of 15);
///   dims [40:45] = mean of body + mind willing.
///
/// All values clamped to `[0, 1]`. The deeper learned spirit tensor lives
/// in `titan-unified-spirit-rs::self_assembly` (C-S4); this MVP daemon
/// produces stable derived state for downstream consumers + filter_down
/// publish input.
pub fn compose_spirit_tensor(
    body: &[f32; 5],
    mind: &[f32; 15],
    cache: &[f32; 5],
) -> [f32; SPIRIT_DIMS] {
    let mut s = [0.0_f32; SPIRIT_DIMS];

    // SAT[0:15] — presence
    for i in 0..5 {
        s[i] = body[i]; // observer slice
        s[5 + i] = (body[i] + mind[i]) * 0.5; // body × thinking
        s[10 + i] = (body[i] + mind[5 + i]) * 0.5; // body × feeling
    }
    // CHIT[15:30] — knowing
    for i in 0..5 {
        s[15 + i] = (mind[i] + mind[5 + i]) * 0.5; // thinking × feeling pair 1
    }
    for i in 0..5 {
        s[20 + i] = (mind[5 + i] + mind[10 + i]) * 0.5; // feeling × willing
    }
    s[25..30].copy_from_slice(&mind[10..15]); // willing direct
                                              // ANANDA[30:45] — bliss/resonance, modulated by sensor cache
    s[30..35].copy_from_slice(&mind[0..5]); // thinking direct
    s[35..40].copy_from_slice(&mind[5..10]); // feeling direct
    for i in 0..5 {
        s[40 + i] = (body[i] + mind[10 + i] + cache[i]) / 3.0;
    }

    // Clamp to [0, 1]
    for v in s.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    s
}

/// Compute LOCAL filter_down multipliers from spirit tensor.
/// Output: (body[5], mind[15], spirit_content[40]) — observer-masked.
pub fn compute_local_filter_down(
    spirit: &[f32; SPIRIT_DIMS],
    ema_body: &mut EmaSmoother,
    ema_mind: &mut EmaSmoother,
    ema_spirit_content: &mut EmaSmoother,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Step 1: derive raw multipliers from spirit. For body + mind use the
    // mean of relevant spirit dims as the bias signal.
    //
    // body bias from SAT body-related slice [0:5] (the observer slice
    // serves as the body's spirit-witness; G8 says observer dims are
    // never PUBLISHED but here we're using them as INPUT to compute body
    // multipliers — that's allowed).
    let mut body_raw = [0.0_f32; 5];
    for i in 0..5 {
        body_raw[i] = 1.0 + (spirit[i] - 0.5) * 0.1;
    }
    // mind bias from CHIT [15:30] — knowing pairs.
    let mut mind_raw = [0.0_f32; 15];
    for i in 0..15 {
        mind_raw[i] = 1.0 + (spirit[15 + i] - 0.5) * 0.1;
    }
    // spirit_content bias from spirit content [5:45] (40D excluding observer).
    let mut content_raw = [0.0_f32; CONTENT_DIMS];
    for i in 0..CONTENT_DIMS {
        content_raw[i] = 1.0 + (spirit[5 + i] - 0.5) * 0.1;
    }

    // Step 2: clamp to [FLOOR, CEIL].
    for v in body_raw.iter_mut() {
        *v = v.clamp(MULTIPLIER_FLOOR, MULTIPLIER_CEIL);
    }
    for v in mind_raw.iter_mut() {
        *v = v.clamp(MULTIPLIER_FLOOR, MULTIPLIER_CEIL);
    }
    for v in content_raw.iter_mut() {
        *v = v.clamp(MULTIPLIER_FLOOR, MULTIPLIER_CEIL);
    }

    // Step 3: spirit-strength scaling on content (per filter_down.py:737-740
    // + Preamble G9 — gentle filter prevents over-steering inner loop).
    apply_spirit_strength(&mut content_raw, SPIRIT_FILTER_STRENGTH_MULT);

    // Step 4: EMA smoothing across publishes (mirrors V5 publish path
    // smoothing per filter_down.py:742-743).
    let body_smoothed = ema_body.update(&body_raw).to_vec();
    let mind_smoothed = ema_mind.update(&mind_raw).to_vec();
    let content_smoothed = ema_spirit_content.update(&content_raw).to_vec();

    (body_smoothed, mind_smoothed, content_smoothed)
}

/// Read the 45D Python-computed spirit tensor from
/// `sensor_cache_inner_spirit.bin`. Returns `Some(45D)` only when the
/// Python sidecar has written at least one full payload; returns `None`
/// when the slot is missing, the read fails, or the payload is short
/// (cold-boot / partial-write window). The caller falls back to the
/// local MVP `compose_spirit_tensor` in the `None` case.
///
/// SPEC §G1 (Inner-Spirit = 45D) + §23.6 formulas + §9.A line 1055.
/// Pre-fix this function returned `[f32; 5]` — that was a SPEC
/// non-compliance bug fixed 2026-05-08 alongside C-S5 closure.
fn read_spirit_cache(sensor_cache: &Option<Slot>) -> Option<[f32; SPIRIT_DIMS]> {
    let slot = sensor_cache.as_ref()?;
    let raw = slot.read().ok()?;
    if raw.is_empty() {
        return None;
    }
    // Step 9 §4.6 schema migration v1→v2: msgpack source-dict
    // {"tensor": [v0..v44]} (Phase C). Falls back to legacy 45-float32-LE
    // for backward compat with pre-deploy.
    let is_msgpack = matches!(raw[0], 0x80..=0x8f | 0xde | 0xdf);
    if is_msgpack {
        return decode_spirit_source_dict(&raw);
    }
    if raw.len() < SPIRIT_DIMS * 4 {
        return None; // partial payload — sidecar still starting up
    }
    let mut out = [0.0_f32; SPIRIT_DIMS];
    for i in 0..SPIRIT_DIMS {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&raw[i * 4..i * 4 + 4]);
        out[i] = f32::from_le_bytes(buf);
    }
    Some(out)
}

/// Decode msgpack source dict {"tensor": [45 floats]} produced by
/// `spirit_worker._provide_spirit_45d` (Step 9 §4.6 schema v2).
fn decode_spirit_source_dict(payload: &[u8]) -> Option<[f32; SPIRIT_DIMS]> {
    use rmpv::Value;
    let v = rmpv::decode::read_value(&mut std::io::Cursor::new(payload)).ok()?;
    let map = match v {
        Value::Map(items) => items,
        _ => return None,
    };
    for (k, val) in map.iter() {
        if let Value::String(s) = k {
            if s.as_str() == Some("tensor") {
                if let Value::Array(items) = val {
                    if items.len() < SPIRIT_DIMS {
                        return None;
                    }
                    let mut out = [0.5_f32; SPIRIT_DIMS];
                    for (i, item) in items.iter().take(SPIRIT_DIMS).enumerate() {
                        out[i] = item.as_f64().unwrap_or(0.5) as f32;
                    }
                    return Some(out);
                }
            }
        }
    }
    None
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

fn encode_spirit_state_payload(spirit: &[f32; SPIRIT_DIMS]) -> Vec<u8> {
    use rmpv::Value;
    let values = Value::Array(spirit.iter().map(|f| Value::F64(*f as f64)).collect());
    let map = Value::Map(vec![
        (Value::String("src".into()), Value::String("inner".into())),
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

fn encode_local_filter_down_payload(
    body_mults: &[f32],
    mind_mults: &[f32],
    spirit_content_mults: &[f32],
) -> Vec<u8> {
    use rmpv::Value;
    fn arr_from_vec(v: &[f32]) -> Value {
        Value::Array(v.iter().map(|f| Value::F64(*f as f64)).collect())
    }
    let multipliers = Value::Map(vec![
        (Value::String("inner_body".into()), arr_from_vec(body_mults)),
        (Value::String("inner_mind".into()), arr_from_vec(mind_mults)),
        (
            Value::String("inner_spirit_content".into()),
            arr_from_vec(spirit_content_mults),
        ),
    ]);
    let map = Value::Map(vec![
        (Value::String("multipliers".into()), multipliers),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ]);
    let mut out = Vec::with_capacity(1024);
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
    use titan_core::constants::INNER_SPIRIT_45D_SCHEMA_VERSION;
    use titan_trinity_daemon::decode_local_filter_down_payload;

    #[test]
    fn schumann_role_spirit_hz_is_locked() {
        // G13 LOCKED: body × 9 = 70.47 Hz (full Schumann spectrum).
        // Sourced from titan-schumann (canonical Schumann library per
        // master plan §7 + C-S3 PLAN §1.1 #2).
        assert_eq!(SchumannRole::Spirit.hz(), 70.47);
    }

    #[test]
    fn compose_spirit_tensor_returns_45d() {
        let body = [0.5; 5];
        let mind = [0.5; 15];
        let cache = [0.5; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        assert_eq!(spirit.len(), 45);
    }

    #[test]
    fn compose_spirit_clamps_to_0_1_range() {
        // Out-of-range inputs → clamped output.
        let body = [2.0_f32; 5]; // > 1.0
        let mind = [-0.5_f32; 15]; // < 0.0
        let cache = [0.0; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        for (i, v) in spirit.iter().enumerate() {
            assert!(*v >= 0.0 && *v <= 1.0, "spirit[{i}]={v} out of range");
        }
    }

    #[test]
    fn compose_spirit_observer_slice_is_body() {
        // SAT[0:5] = body itself (observer slice per G8)
        let body = [0.1, 0.2, 0.3, 0.4, 0.5];
        let mind = [0.0; 15];
        let cache = [0.0; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        for i in 0..5 {
            assert!(
                (spirit[i] - body[i]).abs() < 1e-6,
                "observer dim {i} should equal body[{i}]={}, got {}",
                body[i],
                spirit[i]
            );
        }
    }

    #[test]
    fn observer_dims_never_in_filter_down_output() {
        // G8: filter_down output's inner_spirit_content has 40D, NOT 45D.
        // observer dims [0:5] are absent.
        let body = [0.5; 5];
        let mind = [0.5; 15];
        let cache = [0.5; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        let mut ema_b = EmaSmoother::new(5);
        let mut ema_m = EmaSmoother::new(15);
        let mut ema_c = EmaSmoother::new(40);
        let (b, m, c) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        assert_eq!(b.len(), 5);
        assert_eq!(m.len(), 15);
        assert_eq!(
            c.len(),
            40,
            "inner_spirit_content must be 40D (observer masked per G8)"
        );
    }

    #[test]
    fn local_filter_down_multipliers_within_floor_ceil() {
        // Output multipliers must respect SPEC G7 [FLOOR=0.3, CEIL=3.0].
        let body = [1.0; 5]; // extreme inputs
        let mind = [1.0; 15];
        let cache = [1.0; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        let mut ema_b = EmaSmoother::new(5);
        let mut ema_m = EmaSmoother::new(15);
        let mut ema_c = EmaSmoother::new(40);
        let (b, m, c) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        for v in b.iter().chain(m.iter()).chain(c.iter()) {
            assert!(*v >= MULTIPLIER_FLOOR - 1e-6 && *v <= MULTIPLIER_CEIL + 1e-6);
        }
    }

    #[test]
    fn spirit_content_multipliers_pulled_toward_one() {
        // SPIRIT_FILTER_STRENGTH_MULT=0.3 means content multipliers gently
        // approach 1.0. With body+mind=1.0 (max), spirit_raw deviates from
        // 0.5 by 0.5, raw mult = 1.05; after spirit_strength=0.3:
        // (1.05-1)*0.3+1 = 1.015. EMA first-call replaces directly.
        let body = [1.0; 5];
        let mind = [1.0; 15];
        let cache = [0.0; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        let mut ema_b = EmaSmoother::new(5);
        let mut ema_m = EmaSmoother::new(15);
        let mut ema_c = EmaSmoother::new(40);
        let (_b, _m, c) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        // All content mults should be close to 1.0 (within ~0.05)
        for v in c.iter() {
            assert!((v - 1.0).abs() < 0.05, "content mult {v} too far from 1.0");
        }
    }

    #[test]
    fn local_filter_down_payload_decodes_correctly() {
        let body = [0.5; 5];
        let mind = [0.5; 15];
        let cache = [0.5; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        let mut ema_b = EmaSmoother::new(5);
        let mut ema_m = EmaSmoother::new(15);
        let mut ema_c = EmaSmoother::new(40);
        let (b, m, c) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        let bytes = encode_local_filter_down_payload(&b, &m, &c);
        let decoded = decode_local_filter_down_payload(&bytes).unwrap();
        for i in 0..5 {
            assert!((decoded.body[i] - b[i]).abs() < 1e-5);
        }
        for i in 0..15 {
            assert!((decoded.mind[i] - m[i]).abs() < 1e-5);
        }
        for i in 0..40 {
            assert!((decoded.spirit_content[i] - c[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn spirit_state_payload_msgpack_shape() {
        let spirit = [0.5_f32; 45];
        let bytes = encode_spirit_state_payload(&spirit);
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
        for required in &["src", "type", "values", "ts"] {
            assert!(
                keys.contains(&required.to_string()),
                "missing key: {required}"
            );
        }
    }

    #[test]
    fn spirit_state_payload_src_inner_type_spirit_state() {
        let spirit = [0.0; 45];
        let bytes = encode_spirit_state_payload(&spirit);
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
                        assert_eq!(val, Value::String("SPIRIT_STATE".into()));
                        found_type = true;
                    }
                }
            }
        }
        assert!(found_src && found_type);
    }

    #[test]
    fn spirit_state_payload_values_45d() {
        let spirit: [f32; 45] = std::array::from_fn(|i| (i as f32) * 0.02);
        let bytes = encode_spirit_state_payload(&spirit);
        use rmpv::Value;
        let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(&bytes)).unwrap();
        if let Value::Map(items) = v {
            for (k, val) in items {
                if let Value::String(s) = k {
                    if s.as_str() == Some("values") {
                        if let Value::Array(arr) = val {
                            assert_eq!(arr.len(), 45);
                            return;
                        }
                    }
                }
            }
        }
        panic!("values array of length 45 not found");
    }

    #[test]
    fn slot_write_45d_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inner_spirit_45d.bin");
        let mut slot = Slot::create(&path, INNER_SPIRIT_45D_SCHEMA_VERSION as u32, 180).unwrap();
        let spirit: [f32; 45] = std::array::from_fn(|i| (i as f32) * 0.02);
        let bytes = encode_floats::<45>(&spirit);
        slot.write(&bytes).unwrap();
        let read = slot.read().unwrap();
        assert_eq!(read, bytes);
    }

    #[test]
    fn read_spirit_cache_handles_none() {
        // Updated 2026-05-08 alongside C-S5 closure: read_spirit_cache
        // now returns `Option<[f32; 45]>` per SPEC §G1 (was `[f32; 5]`).
        // None signals "fall back to compose_spirit_tensor MVP".
        let r = read_spirit_cache(&None);
        assert!(r.is_none());
    }

    #[test]
    fn read_spirit_cache_round_trips_45d_payload() {
        // Round-trip: write 45 × f32 LE to a slot, read back, verify
        // byte-identical recovery. Pins SPEC §G1 + §23.6 for
        // sensor_cache_inner_spirit.bin payload shape.
        use std::fs;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sensor_cache_inner_spirit.bin");
        // Variable-size slot up to 4096B (matches Python sidecar +
        // spec.rs declaration).
        let mut slot = Slot::create(&path, 1, 4096).expect("create slot");

        // Build 45 distinct values [0.01, 0.02, ..., 0.45]
        let mut payload: [f32; 45] = [0.0; 45];
        for i in 0..45 {
            payload[i] = (i as f32 + 1.0) * 0.01;
        }
        let mut bytes = Vec::with_capacity(45 * 4);
        for v in payload.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        slot.write(&bytes).expect("write 45D");

        // Reader picks up via Slot::open (separate handle, mirrors
        // titan-inner-spirit-rs runtime path).
        let reopened = Some(Slot::open(&path).expect("reopen slot"));
        let recovered = read_spirit_cache(&reopened).expect("Some(45D)");
        for i in 0..45 {
            assert!(
                (recovered[i] - payload[i]).abs() < 1e-6,
                "dim {i}: expected {}, got {}",
                payload[i],
                recovered[i]
            );
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_spirit_cache_returns_none_on_short_payload() {
        // Partial payload (e.g. cold-boot, sidecar still starting) →
        // None so the daemon falls back to compose_spirit_tensor MVP.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("partial.bin");
        let mut slot = Slot::create(&path, 1, 4096).expect("create slot");
        // Write only 5 floats (the OLD pre-fix layout).
        let mut bytes = Vec::with_capacity(5 * 4);
        for v in &[0.1_f32, 0.2, 0.3, 0.4, 0.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        slot.write(&bytes).expect("write 5×f32");

        let reopened = Some(Slot::open(&path).expect("reopen"));
        let recovered = read_spirit_cache(&reopened);
        assert!(
            recovered.is_none(),
            "short 5×f32 payload must return None (forces MVP fallback)"
        );
    }

    #[test]
    fn ema_smoothing_steady_state_converges() {
        let body = [0.5; 5];
        let mind = [0.5; 15];
        let cache = [0.5; 5];
        let spirit = compose_spirit_tensor(&body, &mind, &cache);
        let mut ema_b = EmaSmoother::new(5);
        let mut ema_m = EmaSmoother::new(15);
        let mut ema_c = EmaSmoother::new(40);
        // First publish — EMA replaces directly.
        let (b1, _, _) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        // Successive publishes converge.
        for _ in 0..50 {
            compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        }
        let (b2, _, _) = compute_local_filter_down(&spirit, &mut ema_b, &mut ema_m, &mut ema_c);
        for i in 0..5 {
            assert!((b2[i] - b1[i]).abs() < 1e-3, "should converge");
        }
    }
}
