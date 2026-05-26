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

// std::sync::atomic removed with the KERNEL_EPOCH_TICK arm path (P0-0a §4 —
// no shim; spirit pulse rising-edge gates the small filter_down DOWN-leg now,
// detected SHM-direct from sphere_clocks.bin).

use titan_bus::{BusClient, InboundEvent};
use titan_core::constants::{INNER_SPIRIT_FIRING_MAX_BYTES, INNER_SPIRIT_FIRING_SCHEMA_VERSION};
use titan_core::small_filter_down::{SmallFilterDownEngine, HALF_DIM};
use titan_schumann::{SchumannGenerator, SchumannRole};
use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_focus_into_enrichment, decode_filter_down_payload,
    decode_gift_at_spirit, encode_floats, load_checkpoint_for_part, load_restoring_cfg, observe,
    open_focus_input_if_present, open_neuromod_slot_if_present, read_dim_slice, read_focus_nudge,
    read_neuromod_gain, stateful_update, write_checkpoint_for_part, CheckpointSnapshot,
    ContentGate, DriftAggregator, FiringSlotWriter, FocusPart, Layer, PulseClockRole, PulseWatcher,
    RestoringCfg, TrinitySide, BODY_FLAG_INNER, INNER_SPIRIT_TOPICS, MIND_FLAG_INNER,
};

/// §G5.2 item 4 checkpoint cadence. Inner-spirit ticks @ 70.47 Hz so 720
/// ticks ≈ 10 s between checkpoint snapshots.
const CHECKPOINT_WRITE_EVERY_N_TICKS: u64 = 720;
const CHECKPOINT_PART: &str = "inner_spirit";

/// §G5.1 UP-leg (PLAN §4): per-content-dim additive bonus added to spirit's
/// enrichment_force when a body/mind sphere-clock pulses. Conservative starting
/// value — the field is small ([0,1] tensor space) so 0.02 is a noticeable
/// nudge without dominating drift/spring dynamics. Tunable via
/// titan_params.toml later when the §3 VERIFY gate observes outer coherence.
const UP_LEG_BONUS_AMPLITUDE: f32 = 0.02;

const SPIRIT_DIMS: usize = 45;
const DRIFT_THRESHOLD_PCT: f64 = 0.005;
/// Content slice (everything except observer) — used for filter_down output.
/// Per G8 + G9: observer dims `[0:5]` are NEVER published as multipliers
/// (mask happens by construction here — `inner_spirit_content` field carries
/// only `[5:45]` = 40D).
const CONTENT_RANGE: std::ops::Range<usize> = 5..45;
const CONTENT_DIMS: usize = 40;

pub async fn run(bus_socket: &Path, authkey: &[u8], shm_dir: &Path, data_dir: &Path) -> Result<()> {
    let client = BusClient::connect(bus_socket, authkey, "inner-spirit")
        .await
        .with_context(|| format!("bus connect to {}", bus_socket.display()))?;
    client
        .subscribe(INNER_SPIRIT_TOPICS)
        .await
        .context("bus subscribe")?;
    info!(event = "BUS_SUBSCRIBED", topics = ?INNER_SPIRIT_TOPICS);

    // Small filter_down learned engine (SPEC §G5.1 / D-SPEC-112 P0-0a — fires
    // on the spirit sphere-clock PULSE EDGE read from `sphere_clocks.bin`,
    // NOT KERNEL_EPOCH_TICK, NOT per Schumann tick). Loads its own per-half
    // trained brain (filter_down_local_inner_*.json) from data_dir.
    let engine = SmallFilterDownEngine::with_defaults(data_dir, "inner")
        .context("init inner small filter_down engine")?;

    let inner_spirit_slot = open_slot(shm_dir, "inner_spirit_45d.bin")?;
    let body_slot = open_slot(shm_dir, "inner_body_5d.bin")?;
    let mind_slot = open_slot(shm_dir, "inner_mind_15d.bin")?;
    // sensor_cache_inner_spirit.bin: created by Python sidecar
    // (inner_spirit_sensor_refresh in spirit_worker heartbeat-stub mode),
    // spawns LATER than this Rust daemon — Option<Slot> at boot, retry
    // open in tick loop. C-S5 closure 2026-05-08.
    let sensor_cache_path = shm_dir.join("sensor_cache_inner_spirit.bin");
    let sensor_cache = Slot::open(&sensor_cache_path).ok();
    info!(
        event = "SHM_OPENED",
        sensor_cache_present = sensor_cache.is_some()
    );

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
        engine,
        shm_dir.to_path_buf(),
        data_dir.to_path_buf(),
    )
    .await;

    shutdown.notify_waiters();
    let _ = dispatcher.await;
    bus.shutdown().await;

    tick_result
}

#[derive(Debug)]
struct DaemonState {
    /// UNIFIED inner_spirit_content multipliers (40D, observer already
    /// masked at publish side).
    unified_spirit_content: Option<[f32; CONTENT_DIMS]>,
    /// Shutdown via KERNEL_SHUTDOWN_ANNOUNCE.
    shutdown_requested: bool,
    /// Last successfully-computed 45D spirit tensor (Sprint 7 — used as
    /// `current_5d` fallback when source dict omits it; matches Python's
    /// `_last_spirit_45d` rolling-state pattern).
    last_spirit: [f32; SPIRIT_DIMS],
    /// Throttle counter for SENSOR_CACHE_ABSENT WARNs. Reset to 0 on every
    /// successful cache read; incremented each absent tick. The daemon ticks
    /// at Schumann spirit (~70 Hz) so an unthrottled WARN floods rsyslog
    /// (~40GB observed fleet-wide). WARN only on the first 3 absences + once
    /// per ~60s (4200 ticks) while persistently absent.
    sensor_cache_absent_count: u64,
    /// P0.5 / D-SPEC-131 §G5.1: amplitude of a BODY_BALANCE_GIFT received from
    /// the inner-body daemon since the last spirit tick. One-shot per
    /// D-SPEC-121 pattern (`.take()` consumes it; if multiple gifts arrive
    /// between two spirit ticks the latest amplitude wins — gifts can only
    /// land at body 7.83 Hz so this is at most one per 9 spirit ticks).
    pending_body_gift_amplitude: Option<f32>,
    /// P0.5 / D-SPEC-131 §G5.1: amplitude of a MIND_BALANCE_GIFT received
    /// from inner-mind. Same one-shot semantic.
    pending_mind_gift_amplitude: Option<f32>,
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            unified_spirit_content: None,
            shutdown_requested: false,
            last_spirit: [0.5; SPIRIT_DIMS],
            sensor_cache_absent_count: 0,
            pending_body_gift_amplitude: None,
            pending_mind_gift_amplitude: None,
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
            _ = shutdown.notified() => { debug!(event = "DISPATCHER_SHUTDOWN_NOTIFIED"); break; }
            event = bus.recv() => match event {
                Some(InboundEvent::Message { msg_type, raw_bytes, .. }) => {
                    handle_bus_message(&msg_type, &raw_bytes, &state);
                    // §G5.1 / P0-0a (D-SPEC-112): the small filter_down DOWN-leg
                    // no longer gates on KERNEL_EPOCH_TICK — the spirit sphere
                    // clock PULSE rising-edge (SHM-direct via PulseWatcher) is
                    // the canonical trigger now. The KERNEL_EPOCH_TICK subscription
                    // is intentionally retained (other consumers / health checks)
                    // but its receipt is no longer routed through this daemon.
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
                        s.unified_spirit_content = Some(p.inner_spirit_content);
                    }
                }
                Err(e) => warn!(err = ?e, "decode UNIFIED_SPIRIT_FILTER_DOWN failed"),
            }
        }
        "BODY_BALANCE_GIFT" | "MIND_BALANCE_GIFT" => {
            // P0.5 / D-SPEC-131 §G5.1: accept gift only if sovereign-half
            // matches this daemon (Inner). Outer gifts are silently
            // discarded — PLAN §6.5.1 sovereign-half lock.
            let payload = match titan_bus::client::extract_payload(raw_bytes) {
                Some(p) => p,
                None => return,
            };
            match decode_gift_at_spirit(&payload) {
                Ok(gift) => {
                    if gift.side != TrinitySide::Inner {
                        return;
                    }
                    if let Ok(mut s) = state.lock() {
                        if msg_type == "BODY_BALANCE_GIFT" {
                            s.pending_body_gift_amplitude = Some(gift.gift_amplitude);
                        } else {
                            s.pending_mind_gift_amplitude = Some(gift.gift_amplitude);
                        }
                    }
                }
                Err(e) => warn!(err = ?e, msg_type, "decode balance gift failed"),
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
    mut inner_spirit_slot: Slot,
    body_slot: Slot,
    mind_slot: Slot,
    mut sensor_cache: Option<Slot>,
    sensor_cache_path: std::path::PathBuf,
    mut firing_writer: FiringSlotWriter,
    mut engine: SmallFilterDownEngine,
    shm_dir: std::path::PathBuf,
    data_dir: std::path::PathBuf,
) -> Result<()> {
    let mut content_gate = ContentGate::new();
    // Previous epoch's 65D half-state — the `s` in the TD(0) transition s→s'.
    let mut prev_half: Option<[f64; HALF_DIM]> = None;
    let mut drift_agg = DriftAggregator::new("spirit", DRIFT_THRESHOLD_PCT);
    // §G5.2 item 4 — restore exact tensor + observable state from checkpoint
    // on boot; cold-start at 0.5 only when sidecar absent/invalid.
    let (mut prev, mut prev2, mut last_obs_restored) =
        match load_checkpoint_for_part::<SPIRIT_DIMS>(&data_dir, CHECKPOINT_PART) {
            Some(CheckpointSnapshot {
                prev,
                prev2,
                last_obs,
                ..
            }) => (prev, prev2, Some(last_obs)),
            None => ([0.5_f32; SPIRIT_DIMS], [0.5_f32; SPIRIT_DIMS], None),
        };
    // §G5.2 item 5 — gains from titan_params.toml [trinity_restoring] sidecar.
    let mut cfg = load_restoring_cfg(&shm_dir, Layer::Spirit);
    // §G5.2 item 2 — live neuromod-gain read per tick.
    let mut neuromod_slot = open_neuromod_slot_if_present(&shm_dir);
    let neuromod_path = shm_dir.join("neuromod_state.bin");
    // §G5.2 item 2 + §G12 — FOCUS cascade nudge slot.
    let mut focus_input_slot = open_focus_input_if_present(&shm_dir);
    let focus_input_path = shm_dir.join("focus_input.bin");
    // §G5.1 Phase 0a (PLAN §4): SHM-direct pulse-edge detector on
    // `sphere_clocks.bin`. DOWN-leg small filter_down fires on inner_spirit
    // rising-edge (replaces the old KERNEL_EPOCH_TICK arm — no shim). UP-leg
    // adds an additive snapshot bonus to spirit's enrichment on body/mind
    // rising-edge. Retry-open at the same cadence as the other sidecars.
    let mut pulse_watcher = PulseWatcher::open(&shm_dir);

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
                        cfg = load_restoring_cfg(&shm_dir, Layer::Spirit);
                    }
                    // Per-tick SHM-direct pulse-edge read (PLAN §4 / G5.1).
                    // D-SPEC-121: also read balanced-edges for the small
                    // filter_down DOWN-leg gate.
                    let (pulse_edges, balanced_pulse_edges) =
                        pulse_watcher.tick_with_balanced();
                    let drift_pct = tick_event.jitter_ns() as f64 / tick_event.period_ns as f64;
                    drift_agg.observe(drift_pct, tick_event.jitter_ns(), tick_event.epoch);
                    if let Err(e) = run_one_tick(
                        &bus, &state, &mut content_gate,
                        &mut engine, &mut prev_half,
                        &mut inner_spirit_slot, &body_slot, &mind_slot, &sensor_cache,
                        &mut firing_writer, &mut prev, &mut prev2,
                        &mut cfg, neuromod_slot.as_ref(), focus_input_slot.as_ref(),
                        &pulse_edges, &balanced_pulse_edges, &mut last_obs_restored,
                    ).await {
                        warn!(err = ?e, "tick failed (continuing)");
                    }
                    tick_count = tick_count.wrapping_add(1);
                    // §G5.2 item 4 — periodic checkpoint write.
                    if tick_count.is_multiple_of(CHECKPOINT_WRITE_EVERY_N_TICKS) {
                        if let Some(o) = last_obs_restored.as_ref() {
                            if let Err(e) = write_checkpoint_for_part::<SPIRIT_DIMS>(
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
            write_checkpoint_for_part::<SPIRIT_DIMS>(&data_dir, CHECKPOINT_PART, &prev, &prev2, o)
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
    engine: &mut SmallFilterDownEngine,
    prev_half: &mut Option<[f64; HALF_DIM]>,
    inner_spirit_slot: &mut Slot,
    body_slot: &Slot,
    mind_slot: &Slot,
    sensor_cache: &Option<Slot>,
    firing_writer: &mut FiringSlotWriter,
    prev: &mut [f32; SPIRIT_DIMS],
    prev2: &mut [f32; SPIRIT_DIMS],
    cfg: &mut RestoringCfg,
    neuromod_slot: Option<&Slot>,
    focus_input_slot: Option<&Slot>,
    // pulse_edges kept in signature for symmetry + future diagnostics; the
    // §G5.1 D-SPEC-121 small filter_down gate uses balanced_pulse_edges only.
    _pulse_edges: &[bool; 6],
    balanced_pulse_edges: &[bool; 6],
    last_obs_restored: &mut Option<titan_trinity_daemon::LayerObs>,
) -> Result<()> {
    // 1. Observer Principle G8: read sibling body + mind slots.
    let body = read_dim_slice::<5>(body_slot).map_err(|e| anyhow!("read body: {e}"))?;
    let mind = read_dim_slice::<15>(mind_slot).map_err(|e| anyhow!("read mind: {e}"))?;

    // 2. Read sensor_cache_inner_spirit. Sprint 7 §4.6: Rust now owns
    //    the 45D formula compute (via `project_inner_spirit_45d`) — the
    //    Python sidecar publishes raw inputs only, no tensor compute.
    //    Backward-compat with the legacy `{tensor:[45D]}` Python-computed
    //    payload is preserved in `project_inner_spirit_45d`.
    //
    //    Sprint 8 (rFP §4.6 closure): the legacy `compose_spirit_tensor`
    //    synthetic fallback (body+mind MVP) is RETIRED — it produced
    //    SPEC-incorrect values that masked Python sidecar outages. When
    //    the sensor cache is absent (cold boot, sidecar starting, Python
    //    writer crashed) we now retain the last successful 45D output
    //    and WARN throttled. Daemon stays resilient via last-known; the
    //    failure is observable rather than silently degraded.
    let last_spirit_snapshot = {
        let s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        s.last_spirit
    };
    let mut spirit: [f32; SPIRIT_DIMS] =
        match read_spirit_cache(sensor_cache, &body, &mind, &last_spirit_snapshot) {
            Some(cache_45d) => {
                // Recovered (or never absent) — reset the throttle counter.
                if let Ok(mut s) = state.lock() {
                    s.sensor_cache_absent_count = 0;
                }
                cache_45d
            }
            None => {
                let count = {
                    let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
                    s.sensor_cache_absent_count = s.sensor_cache_absent_count.saturating_add(1);
                    s.sensor_cache_absent_count
                };
                // Throttle: first 3 absences + once per ~60s (4200 ticks at
                // ~70 Hz). Unthrottled, this WARN floods rsyslog.
                if count <= 3 || count.is_multiple_of(4200) {
                    warn!(
                        event = "SENSOR_CACHE_ABSENT",
                        count = count,
                        consequence = "retain_last_known_45d"
                    );
                }
                last_spirit_snapshot
            }
        };
    // Cache fresh spirit (used as `current_5d` fallback on the next tick
    // when the source dict omits it — matches Python `_last_spirit_45d`).
    if let Ok(mut s) = state.lock() {
        s.last_spirit = spirit;
    }

    // Snapshot raw[t] = un-enriched 45D producer output BEFORE filter_down.
    let raw: [f32; SPIRIT_DIMS] = spirit;

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

    // 4b. §G5.2 traveling-tensor update (Layer::Spirit — gradient .3/.7,
    //     qualitative-led). Applies to ALL 45D incl. observer dims [0:5] (they
    //     travel; observer masking is a filter_down-OUTPUT concern per §G8, not
    //     a tensor-state concern). The restoring spring covers spirit's 45D —
    //     the layer GROUND_UP (§G10) deliberately does not reach.
    //     enrichment = filter_down delta on content dims; 0 on observer dims
    //     (which §G8 forbids from filter_down). Spring is neuromod-modulated
    //     per §G5.2 item 2 (G5.2-neuromod-gain).
    let mut enrichment = [0.0_f32; SPIRIT_DIMS];
    for i in 0..SPIRIT_DIMS {
        enrichment[i] = spirit[i] - raw[i];
    }
    // §G12 FOCUS cascade: amplified nudge composes into enrichment_force.
    let focus = read_focus_nudge::<SPIRIT_DIMS>(focus_input_slot, FocusPart::InnerSpirit);
    compose_focus_into_enrichment(&mut enrichment, &focus);
    // §G5.1 P0-0a / P0.5 (D-SPEC-131): UP-leg balance-gift. The body/mind
    // daemons publish BODY_BALANCE_GIFT / MIND_BALANCE_GIFT on their own
    // sphere clock's balanced rising-edge (sub-1% of their tick rates), each
    // carrying a per-cycle journey digest (path-length / peak-excursion /
    // coherence-climb / direction-stability). The amplitudes are stored in
    // `pending_*_gift_amplitude` on receipt; this tick consumes them ONCE
    // (D-SPEC-121 one-shot pattern) + applies via the Q/L/D mask so each
    // spirit dim receives gift in proportion to its body- vs mind-affinity
    // (PLAN §6.5.4 LOCKED 2026-05-24). RETIRES the v1.54.0 uniform
    // polarity-scalar bonus — no shim per `feedback_no_shim_old_path_must_be_deleted`.
    // Observer dims [0:5] never get gift via mask construction (Q/L/D
    // classification leaves them at the body/mind-zero positions of
    // BODY_FLAG_INNER / MIND_FLAG_INNER, preserved per G8 / observer principle).
    let (body_gift_amp, mind_gift_amp) = {
        let mut s = state.lock().map_err(|e| anyhow!("state lock: {e}"))?;
        (
            s.pending_body_gift_amplitude.take(),
            s.pending_mind_gift_amplitude.take(),
        )
    };
    if let Some(amp) = body_gift_amp {
        for i in 0..SPIRIT_DIMS {
            enrichment[i] += UP_LEG_BONUS_AMPLITUDE * amp * BODY_FLAG_INNER[i];
        }
    }
    if let Some(amp) = mind_gift_amp {
        for i in 0..SPIRIT_DIMS {
            enrichment[i] += UP_LEG_BONUS_AMPLITUDE * amp * MIND_FLAG_INNER[i];
        }
    }
    // `balanced_pulse_edges` continues downstream for the spirit-clock
    // DOWN-leg (small filter_down) — see the INNER_SPIRIT_FILTER_DOWN
    // emit block in step 7.
    cfg.neuromod_gain = read_neuromod_gain(neuromod_slot);
    let obs = observe(&prev[..], &prev2[..]);
    *last_obs_restored = Some(obs);
    let x = stateful_update(&prev[..], &prev2[..], &raw[..], &enrichment[..], &obs, cfg);
    let mut spirit_state = [0.0_f32; SPIRIT_DIMS];
    spirit_state.copy_from_slice(&x[..SPIRIT_DIMS]);
    *prev2 = *prev;
    *prev = spirit_state;
    let spirit = spirit_state;

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
    bus.publish("SPIRIT_STATE", Some("all"), Some(payload))
        .await
        .map_err(|e| anyhow!("publish SPIRIT_STATE: {e}"))?;

    // 7. Small filter_down DOWN-leg — BALANCED-PULSE-gated on the inner-spirit
    //    sphere clock rising edge (SPEC §G5.1 D-SPEC-121, v1.54.0; narrows
    //    D-SPEC-112's "any spirit pulse"). Fires only when the spirit clock's
    //    pulse_count advances AND it was balanced at pulse time
    //    (consecutive_balanced ≥ 1) — what "spirit reaches the Middle Path"
    //    means at the SPEC level. An unbalanced spirit pulse does NOT fire the
    //    small filter_down. Targets inner_body + inner_mind ONLY (sovereign
    //    half — D-SPEC-121 lock; outer-spirit owns the outer half).
    //    Once per balanced spirit pulse: assemble the 65D half-state +
    //    record_transition + maybe_train + compute learned multipliers +
    //    publish INNER_SPIRIT_FILTER_DOWN. Receiving body+mind daemons apply
    //    the multipliers ONCE on the next tick (consume-and-clear; D-SPEC-121
    //    one-shot application replacing v1.36.2's held + per-tick).
    if balanced_pulse_edges[PulseClockRole::InnerSpirit.index()] {
        let mut half = [0.0_f64; HALF_DIM];
        for (i, &v) in body.iter().enumerate() {
            half[i] = v as f64;
        }
        for (i, &v) in mind.iter().enumerate() {
            half[5 + i] = v as f64;
        }
        for (i, &v) in spirit.iter().enumerate() {
            half[20 + i] = v as f64;
        }

        if let Some(prev) = prev_half.as_ref() {
            engine.record_transition(prev, &half);
            let mut rng = rand::thread_rng();
            engine.maybe_train(&mut rng);
        }
        *prev_half = Some(half);

        let mults = engine.compute_multipliers(&half);
        let body_mults: Vec<f32> = mults.body.iter().map(|&v| v as f32).collect();
        let mind_mults: Vec<f32> = mults.mind.iter().map(|&v| v as f32).collect();
        let content_mults: Vec<f32> = mults.spirit_content.iter().map(|&v| v as f32).collect();

        // Observer dims [0:5] NEVER appear in inner_spirit_content (G8: it's
        // already only [5:45] = 40D content).
        let local_payload =
            encode_local_filter_down_payload(&body_mults, &mind_mults, &content_mults);
        bus.publish("INNER_SPIRIT_FILTER_DOWN", Some("all"), Some(local_payload))
            .await
            .map_err(|e| anyhow!("publish INNER_SPIRIT_FILTER_DOWN: {e}"))?;
    }

    Ok(())
}

/// Read sensor_cache_inner_spirit.bin and compute the 45D inner-spirit
/// tensor per SPEC §23.6 `spirit_tensor.collect_spirit_45d`.
///
/// Sprint 7 §4.6 FULL Rust formula port: Python `spirit_worker._provide_
/// spirit_45d` publishes RAW inputs (consciousness / hormone_levels /
/// hormone_fires / unified_spirit_stats / sphere_clocks / memory_stats /
/// birth_state / topology / expression_stats / history / current_5d) and
/// this function computes the 45D Sat-Chit-Ananda tensor in Rust using
/// the freshly-read body + mind tensors as Observer Principle inputs.
///
/// Backward-compat with the pre-Sprint-7 `{tensor: [45 floats]}` legacy
/// payload (Python-computed 45D) is preserved for atomic-deploy safety.
///
/// Returns `Some(45D)` when the cache contains a usable payload (either
/// schema). Returns `None` only when the slot is absent, read errors,
/// or the payload is fundamentally malformed — caller falls back to
/// `compose_spirit_tensor` MVP (body+mind only) in the `None` case.
fn read_spirit_cache(
    sensor_cache: &Option<Slot>,
    body: &[f32; 5],
    mind: &[f32; 15],
    last_spirit: &[f32; SPIRIT_DIMS],
) -> Option<[f32; SPIRIT_DIMS]> {
    let slot = sensor_cache.as_ref()?;
    let raw = slot.read().ok()?;
    if raw.is_empty() {
        return None;
    }
    let is_msgpack = matches!(raw[0], 0x80..=0x8f | 0xde | 0xdf);
    if is_msgpack {
        return project_inner_spirit_45d(&raw, body, mind, last_spirit).ok();
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

/// V6 45D inner-spirit projection. Pure compute over the msgpack source
/// dict + observer-principle body/mind reads. Implements SPEC §23.6
/// `collect_spirit_45d` SAT[0:15] + CHIT[15:30] + ANANDA[30:45].
///
/// Source dict (NEW schema, Sprint 7) — all keys optional:
///   - `current_5d`: [5 × f32] from last spirit tick (for SAT[0,5,12]).
///   - `consciousness`: {epoch_id, density, curvature, dream_quality,
///     fatigue, trajectory_magnitude, ...}
///   - `hormone_levels`: {CURIOSITY, FOCUS, INSPIRATION, IMPULSE,
///     VIGILANCE, ...}
///   - `hormone_fires`: {INTUITION, REFLECTION, CREATIVITY, EMPATHY,
///     CURIOSITY, ...} (counts; sum used for SAT[1])
///   - `unified_spirit_stats`: {velocity, epoch_count}
///   - `sphere_clocks`: {clock_name: {pulse_count}, ...}
///   - `memory_stats`: {action_chains}
///   - `birth_state`: [3+ × f32]
///   - `topology`: {volume, curvature} (optional — SAT[7]/CHIT[12]
///     fall back to default when absent)
///   - `expression_stats`: {sovereignty_ratio, composites:{...}}
///     (CHIT[13] + ANANDA[8])
///   - `history`: {expression:{sovereignty_ratio}} (SAT[2] fallback)
///
/// Source dict (LEGACY schema): `tensor: [45 × f32]` — already-computed
/// 45D passthrough for atomic-deploy safety.
///
/// `body` + `mind` are read by the caller from `inner_body_5d.bin` +
/// `inner_mind_15d.bin` SHM slots (Observer Principle G8); used for
/// body_coh + mind_coh + combined_coh aggregates that feed many dims.
///
/// `last_spirit` provides the fallback for `current_5d` when the source
/// dict omits it (cold-boot tick before Python sidecar has accumulated
/// any tensor state). Same role as Python's `_last_spirit_45d[:5]`.
pub fn project_inner_spirit_45d(
    payload: &[u8],
    body: &[f32; 5],
    mind: &[f32; 15],
    last_spirit: &[f32; SPIRIT_DIMS],
) -> Result<[f32; SPIRIT_DIMS]> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| anyhow!("decode source dict: {e}"))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Ok(*last_spirit),
    };

    // Legacy schema: 45D already computed — pass through.
    if let Some(tensor) = lookup_array(map, "tensor") {
        let mut out = [0.5_f32; SPIRIT_DIMS];
        for (i, item) in tensor.iter().take(SPIRIT_DIMS).enumerate() {
            out[i] = item.as_f64().unwrap_or(0.5) as f32;
        }
        return Ok(out);
    }

    // NEW schema: extract raw inputs.
    let current_5d = lookup_array(map, "current_5d")
        .map(|arr| {
            let mut t = [0.5_f32; 5];
            for (i, item) in arr.iter().take(5).enumerate() {
                t[i] = item.as_f64().unwrap_or(0.5) as f32;
            }
            t
        })
        .unwrap_or_else(|| {
            let mut t = [0.5_f32; 5];
            t.copy_from_slice(&last_spirit[..5]);
            t
        });

    let consciousness = lookup_map(map, "consciousness");
    let hormone_levels = lookup_map(map, "hormone_levels");
    let memory_stats = lookup_map(map, "memory_stats");
    let birth_state = lookup_array(map, "birth_state");

    // Observer Principle aggregates.
    let body_coh = mean_f32(body);
    let mind_coh = mean_f32(mind);
    let combined_coh = (body_coh + mind_coh) / 2.0;

    let mut spirit = [0.5_f32; SPIRIT_DIMS];

    // D-SPEC-101 (rFP Dims Redesign Closure Phase 1): short-window
    // self-observation breath signals (0..1, already normalized by the
    // inner_spirit sidecar's dual-EMA tracker — fast ~90s ÷ slow ~30min
    // baseline). Feed the re-grounded dims that previously saturated on
    // cumulative `min(1, count/N)` or pinned on `epoch/N`. Absent (boot) →
    // 0.0 (quiet); warms up over ~minutes.
    let win = lookup_map(map, "inner_spirit_window");
    // D-SPEC-101 Phase-1 completion (2026-05-21): rich expression rolling-
    // window breath (image/sound/speak/word variety+volume + windowed
    // sovereignty-of-expression) — feeds sovereignty[2], causal_understanding[28],
    // expression_quality[38]. Absent (boot) → neutral defaults at each use.
    let expr_win = lookup_map(map, "expression_window");

    // ── SAT[0..14] — Being/Existence (offset 0) ──
    // [0] self_recognition: cosine_sim(current_5d[:3], birth[:3])
    if let Some(b) = birth_state.as_ref() {
        if b.len() >= 3 {
            let a3 = [
                current_5d[0] as f64,
                current_5d[1] as f64,
                current_5d[2] as f64,
            ];
            let b3 = [
                b[0].as_f64().unwrap_or(0.5),
                b[1].as_f64().unwrap_or(0.5),
                b[2].as_f64().unwrap_or(0.5),
            ];
            spirit[0] = safe_clamp(cosine_sim(&a3, &b3)) as f32;
        }
    }
    // [1] authenticity RE-GROUNDED (D-SPEC-101): how the Titan's
    //   authenticity-cluster dims (self_recognition/sovereignty/essence_purity/
    //   uniqueness/integrity) MOVE over the ~90s window — not a saturating
    //   lifetime fire-count. Source: inner_spirit_window.authenticity_change.
    spirit[1] = safe_clamp(field_or_default(win.as_ref(), "authenticity_change", 0.0)) as f32;
    // [2] sovereignty RE-GROUNDED (D-SPEC-101 Phase-1 completion): windowed
    //   sovereignty-of-expression — self-authored (learned postures) vs total
    //   expressive actions over the recent window. Was the static
    //   history.expression.sovereignty_ratio (empty → pinned 0.5 / 0.0).
    //   Source: expression_window.sovereignty.
    spirit[2] = safe_clamp(field_or_default(expr_win.as_ref(), "sovereignty", 0.5)) as f32;
    // [3] boundary_clarity: (body_coh + mind_coh) / 2
    spirit[3] = safe_clamp((body_coh + mind_coh) / 2.0) as f32;
    // [4] temporal_continuity RE-GROUNDED (D-SPEC-101): steady self ⇒ high
    //   continuity; fast change of the 45D ⇒ low. Was min(1, epoch/3000)
    //   (pinned 1.0). Source: 1 − inner_spirit_window.self_churn.
    spirit[4] = safe_clamp(1.0 - field_or_default(win.as_ref(), "self_churn", 0.0)) as f32;
    // [5] origin_connection: 1 - l2_dist(current_5d, birth) / 3
    if let Some(b) = birth_state.as_ref() {
        if !b.is_empty() {
            let n = current_5d.len().min(b.len());
            let a: Vec<f64> = current_5d.iter().take(n).map(|v| *v as f64).collect();
            let bvec: Vec<f64> = b
                .iter()
                .take(n)
                .map(|v| v.as_f64().unwrap_or(0.5))
                .collect();
            spirit[5] = safe_clamp(1.0 - l2_dist(&a, &bvec) / 3.0) as f32;
        }
    }
    // [6] growth_trajectory RE-GROUNDED (D-SPEC-101): dynamic growth of
    //   inner_spirit's OWN 44 dims over the short window (NOT unified_spirit
    //   velocity — wrong layer). Source: inner_spirit_window.growth.
    spirit[6] = safe_clamp(field_or_default(win.as_ref(), "growth", 0.0)) as f32;
    // [7] spatial_presence RE-GROUNDED (D-SPEC-101): inner topology 10D
    //   change over the ~90s window. Was topo.volume/5 (static). Source:
    //   inner_spirit_window.topo_change.
    spirit[7] = safe_clamp(field_or_default(win.as_ref(), "topo_change", 0.0)) as f32;
    // [8] personality_coherence: body_coh * mind_coh * 2
    spirit[8] = safe_clamp(body_coh * mind_coh * 2.0) as f32;
    // [9] essence_purity: cons.density (default 0.5)
    spirit[9] = safe_clamp(field_or_default(consciousness.as_ref(), "density", 0.5)) as f32;
    // [10] resilience: 1 - |curvature|/π
    let curvature = field_or_default(consciousness.as_ref(), "curvature", 0.0).abs();
    spirit[10] = safe_clamp(1.0 - curvature / std::f64::consts::PI) as f32;
    // [11] adaptability RE-GROUNDED (D-SPEC-101): recent windowed
    //   hormone-deviation RATE (how much hormones are MOVING) — breathing,
    //   not the saturating cumulative |h−0.5| sum. Source:
    //   inner_spirit_window.hormone_velocity.
    spirit[11] = safe_clamp(field_or_default(win.as_ref(), "hormone_velocity", 0.0)) as f32;
    // [12] uniqueness: l2_dist(current_5d, [0.5;5]) / 2
    {
        let a: Vec<f64> = current_5d.iter().map(|v| *v as f64).collect();
        let d: Vec<f64> = vec![0.5; current_5d.len()];
        spirit[12] = safe_clamp(l2_dist(&a, &d) / 2.0) as f32;
    }
    // [13] integrity: (body_coh + mind_coh) / 2 (same shape as [3])
    spirit[13] = safe_clamp((body_coh + mind_coh) / 2.0) as f32;
    // [14] vitality: hormone_activity * 0.4 + body_health * 0.6
    let hormone_activity = mean_map_values(hormone_levels.as_ref());
    let body_health = if body.is_empty() { 0.5 } else { mean_f32(body) };
    spirit[14] = safe_clamp(hormone_activity * 0.4 + body_health * 0.6) as f32;

    // ── CHIT[15..29] — Consciousness/Awareness (offset 15) ──
    // [15] self_awareness_depth RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   DYNAMIC depth = self-observed coherence across the OTHER 44 inner_spirit
    //   dims (1 − var/0.25, smoothed). Was min(1, epoch/5000) (pinned 1.0 — a
    //   lifetime counter, zero variance). Source: inner_spirit_window.coherence_depth.
    spirit[15] = safe_clamp(field_or_default(win.as_ref(), "coherence_depth", 0.5)) as f32;
    // [16] observation_clarity: combined_coh
    spirit[16] = safe_clamp(combined_coh) as f32;
    // [17] discernment_quality RE-GROUNDED (rFP_trinity_dim_resonance, greenlit
    //   2026-05-20): the Titan's judgment apparatus — output_verifier
    //   verified/rejected volume + sovereignty score — plus the original
    //   action_chains as the meta-reasoning chains crystallize. Replaces the
    //   bare `min(1, action_chains/20)` that sat at 0.5 fleet-wide because
    //   action_chains=0 (the known meta-reasoning crystallization gap).
    {
        let ov = lookup_map(map, "output_verifier_stats");
        let verified = field_or_default(ov.as_ref(), "verified_count", 0.0);
        let rejected = field_or_default(ov.as_ref(), "rejected_count", 0.0);
        let sovereignty = field_or_default(ov.as_ref(), "sovereignty_score", 0.5);
        let action_chains = field_or_default(memory_stats.as_ref(), "action_chains", 0.0);
        let judgment_volume = ((verified + rejected) / 50.0).min(1.0);
        spirit[17] = safe_clamp(
            0.4 * judgment_volume + 0.3 * sovereignty + 0.3 * (action_chains / 20.0).min(1.0),
        ) as f32;
    }
    // [18] integration_level: combined_coh
    spirit[18] = safe_clamp(combined_coh) as f32;
    // [19] witness_presence: body_coh * mind_coh * 2
    spirit[19] = safe_clamp(body_coh * mind_coh * 2.0) as f32;
    // [20] pattern_recognition RE-GROUNDED (D-SPEC-101): recent INTUITION
    //   fire-RATE (windowed breath), not the saturating cumulative count.
    //   Source: inner_spirit_window.fire_rate_INTUITION.
    spirit[20] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_INTUITION", 0.0)) as f32;
    // [21] wisdom_accumulation: cons.density (default 0.0 here per Python)
    spirit[21] = safe_clamp(field_or_default(consciousness.as_ref(), "density", 0.0)) as f32;
    // [22] truth_seeking RE-GROUNDED (D-SPEC-101 Phase-1 completion): recent
    //   CURIOSITY fire-RATE (active truth-seeking right now), windowed breath.
    //   Was the hormone LEVEL hlvl.CURIOSITY (saturated ~1.0). Source:
    //   inner_spirit_window.fire_rate_CURIOSITY.
    spirit[22] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_CURIOSITY", 0.0)) as f32;
    // [23] attention_depth: hlvl.FOCUS
    spirit[23] = safe_clamp(field_or_default(hormone_levels.as_ref(), "FOCUS", 0.0)) as f32;
    // [24] reflective_capacity RE-GROUNDED (D-SPEC-101): recent REFLECTION
    //   fire-RATE (windowed breath), not the saturating cumulative count.
    //   Source: inner_spirit_window.fire_rate_REFLECTION.
    spirit[24] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_REFLECTION", 0.0)) as f32;
    // [25] dream_awareness: dream_quality * 0.7 + fatigue * 0.3
    let dream_quality = field_or_default(consciousness.as_ref(), "dream_quality", 0.0);
    let fatigue = field_or_default(consciousness.as_ref(), "fatigue", 0.0);
    spirit[25] = safe_clamp(dream_quality * 0.7 + fatigue * 0.3) as f32;
    // [26] temporal_awareness RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   recent sphere-clock pulse RATE — temporal awareness BREATHES with how
    //   actively the clocks pulse. Was min(1, Σpulse_count/50): the sidecar
    //   reader was also mismapped (summed RADII), and once the canonical
    //   pulse_count is read it saturates 1.0 on any mature Titan. The windowed
    //   rate breathes per the rFP cumulative→recent-rate principle. Source:
    //   inner_spirit_window.clock_pulse_rate.
    spirit[26] = safe_clamp(field_or_default(win.as_ref(), "clock_pulse_rate", 0.0)) as f32;
    // [27] spatial_awareness RE-GROUNDED (D-SPEC-101): inner topology 10D
    //   change over the ~90s window (shares the topo breath with [7]
    //   spatial_presence). Was the static (volume/5 + |curvature|)/2.
    //   Source: inner_spirit_window.topo_change.
    spirit[27] = safe_clamp(field_or_default(win.as_ref(), "topo_change", 0.0)) as f32;
    // [28] causal_understanding RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   breadth of recent expressive output (variety of active modalities).
    //   Was expression_intensity (empty → 0). Source: expression_window.variety.
    spirit[28] = safe_clamp(field_or_default(expr_win.as_ref(), "variety", 0.0)) as f32;
    // [29] meta_cognition: cons.trajectory OR cons.trajectory_magnitude
    let traj = field_or_default(consciousness.as_ref(), "trajectory", -1.0);
    let traj = if traj < 0.0 {
        field_or_default(consciousness.as_ref(), "trajectory_magnitude", 0.0)
    } else {
        traj
    };
    spirit[29] = safe_clamp(traj) as f32;

    // ── ANANDA[30..44] — Bliss/Fulfillment (offset 30) ──
    // [30] purpose_alignment: combined_coh * 0.8 + 0.2
    spirit[30] = safe_clamp(combined_coh * 0.8 + 0.2) as f32;
    // [31] meaning_depth: density * combined_coh * 2
    let density = field_or_default(consciousness.as_ref(), "density", 0.0);
    spirit[31] = safe_clamp(density * combined_coh * 2.0) as f32;
    // [32] creative_joy RE-GROUNDED (D-SPEC-101): recent CREATIVITY
    //   fire-RATE (windowed breath), not the saturating cumulative count.
    //   Source: inner_spirit_window.fire_rate_CREATIVITY.
    spirit[32] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_CREATIVITY", 0.0)) as f32;
    // [33] harmony_seeking: combined_coh
    spirit[33] = safe_clamp(combined_coh) as f32;
    // [34] beauty_perception: body_coh * mind_coh * 2
    spirit[34] = safe_clamp(body_coh * mind_coh * 2.0) as f32;
    // [35] truth_resonance RE-GROUNDED (D-SPEC-101): recent INTUITION
    //   fire-RATE (windowed breath), shares the INTUITION breath with [20].
    //   Source: inner_spirit_window.fire_rate_INTUITION.
    spirit[35] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_INTUITION", 0.0)) as f32;
    // [36] connection_fulfillment RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   recent EMPATHY fire-RATE (windowed breath). Was min(1, EMPATHY/15)
    //   (sat 1.0). Source: inner_spirit_window.fire_rate_EMPATHY.
    spirit[36] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_EMPATHY", 0.0)) as f32;
    // [37] growth_satisfaction RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   own-44-dim short-window growth (same source as [6] growth_trajectory),
    //   NOT unified_spirit velocity (wrong layer, sat 1.0). Source:
    //   inner_spirit_window.growth.
    spirit[37] = safe_clamp(field_or_default(win.as_ref(), "growth", 0.0)) as f32;
    // [38] expression_quality RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   volume×variety blend of the rich expression-window (how MUCH and how
    //   VARIED the recent expressive output). Was expression_intensity*0.5+0.3
    //   (floor 0.30, empty). Source: expression_window.{volume, variety}.
    {
        let vol = field_or_default(expr_win.as_ref(), "volume", 0.0);
        let var = field_or_default(expr_win.as_ref(), "variety", 0.0);
        spirit[38] = safe_clamp(0.5 * (vol + var)) as f32;
    }
    // [39] exploration_joy RE-GROUNDED (D-SPEC-101 Phase-1 completion): recent
    //   CURIOSITY fire-RATE (windowed breath). Was min(1, CURIOSITY/15) (sat
    //   1.0). Source: inner_spirit_window.fire_rate_CURIOSITY.
    spirit[39] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_CURIOSITY", 0.0)) as f32;
    // [40] rest_fulfillment: 1 - fatigue (cons.fatigue default 0.5)
    let fatigue_for_rest = field_or_default(consciousness.as_ref(), "fatigue", 0.5);
    spirit[40] = safe_clamp(1.0 - fatigue_for_rest) as f32;
    // [41] creative_tension RE-GROUNDED (D-SPEC-101 Phase-1 completion): recent
    //   INSPIRATION fire-RATE (windowed breath). Was the hormone LEVEL
    //   hlvl.INSPIRATION (frozen). Source: inner_spirit_window.fire_rate_INSPIRATION.
    spirit[41] = safe_clamp(field_or_default(win.as_ref(), "fire_rate_INSPIRATION", 0.0)) as f32;
    // [42] surrender_capacity: 1 - (IMPULSE + VIGILANCE) / 2
    let impulse = field_or_default(hormone_levels.as_ref(), "IMPULSE", 0.5);
    let vigilance = field_or_default(hormone_levels.as_ref(), "VIGILANCE", 0.5);
    spirit[42] = safe_clamp(1.0 - (impulse + vigilance) / 2.0) as f32;
    // [43] gratitude_depth: fulfillment * combined_coh; fulfillment=(body_health+mind_health)/2
    let mind_health = if mind.is_empty() { 0.5 } else { mean_f32(mind) };
    let fulfillment = (body_health + mind_health) / 2.0;
    spirit[43] = safe_clamp(fulfillment * combined_coh) as f32;
    // [44] transcendence_glimpse RE-GROUNDED (D-SPEC-101 Phase-1 completion):
    //   inner↔outer spirit-pair BIG-PULSE proximity (both spirit sphere clocks
    //   contracted + balanced → BIG PULSE). Was min(1, great_pulse_epochs/5)
    //   (=0, GREAT-circular). Source: inner_spirit_window.spirit_pair_resonance.
    spirit[44] = safe_clamp(field_or_default(win.as_ref(), "spirit_pair_resonance", 0.0)) as f32;

    Ok(spirit)
}

// ── Helpers (inlined per outer-spirit-rs convention; D8 cleanup will
//    extract these to the shared trinity-daemon crate) ──

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

fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na.sqrt() < 1e-10 || nb.sqrt() < 1e-10 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
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

/// Mean of all numeric values in a map (used for hormone_activity).
fn mean_map_values(map: Option<&Vec<(rmpv::Value, rmpv::Value)>>) -> f64 {
    let map = match map {
        Some(m) => m,
        None => return 0.0,
    };
    let mut total = 0.0;
    let mut count = 0;
    for (_, v) in map.iter() {
        if let Some(f) = v.as_f64() {
            total += f;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    total / count as f64
}

fn open_slot(shm_dir: &Path, name: &str) -> Result<Slot> {
    let path = shm_dir.join(name);
    Slot::open(&path).with_context(|| format!("open slot {}", path.display()))
}

/// Build a SPEC §8.5 `SPIRIT_STATE` payload as `rmpv::Value::Map` per
/// SPEC §8.10 line 900 byte-identical guarantee.
fn encode_spirit_state_payload(spirit: &[f32; SPIRIT_DIMS]) -> rmpv::Value {
    use rmpv::Value;
    let values = Value::Array(spirit.iter().map(|f| Value::F64(*f as f64)).collect());
    Value::Map(vec![
        (Value::String("src".into()), Value::String("inner".into())),
        (
            Value::String("type".into()),
            Value::String("SPIRIT_STATE".into()),
        ),
        (Value::String("values".into()), values),
        (Value::String("ts".into()), Value::F64(now_secs())),
    ])
}

/// Build a SPEC §8.6 `INNER_SPIRIT_FILTER_DOWN` payload as `rmpv::Value::Map`.
fn encode_local_filter_down_payload(
    body_mults: &[f32],
    mind_mults: &[f32],
    spirit_content_mults: &[f32],
) -> rmpv::Value {
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
    Value::Map(vec![
        (Value::String("multipliers".into()), multipliers),
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
    fn local_filter_down_payload_decodes_correctly() {
        // Encoder round-trip with representative per-half multipliers
        // (body[5] + mind[15] + spirit_content[40]) as produced by
        // SmallFilterDownEngine::compute_multipliers.
        let b: Vec<f32> = vec![1.1, 0.9, 1.0, 1.2, 0.8];
        let m: Vec<f32> = (0..15).map(|i| 1.0 + (i as f32 - 7.0) * 0.02).collect();
        let c: Vec<f32> = (0..40).map(|i| 1.0 + (i as f32 - 20.0) * 0.005).collect();
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
        let v: Value = bytes; // §4.C-ter: encode_*_payload now returns Value directly
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
        // Sprint 7: extended signature with body/mind/last_spirit for
        // Rust-side formula compute.
        let r = read_spirit_cache(&None, &[0.5; 5], &[0.5; 15], &[0.5; 45]);
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
        let recovered =
            read_spirit_cache(&reopened, &[0.5; 5], &[0.5; 15], &[0.5; 45]).expect("Some(45D)");
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
        let recovered = read_spirit_cache(&reopened, &[0.5; 5], &[0.5; 15], &[0.5; 45]);
        assert!(
            recovered.is_none(),
            "short 5×f32 payload must return None (forces MVP fallback)"
        );
    }

    // ── Sprint 7 §4.6 parity tests: project_inner_spirit_45d ──

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

    fn f_pair(k: &str, v: f64) -> (rmpv::Value, rmpv::Value) {
        (rmpv::Value::String(k.into()), rmpv::Value::F64(v))
    }

    #[test]
    fn project_inner_spirit_45d_empty_dict_returns_python_defaults() {
        // Empty source dict: SPEC §23.6 defaults (_DEFAULT = 0.5 for
        // most dims, with some dims producing computed defaults).
        // Body = [0.5; 5], mind = [0.5; 15] → body_coh = mind_coh = 0.5,
        // combined_coh = 0.5.
        let payload = encode_msgpack_map(vec![]);
        let body = [0.5_f32; 5];
        let mind = [0.5_f32; 15];
        let out = project_inner_spirit_45d(&payload, &body, &mind, &[0.5; 45]).unwrap();

        // SAT[0] no birth_state → stays at default 0.5.
        assert_eq!(out[0], 0.5);
        // SAT[1] authenticity RE-GROUNDED (D-SPEC-101): no inner_spirit_window
        //   → authenticity_change absent → 0.0.
        assert_eq!(out[1], 0.0);
        // SAT[2] sovereignty: no history → 0.5.
        assert_eq!(out[2], 0.5);
        // SAT[3] boundary_clarity: (0.5+0.5)/2 = 0.5.
        assert!((out[3] - 0.5).abs() < 1e-5);
        // SAT[4] temporal_continuity RE-GROUNDED (D-SPEC-101): 1 − self_churn;
        //   no window → self_churn absent 0.0 → continuity 1.0 (steady self).
        assert_eq!(out[4], 1.0);
        // SAT[6] growth_trajectory RE-GROUNDED (D-SPEC-101): no window → 0.0.
        assert!((out[6] - 0.0).abs() < 1e-5);
        // SAT[7] spatial_presence RE-GROUNDED (D-SPEC-101): no window → 0.0.
        assert!((out[7] - 0.0).abs() < 1e-5);
        // SAT[8] personality_coherence: 0.5 * 0.5 * 2 = 0.5.
        assert!((out[8] - 0.5).abs() < 1e-5);
        // SAT[9] essence_purity: default density = 0.5.
        assert!((out[9] - 0.5).abs() < 1e-5);
        // CHIT[16] observation_clarity = combined_coh = 0.5.
        assert!((out[16] - 0.5).abs() < 1e-5);
        // CHIT[17] discernment_quality RE-GROUNDED: empty source → no OV
        //   verified/rejected, sovereignty default 0.5, action_chains=0 →
        //   0.4·0 + 0.3·0.5 + 0.3·0 = 0.15.
        assert!(
            (out[17] - 0.15).abs() < 1e-4,
            "discernment empty = 0.15, got {}",
            out[17]
        );
        // CHIT[19] witness_presence: 0.5*0.5*2 = 0.5.
        assert!((out[19] - 0.5).abs() < 1e-5);
        // ANANDA[30] purpose_alignment: 0.5*0.8 + 0.2 = 0.6.
        assert!((out[30] - 0.6).abs() < 1e-5);
        // ANANDA[33] harmony_seeking = 0.5.
        assert!((out[33] - 0.5).abs() < 1e-5);
        // ANANDA[37] growth_satisfaction RE-GROUNDED (D-SPEC-101): window.growth;
        //   no window → 0.0.
        assert!((out[37] - 0.0).abs() < 1e-5);
        // CHIT[15] self_awareness_depth RE-GROUNDED (D-SPEC-101): window.
        //   coherence_depth; no window → default 0.5.
        assert!((out[15] - 0.5).abs() < 1e-5);
        // ANANDA[38] expression_quality RE-GROUNDED (D-SPEC-101): expr_window
        //   0.5·(vol+var); no window → 0.0.
        assert!((out[38] - 0.0).abs() < 1e-5);
        // ANANDA[40] rest_fulfillment: 1 - default fatigue 0.5 = 0.5.
        assert!((out[40] - 0.5).abs() < 1e-5);
        // ANANDA[42] surrender_capacity: 1 - (0.5+0.5)/2 = 0.5.
        assert!((out[42] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn project_inner_spirit_45d_hormone_fires_no_longer_consumed() {
        // D-SPEC-101 Phase-1 completion: hormone_fires cumulative counts are
        // NO LONGER read — all fire dims read the windowed fire-RATES from the
        // inner_spirit_window (EMPATHY→[36], CURIOSITY→[22]/[39], INSPIRATION→
        // [41], INTUITION→[20]/[35], REFLECTION→[24], CREATIVITY→[32]). Feeding
        // raw fires alone leaves them at the absent-window default 0.0.
        let fires = rmpv::Value::Map(vec![f_pair("EMPATHY", 3.0), f_pair("CURIOSITY", 15.0)]);
        let payload = encode_msgpack_map(vec![("hormone_fires", fires)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        for i in [22usize, 36, 39, 41] {
            assert_eq!(out[i], 0.0, "dim[{i}] should ignore raw hormone_fires");
        }
    }

    #[test]
    fn project_inner_spirit_45d_window_drives_regrounded_dims() {
        // D-SPEC-101: the inner_spirit_window source dict (pre-normalized
        // 0..1 breath signals from the sidecar dual-EMA tracker) drives the
        // 10 re-grounded dims. Verify each reads its own key.
        let win = rmpv::Value::Map(vec![
            f_pair("authenticity_change", 0.42),
            f_pair("self_churn", 0.30),
            f_pair("growth", 0.55),
            f_pair("topo_change", 0.18),
            f_pair("hormone_velocity", 0.66),
            f_pair("fire_rate_INTUITION", 0.71),
            f_pair("fire_rate_REFLECTION", 0.24),
            f_pair("fire_rate_CREATIVITY", 0.38),
            // Phase-1 completion additions:
            f_pair("fire_rate_EMPATHY", 0.12),
            f_pair("fire_rate_CURIOSITY", 0.49),
            f_pair("fire_rate_INSPIRATION", 0.61),
            f_pair("coherence_depth", 0.83),
            f_pair("clock_pulse_rate", 0.27),
            f_pair("spirit_pair_resonance", 0.09),
        ]);
        let payload = encode_msgpack_map(vec![("inner_spirit_window", win)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert!((out[1] - 0.42).abs() < 1e-5, "authenticity={}", out[1]); // [1]
        assert!(
            (out[4] - 0.70).abs() < 1e-5,
            "temporal_continuity={}",
            out[4]
        ); // 1−0.30
        assert!((out[6] - 0.55).abs() < 1e-5); // growth_trajectory
        assert!((out[7] - 0.18).abs() < 1e-5); // spatial_presence
        assert!((out[11] - 0.66).abs() < 1e-5); // adaptability
        assert!((out[20] - 0.71).abs() < 1e-5); // pattern_recognition
        assert!((out[24] - 0.24).abs() < 1e-5); // reflective_capacity
        assert!((out[27] - 0.18).abs() < 1e-5); // spatial_awareness (shares topo)
        assert!((out[32] - 0.38).abs() < 1e-5); // creative_joy
        assert!((out[35] - 0.71).abs() < 1e-5); // truth_resonance (shares INTUITION)
                                                // Phase-1 completion re-groundings:
        assert!((out[15] - 0.83).abs() < 1e-5); // self_awareness_depth = coherence_depth
        assert!((out[22] - 0.49).abs() < 1e-5); // truth_seeking = CURIOSITY rate
        assert!((out[26] - 0.27).abs() < 1e-5); // temporal_awareness = clock_pulse_rate
        assert!((out[36] - 0.12).abs() < 1e-5); // connection_fulfillment = EMPATHY rate
        assert!((out[37] - 0.55).abs() < 1e-5); // growth_satisfaction = growth (shares [6])
        assert!((out[39] - 0.49).abs() < 1e-5); // exploration_joy = CURIOSITY rate
        assert!((out[41] - 0.61).abs() < 1e-5); // creative_tension = INSPIRATION rate
        assert!((out[44] - 0.09).abs() < 1e-5); // transcendence_glimpse = spirit_pair_resonance
    }

    #[test]
    fn project_inner_spirit_45d_consciousness_drives_temporal_density_dims() {
        let cons = rmpv::Value::Map(vec![
            f_pair("epoch_id", 1500.0),
            f_pair("density", 0.8),
            f_pair("curvature", std::f64::consts::FRAC_PI_2),
            f_pair("dream_quality", 0.6),
            f_pair("fatigue", 0.2),
            f_pair("trajectory_magnitude", 0.45),
        ]);
        let payload = encode_msgpack_map(vec![("consciousness", cons)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        // SAT[4] temporal_continuity is now window-driven (D-SPEC-101), not
        //   epoch — verified in project_inner_spirit_45d_window_drives_regrounded.
        // SAT[9] essence_purity = 0.8
        assert!((out[9] - 0.8).abs() < 1e-5);
        // SAT[10] resilience = 1 - 1.5708/π = 0.5
        assert!((out[10] - 0.5).abs() < 1e-3, "SAT[10]={}", out[10]);
        // CHIT[15] self_awareness_depth RE-GROUNDED (D-SPEC-101): now reads
        //   window.coherence_depth (not epoch_count); no window → default 0.5.
        assert!((out[15] - 0.5).abs() < 1e-5);
        // CHIT[21] wisdom_accumulation = 0.8
        assert!((out[21] - 0.8).abs() < 1e-5);
        // CHIT[25] dream_awareness = 0.6*0.7 + 0.2*0.3 = 0.42 + 0.06 = 0.48
        assert!((out[25] - 0.48).abs() < 1e-4);
        // CHIT[29] meta_cognition = trajectory_magnitude = 0.45
        assert!((out[29] - 0.45).abs() < 1e-5);
        // ANANDA[31] meaning_depth = 0.8 * 0.5 * 2 = 0.8 (combined_coh = 0.5)
        assert!((out[31] - 0.8).abs() < 1e-5);
        // ANANDA[40] rest_fulfillment = 1 - 0.2 = 0.8
        assert!((out[40] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn project_inner_spirit_45d_hormone_levels_drive_truth_seeking_and_more() {
        let hlvl = rmpv::Value::Map(vec![
            f_pair("CURIOSITY", 0.7),
            f_pair("FOCUS", 0.4),
            f_pair("INSPIRATION", 0.6),
            f_pair("IMPULSE", 0.3),
            f_pair("VIGILANCE", 0.5),
        ]);
        let payload = encode_msgpack_map(vec![("hormone_levels", hlvl)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        // CHIT[23] attention_depth = hlvl.FOCUS = 0.4 (unchanged — hormone LEVEL).
        assert!((out[23] - 0.4).abs() < 1e-5);
        // ANANDA[42] surrender_capacity = 1 - (0.3+0.5)/2 = 0.6 (unchanged).
        assert!((out[42] - 0.6).abs() < 1e-5);
        // D-SPEC-101 Phase-1 completion: [22] truth_seeking + [41] creative_tension
        //   NO LONGER read hormone LEVELS — they read the windowed fire-RATES
        //   (CURIOSITY/INSPIRATION). With levels-only fed, both stay at 0.0.
        assert_eq!(out[22], 0.0);
        assert_eq!(out[41], 0.0);
    }

    #[test]
    fn project_inner_spirit_45d_birth_state_drives_self_recognition() {
        // Birth = [0.7, 0.8, 0.9]; current_5d = [0.7, 0.8, 0.9, 0.5, 0.5]
        // cosine_sim of [0.7, 0.8, 0.9] with itself = 1.0
        let birth = rmpv::Value::Array(vec![
            rmpv::Value::F64(0.7),
            rmpv::Value::F64(0.8),
            rmpv::Value::F64(0.9),
        ]);
        let current = rmpv::Value::Array(vec![
            rmpv::Value::F64(0.7),
            rmpv::Value::F64(0.8),
            rmpv::Value::F64(0.9),
            rmpv::Value::F64(0.5),
            rmpv::Value::F64(0.5),
        ]);
        let payload = encode_msgpack_map(vec![("birth_state", birth), ("current_5d", current)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        // SAT[0] = cosine_sim = 1.0
        assert!((out[0] - 1.0).abs() < 1e-5, "SAT[0]={}", out[0]);
        // SAT[5] = 1 - l2_dist([0.7,0.8,0.9], [0.7,0.8,0.9])/3 = 1.0
        assert!((out[5] - 1.0).abs() < 1e-5, "SAT[5]={}", out[5]);
    }

    #[test]
    fn project_inner_spirit_45d_observer_principle_body_mind_drive_coh_dims() {
        // body mean = (0.2+0.4+0.6+0.8+1.0)/5 = 0.6
        // mind mean = ((0.1+0.3+0.5+0.7+0.9)*3)/15 = 0.5
        // combined_coh = (0.6 + 0.5) / 2 = 0.55
        let body: [f32; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];
        let mut mind = [0.0_f32; 15];
        for i in 0..15 {
            mind[i] = match i % 5 {
                0 => 0.1,
                1 => 0.3,
                2 => 0.5,
                3 => 0.7,
                _ => 0.9,
            };
        }
        let payload = encode_msgpack_map(vec![]);
        let out = project_inner_spirit_45d(&payload, &body, &mind, &[0.5; 45]).unwrap();
        // SAT[3] = (0.6 + 0.5)/2 = 0.55
        assert!((out[3] - 0.55).abs() < 1e-5);
        // SAT[8] = 0.6 * 0.5 * 2 = 0.6
        assert!((out[8] - 0.6).abs() < 1e-5);
        // SAT[13] = 0.55 (same as [3])
        assert!((out[13] - 0.55).abs() < 1e-5);
        // SAT[14] vitality: hormone_activity=0 → 0*0.4 + 0.6*0.6 = 0.36
        assert!((out[14] - 0.36).abs() < 1e-4);
        // CHIT[16] observation_clarity = 0.55
        assert!((out[16] - 0.55).abs() < 1e-5);
        // CHIT[18] integration_level = 0.55
        assert!((out[18] - 0.55).abs() < 1e-5);
        // CHIT[19] witness_presence = 0.6 * 0.5 * 2 = 0.6
        assert!((out[19] - 0.6).abs() < 1e-5);
        // ANANDA[30] purpose_alignment = 0.55 * 0.8 + 0.2 = 0.64
        assert!((out[30] - 0.64).abs() < 1e-5);
        // ANANDA[33] harmony_seeking = 0.55
        assert!((out[33] - 0.55).abs() < 1e-5);
        // ANANDA[34] beauty_perception = 0.6 * 0.5 * 2 = 0.6
        assert!((out[34] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn project_inner_spirit_45d_sphere_clocks_no_longer_drive_temporal_awareness() {
        // D-SPEC-101 Phase-1 completion: [26] temporal_awareness now reads the
        // windowed clock-pulse RATE (inner_spirit_window.clock_pulse_rate),
        // computed by the sidecar from the CANONICAL sphere_clocks.bin layout —
        // NOT a raw `sphere_clocks` sub-map sum in the source dict (the old
        // reader was also mismapped). Feeding a raw sphere_clocks map no longer
        // affects [26]; it stays at the absent-window default 0.0.
        let clock_a = rmpv::Value::Map(vec![f_pair("pulse_count", 99.0)]);
        let clocks = rmpv::Value::Map(vec![(rmpv::Value::String("inner_spirit".into()), clock_a)]);
        let payload = encode_msgpack_map(vec![("sphere_clocks", clocks)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert_eq!(out[26], 0.0);
    }

    #[test]
    fn project_inner_spirit_45d_expression_window_drives_expression_dims() {
        // D-SPEC-101 Phase-1 completion: the rich expression rolling-window
        // (variety + volume + windowed sovereignty) drives the expressiveness
        // dims — [2] sovereignty, [28] causal_understanding (variety), [38]
        // expression_quality (0.5·(volume+variety)). The old expression_intensity
        // / history.sovereignty paths are deleted.
        let expr_win = rmpv::Value::Map(vec![
            f_pair("variety", 0.50),
            f_pair("volume", 0.30),
            f_pair("sovereignty", 0.77),
        ]);
        let payload = encode_msgpack_map(vec![("expression_window", expr_win)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert!((out[2] - 0.77).abs() < 1e-5, "sovereignty={}", out[2]);
        assert!(
            (out[28] - 0.50).abs() < 1e-5,
            "causal_understanding={}",
            out[28]
        );
        assert!(
            (out[38] - 0.40).abs() < 1e-5,
            "expression_quality={}",
            out[38]
        ); // 0.5·(0.30+0.50)
    }

    #[test]
    fn project_inner_spirit_45d_legacy_tensor_schema_passes_through() {
        let tensor: Vec<rmpv::Value> = (0..45).map(|i| rmpv::Value::F32(i as f32 * 0.02)).collect();
        let payload = encode_msgpack_map(vec![("tensor", rmpv::Value::Array(tensor))]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        for i in 0..45 {
            assert!((out[i] - (i as f32 * 0.02)).abs() < 1e-5);
        }
    }

    #[test]
    fn project_inner_spirit_45d_malformed_envelope_returns_last_spirit() {
        let payload = {
            let v = rmpv::Value::Array(vec![rmpv::Value::F64(1.0)]);
            let mut out = Vec::new();
            rmpv::encode::write_value(&mut out, &v).unwrap();
            out
        };
        let last = [0.42_f32; 45];
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &last).unwrap();
        for i in 0..45 {
            assert!((out[i] - 0.42).abs() < 1e-5);
        }
    }

    #[test]
    fn project_inner_spirit_45d_unified_spirit_stats_no_longer_consumed() {
        // D-SPEC-101 Phase-1 completion: unified_spirit_stats is no longer read
        // (wrong layer). [37] growth_satisfaction now reads window.growth and
        // [44] transcendence_glimpse reads window.spirit_pair_resonance — both
        // covered by window_drives_regrounded. Feeding unified_spirit_stats alone
        // leaves them at the absent-window default 0.0.
        let us = rmpv::Value::Map(vec![f_pair("velocity", 0.7), f_pair("epoch_count", 3.0)]);
        let payload = encode_msgpack_map(vec![("unified_spirit_stats", us)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert_eq!(out[37], 0.0);
        assert_eq!(out[44], 0.0);
    }

    #[test]
    fn project_inner_spirit_45d_memory_action_chains_drive_discernment() {
        // RE-GROUNDED (rFP_trinity_dim_resonance): 0.4·judgment_volume
        //   + 0.3·sovereignty + 0.3·min(1, action_chains/20).
        // action_chains=10 only (no OV): 0.4·0 + 0.3·0.5(default) + 0.3·0.5 = 0.30.
        let mem = rmpv::Value::Map(vec![f_pair("action_chains", 10.0)]);
        let payload = encode_msgpack_map(vec![("memory_stats", mem)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert!(
            (out[17] - 0.30).abs() < 1e-4,
            "discernment chains-only = 0.30, got {}",
            out[17]
        );

        // OV active: verified=30, rejected=20 → vol=min(1,50/50)=1.0; sovereignty=0.8;
        //   action_chains=0 → 0.4·1 + 0.3·0.8 + 0.3·0 = 0.64.
        let ov = rmpv::Value::Map(vec![
            f_pair("verified_count", 30.0),
            f_pair("rejected_count", 20.0),
            f_pair("sovereignty_score", 0.8),
        ]);
        let payload2 = encode_msgpack_map(vec![("output_verifier_stats", ov)]);
        let out2 = project_inner_spirit_45d(&payload2, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert!(
            (out2[17] - 0.64).abs() < 1e-4,
            "discernment OV-active = 0.64, got {}",
            out2[17]
        );
    }

    // (D-SPEC-101: spatial_presence[7] + spatial_awareness[27] are now
    //  driven by inner_spirit_window.topo_change, not the static topology
    //  volume/curvature — covered by project_inner_spirit_45d_window_drives_regrounded.)

    #[test]
    fn project_inner_spirit_45d_history_no_longer_drives_sovereignty() {
        // D-SPEC-101 Phase-1 completion: [2] sovereignty now reads the windowed
        // expression_window.sovereignty (self-authored ratio), not the static
        // history.expression.sovereignty_ratio. Feeding history alone leaves [2]
        // at the expr_window-absent default 0.5.
        let expr_history = rmpv::Value::Map(vec![f_pair("sovereignty_ratio", 0.85)]);
        let history = rmpv::Value::Map(vec![(
            rmpv::Value::String("expression".into()),
            expr_history,
        )]);
        let payload = encode_msgpack_map(vec![("history", history)]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn project_inner_spirit_45d_current_5d_fallback_to_last_spirit() {
        // No current_5d in source dict → use last_spirit[..5].
        // Birth differs from last_spirit[..5] so SAT[5] origin_connection != 1.
        let birth = rmpv::Value::Array(vec![
            rmpv::Value::F64(0.0),
            rmpv::Value::F64(0.0),
            rmpv::Value::F64(0.0),
        ]);
        let payload = encode_msgpack_map(vec![("birth_state", birth)]);
        let mut last = [0.5_f32; 45];
        last[0] = 1.0;
        last[1] = 1.0;
        last[2] = 1.0;
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &last).unwrap();
        // SAT[5] = 1 - l2_dist([1,1,1,0.5,0.5], [0,0,0]) / 3
        //        = 1 - sqrt(1+1+1)/3 = 1 - 1.732/3 = ~0.422
        assert!(out[5] > 0.4 && out[5] < 0.45, "SAT[5]={}", out[5]);
    }

    #[test]
    fn project_inner_spirit_45d_dimensionality_is_45() {
        let payload = encode_msgpack_map(vec![]);
        let out = project_inner_spirit_45d(&payload, &[0.5; 5], &[0.5; 15], &[0.5; 45]).unwrap();
        assert_eq!(out.len(), 45);
    }
}
