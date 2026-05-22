//! runtime — Full unified-spirit-rs binary integration (C4-5).
//!
//! Wires together every C-S4 component into one boot sequence:
//! 1. PR_SET_PDEATHSIG (SPEC §11.C(1)) — die when substrate dies.
//! 2. Open all shm slots (slot_handles.rs from C4-2).
//! 3. Connect to main bus via TITAN_AUTHKEY_HEX HMAC handshake
//!    (titan_bus::client from C4-2c).
//! 4. Subscribe to REQUIRED bus topics (boot.rs from C4-2b1).
//! 5. Initialize ResonanceDetector (resonance.rs C4-2b1) + UnifiedSpirit
//!    (unified_spirit.rs C4-2b2) + FilterDownV5Engine (filter_down.rs
//!    C4-3a/b/c) from data_dir.
//! 6. Spawn body_cycle_loop (orchestration.rs C4-2) — assembles 162D +
//!    SeqLock-writes self_162d / unified_spirit_132d.
//! 7. Spawn body_cycle_publisher task — per cycle: read self_162d slot,
//!    compute multipliers, publish UNIFIED_SPIRIT_SELF_ASSEMBLED +
//!    UNIFIED_SPIRIT_FILTER_DOWN.
//! 8. Spawn bus dispatch loop (boot::run_bus_dispatch_loop) wired with
//!    build_advance_callback → ResonanceDetector → UnifiedSpirit::advance.
//! 9. Spawn 6 daemon placeholders via DaemonSupervisor (C4-4).
//! 10. Wait for SIGTERM / SIGINT / KERNEL_SHUTDOWN_ANNOUNCE → graceful shutdown.
//!
//! Per SPEC §10.A unified-spirit boot order: connect bus + spawn 6
//! daemons within 200ms of substrate spawn. SELF tensor first-write
//! within 200ms triggers substrate's slot-freshness ready signal.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::boot::{build_advance_callback, run_bus_dispatch_loop, REQUIRED_SUBSCRIPTIONS};
use crate::cli::Cli;
use crate::exit::UnifiedSpiritExitCode;
use crate::filter_down::{encode_filter_down_payload, FilterDownV5Engine};
use crate::orchestration::spawn_body_cycle_loop;
use crate::resonance::ResonanceDetector;
use crate::self_assembly::{decode_f32_slice, SELF_DIMS as ASM_SELF_DIMS};
use crate::slot_handles::SlotHandles;
use crate::supervise::{DaemonSupervisor, DaemonSupervisorConfig};
use crate::unified_spirit::UnifiedSpirit;
use titan_bus::client::BusClient;

/// Number of dims in the 162D SELF tensor (re-export for compute_multipliers).
const SELF_162D: usize = 162;

/// Result of `Runtime::run` — drives the binary's exit code per SPEC §15.
pub enum RuntimeOutcome {
    /// Graceful shutdown via SIGTERM / SIGINT / KERNEL_SHUTDOWN_ANNOUNCE.
    Clean,
    /// Boot failure — config / bus connect / shm open.
    BootFailure(UnifiedSpiritExitCode),
}

/// Top-level runtime — owns all integrated components for the binary's
/// lifetime. Construction = boot sequence; `run()` = main wait loop.
pub struct Runtime {
    /// Bus client (Arc — shared between dispatch loop + publisher task).
    pub bus_client: Arc<BusClient>,
    /// Resonance detector (shared with dispatch loop's BIG PULSE callback).
    pub detector: Arc<Mutex<ResonanceDetector>>,
    /// Unified spirit (shared with BIG PULSE callback for advance()).
    pub spirit: Arc<Mutex<UnifiedSpirit>>,
    /// V5 engine (owned by publisher task).
    pub engine: Arc<Mutex<FilterDownV5Engine>>,
    /// 6-daemon supervisor.
    pub daemon_sup: Arc<DaemonSupervisor>,
    /// Shutdown flag — flipped on SIGTERM or KERNEL_SHUTDOWN_ANNOUNCE.
    pub shutdown_flag: Arc<AtomicBool>,
    /// Body-cycle loop join handle + shutdown sender.
    body_cycle_handle: Option<JoinHandle<()>>,
    body_cycle_shutdown_tx: Option<watch::Sender<()>>,
    /// Body-cycle publisher join handle (spawns a separate task that
    /// polls self_162d slot + publishes UNIFIED_SPIRIT_FILTER_DOWN +
    /// UNIFIED_SPIRIT_SELF_ASSEMBLED).
    publisher_handle: Option<JoinHandle<()>>,
    /// Bus dispatch task join handle.
    dispatch_handle: Option<JoinHandle<()>>,
    /// Daemon reaper task join handle — polls every 200ms for child
    /// exits + invokes handle_exit (respawn or escalate). Closes the
    /// production-supervision gap surfaced 2026-05-06 (post-D2): without
    /// this, dead daemons stayed `<defunct>` forever, never respawned.
    reaper_handle: Option<JoinHandle<()>>,
    /// Cached config for shutdown.
    cadence_ms: u64,
}

impl Runtime {
    /// Boot the runtime per SPEC §10.A. On any boot failure returns the
    /// canonical exit code; caller surfaces via `process::exit`.
    pub async fn boot(cli: &Cli) -> Result<Self, UnifiedSpiritExitCode> {
        // Step 1: PDEATHSIG handled in main.rs before runtime boot.

        // Step 2: open shm slots
        let shm_dir = cli.effective_shm_dir();
        let slots = match SlotHandles::open_all(&shm_dir) {
            Ok(s) => s,
            Err(e) => {
                error!(
                    event = "BOOT_FAIL_SHM",
                    err = ?e,
                    shm_dir = ?shm_dir,
                    "failed to open shm slots"
                );
                return Err(UnifiedSpiritExitCode::ShmOpenFailure);
            }
        };
        // Drop slots — body_cycle_loop will re-open. We just verified
        // they exist + have correct sizes.
        drop(slots);

        // Step 3: connect to main bus
        let authkey_hex = match std::env::var("TITAN_AUTHKEY_HEX") {
            Ok(v) => v,
            Err(_) => {
                error!(
                    event = "BOOT_FAIL_AUTHKEY",
                    "TITAN_AUTHKEY_HEX env var missing — substrate must set this"
                );
                return Err(UnifiedSpiritExitCode::BusConnectFailure);
            }
        };
        let authkey = match hex::decode(&authkey_hex) {
            Ok(b) => b,
            Err(e) => {
                error!(
                    event = "BOOT_FAIL_AUTHKEY_DECODE",
                    err = ?e,
                    "TITAN_AUTHKEY_HEX is not valid hex"
                );
                return Err(UnifiedSpiritExitCode::BusConnectFailure);
            }
        };
        let bus_socket = cli.effective_bus_socket();
        let client_name = format!("unified-spirit-{}", cli.titan_id.as_str().to_lowercase());
        let bus_client = match BusClient::connect(&bus_socket, &authkey, &client_name).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                error!(
                    event = "BOOT_FAIL_BUS",
                    err = ?e,
                    socket = ?bus_socket,
                    "bus connect failed"
                );
                return Err(UnifiedSpiritExitCode::BusConnectFailure);
            }
        };

        // Step 4: subscribe
        let topics: Vec<&str> = REQUIRED_SUBSCRIPTIONS.to_vec();
        if let Err(e) = bus_client.subscribe(&topics).await {
            error!(event = "BOOT_FAIL_SUBSCRIBE", err = ?e, "BUS_SUBSCRIBE failed");
            return Err(UnifiedSpiritExitCode::BusConnectFailure);
        }

        // Step 5: init Resonance + UnifiedSpirit + FilterDownV5Engine
        let data_dir = &cli.data_dir;
        let detector = Arc::new(Mutex::new(ResonanceDetector::with_defaults(data_dir)));
        let spirit = Arc::new(Mutex::new(match UnifiedSpirit::with_defaults(data_dir) {
            Ok(s) => s,
            Err(e) => {
                error!(event = "BOOT_FAIL_SPIRIT", err = ?e, "UnifiedSpirit init failed");
                return Err(UnifiedSpiritExitCode::Generic);
            }
        }));
        let engine = match FilterDownV5Engine::with_defaults(data_dir) {
            Ok(e) => Arc::new(Mutex::new(e)),
            Err(err) => {
                error!(event = "BOOT_FAIL_ENGINE", err = ?err, "FilterDownV5Engine init failed");
                return Err(UnifiedSpiritExitCode::Generic);
            }
        };

        // Step 6: spawn body_cycle_loop (slot-write side)
        let cadence_ms = cli.self_assembly_cadence_ms;
        let (body_cycle_handle, body_cycle_shutdown_tx) =
            match spawn_body_cycle_loop(shm_dir.clone(), cadence_ms) {
                Ok(pair) => (pair.0, pair.1),
                Err(e) => {
                    error!(event = "BOOT_FAIL_BODY_CYCLE", err = ?e);
                    return Err(UnifiedSpiritExitCode::ShmOpenFailure);
                }
            };

        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // §G5.1 / D-SPEC-96 D4: GREAT-pulse → unified filter_down channel.
        // The dispatch loop's advance callback sends the GREAT count here;
        // the publisher task computes + publishes UNIFIED_SPIRIT_FILTER_DOWN
        // once per GREAT pulse (event, not per-tick).
        let (great_pulse_tx, great_pulse_rx) = watch::channel(0u64);

        // Step 7: spawn body_cycle_publisher (also publishes Phase B SHM
        // metadata slots per rFP_phase_c_state_read_unification §B).
        let publisher_handle = spawn_publisher_task(
            shm_dir.clone(),
            cadence_ms,
            engine.clone(),
            detector.clone(),
            spirit.clone(),
            bus_client.clone(),
            shutdown_flag.clone(),
            great_pulse_rx,
        );

        // Step 8: spawn bus dispatch loop
        let on_big_pulse = build_advance_callback(spirit.clone(), detector.clone(), great_pulse_tx);
        let dispatch_handle = tokio::spawn({
            let client = bus_client.clone();
            let det = detector.clone();
            let flag = shutdown_flag.clone();
            async move {
                run_bus_dispatch_loop(client, det, on_big_pulse, flag).await;
            }
        });

        // Step 9: spawn 6 daemon placeholders (best-effort — missing
        // binaries log + continue, don't fail boot)
        let daemon_cfg = DaemonSupervisorConfig {
            daemon_binary_dir: cli.daemon_binary_dir.clone(),
            use_placeholder_daemons: cli.use_placeholder_daemons,
            placeholder_binary: None,
            child_env: collect_child_env(cli, &authkey_hex),
        };
        let daemon_sup = match DaemonSupervisor::new(daemon_cfg) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                error!(event = "BOOT_FAIL_SUPERVISOR", err = ?e);
                return Err(UnifiedSpiritExitCode::Generic);
            }
        };
        if let Err(e) = daemon_sup.spawn_all().await {
            warn!(
                event = "DAEMON_SPAWN_PARTIAL",
                err = ?e,
                "one or more daemons failed to spawn — continuing boot, supervisor will retry"
            );
        }

        // Step 9b (post-D2 fix 2026-05-06): spawn reaper task. Without
        // this, dead daemons stay `<defunct>` and never respawn.
        let reaper_handle = daemon_sup.clone().spawn_reaper();

        info!(
            event = "BOOT_COMPLETE",
            binary = "unified-spirit",
            titan_id = cli.titan_id.as_str(),
            cadence_ms = cadence_ms,
            daemon_count = daemon_sup.live_count(),
            "unified-spirit boot complete"
        );

        Ok(Self {
            bus_client,
            detector,
            spirit,
            engine,
            daemon_sup,
            shutdown_flag,
            body_cycle_handle: Some(body_cycle_handle),
            body_cycle_shutdown_tx: Some(body_cycle_shutdown_tx),
            publisher_handle: Some(publisher_handle),
            dispatch_handle: Some(dispatch_handle),
            reaper_handle: Some(reaper_handle),
            cadence_ms,
        })
    }

    /// Wait for SIGTERM / SIGINT / KERNEL_SHUTDOWN_ANNOUNCE → graceful
    /// shutdown. Returns the exit code per SPEC §15.
    pub async fn run(mut self) -> UnifiedSpiritExitCode {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    error!(err = ?e, "install SIGTERM handler failed");
                    return UnifiedSpiritExitCode::Generic;
                }
            };
        let mut intr =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    error!(err = ?e, "install SIGINT handler failed");
                    return UnifiedSpiritExitCode::Generic;
                }
            };
        let shutdown_flag = self.shutdown_flag.clone();
        let mut tick = tokio::time::interval(Duration::from_millis(100));

        let exit_code: UnifiedSpiritExitCode;
        loop {
            tokio::select! {
                _ = term.recv() => {
                    info!(event = "SHUTDOWN", reason = "SIGTERM", "graceful shutdown");
                    exit_code = UnifiedSpiritExitCode::Clean;
                    break;
                }
                _ = intr.recv() => {
                    info!(event = "SHUTDOWN", reason = "SIGINT", "graceful shutdown");
                    exit_code = UnifiedSpiritExitCode::Clean;
                    break;
                }
                _ = tick.tick() => {
                    // KERNEL_SHUTDOWN_ANNOUNCE may have flipped the flag
                    // via dispatch loop.
                    if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        info!(event = "SHUTDOWN", reason = "KERNEL_SHUTDOWN_ANNOUNCE", "graceful shutdown");
                        exit_code = UnifiedSpiritExitCode::Clean;
                        break;
                    }
                }
            }
        }

        // Cascade: stop body_cycle_loop, publisher, dispatch, daemons,
        // then drop bus_client (which closes connection).
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);

        if let Some(tx) = self.body_cycle_shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.body_cycle_handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), h).await;
        }
        if let Some(h) = self.publisher_handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), h).await;
        }
        if let Some(h) = self.dispatch_handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), h).await;
        }
        // Reaper is an infinite poll loop — abort rather than await.
        // Aborting BEFORE shutdown_all so the reaper doesn't try to
        // respawn the daemons we're about to SIGTERM.
        if let Some(h) = self.reaper_handle.take() {
            h.abort();
            let _ = tokio::time::timeout(Duration::from_millis(500), h).await;
        }

        // SIGTERM all 6 daemons
        self.daemon_sup.shutdown_all().await;

        // Persist final state
        if let Err(e) = self.spirit.lock().save_state() {
            warn!(event = "SHUTDOWN_PERSIST_FAIL", subsystem = "spirit", err = ?e);
        }
        if let Err(e) = self.detector.lock().save_state() {
            warn!(event = "SHUTDOWN_PERSIST_FAIL", subsystem = "detector", err = ?e);
        }
        if let Err(e) = self.engine.lock().persist() {
            warn!(event = "SHUTDOWN_PERSIST_FAIL", subsystem = "engine", err = ?e);
        }

        // Drop bus_client (closes socket cleanly)
        self.bus_client.shutdown().await;

        info!(
            event = "RUNTIME_STOPPED",
            cadence_ms = self.cadence_ms,
            exit_code = exit_code.as_str(),
            "runtime fully stopped"
        );
        exit_code
    }
}

/// Body-cycle publisher task — every `cadence_ms`, polls self_162d slot,
/// publishes UNIFIED_SPIRIT_SELF_ASSEMBLED (per-cycle signal), feeds the
/// V5 engine a transition + trains it (§G5.1 / D-SPEC-96 D3 closure), and
/// publishes UNIFIED_SPIRIT_FILTER_DOWN ONLY on a new GREAT pulse
/// (§G5.1 / D4 closure — unified filter_down is a GREAT-gated EVENT).
/// SHM metadata slots are published per-cycle for observability.
#[allow(clippy::too_many_arguments)]
fn spawn_publisher_task(
    shm_dir: PathBuf,
    cadence_ms: u64,
    engine: Arc<Mutex<FilterDownV5Engine>>,
    detector: Arc<Mutex<ResonanceDetector>>,
    spirit: Arc<Mutex<UnifiedSpirit>>,
    bus_client: Arc<BusClient>,
    shutdown_flag: Arc<AtomicBool>,
    great_pulse_rx: watch::Receiver<u64>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Open self_162d slot read-only handle (separate from body_cycle_loop's writer)
        let self_path = shm_dir.join("self_162d.bin");
        let slot = match titan_state::Slot::open(&self_path) {
            Ok(s) => s,
            Err(e) => {
                error!(event = "PUBLISHER_FAIL_OPEN", err = ?e, "cannot open self_162d.bin");
                return;
            }
        };

        // Phase B: open 3 Rust-owned metadata slots (rFP §B / D-SPEC-72).
        // Open-once + write-per-tick mirrors the self_162d pattern above.
        // Failure logged once; publisher continues without metadata slots
        // (B.7 cascade verifies kernel pre-created them).
        let mut metadata_pub =
            match crate::metadata_publisher::MetadataPublisher::open_all(&shm_dir) {
                Ok(p) => Some(p),
                Err(e) => {
                    warn!(
                        event = "METADATA_PUBLISHER_FAIL_OPEN",
                        err = ?e,
                        "Phase B metadata slots unavailable — proceeding with bus-only publish path"
                    );
                    None
                }
            };

        let mut tick = tokio::time::interval(Duration::from_millis(cadence_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut epoch_id: u64 = 0;
        // §G5.1 / D3: previous 162D SELF snapshot for the TD(0) transition.
        let mut prev_self: Option<[f64; ASM_SELF_DIMS]> = None;
        // §G5.1 / D4: GREAT count last published as a filter_down event.
        let mut last_published_great: u64 = 0;
        info!(event = "PUBLISHER_LOOP_START", cadence_ms = cadence_ms);

        loop {
            tick.tick().await;
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                info!(event = "PUBLISHER_LOOP_STOP");
                return;
            }

            // Read self_162d slot
            let payload = match slot.read() {
                Ok(p) => p,
                Err(_) => continue, // SeqLock retries exhausted; skip cycle
            };
            let self_162: [f32; SELF_162D] = match decode_f32_slice::<SELF_162D>(&payload) {
                Some(v) => v,
                None => continue,
            };
            // Convert f32 → f64 for engine
            let self_162_f64: [f64; ASM_SELF_DIMS] = {
                let mut out = [0.0_f64; ASM_SELF_DIMS];
                for (i, &v) in self_162.iter().enumerate() {
                    out[i] = v as f64;
                }
                out
            };

            epoch_id += 1;
            let ts = wall_seconds();

            // Publish UNIFIED_SPIRIT_SELF_ASSEMBLED (signal-only, P1)
            let assembled_payload = encode_assembled_payload(epoch_id, ts);
            if let Err(e) = bus_client
                .publish(
                    "UNIFIED_SPIRIT_SELF_ASSEMBLED",
                    Some("all"),
                    Some(assembled_payload),
                )
                .await
            {
                warn!(event = "PUBLISHER_PUBLISH_FAIL", msg = "UNIFIED_SPIRIT_SELF_ASSEMBLED", err = ?e);
            }

            // §G5.1 / D3: feed the V5 engine a transition s→s' + train.
            // Continuous learning builds `total_train_steps` toward the
            // cold-start floor (2000); `compute_multipliers` keeps emitting
            // all-1.0 until then, so this has no premature dim effect. The
            // 130D `felt_curr` is the first 130 dims of the 162D SELF. The
            // engine internally throttles training via `train_every_n` +
            // `min_transitions`. The lock + thread_rng are confined to this
            // synchronous block (dropped before the next .await) so the task
            // future stays Send.
            if let Some(prev) = prev_self {
                let mut felt_curr = [0.0_f64; 130];
                felt_curr.copy_from_slice(&self_162_f64[0..130]);
                let mut eng = engine.lock();
                eng.record_transition(&prev, &self_162_f64, &felt_curr);
                let mut rng = rand::thread_rng();
                eng.maybe_train(&mut rng);
            }
            prev_self = Some(self_162_f64);

            // §G5.1 / D4: publish UNIFIED_SPIRIT_FILTER_DOWN ONLY on a new
            // GREAT pulse. The advance callback bumps `great_pulse_rx` with
            // the GREAT count; we compute the multipliers from the freshest
            // SELF and publish once per pulse (event, not per Schumann tick).
            let great_count = *great_pulse_rx.borrow();
            if great_count > last_published_great {
                last_published_great = great_count;
                let multipliers = engine.lock().compute_multipliers(&self_162_f64);
                let fd_payload = encode_filter_down_payload(&multipliers, great_count, ts);
                if let Err(e) = bus_client
                    .publish("UNIFIED_SPIRIT_FILTER_DOWN", Some("all"), Some(fd_payload))
                    .await
                {
                    warn!(event = "PUBLISHER_PUBLISH_FAIL", msg = "UNIFIED_SPIRIT_FILTER_DOWN", err = ?e);
                } else {
                    info!(
                        event = "UNIFIED_FILTER_DOWN_PUBLISHED",
                        great_pulse_count = great_count,
                        "GREAT-gated unified filter_down published"
                    );
                }
            }

            // Phase B: publish 3 SHM metadata slots (resonance, unified_spirit,
            // filter_down). G18 SHM-canonical + G21 single-writer; consumers
            // read via Python ShmReaderBank instead of bus subscription /
            // Python-wrapper publishers (B.4 retirement). Per-tick so the
            // climbing `total_train_steps` + `last_loss` stay observable.
            if let Some(mp) = metadata_pub.as_mut() {
                mp.publish(&detector, &spirit, &engine, ts);
            }
        }
    })
}

/// Build UNIFIED_SPIRIT_SELF_ASSEMBLED payload `{epoch_id, ts}` as
/// `rmpv::Value::Map` per SPEC §8.6 + §8.10.
fn encode_assembled_payload(epoch_id: u64, ts: f64) -> rmpv::Value {
    rmpv::Value::Map(vec![
        (
            rmpv::Value::String("epoch_id".into()),
            rmpv::Value::Integer(rmpv::Integer::from(epoch_id)),
        ),
        (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
    ])
}

/// Build child env for daemon spawns — pass through identifying vars.
fn collect_child_env(cli: &Cli, authkey_hex: &str) -> std::collections::HashMap<String, String> {
    let mut env = std::collections::HashMap::new();
    env.insert(
        "TITAN_KERNEL_TITAN_ID".into(),
        cli.titan_id.as_str().to_string(),
    );
    env.insert("TITAN_AUTHKEY_HEX".into(), authkey_hex.to_string());
    env.insert(
        "TITAN_KERNEL_SHM_DIR".into(),
        cli.effective_shm_dir().to_string_lossy().into_owned(),
    );
    env.insert(
        "TITAN_KERNEL_BUS_SOCKET_PATH".into(),
        cli.effective_bus_socket().to_string_lossy().into_owned(),
    );
    env.insert(
        "TITAN_KERNEL_DATA_DIR".into(),
        cli.data_dir.to_string_lossy().into_owned(),
    );
    env
}

fn wall_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_specs::DAEMON_NAMES;

    #[test]
    fn encode_assembled_payload_round_trips() {
        // C4-5 test 1: assembled payload schema {epoch_id, ts}
        let v = encode_assembled_payload(42, 1234567890.5);
        if let rmpv::Value::Map(entries) = v {
            let mut got_epoch = None;
            let mut got_ts = None;
            for (k, val) in entries {
                if let rmpv::Value::String(s) = &k {
                    match s.as_str() {
                        Some("epoch_id") => got_epoch = val.as_u64(),
                        Some("ts") => got_ts = val.as_f64(),
                        _ => {}
                    }
                }
            }
            assert_eq!(got_epoch, Some(42));
            assert!((got_ts.unwrap() - 1234567890.5).abs() < 1e-3);
        } else {
            panic!("expected map");
        }
    }

    #[test]
    fn collect_child_env_includes_required_vars() {
        // C4-5 test 2: child env propagates auth + paths to daemons
        use clap::Parser;
        let cli = Cli::try_parse_from(["titan-unified-spirit-rs", "--titan-id", "T1"]).unwrap();
        let env = collect_child_env(&cli, "abcdef");
        assert_eq!(env.get("TITAN_KERNEL_TITAN_ID"), Some(&"T1".to_string()));
        assert_eq!(env.get("TITAN_AUTHKEY_HEX"), Some(&"abcdef".to_string()));
        assert!(env.contains_key("TITAN_KERNEL_SHM_DIR"));
        assert!(env.contains_key("TITAN_KERNEL_BUS_SOCKET_PATH"));
        assert!(env.contains_key("TITAN_KERNEL_DATA_DIR"));
    }

    #[test]
    fn wall_seconds_is_monotonic_and_finite() {
        // C4-5 test 3: wall_seconds returns finite + non-zero on realtime clock
        let t1 = wall_seconds();
        std::thread::sleep(Duration::from_millis(2));
        let t2 = wall_seconds();
        assert!(t1.is_finite() && t1 > 0.0);
        assert!(t2 >= t1);
    }

    #[test]
    fn daemon_names_consistent_across_modules() {
        // C4-5 test 4: child_specs DAEMON_NAMES set matches what runtime
        // uses (no drift between integration + spec definitions)
        assert_eq!(DAEMON_NAMES.len(), 6);
        for name in DAEMON_NAMES.iter() {
            assert!(name.starts_with("titan-"));
            assert!(name.ends_with("-rs"));
        }
    }
}
