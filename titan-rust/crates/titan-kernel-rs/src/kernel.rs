//! kernel — Top-level `Kernel` struct + `Kernel::run()` boot orchestrator.
//!
//! Per SPEC §10.A boot sequence (T+0ms → T+1000ms steady state). Wires
//! together every C-S2 crate:
//! - titan-core::Identity + titan-core::supervisor + titan-core::atomic_write
//! - titan-state::SlotRegistry
//! - titan-cgn::create_cgn_live_weights
//! - titan-bus::BusBroker
//! - titan-clocks::run_circadian_loop + run_pi_heartbeat_loop
//!
//! Plus C2-6 additions:
//! - identity_load (KernelExitCode mapping)
//! - broker_publisher (BrokerEpochPublisher for KERNEL_EPOCH_TICK)
//! - supervision_log (rotating JSONL writer)
//! - persistence (L0 snapshot writer)
//! - spawn (substrate placeholder + python_main)
//! - shutdown (graceful SIGTERM sequence)

use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;
use tracing::{info, warn};

use titan_bus::BusBroker;
use titan_cgn::create_cgn_live_weights;
use titan_clocks::{run_circadian_loop, run_pi_heartbeat_loop, EpochTickPublisher};
use titan_core::constants::{
    BUS_API_HTTP_PORT_DEFAULT, KERNEL_PYTHON_DRAIN_GRACE_S, KERNEL_SHUTDOWN_GRACE_S, SPEC_VERSION,
};
use titan_state::{Slot, SlotRegistry};

use crate::broker_publisher::BrokerEpochPublisher;
use crate::cli::Cli;
use crate::exit::KernelExitCode;
use crate::fastbus_publisher::spawn_kernel_fastbus_publisher;
use crate::identity_load::load_identity;
use crate::kernel_supervisor::KernelChildSupervisor;
use crate::persistence::{load_or_default, run_snapshot_loop, L0Snapshot, SnapshotState};
use crate::spawn::{SpawnConfig, SpawnedChildren};

/// Kernel boot + run errors.
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    /// Identity load failed → exit 3.
    #[error("identity load failed")]
    Identity,
    /// Slot registry creation failed → exit 5.
    #[error("shm slot registry creation failed: {0}")]
    SlotRegistry(String),
    /// CGN slot creation failed → exit 5.
    #[error("cgn_live_weights creation failed: {0}")]
    CgnSlot(String),
    /// Bus broker bind failed → exit 4.
    #[error("bus broker start failed: {0}")]
    BrokerStart(String),
    /// Fastbus attach failed → exit 5 (shm). C-S3 chunk C3-6 addition.
    #[error("fastbus attach failed: {0}")]
    FastbusAttach(String),
    /// Substrate placeholder spawn failed → exit 6.
    #[error("substrate spawn failed: {0}")]
    SubstrateSpawn(String),
    /// L0 persistence error.
    #[error("L0 persistence error: {0}")]
    Persistence(String),
    /// Kernel-supervisor wiring failed (Phase C C-S7 Gap B).
    #[error("kernel supervisor failed: {0}")]
    Supervisor(String),
}

impl KernelError {
    /// Map to canonical exit code per SPEC §15.
    pub fn to_exit_code(&self) -> KernelExitCode {
        match self {
            KernelError::Identity => KernelExitCode::IdentityLoadFailure,
            KernelError::SlotRegistry(_)
            | KernelError::CgnSlot(_)
            | KernelError::FastbusAttach(_) => KernelExitCode::ShmCreateFailure,
            KernelError::BrokerStart(_) => KernelExitCode::BusBindFailure,
            KernelError::SubstrateSpawn(_) => KernelExitCode::ChildLimitReached,
            KernelError::Persistence(_) => KernelExitCode::Generic,
            KernelError::Supervisor(_) => KernelExitCode::Generic,
        }
    }
}

/// Optional knobs for tests + ops.
pub struct KernelRunOptions {
    /// `false` → skip spawning the substrate placeholder (tests).
    pub spawn_substrate: bool,
    /// `false` → skip spawning `python -u scripts/guardian_hcl.py` (tests).
    pub spawn_guardian_hcl: bool,
    /// Phase 11 §11.I.1 — `false` → skip spawning
    /// `python -u scripts/titan_hcl.py` (tests). Production main.rs
    /// flips TRUE per Phase 11 peer-spawn architecture.
    pub spawn_titan_hcl: bool,
    /// Phase 11 §11.I.1 — `false` → skip spawning
    /// `python -u scripts/titan_hcl_api.py` (tests). Production main.rs
    /// flips TRUE so api becomes a kernel-rs peer to titan_hcl + guardian_hcl.
    pub spawn_titan_hcl_api: bool,
    /// Path to the substrate placeholder binary. None → use a default
    /// resolved relative to `target/debug/`.
    pub substrate_binary: Option<PathBuf>,
    /// Auto-shutdown after this duration (tests). `None` → run until SIGTERM.
    pub auto_shutdown_after: Option<Duration>,
}

impl Default for KernelRunOptions {
    fn default() -> Self {
        Self {
            spawn_substrate: true,
            // spawn_*_hcl* default FALSE so test fixtures don't have to
            // tear down Python children. PRODUCTION main.rs explicitly flips
            // these to TRUE per Phase 11 §11.I.1 peer-spawn architecture.
            spawn_guardian_hcl: false,
            spawn_titan_hcl: false,
            spawn_titan_hcl_api: false,
            substrate_binary: None,
            auto_shutdown_after: None,
        }
    }
}

/// SPEC §11.B.5 — read `[api].host` / `[api].port` from `config.toml` (the
/// `--config` path). Returns `(host, port)` as `Option`s — `None` for any key
/// that is absent, the file unreadable, or the TOML unparseable (the caller
/// then falls back to env / the canonical default). This keeps `config.toml`
/// the single source of truth for the api port the kernel binds.
fn read_api_bind_from_config(config_path: &std::path::Path) -> (Option<String>, Option<u16>) {
    let txt = match std::fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(e) => {
            warn!(path = %config_path.display(), err = %e,
                "B8.5 api socket: config unreadable — using env/default port");
            return (None, None);
        }
    };
    let val: toml::Value = match txt.parse() {
        Ok(v) => v,
        Err(e) => {
            warn!(path = %config_path.display(), err = %e,
                "B8.5 api socket: config TOML parse failed — using env/default port");
            return (None, None);
        }
    };
    let api = val.get("api");
    let host = api
        .and_then(|a| a.get("host"))
        .and_then(|h| h.as_str())
        .map(str::to_owned);
    let port = api
        .and_then(|a| a.get("port"))
        .and_then(toml::Value::as_integer)
        .and_then(|p| u16::try_from(p).ok());
    (host, port)
}

/// SPEC §11.B.5 — bind the kernel-owned api listening socket (socket
/// activation). The kernel binds ONCE and holds the socket for its whole
/// lifetime; every api child (boot + every zero-downtime reload) inherits this
/// exact fd (dup'd to `TITAN_API_LISTEN_FD` in its pre-exec hook) and serves
/// uvicorn on it — so OLD + NEW share ONE never-destroyed accept queue across a
/// swap ⇒ zero dropped connections (OLD closing its dup only decrements the
/// socket refcount; the kernel + NEW keep the queue alive).
///
/// Bind address resolution (each source wins over the next):
/// host — `TITAN_API_HOST` env > `config.toml [api].host` > `0.0.0.0`;
/// port — `TITAN_API_PORT` env (shadow-boot override) > `config.toml [api].port`
/// > [`BUS_API_HTTP_PORT_DEFAULT`]. Reading the configured port is REQUIRED for
/// multi-Titan-per-box hosts (e.g. T2 binds 7777, T3 binds 7778) — a hardcoded
/// default would collide. The socket keeps its default CLOEXEC so it does NOT
/// leak into guardian_hcl / titan_hcl (only the api child's pre-exec dup
/// re-exposes it). Returns `None` on any failure — the api then self-binds
/// (degraded: reloads are no longer zero-downtime, but the api still serves on
/// boot/crash-respawn).
fn bind_api_listen_socket(config_path: &std::path::Path) -> Option<socket2::Socket> {
    use socket2::{Domain, Protocol, SockAddr, Socket, Type};
    let (cfg_host, cfg_port) = read_api_bind_from_config(config_path);
    let host = std::env::var("TITAN_API_HOST")
        .ok()
        .or(cfg_host)
        .unwrap_or_else(|| "0.0.0.0".to_string());
    let port: u16 = std::env::var("TITAN_API_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .or(cfg_port)
        .unwrap_or(BUS_API_HTTP_PORT_DEFAULT as u16);
    let ip: std::net::IpAddr = host.parse().unwrap_or_else(|e| {
        warn!(host = %host, err = %e, "B8.5 api socket: invalid TITAN_API_HOST; using 0.0.0.0");
        std::net::IpAddr::from([0u8, 0, 0, 0])
    });
    let domain = if ip.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };
    let addr = std::net::SocketAddr::new(ip, port);
    let sock = match Socket::new(domain, Type::STREAM, Some(Protocol::TCP)) {
        Ok(s) => s,
        Err(e) => {
            warn!(err = %e, "B8.5 api socket: socket() failed — api will self-bind");
            return None;
        }
    };
    if let Err(e) = sock.set_reuse_address(true) {
        warn!(err = %e, "B8.5 api socket: SO_REUSEADDR failed (continuing)");
    }
    if let Err(e) = sock.bind(&SockAddr::from(addr)) {
        warn!(addr = %addr, err = %e,
            "B8.5 api socket: bind failed — api will self-bind (reloads not zero-downtime)");
        return None;
    }
    // SOMAXCONN-class backlog (matches uvicorn's default 2048-ish accept depth).
    if let Err(e) = sock.listen(1024) {
        warn!(addr = %addr, err = %e, "B8.5 api socket: listen failed — api will self-bind");
        return None;
    }
    info!(
        event = "BOOT_B8_5_API_LISTEN_BOUND",
        addr = %addr,
        fd = sock.as_raw_fd(),
        "B8.5 kernel api listening socket bound (socket activation, SPEC §11.B.5)"
    );
    Some(sock)
}

/// Run the kernel from boot to shutdown. Returns exit code.
///
/// Steps per SPEC §10.A:
/// - **B0-B2**: process start + identity load + HKDF authkey derive
/// - **B3**: shm dir + all kernel slots + CGN slot
/// - **B4**: bus broker bind
/// - **B5**: kernel RPC bind (deferred — Python-only socket in C-S2)
/// - **B6**: circadian + π-heartbeat clocks start
/// - **B7**: L0 snapshot load (resumes last clock state)
/// - **B8**: spawn substrate placeholder
/// - **B9**: spawn python_main (optional)
/// - **B10**: subscribe to SUPERVISION_* + open supervision.jsonl
/// - **B11**: L0 snapshot writer loop
/// - **B12**: steady state — wait for shutdown signal
pub async fn run(cli: &Cli, options: KernelRunOptions) -> Result<KernelExitCode, KernelError> {
    let titan_id = cli.titan_id.as_str();
    let shm_dir = cli.effective_shm_dir();
    let bus_socket = cli.effective_bus_socket();
    let data_dir = cli.data_dir.clone();
    let identity_path = data_dir.join("titan_identity_keypair.json");

    info!(event = "BOOT_B1_IDENTITY_LOAD", path = ?identity_path, "B1 identity load");
    // Phase C C-S7 Gap D (2026-05-05): pass titan_id from CLI as a hint so
    // Solana CLI byte-array-format keypairs (which lack embedded titan_id)
    // can be loaded — production Titans store keypairs in this format
    // today. Struct-format keypairs (canonical SPEC §10.A B1) ignore the
    // hint and use their embedded titan_id.
    let identity = load_identity(&identity_path, titan_id).map_err(|_| KernelError::Identity)?;

    info!(event = "BOOT_B2_AUTHKEY_DERIVE", "B2 HKDF authkey derive");
    // Per PLAN_microkernel_phase_c_s2_kernel.md §7.3: HKDF info is the
    // CONSTANT b"titan-bus", not titan_id. Per-Titan isolation comes from
    // the per-Titan identity secret (different IKM → different authkey).
    // Restored 2026-05-05 after rFP_phase_c_bus_authkey_contract_fix.md
    // diagnosed handshake failures caused by Rust passing "titan_T3" while
    // Python worker passed "T3" → different authkeys → 100% handshake fail.
    let authkey = titan_core::authkey::derive_bus_authkey(identity.secret_seed.as_slice())
        .map_err(|_e| KernelError::Identity)?;
    let authkey_hex = hex::encode(authkey);

    info!(event = "BOOT_B3_SHM_SLOTS", dir = ?shm_dir, "B3 shm slot registry");
    let registry = SlotRegistry::create_all(&shm_dir)
        .map_err(|e| KernelError::SlotRegistry(format!("{e:?}")))?;
    let _cgn_slot: Slot =
        create_cgn_live_weights(&shm_dir).map_err(|e| KernelError::CgnSlot(format!("{e:?}")))?;
    info!(
        event = "BOOT_B3_SHM_SLOTS_DONE",
        kernel_slots = registry.count(),
        "B3 created kernel-owned shm slots + cgn_live_weights"
    );

    info!(event = "BOOT_B4_BUS_BIND", path = ?bus_socket, "B4 bus broker bind");
    let mut broker = BusBroker::new(titan_id, authkey.to_vec());
    broker
        .start(&bus_socket)
        .await
        .map_err(|e| KernelError::BrokerStart(format!("{e:?}")))?;
    let broker = Arc::new(broker);
    info!(
        event = "BOOT_B4_BUS_BIND_DONE",
        path = ?bus_socket,
        "B4 bus broker bound + accept loop running"
    );

    // B7: load L0 snapshot
    let l0_snapshot_path = data_dir.join("l0_snapshot.bin");
    let prev_snapshot = load_or_default(&l0_snapshot_path)
        .map_err(|e| KernelError::Persistence(format!("{e:?}")))?;
    let prev_boot_gen = prev_snapshot
        .as_ref()
        .map(|s| s.boot_generation)
        .unwrap_or(0);
    let boot_generation = prev_boot_gen + 1;
    info!(
        event = "BOOT_B7_SNAPSHOT_LOAD",
        boot_generation,
        prev_existed = prev_snapshot.is_some(),
        "B7 L0 snapshot loaded"
    );

    // B6: clocks. Wrap each kernel-managed slot we'll write to in
    // Arc<AsyncMutex<Slot>> so async loops can lock + write.
    info!(event = "BOOT_B6_CLOCKS_START", "B6 starting clocks");
    let circadian_slot = Arc::new(AsyncMutex::new(
        Slot::open(shm_dir.join("circadian.bin"))
            .map_err(|e| KernelError::SlotRegistry(format!("{e:?}")))?,
    ));
    let pi_slot = Arc::new(AsyncMutex::new(
        Slot::open(shm_dir.join("pi_heartbeat.bin"))
            .map_err(|e| KernelError::SlotRegistry(format!("{e:?}")))?,
    ));
    let epoch_slot = Arc::new(AsyncMutex::new(
        Slot::open(shm_dir.join("epoch_counter.bin"))
            .map_err(|e| KernelError::SlotRegistry(format!("{e:?}")))?,
    ));

    let shutdown = Arc::new(Notify::new());

    let runtime_handle = tokio::runtime::Handle::current();
    let epoch_publisher: Arc<dyn EpochTickPublisher> = Arc::new(BrokerEpochPublisher::new(
        broker.clone(),
        runtime_handle.clone(),
    ));

    let circadian_handle = {
        let slot = circadian_slot.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move { run_circadian_loop(slot, shutdown).await })
    };
    let pi_handle = {
        let pi = pi_slot.clone();
        let epoch = epoch_slot.clone();
        let pub_ = epoch_publisher.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move { run_pi_heartbeat_loop(pi, epoch, pub_, shutdown).await })
    };

    // B11: L0 snapshot loop
    info!(
        event = "BOOT_B11_PERSISTENCE_LOOP",
        "B11 L0 snapshot loop start"
    );
    let snapshot_state = Arc::new(Mutex::new(SnapshotState {
        current: L0Snapshot {
            spec_version: SPEC_VERSION.into(),
            boot_generation,
            ..L0Snapshot::new(SPEC_VERSION)
        },
    }));
    let snapshot_handle = {
        let path = l0_snapshot_path.clone();
        let state = snapshot_state.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move { run_snapshot_loop(path, state, shutdown).await })
    };

    // B7.5 (C-S3 chunk C3-6): kernel-side fastbus producer attach.
    // Must happen BEFORE substrate spawn — substrate's first attach in C3-6
    // sees ring header version=1 (kernel initialized it) rather than racing
    // on init.
    let fastbus_path = shm_dir.join("fastbus.bin");
    info!(
        event = "BOOT_B7_5_FASTBUS_ATTACH",
        path = ?fastbus_path,
        "B7.5 kernel fastbus producer attach"
    );
    let fastbus_handle = spawn_kernel_fastbus_publisher(fastbus_path.clone(), shutdown.clone())
        .map_err(|e| KernelError::FastbusAttach(format!("{e:?}")))?;
    info!(
        event = "BOOT_B7_5_FASTBUS_ATTACH_DONE",
        "B7.5 kernel fastbus producer running"
    );

    // B8: spawn substrate (real titan-trinity-rs since C-S3 chunk C3-3 rename)
    let substrate_binary = options.substrate_binary.unwrap_or_else(|| {
        // Default location: same directory as kernel binary
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.join("titan-trinity-rs")))
            .unwrap_or_else(|| PathBuf::from("titan-trinity-rs"))
    });

    // B8.5: SPEC §11.B.5 — kernel owns the api listening socket (socket
    // activation). Bound ONCE here, held for the kernel's whole lifetime
    // (`api_listen_socket` must NOT be dropped until shutdown — every api child
    // serves uvicorn on this exact fd). OLD + NEW api share this one
    // never-destroyed accept queue across a zero-downtime reload ⇒ zero dropped
    // connections. Also makes api crash-respawn rebind-free. `None` (bind
    // failed, or api spawn disabled) ⇒ the api self-binds (degraded: reloads
    // are no longer zero-downtime, but the api still serves).
    let api_listen_socket = if options.spawn_titan_hcl_api {
        bind_api_listen_socket(&cli.config)
    } else {
        None
    };
    let api_listen_fd = api_listen_socket.as_ref().map(|s| s.as_raw_fd());

    let spawn_config = SpawnConfig {
        titan_id: titan_id.into(),
        boot_generation,
        bus_socket: bus_socket.clone(),
        fastbus_path: shm_dir.join("fastbus.bin"),
        shm_dir: shm_dir.clone(),
        data_dir: data_dir.clone(),
        authkey_hex: authkey_hex.clone(),
        log_level: format!("{:?}", cli.log_level).to_lowercase(),
        substrate_binary,
        python_executable: std::env::var_os("PYTHON")
            .or_else(|| Some("python3".into()))
            .map(PathBuf::from),
        python_cwd: std::env::current_dir().ok(),
        spawn_guardian_hcl: options.spawn_guardian_hcl,
        spawn_titan_hcl: options.spawn_titan_hcl,
        spawn_titan_hcl_api: options.spawn_titan_hcl_api,
        api_listen_fd,
    };

    // Phase C C-S7 Gap B (2026-05-05): wire substrate + python_main spawns
    // through the KernelChildSupervisor so unexpected exits trigger the
    // SPEC §11.B restart cascade (classify → respawn or escalate) instead
    // of leaving the tree dead. The supervisor wraps titan-core's Supervisor
    // primitive (decision logic + escalation) and emits supervision events
    // through JsonlSupervisionPublisher (data/supervision.jsonl + bus broker).
    let kernel_supervisor = KernelChildSupervisor::new(
        spawn_config.clone(),
        boot_generation,
        Some(broker.clone()),
        shutdown.clone(),
        runtime_handle.clone(),
        &data_dir,
    )
    .map_err(|e| KernelError::Supervisor(format!("{e:?}")))?;

    // Legacy SpawnedChildren registry. NOTE: in production it stays EMPTY —
    // every child (substrate + the 3 Python peers) is spawned through
    // KernelChildSupervisor's `spawn_and_watch_*`, whose Child handles live
    // inside the watch tasks, NOT here. So `children.sigterm_all()` is a no-op
    // and the real shutdown SIGTERM goes through `kernel_supervisor
    // .sigterm_children()` (SPEC §18.4 / RFP_supervision_lifecycle §7.D). This
    // empty registry is retained only to future-proof a direct-spawn path.
    let children = SpawnedChildren::new();

    let mut substrate_watch_handle: Option<tokio::task::JoinHandle<()>> = None;
    let mut python_watch_handle: Option<tokio::task::JoinHandle<()>> = None;
    let mut titan_hcl_watch_handle: Option<tokio::task::JoinHandle<()>> = None;
    let mut titan_hcl_api_watch_handle: Option<tokio::task::JoinHandle<()>> = None;

    if options.spawn_substrate {
        info!(
            event = "BOOT_B8_SPAWN_SUBSTRATE",
            "B8 spawning substrate via KernelChildSupervisor"
        );
        match kernel_supervisor.spawn_and_watch_substrate() {
            Ok(handle) => {
                substrate_watch_handle = Some(handle);
                info!("B8 substrate watch task running");
            }
            Err(e) => {
                return Err(KernelError::SubstrateSpawn(format!("{e:?}")));
            }
        }
    } else {
        info!("B8 substrate spawn skipped (test mode)");
    }

    // B9: spawn python_main via supervisor
    if options.spawn_guardian_hcl {
        info!(
            event = "BOOT_B9_SPAWN_PYTHON",
            "B9 spawning python -m titan_hcl via KernelChildSupervisor"
        );
        match kernel_supervisor.spawn_and_watch_python() {
            Ok(Some(handle)) => {
                python_watch_handle = Some(handle);
                info!("B9 python_main watch task running");
            }
            Ok(None) => {
                info!("B9 python_main spawn returned None (disabled)");
            }
            Err(e) => {
                // Per SPEC §11.E: Python crashes are not fatal at boot;
                // log + continue. The supervisor's max_restarts handles
                // sustained-failure mode via escalation.
                warn!(err = ?e, "B9 python_main spawn failed; continuing without");
            }
        }
    } else {
        info!("B9 python_main spawn skipped (default off)");
    }

    // B9.b: Phase 11 §11.I.1 / D-SPEC-141 — kernel-rs peer-spawns
    // titan_hcl (orchestrator) + titan_hcl_api as siblings to guardian_hcl.
    // Phase 11.x (Maker 2026-05-28): SUPERVISED via KernelChildSupervisor
    // (was fire-and-forget direct spawn → zombied on death). The supervisor
    // now watches + respawns all 3 Python peers via their own spawn fns, so
    // `kill -9 titan_hcl|titan_hcl_api` self-recovers (INV-PROC-5). Graceful
    // shutdown SIGTERMs these peers explicitly via
    // `kernel_supervisor.sigterm_children()` at SHUTDOWN_BEGIN (SPEC §18.4 /
    // RFP_supervision_lifecycle §7.D) — NOT via PDEATHSIG, which only fires on
    // kernel exit and would arrive after the broker is already gone. (PDEATHSIG
    // remains the backstop for an UNGRACEFUL kernel death.)
    if options.spawn_titan_hcl {
        info!(
            event = "BOOT_B9b_SPAWN_TITAN_HCL",
            "B9.b spawning titan_hcl (Phase 11 peer, supervised)"
        );
        match kernel_supervisor.spawn_and_watch_titan_hcl() {
            Ok(Some(handle)) => {
                titan_hcl_watch_handle = Some(handle);
                info!("B9.b titan_hcl watch task running");
            }
            Ok(None) => info!("B9.b titan_hcl spawn returned None (disabled)"),
            Err(e) => warn!(err = ?e, "B9.b titan_hcl spawn failed; continuing"),
        }
    }
    if options.spawn_titan_hcl_api {
        info!(
            event = "BOOT_B9c_SPAWN_TITAN_HCL_API",
            "B9.c spawning titan_hcl_api (Phase 11 peer, supervised)"
        );
        match kernel_supervisor.spawn_and_watch_titan_hcl_api() {
            Ok(Some(handle)) => {
                titan_hcl_api_watch_handle = Some(handle);
                info!("B9.c titan_hcl_api watch task running");
            }
            Ok(None) => info!("B9.c titan_hcl_api spawn returned None (disabled)"),
            Err(e) => warn!(err = ?e, "B9.c titan_hcl_api spawn failed; continuing"),
        }
    }

    // B9.d: SPEC §11.B.5 / D-SPEC-149 — the kernel's FIRST inbound bus
    // subscriber. Subscribes `KERNEL_API_RELOAD_REQUEST` over a BusClient to
    // the kernel's own broker socket and forwards each command onto the api
    // watch_loop's reload channel for the zero-downtime swap. Only meaningful
    // when the api peer is supervised; spawned unconditionally (it just idles
    // waiting for a command if the api isn't running).
    let api_reload_subscriber_handle = crate::api_reload_subscriber::spawn_api_reload_subscriber(
        bus_socket.clone(),
        authkey.to_vec(),
        kernel_supervisor.api_reload_sender(),
        shutdown.clone(),
    );

    info!(
        event = "BOOT_COMPLETE",
        boot_generation,
        spawned_substrate = options.spawn_substrate,
        spawned_python = options.spawn_guardian_hcl,
        spawned_titan_hcl = options.spawn_titan_hcl,
        spawned_titan_hcl_api = options.spawn_titan_hcl_api,
        "kernel boot complete; entering steady state"
    );

    // ── B12 Steady state ──────────────────────────────────────────────
    // Phase C C-S7 Gap B: also wake on shutdown.notify_waiters() — the
    // KernelChildSupervisor signals this when an escalation resolves to
    // Terminate (kernel must exit with code 64 → systemd cascades fresh tree).
    if let Some(d) = options.auto_shutdown_after {
        // Test mode: auto-shutdown after N seconds.
        tokio::time::sleep(d).await;
        info!("auto_shutdown_after elapsed; triggering shutdown");
    } else {
        // Production: wait for SIGTERM/SIGINT or supervisor-driven shutdown.
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("install SIGINT handler");
        tokio::select! {
            _ = sigterm.recv() => info!(reason = "SIGTERM", "shutdown signal received"),
            _ = sigint.recv() => info!(reason = "SIGINT", "shutdown signal received"),
            _ = shutdown.notified() => info!(
                reason = "supervisor_escalation",
                terminate_requested = kernel_supervisor.terminate_requested(),
                "supervisor signaled shutdown (escalation cascade)"
            ),
        }
    }

    // ── Graceful shutdown per SPEC §17 + §18.4 ────────────────────────
    // Capture whether the shutdown was triggered by a supervisor escalation
    // (terminate decision) BEFORE notifying the watch tasks — they will
    // observe child exit during shutdown and treat it as clean.
    let supervisor_terminate = kernel_supervisor.terminate_requested();
    info!(
        event = "SHUTDOWN_BEGIN",
        grace_s = KERNEL_SHUTDOWN_GRACE_S,
        supervisor_terminate,
        "kernel shutdown begin"
    );
    kernel_supervisor.mark_shutdown_active();
    shutdown.notify_waiters();

    // SPEC §18.4 / RFP_supervision_lifecycle §7.D (Phase D.2) — explicitly
    // SIGTERM every supervised child (substrate, guardian_hcl, titan_hcl,
    // titan_hcl_api) NOW, while the bus broker is still alive. The legacy
    // `children.sigterm_all()` is a no-op (the Phase-11 peers are owned by
    // KernelChildSupervisor's watch tasks, never registered into the empty
    // SpawnedChildren registry), so the real signal goes through the
    // supervisor's pid bookkeeping. Under `KillMode=mixed` this is the ONLY
    // thing that starts the children draining — systemd signals just the kernel
    // ($MAINPID), and PDEATHSIG fires only on kernel exit (too late: the broker
    // would already be gone). `mark_shutdown_active()` (above) ensures these
    // exits are classified clean (no respawn).
    children.sigterm_all().await; // retained: no-op today, future-proofs the registry path
    kernel_supervisor.sigterm_children().await;

    // Phase 1 — fast L0/L1 drain: the kernel-internal loops resolve on
    // shutdown.notify_waiters(); substrate (L1) + guardian_hcl exit quickly.
    // Bounded by KERNEL_SHUTDOWN_GRACE_S.
    let _ = tokio::time::timeout(Duration::from_secs_f64(KERNEL_SHUTDOWN_GRACE_S), async {
        let _ = circadian_handle.await;
        let _ = pi_handle.await;
        let _ = snapshot_handle.await;
        let _ = fastbus_handle.await;
        if let Some(h) = substrate_watch_handle {
            let _ = h.await;
        }
        if let Some(h) = python_watch_handle {
            let _ = h.await;
        }
        // B9.d api reload subscriber exits on shutdown.notified().
        let _ = api_reload_subscriber_handle.await;
    })
    .await;

    // Phase 2 — Python L2/L3 drain: keep the broker ALIVE until titan_hcl
    // (orchestrator) + titan_hcl_api exit. titan_hcl drains its ~40 modules
    // SEQUENTIALLY, each running a bus SAVE_NOW→SAVE_DONE handshake over THIS
    // broker, so the broker must outlive the drain — that ordering is the whole
    // §18.4 fix. Bounded by KERNEL_PYTHON_DRAIN_GRACE_S (< systemd
    // TimeoutStopSec, the outer SIGKILL backstop); resolves early the instant
    // both peers exit. Phase D.1's bus-independent self-save is the belt to
    // this suspenders: even if the bound is hit, no worker loses data.
    let _ = tokio::time::timeout(
        Duration::from_secs_f64(KERNEL_PYTHON_DRAIN_GRACE_S),
        async {
            if let Some(h) = titan_hcl_watch_handle {
                let _ = h.await;
            }
            if let Some(h) = titan_hcl_api_watch_handle {
                let _ = h.await;
            }
        },
    )
    .await;

    // Stop broker
    let mut broker_owned = match Arc::try_unwrap(broker) {
        Ok(b) => b,
        Err(arc) => {
            warn!(
                refcount = Arc::strong_count(&arc),
                "broker still referenced at shutdown; aborting handles"
            );
            return Ok(KernelExitCode::Clean);
        }
    };
    broker_owned.stop().await;

    info!(event = "SHUTDOWN_COMPLETE", "kernel shutdown complete");
    // Phase C C-S7 Gap B: if shutdown was triggered by supervisor escalation
    // resolving to Terminate, exit with code 64 (escalation range per
    // SPEC §15 + §11.B.1 step 6b) so systemd cascades a fresh tree.
    if supervisor_terminate {
        Ok(KernelExitCode::SupervisorSelfTerminate)
    } else {
        Ok(KernelExitCode::Clean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_api_bind_from_config_extracts_port_and_host() {
        // SPEC §11.B.5 — the kernel must bind each Titan's CONFIGURED port
        // (e.g. T3 = 7778), not a hardcoded default.
        let f = tempfile_with("[network]\nx = 1\n\n[api]\nhost = \"127.0.0.1\"\nport = 7778\n");
        let (host, port) = read_api_bind_from_config(f.path());
        assert_eq!(host.as_deref(), Some("127.0.0.1"));
        assert_eq!(port, Some(7778));
        f.close();
    }

    #[test]
    fn read_api_bind_from_config_missing_keys_and_file() {
        // No [api] section → both None (caller falls back to env/default).
        let f = tempfile_with("[network]\nx = 1\n");
        assert_eq!(read_api_bind_from_config(f.path()), (None, None));
        f.close();
        // Unreadable file → (None, None), never panics.
        assert_eq!(
            read_api_bind_from_config(std::path::Path::new("/nonexistent/config.toml")),
            (None, None)
        );
        // Malformed TOML → (None, None).
        let bad = tempfile_with("this is not = = toml [[[");
        assert_eq!(read_api_bind_from_config(bad.path()), (None, None));
        bad.close();
    }

    /// Minimal self-cleaning temp file (no extra dep — writes under the test
    /// tmp dir keyed by the calling test via a monotonic-ish unique name).
    struct TmpCfg {
        path: std::path::PathBuf,
    }
    impl TmpCfg {
        fn path(&self) -> &std::path::Path {
            &self.path
        }
        fn close(self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
    fn tempfile_with(contents: &str) -> TmpCfg {
        // Unique name from process id + a hash of the contents (Date/rand are
        // unavailable in this workspace's test policy; contents differ per test).
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(contents, &mut hasher);
        let h = std::hash::Hasher::finish(&hasher);
        let path = std::env::temp_dir().join(format!(
            "titan_kernel_api_cfg_test_{}_{}.toml",
            std::process::id(),
            h
        ));
        let mut file = std::fs::File::create(&path).expect("create temp config");
        file.write_all(contents.as_bytes())
            .expect("write temp config");
        TmpCfg { path }
    }

    #[test]
    fn kernel_error_to_exit_code_matches_spec_15() {
        assert_eq!(
            KernelError::Identity.to_exit_code(),
            KernelExitCode::IdentityLoadFailure
        );
        assert_eq!(
            KernelError::SlotRegistry("x".into()).to_exit_code(),
            KernelExitCode::ShmCreateFailure
        );
        assert_eq!(
            KernelError::BrokerStart("x".into()).to_exit_code(),
            KernelExitCode::BusBindFailure
        );
        assert_eq!(
            KernelError::SubstrateSpawn("x".into()).to_exit_code(),
            KernelExitCode::ChildLimitReached
        );
    }
}
