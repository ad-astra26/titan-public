//! titan-outer-body-rs — Outer-body trinity daemon.
//!
//! Per SPEC §9.A `titan-outer-body-rs` row + §18.1 outer cadence
//! resolution + master plan §10.6 chunk C6-3. Jittered ~10s ±20% tick
//! (SPEC §18.1: outer cadences are NOT Schumann-locked), reads
//! sensor_cache_outer_body (msgpack source dict from Python sidecar) +
//! outer_lower topology slice, applies UNIFIED + LOCAL filter_down
//! (multipliers from unified-spirit-rs + outer-spirit-rs), applies
//! ground_up (all 5D per G10 — body grounding scope), writes
//! outer_body_5d.bin (44 bytes total) content-hash gated, publishes
//! BODY_STATE (P1 coalesce-by-(src,type), src=outer) every tick.
//!
//! # Boot sequence (per SPEC §10.A child side)
//!
//! 1. Parse CLI + env (TITAN_KERNEL_TITAN_ID / TITAN_BUS_SOCKET /
//!    TITAN_AUTHKEY_HEX / TITAN_KERNEL_SHM_DIR).
//! 2. Init JSON logging (binary=titan-outer-body-rs, titan_id, pid).
//! 3. Set PR_SET_PDEATHSIG=SIGTERM (SPEC §11.C(1)).
//! 4. Connect to main bus + HMAC handshake + subscribe to OUTER_BODY_TOPICS.
//! 5. Open kernel-created shm slots (sensor_cache + topology + outer_body).
//! 6. Send MODULE_READY.
//! 7. Enter jittered tick loop ~10s ±20%; handle SIGTERM gracefully.

mod tick_loop;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use titan_core::supervisor::prctl_unix::set_pdeathsig;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(version, about = "Titan Outer-Body Trinity Daemon")]
struct Cli {
    /// Titan ID (T1 / T2 / T3). Falls back to TITAN_KERNEL_TITAN_ID env.
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID")]
    titan_id: Option<String>,

    /// Main bus Unix-socket path. Falls back to TITAN_BUS_SOCKET env.
    #[arg(long, env = "TITAN_BUS_SOCKET")]
    bus_socket: Option<PathBuf>,

    /// HKDF authkey as hex. Falls back to TITAN_AUTHKEY_HEX env.
    #[arg(long, env = "TITAN_AUTHKEY_HEX")]
    authkey_hex: Option<String>,

    /// Shm directory. Falls back to TITAN_KERNEL_SHM_DIR env.
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    shm_dir: Option<PathBuf>,

    /// Log level (info / debug / trace).
    #[arg(long, env = "TITAN_KERNEL_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(&cli.log_level);

    let titan_id = match cli.titan_id.clone() {
        Some(t) => t,
        None => {
            error!("missing TITAN_KERNEL_TITAN_ID");
            return ExitCode::from(2);
        }
    };
    let bus_socket = match cli.bus_socket.clone() {
        Some(p) => p,
        None => PathBuf::from(format!("/tmp/titan_bus_{titan_id}.sock")),
    };
    let shm_dir = match cli.shm_dir.clone() {
        Some(p) => p,
        None => PathBuf::from(format!("/dev/shm/titan_{titan_id}")),
    };
    let authkey_hex = match cli.authkey_hex.clone() {
        Some(s) => s,
        None => {
            error!("missing TITAN_AUTHKEY_HEX");
            return ExitCode::from(3);
        }
    };
    let authkey = match hex::decode(&authkey_hex) {
        Ok(k) => k,
        Err(e) => {
            error!(err=%e, "invalid TITAN_AUTHKEY_HEX");
            return ExitCode::from(3);
        }
    };

    info!(
        binary = "titan-outer-body-rs",
        titan_id = %titan_id,
        pid = std::process::id(),
        bus_socket = ?bus_socket,
        shm_dir = ?shm_dir,
        event = "BOOT_START",
        "outer-body daemon boot start"
    );

    if let Err(e) = set_pdeathsig(Signal::SIGTERM) {
        warn!(err = ?e, "set_pdeathsig failed (non-Linux dev?)");
    }

    if let Err(e) = tick_loop::run(&bus_socket, &authkey, &shm_dir).await {
        error!(err = ?e, "outer-body daemon exited with error");
        return ExitCode::from(1);
    }

    info!(event = "SHUTDOWN_CLEAN", "outer-body daemon clean exit");
    ExitCode::SUCCESS
}

fn init_logging(level: &str) {
    let _ = tracing_subscriber::fmt()
        .json()
        .with_target(false)
        .with_thread_ids(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .try_init();
}
