//! titan-outer-mind-rs — Outer-mind trinity daemon.
//!
//! Per SPEC §9.A `titan-outer-mind-rs` row + §18.1 outer cadence
//! resolution + master plan §10.6 chunk C6-4. Jittered ~5s ±20% tick,
//! reads sensor_cache_outer_mind (msgpack) + outer_lower topology slice,
//! applies UNIFIED + LOCAL filter_down to all 15D, applies ground_up
//! **to willing[10:15] ONLY** per SPEC G10 (Thinking + Feeling NOT
//! grounded), writes outer_mind_15d.bin (84 bytes total) content-hash
//! gated, publishes MIND_STATE (P1 coalesce-by-(src,type), src=outer)
//! every tick.

use titan_outer_mind_rs::tick_loop;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use titan_core::supervisor::prctl_unix::set_pdeathsig;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(version, about = "Titan Outer-Mind Trinity Daemon")]
struct Cli {
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID")]
    titan_id: Option<String>,
    #[arg(long, env = "TITAN_BUS_SOCKET")]
    bus_socket: Option<PathBuf>,
    #[arg(long, env = "TITAN_AUTHKEY_HEX")]
    authkey_hex: Option<String>,
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    shm_dir: Option<PathBuf>,
    /// Data directory for disk-persistent state (§G5.2 item 4 checkpoint
    /// sidecars + §24 sovereign-backup chain). Defaults to "data" relative
    /// to cwd. D-SPEC-126.
    #[arg(long, env = "TITAN_KERNEL_DATA_DIR")]
    data_dir: Option<PathBuf>,
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
    let bus_socket = cli
        .bus_socket
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/titan_bus_{titan_id}.sock")));
    let shm_dir = cli
        .shm_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("/dev/shm/titan_{titan_id}")));
    let data_dir = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("data"));
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
        binary = "titan-outer-mind-rs",
        titan_id = %titan_id,
        pid = std::process::id(),
        bus_socket = ?bus_socket,
        shm_dir = ?shm_dir,
        event = "BOOT_START",
        "outer-mind daemon boot start"
    );

    if let Err(e) = set_pdeathsig(Signal::SIGTERM) {
        warn!(err = ?e, "set_pdeathsig failed (non-Linux dev?)");
    }

    if let Err(e) = tick_loop::run(&bus_socket, &authkey, &shm_dir, &data_dir).await {
        error!(err = ?e, "outer-mind daemon exited with error");
        return ExitCode::from(1);
    }

    info!(event = "SHUTDOWN_CLEAN", "outer-mind daemon clean exit");
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
