//! titan-inner-mind-rs — Inner-mind trinity daemon.
//!
//! Per SPEC §9.A `titan-inner-mind-rs` row + master plan §10.5 chunk C5-3:
//! 23.49 Hz Schumann tick (G13 — body × 3), reads sensor_cache_inner_mind +
//! inner_lower topology slice, applies UNIFIED + LOCAL filter_down to
//! mind[0:15], applies ground_up to mind[10:15] WILLING ONLY (G10 —
//! thinking[0:5] + feeling[5:10] are NOT grounded), writes
//! inner_mind_15d.bin (84 bytes total) content-hash gated, publishes
//! MIND_STATE (P1 coalesce-by-(src,type)) every tick.

mod tick_loop;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use titan_core::supervisor::prctl_unix::set_pdeathsig;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(version, about = "Titan Inner-Mind Trinity Daemon")]
struct Cli {
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID")]
    titan_id: Option<String>,
    #[arg(long, env = "TITAN_BUS_SOCKET")]
    bus_socket: Option<PathBuf>,
    #[arg(long, env = "TITAN_AUTHKEY_HEX")]
    authkey_hex: Option<String>,
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    shm_dir: Option<PathBuf>,
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
        binary = "titan-inner-mind-rs",
        titan_id = %titan_id,
        pid = std::process::id(),
        bus_socket = ?bus_socket,
        shm_dir = ?shm_dir,
        event = "BOOT_START",
        "inner-mind daemon boot start"
    );

    if let Err(e) = set_pdeathsig(Signal::SIGTERM) {
        warn!(err = ?e, "set_pdeathsig failed");
    }

    if let Err(e) = tick_loop::run(&bus_socket, &authkey, &shm_dir).await {
        error!(err = ?e, "inner-mind daemon exited with error");
        return ExitCode::from(1);
    }

    info!(event = "SHUTDOWN_CLEAN");
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
