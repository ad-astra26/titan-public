//! titan-inner-spirit-rs — Inner-spirit trinity daemon.
//!
//! Per SPEC §9.A `titan-inner-spirit-rs` row + master plan §10.5 chunk C5-4:
//! 70.47 Hz Schumann tick (G13 — body × 9, the full Schumann spectrum,
//! period 14.19 ms), reads inner_body_5d + inner_mind_15d sibling slots
//! per Observer Principle (G8 — spirit observes Trinity, NOT pure compute),
//! reads sensor_cache_inner_spirit (Python sidecar pre-aggregated state),
//! writes inner_spirit_45d.bin (204 bytes — SAT[0:15] + CHIT[15:30] +
//! ANANDA[30:45]), publishes:
//!
//! - **SPIRIT_STATE** (P1, src=inner) — the full 45D tensor every tick.
//! - **INNER_SPIRIT_FILTER_DOWN** (P1, LOCAL bias to inner_body + inner_mind)
//!   — observer dims `[0:5]` MASKED at output per G8 (the
//!   inner_spirit_content field carries 40D, not 45D).
//!
//! Subscribes only to KERNEL_SHUTDOWN_ANNOUNCE + UNIFIED_SPIRIT_FILTER_DOWN
//! (no topology — Observer is not grounded per G10 + G8).

mod tick_loop;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use titan_core::supervisor::prctl_unix::set_pdeathsig;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(version, about = "Titan Inner-Spirit Trinity Daemon")]
struct Cli {
    #[arg(long, env = "TITAN_KERNEL_TITAN_ID")]
    titan_id: Option<String>,
    #[arg(long, env = "TITAN_BUS_SOCKET")]
    bus_socket: Option<PathBuf>,
    #[arg(long, env = "TITAN_AUTHKEY_HEX")]
    authkey_hex: Option<String>,
    #[arg(long, env = "TITAN_KERNEL_SHM_DIR")]
    shm_dir: Option<PathBuf>,
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
    // data_dir holds the per-half learned filter_down brain
    // (filter_down_local_inner_*.json). Falls back to ./data (daemons run
    // with cwd = TITAN_DIR under systemd).
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
        binary = "titan-inner-spirit-rs",
        titan_id = %titan_id,
        pid = std::process::id(),
        bus_socket = ?bus_socket,
        shm_dir = ?shm_dir,
        event = "BOOT_START",
        "inner-spirit daemon boot start"
    );

    if let Err(e) = set_pdeathsig(Signal::SIGTERM) {
        warn!(err = ?e, "set_pdeathsig failed");
    }

    if let Err(e) = tick_loop::run(&bus_socket, &authkey, &shm_dir, &data_dir).await {
        error!(err = ?e, "inner-spirit daemon exited with error");
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
