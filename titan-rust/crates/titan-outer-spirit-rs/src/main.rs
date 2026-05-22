//! titan-outer-spirit-rs — Outer-spirit trinity daemon.
//!
//! Per SPEC §9.A `titan-outer-spirit-rs` row + §18.1 + master plan
//! §10.6 chunk C6-5. Jittered ~30s ±10% tick, reads outer_body_5d.bin
//! + outer_mind_15d.bin (Observer Principle G8) + sensor_cache_outer_spirit
//! (msgpack pre-aggregated outer-state from Python sidecar), computes
//! 45D Sat-Chit-Ananda Material (SAT[0:15] + CHIT[15:30] + ANANDA[30:45]),
//! writes outer_spirit_45d.bin (204 bytes total) content-hash gated.
//!
//! **Two distinct publications per tick:**
//! 1. SPIRIT_STATE (P1, src=outer, full 45D unmasked) — for L2 consumers
//!    (CGN, reasoning, MSL) that need the complete spirit state.
//! 2. OUTER_SPIRIT_FILTER_DOWN (P1, LOCAL bias) — payload contains
//!    `outer_spirit_content[40]` = `outer_spirit_45d[5:45]` with observer
//!    dims `[0:5]` MASKED per SPEC G8 + §10.F step 3 (Observer Principle:
//!    "Spirit observer is a reflection surface, not a target of filtering").

use titan_outer_spirit_rs::tick_loop;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use titan_core::supervisor::prctl_unix::set_pdeathsig;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(version, about = "Titan Outer-Spirit Trinity Daemon")]
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
        binary = "titan-outer-spirit-rs",
        titan_id = %titan_id,
        pid = std::process::id(),
        bus_socket = ?bus_socket,
        shm_dir = ?shm_dir,
        event = "BOOT_START",
        "outer-spirit daemon boot start"
    );

    if let Err(e) = set_pdeathsig(Signal::SIGTERM) {
        warn!(err = ?e, "set_pdeathsig failed (non-Linux dev?)");
    }

    if let Err(e) = tick_loop::run(&bus_socket, &authkey, &shm_dir, &data_dir).await {
        error!(err = ?e, "outer-spirit daemon exited with error");
        return ExitCode::from(1);
    }

    info!(event = "SHUTDOWN_CLEAN", "outer-spirit daemon clean exit");
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
