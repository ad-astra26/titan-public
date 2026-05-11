//! titan-unified-spirit-rs — Microkernel v2 Phase C Rust L1b binary.
//!
//! Per SPEC §9.A unified-spirit-rs row + §10.A boot sequence + §13 CLI
//! contract + §15 exit codes + §16 logging + §17 process discipline.
//!
//! C-S4 chunks shipped:
//! - **C4-1**: scaffolding (CLI + logging + exit codes + PDEATHSIG + SIGTERM)
//! - **C4-2**: 162D SELF assembly + SeqLock writer + body_cycle_loop
//! - **C4-2b1**: ResonancePair + ResonanceDetector + SPHERE_PULSE subscriber
//! - **C4-2b2**: UnifiedSpirit + GreatEpoch + advance() + persistence
//! - **C4-2c**: shared titan-bus::client + substrate main-bus integration
//! - **C4-3a/b/c**: V5 TrinityValueNet + TransitionBuffer + TD(0) +
//!   FilterDownV5Engine + UNIFIED_SPIRIT_FILTER_DOWN publish
//! - **C4-4**: 6-daemon supervisor (one_for_one + escalation + dep checks)
//! - **C4-5** (this binary): full integration via [`runtime::Runtime`]
//!
//! Binary lifecycle: parse CLI → init logging + PDEATHSIG → boot Runtime
//! (connect bus, subscribe, init engines, spawn body_cycle + publisher +
//! dispatch loops + 6 daemons) → wait for graceful shutdown → cascade
//! exit.
//!
//! Behind `microkernel.l0_rust_enabled = false` flag default per SPEC §3.0
//! Running-Titans Safety Rule until C-S7 first flag-flip.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use tracing::info;

use titan_unified_spirit_rs::cli::Cli;
use titan_unified_spirit_rs::logging;
use titan_unified_spirit_rs::runtime::Runtime;
use titan_unified_spirit_rs::version;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    // SPEC §16 logging init.
    let _log_guard = logging::init(cli.log_level.to_tracing(), cli.titan_id.as_str());

    // SPEC §11.C(1): every Rust child sets PR_SET_PDEATHSIG so it dies
    // when its parent (substrate) does. Best-effort — log + continue if
    // not Linux (e.g. macOS dev box).
    if let Err(e) = titan_core::supervisor::prctl_unix::set_pdeathsig(Signal::SIGTERM) {
        tracing::warn!(err = ?e, "set_pdeathsig failed (non-Linux dev?); continuing");
    }

    info!(
        event = "BOOT_STARTING",
        binary = "unified-spirit",
        titan_id = cli.titan_id.as_str(),
        cargo_version = version::CARGO_VERSION,
        git_sha = version::GIT_SHA,
        spec_version = titan_core::constants::SPEC_VERSION,
        full_version = %version::full_version(),
        pid = std::process::id(),
        "titan-unified-spirit-rs boot starting"
    );
    info!(
        shm_dir = ?cli.effective_shm_dir(),
        bus_socket = ?cli.effective_bus_socket(),
        data_dir = ?cli.data_dir,
        daemon_binary_dir = ?cli.daemon_binary_dir,
        use_placeholder_daemons = cli.use_placeholder_daemons,
        self_assembly_cadence_ms = cli.self_assembly_cadence_ms,
        "boot configuration resolved"
    );

    // Boot the integrated runtime (Steps 2-9 per SPEC §10.A).
    let runtime = match Runtime::boot(&cli).await {
        Ok(r) => r,
        Err(code) => {
            tracing::error!(
                event = "BOOT_FAILED",
                exit_code = code as u8,
                label = code.as_str(),
                "runtime boot failed"
            );
            return code.to_exit_code();
        }
    };

    let exit_code = runtime.run().await;

    info!(
        event = "EXIT",
        binary = "unified-spirit",
        titan_id = cli.titan_id.as_str(),
        code = exit_code as u8,
        label = exit_code.as_str(),
        pid = std::process::id(),
        "unified-spirit exit"
    );
    exit_code.to_exit_code()
}
