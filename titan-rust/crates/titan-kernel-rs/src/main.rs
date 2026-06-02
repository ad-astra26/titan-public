//! titan-kernel-rs — Microkernel v2 Phase C Rust L0 binary.
//!
//! Per SPEC §10.A boot sequence (T+0ms → T+1000ms steady state) +
//! §13 CLI contract + §15 exit codes + §16 logging + §17 process discipline.
//!
//! C-S2 chunks shipping incrementally:
//! - **C2-6.a**: skeleton + CLI + logging + exit codes
//! - **C2-6.b**: boot sequence integration (identity, shm, clocks, bus)
//! - **C2-6.c**: spawn + supervision.jsonl + L0 persistence + shutdown
//!
//! With C2-6.c shipped, the kernel boots end-to-end behind the
//! `microkernel.l0_rust_enabled` flag (default false until C-S7).
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod api_reload_subscriber;
pub mod broker_publisher;
pub mod cli;
pub mod exit;
pub mod fastbus_publisher;
pub mod identity_load;
pub mod kernel;
pub mod kernel_supervisor;
pub mod logging;
pub mod persistence;
pub mod spawn;
pub mod substrate_watch;
pub mod supervision_log;
pub mod version;

use std::process::ExitCode;

use clap::Parser;
use nix::sys::signal::Signal;
use tracing::{error, info};

use crate::cli::Cli;
use crate::kernel::{run, KernelRunOptions};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    // SPEC §16 logging init
    let _log_guard = logging::init(cli.log_level.to_tracing(), cli.titan_id.as_str());

    // SPEC §11.C(1): kernel sets PR_SET_CHILD_SUBREAPER so orphaned
    // descendants get reparented here instead of init/systemd.
    if let Err(e) = titan_core::supervisor::prctl_unix::set_child_subreaper(true) {
        tracing::warn!(err = ?e, "set_child_subreaper failed (non-Linux dev?); continuing");
    }
    // Also set our own PDEATHSIG so we die if our parent (systemd) does.
    if let Err(e) = titan_core::supervisor::prctl_unix::set_pdeathsig(Signal::SIGTERM) {
        tracing::warn!(err = ?e, "set_pdeathsig failed; continuing");
    }

    info!(
        event = "BOOT_STARTING",
        titan_id = cli.titan_id.as_str(),
        cargo_version = version::CARGO_VERSION,
        git_sha = version::GIT_SHA,
        spec_version = titan_core::constants::SPEC_VERSION,
        full_version = %version::full_version(),
        "titan-kernel-rs boot starting"
    );

    info!(
        shm_dir = ?cli.effective_shm_dir(),
        bus_socket = ?cli.effective_bus_socket(),
        kernel_rpc_socket = ?cli.effective_kernel_rpc_socket(),
        data_dir = ?cli.data_dir,
        "boot configuration resolved"
    );

    // Phase C C-S7 activation prep (2026-05-05): production runs MUST spawn
    // the Python plugin (titan_HCL — runs Guardian + L2/L3 modules + API).
    // KernelRunOptions::default() ships with spawn_guardian_hcl=false because tests
    // construct it without overrides; production explicitly flips it on here.
    // Per SPEC §10.A B9 + §9.B titan_HCL row.
    //
    // Cross-language integration tests (rFP_phase_c_bus_authkey_contract_fix.md
    // §3) need to boot a kernel with a tmp-dir data path and check Python
    // worker handshake. They CAN'T spawn python_main (it would conflict with
    // the production T1 PID file at the canonical CWD). Setting env var
    // `TITAN_KERNEL_SKIP_PYTHON=1` disables the spawn for those tests.
    // Production never sets this var.
    let skip_python = std::env::var("TITAN_KERNEL_SKIP_PYTHON")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Phase 11 §11.I.1 / D-SPEC-141 — kernel-rs peer-spawns titan_hcl +
    // titan_hcl_api as siblings to guardian_hcl. Python side shipped:
    //   * scripts/titan_hcl.py owns Orchestrator + start_all + lifecycle
    //   * scripts/guardian_hcl.py reduced to Supervisor (no Popen)
    //   * scripts/titan_hcl_api.py is the L3 peer entry-point
    // INV-PROC-3 / INV-PROC-5 hold: independent crash domains, titan_hcl_api
    // stays UP through titan_hcl restart. TITAN_KERNEL_SKIP_PYTHON remains
    // the test-only escape hatch (rFP_phase_c_bus_authkey_contract_fix §3
    // cross-language integration tests).
    let options = KernelRunOptions {
        spawn_guardian_hcl: !skip_python,
        spawn_titan_hcl: !skip_python,
        spawn_titan_hcl_api: !skip_python,
        ..KernelRunOptions::default()
    };

    match run(&cli, options).await {
        Ok(exit_code) => {
            info!(
                event = "EXIT",
                code = exit_code as u8,
                label = exit_code.as_str(),
                "kernel exit"
            );
            exit_code.to_exit_code()
        }
        Err(e) => {
            let exit_code = e.to_exit_code();
            error!(
                event = "EXIT",
                code = exit_code as u8,
                label = exit_code.as_str(),
                err = ?e,
                "kernel boot/run failed"
            );
            exit_code.to_exit_code()
        }
    }
}
