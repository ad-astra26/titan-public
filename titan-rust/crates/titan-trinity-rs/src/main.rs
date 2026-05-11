//! titan-trinity-rs — Microkernel v2 Phase C trinity-substrate L1a binary.
//!
//! Per PLAN_microkernel_phase_c_s3_substrate.md §7 (substrate boot S0–S14) +
//! SPEC §10.A B8 (substrate side).
//!
//! C-S3 chunks shipping incrementally:
//! - **C3-3**: skeleton + CLI + JSON logging + exit codes + prctl + spawn
//!   unified-spirit-placeholder (this module)
//! - **C3-4**: filter_down + ground_up coordination primitives
//! - **C3-5**: topology engine + sphere clocks + chi state + tick_loop
//! - **C3-6**: fastbus consumer attach + main bus client connect + MODULE_READY

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod body_cycle;
pub mod boot;
pub mod chi_state;
pub mod cli;
pub mod exit;
pub mod fastbus_consumer;
pub mod filter_down;
pub mod ground_up;
pub mod main_bus_publisher;
pub mod sphere_clocks;
pub mod supervise;
pub mod tick_loop;
pub mod topology;
pub mod version;

use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use tracing::{error, info};

use std::sync::Arc;

use tokio::sync::Notify;

use crate::boot::{install_prctl, log_boot_starting, BootConfig};
use crate::cli::Cli;
use crate::exit::SubstrateExitCode;
use crate::fastbus_consumer::{spawn_substrate_fastbus_consumer, ConsumerStats};
use crate::supervise::{SupervisedChildren, UnifiedSpiritSupervisor};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    // Step S3: tracing-subscriber init per SPEC §16
    let titan_id_for_log = if cli.titan_id.is_empty() {
        std::env::var("TITAN_KERNEL_TITAN_ID").unwrap_or_else(|_| "unknown".into())
    } else {
        cli.titan_id.clone()
    };
    let _log_guard = init_logging(cli.log_level.to_tracing(), &titan_id_for_log);

    // Step S2: prctl (PDEATHSIG + CHILD_SUBREAPER)
    if let Err(e) = install_prctl() {
        let code = e.to_exit_code();
        error!(event = "EXIT", code = code as u8, label = code.as_str(), err = ?e, "prctl install failed");
        return code.to_exit_code();
    }

    // Step S1: resolve config from CLI + env
    let cfg = match BootConfig::from_cli(&cli) {
        Ok(c) => c,
        Err(e) => {
            let code = e.to_exit_code();
            error!(event = "EXIT", code = code as u8, label = code.as_str(), err = ?e, "config resolution failed");
            return code.to_exit_code();
        }
    };

    log_boot_starting(&cfg);

    let shutdown = Arc::new(Notify::new());

    // Step S6 (PLAN §7.1): attach to kernel-pre-allocated fastbus.bin +
    // spawn consumer task. Per SPEC §9.A trinity-rs row "consumes circadian
    // + π events from kernel". Failure here = exit 5 (shm).
    let fastbus_stats = Arc::new(ConsumerStats::default());
    let fastbus_handle = match spawn_substrate_fastbus_consumer(
        cfg.fastbus_path.clone(),
        shutdown.clone(),
        fastbus_stats.clone(),
    ) {
        Ok(h) => Some(h),
        Err(e) => {
            let code = SubstrateExitCode::ShmOpenFailure;
            error!(event = "EXIT", code = code as u8, label = code.as_str(), err = ?e, "fastbus attach failed");
            return code.to_exit_code();
        }
    };
    info!(
        event = "BOOT_S6_FASTBUS_ATTACH_DONE",
        "S6 fastbus consumer running"
    );

    let children = SupervisedChildren::new();

    // Step S10: spawn unified-spirit via UnifiedSpiritSupervisor
    // (Phase C C-S7 Gap B — wires SPEC §11.0 row 3 cascade: unexpected
    // unified-spirit exit → respawn or escalate per §11.B). Mirrors the
    // kernel-side KernelChildSupervisor + the leaf DaemonSupervisor.
    let unified_spirit_supervisor = if cfg.spawn_unified_spirit {
        match UnifiedSpiritSupervisor::new(cfg.clone(), shutdown.clone()) {
            Ok(sup) => match sup.spawn_and_watch() {
                Ok(_handle) => {
                    info!("S10 unified-spirit watch task running via supervisor");
                    Some(sup)
                }
                Err(e) => {
                    let code = e.to_exit_code();
                    error!(event = "EXIT", code = code as u8, label = code.as_str(), err = ?e, "unified-spirit spawn failed");
                    return code.to_exit_code();
                }
            },
            Err(e) => {
                error!(err = ?e, "UnifiedSpiritSupervisor::new failed");
                return SubstrateExitCode::Generic.to_exit_code();
            }
        }
    } else {
        info!("S10 unified-spirit spawn skipped (--skip-unified-spirit-spawn)");
        None
    };

    // Step S14: emit MODULE_READY signal. Per SPEC §10.A B8 substrate side.
    info!(
        event = "MODULE_READY",
        binary = "trinity-substrate",
        spawned_unified_spirit = cfg.spawn_unified_spirit,
        fastbus_attached = fastbus_handle.is_some(),
        "substrate MODULE_READY"
    );

    // ── Step S7-S8 + S9 + S13: main bus client + body cycle task ─────
    // Closes rFP_phase_c_close_all_runtime_gaps chunks 9A + 9B (Gap A
    // substrate body cycle wiring + Gap B chi compute). Per SPEC §10.G
    // step 5 (substrate per-body-cycle orchestration), §10.E (telemetry
    // write-then-publish ordering), §11.B (defensive error handling).
    //
    // Pre-9A: `SubstrateState::body_tick` was pure compute with zero
    // production callers (only #[cfg(test)]); topology_30d.bin /
    // sphere_clocks.bin / chi_state.bin sat at version=1 forever.
    let body_cycle_handle = match (cfg.bus_socket.as_ref(), cfg.authkey_hex.as_ref()) {
        (Some(socket_path), Some(authkey_hex)) => {
            match crate::main_bus_publisher::connect_main_bus(
                socket_path,
                authkey_hex,
                &cfg.titan_id,
            )
            .await
            {
                Ok(bus_client) => match crate::body_cycle::BodyCycleSlots::open(&cfg.shm_dir) {
                    Ok(slots) => {
                        let task = tokio::spawn(crate::body_cycle::run_substrate_body_cycle(
                            slots,
                            bus_client,
                            shutdown.clone(),
                        ));
                        info!(
                            event = "BODY_CYCLE_SPAWNED",
                            "substrate body cycle task spawned (rFP chunks 9A+9B)"
                        );
                        Some(task)
                    }
                    Err(e) => {
                        error!(
                            event = "BODY_CYCLE_SLOTS_OPEN_FAILED",
                            err = ?e,
                            "could not open body-cycle slots; substrate continues without body cycle (will mark slots STUCK)"
                        );
                        None
                    }
                },
                Err(e) => {
                    error!(
                        event = "MAIN_BUS_CONNECT_FAILED",
                        err = ?e,
                        "main bus connect failed; substrate continues without body cycle"
                    );
                    None
                }
            }
        }
        _ => {
            info!(
                event = "BODY_CYCLE_SKIPPED",
                bus_socket = ?cfg.bus_socket,
                authkey_present = cfg.authkey_hex.is_some(),
                "skipping body cycle: bus_socket or authkey_hex not configured (test mode)"
            );
            None
        }
    };

    // ── Steady state — wait for SIGTERM/SIGINT or supervisor escalation ─
    if cli.auto_shutdown_after_s > 0 {
        tokio::time::sleep(Duration::from_secs(cli.auto_shutdown_after_s)).await;
        info!("auto_shutdown_after_s elapsed; triggering shutdown");
    } else {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    error!(err = ?e, "failed to install SIGTERM handler");
                    return SubstrateExitCode::Generic.to_exit_code();
                }
            };
        let mut sigint =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    error!(err = ?e, "failed to install SIGINT handler");
                    return SubstrateExitCode::Generic.to_exit_code();
                }
            };
        tokio::select! {
            _ = sigterm.recv() => info!(reason = "SIGTERM", "shutdown signal received"),
            _ = sigint.recv() => info!(reason = "SIGINT", "shutdown signal received"),
            _ = shutdown.notified() => info!(
                reason = "supervisor_escalation",
                "supervisor signaled shutdown (escalation cascade)"
            ),
        }
    }

    // ── Graceful shutdown per SPEC §17 + §18.4 ────────────────────────
    let grace_s = titan_core::constants::DAEMON_SHUTDOWN_GRACE_S;
    let supervisor_terminate = unified_spirit_supervisor
        .as_ref()
        .map(|s| s.terminate_requested())
        .unwrap_or(false);
    info!(
        event = "SHUTDOWN_BEGIN",
        grace_s, supervisor_terminate, "substrate shutdown begin"
    );
    // Notify all spawned tasks (fastbus consumer, watch tasks, future tick loop)
    if let Some(sup) = unified_spirit_supervisor.as_ref() {
        sup.mark_shutdown_active();
    }
    shutdown.notify_waiters();

    children.sigterm_all();
    // Take ownership of the child OUT of the mutex BEFORE the await — clippy
    // forbids holding a MutexGuard across `.await` (per `await_holding_lock`).
    let unified_spirit_child = children.unified_spirit.lock().take();
    if let Some(mut child) = unified_spirit_child {
        let _ = tokio::time::timeout(Duration::from_secs_f64(grace_s), child.wait()).await;
    }
    if let Some(handle) = fastbus_handle {
        let _ = tokio::time::timeout(Duration::from_secs_f64(grace_s), handle).await;
    }
    if let Some(handle) = body_cycle_handle {
        let _ = tokio::time::timeout(Duration::from_secs_f64(grace_s), handle).await;
    }
    let snap = fastbus_stats.snapshot();
    info!(
        circadian_received = snap.circadian_received,
        pi_received = snap.pi_received,
        decoded_ok = snap.decoded_ok,
        decode_errors = snap.decode_errors,
        "substrate fastbus consumer final stats"
    );

    info!(event = "SHUTDOWN_COMPLETE", "substrate shutdown complete");
    // Phase C C-S7 Gap B: exit 64 (SupervisorSelfTerminate range per
    // SPEC §15) when shutdown was triggered by escalation → kernel
    // cascades a fresh substrate.
    if supervisor_terminate {
        SubstrateExitCode::SupervisorSelfTerminate.to_exit_code()
    } else {
        SubstrateExitCode::Clean.to_exit_code()
    }
}

/// Initialize tracing-subscriber with JSON formatter per SPEC §16. Returns
/// a guard for the lifetime of `main()`.
fn init_logging(level: tracing::Level, _titan_id: &str) -> impl Drop {
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    // Allow RUST_LOG override; fall back to passed level
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level.as_str().to_lowercase()));

    let layer = fmt::layer()
        .json()
        .with_target(false)
        .with_thread_ids(false)
        .with_current_span(true)
        .with_span_list(false);

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(layer)
        .try_init();

    DropOnExit
}

struct DropOnExit;
impl Drop for DropOnExit {
    fn drop(&mut self) {
        tracing::debug!("trinity-substrate logger shutting down");
    }
}
