//! substrate_watch — Kernel-side substrate-child supervisor task.
//!
//! Per PLAN §11 substrate supervision contract + SPEC §11.0 + chunk C3-7.
//!
//! C-S3 chunk C3-7 ships the OBSERVABILITY half of substrate supervision:
//!   - Spawns a tokio task that takes ownership of the substrate `Child`
//!   - Awaits `Child::wait()` so SIGCHLD is observed
//!   - On unexpected exit (NOT during shutdown): logs `SUPERVISION_CHILD_DOWN`
//!     event to substrate's structured logs + bumps a respawn-counter for
//!     diagnostics
//!
//! C-S3 does NOT yet implement automatic respawn — that requires extending
//! `SpawnedChildren` with a respawn loop + max_restarts handshake. The full
//! one_for_one supervisor lands in C-S4 when substrate is more stable + has
//! a real bus client. C-S3 logs the exit so OBS-c-s3-substrate-boots-clean
//! has data to gate on (kill substrate manually → kernel logs the event →
//! gate evidence captured).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::process::Child;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Counters surfaced to telemetry + tests.
#[derive(Debug, Default)]
pub struct SubstrateWatchStats {
    /// Number of times substrate has exited unexpectedly (not during shutdown).
    pub unexpected_exits: AtomicU32,
    /// Number of times substrate exited cleanly (during shutdown).
    pub clean_exits: AtomicU32,
}

impl SubstrateWatchStats {
    /// Snapshot of current counters.
    pub fn snapshot(&self) -> SubstrateWatchSnapshot {
        SubstrateWatchSnapshot {
            unexpected_exits: self.unexpected_exits.load(Ordering::Acquire),
            clean_exits: self.clean_exits.load(Ordering::Acquire),
        }
    }
}

/// Plain-old-data snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubstrateWatchSnapshot {
    /// Unexpected-exit count.
    pub unexpected_exits: u32,
    /// Clean-exit count.
    pub clean_exits: u32,
}

/// Spawn a watch task that takes ownership of the substrate child + awaits
/// its exit. On exit, logs the appropriate event + bumps stats.
///
/// The task uses a shared `child_slot` (a `Mutex<Option<Child>>`) so callers
/// can both pass the child INTO the watch task AND retain the ability to
/// SIGTERM it on shutdown via a separate code path. The watch task takes
/// the Child OUT of the slot at startup; subsequent SIGTERM-by-shutdown
/// code finds None and is a no-op (the watch task itself observes the
/// SIGTERM-induced exit).
pub fn spawn_substrate_watch(
    child_slot: Arc<Mutex<Option<Child>>>,
    shutdown: Arc<Notify>,
    stats: Arc<SubstrateWatchStats>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Take ownership of the Child for the lifetime of this task.
        let mut child = match child_slot.lock().take() {
            Some(c) => c,
            None => {
                warn!("substrate_watch: no child to supervise (already taken)");
                return;
            }
        };
        let pid = child.id();
        info!(pid, "substrate_watch: supervising substrate child");

        let shutdown_fut = shutdown.notified();
        tokio::select! {
            wait_result = child.wait() => {
                // Substrate exited on its own — unexpected.
                let exit_code = wait_result.ok().and_then(|s| s.code());
                stats.unexpected_exits.fetch_add(1, Ordering::Release);
                let n = stats.unexpected_exits.load(Ordering::Acquire);
                warn!(
                    event = "SUPERVISION_CHILD_DOWN",
                    child = "trinity-substrate",
                    supervisor = "titan-kernel-rs",
                    pid,
                    exit_code,
                    unexpected_exit_count = n,
                    "substrate exited unexpectedly — automatic respawn deferred to C-S4"
                );
            }
            _ = shutdown_fut => {
                // Shutdown requested — kernel will SIGTERM all children via
                // SpawnedChildren::sigterm_all. Wait for child exit gracefully.
                info!(pid, "substrate_watch: shutdown signaled, awaiting child exit");
                let exit_code = child.wait().await.ok().and_then(|s| s.code());
                stats.clean_exits.fetch_add(1, Ordering::Release);
                info!(
                    event = "SUPERVISION_CHILD_CLEAN_EXIT",
                    child = "trinity-substrate",
                    pid,
                    exit_code,
                    "substrate exited cleanly during kernel shutdown"
                );
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::process::Command;

    /// Test helper: spawn a short-lived dummy child via `sleep`.
    fn spawn_test_child(sleep_secs: &str) -> Child {
        let mut cmd = Command::new("sleep");
        cmd.arg(sleep_secs);
        cmd.kill_on_drop(false);
        cmd.spawn().expect("spawn test sleep child")
    }

    #[test]
    fn stats_default_zero() {
        let s = SubstrateWatchStats::default();
        let snap = s.snapshot();
        assert_eq!(snap.unexpected_exits, 0);
        assert_eq!(snap.clean_exits, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn watch_task_no_child_logs_warn_and_exits() {
        let slot: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
        let shutdown = Arc::new(Notify::new());
        let stats = Arc::new(SubstrateWatchStats::default());
        let handle = spawn_substrate_watch(slot, shutdown, stats.clone());
        let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok());
        // No exit was observed (no child) — counters stay zero
        assert_eq!(stats.snapshot().unexpected_exits, 0);
        assert_eq!(stats.snapshot().clean_exits, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn watch_task_observes_unexpected_child_exit() {
        // Child exits on its own (sleep 0 → immediate exit) — kernel sees this
        // as an unexpected exit since shutdown was never signaled.
        let child = spawn_test_child("0");
        let slot = Arc::new(Mutex::new(Some(child)));
        let shutdown = Arc::new(Notify::new());
        let stats = Arc::new(SubstrateWatchStats::default());
        let handle = spawn_substrate_watch(slot, shutdown, stats.clone());
        // Wait for the watch task to observe the child's quick exit
        let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(
            result.is_ok(),
            "watch task should exit after observing child exit"
        );
        let snap = stats.snapshot();
        assert_eq!(snap.unexpected_exits, 1, "snap = {snap:?}");
        assert_eq!(snap.clean_exits, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn watch_task_observes_clean_exit_during_shutdown() {
        // Long-lived child (60s); shutdown signaled → watch task SIGTERMs +
        // counts as clean exit.
        let child = spawn_test_child("60");
        let pid = child.id();
        let slot = Arc::new(Mutex::new(Some(child)));
        let shutdown = Arc::new(Notify::new());
        let stats = Arc::new(SubstrateWatchStats::default());
        let handle = spawn_substrate_watch(slot, shutdown.clone(), stats.clone());

        // Give it a moment to start watching
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Signal shutdown
        shutdown.notify_waiters();
        // Kill the child externally (simulating SpawnedChildren::sigterm_all)
        if let Some(p) = pid {
            let _ = std::process::Command::new("kill")
                .args(["-TERM", &p.to_string()])
                .status();
        }
        let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(result.is_ok());
        let snap = stats.snapshot();
        assert_eq!(snap.clean_exits, 1, "snap = {snap:?}");
        assert_eq!(snap.unexpected_exits, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn watch_task_distinguishes_unexpected_from_clean_via_shutdown_signal() {
        // Two separate scenarios on independent stats — verifies the
        // discriminator (shutdown signal vs not) is the right axis.
        let stats_unexpected = Arc::new(SubstrateWatchStats::default());
        let stats_clean = Arc::new(SubstrateWatchStats::default());

        // Scenario A: child dies, no shutdown signal
        let slot_a = Arc::new(Mutex::new(Some(spawn_test_child("0"))));
        let shutdown_a = Arc::new(Notify::new());
        let h_a = spawn_substrate_watch(slot_a, shutdown_a, stats_unexpected.clone());
        let _ = tokio::time::timeout(Duration::from_secs(5), h_a).await;

        // Scenario B: long-running child, shutdown signaled
        let child_b = spawn_test_child("30");
        let pid_b = child_b.id();
        let slot_b = Arc::new(Mutex::new(Some(child_b)));
        let shutdown_b = Arc::new(Notify::new());
        let h_b = spawn_substrate_watch(slot_b, shutdown_b.clone(), stats_clean.clone());
        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown_b.notify_waiters();
        if let Some(p) = pid_b {
            let _ = std::process::Command::new("kill")
                .args(["-TERM", &p.to_string()])
                .status();
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), h_b).await;

        assert_eq!(stats_unexpected.snapshot().unexpected_exits, 1);
        assert_eq!(stats_unexpected.snapshot().clean_exits, 0);
        assert_eq!(stats_clean.snapshot().clean_exits, 1);
        assert_eq!(stats_clean.snapshot().unexpected_exits, 0);
    }
}
