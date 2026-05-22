//! fastbus_publisher — Kernel-side fastbus producer task.
//!
//! Per SPEC §9.A kernel row (fast bus produces circadian + π events to
//! substrate) + PLAN_microkernel_phase_c_s3_substrate.md §9 + chunk C3-6.
//!
//! C-S3 chunk C3-6 ships the kernel-side fastbus producer:
//!   - Attaches `/dev/shm/titan_<id>/fastbus.bin` (kernel pre-allocated in
//!     C-S2 boot step B3; we initialize the ring header on first attach)
//!   - Spawns a tokio task that emits 1 Hz `Circadian` ticks + ~3 Hz
//!     `PiHeartbeat` ticks via the SPSC ring
//!   - Substrate-side consumer (titan-trinity-rs::fastbus_consumer in C3-6)
//!     reads + logs received events
//!
//! In C-S3 the substrate doesn't yet act on these events — it just confirms
//! reception. C-S5/C-S6 daemons read circadian + π via fastbus to modulate
//! Schumann phase per Preamble G13 + SPEC §10.H.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use titan_fastbus::{FastbusError, Message, MsgType, Ring};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Cadence for circadian publication: 1 Hz per SPEC §10.H.
const CIRCADIAN_CADENCE_MS: u64 = 1000;
/// Cadence for π-heartbeat publication: target ~3 Hz (333 ms ≈ kernel π rate
/// per SPEC §10.H "~3 Hz").
const PI_HEARTBEAT_CADENCE_MS: u64 = 333;

/// Errors during fastbus publisher setup.
#[derive(Debug, Error)]
pub enum FastbusPublisherError {
    /// Could not attach to the kernel-pre-allocated `fastbus.bin`.
    #[error("fastbus attach failed: {0}")]
    Attach(#[from] FastbusError),
}

/// Spawn the kernel-side fastbus publisher task.
///
/// Owns the [`Ring`] for the lifetime of the task. Emits one `Circadian` tick
/// every `CIRCADIAN_CADENCE_MS` and one `PiHeartbeat` tick every
/// `PI_HEARTBEAT_CADENCE_MS`. Stops cleanly on `shutdown` notification.
///
/// Returns the [`JoinHandle`] so kernel shutdown can `await` clean exit.
pub fn spawn_kernel_fastbus_publisher(
    fastbus_path: PathBuf,
    shutdown: Arc<Notify>,
) -> Result<JoinHandle<()>, FastbusPublisherError> {
    // Attach happens synchronously so attach errors propagate to kernel boot
    // (substrate hasn't started yet — kernel is the first attacher).
    let ring = Ring::attach(&fastbus_path)?;
    let pid = std::process::id() as u64;
    info!(
        path = ?fastbus_path,
        pid,
        "kernel: fastbus producer attached + ring initialized"
    );

    let handle = tokio::spawn(async move {
        let mut ring = ring;
        let mut producer = ring.producer_only();
        let circadian_epoch = AtomicU64::new(0);
        let pi_epoch = AtomicU64::new(0);

        let mut circadian_tick = tokio::time::interval(Duration::from_millis(CIRCADIAN_CADENCE_MS));
        circadian_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut pi_tick = tokio::time::interval(Duration::from_millis(PI_HEARTBEAT_CADENCE_MS));
        pi_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            let shutdown_fut = shutdown.notified();
            tokio::select! {
                _ = circadian_tick.tick() => {
                    let epoch = circadian_epoch.fetch_add(1, Ordering::Release);
                    let ts_ns = wall_now_ns();
                    let msg = Message::new(MsgType::Circadian, ts_ns, epoch, pid);
                    if let Err(e) = producer.publish(&msg.encode()) {
                        // QueueFull = slow consumer (substrate). Rate-limited warn.
                        if epoch.is_multiple_of(10) {
                            warn!(err = ?e, epoch, "kernel: fastbus circadian publish failed (slow substrate?)");
                        }
                    }
                }
                _ = pi_tick.tick() => {
                    let epoch = pi_epoch.fetch_add(1, Ordering::Release);
                    let ts_ns = wall_now_ns();
                    let msg = Message::new(MsgType::PiHeartbeat, ts_ns, epoch, pid);
                    if let Err(e) = producer.publish(&msg.encode()) {
                        if epoch.is_multiple_of(30) {
                            warn!(err = ?e, epoch, "kernel: fastbus pi-heartbeat publish failed (slow substrate?)");
                        }
                    }
                }
                _ = shutdown_fut => {
                    debug!("kernel: fastbus publisher shutdown requested");
                    break;
                }
            }
        }
        info!(
            circadian_emitted = circadian_epoch.load(Ordering::Acquire),
            pi_emitted = pi_epoch.load(Ordering::Acquire),
            "kernel: fastbus publisher exiting"
        );
    });

    Ok(handle)
}

/// Wall-clock nanoseconds since UNIX epoch — matches Python `time.time_ns()`
/// semantics. Used for `Message::ts_ns` field.
pub fn wall_now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    use titan_fastbus::FASTBUS_FILE_TOTAL_BYTES;

    fn make_fastbus_file(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fastbus.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0u8; FASTBUS_FILE_TOTAL_BYTES]).unwrap();
        path
    }

    #[test]
    fn wall_now_ns_returns_nonzero() {
        let now = wall_now_ns();
        assert!(now > 1_700_000_000_000_000_000u64);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn publisher_spawn_attach_succeeds_with_zeroed_file() {
        let tmp = tempdir().unwrap();
        let path = make_fastbus_file(tmp.path());
        let shutdown = Arc::new(Notify::new());
        let handle =
            spawn_kernel_fastbus_publisher(path, shutdown.clone()).expect("attach + spawn");
        // Let publisher run briefly, then shutdown
        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown.notify_waiters();
        // Bounded await — task exits within reasonable time after shutdown
        let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "publisher should exit on shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn publisher_emits_at_least_one_event_in_first_second() {
        let tmp = tempdir().unwrap();
        let path = make_fastbus_file(tmp.path());
        let shutdown = Arc::new(Notify::new());
        // Spawn publisher; let it run for a bit; verify a consumer-side
        // ring read picks up at least one message.
        let path_clone = path.clone();
        let shutdown_clone = shutdown.clone();
        let _handle = spawn_kernel_fastbus_publisher(path, shutdown_clone).expect("attach + spawn");
        // Wait long enough for at least one circadian (1s) OR pi (333ms) tick
        tokio::time::sleep(Duration::from_millis(500)).await;
        // Consumer-side attach (separate Ring instance over same file)
        let mut consumer_ring = Ring::attach(&path_clone).expect("attach consumer");
        let mut consumer = consumer_ring.consumer_only();
        let mut received = 0usize;
        for _ in 0..10 {
            if consumer.recv_and_commit().is_some() {
                received += 1;
            } else {
                break;
            }
        }
        shutdown.notify_waiters();
        assert!(received > 0, "expected at least one fastbus event");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn publisher_attach_fails_for_missing_file() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("nonexistent.bin");
        let shutdown = Arc::new(Notify::new());
        let result = spawn_kernel_fastbus_publisher(path, shutdown);
        assert!(result.is_err());
    }
}
