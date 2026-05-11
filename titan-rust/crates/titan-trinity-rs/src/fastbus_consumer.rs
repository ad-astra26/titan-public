//! fastbus_consumer — Substrate-side fastbus consumer task.
//!
//! Per SPEC §9.A trinity-rs row "Fast bus: produces Schumann epoch events
//! to kernel; consumes circadian + π events from kernel" + PLAN §9 + chunk
//! C3-6.
//!
//! C-S3 chunk C3-6 ships:
//!   - `attach_fastbus()` — substrate boot step S6 per PLAN §7.1; opens
//!     `/dev/shm/titan_<id>/fastbus.bin` (kernel pre-creates the file in
//!     C-S2 step B3 + initialized the ring header in C3-6 step B7.5)
//!   - `spawn_fastbus_consumer_loop()` — consumes incoming kernel events
//!     (Circadian, PiHeartbeat) + logs them. C-S5/C-S6 daemons will use the
//!     same Ring instance to *also* publish SchumannEpoch events back to
//!     kernel via `Producer::publish`.
//!
//! In C-S3 the consumer just confirms reception via tracing logs. C-S4+
//! adds Schumann phase modulation logic that uses these events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use titan_fastbus::{FastbusError, Message, MsgType, Ring};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Substrate-side polling cadence — how often the consumer task wakes to
/// drain the ring. Matches the spirit Schumann tick (≈ 14.2 ms / 70.47 Hz)
/// so we don't lag the producer's max emission rate (kernel publishes at
/// most once per 333 ms = π-heartbeat).
const CONSUMER_POLL_INTERVAL_MS: u64 = 14;

/// Errors during substrate fastbus consumer setup.
#[derive(Debug, Error)]
pub enum FastbusConsumerError {
    /// Could not attach to fastbus.bin (file missing, size mismatch,
    /// magic invalid, etc. — see [`FastbusError`]).
    #[error("fastbus attach failed: {0}")]
    Attach(#[from] FastbusError),
}

/// Counters surfaced for telemetry (Phase D /v4/admin/substrate-state).
#[derive(Debug, Default)]
pub struct ConsumerStats {
    /// Total `Circadian` events received from kernel.
    pub circadian_received: AtomicU64,
    /// Total `PiHeartbeat` events received.
    pub pi_received: AtomicU64,
    /// Slots successfully decoded.
    pub decoded_ok: AtomicU64,
    /// Slots whose msg_type byte didn't match a registered variant.
    pub decode_errors: AtomicU64,
}

impl ConsumerStats {
    /// Read snapshot — useful for tests + diagnostics.
    pub fn snapshot(&self) -> ConsumerStatsSnapshot {
        ConsumerStatsSnapshot {
            circadian_received: self.circadian_received.load(Ordering::Acquire),
            pi_received: self.pi_received.load(Ordering::Acquire),
            decoded_ok: self.decoded_ok.load(Ordering::Acquire),
            decode_errors: self.decode_errors.load(Ordering::Acquire),
        }
    }
}

/// Plain-old-data snapshot of [`ConsumerStats`] — no atomics, easy to compare.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConsumerStatsSnapshot {
    /// Circadian event count.
    pub circadian_received: u64,
    /// π-heartbeat event count.
    pub pi_received: u64,
    /// Successful decodes.
    pub decoded_ok: u64,
    /// Decode errors (unrecognized msg_type).
    pub decode_errors: u64,
}

/// Spawn the substrate-side fastbus consumer task.
///
/// Owns the [`Ring`] for the lifetime of the task. Polls the consumer side
/// at `CONSUMER_POLL_INTERVAL_MS` and decodes any pending messages, updating
/// stats counters + logging events at trace level. Stops on `shutdown`.
pub fn spawn_substrate_fastbus_consumer(
    fastbus_path: PathBuf,
    shutdown: Arc<Notify>,
    stats: Arc<ConsumerStats>,
) -> Result<JoinHandle<()>, FastbusConsumerError> {
    let ring = Ring::attach(&fastbus_path)?;
    info!(
        path = ?fastbus_path,
        "substrate: fastbus consumer attached"
    );

    let handle = tokio::spawn(async move {
        let mut ring = ring;
        let mut consumer = ring.consumer_only();
        let mut poll_interval =
            tokio::time::interval(Duration::from_millis(CONSUMER_POLL_INTERVAL_MS));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            let shutdown_fut = shutdown.notified();
            tokio::select! {
                _ = poll_interval.tick() => {
                    // Drain everything currently available — bounded by ring
                    // capacity (1024) per poll cycle.
                    while let Some(slot) = consumer.recv_and_commit() {
                        match Message::decode(&slot) {
                            Ok(msg) => {
                                stats.decoded_ok.fetch_add(1, Ordering::Release);
                                handle_message(&msg, &stats);
                            }
                            Err(e) => {
                                stats.decode_errors.fetch_add(1, Ordering::Release);
                                let n = stats.decode_errors.load(Ordering::Acquire);
                                if n % 100 == 1 {
                                    warn!(err = ?e, "substrate: fastbus decode error (rate-limited)");
                                }
                            }
                        }
                    }
                }
                _ = shutdown_fut => {
                    debug!("substrate: fastbus consumer shutdown requested");
                    break;
                }
            }
        }
        let snap = stats.snapshot();
        info!(
            circadian = snap.circadian_received,
            pi = snap.pi_received,
            decoded = snap.decoded_ok,
            errors = snap.decode_errors,
            "substrate: fastbus consumer exiting"
        );
    });

    Ok(handle)
}

/// Per-message dispatch — called from the consumer loop. C-S3 just bumps
/// counters + logs at trace level. C-S4+ may upgrade to wake a Schumann
/// phase-modulator or an L2 listener.
fn handle_message(msg: &Message, stats: &ConsumerStats) {
    match msg.msg_type {
        MsgType::Circadian => {
            stats.circadian_received.fetch_add(1, Ordering::Release);
            tracing::trace!(
                event = "FASTBUS_CIRCADIAN_RECEIVED",
                epoch = msg.epoch,
                ts_ns = msg.ts_ns,
                producer_pid = msg.producer_pid,
                "substrate: circadian event"
            );
        }
        MsgType::PiHeartbeat => {
            stats.pi_received.fetch_add(1, Ordering::Release);
            tracing::trace!(
                event = "FASTBUS_PI_HEARTBEAT_RECEIVED",
                epoch = msg.epoch,
                ts_ns = msg.ts_ns,
                producer_pid = msg.producer_pid,
                "substrate: pi-heartbeat event"
            );
        }
        MsgType::SchumannEpoch => {
            // Substrate doesn't consume its own SchumannEpoch publications —
            // those go kernel-ward. If we see one here it's a wiring bug.
            tracing::warn!(
                epoch = msg.epoch,
                "substrate: unexpected SchumannEpoch on consumer side (substrate→kernel direction)"
            );
        }
    }
}

/// Substrate-side helper: publish a SchumannEpoch event back to kernel.
/// Used by C-S5+ inner-spirit-rs daemon when it crosses a body-cycle boundary
/// (every 9 spirit ticks per Preamble G13 ratio).
///
/// Per SPEC §9.A trinity-rs "produces Schumann epoch events to kernel".
/// In C-S3 the substrate's main.rs doesn't yet drive this — daemons will
/// invoke it in C-S5. The function is here in C3-6 so the wiring contract
/// is testable.
pub fn build_schumann_epoch_message(epoch: u64) -> Message {
    let pid = std::process::id() as u64;
    let ts_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Message::new(MsgType::SchumannEpoch, ts_ns, epoch, pid)
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

    fn publish_to_ring(path: &std::path::Path, msg: &Message) {
        let mut ring = Ring::attach(path).expect("attach producer side");
        let mut producer = ring.producer_only();
        producer.publish(&msg.encode()).expect("publish");
    }

    #[test]
    fn build_schumann_epoch_message_uses_substrate_pid() {
        let m = build_schumann_epoch_message(7);
        assert_eq!(m.msg_type, MsgType::SchumannEpoch);
        assert_eq!(m.epoch, 7);
        assert_eq!(m.producer_pid, std::process::id() as u64);
        assert!(m.ts_ns > 0);
    }

    #[test]
    fn handle_message_circadian_increments_circadian_counter() {
        let stats = Arc::new(ConsumerStats::default());
        let m = Message::new(MsgType::Circadian, 100, 1, 999);
        handle_message(&m, &stats);
        let snap = stats.snapshot();
        assert_eq!(snap.circadian_received, 1);
        assert_eq!(snap.pi_received, 0);
    }

    #[test]
    fn handle_message_pi_increments_pi_counter() {
        let stats = Arc::new(ConsumerStats::default());
        let m = Message::new(MsgType::PiHeartbeat, 100, 5, 999);
        handle_message(&m, &stats);
        let snap = stats.snapshot();
        assert_eq!(snap.circadian_received, 0);
        assert_eq!(snap.pi_received, 1);
    }

    #[test]
    fn handle_message_schumann_epoch_does_not_increment_inbound_counters() {
        // Substrate seeing its own SchumannEpoch back is a wiring bug — log
        // warn, don't bump circadian/pi counters.
        let stats = Arc::new(ConsumerStats::default());
        let m = Message::new(MsgType::SchumannEpoch, 100, 1, 999);
        handle_message(&m, &stats);
        let snap = stats.snapshot();
        assert_eq!(snap.circadian_received, 0);
        assert_eq!(snap.pi_received, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consumer_attach_succeeds_and_drains_pre_published_messages() {
        let tmp = tempdir().unwrap();
        let path = make_fastbus_file(tmp.path());
        // Pre-publish: 2 circadian + 1 pi
        publish_to_ring(&path, &Message::new(MsgType::Circadian, 1, 1, 100));
        publish_to_ring(&path, &Message::new(MsgType::Circadian, 2, 2, 100));
        publish_to_ring(&path, &Message::new(MsgType::PiHeartbeat, 3, 1, 100));
        // Spawn consumer
        let stats = Arc::new(ConsumerStats::default());
        let shutdown = Arc::new(Notify::new());
        let _handle =
            spawn_substrate_fastbus_consumer(path, shutdown.clone(), stats.clone()).expect("spawn");
        // Let it drain
        tokio::time::sleep(Duration::from_millis(100)).await;
        let snap = stats.snapshot();
        assert_eq!(snap.circadian_received, 2);
        assert_eq!(snap.pi_received, 1);
        assert_eq!(snap.decoded_ok, 3);
        assert_eq!(snap.decode_errors, 0);
        shutdown.notify_waiters();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consumer_attach_fails_for_missing_file() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("nonexistent.bin");
        let stats = Arc::new(ConsumerStats::default());
        let shutdown = Arc::new(Notify::new());
        let result = spawn_substrate_fastbus_consumer(path, shutdown, stats);
        assert!(result.is_err());
    }
}
