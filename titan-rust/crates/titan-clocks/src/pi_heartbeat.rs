//! pi_heartbeat — ~3 Hz consciousness pulse + `KERNEL_EPOCH_TICK` publisher.
//!
//! Per SPEC §10.H + §7.1 + §8.1:
//! - Tick cadence: `KERNEL_PI_HEARTBEAT_INTERVAL_S = 0.333…` (≈3 Hz)
//! - Slot `pi_heartbeat.bin` = 24B header + 12B payload (`f32 phase | u64 pulse_count`)
//! - Slot `epoch_counter.bin` = 24B header + 8B payload (`u64 epoch_id LE`)
//! - Bus message: `KERNEL_EPOCH_TICK` (P0, never drop) per §8.1
//!   payload: `{epoch_id: u64, ts: f64, dt_s: f64}`
//!
//! Publishing `KERNEL_EPOCH_TICK` is delegated to an injected
//! [`EpochTickPublisher`] trait — keeps the loop testable without a real
//! broker (the kernel binary in C2-6 wires it to titan-bus).

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use tokio::sync::Mutex as AsyncMutex;
use tokio::time::interval;
use tracing::{debug, warn};

use titan_core::constants::KERNEL_PI_HEARTBEAT_INTERVAL_S;
use titan_state::Slot;

/// Snapshot of one π-heartbeat tick — what the publisher receives.
#[derive(Debug, Clone, Copy)]
pub struct PiTickEvent {
    /// Monotonically-increasing epoch ID (matches `epoch_counter.bin`).
    pub epoch_id: u64,
    /// Pulse count (matches `pi_heartbeat.bin` payload).
    pub pulse_count: u64,
    /// Phase in [0.0..1.0).
    pub phase: f32,
    /// Wall-clock timestamp (seconds since UNIX epoch, fractional).
    pub ts: f64,
    /// Time since previous tick in seconds.
    pub dt_s: f64,
}

/// Trait abstracting how `KERNEL_EPOCH_TICK` is published.
///
/// In tests we use a mock that records events. The kernel binary (C2-6)
/// implements this against `titan-bus::BusBroker`.
pub trait EpochTickPublisher: Send + Sync {
    /// Publish one tick event. Returning an error logs a warning but does
    /// NOT halt the heartbeat loop (heartbeat is best-effort under bus
    /// pressure; subscribers can resync from `epoch_counter.bin`).
    fn publish(&self, event: &PiTickEvent) -> Result<(), String>;
}

/// No-op publisher — for tests + boot-time scenarios when no broker is wired.
pub struct NoopPublisher;
impl EpochTickPublisher for NoopPublisher {
    fn publish(&self, _: &PiTickEvent) -> Result<(), String> {
        Ok(())
    }
}

/// Stateful π-heartbeat. Each `tick()` increments pulse_count + epoch_id.
///
/// Phase advances by `KERNEL_PI_HEARTBEAT_INTERVAL_S` per tick (mod 1.0)
/// — this is a non-meaningful local phase variable kept for §7.1
/// compatibility; consciousness-level meaning is in epoch_id.
pub struct PiHeartbeat {
    pulse_count: u64,
    epoch_id: u64,
    phase: f32,
    last_tick: Option<Instant>,
}

impl PiHeartbeat {
    /// Construct fresh (phase=0, pulses=0, epoch=0).
    pub fn new() -> Self {
        Self {
            pulse_count: 0,
            epoch_id: 0,
            phase: 0.0,
            last_tick: None,
        }
    }

    /// Construct from a previously-snapshotted state (boot resume from L0).
    pub fn with_state(pulse_count: u64, epoch_id: u64, phase: f32) -> Self {
        Self {
            pulse_count,
            epoch_id,
            phase: phase.rem_euclid(1.0),
            last_tick: None,
        }
    }

    /// Current pulse count.
    pub fn pulse_count(&self) -> u64 {
        self.pulse_count
    }

    /// Current epoch ID.
    pub fn epoch_id(&self) -> u64 {
        self.epoch_id
    }

    /// Current phase.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Advance one tick. Returns the event payload for publishing.
    pub fn tick(&mut self) -> PiTickEvent {
        self.pulse_count = self.pulse_count.wrapping_add(1);
        self.epoch_id = self.epoch_id.wrapping_add(1);
        let advance = KERNEL_PI_HEARTBEAT_INTERVAL_S as f32;
        let next_phase = self.phase + advance;
        self.phase = if next_phase >= 1.0 {
            next_phase - 1.0
        } else {
            next_phase
        };

        let now = Instant::now();
        let dt_s = self
            .last_tick
            .map(|prev| now.saturating_duration_since(prev).as_secs_f64())
            .unwrap_or(KERNEL_PI_HEARTBEAT_INTERVAL_S);
        self.last_tick = Some(now);

        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        PiTickEvent {
            epoch_id: self.epoch_id,
            pulse_count: self.pulse_count,
            phase: self.phase,
            ts,
            dt_s,
        }
    }

    /// Encode `pi_heartbeat.bin` payload (12 bytes: f32 phase + u64 pulse_count).
    pub fn encode_pi_payload(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&self.phase.to_le_bytes());
        buf[4..12].copy_from_slice(&self.pulse_count.to_le_bytes());
        buf
    }

    /// Encode `epoch_counter.bin` payload (8 bytes: u64 LE).
    pub fn encode_epoch_payload(&self) -> [u8; 8] {
        self.epoch_id.to_le_bytes()
    }
}

impl Default for PiHeartbeat {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the π-heartbeat loop until shutdown.
///
/// Per tick:
/// 1. Advance internal state
/// 2. Write `pi_heartbeat.bin` (SeqLock)
/// 3. Write `epoch_counter.bin` (SeqLock)
/// 4. Publish `KERNEL_EPOCH_TICK` via `publisher`
pub async fn run_pi_heartbeat_loop(
    pi_slot: Arc<AsyncMutex<Slot>>,
    epoch_slot: Arc<AsyncMutex<Slot>>,
    publisher: Arc<dyn EpochTickPublisher>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    let mut hb = PiHeartbeat::new();
    let interval_dur = Duration::from_secs_f64(KERNEL_PI_HEARTBEAT_INTERVAL_S);
    let mut tick_interval = interval(interval_dur);
    tick_interval.tick().await; // consume immediate first tick

    loop {
        tokio::select! {
            _ = tick_interval.tick() => {
                let event = hb.tick();
                let pi_payload = hb.encode_pi_payload();
                let epoch_payload = hb.encode_epoch_payload();

                {
                    let mut s = pi_slot.lock().await;
                    if let Err(e) = s.write(&pi_payload) {
                        warn!(err = ?e, "pi-heartbeat: pi slot write failed");
                    }
                }
                {
                    let mut s = epoch_slot.lock().await;
                    if let Err(e) = s.write(&epoch_payload) {
                        warn!(err = ?e, "pi-heartbeat: epoch slot write failed");
                    }
                }

                if let Err(e) = publisher.publish(&event) {
                    warn!(err = %e, "pi-heartbeat: KERNEL_EPOCH_TICK publish failed");
                }
            }
            _ = shutdown.notified() => {
                debug!("pi-heartbeat loop: shutdown received");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex as SyncMutex;

    /// Recording publisher for tests.
    struct RecordingPublisher {
        events: SyncMutex<Vec<PiTickEvent>>,
    }
    impl RecordingPublisher {
        fn new() -> Self {
            Self {
                events: SyncMutex::new(Vec::new()),
            }
        }
        fn count(&self) -> usize {
            self.events.lock().len()
        }
        fn last(&self) -> Option<PiTickEvent> {
            self.events.lock().last().copied()
        }
    }
    impl EpochTickPublisher for RecordingPublisher {
        fn publish(&self, event: &PiTickEvent) -> Result<(), String> {
            self.events.lock().push(*event);
            Ok(())
        }
    }

    #[test]
    fn new_heartbeat_zero_state() {
        let hb = PiHeartbeat::new();
        assert_eq!(hb.pulse_count(), 0);
        assert_eq!(hb.epoch_id(), 0);
        assert_eq!(hb.phase(), 0.0);
    }

    #[test]
    fn tick_advances_pulse_and_epoch_by_1() {
        let mut hb = PiHeartbeat::new();
        let event = hb.tick();
        assert_eq!(event.pulse_count, 1);
        assert_eq!(event.epoch_id, 1);
        assert_eq!(hb.pulse_count(), 1);
        assert_eq!(hb.epoch_id(), 1);
    }

    #[test]
    fn ten_ticks_yield_pulse_10_epoch_10() {
        let mut hb = PiHeartbeat::new();
        for i in 1..=10 {
            let event = hb.tick();
            assert_eq!(event.pulse_count, i as u64);
            assert_eq!(event.epoch_id, i as u64);
        }
    }

    #[test]
    fn dt_s_first_tick_uses_interval_default() {
        let mut hb = PiHeartbeat::new();
        let event = hb.tick();
        // First tick: no previous → dt = KERNEL_PI_HEARTBEAT_INTERVAL_S
        assert!((event.dt_s - KERNEL_PI_HEARTBEAT_INTERVAL_S).abs() < 1e-6);
    }

    #[test]
    fn dt_s_subsequent_tick_reflects_elapsed() {
        let mut hb = PiHeartbeat::new();
        hb.tick();
        std::thread::sleep(Duration::from_millis(20));
        let event2 = hb.tick();
        // Should be ~0.02 s
        assert!(event2.dt_s >= 0.015 && event2.dt_s <= 0.5);
    }

    #[test]
    fn encode_pi_payload_layout_matches_spec_7_1() {
        // SPEC §7.1: f32 phase + u64 pulse_count = 12 bytes
        let mut hb = PiHeartbeat::new();
        hb.tick();
        let payload = hb.encode_pi_payload();
        assert_eq!(payload.len(), 12);
        assert_eq!(&payload[0..4], &hb.phase().to_le_bytes());
        assert_eq!(&payload[4..12], &hb.pulse_count().to_le_bytes());
    }

    #[test]
    fn encode_epoch_payload_layout_matches_spec_7_1() {
        // SPEC §7.1: u64 LE = 8 bytes
        let mut hb = PiHeartbeat::new();
        hb.tick();
        let payload = hb.encode_epoch_payload();
        assert_eq!(payload.len(), 8);
        assert_eq!(payload, hb.epoch_id().to_le_bytes());
    }

    #[test]
    fn with_state_resumes_from_snapshot() {
        let hb = PiHeartbeat::with_state(1000, 1500, 0.5);
        assert_eq!(hb.pulse_count(), 1000);
        assert_eq!(hb.epoch_id(), 1500);
        assert_eq!(hb.phase(), 0.5);
    }

    #[test]
    fn with_state_normalizes_phase() {
        let hb = PiHeartbeat::with_state(0, 0, 1.5);
        assert!((hb.phase() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn phase_wraps_at_one() {
        // Start near 1.0; ticks should wrap
        let mut hb = PiHeartbeat::with_state(0, 0, 0.99);
        for _ in 0..10 {
            hb.tick();
        }
        // Phase should still be in [0,1)
        assert!(hb.phase() >= 0.0 && hb.phase() < 1.0);
    }

    #[test]
    fn noop_publisher_succeeds() {
        let pub_ = NoopPublisher;
        let event = PiTickEvent {
            epoch_id: 1,
            pulse_count: 1,
            phase: 0.0,
            ts: 0.0,
            dt_s: 0.333,
        };
        assert!(pub_.publish(&event).is_ok());
    }

    #[test]
    fn recording_publisher_captures_event() {
        let pub_ = RecordingPublisher::new();
        let event = PiTickEvent {
            epoch_id: 1,
            pulse_count: 1,
            phase: 0.0,
            ts: 0.0,
            dt_s: 0.333,
        };
        pub_.publish(&event).unwrap();
        assert_eq!(pub_.count(), 1);
        let captured = pub_.last().unwrap();
        assert_eq!(captured.epoch_id, 1);
    }

    #[tokio::test]
    async fn loop_advances_pulse_count_in_slot() {
        use titan_state::Slot;

        let dir = tempfile::tempdir().unwrap();
        let pi_path = dir.path().join("pi.bin");
        let epoch_path = dir.path().join("epoch.bin");
        let pi_slot = Arc::new(AsyncMutex::new(Slot::create(&pi_path, 1, 12).unwrap()));
        let epoch_slot = Arc::new(AsyncMutex::new(Slot::create(&epoch_path, 1, 8).unwrap()));
        let publisher = Arc::new(RecordingPublisher::new());
        let publisher_dyn: Arc<dyn EpochTickPublisher> = publisher.clone();
        let shutdown = Arc::new(tokio::sync::Notify::new());

        let pi_clone = pi_slot.clone();
        let epoch_clone = epoch_slot.clone();
        let shutdown_clone = shutdown.clone();
        let task = tokio::spawn(async move {
            run_pi_heartbeat_loop(pi_clone, epoch_clone, publisher_dyn, shutdown_clone).await;
        });

        // Real-time sleep ~1.1s — allows >2 ticks at ~3 Hz
        tokio::time::sleep(Duration::from_secs_f64(1.1)).await;

        shutdown.notify_waiters();
        let _ = tokio::time::timeout(Duration::from_secs(1), task).await;

        let count = publisher.count();
        assert!(count >= 2, "expected at least 2 ticks in 1.1s, got {count}");

        // Verify slots got updated
        let pi_payload = pi_slot.lock().await.read().unwrap();
        let pulse_from_slot = u64::from_le_bytes(pi_payload[4..12].try_into().unwrap());
        assert!(
            pulse_from_slot >= 2,
            "pulse_count slot should reflect ticks, got {pulse_from_slot}"
        );

        let epoch_payload = epoch_slot.lock().await.read().unwrap();
        let epoch_from_slot = u64::from_le_bytes(epoch_payload[..8].try_into().unwrap());
        assert!(
            epoch_from_slot >= 2,
            "epoch_id slot should reflect ticks, got {epoch_from_slot}"
        );
    }

    #[tokio::test]
    async fn shutdown_signal_stops_loop() {
        use titan_state::Slot;
        let dir = tempfile::tempdir().unwrap();
        let pi_slot = Arc::new(AsyncMutex::new(
            Slot::create(dir.path().join("pi.bin"), 1, 12).unwrap(),
        ));
        let epoch_slot = Arc::new(AsyncMutex::new(
            Slot::create(dir.path().join("epoch.bin"), 1, 8).unwrap(),
        ));
        let publisher: Arc<dyn EpochTickPublisher> = Arc::new(NoopPublisher);
        let shutdown = Arc::new(tokio::sync::Notify::new());

        let pi_clone = pi_slot.clone();
        let epoch_clone = epoch_slot.clone();
        let shutdown_clone = shutdown.clone();
        let task = tokio::spawn(async move {
            run_pi_heartbeat_loop(pi_clone, epoch_clone, publisher, shutdown_clone).await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown.notify_waiters();
        let result = tokio::time::timeout(Duration::from_secs(2), task).await;
        assert!(result.is_ok(), "loop should exit within 2s after shutdown");
    }
}
