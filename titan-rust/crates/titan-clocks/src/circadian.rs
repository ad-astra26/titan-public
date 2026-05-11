//! circadian — 1 Hz tick clock with 24h full-cycle period.
//!
//! Per SPEC §10.H + §7.1:
//! - Tick cadence: `KERNEL_CIRCADIAN_TICK_INTERVAL_S = 1.0` (1 Hz)
//! - Full cycle: `KERNEL_CIRCADIAN_PERIOD_S = 86400.0` (24h)
//! - Slot: `circadian.bin` = `24-byte SeqLock header + 12-byte payload`
//! - Payload layout (12 bytes LE):
//!   - `[0:4]   f32 phase`        (0.0..1.0 cyclic)
//!   - `[4:8]   f32 day_progress` (0.0..1.0 — same as phase for 24h period)
//!   - `[8:12]  f32 reserved`     (zero)

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex as AsyncMutex;
use tokio::time::interval;
use tracing::{debug, warn};

use titan_core::constants::{KERNEL_CIRCADIAN_PERIOD_S, KERNEL_CIRCADIAN_TICK_INTERVAL_S};
use titan_state::Slot;

/// Decoded circadian state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircadianState {
    /// Cyclic phase (0.0..1.0). Same as `day_progress` for 24h period.
    pub phase: f32,
    /// Day progress (0.0..1.0). Wraps at 1.0 → 0.0.
    pub day_progress: f32,
    /// Reserved field (zero at v0.1.2).
    pub reserved: f32,
}

impl CircadianState {
    /// Encode to 12-byte LE payload.
    pub fn encode(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&self.phase.to_le_bytes());
        buf[4..8].copy_from_slice(&self.day_progress.to_le_bytes());
        buf[8..12].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Decode from 12-byte LE payload.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        Some(Self {
            phase: f32::from_le_bytes(bytes[0..4].try_into().ok()?),
            day_progress: f32::from_le_bytes(bytes[4..8].try_into().ok()?),
            reserved: f32::from_le_bytes(bytes[8..12].try_into().ok()?),
        })
    }
}

/// Stateful circadian clock. Advances `day_progress` by
/// `KERNEL_CIRCADIAN_TICK_INTERVAL_S / KERNEL_CIRCADIAN_PERIOD_S` per tick.
pub struct CircadianClock {
    state: CircadianState,
    /// Phase advance per single tick (precomputed).
    advance_per_tick: f32,
}

impl CircadianClock {
    /// Construct with phase=0.
    pub fn new() -> Self {
        let advance = (KERNEL_CIRCADIAN_TICK_INTERVAL_S / KERNEL_CIRCADIAN_PERIOD_S) as f32;
        Self {
            state: CircadianState {
                phase: 0.0,
                day_progress: 0.0,
                reserved: 0.0,
            },
            advance_per_tick: advance,
        }
    }

    /// Construct with a starting phase (used for boot-time L0 snapshot
    /// restoration — kernel reads last-known phase from L0 snapshot and
    /// resumes from there).
    pub fn with_phase(initial_phase: f32) -> Self {
        let mut clock = Self::new();
        let p = initial_phase.rem_euclid(1.0);
        clock.state.phase = p;
        clock.state.day_progress = p;
        clock
    }

    /// Returns the current state without advancing.
    pub fn state(&self) -> CircadianState {
        self.state
    }

    /// Advance the clock by one tick. Returns the new state.
    pub fn tick(&mut self) -> CircadianState {
        let next = self.state.phase + self.advance_per_tick;
        // wrap at 1.0
        let wrapped = if next >= 1.0 { next - 1.0 } else { next };
        self.state.phase = wrapped;
        self.state.day_progress = wrapped;
        self.state
    }
}

impl Default for CircadianClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the circadian clock loop until shutdown is signaled.
///
/// Per SPEC §10.H: 1 Hz tick cadence. Each tick advances the clock + writes
/// the encoded payload to the `circadian.bin` slot via SeqLock.
pub async fn run_circadian_loop(slot: Arc<AsyncMutex<Slot>>, shutdown: Arc<tokio::sync::Notify>) {
    let mut clock = CircadianClock::new();
    let interval_dur = Duration::from_secs_f64(KERNEL_CIRCADIAN_TICK_INTERVAL_S);
    let mut tick_interval = interval(interval_dur);
    tick_interval.tick().await; // consume immediate first tick

    loop {
        tokio::select! {
            _ = tick_interval.tick() => {
                let state = clock.tick();
                let payload = state.encode();
                let mut s = slot.lock().await;
                if let Err(e) = s.write(&payload) {
                    warn!(err = ?e, "circadian: slot write failed");
                }
            }
            _ = shutdown.notified() => {
                debug!("circadian loop: shutdown received");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clock_starts_at_zero() {
        let clock = CircadianClock::new();
        let s = clock.state();
        assert_eq!(s.phase, 0.0);
        assert_eq!(s.day_progress, 0.0);
        assert_eq!(s.reserved, 0.0);
    }

    #[test]
    fn with_phase_normalizes_to_unit_interval() {
        let clock = CircadianClock::with_phase(2.5);
        assert_eq!(clock.state().phase, 0.5);
        let clock2 = CircadianClock::with_phase(-0.3);
        // -0.3 rem_euclid 1.0 = 0.7
        assert!((clock2.state().phase - 0.7).abs() < 1e-6);
    }

    #[test]
    fn tick_advances_by_1_over_86400() {
        let mut clock = CircadianClock::new();
        let s1 = clock.tick();
        let expected = (1.0 / 86400.0) as f32;
        assert!(
            (s1.phase - expected).abs() < 1e-9,
            "phase {} != expected {}",
            s1.phase,
            expected
        );
    }

    #[test]
    fn tick_phase_equals_day_progress() {
        let mut clock = CircadianClock::new();
        for _ in 0..100 {
            let s = clock.tick();
            assert_eq!(s.phase, s.day_progress);
        }
    }

    #[test]
    fn tick_wraps_at_one() {
        // Start near 1.0; one tick should wrap
        let mut clock = CircadianClock::with_phase(0.9999999);
        clock.tick();
        let p = clock.state().phase;
        assert!(p < 0.001, "phase {p} did not wrap");
    }

    #[test]
    fn payload_round_trip() {
        let s = CircadianState {
            phase: 0.5,
            day_progress: 0.5,
            reserved: 0.0,
        };
        let bytes = s.encode();
        assert_eq!(bytes.len(), 12);
        let decoded = CircadianState::decode(&bytes).unwrap();
        assert_eq!(decoded, s);
    }

    #[test]
    fn payload_too_short_returns_none() {
        let bytes = vec![0u8; 11];
        assert!(CircadianState::decode(&bytes).is_none());
    }

    #[test]
    fn payload_byte_layout_matches_spec_7_1() {
        // SPEC §7.1: f32 phase + f32 day_progress + f32 reserved = 12 bytes
        let s = CircadianState {
            phase: 1.0,
            day_progress: 1.0,
            reserved: 0.0,
        };
        let bytes = s.encode();
        assert_eq!(bytes.len(), 12);
        // Byte-exact check: f32(1.0) LE = 00 00 80 3F
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &0.0_f32.to_le_bytes());
    }
}
