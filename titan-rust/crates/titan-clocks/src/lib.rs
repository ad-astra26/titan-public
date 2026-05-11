//! titan-clocks — Circadian + π-heartbeat clocks for Titan microkernel v2.
//!
//! Per SPEC §10.H (consolidated cadence view) + §7.1 (slot byte layouts) +
//! §8.1 (`KERNEL_EPOCH_TICK` is P0, never drop):
//!
//! - **Circadian clock** — 1 Hz logical tick, 24h full-cycle period.
//!   Writes `circadian.bin`: `[f32 phase | f32 day_progress | f32 reserved]`.
//! - **π-heartbeat** — ~3 Hz target tick. Writes `pi_heartbeat.bin`:
//!   `[f32 phase | u64 pulse_count]`. Each tick increments `pulse_count`
//!   and writes `epoch_counter.bin` (`u64 LE`).
//! - **`KERNEL_EPOCH_TICK`** — emitted by π-heartbeat loop on every tick;
//!   payload `{epoch_id, ts, dt_s}` per SPEC §8.1. Caller injects an
//!   `EpochTickPublisher` so the loop is testable without a real broker.
//!
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every value comes from
//! `titan-core::constants`. No magic numbers.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod circadian;
pub mod pi_heartbeat;

pub use crate::circadian::{run_circadian_loop, CircadianClock, CircadianState};
pub use crate::pi_heartbeat::{
    run_pi_heartbeat_loop, EpochTickPublisher, PiHeartbeat, PiTickEvent,
};
