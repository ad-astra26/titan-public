//! titan-bus — Main bus broker for Titan microkernel v2 Phase C.
//!
//! Byte-identical port of Python `titan_hcl/core/bus_socket.py`
//! `BusSocketServer` (B.2 protocol locked 2026-04-27).
//!
//! # Architecture
//!
//! - [`ring`] — `BoundedRing` per-subscriber bounded queue with P0 reserve
//!   region. Mirrors Python `BoundedRing`. The non-P0 deque has maxlen=
//!   capacity-p0_reserve; P0 deque has maxlen=p0_reserve. Drop policy per
//!   priority lane (SPEC §8.0).
//! - [`subscriber`] — Per-connection state: ring + coalesce_index +
//!   subscribed_topics + heartbeat tracking + drop accounting.
//!   (full impl spans C2-2 chunks)
//! - [`drift_bridge`] — D13/D14/D15 dual-emit (canonical ↔ legacy names).
//! - [`server`] (later in C2-2) — Unix-socket accept loop, HMAC handshake,
//!   tokio task per connection.
//! - [`heartbeat`] (later) — BUS_PING/PONG keepalive (`BUS_PING_INTERVAL_S=5`,
//!   `BUS_PING_TIMEOUT_S=15` — 3 missed pings).
//! - [`slow_consumer`] (later) — drop-rate detection +
//!   `BUS_SLOW_CONSUMER` emission when ratio > `BUS_SLOW_CONSUMER_DROP_RATE_RATIO=0.05`.
//!
//! # SPEC discipline
//!
//! Catalog at [`titan_core::bus_specs::SPECS`] (49 messages). Constants from
//! [`titan_core::constants`]. Wire framing from [`titan_core::frame`].
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every value cited.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod boot_buffer;
pub mod broker;
pub mod client;
pub mod drift_bridge;
pub mod heartbeat;
pub mod message;
pub mod ring;
pub mod server;
pub mod slow_consumer;
pub mod subscriber;

// Convenience re-exports
pub use crate::broker::BusBroker;
pub use crate::client::{BusClient, BusClientError, InboundEvent};
pub use crate::ring::{BoundedRing, RingError};
pub use crate::subscriber::{BrokerSubscriber, CoalesceKey};
