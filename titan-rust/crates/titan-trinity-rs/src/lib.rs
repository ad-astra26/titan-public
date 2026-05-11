//! titan-trinity-rs — Trinity substrate library + binary.
//!
//! The library target exposes substrate primitives (boot, supervise,
//! filter_down, ground_up) so integration tests under `tests/` can
//! exercise them and so future C-S5/C-S6 daemons can import the
//! ground_up + filter_down helpers directly per master plan §8 D21.
//!
//! The `main.rs` binary re-uses the same modules via `pub mod` declarations
//! — Cargo allows both a `[[bin]]` and `[lib]` target in the same crate;
//! we keep the modules' canonical home in their own files and have
//! `main.rs` import via `use crate::...` (since `main.rs` IS the binary's
//! crate root + `lib.rs` IS the library's, both must declare the same
//! `pub mod` set).

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
