//! titan-state — Shm slot registry for Titan microkernel v2 Phase C.
//!
//! Per SPEC §7 (Shm Slot Byte Layouts) + §3.1 D04 (slot rename `trinity_state`
//! → `self_162d` with symlink) + §3.1 D05 (per-slot schema versions) +
//! §3.1 D24 (registry header).
//!
//! # Modules
//!
//! - [`spec`] — declarative `SLOT_SPECS` table mirroring SPEC §7.1
//!   (one row per slot with `name`, `schema_version`, `payload_bytes`,
//!   `creator`).
//! - [`slot`] — atomic slot file creation (`Slot::create`), SeqLock-based
//!   writer (`Slot::write`) + reader (`Slot::read`).
//! - [`registry`] — `SlotRegistry::create_all` boots the kernel-owned shm
//!   directory + creates every slot whose creator is `Kernel`. Manages
//!   D04 symlink creation.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod registry;
pub mod slot;
pub mod spec;

pub use crate::registry::{RegistryError, SlotRegistry};
pub use crate::slot::{Slot, SlotIoError};
pub use crate::spec::{SlotCreator, SlotSpec, SLOT_SPECS};
