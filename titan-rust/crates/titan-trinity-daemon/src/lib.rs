//! titan-trinity-daemon — Shared library for the 6 trinity daemons
//! (Phase C C-S5 inner-{body,mind,spirit} + C-S6 outer-{body,mind,spirit}).
//!
//! Per master plan §10.5 chunk C5-1: extracts every primitive that's identical
//! across all 6 daemons so each binary stays small (~250–350 LOC main.rs +
//! tick_loop.rs).
//!
//! # Modules
//!
//! - [`error`] — `DaemonError` + `Result<T>` for the whole crate.
//! - [`ground_up`] — 1:1 port of `titan_hcl/logic/ground_up.py`.
//!   Applies grounding nudges from lower topology to body[0:5] (full)
//!   + mind willing[10:15] (per SPEC G10).
//!
//! Modules pending in this chunk (commits to follow within C5-1):
//! - `content_hash` — `xxh3_64` content-gate (SPEC §7.1: skip slot writes
//!   when payload unchanged).
//! - `filter_apply` — apply UNIFIED + LOCAL filter_down multipliers with
//!   floor/ceil clipping + EMA smoothing + Observer-dim masking (G8).
//! - `slot_io` — typed read/write helpers wrapping `titan-state::Slot` for
//!   `[f32; N]` slot payloads.
//! - `tick` — `SchumannTicker` for fixed-frequency loops at 7.83 / 23.49 /
//!   70.47 Hz (G13) with drift measurement (≤ 0.5% target per master plan §16).
//! - `adoption` — B.2.1 79-byte `ADOPTION_REQUEST` / `ADOPTION_ACK` msgpack
//!   vectors (SPEC §8.4 + D14 + D18).
//!
//! Module pending C-S3 dependency:
//! - `subscriptions` — bus-client subscriber wrapper. Lands after C-S3
//!   ships its bus client (chunk C3-3 boot.rs); see PLAN §0.4.
//!
//! # SPEC discipline
//!
//! Per `feedback_phase_c_spec_enforcement.md` Rule 1: every value in this
//! crate comes from `titan-core::constants` (auto-generated from
//! `SPEC_titan_architecture_constants.toml`) or directly cites a SPEC §.
//! Pre-commit hook (`arch_map phase-c verify --strict`) blocks drift.
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod adoption;
pub mod checkpoint;
pub mod content_hash;
pub mod drift_warn;
pub mod error;
pub mod filter_apply;
pub mod firing_payload;
pub mod ground_up;
pub mod slot_io;
pub mod subscriptions;
// NOTE: There is intentionally NO `tick` module here. Per master plan §7
// (Cargo Workspace Structure) + C-S3 PLAN §1.1 #2: Schumann timer wheels
// are the canonical responsibility of `titan-schumann` — explicitly
// designed as the "shared library used by inner-spirit-rs (C-S5) and
// inner-mind-rs / inner-body-rs daemons (C-S5)". Daemons import
// `titan_schumann::{SchumannGenerator, SchumannRole, TickEvent}` directly.
//
// The "generic tick loop" master plan §7 attributes to titan-trinity-daemon
// is the COLLECTION of helper modules above (filter_apply + ground_up +
// slot_io + content_hash + subscriptions + adoption) — they compose into
// the per-daemon tick body driven by a SchumannGenerator from titan-schumann.

// ── C-S6 outer-side additions (per master plan §7 + PLAN §3.4) ──
//
// `tick.rs` ships `SchumannTicker` for inner-trinity (Schumann-locked,
// SPEC G13). Post-A.S8 D2 the outer daemons are ALSO Schumann-locked
// (each ticks via SchumannGenerator at its role's Schumann harmonic);
// OUTER_*_TICK_BASE_S now drives only the Python sensor-sidecar
// source-refresh cadence + the 3× stale threshold (D-SPEC-100). The
// modules below provide the outer-specific helpers; inner daemons keep
// using SchumannTicker / existing slot_io / filter_apply / ground_up
// unchanged. (The pre-D2 `jittered_tick` ticker was removed D-SPEC-100
// as dead code — never wired into any daemon main loop.)

pub mod focus_input;
pub mod homeostasis;
pub mod neuromod_read;
pub mod observer_mask;
pub mod publish_throttle;
pub mod pulse_watch;
pub mod restoring_cfg;
pub mod sensor_cache_read;

pub use crate::adoption::{
    decode_adoption_ack_payload, decode_adoption_request_payload, encode_adoption_ack_payload,
    encode_adoption_request_payload, AdoptionAck, AdoptionRequest, StartMethod,
    CANONICAL_ADOPTION_ACK_PAYLOAD_BYTES, CANONICAL_ADOPTION_REQUEST_PAYLOAD_BYTES,
};
pub use crate::checkpoint::{
    checkpoint_filename, checkpoint_path, load_for_part as load_checkpoint_for_part,
    payload_bytes as checkpoint_payload_bytes, write_for_part as write_checkpoint_for_part,
    CheckpointSnapshot, CHECKPOINT_SCHEMA_VERSION_F32,
};
pub use crate::content_hash::ContentGate;
pub use crate::drift_warn::{DriftAggregator, DRIFT_WARN_MIN_INTERVAL};
pub use crate::error::{DaemonError, DaemonResult};
pub use crate::filter_apply::{
    apply_multipliers, apply_spirit_strength, compose_multipliers, compose_multipliers_default,
    EmaSmoother, FILTER_DOWN_EMA_SMOOTHING, MULTIPLIER_CEIL, MULTIPLIER_FLOOR,
    SPIRIT_FILTER_STRENGTH_MULT, TENSOR_MAX, TENSOR_MIN,
};
pub use crate::firing_payload::{encode_firing_payload, now_secs, FiringSlotWriter};
pub use crate::focus_input::{
    compose_into as compose_focus_into_enrichment, open_if_present as open_focus_input_if_present,
    read_nudge as read_focus_nudge, FocusNudge, FocusPart, FOCUS_INPUT_PAYLOAD_BYTES,
    FOCUS_INPUT_SIDECAR,
};
pub use crate::ground_up::{
    GroundUpEnricher, GroundUpNudge, Side, GROUND_UP_DEFAULT_DAMPING, GROUND_UP_DEFAULT_STRENGTH,
    GROUND_UP_MAX_NUDGE,
};
pub use crate::homeostasis::{
    gradient, observe, stateful_update, Layer, LayerObs, RestoringCfg, CENTRE, DEFAULT_A_DMAG,
    DEFAULT_A_DRIFT, DEFAULT_A_MAG, DEFAULT_K_DAMP, DEFAULT_K_DIRECTION, DEFAULT_K_DRIVE,
    DEFAULT_K_MOMENTUM, DEFAULT_K_RESTORE,
};
pub use crate::neuromod_read::{
    open_if_present as open_neuromod_slot_if_present, read_gain as read_neuromod_gain,
    NEUROMOD_GAIN_MAX, NEUROMOD_GAIN_MIN, NEUROMOD_GAIN_NEUTRAL,
};
pub use crate::restoring_cfg::{
    load_for_layer as load_restoring_cfg, TRINITY_RESTORING_PAYLOAD_BYTES,
    TRINITY_RESTORING_SIDECAR,
};
pub use crate::slot_io::{
    decode_floats, encode_floats, open_slot, read_dim_slice, read_topology_inner_lower,
    read_topology_outer_lower, read_topology_whole, write_dim_slice,
};
pub use crate::subscriptions::{
    connect_daemon, decode_filter_down_payload, decode_local_filter_down_payload,
    encode_filter_down_payload, InnerFilterDownPayload, LocalFilterDownPayload, INNER_BODY_TOPICS,
    INNER_MIND_TOPICS, INNER_SPIRIT_TOPICS, OUTER_BODY_TOPICS, OUTER_MIND_TOPICS,
    OUTER_SPIRIT_TOPICS,
};

// ── C-S6 re-exports (observer mask + sensor cache) ──
pub use crate::observer_mask::{
    extract_outer_spirit_content, mask_observer_dims_in_place, observer_dims_are_zero,
    CONTENT_DIM_COUNT, OBSERVER_BYTE_END, OBSERVER_BYTE_START, OBSERVER_DIM_COUNT,
};
pub use crate::publish_throttle::PublishThrottle;
pub use crate::pulse_watch::{PulseClockRole, PulseEdges, PulseWatcher};
pub use crate::sensor_cache_read::{
    age_seconds, current_wall_ns, read_sensor_cache, SensorCacheRead,
};
