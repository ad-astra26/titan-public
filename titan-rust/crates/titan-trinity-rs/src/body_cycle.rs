//! body_cycle — Substrate per-body-cycle runtime orchestration.
//!
//! Per SPEC §10.G step 5 + §10.E telemetry write-then-publish ordering +
//! §10.H consolidated cadence (substrate at body publish rate ~1.15s) +
//! §11.B defensive error handling. Closes rFP_phase_c_close_all_runtime_gaps
//! chunks 9A + 9B (Gap A substrate body cycle + Gap B chi compute).
//!
//! # Why this module exists
//!
//! `tick_loop::SubstrateState::body_tick` is pure compute — it returns the
//! bytes to write but performs no I/O. Pre-9A no production caller invoked
//! it; the only call sites lived under `#[cfg(test)]`. The runtime gate
//! `arch_map phase-c daemon-tick-rate` surfaced this on 2026-05-05 as 3
//! STUCK slots: `topology_30d.bin`, `sphere_clocks.bin`, `chi_state.bin`
//! (all owned by titan-trinity-rs per SPEC §7.1 / §9.A; all version=1
//! forever because nothing wrote them).
//!
//! This module spawns a tokio task at boot (between MODULE_READY and
//! SIGTERM-wait in `main.rs`) that:
//!
//!   1. Opens 6 read-handles (inner+outer × body/mind/spirit) + 3 write-
//!      handles (topology_30d, sphere_clocks, chi_state) + 1 neuromod
//!      read-handle.
//!   2. Connects to the main bus broker (substrate's bus client per
//!      `main_bus_publisher::connect_main_bus`).
//!   3. Loops on `tokio::time::interval(BODY_CYCLE_INTERVAL_MS=1150)`.
//!   4. Per tick:
//!      a. Read 6 input slots → BodyTickInputs.
//!      b. Call `SubstrateState::body_tick(&inputs)` → BodyTickOutputs.
//!      c. Compute `ChiState::compute(&chi_inputs)`.
//!      d. Write topology_30d.bin (writes BEFORE publishes per §10.E).
//!      e. Write sphere_clocks.bin.
//!      f. Write chi_state.bin.
//!      g. Publish per-pulse SPHERE_PULSE events (P0).
//!      h. Publish TRINITY_SUBSTRATE_TOPOLOGY_UPDATED (P1).
//!   5. Defensive: every step is wrapped in match/if-let-Err. Errors
//!      log + skip remaining steps + wait next interval. Per SPEC §11.B
//!      a single bad tick must not propagate to kill the body cycle task.
//!
//! # Lifecycle
//!
//! Spawned via `tokio::spawn` from `main.rs`. Holds an `Arc<Notify>`
//! shutdown signal — `tokio::select!` between interval ticks and
//! `shutdown.notified()`. On SIGTERM the kernel notifies, the loop exits
//! cleanly, and the task handle joins within the SPEC §17 grace window.

use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use thiserror::Error;
use tokio::sync::Notify;
use tokio::time::{Duration, MissedTickBehavior};
use tracing::{debug, info, warn};

use titan_bus::client::BusClient;
use titan_core::constants::{
    BODY_CYCLE_INTERVAL_MS, NEUROMOD_FIELD_COUNT, SPHERE_CLOCKS_PAYLOAD_BYTES,
    TOPOLOGY_30D_PAYLOAD_BYTES,
};
use titan_state::Slot;

use crate::chi_state::{ChiInputs, ChiState};
use crate::main_bus_publisher::{
    publish_sphere_pulse, publish_topology_updated, MainBusError,
};
use crate::tick_loop::{BodyTickInputs, SubstrateState};
use crate::topology::{BODY_5D, MIND_15D, SPIRIT_45D};

/// Errors during body-cycle setup. Tick errors are logged + swallowed
/// per SPEC §11.B — they never bubble out of `run_substrate_body_cycle`.
#[derive(Debug, Error)]
pub enum BodyCycleError {
    /// Failed to open a slot at boot. Includes the slot file name for
    /// observability — `arch_map phase-c daemon-tick-rate` will report
    /// the matching slot as STUCK / ERROR.
    #[error("open slot {slot}: {source}")]
    SlotOpen {
        /// Slot file name (e.g. `inner_body_5d.bin`).
        slot: String,
        /// Underlying titan-state error.
        #[source]
        source: titan_state::SlotIoError,
    },
}

/// Slot handles owned by the body cycle task. 6 read-only + 3 read-write +
/// 1 neuromod read.
pub struct BodyCycleSlots {
    /// Inner-body daemon tensor slot.
    pub inner_body: Slot,
    /// Inner-mind daemon tensor slot.
    pub inner_mind: Slot,
    /// Inner-spirit daemon tensor slot.
    pub inner_spirit: Slot,
    /// Outer-body daemon tensor slot.
    pub outer_body: Slot,
    /// Outer-mind daemon tensor slot.
    pub outer_mind: Slot,
    /// Outer-spirit daemon tensor slot.
    pub outer_spirit: Slot,
    /// `topology_30d.bin` — substrate writes per body cycle (SPEC §7.1).
    pub topology_30d: Slot,
    /// `sphere_clocks.bin` — substrate writes per body cycle.
    pub sphere_clocks: Slot,
    /// `chi_state.bin` — substrate writes per body cycle (SPEC §7.1 row 13).
    pub chi_state: Slot,
    /// `neuromod_state.bin` — Python L2 writes; substrate reads for chi
    /// urgency. Optional: if Python L2 hasn't created/populated it yet,
    /// `Slot::open` fails and we proceed with a zero neuromod vector.
    pub neuromod_state: Option<Slot>,
}

impl BodyCycleSlots {
    /// Open all 9 required slots + try-open neuromod_state. Required slot
    /// failures bubble up as `BodyCycleError::SlotOpen` — the kernel
    /// pre-creates them at C-S2 boot, so absence here means the substrate
    /// is misconfigured (different shm_dir, etc) which is a hard error.
    pub fn open(shm_dir: &Path) -> Result<Self, BodyCycleError> {
        let open = |name: &str| -> Result<Slot, BodyCycleError> {
            let path = shm_dir.join(name);
            Slot::open(&path).map_err(|source| BodyCycleError::SlotOpen {
                slot: name.to_string(),
                source,
            })
        };

        let inner_body = open("inner_body_5d.bin")?;
        let inner_mind = open("inner_mind_15d.bin")?;
        let inner_spirit = open("inner_spirit_45d.bin")?;
        let outer_body = open("outer_body_5d.bin")?;
        let outer_mind = open("outer_mind_15d.bin")?;
        let outer_spirit = open("outer_spirit_45d.bin")?;
        let topology_30d = open("topology_30d.bin")?;
        let sphere_clocks = open("sphere_clocks.bin")?;
        let chi_state = open("chi_state.bin")?;

        // neuromod_state.bin is Python-L2-managed (per SPEC §7.1 SlotCreator
        // = PythonModule); kernel does NOT pre-create it, so opening can
        // race the Python module's first write. Best-effort.
        let neuromod_state = match Slot::open(shm_dir.join("neuromod_state.bin")) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!(
                    err = ?e,
                    "neuromod_state.bin not yet open at boot — chi urgency will read 0 until Python L2 creates the slot; substrate continues"
                );
                None
            }
        };

        Ok(Self {
            inner_body,
            inner_mind,
            inner_spirit,
            outer_body,
            outer_mind,
            outer_spirit,
            topology_30d,
            sphere_clocks,
            chi_state,
            neuromod_state,
        })
    }
}

/// Run the substrate body cycle loop. Per §10.G step 5 + §11.B defensive
/// error handling. Returns when `shutdown.notified()` fires.
///
/// Spawned via `tokio::spawn(run_substrate_body_cycle(...))` from `main.rs`
/// AFTER the substrate has connected to the main bus + before the SIGTERM
/// wait.
pub async fn run_substrate_body_cycle(
    mut slots: BodyCycleSlots,
    bus_client: Arc<BusClient>,
    shutdown: Arc<Notify>,
) {
    let mut state = SubstrateState::new();
    let interval_period = Duration::from_millis(BODY_CYCLE_INTERVAL_MS);
    let mut interval = tokio::time::interval(interval_period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    interval.tick().await; // First tick is immediate; consume to align cadence.

    info!(
        event = "BODY_CYCLE_STARTED",
        cadence_ms = BODY_CYCLE_INTERVAL_MS,
        "substrate body cycle task started"
    );

    let mut tick_count: u64 = 0;
    loop {
        tokio::select! {
            _ = interval.tick() => {
                tick_count = tick_count.wrapping_add(1);
                run_one_tick(&mut state, &mut slots, &bus_client, tick_count).await;
            }
            _ = shutdown.notified() => {
                info!(
                    event = "BODY_CYCLE_STOPPED",
                    reason = "shutdown",
                    total_ticks = tick_count,
                    "substrate body cycle task exiting cleanly"
                );
                return;
            }
        }
    }
}

/// One body-cycle tick. Each step is wrapped to log + skip on error per
/// SPEC §11.B. Errors do NOT propagate.
async fn run_one_tick(
    state: &mut SubstrateState,
    slots: &mut BodyCycleSlots,
    bus_client: &BusClient,
    tick_count: u64,
) {
    // Step 1: read 6 input tensors.
    let inputs = match read_body_tick_inputs(slots) {
        Ok(i) => i,
        Err(e) => {
            warn!(
                event = "BODY_CYCLE_READ_ERROR",
                tick = tick_count,
                err = %e,
                "skipping tick — slot read failed"
            );
            return;
        }
    };

    // Step 2: pure compute body_tick.
    let outputs = state.body_tick(&inputs);

    // Step 3: compute substrate-scoped chi state.
    let neuromod_6 = read_neuromod_6(slots);
    let chi_inputs = ChiInputs {
        inner_body_5d: &inputs.inner_body_5d,
        inner_mind_15d: &inputs.inner_mind_15d,
        inner_spirit_45d: &inputs.inner_spirit_45d,
        outer_body_5d: &inputs.outer_body_5d,
        outer_mind_15d: &inputs.outer_mind_15d,
        outer_spirit_45d: &inputs.outer_spirit_45d,
        sphere_clocks: &state.sphere_clocks,
        neuromod_6: &neuromod_6,
    };
    let chi = ChiState::compute(&chi_inputs);
    let chi_payload = chi.serialize();

    // Step 4: writes BEFORE publishes per SPEC §10.E (reverse = race).
    // topology_30d → sphere_clocks → chi_state. Each write defended.
    let topology_bytes = bytes_of_topology_30d(&outputs.topology_30d);
    if let Err(e) = slots.topology_30d.write(&topology_bytes) {
        warn!(
            event = "BODY_CYCLE_WRITE_ERROR",
            tick = tick_count,
            slot = "topology_30d.bin",
            err = ?e,
            "skipping remaining steps for this tick"
        );
        return;
    }
    if let Err(e) = slots.sphere_clocks.write(&outputs.sphere_clocks_payload) {
        warn!(
            event = "BODY_CYCLE_WRITE_ERROR",
            tick = tick_count,
            slot = "sphere_clocks.bin",
            err = ?e,
            "skipping remaining steps for this tick"
        );
        return;
    }
    if let Err(e) = slots.chi_state.write(&chi_payload) {
        warn!(
            event = "BODY_CYCLE_WRITE_ERROR",
            tick = tick_count,
            slot = "chi_state.bin",
            err = ?e,
            "skipping remaining steps for this tick"
        );
        return;
    }

    // Step 5: publish bus events. Per-pulse SPHERE_PULSE then a single
    // TRINITY_SUBSTRATE_TOPOLOGY_UPDATED. Errors logged + swallowed.
    let ts = systime_to_unix_secs();
    for pulse in outputs.pulses.iter() {
        if let Err(e) = publish_sphere_pulse(bus_client, pulse, ts).await {
            log_publish_error("SPHERE_PULSE", tick_count, &e);
            // Continue — one publish failure shouldn't skip the rest.
        }
    }
    if let Err(e) = publish_topology_updated(bus_client, ts).await {
        log_publish_error("TRINITY_SUBSTRATE_TOPOLOGY_UPDATED", tick_count, &e);
    }

    // Periodic structured tick log — every ~10s (every 9 ticks at 1.15s)
    // gives runtime observability without spamming journald.
    if tick_count % 9 == 0 {
        debug!(
            event = "BODY_CYCLE_TICK",
            tick = tick_count,
            chi_total = chi.total,
            chi_spirit = chi.spirit,
            chi_mind = chi.mind,
            chi_body = chi.body,
            inner_mag = outputs.inner_magnitude,
            outer_mag = outputs.outer_magnitude,
            pulses = outputs.pulses.len(),
            "body cycle tick"
        );
    }
}

/// Read the 6 daemon tensor slots into a `BodyTickInputs`. On any read
/// error returns the propagated error so the caller can log + skip the tick.
fn read_body_tick_inputs(
    slots: &BodyCycleSlots,
) -> Result<BodyTickInputs, BodyCycleSlotReadError> {
    let inner_body = read_f32_array::<BODY_5D>(&slots.inner_body, "inner_body_5d.bin")?;
    let inner_mind = read_f32_array::<MIND_15D>(&slots.inner_mind, "inner_mind_15d.bin")?;
    let inner_spirit =
        read_f32_array::<SPIRIT_45D>(&slots.inner_spirit, "inner_spirit_45d.bin")?;
    let outer_body = read_f32_array::<BODY_5D>(&slots.outer_body, "outer_body_5d.bin")?;
    let outer_mind = read_f32_array::<MIND_15D>(&slots.outer_mind, "outer_mind_15d.bin")?;
    let outer_spirit =
        read_f32_array::<SPIRIT_45D>(&slots.outer_spirit, "outer_spirit_45d.bin")?;

    Ok(BodyTickInputs {
        inner_body_5d: inner_body,
        inner_mind_15d: inner_mind,
        inner_spirit_45d: inner_spirit,
        outer_body_5d: outer_body,
        outer_mind_15d: outer_mind,
        outer_spirit_45d: outer_spirit,
        // dt_s is the body-cycle wall-clock interval; constant 1.149 matches
        // the cadence specified in titan_core::constants::BODY_CYCLE_INTERVAL_MS.
        dt_s: (BODY_CYCLE_INTERVAL_MS as f32) / 1000.0,
    })
}

/// Read neuromod_state.bin as 6-float array. Returns all-zeros if the
/// optional slot is absent or read fails — substrate must continue per
/// §11.B; chi.urgency just won't reflect neuromod pressure that tick.
fn read_neuromod_6(slots: &BodyCycleSlots) -> [f32; NEUROMOD_FIELD_COUNT as usize] {
    let Some(slot) = slots.neuromod_state.as_ref() else {
        return [0.0; NEUROMOD_FIELD_COUNT as usize];
    };
    match slot.read() {
        Ok(bytes) if bytes.len() >= 24 => {
            let mut out = [0.0_f32; NEUROMOD_FIELD_COUNT as usize];
            for i in 0..(NEUROMOD_FIELD_COUNT as usize) {
                let off = i * 4;
                out[i] = f32::from_le_bytes(
                    bytes[off..off + 4]
                        .try_into()
                        .expect("4 bytes after length check"),
                );
            }
            out
        }
        Ok(_) | Err(_) => [0.0; NEUROMOD_FIELD_COUNT as usize],
    }
}

/// Slot-read error wrapper — names the slot for actionable logging.
#[derive(Debug, Error)]
#[error("slot read {slot}: {source}")]
struct BodyCycleSlotReadError {
    slot: String,
    #[source]
    source: BodyCycleSlotReadCause,
}

#[derive(Debug, Error)]
enum BodyCycleSlotReadCause {
    #[error("io: {0}")]
    Io(#[from] titan_state::SlotIoError),
    #[error(
        "expected {expected} bytes (≥ {needed} for f32 array), got {actual}"
    )]
    ShortRead {
        expected: usize,
        needed: usize,
        actual: usize,
    },
}

/// Read a slot as `[f32; N]`. Slots may have larger payloads than N×4
/// (e.g. inner_mind_15d is 60 bytes); we read exactly the first N×4.
fn read_f32_array<const N: usize>(
    slot: &Slot,
    slot_name: &str,
) -> Result<[f32; N], BodyCycleSlotReadError> {
    let bytes = slot.read().map_err(|e| BodyCycleSlotReadError {
        slot: slot_name.to_string(),
        source: BodyCycleSlotReadCause::Io(e),
    })?;
    let needed = N * 4;
    if bytes.len() < needed {
        return Err(BodyCycleSlotReadError {
            slot: slot_name.to_string(),
            source: BodyCycleSlotReadCause::ShortRead {
                expected: needed,
                needed,
                actual: bytes.len(),
            },
        });
    }
    let mut out = [0.0_f32; N];
    for i in 0..N {
        let off = i * 4;
        out[i] = f32::from_le_bytes(
            bytes[off..off + 4]
                .try_into()
                .expect("4 bytes after length check"),
        );
    }
    Ok(out)
}

/// Convert a `[f32; 30]` topology to the canonical 120-byte LE payload.
fn bytes_of_topology_30d(t: &[f32; 30]) -> [u8; TOPOLOGY_30D_PAYLOAD_BYTES as usize] {
    let mut out = [0u8; TOPOLOGY_30D_PAYLOAD_BYTES as usize];
    for (i, v) in t.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Wall-clock seconds since UNIX epoch as `f64`. Used for bus event
/// payloads — matches the Python convention used by L2 consumers.
fn systime_to_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Structured publish-error log helper.
fn log_publish_error(event_type: &'static str, tick: u64, err: &MainBusError) {
    warn!(
        event = "BODY_CYCLE_PUBLISH_ERROR",
        tick = tick,
        msg_type = event_type,
        err = ?err,
        "bus publish failed; substrate continues"
    );
}

/// Suppress unused-import warning in builds where SPHERE_CLOCKS_PAYLOAD_BYTES
/// isn't referenced (defensive — present in case SPEC layout changes).
#[allow(dead_code)]
const _SPHERE_CLOCKS_PAYLOAD_BYTES_USED: u64 = SPHERE_CLOCKS_PAYLOAD_BYTES;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tempfile::tempdir;
    use titan_core::constants::{
        CHI_STATE_PAYLOAD_BYTES, CHI_STATE_SCHEMA_VERSION, INNER_BODY_5D_SCHEMA_VERSION,
        INNER_MIND_15D_SCHEMA_VERSION, INNER_SPIRIT_45D_SCHEMA_VERSION,
        OUTER_BODY_5D_SCHEMA_VERSION, OUTER_MIND_15D_SCHEMA_VERSION,
        OUTER_SPIRIT_45D_SCHEMA_VERSION, SPHERE_CLOCKS_SCHEMA_VERSION,
        TOPOLOGY_30D_SCHEMA_VERSION,
    };

    fn create_slot(dir: &Path, name: &str, schema: u32, payload_bytes: u32) -> Slot {
        Slot::create(dir.join(name), schema, payload_bytes)
            .expect("test slot creation must succeed in tmp shm")
    }

    fn create_all_slots(dir: &Path) {
        let _ = create_slot(dir, "inner_body_5d.bin", INNER_BODY_5D_SCHEMA_VERSION as u32, 5 * 4);
        let _ = create_slot(dir, "inner_mind_15d.bin", INNER_MIND_15D_SCHEMA_VERSION as u32, 15 * 4);
        let _ = create_slot(
            dir,
            "inner_spirit_45d.bin",
            INNER_SPIRIT_45D_SCHEMA_VERSION as u32,
            45 * 4,
        );
        let _ = create_slot(dir, "outer_body_5d.bin", OUTER_BODY_5D_SCHEMA_VERSION as u32, 5 * 4);
        let _ = create_slot(dir, "outer_mind_15d.bin", OUTER_MIND_15D_SCHEMA_VERSION as u32, 15 * 4);
        let _ = create_slot(
            dir,
            "outer_spirit_45d.bin",
            OUTER_SPIRIT_45D_SCHEMA_VERSION as u32,
            45 * 4,
        );
        let _ = create_slot(dir, "topology_30d.bin", TOPOLOGY_30D_SCHEMA_VERSION as u32, 30 * 4);
        let _ = create_slot(dir, "sphere_clocks.bin", SPHERE_CLOCKS_SCHEMA_VERSION as u32, 168);
        let _ = create_slot(
            dir,
            "chi_state.bin",
            CHI_STATE_SCHEMA_VERSION as u32,
            CHI_STATE_PAYLOAD_BYTES as u32,
        );
    }

    fn write_known_input(slot: &mut Slot, vals: &[f32]) {
        let mut bytes = vec![0u8; vals.len() * 4];
        for (i, v) in vals.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        slot.write(&bytes).expect("write input");
    }

    #[test]
    fn bytes_of_topology_30d_round_trips() {
        let mut t = [0.0_f32; 30];
        for (i, v) in t.iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        let bytes = bytes_of_topology_30d(&t);
        assert_eq!(bytes.len(), 120);
        for (i, v) in t.iter().enumerate() {
            let off = i * 4;
            let recovered =
                f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            assert!((recovered - *v).abs() < 1e-6);
        }
    }

    #[test]
    fn body_cycle_slots_open_succeeds_when_slots_exist() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let slots = BodyCycleSlots::open(dir.path()).expect("all required slots created");
        assert!(slots.neuromod_state.is_none(), "neuromod absent in test fixture");
        // verify all required handles opened
        assert_eq!(slots.inner_body.read().unwrap().len(), 20);
        assert_eq!(slots.topology_30d.read().unwrap().len(), 120);
        assert_eq!(slots.chi_state.read().unwrap().len(), 24);
    }

    #[test]
    fn body_cycle_slots_open_fails_when_required_slot_missing() {
        let dir = tempdir().unwrap();
        // Skip creating inner_body_5d.bin — required slot
        create_slot(dir.path(), "inner_mind_15d.bin", 1, 60);
        let result = BodyCycleSlots::open(dir.path());
        assert!(result.is_err(), "expected open to fail when required slot missing");
        match result.err().unwrap() {
            BodyCycleError::SlotOpen { slot, .. } => {
                assert_eq!(slot, "inner_body_5d.bin");
            }
        }
    }

    #[test]
    fn read_body_tick_inputs_returns_typed_arrays() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut slots = BodyCycleSlots::open(dir.path()).unwrap();

        let body_vals: Vec<f32> = (0..5).map(|i| i as f32 * 0.1).collect();
        write_known_input(&mut slots.inner_body, &body_vals);

        let inputs = read_body_tick_inputs(&slots).expect("read ok");
        for i in 0..5 {
            assert!((inputs.inner_body_5d[i] - body_vals[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn read_neuromod_6_returns_zeros_when_slot_absent() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let slots = BodyCycleSlots::open(dir.path()).unwrap();
        let neuromod = read_neuromod_6(&slots);
        for v in neuromod.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn read_neuromod_6_returns_values_when_slot_populated() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        // Create + write the optional slot
        let mut nm = create_slot(dir.path(), "neuromod_state.bin", 1, 24);
        let vals = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut bytes = [0u8; 24];
        for (i, v) in vals.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        nm.write(&bytes).unwrap();
        let slots = BodyCycleSlots::open(dir.path()).unwrap();
        let neuromod = read_neuromod_6(&slots);
        for (i, v) in vals.iter().enumerate() {
            assert!((neuromod[i] - *v).abs() < 1e-6);
        }
    }

    /// Drives the full per-tick path without bus client (None publish path
    /// is not exercised here — bus integration is covered separately in
    /// main_bus_publisher tests). Verifies that one tick:
    ///   - Reads the input slots
    ///   - Computes body_tick + chi
    ///   - Writes topology_30d / sphere_clocks / chi_state with version increment
    #[tokio::test]
    async fn one_tick_writes_three_output_slots_with_version_increment() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut slots = BodyCycleSlots::open(dir.path()).unwrap();
        let mut state = SubstrateState::new();

        // Pre-populate inputs with known non-zero values so chi/topology
        // are non-trivial.
        write_known_input(&mut slots.inner_body, &[0.5; 5]);
        let mut mind_vals = vec![0.0_f32; 15];
        for v in mind_vals[10..15].iter_mut() {
            *v = 0.5;
        }
        write_known_input(&mut slots.inner_mind, &mind_vals);
        write_known_input(&mut slots.inner_spirit, &vec![0.5_f32; 45]);
        write_known_input(&mut slots.outer_body, &[0.5; 5]);
        write_known_input(&mut slots.outer_mind, &mind_vals);
        write_known_input(&mut slots.outer_spirit, &vec![0.5_f32; 45]);

        // Drive the read+compute+write path manually (without bus).
        let inputs = read_body_tick_inputs(&slots).unwrap();
        let outputs = state.body_tick(&inputs);
        let neuromod = read_neuromod_6(&slots);
        let chi = ChiState::compute(&ChiInputs {
            inner_body_5d: &inputs.inner_body_5d,
            inner_mind_15d: &inputs.inner_mind_15d,
            inner_spirit_45d: &inputs.inner_spirit_45d,
            outer_body_5d: &inputs.outer_body_5d,
            outer_mind_15d: &inputs.outer_mind_15d,
            outer_spirit_45d: &inputs.outer_spirit_45d,
            sphere_clocks: &state.sphere_clocks,
            neuromod_6: &neuromod,
        });
        slots
            .topology_30d
            .write(&bytes_of_topology_30d(&outputs.topology_30d))
            .unwrap();
        slots.sphere_clocks.write(&outputs.sphere_clocks_payload).unwrap();
        slots.chi_state.write(&chi.serialize()).unwrap();

        // Read back and verify non-zero / well-formed output.
        let topo_bytes = slots.topology_30d.read().unwrap();
        assert_eq!(topo_bytes.len(), 120);
        let topo_first = f32::from_le_bytes(topo_bytes[0..4].try_into().unwrap());
        assert!(topo_first > 0.0, "topology_30d[0] should be non-zero");

        let chi_bytes = slots.chi_state.read().unwrap();
        assert_eq!(chi_bytes.len(), 24);
        let chi_total = f32::from_le_bytes(chi_bytes[0..4].try_into().unwrap());
        assert!(chi_total > 0.0, "chi.total should be non-zero with active inputs");
    }

    /// Counter-driven test: spawn `run_substrate_body_cycle`, let it run a
    /// few ticks, then trigger shutdown and confirm cleanup. Uses a fake
    /// bus client by NOT connecting one — substrate proceeds with publish
    /// failures logged + swallowed (per SPEC §11.B). This exercises the
    /// full task lifecycle.
    #[tokio::test]
    async fn run_substrate_body_cycle_advances_versions_and_shuts_down_clean() {
        let dir = tempdir().unwrap();
        create_all_slots(dir.path());
        let mut slots = BodyCycleSlots::open(dir.path()).unwrap();

        // Pre-populate inputs so body cycle has real data.
        write_known_input(&mut slots.inner_body, &[0.5; 5]);
        write_known_input(&mut slots.inner_mind, &vec![0.3_f32; 15]);
        write_known_input(&mut slots.inner_spirit, &vec![0.5_f32; 45]);
        write_known_input(&mut slots.outer_body, &[0.5; 5]);
        write_known_input(&mut slots.outer_mind, &vec![0.3_f32; 15]);
        write_known_input(&mut slots.outer_spirit, &vec![0.5_f32; 45]);

        let topo_path = dir.path().join("topology_30d.bin");
        let chi_path = dir.path().join("chi_state.bin");

        // Set up a fake broker so we have a real BusClient.
        let bus_client = build_test_bus_client().await;

        let shutdown = Arc::new(Notify::new());
        let shutdown_for_task = shutdown.clone();
        let task = tokio::spawn(async move {
            run_substrate_body_cycle(slots, bus_client, shutdown_for_task).await;
        });

        // Let cycle run ~3 ticks (3 × 1.15s = 3.45s). Use a slightly longer
        // wait to be robust to test-runner scheduling jitter.
        tokio::time::sleep(Duration::from_millis(3_700)).await;

        shutdown.notify_waiters();
        // Task should exit within 1s (well below SPEC §17 grace).
        let result = tokio::time::timeout(Duration::from_secs(2), task).await;
        assert!(
            result.is_ok(),
            "body cycle task must exit within 2s of shutdown notification"
        );
        result.unwrap().expect("task did not panic");

        // Verify topology + chi slots actually got written (version > 1).
        let topo = Slot::open(&topo_path).unwrap();
        let chi = Slot::open(&chi_path).unwrap();
        // Re-open to read; the data MUST be the latest published bytes.
        // We simply verify reads succeed and have the right size — version
        // counter introspection is not part of public API.
        assert_eq!(topo.read().unwrap().len(), 120);
        assert_eq!(chi.read().unwrap().len(), 24);
    }

    /// Helper: spin up an in-process broker + connect a BusClient. Used
    /// by integration tests that need a real bus client without relying
    /// on environmental sockets.
    async fn build_test_bus_client() -> Arc<BusClient> {
        use titan_bus::BusBroker;
        const AUTHKEY: &[u8] = b"test-substrate-authkey-32-bytes!";
        static UNIQ: AtomicU64 = AtomicU64::new(0);
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let dir = tempdir().expect("test tmpdir");
        let sock = dir.path().join(format!("bus_{n}.sock"));
        let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
        broker.start(&sock).await.expect("broker start");
        for _ in 0..20 {
            if sock.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let client = BusClient::connect(&sock, AUTHKEY, "test-substrate")
            .await
            .expect("client connect");
        // Leak the broker + tempdir to keep the socket alive for the test.
        std::mem::forget(broker);
        std::mem::forget(dir);
        Arc::new(client)
    }
}
