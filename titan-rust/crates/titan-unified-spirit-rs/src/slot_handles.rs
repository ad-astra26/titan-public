//! slot_handles — Open the 9 shm slots unified-spirit reads + writes.
//!
//! Per SPEC §9.A unified-spirit-rs row + §7.1 layouts. Slots are
//! kernel-created at boot per C-S2 SLOT_SPECS (`SlotCreator::Kernel`);
//! unified-spirit OPENS them — never CREATES.
//!
//! - **Reads** (7 slots): inner_body_5d, inner_mind_15d, inner_spirit_45d,
//!   outer_body_5d, outer_mind_15d, outer_spirit_45d, topology_30d,
//!   sphere_clocks (8 reads — 7 daemon + sphere_clocks for Journey 2D).
//! - **Writes** (2 slots): unified_spirit_132d, self_162d.
//!
//! Open semantics per `titan-state::Slot::open`: maps the entire mmap
//! region; does not validate header (boot integrity check is the
//! orchestration layer's job). Errors propagate as `UnifiedSpiritExitCode::ShmOpenFailure`
//! (SPEC §15 code 5) at boot.

use std::path::{Path, PathBuf};

use titan_state::Slot;

use crate::self_assembly::{
    self, AssemblyError, TrinitySlotsRead, INNER_BODY_DIMS, INNER_MIND_DIMS, INNER_SPIRIT_DIMS,
    JOURNEY_DIMS, OUTER_BODY_DIMS, OUTER_MIND_DIMS, OUTER_SPIRIT_DIMS, SELF_DIMS, TOPOLOGY_DIMS,
    UNIFIED_SPIRIT_DIMS,
};

/// Errors during slot open / read / write.
#[derive(Debug, thiserror::Error)]
pub enum SlotHandleError {
    /// Slot file missing — kernel did not create it. Surface as
    /// `ExitCode::ShmOpenFailure`.
    #[error("slot {name:?} open failed at {path:?}: {source}")]
    Open {
        /// Slot canonical filename.
        name: &'static str,
        /// Full path attempted.
        path: PathBuf,
        /// Underlying io / mmap error.
        #[source]
        source: titan_state::SlotIoError,
    },
    /// SeqLock read returned a payload of unexpected size. Likely
    /// corruption — caller should escalate.
    #[error("slot {name:?} read returned {actual} bytes, expected {expected}")]
    ReadShape {
        /// Slot canonical filename.
        name: &'static str,
        /// Bytes actually returned.
        actual: usize,
        /// Bytes expected per SPEC §7.1.
        expected: usize,
    },
    /// SeqLock read failed (3-retry cap reached, or io error).
    #[error("slot {name:?} read failed: {source}")]
    Read {
        /// Slot canonical filename.
        name: &'static str,
        /// Underlying io / SeqLock error.
        #[source]
        source: titan_state::SlotIoError,
    },
    /// SeqLock write failed.
    #[error("slot {name:?} write failed: {source}")]
    Write {
        /// Slot canonical filename.
        name: &'static str,
        /// Underlying io / SeqLock error.
        #[source]
        source: titan_state::SlotIoError,
    },
    /// Float-decode failed (wrong byte length per slot spec). Reach this
    /// only if SPEC §7.1 byte counts drift from runtime — should be
    /// caught by `arch_map phase-c verify --strict`.
    #[error("slot {name:?} payload decode failed (expected {expected} f32 elements)")]
    Decode {
        /// Slot canonical filename.
        name: &'static str,
        /// Expected float element count.
        expected: usize,
    },
    /// Assembly error surfacing from `self_assembly` (NaN input).
    #[error("assembly error: {0}")]
    Assembly(#[from] AssemblyError),
}

/// Bundle of opened slot handles. Reader slots are immutable; writer
/// slots are owned mutably (one writer per slot per SPEC §7).
pub struct SlotHandles {
    /// `inner_body_5d.bin` (read-only from unified-spirit's perspective —
    /// kernel-created, daemon-written).
    pub inner_body: Slot,
    /// `inner_mind_15d.bin`.
    pub inner_mind: Slot,
    /// `inner_spirit_45d.bin`.
    pub inner_spirit: Slot,
    /// `outer_body_5d.bin`.
    pub outer_body: Slot,
    /// `outer_mind_15d.bin`.
    pub outer_mind: Slot,
    /// `outer_spirit_45d.bin`.
    pub outer_spirit: Slot,
    /// `topology_30d.bin` — written by trinity-substrate.
    pub topology: Slot,
    /// `sphere_clocks.bin` — written by trinity-substrate; first 7 floats
    /// = clock 0; Journey 2D = `[clock0.phase, clock0.contraction_velocity]`.
    pub sphere_clocks: Slot,
    /// `unified_spirit_132d.bin` — unified-spirit canonical writer.
    pub unified_spirit_132d: Slot,
    /// `self_162d.bin` — unified-spirit canonical writer (THE published
    /// 162D TITAN_SELF tensor that Python L2 consumers read).
    pub self_162d: Slot,
}

impl SlotHandles {
    /// Open all 10 slots from `<shm_dir>`. Returns `SlotHandleError::Open`
    /// on the first slot that fails to open.
    ///
    /// `shm_dir` typically = `/dev/shm/titan_<id>/`. Caller resolves it
    /// from CLI / env per SPEC §5.
    pub fn open_all(shm_dir: &Path) -> Result<Self, SlotHandleError> {
        let open = |name: &'static str| -> Result<Slot, SlotHandleError> {
            let path = shm_dir.join(name);
            Slot::open(&path).map_err(|source| SlotHandleError::Open { name, path, source })
        };

        Ok(SlotHandles {
            inner_body: open("inner_body_5d.bin")?,
            inner_mind: open("inner_mind_15d.bin")?,
            inner_spirit: open("inner_spirit_45d.bin")?,
            outer_body: open("outer_body_5d.bin")?,
            outer_mind: open("outer_mind_15d.bin")?,
            outer_spirit: open("outer_spirit_45d.bin")?,
            topology: open("topology_30d.bin")?,
            sphere_clocks: open("sphere_clocks.bin")?,
            unified_spirit_132d: open("unified_spirit_132d.bin")?,
            self_162d: open("self_162d.bin")?,
        })
    }

    /// Read all 6 daemon slots into a [`TrinitySlotsRead`] bundle.
    /// Each `Slot::read()` retries internally up to 3 times per SPEC §7.0
    /// SeqLock semantics; if any individual slot fails after retries,
    /// surface that error.
    pub fn read_trinity(&self) -> Result<TrinitySlotsRead, SlotHandleError> {
        let inner_body = decode_slot::<{ INNER_BODY_DIMS }>(&self.inner_body, "inner_body_5d.bin")?;
        let inner_mind =
            decode_slot::<{ INNER_MIND_DIMS }>(&self.inner_mind, "inner_mind_15d.bin")?;
        let inner_spirit =
            decode_slot::<{ INNER_SPIRIT_DIMS }>(&self.inner_spirit, "inner_spirit_45d.bin")?;
        let outer_body = decode_slot::<{ OUTER_BODY_DIMS }>(&self.outer_body, "outer_body_5d.bin")?;
        let outer_mind =
            decode_slot::<{ OUTER_MIND_DIMS }>(&self.outer_mind, "outer_mind_15d.bin")?;
        let outer_spirit =
            decode_slot::<{ OUTER_SPIRIT_DIMS }>(&self.outer_spirit, "outer_spirit_45d.bin")?;
        Ok(TrinitySlotsRead {
            inner_body,
            inner_mind,
            inner_spirit,
            outer_body,
            outer_mind,
            outer_spirit,
        })
    }

    /// Read topology_30d slot.
    pub fn read_topology(&self) -> Result<[f32; TOPOLOGY_DIMS], SlotHandleError> {
        decode_slot::<{ TOPOLOGY_DIMS }>(&self.topology, "topology_30d.bin")
    }

    /// Read sphere_clocks slot raw bytes (caller extracts Journey 2D).
    pub fn read_sphere_clocks(&self) -> Result<Vec<u8>, SlotHandleError> {
        self.sphere_clocks
            .read()
            .map_err(|source| SlotHandleError::Read {
                name: "sphere_clocks.bin",
                source,
            })
    }

    /// Read Journey 2D from sphere_clocks (helper).
    pub fn read_journey(&self) -> Result<[f32; JOURNEY_DIMS], SlotHandleError> {
        let payload = self.read_sphere_clocks()?;
        self_assembly::extract_journey_2(&payload).ok_or(SlotHandleError::Decode {
            name: "sphere_clocks.bin",
            expected: 7, // 7 floats minimum for first clock
        })
    }

    /// Write the unified_spirit 132D payload (felt + journey).
    /// Mutates `self.unified_spirit_132d`'s SeqLock.
    pub fn write_unified_spirit_132d(
        &mut self,
        payload: &[f32; UNIFIED_SPIRIT_DIMS],
    ) -> Result<(), SlotHandleError> {
        let bytes = self_assembly::encode_f32_slice(payload);
        self.unified_spirit_132d
            .write(&bytes)
            .map_err(|source| SlotHandleError::Write {
                name: "unified_spirit_132d.bin",
                source,
            })
    }

    /// Write the canonical 162D TITAN_SELF tensor.
    /// Mutates `self.self_162d`'s SeqLock.
    pub fn write_self_162d(&mut self, payload: &[f32; SELF_DIMS]) -> Result<(), SlotHandleError> {
        let bytes = self_assembly::encode_f32_slice(payload);
        self.self_162d
            .write(&bytes)
            .map_err(|source| SlotHandleError::Write {
                name: "self_162d.bin",
                source,
            })
    }
}

/// Read a slot + decode into `[f32; N]`. Wraps `Slot::read()`'s SeqLock
/// retry + propagates a structured error.
fn decode_slot<const N: usize>(
    slot: &Slot,
    name: &'static str,
) -> Result<[f32; N], SlotHandleError> {
    let payload = slot
        .read()
        .map_err(|source| SlotHandleError::Read { name, source })?;
    self_assembly::decode_f32_slice::<N>(&payload).ok_or(SlotHandleError::ReadShape {
        name,
        actual: payload.len(),
        expected: N * self_assembly::F32_BYTES,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    use crate::self_assembly::{encode_f32_slice, F32_BYTES};

    /// Build a minimal kernel-side simulation: create all 10 slots in a
    /// temp dir using `titan-state::Slot::create`. Caller mutates payloads
    /// before tests.
    fn create_all_slots(dir: &Path) -> Result<(), titan_state::SlotIoError> {
        Slot::create(dir.join("inner_body_5d.bin"), 1, 5 * 4)?;
        Slot::create(dir.join("inner_mind_15d.bin"), 1, 15 * 4)?;
        Slot::create(dir.join("inner_spirit_45d.bin"), 1, 45 * 4)?;
        Slot::create(dir.join("outer_body_5d.bin"), 1, 5 * 4)?;
        Slot::create(dir.join("outer_mind_15d.bin"), 1, 15 * 4)?;
        Slot::create(dir.join("outer_spirit_45d.bin"), 1, 45 * 4)?;
        Slot::create(dir.join("topology_30d.bin"), 1, 30 * 4)?;
        Slot::create(dir.join("sphere_clocks.bin"), 1, 6 * 7 * 4)?;
        Slot::create(dir.join("unified_spirit_132d.bin"), 1, 132 * 4)?;
        Slot::create(dir.join("self_162d.bin"), 1, 162 * 4)?;
        Ok(())
    }

    /// Helper: write a payload to an existing slot file.
    fn write_payload(dir: &Path, name: &str, payload: &[u8]) {
        let path = dir.join(name);
        let mut slot = Slot::open(&path).expect("open existing slot");
        slot.write(payload).expect("write slot");
    }

    #[test]
    fn open_all_succeeds_when_kernel_created_slots() {
        // C4-2 SeqLock 1: open_all succeeds against a kernel-style layout
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();
        let handles = SlotHandles::open_all(dir.path()).unwrap();
        // All 10 slot handles present
        let _ = (
            &handles.inner_body,
            &handles.inner_mind,
            &handles.inner_spirit,
            &handles.outer_body,
            &handles.outer_mind,
            &handles.outer_spirit,
            &handles.topology,
            &handles.sphere_clocks,
            &handles.unified_spirit_132d,
            &handles.self_162d,
        );
    }

    #[test]
    fn open_all_fails_when_slot_missing() {
        // C4-2 SeqLock 2: missing slot surfaces SlotHandleError::Open
        let dir = tempdir().unwrap();
        // Create only some slots — leave self_162d.bin missing
        Slot::create(dir.path().join("inner_body_5d.bin"), 1, 5 * 4).unwrap();
        let result = SlotHandles::open_all(dir.path());
        assert!(matches!(result, Err(SlotHandleError::Open { .. })));
    }

    #[test]
    fn read_trinity_returns_decoded_arrays() {
        // C4-2 SeqLock 3: read_trinity correctly decodes per-slot payloads
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();

        // Set inner_body to all 1.0
        let inner_body_payload = encode_f32_slice(&[1.0_f32; 5]);
        write_payload(dir.path(), "inner_body_5d.bin", &inner_body_payload);
        // Set outer_spirit[0] = 7.0
        let mut outer_spirit_arr = [0.0_f32; 45];
        outer_spirit_arr[0] = 7.0;
        let outer_spirit_payload = encode_f32_slice(&outer_spirit_arr);
        write_payload(dir.path(), "outer_spirit_45d.bin", &outer_spirit_payload);

        let handles = SlotHandles::open_all(dir.path()).unwrap();
        let trinity = handles.read_trinity().unwrap();
        assert_eq!(trinity.inner_body, [1.0; 5]);
        assert_eq!(trinity.outer_spirit[0], 7.0);
        assert_eq!(trinity.outer_spirit[1], 0.0);
    }

    #[test]
    fn write_self_162d_round_trips_seqlock() {
        // C4-2 SeqLock 4: SeqLock writer + reader round-trip preserves
        // 162D payload; header has correct payload_bytes; seq is even
        // after write
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();

        let mut payload = [0.0_f32; SELF_DIMS];
        for (i, v) in payload.iter_mut().enumerate() {
            *v = (i as f32) * 0.5;
        }
        handles.write_self_162d(&payload).unwrap();

        // Header sanity (§7.0 v1.0.0): fixed header has schema + capacity;
        // per-buffer metadata has payload_bytes (varies with each publish).
        let h = handles.self_162d.header();
        assert_eq!(h.schema_version, 1);
        assert_eq!(h.payload_capacity, (SELF_DIMS * F32_BYTES) as u32);
        assert!(h.is_initialized(), "version must be > 0 after first write");
        assert!(h.ready_idx_valid(), "ready_idx must be in [0, 2]");
        let buf_meta = handles.self_162d.buffer_meta(h.ready_idx());
        assert_eq!(buf_meta.payload_bytes, (SELF_DIMS * F32_BYTES) as u32);

        // Read back via underlying Slot::read and decode
        let bytes = handles.self_162d.read().unwrap();
        let decoded = self_assembly::decode_f32_slice::<{ SELF_DIMS }>(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn read_journey_extracts_first_clock_phase_and_cvel() {
        // C4-2 orchestration 1: read_journey → first clock phase + cvel
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();

        let mut sphere_payload = vec![0_u8; 6 * 7 * F32_BYTES];
        // Clock 0: radius=0.5, scalar_position=1.5, phase=2.5, cvel=3.5,
        // pulse_count=4.5, consecutive_balanced=5.5, last_pulse_age_s=6.5
        for (i, v) in [0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5].iter().enumerate() {
            sphere_payload[i * F32_BYTES..(i + 1) * F32_BYTES].copy_from_slice(&v.to_le_bytes());
        }
        write_payload(dir.path(), "sphere_clocks.bin", &sphere_payload);

        let handles = SlotHandles::open_all(dir.path()).unwrap();
        let journey = handles.read_journey().unwrap();
        assert_eq!(journey, [2.5, 3.5]);
    }

    #[test]
    fn read_topology_returns_30_floats() {
        // C4-2 orchestration 2: read_topology decodes 30D
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();

        let mut topo = [0.0_f32; TOPOLOGY_DIMS];
        for (i, v) in topo.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let topo_payload = encode_f32_slice(&topo);
        write_payload(dir.path(), "topology_30d.bin", &topo_payload);

        let handles = SlotHandles::open_all(dir.path()).unwrap();
        let topology = handles.read_topology().unwrap();
        assert_eq!(topology, topo);
    }

    #[test]
    fn write_unified_spirit_132d_payload_size_528() {
        // C4-2 orchestration 3: unified_spirit_132d.bin payload = 528 bytes
        let dir = tempdir().unwrap();
        create_all_slots(dir.path()).unwrap();
        let mut handles = SlotHandles::open_all(dir.path()).unwrap();

        let payload = [0.5_f32; UNIFIED_SPIRIT_DIMS];
        handles.write_unified_spirit_132d(&payload).unwrap();
        let h = handles.unified_spirit_132d.header();
        // §7.0 v1.0.0: fixed header carries payload_capacity; per-buffer
        // metadata carries the actual payload_bytes per publish.
        assert_eq!(h.payload_capacity, (UNIFIED_SPIRIT_DIMS * F32_BYTES) as u32);
        assert_eq!(h.payload_capacity, 528);
        let buf_meta = handles.unified_spirit_132d.buffer_meta(h.ready_idx());
        assert_eq!(buf_meta.payload_bytes, 528);
    }

    #[test]
    fn read_shape_error_when_payload_size_wrong() {
        // C4-2 orchestration 4: a slot whose physical payload size doesn't
        // match the expected float count surfaces SlotHandleError::ReadShape
        let dir = tempdir().unwrap();
        // Create inner_body slot with 4 floats instead of 5 (wrong)
        Slot::create(dir.path().join("wrong_size.bin"), 1, 4 * F32_BYTES as u32).unwrap();
        let mut slot = Slot::open(dir.path().join("wrong_size.bin")).unwrap();
        slot.write(&[0_u8; 16]).unwrap();
        // Re-open to pick up the write
        let slot = Slot::open(dir.path().join("wrong_size.bin")).unwrap();
        let result = decode_slot::<5>(&slot, "wrong_size.bin");
        assert!(matches!(result, Err(SlotHandleError::ReadShape { .. })));
    }
}
