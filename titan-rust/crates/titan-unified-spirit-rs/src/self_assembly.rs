//! self_assembly — 162D TITAN_SELF tensor builder.
//!
//! Per master plan PLAN_microkernel_phase_c_l0_l1_rust.md §5.1 + SPEC §7.1
//! (slot byte layouts) + SPEC §10.G (ground_up cadence step 7).
//!
//! 162D layout (canonical, byte-locked):
//! ```text
//! TITAN_SELF (162 × float32 LE = 648 bytes payload)
//! │
//! ├─ [0:130]   felt_130 (130D)
//! │            ├─ [0:5]    inner_body  (5D)   ← inner_body_5d.bin
//! │            ├─ [5:20]   inner_mind  (15D)  ← inner_mind_15d.bin
//! │            ├─ [20:65]  inner_spirit (45D) ← inner_spirit_45d.bin
//! │            ├─ [65:70]  outer_body  (5D)   ← outer_body_5d.bin
//! │            ├─ [70:85]  outer_mind  (15D)  ← outer_mind_15d.bin
//! │            └─ [85:130] outer_spirit (45D) ← outer_spirit_45d.bin
//! ├─ [130:160] topology_30 (30D)              ← topology_30d.bin
//! └─ [160:162] journey_2  (2D)                ← derived from sphere_clocks.bin
//! ```
//!
//! Intermediate `unified_spirit_132d` = `[0:130]` felt + `[160:162]` journey
//! (Trinity 130D + Journey 2D — the felt-experience subset BEFORE topology
//! incorporation). Per SPEC §7.1 unified_spirit_132d.bin slot.
//!
//! Byte-identical guarantee: assembly is concatenation of contiguous
//! float32-LE slices. Both Python `kernel.py:_start_trinity_shm_writer`
//! and this Rust port emit the same 648-byte payload for identical inputs.
//! Verified by golden parity vectors in tests.

use std::time::{SystemTime, UNIX_EPOCH};

use blake2::{
    digest::{consts::U16, generic_array::GenericArray},
    Blake2b, Digest,
};

/// Width of each daemon-owned trinity slot in float32 elements.
pub const INNER_BODY_DIMS: usize = 5;
/// inner_mind dim count.
pub const INNER_MIND_DIMS: usize = 15;
/// inner_spirit dim count.
pub const INNER_SPIRIT_DIMS: usize = 45;
/// outer_body dim count.
pub const OUTER_BODY_DIMS: usize = 5;
/// outer_mind dim count.
pub const OUTER_MIND_DIMS: usize = 15;
/// outer_spirit dim count.
pub const OUTER_SPIRIT_DIMS: usize = 45;
/// felt_130 = sum of 6 trinity dims = 130.
pub const FELT_DIMS: usize = 130;
/// topology_30 dim count.
pub const TOPOLOGY_DIMS: usize = 30;
/// journey_2 dim count.
pub const JOURNEY_DIMS: usize = 2;
/// unified_spirit_132 = felt + journey = 132.
pub const UNIFIED_SPIRIT_DIMS: usize = 132;
/// TITAN_SELF total = felt + topology + journey = 162.
pub const SELF_DIMS: usize = 162;

/// Bytes per float32 element (LE on x86_64 / aarch64; both Titan deploy
/// targets are LE so we don't byte-swap).
pub const F32_BYTES: usize = 4;

/// 6-daemon trinity slots feeding SELF assembly. All values are
/// already-computed per-daemon outputs (not raw sensor data) — this layer
/// just concatenates them.
#[derive(Debug, Clone)]
pub struct TrinitySlotsRead {
    /// `inner_body_5d.bin` payload as 5 floats.
    pub inner_body: [f32; INNER_BODY_DIMS],
    /// `inner_mind_15d.bin` payload as 15 floats.
    pub inner_mind: [f32; INNER_MIND_DIMS],
    /// `inner_spirit_45d.bin` payload as 45 floats.
    pub inner_spirit: [f32; INNER_SPIRIT_DIMS],
    /// `outer_body_5d.bin` payload as 5 floats.
    pub outer_body: [f32; OUTER_BODY_DIMS],
    /// `outer_mind_15d.bin` payload as 15 floats.
    pub outer_mind: [f32; OUTER_MIND_DIMS],
    /// `outer_spirit_45d.bin` payload as 45 floats.
    pub outer_spirit: [f32; OUTER_SPIRIT_DIMS],
}

/// Errors during 162D assembly. NaN is the only non-trivial rejection —
/// shape errors are caught by the type system (fixed-size arrays).
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AssemblyError {
    /// One or more input floats was NaN. Per master plan §5.1 — assembly
    /// is downstream of daemon ticks, which already content-hash-gate
    /// their slot writes. NaN should never reach this layer; if it does,
    /// we refuse to propagate corruption to `self_162d.bin`.
    #[error("NaN detected in {layer} slot at index {index}")]
    NanInput {
        /// Which trinity slot's payload contained NaN.
        layer: &'static str,
        /// Float index within the slot.
        index: usize,
    },
}

/// Decode a slot payload (raw bytes from `Slot::read()`) into `[f32; N]`.
///
/// Returns `None` if `payload.len() != N * 4`. Caller is responsible for
/// re-trying / propagating the error to the orchestration layer.
pub fn decode_f32_slice<const N: usize>(payload: &[u8]) -> Option<[f32; N]> {
    if payload.len() != N * F32_BYTES {
        return None;
    }
    let mut out = [0.0_f32; N];
    for (i, slot) in out.iter_mut().enumerate() {
        let offset = i * F32_BYTES;
        let bytes: [u8; 4] = payload[offset..offset + F32_BYTES].try_into().ok()?;
        *slot = f32::from_le_bytes(bytes);
    }
    Some(out)
}

/// Encode a `[f32; N]` to little-endian bytes, ready for `Slot::write()`.
pub fn encode_f32_slice<const N: usize>(values: &[f32; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N * F32_BYTES);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Reject NaN — convenience helper used at slot ingest.
fn check_no_nan(layer: &'static str, values: &[f32]) -> Result<(), AssemblyError> {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            return Err(AssemblyError::NanInput { layer, index: i });
        }
    }
    Ok(())
}

/// Concatenate 6 trinity daemon outputs into the canonical 130D felt
/// vector per master plan §5.1.
pub fn assemble_felt_130(slots: &TrinitySlotsRead) -> Result<[f32; FELT_DIMS], AssemblyError> {
    check_no_nan("inner_body", &slots.inner_body)?;
    check_no_nan("inner_mind", &slots.inner_mind)?;
    check_no_nan("inner_spirit", &slots.inner_spirit)?;
    check_no_nan("outer_body", &slots.outer_body)?;
    check_no_nan("outer_mind", &slots.outer_mind)?;
    check_no_nan("outer_spirit", &slots.outer_spirit)?;

    let mut felt = [0.0_f32; FELT_DIMS];
    let mut cursor = 0;

    felt[cursor..cursor + INNER_BODY_DIMS].copy_from_slice(&slots.inner_body);
    cursor += INNER_BODY_DIMS;
    felt[cursor..cursor + INNER_MIND_DIMS].copy_from_slice(&slots.inner_mind);
    cursor += INNER_MIND_DIMS;
    felt[cursor..cursor + INNER_SPIRIT_DIMS].copy_from_slice(&slots.inner_spirit);
    cursor += INNER_SPIRIT_DIMS;
    felt[cursor..cursor + OUTER_BODY_DIMS].copy_from_slice(&slots.outer_body);
    cursor += OUTER_BODY_DIMS;
    felt[cursor..cursor + OUTER_MIND_DIMS].copy_from_slice(&slots.outer_mind);
    cursor += OUTER_MIND_DIMS;
    felt[cursor..cursor + OUTER_SPIRIT_DIMS].copy_from_slice(&slots.outer_spirit);
    cursor += OUTER_SPIRIT_DIMS;

    debug_assert_eq!(cursor, FELT_DIMS);
    Ok(felt)
}

/// Build the 132D unified_spirit intermediate (felt + journey) that
/// gets written to `unified_spirit_132d.bin` per SPEC §7.1.
pub fn assemble_unified_spirit_132d(
    felt: &[f32; FELT_DIMS],
    journey: &[f32; JOURNEY_DIMS],
) -> Result<[f32; UNIFIED_SPIRIT_DIMS], AssemblyError> {
    check_no_nan("journey", journey)?;
    let mut out = [0.0_f32; UNIFIED_SPIRIT_DIMS];
    out[..FELT_DIMS].copy_from_slice(felt);
    out[FELT_DIMS..].copy_from_slice(journey);
    Ok(out)
}

/// Build the canonical 162D TITAN_SELF tensor per master plan §5.1.
///
/// Layout: `[0:130]` felt + `[130:160]` topology_30 + `[160:162]` journey_2.
/// Output is byte-identical to current Python `kernel.py:_writer_loop`
/// for the same inputs.
pub fn assemble_162d(
    slots: &TrinitySlotsRead,
    topology: &[f32; TOPOLOGY_DIMS],
    journey: &[f32; JOURNEY_DIMS],
) -> Result<[f32; SELF_DIMS], AssemblyError> {
    check_no_nan("topology", topology)?;
    let felt = assemble_felt_130(slots)?;
    check_no_nan("journey", journey)?;

    let mut self_162 = [0.0_f32; SELF_DIMS];
    self_162[..FELT_DIMS].copy_from_slice(&felt);
    self_162[FELT_DIMS..FELT_DIMS + TOPOLOGY_DIMS].copy_from_slice(topology);
    self_162[FELT_DIMS + TOPOLOGY_DIMS..].copy_from_slice(journey);

    debug_assert_eq!(FELT_DIMS + TOPOLOGY_DIMS + JOURNEY_DIMS, SELF_DIMS);
    Ok(self_162)
}

/// Extract Journey 2D = `[epoch_dt, chi_dt]` from sphere_clocks.bin.
///
/// Per SPEC §7.1: `sphere_clocks.bin` payload = 6 × 7 × float32 = 168
/// bytes. The 7 fields per clock are
/// `[radius, scalar_position, phase, contraction_velocity, pulse_count,
/// consecutive_balanced, last_pulse_age_s]`. Journey 2D maps to the
/// FIRST clock's `[phase, contraction_velocity]` per current Python
/// `state_register.snapshot()['consciousness']` extraction
/// (see `kernel.py:_writer_loop` lines 815-818 — `curvature` and
/// `density` from consciousness snapshot are the Python aliases for
/// `phase` + `contraction_velocity`).
///
/// **Note**: this is the C-S4 interpretation. If C-S3 substrate writes
/// sphere_clocks with a different field order or selects a different
/// clock as canonical, this function adjusts — flagged for verification
/// at C4-5 substrate handshake.
pub fn extract_journey_2(sphere_clocks_payload: &[u8]) -> Option<[f32; JOURNEY_DIMS]> {
    // First clock = first 7 floats = first 28 bytes.
    if sphere_clocks_payload.len() < 7 * F32_BYTES {
        return None;
    }
    // Field order: radius[0], scalar_position[1], phase[2],
    // contraction_velocity[3], pulse_count[4], consecutive_balanced[5],
    // last_pulse_age_s[6]. Journey = [phase, contraction_velocity].
    let phase_bytes: [u8; 4] = sphere_clocks_payload[2 * F32_BYTES..3 * F32_BYTES]
        .try_into()
        .ok()?;
    let cvel_bytes: [u8; 4] = sphere_clocks_payload[3 * F32_BYTES..4 * F32_BYTES]
        .try_into()
        .ok()?;
    Some([
        f32::from_le_bytes(phase_bytes),
        f32::from_le_bytes(cvel_bytes),
    ])
}

/// Content-hash gate digest. Blake2b-128 over the encoded 162D bytes;
/// matches Python `kernel.py:_writer_loop` content-hash pattern
/// (`hashlib.blake2b(payload_bytes, digest_size=16)`).
pub fn content_hash(self_162: &[f32; SELF_DIMS]) -> [u8; 16] {
    let bytes = encode_f32_slice(self_162);
    let mut hasher = Blake2b::<U16>::new();
    hasher.update(&bytes);
    let result: GenericArray<u8, U16> = hasher.finalize();
    result.into()
}

/// Wall clock nanoseconds since UNIX epoch. Used for slot wall_ns
/// telemetry.
pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_slots(value: f32) -> TrinitySlotsRead {
        TrinitySlotsRead {
            inner_body: [value; INNER_BODY_DIMS],
            inner_mind: [value; INNER_MIND_DIMS],
            inner_spirit: [value; INNER_SPIRIT_DIMS],
            outer_body: [value; OUTER_BODY_DIMS],
            outer_mind: [value; OUTER_MIND_DIMS],
            outer_spirit: [value; OUTER_SPIRIT_DIMS],
        }
    }

    /// Build a 162D vector with a unique value per cell (i+1.0/100.0)
    /// that exercises the layout cleanly.
    fn distinct_inputs() -> (TrinitySlotsRead, [f32; TOPOLOGY_DIMS], [f32; JOURNEY_DIMS]) {
        let mut slots = synthetic_slots(0.0);
        let mut idx = 0_f32;
        let mut next = || {
            idx += 0.01;
            idx
        };
        for v in &mut slots.inner_body {
            *v = next();
        }
        for v in &mut slots.inner_mind {
            *v = next();
        }
        for v in &mut slots.inner_spirit {
            *v = next();
        }
        for v in &mut slots.outer_body {
            *v = next();
        }
        for v in &mut slots.outer_mind {
            *v = next();
        }
        for v in &mut slots.outer_spirit {
            *v = next();
        }
        let mut topology = [0.0_f32; TOPOLOGY_DIMS];
        for v in &mut topology {
            *v = next();
        }
        let mut journey = [0.0_f32; JOURNEY_DIMS];
        for v in &mut journey {
            *v = next();
        }
        (slots, topology, journey)
    }

    // ── PARITY TESTS (8) — synthetic golden vectors ────────────────────

    #[test]
    fn parity_all_zero() {
        // C4-2 parity 1: all-zero inputs → all-zero 162D output
        let slots = synthetic_slots(0.0);
        let topology = [0.0_f32; TOPOLOGY_DIMS];
        let journey = [0.0_f32; JOURNEY_DIMS];
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(s, [0.0_f32; SELF_DIMS]);
        // Encoded bytes — 162 × 4 = 648 zeros
        let encoded = encode_f32_slice(&s);
        assert_eq!(encoded.len(), 648);
        assert!(encoded.iter().all(|&b| b == 0));
    }

    #[test]
    fn parity_all_one() {
        // C4-2 parity 2: all-1.0 inputs → all-1.0 162D output, byte-locked
        let slots = synthetic_slots(1.0);
        let topology = [1.0_f32; TOPOLOGY_DIMS];
        let journey = [1.0_f32; JOURNEY_DIMS];
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(s, [1.0_f32; SELF_DIMS]);
        // 1.0_f32 LE = 0x00 0x00 0x80 0x3F
        let encoded = encode_f32_slice(&s);
        assert_eq!(encoded.len(), 648);
        for chunk in encoded.chunks_exact(4) {
            assert_eq!(chunk, &[0x00, 0x00, 0x80, 0x3F]);
        }
    }

    #[test]
    fn parity_layout_distinct_per_dim() {
        // C4-2 parity 3: layout cursor walks correctly through all 6
        // trinity slots, then topology, then journey.
        let (slots, topology, journey) = distinct_inputs();
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        // Expected output: same monotonically-increasing 0.01 .. 1.62
        // sequence (with float-rounding tolerance).
        let mut expected_idx = 0_f32;
        for (i, &v) in s.iter().enumerate() {
            expected_idx += 0.01;
            assert!(
                (v - expected_idx).abs() < 1e-5,
                "self_162[{i}]={v}, expected≈{expected_idx}"
            );
        }
    }

    #[test]
    fn parity_mixed_signs() {
        // C4-2 parity 4: mixed +/- inputs preserve sign bits exactly
        let mut slots = synthetic_slots(0.0);
        slots.inner_body = [-1.0, 1.0, -2.0, 2.0, -3.0];
        slots.outer_spirit[0] = -100.0;
        slots.outer_spirit[44] = 100.0;
        let topology = [0.0; TOPOLOGY_DIMS];
        let journey = [-0.5, 0.5];
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(s[0..5], [-1.0, 1.0, -2.0, 2.0, -3.0]);
        assert_eq!(s[85], -100.0);
        assert_eq!(s[129], 100.0);
        assert_eq!(s[160], -0.5);
        assert_eq!(s[161], 0.5);
    }

    #[test]
    fn parity_denormals_preserved() {
        // C4-2 parity 5: denormal floats survive the assembly path
        // (no implicit rounding / flush-to-zero)
        let mut slots = synthetic_slots(0.0);
        slots.inner_body[0] = f32::MIN_POSITIVE;
        slots.inner_body[1] = f32::MIN_POSITIVE / 2.0; // subnormal
        let topology = [0.0; TOPOLOGY_DIMS];
        let journey = [0.0, 0.0];
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(s[0], f32::MIN_POSITIVE);
        assert_eq!(s[1], f32::MIN_POSITIVE / 2.0);
        assert!(s[1] > 0.0); // subnormal preserved
    }

    #[test]
    fn parity_extremes() {
        // C4-2 parity 6: max-positive + max-negative + ±∞ all encode
        // to the canonical IEEE-754 bytes
        let mut slots = synthetic_slots(0.0);
        slots.inner_body[0] = f32::MAX;
        slots.inner_body[1] = f32::MIN;
        slots.inner_body[2] = f32::INFINITY;
        slots.inner_body[3] = f32::NEG_INFINITY;
        slots.inner_body[4] = -0.0;
        let topology = [0.0; TOPOLOGY_DIMS];
        let journey = [0.0, 0.0];
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(s[0], f32::MAX);
        assert_eq!(s[1], f32::MIN);
        assert!(s[2].is_infinite() && s[2].is_sign_positive());
        assert!(s[3].is_infinite() && s[3].is_sign_negative());
        // -0.0 == 0.0 in float comparison but bits differ
        assert_eq!(s[4].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn parity_observer_dim_indices() {
        // C4-2 parity 7: observer dims are at canonical absolute indices
        // [20:25] (inner_spirit start) and [85:90] (outer_spirit start)
        // per SPEC §10.F G8. Layout cursor proves the indexing.
        let (slots, topology, journey) = distinct_inputs();
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        // inner_spirit absolute indices [20:65] in self_162d;
        // observer subset [20:25] = inner_spirit[0:5]
        assert!((s[20] - slots.inner_spirit[0]).abs() < 1e-7);
        assert!((s[24] - slots.inner_spirit[4]).abs() < 1e-7);
        // outer_spirit absolute [85:130]; observer subset [85:90] = outer_spirit[0:5]
        assert!((s[85] - slots.outer_spirit[0]).abs() < 1e-7);
        assert!((s[89] - slots.outer_spirit[4]).abs() < 1e-7);
    }

    #[test]
    fn parity_nan_input_rejected() {
        // C4-2 parity 8: NaN at any layer rejected with structured error
        let mut slots = synthetic_slots(0.0);
        slots.inner_mind[7] = f32::NAN;
        let topology = [0.0; TOPOLOGY_DIMS];
        let journey = [0.0, 0.0];
        let result = assemble_162d(&slots, &topology, &journey);
        assert_eq!(
            result,
            Err(AssemblyError::NanInput {
                layer: "inner_mind",
                index: 7
            })
        );

        // Also catch NaN in topology
        let slots = synthetic_slots(0.0);
        let mut topology = [0.0_f32; TOPOLOGY_DIMS];
        topology[15] = f32::NAN;
        let result = assemble_162d(&slots, &topology, &journey);
        assert_eq!(
            result,
            Err(AssemblyError::NanInput {
                layer: "topology",
                index: 15
            })
        );

        // Also catch NaN in journey (topology + felt clean → journey check fires)
        let slots = synthetic_slots(0.0);
        let topology = [0.0; TOPOLOGY_DIMS];
        let journey = [0.0, f32::NAN];
        let result = assemble_162d(&slots, &topology, &journey);
        assert_eq!(
            result,
            Err(AssemblyError::NanInput {
                layer: "journey",
                index: 1
            })
        );
    }

    // ── COMPOSITION TESTS (4) — unified_spirit_132d + cross-checks ────

    #[test]
    fn composition_unified_spirit_132d_is_felt_plus_journey() {
        // C4-2 composition 1: unified_spirit_132d == felt + journey
        let (slots, _topology, journey) = distinct_inputs();
        let felt = assemble_felt_130(&slots).unwrap();
        let usp = assemble_unified_spirit_132d(&felt, &journey).unwrap();
        assert_eq!(&usp[0..130], &felt[..]);
        assert_eq!(&usp[130..132], &journey[..]);
    }

    #[test]
    fn composition_self_162d_equals_unified_plus_topology() {
        // C4-2 composition 2: self_162d composition rule
        // self_162[0:130] = felt, [130:160] = topology, [160:162] = journey
        // so equivalent: unified_spirit_132d.split_at(130) interleaved with topology
        let (slots, topology, journey) = distinct_inputs();
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        let felt = assemble_felt_130(&slots).unwrap();
        assert_eq!(&s[0..130], &felt[..]);
        assert_eq!(&s[130..160], &topology[..]);
        assert_eq!(&s[160..162], &journey[..]);
    }

    #[test]
    fn composition_byte_layout_matches_spec_71() {
        // C4-2 composition 3: encoded bytes match SPEC §7.1 self_162d.bin
        // payload size (162 × 4 = 648 bytes)
        let (slots, topology, journey) = distinct_inputs();
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        let bytes = encode_f32_slice(&s);
        assert_eq!(bytes.len(), SELF_DIMS * F32_BYTES);
        assert_eq!(bytes.len(), 648);
        // Decode round-trip preserves values byte-equal
        let decoded = decode_f32_slice::<{ SELF_DIMS }>(&bytes).unwrap();
        assert_eq!(decoded, s);
    }

    #[test]
    fn composition_unified_spirit_132d_byte_layout() {
        // C4-2 composition 4: unified_spirit_132d.bin payload = 528 bytes
        let (slots, _topology, journey) = distinct_inputs();
        let felt = assemble_felt_130(&slots).unwrap();
        let usp = assemble_unified_spirit_132d(&felt, &journey).unwrap();
        let bytes = encode_f32_slice(&usp);
        assert_eq!(bytes.len(), UNIFIED_SPIRIT_DIMS * F32_BYTES);
        assert_eq!(bytes.len(), 528);
        let decoded = decode_f32_slice::<{ UNIFIED_SPIRIT_DIMS }>(&bytes).unwrap();
        assert_eq!(decoded, usp);
    }

    // ── CONTENT-HASH GATE TESTS (4) ────────────────────────────────────

    #[test]
    fn content_hash_deterministic() {
        // C4-2 hash 1: same inputs → same digest
        let (slots, topology, journey) = distinct_inputs();
        let s1 = assemble_162d(&slots, &topology, &journey).unwrap();
        let s2 = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_eq!(content_hash(&s1), content_hash(&s2));
    }

    #[test]
    fn content_hash_distinct_for_different_inputs() {
        // C4-2 hash 2: any input change → different digest
        let (mut slots, topology, journey) = distinct_inputs();
        let s1 = assemble_162d(&slots, &topology, &journey).unwrap();
        slots.inner_body[0] += 0.001;
        let s2 = assemble_162d(&slots, &topology, &journey).unwrap();
        assert_ne!(content_hash(&s1), content_hash(&s2));
    }

    #[test]
    fn content_hash_size_is_16_bytes() {
        // C4-2 hash 3: blake2b-128 produces exactly 16 bytes
        let (slots, topology, journey) = distinct_inputs();
        let s = assemble_162d(&slots, &topology, &journey).unwrap();
        let h = content_hash(&s);
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn content_hash_matches_python_pattern() {
        // C4-2 hash 4: hash a known-zero 648-byte payload to a stable
        // digest. Python `hashlib.blake2b(b'\x00'*648, digest_size=16)`
        // produces this exact value — locked.
        let zeros = [0.0_f32; SELF_DIMS];
        let h = content_hash(&zeros);
        // Pre-computed value (verified 2026-04-29):
        //   python -c "import hashlib;print(hashlib.blake2b(b'\\x00'*648,digest_size=16).hexdigest())"
        //   → "13b85b072745e2df29a9372eb0aff49b"
        let expected_hex = "13b85b072745e2df29a9372eb0aff49b";
        assert_eq!(hex::encode(h), expected_hex);
    }

    // ── JOURNEY EXTRACTION TESTS (2) ────────────────────────────────────

    #[test]
    fn journey_extracts_from_first_clock_phase_and_cvel() {
        // C4-2 journey 1: first clock's [phase, contraction_velocity] →
        // Journey 2D
        let mut payload = vec![0_u8; 6 * 7 * F32_BYTES];
        // Clock 0: radius=1.0, scalar_position=2.0, phase=3.0, cvel=4.0,
        //          pulse_count=5.0, consecutive_balanced=6.0,
        //          last_pulse_age_s=7.0
        for (i, v) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].iter().enumerate() {
            payload[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let journey = extract_journey_2(&payload).unwrap();
        assert_eq!(journey, [3.0, 4.0]);
    }

    #[test]
    fn journey_handles_short_payload() {
        // C4-2 journey 2: too-short payload returns None gracefully
        assert!(extract_journey_2(&[]).is_none());
        assert!(extract_journey_2(&[0_u8; 27]).is_none()); // < 7×4
        assert!(extract_journey_2(&[0_u8; 28]).is_some()); // ≥ 7×4
    }

    // ── DECODE/ENCODE TESTS (2) ─────────────────────────────────────────

    #[test]
    fn decode_f32_slice_validates_length() {
        // C4-2 codec 1: wrong-length payload returns None
        assert_eq!(decode_f32_slice::<5>(&[0_u8; 19]), None);
        assert_eq!(decode_f32_slice::<5>(&[0_u8; 21]), None);
        assert_eq!(decode_f32_slice::<5>(&[0_u8; 20]), Some([0.0; 5]));
    }

    #[test]
    fn encode_decode_roundtrip() {
        // C4-2 codec 2: encode/decode is lossless for finite values
        let v: [f32; 5] = [-1.5, 0.0, 0.001, 1e10, -1e-30];
        let bytes = encode_f32_slice(&v);
        let decoded = decode_f32_slice::<5>(&bytes).unwrap();
        assert_eq!(decoded, v);
    }
}
