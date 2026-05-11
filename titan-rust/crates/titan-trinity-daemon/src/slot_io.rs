//! slot_io — Typed read/write helpers wrapping `titan-state::Slot`.
//!
//! Daemons work with fixed-dim float32 payloads:
//! - inner_body_5d / outer_body_5d: `[f32; 5]` (20 bytes)
//! - inner_mind_15d / outer_mind_15d: `[f32; 15]` (60 bytes)
//! - inner_spirit_45d / outer_spirit_45d: `[f32; 45]` (180 bytes)
//! - topology_30d: `[f32; 30]` (120 bytes; daemons read 10D slices)
//!
//! All multi-byte values are **little-endian** per SPEC §7.4 endianness lock
//! (`<` Python struct format = native LE on x86_64).

use std::path::Path;

use titan_state::Slot;

use crate::error::{DaemonError, DaemonResult};

/// Read a fixed-dim float32 payload from an open slot.
///
/// `N` is the number of float32 dims (e.g. 5 / 15 / 45). The decoded
/// payload size MUST equal `N * 4` bytes; otherwise a [`DaemonError::DimMismatch`]
/// is returned.
pub fn read_dim_slice<const N: usize>(slot: &Slot) -> DaemonResult<[f32; N]> {
    let payload = slot
        .read()
        .map_err(|source| DaemonError::ShmRead { source })?;
    decode_floats::<N>(&payload)
}

/// Write a fixed-dim float32 payload to an open slot.
pub fn write_dim_slice<const N: usize>(slot: &mut Slot, values: &[f32; N]) -> DaemonResult<()> {
    let bytes = encode_floats::<N>(values);
    slot.write(&bytes)
        .map_err(|source| DaemonError::ShmWrite { source })
}

/// Open an existing kernel-created slot file by absolute path.
pub fn open_slot(path: impl AsRef<Path>) -> DaemonResult<Slot> {
    Slot::open(path).map_err(|source| DaemonError::ShmOpen { source })
}

/// Decode `N`-element float32-LE payload from a byte buffer.
/// Public for parity-test reuse.
pub fn decode_floats<const N: usize>(payload: &[u8]) -> DaemonResult<[f32; N]> {
    let expected = N * 4;
    if payload.len() != expected {
        return Err(DaemonError::DimMismatch {
            expected: N,
            expected_bytes: expected,
            actual_bytes: payload.len(),
        });
    }
    let mut out = [0.0_f32; N];
    for (i, chunk) in payload.chunks_exact(4).enumerate() {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        out[i] = f32::from_le_bytes(buf);
    }
    Ok(out)
}

/// Encode `N`-element float32 array to little-endian bytes.
/// Public for parity-test reuse.
pub fn encode_floats<const N: usize>(values: &[f32; N]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(N * 4);
    for v in values.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Read the inner-lower 10D slice from `topology_30d.bin`.
///
/// Per SPEC G4 byte layout: `topology_30d[10:20]` = inner_lower (10D).
/// Inner daemons (body / mind) consume this for ground_up.
pub fn read_topology_inner_lower(slot: &Slot) -> DaemonResult<[f32; 10]> {
    let topology = read_dim_slice::<30>(slot)?;
    let mut out = [0.0_f32; 10];
    out.copy_from_slice(&topology[10..20]);
    Ok(out)
}

/// Read the outer-lower 10D slice from `topology_30d.bin`.
///
/// Per SPEC G4 byte layout: `topology_30d[0:10]` = outer_lower (10D).
/// Outer daemons (body / mind) consume this for ground_up. Provided here
/// for C-S6 reuse.
pub fn read_topology_outer_lower(slot: &Slot) -> DaemonResult<[f32; 10]> {
    let topology = read_dim_slice::<30>(slot)?;
    let mut out = [0.0_f32; 10];
    out.copy_from_slice(&topology[0..10]);
    Ok(out)
}

/// Read the whole/unified-spirit 10D slice from `topology_30d.bin`.
///
/// Per SPEC G4: `topology_30d[20:30]` = whole topology (10D). Used by
/// titan-unified-spirit-rs (C-S4); included here for crate completeness.
pub fn read_topology_whole(slot: &Slot) -> DaemonResult<[f32; 10]> {
    let topology = read_dim_slice::<30>(slot)?;
    let mut out = [0.0_f32; 10];
    out.copy_from_slice(&topology[20..30]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use titan_core::constants::{
        INNER_BODY_5D_SCHEMA_VERSION, INNER_MIND_15D_SCHEMA_VERSION,
        INNER_SPIRIT_45D_SCHEMA_VERSION, TOPOLOGY_30D_SCHEMA_VERSION,
    };

    fn make_slot(name: &str, schema: u32, payload_bytes: u32) -> (tempfile::TempDir, Slot) {
        let dir = tempdir().unwrap();
        let path = dir.path().join(name);
        let slot = Slot::create(&path, schema, payload_bytes).unwrap();
        (dir, slot)
    }

    #[test]
    fn write_then_read_5d_round_trip() {
        let (_d, mut slot) = make_slot(
            "inner_body_5d.bin",
            INNER_BODY_5D_SCHEMA_VERSION as u32,
            5 * 4,
        );
        let payload: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
        write_dim_slice(&mut slot, &payload).unwrap();
        let read = read_dim_slice::<5>(&slot).unwrap();
        for i in 0..5 {
            assert!((read[i] - payload[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn write_then_read_15d_round_trip() {
        let (_d, mut slot) = make_slot(
            "inner_mind_15d.bin",
            INNER_MIND_15D_SCHEMA_VERSION as u32,
            15 * 4,
        );
        let payload: [f32; 15] = [
            0.0, 0.1, 0.2, 0.3, 0.4, // thinking
            0.5, 0.6, 0.7, 0.8, 0.9, // feeling
            1.0, 0.9, 0.8, 0.7, 0.6, // willing
        ];
        write_dim_slice(&mut slot, &payload).unwrap();
        let read = read_dim_slice::<15>(&slot).unwrap();
        for i in 0..15 {
            assert!((read[i] - payload[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn write_then_read_45d_round_trip() {
        let (_d, mut slot) = make_slot(
            "inner_spirit_45d.bin",
            INNER_SPIRIT_45D_SCHEMA_VERSION as u32,
            45 * 4,
        );
        let mut payload = [0.0_f32; 45];
        for i in 0..45 {
            payload[i] = (i as f32) / 45.0;
        }
        write_dim_slice(&mut slot, &payload).unwrap();
        let read = read_dim_slice::<45>(&slot).unwrap();
        for i in 0..45 {
            assert!((read[i] - payload[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn topology_inner_lower_returns_slice_10_to_20() {
        let (_d, mut slot) = make_slot(
            "topology_30d.bin",
            TOPOLOGY_30D_SCHEMA_VERSION as u32,
            30 * 4,
        );
        let mut payload = [0.0_f32; 30];
        for i in 0..30 {
            payload[i] = i as f32; // distinguishable per dim
        }
        write_dim_slice(&mut slot, &payload).unwrap();
        let inner = read_topology_inner_lower(&slot).unwrap();
        // Inner = [10..20]
        for i in 0..10 {
            assert!((inner[i] - (10 + i) as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn topology_outer_lower_returns_slice_0_to_10() {
        let (_d, mut slot) = make_slot(
            "topology_30d.bin",
            TOPOLOGY_30D_SCHEMA_VERSION as u32,
            30 * 4,
        );
        let mut payload = [0.0_f32; 30];
        for i in 0..30 {
            payload[i] = i as f32;
        }
        write_dim_slice(&mut slot, &payload).unwrap();
        let outer = read_topology_outer_lower(&slot).unwrap();
        for i in 0..10 {
            assert!((outer[i] - i as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn topology_whole_returns_slice_20_to_30() {
        let (_d, mut slot) = make_slot(
            "topology_30d.bin",
            TOPOLOGY_30D_SCHEMA_VERSION as u32,
            30 * 4,
        );
        let mut payload = [0.0_f32; 30];
        for i in 0..30 {
            payload[i] = i as f32;
        }
        write_dim_slice(&mut slot, &payload).unwrap();
        let whole = read_topology_whole(&slot).unwrap();
        for i in 0..10 {
            assert!((whole[i] - (20 + i) as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn decode_floats_dim_mismatch_caught() {
        // Asking for 5D from a 15D payload must error.
        let payload = [0u8; 60]; // 15 × float32
        let r = decode_floats::<5>(&payload);
        assert!(matches!(
            r,
            Err(DaemonError::DimMismatch { expected: 5, .. })
        ));
    }

    #[test]
    fn encode_floats_produces_le_bytes() {
        // f32 1.0 = 0x3F800000 little-endian = 00 00 80 3F
        let bytes = encode_floats::<3>(&[1.0, 2.0, 3.0]);
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..4], &[0x00, 0x00, 0x80, 0x3F]);
        // f32 2.0 = 0x40000000 LE = 00 00 00 40
        assert_eq!(&bytes[4..8], &[0x00, 0x00, 0x00, 0x40]);
        // f32 3.0 = 0x40400000 LE = 00 00 40 40
        assert_eq!(&bytes[8..12], &[0x00, 0x00, 0x40, 0x40]);
    }

    #[test]
    fn open_slot_reads_existing_file() {
        let (dir, mut slot) = make_slot(
            "inner_body_5d.bin",
            INNER_BODY_5D_SCHEMA_VERSION as u32,
            5 * 4,
        );
        let path = slot.path().to_path_buf();
        write_dim_slice(&mut slot, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        drop(slot); // close
        let reopened = open_slot(&path).unwrap();
        let read = read_dim_slice::<5>(&reopened).unwrap();
        assert!((read[0] - 1.0).abs() < 1e-7);
        assert!((read[4] - 5.0).abs() < 1e-7);
        drop(dir);
    }

    #[test]
    fn open_slot_fails_on_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");
        let r = open_slot(&path);
        assert!(matches!(r, Err(DaemonError::ShmOpen { .. })));
    }
}
