//! C5-9 e2e harness — exercises the full Phase C C-S5 inner-trinity wire
//! protocol round-trip without spawning subprocesses.
//!
//! Per master plan §10.5 chunk C5-9 + my C-S5 PLAN §4.9 + the parallel-dev
//! adaptation noted in §0.4 v2 (Maker decision 2026-04-29):
//!
//! Subprocess-based kernel-spawn integration (which spawns the 3 daemon
//! binaries + a kernel + a stub substrate publisher) belongs to C-S7's
//! flag-flip prep where the full Rust tree boots end-to-end. This test
//! exercises the LAYER between daemons + bus that's the load-bearing
//! correctness frontier:
//!
//! 1. Stub publisher encodes UNIFIED_SPIRIT_FILTER_DOWN (per SPEC §8.6)
//!    → daemon decodes → daemon's filter_down apply produces expected
//!    output → daemon encodes BODY_STATE / MIND_STATE / SPIRIT_STATE
//!    → observer decodes + verifies wire format.
//! 2. inner-spirit produces INNER_SPIRIT_FILTER_DOWN → inner-body /
//!    inner-mind decode it (LOCAL cascade) → multipliers compose
//!    UNIFIED ⊗ LOCAL correctly.
//! 3. Topology slice handling: substrate's topology_30d[10:20] reads
//!    correctly as inner_lower for body + mind ground_up.
//!
//! All cross-language byte-identical guarantees (SPEC §8.10 + §11.6) are
//! verified by the parity tests in adoption.rs (canonical 60-byte ADOPTION
//! payload + 61-byte ACK).

use titan_state::Slot;
use titan_trinity_daemon::{
    apply_multipliers, compose_multipliers_default, decode_filter_down_payload,
    decode_local_filter_down_payload, encode_filter_down_payload, encode_floats, ContentGate,
    GroundUpEnricher, Side,
};

// ───────────────────────────────────────────────────────────────────────
// 1. Full bus round-trip — UNIFIED_SPIRIT_FILTER_DOWN encode → decode
// ───────────────────────────────────────────────────────────────────────

#[test]
fn unified_filter_down_full_round_trip() {
    // Stub publisher (substitute for titan-unified-spirit-rs in C-S7
    // production tree) encodes a full 6-field UNIFIED_SPIRIT_FILTER_DOWN
    // payload per SPEC §8.6. All 3 inner daemons subscribe; each pulls
    // out the slice it cares about.
    let inner_body = [0.5, 1.5, 1.0, 2.0, 0.8];
    let inner_mind: [f32; 15] = std::array::from_fn(|i| 1.0 + (i as f32) * 0.05);
    let inner_spirit_content: [f32; 40] = std::array::from_fn(|i| 1.0 + (i as f32) * 0.02);
    let outer_body = [1.0; 5];
    let outer_mind = [1.0; 15];
    let outer_spirit_content = [1.0; 40];

    let bytes = encode_filter_down_payload(
        &inner_body,
        &inner_mind,
        &inner_spirit_content,
        &outer_body,
        &outer_mind,
        &outer_spirit_content,
        42,
        1730000000.5,
    );

    let decoded = decode_filter_down_payload(&bytes).expect("decode");

    // Inner daemons see only the inner slice (outer fields decoded but
    // discarded at the boundary).
    for i in 0..5 {
        assert!((decoded.inner_body[i] - inner_body[i]).abs() < 1e-5);
    }
    for i in 0..15 {
        assert!((decoded.inner_mind[i] - inner_mind[i]).abs() < 1e-5);
    }
    for i in 0..40 {
        assert!((decoded.inner_spirit_content[i] - inner_spirit_content[i]).abs() < 1e-5);
    }
    assert_eq!(decoded.epoch_id, 42);
    assert!((decoded.ts - 1730000000.5).abs() < 1e-3);
}

// ───────────────────────────────────────────────────────────────────────
// 2. UNIFIED ⊗ LOCAL compose — what inner_body sees when both arrive
// ───────────────────────────────────────────────────────────────────────

#[test]
fn unified_local_compose_matches_filter_apply_pipeline() {
    // Simulated inner-spirit publishes INNER_SPIRIT_FILTER_DOWN with
    // body[5] = [1.2; 5]. UNIFIED gives body[5] = [1.5; 5].
    // Inner-body subscribes to both + composes.
    let unified_body = [1.5_f32; 5];
    let local_body = [1.2_f32; 5];

    let composed = compose_multipliers_default(&unified_body, &local_body);

    // Each dim: 1.5 * 1.2 = 1.8 (within [FLOOR=0.3, CEIL=3.0])
    for v in composed.iter() {
        assert!((v - 1.8).abs() < 1e-5);
    }

    // Apply to body=0.5: result = 0.5 * 1.8 = 0.9 (within [0, 1])
    let mut body = [0.5_f32; 5];
    apply_multipliers(&mut body, &composed);
    for v in body.iter() {
        assert!((v - 0.9).abs() < 1e-5);
    }
}

#[test]
fn extreme_compose_clips_at_ceil_then_tensor_max() {
    // unified=2.0 * local=2.0 = 4.0 → clamp MULTIPLIER_CEIL=3.0
    // Apply to body=0.5: 0.5 * 3.0 = 1.5 → clamp TENSOR_MAX=1.0
    let unified = [2.0_f32; 5];
    let local = [2.0_f32; 5];
    let composed = compose_multipliers_default(&unified, &local);
    let mut body = [0.5_f32; 5];
    apply_multipliers(&mut body, &composed);
    for v in body.iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

// ───────────────────────────────────────────────────────────────────────
// 3. INNER_SPIRIT_FILTER_DOWN cascade — body + mind both decode
// ───────────────────────────────────────────────────────────────────────

#[test]
fn inner_spirit_filter_down_cascade_to_body_and_mind() {
    // inner-spirit-rs publishes a LOCAL filter_down message; both
    // inner-body and inner-mind decode it (different fields per
    // subscriber).
    use rmpv::Value;
    fn arr<const N: usize>(v: &[f32; N]) -> Value {
        Value::Array(v.iter().map(|f| Value::F64(*f as f64)).collect())
    }

    let body_mults = [1.05_f32; 5];
    let mind_mults: [f32; 15] = std::array::from_fn(|i| 1.0 + (i as f32) * 0.001);
    let spirit_content_mults: [f32; 40] = [1.0; 40];

    let mults_map = Value::Map(vec![
        (Value::String("inner_body".into()), arr(&body_mults)),
        (Value::String("inner_mind".into()), arr(&mind_mults)),
        (
            Value::String("inner_spirit_content".into()),
            arr(&spirit_content_mults),
        ),
    ]);
    let payload_map = Value::Map(vec![
        (Value::String("multipliers".into()), mults_map),
        (Value::String("ts".into()), Value::F64(99.0)),
    ]);
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &payload_map).unwrap();

    // inner-body decodes
    let decoded = decode_local_filter_down_payload(&bytes).expect("decode local");
    for i in 0..5 {
        assert!((decoded.body[i] - body_mults[i]).abs() < 1e-5);
    }
    for i in 0..15 {
        assert!((decoded.mind[i] - mind_mults[i]).abs() < 1e-5);
    }
    assert!((decoded.ts - 99.0).abs() < 1e-3);
}

// ───────────────────────────────────────────────────────────────────────
// 4. Topology slice + ground_up integration (G4 + G10 enforcement)
// ───────────────────────────────────────────────────────────────────────

#[test]
fn topology_30d_inner_lower_slice_drives_body_and_mind_ground_up() {
    // Substrate (titan-trinity-rs from C-S3) writes topology_30d.bin
    // with [10:20] = inner_lower (G4 byte layout). Both inner_body +
    // inner_mind read this slice via read_topology_inner_lower.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("topology_30d.bin");
    let mut topology_slot = Slot::create(
        &path,
        titan_core::constants::TOPOLOGY_30D_SCHEMA_VERSION as u32,
        30 * 4,
    )
    .unwrap();

    // Construct full 30D topology with distinct values per dim:
    // [0:10] outer_lower = 0.x, [10:20] inner_lower = 1.x, [20:30] whole = 2.x
    let mut full = [0.0_f32; 30];
    for i in 0..10 {
        full[i] = 0.01 * (i as f32);
        full[10 + i] = 1.0 + 0.01 * (i as f32);
        full[20 + i] = 2.0 + 0.01 * (i as f32);
    }
    let bytes = encode_floats::<30>(&full);
    topology_slot.write(&bytes).unwrap();

    // inner_lower slice fetch
    let inner_lower =
        titan_trinity_daemon::read_topology_inner_lower(&topology_slot).expect("read");
    for i in 0..10 {
        assert!((inner_lower[i] - (1.0 + 0.01 * (i as f32))).abs() < 1e-5);
    }

    // ground_up Body side (signal[0:5]) drives all 5D body
    let mut g_body = GroundUpEnricher::new(Side::Body);
    let mut body = [0.5_f32; 5];
    g_body.apply_to_body(&mut body, &inner_lower, 1.0).unwrap();
    // Body changed from 0.5 (signal[0:5] is non-zero in this test)
    for i in 0..5 {
        // signal[i] = 1.0 + 0.01*i; clamped at MAX_NUDGE=0.05 after damping;
        // first call: 0.95*0 + 0.05*signal = 0.05*(1.0+0.01*i) > 0.05 →
        // clamped to MAX_NUDGE; delta = 0.05 * 0.1 * 1.0 = 0.005
        assert!((body[i] - 0.505).abs() < 1e-4, "body[{i}]={}", body[i]);
    }

    // ground_up MindWilling side (signal[5:10]) drives mind[10:15] ONLY
    let mut g_mind = GroundUpEnricher::new(Side::MindWilling);
    let mut mind = [0.5_f32; 15];
    g_mind.apply_to_mind(&mut mind, &inner_lower, 1.0).unwrap();
    // Thinking[0:5] + Feeling[5:10] UNTOUCHED (G10)
    for i in 0..10 {
        assert!(
            (mind[i] - 0.5).abs() < 1e-6,
            "mind[{i}] should be untouched"
        );
    }
    // Willing[10:15] nudged by signal[5:10]
    for i in 10..15 {
        // Same formula as body: 0.05 * 0.1 = 0.005
        assert!((mind[i] - 0.505).abs() < 1e-4);
    }
}

// ───────────────────────────────────────────────────────────────────────
// 5. Slot lifecycle — kernel-pre-created, daemons open + write
// ───────────────────────────────────────────────────────────────────────

#[test]
fn full_inner_trinity_slot_lifecycle() {
    // Mirrors what titan-kernel-rs does at boot: pre-create all 3 inner
    // slots; then daemons open (read-only would be ideal but Slot::open
    // returns RW handle by design — daemons trust their writer attribution).
    let dir = tempfile::tempdir().unwrap();
    let body_path = dir.path().join("inner_body_5d.bin");
    let mind_path = dir.path().join("inner_mind_15d.bin");
    let spirit_path = dir.path().join("inner_spirit_45d.bin");

    let mut body_slot = Slot::create(
        &body_path,
        titan_core::constants::INNER_BODY_5D_SCHEMA_VERSION as u32,
        5 * 4,
    )
    .unwrap();
    let mut mind_slot = Slot::create(
        &mind_path,
        titan_core::constants::INNER_MIND_15D_SCHEMA_VERSION as u32,
        15 * 4,
    )
    .unwrap();
    let mut spirit_slot = Slot::create(
        &spirit_path,
        titan_core::constants::INNER_SPIRIT_45D_SCHEMA_VERSION as u32,
        45 * 4,
    )
    .unwrap();

    // Verify SPEC §7.1 byte counts (v1.0.0: 16 fixed header + 3 × (16 buffer meta + payload))
    assert_eq!(std::fs::metadata(&body_path).unwrap().len(), 16 + 3 * (16 + 20));
    assert_eq!(std::fs::metadata(&mind_path).unwrap().len(), 16 + 3 * (16 + 60));
    assert_eq!(std::fs::metadata(&spirit_path).unwrap().len(), 16 + 3 * (16 + 180));

    // Daemons write tick output — body first
    let body: [f32; 5] = std::array::from_fn(|i| (i as f32) * 0.1);
    body_slot.write(&encode_floats::<5>(&body)).unwrap();
    let read_body = body_slot.read().unwrap();
    assert_eq!(read_body, encode_floats::<5>(&body));

    // Mind
    let mind: [f32; 15] = std::array::from_fn(|i| (i as f32) * 0.05);
    mind_slot.write(&encode_floats::<15>(&mind)).unwrap();
    let read_mind = mind_slot.read().unwrap();
    assert_eq!(read_mind, encode_floats::<15>(&mind));

    // Spirit
    let spirit: [f32; 45] = std::array::from_fn(|i| (i as f32) * 0.02);
    spirit_slot.write(&encode_floats::<45>(&spirit)).unwrap();
    let read_spirit = spirit_slot.read().unwrap();
    assert_eq!(read_spirit, encode_floats::<45>(&spirit));

    // Inner-spirit Observer Principle: opens body + mind read-side via
    // a fresh handle (matches what tick_loop::run does at boot).
    let body_observer = Slot::open(&body_path).unwrap();
    let mind_observer = Slot::open(&mind_path).unwrap();
    let observed_body = titan_trinity_daemon::read_dim_slice::<5>(&body_observer).unwrap();
    let observed_mind = titan_trinity_daemon::read_dim_slice::<15>(&mind_observer).unwrap();
    for i in 0..5 {
        assert!((observed_body[i] - body[i]).abs() < 1e-6);
    }
    for i in 0..15 {
        assert!((observed_mind[i] - mind[i]).abs() < 1e-6);
    }
}

// ───────────────────────────────────────────────────────────────────────
// 6. Content-hash gating end-to-end — steady-state suppress + change-write
// ───────────────────────────────────────────────────────────────────────

#[test]
fn content_hash_gates_end_to_end() {
    let mut gate = ContentGate::new();
    // First tick: write
    let body = [0.5_f32; 5];
    let bytes1 = encode_floats::<5>(&body);
    assert!(gate.should_write(&bytes1));

    // Steady state: suppress 1000 identical writes
    for _ in 0..1000 {
        assert!(!gate.should_write(&bytes1));
    }
    assert_eq!(gate.write_count(), 1);
    assert_eq!(gate.suppress_count(), 1000);
    assert!(gate.suppress_ratio() > 0.99);

    // Tiny perturbation: write
    let mut body2 = body;
    body2[0] = 0.5001;
    let bytes2 = encode_floats::<5>(&body2);
    assert!(gate.should_write(&bytes2));
    assert_eq!(gate.write_count(), 2);
}
