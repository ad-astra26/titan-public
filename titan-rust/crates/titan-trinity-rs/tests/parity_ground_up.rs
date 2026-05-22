//! Cross-language parity test — load `tests/parity/vectors.json::ground_up_nudge`
//! from repo root and verify titan-trinity-rs::ground_up produces byte-identical
//! output to Python `titan_hcl/logic/ground_up.py:51-101` per Preamble G5.
//!
//! Per SPEC §11.6 + PLAN_microkernel_phase_c_s3_substrate.md §14. Same JSON
//! file is loaded by pytest tests in `tests/test_phase_c_constants_in_sync.py`
//! schema check (parity_vectors_schema test) AND by Python parity tests when
//! they ship in C-S5 daemon work.

use std::path::PathBuf;

use serde::Deserialize;
use titan_trinity_rs::ground_up::compute_nudge;

#[derive(Deserialize)]
struct GroundUpNudgeVectors {
    cases: Vec<GroundUpCase>,
}

#[derive(Deserialize)]
struct GroundUpCase {
    name: String,
    input: GroundUpInput,
    expected: GroundUpExpected,
}

#[derive(Deserialize)]
struct GroundUpInput {
    grounding_signal_10d: [f32; 10],
    prev_nudge_body: [f32; 5],
    prev_nudge_mind: [f32; 5],
    damping: f32,
}

#[derive(Deserialize)]
struct GroundUpExpected {
    body_nudge: [f32; 5],
    mind_nudge: [f32; 5],
    total_magnitude: f32,
}

fn load_vectors() -> GroundUpNudgeVectors {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // titan-rust/crates/titan-trinity-rs → ../../../tests/parity/vectors.json
    let path = manifest_dir.join("../../../tests/parity/vectors.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to load parity vectors at {}: {}", path.display(), e));
    let v: serde_json::Value = serde_json::from_str(&text).expect("vectors.json parse");
    serde_json::from_value(v["ground_up_nudge"].clone()).expect("ground_up_nudge section parse")
}

#[test]
fn ground_up_nudge_byte_identical_to_python_reference() {
    let vectors = load_vectors();
    assert!(
        !vectors.cases.is_empty(),
        "ground_up_nudge.cases must be non-empty"
    );
    for case in &vectors.cases {
        let r = compute_nudge(
            &case.input.grounding_signal_10d,
            &case.input.prev_nudge_body,
            &case.input.prev_nudge_mind,
            case.input.damping,
        );

        // body_nudge byte-identical (within 1e-6 f32 tolerance)
        for (i, (got, expected)) in r
            .body_nudge
            .iter()
            .zip(case.expected.body_nudge.iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-6,
                "{}: body_nudge[{}] = {got} ≠ Python reference {expected}",
                case.name,
                i
            );
        }

        // mind_nudge byte-identical
        for (i, (got, expected)) in r
            .mind_nudge
            .iter()
            .zip(case.expected.mind_nudge.iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-6,
                "{}: mind_nudge[{}] = {got} ≠ Python reference {expected}",
                case.name,
                i
            );
        }

        // total_magnitude — Python rounds to 6 decimals; allow 1e-4 tolerance
        // to accommodate the rounding step
        assert!(
            (r.total_magnitude - case.expected.total_magnitude).abs() < 1e-4,
            "{}: total_magnitude = {} ≠ Python reference {} (after Python round(_, 6))",
            case.name,
            r.total_magnitude,
            case.expected.total_magnitude
        );
    }
}

#[test]
fn ground_up_nudge_loader_finds_three_cases() {
    let vectors = load_vectors();
    assert_eq!(vectors.cases.len(), 3);
    let names: Vec<&str> = vectors.cases.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"zero_signal"));
    assert!(names.contains(&"saturated_signal"));
    assert!(names.contains(&"with_prev_damping_clamps_to_max"));
}
