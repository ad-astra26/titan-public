//! chi_parity_dump — Cross-language parity helper for substrate-scoped chi.
//!
//! Reads a JSON fixture from stdin (per `tests/parity/test_chi_compute_parity.py`
//! schema), invokes `ChiState::compute(...)`, emits the 6 fields as a JSON
//! object on stdout. Used by `test_chi_compute_parity.py::test_rust_python_parity_within_float32_tolerance`.
//!
//! Build: `cargo build -p titan-trinity-rs --example chi_parity_dump`.

use std::io::Read;

use serde::Deserialize;

use titan_trinity_rs::chi_state::{ChiInputs, ChiState};
use titan_trinity_rs::sphere_clocks::SphereClockSet;

#[derive(Deserialize)]
struct Fixture {
    inner_body_5d: Vec<f32>,
    inner_mind_15d: Vec<f32>,
    inner_spirit_45d: Vec<f32>,
    outer_body_5d: Vec<f32>,
    outer_mind_15d: Vec<f32>,
    outer_spirit_45d: Vec<f32>,
    ib_v: f32,
    ob_v: f32,
    im_v: f32,
    om_v: f32,
    is_v: f32,
    os_v: f32,
    neuromod_6: Vec<f32>,
    #[allow(dead_code)]
    name: Option<String>,
}

fn to_array<const N: usize>(v: &[f32], name: &str) -> [f32; N] {
    if v.len() != N {
        panic!("fixture field {name} expected length {N}, got {}", v.len());
    }
    let mut out = [0.0_f32; N];
    out.copy_from_slice(v);
    out
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin fixture JSON");
    let f: Fixture = serde_json::from_str(&input).expect("valid fixture JSON");

    let inner_body = to_array::<5>(&f.inner_body_5d, "inner_body_5d");
    let inner_mind = to_array::<15>(&f.inner_mind_15d, "inner_mind_15d");
    let inner_spirit = to_array::<45>(&f.inner_spirit_45d, "inner_spirit_45d");
    let outer_body = to_array::<5>(&f.outer_body_5d, "outer_body_5d");
    let outer_mind = to_array::<15>(&f.outer_mind_15d, "outer_mind_15d");
    let outer_spirit = to_array::<45>(&f.outer_spirit_45d, "outer_spirit_45d");
    let neuromod = to_array::<6>(&f.neuromod_6, "neuromod_6");

    let mut sc = SphereClockSet::new();
    sc.inner_body.contraction_velocity = f.ib_v;
    sc.outer_body.contraction_velocity = f.ob_v;
    sc.inner_mind.contraction_velocity = f.im_v;
    sc.outer_mind.contraction_velocity = f.om_v;
    sc.inner_spirit.contraction_velocity = f.is_v;
    sc.outer_spirit.contraction_velocity = f.os_v;

    let chi = ChiState::compute(&ChiInputs {
        inner_body_5d: &inner_body,
        inner_mind_15d: &inner_mind,
        inner_spirit_45d: &inner_spirit,
        outer_body_5d: &outer_body,
        outer_mind_15d: &outer_mind,
        outer_spirit_45d: &outer_spirit,
        sphere_clocks: &sc,
        neuromod_6: &neuromod,
    });

    let out = serde_json::json!({
        "total": chi.total,
        "spirit": chi.spirit,
        "mind": chi.mind,
        "body": chi.body,
        "coherence": chi.coherence,
        "urgency": chi.urgency,
    });
    println!("{out}");
}
