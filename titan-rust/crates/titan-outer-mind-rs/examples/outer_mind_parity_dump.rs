//! outer_mind_parity_dump — Cross-language parity helper for outer-mind 15D.
//!
//! Reads a JSON fixture from stdin (per `tests/parity/test_outer_mind_parity.py`
//! schema), msgpack-encodes the source dict, invokes `project_outer_mind_15d`,
//! emits a JSON `{"mind_15d": [...]}` object on stdout.
//!
//! Build: `cargo build -p titan-outer-mind-rs --example outer_mind_parity_dump`.

use std::io::Read;

use serde::Deserialize;
use serde_json::Value as JsonValue;

use titan_outer_mind_rs::tick_loop::project_outer_mind_15d;

#[derive(Deserialize)]
struct Fixture {
    /// Raw upstream source dict (mirrors `outer_mind_sensor_refresh.py:SOURCE_KEYS`).
    /// Encoded as JSON here; the harness converts JSON → msgpack to match
    /// what the Python sidecar writes to `sensor_cache_outer_mind.bin`.
    sources: JsonValue,
    /// Outer-body 5D vector (slot read freshly each tick by the Rust daemon).
    outer_body: Vec<f32>,
    #[allow(dead_code)]
    name: Option<String>,
}

fn json_to_rmpv(v: &JsonValue) -> rmpv::Value {
    use rmpv::Value as M;
    match v {
        JsonValue::Null => M::Nil,
        JsonValue::Bool(b) => M::Boolean(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                M::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                M::Integer(u.into())
            } else {
                M::F64(n.as_f64().unwrap_or(0.0))
            }
        }
        JsonValue::String(s) => M::String(s.clone().into()),
        JsonValue::Array(arr) => M::Array(arr.iter().map(json_to_rmpv).collect()),
        JsonValue::Object(obj) => M::Map(
            obj.iter()
                .map(|(k, val)| (M::String(k.clone().into()), json_to_rmpv(val)))
                .collect(),
        ),
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin fixture JSON");
    let f: Fixture = serde_json::from_str(&input).expect("valid fixture JSON");

    if f.outer_body.len() != 5 {
        panic!(
            "fixture.outer_body expected length 5, got {}",
            f.outer_body.len()
        );
    }
    let mut outer_body = [0.0_f32; 5];
    outer_body.copy_from_slice(&f.outer_body);

    let mp_value = json_to_rmpv(&f.sources);
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &mp_value).expect("encode source dict");

    let mind = project_outer_mind_15d(&payload, outer_body).expect("project_outer_mind_15d");

    let out = serde_json::json!({
        "mind_15d": mind.to_vec(),
    });
    println!("{out}");
}
