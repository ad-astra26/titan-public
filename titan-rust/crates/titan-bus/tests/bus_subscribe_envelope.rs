//! BUS_SUBSCRIBE byte-wire parity test against Python ground truth.
//!
//! Closes the parity-test gap surfaced by `rFP_worker_broadcast_topics_completion §4.C-ter`
//! (codified 2026-05-13). Pre-existing helper-only unit tests at
//! `client.rs:499 + 538` exercised only `build_subscribe_payload`; the full
//! `encode_simple` envelope path was uncovered, which is why the
//! Binary-vs-Map encoder bug landed without CI catching it.
//!
//! Vectors loaded from `tests/parity/vectors.json :: bus_subscribe_envelope ::
//! canonical_v1` — Python `msgpack.packb(use_bin_type=True)` ground truth.
//! Rust `encode_simple` + `build_subscribe_payload` MUST produce
//! byte-identical output per SPEC §8.2 line 789 + §8.10 line 900.
//!
//! Sibling Python pytest at `tests/parity/test_bus_subscribe_envelope.py`
//! validates the Python side against the same vectors.

use std::path::PathBuf;

use rmpv::Value;
use serde_json::Value as JsonValue;
use titan_bus::message::encode_simple;

/// Load and parse `tests/parity/vectors.json` from the repo root.
fn load_vectors() -> JsonValue {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // titan-rust/crates/titan-bus -> ../../.. -> repo root
    let vectors_path = manifest
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("expected repo root three levels up from CARGO_MANIFEST_DIR")
        .join("tests/parity/vectors.json");
    let raw = std::fs::read_to_string(&vectors_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", vectors_path.display()));
    serde_json::from_str(&raw).expect("vectors.json valid JSON")
}

/// Build BUS_SUBSCRIBE payload Value per SPEC §8.2 line 789 schema.
/// Mirrors `titan_bus::client::build_subscribe_payload` (private helper
/// re-implemented here so tests don't need a `pub` exposure).
fn build_subscribe_payload(name: &str, topics: &[&str], reply_only: bool) -> Value {
    let topic_values: Vec<Value> = topics.iter().map(|t| Value::String((*t).into())).collect();
    Value::Map(vec![
        (Value::String("name".into()), Value::String(name.into())),
        (Value::String("topics".into()), Value::Array(topic_values)),
        (
            Value::String("reply_only".into()),
            Value::Boolean(reply_only),
        ),
    ])
}

/// Encode the full BUS_SUBSCRIBE envelope as the production path does.
fn encode_full_envelope(name: &str, topics: &[&str], reply_only: bool) -> Vec<u8> {
    let payload = build_subscribe_payload(name, topics, reply_only);
    encode_simple("BUS_SUBSCRIBE", Some(name), Some("broker"), Some(payload))
        .expect("encode_simple for known-good BUS_SUBSCRIBE input never fails")
}

#[test]
fn broadcast_consumer_inner_body_matches_python_vector() {
    let vectors = load_vectors();
    let vec_root =
        &vectors["bus_subscribe_envelope"]["canonical_v1"]["broadcast_consumer_inner_body"];
    let expected_hex = vec_root["msgpack_hex"]
        .as_str()
        .expect("msgpack_hex string");
    let expected_bytes: Vec<u8> = (0..expected_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).expect("valid hex byte"))
        .collect();

    // Topics + name pulled from the vector's input so the Rust test stays
    // locked to the same SPEC §9.A REQUIRED list documented in the vector.
    let name = vec_root["input"]["payload"]["name"]
        .as_str()
        .expect("input name");
    let topics: Vec<String> = vec_root["input"]["payload"]["topics"]
        .as_array()
        .expect("input topics array")
        .iter()
        .map(|t| t.as_str().expect("topic str").to_string())
        .collect();
    let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
    let reply_only = vec_root["input"]["payload"]["reply_only"]
        .as_bool()
        .expect("reply_only bool");

    let actual = encode_full_envelope(name, &topic_refs, reply_only);

    let expected_len = vec_root["msgpack_bytes"].as_u64().unwrap() as usize;
    assert_eq!(
        actual.len(),
        expected_len,
        "Rust BUS_SUBSCRIBE envelope length differs from Python ground truth"
    );
    assert_eq!(
        actual,
        expected_bytes,
        "Rust BUS_SUBSCRIBE envelope bytes differ from Python ground truth — \
         SPEC §8.10 byte-identical guarantee violated. Got {} bytes, hex={}",
        actual.len(),
        actual
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
}

#[test]
fn reply_only_titan_hcl_matches_python_vector() {
    let vectors = load_vectors();
    let vec_root = &vectors["bus_subscribe_envelope"]["canonical_v1"]["reply_only_titan_HCL"];
    let expected_hex = vec_root["msgpack_hex"]
        .as_str()
        .expect("msgpack_hex string");
    let expected_bytes: Vec<u8> = (0..expected_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).expect("valid hex byte"))
        .collect();

    let name = vec_root["input"]["payload"]["name"]
        .as_str()
        .expect("input name");
    // topics: []
    let topic_refs: Vec<&str> = Vec::new();
    let reply_only = vec_root["input"]["payload"]["reply_only"]
        .as_bool()
        .expect("reply_only bool");

    let actual = encode_full_envelope(name, &topic_refs, reply_only);

    assert_eq!(
        actual,
        expected_bytes,
        "Rust BUS_SUBSCRIBE (reply_only=true) bytes differ from Python ground truth — \
         SPEC §8.10 byte-identical guarantee violated for D-SPEC-42 row 2 intent. \
         Got {} bytes, hex={}",
        actual.len(),
        actual
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
}

// ─────────────────────────────────────────────────────────────────────────
// §4.C-quater (2026-05-13): decoder-side parity test (closes the gap that
// let the reply_only=false production drift land — encoder parity ONLY
// verified bytes-out, never round-tripped through the decoder).
// Per `feedback_function_parity_vs_contract_parity.md`: contract parity
// requires BOTH encoder bytes match Python ground truth AND decoder
// extracts the original fields back from those bytes.
// ─────────────────────────────────────────────────────────────────────────

use titan_bus::message::decode_bus_subscribe_payload;

#[test]
fn decoder_round_trip_broadcast_consumer_inner_body() {
    let vectors = load_vectors();
    let vec_root =
        &vectors["bus_subscribe_envelope"]["canonical_v1"]["broadcast_consumer_inner_body"];
    let expected_hex = vec_root["msgpack_hex"]
        .as_str()
        .expect("msgpack_hex string");
    let envelope_bytes: Vec<u8> = (0..expected_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).expect("valid hex byte"))
        .collect();

    let expected_name = vec_root["input"]["payload"]["name"]
        .as_str()
        .expect("input name")
        .to_string();
    let expected_topics: Vec<String> = vec_root["input"]["payload"]["topics"]
        .as_array()
        .expect("input topics array")
        .iter()
        .map(|t| t.as_str().expect("topic str").to_string())
        .collect();
    let expected_reply_only = vec_root["input"]["payload"]["reply_only"]
        .as_bool()
        .expect("reply_only bool");

    let (got_name, got_topics, got_reply_only) = decode_bus_subscribe_payload(&envelope_bytes)
        .expect("decoder must succeed on canonical envelope");

    assert_eq!(got_name, Some(expected_name.clone()),
        "decoder DROPPED name field from canonical envelope — SPEC §8.2 v1.4.0 forbidden-regression state");
    assert_eq!(
        got_topics, expected_topics,
        "decoder DROPPED topics field — broadcasts to this subscriber would WARN+drop"
    );
    assert_eq!(
        got_reply_only, expected_reply_only,
        "decoder DROPPED reply_only field — broadcast queue would fill on reply_only subscribers"
    );
}

#[test]
fn decoder_round_trip_reply_only_titan_hcl() {
    let vectors = load_vectors();
    let vec_root = &vectors["bus_subscribe_envelope"]["canonical_v1"]["reply_only_titan_HCL"];
    let expected_hex = vec_root["msgpack_hex"]
        .as_str()
        .expect("msgpack_hex string");
    let envelope_bytes: Vec<u8> = (0..expected_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).expect("valid hex byte"))
        .collect();

    let expected_name = vec_root["input"]["payload"]["name"]
        .as_str()
        .expect("input name")
        .to_string();
    let expected_reply_only = vec_root["input"]["payload"]["reply_only"]
        .as_bool()
        .expect("reply_only bool");

    let (got_name, got_topics, got_reply_only) = decode_bus_subscribe_payload(&envelope_bytes)
        .expect("decoder must succeed on canonical envelope");

    assert_eq!(
        got_name,
        Some(expected_name.clone()),
        "decoder DROPPED name on reply_only=true subscriber — SPEC §8.2 v1.4.0 Row 2 broken"
    );
    assert_eq!(
        got_topics,
        Vec::<String>::new(),
        "decoder extracted topics where there should be none (empty)"
    );
    assert_eq!(got_reply_only, expected_reply_only,
        "decoder DROPPED reply_only=true — this is the production T3 bug — broker enqueues broadcasts to reply-only subscribers");
}
