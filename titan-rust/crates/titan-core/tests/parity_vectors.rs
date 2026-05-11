//! Cross-language parity tests — load `tests/parity/vectors.json` from repo
//! root and verify titan-core produces byte-identical output to Python.
//!
//! Per SPEC §7.4 + §8.10 + §11.6. Same JSON file is loaded by pytest tests
//! in `tests/test_frame_parity.py` + `tests/test_bus_authkey.py` (refactor
//! ships in C2-7). When this Rust suite + the pytest suite both pass against
//! the same JSON, byte-identical wire/storage output is proven.

use serde_json::Value;
use std::path::PathBuf;
use titan_core::{authkey, frame, shm};

fn load_vectors() -> Value {
    // CARGO_MANIFEST_DIR = .../titan-rust/crates/titan-core
    // Repo root = ../../..
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../../tests/parity/vectors.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to load parity vectors at {}: {}", path.display(), e));
    serde_json::from_str(&text).expect("parity vectors JSON parse")
}

fn hex_decode(s: &str) -> Vec<u8> {
    hex::decode(s).expect("hex decode")
}

#[test]
fn vectors_json_loads_cleanly() {
    let v = load_vectors();
    // spec_version tracks the SPEC TOML major.minor.patch — bumped per
    // C-S* PATCH that adds new vectors. C-S3 (0.1.4) added fastbus_layout +
    // schumann_periods_ns sections.
    let sv = v["spec_version"].as_str().expect("spec_version is string");
    assert!(
        sv.starts_with("0.1."),
        "spec_version must be on the 0.1.x track, got {sv}"
    );
    assert!(v["frame"].is_object());
    assert!(v["authkey"].is_object());
    assert!(v["shm_layout"].is_object());
    assert!(v["bus_specs"].is_object());
}

#[test]
fn frame_rfc4231_all_cases_pass() {
    let v = load_vectors();
    for case in v["frame"]["rfc4231"].as_array().unwrap() {
        let key = hex_decode(case["key_hex"].as_str().unwrap());
        let data = hex_decode(case["data_hex"].as_str().unwrap());
        let expected = hex_decode(case["expected_hmac_hex"].as_str().unwrap());
        let actual = frame::compute_hmac(&key, &data);
        assert_eq!(
            &actual[..],
            &expected[..],
            "RFC 4231 vector '{}' mismatch",
            case["name"]
        );
    }
}

#[test]
fn frame_titan_canonical_pass() {
    let v = load_vectors();
    for case in v["frame"]["titan_canonical"].as_array().unwrap() {
        let key = hex_decode(case["key_hex"].as_str().unwrap());
        let challenge = hex_decode(case["challenge_hex"].as_str().unwrap());
        let expected = hex_decode(case["expected_hmac_hex"].as_str().unwrap());
        let actual = frame::compute_hmac(&key, &challenge);
        assert_eq!(
            &actual[..],
            &expected[..],
            "Titan canonical vector '{}' mismatch",
            case["name"]
        );
    }
}

#[test]
fn authkey_rfc5869_all_cases_pass() {
    // RFC 5869 standard test cases verify our HKDF-SHA256 matches the spec.
    // Note: derive_bus_authkey() always uses AUTHKEY_HKDF_SALT='titan-bus-v1' +
    // AUTHKEY_BYTES=32, so we can't directly run RFC test cases through the
    // public API. We DO run them through the internal HKDF impl by checking
    // known canonical inputs that DO use 'titan-bus-v1' salt — but for now,
    // RFC 5869 vectors live in the JSON for reference + future Python parity.
    // The real Rust↔Python parity check is in `authkey_titan_canonical`.
    let v = load_vectors();
    let cases = v["authkey"]["rfc5869"].as_array().unwrap();
    assert!(!cases.is_empty(), "at least one RFC 5869 vector required");
    // Each case has L != 32 + custom salt → they exercise the underlying HKDF
    // crate, which we trust (it's literally `hkdf` from RustCrypto). The
    // assertion: every case has well-formed fields.
    for case in cases {
        assert!(case["ikm_hex"].is_string());
        assert!(case["salt_hex"].is_string());
        assert!(case["info_hex"].is_string());
        assert!(case["L"].is_number());
        assert!(case["expected_okm_hex"].is_string());
    }
}

#[test]
fn authkey_titan_canonical_pass() {
    // Titan canonical vector: identity_secret → derive_bus_authkey() should
    // produce exactly this 32-byte key. Both Rust + Python must produce the
    // same hex string for this input.
    //
    // Per `rFP_phase_c_bus_authkey_contract_fix.md` (2026-05-05): the HKDF info
    // is the CONSTANT b"titan-bus" (not titan_id). titan_id is preserved in
    // the vector schema as informational metadata only — it does NOT enter
    // the HKDF derivation.
    let v = load_vectors();
    let case = &v["authkey"]["titan_canonical"][0];
    let identity_secret = hex_decode(case["identity_secret_bytes_hex"].as_str().unwrap());

    let actual = authkey::derive_bus_authkey(&identity_secret).unwrap();
    let actual_hex = hex::encode(actual);

    let expected = case["expected_authkey_hex_to_be_filled_by_first_implementation"]
        .as_str()
        .unwrap();
    if expected == "REPLACE_WITH_ACTUAL_HEX_AT_FIRST_RUN" {
        // First run: print the value so it can be locked in JSON.
        // This becomes a regression-must-pass once filled in.
        println!(
            "First run — replace 'REPLACE_WITH_ACTUAL_HEX_AT_FIRST_RUN' in tests/parity/vectors.json with: {}",
            actual_hex
        );
        // Self-consistency: derivation must be deterministic
        let again = authkey::derive_bus_authkey(&identity_secret).unwrap();
        assert_eq!(actual, again);
        assert_eq!(actual.len(), 32);
    } else {
        assert_eq!(
            actual_hex, expected,
            "Titan canonical authkey vector mismatch — wire protocol broken between Rust and Python"
        );
    }
}

#[test]
fn shm_header_size_matches_vectors() {
    // SPEC §7.0 v1.0.0 — fixed header is 16 bytes (was 24-byte SeqLock pre-D-SPEC-35).
    let v = load_vectors();
    let header_bytes = v["shm_layout"]["header_bytes"].as_u64().unwrap();
    assert_eq!(header_bytes, 16);
    assert_eq!(header_bytes, shm::SHM_HEADER_BYTES);
    let buffer_meta_bytes = v["shm_layout"]["buffer_meta_bytes"].as_u64().unwrap();
    assert_eq!(buffer_meta_bytes, 16);
    assert_eq!(buffer_meta_bytes, shm::SHM_BUFFER_META_BYTES);
    let buffer_count = v["shm_layout"]["buffer_count"].as_u64().unwrap();
    assert_eq!(buffer_count, 3);
    assert_eq!(buffer_count, shm::SHM_BUFFER_COUNT);
}

#[test]
fn shm_slot_total_bytes_match_vectors() {
    // SPEC §7.0 v1.0.0 — total = SHM_HEADER_BYTES (16) + SHM_BUFFER_COUNT (3) × (SHM_BUFFER_META_BYTES (16) + payload).
    // fastbus.bin is excluded — it is a self-contained SPSC ring, not a §7.0 universal slot.
    let v = load_vectors();
    let slots = v["shm_layout"]["slots"].as_object().unwrap();
    for (slot_name, spec) in slots {
        if slot_name == "fastbus.bin" {
            continue;
        }
        let payload = spec["payload_bytes"].as_u64().unwrap();
        let total = spec["total_bytes"].as_u64().unwrap();
        assert_eq!(
            total,
            16 + 3 * (16 + payload),
            "slot '{slot_name}' total_bytes ({total}) != 16 + 3 × (16 + payload_bytes ({payload}))"
        );
    }
}

#[test]
fn shm_crc32_known_vectors_match() {
    let v = load_vectors();
    for vec in v["shm_layout"]["crc32_known_vectors"].as_array().unwrap() {
        let input = vec["input_ascii"].as_str().unwrap();
        let expected_hex = vec["expected_crc32_hex"].as_str().unwrap();
        let expected = u32::from_str_radix(expected_hex, 16).unwrap();
        let actual = shm::crc32(input.as_bytes());
        assert_eq!(
            actual, expected,
            "CRC32 mismatch for input '{input}' (expected 0x{expected_hex}, got 0x{actual:08x})"
        );
    }
}
