//! adoption — B.2.1 supervision-transfer ADOPTION_REQUEST/ACK msgpack codec.
//!
//! Per SPEC §8.4 + §8.10 + D14 + D18 — locked against Python ground truth
//! at `titan_plugin/core/worker_swap_handler.py:206-216`:
//!
//! ```python
//! req = bus.make_msg(
//!     bus.BUS_WORKER_ADOPT_REQUEST,
//!     state.name,
//!     "guardian",
//!     {
//!         "name": state.name,
//!         "pid": os.getpid(),
//!         "start_method": state.start_method,
//!         "boot_ts": time.time(),
//!     },
//!     rid=rid,
//! )
//! ```
//!
//! **The `rid` field is in the bus ENVELOPE (top-level msg dict), NOT the
//! payload.** The payload itself has 4 fields:
//! `{name: str, pid: int, start_method: "spawn"|"fork", boot_ts: float}`.
//!
//! Earlier SPEC §8.4 row claimed payload includes rid (5 fields); that was
//! a documentation error inherited from the legacy
//! `BUS_WORKER_ADOPT_REQUEST` rename. SPEC v0.1.5 corrects the row + locks
//! the canonical 60-byte msgpack vector here.
//!
//! # Canonical byte-locked vector
//!
//! Input (canonical):
//! ```ignore
//! {"name": "inner-body", "pid": 12345, "start_method": "spawn", "boot_ts": 1730000000.5}
//! ```
//! Output (60 bytes, hex):
//! ```ignore
//! 84a46e616d65aa696e6e65722d626f6479a3706964cd3039ac73746172745f
//! 6d6574686f64a5737061776ea7626f6f745f7473cb41d9c76d20200000
//! ```
//! Stored in `tests/parity/vectors.json::adoption_payload.canonical_v1`
//! for cross-language parity (Python pytest + cargo test both load + assert
//! byte-identical encode + round-trip decode).

use crate::error::{DaemonError, DaemonResult};

/// Total bytes for the canonical ADOPTION_REQUEST payload (`name=inner-body,
/// pid=12345, start_method=spawn, boot_ts=1730000000.5`). Used by tests.
pub const CANONICAL_ADOPTION_REQUEST_PAYLOAD_BYTES: usize = 60;

/// Total bytes for the canonical ADOPTION_ACK payload (`rid=00000000-...,
/// accepted=true, reason=null`). Used by tests. Matches Python wire
/// format (default `use_bin_type=True` → str8 marker for the 36-char UUID).
pub const CANONICAL_ADOPTION_ACK_PAYLOAD_BYTES: usize = 61;

/// Spawn method for B.2.1 supervision transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartMethod {
    /// SPAWN-mode (Phase C default per D13) — child is brand-new, kernel
    /// passes env + connects to bus from scratch.
    Spawn,
    /// FORK-mode (legacy B.1; retired in C-S8) — child copy-on-writes
    /// from kernel address space.
    Fork,
}

impl StartMethod {
    /// String representation used in the msgpack `start_method` field.
    pub fn as_str(self) -> &'static str {
        match self {
            StartMethod::Spawn => "spawn",
            StartMethod::Fork => "fork",
        }
    }

    /// Parse from a msgpack string. Returns Err on unknown variant.
    pub fn parse(s: &str) -> DaemonResult<Self> {
        match s {
            "spawn" => Ok(StartMethod::Spawn),
            "fork" => Ok(StartMethod::Fork),
            other => Err(DaemonError::MsgpackDecode(format!(
                "unknown start_method: {other:?}"
            ))),
        }
    }
}

/// Decoded ADOPTION_REQUEST payload (4 fields per Python ground truth —
/// rid lives in the bus envelope, not the payload).
#[derive(Debug, Clone, PartialEq)]
pub struct AdoptionRequest {
    /// Daemon name (e.g. `"inner-body"`, `"inner-mind"`, etc.).
    pub name: String,
    /// Daemon's current PID (for sanity-check by adopting parent).
    pub pid: i64,
    /// Spawn / fork method.
    pub start_method: StartMethod,
    /// Daemon's boot wall-clock timestamp.
    pub boot_ts: f64,
}

/// Decoded ADOPTION_ACK payload — sent by the adopting parent to confirm
/// or reject the adoption.
#[derive(Debug, Clone, PartialEq)]
pub struct AdoptionAck {
    /// Request UUID being acknowledged (echoed from the original request's
    /// envelope rid).
    pub rid: String,
    /// True if the new parent accepts the daemon.
    pub accepted: bool,
    /// Optional rejection reason. `None` when `accepted=true`.
    pub reason: Option<String>,
}

/// Encode an ADOPTION_REQUEST payload to msgpack bytes.
///
/// The PAYLOAD-only encoding (4 fields, ~60 bytes for canonical inputs) —
/// callers wrap this in a bus envelope via [`titan_bus::BusClient::publish`]
/// which adds type/src/dst/rid/ts at the envelope level.
///
/// Field order: `name → pid → start_method → boot_ts` (matches Python
/// dict-literal order — msgpack preserves insertion order in this Rust
/// encoder + Python's `msgpack.packb` does the same with a regular dict).
pub fn encode_adoption_request_payload(req: &AdoptionRequest) -> DaemonResult<Vec<u8>> {
    use rmpv::Value;
    let map = Value::Map(vec![
        (
            Value::String("name".into()),
            Value::String(req.name.clone().into()),
        ),
        (Value::String("pid".into()), Value::Integer(req.pid.into())),
        (
            Value::String("start_method".into()),
            Value::String(req.start_method.as_str().to_string().into()),
        ),
        (Value::String("boot_ts".into()), Value::F64(req.boot_ts)),
    ]);
    let mut out = Vec::with_capacity(96);
    rmpv::encode::write_value(&mut out, &map)
        .map_err(|e| DaemonError::MsgpackEncode(format!("adoption_request: {e}")))?;
    Ok(out)
}

/// Decode an ADOPTION_REQUEST msgpack payload to typed struct.
pub fn decode_adoption_request_payload(bytes: &[u8]) -> DaemonResult<AdoptionRequest> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes))
        .map_err(|e| DaemonError::MsgpackDecode(format!("adoption_request: {e}")))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => {
            return Err(DaemonError::MsgpackDecode(
                "adoption_request not a map".into(),
            ))
        }
    };
    let mut name: Option<String> = None;
    let mut pid: Option<i64> = None;
    let mut start_method: Option<StartMethod> = None;
    let mut boot_ts: Option<f64> = None;
    for (k, val) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "name" => {
                if let Value::String(s) = val {
                    name = Some(s.as_str().unwrap_or("").to_string());
                }
            }
            "pid" => pid = val.as_i64(),
            "start_method" => {
                if let Value::String(s) = val {
                    start_method = Some(StartMethod::parse(s.as_str().unwrap_or(""))?);
                }
            }
            "boot_ts" => boot_ts = val.as_f64(),
            _ => {}
        }
    }
    Ok(AdoptionRequest {
        name: name.ok_or_else(|| DaemonError::MsgpackDecode("missing name".into()))?,
        pid: pid.ok_or_else(|| DaemonError::MsgpackDecode("missing pid".into()))?,
        start_method: start_method
            .ok_or_else(|| DaemonError::MsgpackDecode("missing start_method".into()))?,
        boot_ts: boot_ts.ok_or_else(|| DaemonError::MsgpackDecode("missing boot_ts".into()))?,
    })
}

/// Encode an ADOPTION_ACK payload. 3 fields: `rid → accepted → reason`.
pub fn encode_adoption_ack_payload(ack: &AdoptionAck) -> DaemonResult<Vec<u8>> {
    use rmpv::Value;
    let reason_val = match &ack.reason {
        Some(s) => Value::String(s.clone().into()),
        None => Value::Nil,
    };
    let map = Value::Map(vec![
        (
            Value::String("rid".into()),
            Value::String(ack.rid.clone().into()),
        ),
        (
            Value::String("accepted".into()),
            Value::Boolean(ack.accepted),
        ),
        (Value::String("reason".into()), reason_val),
    ]);
    let mut out = Vec::with_capacity(96);
    rmpv::encode::write_value(&mut out, &map)
        .map_err(|e| DaemonError::MsgpackEncode(format!("adoption_ack: {e}")))?;
    Ok(out)
}

/// Decode an ADOPTION_ACK msgpack payload.
pub fn decode_adoption_ack_payload(bytes: &[u8]) -> DaemonResult<AdoptionAck> {
    use rmpv::Value;
    let v: Value = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes))
        .map_err(|e| DaemonError::MsgpackDecode(format!("adoption_ack: {e}")))?;
    let map = match &v {
        Value::Map(items) => items,
        _ => return Err(DaemonError::MsgpackDecode("adoption_ack not a map".into())),
    };
    let mut rid: Option<String> = None;
    let mut accepted: Option<bool> = None;
    let mut reason: Option<String> = None;
    for (k, val) in map.iter() {
        let key = match k {
            Value::String(s) => s.as_str().unwrap_or(""),
            _ => continue,
        };
        match key {
            "rid" => {
                if let Value::String(s) = val {
                    rid = Some(s.as_str().unwrap_or("").to_string());
                }
            }
            "accepted" => accepted = val.as_bool(),
            "reason" => match val {
                Value::Nil => reason = None,
                Value::String(s) => reason = Some(s.as_str().unwrap_or("").to_string()),
                _ => {}
            },
            _ => {}
        }
    }
    Ok(AdoptionAck {
        rid: rid.ok_or_else(|| DaemonError::MsgpackDecode("missing rid".into()))?,
        accepted: accepted.ok_or_else(|| DaemonError::MsgpackDecode("missing accepted".into()))?,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical input fixture matching the byte-locked vector in
    /// `tests/parity/vectors.json::adoption_payload.canonical_v1`.
    fn canonical_request() -> AdoptionRequest {
        AdoptionRequest {
            name: "inner-body".to_string(),
            pid: 12345,
            start_method: StartMethod::Spawn,
            boot_ts: 1730000000.5,
        }
    }

    /// Hex of the locked canonical payload (60 bytes).
    /// Generated via Python `msgpack.packb` against the canonical input.
    const CANONICAL_REQUEST_HEX: &str = "84a46e616d65aa696e6e65722d626f6479a3706964cd3039ac73746172745f6d6574686f64a5737061776ea7626f6f745f7473cb41d9c76d20200000";

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn start_method_round_trip() {
        assert_eq!(StartMethod::parse("spawn").unwrap(), StartMethod::Spawn);
        assert_eq!(StartMethod::parse("fork").unwrap(), StartMethod::Fork);
        assert_eq!(StartMethod::Spawn.as_str(), "spawn");
        assert_eq!(StartMethod::Fork.as_str(), "fork");
    }

    #[test]
    fn start_method_rejects_unknown() {
        assert!(StartMethod::parse("teleport").is_err());
    }

    #[test]
    fn canonical_request_byte_locked_against_python() {
        // PARITY: encode the canonical input + assert byte-identical to
        // the Python-generated reference. SPEC §8.10 byte-identical
        // guarantee enforced. Drift = SPEC violation.
        let bytes = encode_adoption_request_payload(&canonical_request()).unwrap();
        assert_eq!(bytes.len(), CANONICAL_ADOPTION_REQUEST_PAYLOAD_BYTES);
        assert_eq!(bytes, hex_to_bytes(CANONICAL_REQUEST_HEX));
    }

    #[test]
    fn canonical_request_decode_matches_input() {
        // Round-trip from the locked hex vector via decode.
        let bytes = hex_to_bytes(CANONICAL_REQUEST_HEX);
        let decoded = decode_adoption_request_payload(&bytes).unwrap();
        assert_eq!(decoded, canonical_request());
    }

    #[test]
    fn encode_decode_request_round_trip_arbitrary() {
        // Non-canonical inputs round-trip cleanly (decoder doesn't depend
        // on key order or compact-encoding choices).
        let req = AdoptionRequest {
            name: "outer-spirit".to_string(),
            pid: 999999,
            start_method: StartMethod::Fork,
            boot_ts: 0.0,
        };
        let bytes = encode_adoption_request_payload(&req).unwrap();
        let decoded = decode_adoption_request_payload(&bytes).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn decode_request_rejects_non_map() {
        use rmpv::Value;
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::String("not a map".into())).unwrap();
        assert!(decode_adoption_request_payload(&bytes).is_err());
    }

    #[test]
    fn decode_request_rejects_missing_field() {
        use rmpv::Value;
        let map = Value::Map(vec![
            (Value::String("name".into()), Value::String("x".into())),
            // missing pid, start_method, boot_ts
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &map).unwrap();
        assert!(decode_adoption_request_payload(&bytes).is_err());
    }

    #[test]
    fn canonical_ack_byte_locked() {
        // PARITY: locked ack vector matching Python's wire format
        // (msgpack default use_bin_type=True → str8 `d9` marker for 36-char
        // UUID, NOT the older str16 `da` marker). Verified via
        // `python3 -c "import msgpack; ..."` against the canonical input.
        // SPEC §8.4.1 v0.1.5 D-SPEC-31.
        const HEX: &str =
            "83a3726964d92430303030303030302d303030302d303030302d303030302d303030303030303030303030a86163636570746564c3a6726561736f6ec0";
        let ack = AdoptionAck {
            rid: "00000000-0000-0000-0000-000000000000".to_string(),
            accepted: true,
            reason: None,
        };
        let bytes = encode_adoption_ack_payload(&ack).unwrap();
        assert_eq!(bytes.len(), CANONICAL_ADOPTION_ACK_PAYLOAD_BYTES);
        assert_eq!(bytes, hex_to_bytes(HEX));
    }

    #[test]
    fn ack_round_trip_with_rejection_reason() {
        let ack = AdoptionAck {
            rid: "abc-def-123".to_string(),
            accepted: false,
            reason: Some("daemon is shutting down".to_string()),
        };
        let bytes = encode_adoption_ack_payload(&ack).unwrap();
        let decoded = decode_adoption_ack_payload(&bytes).unwrap();
        assert_eq!(decoded, ack);
    }

    #[test]
    fn decode_ack_handles_nil_reason() {
        // accepted=true → reason=Nil in Python; decode must produce None.
        let ack = AdoptionAck {
            rid: "x".to_string(),
            accepted: true,
            reason: None,
        };
        let bytes = encode_adoption_ack_payload(&ack).unwrap();
        let decoded = decode_adoption_ack_payload(&bytes).unwrap();
        assert!(decoded.reason.is_none());
    }
}
