//! message — On-the-wire bus message envelope + decoder.
//!
//! Byte-identical to Python `bus_socket.py` on-wire format:
//! ```text
//! [u32 LE length][msgpack-encoded dict with at least: "type", "src", "dst"]
//! ```
//!
//! The broker doesn't need to fully deserialize the payload — it only routes
//! messages by `dst` + `type` + drops them per priority lane. Therefore we
//! decode JUST the header fields we need + keep the original bytes for
//! fanout. Subscribers receive byte-identical msgpack to what was sent.

use rmpv::Value as MpValue;

/// Header fields the broker extracts from an inbound msgpack message.
#[derive(Debug, Clone, Default)]
pub struct MsgHeader {
    /// Message type (e.g. `"BUS_SUBSCRIBE"`, `"BODY_STATE"`).
    pub msg_type: Option<String>,
    /// Source identifier (publisher name).
    pub src: Option<String>,
    /// Destination — `Some("all")` or `None` = fanout; otherwise a specific
    /// subscriber name.
    pub dst: Option<String>,
}

/// Errors during message decoding.
#[derive(Debug, thiserror::Error)]
pub enum MsgError {
    /// msgpack decode failed.
    #[error("msgpack decode: {0}")]
    Decode(String),

    /// Top-level msgpack value is not a map (Python broker rejects non-dict).
    #[error("expected msgpack map at top level, got {got}")]
    NotAMap {
        /// Type name of what was actually received.
        got: &'static str,
    },
}

/// Decode just the header fields from raw msgpack bytes.
///
/// Returns `Ok(MsgHeader)` even if some fields are missing (they're all
/// `Option`). Returns `Err` only if the bytes are not valid msgpack OR the
/// top-level value is not a map (matches Python broker behavior).
///
/// The raw bytes themselves are NOT consumed; caller keeps them for fanout.
pub fn decode_header(bytes: &[u8]) -> Result<MsgHeader, MsgError> {
    let value: MpValue = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes))
        .map_err(|e| MsgError::Decode(format!("{e:?}")))?;
    let map = match value {
        MpValue::Map(m) => m,
        other => {
            return Err(MsgError::NotAMap {
                got: msgpack_type_name(&other),
            })
        }
    };

    let mut hdr = MsgHeader::default();
    for (k, v) in map {
        if let MpValue::String(key_s) = k {
            if let Some(key_str) = key_s.as_str() {
                match key_str {
                    "type" => hdr.msg_type = mp_value_as_string(&v),
                    "src" => hdr.src = mp_value_as_string(&v),
                    "dst" => hdr.dst = mp_value_as_string(&v),
                    _ => {} // ignore unknown
                }
            }
        }
    }
    Ok(hdr)
}

fn mp_value_as_string(v: &MpValue) -> Option<String> {
    match v {
        MpValue::String(s) => s.as_str().map(|s| s.to_string()),
        _ => None,
    }
}

fn msgpack_type_name(v: &MpValue) -> &'static str {
    match v {
        MpValue::Nil => "nil",
        MpValue::Boolean(_) => "boolean",
        MpValue::Integer(_) => "integer",
        MpValue::F32(_) | MpValue::F64(_) => "float",
        MpValue::String(_) => "string",
        MpValue::Binary(_) => "binary",
        MpValue::Array(_) => "array",
        MpValue::Map(_) => "map",
        MpValue::Ext(_, _) => "ext",
    }
}

/// Re-encode a msgpack-encoded message with its `type` field replaced (or
/// inserted) by `new_msg_type`. Other fields (including unknown ones) are
/// preserved byte-identically. Used by the broker's drift-bridge dual-emit
/// path: same payload, different `type` per bridged copy.
///
/// Returns `Err(MsgError::NotAMap)` if the original isn't a top-level map.
pub fn rewrite_msg_type(raw_bytes: &[u8], new_msg_type: &str) -> Result<Vec<u8>, MsgError> {
    let value: MpValue = rmpv::decode::read_value(&mut std::io::Cursor::new(raw_bytes))
        .map_err(|e| MsgError::Decode(format!("{e:?}")))?;
    let mut map = match value {
        MpValue::Map(m) => m,
        other => {
            return Err(MsgError::NotAMap {
                got: msgpack_type_name(&other),
            })
        }
    };

    let mut found = false;
    for (k, v) in map.iter_mut() {
        if let MpValue::String(key_s) = k {
            if key_s.as_str() == Some("type") {
                *v = MpValue::String(new_msg_type.into());
                found = true;
                break;
            }
        }
    }
    if !found {
        // No existing "type" key — insert one
        map.push((
            MpValue::String("type".into()),
            MpValue::String(new_msg_type.into()),
        ));
    }

    let mut out = Vec::with_capacity(raw_bytes.len() + 8);
    rmpv::encode::write_value(&mut out, &MpValue::Map(map))
        .map_err(|e| MsgError::Decode(format!("{e:?}")))?;
    Ok(out)
}

/// Encode a simple `{type, src, dst, payload}` map as msgpack. Used by
/// integration tests + helpers. Production code typically receives bytes
/// pre-encoded by the publisher.
pub fn encode_simple(
    msg_type: &str,
    src: Option<&str>,
    dst: Option<&str>,
    payload: Option<&[u8]>,
) -> Result<Vec<u8>, MsgError> {
    let mut entries = Vec::with_capacity(4);
    entries.push((
        MpValue::String("type".into()),
        MpValue::String(msg_type.into()),
    ));
    if let Some(s) = src {
        entries.push((MpValue::String("src".into()), MpValue::String(s.into())));
    }
    if let Some(d) = dst {
        entries.push((MpValue::String("dst".into()), MpValue::String(d.into())));
    }
    if let Some(p) = payload {
        entries.push((
            MpValue::String("payload".into()),
            MpValue::Binary(p.to_vec()),
        ));
    }
    let val = MpValue::Map(entries);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &val).map_err(|e| MsgError::Decode(format!("{e:?}")))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_simple_message() {
        let bytes = encode_simple(
            "BUS_SUBSCRIBE",
            Some("inner-body"),
            Some("all"),
            Some(b"payload"),
        )
        .unwrap();
        let hdr = decode_header(&bytes).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("BUS_SUBSCRIBE"));
        assert_eq!(hdr.src.as_deref(), Some("inner-body"));
        assert_eq!(hdr.dst.as_deref(), Some("all"));
    }

    #[test]
    fn decode_missing_fields_returns_none() {
        let bytes = encode_simple("BUS_PING", None, None, None).unwrap();
        let hdr = decode_header(&bytes).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("BUS_PING"));
        assert!(hdr.src.is_none());
        assert!(hdr.dst.is_none());
    }

    #[test]
    fn decode_rejects_non_map() {
        // msgpack-encoded array (not a map) — broker rejects
        let bytes = vec![0x91, 0xa3, b'h', b'i', b'!']; // [["hi!"]]
        let result = decode_header(&bytes);
        assert!(matches!(result, Err(MsgError::NotAMap { .. })));
    }

    #[test]
    fn decode_rejects_garbage() {
        // Goal: every garbage-byte input must produce SOME error (either
        // Decode or NotAMap — both are "broker rejects, closes connection"
        // outcomes). rmpv is lenient about reserved bytes (0xc1 → nil), but
        // any path through the broker still rejects via NotAMap.
        for (label, bytes) in [
            ("empty", vec![]),
            ("0xc1 reserved", vec![0xc1]),
            ("truncated map header", vec![0xde, 0xff, 0xff]),
            (
                "negfixint sequence (decodes as int -1, then NotAMap)",
                vec![0xff, 0xff, 0xff, 0xff],
            ),
        ] {
            let result = decode_header(&bytes);
            assert!(result.is_err(), "{label}: expected error; got {result:?}");
        }
    }

    #[test]
    fn encode_decode_round_trip_preserves_bytes() {
        let bytes = encode_simple("BODY_STATE", Some("inner-body"), Some("all"), None).unwrap();
        let bytes2 = encode_simple("BODY_STATE", Some("inner-body"), Some("all"), None).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn rewrite_msg_type_replaces_existing_type() {
        let original =
            encode_simple("EPOCH_TICK", Some("kernel"), Some("all"), Some(b"payload")).unwrap();
        let rewritten = rewrite_msg_type(&original, "KERNEL_EPOCH_TICK").unwrap();
        let hdr = decode_header(&rewritten).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("KERNEL_EPOCH_TICK"));
        // Other fields preserved
        assert_eq!(hdr.src.as_deref(), Some("kernel"));
        assert_eq!(hdr.dst.as_deref(), Some("all"));
    }

    #[test]
    fn rewrite_msg_type_inserts_when_missing() {
        // Build a map that has src but no type
        let original = encode_simple("PLACEHOLDER", Some("x"), None, None).unwrap();
        // Manually construct one without "type": an empty map
        let mut empty_map = Vec::new();
        rmpv::encode::write_value(&mut empty_map, &MpValue::Map(vec![])).unwrap();
        let rewritten = rewrite_msg_type(&empty_map, "INSERTED").unwrap();
        let hdr = decode_header(&rewritten).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("INSERTED"));
        // Original variable used to silence unused-warning
        let _ = original;
    }

    #[test]
    fn rewrite_msg_type_rejects_non_map_input() {
        let bytes = vec![0x91, 0xa3, b'h', b'i', b'!']; // array, not map
        let result = rewrite_msg_type(&bytes, "ANY");
        assert!(matches!(result, Err(MsgError::NotAMap { .. })));
    }
}
