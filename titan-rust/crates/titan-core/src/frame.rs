//! frame — Length-prefix + HMAC-SHA256 challenge-response framing.
//!
//! Byte-identical port of `titan_hcl/core/_frame.py` (B.2 protocol).
//! SPEC §8.10 guarantees Rust output matches Python output bit-for-bit
//! for every input, verified by parity vectors at `tests/parity/vectors.json`.
//!
//! # Wire format (every message, every direction)
//!
//! ```text
//! [0:4]   uint32 LE   length (bytes of payload)
//! [4:N]   bytes       msgpack-encoded payload
//! ```
//!
//! No per-frame HMAC — the HMAC challenge-response is per-connection at
//! handshake time only. See [`compute_hmac`] for the handshake primitive.
//!
//! # HMAC challenge-response handshake (per connection)
//!
//! 1. Server sends [`FRAME_CHALLENGE_BYTES`] random nonce
//! 2. Client sends `HMAC-SHA256(authkey, challenge)` ([`FRAME_AUTH_TAG_BYTES`])
//! 3. Server compares with [`constant_time_eq`]; closes connection on mismatch
//!
//! # Phase C portability
//!
//! All constants here come from [`crate::constants`] (auto-generated from
//! `SPEC_titan_architecture_constants.toml` per SPEC §19.2). Python
//! `_frame.py` constants alias to these names via SPEC §3.1 D07/D23 in
//! `titan_hcl/_phase_c_drift_aliases.py` (C2-7).

use hmac::{Hmac, Mac};
use sha2::Sha256;

// Re-export constants from auto-generated module under their canonical names
// (per SPEC §3.1 D07 + D23). Subtle: the constants file uses canonical names
// (`FRAME_*`); this module just re-exports them with `pub use`.
pub use crate::constants::{
    FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES, FRAME_LENGTH_PREFIX_BYTES, FRAME_MAX_FRAME_BYTES,
};

type HmacSha256 = Hmac<Sha256>;

/// Errors that can occur during frame send/receive.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// Payload exceeds `FRAME_MAX_FRAME_BYTES`.
    #[error("frame {actual}B exceeds FRAME_MAX_FRAME_BYTES {max}B")]
    TooLarge {
        /// Attempted frame length in bytes.
        actual: usize,
        /// Configured maximum.
        max: u64,
    },

    /// Peer closed connection before full frame arrived.
    #[error("peer closed connection after {got}/{expected} bytes")]
    PeerClosed {
        /// Bytes received before close.
        got: usize,
        /// Bytes expected.
        expected: usize,
    },

    /// Underlying I/O error.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Compute HMAC-SHA256 of `data` with `key`. Returns 32 bytes.
///
/// Used at connection-handshake time (server sends random `challenge`, client
/// returns `compute_hmac(authkey, challenge)`).
///
/// # Byte-identical to Python `_frame.compute_hmac()`
///
/// Verified by parity vectors at `tests/parity/vectors.json::frame.rfc4231`
/// + `frame.titan_canonical`.
pub fn compute_hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    let result = mac.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Constant-time equality check on two byte slices. Returns `false` on length
/// mismatch (without reading further). Defends against timing-oracle attacks
/// during HMAC verification.
///
/// Uses `subtle`-style impl via `hmac` crate's internal compare. Falls through
/// to manual XOR-accumulator pattern if lengths match.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Encode a payload into wire bytes: `[length: u32 LE][payload]`.
///
/// Byte-identical to Python `_frame.send_frame()` (which `sendall(prefix +
/// payload)` in one syscall).
///
/// Returns `FrameError::TooLarge` if payload exceeds `FRAME_MAX_FRAME_BYTES`.
pub fn encode_frame(payload: &[u8]) -> Result<Vec<u8>, FrameError> {
    if payload.len() as u64 > FRAME_MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            actual: payload.len(),
            max: FRAME_MAX_FRAME_BYTES,
        });
    }
    let mut buf = Vec::with_capacity(FRAME_LENGTH_PREFIX_BYTES as usize + payload.len());
    let len_le = (payload.len() as u32).to_le_bytes();
    buf.extend_from_slice(&len_le);
    buf.extend_from_slice(payload);
    Ok(buf)
}

/// Decode the length prefix from the first `FRAME_LENGTH_PREFIX_BYTES` bytes.
///
/// Returns the announced payload length. Caller is responsible for reading
/// exactly that many bytes next. Returns `FrameError::TooLarge` if announced
/// length exceeds `FRAME_MAX_FRAME_BYTES` (defensive against malicious peer).
pub fn decode_length_prefix(prefix_bytes: &[u8]) -> Result<u32, FrameError> {
    if prefix_bytes.len() < FRAME_LENGTH_PREFIX_BYTES as usize {
        return Err(FrameError::PeerClosed {
            got: prefix_bytes.len(),
            expected: FRAME_LENGTH_PREFIX_BYTES as usize,
        });
    }
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&prefix_bytes[..4]);
    let n = u32::from_le_bytes(len_bytes);
    if n as u64 > FRAME_MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            actual: n as usize,
            max: FRAME_MAX_FRAME_BYTES,
        });
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants locked (must match SPEC + Python) ─────────────────────

    #[test]
    fn locked_constants_v1() {
        // SPEC §3.1 D07 + D23 + canonical TOML constants
        assert_eq!(FRAME_CHALLENGE_BYTES, 32);
        assert_eq!(FRAME_AUTH_TAG_BYTES, 32);
        assert_eq!(FRAME_LENGTH_PREFIX_BYTES, 4);
        assert_eq!(FRAME_MAX_FRAME_BYTES, 16 * 1024 * 1024);
    }

    // ── HMAC RFC 4231 vectors ───────────────────────────────────────────

    #[test]
    fn rfc4231_test_case_1() {
        // RFC 4231 § Test Case 1: HMAC-SHA256(0x0b * 20, "Hi There")
        let key = vec![0x0bu8; 20];
        let data = b"Hi There";
        let expected =
            hex::decode("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7")
                .unwrap();
        let actual = compute_hmac(&key, data);
        assert_eq!(&actual[..], &expected[..]);
    }

    #[test]
    fn rfc4231_test_case_2() {
        // RFC 4231 § Test Case 2: HMAC-SHA256("Jefe", "what do ya want for nothing?")
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let expected =
            hex::decode("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
                .unwrap();
        let actual = compute_hmac(key, data);
        assert_eq!(&actual[..], &expected[..]);
    }

    // ── Frame encoding ───────────────────────────────────────────────────

    #[test]
    fn encode_empty_frame() {
        let bytes = encode_frame(&[]).unwrap();
        // [0, 0, 0, 0] = u32 LE 0
        assert_eq!(bytes, vec![0, 0, 0, 0]);
    }

    #[test]
    fn encode_known_frame() {
        let payload = b"hello";
        let bytes = encode_frame(payload).unwrap();
        // [5, 0, 0, 0, 'h', 'e', 'l', 'l', 'o']
        assert_eq!(bytes, vec![5, 0, 0, 0, b'h', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn encode_rejects_oversized() {
        let payload = vec![0u8; (FRAME_MAX_FRAME_BYTES + 1) as usize];
        let result = encode_frame(&payload);
        match result {
            Err(FrameError::TooLarge { actual, max }) => {
                assert_eq!(actual, payload.len());
                assert_eq!(max, FRAME_MAX_FRAME_BYTES);
            }
            _ => panic!("expected TooLarge"),
        }
    }

    #[test]
    fn decode_known_prefix() {
        let bytes = vec![42, 0, 0, 0];
        assert_eq!(decode_length_prefix(&bytes).unwrap(), 42);
    }

    #[test]
    fn decode_rejects_oversized_announcement() {
        let bytes = (FRAME_MAX_FRAME_BYTES as u32 + 1).to_le_bytes().to_vec();
        let result = decode_length_prefix(&bytes);
        assert!(matches!(result, Err(FrameError::TooLarge { .. })));
    }

    // ── Constant-time eq ─────────────────────────────────────────────────

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hellow"));
        assert!(constant_time_eq(b"", b""));
    }

    // ── Round-trip ───────────────────────────────────────────────────────

    #[test]
    fn round_trip_known_payload() {
        let payload = b"The quick brown fox jumps over the lazy dog";
        let wire = encode_frame(payload).unwrap();
        let announced_len = decode_length_prefix(&wire[..4]).unwrap();
        assert_eq!(announced_len as usize, payload.len());
        assert_eq!(&wire[4..], payload);
    }
}
