//! authkey — HKDF-SHA256 derivation of the bus authkey from Titan's identity.
//!
//! Byte-identical port of `titan_plugin/core/bus_authkey.py` (B.2 §D1).
//! Uses RustCrypto's `hkdf` + `sha2` — pure Rust, no system dependencies.
//!
//! # Properties
//!
//! - **Recoverable** — Shamir-restore identity → same `secret_bytes` → same
//!   authkey. No second secret to lose.
//! - **Per-titan isolated** — T1, T2, T3 different identity keypair → different
//!   IKM → different authkeys. Per-Titan isolation comes from the IDENTITY
//!   SECRET (different per Titan), NOT from a per-Titan info field. The HKDF
//!   info is a fixed constant `b"titan-bus"` per `PLAN_microkernel_phase_c_s2_kernel.md §7.3`.
//! - **Rotation** — bump `AUTHKEY_HKDF_SALT` to `b"titan-bus-v2"` + restart.
//!   No migration drama.
//! - **Phase C portable** — RustCrypto `hkdf` produces byte-identical output
//!   for the same inputs as Python's stdlib `hmac` + `hashlib.sha256` when
//!   used in the canonical HKDF-SHA256 form (RFC 5869).
//!
//! # Constants
//!
//! Per SPEC §3.1 D06 + canonical TOML:
//! - `AUTHKEY_HKDF_SALT = b"titan-bus-v1"` (version-bumpable; was Python's `BUS_AUTHKEY_SALT`)
//! - `AUTHKEY_HKDF_INFO = b"titan-bus"` (domain separation constant)
//! - `AUTHKEY_BYTES = 32` (was Python's `BUS_AUTHKEY_LEN`)
//!
//! # 2026-05-05 contract fix
//!
//! This file previously took `titan_id: &str` as a parameter and used it as
//! the HKDF info. The runtime call site in `titan-kernel-rs/src/kernel.rs`
//! passed `identity.titan_id.as_namespace()` ("titan_T3") while the Python
//! worker call site in `worker_bus_bootstrap.py` passed `os.environ["TITAN_BUS_TITAN_ID"]`
//! ("T3") — different formats produced different authkeys, breaking the
//! handshake under l0_rust_enabled=true. The PLAN's canonical design (line 681
//! of `PLAN_microkernel_phase_c_s2_kernel.md §7.3`) had `info=b"titan-bus"` as
//! a constant from the beginning; the implementation drifted. This rFP
//! restores the constant. See `titan-docs/rFP_phase_c_bus_authkey_contract_fix.md`.
//!
//! # Parity vectors
//!
//! `tests/parity/vectors.json::T1_authkey_derivation` (Titan-canonical) +
//! `authkey.rfc5869` (RFC 5869 reference). Verified byte-identical to
//! `derive_bus_authkey()` in Python by `cargo test` + `pytest` running
//! against the same JSON file.

use hkdf::Hkdf;
use sha2::Sha256;

pub use crate::constants::{AUTHKEY_BYTES, AUTHKEY_HKDF_INFO, AUTHKEY_HKDF_SALT};

/// Errors during authkey derivation.
#[derive(Debug, thiserror::Error)]
pub enum AuthkeyError {
    /// `identity_secret_bytes` was empty.
    #[error("identity_secret_bytes is empty — cannot derive authkey")]
    EmptyIdentity,
    /// Internal HKDF expand error (only fires for output > 255*32 bytes; not
    /// reachable with our 32-byte output).
    #[error("HKDF-Expand error: {0}")]
    HkdfExpand(String),
}

/// Derive the bus HMAC authkey from Titan's identity secret via HKDF-SHA256.
///
/// # Arguments
///
/// - `identity_secret_bytes` — Ed25519 secret bytes (32 or 64 typical;
///   HKDF accepts arbitrary length). Must be the IKM, NOT a public key.
///
/// # Returns
///
/// `[u8; AUTHKEY_BYTES]` (32 bytes). Deterministic for fixed input +
/// fixed `AUTHKEY_HKDF_SALT` + fixed `AUTHKEY_HKDF_INFO`. Recoverable across
/// kernel swaps as long as the identity keypair is preserved.
///
/// Per-Titan isolation comes from the IDENTITY KEYPAIR (different secret
/// per Titan → different IKM → different authkey). NOT from the info field —
/// that is a fixed `b"titan-bus"` constant per PLAN §7.3.
///
/// # Byte-identical to Python `derive_bus_authkey()`
///
/// Verified by parity vectors at `tests/parity/vectors.json`. Both
/// implementations follow RFC 5869 standard HKDF-SHA256.
pub fn derive_bus_authkey(
    identity_secret_bytes: &[u8],
) -> Result<[u8; AUTHKEY_BYTES as usize], AuthkeyError> {
    if identity_secret_bytes.is_empty() {
        return Err(AuthkeyError::EmptyIdentity);
    }

    let hk = Hkdf::<Sha256>::new(Some(AUTHKEY_HKDF_SALT), identity_secret_bytes);
    let mut okm = [0u8; AUTHKEY_BYTES as usize];
    hk.expand(AUTHKEY_HKDF_INFO, &mut okm)
        .map_err(|e| AuthkeyError::HkdfExpand(format!("{e:?}")))?;
    Ok(okm)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ─────────────────────────────────────────────────────────

    #[test]
    fn constants_match_spec_v1() {
        assert_eq!(AUTHKEY_BYTES, 32);
        assert_eq!(AUTHKEY_HKDF_SALT, b"titan-bus-v1");
        assert_eq!(AUTHKEY_HKDF_INFO, b"titan-bus");
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_inputs() {
        let secret = b"identity-secret-32-bytes-exactly";
        let a = derive_bus_authkey(secret).unwrap();
        let b = derive_bus_authkey(secret).unwrap();
        assert_eq!(a, b);
    }

    // ── Output length ─────────────────────────────────────────────────────

    #[test]
    fn output_always_32_bytes() {
        for secret_len in [1, 16, 32, 64, 128, 1024] {
            let secret = vec![0x42u8; secret_len];
            let key = derive_bus_authkey(&secret).unwrap();
            assert_eq!(key.len(), AUTHKEY_BYTES as usize);
        }
    }

    // ── Per-Titan isolation via identity secrets ──────────────────────────

    #[test]
    fn identity_secret_isolation() {
        // Different identity → different authkeys (the per-Titan isolation path).
        let k1 = derive_bus_authkey(b"identity-secret-AAAAAAAAAAAAAAAAAA").unwrap();
        let k2 = derive_bus_authkey(b"identity-secret-BBBBBBBBBBBBBBBBBB").unwrap();
        assert_ne!(k1, k2);
    }

    // ── Input validation ──────────────────────────────────────────────────

    #[test]
    fn empty_secret_rejected() {
        let result = derive_bus_authkey(b"");
        assert!(matches!(result, Err(AuthkeyError::EmptyIdentity)));
    }
}
