//! identity — Ed25519 keypair load + holder for `titan-kernel-rs`.
//!
//! Per SPEC G16(8) the identity file `data/titan_identity_keypair.json` is
//! **SACRED**: never auto-restored from `.bak`, indefinite backup retention,
//! mode 0600, read-only after boot. Kernel halts boot with exit 3 on
//! corruption (per SPEC §15) — Maker manual recovery required.
//!
//! # Wire format
//!
//! The on-disk JSON file (existing format inherited from Phase A):
//!
//! ```json
//! {
//!   "titan_id": "T1",
//!   "secret_seed_hex": "<64 hex chars = 32 bytes>",
//!   "public_key_hex": "<64 hex chars = 32 bytes>"
//! }
//! ```
//!
//! Optional fields (validated when present): `boot_generation`,
//! `created_at`, `notes`. Unknown fields are ignored (forward-compat).
//!
//! # Zeroize discipline
//!
//! `Identity::secret_seed` is `Zeroizing<[u8; 32]>` — overwrites with zeros
//! when the value is dropped. Prevents accidental key material in core
//! dumps or post-free memory.

use std::path::Path;
use zeroize::Zeroizing;

/// Maximum titan-id length. The id is interpolated into `/dev/shm/titan_<id>/`
/// and the bus/kernel socket paths, so it must stay short + path-safe.
pub const MAX_TITAN_ID_LEN: usize = 32;

/// Titan instance identifier — a validated free-form id. The Maker's fleet
/// uses `T1`/`T2`/`T3`; a sovereign user's Titan picks its own (the public
/// installer defaults to `titan`). This is deliberately NOT a closed enum:
/// hardcoding T1/T2/T3 was a fleet-ism that blocked every Titan not named
/// after the fleet from booting (kernel exited `INVALIDARGUMENT` on
/// `--titan-id titan`). Per SPEC §1 glossary `titan-id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TitanId(String);

/// Validate a titan-id: non-empty, ≤ [`MAX_TITAN_ID_LEN`], and only
/// `[A-Za-z0-9_-]` (path-safe — the id names SHM dirs + UNIX sockets).
/// Returns the trimmed id verbatim on success.
///
/// Shared by [`TitanId::parse`] and the `clap` `value_parser` on every
/// binary's `--titan-id` flag, so the kernel and a spawned worker can never
/// disagree on what a legal id is. `IdentityError` impls `std::error::Error`,
/// so this doubles as a clap value-parser function.
pub fn validate_titan_id(s: &str) -> Result<String, IdentityError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(IdentityError::UnknownTitanId(
            "titan-id is empty".to_string(),
        ));
    }
    if s.len() > MAX_TITAN_ID_LEN {
        return Err(IdentityError::UnknownTitanId(format!(
            "titan-id {s:?} exceeds {MAX_TITAN_ID_LEN} chars ({} given)",
            s.len()
        )));
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(IdentityError::UnknownTitanId(format!(
            "titan-id {s:?} has illegal chars (allowed: A-Z a-z 0-9 _ -)"
        )));
    }
    Ok(s.to_string())
}

impl TitanId {
    /// The id as stored (e.g. `"T1"`, `"titan"`).
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Legacy namespaced form (`"titan_<id>"`). Retained for callers that
    /// still expect it; the bus authkey **no longer** derives from it — the
    /// HKDF info is the constant `b"titan-bus"` (per
    /// `rFP_phase_c_bus_authkey_contract_fix.md`), and per-Titan isolation
    /// comes from the identity keypair, not this string.
    pub fn as_namespace(&self) -> String {
        format!("titan_{}", self.0)
    }

    /// Parse + validate a titan-id from a string. Accepts any path-safe id —
    /// the fleet's `T1`/`T2`/`T3` and a sovereign user's own (`titan`, …).
    pub fn parse(s: &str) -> Result<Self, IdentityError> {
        validate_titan_id(s).map(TitanId)
    }
}

/// Errors during identity load + verification.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// Identity file does not exist or is unreadable.
    #[error("identity file {path}: {kind}")]
    FileAccess {
        /// Path attempted.
        path: String,
        /// I/O error kind.
        kind: String,
    },

    /// Identity file mode is not 0600 (per SPEC G16(8)).
    #[error("identity file {path} has wrong permissions {actual:o} (must be 0600)")]
    BadPermissions {
        /// Path with wrong perms.
        path: String,
        /// Actual mode bits.
        actual: u32,
    },

    /// JSON parse error.
    #[error("identity JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Hex decoding error in `secret_seed_hex` or `public_key_hex`.
    #[error("hex decoding error in field {field}: {err}")]
    HexDecode {
        /// Field name.
        field: &'static str,
        /// Decoding error.
        err: hex::FromHexError,
    },

    /// `secret_seed_hex` length is not 32 bytes (64 hex chars).
    #[error("secret_seed must be 32 bytes, got {actual}")]
    BadSecretSeedLength {
        /// Actual byte count.
        actual: usize,
    },

    /// `public_key_hex` length is not 32 bytes.
    #[error("public_key must be 32 bytes, got {actual}")]
    BadPublicKeyLength {
        /// Actual byte count.
        actual: usize,
    },

    /// `public_key` does not derive from `secret_seed` (sanity check).
    #[error("public_key does not derive from secret_seed")]
    PublicKeyMismatch,

    /// `titan_id` field is missing, empty, too long, or has illegal chars
    /// (must be 1–32 of `[A-Za-z0-9_-]`).
    #[error("invalid titan_id: {0}")]
    UnknownTitanId(String),

    /// Identity file marked corrupted (G16(8) — never auto-restore).
    #[error("identity file appears corrupted; manual Maker recovery required")]
    Corrupted,
}

/// Loaded Titan identity. Holds the Ed25519 keypair + titan_id with zeroize
/// guarantees on the secret material.
pub struct Identity {
    /// Titan instance identifier (the fleet's `T1`/`T2`/`T3`, or a sovereign
    /// user's own id such as `titan`).
    pub titan_id: TitanId,
    /// Ed25519 public key (verifying key) — safe to expose.
    pub verifying_key: ed25519_dalek::VerifyingKey,
    /// Ed25519 secret seed (32 bytes). Zeroized on drop.
    pub secret_seed: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Identity")
            .field("titan_id", &self.titan_id)
            .field(
                "verifying_key_hex",
                &hex::encode(self.verifying_key.as_bytes()),
            )
            .field("secret_seed", &"<redacted>")
            .finish()
    }
}

#[derive(serde::Deserialize)]
struct IdentityFile {
    titan_id: String,
    secret_seed_hex: String,
    public_key_hex: String,
}

impl Identity {
    /// Load an identity from disk (`data/titan_identity_keypair.json` or
    /// equivalent). Backward-compatible alias for
    /// [`Identity::load_from_disk_with_hint`] with `titan_id_hint=None`;
    /// only struct-format files succeed via this path.
    ///
    /// # Errors
    ///
    /// Any failure → kernel exits 3 (identity load failure per SPEC §15).
    /// `.bak` is NEVER tried for identity (G16(8) SACRED file class).
    pub fn load_from_disk(path: &Path) -> Result<Self, IdentityError> {
        Self::load_from_disk_with_hint(path, None)
    }

    /// Load an identity from disk, auto-detecting the on-disk format:
    ///
    /// 1. **Struct format** (canonical, per SPEC §10.A B1):
    ///    ```json
    ///    {
    ///      "titan_id": "T1",
    ///      "secret_seed_hex": "<64 hex chars>",
    ///      "public_key_hex": "<64 hex chars>"
    ///    }
    ///    ```
    ///
    /// 2. **Solana CLI byte-array format** (legacy, what every production
    ///    Titan currently has on disk via `solana-keygen` / migration from
    ///    `~/.config/solana/id.json`):
    ///    ```json
    ///    [byte0, byte1, ..., byte63]
    ///    ```
    ///    First 32 bytes = ed25519 secret seed, last 32 = public key.
    ///    Has no embedded `titan_id` → caller MUST provide `titan_id_hint`
    ///    (typically `cli.titan_id` from systemd `--titan-id %i`).
    ///
    /// Verifies file mode 0600, JSON parse, length checks, and that the
    /// public key derives from the secret seed.
    ///
    /// Per Phase C C-S7 Gap D (2026-05-05) — production Titans store
    /// keypairs in the byte-array format. The Rust kernel must accept both
    /// formats so activation doesn't require a fleet-wide identity
    /// migration. New Titans can adopt the canonical struct format whenever
    /// convenient; both stay supported.
    ///
    /// # Errors
    ///
    /// Any failure → kernel exits 3 (identity load failure per SPEC §15).
    pub fn load_from_disk_with_hint(
        path: &Path,
        titan_id_hint: Option<&str>,
    ) -> Result<Self, IdentityError> {
        // 1. File access + permission check (shared across both formats)
        let meta = std::fs::metadata(path).map_err(|e| IdentityError::FileAccess {
            path: path.display().to_string(),
            kind: e.to_string(),
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = meta.permissions().mode() & 0o777;
            if mode != 0o600 {
                return Err(IdentityError::BadPermissions {
                    path: path.display().to_string(),
                    actual: mode,
                });
            }
        }

        // 2. Read raw bytes
        let text = std::fs::read_to_string(path).map_err(|e| IdentityError::FileAccess {
            path: path.display().to_string(),
            kind: e.to_string(),
        })?;

        // 3. Format dispatch — first non-whitespace char picks the parser:
        //    `{` → struct format (canonical)
        //    `[` → byte-array format (Solana CLI legacy)
        //    anything else → JsonParse error
        let first_char = text.trim_start().chars().next().ok_or_else(|| {
            IdentityError::JsonParse(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "identity file is empty",
            )))
        })?;

        match first_char {
            '{' => Self::parse_struct_format(&text),
            '[' => Self::parse_byte_array_format(&text, titan_id_hint),
            _ => Err(IdentityError::JsonParse(serde_json::Error::io(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "unrecognized identity file format (expected '{{' or '[' \
                         at start of JSON, got {first_char:?})"
                    ),
                ),
            ))),
        }
    }

    /// Parse the canonical struct-format identity file (§10.A B1).
    fn parse_struct_format(text: &str) -> Result<Self, IdentityError> {
        let parsed: IdentityFile = serde_json::from_str(text)?;
        let titan_id = TitanId::parse(&parsed.titan_id)?;

        let secret_seed_bytes =
            hex::decode(&parsed.secret_seed_hex).map_err(|err| IdentityError::HexDecode {
                field: "secret_seed_hex",
                err,
            })?;
        if secret_seed_bytes.len() != 32 {
            return Err(IdentityError::BadSecretSeedLength {
                actual: secret_seed_bytes.len(),
            });
        }
        let mut secret_seed_array = [0u8; 32];
        secret_seed_array.copy_from_slice(&secret_seed_bytes);

        let pk_bytes =
            hex::decode(&parsed.public_key_hex).map_err(|err| IdentityError::HexDecode {
                field: "public_key_hex",
                err,
            })?;
        if pk_bytes.len() != 32 {
            return Err(IdentityError::BadPublicKeyLength {
                actual: pk_bytes.len(),
            });
        }
        let mut pk_array = [0u8; 32];
        pk_array.copy_from_slice(&pk_bytes);

        Self::build_with_pk_check(titan_id, secret_seed_array, pk_array)
    }

    /// Parse a Solana CLI byte-array-format identity file. titan_id comes
    /// from the caller's hint (the file format has no embedded titan_id).
    fn parse_byte_array_format(
        text: &str,
        titan_id_hint: Option<&str>,
    ) -> Result<Self, IdentityError> {
        let bytes: Vec<u8> = serde_json::from_str(text).map_err(IdentityError::JsonParse)?;
        // Solana CLI keypair = 64 bytes total: secret seed (32) + public key (32).
        if bytes.len() != 64 {
            return Err(IdentityError::BadSecretSeedLength {
                actual: bytes.len(),
            });
        }

        let titan_id_str = titan_id_hint.ok_or_else(|| {
            // Byte-array format has no embedded titan_id — caller must
            // provide one. Without a hint we cannot construct an Identity.
            IdentityError::UnknownTitanId(
                "byte-array format requires titan_id hint (pass --titan-id)".into(),
            )
        })?;
        let titan_id = TitanId::parse(titan_id_str)?;

        let mut secret_seed_array = [0u8; 32];
        secret_seed_array.copy_from_slice(&bytes[0..32]);
        let mut pk_array = [0u8; 32];
        pk_array.copy_from_slice(&bytes[32..64]);

        Self::build_with_pk_check(titan_id, secret_seed_array, pk_array)
    }

    /// Final construction step shared by both formats — derives the public
    /// key from the secret seed and verifies it matches the stored public
    /// key (sanity check that secret + public are a valid pair).
    fn build_with_pk_check(
        titan_id: TitanId,
        secret_seed_array: [u8; 32],
        pk_array: [u8; 32],
    ) -> Result<Self, IdentityError> {
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_array)
            .map_err(|_| IdentityError::Corrupted)?;

        // Sanity: derived pk must equal stored pk (G16 invariant).
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret_seed_array);
        if signing.verifying_key().as_bytes() != verifying_key.as_bytes() {
            return Err(IdentityError::PublicKeyMismatch);
        }

        Ok(Identity {
            titan_id,
            verifying_key,
            secret_seed: Zeroizing::new(secret_seed_array),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titan_id_parse() {
        // Fleet ids and a sovereign user's own id all parse verbatim.
        assert_eq!(TitanId::parse("T1").unwrap().as_str(), "T1");
        assert_eq!(TitanId::parse("titan").unwrap().as_str(), "titan");
        assert_eq!(
            TitanId::parse("titan-research_2").unwrap().as_str(),
            "titan-research_2"
        );
        // Genuinely invalid ids are rejected (path-unsafe / empty / too long).
        assert!(matches!(
            TitanId::parse(""),
            Err(IdentityError::UnknownTitanId(_))
        ));
        assert!(matches!(
            TitanId::parse("bad/id"),
            Err(IdentityError::UnknownTitanId(_))
        ));
        assert!(matches!(
            TitanId::parse("has space"),
            Err(IdentityError::UnknownTitanId(_))
        ));
        assert!(matches!(
            TitanId::parse("a.b"),
            Err(IdentityError::UnknownTitanId(_))
        ));
        assert!(matches!(
            TitanId::parse(&"x".repeat(MAX_TITAN_ID_LEN + 1)),
            Err(IdentityError::UnknownTitanId(_))
        ));
    }

    #[test]
    fn titan_id_namespace() {
        assert_eq!(TitanId::parse("T1").unwrap().as_namespace(), "titan_T1");
        assert_eq!(
            TitanId::parse("titan").unwrap().as_namespace(),
            "titan_titan"
        );
    }

    #[test]
    fn titan_id_str() {
        assert_eq!(TitanId::parse("T1").unwrap().as_str(), "T1");
    }

    #[test]
    fn load_succeeds_with_valid_keypair() {
        use rand::RngCore;
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        // Generate a valid Ed25519 keypair for the test (32-byte secret seed)
        let mut secret_seed = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_seed);
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret_seed);
        let secret_seed_hex = hex::encode(signing.to_bytes());
        let public_key_hex = hex::encode(signing.verifying_key().as_bytes());

        let json = format!(
            r#"{{
  "titan_id": "T1",
  "secret_seed_hex": "{secret_seed_hex}",
  "public_key_hex": "{public_key_hex}"
}}"#
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        let identity = Identity::load_from_disk(tmp.path()).unwrap();
        assert_eq!(identity.titan_id.as_str(), "T1");
    }

    #[test]
    fn load_rejects_wrong_permissions() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file()
            .write_all(
                b"{\"titan_id\":\"T1\",\"secret_seed_hex\":\"00\",\"public_key_hex\":\"00\"}",
            )
            .unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o644)).unwrap();

        let result = Identity::load_from_disk(tmp.path());
        assert!(matches!(
            result,
            Err(IdentityError::BadPermissions { actual: 0o644, .. })
        ));
    }

    // ─── Phase C C-S7 Gap D — byte-array format auto-detection tests ───

    /// Helper: write a Solana CLI byte-array-format keypair to a tempfile
    /// with 0600 perms. Returns (tmpfile, secret_seed). The keypair is
    /// generated freshly so every test gets a valid (seed, pk) pair.
    fn write_byte_array_keypair() -> (tempfile::NamedTempFile, [u8; 32]) {
        use rand::RngCore;
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let mut secret_seed = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_seed);
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret_seed);
        let pk = signing.verifying_key().to_bytes();

        // Solana CLI format: 64-byte JSON array (secret seed | public key).
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&secret_seed);
        bytes.extend_from_slice(&pk);
        let json = serde_json::to_string(&bytes).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
        (tmp, secret_seed)
    }

    #[test]
    fn byte_array_format_loads_with_titan_id_hint() {
        let (tmp, _seed) = write_byte_array_keypair();
        let identity = Identity::load_from_disk_with_hint(tmp.path(), Some("T3")).unwrap();
        assert_eq!(identity.titan_id.as_str(), "T3");
    }

    #[test]
    fn byte_array_format_loads_with_sovereign_titan_id() {
        // A sovereign user's Titan (id = "titan") boots from the Solana
        // byte-array keypair — the exact path that exit-2'd before the
        // free-form fix.
        let (tmp, _seed) = write_byte_array_keypair();
        let identity = Identity::load_from_disk_with_hint(tmp.path(), Some("titan")).unwrap();
        assert_eq!(identity.titan_id.as_str(), "titan");
    }

    #[test]
    fn byte_array_format_without_hint_errors_with_unknown_titan_id() {
        // Backward-compat alias `load_from_disk` provides no hint → byte-array
        // file fails with UnknownTitanId (the format has no embedded titan_id).
        let (tmp, _seed) = write_byte_array_keypair();
        let result = Identity::load_from_disk(tmp.path());
        assert!(
            matches!(result, Err(IdentityError::UnknownTitanId(_))),
            "expected UnknownTitanId error for byte-array without hint, got {result:?}"
        );
    }

    #[test]
    fn byte_array_format_with_invalid_titan_id_hint_errors() {
        // "bad/id" has a path-unsafe char → rejected (note: "T9" is now a
        // VALID free-form id, so the invalid case must be genuinely illegal).
        let (tmp, _seed) = write_byte_array_keypair();
        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("bad/id"));
        assert!(matches!(result, Err(IdentityError::UnknownTitanId(_))));
    }

    #[test]
    fn byte_array_format_with_pk_mismatch_errors() {
        use rand::RngCore;
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        // Random secret + RANDOM (mismatched) public key.
        let mut secret_seed = [0u8; 32];
        let mut wrong_pk = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_seed);
        rand::rngs::OsRng.fill_bytes(&mut wrong_pk);

        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&secret_seed);
        bytes.extend_from_slice(&wrong_pk);
        let json = serde_json::to_string(&bytes).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("T1"));
        // Wrong PK first hits the VerifyingKey::from_bytes (Corrupted) OR
        // the signing-derived check (PublicKeyMismatch). Either is acceptable
        // since both indicate the keypair is invalid.
        assert!(
            matches!(
                result,
                Err(IdentityError::PublicKeyMismatch) | Err(IdentityError::Corrupted)
            ),
            "expected pk-mismatch error, got {result:?}"
        );
    }

    #[test]
    fn byte_array_format_with_wrong_byte_count_errors() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let bytes: Vec<u8> = (0..32).collect(); // 32 bytes — wrong length
        let json = serde_json::to_string(&bytes).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("T1"));
        assert!(
            matches!(
                result,
                Err(IdentityError::BadSecretSeedLength { actual: 32 })
            ),
            "expected BadSecretSeedLength{{actual:32}}, got {result:?}"
        );
    }

    #[test]
    fn struct_format_still_works_via_with_hint() {
        // Hint is ignored when struct format is detected — embedded titan_id wins.
        use rand::RngCore;
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let mut secret_seed = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_seed);
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret_seed);
        let secret_seed_hex = hex::encode(signing.to_bytes());
        let public_key_hex = hex::encode(signing.verifying_key().as_bytes());

        let json = format!(
            r#"{{"titan_id":"T2","secret_seed_hex":"{secret_seed_hex}","public_key_hex":"{public_key_hex}"}}"#
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        // Hint says T1 but file says T2 — file wins for struct format.
        let identity = Identity::load_from_disk_with_hint(tmp.path(), Some("T1")).unwrap();
        assert_eq!(identity.titan_id.as_str(), "T2");
    }

    #[test]
    fn byte_array_perm_check_still_enforced() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let bytes: Vec<u8> = (0..64).collect();
        let json = serde_json::to_string(&bytes).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        // 0644 — must still be rejected even before format dispatch.
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o644)).unwrap();

        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("T1"));
        assert!(
            matches!(
                result,
                Err(IdentityError::BadPermissions { actual: 0o644, .. })
            ),
            "expected BadPermissions, got {result:?}"
        );
    }

    #[test]
    fn unrecognized_format_errors() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(b"\"just a string\"").unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("T1"));
        assert!(
            matches!(result, Err(IdentityError::JsonParse(_))),
            "expected JsonParse error for unrecognized format, got {result:?}"
        );
    }
}
