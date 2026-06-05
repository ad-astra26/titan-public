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

/// Titan instance identifier. Constrained to the canonical set per SPEC §1
/// glossary `titan-id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitanId {
    /// Titan 1 (origin instance).
    T1,
    /// Titan 2 (mid-cluster).
    T2,
    /// Titan 3 (T3 cluster).
    T3,
}

impl TitanId {
    /// Returns the canonical lowercase form (e.g. `"T1"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TitanId::T1 => "T1",
            TitanId::T2 => "T2",
            TitanId::T3 => "T3",
        }
    }

    /// Returns the namespaced form used as HKDF info per `bus_authkey.py`
    /// (e.g. `"titan_T1"`).
    pub fn as_namespace(&self) -> &'static str {
        match self {
            TitanId::T1 => "titan_T1",
            TitanId::T2 => "titan_T2",
            TitanId::T3 => "titan_T3",
        }
    }

    /// Parse from a string. Accepts both bare (`"T1"`) and namespaced
    /// (`"titan_T1"`) forms.
    pub fn parse(s: &str) -> Result<Self, IdentityError> {
        match s {
            "T1" | "titan_T1" => Ok(TitanId::T1),
            "T2" | "titan_T2" => Ok(TitanId::T2),
            "T3" | "titan_T3" => Ok(TitanId::T3),
            other => Err(IdentityError::UnknownTitanId(other.to_string())),
        }
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

    /// `titan_id` field is missing or unknown (not T1/T2/T3).
    #[error("unknown or missing titan_id: {0}")]
    UnknownTitanId(String),

    /// Identity file marked corrupted (G16(8) — never auto-restore).
    #[error("identity file appears corrupted; manual Maker recovery required")]
    Corrupted,
}

/// Loaded Titan identity. Holds the Ed25519 keypair + titan_id with zeroize
/// guarantees on the secret material.
pub struct Identity {
    /// Titan instance identifier (T1/T2/T3).
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
        assert_eq!(TitanId::parse("T1").unwrap(), TitanId::T1);
        assert_eq!(TitanId::parse("titan_T2").unwrap(), TitanId::T2);
        assert!(matches!(
            TitanId::parse("T9"),
            Err(IdentityError::UnknownTitanId(_))
        ));
    }

    #[test]
    fn titan_id_namespace() {
        assert_eq!(TitanId::T1.as_namespace(), "titan_T1");
        assert_eq!(TitanId::T2.as_namespace(), "titan_T2");
        assert_eq!(TitanId::T3.as_namespace(), "titan_T3");
    }

    #[test]
    fn titan_id_str() {
        assert_eq!(TitanId::T1.as_str(), "T1");
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
        assert_eq!(identity.titan_id, TitanId::T1);
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
        assert_eq!(identity.titan_id, TitanId::T3);
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
        let (tmp, _seed) = write_byte_array_keypair();
        let result = Identity::load_from_disk_with_hint(tmp.path(), Some("T9"));
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
        assert_eq!(identity.titan_id, TitanId::T2);
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
