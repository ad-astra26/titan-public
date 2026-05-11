//! identity_load — Wraps `titan_core::Identity::load_from_disk` + maps
//! errors to kernel exit codes per SPEC §15.
//!
//! Per SPEC §11.H.4 + G16(8): identity is SACRED. Failure to load = halt
//! boot with exit 3 (`KernelExitCode::IdentityLoadFailure`). NEVER
//! auto-restore from `.bak` (Maker manual recovery required).

use std::path::Path;

use titan_core::identity::{Identity, IdentityError};

use crate::exit::KernelExitCode;

/// Load the kernel's identity from disk + map any error to a kernel exit
/// code per SPEC §15.
///
/// `titan_id_hint` is forwarded to `Identity::load_from_disk_with_hint` so
/// Solana CLI byte-array-format keypairs (which have no embedded
/// `titan_id`) can be loaded — the hint is the value passed via
/// `--titan-id %i` from the systemd unit. Per Phase C C-S7 Gap D
/// (2026-05-05).
pub fn load_identity(path: &Path, titan_id_hint: &str) -> Result<Identity, KernelExitCode> {
    Identity::load_from_disk_with_hint(path, Some(titan_id_hint)).map_err(|e| {
        tracing::error!(
            event = "IDENTITY_LOAD_FAILED",
            err = ?e,
            path = ?path,
            "identity load failed — halting boot per SPEC G16(8)"
        );
        match e {
            IdentityError::FileAccess { .. }
            | IdentityError::BadPermissions { .. }
            | IdentityError::JsonParse(_)
            | IdentityError::HexDecode { .. }
            | IdentityError::BadSecretSeedLength { .. }
            | IdentityError::BadPublicKeyLength { .. }
            | IdentityError::PublicKeyMismatch
            | IdentityError::UnknownTitanId(_)
            | IdentityError::Corrupted => KernelExitCode::IdentityLoadFailure,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn missing_file_maps_to_identity_load_failure() {
        let nonexistent = Path::new("/tmp/this/path/does/not/exist/keypair.json");
        let result = load_identity(nonexistent, "T1");
        assert!(matches!(result, Err(KernelExitCode::IdentityLoadFailure)));
    }

    #[test]
    fn bad_permissions_maps_to_identity_load_failure() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
        let result = load_identity(tmp.path(), "T1");
        assert!(matches!(result, Err(KernelExitCode::IdentityLoadFailure)));
    }

    #[test]
    fn malformed_json_maps_to_identity_load_failure() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(b"not json").unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
        let result = load_identity(tmp.path(), "T1");
        assert!(matches!(result, Err(KernelExitCode::IdentityLoadFailure)));
    }

    #[test]
    fn valid_keypair_loads_successfully() {
        use rand::RngCore;
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

        let identity = load_identity(tmp.path(), "T1").unwrap();
        assert_eq!(identity.titan_id.as_str(), "T1");
    }

    /// Phase C C-S7 Gap D — load_identity accepts Solana CLI byte-array
    /// keypairs when titan_id is provided via the hint argument.
    #[test]
    fn byte_array_format_loads_via_titan_id_hint() {
        use rand::RngCore;
        let mut secret_seed = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret_seed);
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret_seed);
        let pk = signing.verifying_key().to_bytes();

        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&secret_seed);
        bytes.extend_from_slice(&pk);
        let json = serde_json::to_string(&bytes).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(json.as_bytes()).unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600)).unwrap();

        let identity = load_identity(tmp.path(), "T3").unwrap();
        assert_eq!(identity.titan_id.as_str(), "T3");
    }
}
