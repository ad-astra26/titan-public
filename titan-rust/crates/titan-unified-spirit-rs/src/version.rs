//! version — Build-time version metadata for `--version` output per SPEC §13.

use titan_core::constants::{SPEC_SOURCE_SHA256, SPEC_VERSION};

/// Cargo package version (from `[package] version`).
pub const CARGO_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git short SHA, populated by `build.rs`. Falls back to `"unknown"` for
/// non-git builds.
pub const GIT_SHA: &str = match option_env!("TITAN_GIT_SHA") {
    Some(s) => s,
    None => "unknown",
};

/// Multi-line `--version` output per SPEC §13.
pub fn full_version() -> String {
    format!(
        "{cargo} ({sha})\nSPEC v{spec} (sha-256: {spec_sha})",
        cargo = CARGO_VERSION,
        sha = GIT_SHA,
        spec = SPEC_VERSION,
        spec_sha = &SPEC_SOURCE_SHA256[..12],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_version_present() {
        assert!(!CARGO_VERSION.is_empty());
    }

    #[test]
    fn full_version_includes_spec() {
        let v = full_version();
        assert!(v.contains(CARGO_VERSION));
        assert!(v.contains("SPEC v"));
        assert!(v.contains(SPEC_VERSION));
    }
}
