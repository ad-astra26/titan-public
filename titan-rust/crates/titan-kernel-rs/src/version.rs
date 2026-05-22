//! version — Build-time version metadata for `--version` output per SPEC §13.
//!
//! Format:
//! ```text
//! titan-kernel-rs <cargo_version> (<git_sha_short>)
//! SPEC v0.1.x (titan-docs/specs/SPEC_titan_architecture.md)
//! ```

use titan_core::constants::{SPEC_SOURCE_SHA256, SPEC_VERSION};

/// Cargo package version (from `[package] version`).
pub const CARGO_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git short SHA, populated by `build.rs` via `git rev-parse --short HEAD`.
/// Falls back to `"unknown"` when build is not from a git working tree.
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
    fn full_version_includes_spec_version() {
        let s = full_version();
        assert!(s.contains("SPEC v"));
        assert!(s.contains(SPEC_VERSION));
    }

    #[test]
    fn full_version_includes_cargo_version() {
        let s = full_version();
        assert!(s.contains(CARGO_VERSION));
    }

    #[test]
    fn full_version_includes_spec_sha_prefix() {
        let s = full_version();
        // First 12 chars of SPEC SHA must appear
        assert!(s.contains(&SPEC_SOURCE_SHA256[..12]));
    }
}
