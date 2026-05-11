//! version — Cargo + git SHA injected at compile time. Mirrors
//! `titan-kernel-rs/src/version.rs`.

/// Cargo package version (`titan-rust/Cargo.toml` workspace.package.version).
pub const CARGO_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Short git SHA captured by `build.rs`. Falls back to `"unknown"` for
/// tarball builds without `.git` available.
pub const GIT_SHA: &str = env!("TITAN_GIT_SHA");

/// Format the version string per SPEC §13: `<cargo_version> (<git_sha>)`.
pub fn full_version() -> String {
    format!("{CARGO_VERSION} ({GIT_SHA})")
}
