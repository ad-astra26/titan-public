//! Build script: capture git short SHA + build timestamp at compile time
//! per SPEC §13 `--version` output discipline.
//!
//! We don't depend on the `vergen` crate (different API surface across
//! versions and would force pinning); instead we shell out to `git` directly
//! and degrade gracefully when git is missing (e.g. tarball builds).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../../.git/HEAD");

    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=TITAN_GIT_SHA={sha}");
}
