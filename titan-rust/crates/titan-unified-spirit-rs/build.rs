//! Build script: capture git short SHA at compile time per SPEC §13
//! `--version` output discipline.
//!
//! Mirrors `titan-rust/crates/titan-kernel-rs/build.rs` to keep build-time
//! discipline consistent across all Rust binaries. Shells out to `git`
//! directly + degrades gracefully when git is missing (tarball builds).

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
