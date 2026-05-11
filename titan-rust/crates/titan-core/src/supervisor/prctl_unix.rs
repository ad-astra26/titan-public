//! prctl_unix — Linux `prctl` syscall wrappers per SPEC §11.C.
//!
//! Per SPEC §11.C(1):
//! - `PR_SET_PDEATHSIG` — every Rust binary calls this at startup so kernel
//!   children automatically die when their parent dies. Replaces the
//!   monitor-loop pattern Python had to use.
//! - `PR_SET_CHILD_SUBREAPER` — `titan-kernel-rs` ONLY calls this so any
//!   orphaned Python L2/L3 module gets reparented to the kernel rather
//!   than systemd/init. Cleaner than systemd-as-reaper.
//!
//! Both syscalls are no-ops on non-Linux (the binaries are Linux-only per
//! `feedback_t2t3_deployment_via_git_pull.md`; the wrappers degrade
//! gracefully so unit tests can run on macOS/etc. dev machines).

use nix::sys::signal::Signal;

/// Errors during prctl syscalls.
#[derive(Debug, thiserror::Error)]
pub enum PrctlError {
    /// Underlying syscall failure.
    #[error("prctl({op}) failed: {source}")]
    Syscall {
        /// Operation name (`"PR_SET_PDEATHSIG"`, etc.).
        op: &'static str,
        /// nix's errno-wrapped error.
        #[source]
        source: nix::Error,
    },
}

/// Set the parent-death signal for the current process.
///
/// When the parent dies (any way — clean exit, crash, SIGKILL), the kernel
/// will deliver `signal` to this process. By convention, we send `SIGTERM`
/// so the child can shut down gracefully.
///
/// **Important:** the kernel sends the signal exactly once; if the child
/// changes its parent (e.g. via `prctl(PR_SET_CHILD_SUBREAPER)` somewhere
/// in the tree), the signal also fires. Per SPEC §11.C(1) every Rust
/// binary calls this at startup.
#[cfg(target_os = "linux")]
pub fn set_pdeathsig(signal: Signal) -> Result<(), PrctlError> {
    use nix::sys::prctl;
    prctl::set_pdeathsig(signal).map_err(|source| PrctlError::Syscall {
        op: "PR_SET_PDEATHSIG",
        source,
    })
}

/// On non-Linux platforms, this is a no-op (returns `Ok(())` immediately).
/// Lets unit tests run on macOS dev machines without `cfg(target_os)`
/// gates everywhere.
#[cfg(not(target_os = "linux"))]
pub fn set_pdeathsig(_signal: Signal) -> Result<(), PrctlError> {
    Ok(())
}

/// Set this process as a child subreaper.
///
/// When set, descendant processes that lose their parent are reparented to
/// THIS process instead of init/systemd. The kernel binary calls this so
/// orphaned Python L2/L3 modules end up reaped by the kernel-rs supervisor
/// rather than systemd. Per SPEC §11.C(1).
///
/// Only `titan-kernel-rs` should call this. Calling it from a non-kernel
/// Rust binary is harmless but pointless.
#[cfg(target_os = "linux")]
pub fn set_child_subreaper(enabled: bool) -> Result<(), PrctlError> {
    // nix doesn't expose set_child_subreaper directly — call libc::prctl.
    // PR_SET_CHILD_SUBREAPER = 36 per linux/prctl.h
    const PR_SET_CHILD_SUBREAPER: libc::c_int = 36;
    let arg2 = if enabled { 1 } else { 0 };
    // SAFETY: prctl with PR_SET_CHILD_SUBREAPER is safe to call from any
    // thread; the kernel atomically sets the flag.
    let result = unsafe {
        libc::prctl(
            PR_SET_CHILD_SUBREAPER,
            arg2 as libc::c_ulong,
            0_u64,
            0_u64,
            0_u64,
        )
    };
    if result == -1 {
        return Err(PrctlError::Syscall {
            op: "PR_SET_CHILD_SUBREAPER",
            source: nix::Error::last(),
        });
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn set_child_subreaper(_enabled: bool) -> Result<(), PrctlError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_pdeathsig_sigterm_on_current_process() {
        // The test process itself calls set_pdeathsig — when this test
        // process exits, no one cares (test parent is cargo).
        let result = set_pdeathsig(Signal::SIGTERM);
        // On Linux: should succeed (we have permission to set our own
        // pdeathsig). On non-Linux: no-op returns Ok.
        assert!(result.is_ok(), "set_pdeathsig failed: {result:?}");
    }

    #[test]
    fn set_pdeathsig_sigkill_works() {
        // Some tools use SIGKILL as pdeathsig (more aggressive). Ensure
        // the wrapper accepts any Signal value.
        let result = set_pdeathsig(Signal::SIGKILL);
        assert!(result.is_ok());
    }

    // PR_SET_CHILD_SUBREAPER requires no special perms and is idempotent;
    // we can call it freely.
    #[test]
    fn set_child_subreaper_enable_then_disable() {
        let r1 = set_child_subreaper(true);
        assert!(r1.is_ok(), "enable failed: {r1:?}");
        let r2 = set_child_subreaper(false);
        assert!(r2.is_ok(), "disable failed: {r2:?}");
    }
}
