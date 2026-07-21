//! orphan_reaper — reaps orphaned grandchildren adopted via `PR_SET_CHILD_SUBREAPER`.
//!
//! The kernel sets `PR_SET_CHILD_SUBREAPER` (SPEC §11.C(1), `main.rs`) so orphaned
//! descendants of its workers reparent to the kernel instead of pid-1/systemd.
//! tokio's per-worker `Child::wait()` reaps only the workers *we* spawned; an
//! adopted orphan — e.g. `offhost_mirror`'s `rsync -e ssh` grandchild, whose `ssh`
//! can outlive `rsync` — has no `Child` handle and would **zombie forever**, a slow
//! PID leak (measured ~60 zombie `ssh` over ~5 days on T1, 2026-07-21; ≈12/day, the
//! offhost-mirror pull cadence). A `PR_SET_CHILD_SUBREAPER` process that does not
//! reap its adopted orphans is broken by definition — this task closes that gap.
//!
//! **Race-free by construction** — it never steals a tokio-managed worker:
//!   1. it skips any pid in the supervisor's tracked set (`running_pids`); AND
//!   2. it only reaps a zombie that *persisted across two ticks* — a just-exited
//!      worker whose supervisor `wait()` future has not fired yet is reaped by tokio
//!      within milliseconds, far inside one tick, so we never race it.
//! It reaps ONLY `/proc` entries in state `Z` (zombie) with `ppid == our pid`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use nix::sys::wait::{waitpid, WaitPidFlag};
use nix::unistd::Pid;
use tokio::sync::Notify;
use tracing::info;

use crate::kernel_supervisor::KernelChildSupervisor;

/// Reaper poll cadence. Zombies hold only a PID slot (no memory/CPU), so a slow
/// cadence bounds accumulation at negligible cost. With the two-tick grace (see
/// module docs), an orphan is reaped within roughly `2 × REAP_INTERVAL_S`.
pub const REAP_INTERVAL_S: u64 = 15;

/// Scan `/proc` for zombie children of `my_pid` whose pid is NOT in `tracked`
/// (i.e. adopted orphans, not our tokio-managed workers). Reads `/proc` only —
/// it does **not** reap. Returns the set of orphan-zombie pids.
pub fn scan_orphan_zombies(my_pid: i32, tracked: &HashSet<i32>) -> HashSet<i32> {
    let mut out = HashSet::new();
    let dir = match std::fs::read_dir("/proc") {
        Ok(d) => d,
        Err(_) => return out,
    };
    for entry in dir.flatten() {
        let pid: i32 = match entry.file_name().to_str().and_then(|s| s.parse().ok()) {
            Some(p) => p,
            None => continue, // non-numeric /proc entry (e.g. "self", "cpuinfo")
        };
        if tracked.contains(&pid) {
            continue; // a tokio-managed worker — tokio reaps it, never us
        }
        // /proc/<pid>/stat = "pid (comm) state ppid ...". `comm` may contain spaces
        // AND ')', so split on the LAST ')' and read the fixed fields after it.
        let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(s) => s,
            Err(_) => continue, // raced exit / permission — skip
        };
        if let Some(rp) = stat.rfind(')') {
            let mut it = stat[rp + 1..].split_whitespace();
            let state = it.next();
            let ppid: Option<i32> = it.next().and_then(|s| s.parse().ok());
            if state == Some("Z") && ppid == Some(my_pid) {
                out.insert(pid);
            }
        }
    }
    out
}

/// The reaper loop. Spawned once from `kernel::run` after the supervisor exists;
/// runs until shutdown.
pub async fn run_orphan_reaper_loop(supervisor: Arc<KernelChildSupervisor>, shutdown: Arc<Notify>) {
    let my_pid = std::process::id() as i32;
    let mut prev: HashSet<i32> = HashSet::new();
    let mut interval = tokio::time::interval(Duration::from_secs(REAP_INTERVAL_S));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    info!(
        event = "ORPHAN_REAPER_STARTED",
        interval_s = REAP_INTERVAL_S,
        kernel_pid = my_pid
    );
    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                info!(event = "ORPHAN_REAPER_STOPPED");
                break;
            }
            _ = interval.tick() => {
                let tracked = supervisor.tracked_pids();
                let zombies = scan_orphan_zombies(my_pid, &tracked);
                let mut reaped = 0u32;
                for pid in &zombies {
                    // persistence guard: only reap a zombie also seen last tick, so a
                    // just-exited tracked worker (tokio reaps in ms) is never stolen.
                    if prev.contains(pid)
                        && waitpid(Pid::from_raw(*pid), Some(WaitPidFlag::WNOHANG)).is_ok()
                    {
                        reaped += 1;
                    }
                }
                if reaped > 0 {
                    info!(event = "ORPHAN_REAPED", count = reaped);
                }
                prev = zombies;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// FALSIFIER: an orphaned grandchild that zombies under us is FOUND by the scan,
    /// a `tracked` pid is EXCLUDED, and a `waitpid` reap removes it from `/proc`.
    #[test]
    fn reaps_orphan_zombie_but_never_a_tracked_pid() {
        // Become the reaper-of-last-resort for this test process (as the kernel does).
        titan_core::supervisor::prctl_unix::set_child_subreaper(true).expect("set subreaper");
        // `sh` backgrounds a grandchild `sleep` then exits → the sleep is orphaned →
        // reparents to us (the nearest subreaper) → once it exits it is a zombie here.
        let mut sh = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.3 & exit 0")
            .spawn()
            .expect("spawn sh");
        sh.wait().expect("reap sh (our direct child)");
        std::thread::sleep(Duration::from_millis(700)); // let the orphaned sleep exit → zombie
        let my_pid = std::process::id() as i32;

        // (1) the orphan zombie is present
        let z1 = scan_orphan_zombies(my_pid, &HashSet::new());
        assert!(
            !z1.is_empty(),
            "orphaned zombie grandchild must be found by the scan"
        );

        // (2) discrimination: a pid placed in `tracked` is EXCLUDED (never stolen)
        let one = *z1.iter().next().unwrap();
        let mut tracked = HashSet::new();
        tracked.insert(one);
        assert!(
            !scan_orphan_zombies(my_pid, &tracked).contains(&one),
            "a tracked pid must be excluded from the orphan scan"
        );

        // (3) reaping it removes it from /proc
        for pid in &z1 {
            let _ = waitpid(Pid::from_raw(*pid), Some(WaitPidFlag::WNOHANG));
        }
        assert!(
            scan_orphan_zombies(my_pid, &HashSet::new()).is_disjoint(&z1),
            "the orphan zombie must be reaped (gone from /proc after waitpid)"
        );
    }
}
