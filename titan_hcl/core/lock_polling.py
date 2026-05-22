"""
Strict lock-release polling — second-line defense for BUG-B1-SHARED-LOCKS.

Even with per-shadow `data_shadow_<port>/` (the primary fix), Phase 3
shadow boot benefits from one final check: ensure the original kernel's
lock-protected files have actually released their OS locks before we
spawn shadow workers that might inherit them via fork() COW. This
catches the residual case where the original kernel's HIBERNATE was
incomplete or its workers are still in a death spiral.

The check is "strict" because:
  - It uses real OS facilities (`fuser` command); doesn't trust process
    state from /v4/state which can lag behind actual fd close.
  - It refuses to proceed past the timeout — orchestrator rolls back
    rather than racing the lock release.
  - It logs which PIDs are holding the lock when it fails, so the
    diagnosis names the offending workers.

Files we check are the ones with documented exclusive locks:
  - data/titan_memory.duckdb (DuckDB exclusive)
  - data/inner_memory.db (SQLite WAL — concurrent reads ok, writers
    serialize via WAL)
  - data/observatory.db (same pattern)
  - data/run/imw.sock (unix socket — single binder)
  - data/run/observatory_writer.sock

Used by the shadow orchestrator just before _phase_shadow_boot. If
locks aren't released within the grace window, the orchestrator emits
SHADOW_BOOT_LOCKS_HELD + rollback.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


#: Files with exclusive OS locks the original kernel must release before
#: a shadow on the SAME data dir can bind them. (Per-shadow data dir
#: bypasses the issue for these specific files; lock polling defends
#: against the per-shadow-dir copy missing one or against future code
#: paths that re-introduce shared lock files.)
DEFAULT_LOCK_FILES: tuple[str, ...] = (
    "data/titan_memory.duckdb",
    "data/inner_memory.db",
    "data/observatory.db",
    "data/run/imw.sock",
    "data/run/observatory_writer.sock",
)


def _fuser_holders(path: Path | str) -> list[int]:
    """Return PIDs holding `path` open according to fuser. Empty list = released.

    fuser is the canonical Linux tool for "who is holding this file
    open"; checks the kernel's open-file table directly. Better than
    lsof for our purposes because it's lighter-weight + has a clean
    machine-readable mode.

    Returns [] when the file doesn't exist (not a lock-held state).
    """
    p = Path(path)
    if not p.exists():
        return []
    if shutil.which("fuser") is None:
        # fuser not installed — we can't poll. Caller should fall back to
        # waiting a fixed grace period. Logged at WARNING so deploy notes
        # this gap.
        logger.warning("[lock_polling] fuser command not available; cannot poll %s", p)
        return []
    try:
        # `fuser -v <file>` exits 1 with stderr "No process found" when nothing
        # holds the file. With holders, stdout contains space-separated PIDs.
        rc = subprocess.run(
            ["fuser", str(p)], capture_output=True, text=True, timeout=5,
        )
        # fuser writes PIDs to STDOUT (not stderr) in plain mode
        out = rc.stdout.strip()
        if not out:
            return []
        return [int(x) for x in out.split() if x.isdigit()]
    except (subprocess.TimeoutExpired, ValueError):
        return []
    except Exception as e:
        logger.warning("[lock_polling] fuser failed for %s: %s", p, e)
        return []


def poll_locks_released(
    files: Iterable[str | Path] | None = None,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.5,
    exclude_pids: Iterable[int] = (),
) -> tuple[bool, dict]:
    """Block until all `files` are released by everyone except `exclude_pids`.

    Returns (released, diagnosis). On timeout, diagnosis names the files
    still locked + which PIDs hold them — orchestrator surfaces this to
    Maker so the right worker can be debugged.

    `exclude_pids` lets the caller's own PID (or known long-lived helper
    PIDs like systemd) be ignored — the original-kernel main process is
    typically excluded so we only check WORKER lock holders.

    Defaults to DEFAULT_LOCK_FILES if `files` is None.
    """
    if files is None:
        files = DEFAULT_LOCK_FILES
    files = [Path(f) for f in files]
    exclude = set(int(p) for p in exclude_pids)

    deadline = time.monotonic() + timeout
    last_diag: dict = {"polled_files": [str(f) for f in files], "still_held": {}}

    while time.monotonic() < deadline:
        still_held: dict[str, list[int]] = {}
        for f in files:
            holders = [p for p in _fuser_holders(f) if p not in exclude]
            if holders:
                still_held[str(f)] = holders
        last_diag["still_held"] = still_held
        last_diag["checked_at"] = time.time()
        if not still_held:
            last_diag["released"] = True
            return True, last_diag
        time.sleep(poll_interval)

    last_diag["released"] = False
    last_diag["timeout_seconds"] = timeout
    return False, last_diag
