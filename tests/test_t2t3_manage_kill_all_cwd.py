"""C2-11 / BUG-DUPLICATE-KERNELS-FRAGMENT-BUS-20260428.

Verifies t{2,3}_manage.sh `stop` kills ALL titan_main process groups
whose `cwd` matches the manage script's `TITAN_DIR`, not just the
PIDFILE PID. Closes the race where services_watchdog spawned a fresh
titan_main between PIDFILE write and stop call, leaving two parents
fighting for the bus socket.

Per PLAN_microkernel_phase_c_s2_kernel.md §17.3.

Tests run a sandbox version of the stop algorithm (the relevant
function lifted into a portable shell snippet) against fake
titan_main processes (sleep loops with cd into a sandbox dir).
The actual scripts wire this same algorithm with the right TITAN_DIR.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from textwrap import dedent

import pytest


# Pure-bash function under test — same algorithm as t{2,3}_manage.sh stop.
# Single argument: TITAN_DIR. Reads PIDs from PIDFILE env if set.
_STOP_ALGORITHM = r"""
TITAN_DIR="$1"
PIDFILE="${PIDFILE:-/dev/null}"

TITAN_PIDS=""
if [ -f "$PIDFILE" ]; then
    PFPID=$(cat "$PIDFILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PFPID" ] && kill -0 "$PFPID" 2>/dev/null; then
        TITAN_PIDS="$PFPID"
    fi
    rm -f "$PIDFILE"
fi
for p in $(pgrep -f "titan_main.*--server" 2>/dev/null); do
    PCWD=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
    if [ "$PCWD" = "$TITAN_DIR" ]; then
        TITAN_PIDS="$TITAN_PIDS $p"
    fi
done
TITAN_PIDS=$(echo "$TITAN_PIDS" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

if [ -n "$TITAN_PIDS" ]; then
    TITAN_PGIDS=""
    for p in $TITAN_PIDS; do
        PGID=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        if [ -n "$PGID" ] && [ "$PGID" != "0" ] && [ "$PGID" != "1" ]; then
            TITAN_PGIDS="$TITAN_PGIDS $PGID"
        fi
    done
    TITAN_PGIDS=$(echo "$TITAN_PGIDS" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

    for pgid in $TITAN_PGIDS; do
        kill -- -"$pgid" 2>/dev/null
    done
    for _ in 1 2 3 4 5 6; do
        ALIVE=0
        for p in $TITAN_PIDS; do
            if [ -e "/proc/$p" ]; then ALIVE=$((ALIVE+1)); fi
        done
        [ "$ALIVE" -eq 0 ] && break
        sleep 1
    done
    for pgid in $TITAN_PGIDS; do
        kill -9 -- -"$pgid" 2>/dev/null
    done
    for _ in 1 2 3; do
        ALIVE=0
        for p in $TITAN_PIDS; do
            if [ -e "/proc/$p" ]; then ALIVE=$((ALIVE+1)); fi
        done
        [ "$ALIVE" -eq 0 ] && break
        sleep 1
    done
fi

# Final state — print remaining PIDs (empty = success)
echo "REMAINING:"
for p in $TITAN_PIDS; do
    if [ -e "/proc/$p" ]; then echo "$p"; fi
done
"""


# Track Popen handles so we can reap zombies before checking liveness.
# When the test runs `kill -9` (via the algorithm), the spawned bash/sleep
# becomes a zombie — /proc/<pid> entry persists until the parent (Python
# in this test) calls wait(). The actual t2/t3_manage.sh has init as the
# eventual parent (nohup detaches), so init reaps zombies automatically.
# To match the production observation in this test harness, we
# explicitly poll Popen handles to reap the zombies.
_POPENS: dict[int, subprocess.Popen] = {}


def _spawn_fake_titan_main(cwd: Path) -> int:
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = "exec -a 'python -u scripts/titan_main.py --server' sleep 60"
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=str(cwd),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _POPENS[proc.pid] = proc
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            cmdline = Path(f"/proc/{proc.pid}/cmdline").read_text()
            if "titan_main" in cmdline:
                break
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.05)
    return proc.pid


def _reap(pid: int) -> None:
    """Reap a tracked Popen child if it has exited (non-blocking)."""
    p = _POPENS.get(pid)
    if p is not None:
        p.poll()


def _is_alive(pid: int) -> bool:
    """True iff process exists AND is not a zombie."""
    _reap(pid)
    proc_status = Path(f"/proc/{pid}/status")
    if not proc_status.exists():
        return False
    try:
        for line in proc_status.read_text().splitlines():
            if line.startswith("State:"):
                # "State:\tZ (zombie)" or "State:\tR (running)" etc.
                state_char = line.split()[1] if len(line.split()) > 1 else ""
                return state_char != "Z"
    except (FileNotFoundError, OSError):
        return False
    return True


def _wait_for_dead(pids: list[int], timeout: float = 8.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for p in pids:
            _reap(p)
        alive = [p for p in pids if _is_alive(p)]
        if not alive:
            return []
        time.sleep(0.1)
    return [p for p in pids if _is_alive(p)]


def _run_stop(titan_dir: Path, pidfile: Path | None = None) -> str:
    env = os.environ.copy()
    if pidfile is not None:
        env["PIDFILE"] = str(pidfile)
    res = subprocess.run(
        ["bash", "-c", _STOP_ALGORITHM, "_", str(titan_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return res.stdout + res.stderr


@pytest.fixture
def sandbox_titan_dirs(tmp_path):
    titan_dir = tmp_path / "titan"
    titan3_dir = tmp_path / "titan3"
    titan_dir.mkdir()
    titan3_dir.mkdir()
    yield titan_dir, titan3_dir
    # Cleanup any stragglers (paranoid; tests should kill their own)
    for cwd in (titan_dir, titan3_dir):
        for p in subprocess.run(
            ["pgrep", "-f", "titan_main.*--server"],
            capture_output=True, text=True,
        ).stdout.split():
            try:
                pcwd = Path(f"/proc/{p}/cwd").resolve(strict=False)
                if pcwd == cwd:
                    os.kill(int(p), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass


class TestStopKillsAllCwdMatchedTitanMain:
    def test_stop_kills_pidfile_pid(self, sandbox_titan_dirs, tmp_path):
        titan_dir, _ = sandbox_titan_dirs
        pid = _spawn_fake_titan_main(titan_dir)
        pidfile = tmp_path / "titan.pid"
        pidfile.write_text(str(pid))

        try:
            _run_stop(titan_dir, pidfile=pidfile)
            assert _wait_for_dead([pid]) == [], "PIDFILE PID should be killed"
            assert not pidfile.exists(), "PIDFILE should be removed by stop"
        finally:
            if _is_alive(pid):
                os.kill(pid, signal.SIGKILL)

    def test_stop_kills_orphan_titan_main_in_same_cwd(self, sandbox_titan_dirs):
        titan_dir, _ = sandbox_titan_dirs
        # Two orphans in titan_dir, NOT registered in PIDFILE
        orphan1 = _spawn_fake_titan_main(titan_dir)
        orphan2 = _spawn_fake_titan_main(titan_dir)

        try:
            _run_stop(titan_dir, pidfile=None)
            survivors = _wait_for_dead([orphan1, orphan2])
            assert survivors == [], (
                f"both orphans in {titan_dir} should be killed; "
                f"survivors={survivors}"
            )
        finally:
            for p in (orphan1, orphan2):
                if _is_alive(p):
                    os.kill(p, signal.SIGKILL)

    def test_stop_does_not_kill_t3_when_run_on_t2_via_ssh(
        self, sandbox_titan_dirs
    ):
        """Cwd scoping: stopping T2 doesn't touch T3 processes (which
        share the same VPS + would also match `pgrep -f titan_main`)."""
        titan_dir, titan3_dir = sandbox_titan_dirs
        t2_pid = _spawn_fake_titan_main(titan_dir)
        t3_pid = _spawn_fake_titan_main(titan3_dir)

        try:
            _run_stop(titan_dir, pidfile=None)
            # T2's process is dead
            assert _wait_for_dead([t2_pid], timeout=8) == []
            # T3's process is still alive
            assert _is_alive(t3_pid), "T3 process must NOT be killed by T2 stop"
        finally:
            for p in (t2_pid, t3_pid):
                if _is_alive(p):
                    os.kill(p, signal.SIGKILL)

    def test_stop_waits_for_all_pids_to_exit(self, sandbox_titan_dirs):
        """Verifies the stop returns AFTER all targeted PIDs have exited
        (i.e. the post-stop state is clean, no follow-up sleep needed)."""
        titan_dir, _ = sandbox_titan_dirs
        pids = [_spawn_fake_titan_main(titan_dir) for _ in range(3)]

        try:
            _run_stop(titan_dir, pidfile=None)
            # Immediately after stop returns, ALL must be dead (not eventually).
            survivors = [p for p in pids if _is_alive(p)]
            assert survivors == [], (
                f"stop returned with PIDs still alive: {survivors}"
            )
        finally:
            for p in pids:
                if _is_alive(p):
                    os.kill(p, signal.SIGKILL)


class TestManageScriptSourceContract:
    """Source-level checks on the actual scripts."""

    @pytest.mark.parametrize("script", ["t2_manage.sh", "t3_manage.sh"])
    def test_stop_uses_cwd_exact_match(self, script):
        path = Path(__file__).parent.parent / "scripts" / script
        src = path.read_text(encoding="utf-8")
        # Exact-match comparison [ "$PCWD" = "$TITAN_DIR" ] (NOT grep -q substring)
        assert '[ "$PCWD" = "$TITAN_DIR" ]' in src, (
            f"{script} must use exact cwd match, not substring grep"
        )
        # PGID kill of the entire group, not just the leader PID
        assert 'kill -- -"$pgid"' in src or 'kill -- -"${pgid}"' in src, (
            f"{script} stop must kill the process group"
        )
        # Wait-for-exit loop
        assert "/proc/$p" in src, (
            f"{script} stop must wait for each PID to exit"
        )
