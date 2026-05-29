"""Regression guard for native-crash visibility (SPEC §11.I.4 error cascade).

The `@with_error_envelope` cascade only catches Python EXCEPTIONS. A fatal
native signal (SIGSEGV/SIGABRT/SIGBUS/SIGFPE — e.g. a crash in a torch/numpy
C extension) kills a worker WITHOUT an exception, so the cascade can't fire and
the death is otherwise silent (only the Supervisor's `shm_pid_dead`, no cause).
This was the blind spot that hid the cgn shm_pid_dead loop.

Fix: `faulthandler.enable()` in every Python process entry (workers via
`_module_wrapper`, plus the 3 peers) → Python dumps a C+Python traceback to
stderr→journal on a catchable fatal signal; and the orchestrator's `stop()`
logs the dead worker's exit code/signal (catches SIGKILL, which faulthandler
cannot). These guards pin both so native deaths can't go silent again.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _src(rel: str) -> str:
    return (_ROOT / rel).read_text()


def test_every_python_entrypoint_enables_faulthandler():
    """Workers (_module_wrapper) + all 3 peers must enable faulthandler."""
    entrypoints = {
        "titan_hcl/orchestrator/core.py": "_module_wrapper (workers, incl. cgn)",
        "scripts/titan_hcl.py": "orchestrator peer",
        "scripts/guardian_hcl.py": "supervisor peer",
        "titan_hcl/api/api_main.py": "api peer",
    }
    for rel, who in entrypoints.items():
        src = _src(rel)
        assert "faulthandler" in src and ".enable()" in src, (
            f"{rel} ({who}) must enable faulthandler — otherwise a native "
            f"crash (SIGSEGV/SIGABRT/SIGBUS) dies silently with no journal log")


def test_module_wrapper_enables_faulthandler_before_entry():
    """The worker wrapper must enable faulthandler inside the child process
    (so it covers the whole worker lifetime, before heavy imports)."""
    src = _src("titan_hcl/orchestrator/core.py")
    wi = src.index("def _module_wrapper")
    body = src[wi:wi + 2500]
    assert "_faulthandler.enable()" in body, (
        "_module_wrapper must enable faulthandler at the top of the child process")


def test_orchestrator_stop_logs_dead_worker_exit_code():
    """stop() must read + log the dead worker's exit code/signal (the SIGKILL
    path faulthandler can't catch) — was previously always logged as None."""
    src = _src("titan_hcl/orchestrator/core.py")
    si = src.index("def stop(")
    body = src[si:si + 3500]
    assert "info.process.exitcode" in body, (
        "stop() must read info.process.exitcode to surface the death cause")
    assert "DIED by signal" in body or "DIED —" in body, (
        "stop() must log the dead worker's exit code/signal at ERROR")


def test_systemd_template_sets_pythonfaulthandler():
    """Fresh installs must inherit PYTHONFAULTHANDLER=1 (belt-and-braces for
    any Python process incl. ones that crash before their entry code runs)."""
    assert "PYTHONFAULTHANDLER=1" in _src("scripts/setup_titan/systemd_runner.py")
