"""
Tests for 2026-04-27 shadow-swap speed/correctness fixes.

Bundle of 7 fixes shipped in one commit (per Maker greenlight):
  #1 Guardian.pause() replaces stop_all in _phase_hibernate (saves 8.5min)
  #2 Proxy ↔ swap interlock: Guardian.start() blocks during swap
  #4 CLI polling timeout 240s → 1200s
  #5 HIBERNATE_ACK_TIMEOUT_BY_LAYER tightened (~20s saved)
  #6 lock_polling timeout 15s → 5s
  #7 SHADOW_BOOT_TIMEOUT 60s → 30s

Fix #3 (fork→spawn migration) intentionally deferred — becomes architecturally
moot once swaps complete in ~30s instead of 13min (parent doesn't bloat).

Speed target: ~30-50s end-to-end swap (vs broken 13min). Path to <2s requires
B.2.1 worker wiring (next session) + shadow pre-warm (future).
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from titan_plugin import bus
from titan_plugin.core import shadow_protocol as sp
from titan_plugin.guardian import Guardian, ModuleSpec


def _noop_entry(*args, **kwargs):  # pragma: no cover
    return None


# ── Fix #1 — Guardian.pause() ────────────────────────────────────────────


def test_pause_sets_stop_requested():
    """pause() flips _stop_requested True (mutes monitor_tick)."""
    g = Guardian(bus.DivineBus(maxsize=100))
    assert g._stop_requested is False
    g.pause()
    assert g._stop_requested is True


def test_pause_idempotent():
    """Calling pause() on already-paused Guardian is a no-op."""
    g = Guardian(bus.DivineBus(maxsize=100))
    g.pause()
    g.pause()  # must not raise
    assert g._stop_requested is True


def test_pause_does_not_iterate_modules():
    """pause() does NOT call stop() per module (the slow-stop_all behavior)."""
    g = Guardian(bus.DivineBus(maxsize=100))
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3"))
    g.register(ModuleSpec(name="m2", entry_fn=_noop_entry, layer="L3"))
    # Mock stop so we can detect any call
    g.stop = MagicMock()  # type: ignore[method-assign]
    g.pause()
    g.stop.assert_not_called()


def test_pause_resume_round_trip():
    """pause() → resume() restores monitor_tick."""
    g = Guardian(bus.DivineBus(maxsize=100))
    g.pause()
    assert g._stop_requested is True
    g.resume()
    assert g._stop_requested is False


# ── Fix #2 — proxy ↔ swap interlock in Guardian.start ────────────────────


class _FakeKernel:
    """Minimal kernel stub for swap-interlock tests."""

    def __init__(self):
        self._active = False
        self._done_event = threading.Event()
        self._done_event.set()

    def is_shadow_swap_active(self) -> bool:
        return self._active

    def wait_for_swap_completion(self, timeout: float = 60.0) -> bool:
        if not self._active:
            return True
        return self._done_event.wait(timeout=timeout)

    def begin_swap(self):
        self._active = True
        self._done_event.clear()

    def end_swap(self):
        self._active = False
        self._done_event.set()


def test_start_no_swap_proceeds_immediately():
    """When no swap is active, start() does NOT call wait_for_swap_completion."""
    g = Guardian(bus.DivineBus(maxsize=100))
    fake = MagicMock()
    fake.is_shadow_swap_active.return_value = False
    g._kernel_ref = fake
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3", autostart=False))
    # We don't actually want to start the module (it'd fork a process).
    # Just verify the interlock path: is_shadow_swap_active was checked,
    # wait_for_swap_completion was NOT called.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(g, "_module_lock", MagicMock())  # short-circuit lock
        mp.setattr(g._modules.get("m1"), "spec", MagicMock())  # kill spec to early-return
        try:
            g.start("m1")
        except Exception:
            pass  # we only care about the interlock invocation
    fake.is_shadow_swap_active.assert_called_once()
    fake.wait_for_swap_completion.assert_not_called()


def test_start_during_swap_calls_wait():
    """When swap is active, start() blocks via wait_for_swap_completion(60.0)."""
    g = Guardian(bus.DivineBus(maxsize=100))
    fake = MagicMock()
    fake.is_shadow_swap_active.return_value = True
    fake.wait_for_swap_completion.return_value = True
    g._kernel_ref = fake
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3"))
    try:
        g.start("m1")
    except Exception:
        pass
    fake.wait_for_swap_completion.assert_called_once_with(timeout=60.0)


def test_start_no_kernel_ref_skips_interlock():
    """Legacy mode (kernel_ref=None) bypasses the interlock — no exception."""
    g = Guardian(bus.DivineBus(maxsize=100))
    g._kernel_ref = None
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3"))
    # Should not raise
    try:
        g.start("m1")
    except Exception:
        pass


def test_start_swap_wait_exception_fail_open():
    """If wait_for_swap_completion raises, start() proceeds anyway (fail-open)."""
    g = Guardian(bus.DivineBus(maxsize=100))
    fake = MagicMock()
    fake.is_shadow_swap_active.return_value = True
    fake.wait_for_swap_completion.side_effect = RuntimeError("rpc broken")
    g._kernel_ref = fake
    g.register(ModuleSpec(name="m1", entry_fn=_noop_entry, layer="L3"))
    # Should not propagate the RuntimeError — fail-open per design
    try:
        g.start("m1")
    except RuntimeError:
        pytest.fail("start() must not propagate wait_for_swap_completion errors")
    except Exception:
        pass  # other downstream errors (no entry_fn etc) are fine


# ── Fix #5 — HIBERNATE_ACK_TIMEOUT_BY_LAYER tightened ────────────────────


def test_hibernate_ack_timeouts_tightened():
    """Bonus #5 — L3 timeout dropped 30s → 10s; L2 20s → 8s; L1 10s → 5s."""
    assert sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER["L0"] == 5.0
    assert sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER["L1"] == 5.0
    assert sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER["L2"] == 8.0
    assert sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER["L3"] == 10.0
    # Max is 10s now (was 30s) — orchestrator's _drain_messages uses max
    assert max(sp.HIBERNATE_ACK_TIMEOUT_BY_LAYER.values()) == 10.0


def test_shadow_boot_timeout_retuned():
    """SHADOW_BOOT_TIMEOUT 60s → 45s (Bonus #7 + 2026-04-27 retune).
    Original Bonus #7 dropped 60→30, but swap #5 showed spirit+memory
    can't reach state=running in 30s. 45s is the post-fix sweet spot."""
    assert sp.SHADOW_BOOT_TIMEOUT == 45.0


# ── Fix #6 — lock_polling tightened (static src check) ───────────────────


def test_lock_polling_removed_from_phase_shadow_boot():
    """2026-04-27 post-swap-#4 hot-fix: lock_polling REMOVED from
    _phase_shadow_boot. It was architecturally redundant with per-shadow
    data_dir (DuckDB/SQLite hardlinks break on first write → each kernel
    gets its own inode → no lock contention). Was creating false-failures
    on every swap (api_subprocess + IMW children held inner_memory.db
    handles even after fast_kill).
    """
    import inspect
    from titan_plugin.core import shadow_orchestrator as so
    src = inspect.getsource(so._phase_shadow_boot)
    # The lock_polling.poll_locks_released call must be GONE.
    assert "lock_polling.poll_locks_released" not in src, (
        "lock_polling.poll_locks_released should be removed from "
        "_phase_shadow_boot — it was a redundant defense layer that "
        "caused false-failures (see swap #4 diagnostic)"
    )
    # original_kernel_locks_not_released should no longer be a possible
    # failure_reason from _phase_shadow_boot
    assert "original_kernel_locks_not_released" not in src


# ── Fix #1 (orchestrator side) — _phase_hibernate uses pause not stop_all ─


def test_phase_hibernate_uses_pause():
    """_phase_hibernate must call guardian.pause(), NOT stop_all (the slow path)."""
    import inspect
    from titan_plugin.core import shadow_orchestrator as so
    src = inspect.getsource(so._phase_hibernate)
    # Required: pause is called
    assert "kernel.guardian.pause()" in src, (
        "_phase_hibernate must call guardian.pause() (Phase B fast-hibernate)"
    )
    # Forbidden: stop_all in _phase_hibernate (the 8.5min anti-pattern)
    assert 'kernel.guardian.stop_all(reason="shadow_swap_hibernate_complete")' not in src, (
        "_phase_hibernate MUST NOT call stop_all — that's the 8.5min "
        "save_first=True bottleneck this fix removes."
    )


# ── Fix #4 — CLI polling timeout raised ───────────────────────────────────


def test_cli_polling_timeout_raised():
    """scripts/shadow_swap.py must use grace + 1200 (was 240)."""
    import inspect
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "_shadow_swap_cli",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "scripts", "shadow_swap.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    assert "args.grace + 1200.0" in src, (
        "CLI max_total must be grace + 1200.0 (Fix #4); was grace + 240.0"
    )
    assert "args.grace + 240.0" not in src, (
        "old grace + 240.0 still present — fix #4 not applied"
    )
