"""Fault-injection tests for synthesis boot-wiring self-healing.

Proves the DeferredWiringRegistry contract that closes the T2 regression
(2026-07-06): a sub-store whose boot wiring times out under a contended boot
(a `submit_sync` DDL on a saturated SynthesisWriter → `TimeoutError`) is
retried by the export daemon until the writer drains and it wires — the
subsystem comes ONLINE mid-session with NO restart. Verifies:
  1. a transient TimeoutError leaves the wiring pending, then wires on a later
     deferred pass, and runs its finalize hook exactly once;
  2. the attempt cap is bounded (a permanently-timing-out wiring gives up);
  3. the kill-switch (deferred_rewire_enabled=false) suppresses all retry;
  4. a genuine (non-timeout) wiring failure is permanent — dropped, no retry,
     no finalize.
"""
import logging

import pytest

from titan_hcl.modules.synthesis_worker import DeferredWiringRegistry


def _reg(enabled=True, max_attempts=15):
    log = logging.getLogger("test_deferred_rewire")
    return DeferredWiringRegistry(
        enabled_fn=lambda: enabled,
        max_attempts_fn=lambda: max_attempts,
        log=log,
    )


def test_transient_timeout_then_self_heals_and_finalizes_once():
    """Wiring times out twice, then succeeds on the 3rd (2nd deferred) pass."""
    calls = {"wire": 0, "finalize": 0}

    def wire():
        calls["wire"] += 1
        if calls["wire"] < 3:      # boot attempt + 1 deferred = timeouts
            raise TimeoutError()
        return True                # 3rd call wires

    def finalize():
        calls["finalize"] += 1

    reg = _reg()
    reg.register_finalize("phase6", finalize)

    # Boot attempt: times out (call #1) → pending, NO finalize at boot.
    assert reg.attempt("phase6", wire, deferred=False) is False
    assert reg.pending_names == {"phase6"}
    assert calls["finalize"] == 0

    # Deferred pass #1: times out (call #2) → still pending.
    reg.run_deferred()
    assert reg.pending_names == {"phase6"}
    assert calls["finalize"] == 0

    # Deferred pass #2: wires (call #3) → drops pending + runs finalize once.
    reg.run_deferred()
    assert reg.pending_names == set()
    assert calls["wire"] == 3
    assert calls["finalize"] == 1

    # Further passes are a no-op (nothing pending) — finalize never re-runs.
    reg.run_deferred()
    reg.run_deferred()
    assert calls["finalize"] == 1


def test_attempt_cap_gives_up_after_max_attempts():
    """A wiring that never stops timing out is dropped after the cap."""
    calls = {"wire": 0}

    def wire():
        calls["wire"] += 1
        raise TimeoutError()

    reg = _reg(max_attempts=3)
    assert reg.attempt("forks", wire, deferred=False) is False   # boot: 1 timeout
    assert reg.pending_names == {"forks"}

    # 3 deferred passes allowed (attempts 1..3), the 4th trips the cap → dropped.
    reg.run_deferred()   # attempt 1
    reg.run_deferred()   # attempt 2
    reg.run_deferred()   # attempt 3
    assert reg.pending_names == {"forks"}
    reg.run_deferred()   # attempt 4 > cap → give up, drop
    assert reg.pending_names == set()

    # Once given up, further passes never call wire again.
    n = calls["wire"]
    reg.run_deferred()
    assert calls["wire"] == n


def test_kill_switch_suppresses_deferred_retry():
    """deferred_rewire_enabled=false → pending stays, wire never re-runs."""
    calls = {"wire": 0, "finalize": 0}

    def wire():
        calls["wire"] += 1
        raise TimeoutError()

    reg = _reg(enabled=False)
    reg.register_finalize("actr", lambda: calls.__setitem__("finalize", 1))
    assert reg.attempt("actr", wire, deferred=False) is False
    assert calls["wire"] == 1

    reg.run_deferred()
    reg.run_deferred()
    assert calls["wire"] == 1                 # never retried
    assert reg.pending_names == {"actr"}      # still pending (heals if re-enabled)
    assert calls["finalize"] == 0


def test_permanent_failure_is_dropped_no_retry_no_finalize():
    """A non-timeout wiring failure returns None → permanent (missing dep)."""
    calls = {"wire": 0, "finalize": 0}

    def wire():
        calls["wire"] += 1
        return None               # permanent: e.g. missing kuzu_graph dep

    reg = _reg()
    reg.register_finalize("forks", lambda: calls.__setitem__("finalize", 1))
    assert reg.attempt("forks", wire, deferred=False) is False
    assert reg.pending_names == set()         # dropped, not retried

    reg.run_deferred()
    assert calls["wire"] == 1                  # never retried
    assert calls["finalize"] == 0             # finalize never runs on permanent fail


def test_deferred_success_runs_finalize_even_if_boot_finalize_absent():
    """If wiring only succeeds on a deferred pass, the deferred path runs the
    finalize hook (the boot inline finalize saw a None product and no-op'd)."""
    calls = {"wire": 0, "finalize": 0}

    def wire_timeout_then_ok():
        calls["wire"] += 1
        if calls["wire"] == 1:
            raise TimeoutError()
        return True

    reg = _reg()
    reg.register_finalize("phase6", lambda: calls.__setitem__("finalize",
                                                              calls["finalize"] + 1))
    assert reg.attempt("phase6", wire_timeout_then_ok, deferred=False) is False
    assert calls["finalize"] == 0             # boot path never finalizes
    reg.run_deferred()                         # deferred wires → finalize runs
    assert reg.pending_names == set()
    assert calls["finalize"] == 1


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:anchorpy"]))
