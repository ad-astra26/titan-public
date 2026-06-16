"""RFP_supervision_lifecycle §7.D — graceful orchestrated stop.

The kernel's shutdown() drains every module (save_first) then emits a definitive
KERNEL_SHUTDOWN_COMPLETE marker (journal line + marker file) carrying the count
of modules stopped, so a restart can confirm a CLEAN, COMPLETE drain (state
saved, not SIGKILLed mid-save) and gate the next start on memory reclaim.

This proves the feeder: stop_all() returns the stopped count (was None).
The marker emit + the manage-script stop→verify→start gate are verified at
deploy (they need a real kernel process + systemd).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleState


def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L2", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


def test_stop_all_returns_stopped_count():
    """stop_all() returns the number of live modules it stopped (fed into the
    KERNEL_SHUTDOWN_COMPLETE marker), counting only RUNNING/STARTING/UNHEALTHY."""
    g = Guardian(DivineBus())
    for n in ("a", "b", "c"):
        g.register(_spec(n))
        g._modules[n].state = ModuleState.RUNNING
    g.register(_spec("idle"))
    g._modules["idle"].state = ModuleState.STOPPED  # not live → not counted
    g.stop = MagicMock(return_value=None)

    stopped = g.stop_all(reason="unit_test")

    assert stopped == 3, f"expected 3 live modules stopped, got {stopped}"
    assert g.stop.call_count == 3
