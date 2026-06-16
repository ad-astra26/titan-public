"""RFP_supervision_lifecycle §7.F-tail — structured-error-taxonomy USE tests.

Covers the three F-tail gates (§8 GF1/GF2/GF3):

  GF1  DISABLE is critical-only — a resource/non-FATAL condition NEVER disables;
       repeated FATAL (recoverable) ⇒ disable at threshold; FATAL + UNRECOVERABLE
       error_code ⇒ disable on the FIRST occurrence.
  GF2  Greppable journal cascade — every MODULE_ERROR renders
       [ERR][module][code][severity]; a module-down emits MODULE_CRITICAL_DOWN.
  GF3  Error-envelope coverage — the previously-bare worker entries
       (synthesis_worker_main, soul_diary_worker_main) now carry the FATAL
       @with_error_envelope so an uncaught exception becomes a typed ModuleError.

Run isolated (TorchRL mmap bus-error rule):
    python -m pytest tests/test_supervision_lifecycle_phase_f_tail.py -v -p no:anchorpy
"""
from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from titan_hcl.bus import (
    DivineBus,
    MODULE_CRITICAL_DOWN,
    MODULE_ERROR,
    make_msg,
)
from titan_hcl.errors import (
    ModuleError,
    ModuleErrorCode,
    Severity,
    UNRECOVERABLE_CODES,
    is_unrecoverable,
)
from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleState
from titan_hcl.supervisor import Supervisor


# ── fixtures ─────────────────────────────────────────────────────────────────

def _spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name, layer="L2", entry_fn=lambda *a, **kw: None,
        autostart=False, restart_on_crash=True,
    )


def _guardian_with(name: str = "m"):
    """A registered, RUNNING-ish module + a real in-process bus."""
    bus = DivineBus()
    g = Guardian(bus)
    g.register(_spec(name))
    g._modules[name].state = ModuleState.RUNNING
    # stop() touches spawn/queue internals we don't exercise here — the gate's
    # contract is the state transition + the CRITICAL emit, so isolate it.
    g.stop = MagicMock(return_value=True)
    return bus, g


def _supervisor(bus, g, **cfg):
    return Supervisor(bus, g, config=cfg)


def _err_msg(module: str, code, severity: Severity, msg: str = "boom") -> dict:
    err = ModuleError(
        module_name=module, subsystem="entry",
        error_code=str(code.value if isinstance(code, ModuleErrorCode) else code),
        severity=severity, message=msg,
    )
    return make_msg(MODULE_ERROR, module, "all", err.as_wire_dict())


def _feed(sup: Supervisor, msg: dict) -> None:
    """Inject straight onto the supervisor's dedicated error queue + drain."""
    sup._module_error_queue.put(msg)
    sup._process_module_errors()


# ── GF1 — DISABLE is critical-only ─────────────────────────────────────────

def test_non_fatal_module_error_never_disables():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g)
    for sev in (Severity.INFO, Severity.WARN, Severity.ERROR):
        _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, sev))
    assert g._modules["m"].state == ModuleState.RUNNING, \
        "non-FATAL ModuleErrors must NEVER disable (only render journal)"
    g.stop.assert_not_called()


def test_repeated_recoverable_fatal_disables_at_threshold():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, fatal_module_error_disable_threshold=3)
    # LLM_TIMEOUT is recoverable (not in UNRECOVERABLE_CODES).
    assert not is_unrecoverable(ModuleErrorCode.LLM_TIMEOUT.value)
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    assert g._modules["m"].state == ModuleState.RUNNING, "2 < threshold(3) — not yet"
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    assert g._modules["m"].state == ModuleState.DISABLED, "3rd FATAL crosses threshold"
    assert g._modules["m"].disabled_at > 0.0, "auto-re-enable eta marker set (§7.C recoverable)"


def test_unrecoverable_fatal_disables_on_first():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, fatal_module_error_disable_threshold=3)
    assert is_unrecoverable(ModuleErrorCode.BOOT_TIMEOUT.value)
    _feed(sup, _err_msg("m", ModuleErrorCode.BOOT_TIMEOUT, Severity.FATAL))
    assert g._modules["m"].state == ModuleState.DISABLED, \
        "an UNRECOVERABLE FATAL code disables on the FIRST occurrence (restart is futile)"


def test_fatal_errors_outside_window_dont_accumulate():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, fatal_module_error_disable_threshold=3)
    g._restart_window_seconds = 0.05  # tiny window
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    time.sleep(0.06)  # both age out of the window
    _feed(sup, _err_msg("m", ModuleErrorCode.LLM_TIMEOUT, Severity.FATAL))
    assert g._modules["m"].state == ModuleState.RUNNING, \
        "FATALs older than the window must age out — not a crash-loop"


def test_disable_gate_kill_switch_off():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, taxonomy_fatal_disable_gate=False)
    for _ in range(10):
        _feed(sup, _err_msg("m", ModuleErrorCode.BOOT_TIMEOUT, Severity.FATAL))
    assert g._modules["m"].state == ModuleState.RUNNING, "gate=false must never disable"


# ── GF2 — greppable journal cascade ─────────────────────────────────────────

def test_every_module_error_renders_greppable_tag(caplog):
    bus, g = _guardian_with()
    sup = _supervisor(bus, g)
    with caplog.at_level(logging.ERROR):
        _feed(sup, _err_msg("agno_worker", ModuleErrorCode.LLM_TIMEOUT,
                            Severity.ERROR, "upstream 503"))
    assert any("[ERR][agno_worker][LLM_TIMEOUT][ERROR]" in r.getMessage()
               for r in caplog.records), \
        "every ModuleError must render the stable [ERR][module][code][severity] tag"


def test_module_critical_down_emitted_on_disable():
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, fatal_module_error_disable_threshold=1)
    crit_q = bus.subscribe("test_crit_observer", types=[MODULE_CRITICAL_DOWN])
    _feed(sup, _err_msg("m", ModuleErrorCode.BOOT_TIMEOUT, Severity.FATAL))
    got = []
    while True:
        try:
            got.append(crit_q.get_nowait())
        except Exception:
            break
    assert got, "disable must broadcast MODULE_CRITICAL_DOWN"
    payload = got[-1]["payload"]
    assert payload["module"] == "m"
    assert payload["error_code"] == "BOOT_TIMEOUT"
    assert payload["severity"] == "FATAL"


def test_journal_render_kill_switch_off(caplog):
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, module_error_journal_render=False,
                      taxonomy_fatal_disable_gate=False)
    # With both off, the queue isn't even subscribed → nothing to drain.
    assert sup._module_error_queue is None
    with caplog.at_level(logging.ERROR):
        sup._process_module_errors()
    assert not any("[ERR][" in r.getMessage() for r in caplog.records)


# ── GF3 — error-envelope coverage ───────────────────────────────────────────

def test_synthesis_and_soul_diary_entries_have_fatal_envelope():
    """The two spawned-worker entries that were bare now carry the FATAL
    @with_error_envelope so an uncaught exception becomes a typed ModuleError
    instead of a silent heartbeat-timeout (the old grep surface)."""
    from titan_hcl.modules.synthesis_worker import synthesis_worker_main
    from titan_hcl.modules.soul_diary_worker import soul_diary_worker_main
    for fn in (synthesis_worker_main, soul_diary_worker_main):
        assert hasattr(fn, "__wrapped__"), \
            f"{fn.__name__} must be wrapped by @with_error_envelope"


# ── taxonomy invariants ─────────────────────────────────────────────────────

def test_unrecoverable_set_is_conservative():
    # Transient/external faults must stay RECOVERABLE (restart may help).
    for code in (ModuleErrorCode.LLM_TIMEOUT, ModuleErrorCode.EXTERNAL_SVC_UNAVAILABLE,
                 ModuleErrorCode.BUS_PUBLISH_FAILED, ModuleErrorCode.DEPENDENCY_TIMEOUT):
        assert not is_unrecoverable(code.value), f"{code} must be recoverable"
    assert UNRECOVERABLE_CODES, "the unrecoverable set must not be empty"


def test_unknown_error_code_is_recoverable():
    assert is_unrecoverable("SOME_NEW_UNREGISTERED_CODE") is False


# ── integration: the REAL in-process bus path (not a hand-fed queue) ─────────

def test_real_bus_path_publish_to_disable_e2e():
    """Integration — the REAL path a worker actually uses:
    publish_module_error(bus, err) → DivineBus dst='all' broadcast → the
    Supervisor's OWN `guardian_module_errors` subscription (filter=MODULE_ERROR)
    → _process_module_errors drains it → render + DISABLE + MODULE_CRITICAL_DOWN
    traverses the bus to an independent observer. The unit tests above feed the
    queue directly; THIS proves the subscription + routing are wired correctly.
    (The only piece not covered in-process is the Rust-broker SOCKET forward
    declared in guardian_hcl.build_bus_and_client — that needs a live box.)"""
    from titan_hcl.bus import publish_module_error
    bus, g = _guardian_with()
    sup = _supervisor(bus, g, fatal_module_error_disable_threshold=1)
    assert sup._module_error_queue is not None, "Supervisor must hold a real MODULE_ERROR subscription"
    crit_q = bus.subscribe("test_obs_e2e", types=[MODULE_CRITICAL_DOWN])

    err = ModuleError(
        module_name="m", subsystem="entry",
        error_code=ModuleErrorCode.BOOT_TIMEOUT.value,  # unrecoverable → first-occurrence disable
        severity=Severity.FATAL, message="boot exceeded grace",
    )
    assert publish_module_error(bus, err) is True, "the real publish helper must send"

    sup._process_module_errors()  # drains from the supervisor's OWN subscription

    assert g._modules["m"].state == ModuleState.DISABLED, \
        "a FATAL unrecoverable error published over the real bus must disable the module"
    got = []
    while True:
        try:
            got.append(crit_q.get_nowait())
        except Exception:
            break
    assert any(m["payload"]["module"] == "m" and m["payload"]["error_code"] == "BOOT_TIMEOUT"
               for m in got), "MODULE_CRITICAL_DOWN must traverse the bus to an independent observer"
