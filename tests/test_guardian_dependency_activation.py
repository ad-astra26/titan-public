"""Unit tests for SPEC §11.G.2.5 — dependency-driven activation (D-SPEC-90, v1.29.0).

Covers `Guardian._activate_dependencies` without spawning real subprocesses:
- No-op when module has no deps
- Action=PROBE is skipped (only ENSURE_RUNNING activates)
- Severity=SOFT is skipped (only CRITICAL activates)
- Kind != MODULE is skipped (BINARY/SHM_SLOT/etc. don't go through this path)
- Dep RUNNING → no-op + no event emitted
- Dep DISABLED → WARNING + no activation
- Dep unregistered → SUPERVISION_DEPENDENCY_BLOCKED emitted; dependent proceeds
- Dep STOPPED + reachable → SUPERVISION_DEPENDENCY_ACTIVATING emitted; start() called
- Dep READY observed before timeout → activation succeeds
- Dep never READY within timeout → WARNING; dependent proceeds anyway

Integration with meditation_worker is covered live (T3 cascade per
RFP_phase_c_enhancements.md §1.7 OBS gates).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from titan_hcl import bus
from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import (
    Guardian,
    ModuleSpec,
    ModuleState,
)
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
)


def _spec(
    name: str,
    *,
    dependencies: list[Dependency] | None = None,
) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        layer="L3",
        entry_fn=lambda *a, **kw: None,
        autostart=False,
        restart_on_crash=False,
        dependencies=dependencies or [],
    )


def _make_guardian():
    return Guardian(DivineBus())


def _capture_bus_events(g: Guardian, types: tuple[str, ...]) -> list[dict]:
    """Subscribe to bus events of the given types and return the list of
    payloads captured. The list mutates as events arrive."""
    captured: list[dict] = []
    original_publish = g.bus.publish

    def spy_publish(msg, *args, **kwargs):
        if isinstance(msg, dict) and msg.get("type") in types:
            captured.append(msg)
        return original_publish(msg, *args, **kwargs)

    g.bus.publish = spy_publish  # type: ignore[method-assign]
    return captured


# ── No-op paths ──────────────────────────────────────────────────────────


def test_activate_dependencies_no_deps_is_noop():
    g = _make_guardian()
    g.register(_spec("dependent"))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
        bus.SUPERVISION_DEPENDENCY_BLOCKED,
    ))
    g._activate_dependencies("dependent")
    assert captured == []
    g.stop_all()


def test_activate_dependencies_unregistered_module_is_noop():
    g = _make_guardian()
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
        bus.SUPERVISION_DEPENDENCY_BLOCKED,
    ))
    g._activate_dependencies("never_registered")
    assert captured == []
    g.stop_all()


def test_probe_action_is_skipped():
    """PROBE deps stay on the §11.G.2 respawn-only path; no activation."""
    g = _make_guardian()
    g.register(_spec("dep_target"))
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.PROBE,  # ← legacy behavior
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
    ))
    # Monkeypatch start so we'd notice if it was called
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert captured == []
    g.start.assert_not_called()
    g.stop_all()


def test_soft_severity_is_skipped():
    """SOFT deps don't trigger ENSURE_RUNNING — only CRITICAL does."""
    g = _make_guardian()
    g.register(_spec("dep_target"))
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.SOFT,  # ← soft, not critical
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
    ))
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert captured == []
    g.start.assert_not_called()
    g.stop_all()


def test_non_module_kind_is_skipped():
    """ENSURE_RUNNING only applies to MODULE kind; SHM_SLOT/BINARY/etc. ignored."""
    g = _make_guardian()
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="some_shm_slot",
            kind=DependencyKind.SHM_SLOT,  # ← not MODULE
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
        bus.SUPERVISION_DEPENDENCY_BLOCKED,
    ))
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert captured == []
    g.start.assert_not_called()
    g.stop_all()


def test_already_running_dep_is_noop():
    """If dep is already RUNNING, no event + no start call."""
    g = _make_guardian()
    g.register(_spec("dep_target"))
    # Synthesize RUNNING state
    dep_info = g._modules["dep_target"]
    dep_info.state = ModuleState.RUNNING
    dep_info.pid = 11111
    dep_info.ready_time = time.time()

    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
            check=lambda: g.is_running("dep_target"),
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
    ))
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert captured == []
    g.start.assert_not_called()
    g.stop_all()


def test_disabled_dep_skipped_with_warning():
    """DISABLED deps are skipped (not started, not blocked) — Maker can re-enable manually."""
    g = _make_guardian()
    g.register(_spec("dep_target"))
    g._modules["dep_target"].state = ModuleState.DISABLED

    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))
    captured_activating = _capture_bus_events(
        g, (bus.SUPERVISION_DEPENDENCY_ACTIVATING,))
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert captured_activating == []
    g.start.assert_not_called()
    g.stop_all()


# ── Active paths ─────────────────────────────────────────────────────────


def test_unregistered_dep_emits_blocked_event_and_continues():
    """If a dep is declared but never registered, BLOCKED event fires; dependent
    proceeds (start() is not called for the missing dep)."""
    g = _make_guardian()
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="never_registered_dep",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_BLOCKED,
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
    ))
    g.start = MagicMock(name="start_should_not_be_called")  # type: ignore[method-assign]
    g._activate_dependencies("dependent")
    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == bus.SUPERVISION_DEPENDENCY_BLOCKED
    assert msg["payload"]["child_name"] == "dependent"
    assert msg["payload"]["blocked_dependency"] == "never_registered_dep"
    assert msg["payload"]["reason"] == "unregistered_dep"
    g.start.assert_not_called()
    g.stop_all()


def test_stopped_dep_triggers_activation_and_start():
    """The happy path: dep is registered + STOPPED → ACTIVATING event +
    start(dep) called + (simulated) transition to RUNNING completes within
    the activation timeout."""
    g = _make_guardian()
    g.register(_spec("dep_target"))
    # state defaults to STOPPED via Guardian.register
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))
    captured = _capture_bus_events(g, (
        bus.SUPERVISION_DEPENDENCY_ACTIVATING,
    ))

    # Simulate the recursive start succeeding immediately by mutating the
    # registered dep's state inside the mocked start().
    def fake_start(name: str) -> bool:
        assert name == "dep_target", \
            f"Activation must call start() for the dep, got '{name}'"
        info = g._modules[name]
        info.state = ModuleState.RUNNING
        info.pid = 22222
        info.ready_time = time.time()
        return True

    g.start = fake_start  # type: ignore[method-assign]
    g._activate_dependencies("dependent")

    # ACTIVATING event emitted
    assert len(captured) == 1
    msg = captured[0]
    assert msg["payload"]["child_name"] == "dependent"
    assert msg["payload"]["dependency_name"] == "dep_target"
    assert msg["payload"]["dependency_kind"] == DependencyKind.MODULE.value
    assert msg["payload"]["severity"] == DependencySeverity.CRITICAL.value

    # Dep transitioned to RUNNING (via fake_start)
    assert g._modules["dep_target"].state == ModuleState.RUNNING
    g.stop_all()


def test_dep_never_reaches_ready_logs_warning_and_continues(caplog):
    """If dep stays STOPPED past SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S,
    Guardian logs a WARNING and lets the dependent's start proceed anyway —
    §11.G.2 respawn check catches a truly-down dep on the next cycle.

    The wait is normally 30s; this test shortens it to 1s via monkeypatching
    to keep test runtime sane.
    """
    # Post-Phase-11 carve (D-SPEC-141 / v1.65.0 / §11.I.1): the dep-activation
    # constant is read from the OrchestratorDepActivationMixin's own module
    # namespace, not from the package init. Patch at the leaf module — which
    # moved from `titan_hcl.guardian_hcl.dep_activation` to
    # `titan_hcl.orchestrator.dep_activation` per the orchestrator/supervisor
    # role split.
    import titan_hcl.orchestrator.dep_activation as guardian_mod

    g = _make_guardian()
    g.register(_spec("dep_target"))  # stays STOPPED throughout
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))

    # No-op start so the dep stays STOPPED.
    g.start = MagicMock(return_value=True, name="fake_start_noop")  # type: ignore[method-assign]

    original_timeout = guardian_mod.SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S
    try:
        guardian_mod.SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S = 1.0
        with caplog.at_level("WARNING", logger="titan_hcl.orchestrator"):
            t0 = time.time()
            g._activate_dependencies("dependent")
            elapsed = time.time() - t0
        # Timeout should fire ~1s + the 0.2s poll slack
        assert 0.9 < elapsed < 2.5, (
            f"Expected ~1s timeout, got {elapsed:.2f}s — poll/deadline math off?")
        assert any(
            "did not reach READY in" in r.message and "dep_target" in r.message
            for r in caplog.records
        ), f"Expected timeout WARNING, got: {[r.message for r in caplog.records]}"
    finally:
        guardian_mod.SUPERVISION_DEPENDENCY_ACTIVATION_TIMEOUT_S = original_timeout
    g.stop_all()


# ── Hook integration: Guardian.start() calls _activate_dependencies first ──


def test_start_invokes_activate_dependencies_before_spawn():
    """SPEC §11.G.2.5 contract — start(name) must invoke _activate_dependencies(name)
    before the lock-protected spawn body runs. Verified by observing that the
    dep transitions to RUNNING (via the spy's side effect inside
    _activate_dependencies) BEFORE start() touches the dependent's _modules entry
    state machine (visible via _cleanup_module which runs inside the lock).
    """
    g = _make_guardian()
    g.register(_spec("dep_target"))
    g.register(_spec("dependent", dependencies=[
        Dependency(
            name="dep_target",
            kind=DependencyKind.MODULE,
            severity=DependencySeverity.CRITICAL,
            action=DependencyAction.ENSURE_RUNNING,
        ),
    ]))

    call_order: list[str] = []

    # Spy on _activate_dependencies — record the call and simulate dep-up so the
    # subsequent spawn body doesn't sit waiting on a never-ready dep.
    original_activate = g._activate_dependencies

    def spy_activate(name: str) -> None:
        call_order.append(f"activate:{name}")
        if name == "dependent":
            dep_info = g._modules["dep_target"]
            dep_info.state = ModuleState.RUNNING
            dep_info.pid = 33333
            dep_info.ready_time = time.time()
        return original_activate(name)

    g._activate_dependencies = spy_activate  # type: ignore[method-assign]

    # Spy on _cleanup_module — it's invoked early inside the _module_lock body
    # in start() when the dependent's state needs reset. By recording its
    # invocation, we observe that the spawn body started.
    original_cleanup = g._cleanup_module

    def spy_cleanup(name: str) -> None:
        call_order.append(f"cleanup:{name}")
        return original_cleanup(name)

    g._cleanup_module = spy_cleanup  # type: ignore[method-assign]

    # Calling start() will fail at the actual spawn step (no real entry_fn),
    # but only the ordering matters here.
    try:
        g.start("dependent")
    except Exception:  # noqa: BLE001
        pass

    # _activate_dependencies must have run; whether cleanup fired depends on
    # the dependent's prior state, but if it did, it must follow activate.
    assert "activate:dependent" in call_order, (
        f"Expected activate:dependent in call_order; got: {call_order}")
    if any(x.startswith("cleanup:") for x in call_order):
        activate_idx = call_order.index("activate:dependent")
        cleanup_idx = next(
            i for i, x in enumerate(call_order) if x.startswith("cleanup:"))
        assert activate_idx < cleanup_idx, (
            f"Expected activate before cleanup; got: {call_order}")
    g.stop_all()
