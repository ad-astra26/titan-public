"""
Tests for Phase B.2 IPC §D7+§D12 — Guardian skips in-process subscriber
registration when the bus has a socket broker attached.

Background (2026-04-30 incident):
With bus_ipc_socket_enabled=true + spawn_graduated_workers=true both on,
workers connect to the broker via setup_worker_bus and read from
SocketQueue (not info.queue). Guardian._start_module was unconditionally
appending info.queue to bus._subscribers[name], creating an orphan
in-process subscriber. The orphan accumulated every targeted dst=<name>
message + every un-filtered dst="all" broadcast with no consumer →
saturated at maxsize=10000 → 686k+ "Queue full" drops in <30 min.

Fix: gate the registration on `not bus.has_socket_broker`. Phase B.3
deletes both the gate and the gated registration when the legacy
mp.Queue path retires entirely.

These tests pin the contract:
- Socket-broker mode: worker name added to bus._modules but NOT to
  bus._subscribers. info.queue still allocated (legacy fallback safety
  net + still used by Guardian.broadcast_to_modules / call_module).
- Legacy mp.Queue mode: registration unchanged, worker queue subscribed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from titan_hcl.bus import DivineBus
from titan_hcl.guardian_hcl import Guardian, ModuleSpec


def _entry_fn(*args, **kwargs):
    """Spec entry_fn — process never reaches it because we mock spawn."""
    return None


def _make_spec(name: str) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        entry_fn=_entry_fn,
        layer="L1",
        autostart=False,
        rss_limit_mb=500,
        heartbeat_timeout=90.0,
        start_method="spawn",
    )


# ── Socket-broker mode: skip in-process registration ───────────────────


def test_start_module_skips_subscribers_when_socket_broker_attached(monkeypatch):
    """With bus.has_socket_broker=True, _start_module must NOT add info.queue
    to bus._subscribers[name]. Worker connects to broker via socket; the
    in-process queue would be an orphan (worker reads SocketQueue)."""
    bus = DivineBus()
    # Attach a fake broker so has_socket_broker becomes True.
    fake_broker = MagicMock()
    fake_broker.sock_path = "/tmp/test_titan_bus.sock"
    bus.attach_broker(fake_broker)
    assert bus.has_socket_broker is True

    g = Guardian(bus)
    spec = _make_spec("outer_trinity")
    g.register(spec)

    # Mock multiprocessing so we don't actually spawn a process.
    fake_proc = MagicMock()
    fake_proc.pid = 999999
    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda method: MagicMock(
            Queue=lambda maxsize=0: MagicMock(),
            Process=lambda **kwargs: fake_proc,
        ),
    )

    g.start("outer_trinity")

    # Module name registered in bus._modules (still needed for /v3/trinity
    # bus_modules list + arch_map module enumeration).
    assert "outer_trinity" in bus._modules

    # CRITICAL — info.queue must NOT be in bus._subscribers under socket mode.
    # Worker reads SocketQueue (rebound by setup_worker_bus); registering the
    # in-process queue here creates the orphan that caused 686k drops on
    # 2026-04-30.
    assert "outer_trinity" not in bus._subscribers, (
        "Under socket-broker mode, _start_module must not register info.queue "
        "as an in-process bus subscriber — that path is the orphan-queue "
        "regression fixed by Phase B.2 §D7+§D12 gating. Phase B.3 will delete "
        "this path entirely."
    )


def test_start_module_keeps_module_recv_queues_under_socket_mode(monkeypatch):
    """Even under socket mode info.queue is allocated (still used by
    Guardian.broadcast_to_modules / call_module RPC paths). Only the bus
    subscriber registration is skipped."""
    bus = DivineBus()
    fake_broker = MagicMock()
    bus.attach_broker(fake_broker)

    g = Guardian(bus)
    g.register(_make_spec("body"))

    fake_proc = MagicMock()
    fake_proc.pid = 999998
    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda method: MagicMock(
            Queue=lambda maxsize=0: MagicMock(),
            Process=lambda **kwargs: fake_proc,
        ),
    )

    g.start("body")

    info = g._modules["body"]
    # B.3 Stage 1 (2026-05-02): under socket mode, info.queue + info.send_queue
    # are now None (no mp.Queue allocated). Closes fork-at-locked-mp.Queue
    # hazard. See PLAN_microkernel_phase_b3_legacy_path_cleanup.md §1.
    assert info.queue is None, "info.queue must be None under socket mode (B.3 Stage 1)"
    assert info.send_queue is None, "info.send_queue must be None under socket mode (B.3 Stage 1)"
    # _module_recv_queues bookkeeping is also skipped when info.queue is None.
    assert "body" not in g._module_recv_queues, (
        "_module_recv_queues skips workers without mp.Queue (B.3 Stage 1)")


# ── Legacy mp.Queue mode: registration unchanged ───────────────────────


def test_start_module_registers_subscribers_when_no_socket_broker(monkeypatch):
    """Without a socket broker (legacy mp.Queue mode), behavior must be
    unchanged: info.queue registered as in-process bus subscriber so
    targeted dst=<name> messages route correctly."""
    bus = DivineBus()
    assert bus.has_socket_broker is False  # No attach_broker call

    g = Guardian(bus)
    spec = _make_spec("memory")
    g.register(spec)

    fake_queue = MagicMock()
    fake_proc = MagicMock()
    fake_proc.pid = 999997
    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda method: MagicMock(
            Queue=lambda maxsize=0: fake_queue,
            Process=lambda **kwargs: fake_proc,
        ),
    )

    g.start("memory")

    # Legacy path — info.queue MUST be registered so bus.publish(dst="memory")
    # routes to the worker's in-process recv queue.
    assert "memory" in bus._subscribers
    assert fake_queue in bus._subscribers["memory"]
    assert "memory" in bus._modules


# ── Regression guard for B.3 cleanup ────────────────────────────────────


def test_b3_cleanup_seam_documented():
    """Pin the comment that documents the B.3 cleanup chain — if a future
    refactor removes it, this test fails so we revisit the cleanup plan.

    Updated 2026-05-02 for B.3 Stage 1 — the seam now points at the
    fork-at-locked-mp.Queue avoidance comment + the new B.3 PLAN doc.
    """
    import inspect

    from titan_hcl import guardian as g_mod

    src = inspect.getsource(g_mod._start_module if hasattr(g_mod, "_start_module") else g_mod.Guardian)
    assert ("B.3 Stage 1" in src
            or "B.3 cleanup §11.4" in src
            or "Phase B.2 IPC §D7" in src
            or "fork-at-locked-mp.Queue" in src), (
        "The B.3 cleanup seam comment in Guardian._start_module must remain. "
        "Stage 1 (2026-05-02) introduced fork-at-locked-mp.Queue avoidance; "
        "the seam points at PLAN_microkernel_phase_b3_legacy_path_cleanup.md. "
        "Don't remove without updating the cleanup PLAN."
    )
