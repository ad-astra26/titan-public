"""Tests for ModuleSpec.broadcast_topics — per-worker broadcast filter.

Closes the 2026-04-30 backup-queue flood regression: spawn_graduated workers
connected via Unix socket broker received ALL `dst="all"` broadcasts because
the broker treats empty subscribed_topics as "subscribe-all" (legacy mode).

Backup worker only handles MEDITATION_COMPLETE + BACKUP_TRIGGER_MANUAL but
was getting flooded with SPHERE_PULSE/SPIRIT_STATE/SAGE_STATS/PI_HEARTBEAT/
DREAMING_STATE/etc.

Architecture: ModuleSpec.broadcast_topics list → Guardian.spawn → ctx.Process
arg → _module_wrapper → setup_worker_bus(topics=...) → BusSocketClient(topics=...)
→ broker filters at publish time.

Empty list = legacy subscribe-all (preserves backward compat).
"""
from titan_hcl.guardian_hcl import ModuleSpec


def test_broadcast_topics_default_empty():
    """Default ModuleSpec has empty broadcast_topics — legacy subscribe-all."""
    def noop(*args, **kwargs): pass
    spec = ModuleSpec(name="test", entry_fn=noop)
    assert spec.broadcast_topics == []


def test_broadcast_topics_explicit_list():
    """Explicit list passes through unchanged."""
    def noop(*args, **kwargs): pass
    spec = ModuleSpec(
        name="test",
        entry_fn=noop,
        broadcast_topics=["MEDITATION_COMPLETE", "BACKUP_TRIGGER_MANUAL"],
    )
    assert spec.broadcast_topics == ["MEDITATION_COMPLETE", "BACKUP_TRIGGER_MANUAL"]


def test_broadcast_topics_independent_per_spec():
    """Default factory must yield distinct lists per ModuleSpec instance."""
    def noop(*args, **kwargs): pass
    a = ModuleSpec(name="a", entry_fn=noop)
    b = ModuleSpec(name="b", entry_fn=noop)
    a.broadcast_topics.append("X")
    assert b.broadcast_topics == []  # b should not pick up a's mutation


def test_module_wrapper_signature_accepts_topics():
    """_module_wrapper takes broadcast_topics as 7th positional arg."""
    import inspect
    from titan_hcl.guardian_hcl import _module_wrapper
    sig = inspect.signature(_module_wrapper)
    params = list(sig.parameters.keys())
    assert "broadcast_topics" in params, f"broadcast_topics missing from: {params}"


def test_setup_worker_bus_accepts_topics_kwarg():
    """setup_worker_bus accepts topics= keyword (kw-only after env=)."""
    import inspect
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    sig = inspect.signature(setup_worker_bus)
    assert "topics" in sig.parameters, f"topics missing from: {list(sig.parameters)}"
    assert sig.parameters["topics"].kind == inspect.Parameter.KEYWORD_ONLY


def test_setup_worker_bus_legacy_mode_ignores_topics():
    """Legacy mode (no env vars) returns recv_q/send_q unchanged regardless of topics."""
    import multiprocessing
    from titan_hcl.core.worker_bus_bootstrap import setup_worker_bus
    rq = multiprocessing.Queue()
    sq = multiprocessing.Queue()
    new_rq, new_sq, client = setup_worker_bus(
        "test_worker", rq, sq, env={}, topics=["FOO", "BAR"])
    assert new_rq is rq
    assert new_sq is sq
    assert client is None
