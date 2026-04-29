"""Microkernel v2 Phase A §A.4 (S5 amendment) D3 — TitanStateAccessor unit tests.

Covers the StateAccessor abstraction + cache shims that survived from
the Apr 25-26 session work:

  - _CacheGetter / _CallableValue await contracts (defensive cover for
    no-parens `await titan_state.X.Y` legacy patterns)
  - _CacheGetter / _CallableValue serialization safety (no _CallableValue
    leak into JSON — regression from commit 3dd3ae84 → 019881bf)
  - TitanStateAccessor.__getattr__ private-attr cache fallback (D2 fix
    for chat.py reads of plugin._dream_inbox / plugin._current_user_id)
  - _EventBusShim.emit() wraps in _AwaitableValue (D2 — supports
    legacy `await plugin.event_bus.emit(...)` callsites in maker.py +
    webhook.py without rewriting)
  - _BusShim.publish() routes via CommandSender (D2 — supports legacy
    plugin.bus.publish(make_msg(...)) in api/__init__.py + maker module)
  - _BusShim.stats reads bus.stats from cache (kernel snapshot publishes)
  - SpiritAccessor new typed methods (D1 follow-up — 4 new endpoint paths)
  - CommandSender.emit() publishes OBSERVATORY_EVENT bus type (D2)

All tests use fake CachedState + fake send_queue — no real bus, shm, or
subprocess overhead. Run via:

  source test_env/bin/activate
  python -m pytest tests/test_state_accessor.py -v -p no:anchorpy
"""

import asyncio
import json

import pytest

from titan_plugin.api.cached_state import CachedState
from titan_plugin.api.command_sender import CommandSender
from titan_plugin.api.shm_reader_bank import ShmReaderBank
from titan_plugin.api.state_accessor import (
    SpiritAccessor,
    TitanStateAccessor,
    _AwaitableValue,
    _BusShim,
    _CacheGetter,
    _CallableValue,
    _EventBusShim,
)


class _FakeQueue:
    """Captures put() calls so tests can assert on bus traffic without spinning a real DivineBus."""

    def __init__(self):
        self.sent: list = []

    def put(self, msg):
        self.sent.append(msg)


@pytest.fixture
def cache():
    c = CachedState()
    return c


@pytest.fixture
def commands():
    return CommandSender(send_queue=_FakeQueue())


@pytest.fixture
def state(cache, commands):
    """Build a TitanStateAccessor with stub deps. ShmReaderBank is bare;
    sub-accessors that touch shm degrade gracefully (return defaults)."""
    return TitanStateAccessor(
        shm=ShmReaderBank(titan_id="TEST"),
        cache=cache,
        commands=commands,
        full_config={"network": {"vault_program_id": "test_pid"}},
    )


# ── _CacheGetter / _CallableValue serialization + await contracts ─────


def test_cache_getter_call_returns_raw_dict_for_json(cache):
    """_CacheGetter.__call__ must return JSON-serializable raw dict, not
    a _CallableValue wrapper. Regression test for /v4/metabolic-state
    (commit 019881bf)."""
    cache.set("metabolism", {"energy": 0.5})
    g = _CacheGetter("metabolism", cache)
    result = g()
    assert isinstance(result, dict)
    assert result == {"energy": 0.5}
    # Must JSON-serialize cleanly
    json.dumps(result)


def test_cache_getter_call_empty_returns_empty_dict(cache):
    """Missing key → empty dict, not _CallableValue."""
    g = _CacheGetter("nonexistent", cache)
    result = g()
    assert result == {}
    json.dumps(result)


def test_cache_getter_await_resolves_to_value(cache):
    """`await titan_state.X` (no method invocation, no parens) must work."""
    cache.set("metabolism", {"energy": 0.5})
    g = _CacheGetter("metabolism", cache)

    async def _t():
        return await g

    result = asyncio.run(_t())
    assert result == {"energy": 0.5}


def test_callable_value_await_resolves_to_inner(cache):
    """`await _CallableValue(x)` resolves to x — defensive cover for
    legacy `await titan_state.X.Y` (no parens)."""
    cv = _CallableValue({"a": 1})

    async def _t():
        return await cv

    result = asyncio.run(_t())
    assert result == {"a": 1}


def test_callable_value_call_returns_inner_for_json(cache):
    """`titan_state.X.Y()` returns raw inner value, not wrapped — endpoints
    that JSON-serialize must not see _CallableValue leaks."""
    cv = _CallableValue([1, 2, 3])
    result = cv()
    assert result == [1, 2, 3]
    json.dumps(result)


# ── TitanStateAccessor.__getattr__ private fallback (D2) ──────────────


def test_titan_state_private_attr_falls_back_to_cache(state, cache):
    """plugin._dream_inbox → cache.get('plugin._dream_inbox', None) — D2
    fix for chat.py reads in microkernel mode."""
    cache.set("plugin._dream_inbox", ["msg1", "msg2", "msg3"])
    assert state._dream_inbox == ["msg1", "msg2", "msg3"]


def test_titan_state_private_attr_missing_raises_attributeerror(state):
    """Missing private cache key raises AttributeError so hasattr() returns
    False — preserves legacy mode-detection patterns like
    `if hasattr(plugin, '_proxies')`. (D2 v2 — original always-return-None
    broke /v3/trinity warmer detection, caused str-float crash.)"""
    with pytest.raises(AttributeError):
        _ = state._never_published_attr
    assert hasattr(state, "_never_published_attr") is False
    # getattr with default still works for callers that want None
    assert getattr(state, "_never_published_attr", "DEFAULT") == "DEFAULT"


def test_titan_state_public_unknown_attr_returns_cache_getter(state):
    """Public unknown sub-accessor name → _CacheGetter (synthesis fallback)."""
    result = state.metabolism  # not a typed accessor
    assert isinstance(result, _CacheGetter)


# ── _EventBusShim — D2 maker.py + webhook.py compat ───────────────────


def test_event_bus_shim_emit_publishes_observatory_event(commands):
    """plugin.event_bus.emit(type, payload) → CommandSender → OBSERVATORY_EVENT."""
    eb = _EventBusShim(commands)
    rid = eb.emit("directive_update", {"text": "test"})
    assert isinstance(rid, _AwaitableValue)
    sent = commands._send_queue.sent
    assert len(sent) == 1
    assert sent[0]["type"] == "OBSERVATORY_EVENT"
    assert sent[0]["dst"] == "all"
    assert sent[0]["payload"]["event_type"] == "directive_update"
    assert sent[0]["payload"]["data"] == {"text": "test"}


def test_event_bus_shim_emit_is_awaitable(commands):
    """Legacy code does `await plugin.event_bus.emit(...)` — must not crash."""
    eb = _EventBusShim(commands)

    async def _t():
        rid = await eb.emit("test_event", {"k": "v"})
        return rid

    result = asyncio.run(_t())
    assert isinstance(result, str)  # request_id from CommandSender.publish


def test_event_bus_shim_subscriber_count(commands):
    eb = _EventBusShim(commands)
    assert eb.subscriber_count == 0  # microkernel: api_subprocess-managed


# ── _BusShim — D2 api/__init__.py + maker module compat ───────────────


def test_bus_shim_publish_routes_msg_dict(commands):
    """plugin.bus.publish(make_msg(...)) → CommandSender.publish."""
    from titan_plugin.bus import make_msg

    bus = _BusShim(commands, CachedState())
    msg = make_msg("MAKER_PROPOSAL_CREATED", "titan_maker", "all",
                   {"proposal_id": "p1"})
    delivered = bus.publish(msg)
    assert delivered == 1
    sent = commands._send_queue.sent
    assert len(sent) == 1
    assert sent[0]["type"] == "MAKER_PROPOSAL_CREATED"
    assert sent[0]["src"] == "titan_maker"


def test_bus_shim_publish_invalid_returns_zero(commands):
    bus = _BusShim(commands, CachedState())
    assert bus.publish(None) == 0
    assert bus.publish("not a dict") == 0


def test_bus_shim_stats_reads_from_cache(commands):
    cache = CachedState()
    cache.set("bus.stats", {"published": 100, "dropped": 0, "routed": 95})
    bus = _BusShim(commands, cache)
    stats = bus.stats
    assert stats == {"published": 100, "dropped": 0, "routed": 95}


def test_bus_shim_stats_empty_when_no_cache(commands):
    bus = _BusShim(commands, CachedState())
    assert bus.stats == {}


def test_bus_shim_truthy(commands):
    """if plugin.bus: should pass even on empty cache."""
    bus = _BusShim(commands, CachedState())
    assert bool(bus) is True
    assert bus is not None


# ── SpiritAccessor new typed methods (D1 follow-up) ───────────────────


def test_spirit_accessor_get_nervous_system_reads_cache(cache):
    """SpiritAccessor.get_nervous_system → cache.get('spirit.neural_nervous_system')."""
    cache.set("spirit.neural_nervous_system",
              {"version": "v5_neural", "total_transitions": 14_000_000})
    spirit = SpiritAccessor(shm=ShmReaderBank(titan_id="TEST"), cache=cache)
    result = spirit.get_nervous_system()
    assert result["version"] == "v5_neural"
    assert result["total_transitions"] == 14_000_000


def test_spirit_accessor_get_expression_composites_reads_cache(cache):
    cache.set("spirit.expression_composites",
              {"SPEAK": {"urge": 0.5, "fire_count": 7}})
    spirit = SpiritAccessor(shm=ShmReaderBank(titan_id="TEST"), cache=cache)
    result = spirit.get_expression_composites()
    assert result["SPEAK"]["urge"] == 0.5


def test_spirit_accessor_get_unified_spirit_reads_cache(cache):
    cache.set("spirit.unified_spirit", {"epoch_count": 1040, "magnitude": 5.72})
    spirit = SpiritAccessor(shm=ShmReaderBank(titan_id="TEST"), cache=cache)
    result = spirit.get_unified_spirit()
    assert result["epoch_count"] == 1040


def test_spirit_accessor_missing_keys_return_empty_dict(cache):
    """All new typed methods return {} when cache key missing — never crash."""
    spirit = SpiritAccessor(shm=ShmReaderBank(titan_id="TEST"), cache=cache)
    assert spirit.get_nervous_system() == {}
    assert spirit.get_expression_composites() == {}
    assert spirit.get_resonance() == {}
    assert spirit.get_unified_spirit() == {}


# ── CachedState diagnostic surface (powers /v4/cache-staleness) ───────


def test_cached_state_staleness_report_per_key():
    """staleness_report() returns per-key age — what /v4/cache-staleness
    serializes."""
    import time

    c = CachedState()
    c.set("key1", "v1")
    time.sleep(0.05)
    c.set("key2", "v2")
    report = c.staleness_report()
    assert "key1" in report
    assert "key2" in report
    assert report["key1"] >= report["key2"]  # key1 older


def test_cached_state_get_with_age_returns_age_for_present_key():
    c = CachedState()
    c.set("k", "v")
    val, age = c.get_with_age("k")
    assert val == "v"
    assert age is not None
    assert age >= 0


def test_cached_state_get_with_age_returns_none_for_missing():
    c = CachedState()
    val, age = c.get_with_age("missing", default="dflt")
    assert val == "dflt"
    assert age is None


# ── CommandSender — D2 emit() addition ────────────────────────────────


def test_command_sender_emit_routes_observatory_event(commands):
    """CommandSender.emit() — direct call (without shim wrapper)."""
    rid = commands.emit("custom_event", {"data": "x"})
    assert isinstance(rid, str) and rid
    sent = commands._send_queue.sent
    assert sent[0]["type"] == "OBSERVATORY_EVENT"
    assert sent[0]["payload"]["event_type"] == "custom_event"


def test_command_sender_publish_returns_request_id(commands):
    """publish() returns a non-empty request_id when send_queue accepts."""
    rid = commands.publish("TEST_TYPE", "spirit", {"k": "v"})
    assert isinstance(rid, str) and len(rid) >= 8


def test_command_sender_no_queue_publish_noop():
    """CommandSender constructed with send_queue=None must not crash."""
    cs = CommandSender(send_queue=None)
    rid = cs.publish("TEST", "spirit", {})
    assert rid == ""
    rid2 = cs.emit("event", {})
    assert rid2 == ""
