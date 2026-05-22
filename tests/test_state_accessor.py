"""
tests/test_state_accessor.py — TitanStateAccessor surface tests.

Post-Phase-D (D-SPEC-80): the bus-cache → CachedState pipeline is RETIRED.
Every sub-accessor is SHM-direct per Preamble G18; the legacy CachedState +
BusSubscriber files no longer exist. These tests cover the
production-grade contracts left after the retirement:
  - _CacheGetter empty-default fallback for unenumerated sub-accessors
  - _CallableValue serialization + await contract
  - TitanStateAccessor.__getattr__ private/public fallback
  - _BusShim.publish routes via CommandSender (legacy plugin.bus.publish
    callsites in api/__init__.py + maker module)
  - _EventBusShim.emit publishes OBSERVATORY_EVENT
  - SpiritAccessor SHM-direct typed methods (Phase B Rust canonical slots)
  - CommandSender.emit + publish

All tests use fake send_queue + mocked ShmReaderBank — no real bus, shm, or
subprocess overhead. Run via:

  source test_env/bin/activate
  python -m pytest tests/test_state_accessor.py -v -p no:anchorpy
"""

import asyncio
import json

import pytest

from titan_hcl.api.command_sender import CommandSender
from titan_hcl.api.shm_reader_bank import ShmReaderBank
from titan_hcl.api.state_accessor import (
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
def commands():
    return CommandSender(send_queue=_FakeQueue())


@pytest.fixture
def state(commands):
    """Build a TitanStateAccessor with stub deps. ShmReaderBank is bare;
    sub-accessors that touch shm degrade gracefully (return defaults)."""
    return TitanStateAccessor(
        shm=ShmReaderBank(titan_id="TEST"),
        commands=commands,
        full_config={"network": {"vault_program_id": "test_pid"}},
    )


# ── _CacheGetter / _CallableValue serialization + await contracts ─────


def test_cache_getter_call_returns_empty_dict_g18_neutralized():
    """Phase A.4 / Phase D closure (D-SPEC-71 / D-SPEC-80): _CacheGetter
    returns empty default per Preamble G18 (state transport is SHM, never
    bus). The fallback exists only as a backward-compat hedge for
    unenumerated sub-accessor names."""
    g = _CacheGetter("metabolism")
    result = g()
    assert isinstance(result, dict)
    assert result == {}
    json.dumps(result)


def test_cache_getter_call_empty_returns_empty_dict():
    """Missing key → empty dict, not _CallableValue."""
    g = _CacheGetter("nonexistent")
    result = g()
    assert result == {}
    json.dumps(result)


def test_cache_getter_await_resolves_to_empty_g18_neutralized():
    """`await titan_state.X` resolves to empty dict per Preamble G18."""
    g = _CacheGetter("metabolism")

    async def _t():
        return await g

    result = asyncio.run(_t())
    assert result == {}


def test_callable_value_await_resolves_to_inner():
    """`await _CallableValue(x)` resolves to x — defensive cover for
    legacy `await titan_state.X.Y` (no parens)."""
    cv = _CallableValue({"a": 1})

    async def _t():
        return await cv

    result = asyncio.run(_t())
    assert result == {"a": 1}


def test_callable_value_call_returns_inner_for_json():
    """`titan_state.X.Y()` returns raw inner value, not wrapped — endpoints
    that JSON-serialize must not see _CallableValue leaks."""
    cv = _CallableValue([1, 2, 3])
    result = cv()
    assert result == [1, 2, 3]
    json.dumps(result)


# ── TitanStateAccessor.__getattr__ fallback (Phase D) ─────────────────


def test_titan_state_private_attr_raises_attributeerror(state):
    """Missing private attribute raises AttributeError so hasattr() returns
    False — preserves legacy mode-detection patterns like
    `if hasattr(plugin, '_proxies')`. (Phase D retired the cache fallback;
    private names now always AttributeError.)"""
    with pytest.raises(AttributeError):
        _ = state._never_published_attr
    assert hasattr(state, "_never_published_attr") is False
    assert getattr(state, "_never_published_attr", "DEFAULT") == "DEFAULT"


def test_titan_state_public_unknown_attr_returns_cache_getter(state):
    """Public unknown sub-accessor name → _CacheGetter (synthesis fallback)."""
    result = state.metabolism  # not a typed accessor
    assert isinstance(result, _CacheGetter)


# ── _EventBusShim — maker.py + webhook.py compat ──────────────────────


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
    assert isinstance(result, str)


def test_event_bus_shim_subscriber_count(commands):
    eb = _EventBusShim(commands)
    assert eb.subscriber_count == 0


# ── _BusShim — api/__init__.py + maker module compat ──────────────────


def test_bus_shim_publish_routes_msg_dict(commands):
    """plugin.bus.publish(make_msg(...)) → CommandSender.publish."""
    from titan_hcl.bus import make_msg

    bus = _BusShim(commands)
    msg = make_msg("MAKER_PROPOSAL_CREATED", "titan_maker", "all",
                   {"proposal_id": "p1"})
    delivered = bus.publish(msg)
    assert delivered == 1
    sent = commands._send_queue.sent
    assert len(sent) == 1
    assert sent[0]["type"] == "MAKER_PROPOSAL_CREATED"
    assert sent[0]["src"] == "titan_maker"


def test_bus_shim_publish_invalid_returns_zero(commands):
    bus = _BusShim(commands)
    assert bus.publish(None) == 0
    assert bus.publish("not a dict") == 0


def test_bus_shim_stats_empty_dict(commands):
    """Phase A.4 / D closure: _BusShim.stats returns empty dict per
    Preamble G18. Canonical bus stats come from kernel_rpc proxy at
    `app.state.titan_hcl.kernel.bus_broker_stats()` per
    dashboard.py /v4/state."""
    bus = _BusShim(commands)
    assert bus.stats == {}


def test_bus_shim_truthy(commands):
    """if plugin.bus: should pass even on empty cache."""
    bus = _BusShim(commands)
    assert bool(bus) is True
    assert bus is not None


# ── SpiritAccessor SHM-direct typed methods (Phase B Rust canonical) ─


def test_spirit_accessor_get_nervous_system_reads_shm():
    """Phase B.5: SpiritAccessor.get_nervous_system reads SHM-direct via
    ShmReaderBank.read_titanvm_registers() (ns_worker L2 publisher;
    titanvm_registers.bin slot)."""
    from unittest.mock import MagicMock
    expected = {"version": "v5_neural", "total_transitions": 14_000_000}
    shm = MagicMock(spec=ShmReaderBank)
    shm.read_titanvm_registers.return_value = expected
    spirit = SpiritAccessor(shm=shm)
    result = spirit.get_nervous_system()
    assert result["version"] == "v5_neural"
    assert result["total_transitions"] == 14_000_000


def test_spirit_accessor_get_expression_composites_reads_shm():
    """Phase A.3: get_expression_composites reads SHM-direct via
    ShmReaderBank.read_expression_state()."""
    from unittest.mock import MagicMock
    expected = {"SPEAK": {"urge": 0.5, "fire_count": 7}}
    shm = MagicMock(spec=ShmReaderBank)
    shm.read_expression_state.return_value = expected
    spirit = SpiritAccessor(shm=shm)
    result = spirit.get_expression_composites()
    assert result["SPEAK"]["urge"] == 0.5
    assert result["SPEAK"]["fire_count"] == 7


def test_spirit_accessor_get_unified_spirit_reads_shm():
    """Phase B.5: SpiritAccessor.get_unified_spirit reads SHM-direct via
    ShmReaderBank.read_unified_spirit_metadata() (Rust-owned
    unified_spirit_metadata.bin slot, Python→Rust ownership flipped at B.0)."""
    from unittest.mock import MagicMock
    expected = {"epoch_count": 1040, "tensor_magnitude": 5.72}
    shm = MagicMock(spec=ShmReaderBank)
    shm.read_unified_spirit_metadata.return_value = expected
    spirit = SpiritAccessor(shm=shm)
    result = spirit.get_unified_spirit()
    assert result["epoch_count"] == 1040


def test_spirit_accessor_missing_keys_return_empty_dict():
    """All typed methods return {} when SHM slot empty/missing — never crash."""
    spirit = SpiritAccessor(shm=ShmReaderBank(titan_id="TEST"))
    assert spirit.get_nervous_system() == {}
    assert spirit.get_expression_composites() == {}
    assert spirit.get_resonance() == {}
    assert spirit.get_unified_spirit() == {}


# ── CommandSender ─────────────────────────────────────────────────────


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
