"""
Tests for ``titan_plugin.logic.memory_state_publisher``.

Phase C Session 2 of rFP_phase_c_async_shm_consumer_migration §4.B.8.

Run: ``python -m pytest tests/test_memory_state_publisher.py -v -p no:anchorpy``
"""
from __future__ import annotations

import logging
import time

import msgpack
import pytest

from titan_plugin.core.state_registry import StateRegistryReader
from titan_plugin.logic.memory_state_publisher import MemoryStatePublisher
from titan_plugin.logic.memory_state_specs import (
    MEMORY_STATE_SLOT,
    MEMORY_STATE_SPEC,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


class _StubGraph:
    def __init__(self, stats: dict | None = None):
        self._stats = stats or {"Person": 12, "Topic": 5, "BodyEntity": 3}
        self._cached_edge_count = 47

    def get_stats(self):
        return self._stats


class _StubMemory:
    """Stub TieredMemoryGraph-shape with exactly the attrs the publisher reads."""
    def __init__(self, persistent_count=10, mempool=None,
                 cognee_ready=True, node_store=None, graph=None):
        self._persistent_count = persistent_count
        self._mempool = mempool if mempool is not None else [
            {"x": 1}, {"x": 2}, {"x": 3}]
        self._cognee_ready = cognee_ready
        self._node_store = node_store if node_store is not None else {
            f"n{i}": {
                "type": "MemoryNode",
                "status": "persistent",
                "effective_weight": 1.2 if i % 2 == 0 else 0.9,
                "created_at": time.time() - 100,
                "last_accessed": time.time() - 50,
            } for i in range(8)
        }
        self._graph = graph if graph is not None else _StubGraph()

    def get_persistent_count(self):
        return self._persistent_count


# ── 1. Init ───────────────────────────────────────────────────────────


def test_init_logs(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_plugin.logic.memory_state_publisher")
    pub = MemoryStatePublisher(titan_id="T_TEST")
    stats = pub.get_stats()
    assert stats["publish_count"] == 0
    assert stats["writer_attached"] is False
    assert any("initialized" in r.message for r in caplog.records)


# ── 2. Cold-boot publish (memory=None) ───────────────────────────────


def test_publish_cold_boot_with_none_memory(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_plugin.logic.memory_state_publisher")
    pub = MemoryStatePublisher(titan_id="T_TEST")
    pub.publish(memory=None)
    stats = pub.get_stats()
    assert stats["publish_count"] == 1
    assert stats["publish_success"] == 1
    assert stats["write_fails"] == 0
    # First-publish-success log fires
    assert any("FIRST PUBLISH SUCCESS" in r.message for r in caplog.records)


# ── 3. Publish with rich memory state ─────────────────────────────────


def test_publish_rich_memory_writes_correct_payload(shm_root):
    pub = MemoryStatePublisher(titan_id="T_TEST")
    mem = _StubMemory(persistent_count=42)
    pub.publish(mem)
    # Read back
    reader = StateRegistryReader(MEMORY_STATE_SPEC, shm_root)
    raw = reader.read_variable()
    assert raw is not None
    decoded = msgpack.unpackb(raw, raw=False)
    # Schema check
    for key in ("cognee_ready", "persistent_count", "mempool_size",
                "effective_nodes_24h", "high_quality_count",
                "total_persistent_for_growth", "learning_velocity",
                "directive_alignment", "kg_node_count", "kg_edge_count",
                "kg_stats_by_table", "ts"):
        assert key in decoded, f"missing key {key}"
    # Value sanity
    assert decoded["persistent_count"] == 42
    assert decoded["mempool_size"] == 3
    assert decoded["cognee_ready"] is True
    assert decoded["total_persistent_for_growth"] == 8
    # 4 of 8 nodes have effective_weight >= 1.15 (the even-indexed ones)
    assert decoded["high_quality_count"] == 4
    assert decoded["kg_node_count"] == 12 + 5 + 3  # = 20
    assert decoded["kg_edge_count"] == 47
    # learning_velocity bounded [0, 1]
    assert 0.0 <= decoded["learning_velocity"] <= 1.0
    # directive_alignment = high_quality / total * 1.2 = 4/8 * 1.2 = 0.6
    assert decoded["directive_alignment"] == pytest.approx(0.6)


# ── 4. Defensive against partial memory ───────────────────────────────


def test_publish_with_partial_memory_attrs(shm_root):
    """Memory ref with missing attrs (broken init) — publisher should
    publish stub fields without crashing."""
    class _Partial:
        # Missing _node_store, _graph, _cognee_ready, get_persistent_count
        _mempool = []

    pub = MemoryStatePublisher(titan_id="T_TEST")
    pub.publish(memory=_Partial())
    stats = pub.get_stats()
    assert stats["publish_success"] == 1


# ── 5. Heartbeat logs at canonical ticks ──────────────────────────────


def test_heartbeat_at_tick_1_and_10(shm_root, caplog):
    caplog.set_level(logging.INFO,
                     logger="titan_plugin.logic.memory_state_publisher")
    pub = MemoryStatePublisher(titan_id="T_TEST")
    mem = _StubMemory()
    for _ in range(10):
        pub.publish(mem)
    heartbeat_logs = [r for r in caplog.records if "heartbeat" in r.message]
    assert len(heartbeat_logs) >= 2


# ── 6. Monotonic ts advancement (G21 single-writer) ───────────────────


def test_repeated_publish_advances_ts(shm_root):
    pub = MemoryStatePublisher(titan_id="T_TEST")
    mem = _StubMemory()
    pub.publish(mem)
    reader = StateRegistryReader(MEMORY_STATE_SPEC, shm_root)
    raw1 = reader.read_variable()
    decoded1 = msgpack.unpackb(raw1, raw=False)
    time.sleep(0.01)
    pub.publish(mem)
    raw2 = reader.read_variable()
    decoded2 = msgpack.unpackb(raw2, raw=False)
    assert decoded2["ts"] > decoded1["ts"]


# ── 7. Broken graph property (G20 hot-path resilience) ────────────────


def test_publish_with_broken_graph_doesnt_crash(shm_root, caplog):
    """A graph object whose get_stats raises must not prevent the
    publisher from writing the slot — publisher logs warning and
    continues with kg_node_count=0."""
    caplog.set_level(logging.WARNING,
                     logger="titan_plugin.logic.memory_state_publisher")

    class _BadGraph:
        def get_stats(self):
            raise RuntimeError("kuzu connection died")

    mem = _StubMemory(graph=_BadGraph())
    pub = MemoryStatePublisher(titan_id="T_TEST")
    pub.publish(mem)
    stats = pub.get_stats()
    assert stats["publish_success"] == 1  # still wrote slot
    # Read payload — kg_node_count should be 0 (defensive default)
    reader = StateRegistryReader(MEMORY_STATE_SPEC, shm_root)
    decoded = msgpack.unpackb(reader.read_variable(), raw=False)
    assert decoded["kg_node_count"] == 0


# ── 8. get_stats shape ────────────────────────────────────────────────


def test_get_stats_shape(shm_root):
    pub = MemoryStatePublisher(titan_id="T_TEST")
    pub.publish(_StubMemory())
    stats = pub.get_stats()
    for key in ("publish_count", "publish_success", "encode_fails",
                "oversize_fails", "write_fails", "writer_attached"):
        assert key in stats
    assert stats["writer_attached"] is True
