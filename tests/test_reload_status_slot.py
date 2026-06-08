"""Tests for titan_hcl/core/reload_status.py — the SHM-native hot-reload
result slot (RFP_shm_native_hot_reload.md Phase A).

Verifies the orchestrator-owned `reload_status.bin` writer/reader that REPLACES
the deleted `dst="all"` MODULE_RELOAD_ACK bus broadcast (SPEC §8.2 / G18):
per-module ring, dedup-by-swap_id (intermediate→terminal updates in place),
evict-oldest at N, and the result-dict shape the initiator returns.
"""
import shutil

import pytest

from titan_hcl.core.reload_status import (
    RELOAD_STATUS_RING_N,
    ReloadStatusEntry,
    ReloadStatusReader,
    ReloadStatusWriter,
)
from titan_hcl.core.state_registry import resolve_shm_root

_TID = "treloadstatus"


@pytest.fixture
def slot():
    writer = ReloadStatusWriter(titan_id=_TID)
    reader = ReloadStatusReader(titan_id=_TID)
    yield writer, reader
    writer.close()
    reader.close()
    # Clean the test SHM dir so reruns start fresh (no cross-run carryover).
    try:
        shutil.rmtree(resolve_shm_root(_TID))
    except Exception:
        pass


def _entry(swap_id, status, **kw):
    return ReloadStatusEntry(
        module_name="m", swap_id=swap_id, status=status, **kw)


def test_roundtrip_single_entry(slot):
    writer, reader = slot
    writer.write("m", _entry("s1", "ready", new_pid=42, old_pid=7,
                             total_elapsed_ms=15))
    got = reader.read_swap("m", "s1")
    assert got is not None
    assert got.swap_id == "s1"
    assert got.status == "ready"
    assert got.new_pid == 42
    assert got.old_pid == 7
    assert got.is_terminal


def test_intermediate_then_terminal_updates_in_place(slot):
    """spawning → adopted → ready for ONE swap_id collapses to a single ring
    entry holding the latest status (dedup by swap_id)."""
    writer, reader = slot
    writer.write("m", _entry("s1", "spawning"))
    writer.write("m", _entry("s1", "adopted", new_pid=99))
    writer.write("m", _entry("s1", "ready", new_pid=99))
    ring = reader.read("m")
    assert len(ring) == 1
    assert ring[0].status == "ready"
    assert ring[0].new_pid == 99


def test_ring_evicts_oldest_at_N(slot):
    writer, reader = slot
    for i in range(RELOAD_STATUS_RING_N + 3):
        writer.write("m", _entry(f"s{i}", "ready"))
    ring = reader.read("m")
    assert len(ring) == RELOAD_STATUS_RING_N
    swap_ids = [e.swap_id for e in ring]
    assert "s0" not in swap_ids          # oldest evicted
    assert "s1" not in swap_ids
    assert "s2" not in swap_ids
    assert f"s{RELOAD_STATUS_RING_N + 2}" in swap_ids  # newest kept


def test_read_swap_miss_returns_none(slot):
    writer, reader = slot
    writer.write("m", _entry("s1", "ready"))
    assert reader.read_swap("m", "nope") is None
    assert reader.read_swap("other_module", "s1") is None


def test_per_module_isolation(slot):
    writer, reader = slot
    writer.write("a", _entry("sa", "ready"))
    writer.write("b", _entry("sb", "failed", reason="ready_timeout"))
    assert reader.read_swap("a", "sa").status == "ready"
    assert reader.read_swap("b", "sb").status == "failed"
    assert reader.read_swap("b", "sb").reason == "ready_timeout"
    assert len(reader.read("a")) == 1
    assert len(reader.read("b")) == 1


def test_terminal_classification():
    assert not ReloadStatusEntry("m", "s", "spawning").is_terminal
    assert not ReloadStatusEntry("m", "s", "adopted").is_terminal
    assert ReloadStatusEntry("m", "s", "ready").is_terminal
    assert ReloadStatusEntry("m", "s", "failed").is_terminal
    assert ReloadStatusEntry("m", "s", "rolled_back").is_terminal


def test_as_result_dict_shape():
    """The dict GuardianHCLClient.reload_module returns — SPEC §8.3 shape +
    new_pid for pid-swap confirmation."""
    entry = ReloadStatusEntry(
        module_name="m", swap_id="s1", status="ready", reason=None,
        new_pid=5, old_pid=2, total_elapsed_ms=123)
    result = entry.as_result_dict()
    assert set(result.keys()) == {
        "swap_id", "module_name", "status", "reason", "new_pid",
        "total_elapsed_ms", "ts",
    }
    assert result["swap_id"] == "s1"
    assert result["module_name"] == "m"
    assert result["status"] == "ready"
    assert result["new_pid"] == 5
    assert result["reason"] is None
