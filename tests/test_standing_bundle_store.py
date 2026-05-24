"""StandingBundleStore — Phase 2 (D-P2-4) unit tests.

Covers: round-trip maintain→read, idempotency on dup tx_hash, ring-buffer
eviction at ring_size cap, LRU eviction at max_entities cap, atomic
snapshot export, persistence across reopen, stats surface.

PLAN_synthesis_engine_Phase2.md §2B.6.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from titan_hcl.synthesis.standing_store import (
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_SNAPSHOT_NAME,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_RING_SIZE,
    StandingBundleStore,
)


def _record(tx_hash: str, epoch: int = 0, ts: float = 0.0,
            sig: float = 0.5, src: str = "chat") -> dict:
    return {"tx_hash": tx_hash, "epoch_id": epoch, "ts": ts,
            "significance": sig, "source": src}


class TestMaintainRead(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "synthesis.duckdb")
        self.store = StandingBundleStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_maintain_inserts_newest_first(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A", 1))
        self.store.maintain("user", "h1", "conversation", _record("B", 2))
        self.store.maintain("user", "h1", "conversation", _record("C", 3))
        bundle = self.store.read("user", "h1", "conversation")
        assert [r["tx_hash"] for r in bundle] == ["C", "B", "A"]

    def test_read_empty_for_unknown(self) -> None:
        assert self.store.read("user", "missing", "conversation") == []

    def test_read_returns_copy_not_alias(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A"))
        b1 = self.store.read("user", "h1", "conversation")
        b1[0]["tx_hash"] = "MUTATED"
        b2 = self.store.read("user", "h1", "conversation")
        assert b2[0]["tx_hash"] == "A"

    def test_idempotent_on_duplicate_tx_hash(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A", 1))
        self.store.maintain("user", "h1", "conversation", _record("B", 2))
        self.store.maintain("user", "h1", "conversation", _record("A", 1))
        bundle = self.store.read("user", "h1", "conversation")
        # No duplicate insertion — bundle still [B, A], not [A, B, A].
        assert [r["tx_hash"] for r in bundle] == ["B", "A"]

    def test_distinct_forks_keep_distinct_bundles(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A"))
        self.store.maintain("user", "h1", "procedural", _record("X"))
        assert [r["tx_hash"] for r in
                self.store.read("user", "h1", "conversation")] == ["A"]
        assert [r["tx_hash"] for r in
                self.store.read("user", "h1", "procedural")] == ["X"]


class TestRingBufferEviction(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = StandingBundleStore(
            os.path.join(self.tmp.name, "synthesis.duckdb"), ring_size=3)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_ring_evicts_oldest_at_cap(self) -> None:
        for tag in "ABCD":
            self.store.maintain("user", "h1", "conversation",
                                _record(tag))
        bundle = self.store.read("user", "h1", "conversation")
        # Ring=3 → only newest 3 survive ("A" evicted).
        assert [r["tx_hash"] for r in bundle] == ["D", "C", "B"]

    def test_ring_evictions_counted(self) -> None:
        for tag in "ABCDE":
            self.store.maintain("user", "h1", "conversation",
                                _record(tag))
        # 2 inserts past the cap → 2 ring evictions.
        assert self.store.get_stats()["total_ring_evictions"] == 2


class TestLRUEviction(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = StandingBundleStore(
            os.path.join(self.tmp.name, "synthesis.duckdb"),
            ring_size=10, max_entities=2)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_max_entities_lru_evicts_oldest(self) -> None:
        import time
        # h1 first (oldest), then h2, then h3 — h1 should be LRU-evicted.
        self.store.maintain("user", "h1", "conversation", _record("A"))
        time.sleep(0.005)
        self.store.maintain("user", "h2", "conversation", _record("B"))
        time.sleep(0.005)
        self.store.maintain("user", "h3", "conversation", _record("C"))
        assert self.store.entities_tracked() == 2
        # h1 was the oldest → evicted.
        assert self.store.read("user", "h1", "conversation") == []
        assert self.store.read("user", "h2", "conversation")[0]["tx_hash"] == "B"
        assert self.store.read("user", "h3", "conversation")[0]["tx_hash"] == "C"
        assert self.store.get_stats()["total_lru_evictions"] == 1


class TestPersistence(unittest.TestCase):

    def test_bundles_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "synthesis.duckdb")
            s1 = StandingBundleStore(db_path)
            s1.maintain("user", "h1", "conversation", _record("A", 1))
            s1.maintain("user", "h1", "conversation", _record("B", 2))
            s1.close()
            # Reopen — bundle reload from DuckDB.
            s2 = StandingBundleStore(db_path)
            bundle = s2.read("user", "h1", "conversation")
            assert [r["tx_hash"] for r in bundle] == ["B", "A"]
            s2.close()


class TestSnapshotExport(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "synthesis.duckdb")
        self.snap_path = os.path.join(self.tmp.name, BUNDLE_SNAPSHOT_NAME)
        self.store = StandingBundleStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_snapshot_atomic_writes_full_payload(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A"))
        self.store.maintain("user", "h2", "conversation", _record("B"))
        n = self.store.export_snapshot(self.snap_path)
        assert n == 2
        with open(self.snap_path) as f:
            data = json.load(f)
        assert data["version"] == BUNDLE_SCHEMA_VERSION
        assert "user|h1|conversation" in data["bundles"]
        assert "user|h2|conversation" in data["bundles"]
        assert data["bundles"]["user|h1|conversation"][0]["tx_hash"] == "A"

    def test_snapshot_no_tmp_leftover_after_export(self) -> None:
        self.store.maintain("user", "h1", "conversation", _record("A"))
        self.store.export_snapshot(self.snap_path)
        assert os.path.exists(self.snap_path)
        assert not os.path.exists(self.snap_path + ".tmp")

    def test_snapshot_empty_store_writes_empty_bundles(self) -> None:
        n = self.store.export_snapshot(self.snap_path)
        assert n == 0
        data = json.load(open(self.snap_path))
        assert data["bundles"] == {}


class TestStats(unittest.TestCase):

    def test_stats_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = StandingBundleStore(
                os.path.join(tmp, "synthesis.duckdb"),
                ring_size=2, max_entities=5)
            s.maintain("user", "h1", "conversation", _record("A"))
            s.maintain("user", "h1", "conversation", _record("B"))
            s.maintain("user", "h1", "conversation", _record("C"))  # 1 ring evict
            stats = s.get_stats()
            assert stats["entities_tracked"] == 1
            assert stats["max_entities"] == 5
            assert stats["ring_size"] == 2
            assert stats["total_maintains"] == 3
            assert stats["total_ring_evictions"] == 1
            assert stats["total_lru_evictions"] == 0
            s.close()


if __name__ == "__main__":
    unittest.main()
