"""BridgeRecall.read_bundle — Phase 2 D-P2-4 cross-process read tests.

Covers: bundle snapshot read happy path; missing snapshot → []; stale
watermark → []; corrupt snapshot → []; mtime-cache invalidation on
snapshot update.

PLAN_synthesis_engine_Phase2.md §2B.6.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from titan_hcl.synthesis.bridge_recall import BridgeRecall
from titan_hcl.synthesis.standing_store import (
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_SNAPSHOT_NAME,
)


def _write_bundle_snapshot(path: str, bundles: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": BUNDLE_SCHEMA_VERSION,
        "exported_at": time.time(),
        "bundles": bundles,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


class TestReadBundle(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # Point BridgeRecall at the temp data_dir via TITAN_DATA_DIR
        # (its bundle-snapshot resolver consults the env var).
        self._old_env = os.environ.get("TITAN_DATA_DIR")
        os.environ["TITAN_DATA_DIR"] = self.tmp.name
        # db_path can be anything — read_bundle does NOT consult it.
        self.br = BridgeRecall(
            titan_id="T_TEST",
            db_path=os.path.join(self.tmp.name, "titan_memory.duckdb"),
            freshness_window_s=300.0,
        )

    def tearDown(self) -> None:
        self.br.close()
        if self._old_env is None:
            os.environ.pop("TITAN_DATA_DIR", None)
        else:
            os.environ["TITAN_DATA_DIR"] = self._old_env
        self.tmp.cleanup()

    def _snap_path(self) -> str:
        return os.path.join(self.tmp.name, BUNDLE_SNAPSHOT_NAME)

    def test_missing_snapshot_returns_empty(self) -> None:
        # Bypass watermark to isolate the missing-snapshot soft-fail.
        with patch.object(self.br, "is_fresh", return_value=True):
            assert self.br.read_bundle("user", "h1", "conversation") == []

    def test_stale_watermark_returns_empty(self) -> None:
        _write_bundle_snapshot(self._snap_path(), {
            "user|h1|conversation": [{"tx_hash": "A"}]
        })
        # Watermark stale (is_fresh False) → degraded read.
        with patch.object(self.br, "is_fresh", return_value=False):
            assert self.br.read_bundle("user", "h1", "conversation") == []

    def test_happy_path_returns_bundle(self) -> None:
        _write_bundle_snapshot(self._snap_path(), {
            "user|h1|conversation": [
                {"tx_hash": "A", "epoch_id": 1, "ts": 100.0},
                {"tx_hash": "B", "epoch_id": 2, "ts": 200.0},
            ]
        })
        with patch.object(self.br, "is_fresh", return_value=True):
            result = self.br.read_bundle("user", "h1", "conversation")
        assert len(result) == 2
        assert [r["tx_hash"] for r in result] == ["A", "B"]

    def test_missing_key_returns_empty(self) -> None:
        _write_bundle_snapshot(self._snap_path(), {
            "user|other|conversation": [{"tx_hash": "Z"}]
        })
        with patch.object(self.br, "is_fresh", return_value=True):
            assert self.br.read_bundle("user", "missing", "conversation") == []

    def test_corrupt_snapshot_returns_empty(self) -> None:
        # Write garbage that JSON cannot parse.
        with open(self._snap_path(), "w") as f:
            f.write("{not valid json")
        with patch.object(self.br, "is_fresh", return_value=True):
            assert self.br.read_bundle("user", "h1", "conversation") == []

    def test_mtime_cache_invalidates_on_update(self) -> None:
        _write_bundle_snapshot(self._snap_path(), {
            "user|h1|conversation": [{"tx_hash": "A"}]
        })
        with patch.object(self.br, "is_fresh", return_value=True):
            r1 = self.br.read_bundle("user", "h1", "conversation")
            assert [r["tx_hash"] for r in r1] == ["A"]
            # Sleep just enough to ensure mtime is observable, then
            # rewrite the snapshot — cache should refresh.
            time.sleep(0.05)
            _write_bundle_snapshot(self._snap_path(), {
                "user|h1|conversation": [{"tx_hash": "B"}, {"tx_hash": "A"}]
            })
            os.utime(self._snap_path(), None)
            r2 = self.br.read_bundle("user", "h1", "conversation")
            assert [r["tx_hash"] for r in r2] == ["B", "A"]

    def test_read_returns_copy_not_alias(self) -> None:
        _write_bundle_snapshot(self._snap_path(), {
            "user|h1|conversation": [{"tx_hash": "A"}]
        })
        with patch.object(self.br, "is_fresh", return_value=True):
            r1 = self.br.read_bundle("user", "h1", "conversation")
            r1[0]["tx_hash"] = "MUTATED"
            r2 = self.br.read_bundle("user", "h1", "conversation")
            assert r2[0]["tx_hash"] == "A"

    def test_non_dict_bundle_returns_empty(self) -> None:
        # Defensive — if a future writer-side bug stores a non-list, we
        # should soft-fail.
        _write_bundle_snapshot(self._snap_path(), {
            "user|h1|conversation": "not_a_list"
        })
        with patch.object(self.br, "is_fresh", return_value=True):
            assert self.br.read_bundle("user", "h1", "conversation") == []


if __name__ == "__main__":
    unittest.main()
