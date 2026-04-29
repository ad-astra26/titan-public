"""Tests for the rewritten TimeChain.rebuild_index() (2026-04-27 PM).

The rewrite replaces per-block sqlite3 connections with a single transaction
+ executemany INSERT, plus calls _register_primary_forks before scanning
so all 7 primary forks end up registered (the prior version silently left
forks 0..5 unregistered if fork_registry was empty post-DELETE).

Measured speedup on T1's chain set: 30+ minutes → 20 seconds (90× faster).
"""
from __future__ import annotations

import os
import sqlite3
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.logic.timechain import (
    TimeChain, FORK_NAMES, FORK_MAIN, FORK_DECLARATIVE, FORK_EPISODIC,
)


class TestRebuildIndexFast(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "timechain"
        self.data_dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_chain_with_blocks(self, n_main: int = 5, n_decl: int = 3,
                                  n_epi: int = 7) -> TimeChain:
        """Create a TimeChain + commit some blocks to populate chain files."""
        from titan_plugin.logic.timechain import BlockPayload
        tc = TimeChain(data_dir=str(self.data_dir), titan_id="TEST")
        # Genesis (needs content dict)
        tc.create_genesis({"birth_block": True, "test_run": True})
        nm = {"DA": 0.5, "ACh": 0.5, "NE": 0.5, "5HT": 0.5,
              "GABA": 0.5, "endorphin": 0.5}
        # Add blocks to forks
        for i in range(n_decl):
            tc.commit_block(
                FORK_DECLARATIVE, epoch_id=i,
                payload=BlockPayload(
                    thought_type="declarative", source="test",
                    content={"test_idx": i},
                    significance=0.5, tags=["test"]),
                pot_nonce=1, chi_spent=0.001, neuromod_state=nm,
            )
        for i in range(n_epi):
            tc.commit_block(
                FORK_EPISODIC, epoch_id=10 + i,
                payload=BlockPayload(
                    thought_type="episodic", source="test",
                    content={"epi_idx": i},
                    significance=0.5, tags=[]),
                pot_nonce=1, chi_spent=0.001, neuromod_state=nm,
            )
        return tc

    def test_rebuild_recovers_all_forks_in_fork_registry(self):
        """After rebuild on a fresh DB, fork_registry must contain ALL 7
        primary forks (regression: prior implementation only had FORK_SYSTEM
        when fork_registry was empty pre-rebuild)."""
        tc = self._seed_chain_with_blocks()
        # Wipe the index to simulate corruption recovery
        tc._index_db_path.unlink()
        # Reinit: fresh empty DB with only FORK_SYSTEM (via _ensure_system_fork_registered)
        tc._init_index_db()
        # Rebuild
        tc.rebuild_index()

        # Verify all 7 primary forks are in fork_registry
        c = sqlite3.connect(str(tc._index_db_path))
        try:
            forks = c.execute(
                "SELECT fork_id, fork_name FROM fork_registry"
            ).fetchall()
        finally:
            c.close()
        fork_ids = {row[0] for row in forks}
        self.assertEqual(fork_ids, set(FORK_NAMES.keys()),
                         f"Expected all FORK_NAMES, got {fork_ids}")

    def test_rebuild_repopulates_block_index(self):
        """All blocks from .bin files should appear in block_index."""
        tc = self._seed_chain_with_blocks(n_main=5, n_decl=3, n_epi=7)
        # Total blocks: 1 genesis (FORK_MAIN) + 3 declarative + 7 episodic = 11
        original_total = tc._total_blocks
        self.assertGreaterEqual(original_total, 11)

        # Wipe index, rebuild
        tc._index_db_path.unlink()
        tc._init_index_db()
        tc.rebuild_index()

        # block_index should have the same row count
        c = sqlite3.connect(str(tc._index_db_path))
        try:
            count = c.execute(
                "SELECT COUNT(*) FROM block_index").fetchone()[0]
        finally:
            c.close()
        self.assertEqual(count, original_total,
                         f"block_index has {count} rows, expected {original_total}")

    def test_rebuild_updates_fork_tips(self):
        """fork_tips dict reflects max heights from rebuilt index."""
        tc = self._seed_chain_with_blocks(n_main=1, n_decl=3, n_epi=7)
        tc._index_db_path.unlink()
        tc._init_index_db()
        tc.rebuild_index()

        # FORK_DECLARATIVE has 3 blocks (heights 0, 1, 2)
        # FORK_EPISODIC has 7 blocks (heights 0-6)
        self.assertEqual(tc._fork_tips[FORK_DECLARATIVE][0], 2)
        self.assertEqual(tc._fork_tips[FORK_EPISODIC][0], 6)

    def test_rebuild_integrity_check_passes(self):
        """The rebuilt index.db passes PRAGMA integrity_check."""
        tc = self._seed_chain_with_blocks()
        tc._index_db_path.unlink()
        tc._init_index_db()
        tc.rebuild_index()

        c = sqlite3.connect(str(tc._index_db_path))
        try:
            result = c.execute("PRAGMA integrity_check").fetchall()
        finally:
            c.close()
        self.assertEqual(result, [("ok",)])

    def test_rebuild_idempotent(self):
        """Calling rebuild twice produces the same end state."""
        tc = self._seed_chain_with_blocks()

        tc.rebuild_index()
        c = sqlite3.connect(str(tc._index_db_path))
        try:
            count1 = c.execute("SELECT COUNT(*) FROM block_index").fetchone()[0]
        finally:
            c.close()

        tc.rebuild_index()
        c = sqlite3.connect(str(tc._index_db_path))
        try:
            count2 = c.execute("SELECT COUNT(*) FROM block_index").fetchone()[0]
        finally:
            c.close()

        self.assertEqual(count1, count2)

    def test_rebuild_speed_sanity(self):
        """Sanity: rebuild on a small chain completes in <2s. Bigger
        speedup (~90×) is verified manually on T1 production data."""
        tc = self._seed_chain_with_blocks(n_main=1, n_decl=20, n_epi=50)
        tc._index_db_path.unlink()
        tc._init_index_db()
        t0 = time.time()
        tc.rebuild_index()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 2.0,
                        f"rebuild took {elapsed:.2f}s — should be <2s for "
                        f"~70 blocks (per-block sqlite-connect would be ~5s+)")

    def test_rebuild_after_main_db_unlink(self):
        """Rebuild after unlinking + re-init recovers usable state.
        Simulates the recovery flow used 2026-04-27 PM on T1."""
        tc = self._seed_chain_with_blocks()

        # Save reference state
        original_blocks = tc._total_blocks
        original_decl_tip = tc._fork_tips[FORK_DECLARATIVE]

        # Wipe + reinit + rebuild
        tc._index_db_path.unlink()
        tc._init_index_db()
        tc.rebuild_index()

        # State recovered
        self.assertEqual(tc._total_blocks, original_blocks)
        self.assertEqual(tc._fork_tips[FORK_DECLARATIVE], original_decl_tip)

        # Re-instantiating TimeChain on the rebuilt index sees the same forks
        tc2 = TimeChain(data_dir=str(self.data_dir), titan_id="TEST")
        self.assertGreaterEqual(len(tc2._fork_tips), 7)


if __name__ == "__main__":
    unittest.main()
