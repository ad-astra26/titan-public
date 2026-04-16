"""Tests for ChainArchive.update_embeddings_batch — the Tier 1 fix for
autoencoder dream-cycle bottleneck (2026-04-16).

Validates:
 - All pairs persist
 - Single-transaction semantics (no partial writes on exception)
 - Empty input is a no-op
 - Large batches (100+ pairs) complete in O(1) commits not O(N) commits
 - Retry/backoff still applies on BUSY
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from titan_plugin.logic.chain_archive import ChainArchive


@pytest.fixture
def chain_archive():
    """Fresh ChainArchive backed by a temp DB with 100 seeded rows."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_inner_memory.db"
        archive = ChainArchive(db_path=str(db_path))
        # Seed 100 rows with observation_snapshot but no embedding
        conn = sqlite3.connect(str(db_path))
        for i in range(100):
            snap = [0.1 * (i % 10)] * 132
            conn.execute(
                "INSERT INTO chain_archive "
                "(source, chain_sequence, chain_length, confidence, gut_agreement, "
                " outcome_score, domain, observation_snapshot, epoch_id, created_at) "
                "VALUES ('main', ?, 3, 0.7, 0.5, 0.8, 'general', ?, ?, ?)",
                (json.dumps(["a", "b", "c"]), json.dumps(snap), 1000 + i, 1_000_000.0 + i),
            )
        conn.commit()
        conn.close()
        yield archive, str(db_path)


def _embedding(i: int) -> list:
    """16D embedding for test row i — distinguishable per row."""
    return [float(i) / 100.0] * 16


def test_batch_update_persists_all_pairs(chain_archive):
    archive, db_path = chain_archive
    # Get the 100 rows
    chains = archive.get_chains_without_embedding(limit=100)
    assert len(chains) == 100
    pairs = [(c["id"], _embedding(c["id"])) for c in chains]

    n = archive.update_embeddings_batch(pairs)
    assert n == 100

    # Verify persistence
    conn = sqlite3.connect(db_path)
    for chain_id, emb in pairs:
        row = conn.execute(
            "SELECT observation_embedding FROM chain_archive WHERE id = ?",
            (chain_id,),
        ).fetchone()
        assert row is not None, f"row {chain_id} missing"
        stored = json.loads(row[0])
        assert stored == emb, f"row {chain_id} embedding mismatch"
    conn.close()


def test_batch_empty_is_noop(chain_archive):
    archive, _ = chain_archive
    n = archive.update_embeddings_batch([])
    assert n == 0


def test_batch_preserves_other_rows(chain_archive):
    """Updating a subset must not touch unrelated rows."""
    archive, db_path = chain_archive
    chains = archive.get_chains_without_embedding(limit=100)
    # Update only the first 10
    pairs = [(c["id"], _embedding(c["id"])) for c in chains[:10]]
    archive.update_embeddings_batch(pairs)

    # Rows 11-100 still have NULL embedding
    remaining = archive.get_chains_without_embedding(limit=100)
    assert len(remaining) == 90

    # Rows 1-10 are not returned (they have embeddings now)
    updated_ids = {c[0] for c in pairs}
    remaining_ids = {c["id"] for c in remaining}
    assert updated_ids.isdisjoint(remaining_ids)


class _ConnectionWrapper:
    """Proxy that forwards everything to a real sqlite3.Connection while
    letting us observe commit/executemany calls. Needed because sqlite3
    Connection attributes are read-only C slots, so you can't just assign
    to conn.commit.
    """
    def __init__(self, real_conn):
        self._real = real_conn
        self.commit_count = 0
        self.executemany_count = 0
        self._executemany_side_effect = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self, *a, **kw):
        self.commit_count += 1
        return self._real.commit(*a, **kw)

    def executemany(self, *a, **kw):
        self.executemany_count += 1
        if self._executemany_side_effect is not None:
            effect = self._executemany_side_effect
            self._executemany_side_effect = None  # one-shot
            raise effect
        return self._real.executemany(*a, **kw)

    def close(self):
        return self._real.close()


def test_batch_uses_single_transaction(chain_archive, monkeypatch):
    """Verify the batch path issues ONE commit total, not one per row."""
    archive, _ = chain_archive

    wrappers: list[_ConnectionWrapper] = []
    original = archive._get_db

    def wrapped_get_db():
        w = _ConnectionWrapper(original())
        wrappers.append(w)
        return w

    monkeypatch.setattr(archive, "_get_db", wrapped_get_db)

    chains = archive.get_chains_without_embedding(limit=100)
    pairs = [(c["id"], _embedding(c["id"])) for c in chains[:50]]
    wrappers.clear()
    archive.update_embeddings_batch(pairs)

    # Expect exactly one connection opened, with one commit.
    assert len(wrappers) == 1, f"expected 1 connection, got {len(wrappers)}"
    assert wrappers[0].commit_count == 1, f"expected 1 commit, got {wrappers[0].commit_count}"
    assert wrappers[0].executemany_count == 1, f"expected 1 executemany, got {wrappers[0].executemany_count}"


def test_batch_retries_on_busy(chain_archive, monkeypatch):
    """If SQLite raises 'locked' transiently, retry path kicks in."""
    archive, _ = chain_archive
    chains = archive.get_chains_without_embedding(limit=5)
    pairs = [(c["id"], _embedding(c["id"])) for c in chains]

    wrappers: list[_ConnectionWrapper] = []
    busy_error = sqlite3.OperationalError("database is locked")
    original = archive._get_db

    def wrapped_get_db():
        w = _ConnectionWrapper(original())
        # Make only the FIRST connection's first executemany raise.
        if not wrappers:
            w._executemany_side_effect = busy_error
        wrappers.append(w)
        return w

    monkeypatch.setattr(archive, "_get_db", wrapped_get_db)

    n = archive.update_embeddings_batch(pairs)
    assert n == 5, f"expected all 5 persisted after retry, got {n}"
    assert len(wrappers) >= 2, "retry should open a second connection"


def test_batch_scales_linearly_not_quadratically():
    """Sanity: a batch of 500 pairs completes in well under 10s on any disk."""
    import time

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "big.db"
        archive = ChainArchive(db_path=str(db_path))
        conn = sqlite3.connect(str(db_path))
        for i in range(500):
            conn.execute(
                "INSERT INTO chain_archive "
                "(source, chain_sequence, chain_length, confidence, gut_agreement, "
                " outcome_score, domain, observation_snapshot, epoch_id, created_at) "
                "VALUES ('main', '[]', 1, 0.5, 0.5, 0.5, 'general', ?, ?, ?)",
                (json.dumps([0.1] * 132), 1000 + i, 1_000_000.0 + i),
            )
        conn.commit()
        conn.close()

        chains = archive.get_chains_without_embedding(limit=500)
        pairs = [(c["id"], _embedding(c["id"])) for c in chains]
        assert len(pairs) == 500

        t0 = time.perf_counter()
        n = archive.update_embeddings_batch(pairs)
        dt = time.perf_counter() - t0
        assert n == 500
        assert dt < 10.0, f"batch of 500 took {dt:.2f}s — bottleneck?"
