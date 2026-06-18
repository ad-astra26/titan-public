"""Phase B — dream-time rollup. Proves §7.B for `titan_hcl/synthesis/presence_rollup.py`:

  • fold N raw person_interactions atoms → one presence_cycle_rollup row per person with
    correct first/last_seen_epoch, count, and the STRONGEST evidence (honesty gradient);
  • idempotent — folding twice yields identical rows (no double-count);
  • raw person_interactions atoms are UNTOUCHED (the rollup is a derived index, INV-PAM-NO-GAPS);
  • cycle partition — atoms below cycle_start_epoch are excluded (Titan-time key);
  • keyed by cycle_id (read from the Titan-time counter file), never wall-clock.

Real DuckDB conn + InlineWriter + a controlled cycle_state.json (so cycle_id/start_epoch
are deterministic).
"""
import json
import os

import duckdb
import pytest

from titan_hcl.synthesis.presence_rollup import PresenceRollup
from titan_hcl.synthesis.writer import InlineWriter


def _seed_atoms(conn, atoms):
    """atoms: list of (tx_hash, person_id, evidence_strength, age_epochs)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS person_interactions ("
        " tx_hash VARCHAR PRIMARY KEY, person_id VARCHAR NOT NULL, person_ref VARCHAR,"
        " evidence_strength VARCHAR NOT NULL, channel VARCHAR, age_epochs BIGINT NOT NULL,"
        " ts_utc DOUBLE)")
    for tx, pid, ev, ep in atoms:
        conn.execute(
            "INSERT INTO person_interactions VALUES (?,?,?,?,?,?,?)",
            [tx, pid, "", ev, "web", ep, float(ep)])


def _write_cycle(tmp_path, cycle_id, cycle_start_epoch):
    p = tmp_path / "cycle_state.json"
    p.write_text(json.dumps({"cycle_id": cycle_id, "cycle_start_epoch": cycle_start_epoch,
                             "armed": False, "last_latch_ts": 0.0}))


@pytest.fixture
def setup(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    writer = InlineWriter()
    pr = PresenceRollup(conn, writer, save_dir=str(tmp_path))
    assert pr.ensure_schema() is True
    return pr, conn, tmp_path


def _rollup(conn):
    return conn.execute(
        "SELECT cycle_id, person_id, first_seen_epoch, last_seen_epoch, count, "
        "evidence_strength FROM presence_cycle_rollup ORDER BY person_id").fetchall()


def test_fold_aggregates_and_strongest_evidence(setup):
    pr, conn, tmp = setup
    _write_cycle(tmp, cycle_id=7, cycle_start_epoch=1000)
    _seed_atoms(conn, [
        ("a", "maker", "asserted_identity", 1010),       # maker seen weak first…
        ("b", "maker", "crypto_verified_maker", 1020),    # …then crypto-verified (strongest wins)
        ("c", "maker", "asserted_identity", 1030),
        ("d", "alice", "asserted_identity", 1015),
    ])
    n = pr.fold()
    assert n == 2
    rows = _rollup(conn)
    # alice
    assert rows[0] == (7, "alice", 1015, 1015, 1, "asserted_identity")
    # maker: first=1010, last=1030, count=3, strongest=crypto_verified_maker
    assert rows[1] == (7, "maker", 1010, 1030, 3, "crypto_verified_maker")


def test_idempotent_refold(setup):
    pr, conn, tmp = setup
    _write_cycle(tmp, cycle_id=1, cycle_start_epoch=0)
    _seed_atoms(conn, [("a", "bob", "asserted_identity", 5),
                       ("b", "bob", "asserted_identity", 9)])
    pr.fold()
    pr.fold()   # second fold = recompute, no double-count
    rows = _rollup(conn)
    assert len(rows) == 1
    assert rows[0] == (1, "bob", 5, 9, 2, "asserted_identity")


def test_raw_atoms_untouched(setup):
    pr, conn, tmp = setup
    _write_cycle(tmp, cycle_id=2, cycle_start_epoch=0)
    _seed_atoms(conn, [("a", "x", "asserted_identity", 1),
                       ("b", "x", "asserted_identity", 2),
                       ("c", "y", "crypto_verified_device", 3)])
    pr.fold()
    raw = conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
    assert raw == 3                      # INV-PAM-NO-GAPS — rollup never drops atoms


def test_cycle_partition_excludes_prior_atoms(setup):
    pr, conn, tmp = setup
    # cycle 9 opened at epoch 2000 — atoms below it belong to earlier cycles
    _write_cycle(tmp, cycle_id=9, cycle_start_epoch=2000)
    _seed_atoms(conn, [
        ("old1", "carol", "asserted_identity", 1500),   # prior cycle — excluded
        ("new1", "carol", "asserted_identity", 2100),
        ("new2", "carol", "crypto_verified_device", 2200),
    ])
    pr.fold()
    rows = _rollup(conn)
    assert len(rows) == 1
    # only the in-cycle atoms count: first=2100, last=2200, count=2
    assert rows[0] == (9, "carol", 2100, 2200, 2, "crypto_verified_device")


def test_empty_cycle_folds_nothing(setup):
    pr, conn, tmp = setup
    _write_cycle(tmp, cycle_id=3, cycle_start_epoch=0)
    _seed_atoms(conn, [])                # table exists, no atoms
    assert pr.fold() == 0
    assert _rollup(conn) == []
