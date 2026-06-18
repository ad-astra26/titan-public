"""Phase D — presence recall. Proves §7.D for `AutobiographySeal.export_recall_snapshot`
+ `logic/presence_recall.py`:

  • the recall snapshot picks each person's MOST-RECENT presence (across cycles) +
    the chain_status of the cycle that holds it (CHAINED › WIRED › UNSEALED);
  • recall computes the gap in EPOCHS; human time appears ONLY in `gap_human`
    (the translator runs at the narration edge — INV-PAM-TITAN-TIME / G5);
  • evidence_strength + chain_status are carried end-to-end (honesty gradient);
  • a person with no anchored prior presence → None (honest non-recognition);
  • full round-trip: seal → export → recall sees WIRED, then CHAINED after the
    fork-main block seals.

Real DuckDB + InlineWriter + the real OuterMemoryWriter/AutobiographySeal; a fake
translator so the gap-translation is asserted deterministically.
"""
import hashlib
import json
import os

import duckdb
import pytest

from titan_hcl import bus
from titan_hcl.logic.presence_recall import PresenceRecall, SNAPSHOT_NAME
from titan_hcl.synthesis.autobiography_seal import AutobiographySeal
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.presence_rollup import PresenceRollup
from titan_hcl.synthesis.writer import InlineWriter


class _CapturingQueue:
    def __init__(self):
        self.msgs = []

    def put(self, msg):
        self.msgs.append(msg)


class _FakeTranslator:
    """Deterministic translation so the gap-epochs→human edge is assertable."""

    def to_human(self, gap_epochs):
        return f"~{gap_epochs}-epochs-ago"


def _hx(seed):
    return hashlib.sha256(seed.encode()).hexdigest()


def _seed_atoms(conn, atoms):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS person_interactions ("
        " tx_hash VARCHAR PRIMARY KEY, person_id VARCHAR NOT NULL, person_ref VARCHAR,"
        " evidence_strength VARCHAR NOT NULL, channel VARCHAR, age_epochs BIGINT NOT NULL,"
        " ts_utc DOUBLE)")
    for tx, pid, ev, ep in atoms:
        conn.execute("INSERT INTO person_interactions VALUES (?,?,?,?,?,?,?)",
                     [tx, pid, "", ev, "web", ep, float(ep)])


@pytest.fixture
def setup(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    writer = InlineWriter()
    rollup = PresenceRollup(conn, writer, save_dir=str(tmp_path))
    assert rollup.ensure_schema() is True
    omw = OuterMemoryWriter(send_queue=_CapturingQueue(), src="autobiography_seal")
    seal = AutobiographySeal(conn, writer, rollup, omw, save_dir=str(tmp_path))
    snap_path = str(tmp_path / SNAPSHOT_NAME)
    recall = PresenceRecall(snapshot_path=snap_path, translator=_FakeTranslator())
    return seal, recall, conn, tmp_path, snap_path


def test_export_picks_latest_presence_and_chain_status(setup):
    seal, recall, conn, tmp, snap_path = setup
    # maker seen in cycle 1 (older) and cycle 2 (newer); alice only in cycle 1
    _seed_atoms(conn, [
        (_hx("m1"), "maker", "crypto_verified_maker", 100),
        (_hx("a1"), "alice", "asserted_identity", 110),
        (_hx("m2"), "maker", "crypto_verified_maker", 250),
    ])
    seal.seal_closed_cycle(1, 0, 200)      # captures maker@100, alice@110
    seal.seal_closed_cycle(2, 200, 300)    # captures maker@250
    n = seal.export_recall_snapshot()
    assert n == 2
    persons = json.load(open(snap_path))["persons"]
    # maker's LATEST presence is cycle 2 @ 250 (not cycle 1 @ 100)
    assert persons["maker"]["last_seen_epoch"] == 250
    assert persons["maker"]["cycle_id"] == 2
    assert persons["maker"]["chain_status"] == "WIRED"      # sealed, block pending
    assert persons["alice"]["last_seen_epoch"] == 110
    assert persons["alice"]["cycle_id"] == 1


def test_recall_gap_in_epochs_human_only_in_string(setup):
    seal, recall, conn, tmp, snap_path = setup
    _seed_atoms(conn, [(_hx("m"), "maker", "crypto_verified_maker", 1000)])
    seal.seal_closed_cycle(5, 0, 2000)
    seal.export_recall_snapshot()
    # Maker returns at age 1500 → gap = 500 epochs (Titan-time)
    rec = recall.recall("maker", now_age_epochs=1500)
    assert rec is not None
    assert rec["gap_epochs"] == 500                          # epochs, INV-PAM-TITAN-TIME
    assert rec["gap_human"] == "~500-epochs-ago"             # human ONLY here (edge)
    assert rec["last_seen_epoch"] == 1000
    assert rec["evidence_strength"] == "crypto_verified_maker"


def test_recall_absent_person_is_none(setup):
    seal, recall, conn, tmp, snap_path = setup
    _seed_atoms(conn, [(_hx("m"), "maker", "crypto_verified_maker", 100)])
    seal.seal_closed_cycle(1, 0, 200)
    seal.export_recall_snapshot()
    assert recall.recall("stranger", now_age_epochs=300) is None   # honest non-recognition
    # no snapshot file at all → also None (soft)
    recall2 = PresenceRecall(snapshot_path=str(tmp / "nope.json"),
                             translator=_FakeTranslator())
    assert recall2.recall("maker", now_age_epochs=300) is None


def test_chain_status_flips_chained_after_block_seal(setup):
    seal, recall, conn, tmp, snap_path = setup
    _seed_atoms(conn, [(_hx("m"), "maker", "crypto_verified_maker", 100)])
    entry = seal.seal_closed_cycle(3, 0, 200)
    seal.export_recall_snapshot()
    assert recall.recall("maker", now_age_epochs=300)["chain_status"] == "WIRED"
    # the fork-main block seals AFTER our emit → CHAINED + anchored to the block
    seal.note_fork_main_sealed(99, "blockX", entry["emit_ts"] + 1.0)
    seal.export_recall_snapshot()
    rec = recall.recall("maker", now_age_epochs=300)
    assert rec["chain_status"] == "CHAINED"
    assert rec["anchor"] == "blockX"                         # block_hash is the anchor


def test_empty_rollup_exports_no_persons(setup):
    seal, recall, conn, tmp, snap_path = setup
    _seed_atoms(conn, [])                       # table exists, no atoms
    seal.seal_closed_cycle(1, 0, 200)           # empty cycle still seals
    assert seal.export_recall_snapshot() == 0   # no persons → empty surface
    assert recall.recall("maker", now_age_epochs=300) is None
