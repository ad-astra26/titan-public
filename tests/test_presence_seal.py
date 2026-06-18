"""Phase C — circadian Merkle-seal to fork 0. Proves §7.C for
`titan_hcl/synthesis/autobiography_seal.py` + `OuterMemoryWriter.write_presence_seal`
+ `PresenceRollup.fold_closed_cycle`:

  • the closed cycle is final-folded (end-bounded) + Merkle-rooted over its interaction
    tx_hashes, and ONE `presence_seal` fork-main TX is emitted with the right content;
  • the seal ledger records the cycle WIRED (idempotent: a re-seal is a no-op);
  • an EMPTY cycle still seals (merkle = SHA-256(b""), empty rollups) — INV-PAM-NO-GAPS;
  • end-bound partitions: atoms at/after end_epoch are excluded from merkle + rollups;
  • CHAINED flips on a fork=main TIMECHAIN_SEALED with ts ≥ emit_ts (not before).

Real DuckDB conn + InlineWriter + the real OuterMemoryWriter wired to a capturing
queue (so the genuine TIMECHAIN_COMMIT emit path is exercised, not a mock).
"""
import hashlib

import duckdb
import pytest

from titan_hcl import bus
from titan_hcl.synthesis.autobiography_seal import AutobiographySeal
from titan_hcl.synthesis.merkle import merkle_root_hex
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.presence_rollup import PresenceRollup
from titan_hcl.synthesis.writer import InlineWriter


class _CapturingQueue:
    """Stands in for the worker send_queue — captures emitted bus messages."""

    def __init__(self):
        self.msgs = []

    def put(self, msg):
        self.msgs.append(msg)

    def seals(self):
        return [m for m in self.msgs
                if m.get("type") == bus.TIMECHAIN_COMMIT
                and (m.get("payload") or {}).get("thought_type") == "presence_seal"]


def _hx(seed: str) -> str:
    """A valid 64-char hex tx_hash (merkle_root_hex parses leaves as hex)."""
    return hashlib.sha256(seed.encode()).hexdigest()


def _seed_atoms(conn, atoms):
    """atoms: list of (tx_hash, person_id, evidence_strength, age_epochs)."""
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
    q = _CapturingQueue()
    omw = OuterMemoryWriter(send_queue=q, src="autobiography_seal")
    seal = AutobiographySeal(conn, writer, rollup, omw, save_dir=str(tmp_path))
    return seal, rollup, conn, q, tmp_path


def test_seal_folds_merkles_emits(setup):
    seal, rollup, conn, q, _ = setup
    a, b, c, d = _hx("a"), _hx("b"), _hx("c"), _hx("d")
    _seed_atoms(conn, [
        (a, "maker", "asserted_identity", 1010),
        (b, "maker", "crypto_verified_maker", 1020),
        (c, "maker", "asserted_identity", 1030),
        (d, "alice", "asserted_identity", 1015),
    ])
    entry = seal.seal_closed_cycle(7, 1000, 1100)
    assert entry is not None
    # the closed cycle was final-folded into the rollup (strongest evidence wins)
    rows = conn.execute(
        "SELECT cycle_id, person_id, first_seen_epoch, last_seen_epoch, count, "
        "evidence_strength FROM presence_cycle_rollup ORDER BY person_id").fetchall()
    assert rows[0] == (7, "alice", 1015, 1015, 1, "asserted_identity")
    assert rows[1] == (7, "maker", 1010, 1030, 3, "crypto_verified_maker")
    # exactly ONE presence_seal fork-main TX emitted, with the right content
    seals = q.seals()
    assert len(seals) == 1
    payload = seals[0]["payload"]
    assert payload["fork"] == "main"
    content = payload["content"]
    assert content["cycle_id"] == 7
    assert content["age_epoch_range"] == [1000, 1100]
    assert content["interaction_count"] == 4
    # merkle root = root over the cycle's tx_hashes (ORDER BY tx_hash)
    assert content["merkle_root"] == merkle_root_hex(sorted([a, b, c, d]))
    pids = {r["person_id"] for r in content["person_rollups"]}
    assert pids == {"maker", "alice"}
    # ledger: WIRED, anchored content-hash recorded
    assert entry["chain_status"] == "WIRED"
    assert seal.chain_status(7) == "WIRED"


def test_empty_cycle_still_seals(setup):
    seal, rollup, conn, q, _ = setup
    _seed_atoms(conn, [])  # table exists, no atoms in the cycle
    entry = seal.seal_closed_cycle(3, 0, 500)
    assert entry is not None                       # a "met no one" cycle STILL seals
    seals = q.seals()
    assert len(seals) == 1
    content = seals[0]["payload"]["content"]
    assert content["person_rollups"] == []
    assert content["interaction_count"] == 0
    # empty set → deterministic SHA-256(b"") root (proves "the set was empty")
    assert content["merkle_root"] == hashlib.sha256(b"").hexdigest()
    assert seal.chain_status(3) == "WIRED"


def test_idempotent_reseal_is_noop(setup):
    seal, rollup, conn, q, _ = setup
    _seed_atoms(conn, [(_hx("x"), "bob", "asserted_identity", 10)])
    first = seal.seal_closed_cycle(5, 0, 100)
    assert first is not None
    assert len(q.seals()) == 1
    again = seal.seal_closed_cycle(5, 0, 100)       # same cycle
    assert again is None                            # INV-PAM-SEAL-IDEMPOTENT
    assert len(q.seals()) == 1                      # no second emit


def test_end_bound_partitions_the_cycle(setup):
    seal, rollup, conn, q, _ = setup
    in1, in2, after = _hx("in1"), _hx("in2"), _hx("after")
    _seed_atoms(conn, [
        (in1, "carol", "asserted_identity", 2100),
        (in2, "carol", "crypto_verified_device", 2200),
        (after, "carol", "crypto_verified_maker", 2500),   # at/after end → excluded
    ])
    seal.seal_closed_cycle(9, 2000, 2300)
    content = q.seals()[0]["payload"]["content"]
    assert content["interaction_count"] == 2                # `after` excluded
    assert content["merkle_root"] == merkle_root_hex(sorted([in1, in2]))
    roll = content["person_rollups"][0]
    assert roll["last_seen_epoch"] == 2200                  # not 2500
    assert roll["evidence_strength"] == "crypto_verified_device"


def test_chained_on_fork_main_seal(setup):
    seal, rollup, conn, q, _ = setup
    _seed_atoms(conn, [(_hx("p"), "dave", "asserted_identity", 50)])
    entry = seal.seal_closed_cycle(2, 0, 100)
    emit_ts = entry["emit_ts"]
    # a fork-main block sealed BEFORE our emit → does NOT chain us
    assert seal.note_fork_main_sealed(41, "blockA", emit_ts - 10.0) == 0
    assert seal.chain_status(2) == "WIRED"
    # a fork-main block sealed AT/AFTER our emit → chains us (+ records the block)
    assert seal.note_fork_main_sealed(42, "blockB", emit_ts + 5.0) == 1
    assert seal.chain_status(2) == "CHAINED"
    # idempotent: a later seal does not re-upgrade an already-CHAINED entry
    assert seal.note_fork_main_sealed(43, "blockC", emit_ts + 50.0) == 0
