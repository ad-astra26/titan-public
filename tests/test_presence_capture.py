"""Phase A — presence capture. Proves the §7.A.5 DONE-contract for
`titan_hcl/synthesis/presence_capture.py`:

  • a verified Maker turn → a real `person_interactions` row with the correct
    evidence_strength (crypto_verified_maker) + a real anchoring tx_hash, AND an
    episodic-fork TIMECHAIN_COMMIT is emitted;
  • a non-verified visitor turn → a row with `asserted_identity`;
  • an invalid evidence string is downgraded honestly (never upgraded);
  • a multi-interaction soak drops ZERO rows (INV-PAM-NO-GAPS);
  • a byte-identical re-record is idempotent (PK collision → no double row).

Real DuckDB conn + InlineWriter (inline) + the REAL OuterMemoryWriter (fake queue),
so the tx_hash + row are genuinely produced (no mock of the hash path).
"""
import duckdb
import pytest

from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter
from titan_hcl.synthesis.presence_capture import PresenceCapture
from titan_hcl.synthesis.writer import InlineWriter


class _FakeQueue:
    """Captures emitted TIMECHAIN_COMMIT messages (OuterMemoryWriter.emit → .put)."""

    def __init__(self):
        self.msgs = []

    def put(self, m):
        self.msgs.append(m)


class _FakeAgeReader:
    def __init__(self, epochs=812300):
        self.epochs = epochs

    def get_age_epochs(self):
        return self.epochs


@pytest.fixture
def cap(tmp_path):
    conn = duckdb.connect(str(tmp_path / "synthesis.duckdb"))
    writer = InlineWriter()
    queue = _FakeQueue()
    omw = OuterMemoryWriter(queue, src="test_presence")
    pc = PresenceCapture(conn, writer, omw_writer=omw, age_reader=_FakeAgeReader(812300))
    assert pc.ensure_schema() is True
    return pc, conn, queue


def _rows(conn):
    return conn.execute(
        "SELECT tx_hash, person_id, person_ref, evidence_strength, channel, "
        "age_epochs, ts_utc FROM person_interactions ORDER BY ts_utc").fetchall()


def test_verified_maker_row_and_tx(cap):
    pc, conn, queue = cap
    tx = pc.record(person_id="maker", evidence_strength="crypto_verified_maker",
                   channel="web", person_ref="Bsg2sw", ts=1000.0)
    assert tx and isinstance(tx, str) and len(tx) == 64   # sha256 hex
    rows = _rows(conn)
    assert len(rows) == 1
    r = rows[0]
    assert r[0] == tx                       # tx_hash on the row == anchored TX
    assert r[1] == "maker"
    assert r[2] == "Bsg2sw"
    assert r[3] == "crypto_verified_maker"  # honesty gradient recorded
    assert r[4] == "web"
    assert r[5] == 812300                   # age_epochs is the Titan-time key
    assert r[6] == 1000.0

    # an episodic-fork TIMECHAIN_COMMIT was emitted for the same atom
    assert len(queue.msgs) == 1
    p = queue.msgs[0]["payload"]
    assert p["fork"] == "episodic"
    assert p["thought_type"] == "presence_interaction"
    assert p["content"]["evidence_strength"] == "crypto_verified_maker"


def test_nonverified_visitor_is_asserted(cap):
    pc, conn, _ = cap
    tx = pc.record(person_id="visitor_42", evidence_strength="asserted_identity",
                   channel="web", ts=2000.0)
    assert tx
    r = _rows(conn)[0]
    assert r[1] == "visitor_42"
    assert r[3] == "asserted_identity"      # never over-claimed


def test_invalid_evidence_downgrades(cap):
    pc, conn, _ = cap
    tx = pc.record(person_id="x", evidence_strength="totally_sure_its_them",
                   channel="app", ts=3000.0)
    assert tx
    assert _rows(conn)[0][3] == "asserted_identity"   # downgraded, never upgraded


def test_no_gaps_over_soak(cap):
    pc, conn, _ = cap
    n = 50
    seen = set()
    for i in range(n):
        tx = pc.record(person_id=f"p{i % 5}", evidence_strength="asserted_identity",
                       channel="web", ts=10_000.0 + i)   # distinct ts → distinct atom
        assert tx
        seen.add(tx)
    assert len(seen) == n                    # n distinct tx_hashes
    cnt = conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
    assert cnt == n                          # ZERO dropped rows (INV-PAM-NO-GAPS)


def test_idempotent_rerecord(cap):
    pc, conn, _ = cap
    a = pc.record(person_id="maker", evidence_strength="crypto_verified_maker",
                  channel="web", person_ref="Bsg2sw", ts=5000.0)
    b = pc.record(person_id="maker", evidence_strength="crypto_verified_maker",
                  channel="web", person_ref="Bsg2sw", ts=5000.0)   # byte-identical
    assert a == b                            # same content → same content-hash
    cnt = conn.execute("SELECT COUNT(*) FROM person_interactions").fetchone()[0]
    assert cnt == 1                          # PK collision → no double row
