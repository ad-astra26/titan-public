"""SPEC §11.H.1.bis — consciousness.db row-vector BLOB f32-LE encoding.

Verifies the lean Phase 8 migration: writer packs f32 BLOB, reader dual-reads
BLOB (new) + TEXT-JSON (legacy/rollback), and a DB with MIXED rows reads
cleanly (the property that removes the lockstep fragility which rolled back
the 2026-05-26 attempt).
"""
import json
import struct

import pytest

from titan_hcl.logic.consciousness import (
    ConsciousnessDB,
    EpochRecord,
    pack_vector,
    unpack_vector,
)


def test_pack_unpack_roundtrip_lossless_f32():
    vec = [0.0, 1.0, 0.5, -0.25, 0.123456, 0.999999]
    blob = pack_vector(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == len(vec) * 4  # N×4 f32-LE
    out = unpack_vector(blob)
    assert isinstance(out, list)
    # f32 precision (atol 1e-6 per INV-CDB-3 verify spec)
    assert all(abs(a - b) < 1e-6 for a, b in zip(vec, out))


def test_pack_is_little_endian_f32():
    assert pack_vector([1.0]) == struct.pack("<f", 1.0)
    assert pack_vector([1.0, 2.0]) == struct.pack("<2f", 1.0, 2.0)


def test_unpack_dual_read_blob_and_text_and_edges():
    # BLOB (new)
    assert unpack_vector(pack_vector([0.1, 0.2, 0.3])) == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    # TEXT-JSON (legacy / rollback)
    assert unpack_vector(json.dumps([0.4, 0.5])) == [0.4, 0.5]
    # already-decoded list (defensive)
    assert unpack_vector([0.6, 0.7]) == [0.6, 0.7]
    # None / empty
    assert unpack_vector(None) == []


def test_unpack_rejects_corrupt_blob_length():
    with pytest.raises(ValueError):
        unpack_vector(b"\x00\x00\x00")  # 3 bytes — not a multiple of 4


def test_writer_reader_roundtrip_blob(tmp_path):
    db = ConsciousnessDB(str(tmp_path / "c.db"))
    rec = EpochRecord(
        epoch_id=1, timestamp=1.0, block_hash="h",
        state_vector=[i * 0.01 for i in range(130)],
        drift_vector=[0.1] * 9, trajectory_vector=[0.2] * 9,
        journey_point=(0.1, 0.2, 0.3), curvature=0.0, density=0.0,
        distillation="", anchored_tx="",
    )
    db.insert_epoch(rec)
    # raw column is BLOB bytes (not TEXT)
    raw = db._conn.execute(
        "SELECT state_vector FROM epochs WHERE epoch_id=1").fetchone()[0]
    assert isinstance(raw, (bytes, bytearray))
    assert len(raw) == 130 * 4
    # reads back lossless
    got = db.get_recent_epochs(1)[0]
    assert got.state_vector == pytest.approx(rec.state_vector, abs=1e-6)
    assert got.drift_vector == pytest.approx(rec.drift_vector, abs=1e-6)


def test_reader_handles_mixed_legacy_text_and_blob_rows(tmp_path):
    """The lockstep-killer: a DB with an old TEXT row AND a new BLOB row must
    read cleanly — so a pre-migration DB + the new writer coexist with no
    column rename / no INSERT failure (the 2026-05-26 root cause)."""
    db = ConsciousnessDB(str(tmp_path / "c.db"))
    # legacy TEXT row inserted directly (simulates pre-migration data)
    db._conn.execute(
        "INSERT INTO epochs (epoch_id,timestamp,block_hash,state_vector,"
        "drift_vector,trajectory_vector,journey_x,journey_y,journey_z,"
        "curvature,density,distillation,anchored_tx) "
        "VALUES (1,1.0,'',?,?,?,0,0,0,0,0,'','')",
        (json.dumps([0.5] * 130), json.dumps([0.1] * 9), json.dumps([0.2] * 9)),
    )
    db._conn.commit()
    # new BLOB row via the production writer
    db.insert_epoch(EpochRecord(
        epoch_id=2, timestamp=2.0, block_hash="",
        state_vector=[0.7] * 130, drift_vector=[0.3] * 9,
        trajectory_vector=[0.4] * 9, journey_point=(0, 0, 0),
        curvature=0.0, density=0.0, distillation="", anchored_tx="",
    ))
    rows = db.get_recent_epochs(2)
    by_id = {r.epoch_id: r for r in rows}
    assert by_id[1].state_vector == pytest.approx([0.5] * 130, abs=1e-6)  # TEXT legacy
    assert by_id[2].state_vector == pytest.approx([0.7] * 130, abs=1e-6)  # BLOB new
