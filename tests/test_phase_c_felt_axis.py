"""Phase C — felt-at-lived-time plumbing tests (RFP_synthesis_engram_grounding §7.C / G3).

Covers the pure units of the felt chain: the felt-coverage math
(`(members_with_felt/total) × mean_magnitude`, magnitude = ‖levels − 0.5‖),
the sidecar `felt` round-trip + the idempotent backward-compat ALTER, and the
promotion-anchor determinism guarantee (neuromods is NOT a tx_hash input). The
chat→add_to_mempool→promotion→consolidation integration is verified live on T3.
"""
import json
import os
import sqlite3

import pytest

from titan_hcl.synthesis.consolidation import (
    TxCandidate,
    _felt_magnitude,
    _parse_felt,
    felt_coverage_from_members,
)
from titan_hcl.synthesis.promotion_anchor import build_promotion_tx
from titan_hcl.synthesis.thought_sidecar import (
    ThoughtSidecar,
    ThoughtSidecarReader,
)


def _tx(felt=None):
    return TxCandidate(tx_hash="h", fork="episodic", tags=(), embedding=None, felt=felt)


# ── felt magnitude (intensity = deviation from the 0.5 setpoint centre) ──

def test_felt_magnitude_empty_and_metadata_only_is_zero():
    assert _felt_magnitude({}) == 0.0
    assert _felt_magnitude({"emotion": "joy", "ts": 1.0,
                            "emotion_confidence": 0.9, "dream_cycle": 3}) == 0.0


def test_felt_magnitude_at_centre_is_zero():
    assert _felt_magnitude({"DA": 0.5, "5HT": 0.5, "NE": 0.5}) == 0.0


def test_felt_magnitude_extremes_is_one():
    assert _felt_magnitude(
        {"DA": 1.0, "5HT": 0.0, "NE": 1.0, "ACh": 0.0}) == pytest.approx(1.0)


def test_felt_magnitude_excludes_metadata_keys():
    bare = _felt_magnitude({"DA": 1.0})
    with_meta = _felt_magnitude(
        {"DA": 1.0, "emotion": "x", "ts": 9.0, "emotion_confidence": 0.5,
         "dream_cycle": 2})
    assert bare == with_meta == pytest.approx(1.0)


# ── parse felt (sidecar JSON string | dict | None) ──

def test_parse_felt_variants():
    assert _parse_felt(None) == {}
    assert _parse_felt('{"DA":0.7}') == {"DA": 0.7}
    assert _parse_felt({"DA": 0.7}) == {"DA": 0.7}
    assert _parse_felt("not-json") == {}
    assert _parse_felt("[1,2]") == {}  # valid json but not a dict


# ── felt_coverage = consistency × intensity ──

def test_felt_coverage_no_felt_is_zero():
    assert felt_coverage_from_members([_tx(), _tx()]) == 0.0
    assert felt_coverage_from_members([]) == 0.0


def test_felt_coverage_full_max_intensity_is_one():
    f = json.dumps({"DA": 1.0, "5HT": 0.0})
    assert felt_coverage_from_members([_tx(f), _tx(f)]) == pytest.approx(1.0)


def test_felt_coverage_partial_coverage_scales_by_fraction():
    f = json.dumps({"DA": 1.0, "5HT": 0.0})  # max intensity
    # 1 of 2 members felt-laden → 0.5 coverage × 1.0 intensity
    assert felt_coverage_from_members([_tx(f), _tx(None)]) == pytest.approx(0.5)


def test_felt_coverage_centre_felt_is_zero_intensity():
    # felt present but neutral (at centre) → magnitude 0 → not counted
    f = json.dumps({"DA": 0.5, "5HT": 0.5})
    assert felt_coverage_from_members([_tx(f), _tx(f)]) == 0.0


# ── promotion-anchor determinism: neuromods is NOT a tx_hash input ──

def test_promotion_neuromods_is_not_a_hash_input():
    base = {"id": 1, "user_prompt": "p", "agent_response": "r", "source_id": "s"}
    felt = dict(base, neuromod_context={"DA": 0.9, "NE": 0.8})
    p0, h0 = build_promotion_tx(base, now=1000.0)
    p1, h1 = build_promotion_tx(felt, now=1000.0)
    assert h0 == h1                       # same hash regardless of neuromods
    assert p0["neuromods"] == {}
    assert p1["neuromods"] == {"DA": 0.9, "NE": 0.8}


# ── sidecar felt round-trip + backward-compat ALTER ──

def test_sidecar_felt_roundtrip(tmp_path):
    sc = ThoughtSidecar(str(tmp_path))
    f = json.dumps({"DA": 0.8})
    sc.put(tx_hash="tx1", node_id=1, user_prompt="p", agent_response="r",
           memory_type="episodic", fork="episodic", ts=1.0, felt=f)
    sc.put(tx_hash="tx2", node_id=2, user_prompt="p2", agent_response="r2",
           memory_type="episodic", fork="episodic", ts=2.0, felt=None)
    sc.close()
    rd = ThoughtSidecarReader(str(tmp_path))
    assert rd.get("tx1")["felt"] == f
    assert rd.get("tx2")["felt"] is None
    by = {r["tx_hash"]: r for r in rd.iter_since(since_ts=0.0)}
    assert by["tx1"]["felt"] == f and by["tx2"]["felt"] is None
    rd.close()


def test_sidecar_backward_compat_alter_adds_felt(tmp_path):
    """A pre-Phase-B sidecar (no felt column) gains it via the idempotent ALTER on
    re-open; the old row reads felt=None (no data loss)."""
    p = os.path.join(str(tmp_path), "thought_sidecar.db")
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE thought_content (tx_hash TEXT PRIMARY KEY, node_id INTEGER,"
        " user_prompt TEXT, agent_response TEXT, memory_type TEXT, fork TEXT,"
        " ts DOUBLE)")
    c.execute("INSERT INTO thought_content VALUES "
              "('old',1,'p','r','episodic','episodic',1.0)")
    c.commit()
    c.close()
    ThoughtSidecar(str(tmp_path)).close()  # runs the idempotent ADD COLUMN felt
    rd = ThoughtSidecarReader(str(tmp_path))
    row = rd.get("old")
    assert row is not None and row["felt"] is None
    rd.close()
