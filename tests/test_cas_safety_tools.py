"""Tests for the read-only CAS safety net (Phase 0 / 0B): cas_audit + migration dry-run.

Run isolated:
    python -m pytest tests/test_cas_safety_tools.py -v -p no:anchorpy --tb=short
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from titan_hcl.logic.timechain import TimeChain
from titan_hcl.logic.timechain_v2 import BlockBuilder, Mempool, Transaction
from titan_hcl.synthesis.content_store import ContentStore

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cas_audit = _load("cas_audit")
migrate = _load("migrate_outer_memory_to_cas")
content_dedup = _load("content_dedup_audit")


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="cas_safety_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _payload(i: int) -> dict:
    return {
        "fork": "episodic", "source": "test_source", "significance": 0.5,
        "thought_type": "episodic", "tags": ["test", f"t:{i}"], "epoch_id": 1000 + i,
        "content": {"data": f"ep {i}"},
        "neuromods": {"DA": 0.5, "5HT": 0.8, "NE": 0.6, "GABA": 0.2, "ACh": 0.7},
        "chi_available": 0.5, "attention": 0.5, "i_confidence": 0.5,
        "chi_coherence": 0.3, "timestamp": 1_700_000_000.0 + i,
    }


def _seal(data_dir, *, slim, store=None, n=5):
    mp = Mempool(data_dir, {"aggregate_sources": []})
    for i in range(n):
        mp.submit(Transaction.from_commit_payload(_payload(i)))
    tc = TimeChain(data_dir, "T1")
    tc.create_genesis({"birth": "t"}, birth_timestamp=1_699_000_000.0)
    builder = BlockBuilder(tc, {"cas_payload_slimming_enabled": slim}, content_store=store)
    return builder.seal_fork(mp, "episodic", "time", 1010, None, "test")


# ── cas_audit ─────────────────────────────────────────────────────────

def test_audit_clean_when_slimmed_block_resolves(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    _seal(tmp_dir, slim=True, store=store)
    r = cas_audit.audit(Path(tmp_dir), Path(tmp_dir) / "cas")
    assert r["referenced"] == 1
    assert r["ok"] == 1
    assert r["missing"] == 0 and r["corrupt"] == 0
    assert r["clean"] is True


def test_audit_detects_missing_blob(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    block = _seal(tmp_dir, slim=True, store=store)
    h = block.payload.content["content_summaries_hash"]
    store._path_for(h).unlink()  # simulate a lost blob
    r = cas_audit.audit(Path(tmp_dir), Path(tmp_dir) / "cas")
    assert r["missing"] == 1
    assert r["clean"] is False


def test_audit_detects_corrupt_blob(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    block = _seal(tmp_dir, slim=True, store=store)
    h = block.payload.content["content_summaries_hash"]
    store._path_for(h).write_bytes(b"tampered")
    r = cas_audit.audit(Path(tmp_dir), Path(tmp_dir) / "cas")
    assert r["corrupt"] == 1
    assert r["clean"] is False


def test_audit_clean_on_inline_only_chain(tmp_dir):
    # Inline (unslimmed) chain references no CAS — audit is trivially clean.
    _seal(tmp_dir, slim=False)
    r = cas_audit.audit(Path(tmp_dir), Path(tmp_dir) / "cas")
    assert r["referenced"] == 0 and r["clean"] is True


def test_audit_reports_orphans(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    _seal(tmp_dir, slim=False)  # inline chain, no references
    store.put(b"an unreferenced blob")  # orphan
    r = cas_audit.audit(Path(tmp_dir), Path(tmp_dir) / "cas")
    assert r["orphans"] == 1
    assert r["clean"] is True  # orphans are not a fault


# ── migration dry-run ─────────────────────────────────────────────────

def test_dryrun_counts_inline_blocks_writes_nothing(tmp_dir):
    _seal(tmp_dir, slim=False, n=5)
    cas_before = Path(tmp_dir) / "content_blobs"
    r = migrate.plan(Path(tmp_dir))
    assert r["inline_blocks_total"] == 1
    assert r["total_inline_bytes"] > 0
    assert r["unique_blobs"] == 1
    # The dry-run must not have created a CAS.
    assert not cas_before.exists()


def test_dryrun_skips_already_slim_blocks(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    _seal(tmp_dir, slim=True, store=store)
    r = migrate.plan(Path(tmp_dir))
    assert r["already_slim_blocks"] == 1
    assert r["inline_blocks_total"] == 0


# ── content-dedup audit ───────────────────────────────────────────────

def test_content_dedup_audit_runs_and_reports_fields(tmp_dir):
    _seal(tmp_dir, slim=False, n=5)
    r = content_dedup.audit(Path(tmp_dir), min_field_bytes=64)
    # tx_summaries should appear as a large field; unique == occurrences (no dedup).
    fields = {row["field"]: row for row in r["rows"]}
    assert "tx_summaries" in fields
    assert fields["tx_summaries"]["dedupable_bytes"] == 0
    assert r["total_field_bytes"] > 0
