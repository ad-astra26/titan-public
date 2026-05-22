"""Replay-equivalence + integrity gate for CAS payload slimming (Phase 0 / 0B, D-SPEC-102).

Proves the two non-negotiables agreed with the Maker:
  (a) a batch block sealed via the CAS path validates through the UNCHANGED
      `TimeChain.verify_fork` (chain mechanics untouched), and
  (b) reconstructing tx_summaries from the CAS yields content byte-identical to
      the old inline path.

Run isolated:
    python -m pytest tests/test_timechain_v2_cas_slimming.py -v -p no:anchorpy --tb=short
"""
import shutil
import tempfile

import pytest

from titan_hcl.logic.timechain import TimeChain
from titan_hcl.logic.timechain_v2 import (
    BlockBuilder,
    FORK_IDS,
    Mempool,
    Transaction,
    resolve_batch_summaries,
)
from titan_hcl.synthesis.content_store import ContentStore


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="tc_v2_cas_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _payload(i: int) -> dict:
    # Fixed timestamp/content per index → deterministic TX hashes, so the inline
    # and CAS runs build identical tx_summaries.
    return {
        "fork": "episodic",
        "source": "test_source",
        "significance": 0.5,
        "thought_type": "episodic",
        "tags": ["test", f"turn:{i}"],
        "epoch_id": 1000 + i,
        "content": {"data": f"episode {i}", "n": i},
        "neuromods": {"DA": 0.5, "5HT": 0.8, "NE": 0.6, "GABA": 0.2, "ACh": 0.7},
        "chi_available": 0.5,
        "attention": 0.5,
        "i_confidence": 0.5,
        "chi_coherence": 0.3,
        "timestamp": 1_700_000_000.0 + i,
    }


def _populate(data_dir: str, n: int) -> Mempool:
    mp = Mempool(data_dir, {"aggregate_sources": []})
    for i in range(n):
        mp.submit(Transaction.from_commit_payload(_payload(i)))
    return mp


def _seal(data_dir: str, mempool: Mempool, *, slim: bool, store=None):
    tc = TimeChain(data_dir, "T1")
    # Genesis registers the primary forks (declarative/procedural/episodic/...).
    tc.create_genesis({"birth": "test", "maker": "test"}, birth_timestamp=1_699_000_000.0)
    cfg = {"cas_payload_slimming_enabled": slim}
    builder = BlockBuilder(tc, cfg, content_store=store)
    block = builder.seal_fork(
        mempool, "episodic", trigger="time", current_epoch=1010,
        send_queue=None, worker_name="test",
    )
    return tc, block


def test_inline_path_unchanged(tmp_dir):
    tc, block = _seal(tmp_dir, _populate(tmp_dir, 5), slim=False)
    assert block is not None
    content = block.payload.content
    assert "tx_summaries" in content
    assert "content_summaries_hash" not in content
    assert len(content["tx_summaries"]) == 5
    ok, msg = tc.verify_fork(FORK_IDS["episodic"])
    assert ok, msg


def test_cas_path_validates_and_reconstructs(tmp_dir):
    # Inline reference run (separate tc + mempool with identical TXs).
    dir_inline = tempfile.mkdtemp(prefix="tc_inline_")
    try:
        _, inline_block = _seal(dir_inline, _populate(dir_inline, 5), slim=False)
        inline_summaries = inline_block.payload.content["tx_summaries"]
    finally:
        shutil.rmtree(dir_inline, ignore_errors=True)

    # CAS run.
    store = ContentStore(root=tmp_dir + "/cas")
    tc, cas_block = _seal(tmp_dir, _populate(tmp_dir, 5), slim=True, store=store)
    content = cas_block.payload.content

    # Shape: hash ref present, inline list gone.
    assert "content_summaries_hash" in content
    assert "tx_summaries" not in content
    assert content["tx_count"] == 5

    # (a) Integrity through the UNCHANGED verify_fork.
    ok, msg = tc.verify_fork(FORK_IDS["episodic"])
    assert ok, msg

    # (b) Byte/structure-identical reconstruction from the CAS.
    reconstructed = resolve_batch_summaries(content, store)
    assert reconstructed == inline_summaries

    # The blob is actually in the store and self-verifies.
    assert store.exists(content["content_summaries_hash"])


def test_resolver_handles_both_shapes(tmp_dir):
    store = ContentStore(root=tmp_dir + "/cas")
    # Old/inline shape → returned as-is, no store needed.
    inline = {"tx_summaries": [{"hash": "ab", "type": "episodic"}]}
    assert resolve_batch_summaries(inline) == inline["tx_summaries"]
    # New/CAS shape → fetched + decoded from the store.
    import msgpack
    summaries = [{"hash": "cd", "type": "episodic", "sig": 0.5}]
    h = store.put(msgpack.packb(summaries, use_bin_type=True))
    assert resolve_batch_summaries({"content_summaries_hash": h}, store) == summaries
    # Neither shape → empty.
    assert resolve_batch_summaries({"tx_count": 0}) == []


def test_slimmed_payload_is_smaller(tmp_dir):
    """The whole point: the on-chain payload shrinks when summaries move to CAS."""
    dir_inline = tempfile.mkdtemp(prefix="tc_inline2_")
    try:
        _, inline_block = _seal(dir_inline, _populate(dir_inline, 50), slim=False)
        inline_bytes = len(inline_block.payload.to_bytes())
    finally:
        shutil.rmtree(dir_inline, ignore_errors=True)

    store = ContentStore(root=tmp_dir + "/cas")
    _, cas_block = _seal(tmp_dir, _populate(tmp_dir, 50), slim=True, store=store)
    cas_bytes = len(cas_block.payload.to_bytes())

    assert cas_bytes < inline_bytes
