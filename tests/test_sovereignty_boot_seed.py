"""G9 (INV-Syn-25): SovereigntyRatioMeter boot-seed from conversation-fork TXs.

The meter's rolling-window marks are in-memory; a worker respawn zeros them
(crash-loop audit §5.3 "respawn zeros windows"). INV-Syn-25 makes metrics a
rebuildable projection over the Timechain, so the per-turn
`sovereignty{needed,satisfied}` signal now rides arch §7 conversation-fork TXs
and the worker boot-seeds the meter by replaying in-window conv TXs.

These pin that reseed against a fixture `block_index` (real SQL over the
conversation-fork resolved BY NAME per INV-Syn-26) + the single-source
`knowledge_moment_signal` helper. The binary chain read (`read_block_content_at`)
is stubbed — it has its own dedicated tests.
"""
import sqlite3

import titan_hcl.synthesis.chain_reader as chain_reader
from titan_hcl.synthesis.cited_use import (
    CitedUseDetector,
    SurfacedItem,
    knowledge_moment_signal,
)
from titan_hcl.synthesis.sovereignty_meter import (
    SovereigntyRatioMeter,
    boot_seed_from_conversation_chain,
)


def _build_conv_index(idx_path, rows):
    """rows: list of (block_hash, file_offset, timestamp, thought_type). All on
    the conversation fork (id 5). An episodic fork is also registered so the
    fork-name JOIN is exercised."""
    conn = sqlite3.connect(str(idx_path))
    conn.execute("CREATE TABLE fork_registry (fork_id INTEGER, fork_name TEXT, "
                 "fork_type TEXT)")
    conn.execute("CREATE TABLE block_index (block_hash TEXT, fork_id INTEGER, "
                 "block_height INTEGER, file_offset INTEGER, thought_type TEXT, "
                 "tags TEXT, timestamp REAL)")
    conn.execute("INSERT INTO fork_registry VALUES (5, 'conversation', 'conv')")
    conn.execute("INSERT INTO fork_registry VALUES (3, 'episodic', 'epi')")
    for (bh, off, ts, tt) in rows:
        conn.execute("INSERT INTO block_index VALUES (?, 5, ?, ?, ?, '[]', ?)",
                     (bh, off, off, tt, ts))
    conn.commit()
    conn.close()


def _fake_read(content_by_offset):
    def fake(data_dir, fork_id, file_offset):
        return content_by_offset.get(int(file_offset))
    return fake


# ── boot-seed ──────────────────────────────────────────────────────────────

def test_boot_seed_replays_needed_and_satisfied(tmp_path, monkeypatch):
    idx = tmp_path / "index.db"
    _build_conv_index(idx, [
        ("h0", 0, 1000.0, "conversation"),
        ("h1", 1, 1100.0, "conversation"),
        ("h2", 2, 1200.0, "conversation"),
    ])
    content = {
        0: {"agent_response": "a",
            "sovereignty": {"needed": True, "satisfied": True}},
        1: {"agent_response": "b",
            "sovereignty": {"needed": True, "satisfied": False}},
        2: {"agent_response": "c",
            "sovereignty": {"needed": False, "satisfied": False}},
    }
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read(content))

    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1300.0)
    summary = boot_seed_from_conversation_chain(
        meter, data_dir=str(tmp_path), since_ts=0.0, index_db_path=str(idx))

    assert summary["scanned"] == 3
    assert summary["knowledge_moments"] == 2   # h0, h1 (needed)
    assert summary["recall_satisfied"] == 1    # h0 (satisfied)
    assert summary["capped"] is False

    stats = meter.compute(1300.0)["all"]
    assert stats["knowledge_moments"] == 2
    assert stats["recall_satisfied"] == 1
    assert stats["cited_recalls"] == 1
    assert stats["ratio"] == 0.5               # the meter recomputed honestly


def test_boot_seed_respects_since_ts(tmp_path, monkeypatch):
    idx = tmp_path / "index.db"
    _build_conv_index(idx, [
        ("old", 0, 100.0, "conversation"),
        ("new", 1, 9000.0, "conversation"),
    ])
    content = {
        0: {"sovereignty": {"needed": True, "satisfied": True}},
        1: {"sovereignty": {"needed": True, "satisfied": True}},
    }
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read(content))
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 9100.0)
    summary = boot_seed_from_conversation_chain(
        meter, data_dir=str(tmp_path), since_ts=5000.0, index_db_path=str(idx))
    assert summary["scanned"] == 1             # only 'new' (ts > 5000)
    assert summary["knowledge_moments"] == 1


def test_boot_seed_skips_blocks_without_sovereignty(tmp_path, monkeypatch):
    idx = tmp_path / "index.db"
    _build_conv_index(idx, [
        ("h0", 0, 1000.0, "conversation"),
        ("h1", 1, 1100.0, "conversation"),   # pre-G9 TX / batch envelope
    ])
    content = {
        0: {"sovereignty": {"needed": True, "satisfied": True}},
        1: {"v2": True, "tx_summaries": [{"hash": "x", "type": "conversation"}]},
    }
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read(content))
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1200.0)
    summary = boot_seed_from_conversation_chain(
        meter, data_dir=str(tmp_path), since_ts=0.0, index_db_path=str(idx))
    assert summary["scanned"] == 2
    assert summary["knowledge_moments"] == 1   # h1 lacks sovereignty → skipped
    assert summary["recall_satisfied"] == 1


def test_boot_seed_cap_flag(tmp_path, monkeypatch):
    idx = tmp_path / "index.db"
    rows = [(f"h{i}", i, 1000.0 + i, "conversation") for i in range(5)]
    _build_conv_index(idx, rows)
    content = {i: {"sovereignty": {"needed": True, "satisfied": False}}
               for i in range(5)}
    monkeypatch.setattr(chain_reader, "read_block_content_at",
                        _fake_read(content))
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 2000.0)
    summary = boot_seed_from_conversation_chain(
        meter, data_dir=str(tmp_path), since_ts=0.0, cap=3,
        index_db_path=str(idx))
    assert summary["capped"] is True           # surfaced, not silent
    assert summary["scanned"] == 3
    assert summary["knowledge_moments"] == 3


def test_boot_seed_missing_index_returns_empty(tmp_path):
    meter = SovereigntyRatioMeter(windows=["all"], clock=lambda: 1.0)
    summary = boot_seed_from_conversation_chain(
        meter, data_dir=str(tmp_path), since_ts=0.0,
        index_db_path=str(tmp_path / "nope.db"))
    assert summary["scanned"] == 0
    assert summary["knowledge_moments"] == 0


# ── knowledge_moment_signal (single source: live emit AND conv-TX field) ────

def test_knowledge_moment_signal_needed_and_satisfied():
    det = CitedUseDetector()
    items = [SurfacedItem(item_id="tx:1", title="solana minting guide",
                          content_snippet="how to mint a solana nft")]
    needed, satisfied, cited = knowledge_moment_signal(
        det, items, "Here is how solana minting works.")
    assert needed is True
    assert satisfied is True
    assert cited == ["tx:1"]


def test_knowledge_moment_signal_needed_not_satisfied():
    det = CitedUseDetector()
    items = [SurfacedItem(item_id="tx:1", title="quantum chromodynamics",
                          content_snippet="gluon color confinement")]
    needed, satisfied, cited = knowledge_moment_signal(
        det, items, "I like cats and dogs.")
    assert needed is True
    assert satisfied is False
    assert cited == []


def test_knowledge_moment_signal_not_needed_when_empty():
    det = CitedUseDetector()
    needed, satisfied, cited = knowledge_moment_signal(det, [], "anything")
    assert needed is False
    assert satisfied is False
    assert cited == []
