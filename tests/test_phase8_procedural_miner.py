"""Phase 8 — ProceduralMiner unit tests (D-SPEC-PHASE8).

Covers:
- canonicalize_call: same tool_id + same arg-shape collapse; value-agnostic
- compute_skill_id: deterministic + kind-discriminating
- group_by_parent: parent_chat_tx preferred, parent_goal fallback, ts-ordered
- cluster_sequences: sliding windows; len boundaries respected
- filter_recurrent: keeps clusters ≥min_occurrences; sorted longest+highest first
- split_success_failure: positive iff success AND scored_by ∈ {oracle, llm}
                        AND no failure-tag/exception
- abstract_cluster: returns None on missing nl_desc; collects unique compiled_from hashes
- mine_pass: end-to-end with mocked tool_call_reader + llm_proposer
- mine_pass respects max_skills_per_pass cap
- mine_pass idempotency: same window re-mined → same skill_ids (no duplicates)
- mine_pass emits META_SKILL_COMPILED per skill; anchors summary
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from titan_hcl.synthesis.procedural_miner import (
    DEFAULT_MAX_SKILLS_PER_PASS,
    ProceduralMiner,
    canonicalize_call,
    compute_skill_id,
)
from titan_hcl.synthesis.skill_store import ProceduralSkillStore


# ── Helpers ────────────────────────────────────────────────────────────


def _make_tx(*, tool_id: str, args: dict, success: bool, scored_by: str | None,
             parent_chat_tx: str = "chat_a", ts: float = 1.0,
             tx_hash: str | None = None, tags: list | None = None,
             exception: str | None = None) -> dict:
    content = {
        "tool_id": tool_id, "args": args, "success": success,
        "scored_by": scored_by, "parent_chat_tx": parent_chat_tx, "ts": ts,
    }
    if exception:
        content["exception"] = exception
    return {
        "tx_hash": tx_hash or f"tx_{tool_id}_{ts}",
        "content": content,
        "tags": tags or [],
    }


@pytest.fixture()
def store(tmp_path):
    conn = duckdb.connect(":memory:")
    return ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=None,
    )


def _identity_proposer(cluster_meta, kind):
    """Test proposer: deterministic abstraction so the miner is exercise-able."""
    seq = cluster_meta["sequence"]
    nl = f"{kind}: {len(seq)}-step sequence " + " → ".join(s[0] for s in seq)
    return {
        "nl_description": nl,
        "executable_spec": {"steps": [{"tool": t, "args_shape": h} for t, h in seq]},
        "preconditions": ["pre"],
        "postconditions": ["post"],
    }


# ── canonicalize_call ──────────────────────────────────────────────────


def test_canonicalize_same_shape_collapses():
    a = _make_tx(tool_id="read_buffer", args={"name": "goal"}, success=True, scored_by="oracle")
    b = _make_tx(tool_id="read_buffer", args={"name": "retrieval"}, success=True, scored_by="oracle")
    assert canonicalize_call(a) == canonicalize_call(b)


def test_canonicalize_different_tool_differs():
    a = _make_tx(tool_id="read_buffer", args={"name": "goal"}, success=True, scored_by="oracle")
    b = _make_tx(tool_id="write_buffer", args={"name": "goal"}, success=True, scored_by="oracle")
    assert canonicalize_call(a) != canonicalize_call(b)


def test_canonicalize_value_agnostic_type_aware():
    """Same arg keys + same value types = same hash; different types = different."""
    a = _make_tx(tool_id="t", args={"x": 1, "y": "s"}, success=True, scored_by="oracle")
    b = _make_tx(tool_id="t", args={"x": 99, "y": "z"}, success=True, scored_by="oracle")
    c = _make_tx(tool_id="t", args={"x": "1", "y": "s"}, success=True, scored_by="oracle")  # x type changed
    assert canonicalize_call(a) == canonicalize_call(b)
    assert canonicalize_call(a) != canonicalize_call(c)


def test_canonicalize_handles_malformed_input():
    assert canonicalize_call({}) == ("unknown", canonicalize_call({})[1])


# ── compute_skill_id ────────────────────────────────────────────────────


def test_skill_id_deterministic():
    seq = (("a", "deadbeef"), ("b", "cafebabe"))
    assert compute_skill_id(seq, "positive") == compute_skill_id(seq, "positive")


def test_skill_id_kind_discriminates():
    seq = (("a", "deadbeef"),)
    assert compute_skill_id(seq, "positive") != compute_skill_id(seq, "negative")


def test_skill_id_prefix():
    assert compute_skill_id((("a", "1"),), "positive").startswith("skill_")


# ── group_by_parent ─────────────────────────────────────────────────────


def test_group_by_parent_chat_tx_preferred(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    txs = [
        _make_tx(tool_id="a", args={}, success=True, scored_by="oracle", parent_chat_tx="C1", ts=2.0),
        _make_tx(tool_id="b", args={}, success=True, scored_by="oracle", parent_chat_tx="C1", ts=1.0),
        _make_tx(tool_id="c", args={}, success=True, scored_by="oracle", parent_chat_tx="C2", ts=1.5),
    ]
    grouped = miner.group_by_parent(txs)
    assert set(grouped.keys()) == {"C1", "C2"}
    # ts-sorted within group
    assert [t["content"]["tool_id"] for t in grouped["C1"]] == ["b", "a"]


def test_group_by_parent_goal_fallback(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    tx = {
        "tx_hash": "x", "content": {"tool_id": "t", "args": {}, "success": True,
                                    "scored_by": "oracle", "parent_goal": "G1", "ts": 1.0},
    }
    grouped = miner.group_by_parent([tx])
    assert "G1" in grouped


# ── cluster_sequences ──────────────────────────────────────────────────


def test_cluster_sequences_respects_min_max_len(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_seq_len=2, max_seq_len=3,
    )
    groups = {
        "C1": [
            _make_tx(tool_id="a", args={}, success=True, scored_by="oracle", ts=1),
            _make_tx(tool_id="b", args={}, success=True, scored_by="oracle", ts=2),
            _make_tx(tool_id="c", args={}, success=True, scored_by="oracle", ts=3),
            _make_tx(tool_id="d", args={}, success=True, scored_by="oracle", ts=4),
        ],
    }
    clusters = miner.cluster_sequences(groups)
    lengths = {len(seq) for seq in clusters}
    assert lengths.issubset({2, 3})
    assert 2 in lengths and 3 in lengths


def test_cluster_sequences_skips_short_groups(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_seq_len=2,
    )
    groups = {"C1": [_make_tx(tool_id="a", args={}, success=True, scored_by="oracle")]}
    assert miner.cluster_sequences(groups) == {}


# ── filter_recurrent ───────────────────────────────────────────────────


def test_filter_recurrent_keeps_above_threshold(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_occurrences=3,
    )
    seq = (("a", "h1"), ("b", "h2"))
    clusters = {seq: [[], [], []]}
    out = miner.filter_recurrent(clusters)
    assert len(out) == 1
    assert out[0]["sequence"] == seq


def test_filter_recurrent_drops_below_threshold(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_occurrences=3,
    )
    seq = (("a", "h1"),)
    out = miner.filter_recurrent({seq: [[], []]})
    assert out == []


def test_filter_recurrent_orders_longest_first(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_occurrences=3,
    )
    short = (("a", "h1"),)
    long_ = (("a", "h1"), ("b", "h2"), ("c", "h3"))
    clusters = {short: [[], [], []], long_: [[], [], []]}
    out = miner.filter_recurrent(clusters)
    assert out[0]["sequence"] == long_  # longest first


# ── split_success_failure ──────────────────────────────────────────────


def test_split_positive_when_success_and_scored(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    members = [[
        _make_tx(tool_id="a", args={}, success=True, scored_by="oracle"),
    ]]
    cluster = {"sequence": (("a", "h"),), "members": members}
    split = miner.split_success_failure(cluster)
    assert len(split["positive"]) == 1
    assert split["negative"] == []


def test_split_negative_when_terminal_failure(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    members = [[
        _make_tx(tool_id="a", args={}, success=False, scored_by="oracle"),
    ]]
    cluster = {"sequence": (("a", "h"),), "members": members}
    split = miner.split_success_failure(cluster)
    assert split["positive"] == []
    assert len(split["negative"]) == 1


def test_split_drops_unscored_members(store):
    """rFP §11.4 mines only Tier-1-verified TXs (scored_by ∈ {oracle, llm})."""
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    members = [[
        _make_tx(tool_id="a", args={}, success=True, scored_by=None),
    ]]
    cluster = {"sequence": (("a", "h"),), "members": members}
    split = miner.split_success_failure(cluster)
    assert split["positive"] == []
    assert split["negative"] == []


def test_split_negative_on_exception(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    tx = _make_tx(tool_id="a", args={}, success=True, scored_by="oracle", exception="boom")
    cluster = {"sequence": (("a", "h"),), "members": [[tx]]}
    split = miner.split_success_failure(cluster)
    assert split["negative"] and not split["positive"]


# ── abstract_cluster ───────────────────────────────────────────────────


def test_abstract_cluster_collects_unique_compiled_from(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    members = [
        [_make_tx(tool_id="a", args={}, success=True, scored_by="oracle", tx_hash="tx_1")],
        [_make_tx(tool_id="a", args={}, success=True, scored_by="oracle", tx_hash="tx_2")],
        [_make_tx(tool_id="a", args={}, success=True, scored_by="oracle", tx_hash="tx_1")],  # dup
    ]
    cluster = {"sequence": (("a", "h"),), "members": members}
    out = miner.abstract_cluster(cluster, members, "positive")
    assert out is not None
    assert sorted(out["compiled_from"]) == ["tx_1", "tx_2"]


def test_abstract_cluster_returns_none_on_bad_proposer():
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn, faiss_path="/tmp/_.faiss", snapshot_path="/tmp/_.json",
        embedder=None,
    )
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=lambda meta, kind: None,
        outer_memory_writer=MagicMock(),
    )
    cluster = {"sequence": (("a", "h"),), "members": [[_make_tx(tool_id="a", args={}, success=True, scored_by="oracle")]]}
    assert miner.abstract_cluster(cluster, cluster["members"], "positive") is None


# ── mine_pass end-to-end ───────────────────────────────────────────────


def test_mine_pass_compiles_recurrent_sequence(store, tmp_path):
    """3 occurrences of (read_buffer, write_buffer) sequence → 1 positive skill."""
    txs = []
    for i, parent in enumerate(["C1", "C2", "C3"]):
        ts_base = float(i * 10)
        txs.append(_make_tx(tool_id="read_buffer", args={"name": "goal"},
                            success=True, scored_by="oracle",
                            parent_chat_tx=parent, ts=ts_base,
                            tx_hash=f"tx_r_{i}"))
        txs.append(_make_tx(tool_id="write_buffer", args={"name": "retrieval", "content": "x"},
                            success=True, scored_by="oracle",
                            parent_chat_tx=parent, ts=ts_base + 1,
                            tx_hash=f"tx_w_{i}"))

    emitted: list[tuple[str, dict]] = []
    miner = ProceduralMiner(
        skill_store=store,
        tool_call_reader=lambda since, lim: txs,
        llm_proposer=_identity_proposer,
        outer_memory_writer=MagicMock(),
        bus_emit=lambda ev, p: emitted.append((ev, p)),
        min_occurrences=3, min_seq_len=2, max_seq_len=3,
    )
    summary = miner.mine_pass(dream_pass_id="dp_001")
    assert summary["txs_scanned"] == 6
    assert summary["clusters_recurrent"] >= 1
    assert summary["positive_skills_compiled"] >= 1
    assert any(ev == "META_SKILL_COMPILED" for ev, _ in emitted)


def test_mine_pass_respects_max_skills_cap(store):
    """Many recurrent shapes — cap at max_skills_per_pass."""
    # Build 5 distinct recurrent shapes (tool_id A → tool_id A+i), each 3×.
    txs = []
    for shape_i in range(5):
        for occ in range(3):
            parent = f"C_{shape_i}_{occ}"
            txs.append(_make_tx(tool_id=f"tool_{shape_i}_a", args={},
                                success=True, scored_by="oracle",
                                parent_chat_tx=parent, ts=1.0,
                                tx_hash=f"tx1_{shape_i}_{occ}"))
            txs.append(_make_tx(tool_id=f"tool_{shape_i}_b", args={},
                                success=True, scored_by="oracle",
                                parent_chat_tx=parent, ts=2.0,
                                tx_hash=f"tx2_{shape_i}_{occ}"))
    miner = ProceduralMiner(
        skill_store=store,
        tool_call_reader=lambda since, lim: txs,
        llm_proposer=_identity_proposer,
        outer_memory_writer=MagicMock(),
        max_skills_per_pass=2,
        min_occurrences=3, min_seq_len=2, max_seq_len=2,
    )
    summary = miner.mine_pass()
    total = summary["positive_skills_compiled"] + summary["negative_skills_compiled"]
    assert total <= 2


def test_mine_pass_idempotent(store):
    """Re-mining the same window produces the same skill_ids — no duplicates."""
    txs = []
    for i in range(3):
        parent = f"C_{i}"
        txs.append(_make_tx(tool_id="a", args={}, success=True, scored_by="oracle",
                            parent_chat_tx=parent, ts=1.0, tx_hash=f"a_{i}"))
        txs.append(_make_tx(tool_id="b", args={}, success=True, scored_by="oracle",
                            parent_chat_tx=parent, ts=2.0, tx_hash=f"b_{i}"))
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: txs,
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
        min_occurrences=3, min_seq_len=2, max_seq_len=2,
    )
    s1 = miner.mine_pass()
    s2 = miner.mine_pass()
    # Both passes should emit the same skill_ids
    assert sorted(s1["compiled_ids"]) == sorted(s2["compiled_ids"])
    # Table count should be deterministic
    rows = store.list_all()
    assert len({r["skill_id"] for r in rows}) == len(s1["compiled_ids"])


def test_mine_pass_anchors_summary(store):
    writer = MagicMock()
    writer.write_skill_mining_pass = MagicMock(return_value="anchor_tx_x")
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=writer,
    )
    miner.mine_pass(dream_pass_id="dp_X")
    assert writer.write_skill_mining_pass.called


def test_mine_pass_empty_window_safe(store):
    miner = ProceduralMiner(
        skill_store=store, tool_call_reader=lambda *_: [],
        llm_proposer=_identity_proposer, outer_memory_writer=MagicMock(),
    )
    summary = miner.mine_pass()
    assert summary["txs_scanned"] == 0
    assert summary["positive_skills_compiled"] == 0
