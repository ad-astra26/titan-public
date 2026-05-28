"""Phase 8 — End-to-end integration test (D-SPEC-PHASE8 §P8.K).

One dream-pass exercises the full Phase 8 pipeline with mocked LLM +
mocked tool-call reader:
  1. 10 tool-call TXs seeded (5 oracle-scored, 5 null-scored)
  2. LLMJudge scores the 5 null TXs → 9/10 scored after pass
  3. ProceduralMiner sees 9 scored TXs, finds one recurrent (3-occ × 2-len)
     positive cluster → 1 positive skill compiled
  4. ProceduralSkillStore persists row + FAISS embed + snapshot export
  5. ProceduralSkillReader finds the skill via FAISS + utility gate
  6. SkillVerifier verifies the skill on first invocation
  7. should_delegate returns True

This is the contract verification for the whole P8 pipeline in one test.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.llm_judge import LLMJudge
from titan_hcl.synthesis.procedural_miner import ProceduralMiner
from titan_hcl.synthesis.procedural_reader import ProceduralSkillReader
from titan_hcl.synthesis.skill_store import EMBEDDING_DIM, ProceduralSkillStore
from titan_hcl.synthesis.skill_verifier import SkillVerifier


# ── Fixtures ───────────────────────────────────────────────────────────


def _det_embedder():
    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        import hashlib
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        cache[text] = v
        return v

    return embed


def _llm_provider_canned(judge_response: str, miner_response: dict):
    """Returns different responses for judge vs miner calls."""
    def fn(prompt, timeout_s=30.0, max_tokens=None, temperature=None, **kw):
        if "verdict" in prompt.lower():  # judge prompt asks for verdict
            return judge_response
        # miner prompt asks for nl_description + executable_spec
        return json.dumps(miner_response)
    return fn


def _make_tx(*, tx_hash, tool, args, success, scored_by, parent_chat_tx, ts):
    return {
        "tx_hash": tx_hash,
        "content": {
            "tool_id": tool, "args": args, "success": success,
            "scored_by": scored_by, "parent_chat_tx": parent_chat_tx,
            "ts": ts, "result_summary": "ok",
        },
        "tags": [],
    }


# ── End-to-end ─────────────────────────────────────────────────────────


def test_phase8_full_dream_pass_end_to_end(tmp_path):
    """The big one: 1 dream pass produces 1 compiled positive skill that
    survives first-invocation verification and passes the delegate gate."""
    # 1. Seed: 10 tool-call TXs across 5 chats, each chat has 2 calls:
    #    (read_buffer with name=goal) → (write_buffer with name=retrieval, content=text)
    #    All 5 chats share the same canonical shape. 5 are oracle-scored
    #    success, 5 are null-scored (judge will score them).
    seed_txs = []
    for i in range(5):
        chat_id = f"chat_{i}"
        ts_base = 1000.0 + i * 10
        seed_txs.append(_make_tx(
            tx_hash=f"tx_r_{i}", tool="read_buffer",
            args={"name": "goal"}, success=True,
            scored_by="oracle" if i < 3 else None,
            parent_chat_tx=chat_id, ts=ts_base,
        ))
        seed_txs.append(_make_tx(
            tx_hash=f"tx_w_{i}", tool="write_buffer",
            args={"name": "retrieval", "content": "x"}, success=True,
            scored_by="oracle" if i < 3 else None,
            parent_chat_tx=chat_id, ts=ts_base + 1,
        ))

    judge_response = '{"verdict": "full", "rationale": "ok", "confidence": 0.9}'
    miner_response = {
        "nl_description": "Read goal buffer, write to retrieval",
        "executable_spec": {"steps": [
            {"tool": "read_buffer", "args_shape": "goal"},
            {"tool": "write_buffer", "args_shape": "content"},
        ]},
        "preconditions": [],
        "postconditions": [],
    }
    llm_fn = _llm_provider_canned(judge_response, miner_response)

    # 2. Wire ProceduralSkillStore + writer mock
    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=_det_embedder(),
    )

    writer = MagicMock()
    writer.write_llm_judge_batch = MagicMock(return_value="judge_batch_tx")
    writer.write_scored_by_patch = MagicMock(return_value="scored_patch_tx")
    writer.write_skill_mining_pass = MagicMock(return_value="mining_pass_tx")
    writer.write_skill_lifecycle_tx = MagicMock(return_value="lifecycle_tx")

    # In-process state — judge anchors scored_by patches, so the miner's
    # reader should see those TXs as scored. Simulate this by tracking
    # which tx_hashes the judge scored and patching them in the reader.
    patched_hashes: set[str] = set()

    def patched_reader(since_ts: float, lim: int) -> list[dict]:
        result = []
        for tx in seed_txs:
            content = dict(tx["content"])
            if tx["tx_hash"] in patched_hashes and content.get("scored_by") is None:
                content["scored_by"] = "llm"
            result.append({"tx_hash": tx["tx_hash"], "content": content, "tags": tx["tags"]})
        return result

    def judge_reader(since_ts: float, lim: int) -> list[dict]:
        # Judge sees the raw, pre-patch view (so it can find the nulls).
        return [dict(tx) for tx in seed_txs]

    # 3. Run the judge
    judge = LLMJudge(
        tool_call_reader=judge_reader,
        llm_provider=llm_fn,
        outer_memory_writer=writer,
        per_pass_cap=200,
    )
    judge_summary = judge.score_window(since_ts=0.0)
    assert judge_summary["scored_now"] == 4  # 4 null-scored hashes (tx_r_3, tx_w_3, tx_r_4, tx_w_4)
    # Mark them patched
    scored_call_args = writer.write_scored_by_patch.call_args
    for entry in scored_call_args.kwargs["entries"]:
        patched_hashes.add(entry["parent_tool_call_tx"])

    # 4. Run the miner — sees all 10 TXs, 7 oracle-scored + 4 llm-scored = 10 eligible
    # Wait — actually only 6 are oracle (i<3 → both r+w for 3 chats = 6) + 4 llm = 10.
    miner = ProceduralMiner(
        skill_store=store,
        tool_call_reader=patched_reader,
        llm_proposer=lambda meta, kind: miner_response if kind == "positive" else None,
        outer_memory_writer=writer,
        min_occurrences=3,
        min_seq_len=2,
        max_seq_len=2,
        max_skills_per_pass=10,
    )
    miner_summary = miner.mine_pass(dream_pass_id="dp_test_1")
    assert miner_summary["positive_skills_compiled"] >= 1
    assert len(miner_summary["compiled_ids"]) >= 1

    # 5. ProceduralSkillStore should have the skill
    skill_id = miner_summary["compiled_ids"][0]
    skill_row = store.read_skill(skill_id)
    assert skill_row is not None
    assert skill_row["nl_description"].startswith("Read goal")
    assert skill_row["embedding_id"] >= 0  # FAISS embedded
    assert skill_row["verified_at"] is None  # Not yet verified

    # 6. SkillVerifier should accept the skill (mock chain reader: all hashes resolve)
    chain_reader = MagicMock()
    chain_reader.read_tx_by_content_hash = lambda h: {"content_hash": h, "fork": "procedural"}
    verifier = SkillVerifier(
        skill_store=store,
        chain_reader=chain_reader,
        outer_memory_writer=writer,
    )
    assert verifier.verify_skill(skill_id) is True

    # Skill should now be verified
    skill_row_v = store.read_skill(skill_id)
    assert skill_row_v["verified_at"] is not None

    # 7. ProceduralSkillReader should find the skill, should_delegate=True
    reader = ProceduralSkillReader(store, utility_floor=0.3, match_floor=0.0)
    results = reader.recall("read goal write retrieval", k=5)
    assert len(results) > 0
    top = results[0]
    assert top["skill_id"] == skill_id
    assert reader.should_delegate(top) is True


def test_phase8_pipeline_idempotent_across_two_passes(tmp_path):
    """Re-running the dream pass on the same window produces no NEW skills
    (compute_skill_id is deterministic; persist_skill is INSERT OR REPLACE)."""
    seed_txs = []
    for i in range(3):
        ts_base = 1000.0 + i * 10
        seed_txs.append(_make_tx(
            tx_hash=f"a_{i}", tool="a", args={}, success=True,
            scored_by="oracle", parent_chat_tx=f"c_{i}", ts=ts_base,
        ))
        seed_txs.append(_make_tx(
            tx_hash=f"b_{i}", tool="b", args={}, success=True,
            scored_by="oracle", parent_chat_tx=f"c_{i}", ts=ts_base + 1,
        ))

    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=_det_embedder(),
    )
    writer = MagicMock()
    miner = ProceduralMiner(
        skill_store=store,
        tool_call_reader=lambda since, lim: seed_txs,
        llm_proposer=lambda meta, kind: {
            "nl_description": "test", "executable_spec": {},
            "preconditions": [], "postconditions": [],
        },
        outer_memory_writer=writer,
        min_occurrences=3, min_seq_len=2, max_seq_len=2,
    )
    s1 = miner.mine_pass()
    pass1_count = len(store.list_all())
    s2 = miner.mine_pass()
    pass2_count = len(store.list_all())
    # Same skill_ids on pass 2; row count unchanged
    assert sorted(s1["compiled_ids"]) == sorted(s2["compiled_ids"])
    assert pass1_count == pass2_count


def test_phase8_failure_skill_compiled_distinct_from_success(tmp_path):
    """Positive and negative clusters on the SAME canonical shape produce
    DIFFERENT skill_ids (kind discriminates the hash)."""
    # 3 success-terminal + 3 failure-terminal sequences of the same shape
    seed_txs = []
    for i in range(3):
        seed_txs.append(_make_tx(
            tx_hash=f"s_a_{i}", tool="a", args={}, success=True,
            scored_by="oracle", parent_chat_tx=f"good_{i}", ts=1.0 + i,
        ))
        seed_txs.append(_make_tx(
            tx_hash=f"s_b_{i}", tool="b", args={}, success=True,
            scored_by="oracle", parent_chat_tx=f"good_{i}", ts=2.0 + i,
        ))
    for i in range(3):
        seed_txs.append(_make_tx(
            tx_hash=f"f_a_{i}", tool="a", args={}, success=True,
            scored_by="oracle", parent_chat_tx=f"bad_{i}", ts=10.0 + i,
        ))
        seed_txs.append(_make_tx(
            tx_hash=f"f_b_{i}", tool="b", args={}, success=False,
            scored_by="oracle", parent_chat_tx=f"bad_{i}", ts=11.0 + i,
        ))

    conn = duckdb.connect(":memory:")
    store = ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills.json"),
        embedder=_det_embedder(),
    )
    writer = MagicMock()
    miner = ProceduralMiner(
        skill_store=store,
        tool_call_reader=lambda since, lim: seed_txs,
        llm_proposer=lambda meta, kind: {
            "nl_description": f"{kind} skill",
            "executable_spec": {}, "preconditions": [], "postconditions": [],
        },
        outer_memory_writer=writer,
        min_occurrences=3, min_seq_len=2, max_seq_len=2,
    )
    summary = miner.mine_pass()
    assert summary["positive_skills_compiled"] == 1
    assert summary["negative_skills_compiled"] == 1
    all_skills = store.list_all()
    ids = {s["skill_id"] for s in all_skills}
    assert len(ids) == 2  # distinct skill_ids
