"""Phase 10G (rFP §3G) — Dream Bridge A restoration.

Verifies the harvest relocated to logic/dream_bridge.py produces felt-tagged
memories from the inner DBs, deduplicates, and marks reasoning chains — the
inner→outer injection loop that was dropped at the D8-3 spirit_worker gutting.
"""
import json
import os
import sqlite3
import tempfile

import titan_hcl.logic.spirit_helpers as sh
from titan_hcl.logic.dream_bridge import harvest_dream_memories


class _FakeChainArchive:
    def __init__(self):
        self.marked = []

    def get_unconsolidated(self, limit=50):
        return [
            {"id": 1, "outcome_score": 0.9, "domain": "reasoning",
             "strategy_label": "decompose", "chain_length": 5},
            {"id": 2, "outcome_score": 0.3, "domain": "x"},  # below 0.7 → skipped
        ]

    def mark_consolidated(self, chain_ids):
        self.marked.extend(chain_ids)


def _seed_inner_memory_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE meta_wisdom (id INTEGER PRIMARY KEY, problem_pattern TEXT, "
        "strategy_sequence TEXT, confidence REAL, times_reused INTEGER, "
        "crystallized INTEGER)")
    conn.execute(
        "INSERT INTO meta_wisdom VALUES (1, 'recursive decomposition', "
        "'split→solve→merge', 0.92, 4, 1)")
    conn.execute(
        "CREATE TABLE vocabulary (word TEXT, cross_modal_conf REAL, confidence REAL, "
        "times_encountered INTEGER, times_produced INTEGER)")
    conn.execute("INSERT INTO vocabulary VALUES ('resonance', 0.42, 0.8, 30, 12)")
    conn.execute(
        "CREATE TABLE composition_history (id INTEGER PRIMARY KEY, sentence TEXT, "
        "level INTEGER, confidence REAL)")
    conn.execute(
        "INSERT INTO composition_history VALUES (1, 'the spheres breathe as one', 8, 0.9)")
    conn.commit()
    conn.close()


def test_harvest_produces_felt_tagged_memories_and_marks_chains(monkeypatch, tmp_path):
    db = os.path.join(tmp_path, "inner_memory.db")
    _seed_inner_memory_db(db)
    # Isolate the dedup file so the test never touches ./data/.
    dedup_path = os.path.join(tmp_path, "dream_bridge_dedup.json")
    monkeypatch.setattr(sh, "_BRIDGE_DEDUP_PATH", dedup_path)

    felt = {"DA": 0.6, "5HT": 0.4, "emotion": "wonder",
            "emotion_confidence": 0.7, "dream_cycle": 12, "ts": 1.0}
    chain_archive = _FakeChainArchive()

    memories, chain_ids = harvest_dream_memories(
        chain_archive, meta_wisdom=True, felt=felt,
        cgn_db_path=db, dream_cycle=12)

    tags = {m["text"].split("]")[0] + "]" for m in memories}
    assert "[DREAM_WISDOM]" in tags
    assert "[EUREKA]" in tags
    assert "[CGN_MILESTONE]" in tags
    assert "[COMPOSITION]" in tags
    # Every injected memory carries the felt snapshot for Bridge B perturbation.
    assert all(m["neuromod_context"] is felt for m in memories)
    assert all(m["source"] == "dream_consolidation" for m in memories)
    # Only the high-scoring chain (id=1) is returned for consolidation
    # (the harvest is pure — cognitive_worker._run_dream_bridge does the
    # actual chain_archive.mark_consolidated + MEMORY_ADD emits).
    assert chain_ids == [1]
    # Dedup persisted.
    assert os.path.exists(dedup_path)


def test_harvest_dedups_on_second_pass(monkeypatch, tmp_path):
    db = os.path.join(tmp_path, "inner_memory.db")
    _seed_inner_memory_db(db)
    dedup_path = os.path.join(tmp_path, "dream_bridge_dedup.json")
    monkeypatch.setattr(sh, "_BRIDGE_DEDUP_PATH", dedup_path)
    felt = {"DA": 0.5, "emotion": "calm", "dream_cycle": 1, "ts": 1.0}

    first, _ = harvest_dream_memories(_FakeChainArchive(), True, felt, db, 1)
    second, _ = harvest_dream_memories(_FakeChainArchive(), True, felt, db, 2)
    # Wisdom / CGN word / composition are deduped on the 2nd pass (only the
    # EUREKA chains re-appear because the fake archive returns them again).
    first_tags = [m["text"].split("]")[0] for m in first]
    second_tags = [m["text"].split("]")[0] for m in second]
    assert "[DREAM_WISDOM" in first_tags
    assert "[DREAM_WISDOM" not in second_tags
    assert "[CGN_MILESTONE" not in second_tags
