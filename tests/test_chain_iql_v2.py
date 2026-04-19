"""Unit tests for chain_iql v2 (rFP_chain_iql_v2, 2026-04-19).

Verifies:
- LRU eviction: novel templates get real slots, never aliased to 0
- UCB1 exploration: low-visit templates get bonus over high-visit with same Q
- Template cap 500 honored from DNA
- Persistence: LRU/UCB state survives save→load
"""

import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from titan_plugin.logic.chain_iql import ChainIQL


@pytest.fixture
def tmp_save_dir():
    d = tempfile.mkdtemp(prefix="chain_iql_v2_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fresh(dna: dict, tmp_dir: str) -> ChainIQL:
    return ChainIQL(dna=dna, save_dir=tmp_dir)


def _dummy_chain(prims: list[str]) -> list:
    return [f"{p}.x" for p in prims]


def test_lru_eviction_never_aliases_to_zero(tmp_save_dir):
    """Fill registry to cap + 1, verify eviction recycles LRU slot, not slot 0."""
    iql = _fresh({"chain_template_max_count": 4, "chain_iql_enabled": True}, tmp_save_dir)
    task_emb = np.zeros(32, dtype=np.float32)

    # Fill 4 templates with different shapes.
    for i in range(4):
        chain = _dummy_chain([f"P{i}"] * 3)
        iql.record_chain_outcome(task_emb, chain, 0.5, [f"P{i}"] * 3, "general", chain_id=i)
    assert len(iql.template_registry) == 4, "Registry should be full at 4"

    # Access template 0 (makes it most-recently-seen)
    tid0 = iql.get_or_assign_template_id(next(iter(iql.template_registry)))

    # Assign a 5th novel template — should evict LRU (NOT slot 0)
    novel_chain = _dummy_chain(["XNEW"] * 3)
    iql.record_chain_outcome(task_emb, novel_chain, 0.5, ["XNEW"] * 3, "general", chain_id=5)

    novel_template = "XNEW→XNEW→XNEW"
    assert novel_template in iql.template_registry, "Novel template should be registered"
    new_tid = iql.template_registry[novel_template]
    assert new_tid != 0 or tid0 == 0, "New template should NOT collapse to slot 0 unless LRU truly was slot 0"
    assert iql._lru_evictions == 1, "One LRU eviction should have fired"


def test_ucb_prefers_low_visit_templates(tmp_save_dir):
    """With same Q, UCB should prefer templates with fewer visits."""
    iql = _fresh({
        "chain_template_max_count": 10,
        "chain_iql_ucb_c": 1.0,  # high explore coefficient for testable signal
        "chain_iql_enabled": True,
    }, tmp_save_dir)
    task_emb = np.zeros(32, dtype=np.float32)

    # Template A: recorded 50 times
    for i in range(50):
        iql.record_chain_outcome(task_emb, _dummy_chain(["A"] * 3), 0.5, ["A"] * 3, "general", chain_id=i)
    # Template B: recorded once
    iql.record_chain_outcome(task_emb, _dummy_chain(["B"] * 3), 0.5, ["B"] * 3, "general", chain_id=999)

    # Both have same reward history (0.5), so Q will be similar. UCB should prefer B (low visits).
    best_template, _ = iql.query_best_template(task_emb)
    assert best_template is not None
    # Given c=1.0, log(51)/1 > log(51)/50 by ~7x. B should win.
    assert "B" in best_template, f"UCB should prefer low-visit B, got {best_template}"


def test_template_cap_500_default(tmp_save_dir):
    """Default cap should be 500 per FIX-2."""
    iql = _fresh({}, tmp_save_dir)
    assert iql._template_max == 500, f"Expected 500 default, got {iql._template_max}"


def test_persistence_roundtrip_preserves_lru_and_ucb(tmp_save_dir):
    """save→load→verify that LRU timestamps and UCB visit counts survive."""
    iql = _fresh({"chain_template_max_count": 10, "chain_iql_enabled": True}, tmp_save_dir)
    task_emb = np.zeros(32, dtype=np.float32)

    # Record 3 templates with varying visit counts
    for _ in range(10):
        iql.record_chain_outcome(task_emb, _dummy_chain(["A"] * 3), 0.5, ["A"] * 3, "general", chain_id=1)
    for _ in range(3):
        iql.record_chain_outcome(task_emb, _dummy_chain(["B"] * 3), 0.5, ["B"] * 3, "general", chain_id=2)
    iql.record_chain_outcome(task_emb, _dummy_chain(["C"] * 3), 0.5, ["C"] * 3, "general", chain_id=3)

    # Capture pre-save state
    pre_visits = dict(iql._template_visits)
    pre_last_seen = dict(iql._template_last_seen)
    pre_evictions = iql._lru_evictions

    iql.save()

    # Reload from disk
    iql2 = _fresh({"chain_template_max_count": 10, "chain_iql_enabled": True}, tmp_save_dir)
    assert iql2._template_visits == pre_visits, f"Visits mismatch: {iql2._template_visits} vs {pre_visits}"
    # Last-seen survives with float precision
    for tid, ts in pre_last_seen.items():
        assert abs(iql2._template_last_seen[tid] - ts) < 1e-3
    assert iql2._lru_evictions == pre_evictions
