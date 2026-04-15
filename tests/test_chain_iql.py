"""
Tests for TUNING-012 v2 Sub-phase B — Chain-Level IQL Hierarchy.

Verifies:
  - task_embedding helpers (encode_task, extract_chain_template)
  - ChainTemplateQNet forward/backward
  - ChainIQL buffer + template registry + training + best-template lookup
  - JSON persistence round-trip
"""

import json
import os

import numpy as np
import pytest

from titan_plugin.logic.task_embedding import (
    encode_task,
    extract_chain_template,
    template_to_primitive_list,
)
from titan_plugin.logic.chain_iql import ChainTemplateQNet, ChainIQL


# ── task_embedding ────────────────────────────────────────────────


class TestTaskEmbedding:
    def test_encode_task_returns_correct_shape(self):
        emb = encode_task("inner_spirit", "low_commit_rate", [0.5] * 132, dim=32)
        assert emb.shape == (32,)
        assert emb.dtype == np.float32

    def test_encode_task_deterministic(self):
        """Same inputs → same embedding (across calls)."""
        e1 = encode_task("inner_spirit", "trigger_a", [0.5] * 132, dim=32)
        e2 = encode_task("inner_spirit", "trigger_a", [0.5] * 132, dim=32)
        np.testing.assert_array_equal(e1, e2)

    def test_encode_task_distinguishes_domain(self):
        """Different domains produce different embeddings."""
        e1 = encode_task("inner_spirit", "trigger_a", [0.5] * 132, dim=32)
        e2 = encode_task("body_mind", "trigger_a", [0.5] * 132, dim=32)
        assert not np.array_equal(e1, e2)

    def test_encode_task_distinguishes_state(self):
        """Different state vectors produce different embeddings (state half changes)."""
        e1 = encode_task("inner_spirit", "trigger_a", [0.1] * 132, dim=32)
        e2 = encode_task("inner_spirit", "trigger_a", [0.9] * 132, dim=32)
        # Categorical half should match, state half should differ
        np.testing.assert_array_equal(e1[:16], e2[:16])
        assert not np.array_equal(e1[16:], e2[16:])

    def test_encode_task_handles_short_state(self):
        """Short state vectors should be padded."""
        emb = encode_task("inner_spirit", "trigger", [0.5] * 10, dim=32)
        assert emb.shape == (32,)

    def test_encode_task_handles_none_state(self):
        emb = encode_task("inner_spirit", "trigger", None, dim=32)
        assert emb.shape == (32,)
        # State half should be all zero when no state provided
        np.testing.assert_array_almost_equal(emb[16:], np.zeros(16))

    def test_encode_task_rejects_odd_dim(self):
        with pytest.raises(ValueError):
            encode_task("a", "b", [0.5] * 132, dim=31)

    def test_extract_chain_template(self):
        chain = ["FORMULATE.define", "RECALL.chain_archive", "EVALUATE.check_progress"]
        assert extract_chain_template(chain) == "FORMULATE→RECALL→EVALUATE"

    def test_extract_chain_template_empty(self):
        assert extract_chain_template([]) == ""

    def test_template_to_primitive_list_roundtrip(self):
        chain = ["FORMULATE.define", "RECALL.chain_archive", "EVALUATE.check_progress"]
        tmpl = extract_chain_template(chain)
        prims = template_to_primitive_list(tmpl)
        assert prims == ["FORMULATE", "RECALL", "EVALUATE"]


# ── ChainTemplateQNet ─────────────────────────────────────────────


class TestChainTemplateQNet:
    def test_forward_returns_scalar(self):
        net = ChainTemplateQNet(task_dim=32, template_count=10, hidden=16)
        task_emb = np.random.randn(32).astype(np.float32)
        q, cache = net.forward(task_emb, 0)
        assert isinstance(q, float)
        assert "x" in cache

    def test_predict_clipped_to_unit_range(self):
        net = ChainTemplateQNet(task_dim=32, template_count=10, hidden=16)
        task_emb = np.random.randn(32).astype(np.float32) * 100  # extreme input
        q = net.predict(task_emb, 0)
        assert 0.0 <= q <= 1.0

    def test_train_step_decreases_loss(self):
        """Training should reduce loss for a single (task, template, target) triple."""
        net = ChainTemplateQNet(task_dim=32, template_count=10, hidden=16, lr=0.05)
        task_emb = np.random.RandomState(7).randn(32).astype(np.float32)
        # Train against a fixed target multiple times
        target = 0.8
        losses = []
        for _ in range(50):
            losses.append(net.train_step(task_emb, 3, target))
        assert losses[-1] < losses[0]
        # After many steps, prediction should approach target
        q = net.predict(task_emb, 3)
        assert abs(q - target) < 0.2

    def test_persistence_roundtrip(self):
        net1 = ChainTemplateQNet(task_dim=32, template_count=10, hidden=16)
        task_emb = np.random.RandomState(11).randn(32).astype(np.float32)
        # Train a bit
        for _ in range(20):
            net1.train_step(task_emb, 2, 0.7)
        d = net1.to_dict()
        # Restore into a fresh net
        net2 = ChainTemplateQNet(task_dim=32, template_count=10, hidden=16)
        net2.from_dict(d)
        q1 = net1.predict(task_emb, 2)
        q2 = net2.predict(task_emb, 2)
        assert abs(q1 - q2) < 1e-6
        assert net2.total_updates == net1.total_updates


# ── ChainIQL ──────────────────────────────────────────────────────


def _dna() -> dict:
    return {
        "task_embedding_dim": 32,
        "chain_template_max_count": 50,
        "chain_qnet_lr": 0.01,
        "chain_iql_buffer_size": 100,
        "chain_blend_alpha": 0.5,
        "chain_iql_enabled": True,
    }


class TestChainIQL:
    def test_record_and_buffer_grows(self, tmp_path):
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("inner_spirit", "trigger", [0.5] * 132)
        chain = ["FORMULATE.define", "RECALL.chain_archive", "EVALUATE.check_progress"]
        iql.record_chain_outcome(task_emb, chain, task_success=0.7,
                                 primitives=["FORMULATE", "RECALL", "EVALUATE"], domain="meta")
        assert len(iql.buffer) == 1
        assert "FORMULATE→RECALL→EVALUATE" in iql.template_registry

    def test_query_best_template_returns_none_when_empty(self, tmp_path):
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("inner_spirit", "trigger", [0.5] * 132)
        best, q = iql.query_best_template(task_emb)
        assert best is None
        assert q == 0.0

    def test_consolidate_during_dream_trains_qnet(self, tmp_path):
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "test_trigger", [0.5] * 132)
        # Add 50 chain outcomes — same template, varying success
        for i in range(50):
            success = 0.8 if i % 2 == 0 else 0.3
            iql.record_chain_outcome(
                task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
                task_success=success, primitives=["FORMULATE", "RECALL"], domain="meta",
            )
        result = iql.consolidate_during_dream(batch_size=32)
        assert result["trained"] is True
        assert result["samples"] == 32
        assert result["template_count"] == 1
        assert result["total_updates"] == 32

    def test_query_best_template_after_training(self, tmp_path):
        """After training on outcomes, the best template's Q should reflect avg success."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "test_trigger", [0.5] * 132)
        # Add MANY outcomes with template_A always succeeding
        for _ in range(40):
            iql.record_chain_outcome(
                task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
                task_success=0.9, primitives=["FORMULATE", "RECALL"], domain="meta",
            )
        # Add MANY outcomes with template_B always failing
        for _ in range(40):
            iql.record_chain_outcome(
                task_emb, ["BREAK.rewind_last", "RECALL.experience"],
                task_success=0.1, primitives=["BREAK", "RECALL"], domain="meta",
            )
        # Train multiple dream cycles
        for _ in range(20):
            iql.consolidate_during_dream(batch_size=32)
        # After training, template_A should rank higher
        best, q_best = iql.query_best_template(task_emb)
        assert best == "FORMULATE→RECALL"
        # Q for template_A should be MUCH higher than Q for template_B
        tid_a = iql.template_registry["FORMULATE→RECALL"]
        tid_b = iql.template_registry["BREAK→RECALL"]
        q_a = iql.qnet.predict(task_emb, tid_a)
        q_b = iql.qnet.predict(task_emb, tid_b)
        assert q_a > q_b

    def test_save_and_load_roundtrip(self, tmp_path):
        iql1 = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        for _ in range(10):
            iql1.record_chain_outcome(
                task_emb, ["FORMULATE.define"], task_success=0.6,
                primitives=["FORMULATE"], domain="meta",
            )
        iql1.consolidate_during_dream(batch_size=10) if len(iql1.buffer) >= 10 else None
        # Save explicitly
        iql1.save()
        assert os.path.exists(iql1._save_path)

        # Reload into fresh instance
        iql2 = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        assert "FORMULATE" in iql2.template_registry
        assert len(iql2.buffer) == 10

    def test_disabled_iql_skips_recording(self, tmp_path):
        dna = _dna()
        dna["chain_iql_enabled"] = False
        iql = ChainIQL(dna=dna, save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define"], task_success=0.5,
            primitives=["FORMULATE"], domain="meta",
        )
        assert len(iql.buffer) == 0

    def test_template_registry_assigns_unique_ids(self, tmp_path):
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        id_a = iql.get_or_assign_template_id("A→B→C")
        id_b = iql.get_or_assign_template_id("X→Y→Z")
        id_a_again = iql.get_or_assign_template_id("A→B→C")
        assert id_a != id_b
        assert id_a == id_a_again
        assert id_a == 0
        assert id_b == 1

    def test_stratified_sampling_reports_groups(self, tmp_path):
        """Phase D pre-flight #4b: stratified sampling reports group count and
        ensures rare templates aren't drowned out by a dominant template."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        # Skewed buffer: 90 of template A, 10 of template B (matches T1 collapse)
        for _ in range(90):
            iql.record_chain_outcome(
                task_emb, ["FORMULATE.define"],
                task_success=0.2, primitives=["FORMULATE"], domain="meta",
            )
        for _ in range(10):
            iql.record_chain_outcome(
                task_emb, ["RECALL.chain_archive"],
                task_success=0.9, primitives=["RECALL"], domain="meta",
            )
        result = iql.consolidate_during_dream(batch_size=20)
        assert result["trained"] is True
        # New stat: stratified_groups should match number of distinct templates
        assert result["stratified_groups"] == 2
        # batch should be exactly batch_size (20), not under or over
        assert result["samples"] == 20

    def test_stratified_with_single_template_unchanged(self, tmp_path):
        """Backwards compatibility: when there's only 1 template, behavior is
        equivalent to uniform random — full batch from the single group."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        for _ in range(50):
            iql.record_chain_outcome(
                task_emb, ["FORMULATE.define"],
                task_success=0.5, primitives=["FORMULATE"], domain="meta",
            )
        result = iql.consolidate_during_dream(batch_size=32)
        assert result["trained"] is True
        assert result["stratified_groups"] == 1
        assert result["samples"] == 32

    def test_stratified_rare_template_gets_signal(self, tmp_path):
        """The whole point of #4b: a rare-but-strong template should be
        learnable even when the buffer is dominated by a frequent-mediocre one.
        """
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        # 90 entries of FORMULATE with mediocre 0.3 success
        for _ in range(90):
            iql.record_chain_outcome(
                task_emb, ["FORMULATE.define"],
                task_success=0.3, primitives=["FORMULATE"], domain="meta",
            )
        # 10 entries of RECALL with strong 0.9 success
        for _ in range(10):
            iql.record_chain_outcome(
                task_emb, ["RECALL.chain_archive"],
                task_success=0.9, primitives=["RECALL"], domain="meta",
            )
        # Train multiple dream cycles
        for _ in range(15):
            iql.consolidate_during_dream(batch_size=20)
        # Q for the rare-strong template should be HIGHER than the frequent-mediocre
        tid_strong = iql.template_registry["RECALL"]
        tid_weak = iql.template_registry["FORMULATE"]
        q_strong = iql.qnet.predict(task_emb, tid_strong)
        q_weak = iql.qnet.predict(task_emb, tid_weak)
        assert q_strong > q_weak, (
            f"Rare-strong template should outrank frequent-mediocre: "
            f"q_strong={q_strong:.3f} q_weak={q_weak:.3f}"
        )


# ── Phase D.1 — External Reward Injection (META_LANGUAGE loop) ───────


class TestExternalRewardInjection:
    """Phase D.1: chain_iql.apply_external_reward blends external rewards
    from the META_LANGUAGE closed loop into buffer entries (Option B)."""

    def test_record_chain_outcome_persists_chain_id(self, tmp_path):
        """D.1: chain_id is carried into the buffer entry."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.5, primitives=["FORMULATE", "RECALL"],
            domain="meta", chain_id=42,
        )
        assert len(iql.buffer) == 1
        assert iql.buffer[-1]["chain_id"] == 42

    def test_record_chain_outcome_default_chain_id_is_minus_one(self, tmp_path):
        """Legacy callers that don't pass chain_id get -1 (no tracking)."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.5, primitives=["FORMULATE", "RECALL"], domain="meta",
        )
        assert iql.buffer[-1]["chain_id"] == -1

    def test_apply_external_reward_blends_buffer_entry(self, tmp_path):
        """alpha=0.5: new = 0.4 * 0.5 + 0.8 * 0.5 = 0.6."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.4, primitives=["FORMULATE", "RECALL"],
            domain="meta", chain_id=7,
        )
        applied = iql.apply_external_reward(chain_id=7, external_reward=0.8, alpha=0.5)
        assert applied is True
        entry = iql.buffer[-1]
        assert abs(entry["task_success"] - 0.6) < 1e-6
        assert entry.get("external_applied") is True

    def test_apply_external_reward_unknown_chain_id_noop(self, tmp_path):
        """Unknown chain_id → False + late_drops counter increments."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.5, primitives=["FORMULATE", "RECALL"],
            domain="meta", chain_id=1,
        )
        before = iql._external_reward_late_drops
        applied = iql.apply_external_reward(
            chain_id=999, external_reward=0.8, alpha=0.5)
        assert applied is False
        assert iql._external_reward_late_drops == before + 1
        # Existing entry untouched
        assert iql.buffer[-1]["task_success"] == 0.5

    def test_apply_external_reward_clips_to_unit_interval(self, tmp_path):
        """reward=1.5 → clipped to 1.0 before blend; alpha=-0.1 → 0."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.2, primitives=["FORMULATE", "RECALL"],
            domain="meta", chain_id=3,
        )
        # ext clipped to 1.0, alpha 0.5 → new = 0.2*0.5 + 1.0*0.5 = 0.6
        iql.apply_external_reward(chain_id=3, external_reward=1.5, alpha=0.5)
        assert abs(iql.buffer[-1]["task_success"] - 0.6) < 1e-6

    def test_apply_external_reward_negative_chain_id_noop(self, tmp_path):
        """chain_id=-1 (legacy marker) must not match anything."""
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        task_emb = encode_task("meta", "trig", [0.5] * 132)
        iql.record_chain_outcome(
            task_emb, ["FORMULATE.define", "RECALL.chain_archive"],
            task_success=0.5, primitives=["FORMULATE", "RECALL"], domain="meta",
        )
        # buffer entry has chain_id=-1 by default, but we still refuse
        # to apply because negative chain_ids are the "no tracking" sentinel
        assert iql.apply_external_reward(
            chain_id=-1, external_reward=0.8, alpha=0.5) is False

    def test_get_stats_exposes_late_drops(self, tmp_path):
        iql = ChainIQL(dna=_dna(), save_dir=str(tmp_path))
        assert iql.get_stats()["external_reward_late_drops"] == 0
        iql.apply_external_reward(
            chain_id=999, external_reward=0.5, alpha=0.5)
        assert iql.get_stats()["external_reward_late_drops"] == 1
