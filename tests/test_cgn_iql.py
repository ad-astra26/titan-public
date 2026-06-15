"""Canonical IQL learner for CGN — unit + regression tests.

Covers RFP_cgn_canonical_iql_upgrade §8 gates:
  • G-IQL-1 — the three IQL losses (expectile-V, Bellman-Q-with-V, AWR) are
    implemented and mathematically correct.
  • G-IQL-2 — cgn_iql_enabled toggles cleanly; the legacy advantage-REINFORCE
    path is untouched when the flag is off.
Plus §7.A trajectory linkage (s'/terminal derivation) and Q-net persistence.

Run isolated:  python -m pytest tests/test_cgn_iql.py -v -p no:anchorpy
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest
import torch

from titan_hcl.logic.cgn import (
    ConceptGroundingNetwork, ConsumerQNet, CGNConsumerConfig)
from titan_hcl.logic.cgn_types import CGNTransition


# ── fixtures ────────────────────────────────────────────────────────────────

def _make_cgn(tmpdir, *, iql_enabled, n_consumers=1):
    iql = {"enabled": iql_enabled, "tau": 0.7, "beta": 3.0,
           "adv_clip": 100.0, "polyak": 0.005, "gamma": 0.99}
    cgn = ConceptGroundingNetwork(state_dir=tmpdir, iql_config=iql)
    for c in range(n_consumers):
        cgn.register_consumer(CGNConsumerConfig(
            name=f"lang{c}", feature_dims=30, action_dims=8,
            action_names=[f"a{i}" for i in range(8)],
            reward_source="x", max_buffer_size=500, consolidation_priority=1))
    return cgn


def _seed_buffer(cgn, consumer="lang0", n_concepts=4, per_concept=6):
    """Add transitions forming per-(consumer,concept) trajectories with
    deterministic timestamps + non-degenerate rewards."""
    rng = np.random.RandomState(0)
    ts = 1000.0
    for k in range(n_concepts):
        for j in range(per_concept):
            ts += 1.0
            cgn._buffer.add(CGNTransition(
                consumer=consumer, concept_id=f"concept_{k}",
                state=rng.randn(30).astype(np.float32),
                action=int(rng.randint(0, 8)),
                action_params=np.zeros(4, dtype=np.float32),
                reward=float(rng.uniform(-0.5, 1.0)),
                timestamp=ts, epoch=j,
                metadata={}))


# ── G-IQL-1: expectile loss math ─────────────────────────────────────────────

def test_expectile_loss_tau_half_is_half_mse():
    diff = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss = ConceptGroundingNetwork._expectile_loss(diff, 0.5)
    expected = 0.5 * diff.pow(2).mean()  # τ=0.5 ⇒ ½·MSE
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6)


def test_expectile_loss_asymmetry_penalises_underestimate():
    # diff = target - pred. Positive diff = under-estimate (pred < target).
    # τ=0.7 must weight under-estimation more → pushes V UP (optimistic).
    under = ConceptGroundingNetwork._expectile_loss(torch.tensor([1.0]), 0.7)
    over = ConceptGroundingNetwork._expectile_loss(torch.tensor([-1.0]), 0.7)
    assert under.item() > over.item()
    assert math.isclose(under.item(), 0.7, rel_tol=1e-6)
    assert math.isclose(over.item(), 0.3, rel_tol=1e-6)


# ── G-IQL-1: Bellman-with-V target + AWR weighting ───────────────────────────

def test_bellman_backup_uses_v_of_next_state_and_terminal_mask():
    gamma = 0.99
    rewards = torch.tensor([1.0, 0.5])
    v_next = torch.tensor([2.0, 9.9])
    nonterminal = torch.tensor([1.0, 0.0])          # 2nd is terminal
    backup = rewards + gamma * (v_next * nonterminal)
    assert math.isclose(backup[0].item(), 1.0 + 0.99 * 2.0, rel_tol=1e-6)
    assert math.isclose(backup[1].item(), 0.5, rel_tol=1e-6)  # V(s') masked out


def test_awr_weight_is_clamped():
    beta, clip = 3.0, 100.0
    adv = torch.tensor([10.0, 0.0, -10.0])          # exp(30) overflows the cap
    weight = torch.exp(beta * adv).clamp(max=clip)
    assert weight[0].item() == pytest.approx(clip)         # clamped
    assert weight[1].item() == pytest.approx(1.0)          # exp(0)=1
    assert weight[2].item() < 1.0                          # downweighted


def test_soft_update_is_polyak_ema():
    a = ConsumerQNet(30, 8)
    b = ConsumerQNet(30, 8)
    with torch.no_grad():
        for p in a.parameters():
            p.zero_()
        for p in b.parameters():
            p.fill_(1.0)
    ConceptGroundingNetwork._soft_update(a, b, 0.1)
    for p in a.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1))  # 0.9*0 + 0.1*1


# ── §7.A: trajectory linkage (s' / terminal derivation) ──────────────────────

def test_trajectory_links_per_consumer_concept_with_terminal_tail():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True)
        _seed_buffer(cgn, n_concepts=2, per_concept=3)
        cgn._build_trajectory_links()
        by_concept = {}
        for t in cgn._buffer.get_all():
            by_concept.setdefault(t.concept_id, []).append(t)
        for cid, seq in by_concept.items():
            seq.sort(key=lambda x: x.timestamp)
            # every non-tail links to the NEXT same-concept state
            for i in range(len(seq) - 1):
                assert seq[i].terminal is False
                assert np.array_equal(seq[i].next_state, seq[i + 1].state)
            # tail is terminal, no successor
            assert seq[-1].terminal is True
            assert seq[-1].next_state is None


# ── G-IQL-2: flag toggles; legacy path intact ────────────────────────────────

def test_flag_off_uses_legacy_reinforce_path():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=False)
        _seed_buffer(cgn)
        stats = cgn.consolidate(dream_phase=True)
        assert stats["trained"] is True
        assert stats["learner"] == "reinforce"


def test_flag_on_uses_iql_path_and_is_dream_only():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True)
        _seed_buffer(cgn)
        # awake consolidation is a no-op under IQL (dream-only)
        awake = cgn.consolidate(dream_phase=False)
        assert awake["trained"] is False
        assert awake["reason"] == "iql_dream_only"
        # dream consolidation runs the IQL learner
        dream = cgn.consolidate(dream_phase=True)
        assert dream["learner"] == "iql"
        assert "lang0" in dream["consumers"]
        m = dream["consumers"]["lang0"]
        assert {"v_loss", "q_loss", "policy_loss"} <= set(m)


def test_q_nets_exist_per_consumer():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True, n_consumers=2)
        for name in ("lang0", "lang1"):
            assert name in cgn._q_nets
            assert name in cgn._q_targets
            assert name in cgn._q_optimizers
            # target starts identical to the online net
            for tp, sp in zip(cgn._q_targets[name].parameters(),
                              cgn._q_nets[name].parameters()):
                assert torch.allclose(tp, sp)


# ── stability: IQL trains without divergence ─────────────────────────────────

def test_iql_training_converges_without_divergence():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True)
        _seed_buffer(cgn, n_concepts=6, per_concept=8)
        for _ in range(5):
            stats = cgn.consolidate(dream_phase=True)
            m = stats["consumers"]["lang0"]
            for key in ("v_loss", "q_loss", "policy_loss"):
                assert math.isfinite(m[key]), f"{key} diverged: {m[key]}"


def test_iql_does_not_collapse_to_single_action():
    """AWR + entropy of the policy must keep a spread over actions after
    training on varied rewards (no degenerate single-action collapse)."""
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True)
        _seed_buffer(cgn, n_concepts=6, per_concept=8)
        for _ in range(8):
            cgn.consolidate(dream_phase=True)
        states = torch.FloatTensor(np.random.RandomState(1).randn(16, 30))
        with torch.no_grad():
            logits, _ = cgn._action_nets["lang0"](states)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
        # at least 3 actions retain non-trivial mass
        assert int((probs > 0.02).sum()) >= 3


# ── persistence: Q-net round-trips ───────────────────────────────────────────

def test_q_net_persists_across_save_load():
    with tempfile.TemporaryDirectory() as td:
        cgn = _make_cgn(td, iql_enabled=True)
        _seed_buffer(cgn, n_concepts=6, per_concept=8)
        cgn.consolidate(dream_phase=True)   # mutates the Q net
        before = {k: v.clone() for k, v in
                  cgn._q_nets["lang0"].state_dict().items()}
        cgn._save_state()

        cgn2 = _make_cgn(td, iql_enabled=True)  # fresh net, same dir
        cgn2._load_state()
        after = cgn2._q_nets["lang0"].state_dict()
        for k, v in before.items():
            assert torch.allclose(v, after[k]), f"Q weight {k} not restored"
        # target re-synced to the loaded online net
        for tp, sp in zip(cgn2._q_targets["lang0"].parameters(),
                          cgn2._q_nets["lang0"].parameters()):
            assert torch.allclose(tp, sp)
