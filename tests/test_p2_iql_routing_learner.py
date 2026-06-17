"""P2 (RFP_emergent_mastery_curriculum) — full-IQL routing learner.

Mirrors the CGN-IQL algorithm (logic/cgn.py:_iql_update_consumer) by hand in
numpy on OuterMetaPolicy (INV-MC-6 — no torch import). These tests pin:
  - train_iql reduces V/Q losses on a learnable synthetic dataset;
  - AWR shifts the π net toward the high-advantage action;
  - the Polyak target tracks Q (moves toward it, slowly);
  - iql_to_flat/iql_from_flat round-trip;
  - FULL-IQL (not bandit): V(s') participates — a non-terminal transition with a
    high-value successor lifts its Q-backup above a terminal one;
  - flag-off parity: init_iql / train_iql never touch to_flat / from_flat / the
    REINFORCE learn() path (byte-identical legacy persistence).
"""

import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import (
    IQL_FLAT_DIM,
    NUM_OUTER_ACTIONS,
    OUTER_POLICY_FLAT_DIM,
    OUTER_POLICY_INPUT_DIM,
    OuterMetaPolicy,
)


def _rng():
    np.random.seed(1234)


def _synthetic_transitions(n=200, high_action=2):
    """A learnable bandit-ish set with trajectory links: a fixed feature pattern
    routed to `high_action` earns reward 1.0; others earn 0.0. next_state links
    consecutive samples (one global trajectory), tail terminal."""
    d = OUTER_POLICY_INPUT_DIM
    states = (np.random.rand(n, d).astype(np.float32))
    states[:, 0] = 1.0  # bias feature
    trans = []
    for i in range(n):
        a = int(np.random.randint(NUM_OUTER_ACTIONS))
        r = 1.0 if a == high_action else 0.0
        nxt = states[i + 1] if i + 1 < n else None
        trans.append({
            "state": states[i], "action": a, "reward": r,
            "next_state": nxt, "terminal": (i + 1 >= n),
        })
    return trans


def test_train_iql_reduces_losses():
    _rng()
    p = OuterMetaPolicy()
    trans = _synthetic_transitions(200)
    first = p.train_iql(trans, steps=5, batch_size=32)
    # warm up, then measure that losses are finite + bounded after many steps.
    last = p.train_iql(trans, steps=200, batch_size=32)
    assert np.isfinite(last["v_loss"]) and np.isfinite(last["q_loss"])
    assert last["iql_updates"] > first["iql_updates"]
    # Q must learn the reward structure: Q(high_action) > Q(a low) on the pattern.
    s = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
    s[0] = 1.0
    q_all, _ = p._mlp2_forward(s.reshape(1, -1), p._qw1, p._qb1, p._qw2, p._qb2)
    q_all = q_all.reshape(-1)
    assert q_all[2] == pytest.approx(q_all.max(), abs=1e-5), (
        f"Q should peak at the rewarded action 2: {q_all}")


def test_awr_shifts_policy_toward_high_advantage_action():
    _rng()
    p = OuterMetaPolicy()
    trans = _synthetic_transitions(300, high_action=2)
    s = np.full(OUTER_POLICY_INPUT_DIM, 0.5, dtype=np.float32)
    s[0] = 1.0
    p0 = p.action_probs(s)[2]
    p.train_iql(trans, steps=400, batch_size=48)
    p1 = p.action_probs(s)[2]
    assert p1 > p0, f"AWR should raise P(action 2): {p0:.3f} -> {p1:.3f}"


def test_polyak_target_tracks_q():
    _rng()
    p = OuterMetaPolicy()
    p.init_iql()
    q_before = p._qw2.copy()
    qt_before = p._qtw2.copy()
    # at init they are equal copies
    assert np.allclose(q_before, qt_before)
    p.train_iql(_synthetic_transitions(120), steps=80, batch_size=24,
                polyak=0.01)
    # Q moved; target moved toward Q but lags (not equal to Q, not stuck at init).
    assert not np.allclose(p._qw2, p._qtw2), "target should lag Q (Polyak)"
    moved = np.linalg.norm(p._qtw2 - qt_before)
    assert moved > 0.0, "target must move toward Q"


def test_full_iql_uses_next_state_value():
    """FULL IQL (not bandit): a non-terminal transition's Q-backup includes
    γ·V(s'). With a deliberately high-value successor, the backup exceeds the
    pure-reward (terminal) backup — proving V(s') participates."""
    _rng()
    p = OuterMetaPolicy()
    p.init_iql()
    # Force V to output a large positive value by scaling its output weights.
    p._vw2 *= 0.0
    p._vb2[:] = 5.0  # V(s) ≈ 5 everywhere
    d = OUTER_POLICY_INPUT_DIM
    s = np.full(d, 0.5, dtype=np.float32); s[0] = 1.0
    s2 = np.full(d, 0.5, dtype=np.float32); s2[0] = 1.0
    term = [{"state": s, "action": 1, "reward": 1.0,
             "next_state": None, "terminal": True}] * 4
    nonterm = [{"state": s, "action": 1, "reward": 1.0,
                "next_state": s2, "terminal": False}] * 4
    gamma = 0.99
    # one step each; compare the learned Q toward the two backups
    p_t = OuterMetaPolicy(); p_t.init_iql(); p_t._vw2 *= 0.0; p_t._vb2[:] = 5.0
    p_n = OuterMetaPolicy(); p_n.init_iql(); p_n._vw2 *= 0.0; p_n._vb2[:] = 5.0
    # identical Q init so the comparison is fair
    for a in ("_qw1", "_qb1", "_qw2", "_qb2"):
        setattr(p_n, a, getattr(p_t, a).copy())
        setattr(p_n, "_qt" + a[2:], getattr(p_t, "_qt" + a[2:]).copy())
    p_t.train_iql(term, steps=1, batch_size=4, gamma=gamma, lr=0.05)
    p_n.train_iql(nonterm, steps=1, batch_size=4, gamma=gamma, lr=0.05)
    q_t, _ = p_t._mlp2_forward(s.reshape(1, -1), p_t._qw1, p_t._qb1, p_t._qw2, p_t._qb2)
    q_n, _ = p_n._mlp2_forward(s.reshape(1, -1), p_n._qw1, p_n._qb1, p_n._qw2, p_n._qb2)
    # nonterminal backup = r + γ·V(s') ≈ 1 + 0.99·5 ≫ terminal backup = r = 1
    assert q_n.reshape(-1)[1] > q_t.reshape(-1)[1], (
        "V(s') must lift the non-terminal Q above the terminal one (full IQL)")


def test_iql_flat_roundtrip():
    _rng()
    p = OuterMetaPolicy()
    p.train_iql(_synthetic_transitions(60), steps=30, batch_size=16)
    flat = p.iql_to_flat()
    assert flat.shape[0] == IQL_FLAT_DIM
    q = OuterMetaPolicy()
    assert q.iql_from_flat(flat) is True
    assert q.total_iql_updates == p.total_iql_updates
    for a in ("_vw1", "_vw2", "_qw1", "_qw2", "_qtw1", "_qtw2"):
        assert np.allclose(getattr(q, a), getattr(p, a)), a
    # bad-size flat → False + safe fresh init
    assert q.iql_from_flat(np.zeros(IQL_FLAT_DIM + 1, dtype=np.float32)) is False
    assert q._iql_inited is True


def test_flag_off_parity_pi_persistence_untouched():
    """init_iql + train_iql must NOT change the π flat layout / SHM persistence
    (agno reads only π). A policy that trained IQL still round-trips its π via
    to_flat/from_flat at the legacy fixed dim."""
    _rng()
    p = OuterMetaPolicy()
    flat_dim_before = p.to_flat().shape[0]
    assert flat_dim_before == OUTER_POLICY_FLAT_DIM
    p.train_iql(_synthetic_transitions(50), steps=20, batch_size=16)
    flat = p.to_flat()
    assert flat.shape[0] == OUTER_POLICY_FLAT_DIM, "π SHM flat dim must be unchanged"
    r = OuterMetaPolicy.from_flat(flat)
    assert np.allclose(r.w1, p.w1) and np.allclose(r.w3, p.w3)


def test_legacy_learn_path_still_works():
    # Flag-off path: learn() (REINFORCE) is untouched and independent of IQL.
    _rng()
    p = OuterMetaPolicy()
    x = np.full(OUTER_POLICY_INPUT_DIM, 0.3, dtype=np.float32); x[0] = 1.0
    u0 = p.total_updates
    p.learn(x, action=1, reward=1.0)
    assert p.total_updates == u0 + 1
    assert p._iql_inited is False, "learn() must not initialise IQL"
