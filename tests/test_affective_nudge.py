"""Affective Grounding Loop §7.A — Phase A offline tests.

Covers the RFP §8 gates that are verifiable offline:
  G2  symmetric/honest  — a below-baseline (worse-competence) pass yields a
                          NEGATIVE-valence nudge; above-baseline a positive one.
  G3  habituation        — a repeated deviation shrinks the nudge as the baseline
                          catches up (no hardcoded decay; emerges from the EMA).
  G4  per-Titan divergence — two independent baselines (separate state files) →
                          different signal→affect sensitivity for the same probe.
  G6  composes-not-clobbers / flag-off ≡ baseline — config defaults OFF; the
                          worker's nudge block is gated on `enabled` (here we
                          assert the default), so flag-off changes nothing.
Plus the honest "no real movement → no nudge" cases (no outcomes / cold baseline
/ exactly-on-baseline) and the drain-tally additivity the loop depends on.
"""
import duckdb
import pytest

from titan_hcl.logic.affective_nudge import (
    AffectiveConfig, Nudge, compute_skill_score_nudge, load_affective_config,
    SKILL_SCORE_MODULATOR,
)
from titan_hcl.synthesis.skill_store import ProceduralSkillStore


CFG = AffectiveConfig(enabled=True, k_surprise=0.04, max_mag=0.06,
                      ema_alpha=0.15, sigma_init=0.25, eps=1e-3, min_samples=2)


def _path(tmp_path, name="aff.json"):
    return str(tmp_path / name)


def _warm(path, rate_succ, rate_fail, n=2):
    """Fold `n` identical observations to get the baseline past min_samples."""
    for _ in range(n):
        compute_skill_score_nudge(rate_succ, rate_fail, path, cfg=CFG)


# ── honest "no real movement → no nudge" ─────────────────────────────────────

def test_no_outcomes_returns_none(tmp_path):
    assert compute_skill_score_nudge(0, 0, _path(tmp_path), cfg=CFG) is None


def test_first_observation_seeds_but_emits_nothing(tmp_path):
    p = _path(tmp_path)
    # First ever: no prior → surprise undefined → claiming a magnitude would be a
    # hardcoded nudge. Must seed the baseline and return None.
    assert compute_skill_score_nudge(3, 1, p, cfg=CFG) is None
    import json
    with open(p) as f:
        st = json.load(f)
    assert st["skill_score"]["n"] == 1
    assert st["skill_score"]["mu"] == pytest.approx(0.75)


def test_warming_up_emits_nothing(tmp_path):
    p = _path(tmp_path)
    compute_skill_score_nudge(1, 1, p, cfg=CFG)            # seed (n=1)
    # n_before == 1 < min_samples(2): even a deviation must not nudge yet.
    assert compute_skill_score_nudge(4, 0, p, cfg=CFG) is None


def test_exactly_on_baseline_no_nudge(tmp_path):
    p = _path(tmp_path)
    _warm(p, 1, 1, n=2)                                    # baseline μ≈0.5, n=2
    # same 0.5 rate → deviation 0 → no real movement → None
    assert compute_skill_score_nudge(1, 1, p, cfg=CFG) is None


# ── G2: honest + symmetric valence ───────────────────────────────────────────

def test_above_baseline_positive_valence_pulls_DA_up(tmp_path):
    p = _path(tmp_path)
    _warm(p, 1, 1, n=2)                                    # μ≈0.5
    n = compute_skill_score_nudge(10, 0, p, cfg=CFG)       # rate 1.0 > μ
    assert isinstance(n, Nudge)
    assert n.valence == +1
    assert n.target == 1.0                                 # pull DA toward max
    assert n.magnitude > 0.0


def test_below_baseline_negative_valence_pulls_DA_down(tmp_path):
    p = _path(tmp_path)
    _warm(p, 1, 1, n=2)                                    # μ≈0.5
    n = compute_skill_score_nudge(0, 10, p, cfg=CFG)       # rate 0.0 < μ
    assert isinstance(n, Nudge)
    assert n.valence == -1
    assert n.target == 0.0                                 # pull DA toward floor
    assert n.magnitude > 0.0


# ── magnitude is surprise-scaled and capped (INV-AFF-EMERGENT, gentle) ────────

def test_magnitude_capped_at_max_mag(tmp_path):
    p = _path(tmp_path)
    _warm(p, 0, 2, n=2)                                    # μ≈0.0
    n = compute_skill_score_nudge(10, 0, p, cfg=CFG)       # rate 1.0: huge surprise
    assert n is not None
    assert n.magnitude == pytest.approx(CFG.max_mag)       # clamped, never a spike


def test_magnitude_scales_with_surprise(tmp_path):
    # A bigger deviation from the same baseline → a bigger (pre-cap) nudge.
    p_small = _path(tmp_path, "s.json")
    p_big = _path(tmp_path, "b.json")
    _warm(p_small, 1, 1, n=2)                              # μ≈0.5
    _warm(p_big, 1, 1, n=2)                                # μ≈0.5
    small = compute_skill_score_nudge(6, 4, p_small, cfg=CFG)   # rate 0.6
    big = compute_skill_score_nudge(10, 0, p_big, cfg=CFG)      # rate 1.0
    assert small is not None and big is not None
    assert big.surprise > small.surprise
    assert big.magnitude >= small.magnitude


# ── G3: habituation emerges (no hardcoded decay) ─────────────────────────────

def test_habituation_shrinks_repeated_deviation(tmp_path):
    p = _path(tmp_path)
    _warm(p, 1, 1, n=2)                                    # μ≈0.5
    mags = []
    for _ in range(30):
        n = compute_skill_score_nudge(10, 0, p, cfg=CFG)   # keep hitting rate 1.0
        mags.append(n.magnitude if n is not None else 0.0)
    # As μ creeps toward 1.0 the deviation collapses → the nudge fades to ~0.
    assert mags[0] > 0.0
    assert mags[-1] < mags[0] * 0.5                        # at least halved
    assert mags[-1] < 0.01                                 # effectively habituated


# ── G4: per-Titan divergence (independent baselines, no shared weights) ───────

def test_per_titan_divergence(tmp_path):
    # Titan A has lived mostly-success (high μ); Titan B mostly-failure (low μ).
    pa = _path(tmp_path, "titanA.json")
    pb = _path(tmp_path, "titanB.json")
    for _ in range(6):
        compute_skill_score_nudge(9, 1, pa, cfg=CFG)       # A: rate ~0.9
        compute_skill_score_nudge(1, 9, pb, cfg=CFG)       # B: rate ~0.1
    # Same probe (a perfect pass) lands very differently per Titan:
    na = compute_skill_score_nudge(10, 0, pa, cfg=CFG)     # near A's baseline → small/None
    nb = compute_skill_score_nudge(10, 0, pb, cfg=CFG)     # far above B's baseline → strong
    mag_a = na.magnitude if na is not None else 0.0
    mag_b = nb.magnitude if nb is not None else 0.0
    assert mag_b > mag_a                                   # B still moved by it, A blasé


# ── persistence: baseline survives across calls (file-backed) ────────────────

def test_state_persists_across_calls(tmp_path):
    p = _path(tmp_path)
    compute_skill_score_nudge(1, 1, p, cfg=CFG)
    import json
    with open(p) as f:
        n_after_one = json.load(f)["skill_score"]["n"]
    compute_skill_score_nudge(1, 1, p, cfg=CFG)
    with open(p) as f:
        n_after_two = json.load(f)["skill_score"]["n"]
    assert n_after_two == n_after_one + 1                  # folded, not reset


# ── G6: config defaults OFF + the modulator mapping is DA ─────────────────────

def test_config_default_disabled():
    # The SAFETY invariant (RFP §5) is the in-CODE default — a fresh install must
    # boot with the loop OFF. The live titan_params.toml is deliberately flipped
    # ON for the fleet soak (2026-06-13); that's the intended runtime state, not a
    # regression — so assert the dataclass default, not the live-file value.
    assert AffectiveConfig().enabled is False              # fresh-install OFF until proven


def test_skill_score_maps_to_DA():
    # RFP D4: achievement/competence → DA.
    assert SKILL_SCORE_MODULATOR == "DA"


# ── drain-tally additivity the loop depends on (INV: purely additive) ─────────

def _store(tmp_path):
    conn = duckdb.connect(":memory:")
    return ProceduralSkillStore(
        duckdb_conn=conn,
        faiss_path=str(tmp_path / "skills.faiss"),
        snapshot_path=str(tmp_path / "skills_snapshot.json"),
        embedder=None,
    )


def test_drain_reports_success_failure_tally(tmp_path):
    store = _store(tmp_path)
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi", success=True,
        parent_tool_call_tx="tx_ok1")
    store.enqueue_score_event(
        oracle_id="web_api_oracle", goal_class="defi-lookup",
        task_shape="informational|searxng-search|defi2", success=True,
        parent_tool_call_tx="tx_ok2")
    store.enqueue_score_event(
        oracle_id="coding_sandbox", goal_class="impl",
        task_shape="generative|code|x", success=False,
        parent_tool_call_tx="tx_bad1")
    summary = store.drain_score_events()
    assert summary["drained"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1


def test_empty_drain_tally_zero(tmp_path):
    store = _store(tmp_path)
    summary = store.drain_score_events()
    assert summary["drained"] == 0
    assert summary["successes"] == 0
    assert summary["failures"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# PHASE B — the emergent AffectiveNudgeNet (RFP §7.B)
# ═════════════════════════════════════════════════════════════════════════════
import numpy as np

from titan_hcl.logic.affective_nudge import (
    AffectiveNudgeNet, AffectiveNudgeRuntime, FEATURE_DIM, build_features,
    signed_delta_target,
)

CFGB = AffectiveConfig(enabled=True, k_surprise=0.04, max_mag=0.06,
                       ema_alpha=0.15, sigma_init=0.25, eps=1e-3, min_samples=2,
                       net_enabled=True, net_lr=0.05, net_l2=1e-4, net_hidden=16)


class _Emot:
    """Mutable fake emot read_state — set .v / .idx to simulate live emot drift
    (attribution rides V_blended; see signed_delta_target). Callable like the
    injected emot_state_reader."""
    def __init__(self, v_blended=0.3, dominant_idx=0):
        self.v = v_blended
        self.idx = dominant_idx

    def __call__(self):
        return {"V_blended": self.v, "dominant_idx": self.idx}


def _emot(v_blended=0.3, dominant_idx=0):
    return _Emot(v_blended, dominant_idx)


# ── the net: forward / numpy-SGD backprop / persistence ──────────────────────

def test_net_forward_shape_scalar():
    net = AffectiveNudgeNet(hidden=16, seed=0)
    out = net.forward(np.zeros(FEATURE_DIM))
    assert isinstance(out, float)
    assert net.W1.shape == (16, FEATURE_DIM)
    assert net.W2.shape == (1, 16)


def test_net_untrained_then_trained_flag():
    net = AffectiveNudgeNet(hidden=8, seed=1)
    assert net.is_trained is False
    net.train_step(np.ones((3, FEATURE_DIM)), np.zeros(3), lr=0.05)
    assert net.is_trained is True
    assert net.trained_steps == 1


def test_net_sgd_reduces_mse_toward_target():
    # The numpy backprop must actually learn: fit a fixed (X→y) batch and watch
    # the loss fall monotonically-ish to near zero.
    rng = np.random.default_rng(7)
    X = rng.random((16, FEATURE_DIM))
    w_true = rng.standard_normal(FEATURE_DIM) * 0.1
    y = X @ w_true                                           # a learnable (linear) signal
    net = AffectiveNudgeNet(hidden=16, seed=2)
    first = net.train_step(X, y, lr=0.1)
    for _ in range(1500):
        last = net.train_step(X, y, lr=0.1)
    assert last < first * 0.1                                # ≥10× loss reduction = genuine learning
    assert last < 1e-3                                       # genuinely fit a learnable target


def test_net_learns_constant_magnitude_target():
    # Habituation basis: a flattened observed drift (target→small) trains the
    # predicted magnitude DOWN; a large drift trains it UP. Train two nets on
    # constant targets and confirm forward tracks the target.
    rng = np.random.default_rng(3)
    X = rng.random((8, FEATURE_DIM))
    lo = AffectiveNudgeNet(hidden=16, seed=4)
    hi = AffectiveNudgeNet(hidden=16, seed=4)
    for _ in range(500):
        lo.train_step(X, np.zeros(8), lr=0.05)
        hi.train_step(X, np.full(8, 0.5), lr=0.05)
    probe = X[0]
    assert abs(lo.forward(probe)) < 0.05                    # learned ~0 (habituated)
    assert hi.forward(probe) > 0.3                           # learned strong


def test_net_npz_roundtrip(tmp_path):
    net = AffectiveNudgeNet(hidden=12, seed=5)
    net.train_step(np.ones((4, FEATURE_DIM)), np.full(4, 0.2), lr=0.05)
    path = str(tmp_path / "affective" / "net.npz")
    net.save_npz(path)
    probe = np.linspace(0, 1, FEATURE_DIM)
    loaded = AffectiveNudgeNet.load_npz(path, hidden=12)
    assert loaded.trained_steps == net.trained_steps
    assert loaded.forward(probe) == pytest.approx(net.forward(probe), abs=1e-9)
    assert loaded.is_trained is True


def test_net_load_missing_returns_fresh(tmp_path):
    net = AffectiveNudgeNet.load_npz(str(tmp_path / "nope.npz"), hidden=10)
    assert net.is_trained is False
    assert net.hidden == 10


# ── features + attribution target ────────────────────────────────────────────

def test_build_features_layout():
    n = Nudge(magnitude=0.05, target=1.0, surprise=10.0, valence=1,
              rate=0.9, mu_before=0.5, n=4)
    f = build_features(n, {"V_blended": 0.4, "dominant_idx": 2})
    assert f.shape == (FEATURE_DIM,)
    assert f[0] == pytest.approx(min(3.0, 10.0 / 5.0))      # surprise scaled
    assert f[2] == pytest.approx(abs(0.9 - 0.5))            # |deviation|
    assert f[3] == pytest.approx(0.4)                       # V_blended
    assert f[4 + 2] == 1.0                                   # dominant one-hot @ idx 2
    assert f[4:12].sum() == 1.0                              # exactly one dominant


def test_build_features_missing_emot_safe():
    n = Nudge(magnitude=0.05, target=0.0, surprise=2.0, valence=-1,
              rate=0.1, mu_before=0.5, n=3)
    f = build_features(n, None)                              # emot SHM unavailable
    assert f.shape == (FEATURE_DIM,)
    assert f[3] == 0.0
    assert f[4:12].sum() == 0.0                              # no dominant claimed


def test_signed_delta_target_sign_and_magnitude():
    # Attribution = signed delta of the dominant blended valence (V_blended scalar).
    assert signed_delta_target(0.1, 0.3) == pytest.approx(0.2)   # rose → +
    assert signed_delta_target(0.3, 0.1) == pytest.approx(-0.2)  # fell → −
    assert signed_delta_target(0.2, None) is None                # missing → skip
    assert signed_delta_target(None, 0.3) is None
    assert signed_delta_target(0.2, 0.2) == 0.0                  # no movement


# ── the runtime: cold-start fallback → net takeover, pending attribution ──────

def _runtime(tmp_path, name="r", emot_v=0.3):
    d = tmp_path / name
    return AffectiveNudgeRuntime(
        CFGB,
        str(d / "state.json"),
        str(d / "net.npz"),
        emot_state_reader=_emot(emot_v),
    )


def test_runtime_cold_start_uses_formula_magnitude(tmp_path):
    # Untrained net → observe_drain must return the SAME magnitude the Phase-A
    # formula would (cold-start fallback, no regression vs Phase A).
    rt = _runtime(tmp_path, "rt")
    direct = _path(tmp_path, "direct.json")
    # warm both baselines identically (n past min_samples)
    for _ in range(2):
        rt.observe_drain(1, 1, ts=0.0)
        compute_skill_score_nudge(1, 1, direct, cfg=CFGB)
    assert rt.net.is_trained is False
    got = rt.observe_drain(10, 0, ts=1.0)
    exp = compute_skill_score_nudge(10, 0, direct, cfg=CFGB)
    assert got is not None and exp is not None
    assert got.magnitude == pytest.approx(exp.magnitude)    # formula, not net


def test_runtime_records_pending_and_trains_on_dream(tmp_path):
    emot = _emot(0.1)
    rt = AffectiveNudgeRuntime(CFGB, str(tmp_path / "s.json"),
                               str(tmp_path / "n.npz"), emot_state_reader=emot)
    for _ in range(2):
        rt.observe_drain(1, 1, ts=0.0)                       # warm baseline
    rt.observe_drain(10, 0, ts=1.0)                          # a real nudge fires
    assert len(rt._pending) >= 1
    emot.v = 0.4                                             # emot valence drifted post-cycle
    summary = rt.train_on_dream()
    assert summary["trained"] >= 1
    assert rt.net.is_trained is True
    assert rt._pending == []                                 # buffer cleared


def test_runtime_net_takes_over_after_training(tmp_path):
    emot = _emot(0.1)
    rt = AffectiveNudgeRuntime(CFGB, str(tmp_path / "s.json"),
                               str(tmp_path / "n.npz"), emot_state_reader=emot)
    for _ in range(2):
        rt.observe_drain(1, 1, ts=0.0)
    rt.observe_drain(9, 1, ts=1.0)
    emot.v = 0.35                                            # post-cycle drift
    rt.train_on_dream()
    assert rt.net.is_trained
    # next nudge magnitude now comes from the net (still clamped gentle)
    got = rt.observe_drain(10, 0, ts=2.0)
    assert got is not None
    assert 0.0 < got.magnitude <= CFGB.max_mag


def test_runtime_train_no_pending_is_noop(tmp_path):
    rt = _runtime(tmp_path, "rt4")
    summary = rt.train_on_dream()
    assert summary.get("trained", 0) == 0
    assert rt.net.is_trained is False


def test_runtime_train_skips_when_no_emot(tmp_path):
    # emot SHM unavailable → no pre/post valence snapshot → cannot attribute →
    # skip (never fabricate a delta, INV-AFF-HONEST).
    rt = AffectiveNudgeRuntime(
        CFGB, str(tmp_path / "s.json"), str(tmp_path / "n.npz"),
        emot_state_reader=lambda: None)
    for _ in range(2):
        rt.observe_drain(1, 1, ts=0.0)
    rt.observe_drain(10, 0, ts=1.0)
    summary = rt.train_on_dream()
    assert summary["trained"] == 0
    assert summary["skipped"] >= 1
    assert rt.net.is_trained is False


def test_runtime_per_titan_net_divergence(tmp_path):
    # Same nudge stimulus, DIFFERENT emot trajectories → divergent net weights
    # (INV-AFF-SELF-SOVEREIGN; no shared weights).
    ea, eb = _emot(0.1), _emot(0.1)
    a = AffectiveNudgeRuntime(CFGB, str(tmp_path / "A.json"),
                              str(tmp_path / "A.npz"), emot_state_reader=ea)
    b = AffectiveNudgeRuntime(CFGB, str(tmp_path / "B.json"),
                              str(tmp_path / "B.npz"), emot_state_reader=eb)
    for rt, emot, post in ((a, ea, 0.5), (b, eb, 0.12)):     # A's valence moves a lot; B's barely
        for _ in range(2):
            rt.observe_drain(1, 1, ts=0.0)
        for k in range(5):
            emot.v = 0.1
            rt.observe_drain(9, 1, ts=float(k + 1))
            emot.v = post                                    # post-cycle drift differs per Titan
            rt.train_on_dream()
    probe = build_features(
        Nudge(0.05, 1.0, 5.0, 1, 0.9, 0.5, 6), {"V_blended": 0.3, "dominant_idx": 0})
    assert a.net.forward(probe) != b.net.forward(probe)     # diverged


# ─────────────────────────────────────────────────────────────────────────────
# §7.C — broadened signals: signal_type one-hot + compute_event_nudge + the
# generalized runtime observe_signal path. (sol_receipt + maker_bond = Tier 1.)
# ─────────────────────────────────────────────────────────────────────────────

from titan_hcl.logic.affective_nudge import (   # noqa: E402
    SIGNAL_TYPES, SIGNAL_INDEX, N_SIGNAL_TYPES, SIGNAL_MODULATOR,
    compute_event_nudge, _BASE_FEATURE_DIM,
)


def test_feature_dim_grows_by_signal_count():
    assert FEATURE_DIM == _BASE_FEATURE_DIM + N_SIGNAL_TYPES
    assert N_SIGNAL_TYPES == len(SIGNAL_TYPES)
    # stable positional order — skill_score MUST stay index 0 (saved-net columns).
    assert SIGNAL_INDEX["skill_score"] == 0


def test_build_features_signal_one_hot_per_type():
    n = Nudge(magnitude=0.05, target=1.0, surprise=5.0, valence=1,
              rate=0.9, mu_before=0.5, n=4)
    for st in SIGNAL_TYPES:
        f = build_features(n, {"V_blended": 0.2, "dominant_idx": 1}, signal_type=st)
        assert f.shape == (FEATURE_DIM,)
        # exactly one signal one-hot, at the right index, in the [12:] block.
        assert f[_BASE_FEATURE_DIM:].sum() == 1.0
        assert f[_BASE_FEATURE_DIM + SIGNAL_INDEX[st]] == 1.0
        # the base block is untouched by the signal one-hot.
        assert f[4:12].sum() == 1.0          # dominant one-hot still there


def test_build_features_unknown_signal_no_one_hot():
    n = Nudge(0.05, 1.0, 5.0, 1, 0.9, 0.5, 4)
    f = build_features(n, None, signal_type="not_a_signal")
    assert f[_BASE_FEATURE_DIM:].sum() == 0.0   # no signal claimed (robust)


def test_all_signals_map_to_DA_phase_c_v1():
    # Maker-decided: all Phase C signals → DA in v1 (multi-modulator = C.2).
    assert set(SIGNAL_MODULATOR.values()) == {"DA"}
    assert SIGNAL_MODULATOR["sol_receipt"] == "DA"
    assert SIGNAL_MODULATOR["maker_bond"] == "DA"


def test_event_nudge_zero_delta_none(tmp_path):
    assert compute_event_nudge(0.0, _path(tmp_path), signal_type="sol_receipt",
                               cfg=CFG) is None


def test_event_nudge_first_obs_seeds_then_emits(tmp_path):
    p = _path(tmp_path)
    # first ever → seed only (no prior → no definable surprise)
    assert compute_event_nudge(0.05, p, signal_type="sol_receipt", cfg=CFG) is None
    # warm past min_samples
    compute_event_nudge(0.05, p, signal_type="sol_receipt", cfg=CFG)
    n = compute_event_nudge(0.5, p, signal_type="sol_receipt", cfg=CFG)  # bigger receipt
    assert n is not None and n.magnitude > 0.0


def test_sol_receipt_symmetric_valence(tmp_path):
    # A receipt (+Δ) → +valence (pull DA up); a spend (−Δ) → −valence.
    p_pos = _path(tmp_path, "pos.json")
    for _ in range(2):
        compute_event_nudge(0.01, p_pos, signal_type="sol_receipt", cfg=CFG)
    up = compute_event_nudge(0.5, p_pos, signal_type="sol_receipt", cfg=CFG)
    assert up is not None and up.valence == 1 and up.target == 1.0

    p_neg = _path(tmp_path, "neg.json")
    for _ in range(2):
        compute_event_nudge(-0.01, p_neg, signal_type="sol_receipt", cfg=CFG)
    down = compute_event_nudge(-0.5, p_neg, signal_type="sol_receipt", cfg=CFG)
    assert down is not None and down.valence == -1 and down.target == 0.0


def test_maker_bond_intrinsic_positive(tmp_path):
    # maker_bond forces + valence even though the magnitude EMA is direction-blind.
    p = _path(tmp_path)
    for _ in range(2):
        compute_event_nudge(0.01, p, signal_type="maker_bond", cfg=CFG,
                            intrinsic_positive=True)
    n = compute_event_nudge(0.5, p, signal_type="maker_bond", cfg=CFG,
                            intrinsic_positive=True)
    assert n is not None and n.valence == 1


def test_event_signal_habituation_per_signal(tmp_path):
    # Warm the baseline at a SMALL magnitude, then fire a LARGE receipt repeatedly:
    # the first deviates strongly (big nudge), and as μ catches up to the large
    # magnitude the nudge shrinks → habituation emerges per-signal (no hardcoded
    # decay). A constant-from-cold stream never deviates (μ seeds to it), so the
    # deviation must come from a baseline shift — exactly the EMA mechanic.
    p = _path(tmp_path)
    for _ in range(2):
        compute_event_nudge(0.001, p, signal_type="sol_receipt", cfg=CFG)  # small
    mags = []
    for _ in range(30):
        n = compute_event_nudge(0.5, p, signal_type="sol_receipt", cfg=CFG)  # big
        if n is not None:
            mags.append(n.magnitude)
    assert mags, "expected at least one nudge before habituation"
    assert mags[-1] < mags[0]             # shrinks as μ catches up to the big size


def test_event_signals_keep_independent_baselines(tmp_path):
    # sol_receipt and maker_bond share ONE state file but separate _SignalBaseline
    # keys → folding one does not move the other's surprise.
    p = _path(tmp_path)
    for _ in range(3):
        compute_event_nudge(0.2, p, signal_type="sol_receipt", cfg=CFG)
    # maker_bond still cold here → first real obs seeds, emits None
    assert compute_event_nudge(0.2, p, signal_type="maker_bond", cfg=CFG,
                               intrinsic_positive=True) is None
    import json
    state = json.loads(open(p).read())
    assert "sol_receipt" in state and "maker_bond" in state
    assert state["sol_receipt"]["n"] >= 3 and state["maker_bond"]["n"] == 1


def test_runtime_observe_signal_records_pending_with_one_hot(tmp_path):
    emot = _emot(0.1)
    rt = AffectiveNudgeRuntime(CFGB, str(tmp_path / "s.json"),
                               str(tmp_path / "n.npz"), emot_state_reader=emot)
    for _ in range(2):
        rt.observe_signal("sol_receipt", 0.01, ts=0.0)     # warm
    got = rt.observe_signal("sol_receipt", 0.6, ts=1.0)    # real event
    assert got is not None and got.magnitude > 0.0
    assert len(rt._pending) >= 1
    # the pending features carry the sol_receipt one-hot
    feat = rt._pending[-1].features
    assert feat[_BASE_FEATURE_DIM + SIGNAL_INDEX["sol_receipt"]] == 1.0


def test_net_load_dim_guard_discards_incompatible(tmp_path):
    # A pre-Phase-C 12-D net on disk must be discarded (fresh 17-D net), not crash.
    p = str(tmp_path / "old.npz")
    old = AffectiveNudgeNet(hidden=16, in_dim=12)   # legacy dim
    old.train_step(np.ones((2, 12)), np.zeros(2), lr=0.05)
    old.save_npz(p)
    fresh = AffectiveNudgeNet.load_npz(p, hidden=16)
    assert fresh.in_dim == FEATURE_DIM               # rebuilt at the new dim
    assert fresh.is_trained is False                 # discarded the stale weights


def test_skill_score_path_byte_identical_after_refactor(tmp_path):
    # The §7.C refactor must not change skill_score behaviour: a known scenario
    # yields the same valence/target/sign as the documented Phase-A contract.
    p = _path(tmp_path)
    for _ in range(2):
        compute_skill_score_nudge(1, 1, p, cfg=CFG)     # baseline rate 0.5
    up = compute_skill_score_nudge(10, 0, p, cfg=CFG)   # rate 1.0 > μ
    assert up is not None and up.valence == 1 and up.target == 1.0
    assert up.rate == pytest.approx(1.0)


# ── §7.C Tier 2: cross-process source readers (events_teacher / inner_memory) ──

from titan_hcl.logic.affective_nudge import (   # noqa: E402
    read_engagement_delta,
)


def _mk_engagement_db(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE engagement_snapshots (id INTEGER PRIMARY KEY, "
              "tweet_id TEXT, delta_likes INTEGER, delta_replies INTEGER, "
              "delta_quotes INTEGER, checked_at REAL)")
    for tid, dl, dr, dq, ts in rows:
        c.execute("INSERT INTO engagement_snapshots (tweet_id, delta_likes, "
                  "delta_replies, delta_quotes, checked_at) VALUES (?,?,?,?,?)",
                  (tid, dl, dr, dq, ts))
    c.commit(); c.close()


def test_read_engagement_delta_establishes_then_sums(tmp_path):
    p = str(tmp_path / "events_teacher.db")
    _mk_engagement_db(p, [("t1", 3, 1, 0, 100.0), ("t2", 5, 0, 2, 200.0)])
    # first call (cursor None) → establish at MAX(checked_at), NO historical flood
    delta, cur = read_engagement_delta(p, None)
    assert delta == 0 and cur == 200.0
    # a new snapshot after the cursor → its aggregate delta, cursor advances
    import sqlite3
    c = sqlite3.connect(p)
    c.execute("INSERT INTO engagement_snapshots (tweet_id, delta_likes, "
              "delta_replies, delta_quotes, checked_at) VALUES ('t3',4,2,1,300.0)")
    c.commit(); c.close()
    delta, cur = read_engagement_delta(p, cur)
    assert delta == 7 and cur == 300.0
    # no new snapshots → delta 0, cursor unchanged
    delta, cur = read_engagement_delta(p, cur)
    assert delta == 0 and cur == 300.0


def test_read_engagement_delta_missing_db_soft(tmp_path):
    # missing db → soft (0, last_cursor or 0.0); never raises
    assert read_engagement_delta(str(tmp_path / "nope.db"), 50.0) == (0, 50.0)
    assert read_engagement_delta(str(tmp_path / "nope.db"), None) == (0, 0.0)


def test_chat_reuse_counter_drains(tmp_path):
    # chain_reuse (repointed to outer agno reuse): note_chat_reuse accumulates
    # skill_delegate turns; the drain loop drains the count (+resets) → one
    # chain_reuse event per drain tick. Thread-safe single-consumer.
    rt = _runtime(tmp_path, "reuse")
    assert rt.drain_chat_reuse() == 0          # nothing yet
    rt.note_chat_reuse()                        # one skill_delegate turn
    rt.note_chat_reuse(2)                        # two more (n arg)
    assert rt.drain_chat_reuse() == 3           # accumulated
    assert rt.drain_chat_reuse() == 0           # reset after drain
    rt.note_chat_reuse(0)                        # no-op (guard)
    rt.note_chat_reuse(-5)                       # no-op (guard)
    assert rt.drain_chat_reuse() == 0


def test_tier2_signals_event_nudge_path(tmp_path):
    # x_engagement / chain_reuse use compute_event_nudge with intrinsic_positive →
    # always + valence, magnitude from the per-signal EMA (same proven path).
    for sig in ("x_engagement", "chain_reuse"):
        p = _path(tmp_path, f"{sig}.json")
        for _ in range(2):
            compute_event_nudge(2.0, p, signal_type=sig, cfg=CFG,
                                intrinsic_positive=True)
        n = compute_event_nudge(20.0, p, signal_type=sig, cfg=CFG,
                                intrinsic_positive=True)
        assert n is not None and n.valence == 1 and n.target == 1.0
