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
    c = load_affective_config()
    assert c.enabled is False                              # spine OFF until proven


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
