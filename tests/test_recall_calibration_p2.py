"""P2 self-calibrating recall floors — gibberish baseline (RFP_synthesis_decision_authority).

Pins the noise-floor calibration: a gibberish probe's top cosine sets the floors
so real recall (above the noise floor) → Sovereign and at-noise → Collaborative,
adapting to the embedder's compressed-high band (D5) instead of a fixed floor.
"""
from titan_hcl.synthesis.recall_calibration import (
    GIBBERISH_PROMPTS,
    GibberishBaseline,
)


def test_cold_start_seed():
    b = GibberishBaseline(initial_ceiling=0.74, margin=0.04)
    known, present = b.floors()
    assert present == 0.74 and round(known, 4) == 0.78
    assert b.samples == 0


def test_first_sample_replaces_seed():
    b = GibberishBaseline(initial_ceiling=0.74, margin=0.04, ema_alpha=0.3)
    b.update(0.72)  # live gibberish floor
    assert b.ceiling == 0.72  # first sample replaces the cold-start seed
    known, present = b.floors()
    assert present == 0.72 and round(known, 4) == 0.76
    assert b.samples == 1


def test_ema_smooths_subsequent_samples():
    b = GibberishBaseline(initial_ceiling=0.74, margin=0.04, ema_alpha=0.5)
    b.update(0.70)            # first → replace → 0.70
    b.update(0.80)            # EMA: 0.5*0.70 + 0.5*0.80 = 0.75
    assert round(b.ceiling, 4) == 0.75
    assert b.samples == 2


def test_floors_track_a_high_noise_floor():
    """If the embedder's gibberish floor is high (0.76), the known floor rises
    with it — the D5 'compressed band' case the fixed 0.65 floor mishandled."""
    b = GibberishBaseline(margin=0.04)
    b.update(0.76)
    known, present = b.floors()
    assert present == 0.76 and round(known, 4) == 0.80
    # A real-recall cosine of 0.82 clears 'known' → Sovereign; 0.77 is between
    # present and known → Collaborative; 0.70 is below noise → research.
    assert 0.82 >= known
    assert present <= 0.77 < known
    assert 0.70 < present


def test_clamped_to_unit_interval():
    b = GibberishBaseline(initial_ceiling=0.98, margin=0.10)
    known, present = b.floors()
    assert known == 1.0 and present == 0.98
    b2 = GibberishBaseline(initial_ceiling=0.74)
    b2.update(-0.5)
    assert b2.ceiling == 0.0


def test_garbage_input_safe():
    b = GibberishBaseline()
    b.update("not-a-number")
    assert 0.0 <= b.ceiling <= 1.0


def test_snapshot_shape():
    b = GibberishBaseline(initial_ceiling=0.74, margin=0.04)
    b.update(0.73)
    snap = b.snapshot()
    assert snap["gibberish_ceiling"] == 0.73
    assert snap["present_floor"] == 0.73
    assert snap["known_floor"] == 0.77
    assert snap["samples"] == 1


def test_gibberish_prompts_are_nonempty_strings():
    assert len(GIBBERISH_PROMPTS) >= 1
    assert all(isinstance(p, str) and p.strip() for p in GIBBERISH_PROMPTS)


# ── P2 decision table — calibrated floors route on cosine RELEVANCE ───────────
# (RFP §7.P2.3: the decision is now keyed on the EngineRecall cosine; the
# relevance-blind regression — gibberish → Sovereign — is killed.)

def _thresholds_from(baseline):
    from titan_hcl.logic.sage.grounded_router import RouterThresholds
    known, present = baseline.floors()
    return RouterThresholds(
        recall_known_floor=known, recall_present_floor=present,
        engram_ground_floor=0.30, skill_promote_floor=0.70)


def _route(cosine, thr, *, informational=True):
    from titan_hcl.logic.sage.grounded_router import (
        GroundedReadout, grounded_route,
    )
    r = GroundedReadout(
        recall_score=cosine, engram_ground=0.0, skill_utility=None,
        requires_tool=False, is_informational=informational,
        can_afford_research=True)
    return grounded_route(r, thr).mode


def test_known_cosine_routes_sovereign():
    # Embedder gibberish floor ~0.74 (the compressed-high D5 band).
    b = GibberishBaseline(margin=0.04)
    b.update(0.74)                       # ceiling 0.74 → floors 0.78 / 0.74
    thr = _thresholds_from(b)
    assert _route(0.82, thr) == "Sovereign"   # 0.82 ≥ known 0.78 → strong substrate


def test_partial_cosine_routes_collaborative():
    b = GibberishBaseline(margin=0.04)
    b.update(0.74)                       # floors 0.78 / 0.74
    thr = _thresholds_from(b)
    # 0.76 is between present (0.74) and known (0.78) → Collaborative.
    assert _route(0.76, thr) == "Collaborative"


def test_gibberish_cosine_routes_research_not_sovereign():
    """THE relevance-blind regression: a noise-floor cosine (0.70 < present 0.74)
    must route to Research, NOT Sovereign. The fixed 0.65 floor mis-routed it."""
    b = GibberishBaseline(margin=0.04)
    b.update(0.74)                       # floors 0.78 / 0.74
    thr = _thresholds_from(b)
    mode = _route(0.70, thr)
    assert mode == "STATE_NEED_RESEARCH"
    # And it would have been Sovereign under the legacy fixed 0.65 floor:
    from titan_hcl.logic.sage.grounded_router import RouterThresholds
    legacy = RouterThresholds(recall_known_floor=0.65, recall_present_floor=0.30,
                              engram_ground_floor=0.30, skill_promote_floor=0.70)
    assert _route(0.70, legacy) != "STATE_NEED_RESEARCH"  # legacy = mis-routed
