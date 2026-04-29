"""
Regression tests for BUG #11: inner_spirit 45D had 19 dead dims because
collect_spirit_45d() was called from _publish_spirit_state() without
hormone_levels / hormone_fires / sphere_clocks / unified_spirit_stats /
memory_stats. The 19 dead dims were 67% of CHIT (15 dims consciousness)
+ 40% of ANANDA (15 dims fulfillment) + partial SAT.

Pre-fix: spirit_loop.py:2257 had TODO comment "Populated when hormonal
system wired" — the wiring happened 2026-04-24 as part of the fuller
EMOT-CGN quality effort.
"""
import pytest
from unittest.mock import MagicMock


def test_collect_spirit_45d_with_full_inputs_activates_chit_dims():
    """When hormone_levels + hormone_fires + sphere_clocks are populated,
    the CHIT segment (dims 15-29 within 45D inner_spirit) shows non-zero
    variance across the expected dims (pattern_recognition, truth_seeking,
    attention_depth, reflective_capacity, temporal_awareness, etc)."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    # Realistic inputs that mimic what spirit_worker NOW passes
    hormone_levels = {
        "DA": 0.6, "NE": 0.4, "5HT": 0.7, "ACh": 0.5,
        "Endorphin": 0.3, "GABA": 0.4,
        "CURIOSITY": 0.7, "FOCUS": 0.5, "INSPIRATION": 0.6,
        "IMPULSE": 0.3, "VIGILANCE": 0.4,
    }
    hormone_fires = {
        "INTUITION": 10, "REFLECTION": 5, "CREATIVITY": 8,
        "EMPATHY": 12, "CURIOSITY": 15,
    }
    sphere_clocks = {
        "body": {"pulse_count": 20, "phase": 0.3},
        "mind": {"pulse_count": 15, "phase": 0.7},
        "spirit": {"pulse_count": 10, "phase": 0.5},
    }
    consciousness = {"epoch_count": 1000, "density": 0.6, "trajectory": 0.4,
                     "curvature": 0.2, "dream_quality": 0.5, "fatigue": 0.3}
    topology = {"volume": 2.0, "curvature": 0.3}

    result = collect_spirit_45d(
        current_5d=[0.5, 0.5, 0.5, 0.5, 0.5],
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness=consciousness,
        topology=topology,
        hormone_levels=hormone_levels,
        hormone_fires=hormone_fires,
        sphere_clocks=sphere_clocks,
    )
    assert len(result) == 45

    # CHIT dims that previously returned 0 with None inputs
    chit_pattern_recognition = result[15 + 5]  # chit[5]
    chit_truth_seeking = result[15 + 7]         # chit[7] — uses CURIOSITY hlvl
    chit_attention_depth = result[15 + 8]       # chit[8] — uses FOCUS hlvl
    chit_reflective_capacity = result[15 + 9]   # chit[9] — uses REFLECTION fires
    chit_temporal_awareness = result[15 + 11]   # chit[11] — uses sphere_clocks
    chit_spatial_awareness = result[15 + 12]    # chit[12] — uses topology

    # All should be > 0 now (were 0 pre-fix)
    assert chit_pattern_recognition > 0, "CHIT[5] pattern_recognition should light up with hormone_fires.INTUITION"
    assert chit_truth_seeking > 0, "CHIT[7] truth_seeking should light up with hormone_levels.CURIOSITY"
    assert chit_attention_depth > 0, "CHIT[8] attention_depth should light up with hormone_levels.FOCUS"
    assert chit_reflective_capacity > 0, "CHIT[9] reflective_capacity should light up with hormone_fires.REFLECTION"
    assert chit_temporal_awareness > 0, "CHIT[11] temporal_awareness should light up with sphere_clocks"
    assert chit_spatial_awareness > 0, "CHIT[12] spatial_awareness should light up with topology.volume"


def test_collect_spirit_45d_ananda_dims_activate():
    """ANANDA dims depending on hormone_fires / levels should activate."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    hormone_fires = {"CREATIVITY": 20, "INTUITION": 15, "EMPATHY": 18, "CURIOSITY": 25}
    hormone_levels = {"INSPIRATION": 0.8, "IMPULSE": 0.3, "VIGILANCE": 0.3}
    unified_spirit_stats = {"velocity": 0.7, "epoch_count": 3}

    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"density": 0.5, "fatigue": 0.3},
        hormone_fires=hormone_fires,
        hormone_levels=hormone_levels,
        unified_spirit_stats=unified_spirit_stats,
    )

    # ANANDA dims 30-44
    ananda_creative_joy = result[30 + 2]         # uses CREATIVITY fires
    ananda_truth_resonance = result[30 + 5]      # uses INTUITION fires
    ananda_connection_fulfillment = result[30 + 6]  # uses EMPATHY fires
    ananda_exploration_joy = result[30 + 9]      # uses CURIOSITY fires
    ananda_creative_tension = result[30 + 11]    # uses INSPIRATION level
    ananda_transcendence = result[30 + 14]       # uses us.epoch_count

    assert ananda_creative_joy > 0
    assert ananda_truth_resonance > 0
    assert ananda_connection_fulfillment > 0
    assert ananda_exploration_joy > 0
    assert ananda_creative_tension > 0
    assert ananda_transcendence > 0


def test_collect_spirit_45d_with_none_inputs_matches_pre_bug11_behavior():
    """Backwards compat: if the new optional args are None (like pre-fix),
    the affected dims return 0 as before. Pre-existing behavior preserved."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"epoch_count": 100, "density": 0.5},
        # All the new-fixed inputs left as None (default)
    )
    assert len(result) == 45

    # The dims that depend ONLY on None inputs should compute to 0
    chit_pattern_recognition = result[15 + 5]  # needs hormone_fires (None)
    chit_truth_seeking = result[15 + 7]         # needs hormone_levels (None)
    assert chit_pattern_recognition == 0.0
    assert chit_truth_seeking == 0.0


def test_publish_spirit_state_signature_accepts_hormonal_kwargs():
    """_publish_spirit_state now accepts neural_nervous_system, sphere_clock,
    unified_spirit, e_mem kwargs. Signature check ensures callers don't
    break when passing these."""
    import inspect
    from titan_plugin.modules.spirit_loop import _publish_spirit_state

    sig = inspect.signature(_publish_spirit_state)
    params = sig.parameters
    assert "neural_nervous_system" in params, \
        "BUG #11 fix: _publish_spirit_state must accept neural_nervous_system kwarg"
    assert "sphere_clock" in params, \
        "BUG #11 fix: _publish_spirit_state must accept sphere_clock kwarg"
    assert "unified_spirit" in params, \
        "BUG #11 fix: _publish_spirit_state must accept unified_spirit kwarg"
    assert "e_mem" in params, \
        "BUG #11 fix: _publish_spirit_state must accept e_mem kwarg"

    # Defaults should be None (no behavior change for old callers)
    assert params["neural_nervous_system"].default is None
    assert params["sphere_clock"].default is None


def test_publish_spirit_state_harvests_hormonal_state():
    """End-to-end: _publish_spirit_state with real-ish mocks pulls hormone
    levels + fires from neural_nervous_system._hormonal and passes them to
    collect_spirit_45d. Verified by inspecting the SPIRIT_STATE payload's
    values_45d — CHIT[7] (truth_seeking) should be non-zero after the fix."""
    import queue
    from titan_plugin.modules.spirit_loop import _publish_spirit_state

    # Mock hormonal system
    mock_hormone = MagicMock()
    mock_hormone.fire_count = 10
    mock_hormonal = MagicMock()
    mock_hormonal.get_levels.return_value = {"DA": 0.6, "CURIOSITY": 0.8, "FOCUS": 0.7}
    mock_hormonal._hormones = {"DA": mock_hormone, "CURIOSITY": mock_hormone}
    mock_nns = MagicMock()
    mock_nns._hormonal_enabled = True
    mock_nns._hormonal = mock_hormonal

    # Mock sphere clock
    mock_clock = MagicMock()
    mock_clock.pulse_count = 50
    mock_clock.current_phase = 0.3
    mock_sphere_clock = MagicMock()
    mock_sphere_clock.clocks = {"body": mock_clock, "mind": mock_clock}

    q = queue.Queue()
    _publish_spirit_state(
        send_queue=q,
        name="test",
        tensor=[0.5] * 5,
        consciousness={"latest_epoch": {"epoch_count": 1000, "density": 0.5}},
        body_state={"values": [0.5] * 5},
        mind_state={"values": [0.5] * 5, "values_15d": [0.5] * 15},
        neural_nervous_system=mock_nns,
        sphere_clock=mock_sphere_clock,
    )

    # Drain queue — should have one SPIRIT_STATE message
    msg = q.get_nowait()
    assert msg["type"] == "SPIRIT_STATE"
    payload = msg["payload"]
    assert "values_45d" in payload
    values_45d = payload["values_45d"]
    assert len(values_45d) == 45

    # CHIT[7] truth_seeking depends on CURIOSITY hormone level → 0.8 → should be > 0
    assert values_45d[15 + 7] > 0, \
        f"CHIT[7] truth_seeking should activate with CURIOSITY=0.8; got {values_45d[15+7]}"
    # CHIT[11] temporal_awareness depends on sphere_clock pulse_count totals
    assert values_45d[15 + 11] > 0, \
        f"CHIT[11] temporal_awareness should activate with clock pulses; got {values_45d[15+11]}"


def test_expression_intensity_falls_back_to_composites():
    """`_expression_intensity` should prefer `sovereignty_ratio` when the
    legacy ExpressionTranslator key is present, but fall back to a
    composite-level proxy (mean urge/threshold ratio) when only the
    spirit_worker-visible ExpressionManager stats are available."""
    from titan_plugin.logic.spirit_tensor import _expression_intensity

    # 1. Legacy translator-style stats — sovereignty_ratio wins
    assert _expression_intensity({"sovereignty_ratio": 0.7}) == pytest.approx(0.7)
    # NaN/None defends to clamped fallback then composites lookup → 0
    assert _expression_intensity({"sovereignty_ratio": None, "composites": {}}) == 0.0

    # 2. Composite-only stats (what expression_manager.get_stats() returns)
    composite_stats = {
        "total_composites": 3,
        "composites": {
            "speak": {"urge": 0.8, "threshold": 1.0, "fire_count": 5},
            "art":   {"urge": 0.3, "threshold": 1.0, "fire_count": 2},
            "music": {"urge": 1.5, "threshold": 1.0, "fire_count": 9},  # urge > thr → clamps to 1.0
        },
    }
    # mean(0.8, 0.3, 1.0) = 0.7
    assert _expression_intensity(composite_stats) == pytest.approx(0.7)

    # 3. Empty / malformed → 0
    assert _expression_intensity({}) == 0.0
    assert _expression_intensity({"composites": "not-a-dict"}) == 0.0
    # Threshold=0 entries are skipped (avoid div-by-zero); other entries still count
    skip_zero = {"composites": {
        "a": {"urge": 0.5, "threshold": 0.0},
        "b": {"urge": 0.6, "threshold": 1.0},
    }}
    assert _expression_intensity(skip_zero) == pytest.approx(0.6)


def test_collect_spirit_45d_chit13_ananda8_activate_from_composites():
    """dim 48 (chit[13] causal_understanding) + dim 28+30=58 (ananda[8]
    expression_quality) must activate when expression_stats carries the
    ExpressionManager composite shape (no `sovereignty_ratio` field)."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    expression_stats = {
        "total_composites": 6,
        "composites": {
            "speak":     {"urge": 0.6, "threshold": 1.0, "fire_count": 3},
            "art":       {"urge": 0.4, "threshold": 1.0, "fire_count": 1},
            "music":     {"urge": 0.8, "threshold": 1.0, "fire_count": 7},
            "social":    {"urge": 0.5, "threshold": 1.0, "fire_count": 4},
            "kin_sense": {"urge": 0.7, "threshold": 1.0, "fire_count": 2},
            "longing":   {"urge": 0.9, "threshold": 1.0, "fire_count": 5},
        },
    }
    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"epoch_id": 100, "density": 0.5},
        expression_stats=expression_stats,
    )
    # mean = (0.6+0.4+0.8+0.5+0.7+0.9)/6 = 0.65
    assert result[15 + 13] == pytest.approx(0.65), \
        f"chit[13] should equal mean composite ratio; got {result[15+13]}"
    # ananda[8] = 0.65 * 0.5 + 0.3 = 0.625
    assert result[30 + 8] == pytest.approx(0.625), \
        f"ananda[8] should equal intensity*0.5 + 0.3; got {result[30+8]}"


def test_collect_spirit_45d_reads_trajectory_magnitude_when_trajectory_missing():
    """Same producer/consumer name-mismatch class as epoch_count/epoch_id:
    `_run_consciousness_epoch` writes `trajectory_magnitude` but
    `chit[14]` was reading `trajectory`. Defensive fallback to the
    producer key."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={
            "epoch_id": 100,
            "trajectory_magnitude": 0.42,
            "density": 0.5,
        },
    )
    # chit[14] (= dim 49 in 130D buffer) should pick up trajectory_magnitude
    assert result[15 + 14] == pytest.approx(0.42), \
        f"chit[14] should read trajectory_magnitude fallback; got {result[15+14]}"

    # And explicit trajectory key still wins when present (back-compat)
    result2 = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"trajectory": 0.9, "trajectory_magnitude": 0.1},
    )
    assert result2[15 + 14] == pytest.approx(0.9)


def test_collect_spirit_45d_reads_epoch_id_when_epoch_count_missing():
    """Regression for the post-BUG#11 finding (2026-04-26): the producer
    `_run_consciousness_epoch` writes the epoch counter as `epoch_id`, but
    `collect_spirit_45d` was reading `epoch_count`. Result: dims 24
    (sat[4] temporal_continuity) and 35 (chit[0] self_awareness_depth) sat
    at 0 across all 3 Titans even with millions of epochs accumulated.
    Fix is a defensive fallback to `epoch_id` so both producer keys work.
    """
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    # Realistic latest_epoch dict — exact shape produced by spirit_loop:1266
    latest_epoch = {
        "epoch_id": 600000,
        "state_vector": [0.5] * 67,
        "drift_magnitude": 0.1,
        "trajectory_magnitude": 0.05,
        "curvature": 0.2,
        "density": 0.6,
    }
    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness=latest_epoch,
    )
    assert len(result) == 45
    # sat[4] = clamp(min(1.0, epoch_id/3000.0)) → 600000/3000 → clamped to 1.0
    assert result[4] == 1.0, \
        f"sat[4] temporal_continuity should saturate at high epoch_id; got {result[4]}"
    # chit[0] = clamp(min(1.0, epoch_id/5000.0)) → likewise saturates
    assert result[15 + 0] == 1.0, \
        f"chit[0] self_awareness_depth should saturate at high epoch_id; got {result[15+0]}"

    # And explicit `epoch_count` still wins when the legacy key is present
    legacy = {"epoch_count": 1500, "epoch_id": 999_999}
    result2 = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness=legacy,
    )
    # 1500/3000 = 0.5 — proves epoch_count took precedence over epoch_id
    assert result2[4] == pytest.approx(0.5), \
        f"explicit epoch_count must override epoch_id fallback; got {result2[4]}"


# ─────────────────────────────────────────────────────────────────────────
# 2026-04-27 follow-up: dims 45 / 47 / 48 still dead post-611fda0f
# Three more producer/consumer name-mismatches surfaced by live audit:
#   dim 45 (chit[10]) — dream_quality + fatigue not in latest_epoch
#   dim 47 (chit[12]) — topology never reached _publish_spirit_state
#   dim 48 (chit[13]) — _expression_intensity expected `urge`, prod has `last_urge`
# ─────────────────────────────────────────────────────────────────────────


def test_expression_intensity_accepts_last_urge_from_live_em_stats():
    """Live `expression_manager.get_stats()` emits per-composite dicts with
    `last_urge` (most recent eval) — NOT `urge`. The 611fda0f helper read
    `urge` only, so dim 48 stayed dead post-fix on all 3 Titans.
    Regression: helper must accept both keys."""
    from titan_plugin.logic.spirit_tensor import _expression_intensity

    # Live shape from /v4/inner-trinity expression_composites on T1
    live_em_stats = {
        "total_composites": 6,
        "composites": {
            "SPEAK":     {"name": "SPEAK", "fire_count": 1, "threshold": 0.5,
                          "last_urge": 1.0, "peak_urge": 2.7},
            "ART":       {"name": "ART", "fire_count": 111, "threshold": 2.0,
                          "last_urge": 2.0, "peak_urge": 2.16},
            "MUSIC":     {"name": "MUSIC", "fire_count": 147, "threshold": 2.0,
                          "last_urge": 1.0, "peak_urge": 2.26},
        },
    }
    # SPEAK: 1.0/0.5 → clamped to 1.0
    # ART:   2.0/2.0 → 1.0
    # MUSIC: 1.0/2.0 → 0.5
    # mean = (1.0 + 1.0 + 0.5) / 3 ≈ 0.833
    assert _expression_intensity(live_em_stats) == pytest.approx(0.833, abs=0.01)

    # Mixed shapes: explicit `urge` still wins when present
    mixed = {"composites": {
        "a": {"urge": 0.4, "last_urge": 0.9, "threshold": 1.0},
        "b": {"last_urge": 0.6, "threshold": 1.0},
    }}
    # a: urge=0.4 → 0.4 (urge wins over last_urge)
    # b: last_urge=0.6 → 0.6 (only last_urge present)
    # mean = 0.5
    assert _expression_intensity(mixed) == pytest.approx(0.5)


def test_collect_spirit_45d_chit10_activates_with_dream_quality_and_fatigue():
    """dim 45 (chit[10] dream_awareness) reads `dream_quality` (0.7 weight)
    + `fatigue` (0.3 weight) from consciousness epoch. Both producer-side
    sources exist (e_mem.get_dream_quality + coordinator.dreaming.last_fatigue)
    but were never enriched into the dict that flows into `_publish_spirit_state`."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    cons = {
        "epoch_id": 1000,
        "dream_quality": 0.8,
        "fatigue": 0.4,
        "density": 0.5,
    }
    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness=cons,
    )
    # chit[10] = clamp(0.8*0.7 + 0.4*0.3) = clamp(0.56 + 0.12) = 0.68
    assert result[15 + 10] == pytest.approx(0.68), \
        f"chit[10] dream_awareness should activate from dream_quality+fatigue; got {result[15+10]}"

    # When both missing → 0 (pre-fix behavior preserved)
    result_empty = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"epoch_id": 1000, "density": 0.5},
    )
    assert result_empty[15 + 10] == 0.0


def test_collect_spirit_45d_chit12_activates_with_topology_block():
    """dim 47 (chit[12] spatial_awareness) reads `volume` + `curvature`
    from the `topology` arg. Live values exist on coordinator
    (`_last_extended_topology`) but the call site was reading
    `body_state.get("topology")` — body_state never had that key."""
    from titan_plugin.logic.spirit_tensor import collect_spirit_45d

    # Live shape from /v4/inner-trinity topology on T1
    topology = {
        "volume": 5.54,
        "curvature": -1.7e-5,
        "cluster_count": 30,
    }
    result = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"epoch_id": 1000, "density": 0.5},
        topology=topology,
    )
    # chit[12] = clamp((volume/5.0 + abs(curvature)) / 2.0)
    #          = clamp((1.108 + 1.7e-5) / 2.0) ≈ 0.554
    assert result[15 + 12] == pytest.approx(0.554, abs=0.01), \
        f"chit[12] spatial_awareness should activate from topology; got {result[15+12]}"

    # Missing topology → 0 (pre-fix preserved)
    result_no_topo = collect_spirit_45d(
        current_5d=[0.5] * 5,
        body_tensor=[0.5] * 5,
        mind_tensor=[0.5] * 15,
        consciousness={"epoch_id": 1000, "density": 0.5},
    )
    assert result_no_topo[15 + 12] == 0.0


def test_enrich_spirit_inputs_for_bug11_routes_dream_fatigue_topology():
    """The spirit_worker enrichment helper bridges the call-site gap by
    pulling `dream_quality` (e_mem) + `fatigue` (coordinator.dreaming) into
    consciousness.latest_epoch and `topology` (coordinator) into body_state,
    in-place. Mutation must be idempotent."""
    from titan_plugin.modules.spirit_worker import _enrich_spirit_inputs_for_bug11

    class _FakeEMem:
        def get_dream_quality(self, last_n_cycles=5):
            return 0.75

    class _FakeDreaming:
        last_fatigue = 0.42

    class _FakeCoordinator:
        dreaming = _FakeDreaming()
        _last_topology_volcurv = {"volume": 4.5, "curvature": 0.1}

    consciousness = {"latest_epoch": {"epoch_id": 100, "density": 0.5}}
    body_state = {"values": [0.5] * 5}

    _enrich_spirit_inputs_for_bug11(consciousness, body_state, _FakeEMem(), _FakeCoordinator())

    # consciousness.latest_epoch enriched
    le = consciousness["latest_epoch"]
    assert le["dream_quality"] == pytest.approx(0.75)
    assert le["fatigue"] == pytest.approx(0.42)
    # body_state gains topology key
    assert body_state["topology"] == {"volume": 4.5, "curvature": 0.1}

    # Idempotent — repeat call mutates same values, no errors
    _enrich_spirit_inputs_for_bug11(consciousness, body_state, _FakeEMem(), _FakeCoordinator())
    assert le["dream_quality"] == pytest.approx(0.75)
    assert body_state["topology"] == {"volume": 4.5, "curvature": 0.1}

    # Defensive — None coordinator/e_mem doesn't raise
    consciousness2 = {"latest_epoch": {"epoch_id": 200}}
    body_state2 = {"values": [0.5] * 5}
    _enrich_spirit_inputs_for_bug11(consciousness2, body_state2, None, None)
    assert "dream_quality" not in consciousness2["latest_epoch"]
    assert "topology" not in body_state2

    # Pre-existing topology not overwritten (allows test injection / explicit override)
    body_state3 = {"topology": {"volume": 99.9, "curvature": 0.9}}
    consciousness3 = {"latest_epoch": {}}
    _enrich_spirit_inputs_for_bug11(consciousness3, body_state3, None, _FakeCoordinator())
    assert body_state3["topology"]["volume"] == 99.9  # not overwritten


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
