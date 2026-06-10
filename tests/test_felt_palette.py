"""P7 — felt-driven procedural art (`RFP_titan_authored_soul_diary` §7.P7 / INV-SD-4).

Two layers:
  1. ``felt_palette`` — the pure felt→render-params core (circumplex + neuromod
     palette): valence→hue, arousal→energy, neuromod profile→accent, coherence→
     order; deterministic; generalizes to unknown modulators.
  2. ``ProceduralArtGen.generate_flow_field`` — felt drives the render; the
     legacy ``felt=None`` path is unchanged (backward-compat); same (seed, felt)
     renders identically (INV-SD-4, no image-LLM).
"""
from titan_hcl.expressive import felt_palette as fp
from titan_hcl.expressive.art import ProceduralArtGen


# ── 1 · felt_palette core ────────────────────────────────────────────────────

def test_normalize_felt_accepts_gather_shape_and_clamps():
    # the soul_diary _gather_felt shape
    out = fp.normalize_felt({"valence": 0.6, "intensity": 0.4,
                             "neuromod_levels": {"dopamine": 0.8, "cortisol": 0.2}})
    assert out["valence"] == 0.6 and out["arousal"] == 0.4
    assert out["neuromods"] == {"dopamine": 0.8, "cortisol": 0.2}
    assert 0.0 <= out["coherence"] <= 1.0          # derived from neuromod balance
    # direct shape + clamping out-of-range inputs
    out2 = fp.normalize_felt({"valence": 5.0, "arousal": -1.0, "neuromods": {},
                              "coherence": 0.3})
    assert out2["valence"] == 1.0 and out2["arousal"] == 0.0 and out2["coherence"] == 0.3
    # no usable signal → None
    assert fp.normalize_felt({}) is None
    assert fp.normalize_felt(None) is None


def test_valence_drives_hue_warm_to_cool():
    warm = fp.felt_to_render_params({"valence": 1.0, "arousal": 0.5, "neuromods": {}})
    neutral = fp.felt_to_render_params({"valence": 0.0, "arousal": 0.5, "neuromods": {}})
    cool = fp.felt_to_render_params({"valence": -1.0, "arousal": 0.5, "neuromods": {}})
    # positive valence → lower hue (gold/warm), negative → higher (indigo/cool)
    assert warm["base_hue"] == 45.0
    assert cool["base_hue"] == 255.0
    assert warm["base_hue"] < neutral["base_hue"] < cool["base_hue"]


def test_arousal_drives_field_energy():
    calm = fp.felt_to_render_params({"valence": 0.0, "arousal": 0.0, "neuromods": {}})
    hot = fp.felt_to_render_params({"valence": 0.0, "arousal": 1.0, "neuromods": {}})
    assert hot["particle_mult"] > calm["particle_mult"]    # denser when activated
    assert hot["steps"] > calm["steps"]                    # longer streaks
    assert hot["turbulence_amp"] > calm["turbulence_amp"]  # more turbulent
    assert hot["jitter"] > calm["jitter"]


def test_neuromod_profile_drives_accent_hue():
    dop = fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {"dopamine": 0.9}})
    ser = fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {"serotonin": 0.9}})
    assert abs(dop["accent_hue"] - fp.NEUROMOD_HUES["dopamine"]) < 1.0
    assert abs(ser["accent_hue"] - fp.NEUROMOD_HUES["serotonin"]) < 1.0
    assert dop["accent_hue"] != ser["accent_hue"]
    # unknown modulator still gets a stable hue (generalizes, no silent drop)
    unknown = fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {"mystery_mod": 0.9}})
    assert unknown["accent_hue"] is not None
    assert (fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {"mystery_mod": 0.9}})["accent_hue"]
        == unknown["accent_hue"])                          # deterministic fallback


def test_coherence_drives_order():
    ordered = fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {}, "coherence": 1.0})
    chaotic = fp.felt_to_render_params(
        {"valence": 0.0, "arousal": 0.5, "neuromods": {}, "coherence": 0.2})
    assert ordered["order"] == 1.0 and chaotic["order"] == 0.2


def test_params_are_valid_rgb_and_deterministic():
    felt = {"valence": 0.3, "arousal": 0.6, "neuromods": {"dopamine": 0.5, "gaba": 0.3},
            "coherence": 0.7}
    p1 = fp.felt_to_render_params(felt)
    p2 = fp.felt_to_render_params(dict(felt))
    assert p1 == p2                                        # deterministic
    for key in ("bg_color", "line_color", "accent_color"):
        assert len(p1[key]) == 3
        assert all(0 <= ch <= 255 for ch in p1[key])


# ── 2 · generate_flow_field — felt drives render, legacy unchanged ───────────

def _img_bytes(gen, *, felt=None):
    return gen.generate_flow_field("seed_abc", 40, 5, return_image=True,
                                   resolution=96, felt=felt).tobytes()


def test_render_deterministic_legacy_and_felt(tmp_path):
    gen = ProceduralArtGen(output_dir=str(tmp_path))
    # legacy (felt=None): two identical calls → identical pixels
    assert _img_bytes(gen) == _img_bytes(gen)
    felt = {"valence": 0.7, "arousal": 0.6, "neuromods": {"dopamine": 0.8}}
    # felt: same (seed, felt) → identical (INV-SD-4)
    assert _img_bytes(gen, felt=felt) == _img_bytes(gen, felt=dict(felt))


def test_felt_changes_render_and_distinct_felt_distinct_art(tmp_path):
    gen = ProceduralArtGen(output_dir=str(tmp_path))
    legacy = _img_bytes(gen)
    warm = _img_bytes(gen, felt={"valence": 0.9, "arousal": 0.8,
                                 "neuromods": {"dopamine": 0.9}})
    cool = _img_bytes(gen, felt={"valence": -0.9, "arousal": 0.2,
                                 "neuromods": {"serotonin": 0.9}})
    assert warm != legacy            # felt actually drives the render
    assert warm != cool              # distinct felt-days → distinct art (G10)


def test_empty_felt_falls_back_to_legacy(tmp_path):
    """A felt dict with no usable signal normalizes to None → legacy palette
    (identical to felt=None) — no crash, graceful degradation."""
    gen = ProceduralArtGen(output_dir=str(tmp_path))
    assert _img_bytes(gen, felt={}) == _img_bytes(gen, felt=None)
