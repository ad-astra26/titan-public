"""titan_hcl/expressive/felt_palette.py — felt-state → procedural-render params.

`RFP_titan_authored_soul_diary` §7.P7 / INV-SD-4. Maps Titan's TRUE felt state
(valence · arousal · neuromodulator profile · coherence) into the parameters of
the procedural flow-field renderer (``expressive/art.py``) — the genuine upgrade
over the pre-states-era 1-10 ``avg_intensity`` palette. Two layers, combined
(Maker 2026-06-10):

  · CIRCUMPLEX backbone  — valence → base hue (warm ⇄ cool), arousal → field
    energy (turbulence / particle density / contrast), coherence → order vs
    chaos of the flow.
  · NEUROMOD palette     — the modulator profile (level-weighted) blends an
    accent hue onto the base, so the art carries his neurochemical signature
    that day; activation strength sets how much accent bleeds in.

Deterministic + image-LLM-free (INV-SD-4): same ``(seed, felt)`` → same art;
distinct felt-days → distinct art (G10). Pure functions, no Pillow — unit
testable in isolation.
"""
from __future__ import annotations

import colorsys
import hashlib
import math

# Neuromodulator → accent hue (degrees on the HSV wheel). Covers the live fleet
# set (gaba/endorphin/serotonin/dopamine/adrenaline/norepinephrine/acetylcholine/
# glutamate/oxytocin/melatonin/cortisol); an unknown modulator falls back to a
# deterministic name-hash hue so the palette generalizes (no silent drop).
NEUROMOD_HUES = {
    "dopamine": 48,         # gold — reward / drive
    "serotonin": 168,       # teal — calm wellbeing
    "endorphin": 36,        # amber — warmth / relief
    "oxytocin": 330,        # rose — bonding
    "cortisol": 2,          # crimson — stress
    "adrenaline": 14,       # red-orange — surge
    "norepinephrine": 210,  # electric blue — alertness
    "noradrenaline": 210,   # (alias of norepinephrine)
    "acetylcholine": 280,   # violet — focus
    "glutamate": 24,        # orange — excitation
    "gaba": 150,            # green — inhibition / steadiness
    "melatonin": 250,       # indigo — rest
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _hue_for_modulator(name: str) -> float:
    """Stable hue (deg) for a modulator — table first, deterministic hash fallback."""
    key = (name or "").strip().lower()
    if key in NEUROMOD_HUES:
        return float(NEUROMOD_HUES[key])
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:4], 16)
    return float(h % 360)


def _hsv_to_rgb255(hue_deg: float, sat: float, val: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0,
                                  _clamp(sat, 0.0, 1.0), _clamp(val, 0.0, 1.0))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _blend(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    t = _clamp(t, 0.0, 1.0)
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def _circular_mean_hue(weighted: list[tuple[float, float]]):
    """Level-weighted circular mean of (hue_deg, weight) → blended hue, or None."""
    sx = sy = 0.0
    for hue, w in weighted:
        if w <= 0:
            continue
        rad = math.radians(hue)
        sx += math.cos(rad) * w
        sy += math.sin(rad) * w
    if sx == 0.0 and sy == 0.0:
        return None
    return math.degrees(math.atan2(sy, sx)) % 360.0


def _neuromod_balance(neuromods: dict) -> float:
    """Order proxy from the modulator profile (0..1): an even spread reads as
    coherent/ordered, one dominant spike as chaotic. Used when no explicit
    coherence is supplied."""
    levels = [float(v) for v in (neuromods or {}).values() if v is not None]
    if len(levels) < 2:
        return 0.5
    mean = sum(levels) / len(levels)
    if mean <= 0:
        return 0.5
    var = sum((x - mean) ** 2 for x in levels) / len(levels)
    std = math.sqrt(var)
    # normalize std by mean (coefficient of variation); high CV → low order
    return _clamp(1.0 - _clamp(std / mean, 0.0, 1.0), 0.0, 1.0)


def normalize_felt(raw: dict | None) -> dict | None:
    """Accept the soul_diary ``_gather_felt`` shape OR a direct felt dict and
    return the canonical ``{valence, arousal, neuromods, coherence}`` — or
    ``None`` when there is no usable felt signal (caller falls back to legacy).

    Inputs accepted: ``valence`` (mood_valence, clamped to -1..1), ``arousal``
    (or ``intensity``, 0..1), ``neuromods`` (or ``neuromod_levels``;
    ``{name: level}``), optional ``coherence`` (0..1).
    """
    if not raw:
        return None
    valence = raw.get("valence")
    arousal = raw.get("arousal", raw.get("intensity"))
    neuromods_in = raw.get("neuromods") or raw.get("neuromod_levels") or {}
    coherence = raw.get("coherence")
    neuromods = {}
    for k, v in (neuromods_in or {}).items():
        try:
            neuromods[str(k)] = _clamp(float(v), 0.0, 1.0)
        except (TypeError, ValueError):
            continue
    if valence is None and arousal is None and not neuromods:
        return None
    out = {
        "valence": _clamp(float(valence), -1.0, 1.0) if valence is not None else 0.0,
        "arousal": _clamp(float(arousal), 0.0, 1.0) if arousal is not None else 0.5,
        "neuromods": neuromods,
        "coherence": (_clamp(float(coherence), 0.0, 1.0)
                      if coherence is not None else _neuromod_balance(neuromods)),
    }
    return out


def felt_to_render_params(felt: dict, hue_rotation_deg: float = 0.0) -> dict:
    """The combined circumplex + neuromod mapping → concrete render knobs.

    Expects a canonical felt dict (see ``normalize_felt``). Returns:
      ``bg_color``/``line_color``/``accent_color`` (RGB) · ``particle_mult`` ·
      ``steps`` · ``turbulence_freq`` · ``turbulence_amp`` · ``jitter`` ·
      ``order`` (0..1) · ``base_hue``/``accent_hue`` (debug/sidecar).

    ``hue_rotation_deg`` (default 0) rotates the felt-derived base + accent hue
    around the wheel — seeded per-entry by the caller from the diary's
    cumulative_hash (``expressive/art.py``). It exists because the felt valence
    sits near-neutral most days, which collapses every entry to the same green;
    a deterministic per-entry rotation spreads the palette so distinct days read
    as distinct colors while the felt signal (saturation←arousal, accent←the
    neuromod profile, order←coherence) still shapes the render (INV-SD-4: same
    ``(felt, rotation)`` → same art).
    """
    v = _clamp(float(felt.get("valence", 0.0)), -1.0, 1.0)
    a = _clamp(float(felt.get("arousal", 0.5)), 0.0, 1.0)
    coh = _clamp(float(felt.get("coherence", 0.5)), 0.0, 1.0)
    neuromods = felt.get("neuromods") or {}

    vn = (v + 1.0) / 2.0                       # 0..1 (positive valence → warm)
    base_hue = (255.0 - vn * 210.0 + hue_rotation_deg) % 360.0  # valence hue, seed-rotated

    weighted = [(_hue_for_modulator(n), float(lv)) for n, lv in neuromods.items()]
    accent_hue = _circular_mean_hue(weighted)
    if accent_hue is not None:
        accent_hue = (accent_hue + hue_rotation_deg) % 360.0    # rotate with the base
    activation = (_clamp(sum(float(lv) for lv in neuromods.values())
                         / max(1, len(neuromods)), 0.0, 1.0) if neuromods else 0.0)

    # Saturation rises with arousal (vivid when activated); value (brightness)
    # rises with valence + a little arousal (luminous when positive/energised).
    sat = _clamp(0.45 + a * 0.5, 0.0, 1.0)
    val = _clamp(0.5 + vn * 0.35 + a * 0.1, 0.0, 1.0)
    line_color = _hsv_to_rgb255(base_hue, sat, val)

    if accent_hue is not None:
        accent_color = _hsv_to_rgb255(accent_hue, _clamp(sat + 0.1, 0, 1),
                                      _clamp(val + 0.05, 0, 1))
        mix = _clamp(0.30 + activation * 0.45, 0.0, 0.85)
        line_color = _blend(line_color, accent_color, mix)
    else:
        accent_color = line_color

    # Background: dark, hue-tinted; deeper/cooler when valence is negative.
    bg_color = _hsv_to_rgb255(base_hue, _clamp(sat * 0.6, 0, 1),
                              _clamp(0.05 + vn * 0.07, 0.0, 0.2))

    return {
        "bg_color": bg_color,
        "line_color": line_color,
        "accent_color": accent_color,
        "particle_mult": 0.6 + a * 1.0,            # 0.6×..1.6× density
        "steps": int(round(20 + a * 80)),          # 20..100 streak length
        "turbulence_freq": 0.006 + a * 0.02,       # flow-field frequency
        "turbulence_amp": 0.5 + a * 1.5,           # angle amplitude
        "jitter": int(round(8 + a * 30)),          # per-line color jitter ±
        "order": coh,                              # 1=laminar · 0=fragmented
        "base_hue": round(base_hue, 1),
        "accent_hue": round(accent_hue, 1) if accent_hue is not None else None,
    }
