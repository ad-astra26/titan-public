"""
expression_window_tracker — shared rich-expression rolling-window breath.

D-SPEC (rFP Dims Redesign Closure — Phase 1 inner completion + Phase 2 outer).

Both spirit layers carry "expressiveness" dims (inner_spirit sovereignty[2],
causal_understanding[28], expression_quality[38]; outer_spirit
expressive_authenticity / creative_expression / expression_reach /
beauty_create / harmony). Pre-fix these read a STATIC ratio or an empty
`expression_intensity`, contributing zero variance. Per the Maker's §86
directive — "all expressiveness-related spirit dims should source from the
*variety + volume of music / speak / image generation* (the spirit's
expressive output)" — this tracker turns the Titan's own cumulative
expressive-output counters into a breathing rate + variety signal.

# Mechanism (mirrors InnerSpiritWindowTracker's dual-EMA breath)

Per modality it keeps a FAST time-decay EMA (~90s, the felt recent window)
and a SLOW baseline EMA (~30 min) of the per-second output RATE (cumulative
counter delta ÷ dt; reset-safe). The emitted breath is

    breath = fast / (fast + baseline + eps)   ∈ [0, 1)

~0 when that modality is quiet, rising on a fresh burst, settling to ~0.5
under sustained output (baseline habituates) — self-calibrating per Titan,
no magic rate-scale. Cadence-agnostic: the time-decay EMA adapts to dt, so
the SAME tracker serves the inner sidecar (~70 Hz) and the outer source
gather (~5 s) correctly.

# Emitted source dict

    {image_rate, sound_rate, speak_rate, word_rate, variety, volume}

- `<modality>_rate`: breath for that modality (image←ART, sound←MUSIC,
  speak←SPEAK, word←language vocab_total).
- `variety`  = fraction of tracked modalities currently active (breath >
  ACTIVE_EPS) ∈ [0,1] — breadth of expressive output.
- `volume`   = mean breath across modalities — overall expressive intensity.

State is per-process and resets on restart (a felt window has no meaning
across rebirth); warms up over ~minutes.
"""
from __future__ import annotations

import math
from typing import Optional

#: Tracked expressive modalities (order fixed for stable emit keys).
MODALITIES = ("image", "sound", "speak", "word")
_FAST_HALF_LIFE_S = 90.0
_SLOW_HALF_LIFE_S = 1800.0
_DT_MIN_S = 1e-3
_DT_MAX_S = 30.0  # outer gather cadence is ~5s; cap generously
_BASELINE_EPS = 1e-6
#: a modality counts toward `variety` once its breath rises above this.
_ACTIVE_EPS = 0.05


def _alpha(dt: float, half_life_s: float) -> float:
    tau = half_life_s / math.log(2.0)
    return 1.0 - math.exp(-dt / tau)


class ExpressionWindowTracker:
    """Dual-EMA breath over the Titan's cumulative expressive-output counts."""

    __slots__ = ("_last_ts", "_last_counts", "_fast", "_slow", "_sovereignty",
                 "_modalities")

    def __init__(self, modalities: tuple = MODALITIES) -> None:
        #: tracked modalities — defaults to the expression set (image/sound/
        #: speak/word) but configurable so the same dual-EMA breath serves
        #: the outer_mind willing dims (action/social/creative/protective/
        #: exploration) — D-SPEC-101 Phase-2.
        self._modalities: tuple = tuple(modalities)
        self._last_ts: Optional[float] = None
        self._last_counts: dict = {}
        self._fast: dict = {}
        self._slow: dict = {}
        #: smoothed windowed sovereignty-of-expression ratio (None until warm).
        self._sovereignty: Optional[float] = None

    def _observe(self, name: str, value: float, a_fast: float, a_slow: float) -> float:
        f = self._fast.get(name)
        s = self._slow.get(name)
        if f is None:
            self._fast[name] = value
            self._slow[name] = value
            return 0.0  # no breath until a baseline forms
        f += a_fast * (value - f)
        s += a_slow * (value - s)
        self._fast[name] = f
        self._slow[name] = s
        return f / (f + s + _BASELINE_EPS)

    def update(self, now: float, counts: Optional[dict]) -> dict:
        """Advance EMAs from cumulative `counts` and emit the source dict.

        `counts` maps modality → lifetime cumulative count (monotonic,
        reset-safe). Absent modalities are treated as unchanged.
        """
        counts = counts or {}
        mods = self._modalities
        out = {f"{m}_rate": 0.0 for m in mods}
        out["variety"] = 0.0
        out["volume"] = 0.0
        out["sovereignty"] = 0.5  # neutral until a baseline forms

        _track = mods + ("self_authored", "total")
        if self._last_ts is None:
            self._last_ts = now
            self._last_counts = {m: float(counts.get(m, 0.0)) for m in _track}
            return out

        dt = min(max(now - self._last_ts, _DT_MIN_S), _DT_MAX_S)
        a_f = _alpha(dt, _FAST_HALF_LIFE_S)
        a_s = _alpha(dt, _SLOW_HALF_LIFE_S)

        active = 0
        breath_sum = 0.0
        for m in mods:
            cur = float(counts.get(m, self._last_counts.get(m, 0.0)))
            prev = float(self._last_counts.get(m, cur))
            rate = max(0.0, cur - prev) / dt  # reset-safe (negative → 0)
            breath = self._observe(m, rate, a_f, a_s)
            out[f"{m}_rate"] = breath
            breath_sum += breath
            if breath > _ACTIVE_EPS:
                active += 1
            self._last_counts[m] = cur

        out["variety"] = active / len(mods)
        out["volume"] = breath_sum / len(mods)

        # windowed sovereignty-of-expression: self-authored / total expression
        # over the recent window (Δself ÷ Δtotal), fast-EMA smoothed. Self-
        # authored = the Titan's own composite fires; total = + LLM-mediated.
        if "self_authored" in counts and "total" in counts:
            sa_cur = float(counts.get("self_authored", 0.0))
            sa_prev = float(self._last_counts.get("self_authored", sa_cur))
            tot_cur = float(counts.get("total", 0.0))
            tot_prev = float(self._last_counts.get("total", tot_cur))
            d_sa = max(0.0, sa_cur - sa_prev)
            d_tot = max(0.0, tot_cur - tot_prev)
            if d_tot > 0.0:
                ratio = min(1.0, d_sa / d_tot)
                if self._sovereignty is None:
                    self._sovereignty = ratio
                else:
                    self._sovereignty += a_f * (ratio - self._sovereignty)
            self._last_counts["self_authored"] = sa_cur
            self._last_counts["total"] = tot_cur
        if self._sovereignty is not None:
            out["sovereignty"] = self._sovereignty

        self._last_ts = now
        return out


class ChangeBreathTracker:
    """Dual-EMA breath over the RATE-OF-CHANGE of named LEVEL signals.

    Where ``ExpressionWindowTracker`` breathes on cumulative-counter rates,
    this breathes on |Δlevel|/dt — for dims the Maker re-grounded from an
    *instantaneous value* to its *rate of change over a minutes-scale window*
    (outer_body entropy[68] / thermal[69], D-SPEC-101 Phase-2). A steady level
    → ~0; a moving level → rises; sustained churn → ~0.5 (baseline habituates).
    """

    __slots__ = ("_last_ts", "_last_level", "_fast", "_slow")

    def __init__(self) -> None:
        self._last_ts: Optional[float] = None
        self._last_level: dict = {}
        self._fast: dict = {}
        self._slow: dict = {}

    def update(self, now: float, levels: Optional[dict]) -> dict:
        levels = levels or {}
        out = {f"{k}_change": 0.0 for k in levels}
        if self._last_ts is None:
            self._last_ts = now
            self._last_level = {k: float(v) for k, v in levels.items()}
            return out
        dt = min(max(now - self._last_ts, _DT_MIN_S), _DT_MAX_S)
        a_f = _alpha(dt, _FAST_HALF_LIFE_S)
        a_s = _alpha(dt, _SLOW_HALF_LIFE_S)
        for k, v in levels.items():
            cur = float(v)
            prev = float(self._last_level.get(k, cur))
            rate = abs(cur - prev) / dt
            f = self._fast.get(k)
            s = self._slow.get(k)
            if f is None:
                self._fast[k] = rate
                self._slow[k] = rate
                out[f"{k}_change"] = 0.0
            else:
                f += a_f * (rate - f)
                s += a_s * (rate - s)
                self._fast[k] = f
                self._slow[k] = s
                out[f"{k}_change"] = f / (f + s + _BASELINE_EPS)
            self._last_level[k] = cur
        self._last_ts = now
        return out


class EmaVarianceTracker:
    """Scale-free running variability of a scalar over a long window.

    For outer_body interoception[65] = π-cluster heartbeat variance over a
    rolling 24h window (HRV-like). Uses EMA mean + EMA of squared-deviation
    (var ≈ E[(x−mean)²]) with a 24h half-life — O(1) memory, no 8.6k-sample
    deque. Emits the squared coefficient of variation ``var / mean²`` clamped
    to [0,1] — scale-free (no magic norm constant): a steady internal rhythm
    → ~0, a variable one → rises. This is the HRV-of-the-π-heartbeat: high
    interoception = the Titan keenly feels its own fluctuating inner cadence.
    """

    __slots__ = ("_half_life_s", "_last_ts", "_mean", "_var")

    def __init__(self, half_life_s: float = 86400.0) -> None:
        self._half_life_s = float(half_life_s)
        self._last_ts: Optional[float] = None
        self._mean: Optional[float] = None
        self._var: float = 0.0

    def _cv2(self) -> float:
        if self._mean is None:
            return 0.0
        return min(1.0, self._var / (self._mean * self._mean + _BASELINE_EPS))

    def update(self, now: float, value: Optional[float]) -> float:
        """Feed a new sample; return scale-free variability (CV²) ∈ [0,1]."""
        if value is None:
            return self._cv2()
        x = float(value)
        if self._mean is None:
            self._mean = x
            self._last_ts = now
            return 0.0
        dt = max(now - (self._last_ts or now), 0.0)
        a = _alpha(dt, self._half_life_s) if dt > 0 else 0.0
        dev = x - self._mean
        self._mean += a * dev
        self._var += a * (dev * dev - self._var)
        self._last_ts = now
        return self._cv2()
