"""
inner_spirit_sidecar — shared Sprint 7 §4.6 raw-inputs sensor cache writer
for inner_spirit_45d.

Encapsulates the InnerSpiritSensorRefresh wiring + `_provide_spirit_45d`
callback so the same code path runs on:
  - T1 + T2 (Phase A+B): hosted inside `spirit_worker` subprocess
  - T3 (Phase C, D8-3 partial retirement): hosted inside `cognitive_worker`
    subprocess (spirit_worker is no longer a separate process on T3 —
    per `feedback_phase_c_spirit_worker_d8_retirement.md` + the D8-3
    chunk in rFP_microkernel_v2_definitive_closure)

Per `feedback_verify_worker_runs_on_target_before_implementing.md` —
this module exists specifically because Sprint 7+8 of
rFP_phase_c_130d_rust_l1_port were initially wired ONLY into
spirit_worker.py, which left T3 (the deploy target) with a missing
sensor_cache_inner_spirit.bin → all 45 inner_spirit dims stuck at 0.5
PARTIAL. This shared sidecar relocates the logic so it runs wherever
the host worker boots it, independent of which worker that is per
target.

# Usage

```python
from titan_hcl.logic.inner_spirit_sidecar import (
    start_inner_spirit_sensor_refresh)

stop_event = threading.Event()
sensor_thread = start_inner_spirit_sensor_refresh(
    config=config,
    stop_event=stop_event,
    logger=logger,
    log_prefix="[CognitiveWorker]",
)
```

Returns the thread handle (or None on failure). The caller is
responsible for stop_event lifecycle.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional
from titan_hcl.params import get_params


# ── D-SPEC-101 (rFP Dims Redesign Closure Phase 1) ──────────────────────
#: Identity cluster (inner_spirit local idx) whose movement defines how the
#: Titan's *authenticity* breathes — excludes authenticity[1] itself.
_AUTHENTICITY_CLUSTER = (0, 2, 9, 12, 13)  # self_recog, sovereignty, essence, uniqueness, integrity
#: hormones whose recent FIRE-RATE feeds re-grounded inner_spirit dims.
#: D-SPEC-101 Phase-1 completion (2026-05-21): extended from the original
#: INTUITION/REFLECTION/CREATIVITY trio with EMPATHY/CURIOSITY/INSPIRATION so
#: the second class of saturated counters (connection_fulfillment[36],
#: exploration_joy[39], truth_seeking[22], creative_tension[41]) can read a
#: breathing fire-RATE instead of a pinned cumulative count / frozen level.
_TRACKED_FIRE_HORMONES = (
    "INTUITION", "REFLECTION", "CREATIVITY",
    "EMPATHY", "CURIOSITY", "INSPIRATION",
)
#: fast/slow EMA half-lives. Fast = the felt "recent window" (~90s, the
#: rFP's 1–2 min at the 70 Hz spirit cadence). Slow = a per-Titan activity
#: baseline (~30 min) so dims read CHANGE-relative-to-typical, not an
#: absolute magic-scale rate → self-calibrating, no per-host tuning.
_FAST_HALF_LIFE_S = 90.0
_SLOW_HALF_LIFE_S = 1800.0
_DT_MIN_S = 1e-3
_DT_MAX_S = 5.0
_BASELINE_EPS = 1e-6


def _alpha(dt: float, half_life_s: float) -> float:
    """EMA smoothing factor for a time-decay EMA with the given half-life."""
    tau = half_life_s / math.log(2.0)
    return 1.0 - math.exp(-dt / tau)


class InnerSpiritWindowTracker:
    """Short-window self-observation tracker for inner_spirit (D-SPEC-101).

    Sampled per ``_provide_spirit_45d`` call (~70 Hz Schumann spirit). For
    each tracked signal it maintains a FAST time-decay EMA (~90s, the felt
    recent window) and a SLOW baseline EMA (~30 min). The emitted value is

        breath = fast / (fast + baseline + eps)   ∈ [0, 1)

    which reads ~0 when quiet, rises toward 1 on a fresh burst, and settles
    to ~0.5 under sustained activity (the baseline habituates). This makes
    every fed dim *breathe* with current activity — true variance from
    self-observation — without a fixed magic rate-scale (self-calibrating
    per Titan + per signal). Replaces the saturating ``min(1, count/N)`` and
    pinned ``epoch/N`` formulas that contributed zero variance.

    All state is per-process and resets on restart (a felt "recent window"
    has no meaning across a rebirth) — dims warm up over ~minutes.
    """

    __slots__ = (
        "_last_ts", "_last_fires", "_last_45d", "_last_topo10",
        "_last_levels", "_last_clock_pulses", "_fast", "_slow", "_level",
    )

    def __init__(self) -> None:
        self._last_ts: Optional[float] = None
        self._last_fires: dict = {}
        self._last_45d: Optional[list] = None
        self._last_topo10: Optional[list] = None
        self._last_levels: dict = {}
        self._last_clock_pulses: Optional[float] = None
        # signal-name → EMA value (breath dims: fast/(fast+slow))
        self._fast: dict = {}
        self._slow: dict = {}
        # signal-name → plain fast-EMA value (level dims that want a smoothed
        # MAGNITUDE, not a breath ratio — e.g. coherence_depth).
        self._level: dict = {}

    def _smooth(self, name: str, value: float, alpha: float) -> float:
        """Plain fast-EMA of an absolute level (no baseline-relative breath)."""
        prev = self._level.get(name)
        if prev is None:
            self._level[name] = value
            return value
        nv = prev + alpha * (value - prev)
        self._level[name] = nv
        return nv

    @staticmethod
    def _coherence_depth(spirit_45d: list, exclude_idx: int) -> float:
        """Self-observed coherence across the OTHER 44 inner_spirit dims:
        ``1 − var(44)/0.25`` (var of values in [0,1] maxes at 0.25). High when
        the felt dims hold together (deep self-awareness), low when they
        scatter. Excludes the dim this feeds (self_awareness_depth[15])."""
        vals = [float(spirit_45d[i]) for i in range(len(spirit_45d))
                if i != exclude_idx]
        n = len(vals)
        if n == 0:
            return 0.5
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return max(0.0, min(1.0, 1.0 - var / 0.25))

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

    @staticmethod
    def _mean_abs_delta(cur: list, prev: list, idxs) -> float:
        n = 0
        acc = 0.0
        for i in idxs:
            if i < len(cur) and i < len(prev):
                acc += abs(float(cur[i]) - float(prev[i]))
                n += 1
        return acc / n if n else 0.0

    def update(
        self,
        now: float,
        fires: Optional[dict],
        spirit_45d: list,
        topo10: Optional[list],
        levels: Optional[dict],
        clock_pulses: Optional[float] = None,
    ) -> dict:
        """Advance all EMAs and return the normalized source dict (0..1)."""
        fires = fires or {}
        levels = levels or {}
        out = {
            "fire_rate_INTUITION": 0.0,
            "fire_rate_REFLECTION": 0.0,
            "fire_rate_CREATIVITY": 0.0,
            "fire_rate_EMPATHY": 0.0,
            "fire_rate_CURIOSITY": 0.0,
            "fire_rate_INSPIRATION": 0.0,
            "self_churn": 0.0,
            "growth": 0.0,
            "authenticity_change": 0.0,
            "topo_change": 0.0,
            "hormone_velocity": 0.0,
            "coherence_depth": 0.5,
            "clock_pulse_rate": 0.0,
        }
        if self._last_ts is None:
            self._last_ts = now
            self._last_fires = dict(fires)
            self._last_45d = list(spirit_45d)
            self._last_topo10 = list(topo10) if topo10 else None
            self._last_levels = dict(levels)
            self._last_clock_pulses = (
                float(clock_pulses) if clock_pulses is not None else None)
            return out

        dt = min(max(now - self._last_ts, _DT_MIN_S), _DT_MAX_S)
        a_f = _alpha(dt, _FAST_HALF_LIFE_S)
        a_s = _alpha(dt, _SLOW_HALF_LIFE_S)

        # hormone fire RATES (cumulative counter delta ÷ dt; reset-safe)
        for h in _TRACKED_FIRE_HORMONES:
            cur = float(fires.get(h, 0.0))
            prev = float(self._last_fires.get(h, cur))
            rate = max(0.0, cur - prev) / dt
            out[f"fire_rate_{h}"] = self._observe(f"fire_{h}", rate, a_f, a_s)

        # self-velocity over the 45D (churn → temporal_continuity; growth)
        if self._last_45d is not None:
            churn = self._mean_abs_delta(
                spirit_45d, self._last_45d,
                [i for i in range(45) if i != 4]) / dt
            growth = self._mean_abs_delta(
                spirit_45d, self._last_45d,
                [i for i in range(45) if i != 6]) / dt
            ident = self._mean_abs_delta(
                spirit_45d, self._last_45d, _AUTHENTICITY_CLUSTER) / dt
            out["self_churn"] = self._observe("self_churn", churn, a_f, a_s)
            out["growth"] = self._observe("growth", growth, a_f, a_s)
            out["authenticity_change"] = self._observe(
                "authenticity", ident, a_f, a_s)

        # inner topology 10D change
        if topo10 is not None and self._last_topo10 is not None:
            tv = self._mean_abs_delta(
                topo10, self._last_topo10, range(min(10, len(topo10)))) / dt
            out["topo_change"] = self._observe("topo", tv, a_f, a_s)

        # hormone-level deviation movement (adaptability — breathing)
        if levels and self._last_levels:
            keys = set(levels) & set(self._last_levels)
            if keys:
                hv = sum(
                    abs(float(levels[k]) - float(self._last_levels[k]))
                    for k in keys) / len(keys) / dt
                out["hormone_velocity"] = self._observe(
                    "hormone_vel", hv, a_f, a_s)

        # clock_pulse_rate (feeds [26] temporal_awareness): recent sphere-clock
        # pulse RATE — temporal awareness BREATHES with how actively the clocks
        # are pulsing, replacing the cumulative min(1, Σpulses/50) that
        # saturates once the canonical pulse_count is read correctly.
        if clock_pulses is not None and self._last_clock_pulses is not None:
            cp_rate = max(0.0, float(clock_pulses) - self._last_clock_pulses) / dt
            out["clock_pulse_rate"] = self._observe(
                "clock_pulse", cp_rate, a_f, a_s)
        if clock_pulses is not None:
            self._last_clock_pulses = float(clock_pulses)

        # coherence_depth (feeds [15] self_awareness_depth): self-observed
        # coherence across the OTHER 44 inner_spirit dims, fast-EMA smoothed.
        # 1 − var/0.25 — deep self-awareness = the felt dims hold together.
        out["coherence_depth"] = self._smooth(
            "coherence_depth",
            self._coherence_depth(spirit_45d, exclude_idx=15),
            a_f)

        self._last_ts = now
        self._last_fires = dict(fires)
        self._last_45d = list(spirit_45d)
        if topo10 is not None:
            self._last_topo10 = list(topo10)
        self._last_levels = dict(levels)
        return out


def start_inner_spirit_sensor_refresh(
    config: dict,
    stop_event: threading.Event,
    logger,
    log_prefix: str = "[InnerSpiritSidecar]",
) -> Optional[threading.Thread]:
    """
    Boot the InnerSpiritSensorRefresh sidecar that publishes the
    raw-inputs source dict to sensor_cache_inner_spirit.bin at
    Schumann spirit rate. Idempotent — safe to call once per host
    worker boot.

    Per SPEC §G1 (Inner-Spirit 45D) + §23.6 collect_spirit_45d. Phase C
    is the canonical architecture (l0_rust permanently true) — Rust
    inner-spirit-rs reads this sensor cache; the legacy Phase A+B inline
    writer in spirit_worker was retired (D-SPEC-116 / config-shm Phase D).

    Returns:
        threading.Thread on success, None on failure (the host worker
        should log_critical when None — Rust inner-spirit-rs will starve
        without this writer).
    """

    try:
        from titan_hcl.logic.inner_spirit_sensor_refresh import (
            InnerSpiritSensorRefresh)
        from titan_hcl.core.state_registry import (
            INNER_BODY_5D, INNER_MIND_15D, INNER_SPIRIT_45D,
            NNS_HORMONAL_STATE, SPHERE_CLOCKS_STATE, TOPOLOGY_30D,
            RegistryBank)
        from titan_hcl.logic.spirit_state_specs import (
            HORMONE_FIRES_SPEC, CONSCIOUSNESS_STATE_SPEC,
            UNIFIED_SPIRIT_METADATA_SPEC,
        )
        from titan_hcl.logic.memory_state_specs import MEMORY_STATE_SPEC
        from titan_hcl.logic.expression_state_specs import (
            EXPRESSION_STATE_SPEC,
        )
        # rFP_trinity_dim_resonance — output_verifier_state.bin for
        # inner_spirit[17] discernment_quality re-grounding.
        from titan_hcl.logic.session3_state_specs import (
            OUTPUT_VERIFIER_STATE_SPEC,
        )
        # D-SPEC-101 Phase-1 completion: language_state.bin for the rich
        # expression-window word_rate (vocab_total cumulative) feeding
        # inner_spirit sovereignty[2] / causal_understanding[28] /
        # expression_quality[38].
        from titan_hcl.logic.session4_state_specs import (
            LANGUAGE_STATE_SPEC,
        )
        from titan_hcl.logic.expression_window_tracker import (
            ExpressionWindowTracker,
        )
        from titan_hcl.logic.spirit_helpers import _load_birth_state  # Phase 10C relocation
        # Phase B.5 (2026-05-18): HORMONE_NAMES tuple migrated from the
        # retired spirit_proxy.py to spirit_state_specs.py as
        # SPIRIT_PROXY_LEGACY_HORMONE_NAMES — semantics unchanged
        # (preserved verbatim for back-compat with this sidecar's
        # hormonal_state.bin row labeling).
        from titan_hcl.logic.spirit_state_specs import (
            SPIRIT_PROXY_LEGACY_HORMONE_NAMES as HORMONE_NAMES,
        )
        import msgpack as _msgpack

        _shm_bank = RegistryBank(titan_id=None, config=config)
        _body_reader = _shm_bank.reader(INNER_BODY_5D)
        _mind_reader = _shm_bank.reader(INNER_MIND_15D)
        # Phase 3.A wave 2 (D-SPEC-86 follow-up, 2026-05-18): read
        # NNS_HORMONAL_STATE (cognitive_worker's authoritative in-process
        # NeuralNervousSystem hormonal state — all 11 hormones populated
        # by NS evaluate ticks) instead of HORMONAL_STATE (only contains
        # IMPULSE from ns_worker's HORMONE_STIMULUS bus stream; other 10
        # hormones perpetually 0.0). Live audit 2026-05-18 confirmed:
        # HORMONAL_STATE.IMPULSE=0.6, all others=0.0; NNS_HORMONAL_STATE
        # has CURIOSITY=2.68, REFLECTION=2.71, FOCUS=0.26, INSPIRATION=0.02
        # all alive. Unblocks inner_spirit dims [22] truth_seeking,
        # [23] attention_depth, [41] creative_tension fleet-wide. Mirrors
        # the §4.B Track 3 expression_worker rationale documented at
        # state_registry.py:691.
        _hormonal_reader = _shm_bank.reader(NNS_HORMONAL_STATE)
        _sphere_clocks_reader = _shm_bank.reader(SPHERE_CLOCKS_STATE)
        _topology_reader = _shm_bank.reader(TOPOLOGY_30D)
        _hormone_fires_reader = _shm_bank.reader(HORMONE_FIRES_SPEC)
        _consciousness_reader = _shm_bank.reader(CONSCIOUSNESS_STATE_SPEC)
        _unified_spirit_reader = _shm_bank.reader(
            UNIFIED_SPIRIT_METADATA_SPEC)
        _memory_state_reader = _shm_bank.reader(MEMORY_STATE_SPEC)
        _expression_state_reader = _shm_bank.reader(EXPRESSION_STATE_SPEC)
        # rFP_trinity_dim_resonance — output_verifier judgment stats feed
        # inner_spirit[17] discernment_quality (re-grounded from action_chains).
        _ov_reader = _shm_bank.reader(OUTPUT_VERIFIER_STATE_SPEC)
        # language_state.bin (LanguageTeacher.get_stats) — vocab_total cumulative
        # feeds the expression-window word_rate.
        _language_reader = _shm_bank.reader(LANGUAGE_STATE_SPEC)
        # Wave 4a Cat A fix (D-SPEC-89, 2026-05-18): inner_spirit_45d.bin
        # reader for `current_5d` self-update. Pre-fix `_last_spirit_45d`
        # was declared nonlocal but never assigned, staying [0.5]*45
        # permanently → uniqueness dim=0 fleet-wide (l2_dist of [0.5]*5
        # vs [0.5]*5 = 0). Read back the Rust-computed 45D each tick so
        # current_5d reflects actual spirit drift.
        _inner_spirit_45d_reader = _shm_bank.reader(INNER_SPIRIT_45D)
        _birth_state_cached = _load_birth_state()

        # SPEC §7.1 sphere_clocks.bin canonical layout (6 clocks × 7 fields).
        # FIX (D-SPEC-101 Phase-1 completion, 2026-05-21): the prior names
        # (body/spirit/mind/energy/fire/soul) + field order (pulse_count first)
        # were WRONG — both writers (spirit_worker._write_sphere_clocks_shm +
        # titan-trinity-rs) emit rows inner_body…outer_spirit with fields
        # [radius, scalar_position, phase, contraction_velocity, pulse_count,
        # consecutive_balanced, last_pulse_age_s]. The mismap fed [26]
        # temporal_awareness a sum of RADII (≈const 0.1) instead of pulses,
        # and blocked any inner_spirit/outer_spirit resonance read.
        _SPHERE_CLOCK_NAMES = ("inner_body", "inner_mind", "inner_spirit",
                                "outer_body", "outer_mind", "outer_spirit")
        _SPHERE_CLOCK_FIELDS = ("radius", "scalar_position", "phase",
                                 "contraction_velocity", "pulse_count",
                                 "consecutive_balanced", "last_pulse_age_s")
        #: sphere clock geometry (SPEC §7.1) — min radius a contracted clock
        #: reaches; BIG PULSE fires after this many consecutive balanced cycles.
        _CLOCK_MIN_RADIUS = 0.3
        _BIG_PULSE_CONSEC = 3.0

        def _read_msgpack_slot(reader):
            try:
                blob = reader.read_variable()
                if blob:
                    decoded = _msgpack.unpackb(blob, raw=False)
                    if isinstance(decoded, dict):
                        return decoded
            except Exception:
                pass
            return None

        def _read_hormone_fires():
            """hormone_fires.bin payload is `{"fires":{HORMONE:count}, "ts":}`.
            Unwrap the inner dict so consumers see flat `{HORMONE:count}`."""
            outer = _read_msgpack_slot(_hormone_fires_reader)
            if outer is None:
                return None
            inner = outer.get("fires")
            if isinstance(inner, dict):
                return inner
            return None

        def _read_hormone_levels():
            try:
                arr = _hormonal_reader.read()
                if arr is None or arr.shape[0] < len(HORMONE_NAMES):
                    return None
                return {
                    name: float(arr[i][0])
                    for i, name in enumerate(HORMONE_NAMES)
                }
            except Exception:
                return None

        def _read_sphere_clocks():
            try:
                arr = _sphere_clocks_reader.read()
                if arr is None or arr.shape != (6, 7):
                    return None
                return {
                    _SPHERE_CLOCK_NAMES[i]: {
                        _SPHERE_CLOCK_FIELDS[j]: float(arr[i][j])
                        for j in range(7)
                    }
                    for i in range(6)
                }
            except Exception:
                return None

        def _clock_proximity(clk):
            """Per-clock BIG-PULSE readiness ∈ [0,1]: contraction toward min
            radius + balance toward the 3-consecutive threshold."""
            if not isinstance(clk, dict):
                return 0.0
            radius = float(clk.get("radius", 1.0))
            consec = float(clk.get("consecutive_balanced", 0.0))
            contraction = (1.0 - radius) / (1.0 - _CLOCK_MIN_RADIUS)
            contraction = max(0.0, min(1.0, contraction))
            balance = min(1.0, consec / _BIG_PULSE_CONSEC)
            return 0.5 * contraction + 0.5 * balance

        def _spirit_pair_resonance(clocks):
            """[44] transcendence_glimpse source: inner_spirit ↔ outer_spirit
            sphere-clock pair resonance = BIG-PULSE proximity. A pair is only
            as resonant as its weaker clock, so take the MIN — the BIG PULSE
            fires only when BOTH spirit clocks are contracted + balanced."""
            if not isinstance(clocks, dict):
                return 0.0
            inner = _clock_proximity(clocks.get("inner_spirit"))
            outer = _clock_proximity(clocks.get("outer_spirit"))
            return min(inner, outer)

        def _total_clock_pulses(clocks):
            if not isinstance(clocks, dict):
                return 0.0
            return sum(float(c.get("pulse_count", 0.0))
                       for c in clocks.values() if isinstance(c, dict))

        def _trim_expression_stats(expr_state):
            """Keep only the fields SPEC §23.6 + outer_spirit formulas
            consume. Drops top_mappings (largest payload component, no
            SPEC consumer) to keep sensor_cache_inner_spirit under cap.
            """
            if not isinstance(expr_state, dict):
                return None
            keep_keys = (
                "sovereignty_ratio",
                "intensity",
                "composites",
                "posture_authenticity_ratio_30",
                "learned_actions",
                "llm_actions",
                "total_actions",
            )
            return {k: expr_state[k] for k in keep_keys if k in expr_state}

        def _build_expression_history(expr_state):
            """SAT[2] sovereignty path: spirit_tensor.collect_spirit_45d
            reads `history.expression.sovereignty_ratio` (path 1) before
            falling through to expression_stats. Construct that nested
            shape so the primary path activates cleanly."""
            if not isinstance(expr_state, dict):
                return None
            sov = expr_state.get("sovereignty_ratio")
            if sov is None:
                return None
            return {"expression": {"sovereignty_ratio": float(sov)}}

        def _read_topology():
            try:
                arr = _topology_reader.read()
                if arr is None or arr.shape != (30,):
                    return None
                # SPEC §G14: topology_30d flat array carries summary
                # stats. Index 0 = volume, index 1 = curvature.
                v = arr.tolist()
                curv = float(v[1]) if len(v) > 1 else 0.0
                return {
                    "volume": float(v[0]) if len(v) > 0 else 0.0,
                    "curvature": curv,
                    "curvature_norm": curv,
                }
            except Exception:
                return None

        def _read_topology_10d():
            """Inner topology 10D (topology_30d[0:10] = inner_lower per
            §G14) for D-SPEC-101 spatial-dim short-window change."""
            try:
                arr = _topology_reader.read()
                if arr is None or arr.shape != (30,):
                    return None
                return [float(v) for v in arr[:10]]
            except Exception:
                return None

        def _read_expr_counts():
            """Cumulative expressive-output counters → {image, sound, speak,
            word} for the rich expression-window. image←ART, sound←MUSIC,
            speak←SPEAK composite fire_count (cumulative, persisted across
            restart per expression_translator); word←language vocab_total."""
            counts = {"image": 0.0, "sound": 0.0, "speak": 0.0, "word": 0.0}
            expr = _read_msgpack_slot(_expression_state_reader)
            if isinstance(expr, dict):
                comps = expr.get("composites")
                if isinstance(comps, dict):
                    for mod, name in (("image", "ART"), ("sound", "MUSIC"),
                                      ("speak", "SPEAK")):
                        c = comps.get(name)
                        if isinstance(c, dict):
                            counts[mod] = float(c.get("fire_count", 0.0) or 0.0)
                # sovereignty-of-expression: self-authored (learned postures)
                # vs total expressive actions — windowed Δ ratio (inner [2]).
                if "learned_actions" in expr:
                    counts["self_authored"] = float(
                        expr.get("learned_actions", 0.0) or 0.0)
                    counts["total"] = float(expr.get("total_actions", 0.0) or 0.0)
            lang = _read_msgpack_slot(_language_reader)
            if isinstance(lang, dict):
                counts["word"] = float(lang.get("vocab_total", 0.0) or 0.0)
            return counts

        _last_spirit_45d: list = [0.5] * 45  # current_5d bootstrap
        # D-SPEC-101 short-window self-observation tracker (Phase 1).
        _window_tracker = InnerSpiritWindowTracker()
        # D-SPEC-101 Phase-1 completion: rich expression rolling-window breath.
        _expr_window_tracker = ExpressionWindowTracker()
        # RFP_synthesis_decision_authority P3/P5 (INV-SDA-3): the ONE sovereignty
        # metric S = 0.7·E + 0.3·V feeds the live Rust CHIT[17] discernment_quality
        # dim (replacing the retired output_verifier sovereignty_score). The rolling
        # S lives in the synthesis metrics snapshot (a cross-process FILE) — too
        # expensive to read on this 70 Hz path, so TTL-cache it and pass it into the
        # source dict as `sovereignty.s`. `replies==0` (no scored reply yet) → the
        # key is omitted → Rust uses its neutral 0.5 default (dim never collapses).
        _sov_cache = {"s": 0.0, "replies": 0, "ts": 0.0}
        _SOV_TTL_S = 30.0

        def _provide_spirit_45d():
            """tensor_provider for InnerSpiritSensorRefresh — Sprint 7
            §4.6 FULL Rust formula port. Publishes raw inputs as
            msgpack source dict for titan-inner-spirit-rs."""
            nonlocal _last_spirit_45d
            # Wave 4a Cat A fix (D-SPEC-89) — read back the Rust-computed
            # 45D so current_5d reflects actual spirit drift. Pre-fix
            # _last_spirit_45d stayed [0.5]*45 permanently → uniqueness
            # dim=0 fleet-wide. Cheap O(1) SHM read.
            try:
                _spirit_arr = _inner_spirit_45d_reader.read()
                if _spirit_arr is not None and _spirit_arr.shape == (45,):
                    _last_spirit_45d = [float(v) for v in _spirit_arr]
            except Exception:
                pass  # fall through with prior _last_spirit_45d
            current_5d = [float(v) for v in _last_spirit_45d[:5]]
            try:
                _hlevels = _read_hormone_levels()
                _hfires = _read_hormone_fires()
                # D-SPEC-101: advance the short-window self-observation
                # tracker and emit normalized 0..1 breath signals that feed
                # the re-grounded inner_spirit dims (replaces saturating
                # cumulative-count / pinned-epoch formulas).
                _now = time.time()
                # P3/P5 (INV-SDA-3): refresh the rolling-sovereignty cache at
                # most every _SOV_TTL_S — the synthesis metrics snapshot is a
                # cross-process file; never read it per-tick on this 70 Hz path.
                if _now - _sov_cache["ts"] > _SOV_TTL_S:
                    try:
                        from titan_hcl.synthesis.sovereignty_readout import (
                            read_rolling_sovereignty)
                        _sov = read_rolling_sovereignty()
                        _sov_cache["s"] = float(_sov.get("s", 0.0) or 0.0)
                        _sov_cache["replies"] = int(_sov.get("replies", 0) or 0)
                    except Exception:
                        pass  # keep last-known; never break the 45D provider
                    _sov_cache["ts"] = _now
                _clocks = _read_sphere_clocks()
                _window = _window_tracker.update(
                    now=_now,
                    fires=_hfires,
                    spirit_45d=_last_spirit_45d,
                    topo10=_read_topology_10d(),
                    levels=_hlevels,
                    clock_pulses=_total_clock_pulses(_clocks),
                )
                # [44] transcendence_glimpse: inner↔outer spirit-pair BIG-PULSE
                # proximity from the sphere clocks (injected into the window map
                # so all re-grounded source keys ride one dict).
                _window["spirit_pair_resonance"] = _spirit_pair_resonance(_clocks)
                # rich expression-window breath (image/sound/speak/word
                # variety+volume) — feeds sovereignty[2], causal_understanding[28],
                # expression_quality[38].
                _expr_window = _expr_window_tracker.update(
                    now=_now, counts=_read_expr_counts())
                payload = {
                    "current_5d": current_5d,
                    "consciousness": _read_msgpack_slot(
                        _consciousness_reader),
                    "hormone_levels": _hlevels,
                    "hormone_fires": _hfires,
                    "inner_spirit_window": _window,
                    "expression_window": _expr_window,
                    "unified_spirit_stats": _read_msgpack_slot(
                        _unified_spirit_reader),
                    "sphere_clocks": _clocks,
                    "memory_stats": _read_msgpack_slot(_memory_state_reader),
                    "output_verifier_stats": _read_msgpack_slot(_ov_reader),
                    "topology": _read_topology(),
                    "expression_stats": _trim_expression_stats(
                        _read_msgpack_slot(_expression_state_reader)),
                    "birth_state": (
                        [float(v) for v in _birth_state_cached]
                        if _birth_state_cached else None),
                    "history": _build_expression_history(
                        _read_msgpack_slot(_expression_state_reader)),
                }
                # INV-SDA-3: feed the ONE synthesis sovereignty S → Rust CHIT[17].
                # Omit until the first scored reply so Rust keeps its neutral 0.5.
                if _sov_cache["replies"] > 0:
                    payload["sovereignty"] = {"s": _sov_cache["s"]}
                payload = {k: v for k, v in payload.items() if v is not None}
                return _msgpack.packb(payload, use_bin_type=True)
            except Exception as _e:
                logger.warning(
                    "%s _provide_spirit_45d input gathering failed: %s "
                    "— emitting legacy {tensor:[45D]} with last-known",
                    log_prefix, _e)
                return _msgpack.packb(
                    {"tensor": [float(v) for v in _last_spirit_45d]},
                    use_bin_type=True)

        _sensor_writer = InnerSpiritSensorRefresh(
            tensor_provider=_provide_spirit_45d,
            titan_id=None,
        )
        _sensor_thread = _sensor_writer.start_thread(stop_event)
        logger.info(
            "%s sensor_cache_inner_spirit.bin writer started "
            "(45×f32 @ Schumann spirit rate; SPEC §G1 + §23.6)",
            log_prefix)
        return _sensor_thread
    except Exception:
        logger.critical(
            "%s failed to start inner_spirit sensor cache writer — "
            "Rust inner-spirit-rs will starve on constant-zero input. "
            "Investigate before relying on inner_spirit_45d.bin "
            "downstream.", log_prefix, exc_info=True)
        return None
