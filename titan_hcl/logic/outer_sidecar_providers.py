"""
outer_sidecar_providers — per-layer source providers for the outer
sensor-refresh sidecars (Phase C dissolution of _gather_outer_sources).

Each factory returns a stateless-to-call provider closure that the matching
sidecar uses as its ``sources_provider``. The closure:
  1. assembles the layer's base source keys SHM-direct via
     ``outer_source_assembly.assemble_outer_sources`` (no bus, no parent gather);
  2. computes the layer's stateful **breath** signal(s) — the rolling-window /
     EMA trackers that were ``TitanHCL._gather_outer_window_signals``, now
     RE-HOMED here into the owning sidecar (they accumulate state across calls,
     so they live in the closure, not the stateless assembler);
  3. merges the breath keys into the dict.

The Rust outer daemons already read every breath key (verified in
titan-outer-{body,mind,spirit}-rs tick_loop.rs), falling back to defaults today
because the breath keys were never in the sidecar SOURCE_KEYS — only on the
deleted OUTER_SOURCES_SNAPSHOT bus broadcast consumed by the (now unspawned)
Python workers. Wiring them here closes that latent D-SPEC-101 gap.

Breath ownership (one tracker, one sidecar):
  - outer_body   : outer_body_change (ChangeBreathTracker), pi_heartbeat_hrv (EmaVariance)
  - outer_mind   : willing_window (ExpressionWindowTracker over volitional counters)
  - outer_spirit : expr_window (ExpressionWindowTracker), outer_spirit_self_change (ChangeBreath)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

from titan_hcl.logic.outer_source_assembly import (
    OuterSourceContext, assemble_outer_sources,
)
from titan_hcl.logic.expression_window_tracker import (
    ExpressionWindowTracker, ChangeBreathTracker, EmaVarianceTracker,
)

logger = logging.getLogger(__name__)

Provider = Callable[[], dict]


def _composite_fire_counts(ctx: OuterSourceContext) -> dict:
    """fire_count per expression composite from expression_state SHM."""
    try:
        es = ctx.shm_bank.read_expression_state() or {}
        comps = es.get("composites") or {}
        if isinstance(comps, dict):
            return {
                name: float(c.get("fire_count", 0.0) or 0.0)
                for name, c in comps.items() if isinstance(c, dict)
            }
    except Exception as _e:
        logger.debug("[OuterProviders] composite fire counts: %s", _e)
    return {}


def make_outer_body_provider(ctx: OuterSourceContext,
                             source_keys: Iterable[str]) -> Provider:
    keys = set(source_keys)
    change_tracker = ChangeBreathTracker()
    pi_hrv_tracker = EmaVarianceTracker(half_life_s=86400.0)
    st: dict = {"pi_last": None}  # (ts, pulse_count)

    def provider() -> dict:
        sources = assemble_outer_sources(keys, ctx)
        now = time.time()
        # entropy[68] + thermal[69] minutes-scale rate-of-change breath.
        _net = sources.get("network_monitor_stats") or {}
        _sys = sources.get("system_sensor_stats") or {}
        _hl = sources.get("hormone_levels") or {}
        _ag = sources.get("agency_stats") or {}
        _tot = float(_ag.get("total_actions", 0.0) or 0.0)
        _fail = float(_ag.get("failed_actions", 0.0) or 0.0)
        _err_rate = 1.0 - ((_tot - _fail) / _tot) if _tot > 0 else 0.0
        _entropy = (0.4 * float(_net.get("ping_variance", 0.5) or 0.5)
                    + 0.3 * float(_net.get("bus_drop_rate", 0.0) or 0.0)
                    + 0.3 * _err_rate)
        _hormonal_heat = (float(_hl.get("IMPULSE", 0.5) or 0.5)
                          + float(_hl.get("VIGILANCE", 0.5) or 0.5)) / 2.0
        _thermal = (0.35 * float(_sys.get("cpu_thermal", 0.5) or 0.5)
                    + 0.25 * float(_sys.get("circadian_phase", 0.5) or 0.5)
                    + 0.40 * _hormonal_heat)
        sources["outer_body_change"] = change_tracker.update(
            now, {"entropy": _entropy, "thermal": _thermal})
        # interoception[65] = π-heartbeat 24h HRV.
        try:
            _pi = ctx.shm_bank.read_pi_heartbeat()
            if isinstance(_pi, dict) and _pi.get("pulse_count") is not None:
                _pc = float(_pi["pulse_count"])
                _rate = None
                if st["pi_last"] is not None:
                    _dt = max(now - st["pi_last"][0], 1e-3)
                    _rate = max(0.0, _pc - st["pi_last"][1]) / _dt
                st["pi_last"] = (now, _pc)
                sources["pi_heartbeat_hrv"] = pi_hrv_tracker.update(now, _rate)
        except Exception as _e:
            logger.debug("[OuterProviders] body pi_hrv: %s", _e)
        return sources

    return provider


def make_outer_mind_provider(ctx: OuterSourceContext,
                             source_keys: Iterable[str]) -> Provider:
    keys = set(source_keys)
    willing_tracker = ExpressionWindowTracker(modalities=(
        "action", "social", "creative", "protective", "exploration"))

    def provider() -> dict:
        sources = assemble_outer_sources(keys, ctx)
        now = time.time()
        _fc = _composite_fire_counts(ctx)
        _ag = sources.get("agency_stats") or {}
        _ov = sources.get("output_verifier_stats") or {}
        _mc = sources.get("meta_cgn_stats") or {}
        _lang = sources.get("language_stats") or {}
        _vocab = float(_lang.get("vocab_total", 0.0) or 0.0)
        sources["willing_window"] = willing_tracker.update(now, {
            "action": float(_ag.get("total_actions", 0.0) or 0.0),
            "social": _fc.get("SOCIAL", 0.0),
            "creative": _fc.get("ART", 0.0) + _fc.get("MUSIC", 0.0),
            "protective": float(_ov.get("rejected_count", 0.0) or 0.0),
            "exploration": _vocab + float(_mc.get("primitives_grounded", 0.0) or 0.0),
        })
        return sources

    return provider


def make_outer_spirit_provider(ctx: OuterSourceContext,
                               source_keys: Iterable[str]) -> Provider:
    # OSH ingests need system_sensor_stats (cpu_thermal/spike/circadian) — add
    # it to the assemble set even though it's not a spirit tensor SOURCE_KEY.
    keys = set(source_keys) | {"system_sensor_stats", "assessment_stats"}
    expr_tracker = ExpressionWindowTracker()
    osr_change_tracker = ChangeBreathTracker()

    def provider() -> dict:
        sources = assemble_outer_sources(keys, ctx)
        now = time.time()
        # ── outer_spirit_history_stats (OSH accumulator, re-homed here) ──
        osh = ctx.outer_spirit_history
        if osh is not None:
            try:
                _sys = sources.get("system_sensor_stats") or {}
                _assess = sources.get("assessment_stats") or {}
                osh.ingest_assessments(
                    _assess.get("recent") or [],
                    float(_sys.get("cpu_thermal", 0.0) or 0.0),
                    float(_sys.get("cpu_spike_rate", 0.0) or 0.0),
                    float(_sys.get("circadian_phase", 0.5) or 0.5))
                # π-pulse cadence for CHIT[26] circadian_alignment.
                _pi = ctx.shm_bank.read_pi_heartbeat()
                _ep = ctx.shm_bank.read_epoch()
                if isinstance(_pi, dict) and isinstance(_ep, dict):
                    osh.ingest_pi_cluster(_pi.get("pulse_count"), _ep.get("epoch"))
                # CHIT[14] self_trajectory ingest of the 45D snapshot.
                _os45 = ctx.shm_bank.read_outer_spirit_45d()
                _v45 = _os45.get("values") if isinstance(_os45, dict) else None
                if not _v45 and isinstance(_os45, dict):
                    _sat = _os45.get("SAT") or []
                    _chit = _os45.get("CHIT") or []
                    _ananda = _os45.get("ANANDA") or []
                    if len(_sat) == 15 and len(_chit) == 15 and len(_ananda) == 15:
                        _v45 = list(_sat) + list(_chit) + list(_ananda)
                if _v45 and len(_v45) == 45:
                    osh.ingest_outer_spirit_45d(_v45)
                sources["outer_spirit_history_stats"] = osh.get_stats()
            except Exception as _e:
                logger.debug("[OuterProviders] OSH ingest: %s", _e)
        _fc = _composite_fire_counts(ctx)
        _et = sources.get("expression_translator_stats") or {}
        _lang = sources.get("language_stats") or {}
        _vocab = float(_lang.get("vocab_total", 0.0) or 0.0)
        # expr_window — rich expression rolling-window (expression cluster).
        sources["expr_window"] = expr_tracker.update(now, {
            "image": _fc.get("ART", 0.0), "sound": _fc.get("MUSIC", 0.0),
            "speak": _fc.get("SPEAK", 0.0), "word": _vocab,
            "self_authored": float(_et.get("learned_actions", 0.0) or 0.0),
            "total": float(_et.get("total_actions", 0.0) or 0.0),
        })
        # outer_spirit_self_change — SAT[0] world_recognition = fast |Δ| of the
        # OTHER 44 outer_spirit dims (exclude observer [0]).
        try:
            _os45 = ctx.shm_bank.read_outer_spirit_45d()
            _vals = None
            if isinstance(_os45, dict):
                _vals = _os45.get("values")
                if not _vals:
                    _sat = _os45.get("SAT") or []
                    _chit = _os45.get("CHIT") or []
                    _ananda = _os45.get("ANANDA") or []
                    if len(_sat) == 15 and len(_chit) == 15 and len(_ananda) == 15:
                        _vals = list(_sat) + list(_chit) + list(_ananda)
            if _vals and len(_vals) == 45:
                _levels = {f"d{i}": float(_vals[i]) for i in range(1, 45)}
                _ch = osr_change_tracker.update(now, _levels)
                _changes = [_ch[f"d{i}_change"] for i in range(1, 45)]
                sources["outer_spirit_self_change"] = (
                    sum(_changes) / len(_changes) if _changes else 0.0)
        except Exception as _e:
            logger.debug("[OuterProviders] spirit self_change: %s", _e)
        return sources

    return provider
