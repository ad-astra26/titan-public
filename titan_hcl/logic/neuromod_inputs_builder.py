"""Neuromod inputs builder — cognitive_worker side of the §4.Q bridge.

Lifted from `spirit_worker.py:4074-4327` (DEAD under Phase C) with the same
math preserved exactly. Builds the 11 emergent inputs that
`compute_emergent_inputs(...)` consumes, plus chi_health + topology_velocity
+ dt, and packs them into a single payload that cognitive_worker writes to
the `neuromod_inputs.bin` SHM slot for `neuromod_worker` to consume in its
evaluate driver.

The state cache (prev_drift, prev_curvature, EMA epoch interval/regularity,
prev NS transitions, prev filter_down count, last epoch time/id) was
formerly attached to `NeuromodulatorSystem` (the live in-process instance
in spirit_worker). Under §4.Q, that instance lives in neuromod_worker —
cognitive_worker keeps its OWN cache here because the deltas are computed
on cognitive_worker's epoch tick, not on neuromod_worker's tick.

Added 2026-05-15 (§4.Q neuromod_worker.evaluate migration) — see
`titan-docs/PLAN_microkernel_phase_c_neuromod_worker_evaluate_migration.md`.
"""
from __future__ import annotations

import math
import time
from typing import Any

from titan_hcl.logic.neuromodulator import compute_emergent_inputs


class NeuromodInputsBuilder:
    """Per-tick aggregator + state cache for the 11 emergent neuromod inputs.

    Construction is cheap (no I/O). One instance per cognitive_worker. Call
    `build(...)` each consciousness epoch with the current engine snapshot
    + a cached bundle of bus-event payloads (expression composites,
    prediction stats, kin signature).
    """

    def __init__(self, dna: dict | None = None):
        # DNA — neuromodulator coupling weights from titan_params.toml
        # [neuromodulator_dna]. Default = {}; compute_emergent_inputs uses
        # per-modulator fallback values when keys are absent.
        self.dna: dict = dict(dna or {})

        # State cache (mirrors what NeuromodulatorSystem held pre-§4.Q).
        self._prev_drift: float = 0.0
        self._prev_curvature: float = 0.0
        self._prev_ns_transitions: int = 0
        self._prev_filter_down_count: int = 0
        self._last_epoch_time: float = time.time()
        self._last_epoch_id: int | None = None
        self._ema_epoch_interval: float = 7.5
        self._ema_epoch_regularity: float = 0.5

    # ─────────────────────────────────────────────────────────────────────
    # Snapshot extraction helpers — each returns one of the 9 input subdicts.
    # All defensive against None engines (default values match the spirit
    # implementation pre-§4.Q).
    # ─────────────────────────────────────────────────────────────────────

    def _sphere_balance(self, sphere_clocks_snap: dict | None) -> dict[str, float]:
        """sphere_balance dict from SHM-read sphere_clocks payload.

        sphere_clocks_snap shape:
            {"inner_body": {"consecutive_balanced": int, ...},
             "outer_body": {...}, "inner_mind": {...}, "outer_mind": {...}, ...}
        """
        bal_norm = float(self.dna.get("balance_streak_normalization", 100))
        out: dict[str, float] = {}
        for name in ("inner_body", "inner_mind", "outer_body", "outer_mind"):
            snap = (sphere_clocks_snap or {}).get(name) or {}
            streak = float(snap.get("consecutive_balanced", 0) or 0)
            out[name] = min(1.0, streak / bal_norm) if bal_norm > 0 else 0.0
        return out

    def _trinity_coherence(self, latest_epoch: dict | None) -> dict[str, float]:
        """Trinity coherence dict from coordinator.consciousness latest_epoch."""
        out = {"inner": 0.5, "outer": 0.5}
        if not latest_epoch:
            return out
        for tri_key, tri_dims in (("inner", ("iB", "iM", "iS")),
                                  ("outer", ("oB", "oM", "oS"))):
            vals = [float(latest_epoch.get(d, 0.5)) for d in tri_dims]
            mean = sum(vals) / 3.0
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 3.0)
            out[tri_key] = max(0.0, 1.0 - std / 0.4)
        return out

    def _chi_state(self, life_force_engine) -> dict[str, float]:
        """chi_state dict from life_force_engine._prev + _metabolic_drain."""
        prev = getattr(life_force_engine, "_prev", {}) if life_force_engine else {}
        vals = [float(prev.get(k, 0.5) or 0.5) for k in ("spirit", "mind", "body")]
        mean = sum(vals) / 3.0
        var = sum((v - mean) ** 2 for v in vals) / 3.0
        return {
            "total": mean,
            "body": float(prev.get("body", 0.5) or 0.5),
            "circulation": max(0.0, 1.0 - var / 0.08),
            "drain": float(getattr(life_force_engine, "_metabolic_drain", 0.0) or 0.0)
            if life_force_engine else 0.0,
        }

    def _consciousness_dynamics(self, latest_epoch: dict | None, now: float) -> dict[str, float]:
        """consciousness_dynamics dict — uses state cache for drift_delta + epoch_gap_ratio."""
        latest = latest_epoch or {}
        drift = float(latest.get("drift_magnitude", 0.0) or 0.0)
        drift_delta = min(1.0, abs(drift - self._prev_drift) * 100.0)
        self._prev_drift = drift
        density = float(latest.get("density", 0.0) or 0.0)

        # Epoch gap ratio: how close are epochs firing to MIN_GAP?
        epoch_gap = max(1.0, now - self._last_epoch_time)
        epoch_id = latest.get("epoch_id")
        if epoch_id is not None and epoch_id != self._last_epoch_id:
            self._last_epoch_time = now
            self._last_epoch_id = epoch_id
        epoch_gap_ratio = min(1.0, 8.0 / max(1.0, epoch_gap))

        return {
            "drift_magnitude": drift,
            "drift_delta": drift_delta,
            "density": density,
            "epoch_gap_ratio": epoch_gap_ratio,
        }

    def _pi_state(self, pi_monitor, latest_epoch: dict | None, now: float) -> dict[str, float]:
        """pi_state dict — uses state cache for EMAs + curvature delta."""
        latest = latest_epoch or {}
        epoch_id = latest.get("epoch_id")

        # Update interval EMA on new-epoch transitions only (skip restart outliers).
        epoch_gap = max(1.0, now - self._last_epoch_time)
        is_new_epoch = epoch_id is not None and epoch_id != self._last_epoch_id
        # NOTE: _last_epoch_id update is in _consciousness_dynamics — order of
        # calls matters in build(). We read it BEFORE the dynamics call.
        if is_new_epoch and epoch_gap < 120:
            self._ema_epoch_interval = self._ema_epoch_interval * 0.98 + epoch_gap * 0.02
            dev = abs(epoch_gap - self._ema_epoch_interval) / max(self._ema_epoch_interval, 1.0)
            instant_reg = max(0.0, 1.0 - dev)
            self._ema_epoch_regularity = self._ema_epoch_regularity * 0.95 + instant_reg * 0.05

        curv_now = float(latest.get("curvature", 0.0) or 0.0)
        curv_delta = min(1.0, abs(curv_now - self._prev_curvature))
        self._prev_curvature = curv_now

        epoch_id_int = int(epoch_id or 0)
        epoch_maturity = min(1.0, epoch_id_int / 50000.0)

        return {
            "regularity": float(getattr(pi_monitor, "heartbeat_ratio", 0.0) or 0.0)
            if pi_monitor else 0.0,
            "epoch_regularity": self._ema_epoch_regularity,
            "cluster_streak": min(
                1.0,
                float(getattr(pi_monitor, "_current_pi_streak", 0) or 0) / 20.0)
            if pi_monitor else 0.0,
            "developmental_age": float(getattr(pi_monitor, "developmental_age", 0.0) or 0.0)
            if pi_monitor else 0.0,
            "curvature_delta": curv_delta,
            "epoch_maturity": epoch_maturity,
        }

    def _prediction_state(self, prediction_stats: dict | None, ex_mem) -> dict[str, float]:
        """prediction_state dict from cached PREDICTION_STATS_UPDATED + ex_mem stats."""
        # surprise — from cached PREDICTION_STATS_UPDATED bus event (now lives
        # in self_reflection_worker per Track 2). Falls back to 0.0 on cold start.
        surprise = float((prediction_stats or {}).get("novelty_signal", 0.0) or 0.0)

        action_outcome = 0.5
        success_rate = 0.5
        if ex_mem is not None:
            try:
                stats = ex_mem.get_stats() or {}
                types = stats.get("by_type", {}) or {}
                if types:
                    scores = [t.get("avg_score", 0.5) for t in types.values()
                              if isinstance(t, dict) and t.get("avg_score") is not None]
                    if scores:
                        action_outcome = sum(scores) / len(scores)
                    srs = [t.get("success_rate", 0.5) for t in types.values()
                           if isinstance(t, dict) and t.get("success_rate") is not None]
                    if srs:
                        success_rate = sum(srs) / len(srs)
            except Exception:
                pass

        return {
            "surprise": surprise,
            "action_outcome": action_outcome,
            "success_rate": success_rate,
        }

    def _ns_state(self, neural_nervous_system, filter_down_count: int) -> dict[str, float]:
        """ns_state dict — uses state cache for transition + filter_down deltas."""
        transition_delta = 0.0
        if neural_nervous_system is not None:
            trans_now = int(getattr(neural_nervous_system, "_total_transitions", 0) or 0)
            transition_delta = min(1.0, max(0, trans_now - self._prev_ns_transitions) / 20.0)
            self._prev_ns_transitions = trans_now

        fd_delta = min(1.0, max(0, filter_down_count - self._prev_filter_down_count) / 3.0)
        self._prev_filter_down_count = filter_down_count

        return {
            "transition_delta": transition_delta,
            "filter_down_writes": fd_delta,
        }

    def _expression_state(self, expression_stats: dict | None) -> dict[str, float]:
        """expression_state dict from cached EXPRESSION_COMPOSITES_UPDATED payload.

        Expected payload shape (per expression_worker §4.B): a dict keyed by
        composite name → per-composite stats dict containing fire_count +
        evaluation_count fields. Default (no composites): fire_rate=0.0,
        alignment=0.5.
        """
        fire_rate = 0.0
        alignment = 0.5
        if expression_stats:
            try:
                fc = [c.get("fire_count", 0) for c in expression_stats.values()
                      if isinstance(c, dict)]
                ec = [c.get("evaluation_count", 1) for c in expression_stats.values()
                      if isinstance(c, dict)]
                if ec and sum(ec) > 0:
                    alignment = sum(fc) / sum(ec)
                    fire_rate = alignment  # fire/eval ratio IS the fire rate (pre-§4.Q math)
            except Exception:
                pass
        return {"fire_rate": min(1.0, fire_rate), "alignment": alignment}

    def _resonance_state(self, kin_signature: dict | None, resonance_count: int) -> dict[str, float]:
        """resonance_state dict — Endorphin-source field from kin resonance."""
        # resonant_fraction = resonance.resonant_count() / 3.0 in spirit pre-§4.Q.
        # Under Phase C, the kin signature payload (from outer_interface_worker)
        # carries resonant_count + last_resonance.
        return {"resonant_fraction": min(1.0, resonance_count / 3.0)}

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        coordinator,
        neural_nervous_system,
        life_force_engine,
        pi_monitor,
        ex_mem,
        sphere_clocks_snap: dict | None,
        latest_epoch: dict | None,
        is_dreaming: bool,
        prediction_stats: dict | None,
        expression_stats: dict | None,
        kin_signature: dict | None,
        filter_down_count: int = 0,
        resonance_count: int = 0,
        topology_velocity: float = 0.3,
        dt: float = 1.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Build the SHM payload for `neuromod_inputs.bin`.

        Returns a msgpack-friendly dict with:
            inputs: {DA, 5HT, NE, ACh, Endorphin, GABA} ∈ [0, 1]
            chi_health: float ∈ [0.1, 1.0]
            topology_velocity: float ∈ [0, 1]
            developmental_age: float
            dt: float
            kin_overrides: optional dict for the 4 dead-but-preserved kin_*
                           keys (kin_da/kin_endorphin/kin_5ht/kin_ne — never
                           read by NeuromodulatorSystem.evaluate but written
                           for pre-§4.Q parity)
            ts: float (wall time of build)
        """
        now_t = float(now or time.time())

        # NOTE: call order matters — _pi_state reads self._last_epoch_id BEFORE
        # _consciousness_dynamics updates it. Preserve this order.
        pi_state = self._pi_state(pi_monitor, latest_epoch, now_t)
        consciousness_dynamics = self._consciousness_dynamics(latest_epoch, now_t)
        sphere_balance = self._sphere_balance(sphere_clocks_snap)
        trinity_coherence = self._trinity_coherence(latest_epoch)
        chi_state = self._chi_state(life_force_engine)
        prediction_state = self._prediction_state(prediction_stats, ex_mem)
        ns_state = self._ns_state(neural_nervous_system, filter_down_count)
        expression_state = self._expression_state(expression_stats)
        resonance_state = self._resonance_state(kin_signature, resonance_count)

        # Compute the 6-modulator inputs (the actual evaluate(...) signature).
        inputs = compute_emergent_inputs(
            sphere_balance=sphere_balance,
            trinity_coherence=trinity_coherence,
            chi_state=chi_state,
            consciousness_dynamics=consciousness_dynamics,
            pi_state=pi_state,
            prediction_state=prediction_state,
            ns_state=ns_state,
            expression_state=expression_state,
            resonance_state=resonance_state,
            is_dreaming=bool(is_dreaming),
            dna=self.dna,
        )

        # chi_health = derived from metabolic drain (preserves pre-§4.Q math).
        drain = chi_state.get("drain", 0.0)
        chi_health = max(0.1, 1.0 - drain * 0.6)

        # kin signal → 4 boost overrides (dead code in evaluate but preserved).
        kin_overrides: dict[str, float] = {}
        kin = kin_signature or {}
        kin_res = float(kin.get("last_resonance", 0.0) or 0.0)
        kin_recency = max(0.0, 1.0 - (now_t - float(kin.get("last_exchange_ts", 0) or 0)) / 3600.0)
        kin_signal = kin_res * kin_recency
        if kin_signal > 0.01:
            kin_dna = self.dna.get("kin", {}).get("dna", {}) if isinstance(
                self.dna.get("kin"), dict) else {}
            kin_overrides = {
                "kin_da": kin_signal * float(kin_dna.get("da_boost", 0.25)),
                "kin_endorphin": kin_signal * float(kin_dna.get("endorphin_boost", 0.20)),
                "kin_5ht": kin_signal * float(kin_dna.get("sht_boost", 0.15)),
                "kin_ne": kin_signal * float(kin_dna.get("ne_boost", 0.10)),
            }

        # Developmental age — for downstream nudge gating in neuromod_worker.
        developmental_age = pi_state.get("developmental_age", 0.0)

        return {
            "inputs": inputs,
            "chi_health": float(chi_health),
            "topology_velocity": float(topology_velocity),
            "developmental_age": float(developmental_age),
            "dt": float(dt),
            "kin_overrides": kin_overrides,
            "is_dreaming": bool(is_dreaming),
            "ts": now_t,
        }
