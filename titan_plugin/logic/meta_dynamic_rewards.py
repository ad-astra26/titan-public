"""
Meta-Reasoning Dynamic Rewards — accumulator + α-ramp + blended step reward.

Emergent-reward layer (rFP §7 / Upgrade D). Replaces the hand-crafted
STEP_REWARDS table with a consumer-outcome-driven rolling mean per
(consumer, primitive, sub_mode) tuple. Signed rewards in [-1, +1] per
Maker decision — negative feedback teaches meta what NOT to do.

Session 1 (this commit) ships:
    - accumulator infrastructure (collecting outcomes from META_REASON_OUTCOME)
    - α ramp calculation per count-based schedule (rFP §7.3)
    - compute_step_reward(primitive, sub_mode, consumer_context=None)
      with cold-start guard and honest α=0.0 enforcement when ramp disabled
    - get_stats() for /v4/meta-service/rewards
    - α HARD-WIRED to 0.0 via config [meta_service_interface] alpha_ramp_enabled
      — Session 3 flips to true after first 500-outcome threshold

The blend hook is NOT yet called from meta_reasoning's chain loop in Session
1 (chains are still self-driven; consumer_context threads through in
Session 2 when real META_REASON_REQUEST processing lands). This commit
ships the accumulator + schedule so outcome data accumulates now, and
Session 2's blend logic lands with a full dataset.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DynamicRewardAccumulator:
    """Rolling-mean outcome tracker per (consumer, primitive, sub_mode).

    Thread-safe. Bounded memory via implicit upper bound on the tuple
    cardinality (9 consumers × 9 primitives × ≤8 sub-modes = ~648 tuples).
    """

    def __init__(
        self,
        alpha_ramp_enabled: bool = False,
        phase_0_end: int = 500,
        phase_1_end: int = 2000,
        phase_2_end: int = 5000,
        phase_3_end: int = 10000,
        cold_start_n: int = 10,
    ):
        self._alpha_ramp_enabled = bool(alpha_ramp_enabled)
        self._phase_boundaries = (
            int(phase_0_end),
            int(phase_1_end),
            int(phase_2_end),
            int(phase_3_end),
        )
        self._cold_start_n = int(cold_start_n)

        # (consumer, primitive, sub_mode) → rolling mean
        self._rolling_mean: dict = {}
        # (consumer, primitive, sub_mode) → count
        self._count: dict = {}
        # Total outcomes across all tuples (drives α ramp)
        self._total_outcomes = 0
        # Per-consumer negative-outcome rate (for monitoring)
        self._neg_count: dict = collections.Counter()
        self._pos_count: dict = collections.Counter()

        self._lock = threading.Lock()
        self._t_boot = time.time()

    # ── Outcome ingestion ───────────────────────────────────────────

    def record_outcome(
        self,
        consumer_id: str,
        primitive_sequence: list,
        sub_modes: Optional[list],
        outcome_reward: float,
    ) -> int:
        """Apply outcome to each (consumer, primitive, sub_mode) tuple
        in the chain. Returns the number of tuples updated.

        primitive_sequence: ["FORMULATE", "RECALL", "HYPOTHESIZE", ...]
        sub_modes: ["define", "wisdom", "generate", ...] — must be same
            length as primitive_sequence. If None, each sub_mode is "_all".

        The chain outcome reward propagates to every step. This is a
        simple credit-assignment model — Session 3 can add step-level
        differential rewards once we see data.
        """
        try:
            reward = max(-1.0, min(1.0, float(outcome_reward)))
        except (TypeError, ValueError):
            return 0
        if not primitive_sequence:
            return 0
        modes = (list(sub_modes) if sub_modes
                 else ["_all"] * len(primitive_sequence))
        # Pad/truncate modes to match primitives
        if len(modes) < len(primitive_sequence):
            modes.extend(["_all"] * (len(primitive_sequence) - len(modes)))
        elif len(modes) > len(primitive_sequence):
            modes = modes[:len(primitive_sequence)]

        updated = 0
        with self._lock:
            for prim, mode in zip(primitive_sequence, modes):
                key = (consumer_id, prim, mode)
                n = self._count.get(key, 0)
                prev = self._rolling_mean.get(key, 0.0)
                # Incremental mean
                new_mean = prev + (reward - prev) / (n + 1)
                self._rolling_mean[key] = new_mean
                self._count[key] = n + 1
                updated += 1
            self._total_outcomes += 1
            if reward < 0:
                self._neg_count[consumer_id] += 1
            elif reward > 0:
                self._pos_count[consumer_id] += 1
        return updated

    def record_single_step(
        self,
        consumer_id: str,
        primitive: str,
        sub_mode: str,
        outcome_reward: float,
    ) -> bool:
        """Apply outcome to a single (consumer, primitive, sub_mode). Returns
        True on accept, False on invalid input.

        Used when Session 2's chain execution threads per-step outcomes
        through (rather than the whole-chain summary reward).
        """
        try:
            reward = max(-1.0, min(1.0, float(outcome_reward)))
        except (TypeError, ValueError):
            return False
        if not consumer_id or not primitive or not sub_mode:
            return False
        key = (consumer_id, primitive, sub_mode)
        with self._lock:
            n = self._count.get(key, 0)
            prev = self._rolling_mean.get(key, 0.0)
            self._rolling_mean[key] = prev + (reward - prev) / (n + 1)
            self._count[key] = n + 1
            self._total_outcomes += 1
            if reward < 0:
                self._neg_count[consumer_id] += 1
            elif reward > 0:
                self._pos_count[consumer_id] += 1
        return True

    def ingest_outcome_record(self, record: dict) -> int:
        """Wire target for MetaService.outcome_sink. Expects the record
        shape emitted by MetaService.handle_outcome (or None-safe).

        Single-step update: no primitive_sequence yet in Session 1 outcomes
        (real chains haven't run). We still increment the outcome counter
        so the α ramp can track total receipts, and record_single_step
        is used IF actual_primitive_used is provided in the record.
        """
        if not isinstance(record, dict):
            return 0
        consumer_id = record.get("consumer_id", "")
        reward = record.get("outcome_reward")
        prim = record.get("actual_primitive_used")
        if consumer_id and prim and reward is not None:
            ok = self.record_single_step(consumer_id, prim, "_all", reward)
            return 1 if ok else 0
        # No primitive reported — just bump the totals.
        try:
            r = max(-1.0, min(1.0, float(reward)))
        except (TypeError, ValueError):
            return 0
        with self._lock:
            self._total_outcomes += 1
            if r < 0:
                self._neg_count[consumer_id] += 1
            elif r > 0:
                self._pos_count[consumer_id] += 1
        return 1

    # ── α ramp ──────────────────────────────────────────────────────

    def current_alpha(self) -> float:
        """Return current blend α per rFP §7.3 count-based schedule.

        When alpha_ramp_enabled is False → returns 0.0 regardless (Session 1
        hard-wire, honest about "infrastructure present but not active").
        """
        if not self._alpha_ramp_enabled:
            return 0.0
        n = self._total_outcomes
        p0, p1, p2, p3 = self._phase_boundaries
        if n < p0:
            return 0.0
        if n < p1:
            return 0.25
        if n < p2:
            return 0.50
        if n < p3:
            return 0.75
        return 1.0

    def current_phase(self) -> str:
        """String label for the current ramp phase — diagnostic output."""
        if not self._alpha_ramp_enabled:
            return "disabled"
        n = self._total_outcomes
        p0, p1, p2, p3 = self._phase_boundaries
        if n < p0:
            return "warm_up"
        if n < p1:
            return "phase_1"
        if n < p2:
            return "phase_2"
        if n < p3:
            return "phase_3"
        return "steady"

    # ── Blended step reward (call site in meta_reasoning) ───────────

    def blend_step_reward(
        self,
        static_reward: float,
        primitive: str,
        sub_mode: str,
        consumer_context: Optional[str] = None,
    ) -> float:
        """Return blended step reward per rFP §7.2.

            (1 - α) * static + α * dynamic_mean  (if n ≥ cold_start_n)
            static                               (otherwise)

        When consumer_context is None (chain is self-driven, not
        consumer-requested) — returns static unchanged. Same when α=0.

        Meta-reasoning's chain loop will call this in Session 2 with a real
        consumer_context when the chain originates from a META_REASON_REQUEST.
        Session 1 chains still pass consumer_context=None so this is a no-op.
        """
        alpha = self.current_alpha()
        if alpha <= 0.0 or consumer_context is None:
            return float(static_reward)
        key = (consumer_context, primitive, sub_mode)
        with self._lock:
            n = self._count.get(key, 0)
            if n < self._cold_start_n:
                return float(static_reward)
            dynamic = self._rolling_mean.get(key, 0.0)
        return (1.0 - alpha) * float(static_reward) + alpha * float(dynamic)

    # ── Stats export ────────────────────────────────────────────────

    def get_stats(self, top_n: int = 10) -> dict:
        """Snapshot for /v4/meta-service/rewards. Includes α state, phase,
        top-N tuples by count, per-consumer negative-outcome rate."""
        with self._lock:
            total = self._total_outcomes
            tuples_tracked = len(self._count)
            # Top N tracked tuples by observation count
            top = sorted(
                self._count.items(), key=lambda kv: -kv[1])[:top_n]
            top_list = []
            for (consumer, prim, mode), n in top:
                mean = self._rolling_mean.get((consumer, prim, mode), 0.0)
                top_list.append({
                    "consumer": consumer,
                    "primitive": prim,
                    "sub_mode": mode,
                    "n": n,
                    "rolling_mean_reward": round(mean, 4),
                })
            neg_rate = {}
            pos_rate = {}
            for c, nn in self._neg_count.items():
                nn = float(nn)
                nd = float(self._neg_count.get(c, 0) + self._pos_count.get(c, 0))
                if nd > 0:
                    neg_rate[c] = round(nn / nd, 3)
            for c, pp in self._pos_count.items():
                nn = float(pp)
                nd = float(self._neg_count.get(c, 0) + self._pos_count.get(c, 0))
                if nd > 0:
                    pos_rate[c] = round(nn / nd, 3)

        return {
            "alpha_ramp_enabled": self._alpha_ramp_enabled,
            "current_alpha": self.current_alpha(),
            "current_phase": self.current_phase(),
            "phase_boundaries": list(self._phase_boundaries),
            "total_outcomes": total,
            "tuples_tracked": tuples_tracked,
            "cold_start_n": self._cold_start_n,
            "top_by_count": top_list,
            "per_consumer_negative_rate": neg_rate,
            "per_consumer_positive_rate": pos_rate,
            "uptime_seconds": round(time.time() - self._t_boot, 1),
        }
