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
        phase_warmup_end: int = 500,    # NEW: α=0.10 warm-up tier
        phase_0_end: int = 2000,         # was 500 — α=0.25
        phase_1_end: int = 5000,         # was 2000 — α=0.50
        phase_2_end: int = 10000,        # was 5000 — α=0.75
        phase_3_end: int = 20000,        # was 10000 — α=1.00 steady
        cold_start_n: int = 10,
        # RFP_meta-reasoning_CGN_FIX.md §4.4 time-escape hatch — if outcome
        # rate stalls, force α += increment every N seconds anchored to
        # last legitimate tier promotion. Prevents stuck-at-α=0.10 forever
        # if dispatch fires but consumer outcomes are sparse. Worst case:
        # α reaches 1.0 within 9 weeks of activation regardless of rate.
        time_escape_enabled: bool = True,
        time_escape_seconds_per_step: float = 604800.0,  # 7 days
        time_escape_increment: float = 0.10,
        time_escape_cap: float = 1.0,
    ):
        self._alpha_ramp_enabled = bool(alpha_ramp_enabled)
        # 5 boundaries: warmup, p0, p1, p2, p3 (5 tiers: 0.10/0.25/0.50/0.75/1.0)
        self._phase_boundaries = (
            int(phase_warmup_end),
            int(phase_0_end),
            int(phase_1_end),
            int(phase_2_end),
            int(phase_3_end),
        )
        self._cold_start_n = int(cold_start_n)

        # Time-escape hatch state — RFP §4.4
        self._time_escape_enabled = bool(time_escape_enabled)
        self._time_escape_seconds = float(time_escape_seconds_per_step)
        self._time_escape_increment = float(time_escape_increment)
        self._time_escape_cap = float(time_escape_cap)
        self._time_escape_alpha_boost = 0.0  # cumulative time-escape addition
        # Anchor for the 7-day timer. Reset on every legitimate tier
        # promotion so normal-cadence ramping doesn't double-step.
        self._last_tier_promotion_ts = time.time()
        self._last_observed_tier_index = 0  # 0=disabled, 1=warm, 2=p0, ...

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
        """Return current blend α per rFP §7.3 + RFP_meta-reasoning_CGN_FIX.md
        §4.4 count-based schedule with optional time-escape additive.

        Schedule (RFP §4.4 — gentler entry to avoid policy thrashing when
        the policy is currently 100% FORMULATE-dominant):
          0–500   → α = 0.10  (NEW gentler warm-up tier)
          500–2000 → α = 0.25
          2000–5000 → α = 0.50
          5000–10000 → α = 0.75
          10000+ → α = 1.00 (steady)
        Plus time-escape boost (capped at time_escape_cap, default 1.0).

        When alpha_ramp_enabled is False → returns 0.0 regardless.
        """
        if not self._alpha_ramp_enabled:
            return 0.0
        base = self._count_based_alpha()
        if self._time_escape_enabled:
            base = min(self._time_escape_cap,
                       base + self._compute_time_escape_boost())
        return base

    def _count_based_alpha(self) -> float:
        """5-tier count-based α (no time-escape applied)."""
        n = self._total_outcomes
        pw, p0, p1, p2, p3 = self._phase_boundaries
        if n < pw:
            return 0.10  # NEW warm-up tier
        if n < p0:
            return 0.25
        if n < p1:
            return 0.50
        if n < p2:
            return 0.75
        # n >= p2 → steady at 1.0 (count-based; time-escape can't go higher)
        return 1.0

    def _current_tier_index(self) -> int:
        """Return integer tier index for the current outcome count.
        0=disabled / 1=warm_up / 2=p0 / 3=p1 / 4=p2 / 5=steady. Used to
        detect tier promotions for time-escape anchor reset.
        """
        if not self._alpha_ramp_enabled:
            return 0
        n = self._total_outcomes
        pw, p0, p1, p2, p3 = self._phase_boundaries
        if n < pw:
            return 1
        if n < p0:
            return 2
        if n < p1:
            return 3
        if n < p2:
            return 4
        return 5

    def _compute_time_escape_boost(self) -> float:
        """RFP_meta-reasoning_CGN_FIX.md §4.4 time-escape hatch.

        Every `time_escape_seconds_per_step` (default 7 days) since the
        last legitimate tier promotion, add `time_escape_increment`
        (default 0.10) to the base count-based α. Cumulative boost capped
        at `time_escape_cap` (default 1.0).

        Tier promotion (via _count_based_alpha crossing a boundary)
        resets the timer + clears the accumulated boost so normal-cadence
        ramping doesn't double-step.
        """
        if not self._time_escape_enabled:
            return 0.0
        # First-call initialization: capture current tier without claiming
        # a promotion event. Only subsequent strict-greater transitions
        # count as promotions that reset the escape timer.
        current_tier = self._current_tier_index()
        if self._last_observed_tier_index == 0:
            self._last_observed_tier_index = current_tier
        elif current_tier > self._last_observed_tier_index:
            self._last_observed_tier_index = current_tier
            self._last_tier_promotion_ts = time.time()
            self._time_escape_alpha_boost = 0.0
            logger.info(
                "[DynamicRewards] α tier promoted to %d (count-based "
                "α=%.2f) — time-escape timer reset",
                current_tier, self._count_based_alpha())
            return 0.0
        # Accumulate +increment every N seconds anchored to last promotion
        elapsed = time.time() - self._last_tier_promotion_ts
        if elapsed >= self._time_escape_seconds:
            steps = int(elapsed // self._time_escape_seconds)
            new_boost = min(self._time_escape_cap,
                            steps * self._time_escape_increment)
            if new_boost > self._time_escape_alpha_boost:
                logger.info(
                    "[DynamicRewards] time-escape boost %.2f → %.2f "
                    "(%d×%.2f after %.1f days since tier %d promotion at "
                    "count=%d) — RFP §4.4 escape hatch active",
                    self._time_escape_alpha_boost, new_boost,
                    steps, self._time_escape_increment,
                    elapsed / 86400.0, self._last_observed_tier_index,
                    self._total_outcomes)
                self._time_escape_alpha_boost = new_boost
        return self._time_escape_alpha_boost

    def current_phase(self) -> str:
        """String label for the current ramp phase — diagnostic output."""
        if not self._alpha_ramp_enabled:
            return "disabled"
        n = self._total_outcomes
        pw, p0, p1, p2, p3 = self._phase_boundaries
        if n < pw:
            return "warm_up"     # α=0.10
        if n < p0:
            return "phase_0"     # α=0.25
        if n < p1:
            return "phase_1"     # α=0.50
        if n < p2:
            return "phase_2"     # α=0.75
        return "steady"          # α=1.00

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
