"""
titan_plugin/logic/meta_teacher.py — Philosopher-critic for meta-reasoning chains.

Observes completed meta-reasoning chains, evaluates each along principled
reasoning criteria via LLM critic, emits feedback for IQL reward shaping +
META-CGN β-posterior nudging.

Pure-logic class (no bus, no subprocess imports) — called from
meta_teacher_worker. Follows LanguageTeacher pattern for testability.

rFP: titan-docs/rFP_titan_meta_reasoning_teacher.md
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import deque
from typing import Optional

from titan_plugin.logic.meta_teacher_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    build_user_prompt,
)

logger = logging.getLogger("titan.meta_teacher")


class MetaTeacher:
    """Philosopher-critic evaluating meta-reasoning chains.

    Responsibilities:
      - Decide whether to critique a chain (sampling: uncertainty-gated +
        random + domain balance + rate cap)
      - Call LLM via provided ainvoke() coroutine; parse JSON response
      - Compute IQL reward_weight per ramp schedule (rFP §5.1)
      - Track adoption rate EMAs per (domain, context_type) to close the
        collaboration loop (rFP §5.3)

    The worker wraps this with bus I/O + persistence.
    """

    _REWARD_WEIGHT_HARD_CAP = 0.30  # §8 safeguard — hard-coded, not just config

    def __init__(self, config: dict):
        """Build the teacher. config is the [meta_teacher] TOML section."""
        self._enabled = bool(config.get("enabled", False))
        self._sample_mode = str(config.get("sample_mode", "uncertainty_plus_random"))
        self._uncertainty_threshold = float(config.get("uncertainty_threshold", 0.4))
        self._random_rate = float(config.get("random_sample_rate", 0.15))
        self._max_per_hour = int(config.get("max_critiques_per_hour", 30))
        self._domain_floor = float(config.get("domain_balance_floor", 0.05))
        self._reward_weight_config = float(config.get("reward_weight", 0.05))
        self._reward_weight_cap = min(
            self._REWARD_WEIGHT_HARD_CAP,
            float(config.get("reward_weight_cap", 0.30)))
        self._grounding_weight = float(config.get("grounding_weight", 0.15))
        self._ramp_p1 = int(config.get("ramp_phase1_critiques", 1000))
        self._ramp_p2 = int(config.get("ramp_phase2_critiques", 1500))
        self._llm_timeout = float(config.get("llm_timeout_s", 30.0))
        self._task_key = str(config.get("task_key", "meta_teacher"))

        # Sliding 1-hour window of critique timestamps for rate cap
        self._critique_times: deque = deque()
        # Per-domain critique count in rolling 1-hour window (for balance)
        self._critique_times_by_domain: dict[str, deque] = {}
        # Adoption metrics (rFP §5.3): EMA of whether Titan adopts suggestions
        # Keyed by domain → EMA in [0, 1]. Updated externally by worker.
        self.adoption_ema_by_domain: dict[str, float] = {}
        # Total critiques issued (lifetime, load/save via worker)
        self.total_critiques: int = 0
        # Total chains observed (for sample-rate telemetry)
        self.total_observed: int = 0
        # Last-N critiques for /v4 API (in-memory, worker also persists to jsonl)
        self.recent_critiques: deque = deque(maxlen=100)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reward_weight_cap(self) -> float:
        return self._reward_weight_cap

    def compute_reward_weight(self) -> float:
        """Ramp schedule per rFP §5.1.

        Phase 0 (0 → ramp_phase1): flat at reward_weight_config (0.05 default)
        Phase 1 (ramp_phase1 → ramp_phase2): linear ramp to target (0.15 default)
        Phase 2 (> ramp_phase2): stable at target

        Hard-capped at reward_weight_cap in all phases.
        """
        n = self.total_critiques
        start = self._reward_weight_config
        target = min(self._reward_weight_cap, 0.15)  # target phase-2 value
        if n < self._ramp_p1:
            w = start
        elif n < self._ramp_p2:
            span = max(1, self._ramp_p2 - self._ramp_p1)
            frac = (n - self._ramp_p1) / span
            w = start + (target - start) * frac
        else:
            w = target
        return float(min(self._reward_weight_cap, max(0.0, w)))

    def _prune_old(self, now: float) -> None:
        """Drop entries older than 1 hour from rate-cap windows."""
        cutoff = now - 3600.0
        while self._critique_times and self._critique_times[0] < cutoff:
            self._critique_times.popleft()
        for d, dq in self._critique_times_by_domain.items():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def should_sample(self, payload: dict, rng: Optional[random.Random] = None) -> tuple[bool, str]:
        """Decide whether to critique this chain.

        Returns (sample: bool, reason: str) — reason is "uncertainty" |
        "random" | "rate_cap" | "disabled" | "domain_starved" | "skipped".
        """
        self.total_observed += 1
        if not self._enabled:
            return False, "disabled"

        rng = rng or random
        now = time.time()
        self._prune_old(now)

        # Hard rate cap comes first
        if len(self._critique_times) >= self._max_per_hour:
            return False, "rate_cap"

        domain = str(payload.get("domain", "general"))
        chain_iql_conf = float(payload.get("chain_iql_confidence", 0.5))

        # Domain-balance boost: if any domain is under its floor and THIS chain
        # is from that starved domain, force-sample to maintain coverage.
        total_recent = max(1, len(self._critique_times))
        domain_share = (
            len(self._critique_times_by_domain.get(domain, ())) / total_recent)
        if total_recent >= 10 and domain_share < self._domain_floor:
            return True, "domain_starved"

        mode = self._sample_mode
        if mode == "random_only":
            if rng.random() < self._random_rate:
                return True, "random"
            return False, "skipped"
        if mode == "uncertainty_only":
            if chain_iql_conf < self._uncertainty_threshold:
                return True, "uncertainty"
            return False, "skipped"
        # Default: uncertainty_plus_random
        if chain_iql_conf < self._uncertainty_threshold:
            return True, "uncertainty"
        if rng.random() < self._random_rate:
            return True, "random"
        return False, "skipped"

    def _record_sample(self, payload: dict) -> None:
        """Mark that a critique was sampled (called after LLM succeeds)."""
        now = time.time()
        self._critique_times.append(now)
        domain = str(payload.get("domain", "general"))
        self._critique_times_by_domain.setdefault(domain, deque()).append(now)
        self.total_critiques += 1

    def parse_critique(
        self, llm_response: str,
        used_primitives: Optional[list] = None,
    ) -> Optional[dict]:
        """Parse LLM JSON response into a validated critique dict.

        Returns None if parsing fails or fields are malformed. Worker treats
        None as "neutral" (quality=0.5, confidence=0.0) per rFP §4.4 fallback.

        v2 (2026-04-24): `used_primitives` is optional but recommended —
        when supplied, `suggested_primitives` is defensively filtered to
        remove any primitive that appears in `used_primitives` (per v2
        prompt contract: suggestions must be from the NOT-USED set). This
        guards against LLM non-compliance; the critique_text field keeps
        its full content regardless.
        """
        if not llm_response:
            return None
        # Handle LLMs that wrap JSON in code fences or prose
        text = llm_response.strip()
        # Extract first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            quality = float(data.get("quality_score", 0.5))
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            return None
        cats = data.get("critique_categories") or []
        if not isinstance(cats, list):
            cats = []
        principles = data.get("principles_invoked") or []
        if not isinstance(principles, list):
            principles = []
        suggested = data.get("suggested_primitives") or []
        if not isinstance(suggested, list):
            suggested = []
        # v2: defensive filter — strip any suggested primitive that was
        # already used in the chain. v2 prompt demands "NOT USED only".
        suggested = [str(p)[:30] for p in suggested][:10]
        filtered_violations = 0
        if used_primitives is not None:
            used_set = {str(p) for p in used_primitives}
            before = len(suggested)
            suggested = [p for p in suggested if p not in used_set]
            filtered_violations = before - len(suggested)
        # Cap at 3 per v2 contract (LLMs occasionally still return more).
        if len(suggested) > 3:
            suggested = suggested[:3]
        critique_text = str(data.get("critique_text", ""))[:300]
        result = {
            "quality_score": max(0.0, min(1.0, quality)),
            "critique_categories": [str(c)[:40] for c in cats][:4],
            "critique_text": critique_text,
            "suggested_primitives": suggested,
            "confidence": max(0.0, min(1.0, confidence)),
            "principles_invoked": [str(p)[:30] for p in principles][:6],
        }
        if filtered_violations:
            result["filtered_v2_violations"] = filtered_violations
        return result

    def build_feedback_payload(
        self, payload: dict, critique: Optional[dict],
        retrieved_context_ids: Optional[list] = None,
    ) -> dict:
        """Build META_TEACHER_FEEDBACK payload from a (chain, critique) pair.

        On critique=None (LLM failed), emits neutral feedback (quality=0.5,
        confidence=0.0, reward_bonus=0) so downstream never blocks.

        rFP_meta_teacher_v2 Phase B: `retrieved_context_ids` enumerates the
        topic_keys of memory entries that informed this critique
        (observability only — chain_iql does not act on these). When
        teaching memory is disabled or no hits, omitted from payload.
        """
        chain_id = int(payload.get("chain_id", 0))
        if critique is None:
            out = {
                "chain_id": chain_id,
                "quality_score": 0.5,
                "critique_categories": [],
                "critique_text": "",
                "suggested_primitives": None,
                "confidence": 0.0,
                "reward_bonus": 0.0,
                "principles_invoked": [],
                "prompt_version": SYSTEM_PROMPT_VERSION,
                "llm_ok": False,
            }
            if retrieved_context_ids:
                out["retrieved_context_ids"] = list(retrieved_context_ids)
            return out
        reward_w = self.compute_reward_weight()
        out = {
            "chain_id": chain_id,
            "quality_score": critique["quality_score"],
            "critique_categories": critique["critique_categories"],
            "critique_text": critique["critique_text"],
            "suggested_primitives": (
                critique["suggested_primitives"] or None),
            "confidence": critique["confidence"],
            "reward_bonus": reward_w,  # alpha blend weight for apply_external_reward
            "principles_invoked": critique["principles_invoked"],
            "prompt_version": SYSTEM_PROMPT_VERSION,
            "llm_ok": True,
        }
        if retrieved_context_ids:
            out["retrieved_context_ids"] = list(retrieved_context_ids)
        return out

    def build_grounding_payloads(
        self, payload: dict, critique: Optional[dict]
    ) -> list[dict]:
        """One META_TEACHER_GROUNDING per primitive in the chain.

        Per rFP §5.2. Each entry feeds meta_cgn.handle_teacher_grounding()
        which does a β-posterior nudge with label_quality = critique score.
        Returns empty list if critique is None or teacher is disabled.
        """
        if critique is None or not self._enabled:
            return []
        chain_id = int(payload.get("chain_id", 0))
        primitives = list(payload.get("primitives_used", []))
        domain = str(payload.get("domain", "general"))
        ctx = payload.get("context_summary") or {}
        label_quality = critique["quality_score"]
        # Context fingerprint: hash of domain + dominant_emotion + impasse —
        # enough to cluster "similar situations" for the β nudge.
        emotion = str(ctx.get("dominant_emotion", ""))
        impasse = str(ctx.get("impasse_state", "none"))
        ctx_fp = f"{domain}|{emotion}|{impasse}"
        return [
            {
                "chain_id": chain_id,
                "primitive_id": p,
                "label_quality": label_quality,
                "ctx_fingerprint": ctx_fp,
                "grounding_weight": self._grounding_weight,
            }
            for p in primitives
        ]

    def update_adoption(self, domain: str, adopted: bool) -> float:
        """Update adoption-rate EMA for a domain. Returns new EMA.

        Called by worker when it observes a subsequent chain in `domain` and
        can compare against a prior teacher suggestion. alpha=0.1 for slow EMA.

        v2 (2026-04-24): semantics changed — caller is now expected to pass
        adopted=True iff Titan used at least one of the PRIORLY-SUGGESTED
        MISSING primitives (non-empty intersection of suggested ∩ actual).
        Caller must SKIP the update entirely when the prior suggestion list
        was empty (no signal to measure). This method does not know the
        semantics — it just applies the EMA. The worker's adoption-check
        block is the authoritative v2 semantics site.
        """
        cur = self.adoption_ema_by_domain.get(domain, 0.5)
        new = 0.9 * cur + 0.1 * (1.0 if adopted else 0.0)
        self.adoption_ema_by_domain[domain] = new
        return new

    def telemetry(self) -> dict:
        """Structured telemetry for /v4/meta-teacher/status."""
        now = time.time()
        self._prune_old(now)
        recent = list(self.recent_critiques)
        avg_q = (
            sum(r.get("quality_score", 0.5) for r in recent) / max(1, len(recent)))
        cat_counts: dict[str, int] = {}
        for r in recent:
            for c in r.get("critique_categories", []):
                cat_counts[c] = cat_counts.get(c, 0) + 1
        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
        return {
            "enabled": self._enabled,
            "prompt_version": SYSTEM_PROMPT_VERSION,
            "sample_mode": self._sample_mode,
            "critiques_lifetime": self.total_critiques,
            "observed_lifetime": self.total_observed,
            "critiques_1h": len(self._critique_times),
            "max_per_hour": self._max_per_hour,
            "avg_quality_recent": round(avg_q, 3),
            "top_critique_categories": top_cats,
            "adoption_rate_by_domain": dict(self.adoption_ema_by_domain),
            "adoption_rate_overall": (
                sum(self.adoption_ema_by_domain.values()) /
                max(1, len(self.adoption_ema_by_domain))),
            "current_reward_weight": round(self.compute_reward_weight(), 4),
            "reward_weight_cap": self._reward_weight_cap,
            "reward_weight_schedule": self._schedule_phase_name(),
            "task_key": self._task_key,
        }

    def _schedule_phase_name(self) -> str:
        if self.total_critiques < self._ramp_p1:
            return "phase_0_flat"
        if self.total_critiques < self._ramp_p2:
            return "phase_1_ramp"
        return "phase_2_stable"

    def record_critique_entry(self, entry: dict) -> None:
        """Append a full critique entry (dict) to the in-memory ring buffer.

        Worker also persists to critiques.jsonl. Ring buffer serves /v4 API.
        """
        self.recent_critiques.append(entry)


def build_system_prompt() -> str:
    """Return the current teacher system prompt. Indirected so worker can
    swap in per-session prompt variants for A/B testing without touching
    the teacher's core (rFP §12 Q1 future work)."""
    return SYSTEM_PROMPT
