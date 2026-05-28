"""SkillFailureTracker — repair-fork-on-failure orchestration (Synthesis Engine Phase 9).

arch §9.3 / rFP §11.5: "after N consecutive failures, auto-flag for LLM re-review
(or spawn a repair fork)." This wires the failure counter to the P5 hypothesis-fork
machinery — no new fork class, no new invariant (it orchestrates the already-RATIFIED
INV-3 / INV-9 / INV-10).

When a delegated compiled skill (P8) fails `failure_threshold` times in a row, Titan
spawns a **repair fork** rooted at the skill's parent concept (resolved via the skill's
`compiled_from` lineage → spine concept). The fork explores the failure without polluting
the canonical chain; on oracle-verified resolution it graduates to a parent version bump
(INV-10), on abandonment it leaves a tombstone scar (INV-3). The worked example (arch §9.4)
is the Metaplex undocumented-bug loop made autonomous + verifiable.

Sole writer is synthesis_worker (the fork is created via HypothesisForkStore — INV-Syn-8).
Guard rail (§9.4): spawning is computational only; a fork whose exploration would execute
a consequential action inherits the existing metabolic + approval gates. P9 spawns + records;
it does NOT auto-execute.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["SkillFailureTracker"]


class SkillFailureTracker:
    """Counts consecutive failures per skill_id; spawns a repair fork at threshold.

    `concept_resolver(skill_id) -> Optional[tuple[root_anchor_tx, parent_concept_id]]`
    resolves the skill's parent concept for a repair fork. Returns None when the skill
    has no spine concept (→ net-new exploration fork, both anchors None).
    """

    def __init__(
        self,
        *,
        fork_store,
        concept_resolver: Callable[[str], Optional[tuple]],
        bus_emit: Optional[Callable[[str, dict], None]] = None,
        failure_threshold: int = 3,
        clock: Callable[[], float] = None,
    ) -> None:
        self._fork_store = fork_store
        self._concept_resolver = concept_resolver
        self._bus_emit = bus_emit
        self._failure_threshold = max(1, int(failure_threshold))
        import time as _time
        self._clock = clock or _time.time
        # skill_id -> consecutive failure count.
        self._consecutive: dict[str, int] = {}
        # skill_id -> live repair fork_id (idempotency guard: no double-spawn
        # while a repair fork for this skill is still open).
        self._live_repair_fork: dict[str, str] = {}

    # ── public API ────────────────────────────────────────────────────

    def record_outcome(self, skill_id: str, *, success: bool) -> Optional[str]:
        """Record one delegated-skill invocation outcome.

        success → reset the consecutive counter, return None.
        failure → increment; at threshold (and no live repair fork) spawn a
                  repair fork, emit SKILL_REPAIR_FORK_SPAWNED, reset counter,
                  return the new fork_id. Otherwise return None.
        Soft, total: never raises (a fork-spawn failure logs + returns None).
        """
        if not skill_id:
            return None

        if success:
            self._consecutive.pop(skill_id, None)
            return None

        count = self._consecutive.get(skill_id, 0) + 1
        self._consecutive[skill_id] = count

        if count < self._failure_threshold:
            return None

        # Idempotency: a live repair fork for this skill is already exploring.
        if skill_id in self._live_repair_fork:
            logger.debug(
                "[SkillFailureTracker] skill=%s hit threshold but a repair "
                "fork (%s) is already live — not double-spawning",
                skill_id, self._live_repair_fork[skill_id],
            )
            return None

        fork_id = self._spawn_repair_fork(skill_id)
        # Reset the counter regardless of spawn outcome — a failed spawn
        # shouldn't wedge the counter at threshold forever.
        self._consecutive[skill_id] = 0
        return fork_id

    def resolve_repair_fork(self, skill_id: str) -> None:
        """Clear the live-repair-fork guard for a skill once its repair fork
        graduated or was abandoned (called by the fork lifecycle). Idempotent."""
        self._live_repair_fork.pop(skill_id, None)

    def consecutive_failures(self, skill_id: str) -> int:
        """Current consecutive-failure count for a skill (telemetry/tests)."""
        return self._consecutive.get(skill_id, 0)

    # ── internals ─────────────────────────────────────────────────────

    def _spawn_repair_fork(self, skill_id: str) -> Optional[str]:
        intent = f"repair_skill:{skill_id}"
        root_anchor: Optional[str] = None
        parent_concept_id: Optional[str] = None
        try:
            resolved = self._concept_resolver(skill_id)
            if resolved is not None:
                root_anchor, parent_concept_id = resolved
        except Exception as exc:
            logger.warning(
                "[SkillFailureTracker] concept_resolver(%s) raised: %s — "
                "spawning net-new exploration fork", skill_id, exc,
            )
            root_anchor, parent_concept_id = None, None

        # Both-or-neither contract of create_fork (repair vs net-new).
        if (root_anchor is None) != (parent_concept_id is None):
            logger.warning(
                "[SkillFailureTracker] resolver returned half a repair anchor "
                "for skill=%s (root=%r concept=%r) — falling back to net-new",
                skill_id, root_anchor, parent_concept_id,
            )
            root_anchor, parent_concept_id = None, None

        try:
            fork_id = self._fork_store.create_fork(
                intent=intent,
                root_anchor=root_anchor,
                parent_concept_id=parent_concept_id,
            )
        except Exception as exc:
            logger.error(
                "[SkillFailureTracker] create_fork failed for skill=%s: %s",
                skill_id, exc, exc_info=True,
            )
            return None

        is_repair = parent_concept_id is not None
        if is_repair:
            self._live_repair_fork[skill_id] = fork_id

        logger.info(
            "[SkillFailureTracker] skill=%s reached %d consecutive failures "
            "→ spawned %s fork %s (parent_concept=%s)",
            skill_id, self._failure_threshold,
            "repair" if is_repair else "net-new", fork_id, parent_concept_id,
        )

        if self._bus_emit is not None:
            try:
                self._bus_emit("SKILL_REPAIR_FORK_SPAWNED", {
                    "skill_id": skill_id,
                    "fork_id": fork_id,
                    "parent_concept_id": parent_concept_id,
                    "root_anchor": root_anchor,
                    "kind": "repair" if is_repair else "net_new",
                    "consecutive_failures": self._failure_threshold,
                    "ts": self._clock(),
                })
            except Exception as exc:
                logger.debug(
                    "[SkillFailureTracker] SKILL_REPAIR_FORK_SPAWNED emit "
                    "failed: %s", exc,
                )
        return fork_id
