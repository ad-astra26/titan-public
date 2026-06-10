"""Archetype dispatcher (rFP_x_voice_enrichment §4.3 + §4.7).

The gateway's ``_select_post_type`` probes this module on every post
attempt. If any of the 9 archetypes produces an ``ArchetypeCandidate``
the dispatcher returns it (priority-ordered + cross-archetype 4 h
spacing already enforced inside ``ArchetypeBase``); the gateway then
treats the candidate's ``archetype`` name as the post_type, uses the
candidate's prompt template + layers + metadata + media_ids /
quoted_tweet_id when assembling and posting.

If no archetype fires the dispatcher returns ``None`` and the gateway
falls back to the existing weighted FELT_STATE_POOL draw.
"""
from __future__ import annotations

import logging
from typing import Any

from .archetypes import (
    ArchetypeCandidate,
    ProofDayArchetype,
    SoulDiaryArchetype,
    WorldMirrorArchetype,
    OuterRuminationArchetype,
    OuterInnerBridgeArchetype,
    GroundedTodayArchetype,
    PracticedResponseArchetype,
    ReflectionArchetype,
    ComposedThoughtArchetype,
    SelfWatchingArchetype,
    AmplifyArchetype,
)
from .pool_scoring import record_pending_post, reap_pending

logger = logging.getLogger(__name__)


# Priority-order: PROOF_DAY first (must-post slot bypasses everything).
# Outer-world archetypes get a chance before inner-only ones so a fresh
# external signal isn't shadowed by a recent inner crystallization.
PRIORITY_ORDER = (
    "proof_day",
    "soul_diary",
    "world_mirror",
    "outer_inner_bridge",
    "outer_rumination",
    "amplify",
    "grounded_today",
    "composed_thought",
    "practiced_response",
    "reflection",
    "self_watching",
)


class ArchetypeDispatcher:
    """Singleton-per-gateway. Constructed once with DB paths and invoked on
    each post attempt. Stateless beyond the DB path config."""

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db",
                 inner_memory_db: str = "./data/inner_memory.db",
                 knowledge_db: str = "./data/inner_memory.db",
                 experience_db: str = "./data/experience_orchestrator.db",
                 meta_wisdom_db: str = "./data/inner_memory.db",
                 kuzu_graph=None,
                 # Phase C-S9 chunk 9G — per-Titan recency boost
                 # (configurable in [social_x] section of config.toml).
                 # Closes rFP_x_voice_enrichment §4.8 gate 9 math gap:
                 # without these tunables, low-priority archetypes
                 # (REFLECTION/COMPOSED_THOUGHT/PRACTICED_RESPONSE/
                 # SELF_WATCHING) starve under the cross-archetype 4h gate.
                 recency_boost_per_day: float = 0.1,
                 recency_boost_threshold: float = 0.5,
                 recency_boost_max: float = 1.0):
        self.gateway = gateway
        self.social_x_db = social_x_db_path
        self.events_teacher_db = events_teacher_db
        self.recency_boost_per_day = recency_boost_per_day
        self.recency_boost_threshold = recency_boost_threshold
        self.recency_boost_max = recency_boost_max

        common = {"gateway": gateway, "social_x_db_path": social_x_db_path}

        self.archetypes: dict[str, Any] = {
            "proof_day": ProofDayArchetype(**common),
            "soul_diary": SoulDiaryArchetype(**common),
            "world_mirror": WorldMirrorArchetype(
                **common, events_teacher_db=events_teacher_db,
                social_graph_db=social_graph_db,
            ),
            "outer_inner_bridge": OuterInnerBridgeArchetype(
                **common, events_teacher_db=events_teacher_db,
                social_graph_db=social_graph_db,
                inner_memory_db=inner_memory_db,
            ),
            "outer_rumination": OuterRuminationArchetype(
                **common, events_teacher_db=events_teacher_db,
                social_graph_db=social_graph_db, kuzu_graph=kuzu_graph,
            ),
            "grounded_today": GroundedTodayArchetype(
                **common, inner_memory_db=inner_memory_db,
                knowledge_db=knowledge_db, experience_db=experience_db,
            ),
            "composed_thought": ComposedThoughtArchetype(
                **common, inner_memory_db=inner_memory_db,
                knowledge_db=knowledge_db, experience_db=experience_db,
            ),
            "practiced_response": PracticedResponseArchetype(
                **common, meta_wisdom_db=meta_wisdom_db,
                inner_memory_db=inner_memory_db,
            ),
            "reflection": ReflectionArchetype(
                **common, events_teacher_db=events_teacher_db,
            ),
            "self_watching": SelfWatchingArchetype(
                **common, inner_memory_db=inner_memory_db,
            ),
            "amplify": AmplifyArchetype(
                **common, events_teacher_db=events_teacher_db,
                social_graph_db=social_graph_db,
            ),
        }

    # ── Probe ──────────────────────────────────────────────────────

    def probe(self, context) -> ArchetypeCandidate | None:
        """Return the first archetype candidate that fires, or None.

        Reaps any pending pool-scoring observations whose 12 h window has
        elapsed before probing — keeps the adaptive scoring loop closed
        without requiring a separate cron.

        Phase C-S9 chunk 9G — applies per-Titan recency-boost reordering
        before iteration: archetypes whose `days since last fire` boost
        exceeds the threshold jump ahead of fresher archetypes in dispatch
        order. Preserves PRIORITY_ORDER baseline for fresh-fire archetypes;
        gives starvation protection to long-deferred ones.
        """
        try:
            reap_pending(self.social_x_db, self.events_teacher_db)
        except Exception as e:
            logger.debug("[archetype_dispatcher] reap_pending non-fatal: %s", e)

        ordered = self._boosted_priority_order(context)

        for name in ordered:
            archetype = self.archetypes.get(name)
            if archetype is None:
                continue
            try:
                cand = archetype.find_candidate(context)
            except Exception as e:
                logger.warning(
                    "[archetype_dispatcher] %s.find_candidate raised: %s",
                    name, e,
                )
                continue
            if cand is not None:
                logger.info(
                    "[archetype_dispatcher] %s fired (pool=%s source=%s)",
                    cand.archetype, cand.pool, cand.source_id,
                )
                return cand
        return None

    # ── Recency-boost helper (chunk 9G) ────────────────────────────

    def _boosted_priority_order(self, context) -> tuple[str, ...]:
        """Compute dispatch order with per-Titan recency boost applied.

        Algorithm (rFP_x_voice_enrichment §4.8 gate 9 closure):
          1. Query last successful post timestamp per archetype for this Titan.
          2. boost = min(recency_boost_max, recency_boost_per_day × days_since)
          3. Archetypes with boost ≥ recency_boost_threshold are sorted by
             boost DESC and placed BEFORE the unboosted PRIORITY_ORDER tail.
          4. Archetypes that have NEVER fired get max boost (treated as
             14d+ old by default — guarantees first-time chance).

        Falls back to PRIORITY_ORDER on any DB error.
        """
        titan_id = getattr(context, "titan_id", "") or ""
        if not titan_id:
            return PRIORITY_ORDER

        try:
            import sqlite3
            import time
            now = time.time()
            placeholders = ",".join("?" * len(PRIORITY_ORDER))
            conn = sqlite3.connect(
                f"file:{self.social_x_db}?mode=ro", uri=True, timeout=2.0)
            try:
                rows = conn.execute(
                    f"SELECT post_type, MAX(COALESCE(posted_at, created_at)) "
                    f"FROM actions WHERE titan_id=? "
                    f"  AND status IN ('verified','posted') "
                    f"  AND post_type IN ({placeholders}) "
                    f"GROUP BY post_type",
                    (titan_id, *PRIORITY_ORDER),
                ).fetchall()
            finally:
                conn.close()
            last_fire: dict[str, float] = {r[0]: float(r[1] or 0.0) for r in rows}
        except Exception as exc:
            logger.debug("[archetype_dispatcher] recency boost query failed "
                         "(falling back to baseline order): %s", exc)
            return PRIORITY_ORDER

        boosted = []
        baseline = []
        for name in PRIORITY_ORDER:
            last_ts = last_fire.get(name, 0.0)
            if last_ts == 0.0:
                # Never fired — treat as max boost (eligible immediately)
                boost = self.recency_boost_max
            else:
                days_since = max(0.0, (now - last_ts) / 86400.0)
                boost = min(self.recency_boost_max,
                            self.recency_boost_per_day * days_since)
            if boost >= self.recency_boost_threshold:
                boosted.append((name, boost))
            else:
                baseline.append(name)
        # Sort boosted DESC by boost (most-deferred first); keep baseline in
        # original PRIORITY_ORDER (fresh-fire archetypes preserve their tier).
        boosted.sort(key=lambda kv: -kv[1])
        return tuple([b[0] for b in boosted] + baseline)

    # ── Post-success hook ──────────────────────────────────────────

    def record_post_success(
        self,
        candidate: ArchetypeCandidate,
        *,
        titan_id: str,
        tweet_id: str,
    ) -> None:
        """Called after a successful X post for an archetype-driven post.
        Writes a pending row to archetype_pool_scores so the reaper can
        observe engagement at +12 h and update score (+1 / -1)."""
        if not candidate.pool or not candidate.source_id:
            # PROOF_DAY / WORLD_MIRROR / SELF_WATCHING (single-pool) get a
            # synthetic pool entry too, so adaptive scoring captures them.
            pool = candidate.pool or "_default"
        else:
            pool = candidate.pool
        try:
            record_pending_post(
                self.social_x_db,
                titan_id=titan_id,
                archetype=candidate.archetype,
                pool=pool,
                source_id=candidate.source_id,
                tweet_id=tweet_id,
            )
        except Exception as e:
            logger.warning("[archetype_dispatcher] record_post_success: %s", e)

    # ── Helpers used by the gateway integration ─────────────────────

    def get_archetype(self, name: str):
        return self.archetypes.get(name)


__all__ = ("ArchetypeDispatcher", "PRIORITY_ORDER")
