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
    WorldMirrorArchetype,
    OuterRuminationArchetype,
    OuterInnerBridgeArchetype,
    GroundedTodayArchetype,
    PracticedResponseArchetype,
    ReflectionArchetype,
    ComposedThoughtArchetype,
    SelfWatchingArchetype,
)
from .pool_scoring import record_pending_post, reap_pending

logger = logging.getLogger(__name__)


# Priority-order: PROOF_DAY first (must-post slot bypasses everything).
# Outer-world archetypes get a chance before inner-only ones so a fresh
# external signal isn't shadowed by a recent inner crystallization.
PRIORITY_ORDER = (
    "proof_day",
    "world_mirror",
    "outer_inner_bridge",
    "outer_rumination",
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
                 knowledge_db: str = "./data/knowledge.db",
                 experience_db: str = "./data/experience.db",
                 meta_wisdom_db: str = "./data/meta_wisdom.db",
                 kuzu_graph=None):
        self.gateway = gateway
        self.social_x_db = social_x_db_path
        self.events_teacher_db = events_teacher_db

        common = {"gateway": gateway, "social_x_db_path": social_x_db_path}

        self.archetypes: dict[str, Any] = {
            "proof_day": ProofDayArchetype(**common),
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
        }

    # ── Probe ──────────────────────────────────────────────────────

    def probe(self, context) -> ArchetypeCandidate | None:
        """Return the first archetype candidate that fires, or None.

        Reaps any pending pool-scoring observations whose 12 h window has
        elapsed before probing — keeps the adaptive scoring loop closed
        without requiring a separate cron.
        """
        try:
            reap_pending(self.social_x_db, self.events_teacher_db)
        except Exception as e:
            logger.debug("[archetype_dispatcher] reap_pending non-fatal: %s", e)

        for name in PRIORITY_ORDER:
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
