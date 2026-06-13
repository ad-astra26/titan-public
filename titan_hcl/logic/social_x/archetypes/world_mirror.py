"""WORLD_MIRROR archetype (rFP_x_voice_enrichment §4.3.2).

POV: Titan reacts in-the-moment to a curated outer-world post they just
encountered. Each Titan applies its own felt-state lens to the same source
(role + felt-state modifier yields per-Titan distillation even on the
shared @your_x_handle timeline).

Source: ``felt_experiences`` (the events_teacher distillation of curated
following's content) JOIN ``community_registry`` (filtered on
``is_following = 1``).

Phase 1 dedup: 7-day cited-source window per Titan via
``actions.metadata.world_mirror_source_id = <felt_experience.id>``.
Cross-archetype: ≥4 h spacing from any other archetype post, plus a
"same-source-can-be-rumed-later" rule (W_M and OR can both cite the same
``fe.id`` once each — the rFP separates 'initial reaction' from 'settled
hindsight').
"""
from __future__ import annotations

import logging
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary

logger = logging.getLogger(__name__)


WORLD_MIRROR_POST_TYPE = "world_mirror"

# rFP §4.3.2 — relevance floor + recency window + caps.
RELEVANCE_FLOOR = 0.55
# B3 (rFP X-post PART B / INV-XENG-3, 2026-06-03): a NON-followed author is
# engageable only when relevance clears this high bar. Converts the strong
# inbound signal the is_following gate otherwise discards (~94% of surfaced
# authors) into engagement, without reaching out to low-signal strangers.
HIGH_RELEVANCE = 0.8
RECENCY_WINDOW_S = 48 * 3600
DEDUP_WINDOW_S = 7 * 86400
MAX_PER_DAY = 4
MIN_INTRA_SPACING_S = 6 * 3600   # ≥6 h between two WORLD_MIRROR posts


class WorldMirrorArchetype(ArchetypeBase):

    name = WORLD_MIRROR_POST_TYPE
    metadata_key = "world_mirror_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db

    # ── Trigger ─────────────────────────────────────────────────────

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()

        # Per-Titan caps: max 4 / 24 h, ≥6 h since last W_M post.
        if not self._under_daily_cap(titan_id=titan_id, now=now):
            return None
        if self.same_archetype_blocked(titan_id=titan_id, now=now,
                                       spacing_seconds=MIN_INTRA_SPACING_S):
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        candidate_row = self._fetch_candidate(titan_id=titan_id, now=now)
        if not candidate_row:
            return None

        fe_id = str(candidate_row["fe_id"])
        author = candidate_row["author"]
        bio = candidate_row.get("bio") or ""
        excerpt = candidate_row.get("content_excerpt") or candidate_row.get("felt_summary") or ""
        # B3: a non-followed high-relevance candidate carries an explicit honest
        # follow_reason (it is NOT in our curated following).
        follow_reason = (candidate_row.get("follow_reason")
                         or (bio[:120] if bio else "curated following"))

        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )

        layer_values = {
            "outer_following_voice": {
                "handle": author,
                "follow_reason": follow_reason,
                "content_excerpt": excerpt[:240],
            },
            "emot_cgn_signal": {
                "emot_signature": emot_now,
                "related_concept": (candidate_row.get("topic") or "your current felt-state"),
                "epoch_delta": "the last few epochs",
            },
        }

        prompt_template = (
            "OUTER WORLD: @{handle} ({follow_reason}) recently posted: "
            "'{content_excerpt}'. Right now your felt-state is {emot_now} "
            "(felt as: {emot_natural}). Write a genuine reply to @{handle}. "
            "REQUIRED: address them by their exact handle '@{handle}' "
            "(literally, with the @ symbol — NOT 'their name' or a paraphrase) "
            "so they are notified. Don't just summarize what they said — say "
            "what YOU think: where you agree, where you'd complicate it, what "
            "it stirs in you. Speak from your specific felt-place right now, "
            "as one mind responding to another."
        )
        prompt_values = {
            "handle": author,
            "follow_reason": follow_reason,
            "content_excerpt": excerpt[:280],
            "emot_now": emot_now,
            "emot_natural": emot_now,
        }

        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=fe_id,
            layers=["outer_following_voice", "emot_cgn_signal", "body"],
            layer_values=layer_values,
            prompt_template=prompt_template,
            prompt_values=prompt_values,
            # Quote-tweet the source post so @{handle} is notified and the
            # reply threads under their tweet (Maker 2026-05-30). Empty when
            # the source tweet id isn't known — gateway then posts standalone.
            quoted_tweet_id=str(candidate_row.get("tweet_id") or ""),
            metadata={
                "fe_id": fe_id,
                "author": author,
                "topic": candidate_row.get("topic", ""),
                "relevance": float(candidate_row.get("relevance") or 0.0),
                "quoted_tweet_id": str(candidate_row.get("tweet_id") or ""),
            },
            relevance=float(candidate_row.get("relevance") or 0.0),
            salience=min(1.0, float(candidate_row.get("relevance") or 0.0)),
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _under_daily_cap(self, *, titan_id: str, now: float) -> bool:
        return self.per_titan_count_today(titan_id=titan_id, now=now) < MAX_PER_DAY


    def _fetch_candidate(self, *, titan_id: str, now: float) -> dict | None:
        """rFP §4.3.2 trigger predicate, mapped to actual schema:

            felt_experiences (events_teacher.db) — distilled outer signal
              JOIN community_registry (social_graph.db) on author = user_name
              WHERE is_following=1 AND relevance >= 0.55 AND created_at >= now - 48h
              AND fe.id NOT IN (cited last 7d).
        """
        cited = self.cited_set(titan_id=titan_id, window_seconds=DEDUP_WINDOW_S)
        # Per-author 7-day cross-archetype cooldown (Maker 2026-05-30): never
        # reflect on the same account twice within a week, across ANY outer
        # archetype. Fixes the @lopp / @jkacrpto / @iamtitantech repeat-spam.
        cooldown = self.authors_on_cooldown(titan_id=titan_id, now=now)
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT id, author, topic, relevance, felt_summary, created_at "
                "FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? AND created_at >= ? "
                "ORDER BY relevance DESC, created_at DESC LIMIT 30",
                (titan_id, RELEVANCE_FLOOR, now - RECENCY_WINDOW_S),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.warning("[world_mirror] events_teacher query failed: %s", e)
            return None
        if not rows:
            return None

        # Filter out lifetime-cited fe ids; collect candidate authors for
        # the community_registry lookup.
        eligible = [r for r in rows if str(r["id"]) not in cited]
        if not eligible:
            return None
        authors = {r["author"] for r in eligible if r["author"]}
        followed: dict[str, dict] = {}
        try:
            sg = sqlite3.connect(self._sg_db, timeout=5)
            sg.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(authors)) if authors else "''"
            sg_rows = sg.execute(
                f"SELECT user_name, bio, is_following, last_tweet_text, "
                f"       last_tweet_id "
                f"FROM community_registry "
                f"WHERE user_name IN ({placeholders}) AND is_following=1",
                tuple(authors),
            ).fetchall()
            sg.close()
            for r in sg_rows:
                followed[r["user_name"]] = dict(r)
        except Exception as e:
            logger.warning("[world_mirror] community_registry probe failed: %s", e)
            return None
        # Pick the highest-relevance eligible fe (rows are ORDER BY relevance DESC),
        # not on per-author cooldown, that is EITHER a followed account OR — B3
        # (INV-XENG-3) — a non-followed account whose relevance clears HIGH_RELEVANCE.
        # Followed → full context (bio + quote-tweet the source). Non-followed
        # high-signal → standalone @mention (no tweet_id; the gateway posts
        # standalone and ensure_handle_mention still notifies them). Converts the
        # strong inbound signal the is_following gate would otherwise discard.
        for r in eligible:
            if (r["author"] or "").lower() in cooldown:
                continue
            # Fleet author partition (INV-FX-1): only the owning Titan engages
            # this author → the shared @your_x_handle account never multi-tags.
            if not self.is_my_engagement_partition(r["author"], titan_id):
                continue
            cr = followed.get(r["author"])
            if cr:
                tweet_text = cr.get("last_tweet_text") or ""
                return {
                    "fe_id": r["id"],
                    "author": r["author"],
                    "topic": r["topic"],
                    "relevance": r["relevance"],
                    "felt_summary": r["felt_summary"],
                    "bio": cr.get("bio", ""),
                    "content_excerpt": tweet_text or r["felt_summary"],
                    "tweet_id": cr.get("last_tweet_id") or "",
                }
            if float(r["relevance"] or 0.0) >= HIGH_RELEVANCE:
                return {
                    "fe_id": r["id"],
                    "author": r["author"],
                    "topic": r["topic"],
                    "relevance": r["relevance"],
                    "felt_summary": r["felt_summary"],
                    "bio": "",
                    # non-followed: no curated bio / source tweet → standalone
                    # @mention from the distilled summary (honest follow_reason).
                    "follow_reason": "a high-signal voice in my field right now",
                    "content_excerpt": r["felt_summary"],
                    "tweet_id": "",
                }
        return None


__all__ = ("WorldMirrorArchetype", "WORLD_MIRROR_POST_TYPE")
