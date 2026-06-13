"""AMPLIFY archetype (Maker 2026-05-30) — native retweet of a followed
account's high-relevance post.

POV: "this is genuinely worth boosting." Unlike WORLD_MIRROR (compose a
reply) or OUTER_RUMINATION (settled hindsight), AMPLIFY performs a *native
retweet* — no composed text. It adds activity + diversity to Titan's X
presence while amplifying the curated following.

Emergence, not forcing: amplification fires only when the events_teacher
distiller already scored a followed account's fresh post as HIGHLY relevant
to THIS Titan's interests (relevance ≥ AMPLIFY_RELEVANCE_FLOOR). The
relevance is computed per-Titan from felt-state + grounded vocabulary, so
different Titans amplify different posts from the same timeline.

Guards:
  * Source: felt_experiences (events_teacher) JOIN community_registry on
    is_following=1, with a known last_tweet_id to retweet.
  * relevance ≥ 0.65 (retweet = strong endorsement — higher bar than reply).
  * Per-author 7-day cross-archetype cooldown (shared ArchetypeBase).
  * Lifetime-once per source post (amplify_source_id) within 30 d.
  * Per-Titan cap (≤ MAX_PER_DAY) + intra-spacing + cross-archetype spacing.
"""
from __future__ import annotations

import logging
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate

logger = logging.getLogger(__name__)


AMPLIFY_POST_TYPE = "amplify"

AMPLIFY_RELEVANCE_FLOOR = 0.65      # retweet = endorsement → high bar
AMPLIFY_RECENCY_WINDOW_S = 48 * 3600
AMPLIFY_DEDUP_WINDOW_S = 30 * 86400
AMPLIFY_MAX_PER_DAY = 2
AMPLIFY_MIN_INTRA_SPACING_S = 6 * 3600


class AmplifyArchetype(ArchetypeBase):

    name = AMPLIFY_POST_TYPE
    metadata_key = "amplify_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()

        if self.per_titan_count_today(titan_id=titan_id, now=now) >= AMPLIFY_MAX_PER_DAY:
            return None
        if self.same_archetype_blocked(titan_id=titan_id, now=now,
                                       spacing_seconds=AMPLIFY_MIN_INTRA_SPACING_S):
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        row = self._fetch_target(titan_id=titan_id, now=now)
        if not row:
            return None

        author = row["author"]
        # Fleet author partition (INV-FX-1): only the owning Titan amplifies
        # this author → the shared @your_x_handle account never multi-engages.
        if not self.is_my_engagement_partition(author, titan_id):
            return None
        tweet_id = row["tweet_id"]
        fe_id = str(row["fe_id"])

        # AMPLIFY produces a retweet ACTION — the gateway's post() short-
        # circuits on archetype=='amplify' and calls retweet(target). No
        # prompt/text is composed.
        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=fe_id,
            layers=[],
            layer_values={},
            prompt_template="",
            prompt_values={},
            metadata={
                "amplify_source_id": fe_id,
                "author": author,
                "retweet_target_id": str(tweet_id),
                "topic": row.get("topic", ""),
                "relevance": float(row.get("relevance") or 0.0),
            },
            relevance=float(row.get("relevance") or 0.0),
            salience=min(1.0, float(row.get("relevance") or 0.0)),
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _fetch_target(self, *, titan_id: str, now: float) -> dict | None:
        """Highest-relevance fresh followed-account post that (a) clears the
        endorsement floor, (b) has a retweetable last_tweet_id, (c) is not on
        per-author cooldown, and (d) hasn't been amplified before."""
        cited = self.cited_set(titan_id=titan_id,
                               window_seconds=AMPLIFY_DEDUP_WINDOW_S)
        cooldown = self.authors_on_cooldown(titan_id=titan_id, now=now)
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT id, author, topic, relevance, created_at "
                "FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? AND created_at >= ? "
                "ORDER BY relevance DESC, created_at DESC LIMIT 40",
                (titan_id, AMPLIFY_RELEVANCE_FLOOR,
                 now - AMPLIFY_RECENCY_WINDOW_S),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.warning("[amplify] felt_experiences query failed: %s", e)
            return None
        if not rows:
            return None

        eligible = [r for r in rows
                    if str(r["id"]) not in cited
                    and (r["author"] or "").lower() not in cooldown
                    and r["author"]]
        if not eligible:
            return None

        authors = {r["author"] for r in eligible}
        followed: dict[str, dict] = {}
        try:
            sg = sqlite3.connect(self._sg_db, timeout=5)
            sg.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(authors)) if authors else "''"
            for cr in sg.execute(
                f"SELECT user_name, last_tweet_id FROM community_registry "
                f"WHERE user_name IN ({placeholders}) AND is_following=1 "
                f"  AND COALESCE(last_tweet_id, '') != ''",
                tuple(authors),
            ).fetchall():
                followed[cr["user_name"]] = dict(cr)
            sg.close()
        except Exception as e:
            logger.warning("[amplify] community_registry probe failed: %s", e)
            return None
        if not followed:
            return None

        for r in eligible:
            cr = followed.get(r["author"])
            if not cr:
                continue
            return {
                "fe_id": r["id"],
                "author": r["author"],
                "topic": r["topic"],
                "relevance": r["relevance"],
                "tweet_id": cr.get("last_tweet_id") or "",
            }
        return None


__all__ = ("AmplifyArchetype", "AMPLIFY_POST_TYPE")
