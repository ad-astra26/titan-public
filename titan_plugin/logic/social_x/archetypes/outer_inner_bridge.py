"""OUTER_INNER_BRIDGE archetype (rFP_x_voice_enrichment §4.3.4) — keystone
synthesis archetype.

POV: ``This outside thing lands against my inside grounding — a SHAPE
emerges when the two meet.`` Pre-bridge proof-of-concept that the broader
``rFP_inner_outer_bridge`` rFP unlocks at full power later.

Trigger (both must hold):
  1. Fresh outer signal: ``felt_experiences`` row with ``relevance ≥ 0.5``
     within the last 72 h, from an ``is_following=1`` author, not previously
     bridged.
  2. Recent inner grounding: ``vocabulary`` row with
     ``learning_phase='producible'`` AND ``times_encountered ≥ 5`` AND
     ``last_encountered`` within 14 d (or a Kuzu MindEntity concept fallback).
  3. Match: the inner concept's name appears in the outer signal's
     ``concept_signals`` JSON list (Phase 1 symbolic-overlap; Phase 2b
     swaps in embedding similarity).

Idempotency: ``outer_inner_bridge_source_id = <fe.id>`` lifetime; same
inner concept CAN appear in multiple bridges if surfaced by different
outer posts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary

logger = logging.getLogger(__name__)


OINB_POST_TYPE = "outer_inner_bridge"

OUTER_RELEVANCE_FLOOR = 0.5
OUTER_WINDOW_S = 72 * 3600
INNER_TIMES_ENCOUNTERED_MIN = 5
INNER_WINDOW_S = 14 * 86400
CONCEPT_GT_DEDUP_S = 4 * 86400  # cross-archetype with GROUNDED_TODAY (§4.3.5)


class OuterInnerBridgeArchetype(ArchetypeBase):

    name = OINB_POST_TYPE
    metadata_key = "outer_inner_bridge_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db",
                 inner_memory_db: str = "./data/inner_memory.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db
        self._im_db = inner_memory_db

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 1:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        cited = self.cited_set(titan_id=titan_id)             # lifetime once-bridged
        gt_concepts = self._recent_gt_concepts(titan_id=titan_id, now=now)

        outer_candidates = self._fetch_fresh_outer(titan_id=titan_id, now=now,
                                                     cited=cited)
        if not outer_candidates:
            return None
        # Inner concept index: producible vocabulary recently encountered
        inner_index = self._inner_vocabulary_index(now=now)
        if not inner_index:
            return None

        # Symbolic-overlap match — first outer signal whose concept_signals
        # JSON contains a known inner concept (and that concept hasn't been
        # GROUNDED_TODAY'd in the last 4d).
        for outer in outer_candidates:
            try:
                concept_signals = json.loads(outer.get("concept_signals") or "[]")
            except Exception:
                concept_signals = []
            for raw_c in concept_signals:
                key = str(raw_c).lower().strip()
                if not key or key in gt_concepts:
                    continue
                inner = inner_index.get(key)
                if not inner:
                    continue
                return self._build_candidate(outer, inner, context, now)
        return None

    # ── Helpers ─────────────────────────────────────────────────────

    def _fetch_fresh_outer(self, *, titan_id: str, now: float,
                            cited: set[str]) -> list[dict]:
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT id, author, topic, relevance, felt_summary, "
                "       concept_signals, created_at "
                "FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? "
                "  AND created_at >= ? "
                "ORDER BY relevance DESC LIMIT 50",
                (titan_id, OUTER_RELEVANCE_FLOOR, now - OUTER_WINDOW_S),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.warning("[oinb] outer fetch failed: %s", e)
            return []
        if not rows:
            return []
        # Filter by is_following + not-cited.
        authors = {r["author"] for r in rows if r["author"]}
        followed: dict[str, dict] = {}
        try:
            sg = sqlite3.connect(self._sg_db, timeout=5)
            sg.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(authors)) if authors else "''"
            for r in sg.execute(
                f"SELECT user_name, bio, last_tweet_text FROM community_registry "
                f"WHERE user_name IN ({placeholders}) AND is_following=1",
                tuple(authors),
            ).fetchall():
                followed[r["user_name"]] = dict(r)
            sg.close()
        except Exception as e:
            logger.warning("[oinb] community_registry probe failed: %s", e)
            return []
        out: list[dict] = []
        for r in rows:
            if str(r["id"]) in cited:
                continue
            if r["author"] not in followed:
                continue
            d = dict(r)
            d["bio"] = followed[r["author"]].get("bio", "")
            d["content_excerpt"] = (followed[r["author"]].get("last_tweet_text")
                                     or r["felt_summary"])
            out.append(d)
        return out

    def _inner_vocabulary_index(self, *, now: float) -> dict[str, dict]:
        cutoff = now - INNER_WINDOW_S
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, word, times_encountered, grounded_at, "
                "       grounded_felt_summary, last_encountered "
                "FROM vocabulary "
                "WHERE learning_phase='producible' "
                "  AND times_encountered >= ? "
                "  AND COALESCE(last_encountered, 0) >= ?",
                (INNER_TIMES_ENCOUNTERED_MIN, cutoff),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("[oinb] inner vocab index failed: %s", e)
            return {}
        return {str(r["word"]).lower(): dict(r) for r in rows if r["word"]}

    def _recent_gt_concepts(self, *, titan_id: str, now: float) -> set[str]:
        """Concepts that were GROUNDED_TODAY'd in the last 4 days."""
        cutoff = now - CONCEPT_GT_DEDUP_S
        out: set[str] = set()
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT metadata FROM actions WHERE titan_id=? AND post_type=? "
                "AND created_at >= ?",
                (titan_id, "grounded_today", cutoff),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            try:
                m = json.loads(r["metadata"] or "{}")
            except Exception:
                continue
            ck = m.get("concept_key") or m.get("concept", "")
            if ck:
                out.add(str(ck).lower())
        return out

    def _build_candidate(self, outer: dict, inner: dict, context,
                          now: float) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        epochs_ago = max(1, int((now - float(inner.get("grounded_at") or now)) / 60))
        layer_values = {
            "outer_following_voice": {
                "handle": outer["author"],
                "follow_reason": (outer.get("bio") or "curated following")[:120],
                "content_excerpt": (outer.get("content_excerpt") or "")[:240],
            },
            "cgn_grounded_today": {
                "concept": inner["word"],
                "pool_name": "vocabulary",
                "meta": f"encountered {inner.get('times_encountered', 0)}× now",
                "grounded_felt_summary": inner.get("grounded_felt_summary") or emot_now,
            },
        }
        prompt_template = (
            "OUTER + INNER: @{handle} just posted: '{content_excerpt}'. "
            "This lands against your inner grounding of '{concept}' from "
            "{epochs_ago} epochs ago (encountered {times_encountered}× since). "
            "At grounding moment your felt-state was: '{grounding_felt}'. "
            "Right now your felt-state is: '{emot_now}'. What does the "
            "OUTSIDE see that you've been touching from the inside via "
            "{concept}? Speak from the synthesis — what's the SHAPE that "
            "emerges when these two meet INSIDE YOU? Not summary; emergence."
        )
        prompt_values = {
            "handle": outer["author"],
            "content_excerpt": (outer.get("content_excerpt") or "")[:240],
            "concept": inner["word"],
            "epochs_ago": epochs_ago,
            "times_encountered": inner.get("times_encountered", 0),
            "grounding_felt": inner.get("grounded_felt_summary") or emot_now,
            "emot_now": emot_now,
        }
        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=str(outer["id"]),
            layers=["outer_following_voice", "cgn_grounded_today",
                    "meta_insight", "body"],
            layer_values=layer_values,
            prompt_template=prompt_template,
            prompt_values=prompt_values,
            metadata={
                "outer_id": outer["id"],
                "author": outer["author"],
                "concept": inner["word"],
                "concept_key": str(inner["word"] or "").lower(),
                "match_method": "concept_signals_overlap",
            },
            relevance=float(outer.get("relevance") or 0.0),
            salience=min(1.0, float(outer.get("relevance") or 0.0)),
        )


__all__ = ("OuterInnerBridgeArchetype", "OINB_POST_TYPE")
