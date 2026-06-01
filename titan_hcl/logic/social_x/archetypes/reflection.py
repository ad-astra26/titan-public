"""REFLECTION archetype (rFP_x_voice_enrichment §4.3.7).

POV: Looking at past output. **Phase 1 own-only**; Phase 2c adds cross-Titan.

Three sub-pools (multi-criteria selection):
  A. Recent Delta    — posted 24-72h ago, highest felt-state delta
  B. Mid Resonance   — posted 3-7d ago, highest felt-state similarity
  C. Long-tail Hindsight — posted 7-14d ago, highest engagement × age

Multi-criteria score per candidate:
  score = 0.35*engagement + 0.30*felt_match + 0.20*reflectability + 0.15*recency

Quote mechanism: X native quote-tweet via twitterapi.io's
``create_tweet_v2`` ``quoted_tweet_id`` parameter. Lifetime dedup per
``tweet_id``.

Post-type whitelist: ``REFLECTABLE_POST_TYPES`` from
``social_x_gateway.SocialXGateway`` (PROOF_DAY excluded).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary

logger = logging.getLogger(__name__)


REFLECTION_POST_TYPE = "reflection"

POOL_A = "A_recent_delta"
POOL_B = "B_mid_resonance"
POOL_C = "C_long_hindsight"

POOL_A_MIN_S = 24 * 3600
POOL_A_MAX_S = 72 * 3600
POOL_B_MIN_S = 3 * 86400
POOL_B_MAX_S = 7 * 86400
POOL_C_MIN_S = 7 * 86400
POOL_C_MAX_S = 14 * 86400

# Multi-criteria weights
W_ENGAGEMENT = 0.35
W_FELT_MATCH = 0.30
W_REFLECTABILITY = 0.20
W_RECENCY = 0.15


class ReflectionArchetype(ArchetypeBase):

    name = REFLECTION_POST_TYPE
    metadata_key = "reflection_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 2:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        whitelist = SocialXGateway.REFLECTABLE_POST_TYPES
        excluded = SocialXGateway.REFLECTION_EXCLUDED_POST_TYPES

        # F-3 (2026-05-17): 30-day window per tweet_id (was lifetime).
        # Reflection re-cites own posts after 30d once felt-state has
        # likely shifted enough to produce a different reflection.
        cited = self.cited_set(titan_id=titan_id, window_seconds=30 * 86400)
        cur_nm = getattr(context, "neuromods", {}) or {}
        cur_vec = _vec(cur_nm)

        engagement_index = self._engagement_index(titan_id=titan_id)
        max_engagement = max(engagement_index.values(), default=1) or 1

        candidates_per_pool: dict[str, dict] = {}
        for pool, lo, hi in (
            (POOL_A, POOL_A_MIN_S, POOL_A_MAX_S),
            (POOL_B, POOL_B_MIN_S, POOL_B_MAX_S),
            (POOL_C, POOL_C_MIN_S, POOL_C_MAX_S),
        ):
            best = self._best_in_pool(
                titan_id=titan_id, now=now, lo=lo, hi=hi,
                whitelist=whitelist, excluded=excluded, cited=cited,
                cur_vec=cur_vec, engagement_index=engagement_index,
                max_engagement=max_engagement, pool=pool,
            )
            if best:
                candidates_per_pool[pool] = best
        if not candidates_per_pool:
            return None
        # Pick the pool with the highest multi-criteria score, with
        # archetype_pool_scores as a secondary tie-breaker.
        scored = sorted(
            candidates_per_pool.items(),
            key=lambda kv: (kv[1]["score"], kv[1]["engagement_norm"]),
            reverse=True,
        )
        chosen_pool, cand = scored[0]
        return self._build_candidate(chosen_pool, cand, context, now)

    # ── Pool selection ──────────────────────────────────────────────

    def _engagement_index(self, *, titan_id: str) -> dict[str, int]:
        """tweet_id → engagement (likes + 2*replies + 3*quotes) from
        events_teacher.engagement_snapshots most-recent snapshot per tweet."""
        try:
            et = sqlite3.connect(f"file:{self._et_db}?mode=ro", uri=True, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT tweet_id, likes, replies, quotes FROM engagement_snapshots "
                "WHERE titan_id=? ORDER BY checked_at DESC",
                (titan_id,),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.debug("[reflection] engagement index unavailable: %s", e)
            return {}
        out: dict[str, int] = {}
        for r in rows:
            tid = str(r["tweet_id"])
            if tid in out:
                continue
            out[tid] = (int(r["likes"] or 0) + 2 * int(r["replies"] or 0)
                         + 3 * int(r["quotes"] or 0))
        return out

    def _best_in_pool(self, *, titan_id: str, now: float, lo: float, hi: float,
                       whitelist, excluded, cited: set[str],
                       cur_vec: list[float],
                       engagement_index: dict[str, int],
                       max_engagement: int, pool: str) -> dict | None:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT tweet_id, post_type, text, neuromods, emotion, "
                "       posted_at, created_at "
                "FROM actions WHERE titan_id=? "
                "  AND status IN ('posted','verified') "
                "  AND tweet_id IS NOT NULL "
                "  AND COALESCE(posted_at, created_at) <= ? "
                "  AND COALESCE(posted_at, created_at) >= ? "
                "ORDER BY COALESCE(posted_at, created_at) DESC LIMIT 200",
                (titan_id, now - lo, now - hi),
            ).fetchall()
        finally:
            conn.close()
        best = None
        for r in rows:
            tid = str(r["tweet_id"])
            if not tid or tid in cited:
                continue
            ptype = (r["post_type"] or "").lower()
            if ptype in excluded:
                continue
            reflectability = 1.0 if (ptype in whitelist) else 0.0
            if reflectability == 0.0:
                continue
            try:
                past_nm = json.loads(r["neuromods"] or "{}")
            except Exception:
                past_nm = {}
            past_vec = _vec(past_nm)
            sim = _cosine(cur_vec, past_vec)
            delta = 1.0 - sim
            felt_match = delta if pool == POOL_A else (sim if pool == POOL_B else 0.5)
            engagement_norm = engagement_index.get(tid, 0) / max_engagement
            age_s = now - float(r["posted_at"] or r["created_at"])
            recency_w = (
                1.0 - (age_s - lo) / max(1.0, hi - lo)
                if pool != POOL_C else (age_s / hi)
            )
            recency_w = max(0.0, min(1.0, recency_w))
            score = (
                W_ENGAGEMENT * engagement_norm
                + W_FELT_MATCH * felt_match
                + W_REFLECTABILITY * reflectability
                + W_RECENCY * recency_w
            )
            cand = {
                "tweet_id": tid,
                "post_type": ptype,
                "text": r["text"] or "",
                "neuromods": past_nm,
                "emotion": r["emotion"] or "",
                "posted_at": float(r["posted_at"] or r["created_at"]),
                "engagement": engagement_index.get(tid, 0),
                "engagement_norm": engagement_norm,
                "felt_match": felt_match,
                "delta_or_similarity": delta if pool == POOL_A else sim,
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
        return best

    # ── Build ───────────────────────────────────────────────────────

    def _build_candidate(self, pool: str, cand: dict, context,
                          now: float) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        emot_then = compact_felt_summary(cand["neuromods"], cand["emotion"] or "")
        days_ago = max(1, int((now - cand["posted_at"]) / 3600))
        delta_or_sim_label = "delta" if pool == POOL_A else "similarity"

        body = (
            f"REFLECTION (own post, pool {pool}): {days_ago}h ago you "
            f"posted this:\n'{cand['text']}'\n"
            f"At posting moment your felt-state was {cand['emotion']} "
            f"({emot_then}). Right now you're feeling {emot_now}. "
            f"The felt-{delta_or_sim_label} between then and now is "
            f"{cand['delta_or_similarity']:.2f} — that's why you're "
            f"returning to this. Speak from CHANGED-MIND register. "
            f"Quote-tweet shape: short, sharp, hindsight."
        )
        layer_values = {
            "own_post_quote": {
                "days_ago": days_ago,
                "posted_emotion": cand["emotion"] or "neutral",
                "post_text": cand["text"][:240],
                "posted_neuromods_summary": emot_then,
                "emot_now_natural": emot_now,
                "delta": cand["delta_or_similarity"],
            },
            "emot_cgn_signal": {
                "emot_signature": emot_now,
                "related_concept": cand["post_type"],
                "epoch_delta": f"{days_ago} hours",
            },
        }
        return ArchetypeCandidate(
            archetype=self.name,
            pool=pool,
            source_id=cand["tweet_id"],
            layers=["own_post_quote", "meta_insight", "emot_cgn_signal", "body"],
            layer_values=layer_values,
            prompt_template=body,
            prompt_values={},
            metadata={
                "pool": pool,
                "tweet_id": cand["tweet_id"],
                "selection_score": cand["score"],
                "engagement_norm": cand["engagement_norm"],
                "felt_match": cand["felt_match"],
            },
            quoted_tweet_id=cand["tweet_id"],
            relevance=cand["score"],
            salience=cand["engagement_norm"],
        )


def _vec(nm: dict) -> list[float]:
    keys = ("DA", "5HT", "NE", "ACh", "GABA", "Endorphin", "Glutamate")
    out = []
    for k in keys:
        try:
            out.append(float(nm.get(k, 0.5)))
        except (TypeError, ValueError):
            out.append(0.5)
    return out


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


__all__ = ("ReflectionArchetype", "REFLECTION_POST_TYPE")
