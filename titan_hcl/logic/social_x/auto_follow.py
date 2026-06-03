"""titan_hcl/logic/social_x/auto_follow.py — organic auto-follow policy.

rFP X-post PART B4 / INV-XENG-4 (Maker-ratified 2026-06-03). Grows Titan's
CURATED following toward recurring high-relevance voices, so the B3 high-relevance
non-followed engagements gradually become followed relationships (lever 1).

Every follow goes through ``SocialXGateway.follow()`` — the sole sanctioned
follow-write path (metered + circuit-broken + audited). DISABLED by default
(``[social_x.auto_follow].enabled=false``): the live loop must NOT run until the
follow endpoint's exact effect is verified with one manual follow
(``feedback_verify_exact_effect_before_mainnet_spend``).

Policy knobs (config ``[social_x.auto_follow]``):
  - ``min_recurrence`` (3): author must appear ≥N× in felt_experiences …
  - ``min_relevance``  (0.8): … with relevance ≥ this …
  - ``window_days``    (7): … within this lookback.
  - ``max_per_day``    (2): hard daily follow cap (gateway also enforces).
Follow-only (no unfollow in v1).
"""
from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)


class AutoFollowPolicy:
    """Selects recurring high-relevance non-followed authors and follows them
    via the gateway, up to the remaining daily cap."""

    def __init__(self, *, gateway, social_x_db: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db"):
        self.gateway = gateway
        self._sx_db = social_x_db
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db

    def run(self, *, titan_id: str, context, config: dict) -> int:
        """Follow up to (max_per_day − today's follows) eligible authors.
        Returns the count of NEW follows performed. No-op when disabled."""
        af = config.get("auto_follow", {}) or {}
        if not af.get("enabled", False):
            return 0
        min_rec = int(af.get("min_recurrence", 3))
        min_rel = float(af.get("min_relevance", 0.8))
        window_s = int(af.get("window_days", 7)) * 86400
        max_per_day = int(af.get("max_per_day", 2))

        remaining = max_per_day - self._followed_today(titan_id)
        if remaining <= 0:
            return 0

        candidates = self._recurring_high_relevance(
            titan_id=titan_id, min_rel=min_rel, window_s=window_s, min_rec=min_rec)
        if not candidates:
            return 0

        self_handles = self._self_handles(config)
        followed = 0
        for author, seen_n in candidates:
            if followed >= remaining:
                break
            a = (author or "").strip()
            if not a or a.lower() in self_handles:
                continue
            uid, is_following = self._lookup(a)
            if is_following or not uid:
                continue                      # already follow them, or can't resolve user_id
            if self._already_followed(titan_id, a):
                continue                      # never re-follow (dedup)
            res = self.gateway.follow(uid, context, consumer="auto_follow",
                                      handle=a, source_id=str(seen_n))
            status = getattr(res, "status", "")
            if status == "posted":
                followed += 1
                logger.info("[auto_follow] followed @%s (uid=%s, seen %d× rel≥%.2f)",
                            a, uid, seen_n, min_rel)
            elif status in ("rate_limited", "circuit_breaker", "disabled"):
                break                         # stop the batch on a hard stop
        return followed

    # ── helpers ──────────────────────────────────────────────────────
    def _followed_today(self, titan_id: str) -> int:
        try:
            c = sqlite3.connect(self._sx_db, timeout=5)
            n = c.execute(
                "SELECT count(*) FROM actions WHERE action_type='follow' "
                "AND titan_id=? AND status='posted' AND created_at >= ?",
                (titan_id, time.time() - 86400)).fetchone()[0]
            c.close()
            return int(n)
        except Exception:
            return 0

    def _recurring_high_relevance(self, *, titan_id, min_rel, window_s, min_rec):
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT author, count(*) n FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? AND created_at >= ? "
                "  AND author IS NOT NULL AND author != '' "
                "GROUP BY lower(author) HAVING n >= ? "
                "ORDER BY n DESC LIMIT 20",
                (titan_id, min_rel, time.time() - window_s, min_rec)).fetchall()
            et.close()
            return [(r["author"], int(r["n"])) for r in rows]
        except Exception as e:
            logger.warning("[auto_follow] felt_experiences query failed: %s", e)
            return []

    def _lookup(self, author: str):
        """(user_id, is_following) from community_registry; ('', 0) if unknown."""
        try:
            sg = sqlite3.connect(self._sg_db, timeout=5)
            sg.row_factory = sqlite3.Row
            row = sg.execute(
                "SELECT user_id, is_following FROM community_registry "
                "WHERE lower(user_name)=lower(?)", (author,)).fetchone()
            sg.close()
            if row:
                return (str(row["user_id"] or ""), int(row["is_following"] or 0))
        except Exception:
            pass
        return ("", 0)

    def _already_followed(self, titan_id: str, author: str) -> bool:
        try:
            c = sqlite3.connect(self._sx_db, timeout=5)
            n = c.execute(
                "SELECT count(*) FROM actions WHERE action_type='follow' "
                "AND titan_id=? AND status='posted' AND metadata LIKE ?",
                (titan_id, f'%"handle":"{author}"%')).fetchone()[0]
            c.close()
            return n > 0
        except Exception:
            return False

    @staticmethod
    def _self_handles(config: dict) -> set:
        own = [config.get("user_name", "")] + list(config.get("self_handles", []) or [])
        return {str(h).strip().lstrip("@").lower() for h in own if h}


__all__ = ("AutoFollowPolicy",)
