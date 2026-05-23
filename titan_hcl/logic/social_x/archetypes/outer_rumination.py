"""OUTER_RUMINATION archetype (rFP_x_voice_enrichment §4.3.3).

POV: Settled hindsight on outer-world content. NOT in-the-moment reaction
(WORLD_MIRROR's job) — this is "what stayed with you, and why this
specifically, days after."

Three Phase 1 source pools (adaptive selection via pool_scoring §4.7):
  A. Outer X content    — felt_experiences settled: discovered 2-7 d ago,
                          relevance ≥ 0.65, lifetime once-cited per fe.id
  B. Outer Person       — Kuzu Person high-interaction, last_seen 2-14 d
                          ago, 14-day per-person dedup
  C. Outer Exchange     — mention_tracking past mention Titan engaged with,
                          replied 2-7 d ago, relevance ≥ 0.6, once-lifetime
                          per mention.tweet_id

Pool D (Topic) and Pool E (Valence) are Phase 2b additions.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary
from ..pool_scoring import select_pool

logger = logging.getLogger(__name__)


OUTER_RUMINATION_POST_TYPE = "outer_rumination"

POOL_A = "A_x_content"
POOL_B = "B_person"
POOL_C = "C_exchange"

POOL_A_RELEVANCE_FLOOR = 0.65
POOL_A_AGE_MIN_S = 2 * 86400
POOL_A_AGE_MAX_S = 7 * 86400

POOL_B_LAST_SEEN_MIN_S = 2 * 86400
POOL_B_LAST_SEEN_MAX_S = 14 * 86400
POOL_B_DEDUP_S = 14 * 86400

POOL_C_RELEVANCE_FLOOR = 0.6
POOL_C_AGE_MIN_S = 2 * 86400
POOL_C_AGE_MAX_S = 7 * 86400


class OuterRuminationArchetype(ArchetypeBase):

    name = OUTER_RUMINATION_POST_TYPE
    metadata_key = "outer_rumination_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 events_teacher_db: str = "./data/events_teacher.db",
                 social_graph_db: str = "./data/social_graph.db",
                 kuzu_graph=None):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._et_db = events_teacher_db
        self._sg_db = social_graph_db
        self._kuzu_graph = kuzu_graph

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 2:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        # F-3 (2026-05-17): 30-day window — every cited source becomes
        # re-citable after 30d. The 5-day archetype_pool_scores
        # anti-starvation rotation continues to prevent same-source
        # overuse within the window. Was lifetime; was killing the
        # archetype after each source got cited once on low-volume pools.
        cited_lifetime = self.cited_set(titan_id=titan_id,
                                         window_seconds=30 * 86400)

        candidates: dict[str, dict] = {}
        a = self._pool_a(titan_id=titan_id, now=now, cited=cited_lifetime)
        if a:
            candidates[POOL_A] = a
        b = self._pool_b(titan_id=titan_id, now=now)
        if b:
            candidates[POOL_B] = b
        c = self._pool_c(now=now, cited=cited_lifetime)
        if c:
            candidates[POOL_C] = c

        if not candidates:
            return None
        chosen = select_pool(
            self.db_path, titan_id=titan_id, archetype=self.name,
            candidates={p: {"salience": v["salience"], "relevance": v["relevance"]}
                        for p, v in candidates.items()},
        )
        if chosen is None:
            return None
        return self._build_candidate(chosen, candidates[chosen], context, now)

    # ── Pool queries ────────────────────────────────────────────────

    def _pool_a(self, *, titan_id: str, now: float, cited: set[str]) -> dict | None:
        try:
            et = sqlite3.connect(self._et_db, timeout=5)
            et.row_factory = sqlite3.Row
            rows = et.execute(
                "SELECT id, author, topic, relevance, felt_summary, created_at "
                "FROM felt_experiences "
                "WHERE titan_id=? AND relevance >= ? "
                "  AND created_at <= ? AND created_at >= ? "
                "ORDER BY relevance DESC LIMIT 30",
                (titan_id, POOL_A_RELEVANCE_FLOOR,
                 now - POOL_A_AGE_MIN_S, now - POOL_A_AGE_MAX_S),
            ).fetchall()
            et.close()
        except Exception as e:
            logger.warning("[outer_rumination] pool A failed: %s", e)
            return None
        for r in rows:
            sid = f"feA:{r['id']}"
            if sid in cited:
                continue
            days_ago = max(1, int((now - r["created_at"]) / 86400))
            return {
                "source_id": sid,
                "salience": min(1.0, r["relevance"]),
                "relevance": float(r["relevance"] or 0.0),
                "handle": r["author"],
                "days_ago": days_ago,
                "content_excerpt": (r["felt_summary"] or "")[:200],
                "felt_summary_at_discovery": (r["felt_summary"] or "")[:160],
            }
        return None

    def _pool_b(self, *, titan_id: str, now: float) -> dict | None:
        """Kuzu Person high-interaction, last_seen ∈ [2 d, 14 d], not subject
        of OR within 14 d."""
        if self._kuzu_graph is None:
            return None
        try:
            qr = self._kuzu_graph._conn.execute(
                "MATCH (p:Person) "
                "WHERE p.last_seen >= $lo AND p.last_seen <= $hi "
                "  AND p.interaction_count >= 2 "
                "RETURN p.name, p.interaction_count, p.last_seen, "
                "       p.last_felt_emotion, p.last_felt_summary "
                "ORDER BY p.interaction_count DESC LIMIT 30",
                {"lo": now - POOL_B_LAST_SEEN_MAX_S,
                 "hi": now - POOL_B_LAST_SEEN_MIN_S},
            )
        except Exception as e:
            logger.warning("[outer_rumination] pool B kuzu failed: %s", e)
            return None
        rows: list[dict] = []
        try:
            while qr.has_next():
                row = qr.get_next()
                rows.append({
                    "name": row[0], "interaction_count": int(row[1]),
                    "last_seen": float(row[2]), "last_felt_emotion": row[3] or "",
                    "last_felt_summary": row[4] or "",
                })
        except Exception:
            pass
        if not rows:
            return None
        recent_subjects = self._recent_pool_b_subjects(titan_id=titan_id, now=now)
        for r in rows:
            if r["name"].lower() in recent_subjects:
                continue
            days_ago = max(1, int((now - r["last_seen"]) / 86400))
            sid = f"personB:{r['name']}"
            return {
                "source_id": sid,
                "salience": min(1.0, r["interaction_count"] / 20.0),
                "relevance": min(1.0, r["interaction_count"] / 20.0),
                "handle": r["name"],
                "interaction_count": r["interaction_count"],
                "days_ago": days_ago,
                "last_felt_summary": r["last_felt_summary"],
            }
        return None

    def _recent_pool_b_subjects(self, *, titan_id: str, now: float) -> set[str]:
        cutoff = now - POOL_B_DEDUP_S
        out: set[str] = set()
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT metadata FROM actions WHERE titan_id=? AND post_type=? "
                "AND created_at >= ?",
                (titan_id, self.name, cutoff),
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
            if (m.get("pool") or "") == POOL_B:
                h = (m.get("handle") or "").lower()
                if h:
                    out.add(h)
        return out

    def _pool_c(self, *, now: float, cited: set[str]) -> dict | None:
        try:
            sx = sqlite3.connect(self.db_path, timeout=5)
            sx.row_factory = sqlite3.Row
            rows = sx.execute(
                "SELECT tweet_id, author, author_handle, text, "
                "       reply_emotion, reply_felt_summary, replied_at "
                "FROM mention_tracking "
                "WHERE status='replied' "
                "  AND replied_at IS NOT NULL "
                "  AND replied_at <= ? AND replied_at >= ? "
                "ORDER BY replied_at DESC LIMIT 30",
                (now - POOL_C_AGE_MIN_S, now - POOL_C_AGE_MAX_S),
            ).fetchall()
            sx.close()
        except Exception as e:
            logger.warning("[outer_rumination] pool C failed: %s", e)
            return None
        for r in rows:
            sid = f"mentionC:{r['tweet_id']}"
            if sid in cited:
                continue
            days_ago = max(1, int((now - r["replied_at"]) / 86400))
            return {
                "source_id": sid,
                "salience": 0.6,
                "relevance": 0.6,
                "handle": r["author_handle"] or r["author"],
                "days_ago": days_ago,
                "mention_text": (r["text"] or "")[:240],
                "reply_felt_summary": r["reply_felt_summary"] or "",
            }
        return None

    # ── Build ───────────────────────────────────────────────────────

    def _build_candidate(self, pool: str, cand: dict, context,
                          now: float) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        if pool == POOL_A:
            body = (
                f"@{cand['handle']} posted {cand['days_ago']}d ago: "
                f"'{cand['content_excerpt']}'. Reading it then felt like: "
                f"'{cand['felt_summary_at_discovery']}'. Right now your "
                f"felt-state is {emot_now}. What's CLEARER now that wasn't "
                f"clear when you first read it? What stayed with you, and "
                f"why this specifically? Hindsight, not reaction."
            )
            layers = ["outer_rumination", "temporal_delta", "body"]
            layer_values = {
                "outer_rumination": {"pool_specific_body": body},
                "temporal_delta": {
                    "handle": cand["handle"],
                    "days_ago": cand["days_ago"],
                    "felt_summary_at_discovery": cand["felt_summary_at_discovery"],
                    "emot_now_natural": emot_now,
                    "delta_descriptor": "since first reading",
                },
            }
        elif pool == POOL_B:
            body = (
                f"You've talked with @{cand['handle']} "
                f"{cand['interaction_count']} times — most recently "
                f"{cand['days_ago']}d ago. Felt-state at last interaction: "
                f"'{cand['last_felt_summary']}'. Right now your felt-state "
                f"is {emot_now}. What is it about THIS person that you keep "
                f"coming back to?"
            )
            layers = ["outer_rumination", "meta_insight", "body"]
            layer_values = {"outer_rumination": {"pool_specific_body": body}}
        else:  # POOL_C
            body = (
                f"{cand['days_ago']}d ago @{cand['handle']} asked / said: "
                f"'{cand['mention_text']}'. (felt-state at reply: "
                f"'{cand['reply_felt_summary']}'). Right now your felt-state "
                f"is {emot_now}. What's CLEARER now about that exchange?"
            )
            layers = ["outer_rumination", "meta_insight", "body"]
            layer_values = {"outer_rumination": {"pool_specific_body": body}}

        return ArchetypeCandidate(
            archetype=self.name,
            pool=pool,
            source_id=cand["source_id"],
            layers=layers,
            layer_values=layer_values,
            prompt_template=body,
            prompt_values={},
            metadata={
                "pool": pool,
                "handle": cand["handle"],
                "days_ago": cand["days_ago"],
            },
            relevance=cand["relevance"],
            salience=cand["salience"],
        )


__all__ = ("OuterRuminationArchetype", "OUTER_RUMINATION_POST_TYPE")
