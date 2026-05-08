"""SELF_WATCHING archetype (rFP_x_voice_enrichment §4.3.9).

POV: Titan looking at its own *behavioral patterns*, not outer content.
``I keep doing X. Watching myself, here's what I see.``

Source: ``inner_memory.db.self_insights`` — a recent high-confidence
insight (``confidence ≥ 0.7``, ``timestamp`` within last 24 h) that has
not yet been cited by SELF_WATCHING (lifetime once-cited per row).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary

logger = logging.getLogger(__name__)

SELF_WATCHING_POST_TYPE = "self_watching"

CONFIDENCE_FLOOR = 0.7
RECENCY_S = 24 * 3600


class SelfWatchingArchetype(ArchetypeBase):

    name = SELF_WATCHING_POST_TYPE
    metadata_key = "self_watching_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 inner_memory_db: str = "./data/inner_memory.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
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

        cited = self.cited_set(titan_id=titan_id)  # lifetime

        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, sub_mode, epoch, timestamp, data, confidence "
                "FROM self_insights "
                "WHERE timestamp >= ? AND confidence >= ? "
                "ORDER BY confidence DESC, timestamp DESC LIMIT 30",
                (now - RECENCY_S, CONFIDENCE_FLOOR),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[self_watching] insights query failed: %s", e)
            return None
        if not rows:
            return None

        for r in rows:
            sid = f"insight:{r['id']}"
            if sid in cited:
                continue
            try:
                data_obj = json.loads(r["data"] or "{}")
            except Exception:
                data_obj = {}
            insight_summary = (
                data_obj.get("summary")
                or data_obj.get("insight")
                or data_obj.get("description")
                or "(see insight data)"
            )[:240]
            data_excerpt = json.dumps(
                {k: v for k, v in data_obj.items()
                 if k in ("pattern", "metric", "score", "n_observations")}
            )[:160] or json.dumps(data_obj)[:160]
            epoch_span = data_obj.get("epoch_span") or "recent"
            return self._build_candidate(
                source_id=sid,
                titan_id=titan_id,
                row=r,
                insight_summary=insight_summary,
                data_excerpt=data_excerpt,
                epoch_span=epoch_span,
                context=context,
            )
        return None

    def _build_candidate(self, *, source_id: str, titan_id: str, row,
                          insight_summary: str, data_excerpt: str,
                          epoch_span: str, context) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        register_by_titan = {
            "T1": ("I've noticed I keep doing X. Watching myself, the question "
                   "is — is that the move, or am I just defaulting?"),
            "T2": ("Pattern observed: when state Y, I sequence Z. Why this "
                   "sequence and not others? What's selecting it?"),
            "T3": ("Sat with my own behavior over [N] epochs. Here's what I see "
                   "about how I tend to..."),
        }
        register_hint = register_by_titan.get(titan_id, register_by_titan["T1"])

        body = (
            f"SELF-WATCHING (sub_mode '{row['sub_mode']}', epoch {row['epoch']}): "
            f"You noticed something about your own behavior — "
            f"'{insight_summary}'. Confidence {float(row['confidence']):.2f}. "
            f"Pattern observed over {epoch_span} (data: {data_excerpt}). "
            f"Right now you're feeling: {emot_now}. "
            f"Speak from inside the noticing. Voice register hint: "
            f"{register_hint}"
        )
        layer_values = {
            "self_insight_layer": {
                "sub_mode": row["sub_mode"],
                "when": row["epoch"],
                "insight_summary": insight_summary,
                "confidence": float(row["confidence"] or 0.0),
                "epoch_span": epoch_span,
            },
        }
        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=source_id,
            layers=["self_insight_layer", "meta_insight", "body"],
            layer_values=layer_values,
            prompt_template=body,
            prompt_values={},
            metadata={
                "insight_id": row["id"],
                "sub_mode": row["sub_mode"],
                "confidence": float(row["confidence"] or 0.0),
            },
            relevance=float(row["confidence"] or 0.0),
            salience=float(row["confidence"] or 0.0),
        )


__all__ = ("SelfWatchingArchetype", "SELF_WATCHING_POST_TYPE")
