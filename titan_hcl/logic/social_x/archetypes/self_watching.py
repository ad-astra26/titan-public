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

# CONFIDENCE_FLOOR 2026-05-12: lowered 0.7 → 0.5 to match live
# self_insights confidence distribution. Empirical check on T1
# inner_memory.db (2026-05-12 dry-run probe): in the last 24h the
# 3 BEHAVIORAL_SUB_MODES yield max(state_audit.confidence)=0.6794,
# max(prediction)=0.6, max(coherence_check)=0.6. The original 0.7
# floor was aspirational and silently rejected ALL behavioral
# self_insights → archetype could never fire (blocking rFP §4.8
# gate-9 14-day acceptance). 0.5 still excludes low-quality noise
# (most insights cluster in 0.55-0.68 band) while admitting the
# behavioral signal the archetype was designed to surface.
CONFIDENCE_FLOOR = 0.5
# F-2-finish (rFP_social_x_improvements §B.3.F-2, 2026-05-17): widened
# 24h → 72h to match F-2 Pool C convention (used by grounded_today for
# distilled_wisdom). Behavioral self_insights write rate is bursty + slow
# — live probe on T1 2026-05-17 found 0 candidates in 24h but 5 in 72h
# (T1 self_watching went dark 3 days after its 3-post burst on
# 2026-05-12..14). 72h matches the "slow source pool" Pool C semantics.
RECENCY_S = 72 * 3600

# Lifetime dedup window — bounded at 30 days (rFP_archetype_execution_recovery
# F-3, 2026-05-16): unbounded lifetime cited_set was a safe default when
# self_insights was a thin table, but with 8945 rows lifetime dedup adds
# unnecessary DB scan cost on every probe. 30 d window matches the human
# rhythm of revisiting prior self-observations after felt-context shifts.
CITED_SET_WINDOW_S = 30 * 86400

# rFP §4.3.9: "POV = Titan looking at its own *behavioral patterns*,
# not outer content." self_reasoning.INTROSPECT_SUB_MODES has 5 modes;
# vocabulary_probe + architecture_query are pure telemetry (vocab stats,
# architecture snapshot — NOT behavioral). prediction, coherence_check,
# state_audit are the behavioral subset.
BEHAVIORAL_SUB_MODES = ("prediction", "coherence_check", "state_audit")


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
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 2:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None
        if self.same_archetype_blocked(titan_id=titan_id, now=now):
            return None

        cited = self.cited_set(titan_id=titan_id,
                                window_seconds=CITED_SET_WINDOW_S)

        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(BEHAVIORAL_SUB_MODES))
            rows = conn.execute(
                "SELECT id, sub_mode, epoch, timestamp, data, confidence "
                "FROM self_insights "
                "WHERE timestamp >= ? AND confidence >= ? "
                f"  AND sub_mode IN ({placeholders}) "
                "ORDER BY confidence DESC, timestamp DESC LIMIT 30",
                (now - RECENCY_S, CONFIDENCE_FLOOR, *BEHAVIORAL_SUB_MODES),
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
            insight_summary = _summarize_for_sub_mode(
                r["sub_mode"], data_obj
            )[:240]
            data_excerpt = json.dumps(
                {k: v for k, v in data_obj.items()
                 if k in ("pattern", "metric", "score", "n_observations")}
            )[:160] or json.dumps(data_obj)[:160]
            # Escape literal {/} in data_excerpt before f-string inlining.
            # Downstream `ArchetypeCandidate.render_prompt()` calls
            # `prompt_template.format(**prompt_values)` — JSON braces would
            # be interpreted as format placeholders ({0} → "index 0 out of
            # range for positional args tuple"). Double them so format()
            # restores them as literal {} during render. Same escape applied
            # to insight_summary as a defense-in-depth measure (sub-mode
            # summaries that include JSON-ish output would otherwise hit
            # the same trap).
            data_excerpt = data_excerpt.replace("{", "{{").replace("}", "}}")
            insight_summary = (insight_summary
                                .replace("{", "{{")
                                .replace("}", "}}"))
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


def _summarize_for_sub_mode(sub_mode: str, data_obj: dict) -> str:
    """Sub_mode-aware insight extractor. Production self_insights rows lack
    a generic ``summary`` field; each sub_mode has its own data shape, so
    we read the field that carries behavioral content for that mode.

    Falls through to the generic summary/insight/description keys, then to
    a sub_mode-aware terse fallback (NEVER returns the unhelpful
    ``(see insight data)`` placeholder for behavioral sub_modes)."""
    # First try the generic keys (in case future sub_modes adopt them).
    for k in ("summary", "insight", "description"):
        v = data_obj.get(k)
        if v:
            return str(v)

    if sub_mode == "coherence_check":
        gaps = data_obj.get("gaps") or []
        if gaps:
            interp = (gaps[0] or {}).get("interpretation")
            if interp:
                # Prepend gap count for the "I noticed N drifts" register
                n = data_obj.get("gaps_found") or len(gaps)
                return f"{n} drifts since last self-profile; first: {interp}"
        return "self-profile coherence checked — no gaps significant enough to name"

    if sub_mode == "prediction":
        # Common shapes: {predicted_state: ..., trajectory: [...]}
        pred = data_obj.get("predicted_state") or data_obj.get("prediction")
        traj = data_obj.get("trajectory")
        if pred:
            return f"predicting next-state: {pred}"
        if traj:
            return f"trajectory observed across {len(traj)} steps"
        return "next-state prediction crystallized"

    if sub_mode == "state_audit":
        # Common shape: {neuromods: {...}, mood: ..., chi: ...}
        mood = data_obj.get("mood") or data_obj.get("emotion")
        chi = data_obj.get("chi")
        if mood and chi is not None:
            return f"state audit: mood={mood}, chi={chi}"
        if mood:
            return f"state audit: mood={mood}"
        return "state-audit snapshot crystallized"

    return "(see insight data)"


__all__ = ("SelfWatchingArchetype", "SELF_WATCHING_POST_TYPE")
