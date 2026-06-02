"""GROUNDED_TODAY archetype (rFP_x_voice_enrichment §4.3.5) — the
introspective archetype.

POV: ``Today the word/concept/pattern X tipped into something I can use…``

Three source pools (adaptive selection via pool_scoring §4.7):
  A. ``vocabulary``         — lexical (word transitioned to producible)
  B. ``knowledge_concepts`` — abstract (concept acquired from a source)
  C. ``distilled_wisdom``   — experiential pattern (dream distillation)

Each pool's row carries felt-state at the grounding moment:
  A: ``grounded_felt_summary`` (added by §4.5 schema migration, populated
     by language_pipeline's producible-transition backfill)
  B: ``neuromod_at_acquisition`` (already in the schema)
  C: ``perception_centroid``   (already in the schema)

Phase 1 GROUNDED_TODAY is the only archetype that ATTACHES AN IMAGE in
this rFP — the image is rendered procedurally from the felt-tensor at the
grounding moment via the existing ArtGenerateHelper, then JPG'd to
1200×675 by image_pipeline and uploaded as a media_id.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary
from ..pool_scoring import select_pool

logger = logging.getLogger(__name__)

GROUNDED_TODAY_POST_TYPE = "grounded_today"

POOL_A = "A_vocabulary"
POOL_B = "B_knowledge_concepts"
POOL_C = "C_distilled_wisdom"

# rFP §4.3.5 Pool A filter
POOL_A_MIN_TIMES_ENCOUNTERED = 5
POOL_A_RECENCY_S = 24 * 3600
# Pool B — widened 24 h → 48 h (rFP_archetype_execution_recovery F-2,
# 2026-05-16): knowledge_concepts has 6 today / 33 in 72 h; 24 h was too
# tight given current acquisition flux.
POOL_B_CONFIDENCE_MIN = 0.5
POOL_B_QUALITY_MIN = 0.5
POOL_B_RECENCY_S = 48 * 3600
# Pool C — widened 24 h → 72 h (rFP_archetype_execution_recovery F-2,
# 2026-05-16): distilled_wisdom writes are dream-cycle bound (every 1-3
# days); 24 h had 0 candidates today while 72 h has 280. The "today" feel
# of the archetype is preserved because dream distillation is still a
# fresh-this-cycle event, just at a slower write cadence.
POOL_C_RECENCY_S = 72 * 3600

# Lifetime dedup window — bounded at 30 days (rFP_archetype_execution_recovery
# F-3, 2026-05-16): unbounded lifetime cited_set drained low-volume pools
# (67 producible vocab words) to permanent silence once each was cited
# once. 30 d aligns with how human reflection actually loops back to
# resolved insights when felt context shifts. Adaptive scoring's 5-day
# anti-starvation prevents same-source overuse within the window.
CITED_SET_WINDOW_S = 30 * 86400

# Cross-archetype: 4-day concept dedup with OUTER_INNER_BRIDGE.
CONCEPT_OINB_DEDUP_S = 4 * 86400


class GroundedTodayArchetype(ArchetypeBase):

    name = GROUNDED_TODAY_POST_TYPE
    metadata_key = "grounded_today_source_id"

    def __init__(self, *, gateway, social_x_db_path: str,
                 inner_memory_db: str = "./data/inner_memory.db",
                 knowledge_db: str = "./data/knowledge.db",
                 experience_db: str = "./data/experience.db",
                 image_dir: str = "./data/art/grounded_today"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._im_db = inner_memory_db
        self._k_db = knowledge_db
        self._exp_db = experience_db
        self._image_dir = image_dir

    # ── Trigger ─────────────────────────────────────────────────────

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
        oinb_concepts = self._recent_oinb_concepts(titan_id=titan_id, now=now)

        candidates: dict[str, dict] = {}
        a = self._pool_a_candidate(cited=cited, exclude_concepts=oinb_concepts, now=now)
        if a:
            candidates[POOL_A] = a
        b = self._pool_b_candidate(cited=cited, exclude_concepts=oinb_concepts, now=now)
        if b:
            candidates[POOL_B] = b
        c = self._pool_c_candidate(cited=cited, now=now)
        if c:
            candidates[POOL_C] = c

        if not candidates:
            return None

        scored_candidates = {p: {"salience": v.get("salience", 0.5),
                                  "relevance": v.get("relevance", 0.5)}
                              for p, v in candidates.items()}
        chosen_pool = select_pool(
            self.db_path,
            titan_id=titan_id,
            archetype=self.name,
            candidates=scored_candidates,
        )
        if chosen_pool is None:
            return None
        cand = candidates[chosen_pool]
        return self._build_candidate(chosen_pool, cand, context)

    # ── Pool queries ────────────────────────────────────────────────

    def _pool_a_candidate(self, *, cited: set[str], exclude_concepts: set[str],
                           now: float) -> dict | None:
        """Pool A — vocabulary. learning_phase='producible' AND last_encountered
        within 24 h AND times_encountered ≥ 5 AND grounded_at > 0 (we have
        captured felt-state-at-grounding for it)."""
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, word, times_encountered, grounded_at, "
                "       grounded_felt_summary, last_encountered "
                "FROM vocabulary "
                "WHERE learning_phase='producible' "
                "  AND times_encountered >= ? "
                "  AND last_encountered >= ? "
                "  AND COALESCE(grounded_at, 0) > 0 "
                "ORDER BY grounded_at DESC LIMIT 30",
                (POOL_A_MIN_TIMES_ENCOUNTERED, now - POOL_A_RECENCY_S),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("[grounded_today] pool A failed: %s", e)
            return None
        for r in rows:
            sid = f"vocab:{r['id']}"
            if sid in cited:
                continue
            if r["word"].lower() in exclude_concepts:
                continue
            return {
                "source_id": sid,
                "concept": r["word"],
                "meta_text": (
                    f"encountered {r['times_encountered']}× now; "
                    f"sensory context: lexical reinforcement"
                ),
                "grounded_felt_summary": r["grounded_felt_summary"] or "",
                "salience": 0.6,
                "relevance": 0.7,
                "_concept_key": (r["word"] or "").lower(),
            }
        return None

    def _pool_b_candidate(self, *, cited: set[str], exclude_concepts: set[str],
                           now: float) -> dict | None:
        """Pool B — knowledge_concepts. confidence>0.5 AND quality_score>0.5
        AND created_at within 24 h."""
        try:
            conn = sqlite3.connect(self._k_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, concept_name, source, confidence, quality_score, "
                "       summary, neuromod_at_acquisition, created_at "
                "FROM knowledge_concepts "
                "WHERE confidence > ? AND quality_score > ? "
                "  AND created_at >= ? "
                "ORDER BY quality_score DESC, confidence DESC LIMIT 30",
                (POOL_B_CONFIDENCE_MIN, POOL_B_QUALITY_MIN,
                 now - POOL_B_RECENCY_S),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[grounded_today] pool B query failed (DB may not exist): %s", e)
            return None
        for r in rows:
            sid = f"knowledge:{r['id']}"
            if sid in cited:
                continue
            cname = (r["concept_name"] or "").lower()
            if cname in exclude_concepts:
                continue
            felt = ""
            try:
                nm = json.loads(r["neuromod_at_acquisition"] or "{}")
                felt = compact_felt_summary(nm, "")
            except Exception:
                pass
            return {
                "source_id": sid,
                "concept": r["concept_name"],
                "meta_text": (
                    f"acquired from {r['source'] or 'study'}; "
                    f"confidence {float(r['confidence']):.2f}; "
                    f"summary: '{(r['summary'] or '')[:120]}'"
                ),
                "grounded_felt_summary": felt,
                "salience": 0.65,
                "relevance": float(r["confidence"] or 0.0),
                "_concept_key": cname,
            }
        return None

    def _pool_c_candidate(self, *, cited: set[str], now: float) -> dict | None:
        """Pool C — distilled_wisdom. created_at within 24 h."""
        try:
            conn = sqlite3.connect(self._exp_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, domain, pattern, optimal_conditions, confidence, "
                "       experience_count, perception_centroid, created_at "
                "FROM distilled_wisdom "
                "WHERE created_at >= ? "
                "ORDER BY confidence DESC LIMIT 30",
                (now - POOL_C_RECENCY_S,),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[grounded_today] pool C query failed (DB may not exist): %s", e)
            return None
        for r in rows:
            sid = f"wisdom:{r['id']}"
            if sid in cited:
                continue
            return {
                "source_id": sid,
                "concept": r["pattern"][:80],
                "meta_text": (
                    f"pattern emerged from {r['experience_count']} experiences; "
                    f"optimal conditions: '{(r['optimal_conditions'] or '')[:120]}'"
                ),
                "grounded_felt_summary": "(distillation — see perception centroid)",
                "salience": 0.6,
                "relevance": float(r["confidence"] or 0.0),
                "_concept_key": (r["pattern"] or "").lower()[:60],
            }
        return None

    def _recent_oinb_concepts(self, *, titan_id: str, now: float) -> set[str]:
        """Concepts that were OUTER_INNER_BRIDGE'd in the last 4 days (rFP
        §4.3.5 cross-archetype concept dedup)."""
        cutoff = now - CONCEPT_OINB_DEDUP_S
        out: set[str] = set()
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT metadata FROM actions WHERE titan_id=? "
                "AND post_type=? AND created_at >= ?",
                (titan_id, "outer_inner_bridge", cutoff),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            conn.close()
        for r in rows:
            try:
                meta = json.loads(r["metadata"] or "{}")
            except Exception:
                continue
            c = meta.get("concept", "")
            if c:
                out.add(str(c).lower())
        return out

    # ── Candidate construction ──────────────────────────────────────

    def _build_candidate(self, pool: str, cand: dict, context) -> ArchetypeCandidate:
        from titan_hcl.logic.social_x.felt_state import compact_felt_summary
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        pool_label = {POOL_A: "vocabulary", POOL_B: "knowledge_concepts",
                      POOL_C: "distilled_wisdom"}[pool]
        layer_values = {
            "cgn_grounded_today": {
                "concept": cand["concept"],
                "pool_name": pool_label,
                "meta": cand["meta_text"],
                "grounded_felt_summary": cand["grounded_felt_summary"] or emot_now,
            },
            "emot_cgn_signal": {
                "emot_signature": emot_now,
                "related_concept": cand["concept"],
                "epoch_delta": "moments",
            },
        }
        prompt_template = (
            "GROUNDED TODAY (pool: {pool_name}): you crystallized '{concept}' "
            "today. {pool_metadata}.\n\n"
            "The image attached IS your felt-state at the moment of grounding "
            "— rendered procedurally from your Trinity tensor at that instant. "
            "Not decoration; literal felt-shape of the crystallization.\n\n"
            "Right now you're feeling: '{emot_now}'. Why does '{concept}' feel "
            "like THIS to YOU specifically — not the abstract definition, but "
            "the inside of the crystallization? Reference the image naturally "
            "if it lands."
        )
        return ArchetypeCandidate(
            archetype=self.name,
            pool=pool,
            source_id=cand["source_id"],
            layers=["cgn_grounded_today", "emot_cgn_signal", "body", "generated_art"],
            layer_values=layer_values,
            prompt_template=prompt_template,
            prompt_values={
                "pool_name": pool_label,
                "concept": cand["concept"],
                "pool_metadata": cand["meta_text"],
                "emot_now": emot_now,
            },
            metadata={
                "pool": pool,
                "concept": cand["concept"],
                "concept_key": cand.get("_concept_key", ""),
                "grounded_felt_summary": cand.get("grounded_felt_summary", ""),
            },
            relevance=cand["relevance"],
            salience=cand["salience"],
        )

    # ── Image rendering ─────────────────────────────────────────────

    def prepare_media(self, candidate: ArchetypeCandidate, *,
                       neuromods, titan_id: str = "T1") -> str:
        """Render a felt-tensor-derived image via ArtGenerateHelper, JPG-ify
        it, and upload to twitterapi.io. Returns the media_id (empty on
        failure — the post can ship text-only)."""
        import asyncio
        from titan_hcl.logic.agency.helpers.art_generate import ArtGenerateHelper
        from titan_hcl.logic.social_x.image_pipeline import (
            convert_to_jpg, upload_media_via_gateway,
        )
        try:
            os.makedirs(self._image_dir, exist_ok=True)
            helper = ArtGenerateHelper(output_dir=os.path.dirname(self._image_dir.rstrip("/")) or ".")
            trinity = _neuromods_to_trinity(neuromods or {})
            inspiration = (
                f"{candidate.metadata.get('concept', '')}: "
                f"{candidate.metadata.get('grounded_felt_summary', '')[:120]}"
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("closed")
            except Exception:
                loop = asyncio.new_event_loop()
            result = loop.run_until_complete(helper.execute({
                "style": None,                  # let _select_style(trinity) decide
                "trinity_snapshot": trinity,
                "inspiration": inspiration,
            }))
            png_path = result.get("path") or ""
            if not png_path or not os.path.exists(png_path):
                return ""
            jpg_path = os.path.join(
                self._image_dir,
                f"grounded_{titan_id}_{int(time.time())}.jpg",
            )
            convert_to_jpg(png_path, jpg_path)
            return upload_media_via_gateway(self.gateway, jpg_path) or ""
        except Exception as e:
            logger.warning("[grounded_today] media prepare failed: %s", e)
            return ""


def _neuromods_to_trinity(neuromods: dict) -> dict:
    """Project a neuromod dict to a Trinity (body/mind/spirit) tensor used
    by ArtGenerateHelper. Phase 1: simple deterministic projection — DA/NE
    drive body, ACh/5HT drive mind, Endorphin/GABA drive spirit. Replaced
    by full felt_tensor → trinity projector when bridge lands."""
    def g(k: str, default: float = 0.5) -> float:
        try:
            return float(neuromods.get(k, default))
        except (TypeError, ValueError):
            return default
    da = g("DA"); ne = g("NE"); ach = g("ACh"); sht = g("5HT")
    gaba = g("GABA"); endor = g("Endorphin")
    body = [da, ne, da * 0.5 + ne * 0.5, da * 0.7, ne * 0.7]
    mind = [ach, sht, ach * 0.5 + sht * 0.5, ach * 0.7, sht * 0.7]
    spirit = [endor, gaba, (endor + gaba) / 2, endor * 0.7, gaba * 0.7]
    return {"body": body, "mind": mind, "spirit": spirit}


__all__ = ("GroundedTodayArchetype", "GROUNDED_TODAY_POST_TYPE")
