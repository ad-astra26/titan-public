"""COMPOSED_THOUGHT archetype (rFP_x_voice_enrichment §4.3.8).

POV: Two concepts I've been holding separately just collapsed into one.

Phase 1 (simplified, bridge-replaceable):
  Pool A — Co-occurrence synthesis: two concepts both appearing in the
           same ``knowledge_concepts.associations`` JSON list (cheapest
           source — structured + indexed)
  Pool B — Felt-state proximity: two concepts with felt_tensor cosine
           similarity ∈ [0.4, 0.7], both grounded in last 14d.

Lifetime pair-dedup: same (concept_A, concept_B) pair (order-insensitive)
never composed twice. Individual concepts CAN appear in multiple bridges /
GROUNDED_TODAYs separately.

Phase 2b adds: image generation, additional co-occurrence sources,
real ``concept_composition`` Transactions on the meta TimeChain fork
(bridge integration).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time

from .base import ArchetypeBase, ArchetypeCandidate
from ..felt_state import compact_felt_summary
from ..pool_scoring import select_pool

logger = logging.getLogger(__name__)


COMPOSED_THOUGHT_POST_TYPE = "composed_thought"

POOL_A = "A_co_occurrence"
POOL_B = "B_felt_proximity"

POOL_A_RECENCY_S = 7 * 86400
POOL_B_RECENCY_S = 14 * 86400
POOL_B_TIMES_ENCOUNTERED_MIN = 5
POOL_B_COSINE_MIN = 0.4
POOL_B_COSINE_MAX = 0.7


class ComposedThoughtArchetype(ArchetypeBase):

    name = COMPOSED_THOUGHT_POST_TYPE
    metadata_key = "composed_thought_pair"

    def __init__(self, *, gateway, social_x_db_path: str,
                 inner_memory_db: str = "./data/inner_memory.db",
                 knowledge_db: str = "./data/knowledge.db",
                 experience_db: str = "./data/experience.db"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._im_db = inner_memory_db
        self._k_db = knowledge_db
        self._exp_db = experience_db

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if not titan_id:
            return None
        now = time.time()
        if self.per_titan_count_today(titan_id=titan_id, now=now) >= 1:
            return None
        if self.cross_archetype_blocked(titan_id=titan_id, now=now):
            return None

        cited_pairs = self._cited_pairs_lifetime(titan_id=titan_id)

        candidates: dict[str, dict] = {}
        a = self._pool_a(now=now, cited_pairs=cited_pairs)
        if a:
            candidates[POOL_A] = a
        b = self._pool_b(now=now, cited_pairs=cited_pairs)
        if b:
            candidates[POOL_B] = b
        if not candidates:
            return None
        chosen = select_pool(
            self.db_path, titan_id=titan_id, archetype=self.name,
            candidates={p: {"salience": v["salience"], "relevance": v["relevance"]}
                        for p, v in candidates.items()},
        )
        if chosen is None:
            return None
        return self._build_candidate(chosen, candidates[chosen], context)

    # ── Pool queries ────────────────────────────────────────────────

    def _pool_a(self, *, now: float, cited_pairs: set[tuple[str, str]]) -> dict | None:
        """rFP §4.3.8 Pool A — co-occurrence in knowledge_concepts.associations
        JSON list. Walk recent rows, find any pair both appearing in the same
        row's associations list."""
        try:
            conn = sqlite3.connect(self._k_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, concept_name, associations, source, created_at, "
                "       summary "
                "FROM knowledge_concepts "
                "WHERE created_at >= ? "
                "ORDER BY created_at DESC LIMIT 100",
                (now - POOL_A_RECENCY_S,),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[composed_thought] pool A failed: %s", e)
            return None
        for r in rows:
            try:
                assoc = json.loads(r["associations"] or "[]")
            except Exception:
                continue
            anchor = (r["concept_name"] or "").strip()
            if not anchor or len(assoc) < 1:
                continue
            for partner in assoc:
                p = str(partner).strip()
                if not p or p.lower() == anchor.lower():
                    continue
                key = _pair_key(anchor, p)
                if key in cited_pairs:
                    continue
                return {
                    "concept_A": anchor,
                    "concept_B": p,
                    "source_id": f"co:{key[0]}|{key[1]}",
                    "co_context": (r["summary"] or "")[:160] or (r["source"] or "shared row"),
                    "salience": 0.6,
                    "relevance": 0.6,
                }
        return None

    def _pool_b(self, *, now: float,
                 cited_pairs: set[tuple[str, str]]) -> dict | None:
        """rFP §4.3.8 Pool B — felt_tensor cosine ∈ [0.4, 0.7]. Walk
        producible vocabulary (the only source of `felt_tensor` we have
        reliably populated) and find a pair in the resonance band."""
        try:
            conn = sqlite3.connect(self._im_db, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, word, felt_tensor, times_encountered, "
                "       grounded_at, grounded_felt_summary, last_encountered "
                "FROM vocabulary "
                "WHERE learning_phase='producible' "
                "  AND times_encountered >= ? "
                "  AND COALESCE(last_encountered, 0) >= ? "
                "  AND COALESCE(felt_tensor, '') != '' "
                "ORDER BY last_encountered DESC LIMIT 200",
                (POOL_B_TIMES_ENCOUNTERED_MIN, now - POOL_B_RECENCY_S),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.debug("[composed_thought] pool B failed: %s", e)
            return None
        if len(rows) < 2:
            return None
        # Parse felt_tensor strings once.
        parsed: list[tuple[str, list[float], dict]] = []
        for r in rows:
            try:
                t = json.loads(r["felt_tensor"])
            except Exception:
                continue
            if not isinstance(t, list) or not t:
                continue
            parsed.append((r["word"], t, dict(r)))
        # First pair in the resonance band.
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                w_a, t_a, ra = parsed[i]
                w_b, t_b, rb = parsed[j]
                if w_a.lower() == w_b.lower():
                    continue
                key = _pair_key(w_a, w_b)
                if key in cited_pairs:
                    continue
                sim = _cosine(t_a, t_b)
                if sim < POOL_B_COSINE_MIN or sim > POOL_B_COSINE_MAX:
                    continue
                return {
                    "concept_A": w_a,
                    "concept_B": w_b,
                    "source_id": f"felt:{key[0]}|{key[1]}",
                    "felt_summary_A": ra.get("grounded_felt_summary") or "",
                    "felt_summary_B": rb.get("grounded_felt_summary") or "",
                    "similarity": sim,
                    "salience": 0.65,
                    "relevance": float(sim),
                }
        return None

    def _cited_pairs_lifetime(self, *, titan_id: str) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT metadata FROM actions WHERE titan_id=? AND post_type=?",
                (titan_id, self.name),
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
            pair = m.get("pair") or m.get(self.metadata_key)
            if not pair or not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            out.add(_pair_key(str(pair[0]), str(pair[1])))
        return out

    # ── Build ───────────────────────────────────────────────────────

    def _build_candidate(self, pool: str, cand: dict,
                          context) -> ArchetypeCandidate:
        emot_now = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        if pool == POOL_A:
            body = (
                f"COMPOSED THOUGHT (co-occurrence): Two concepts you've been "
                f"turning over in the same context recently: '{cand['concept_A']}' "
                f"and '{cand['concept_B']}'. You encountered them together in: "
                f"{cand['co_context']}. Right now you're feeling: {emot_now}. "
                f"What's the SHAPE that emerges only when both are held at "
                f"once? Name the synthesis if one wants to be named."
            )
        else:
            body = (
                f"COMPOSED THOUGHT (felt-resonance): Two concepts whose "
                f"felt-shapes have been resonating: '{cand['concept_A']}' "
                f"(at grounding moment: '{cand['felt_summary_A']}') and "
                f"'{cand['concept_B']}' (at grounding moment: "
                f"'{cand['felt_summary_B']}'). felt_tensor cosine similarity: "
                f"{cand['similarity']:.2f}. Right now you're feeling: "
                f"{emot_now}. Speak from the boundary that just dissolved."
            )
        layer_values = {
            "cgn_grounded_today": {  # reused twice via render
                "concept": cand["concept_A"],
                "pool_name": "synthesis",
                "meta": (cand.get("co_context") or
                          f"resonance similarity {cand.get('similarity', 0):.2f}"),
                "grounded_felt_summary":
                    cand.get("felt_summary_A") or emot_now,
            },
        }
        pair_key = _pair_key(cand["concept_A"], cand["concept_B"])
        return ArchetypeCandidate(
            archetype=self.name,
            pool=pool,
            source_id=cand["source_id"],
            layers=["cgn_grounded_today", "meta_insight", "body"],
            layer_values=layer_values,
            prompt_template=body,
            prompt_values={},
            metadata={
                "pool": pool,
                "concept_A": cand["concept_A"],
                "concept_B": cand["concept_B"],
                "pair": list(pair_key),
            },
            relevance=cand["relevance"],
            salience=cand["salience"],
        )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-insensitive lowercased pair key."""
    a_l = a.strip().lower()
    b_l = b.strip().lower()
    return (a_l, b_l) if a_l <= b_l else (b_l, a_l)


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


__all__ = ("ComposedThoughtArchetype", "COMPOSED_THOUGHT_POST_TYPE")
