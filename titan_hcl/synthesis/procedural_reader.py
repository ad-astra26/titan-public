"""ProceduralSkillReader — read surface for `match_procedural_skill` (Phase 8).

Per `ARCHITECTURE_synthesis_engine.md §8.5` + `PLAN_synthesis_engine_Phase8.md §P8.D`.

Lives synthesis-worker-side (sole-writer process; no watermark needed). The
agno-side `match_procedural_skill` tool consumes the matching path via
BridgeRecall + watermark — this reader is the in-process engine-side
implementation behind `EngineRecall.recall(granularity="procedural")`.

Scoring:
  FAISS top-K * 4 candidates from skills_vectors.faiss → DuckDB join on
  procedural_skills filtered by `utility_score >= utility_floor` +
  `verified_at IS NOT NULL` → composite re-rank.

  composite_score(skill, query) = (1 - faiss_distance/normalizer)
                                  * (utility_score)
                                  * (1 + name_match_boost)

  - cosine surrogate: 1 - dist/normalizer (FAISS IndexFlatL2 returns squared
    L2 in [0..4] for unit vectors; normalizer = 4.0).
  - utility carries through (already in [-1, 1]); soft-retired skills
    pre-filtered.
  - name_match_boost = 0.2 when any ≥3-char query token is a case-folded
    substring of the skill name.

`should_delegate(top)` is the gate the agno tool checks:
  match_score >= match_floor AND skill.utility_score >= utility_floor
  AND skill.verified_at IS NOT NULL AND skill.utility_score >= 0.0.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Defaults from titan_params.toml [synthesis.skill]
DEFAULT_UTILITY_FLOOR: float = 0.3
DEFAULT_MATCH_FLOOR: float = 0.65

# FAISS IndexFlatL2 distance normalizer for cosine surrogate. Unit-norm
# vectors have ||a - b||² ∈ [0, 4]; dividing by 4 maps to [0, 1] then we
# invert to get a "match score" in [0, 1] where 1 = identical.
FAISS_L2_NORMALIZER: float = 4.0

NAME_MATCH_BOOST: float = 0.2


class ProceduralSkillReader:
    """Read-only consumer of ProceduralSkillStore. Synthesis-worker-side."""

    def __init__(
        self,
        skill_store: Any,
        *,
        utility_floor: float = DEFAULT_UTILITY_FLOOR,
        match_floor: float = DEFAULT_MATCH_FLOOR,
    ):
        self._store = skill_store
        self._utility_floor = float(utility_floor)
        self._match_floor = float(match_floor)

    def recall(
        self,
        query_text: str,
        *,
        k: int = 5,
    ) -> list[dict]:
        """Return up to `k` ranked skill dicts.

        Each result dict is the full skill row enriched with:
          - match_score (float in [0, 1])
          - name_match_boost (float)
          - cosine_surrogate (float)

        Empty list when:
          - FAISS index is empty
          - embedder failed
          - no skills pass utility + verified gate.
        """
        if not query_text:
            return []
        query_vec = self._store.embed_query(query_text)
        if query_vec is None:
            # No embedder wired — fall back to read_for_match utility-only ordering
            rows = self._store.read_for_match(
                utility_floor=self._utility_floor, k=k, verified_only=True,
            )
            return [
                {
                    **row,
                    "match_score": row["utility_score"],
                    "name_match_boost": 0.0,
                    "cosine_surrogate": 0.0,
                }
                for row in rows[:k]
            ]

        # FAISS top-K * 4 (over-fetch for the DuckDB-join filter step)
        try:
            hits = self._store.faiss_search(query_vec, top_k=max(1, k * 4))
        except Exception as e:
            logger.warning("[ProceduralSkillReader] faiss_search raised: %s", e)
            hits = []

        if not hits:
            return []

        # Pull eligible skill rows; we filter by (utility, verified) DB-side.
        all_rows = self._store.read_for_match(
            utility_floor=self._utility_floor,
            k=max(1, k * 4),
            verified_only=True,
        )
        # Index by embedding_id for FAISS join
        by_emb_id: dict[int, dict] = {
            int(row["embedding_id"]): row
            for row in all_rows
            if row.get("embedding_id") is not None and int(row["embedding_id"]) >= 0
        }

        query_tokens = {
            t.lower().strip(".,!?:;()[]\"'") for t in (query_text or "").split()
            if len(t) >= 3
        }

        scored: list[tuple[float, dict]] = []
        for emb_id, dist in hits:
            row = by_emb_id.get(emb_id)
            if row is None:
                continue
            # cosine surrogate from L2 distance on unit vectors
            cosine_surrogate = max(0.0, 1.0 - (float(dist) / FAISS_L2_NORMALIZER))
            name_lower = (row.get("name") or "").lower()
            nl_lower = (row.get("nl_description") or "").lower()
            name_match = any(t in name_lower or t in nl_lower for t in query_tokens)
            name_boost = NAME_MATCH_BOOST if name_match else 0.0
            utility = float(row.get("utility_score") or 0.0)
            match_score = max(0.0, min(1.0, cosine_surrogate * max(0.0, utility) * (1.0 + name_boost)))
            enriched = {
                **row,
                "match_score": match_score,
                "name_match_boost": name_boost,
                "cosine_surrogate": cosine_surrogate,
            }
            scored.append((match_score, enriched))

        scored.sort(key=lambda x: -x[0])
        return [enriched for _, enriched in scored[:k]]

    def should_delegate(self, top: Optional[dict]) -> bool:
        """Q6 delegate gate: utility_score ≥ utility_floor AND match_score ≥ match_floor
        AND verified_at IS NOT NULL AND utility_score ≥ 0.0 (rejected skills excluded)."""
        if not top:
            return False
        match_score = float(top.get("match_score") or 0.0)
        utility = float(top.get("utility_score") or 0.0)
        verified_at = top.get("verified_at")
        if verified_at is None:
            return False
        if utility < 0.0:
            return False
        if utility < self._utility_floor:
            return False
        if match_score < self._match_floor:
            return False
        return True
