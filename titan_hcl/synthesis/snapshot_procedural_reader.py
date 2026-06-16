"""SnapshotProceduralReader — agno-side (cross-process) procedural skill match.

Break F (RFP_synthesis_reuse_and_routing_revival): the synthesis worker holds the
`procedural_skills` DuckDB exclusive lock (G21 single-writer), so the agno process
CANNOT open it. Before this reader, `agno_worker` built `EngineRecall` with
`procedural_reader=None` → `recall(granularity="procedural")` returned None
unconditionally → the `match_procedural_skill` tool always answered "no match" and
the OML `skill_utility`/`skill_matched` features never lit (skill_delegate=0
fleet-wide).

This reconstructs the ProceduralSkillStore match surface from the two atomic,
G18-pure files the synthesis worker publishes:
  - data/skills_snapshot.json  (skill metadata incl. embedding_id, utility, verified)
  - data/skills_vectors.faiss  (the skill embedding index — read-only mmap)
and delegates ALL scoring to the canonical `ProceduralSkillReader`, so the agno-side
match is byte-identical to the engine-side one (no logic divergence). Mirrors the
`SnapshotSpineReader` pattern the chat path already uses for concept/self recall.

Both processes embed with the same `get_text_embedder()` singleton, so the FAISS
cosine is valid cross-process; the FAISS row id equals `embedding_id` (same file,
same search), so the join is exact.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from titan_hcl.synthesis.procedural_reader import (
    DEFAULT_MATCH_FLOOR,
    DEFAULT_UTILITY_FLOOR,
    ProceduralSkillReader,
)

logger = logging.getLogger(__name__)


class _SnapshotSkillStoreView:
    """A read-only, file-backed stand-in for ProceduralSkillStore exposing only the
    three primitives `ProceduralSkillReader` consumes: `embed_query`, `faiss_search`,
    `read_for_match`. Reloads each file when its mtime changes (the synthesis worker
    rewrites them atomically tmp+rename)."""

    def __init__(self, data_dir: str, embedder: Optional[Callable[[str], Any]]):
        self._data_dir = data_dir
        self._embedder = embedder
        self._faiss = None
        self._faiss_mtime: Optional[float] = None
        self._snap: Optional[dict] = None
        self._snap_mtime: Optional[float] = None

    def _snapshot_path(self) -> str:
        return os.path.join(self._data_dir, "skills_snapshot.json")

    def _faiss_path(self) -> str:
        return os.path.join(self._data_dir, "skills_vectors.faiss")

    def _load_snapshot(self) -> Optional[dict]:
        path = self._snapshot_path()
        try:
            mt = os.path.getmtime(path)
        except OSError:
            self._snap = None
            return None
        if self._snap is None or mt != self._snap_mtime:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._snap = json.load(fh)
                self._snap_mtime = mt
            except Exception as e:  # noqa: BLE001
                logger.debug("[SnapshotProceduralReader] snapshot load failed: %s", e)
                self._snap = None
        return self._snap

    def embed_query(self, text: str) -> Optional[Any]:
        if self._embedder is None or not text:
            return None
        try:
            return self._embedder(text)
        except Exception as e:  # noqa: BLE001
            logger.debug("[SnapshotProceduralReader] embed_query failed: %s", e)
            return None

    def _ensure_faiss(self) -> None:
        path = self._faiss_path()
        try:
            mt = os.path.getmtime(path)
        except OSError:
            self._faiss = None
            return
        if self._faiss is None or mt != self._faiss_mtime:
            try:
                import faiss  # type: ignore
                self._faiss = faiss.read_index(path)
                self._faiss_mtime = mt
            except Exception as e:  # noqa: BLE001
                logger.debug("[SnapshotProceduralReader] faiss read failed: %s", e)
                self._faiss = None

    def faiss_search(self, query_vec: Any, top_k: int = 20) -> list[tuple[int, float]]:
        try:
            import numpy as np
        except ImportError:
            return []
        self._ensure_faiss()
        if self._faiss is None or self._faiss.ntotal == 0:
            return []
        try:
            vec = np.asarray(query_vec, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            k = min(int(top_k), self._faiss.ntotal)
            dists, ids = self._faiss.search(vec, k)
            return [(int(ids[0][i]), float(dists[0][i]))
                    for i in range(k) if ids[0][i] >= 0]
        except Exception as e:  # noqa: BLE001
            logger.debug("[SnapshotProceduralReader] faiss_search failed: %s", e)
            return []

    def read_for_match(self, *, utility_floor: float, k: int,
                       verified_only: bool = True) -> list[dict]:
        """Mirror ProceduralSkillStore.read_for_match over the snapshot: positives
        only (the utility_floor + verified gate excludes [negative]/unproven cells,
        whose snapshot `utility_score` is 0.0 and `verified_at` is None), ordered by
        utility, capped at k. Rows carry `embedding_id` for the FAISS join."""
        snap = self._load_snapshot()
        if not snap:
            return []
        rows: list[dict] = []
        for s in snap.get("skills", []):
            util = float(s.get("utility_score") or 0.0)
            if util < utility_floor:
                continue
            if verified_only and s.get("verified_at") is None:
                continue
            emb_id = int(s.get("embedding_id", -1))
            if emb_id < 0:
                continue
            rows.append({
                **s,
                "embedding_id": emb_id,
                # utility_score (= best positive cell time_cost) is the proficiency.
                "utility_score": util,
                # gate at source returns positives only (INV-EEL-5); set explicitly
                # so the reader's polarity guard is satisfied.
                "polarity": "positive",
            })
        rows.sort(key=lambda r: -float(r.get("utility_score") or 0.0))
        return rows[:k]


class SnapshotProceduralReader:
    """The `procedural_reader` EngineRecall calls (`.recall(query_text, k=)`),
    backed by the snapshot+faiss files and the canonical scoring."""

    def __init__(self, data_dir: str, embedder: Optional[Callable[[str], Any]], *,
                 utility_floor: float = DEFAULT_UTILITY_FLOOR,
                 match_floor: float = DEFAULT_MATCH_FLOOR):
        self._view = _SnapshotSkillStoreView(data_dir, embedder)
        self._reader = ProceduralSkillReader(
            self._view, utility_floor=utility_floor, match_floor=match_floor)

    def recall(self, query_text: str, *, k: int = 5) -> list[dict]:
        return self._reader.recall(query_text, k=k)

    def should_delegate(self, top: Optional[dict]) -> bool:
        return self._reader.should_delegate(top)
