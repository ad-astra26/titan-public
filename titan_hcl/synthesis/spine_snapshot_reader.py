"""titan_hcl/synthesis/spine_snapshot_reader.py — read-only spine reader over the
JSON snapshot (RFP_titan_authored_soul_diary §7.P4 — chat-path self-recall).

The agno chat worker can NOT open the live Kuzu spine (Kuzu 0.11's `read_only`
flag still acquires the exclusive write-lock vs the synthesis_worker writer — the
same constraint that motivated the `spine_snapshot.json` pattern). So to run
concept- AND self-granularity recall in the CHAT path, `EngineRecall` needs a
`kuzu_reader` — this class is a faithful read-only one backed by the atomic
`spine_snapshot.json` synthesis_worker re-exports every ~60s. G18-pure (a file
read; no Kuzu handle, no bus/RPC, no lock).

It implements the two methods EngineRecall's spine granularities call:
  · `spine_list_concepts(limit, offset, memory_type)` → concept-granularity
    (matches `TitanKnowledgeGraph.spine_list_concepts`: latest version per
    concept_id, groundedness DESC).
  · `spine_self_recall()` → self-granularity (the SELF hub). Its engrams are the
    concepts with `domain_hint == "self"` — the SAME predicate the live
    `SELF_HAS_ENGRAM` linker uses (`direct_memory.py:828`
    `WHERE c.domain_hint = 'self'`) — so the snapshot set is faithful to the hub.
    Skills (`Production` nodes) ride the live spine only; they are not in the
    concept snapshot → empty here (forward-compatible; skills are ~empty today).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SNAPSHOT_NAME = "spine_snapshot.json"
_STALE_SECONDS = 600  # 10 min — far above the 60s export cadence; logged, not fatal


class SnapshotSpineReader:
    """Read-only `kuzu_reader` over `spine_snapshot.json` for the chat-path
    `EngineRecall` (concept + self granularity). mtime-cached (cheap stat)."""

    def __init__(self, data_dir: str = "data", *,
                 snapshot_path: Optional[str] = None):
        self._path = snapshot_path or os.path.join(data_dir, _SNAPSHOT_NAME)
        self._cache_mtime: float = -1.0
        self._latest: list[dict] = []  # latest-version-per-concept_id, cached

    def _load_latest(self) -> list[dict]:
        """Latest-version-per-concept_id concept rows from the snapshot,
        mtime-cached. Soft-fail → []."""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return []
        if mtime == self._cache_mtime and self._latest:
            return self._latest
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.debug("[SnapshotSpineReader] snapshot read failed: %s", e)
            return []
        age = time.time() - mtime
        if age > _STALE_SECONDS:
            logger.info("[SnapshotSpineReader] snapshot is %.0fs stale (>%ds)",
                        age, _STALE_SECONDS)
        latest: dict[str, dict] = {}
        for c in (snap.get("concepts") or []):
            if not isinstance(c, dict):
                continue
            cid = c.get("concept_id")
            if not cid:
                continue
            try:
                ver = int(c.get("version", 0) or 0)
            except (TypeError, ValueError):
                continue
            cur = latest.get(cid)
            if cur is None or ver > int(cur.get("version", 0) or 0):
                latest[cid] = c
        self._latest = list(latest.values())
        self._cache_mtime = mtime
        return self._latest

    def spine_list_concepts(self, limit: int = 100, offset: int = 0,
                            memory_type: Optional[str] = None) -> list[dict]:
        """Latest-version-per-concept_id, ordered by groundedness DESC — mirrors
        `TitanKnowledgeGraph.spine_list_concepts` (the shape EngineRecall's
        concept granularity consumes)."""
        out: list[dict] = []
        for c in self._load_latest():
            if memory_type is not None and c.get("memory_type") != memory_type:
                continue
            out.append({
                "concept_id": c.get("concept_id"),
                "version": int(c.get("version", 0) or 0),
                "name": c.get("name") or "",
                "memory_type": c.get("memory_type"),
                "groundedness": float(c.get("groundedness", 0.0) or 0.0),
                "anchor_tx": c.get("anchor_tx"),
                "created_at": float(c.get("created_at", 0.0) or 0.0),
            })
        out.sort(key=lambda r: r.get("groundedness", 0.0), reverse=True)
        return out[offset:offset + limit]

    def spine_self_recall(self) -> dict:
        """The SELF hub over the snapshot: engrams = concepts with
        `domain_hint == "self"` (the live `SELF_HAS_ENGRAM` predicate),
        newest-first; skills ride the live spine only (empty here). Shape matches
        `TitanKnowledgeGraph.spine_self_recall`."""
        engrams = [c for c in self._load_latest()
                   if (c.get("domain_hint") or "") == "self"]
        engrams.sort(key=lambda c: float(c.get("created_at", 0.0) or 0.0),
                     reverse=True)
        return {
            "engrams": [{
                "concept_id": c.get("concept_id"),
                "version": int(c.get("version", 0) or 0),
                "name": c.get("name") or "",
                "domain_hint": c.get("domain_hint") or "",
                "anchor_tx": c.get("anchor_tx") or "",
                "groundedness": float(c.get("groundedness", 0.0) or 0.0),
            } for c in engrams],
            "skills": [],
        }
