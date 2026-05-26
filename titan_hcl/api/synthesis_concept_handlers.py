"""API handlers for `/v6/synthesis/concepts/*` (Phase 4 §P4.I).

Backs the Observatory's ConceptSpinePanel: list spines, get one spine's
full version history, fetch the groundedness heatmap.

The handlers open the canonical Kuzu graph (`data/knowledge_graph.kuzu`)
in **read-only** mode — the api process is a separate process from
synthesis_worker (which writes the spine), so the read-only open is the
G18 watermark-gated cross-process pattern in miniature. Kuzu 0.11
supports concurrent read-only opens against an active writer process.

All handlers soft-fail to a structured `{"error": ..., "ok": false}`
shape on read errors so the frontend can render a friendly empty state
without a 500 cascade. Successful responses use `{"ok": true, ...}`.

`api/v6.py` mounts these handlers via ROUTE_TABLE entries.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_KUZU_PATH = "data/knowledge_graph.kuzu"


# ── Cache (read-only Kuzu handle is moderately expensive to open) ────


_KUZU_HANDLE_CACHE: dict[str, Any] = {}


def _resolve_kuzu_path() -> str:
    """Pick up TITAN_DATA_DIR like the rest of the stack so tests + shadow
    directories work without monkey-patching."""
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, "knowledge_graph.kuzu")


def _get_kuzu_reader(path: Optional[str] = None) -> Optional[Any]:
    """Lazy-open a read-only TitanKnowledgeGraph against the canonical
    Kuzu file. Cached per-path so repeat handler calls share the handle.
    Returns None on open failure (handler returns a friendly empty
    response in that case)."""
    if path is None:
        path = _resolve_kuzu_path()
    cached = _KUZU_HANDLE_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        logger.debug(
            "[synthesis_concept_handlers] Kuzu file missing: %s", path,
        )
        return None
    try:
        # TitanKnowledgeGraph re-opens read-write by default; for the
        # api process we want read-only. Kuzu 0.11's read_only flag on
        # Database lets a reader coexist with synthesis_worker's writer.
        # We bypass the TitanKnowledgeGraph wrapper here and open Kuzu
        # directly — the wrapper would attempt schema bootstrap which is
        # forbidden in a read-only handle.
        import kuzu
        db = kuzu.Database(path, read_only=True)
        conn = kuzu.Connection(db)
        # Wrap in a minimal facade matching the spine_* method names so
        # the handlers can call them uniformly.
        reader = _ReadOnlyKuzuReader(db, conn)
        _KUZU_HANDLE_CACHE[path] = reader
        return reader
    except Exception as e:
        logger.warning(
            "[synthesis_concept_handlers] Kuzu read-only open failed (%s): "
            "%s — handlers will return empty",
            path, e,
        )
        return None


def _reset_cache_for_tests() -> None:
    """Test-only: drop the cached Kuzu handle. Production must never
    call this (handles are intentionally process-lifetime cached)."""
    _KUZU_HANDLE_CACHE.clear()


class _ReadOnlyKuzuReader:
    """Minimal read-only facade — exposes just the spine_* methods the
    handlers call. Reuses the Cypher queries from TitanKnowledgeGraph but
    against a `read_only=True` Database so the api process can coexist
    with synthesis_worker's writer."""

    def __init__(self, db, conn):
        self._db = db
        self._conn = conn

    @staticmethod
    def _pk(concept_id: str, version: int) -> str:
        return f"{concept_id}:v{int(version)}"

    def spine_get_concept_version(
        self, concept_id: str, version: int,
    ) -> Optional[dict]:
        try:
            qr = self._conn.execute(
                "MATCH (c:Concept {pk: $pk}) "
                "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                "c.groundedness, c.anchor_tx, c.created_at",
                {"pk": self._pk(concept_id, version)},
            )
            if not qr.has_next():
                return None
            row = qr.get_next()
            return {
                "concept_id": row[0], "version": int(row[1]),
                "name": row[2], "memory_type": row[3],
                "groundedness": float(row[4]),
                "anchor_tx": row[5], "created_at": float(row[6]),
            }
        except Exception:
            return None

    def spine_list_concepts(
        self, limit: int = 100, offset: int = 0,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        all_rows: list[dict] = []
        try:
            if memory_type is not None:
                qr = self._conn.execute(
                    "MATCH (c:Concept) WHERE c.memory_type = $mt "
                    "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                    "c.groundedness, c.anchor_tx, c.created_at",
                    {"mt": memory_type},
                )
            else:
                qr = self._conn.execute(
                    "MATCH (c:Concept) "
                    "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                    "c.groundedness, c.anchor_tx, c.created_at"
                )
            while qr.has_next():
                row = qr.get_next()
                all_rows.append({
                    "concept_id": row[0], "version": int(row[1]),
                    "name": row[2], "memory_type": row[3],
                    "groundedness": float(row[4]),
                    "anchor_tx": row[5], "created_at": float(row[6]),
                })
        except Exception as e:
            logger.debug(
                "[ReadOnlyKuzuReader] spine_list_concepts: %s", e,
            )
            return []
        latest: dict[str, dict] = {}
        for r in all_rows:
            existing = latest.get(r["concept_id"])
            if existing is None or r["version"] > existing["version"]:
                latest[r["concept_id"]] = r
        ordered = sorted(
            latest.values(),
            key=lambda r: r.get("groundedness", 0.0), reverse=True,
        )
        return ordered[offset: offset + limit]

    def spine_get_all_versions(self, concept_id: str) -> list[dict]:
        """Return every version of a concept, ordered by version ASC.
        Specific to the /v6/synthesis/concepts/<id> endpoint."""
        try:
            qr = self._conn.execute(
                "MATCH (c:Concept {concept_id: $cid}) "
                "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                "c.groundedness, c.anchor_tx, c.created_at",
                {"cid": concept_id},
            )
            out: list[dict] = []
            while qr.has_next():
                row = qr.get_next()
                out.append({
                    "concept_id": row[0], "version": int(row[1]),
                    "name": row[2], "memory_type": row[3],
                    "groundedness": float(row[4]),
                    "anchor_tx": row[5], "created_at": float(row[6]),
                })
            out.sort(key=lambda r: r["version"])
            return out
        except Exception as e:
            logger.debug(
                "[ReadOnlyKuzuReader] spine_get_all_versions: %s", e,
            )
            return []

    def spine_get_composition_edges(
        self, concept_id: str, version: int,
    ) -> dict:
        """Return {composed_from: [...], composed_into: [...]} for a
        specific (concept_id, version)."""
        anchor_pk = self._pk(concept_id, version)
        out: dict[str, list[dict]] = {"composed_from": [], "composed_into": []}
        for rel, key in (("COMPOSED_FROM", "composed_from"),
                         ("COMPOSED_INTO", "composed_into")):
            try:
                qr = self._conn.execute(
                    f"MATCH (a:Concept {{pk: $apk}})-[:{rel}]->(b:Concept) "
                    f"RETURN b.concept_id, b.version, b.name",
                    {"apk": anchor_pk},
                )
                while qr.has_next():
                    row = qr.get_next()
                    out[key].append({
                        "concept_id": row[0], "version": int(row[1]),
                        "name": row[2],
                    })
            except Exception:
                pass
        return out


# ── Public handler functions (wired into v6.py ROUTE_TABLE) ─────────


def get_synthesis_concepts(
    limit: int = 50, offset: int = 0,
    memory_type: Optional[str] = None,
) -> dict:
    """GET /v6/synthesis/concepts — paginated list of spines (latest
    version per concept_id), ordered by groundedness DESC. Optional
    `memory_type` filter (declarative|procedural|episodic|meta)."""
    reader = _get_kuzu_reader()
    if reader is None:
        return {"ok": True, "concepts": [], "total": 0, "kuzu": "missing"}
    try:
        # Fetch larger pool first for accurate `total`.
        all_concepts = reader.spine_list_concepts(
            limit=10_000, offset=0, memory_type=memory_type,
        )
        total = len(all_concepts)
        page = all_concepts[offset: offset + max(0, min(int(limit), 500))]
        return {
            "ok": True,
            "concepts": page,
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "memory_type": memory_type,
        }
    except Exception as e:
        logger.warning("[handlers] get_synthesis_concepts failed: %s", e)
        return {"ok": False, "error": str(e), "concepts": [], "total": 0}


def get_synthesis_concept(concept_id: str) -> dict:
    """GET /v6/synthesis/concepts/<concept_id> — full spine of one
    concept: every version + composition edges (both directions) keyed
    by the latest version."""
    reader = _get_kuzu_reader()
    if reader is None:
        return {"ok": True, "concept_id": concept_id,
                "versions": [], "kuzu": "missing"}
    if not concept_id:
        return {"ok": False, "error": "empty_concept_id"}
    try:
        versions = reader.spine_get_all_versions(concept_id)
        if not versions:
            return {"ok": True, "concept_id": concept_id,
                    "versions": [], "exists": False}
        latest = versions[-1]
        edges = reader.spine_get_composition_edges(
            concept_id, latest["version"],
        )
        return {
            "ok": True,
            "concept_id": concept_id,
            "exists": True,
            "latest_version": latest["version"],
            "versions": versions,
            "composed_from": edges["composed_from"],
            "composed_into": edges["composed_into"],
        }
    except Exception as e:
        logger.warning(
            "[handlers] get_synthesis_concept(%s) failed: %s",
            concept_id, e,
        )
        return {"ok": False, "error": str(e),
                "concept_id": concept_id, "versions": []}


def get_synthesis_concepts_heatmap() -> dict:
    """GET /v6/synthesis/concepts/heatmap — 4×10 grid of concept counts
    bucketed by (memory_type, groundedness_decile). memory_type rows:
    declarative, procedural, episodic, meta. Columns: decile 0..9 where
    column k contains concepts with k/10 ≤ groundedness < (k+1)/10."""
    reader = _get_kuzu_reader()
    if reader is None:
        return {
            "ok": True,
            "heatmap": _empty_heatmap(),
            "kuzu": "missing",
        }
    try:
        concepts = reader.spine_list_concepts(limit=10_000)
        heatmap = _empty_heatmap()
        for c in concepts:
            mt = c.get("memory_type", "meta")
            if mt not in heatmap:
                mt = "meta"
            g = max(0.0, min(0.9999, float(c.get("groundedness", 0.0) or 0.0)))
            decile = int(g * 10)
            heatmap[mt][decile] += 1
        return {
            "ok": True,
            "heatmap": heatmap,
            "total": len(concepts),
        }
    except Exception as e:
        logger.warning("[handlers] heatmap failed: %s", e)
        return {"ok": False, "error": str(e), "heatmap": _empty_heatmap()}


def _empty_heatmap() -> dict[str, list[int]]:
    return {
        "declarative": [0] * 10,
        "procedural": [0] * 10,
        "episodic": [0] * 10,
        "meta": [0] * 10,
    }


__all__ = (
    "get_synthesis_concepts",
    "get_synthesis_concept",
    "get_synthesis_concepts_heatmap",
    "_get_kuzu_reader",
    "_reset_cache_for_tests",
    "_ReadOnlyKuzuReader",
)
