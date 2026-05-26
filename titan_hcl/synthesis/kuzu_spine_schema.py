"""Phase 4 — Kuzu Concept-spine schema bootstrap.

Per `ARCHITECTURE_synthesis_engine.md` §6.1 + `PLAN_synthesis_engine_Phase4.md` §P4.A.

Creates the synthesis-engine multi-modal spine tables on the project's
single canonical Kuzu graph (the same graph that hosts Person/Topic/Trinity
entities — INV-13 outer-memory substrate). Tables are additive + idempotent:

NODE TABLES
- Concept(concept_id, version)    — versioned multi-modal spine root (§10)
- Production                       — procedural skill node (populated Phase 8)
- ActionChain                      — recurrent tool-call shape (populated Phase 8)
- HypothesisFork                   — probationary fork index (populated Phase 5)

REL TABLES
- COMPOSED_FROM (Concept → Concept) — decompile / traverse down (§10)
- COMPOSED_INTO (Concept → Concept) — recompile / traverse up (§10)
- USES_SKILL (Concept → Production)
- COMPILED_FROM (Production → ActionChain)
- EXPLORES (HypothesisFork → Concept) — repair-fork root anchor (§9.3)

Bootstrap is a one-shot called from `TitanKnowledgeGraph._init_schema()` after
the trinity schema. Empty Production/ActionChain/HypothesisFork tables ship in
P4 so consumers can issue Cypher without a schema-missing branch; population
lands in P5 / P8.

Invariants:
- Concept has composite PRIMARY KEY(concept_id, version) — two rows with the
  same concept_id but different versions coexist (§10 versioning; INV-3).
- All I/O via the single canonical Kuzu interface (`direct_memory.py` —
  arch §6.1 footnote).
- INV-Syn-3 (extended via P4.K proposed INV-Syn-7): synthesis_worker is sole
  writer of these tables; cross-process readers go through BridgeRecall (G18
  watermark-gated).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Schema definitions ──────────────────────────────────────────────

_NODE_TABLES: tuple[tuple[str, str], ...] = (
    (
        # Kuzu 0.11 does NOT support composite PRIMARY KEY in CREATE NODE
        # TABLE; we synthesize a single-column PK `pk = f"{concept_id}:v{version}"`
        # so the (concept_id, version) tuple stays canonical at the API
        # surface while honoring Kuzu's single-PK constraint. `concept_id`
        # + `version` stay as regular indexed properties for filter queries.
        "Concept",
        "pk STRING, concept_id STRING, version INT64, name STRING, "
        "memory_type STRING, groundedness DOUBLE, anchor_tx STRING, "
        "created_at DOUBLE, PRIMARY KEY(pk)",
    ),
    (
        "Production",
        "skill_id STRING, name STRING, utility_score DOUBLE, anchor_tx STRING, "
        "PRIMARY KEY(skill_id)",
    ),
    (
        "ActionChain",
        "chain_id STRING, shape STRING, success_count INT64, failure_count INT64, "
        "PRIMARY KEY(chain_id)",
    ),
    (
        "HypothesisFork",
        "fork_id STRING, root_anchor STRING, activation DOUBLE, status STRING, "
        "PRIMARY KEY(fork_id)",
    ),
)


# (rel_name, from_table, to_table). No properties on these rels in P4 — the
# spine relationship is identity-only; per-edge metadata (e.g. version-link
# context) lives on the Concept node row itself.
_REL_TABLES: tuple[tuple[str, str, str], ...] = (
    ("COMPOSED_FROM", "Concept", "Concept"),
    ("COMPOSED_INTO", "Concept", "Concept"),
    ("USES_SKILL", "Concept", "Production"),
    ("COMPILED_FROM", "Production", "ActionChain"),
    ("EXPLORES", "HypothesisFork", "Concept"),
)


# ── Probes (Kuzu CALL TABLE_INFO / CALL SHOW_TABLES) ─────────────────

def _table_exists(conn, table_name: str) -> bool:
    """Return True if a Kuzu node/rel table of this name exists."""
    try:
        qr = conn.execute("CALL SHOW_TABLES() RETURN *")
        while qr.has_next():
            row = qr.get_next()
            # SHOW_TABLES row shape varies by version; name is the first or
            # second column. Compare against both for portability.
            if not row:
                continue
            if (len(row) >= 1 and row[0] == table_name) or (
                len(row) >= 2 and row[1] == table_name
            ):
                return True
    except Exception as e:
        logger.debug("[kuzu_spine_schema] SHOW_TABLES probe failed: %s", e)
    return False


# ── Bootstrap entry-point ───────────────────────────────────────────

def bootstrap_spine_schema(graph: Any) -> dict:
    """Create the 4 node tables + 5 rel tables on the given Kuzu graph.

    `graph` is a TitanKnowledgeGraph (or anything exposing `_conn` that runs
    Cypher). Idempotent: re-running is a no-op (each CREATE is wrapped in a
    try-except + a CALL SHOW_TABLES presence probe first).

    Returns a summary dict for logging / test assertions:
      {"created_nodes": [...], "created_rels": [...], "errors": [...]}
    """
    out: dict = {"created_nodes": [], "created_rels": [], "errors": []}
    conn = getattr(graph, "_conn", None)
    if conn is None:
        logger.warning(
            "[kuzu_spine_schema] graph has no _conn — skipping spine bootstrap"
        )
        out["errors"].append("no_conn")
        return out

    for table_name, schema in _NODE_TABLES:
        if _table_exists(conn, table_name):
            continue
        try:
            conn.execute(f"CREATE NODE TABLE {table_name}({schema})")
            out["created_nodes"].append(table_name)
            logger.info("[kuzu_spine_schema] created node table %s", table_name)
        except Exception as e:
            # Tolerate the rare race / version-quirk where _table_exists
            # missed it but CREATE raises "already exists".
            msg = str(e).lower()
            if "already exists" in msg or "binder exception" in msg:
                continue
            logger.warning(
                "[kuzu_spine_schema] CREATE NODE TABLE %s failed: %s",
                table_name, e,
            )
            out["errors"].append(f"node:{table_name}:{e}")

    for rel_name, src, dst in _REL_TABLES:
        if _table_exists(conn, rel_name):
            continue
        try:
            conn.execute(
                f"CREATE REL TABLE {rel_name}(FROM {src} TO {dst})"
            )
            out["created_rels"].append(rel_name)
            logger.info("[kuzu_spine_schema] created rel table %s", rel_name)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "binder exception" in msg:
                continue
            logger.warning(
                "[kuzu_spine_schema] CREATE REL TABLE %s failed: %s",
                rel_name, e,
            )
            out["errors"].append(f"rel:{rel_name}:{e}")

    return out


__all__ = ("bootstrap_spine_schema",)
