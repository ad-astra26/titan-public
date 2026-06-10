"""Phase 4 — Kuzu Engram-spine schema bootstrap (+ Phase-B Concept→Engram migration).

Per `ARCHITECTURE_synthesis_engine.md` §6.1/§6.2 + `PLAN_synthesis_engine_Phase4.md`
§P4.A + `RFP_synthesis_engram_grounding.md` §7.B (the spine node "concept" was
renamed → **Engram**; +4 decomposed grounding axes + advisory domain_hint).

Creates the synthesis-engine multi-modal spine tables on the project's
single canonical Kuzu graph (the same graph that hosts Person/Topic/Trinity
entities — INV-13 outer-memory substrate). Tables are additive + idempotent:

NODE TABLES
- Engram(concept_id, version)     — versioned multi-modal spine root (§10; renamed
                                    from Concept, +axis_used/verified/felt/fluent
                                    DOUBLE + domain_hint STRING — RFP §7.B)
- Production                       — procedural skill node (populated Phase 8)
- ActionChain                      — recurrent tool-call shape (populated Phase 8)
- HypothesisFork                   — probationary fork index (populated Phase 5)

REL TABLES
- COMPOSED_FROM (Engram → Engram)  — decompile / traverse down (§10)
- COMPOSED_INTO (Engram → Engram)  — recompile / traverse up (§10)
- USES_SKILL (Engram → Production)
- COMPILED_FROM (Production → ActionChain)  — does NOT reference the spine node
- EXPLORES (HypothesisFork → Engram) — repair-fork root anchor (§9.3)

Bootstrap is a one-shot called from `TitanKnowledgeGraph._init_schema()` after
the trinity schema; `migrate_concept_to_engram` runs immediately BEFORE it (same
hook) so an existing `Concept`-schema graph is copy-migrated to `Engram` before
the bootstrap presence-checks. Empty Production/ActionChain/HypothesisFork tables
ship in P4 so consumers can issue Cypher without a schema-missing branch.

Invariants:
- Engram has composite PRIMARY KEY(concept_id, version) — two rows with the
  same concept_id but different versions coexist (§10 versioning; INV-3).
- All I/O via the single canonical Kuzu interface (`direct_memory.py` —
  arch §6.1 footnote).
- INV-Syn-3 (extended via P4.K proposed INV-Syn-7): synthesis_worker is sole
  writer of these tables; cross-process readers go through BridgeRecall (G18
  watermark-gated).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Schema definitions ──────────────────────────────────────────────

_NODE_TABLES: tuple[tuple[str, str], ...] = (
    (
        # Kuzu 0.11 does NOT support composite PRIMARY KEY in CREATE NODE
        # TABLE; we synthesize a single-column PK `pk = f"{concept_id}:v{version}"`
        # so the (concept_id, version) tuple stays canonical at the API
        # surface while honoring Kuzu's single-PK constraint. `concept_id`
        # + `version` stay as regular indexed properties for filter queries.
        "Engram",
        "pk STRING, concept_id STRING, version INT64, name STRING, "
        "memory_type STRING, groundedness DOUBLE, anchor_tx STRING, "
        "created_at DOUBLE, "
        # Phase B (RFP_synthesis_engram_grounding §7.B) — decomposed grounding
        # axes (BRAIN §3.4-native: used→B_i, verified→c, felt, fluent→time_cost)
        # + advisory domain_hint. Born here for fresh installs; existing Titans
        # populated by migrate_concept_to_engram. Unpopulated (0.0/NULL) until
        # Phase C/D/F. `groundedness` stays as the derived recall-ranking scalar.
        "axis_used DOUBLE, axis_verified DOUBLE, axis_felt DOUBLE, "
        "axis_fluent DOUBLE, domain_hint STRING, "
        "PRIMARY KEY(pk)",
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
    (
        # Self — the self-knowledge HUB (RFP_titan_authored_soul_diary §7.P3a,
        # INV-SD-16). A singleton per Titan (one Kuzu graph = one Titan) that
        # LINKS all Titan-about-himself data so "what have I learned / what can
        # I do?" resolves in one hop (SELF_HAS_ENGRAM + SELF_HAS_SKILL). The
        # outward-expression rels (SELF_HAS_EXPRESSION) + the Persona node
        # (SELF_HAS_PERSONA) are a DEFERRED later step (see the RFP §7.P3 +
        # frontmatter scope_decisions — persona mechanic undesigned).
        "Self",
        "id STRING, created_at DOUBLE, PRIMARY KEY(id)",
    ),
)

# Canonical id for the per-graph Self singleton (one Kuzu graph = one Titan).
SELF_NODE_ID = "self"


# (rel_name, from_table, to_table). No properties on these rels in P4 — the
# spine relationship is identity-only; per-edge metadata (e.g. version-link
# context) lives on the Concept node row itself.
_REL_TABLES: tuple[tuple[str, str, str], ...] = (
    ("COMPOSED_FROM", "Engram", "Engram"),
    ("COMPOSED_INTO", "Engram", "Engram"),
    ("USES_SKILL", "Engram", "Production"),
    ("COMPILED_FROM", "Production", "ActionChain"),  # does NOT touch the spine node
    ("EXPLORES", "HypothesisFork", "Engram"),
    # SELF hub edges (§7.P3a, INV-SD-16): his self-knowledge in one hop.
    ("SELF_HAS_ENGRAM", "Self", "Engram"),        # diary entries + self-about engrams
    ("SELF_HAS_SKILL", "Self", "Production"),     # what he can do (forward-compat)
)

# The 4 rel tables that reference the spine node (Concept→Engram migration must
# drop+recreate these; COMPILED_FROM is untouched — Production→ActionChain).
_SPINE_REL_TABLES: tuple[tuple[str, str, str], ...] = (
    ("COMPOSED_FROM", "Engram", "Engram"),
    ("COMPOSED_INTO", "Engram", "Engram"),
    ("USES_SKILL", "Engram", "Production"),
    ("EXPLORES", "HypothesisFork", "Engram"),
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
    """Create the 5 node tables + 7 rel tables on the given Kuzu graph.

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


# ── Phase B — Concept → Engram copy-migration (RFP §7.B) ────────────

def _count_nodes(conn, table: str) -> int:
    try:
        qr = conn.execute(f"MATCH (n:{table}) RETURN COUNT(n)")
        if qr.has_next():
            return int(qr.get_next()[0])
    except Exception:
        pass
    return 0


def _snapshot_kuzu_dir(db_path: Optional[str]) -> Optional[str]:
    """One-time pre-migration insurance copy of the Kuzu store (memory-
    preservation — a destructive DROP follows). Best-effort: a failed snapshot
    logs + returns None but does NOT abort (the copy-then-verify-before-drop
    ordering is the primary node-data safety)."""
    import shutil
    try:
        if not db_path or not os.path.exists(db_path):
            return None
        dst = f"{db_path}.bak_pre_engram_{int(time.time())}"
        if os.path.isdir(db_path):
            shutil.copytree(db_path, dst)
        else:
            shutil.copy2(db_path, dst)
        logger.info("[engram_migration] pre-migration snapshot → %s", dst)
        return dst
    except Exception as e:
        logger.warning(
            "[engram_migration] snapshot failed (proceeding — verify-before-drop "
            "still protects node data): %s", e)
        return None


# (rel, MATCH-pattern, from-key-expr, to-key-expr) for capture; and the
# (src_table, src_key, dst_table, dst_key) for re-insert onto the Engram rels.
_SPINE_EDGE_CAPTURE: tuple = (
    ("COMPOSED_FROM", "(a:Concept)-[:COMPOSED_FROM]->(b:Concept)", "a.pk", "b.pk"),
    ("COMPOSED_INTO", "(a:Concept)-[:COMPOSED_INTO]->(b:Concept)", "a.pk", "b.pk"),
    ("USES_SKILL", "(a:Concept)-[:USES_SKILL]->(b:Production)", "a.pk", "b.skill_id"),
    ("EXPLORES", "(a:HypothesisFork)-[:EXPLORES]->(b:Concept)", "a.fork_id", "b.pk"),
)
_SPINE_EDGE_REINSERT: dict = {
    "COMPOSED_FROM": ("Engram", "pk", "Engram", "pk"),
    "COMPOSED_INTO": ("Engram", "pk", "Engram", "pk"),
    "USES_SKILL": ("Engram", "pk", "Production", "skill_id"),
    "EXPLORES": ("HypothesisFork", "fork_id", "Engram", "pk"),
}


def migrate_concept_to_engram(graph: Any) -> dict:
    """Copy-migrate a legacy `Concept` spine to `Engram` (RFP §7.B). Kuzu 0.11
    cannot rename a node table, so we copy rows + edges, then drop the old
    tables. Idempotent + guarded; **data-safe by ordering** — `Engram` is fully
    populated + count-verified BEFORE anything is dropped.

    Runs in `TitanKnowledgeGraph._init_schema()` immediately before
    `bootstrap_spine_schema`, single-threaded at graph-open (INV-Syn-7/28 — no
    writer thread is live yet). Both kuzu files run it: `synthesis_spine.kuzu`
    (real rows) + `knowledge_graph.kuzu` (empty Concept → trivial 0-row).
    """
    out: dict = {"migrated": False, "reason": None, "nodes": 0, "edges": 0,
                 "snapshot": None}
    conn = getattr(graph, "_conn", None)
    if conn is None:
        out["reason"] = "no_conn"
        return out

    # Guard 1 — already migrated (re-run no-op).
    if _table_exists(conn, "Engram"):
        out["reason"] = "engram_exists"
        return out
    # Guard 2 — fresh install (no legacy Concept) → bootstrap makes Engram.
    if not _table_exists(conn, "Concept"):
        out["reason"] = "no_concept_fresh"
        return out

    # 1. Capture all Concept rows (the 8 legacy columns).
    rows: list = []
    try:
        qr = conn.execute(
            "MATCH (c:Concept) RETURN c.pk, c.concept_id, c.version, c.name, "
            "c.memory_type, c.groundedness, c.anchor_tx, c.created_at"
        )
        while qr.has_next():
            rows.append(qr.get_next())
    except Exception as e:
        logger.error("[engram_migration] Concept capture failed — abort: %s", e)
        out["reason"] = f"capture_failed:{e}"
        return out

    # 2. Capture all spine edges (key pairs) for the 4 spine rel tables.
    captured_edges: dict = {}
    for rel, pattern, fk, tk in _SPINE_EDGE_CAPTURE:
        pairs: list = []
        try:
            qr = conn.execute(f"MATCH {pattern} RETURN {fk}, {tk}")
            while qr.has_next():
                pairs.append(qr.get_next())
        except Exception as e:
            logger.warning("[engram_migration] edge capture %s failed: %s", rel, e)
        captured_edges[rel] = pairs
    total_edges = sum(len(v) for v in captured_edges.values())

    # 3. Snapshot ONLY when there is data to lose (the empty knowledge_graph.kuzu
    #    Concept table is a trivial 0-row migration — no big shared-graph copy).
    if rows:
        out["snapshot"] = _snapshot_kuzu_dir(getattr(graph, "_db_path", None))

    # 4. CREATE Engram (Phase-B schema, with axes) + copy rows (axes 0.0,
    #    domain_hint NULL).
    engram_schema = dict(_NODE_TABLES)["Engram"]
    try:
        conn.execute(f"CREATE NODE TABLE Engram({engram_schema})")
    except Exception as e:
        if "already exists" not in str(e).lower() and "binder" not in str(e).lower():
            logger.error("[engram_migration] CREATE Engram failed — abort: %s", e)
            out["reason"] = f"create_engram_failed:{e}"
            return out
    for r in rows:
        try:
            conn.execute(
                "CREATE (e:Engram {pk:$pk, concept_id:$cid, version:$ver, "
                "name:$name, memory_type:$mt, groundedness:$g, anchor_tx:$tx, "
                "created_at:$ca, axis_used:0.0, axis_verified:0.0, "
                "axis_felt:0.0, axis_fluent:0.0})",
                {"pk": r[0], "cid": r[1], "ver": r[2], "name": r[3],
                 "mt": r[4], "g": r[5], "tx": r[6], "ca": r[7]},
            )
        except Exception as e:
            logger.error("[engram_migration] row copy failed (%s) — abort + drop "
                         "partial Engram (Concept intact): %s", r[0], e)
            try:
                conn.execute("DROP TABLE Engram")
            except Exception:
                pass
            out["reason"] = f"copy_failed:{e}"
            return out

    # 5. VERIFY node count (G2) BEFORE any destructive drop.
    migrated = _count_nodes(conn, "Engram")
    if migrated != len(rows):
        logger.error("[engram_migration] count mismatch Engram=%d != Concept=%d "
                     "— abort + drop partial Engram (Concept intact)",
                     migrated, len(rows))
        try:
            conn.execute("DROP TABLE Engram")
        except Exception:
            pass
        out["reason"] = f"count_mismatch:{migrated}!={len(rows)}"
        return out

    # 6. DROP the 4 spine rels, then DROP Concept (rels FIRST — Kuzu refuses to
    #    drop a node table still referenced by a rel table).
    for rel in ("COMPOSED_FROM", "COMPOSED_INTO", "USES_SKILL", "EXPLORES"):
        try:
            conn.execute(f"DROP TABLE {rel}")
        except Exception as e:
            logger.warning("[engram_migration] DROP rel %s: %s", rel, e)
    try:
        conn.execute("DROP TABLE Concept")
    except Exception as e:
        logger.error("[engram_migration] DROP Concept failed (Engram populated; "
                     "next-boot guard will see Engram): %s", e)
        out["reason"] = f"drop_concept_failed:{e}"
        return out

    # 7. CREATE the 4 Engram rels + re-insert captured edges.
    for rel, src, dst in _SPINE_REL_TABLES:
        try:
            conn.execute(f"CREATE REL TABLE {rel}(FROM {src} TO {dst})")
        except Exception as e:
            if "already exists" not in str(e).lower() and "binder" not in str(e).lower():
                logger.warning("[engram_migration] CREATE rel %s: %s", rel, e)
    reinserted = 0
    for rel, pairs in captured_edges.items():
        src_t, src_k, dst_t, dst_k = _SPINE_EDGE_REINSERT[rel]
        for fk_v, tk_v in pairs:
            try:
                conn.execute(
                    f"MATCH (a:{src_t} {{{src_k}: $av}}), "
                    f"(b:{dst_t} {{{dst_k}: $bv}}) CREATE (a)-[:{rel}]->(b)",
                    {"av": fk_v, "bv": tk_v},
                )
                reinserted += 1
            except Exception as e:
                logger.warning("[engram_migration] edge reinsert %s failed: %s", rel, e)

    out.update({"migrated": True, "reason": "ok", "nodes": migrated,
                "edges": reinserted})
    logger.info("[engram_migration] Concept→Engram migrated: %d nodes, %d/%d edges, "
                "snapshot=%s", migrated, reinserted, total_edges, out["snapshot"])
    return out


__all__ = ("bootstrap_spine_schema", "migrate_concept_to_engram")
