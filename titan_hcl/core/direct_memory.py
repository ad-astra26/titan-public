"""
core/direct_memory.py
Direct memory backend — replaces Cognee with FAISS + Kuzu + DuckDB.

Components:
  - TitanDuckDB: Unified metadata store (replaces SQLite memory_nodes.db)
  - TitanVectorIndex: FAISS flat index for semantic search (replaces LanceDB)
  - TitanKnowledgeGraph: Kuzu direct graph (replaces Cognee middleware)
  - TitanCognify: Custom Trinity-typed entity extraction (replaces cognee.cognify())
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from titan_hcl.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DuckDB Manager
# ---------------------------------------------------------------------------

class TitanDuckDB:
    """Unified metadata store for memory nodes. Replaces SQLite memory_nodes.db."""

    def __init__(self, db_path: str):
        import duckdb
        self._path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path, config={
            "memory_limit": "256MB",   # Cap in-memory buffers (was unbounded)
            "threads": "4",            # Limit thread pool
        })
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id INTEGER PRIMARY KEY,
                user_prompt TEXT,
                agent_response TEXT,
                source_id TEXT,
                status TEXT DEFAULT 'mempool',
                score REAL DEFAULT 0,
                on_chain_tx TEXT,
                base_weight REAL DEFAULT 1.0,
                anchor_bonus REAL DEFAULT 0.0,
                reinforcement_count INTEGER DEFAULT 0,
                emotional_intensity INTEGER DEFAULT 0,
                mempool_weight REAL DEFAULT 1.0,
                mempool_reinforcements INTEGER DEFAULT 0,
                effective_weight REAL DEFAULT 1.0,
                created_at DOUBLE,
                last_accessed DOUBLE,
                last_reinforced DOUBLE,
                embedding_id INTEGER DEFAULT -1,
                cognified BOOLEAN DEFAULT FALSE,
                neuromod_context TEXT,
                memory_type TEXT DEFAULT 'episodic'
            )
        """)
        # Synthesis Engine Phase 1 / D-SPEC-123 — additive memory_type column
        # for existing installs (CREATE TABLE above already carries it for
        # new installs). Idempotent: DuckDB silent-skips ADD COLUMN IF NOT
        # EXISTS when the column is already present.
        self._conn.execute(
            "ALTER TABLE memory_nodes "
            "ADD COLUMN IF NOT EXISTS memory_type TEXT DEFAULT 'episodic'"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS identity_nodes (
                id TEXT PRIMARY KEY,
                identifier TEXT,
                created_at DOUBLE
            )
        """)
        # Synthesis Engine Phase 1 / D-SPEC-123 — activation_state DDL
        # RELOCATED 2026-05-23 to data/synthesis.duckdb (owned by
        # synthesis_worker per G21 / INV-Syn-3). It lived here briefly
        # in the substrate commit (b1a04736) but DuckDB v0.8+ rejects two
        # R/W connections to one file across processes, so memory_worker
        # (this DB's R/W owner) and synthesis_worker can't share the
        # file — synthesis_worker now owns its own file. memory_worker
        # reads activation_state R/O via _in_process_activation_lookup
        # opening data/synthesis.duckdb in read_only mode.

    def insert_node(self, node: dict) -> int:
        """Insert a memory node. Returns node id."""
        cols = [
            "id", "user_prompt", "agent_response", "source_id", "status",
            "score", "on_chain_tx", "base_weight", "anchor_bonus",
            "reinforcement_count", "emotional_intensity", "mempool_weight",
            "mempool_reinforcements", "effective_weight", "created_at",
            "last_accessed", "last_reinforced", "embedding_id", "cognified",
            "neuromod_context",
        ]
        vals = []
        for c in cols:
            v = node.get(c)
            if c == "neuromod_context" and isinstance(v, dict):
                v = json.dumps(v)
            vals.append(v)
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        self._conn.execute(
            f"INSERT OR REPLACE INTO memory_nodes ({col_str}) VALUES ({placeholders})",
            vals,
        )
        return node.get("id", 0)

    def update_node(self, node_id: int, **fields):
        """Update specific fields on a node."""
        if not fields:
            return
        sets = []
        vals = []
        for k, v in fields.items():
            if k == "neuromod_context" and isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(node_id)
        self._conn.execute(
            f"UPDATE memory_nodes SET {', '.join(sets)} WHERE id = ?",
            vals,
        )

    def get_node(self, node_id: int) -> Optional[dict]:
        result = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?", [node_id]
        ).fetchone()
        if not result:
            return None
        cols = [d[0] for d in self._conn.description]
        return dict(zip(cols, result))

    def get_all_nodes(self) -> List[dict]:
        """Load all memory nodes."""
        result = self._conn.execute("SELECT * FROM memory_nodes").fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row)) for row in result]

    def get_nodes_by_status(self, status: str) -> List[dict]:
        result = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE status = ?", [status]
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row)) for row in result]

    def insert_identity(self, node_id: str, identifier: str, created_at: float):
        self._conn.execute(
            "INSERT OR IGNORE INTO identity_nodes (id, identifier, created_at) VALUES (?, ?, ?)",
            [node_id, identifier, created_at],
        )

    def get_all_identities(self) -> List[dict]:
        result = self._conn.execute("SELECT * FROM identity_nodes").fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row)) for row in result]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
        by_status = self._conn.execute(
            "SELECT status, COUNT(*) FROM memory_nodes GROUP BY status"
        ).fetchall()
        return {
            "total": total,
            "by_status": {s: c for s, c in by_status},
        }

    def export_parquet(self, path: str):
        """Export memory_nodes to Parquet for Arweave backup."""
        self._conn.execute(
            f"COPY memory_nodes TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        logger.info("[DuckDB] Exported to %s", path)

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# 2. FAISS Vector Index
# ---------------------------------------------------------------------------

class TitanVectorIndex:
    """FAISS flat index for semantic search. Lazy-loads Fastembed model."""

    def __init__(self, index_path: str, dim: int = 384):
        self._index_path = index_path
        self._id_map_path = index_path + ".idmap.json"
        self._dim = dim
        self._model = None  # Lazy-loaded
        self._index = None
        self._id_map: List[int] = []  # FAISS position → node_id
        self._load()

    def _load(self):
        """Load existing FAISS index and id map from disk.

        Defends against 0-byte / truncated index files (observed 2026-04-14
        on T2+T3 when disk filled to 100% mid-save — FAISS write succeeded
        creating the file, then ran out of space before writing any
        content, leaving a 0-byte file that blocked memory worker boot in
        a crash loop). A size-zero file on disk is semantically equivalent
        to no file at all, but faiss.read_index() throws a RuntimeError
        instead of handling it. We detect and recover here: move the
        corrupt file aside for forensics, then initialize a fresh index."""
        import faiss
        needs_fresh = True
        if os.path.exists(self._index_path):
            try:
                sz = os.path.getsize(self._index_path)
            except OSError:
                sz = 0
            if sz > 0:
                try:
                    self._index = faiss.read_index(self._index_path)
                    if os.path.exists(self._id_map_path):
                        with open(self._id_map_path, "r") as f:
                            self._id_map = json.load(f)
                    logger.info(
                        "[VectorIndex] Loaded %d vectors from %s",
                        self._index.ntotal, self._index_path,
                    )
                    needs_fresh = False
                except Exception as e:
                    quarantine = f"{self._index_path}.corrupt_{int(os.path.getmtime(self._index_path))}"
                    try:
                        os.rename(self._index_path, quarantine)
                        if os.path.exists(self._id_map_path):
                            os.rename(self._id_map_path, quarantine + ".idmap.json")
                    except OSError:
                        pass
                    logger.warning(
                        "[VectorIndex] Corrupt index at %s (%s) — quarantined to %s, initializing fresh",
                        self._index_path, e, quarantine,
                    )
            else:
                try:
                    quarantine = f"{self._index_path}.corrupt_empty_{int(os.path.getmtime(self._index_path))}"
                    os.rename(self._index_path, quarantine)
                except OSError:
                    pass
                logger.warning(
                    "[VectorIndex] Empty (0-byte) index at %s — disk-full corruption suspected; initializing fresh",
                    self._index_path,
                )
        if needs_fresh:
            self._index = faiss.IndexFlatIP(self._dim)
            self._id_map = []
            logger.info("[VectorIndex] Created new empty index (dim=%d)", self._dim)

    def _ensure_model(self):
        """Lazy-bind the fleet-standard llama.cpp embedder singleton on first use
        (Phase 13 §3J.1 — shared bge-small, torch-free)."""
        if self._model is not None:
            return
        from titan_hcl.utils.text_embedder import get_text_embedder
        self._model = get_text_embedder()
        logger.info("[VectorIndex] llama.cpp embedder bound (lazy)")

    def embed(self, text: str) -> np.ndarray:
        """Embed text to 384D normalized vector (singleton normalizes already)."""
        self._ensure_model()
        return np.asarray(self._model.encode(text), dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts at once (singleton normalizes each row already)."""
        self._ensure_model()
        return np.asarray(self._model.encode(texts), dtype=np.float32)

    def add(self, embedding: np.ndarray, node_id: int):
        """Add a single vector to the index."""
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._id_map.append(node_id)

    def add_batch(self, embeddings: np.ndarray, node_ids: List[int]):
        """Add multiple vectors at once."""
        vecs = embeddings.astype(np.float32)
        self._index.add(vecs)
        self._id_map.extend(node_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for nearest neighbors. Returns [(node_id, score), ...]."""
        if self._index.ntotal == 0:
            return []
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def save(self):
        """Persist index and id map to disk."""
        import faiss
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._id_map_path, "w") as f:
            json.dump(self._id_map, f)
        logger.info("[VectorIndex] Saved %d vectors to %s", self._index.ntotal, self._index_path)

    @property
    def count(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# 3. Kuzu Knowledge Graph
# ---------------------------------------------------------------------------

class TitanKnowledgeGraph:
    """Direct Kuzu graph with Trinity-typed + Universal entity schema."""

    def __init__(self, db_path: str):
        import kuzu
        self._db_path = db_path
        # Kuzu 0.11+ creates its own file/directory — ensure parent exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self):
        """Create node and relationship tables if they don't exist."""
        node_tables = {
            # Universal entities
            "Person": "name STRING, user_id STRING, first_seen DOUBLE, last_seen DOUBLE, interaction_count INT64, PRIMARY KEY(name)",
            "Topic": "name STRING, domain STRING, first_seen DOUBLE, relevance DOUBLE, PRIMARY KEY(name)",
            "Media": "name STRING, media_type STRING, source_node INT64, created_at DOUBLE, PRIMARY KEY(name)",
            # Trinity entities
            "BodyEntity": "name STRING, subtype STRING, source_node INT64, created_at DOUBLE, PRIMARY KEY(name)",
            "MindEntity": "name STRING, subtype STRING, source_node INT64, created_at DOUBLE, PRIMARY KEY(name)",
            "SpiritEntity": "name STRING, subtype STRING, source_node INT64, created_at DOUBLE, PRIMARY KEY(name)",
        }

        for table_name, schema in node_tables.items():
            try:
                self._conn.execute(f"CREATE NODE TABLE {table_name}({schema})")
                logger.info("[KnowledgeGraph] Created table %s", table_name)
            except Exception:
                pass  # Table already exists

        try:
            from titan_hcl.logic.social_x.schema_migrations import (
                apply_kuzu_person_migrations,
            )
            apply_kuzu_person_migrations(self)
        except Exception as exc:
            logger.warning(
                "[KnowledgeGraph] X-voice Person migration skipped: %s", exc
            )

        # Phase 4 — synthesis-engine Concept-spine schema (§6.1 / §10).
        # 4 node tables + 5 rel tables, additive + idempotent. Empty
        # Production/ActionChain/HypothesisFork ship in P4 so consumers
        # can issue Cypher without a schema-missing branch; population
        # lands in P5/P8.
        try:
            from titan_hcl.synthesis.kuzu_spine_schema import (
                bootstrap_spine_schema,
            )
            bootstrap_spine_schema(self)
        except Exception as exc:
            logger.warning(
                "[KnowledgeGraph] synthesis spine bootstrap skipped: %s", exc
            )

        # Relationship tables — we use a generic rel table per node-type pair
        # to keep schema manageable. Kuzu requires explicit FROM/TO types.
        all_tables = list(node_tables.keys())
        for src in all_tables:
            for dst in all_tables:
                rel_name = f"REL_{src}_{dst}"
                try:
                    self._conn.execute(
                        f"CREATE REL TABLE {rel_name}("
                        f"FROM {src} TO {dst}, "
                        f"rel_type STRING, da DOUBLE, serotonin DOUBLE, "
                        f"emotion STRING, source_node INT64, created_at DOUBLE)"
                    )
                except Exception:
                    pass  # Already exists

    def _get_table_for_type(self, entity_type: str) -> str:
        """Map entity type/subtype to Kuzu node table."""
        type_map = {
            # Universal
            "person": "Person",
            "topic": "Topic",
            "media": "Media",
            "organization": "Topic",  # Organizations stored as Topics with domain="org"
            "location": "Topic",      # Locations stored as Topics with domain="location"
            # Body
            "sensation": "BodyEntity",
            "action": "BodyEntity",
            "state": "BodyEntity",
            # Mind
            "concept": "MindEntity",
            "decision": "MindEntity",
            "pattern": "MindEntity",
            # Spirit
            "identity": "SpiritEntity",
            "relationship": "SpiritEntity",
            "purpose": "SpiritEntity",
        }
        return type_map.get(entity_type.lower(), "Topic")

    def add_entity(
        self, name: str, entity_type: str, source_node: int = 0,
        attributes: dict = None,
    ):
        """Add or merge an entity into the graph."""
        table = self._get_table_for_type(entity_type)
        now = time.time()
        subtype = entity_type.lower()

        try:
            if table == "Person":
                attrs = attributes or {}
                user_id = attrs.get("user_id", "")
                # rFP_x_voice_enrichment §4.5: capture felt-state-at-last-interaction
                # for OUTER_RUMINATION Pool B prompt template.
                neuromods = attrs.get("neuromods")
                emotion = attrs.get("emotion", "")
                last_felt_summary = ""
                last_felt_neuromods_json = "{}"
                if neuromods or emotion:
                    try:
                        from titan_hcl.logic.social_x.felt_state import (
                            compact_felt_summary, neuromods_to_json,
                        )
                        last_felt_summary = compact_felt_summary(neuromods, emotion)
                        last_felt_neuromods_json = neuromods_to_json(neuromods)
                    except Exception:
                        pass
                self._conn.execute(
                    f"MERGE (p:{table} {{name: $name}}) "
                    f"ON CREATE SET p.user_id = $uid, p.first_seen = $ts, "
                    f"p.last_seen = $ts, p.interaction_count = 1, "
                    f"p.last_felt_emotion = $emo, "
                    f"p.last_felt_summary = $sum, "
                    f"p.last_felt_neuromods_json = $njs "
                    f"ON MATCH SET p.last_seen = $ts, "
                    f"p.interaction_count = p.interaction_count + 1, "
                    f"p.last_felt_emotion = $emo, "
                    f"p.last_felt_summary = $sum, "
                    f"p.last_felt_neuromods_json = $njs",
                    {"name": name, "uid": user_id, "ts": now,
                     "emo": emotion, "sum": last_felt_summary,
                     "njs": last_felt_neuromods_json},
                )
            elif table == "Topic":
                domain = subtype if subtype in ("organization", "location") else "general"
                self._conn.execute(
                    f"MERGE (t:{table} {{name: $name}}) "
                    f"ON CREATE SET t.domain = $domain, t.first_seen = $ts, t.relevance = 1.0 "
                    f"ON MATCH SET t.relevance = t.relevance + 0.1",
                    {"name": name, "domain": domain, "ts": now},
                )
            elif table == "Media":
                media_type = (attributes or {}).get("media_type", "unknown")
                self._conn.execute(
                    f"MERGE (m:{table} {{name: $name}}) "
                    f"ON CREATE SET m.media_type = $mt, m.source_node = $sn, m.created_at = $ts",
                    {"name": name, "mt": media_type, "sn": source_node, "ts": now},
                )
            else:
                # Trinity entities (Body/Mind/Spirit)
                self._conn.execute(
                    f"MERGE (e:{table} {{name: $name}}) "
                    f"ON CREATE SET e.subtype = $st, e.source_node = $sn, e.created_at = $ts",
                    {"name": name, "st": subtype, "sn": source_node, "ts": now},
                )
        except Exception as e:
            swallow_warn(f"[KnowledgeGraph] Entity insert error for '{name}'", e,
                         key="core.direct_memory.entity_insert_error_for", throttle=100)

    def add_relationship(
        self, src_name: str, src_type: str, dst_name: str, dst_type: str,
        rel_type: str, neuromod_context: dict = None, source_node: int = 0,
    ):
        """Add a relationship between two entities."""
        src_table = self._get_table_for_type(src_type)
        dst_table = self._get_table_for_type(dst_type)
        rel_table = f"REL_{src_table}_{dst_table}"
        ctx = neuromod_context or {}
        now = time.time()

        try:
            self._conn.execute(
                f"MATCH (a:{src_table} {{name: $src}}), (b:{dst_table} {{name: $dst}}) "
                f"CREATE (a)-[:{rel_table} {{rel_type: $rt, da: $da, serotonin: $st, "
                f"emotion: $em, source_node: $sn, created_at: $ts}}]->(b)",
                {
                    "src": src_name, "dst": dst_name, "rt": rel_type,
                    "da": ctx.get("DA", 0.5), "st": ctx.get("5-HT", 0.5),
                    "em": ctx.get("emotion", "neutral"),
                    "sn": source_node, "ts": now,
                },
            )
        except Exception as e:
            logger.debug("[KnowledgeGraph] Relationship error '%s'->'%s': %s", src_name, dst_name, e)

    def search_entities(self, query: str, limit: int = 10) -> List[dict]:
        """Search entities by name substring across all tables."""
        results = []
        for table in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity", "Media"]:
            try:
                qr = self._conn.execute(
                    f"MATCH (e:{table}) WHERE e.name CONTAINS $q RETURN e.name, '{table}' "
                    f"LIMIT $lim",
                    {"q": query, "lim": limit},
                )
                while qr.has_next():
                    row = qr.get_next()
                    results.append({"name": row[0], "table": row[1]})
            except Exception:
                pass
        return results[:limit]

    def traverse(self, entity_name: str, depth: int = 2) -> List[dict]:
        """Traverse relationships from an entity up to given depth."""
        results = []
        for src_table in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity"]:
            for dst_table in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity"]:
                rel_table = f"REL_{src_table}_{dst_table}"
                try:
                    qr = self._conn.execute(
                        f"MATCH (a:{src_table} {{name: $name}})-[r:{rel_table}]->(b:{dst_table}) "
                        f"RETURN a.name, r.rel_type, b.name LIMIT 20",
                        {"name": entity_name},
                    )
                    while qr.has_next():
                        row = qr.get_next()
                        results.append({
                            "src": row[0],
                            "rel": row[1],
                            "dst": row[2],
                        })
                except Exception:
                    pass
        return results

    def get_stats(self) -> dict:
        counts = {}
        for table in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity", "Media"]:
            try:
                result = self._conn.execute(f"MATCH (e:{table}) RETURN COUNT(e)").get_next()
                counts[table] = int(result[0]) if result else 0
            except Exception:
                counts[table] = 0
        return counts

    def export_json(self, path: str):
        """Export all entities and relationships to JSON for Arweave backup."""
        data = {"entities": [], "relationships": []}
        for table in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity", "Media"]:
            try:
                df = self._conn.execute(f"MATCH (e:{table}) RETURN e.*").get_as_df()
                for _, row in df.iterrows():
                    data["entities"].append({"table": table, **row.to_dict()})
            except Exception:
                pass
        # Export relationships
        for src_t in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity"]:
            for dst_t in ["Person", "Topic", "BodyEntity", "MindEntity", "SpiritEntity"]:
                rel = f"REL_{src_t}_{dst_t}"
                try:
                    df = self._conn.execute(
                        f"MATCH (a:{src_t})-[r:{rel}]->(b:{dst_t}) "
                        f"RETURN a.name, r.rel_type, b.name"
                    ).get_as_df()
                    for _, row in df.iterrows():
                        data["relationships"].append({
                            "src": row.iloc[0], "rel": row.iloc[1], "dst": row.iloc[2],
                        })
                except Exception:
                    pass
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("[KnowledgeGraph] Exported %d entities, %d rels to %s",
                     len(data["entities"]), len(data["relationships"]), path)

    # ─── Phase 4 — synthesis-engine Concept-spine helpers (§6.1 / §10) ───
    #
    # Low-level Cypher wrappers consumed by `titan_hcl/synthesis/concept_store.py`
    # (the sole writer per INV-Syn-3 extended). High-level invariants
    # (INV-3 no parent mutation, INV-4 single canonical write path, INV-10
    # parent must exist) live in ConceptStore; these helpers are intentionally
    # primitive so they're also safe for read-only consumers (BridgeRecall +
    # observatory endpoints).

    @staticmethod
    def _spine_pk(concept_id: str, version: int) -> str:
        """Synthetic single-column PK for the Concept node table — needed
        because Kuzu 0.11 does not support composite PRIMARY KEY. The format
        is `<concept_id>:v<version>`; the API surface still takes (concept_id,
        version) so callers never see this implementation detail."""
        return f"{concept_id}:v{int(version)}"

    def spine_create_concept_node(
        self, concept_id: str, version: int, name: str, memory_type: str,
        groundedness: float, anchor_tx: str, created_at: float,
    ) -> bool:
        """INSERT one Concept row keyed by synthetic pk = `<id>:v<ver>` so the
        same concept_id can carry multiple versions (§10 versioning invariant).

        Returns True on insert, False if the row already exists (idempotent —
        safe to call on replay). Raises only on unexpected Cypher errors.
        """
        pk = self._spine_pk(concept_id, version)
        try:
            self._conn.execute(
                "CREATE (c:Concept {pk: $pk, concept_id: $cid, version: $ver, "
                "name: $name, memory_type: $mt, groundedness: $g, "
                "anchor_tx: $atx, created_at: $ts})",
                {"pk": pk, "cid": concept_id, "ver": int(version),
                 "name": name, "mt": memory_type,
                 "g": float(groundedness), "atx": anchor_tx,
                 "ts": float(created_at)},
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if (
                "primary key" in msg or "duplicate" in msg
                or "constraint" in msg or "violates" in msg
            ):
                return False
            logger.warning(
                "[KnowledgeGraph] spine_create_concept_node(%s,v%d) failed: %s",
                concept_id, version, e,
            )
            raise

    def spine_get_concept_version(
        self, concept_id: str, version: int,
    ) -> dict | None:
        """Return the row dict for (concept_id, version) or None if missing."""
        pk = self._spine_pk(concept_id, version)
        try:
            qr = self._conn.execute(
                "MATCH (c:Concept {pk: $pk}) "
                "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                "c.groundedness, c.anchor_tx, c.created_at",
                {"pk": pk},
            )
            if not qr.has_next():
                return None
            row = qr.get_next()
            return {
                "concept_id": row[0], "version": int(row[1]), "name": row[2],
                "memory_type": row[3], "groundedness": float(row[4]),
                "anchor_tx": row[5], "created_at": float(row[6]),
            }
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] spine_get_concept_version(%s,v%d) failed: %s",
                concept_id, version, e,
            )
            return None

    def spine_get_latest_concept(self, concept_id: str) -> dict | None:
        """Return the highest-version row for concept_id, or None if no
        version exists. Used by ConceptStore.bump_version() to compute v+1
        and by spine recall (P4.H) to pick the latest spine root."""
        try:
            qr = self._conn.execute(
                "MATCH (c:Concept {concept_id: $cid}) "
                "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                "c.groundedness, c.anchor_tx, c.created_at "
                "ORDER BY c.version DESC LIMIT 1",
                {"cid": concept_id},
            )
            if not qr.has_next():
                return None
            row = qr.get_next()
            return {
                "concept_id": row[0], "version": int(row[1]), "name": row[2],
                "memory_type": row[3], "groundedness": float(row[4]),
                "anchor_tx": row[5], "created_at": float(row[6]),
            }
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] spine_get_latest_concept(%s) failed: %s",
                concept_id, e,
            )
            return None

    def spine_update_groundedness(
        self, concept_id: str, version: int, new_groundedness: float,
    ) -> bool:
        """UPDATE one Concept row's groundedness column. Allowed by INV-3
        because groundedness is a *derived metric column*, not the row's
        identity / version / lineage — it can be recomputed at any time
        without violating the immutability of the version itself. Returns
        True on update, False if the row is missing."""
        pk = self._spine_pk(concept_id, version)
        try:
            # Verify row exists first (Kuzu MATCH+SET silently succeeds with
            # zero rows; we want a definite signal).
            if self.spine_get_concept_version(concept_id, version) is None:
                return False
            self._conn.execute(
                "MATCH (c:Concept {pk: $pk}) SET c.groundedness = $g",
                {"pk": pk, "g": float(new_groundedness)},
            )
            return True
        except Exception as e:
            logger.warning(
                "[KnowledgeGraph] spine_update_groundedness(%s,v%d) failed: %s",
                concept_id, version, e,
            )
            return False

    def spine_add_composition_edge(
        self,
        from_concept_id: str, from_version: int,
        to_concept_id: str, to_version: int,
        direction: str = "from",
    ) -> bool:
        """Create a COMPOSED_FROM or COMPOSED_INTO edge between two Concept
        rows. `direction="from"` means the FROM-node was composed FROM the
        TO-node (decompile / down); `direction="into"` means the FROM-node
        composes INTO the TO-node (recompile / up). Per §10, both directions
        are maintained when a version bump consumes base concepts.

        Both endpoints must already exist; returns False if either is missing
        or the edge already exists. Idempotent on duplicate edge attempts.
        """
        rel = "COMPOSED_FROM" if direction == "from" else "COMPOSED_INTO"
        if direction not in ("from", "into"):
            logger.warning(
                "[KnowledgeGraph] spine_add_composition_edge: bad direction %r",
                direction,
            )
            return False
        # Existence check — Kuzu's CREATE on a missing MATCH silently no-ops,
        # so we verify endpoints first for a definitive signal.
        if self.spine_get_concept_version(
            from_concept_id, from_version,
        ) is None:
            return False
        if self.spine_get_concept_version(
            to_concept_id, to_version,
        ) is None:
            return False
        from_pk = self._spine_pk(from_concept_id, from_version)
        to_pk = self._spine_pk(to_concept_id, to_version)
        try:
            self._conn.execute(
                f"MATCH (a:Concept {{pk: $apk}}), (b:Concept {{pk: $bpk}}) "
                f"CREATE (a)-[:{rel}]->(b)",
                {"apk": from_pk, "bpk": to_pk},
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" in msg or "constraint" in msg:
                return False
            logger.warning(
                "[KnowledgeGraph] spine_add_composition_edge(%s v%d -[%s]-> "
                "%s v%d) failed: %s",
                from_concept_id, from_version, rel,
                to_concept_id, to_version, e,
            )
            return False

    def spine_count_concepts(self) -> int:
        """Total Concept rows across all (concept_id, version) tuples.
        Used by the fleet E2E test P4.kuzu-spine-active check (§P4.J)."""
        try:
            qr = self._conn.execute("MATCH (c:Concept) RETURN COUNT(c)")
            if qr.has_next():
                return int(qr.get_next()[0])
        except Exception as e:
            logger.debug("[KnowledgeGraph] spine_count_concepts failed: %s", e)
        return 0

    def spine_count_composition_edges(self, direction: str = "from") -> int:
        """Total COMPOSED_FROM or COMPOSED_INTO edges across the graph.
        Used by the fleet E2E test P4.composition-edges-present check."""
        rel = "COMPOSED_FROM" if direction == "from" else "COMPOSED_INTO"
        try:
            qr = self._conn.execute(
                f"MATCH ()-[r:{rel}]->() RETURN COUNT(r)"
            )
            if qr.has_next():
                return int(qr.get_next()[0])
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] spine_count_composition_edges(%s) failed: %s",
                direction, e,
            )
        return 0

    def spine_concept_neighbors(
        self, concept_id: str, version: int | None = None,
        limit: int = 20,
    ) -> list[tuple[str, int]]:
        """Return up-to-`limit` neighbor concept (id, version) tuples reachable
        via COMPOSED_FROM or COMPOSED_INTO from the given concept. If `version`
        is None, anchors on the latest version (typical spreading-activation
        usage from §P4.F kuzu_spreading_lookup).

        The spreading-activation formula `S - ln(fan_j)` needs the fan-out of
        each buffer-entity concept; the consumer counts the returned tuples to
        get `fan_j`. Both COMPOSED_FROM and COMPOSED_INTO are walked because
        spreading should pick up sibling concepts in either direction.
        """
        anchor_version = version
        if anchor_version is None:
            latest = self.spine_get_latest_concept(concept_id)
            if latest is None:
                return []
            anchor_version = latest["version"]

        seen: list[tuple[str, int]] = []
        anchor_pk = self._spine_pk(concept_id, anchor_version)
        for rel in ("COMPOSED_FROM", "COMPOSED_INTO"):
            try:
                qr = self._conn.execute(
                    f"MATCH (a:Concept {{pk: $apk}})-[:{rel}]->(b:Concept) "
                    f"RETURN b.concept_id, b.version LIMIT $lim",
                    {"apk": anchor_pk, "lim": int(limit)},
                )
                while qr.has_next():
                    row = qr.get_next()
                    pair = (row[0], int(row[1]))
                    if pair not in seen:
                        seen.append(pair)
            except Exception as e:
                logger.debug(
                    "[KnowledgeGraph] spine_concept_neighbors(%s,v%d,%s) failed: %s",
                    concept_id, anchor_version, rel, e,
                )
        return seen[:limit]

    def spine_list_concepts(
        self, limit: int = 100, offset: int = 0,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Paginated list of concepts (latest version per concept_id), ordered
        by groundedness DESC. Backs the Observatory /v6/synthesis/concepts
        endpoint (§P4.I).

        Kuzu's Cypher dialect doesn't provide MAX+GROUP BY in 0.11, so we
        fetch all rows + collapse to latest-per-concept_id in Python. Cheap
        even at fleet scale because total concept-version count stays bounded
        by the dream-boundary cap in §P4.G (max_concepts_per_pass).
        """
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
            logger.debug("[KnowledgeGraph] spine_list_concepts failed: %s", e)
            return []

        # Collapse to latest-version-per-concept_id.
        latest: dict[str, dict] = {}
        for r in all_rows:
            existing = latest.get(r["concept_id"])
            if existing is None or r["version"] > existing["version"]:
                latest[r["concept_id"]] = r

        ordered = sorted(
            latest.values(),
            key=lambda r: r.get("groundedness", 0.0),
            reverse=True,
        )
        return ordered[offset : offset + limit]

    # ── Phase 5 — HypothesisFork node + EXPLORES edge helpers ─────────
    #
    # The HypothesisFork table + EXPLORES rel were declared in Phase 4
    # (`kuzu_spine_schema.py`) but empty. Phase 5 populates them via the
    # primitives below. All write ops are funnelled through HypothesisForkStore
    # (INV-Syn-8); these are the bare Kuzu helpers it calls.

    def fork_create_node(
        self, fork_id: str, root_anchor: str, activation: float, status: str,
    ) -> bool:
        """INSERT one HypothesisFork row. Returns True on insert, False on
        duplicate. `root_anchor` is the empty string ("") for net-new forks
        (Kuzu STRING is non-nullable by Phase 4 DDL; we use "" to mean ∅)."""
        try:
            self._conn.execute(
                "CREATE (f:HypothesisFork {fork_id: $fid, root_anchor: $ra, "
                "activation: $a, status: $s})",
                {"fid": fork_id, "ra": root_anchor or "",
                 "a": float(activation), "s": status},
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if (
                "primary key" in msg or "duplicate" in msg
                or "constraint" in msg or "violates" in msg
            ):
                return False
            logger.warning(
                "[KnowledgeGraph] fork_create_node(%s) failed: %s",
                fork_id, e,
            )
            raise

    def fork_get_node(self, fork_id: str) -> dict | None:
        """Return row dict for fork_id or None."""
        try:
            qr = self._conn.execute(
                "MATCH (f:HypothesisFork {fork_id: $fid}) "
                "RETURN f.fork_id, f.root_anchor, f.activation, f.status",
                {"fid": fork_id},
            )
            if not qr.has_next():
                return None
            row = qr.get_next()
            return {
                "fork_id": row[0],
                "root_anchor": row[1] or None,
                "activation": float(row[2]),
                "status": row[3],
            }
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] fork_get_node(%s) failed: %s", fork_id, e,
            )
            return None

    def fork_update_status(
        self, fork_id: str, status: str, activation: float | None = None,
    ) -> bool:
        """Update mutable columns on a HypothesisFork row. `status` always
        updated; `activation` optional. Returns True on update, False if the
        row is missing.

        Allowed by INV-Syn-8: HypothesisFork is a *probationary index row*
        (not canonical chain data) — its status + activation are derived
        metrics that may be updated by the sole writer. INV-3 says canonical
        data is never deleted; fork-node rows are NOT canonical (only their
        tombstone TXs on graduation/abandonment are).
        """
        if self.fork_get_node(fork_id) is None:
            return False
        try:
            if activation is not None:
                self._conn.execute(
                    "MATCH (f:HypothesisFork {fork_id: $fid}) "
                    "SET f.status = $s, f.activation = $a",
                    {"fid": fork_id, "s": status, "a": float(activation)},
                )
            else:
                self._conn.execute(
                    "MATCH (f:HypothesisFork {fork_id: $fid}) SET f.status = $s",
                    {"fid": fork_id, "s": status},
                )
            return True
        except Exception as e:
            logger.warning(
                "[KnowledgeGraph] fork_update_status(%s,%s) failed: %s",
                fork_id, status, e,
            )
            return False

    def fork_delete_node(self, fork_id: str) -> bool:
        """DETACH DELETE a HypothesisFork row + all its incident EXPLORES
        edges. Used by the cascade-GC sweep after the lifecycle ends
        (graduated or abandoned) — the row is no longer needed as a hot
        index; the canonical record lives in the chain (graduation TX or
        tombstone TX). Returns True if a row was deleted."""
        if self.fork_get_node(fork_id) is None:
            return False
        try:
            self._conn.execute(
                "MATCH (f:HypothesisFork {fork_id: $fid}) DETACH DELETE f",
                {"fid": fork_id},
            )
            return True
        except Exception as e:
            logger.warning(
                "[KnowledgeGraph] fork_delete_node(%s) failed: %s", fork_id, e,
            )
            return False

    def fork_add_explores_edge(
        self, fork_id: str, concept_id: str, version: int,
    ) -> bool:
        """Create an EXPLORES edge from a HypothesisFork to a Concept row.
        Both endpoints must exist. Returns False if either is missing or the
        edge already exists (idempotent on duplicates)."""
        if self.fork_get_node(fork_id) is None:
            return False
        if self.spine_get_concept_version(concept_id, version) is None:
            return False
        concept_pk = self._spine_pk(concept_id, version)
        try:
            self._conn.execute(
                "MATCH (f:HypothesisFork {fork_id: $fid}), "
                "(c:Concept {pk: $cpk}) "
                "CREATE (f)-[:EXPLORES]->(c)",
                {"fid": fork_id, "cpk": concept_pk},
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" in msg or "constraint" in msg:
                return False
            logger.warning(
                "[KnowledgeGraph] fork_add_explores_edge(%s → %s v%d) failed: %s",
                fork_id, concept_id, version, e,
            )
            return False

    def fork_list_all(self, status: str | None = None) -> list[dict]:
        """Return all HypothesisFork rows, optionally filtered by status.
        Cheap because the table is GC-bounded (graduated/abandoned rows are
        DETACH DELETE'd by the nightly sweep)."""
        try:
            if status is not None:
                qr = self._conn.execute(
                    "MATCH (f:HypothesisFork) WHERE f.status = $s "
                    "RETURN f.fork_id, f.root_anchor, f.activation, f.status",
                    {"s": status},
                )
            else:
                qr = self._conn.execute(
                    "MATCH (f:HypothesisFork) "
                    "RETURN f.fork_id, f.root_anchor, f.activation, f.status"
                )
            out: list[dict] = []
            while qr.has_next():
                row = qr.get_next()
                out.append({
                    "fork_id": row[0],
                    "root_anchor": row[1] or None,
                    "activation": float(row[2]),
                    "status": row[3],
                })
            return out
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] fork_list_all(status=%r) failed: %s",
                status, e,
            )
            return []

    def fork_count(self, status: str | None = None) -> int:
        """Count HypothesisFork rows, optionally by status. Cheap; used by
        the fleet E2E test §P5.J.1 check + Observatory metrics."""
        try:
            if status is not None:
                qr = self._conn.execute(
                    "MATCH (f:HypothesisFork) WHERE f.status = $s "
                    "RETURN COUNT(f)",
                    {"s": status},
                )
            else:
                qr = self._conn.execute(
                    "MATCH (f:HypothesisFork) RETURN COUNT(f)"
                )
            if qr.has_next():
                return int(qr.get_next()[0])
        except Exception as e:
            logger.debug("[KnowledgeGraph] fork_count failed: %s", e)
        return 0

    def fork_explores_targets(self, fork_id: str) -> list[tuple[str, int]]:
        """Return the (concept_id, version) tuples the given fork EXPLORES.
        Used by the cascade-GC predicate's "sole-inbound" check and by
        repair-fork graduation to resolve the parent concept."""
        try:
            qr = self._conn.execute(
                "MATCH (f:HypothesisFork {fork_id: $fid})-[:EXPLORES]->(c:Concept) "
                "RETURN c.concept_id, c.version",
                {"fid": fork_id},
            )
            out: list[tuple[str, int]] = []
            while qr.has_next():
                row = qr.get_next()
                out.append((row[0], int(row[1])))
            return out
        except Exception as e:
            logger.debug(
                "[KnowledgeGraph] fork_explores_targets(%s) failed: %s",
                fork_id, e,
            )
            return []

    def close(self):
        del self._conn
        del self._db


# ---------------------------------------------------------------------------
# 4. Custom Cognify — Trinity-typed Entity Extraction
# ---------------------------------------------------------------------------

# Entity extraction prompt
_COGNIFY_SYSTEM = """You are an entity extraction system for a sovereign AI agent called Titan.
Extract entities and relationships from the given memory text.

Entity types:
  UNIVERSAL: person, topic, organization, location, media
  BODY (physical/sensory): sensation, action, state
  MIND (cognitive/analytical): concept, decision, pattern
  SPIRIT (identity/relational): identity, relationship, purpose

Relationship types: causes, correlates_with, part_of, precedes, contradicts, enriches, discusses, created_by

Return ONLY valid JSON:
{"entities": [{"name": "...", "type": "..."}], "relationships": [{"src": "...", "dst": "...", "type": "..."}]}

Rules:
- Entity names should be lowercase, concise (1-3 words)
- Extract the user's identifier as a "person" entity when present
- Extract topics discussed as "topic" entities
- Extract emotional/body states as appropriate Trinity types
- Keep total entities under 10 per memory
- Only create relationships between entities you extracted"""


class TitanCognify:
    """Custom entity extraction using LLM, optimized for Trinity architecture.

    Phase 3 Chunk χ (D-SPEC-88, 2026-05-18): direct OllamaCloudProvider
    construction REMOVED. Entity-extraction LLM calls now route through
    POST /v4/llm-distill so all LLM traffic appears in llm_state.bin.
    The `llm_client` param is preserved as None-default for back-compat
    with any test that constructs the engine inline; production wiring
    in TieredMemoryGraph supplies api_base + internal_key only.
    """

    def __init__(self, llm_client=None, graph: TitanKnowledgeGraph = None,
                 *, api_base: str = "http://127.0.0.1:7777",
                 internal_key: str = ""):
        """
        Args:
            llm_client: legacy param kept for back-compat (ignored; cognify
                        now uses HTTP /v4/llm-distill instead).
            graph: TitanKnowledgeGraph instance.
            api_base: Titan API root (e.g. http://127.0.0.1:7777).
            internal_key: X-Titan-Internal-Key for /v4/llm-distill auth.
        """
        self._llm = llm_client  # legacy / unused — kept so callers don't break
        self._graph = graph
        self._model = "gemma4:31b"  # Fast, efficient for extraction
        self._api_base = api_base.rstrip("/")
        self._internal_key = internal_key

    async def cognify_node(
        self, node_id: int, text: str, neuromod_context: dict = None,
    ) -> List[dict]:
        """
        Extract Trinity-typed entities from a single memory node.
        Incremental: ~3-5s per node via Ollama Cloud (now routed through
        /v4/llm-distill → llm_worker → Ollama Cloud).
        """
        if not self._internal_key:
            return []

        prompt = f"Memory text:\n{text[:2000]}"  # Cap at 2000 chars

        try:
            from titan_hcl.logic.llm_distill_client import (
                distill_via_http_async)
            response = await distill_via_http_async(
                text=prompt,
                instruction=_COGNIFY_SYSTEM,
                api_base=self._api_base,
                internal_key=self._internal_key,
                model=self._model,
                max_tokens=500,
                temperature=0.1,
                consumer="cognify_entity_extraction",
                timeout_s=30.0,
            )
            if not response:
                return []
            entities, relationships = self._parse_response(response)
        except Exception as e:
            logger.warning("[Cognify] LLM call failed for node %d: %s", node_id, e)
            return []

        # Insert entities into graph
        for ent in entities:
            self._graph.add_entity(
                name=ent["name"],
                entity_type=ent["type"],
                source_node=node_id,
            )

        # Insert relationships
        for rel in relationships:
            src_type = self._find_entity_type(rel["src"], entities)
            dst_type = self._find_entity_type(rel["dst"], entities)
            if src_type and dst_type:
                self._graph.add_relationship(
                    src_name=rel["src"], src_type=src_type,
                    dst_name=rel["dst"], dst_type=dst_type,
                    rel_type=rel["type"],
                    neuromod_context=neuromod_context,
                    source_node=node_id,
                )

        logger.debug(
            "[Cognify] Node %d: %d entities, %d relationships",
            node_id, len(entities), len(relationships),
        )
        return entities

    def _parse_response(self, response: str) -> Tuple[List[dict], List[dict]]:
        """Parse LLM JSON response into entities and relationships."""
        entities = []
        relationships = []

        # Find JSON in response (may have surrounding text)
        try:
            # Try direct parse first
            data = json.loads(response)
        except json.JSONDecodeError:
            # Find JSON block in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(response[start:end])
                except json.JSONDecodeError:
                    return [], []
            else:
                return [], []

        for ent in data.get("entities", []):
            if isinstance(ent, dict) and "name" in ent and "type" in ent:
                entities.append({
                    "name": str(ent["name"]).lower().strip(),
                    "type": str(ent["type"]).lower().strip(),
                })

        for rel in data.get("relationships", []):
            if isinstance(rel, dict) and "src" in rel and "dst" in rel and "type" in rel:
                relationships.append({
                    "src": str(rel["src"]).lower().strip(),
                    "dst": str(rel["dst"]).lower().strip(),
                    "type": str(rel["type"]).lower().strip(),
                })

        return entities, relationships

    def _find_entity_type(self, name: str, entities: List[dict]) -> Optional[str]:
        """Find the type of an entity by name from the extracted list."""
        name_lower = name.lower().strip()
        for ent in entities:
            if ent["name"] == name_lower:
                return ent["type"]
        return None
