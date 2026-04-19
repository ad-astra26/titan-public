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
                neuromod_context TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS identity_nodes (
                id TEXT PRIMARY KEY,
                identifier TEXT,
                created_at DOUBLE
            )
        """)

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
        """Lazy-load Fastembed model on first use."""
        if self._model is not None:
            return
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("[VectorIndex] Fastembed model loaded (lazy)")

    def embed(self, text: str) -> np.ndarray:
        """Embed text to 384D normalized vector."""
        self._ensure_model()
        embeddings = list(self._model.embed([text]))
        vec = np.array(embeddings[0], dtype=np.float32)
        # Normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts at once for efficiency."""
        self._ensure_model()
        embeddings = list(self._model.embed(texts))
        vecs = np.array(embeddings, dtype=np.float32)
        # Normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        return vecs

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
                user_id = (attributes or {}).get("user_id", "")
                self._conn.execute(
                    f"MERGE (p:{table} {{name: $name}}) "
                    f"ON CREATE SET p.user_id = $uid, p.first_seen = $ts, "
                    f"p.last_seen = $ts, p.interaction_count = 1 "
                    f"ON MATCH SET p.last_seen = $ts, "
                    f"p.interaction_count = p.interaction_count + 1",
                    {"name": name, "uid": user_id, "ts": now},
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
            logger.debug("[KnowledgeGraph] Entity insert error for '%s': %s", name, e)

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
    """Custom entity extraction using LLM, optimized for Trinity architecture."""

    def __init__(self, llm_client, graph: TitanKnowledgeGraph):
        """
        Args:
            llm_client: OllamaCloudClient instance (or None for disabled cognify)
            graph: TitanKnowledgeGraph instance
        """
        self._llm = llm_client
        self._graph = graph
        self._model = "gemma4:31b"  # Fast, efficient for extraction

    async def cognify_node(
        self, node_id: int, text: str, neuromod_context: dict = None,
    ) -> List[dict]:
        """
        Extract Trinity-typed entities from a single memory node.
        Incremental: ~3-5s per node via Ollama Cloud.
        """
        if self._llm is None:
            return []

        prompt = f"Memory text:\n{text[:2000]}"  # Cap at 2000 chars

        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model,
                system=_COGNIFY_SYSTEM,
                temperature=0.1,
                max_tokens=500,
                timeout=30.0,
            )
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
