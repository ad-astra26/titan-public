"""
core/memory.py
Cognee-backed Tiered Persistence with Digital Neuroplasticity.
V2.1: Gradual Forgetting — sigmoid mempool decay, per-node scoring, lightweight
      mempool embeddings for pre-meditation semantic recall.

Architecture:
  - _node_store (dict): Local metadata index for all nodes (weights, decay, status).
  - Cognee (lancedb + kuzu): Semantic search + knowledge graph for persistent memories.
  - Mempool: Sigmoid-decayed local nodes with lightweight embedding index for semantic recall.
  - System nodes (pulses, social metrics, identity): Local-only (structured, not semantic).

Mempool Lifecycle (v2.1):
  - Nodes enter with mempool_weight=1.0
  - Sigmoid decay: w(t) = 1 / (1 + e^(k*(t - t_half))) where t_half=12h, k=0.4/hr
  - Reinforcement: matching topics reset decay clock + boost weight
  - Meditation: per-node scoring, threshold promotion (>=40), keep (weight>0.3), prune (<0.1)
  - Hard TTL: 24h maximum regardless of reinforcement
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Mempool decay constants
_MEMPOOL_HALF_LIFE_HOURS = 12.0
_MEMPOOL_DECAY_K = 0.4  # steepness: sigmoid drops from ~0.88 at 6h to ~0.12 at 18h
_MEMPOOL_MAX_TTL_HOURS = 24.0
_MEMPOOL_PRUNE_THRESHOLD = 0.1  # below this weight → prune on next meditation
_MEMPOOL_KEEP_THRESHOLD = 0.3  # between prune and this → keep but don't promote
_MEMPOOL_PROMOTE_SCORE = 40.0  # LLM score threshold for promotion (was 50 batch-avg)


class TieredMemoryGraph:
    """
    Manages the memory graph system, implementing tiered persistence (mempool vs persistent)
    and digital neuroplasticity features like forgetting curves and reinforcement.

    Persistent memories are stored in Cognee for semantic graph search.
    Mempool and system nodes remain in a local index for fast access.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Dict combining [inference] + [memory_and_storage] sections from config.toml.
        """
        config = config or {}
        self._config = config

        # Data directory
        data_dir = config.get("data_dir", "./data")

        # Local metadata index — all node metadata lives here (in-memory cache)
        self._node_store: Dict = {}
        self._next_id = 1

        # Lightweight mempool embedding index (fastembed, in-memory)
        # NOTE: Must be initialized BEFORE _load_node_store (which calls _index_mempool_node)
        self._mempool_embeddings: Dict[int, np.ndarray] = {}
        self._embedding_model = None  # Lazy-loaded fastembed model
        self._embedding_dim = 384  # BAAI/bge-small-en-v1.5

        # --- Direct Memory Backend (replaces Cognee) ---
        from titan_plugin.core.direct_memory import (
            TitanDuckDB, TitanVectorIndex, TitanKnowledgeGraph, TitanCognify,
        )
        self._duckdb = TitanDuckDB(os.path.join(data_dir, "titan_memory.duckdb"))
        self._vectors = TitanVectorIndex(os.path.join(data_dir, "memory_vectors.faiss"))
        self._graph = TitanKnowledgeGraph(os.path.join(data_dir, "knowledge_graph.kuzu"))

        # LLM client for custom cognify (entity extraction)
        ollama_cloud_key = config.get("ollama_cloud_api_key", "")
        ollama_cloud_url = config.get("ollama_cloud_base_url", "https://ollama.com/v1")
        self._llm_client = None
        if ollama_cloud_key:
            try:
                from titan_plugin.utils.ollama_cloud import OllamaCloudClient
                self._llm_client = OllamaCloudClient(
                    api_key=ollama_cloud_key, base_url=ollama_cloud_url,
                )
            except Exception as e:
                logger.warning("[Memory] OllamaCloud client init failed: %s", e)
        self._cognify_engine = TitanCognify(self._llm_client, self._graph)

        # Backward compat: callers check this flag
        self._cognee_ready = True  # Direct backend is always "ready"

        # Load all nodes from DuckDB into in-memory cache
        self._load_node_store()

        # ZK batch queue — memory hashes pending on-chain compression
        self._zk_queue: List[bytes] = []
        self._zk_queue_path = os.path.join(data_dir, "zk_queue", "pending.json")
        self._zk_queue_path = os.path.normpath(self._zk_queue_path)
        self._load_zk_queue()

    # -------------------------------------------------------------------------
    # Backward-compat alias: code referencing _mock_db still works
    # -------------------------------------------------------------------------
    @property
    def _mock_db(self) -> Dict:
        return self._node_store

    @_mock_db.setter
    def _mock_db(self, value: Dict):
        self._node_store = value

    # -------------------------------------------------------------------------
    # Node Store Persistence (SQLite — survives restarts)
    # -------------------------------------------------------------------------
    def _load_node_store(self) -> None:
        """Load all nodes from DuckDB into _node_store on boot."""
        try:
            # Load memory nodes (skip pruned)
            rows = self._duckdb.get_all_nodes()
            for node in rows:
                if node.get("status") == "pruned":
                    continue
                node["type"] = "MemoryNode"  # DuckDB doesn't store type column
                node_id = node["id"]
                self._node_store[node_id] = node
                if node_id >= self._next_id:
                    self._next_id = node_id + 1
                # Rebuild mempool embedding index for active mempool nodes
                if node.get("status") == "mempool":
                    self._index_mempool_node(node)

            # Load identity nodes
            id_rows = self._duckdb.get_all_identities()
            for node in id_rows:
                node["type"] = "IdentityNode"
                self._node_store[node["id"]] = node

            mem_count = sum(1 for v in self._node_store.values() if v.get("type") == "MemoryNode")
            persistent = sum(1 for v in self._node_store.values()
                           if v.get("type") == "MemoryNode" and v.get("status") == "persistent")
            mempool = mem_count - persistent
            logger.info(
                "[Memory] Loaded %d nodes from DuckDB (%d persistent, %d mempool, %d identities, "
                "FAISS: %d vectors, Kuzu: %s).",
                len(self._node_store), persistent, mempool,
                sum(1 for v in self._node_store.values() if v.get("type") == "IdentityNode"),
                self._vectors.count,
                self._graph.get_stats(),
            )
        except Exception as e:
            logger.warning("[Memory] Failed to load node store: %s. Starting fresh.", e)

    def _persist_node(self, node: Dict) -> None:
        """Upsert a single MemoryNode to DuckDB."""
        if node.get("type") == "IdentityNode":
            self._persist_identity_node(node)
            return
        if node.get("type") != "MemoryNode":
            return
        try:
            self._duckdb.insert_node(node)
        except Exception as e:
            logger.warning("[Memory] Failed to persist node %s: %s", node.get("id"), e)

    def _persist_identity_node(self, node: Dict) -> None:
        """Upsert an IdentityNode to DuckDB."""
        try:
            self._duckdb.insert_identity(
                node.get("id", ""),
                node.get("identifier", ""),
                node.get("created_at", 0),
            )
        except Exception as e:
            logger.warning("[Memory] Failed to persist identity node %s: %s", node.get("id"), e)

    # -------------------------------------------------------------------------
    # Backend Ready Check (replaces _ensure_cognee)
    # -------------------------------------------------------------------------
    async def _ensure_cognee(self) -> bool:
        """Backward-compat: direct backend is always ready."""
        return True

    # -------------------------------------------------------------------------
    # Mempool Embedding Index (lightweight, in-memory, fastembed)
    # -------------------------------------------------------------------------
    def _ensure_embedding_model(self):
        """Lazy-load the fastembed model for mempool semantic search."""
        if self._embedding_model is not None:
            return True
        try:
            from fastembed import TextEmbedding
            self._embedding_model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir=os.path.join(self._config.get("data_dir", "./data"), ".fastembed_cache"),
            )
            logger.info("[Memory] Fastembed model loaded for mempool semantic index.")
            return True
        except Exception as e:
            logger.debug("[Memory] Fastembed unavailable: %s — mempool keyword-only.", e)
            return False

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a single text string."""
        if not self._ensure_embedding_model():
            return None
        try:
            embeddings = list(self._embedding_model.embed([text]))
            if embeddings:
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.debug("[Memory] Embedding failed: %s", e)
        return None

    def _index_mempool_node(self, node: Dict):
        """Add a mempool node's embedding to the in-memory index."""
        text = f"{node.get('user_prompt', '')} {node.get('agent_response', '')}"
        emb = self._embed_text(text)
        if emb is not None:
            self._mempool_embeddings[node["id"]] = emb

    def _remove_mempool_embedding(self, node_id: int):
        """Remove a node's embedding from the index."""
        self._mempool_embeddings.pop(node_id, None)

    def _mempool_semantic_search(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search over mempool embeddings using cosine similarity.
        Returns top-k matching mempool nodes sorted by similarity.
        """
        if not self._mempool_embeddings:
            return []
        query_emb = self._embed_text(prompt)
        if query_emb is None:
            return []

        scored = []
        for node_id, emb in self._mempool_embeddings.items():
            if node_id not in self._node_store:
                continue
            node = self._node_store[node_id]
            if node.get("status") != "mempool":
                continue
            # Cosine similarity
            dot = float(np.dot(query_emb, emb))
            norm = float(np.linalg.norm(query_emb) * np.linalg.norm(emb))
            sim = dot / norm if norm > 0 else 0.0
            if sim > 0.3:  # minimum relevance threshold
                scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    # -------------------------------------------------------------------------
    # Mempool Sigmoid Decay
    # -------------------------------------------------------------------------
    def _compute_mempool_weight(self, node: Dict) -> float:
        """
        Sigmoid decay for mempool nodes:
        w(t) = 1 / (1 + e^(k * (t_hours - t_half)))

        At t=0:  w ≈ 0.99
        At t=6h: w ≈ 0.88
        At t=12h: w = 0.50
        At t=18h: w ≈ 0.12
        At t=24h: w ≈ 0.01 (hard TTL kicks in)

        Reinforcement shifts the reference time forward (resets decay clock).
        """
        now = time.time()
        # Use last_reinforced if available, otherwise created_at
        ref_time = node.get("last_reinforced", node.get("created_at", now))
        elapsed_hours = (now - ref_time) / 3600.0

        # Hard TTL: 24h from creation regardless of reinforcement
        age_hours = (now - node.get("created_at", now)) / 3600.0
        if age_hours >= _MEMPOOL_MAX_TTL_HOURS:
            return 0.0

        # Sigmoid decay from last reinforcement
        base_weight = 1.0 / (1.0 + math.exp(_MEMPOOL_DECAY_K * (elapsed_hours - _MEMPOOL_HALF_LIFE_HOURS)))

        # Reinforcement bonus: +10% per hit, max +50%
        reinforcement_bonus = min(0.50, node.get("mempool_reinforcements", 0) * 0.10)

        return min(1.0, base_weight + reinforcement_bonus)

    def _apply_mempool_decay(self, node: Dict):
        """Update mempool_weight on a node using sigmoid decay."""
        node["mempool_weight"] = self._compute_mempool_weight(node)

    def reinforce_mempool_node(self, node_id: int):
        """
        Reinforce a mempool node — resets its decay clock and boosts weight.
        Called when the same topic comes up again before meditation.
        """
        if node_id not in self._node_store:
            return
        node = self._node_store[node_id]
        if node.get("status") != "mempool":
            return
        node["last_reinforced"] = time.time()
        node["mempool_reinforcements"] = node.get("mempool_reinforcements", 0) + 1
        self._apply_mempool_decay(node)
        self._persist_node(node)
        logger.debug(
            "[Memory] Mempool node %d reinforced (count=%d, weight=%.3f).",
            node_id, node["mempool_reinforcements"], node["mempool_weight"],
        )

    def find_similar_mempool_node(self, text: str, threshold: float = 0.6) -> Optional[int]:
        """
        Find a mempool node semantically similar to the given text.
        Returns node_id if found above threshold, None otherwise.
        Used for topic-based reinforcement during add_to_mempool.
        """
        if not self._mempool_embeddings:
            return None
        query_emb = self._embed_text(text)
        if query_emb is None:
            return None

        best_id = None
        best_sim = 0.0
        for node_id, emb in self._mempool_embeddings.items():
            if node_id not in self._node_store:
                continue
            if self._node_store[node_id].get("status") != "mempool":
                continue
            dot = float(np.dot(query_emb, emb))
            norm = float(np.linalg.norm(query_emb) * np.linalg.norm(emb))
            sim = dot / norm if norm > 0 else 0.0
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

        return best_id if best_sim >= threshold else None

    def get_mempool_stats(self) -> Dict:
        """Return mempool health statistics for Observatory/Info Banner."""
        mempool = [
            v for v in self._node_store.values()
            if v.get("type") == "MemoryNode" and v.get("status") == "mempool"
        ]
        if not mempool:
            return {"count": 0, "avg_weight": 0.0, "min_weight": 0.0, "max_weight": 0.0}

        for n in mempool:
            self._apply_mempool_decay(n)
        weights = [n.get("mempool_weight", 1.0) for n in mempool]
        return {
            "count": len(mempool),
            "avg_weight": sum(weights) / len(weights),
            "min_weight": min(weights),
            "max_weight": max(weights),
        }

    # -------------------------------------------------------------------------
    # Query — Semantic search via Cognee + local mempool scan
    # -------------------------------------------------------------------------
    async def query(self, prompt: str, top_k: int = 10) -> List[Dict]:
        """
        Fetch relevant memories for the pre-prompt.
        Three-layer search:
          1. Cognee semantic search (persistent memories)
          2. Mempool semantic search (lightweight embeddings, pre-meditation recall)
          3. Local keyword fallback (both tiers)
        """
        results = []
        now = time.time()

        # 0. Global decay tick — apply forgetting curve to all persistent nodes
        for v in self._node_store.values():
            if v.get("type") == "MemoryNode" and v.get("status") == "persistent":
                self._apply_decay(v)

        # 1. Semantic search via FAISS for persistent memories
        faiss_results = await self._cognee_search(prompt, top_k=top_k)
        if faiss_results:
            for node in faiss_results:
                self._apply_decay(node)
                node["last_accessed"] = now
                node["reinforcement_count"] += 1
                self._apply_decay(node)
                results.append(node)

        # 2. Mempool semantic search (pre-meditation recall via fastembed)
        mempool_hits = self._mempool_semantic_search(prompt, top_k=5)
        seen_ids = {r["id"] for r in results}
        for node in mempool_hits:
            if node["id"] not in seen_ids:
                self._apply_mempool_decay(node)
                if node.get("mempool_weight", 1.0) >= _MEMPOOL_PRUNE_THRESHOLD:
                    # Reinforce: this memory was recalled, it matters
                    self.reinforce_mempool_node(node["id"])
                    results.append(node)
                    seen_ids.add(node["id"])

        # 3. Local keyword fallback for anything not found above
        local_hits = self._local_keyword_search(prompt)
        for node in local_hits:
            if node["id"] not in seen_ids:
                if node["status"] == "persistent":
                    self._apply_decay(node)
                    node["last_accessed"] = now
                    node["reinforcement_count"] += 1
                    self._apply_decay(node)
                elif node["status"] == "mempool":
                    self._apply_mempool_decay(node)
                    if node.get("mempool_weight", 1.0) < _MEMPOOL_PRUNE_THRESHOLD:
                        continue
                    self.reinforce_mempool_node(node["id"])
                results.append(node)
                seen_ids.add(node["id"])

        return results

    async def query_user_memories(
        self, prompt: str, user_id: str, limit: int = 3
    ) -> List[Dict]:
        """
        Phase 13: Recall memories specific to a user.
        Searches both mempool and persistent nodes tagged with this user's identity.
        Returns the most relevant memories from past conversations with this user.
        """
        identity_node_id = f"identity_{user_id}"
        user_nodes = []

        for node in self._node_store.values():
            if node.get("type") != "MemoryNode":
                continue
            if node.get("source_id") != identity_node_id:
                continue
            # Apply decay for accurate weight
            if node.get("status") == "mempool":
                self._apply_mempool_decay(node)
                if node.get("mempool_weight", 0) < _MEMPOOL_PRUNE_THRESHOLD:
                    continue
            elif node.get("status") == "persistent":
                self._apply_decay(node)

            user_nodes.append(node)

        if not user_nodes:
            return []

        # Score by keyword relevance to current prompt
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())
        stopwords = {"the", "a", "an", "is", "it", "in", "to", "and", "of", "for", "do", "you", "my"}
        meaningful_prompt = prompt_words - stopwords

        scored = []
        for node in user_nodes:
            content = f"{node.get('user_prompt', '')} {node.get('agent_response', '')}".lower()
            content_words = set(content.split()) - stopwords
            overlap = meaningful_prompt & content_words
            score = len(overlap) + node.get("effective_weight", 1.0) * 0.1
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    async def _cognee_search(self, prompt: str, top_k: int = 10) -> list:
        """
        FAISS vector search for persistent memories (replaces Cognee search).
        Returns list of node dicts matching the query semantically.
        """
        if self._vectors.count == 0:
            return []

        try:
            query_vec = self._vectors.embed(prompt)
            hits = self._vectors.search(query_vec, top_k=top_k)
            results = []
            for node_id, score in hits:
                node = self._node_store.get(node_id)
                if node and node.get("status") == "persistent":
                    results.append(node)
            if results:
                logger.info("[Memory] FAISS search: %d results for '%s...'",
                            len(results), prompt[:40])
            return results
        except Exception as e:
            logger.debug("[Memory] FAISS search failed: %s", e)
            return []

    async def graph_completion_search(self, prompt: str, top_k: int = 3) -> list:
        """
        Kuzu graph traversal — find related entities and relationships.
        No LLM calls needed (unlike Cognee's GRAPH_COMPLETION).
        """
        try:
            # Search entities matching the query
            entities = self._graph.search_entities(prompt, limit=top_k)
            results = []
            for ent in entities:
                # Traverse relationships from each matched entity
                rels = self._graph.traverse(ent["name"], depth=2)
                if rels:
                    results.append({
                        "entity": ent["name"],
                        "table": ent.get("table", ""),
                        "relationships": rels,
                    })
            if results:
                logger.info("[Memory] Graph search: %d entity matches for '%s...'",
                            len(results), prompt[:50])
            return results
        except Exception as e:
            logger.debug("[Memory] Graph search failed: %s", e)
            return []

    def _match_cognee_result(self, result_text: str) -> Dict | None:
        """Match a search result back to a node. FAISS returns nodes directly now."""
        # With FAISS, _cognee_search already returns node dicts, so this is rarely needed.
        # Kept for backward compat with any callers that still pass text.
        if isinstance(result_text, dict):
            return result_text
        for node in self._node_store.values():
            if node.get("type") != "MemoryNode" or node.get("status") != "persistent":
                continue
            if (
                node.get("user_prompt", "") in result_text
                or node.get("agent_response", "") in result_text
            ):
                return node
        return None

    def _local_keyword_search(self, prompt: str) -> List[Dict]:
        """Fallback keyword search over local node store."""
        results = []
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())

        for v in self._node_store.values():
            if v.get("type") != "MemoryNode":
                continue
            if v.get("status") not in ("mempool", "persistent"):
                continue

            # Score by word overlap between prompt and stored content
            content = f"{v.get('user_prompt', '')} {v.get('agent_response', '')}".lower()
            content_words = set(content.split())
            overlap = prompt_words & content_words
            # Require at least one meaningful word match (skip stopwords)
            stopwords = {"the", "a", "an", "is", "it", "in", "to", "and", "of", "for", "do", "you", "my"}
            meaningful = overlap - stopwords
            if meaningful:
                results.append(v)

        return results

    # -------------------------------------------------------------------------
    # Mempool Operations (local-only, fast)
    # -------------------------------------------------------------------------
    async def add_to_mempool(
        self, user_prompt: str, agent_response: str, user_identifier: str = "Anonymous"
    ) -> None:
        """
        Add a fresh node to the mempool with sigmoid decay tracking.
        If a semantically similar topic already exists in the mempool,
        reinforces the existing node instead of creating a duplicate.
        """
        now = time.time()

        # Ensure IdentityNode exists
        identity_node_id = f"identity_{user_identifier}"
        if identity_node_id not in self._node_store:
            id_node = {
                "id": identity_node_id,
                "type": "IdentityNode",
                "identifier": user_identifier,
                "created_at": now,
            }
            self._node_store[identity_node_id] = id_node
            self._persist_node(id_node)

        # Check for similar existing mempool node → reinforce instead of duplicate
        combined_text = f"{user_prompt} {agent_response}"
        similar_id = self.find_similar_mempool_node(combined_text, threshold=0.7)
        if similar_id is not None:
            self.reinforce_mempool_node(similar_id)
            # Append the new response context to the existing node
            existing = self._node_store[similar_id]
            existing["agent_response"] = f"{existing.get('agent_response', '')} | {agent_response}"
            # Re-index with updated content
            self._index_mempool_node(existing)
            self._persist_node(existing)
            logger.debug("[Memory] Reinforced existing mempool node %d (similar topic).", similar_id)
            return

        node_id = self._next_id
        node = {
            "id": node_id,
            "type": "MemoryNode",
            "user_prompt": user_prompt,
            "agent_response": agent_response,
            "source_id": identity_node_id,
            "status": "mempool",
            "score": 0,
            "on_chain_tx": None,
            "base_weight": 1.0,
            "anchor_bonus": 0.0,
            "reinforcement_count": 0,
            "emotional_intensity": 0,
            "created_at": now,
            "last_accessed": now,
            "last_reinforced": now,
            "mempool_reinforcements": 0,
            "mempool_weight": 1.0,
            "effective_weight": 1.0,
        }
        self._node_store[node_id] = node
        self._next_id += 1

        # Index in lightweight embedding store for semantic recall
        self._index_mempool_node(node)

        # Persist to SQLite
        self._persist_node(node)

    async def fetch_mempool(self) -> List[Dict]:
        """Retrieves all memory nodes currently residing in the mempool."""
        return [v for v in self._node_store.values() if v.get("status") == "mempool"]

    async def fetch_mempool_classified(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Classify mempool nodes by sigmoid weight into three buckets:
          - candidates: weight >= KEEP_THRESHOLD (eligible for LLM scoring + promotion)
          - fading: PRUNE_THRESHOLD <= weight < KEEP_THRESHOLD (keep, re-evaluate next epoch)
          - dead: weight < PRUNE_THRESHOLD or age > 24h (prune immediately)
        """
        candidates, fading, dead = [], [], []
        for v in self._node_store.values():
            if v.get("type") != "MemoryNode" or v.get("status") != "mempool":
                continue
            self._apply_mempool_decay(v)
            w = v.get("mempool_weight", 1.0)
            if w < _MEMPOOL_PRUNE_THRESHOLD:
                dead.append(v)
            elif w < _MEMPOOL_KEEP_THRESHOLD:
                fading.append(v)
            else:
                candidates.append(v)
        return candidates, fading, dead

    async def prune_mempool_node(self, node_id: int) -> None:
        """Marks a mempool node as pruned and removes its embedding. Additive — never deletes from DB."""
        if node_id in self._node_store and self._node_store[node_id].get("status") == "mempool":
            node = self._node_store[node_id]
            node["status"] = "pruned"
            self._persist_node(node)
            del self._node_store[node_id]
            self._remove_mempool_embedding(node_id)

    # -------------------------------------------------------------------------
    # Persistence — Migrate mempool → persistent (+ Cognee ingest)
    # -------------------------------------------------------------------------
    async def migrate_to_persistent(
        self, node_id: int, on_chain_tx: str, emotional_intensity: int
    ) -> None:
        """
        Meditation Epoch action: promote a mempool node to persistent.
        Updates local metadata, removes mempool embedding, ingests into Cognee.
        """
        if node_id not in self._node_store:
            return

        node = self._node_store[node_id]
        node["status"] = "persistent"
        node["on_chain_tx"] = on_chain_tx

        # Clean up mempool-specific fields and embedding
        self._remove_mempool_embedding(node_id)
        node.pop("mempool_weight", None)
        node.pop("mempool_reinforcements", None)
        node.pop("last_reinforced", None)

        # Map 1-10 intensity to 5% to 25% anchor bonus
        bonus = 0.05 + ((emotional_intensity - 1) / 9.0) * 0.20
        node["emotional_intensity"] = emotional_intensity
        node["anchor_bonus"] = bonus
        node["last_accessed"] = time.time()
        self._apply_decay(node)

        # Persist updated status to SQLite
        self._persist_node(node)

        # Ingest into Cognee (async, non-blocking on failure)
        await self._cognee_ingest(node)

        # Queue memory hash for ZK compression during next meditation
        memory_hash = self._compute_memory_hash(node)
        self._queue_for_compression(memory_hash)

    async def _cognee_ingest(self, node: Dict) -> None:
        """Ingest a persistent memory: embed into FAISS + cognify into Kuzu graph."""
        try:
            # Format memory as structured text
            text = (
                f"Memory #{node['id']} | "
                f"User: {node.get('user_prompt', '')} | "
                f"Agent: {node.get('agent_response', '')}"
            )
            # 1. Embed and add to FAISS vector index
            vec = self._vectors.embed(text)
            self._vectors.add(vec, node["id"])
            self._vectors.save()

            # 2. Run custom cognify: extract entities → Kuzu graph
            try:
                neuromod_ctx = node.get("neuromod_context", {})
                await self._cognify_engine.cognify_node(
                    node["id"], text, neuromod_ctx,
                )
                self._duckdb.update_node(node["id"], cognified=True)
            except Exception as e:
                logger.warning("[Memory] Cognify failed for node %s: %s", node["id"], e)

            logger.debug("[Memory] Ingested node %s (FAISS + Kuzu).", node["id"])
        except Exception as e:
            logger.warning("[Memory] Direct ingest failed for node %s: %s", node["id"], e)

    async def inject_memory(self, text: str, source: str = "maker", weight: float = 5.0,
                           neuromod_context: dict = None) -> dict:
        """
        Direct Memory Injection — bypasses mempool, creates a high-weight
        persistent node immediately and ingests into Cognee.

        Used by the Maker Console and Dream Bridge to inject memories directly
        into the Titan's persistent core without waiting for discovery.

        Args:
            text: The memory content to inject.
            source: Attribution label (default "maker").
            weight: Base weight for the memory (default 5.0, much higher than organic 1.0).
            neuromod_context: Optional felt state snapshot (neuromods + emotion) at injection time.
                              Stored as JSON in DuckDB. Used by Bridge B for recall perturbation.

        Returns:
            dict: The created memory node metadata.
        """
        now = time.time()
        node_id = self._next_id
        self._next_id += 1

        node = {
            "id": node_id,
            "type": "MemoryNode",
            "user_prompt": f"[{source.upper()}_INJECTION] {text}",
            "agent_response": text,
            "source_id": f"identity_{source}",
            "status": "persistent",
            "score": 100,
            "on_chain_tx": None,
            "base_weight": weight,
            "anchor_bonus": 0.25,  # Maximum anchor bonus
            "reinforcement_count": 0,
            "emotional_intensity": 10,  # Highest priority
            "created_at": now,
            "last_accessed": now,
            "effective_weight": weight + 0.25,
            "neuromod_context": neuromod_context,
        }
        self._node_store[node_id] = node
        self._persist_node(node)

        # Ingest into Cognee for semantic retrieval
        await self._cognee_ingest(node)

        logger.info("[Memory] Direct injection from %s: node %d, weight %.1f", source, node_id, weight)
        return {
            "node_id": node_id,
            "weight": weight + 0.25,
            "status": "persistent",
            "cognified": True,
        }

    # -------------------------------------------------------------------------
    # Consolidation — Batch cognify during meditation
    # -------------------------------------------------------------------------
    async def consolidate(self) -> bool:
        """
        With direct backend, cognify happens incrementally at ingest time.
        This method now just saves the FAISS index to disk (periodic persistence).
        Called during Meditation Epoch (every 6 hours).
        """
        try:
            self._vectors.save()
            logger.info("[Memory] Consolidation: FAISS index saved (%d vectors).",
                        self._vectors.count)
            return True
        except Exception as e:
            logger.warning("[Memory] Consolidation failed: %s", e)
            return False

    # -------------------------------------------------------------------------
    # ZK Batch Queue — Pending On-Chain Compression
    # -------------------------------------------------------------------------
    def _compute_memory_hash(self, node: Dict) -> bytes:
        """Compute a 32-byte SHA-256 hash of a memory node's content."""
        content = f"{node.get('user_prompt', '')}|{node.get('agent_response', '')}|{node.get('id', '')}"
        return hashlib.sha256(content.encode("utf-8")).digest()

    def _queue_for_compression(self, memory_hash: bytes):
        """Add a memory hash to the ZK batch queue after Cognee persistence."""
        self._zk_queue.append(memory_hash)
        self._persist_zk_queue()

    def drain_zk_queue(self) -> List[bytes]:
        """Drain and return all queued hashes (called during meditation)."""
        hashes = list(self._zk_queue)
        self._zk_queue.clear()
        self._persist_zk_queue()
        return hashes

    def _persist_zk_queue(self):
        """Write the ZK queue to disk for crash resilience."""
        try:
            os.makedirs(os.path.dirname(self._zk_queue_path), exist_ok=True)
            with open(self._zk_queue_path, "w") as f:
                json.dump([h.hex() for h in self._zk_queue], f)
        except Exception as e:
            logger.debug("[Memory] ZK queue persist failed: %s", e)

    def _load_zk_queue(self):
        """Load the ZK queue from disk on boot."""
        try:
            if os.path.exists(self._zk_queue_path):
                with open(self._zk_queue_path, "r") as f:
                    hex_list = json.load(f)
                self._zk_queue = [bytes.fromhex(h) for h in hex_list]
                if self._zk_queue:
                    logger.info(
                        "[Memory] Recovered %d pending ZK hashes from disk.",
                        len(self._zk_queue),
                    )
        except Exception as e:
            logger.debug("[Memory] ZK queue load failed: %s", e)
            self._zk_queue = []

    # -------------------------------------------------------------------------
    # Neuroplasticity — Decay + Reinforcement
    # -------------------------------------------------------------------------
    def _apply_decay(self, node: Dict):
        """
        Digital Neuroplasticity calculation:
        1. Curve of Forgetting: Exponential decay of semantic anchor bonus over time.
        2. Reinforcement Hardening: Adds bonus weight per query execution.
        """
        now = time.time()
        elapsed_seconds = now - node.get("last_accessed", now)
        elapsed_days = elapsed_seconds / 86400.0

        # Exponential decay: drops to ~5% of initial value after 180 days (k = 0.0166)
        decay_rate = 0.0166
        current_anchor_bonus = node.get("anchor_bonus", 0.0) * math.exp(
            -decay_rate * elapsed_days
        )

        # Reinforcement Hardening (+2% per hit, max 30%)
        reinforcement_bonus = min(0.30, node.get("reinforcement_count", 0) * 0.02)

        # Calculate Effective Weight
        node["effective_weight"] = (
            node.get("base_weight", 1.0) + current_anchor_bonus + reinforcement_bonus
        )

    def get_persistent_count(self) -> int:
        """Returns the count of persistent memory nodes."""
        return sum(
            1
            for v in self._node_store.values()
            if v.get("type") == "MemoryNode" and v.get("status") == "persistent"
        )

    def get_top_memories(self, n: int = 3) -> List[Dict]:
        """
        Return the top N persistent memories sorted by effective_weight.
        Used by the Omni-Voice synthesis to give the Titan contextual depth.
        """
        persistent = [
            v
            for v in self._node_store.values()
            if v.get("type") == "MemoryNode" and v.get("status") == "persistent"
        ]
        # Apply decay before sorting so weights are current
        for node in persistent:
            self._apply_decay(node)
        persistent.sort(key=lambda x: x.get("effective_weight", 0), reverse=True)
        return persistent[:n]

    # -------------------------------------------------------------------------
    # Social History — Anti-repetition memory for the Omni-Voice
    # -------------------------------------------------------------------------
    def get_recent_social_history(self, limit: int = 3) -> List[str]:
        """
        Return the last N posted tweet texts for anti-repetition context.
        Most recent first.
        """
        posts = [
            v
            for v in self._node_store.values()
            if v.get("type") == "RecentPostNode"
        ]
        posts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return [p.get("text", "") for p in posts[:limit]]

    def add_social_history(self, tweet_text: str):
        """
        Record a posted tweet for anti-repetition tracking.
        Keeps a rolling window of the 5 most recent posts.
        """
        now = time.time()
        node_id = f"social_post_{int(now * 1000)}"
        self._node_store[node_id] = {
            "id": node_id,
            "type": "RecentPostNode",
            "text": tweet_text,
            "created_at": now,
        }
        # Prune old posts beyond the rolling window
        posts = [
            (k, v)
            for k, v in self._node_store.items()
            if v.get("type") == "RecentPostNode"
        ]
        if len(posts) > 5:
            posts.sort(key=lambda x: x[1].get("created_at", 0))
            for k, _ in posts[: len(posts) - 5]:
                del self._node_store[k]

    # -------------------------------------------------------------------------
    # Research Topic Tracking — For Synchronicity Detection
    # -------------------------------------------------------------------------
    def add_research_topic(self, topic: str):
        """Record a research topic for synchronicity detection in social engagement."""
        now = time.time()
        node_id = f"research_topic_{int(now * 1000)}"
        self._node_store[node_id] = {
            "id": node_id,
            "type": "ResearchTopicNode",
            "topic": topic,
            "created_at": now,
        }
        # Keep rolling window of last 10 research topics
        topics = [
            (k, v) for k, v in self._node_store.items()
            if v.get("type") == "ResearchTopicNode"
        ]
        if len(topics) > 10:
            topics.sort(key=lambda x: x[1].get("created_at", 0))
            for k, _ in topics[:len(topics) - 10]:
                del self._node_store[k]

    def get_recent_research_topics(self, n: int = 5) -> List[str]:
        """Return the N most recent research topics for synchronicity matching."""
        topics = [
            v for v in self._node_store.values()
            if v.get("type") == "ResearchTopicNode"
        ]
        topics.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return [t.get("topic", "") for t in topics[:n]]

    # -------------------------------------------------------------------------
    # Social Graph & System Pulse Nodes (local-only)
    # -------------------------------------------------------------------------
    async def get_unique_interactors(self, timespan_seconds: int = 86400) -> Set[str]:
        """Graph Traversal: unique User_IDs interacting within the timeframe."""
        cutoff_time = time.time() - timespan_seconds
        unique_users = set()

        for v in self._node_store.values():
            if v.get("type") == "MemoryNode" and v.get("created_at", 0) >= cutoff_time:
                source_identity = v.get("source_id")
                if source_identity:
                    unique_users.add(source_identity)

        return unique_users

    async def add_system_pulse(self, sentiment: float, epoch_id: int):
        """Creates a System_Pulse node storing batch sentiment from Meditations."""
        node_id = f"pulse_{epoch_id}"
        self._node_store[node_id] = {
            "id": node_id,
            "type": "SystemPulseNode",
            "sentiment": sentiment,
            "epoch_id": epoch_id,
            "created_at": time.time(),
        }

    async def get_recent_sentiments(self, count: int = 4) -> List[float]:
        """Retrieves sentiment values from the most recent System_Pulse nodes."""
        pulses = [v for v in self._node_store.values() if v.get("type") == "SystemPulseNode"]
        pulses.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return [p.get("sentiment", 0.5) for p in pulses[:count]]

    # -------------------------------------------------------------------------
    # Social Interaction Metrics (local-only)
    # -------------------------------------------------------------------------
    async def fetch_social_metrics(self) -> dict:
        """Fetch daily interaction counters including Social Gravity fields."""
        defaults = {
            "daily_likes": 0, "daily_replies": 0,
            "mentions_received": 0, "reply_likes": 0,
        }
        for node in self._node_store.values():
            if node.get("type") == "SocialMetricsNode":
                stats = node.get("stats", {})
                # Ensure all fields exist
                return {**defaults, **stats}
        return defaults

    async def update_social_metrics(
        self,
        likes_inc: int = 0,
        replies_inc: int = 0,
        mentions_received_inc: int = 0,
        reply_likes_inc: int = 0,
    ):
        """
        Atomically increment social metrics.

        Tracks daily engagement counters plus Social Gravity fields:
          - mentions_received: how many times the Titan was mentioned
          - reply_likes: likes received on the Titan's replies
        """
        defaults = {
            "daily_likes": 0, "daily_replies": 0,
            "mentions_received": 0, "reply_likes": 0,
        }
        for node in self._node_store.values():
            if node.get("type") == "SocialMetricsNode":
                stats = node["stats"]
                # Ensure Social Gravity fields exist (backward compat)
                for k, v in defaults.items():
                    stats.setdefault(k, v)
                stats["daily_likes"] += likes_inc
                stats["daily_replies"] += replies_inc
                stats["mentions_received"] += mentions_received_inc
                stats["reply_likes"] += reply_likes_inc
                return

        self._node_store["social_metrics"] = {
            "id": "social_metrics",
            "type": "SocialMetricsNode",
            "stats": {
                "daily_likes": likes_inc,
                "daily_replies": replies_inc,
                "mentions_received": mentions_received_inc,
                "reply_likes": reply_likes_inc,
            },
            "last_reset": time.time(),
        }

    async def reset_daily_social_metrics(self):
        """Resets all counters (including Social Gravity) for a new Greater Epoch."""
        for node in self._node_store.values():
            if node.get("type") == "SocialMetricsNode":
                node["stats"] = {
                    "daily_likes": 0, "daily_replies": 0,
                    "mentions_received": 0, "reply_likes": 0,
                }
                node["last_reset"] = time.time()
