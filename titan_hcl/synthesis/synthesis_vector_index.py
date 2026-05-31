"""SynthesisVectorStore + FaissReader — the tx_hash-native retrieval substrate.

Synthesis Engine Operator Closure, Phase A (arch §3.6 / INV-15 — the tx_hash
spine). This is the substrate the SPEC always specified but the implementation
never built: outer-memory vectors **addressed BY the anchoring `tx_hash`**, not
by a legacy autoincrement `node_id`. SEARCH/recall return `tx_hashes`; the
operator dereferences `tx_hash → outer-memory content`.

Two roles, one shard convention:

  * **SynthesisVectorStore** — the SOLE WRITER (synthesis_worker; G21 / the
    INV-Syn-3 sole-writer family). On seal of a meaningful TX it embeds the TX
    content (the same fastembed BAAI/bge-small-en-v1.5 path the rest of the
    engine uses) and appends `(tx_hash → vector)` to the per-fork shard. It
    also serves as an in-process `faiss_reader` for synthesis_worker's own
    EngineRecall (so the SEARCH op finally returns hits inside the worker).

  * **FaissReader** — the read-only adapter every consumer process (agno,
    cognitive) opens. It exposes exactly the `.knn(fork, vec, k, min_similarity)
    -> [{tx_hash, score, fork}]` contract the RuleEvaluator `SEARCH` op was
    waiting for ("wired in 2D" — finally done here, arch §12.1). Cross-process
    reads follow the established `skills_vectors.faiss` pattern: each reader
    opens its own `faiss.read_index()` copy and reloads on mtime change; the
    writer saves atomically (tmp + rename), so a reader never observes a torn
    index.

**Per-fork shards (PLAN §4 Q1 lean).** One `IndexFlatIP` per indexed fork at
`data/synthesis_vectors_<fork>.faiss`, with a position-aligned id_map of
**tx_hash hex strings** persisted alongside as `<...>.idmap.json`. Storing the
tx_hash string directly (the existing `TitanVectorIndex._id_map` pattern)
sidesteps the int64-truncation collision problem of a single combined index
entirely — a 256-bit tx_hash never has to be squeezed into an int64 key.

**Cosine via inner product.** Vectors are L2-normalized on both the add and the
query side (defensively — independent of whether the injected embedder
normalizes), so `IndexFlatIP` inner product == cosine similarity, matching the
legacy `memory_vectors` path and arch §5.3 ("cosine = FAISS L2-norm inner
product"). `min_similarity` therefore thresholds cosine in [-1, 1].
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# 384D BAAI/bge-small-en-v1.5 — the one embedding model the engine uses
# (arch §4: "384D BAAI/bge-small-en-v1.5"). Matches memory_vectors +
# skills_vectors so a query embedded once is comparable across all indices.
EMBEDDING_DIM = 384

# The forks that participate in tx_hash-native retrieval / synthesis (PLAN §4 Q4):
#   conversation — chat recall (Gate B primary path)
#   declarative  — concepts (consolidation clusters by cosine — W4)
#   procedural   — tool-call TXs (oracle -> verdict -> skill loop — Gate C)
# episodic (inner-expression flood, 166k blocks image/speak/sound — audit §84)
# and meta (book-keeping) are intentionally NOT indexed.
INDEXED_FORKS = ("conversation", "declarative", "procedural")


def _l2_normalize(vec: "Any") -> "Any":
    """Return an L2-normalized float32 (1, dim) row. Zero-norm → unchanged
    (an all-zero embedding stays all-zero rather than dividing by zero — the
    fleet zero-embedding bug b621e80c is fixed upstream; this is just defensive
    so a degenerate vector never raises)."""
    import numpy as np
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


class _FaissShard:
    """One fork's tx_hash-keyed FAISS shard: an IndexFlatIP + a position-aligned
    list of tx_hash hex strings. Handles lazy load, mtime-reload (reader side),
    append + atomic save (writer side), knn, and reconstruct-by-tx_hash."""

    def __init__(self, faiss_path: str, dim: int = EMBEDDING_DIM):
        self._faiss_path = str(faiss_path)
        self._idmap_path = self._faiss_path + ".idmap.json"
        self._dim = int(dim)
        self._index = None
        self._id_map: list[str] = []          # FAISS position -> tx_hash hex
        self._pos_by_hash: dict[str, int] = {}  # tx_hash hex -> FAISS position
        self._loaded_mtime: float = -1.0

    # ── load / reload ────────────────────────────────────────────────
    def _index_mtime(self) -> float:
        try:
            return os.path.getmtime(self._faiss_path)
        except OSError:
            return -1.0

    def _ensure_loaded(self, *, allow_create: bool) -> None:
        """Load the shard if not yet loaded; on the reader side, transparently
        reload when the on-disk index mtime advances (the writer saved a newer
        copy). `allow_create` is True for the writer so a missing shard starts
        as a fresh empty index; False for a reader (a missing shard stays None
        → knn returns [])."""
        import faiss
        disk_mtime = self._index_mtime()
        if self._index is not None and disk_mtime == self._loaded_mtime:
            return  # current
        if not os.path.exists(self._faiss_path) or disk_mtime < 0:
            if allow_create and self._index is None:
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_map = []
                self._pos_by_hash = {}
                self._loaded_mtime = -1.0
            return
        # (Re)load index + idmap from disk.
        try:
            sz = os.path.getsize(self._faiss_path)
        except OSError:
            sz = 0
        if sz <= 0:
            if allow_create and self._index is None:
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_map = []
                self._pos_by_hash = {}
            return
        try:
            idx = faiss.read_index(self._faiss_path)
            if idx.d != self._dim:
                logger.warning(
                    "[SynthesisVectorIndex] %s has dim=%d, expected %d — "
                    "treating as empty", self._faiss_path, idx.d, self._dim)
                if allow_create:
                    self._index = faiss.IndexFlatIP(self._dim)
                    self._id_map = []
                    self._pos_by_hash = {}
                return
            id_map: list[str] = []
            if os.path.exists(self._idmap_path):
                with open(self._idmap_path) as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    id_map = [str(x) for x in raw]
            # Guard: idmap length must match index size; on mismatch the pair is
            # inconsistent (interrupted save) — drop to empty rather than serve
            # mis-keyed hits.
            if len(id_map) != idx.ntotal:
                logger.warning(
                    "[SynthesisVectorIndex] %s idmap len=%d != ntotal=%d — "
                    "inconsistent shard, treating as empty",
                    self._faiss_path, len(id_map), idx.ntotal)
                if allow_create:
                    self._index = faiss.IndexFlatIP(self._dim)
                    self._id_map = []
                    self._pos_by_hash = {}
                return
            self._index = idx
            self._id_map = id_map
            self._pos_by_hash = {h: i for i, h in enumerate(id_map)}
            self._loaded_mtime = disk_mtime
        except Exception as e:
            logger.warning(
                "[SynthesisVectorIndex] failed to load %s (%s) — empty",
                self._faiss_path, e)
            if allow_create:
                import faiss as _f
                self._index = _f.IndexFlatIP(self._dim)
                self._id_map = []
                self._pos_by_hash = {}

    # ── writer surface ───────────────────────────────────────────────
    def has(self, tx_hash: str) -> bool:
        self._ensure_loaded(allow_create=True)
        return tx_hash in self._pos_by_hash

    def add(self, tx_hash: str, vec: "Any") -> bool:
        """Append (tx_hash -> normalized vec). Idempotent: a tx_hash already in
        the shard is skipped (returns False). Returns True on a real add.
        Does NOT save — caller batches saves via save()."""
        if not tx_hash:
            return False
        self._ensure_loaded(allow_create=True)
        if tx_hash in self._pos_by_hash:
            return False
        arr = _l2_normalize(vec)
        if arr.shape[1] != self._dim:
            logger.warning(
                "[SynthesisVectorIndex] add: vec dim=%d != %d — skipping %s",
                arr.shape[1], self._dim, tx_hash[:12])
            return False
        pos = self._index.ntotal
        self._index.add(arr)
        self._id_map.append(tx_hash)
        self._pos_by_hash[tx_hash] = pos
        return True

    def save(self) -> None:
        """Atomic tmp+rename of both the FAISS index and the idmap (the pair
        must move together so a reader never sees a length mismatch)."""
        if self._index is None:
            return
        import faiss
        try:
            os.makedirs(os.path.dirname(self._faiss_path) or ".", exist_ok=True)
            tmp_faiss = self._faiss_path + ".tmp"
            tmp_idmap = self._idmap_path + ".tmp"
            faiss.write_index(self._index, tmp_faiss)
            with open(tmp_idmap, "w") as f:
                json.dump(self._id_map, f)
            # Write idmap first, then index: a reader keys off the index mtime,
            # so the index must be the last file to land.
            os.replace(tmp_idmap, self._idmap_path)
            os.replace(tmp_faiss, self._faiss_path)
            self._loaded_mtime = self._index_mtime()
        except Exception as e:
            logger.warning(
                "[SynthesisVectorIndex] save failed (%s): %s",
                self._faiss_path, e)

    def get_vector(self, tx_hash: str) -> Optional["Any"]:
        """Reconstruct the stored (normalized) vector for tx_hash, or None.
        Used by ConsolidationPass to cluster by cosine (W4)."""
        self._ensure_loaded(allow_create=False)
        if self._index is None:
            return None
        pos = self._pos_by_hash.get(tx_hash)
        if pos is None:
            return None
        try:
            import numpy as np
            return np.asarray(self._index.reconstruct(int(pos)), dtype=np.float32)
        except Exception as e:
            logger.debug("[SynthesisVectorIndex] reconstruct failed: %s", e)
            return None

    # ── read surface ─────────────────────────────────────────────────
    def knn(self, vec: "Any", k: int, min_similarity: float,
            *, allow_create: bool) -> list[tuple[str, float]]:
        """Return up to k (tx_hash, cosine) pairs above min_similarity, sorted
        descending. Empty list when the shard is missing/empty."""
        self._ensure_loaded(allow_create=allow_create)
        if self._index is None or self._index.ntotal == 0:
            return []
        import numpy as np
        q = _l2_normalize(vec)
        if q.shape[1] != self._dim:
            return []
        n = min(int(k), self._index.ntotal)
        if n <= 0:
            return []
        scores, idxs = self._index.search(q, n)
        out: list[tuple[str, float]] = []
        for score, pos in zip(scores[0], idxs[0]):
            if pos < 0 or pos >= len(self._id_map):
                continue
            s = float(score)
            if s < float(min_similarity):
                continue
            out.append((self._id_map[pos], s))
        return out

    def ntotal(self) -> int:
        self._ensure_loaded(allow_create=False)
        return int(self._index.ntotal) if self._index is not None else 0


class SynthesisVectorStore:
    """Sole writer of the per-fork tx_hash-keyed FAISS shards (synthesis_worker;
    G21 / INV-Syn family). Embeds TX content and binds it to the chain by the
    anchoring tx_hash (arch §3.6). Also serves as an in-process `faiss_reader`
    for synthesis_worker's own EngineRecall."""

    def __init__(
        self,
        *,
        data_dir: str,
        embedder: Optional[Callable[[str], "Any"]] = None,
        forks: Iterable[str] = INDEXED_FORKS,
        dim: int = EMBEDDING_DIM,
    ):
        self._data_dir = str(data_dir)
        self._embedder = embedder
        self._dim = int(dim)
        self._forks = tuple(forks)
        self._shards: dict[str, _FaissShard] = {
            fork: _FaissShard(_shard_path(self._data_dir, fork), dim=self._dim)
            for fork in self._forks
        }
        self._dirty: set[str] = set()

    def _shard(self, fork: str) -> Optional[_FaissShard]:
        return self._shards.get(fork)

    def has(self, fork: str, tx_hash: str) -> bool:
        sh = self._shard(fork)
        return sh.has(tx_hash) if sh is not None else False

    def add_text(self, fork: str, tx_hash: str, text: str) -> bool:
        """Embed `text` and bind it under `tx_hash` in the fork's shard.
        No-op (False) when the fork is not indexed, the embedder is absent,
        the text is empty, or the tx_hash is already present (idempotent)."""
        sh = self._shard(fork)
        if sh is None or self._embedder is None or not text or not tx_hash:
            return False
        if sh.has(tx_hash):
            return False
        try:
            vec = self._embedder(text)
        except Exception as e:
            logger.debug("[SynthesisVectorStore] embed failed for %s: %s",
                         tx_hash[:12], e)
            return False
        if vec is None:
            return False
        added = sh.add(tx_hash, vec)
        if added:
            self._dirty.add(fork)
        return added

    def add_vector(self, fork: str, tx_hash: str, vec: "Any") -> bool:
        """Bind a precomputed vector under tx_hash (backfill from an existing
        index — A4 reuses already-computed embeddings where available)."""
        sh = self._shard(fork)
        if sh is None or vec is None or not tx_hash:
            return False
        added = sh.add(tx_hash, vec)
        if added:
            self._dirty.add(fork)
        return added

    def save(self, fork: Optional[str] = None) -> None:
        """Persist dirty shards atomically. `fork=None` saves all dirty shards."""
        targets = (fork,) if fork is not None else tuple(self._dirty)
        for f in targets:
            sh = self._shard(f)
            if sh is not None:
                sh.save()
            self._dirty.discard(f)

    def get_vector(self, fork: str, tx_hash: str) -> Optional["Any"]:
        sh = self._shard(fork)
        return sh.get_vector(tx_hash) if sh is not None else None

    # In-process faiss_reader contract (RuleEvaluator SEARCH op) ----------
    def knn(self, fork: str, vec: "Any", k: int,
            min_similarity: float = 0.0) -> list[dict]:
        return _knn_over(
            self._shards, fork, vec, k, min_similarity, allow_create=True)

    def stats(self) -> dict:
        return {f: sh.ntotal() for f, sh in self._shards.items()}


class FaissReader:
    """Read-only `.knn(fork, vec, k, min_similarity) -> [{tx_hash, score, fork}]`
    adapter over the per-fork tx_hash shards — the concrete impl the RuleEvaluator
    SEARCH op (arch §12.1) always specified. One per consumer process (agno,
    cognitive); reloads on mtime so it tracks the writer's atomic saves. Never
    writes; a missing shard simply yields no hits."""

    def __init__(
        self,
        *,
        data_dir: str,
        forks: Iterable[str] = INDEXED_FORKS,
        dim: int = EMBEDDING_DIM,
    ):
        self._shards: dict[str, _FaissShard] = {
            fork: _FaissShard(_shard_path(str(data_dir), fork), dim=int(dim))
            for fork in forks
        }

    def knn(self, fork: str, vec: "Any", k: int,
            min_similarity: float = 0.0) -> list[dict]:
        return _knn_over(
            self._shards, fork, vec, k, min_similarity, allow_create=False)

    def ntotal(self, fork: str) -> int:
        sh = self._shards.get(fork)
        return sh.ntotal() if sh is not None else 0


def _shard_path(data_dir: str, fork: str) -> str:
    return os.path.join(data_dir, f"synthesis_vectors_{fork}.faiss")


def _knn_over(
    shards: dict[str, _FaissShard],
    fork: str,
    vec: "Any",
    k: int,
    min_similarity: float,
    *,
    allow_create: bool,
) -> list[dict]:
    """Shared knn for store + reader. `fork="auto"` searches every indexed shard
    and merges the global top-k; a named fork searches just that shard. Returns
    the SEARCH-op shape: [{tx_hash, score, fork}] sorted by score desc."""
    if fork == "auto":
        merged: list[dict] = []
        for f, sh in shards.items():
            for tx_hash, score in sh.knn(
                    vec, k, min_similarity, allow_create=allow_create):
                merged.append({"tx_hash": tx_hash, "score": score, "fork": f})
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[: int(k)]
    sh = shards.get(fork)
    if sh is None:
        return []
    return [
        {"tx_hash": tx_hash, "score": score, "fork": fork}
        for tx_hash, score in sh.knn(
            vec, k, min_similarity, allow_create=allow_create)
    ]


__all__ = [
    "SynthesisVectorStore",
    "FaissReader",
    "EMBEDDING_DIM",
    "INDEXED_FORKS",
]
