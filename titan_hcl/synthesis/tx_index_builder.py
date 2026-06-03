"""TxIndexBuilder — populate the tx_hash-native FAISS index from the chain.

Synthesis Engine Operator Closure, Phase A (A1 + A2 + A4). This is the bridge
that binds the chain spine to the outer-memory vectors: it walks the canonical
`block_index` for the indexed forks, dereferences each anchored thought's content
from the chain `.bin` at its `file_offset`, embeds the content, and adds it to
`SynthesisVectorStore` keyed by the block's canonical hash (`block_index`'s PK —
the dereferenceable key the SEARCH op returns and the operator resolves).

Why a pull (vs. a seal-time push): the canonical hash + the content live together
only on the chain after a block is written; `block_index` carries the hash +
`file_offset` but not the content, and the v1-commit path writes immediately.
A read-only pull over `block_index` + the chain files (the same shape
`consolidation_defaults.mine` already uses) keeps this fully decoupled from the
timechain worker — no new bus events, no coupling, idempotent (a tx already in
the shard is skipped), and naturally uniform across forks.

  * **Incremental (A2):** synthesis_worker calls `run(...)` on a cadence /
    after seal; a per-fork `(fork_id → max block_height indexed)` watermark
    bounds the scan to new blocks.
  * **Backfill (A4):** `scripts/backfill_synthesis_tx_index.py` constructs the
    same builder with a zeroed watermark + `dry_run` to report-then-populate
    the full history. Read-only on the chain (adds the new index, never mutates
    `chain_*.bin`).

The builder is the SOLE writer's helper — it runs inside synthesis_worker (the
`SynthesisVectorStore` owner, G21). Cross-process consumers read via `FaissReader`.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from titan_hcl.synthesis.chain_reader import read_block_content_at
from titan_hcl.synthesis.synthesis_vector_index import INDEXED_FORKS
from titan_hcl.synthesis.thought_sidecar import ThoughtSidecarReader

logger = logging.getLogger(__name__)

# Watermark sidecar (per-fork max block_height already indexed). Lives next to
# the shards so a fresh boot resumes where it left off without re-reading the
# whole chain. JSON, atomic tmp+rename.
WATERMARK_NAME = "synthesis_tx_index_watermark.json"

# Embed BATCH_SIZE texts per fastembed call (one call >> N single calls,
# especially on a small/contended box). Persist + advance the watermark every
# SAVE_EVERY indexed so an interruption mid-backfill keeps progress.
BATCH_SIZE = 256
SAVE_EVERY = 2048


def _embeddable_text(fork: str, content: dict) -> str:
    """Extract the semantically rich text to embed from a TX content dict.

    Fork-aware with robust fallbacks (TX shapes vary by producer):
      conversation → the chat turn (user msg + agent response)
      declarative  → the concept name / id
      procedural   → the tool id + result summary
    Falls back to common text keys, then a bounded JSON dump, so no anchored
    thought is silently un-embeddable.
    """
    if not isinstance(content, dict) or not content:
        return ""
    parts: list[str] = []

    def _add(*keys: str) -> None:
        for k in keys:
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

    if fork == "conversation":
        _add("user_msg", "user_prompt", "user", "prompt")
        _add("agent_response", "response", "text", "summary")
    elif fork == "procedural":
        _add("tool_id", "name")
        _add("result_summary", "summary", "nl_description", "text")
    else:  # declarative + any other indexed fork
        _add("name", "concept_id")
        _add("summary", "text", "nl_description", "description")

    if not parts:
        # Last-resort fallbacks before giving up — common generic keys, then a
        # bounded canonical dump so a clustered/odd-shaped TX still gets a vector.
        _add("summary", "text", "name", "description", "content", "message")
    if not parts:
        try:
            dump = json.dumps(content, sort_keys=True, separators=(",", ":"))
        except Exception:
            dump = str(content)
        parts.append(dump[:512])

    return "  ".join(parts)[:2048]


class TxIndexBuilder:
    """Populate a `SynthesisVectorStore` from the canonical chain, idempotently."""

    def __init__(
        self,
        *,
        store,                              # SynthesisVectorStore (sole writer)
        data_dir: str,
        index_db: Optional[sqlite3.Connection] = None,
        forks=INDEXED_FORKS,
    ):
        self._store = store
        self._data_dir = str(data_dir)
        self._forks = tuple(forks)
        self._chain_dir = os.path.join(self._data_dir, "timechain")
        self._wm_path = os.path.join(self._data_dir, WATERMARK_NAME)
        # Read-only index.db handle (caller may inject a shared one; else open).
        self._owns_conn = index_db is None
        if index_db is not None:
            self._conn = index_db
            # Defensive: an injected conn may lack row_factory; our queries use
            # string-key access. Row is tuple-compatible so this is safe for any
            # positional consumer sharing the conn. (2026-06-01 — was the
            # "tuple indices must be integers or slices, not str" boot failure.)
            try:
                self._conn.row_factory = sqlite3.Row
            except Exception:
                pass
        else:
            self._conn = self._open_index_db()
        self._fork_ids: dict[int, str] = {}   # fork_id -> fork_name (indexed only)
        self._watermark: dict[int, int] = {}  # fork_id -> max block_height indexed
        self._resolve_fork_ids()
        self._load_watermark()

    # ── setup ─────────────────────────────────────────────────────────
    def _open_index_db(self) -> Optional[sqlite3.Connection]:
        path = os.path.join(self._chain_dir, "index.db")
        if not os.path.exists(path):
            logger.info("[TxIndexBuilder] index.db absent at %s — no-op until it exists", path)
            return None
        try:
            conn = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True,
                check_same_thread=False, timeout=2.0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.warning("[TxIndexBuilder] index.db open failed: %s", e)
            return None

    def _resolve_fork_ids(self) -> None:
        """Resolve the indexed fork NAMES → chain-local ids via fork_registry
        (INV-Syn-26 — ids are chain-local; conversation is 5 on T1, 115 on T3).

        Orders `_fork_ids` by `self._forks` (INDEXED_FORKS), NOT by fork_id, so
        the SMALL + chat-critical `conversation` fork is indexed FIRST — chat
        recall fires as soon as the backfill starts, rather than after the
        large declarative/procedural forks (fork_id order would do conversation
        last on T3, where it is sidechain id 115)."""
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT fork_id, fork_name FROM fork_registry").fetchall()
        except Exception as e:
            logger.warning("[TxIndexBuilder] fork_registry read failed: %s", e)
            return
        name_to_id = {str(r["fork_name"]): int(r["fork_id"]) for r in rows}
        # Insert in INDEXED_FORKS order (conversation first) — dict preserves it.
        for name in self._forks:
            fid = name_to_id.get(name)
            if fid is not None:
                self._fork_ids[fid] = name

    def _load_watermark(self) -> None:
        try:
            if os.path.exists(self._wm_path):
                with open(self._wm_path) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._watermark = {int(k): int(v) for k, v in raw.items()}
        except Exception as e:
            logger.debug("[TxIndexBuilder] watermark load failed: %s", e)
            self._watermark = {}

    def _save_watermark(self) -> None:
        try:
            tmp = self._wm_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({str(k): v for k, v in self._watermark.items()}, f)
            os.replace(tmp, self._wm_path)
        except Exception as e:
            logger.debug("[TxIndexBuilder] watermark save failed: %s", e)

    # ── main pass ─────────────────────────────────────────────────────
    def run(
        self,
        *,
        max_blocks: int = 5000,
        from_scratch: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Index up to `max_blocks` not-yet-indexed blocks across the indexed
        forks. Returns a summary dict. Idempotent: a block already in the shard
        (or at/below the watermark) is skipped. `from_scratch=True` ignores the
        watermark (backfill). `dry_run=True` reads + counts but never embeds or
        writes (the A4 report pass)."""
        summary = {
            "scanned": 0, "indexed": 0, "skipped": 0, "no_content": 0,
            "by_fork": {}, "dry_run": dry_run, "ts": time.time(),
        }
        if self._conn is None or not self._fork_ids:
            summary["note"] = "no index_db or no indexed forks resolved"
            return summary

        budget = int(max_blocks)
        for fork_id, fork_name in self._fork_ids.items():
            if budget <= 0:
                break
            wm = 0 if from_scratch else int(self._watermark.get(fork_id, 0))
            fork_stats = self._index_fork(
                fork_id, fork_name, wm, budget, dry_run, summary)
            summary["by_fork"][fork_name] = fork_stats
            budget -= fork_stats["scanned"]

        if not dry_run and summary["indexed"] > 0:
            self._store.save()
            self._save_watermark()
        return summary

    def _index_fork(
        self, fork_id: int, fork_name: str, watermark: int,
        budget: int, dry_run: bool, summary: dict,
    ) -> dict:
        stats = {"scanned": 0, "indexed": 0, "skipped": 0, "no_content": 0,
                 "max_height": watermark}
        try:
            rows = self._conn.execute(
                "SELECT block_hash, block_height, file_offset "
                "FROM block_index WHERE fork_id = ? AND block_height > ? "
                "ORDER BY block_height ASC LIMIT ?",
                (fork_id, int(watermark), int(budget)),
            ).fetchall()
        except Exception as e:
            logger.warning(
                "[TxIndexBuilder] block_index scan failed for fork %s: %s",
                fork_name, e)
            return stats

        # Accumulate (tx_hash, text) and flush to the store in batches — ONE
        # fastembed call per batch is dramatically faster + lighter than per-text
        # (critical on a small/contended box). Periodic save + watermark advance
        # so an interruption keeps progress (the index saves only on flush).
        batch: list[tuple[str, str]] = []
        since_save = 0

        def _flush() -> None:
            nonlocal batch, since_save
            if not batch:
                return
            added = self._store.add_texts(fork_name, batch)
            stats["indexed"] += added
            summary["indexed"] += added
            # Anything in the batch not added was a dedup skip.
            skipped = len(batch) - added
            stats["skipped"] += skipped
            summary["skipped"] += skipped
            since_save += added
            batch = []
            # Persist + advance watermark every ~SAVE_EVERY indexed so a crash
            # mid-backfill doesn't discard the whole pass.
            if since_save >= SAVE_EVERY:
                self._store.save(fork_name)
                self._watermark[fork_id] = stats["max_height"]
                self._save_watermark()
                since_save = 0

        for r in rows:
            stats["scanned"] += 1
            summary["scanned"] += 1
            bh = r["block_hash"]
            tx_hash = bh.hex() if isinstance(bh, (bytes, bytearray)) else str(bh)
            height = int(r["block_height"])
            stats["max_height"] = max(stats["max_height"], height)

            if self._store.has(fork_name, tx_hash):
                stats["skipped"] += 1
                summary["skipped"] += 1
                continue

            content = read_block_content_at(
                Path(self._data_dir), fork_id, int(r["file_offset"]))
            text = _embeddable_text(fork_name, content or {})
            if not text:
                stats["no_content"] += 1
                summary["no_content"] += 1
                continue

            if dry_run:
                stats["indexed"] += 1
                summary["indexed"] += 1
                continue

            batch.append((tx_hash, text))
            if len(batch) >= BATCH_SIZE:
                _flush()

        if not dry_run:
            _flush()
            # Advance the watermark to the highest height we scanned (even skips —
            # they are already indexed, so we never need to revisit them).
            if stats["scanned"] > 0:
                self._watermark[fork_id] = stats["max_height"]
        return stats

    def run_sidecar(self, *, max_items: int = 2000, dry_run: bool = False) -> dict:
        """Index PROMOTED thoughts from the content sidecar (RFP spine Phase C).

        Embeds the REAL thought (`user_prompt` / `agent_response` from
        `thought_sidecar.db`) — NOT the chain envelope — keyed by its per-TX hash,
        into the per-fork vector store, so SEARCH returns promoted-thought
        tx_hashes the deref resolves to real content. Idempotent (`store.has`
        skip). Forks outside `INDEXED_FORKS` (e.g. episodic) clamp to
        `declarative` so nothing promoted is left unsearchable."""
        summary = {"scanned": 0, "indexed": 0, "skipped": 0, "no_content": 0,
                   "by_fork": {}, "dry_run": dry_run, "ts": time.time()}
        try:
            reader = ThoughtSidecarReader(self._data_dir)
        except Exception as e:
            summary["note"] = f"sidecar reader init failed: {e}"
            return summary
        rows = reader.iter_all(limit=int(max_items))
        batch_by_fork: dict[str, list] = {}
        for row in rows:
            summary["scanned"] += 1
            txh = row.get("tx_hash")
            if not txh:
                continue
            fork = row.get("fork") or "declarative"
            if fork not in INDEXED_FORKS:
                fork = "declarative"
            if self._store.has(fork, txh):
                summary["skipped"] += 1
                continue
            text = _embeddable_text("conversation", {
                "user_prompt": row.get("user_prompt"),
                "agent_response": row.get("agent_response"),
            })
            if not text:
                summary["no_content"] += 1
                continue
            if dry_run:
                summary["indexed"] += 1
                continue
            batch_by_fork.setdefault(fork, []).append((txh, text))
        for fork, items in batch_by_fork.items():
            try:
                added = self._store.add_texts(fork, items)
            except Exception as e:
                logger.warning(
                    "[TxIndexBuilder] sidecar add_texts(%s) failed: %s", fork, e)
                added = 0
            summary["indexed"] += added
            summary["by_fork"][fork] = added
        if not dry_run and summary["indexed"] > 0:
            try:
                self._store.save()
            except Exception as e:
                logger.warning("[TxIndexBuilder] sidecar store.save failed: %s", e)
        reader.close()
        return summary

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


class TxContentDeref:
    """Read-only `tx_hash → outer-memory content` dereference (arch §3.6).

    SEARCH/recall return `tx_hashes`; the OPERATOR dereferences them into content
    to assemble context. This is that dereference, for any consumer process
    (agno / cognitive): look up the block by its canonical hash in the read-only
    `block_index` (→ fork_id + file_offset), read the content from the chain
    `.bin`, and return a bounded snippet. Caches a small LRU so a hot chat turn
    that surfaces the same tx repeatedly doesn't re-read the chain.

    Soft-fail throughout — a missing index.db / block / file yields None so the
    operator simply drops that candidate from the injected context.
    """

    def __init__(self, *, data_dir: str, index_db: Optional[sqlite3.Connection] = None,
                 cache_size: int = 512):
        self._data_dir = str(data_dir)
        # Phase C: prefer the lock-free content sidecar (real promoted thoughts)
        # over the chain envelope. Soft — an absent sidecar just means chain-only.
        try:
            self._sidecar: Optional[ThoughtSidecarReader] = ThoughtSidecarReader(self._data_dir)
        except Exception:
            self._sidecar = None
        self._owns_conn = index_db is None
        if index_db is not None:
            self._conn = index_db
        else:
            self._conn = self._open()
        self._cache: dict[str, Optional[str]] = {}
        self._cache_order: list[str] = []
        self._cache_size = int(cache_size)

    def _open(self) -> Optional[sqlite3.Connection]:
        path = os.path.join(self._data_dir, "timechain", "index.db")
        if not os.path.exists(path):
            return None
        try:
            conn = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True,
                check_same_thread=False, timeout=2.0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:
            return None

    def snippet(self, tx_hash: str, fork: str = "", *, max_chars: int = 512) -> Optional[str]:
        """Return a bounded content snippet for `tx_hash`, or None.

        Phase C (RFP spine): prefer the lock-free content SIDECAR — a promoted
        thought's REAL text keyed by its per-TX hash — and fall back to the chain
        block ENVELOPE only for legacy / non-promoted keys (block_hashes)."""
        if not tx_hash:
            return None
        if tx_hash in self._cache:
            return self._cache[tx_hash]
        snip: Optional[str] = None
        # 1) sidecar — the real promoted thought (canonical recall content).
        if self._sidecar is not None:
            try:
                row = self._sidecar.get(tx_hash)
                if row is not None:
                    text = _embeddable_text("conversation", {
                        "user_prompt": row.get("user_prompt"),
                        "agent_response": row.get("agent_response"),
                    })
                    snip = text[:max_chars] if text else None
            except Exception as e:
                logger.debug("[TxContentDeref] sidecar deref %s failed: %s",
                             str(tx_hash)[:12], e)
                snip = None
        # 2) fallback — chain block envelope (legacy block_hash keys).
        if snip is None and self._conn is not None:
            try:
                row = self._conn.execute(
                    "SELECT fork_id, file_offset FROM block_index WHERE block_hash = ? LIMIT 1",
                    (bytes.fromhex(tx_hash),),
                ).fetchone()
                if row is not None:
                    content = read_block_content_at(
                        Path(self._data_dir), int(row["fork_id"]), int(row["file_offset"]))
                    text = _embeddable_text(fork or "", content or {})
                    snip = text[:max_chars] if text else None
            except Exception as e:
                logger.debug("[TxContentDeref] deref %s failed: %s",
                             str(tx_hash)[:12], e)
                snip = None
        self._cache_put(tx_hash, snip)
        return snip

    def _cache_put(self, key: str, val: Optional[str]) -> None:
        if key in self._cache:
            return
        self._cache[key] = val
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._sidecar is not None:
            try:
                self._sidecar.close()
            except Exception:
                pass


__all__ = ["TxIndexBuilder", "TxContentDeref", "WATERMARK_NAME"]
