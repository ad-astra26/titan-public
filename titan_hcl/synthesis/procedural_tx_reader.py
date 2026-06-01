"""Procedural tool-call TX reader + chain content-hash reader (Phase 8).

Two thin helpers used by the dream-time pipeline:

- `default_procedural_tool_call_reader(since_ts, limit, ...)` — queries
  the chain index for procedural-fork tool-call TXs in window. Used by
  LLMJudge.score_window and ProceduralMiner.mine_pass.

- `ChainContentHashReader(chain_path)` — wraps `chain_reader.iter_block_contents`
  to provide INV-Syn-20's `read_tx_by_content_hash(h)` lookup for SkillVerifier.

Both are read-only and never block the chain writer (sqlite WAL + read-only
opens; chain_*.bin opened separately).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# Match the consolidation_defaults pattern: read-only sqlite URI open.
_DEFAULT_INDEX_DB = "data/timechain/index.db"
# Default cap matches PLAN P8.B (mine_pass scans up to 5000 TXs per window).
_DEFAULT_ROW_CAP = 5000

# Default tool-call procedural fork name. Production sets this via
# the timechain fork_registry; the constant here is the fallback.
_PROCEDURAL_FORK_NAME = "procedural"

# Default chain payload directory (chain_<fork>.bin lives here).
_DEFAULT_CHAIN_DIR = "data/timechain"

# Tool-call index query. ALL columns are table-qualified — both block_index and
# fork_registry carry a `fork_id` column, so an unqualified `fork_id` in the
# SELECT raises sqlite3 "ambiguous column name" (silently swallowed → []),
# which is why oracle coverage read 0 for the entire Phase-6 lifetime. Pinned by
# tests/test_procedural_tx_reader.py (2026-06-01).
_TOOL_CALL_INDEX_SQL = (
    "SELECT bi.block_hash, bi.fork_id, bi.block_height, bi.file_offset, "
    "bi.thought_type, bi.tags, bi.timestamp "
    "FROM block_index bi "
    "JOIN fork_registry fr ON bi.fork_id = fr.fork_id "
    "WHERE fr.fork_name = ? AND bi.timestamp > ? "
    "  AND bi.thought_type = 'tool_call' "
    "ORDER BY bi.timestamp DESC LIMIT ?"
)


def default_procedural_tool_call_reader(
    since_ts: float,
    limit: int = _DEFAULT_ROW_CAP,
    *,
    index_db_path: str = _DEFAULT_INDEX_DB,
    chain_dir: str = _DEFAULT_CHAIN_DIR,
    fork_name: str = _PROCEDURAL_FORK_NAME,
) -> list[dict]:
    """Return tool-call TXs from the procedural fork since `since_ts`.

    Each dict has shape compatible with the LLMJudge / ProceduralMiner:
      {
        "tx_hash": "<content_hash hex>",
        "content": {
          "tool_id": str, "args": dict, "success": bool, "scored_by": str|None,
          "parent_chat_tx": str|None, "parent_goal": str|None, "ts": float,
          "result_summary": str, "exception": str|None, "latency_ms": int,
        },
        "tags": list[str],
      }

    Returns [] on any I/O error (read-only chain access; never raises to
    the dream-pass caller).
    """
    if not os.path.exists(index_db_path):
        return []
    try:
        conn = sqlite3.connect(
            f"file:{index_db_path}?mode=ro&immutable=0",
            uri=True, timeout=5.0,
        )
    except Exception as e:
        logger.warning("[procedural_tx_reader] index open failed: %s", e)
        return []
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            _TOOL_CALL_INDEX_SQL,
            [fork_name, float(since_ts), int(limit)],
        )
        rows = list(cur.fetchall())
    except sqlite3.Error as e:
        logger.warning("[procedural_tx_reader] query failed: %s", e)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        return []

    # v2 chain shape (2026-06-01): the procedural fork seals TXs into BATCH
    # blocks whose content is a `{v2, tx_count, tx_merkle_root, tx_summaries}`
    # envelope — NOT the per-TX content. Each `tx_summaries` entry carries
    # `{hash, type, source, tags}` where the tags hold `tool:<id>` +
    # `scored_by:<v>`. That is exactly what coverage (INV-Syn-15) needs; full
    # args/result are not on-chain in v2 (the prior reader expected per-TX
    # content and so returned nothing — coverage read 0 for Phase-6's whole
    # life). We resolve each indexed block's content at its `file_offset` and
    # walk its tx_summaries via the canonical `resolve_batch_summaries` helper
    # (handles inline + CAS-slimmed shapes). Block timestamp (from the index)
    # is the TX timestamp.
    from titan_hcl.synthesis.chain_reader import read_block_content_at
    from titan_hcl.logic.timechain_v2 import resolve_batch_summaries

    data_dir = Path(chain_dir).parent  # read_block_content_at takes the DATA dir
    out: list[dict] = []
    for r in rows:  # already ORDER BY timestamp DESC LIMIT in the SQL
        block_ts = float(r["timestamp"])
        try:
            content = read_block_content_at(
                data_dir, int(r["fork_id"]), int(r["file_offset"]))
        except Exception:
            continue
        if not isinstance(content, dict):
            continue
        try:
            summaries = resolve_batch_summaries(content)
        except Exception:
            # Pre-v2 / non-batch block: the content may itself be the per-TX
            # payload (legacy shape) — fall back to treating it as one TX.
            summaries = [content] if content.get("tool_id") else []
        for s in summaries or []:
            if not isinstance(s, dict):
                continue
            if s.get("type", "tool_call") != "tool_call" and "tool_id" not in s:
                continue
            tags = list(s.get("tags") or [])
            scored_by = _scored_by_from_tags(tags)
            tool_id = _tool_id_from_tags(tags) or s.get("tool_id", "")
            out.append({
                "tx_hash": s.get("hash") or s.get("tx_hash") or "",
                "content": {
                    "tool_id": tool_id,
                    "args": s.get("args", {}),
                    "success": s.get("success"),
                    "scored_by": scored_by,
                    "ts": block_ts,
                    "result_summary": s.get("result_summary", ""),
                },
                "tags": tags,
            })
            if len(out) >= limit:
                return out

    return out


def _scored_by_from_tags(tags: list) -> Optional[str]:
    """Extract the `scored_by` verdict from TX tags. `scored_by:none` → None
    (unscored); `scored_by:oracle`/`llm` → that value."""
    for t in tags:
        if isinstance(t, str) and t.startswith("scored_by:"):
            v = t.split(":", 1)[1]
            return None if v in ("none", "") else v
    return None


def _tool_id_from_tags(tags: list) -> str:
    for t in tags:
        if isinstance(t, str) and t.startswith("tool:"):
            return t.split(":", 1)[1]
    return ""


class ChainContentHashReader:
    """INV-Syn-20: resolve a content hash → block dict (read-only)."""

    def __init__(self, *, chain_dir: str = _DEFAULT_CHAIN_DIR, fork_names: Optional[list] = None):
        self._chain_dir = chain_dir
        self._fork_names = fork_names or ["procedural", "declarative", "episodic",
                                            "meta", "conversation", "main"]
        # In-memory cache so repeated SkillVerifier passes on the same window
        # don't rescan chain_*.bin every time.
        self._cache: dict[str, Optional[dict]] = {}

    def read_tx_by_content_hash(self, h: str) -> Optional[dict]:
        """Return a dict with content_hash + fork + height fields, or None on miss.

        Walks chain_*.bin files for the candidate forks. The first match
        wins. Soft-fail on I/O errors (returns None — SkillVerifier
        interprets as 'miss → reject')."""
        if not h:
            return None
        if h in self._cache:
            return self._cache[h]
        try:
            from titan_hcl.synthesis.chain_reader import iter_block_contents
        except Exception:
            self._cache[h] = None
            return None
        try:
            import hashlib
        except Exception:
            return None

        for fork in self._fork_names:
            chain_path = Path(self._chain_dir) / f"chain_{fork}.bin"
            if not chain_path.exists():
                continue
            try:
                for height, thought_type, source, content in iter_block_contents(chain_path):
                    if not isinstance(content, dict):
                        continue
                    stored_hash = content.get("tx_hash") or content.get("content_hash")
                    if not stored_hash:
                        try:
                            canonical = json.dumps(
                                content, sort_keys=True, separators=(",", ":"),
                            ).encode()
                            stored_hash = hashlib.sha256(canonical).hexdigest()
                        except Exception:
                            continue
                    if stored_hash == h:
                        rec = {
                            "content_hash": stored_hash,
                            "fork": fork,
                            "height": height,
                            "thought_type": thought_type,
                        }
                        self._cache[h] = rec
                        return rec
            except Exception as e:
                logger.debug("[ChainContentHashReader] %s scan failed: %s", chain_path, e)
                continue

        self._cache[h] = None
        return None
