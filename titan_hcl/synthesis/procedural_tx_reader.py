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
            "SELECT block_hash, fork_id, block_height, file_offset, "
            "thought_type, tags, timestamp "
            "FROM block_index bi "
            "JOIN fork_registry fr ON bi.fork_id = fr.fork_id "
            "WHERE fr.fork_name = ? AND bi.timestamp > ? "
            "  AND bi.thought_type = 'tool_call' "
            "ORDER BY bi.timestamp DESC LIMIT ?",
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

    # Resolve the chain payload path for the procedural fork. Convention:
    # chain_<fork>.bin alongside index.db.
    chain_path = Path(chain_dir) / f"chain_{fork_name}.bin"
    if not chain_path.exists():
        logger.debug("[procedural_tx_reader] chain file missing: %s", chain_path)
        return []

    # For each block index row, seek into chain_path at file_offset and parse
    # the payload. Use the existing chain_reader iteration as the safe path
    # (block size varies; sequential read is the canonical pattern).
    # We perform ONE pass collecting by file_offset for efficiency.
    offsets = {int(r["file_offset"]): r for r in rows}
    out: list[dict] = []
    try:
        from titan_hcl.synthesis.chain_reader import iter_block_contents  # local import (heavy)
    except Exception:
        return []

    # iter_block_contents yields (height, thought_type, source, content)
    # in chain order. We accumulate matching tool_call records.
    matched_count = 0
    for height, thought_type, source, content in iter_block_contents(chain_path):
        if thought_type != "tool_call":
            continue
        if not isinstance(content, dict):
            continue
        # Synthesize the dict expected by LLMJudge / ProceduralMiner. The
        # content already carries all expected fields per OuterMemoryWriter.write_tool_call.
        # We use the content-hash from the payload (the tx_hash field if present;
        # otherwise sha256 of canonical content).
        content_hash = (
            content.get("tx_hash")
            or content.get("content_hash")
        )
        if not content_hash:
            # Best-effort fallback: compute the content hash like OuterMemoryWriter does
            try:
                import hashlib
                canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
                content_hash = hashlib.sha256(canonical).hexdigest()
            except Exception:
                continue
        if float(content.get("ts") or 0.0) <= float(since_ts):
            continue
        out.append({
            "tx_hash": content_hash,
            "content": dict(content),
            "tags": list(content.get("tags") or []),
        })
        matched_count += 1
        if matched_count >= limit:
            break

    return out


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
