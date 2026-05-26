"""Production-grade default implementations of the ConsolidationPass
mine + LLM-propose callables (§P4.G).

`titan_hcl/synthesis/consolidation.py` holds the pure orchestration; this
module provides the two REAL collaborators production wires at synthesis
worker boot. Tests use fakes (see test_phase4_consolidation.py).

- `default_mine_recent_txs(since_ts, exclude_forks)` — reads canonical
  TXs from `data/timechain/index.db.block_index` via a read-only SQLite
  query (the Phase 2 FORK_READ infra). Returns TxCandidate per row.
  Embedding is left None because the chain doesn't hold raw 132D vectors
  (only an `embedding_hash` per arch §7); clustering falls back to
  Jaccard-only mode for tag-coherent groups, which is functional for P4.
  Phase 7+ wires real FAISS-by-tx_hash embedding fetch when the chat-TX
  shape's full 132D vector becomes available cross-worker.

- `default_llm_propose(cluster, provider)` — calls the inference module's
  `complete()` to ask the model whether the cluster represents a new
  concept, a version-bump of an existing one, or nothing coherent.
  Output is parsed via a strict line-prefix protocol so a verbose LLM
  response cannot break clustering. Provider failure / parse error →
  LLMProposal(action="reject", reason=<diagnostic>) — never raises.

Both functions are sync; the LLM call is bridged via `asyncio.run` since
synthesis_worker's bus loop is thread-based (no surrounding event loop).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any, Optional

from titan_hcl.synthesis.consolidation import (
    Cluster,
    LLMProposal,
    TxCandidate,
)

logger = logging.getLogger(__name__)

# Hard-cap rows pulled per mine call. The dream window is bounded but
# autonomous tool-call volume can be large; capping protects the dream-
# boundary budget. Per arch §16.1 tiered anchoring caps individual-anchor
# growth so the absolute number stays moderate; 500 is comfortably above
# any organic-traffic 24h window seen in P3 fleet verify.
_MINE_ROW_CAP = 500


# ── Default mine ────────────────────────────────────────────────────


def default_mine_recent_txs(
    *,
    since_ts: float,
    exclude_forks: set[str],
    index_db_path: str = "data/timechain/index.db",
    row_cap: int = _MINE_ROW_CAP,
) -> list[TxCandidate]:
    """Read recent canonical TXs from the chain index DB. Read-only; never
    blocks the chain writer (sqlite WAL semantics + read_only open)."""
    out: list[TxCandidate] = []
    try:
        # uri-mode read-only open ensures we never accidentally take a
        # write lock against the chain worker's writer.
        conn = sqlite3.connect(
            f"file:{index_db_path}?mode=ro&immutable=0",
            uri=True, timeout=5.0,
        )
    except Exception as e:
        logger.warning(
            "[consolidation_defaults] mine: failed to open %s: %s",
            index_db_path, e,
        )
        return out

    try:
        conn.row_factory = sqlite3.Row
        # block_index schema (from timechain_v2.py:1901): tx_hash BLOB PK,
        # fork_name, tx_type, source, significance, tags (JSON string), ts.
        # Filter by ts > since_ts + exclude noisy forks.
        placeholders = ",".join("?" * len(exclude_forks)) if exclude_forks else ""
        excluded = list(exclude_forks)
        if excluded:
            sql = (
                "SELECT tx_hash, fork_name, tags, ts FROM block_index "
                f"WHERE ts > ? AND fork_name NOT IN ({placeholders}) "
                "ORDER BY ts DESC LIMIT ?"
            )
            params: list[Any] = [float(since_ts), *excluded, int(row_cap)]
        else:
            sql = (
                "SELECT tx_hash, fork_name, tags, ts FROM block_index "
                "WHERE ts > ? ORDER BY ts DESC LIMIT ?"
            )
            params = [float(since_ts), int(row_cap)]
        cursor = conn.execute(sql, params)
        for row in cursor.fetchall():
            tx_hash_blob = row["tx_hash"]
            if isinstance(tx_hash_blob, bytes):
                tx_hex = tx_hash_blob.hex()
            else:
                tx_hex = str(tx_hash_blob)
            tags_raw = row["tags"]
            tags: tuple[str, ...] = ()
            if isinstance(tags_raw, str) and tags_raw:
                try:
                    parsed = json.loads(tags_raw)
                    if isinstance(parsed, list):
                        tags = tuple(t for t in parsed if isinstance(t, str))
                except json.JSONDecodeError:
                    pass
            out.append(TxCandidate(
                tx_hash=tx_hex,
                fork=str(row["fork_name"]),
                tags=tags,
                embedding=None,  # raw vectors not in chain; FAISS fetch is Phase 7
                content_summary="",
            ))
    except Exception as e:
        logger.warning(
            "[consolidation_defaults] mine: query failed (%s) — returning %d rows",
            e, len(out),
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return out


# ── Default LLM propose ────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = (
    "You are part of Titan's synthesis engine. You will receive a cluster "
    "of related thoughts (chain TXs) that may represent ONE coherent "
    "concept in Titan's experience.\n\n"
    "Decide whether the cluster is:\n"
    "  - a NEW concept that should be materialized (give it a short, "
    "human-readable concept_id like 'linux_terminal' or "
    "'metaplex_nft_minting'),\n"
    "  - a VERSION_BUMP of an existing concept (you'll be told which "
    "concept_ids already exist in the cluster's tag set),\n"
    "  - or REJECT (clusters that are noise or already represented).\n\n"
    "Respond in this EXACT format (one field per line):\n"
    "ACTION: new_concept | version_bump | reject\n"
    "CONCEPT_ID: <id_with_underscores>\n"
    "NAME: <human-readable name, empty for reject>\n"
    "MEMORY_TYPE: declarative | procedural | episodic | meta\n"
    "REASON: <one short sentence>\n"
)


def _build_cluster_prompt(cluster: Cluster, max_tags: int = 30) -> str:
    """Compact prompt describing a cluster — TX hashes (first 12 chars),
    fork distribution, top tags. Bounded so the prompt stays small."""
    n = len(cluster.members)
    tag_counts: dict[str, int] = {}
    fork_counts: dict[str, int] = {}
    for tx in cluster.members:
        fork_counts[tx.fork] = fork_counts.get(tx.fork, 0) + 1
        for t in tx.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(
        tag_counts.items(), key=lambda kv: -kv[1],
    )[:max_tags]
    top_tag_str = ", ".join(f"{t}×{c}" for t, c in top_tags)
    forks_str = ", ".join(f"{f}×{c}" for f, c in fork_counts.items())
    sample_hashes = ", ".join(
        m.tx_hash[:12] for m in cluster.members[:8]
    )
    return (
        f"Cluster size: {n} thought(s)\n"
        f"Forks: {forks_str}\n"
        f"Top tags: {top_tag_str}\n"
        f"Sample TX prefixes: {sample_hashes}\n"
    )


def _parse_llm_response(text: str) -> LLMProposal:
    """Parse the LLM's line-prefix response. Lenient: missing fields →
    safe defaults that still produce a valid LLMProposal. Unknown ACTION
    → reject. Never raises."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        if key in ("ACTION", "CONCEPT_ID", "NAME", "MEMORY_TYPE", "REASON"):
            fields[key] = val.strip()

    action_raw = fields.get("ACTION", "reject").lower()
    if action_raw not in ("new_concept", "version_bump", "reject"):
        action_raw = "reject"

    if action_raw == "reject":
        return LLMProposal(
            action="reject",
            reason=fields.get("REASON", ""),
        )

    concept_id = fields.get("CONCEPT_ID", "").strip()
    if not concept_id:
        return LLMProposal(
            action="reject",
            reason="llm_returned_empty_concept_id",
        )
    # Sanitize concept_id: lowercase + underscores; the chain tag space
    # treats concept_ids as identifiers, not freeform names.
    concept_id = concept_id.lower().replace(" ", "_").replace("-", "_")

    memory_type = fields.get("MEMORY_TYPE", "meta").lower()
    if memory_type not in ("declarative", "procedural", "episodic", "meta"):
        memory_type = "meta"

    return LLMProposal(
        action=action_raw,  # type: ignore[arg-type]
        concept_id=concept_id,
        proposed_name=fields.get("NAME", concept_id) or concept_id,
        memory_type=memory_type,
        base_concept_refs=(),
        reason=fields.get("REASON", ""),
    )


def make_default_llm_propose(provider: Any) -> Any:
    """Bind a provider into a sync `llm_propose_fn(cluster) -> LLMProposal`
    callable. The provider must conform to the
    `titan_hcl.inference.base.InferenceProvider` async interface — we
    bridge via asyncio.run().

    Provider failure / parse error → REJECT with a diagnostic reason
    (never raises)."""

    def _propose(cluster: Cluster) -> LLMProposal:
        prompt = _build_cluster_prompt(cluster)
        try:
            text = asyncio.run(provider.complete(
                prompt=prompt,
                system=_LLM_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=300,
                timeout=45.0,
            ))
        except Exception as e:
            logger.warning(
                "[consolidation_defaults] llm_propose provider failed: %s", e,
            )
            return LLMProposal(
                action="reject",
                reason=f"llm_provider_error: {type(e).__name__}",
            )

        return _parse_llm_response(text or "")

    return _propose


__all__ = (
    "default_mine_recent_txs",
    "make_default_llm_propose",
    "_parse_llm_response",
    "_build_cluster_prompt",
)
