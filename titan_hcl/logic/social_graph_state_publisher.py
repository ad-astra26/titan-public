"""
social_graph_state_publisher — owns social_graph_state.bin SHM writer.

Phase C v1.7.1 (D-SPEC-50) per rFP_titan_hcl_l2_separation_strategy §4.P.

Invoked from social_graph_worker's periodic publisher thread @ 1 Hz.
Single-threaded (G21 single-writer-per-slot). Closes the legacy
`get_social_stats` orphan-handler (G22 violation documented in
phase_c_rpc_exemptions.yaml::orphan_handler_allowlist with rationale
"SocialGraph stats; full migration deferred") — consumers now read
SHM per G18 instead.

Cold-boot safe — if SocialGraph instance is None or DB query fails,
publish a stub payload with zeros + cold-boot ts. Consumers treat
defaults as "cold" and proceed.

Failure modes mirror MemoryStatePublisher precedent:
  - encode/oversize/write fails handled per-slot; first WARN with
    exc_info, subsequent throttled to every 60 ticks.
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.social_graph_state_specs import (
    SOCIAL_GRAPH_STATE_SLOT,
    SOCIAL_GRAPH_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class SocialGraphStatePublisher:
    """Owns social_graph_state.bin SHM writer (G21 single-writer)."""

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[SocialGraphStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.7.1 / Preamble G18 + G21)",
            titan_id, self._shm_root, SOCIAL_GRAPH_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(SOCIAL_GRAPH_STATE_SPEC, self._shm_root)
        logger.info(
            "[SocialGraphStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            SOCIAL_GRAPH_STATE_SLOT, SOCIAL_GRAPH_STATE_SPEC.payload_bytes,
            SOCIAL_GRAPH_STATE_SPEC.schema_version,
            self._shm_root / f"{SOCIAL_GRAPH_STATE_SLOT}.bin")
        return self._writer

    def publish(self, social_graph: Any) -> None:
        """Compute stats payload from `social_graph` and write to SHM slot.

        `social_graph` is the in-process SocialGraph reference held by
        social_graph_worker (NOT a proxy — direct access). Cold-boot safe.
        """
        self._publish_count += 1
        payload = self._compute_payload(social_graph)
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[SocialGraphStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d}",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails)

    def _compute_payload(self, social_graph: Any) -> dict[str, Any]:
        """Build the msgpack payload via SocialGraph.get_stats() + an extra
        engagement_ledger COUNT-since-midnight. Defensive at cold-boot per G20."""
        now_ts = time.time()
        if social_graph is None:
            return {
                "users": 0,
                "edges": 0,
                "donations": 0,
                "total_donated_sol": 0.0,
                "inspirations": 0,
                "engagement_ledger_today": 0,
                "schema_version": SOCIAL_GRAPH_STATE_SPEC.schema_version,
                "ts": now_ts,
            }

        try:
            stats = social_graph.get_stats()
        except Exception as e:
            logger.warning(
                "[SocialGraphStatePublisher] get_stats raised: %s",
                e, exc_info=True)
            stats = {}

        ledger_today = 0
        try:
            db_path = getattr(social_graph, "_db_path", None)
            if db_path:
                midnight = datetime.datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0).timestamp()
                with sqlite3.connect(
                    f"file:{db_path}?mode=ro", uri=True, timeout=2
                ) as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM engagement_ledger "
                        "WHERE timestamp > ?",
                        (midnight,),
                    ).fetchone()
                    ledger_today = int(row[0] if row and row[0] else 0)
        except Exception as e:
            logger.warning(
                "[SocialGraphStatePublisher] engagement_ledger COUNT raised: %s",
                e, exc_info=True)

        return {
            "users": int(stats.get("users", 0) or 0),
            "edges": int(stats.get("edges", 0) or 0),
            "donations": int(stats.get("donations", 0) or 0),
            "total_donated_sol": float(stats.get("total_donated_sol", 0.0) or 0.0),
            "inspirations": int(stats.get("inspirations", 0) or 0),
            "engagement_ledger_today": ledger_today,
            "schema_version": SOCIAL_GRAPH_STATE_SPEC.schema_version,
            "ts": now_ts,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[SocialGraphStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > SOCIAL_GRAPH_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[SocialGraphStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), SOCIAL_GRAPH_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[SocialGraphStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d (consumers can now read; "
                    "closes G22 get_social_stats orphan-handler deferral "
                    "per rFP §4.P + D-SPEC-50)",
                    SOCIAL_GRAPH_STATE_SLOT, len(encoded))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[SocialGraphStatePublisher] shm write failed (#%d): %s",
                    self._write_fails, e, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
        }
