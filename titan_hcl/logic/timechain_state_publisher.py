"""
timechain_state_publisher — Phase C Session 3 §4.B.9.

Publishes timechain_state.bin from a ``TimeChain`` instance + module-level
``get_tx_latency_stats`` / ``get_block_delta_stats`` accessors. Owned by
timechain_worker.

Source: TimeChain instance attrs (total_blocks, chi_spent total, fork
summary via ContractStore + Mempool) + module-level latency/delta stats.
"""
from __future__ import annotations

from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.session3_state_specs import (
    TIMECHAIN_STATE_SLOT,
    TIMECHAIN_STATE_SPEC,
)


class TimechainStatePublisher(BaseStatePublisher):
    slot_name = TIMECHAIN_STATE_SLOT
    slot_spec = TIMECHAIN_STATE_SPEC

    def _compute_payload(self, tc: Any) -> dict[str, Any]:
        import time
        if tc is None:
            return self._cold_boot_payload()

        # TX-latency + block-delta come from module-level accessors
        # (rolling buffers — no per-call cost).
        tx_lat: dict[str, Any] = {}
        block_delta: dict[str, Any] = {}
        try:
            from titan_hcl.logic.timechain_v2 import (
                get_tx_latency_stats, get_block_delta_stats)
            tx_lat = get_tx_latency_stats() or {}
            block_delta = get_block_delta_stats() or {}
        except Exception:
            # Defensive — module-level functions may have side-effect setup
            pass

        tx_latency_norm = float(tx_lat.get("normalized", 0.5) or 0.5)
        block_delta_norm = float(block_delta.get("normalized", 0.5) or 0.5)

        # Anchor freshness — read anchor_state.json mtime if available
        recent_anchor_age_s = 0.0
        try:
            import os
            anchor_path = os.path.normpath(os.path.join(
                os.path.dirname(__file__), "..", "..", "data",
                "anchor_state.json"))
            if os.path.exists(anchor_path):
                recent_anchor_age_s = float(
                    time.time() - os.path.getmtime(anchor_path))
        except Exception:
            pass

        # TimeChain instance attrs
        total_blocks = int(getattr(tc, "total_blocks", 0) or 0)
        chi_spent_total = float(getattr(tc, "chi_spent_total", 0.0) or 0.0)

        # Fork summary — best-effort via mempool.get_pending_forks
        fork_summary: list[dict[str, Any]] = []
        try:
            mempool = getattr(tc, "_mempool", None)
            if mempool is not None and hasattr(mempool, "get_pending_forks"):
                pending_forks = mempool.get_pending_forks() or []
                for fork_name in pending_forks[:7]:  # cap at 7
                    fork_summary.append({"name": str(fork_name)})
        except Exception:
            pass

        # Integrity status — best-effort via _last_integrity_status if set
        integrity_status = str(
            getattr(tc, "_last_integrity_status", "unknown"))

        return {
            "tx_latency_norm": tx_latency_norm,
            "block_delta_norm": block_delta_norm,
            "recent_anchor_age_s": round(recent_anchor_age_s, 2),
            "fork_summary": fork_summary,
            "integrity_status": integrity_status,
            "total_blocks": total_blocks,
            "chi_spent_total": round(chi_spent_total, 4),
            "tx_latency_samples": int(tx_lat.get("samples", 0) or 0),
            "tx_latency_median_s": float(tx_lat.get("median_s", 0.0) or 0.0),
            "tx_latency_p95_s": float(tx_lat.get("p95_s", 0.0) or 0.0),
            "ts": time.time(),
        }

    def _cold_boot_payload(self) -> dict[str, Any]:
        import time
        return {
            "tx_latency_norm": 0.5,
            "block_delta_norm": 0.5,
            "recent_anchor_age_s": 0.0,
            "fork_summary": [],
            "integrity_status": "unknown",
            "total_blocks": 0,
            "chi_spent_total": 0.0,
            "tx_latency_samples": 0,
            "tx_latency_median_s": 0.0,
            "tx_latency_p95_s": 0.0,
            "ts": time.time(),
        }
