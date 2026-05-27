"""Phase 6 — Oracle snapshot exporter (§P6.K).

Writes ``data/oracles_snapshot.json`` for the api process to read.
Mirrors the P4 ``spine_snapshot.json`` + P5 ``forks_snapshot.json``
patterns: synthesis_worker is the sole writer (INV-Syn-3 / G21);
cross-process readers (Observatory endpoints) consume an atomic JSON
snapshot, never the live in-memory state.

Snapshot schema (version 1):
    {
      "version": 1,
      "exported_at": <wall-clock seconds>,
      "router": [{oracle_id, cost_class}, ...],
      "budget": {date: <YYYY-MM-DD>, per_oracle: [{oracle_id,
                 spent_sol, n_calls, daily_budget_sol, remaining_sol}, ...]},
      "coverage": {window_seconds, total_tool_call_txs, scored_by_oracle,
                   scored_by_llm, unscored, coverage_ratio, a6_gate_passes},
      "recent_verdicts": [{tx_hash, ts, oracle_id, verdict,
                            claim_domain, evidence_ref, cost, fork}, ...],
      "recent_proofs": [{ts, strategy, commitment_hex, payload_ref, cost}, ...]
    }

Synthesis_worker calls ``export(...)`` from its 60s recompute loop +
on demand after any router state change.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from titan_hcl.synthesis.oracle_coverage import CoverageAnalyzer
    from titan_hcl.synthesis.oracle_gate import OracleGateConfig
    from titan_hcl.synthesis.oracle_router import OracleRouter, OracleSpendStore

logger = logging.getLogger(__name__)


DEFAULT_SNAPSHOT_PATH = "data/oracles_snapshot.json"
SNAPSHOT_VERSION = 1


def resolve_snapshot_path() -> str:
    data_dir = os.environ.get("TITAN_DATA_DIR", "data")
    return os.path.join(data_dir, "oracles_snapshot.json")


class OracleSnapshotExporter:
    """Atomic JSON writer for ``oracles_snapshot.json``.

    Constructed by synthesis_worker at boot with the router + spend
    store + gate config + coverage analyzer. ``export()`` writes a
    fresh snapshot to disk via tmp+rename so cross-process readers
    never see a partial file.
    """

    def __init__(
        self,
        *,
        router: "OracleRouter",
        spend_store: "OracleSpendStore",
        gate_config: "OracleGateConfig",
        coverage_analyzer: Optional["CoverageAnalyzer"] = None,
        snapshot_path: Optional[str] = None,
        recent_verdict_buffer: Optional[list] = None,
        recent_proof_buffer: Optional[list] = None,
        max_recent: int = 50,
    ):
        self._router = router
        self._spend = spend_store
        self._gate_config = gate_config
        self._coverage = coverage_analyzer
        self._snapshot_path = snapshot_path or resolve_snapshot_path()
        # Optional in-memory ring buffers fed by router instrumentation +
        # proof commits (synthesis_worker maintains these). If not
        # provided, snapshot's recent_verdicts / recent_proofs stay empty.
        self._recent_verdicts = recent_verdict_buffer if recent_verdict_buffer is not None else []
        self._recent_proofs = recent_proof_buffer if recent_proof_buffer is not None else []
        self._max_recent = int(max_recent)

    def build_payload(self, *, now: Optional[float] = None) -> dict:
        """Construct the snapshot payload in-memory. Pure (no disk I/O)
        so unit tests can assert shape without filesystem fixtures."""
        now = now or time.time()
        # Router registry
        router_listing = list(self._router.registered_oracles())
        # Budget — per-oracle spent_today + remaining
        spend_snapshot = self._spend.export_snapshot(now=now)
        per_oracle_budget = []
        seen_oracles = set()
        for row in spend_snapshot.get("per_oracle", []):
            oracle_id = row.get("oracle_id")
            seen_oracles.add(oracle_id)
            budget = self._gate_config.daily_budget_for(oracle_id)
            spent = float(row.get("spent_sol", 0.0))
            per_oracle_budget.append({
                "oracle_id": oracle_id,
                "spent_sol": spent,
                "n_calls": int(row.get("n_calls", 0)),
                "daily_budget_sol": budget,
                "remaining_sol": max(0.0, budget - spent),
            })
        # Also surface registered oracles with no spend today (so the UI
        # can show 0/budget rather than missing rows).
        for oracle in router_listing:
            oid = oracle.get("oracle_id")
            if oid in seen_oracles or oracle.get("cost_class") == "free":
                continue
            budget = self._gate_config.daily_budget_for(oid)
            per_oracle_budget.append({
                "oracle_id": oid,
                "spent_sol": 0.0,
                "n_calls": 0,
                "daily_budget_sol": budget,
                "remaining_sol": budget,
            })

        # Coverage (§A.6)
        coverage: dict = {}
        if self._coverage is not None:
            try:
                coverage = self._coverage.report_dict()
            except Exception:
                logger.exception("[oracle_snapshot] coverage report failed")
                coverage = {}

        return {
            "version": SNAPSHOT_VERSION,
            "exported_at": now,
            "router": router_listing,
            "budget": {
                "date": spend_snapshot.get("date"),
                "per_oracle": per_oracle_budget,
            },
            "coverage": coverage,
            "recent_verdicts": list(self._recent_verdicts[-self._max_recent:]),
            "recent_proofs": list(self._recent_proofs[-self._max_recent:]),
        }

    def export(self, *, now: Optional[float] = None) -> bool:
        """Atomic tmp+rename write. Returns True on success, False on
        any failure (logs WARN; synthesis_worker continues — snapshot
        export is best-effort by design)."""
        try:
            payload = self.build_payload(now=now)
        except Exception:
            logger.exception("[oracle_snapshot] build_payload failed")
            return False
        try:
            os.makedirs(
                os.path.dirname(self._snapshot_path) or ".", exist_ok=True,
            )
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self._snapshot_path) or ".",
                prefix=".oracles_snapshot_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, separators=(",", ":"))
                os.replace(tmp, self._snapshot_path)
                return True
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            logger.exception("[oracle_snapshot] write failed")
            return False

    def record_verdict(
        self,
        *,
        tx_hash: Optional[str],
        oracle_id: str,
        verdict: str,
        claim_domain: str,
        evidence_ref: str,
        cost: float,
        fork: str,
        ts: Optional[float] = None,
    ) -> None:
        """Append a verdict to the recent-verdicts ring (synthesis_worker
        calls this after each OracleRouter.verify() that anchored a
        standalone verdict). Keeps last ``max_recent`` entries."""
        self._recent_verdicts.append({
            "tx_hash": tx_hash,
            "ts": ts or time.time(),
            "oracle_id": oracle_id,
            "verdict": verdict,
            "claim_domain": claim_domain,
            "evidence_ref": evidence_ref,
            "cost": float(cost),
            "fork": fork,
        })
        # Bound the buffer in-place so memory stays bounded across long soaks.
        if len(self._recent_verdicts) > self._max_recent * 2:
            del self._recent_verdicts[: -self._max_recent]

    def record_proof(
        self,
        *,
        strategy: str,
        commitment_hex: str,
        payload_ref: Optional[str],
        cost: float,
        ts: Optional[float] = None,
    ) -> None:
        """Append a proof commit to the recent-proofs ring."""
        self._recent_proofs.append({
            "ts": ts or time.time(),
            "strategy": strategy,
            "commitment_hex": commitment_hex,
            "payload_ref": payload_ref,
            "cost": float(cost),
        })
        if len(self._recent_proofs) > self._max_recent * 2:
            del self._recent_proofs[: -self._max_recent]


__all__ = (
    "OracleSnapshotExporter",
    "DEFAULT_SNAPSHOT_PATH",
    "SNAPSHOT_VERSION",
    "resolve_snapshot_path",
)
