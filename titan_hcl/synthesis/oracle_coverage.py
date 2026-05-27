"""Phase 6 — `scored_by` coverage analyzer (§P6.J; INV-Syn-15).

The §A.6 acceptance gate requires `scored_by` ≥ 95% on tool-call
procedural TXs. The instrumentation is split:

1. **Write-time** (P6.I `write_tool_call`): every tool invocation lands
   a procedural-fork TX with ``content.scored_by`` initialized to None.
2. **Companion-verdict-time** (P6.F `OracleRouter.flush_companion_batches`):
   each OracleVerdictBatch TX carries ``parent_tool_call_tx`` per entry
   — so the chain itself records WHICH tool calls got an oracle verdict.
3. **Read-time** (this module): the `CoverageAnalyzer` walks the chain
   (or its index) and computes the coverage ratio by joining tool-call
   TX hashes against batch parent_tool_call_tx references.

The model is *retrospective* — `scored_by` is computed at read time by
joining two chain-anchored streams, not by mutating the original TX
(which would violate the chain immutability). This is the same pattern
P5 uses for fork-graduation status (queried via `concept:<id>:v<n>`
tags).

The analyzer takes injected readers so unit tests are deterministic:
- ``tool_call_reader(since_ts) -> list[dict]`` — returns tool_call TXs
  in window. Each dict carries at least ``tx_hash``, ``scored_by``
  (write-time value; usually None).
- ``batch_reader(since_ts) -> list[dict]`` — returns OracleVerdictBatch
  TXs in window. Each carries ``entries: [{parent_tool_call_tx, ...}]``.

Phase 8's skill miner will also consume this surface (per arch §8.4 +
SPEC §25.1 INV-Syn-15: "Phase 8 skill miner consumes scored_by for
skill-candidate weighting").
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ``tool_call_reader(since_ts: float, limit: int) -> list[dict]``
# Returns tool_call TXs in window. Dict shape:
#   {"tx_hash": str, "scored_by": "oracle"|"llm"|None, "ts": float,
#    "tool_id": str, ...}
ToolCallReader = Callable[[float, int], list[dict]]

# ``batch_reader(since_ts: float, limit: int) -> list[dict]``
# Returns OracleVerdictBatch TXs in window. Dict shape:
#   {"tx_hash": str, "ts": float, "entries": [{"parent_tool_call_tx": ..., ...}]}
BatchReader = Callable[[float, int], list[dict]]


def _default_tool_call_reader(since_ts: float, limit: int) -> list[dict]:
    return []


def _default_batch_reader(since_ts: float, limit: int) -> list[dict]:
    return []


@dataclass
class CoverageReport:
    """The §A.6 measurable surface — returned by analyze() and exposed
    via Observatory ``/v6/synthesis/oracles/coverage`` (P6.K)."""

    window_seconds: float
    as_of: float
    total_tool_call_txs: int
    scored_by_oracle: int          # write-time "oracle" + retrospective oracle-batch joins
    scored_by_llm: int             # Phase 8 will populate this
    unscored: int
    coverage_ratio: float          # (oracle + llm) / total; 0.0 if total==0


class CoverageAnalyzer:
    """Computes the §A.6 coverage ratio over a sliding window.

    The analyzer is **stateless** between calls; the synthesis_worker
    constructs one at boot and calls ``analyze(window_s)`` from the
    Observatory route handler (P6.K) on each request. Cheap enough to
    run per-request — typically O(N) over N tool-call TXs in the
    window (default 30 days = ~thousands of TXs).
    """

    def __init__(
        self,
        *,
        tool_call_reader: ToolCallReader = _default_tool_call_reader,
        batch_reader: BatchReader = _default_batch_reader,
        now_fn: Callable[[], float] = time.time,
        default_window_seconds: float = 30 * 24 * 3600.0,
        default_limit: int = 10_000,
    ):
        self._tool_call_reader = tool_call_reader
        self._batch_reader = batch_reader
        self._now_fn = now_fn
        self._default_window_seconds = float(default_window_seconds)
        self._default_limit = int(default_limit)

    def analyze(
        self,
        *,
        window_seconds: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> CoverageReport:
        """Compute the coverage report over the window.

        Joins write-time ``scored_by`` (set at TX write) with
        retrospective oracle-batch joins (any tool_call TX referenced as
        ``parent_tool_call_tx`` in an OracleVerdictBatch TX in the same
        window is scored "oracle"). LLM-scored will land via the Phase
        8 skill miner's follow-up TX shape; Phase 6 only counts what's
        already on the chain.
        """
        now = self._now_fn()
        window = float(window_seconds if window_seconds is not None else self._default_window_seconds)
        cap = int(limit if limit is not None else self._default_limit)
        since_ts = now - window

        # 1. Tool-call TXs in window
        try:
            tool_calls = list(self._tool_call_reader(since_ts, cap) or [])
        except Exception:
            logger.exception("[coverage] tool_call_reader raised")
            tool_calls = []

        # 2. Oracle-batch TXs in window → set of parent_tool_call_tx hashes
        try:
            batches = list(self._batch_reader(since_ts, cap) or [])
        except Exception:
            logger.exception("[coverage] batch_reader raised")
            batches = []

        oracle_scored_via_batch: set[str] = set()
        for batch in batches:
            entries = batch.get("entries") if isinstance(batch, dict) else None
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                # Only count verdicts that actually fired (verdict ∈
                # {true, false}); "unknown" verdicts are NOT an
                # oracle-scored signal (per arch §11.1 "verdict is true,
                # false, OR unknown" — the latter is non-evidence).
                verdict = str(entry.get("verdict", "unknown")).lower()
                if verdict not in ("true", "false"):
                    continue
                ref = entry.get("parent_tool_call_tx")
                if isinstance(ref, str) and ref:
                    oracle_scored_via_batch.add(ref)

        # 3. Classify each tool_call
        total = len(tool_calls)
        oracle_n = 0
        llm_n = 0
        unscored_n = 0
        for tc in tool_calls:
            tx_hash = tc.get("tx_hash") if isinstance(tc, dict) else None
            write_time_scored = tc.get("scored_by") if isinstance(tc, dict) else None
            # Retrospective oracle path joins first
            if tx_hash in oracle_scored_via_batch:
                oracle_n += 1
                continue
            if write_time_scored == "oracle":
                oracle_n += 1
            elif write_time_scored == "llm":
                llm_n += 1
            else:
                unscored_n += 1

        scored_total = oracle_n + llm_n
        ratio = (scored_total / total) if total > 0 else 0.0

        return CoverageReport(
            window_seconds=window,
            as_of=now,
            total_tool_call_txs=total,
            scored_by_oracle=oracle_n,
            scored_by_llm=llm_n,
            unscored=unscored_n,
            coverage_ratio=ratio,
        )

    def report_dict(
        self,
        *,
        window_seconds: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Convenience for the Observatory route — returns a plain
        dict ready to serialize as JSON."""
        r = self.analyze(window_seconds=window_seconds, limit=limit)
        return {
            "window_seconds": r.window_seconds,
            "as_of": r.as_of,
            "total_tool_call_txs": r.total_tool_call_txs,
            "scored_by_oracle": r.scored_by_oracle,
            "scored_by_llm": r.scored_by_llm,
            "unscored": r.unscored,
            "coverage_ratio": r.coverage_ratio,
            "a6_gate_passes": r.coverage_ratio >= 0.95,
        }


__all__ = (
    "CoverageAnalyzer",
    "CoverageReport",
    "ToolCallReader",
    "BatchReader",
)
