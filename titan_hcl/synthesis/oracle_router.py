"""Phase 6 — OracleRouter (§P6.F).

The synthesis_worker-owned orchestration layer for `TruthOraclePlug`
invocations. Per SPEC §25.1 INV-Syn-12 + INV-Syn-13:

- **Routes** ``OracleClaim.domain → plug.can_handle()`` selecting which
  registered `TruthOraclePlug` handles the claim.
- **Applies** the INV-Syn-13 metabolic gate (via `OracleGate` from
  §P6.A) before any metered plug fires; gate-denied claims STILL get
  an anchored ``OracleVerdict(verdict="unknown", evidence_ref=
  "metabolic_gate_denied"|"daily_budget_exhausted")`` per the
  "rejection is auditable" rule.
- **Tracks** per-day spend in ``data/synthesis.duckdb ::
  oracle_daily_spend`` (sole writer = synthesis_worker per INV-Syn-3
  / G21).
- **Anchors** the verdict per INV-Syn-12 routing:
    - **Standalone** verdicts (caller did NOT pass a
      ``parent_tool_call_tx``) ride the canonical fork that matches
      the claim domain (table below).
    - **Tool-call companion** verdicts (caller passed a
      ``parent_tool_call_tx``) are buffered + Merkle-batched into a
      single ``OracleVerdictBatch`` TX per dream window per fork
      (per §16.1 micro-event tier; bounds chain growth per §B.7).
- **Is the sole emitter** of standalone OracleVerdict TXs — INV-4
  single-canonical-write-path extends through the router via
  ``OuterMemoryWriter``.

Construction takes injected dependencies so unit tests can stub
each independently:
- ``gate``: an ``OracleGate`` (already configured from §P6.A).
- ``spend_store``: an ``OracleSpendStore`` reading/writing
  ``oracle_daily_spend``.
- ``outer_memory_writer``: the synthesis_worker's
  ``OuterMemoryWriter`` instance (sole canonical write path —
  INV-4).
- ``balance_provider``: callable ``() -> float`` returning current
  Titan SOL balance (the worker passes a closure over
  ``network.balance`` typically).
- ``now_fn``: callable ``() -> float`` returning wall-clock epoch
  seconds (injectable for deterministic tests around the dream-
  window batch flush cadence).
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from titan_hcl.synthesis.merkle import merkle_root_hex
from titan_hcl.synthesis.oracle_gate import (
    DENY_REASON_BUDGET,
    DENY_REASON_THRESHOLD,
    GateDecision,
    OracleGate,
    ensure_oracle_daily_spend_table,
)
from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict, TruthOraclePlug

if TYPE_CHECKING:
    from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# INV-Syn-12 — claim domain → canonical fork routing table.
# Tool-call companion verdicts override this and ride the parent fork.
# ─────────────────────────────────────────────────────────────────────────

CLAIM_DOMAIN_TO_FORK: dict[str, str] = {
    # P6.B coding_sandbox
    "code_correctness": "procedural",
    "math_correctness": "procedural",
    # P6.C solana_rpc
    "solana_tx_confirmed": "procedural",
    "solana_account_balance_gte": "procedural",
    "solana_program_invoked": "procedural",
    # P6.D web_api
    "web_fact": "declarative",
    "wiki_fact": "declarative",
    # P6.E x_oracle
    "topic_trending": "declarative",
    "account_exists": "declarative",
    "post_real": "declarative",
    "x_event_real": "declarative",
    # Future: episodic claims from chat
    "chat_factual_claim": "episodic",
}


# Default fork when the router sees a claim domain not in the table
# (e.g. a plug registers a brand-new domain via can_handle pre-route-table
# update). Choose "meta" — symmetric with tombstone routing per INV-Syn-9.
_UNKNOWN_DOMAIN_FORK: str = "meta"


# ─────────────────────────────────────────────────────────────────────────
# OracleSpendStore — sole writer to oracle_daily_spend (INV-Syn-3 / G21)
# ─────────────────────────────────────────────────────────────────────────


class OracleSpendStore:
    """Wraps the ``oracle_daily_spend`` table on ``data/synthesis.duckdb``.

    All writes are serialized through an internal lock so the router
    can be invoked from multiple coroutines/threads inside the
    synthesis_worker process; INV-Syn-3 sole-writer is preserved
    because only this process holds the DB handle.
    """

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()
        # Ensure the table exists — idempotent CREATE.
        ensure_oracle_daily_spend_table(conn)

    @staticmethod
    def _today(now: float) -> str:
        """UTC YYYY-MM-DD for grouping; matches DuckDB DATE literal."""
        return datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    def spent_today(self, oracle_id: str, *, now: Optional[float] = None) -> float:
        """SOL spent on ``oracle_id`` for today (UTC). Zero if no row."""
        now = now or time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT spent_sol FROM oracle_daily_spend "
                "WHERE oracle_id = ? AND spend_date = ?",
                (oracle_id, self._today(now)),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def record_spend(
        self,
        oracle_id: str,
        cost_sol: float,
        *,
        now: Optional[float] = None,
    ) -> float:
        """UPSERT today's row; return the new cumulative ``spent_sol``.

        ``cost_sol = 0.0`` is a valid call (free oracle / gate-denied path
        still bumps ``n_calls`` so the audit log shows the attempt).
        """
        now = now or time.time()
        today = self._today(now)
        with self._lock:
            row = self._conn.execute(
                "SELECT spent_sol, n_calls FROM oracle_daily_spend "
                "WHERE oracle_id = ? AND spend_date = ?",
                (oracle_id, today),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO oracle_daily_spend "
                    "(oracle_id, spend_date, spent_sol, n_calls, updated_at) "
                    "VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)",
                    (oracle_id, today, float(cost_sol)),
                )
                return float(cost_sol)
            new_total = float(row[0]) + float(cost_sol)
            new_calls = int(row[1]) + 1
            self._conn.execute(
                "UPDATE oracle_daily_spend "
                "SET spent_sol = ?, n_calls = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE oracle_id = ? AND spend_date = ?",
                (new_total, new_calls, oracle_id, today),
            )
            return new_total

    def export_snapshot(self, *, now: Optional[float] = None) -> dict:
        """Today's per-oracle spend snapshot for the Observatory
        ``/v6/synthesis/oracles/budget`` route (P6.K)."""
        now = now or time.time()
        today = self._today(now)
        with self._lock:
            rows = self._conn.execute(
                "SELECT oracle_id, spent_sol, n_calls FROM oracle_daily_spend "
                "WHERE spend_date = ?",
                (today,),
            ).fetchall()
        return {
            "as_of": now,
            "date": today,
            "per_oracle": [
                {"oracle_id": r[0], "spent_sol": float(r[1]), "n_calls": int(r[2])}
                for r in rows
            ],
        }


# ─────────────────────────────────────────────────────────────────────────
# Companion-batch buffer — INV-Syn-12 tool-call companion verdicts
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _CompanionEntry:
    """One buffered (parent_tool_call_tx_hash, verdict_hash) leaf for the
    next ``OracleVerdictBatch`` TX flush. ``verdict`` is the full verdict
    payload — included so the flushed batch carries enough audit detail
    (oracle_id, evidence_ref, cost) without needing to walk CAS.
    """

    parent_tool_call_tx: str
    verdict: OracleVerdict
    fork: str           # parent tool-call's fork — INV-Syn-12 routing


@dataclass
class _CompanionBuffer:
    """In-memory buffer per (fork). Flushed at dream boundary by the
    synthesis_worker into one ``OracleVerdictBatch`` TX per fork.
    """

    by_fork: dict[str, list[_CompanionEntry]] = field(default_factory=lambda: defaultdict(list))
    lock: threading.Lock = field(default_factory=threading.Lock)


# ─────────────────────────────────────────────────────────────────────────
# OracleRouter — the orchestration layer
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class RouterResult:
    """What the router returns to the caller — both the verdict (for any
    downstream logic the caller may want) AND the anchored TX hash
    (so the caller can FORK_READ it back, attach it to a procedural
    record, etc.). For tool-call companion path: ``anchor_tx`` is None
    until the dream-boundary batch flush emits the rollup TX.
    """

    verdict: OracleVerdict
    gate_decision: GateDecision
    anchor_tx: Optional[str]    # standalone: TX hash; companion: None (anchored at flush)
    fork: str                   # where the verdict landed (or will land for companion)


class OracleRouter:
    """Sole emitter of standalone OracleVerdict TXs (INV-Syn-12, extending INV-4).

    Invoked by:
    - the synthesis_worker's verify-claim entry point (standalone)
    - the ToolPlug invocation path in P6.I (tool-call companion)
    - the §P5 graduation path (NOT this router — Phase 5 uses
      ``OuterMemoryWriter.write_concept_version_with_proof`` directly;
      Phase 6 does NOT change that path).
    """

    def __init__(
        self,
        *,
        gate: OracleGate,
        spend_store: OracleSpendStore,
        outer_memory_writer: "OuterMemoryWriter",
        balance_provider: Callable[[], float],
        now_fn: Callable[[], float] = time.time,
    ):
        self._gate = gate
        self._spend_store = spend_store
        self._writer = outer_memory_writer
        self._balance_provider = balance_provider
        self._now_fn = now_fn

        # Registered plugs indexed by oracle_id. The router asks each plug
        # `can_handle(domain)` at verify time — domain→plug is dynamic so
        # plugs can be hot-swapped (P4.K future).
        self._plugs: dict[str, TruthOraclePlug] = {}

        # Companion-batch buffer for tool-call verdicts (INV-Syn-12).
        self._companions = _CompanionBuffer()

    # ── registry ────────────────────────────────────────────────────────

    def register(self, plug: TruthOraclePlug) -> None:
        """Register a TruthOraclePlug instance. ``oracle_id`` must be unique
        (registering the same id twice overwrites — last-write-wins; the
        synthesis_worker registers each plug exactly once at boot, so
        this is intentional simplicity)."""
        self._plugs[plug.oracle_id] = plug
        logger.info(
            "[OracleRouter] registered plug oracle_id=%s cost_class=%s",
            plug.oracle_id, getattr(plug, "cost_class", "unknown"),
        )

    def registered_oracles(self) -> list[dict]:
        """Snapshot for ``/v6/synthesis/oracles/router`` (P6.K)."""
        return [
            {
                "oracle_id": p.oracle_id,
                "cost_class": getattr(p, "cost_class", "unknown"),
            }
            for p in self._plugs.values()
        ]

    # ── verify entry ────────────────────────────────────────────────────

    def verify(
        self,
        claim: OracleClaim,
        *,
        parent_tool_call_tx: Optional[str] = None,
        parent_tool_call_fork: Optional[str] = None,
    ) -> RouterResult:
        """Run a claim through gate + plug + verdict emission.

        ``parent_tool_call_tx`` set → tool-call companion path (verdict
        buffered for batch emission at dream boundary). ``parent_tool_call_
        fork`` is the parent tool-call TX's fork (typically ``procedural``
        for tool calls); the batched verdict rides the same fork per
        INV-Syn-12. Defaults to ``procedural`` when not given.

        ``parent_tool_call_tx`` unset → standalone path (verdict anchored
        immediately on the fork determined by claim.domain).
        """
        # ── 1. plug selection ──
        plug = self._select_plug(claim.domain)
        if plug is None:
            return self._anchor_unsupported(claim, parent_tool_call_tx, parent_tool_call_fork)

        # ── 2. metabolic gate (INV-Syn-13) ──
        oracle_id = plug.oracle_id
        balance = float(self._balance_provider() or 0.0)
        # remaining_daily_sol pulled from the spend ledger
        budget = self._gate.config.daily_budget_for(oracle_id)
        spent = self._spend_store.spent_today(oracle_id, now=self._now_fn())
        remaining = max(0.0, budget - spent)

        decision = self._gate.admit(
            claim, plug,
            balance_sol=balance,
            remaining_daily_sol=remaining,
        )

        # ── 3. fire plug if admitted; else build deny verdict ──
        if decision.admit:
            try:
                verdict = plug.verify(claim)
            except Exception:
                logger.exception(
                    "[OracleRouter] plug %s.verify() raised; emitting unknown verdict",
                    oracle_id,
                )
                verdict = OracleVerdict(
                    oracle_id=oracle_id,
                    verdict="unknown",
                    evidence_ref="plug_exception",
                    cost=0.0,
                    latency_ms=0,
                    ts=self._now_fn(),
                )
        else:
            # Gate denied — anchored unknown verdict carries the reason.
            verdict = OracleVerdict(
                oracle_id=oracle_id,
                verdict="unknown",
                evidence_ref=decision.reason or DENY_REASON_THRESHOLD,
                cost=0.0,
                latency_ms=decision.latency_ms,
                ts=self._now_fn(),
            )

        # ── 4. spend bookkeeping ──
        # Record even on free oracles + denies (so n_calls audits the
        # attempt rate, useful for soak diagnosis).
        try:
            self._spend_store.record_spend(oracle_id, verdict.cost, now=self._now_fn())
        except Exception:
            logger.exception("[OracleRouter] record_spend failed (oracle=%s)", oracle_id)

        # ── 5. route + anchor per INV-Syn-12 ──
        if parent_tool_call_tx is not None:
            return self._buffer_companion(verdict, parent_tool_call_tx, parent_tool_call_fork)
        return self._anchor_standalone(verdict, claim)

    # ── routing helpers ─────────────────────────────────────────────────

    def _select_plug(self, domain: str) -> Optional[TruthOraclePlug]:
        for p in self._plugs.values():
            try:
                if p.can_handle(domain):
                    return p
            except Exception:
                logger.exception(
                    "[OracleRouter] plug %s.can_handle(%r) raised — skipping",
                    p.oracle_id, domain,
                )
        return None

    def _anchor_unsupported(
        self,
        claim: OracleClaim,
        parent_tool_call_tx: Optional[str],
        parent_tool_call_fork: Optional[str],
    ) -> RouterResult:
        """No registered plug handles this domain — anchor an unknown
        verdict so the unsupported attempt is auditable (still per INV-4
        single canonical write path)."""
        verdict = OracleVerdict(
            oracle_id="router",
            verdict="unknown",
            evidence_ref="no_plug_for_domain",
            cost=0.0,
            latency_ms=0,
            ts=self._now_fn(),
        )
        decision = GateDecision(admit=False, reason="", admit_score=0.0, latency_ms=0)
        if parent_tool_call_tx is not None:
            return self._buffer_companion(verdict, parent_tool_call_tx, parent_tool_call_fork)
        # Standalone unsupported → ride the routing-table fork or meta default.
        fork = CLAIM_DOMAIN_TO_FORK.get(claim.domain, _UNKNOWN_DOMAIN_FORK)
        tx = self._writer.write_oracle_verdict_standalone(verdict=verdict, claim_domain=claim.domain, fork=fork)
        return RouterResult(verdict=verdict, gate_decision=decision, anchor_tx=tx, fork=fork)

    def _anchor_standalone(self, verdict: OracleVerdict, claim: OracleClaim) -> RouterResult:
        fork = CLAIM_DOMAIN_TO_FORK.get(claim.domain, _UNKNOWN_DOMAIN_FORK)
        try:
            tx = self._writer.write_oracle_verdict_standalone(
                verdict=verdict,
                claim_domain=claim.domain,
                fork=fork,
            )
        except Exception:
            logger.exception("[OracleRouter] standalone anchor failed")
            tx = None
        decision = GateDecision(admit=True, reason="", admit_score=0.0, latency_ms=0)
        return RouterResult(verdict=verdict, gate_decision=decision, anchor_tx=tx, fork=fork)

    def _buffer_companion(
        self,
        verdict: OracleVerdict,
        parent_tool_call_tx: str,
        parent_tool_call_fork: Optional[str],
    ) -> RouterResult:
        fork = parent_tool_call_fork or "procedural"
        with self._companions.lock:
            self._companions.by_fork[fork].append(
                _CompanionEntry(
                    parent_tool_call_tx=parent_tool_call_tx,
                    verdict=verdict,
                    fork=fork,
                )
            )
        decision = GateDecision(admit=True, reason="", admit_score=0.0, latency_ms=0)
        return RouterResult(verdict=verdict, gate_decision=decision, anchor_tx=None, fork=fork)

    def record_companion_verdict(
        self,
        *,
        parent_tool_call_tx: str,
        oracle_id: str,
        verdict: str,
        evidence_ref: str = "",
        cost: float = 0.0,
        latency_ms: int = 0,
        ts: Optional[float] = None,
        fork: Optional[str] = "procedural",
    ) -> None:
        """Buffer a PRE-COMPUTED companion verdict (operator-closure C2 / W7).

        For a tool that IS its own truth oracle (e.g. coding_sandbox — §11.1),
        the tool's execution already produced the verdict; re-running it through
        ``verify()`` would double-execute (and, for the sandbox, block up to 30s,
        violating G19). The chat-time tool emits its already-known verdict here
        (via the bus → synthesis_worker), and it is buffered for the same
        dream-boundary OracleVerdictBatch flush as router-produced verdicts —
        so chat-driven tool calls count toward §A.6 coverage. No plug re-run.
        """
        v = OracleVerdict(
            oracle_id=str(oracle_id),
            verdict=str(verdict),
            evidence_ref=str(evidence_ref),
            cost=float(cost),
            latency_ms=int(latency_ms),
            ts=float(ts if ts is not None else self._now_fn()),
        )
        self._buffer_companion(v, str(parent_tool_call_tx), fork or "procedural")

    # ── companion batch flush (called at dream boundary) ────────────────

    def flush_companion_batches(self) -> dict[str, str]:
        """Emit one ``OracleVerdictBatch`` TX per fork that has buffered
        entries. Returns ``{fork_name: anchor_tx_hash}`` for the synthesis_
        worker's audit log.

        Called from the synthesis_worker's DREAM_STATE_CHANGED dreaming=True
        handler (mirrors P4 ConsolidationPass + P5 ForkGC cadence).
        """
        with self._companions.lock:
            # Snapshot + drain atomically.
            snapshot = {k: v[:] for k, v in self._companions.by_fork.items() if v}
            self._companions.by_fork.clear()

        anchored: dict[str, str] = {}
        for fork, entries in snapshot.items():
            if not entries:
                continue
            leaves = [
                hashlib.sha256(
                    (e.parent_tool_call_tx + ":" + e.verdict.evidence_ref).encode("utf-8")
                ).hexdigest()
                for e in entries
            ]
            merkle_root = merkle_root_hex(leaves)
            try:
                tx = self._writer.write_oracle_verdict_batch(
                    fork=fork,
                    merkle_root=merkle_root,
                    entries=[
                        {
                            "parent_tool_call_tx": e.parent_tool_call_tx,
                            "oracle_id": e.verdict.oracle_id,
                            "verdict": e.verdict.verdict,
                            "evidence_ref": e.verdict.evidence_ref,
                            "cost": e.verdict.cost,
                            "ts": e.verdict.ts,
                        }
                        for e in entries
                    ],
                )
            except Exception:
                logger.exception("[OracleRouter] companion batch flush failed (fork=%s)", fork)
                continue
            anchored[fork] = tx
            logger.info(
                "[OracleRouter] flushed companion batch fork=%s n=%d merkle=%s tx=%s",
                fork, len(entries), merkle_root[:16], tx[:16],
            )
        return anchored

    def companion_buffer_size(self) -> int:
        """Total buffered companion entries across all forks (Observatory)."""
        with self._companions.lock:
            return sum(len(v) for v in self._companions.by_fork.values())


__all__ = (
    "OracleRouter",
    "OracleSpendStore",
    "RouterResult",
    "CLAIM_DOMAIN_TO_FORK",
)
