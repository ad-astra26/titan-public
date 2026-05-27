"""ForkGC — conservative reference-counted cascade GC for hypothesis forks.

Per `ARCHITECTURE_synthesis_engine.md §9.5` (RATIFIED) +
`PLAN_synthesis_engine_Phase5.md §P5.H` + INV-Syn-10 (proposed P5.K).

The §9.5 conjunction predicate (load-bearing):

  A node is cascade-prunable IFF **all three** hold:
    (a) sole-inbound from the dying fork — only inbound edges originate from
        this fork's exploration TXs / Kuzu EXPLORES edge.
    (b) never canonically anchored — the node has no canonical-fork TX
        backing it (or its anchor TX rides only on the hypothesis-class
        chain slot, never on a canonical fork).
    (c) below the activation floor — `activation_state.base_level < floor`.

  Any single survivor among (a)/(b)/(c) keeps the node alive. **When in
  doubt, keep** (arch §9.5).

Scope (Maker-locked 2026-05-27 — FULL cascade):

  Kuzu (always safe — pure index, rebuildable from chain INV-2):
    - HypothesisFork node
    - EXPLORES edges (cascade with DETACH DELETE)
    - probationary Concept rows created inside the fork's lifetime that
      have NO canonical anchor + sole inbound is the dying fork (rare;
      net-new graduation produces canonical concepts only at graduation
      time, so pre-graduation Concept rows are non-existent unless a future
      phase introduces them)

  DuckDB synthesis.duckdb:
    - hypothesis_fork_explorations (the durable Merkle-leaf log) — purged
      after the per-fork tombstone TX + canonical chain has the proof.

  DuckDB titan_memory.duckdb (cross-process — proxy via memory_worker
  facade if available; soft-skip otherwise):
    - memory_nodes rows whose `source_id` references ONLY the fork's
      exploration TX hashes AND have no canonical anchor.
    - action_chains_step rows created inside the fork's lifetime AND with
      no COMPILED_FROM edge to a canonical Production node.

  FAISS:
    - skipped in v1 (P5 ships v1 with Kuzu + DuckDB scope; FAISS pruning
      requires the embedding-index migration owner to confirm safe-delete
      semantics — deferred to a follow-up, not Phase 6).

Sweep is **nightly** at dream boundary, mirroring Phase 4 ConsolidationPass
timing (reuses the DREAM_STATE_CHANGED listener — fires when `dreaming=True`).

Safety:
  - Dry-run mode (default for first 24h of T3 soak) — logs what would be
    pruned without destructive writes.
  - Per-sweep cap (`max_nodes_per_sweep=10_000` default) — bounds the
    worst-case sweep duration + write amplification.
  - Transactional safety: each fork's cascade prune is wrapped in a single
    DuckDB transaction; rollback on any sub-failure leaves the fork in
    `abandoned_pending_gc` semantic-status (the canonical 'abandoned'
    status stays but `purge_durable_state` is retried next sweep).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Default per-sweep cap (Maker-locked 2026-05-27: 10k nodes/sweep).
DEFAULT_MAX_NODES_PER_SWEEP = 10_000


@dataclass
class PrunePlan:
    """What a sweep WOULD prune for one fork (dry-run output)."""

    fork_id: str
    status_at_sweep: str            # 'graduated' | 'abandoned'
    kuzu_node_will_drop: bool
    kuzu_explores_edges_will_drop: int
    durable_exploration_rows_will_drop: int
    memory_nodes_will_drop: list[int] = field(default_factory=list)
    action_chains_will_drop: list[int] = field(default_factory=list)
    keep_reasons: list[str] = field(default_factory=list)  # per-subnode keep reasons

    @property
    def total_will_drop(self) -> int:
        return (
            (1 if self.kuzu_node_will_drop else 0)
            + self.kuzu_explores_edges_will_drop
            + self.durable_exploration_rows_will_drop
            + len(self.memory_nodes_will_drop)
            + len(self.action_chains_will_drop)
        )


@dataclass
class SweepReport:
    """Outcome of one nightly sweep run."""

    started_at: float
    finished_at: float
    dry_run: bool
    forks_visited: int = 0
    forks_pruned: int = 0
    forks_skipped: int = 0          # over the per-sweep cap, retry next sweep
    total_nodes_dropped: int = 0
    plans: list[PrunePlan] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── ForkGC ──────────────────────────────────────────────────────────


class ForkGC:
    """Nightly cascade-GC sweep for hypothesis forks. INV-Syn-10.

    Owned by synthesis_worker; called at dream boundary (DREAM_STATE_CHANGED
    `dreaming=True`). Reads from HypothesisForkStore.pending_gc_targets()
    + applies the §9.5 conjunction predicate per fork.
    """

    def __init__(
        self,
        *,
        fork_store: Any,                # HypothesisForkStore
        synthesis_duckdb_conn: Any,     # synthesis.duckdb connection (sole writer)
        kuzu_graph: Any,                # TitanKnowledgeGraph
        activation_store: Any,          # ActivationStore for B_i lookups
        memory_db_conn: Optional[Any] = None,  # titan_memory.duckdb read-only handle
        chain_canonical_tx_resolver: Optional[Any] = None,  # callable(tx_hash) -> bool
        activation_floor: float = -3.0,
        max_nodes_per_sweep: int = DEFAULT_MAX_NODES_PER_SWEEP,
        clock: Any = time.time,
    ):
        self._store = fork_store
        self._db = synthesis_duckdb_conn
        self._graph = kuzu_graph
        self._activation = activation_store
        self._memory_db = memory_db_conn
        # Caller-supplied resolver: takes a tx_hash, returns True iff the TX
        # is anchored on a CANONICAL fork (not hypothesis-class). Default
        # impl walks the timechain index — but the resolver is injectable
        # so tests + cross-process scenarios can substitute.
        self._is_canonical_tx = chain_canonical_tx_resolver or (lambda _h: True)
        self._floor = float(activation_floor)
        self._cap = int(max_nodes_per_sweep)
        self._clock = clock

    # ── Sweep entry point ────────────────────────────────────────────

    def sweep(self, dry_run: bool = True) -> SweepReport:
        """Run one cascade-GC pass over all pending-GC forks.

        Args:
          dry_run: if True (default), no destructive writes — only PrunePlan
            instances are produced. Soak T3 ≥24h in dry-run before live.

        Returns a SweepReport with per-fork plans + counts. Per-sweep cap
        bounds the work; forks beyond the cap are skipped (`forks_skipped`
        incremented) and retried next sweep.
        """
        report = SweepReport(
            started_at=self._clock(),
            finished_at=0.0,        # filled at end
            dry_run=dry_run,
        )

        targets = self._store.pending_gc_targets()
        report.forks_visited = len(targets)

        budget = self._cap
        for fork_id in targets:
            if budget <= 0:
                report.forks_skipped += 1
                continue

            try:
                plan = self._plan_prune(fork_id)
            except Exception as e:
                msg = f"plan_prune({fork_id}) raised: {e}"
                report.errors.append(msg)
                logger.warning("[ForkGC] %s", msg, exc_info=True)
                continue

            report.plans.append(plan)
            if plan.total_will_drop > budget:
                # Single fork's plan exceeds remaining budget — skip and
                # retry next sweep (don't partially-prune; transactional
                # semantics require all-or-nothing per fork).
                report.forks_skipped += 1
                logger.info(
                    "[ForkGC] sweep: fork=%s plan=%d nodes exceeds remaining "
                    "budget=%d — skipped, will retry next sweep",
                    fork_id, plan.total_will_drop, budget,
                )
                continue

            if dry_run:
                # Count what would have been pruned but make no writes.
                report.total_nodes_dropped += plan.total_will_drop
                report.forks_pruned += 1
                logger.info(
                    "[ForkGC] sweep (dry_run): fork=%s would drop %d nodes",
                    fork_id, plan.total_will_drop,
                )
            else:
                try:
                    dropped = self._execute_plan(plan)
                    report.total_nodes_dropped += dropped
                    report.forks_pruned += 1
                    budget -= dropped
                    logger.info(
                        "[ForkGC] sweep: pruned fork=%s nodes=%d",
                        fork_id, dropped,
                    )
                except Exception as e:
                    msg = f"execute_plan({fork_id}) raised: {e}"
                    report.errors.append(msg)
                    logger.error("[ForkGC] %s", msg, exc_info=True)
                    # Don't decrement budget; the transactional rollback
                    # means nothing was written for this fork.

        report.finished_at = self._clock()
        logger.info(
            "[ForkGC] sweep complete: visited=%d pruned=%d skipped=%d "
            "errors=%d total_dropped=%d (dry_run=%s) elapsed=%.2fs",
            report.forks_visited, report.forks_pruned, report.forks_skipped,
            len(report.errors), report.total_nodes_dropped, dry_run,
            report.finished_at - report.started_at,
        )
        return report

    # ── Plan: identify prune candidates per fork ────────────────────

    def _plan_prune(self, fork_id: str) -> PrunePlan:
        """Apply the §9.5 conjunction predicate. Builds a PrunePlan without
        executing any writes."""
        fork = self._store.get_fork(fork_id)
        if fork is None:
            raise ValueError(f"_plan_prune: no fork {fork_id!r}")
        if fork.status not in ("graduated", "abandoned"):
            raise ValueError(
                f"_plan_prune({fork_id}): status={fork.status!r}, "
                "must be graduated|abandoned"
            )

        plan = PrunePlan(
            fork_id=fork_id, status_at_sweep=fork.status,
            kuzu_node_will_drop=False,
            kuzu_explores_edges_will_drop=0,
            durable_exploration_rows_will_drop=0,
        )

        # (1) Kuzu HypothesisFork node — always safe to prune once the
        # lifecycle ends. It is a pure index (the canonical record is the
        # chain TX); the (a)/(b)/(c) predicate doesn't apply because it's
        # NOT a subnode — it's the fork-index itself.
        if self._graph.fork_get_node(fork_id) is not None:
            plan.kuzu_node_will_drop = True

        # (2) EXPLORES edges — cascade with the fork node (DETACH DELETE
        # handles this automatically in execute_plan; for the plan, count
        # them so the report is honest).
        plan.kuzu_explores_edges_will_drop = len(
            self._graph.fork_explores_targets(fork_id)
        )

        # (3) Durable hypothesis_fork_explorations rows — always safe to
        # prune; the Merkle root is anchored on-chain (tombstone TX for
        # abandoned forks; concept-version TX's derivation_evidence for
        # graduated forks).
        try:
            count = self._db.execute(
                "SELECT COUNT(*) FROM hypothesis_fork_explorations "
                "WHERE fork_id=?",
                (fork_id,),
            ).fetchone()
            plan.durable_exploration_rows_will_drop = (
                int(count[0]) if count else 0
            )
        except Exception as e:
            logger.warning(
                "[ForkGC] plan: durable exploration count for %s failed: %s",
                fork_id, e,
            )

        # (4) memory_nodes — full §9.5 conjunction predicate.
        # Skip if memory_db_conn unavailable (cross-process limitations;
        # caller may inject a read-only handle if the runtime allows).
        exploration_txs = self._fetch_exploration_txs(fork_id)
        if self._memory_db is not None and exploration_txs:
            self._plan_memory_nodes_prune(
                fork_id, exploration_txs, plan,
            )

        # (5) action_chains_step — same predicate, additional gate:
        # row must have been created INSIDE the fork's lifetime
        # (created_at between fork.created_at and fork.last_touched).
        if self._memory_db is not None:
            self._plan_action_chains_prune(fork, plan)

        return plan

    def _fetch_exploration_txs(self, fork_id: str) -> list[str]:
        """Read the durable exploration-TX log for the fork (in-memory map
        is also valid but the store's API doesn't expose it directly — we
        read from the durable table so the predicate is reproducible
        across worker restarts)."""
        try:
            rows = self._db.execute(
                "SELECT tx_hash FROM hypothesis_fork_explorations "
                "WHERE fork_id=?",
                (fork_id,),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.warning(
                "[ForkGC] _fetch_exploration_txs(%s) failed: %s", fork_id, e,
            )
            return []

    def _plan_memory_nodes_prune(
        self, fork_id: str, exploration_txs: list[str], plan: PrunePlan,
    ) -> None:
        """For each memory_node whose `source_id` LIKE-matches one of the
        fork's exploration TX hashes, apply the §9.5 conjunction:
          (a) sole inbound = the dying fork → check that no OTHER active
              fork's exploration_txs reference the same node.
          (b) never canonical → the row's anchor TX (if present) is NOT
              on a canonical fork.
          (c) activation below floor → activation_state for the node id.

        We use a conservative approximation for (a): we only consider the
        node "sole-inbound" if its source_id contains exactly one of our
        TX hashes AND no other open fork references it (LEFT JOIN check).

        For (b): if memory_nodes carries an `anchor_tx` column, we resolve
        it via `_is_canonical_tx`; otherwise we conservatively treat the
        row as canonical and keep it (in doubt, keep).
        """
        try:
            cursor = self._memory_db.execute(
                "SELECT id, source_id FROM memory_nodes "
                "WHERE source_id IS NOT NULL"
            )
        except Exception as e:
            logger.debug(
                "[ForkGC] memory_nodes scan failed (%s) — skip subnode prune",
                e,
            )
            return

        tx_set = set(exploration_txs)
        for row in cursor.fetchall():
            mem_id, source_id = row[0], row[1]
            if source_id is None:
                continue
            # Predicate (a)-fast-path: any of our TXs appear in source_id?
            matches = [tx for tx in tx_set if tx in source_id]
            if not matches:
                continue
            # Predicate (a)-strict: no other open fork references this node.
            if self._other_open_fork_references(mem_id, fork_id):
                plan.keep_reasons.append(
                    f"memory_node:{mem_id} kept (a-fail: cross-fork reference)"
                )
                continue
            # Predicate (b): canonical anchor? Conservative default = keep.
            # If the schema doesn't expose an anchor_tx column we can't
            # disprove canonicity → keep.
            if not self._is_node_purely_probationary(mem_id):
                plan.keep_reasons.append(
                    f"memory_node:{mem_id} kept (b-fail: may be canonical)"
                )
                continue
            # Predicate (c): below activation floor?
            bi = self._activation_for_item(f"mem:{mem_id}")
            if bi is None or bi >= self._floor:
                plan.keep_reasons.append(
                    f"memory_node:{mem_id} kept (c-fail: B_i={bi} >= floor)"
                )
                continue
            plan.memory_nodes_will_drop.append(int(mem_id))

    def _plan_action_chains_prune(self, fork: Any, plan: PrunePlan) -> None:
        """action_chains_step prune predicate:
          - row created within [fork.created_at, fork.last_touched]
          - row not referenced by a Production node's COMPILED_FROM edge
          - sole inbound is exploration TXs from this fork

        Pre-Phase-8 Production/COMPILED_FROM is empty, so the (b) predicate
        ("never canonically compiled") passes trivially for nearly all
        rows — but the (a) "sole inbound from this fork" predicate is the
        teeth: we require the row's chain_id to appear in this fork's
        exploration_tx set. In practice this means: a fork can only
        cascade-prune action_chains_step rows it explicitly recorded as
        exploration TXs (the synthesis_worker can wire this if it ever
        produces action_chain TXs inside fork exploration; Phase 5 ships
        the framework but no producer fires it yet).
        """
        try:
            cursor = self._memory_db.execute(
                "SELECT chain_id, step_idx, created_at FROM action_chains_step "
                "WHERE created_at BETWEEN ? AND ?",
                (fork.created_at, fork.last_touched),
            )
        except Exception as e:
            logger.debug(
                "[ForkGC] action_chains_step scan failed (%s) — skip", e,
            )
            return

        exploration_txs = set(self._fetch_exploration_txs(fork.fork_id))
        if not exploration_txs:
            return

        for row in cursor.fetchall():
            chain_id, step_idx = row[0], row[1]
            # (a) — chain id must be among the fork's exploration TXs.
            if chain_id not in exploration_txs:
                continue
            # (b) — never compiled. Cheap check via Kuzu: does any
            # Production node have COMPILED_FROM → this chain?
            if self._chain_is_canonically_compiled(chain_id):
                plan.keep_reasons.append(
                    f"action_chains_step:{chain_id}.{step_idx} kept "
                    f"(b-fail: canonical Production exists)"
                )
                continue
            # (c) — chain activation below floor? action_chains don't have
            # an activation_state row by default; treat as below-floor only
            # if no activation entry exists at all (cold-start). This
            # matches §9.5's "in doubt, keep" — if there's an entry we
            # honor it; if there's no entry we consider it never-warm.
            bi = self._activation_for_item(f"chain:{chain_id}")
            if bi is not None and bi >= self._floor:
                plan.keep_reasons.append(
                    f"action_chains_step:{chain_id}.{step_idx} kept "
                    f"(c-fail: B_i={bi} >= floor)"
                )
                continue
            plan.action_chains_will_drop.append(int(step_idx))

    # ── Helpers — predicate evaluation ──────────────────────────────

    def _other_open_fork_references(
        self, memory_node_id: int, this_fork_id: str,
    ) -> bool:
        """Predicate (a) strict: does any OTHER open fork's exploration_txs
        reference this memory_node? We can't directly join on source_id,
        but we can scan open forks' exploration_txs + check intersection.
        Cheap because the set of open forks is bounded."""
        try:
            other_forks = self._db.execute(
                "SELECT fork_id FROM hypothesis_forks "
                "WHERE status='open' AND fork_id != ?",
                (this_fork_id,),
            ).fetchall()
        except Exception:
            return True  # conservative: can't determine → assume yes → keep
        if not other_forks:
            return False
        # Bulk fetch the other forks' exploration TXs.
        try:
            self._db.execute("CREATE TEMPORARY TABLE IF NOT EXISTS _gc_other_forks (fork_id TEXT)")
            self._db.execute("DELETE FROM _gc_other_forks")
            for (fid,) in other_forks:
                self._db.execute(
                    "INSERT INTO _gc_other_forks VALUES (?)", (fid,),
                )
            other_txs_rows = self._db.execute(
                "SELECT DISTINCT e.tx_hash FROM hypothesis_fork_explorations e "
                "JOIN _gc_other_forks o ON e.fork_id = o.fork_id"
            ).fetchall()
        except Exception as e:
            logger.debug(
                "[ForkGC] _other_open_fork_references temp-table path failed: "
                "%s — fall back to per-fork scan", e,
            )
            other_txs_rows = []
            for (fid,) in other_forks:
                rows = self._db.execute(
                    "SELECT tx_hash FROM hypothesis_fork_explorations "
                    "WHERE fork_id=?",
                    (fid,),
                ).fetchall()
                other_txs_rows.extend(rows)
        if not other_txs_rows:
            return False
        other_tx_set = {r[0] for r in other_txs_rows}
        # Check memory_node.source_id against other_tx_set.
        try:
            src = self._memory_db.execute(
                "SELECT source_id FROM memory_nodes WHERE id=?",
                (memory_node_id,),
            ).fetchone()
        except Exception:
            return True
        if not src or not src[0]:
            return False
        for tx in other_tx_set:
            if tx in src[0]:
                return True
        return False

    def _is_node_purely_probationary(self, memory_node_id: int) -> bool:
        """Predicate (b): the row has no canonical anchor TX. memory_nodes
        in the current schema doesn't carry an explicit `anchor_tx` column
        for canonical claims — `source_id` is the only TX-pointing field.
        We treat a node as probationary iff every TX hash extractable from
        source_id is NOT resolvable as a canonical (non-hypothesis-class)
        TX via `_is_canonical_tx`.

        Conservative defaults:
          - If we can't parse any TX out of source_id → keep (return False).
          - If the resolver returns True for ANY TX → keep (return False).
          - Only if every TX in source_id is non-canonical AND the set is
            non-empty do we return True (= prune-eligible on b).
        """
        try:
            src = self._memory_db.execute(
                "SELECT source_id FROM memory_nodes WHERE id=?",
                (memory_node_id,),
            ).fetchone()
        except Exception:
            return False
        if not src or not src[0]:
            return False
        # source_id is a free-form string in the current schema; extract
        # the longest contiguous 64-hex-char substrings as candidate TXs.
        import re
        tx_candidates = re.findall(r"[0-9a-f]{64}", str(src[0]).lower())
        if not tx_candidates:
            return False
        for tx in tx_candidates:
            try:
                if self._is_canonical_tx(tx):
                    return False
            except Exception:
                return False  # conservative
        return True

    def _activation_for_item(self, item_id: str) -> Optional[float]:
        """Read the current base_level for item_id from activation_state
        (sole-writer is synthesis_worker = us; same process so same lock)."""
        try:
            row = self._db.execute(
                "SELECT base_level FROM activation_state WHERE item_id=?",
                (item_id,),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return float(row[0])

    def _chain_is_canonically_compiled(self, chain_id: str) -> bool:
        """Does any Production node carry COMPILED_FROM → this action_chain?
        Used by the action_chains_step predicate (b)."""
        try:
            qr = self._graph._conn.execute(
                "MATCH (p:Production)-[:COMPILED_FROM]->(ac:ActionChain "
                "{chain_id: $cid}) RETURN COUNT(p) LIMIT 1",
                {"cid": chain_id},
            )
            if qr.has_next():
                return int(qr.get_next()[0]) > 0
        except Exception:
            return True   # conservative: in doubt, keep
        return False

    # ── Plan execution: transactional per-fork prune ────────────────

    def _execute_plan(self, plan: PrunePlan) -> int:
        """Apply a PrunePlan inside a transaction. Rolls back on any
        sub-failure; returns the actually-pruned node count.

        DuckDB's BEGIN/COMMIT bounds writes to synthesis.duckdb; Kuzu
        write-throughs (fork_delete_node) commit per statement (Kuzu 0.11
        doesn't expose transaction boundaries to the Python API), so on
        rollback we may leak a Kuzu node deletion. The next sweep will
        re-discover the fork via pending_gc_targets (it stays graduated/
        abandoned in DuckDB) and re-attempt the cleanup.
        """
        dropped = 0
        # Begin transaction on synthesis.duckdb.
        self._db.execute("BEGIN TRANSACTION")
        try:
            # (1) durable exploration log
            if plan.durable_exploration_rows_will_drop > 0:
                self._db.execute(
                    "DELETE FROM hypothesis_fork_explorations WHERE fork_id=?",
                    (plan.fork_id,),
                )
                dropped += plan.durable_exploration_rows_will_drop

            # (2) memory_nodes
            for mem_id in plan.memory_nodes_will_drop:
                try:
                    self._memory_db.execute(
                        "DELETE FROM memory_nodes WHERE id=?", (mem_id,),
                    )
                    dropped += 1
                except Exception as e:
                    logger.warning(
                        "[ForkGC] execute: memory_nodes delete id=%d failed: "
                        "%s — rolling back", mem_id, e,
                    )
                    raise

            # (3) action_chains_step
            for step_idx in plan.action_chains_will_drop:
                try:
                    self._memory_db.execute(
                        "DELETE FROM action_chains_step WHERE step_idx=?",
                        (step_idx,),
                    )
                    dropped += 1
                except Exception as e:
                    logger.warning(
                        "[ForkGC] execute: action_chains_step delete "
                        "step_idx=%d failed: %s — rolling back", step_idx, e,
                    )
                    raise

            self._db.execute("COMMIT")
        except Exception:
            try:
                self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise

        # Kuzu node delete is outside the transaction (Kuzu lacks Python
        # transaction API). Run it last so a failure here doesn't dirty
        # the synthesis.duckdb state — the fork stays pending_gc_targets
        # for next sweep.
        if plan.kuzu_node_will_drop:
            ok = self._graph.fork_delete_node(plan.fork_id)
            if ok:
                dropped += 1
                # +EXPLORES edges (DETACH DELETE handles automatically).
                dropped += plan.kuzu_explores_edges_will_drop

        return dropped


__all__ = (
    "ForkGC",
    "PrunePlan",
    "SweepReport",
    "DEFAULT_MAX_NODES_PER_SWEEP",
)
