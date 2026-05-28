"""HypothesisForkStore — sole writer of the hypothesis-fork lifecycle (Phase 5).

Per `ARCHITECTURE_synthesis_engine.md §9` (RATIFIED — INV-3 locked) +
`PLAN_synthesis_engine_Phase5.md §P5.A` + INV-Syn-8 (proposed P5.K).

This module is the ONLY surface authorized to create, transition, or close
hypothesis forks. The synthesis_worker instantiates a singleton at startup;
all reads from other processes go via `forks_snapshot.json` (P5.I).

Invariants enforced HERE:

- **INV-3** — only never-canonical probationary data is collected. The
  fork's `hypothesis_forks` row is updated through its lifecycle but the
  row history (`graduated_anchor_tx`, `abandoned_tombstone_tx`) is preserved
  for audit; only the Kuzu `HypothesisFork` *index* node is DETACH-DELETE'd
  by the cascade-GC sweep (it's a hot index, not canonical data — the
  canonical record is the chain TX).

- **INV-10** — a repair fork NEVER mutates its parent concept. Reuses
  `ConceptStore.bump_version()` insert-only contract (Phase 4 hard test).

- **INV-Syn-3** — synthesis_worker is the sole writer to `synthesis.duckdb`.
  HypothesisForkStore is only ever instantiated inside that process.

- **INV-Syn-8 (proposed P5.K)** — sole writer to `hypothesis_forks` (DuckDB)
  + `HypothesisFork` Kuzu nodes + `EXPLORES` edges.

- **INV-Syn-9 (proposed P5.K)** — tombstone TX is canonical (Arweave-
  eligible) even though the fork's exploration TXs are not. The scar is
  provably-non-empty.

- **INV-Syn-10 (proposed P5.K)** — cascade-GC predicate is conjunctive
  (sole-inbound AND never-canonical AND below-floor). Implemented in
  the companion `fork_gc.py` (P5.H).

- **INV-Syn-11 (proposed P5.K)** — repair-fork graduation reuses Phase 4
  insert-only `bump_version`. Asserted by the `_graduate_repair` path.

Constructor dependencies (all duck-typed — tests inject fakes):

- `duckdb_conn`: open DuckDB connection to `synthesis.duckdb` (the ActivationStore's
  connection is reused — same process, same lock).
- `kuzu_graph`: `TitanKnowledgeGraph` with the Phase 5 fork-helper methods
  (`fork_create_node`, `fork_update_status`, ...).
- `concept_store`: `ConceptStore` instance for graduation-path concept writes
  (reuses Phase 4's create_concept / bump_version — INV-10 enforced there).
- `outer_memory_writer`: `OuterMemoryWriter` for tombstone + graduation TXs.
- `activation_store`: `ActivationStore` (synthesis_worker.py) for the
  `fork:<id>` namespace + B_i tracking.
- `clock`: injectable `time.time`-compatible callable.

Public API surface (see `PLAN_synthesis_engine_Phase5.md §P5.A`):

  - `create_fork(...)` → fork_id
  - `graduate_oracle(...)` → concept_anchor_tx
  - `graduate_used(...)` → concept_anchor_tx
  - `abandon(...)` → tombstone_anchor_tx
  - `on_fork_read(fork_id)` → None (FORK_READ observer; auto-triggers graduate_used at threshold)
  - `find_below_floor(floor, window_sec)` → list[fork_id]
  - `get_fork(fork_id)` → HypothesisFork | None
  - `list_active()` → list[HypothesisFork]
  - `export_snapshot(path)` → int (atomic JSON, mirrors ConceptStore.export_snapshot)
  - `pending_gc_targets()` → list[fork_id] (graduated/abandoned awaiting cascade sweep)

`fork_class="hypothesis"` is encoded on the chain TX *metadata* (the
`OuterMemoryEvent.fork` field) — a hypothesis-fork's exploration TXs MUST
ride a dedicated logical fork that the Arweave-cascade encoder filters out
(§P5.D Maker-greenlit decision). Net-new vs repair forks share the same
fork-id namespace; the differentiator is `root_anchor` (None vs canonical TX).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────


class ForkNotFound(Exception):
    """Fork id has no row in hypothesis_forks."""


class ForkStateError(Exception):
    """Lifecycle transition rejected (e.g. graduate-twice, abandon-graduated)."""


class GraduationGateError(Exception):
    """Graduation preconditions not met (oracle verdict false / use_count <3)."""


# ── Use-count threshold for graduate_used (arch §8.5 default; P5.F) ──

USE_COUNT_GRADUATION_THRESHOLD = 3


# ── Default activation TTL params (Maker-locked 2026-05-27) ────────

DEFAULT_ACTIVATION_FLOOR = -3.0    # B_i below -3.0 = roughly one access >1 day stale
DEFAULT_WINDOW_SEC = 7.0 * 86400.0  # 7 days quiet → abandon-eligible


# ── Public dataclasses ─────────────────────────────────────────────


@dataclass(frozen=True)
class HypothesisFork:
    """The materialized state of one fork. Immutable snapshot; mutations
    happen via HypothesisForkStore methods which write through to DuckDB."""

    fork_id: str
    root_anchor: Optional[str]              # None = net-new; tx_hash = repair fork
    parent_concept_id: Optional[str]        # required when root_anchor != None
    intent: str
    status: str                              # 'open' | 'graduated' | 'abandoned'
    created_at: float
    last_touched: float
    use_count: int
    activation: float                        # last computed B_i (snapshot)
    graduated_at: Optional[float] = None
    graduated_concept_id: Optional[str] = None
    graduated_anchor_tx: Optional[str] = None
    abandoned_at: Optional[float] = None
    abandoned_tombstone_tx: Optional[str] = None
    abandonment_reason: Optional[str] = None


# ── Duck-typed dependency protocols (for unit-test substitution) ──


class _ActivationStoreLike(Protocol):
    def record_access(self, item_id: str, ts: float) -> None: ...


class _ConceptStoreLike(Protocol):
    def create_concept(
        self, concept_id: str, name: str, memory_type: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
    ) -> Any: ...

    def bump_version(
        self, concept_id: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
        groundedness_at_bump: Optional[float] = None,
    ) -> Any: ...


class _OuterMemoryWriterLike(Protocol):
    def write_concept_version_with_proof(
        self, *, concept_id: str, version: int, name: str, memory_type: str,
        parent_version_tx: Optional[str],
        composed_from: list[tuple[str, int]],
        derivation_evidence: list[str], groundedness: float,
        derivation_merkle_root: str, oracle_verdict: dict, **kwargs,
    ) -> tuple[str, str]: ...

    def write_tombstone(
        self, *, fork_id: str, root_anchor: Optional[str], intent: str,
        explored_from: float, explored_to: float, exploration_root: str,
        abandonment_reason: str, reference_count_pruned: int, **kwargs,
    ) -> str: ...


# ── Fork-id generation (deterministic, but no key collisions) ────


def _make_fork_id(intent: str, ts: float, root_anchor: Optional[str]) -> str:
    """sha256(intent || ts || root_anchor or '')[:16] — 64-bit space, collision
    probability negligible at fleet scale, opaque to outside observers
    (intent + ts not directly reversible). The 16-hex (8-byte) length matches
    the convention used by Phase-4 ConceptStore for concept_id collisions."""
    seed = f"{intent}|{ts:.6f}|{root_anchor or ''}".encode()
    return hashlib.sha256(seed).hexdigest()[:16]


# ── HypothesisForkStore ─────────────────────────────────────────────


class HypothesisForkStore:
    """Sole writer of hypothesis-fork lifecycle. INV-Syn-8."""

    def __init__(
        self,
        *,
        duckdb_conn: Any,                    # open DuckDB connection (synthesis.duckdb)
        kuzu_graph: Any,                     # TitanKnowledgeGraph
        concept_store: _ConceptStoreLike,
        outer_memory_writer: _OuterMemoryWriterLike,
        activation_store: _ActivationStoreLike,
        activation_floor: float = DEFAULT_ACTIVATION_FLOOR,
        ttl_window_sec: float = DEFAULT_WINDOW_SEC,
        clock: Any = time.time,
        # P8.X (D-SPEC-PHASE8 fold-in): write-through snapshot path. When set,
        # every lifecycle mutator (create/record/graduate/abandon) calls
        # `export_snapshot(snapshot_path)` synchronously after the DuckDB
        # transaction commits. Closes the "new fork visible in snapshot
        # never appeared after 6s" P5 cascade flake (the 60s recompute-loop
        # snapshot stays as a heartbeat but is no longer load-bearing for
        # visibility). Mirrors the P7 ActrBufferStore.persist() pattern.
        snapshot_path: Optional[str] = None,
    ):
        self._db = duckdb_conn
        self._graph = kuzu_graph
        self._concepts = concept_store
        self._writer = outer_memory_writer
        self._activation = activation_store
        self._floor = float(activation_floor)
        self._window = float(ttl_window_sec)
        self._clock = clock
        self._snapshot_path = snapshot_path
        # Exploration-TX tracker: in-memory map of fork_id -> list[tx_hash].
        # Populated by `record_exploration_tx()` (called by the synthesis
        # engine when it anchors a TX inside a fork) and consumed at
        # graduation (Merkle root over the list) + abandonment (Merkle
        # root over the list before cascade-prune).
        # NOT persisted to DuckDB pre-graduation: per §9.2 the exploration
        # data is HDD-only and probationary. After graduation, the Merkle
        # root + the TX hash list (if needed for audit replay) survive as
        # part of the canonical concept-version TX's `derivation_evidence`.
        # After abandonment, only the Merkle root survives (in the tombstone
        # TX's `exploration_root`); the list itself is pruned.
        self._exploration_txs: dict[str, list[str]] = {}
        self._init_schema()
        self._load_exploration_log_from_durable_state()

    # ── Schema bootstrap (P5.B) ──────────────────────────────────────

    def _init_schema(self) -> None:
        """CREATE TABLE IF NOT EXISTS hypothesis_forks (idempotent)."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_forks (
                fork_id        TEXT PRIMARY KEY,
                root_anchor    TEXT,
                parent_concept_id TEXT,
                intent         TEXT NOT NULL,
                status         TEXT NOT NULL CHECK(status IN ('open','graduated','abandoned')),
                created_at     DOUBLE NOT NULL,
                last_touched   DOUBLE NOT NULL,
                use_count      INTEGER NOT NULL DEFAULT 0,
                activation     DOUBLE,
                graduated_at   DOUBLE,
                graduated_concept_id TEXT,
                graduated_anchor_tx  TEXT,
                abandoned_at   DOUBLE,
                abandoned_tombstone_tx TEXT,
                abandonment_reason TEXT
            )
        """)
        # Indices for hot paths: status-scan (sweep), last_touched-scan (TTL).
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS hf_status_idx "
            "ON hypothesis_forks(status)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS hf_last_touched_idx "
            "ON hypothesis_forks(last_touched)"
        )
        # Companion table: persisted exploration-TX log for crash resilience.
        # Without this, a synthesis_worker crash mid-fork would lose the
        # Merkle leaves and the eventual graduation/tombstone could not
        # cryptographically prove what was explored. The log is durable but
        # SMALL (just (fork_id, tx_hash) pairs); GC'd alongside the fork
        # at graduation/abandonment by `_purge_exploration_log`.
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_fork_explorations (
                fork_id   TEXT NOT NULL,
                tx_hash   TEXT NOT NULL,
                recorded_at DOUBLE NOT NULL,
                PRIMARY KEY (fork_id, tx_hash)
            )
        """)

    def _load_exploration_log_from_durable_state(self) -> None:
        """Rehydrate the in-memory exploration map from
        hypothesis_fork_explorations after a worker restart."""
        try:
            rows = self._db.execute(
                "SELECT fork_id, tx_hash FROM hypothesis_fork_explorations "
                "ORDER BY recorded_at"
            ).fetchall()
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] exploration-log rehydrate failed: %s", e,
            )
            return
        for fork_id, tx_hash in rows:
            self._exploration_txs.setdefault(fork_id, []).append(tx_hash)
        if rows:
            logger.info(
                "[HypothesisForkStore] rehydrated %d exploration TXs across "
                "%d forks", len(rows), len(self._exploration_txs),
            )

    # ── Public surface — lifecycle ────────────────────────────────────

    def create_fork(
        self,
        *,
        intent: str,
        root_anchor: Optional[str] = None,
        parent_concept_id: Optional[str] = None,
    ) -> str:
        """Create a fresh hypothesis fork (status='open') and return its id.

        Net-new fork: `root_anchor` and `parent_concept_id` both None.
        Repair fork: BOTH non-None; `parent_concept_id` MUST resolve in Kuzu
        Concept (latest version is auto-resolved at graduation time).

        Side effects:
          1. INSERT hypothesis_forks row (status='open', use_count=0,
             activation=0.0, created_at=last_touched=now).
          2. INSERT Kuzu HypothesisFork node (status='open').
          3. If repair fork: INSERT Kuzu EXPLORES edge to the parent concept's
             *latest* version.
          4. Touch the activation_state row for `fork:<fork_id>` (initial
             access). Subsequent B_i computes happen in the synthesis_worker
             60s recompute loop.

        Returns the fork_id. Raises:
          - ValueError if intent is empty or root_anchor/parent_concept_id
            args are inconsistent.
          - ForkStateError if parent_concept_id doesn't resolve in Kuzu.
        """
        if not intent or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if (root_anchor is None) != (parent_concept_id is None):
            raise ValueError(
                "root_anchor and parent_concept_id must both be None "
                "(net-new fork) or both non-None (repair fork)"
            )
        # Repair-fork: parent must resolve.
        if parent_concept_id is not None:
            latest = self._graph.spine_get_latest_concept(parent_concept_id)
            if latest is None:
                raise ForkStateError(
                    f"repair fork: parent_concept_id={parent_concept_id!r} "
                    "has no Concept rows in spine (call create_concept first)"
                )

        now = self._clock()
        fork_id = _make_fork_id(intent, now, root_anchor)

        # Step 1: DuckDB INSERT. PRIMARY KEY collision (same intent+ts+root
        # within the microsecond, vanishingly improbable) → caller retries.
        self._db.execute(
            "INSERT INTO hypothesis_forks "
            "(fork_id, root_anchor, parent_concept_id, intent, status, "
            " created_at, last_touched, use_count, activation) "
            "VALUES (?, ?, ?, ?, 'open', ?, ?, 0, 0.0)",
            (fork_id, root_anchor, parent_concept_id, intent, now, now),
        )

        # Step 2: Kuzu HypothesisFork node. If this fails AFTER the DuckDB
        # row landed, we keep going (the spine is rebuildable from chain;
        # the DuckDB row is the source of truth for lifecycle state).
        created_in_kuzu = self._graph.fork_create_node(
            fork_id=fork_id, root_anchor=root_anchor or "",
            activation=0.0, status="open",
        )
        if not created_in_kuzu:
            logger.warning(
                "[HypothesisForkStore] create_fork(%s): Kuzu node insert "
                "skipped (probably duplicate) — DuckDB stays canonical",
                fork_id,
            )

        # Step 3: EXPLORES edge for repair forks.
        if parent_concept_id is not None:
            latest = self._graph.spine_get_latest_concept(parent_concept_id)
            if latest is not None:
                ok = self._graph.fork_add_explores_edge(
                    fork_id=fork_id,
                    concept_id=parent_concept_id,
                    version=latest["version"],
                )
                if not ok:
                    logger.warning(
                        "[HypothesisForkStore] create_fork(%s): EXPLORES "
                        "edge to %s v%d failed — spine + lifecycle may "
                        "diverge until next sweep",
                        fork_id, parent_concept_id, latest["version"],
                    )

        # Step 4: initial activation touch.
        self._activation.record_access(f"fork:{fork_id}", now)

        logger.info(
            "[HypothesisForkStore] created fork=%s type=%s root_anchor=%s "
            "intent=%r",
            fork_id,
            "repair" if root_anchor else "net_new",
            (root_anchor[:16] + "...") if root_anchor else "∅",
            intent[:64],
        )
        # P8.X write-through: closes P5 cascade flake (snapshot lag on create)
        self._maybe_write_through_snapshot()
        return fork_id

    def record_exploration_tx(self, fork_id: str, tx_hash: str) -> None:
        """Track one exploration TX hash inside an active fork. Called by
        the synthesis engine when it anchors a TX with `fork_class="hypothesis"`
        + matching fork_id. Stored in-memory + DuckDB durable; consumed at
        graduation (Merkle root over the list) or abandonment (proof of
        what existed before pruning)."""
        fork = self.get_fork(fork_id)
        if fork is None:
            raise ForkNotFound(
                f"record_exploration_tx({fork_id}): no such fork"
            )
        if fork.status != "open":
            raise ForkStateError(
                f"record_exploration_tx({fork_id}): fork is {fork.status!r}, "
                "must be 'open'"
            )
        self._exploration_txs.setdefault(fork_id, []).append(tx_hash)
        try:
            self._db.execute(
                "INSERT INTO hypothesis_fork_explorations "
                "(fork_id, tx_hash, recorded_at) VALUES (?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                (fork_id, tx_hash, self._clock()),
            )
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] record_exploration_tx(%s,%s) durable "
                "write failed: %s — in-memory list preserved",
                fork_id, tx_hash, e,
            )
        # P8.X write-through: closes P5 cascade flake (snapshot lag on record)
        self._maybe_write_through_snapshot()

    def graduate_oracle(
        self,
        *,
        fork_id: str,
        oracle_verdict: dict,
        concept_name: Optional[str] = None,
    ) -> str:
        """Graduate a fork via oracle-verified path (arch §9.3 + §11.2).

        Preconditions:
          - fork.status == 'open'
          - oracle_verdict['verdict'] == 'true' (string literal per plugs.py
            `OracleVerdict.verdict: Literal["true","false","unknown"]`)

        Side effects:
          - Computes Merkle root over the fork's exploration TXs.
          - Repair fork: calls ConceptStore.bump_version (INV-10, INV-Syn-11).
          - Net-new fork: calls ConceptStore.create_concept (caller must
            supply `concept_name`; raises ValueError if missing).
          - Calls OuterMemoryWriter.write_concept_version_with_proof
            (anchors concept-version TX + OracleVerdict TX on chain).
          - Marks fork status='graduated', records anchor_tx + graduated_at.
          - Removes EXPLORES edges, leaves the HypothesisFork node + DuckDB
            row in place for the cascade-GC sweep to clean up alongside any
            never-canonical subnodes (P5.H).

        Returns the concept-version anchor_tx.
        """
        fork = self.get_fork(fork_id)
        if fork is None:
            raise ForkNotFound(f"graduate_oracle: no fork {fork_id!r}")
        if fork.status != "open":
            raise ForkStateError(
                f"graduate_oracle({fork_id}): status={fork.status!r}, "
                "must be 'open'"
            )
        if str(oracle_verdict.get("verdict", "unknown")).lower() != "true":
            raise GraduationGateError(
                f"graduate_oracle({fork_id}): verdict="
                f"{oracle_verdict.get('verdict')!r}, must be 'true'"
            )
        return self._graduate(
            fork=fork, oracle_verdict=oracle_verdict,
            concept_name=concept_name,
        )

    def graduate_used(
        self,
        *,
        fork_id: str,
        concept_name: Optional[str] = None,
    ) -> str:
        """Graduate a fork via use-count path (arch §9.3, threshold from
        P5.F = 3).

        Preconditions:
          - fork.status == 'open'
          - fork.use_count >= USE_COUNT_GRADUATION_THRESHOLD

        The use_count is the store's own observed count (incremented only by
        `on_fork_read()`), so callers cannot fake it.

        Constructs a synthetic OracleVerdict for the audit trail:
        `{oracle_id: "use_threshold", verdict: "true", evidence_ref:
        "use_count=<n>", cost: 0, latency_ms: 0, ts: now}`.

        Returns the concept-version anchor_tx.
        """
        fork = self.get_fork(fork_id)
        if fork is None:
            raise ForkNotFound(f"graduate_used: no fork {fork_id!r}")
        if fork.status != "open":
            raise ForkStateError(
                f"graduate_used({fork_id}): status={fork.status!r}, "
                "must be 'open'"
            )
        if fork.use_count < USE_COUNT_GRADUATION_THRESHOLD:
            raise GraduationGateError(
                f"graduate_used({fork_id}): use_count={fork.use_count}, "
                f"threshold={USE_COUNT_GRADUATION_THRESHOLD}"
            )
        synthetic_verdict = {
            "oracle_id": "use_threshold",
            "verdict": "true",
            "evidence_ref": f"use_count={fork.use_count}",
            "cost": 0.0,
            "latency_ms": 0,
            "ts": self._clock(),
        }
        return self._graduate(
            fork=fork, oracle_verdict=synthetic_verdict,
            concept_name=concept_name,
        )

    def abandon(
        self,
        *,
        fork_id: str,
        reason: str = "activation_below_floor",
    ) -> Optional[str]:
        """Abandon a fork: write tombstone TX, mark status='abandoned'.

        Idempotent: if status is already 'abandoned' or 'graduated', returns
        None and logs at INFO (no-op). Otherwise returns the tombstone TX hash.

        Cascade-pruning of subnodes is NOT done here — it's the job of the
        nightly sweep in `fork_gc.py` (P5.H), which runs the conservative
        reference-counted predicate and writes a tombstone batch event. The
        per-fork tombstone TX (this method's output) is the per-fork scar;
        the batch event is the operational audit log.
        """
        fork = self.get_fork(fork_id)
        if fork is None:
            raise ForkNotFound(f"abandon: no fork {fork_id!r}")
        if fork.status in ("abandoned", "graduated"):
            logger.info(
                "[HypothesisForkStore] abandon(%s) idempotent no-op "
                "(status=%s)", fork_id, fork.status,
            )
            return None

        # Compute the Merkle root over exploration TXs — proves what was
        # tried even after the underlying TXs are forgotten (§9.3, §P5.G).
        from titan_hcl.synthesis.merkle import merkle_root_hex
        exploration_tx_list = list(self._exploration_txs.get(fork_id, ()))
        exploration_root = merkle_root_hex(exploration_tx_list)

        now = self._clock()
        tombstone_tx = self._writer.write_tombstone(
            fork_id=fork_id,
            root_anchor=fork.root_anchor,
            intent=fork.intent,
            explored_from=fork.created_at,
            explored_to=fork.last_touched,
            exploration_root=exploration_root,
            abandonment_reason=reason,
            reference_count_pruned=0,   # populated by the GC sweep batch
        )

        # Update lifecycle row.
        self._db.execute(
            "UPDATE hypothesis_forks SET status='abandoned', "
            "abandoned_at=?, abandoned_tombstone_tx=?, abandonment_reason=? "
            "WHERE fork_id=?",
            (now, tombstone_tx, reason, fork_id),
        )
        self._graph.fork_update_status(
            fork_id=fork_id, status="abandoned", activation=fork.activation,
        )

        # Drop the in-memory exploration log immediately — the Merkle root
        # is anchored on-chain. The durable companion table will be cleaned
        # up by the cascade-GC sweep (which fires the per-fork batch event).
        self._exploration_txs.pop(fork_id, None)

        logger.info(
            "[HypothesisForkStore] abandoned fork=%s reason=%s "
            "tombstone_tx=%s exploration_root=%s exploration_count=%d",
            fork_id, reason, tombstone_tx[:16], exploration_root[:16],
            len(exploration_tx_list),
        )
        # P8.X write-through: closes P5 cascade flake (snapshot lag on abandon)
        self._maybe_write_through_snapshot()
        return tombstone_tx

    # ── Public surface — FORK_READ observer (P5.F) ────────────────────

    def on_fork_read(self, fork_id: str) -> Optional[str]:
        """Increment use_count + touch activation for a fork the SC handler
        just FORK_READ-ed. Auto-fires graduate_used() at the threshold.

        Returns the concept_anchor_tx if graduation auto-fired this call,
        else None.

        Per arch §5.4 + §P5.F: only deliberate FORK_READ invocations count
        — incidental composite-score retrieval does NOT touch this (rich-
        get-richer trap).
        """
        fork = self.get_fork(fork_id)
        if fork is None:
            raise ForkNotFound(f"on_fork_read: no fork {fork_id!r}")
        if fork.status != "open":
            # Not an error — FORK_READ on a graduated fork is a legitimate
            # query, it just doesn't bump use_count.
            return None

        now = self._clock()
        new_count = fork.use_count + 1
        self._db.execute(
            "UPDATE hypothesis_forks SET use_count=?, last_touched=? "
            "WHERE fork_id=?",
            (new_count, now, fork_id),
        )
        self._activation.record_access(f"fork:{fork_id}", now)

        # Auto-graduate exactly once at threshold crossing.
        if new_count >= USE_COUNT_GRADUATION_THRESHOLD:
            try:
                return self.graduate_used(
                    fork_id=fork_id, concept_name=fork.intent[:64],
                )
            except GraduationGateError:
                # Race: another caller already graduated. Soft-fail.
                logger.info(
                    "[HypothesisForkStore] on_fork_read(%s) auto-graduate "
                    "raced — already graduated", fork_id,
                )
        return None

    # ── Public surface — readers ──────────────────────────────────────

    def get_fork(self, fork_id: str) -> Optional[HypothesisFork]:
        rows = self._db.execute(
            "SELECT fork_id, root_anchor, parent_concept_id, intent, status, "
            "created_at, last_touched, use_count, activation, "
            "graduated_at, graduated_concept_id, graduated_anchor_tx, "
            "abandoned_at, abandoned_tombstone_tx, abandonment_reason "
            "FROM hypothesis_forks WHERE fork_id = ?",
            (fork_id,),
        ).fetchall()
        if not rows:
            return None
        row = rows[0]
        return HypothesisFork(
            fork_id=row[0], root_anchor=row[1], parent_concept_id=row[2],
            intent=row[3], status=row[4], created_at=float(row[5]),
            last_touched=float(row[6]), use_count=int(row[7] or 0),
            activation=float(row[8] or 0.0),
            graduated_at=float(row[9]) if row[9] is not None else None,
            graduated_concept_id=row[10],
            graduated_anchor_tx=row[11],
            abandoned_at=float(row[12]) if row[12] is not None else None,
            abandoned_tombstone_tx=row[13],
            abandonment_reason=row[14],
        )

    def list_active(self) -> list[HypothesisFork]:
        return self._list_by_status("open")

    def list_by_status(self, status: str) -> list[HypothesisFork]:
        return self._list_by_status(status)

    def _list_by_status(self, status: str) -> list[HypothesisFork]:
        ids = [
            row[0] for row in self._db.execute(
                "SELECT fork_id FROM hypothesis_forks WHERE status = ? "
                "ORDER BY last_touched DESC",
                (status,),
            ).fetchall()
        ]
        out: list[HypothesisFork] = []
        for fid in ids:
            f = self.get_fork(fid)
            if f is not None:
                out.append(f)
        return out

    def find_below_floor(
        self,
        floor: Optional[float] = None,
        window_sec: Optional[float] = None,
    ) -> list[str]:
        """Return fork_ids whose activation has been below `floor` with no
        touch within `window_sec`. Both args default to the store's locked-
        in TTL params (Maker 2026-05-27).

        Implementation: the activation column is updated by the recompute
        loop (synthesis_worker.py extension P5.C) at each 60s pass; this
        method simply scans rows where status='open' AND activation < floor
        AND last_touched < (now - window_sec).
        """
        floor_v = self._floor if floor is None else float(floor)
        window_v = self._window if window_sec is None else float(window_sec)
        cutoff = self._clock() - window_v
        rows = self._db.execute(
            "SELECT fork_id FROM hypothesis_forks "
            "WHERE status='open' AND activation < ? AND last_touched < ? "
            "ORDER BY last_touched",
            (floor_v, cutoff),
        ).fetchall()
        return [r[0] for r in rows]

    def pending_gc_targets(self) -> list[str]:
        """Forks whose lifecycle has ended (graduated or abandoned) but the
        Kuzu HypothesisFork node + cascade subnodes still exist. The nightly
        sweep (P5.H) reads this and pruges them."""
        rows = self._db.execute(
            "SELECT fork_id FROM hypothesis_forks "
            "WHERE status IN ('graduated','abandoned') "
            "ORDER BY graduated_at NULLS LAST, abandoned_at NULLS LAST"
        ).fetchall()
        # Filter to forks whose Kuzu node still exists (i.e. not yet swept).
        out: list[str] = []
        for (fid,) in rows:
            if self._graph.fork_get_node(fid) is not None:
                out.append(fid)
        return out

    def update_activation(self, fork_id: str, base_level: float) -> None:
        """Called by the synthesis_worker recompute loop to persist the
        fork's latest B_i to DuckDB + Kuzu (so find_below_floor and the
        Observatory readout reflect current activation)."""
        if not self._db.execute(
            "SELECT 1 FROM hypothesis_forks WHERE fork_id=?",
            (fork_id,),
        ).fetchone():
            return
        self._db.execute(
            "UPDATE hypothesis_forks SET activation=? WHERE fork_id=?",
            (float(base_level), fork_id),
        )
        # Best-effort Kuzu mirror; status unchanged.
        node = self._graph.fork_get_node(fork_id)
        if node is not None:
            self._graph.fork_update_status(
                fork_id=fork_id, status=node["status"],
                activation=float(base_level),
            )

    def purge_durable_state(self, fork_id: str) -> None:
        """Called by the cascade-GC sweep after Kuzu cleanup: drop the
        durable exploration log + Kuzu fork node. The DuckDB lifecycle row
        is preserved (canonical audit trail; see INV-3 comment in module
        docstring)."""
        try:
            self._db.execute(
                "DELETE FROM hypothesis_fork_explorations WHERE fork_id=?",
                (fork_id,),
            )
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] purge_durable_state(%s) explorations "
                "delete failed: %s", fork_id, e,
            )
        try:
            self._graph.fork_delete_node(fork_id)
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] purge_durable_state(%s) Kuzu node "
                "delete failed: %s", fork_id, e,
            )

    # ── Snapshot export (cross-process read surface — mirrors P4 FU-1) ─

    def _maybe_write_through_snapshot(self) -> None:
        """P8.X write-through: if `snapshot_path` was supplied at construction,
        export the snapshot synchronously after a lifecycle mutator commits.
        Soft-fail — logs WARN; never raises (export errors must not block
        the caller's transaction)."""
        if not self._snapshot_path:
            return
        try:
            self.export_snapshot(self._snapshot_path)
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] write-through snapshot export failed: %s", e,
            )

    def export_snapshot(self, snapshot_path: str) -> int:
        """Atomic JSON export of all hypothesis_forks rows + summary stats.

        Schema:
            {
              "version": 1,
              "exported_at": <wall-clock seconds>,
              "forks": [
                {fork_id, root_anchor, parent_concept_id, intent, status,
                 created_at, last_touched, use_count, activation,
                 graduated_at, graduated_concept_id, graduated_anchor_tx,
                 abandoned_at, abandoned_tombstone_tx, abandonment_reason},
                ...
              ],
              "summary": {open: int, graduated: int, abandoned: int}
            }

        Atomic via tmp + os.replace. Returns total fork-row count written.
        """
        import json
        import os

        rows = self._db.execute(
            "SELECT fork_id, root_anchor, parent_concept_id, intent, status, "
            "created_at, last_touched, use_count, activation, "
            "graduated_at, graduated_concept_id, graduated_anchor_tx, "
            "abandoned_at, abandoned_tombstone_tx, abandonment_reason "
            "FROM hypothesis_forks ORDER BY created_at"
        ).fetchall()

        forks: list[dict] = []
        summary = {"open": 0, "graduated": 0, "abandoned": 0}
        for row in rows:
            status = row[4]
            if status in summary:
                summary[status] += 1
            forks.append({
                "fork_id": row[0],
                "root_anchor": row[1],
                "parent_concept_id": row[2],
                "intent": row[3],
                "status": status,
                "created_at": float(row[5]),
                "last_touched": float(row[6]),
                "use_count": int(row[7] or 0),
                "activation": float(row[8] or 0.0),
                "graduated_at": float(row[9]) if row[9] is not None else None,
                "graduated_concept_id": row[10],
                "graduated_anchor_tx": row[11],
                "abandoned_at": float(row[12]) if row[12] is not None else None,
                "abandoned_tombstone_tx": row[13],
                "abandonment_reason": row[14],
            })

        payload = {
            "version": 1,
            "exported_at": self._clock(),
            "forks": forks,
            "summary": summary,
        }
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = snapshot_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp_path, snapshot_path)
        except Exception as e:
            logger.warning(
                "[HypothesisForkStore] export_snapshot write failed: %s", e,
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return 0
        return len(forks)

    # ── Internal: graduate-path orchestration ─────────────────────────

    def _graduate(
        self,
        *,
        fork: HypothesisFork,
        oracle_verdict: dict,
        concept_name: Optional[str],
    ) -> str:
        """Shared graduation path consumed by both `graduate_oracle` and
        `graduate_used`. Encapsulates the repair-vs-net-new branching +
        Merkle-root computation + canonical-TX emission + DuckDB/Kuzu state
        update.

        Returns the concept-version anchor_tx.
        """
        from titan_hcl.synthesis.merkle import merkle_root_hex
        exploration_tx_list = list(self._exploration_txs.get(fork.fork_id, ()))
        derivation_merkle_root = merkle_root_hex(exploration_tx_list)

        is_repair = fork.root_anchor is not None and fork.parent_concept_id is not None
        if not is_repair and concept_name is None:
            raise ValueError(
                "graduate (net-new): concept_name is required to materialize "
                "the brand-new concept; received None"
            )

        if is_repair:
            # Reuse Phase 4 ConceptStore.bump_version — INV-10 enforced by
            # the existing hard test (`test_bump_version_inserts_new_row_"
            # "without_mutating_parent`).
            parent_id = fork.parent_concept_id
            assert parent_id is not None  # narrowing for typecheck
            latest = self._graph.spine_get_latest_concept(parent_id)
            if latest is None:
                raise ForkStateError(
                    f"graduate (repair): parent {parent_id} has no spine row"
                )

            # Use bump_version directly — it handles the OuterMemoryWriter
            # call internally via the standard Phase 4 path, returning a
            # ConceptVersion whose anchor_tx is the canonical write hash.
            new_cv = self._concepts.bump_version(
                parent_id,
                composed_from=[],
                derivation_evidence=exploration_tx_list,
            )
            # Then emit the OracleVerdict TX so the verification is
            # provenanced (arch §11.1). The concept-version TX is already
            # anchored; we add a co-located verdict TX referencing it.
            self._writer.write_concept_version_with_proof(
                concept_id=new_cv.concept_id,
                version=new_cv.version,
                name=new_cv.name,
                memory_type=new_cv.memory_type,
                parent_version_tx=latest["anchor_tx"],
                composed_from=[],
                derivation_evidence=exploration_tx_list,
                groundedness=new_cv.groundedness,
                derivation_merkle_root=derivation_merkle_root,
                oracle_verdict=oracle_verdict,
            )
            # The first call (bump_version) emitted the canonical concept-
            # version TX; the second (write_concept_version_with_proof)
            # emitted a duplicate-shape concept-version TX. Both are
            # anchored, idempotent on-chain due to content-hashing; the
            # duplicate carries the proof + verdict. The anchor_tx that
            # ConceptStore returned (from bump_version's bare write) is
            # canonical; we use it as the chain handle.
            concept_anchor_tx = new_cv.anchor_tx
            new_concept_id = new_cv.concept_id
            new_version = new_cv.version
        else:
            # Net-new fork: synthesize a deterministic concept_id from the
            # name + a creation timestamp so two parallel forks with the
            # same intent don't collide.
            now = self._clock()
            seed = f"{concept_name}|{now:.6f}|{fork.fork_id}".encode()
            new_concept_id = hashlib.sha256(seed).hexdigest()[:16]
            new_cv = self._concepts.create_concept(
                new_concept_id,
                str(concept_name),
                "declarative",   # net-new earned concepts default to declarative
                composed_from=[],
                derivation_evidence=exploration_tx_list,
            )
            self._writer.write_concept_version_with_proof(
                concept_id=new_cv.concept_id,
                version=new_cv.version,
                name=new_cv.name,
                memory_type=new_cv.memory_type,
                parent_version_tx=None,
                composed_from=[],
                derivation_evidence=exploration_tx_list,
                groundedness=new_cv.groundedness,
                derivation_merkle_root=derivation_merkle_root,
                oracle_verdict=oracle_verdict,
            )
            concept_anchor_tx = new_cv.anchor_tx
            new_version = new_cv.version

        # Update lifecycle row.
        now = self._clock()
        self._db.execute(
            "UPDATE hypothesis_forks SET status='graduated', "
            "graduated_at=?, graduated_concept_id=?, graduated_anchor_tx=? "
            "WHERE fork_id=?",
            (now, new_concept_id, concept_anchor_tx, fork.fork_id),
        )
        self._graph.fork_update_status(
            fork_id=fork.fork_id, status="graduated",
            activation=fork.activation,
        )

        # Drop the in-memory exploration log; durable companion table is
        # cleaned by the cascade-GC sweep via purge_durable_state.
        self._exploration_txs.pop(fork.fork_id, None)

        logger.info(
            "[HypothesisForkStore] graduated fork=%s type=%s → "
            "concept=%s v%d anchor_tx=%s oracle=%s",
            fork.fork_id,
            "repair" if is_repair else "net_new",
            new_concept_id, new_version, concept_anchor_tx[:16],
            oracle_verdict.get("oracle_id", "?"),
        )
        # P8.X write-through: closes P5 cascade flake (snapshot lag on graduate)
        self._maybe_write_through_snapshot()
        return concept_anchor_tx


__all__ = (
    "HypothesisForkStore",
    "HypothesisFork",
    "ForkNotFound",
    "ForkStateError",
    "GraduationGateError",
    "USE_COUNT_GRADUATION_THRESHOLD",
    "DEFAULT_ACTIVATION_FLOOR",
    "DEFAULT_WINDOW_SEC",
)
