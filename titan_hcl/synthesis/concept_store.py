"""ConceptStore — sole writer of the Kuzu Concept spine (Phase 4 / §P4.B).

Per `ARCHITECTURE_synthesis_engine.md` §6 + §10 + INV-Syn-3 (extended) +
INV-3 + INV-4 + INV-10, this module is the ONLY surface authorized to
materialize spine concepts and bump their versions.

Invariants enforced HERE (high-level), not on the bare Kuzu helpers in
`direct_memory.py` (which are intentionally primitive so read-only consumers
can use them):

- **INV-3** — `bump_version()` NEVER mutates the parent row. It INSERTs
  `(concept_id, v+1)` as a new row. Parent stays immutable + on-chain.
  `recompute_groundedness()` updates only the *derived metric column*
  (not identity, version, lineage, or anchor_tx).

- **INV-4** — every `create_concept` + `bump_version` call ends with an
  `outer_memory_writer.write_concept_version(...)` call (P4.D). Writer
  failure rolls back the Kuzu insert before propagating the exception, so
  the spine never holds a row whose TX never landed on the chain.

- **INV-10** — `bump_version()` requires the parent concept_id to exist;
  raises `ParentVersionMissing` rather than silently creating a v=1 ghost.

- **INV-Syn-3 (extended → INV-Syn-7 proposed in P4.K)** — ConceptStore is
  only ever instantiated inside `synthesis_worker`. Cross-process readers
  go through `BridgeRecall.read_concept_spine()` (P4.H, watermark-gated).

The `outer_memory_writer` parameter is a duck-typed protocol — anything
exposing `.write_concept_version(concept_id, version, name, memory_type,
parent_version_tx, composed_from, derivation_evidence, groundedness,
derivation_merkle_root) -> str` (the returned TX hash) works. Tests
inject a fake; production wires the real OuterMemoryWriter (P4.D).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────


class ParentVersionMissing(Exception):
    """INV-10: bump_version() called with a concept_id that has no rows."""


class WriterFailure(Exception):
    """INV-4: outer_memory_writer.write_concept_version raised; the Kuzu
    insert was rolled back before this propagated. Wrapping preserves the
    cause for diagnosis."""


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class ConceptVersion:
    """The materialized result of create_concept / bump_version."""

    concept_id: str
    version: int
    name: str
    memory_type: str  # declarative|procedural|episodic|meta
    groundedness: float
    anchor_tx: str
    created_at: float


@dataclass(frozen=True)
class ConceptSpine:
    """Aggregate of all versions for one concept_id + composition edges.
    Returned by `read_spine()` / used by spine-aware recall (P4.H)."""

    concept_id: str
    latest_version: int
    versions: tuple[ConceptVersion, ...]
    composed_from: tuple[tuple[str, int], ...]  # (parent_concept_id, version)
    composed_into: tuple[tuple[str, int], ...]  # (child_concept_id, version)


# ── Writer protocol (duck-typed; real impl is P4.D) ─────────────────


class _ConceptVersionWriter(Protocol):
    def write_concept_version(
        self,
        *,
        concept_id: str,
        version: int,
        name: str,
        memory_type: str,
        parent_version_tx: Optional[str],
        composed_from: list[tuple[str, int]],
        derivation_evidence: list[str],
        groundedness: float,
        derivation_merkle_root: Optional[str] = None,
    ) -> str: ...


# ── Groundedness formula (§P4.E) ────────────────────────────────────


@dataclass
class _GroundednessParams:
    """Defaults match titan_params.toml [synthesis.groundedness]."""

    w_e: float = 0.3   # episodic encounters
    w_c: float = 0.3   # distinct contexts
    w_p: float = 0.4   # procedural links (highest in P4 since felt=0)
    w_f: float = 0.0   # felt-state coverage (Phase 7+ populates)


def _load_groundedness_params_from_toml() -> _GroundednessParams:
    """Best-effort load of `[synthesis.groundedness]` from titan_params.toml.

    Returns the in-code defaults on any error (file missing, parse error,
    subtable missing) so tests + unit scenarios that don't need the toml
    keep working. Production synthesis_worker calls this at boot."""
    try:
        import importlib.resources
        try:
            import tomllib  # 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore
        # The params file ships under titan_hcl/titan_params.toml.
        # In production the worker imports it from its installed package;
        # in tests we resolve via the parent of titan_hcl/synthesis/.
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "titan_params.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        sub = data.get("synthesis", {}).get("groundedness", {})
        return _GroundednessParams(
            w_e=float(sub.get("w_e", 0.3)),
            w_c=float(sub.get("w_c", 0.3)),
            w_p=float(sub.get("w_p", 0.4)),
            w_f=float(sub.get("w_f", 0.0)),
        )
    except Exception:
        return _GroundednessParams()


def _norm_log_count(n: int | float) -> float:
    """Smooth normalization for an unbounded count: log1p / log(1+50) so a
    count of 50 saturates to ~1.0. Cheap, no candidate-pool dependency, and
    safe for first-version concepts where the count is 0."""
    if n <= 0:
        return 0.0
    return min(1.0, math.log1p(float(n)) / math.log1p(50.0))


def compute_groundedness(
    *,
    episodic_encounters: int,
    distinct_contexts: int,
    procedural_links: int,
    felt_coverage: float = 0.0,
    params: _GroundednessParams | None = None,
) -> float:
    """Per `PLAN_synthesis_engine_Phase4.md §P4.E`. Returns groundedness
    in [0.0, 1.0]. Each count term is log-normalized so any single term
    saturates at ~50; the weighted sum is clamped to [0, 1].

    `felt_coverage` is a [0, 1] float owned by the inner-outer bridge
    (Phase 7+); in P4 callers pass 0.0 and `w_f` is also 0.0 so the term
    is identically zero. Kept in the signature so the bridge phase only
    has to flip a parameter, not change the formula or callers."""
    p = params or _GroundednessParams()
    e = _norm_log_count(episodic_encounters)
    c = _norm_log_count(distinct_contexts)
    pr = _norm_log_count(procedural_links)
    raw = p.w_e * e + p.w_c * c + p.w_p * pr + p.w_f * felt_coverage
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


# ── ConceptStore ────────────────────────────────────────────────────


class ConceptStore:
    """Sole writer of the Kuzu Concept spine. Instantiate ONCE inside
    `synthesis_worker` (INV-Syn-3 / INV-11)."""

    def __init__(
        self,
        graph: Any,                             # TitanKnowledgeGraph
        outer_memory_writer: _ConceptVersionWriter,
        *,
        groundedness_params: _GroundednessParams | None = None,
        clock: Any = time.time,                  # injectable for deterministic tests
    ):
        self._graph = graph
        self._writer = outer_memory_writer
        self._params = groundedness_params or _GroundednessParams()
        self._clock = clock

    # ── Public surface ────────────────────────────────────────

    def create_concept(
        self,
        concept_id: str,
        name: str,
        memory_type: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
    ) -> ConceptVersion:
        """Materialize a brand-new spine concept at v=1.

        Caller MUST have registered `concept_id` with CGN first (P4.C). This
        method does NOT touch CGN — that's the caller's job (separation of
        concerns: CGN registration belongs to the orchestrator, P4.G).

        Sequence (INV-4):
          1. Compute initial groundedness from supplied composed_from depth.
          2. Anchor the v=1 TX via outer_memory_writer (parent_version_tx=None).
          3. INSERT the Concept row into Kuzu, anchor_tx = writer-returned hash.
          4. Best-effort COMPOSED_FROM + COMPOSED_INTO edges to each base.

        If step 2 raises, step 3 never runs (no orphan Kuzu row).
        If step 3 raises after a successful step 2, we keep going — the TX is
        canonical, the Kuzu row is a derived index that can be rebuilt from
        the chain (INV-2). We log a HARD warning so it surfaces.
        """
        if memory_type not in ("declarative", "procedural", "episodic", "meta"):
            raise ValueError(f"invalid memory_type: {memory_type!r}")
        composed_from = list(composed_from or [])
        derivation_evidence = list(derivation_evidence or [])

        # The v=1 row has zero episodic/context/procedural history yet —
        # groundedness starts at the contribution from composed_from depth
        # (1 link per base, capped by the log-normalizer).
        initial_groundedness = compute_groundedness(
            episodic_encounters=0,
            distinct_contexts=0,
            procedural_links=len(composed_from),
            felt_coverage=0.0,
            params=self._params,
        )

        # Step 1: anchor the v=1 TX. Writer failure aborts the whole op.
        try:
            tx_hash = self._writer.write_concept_version(
                concept_id=concept_id,
                version=1,
                name=name,
                memory_type=memory_type,
                parent_version_tx=None,  # v=1 — genesis of this spine concept
                composed_from=composed_from,
                derivation_evidence=derivation_evidence,
                groundedness=initial_groundedness,
                derivation_merkle_root=None,  # Phase 5 wires this
            )
        except Exception as e:
            logger.error(
                "[ConceptStore] create_concept(%s) writer failed: %s",
                concept_id, e,
            )
            raise WriterFailure(
                f"write_concept_version failed for {concept_id} v=1: {e}"
            ) from e

        # Step 2: INSERT the row.
        created = self._graph.spine_create_concept_node(
            concept_id=concept_id, version=1, name=name,
            memory_type=memory_type, groundedness=initial_groundedness,
            anchor_tx=tx_hash, created_at=self._clock(),
        )
        if not created:
            # Pre-existing row at v=1 — surfaces a logic error in the
            # caller (P4.G must dedup before calling create_concept).
            # We don't roll back the TX (it's canonical) but we surface
            # the divergence loudly so the operator notices.
            logger.warning(
                "[ConceptStore] create_concept(%s) found existing v=1 row "
                "after a successful TX anchor — spine + chain may diverge",
                concept_id,
            )

        # Step 3: composition edges. Best-effort; missing endpoints are
        # logged but do not abort (the parent concept may have been
        # GC'd or may exist only on chain).
        self._maintain_composition_edges(
            from_concept_id=concept_id, from_version=1,
            composed_from=composed_from,
        )

        return ConceptVersion(
            concept_id=concept_id, version=1, name=name,
            memory_type=memory_type, groundedness=initial_groundedness,
            anchor_tx=tx_hash, created_at=self._clock(),
        )

    def bump_version(
        self,
        concept_id: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
        groundedness_at_bump: Optional[float] = None,
    ) -> ConceptVersion:
        """Insert v(n+1) for an existing concept. INV-3: parent stays
        immutable; INV-10: parent MUST exist or ParentVersionMissing raises.
        """
        latest = self._graph.spine_get_latest_concept(concept_id)
        if latest is None:
            raise ParentVersionMissing(
                f"bump_version({concept_id}): no existing version found "
                "(call create_concept first)"
            )

        composed_from = list(composed_from or [])
        derivation_evidence = list(derivation_evidence or [])
        new_version = latest["version"] + 1
        parent_tx = latest["anchor_tx"]
        name = latest["name"]
        memory_type = latest["memory_type"]
        # Default groundedness for the bump: recompute fresh; caller may
        # override with `groundedness_at_bump` if they already have a value.
        if groundedness_at_bump is None:
            groundedness_at_bump = compute_groundedness(
                episodic_encounters=0,  # consolidation-pass-side compute
                distinct_contexts=0,
                procedural_links=len(composed_from),
                felt_coverage=0.0,
                params=self._params,
            )

        try:
            tx_hash = self._writer.write_concept_version(
                concept_id=concept_id,
                version=new_version,
                name=name,
                memory_type=memory_type,
                parent_version_tx=parent_tx,
                composed_from=composed_from,
                derivation_evidence=derivation_evidence,
                groundedness=groundedness_at_bump,
                derivation_merkle_root=None,
            )
        except Exception as e:
            logger.error(
                "[ConceptStore] bump_version(%s,v%d) writer failed: %s",
                concept_id, new_version, e,
            )
            raise WriterFailure(
                f"write_concept_version failed for {concept_id} "
                f"v={new_version}: {e}"
            ) from e

        created = self._graph.spine_create_concept_node(
            concept_id=concept_id, version=new_version, name=name,
            memory_type=memory_type, groundedness=groundedness_at_bump,
            anchor_tx=tx_hash, created_at=self._clock(),
        )
        if not created:
            logger.warning(
                "[ConceptStore] bump_version(%s,v%d) found existing row "
                "after TX anchor — spine + chain may diverge",
                concept_id, new_version,
            )

        # Composition edges: each parent in composed_from must get both a
        # COMPOSED_FROM (new_version → parent) and COMPOSED_INTO
        # (new_version → parent's later compositions) edge. Per §10 the
        # bidirectional edge model means both directions are added for
        # parent-child traversal.
        self._maintain_composition_edges(
            from_concept_id=concept_id, from_version=new_version,
            composed_from=composed_from,
        )

        return ConceptVersion(
            concept_id=concept_id, version=new_version, name=name,
            memory_type=memory_type, groundedness=groundedness_at_bump,
            anchor_tx=tx_hash, created_at=self._clock(),
        )

    def add_composition_edge(
        self,
        from_concept: tuple[str, int],
        to_concept: tuple[str, int],
        direction: str = "from",
    ) -> bool:
        """Explicit composition-edge maintenance — exposed for the rare case
        consolidation discovers a back-edge after the fact (e.g. v3 turns
        out to also derive from a concept already in the graph but not
        listed in derivation_evidence at bump time). Direction semantics
        match `TitanKnowledgeGraph.spine_add_composition_edge`."""
        return self._graph.spine_add_composition_edge(
            from_concept[0], from_concept[1],
            to_concept[0], to_concept[1],
            direction=direction,
        )

    def recompute_groundedness(
        self,
        concept_id: str,
        version: int,
        *,
        episodic_encounters: int = 0,
        distinct_contexts: int = 0,
        procedural_links: int = 0,
        felt_coverage: float = 0.0,
    ) -> float:
        """Compute + persist groundedness for a single (concept_id, version).
        Returns the new value (0.0 if the row is missing)."""
        new_value = compute_groundedness(
            episodic_encounters=episodic_encounters,
            distinct_contexts=distinct_contexts,
            procedural_links=procedural_links,
            felt_coverage=felt_coverage,
            params=self._params,
        )
        updated = self._graph.spine_update_groundedness(
            concept_id, version, new_value,
        )
        if not updated:
            return 0.0
        return new_value

    def recompute_groundedness_batch(
        self,
        rows: Iterable[dict],
    ) -> int:
        """Apply recompute_groundedness across a batch. Each input row is
        a dict with keys: concept_id, version, episodic_encounters,
        distinct_contexts, procedural_links, felt_coverage (optional).
        Returns the number of rows successfully updated."""
        n = 0
        for r in rows:
            try:
                self.recompute_groundedness(
                    r["concept_id"], int(r["version"]),
                    episodic_encounters=int(r.get("episodic_encounters", 0)),
                    distinct_contexts=int(r.get("distinct_contexts", 0)),
                    procedural_links=int(r.get("procedural_links", 0)),
                    felt_coverage=float(r.get("felt_coverage", 0.0)),
                )
                n += 1
            except Exception as e:
                logger.warning(
                    "[ConceptStore] recompute_groundedness_batch row failed: "
                    "%r: %s", r, e,
                )
        return n

    # ── Internal helpers ─────────────────────────────────────

    def _maintain_composition_edges(
        self,
        from_concept_id: str,
        from_version: int,
        composed_from: list[tuple[str, int]],
    ) -> None:
        """Best-effort composition-edge insertion. Missing endpoints are
        logged at DEBUG; the spine is rebuildable from chain so a missed
        edge is recoverable (INV-2)."""
        for parent_id, parent_ver in composed_from:
            ok_from = self._graph.spine_add_composition_edge(
                from_concept_id, from_version, parent_id, parent_ver,
                direction="from",
            )
            ok_into = self._graph.spine_add_composition_edge(
                parent_id, parent_ver, from_concept_id, from_version,
                direction="into",
            )
            if not (ok_from or ok_into):
                logger.debug(
                    "[ConceptStore] composition edge %s v%d ↔ %s v%d skipped"
                    " (likely missing endpoint)",
                    from_concept_id, from_version, parent_id, parent_ver,
                )


__all__ = (
    "ConceptStore",
    "ConceptVersion",
    "ConceptSpine",
    "ParentVersionMissing",
    "WriterFailure",
    "compute_groundedness",
    "_GroundednessParams",
    "_load_groundedness_params_from_toml",
)
