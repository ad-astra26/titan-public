"""EngramStore — sole writer of the Kuzu Concept spine (Phase 4 / §P4.B).

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

- **INV-Syn-3 (extended → INV-Syn-7 proposed in P4.K)** — EngramStore is
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
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from titan_hcl.synthesis.writer import on_writer, resolve_writer
from titan_hcl.synthesis.grounding_combiner import GroundingCombiner

logger = logging.getLogger(__name__)

# §7.E — default persisted location of the learned grounding combiner. CWD-
# relative `data/`, matching synthesis.duckdb's default (synthesis_worker runs
# with CWD=repo root); overridable via EngramStore(combiner_path=…) for tests.
_DEFAULT_COMBINER_PATH = os.path.join("data", "grounding_combiner.json")


# ── Exceptions ──────────────────────────────────────────────────────


class ParentVersionMissing(Exception):
    """INV-10: bump_version() called with a concept_id that has no rows."""


class WriterFailure(Exception):
    """INV-4: outer_memory_writer.write_concept_version raised; the Kuzu
    insert was rolled back before this propagated. Wrapping preserves the
    cause for diagnosis."""


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class Engram:
    """The materialized result of create_concept / bump_version."""

    concept_id: str
    version: int
    name: str
    memory_type: str  # declarative|procedural|episodic|meta
    groundedness: float
    anchor_tx: str
    created_at: float


@dataclass(frozen=True)
class EngramSpine:
    """Aggregate of all versions for one concept_id + composition edges.
    Returned by `read_spine()` / used by spine-aware recall (P4.H)."""

    concept_id: str
    latest_version: int
    versions: tuple[Engram, ...]
    composed_from: tuple[tuple[str, int], ...]  # (parent_concept_id, version)
    composed_into: tuple[tuple[str, int], ...]  # (child_concept_id, version)


# ── Writer protocol (duck-typed; real impl is P4.D) ─────────────────


class _EngramWriter(Protocol):
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


# ── Phase D — decomposed axes + population-percentile reduction (§7.D) ────────
# BRAIN §3.4 anti-collapse: store the 4 grounding axes ABSOLUTE/independent; the
# recall-ranking scalar is a derived percentile-normalized blend recomputed across
# the live population at the dream boundary (guaranteed discrimination, no magic
# saturation constant). At launch only `used` varies (verified/felt sparse,
# fluent=0); the variance-gated blend lets `used`'s percentile spread drive the
# scalar across [0,1], with the other axes enriching forward.

_AXIS_NAMES = ("used", "verified", "felt", "fluent")


@dataclass
class _BlendParams:
    """[synthesis.engram.grounding] — percentile-blend weights + NARS horizon.
    Weights lean on `used` at launch (the live signal); variance-gating means a
    flat axis is excluded + the remaining weights renormalize, so the spread is
    always driven by whatever actually varies."""

    w_used: float = 0.40
    w_verified: float = 0.0   # procedural-only; N/A on thought-Engrams (§6.2.3 spine-partition / BUG-ENGRAM-AXIS-VERIFIED) — structurally 0, excluded from the thought-Engram blend
    w_felt: float = 0.20
    w_fluent: float = 0.15
    nars_k: float = 1.0


def _load_blend_params_from_toml() -> _BlendParams:
    """Best-effort load of `[synthesis.engram.grounding]`; in-code defaults on any error."""
    try:
        try:
            import tomllib  # 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "titan_params.toml"), "rb") as f:
            data = tomllib.load(f)
        sub = data.get("synthesis", {}).get("engram", {}).get("grounding", {})
        return _BlendParams(
            w_used=float(sub.get("w_used", 0.40)),
            w_verified=float(sub.get("w_verified", 0.0)),
            w_felt=float(sub.get("w_felt", 0.20)),
            w_fluent=float(sub.get("w_fluent", 0.15)),
            nars_k=float(sub.get("nars_k", 1.0)),
        )
    except Exception:
        return _BlendParams()


def compute_axes(
    *,
    provisional: float = 0.0,
    oracle_evidence: int = 0,
    felt_coverage: float = 0.0,
    fluent: float = 0.0,
    nars_k: float = 1.0,
) -> dict[str, float]:
    """The 4 grounding axes stored on the Engram (§6.2.3 / §7.D, option-1 v1):
      used     = the **provisional grounding blend** (`compute_groundedness`) — the
                 consistent, stable rank base across the WHOLE population (pre-D
                 Engrams are backfilled from their stored `groundedness`, same
                 scale, no magic constant). The pure log1p-B_i decomposition
                 graduates with the §7.E learned reduction.
      verified = NARS c = n/(n+k), n = oracle scored_by evidence count.
                 PROCEDURAL-SPINE ONLY (§6.2.3 spine-partition): thought-Engram
                 consolidation passes no oracle_evidence (members are thoughts →
                 no scored_by) → 0; w_verified=0 in the blend. Column kept for
                 BRAIN-compat (BRAIN derives c from procedural TXs at ingest).
      felt     = felt_coverage (§7.C; [0,1])
      fluent   = recall-citation rate ([0,1]; 0 at launch — attribution deferred)
    The recall scalar is the population percentile-blend of these (reduce step)."""
    n = max(0, int(oracle_evidence))
    return {
        "used": float(provisional),
        "verified": (n / (n + nars_k)) if (n + nars_k) > 0 else 0.0,
        "felt": max(0.0, min(1.0, float(felt_coverage))),
        "fluent": max(0.0, min(1.0, float(fluent))),
    }


def _percentile_ranks(values: list[float]) -> list[float]:
    """Empirical-CDF percentile rank ∈ (0,1] for each value (fraction ≤ v). A
    constant population maps every value to 1.0 (no discrimination — the caller
    variance-gates flat axes out of the blend)."""
    import bisect
    n = len(values)
    if n == 0:
        return []
    srt = sorted(values)
    return [bisect.bisect_right(srt, v) / n for v in values]


def reduce_population_to_scalars(
    axes_rows: list[dict], params: _BlendParams | None = None,
) -> dict[str, float]:
    """Map [{"key", used, verified, felt, fluent}] → {key: groundedness scalar}.
    Percentile-normalize EACH axis across the population, then blend ONLY the axes
    that actually vary (variance > 1e-9), renormalizing their weights — so the
    scalar's spread is driven by whatever signal exists (at launch: `used` alone →
    full [0,1]). All-flat population → all 0.0 (nothing to discriminate on)."""
    p = params or _BlendParams()
    if not axes_rows:
        return {}
    weights = {"used": p.w_used, "verified": p.w_verified,
               "felt": p.w_felt, "fluent": p.w_fluent}
    pct: dict[str, list[float]] = {}
    active: list[str] = []
    for ax in _AXIS_NAMES:
        vals = [float(r.get(ax, 0.0) or 0.0) for r in axes_rows]
        pct[ax] = _percentile_ranks(vals)
        # An axis joins the blend only if it BOTH varies AND carries positive
        # weight. The weight>0 guard makes a zero-weight axis (today: `verified`,
        # procedural-only per §6.2.3 spine-partition) a true exclusion — even if a
        # stray value ever makes it vary, it cannot drive the scalar (which would
        # otherwise collapse all scalars to 0 when it is the sole varying axis).
        if (max(vals) - min(vals)) > 1e-9 and weights[ax] > 1e-12:
            active.append(ax)
    if not active:
        # No axis varies (single Engram, or a population not yet carrying axis
        # data) → percentile can't discriminate. Return {} so the caller KEEPS
        # each Engram's provisional scalar rather than zeroing it.
        return {}
    wsum = sum(weights[ax] for ax in active) or 1.0
    out: dict[str, float] = {}
    for i, r in enumerate(axes_rows):
        s = sum((weights[ax] / wsum) * pct[ax][i] for ax in active)
        out[r["key"]] = max(0.0, min(1.0, s))
    return out


# ── EngramStore ────────────────────────────────────────────────────


class EngramStore:
    """Sole writer of the Kuzu Concept spine. Instantiate ONCE inside
    `synthesis_worker` (INV-Syn-3 / INV-11)."""

    def __init__(
        self,
        graph: Any,                             # TitanKnowledgeGraph
        outer_memory_writer: _EngramWriter,
        *,
        groundedness_params: _GroundednessParams | None = None,
        clock: Any = time.time,                  # injectable for deterministic tests
        db_writer: Any = None,
        combiner_path: str | None = None,
    ):
        self._graph = graph
        self._writer = outer_memory_writer
        # Single-writer-thread (Option C): the Kuzu spine graph is touched only
        # on the one SynthesisWriter thread (the @on_writer methods below) —
        # consolidation writes, recompute exports, and fork-gc prunes can no
        # longer race on it. Named `_db_writer` because `_writer` holds the
        # bus-only OuterMemoryWriter. INV-Syn-7 sole writer preserved. Tests
        # inject none → InlineWriter runs ops inline.
        self._db_writer = resolve_writer(db_writer)
        self._params = groundedness_params or _GroundednessParams()
        self._blend = _load_blend_params_from_toml()  # §7.D percentile-blend params
        self._clock = clock
        # §7.E — the learned grounding combiner. Loaded once; inactive (→ §7.D
        # blend) until a train-step beats the blend on held-out citation AUC.
        self._combiner_path = combiner_path or _DEFAULT_COMBINER_PATH
        self._combiner = GroundingCombiner.load(self._combiner_path)

    # ── Public surface ────────────────────────────────────────

    @on_writer
    def create_concept(
        self,
        concept_id: str,
        name: str,
        memory_type: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
        derivation_merkle_root: Optional[str] = None,
        oracle_verdict: Optional[dict] = None,
        domain_hint: str = "",
    ) -> Engram:
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

        # Step 1: anchor the v=1 TX. Writer failure aborts the whole op. G4
        # (AUDIT §5.3): when a proof (oracle_verdict) is supplied — fork
        # graduation — emit the SINGLE canonical concept-version TX WITH the
        # derivation merkle root + companion verdict (INV-4 single write), NOT a
        # bare TX the caller then DUPLICATES with a second
        # write_concept_version_with_proof.
        try:
            if oracle_verdict is not None:
                tx_hash, _verdict_tx = self._writer.write_concept_version_with_proof(
                    concept_id=concept_id,
                    version=1,
                    name=name,
                    memory_type=memory_type,
                    parent_version_tx=None,
                    composed_from=composed_from,
                    derivation_evidence=derivation_evidence,
                    groundedness=initial_groundedness,
                    derivation_merkle_root=derivation_merkle_root,
                    oracle_verdict=oracle_verdict,
                )
            else:
                tx_hash = self._writer.write_concept_version(
                    concept_id=concept_id,
                    version=1,
                    name=name,
                    memory_type=memory_type,
                    parent_version_tx=None,  # v=1 — genesis of this spine concept
                    composed_from=composed_from,
                    derivation_evidence=derivation_evidence,
                    groundedness=initial_groundedness,
                    derivation_merkle_root=derivation_merkle_root,
                )
        except Exception as e:
            logger.error(
                "[EngramStore] create_concept(%s) writer failed: %s",
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
            domain_hint=domain_hint,
        )
        if not created:
            # Pre-existing row at v=1 — surfaces a logic error in the
            # caller (P4.G must dedup before calling create_concept).
            # We don't roll back the TX (it's canonical) but we surface
            # the divergence loudly so the operator notices.
            logger.warning(
                "[EngramStore] create_concept(%s) found existing v=1 row "
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

        return Engram(
            concept_id=concept_id, version=1, name=name,
            memory_type=memory_type, groundedness=initial_groundedness,
            anchor_tx=tx_hash, created_at=self._clock(),
        )

    @on_writer
    def bump_version(
        self,
        concept_id: str,
        composed_from: Optional[list[tuple[str, int]]] = None,
        derivation_evidence: Optional[list[str]] = None,
        groundedness_at_bump: Optional[float] = None,
        derivation_merkle_root: Optional[str] = None,
        oracle_verdict: Optional[dict] = None,
        domain_hint: str = "",
    ) -> Engram:
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
            # G4 (AUDIT §5.3): with a proof (oracle_verdict, fork graduation),
            # emit the SINGLE canonical version TX WITH merkle root + companion
            # verdict (INV-4 single write); else a bare version TX.
            if oracle_verdict is not None:
                tx_hash, _verdict_tx = self._writer.write_concept_version_with_proof(
                    concept_id=concept_id,
                    version=new_version,
                    name=name,
                    memory_type=memory_type,
                    parent_version_tx=parent_tx,
                    composed_from=composed_from,
                    derivation_evidence=derivation_evidence,
                    groundedness=groundedness_at_bump,
                    derivation_merkle_root=derivation_merkle_root,
                    oracle_verdict=oracle_verdict,
                )
            else:
                tx_hash = self._writer.write_concept_version(
                    concept_id=concept_id,
                    version=new_version,
                    name=name,
                    memory_type=memory_type,
                    parent_version_tx=parent_tx,
                    composed_from=composed_from,
                    derivation_evidence=derivation_evidence,
                    groundedness=groundedness_at_bump,
                    derivation_merkle_root=derivation_merkle_root,
                )
        except Exception as e:
            logger.error(
                "[EngramStore] bump_version(%s,v%d) writer failed: %s",
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
            domain_hint=domain_hint,
        )
        if not created:
            logger.warning(
                "[EngramStore] bump_version(%s,v%d) found existing row "
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

        return Engram(
            concept_id=concept_id, version=new_version, name=name,
            memory_type=memory_type, groundedness=groundedness_at_bump,
            anchor_tx=tx_hash, created_at=self._clock(),
        )

    @on_writer
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

    @on_writer
    def recompute_groundedness(
        self,
        concept_id: str,
        version: int,
        *,
        episodic_encounters: int = 0,
        distinct_contexts: int = 0,
        procedural_links: int = 0,
        felt_coverage: float = 0.0,
        oracle_evidence: int = 0,
    ) -> float:
        """Store the provisional grounding blend + the 4 axes (§7.D, option-1) for
        this (concept_id, version). `axis_used` = the provisional blend (the stable
        rank base); the scalar is refined to the population percentile-blend by
        `recompute_population_groundedness` at the dream boundary (this provisional
        keeps recall sane between passes). Returns the provisional (0.0 if missing)."""
        new_value = compute_groundedness(
            episodic_encounters=episodic_encounters,
            distinct_contexts=distinct_contexts,
            procedural_links=procedural_links,
            felt_coverage=felt_coverage,
            params=self._params,
        )
        axes = compute_axes(
            provisional=new_value,  # the consistent, stable rank base (§7.D option-1)
            oracle_evidence=oracle_evidence,
            felt_coverage=felt_coverage,
            fluent=0.0,  # §7.E.0: fresh-version seed; the LIVE recall-citation rate
                         # is injected at the dream population recompute (fluent_lookup)
            nars_k=self._blend.nars_k,
        )
        updated = self._graph.spine_update_groundedness(
            concept_id, version, new_value, axes=axes,
        )
        if not updated:
            return 0.0
        return new_value

    @on_writer
    def recompute_population_groundedness(
        self,
        fluent_lookup: Optional[Callable[[str, int], Optional[float]]] = None,
        axes_sink: Optional[Callable[[list[dict]], None]] = None,
    ) -> int:
        """Dream-boundary pass (§7.D, option-1): read EVERY Engram's axes, percentile-
        normalize the population (variance-gated blend) → write the `groundedness`
        scalar back. `axis_used` is the consistent rank base (the provisional blend);
        pre-D Engrams whose `axis_used` is still 0 are BACKFILLED once from their
        stored `groundedness` (same scale) so the WHOLE population discriminates from
        the first pass. This is what makes groundedness DISCRIMINATE (vs the old
        saturate-at-50 scalar). Returns rows updated.

        §7.E.0: `fluent_lookup(concept_id, version) → float|None` injects the LIVE
        recall-citation rate over the stored `axis_fluent` so the 4th axis varies and
        enters the blend; `axes_sink(rows)` receives the final per-Engram axes for the
        recall-event snapshot cache. Both optional/soft — None = §7.D behaviour."""
        try:
            rows = self._graph.spine_read_engram_axes()
        except Exception as e:
            logger.warning("[EngramStore] recompute_population: read axes failed: %s", e)
            return 0
        if not rows:
            return 0
        for r in rows:
            r["key"] = (r["concept_id"], int(r["version"]))
            # §7.E.0 — override the stored (stale) fluent with the live citation rate.
            if fluent_lookup is not None:
                try:
                    fl = fluent_lookup(r["concept_id"], int(r["version"]))
                except Exception:
                    fl = None
                if fl is not None:
                    r["fluent"] = float(fl)
                    r["_fluent_updated"] = True
            # One-time backfill: a pre-D Engram has axis_used==0 but a real
            # provisional `groundedness` — seed axis_used from it (same scale) so
            # the population shares one rank base. Marked for persistence below.
            if (r.get("used") or 0.0) <= 0.0 and (r.get("groundedness") or 0.0) > 0.0:
                r["used"] = float(r["groundedness"])
                r["_backfill_used"] = True
        # §7.E — reduce to the recall scalar via the LEARNED combiner when it is
        # active (O(1) per-Engram dot-product over the raw axes), else the §7.D
        # population percentile-blend. The combiner only activates after beating
        # the blend on held-out citation AUC, so this is a safe, self-gated swap.
        if self._combiner.is_active():
            scalars = {r["key"]: self._combiner.score(r) for r in rows}
        else:
            scalars = reduce_population_to_scalars(rows, self._blend)
        n = 0
        for r in rows:
            sc = scalars.get(r["key"])
            if sc is None:
                continue
            # Persist the backfilled axis_used + live fluent alongside the new
            # scalar (one write per row).
            axes: dict = {}
            if r.get("_backfill_used"):
                axes["used"] = r["used"]
            if r.get("_fluent_updated"):
                axes["fluent"] = r["fluent"]
            try:
                if self._graph.spine_update_groundedness(
                        r["concept_id"], int(r["version"]), float(sc),
                        axes=(axes or None)):
                    n += 1
            except Exception as e:
                logger.warning(
                    "[EngramStore] recompute_population: update %s v%s failed: %s",
                    r["concept_id"], r["version"], e)
        # §7.E.0 — denormalize the fresh axes for the recall-event snapshot cache.
        if axes_sink is not None:
            try:
                axes_sink([
                    {"concept_id": r["concept_id"], "version": int(r["version"]),
                     "used": float(r.get("used") or 0.0),
                     "verified": float(r.get("verified") or 0.0),
                     "felt": float(r.get("felt") or 0.0),
                     "fluent": float(r.get("fluent") or 0.0)}
                    for r in rows])
            except Exception as e:
                logger.debug("[EngramStore] axes_sink soft-fail: %s", e)
        logger.info(
            "[EngramStore] population groundedness recomputed: %d/%d Engrams "
            "(percentile-blend, §7.D%s)", n, len(rows),
            "; fluent live" if fluent_lookup is not None else "")
        return n

    def train_grounding_combiner(self, events: Iterable[tuple]) -> dict:
        """§7.E — offline train-step (dream boundary / boot; idle-gated by the
        caller). Retrain the learned combiner on the recall-citation `events`
        ([(axes-record, cited-bool)]); it SELF-GATES (activates only if it beats
        the §7.D percentile-blend on held-out citation AUC) and is hot-swapped +
        persisted in place. The blend baseline is the very reduction it would
        replace, computed via `reduce_population_to_scalars` on the holdout (the
        callable is injected to avoid a grounding_combiner→engram_store circular
        import). Soft-fail → keep prior state. Returns the metrics dict."""
        try:
            def _blend_scorer(axes_list: list[list[float]]) -> list[float]:
                rows = [{"key": i, "used": a[0], "verified": a[1],
                         "felt": a[2], "fluent": a[3]}
                        for i, a in enumerate(axes_list)]
                sc = reduce_population_to_scalars(rows, self._blend)
                # {} (no axis varies) → neutral 0.5 baseline (chance).
                return [sc.get(i, 0.5) for i in range(len(axes_list))]
            metrics = self._combiner.train(
                events, blend_scorer=_blend_scorer, clock=self._clock)
            self._combiner.save(self._combiner_path)
            return metrics
        except Exception as e:
            logger.warning("[EngramStore] train_grounding_combiner failed: %s", e)
            return {"error": str(e), "activated": False}

    def backfill_domain_hints(self, derive_fn: Callable[..., str]) -> int:
        """One-time content backfill of the advisory §7.F `domain_hint` for pre-
        Phase-F Engrams (BUG-ENGRAM-DOMAIN-HINT-NOT-BACKFILLED). Phase F set
        `domain_hint` only at consolidation time (the LLM `DOMAIN:` line);
        Engrams that predate it carry "". This pass reads every Engram and, for
        each whose `domain_hint` is blank, derives a coarse advisory hint from
        its NAME via `derive_fn(name, memory_type) -> str` (the cheap classifier
        sanctioned by the §7.F fix-plan as the alternative to re-running the
        LLM) and SETs it. Engrams that already carry a hint are NEVER touched; a
        derive of "" leaves the row blank (test artifacts / low-confidence).

        IDEMPOTENT + cheap (bounded Engram count) — safe to run every boot,
        mirroring the §7.D population-recompute precedent. Runs INSIDE
        synthesis_worker (the sole SynthesisWriter, G21) so it owns the Kuzu RW
        handle — no external-process lock conflict. Returns rows updated."""
        try:
            rows = self._graph.spine_read_engram_domain_rows()
        except Exception as e:
            logger.warning("[EngramStore] backfill_domain_hints: read failed: %s", e)
            return 0
        if not rows:
            return 0
        n = 0
        for r in rows:
            if (r.get("domain_hint") or "").strip():
                continue  # already labeled — never overwrite (mutable, advisory)
            try:
                dh = derive_fn(r.get("name") or "", r.get("memory_type") or "")
            except Exception as e:
                logger.debug(
                    "[EngramStore] backfill_domain_hints: derive %s failed: %s",
                    r.get("concept_id"), e)
                continue
            if not dh:
                continue  # leave blank (test artifact / low-confidence)
            try:
                if self._graph.spine_set_domain_hint(
                        r["concept_id"], int(r["version"]), dh):
                    n += 1
            except Exception as e:
                logger.warning(
                    "[EngramStore] backfill_domain_hints: set %s v%s failed: %s",
                    r["concept_id"], r["version"], e)
        if n:
            logger.info(
                "[EngramStore] domain_hint backfill: %d/%d Engrams labeled "
                "(§7.F)", n, len(rows))
        return n

    @on_writer
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
                    "[EngramStore] recompute_groundedness_batch row failed: "
                    "%r: %s", r, e,
                )
        return n

    # ── Snapshot export (cross-process read surface — FU-1) ──────────

    @on_writer
    def export_snapshot(self, snapshot_path: str) -> int:
        """Atomic JSON export of the full spine state for cross-process
        readers. Mirrors the standing_store + activation_snapshot pattern:
        synthesis_worker is the sole writer (G21); cross-process consumers
        (api process for `/v6/synthesis/engrams/*`) read this JSON
        snapshot — the same Kuzu file CANNOT be opened read-only against
        an active RW writer in Kuzu 0.11 (read_only=True still acquires
        the exclusive lock), so the snapshot pattern is the canonical
        cross-process read surface.

        Schema:
            {
              "version": 1,
              "exported_at": <wall-clock seconds>,
              "concepts": [
                {concept_id, version, name, memory_type, groundedness,
                 anchor_tx, created_at},
                ...
              ],  # ALL versions of every concept (sorted concept_id asc, version asc)
              "composition_edges": {
                "from": [[(from_id, from_ver), (to_id, to_ver)], ...],
                "into": [[(from_id, from_ver), (to_id, to_ver)], ...]
              }
            }

        Atomic write via tmp + os.replace. Returns the total concept-row
        count in the snapshot (NOT the latest-version count — full
        history). Empty graph → returns 0 and writes a minimal payload
        so the api endpoint can distinguish "spine empty" from "snapshot
        missing".
        """
        concepts: list[dict] = []
        try:
            qr = self._graph._conn.execute(
                "MATCH (c:Engram) "
                "RETURN c.concept_id, c.version, c.name, c.memory_type, "
                "c.groundedness, c.anchor_tx, c.created_at, c.axis_used, "
                "c.axis_verified, c.axis_felt, c.axis_fluent, c.domain_hint"
            )
            while qr.has_next():
                row = qr.get_next()
                concepts.append({
                    "concept_id": row[0],
                    "version": int(row[1]),
                    "name": row[2],
                    "memory_type": row[3],
                    "groundedness": float(row[4]),
                    "anchor_tx": row[5],
                    "created_at": float(row[6]),
                    # The 4 decomposed grounding axes (§7.D), visible on the API.
                    "axis_used": float(row[7]) if row[7] is not None else 0.0,
                    "axis_verified": float(row[8]) if row[8] is not None else 0.0,
                    "axis_felt": float(row[9]) if row[9] is not None else 0.0,
                    "axis_fluent": float(row[10]) if row[10] is not None else 0.0,
                    # §7.F advisory domain hint (free-text; "" when unset).
                    "domain_hint": row[11] if row[11] is not None else "",
                })
        except Exception as e:
            logger.warning(
                "[EngramStore] export_snapshot: concept fetch failed: %s", e,
            )

        concepts.sort(key=lambda r: (r["concept_id"], r["version"]))

        edges_from: list[list[list]] = []
        edges_into: list[list[list]] = []
        for rel, bucket in (("COMPOSED_FROM", edges_from),
                            ("COMPOSED_INTO", edges_into)):
            try:
                qr = self._graph._conn.execute(
                    f"MATCH (a:Engram)-[:{rel}]->(b:Engram) "
                    f"RETURN a.concept_id, a.version, b.concept_id, b.version"
                )
                while qr.has_next():
                    row = qr.get_next()
                    bucket.append([
                        [row[0], int(row[1])],
                        [row[2], int(row[3])],
                    ])
            except Exception as e:
                logger.debug(
                    "[EngramStore] export_snapshot: %s edge fetch failed: %s",
                    rel, e,
                )

        payload = {
            "version": 1,
            "exported_at": self._clock(),
            "concepts": concepts,
            "composition_edges": {
                "from": edges_from,
                "into": edges_into,
            },
        }

        import json
        import os
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = snapshot_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp_path, snapshot_path)
        except Exception as e:
            logger.warning(
                "[EngramStore] export_snapshot atomic write failed: %s", e,
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return 0
        return len(concepts)

    @on_writer
    def read_spine_strands(self, concept_id: str, version: int) -> Optional[dict]:
        """Four spine strands for a (concept_id, version) as Timechain-anchor
        (tx_hash) lists — the read surface CGNMeaningOracle.meaning_of consumes
        (INV-Syn-1 read path; G3, AUDIT §5.3).

        Per ARCHITECTURE_synthesis_engine.md §6 ("Episodic + declarative strands
        of the spine are Timechain-ref'd, not separate Kuzu nodes — Timechain is
        canonical, INV-2") the strands are NOT separate Kuzu tables; they are
        anchor-hash lists derived from the populated spine — the concept-
        version's own anchor_tx (its definitional/declarative TX, §10) plus its
        COMPOSED_FROM parents' anchor_tx bucketed by each parent's memory_type:
          declarative := [self.anchor_tx] + declarative-typed parents
          procedural  := procedural-typed parents (the USES_SKILL strand is
                          schema-only/unpopulated; composed-from procedural
                          bases are the faithful Timechain-anchored proxy)
          episodic    := episodic-typed parents
          felt        := [] (Phase-7 inner-outer-bridge strand; no Kuzu data
                          backs felt yet → felt coverage = 0; INV-Syn-1: never
                          invent a grounding. The bridge extends Concept with
                          felt anchors later, §15.6.)
        meta-typed parents are not one of the four strands (§6) and are skipped.

        version <= 0 → latest (matches ConceptRef default version=0); >= 1 →
        exact. Missing concept → None (CGNMeaningOracle treats None as an empty
        MeaningStrand). Runs on the single writer thread (@on_writer): Kuzu is
        writer-owned, so off-thread reads raise WriterThreadViolation."""
        if version is None or int(version) <= 0:
            row = self._graph.spine_get_latest_concept(concept_id)
        else:
            row = self._graph.spine_get_concept_version(concept_id, int(version))
        if not row:
            return None
        resolved_version = int(row.get("version") or 0)
        own_anchor = row.get("anchor_tx") or ""

        declarative: list[str] = []
        procedural: list[str] = []
        episodic: list[str] = []
        if own_anchor:
            declarative.append(own_anchor)
        try:
            qr = self._graph._conn.execute(
                "MATCH (a:Engram {concept_id: $cid, version: $v})"
                "-[:COMPOSED_FROM]->(b:Engram) "
                "RETURN b.memory_type, b.anchor_tx",
                {"cid": concept_id, "v": resolved_version},
            )
            while qr.has_next():
                mrow = qr.get_next()
                mtype, atx = mrow[0], (mrow[1] or "")
                if not atx:
                    continue
                if mtype == "declarative":
                    declarative.append(atx)
                elif mtype == "procedural":
                    procedural.append(atx)
                elif mtype == "episodic":
                    episodic.append(atx)
                # 'meta' parents are not one of the four spine strands — skip.
        except Exception as e:
            logger.debug(
                "[EngramStore] read_spine_strands: COMPOSED_FROM read failed "
                "for %s:%s: %s", concept_id, resolved_version, e)
        return {
            "declarative_anchors": declarative,
            "procedural_anchors": procedural,
            "episodic_anchors": episodic,
            "felt_anchors": [],
        }

    # ── Internal helpers ─────────────────────────────────────

    @on_writer
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
                    "[EngramStore] composition edge %s v%d ↔ %s v%d skipped"
                    " (likely missing endpoint)",
                    from_concept_id, from_version, parent_id, parent_ver,
                )


__all__ = (
    "EngramStore",
    "Engram",
    "EngramSpine",
    "ParentVersionMissing",
    "WriterFailure",
    "compute_groundedness",
    "_GroundednessParams",
    "_load_groundedness_params_from_toml",
)
