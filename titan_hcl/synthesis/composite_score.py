"""ACT-R composite retrieval score (arch §5.3).

Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0 §25).

Formula:
    retrieval_score(i, q) =
        w_b · norm(B_i)
      + w_s · spreading(i, buffers)
      + w_r · cosine(embed(q), embed(i))
      + w_p · importance(i)

Phase 1 scope: w_b + w_r + w_p only. Spreading activation depends on the
Kuzu Concept spine (arch §6.1) which lands Phase 4 — Phase 1 silently
ignores `w_s` (returns 0 from spreading_lookup). actr_buffers + the
`buffer_entities` substrate for spreading also ship Phase 7.

- `norm(B_i)` — z-scored over the candidate pool (NOT z-scored over the
  whole `activation_state` table — per arch §5.3 normalization is local
  to the result set so cold items don't drag the mean).
- `cosine` — FAISS L2-norm inner product (caller supplies this).
- `importance(i)` — Phase 1: per arch §5.3 + arch §15.6 "consumes the
  bridge salience formula." Bridge salience is Phase 4+; for now we
  default to a per-node `effective_weight` field if present, else 0.5.
  This keeps the weight present + tunable without front-running the
  bridge contract.
- Cold-start: items absent from `activation_state` get `B_i = -inf` from
  base_level(); norm() substitutes the configured cold-start default
  (0.5 per arch §5.3) so cold items still rank by cosine + importance.

Default weights all = 1.0 per arch §5.3; tunable post-soak via
`titan_params.toml [synthesis]`.

INV-Syn-3: this is a PURE function — no I/O, no DuckDB, no SHM. The
caller (synthesis-aware retrieval wrapper) supplies the activation
lookup (BridgeRecall) + the cosine scores. Stays unit-testable without
a worker process or DuckDB connection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


# Defaults match arch §5.3; runtime values come from titan_params.toml.
DEFAULT_W_B = 1.0      # base-level activation weight
DEFAULT_W_S = 1.0      # spreading activation weight (Phase 4+)
DEFAULT_W_R = 1.0      # cosine relevance weight
DEFAULT_W_P = 1.0      # importance weight
DEFAULT_COLD_START_B = 0.5     # arch §5.3 — cold items default to 0.5

# Floor for `(now - t_j)^(-d)` etc. inside base_level; mirrored here for
# the cold-start substitution boundary (any base_level ≤ this is treated
# as cold-start).
COLD_START_SENTINEL = float("-inf")


@dataclass
class Candidate:
    """One retrieval candidate fed into composite_score.

    `item_id` namespaces match arch §5.2 + synthesis_worker.activation_state
    keys: "kuzu:NODE" | "tc:TX" | "skill:ID" | "fork:ID" | "mem:<memory_nodes.id>".

    `cosine` is the FAISS-returned similarity in [0, 1]; values outside
    this range are clamped to [0, 1] at scoring time (defensive — some
    encoders return raw inner products).

    `importance` is per arch §5.3 / arch §15.6 — Phase 1 default 0.5 if
    the source has no explicit value; Phase 4+ consumes bridge salience.
    """

    item_id: str
    cosine: float
    importance: float = 0.5
    payload: object = None     # opaque pointer (the original node dict / TX hash)


@dataclass
class ScoredCandidate:
    """A candidate annotated with the full composite breakdown for
    observability + debugging. Sort by `.score` descending."""

    candidate: Candidate
    score: float
    base_level: float              # raw B_i from activation lookup (may be -inf)
    norm_base_level: float         # z-scored over the candidate pool
    cosine: float                  # clamped [0,1]
    importance: float
    weights: tuple[float, float, float, float]    # (w_b, w_s, w_r, w_p)


def _zscore(values: list[float], cold_default: float) -> list[float]:
    """Z-score normalize a list of values. Cold-start sentinels (-inf)
    are SUBSTITUTED with `cold_default` BEFORE normalization so they
    don't poison the mean/std.

    Empty / all-cold input → all-zero output (no signal to normalize).
    Single-element input → 0.0 (z-score of a single point is undefined).
    """
    if not values:
        return []
    substituted = [
        cold_default if v == COLD_START_SENTINEL or not math.isfinite(v) else v
        for v in values
    ]
    n = len(substituted)
    if n == 1:
        return [0.0]
    mean = sum(substituted) / n
    var = sum((x - mean) ** 2 for x in substituted) / n
    std = math.sqrt(var)
    if std == 0.0:
        # All values identical — no signal in the activation axis.
        return [0.0] * n
    return [(x - mean) / std for x in substituted]


def composite_score(
    candidates: Iterable[Candidate],
    activation_lookup: Callable[[list[str]], dict[str, float]],
    spreading_lookup: Optional[Callable[[list[str]], dict[str, float]]] = None,
    *,
    w_b: float = DEFAULT_W_B,
    w_s: float = DEFAULT_W_S,
    w_r: float = DEFAULT_W_R,
    w_p: float = DEFAULT_W_P,
    cold_start_b: float = DEFAULT_COLD_START_B,
) -> list[ScoredCandidate]:
    """Score + sort candidates by composite retrieval score (arch §5.3).

    `activation_lookup(item_ids) -> {item_id: base_level}` — Phase 1
    callers pass BridgeRecall.activation_lookup. Cold items (absent from
    activation_state) should be omitted from the returned dict — we
    substitute `cold_start_b` per arch §5.3.

    `spreading_lookup` — Phase 4+ Kuzu Concept-spine spreading. Phase 1
    callers pass None → spreading contribution is 0.

    Returns ScoredCandidate list sorted by `.score` descending. Ties broken
    by candidate order (Python sort is stable) → preserves cosine ordering
    for items with identical composite scores.
    """
    cands = list(candidates)
    if not cands:
        return []

    ids = [c.item_id for c in cands]
    activations = activation_lookup(ids) or {}
    base_levels = [activations.get(c.item_id, COLD_START_SENTINEL) for c in cands]
    norm_bs = _zscore(base_levels, cold_default=cold_start_b)

    if spreading_lookup is not None:
        spreading = spreading_lookup(ids) or {}
        spreads = [spreading.get(c.item_id, 0.0) for c in cands]
    else:
        spreads = [0.0] * len(cands)

    scored: list[ScoredCandidate] = []
    for c, b, nb, sp in zip(cands, base_levels, norm_bs, spreads):
        cos = max(0.0, min(1.0, c.cosine))     # defensive clamp
        imp = c.importance
        score = w_b * nb + w_s * sp + w_r * cos + w_p * imp
        scored.append(ScoredCandidate(
            candidate=c, score=score,
            base_level=b, norm_base_level=nb,
            cosine=cos, importance=imp,
            weights=(w_b, w_s, w_r, w_p),
        ))

    scored.sort(key=lambda sc: sc.score, reverse=True)
    return scored


__all__ = [
    "Candidate", "ScoredCandidate", "composite_score",
    "DEFAULT_W_B", "DEFAULT_W_S", "DEFAULT_W_R", "DEFAULT_W_P",
    "DEFAULT_COLD_START_B", "COLD_START_SENTINEL",
    "make_kuzu_spreading_lookup",
]


# ─── Phase 4 — Kuzu Concept-spine spreading-activation lookup (§P4.F) ────


def _candidate_id_to_concept_id(item_id: str) -> str | None:
    """Extract the spine concept_id from a candidate item_id, or None if
    the item isn't concept-keyed. P4 spreading contributes ONLY to
    concept-keyed candidates because that's where the spine carries
    edges. Other item_id namespaces (`mem:`, `tc:`, `skill:`, `fork:`)
    get spreading = 0 in P4; later phases may extend the mapping as
    consolidation materializes more concept-spine entries."""
    if not item_id:
        return None
    if item_id.startswith("concept:"):
        rest = item_id[len("concept:"):]
        return rest.split(":v", 1)[0] or None
    return None


def make_kuzu_spreading_lookup(
    kuzu_reader,
    buffer_entities: list[str],
    *,
    S: float = 2.0,
    neighbor_fetch_cap: int = 50,
):
    """Build a spreading_lookup function bound to the current buffer state.

    Per arch §5.3:
        spreading(i, buffers) =
            Σ_{j ∈ buffer_entities} (S - ln(fan_j)) · 1[edge(j, i) ∈ Kuzu]

    where `fan_j` is the COMPOSED_FROM + COMPOSED_INTO neighbor count of
    concept `j` in the Kuzu spine. P4 computes contributions ONLY for
    candidate item_ids that are concept-keyed (`concept:<id>` or
    `concept:<id>:v<n>`); other namespaces contribute 0 (consistent with
    "spine carries the structure" — non-concept items don't have spine
    neighbors yet).

    Pre-computes the reachable-concept_id map ONCE per recall (buffer
    state is fixed per recall) so the per-candidate lookup is O(1).

    `kuzu_reader` is any object with `spine_concept_neighbors(concept_id,
    limit=...)` returning `list[(concept_id, version)]` — typically a
    `TitanKnowledgeGraph` or a BridgeRecall-wrapped read-only handle.

    Returns a callable `lookup(item_ids) -> {item_id: spreading_score}`
    that fits composite_score's `spreading_lookup` parameter.
    """
    import math

    reachable: dict[str, float] = {}
    for buf in buffer_entities or []:
        if not buf:
            continue
        try:
            neighbors = kuzu_reader.spine_concept_neighbors(
                buf, limit=neighbor_fetch_cap,
            )
        except Exception:
            # Soft-fail per BridgeRecall contract (INV-Syn-4 / G18).
            continue
        if not neighbors:
            continue
        fan_j = len(neighbors)
        contribution = S - math.log(fan_j)
        if contribution <= 0:
            # Very dense node (ln(fan_j) >= S) → no contribution. Matches
            # arch §5.3 "saturation" intuition: ubiquitous concepts can't
            # discriminate, so they shouldn't pull the score.
            continue
        for nb_id, _nb_version in neighbors:
            reachable[nb_id] = reachable.get(nb_id, 0.0) + contribution

    def _lookup(item_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for iid in item_ids:
            cid = _candidate_id_to_concept_id(iid)
            if cid is None:
                continue
            score = reachable.get(cid)
            if score is not None:
                out[iid] = score
        return out

    return _lookup
