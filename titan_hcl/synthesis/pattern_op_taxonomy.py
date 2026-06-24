"""pattern_op_taxonomy — the abstract-operation registry for `pattern_logic`.

RFP_pattern_logic.md §6/Q1 (LOCKED 2026-06-24). A verified-transition records
*"in CONTEXT, OPERATION led to OUTCOME (verified true/false)"*. The `operation`
is the **procedural-action verb** that produced the outcome — the axis along which
an inner CGN HAOV-action and an outer OracleVerdict-action are recognised as "the
same kind of operation". This is DISTINCT from BRAIN's *relational* Link starter
set (`ASSOCIATE/RELATE/OPPOSITE/EQUIV/COMPARE/COMPLEMENT`, which describe
concept↔concept relations — only COMPARE overlaps). Per BRAIN-INV-10 the Link
registry is extensible with *new bounded-op mini-programs*: these procedural verbs
are registered as such, coexisting with the relational starter set — forward
compatible, NOT force-mapped onto the relational names.

Proto stance (RFP §3 DEFER): the op is a symbolic label now; it *graduates* to a
TitanVM Link mini-program when BRAIN lands. No executor is built here.

The seed is **not a fixed list** (INV: emergence over determinism) — `register_op`
adds new verbs as new substrate sources come online (e.g. ARC in Phase 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "OpSpec",
    "SEED_OPS",
    "op_for_outer_action",
    "register_op",
    "is_known_op",
    "all_ops",
    "describe_op",
]


class OpSpec:
    """One abstract operation = a procedural-action verb (proto-BRAIN Link program)."""

    __slots__ = ("name", "description", "phase")

    def __init__(self, name: str, description: str, phase: int) -> None:
        self.name = name
        self.description = description
        # Phase the op's source stream comes online (1 = live outer/inner now;
        # 2 = ARC / widened streams). Pure metadata — does not gate registration.
        self.phase = phase

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"OpSpec({self.name!r}, phase={self.phase})"


# ── The 9-verb seed registry (Q1) ───────────────────────────────────────────
# Phase-1 exercised: RECALL/RESEARCH/TOOL/SKILL/COMPARE. The rest are seeded for
# Phase-2 sources (ARC grid: TRANSFORM/COMPOSE; deterministic: COMPUTE; VERIFY).
_SEED: List[OpSpec] = [
    OpSpec("RECALL", "answer/derive from existing grounded memory (OML 'direct')", 1),
    OpSpec("RESEARCH", "fetch fresh external knowledge (OML 'research'/web_search)", 1),
    OpSpec("TOOL", "run a deterministic tool/oracle (OML 'tool'/coding_sandbox)", 1),
    OpSpec("SKILL", "apply a verified learned skill (OML 'skill_delegate')", 1),
    OpSpec("COMPARE", "find the delta/difference between two quantities (cross-domain)", 1),
    OpSpec("COMPUTE", "deterministic calculation over known inputs", 2),
    OpSpec("COMPOSE", "chain/combine sub-operations into a sequence", 2),
    OpSpec("TRANSFORM", "map/restructure a representation (e.g. ARC grid ops)", 2),
    OpSpec("VERIFY", "check/validate a claim against ground truth", 2),
]

SEED_OPS: Dict[str, OpSpec] = {o.name: o for o in _SEED}

# Live registry — starts as the seed, grows via register_op (BRAIN-INV-10).
_REGISTRY: Dict[str, OpSpec] = dict(SEED_OPS)

# ── OML routing action → abstract op (the outer-substrate projection) ────────
# OUTER_ACTIONS = (direct, tool, skill_delegate, research, IDK). 'IDK' is an
# abstention, not an operation that produced an outcome → it has no op (None).
_OUTER_ACTION_TO_OP: Dict[str, str] = {
    "direct": "RECALL",
    "tool": "TOOL",
    "skill_delegate": "SKILL",
    "research": "RESEARCH",
    # "IDK": intentionally absent — no verified-transition op for an abstention.
}


def op_for_outer_action(action: Optional[str]) -> Optional[str]:
    """Map an OML routing action (outer_meta_policy.OUTER_ACTIONS) to an abstract op.

    Returns None for abstentions ('IDK'), unknown actions, or None — the caller
    skips those (they are not operations that produced a verifiable outcome).
    """
    if not action:
        return None
    return _OUTER_ACTION_TO_OP.get(action)


def register_op(name: str, description: str, *, phase: int = 2) -> OpSpec:
    """Register a new abstract op (BRAIN-INV-10 extensibility). Idempotent on name.

    Used when a new substrate source introduces an operation not in the seed
    (e.g. a future domain). Re-registering an existing name returns the existing
    spec unchanged (mutate-not-update discipline: never silently overwrite).
    """
    name = name.strip().upper()
    if not name:
        raise ValueError("op name must be non-empty")
    existing = _REGISTRY.get(name)
    if existing is not None:
        return existing
    spec = OpSpec(name, description, phase)
    _REGISTRY[name] = spec
    return spec


def is_known_op(name: Optional[str]) -> bool:
    return bool(name) and name.strip().upper() in _REGISTRY


def all_ops() -> List[str]:
    """All currently-registered op names (seed + emergently registered)."""
    return list(_REGISTRY.keys())


def describe_op(name: str) -> Optional[str]:
    spec = _REGISTRY.get(name.strip().upper()) if name else None
    return spec.description if spec else None
