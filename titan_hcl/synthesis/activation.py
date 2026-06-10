"""ACT-R base-level activation B_i — pure math module.

Synthesis Engine Phase 1 (D-SPEC-123, SPEC v1.56.0 §25 / arch §5.2).

Formula:
    B_i = ln( Σ_{j=1..n} (now − t_j)^(−d) )

Petrov 2006 O(1) tail (constant-time approximation for the unbounded
history beyond the last `n` retained access timestamps):

    B_tail ≈ ln( n_total · (now − t_first)^(−d) / (1 − d) )

`d = 0.5` is the ACT-R canonical decay constant; exposed via
`titan_params.toml [synthesis] d` for tuning post-30d-soak. `n` (the
sliding-window retention cap on `access_log`) defaults to 20 per arch
§5.2.

INV-Syn-3: only the synthesis_worker process calls these; the math is
pure (no I/O) so it lives in `titan_hcl/synthesis/` rather than the
worker module itself for clean unit-testability.

INV-Syn-5: reinforcement is use-gated — `record_access(item)` is only
called when the LLM cited or acted upon `item`, never on mere surfacing.
That gate is enforced upstream (the `MEMORY_RETRIEVAL_USED` bus event,
which the synthesis_worker subscribes to, is emitted only by the agno
post-hook on cited items — Phase 1.5+).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Defaults match arch §5.2; runtime values come from titan_params.toml.
DEFAULT_DECAY_D = 0.5
DEFAULT_WINDOW_N = 20

# Numerical floor for (now − t_j) to avoid -inf from a same-tick repeat
# access. Equivalent to "treat any access within the last 1ms as 1ms ago"
# — the integer-millisecond granularity of consumer-side timestamps makes
# this safe.
EPSILON_AGE_S = 1e-3


@dataclass
class ActivationState:
    """In-memory representation of one activation_state row.

    Mirrors the DuckDB schema in titan_hcl/core/direct_memory.py:
        item_id TEXT PRIMARY KEY,
        last_access DOUBLE,
        access_log BLOB,         -- msgpack list[float], last `n`
        access_count INTEGER,    -- TOTAL all-time count (n_total, not len(access_log))
        first_access DOUBLE,
        base_level DOUBLE,
        last_recompute DOUBLE
    """

    item_id: str
    last_access: float = 0.0
    access_log: list[float] = field(default_factory=list)   # last `n` access timestamps, ascending
    access_count: int = 0
    first_access: float = 0.0
    base_level: float = 0.0
    last_recompute: float = 0.0


def record_access(state: ActivationState, ts: float, window_n: int = DEFAULT_WINDOW_N) -> None:
    """Use-gated reinforcement (INV-Syn-5). Mutates `state` in place.

    Maintains `access_log` as the last `window_n` access timestamps in
    ascending order; `access_count` is the all-time total (used for the
    Petrov tail correction).
    """
    state.access_log.append(ts)
    if len(state.access_log) > window_n:
        state.access_log = state.access_log[-window_n:]
    state.last_access = ts
    state.access_count += 1
    # Pin first_access on the very first access only (using access_count
    # after the increment — never use `first_access == 0.0` as a sentinel
    # since 0.0 is a legitimate timestamp value).
    if state.access_count == 1:
        state.first_access = ts


def base_level(
    state: ActivationState,
    now: float,
    d: float = DEFAULT_DECAY_D,
    window_n: int = DEFAULT_WINDOW_N,
) -> float:
    """Compute B_i for `state` at time `now`.

    Sum-of-power-law over the retained `access_log` PLUS the Petrov O(1)
    tail correction for the remaining (access_count − len(access_log))
    older accesses.

    Returns -inf safely: a never-accessed item (empty log, count=0)
    returns -math.inf, signalling "no activation yet, treat as cold-
    start" to ranking consumers (which substitute their cold-start
    default per arch §5.3).
    """
    if state.access_count == 0 or not state.access_log:
        return float("-inf")

    sum_powers = 0.0
    for t_j in state.access_log:
        age = max(now - t_j, EPSILON_AGE_S)
        sum_powers += age ** (-d)

    retained = len(state.access_log)
    tail_count = state.access_count - retained
    if tail_count > 0:
        # `first_access` is pinned on the very first record_access call and
        # never touched afterward — see record_access — so this is reliable
        # even when first_access == 0.0 (a legitimate timestamp).
        age_first = max(now - state.first_access, EPSILON_AGE_S)
        # Petrov 2006 closed-form approximation for the missing tail:
        #   ∑_{j=retained+1..n_total} (now − t_j)^(−d)
        # ≈ tail_count · (now − t_first)^(−d) / (1 − d)
        # Numerically stable for d in (0, 1) — ACT-R d=0.5 is well inside.
        if d < 1.0:
            sum_powers += tail_count * (age_first ** (-d)) / (1.0 - d)
        else:
            # Degenerate range — fall back to a per-item average. Not
            # canonical-ACT-R; only protects against bad config.
            sum_powers += tail_count * (age_first ** (-d))

    if sum_powers <= 0.0:
        return float("-inf")
    return math.log(sum_powers)


def recompute_all(
    states: list[ActivationState],
    now: float,
    d: float = DEFAULT_DECAY_D,
    window_n: int = DEFAULT_WINDOW_N,
) -> int:
    """Recompute B_i for every state. Returns the count actually touched.

    Called by the synthesis_worker's 60s recompute loop. Mutates each
    `state.base_level` + `state.last_recompute` in place; the caller is
    responsible for persisting to DuckDB.
    """
    n_touched = 0
    for state in states:
        new_bi = base_level(state, now, d=d, window_n=window_n)
        if new_bi != state.base_level or state.last_recompute == 0.0:
            state.base_level = new_bi
            state.last_recompute = now
            n_touched += 1
    return n_touched
