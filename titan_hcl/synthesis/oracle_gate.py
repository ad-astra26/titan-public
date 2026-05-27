"""Phase 6 — Metabolic gate for metered TruthOraclePlug invocations (INV-Syn-13).

The gate is **pure functional**. It does not own state; the per-day spend
ledger lives in `data/synthesis.duckdb :: oracle_daily_spend` (DDL helper
here, but the sole writer is `synthesis_worker` via `OracleRouter` per
INV-Syn-3 / G21 — see P6.F).

The contract (per SPEC §25.1 INV-Syn-13 + `titan_params.toml [synthesis.oracle]`):

    free oracles (cost_class="free"; e.g. coding_sandbox, pure math,
                  sovereign-light-node [TARGET]):
        always admit; no spend bookkeeping.

    metered oracles (cost_class="metered"; e.g. helius_rpc, web_api,
                     x_api, zk_prover):
        admit_score = importance × (balance_sol / balance_sol_baseline)
        admit IFF
            admit_score    ≥ params.admit_threshold
            AND
            daily_remaining[oracle_id] ≥ claim.cost_estimate_sol

A gate-denied claim STILL produces an anchored verdict (per INV-Syn-13):
the OracleRouter constructs `OracleVerdict(verdict="unknown",
evidence_ref="metabolic_gate_denied"|"daily_budget_exhausted", cost=0.0,
latency_ms=<gate-check ms>)` — the rejection itself is auditable on-chain.

Used by `OracleRouter` (P6.F). Unit-test surface here is a pure call:
`OracleGate.admit(claim, oracle, balance_sol, remaining_daily_sol) -> Decision`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from titan_hcl.synthesis.plugs import OracleClaim, TruthOraclePlug


# ─────────────────────────────────────────────────────────────────────────
# Defaults (mirrored from titan_params.toml [synthesis.oracle]; the runtime
# gate reads from the merged config, these constants exist so unit tests
# can construct a gate without a config file present).
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_BALANCE_SOL_BASELINE: float = 1.0
DEFAULT_ADMIT_THRESHOLD: float = 0.15
DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL: float = 0.1

# Reason strings used in evidence_ref of gate-denied verdicts. Keep stable —
# downstream observability + analytics group by these literals.
DENY_REASON_THRESHOLD = "metabolic_gate_denied"
DENY_REASON_BUDGET = "daily_budget_exhausted"


# ─────────────────────────────────────────────────────────────────────────
# Decision record
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class GateDecision:
    """The gate's verdict on a single OracleClaim."""

    admit: bool
    reason: Literal["", "metabolic_gate_denied", "daily_budget_exhausted"]
    admit_score: float          # importance × (balance/baseline); always reported, even for free oracles (1.0)
    latency_ms: int             # wall-clock spent inside the gate (so anchored unknown verdicts carry real latency)


# ─────────────────────────────────────────────────────────────────────────
# Gate config snapshot
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OracleGateConfig:
    """Frozen snapshot of [synthesis.oracle] params.

    Built once by the synthesis_worker at boot from the merged Titan
    config; passed to OracleGate. Per-oracle daily budgets are a
    `{oracle_id: sol_per_day}` dict — missing keys fall back to
    `default_daily_sol_budget`.
    """

    balance_sol_baseline: float = DEFAULT_BALANCE_SOL_BASELINE
    admit_threshold: float = DEFAULT_ADMIT_THRESHOLD
    default_daily_sol_budget: float = DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL
    daily_sol_budget: dict[str, float] = None  # type: ignore[assignment]  # frozen + mutable default → built via factory

    def daily_budget_for(self, oracle_id: str) -> float:
        """Look up the per-oracle daily SOL cap; fall back to default."""
        if self.daily_sol_budget is None:
            return self.default_daily_sol_budget
        return float(self.daily_sol_budget.get(oracle_id, self.default_daily_sol_budget))


def build_gate_config(merged_config: dict | None) -> OracleGateConfig:
    """Construct an OracleGateConfig from the merged Titan config dict.

    `merged_config` is the dict returned by `load_titan_config()` /
    `load_titan_params()` (the 4-layer merge — titan_params.toml <
    config.toml < secrets.toml < ~/.titan/microkernel_<TID>.toml). The
    relevant block is `[synthesis.oracle]` with subtables
    `[synthesis.oracle.daily_sol_budget]` and `[synthesis.oracle.zk]`.

    Missing keys fall back to module defaults so this never raises — the
    gate degrades to "everything admits at default threshold" if config
    is corrupt, NOT to crashing the synthesis_worker.
    """
    block: dict = ((merged_config or {}).get("synthesis", {}) or {}).get("oracle", {}) or {}
    daily: dict = block.get("daily_sol_budget", {}) or {}
    # Filter to numeric values so a typo'd string entry doesn't crash float() later.
    daily_clean = {
        str(k): float(v)
        for k, v in daily.items()
        if isinstance(v, (int, float))
    }
    return OracleGateConfig(
        balance_sol_baseline=float(block.get("balance_sol_baseline", DEFAULT_BALANCE_SOL_BASELINE)),
        admit_threshold=float(block.get("admit_threshold", DEFAULT_ADMIT_THRESHOLD)),
        default_daily_sol_budget=DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL,
        daily_sol_budget=daily_clean,
    )


def zk_privacy_domains(merged_config: dict | None) -> frozenset[str]:
    """Return the INV-Syn-14 privacy-domain whitelist from `[synthesis.oracle.zk]`.

    Defined here (not in proofs/) because the whitelist is a Phase-6
    config artifact alongside the gate; the proof strategy registry
    (P6.G) consumes it. Missing config → empty set (no auto-promotion;
    Merkle stays default everywhere; the per-fork explicit `zk` flag
    remains the other ZK trigger).
    """
    block: dict = ((merged_config or {}).get("synthesis", {}) or {}).get("oracle", {}) or {}
    zk_block: dict = block.get("zk", {}) or {}
    domains = zk_block.get("privacy_domains", []) or []
    return frozenset(str(d) for d in domains)


# ─────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────

class OracleGate:
    """Pure functional INV-Syn-13 gate.

    Construction is cheap; the gate carries only the config snapshot.
    Per-day spend tracking lives in the OracleRouter (P6.F) which owns
    `oracle_daily_spend` in `data/synthesis.duckdb` — the router queries
    "how much have we spent on `helius_rpc` today?" and passes
    `remaining_daily_sol = budget - spent_today` into `admit()`.
    """

    def __init__(self, config: OracleGateConfig):
        self._config = config

    @property
    def config(self) -> OracleGateConfig:
        return self._config

    def admit(
        self,
        claim: "OracleClaim",
        oracle: "TruthOraclePlug",
        *,
        balance_sol: float,
        remaining_daily_sol: float,
    ) -> GateDecision:
        """Apply INV-Syn-13 to a single (claim, oracle) pair.

        `balance_sol` — current Titan SOL balance (from `core/network.py`
        or `core/soul.py`; caller's responsibility to provide a fresh
        value; the gate does NOT call out for it).

        `remaining_daily_sol` — `daily_sol_budget[oracle.oracle_id] -
        spent_today`. Caller (OracleRouter) tracks `spent_today` in
        `oracle_daily_spend` and computes this before the gate call. For
        free oracles this argument is irrelevant — the cost_class branch
        short-circuits before reading it.

        Returns a GateDecision; the caller anchors an OracleVerdict
        accordingly (admitted → invoke plug.verify(); denied → emit
        verdict_unknown with evidence_ref=decision.reason).
        """
        t0 = time.perf_counter()

        # Free oracles bypass the gate entirely (INV-Syn-13).
        if getattr(oracle, "cost_class", "metered") == "free":
            return GateDecision(
                admit=True,
                reason="",
                admit_score=1.0,
                latency_ms=int((time.perf_counter() - t0) * 1000.0),
            )

        importance = float(getattr(claim, "importance", 0.5))
        baseline = self._config.balance_sol_baseline
        # Guard against pathological baseline=0 (Maker typo): treat as 1.0
        # so the gate never divides by zero and never spuriously admits
        # everything because of a config bug.
        if baseline <= 0.0:
            baseline = DEFAULT_BALANCE_SOL_BASELINE
        admit_score = importance * (float(balance_sol) / baseline)

        # First gate — admit threshold (INV-Syn-13 first conjunct).
        if admit_score < self._config.admit_threshold:
            return GateDecision(
                admit=False,
                reason=DENY_REASON_THRESHOLD,
                admit_score=admit_score,
                latency_ms=int((time.perf_counter() - t0) * 1000.0),
            )

        # Second gate — daily budget (INV-Syn-13 second conjunct).
        # The claim's payload may carry an explicit `cost_estimate_sol`
        # override; otherwise treat as 0.0 (some oracles report cost only
        # post-call). A claim with no cost estimate passes this gate IFF
        # remaining_daily_sol > 0 — i.e. the per-oracle daily budget has
        # not been exhausted yet.
        payload = getattr(claim, "payload", {}) or {}
        cost_estimate = float(payload.get("cost_estimate_sol", 0.0))
        if remaining_daily_sol < cost_estimate or remaining_daily_sol <= 0.0:
            return GateDecision(
                admit=False,
                reason=DENY_REASON_BUDGET,
                admit_score=admit_score,
                latency_ms=int((time.perf_counter() - t0) * 1000.0),
            )

        return GateDecision(
            admit=True,
            reason="",
            admit_score=admit_score,
            latency_ms=int((time.perf_counter() - t0) * 1000.0),
        )


# ─────────────────────────────────────────────────────────────────────────
# DuckDB DDL helper for `oracle_daily_spend` (the spend ledger)
# ─────────────────────────────────────────────────────────────────────────

# The table lives on `data/synthesis.duckdb` — the same DB the
# synthesis_worker already owns per INV-Syn-3. Sole writer is the
# synthesis_worker via OracleRouter (P6.F). Cross-process readers go
# through atomic JSON snapshot — see P6.K for the Observatory route.
ORACLE_DAILY_SPEND_DDL = """
CREATE TABLE IF NOT EXISTS oracle_daily_spend (
    oracle_id   VARCHAR    NOT NULL,
    spend_date  DATE       NOT NULL,
    spent_sol   DOUBLE     NOT NULL DEFAULT 0.0,
    n_calls     INTEGER    NOT NULL DEFAULT 0,
    updated_at  TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (oracle_id, spend_date)
);
"""


def ensure_oracle_daily_spend_table(conn) -> None:
    """Create `oracle_daily_spend` if missing.

    Idempotent (CREATE TABLE IF NOT EXISTS). Safe to call at every
    synthesis_worker boot. `conn` is a DuckDB connection — callers
    pass the synthesis_worker's existing `synthesis.duckdb` handle so
    INV-Syn-3 sole-writer is preserved.
    """
    conn.execute(ORACLE_DAILY_SPEND_DDL)


__all__ = (
    "OracleGate",
    "OracleGateConfig",
    "GateDecision",
    "build_gate_config",
    "zk_privacy_domains",
    "ensure_oracle_daily_spend_table",
    "ORACLE_DAILY_SPEND_DDL",
    "DENY_REASON_THRESHOLD",
    "DENY_REASON_BUDGET",
    "DEFAULT_BALANCE_SOL_BASELINE",
    "DEFAULT_ADMIT_THRESHOLD",
    "DEFAULT_PER_ORACLE_DAILY_BUDGET_SOL",
)
