"""EngineRecall — contract-driven episodic recall.

Synthesis Engine Phase 2 (PLAN_synthesis_engine_Phase2.md §2D, D-P2-1).

The recall pipeline is encoded END-TO-END inside the
`actr_episodic_recall_helper` contract (PLAN §2C.a). EngineRecall is the
thin coordinator that:

  1. embeds the query text (caller-supplied embedder; SPEC-correct per
     arch §3.5 — RuleEvaluator stays model-free, D-P2-3),
  2. loads the contract,
  3. invokes `RuleEvaluator.evaluate(rules, ctx, initial_variables=...)`
     with `$query_embedding` + `$current_chat_tx` pre-seeded,
  4. consumes the `rank_composite` action by:
       - merging the candidate-source $vars (e.g. $base + $semantic +
         $threaded) — dedup by tx_hash,
       - building `Candidate` objects (cosine from FAISS score if
         present, else default; importance default 0.5 per arch §5.3),
       - running `composite_score()` with the contract-supplied weights,
  5. returns the top-K ScoredCandidate-equivalent dicts.

Contract-driven design (D-P2-1): Maker tunes ranking end-to-end by
editing the JSON + re-signing — no Python change needed. Adding new
candidate sources (a fourth fork to search, a new cross-ref via_field)
is purely a JSON edit.

Source-of-truth for contracts: this class reads them DIRECTLY from
`titan_hcl/contracts/meta_cognitive/*.json` (not from the meta-fork
chain). Synthesis_worker is a CONSUMER of contracts, not a writer; the
JSONs on disk are the runtime source. The chain stores them for audit
+ Maker ceremony (R8 bundle), but the engine doesn't depend on the
chain to function — keeps synthesis_worker free of any sync `bus.request`
to timechain_worker (INV-Syn-2 / G19).

Fallback policy (PLAN §2D): when the helper contract is absent /
disabled / fails to fire, EngineRecall returns `None` — the caller
should degrade to cosine-only ranking. NEVER raise to the caller.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from titan_hcl.synthesis.composite_score import (
    Candidate,
    ScoredCandidate,
    composite_score,
    DEFAULT_W_B,
    DEFAULT_W_S,
    DEFAULT_W_R,
    DEFAULT_W_P,
)

logger = logging.getLogger(__name__)


# Contract identifier — must match the JSON's contract_id field.
HELPER_CONTRACT_ID = "actr_episodic_recall_helper"
# Default candidate-source $var prefix used by the helper contract. The
# contract action's `candidates_from` list overrides this — kept here
# only so a test fixture can reference the canonical names.
DEFAULT_CANDIDATES_FROM = ("$base", "$semantic", "$threaded")
DEFAULT_K = 8


# ── Recall result row (engine API) ───────────────────────────────────────

@dataclass
class RecallResult:
    """One ranked retrieval result. Stable surface for consumers (the
    agno tool layer + the future bridge consumer). Maps onto
    ScoredCandidate from composite_score but flattens the dataclass
    nesting for an at-a-glance JSON-shape."""

    tx_hash: str
    score: float
    fork: str = ""
    source: str = ""
    summary: str = ""
    # Internal scoring breakdown (observability — Observatory panel later).
    cosine: float = 0.0
    base_level: float = 0.0
    norm_base_level: float = 0.0
    importance: float = 0.0


# ── Engine recall ────────────────────────────────────────────────────────

class EngineRecall:
    """Contract-driven episodic recall coordinator.

    Construct once per process. Hot path: `.recall(query_text, ...)`.
    """

    def __init__(
        self,
        rule_evaluator,
        activation_lookup: Callable[[list[str]], dict[str, float]],
        embedder: Optional[Callable[[str], list[float]]] = None,
        contracts_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            rule_evaluator: A RuleEvaluator pre-wired with faiss_reader +
                index_db substrate handles (per PLAN §2A). The evaluator's
                chi accounting flows through here unchanged.
            activation_lookup: Callable `(item_ids) -> {item_id: base_level}`.
                Synthesis_worker binds this to ActivationStore (in-process,
                no DuckDB lock conflict). Cross-process consumers would
                pass BridgeRecall.activation_lookup.
            embedder: Callable `(text) -> list[float]` — caller-supplied
                so RuleEvaluator stays model-free (D-P2-3). None disables
                recall entirely (returns None — caller falls back).
            contracts_dir: Override the default
                `titan_hcl/contracts/meta_cognitive/`. Tests use this to
                point at a tmp directory with a fixture contract.
        """
        self._evaluator = rule_evaluator
        self._activation_lookup = activation_lookup
        self._embedder = embedder
        if contracts_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            # titan_hcl/synthesis → titan_hcl/contracts/meta_cognitive
            pkg = os.path.dirname(here)
            contracts_dir = os.path.join(pkg, "contracts", "meta_cognitive")
        self._contracts_dir = contracts_dir
        # In-memory contract cache (id -> parsed dict). Lazy-loaded on
        # first recall(); reloaded on mtime change (cheap stat).
        self._contracts: dict[str, dict] = {}
        self._contracts_mtimes: dict[str, float] = {}
        # Aggregate stats for Observatory.
        self._total_recall_calls = 0
        self._total_contract_hits = 0
        self._total_fallbacks = 0

    # ── contract loading (lazy + mtime-cached) ────────────────────────

    def _load_contract(self, contract_id: str) -> Optional[dict]:
        """Read `<contract_id>.json` from contracts_dir, mtime-cached.
        Returns the parsed JSON dict or None if missing / unparseable /
        inactive. Soft-fail; never raises."""
        fname = f"{contract_id}.json"
        path = os.path.join(self._contracts_dir, fname)
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._contracts.get(contract_id)
        if cached is not None and self._contracts_mtimes.get(contract_id) == mtime:
            return cached
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                "[EngineRecall] failed to parse %s: %s", fname, exc)
            return None
        if data.get("status") != "active":
            return None
        self._contracts[contract_id] = data
        self._contracts_mtimes[contract_id] = mtime
        return data

    # ── public hot path ───────────────────────────────────────────────

    def recall(
        self,
        query_text: str,
        *,
        current_chat_tx: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> Optional[list[RecallResult]]:
        """Contract-driven episodic recall.

        Returns:
            list[RecallResult] sorted by composite score descending, or
            None when the helper contract is absent / disabled / fails
            to fire — caller falls back to cosine-only ranking.
        """
        self._total_recall_calls += 1

        if not self._embedder:
            logger.debug("[EngineRecall] no embedder injected — fallback")
            self._total_fallbacks += 1
            return None

        contract = self._load_contract(HELPER_CONTRACT_ID)
        if contract is None:
            self._total_fallbacks += 1
            return None

        # 1. Embed the query.
        try:
            qe = list(self._embedder(query_text))
        except Exception as exc:
            logger.warning(
                "[EngineRecall] embedder failed: %s — fallback", exc)
            self._total_fallbacks += 1
            return None
        if not qe:
            self._total_fallbacks += 1
            return None

        # 2. Build the eval context + variable preamble.
        ctx = {
            "event": "retrieval_request",
            "query_text_len": len(query_text or ""),
            "k": k,
        }
        initial_vars: dict[str, Any] = {
            "$query_embedding": qe,
            "$current_chat_tx": current_chat_tx or "",
        }

        # 3. Evaluate via RuleEvaluator. The contract's binding rules
        #    (FORK_READ / SEARCH / CROSS_REF) populate variables; the
        #    final OR-gate emits the rank_composite action.
        try:
            action = self._evaluator.evaluate(
                contract["rules"], ctx, initial_variables=initial_vars)
        except Exception as exc:
            logger.warning(
                "[EngineRecall] contract evaluation failed: %s — fallback",
                exc, exc_info=True)
            self._total_fallbacks += 1
            return None

        if not action:
            # OR-gate fell through — all sources empty. Caller falls back.
            self._total_fallbacks += 1
            return None
        if action.get("action") == "chi_budget_exhausted":
            logger.warning(
                "[EngineRecall] chi budget exhausted during recall "
                "(spent=%s cap=%s) — fallback",
                action.get("spent"), action.get("cap"))
            self._total_fallbacks += 1
            return None
        if action.get("action") != "rank_composite":
            logger.debug(
                "[EngineRecall] unexpected action %r from %s — fallback",
                action.get("action"), HELPER_CONTRACT_ID)
            self._total_fallbacks += 1
            return None

        self._total_contract_hits += 1

        # 4. Consume rank_composite — merge candidate $vars + dedup +
        #    score + return top-K.
        return self._consume_rank_composite(action, k)

    # ── action consumption ────────────────────────────────────────────

    def _consume_rank_composite(
        self, action: dict, k_override: int,
    ) -> list[RecallResult]:
        """Apply the rank_composite action: merge candidate-source $vars,
        dedup by tx_hash, score via composite_score(), return top-K."""
        # Pull the contract-supplied parameters. `action_limit` is the
        # contract's `limit` field; we honor min(action_limit, k_override)
        # so the caller can request fewer than the contract allows.
        candidate_sources = action.get(
            "candidates_from", list(DEFAULT_CANDIDATES_FROM))
        weights = action.get("weights") or {}
        action_limit = int(action.get("limit", DEFAULT_K))
        k_final = max(1, min(action_limit, int(k_override)))

        w_b = float(weights.get("w_b", DEFAULT_W_B))
        w_s = float(weights.get("w_s", DEFAULT_W_S))
        w_r = float(weights.get("w_r", DEFAULT_W_R))
        w_p = float(weights.get("w_p", DEFAULT_W_P))

        # Pull merged candidate dicts from RuleEvaluator's bound variables.
        # We access the evaluator's internal `variables` dict from the
        # last evaluate() call. Cleanest hack: stash variables on the
        # evaluator after each call, then reach in here. The internal
        # contract is local + tested below — see RuleEvaluator._last_vars.
        last_vars = getattr(self._evaluator, "_last_variables", {}) or {}

        seen: set[str] = set()
        candidates: list[Candidate] = []
        for src_var in candidate_sources:
            rows = last_vars.get(src_var) or []
            if not isinstance(rows, (list, tuple)):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                tx_hash = (row.get("tx_hash")
                           or row.get("block_hash")
                           or row.get("id")
                           or "")
                if not tx_hash or tx_hash in seen:
                    continue
                seen.add(tx_hash)
                # cosine comes from SEARCH's `score` field (FAISS similarity).
                # FORK_READ / CROSS_REF rows have no cosine — default to
                # 0.5 so they rank by activation + importance.
                cosine = row.get("score")
                if cosine is None:
                    cosine = 0.5
                importance = float(row.get("importance", 0.5))
                candidates.append(Candidate(
                    item_id=f"tc:{tx_hash}",
                    cosine=float(cosine),
                    importance=importance,
                    payload=row,
                ))

        if not candidates:
            return []

        scored = composite_score(
            candidates,
            activation_lookup=self._activation_lookup,
            spreading_lookup=None,    # Phase 4+ Kuzu spine
            w_b=w_b, w_s=w_s, w_r=w_r, w_p=w_p,
        )

        results: list[RecallResult] = []
        for sc in scored[:k_final]:
            row = sc.candidate.payload or {}
            results.append(RecallResult(
                tx_hash=row.get("tx_hash") or row.get("block_hash") or "",
                score=sc.score,
                fork=str(row.get("fork", "")),
                source=str(row.get("source", "")),
                summary=str(row.get("summary", ""))[:160],
                cosine=sc.cosine,
                base_level=sc.base_level,
                norm_base_level=sc.norm_base_level,
                importance=sc.importance,
            ))
        return results

    # ── stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_recall_calls": self._total_recall_calls,
            "total_contract_hits": self._total_contract_hits,
            "total_fallbacks": self._total_fallbacks,
            "helper_contract_loaded": HELPER_CONTRACT_ID in self._contracts,
        }


__all__ = [
    "EngineRecall",
    "RecallResult",
    "HELPER_CONTRACT_ID",
    "DEFAULT_CANDIDATES_FROM",
    "DEFAULT_K",
]
