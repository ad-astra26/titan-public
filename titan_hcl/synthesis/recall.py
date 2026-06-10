"""EngineRecall — contract-driven episodic recall.

Synthesis Engine Phase 2 (PLAN_synthesis_engine_Phase2.md §2D, D-P2-1)
+ Phase 3 (PLAN_synthesis_engine_Phase3.md P3.D — granularity-aware
retrieval, D-SPEC-127).

The recall pipeline is encoded END-TO-END inside the
`actr_episodic_recall_helper` contract (PLAN §2C.a). EngineRecall is the
thin coordinator that:

  1. embeds the query text (caller-supplied embedder; SPEC-correct per
     arch §3.5 — RuleEvaluator stays model-free, D-P2-3),
  2. loads the contract,
  3. **(Phase 3 P3.D)** augments the rule list with a granularity-scoped
     FORK_READ when the caller requests `granularity={turn,topic,session}`,
  4. invokes `RuleEvaluator.evaluate(rules, ctx, initial_variables=...)`
     with `$query_embedding` + `$current_chat_tx` (+ P3 `$granularity_tag`)
     pre-seeded,
  5. consumes the `rank_composite` action by:
       - merging the candidate-source $vars (e.g. $base + $semantic +
         $threaded + P3 $granularity_filtered) — dedup by tx_hash,
       - building `Candidate` objects (cosine from FAISS score if
         present, else default; importance default 0.5 per arch §5.3),
       - running `composite_score()` with the contract-supplied weights,
  6. returns the top-K ScoredCandidate-equivalent dicts.

**Granularity (arch §7 — P3.D):** `{turn, topic, session}` as a
query-time parameter (arch §7: "no extra storage"). EngineRecall
augments the contract rule list with a conditional FORK_READ scoped to
the granularity tag (`chat:<chat_id>` for turn/session, `topic:<X>`
for topic). The candidate source `$granularity_filtered` is added to
`candidates_from` only when granularity is requested — base contract
JSON unchanged, granularity-less callers see legacy P2 semantics
byte-identical.

Contract-driven design (D-P2-1 / INV-Syn-6): Maker tunes ranking
end-to-end by editing the JSON + re-signing. Granularity is a
*parametric extension by the consumer* (the recipe lives in the
contract; granularity prunes to a runtime context the contract can't
know ahead of time).

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

# Phase 3 P3.D — granularity-aware retrieval constants.
GRANULARITY_TURN = "turn"
GRANULARITY_TOPIC = "topic"
GRANULARITY_SESSION = "session"
# Phase 9 §13.2 — RECALL sub-mode granularities (FORK_READ-based, no embedder).
# archive          → RECALL.chain_archive          → FORK_READ(meta) + CROSS_REF
# autobiographical → RECALL.autobiographical_relevant → FORK_READ(main genesis) + DIFF
GRANULARITY_ARCHIVE = "archive"
GRANULARITY_AUTOBIOGRAPHICAL = "autobiographical"
# RFP_titan_authored_soul_diary §7.P4 — self-recall: traverse the Kuzu `Self`
# hub (SELF_HAS_ENGRAM diary+self-engrams, SELF_HAS_SKILL skills), per-spine like
# "concept"/"procedural" (NOT a FORK_READ mode). Requires kuzu_reader.
GRANULARITY_SELF = "self"
_VALID_GRANULARITIES = frozenset(
    {GRANULARITY_TURN, GRANULARITY_TOPIC, GRANULARITY_SESSION})
# Default recency windows (hours) for the FORK_READ-based modes.
ARCHIVE_WINDOW_H = 720           # meta-fork archive: 30d
AUTOBIOGRAPHICAL_WINDOW_H = 8760  # genesis self-journey: ~1y

# $var used by the granularity-augmented FORK_READ. Public constant so
# tests + a future Maker-tuned contract version can reference it.
GRANULARITY_SOURCE_VAR = "$granularity_filtered"
# Recency window (hours) for the granularity-scoped FORK_READ. Tighter
# than the default `$base` 168h because granularity = "this conversation
# / this topic, recent". Maker-tunable via the contract once promoted.
GRANULARITY_DEFAULT_WINDOW_H = {
    GRANULARITY_TURN: 24,     # "this turn" → last 24h of this chat
    GRANULARITY_TOPIC: 168,   # "this topic" → 1 week of the topic
    GRANULARITY_SESSION: 720, # "this session" → 30 days of chat_id
}
# Per-granularity row cap. Turn is the tightest (most recent few);
# session opens the widest aperture.
GRANULARITY_DEFAULT_LIMIT = {
    GRANULARITY_TURN: 10,
    GRANULARITY_TOPIC: 30,
    GRANULARITY_SESSION: 50,
}


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
        # Phase 4 §P4.H — kuzu reader for granularity="concept" recall.
        # Anything exposing spine_list_concepts + spine_get_latest_concept
        # (TitanKnowledgeGraph in production; BridgeRecall later for
        # cross-process readers — INV-Syn-4 / G18). None disables concept-
        # granularity recall (returns None — caller falls back).
        kuzu_reader: Optional[Any] = None,
        # Phase 8 §P8.D — procedural reader for granularity="procedural"
        # recall. ProceduralSkillReader (in synthesis_worker) or BridgeRecall
        # wrapper (cross-process). None disables that granularity.
        procedural_reader: Optional[Any] = None,
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
            kuzu_reader: Phase 4 — optional Kuzu graph handle for
                granularity="concept" recall. When supplied, recall()
                with granularity="concept" returns concept-spine results
                ranked by groundedness; when None, that granularity
                returns None (caller falls back to per-TX recall).
        """
        self._evaluator = rule_evaluator
        self._activation_lookup = activation_lookup
        self._embedder = embedder
        self._kuzu_reader = kuzu_reader
        self._procedural_reader = procedural_reader
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
        # Phase 3 P3.D — granularity-aware retrieval (arch §7).
        granularity: Optional[str] = None,
        chat_id: Optional[str] = None,
        topic_tag: Optional[str] = None,
        # P4 (RFP_synthesis_decision_authority) embed-once: a caller-supplied
        # shared prompt vector (get_text_embedder() output threaded from the
        # agno PreHook). When provided it is reused verbatim for the SEARCH
        # cosine — the injected embedder is NOT invoked (1 embed/turn, G9).
        query_vec: Optional[list] = None,
    ) -> Optional[list[RecallResult]]:
        """Contract-driven episodic recall.

        Args:
            query_text: text to embed for SEARCH cosine.
            current_chat_tx: pivot tx_hash for CROSS_REF threading
                (P2 behavior — pre-existing).
            k: max results to return (post-composite-rank cap).
            granularity: P3.D — one of `{"turn", "topic", "session"}`
                or None (default). When set, augments the contract's
                rule list with a granularity-scoped FORK_READ source
                and includes it in the rank_composite candidate set.
                Unknown values are treated as None with a WARN log.
            chat_id: required when granularity in {turn, session} —
                the session whose TXs the scoped FORK_READ filters to
                (`tags_include=["chat:<chat_id>"]`).
            topic_tag: required when granularity == "topic" — the
                topic tag to scope to (auto-prefixed `topic:` if the
                caller didn't include it).

        Returns:
            list[RecallResult] sorted by composite score descending, or
            None when the helper contract is absent / disabled / fails
            to fire — caller falls back to cosine-only ranking.
        """
        self._total_recall_calls += 1

        # Phase 4 §P4.H — concept-granularity recall. Bypasses the
        # contract pipeline because the result type is per-spine, not
        # per-TX (the Kuzu spine IS the per-concept index by arch §6.1).
        # The embedder is NOT required for concept granularity (P4 ranks
        # by groundedness + name substring match; cosine-over-names is
        # Phase 5+); kuzu_reader IS required.
        if granularity == "concept":
            results = self._concept_granularity_recall(
                query_text=query_text, k=k,
            )
            if results is None:
                self._total_fallbacks += 1
            else:
                self._total_contract_hits += 1
            return results

        # §7.P4 — self-recall: traverse the Kuzu `Self` hub (per-spine, like
        # concept; bypasses the contract pipeline). kuzu_reader-gated.
        if granularity == GRANULARITY_SELF:
            results = self._self_granularity_recall(
                query_text=query_text, k=k,
            )
            if results is None:
                self._total_fallbacks += 1
            else:
                self._total_contract_hits += 1
            return results

        # Phase 8 §P8.D — procedural-granularity recall. Bypasses the
        # contract pipeline (the result type is per-skill, not per-TX —
        # `procedural_skills` is its own index by arch §8.1). Uses the
        # injected procedural_reader (binds to ProceduralSkillStore in
        # synthesis_worker); None disables this granularity (returns
        # None, caller falls back to per-TX recall).
        if granularity == "procedural":
            if self._procedural_reader is None:
                logger.debug(
                    "[EngineRecall] procedural granularity requested but "
                    "procedural_reader is None — caller falls back",
                )
                self._total_fallbacks += 1
                return None
            try:
                skill_rows = self._procedural_reader.recall(query_text, k=k)
            except Exception as e:
                logger.warning(
                    "[EngineRecall] procedural_reader raised: %s — fallback", e,
                )
                self._total_fallbacks += 1
                return None
            if not skill_rows:
                self._total_fallbacks += 1
                return []
            self._total_contract_hits += 1
            # Map procedural skill dicts onto uniform RecallResult shape so
            # downstream callers (engine, agno tool) handle one type.
            return [
                RecallResult(
                    tx_hash=row.get("skill_id", ""),
                    score=float(row.get("match_score") or 0.0),
                    fork="procedural_skill",
                    source="synthesis_procedural_skill",
                    summary=row.get("name") or row.get("nl_description") or "",
                    cosine=float(row.get("cosine_surrogate") or 0.0),
                    importance=float(row.get("utility_score") or 0.0),
                )
                for row in skill_rows
            ]

        # Phase 9 §13.2 — FORK_READ-based RECALL sub-modes (INV-Syn-22). These
        # bypass the embedder + helper-contract path: they read a fork directly
        # (meta archive / FORK_MAIN genesis) and rank by activation + importance
        # (cosine defaults to 0.5 — recency/frequency carries the ranking).
        if granularity == GRANULARITY_ARCHIVE:
            results = self._fork_scoped_recall(
                fork="meta", k=k, current_chat_tx=current_chat_tx,
                with_cross_ref=True, since_hours=ARCHIVE_WINDOW_H,
                src_label="synthesis_archive",
            )
            self._tally(results)
            return results
        if granularity == GRANULARITY_AUTOBIOGRAPHICAL:
            results = self._fork_scoped_recall(
                fork="main", k=k, current_chat_tx=current_chat_tx,
                with_cross_ref=False, since_hours=AUTOBIOGRAPHICAL_WINDOW_H,
                src_label="synthesis_autobiographical",
            )
            self._tally(results)
            return results

        # P4 embed-once: a caller-supplied query_vec (the shared
        # get_text_embedder() vector threaded from the agno PreHook) is reused
        # verbatim; only when it is absent do we require + invoke the embedder.
        if query_vec is None and not self._embedder:
            logger.debug("[EngineRecall] no embedder injected — fallback")
            self._total_fallbacks += 1
            return None

        contract = self._load_contract(HELPER_CONTRACT_ID)
        if contract is None:
            self._total_fallbacks += 1
            return None

        # 1. Embed the query (or reuse the shared embed-once vector).
        if query_vec is not None:
            qe = list(query_vec)
        else:
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

        # 2. Resolve the granularity tag + build augmented rule list.
        #    None / unknown granularity → empty tag → no augmentation
        #    (legacy P2 behavior byte-identical).
        granularity_tag = _resolve_granularity_tag(
            granularity, chat_id, topic_tag)
        rules = list(contract["rules"])
        if granularity_tag:
            granularity_rule, gran_var = _build_granularity_rule(
                granularity, granularity_tag)
            rules.append(granularity_rule)
            # The contract's OR-gate is the LAST rule (per
            # actr_episodic_recall_helper convention). To include the
            # new granularity source in the rank_composite action, we
            # mutate the OR-gate's action.candidates_from to append
            # gran_var. The OR clauses also gain an IF for length>=1
            # so the OR fires when granularity alone yields rows.
            _augment_or_gate(rules, gran_var)

        # 3. Build the eval context + variable preamble.
        ctx = {
            "event": "retrieval_request",
            "query_text_len": len(query_text or ""),
            "k": k,
            "granularity": granularity or "",
        }
        initial_vars: dict[str, Any] = {
            "$query_embedding": qe,
            "$current_chat_tx": current_chat_tx or "",
            # Always seed even when unused — keeps the contract rule
            # references resolvable when Maker later promotes the
            # granularity hook into the JSON.
            "$granularity_tag": granularity_tag,
        }

        # 4. Evaluate via RuleEvaluator. The contract's binding rules
        #    (FORK_READ / SEARCH / CROSS_REF) populate variables; the
        #    final OR-gate emits the rank_composite action.
        try:
            action = self._evaluator.evaluate(
                rules, ctx, initial_variables=initial_vars)
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

    # ── Phase 9 FORK_READ-based RECALL sub-modes (INV-Syn-22) ──────────

    def _tally(self, results: Optional[list]) -> None:
        if results is None:
            self._total_fallbacks += 1
        else:
            self._total_contract_hits += 1

    def _fork_scoped_recall(
        self,
        *,
        fork: str,
        k: int,
        current_chat_tx: Optional[str],
        with_cross_ref: bool,
        since_hours: int,
        src_label: str,
    ) -> Optional[list[RecallResult]]:
        """RECALL.chain_archive / .autobiographical_relevant (§13.2).

        Runs a FORK_READ on `fork` (+ optional CROSS_REF threading through the
        current chat TX), then ranks the rows by the ACT-R composite (activation
        + importance; cosine defaults to 0.5 since there is no query embedding —
        archive/autobiographical recall is recency/frequency-driven). Reuses the
        evaluator's binding-op machinery + `_consume_rank_composite`. Returns
        None on evaluator failure (caller falls back); [] when the fork is empty.
        """
        src_var = "$fork_scoped_src"
        rules: list[dict] = [{
            "op": "FORK_READ", "fork": fork,
            "since_hours": since_hours, "limit": 50, "store": src_var,
        }]
        candidates_from = [src_var]
        initial_vars: dict[str, Any] = {"$current_chat_tx": current_chat_tx or ""}
        if with_cross_ref and current_chat_tx:
            xref_var = "$fork_scoped_xref"
            rules.append({
                "op": "CROSS_REF", "tx_hash": "$current_chat_tx",
                "via_field": "parent_chat_tx", "limit": 20, "store": xref_var,
            })
            candidates_from.append(xref_var)

        ctx = {"event": "retrieval_request", "k": k}
        try:
            # Binding ops populate the evaluator's $vars; no action fires (we
            # synthesize the rank_composite action ourselves below).
            self._evaluator.evaluate(rules, ctx, initial_variables=initial_vars)
        except Exception as exc:
            logger.warning(
                "[EngineRecall] %s fork-scoped eval failed: %s — fallback",
                src_label, exc, exc_info=True,
            )
            return None

        action = {
            "action": "rank_composite",
            "candidates_from": candidates_from,
            "weights": {
                "w_b": DEFAULT_W_B, "w_s": DEFAULT_W_S,
                "w_r": DEFAULT_W_R, "w_p": DEFAULT_W_P,
            },
            "limit": k,
        }
        results = self._consume_rank_composite(action, k)
        # Stamp the RECALL-mode label for observability (which sub-mode produced
        # this result), overriding the per-row producer source.
        for r in results:
            r.source = src_label
        return results

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

    # ── Phase 4 §P4.H — concept-granularity recall ───────────────────

    def _concept_granularity_recall(
        self,
        *,
        query_text: str,
        k: int,
    ) -> Optional[list[RecallResult]]:
        """Return up to `k` Concept spines ranked by composite score.

        Scoring:
          score = groundedness * (1 + name_match_boost)

        where `name_match_boost = 0.5` when any token of query_text is a
        case-folded substring of the concept name (cheap, no embedder
        needed); 0 otherwise. This makes high-groundedness concepts
        always rank well + lets a query about "linux" surface
        linux-named concepts above unrelated high-groundedness ones.

        Returned RecallResult shape — uniform with per-TX recall so
        downstream consumers handle one type:
          tx_hash = the spine's latest anchor_tx
          fork    = "concept_spine"  (disambiguator)
          source  = "synthesis_concept_spine"
          summary = the concept's human-readable name
          importance = the concept's groundedness  (so consumers that
                       only look at importance still get a useful signal)

        Returns None if no kuzu_reader is wired (caller falls back to
        legacy P2/P3 recall) or if the spine is empty.
        """
        if self._kuzu_reader is None:
            logger.debug(
                "[EngineRecall] concept granularity requested but kuzu_reader "
                "is None — caller falls back to per-TX recall",
            )
            return None

        try:
            # Pull a generous candidate pool; concept count is bounded by
            # the dream-boundary cap (§P4.G) so this is cheap.
            candidates = self._kuzu_reader.spine_list_concepts(limit=200)
        except Exception as e:
            logger.warning(
                "[EngineRecall] spine_list_concepts failed: %s — fallback", e,
            )
            return None

        if not candidates:
            return None

        # Cheap query-token match against concept name. Case-folded
        # substring; tokens shorter than 3 chars dropped (stop-word guard).
        query_tokens = {
            t.lower() for t in (query_text or "").split()
            if len(t) >= 3
        }

        scored: list[tuple[float, dict]] = []
        for row in candidates:
            g = float(row.get("groundedness", 0.0) or 0.0)
            name = str(row.get("name", "") or "")
            name_lc = name.lower()
            boost = 0.5 if any(t in name_lc for t in query_tokens) else 0.0
            score = g * (1.0 + boost)
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        top = scored[:k]

        results: list[RecallResult] = []
        for score, row in top:
            results.append(RecallResult(
                tx_hash=str(row.get("anchor_tx", "")),
                score=score,
                fork="concept_spine",
                source="synthesis_concept_spine",
                summary=str(row.get("name", "")),
                cosine=0.0,
                base_level=0.0,
                norm_base_level=0.0,
                importance=float(row.get("groundedness", 0.0) or 0.0),
            ))
        return results

    def _self_granularity_recall(
        self,
        *,
        query_text: str,
        k: int,
    ) -> Optional[list[RecallResult]]:
        """RECALL.self — the focused self-recall (RFP_titan_authored_soul_diary
        §7.P4, decided mechanic): traverse the `Self` hub
        (`TitanKnowledgeGraph.spine_self_recall`, built P3a) — SELF_HAS_ENGRAM
        (diary + self-engrams) + SELF_HAS_SKILL (skills) — to surface his path +
        abilities in ONE hop, WITHOUT scanning whole memory. Ranks by
        groundedness/utility × (1 + name-match boost); a small floor keeps a
        fresh (groundedness-0) diary engram surfacing on a self-query (the hub
        traversal already ordered engrams newest-first).

        Returned RecallResult shape — uniform with per-TX recall:
          engram → tx_hash=anchor_tx, fork="self_hub",  source="synthesis_self_recall"
          skill  → tx_hash=skill_id,  fork="self_skill", source="synthesis_self_skill"

        Returns None if no kuzu_reader is wired / it lacks `spine_self_recall`
        (caller falls back to per-TX recall), or the hub is empty.
        """
        reader = self._kuzu_reader
        if reader is None or not hasattr(reader, "spine_self_recall"):
            logger.debug(
                "[EngineRecall] self granularity requested but no kuzu_reader "
                "with spine_self_recall — caller falls back",
            )
            return None
        try:
            hub = reader.spine_self_recall() or {}
        except Exception as e:
            logger.warning(
                "[EngineRecall] spine_self_recall failed: %s — fallback", e,
            )
            return None

        engrams = hub.get("engrams") or []
        skills = hub.get("skills") or []
        if not engrams and not skills:
            return None

        query_tokens = {
            t.lower() for t in (query_text or "").split() if len(t) >= 3
        }

        def _boost(name: str) -> float:
            nlc = name.lower()
            return 0.5 if any(t in nlc for t in query_tokens) else 0.0

        scored: list[tuple[float, RecallResult]] = []
        for row in engrams:
            g = float(row.get("groundedness", 0.0) or 0.0)
            name = str(row.get("name", "") or "")
            score = max(g, 0.05) * (1.0 + _boost(name))
            scored.append((score, RecallResult(
                tx_hash=str(row.get("anchor_tx", "") or ""),
                score=score, fork="self_hub", source="synthesis_self_recall",
                summary=name, cosine=0.0, base_level=0.0, norm_base_level=0.0,
                importance=g,
            )))
        for row in skills:
            u = float(row.get("utility_score", 0.0) or 0.0)
            name = str(row.get("name", "") or "")
            score = max(u, 0.05) * (1.0 + _boost(name))
            scored.append((score, RecallResult(
                tx_hash=str(row.get("skill_id", "") or ""),
                score=score, fork="self_skill", source="synthesis_self_skill",
                summary=name, cosine=0.0, base_level=0.0, norm_base_level=0.0,
                importance=u,
            )))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [rr for _, rr in scored[:k]]

    # ── stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_recall_calls": self._total_recall_calls,
            "total_contract_hits": self._total_contract_hits,
            "total_fallbacks": self._total_fallbacks,
            "helper_contract_loaded": HELPER_CONTRACT_ID in self._contracts,
        }


def _resolve_granularity_tag(
    granularity: Optional[str],
    chat_id: Optional[str],
    topic_tag: Optional[str],
) -> str:
    """Map P3.D granularity hint → the tag the FORK_READ filters on.

    Returns "" (sentinel for "no augmentation, legacy P2 behavior")
    when:
      - granularity is None or unknown
      - the required parameter for the granularity is missing
        (turn/session need chat_id; topic needs topic_tag)

    No exceptions; missing context logs DEBUG (not WARN — granularity
    is opt-in and absent context is the legacy path, not an error).
    """
    if not granularity:
        return ""
    if granularity not in _VALID_GRANULARITIES:
        logger.warning(
            "[EngineRecall] unknown granularity %r — degrading to legacy "
            "(valid: %s)", granularity, sorted(_VALID_GRANULARITIES))
        return ""
    if granularity in (GRANULARITY_TURN, GRANULARITY_SESSION):
        if not chat_id:
            logger.debug(
                "[EngineRecall] granularity=%s but no chat_id — degrading",
                granularity)
            return ""
        return f"chat:{chat_id}"
    # GRANULARITY_TOPIC
    if not topic_tag:
        logger.debug(
            "[EngineRecall] granularity=topic but no topic_tag — degrading")
        return ""
    # Auto-prefix `topic:` if the caller passed the bare topic.
    return topic_tag if topic_tag.startswith("topic:") else f"topic:{topic_tag}"


def _build_granularity_rule(
    granularity: str, granularity_tag: str,
) -> tuple[dict, str]:
    """Construct the FORK_READ rule scoped to the granularity tag.

    Returns (rule_dict, candidate_var_name). The rule reads from the
    conversation fork constrained by `tags_include=[granularity_tag]`;
    window + limit come from the per-granularity defaults
    (`GRANULARITY_DEFAULT_WINDOW_H` / `GRANULARITY_DEFAULT_LIMIT`).
    The result is stored in `GRANULARITY_SOURCE_VAR` (default
    `$granularity_filtered`).
    """
    window_h = GRANULARITY_DEFAULT_WINDOW_H.get(granularity, 168)
    limit = GRANULARITY_DEFAULT_LIMIT.get(granularity, 30)
    rule = {
        "op": "FORK_READ",
        "fork": "conversation",
        "filter": {"tags_include": [granularity_tag]},
        "since_hours": window_h,
        "limit": limit,
        "store": GRANULARITY_SOURCE_VAR,
    }
    return rule, GRANULARITY_SOURCE_VAR


def _augment_or_gate(rules: list, gran_var: str) -> None:
    """Mutate the last OR-gate in `rules` to include gran_var.

    Adds:
      - a new IF clause `{"op":"IF","field":"<gran_var>.length",
        "cmp":"GTE","value":1}` to the OR.clauses list, so the gate
        fires when ONLY granularity matches yielded rows.
      - the gran_var to action.candidates_from.

    Idempotent on duplicate gran_var. Soft-fails silently if the
    last rule is not an OR-gate with the expected shape — the
    granularity source is then still computed but won't participate
    in ranking (legacy P2 candidates dominate). Logs DEBUG on
    soft-fail so the issue surfaces if a contract is restructured.
    """
    if not rules:
        return
    last = rules[-1]
    if not isinstance(last, dict) or last.get("op") != "OR":
        logger.debug(
            "[EngineRecall] granularity augment: last rule is not OR — "
            "leaving gate unchanged")
        return
    clauses = last.get("clauses")
    then = last.get("then")
    if not isinstance(clauses, list) or not isinstance(then, dict):
        logger.debug(
            "[EngineRecall] granularity augment: OR-gate has unexpected "
            "shape — leaving unchanged")
        return
    cands = then.get("candidates_from")
    if not isinstance(cands, list):
        logger.debug(
            "[EngineRecall] granularity augment: action has no list "
            "candidates_from — leaving unchanged")
        return
    if gran_var in cands:
        return  # already augmented (idempotent)
    # Append the IF clause + the candidate var.
    clauses.append({
        "op": "IF",
        "field": f"{gran_var}.length",
        "cmp": "GTE",
        "value": 1,
    })
    cands.append(gran_var)


__all__ = [
    "EngineRecall",
    "RecallResult",
    "HELPER_CONTRACT_ID",
    "DEFAULT_CANDIDATES_FROM",
    "DEFAULT_K",
    # P3.D
    "GRANULARITY_TURN",
    "GRANULARITY_TOPIC",
    "GRANULARITY_SESSION",
    "GRANULARITY_ARCHIVE",
    "GRANULARITY_AUTOBIOGRAPHICAL",
    "GRANULARITY_SOURCE_VAR",
    "GRANULARITY_DEFAULT_WINDOW_H",
    "GRANULARITY_DEFAULT_LIMIT",
]
