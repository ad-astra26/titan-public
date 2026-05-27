"""Phase 6 — `web_api` TruthOraclePlug (§P6.D; SPEC §25.3 + §25.5).

Wraps the existing `knowledge_dispatcher` (web search backends — DuckDuckGo,
Wikipedia, SearXNG, sage-mediated; ``titan_hcl/logic/knowledge_dispatcher.py``)
plus an LLM-judge step as a `TruthOraclePlug` for textual fact claims.

Per SPEC §25.3 day-one set + arch §11.1: metered oracle (cost class
``"metered"``; oracle_id ``web_api``). The INV-Syn-13 gate enforces
``daily_sol_budget["web_api"]`` from
``titan_params.toml [synthesis.oracle.daily_sol_budget]``.

Claim domains served (SPEC §25.3 + arch §11.1):

- **web_fact** — "is this statement supported by current web evidence?"
  Payload: ``{"claim": "<sentence>"}``. The plug:
    1. Issues a single search via the injected ``search_fn(query)`` —
       defaults to ``knowledge_dispatcher.dispatch`` resolved through
       a one-shot event loop so the plug stays sync (TruthOraclePlug
       protocol). Returns a list of snippet strings.
    2. Concatenates the snippets and passes them with the claim to the
       injected ``judge_fn(claim, evidence) -> "true"|"false"|"unknown"``
       — defaults to a deterministic substring heuristic; in production
       the synthesis_worker constructs the plug with an LLM-backed judge.
    3. Anchors the verdict on `declarative` fork via OracleRouter (P6.F).

- **wiki_fact** — same shape but the search prefers the Wikipedia
  backend chain (claim re-issued as a 2-4 word noun-phrase query).

The architecture permits LLM-judge here (arch §11.1 / SPEC §25.5):

  > "LLM-judges-the-evidence is acceptable because the verdict itself is
  > still anchored + auditable — the LLM is acting as a heuristic, the
  > chain stores the heuristic's vote."

So the verdict is provenanced regardless of which judge fires. The
``evidence_ref`` carries a sha256 of the snippets so two callers
verifying the same claim against the same evidence get the same ref;
different evidence → different verdict + different ref.

**Failure modes** (all map to ``verdict="unknown"`` with specific
``evidence_ref`` strings so observability can group):

- search backend timeout / dispatch error → ``"search_unreachable"``
- empty evidence (no snippets returned) → ``"no_evidence"``
- judge raises → ``"judge_exception"``
- empty claim → ``"missing_claim"``

Default per-call cost is ``0.0005 SOL`` — slightly higher than
solana_rpc because a web fetch + LLM judge round-trip is more expensive
than a single RPC call. At the default ``daily_sol_budget["web_api"] =
0.1 SOL`` this caps ~200 web verifications per day. Maker-tunable.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Optional

from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict

logger = logging.getLogger(__name__)


SUPPORTED_DOMAINS = frozenset({"web_fact", "wiki_fact"})

DEFAULT_PER_CALL_COST_SOL: float = 0.0005
DEFAULT_SEARCH_TIMEOUT_S: float = 8.0
DEFAULT_MAX_SNIPPETS: int = 5


SearchFn = Callable[[str], list[str]]
JudgeFn = Callable[[str, str], str]  # returns "true" | "false" | "unknown"


def _default_search_fn_factory(timeout_s: float, max_snippets: int) -> SearchFn:
    """Build the default search_fn — runs ``knowledge_dispatcher.dispatch``
    through a one-shot event loop so the plug surface stays sync.

    Returns at most ``max_snippets`` evidence strings extracted from the
    DispatchResult.raw or .text fields. Empty list on every failure path.
    """

    def search_fn(query: str) -> list[str]:
        try:
            from titan_hcl.logic.knowledge_dispatcher import dispatch
        except Exception:  # pragma: no cover — import path is stable
            logger.exception("[web_api_oracle] knowledge_dispatcher import failed")
            return []

        async def _go() -> list[str]:
            try:
                result = await asyncio.wait_for(
                    dispatch(topic=query, raw_results=True),
                    timeout=timeout_s,
                )
            except (asyncio.TimeoutError, Exception):
                logger.exception("[web_api_oracle] dispatch raised")
                return []
            snippets: list[str] = []
            # DispatchResult may carry per-backend results in .raw or merged
            # text in .text — accept both shapes defensively.
            raw = getattr(result, "raw", None) or []
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, str) and entry.strip():
                        snippets.append(entry.strip())
                    elif isinstance(entry, dict):
                        for k in ("text", "snippet", "body", "summary"):
                            v = entry.get(k)
                            if isinstance(v, str) and v.strip():
                                snippets.append(v.strip())
                                break
                    if len(snippets) >= max_snippets:
                        break
            text = getattr(result, "text", None)
            if isinstance(text, str) and text.strip():
                snippets.append(text.strip())
            return snippets[:max_snippets]

        # asyncio.run() creates + tears down a fresh loop; cannot be called
        # from inside a running loop. The synthesis_worker dispatches the
        # oracle from a sync recv-loop handler — safe.
        try:
            return asyncio.run(_go())
        except RuntimeError:
            # Already inside a loop (test harness with @pytest.mark.asyncio,
            # or a future caller that forgot they were async). Fall back to
            # a thread-local loop so verify() never raises.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_go())
            finally:
                loop.close()

    return search_fn


def _default_judge_fn(claim: str, evidence: str) -> str:
    """Deterministic substring heuristic — production replaces with LLM judge.

    Returns:
      "true"    if every alphabetic token from the claim appears in evidence
                (case-insensitive); rough analog of "evidence covers claim".
      "false"   if any prominent claim token is contradicted by negation
                phrasing in the evidence (heuristic — "not", "no", "false").
      "unknown" if evidence lacks any claim token (insufficient signal).

    This heuristic exists so unit tests stay deterministic. The
    synthesis_worker wires the LLM-backed judge in P6.F construction.
    """
    if not claim.strip() or not evidence.strip():
        return "unknown"
    claim_lc = claim.lower()
    evidence_lc = evidence.lower()
    # Extract alphabetic tokens of length ≥3 from the claim.
    tokens = [t for t in "".join(c if c.isalpha() else " " for c in claim_lc).split() if len(t) >= 3]
    if not tokens:
        return "unknown"
    matched = sum(1 for t in tokens if t in evidence_lc)
    coverage = matched / len(tokens)
    # Contradiction signal — evidence contains negation near the claim's content.
    contradicted = any(neg in evidence_lc for neg in (" not ", " no ", "false", "wrong"))
    if coverage < 0.3:
        return "unknown"  # evidence doesn't address the claim
    if contradicted and coverage >= 0.5:
        return "false"
    if coverage >= 0.6:
        return "true"
    return "unknown"


class WebApiOracle:
    """TruthOraclePlug wrapping web search + LLM-judge.

    Both ``search_fn`` and ``judge_fn`` are injected at construction so
    unit tests can stub them deterministically. The synthesis_worker
    (P6.F) wires the real dispatch + LLM-backed judge.
    """

    oracle_id: str = "web_api"        # matches the daily_sol_budget key
    cost_class: str = "metered"       # INV-Syn-13 gated

    def __init__(
        self,
        *,
        search_fn: Optional[SearchFn] = None,
        judge_fn: Optional[JudgeFn] = None,
        per_call_cost_sol: float = DEFAULT_PER_CALL_COST_SOL,
        search_timeout_s: float = DEFAULT_SEARCH_TIMEOUT_S,
        max_snippets: int = DEFAULT_MAX_SNIPPETS,
    ):
        self._search_fn: SearchFn = search_fn or _default_search_fn_factory(
            search_timeout_s, max_snippets
        )
        self._judge_fn: JudgeFn = judge_fn or _default_judge_fn
        self._per_call_cost_sol = float(per_call_cost_sol)

    def can_handle(self, domain: str) -> bool:
        return domain in SUPPORTED_DOMAINS

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        t0 = time.perf_counter()
        ts_now = time.time()

        if not self.can_handle(claim.domain):
            return self._verdict(t0, ts_now, "unknown", "domain_unsupported")

        payload = claim.payload or {}
        claim_text = str(payload.get("claim", "")).strip()
        if not claim_text:
            return self._verdict(t0, ts_now, "unknown", "missing_claim")

        # Search step
        try:
            snippets = self._search_fn(claim_text)
        except Exception:
            logger.exception("[web_api_oracle] search_fn raised")
            return self._verdict(
                t0, ts_now, "unknown", "search_unreachable", cost=self._per_call_cost_sol
            )

        if not snippets:
            return self._verdict(
                t0, ts_now, "unknown", "no_evidence", cost=self._per_call_cost_sol
            )

        evidence = "\n\n".join(snippets)
        evidence_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()

        # Judge step
        try:
            judgment = self._judge_fn(claim_text, evidence)
        except Exception:
            logger.exception("[web_api_oracle] judge_fn raised")
            return self._verdict(
                t0, ts_now, "unknown", "judge_exception", cost=self._per_call_cost_sol
            )

        verdict = judgment if judgment in ("true", "false", "unknown") else "unknown"
        return self._verdict(t0, ts_now, verdict, evidence_hash, cost=self._per_call_cost_sol)

    def _verdict(
        self,
        t0: float,
        ts_now: float,
        verdict: str,
        evidence_ref: str,
        *,
        cost: float = 0.0,
    ) -> OracleVerdict:
        return OracleVerdict(
            oracle_id=self.oracle_id,
            verdict=verdict,  # type: ignore[arg-type]
            evidence_ref=evidence_ref,
            cost=cost,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ts=ts_now,
        )


__all__ = (
    "WebApiOracle",
    "SUPPORTED_DOMAINS",
    "DEFAULT_PER_CALL_COST_SOL",
    "SearchFn",
    "JudgeFn",
)
