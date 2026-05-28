"""CitedUseDetector — strict "LLM cited" gate heuristic (Synthesis Engine Phase 9).

INV-Syn-23: use-gated reinforcement requires `used_by_llm = true`. The soft
P1 gate emitted `MEMORY_RETRIEVAL_USED` for every *surfaced* item at retrieval
time; that lets frequently-surfaced items crowd out the long tail (the
rich-get-richer runaway, rFP §240). The strict gate reinforces only items the
LLM response *actually used*.

This detector runs agno-side, post-LLM (after `response_text` finalizes, before
send), and decides — heuristically, with no extra LLM call — which surfaced
retrieval items the response cited. Maker decision 2026-05-28: heuristic match
(response-text substring OR ≥1 CGN concept-id overlap). Conservative by design:
an unmatched cited item simply isn't reinforced (it decays naturally) — there is
no false-negative correctness harm, only a missed reinforcement.

The agno caller emits `MEMORY_RETRIEVAL_USED{item_id, ts, used_by_llm}` for
EVERY surfaced item — `True` for the cited subset returned here, `False` for the
rest (surfaced-not-cited; counted for telemetry, never reinforced). The
synthesis_worker consumer gates `record_access` on `used_by_llm=True`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["SurfacedItem", "CitedUseDetector"]


@dataclass
class SurfacedItem:
    """One retrieval item that was surfaced into LLM context for a chat turn.

    Assembled by the agno caller from the `retrieval` buffer rows (P7) + the
    VCB-built context for the turn. `item_id` is the stable activation key
    already used by the P1 emit (`mem:<id>` / `tx:<hash>`).
    """

    item_id: str
    title: str = ""
    content_snippet: str = ""
    concept_ids: list[str] = field(default_factory=list)


# Token splitter: words of letters/digits/underscore. Used for the substring
# heuristic so we match on whole salient tokens, not arbitrary fragments.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class CitedUseDetector:
    """Heuristic post-LLM cited-use detector (INV-Syn-23).

    Deterministic, no LLM call. A surfaced item is "cited" when EITHER:
      • a salient token (length ≥ `min_token_len`) from its title/content
        appears as a whole token in the (case-folded) response, OR
      • ≥ `concept_overlap_min` of its CGN concept_ids appear in the
        response-grounded concept_ids supplied by the caller.
    """

    def __init__(
        self,
        *,
        concept_overlap_min: int = 1,
        min_token_len: int = 4,
        max_salient_tokens: int = 40,
    ) -> None:
        self._concept_overlap_min = max(1, int(concept_overlap_min))
        self._min_token_len = max(1, int(min_token_len))
        self._max_salient_tokens = max(1, int(max_salient_tokens))

    # ── public API ────────────────────────────────────────────────────

    def detect(
        self,
        *,
        response_text: str,
        surfaced_items: Iterable[SurfacedItem],
        response_concept_ids: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """Return the subset of `item_id`s that the response cited.

        Args:
            response_text: the finalized LLM response.
            surfaced_items: items surfaced into context this turn.
            response_concept_ids: CGN concept_ids grounded from the response
                (optional; enables the concept-overlap match path). When
                absent, only the substring path fires.

        Soft, total: never raises; bad input → empty list.
        """
        items = list(surfaced_items or [])
        if not items:
            return []

        resp = response_text or ""
        if not resp.strip():
            return []

        resp_tokens = self._tokenize_set(resp)
        resp_concepts = {str(c) for c in (response_concept_ids or []) if c}

        cited: list[str] = []
        for item in items:
            if not isinstance(item, SurfacedItem) or not item.item_id:
                continue
            if self._is_cited(item, resp_tokens, resp_concepts):
                cited.append(item.item_id)
        # Dedup, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for iid in cited:
            if iid not in seen:
                seen.add(iid)
                out.append(iid)
        return out

    # ── heuristics ────────────────────────────────────────────────────

    def _is_cited(
        self,
        item: SurfacedItem,
        resp_tokens: set[str],
        resp_concepts: set[str],
    ) -> bool:
        # Concept-overlap path.
        if resp_concepts and item.concept_ids:
            overlap = sum(
                1 for c in item.concept_ids if str(c) in resp_concepts
            )
            if overlap >= self._concept_overlap_min:
                return True
        # Substring (whole-token) path.
        salient = self._salient_tokens(item)
        if salient and salient & resp_tokens:
            return True
        return False

    def _salient_tokens(self, item: SurfacedItem) -> set[str]:
        """Salient tokens from the item's title + content snippet — the
        tokens whose presence in the response signals the item was used.
        Capped to bound work on long snippets."""
        text = f"{item.title} {item.content_snippet}"
        toks = [
            t.lower()
            for t in _TOKEN_RE.findall(text)
            if len(t) >= self._min_token_len
        ]
        if len(toks) > self._max_salient_tokens:
            toks = toks[: self._max_salient_tokens]
        return set(toks)

    def _tokenize_set(self, text: str) -> set[str]:
        return {
            t.lower()
            for t in _TOKEN_RE.findall(text)
            if len(t) >= self._min_token_len
        }
