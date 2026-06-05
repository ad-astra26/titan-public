"""Sovereignty score — the ONE token-economy metric (RFP_synthesis_decision_authority P3).

`S = w_e·E + w_v·V`, the single per-reply sovereignty score that replaces the four
disagreeing legacy scores (gatekeeper / output_verifier / meditation / the old
recall-ratio meter). It measures **how much of each reply Titan supplied from his
own substrate** vs fresh LLM generation. The LLM always stays in the loop
(INV-SDA-7) — S *measures* the saving, it does not skip the LLM.

Terms (both in [0,1]):

  • **E — substrate-cited token share.** Of the salient tokens in the response,
    the fraction that trace to a *cited* substrate item. We take the union of
    each cited item's salient tokens that actually appear in the response
    (the `CitedUseDetector` substring heuristic, reused for tokenizer parity),
    divided by the count of salient response tokens. High E = most of the reply's
    vocabulary came from Titan's recalled/known substrate, not LLM construction.

  • **V — inner-state share.** Of the cited substrate items, the fraction sourced
    from VCB inner-titan state (`SurfacedItem.source == "vcb"`) vs all cited items.
    High V = the reply leaned on Titan's *own current inner state*, not just
    external recall.

  S = clamp(w_e·E + w_v·V, 0, 1). Default weights 0.7 / 0.3 (RFP §6 D; tweakable
  in ONE place). Pure + deterministic — no LLM call, no I/O, never raises. Computed
  once at the post-LLM OVG boundary (cheap, O(response tokens)) so the reply's
  TIMECHAIN_COMMIT can anchor S; persistence + rolling aggregation run off the hot
  path in the synthesis post-turn pipeline (INV-SDA-11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from titan_hcl.synthesis.cited_use import CitedUseDetector, SurfacedItem

__all__ = [
    "SovereigntyScore",
    "WEIGHT_E",
    "WEIGHT_V",
    "VCB_SOURCE",
    "compute_sovereignty_score",
]

# The ONE place the weights live (RFP §6 D — calibration is a follow-up that
# edits only these two constants / their override).
WEIGHT_E = 0.7
WEIGHT_V = 0.3

# The source tag that marks a surfaced item as VCB inner-titan state (the V
# numerator). Kept as a constant so the producer (agno surfaced-item
# registration) and this consumer never drift on the string.
VCB_SOURCE = "vcb"


@dataclass(frozen=True)
class SovereigntyScore:
    """One reply's sovereignty decomposition. `s` is the anchored headline."""

    s: float                  # clamp(w_e·E + w_v·V, 0, 1) — the metric
    e: float                  # substrate-cited token share
    v: float                  # inner-state (VCB) share of cited substrate
    cited_count: int          # |cited substrate items|
    vcb_cited_count: int      # |cited items with source == VCB_SOURCE|
    response_token_count: int # |salient response tokens| (E denominator)


def compute_sovereignty_score(
    *,
    response_text: str,
    surfaced_items: Iterable[SurfacedItem],
    cited_item_ids: Iterable[str],
    detector: Optional[CitedUseDetector] = None,
    w_e: float = WEIGHT_E,
    w_v: float = WEIGHT_V,
) -> SovereigntyScore:
    """Compute the per-reply sovereignty score `S = w_e·E + w_v·V`.

    Args:
        response_text: the finalized LLM response.
        surfaced_items: every item surfaced into context this turn (each tagged
            with `.source`). The cited subset is selected by `cited_item_ids`.
        cited_item_ids: the `item_id`s the response cited (from
            `CitedUseDetector.detect` / `knowledge_moment_signal`) — the single
            cited-use computation is reused, never recomputed here.
        detector: a `CitedUseDetector` whose tokenizer defines "salient token"
            (defaults to a fresh one so the function is self-contained + the
            tokenization matches the cited-use gate exactly).
        w_e, w_v: weights (default 0.7 / 0.3).

    Soft + total: never raises; degenerate input → an all-zero score. A reply
    that cited nothing (or surfaced nothing) is S=0 — Titan supplied none of it
    from substrate, which is the honest reading.
    """
    det = detector or CitedUseDetector()
    items = [it for it in (surfaced_items or [])
             if isinstance(it, SurfacedItem) and it.item_id]
    cited_ids = {str(i) for i in (cited_item_ids or []) if i}

    resp_tokens = det._tokenize_set(response_text or "")
    n_resp = len(resp_tokens)

    cited_items = [it for it in items if it.item_id in cited_ids]
    cited_count = len(cited_items)
    vcb_cited = sum(1 for it in cited_items if it.source == VCB_SOURCE)

    # E — union of cited items' salient tokens that actually appear in the
    # response (union so a token shared by two cited items is counted once),
    # over the salient response-token count.
    if n_resp and cited_items:
        attributed: set[str] = set()
        for it in cited_items:
            attributed |= det._salient_tokens(it) & resp_tokens
        e = len(attributed) / n_resp
    else:
        e = 0.0

    # V — fraction of cited substrate that is VCB inner-titan state.
    v = (vcb_cited / cited_count) if cited_count else 0.0

    e = _clamp01(e)
    v = _clamp01(v)
    s = _clamp01(w_e * e + w_v * v)

    return SovereigntyScore(
        s=s, e=e, v=v,
        cited_count=cited_count,
        vcb_cited_count=vcb_cited,
        response_token_count=n_resp,
    )


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
