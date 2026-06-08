"""Context assembler — the single place that merges a chat turn's recall sources
into ONE source-tagged, content-hash-deduplicated surfaced-item list
(RFP_synthesis_decision_authority P4; realises INV-SDA-5).

Three substrate partitions feed a routed chat turn's grounded context:

  • ``"vcb"``    — VerifiedContextBuilder inner-titan state (vocabulary,
    knowledge_concepts, chain_archive, meta_wisdom, …). Titan's *current inner
    state* — the sovereignty **V** numerator.
  • ``"memory"`` — legacy ``core.memory.query`` (cognee-FAISS + mempool + keyword);
    the reflex / no-VCB fallback recall path.
  • ``"recall"`` — the synthesis tx_hash spine (EngineRecall); Titan's own
    timechain-anchored verified experience.

Before P4 only ``"recall"`` items were registered as surfaced items, so the
sovereignty **V** term (``vcb_cited / cited``) and the **E** term could never
credit VCB or memory content — V was structurally pinned at 0. The assembler
itemises ALL surfaced substrate, tags each at its source (the
``SurfacedItem.source`` the ``CitedUseDetector`` / ``compute_sovereignty_score``
consume), and de-duplicates by a normalised content hash so a fact surfaced by
two paths is delivered **and** counted exactly once (gate G6 — no content
twice).

De-dup collisions are resolved by ``SOURCE_PRIORITY``: the partition that is
injected into the prompt *wholesale* wins, so the surfaced-item set (what gets
*counted*) matches what the LLM actually *sees*. In the live PreHook the VCB (or
its memory fallback) block is injected as a whole formatted block, while the
recall block is rendered line-by-line and so can be filtered — therefore
``vcb``/``memory`` keep a colliding item and the redundant ``recall`` duplicate
line is suppressed. Display == count.

Pure + deterministic — no I/O, no LLM, never raises. The hot path imports
``assemble(...)``; persistence + scoring stay downstream (INV-SDA-11).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from titan_hcl.synthesis.cited_use import SurfacedItem

__all__ = [
    "AssembledItem",
    "assemble",
    "SOURCE_PRIORITY",
    "content_hash",
]

# De-dup keep-order: the partition injected into the prompt wholesale wins a
# content collision (see module docstring). vcb / memory are mutually exclusive
# in the live path, so their relative order is immaterial; both beat recall.
SOURCE_PRIORITY = ("vcb", "memory", "recall")

_WS_RE = re.compile(r"\s+")
_HASH_ID_LEN = 16          # stable id suffix for partitions with no native id
_SNIPPET_CAP = 512         # SurfacedItem content cap (matches PreHook stash)


@dataclass(frozen=True)
class AssembledItem:
    """One de-duplicated, source-tagged surfaced item.

    ``content_hash`` is the normalised-content dedup key; ``source`` is the
    sovereignty partition tag {"vcb","memory","recall"}.
    """

    item_id: str
    source: str
    content: str
    content_hash: str
    title: str = ""
    weight: float = 0.0
    concept_ids: tuple[str, ...] = ()

    def to_surfaced(self) -> SurfacedItem:
        """Project to the ``SurfacedItem`` the cited-use / sovereignty path eats."""
        return SurfacedItem(
            item_id=self.item_id,
            title=self.title,
            content_snippet=self.content[:_SNIPPET_CAP],
            concept_ids=list(self.concept_ids),
            source=self.source,
        )

    def to_stash_dict(self) -> dict[str, Any]:
        """Project to the dict shape the agno PreHook stashes in
        ``plugin._last_surfaced_items`` (read back via ``s.get(...)`` in the
        PostHook)."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "content_snippet": self.content[:_SNIPPET_CAP],
            "concept_ids": list(self.concept_ids),
            "source": self.source,
        }


def content_hash(text: str) -> str:
    """Normalised (case-folded, whitespace-collapsed, stripped) SHA-256 of the
    content — the dedup key. Same fact rendered with different spacing/case
    hashes identically."""
    norm = _WS_RE.sub(" ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _row_content(row: Mapping[str, Any]) -> str:
    """Extract the dedup/display content from a partition row. Accepts the
    common keys across the three native shapes the caller adapts:
    ``content`` (assembler-native / VCB-adapted) or ``content_snippet``
    (the recall stash dict)."""
    return str(row.get("content") or row.get("content_snippet") or "").strip()


def assemble(
    *,
    vcb: Optional[Sequence[Mapping[str, Any]]] = None,
    memory: Optional[Sequence[Mapping[str, Any]]] = None,
    recall: Optional[Sequence[Mapping[str, Any]]] = None,
    max_items: int = 40,
) -> list[AssembledItem]:
    """Merge the three substrate partitions into one deduped, source-tagged list.

    Each argument is a sequence of plain dict rows (the caller adapts the native
    VerifiedRecord / memory-node / recall shapes → dicts) with keys:
    ``content`` | ``content_snippet`` (required, non-empty), and optionally
    ``item_id``, ``title``, ``weight``, ``concept_ids``. A row missing content is
    skipped. Rows are visited in ``SOURCE_PRIORITY`` order; the first occurrence
    of a ``content_hash`` wins, later duplicates (any partition) are dropped.

    Returns up to ``max_items`` ``AssembledItem``s in priority-then-input order.
    Soft + total: never raises — a malformed row is skipped, not fatal.
    """
    by_source = {"vcb": vcb, "memory": memory, "recall": recall}
    seen: dict[str, AssembledItem] = {}
    order: list[str] = []

    for source in SOURCE_PRIORITY:
        rows = by_source.get(source) or []
        for row in rows:
            if len(order) >= max_items:
                return [seen[h] for h in order]
            if not isinstance(row, Mapping):
                continue
            content = _row_content(row)
            if not content:
                continue
            h = content_hash(content)
            if h in seen:
                continue  # a higher-priority partition already took this content
            item_id = str(row.get("item_id") or "").strip() or f"{source}:{h[:_HASH_ID_LEN]}"
            try:
                weight = float(row.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            concept_ids = tuple(str(c) for c in (row.get("concept_ids") or ()) if c)
            item = AssembledItem(
                item_id=item_id,
                source=source,
                content=content,
                content_hash=h,
                title=str(row.get("title") or content[:120]),
                weight=weight,
                concept_ids=concept_ids,
            )
            seen[h] = item
            order.append(h)

    return [seen[h] for h in order]
