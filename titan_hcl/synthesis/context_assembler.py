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
sovereignty **E** term (cited-substrate share) and **V** term could never
credit VCB or memory content. The assembler itemises ALL surfaced substrate so
E ("was it his") credits every own-substrate source, while the refined **V**
("was it proven") counts only the timechain-anchored ``TX_ANCHORED_SOURCES``
(recall + synthesis spine) — see ``compute_sovereignty_score``. It tags each
item at its source (the
``SurfacedItem.source`` the ``CitedUseDetector`` / ``compute_sovereignty_score``
consume), and de-duplicates by a normalised content hash so a fact surfaced by
two paths is delivered **and** counted exactly once (gate G6 — no content
twice).

De-dup collisions are resolved by ``SOURCE_PRIORITY`` = **recall-wins** (D2 /
INV-SDA-5): the synthesis-supplied ``recall`` partition (Titan's own
timechain-anchored experience) is authoritative, so on a content collision the
``recall`` item is kept and the enriching ``vcb`` / ``memory`` duplicate is
dropped — from BOTH the surfaced-item set (what gets *counted*) and the rendered
grounded block (what the LLM *sees*). Since P4 the assembler DRIVES the one
grounded prompt: ``render_grounded_block(...)`` renders the deduped items into a
single block, so display == count by construction (no separate wholesale-VCB +
line-by-line-recall blocks that can fall out of sync).

Pure + deterministic — no I/O, no LLM, never raises. The hot path imports
``assemble(...)`` + ``render_grounded_block(...)``; persistence + scoring stay
downstream (INV-SDA-11).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from titan_hcl.synthesis.cited_use import SurfacedItem

__all__ = [
    "AssembledItem",
    "assemble",
    "render_grounded_block",
    "SOURCE_PRIORITY",
    "content_hash",
]

# De-dup keep-order (D2 / INV-SDA-5): RECALL WINS. The synthesis-supplied recall
# partition is authoritative — on a content collision its item is kept and the
# enriching vcb / memory duplicate is dropped. vcb precedes memory only as a
# stable tiebreak between the two enrichers (they are disjoint by DB per §5.1,
# so a vcb↔memory content collision is not expected by construction).
SOURCE_PRIORITY = ("recall", "vcb", "memory")

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
    # Source-specific render hints carried through verbatim (chain_status,
    # block_height, timestamp for vcb; the assembler never interprets them —
    # only ``render_grounded_block`` reads them). NOT projected to SurfacedItem
    # (sovereignty counting is render-agnostic).
    meta: Mapping[str, Any] = field(default_factory=dict)

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
            _meta = row.get("meta")
            item = AssembledItem(
                item_id=item_id,
                source=source,
                content=content,
                content_hash=h,
                title=str(row.get("title") or content[:120]),
                weight=weight,
                concept_ids=concept_ids,
                meta=dict(_meta) if isinstance(_meta, Mapping) else {},
            )
            seen[h] = item
            order.append(h)

    return [seen[h] for h in order]


# ─────────────────────────────────────────────────────────────────────────
# Render — the assembler DRIVES the one grounded prompt (P4 / handoff §1.3).
# ─────────────────────────────────────────────────────────────────────────

_VCB_STATUS_TAG = {
    "CHAINED": "[CHAINED ✓]",
    "PARTIAL": "[PARTIAL]",
    "WIRED": "[WIRED]",
    "NOT_COVERED": "[unverified]",
}


def _vcb_store(item: "AssembledItem") -> str:
    """The store key a vcb item came from (for source-grouped display). The
    title carries VCB's ``db_ref`` (``"knowledge_concepts:solana"`` →
    ``knowledge_concepts``)."""
    ref = (item.title or "").strip()
    if ":" in ref:
        return ref.split(":", 1)[0]
    return ref or "inner state"


def render_grounded_block(
    items: Sequence["AssembledItem"], *, max_tokens: int = 2000
) -> str:
    """Render the deduped, source-tagged ``AssembledItem``s into ONE grounded
    markdown block for LLM injection.

    Since P4 this REPLACES the separate ``memory_context`` (VCB wholesale text)
    and ``synthesis_recall_context`` blocks: the LLM sees exactly the deduped
    item set the ``CitedUseDetector`` counts (display == count, gate G6).

    Sections, in display order:
      • **vcb**    — inner-titan state, grouped by store, each line stamped with
        its TimeChain ``chain_status`` (``[CHAINED ✓]`` …) + block height, the
        way ``VerifiedContextBuilder._assemble_text`` rendered it before P4 (no
        chat-quality regression).
      • **recall** — Titan's own verified experience (tx_hash spine), scored.
      • **memory** — consolidated legacy graph recall.

    Closes with the anti-hallucination footer. Pure + deterministic; returns
    ``""`` when there are no items (matches the pre-P4 "no block" behaviour —
    the empty-VCB honesty notice was never injected on the live path).
    """
    if not items:
        return ""

    lines: list[str] = ["### Verified Memory Recall"]
    token_est = 0.0
    _truncated = False

    def _emit(line: str) -> bool:
        """Append a line under the rolling token budget. Returns False once the
        budget is hit (caller stops)."""
        nonlocal token_est, _truncated
        token_est += len(line.split()) * 1.3
        if token_est > max_tokens:
            lines.append("- *(truncated — token budget reached)*")
            _truncated = True
            return False
        lines.append(line)
        return True

    # ── vcb — grouped by store, chain-stamped ──────────────────────────────
    vcb_items = [it for it in items if it.source == "vcb"]
    by_store: dict[str, list["AssembledItem"]] = {}
    for it in vcb_items:
        by_store.setdefault(_vcb_store(it), []).append(it)
    for store, recs in by_store.items():
        if _truncated:
            break
        lines.append(f"**{store}:**")
        for it in recs:
            meta = it.meta or {}
            status_tag = _VCB_STATUS_TAG.get(str(meta.get("chain_status") or ""), "")
            ts_str = ""
            _ts = meta.get("timestamp")
            if _ts:
                try:
                    ts_str = datetime.fromtimestamp(
                        float(_ts), tz=timezone.utc).strftime(" (%Y-%m-%d %H:%M)")
                except (OSError, ValueError, TypeError):
                    ts_str = ""
            _bh = meta.get("block_height")
            block_ref = f" Block #{_bh}" if _bh else ""
            if not _emit(f"- {it.content}{ts_str} {status_tag}{block_ref}".rstrip()):
                break
        lines.append("")

    # ── recall — Titan's own verified experience ───────────────────────────
    recall_items = [it for it in items if it.source == "recall"]
    if recall_items and not _truncated:
        lines.append("**Your own verified experience (tx_hash spine):**")
        for it in recall_items:
            if not _emit(f"- [{it.weight:.2f}] {it.content[:300]}"):
                break
        lines.append("")

    # ── memory — consolidated legacy recall ────────────────────────────────
    memory_items = [it for it in items if it.source == "memory"]
    if memory_items and not _truncated:
        lines.append("**Recalled memory:**")
        for it in memory_items:
            if not _emit(f"- {it.content[:300]}"):
                break
        lines.append("")

    lines.append("⚠ These memories are verified against your TimeChain.")
    lines.append("Only reference what is provided here. If asked about "
                 "something not in your memories, say honestly that you "
                 "don't recall.")
    return "\n".join(lines) + "\n\n"
