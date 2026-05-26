"""Buffer-entity source stub for spreading-activation lookup (Phase 4 / §P4.F).

Phase 1 deferred `w_s` because the buffer-entity source didn't exist yet.
Phase 4 ships a STUB source so `composite_score` can be exercised end-to-end
+ the B.1 acceptance gate becomes measurable. The §14 `actr_buffers` table
(Phase 7) replaces this stub via a single-import swap — config + caller
shape stays.

Source per PLAN §P4.F:
  - topic_tags from the in-flight chat TX (caller-supplied)
  - last-N-turns CGN concept_ids touched (read from a CGN handle if
    available; falls back to empty silently if the handle is missing or
    its internal shape changes)
  - extra explicit concept_ids the caller wants in the buffer (escape
    hatch for `actr_procedural_skill_proposer`-style callers)

The union is dedup'd, ordered by recency (chat-topic tags first), and
capped at `max_entities`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _strip_tag_prefix(tag: str) -> str:
    """Normalize a topic-tag string into a bare concept_id. Tags arrive as
    e.g. "topic:linux_terminal" or "concept:linux_terminal:v3" or just
    "linux_terminal" — buffer entities are bare concept_ids."""
    if tag.startswith("concept:"):
        rest = tag[len("concept:"):]
        # Strip optional :v<n> suffix.
        return rest.split(":v", 1)[0]
    if tag.startswith("topic:"):
        return tag[len("topic:"):]
    return tag


class BufferStub:
    """Buffer-entity source for §P4.F spreading-activation lookup.

    `cgn_handle` is the live CGN instance (the synthesis_worker has access
    to it). If None or the handle's internal shape changes, the CGN-history
    contribution silently degrades to empty — the topic-tags + extra-concepts
    contributions still work."""

    def __init__(
        self,
        cgn_handle: Any = None,
        *,
        history_window_turns: int = 3,
        max_entities: int = 20,
    ):
        self._cgn = cgn_handle
        # CGN doesn't keep a "per turn" history; the window scales linearly
        # with how many concepts we sample per turn. ~5 concepts/turn is a
        # reasonable estimate so window=3 → ~15 CGN concepts considered.
        self._cgn_sample_size = max(1, history_window_turns * 5)
        self._max = max_entities

    def current_entities(
        self,
        topic_tags: Optional[list[str]] = None,
        extra_concepts: Optional[list[str]] = None,
    ) -> list[str]:
        """Return the buffer-entity list (bare concept_ids) for the current
        recall. Cap at max_entities; order = topic_tags first, then extra
        explicit concepts, then CGN-history fallback."""
        out: list[str] = []

        def _add(cid: str) -> bool:
            """Append `cid` if non-empty + not already present. Returns
            True if cap reached (caller should stop adding)."""
            if not cid or cid in out:
                return False
            out.append(cid)
            return len(out) >= self._max

        # 1. Chat-topic tags (most recently relevant).
        for tag in topic_tags or []:
            cid = _strip_tag_prefix(tag)
            if _add(cid):
                return out

        # 2. Caller-supplied extras (escape hatch).
        for cid in extra_concepts or []:
            if _add(cid):
                return out

        # 3. CGN-history fallback — sample most recently-touched concept_ids
        # from CGN's per-concept journey ledger. Defensive: any AttributeError
        # / KeyError / type mismatch silently degrades to no contribution.
        if self._cgn is not None:
            try:
                journeys = getattr(self._cgn, "_concept_journeys", None)
                if isinstance(journeys, dict) and journeys:
                    # Sort by last-seen timestamp (or any "recency"-like field).
                    def _recency(kv) -> float:
                        v = kv[1]
                        if not isinstance(v, dict):
                            return 0.0
                        for key in ("last_seen", "last_outcome_at",
                                    "last_touched", "updated_at"):
                            ts = v.get(key)
                            if isinstance(ts, (int, float)):
                                return float(ts)
                        return 0.0

                    sorted_concepts = sorted(
                        journeys.items(), key=_recency, reverse=True,
                    )
                    for cid, _ in sorted_concepts[:self._cgn_sample_size]:
                        if isinstance(cid, str) and _add(cid):
                            return out
            except Exception as e:
                logger.debug(
                    "[BufferStub] CGN history sampling skipped: %s", e,
                )

        return out


__all__ = ("BufferStub", "_strip_tag_prefix")
