"""Cross-consumer composition layer (Phase B / RFP_cgn_enhancements §9.2).

The output of a Level-B abstraction chain: a higher-order pattern synthesized
ACROSS consumers (e.g. "boundary" abstracted from language:COLD + reasoning:edge
+ social:distance). Per CGN invariants §11.3 + §12.2, cross-consumer compositions
are built ABOVE CGN by meta-reasoning and are NOT grounded as CGN concept rows —
so they live here, in a meta-reasoning-owned store, referenced by RECALL.wisdom
but never written back into the CGN matrix.

Pure-logic class (no bus, no worker imports) — JSON-persisted, bounded, testable.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger("titan.cross_consumer_composition")

DEFAULT_SAVE_PATH = "data/meta_cgn/cross_consumer_compositions.json"


@dataclass
class CrossConsumerComposition:
    """One higher-order binding synthesized across consumers."""
    binding_id: str
    member_concepts: list = field(default_factory=list)   # concepts abstracted
    member_consumers: list = field(default_factory=list)  # contributing consumers
    abstraction_label: str = ""                           # the pattern name
    confidence: float = 0.3                               # grows with corroboration
    lineage: list = field(default_factory=list)           # source chain_ids / bindings
    n_reinforced: int = 0
    ts_created: float = 0.0
    ts_updated: float = 0.0


def _binding_id(member_concepts: list) -> str:
    """Deterministic id from the concept SET (re-abstracting the same set updates,
    not duplicates)."""
    key = "|".join(sorted(str(c).lower() for c in member_concepts if c))
    return "ccc_" + hashlib.sha1(key.encode()).hexdigest()[:16]


class CrossConsumerCompositionStore:
    """Bounded, persisted store of cross-consumer compositions."""

    def __init__(self, save_path: str = DEFAULT_SAVE_PATH, max_size: int = 2000):
        self._by_id: dict = {}            # binding_id → CrossConsumerComposition
        self._save_path = save_path
        self._max_size = int(max_size)
        self._load()

    # ── Mutation ────────────────────────────────────────────────────────
    def add(self, member_concepts: list, member_consumers: list,
            abstraction_label: str, confidence: float = 0.3,
            lineage: Optional[list] = None, now: Optional[float] = None) -> str:
        """Add or reinforce a composition. Idempotent on the concept set:
        re-adding the same set bumps confidence + n_reinforced rather than
        duplicating. Returns the binding_id."""
        if not member_concepts:
            return ""
        _now = float(now) if now is not None else time.time()
        bid = _binding_id(member_concepts)
        existing = self._by_id.get(bid)
        if existing is not None:
            # Reinforce — bounded confidence climb, merge consumers/lineage.
            existing.n_reinforced += 1
            existing.confidence = min(1.0, existing.confidence + 0.1 * (1.0 - existing.confidence))
            existing.member_consumers = sorted(
                set(existing.member_consumers) | set(member_consumers or []))
            for src in (lineage or []):
                if src not in existing.lineage:
                    existing.lineage.append(src)
            existing.lineage = existing.lineage[-20:]
            existing.ts_updated = _now
            if abstraction_label and not existing.abstraction_label:
                existing.abstraction_label = abstraction_label
            return bid
        comp = CrossConsumerComposition(
            binding_id=bid,
            member_concepts=sorted(set(str(c) for c in member_concepts if c)),
            member_consumers=sorted(set(member_consumers or [])),
            abstraction_label=str(abstraction_label or ""),
            confidence=max(0.0, min(1.0, float(confidence))),
            lineage=list(lineage or [])[-20:],
            n_reinforced=0,
            ts_created=_now,
            ts_updated=_now,
        )
        self._by_id[bid] = comp
        self._evict_if_needed()
        return bid

    def _evict_if_needed(self) -> None:
        if len(self._by_id) <= self._max_size:
            return
        # Evict the oldest-updated, lowest-confidence first.
        victims = sorted(self._by_id.values(),
                         key=lambda c: (c.confidence, c.ts_updated))
        for v in victims[: len(self._by_id) - self._max_size]:
            self._by_id.pop(v.binding_id, None)

    # ── Query (RECALL.wisdom reads these) ───────────────────────────────
    def get(self, binding_id: str) -> Optional[CrossConsumerComposition]:
        return self._by_id.get(binding_id)

    def all(self) -> list:
        return list(self._by_id.values())

    def recent(self, k: int = 5) -> list:
        """Top-k most recently updated — the abstraction wisdom RECALL pulls."""
        return sorted(self._by_id.values(),
                      key=lambda c: c.ts_updated, reverse=True)[:max(0, int(k))]

    def count(self) -> int:
        return len(self._by_id)

    # ── Persistence (caller drives the non-blocking save tick) ──────────
    def to_dict(self) -> dict:
        return {"compositions": [asdict(c) for c in self._by_id.values()]}

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
            tmp = self._save_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self._save_path)   # atomic
            return True
        except Exception as e:
            logger.warning("[CrossConsumerComposition] save failed: %s", e)
            return False

    def _load(self) -> None:
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            for cd in data.get("compositions", []):
                comp = CrossConsumerComposition(**cd)
                self._by_id[comp.binding_id] = comp
        except Exception as e:
            logger.warning("[CrossConsumerComposition] load failed (fresh): %s", e)
