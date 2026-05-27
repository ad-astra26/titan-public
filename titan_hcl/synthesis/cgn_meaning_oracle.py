"""Phase 6 — `CGNMeaningOracle` (§P6.H — graduates the P4 stub).

Concrete `MeaningOraclePlug` per SPEC §25.3 + §25.5 + INV-Syn-1 / INV-1:

  > CGN remains the sole grounding authority; the synthesis engine
  > REQUESTS groundings, never creates them.

This module **does not** replace ``CGNRegistrationBridge`` (the P4
spine-registry bridge); it sits alongside it as the
``MeaningOraclePlug`` surface the OracleRouter + synthesis_worker
consume. ``CGNRegistrationBridge.ensure_grounded()`` returns a stub
``Grounding(grounded=False, note="phase4_stub")``; ``CGNMeaningOracle``
graduates that to:

- ``meaning_of(concept)`` — reads the four spine strands
  (declarative / procedural / episodic / felt) from the Kuzu Concept
  spine via an injected ``concept_reader`` callable. Returns a
  ``MeaningStrand`` whose strand lists carry concept anchor TX hashes.
- ``ground(concept, felt)`` — delegates to the live CGN process via an
  injected ``cgn_grounder`` callable (the synthesis_worker wires this
  to CGN's `ground_concept` bus surface). CGN authors the grounding;
  this oracle never creates one.

Both injections soft-fail to a stub when CGN is degraded — keeping
INV-Syn-1 honored (we never invent a grounding) while preserving
production stability (a CGN restart should NOT cascade-crash the
synthesis_worker).

The plug exposes ``oracle_id="cgn"`` for the Observatory router
listing (P6.K). Free cost class — CGN's compute is amortized in the
inner-trinity ticks; no per-call SOL cost.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from titan_hcl.synthesis.plugs import (
    ConceptRef,
    FeltContext,
    Grounding,
    MeaningStrand,
)

logger = logging.getLogger(__name__)


# ``concept_reader(concept_id, version) -> dict | None`` —
# Returns a dict carrying the spine concept's strand arrays:
#   {
#     "declarative_anchors": [tx_hash, ...],
#     "procedural_anchors":  [tx_hash, ...],
#     "episodic_anchors":    [tx_hash, ...],
#     "felt_anchors":        [tx_hash, ...],
#   }
# Missing concept → None. Synthesis_worker wires this to
# ``ConceptStore.read_spine_strands(...)`` (or a Kuzu read wrapper).
ConceptReader = Callable[[str, int], Optional[dict]]

# ``cgn_grounder(concept_id, version, valence, arousal, neuromods) -> dict``
# Returns:
#   {"grounding_id": <str>, "strength": <float [0..1]>, "ts": <float>}
# OR raises / returns None when CGN is degraded. Synthesis_worker wires
# this to CGN's bus surface.
CGNGrounder = Callable[[str, int, float, float, dict], Optional[dict]]


def _default_concept_reader(concept_id: str, version: int) -> Optional[dict]:
    """No-op reader — synthesis_worker MUST inject the real one at boot.
    Surfaces clearly in tests + logs (returns None → empty strand).
    """
    logger.debug(
        "[cgn_meaning_oracle] default concept_reader called for %s:v%d — "
        "synthesis_worker should inject a real reader", concept_id, version,
    )
    return None


def _default_cgn_grounder(
    concept_id: str, version: int, valence: float, arousal: float, neuromods: dict,
) -> Optional[dict]:
    """No-op grounder — synthesis_worker MUST inject the real one at boot."""
    logger.debug(
        "[cgn_meaning_oracle] default cgn_grounder called — synthesis_worker "
        "should inject a real grounder; returning degraded stub",
    )
    return None


class CGNMeaningOracle:
    """`MeaningOraclePlug` graduating `cgn_bridge.py`'s P4 stub.

    INV-Syn-1 / INV-1: CGN is the sole grounding authority. This class
    requests groundings; it never invents them. On CGN degraded the
    grounder returns None and we surface a degraded Grounding
    (``strength=0.0``, ``grounding_id=""``) — the consumer treats it
    the same as the P4 stub for groundedness-formula purposes (felt
    coverage = 0).
    """

    oracle_id: str = "cgn"
    cost_class: str = "free"

    def __init__(
        self,
        *,
        concept_reader: ConceptReader = _default_concept_reader,
        cgn_grounder: CGNGrounder = _default_cgn_grounder,
        now_fn: Callable[[], float] = time.time,
    ):
        self._concept_reader = concept_reader
        self._cgn_grounder = cgn_grounder
        self._now_fn = now_fn

    # ── meaning_of ──────────────────────────────────────────────────────

    def meaning_of(self, concept: ConceptRef) -> MeaningStrand:
        """Read the four spine strands for a concept.

        Missing concept or concept_reader degraded → returns a
        MeaningStrand with all four strand lists empty (consumer
        treats this as "no meaning yet" — same shape as a freshly
        materialized concept).
        """
        try:
            row = self._concept_reader(concept.concept_id, concept.version)
        except Exception:
            logger.exception(
                "[cgn_meaning_oracle] concept_reader raised for %s:v%d",
                concept.concept_id, concept.version,
            )
            row = None

        if not isinstance(row, dict):
            return MeaningStrand(concept=concept)

        return MeaningStrand(
            concept=concept,
            declarative_anchors=list(row.get("declarative_anchors", []) or []),
            procedural_anchors=list(row.get("procedural_anchors", []) or []),
            episodic_anchors=list(row.get("episodic_anchors", []) or []),
            felt_anchors=list(row.get("felt_anchors", []) or []),
        )

    # ── ground ──────────────────────────────────────────────────────────

    def ground(self, concept: ConceptRef, felt: FeltContext) -> Grounding:
        """Request CGN to ground this concept in the felt context.

        CGN authors the Grounding (INV-Syn-1). On CGN degraded / grounder
        raises / grounder returns None, we return a degraded Grounding
        with ``strength=0.0`` so the caller sees "no felt strand attached
        yet" — never a fake grounding.
        """
        valence = float(felt.valence)
        arousal = float(felt.arousal)
        neuromods = dict(felt.neuromods or {})

        result: Optional[dict] = None
        try:
            result = self._cgn_grounder(
                concept.concept_id, concept.version, valence, arousal, neuromods,
            )
        except Exception:
            logger.exception(
                "[cgn_meaning_oracle] cgn_grounder raised for %s:v%d",
                concept.concept_id, concept.version,
            )
            result = None

        if not isinstance(result, dict):
            # CGN degraded — surface a degraded Grounding (strength=0)
            # so consumer knows the felt strand is unattached. NEVER
            # invent a real grounding — INV-Syn-1 / INV-1.
            return Grounding(
                concept=concept,
                grounding_id="",
                strength=0.0,
                ts=self._now_fn(),
            )

        grounding_id = str(result.get("grounding_id") or "")
        # Authored by CGN — if CGN didn't supply an id, we don't make one
        # up (the consumer can detect the missing id and re-request later).
        strength = float(result.get("strength", 0.0) or 0.0)
        # Clamp strength to [0, 1] defensively — out-of-range CGN reply
        # likely means schema drift; treat as degraded grounding.
        if not (0.0 <= strength <= 1.0):
            logger.warning(
                "[cgn_meaning_oracle] CGN returned strength=%s outside [0,1]; "
                "clamping to 0.0 (degraded)", strength,
            )
            strength = 0.0
        ts = float(result.get("ts", self._now_fn()) or self._now_fn())

        return Grounding(
            concept=concept,
            grounding_id=grounding_id,
            strength=strength,
            ts=ts,
        )


__all__ = (
    "CGNMeaningOracle",
    "ConceptReader",
    "CGNGrounder",
)
