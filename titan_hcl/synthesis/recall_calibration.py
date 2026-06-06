"""Self-calibrating recall floors — gibberish-baseline (RFP_synthesis_decision_authority P2).

The execution-mode decision routes on the EngineRecall top cosine. But the
embedder's cosine band is COMPRESSED + HIGH — the D5 live probe found
known-topic 0.81-0.84 > **gibberish 0.71-0.74** > off-spine real 0.68, a ~0.06
margin. A fixed floor (the legacy 0.65) sits BELOW even the gibberish noise
floor, so it routes *everything* — including nonsense — to Sovereign (the exact
relevance-blind bug INV-SDA-2 targets).

This calibrates the floors against the embedder's ACTUAL noise floor. A known-
gibberish prompt's top cosine = the `gibberish_ceiling` — the recall an irrelevant
query gets by chance against the spine. Real recall sits ABOVE it:

    known_floor   = gibberish_ceiling + margin   (clearly above noise → Sovereign)
    present_floor = gibberish_ceiling             (at/above noise → Collaborative;
                                                    below → research / honest-IDK)

The ceiling is an EMA over periodic gibberish probes, so the floors TRACK the
live embedder/spine (re-index, model swap, spine growth) with zero hardcoded
tuning — emergence over determinism. Pure + deterministic: no I/O, no recall, no
LLM; the caller supplies the gibberish probe's measured top cosine.
"""
from __future__ import annotations

__all__ = ["GibberishBaseline", "GIBBERISH_PROMPTS"]

# Fixed nonsense prompts — pronounceable non-words so the embedder produces a
# normal (not degenerate) vector, measuring the genuine "irrelevant query" noise
# floor against the spine. Averaged over a few to smooth single-prompt quirks.
GIBBERISH_PROMPTS = (
    "zorblax quenfium thractle wobbergquist",
    "flimmerdax plonquibble sernathy grumvox",
    "vextr- quommadge fnordleby skritchwane",
)


def _clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class GibberishBaseline:
    """EMA of the gibberish noise-floor cosine → self-calibrating decision floors.

    Thread-unsafe by design (single owner, the agno PreHook); the caller probes
    periodically and reads `floors()` per turn (cheap, no recall)."""

    def __init__(
        self,
        *,
        initial_ceiling: float = 0.74,   # D5 gibberish observation (cold-start seed)
        margin: float = 0.04,            # known-recall must clear the noise floor by this
        ema_alpha: float = 0.3,          # EMA weight on each new probe (0..1)
    ) -> None:
        self._ceiling = _clamp01(initial_ceiling)
        self._margin = max(0.0, float(margin))
        self._alpha = _clamp01(ema_alpha)
        self._samples = 0

    def update(self, gibberish_top_cosine: float) -> None:
        """Fold one gibberish-probe top cosine into the EMA ceiling. The first
        real sample REPLACES the cold-start seed (so we converge to the live
        embedder fast); subsequent samples EMA-smooth."""
        c = _clamp01(gibberish_top_cosine)
        if self._samples == 0:
            self._ceiling = c
        else:
            self._ceiling = (1.0 - self._alpha) * self._ceiling + self._alpha * c
        self._samples += 1

    def floors(self) -> tuple[float, float]:
        """Return `(known_floor, present_floor)` for the grounded router.

        known = ceiling + margin (clear-of-noise → Sovereign); present = ceiling
        (at-the-noise-floor → Collaborative). Both clamped to [0,1]."""
        known = _clamp01(self._ceiling + self._margin)
        present = _clamp01(self._ceiling)
        return known, present

    @property
    def ceiling(self) -> float:
        return self._ceiling

    @property
    def samples(self) -> int:
        return self._samples

    def snapshot(self) -> dict:
        """Observable state (for the journal / telemetry)."""
        known, present = self.floors()
        return {
            "gibberish_ceiling": round(self._ceiling, 4),
            "known_floor": round(known, 4),
            "present_floor": round(present, 4),
            "margin": round(self._margin, 4),
            "samples": self._samples,
        }
