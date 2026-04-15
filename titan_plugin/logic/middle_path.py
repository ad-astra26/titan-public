"""
titan_plugin/logic/middle_path.py — Middle Path coherence and loss functions.

The Divine Center is [0.5, 0.5, ..., 0.5] in each tensor space.

Coherence (T1 upgrade): measures internal alignment of a tensor.
  1.0 = all dimensions aligned (coherent), 0.0 = maximally incoherent.
  This is the PRIMARY metric used by sphere clocks, observables, and topology.

Legacy layer_loss: L2 distance from center. Kept for backward compatibility
  (FILTER_DOWN reward, FOCUS PID). Expressible as 1.0 - coherence only for
  uniform tensors, so both are retained.

Used by:
  - Spirit: overall health metric published in SPIRIT_STATE
  - FILTER_DOWN: reward signal (lower loss = better outcome)
  - FOCUS: PID error term
  - Sphere clocks: coherence drives contraction velocity (T1)
  - Observables: coherence is observable #1 (T1)
"""
import math
from typing import Sequence


# Divine Center — perfect equilibrium
CENTER = 0.5

# Per-layer weights (Body, Mind, Spirit)
# Body and Mind weighted equally; Spirit slightly higher because it
# aggregates the other two and adds consciousness metrics.
DEFAULT_WEIGHTS = (1.0, 1.0, 1.2)

# Max possible variance for values in [0, 1]: half at 0, half at 1 → 0.25
_MAX_VARIANCE = 0.25


def layer_coherence(tensor: Sequence[float]) -> float:
    """
    Coherence of a single layer's 5DT tensor.

    Measures how aligned the dimensions are with each other (low variance).
    Returns 1.0 when all dimensions have the same value (perfectly coherent),
    0.0 when variance is maximal (half at 0, half at 1).

    Unlike layer_loss, coherence is position-independent: [0.1, 0.1, 0.1]
    and [0.9, 0.9, 0.9] are both perfectly coherent.
    """
    if len(tensor) < 2:
        return 1.0
    mean_val = sum(tensor) / len(tensor)
    variance = sum((v - mean_val) ** 2 for v in tensor) / len(tensor)
    return max(0.0, 1.0 - variance / _MAX_VARIANCE)


def layer_loss(tensor: Sequence[float]) -> float:
    """
    L2 distance from center for a single layer's 5DT tensor.

    Returns a value >= 0.0 where 0.0 = perfect equilibrium.
    Max theoretical = sqrt(5 * 0.25) ≈ 1.118 for a 5-dim tensor at extremes.

    Note: This is NOT simply 1.0 - coherence. Coherence measures internal
    alignment (variance), loss measures distance from center (L2). Both are
    retained because FOCUS PID needs directional error, not just alignment.
    """
    return math.sqrt(sum((v - CENTER) ** 2 for v in tensor))


def middle_path_loss(
    body: Sequence[float],
    mind: Sequence[float],
    spirit: Sequence[float],
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Combined weighted Middle Path loss across the Trinity.

    Returns a normalized value (0.0 = perfect equilibrium, 1.0 = max distress).
    The normalization uses the theoretical maximum for 5-dim tensors.
    """
    wb, wm, ws = weights
    total_weight = wb + wm + ws

    body_l = layer_loss(body)
    mind_l = layer_loss(mind)
    spirit_l = layer_loss(spirit)

    # Theoretical max L2 for 5-dim tensor at all-0 or all-1
    max_l2 = math.sqrt(5 * 0.25)  # ~1.118

    # Weighted average, normalized to 0-1
    raw = (wb * body_l + wm * mind_l + ws * spirit_l) / total_weight
    return min(1.0, raw / max_l2)


def per_dim_loss(tensor: Sequence[float]) -> list[float]:
    """Per-dimension squared distance from center. Used by FOCUS PID."""
    return [(v - CENTER) ** 2 for v in tensor]
