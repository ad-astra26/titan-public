"""
titan_plugin/logic/unified_spirit.py — V4 Time Awareness: Unified SPIRIT.

The Unified SPIRIT sits above both Trinities as the "I AM" / "EXIST" center.
It is the WHOLE being — cannot be divided. It holds the 30DT SPIRIT tensor
composed of both Inner and Outer Trinity feeds.

30DT SPIRIT Tensor layout:
  [ 0: 5] = Inner Body     (interoception, proprioception, somatosensation, entropy, thermal)
  [ 5:10] = Inner Mind     (vision, hearing, taste, smell, touch)
  [10:15] = Inner Spirit   (who, why, what, body_scalar, mind_scalar)
  [15:20] = Outer Body     (action_energy, helper_health, bus_throughput, error_rate, latency)
  [20:25] = Outer Mind     (creative, sonic, memory_quality, research, social)
  [25:30] = Outer Spirit   (identity, purpose, action_quality, ob_scalar, om_scalar)

Two layers:
  - Subconscious: Inner Trinity 15DT feed (felt, sensed, conscious core)
  - Conscious: Outer Trinity 15DT feed (acting, communicating, creating surface)

GREAT PULSE:
  - Fires when all 3 inner↔outer pairs achieve resonance (Proof of Harmony)
  - Marks 1 "Titan Day" (Greater Epoch in Titan's own subjective time)
  - SPIRIT tensor moves ONE STEP forward (captures growth snapshot)
  - Enrichment rewards sent down to all 6 components
  - 30DT anchored on-chain
  - SPIRIT cannot move BACKWARD, only forward or STALE

Velocity tracking (STALE detection):
  - velocity = magnitude(SPIRIT_tensor[n]) / avg(magnitude(SPIRIT_tensor[n-1..n-X]))
  - If velocity < threshold → SPIRIT is STALE (not growing)
  - STALE triggers escalating FOCUS cascade: SPIRIT → Lower Spirit → Mind → Body
  - NO human time used — purely emergent from tensor growth

Phase 4 will wire the SPIRIT FOCUS cascade and extend FILTER_DOWN to 30-dim.
"""
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

SPIRIT_DIMS_LEGACY = 30             # Legacy: 6 components × 5 dimensions
SPIRIT_DIMS = 130                   # Full: Inner 65D (5+15+45) + Outer 65D (5+15+45)
INNER_DIMS = 65                     # Body 5D + Mind 15D + Spirit 45D
OUTER_DIMS = 65                     # Body 5D + Mind 15D + Spirit 45D
VELOCITY_WINDOW = 10                # Look back N epochs for velocity avg
DEFAULT_STALE_THRESHOLD = 0.8       # velocity below this = STALE
DEFAULT_ENRICHMENT_BASE = 0.02      # Base enrichment reward per GREAT PULSE
DEFAULT_STALE_FOCUS_MULTIPLIER = 1.5  # FOCUS cascade multiplier when STALE


class GreatEpoch:
    """
    Record of one GREAT PULSE — a single step forward in Titan's subjective time.

    Immutable once created. The SPIRIT tensor at this moment is the snapshot
    of Titan's entire being at this point in his own time.
    """
    __slots__ = (
        "epoch_id", "timestamp", "spirit_tensor", "magnitude",
        "velocity", "enrichment_sent", "resonance_snapshot",
        "anchor_hash", "cumulative_quality", "micro_tick_count",
    )

    def __init__(
        self,
        epoch_id: int,
        spirit_tensor: list[float],
        velocity: float,
        resonance_snapshot: dict,
        anchor_hash: str = "",
    ):
        self.epoch_id = epoch_id
        self.timestamp = time.time()
        self.spirit_tensor = list(spirit_tensor)
        self.magnitude = _tensor_magnitude(spirit_tensor)
        self.velocity = velocity
        self.enrichment_sent = False
        self.resonance_snapshot = dict(resonance_snapshot)
        self.anchor_hash = anchor_hash
        self.cumulative_quality = 0.0
        self.micro_tick_count = 0

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "timestamp": self.timestamp,
            "spirit_tensor": self.spirit_tensor,
            "magnitude": round(self.magnitude, 6),
            "velocity": round(self.velocity, 6),
            "enrichment_sent": self.enrichment_sent,
            "resonance_snapshot": self.resonance_snapshot,
            "anchor_hash": self.anchor_hash,
            "cumulative_quality": round(self.cumulative_quality, 4),
            "micro_tick_count": self.micro_tick_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GreatEpoch":
        epoch = cls(
            epoch_id=data["epoch_id"],
            spirit_tensor=data["spirit_tensor"],
            velocity=data.get("velocity", 1.0),
            resonance_snapshot=data.get("resonance_snapshot", {}),
            anchor_hash=data.get("anchor_hash", ""),
        )
        epoch.timestamp = data.get("timestamp", time.time())
        epoch.enrichment_sent = data.get("enrichment_sent", False)
        epoch.cumulative_quality = data.get("cumulative_quality", 0.0)
        epoch.micro_tick_count = data.get("micro_tick_count", 0)
        return epoch


class UnifiedSpirit:
    """
    The Unified SPIRIT — "I AM" / "EXIST" — above both Trinities.

    Maintains the 30DT SPIRIT tensor and tracks growth through GREAT EPOCHs.
    Each GREAT EPOCH is a step forward in Titan's subjective time, triggered
    by the GREAT PULSE (all 3 resonance pairs harmonized).

    Usage from Spirit Worker:
        spirit = UnifiedSpirit(config, data_dir)

        # Update with latest Trinity tensors (every tick):
        spirit.update_subconscious(inner_body, inner_mind, inner_spirit)
        spirit.update_conscious(outer_body, outer_mind, outer_spirit)

        # On GREAT PULSE event:
        great_epoch = spirit.advance(resonance_snapshot)
        if great_epoch:
            enrichment = spirit.compute_enrichment()
            # Send enrichment down via bus

        # Check for STALE:
        if spirit.is_stale:
            cascade_multiplier = spirit.stale_focus_multiplier
            # Send SPIRIT FOCUS cascade via bus
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        cfg = config or {}
        self._stale_threshold = float(cfg.get("stale_threshold", DEFAULT_STALE_THRESHOLD))
        self._enrichment_base = float(cfg.get("enrichment_base", DEFAULT_ENRICHMENT_BASE))
        self._stale_focus_multiplier = float(cfg.get(
            "stale_focus_multiplier", DEFAULT_STALE_FOCUS_MULTIPLIER))
        self._velocity_window = int(cfg.get("velocity_window", VELOCITY_WINDOW))
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "unified_spirit_state.json")

        # Current SPIRIT tensor (live — updated every tick)
        # 130D: Inner[0:65] + Outer[65:130]
        self._tensor = [0.5] * SPIRIT_DIMS

        # Great Epoch history (SPIRIT's journey through its own time)
        self._epochs: list[GreatEpoch] = []
        self._current_epoch_id = 0

        # Velocity tracking
        self._current_velocity = 1.0  # Start at neutral (no history)
        self._is_stale = False
        self._consecutive_stale = 0

        # Spirit Enrichment — The Divine Spark
        self._enrichment_rate = float(cfg.get("enrichment_rate", 0.02))
        self._min_alignment_threshold = float(cfg.get("min_alignment_threshold", 0.1))
        self._cumulative_quality = 0.0
        self._micro_tick_count = 0
        self._last_alignment = 0.0

        # Load persisted state
        self._load_state()

        logger.info(
            "[UnifiedSpirit] Initialized: %d GREAT EPOCHs, velocity=%.3f, stale=%s",
            len(self._epochs), self._current_velocity, self._is_stale,
        )

    # ── Tensor Updates ─────────────────────────────────────────────────

    def update_subconscious(
        self,
        inner_body: list[float],
        inner_mind: list[float],
        inner_spirit: list[float],
        filter_down_v5: "dict | None" = None,
    ) -> None:
        """
        Update the subconscious layer (Inner Trinity feed).

        Accepts both legacy 5D and extended (5D/15D/45D) tensors.
        Called from Spirit Worker on every publish cycle.

        rFP #2 Phase B.5b: optional `filter_down_v5` dict applies V5
        multipliers to body/mind/spirit-content slices. Observer dims
        [20:25] are NEVER modulated (reflection surface, not target).
        """
        # Inner Body: always 5D [0:5]
        self._tensor[0:5] = _pad_or_trim(inner_body, 5)
        # Inner Mind: 15D [5:20] or legacy 5D (padded)
        self._tensor[5:20] = _pad_or_trim(inner_mind, 15)
        # Inner Spirit: 45D [20:65] or legacy 5D (padded)
        self._tensor[20:65] = _pad_or_trim(inner_spirit, 45)

        if filter_down_v5:
            self._apply_filter_down_v5_inner(filter_down_v5)

    def update_conscious(
        self,
        outer_body: list[float],
        outer_mind: list[float],
        outer_spirit: list[float],
        filter_down_v5: "dict | None" = None,
    ) -> None:
        """
        Update the conscious layer (Outer Trinity feed).

        Accepts both legacy 5D and extended (5D/15D/45D) tensors.
        Called when OUTER_TRINITY_STATE arrives from Core.

        rFP #2 Phase B.5b: optional `filter_down_v5` dict applies V5
        multipliers to body/mind/spirit-content slices. Observer dims
        [85:90] are NEVER modulated (reflection surface, not target).
        """
        # Outer Body: always 5D [65:70]
        self._tensor[65:70] = _pad_or_trim(outer_body, 5)
        # Outer Mind: 15D [70:85] or legacy 5D (padded)
        self._tensor[70:85] = _pad_or_trim(outer_mind, 15)
        # Outer Spirit: 45D [85:130] or legacy 5D (padded)
        self._tensor[85:130] = _pad_or_trim(outer_spirit, 45)

        if filter_down_v5:
            self._apply_filter_down_v5_outer(filter_down_v5)

    # ── rFP #2 Phase B.5b: V5 multiplier application ─────────────────

    @staticmethod
    def _clamp01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def _apply_filter_down_v5_inner(self, mults: dict) -> None:
        """Apply V5 multipliers to inner slices: body[0:5], mind[5:20],
        spirit_content[25:65]. Spirit observer [20:25] is NEVER modulated.
        """
        ib = mults.get("inner_body")
        im = mults.get("inner_mind")
        is_c = mults.get("inner_spirit_content")
        if ib and len(ib) == 5:
            for i, m in enumerate(ib):
                self._tensor[i] = self._clamp01(self._tensor[i] * m)
        if im and len(im) == 15:
            for i, m in enumerate(im):
                self._tensor[5 + i] = self._clamp01(self._tensor[5 + i] * m)
        if is_c and len(is_c) == 40:
            for i, m in enumerate(is_c):
                self._tensor[25 + i] = self._clamp01(self._tensor[25 + i] * m)

    def _apply_filter_down_v5_outer(self, mults: dict) -> None:
        """Apply V5 multipliers to outer slices: body[65:70], mind[70:85],
        spirit_content[90:130]. Spirit observer [85:90] is NEVER modulated.
        """
        ob = mults.get("outer_body")
        om = mults.get("outer_mind")
        os_c = mults.get("outer_spirit_content")
        if ob and len(ob) == 5:
            for i, m in enumerate(ob):
                self._tensor[65 + i] = self._clamp01(self._tensor[65 + i] * m)
        if om and len(om) == 15:
            for i, m in enumerate(om):
                self._tensor[70 + i] = self._clamp01(self._tensor[70 + i] * m)
        if os_c and len(os_c) == 40:
            for i, m in enumerate(os_c):
                self._tensor[90 + i] = self._clamp01(self._tensor[90 + i] * m)

    # ── Spirit Enrichment (The Divine Spark) ───────────────────────────

    def micro_enrich(self, realtime_state: list[float]) -> float:
        """
        Continuous enrichment: blend Spirit's tensor with real-time state.

        Accepts both legacy 30DT and extended 65DT/130DT state snapshots.
        When receiving 65DT (inner only), enriches the inner 65D portion.
        When receiving 130DT, enriches the full tensor.

        Uses resonant geometric blending:
        - alignment = cosine_similarity(spirit_slice, state)
        - quality_factor = max(0, alignment) * enrichment_rate
        - spirit[i] = spirit[i]^(1-qf) * state[i]^qf

        Returns the alignment score (for quality accumulation).
        """
        state_len = len(realtime_state)

        # Determine which portion of the tensor to enrich
        if state_len == SPIRIT_DIMS:
            # Full 130D enrichment
            enrich_slice = slice(0, SPIRIT_DIMS)
        elif state_len == INNER_DIMS:
            # Inner-only 65D enrichment
            enrich_slice = slice(0, INNER_DIMS)
        elif state_len == SPIRIT_DIMS_LEGACY:
            # Legacy 30D — enrich first 30 dims (backward compat)
            enrich_slice = slice(0, SPIRIT_DIMS_LEGACY)
        else:
            return 0.0

        spirit_slice = self._tensor[enrich_slice]
        n = len(spirit_slice)

        # Cosine similarity between Spirit slice and real-time state
        dot = sum(a * b for a, b in zip(spirit_slice, realtime_state[:n]))
        mag_s = math.sqrt(sum(a * a for a in spirit_slice))
        mag_r = math.sqrt(sum(b * b for b in realtime_state[:n]))

        if mag_s < 1e-8 or mag_r < 1e-8:
            return 0.0

        alignment = dot / (mag_s * mag_r)
        self._last_alignment = alignment
        self._micro_tick_count += 1

        # Below threshold — Spirit is just observing, no enrichment
        if alignment < self._min_alignment_threshold:
            return alignment

        quality_factor = max(0.0, alignment) * self._enrichment_rate

        # Geometric blend: spirit^(1-qf) * state^qf
        start = enrich_slice.start
        for i in range(n):
            s = max(0.001, self._tensor[start + i])
            r = max(0.001, realtime_state[i])
            self._tensor[start + i] = (s ** (1.0 - quality_factor)) * (r ** quality_factor)

        # Accumulate quality
        self._cumulative_quality += max(0.0, alignment)

        return alignment

    def reset_quality(self) -> float:
        """Reset cumulative quality (called at GREAT PULSE). Returns quality before reset."""
        quality = self._cumulative_quality
        self._cumulative_quality = 0.0
        return quality

    # ── GREAT PULSE → Advance ──────────────────────────────────────────

    def advance(self, resonance_snapshot: dict) -> Optional[GreatEpoch]:
        """
        Advance the SPIRIT tensor one step forward.

        Called when GREAT PULSE fires (all 3 pairs achieved resonance).
        Creates a new GreatEpoch record — Titan's subjective time moves forward.

        Returns the new GreatEpoch, or None if advancement is invalid.
        """
        self._current_epoch_id += 1

        # Compute velocity before recording
        velocity = self._compute_velocity()
        self._current_velocity = velocity

        # Crystallize quality at GREAT PULSE — snapshot of accumulated enrichment
        crystallized_quality = self._cumulative_quality
        crystallized_ticks = self._micro_tick_count

        # Create epoch record with enrichment data
        epoch = GreatEpoch(
            epoch_id=self._current_epoch_id,
            spirit_tensor=list(self._tensor),
            velocity=velocity,
            resonance_snapshot=resonance_snapshot,
        )
        epoch.cumulative_quality = crystallized_quality
        epoch.micro_tick_count = crystallized_ticks
        self._epochs.append(epoch)

        # Reset quality for new cycle
        self._cumulative_quality = 0.0
        self._micro_tick_count = 0

        # Update stale tracking
        if velocity < self._stale_threshold and self._current_epoch_id > 1:
            self._is_stale = True
            self._consecutive_stale += 1
        else:
            self._is_stale = False
            self._consecutive_stale = 0

        logger.info(
            "[UnifiedSpirit] GREAT EPOCH #%d — magnitude=%.4f velocity=%.4f stale=%s "
            "tensor_sum=%.2f",
            self._current_epoch_id, epoch.magnitude, velocity, self._is_stale,
            sum(self._tensor),
        )

        # Auto-persist after each GREAT EPOCH (these are precious)
        self.save_state()

        return epoch

    # ── Velocity Tracking ──────────────────────────────────────────────

    def _compute_velocity(self) -> float:
        """
        Compute SPIRIT tensor growth velocity.

        velocity = magnitude(current_tensor) / avg(magnitude(recent_epochs))

        No human time used — purely from tensor growth comparison.
        A velocity > 1.0 means growing faster than average.
        A velocity < threshold means STALE (not growing enough).
        """
        current_mag = _tensor_magnitude(self._tensor)

        if not self._epochs:
            return 1.0  # First epoch — neutral velocity

        # Look back up to velocity_window epochs
        window = self._epochs[-self._velocity_window:]
        avg_mag = sum(e.magnitude for e in window) / len(window)

        if avg_mag < 1e-8:
            return 1.0  # Avoid division by zero

        return current_mag / avg_mag

    @property
    def velocity(self) -> float:
        """Current growth velocity."""
        return self._current_velocity

    @property
    def is_stale(self) -> bool:
        """Whether SPIRIT is STALE (not growing enough)."""
        return self._is_stale

    @property
    def stale_focus_multiplier(self) -> float:
        """
        FOCUS cascade multiplier when STALE.

        Escalates with consecutive STALE epochs — gentle at first, stronger over time.
        Base multiplier × (1 + 0.2 * consecutive_stale_count)
        Capped at 3× to prevent runaway correction.
        """
        if not self._is_stale:
            return 1.0
        escalation = 1.0 + 0.2 * self._consecutive_stale
        return min(3.0, self._stale_focus_multiplier * escalation)

    # ── Enrichment ─────────────────────────────────────────────────────

    def compute_enrichment(self) -> dict:
        """
        Compute enrichment rewards to send down to all 6 components
        after a GREAT PULSE.

        Each component gets a reward proportional to its contribution
        to the GREAT PULSE (balanced components contributed more).

        Returns dict with per-component enrichment deltas.
        """
        if not self._epochs:
            return {}

        latest = self._epochs[-1]
        tensor = latest.spirit_tensor

        # Compute per-component balance scores (closer to 0.5 = more balanced)
        # Full 130D: Inner[0:65] (B5+M15+S45) + Outer[65:130] (B5+M15+S45)
        # Legacy 30D: 6 components × 5D each
        if len(tensor) >= 130:
            components = {
                "inner_body":   tensor[0:5],       # 5D
                "inner_mind":   tensor[5:20],      # 15D
                "inner_spirit": tensor[20:65],     # 45D
                "outer_body":   tensor[65:70],     # 5D
                "outer_mind":   tensor[70:85],     # 15D
                "outer_spirit": tensor[85:130],    # 45D
            }
        elif len(tensor) >= 30:
            # Legacy 30D fallback (6 × 5D)
            components = {
                "inner_body":   tensor[0:5],
                "inner_mind":   tensor[5:10],
                "inner_spirit": tensor[10:15],
                "outer_body":   tensor[15:20],
                "outer_mind":   tensor[20:25],
                "outer_spirit": tensor[25:30],
            }
        else:
            return {}

        enrichment = {}
        for comp_name, comp_tensor in components.items():
            # Balance score: 1.0 = perfect center, 0.0 = maximally imbalanced
            avg_delta = sum(abs(v - 0.5) for v in comp_tensor) / len(comp_tensor)
            balance_score = max(0.0, 1.0 - avg_delta * 2.0)

            # Quality bonus from continuous enrichment between GREAT PULSEs
            quality_bonus = min(2.0, 1.0 + latest.cumulative_quality / 100.0)

            # Enrichment = base × balance_score × velocity_bonus × quality_bonus
            velocity_bonus = min(2.0, max(0.5, latest.velocity))
            reward = self._enrichment_base * balance_score * velocity_bonus * quality_bonus

            enrichment[comp_name] = {
                "reward": round(reward, 6),
                "balance_score": round(balance_score, 4),
                "velocity_bonus": round(velocity_bonus, 4),
                "quality_bonus": round(quality_bonus, 4),
            }

        latest.enrichment_sent = True
        return enrichment

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def tensor(self) -> list[float]:
        """Current 30DT SPIRIT tensor (live)."""
        return list(self._tensor)

    @property
    def inner_tensor(self) -> list[float]:
        """Inner Trinity 15DT (subconscious layer)."""
        return list(self._tensor[0:INNER_DIMS])

    @property
    def outer_tensor(self) -> list[float]:
        """Outer Trinity 15DT (conscious layer)."""
        return list(self._tensor[INNER_DIMS:SPIRIT_DIMS])

    @property
    def epoch_count(self) -> int:
        """Total GREAT EPOCHs (Titan's subjective age)."""
        return len(self._epochs)

    @property
    def latest_epoch(self) -> Optional[GreatEpoch]:
        """Most recent GREAT EPOCH."""
        return self._epochs[-1] if self._epochs else None

    def get_epoch(self, epoch_id: int) -> Optional[GreatEpoch]:
        """Get a specific GREAT EPOCH by ID."""
        for e in self._epochs:
            if e.epoch_id == epoch_id:
                return e
        return None

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        latest = self._epochs[-1].to_dict() if self._epochs else None
        return {
            "epoch_count": len(self._epochs),
            "current_epoch_id": self._current_epoch_id,
            "velocity": round(self._current_velocity, 4),
            "is_stale": self._is_stale,
            "consecutive_stale": self._consecutive_stale,
            "stale_focus_multiplier": round(self.stale_focus_multiplier, 4),
            "tensor_magnitude": round(_tensor_magnitude(self._tensor), 4),
            "tensor_sum": round(sum(self._tensor), 4),
            "latest_epoch": latest,
            "cumulative_quality": round(self._cumulative_quality, 4),
            "micro_tick_count": self._micro_tick_count,
            "last_alignment": round(self._last_alignment, 4),
            "enrichment_rate": self._enrichment_rate,
            "config": {
                "stale_threshold": self._stale_threshold,
                "enrichment_base": self._enrichment_base,
                "velocity_window": self._velocity_window,
                "enrichment_rate": self._enrichment_rate,
                "min_alignment_threshold": self._min_alignment_threshold,
            },
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist SPIRIT state and epoch history to disk."""
        try:
            Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                "tensor": self._tensor,
                "current_epoch_id": self._current_epoch_id,
                "current_velocity": self._current_velocity,
                "is_stale": self._is_stale,
                "consecutive_stale": self._consecutive_stale,
                "cumulative_quality": self._cumulative_quality,
                "micro_tick_count": self._micro_tick_count,
                "last_alignment": self._last_alignment,
                "epochs": [e.to_dict() for e in self._epochs],
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning("[UnifiedSpirit] Save failed: %s", e)

    def _load_state(self) -> None:
        """Restore SPIRIT state from disk. Handles 30D → 130D migration."""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                state = json.load(f)
            saved_tensor = state.get("tensor", [0.5] * SPIRIT_DIMS)
            # Migrate: if saved tensor is legacy 30D, expand to 130D
            if len(saved_tensor) < SPIRIT_DIMS:
                logger.info("[UnifiedSpirit] Migrating tensor from %dD → %dD",
                            len(saved_tensor), SPIRIT_DIMS)
                # Legacy layout: [iB5, iM5, iS5, oB5, oM5, oS5]
                # New layout: [iB5, iM15, iS45, oB5, oM15, oS45]
                # Copy what we can, pad the rest
                new_tensor = [0.5] * SPIRIT_DIMS
                if len(saved_tensor) >= 30:
                    # Inner body [0:5]
                    new_tensor[0:5] = saved_tensor[0:5]
                    # Inner mind [5:10] → [5:20] (pad 10 new dims)
                    new_tensor[5:10] = saved_tensor[5:10]
                    # Inner spirit [10:15] → [20:25] (pad 40 new dims)
                    new_tensor[20:25] = saved_tensor[10:15]
                    # Outer body [15:20] → [65:70]
                    new_tensor[65:70] = saved_tensor[15:20]
                    # Outer mind [20:25] → [70:75] (pad 10 new dims)
                    new_tensor[70:75] = saved_tensor[20:25]
                    # Outer spirit [25:30] → [85:90] (pad 40 new dims)
                    new_tensor[85:90] = saved_tensor[25:30]
                saved_tensor = new_tensor
            self._tensor = saved_tensor
            self._current_epoch_id = state.get("current_epoch_id", 0)
            self._current_velocity = state.get("current_velocity", 1.0)
            self._is_stale = state.get("is_stale", False)
            self._consecutive_stale = state.get("consecutive_stale", 0)
            self._cumulative_quality = state.get("cumulative_quality", 0.0)
            self._micro_tick_count = state.get("micro_tick_count", 0)
            self._last_alignment = state.get("last_alignment", 0.0)
            self._epochs = [
                GreatEpoch.from_dict(d) for d in state.get("epochs", [])
            ]
            logger.info(
                "[UnifiedSpirit] Restored: %d GREAT EPOCHs, velocity=%.3f",
                len(self._epochs), self._current_velocity,
            )
        except Exception as e:
            logger.warning("[UnifiedSpirit] Load failed: %s", e)


# ── Utility ───────────────────────────────────────────────────────────

def _tensor_magnitude(tensor: list[float]) -> float:
    """L2 magnitude of a tensor."""
    return math.sqrt(sum(v * v for v in tensor))


def _pad_or_trim(values: list[float], length: int) -> list[float]:
    """Ensure a list is exactly `length` items, padding with 0.5 or trimming."""
    if len(values) >= length:
        return values[:length]
    return values + [0.5] * (length - len(values))
