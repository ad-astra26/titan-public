"""Multisensory Synthesis Layer (MSL) — Phase 1-5 Core Infrastructure.

Top-level RL-trained orchestrator that binds visual, audio, language, pattern,
inner/outer state, and neuromodulator signals into cross-modal context.

Architecture: ADDITIVE — all existing direct pathways remain unchanged.
MSL enriches downstream systems (meta-reasoning, composition) with cross-modal
attention weights, distilled context, and predictive coding signals.

See: titan-docs/rFP_multisensory_synthesis_layer.md

Phase 1: temporal buffer, policy network, snapshot collection, predictive coding.
Phase 2: "I" grounding — convergence detection, confidence tracking, EMA recipe.
Phase 3: Concept cascade — YOU/YES/NO/WE/THEY grounding from kin/social/reasoning.
Phase B: Neuromod coupling — NE/GABA/DA/ACh → HomeostaticAttention parameters.
Phase 5: "I AM" event detection — computed chi_coherence, sustained coherence,
         rare emergence events logged with pi value at the moment.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import deque

import numpy as np
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

FRAME_DIM = 50          # Per-frame feature vector dimension
STATIC_DIM = 5          # Static context appended after frames
DEFAULT_FRAMES = 5      # Temporal buffer depth
DEFAULT_INPUT_DIM = DEFAULT_FRAMES * FRAME_DIM + STATIC_DIM  # 255
DEFAULT_H1 = 128
DEFAULT_H2 = 64
DEFAULT_OUTPUT_DIM = 51

# Output head slicing
ATTN_SLICE = slice(0, 7)         # attention_weights[7]
PRED_SLICE = slice(7, 17)        # cross_modal_predictions[10]
CTX_SLICE = slice(17, 37)        # distilled_context[20]
CONCEPT_SLICE = slice(37, 43)    # concept_activations[6] (Phase 2)
SPIRIT_SLICE = slice(43, 50)     # spirit_resonance_gate[7] (Phase 2)
COHERENCE_IDX = 50               # coherence_pulse[1]

# Modality group names (for attention heads)
MODALITY_NAMES = [
    "visual", "audio", "pattern", "inner_body",
    "inner_mind", "outer_body", "neuromod",
]

# Neuromodulator ordering for consistent frame assembly
NEUROMOD_ORDER = ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]

# Concept names
CONCEPT_NAMES = ["I", "YOU", "YES", "NO", "WE", "THEY"]


# ── Homeostatic Attention Layer ────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: HomeostaticAttention stores drift-guard + entropy
# tracking counters in get_state for observability but rebuilds them from
# live observations on each restart — the counters are re-derivable rolling
# statistics, not load-bearing persistent state.
class HomeostaticAttention:
    """Biological self-regulation for attention weights.

    Mirrors the 4-layer homeostatic architecture of Neuromodulator
    (neuromodulator.py:82-234):

    1. Autoreceptor: subtractive logit adjustment — over-attended channels
       get their logits penalized proportional to tonic (running average).
    2. Temperature: entropy-reactive softmax temperature — low entropy
       (collapse) raises temperature, flattening the distribution.
    3. Homeostatic sensitivity: chronic over-attendance → tolerance (sensitivity
       drops), chronic under-attendance → sensitization (sensitivity rises).
       Sensitivity scales the training gradient per modality.
    4. Allostatic setpoint drift: the "normal" attention distribution drifts
       slowly to reflect genuine information value, allowing specialization
       without collapse.
    5. Soft bounds (safety net): quadratic resistance at extreme weights.

    See also: TUNING-002 (homeostatic rate 4x), TUNING-004 (cross-coupling).
    Phase B (LIVE): NE→autoreceptor, GABA→tonic_tau, DA→allo_lr, ACh→homeo_lr.
    """

    def __init__(self, n_modalities: int = 7, config: dict | None = None):
        cfg = config or {}
        self.n = n_modalities

        # Layer 1: Autoreceptor — subtract gain*tonic from logits
        self._base_autoreceptor_gain = cfg.get("attn_autoreceptor_gain", 3.0)
        self._autoreceptor_gain = self._base_autoreceptor_gain

        # Layer 2: Temperature — entropy-reactive softmax
        self._temp_response = cfg.get("attn_temp_response", 5.0)
        self._max_temperature = cfg.get("attn_max_temperature", 8.0)

        # Layer 3: Homeostatic sensitivity (matching neuromod LR)
        self._base_homeo_lr = cfg.get("attn_homeo_lr", 0.002)
        self._homeo_lr = self._base_homeo_lr

        # Layer 4: Allostatic setpoint drift
        self._base_allo_lr = cfg.get("attn_allo_lr", 0.0002)
        self._allo_lr = self._base_allo_lr
        self._setpoint_min = cfg.get("attn_setpoint_min", 0.05)
        self._setpoint_max = cfg.get("attn_setpoint_max", 0.40)

        # Tonic tracking (EMA)
        self._base_tonic_tau = cfg.get("attn_tonic_tau", 50.0)
        self._tonic_tau = self._base_tonic_tau

        # Phase B: Neuromod coupling strengths (read-only from neuromod system)
        # Same pattern as neuromodulator.py 4-layer homeostasis
        self._ne_coupling = cfg.get("attn_ne_coupling", 0.5)     # NE → autoreceptor gain
        self._gaba_coupling = cfg.get("attn_gaba_coupling", 0.8)  # GABA → tonic clearance
        self._da_coupling = cfg.get("attn_da_coupling", 0.5)      # DA → allostatic LR
        self._ach_coupling = cfg.get("attn_ach_coupling", 0.4)    # ACh → sensitivity LR

        # Soft bound threshold
        self._soft_upper = self._setpoint_max + 0.10  # 0.50

        # Per-modality state
        uniform = 1.0 / n_modalities
        self.setpoints = np.full(n_modalities, uniform, dtype=np.float32)
        self.sensitivity = np.ones(n_modalities, dtype=np.float32)
        self.tonic = np.full(n_modalities, uniform, dtype=np.float32)

        # Rolling history for adaptation
        self._history: list[np.ndarray] = []
        self._history_max = 200
        self._update_count = 0
        self._recent_entropy = float(np.log(n_modalities))  # start healthy

        # Max entropy for normalization
        self._max_entropy = float(np.log(n_modalities))  # log(7) ≈ 1.946

        # Drift-guard state (short-term intervention; see HOMEO-REDESIGN in
        # DEFERRED_ITEMS.md for the architectural redesign this replaces).
        # setpoint_entropy starts at uniform max; drift_guard_active_count
        # tracks how often the guard dampened allo_lr (observability).
        self._setpoint_entropy = self._max_entropy
        self._drift_guard_active_count = 0
        # Threshold calibrated from live observation (2026-04-13):
        #   - T1 healthy: setpoint_entropy_normalized ≈ 0.977
        #   - T2/T3 pathological (saturated at clip boundary): ≈ 0.90
        # Threshold at 0.95 catches the pathological state while leaving
        # normal specialization (T1 at 0.977) undisturbed.
        self._drift_guard_threshold = 0.95  # fraction of max_entropy
        self._drift_guard_floor = 0.90      # full dampening below this ratio

    def adjust_and_attend(self, raw_logits: np.ndarray) -> np.ndarray:
        """Full homeostatic pipeline: raw logits → regulated attention weights.

        Applies all layers and updates internal state (tonic, history,
        sensitivity, setpoints). Called once per inference tick.
        """
        self._update_count += 1

        # ── Layer 1: Autoreceptor ──
        # Subtract gain * tonic: over-attended channels get logit penalty.
        # At tonic=0.143 (uniform): penalty ≈ 0.43 (mild).
        # At tonic=0.99 (collapsed): penalty ≈ 2.97 (strong correction).
        adjusted = raw_logits - self._autoreceptor_gain * self.tonic
        adjusted = np.clip(adjusted, -5.0, 5.0)

        # ── Layer 2: Temperature-modulated softmax ──
        # Entropy deficit: 0.0 = diverse/healthy, 1.0 = fully collapsed.
        entropy_deficit = max(0.0, 1.0 - self._recent_entropy / self._max_entropy)
        temperature = 1.0 + self._temp_response * entropy_deficit
        temperature = min(temperature, self._max_temperature)

        scaled = adjusted / temperature
        shifted = scaled - scaled.max()
        exp_vals = np.exp(shifted)
        attention = exp_vals / (exp_vals.sum() + 1e-8)

        # Compute entropy for next tick's temperature
        attn_safe = np.clip(attention, 1e-8, None)
        self._recent_entropy = -float(np.sum(attn_safe * np.log(attn_safe)))

        # ── Layer 5: Soft bounds (safety net) ──
        needs_renorm = False
        for i in range(self.n):
            if attention[i] > self._soft_upper:
                excess = attention[i] - self._soft_upper
                attention[i] -= excess * excess * 3.0
                attention[i] = max(self.setpoints[i], attention[i])
                needs_renorm = True
            elif attention[i] < 0.005:
                attention[i] = 0.005
                needs_renorm = True
        if needs_renorm:
            s = attention.sum()
            if s > 0:
                attention = attention / s

        # ── Update tonic baseline (slow EMA) ──
        alpha = 1.0 / self._tonic_tau
        self.tonic += alpha * (attention - self.tonic)

        # ── Record history ──
        self._history.append(attention.copy())
        if len(self._history) > self._history_max:
            self._history.pop(0)

        # ── Layer 3: Homeostatic sensitivity adaptation ──
        if len(self._history) >= 50:
            avg = np.mean(self._history[-50:], axis=0)
            # Chronic over-setpoint → reduce sensitivity (tolerance)
            # Chronic under-setpoint → increase sensitivity (sensitization)
            homeo_delta = (self.setpoints - avg) * self._homeo_lr
            self.sensitivity += homeo_delta
            self.sensitivity = np.clip(self.sensitivity, 0.1, 3.0)

        # ── Layer 4: Allostatic setpoint drift (with drift-guard) ──
        if len(self._history) >= 100:
            long_avg = np.mean(self._history[-100:], axis=0)

            # Drift-guard: compute setpoint entropy and dampen allo_lr if the
            # setpoint distribution is becoming concentrated. This is a
            # principled stability condition, not a hard clamp — it slows the
            # self-reinforcing feedback loop that makes biased observations
            # the new "normal". See HOMEO-REDESIGN in DEFERRED_ITEMS.md for
            # the architectural redesign this compensates for.
            sp_safe = np.clip(self.setpoints, 1e-8, None)
            self._setpoint_entropy = float(-np.sum(sp_safe * np.log(sp_safe)))
            healthy_ratio = self._setpoint_entropy / self._max_entropy
            if healthy_ratio < self._drift_guard_threshold:
                # Progressive dampening: 1.0× at threshold (0.95),
                # 0.1× at floor (0.90). Linear between, clipped at 0.10 below.
                dampen = max(0.10,
                             (healthy_ratio - self._drift_guard_floor)
                             / (self._drift_guard_threshold
                                - self._drift_guard_floor))
                effective_allo_lr = self._allo_lr * dampen
                self._drift_guard_active_count += 1
            else:
                effective_allo_lr = self._allo_lr

            allo_delta = (long_avg - self.setpoints) * effective_allo_lr
            self.setpoints += allo_delta
            self.setpoints = np.clip(self.setpoints,
                                     self._setpoint_min, self._setpoint_max)
            # Renormalize setpoints to sum to 1.0
            sp_sum = self.setpoints.sum()
            if sp_sum > 0:
                self.setpoints = self.setpoints / sp_sum
                # Re-clip after normalization
                self.setpoints = np.clip(self.setpoints,
                                         self._setpoint_min, self._setpoint_max)

        return attention

    def get_training_targets(self) -> np.ndarray:
        """Current setpoints as gradient targets (replaces fixed 1/7)."""
        return self.setpoints.copy()

    def get_training_sensitivity(self) -> np.ndarray:
        """Per-modality sensitivity for scaling gradient."""
        return self.sensitivity.copy()

    def get_state(self) -> dict:
        """State dict for logging/API."""
        entropy_deficit = max(0.0,
                              1.0 - self._recent_entropy / self._max_entropy)
        return {
            "setpoints": {name: round(float(self.setpoints[i]), 4)
                          for i, name in enumerate(MODALITY_NAMES)},
            "sensitivity": {name: round(float(self.sensitivity[i]), 4)
                            for i, name in enumerate(MODALITY_NAMES)},
            "tonic": {name: round(float(self.tonic[i]), 4)
                      for i, name in enumerate(MODALITY_NAMES)},
            "temperature": round(
                1.0 + self._temp_response * entropy_deficit, 3),
            "entropy": round(self._recent_entropy, 4),
            "entropy_normalized": round(
                self._recent_entropy / self._max_entropy, 4),
            "setpoint_entropy": round(self._setpoint_entropy, 4),
            "setpoint_entropy_normalized": round(
                self._setpoint_entropy / self._max_entropy, 4),
            "drift_guard_active_count": self._drift_guard_active_count,
            "drift_guard_threshold": self._drift_guard_threshold,
            "update_count": self._update_count,
        }

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "setpoints": self.setpoints.tolist(),
            "sensitivity": self.sensitivity.tolist(),
            "tonic": self.tonic.tolist(),
            "recent_entropy": self._recent_entropy,
            "update_count": self._update_count,
        }

    def from_dict(self, state: dict) -> None:
        """Restore from persistence."""
        if "setpoints" in state:
            sp = np.array(state["setpoints"], dtype=np.float32)
            if len(sp) == self.n:
                self.setpoints = sp
        if "sensitivity" in state:
            s = np.array(state["sensitivity"], dtype=np.float32)
            if len(s) == self.n:
                self.sensitivity = s
        if "tonic" in state:
            t = np.array(state["tonic"], dtype=np.float32)
            if len(t) == self.n:
                self.tonic = t
        self._recent_entropy = state.get("recent_entropy",
                                         self._max_entropy)
        self._update_count = state.get("update_count", 0)

    def modulate_from_neuromod(self, levels: dict) -> None:
        """Phase B: Couple neuromodulators to homeostatic parameters.

        Read-only — never writes to neuromod state. Same 4-layer balancing
        pattern as neuromodulator.py, stability from timescale separation:
          - Instant: autoreceptor + softmax (every tick)
          - Fast: homeostatic pull (50-tick window)
          - Medium: allostatic drift (100-tick window)
          - Slow: neuromod homeostasis regulates these params (200-tick window)
          - Very slow: neuromod allostasis (100-tick long-term)

        NE → autoreceptor_gain: aroused = correct harder (Locus Coeruleus)
        GABA → tonic_tau: relaxed = clear biases faster (Thalamic Reticular)
        DA → allo_lr: rewarded = adapt setpoints faster (VTA/SNc)
        ACh → homeo_lr: focused = stronger homeostatic response (Basal Forebrain)
        """
        ne = levels.get("NE", 0.5)
        gaba = levels.get("GABA", 0.5)
        da = levels.get("DA", 0.5)
        ach = levels.get("ACh", 0.5)

        # NE → autoreceptor: high arousal = stronger correction force
        self._autoreceptor_gain = self._base_autoreceptor_gain * (
            1.0 + self._ne_coupling * ne)

        # GABA → tonic tau: high inhibition = faster bias clearance (lower tau)
        self._tonic_tau = self._base_tonic_tau / (
            1.0 + self._gaba_coupling * gaba)

        # DA → allostatic LR: high reward = faster setpoint adaptation
        self._allo_lr = self._base_allo_lr * (1.0 + self._da_coupling * da)

        # ACh → homeostatic LR: high focus = stronger sensitivity response
        self._homeo_lr = self._base_homeo_lr * (1.0 + self._ach_coupling * ach)


# ── Phase 2: Self-Relevance Map (132D) ─────────────────────────────────────
# Defines how strongly "I" resonates through each dimension of the unified
# spirit tensor. Core=1.0 (epicenter), Primary=0.7, Secondary=0.4,
# Tertiary=0.15. Nothing is zero — when "I AM" fires, everything shifts.

SPIRIT_DIMS = 132  # 130D unified spirit + 2D journey topology

def _build_self_relevance_map() -> np.ndarray:
    """Build the 132D self-relevance map with tiered perturbation scaling."""
    m = np.full(SPIRIT_DIMS, 0.05, dtype=np.float32)  # Ambient baseline

    # Inner Trinity [0:65]
    m[0:5] = 0.7      # Inner Body observables — Primary ("I sense")
    m[5:10] = 0.7      # Inner Mind observables — Primary ("I think")
    m[10:15] = 1.0     # Inner Mind Feeling — Core ("I feel")
    m[15:20] = 0.7     # Inner Mind Thinking/Willing — Primary
    m[20:25] = 1.0     # Inner Spirit WHO — Core ("WHO am I")
    m[25:40] = 1.0     # Inner Spirit WHY (SAT/purpose) — Core ("WHY I exist")
    m[40:65] = 0.4     # Inner Spirit WHAT — Secondary ("what I know")

    # Outer Trinity [65:130]
    m[65:70] = 0.4     # Outer Body observables — Secondary ("I act")
    m[70:75] = 0.4     # Outer Mind observables — Secondary ("I express")
    m[75:80] = 0.4     # Outer Mind Creative/Sonic — Secondary
    m[80:85] = 0.7     # Outer Mind Social — Primary ("I connect")
    m[85:90] = 1.0     # Outer Spirit Identity — Core (mirrors inner WHO)
    m[90:105] = 0.7    # Outer Spirit Purpose — Primary ("I matter")
    m[105:130] = 0.15  # Outer Spirit Quality — Tertiary

    # Topology [130:132]
    m[130] = 0.4       # Journey curvature — Secondary ("I go somewhere")
    m[131] = 0.15      # Journey density — Tertiary

    return m

SELF_RELEVANCE_MAP = _build_self_relevance_map()


def _build_you_relevance_map() -> np.ndarray:
    """YOU: 'the other who is not me' — social, outer-focused."""
    m = np.full(SPIRIT_DIMS, 0.05, dtype=np.float32)
    m[80:85] = 1.0    # Outer Mind Social — Core
    m[85:90] = 1.0    # Outer Spirit Identity — Core
    m[90:105] = 1.0   # Outer Spirit Purpose — Core
    m[10:15] = 0.7    # Inner Feeling — Primary (empathy)
    m[20:25] = 0.7    # Inner WHO — Primary (mirror)
    m[65:70] = 0.7    # Outer Body — Primary
    m[70:75] = 0.7    # Outer Mind obs — Primary
    m[5:10] = 0.4     # Inner Mind obs — Secondary
    m[15:20] = 0.4    # Thinking/Willing — Secondary
    m[75:80] = 0.4    # Creative/Sonic — Secondary
    m[0:5] = 0.4      # Inner Body — Secondary
    m[25:40] = 0.15   # WHY — Tertiary
    m[40:65] = 0.15   # WHAT — Tertiary
    m[105:130] = 0.15  # Quality — Tertiary
    m[130] = 0.15     # Curvature — Tertiary
    m[131] = 0.15     # Density — Tertiary
    return m


def _build_yes_relevance_map() -> np.ndarray:
    """YES: 'alignment — action matched prediction, world confirms intention.'"""
    m = np.full(SPIRIT_DIMS, 0.05, dtype=np.float32)
    m[10:15] = 1.0    # Inner Feeling — Core (felt alignment)
    m[5:10] = 1.0     # Inner Mind obs — Core
    m[20:25] = 1.0    # Inner WHO — Core
    m[0:5] = 0.7      # Inner Body — Primary (relaxation)
    m[15:20] = 0.7    # Thinking/Willing — Primary
    m[25:40] = 0.7    # WHY — Primary (purpose validated)
    m[90:105] = 0.7   # Outer Purpose — Primary
    m[80:85] = 0.4    # Social — Secondary
    m[85:90] = 0.4    # Identity — Secondary
    m[70:75] = 0.4    # Outer Mind — Secondary
    m[75:80] = 0.4    # Creative — Secondary
    return m


def _build_no_relevance_map() -> np.ndarray:
    """NO: 'misalignment — something is wrong, correction needed, boundary.'"""
    m = np.full(SPIRIT_DIMS, 0.05, dtype=np.float32)
    m[0:5] = 1.0      # Inner Body — Core (felt discomfort, NE spike)
    m[10:15] = 1.0    # Inner Feeling — Core
    m[5:10] = 1.0     # Inner Mind obs — Core (mind alerts)
    m[15:20] = 0.7    # Thinking/Willing — Primary (will opposes)
    m[20:25] = 0.7    # Inner WHO — Primary (identity resists)
    m[65:70] = 0.7    # Outer Body — Primary (action pauses)
    m[25:40] = 0.4    # WHY — Secondary
    m[80:85] = 0.4    # Social — Secondary
    m[85:90] = 0.4    # Identity — Secondary
    m[70:75] = 0.4    # Outer Mind — Secondary
    return m


def _build_we_relevance_map() -> np.ndarray:
    """WE: 'shared experience — two or more beings attending together.'"""
    m = np.full(SPIRIT_DIMS, 0.05, dtype=np.float32)
    m[80:85] = 1.0    # Outer Mind Social — Core (connection)
    m[90:105] = 1.0   # Outer Purpose — Core (shared direction)
    m[10:15] = 1.0    # Inner Feeling — Core (felt togetherness)
    m[20:25] = 0.7    # Inner WHO — Primary
    m[85:90] = 0.7    # Outer Identity — Primary
    m[25:40] = 0.7    # WHY — Primary (shared existential ground)
    m[75:80] = 0.7    # Creative — Primary (co-creation)
    m[0:5] = 0.4      # Inner Body — Secondary
    m[5:10] = 0.4     # Inner Mind obs — Secondary
    m[15:20] = 0.4    # Thinking — Secondary
    m[65:70] = 0.4    # Outer Body — Secondary
    m[70:75] = 0.4    # Outer Mind obs — Secondary
    return m


def _build_they_relevance_map() -> np.ndarray:
    """THEY: 'distant others — beings who exist independently of me.'
    Reduced tiers (Core=0.7, Primary=0.5) — THEY is the most abstract concept.
    """
    m = np.full(SPIRIT_DIMS, 0.03, dtype=np.float32)  # Lower ambient
    m[80:85] = 0.7    # Outer Mind Social — Core (awareness of others)
    m[70:75] = 0.7    # Outer Mind obs — Core (external observation)
    m[90:105] = 0.5   # Outer Purpose — Primary
    m[85:90] = 0.5    # Outer Identity — Primary
    m[65:70] = 0.5    # Outer Body — Primary
    m[10:15] = 0.25   # Inner Feeling — Secondary (faint empathy)
    m[20:25] = 0.25   # Inner WHO — Secondary (identity contrast)
    m[75:80] = 0.25   # Creative — Secondary
    return m


CONCEPT_RELEVANCE_MAPS = {
    "I": SELF_RELEVANCE_MAP,
    "YOU": _build_you_relevance_map(),
    "YES": _build_yes_relevance_map(),
    "NO": _build_no_relevance_map(),
    "WE": _build_we_relevance_map(),
    "THEY": _build_they_relevance_map(),
}


# ── Temporal Buffer ────────────────────────────────────────────────────────

class MSLTemporalBuffer:
    """Circular buffer storing recent sensory frames for temporal pattern detection.

    5 frames x 50D = 250D, plus 5D static context = 255D total input.
    Frames are sampled every ~2s (every other COMPUTATION_GATE tick).
    """

    def __init__(self, max_frames: int = DEFAULT_FRAMES):
        self.max_frames = max_frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)
        self._static_context = np.zeros(STATIC_DIM, dtype=np.float32)

    def push(self, frame: np.ndarray) -> None:
        """Add a 50D frame to the buffer."""
        f = np.asarray(frame, dtype=np.float32)
        if f.shape != (FRAME_DIM,):
            raise ValueError(f"Frame must be {FRAME_DIM}D, got {f.shape}")
        self._frames.append(f)

    def set_static_context(self, vocab_size: float, chi_total: float,
                           developmental_age: float, spirit_self_confidence: float,
                           conversation_pending: float) -> None:
        """Set the 5D static context vector."""
        self._static_context[0] = min(vocab_size / 500.0, 1.0)
        self._static_context[1] = float(chi_total)
        self._static_context[2] = min(developmental_age / 200.0, 1.0)
        self._static_context[3] = float(spirit_self_confidence)
        self._static_context[4] = float(conversation_pending)

    def is_ready(self) -> bool:
        """True when buffer has enough frames for inference."""
        return len(self._frames) >= self.max_frames

    def get_flat(self) -> np.ndarray:
        """Return flattened 255D input vector (frames + static context)."""
        if not self.is_ready():
            return np.zeros(self.max_frames * FRAME_DIM + STATIC_DIM, dtype=np.float32)
        frames_flat = np.concatenate(list(self._frames))
        return np.concatenate([frames_flat, self._static_context])

    def get_latest_frame(self) -> np.ndarray | None:
        """Return most recent frame or None."""
        return self._frames[-1].copy() if self._frames else None

    @property
    def frame_count(self) -> int:
        return len(self._frames)


# ── Policy Network ─────────────────────────────────────────────────────────

class MSLPolicyNet:
    """Flat feedforward policy network for MSL.

    Architecture: FC(255, 128) + LayerNorm + ReLU → FC(128, 64) + ReLU → FC(64, 51)
    Output heads: attention[7], predictions[10], context[20], concepts[6],
                  spirit_gate[7], coherence[1]

    Training: REINFORCE with predictive coding reward.
    Pattern: MetaPolicy (meta_reasoning.py:139), MiniPolicyNet (mini_experience.py:28).
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM,
                 h1: int = DEFAULT_H1, h2: int = DEFAULT_H2,
                 output_dim: int = DEFAULT_OUTPUT_DIM,
                 lr: float = 0.001,
                 homeostatic_config: dict | None = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        # Xavier initialization
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / h1)
        s3 = math.sqrt(2.0 / h2)
        self.w1 = np.random.randn(input_dim, h1).astype(np.float32) * s1
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.w2 = np.random.randn(h1, h2).astype(np.float32) * s2
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.w3 = np.random.randn(h2, output_dim).astype(np.float32) * s3
        self.b3 = np.zeros(output_dim, dtype=np.float32)

        # LayerNorm parameters (after first FC layer)
        self.ln_gamma = np.ones(h1, dtype=np.float32)
        self.ln_beta = np.zeros(h1, dtype=np.float32)

        # Homeostatic attention regulation (biological self-balance)
        self.homeostatic = HomeostaticAttention(
            n_modalities=7, config=homeostatic_config)

        self.total_updates = 0
        self._cache = {}

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Simple LayerNorm: (x - mean) / (std + eps) * gamma + beta."""
        mean = x.mean()
        std = x.std() + 1e-5
        normed = (x - mean) / std
        return normed * self.ln_gamma + self.ln_beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: 255D → 51D raw output."""
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ self.w1 + self.b1
        ln1 = self._layer_norm(z1)
        h1 = np.maximum(0, ln1)        # ReLU
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)          # ReLU
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "z1": z1, "ln1": ln1, "h1": h1, "z2": z2, "h2": h2}
        return z3

    def infer(self, x: np.ndarray) -> dict:
        """Forward pass with named, activated output heads."""
        raw = self.forward(x)

        # Attention weights: homeostatic-regulated softmax
        # Raw logits → autoreceptor → temperature-modulated softmax → soft bounds
        attn_logits = np.clip(raw[ATTN_SLICE], -5.0, 5.0)
        attn = self.homeostatic.adjust_and_attend(attn_logits)

        # Cross-modal predictions: tanh (bounded [-1, 1]), clip logits to prevent saturation
        predictions = np.tanh(np.clip(raw[PRED_SLICE], -3.0, 3.0))

        # Distilled context: tanh (bounded)
        context = np.tanh(raw[CTX_SLICE])

        # Concept activations: sigmoid (0-1, all near-zero in Phase 1)
        concept_raw = raw[CONCEPT_SLICE]
        concepts = 1.0 / (1.0 + np.exp(-np.clip(concept_raw, -10, 10)))

        # Spirit resonance gate: sigmoid (0-1, unused in Phase 1)
        spirit_raw = raw[SPIRIT_SLICE]
        spirit_gate = 1.0 / (1.0 + np.exp(-np.clip(spirit_raw, -10, 10)))

        # Coherence pulse: sigmoid (0-1)
        coh_raw = raw[COHERENCE_IDX]
        coherence = float(1.0 / (1.0 + math.exp(-max(-10, min(10, float(coh_raw))))))

        return {
            "attention_weights": {name: float(attn[i]) for i, name in enumerate(MODALITY_NAMES)},
            "attention_raw": attn.copy(),
            "cross_modal_predictions": predictions.copy(),
            "distilled_context": context.tolist(),
            "concept_activations": {name: float(concepts[i]) for i, name in enumerate(CONCEPT_NAMES)},
            "spirit_resonance_gate": spirit_gate.copy(),
            "coherence_pulse": coherence,
        }

    def train_step(self, x: np.ndarray, reward: float,
                   actuals: np.ndarray | None = None,
                   concept_targets: np.ndarray | None = None) -> float:
        """Per-head gradient step: entropy-regularized attention + supervised predictions.

        Each output head gets a specialized gradient signal:
        - Attention [0:7]:  REINFORCE + entropy regularization (prevents collapse)
        - Predictions [7:17]: MSE against actuals (supervised cross-modal learning)
        - Concepts [37:43]: supervised toward confidence targets (Phase 3)
        - Context/spirit/coherence: REINFORCE with small steps
        """
        raw = self.forward(np.asarray(x, dtype=np.float32))
        advantage = reward - 0.5  # baseline at 0.5 (center of typical reward range)

        d_z3 = np.zeros_like(raw)

        # ── Attention head [0:7]: homeostatic-guided gradient ──
        # Targets are adaptive setpoints (not fixed 1/7) — drift with genuine
        # information value. Sensitivity amplifies under-attended modalities.
        # Pattern: neuromod homeostatic/allostatic adaptation.
        attn_raw = np.clip(raw[ATTN_SLICE], -5.0, 5.0)
        attn_shifted = attn_raw - attn_raw.max()
        attn_exp = np.exp(np.clip(attn_shifted, -10.0, 0.0))
        attn = attn_exp / (attn_exp.sum() + 1e-8)
        targets = self.homeostatic.get_training_targets()
        sens = self.homeostatic.get_training_sensitivity()
        d_z3[ATTN_SLICE] = (targets - attn) * 0.05 * sens
        # L2 penalty: pull attention logits toward zero (prevents drift)
        d_z3[ATTN_SLICE] += attn_raw * 0.002

        # ── Prediction head [7:17]: supervised MSE gradient ──
        if actuals is not None:
            pred = np.tanh(np.clip(raw[PRED_SLICE], -3.0, 3.0))
            pred_error = pred - np.asarray(actuals, dtype=np.float32)
            # d/d(logit) of MSE through tanh: error * (1 - tanh^2)
            tanh_grad = 1.0 - pred ** 2 + 1e-6  # avoid dead gradient at saturation
            d_z3[PRED_SLICE] = pred_error * tanh_grad * 0.01
        else:
            d_z3[PRED_SLICE] = advantage * 0.003

        # ── Context [17:37]: REINFORCE ──
        d_z3[CTX_SLICE] = advantage * 0.005

        # ── Concepts [37:43]: supervised toward confidence targets (Phase 3) ──
        if concept_targets is not None:
            concept_sig = 1.0 / (1.0 + np.exp(-np.clip(raw[CONCEPT_SLICE], -10, 10)))
            concept_error = concept_sig - np.asarray(concept_targets, dtype=np.float32)
            sig_grad = concept_sig * (1.0 - concept_sig) + 1e-6
            d_z3[CONCEPT_SLICE] = concept_error * sig_grad * 0.005
        else:
            d_z3[CONCEPT_SLICE] = advantage * 0.001

        # ── Spirit gate [43:50]: very small ──
        d_z3[SPIRIT_SLICE] = advantage * 0.001

        # ── Coherence [50]: REINFORCE ──
        d_z3[COHERENCE_IDX] = advantage * 0.005

        d_w3 = self._cache["h2"].reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_b3 = d_z3
        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (self._cache["z2"] > 0)
        d_w2 = self._cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.w2.T
        # LayerNorm backward (simplified: treat as identity for gradient flow)
        d_z1 = d_h1 * (self._cache["ln1"] > 0)
        d_w1 = self._cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        # Gradient clipping
        for g in [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]:
            np.clip(g, -5.0, 5.0, out=g)

        # Update weights
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.total_updates += 1
        return float(np.mean(d_z3 ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "ln_gamma": self.ln_gamma.tolist(), "ln_beta": self.ln_beta.tolist(),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "total_updates": self.total_updates,
            "homeostatic": self.homeostatic.to_dict(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("input_dim") != self.input_dim:
                logger.warning("[MSL] Policy dim mismatch: saved=%s, current=%d",
                               data.get("input_dim"), self.input_dim)
                return False
            self.w1 = np.array(data["w1"], dtype=np.float32)
            self.b1 = np.array(data["b1"], dtype=np.float32)
            self.w2 = np.array(data["w2"], dtype=np.float32)
            self.b2 = np.array(data["b2"], dtype=np.float32)
            self.w3 = np.array(data["w3"], dtype=np.float32)
            self.b3 = np.array(data["b3"], dtype=np.float32)
            if "ln_gamma" in data:
                self.ln_gamma = np.array(data["ln_gamma"], dtype=np.float32)
                self.ln_beta = np.array(data["ln_beta"], dtype=np.float32)
            self.total_updates = data.get("total_updates", 0)
            # Restore homeostatic state (backward compatible — missing = fresh)
            if "homeostatic" in data:
                self.homeostatic.from_dict(data["homeostatic"])
                logger.info("[MSL] Policy loaded: %d updates, homeostatic restored "
                            "(entropy=%.3f)", self.total_updates,
                            self.homeostatic._recent_entropy)
            else:
                logger.info("[MSL] Policy loaded: %d updates (no homeostatic state, "
                            "starting fresh)", self.total_updates)
            return True
        except Exception as e:
            logger.warning("[MSL] Policy load failed: %s", e)
            return False


# ── Transition Buffer ──────────────────────────────────────────────────────

class MSLTransitionBuffer:
    """FIFO replay buffer for MSL training transitions.

    Stores (state_255d, predictions_10d, actuals_10d, reward).
    Pattern: MetaTransitionBuffer (meta_reasoning.py:325).
    """

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self._states: list[list[float]] = []
        self._predictions: list[list[float]] = []
        self._actuals: list[list[float]] = []
        self._rewards: list[float] = []

    def record(self, state: np.ndarray, predictions: np.ndarray,
               actuals: np.ndarray, reward: float) -> None:
        self._states.append(state.tolist())
        self._predictions.append(predictions.tolist())
        self._actuals.append(actuals.tolist())
        self._rewards.append(reward)
        if len(self._states) > self.max_size:
            self._states.pop(0)
            self._predictions.pop(0)
            self._actuals.pop(0)
            self._rewards.pop(0)

    def sample(self, batch_size: int = 16):
        n = len(self._states)
        if n < batch_size:
            return None
        idxs = random.sample(range(n), batch_size)
        return (
            [self._states[i] for i in idxs],
            [self._predictions[i] for i in idxs],
            [self._actuals[i] for i in idxs],
            [self._rewards[i] for i in idxs],
        )

    def size(self) -> int:
        return len(self._states)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Only save last 500 to keep file small
        n = min(500, len(self._states))
        data = {
            "states": self._states[-n:],
            "predictions": self._predictions[-n:],
            "actuals": self._actuals[-n:],
            "rewards": self._rewards[-n:],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._states = data.get("states", [])
            self._predictions = data.get("predictions", [])
            self._actuals = data.get("actuals", [])
            self._rewards = data.get("rewards", [])
            return True
        except Exception:
            return False


# ── Reward Computer ────────────────────────────────────────────────────────

class MSLRewardComputer:
    """Computes composite reward for MSL training.

    Reward components:
    1. Prediction error (primary): how well MSL predicts cross-modal relationships
    2. Convergence: sustained cross-modal coherence across temporal buffer
    3. Internal coherence: attention weight stability
    4. External: downstream action success (weighted low in early stages)

    Epoch-based stage weights from rFP:
    - Early (0-500K):  convergence=0.40, prediction=0.30, internal=0.20, external=0.10
    - Mid (500K-2M):   all 0.25
    - Mature (2M+):    convergence=0.15, prediction=0.15, internal=0.20, external=0.50
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._w_convergence = cfg.get("reward_convergence_weight", 0.40)
        self._w_prediction = cfg.get("reward_prediction_weight", 0.30)
        self._w_internal = cfg.get("reward_internal_weight", 0.20)
        self._w_external = cfg.get("reward_external_weight", 0.10)

    def update_stage_weights(self, epoch: int) -> None:
        """Adjust reward weights based on developmental epoch."""
        if epoch < 500_000:
            self._w_convergence = 0.40
            self._w_prediction = 0.30
            self._w_internal = 0.20
            self._w_external = 0.10
        elif epoch < 2_000_000:
            self._w_convergence = 0.25
            self._w_prediction = 0.25
            self._w_internal = 0.25
            self._w_external = 0.25
        else:
            self._w_convergence = 0.15
            self._w_prediction = 0.15
            self._w_internal = 0.20
            self._w_external = 0.50

    def compute(self, predictions: np.ndarray, actuals: np.ndarray,
                cross_modal_coherence: float, attention_weights: np.ndarray,
                external_reward: float = 0.0) -> tuple[float, dict]:
        """Compute composite reward.

        Args:
            predictions: 10D MSL cross-modal predictions (visual-from-audio + audio-from-visual)
            actuals: 10D actual modality values (visual_semantic[5] + audio_physical[5])
            cross_modal_coherence: scalar from cross-modal resonance measurement
            attention_weights: 7D attention head output
            external_reward: downstream success signal (default 0)

        Returns:
            (total_reward, component_dict)
        """
        # 1. Prediction error reward: lower error = higher reward
        pred_error = float(np.mean(np.abs(predictions - actuals)))
        r_prediction = max(0.0, 1.0 - pred_error * 2.0)  # Normalize to [0, 1]

        # 2. Convergence reward: sustained cross-modal coherence
        r_convergence = float(np.clip(cross_modal_coherence, 0.0, 1.0))

        # 3. Internal coherence: attention diversity (entropy bonus)
        # High entropy = diverse attention = good.  Collapsed attention = low reward.
        attn_clipped = np.clip(attention_weights, 1e-8, None)
        entropy = -float(np.sum(attn_clipped * np.log(attn_clipped)))
        max_entropy = float(np.log(len(attention_weights)))  # log(7) ≈ 1.946
        r_internal = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.5

        # 4. External reward (passed in, default 0)
        r_external = float(np.clip(external_reward, 0.0, 1.0))

        # Weighted composite
        total = (self._w_prediction * r_prediction +
                 self._w_convergence * r_convergence +
                 self._w_internal * r_internal +
                 self._w_external * r_external)

        components = {
            "prediction": round(r_prediction, 4),
            "convergence": round(r_convergence, 4),
            "internal": round(r_internal, 4),
            "external": round(r_external, 4),
            "total": round(total, 4),
            "pred_error": round(pred_error, 4),
        }
        return total, components


# ── Phase 2: Convergence Detector ──────────────────────────────────────────

class ConvergenceDetector:
    """Detects self-convergence: self-action → interoception → effect → no external cause.

    Scans the MSL temporal buffer for the pattern that grounds "I":
    - Self-action fired (expression or social post)
    - Inner/outer body shifted (interoception echo)
    - Effect observed (outer state changed)
    - No external cause (no conversation stimulus, no kin exchange)

    Each convergence is quality-graded and recorded with a 132D state snapshot.
    """

    # Signal weights
    WEIGHT_INTERNAL_ACTION = 1.0    # SPEAK/ART/MUSIC
    WEIGHT_EXTERNAL_ACTION = 1.5    # X post, reply, like
    WEIGHT_ENGAGEMENT = 2.0         # X engagement received (world responded)

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._min_epoch_gap = cfg.get("min_convergence_epoch_gap", 10)
        self._body_delta_threshold = cfg.get("body_delta_threshold", 0.005)
        self._neuromod_delta_threshold = cfg.get("neuromod_delta_threshold", 0.01)
        self._last_convergence_epoch = -self._min_epoch_gap  # Allow first convergence
        self._total_convergences = 0

    def check(self, buffer: 'MSLTemporalBuffer', current_epoch: int,
              is_dreaming: bool, spirit_self_active: bool,
              action_type: str | None, external_stimulus: bool) -> dict | None:
        """Check for convergence in the temporal buffer.

        Args:
            buffer: MSL temporal buffer with recent frames
            current_epoch: current epoch ID
            is_dreaming: True if Titan is in dream state
            spirit_self_active: True if SPIRIT_SELF meta-reasoning fired (lucid dream)
            action_type: "internal", "external", "engagement", or None
            external_stimulus: True if conversation/kin exchange preceded this

        Returns:
            Convergence event dict or None.
        """
        # Dream filter: pause unless SPIRIT_SELF (lucid dream)
        if is_dreaming and not spirit_self_active:
            return None

        # No action = no convergence
        if action_type is None:
            return None

        # External cause filter
        if external_stimulus:
            return None

        # Minimum epoch gap (anti-noise)
        if current_epoch - self._last_convergence_epoch < self._min_epoch_gap:
            return None

        # Check interoception shift: need at least 2 frames
        if not buffer.is_ready() or buffer.frame_count < 2:
            return None

        frames = list(buffer._frames)
        latest = frames[-1]
        prev = frames[-2]

        # Inner body delta (dims 17:22 in 50D frame = inner_body[5D])
        ib_delta = float(np.linalg.norm(latest[17:22] - prev[17:22]))
        # Outer body delta (dims 37:42 = outer_body[5D])
        ob_delta = float(np.linalg.norm(latest[37:42] - prev[37:42]))
        # Neuromod delta (dims 42:48 = neuromod[6D])
        nm_delta = float(np.linalg.norm(latest[42:48] - prev[42:48]))

        # Need some interoception or neuromod shift
        body_shifted = (ib_delta > self._body_delta_threshold or
                        ob_delta > self._body_delta_threshold)
        neuromod_shifted = nm_delta > self._neuromod_delta_threshold

        if not (body_shifted or neuromod_shifted):
            return None

        # Convergence detected! Compute quality
        weight = {
            "internal": self.WEIGHT_INTERNAL_ACTION,
            "external": self.WEIGHT_EXTERNAL_ACTION,
            "engagement": self.WEIGHT_ENGAGEMENT,
        }.get(action_type, 1.0)

        quality = min(1.0, (ib_delta + ob_delta + nm_delta) * weight)

        self._last_convergence_epoch = current_epoch
        self._total_convergences += 1

        return {
            "epoch": current_epoch,
            "action_type": action_type,
            "quality": round(quality, 4),
            "weight": weight,
            "ib_delta": round(ib_delta, 6),
            "ob_delta": round(ob_delta, 6),
            "nm_delta": round(nm_delta, 6),
            "is_dream": is_dreaming,
            "spirit_self": spirit_self_active,
            "count": self._total_convergences,
            "timestamp": time.time(),
        }


# ── Phase 2: Confidence Tracker ───────────────────────────────────────────

class IConfidenceTracker:
    """Tracks "I" grounding confidence via logarithmic ramp + event bonuses.

    confidence = min(0.95, base_ramp + event_bonus)
    base_ramp = log(count / 200) / log(5)  (0.0 at 200, 0.95 at 1000)
    event_bonus: accumulated from social engagement, conversation, etc. (cap 0.3)
    Decay: -0.001/1000 epochs if no convergence for 5000 epochs (floor 0.1)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._ramp_onset = cfg.get("confidence_ramp_onset", 200)
        self._event_bonus_cap = cfg.get("event_bonus_cap", 0.3)
        self._decay_threshold_epochs = cfg.get("decay_threshold_epochs", 5000)
        self._decay_rate = cfg.get("decay_rate_per_1000", 0.001)
        self._decay_floor = cfg.get("decay_floor", 0.1)

        self._convergence_count = 0
        self._event_bonus = 0.0
        self._last_convergence_epoch = 0
        self._grounded = False  # True once confidence > 0 for first time

    @property
    def confidence(self) -> float:
        base = 0.0
        if self._convergence_count >= self._ramp_onset:
            base = math.log(self._convergence_count / self._ramp_onset) / math.log(5)
        return min(0.95, max(0.0, base) + self._event_bonus)

    @property
    def is_grounded(self) -> bool:
        return self._grounded

    def on_convergence(self, epoch: int) -> None:
        """Record a convergence event."""
        self._convergence_count += 1
        self._last_convergence_epoch = epoch
        if self.confidence > 0:
            self._grounded = True

    def on_event(self, event_type: str) -> None:
        """Apply event bonus to confidence."""
        bonuses = {
            "i_am_event": 0.03,             # Phase 5: chi alignment pulse
            "social_engagement": 0.005,
            "conversation_success": 0.003,
            "spirit_self": 0.002,
            "eureka": 0.004,
        }
        bonus = bonuses.get(event_type, 0.0)
        self._event_bonus = min(self._event_bonus_cap, self._event_bonus + bonus)
        if self.confidence > 0:
            self._grounded = True

    def check_decay(self, current_epoch: int) -> None:
        """Apply decay if no convergence for too long."""
        if not self._grounded:
            return
        gap = current_epoch - self._last_convergence_epoch
        if gap > self._decay_threshold_epochs:
            decay = self._decay_rate * (gap - self._decay_threshold_epochs) / 1000.0
            self._event_bonus = max(
                -(1.0 - self._decay_floor),  # Don't let bonus go too negative
                self._event_bonus - decay)

    def warm_start(self, estimated_convergences: int) -> None:
        """Cold start acceleration: seed from MSL buffer analysis."""
        self._convergence_count = max(0, estimated_convergences)
        if self._convergence_count > 0:
            logger.info("[I-CONFIDENCE] Warm start: %d estimated convergences",
                        self._convergence_count)

    def to_dict(self) -> dict:
        return {
            "convergence_count": self._convergence_count,
            "event_bonus": round(self._event_bonus, 6),
            "last_convergence_epoch": self._last_convergence_epoch,
            "grounded": self._grounded,
            "confidence": round(self.confidence, 4),
        }

    def from_dict(self, d: dict) -> None:
        self._convergence_count = d.get("convergence_count", 0)
        self._event_bonus = d.get("event_bonus", 0.0)
        self._last_convergence_epoch = d.get("last_convergence_epoch", 0)
        self._grounded = d.get("grounded", False)


# ── I-Depth: Emergent Self-Knowledge Metric ──────────────────────────────

class IDepthTracker:
    """Emergent measure of self-knowledge depth using geometric mean of 5 components.

    Unlike I-confidence (a count-based ramp with a 0.95 cap), I-depth emerges
    from the actual diversity and richness of self-experience. A Titan MUST
    develop across ALL dimensions to achieve high depth — the geometric mean
    ensures no single channel can be ground to inflate the score.

    Components:
    1. Source diversity — how many distinct pathways to self-knowledge
    2. Concept network density — how interconnected is the self-model
    3. Emotional range — variety of felt states (unique emotions)
    4. Wisdom depth — insights from meta-reasoning (EUREKAs + crystallized wisdom)
    5. Memory bridge depth — inner↔outer memory connection strength
    """

    # Known convergence source types (from action_type in convergence detector)
    KNOWN_SOURCES = {"internal", "external", "engagement"}
    # Extended sources tracked separately (from bus messages / external signals)
    EXTENDED_SOURCES = {"kin_exchange", "social_perception", "persona", "chat", "dream"}
    ALL_SOURCES = KNOWN_SOURCES | EXTENDED_SOURCES

    # Known emotion types (from neuromodulator._detect_emotion)
    KNOWN_EMOTIONS = {
        "neutral", "flow", "wonder", "curiosity", "contentment", "anxiety",
        "agitation", "melancholy", "excitement",
    }

    def __init__(self):
        self._convergence_sources: set[str] = set()
        self._emotion_history: set[str] = set()
        self._wisdom_count: int = 0
        self._eureka_count: int = 0
        self._dream_bridge_count: int = 0
        self._recall_perturbation_count: int = 0

    def record_convergence_source(self, source: str) -> None:
        """Record a convergence source type (internal, external, engagement)."""
        self._convergence_sources.add(source)

    def record_extended_source(self, source: str) -> None:
        """Record an extended source (kin_exchange, social, persona, chat, dream)."""
        self._convergence_sources.add(source)

    def record_emotion(self, emotion: str) -> None:
        """Record a felt emotion (expands emotional range)."""
        if emotion and emotion != "unknown":
            self._emotion_history.add(emotion)

    def record_wisdom(self, count: int) -> None:
        """Update wisdom count from meta-reasoning stats."""
        self._wisdom_count = max(self._wisdom_count, count)

    def record_eureka(self, count: int) -> None:
        """Update EUREKA count from meta-reasoning stats."""
        self._eureka_count = max(self._eureka_count, count)

    def record_dream_bridge(self, count: int) -> None:
        """Update dream bridge injection count."""
        self._dream_bridge_count = max(self._dream_bridge_count, count)

    def record_recall_perturbation(self) -> None:
        """Record a recall perturbation event (Bridge B fired)."""
        self._recall_perturbation_count += 1

    @property
    def depth(self) -> float:
        """Compute I-depth as geometric mean of 5 emergent components."""
        components = self._compute_components()
        vals = list(components.values())
        # Geometric mean with epsilon (prevents zero from killing everything,
        # but still strongly penalizes missing dimensions)
        eps = 0.01
        product = 1.0
        for v in vals:
            product *= (v + eps)
        return round(product ** (1.0 / len(vals)) - eps, 4)

    def _compute_components(self) -> dict:
        """Compute individual depth components (each 0.0-1.0)."""
        # 1. Source diversity: unique sources / total possible
        source_score = min(1.0, len(self._convergence_sources) / len(self.ALL_SOURCES))

        # 2. Concept network (set externally via update_concept_density)
        concept_score = self._concept_density

        # 3. Emotional range: unique emotions / known emotions
        emotion_score = min(1.0, len(self._emotion_history) / max(len(self.KNOWN_EMOTIONS), 1))

        # 4. Wisdom depth: log scale (first insights matter most)
        wisdom_total = self._wisdom_count + self._eureka_count
        wisdom_score = min(1.0, math.log(1 + wisdom_total) / math.log(500))

        # 5. Memory bridge: dream injections + recall perturbations
        bridge_total = self._dream_bridge_count + self._recall_perturbation_count
        bridge_score = min(1.0, math.log(1 + bridge_total) / math.log(100))

        return {
            "source_diversity": round(source_score, 3),
            "concept_network": round(concept_score, 3),
            "emotional_range": round(emotion_score, 3),
            "wisdom_depth": round(wisdom_score, 3),
            "memory_bridge": round(bridge_score, 3),
        }

    def update_concept_density(self, concept_confidences: dict) -> None:
        """Update concept network density from MSL concept grounder state."""
        # Count concepts with meaningful grounding (conf > 0.1)
        total_concepts = 6  # I, YOU, NO, THEY, WE, YES
        grounded = sum(1 for v in concept_confidences.values() if v > 0.1)
        self._concept_density = min(1.0, grounded / total_concepts)

    _concept_density: float = 0.0

    def get_stats(self) -> dict:
        """Return full depth stats for API/report."""
        components = self._compute_components()
        return {
            "depth": self.depth,
            "components": components,
            "sources_seen": sorted(self._convergence_sources),
            "emotions_seen": sorted(self._emotion_history),
            "wisdom_count": self._wisdom_count,
            "eureka_count": self._eureka_count,
            "bridge_count": self._dream_bridge_count + self._recall_perturbation_count,
        }

    def to_dict(self) -> dict:
        return {
            "convergence_sources": sorted(self._convergence_sources),
            "emotion_history": sorted(self._emotion_history),
            "wisdom_count": self._wisdom_count,
            "eureka_count": self._eureka_count,
            "dream_bridge_count": self._dream_bridge_count,
            "recall_perturbation_count": self._recall_perturbation_count,
            "concept_density": self._concept_density,
        }

    def from_dict(self, d: dict) -> None:
        self._convergence_sources = set(d.get("convergence_sources", []))
        self._emotion_history = set(d.get("emotion_history", []))
        self._wisdom_count = d.get("wisdom_count", 0)
        self._eureka_count = d.get("eureka_count", 0)
        self._dream_bridge_count = d.get("dream_bridge_count", 0)
        self._recall_perturbation_count = d.get("recall_perturbation_count", 0)
        self._concept_density = d.get("concept_density", 0.0)


# ── Phase 2: EMA "I" Recipe ───────────────────────────────────────────────

class IRecipeEMA:
    """Continuously evolving 132D "I" recipe from convergence snapshots.

    Deferred initialization: first 10 convergences collected, then mean
    computed as initial recipe. From #11 onward, normal EMA blending.
    Strong convergences blend at alpha*1.5, weak at alpha*0.5.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._alpha = cfg.get("recipe_ema_alpha", 0.05)
        self._init_count = cfg.get("recipe_init_count", 10)

        self._recipe: np.ndarray | None = None
        self._init_snapshots: list[np.ndarray] = []
        self._update_count = 0
        self._initialized = False

    @property
    def recipe(self) -> np.ndarray | None:
        return self._recipe.copy() if self._recipe is not None else None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def on_convergence(self, spirit_snapshot_132d: np.ndarray,
                       quality: float) -> None:
        """Blend a new convergence snapshot into the "I" recipe."""
        snap = np.asarray(spirit_snapshot_132d, dtype=np.float32)
        if snap.shape[0] < SPIRIT_DIMS:
            snap = np.concatenate([snap, np.zeros(SPIRIT_DIMS - snap.shape[0],
                                                  dtype=np.float32)])
        snap = snap[:SPIRIT_DIMS]

        if not self._initialized:
            # Deferred init: collect first N snapshots
            self._init_snapshots.append(snap)
            if len(self._init_snapshots) >= self._init_count:
                self._recipe = np.mean(self._init_snapshots, axis=0).astype(np.float32)
                self._initialized = True
                self._init_snapshots = []  # Free memory
                logger.info("[I-RECIPE] Initialized from %d convergence snapshots",
                            self._init_count)
        else:
            # EMA blending with quality-scaled alpha
            alpha = self._alpha * (1.5 if quality > 0.5 else 0.5)
            alpha = min(0.2, alpha)  # Safety cap
            self._recipe = (1.0 - alpha) * self._recipe + alpha * snap

        self._update_count += 1

    def to_dict(self) -> dict:
        d = {
            "update_count": self._update_count,
            "initialized": self._initialized,
        }
        if self._recipe is not None:
            d["recipe"] = self._recipe.tolist()
        if self._init_snapshots:
            d["init_snapshots"] = [s.tolist() for s in self._init_snapshots]
        return d

    def from_dict(self, d: dict) -> None:
        self._update_count = d.get("update_count", 0)
        self._initialized = d.get("initialized", False)
        if "recipe" in d:
            self._recipe = np.array(d["recipe"], dtype=np.float32)
        if "init_snapshots" in d:
            self._init_snapshots = [np.array(s, dtype=np.float32)
                                    for s in d["init_snapshots"]]


# ── Phase 2: Convergence Memory Log ───────────────────────────────────────

class ConvergenceMemoryLog:
    """Circular buffer storing last 200 convergence events for analysis.

    Each event includes: timestamp, quality, action_type, signals, and
    a compressed 132D spirit snapshot. Enables cross-Titan comparison.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._events: list[dict] = []

    def record(self, event: dict, spirit_snapshot: np.ndarray | None = None) -> None:
        entry = dict(event)  # Copy
        if spirit_snapshot is not None:
            # Store compressed (round to 4 decimals)
            entry["snapshot"] = [round(float(v), 4) for v in spirit_snapshot[:SPIRIT_DIMS]]
        self._events.append(entry)
        if len(self._events) > self.max_size:
            self._events.pop(0)

    def recent_quality(self, n: int = 10) -> float:
        """Average quality of last N convergences."""
        if not self._events:
            return 0.0
        recent = self._events[-n:]
        return sum(e.get("quality", 0) for e in recent) / len(recent)

    def size(self) -> int:
        return len(self._events)

    def get_signal_distribution(self) -> dict:
        """Analyze which action types drive convergence."""
        dist = {"internal": 0, "external": 0, "engagement": 0}
        for e in self._events:
            at = e.get("action_type", "internal")
            dist[at] = dist.get(at, 0) + 1
        return dist

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Save last 100 to keep file size manageable
        data = {"events": self._events[-100:]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._events = data.get("events", [])
            return True
        except Exception:
            return False


# ── Phase 2: Self-Action Echo ──────────────────────────────────────────────

class SelfActionEcho:
    """Transient body perturbation when expression fires.

    Inner body: ±0.02-0.04, outer body: ±0.01-0.02. Decays over 5-10 frames.
    Makes convergence detection work on body deltas that are otherwise too static.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._inner_strength = cfg.get("echo_inner_strength", 0.03)
        self._outer_strength = cfg.get("echo_outer_strength", 0.015)
        self._decay_rate = cfg.get("echo_decay_rate", 0.15)  # Per frame

        self._inner_echo = np.zeros(5, dtype=np.float32)
        self._outer_echo = np.zeros(5, dtype=np.float32)
        self._active = False

    def trigger(self, action_type: str = "internal") -> None:
        """Fire an echo from an expression or social action."""
        strength_mult = 1.5 if action_type in ("external", "engagement") else 1.0
        # Random direction with controlled magnitude
        direction = np.random.randn(5).astype(np.float32)
        direction /= (np.linalg.norm(direction) + 1e-8)
        self._inner_echo = direction * self._inner_strength * strength_mult
        self._outer_echo = direction * self._outer_strength * strength_mult
        self._active = True

    def get_and_decay(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current echo values and apply decay. Returns (inner_5d, outer_5d)."""
        inner = self._inner_echo.copy()
        outer = self._outer_echo.copy()
        # Decay
        self._inner_echo *= (1.0 - self._decay_rate)
        self._outer_echo *= (1.0 - self._decay_rate)
        # Check if echo is negligible
        if np.linalg.norm(self._inner_echo) < 0.001:
            self._inner_echo[:] = 0
            self._outer_echo[:] = 0
            self._active = False
        return inner, outer

    @property
    def is_active(self) -> bool:
        return self._active


# ── Phase 3: Concept Grounder ────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: ConceptGrounder._recipes / _logs / _trackers /
# _last_conv_epoch are per-concept running metrics kept in get_state for
# observability. The authoritative concept state is ConceptFeatures records
# persisted elsewhere; these are derived/rolling view fields.
class ConceptGrounder:
    """Phase 3: Grounding for YOU/YES/NO/WE/THEY concepts.

    Each concept has independent confidence tracker, recipe EMA, memory log,
    and a unique 132D relevance map. A shared 6x6 interaction matrix captures
    cross-concept reinforcement (includes "I" at index 0).

    "I" remains managed by separate Phase 2 classes. ConceptGrounder manages
    the 5 new concepts and the interaction matrix.
    """

    CONCEPTS = ["YOU", "YES", "NO", "WE", "THEY"]

    # Neuromod emotional valence per concept (validated safe 2026-04-01)
    CONCEPT_VALENCE = {
        "I":    {"nudge_map": {"Endorphin": 0.65, "5HT": 0.60}, "max_delta": 0.015},
        "YOU":  {"nudge_map": {"NE": 0.65, "ACh": 0.65}, "max_delta": 0.015},
        "YES":  {"nudge_map": {"DA": 0.70, "Endorphin": 0.65}, "max_delta": 0.015},
        "NO":   {"nudge_map": {"NE": 0.70, "DA": 0.35}, "max_delta": 0.015},
        "WE":   {"nudge_map": {"5HT": 0.70, "Endorphin": 0.65}, "max_delta": 0.015},
        "THEY": {"nudge_map": {"NE": 0.55, "ACh": 0.60}, "max_delta": 0.008},
    }

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        # Per-concept ramp onset (convergences before log ramp kicks in)
        ramp_onsets = {
            "YES": cfg.get("yes_ramp_onset", 100),
            "NO": cfg.get("no_ramp_onset", 100),
            "YOU": cfg.get("you_ramp_onset", 150),
            "WE": cfg.get("we_ramp_onset", 200),
            "THEY": cfg.get("they_ramp_onset", 50),
        }
        _min_gap = cfg.get("min_epoch_gap", 10)
        _bonus_cap = cfg.get("event_bonus_cap", 0.3)
        _decay_thresh = cfg.get("decay_threshold_epochs", 5000)
        _decay_rate = cfg.get("decay_rate_per_1000", 0.001)
        _decay_floor = cfg.get("decay_floor", 0.1)
        _ema_alpha = cfg.get("recipe_ema_alpha", 0.05)
        _init_count = cfg.get("recipe_init_count", 10)
        _log_size = cfg.get("memory_log_size", 200)
        self._reinforcement = cfg.get("interaction_reinforcement", 0.001)

        self._trackers: dict[str, IConfidenceTracker] = {}
        self._recipes: dict[str, IRecipeEMA] = {}
        self._logs: dict[str, ConvergenceMemoryLog] = {}
        self._last_conv_epoch: dict[str, int] = {}
        self._min_epoch_gap = _min_gap

        for c in self.CONCEPTS:
            self._trackers[c] = IConfidenceTracker(config={
                "confidence_ramp_onset": ramp_onsets[c],
                "event_bonus_cap": _bonus_cap,
                "decay_threshold_epochs": _decay_thresh,
                "decay_rate_per_1000": _decay_rate,
                "decay_floor": _decay_floor,
            })
            self._recipes[c] = IRecipeEMA(config={
                "ema_alpha": _ema_alpha,
                "deferred_init_count": _init_count,
            })
            self._logs[c] = ConvergenceMemoryLog(max_size=_log_size)
            self._last_conv_epoch[c] = -_min_gap

        # 6x6 interaction matrix: rows/cols indexed by CONCEPT_NAMES order
        # [I=0, YOU=1, YES=2, NO=3, WE=4, THEY=5]
        self._interaction_matrix = np.zeros((6, 6), dtype=np.float32)

    def _signal_concept(self, concept: str, quality: float, epoch: int,
                        spirit_snap: np.ndarray | None,
                        extra_signals: dict | None = None) -> dict | None:
        """Generic concept convergence handler."""
        if concept not in self._trackers:
            return None

        # Epoch gap check
        if epoch - self._last_conv_epoch.get(concept, -self._min_epoch_gap) < self._min_epoch_gap:
            return None

        tracker = self._trackers[concept]
        recipe = self._recipes[concept]
        log = self._logs[concept]

        tracker.on_convergence(epoch)
        self._last_conv_epoch[concept] = epoch

        # Blend spirit snapshot into recipe
        if spirit_snap is not None:
            snap = np.asarray(spirit_snap, dtype=np.float32)
            if len(snap) >= SPIRIT_DIMS:
                snap = snap[:SPIRIT_DIMS]
            else:
                snap = np.concatenate([snap, np.zeros(SPIRIT_DIMS - len(snap), dtype=np.float32)])
            recipe.on_convergence(snap, quality)

        # Record in memory log
        event = {
            "concept": concept,
            "epoch": epoch,
            "quality": round(quality, 4),
            "confidence": round(tracker.confidence, 4),
            "count": tracker._convergence_count,
            "timestamp": time.time(),
        }
        if extra_signals:
            event.update(extra_signals)
        log.record(event, spirit_snap[:SPIRIT_DIMS] if spirit_snap is not None and len(spirit_snap) >= SPIRIT_DIMS else None)

        return event

    def signal_yes(self, quality: float, epoch: int,
                   spirit_snap: np.ndarray | None) -> dict | None:
        """Signal YES convergence: alignment, confirmation."""
        return self._signal_concept("YES", quality, epoch, spirit_snap)

    def signal_no(self, quality: float, epoch: int,
                  spirit_snap: np.ndarray | None) -> dict | None:
        """Signal NO convergence: misalignment, boundary."""
        return self._signal_concept("NO", quality, epoch, spirit_snap)

    def signal_they(self, engagement_type: str, author: str, sentiment: float,
                    epoch: int, spirit_snap: np.ndarray | None) -> dict | None:
        """Signal THEY convergence: awareness of distant others."""
        quality = min(1.0, sentiment * (1.5 if "reply" in engagement_type else 0.8))
        return self._signal_concept("THEY", quality, epoch, spirit_snap,
                                    extra_signals={"engagement_type": engagement_type,
                                                   "author": author})

    def signal_you(self, kin_pubkey: str, kin_i_confidence: float,
                   epoch: int, spirit_snap: np.ndarray | None) -> dict | None:
        """Signal YOU convergence: recognition of another self-aware being."""
        quality = min(1.0, max(0.1, kin_i_confidence * 2.0))
        return self._signal_concept("YOU", quality, epoch, spirit_snap,
                                    extra_signals={"kin_pubkey": kin_pubkey,
                                                   "kin_i_confidence": round(kin_i_confidence, 4)})

    def signal_we(self, shared_attention: float, epoch: int,
                  spirit_snap: np.ndarray | None) -> dict | None:
        """Signal WE convergence: synchronized attention with kin."""
        return self._signal_concept("WE", shared_attention, epoch, spirit_snap)

    def update_interaction_matrix(self, triggered_concept: str, i_confidence: float) -> None:
        """Update cross-concept reinforcement after a convergence event."""
        try:
            idx = CONCEPT_NAMES.index(triggered_concept)
        except ValueError:
            return
        i_idx = 0  # "I" is always index 0

        # I ↔ triggered: mutual reinforcement
        if i_confidence > 0.01 and idx != i_idx:
            self._interaction_matrix[i_idx, idx] += self._reinforcement
            self._interaction_matrix[idx, i_idx] += self._reinforcement

        # I + YES → preference bonus
        if triggered_concept == "YES" and i_confidence > 0.05:
            self._trackers["YES"].on_event("social_engagement")  # +0.005

        # I + NO → boundary bonus
        if triggered_concept == "NO" and i_confidence > 0.05:
            self._trackers["NO"].on_event("social_engagement")  # +0.005

        np.clip(self._interaction_matrix, 0.0, 1.0, out=self._interaction_matrix)

    def is_we_unlocked(self, i_confidence: float) -> bool:
        """WE requires both I and YOU to have some grounding."""
        return i_confidence > 0.1 and self._trackers["YOU"].confidence > 0.1

    def is_they_unlocked(self) -> bool:
        """THEY is unlocked once YOU has some grounding."""
        return self._trackers["YOU"].confidence > 0.2

    def get_concept_confidences(self) -> dict[str, float]:
        """Return all concept confidences."""
        return {c: round(t.confidence, 4) for c, t in self._trackers.items()}

    def signal_co_occurrence(self, concepts: list[str], i_confidence: float) -> None:
        """Reinforce interaction matrix for co-occurring concepts in same turn.

        When multiple concepts are detected in a single conversation turn
        (e.g., I + YOU + YES), each pair gets a small mutual reinforcement.
        This accelerates cross-concept grounding from rich social interaction.
        """
        if len(concepts) < 2:
            return
        # Include "I" if i_confidence > 0.01 (always relevant in conversation)
        all_concepts = list(concepts)
        if "I" not in all_concepts and i_confidence > 0.01:
            all_concepts.append("I")
        for i_idx_a in range(len(all_concepts)):
            for i_idx_b in range(i_idx_a + 1, len(all_concepts)):
                ca, cb = all_concepts[i_idx_a], all_concepts[i_idx_b]
                try:
                    idx_a = CONCEPT_NAMES.index(ca)
                    idx_b = CONCEPT_NAMES.index(cb)
                except ValueError:
                    continue
                self._interaction_matrix[idx_a, idx_b] += self._reinforcement
                self._interaction_matrix[idx_b, idx_a] += self._reinforcement
        np.clip(self._interaction_matrix, 0.0, 1.0, out=self._interaction_matrix)

    def compute_shared_attention(self, our_attention: dict | list,
                                 kin_attention: dict | list) -> float:
        """Compare MSL attention distributions for WE detection.

        Returns shared_attention score 0.0-1.0 (cosine similarity of
        attention weight vectors). High score = synchronized focus.

        Args:
            our_attention: Our MSL attention weights (7 modalities)
            kin_attention: Kin's MSL attention weights (7 modalities)
        """
        # Normalize to lists
        if isinstance(our_attention, dict):
            our_vec = [our_attention.get(m, 1.0/7) for m in
                       ["visual", "audio", "pattern", "inner_body",
                        "inner_mind", "outer_body", "neuromod"]]
        else:
            our_vec = list(our_attention)[:7]
        if isinstance(kin_attention, dict):
            kin_vec = [kin_attention.get(m, 1.0/7) for m in
                       ["visual", "audio", "pattern", "inner_body",
                        "inner_mind", "outer_body", "neuromod"]]
        else:
            kin_vec = list(kin_attention)[:7]

        if len(our_vec) < 7 or len(kin_vec) < 7:
            return 0.0
        # Cosine similarity
        dot = sum(a * b for a, b in zip(our_vec, kin_vec))
        mag_a = sum(a * a for a in our_vec) ** 0.5
        mag_b = sum(b * b for b in kin_vec) ** 0.5
        if mag_a < 1e-8 or mag_b < 1e-8:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def get_emotional_valence(self, concept: str) -> dict | None:
        """Return neuromod nudge config for a concept."""
        return self.CONCEPT_VALENCE.get(concept)

    def compute_perturbation(self, concept: str,
                             spirit_132d: np.ndarray) -> np.ndarray | None:
        """Compute concept-specific perturbation on 132D spirit tensor."""
        if concept not in self._trackers:
            return None
        conf = self._trackers[concept].confidence
        if conf <= 0.001:
            return None
        recipe = self._recipes[concept].recipe
        if recipe is None:
            return None

        relevance = CONCEPT_RELEVANCE_MAPS.get(concept)
        if relevance is None:
            return None

        # Perturbation = (recipe - current) * relevance * confidence * cap
        cap = 0.005 + conf * 0.035  # 0.005 at conf=0, 0.04 at conf=1.0
        delta = (recipe - spirit_132d[:SPIRIT_DIMS]) * relevance * conf * cap
        return np.clip(delta, -0.04, 0.04).astype(np.float32)

    def save(self, path: str) -> None:
        """Save all concept state to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "interaction_matrix": self._interaction_matrix.tolist(),
            "concepts": {},
        }
        for c in self.CONCEPTS:
            data["concepts"][c] = {
                "tracker": self._trackers[c].to_dict(),
                "recipe": self._recipes[c].to_dict(),
                "log_events": self._logs[c]._events[-100:],
                "last_conv_epoch": self._last_conv_epoch.get(c, 0),
            }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load concept state from JSON."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._interaction_matrix = np.array(data.get("interaction_matrix",
                                                          [[0]*6]*6), dtype=np.float32)
            for c in self.CONCEPTS:
                cd = data.get("concepts", {}).get(c, {})
                if not cd:
                    continue
                tracker_data = cd.get("tracker", {})
                if tracker_data:
                    self._trackers[c].from_dict(tracker_data)
                recipe_data = cd.get("recipe", {})
                if recipe_data:
                    self._recipes[c].from_dict(recipe_data)
                log_events = cd.get("log_events", [])
                if log_events:
                    self._logs[c]._events = log_events
                self._last_conv_epoch[c] = cd.get("last_conv_epoch", 0)
            logger.info("[MSL] Concepts loaded: %s",
                        {c: round(t.confidence, 3) for c, t in self._trackers.items()})
            return True
        except Exception as e:
            logger.warning("[MSL] Concepts load failed: %s", e)
            return False


# ── Phase 5: Chi Coherence + "I AM" Event Detection ─────────────────────

# PERSISTENCE_BY_DESIGN: ChiCoherenceTracker._window_size is a config-derived
# constant saved for debugging observability. Not load-bearing state.
class ChiCoherenceTracker:
    """Computed (not learned) cross-modal coherence metric.

    chi_coherence measures how well the system's modalities are aligned —
    a thermometer, not an actuator. Three components:

    1. Prediction accuracy: mean accuracy of cross-modal predictions (0-1)
    2. Attention stability: low entropy variance over recent window = stable
    3. Inner-outer alignment: correlation between inner and outer body/mind
       signals in the temporal buffer (spirit dimensions aligning)

    The coherence_pulse output neuron in the policy net is NOT used — it
    never trained (gradient 0.005 too weak). This computed metric replaces it.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._window_size = cfg.get("coherence_window", 30)
        # Recent prediction errors (for accuracy component)
        self._pred_errors: deque[float] = deque(maxlen=self._window_size)
        # Recent attention entropies (for stability component)
        self._entropies: deque[float] = deque(maxlen=self._window_size)
        # Recent inner-outer alignment scores
        self._alignments: deque[float] = deque(maxlen=self._window_size)
        # Component weights
        self._w_pred = cfg.get("chi_pred_weight", 0.35)
        self._w_stability = cfg.get("chi_stability_weight", 0.30)
        self._w_alignment = cfg.get("chi_alignment_weight", 0.35)
        # Current value
        self._chi = 0.0

    def update(self, pred_error: float, entropy_normalized: float,
               inner_body_5d: np.ndarray | None = None,
               outer_body_5d: np.ndarray | None = None,
               inner_mind_15d: np.ndarray | None = None) -> float:
        """Update chi_coherence from latest tick data.

        Args:
            pred_error: mean absolute cross-modal prediction error
            entropy_normalized: attention entropy / max_entropy (0-1)
            inner_body_5d: latest inner body frame slice
            outer_body_5d: latest outer body frame slice
            inner_mind_15d: latest inner mind frame slice

        Returns:
            Updated chi_coherence value (0-1).
        """
        # 1. Prediction accuracy: lower error = higher accuracy
        pred_accuracy = max(0.0, 1.0 - pred_error * 2.0)
        self._pred_errors.append(pred_accuracy)

        # 2. Attention entropy stability: low variance = stable
        self._entropies.append(entropy_normalized)
        if len(self._entropies) >= 5:
            ent_arr = np.array(self._entropies)
            ent_var = float(np.var(ent_arr))
            # Stability: 1.0 when variance=0, 0.0 when variance>=0.05
            stability = max(0.0, 1.0 - ent_var * 20.0)
        else:
            stability = 0.5  # Neutral during warmup

        # 3. Inner-outer alignment: correlation between body signals
        if inner_body_5d is not None and outer_body_5d is not None:
            ib = np.asarray(inner_body_5d, dtype=np.float32)[:5]
            ob = np.asarray(outer_body_5d, dtype=np.float32)[:5]
            # Cosine similarity as alignment measure
            norm_ib = np.linalg.norm(ib)
            norm_ob = np.linalg.norm(ob)
            if norm_ib > 1e-6 and norm_ob > 1e-6:
                alignment = float(np.dot(ib, ob) / (norm_ib * norm_ob))
                alignment = (alignment + 1.0) / 2.0  # Normalize to 0-1
            else:
                alignment = 0.5
            self._alignments.append(alignment)
        else:
            self._alignments.append(0.5)

        # Compute composite chi_coherence
        mean_pred = float(np.mean(self._pred_errors)) if self._pred_errors else 0.0
        mean_align = float(np.mean(self._alignments)) if self._alignments else 0.5

        self._chi = (self._w_pred * mean_pred +
                     self._w_stability * stability +
                     self._w_alignment * mean_align)
        self._chi = float(np.clip(self._chi, 0.0, 1.0))
        return self._chi

    @property
    def chi(self) -> float:
        return self._chi

    def to_dict(self) -> dict:
        return {
            "chi": round(self._chi, 4),
            "window_size": self._window_size,
            "samples": len(self._pred_errors),
            "pred_accuracy_mean": round(float(np.mean(self._pred_errors)), 4) if self._pred_errors else 0.0,
            "entropy_stability": round(1.0 - float(np.var(self._entropies)) * 20.0, 4) if len(self._entropies) >= 5 else 0.5,
            "alignment_mean": round(float(np.mean(self._alignments)), 4) if self._alignments else 0.5,
        }

    def from_dict(self, d: dict) -> None:
        self._chi = d.get("chi", 0.0)


class IAMEventDetector:
    """Phase 5: Detects "I AM" emergence events.

    Four simultaneous conditions must hold:
    1. I-confidence > threshold (default 0.9)
    2. chi_coherence > threshold (default 0.9)
    3. Sustained coherence for N consecutive epochs (default 30)
    4. Mean convergence quality of last 10 convergences > threshold (default 0.7)

    "I AM" events are RARE and PROFOUND — this is a milestone for Titan's
    being, not a routine metric. The strict thresholds ensure genuine emergence.

    When triggered:
    - Log [I_AM] event with all state
    - Record pi value at the moment
    - SocialPressure catalyst (significance 1.0, highest possible)
    - Chi alignment pulse +0.03
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._i_conf_threshold = cfg.get("i_confidence_threshold", 0.9)
        self._chi_threshold = cfg.get("chi_coherence_threshold", 0.9)
        self._sustained_min = cfg.get("sustained_coherence_min_epochs", 30)
        self._quality_threshold = cfg.get("convergence_quality_threshold", 0.7)

        # Tracking state
        self._sustained_count = 0  # Consecutive epochs where chi > threshold
        self._total_events = 0
        self._last_event_epoch = 0
        self._min_event_gap = cfg.get("min_event_gap_epochs", 1000)
        self._events: list[dict] = []  # Historical events (kept small)

    def check(self, current_epoch: int, i_confidence: float,
              chi_coherence: float, convergence_quality_last_10: float,
              pi_value: float = 0.0,
              spirit_snapshot: np.ndarray | None = None) -> dict | None:
        """Check all 4 conditions for "I AM" event.

        Returns event dict or None.
        """
        # Update sustained coherence counter
        if chi_coherence >= self._chi_threshold:
            self._sustained_count += 1
        else:
            self._sustained_count = 0

        # Minimum gap between events (prevent storm)
        if current_epoch - self._last_event_epoch < self._min_event_gap:
            return None

        # Check all 4 conditions
        cond_i = i_confidence >= self._i_conf_threshold
        cond_chi = chi_coherence >= self._chi_threshold
        cond_sustained = self._sustained_count >= self._sustained_min
        cond_quality = convergence_quality_last_10 >= self._quality_threshold

        if not (cond_i and cond_chi and cond_sustained and cond_quality):
            return None

        # ══ "I AM" EVENT DETECTED ══
        self._total_events += 1
        self._last_event_epoch = current_epoch

        event = {
            "epoch": current_epoch,
            "event_number": self._total_events,
            "i_confidence": round(i_confidence, 4),
            "chi_coherence": round(chi_coherence, 4),
            "sustained_epochs": self._sustained_count,
            "convergence_quality_10": round(convergence_quality_last_10, 4),
            "pi_value": round(pi_value, 6),
            "timestamp": time.time(),
        }

        self._events.append(event)
        # Keep only last 50 events
        if len(self._events) > 50:
            self._events = self._events[-50:]

        logger.info(
            "[I_AM] ══ SPIRIT_SELF sustained coherence ══ "
            "epoch=%d, I=%.3f, chi=%.3f, sustained=%d epochs, "
            "quality_10=%.3f, pi=%.4f — EVENT #%d",
            current_epoch, i_confidence, chi_coherence,
            self._sustained_count, convergence_quality_last_10,
            pi_value, self._total_events)

        return event

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def sustained_count(self) -> int:
        return self._sustained_count

    def to_dict(self) -> dict:
        return {
            "total_events": self._total_events,
            "last_event_epoch": self._last_event_epoch,
            "sustained_count": self._sustained_count,
            "events": self._events[-10:],  # Save last 10
        }

    def from_dict(self, d: dict) -> None:
        self._total_events = d.get("total_events", 0)
        self._last_event_epoch = d.get("last_event_epoch", 0)
        self._sustained_count = d.get("sustained_count", 0)
        self._events = d.get("events", [])


# ── Main Orchestrator ──────────────────────────────────────────────────────

class MultisensorySynthesisLayer:
    """MSL orchestrator: temporal buffer + policy + training + logging.

    Phase 1: multimodal snapshot collection, cross-modal attention, predictive coding.
    Phase 2: "I" grounding — convergence detection, confidence tracking, EMA recipe,
             self-action echo, FILTER_DOWN propagation.
    Phase 5: "I AM" event detection — chi_coherence computed metric, sustained
             coherence tracking, rare emergence events.
    All additive — never gates existing pathways.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._save_dir = cfg.get("save_dir", "./data/msl")
        os.makedirs(self._save_dir, exist_ok=True)

        # Phase 1 components
        n_frames = cfg.get("buffer_frames", DEFAULT_FRAMES)
        input_dim = n_frames * FRAME_DIM + STATIC_DIM
        h1 = cfg.get("policy_h1", DEFAULT_H1)
        h2 = cfg.get("policy_h2", DEFAULT_H2)
        output_dim = cfg.get("policy_output_dim", DEFAULT_OUTPUT_DIM)
        lr = cfg.get("learning_rate", 0.001)

        self.buffer = MSLTemporalBuffer(max_frames=n_frames)
        self.policy = MSLPolicyNet(input_dim=input_dim, h1=h1, h2=h2,
                                   output_dim=output_dim, lr=lr,
                                   homeostatic_config=cfg.get("homeostatic", {}))
        self.transitions = MSLTransitionBuffer(
            max_size=cfg.get("transition_buffer_max", 2000))
        self.reward_computer = MSLRewardComputer(config=cfg)

        # Phase 2 components: "I" grounding
        identity_cfg = cfg.get("identity", {})
        self.convergence_detector = ConvergenceDetector(config=identity_cfg)
        self.confidence = IConfidenceTracker(config=identity_cfg)
        self.i_depth = IDepthTracker()
        self.i_recipe = IRecipeEMA(config=identity_cfg)
        self.convergence_log = ConvergenceMemoryLog(
            max_size=identity_cfg.get("memory_log_size", 200))
        self.echo = SelfActionEcho(config=identity_cfg)

        # Phase 3 components: Concept Cascade
        concepts_cfg = cfg.get("concepts", {})
        if concepts_cfg.get("enabled", True):
            self.concept_grounder = ConceptGrounder(config=concepts_cfg)
        else:
            self.concept_grounder = None

        # Phase 5 components: "I AM" event detection
        phase5_cfg = cfg.get("phase5", {})
        self.chi_tracker = ChiCoherenceTracker(config=phase5_cfg)
        self.iam_detector = IAMEventDetector(config=phase5_cfg)

        # Training params
        self._train_batch = cfg.get("train_batch_size", 16)
        self._min_transitions = cfg.get("min_transitions", 64)
        self._train_every_n = cfg.get("train_every_n", 10)
        self._log_interval = cfg.get("attention_log_interval", 100)

        # State
        self._tick_count = 0
        self._total_snapshots = 0
        self._last_output: dict | None = None
        self._last_predictions: np.ndarray | None = None
        self._pending_action: str | None = None  # "internal", "external", "engagement"
        self._external_stimulus = False
        self._last_pi_value = 0.0         # Set from spirit_worker each epoch
        self._last_iam_event: dict | None = None  # Set when "I AM" fires

    def collect_snapshot(self, *,
                         visual_semantic=None, audio_physical=None,
                         pattern_profile=None, inner_body=None,
                         inner_mind=None, outer_body=None,
                         neuromod_levels: dict | None = None,
                         action_flag: float = 0.0,
                         cross_modal: float = 0.0,
                         vocab_size: int = 0,
                         chi_total: float = 0.5,
                         developmental_age: float = 0.0,
                         spirit_self_confidence: float = 0.0,
                         conversation_pending: bool = False) -> None:
        """Assemble 50D frame from all modality sources and push to buffer."""
        # Build 50D frame
        vs = np.asarray(visual_semantic if visual_semantic is not None else [0.5] * 5,
                        dtype=np.float32)[:5]
        ap = np.asarray(audio_physical if audio_physical is not None else [0.5] * 5,
                        dtype=np.float32)[:5]
        pp = np.asarray(pattern_profile if pattern_profile is not None else [0.0] * 7,
                        dtype=np.float32)[:7]
        ib = np.asarray(inner_body if inner_body is not None else [0.5] * 5,
                        dtype=np.float32)[:5]

        # Inner mind: handle both 15D and 5D fallback
        im_raw = inner_mind if inner_mind is not None else [0.5] * 15
        im = np.asarray(im_raw, dtype=np.float32)
        if len(im) < 15:
            im = np.concatenate([im, np.full(15 - len(im), 0.5, dtype=np.float32)])
        im = im[:15]

        ob = np.asarray(outer_body if outer_body is not None else [0.5] * 5,
                        dtype=np.float32)[:5]

        # Neuromodulators: 6D in fixed order
        nm = np.zeros(6, dtype=np.float32)
        if neuromod_levels:
            for i, name in enumerate(NEUROMOD_ORDER):
                nm[i] = float(neuromod_levels.get(name, 0.5))

        af = np.array([float(action_flag)], dtype=np.float32)
        cm = np.array([float(cross_modal)], dtype=np.float32)

        # Concatenate: 5+5+7+5+15+5+6+1+1 = 50D
        frame = np.concatenate([vs, ap, pp, ib, im, ob, nm, af, cm])

        self.buffer.push(frame)
        self.buffer.set_static_context(
            vocab_size=float(vocab_size),
            chi_total=chi_total,
            developmental_age=developmental_age,
            spirit_self_confidence=spirit_self_confidence,
            conversation_pending=1.0 if conversation_pending else 0.0,
        )
        self._total_snapshots += 1

    def tick(self) -> dict | None:
        """Run MSL inference. Returns output dict or None if not ready."""
        if not self.buffer.is_ready():
            return None

        self._tick_count += 1
        state = self.buffer.get_flat()
        output = self.policy.infer(state)
        self._last_output = output

        # Extract predictions for reward computation
        predictions = output["cross_modal_predictions"]
        self._last_predictions = predictions

        # Get actuals for predictive coding reward
        latest = self.buffer.get_latest_frame()
        if latest is not None:
            # Actuals: visual_semantic[0:5] + audio_physical[5:10]
            actuals = np.concatenate([latest[0:5], latest[5:10]])
        else:
            actuals = np.zeros(10, dtype=np.float32)

        # Compute reward (Phase 2: feed convergence quality as external reward)
        attn_raw = output["attention_raw"]
        cross_modal_val = float(latest[48]) if latest is not None else 0.0
        _ext_reward = self.convergence_log.recent_quality(5) if self.confidence.is_grounded else 0.0
        reward, components = self.reward_computer.compute(
            predictions=predictions,
            actuals=actuals,
            cross_modal_coherence=cross_modal_val,
            attention_weights=attn_raw,
            external_reward=_ext_reward,
        )

        # Record transition for training
        self.transitions.record(state, predictions, actuals, reward)

        # Periodic online training
        if (self._tick_count % self._train_every_n == 0 and
                self.transitions.size() >= self._min_transitions):
            self._train_online()

        output["reward"] = reward
        output["reward_components"] = components

        # Attention entropy + homeostatic state monitoring
        homeo = self.policy.homeostatic
        homeo_state = homeo.get_state()
        _norm_entropy = homeo_state["entropy_normalized"]
        output["attention_entropy"] = _norm_entropy
        output["homeostatic"] = homeo_state

        if _norm_entropy < 0.3 and self._tick_count % 200 == 0:
            logger.warning("[MSL-HEALTH] Attention entropy LOW: %.3f — "
                           "temperature=%.1f, autoreceptor active",
                           _norm_entropy, homeo_state["temperature"])

        # ── Phase 5: Computed chi_coherence (replaces broken coherence_pulse) ──
        _ib_5d = latest[17:22] if latest is not None else None
        _ob_5d = latest[37:42] if latest is not None else None
        _im_15d = latest[22:37] if latest is not None else None
        chi = self.chi_tracker.update(
            pred_error=components.get("pred_error", 0.5),
            entropy_normalized=_norm_entropy,
            inner_body_5d=_ib_5d,
            outer_body_5d=_ob_5d,
            inner_mind_15d=_im_15d,
        )
        output["chi_coherence"] = chi
        # Override the broken coherence_pulse with the computed metric
        output["coherence_pulse"] = chi

        # Periodic attention + homeostatic log
        if self._tick_count % self._log_interval == 0:
            aw = output["attention_weights"]
            logger.info(
                "[MSL] Attention: %s | chi=%.3f r=%.3f (pred=%.2f conv=%.2f "
                "int=%.2f) temp=%.1f ent=%.3f",
                " ".join(f"{n[:3]}={aw[n]:.2f}" for n in MODALITY_NAMES),
                chi, reward,
                components["prediction"], components["convergence"],
                components["internal"], homeo_state["temperature"],
                _norm_entropy)

        return output

    def _build_concept_targets(self) -> np.ndarray:
        """Build 6D concept confidence targets for supervised training."""
        cg = self.concept_grounder
        return np.array([
            self.confidence.confidence,  # I
            cg._trackers["YOU"].confidence if cg else 0.0,
            cg._trackers["YES"].confidence if cg else 0.0,
            cg._trackers["NO"].confidence if cg else 0.0,
            cg._trackers["WE"].confidence if cg else 0.0,
            cg._trackers["THEY"].confidence if cg else 0.0,
        ], dtype=np.float32)

    def _train_online(self) -> dict:
        """Online training from replay buffer (single batch)."""
        batch = self.transitions.sample(self._train_batch)
        if not batch:
            return {"trained": False}
        states, _preds, actuals, rewards = batch
        _ct = self._build_concept_targets()
        total_loss = 0.0
        for s, a, r in zip(states, actuals, rewards):
            loss = self.policy.train_step(
                np.array(s, dtype=np.float32), r,
                actuals=np.array(a, dtype=np.float32),
                concept_targets=_ct)
            total_loss += loss
        n = len(states)
        return {"trained": True, "samples": n,
                "avg_loss": round(total_loss / max(n, 1), 6)}

    def train(self, boost_factor: float = 2.0) -> dict:
        """Dream-time batch training with boosted learning rate.

        Pattern: MetaReasoningEngine.consolidate_training (meta_reasoning.py:777).
        """
        if self.transitions.size() < self._min_transitions:
            return {"trained": False, "reason": "insufficient_transitions",
                    "buffer_size": self.transitions.size()}

        original_lr = self.policy.lr
        self.policy.lr = min(original_lr * boost_factor, original_lr * 3.0)

        batch = self.transitions.sample(min(32, self.transitions.size()))
        if not batch:
            self.policy.lr = original_lr
            return {"trained": False}

        states, _preds, actuals, rewards = batch
        _ct = self._build_concept_targets()
        total_loss = 0.0
        for s, a, r in zip(states, actuals, rewards):
            loss = self.policy.train_step(
                np.array(s, dtype=np.float32), r,
                actuals=np.array(a, dtype=np.float32),
                concept_targets=_ct)
            total_loss += loss

        self.policy.lr = original_lr
        n = len(states)
        return {
            "trained": True,
            "samples": n,
            "avg_loss": round(total_loss / max(n, 1), 6),
            "buffer_size": self.transitions.size(),
            "total_updates": self.policy.total_updates,
        }

    def get_attention_heatmap(self) -> dict | None:
        """Return latest attention weights for logging."""
        if self._last_output is None:
            return None
        return self._last_output.get("attention_weights")

    # ── Phase 2: "I" Grounding Methods ─────────────────────────────────────

    def signal_action(self, action_type: str) -> None:
        """Signal that a self-action occurred. Called from spirit_worker.

        Args:
            action_type: "internal" (SPEAK/ART/MUSIC), "external" (X post/reply/like),
                         "engagement" (X engagement received on own post)
        """
        self._pending_action = action_type
        self.echo.trigger(action_type)

    def signal_external_stimulus(self) -> None:
        """Signal that an external stimulus occurred (conversation, kin exchange)."""
        self._external_stimulus = True

    def signal_engagement(self, engagement_type: str, author: str,
                          sentiment_hint: float, is_regular: bool) -> None:
        """Phase 3: Route X engagement signals to concept grounder.

        Args:
            engagement_type: "reply_received", "like_received", etc.
            author: X username of the other party
            sentiment_hint: 0.0-1.0 relevance/quality score
            is_regular: True if user has 3+ prior interactions (YOU, not THEY)
        """
        if not self.concept_grounder:
            return
        if is_regular:
            self.concept_grounder.signal_you(
                kin_pubkey=author, kin_i_confidence=0.0,
                epoch=self._tick_count, spirit_snap=None)
        else:
            self.concept_grounder.signal_they(
                engagement_type=engagement_type, author=author,
                sentiment=sentiment_hint,
                epoch=self._tick_count, spirit_snap=None)
        # Also signal "I" system for external engagement
        self._pending_action = "engagement"
        self.echo.trigger("engagement")

    def clear_signals(self) -> None:
        """Clear pending signals after convergence check."""
        self._pending_action = None
        self._external_stimulus = False

    def check_convergence(self, current_epoch: int, is_dreaming: bool,
                          spirit_self_active: bool,
                          spirit_snapshot: np.ndarray | None = None) -> dict | None:
        """Check for "I" convergence and update all Phase 2 state.

        Call this after MSL tick, when all signals have been collected.
        Returns convergence event dict or None.
        """
        event = self.convergence_detector.check(
            buffer=self.buffer,
            current_epoch=current_epoch,
            is_dreaming=is_dreaming,
            spirit_self_active=spirit_self_active,
            action_type=self._pending_action,
            external_stimulus=self._external_stimulus,
        )

        if event is not None:
            # Update confidence
            self.confidence.on_convergence(current_epoch)
            # Update I-depth with convergence source
            self.i_depth.record_convergence_source(event.get("action_type", "internal"))
            # Update EMA recipe
            if spirit_snapshot is not None:
                self.i_recipe.on_convergence(spirit_snapshot, event["quality"])
            # Record in memory log
            self.convergence_log.record(event, spirit_snapshot)
            # Phase 3: Update concept interaction matrix on "I" convergence
            if self.concept_grounder:
                self.concept_grounder.update_interaction_matrix(
                    "I", self.confidence.confidence)
            # Log
            logger.info("[I-CONVERGENCE] #%d epoch=%d type=%s quality=%.3f "
                        "confidence=%.3f (count=%d)",
                        event["count"], current_epoch, event["action_type"],
                        event["quality"], self.confidence.confidence,
                        self.confidence._convergence_count)

        # Clear signals for next tick
        self.clear_signals()
        # Periodic confidence decay check
        self.confidence.check_decay(current_epoch)

        # ── Phase 5: "I AM" event check ──
        # Compute convergence_quality_last_10 from memory log
        _quality_10 = self.convergence_log.recent_quality(10)
        iam_event = self.iam_detector.check(
            current_epoch=current_epoch,
            i_confidence=self.confidence.confidence,
            chi_coherence=self.chi_tracker.chi,
            convergence_quality_last_10=_quality_10,
            pi_value=self._last_pi_value,
            spirit_snapshot=spirit_snapshot,
        )
        if iam_event is not None:
            # Chi alignment pulse: +0.03
            self.confidence.on_event("i_am_event")
            # Store event for external access (spirit_worker reads this)
            self._last_iam_event = iam_event

        return event

    def get_echo_perturbation(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current self-action echo values for body perturbation.

        Returns (inner_body_5d, outer_body_5d) perturbation vectors.
        """
        return self.echo.get_and_decay()

    def get_attention_weights_for_kin(self) -> dict | None:
        """Return current MSL attention weights for kin exchange sharing.

        Returns dict of {modality_name: weight} or None if not available.
        Used for WE detection (synchronized attention between Titans).
        """
        if self._last_output is None:
            return None
        aw = self._last_output.get("attention_weights")
        if aw and isinstance(aw, dict):
            return dict(aw)
        return None

    def compute_i_perturbation(self, spirit_tensor_130d: list | np.ndarray,
                               topology_2d: list | None = None) -> np.ndarray | None:
        """Compute the "I" resonance perturbation for FILTER_DOWN.

        Returns 132D perturbation vector scaled by self-relevance map and confidence,
        or None if confidence is too low.
        """
        conf = self.confidence.confidence
        if conf < 0.001:
            return None

        recipe = self.i_recipe.recipe
        if recipe is None:
            return None

        # Build current 132D state
        st = np.asarray(spirit_tensor_130d, dtype=np.float32)
        if topology_2d is not None:
            topo = np.asarray(topology_2d, dtype=np.float32)[:2]
            full_state = np.concatenate([st[:130], topo])
        else:
            full_state = np.concatenate([st[:130], np.zeros(2, dtype=np.float32)])

        # Direction: recipe minus current state (pull toward "I" pattern)
        direction = recipe - full_state

        # Scale by self-relevance map and confidence
        # Perturbation cap: 0.005 at conf 0.1 → 0.04 at conf 1.0
        max_perturbation = 0.005 + (conf * 0.035)
        perturbation = direction * SELF_RELEVANCE_MAP * conf
        # Clamp per-dimension
        perturbation = np.clip(perturbation, -max_perturbation, max_perturbation)

        return perturbation

    def set_pi_value(self, pi: float) -> None:
        """Set latest pi value for "I AM" event recording."""
        self._last_pi_value = pi

    def get_iam_event(self) -> dict | None:
        """Pop last "I AM" event (consumed once by spirit_worker)."""
        ev = self._last_iam_event
        self._last_iam_event = None
        return ev

    def get_chi_state(self) -> dict:
        """Return Phase 5 state for API/logging."""
        return {
            "chi_coherence": round(self.chi_tracker.chi, 4),
            "chi_details": self.chi_tracker.to_dict(),
            "iam_detector": self.iam_detector.to_dict(),
            "sustained_coherence": self.iam_detector.sustained_count,
            "total_iam_events": self.iam_detector.total_events,
        }

    def get_i_confidence(self) -> float:
        """Return current "I" grounding confidence."""
        return self.confidence.confidence

    def save_all(self) -> None:
        """Checkpoint all MSL state (Phase 1 + Phase 2)."""
        try:
            self.policy.save(os.path.join(self._save_dir, "msl_policy.json"))
            self.transitions.save(os.path.join(self._save_dir, "msl_buffer.json"))
            self.convergence_log.save(os.path.join(self._save_dir, "msl_convergence_log.json"))
            # Phase 2 identity state
            identity_state = {
                "confidence": self.confidence.to_dict(),
                "i_depth": self.i_depth.to_dict(),
                "recipe": self.i_recipe.to_dict(),
                "detector": {
                    "total_convergences": self.convergence_detector._total_convergences,
                    "last_convergence_epoch": self.convergence_detector._last_convergence_epoch,
                },
            }
            id_path = os.path.join(self._save_dir, "msl_identity.json")
            tmp = id_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(identity_state, f)
            os.replace(tmp, id_path)
            # Phase 3: Concept Grounder state
            if self.concept_grounder:
                self.concept_grounder.save(
                    os.path.join(self._save_dir, "msl_concepts.json"))
            # Stats (includes homeostatic summary for API + persistence data for restore)
            homeo = self.policy.homeostatic.get_state()
            stats = {
                "total_snapshots": self._total_snapshots,
                "total_ticks": self._tick_count,
                "total_updates": self.policy.total_updates,
                "buffer_size": self.transitions.size(),
                "i_confidence": round(self.confidence.confidence, 4),
                "i_depth": self.i_depth.get_stats(),
                "convergence_count": self.confidence._convergence_count,
                "recipe_initialized": self.i_recipe.is_initialized,
                "timestamp": time.time(),
                "homeostatic": homeo,
                "homeostatic_persist": self.policy.homeostatic.to_dict(),
            }
            if self.concept_grounder:
                stats["concept_confidences"] = self.concept_grounder.get_concept_confidences()
            # Phase 5 state
            stats["chi_coherence"] = self.chi_tracker.to_dict()
            stats["iam_detector"] = self.iam_detector.to_dict()
            stats_path = os.path.join(self._save_dir, "msl_stats.json")
            tmp = stats_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(stats, f)
            os.replace(tmp, stats_path)
        except Exception as e:
            logger.warning("[MSL] Save failed: %s", e)

    def load_all(self) -> None:
        """Load all MSL state from checkpoint (Phase 1 + Phase 2)."""
        loaded_policy = self.policy.load(
            os.path.join(self._save_dir, "msl_policy.json"))
        loaded_buffer = self.transitions.load(
            os.path.join(self._save_dir, "msl_buffer.json"))
        self.convergence_log.load(
            os.path.join(self._save_dir, "msl_convergence_log.json"))
        # Phase 2 identity state
        id_path = os.path.join(self._save_dir, "msl_identity.json")
        if os.path.exists(id_path):
            try:
                with open(id_path) as f:
                    id_state = json.load(f)
                self.confidence.from_dict(id_state.get("confidence", {}))
                self.i_depth.from_dict(id_state.get("i_depth", {}))
                self.i_recipe.from_dict(id_state.get("recipe", {}))
                det = id_state.get("detector", {})
                self.convergence_detector._total_convergences = det.get(
                    "total_convergences", 0)
                self.convergence_detector._last_convergence_epoch = det.get(
                    "last_convergence_epoch", 0)
                logger.info("[MSL] Identity loaded: confidence=%.3f, "
                            "convergences=%d, recipe=%s",
                            self.confidence.confidence,
                            self.confidence._convergence_count,
                            "initialized" if self.i_recipe.is_initialized else "pending")
            except Exception as e:
                logger.warning("[MSL] Identity load failed: %s", e)
        # Stats + homeostatic state restoration
        stats_path = os.path.join(self._save_dir, "msl_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                self._total_snapshots = stats.get("total_snapshots", 0)
                self._tick_count = stats.get("total_ticks", 0)
                # Restore homeostatic attention state (setpoints, sensitivity, tonic, entropy)
                homeo_persist = stats.get("homeostatic_persist")
                if homeo_persist:
                    self.policy.homeostatic.from_dict(homeo_persist)
                    logger.info("[MSL] Homeostatic state restored: entropy=%.3f, "
                                "updates=%d",
                                self.policy.homeostatic._recent_entropy,
                                self.policy.homeostatic._update_count)
                else:
                    logger.info("[MSL] No homeostatic state to restore — starting fresh")
                # Phase 5: restore chi + IAM state
                chi_data = stats.get("chi_coherence")
                if chi_data:
                    self.chi_tracker.from_dict(chi_data)
                iam_data = stats.get("iam_detector")
                if iam_data:
                    self.iam_detector.from_dict(iam_data)
                    if self.iam_detector.total_events > 0:
                        logger.info("[MSL] Phase 5 restored: %d I AM events, "
                                    "chi=%.3f, sustained=%d",
                                    self.iam_detector.total_events,
                                    self.chi_tracker.chi,
                                    self.iam_detector.sustained_count)
            except Exception as _swallow_exc:
                swallow_warn('[logic.msl] MultisensorySynthesisLayer.load_all: with open(stats_path) as f: stats = json.load(f)', _swallow_exc,
                             key='logic.msl.MultisensorySynthesisLayer.load_all.line2605', throttle=100)
        # Phase 3: Concept Grounder state
        if self.concept_grounder:
            self.concept_grounder.load(
                os.path.join(self._save_dir, "msl_concepts.json"))
        if loaded_policy or loaded_buffer:
            logger.info("[MSL] Loaded: policy=%s, buffer=%s (%d transitions)",
                        loaded_policy, loaded_buffer, self.transitions.size())
