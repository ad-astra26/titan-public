"""
titan_plugin/logic/reasoning.py — Mind's Deliberate Cognition.

Reasoning engine with 7 composable logic primitives, cognitive EMA
for neuromod stabilization, policy network for primitive selection,
and Spirit observation protocol.

Architecture:
  Body (raw neuromods, NS programs) → gut signals rise via GroundUp
  Mind (EMA-smoothed neuromods) → reasoning chains via logic primitives
  Spirit (very slow EMA) → gentle nudges when reasoning stuck/exhausted

The reasoning layer IS Mind's Thinking dimension properly realized.
Feeling provides confidence. Willing provides commitment.
"""
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

# The 7 logic primitives + CONCLUDE action
PRIMITIVES = [
    "COMPARE",      # Detect similarity/difference between state slices
    "IF_THEN",      # Conditional check → record conclusion
    "SEQUENCE",     # Chain 2-3 sub-observations in order
    "ASSOCIATE",    # Query working memory for related items
    "DECOMPOSE",    # Split observation into labeled sub-vectors
    "LOOP",         # Re-execute last primitive (up to 3x)
    "NEGATE",       # Invert last result / reverse hypothesis
    "CONCLUDE",     # End chain, commit or abandon
]

NUM_ACTIONS = len(PRIMITIVES)  # 8

# Observation decomposition labels for DECOMPOSE primitive
DECOMPOSE_LABELS = {
    "body":     (0, 15),    # inner_body + outer_body observables
    "mind":     (15, 25),   # inner_mind + outer_mind + inner/outer spirit
    "spirit":   (25, 30),   # last 5 of tier1
    "temporal":  (30, 55),  # tier2
    "neuro":    (55, 67),   # tier5 (in enriched layout)
    "dynamics": (67, 75),   # tier6 (in enriched layout, pre-extension)
}

# ── Policy Network ────────────────────────────────────────────────


class ReasoningPolicyNet:
    """Small feedforward network selecting which primitive to execute.

    Input: observation + gut_signals + mind_neuromods + chain_state
    Output: preference score per action (softmax → selection)

    Architecture follows NeuralReflexNet pattern (manual numpy backprop,
    Xavier init, gradient clipping) but outputs 8 action scores.
    """

    def __init__(self, input_dim: int, hidden_1: int = 48, hidden_2: int = 24,
                 learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.h1 = hidden_1
        self.h2 = hidden_2
        self.output_dim = NUM_ACTIONS
        self.lr = learning_rate

        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_1)
        scale3 = math.sqrt(2.0 / hidden_2)

        self.w1 = np.random.randn(input_dim, hidden_1) * scale1
        self.b1 = np.zeros(hidden_1)
        self.w2 = np.random.randn(hidden_1, hidden_2) * scale2
        self.b2 = np.zeros(hidden_2)
        self.w3 = np.random.randn(hidden_2, NUM_ACTIONS) * scale3
        self.b3 = np.zeros(NUM_ACTIONS)

        # Training stats
        self.total_updates = 0
        self.last_loss = 0.0

        # Cached activations for backprop
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass → raw action scores (pre-softmax)."""
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)  # ReLU
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)  # ReLU
        z3 = h2 @ self.w3 + self.b3  # Raw scores

        self._cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3}
        return z3

    def select_action(self, x: np.ndarray, temperature: float = 1.0,
                      strategy_bias: np.ndarray = None,
                      intuition_bias: np.ndarray = None) -> int:
        """Select action via softmax with temperature and optional biases."""
        scores = self.forward(x)
        if strategy_bias is not None:
            scores = scores + strategy_bias  # Strong: up to ±3.0 from DELEGATE
        if intuition_bias is not None:
            scores = scores + intuition_bias  # Soft: up to ±0.3 from convergence
        # Temperature-scaled softmax
        t = max(0.1, temperature)
        exp_scores = np.exp((scores - scores.max()) / t)
        probs = exp_scores / (exp_scores.sum() + 1e-8)
        return int(np.random.choice(NUM_ACTIONS, p=probs))

    def train_step(self, x: np.ndarray, target_action: int,
                   advantage: float) -> float:
        """Train via policy gradient (REINFORCE-style).

        advantage > 0: encourage this action
        advantage < 0: discourage this action
        """
        scores = self.forward(x)

        # Softmax probabilities
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-8)

        # Policy gradient: d_loss/d_score = probs - one_hot(target) * sign(advantage)
        target_one_hot = np.zeros(NUM_ACTIONS)
        target_one_hot[target_action] = 1.0

        # Gradient of cross-entropy weighted by advantage
        d_z3 = (probs - target_one_hot) * abs(advantage)
        if advantage < 0:
            d_z3 = -d_z3  # Flip to discourage

        # Backprop through layers
        d_w3 = self._cache["h2"].reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_b3 = d_z3

        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (self._cache["z2"] > 0)  # ReLU gradient
        d_w2 = self._cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2

        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (self._cache["z1"] > 0)
        d_w1 = self._cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        # Gradient clipping
        for g in [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]:
            np.clip(g, -5.0, 5.0, out=g)

        # SGD update
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3

        self.total_updates += 1
        loss = -np.log(probs[target_action] + 1e-8) * abs(advantage)
        self.last_loss = float(loss)
        return float(loss)

    def save(self, path: str) -> None:
        """Save weights to JSON."""
        data = {
            "input_dim": self.input_dim, "h1": self.h1, "h2": self.h2,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "total_updates": self.total_updates, "last_loss": self.last_loss,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load weights. Returns True if loaded, False if file missing."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            saved_dim = data.get("input_dim", self.input_dim)
            if saved_dim != self.input_dim:
                logger.warning("[Reasoning] Input dim mismatch: saved=%d current=%d — migrating",
                               saved_dim, self.input_dim)
                self._migrate_weights(data, saved_dim)
            else:
                self.w1 = np.array(data["w1"])
                self.b1 = np.array(data["b1"])
                self.w2 = np.array(data["w2"])
                self.b2 = np.array(data["b2"])
                self.w3 = np.array(data["w3"])
                self.b3 = np.array(data["b3"])
            self.total_updates = data.get("total_updates", 0)
            self.last_loss = data.get("last_loss", 0.0)
            return True
        except Exception as e:
            logger.error("[Reasoning] Failed to load policy net: %s", e)
            return False

    def _migrate_weights(self, data: dict, saved_dim: int) -> None:
        """Migrate weights when input dimension changes."""
        old_w1 = np.array(data["w1"])
        old_b1 = np.array(data["b1"])
        old_h1 = old_w1.shape[1]

        # Preserve old rows, Xavier-init new rows scaled by 0.1
        scale = math.sqrt(2.0 / self.input_dim) * 0.1
        new_w1 = np.random.randn(self.input_dim, self.h1) * scale
        copy_rows = min(saved_dim, self.input_dim)
        copy_cols = min(old_h1, self.h1)
        new_w1[:copy_rows, :copy_cols] = old_w1[:copy_rows, :copy_cols]
        self.w1 = new_w1

        new_b1 = np.zeros(self.h1)
        new_b1[:copy_cols] = old_b1[:copy_cols]
        self.b1 = new_b1

        # w2, b2 — preserve if hidden dims match
        old_w2 = np.array(data["w2"])
        old_b2 = np.array(data["b2"])
        if old_h1 == self.h1:
            self.w2 = old_w2
            self.b2 = old_b2
        else:
            scale2 = math.sqrt(2.0 / self.h1) * 0.1
            new_w2 = np.random.randn(self.h1, self.h2) * scale2
            cr = min(old_h1, self.h1)
            cc = min(old_w2.shape[1], self.h2)
            new_w2[:cr, :cc] = old_w2[:cr, :cc]
            self.w2 = new_w2
            new_b2 = np.zeros(self.h2)
            new_b2[:cc] = old_b2[:cc]
            self.b2 = new_b2

        # w3, b3 — output dim is always NUM_ACTIONS, try to preserve
        old_w3 = np.array(data["w3"])
        old_b3 = np.array(data["b3"])
        if old_w3.shape[0] == self.h2:
            self.w3 = old_w3
            self.b3 = old_b3
        else:
            scale3 = math.sqrt(2.0 / self.h2) * 0.1
            new_w3 = np.random.randn(self.h2, NUM_ACTIONS) * scale3
            cr = min(old_w3.shape[0], self.h2)
            new_w3[:cr, :] = old_w3[:cr, :]
            self.w3 = new_w3
            self.b3 = old_b3.copy() if len(old_b3) == NUM_ACTIONS else np.zeros(NUM_ACTIONS)

        logger.info("[Reasoning] Migrated policy net: %dD→%dD, h1=%d→%d",
                    saved_dim, self.input_dim, old_h1, self.h1)


# ── Cognitive EMA ─────────────────────────────────────────────────


class CognitivePerception:
    """EMA-filtered neuromod perception for Body/Mind/Spirit levels.

    Body: raw (α=1.0) — reactive, per-tick
    Mind: smoothed (α=0.05) — stable for reasoning chains
    Spirit: very smooth (α=0.01) — patient, trend-aware

    Validated with real data (2026-03-27):
    GABA raw 0.16-0.77 → Mind ~0.35-0.55 → Spirit ~0.40-0.50
    """

    NEUROMOD_NAMES = ["DA", "5-HT", "NE", "ACh", "Endorphin", "GABA"]

    def __init__(self, mind_alpha: float = 0.05, spirit_alpha: float = 0.01):
        self.mind_alpha = mind_alpha
        self.spirit_alpha = spirit_alpha

        # Initialize at sensible defaults
        self.body: dict[str, float] = {nm: 0.5 for nm in self.NEUROMOD_NAMES}
        self.mind: dict[str, float] = {nm: 0.5 for nm in self.NEUROMOD_NAMES}
        self.spirit: dict[str, float] = {nm: 0.5 for nm in self.NEUROMOD_NAMES}
        self._initialized = False

    def update(self, raw_levels: dict[str, float]) -> None:
        """Update all three perception levels from raw neuromod values."""
        for nm in self.NEUROMOD_NAMES:
            raw = float(raw_levels.get(nm, 0.5))
            raw = max(0.0, min(1.0, raw))

            # Body = raw (no smoothing)
            self.body[nm] = raw

            if not self._initialized:
                # First update: set all levels to raw
                self.mind[nm] = raw
                self.spirit[nm] = raw
            else:
                # EMA smoothing
                self.mind[nm] = self.mind[nm] * (1.0 - self.mind_alpha) + raw * self.mind_alpha
                self.spirit[nm] = self.spirit[nm] * (1.0 - self.spirit_alpha) + raw * self.spirit_alpha

        self._initialized = True

    def get_mind_vector(self) -> np.ndarray:
        """6D vector of Mind-smoothed neuromod levels."""
        return np.array([self.mind[nm] for nm in self.NEUROMOD_NAMES])

    def get_reasoning_persistence(self) -> float:
        """5-HT/GABA ratio — determines reasoning style.

        High (>1.5): patient, thorough exploration
        Low (<0.8): quick, decisive — cuts chains short
        Balanced (~1.0): normal reasoning depth
        """
        sht = self.mind.get("5-HT", 0.5)
        gaba = self.mind.get("GABA", 0.5)
        return sht / (gaba + 0.1)

    def get_reasoning_temperature(self) -> float:
        """Neuromod-derived temperature for action selection.

        High DA → more exploratory (higher temperature)
        High GABA → more conservative (lower temperature)
        High NE → more focused (lower temperature)
        """
        da = self.mind.get("DA", 0.5)
        gaba = self.mind.get("GABA", 0.5)
        ne = self.mind.get("NE", 0.5)
        # Range roughly [0.3, 2.0]
        return 0.5 + da * 1.0 - gaba * 0.3 - ne * 0.2

    def get_state(self) -> dict:
        """Serialize for hot-reload."""
        return {
            "mind": dict(self.mind),
            "spirit": dict(self.spirit),
            "initialized": self._initialized,
        }

    def restore_state(self, state: dict) -> None:
        """Restore from hot-reload."""
        if state.get("initialized"):
            self.mind = state.get("mind", self.mind)
            self.spirit = state.get("spirit", self.spirit)
            self._initialized = True


# ── Spirit Observer ───────────────────────────────────────────────


class SpiritReasoningObserver:
    """Spirit watches reasoning quality. Gentle nudges only (max 2%).

    Spirit perceives neuromods through very slow EMA (α=0.01),
    sees long-term trends, not tick-by-tick noise.
    """

    def __init__(self, max_nudge: float = 0.02):
        self.max_nudge = max_nudge
        self._last_chains: list[list[str]] = []  # recent chain patterns
        self._nudge_count = 0
        self._positive_nudges = 0
        self._negative_nudges = 0
        self._call_count = 0  # total observe() invocations (observability)

    def observe(self, reasoning_state: dict) -> float:
        """Observe reasoning and return nudge [-max, +max].

        Returns:
            0.0: silent observation (default)
            +max: "this feels right, keep going"
            -max: "something is off, try different"
        """
        self._call_count += 1
        chain = reasoning_state.get("chain", [])
        confidence = reasoning_state.get("confidence", 0.5)
        gut_agreement = reasoning_state.get("gut_agreement", 0.5)
        chain_length = len(chain)

        nudge = 0.0
        nudge_reason = ""

        # Stuck detection: same primitive repeated within current chain.
        # Recalibrated 2026-04-08: chains in current operating mode rarely
        # exceed 3 steps. Old threshold (chain≥6, last_3==prev_3) was
        # structurally unreachable. New rule fires when the latest two
        # primitives are identical at chain length ≥3 — the smallest
        # reliable repetition signal in a 3-step chain.
        if chain_length >= 3 and chain[-1] == chain[-2]:
            nudge = -self.max_nudge  # "try something different"
            nudge_reason = "stuck"
            self._negative_nudges += 1

        # Exhaustion detection: established chain with sub-baseline confidence.
        # Recalibrated 2026-04-08: chains rarely exceed 3 steps so the old
        # chain>8 threshold never fired. New rule fires when chain has at
        # least 2 steps and confidence is below the 0.5 baseline.
        elif chain_length >= 2 and confidence < 0.45:
            nudge = -self.max_nudge  # "rest, let dreaming consolidate"
            nudge_reason = "exhausted"
            self._negative_nudges += 1

        # Success reinforcement: clearly good outcome.
        # Recalibrated 2026-04-08: actual operating range is conf 0.50-0.70,
        # gut 0.30-0.45. Old thresholds (conf>0.7 AND gut>0.6) were never
        # met. New thresholds fire on the high end of the actual range —
        # intentionally tighter than negative branches (asymmetric: easier
        # to correct, harder to reinforce, biases toward exploration).
        elif confidence > 0.60 and gut_agreement > 0.40:
            nudge = +self.max_nudge  # "this feels right"
            nudge_reason = "success"
            self._positive_nudges += 1

        # Body distress during reasoning (would be checked externally)
        # Spirit just flags it — the actual veto is in Willing

        if nudge != 0.0:
            self._nudge_count += 1
            # Track chain pattern for repetition detection
            self._last_chains.append(list(chain))
            if len(self._last_chains) > 10:
                self._last_chains.pop(0)
            logger.info(
                "[SpiritObs] NUDGE %+.4f reason=%s conf=%.3f gut=%.3f chain_len=%d",
                nudge, nudge_reason, confidence, gut_agreement, chain_length,
            )

        # Periodic observability — exposes silent-observer mode (call count >> nudge count)
        if self._call_count % 200 == 0:
            logger.info(
                "[SpiritObs] %d obs | %d nudges (%d+ %d-) | sample: chain_len=%d conf=%.2f gut=%.2f",
                self._call_count, self._nudge_count, self._positive_nudges,
                self._negative_nudges, chain_length, confidence, gut_agreement,
            )

        return nudge

    def get_stats(self) -> dict:
        return {
            "total_nudges": self._nudge_count,
            "positive": self._positive_nudges,
            "negative": self._negative_nudges,
            "call_count": self._call_count,
        }

    def get_state(self) -> dict:
        return {
            "nudge_count": self._nudge_count,
            "positive": self._positive_nudges,
            "negative": self._negative_nudges,
            "call_count": self._call_count,
        }

    def restore_state(self, state: dict) -> None:
        self._nudge_count = state.get("nudge_count", 0)
        self._positive_nudges = state.get("positive", 0)
        self._negative_nudges = state.get("negative", 0)
        self._call_count = state.get("call_count", 0)


# ── Reasoning Transition Buffer ───────────────────────────────────


class ReasoningTransitionBuffer:
    """Stores reasoning transitions for IQL training during dreams.

    Each transition: (state, action, reward, next_state, done)
    where state = policy network input vector.
    """

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self._states: list[list[float]] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._next_states: list[list[float]] = []
        self._dones: list[bool] = []

    def record(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Record one transition."""
        self._states.append(state.tolist())
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state.tolist())
        self._dones.append(done)

        # Circular buffer
        if len(self._states) > self.max_size:
            self._states.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._next_states.pop(0)
            self._dones.pop(0)

    def sample(self, batch_size: int = 16) -> Optional[tuple]:
        """Random sample for training. Returns None if buffer too small."""
        n = len(self._states)
        if n < batch_size:
            return None
        indices = np.random.choice(n, size=batch_size, replace=False)
        return (
            [self._states[i] for i in indices],
            [self._actions[i] for i in indices],
            [self._rewards[i] for i in indices],
            [self._next_states[i] for i in indices],
            [self._dones[i] for i in indices],
        )

    def update_last_reward(self, reward: float) -> None:
        """Update reward of most recent transition (credit assignment)."""
        if self._rewards:
            self._rewards[-1] = reward

    def size(self) -> int:
        return len(self._states)

    def save(self, path: str) -> None:
        data = {
            "states": self._states[-500:],  # Keep last 500 for file size
            "actions": self._actions[-500:],
            "rewards": self._rewards[-500:],
            "next_states": self._next_states[-500:],
            "dones": self._dones[-500:],
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
            self._actions = data.get("actions", [])
            self._rewards = data.get("rewards", [])
            self._next_states = data.get("next_states", [])
            self._dones = data.get("dones", [])
            return True
        except Exception as e:
            logger.error("[Reasoning] Failed to load buffer: %s", e)
            return False


# ── The Seven Logic Primitives ────────────────────────────────────


def _primitive_compare(observation: np.ndarray, working_memory: list,
                       mind_neuromods: dict) -> dict:
    """COMPARE: Detect similarity/difference between state regions.

    ACh (attention) modulates granularity — high ACh → finer distinctions.
    Compares inner Trinity [0:15] vs outer Trinity [15:30] — alignment
    between internal state and external engagement.
    """
    ach = mind_neuromods.get("ACh", 0.5)
    threshold = 0.3 + (1.0 - ach) * 0.4  # High ACh → lower threshold → finer

    # Compare inner vs outer regions of Tier 1 observation
    if len(observation) >= 30:
        inner_region = observation[:15]
        outer_region = observation[15:30]
        # Use L2 difference as primary metric (more robust than cosine for mixed-scale data)
        difference = float(np.linalg.norm(inner_region - outer_region))
        # Normalized difference: scale by region magnitude for interpretability
        avg_mag = (float(np.linalg.norm(inner_region)) + float(np.linalg.norm(outer_region))) / 2 + 1e-8
        norm_diff = difference / avg_mag
        # Cosine similarity as secondary
        dot = float(np.dot(inner_region, outer_region))
        norm_i = float(np.linalg.norm(inner_region)) + 1e-8
        norm_o = float(np.linalg.norm(outer_region)) + 1e-8
        similarity = dot / (norm_i * norm_o)
        # Significant when there's meaningful divergence OR alignment
        # Either high similarity (inner/outer aligned) or high difference (divergence)
        significant = abs(similarity) > threshold or norm_diff > 0.5
    else:
        similarity = 0.5
        difference = 0.0
        norm_diff = 0.0
        significant = False

    return {
        "primitive": "COMPARE",
        "similarity": round(similarity, 4),
        "difference": round(difference, 4),
        "norm_diff": round(norm_diff, 4) if norm_diff else 0.0,
        "threshold": round(threshold, 4),
        "significant": significant,
    }


def _primitive_if_then(observation: np.ndarray, working_memory: list,
                       mind_neuromods: dict) -> dict:
    """IF_THEN: Conditional check based on observation thresholds.

    DA (reward) drives acting on conclusions.
    GABA (inhibition) gates: "strong enough evidence to act?"
    """
    da = mind_neuromods.get("DA", 0.5)
    gaba = mind_neuromods.get("GABA", 0.5)
    act_threshold = 0.3 + gaba * 0.4  # High GABA → harder to trigger action

    # Check recent working memory for actionable conclusions
    has_significant = any(
        item.get("content", {}).get("significant", False)
        for item in working_memory
    )

    # DA drives motivation to act
    motivation = da * 0.6 + (0.4 if has_significant else 0.0)

    return {
        "primitive": "IF_THEN",
        "condition_met": motivation > act_threshold,
        "motivation": round(motivation, 4),
        "threshold": round(act_threshold, 4),
        "action": "proceed" if motivation > act_threshold else "wait",
    }


def _primitive_sequence(observation: np.ndarray, working_memory: list,
                        mind_neuromods: dict) -> dict:
    """SEQUENCE: Ordered observation of multiple state aspects.

    5-HT (patience) determines how many steps we complete.
    """
    sht = mind_neuromods.get("5-HT", 0.5)
    max_steps = max(2, int(2 + sht * 3))  # 2-5 steps based on patience

    # Observe state aspects in sequence
    aspects = []
    labels = ["body_state", "mind_state", "temporal", "neuro", "dynamics"]
    slices = [(0, 15), (15, 30), (30, 55), (55, 67), (67, 75)]

    for i, (label, (start, end)) in enumerate(zip(labels, slices)):
        if i >= max_steps:
            break
        if end <= len(observation):
            region = observation[start:end]
            aspects.append({
                "label": label,
                "mean": round(float(np.mean(region)), 4),
                "std": round(float(np.std(region)), 4),
                "max_idx": int(np.argmax(region)),
            })

    return {
        "primitive": "SEQUENCE",
        "steps_completed": len(aspects),
        "max_steps": max_steps,
        "aspects": aspects,
    }


def _primitive_associate(observation: np.ndarray, working_memory: list,
                         mind_neuromods: dict, mini_registry=None) -> dict:
    """ASSOCIATE: Find most relevant working memory item OR mini-reasoner insight.

    DA (reward) + Endorphin (recognition) drive association quality.
    Hierarchical search: working memory first, then mini-experience summaries.
    """
    endorphin = mind_neuromods.get("Endorphin", 0.5)

    # Search working memory
    best_item = None
    best_relevance = 0.0
    best_source = "none"
    for item in (working_memory or []):
        strength = item.get("strength", 0.0)
        relevance = strength * (0.5 + endorphin * 0.5)
        if relevance > best_relevance:
            best_relevance = relevance
            best_item = item
            best_source = "working_memory"

    # Search mini-reasoner summaries (hierarchical ASSOCIATE)
    if mini_registry:
        for domain, summary in mini_registry.query_all().items():
            if summary.get("ticks", 0) == 0:
                continue
            mini_rel = summary.get("relevance", 0.0) * (0.5 + endorphin * 0.5)
            if mini_rel > best_relevance:
                best_relevance = mini_rel
                best_item = {"type": f"mini_{domain}", "key": summary.get("primitive", ""),
                             "strength": mini_rel, "summary": summary}
                best_source = f"mini_{domain}"

    return {
        "primitive": "ASSOCIATE",
        "found": best_item is not None,
        "item_type": best_item.get("type", "unknown") if best_item else None,
        "item_key": best_item.get("key", "unknown") if best_item else None,
        "relevance": round(best_relevance, 4),
        "eureka": best_relevance > 0.7,
        "source": best_source,
    }


def _primitive_decompose(observation: np.ndarray, working_memory: list,
                         mind_neuromods: dict) -> dict:
    """DECOMPOSE: Split observation into labeled sub-vectors.

    NE (analytical alertness) determines decomposition depth.
    """
    ne = mind_neuromods.get("NE", 0.5)

    parts = {}
    for label, (start, end) in DECOMPOSE_LABELS.items():
        if end <= len(observation):
            region = observation[start:end]
            parts[label] = {
                "mean": round(float(np.mean(region)), 4),
                "energy": round(float(np.sum(region ** 2)), 4),
                "active_dims": int(np.sum(region > 0.5)),
                "total_dims": end - start,
            }

    # Higher NE → also compute cross-region correlations
    cross_correlations = {}
    if ne > 0.6 and "body" in parts and "mind" in parts:
        body_r = observation[DECOMPOSE_LABELS["body"][0]:DECOMPOSE_LABELS["body"][1]]
        mind_r = observation[DECOMPOSE_LABELS["mind"][0]:DECOMPOSE_LABELS["mind"][1]]
        min_len = min(len(body_r), len(mind_r))
        if min_len > 0:
            corr = float(np.corrcoef(body_r[:min_len], mind_r[:min_len])[0, 1])
            if not math.isnan(corr):
                cross_correlations["body_mind"] = round(corr, 4)

    return {
        "primitive": "DECOMPOSE",
        "parts": parts,
        "depth": "deep" if ne > 0.6 else "surface",
        "cross_correlations": cross_correlations,
    }


def _primitive_loop(observation: np.ndarray, working_memory: list,
                    mind_neuromods: dict, last_result: Optional[dict] = None) -> dict:
    """LOOP: Re-evaluate last result, looking for change.

    GABA (stop) + 5-HT (persist) ratio determines whether to continue.
    """
    gaba = mind_neuromods.get("GABA", 0.5)
    sht = mind_neuromods.get("5-HT", 0.5)

    should_continue = sht > (gaba + 0.1)  # Persistence > inhibition

    return {
        "primitive": "LOOP",
        "continue": should_continue,
        "persistence_ratio": round(sht / (gaba + 0.1), 4),
        "last_primitive": last_result.get("primitive", "none") if last_result else "none",
        "iteration": 1,  # Incremented externally if looping
    }


def _primitive_negate(observation: np.ndarray, working_memory: list,
                      mind_neuromods: dict, last_result: Optional[dict] = None) -> dict:
    """NEGATE: Invert the last hypothesis.

    Pure GABA-driven: "NOT this path — try the opposite."
    """
    gaba = mind_neuromods.get("GABA", 0.5)

    inverted = {}
    if last_result:
        # Invert boolean conclusions
        for key, val in last_result.items():
            if isinstance(val, bool):
                inverted[key] = not val
            elif isinstance(val, (int, float)) and key in ("similarity", "motivation", "relevance"):
                inverted[key] = round(1.0 - val, 4)

    return {
        "primitive": "NEGATE",
        "inverted_fields": inverted,
        "negation_strength": round(gaba, 4),
        "original_primitive": last_result.get("primitive", "none") if last_result else "none",
    }


# Primitive dispatch table
PRIMITIVE_FUNCTIONS = {
    "COMPARE": _primitive_compare,
    "IF_THEN": _primitive_if_then,
    "SEQUENCE": _primitive_sequence,
    "ASSOCIATE": _primitive_associate,
    "DECOMPOSE": _primitive_decompose,
    "LOOP": _primitive_loop,
    "NEGATE": _primitive_negate,
}


# ── Main Reasoning Engine ─────────────────────────────────────────


class ReasoningEngine:
    """Mind's deliberate cognition through composable logic primitives.

    Integrates:
    - Policy network (selects primitives)
    - Cognitive EMA (stabilized neuromod perception)
    - Spirit observer (gentle nudges)
    - Working memory (reasoning context)
    - Transition buffer (IQL training data)

    Called once per Mind-rate tick in spirit_worker.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        # Core parameters
        self.max_chain_length = cfg.get("max_chain_length", 10)
        self.min_chain_length = cfg.get("min_chain_length", 3)  # Must think before concluding
        self.confidence_threshold = cfg.get("confidence_threshold", 0.6)
        self.policy_input_dim = cfg.get("policy_input_dim", 111)  # 95 base + 16 mini-summaries
        self.policy_h1 = cfg.get("policy_h1", 64)
        self.policy_h2 = cfg.get("policy_h2", 32)
        self.learning_rate = cfg.get("learning_rate", 0.001)

        # Save path
        self.save_dir = cfg.get("save_dir", "data/reasoning")
        os.makedirs(self.save_dir, exist_ok=True)

        # Components
        self.perception = CognitivePerception(
            mind_alpha=cfg.get("mind_ema_alpha", 0.05),
            spirit_alpha=cfg.get("spirit_ema_alpha", 0.01),
        )
        self.observer = SpiritReasoningObserver(
            max_nudge=cfg.get("spirit_max_nudge", 0.02),
        )
        self.policy = ReasoningPolicyNet(
            input_dim=self.policy_input_dim,
            hidden_1=self.policy_h1,
            hidden_2=self.policy_h2,
            learning_rate=self.learning_rate,
        )
        self.buffer = ReasoningTransitionBuffer(
            max_size=cfg.get("buffer_size", 2000),
        )

        # Mini-reasoner registry (set via set_mini_registry)
        self._mini_registry = None

        # Reasoning state
        self.chain: list[str] = []          # Current chain of primitives executed
        self.chain_results: list[dict] = [] # Results from each primitive
        self.confidence: float = 0.5
        self.gut_agreement: float = 0.5
        self.is_active: bool = False
        self.spirit_nudge: float = 0.0
        self._last_result: Optional[dict] = None
        self._loop_count: int = 0
        self._total_chains: int = 0
        self._total_conclusions: int = 0
        self._total_reasoning_steps: int = 0
        self._chain_start_time: float = 0.0
        self._strategy_bias: Optional[np.ndarray] = None  # 8D, set by meta-reasoning DELEGATE
        self._intuition_bias: Optional[np.ndarray] = None  # 8D, set by vertical convergence (M12)

        # Load saved state
        self.policy.load(os.path.join(self.save_dir, "policy_net.json"))
        self.buffer.load(os.path.join(self.save_dir, "buffer.json"))
        # 2026-04-13: persist lifetime totals across restarts so commit_rate,
        # historical observability, and the "dormant reasoning" bug don't
        # reappear after every spirit/titan_main restart. Previously only
        # policy+buffer were saved, losing chain counters on every reboot.
        self._load_totals()

    def tick(self, observation: np.ndarray, gut_signals: dict,
             body_state: dict, raw_neuromods: dict,
             working_memory_items: list, dt: float = 1.0) -> dict:
        """One reasoning step per Mind-rate tick.

        Args:
            observation: enriched observation vector (75-79D)
            gut_signals: {program_name: last_urgency} from NS programs
            body_state: {fatigue, chi_total, metabolic_drain, is_dreaming}
            raw_neuromods: {DA, 5-HT, NE, ACh, Endorphin, GABA}
            working_memory_items: list of dicts from working_memory.get_context()
            dt: delta time (for dt-parameterized operations)

        Returns:
            dict with action ("CONTINUE", "COMMIT", "HOLD", "ABANDON") and details
        """
        # Don't reason during dreams
        if body_state.get("is_dreaming", False):
            if self.is_active:
                return self._abandon_chain("dreaming")
            return {"action": "IDLE", "reason": "dreaming"}

        # 1. Update cognitive perception (EMA smoothing)
        self.perception.update(raw_neuromods)

        # 2. Check METABOLISM energy budget
        metabolism_urgency = gut_signals.get("METABOLISM", 0.0)
        max_chain = self.max_chain_length
        if metabolism_urgency > 0.7:
            max_chain = 3  # Conserve energy
        elif metabolism_urgency > 0.4:
            max_chain = int(self.max_chain_length * (1.0 - metabolism_urgency))

        # 3. Build policy input
        policy_input = self._build_policy_input(
            observation, gut_signals, working_memory_items)

        # 4. Start new chain if not active
        if not self.is_active:
            # Check if there's enough stimulus to start reasoning
            if not self._should_start_reasoning(gut_signals, body_state):
                return {"action": "IDLE", "reason": "no_stimulus"}
            self._start_chain()

        # 5. Select and execute primitive
        temperature = self.perception.get_reasoning_temperature()
        action_idx = self.policy.select_action(policy_input, temperature,
                                               strategy_bias=self._strategy_bias,
                                               intuition_bias=self._intuition_bias)
        action_name = PRIMITIVES[action_idx]

        # 5b. Enforce minimum chain length — redirect premature CONCLUDE
        if action_name == "CONCLUDE" and len(self.chain) < self.min_chain_length:
            # Pick a non-CONCLUDE primitive (exclude idx 7=CONCLUDE)
            non_conclude = [i for i in range(NUM_ACTIONS - 1)]
            action_idx = int(np.random.choice(non_conclude))
            action_name = PRIMITIVES[action_idx]

        # 6. Handle CONCLUDE
        if action_name == "CONCLUDE" or len(self.chain) >= max_chain:
            return self._conclude_chain(observation, gut_signals, body_state, policy_input)

        # 7. Execute the selected primitive
        mind_nm = dict(self.perception.mind)

        if action_name == "LOOP":
            result = _primitive_loop(observation, working_memory_items,
                                     mind_nm, self._last_result)
            if not result["continue"] or self._loop_count >= 3:
                # GABA says stop or max loops reached — treat as no-op
                result["continue"] = False
            else:
                self._loop_count += 1
        elif action_name == "NEGATE":
            result = _primitive_negate(observation, working_memory_items,
                                       mind_nm, self._last_result)
        elif action_name == "ASSOCIATE":
            result = _primitive_associate(observation, working_memory_items,
                                          mind_nm, mini_registry=self._mini_registry)
        else:
            prim_fn = PRIMITIVE_FUNCTIONS.get(action_name)
            if prim_fn:
                result = prim_fn(observation, working_memory_items, mind_nm)
            else:
                result = {"primitive": action_name, "error": "unknown"}

        # 8. Record in chain
        self.chain.append(action_name)
        self.chain_results.append(result)
        self._last_result = result
        self._total_reasoning_steps += 1

        if action_name != "LOOP":
            self._loop_count = 0  # Reset loop counter after non-loop primitive

        # 9. Update confidence (Mind Feeling)
        self._update_confidence(result, gut_signals)

        # 10. Spirit observes
        self.spirit_nudge = self.observer.observe(self.get_reasoning_state())

        # 11. Record transition for IQL
        next_input = self._build_policy_input(
            observation, gut_signals, working_memory_items)
        self.buffer.record(
            state=policy_input, action=action_idx, reward=0.0,
            next_state=next_input, done=False,
        )

        return {
            "action": "CONTINUE",
            "primitive": action_name,
            "result": result,
            "chain_length": len(self.chain),
            "confidence": round(self.confidence, 4),
            "gut_agreement": round(self.gut_agreement, 4),
            "spirit_nudge": round(self.spirit_nudge, 4),
            "persistence": round(self.perception.get_reasoning_persistence(), 4),
        }

    def _should_start_reasoning(self, gut_signals: dict, body_state: dict) -> bool:
        """Determine if there's enough stimulus to begin reasoning.

        Reasoning starts when:
        - Any NS program has high urgency (gut feeling says "pay attention")
        - Or METABOLISM signals energy availability
        - AND body is not too fatigued
        """
        if body_state.get("fatigue", 0) > 0.6:
            return False
        if body_state.get("chi_total", 0.5) < 0.25:
            return False

        # Start reasoning when any gut signal present or enough ticks have passed
        # After dimension migration, urgencies are low (new weights scaled 0.1×)
        # Use a low threshold that allows reasoning to begin learning
        max_gut = max(gut_signals.values()) if gut_signals else 0.0
        if max_gut > 0.15:
            return True
        # Fallback: reason periodically even without gut signals
        # Uses a tick counter (incremented each call) — every ~20 ticks (~70s)
        if not hasattr(self, '_idle_ticks'):
            self._idle_ticks = 0
        self._idle_ticks += 1
        if self._idle_ticks >= 20:
            self._idle_ticks = 0
            return True
        return False

    def _start_chain(self) -> None:
        """Initialize a new reasoning chain."""
        self.chain = []
        self.chain_results = []
        self.confidence = 0.5
        self.gut_agreement = 0.5
        self.is_active = True
        self._last_result = None
        self._loop_count = 0
        self._chain_start_time = time.time()
        self._total_chains += 1

    def _conclude_chain(self, observation: np.ndarray, gut_signals: dict,
                        body_state: dict, policy_input: np.ndarray) -> dict:
        """End reasoning chain — commit, hold, or abandon."""
        # Check Mind Willing: body ready?
        body_ready = self._body_ready(body_state)

        # Compute final gut agreement
        self._update_gut_agreement(gut_signals)

        # Determine action
        if self.confidence >= self.confidence_threshold and body_ready:
            action = "COMMIT"
            self._total_conclusions += 1
        elif not body_ready:
            action = "HOLD"
        else:
            action = "ABANDON"

        # Build conclusion
        conclusion = {
            "action": action,
            "confidence": round(self.confidence, 4),
            "gut_agreement": round(self.gut_agreement, 4),
            "chain_length": len(self.chain),
            "chain": list(self.chain),
            "duration_s": round(time.time() - self._chain_start_time, 2),
            "reasoning_plan": self._extract_plan(),
        }

        # Terminal reward for IQL
        reward = self._compute_terminal_reward(conclusion, gut_signals)

        # M13: Intuition reward shaping — learn to trust vertical convergence
        if self._intuition_bias is not None and len(self.chain) > 0:
            from titan_plugin.logic.intuition_convergence import (
                compute_intuition_reward_shaping)
            last_action = PRIMITIVES.index(self.chain[-1]) if self.chain[-1] in PRIMITIVES else 0
            _intuit_shape = compute_intuition_reward_shaping(
                action_taken=last_action,
                intuition_bias=self._intuition_bias,
                outcome_score=conclusion["confidence"],
            )
            if abs(_intuit_shape) > 0.001:
                reward = max(0.0, reward + _intuit_shape)
                logger.info("[Reasoning] Intuition shaping: %+.3f (aligned=%s, conf=%.3f)",
                            _intuit_shape,
                            self.chain[-1] == PRIMITIVES[int(np.argmax(self._intuition_bias))]
                            if self._intuition_bias is not None else False,
                            conclusion["confidence"])

        self.buffer.record(
            state=policy_input,
            action=PRIMITIVES.index("CONCLUDE"),
            reward=reward,
            next_state=policy_input,  # Terminal
            done=True,
        )

        # Reset chain state
        self.is_active = False
        self.chain = []
        self.chain_results = []
        self._last_result = None
        self._strategy_bias = None   # Clear meta-reasoning bias on chain end
        self._intuition_bias = None  # Clear vertical convergence bias on chain end

        logger.info("[Reasoning] %s — conf=%.3f gut=%.3f chain=%d dur=%.1fs reward=%.3f",
                    action, conclusion["confidence"], conclusion["gut_agreement"],
                    conclusion["chain_length"], conclusion["duration_s"], reward)

        return conclusion

    def _abandon_chain(self, reason: str) -> dict:
        """Abandon current chain (e.g., dreaming started)."""
        result = {
            "action": "ABANDON",
            "reason": reason,
            "chain_length": len(self.chain),
        }
        self.is_active = False
        self.chain = []
        self.chain_results = []
        self._last_result = None
        self._strategy_bias = None   # Clear meta-reasoning bias on chain end
        self._intuition_bias = None  # Clear vertical convergence bias on chain end
        return result

    def _body_ready(self, body_state: dict) -> bool:
        """Check if Body can support action execution.

        Uses chi_total and metabolic_drain (NOT chi_circulation which is
        structurally near-zero at 0.29Hz tick rate until Schumann fix).
        """
        if body_state.get("is_dreaming", False):
            return False
        if body_state.get("fatigue", 0) > 0.7:
            return False
        if body_state.get("chi_total", 0.5) < 0.2:
            return False
        if body_state.get("metabolic_drain", 0) > 0.8:
            return False
        return True

    def _update_confidence(self, result: dict, gut_signals: dict) -> None:
        """Update Mind Feeling: confidence in current reasoning direction."""
        primitive = result.get("primitive", "")

        # Strong confidence boosts: clear success signals
        if result.get("significant") or result.get("eureka") or result.get("condition_met"):
            self.confidence = min(1.0, self.confidence + 0.1)
        # Moderate confidence boosts: productive observation
        elif primitive == "DECOMPOSE" and result.get("parts"):
            # DECOMPOSE is productive when it finds active dimensions
            active_total = sum(p.get("active_dims", 0) for p in result["parts"].values())
            if active_total > 5:
                self.confidence = min(1.0, self.confidence + 0.05)
        elif primitive == "SEQUENCE" and result.get("steps_completed", 0) >= 3:
            # SEQUENCE completing 3+ steps means thorough observation
            self.confidence = min(1.0, self.confidence + 0.05)
        elif primitive == "ASSOCIATE" and result.get("found") and result.get("relevance", 0) > 0.3:
            # Found relevant memory item even if not eureka
            self.confidence = min(1.0, self.confidence + 0.03)
        # Confidence drops: failed attempts
        elif result.get("found") is False or result.get("continue") is False:
            self.confidence = max(0.0, self.confidence - 0.05)

        # Spirit nudge influences confidence
        self.confidence = max(0.0, min(1.0,
                              self.confidence + self.spirit_nudge))

        # Update gut agreement
        self._update_gut_agreement(gut_signals)

    def _update_gut_agreement(self, gut_signals: dict) -> None:
        """How well does reasoning align with NS program gut signals?"""
        if not gut_signals:
            self.gut_agreement = 0.5
            return

        # High-urgency programs "agree" when reasoning confidence is high
        # Low-urgency programs "agree" when reasoning is cautious
        urgencies = list(gut_signals.values())
        mean_urgency = sum(urgencies) / len(urgencies)

        # Agreement = how similar reasoning confidence is to gut urgency average
        self.gut_agreement = 1.0 - abs(self.confidence - mean_urgency)

    def _extract_plan(self) -> dict:
        """Extract a reasoning plan from the chain results.

        This plan feeds into L8/L9 language generation.
        """
        plan = {
            "intent": "reflect",
            "elements": [],
            "structure": "observation",
            "confidence": round(self.confidence, 4),
        }

        for result in self.chain_results:
            prim = result.get("primitive", "")
            if prim == "DECOMPOSE":
                parts = result.get("parts", {})
                active_parts = [k for k, v in parts.items()
                                if v.get("active_dims", 0) > v.get("total_dims", 1) * 0.5]
                plan["elements"].extend(active_parts)
                plan["structure"] = "decomposition"
            elif prim == "COMPARE":
                if result.get("significant"):
                    plan["intent"] = "express_observation"
                    plan["structure"] = "comparison"
            elif prim == "ASSOCIATE" and result.get("eureka"):
                plan["intent"] = "express_recognition"
                plan["structure"] = "association"
            elif prim == "IF_THEN" and result.get("condition_met"):
                plan["intent"] = "express_decision"
                plan["structure"] = "conditional"
            elif prim == "SEQUENCE":
                plan["structure"] = "sequential"

        return plan

    def _compute_terminal_reward(self, conclusion: dict, gut_signals: dict) -> float:
        """Compute reward for completed reasoning chain.

        Components:
        - Semantic fidelity (0.30) — did we address the stimulus?
        - Gut agreement (0.20) — intuition-reasoning alignment
        - Exploration (0.15) — reward thinking depth, penalize premature exit
        - Neuromod satisfaction (0.10) — did it feel right?
        """
        confidence = conclusion["confidence"]
        gut = conclusion["gut_agreement"]
        chain_len = conclusion["chain_length"]

        # Semantic fidelity: confidence as proxy
        semantic = confidence * 0.30

        # Gut agreement
        gut_reward = gut * 0.20

        # Exploration: reward chains that actually think (not premature exit)
        # Longer chains up to max get progressively more exploration reward
        chain_depth = min(chain_len, self.max_chain_length)
        exploration = (chain_depth / self.max_chain_length) * 0.15

        # Premature termination penalty: very short chains get penalized
        if chain_len < self.min_chain_length:
            exploration = -0.10  # Negative reward for not even trying

        # Neuromod satisfaction: DA level as proxy for reward signal
        da = self.perception.mind.get("DA", 0.5)
        neuromod = da * 0.10

        # Action bonus
        if conclusion["action"] == "COMMIT":
            action_bonus = 0.25  # Successful commitment
        elif conclusion["action"] == "HOLD":
            action_bonus = 0.10  # Reasonable caution
        else:
            action_bonus = 0.0   # Abandoned

        total = semantic + gut_reward + exploration + neuromod + action_bonus
        return round(max(0.0, total), 4)

    # ── Mini-Reasoner Integration ────────────────────────────────────

    def set_mini_registry(self, registry) -> None:
        """Connect mini-reasoner registry for hierarchical consultation."""
        self._mini_registry = registry

    def set_strategy_bias(self, bias: np.ndarray) -> None:
        """Set strategy bias from meta-reasoning DELEGATE. Cleared on chain end."""
        self._strategy_bias = np.array(bias, dtype=np.float32) if bias is not None else None
        if self._strategy_bias is not None:
            logger.info("[Reasoning] Strategy bias set by meta-reasoning: %s",
                        {PRIMITIVES[i]: round(float(b), 2)
                         for i, b in enumerate(self._strategy_bias) if abs(b) > 0.01})

    def clear_strategy_bias(self) -> None:
        """Clear strategy bias (called after chain concludes)."""
        self._strategy_bias = None

    def set_intuition_bias(self, bias: np.ndarray) -> None:
        """Set soft intuition bias from vertical convergence (M12).

        Much softer than strategy_bias (±0.3 vs ±3.0). Additive with strategy.
        When DELEGATE is active AND intuition agrees, signals reinforce.
        When they disagree, intuition is too soft to override DELEGATE.
        Cleared on chain end.
        """
        self._intuition_bias = np.clip(
            np.array(bias, dtype=np.float32), -0.3, 0.3
        ) if bias is not None else None

    def clear_intuition_bias(self) -> None:
        """Clear intuition bias (called after chain concludes)."""
        self._intuition_bias = None

    def query_mini(self, domain: str = None) -> dict:
        """Query mini-reasoner summaries. None = all domains."""
        if not self._mini_registry:
            return {}
        if domain:
            return self._mini_registry.query(domain)
        return self._mini_registry.query_all()

    def _build_mini_summary_vector(self) -> np.ndarray:
        """Build 16D vector from 4 mini-reasoner summaries (4D each).

        Each domain contributes: [relevance, confidence, primitive_idx_norm, active_flag]
        Order: spatial, observation, language, self_exploration
        """
        vec = np.zeros(16)
        if not self._mini_registry:
            return vec
        domains = ["spatial", "observation", "language", "self_exploration"]
        for i, domain in enumerate(domains):
            summary = self._mini_registry.query(domain)
            if summary and summary.get("ticks", 0) > 0:
                offset = i * 4
                vec[offset] = summary.get("relevance", 0.0)
                vec[offset + 1] = summary.get("confidence", 0.0)
                # Normalize primitive index
                primitives = summary.get("primitives", [None, None, None])
                prim = summary.get("primitive", "")
                prim_idx = primitives.index(prim) if prim in primitives else 0
                vec[offset + 2] = prim_idx / max(1, len(primitives) - 1)
                vec[offset + 3] = 1.0  # Active flag
        return vec

    def _build_policy_input(self, observation: np.ndarray,
                            gut_signals: dict,
                            working_memory_items: list) -> np.ndarray:
        """Build input vector for policy network.

        Components:
        - observation (75-79D from enriched feature set)
        - gut_signals summary (11D: one urgency per NS program)
        - mind_neuromods EMA (6D)
        - chain_state (3D: chain_length_norm, confidence, persistence_ratio)
        """
        parts = [observation]

        # Gut signals — fixed order for 11 programs
        gut_order = ["REFLEX", "FOCUS", "INTUITION", "IMPULSE", "VIGILANCE",
                     "CREATIVITY", "CURIOSITY", "EMPATHY", "INSPIRATION",
                     "REFLECTION", "METABOLISM"]
        gut_vec = np.array([gut_signals.get(p, 0.0) for p in gut_order])
        parts.append(gut_vec)

        # Mind neuromods (EMA smoothed)
        parts.append(self.perception.get_mind_vector())

        # Chain state
        chain_state = np.array([
            len(self.chain) / max(1, self.max_chain_length),  # normalized length
            self.confidence,
            min(2.0, self.perception.get_reasoning_persistence()) / 2.0,  # normalized
        ])
        parts.append(chain_state)

        # Mini-reasoner summaries (16D: 4 domains × 4D each)
        # Appended AFTER existing dimensions — existing policy weights stay valid
        # (pad/truncate below handles backward compat with old policy_input_dim)
        parts.append(self._build_mini_summary_vector())

        combined = np.concatenate(parts)

        # Pad or truncate to expected input dim
        if len(combined) < self.policy_input_dim:
            combined = np.concatenate([combined,
                                       np.zeros(self.policy_input_dim - len(combined))])
        elif len(combined) > self.policy_input_dim:
            combined = combined[:self.policy_input_dim]

        return combined

    def consolidate_training(self, boost_factor: float = 2.0) -> dict:
        """Dream-time training. Called during dream cycles.

        Samples from transition buffer and trains policy network
        with boosted learning rate.
        """
        sample = self.buffer.sample(batch_size=32)
        if sample is None:
            return {"trained": False, "reason": "insufficient_data",
                    "buffer_size": self.buffer.size()}

        states, actions, rewards, next_states, dones = sample

        # Boost LR temporarily
        original_lr = self.policy.lr
        self.policy.lr = min(original_lr * boost_factor, original_lr * 3.0)

        total_loss = 0.0
        for i in range(len(states)):
            state = np.array(states[i])
            action = actions[i]
            reward = rewards[i]

            # Advantage: reward - baseline (0.3 as typical reward)
            advantage = reward - 0.3
            loss = self.policy.train_step(state, action, advantage)
            total_loss += loss

        # Restore LR
        self.policy.lr = original_lr

        avg_loss = total_loss / len(states)
        logger.info("[Reasoning] Dream training: %d samples, avg_loss=%.4f, buffer=%d",
                    len(states), avg_loss, self.buffer.size())

        return {
            "trained": True,
            "samples": len(states),
            "avg_loss": round(avg_loss, 6),
            "buffer_size": self.buffer.size(),
            "total_updates": self.policy.total_updates,
        }

    def save_all(self) -> None:
        """Persist policy network, transition buffer, and lifetime totals."""
        self.policy.save(os.path.join(self.save_dir, "policy_net.json"))
        self.buffer.save(os.path.join(self.save_dir, "buffer.json"))
        # 2026-04-13: persist reasoning-engine lifetime counters so they
        # survive restarts. save_all previously dropped these — which meant
        # commit_rate shown in social posts was session-lifetime only and
        # often looked like "1% dormant" after a fresh boot.
        self._save_totals()

    def _save_totals(self) -> None:
        """Persist _total_chains / _total_conclusions / _total_reasoning_steps
        to reasoning_totals.json. Small file (3 ints + timestamp), atomic
        replace. Called from save_all() on every persist cycle."""
        try:
            path = os.path.join(self.save_dir, "reasoning_totals.json")
            data = {
                "version": 1,
                "saved_ts": time.time(),
                "total_chains": int(self._total_chains),
                "total_conclusions": int(self._total_conclusions),
                "total_reasoning_steps": int(self._total_reasoning_steps),
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("[Reasoning] _save_totals failed: %s", e)

    def _load_totals(self) -> None:
        """Restore lifetime counters from reasoning_totals.json. Safe on
        first boot (missing file → counters stay at 0). Logs if file was
        written by a different schema version."""
        try:
            path = os.path.join(self.save_dir, "reasoning_totals.json")
            if not os.path.exists(path):
                return
            with open(path) as f:
                data = json.load(f)
            if int(data.get("version", 0)) != 1:
                logger.warning(
                    "[Reasoning] _load_totals: unexpected schema v%s — "
                    "ignoring", data.get("version"))
                return
            self._total_chains = int(data.get("total_chains", 0))
            self._total_conclusions = int(data.get("total_conclusions", 0))
            self._total_reasoning_steps = int(
                data.get("total_reasoning_steps", 0))
            logger.info(
                "[Reasoning] Restored lifetime totals: chains=%d "
                "conclusions=%d steps=%d (rate=%.1f%%)",
                self._total_chains, self._total_conclusions,
                self._total_reasoning_steps,
                100.0 * self._total_conclusions
                / max(1, self._total_chains))
        except Exception as e:
            logger.warning("[Reasoning] _load_totals failed: %s", e)

    def get_reasoning_state(self) -> dict:
        """Current reasoning state for Spirit observer and API."""
        return {
            "is_active": self.is_active,
            "chain": list(self.chain),
            "chain_length": len(self.chain),
            "confidence": self.confidence,
            "gut_agreement": self.gut_agreement,
            "spirit_nudge": self.spirit_nudge,
            "persistence": self.perception.get_reasoning_persistence(),
        }

    def get_observation_features(self) -> dict:
        """Reasoning features for observation space T6 extension."""
        return {
            "is_active": 1.0 if self.is_active else 0.0,
            "chain_length_norm": len(self.chain) / max(1, self.max_chain_length),
            "confidence": self.confidence,
            "gut_agreement": self.gut_agreement,
        }

    def get_stats(self) -> dict:
        """Stats for API/monitoring."""
        return {
            "total_chains": self._total_chains,
            "total_conclusions": self._total_conclusions,
            "total_reasoning_steps": self._total_reasoning_steps,
            "is_active": self.is_active,
            "chain_length": len(self.chain),
            "confidence": round(self.confidence, 4),
            "gut_agreement": round(self.gut_agreement, 4),
            "spirit_nudge": round(self.spirit_nudge, 4),
            "persistence": round(self.perception.get_reasoning_persistence(), 4),
            "buffer_size": self.buffer.size(),
            "policy_updates": self.policy.total_updates,
            "policy_loss": round(self.policy.last_loss, 6),
            "spirit_observer": self.observer.get_stats(),
            "mind_neuromods": {k: round(v, 4) for k, v in self.perception.mind.items()},
        }

    def get_state(self) -> dict:
        """Full state for hot-reload preservation."""
        return {
            "perception": self.perception.get_state(),
            "observer": self.observer.get_state(),
            "total_chains": self._total_chains,
            "total_conclusions": self._total_conclusions,
            "total_reasoning_steps": self._total_reasoning_steps,
            "is_active": self.is_active,
            "chain": list(self.chain),
            "confidence": self.confidence,
            "gut_agreement": self.gut_agreement,
        }

    def restore_state(self, state: dict) -> None:
        """Restore from hot-reload."""
        # I-007 fix: explicit logging when reasoning state is restored.
        # Previously this method silently reset counters to 0 if the state
        # dict was missing the keys, which masked the cause of the T1
        # 4→0 chain counter incident on 2026-04-08 (Guardian sub-restart vs
        # genuine state loss couldn't be distinguished).
        prev_chains = self._total_chains
        new_chains = state.get("total_chains", 0)
        prev_conclusions = self._total_conclusions
        new_conclusions = state.get("total_conclusions", 0)

        self.perception.restore_state(state.get("perception", {}))
        self.observer.restore_state(state.get("observer", {}))
        self._total_chains = new_chains
        self._total_conclusions = new_conclusions
        self._total_reasoning_steps = state.get("total_reasoning_steps", 0)
        # Don't restore mid-chain state — start fresh after reload
        self.is_active = False
        self.chain = []
        self.chain_results = []

        if prev_chains > 0 and new_chains < prev_chains:
            logger.warning(
                "[Reasoning] STATE REGRESSION on restore_state: "
                "chains %d → %d, conclusions %d → %d. "
                "Likely missing keys in state dict (defaulted to 0). "
                "Investigate caller — this is the I-007 pattern.",
                prev_chains, new_chains, prev_conclusions, new_conclusions,
            )
        elif prev_chains > 0 or new_chains > 0:
            logger.info(
                "[Reasoning] State restored: chains=%d→%d conclusions=%d→%d",
                prev_chains, new_chains, prev_conclusions, new_conclusions,
            )
