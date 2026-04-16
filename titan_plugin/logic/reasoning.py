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

# rFP β Phase 3 — Primitive-Affinity table for gut formula redesign.
# Each NS program has visceral character that aligns with specific reasoning
# primitives. Gut agreement is no longer the degenerate (1 - confidence)
# formula but a measurement of how well firing programs match the chosen
# primitive's semantic role.
#
# Encoding:
#   1.0 = primary (program's defining cognitive operation)
#   0.6 = secondary (program also resonates with this primitive)
#   0.3 = explicit anti-affinity (program OPPOSES this primitive)
#   not listed = 0.5 (true neutral — no contribution to gut)
#
# Formula: contribution = urgency × (affinity - 0.5) × 2  ∈ [-urgency, +urgency]
# Symmetric by construction (Q4 lock-in 2026-04-16).
PRIMITIVE_AFFINITY = {
    # Inner / autonomic programs
    "REFLEX":      {"CONCLUDE": 1.0},                        # end fast
    "FOCUS":       {"DECOMPOSE": 1.0, "SEQUENCE": 0.6},      # depth-first
    "INTUITION":   {"ASSOCIATE": 1.0, "LOOP": 0.6},          # gut-guided
    "IMPULSE":     {"CONCLUDE": 1.0, "IF_THEN": 0.6},        # act-now or check-then-act
    "METABOLISM":  {"CONCLUDE": 1.0, "LOOP": 0.3},           # end fast; OPPOSES deep loops
    "VIGILANCE":   {"IF_THEN": 1.0, "COMPARE": 0.6},         # condition-checking
    # Outer / personality programs
    "INSPIRATION": {"ASSOCIATE": 1.0, "DECOMPOSE": 0.6},     # find connections
    "CREATIVITY":  {"DECOMPOSE": 1.0, "ASSOCIATE": 0.6, "NEGATE": 0.6},  # generate variations
    "CURIOSITY":   {"DECOMPOSE": 1.0, "IF_THEN": 0.6, "CONCLUDE": 0.3},  # explore; OPPOSES early CONCLUDE
    "EMPATHY":     {"COMPARE": 1.0, "SEQUENCE": 0.6},        # perspective-taking
    "REFLECTION":  {"LOOP": 1.0, "COMPARE": 0.6, "ASSOCIATE": 0.6},  # revisit + connect
}
NEUTRAL_AFFINITY = 0.5  # default for programs with no listed entry for this primitive

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


# ── rFP α Mechanism A: Sequence-Quality Store ─────────────────────
# Per-prefix EMA of terminal outcome score. Keys on tuple(chain[:step_idx]).
# Visit-count-weighted for first N visits (true running mean), then fixed α.
# LRU eviction at cap. No state bucketing in v1 (rFP §2a "start per-primitive").


class SequenceQualityStore:
    """Mechanism A — EMA table mapping primitive-prefix → expected terminal reward.

    Built 2026-04-16 per rFP α §2a. Used to provide intermediate-step rewards
    during reasoning chains: at step t with chain-so-far C[:t], query
    `sequence_quality[C[:t]]` → lookup value multiplied by schedule weight.

    Update cadence: on each chain conclusion, update the EMA for every
    prefix of the completed chain with the terminal reward.

    Count-weighted ramp:
        visits < ramp_cutoff:  ema = running_mean (exact 1/n — trusts early aggregate)
        visits ≥ ramp_cutoff:  ema = α × new + (1-α) × old (fixed rate, responsive)

    Visit gate: query returns None (no signal) for prefixes with < visit_gate
    visits — prevents spurious signal from under-sampled keys.

    LRU eviction: when table exceeds cap, drop least-recently-updated entry.
    Monitored via telemetry; rFP §2a anticipates growth to hundreds-thousands
    of unique prefixes, cap 10k is generous.
    """

    def __init__(self, cap: int = 10000, visit_gate: int = 3,
                 ema_alpha: float = 0.1, ramp_cutoff: int = 20):
        self._table: dict[tuple, dict] = {}  # {seq: {"ema": float, "n": int, "last_ts": float}}
        self._cap = cap
        self._visit_gate = visit_gate
        self._ema_alpha = ema_alpha
        self._ramp_cutoff = ramp_cutoff
        self._evictions = 0  # Counter for telemetry

    def update_chain_prefixes(self, chain: list[str], terminal_reward: float) -> int:
        """Update EMA for every prefix of the completed chain.

        Returns number of prefixes updated.
        """
        if not chain:
            return 0
        now = time.time()
        count = 0
        for i in range(1, len(chain) + 1):
            prefix = tuple(chain[:i])
            self._update_one(prefix, terminal_reward, now)
            count += 1
        self._maybe_evict()
        return count

    def _update_one(self, prefix: tuple, reward: float, ts: float) -> None:
        entry = self._table.get(prefix)
        if entry is None:
            self._table[prefix] = {"ema": float(reward), "n": 1, "last_ts": ts}
            return
        n = entry["n"] + 1
        if n <= self._ramp_cutoff:
            # Count-weighted: exact running mean, trusts aggregate
            entry["ema"] = entry["ema"] + (reward - entry["ema"]) / n
        else:
            # Fixed α EMA: responsive
            entry["ema"] = (1.0 - self._ema_alpha) * entry["ema"] + self._ema_alpha * reward
        entry["n"] = n
        entry["last_ts"] = ts

    def query(self, prefix: tuple) -> Optional[float]:
        """Return EMA if visit_gate satisfied, else None (no signal).

        Updates last_ts for LRU ordering.
        """
        entry = self._table.get(prefix)
        if entry is None or entry["n"] < self._visit_gate:
            return None
        entry["last_ts"] = time.time()
        return float(entry["ema"])

    def query_blended(self, chain_prefix: list[str], last_k: int,
                      horizon_entire: float, horizon_last_k: float) -> float:
        """Horizon-blended query per rFP §3 lock: 0.4 × entire + 0.6 × last_K.

        Returns 0.0 if neither path returns signal (under-visited).
        """
        entire = self.query(tuple(chain_prefix))
        lk_prefix = tuple(chain_prefix[-last_k:]) if len(chain_prefix) >= last_k else tuple(chain_prefix)
        last_k_v = self.query(lk_prefix)
        if entire is None and last_k_v is None:
            return 0.0
        if entire is None:
            return float(last_k_v) * horizon_last_k
        if last_k_v is None:
            return float(entire) * horizon_entire
        return float(entire) * horizon_entire + float(last_k_v) * horizon_last_k

    def _maybe_evict(self) -> None:
        """LRU eviction when over cap. Drops oldest last_ts."""
        if len(self._table) <= self._cap:
            return
        over = len(self._table) - self._cap
        sorted_items = sorted(self._table.items(), key=lambda kv: kv[1]["last_ts"])
        for key, _ in sorted_items[:over]:
            del self._table[key]
            self._evictions += 1

    def stats(self) -> dict:
        gated = sum(1 for e in self._table.values() if e["n"] >= self._visit_gate)
        return {
            "size": len(self._table),
            "cap": self._cap,
            "gated_size": gated,
            "visit_gate": self._visit_gate,
            "evictions": self._evictions,
            "ema_alpha": self._ema_alpha,
        }

    def save(self, path: str) -> None:
        """Save table to JSON. Keys become stringified tuples (json-safe)."""
        data = {
            "schema": 1,
            "visit_gate": self._visit_gate,
            "ema_alpha": self._ema_alpha,
            "ramp_cutoff": self._ramp_cutoff,
            "evictions": self._evictions,
            "entries": [
                {"seq": list(k), "ema": v["ema"], "n": v["n"], "last_ts": v["last_ts"]}
                for k, v in self._table.items()
            ],
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
            self._table.clear()
            for entry in data.get("entries", []):
                key = tuple(entry["seq"])
                self._table[key] = {
                    "ema": float(entry["ema"]),
                    "n": int(entry["n"]),
                    "last_ts": float(entry["last_ts"]),
                }
            self._evictions = int(data.get("evictions", 0))
            return True
        except Exception as e:
            logger.error("[Reasoning/MechA] Failed to load seq_quality: %s", e)
            return False

    def seed_from_archive(self, archive_rows: list[dict]) -> int:
        """Bulk-seed from chain_archive rows. Used for warm-start on T1.

        Each row: {"chain_sequence": [str], "outcome_score": float}.
        Updates every prefix of every chain with its outcome_score.
        """
        seeded = 0
        now = time.time()
        for row in archive_rows:
            seq = row.get("chain_sequence") or row.get("chain") or []
            score = float(row.get("outcome_score", 0.0))
            if not seq or score < 0:
                continue
            for i in range(1, len(seq) + 1):
                self._update_one(tuple(seq[:i]), score, now)
                seeded += 1
        self._maybe_evict()
        return seeded


# ── rFP α Mechanism B: Step Value Net ─────────────────────────────
# Tiny TD-regression net: V(state, chain_len, last_primitive) → terminal reward.
# Input dim: policy_input_dim + 11 (conf + gut_agreement + chain_len_norm + last-prim one-hot 8).
# Output: scalar predicted terminal reward.
# Training: MSE vs z-score normalized terminal reward, running μ/σ via EMA.


class StepValueNet:
    """Mechanism B — tiny value net predicting terminal reward from mid-chain state.

    Architecture (rFP α D1/D2 lock):
        Input: policy_input + confidence + gut_agreement + chain_len_norm +
               last_primitive_onehot(8)  →  total = policy_input_dim + 11
        Hidden: 2 layers (default 64 + 32) ReLU, Xavier init
        Output: 1 scalar (normalized), denormalized at inference

    Training: TD regression — trains on completed chains during chain
    conclusion. V-targets z-score normalized via running EMA (μ, σ) to
    handle terminal-reward distribution drift (D25).

    Cold-started in v1; warm-start path via offline pre-training on
    action_chains_step table (Phase 0.5 persistence).
    """

    def __init__(self, policy_input_dim: int, hidden_1: int = 64, hidden_2: int = 32,
                 learning_rate: float = 0.001, vtarget_ema_alpha: float = 0.01):
        # Input: policy dim + 2 (conf, gut_agreement) + 1 (chain_len_norm) + 8 (last_prim one-hot)
        self.input_dim = policy_input_dim + 11
        self.policy_input_dim = policy_input_dim  # stored for audit
        self.h1 = hidden_1
        self.h2 = hidden_2
        self.lr = learning_rate

        scale1 = math.sqrt(2.0 / self.input_dim)
        scale2 = math.sqrt(2.0 / hidden_1)
        scale3 = math.sqrt(2.0 / hidden_2)

        self.w1 = np.random.randn(self.input_dim, hidden_1) * scale1
        self.b1 = np.zeros(hidden_1)
        self.w2 = np.random.randn(hidden_1, hidden_2) * scale2
        self.b2 = np.zeros(hidden_2)
        self.w3 = np.random.randn(hidden_2, 1) * scale3
        self.b3 = np.zeros(1)

        # V-target running normalization (D25)
        self._vtarget_mean = 0.3   # seed near observed terminal-reward mean (~0.3)
        self._vtarget_std = 0.15   # seed std near observed range (~0.15)
        self._vtarget_alpha = vtarget_ema_alpha

        # Training stats
        self.total_updates = 0
        self.last_loss = 0.0

        # Backprop cache
        self._cache = {}

    def build_input(self, policy_input: np.ndarray, confidence: float,
                    gut_agreement: float, chain_len: int, max_chain: int,
                    last_primitive: Optional[str]) -> np.ndarray:
        """Compose the 122D (or policy_input_dim+11) input vector.

        last_primitive: None at chain start → all-zero one-hot.
        """
        chain_len_norm = float(min(chain_len, max_chain)) / max(1, max_chain)
        last_prim_oh = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if last_primitive and last_primitive in PRIMITIVES:
            last_prim_oh[PRIMITIVES.index(last_primitive)] = 1.0
        extras = np.array([float(confidence), float(gut_agreement), chain_len_norm],
                          dtype=np.float64)
        return np.concatenate([policy_input.astype(np.float64), extras, last_prim_oh])

    def forward(self, x: np.ndarray) -> float:
        """Forward pass → normalized V prediction (denormalized via helper)."""
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3  # shape (1,)
        self._cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return float(z3[0])

    def predict(self, x: np.ndarray) -> float:
        """Denormalized V prediction — multiply by running σ, add running μ."""
        v_norm = self.forward(x)
        return v_norm * self._vtarget_std + self._vtarget_mean

    def train_step(self, x: np.ndarray, target_v: float) -> float:
        """TD regression step. Updates running μ/σ then MSE-backprops.

        target_v is the raw terminal reward; we z-score normalize for training.
        """
        # Update running μ/σ (EMA)
        delta = target_v - self._vtarget_mean
        self._vtarget_mean += self._vtarget_alpha * delta
        variance = (1.0 - self._vtarget_alpha) * (self._vtarget_std ** 2) \
                   + self._vtarget_alpha * (delta ** 2)
        self._vtarget_std = max(1e-4, math.sqrt(variance))

        # Normalize target
        target_norm = (target_v - self._vtarget_mean) / (self._vtarget_std + 1e-5)

        # Forward + MSE gradient
        v_pred = self.forward(x)
        d_z3 = np.array([v_pred - target_norm])  # shape (1,)

        # Backprop
        d_w3 = self._cache["h2"].reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_b3 = d_z3
        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (self._cache["z2"] > 0)
        d_w2 = self._cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (self._cache["z1"] > 0)
        d_w1 = self._cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        for g in [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]:
            np.clip(g, -5.0, 5.0, out=g)

        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3

        self.total_updates += 1
        loss = float(0.5 * (v_pred - target_norm) ** 2)
        self.last_loss = loss
        return loss

    def save(self, path: str) -> None:
        data = {
            "schema": 1,
            "input_dim": self.input_dim,
            "policy_input_dim": self.policy_input_dim,
            "h1": self.h1, "h2": self.h2,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "vtarget_mean": self._vtarget_mean,
            "vtarget_std": self._vtarget_std,
            "vtarget_alpha": self._vtarget_alpha,
            "total_updates": self.total_updates,
            "last_loss": self.last_loss,
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
            saved_dim = data.get("input_dim", self.input_dim)
            if saved_dim != self.input_dim:
                logger.warning("[Reasoning/MechB] Input dim mismatch: saved=%d current=%d — discarding",
                               saved_dim, self.input_dim)
                return False
            self.w1 = np.array(data["w1"])
            self.b1 = np.array(data["b1"])
            self.w2 = np.array(data["w2"])
            self.b2 = np.array(data["b2"])
            self.w3 = np.array(data["w3"])
            self.b3 = np.array(data["b3"])
            self._vtarget_mean = float(data.get("vtarget_mean", 0.3))
            self._vtarget_std = max(1e-4, float(data.get("vtarget_std", 0.15)))
            self._vtarget_alpha = float(data.get("vtarget_alpha", 0.01))
            self.total_updates = int(data.get("total_updates", 0))
            self.last_loss = float(data.get("last_loss", 0.0))
            return True
        except Exception as e:
            logger.error("[Reasoning/MechB] Failed to load value_head: %s", e)
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

        # ── rFP α Reasoning Rewards (Phase 1 infrastructure) ──────────
        # Config section [reasoning_rewards] in titan_params.toml.
        # v1 ships ALL code; phase schedule controlled by weights.
        # Phase 1 has weights=0 → telemetry only, no behavior change.
        _rr = cfg.get("reasoning_rewards") or {}
        self._rr_enabled = bool(_rr.get("enabled", True))
        self._rr_publish = bool(_rr.get("publish_enabled", False))
        self._rr_cap = float(_rr.get("intermediate_cap", 0.2))
        self._rr_phase1_end = int(_rr.get("schedule_phase1_chains", 100))
        self._rr_phase2_end = int(_rr.get("schedule_phase2_chains", 500))
        self._rr_w_a_p2 = float(_rr.get("weight_a_phase2", 0.5))
        self._rr_w_b_p2 = float(_rr.get("weight_b_phase2", 0.0))
        self._rr_w_a_p3 = float(_rr.get("weight_a_phase3", 0.3))
        self._rr_w_b_p3 = float(_rr.get("weight_b_phase3", 0.7))
        self._rr_conf_coeff = float(_rr.get("confidence_growth_coeff", 0.3))
        self._rr_conf_step_cap = float(_rr.get("confidence_growth_step_cap", 0.05))
        self._rr_thresh_bonus = float(_rr.get("threshold_cross_bonus", 0.1))
        self._rr_thresh_point = float(_rr.get("threshold_cross_point", 0.6))
        self._rr_horizon_entire = float(_rr.get("mech_a_horizon_entire", 0.4))
        self._rr_horizon_last_k = float(_rr.get("mech_a_horizon_last_k", 0.6))
        self._rr_horizon_k = int(_rr.get("mech_a_horizon_k", 3))
        self._rr_cgn_threshold = float(_rr.get("cgn_emission_threshold", 0.55))

        # Mechanism A — sequence-quality EMA store
        self.seq_quality_store = SequenceQualityStore(
            cap=int(_rr.get("mech_a_table_cap", 10000)),
            visit_gate=int(_rr.get("mech_a_visit_gate", 3)),
            ema_alpha=float(_rr.get("mech_a_ema_alpha", 0.1)),
            ramp_cutoff=int(_rr.get("mech_a_visit_ramp_cutoff", 20)),
        )
        self.seq_quality_store.load(os.path.join(self.save_dir, "sequence_quality.json"))

        # Mechanism B — StepValueNet (input = policy_input_dim + 11)
        self.step_value_net = StepValueNet(
            policy_input_dim=self.policy_input_dim,
            hidden_1=int(_rr.get("mech_b_h1", 64)),
            hidden_2=int(_rr.get("mech_b_h2", 32)),
            learning_rate=float(_rr.get("mech_b_learning_rate", 0.001)),
            vtarget_ema_alpha=float(_rr.get("mech_b_vtarget_ema_alpha", 0.01)),
        )
        self.step_value_net.load(os.path.join(self.save_dir, "value_head.json"))

        # Per-chain rFP α state — reset by _start_chain
        self._rr_cum_bonus: float = 0.0   # Σ intermediate reward this chain (cap = _rr_cap)
        self._rr_threshold_crossed: bool = False  # one-shot per chain
        self._rr_prev_confidence: float = 0.5
        # Per-chain step snapshots for Mechanism B online training at conclusion
        # and for action_chains_step SQLite persistence (Phase 0.5).
        self._rr_step_snapshots: list[dict] = []

        # rFP α: chains_at_activation — enables the phase schedule to be
        # RELATIVE to when publish_enabled was first turned on, not absolute
        # on the lifetime counter. Without this, a Titan with lifetime
        # _total_chains=4881 would skip phase 1+2 on first activation and
        # jump to phase 3 with cold Mechanism B. Persisted via reasoning_totals.json.
        self._rr_chains_at_activation: Optional[int] = None
        self._rr_load_activation_offset()

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

        # 10b. rFP α — compute intermediate reward (Phase 1 telemetry-only
        # unless publish_enabled + past phase1 gate). See _compute_step_reward.
        next_input = self._build_policy_input(
            observation, gut_signals, working_memory_items)
        step_reward, rr_tele = self._compute_step_reward(
            policy_input=policy_input, action_name=action_name,
        )
        # Snapshot for Mechanism B online training at conclusion + Phase 0.5
        # persistence. Keeps per-step state so V_step(s_k, chain[:k]) can be
        # reconstructed accurately for TD regression.
        if self._rr_enabled and len(self._rr_step_snapshots) < 40:
            self._rr_step_snapshots.append({
                "step_idx": len(self.chain) - 1,
                "policy_input": policy_input.tolist(),
                "chain_prefix": list(self.chain),
                "confidence": float(self.confidence),
                "gut_agreement": float(self.gut_agreement),
                "last_primitive": action_name,
                "intermediate_reward": step_reward,
            })

        # 11. Record transition for IQL — now uses shaped reward (Phase 2+)
        self.buffer.record(
            state=policy_input, action=action_idx, reward=step_reward,
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
            "step_reward": round(step_reward, 4),
            "reward_telemetry": rr_tele,
        }

    def _rr_current_phase_weights(self) -> tuple[float, float]:
        """Return (weight_A, weight_B) based on schedule phase.

        Phase gate counts are RELATIVE to chains_at_activation, not absolute
        on the lifetime counter. On first observation of publish_enabled=true,
        chains_at_activation is latched to the current lifetime _total_chains,
        so the phase schedule runs its full arc from that moment.

        Phase 1 (offset < phase1_end):           (0, 0)   — telemetry only
        Phase 2 (phase1_end ≤ offset < phase2_end):  (w_a_p2, w_b_p2)
        Phase 3 (offset ≥ phase2_end):           (w_a_p3, w_b_p3)
        """
        if not self._rr_enabled or not self._rr_publish:
            return (0.0, 0.0)
        # Latch activation baseline on first active tick after publish_enabled=true
        if self._rr_chains_at_activation is None:
            self._rr_chains_at_activation = int(self._total_chains)
            logger.info(
                "[Reasoning/rFP-α] phase schedule anchored — "
                "chains_at_activation=%d (phase1_end=+%d, phase2_end=+%d)",
                self._rr_chains_at_activation,
                self._rr_phase1_end, self._rr_phase2_end)
            self._rr_save_activation_offset()
        offset = int(self._total_chains) - int(self._rr_chains_at_activation)
        if offset < self._rr_phase1_end:
            return (0.0, 0.0)
        if offset < self._rr_phase2_end:
            return (self._rr_w_a_p2, self._rr_w_b_p2)
        return (self._rr_w_a_p3, self._rr_w_b_p3)

    def _rr_activation_state_path(self) -> str:
        return os.path.join(self.save_dir, "rfp_alpha_activation.json")

    def _rr_save_activation_offset(self) -> None:
        """Persist chains_at_activation + schedule params to disk."""
        try:
            path = self._rr_activation_state_path()
            data = {
                "version": 1,
                "saved_ts": time.time(),
                "chains_at_activation": int(self._rr_chains_at_activation) if self._rr_chains_at_activation is not None else None,
                "phase1_end": int(self._rr_phase1_end),
                "phase2_end": int(self._rr_phase2_end),
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("[Reasoning/rFP-α] save activation offset failed: %s", e)

    def _rr_load_activation_offset(self) -> None:
        """Restore chains_at_activation from disk. Safe on first boot
        (missing file → stays None, will be latched on first active tick)."""
        try:
            path = self._rr_activation_state_path()
            if not os.path.exists(path):
                return
            with open(path) as f:
                data = json.load(f)
            val = data.get("chains_at_activation")
            if val is not None:
                self._rr_chains_at_activation = int(val)
                logger.info(
                    "[Reasoning/rFP-α] chains_at_activation restored from disk: %d",
                    self._rr_chains_at_activation)
        except Exception as e:
            logger.warning("[Reasoning/rFP-α] load activation offset failed: %s", e)

    def _compute_step_reward(self, policy_input: np.ndarray,
                             action_name: str) -> tuple[float, dict]:
        """rFP α §2a — intermediate reward at current step.

        Always computes all four components for telemetry. Returns actual
        injected reward (0.0 in Phase 1) and a telemetry dict.

        Components:
          - mech_a: blended sequence-quality lookup (0.4×entire + 0.6×last_K)
          - mech_b: StepValueNet(state + chain_len + last_prim) denormalized
          - bonus_growth: max(0, Δconf) × coeff, cap 0.05/step
          - bonus_thresh: 0.1 one-shot when confidence first crosses 0.6

        Cap: cumulative intermediate reward per chain bounded at _rr_cap (0.2).
        """
        wa, wb = self._rr_current_phase_weights()

        # Mechanism A — horizon-blended prefix lookup
        mech_a_signal = self.seq_quality_store.query_blended(
            chain_prefix=list(self.chain),
            last_k=self._rr_horizon_k,
            horizon_entire=self._rr_horizon_entire,
            horizon_last_k=self._rr_horizon_last_k,
        )

        # Mechanism B — V(state, chain_len, last_prim) denormalized prediction
        try:
            mech_b_input = self.step_value_net.build_input(
                policy_input=policy_input,
                confidence=self.confidence,
                gut_agreement=self.gut_agreement,
                chain_len=len(self.chain),
                max_chain=self.max_chain_length,
                last_primitive=action_name,
            )
            mech_b_signal = self.step_value_net.predict(mech_b_input)
        except Exception as e:
            logger.warning("[Reasoning/MechB] predict failed: %s", e)
            mech_b_signal = 0.0

        # Companion bonus 1 — confidence growth (D12)
        conf_delta = self.confidence - self._rr_prev_confidence
        bonus_growth = min(self._rr_conf_step_cap,
                           max(0.0, conf_delta) * self._rr_conf_coeff)

        # Companion bonus 2 — threshold crossing (D13, one-shot per chain)
        bonus_thresh = 0.0
        if (not self._rr_threshold_crossed
                and self.confidence >= self._rr_thresh_point
                and self._rr_prev_confidence < self._rr_thresh_point):
            bonus_thresh = self._rr_thresh_bonus
            self._rr_threshold_crossed = True

        # Gross intermediate reward before cap (computed for telemetry always)
        gross = (wa * mech_a_signal + wb * mech_b_signal
                 + bonus_growth + bonus_thresh)

        # Phase gate: if publish_enabled=false OR chain < phase1_end →
        # weights are (0,0). Zero INJECTED reward, but all components still
        # logged via telemetry. Bonuses also suppressed in this window so
        # that Phase 1 is a clean "telemetry-only" validation run.
        is_active_phase = (wa > 0.0 or wb > 0.0)
        if is_active_phase:
            remaining = max(0.0, self._rr_cap - self._rr_cum_bonus)
            actual = max(0.0, min(gross, remaining))
            self._rr_cum_bonus += actual
        else:
            actual = 0.0

        # Update prev_confidence for next step's delta
        self._rr_prev_confidence = self.confidence

        tele = {
            "mech_a": round(mech_a_signal, 4),
            "mech_b": round(mech_b_signal, 4),
            "bonus_growth": round(bonus_growth, 4),
            "bonus_thresh": round(bonus_thresh, 4),
            "weight_a": wa,
            "weight_b": wb,
            "gross": round(gross, 4),
            "actual": round(actual, 4),
            "cum_sum": round(self._rr_cum_bonus, 4),
            "cap": self._rr_cap,
        }
        # Per-step log (INFO level — keep concise to avoid spam)
        logger.info(
            "[Reasoning/rFP-α] step=%d prim=%s conf=%.3f Δconf=%+.3f "
            "mech_a=%.3f mech_b=%.3f bonus_g=%.3f bonus_t=%.3f "
            "w=(%.2f,%.2f) actual=%.3f cum=%.3f/%.2f",
            len(self.chain), action_name, self.confidence, conf_delta,
            mech_a_signal, mech_b_signal, bonus_growth, bonus_thresh,
            wa, wb, actual, self._rr_cum_bonus, self._rr_cap,
        )
        return actual, tele

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
        # rFP α — reset per-chain reward state
        self._rr_cum_bonus = 0.0
        self._rr_threshold_crossed = False
        self._rr_prev_confidence = 0.5   # matches initial self.confidence
        self._rr_step_snapshots = []

    def _conclude_chain(self, observation: np.ndarray, gut_signals: dict,
                        body_state: dict, policy_input: np.ndarray) -> dict:
        """End reasoning chain — commit, hold, or abandon."""
        # Check Mind Willing: body ready?
        body_ready = self._body_ready(body_state)

        # Compute final gut agreement against CONCLUDE primitive (Phase 3)
        self._update_gut_agreement(gut_signals, current_primitive="CONCLUDE")

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

        # ── rFP α — update Mechanism A + train Mechanism B ─────────────
        # Mechanism A: update EMA for every prefix of the completed chain.
        # Always runs (independent of publish_enabled) so table accumulates
        # during Phase 1 telemetry window, ready for Phase 2 activation.
        outcome_score_for_mech_a = float(reward)
        a_updated = 0
        if self._rr_enabled and self.chain:
            try:
                a_updated = self.seq_quality_store.update_chain_prefixes(
                    list(self.chain), outcome_score_for_mech_a
                )
            except Exception as e:
                logger.warning("[Reasoning/MechA] update failed: %s", e)

        # Mechanism B: TD regression on per-step snapshots toward terminal reward.
        # Always runs so B learns continuously from Phase 1 onward.
        b_trained = 0
        if self._rr_enabled and self._rr_step_snapshots:
            try:
                for snap in self._rr_step_snapshots:
                    pi = np.array(snap["policy_input"], dtype=np.float64)
                    v_input = self.step_value_net.build_input(
                        policy_input=pi,
                        confidence=snap["confidence"],
                        gut_agreement=snap["gut_agreement"],
                        chain_len=snap["step_idx"] + 1,
                        max_chain=self.max_chain_length,
                        last_primitive=snap["last_primitive"],
                    )
                    self.step_value_net.train_step(v_input, reward)
                    b_trained += 1
            except Exception as e:
                logger.warning("[Reasoning/MechB] train failed: %s", e)

        # Phase 0.5 step-snapshot persistence (D3+D18) — append to SQLite
        # `action_chains_step` table for offline Mechanism B warm-start path.
        # Table auto-created on first write; 7-day retention on engine save.
        if self._rr_enabled and self._rr_step_snapshots:
            try:
                self._rr_persist_step_snapshots(reward, action)
            except Exception as e:
                # Non-fatal — step persistence is for future B training; live
                # Mechanism B already trained in-memory above.
                if self._total_chains % 20 == 0:
                    logger.warning("[Reasoning/StepPersist] %s", e)

        # rFP α §2b — CGN reasoning_strategy emission prep.
        # On COMMIT with outcome_score > cgn_emission_threshold (default 0.55),
        # attach a payload to the conclusion dict. The CALLER (spirit_worker)
        # is responsible for actually emitting on the bus — reasoning engine
        # has no bus handle. META-CGN observes via normal CGN→META flow (D16).
        if action == "COMMIT" and self._rr_enabled:
            try:
                if reward >= self._rr_cgn_threshold:
                    # state_embedding = tier1 30D slice (D15 lock)
                    _tier1 = observation[:30].tolist() if len(observation) >= 30 else observation.tolist()
                    # neuromod_state = current EMA'd mind-level perception
                    _nm_state = {
                        k: float(self.perception.mind.get(k, 0.0))
                        for k in ("DA", "5-HT", "NE", "ACh", "Endorphin", "GABA")
                    }
                    conclusion["cgn_reasoning_strategy"] = {
                        "chain_signature": list(self.chain),
                        "outcome_score": round(float(reward), 4),
                        "confidence_final": round(float(self.confidence), 4),
                        "gut_agreement_final": round(float(self.gut_agreement), 4),
                        "chain_length": len(self.chain),
                        "state_embedding_tier1": _tier1,
                        "neuromod_state": _nm_state,
                        "source": "reasoning.chain_commit",
                        "mech_a_size": len(self.seq_quality_store._table),
                        "mech_b_updates": int(self.step_value_net.total_updates),
                    }
            except Exception as e:
                logger.warning("[Reasoning/CGN] emission prep failed: %s", e)

        # Reset chain state
        self.is_active = False
        self.chain = []
        self.chain_results = []
        self._last_result = None
        self._strategy_bias = None   # Clear meta-reasoning bias on chain end
        self._intuition_bias = None  # Clear vertical convergence bias on chain end
        self._rr_step_snapshots = []  # rFP α — clear per-chain snapshots

        logger.info("[Reasoning] %s — conf=%.3f gut=%.3f chain=%d dur=%.1fs reward=%.3f "
                    "cum_bonus=%.3f mech_a_updates=%d mech_b_train=%d seq_size=%d",
                    action, conclusion["confidence"], conclusion["gut_agreement"],
                    conclusion["chain_length"], conclusion["duration_s"], reward,
                    self._rr_cum_bonus, a_updated, b_trained,
                    len(self.seq_quality_store._table))

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

        # Update gut agreement against the just-executed primitive (Phase 3)
        self._update_gut_agreement(gut_signals, current_primitive=primitive)

    def _update_gut_agreement(self, gut_signals: dict,
                              current_primitive: str | None = None) -> None:
        """rFP β Phase 3 — primitive-affinity gut formula.

        Replaces the degenerate (1 - abs(conf - mean_urgency)) formula
        which collapsed to (1 - conf) when all NS urgencies were 0.

        New formula: gut measures how well firing programs align with the
        currently-executing reasoning primitive. Each program has primitive
        affinities (PRIMITIVE_AFFINITY table) — primary at 1.0, secondary
        at 0.6, anti-affinity at 0.3, neutral default at 0.5.

        Symmetric formula: contribution = urgency × (affinity - 0.5) × 2
            affinity 1.0 → +urgency
            affinity 0.6 → +0.2 × urgency
            affinity 0.5 → 0 (neutral, no contribution)
            affinity 0.3 → -0.4 × urgency
            affinity 0   → -urgency

        gut_agreement = 0.5 + 0.5 × (sum_contribution / total_urgency)
            → in [0, 1], 0.5 = no information / neutral

        Backward compat: if current_primitive=None (e.g. legacy callers),
        use mean-urgency mode for graceful fallback.
        """
        if not gut_signals:
            self.gut_agreement = 0.5
            return

        # Filter noise floor — only programs with meaningful signal contribute
        active = {p: u for p, u in gut_signals.items() if abs(u) >= 0.05}
        if not active:
            self.gut_agreement = 0.5
            return

        # No primitive specified → fall back to mean-urgency agreement
        # (graceful for callers that don't pass primitive — legacy tests etc.)
        if not current_primitive:
            mean_urgency = sum(active.values()) / len(active)
            self.gut_agreement = 1.0 - abs(self.confidence - mean_urgency)
            return

        # Primitive-affinity gut (Phase 3)
        prim = current_primitive.upper()
        total_contribution = 0.0
        total_urgency = 0.0
        for prog, urgency in active.items():
            aff = PRIMITIVE_AFFINITY.get(prog, {}).get(prim, NEUTRAL_AFFINITY)
            # Symmetric contribution in [-urgency, +urgency]
            total_contribution += urgency * (aff - 0.5) * 2.0
            total_urgency += urgency

        if total_urgency < 0.01:
            self.gut_agreement = 0.5
            return

        net = total_contribution / total_urgency  # in [-1, +1]
        # Map to [0, 1] with 0.5 = neutral
        new_gut = 0.5 + 0.5 * max(-1.0, min(1.0, net))

        # Phase 3 idea #2: time-smoothed gut via EMA. Less reactive to
        # single-tick fluctuations; closer to "felt sense" than instantaneous
        # measurement. Alpha = 0.3 → ~3-tick effective half-life.
        gut_ema_alpha = getattr(self, '_gut_ema_alpha', 0.3)
        prev_gut = getattr(self, 'gut_agreement', 0.5)
        self.gut_agreement = (1.0 - gut_ema_alpha) * prev_gut + gut_ema_alpha * new_gut

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
        # rFP α: persist Mechanism A table + Mechanism B weights + activation offset
        if self._rr_enabled:
            try:
                self.seq_quality_store.save(
                    os.path.join(self.save_dir, "sequence_quality.json"))
                self.step_value_net.save(
                    os.path.join(self.save_dir, "value_head.json"))
                self._rr_save_activation_offset()
            except Exception as e:
                logger.warning("[Reasoning/rFP-α] save_all failed: %s", e)
            # Phase 0.5 retention trim (on engine save, not per-chain)
            try:
                self._rr_trim_step_snapshots_retention()
            except Exception as e:
                logger.warning("[Reasoning/StepPersist] trim failed: %s", e)

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

    # ── rFP α Phase 0.5 — action_chains_step persistence ──────────────
    # Stores per-step snapshots for offline Mechanism B pre-training.
    # Schema kept minimal: chain_id + step_idx keyed, JSON blob for details.
    # 7-day retention, trimmed on save_all().

    _RR_STEP_TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS action_chains_step (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id INTEGER NOT NULL,
            step_idx INTEGER NOT NULL,
            created_at REAL NOT NULL,
            terminal_reward REAL NOT NULL,
            outcome_action TEXT NOT NULL,
            confidence REAL NOT NULL,
            gut_agreement REAL NOT NULL,
            chain_prefix TEXT NOT NULL,
            last_primitive TEXT NOT NULL,
            policy_input_json TEXT NOT NULL,
            intermediate_reward REAL DEFAULT 0.0
        )
    """
    _RR_STEP_INDEX_DDL = """
        CREATE INDEX IF NOT EXISTS ix_step_chain_id
            ON action_chains_step(chain_id, step_idx)
    """
    _RR_STEP_CREATED_DDL = """
        CREATE INDEX IF NOT EXISTS ix_step_created
            ON action_chains_step(created_at)
    """

    def _rr_step_db_conn(self):
        """Open SQLite connection to inner_memory.db, ensuring tables exist.

        Uses same DB as action_chains/chain_archive (single-file convenience).
        Caller responsible for closing; we do this inline with `with`.
        """
        import sqlite3
        # Use same path pattern as existing inner_memory usage — project data/
        db_path = "data/inner_memory.db"
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute(self._RR_STEP_TABLE_DDL)
        conn.execute(self._RR_STEP_INDEX_DDL)
        conn.execute(self._RR_STEP_CREATED_DDL)
        return conn

    def _rr_persist_step_snapshots(self, reward: float, action: str) -> None:
        """Insert current chain's step snapshots into action_chains_step.

        Only called if _rr_step_snapshots is non-empty.
        Uses _total_chains as chain_id (lifetime counter, monotonic).
        """
        if not self._rr_step_snapshots:
            return
        now = time.time()
        chain_id = self._total_chains  # lifetime counter, unique per chain
        rows = []
        for snap in self._rr_step_snapshots:
            rows.append((
                chain_id,
                int(snap["step_idx"]),
                now,
                float(reward),
                str(action),
                float(snap["confidence"]),
                float(snap["gut_agreement"]),
                json.dumps(snap["chain_prefix"]),
                str(snap["last_primitive"]),
                json.dumps(snap["policy_input"]),
                float(snap.get("intermediate_reward", 0.0)),
            ))
        try:
            with self._rr_step_db_conn() as conn:
                conn.executemany("""
                    INSERT INTO action_chains_step (
                        chain_id, step_idx, created_at, terminal_reward,
                        outcome_action, confidence, gut_agreement,
                        chain_prefix, last_primitive, policy_input_json,
                        intermediate_reward
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
        except Exception as e:
            raise RuntimeError(f"step_snapshot insert: {e}") from e

    def _rr_trim_step_snapshots_retention(self, retention_days: int = 7) -> int:
        """Delete step snapshots older than retention window. Called from save_all().

        Returns count of rows deleted.
        """
        cutoff = time.time() - (retention_days * 86400)
        try:
            with self._rr_step_db_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM action_chains_step WHERE created_at < ?",
                    (cutoff,))
                deleted = cur.rowcount
                conn.commit()
            if deleted > 0:
                logger.info("[Reasoning/StepPersist] trimmed %d rows older than %dd",
                            deleted, retention_days)
            return deleted
        except Exception:
            return 0

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
