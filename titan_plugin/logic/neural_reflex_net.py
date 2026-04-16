"""
titan_plugin/logic/neural_reflex_net.py — V5 Neural Reflex Micro-Network.

Tiny feedforward network for one nervous system program.
Pure NumPy — no PyTorch dependency (follows FilterDown pattern).

Architecture: input_dim → hidden_1 → hidden_2 → 1 (sigmoid)
Default: 55 → 48 → 24 → 1 (~2,900 parameters, <0.1ms inference)

Each of the 5 nervous system programs (REFLEX, FOCUS, INTUITION, IMPULSE,
INSPIRATION) gets its own independent NeuralReflexNet that learns from
experience which situations deserve which urgency response.
"""
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_INPUT_DIM = 55
DEFAULT_HIDDEN_1 = 48
DEFAULT_HIDDEN_2 = 24
DEFAULT_LR = 0.001
DEFAULT_FIRE_THRESHOLD = 0.3
GRAD_CLIP = 5.0


class NeuralReflexNet:
    """
    Tiny feedforward network for one nervous system program.

    Forward:  input → Linear(H1) → ReLU → Linear(H2) → ReLU → Linear(1) → Sigmoid
    Training: MSE loss with manual backprop, gradient clipping.
    Persistence: JSON (weights as lists, same as FilterDown).
    """

    def __init__(
        self,
        name: str,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_1: int = DEFAULT_HIDDEN_1,
        hidden_2: int = DEFAULT_HIDDEN_2,
        learning_rate: float = DEFAULT_LR,
        fire_threshold: float = DEFAULT_FIRE_THRESHOLD,
    ):
        self.name = name
        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.lr = learning_rate
        self.fire_threshold = fire_threshold
        self._feature_set = "standard"

        # Xavier initialization
        self.w1 = np.random.randn(input_dim, hidden_1).astype(np.float64) * math.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_1, dtype=np.float64)
        self.w2 = np.random.randn(hidden_1, hidden_2).astype(np.float64) * math.sqrt(2.0 / hidden_1)
        self.b2 = np.zeros(hidden_2, dtype=np.float64)
        self.w3 = np.random.randn(hidden_2, 1).astype(np.float64) * math.sqrt(2.0 / hidden_2)
        self.b3 = np.zeros(1, dtype=np.float64)

        # Training stats
        self.total_updates: int = 0
        self.last_loss: float = 0.0
        self.fire_count: int = 0

    def forward(self, x) -> float:
        """Forward pass → urgency ∈ [0, 1]."""
        x = np.asarray(x, dtype=np.float64).ravel()
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        z = float((h2 @ self.w3 + self.b3).item())
        return _sigmoid(z)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch forward for training. X shape: (batch, input_dim)."""
        h1 = np.maximum(0.0, X @ self.w1 + self.b1)          # (B, H1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)         # (B, H2)
        z = h2 @ self.w3 + self.b3                             # (B, 1)
        return _sigmoid_vec(z)                                  # (B, 1)

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Single training step with MSE loss + backprop.

        Args:
            inputs:  (batch, input_dim) observation vectors
            targets: (batch, 1) target urgencies ∈ [0, 1]

        Returns:
            MSE loss value.
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        batch_size = inputs.shape[0]

        # ── Forward pass (cache activations) ──
        z1 = inputs @ self.w1 + self.b1                        # (B, H1)
        h1 = np.maximum(0.0, z1)                               # ReLU
        z2 = h1 @ self.w2 + self.b2                            # (B, H2)
        h2 = np.maximum(0.0, z2)                               # ReLU
        z3 = h2 @ self.w3 + self.b3                            # (B, 1)
        out = _sigmoid_vec(z3)                                  # (B, 1)

        # ── Loss ──
        diff = out - targets
        loss = float(np.mean(diff ** 2))

        # ── Backward pass ──
        # d_loss/d_out = 2 * (out - target) / batch_size
        d_out = (2.0 / batch_size) * diff                      # (B, 1)

        # Sigmoid derivative: sigmoid * (1 - sigmoid)
        d_z3 = d_out * out * (1.0 - out)                       # (B, 1)

        # Layer 3
        d_w3 = h2.T @ d_z3                                     # (H2, 1)
        d_b3 = np.sum(d_z3, axis=0)                            # (1,)
        d_h2 = d_z3 @ self.w3.T                                # (B, H2)

        # ReLU derivative
        d_z2 = d_h2 * (z2 > 0).astype(np.float64)             # (B, H2)

        # Layer 2
        d_w2 = h1.T @ d_z2                                     # (H1, H2)
        d_b2 = np.sum(d_z2, axis=0)                            # (H2,)
        d_h1 = d_z2 @ self.w2.T                                # (B, H1)

        # ReLU derivative
        d_z1 = d_h1 * (z1 > 0).astype(np.float64)             # (B, H1)

        # Layer 1
        d_w1 = inputs.T @ d_z1                                 # (in, H1)
        d_b1 = np.sum(d_z1, axis=0)                            # (H1,)

        # ── Gradient clipping ──
        for grad in [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]:
            np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

        # ── Weight update ──
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3

        self.total_updates += 1
        self.last_loss = loss
        return loss

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return (self.input_dim * self.hidden_1 + self.hidden_1 +
                self.hidden_1 * self.hidden_2 + self.hidden_2 +
                self.hidden_2 * 1 + 1)

    def save(self, path: str) -> None:
        """Save weights + stats to JSON. Atomic write (tmp→rename)."""
        data = {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_1": self.hidden_1,
            "hidden_2": self.hidden_2,
            "lr": self.lr,
            "fire_threshold": self.fire_threshold,
            "feature_set": self._feature_set,
            "total_updates": self.total_updates,
            "last_loss": self.last_loss,
            "fire_count": self.fire_count,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load weights + stats from JSON. Supports dimension migration."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)

            saved_dim = data.get("input_dim", DEFAULT_INPUT_DIM)
            saved_h1 = data.get("hidden_1", DEFAULT_HIDDEN_1)
            saved_h2 = data.get("hidden_2", DEFAULT_HIDDEN_2)

            # Load stats (always preserved)
            self.total_updates = data.get("total_updates", 0)
            self.last_loss = data.get("last_loss", 0.0)
            self.fire_count = data.get("fire_count", 0)
            self._feature_set = data.get("feature_set", "standard")

            if saved_dim == self.input_dim and saved_h1 == self.hidden_1:
                # Exact match — direct load
                self.w1 = np.array(data["w1"], dtype=np.float64)
                self.b1 = np.array(data["b1"], dtype=np.float64)
                self.w2 = np.array(data["w2"], dtype=np.float64)
                self.b2 = np.array(data["b2"], dtype=np.float64)
                self.w3 = np.array(data["w3"], dtype=np.float64)
                self.b3 = np.array(data["b3"], dtype=np.float64)
                logger.info("[NeuralReflexNet] Loaded %s: %d updates, loss=%.6f",
                            self.name, self.total_updates, self.last_loss)
            else:
                # Dimension migration — preserve learned knowledge
                self._migrate_weights(data, saved_dim, saved_h1, saved_h2)

            return True
        except Exception as e:
            logger.warning("[NeuralReflexNet] Failed to load %s: %s", path, e)
            return False

    def _migrate_weights(self, data: dict, old_dim: int, old_h1: int, old_h2: int) -> None:
        """Migrate weights when input_dim or hidden sizes changed.

        Strategy: preserve as much learned knowledge as possible.
        - w1: copy old rows, Xavier-init new rows scaled ×0.1 (minimize disruption)
        - b1: copy old values, zero-init new
        - w2: copy old block, Xavier-init expansion scaled ×0.1
        - b2, w3, b3: copy if h2 unchanged, expand if needed
        """
        old_w1 = np.array(data["w1"], dtype=np.float64)
        old_b1 = np.array(data["b1"], dtype=np.float64)
        old_w2 = np.array(data["w2"], dtype=np.float64)
        old_b2 = np.array(data["b2"], dtype=np.float64)

        # w1: (old_dim, old_h1) → (new_dim, new_h1)
        # Fresh Xavier init, then copy preserved rows/cols
        copy_rows = min(old_dim, self.input_dim)
        copy_h1 = min(old_h1, self.hidden_1)
        self.w1[:copy_rows, :copy_h1] = old_w1[:copy_rows, :copy_h1]
        # Scale new rows small to avoid disrupting existing dynamics
        if self.input_dim > old_dim:
            self.w1[old_dim:, :] *= 0.1
        if self.hidden_1 > old_h1:
            self.w1[:, old_h1:] *= 0.1

        # b1: copy preserved, rest stays zero
        self.b1[:copy_h1] = old_b1[:copy_h1]

        # w2: (old_h1, old_h2) → (new_h1, new_h2)
        copy_h2 = min(old_h2, self.hidden_2)
        self.w2[:copy_h1, :copy_h2] = old_w2[:copy_h1, :copy_h2]
        if self.hidden_1 > old_h1:
            self.w2[old_h1:, :] *= 0.1
        if self.hidden_2 > old_h2:
            self.w2[:, old_h2:] *= 0.1

        # b2: copy preserved
        self.b2[:copy_h2] = old_b2[:copy_h2]

        # w3 + b3: (old_h2, 1) → (new_h2, 1) — preserve output layer
        old_w3 = np.array(data["w3"], dtype=np.float64)
        old_b3 = np.array(data["b3"], dtype=np.float64)
        self.w3[:copy_h2, :] = old_w3[:copy_h2, :]
        if self.hidden_2 > old_h2:
            self.w3[old_h2:, :] *= 0.1
        self.b3[:] = old_b3[:]

        logger.info(
            "[NeuralReflexNet] MIGRATED %s: %dD(%d/%d)→%dD(%d/%d) "
            "(preserved %d input rows, %d/%d hidden — %d updates retained)",
            self.name, old_dim, old_h1, old_h2,
            self.input_dim, self.hidden_1, self.hidden_2,
            copy_rows, copy_h1, copy_h2, self.total_updates)

    def get_stats(self) -> dict:
        """Return stats for API."""
        return {
            "name": self.name,
            "type": "neural",
            "input_dim": self.input_dim,
            "hidden": [self.hidden_1, self.hidden_2],
            "param_count": self.param_count(),
            "feature_set": self._feature_set,
            "fire_threshold": self.fire_threshold,
            "total_updates": self.total_updates,
            "last_loss": round(self.last_loss, 6),
            "fire_count": self.fire_count,
            "learning_rate": self.lr,
        }


# ── Transition Buffer ──────────────────────────────────────────────

class NervousTransitionBuffer:
    """
    Per-program transition buffer for neural training.
    Follows TransitionBuffer pattern from filter_down.py.
    """

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self._observations: list[list[float]] = []
        self._urgencies: list[float] = []
        self._vm_baselines: list[float] = []
        self._rewards: list[float] = []
        self._fired: list[bool] = []
        self._last_fired_idx: int = -1

    def __len__(self) -> int:
        return len(self._observations)

    @property
    def last_fired(self) -> bool:
        """Whether the most recent transition was a fired signal."""
        if self._last_fired_idx >= 0 and self._last_fired_idx < len(self._fired):
            return self._fired[self._last_fired_idx]
        return False

    def add(self, observation: list, urgency: float, vm_baseline: float,
            reward: float = 0.0, fired: bool = False) -> None:
        """Record one transition."""
        self._observations.append(list(observation))
        self._urgencies.append(urgency)
        self._vm_baselines.append(vm_baseline)
        self._rewards.append(reward)
        self._fired.append(fired)

        if fired:
            self._last_fired_idx = len(self._observations) - 1

        # Evict oldest if over capacity
        if len(self._observations) > self.max_size:
            self._observations.pop(0)
            self._urgencies.pop(0)
            self._vm_baselines.pop(0)
            self._rewards.pop(0)
            self._fired.pop(0)
            if self._last_fired_idx >= 0:
                self._last_fired_idx -= 1

    def update_last_reward(self, reward: float) -> None:
        """Update reward for the most recent fired transition."""
        if self._last_fired_idx >= 0 and self._last_fired_idx < len(self._rewards):
            self._rewards[self._last_fired_idx] = reward

    def update_recent_rewards(self, reward: float, k: int = 1,
                              decay: float = 0.5) -> int:
        """rFP β Stage 2: eligibility-trace credit assignment.

        Apply `reward` to the last K fired transitions with exponential
        decay: reward(0)=reward, reward(1)=reward*decay, reward(2)=reward*decay^2,
        etc. Walks backward from the most recent fire.

        Returns the number of transitions actually updated (≤ K).
        """
        updated = 0
        decay_factor = 1.0
        # Walk backward through buffer finding fired transitions
        for i in range(len(self._fired) - 1, -1, -1):
            if updated >= k:
                break
            if self._fired[i]:
                self._rewards[i] = reward * decay_factor
                decay_factor *= decay
                updated += 1
        return updated

    def update_soft_reward(self, reward: float, fire_threshold: float,
                            soft_factor: float = 0.5) -> bool:
        """rFP β Stage 2: soft-fire propagation.

        For a NOT-FIRED most-recent transition, apply a scaled reward
        proportional to how close the urgency was to the fire threshold.

        urgency_ratio = last_urgency / fire_threshold (clamped [0, 1])
        soft_reward = reward * urgency_ratio * soft_factor

        This breaks the class imbalance (target=0 dominance for not-fired
        samples) by giving the network gradient signal for "close to firing"
        states without polluting the genuinely-restraint cases.

        Returns True if a soft reward was applied (urgency was meaningful).
        """
        if not self._urgencies:
            return False
        last_urg = self._urgencies[-1]
        if fire_threshold <= 0:
            return False
        urgency_ratio = max(0.0, min(1.0, last_urg / fire_threshold))
        if urgency_ratio < 0.1:  # too far from threshold — true restraint
            return False
        soft = reward * urgency_ratio * soft_factor
        self._rewards[-1] = soft
        return True

    def sample_stratified(self, batch_size: int) -> tuple:
        """rFP β Stage 2: 50/50 fired-vs-not-fired stratified sampling.

        Combats class imbalance during training (97-99% of transitions
        are not-fired by default). Falls back to uniform `sample()` if
        one class is empty (e.g., very early training).

        Returns same tuple shape as sample(): (obs, urgencies, vm_baselines,
        rewards, fired) — each as np.ndarray.
        """
        import numpy as _np
        fired_idx = [i for i, f in enumerate(self._fired) if f]
        not_fired_idx = [i for i, f in enumerate(self._fired) if not f]

        if not fired_idx or not not_fired_idx:
            return self.sample(batch_size)

        half = batch_size // 2
        n_fired = min(half, len(fired_idx))
        n_not_fired = min(batch_size - n_fired, len(not_fired_idx))

        fired_sample = _np.random.choice(fired_idx, n_fired, replace=False)
        not_fired_sample = _np.random.choice(not_fired_idx, n_not_fired, replace=False)
        indices = _np.concatenate([fired_sample, not_fired_sample])
        _np.random.shuffle(indices)

        obs = _np.array([self._observations[i] for i in indices], dtype=_np.float64)
        return (
            obs,
            _np.array([self._urgencies[i] for i in indices], dtype=_np.float64),
            _np.array([self._vm_baselines[i] for i in indices], dtype=_np.float64),
            _np.array([self._rewards[i] for i in indices], dtype=_np.float64),
            _np.array([self._fired[i] for i in indices], dtype=bool),
        )

    def sample(self, batch_size: int) -> tuple:
        """
        Random sample for training.

        Returns:
            (observations, urgencies, vm_baselines, rewards, fired)
            Each as np.ndarray.
        """
        n = len(self._observations)
        if n == 0:
            return (np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]))

        indices = random.sample(range(n), min(batch_size, n))
        return (
            np.array([self._observations[i] for i in indices], dtype=np.float64),
            np.array([self._urgencies[i] for i in indices], dtype=np.float64),
            np.array([self._vm_baselines[i] for i in indices], dtype=np.float64),
            np.array([self._rewards[i] for i in indices], dtype=np.float64),
            np.array([self._fired[i] for i in indices], dtype=bool),
        )

    def save(self, path: str) -> None:
        """Persist to JSON. Atomic write (tmp→rename)."""
        data = {
            "observations": self._observations[-self.max_size:],
            "urgencies": self._urgencies[-self.max_size:],
            "vm_baselines": self._vm_baselines[-self.max_size:],
            "rewards": self._rewards[-self.max_size:],
            # Cast fired to plain bool (numpy.bool_ is not JSON-serializable)
            "fired": [bool(f) for f in self._fired[-self.max_size:]],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load from JSON. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._observations = data.get("observations", [])
            self._urgencies = data.get("urgencies", [])
            self._vm_baselines = data.get("vm_baselines", [])
            self._rewards = data.get("rewards", [])
            self._fired = data.get("fired", [])
            # Find last fired index
            self._last_fired_idx = -1
            for i in range(len(self._fired) - 1, -1, -1):
                if self._fired[i]:
                    self._last_fired_idx = i
                    break
            return True
        except Exception as e:
            logger.warning("[NervousBuffer] Failed to load %s: %s", path, e)
            return False


# ── Helpers ────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized numerically stable sigmoid."""
    x = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x))
