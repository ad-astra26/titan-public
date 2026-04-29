"""
titan_plugin/logic/filter_down.py — FILTER_DOWN: learned attention for the Trinity.

A lightweight value network learns V(state) → expected future equilibrium quality
from accumulated Trinity tensor transitions. Attention gradients ∂V/∂state reveal
which sensors matter most, producing severity_multipliers for Body/Mind.

Engineered weights become initial weights; FILTER_DOWN reshapes them from
lived experience. This is how Titan learns what hurts.

Architecture (V3 — 15-dim Inner Trinity):
  - State: 15-dim Trinity tensor (5 Body + 5 Mind + 5 Spirit)
  - Network: 15→32→16→1
  - Output: body_multipliers[5] + mind_multipliers[5]

Architecture (V4 — 30-dim Full Trinity):
  - State: 30-dim SPIRIT tensor (15 Inner + 15 Outer)
  - Network: 30→64→32→1
  - Output: inner_body[5] + inner_mind[5] + outer_body[5] + outer_mind[5]
  - Spirit dimensions (inner[10:15] + outer[25:30]) still not published (observes)

Both architectures coexist — V4 network uses separate weight/buffer files.
"""
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

STATE_DIM = 15           # 5 Body + 5 Mind + 5 Spirit (V3)
HIDDEN_1 = 32
HIDDEN_2 = 16

# V4 constants (30-dim Full Trinity)
V4_STATE_DIM = 30        # 15 Inner + 15 Outer
V4_HIDDEN_1 = 64
V4_HIDDEN_2 = 32
GAMMA = 0.95             # Discount factor
LR = 0.001               # Learning rate
BATCH_SIZE = 16           # Mini-batch for training
BUFFER_MAX = 2000         # Max stored transitions
MIN_TRANSITIONS = 32      # Don't train until this many transitions
TRAIN_EVERY_N = 5         # Train every N new transitions
MULTIPLIER_FLOOR = 0.3    # Minimum severity multiplier (never fully mute a sense)
MULTIPLIER_CEIL = 3.0     # Maximum severity multiplier
SMOOTHING = 0.9           # EMA smoothing for multiplier updates


def _resolve_filter_down_cfg(config: Optional[dict]) -> dict:
    """Resolve [filter_down] TOML section into per-instance params with module-
    constant fallbacks. Shared by V3/V4/V5. Keys not present fall back to the
    module constants above (which match the toml defaults, so behavior is
    unchanged for callers that pass config=None).
    """
    section = (config or {}).get("filter_down", {}) if isinstance(config, dict) else {}
    return {
        "gamma":            float(section.get("gamma", GAMMA)),
        "lr":               float(section.get("learning_rate", LR)),
        "batch_size":       int(section.get("batch_size", BATCH_SIZE)),
        "buffer_max":       int(section.get("buffer_max", BUFFER_MAX)),
        "min_transitions":  int(section.get("min_transitions", MIN_TRANSITIONS)),
        "train_every_n":    int(section.get("train_every_n", TRAIN_EVERY_N)),
        "multiplier_floor": float(section.get("multiplier_floor", MULTIPLIER_FLOOR)),
        "multiplier_ceil":  float(section.get("multiplier_ceil", MULTIPLIER_CEIL)),
        "smoothing":        float(section.get("smoothing", SMOOTHING)),
    }

# ── Pure numpy value network ──────────────────────────────────────────
# Using numpy instead of PyTorch to avoid ~150MB memory overhead in Spirit process.

def _relu(x):
    """Element-wise ReLU."""
    import numpy as np
    return np.maximum(0, x)


def _relu_grad(x):
    """Gradient of ReLU."""
    import numpy as np
    return (x > 0).astype(np.float64)


class TrinityValueNet:
    """
    Tiny feedforward value network: 15 → 32 → 16 → 1.

    Predicts expected future middle_path_loss from current Trinity state.
    All numpy — no PyTorch dependency.
    """

    def __init__(self, state_dim: int = STATE_DIM):
        import numpy as np

        self.state_dim = state_dim
        # Xavier init
        self.w1 = np.random.randn(state_dim, HIDDEN_1).astype(np.float64) * math.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(HIDDEN_1, dtype=np.float64)
        self.w2 = np.random.randn(HIDDEN_1, HIDDEN_2).astype(np.float64) * math.sqrt(2.0 / HIDDEN_1)
        self.b2 = np.zeros(HIDDEN_2, dtype=np.float64)
        self.w3 = np.random.randn(HIDDEN_2, 1).astype(np.float64) * math.sqrt(2.0 / HIDDEN_2)
        self.b3 = np.zeros(1, dtype=np.float64)

    def forward(self, state):
        """Forward pass. state: (state_dim,) → scalar value."""
        import numpy as np
        s = np.asarray(state, dtype=np.float64)
        self._z1 = s @ self.w1 + self.b1
        self._a1 = _relu(self._z1)
        self._z2 = self._a1 @ self.w2 + self.b2
        self._a2 = _relu(self._z2)
        self._z3 = self._a2 @ self.w3 + self.b3
        return float(self._z3[0])

    def forward_batch(self, states):
        """Forward pass for batch. states: (N, state_dim) → (N,)."""
        import numpy as np
        S = np.asarray(states, dtype=np.float64)
        z1 = S @ self.w1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.w3 + self.b3
        return z3.flatten()

    def gradient_wrt_input(self, state) -> list[float]:
        """
        Compute ∂V/∂state via manual backprop through ReLU layers.
        Returns 15-dim gradient vector (attention signal).
        """
        import numpy as np

        s = np.asarray(state, dtype=np.float64)
        # Forward
        z1 = s @ self.w1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = _relu(z2)
        # Output gradient is 1 (scalar output)
        # Backprop: dL/dz3 = 1
        dz3 = np.ones(1, dtype=np.float64)
        # dL/da2 = dz3 @ w3^T
        da2 = dz3 @ self.w3.T  # (HIDDEN_2,)
        dz2 = da2 * _relu_grad(z2)
        da1 = dz2 @ self.w2.T  # (HIDDEN_1,)
        dz1 = da1 * _relu_grad(z1)
        # dL/ds = dz1 @ w1^T
        ds = dz1 @ self.w1.T  # (state_dim,)
        return ds.tolist()

    def train_step(self, states, rewards, next_states, lr: float = LR) -> float:
        """
        One TD(0) batch update.

        Target: r + γ·V(s')
        Loss: MSE(V(s), target)

        Returns mean loss.
        """
        import numpy as np

        S = np.asarray(states, dtype=np.float64)
        R = np.asarray(rewards, dtype=np.float64)
        S_next = np.asarray(next_states, dtype=np.float64)
        N = len(S)

        # Forward pass for current states
        z1 = S @ self.w1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = _relu(z2)
        values = (a2 @ self.w3 + self.b3).flatten()  # (N,)

        # Forward pass for next states (no grad needed — target network)
        z1n = S_next @ self.w1 + self.b1
        a1n = _relu(z1n)
        z2n = a1n @ self.w2 + self.b2
        a2n = _relu(z2n)
        next_values = (a2n @ self.w3 + self.b3).flatten()

        # TD target
        targets = R + GAMMA * next_values

        # Loss and gradient
        errors = values - targets  # (N,)
        loss = float(np.mean(errors ** 2))

        # Backprop through value network (current states only)
        # dL/dvalues = 2 * errors / N
        dv = 2.0 * errors / N  # (N,)

        # Layer 3
        dz3 = dv.reshape(-1, 1)  # (N, 1)
        dw3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        # Layer 2
        da2 = dz3 @ self.w3.T
        dz2 = da2 * _relu_grad(z2)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Layer 1
        da1 = dz2 @ self.w2.T
        dz1 = da1 * _relu_grad(z1)
        dw1 = S.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient descent
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        return loss

    def save(self, path: str) -> None:
        """Save weights to JSON."""
        import numpy as np
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load weights from JSON. Returns True on success."""
        import numpy as np
        try:
            with open(path) as f:
                data = json.load(f)
            self.w1 = np.array(data["w1"], dtype=np.float64)
            self.b1 = np.array(data["b1"], dtype=np.float64)
            self.w2 = np.array(data["w2"], dtype=np.float64)
            self.b2 = np.array(data["b2"], dtype=np.float64)
            self.w3 = np.array(data["w3"], dtype=np.float64)
            self.b3 = np.array(data["b3"], dtype=np.float64)
            return True
        except Exception as e:
            logger.warning("[FilterDown] Could not load weights: %s", e)
            return False


# ── Transition Buffer ──────────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: TransitionBuffer._write_idx is ring-buffer cursor
# position recomputed from buffer length on load (`len(buffer) % max_size`).
# Not independently-persistable state.
class TransitionBuffer:
    """Ring buffer of (state, reward, next_state) tuples."""

    def __init__(self, max_size: int = BUFFER_MAX):
        self._buffer: list[tuple[list[float], float, list[float]]] = []
        self._max_size = max_size
        self._write_idx = 0

    def add(self, state: list[float], reward: float, next_state: list[float]) -> None:
        if len(self._buffer) < self._max_size:
            self._buffer.append((state, reward, next_state))
        else:
            self._buffer[self._write_idx] = (state, reward, next_state)
        self._write_idx = (self._write_idx + 1) % self._max_size

    def sample(self, batch_size: int) -> tuple[list, list, list]:
        """Random mini-batch. Returns (states, rewards, next_states)."""
        import numpy as np
        n = min(batch_size, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        states, rewards, next_states = [], [], []
        for i in indices:
            s, r, ns = self._buffer[i]
            states.append(s)
            rewards.append(r)
            next_states.append(ns)
        return states, rewards, next_states

    def __len__(self) -> int:
        return len(self._buffer)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._buffer, f)

    def load(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            self._buffer = [(s, r, ns) for s, r, ns in data]
            self._write_idx = len(self._buffer) % self._max_size
            return True
        except Exception:
            return False


# ── FilterDown Engine ──────────────────────────────────────────────────

class FilterDownEngine:
    """
    Orchestrates transition collection, training, and gradient computation.

    Lifecycle (called from Spirit worker):
      1. record_transition() — after each consciousness epoch
      2. maybe_train() — periodically trains value network
      3. compute_multipliers() — gradient-based severity multipliers
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        p = _resolve_filter_down_cfg(config)
        self._gamma            = p["gamma"]
        self._lr               = p["lr"]
        self._batch_size       = p["batch_size"]
        self._min_transitions  = p["min_transitions"]
        self._train_every_n    = p["train_every_n"]
        self._multiplier_floor = p["multiplier_floor"]
        self._multiplier_ceil  = p["multiplier_ceil"]
        self._smoothing        = p["smoothing"]

        self._net = TrinityValueNet()
        self._buffer = TransitionBuffer(max_size=p["buffer_max"])
        self._data_dir = data_dir

        self._weights_path = os.path.join(data_dir, "filter_down_weights.json")
        self._buffer_path = os.path.join(data_dir, "filter_down_buffer.json")

        # Current severity multipliers (start at 1.0 = no modulation)
        self._body_multipliers = [1.0] * 5
        self._mind_multipliers = [1.0] * 5

        # Training stats
        self._transitions_since_train = 0
        self._total_train_steps = 0
        self._last_loss = 0.0

        # Load persisted state
        self._net.load(self._weights_path)
        self._buffer.load(self._buffer_path)
        logger.info("[FilterDown] Initialized: %d transitions in buffer, weights=%s",
                     len(self._buffer), "loaded" if os.path.exists(self._weights_path) else "random")

    def record_transition(
        self,
        body: list[float],
        mind: list[float],
        spirit: list[float],
        next_body: list[float],
        next_mind: list[float],
        next_spirit: list[float],
    ) -> None:
        """Record a Trinity state transition after a consciousness epoch."""
        from .middle_path import middle_path_loss

        state = body + mind + spirit  # 15-dim
        next_state = next_body + next_mind + next_spirit

        # Reward = negative middle path loss of next state (lower loss = higher reward)
        next_loss = middle_path_loss(next_body, next_mind, next_spirit)
        reward = -next_loss

        self._buffer.add(state, reward, next_state)
        self._transitions_since_train += 1

    def maybe_train(self) -> Optional[float]:
        """Train if enough new transitions have accumulated. Returns loss or None."""
        if len(self._buffer) < self._min_transitions:
            return None
        if self._transitions_since_train < self._train_every_n:
            return None

        states, rewards, next_states = self._buffer.sample(self._batch_size)
        loss = self._net.train_step(states, rewards, next_states, lr=self._lr)

        self._total_train_steps += 1
        self._last_loss = loss
        self._transitions_since_train = 0

        # Persist periodically
        if self._total_train_steps % 10 == 0:
            self._persist()

        logger.info("[FilterDown] Train step %d: loss=%.6f buffer=%d",
                     self._total_train_steps, loss, len(self._buffer))
        return loss

    def compute_multipliers(
        self,
        body: list[float],
        mind: list[float],
        spirit: list[float],
    ) -> tuple[list[float], list[float]]:
        """
        Compute severity multipliers from attention gradients.

        ∂V/∂state → 15-dim importance → split into body[5] + mind[5].
        Higher gradient magnitude = more important sensor = higher multiplier.
        Spirit multipliers are not published (Spirit is the observer, not the observed).
        """
        if self._total_train_steps < 1:
            # Not trained yet — return defaults
            return list(self._body_multipliers), list(self._mind_multipliers)

        state = body + mind + spirit
        grad = self._net.gradient_wrt_input(state)

        # Split into body/mind/spirit gradients
        body_grad = [abs(g) for g in grad[:5]]
        mind_grad = [abs(g) for g in grad[5:10]]
        # spirit_grad = grad[10:15]  — not used for severity (Spirit observes)

        # Normalize: divide by max to get relative importance, then scale
        all_grad = body_grad + mind_grad
        max_grad = max(all_grad) if all_grad else 1.0
        if max_grad < 1e-8:
            max_grad = 1.0

        new_body = [max(self._multiplier_floor, min(self._multiplier_ceil, g / max_grad * 2.0)) for g in body_grad]
        new_mind = [max(self._multiplier_floor, min(self._multiplier_ceil, g / max_grad * 2.0)) for g in mind_grad]

        # EMA smoothing to prevent jerky changes
        self._body_multipliers = [
            self._smoothing * old + (1 - self._smoothing) * new
            for old, new in zip(self._body_multipliers, new_body)
        ]
        self._mind_multipliers = [
            self._smoothing * old + (1 - self._smoothing) * new
            for old, new in zip(self._mind_multipliers, new_mind)
        ]

        return (
            [round(m, 4) for m in self._body_multipliers],
            [round(m, 4) for m in self._mind_multipliers],
        )

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self._buffer),
            "total_train_steps": self._total_train_steps,
            "last_loss": round(self._last_loss, 6),
            "body_multipliers": [round(m, 4) for m in self._body_multipliers],
            "mind_multipliers": [round(m, 4) for m in self._mind_multipliers],
        }

    def _persist(self) -> None:
        """Save weights and buffer to disk."""
        try:
            self._net.save(self._weights_path)
            self._buffer.save(self._buffer_path)
        except Exception as e:
            logger.warning("[FilterDown] Persist failed: %s", e)


# ── V4 FilterDown Engine (30-dim) — RETIRED 2026-04-25 ─────────────────
#
# V4 was a 30D learner that became silently dead when unified_spirit.tensor
# was upgraded 30D → 130D in commit 5d2774b8. record_transition + matmul
# failed on every consciousness epoch with size 30 vs 130 mismatch, swallowed
# at DEBUG level. Pattern C migration 2026-04-25 surfaced the swallow as a
# WARNING, Maker greenlit retirement. V5 (FilterDownV5Engine, 162D TITAN_SELF)
# is the sole FilterDown learner now.
#
# State files data/filter_down_v4_{weights,buffer}.json preserved on disk
# per directive_memory_preservation.md but no longer read or written.



# ── V5 FilterDown Engine (162-dim TITAN_SELF) ──────────────────────────
# rFP #2: V5 consumes full 162D TITAN_SELF (130D felt + 2D journey + 30D
# topology, all pre-weighted by consciousness) and produces 120 multipliers
# at extended resolution. Observer dims (iS[20:25], oS[85:90]) are masked —
# computed but never published, per doctrine "Spirit observer is a reflection
# surface, not a target of filtering."

V5_INPUT_DIM = 162       # 130D felt + 2D journey + 30D topology (pre-weighted)
V5_HIDDEN_1 = 128
V5_HIDDEN_2 = 64
V5_OUTPUT_DIM = 120      # 5+15+40+5+15+40 — observer 10 dims MASKED


class FilterDownV5Engine:
    """
    V5 extension: 162-dim FILTER_DOWN covering full TITAN_SELF.

    Input:  162D TITAN_SELF = weighted(130D felt + 2D journey + 30D topology).
    Network: 162 → 128 → 64 → 1 (larger than V4's 30→64→32→1).
    Output: 120 multipliers at EXTENDED resolution, observer dims masked:
      - inner_body[5], inner_mind[15], inner_spirit_content[40] = 60
      - outer_body[5], outer_mind[15], outer_spirit_content[40] = 60
      - 10 observer dims (iS[20:25], oS[85:90]) NEVER published.

    Spirit content multipliers are scaled by spirit_filter_strength_multiplier
    (default 0.3) toward 1.0 — spirit modulates slowly, so gentler filter.

    Trained via TD(0) value network on middle_path_loss of raw felt (RAW,
    not weighted TITAN_SELF). Uses TRUE s→s' transitions (not V4's self-
    transition quirk).

    Weight/buffer persisted to data/filter_down_v5_{weights,buffer}.json.

    Training gates:
      MIN_TRANSITIONS (32): don't train on empty buffer.
      cold_start_floor_epochs (default 2000, configurable): don't publish
         multipliers until net has trained enough to produce meaningful
         gradients. Returns all-1.0 multipliers until floor cleared.
         Separate from MIN_TRANSITIONS because buffer fills fast (~20 min at
         1-2 Hz) but net convergence takes longer.

    Reward semantics:
      record_transition takes BOTH weighted 162D TITAN_SELF (network input/
      target) AND raw 130D felt (reward computation). Reward is always
      -middle_path_loss of raw felt — independent of TITAN_SELF weight changes.
      If [titan_self] weights change later, network retrains on new input
      distribution but reward semantics stay stable.
    """

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        import numpy as np
        cfg = (config or {}).get("filter_down_v5", {})
        base = _resolve_filter_down_cfg(config)

        # Shared base params — drive training cadence and multiplier clipping
        # across V3/V4/V5. V5-specific params remain in [filter_down_v5].
        self._gamma            = base["gamma"]
        self._lr               = base["lr"]
        self._batch_size       = base["batch_size"]
        self._min_transitions  = base["min_transitions"]
        self._train_every_n    = base["train_every_n"]
        self._multiplier_floor = base["multiplier_floor"]
        self._multiplier_ceil  = base["multiplier_ceil"]
        self._smoothing        = base["smoothing"]

        self._spirit_filter_strength_mult = float(cfg.get(
            "spirit_filter_strength_multiplier", 0.3))
        self._cold_start_floor_epochs = int(cfg.get(
            "cold_start_floor_epochs", 2000))
        self._publish_enabled = bool(cfg.get("publish_enabled", False))

        # Value network 162 → 128 → 64 → 1 (override TrinityValueNet's default dims)
        self._net = TrinityValueNet(state_dim=V5_INPUT_DIM)
        self._net.w1 = np.random.randn(V5_INPUT_DIM, V5_HIDDEN_1).astype(np.float64) * math.sqrt(2.0 / V5_INPUT_DIM)
        self._net.b1 = np.zeros(V5_HIDDEN_1, dtype=np.float64)
        self._net.w2 = np.random.randn(V5_HIDDEN_1, V5_HIDDEN_2).astype(np.float64) * math.sqrt(2.0 / V5_HIDDEN_1)
        self._net.b2 = np.zeros(V5_HIDDEN_2, dtype=np.float64)
        self._net.w3 = np.random.randn(V5_HIDDEN_2, 1).astype(np.float64) * math.sqrt(2.0 / V5_HIDDEN_2)
        self._net.b3 = np.zeros(1, dtype=np.float64)

        self._buffer = TransitionBuffer(max_size=base["buffer_max"])
        self._data_dir = data_dir

        self._weights_path = os.path.join(data_dir, "filter_down_v5_weights.json")
        self._buffer_path = os.path.join(data_dir, "filter_down_v5_buffer.json")
        self._state_path = os.path.join(data_dir, "filter_down_v5_state.json")

        self._ib_mults = [1.0] * 5    # inner_body
        self._im_mults = [1.0] * 15   # inner_mind
        self._is_content_mults = [1.0] * 40  # inner_spirit_content [25:65]
        self._ob_mults = [1.0] * 5    # outer_body
        self._om_mults = [1.0] * 15   # outer_mind
        self._os_content_mults = [1.0] * 40  # outer_spirit_content [90:130]

        self._transitions_since_train = 0
        self._total_train_steps = 0
        self._last_loss = 0.0
        self._recent_losses: list[float] = []
        self._buffer_full_logged = False
        self._phase8_snapshot_taken = False

        self._net.load(self._weights_path)
        self._buffer.load(self._buffer_path)
        self._load_state()
        logger.info(
            "[FilterDownV5] Initialized: %d transitions, %d train_steps, weights=%s, publish_enabled=%s",
            len(self._buffer),
            self._total_train_steps,
            "loaded" if os.path.exists(self._weights_path) else "random",
            self._publish_enabled,
        )

    def _load_state(self) -> None:
        """Load train_steps / last_loss / recent_losses / phase8_snapshot_taken
        from sidecar state file. Weights + buffer are loaded separately.

        rFP #2 graduation gate (RFP2-PHASE8) requires total_train_steps to
        cross cold_start_floor_epochs. Before 2026-04-15 this counter reset
        to 0 on every spirit_worker restart (Guardian respawns also reset it),
        making the gate effectively unreachable. This persists the counter.
        """
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                self._total_train_steps = int(state.get("total_train_steps", 0))
                self._last_loss = float(state.get("last_loss", 0.0))
                rl = state.get("recent_losses", [])
                if isinstance(rl, list):
                    self._recent_losses = [float(x) for x in rl[-20:]]
                self._phase8_snapshot_taken = bool(state.get("phase8_snapshot_taken", False))
                # EMA multiplier state — restored if present. Older state files
                # won't have this key; __init__ defaults to [1.0,...] so the
                # fallback is safe.
                ema = state.get("multipliers_ema") or {}
                if isinstance(ema, dict):
                    def _restore(name, attr, expected_len):
                        vals = ema.get(name)
                        if isinstance(vals, list) and len(vals) == expected_len:
                            setattr(self, attr, [float(x) for x in vals])
                    _restore("inner_body",           "_ib_mults",         5)
                    _restore("inner_mind",           "_im_mults",        15)
                    _restore("inner_spirit_content", "_is_content_mults", 40)
                    _restore("outer_body",           "_ob_mults",         5)
                    _restore("outer_mind",           "_om_mults",        15)
                    _restore("outer_spirit_content", "_os_content_mults", 40)
        except Exception as e:
            logger.warning("[FilterDownV5] state load failed: %s (starting fresh)", e)

    def record_transition(
        self,
        titan_self_prev: list,
        titan_self_curr: list,
        felt_prev: list,
        felt_curr: list,
    ) -> None:
        """Record a TRUE s→s' transition of 162D TITAN_SELF.

        Reward computed from RAW felt_curr (130D) via middle_path_loss —
        independent of any future weight changes to TITAN_SELF composition.
        """
        from .middle_path import middle_path_loss
        if len(titan_self_prev) != V5_INPUT_DIM or len(titan_self_curr) != V5_INPUT_DIM:
            return
        if len(felt_curr) != 130:
            return

        # Reward from raw felt (130D = iB5+iM15+iS45+oB5+oM15+oS45).
        # middle_path_loss expects 3 sequences of 5 dims; we take the core
        # 5-dim slices that align with the original Trinity middle-path
        # definition: body[0:5], mind[5:10] (mind core), spirit[20:25] (iS
        # observer/core region). Symmetrical pair for outer.
        ib   = felt_curr[0:5]
        im_5 = felt_curr[5:10]
        is_c = felt_curr[20:25]
        ob   = felt_curr[65:70]
        om_5 = felt_curr[70:75]
        os_c = felt_curr[85:90]
        inner_loss = middle_path_loss(ib, im_5, is_c)
        outer_loss = middle_path_loss(ob, om_5, os_c)
        reward = -(inner_loss + outer_loss) / 2.0

        self._buffer.add(list(titan_self_prev), reward, list(titan_self_curr))
        self._transitions_since_train += 1

        if len(self._buffer) >= self._buffer._max_size and not self._buffer_full_logged:
            logger.info(
                "[FilterDownV5] Transition buffer reached BUFFER_MAX=%d; "
                "further adds will evict oldest entries.", self._buffer._max_size,
            )
            self._buffer_full_logged = True

    def maybe_train(self) -> Optional[float]:
        """Train the value network on a random mini-batch, if enough data."""
        if len(self._buffer) < self._min_transitions:
            return None
        if self._transitions_since_train < self._train_every_n:
            return None

        states, rewards, next_states = self._buffer.sample(self._batch_size)
        loss = self._net.train_step(states, rewards, next_states, lr=self._lr)
        self._total_train_steps += 1
        self._last_loss = loss
        self._recent_losses.append(float(loss))
        if len(self._recent_losses) > 20:
            self._recent_losses = self._recent_losses[-20:]
        self._transitions_since_train = 0

        if self._total_train_steps % 10 == 0:
            self._persist()

        logger.info(
            "[FilterDownV5] Train step %d: loss=%.6f buffer=%d",
            self._total_train_steps, loss, len(self._buffer),
        )
        return loss

    def compute_multipliers(self, titan_self_162d: list) -> dict:
        """Compute severity multipliers from 162D gradient attention.

        Returns a dict with slices at extended resolution. Observer dims
        (iS[20:25] + oS[85:90]) are computed but NEVER included. Spirit
        content multipliers are scaled toward 1.0 by the spirit filter
        strength coefficient.

        If the network hasn't been trained enough (below cold_start_floor),
        returns all-1.0 multipliers (no modulation).
        """
        if self._total_train_steps < self._cold_start_floor_epochs:
            return self._default_multipliers_dict()
        if len(titan_self_162d) != V5_INPUT_DIM:
            return self._default_multipliers_dict()

        grad = self._net.gradient_wrt_input(titan_self_162d)

        # Felt portion is [0:130] — extract observable-slice gradients.
        # Observer dims [20:25] and [85:90] are SKIPPED (never published).
        ib_grad         = [abs(g) for g in grad[0:5]]
        im_grad         = [abs(g) for g in grad[5:20]]
        is_content_grad = [abs(g) for g in grad[25:65]]   # skip [20:25]
        ob_grad         = [abs(g) for g in grad[65:70]]
        om_grad         = [abs(g) for g in grad[70:85]]
        os_content_grad = [abs(g) for g in grad[90:130]]  # skip [85:90]

        all_grad = (ib_grad + im_grad + is_content_grad
                    + ob_grad + om_grad + os_content_grad)
        max_grad = max(all_grad) if all_grad else 1.0
        if max_grad < 1e-8:
            max_grad = 1.0

        def _scale(grads):
            return [max(self._multiplier_floor, min(self._multiplier_ceil, g / max_grad * 2.0))
                    for g in grads]

        new_ib = _scale(ib_grad)
        new_im = _scale(im_grad)
        new_is_c = _scale(is_content_grad)
        new_ob = _scale(ob_grad)
        new_om = _scale(om_grad)
        new_os_c = _scale(os_content_grad)

        # Spirit strength coefficient — pull spirit multipliers toward 1.0.
        k = self._spirit_filter_strength_mult
        new_is_c = [(m - 1.0) * k + 1.0 for m in new_is_c]
        new_os_c = [(m - 1.0) * k + 1.0 for m in new_os_c]

        def _ema(old, new):
            return [self._smoothing * o + (1 - self._smoothing) * n for o, n in zip(old, new)]

        self._ib_mults = _ema(self._ib_mults, new_ib)
        self._im_mults = _ema(self._im_mults, new_im)
        self._is_content_mults = _ema(self._is_content_mults, new_is_c)
        self._ob_mults = _ema(self._ob_mults, new_ob)
        self._om_mults = _ema(self._om_mults, new_om)
        self._os_content_mults = _ema(self._os_content_mults, new_os_c)

        return {
            "inner_body":           [round(m, 4) for m in self._ib_mults],
            "inner_mind":           [round(m, 4) for m in self._im_mults],
            "inner_spirit_content": [round(m, 4) for m in self._is_content_mults],
            "outer_body":           [round(m, 4) for m in self._ob_mults],
            "outer_mind":           [round(m, 4) for m in self._om_mults],
            "outer_spirit_content": [round(m, 4) for m in self._os_content_mults],
        }

    def _default_multipliers_dict(self) -> dict:
        return {
            "inner_body":           [1.0] * 5,
            "inner_mind":           [1.0] * 15,
            "inner_spirit_content": [1.0] * 40,
            "outer_body":           [1.0] * 5,
            "outer_mind":           [1.0] * 15,
            "outer_spirit_content": [1.0] * 40,
        }

    def snapshot_phase8_baseline(self) -> None:
        """On first publish post-flag-flip, snapshot weights as immutable baseline."""
        if self._phase8_snapshot_taken:
            return
        import shutil
        try:
            if os.path.exists(self._weights_path):
                dst = self._weights_path.replace(".json", "_phase8_swap.json")
                shutil.copy(self._weights_path, dst)
                self._phase8_snapshot_taken = True
                logger.info("[FilterDownV5] Phase 8 baseline snapshot saved: %s", dst)
        except Exception as e:
            logger.warning("[FilterDownV5] Baseline snapshot failed: %s", e)

    def get_stats(self) -> dict:
        return {
            "version": "v5",
            "input_dim": V5_INPUT_DIM,
            "output_dim": V5_OUTPUT_DIM,
            "buffer_size": len(self._buffer),
            "total_train_steps": self._total_train_steps,
            "last_loss": round(self._last_loss, 6),
            "publish_enabled": self._publish_enabled,
            "spirit_filter_strength": self._spirit_filter_strength_mult,
            "cold_start_floor": self._cold_start_floor_epochs,
            "multipliers_mean": {
                "inner_body":           sum(self._ib_mults) / max(1, len(self._ib_mults)),
                "inner_mind":           sum(self._im_mults) / max(1, len(self._im_mults)),
                "inner_spirit_content": sum(self._is_content_mults) / max(1, len(self._is_content_mults)),
                "outer_body":           sum(self._ob_mults) / max(1, len(self._ob_mults)),
                "outer_mind":           sum(self._om_mults) / max(1, len(self._om_mults)),
                "outer_spirit_content": sum(self._os_content_mults) / max(1, len(self._os_content_mults)),
            },
            "multipliers": {
                "inner_body":           [round(m, 4) for m in self._ib_mults],
                "inner_mind":           [round(m, 4) for m in self._im_mults],
                "inner_spirit_content": [round(m, 4) for m in self._is_content_mults],
                "outer_body":           [round(m, 4) for m in self._ob_mults],
                "outer_mind":           [round(m, 4) for m in self._om_mults],
                "outer_spirit_content": [round(m, 4) for m in self._os_content_mults],
            },
        }

    def _persist(self) -> None:
        try:
            self._net.save(self._weights_path)
            self._buffer.save(self._buffer_path)
            # Persist graduation-critical counters alongside weights/buffer.
            # EMA multiplier state is persisted so Gate #9 divergence progress
            # survives restarts — without this, every restart resets the EMAs
            # to [1.0,...] and any drift accumulated during silent-mode compute
            # is lost.
            state = {
                "total_train_steps": int(self._total_train_steps),
                "last_loss": float(self._last_loss),
                "recent_losses": [float(x) for x in self._recent_losses[-20:]],
                "phase8_snapshot_taken": bool(self._phase8_snapshot_taken),
                "multipliers_ema": {
                    "inner_body":           [float(x) for x in self._ib_mults],
                    "inner_mind":           [float(x) for x in self._im_mults],
                    "inner_spirit_content": [float(x) for x in self._is_content_mults],
                    "outer_body":           [float(x) for x in self._ob_mults],
                    "outer_mind":           [float(x) for x in self._om_mults],
                    "outer_spirit_content": [float(x) for x in self._os_content_mults],
                },
            }
            _tmp = self._state_path + ".tmp"
            with open(_tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(_tmp, self._state_path)
        except Exception as e:
            logger.warning("[FilterDownV5] Persist failed: %s", e)
