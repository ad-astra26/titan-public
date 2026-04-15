"""
Mini-Reasoning Framework — distributed intelligence for Titan's cognitive architecture.

Each perceptual subsystem gets its own lightweight reasoning engine + persistent
mini-experience memory. These run at Body/Mind/Spirit rate (fast, reflexive) and
produce structured conclusions that the main Mind reasoning can query.

Architecture mirrors DomainInterpreter (reasoning_interpreter.py) with:
- MiniPolicyNet: tiny NN selecting which of 3 domain primitives to fire
- MiniExperienceBuffer: FIFO buffer for REINFORCE + dream IQL training
- MiniReasoner ABC: pluggable base for domain-specific mini-reasoning
- MiniReasonerRegistry: auto-discovery registry for all mini-modules
"""
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# ── Mini Policy Network ─────────────────────────────────────────────────────

class MiniPolicyNet:
    """Tiny NN selecting which of 3 domain primitives to fire.

    Architecture: observation → 16 → 8 → 3 primitive scores.
    Training: REINFORCE with delayed reward from main reasoning outcome.
    Pattern: InterpreterPolicyNet (reasoning_interpreter.py:38).
    """

    def __init__(self, input_dim: int, output_dim: int = 3,
                 hidden_1: int = 16, hidden_2: int = 8,
                 learning_rate: float = 0.002):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Xavier initialization
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / hidden_1)
        s3 = math.sqrt(2.0 / hidden_2)
        self.w1 = np.random.randn(input_dim, hidden_1) * s1
        self.b1 = np.zeros(hidden_1)
        self.w2 = np.random.randn(hidden_1, hidden_2) * s2
        self.b2 = np.zeros(hidden_2)
        self.w3 = np.random.randn(hidden_2, output_dim) * s3
        self.b3 = np.zeros(output_dim)
        self.total_updates = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass → primitive scores."""
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "h1": h1, "z1": z1, "h2": h2, "z2": z2}
        return z3

    def select(self, x: np.ndarray, temperature: float = 1.0) -> int:
        """Select primitive via softmax with temperature."""
        scores = self.forward(x)
        # Softmax selection (not epsilon-greedy — primitives have natural ordering)
        scores_shifted = scores - np.max(scores)
        exp_s = np.exp(scores_shifted / max(0.01, temperature))
        probs = exp_s / (np.sum(exp_s) + 1e-10)
        return int(np.random.choice(len(probs), p=probs))

    def train_step(self, x: np.ndarray, action: int, advantage: float) -> float:
        """REINFORCE policy gradient step."""
        scores = self.forward(x)
        # Softmax probabilities
        scores_shifted = scores - np.max(scores)
        exp_s = np.exp(scores_shifted)
        probs = exp_s / (np.sum(exp_s) + 1e-10)

        # Policy gradient: d_log_pi * advantage
        d_z3 = probs.copy()
        d_z3[action] -= 1.0  # grad of -log(pi) w.r.t. logits
        d_z3 *= -advantage   # REINFORCE: -advantage * grad(log_pi)

        # Backprop (same pattern as InterpreterPolicyNet.learn)
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
        return float(np.mean(d_z3 ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "total_updates": self.total_updates,
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
                logger.warning("[MiniPolicy] Dimension mismatch: saved=%d, current=%d",
                               data.get("input_dim"), self.input_dim)
                return False
            self.w1 = np.array(data["w1"])
            self.b1 = np.array(data["b1"])
            self.w2 = np.array(data["w2"])
            self.b2 = np.array(data["b2"])
            self.w3 = np.array(data["w3"])
            self.b3 = np.array(data["b3"])
            self.total_updates = data.get("total_updates", 0)
            return True
        except Exception:
            return False


# ── Mini Experience Buffer ───────────────────────────────────────────────────

class MiniExperienceBuffer:
    """FIFO buffer for mini-reasoner transitions.

    Stores (observation, primitive_idx, outcome) tuples for training.
    Pattern: NervousTransitionBuffer (neural_reflex_net.py:301).
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._observations: list[list[float]] = []
        self._actions: list[int] = []
        self._outcomes: list[float] = []
        self._consulted: list[bool] = []
        self._timestamps: list[float] = []

    def __len__(self) -> int:
        return len(self._observations)

    def add(self, observation: list, action: int, outcome: float = 0.0,
            consulted: bool = False) -> None:
        self._observations.append(list(observation))
        self._actions.append(action)
        self._outcomes.append(outcome)
        self._consulted.append(consulted)
        self._timestamps.append(time.time())
        # FIFO eviction
        if len(self._observations) > self.max_size:
            self._observations.pop(0)
            self._actions.pop(0)
            self._outcomes.pop(0)
            self._consulted.pop(0)
            self._timestamps.pop(0)

    def update_last_outcome(self, outcome: float, consulted: bool = True) -> None:
        if self._outcomes:
            self._outcomes[-1] = outcome
            self._consulted[-1] = consulted

    def sample(self, batch_size: int) -> tuple:
        n = len(self._observations)
        if n == 0:
            return (np.array([]), np.array([]), np.array([]))
        indices = random.sample(range(n), min(batch_size, n))
        return (
            np.array([self._observations[i] for i in indices], dtype=np.float64),
            np.array([self._actions[i] for i in indices], dtype=np.int64),
            np.array([self._outcomes[i] for i in indices], dtype=np.float64),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "observations": self._observations[-self.max_size:],
            "actions": self._actions[-self.max_size:],
            "outcomes": self._outcomes[-self.max_size:],
            "consulted": [bool(c) for c in self._consulted[-self.max_size:]],
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
            self._observations = data.get("observations", [])
            self._actions = data.get("actions", [])
            self._outcomes = data.get("outcomes", [])
            self._consulted = data.get("consulted", [])
            self._timestamps = [0.0] * len(self._observations)
            return True
        except Exception as e:
            logger.warning("[MiniBuffer] Failed to load %s: %s", path, e)
            return False


# ── Mini Reasoner ABC ────────────────────────────────────────────────────────

class MiniReasoner(ABC):
    """Base class for domain-specific mini-reasoning modules.

    Subclass this to add new cognitive domains to Titan.
    Registration is automatic via the MiniReasonerRegistry.

    Each subclass defines:
      - domain: str — unique identifier ("spatial", "observation", ...)
      - primitives: list[str] — 3 domain-specific primitive names
      - rate_tier: str — "body" | "mind" | "spirit" (determines tick frequency)
      - observation_dim: int — input dimension for this domain
    """
    domain: str = "base"
    primitives: list = []
    rate_tier: str = "body"
    observation_dim: int = 20

    def __init__(self, save_dir: str = "./data/mini_reasoning"):
        self._save_dir = save_dir
        self._policy = MiniPolicyNet(
            input_dim=self.observation_dim,
            output_dim=len(self.primitives),
        )
        self._buffer = MiniExperienceBuffer(max_size=500)
        self._latest_summary: dict = {}
        self._latest_observation: np.ndarray = np.zeros(self.observation_dim)
        self._total_ticks: int = 0
        self._total_consulted: int = 0
        self._temperature: float = 1.5  # Start exploratory, decay over time

    # ── Abstract methods (subclass must implement) ───────────────────

    @abstractmethod
    def perceive(self, context: dict) -> np.ndarray:
        """Extract domain-specific observation from global context.
        Returns: numpy array of shape (observation_dim,)
        """

    @abstractmethod
    def execute_primitive(self, primitive_idx: int, observation: np.ndarray) -> dict:
        """Execute one domain primitive. Returns structured conclusion dict.
        Must include: {"relevance": float, "confidence": float, ...domain_specific}
        """

    @abstractmethod
    def format_summary(self) -> dict:
        """Format latest conclusion for main reasoning query.
        Must include: {"relevance": float, "confidence": float, "primitive": str}
        """

    # ── Provided by base class ───────────────────────────────────────

    def tick(self, context: dict) -> dict:
        """Select primitive, execute, store in buffer. Called at rate_tier frequency."""
        try:
            observation = self.perceive(context)
            if observation is None or len(observation) == 0:
                return self._latest_summary
            # Pad/truncate to expected dim
            if len(observation) < self.observation_dim:
                observation = np.concatenate([observation, np.zeros(self.observation_dim - len(observation))])
            elif len(observation) > self.observation_dim:
                observation = observation[:self.observation_dim]
            self._latest_observation = observation

            # Select primitive via policy
            primitive_idx = self._policy.select(observation, temperature=self._temperature)
            primitive_idx = min(primitive_idx, len(self.primitives) - 1)

            # Execute
            conclusion = self.execute_primitive(primitive_idx, observation)
            conclusion["primitive"] = self.primitives[primitive_idx]
            conclusion["primitive_idx"] = primitive_idx

            # Store in buffer (outcome updated later via feedback)
            self._buffer.add(
                observation=observation.tolist(),
                action=primitive_idx,
                outcome=0.0,
                consulted=False,
            )

            # Update summary
            self._latest_summary = conclusion
            self._total_ticks += 1

            # Decay temperature (more exploitative over time)
            if self._total_ticks % 100 == 0 and self._temperature > 0.5:
                self._temperature *= 0.995

            return conclusion
        except Exception as e:
            logger.debug("[MiniReasoner:%s] tick error: %s", self.domain, e)
            return self._latest_summary

    def query(self) -> dict:
        """Return latest summary for main reasoning consultation."""
        summary = self.format_summary()
        summary["domain"] = self.domain
        summary["ticks"] = self._total_ticks
        return summary

    def feedback(self, outcome: float) -> float:
        """Receive outcome from main reasoning. Train policy online."""
        self._buffer.update_last_outcome(outcome, consulted=True)
        self._total_consulted += 1
        # Online REINFORCE step
        if len(self._buffer) > 0 and self._latest_observation is not None:
            obs, acts, _ = self._buffer.sample(1)
            if len(obs) > 0:
                advantage = outcome - 0.3  # Baseline
                loss = self._policy.train_step(
                    self._latest_observation, acts[0], advantage)
                return loss
        return 0.0

    def consolidate(self, boost_factor: float = 2.0) -> dict:
        """Dream consolidation: IQL batch training from experience buffer."""
        if len(self._buffer) < 10:
            return {"domain": self.domain, "samples": 0}

        batch_size = min(32, len(self._buffer))
        obs, actions, outcomes = self._buffer.sample(batch_size)
        if len(obs) == 0:
            return {"domain": self.domain, "samples": 0}

        total_loss = 0.0
        trained = 0
        # Boost learning rate during dreams
        original_lr = self._policy.lr
        self._policy.lr = min(original_lr * boost_factor, 0.01)

        for i in range(len(obs)):
            advantage = outcomes[i] - 0.3
            loss = self._policy.train_step(obs[i], int(actions[i]), advantage)
            total_loss += loss
            trained += 1

        self._policy.lr = original_lr
        avg_loss = total_loss / max(1, trained)
        return {
            "domain": self.domain,
            "samples": trained,
            "loss": round(avg_loss, 6),
            "buffer_size": len(self._buffer),
            "total_updates": self._policy.total_updates,
        }

    def save(self) -> None:
        domain_dir = os.path.join(self._save_dir, self.domain)
        os.makedirs(domain_dir, exist_ok=True)
        self._policy.save(os.path.join(domain_dir, "policy.json"))
        self._buffer.save(os.path.join(domain_dir, "buffer.json"))
        # Save metadata
        meta = {
            "domain": self.domain, "total_ticks": self._total_ticks,
            "total_consulted": self._total_consulted,
            "temperature": self._temperature,
        }
        meta_path = os.path.join(domain_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def load(self) -> bool:
        domain_dir = os.path.join(self._save_dir, self.domain)
        loaded_policy = self._policy.load(os.path.join(domain_dir, "policy.json"))
        loaded_buffer = self._buffer.load(os.path.join(domain_dir, "buffer.json"))
        meta_path = os.path.join(domain_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self._total_ticks = meta.get("total_ticks", 0)
                self._total_consulted = meta.get("total_consulted", 0)
                self._temperature = meta.get("temperature", 1.5)
            except Exception:
                pass
        if loaded_policy:
            logger.info("[MiniReasoner:%s] Loaded policy (%d updates) + buffer (%d items)",
                        self.domain, self._policy.total_updates, len(self._buffer))
        return loaded_policy

    def get_stats(self) -> dict:
        return {
            "domain": self.domain,
            "primitives": self.primitives,
            "rate_tier": self.rate_tier,
            "observation_dim": self.observation_dim,
            "total_ticks": self._total_ticks,
            "total_consulted": self._total_consulted,
            "policy_updates": self._policy.total_updates,
            "buffer_size": len(self._buffer),
            "temperature": round(self._temperature, 3),
            "latest_summary": self._latest_summary,
        }


# ── Mini Reasoner Registry ──────────────────────────────────────────────────

class MiniReasonerRegistry:
    """Auto-discovery registry for mini-reasoning modules.

    Spirit worker calls registry.tick_all(context, rate_tier) each computation gate.
    Main reasoning calls registry.query(domain) for summaries.
    Dream consolidation calls registry.consolidate_all().
    """

    def __init__(self, save_dir: str = "./data/mini_reasoning"):
        self._reasoners: dict[str, MiniReasoner] = {}
        self._save_dir = save_dir

    def register(self, reasoner: MiniReasoner) -> None:
        reasoner._save_dir = self._save_dir
        self._reasoners[reasoner.domain] = reasoner
        logger.info("[MiniRegistry] Registered: %s (%dD, %s-rate, primitives=%s)",
                    reasoner.domain, reasoner.observation_dim,
                    reasoner.rate_tier, reasoner.primitives)

    def get(self, domain: str):
        return self._reasoners.get(domain)

    def all(self) -> list:
        return list(self._reasoners.values())

    def tick_all(self, context: dict, rate_tier: str) -> dict:
        """Tick all mini-reasoners matching the given rate tier."""
        results = {}
        for domain, reasoner in self._reasoners.items():
            if reasoner.rate_tier == rate_tier:
                result = reasoner.tick(context)
                results[domain] = result
        return results

    def query(self, domain: str) -> dict:
        """Query a single mini-reasoner's latest summary."""
        reasoner = self._reasoners.get(domain)
        if reasoner:
            return reasoner.query()
        return {}

    def query_all(self) -> dict:
        """Query all mini-reasoners. Returns {domain: summary}."""
        return {domain: r.query() for domain, r in self._reasoners.items()}

    def feedback(self, domain: str, outcome: float) -> None:
        """Feed outcome from main reasoning back to a mini-reasoner."""
        reasoner = self._reasoners.get(domain)
        if reasoner:
            reasoner.feedback(outcome)

    def feedback_all(self, outcome: float) -> None:
        """Feed outcome to all mini-reasoners (when domain is unknown)."""
        for reasoner in self._reasoners.values():
            reasoner.feedback(outcome)

    def consolidate_all(self, boost_factor: float = 2.0) -> dict:
        """Dream consolidation for all mini-reasoners."""
        results = {}
        for domain, reasoner in self._reasoners.items():
            stats = reasoner.consolidate(boost_factor=boost_factor)
            results[domain] = stats
        return results

    def save_all(self) -> None:
        for reasoner in self._reasoners.values():
            try:
                reasoner.save()
            except Exception as e:
                logger.warning("[MiniRegistry] Save failed for %s: %s", reasoner.domain, e)

    def load_all(self) -> None:
        for reasoner in self._reasoners.values():
            try:
                reasoner.load()
            except Exception as e:
                logger.warning("[MiniRegistry] Load failed for %s: %s", reasoner.domain, e)

    def get_stats(self) -> dict:
        return {
            domain: r.get_stats() for domain, r in self._reasoners.items()
        }
