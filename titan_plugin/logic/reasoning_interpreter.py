"""
titan_plugin/logic/reasoning_interpreter.py — V6 Interpreter Layer.

The "premotor cortex" of Titan's cognitive architecture.
Translates abstract reasoning conclusions (COMMIT/ABANDON/plan)
into concrete domain-specific actions.

Architecture:
  Main Reasoning → COMMIT with plan
      ↓
  ReasoningInterpreter → routes to DomainInterpreter
      ↓
  DomainInterpreter → concrete action + confidence
      ↓
  Action system executes (ARC, composition, expression, self-explore)
      ↓
  Outcome feeds back → interpreter learns

Pluggable: new domains added via InterpreterRegistry.register().
Each domain interpreter has its own learning NN (contextual bandit).
"""
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Interpreter NN (Contextual Bandit) ────────────────────────────

class InterpreterPolicyNet:
    """Small NN mapping reasoning plan → action preference scores.

    Fastest learning loop: direct reward per interpretation.
    Architecture: plan_features → h1 → h2 → action_scores.
    """

    def __init__(self, input_dim: int, output_dim: int,
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
        """Forward pass → action scores."""
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "h1": h1, "z1": z1, "h2": h2, "z2": z2}
        return z3

    def select(self, x: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action via epsilon-greedy."""
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        scores = self.forward(x)
        return int(np.argmax(scores))

    def learn(self, x: np.ndarray, action: int, reward: float) -> float:
        """Direct reward learning (contextual bandit)."""
        scores = self.forward(x)
        target = scores.copy()
        target[action] = reward
        error = scores - target

        # Backprop
        d_z3 = error
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
        return float(np.mean(error ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "total_updates": self.total_updates,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("input_dim") != self.input_dim:
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


# ── Domain Interpreter Base Class ─────────────────────────────────

class DomainInterpreter(ABC):
    """Base class for domain-specific reasoning interpretation.

    Subclass this to add new action domains.
    Each domain has its own policy NN that learns from outcomes.
    """

    domain: str = "base"
    action_names: list[str] = []

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._save_dir = cfg.get("save_dir", "./data/interpreter")
        os.makedirs(self._save_dir, exist_ok=True)

        input_dim = self._get_input_dim()
        output_dim = len(self.action_names) if self.action_names else 4
        self._policy = InterpreterPolicyNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_1=cfg.get("policy_hidden_1", 16),
            hidden_2=cfg.get("policy_hidden_2", 8),
            learning_rate=cfg.get("learning_rate", 0.002),
        )
        self._epsilon = cfg.get("epsilon_start", 0.5)
        self._epsilon_decay = cfg.get("epsilon_decay", 0.995)
        self._total_interpretations = 0
        self._total_successes = 0

        # Load saved weights
        self._policy.load(os.path.join(self._save_dir, f"{self.domain}_policy.json"))

    def _get_input_dim(self) -> int:
        """Override to specify input dimension for this domain."""
        return 24  # default: 16D plan features + 8D domain context

    @abstractmethod
    def build_features(self, reasoning_plan: dict, context: dict) -> np.ndarray:
        """Build NN input from reasoning plan + domain context."""

    @abstractmethod
    def decode_action(self, action_idx: int, context: dict) -> dict:
        """Translate action index → domain-specific action dict."""

    def interpret(self, reasoning_output: dict, context: dict) -> dict:
        """Main entry: translate reasoning conclusion → concrete action.

        Returns dict with: domain, action, action_name, confidence, plan_intent
        """
        plan = reasoning_output.get("reasoning_plan", {})
        confidence = reasoning_output.get("confidence", 0.5)

        features = self.build_features(plan, context)
        action_idx = self._policy.select(features, self._epsilon)

        action = self.decode_action(action_idx, context)
        action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"action_{action_idx}"

        self._total_interpretations += 1
        # Decay epsilon
        self._epsilon = max(0.05, self._epsilon * self._epsilon_decay)

        return {
            "domain": self.domain,
            "action": action,
            "action_idx": action_idx,
            "action_name": action_name,
            "confidence": confidence,
            "plan_intent": plan.get("intent", "reflect"),
            "features": features,  # cached for learn()
        }

    def learn(self, interpretation: dict, outcome: float) -> float:
        """Learn from action outcome. Returns loss."""
        features = interpretation.get("features")
        action_idx = interpretation.get("action_idx", 0)
        if features is None:
            return 0.0
        loss = self._policy.learn(features, action_idx, outcome)
        if outcome > 0.5:
            self._total_successes += 1
        return loss

    def save(self) -> None:
        self._policy.save(os.path.join(self._save_dir, f"{self.domain}_policy.json"))

    def get_stats(self) -> dict:
        return {
            "domain": self.domain,
            "total_interpretations": self._total_interpretations,
            "total_successes": self._total_successes,
            "success_rate": self._total_successes / max(1, self._total_interpretations),
            "epsilon": round(self._epsilon, 4),
            "policy_updates": self._policy.total_updates,
        }


# ── Concrete Domain Interpreters ──────────────────────────────────

def _plan_to_features(plan: dict) -> np.ndarray:
    """Extract 16D feature vector from reasoning plan."""
    intent_map = {"reflect": 0, "express_observation": 1, "express_recognition": 2,
                  "express_decision": 3, "observation": 0}
    structure_map = {"observation": 0, "decomposition": 1, "comparison": 2,
                     "association": 3, "conditional": 4, "sequential": 5}

    intent_idx = intent_map.get(plan.get("intent", "reflect"), 0)
    structure_idx = structure_map.get(plan.get("structure", "observation"), 0)
    confidence = plan.get("confidence", 0.5)
    num_elements = len(plan.get("elements", []))

    # One-hot intent (4D) + one-hot structure (6D) + confidence (1D) + elements (1D) + padding (4D)
    vec = np.zeros(16)
    vec[min(intent_idx, 3)] = 1.0
    vec[4 + min(structure_idx, 5)] = 1.0
    vec[10] = confidence
    vec[11] = min(1.0, num_elements / 5.0)
    return vec


class ArcInterpreter(DomainInterpreter):
    """Translates reasoning → ARC game actions (1-7)."""
    domain = "arc"
    action_names = ["action_1", "action_2", "action_3", "action_4",
                    "action_5", "action_6", "action_7"]

    def _get_input_dim(self) -> int:
        return 24  # 16D plan + 8D grid context

    def build_features(self, plan: dict, context: dict) -> np.ndarray:
        plan_f = _plan_to_features(plan)
        # Grid context: stuck_ratio, exploration_rate, reward_trend, available_actions_norm, + 4 spatial
        grid = context.get("grid_features", {})
        spirit = grid.get("inner_spirit", [0.5] * 5)
        spatial = grid.get("spatial", [0.5] * 5)
        ctx = np.array([
            context.get("stuck_ratio", 0.5),
            spirit[0] if len(spirit) > 0 else 0.5,  # exploration
            spirit[1] if len(spirit) > 1 else 0.5,  # reward_trend
            len(context.get("available_actions", [])) / 7.0,
            spatial[0] if len(spatial) > 0 else 0.5,
            spatial[1] if len(spatial) > 1 else 0.5,
            spatial[3] if len(spatial) > 3 else 0.5,  # direction
            spatial[4] if len(spatial) > 4 else 0.5,  # new_colors
        ])
        return np.concatenate([plan_f, ctx])

    def decode_action(self, action_idx: int, context: dict) -> dict:
        available = context.get("available_actions", list(range(1, 8)))
        action_id = available[action_idx % len(available)] if available else action_idx + 1
        return {"action_id": action_id, "source": "interpreter"}


class LanguageInterpreter(DomainInterpreter):
    """Translates reasoning → language composition bias."""
    domain = "language"
    action_names = ["boost_creative", "boost_analytical", "boost_emotional",
                    "boost_observational", "default"]

    def _get_input_dim(self) -> int:
        return 24  # 16D plan + 8D language context

    def build_features(self, plan: dict, context: dict) -> np.ndarray:
        plan_f = _plan_to_features(plan)
        # Language context
        vocab_size = len(context.get("vocabulary", [])) / 200.0
        avg_conf = context.get("avg_composition_confidence", 0.5)
        queue_size = len(context.get("composition_queue", [])) / 10.0
        ctx = np.array([
            vocab_size, avg_conf, queue_size, 0.5, 0.5, 0.5, 0.5, 0.5
        ])
        return np.concatenate([plan_f, ctx])

    def decode_action(self, action_idx: int, context: dict) -> dict:
        biases = {
            0: {"word_boost": ["create", "express", "flow", "light"], "template_bias": "creative"},
            1: {"word_boost": ["think", "observe", "know", "see"], "template_bias": "analytical"},
            2: {"word_boost": ["feel", "warm", "alive", "deep"], "template_bias": "emotional"},
            3: {"word_boost": ["change", "drift", "explore", "new"], "template_bias": "observational"},
            4: {"word_boost": [], "template_bias": "default"},
        }
        return biases.get(action_idx, biases[4])


class ExpressionInterpreter(DomainInterpreter):
    """Translates reasoning → expression trigger context."""
    domain = "expression"
    action_names = ["trigger_speak", "trigger_art", "trigger_music",
                    "trigger_social", "no_trigger"]

    def _get_input_dim(self) -> int:
        return 24

    def build_features(self, plan: dict, context: dict) -> np.ndarray:
        plan_f = _plan_to_features(plan)
        # Expression context
        neuromods = context.get("neuromod_state", {})
        ctx = np.array([
            neuromods.get("DA", 0.5), neuromods.get("5-HT", 0.5),
            neuromods.get("NE", 0.5), neuromods.get("GABA", 0.3),
            neuromods.get("ACh", 0.5), neuromods.get("Endorphin", 0.5),
            context.get("chi_total", 0.5), context.get("fatigue", 0.3),
        ])
        return np.concatenate([plan_f, ctx])

    def decode_action(self, action_idx: int, context: dict) -> dict:
        triggers = {
            0: {"composite": "SPEAK", "reasoning_context": True},
            1: {"composite": "ART", "reasoning_context": True},
            2: {"composite": "MUSIC", "reasoning_context": True},
            3: {"composite": "SOCIAL", "reasoning_context": True},
            4: {"composite": None, "reasoning_context": False},
        }
        return triggers.get(action_idx, triggers[4])


class SelfExplorationInterpreter(DomainInterpreter):
    """Translates reasoning → self-exploration actions."""
    domain = "self_exploration"
    action_names = ["introspect_state", "adjust_attention", "seek_novelty",
                    "consolidate", "rest"]

    def _get_input_dim(self) -> int:
        return 24

    def build_features(self, plan: dict, context: dict) -> np.ndarray:
        plan_f = _plan_to_features(plan)
        ctx = np.array([
            context.get("fatigue", 0.3),
            context.get("chi_total", 0.5),
            context.get("epochs_since_dream", 0) / 1000.0,
            context.get("reasoning_commit_rate", 0.5),
            context.get("expression_fire_rate", 0.5),
            context.get("pi_rate", 0.05),
            context.get("stuck_indicator", 0.0),
            context.get("curiosity_level", 0.5),
        ])
        return np.concatenate([plan_f, ctx])

    def decode_action(self, action_idx: int, context: dict) -> dict:
        actions = {
            0: {"type": "introspect", "target": "current_state"},
            1: {"type": "adjust_attention", "direction": "broaden"},
            2: {"type": "seek_novelty", "urgency": 0.5},
            3: {"type": "consolidate", "trigger_distillation": True},
            4: {"type": "rest", "reduce_activity": True},
        }
        return actions.get(action_idx, actions[4])


class CodingInterpreter(DomainInterpreter):
    """Translates reasoning → coding exercise actions."""
    domain = "coding"
    action_names = ["decompose", "abstract", "implement",
                    "test", "refactor", "compose"]

    def _get_input_dim(self) -> int:
        return 24  # 16D plan + 8D coding context

    def build_features(self, plan: dict, context: dict) -> np.ndarray:
        plan_f = _plan_to_features(plan)
        # Coding context
        ctx = np.array([
            context.get("success_rate", 0.5),
            context.get("concepts_attempted", 0) / 8.0,
            context.get("cooldown_pct", 0.0),
            context.get("sandbox_available", 1.0),
            context.get("chi_total", 0.5),
            context.get("da_level", 0.5),
            context.get("curiosity_level", 0.5),
            context.get("fatigue", 0.3),
        ])
        return np.concatenate([plan_f, ctx])

    def decode_action(self, action_idx: int, context: dict) -> dict:
        actions = {
            0: {"type": "decompose", "target": "problem_analysis"},
            1: {"type": "abstract", "target": "pattern_extraction"},
            2: {"type": "implement", "target": "code_generation"},
            3: {"type": "test", "target": "sandbox_execution"},
            4: {"type": "refactor", "target": "code_improvement"},
            5: {"type": "compose", "target": "solution_assembly"},
        }
        return actions.get(action_idx, actions[2])


# ── Interpreter Registry ──────────────────────────────────────────

class InterpreterRegistry:
    """Registry for domain interpreters. Pluggable — add new domains without core changes."""

    def __init__(self):
        self._interpreters: dict[str, DomainInterpreter] = {}

    def register(self, interpreter: DomainInterpreter) -> None:
        self._interpreters[interpreter.domain] = interpreter
        logger.info("[InterpreterRegistry] Registered: %s (%d actions, input=%dD)",
                    interpreter.domain, len(interpreter.action_names),
                    interpreter._policy.input_dim)

    def get(self, domain: str) -> Optional[DomainInterpreter]:
        return self._interpreters.get(domain)

    def all(self) -> list[DomainInterpreter]:
        return list(self._interpreters.values())

    def save_all(self) -> None:
        for interp in self._interpreters.values():
            interp.save()

    def get_stats(self) -> dict:
        return {d: i.get_stats() for d, i in self._interpreters.items()}


# ── Main Reasoning Interpreter ────────────────────────────────────

class ReasoningInterpreter:
    """Orchestrates interpretation across all registered domains.

    Routes reasoning COMMIT output to the most appropriate domain interpreter
    based on context. Falls through to default behavior if no interpreter
    has confidence.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.registry = InterpreterRegistry()
        self._save_dir = cfg.get("save_dir", "./data/interpreter")

        # Register built-in domain interpreters
        self.registry.register(ArcInterpreter(cfg))
        self.registry.register(LanguageInterpreter(cfg))
        self.registry.register(ExpressionInterpreter(cfg))
        self.registry.register(SelfExplorationInterpreter(cfg))
        self.registry.register(CodingInterpreter(cfg))

        self._total_interpretations = 0
        self._last_interpretation: Optional[dict] = None

    def interpret(self, reasoning_output: dict, context: dict) -> Optional[dict]:
        """Interpret reasoning output for the appropriate domain.

        Args:
            reasoning_output: From ReasoningEngine.tick() with action=COMMIT
            context: Domain context dict with "domain" key to route

        Returns:
            Interpreted action dict, or None if no interpreter matches
        """
        if reasoning_output.get("action") != "COMMIT":
            return None

        domain = context.get("domain", "expression")  # default to expression
        interpreter = self.registry.get(domain)

        if not interpreter:
            # Fallback: try expression (always available)
            interpreter = self.registry.get("expression")

        if not interpreter:
            return None

        result = interpreter.interpret(reasoning_output, context)
        self._total_interpretations += 1
        self._last_interpretation = result

        logger.info("[INTERPRET] %s → %s (conf=%.2f, intent=%s, eps=%.3f)",
                    domain, result.get("action_name", "?"),
                    result.get("confidence", 0),
                    result.get("plan_intent", "?"),
                    interpreter._epsilon)

        return result

    def learn_outcome(self, domain: str, interpretation: dict, outcome: float) -> None:
        """Feed action outcome back to the domain interpreter."""
        interpreter = self.registry.get(domain)
        if interpreter and interpretation:
            loss = interpreter.learn(interpretation, outcome)
            logger.debug("[INTERPRET] %s learned: outcome=%.3f loss=%.4f",
                         domain, outcome, loss)

    def save_all(self) -> None:
        self.registry.save_all()

    def get_stats(self) -> dict:
        return {
            "total_interpretations": self._total_interpretations,
            "domains": self.registry.get_stats(),
        }
