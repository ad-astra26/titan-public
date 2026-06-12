"""titan_hcl/synthesis/outer_reasoning.py — OML Phase C, piece 3.

The OUTER lower chain-builder that ``OuterMetaReasoningEngine``'s ``DELEGATE``
targets (RFP_synthesis_self_learning_meta_reasoning §7.C, both-engines reuse).

It REUSES ``logic/reasoning.py``'s logic-primitive machinery with **zero inner
touch** — it only *imports* the module-level primitive functions + the
``ReasoningPolicyNet`` / ``ReasoningTransitionBuffer`` classes; it never
instantiates ``ReasoningEngine`` (the inner, neuromod/spirit-coupled engine
that lives in ``cognitive_worker``). The inner Titan reasoning loop is therefore
untouched (the §7.C "structural-safety" binding).

What is OUTER here (vs the inner engine):
  • the observation is an **outer-problem signature** — the 30-D outer feature
    vector (``OUTER_POLICY_INPUT_DIM``; what ``mean_features`` /
    ``recent_reward_tuples`` produce, RFP §7.C #4) — NOT the inner
    ``CognitivePerception`` / ``gut_signals`` / ``body_state``.
  • there is **no neuromod / spirit coupling** — the primitive functions are
    fed a neutral constant neuromod dict (0.5 = honest "no signal"; nothing
    fabricated). No ``CognitivePerception``, no ``SpiritReasoningObserver``,
    no rFP-α reward shaping.
  • the terminal reward is supplied from OUTSIDE — the OuterMeta's oracle
    ``EVALUATE`` (piece 5) calls ``train_terminal(reward)``.

The OuterMeta ``DELEGATE.{full,quick,biased}_chain`` steers primitive selection
via ``set_strategy_bias(8D)`` — the SAME 8-D bias surface the inner engine
exposes at ``reasoning.py:set_strategy_bias`` — so ``DELEGATE`` stays WHOLE
(no cherry-pick).
"""
import logging
import os
import time
from typing import Optional

import numpy as np

# Zero-inner-touch reuse: import the primitive FUNCTIONS + the policy/buffer
# classes from the inner module. We never construct ``ReasoningEngine`` itself.
from titan_hcl.logic.reasoning import (
    NUM_ACTIONS,
    PRIMITIVE_FUNCTIONS,
    PRIMITIVES,
    ReasoningPolicyNet,
    ReasoningTransitionBuffer,
    _primitive_associate,
    _primitive_loop,
    _primitive_negate,
)
from titan_hcl.synthesis.outer_meta_policy import OUTER_POLICY_INPUT_DIM

logger = logging.getLogger(__name__)

# ── Dimensions ────────────────────────────────────────────────────────
# The outer-problem observation IS the 30-D outer feature vector (the same
# space OuterMetaPolicy operates in — single source of truth).
OUTER_OBS_DIM = OUTER_POLICY_INPUT_DIM            # 30
_CHAIN_STATE_DIM = 3                              # [len_norm, confidence, momentum]
OUTER_REASON_INPUT_DIM = OUTER_OBS_DIM + _CHAIN_STATE_DIM  # 33

# Outer plane has NO neuromod coupling. The reused primitives read a neuromod
# dict to set thresholds; 0.5 is the honest neutral ("no signal") — it makes
# every primitive's threshold land at its midpoint and fabricates nothing.
_NEUTRAL_NEUROMODS = {
    "DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.5,
}

_CONCLUDE_IDX = PRIMITIVES.index("CONCLUDE")


class OuterReasoningEngine:
    """Outer logic-primitive chain-builder (piece 3).

    Mirrors the inner ``ReasoningEngine`` chain loop — select a primitive →
    execute → update confidence → conclude COMMIT/HOLD/ABANDON — but over an
    outer-problem observation, with its OWN fresh ``ReasoningPolicyNet`` +
    ``ReasoningTransitionBuffer`` and no inner neuromod/spirit/body coupling.
    """

    def __init__(self, config: Optional[dict] = None, save_dir: Optional[str] = None):
        cfg = config or {}
        self.max_chain_length = int(cfg.get("max_chain_length", 8))
        self.min_chain_length = int(cfg.get("min_chain_length", 3))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.6))
        # Below COMMIT threshold but still promising → HOLD (keep the chain
        # alive for a later explore tick) rather than ABANDON. The outer analog
        # of the inner engine's body-not-ready HOLD (outer has no body_state).
        self.hold_floor = float(cfg.get("hold_floor", 0.45))

        self.policy = ReasoningPolicyNet(
            input_dim=OUTER_REASON_INPUT_DIM,
            hidden_1=int(cfg.get("policy_h1", 32)),
            hidden_2=int(cfg.get("policy_h2", 16)),
            learning_rate=float(cfg.get("learning_rate", 0.001)),
        )
        self.buffer = ReasoningTransitionBuffer(max_size=int(cfg.get("buffer_size", 2000)))

        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.policy.load(os.path.join(save_dir, "outer_reasoning_policy.json"))
            self.buffer.load(os.path.join(save_dir, "outer_reasoning_buffer.json"))

        # Seeded problem observation (30-D) — set per chain via set_problem().
        self._problem_obs = np.zeros(OUTER_OBS_DIM, dtype=np.float64)

        # Per-chain state.
        self.chain: list[str] = []
        self.chain_results: list[dict] = []
        self.confidence: float = 0.5
        self.is_active: bool = False
        self._last_result: Optional[dict] = None
        self._loop_count: int = 0
        self._confidence_gains: int = 0
        self._chain_start_time: float = 0.0
        self._strategy_bias: Optional[np.ndarray] = None  # 8-D, set by OuterMeta DELEGATE
        # Chain transitions for terminal credit assignment (filled by train_terminal).
        self._chain_transitions: list[tuple] = []

        # Lifetime telemetry.
        self._total_chains: int = 0
        self._action_counts: dict[str, int] = {"COMMIT": 0, "HOLD": 0, "ABANDON": 0}
        self._last_action: str = ""

    # ── DELEGATE injection surface (parity with reasoning.py set_strategy_bias) ──
    def set_strategy_bias(self, bias) -> None:
        """Set the 8-D strategy bias from the OuterMeta DELEGATE. Applied to
        primitive selection; cleared on chain conclusion."""
        self._strategy_bias = (
            np.array(bias, dtype=np.float32) if bias is not None else None)
        if self._strategy_bias is not None:
            logger.debug(
                "[OuterReasoning] strategy bias set: %s",
                {PRIMITIVES[i]: round(float(b), 2)
                 for i, b in enumerate(self._strategy_bias) if abs(b) > 0.01})

    def clear_strategy_bias(self) -> None:
        self._strategy_bias = None

    def set_problem(self, problem_obs) -> None:
        """Seed the chain with an outer-problem observation — the 30-D outer
        feature vector (``mean_features`` / a recalled ``Reasoning`` signature).
        Pad/trim/NaN-safe so a malformed input never breaks a chain."""
        arr = np.asarray(list(problem_obs) if problem_obs is not None else [],
                         dtype=np.float64).ravel()
        obs = np.zeros(OUTER_OBS_DIM, dtype=np.float64)
        n = min(OUTER_OBS_DIM, arr.shape[0])
        if n:
            obs[:n] = np.nan_to_num(arr[:n], nan=0.0, posinf=0.0, neginf=0.0)
        self._problem_obs = obs

    # ── Chain orchestration ────────────────────────────────────────────
    def start_chain(self) -> None:
        """Initialize a new reasoning chain (parity with reasoning.py:_start_chain)."""
        self.chain = []
        self.chain_results = []
        self.confidence = 0.5
        self.is_active = True
        self._last_result = None
        self._loop_count = 0
        self._confidence_gains = 0
        self._chain_start_time = time.time()
        self._chain_transitions = []
        self._total_chains += 1

    def _build_policy_input(self) -> np.ndarray:
        """Outer input-builder — the OUTER analog of reasoning.py:_build_policy_input.

        [ problem_obs (30-D outer feature signature) ]
        ⊕ [ chain_len_norm, confidence, momentum ]   (3-D generic chain state)

        ``momentum`` = fraction of executed steps that raised confidence — an
        emergent, neuromod-free replacement for the inner persistence_ratio
        (which was 5-HT/GABA-derived; outer has no neuromods).
        """
        n = max(1, len(self.chain))
        momentum = self._confidence_gains / n
        chain_state = np.array([
            len(self.chain) / max(1, self.max_chain_length),
            self.confidence,
            momentum,
        ], dtype=np.float64)
        return np.concatenate([self._problem_obs, chain_state])

    def _run_primitive(self, action_name: str) -> dict:
        """Execute one reused logic primitive over the outer observation."""
        obs = self._problem_obs
        wm = self.chain_results
        nm = _NEUTRAL_NEUROMODS

        if action_name == "LOOP":
            result = _primitive_loop(obs, wm, nm, self._last_result)
            if not result.get("continue") or self._loop_count >= 3:
                result["continue"] = False
            else:
                self._loop_count += 1
        elif action_name == "NEGATE":
            result = _primitive_negate(obs, wm, nm, self._last_result)
        elif action_name == "ASSOCIATE":
            # Outer has no mini-reasoner registry — working-memory search only.
            result = _primitive_associate(obs, wm, nm, mini_registry=None)
        else:
            fn = PRIMITIVE_FUNCTIONS.get(action_name)
            result = fn(obs, wm, nm) if fn else {"primitive": action_name, "error": "unknown"}
        return result

    def _update_confidence(self, result: dict) -> None:
        """Outer Mind-Feeling update — the inner _update_confidence logic MINUS
        the spirit nudge + gut-agreement (those are inner-coupled)."""
        primitive = result.get("primitive", "")
        gained = False
        if result.get("significant") or result.get("eureka") or result.get("condition_met"):
            self.confidence = min(1.0, self.confidence + 0.1)
            gained = True
        elif primitive == "DECOMPOSE" and result.get("parts"):
            active_total = sum(p.get("active_dims", 0) for p in result["parts"].values())
            if active_total > 5:
                self.confidence = min(1.0, self.confidence + 0.05)
                gained = True
        elif primitive == "SEQUENCE" and result.get("steps_completed", 0) >= 3:
            self.confidence = min(1.0, self.confidence + 0.05)
            gained = True
        elif (primitive == "ASSOCIATE" and result.get("found")
              and result.get("relevance", 0) > 0.3):
            self.confidence = min(1.0, self.confidence + 0.03)
            gained = True
        elif result.get("found") is False or result.get("continue") is False:
            self.confidence = max(0.0, self.confidence - 0.05)
        if gained:
            self._confidence_gains += 1

    def _extract_plan(self) -> list[str]:
        """A compact, ordered summary of the primitives executed this chain."""
        return list(self.chain)

    def step(self, temperature: float = 1.0) -> dict:
        """Advance the chain one primitive (or conclude). Returns a dict whose
        ``action`` is CONTINUE / COMMIT / HOLD / ABANDON."""
        if not self.is_active:
            self.start_chain()

        policy_input = self._build_policy_input()
        action_idx = int(self.policy.select_action(
            policy_input, temperature, strategy_bias=self._strategy_bias))
        action_name = PRIMITIVES[action_idx]

        # Enforce minimum chain length — redirect a premature CONCLUDE to a
        # non-CONCLUDE primitive (parity with reasoning.py:1416).
        if action_name == "CONCLUDE" and len(self.chain) < self.min_chain_length:
            action_idx = int(np.random.choice(NUM_ACTIONS - 1))
            action_name = PRIMITIVES[action_idx]

        if action_name == "CONCLUDE" or len(self.chain) >= self.max_chain_length:
            return self._conclude(policy_input)

        result = self._run_primitive(action_name)
        self.chain.append(action_name)
        self.chain_results.append(result)
        self._last_result = result
        if action_name != "LOOP":
            self._loop_count = 0
        self._update_confidence(result)

        next_input = self._build_policy_input()
        # Reward is filled at terminal (train_terminal) — intermediate=0.
        self.buffer.record(state=policy_input, action=action_idx, reward=0.0,
                           next_state=next_input, done=False)
        self._chain_transitions.append((policy_input, action_idx))

        return {
            "action": "CONTINUE",
            "primitive": action_name,
            "result": result,
            "chain_length": len(self.chain),
            "confidence": round(self.confidence, 4),
        }

    def _conclude(self, policy_input: np.ndarray) -> dict:
        """End the chain — COMMIT (confident), HOLD (promising), or ABANDON."""
        if self.confidence >= self.confidence_threshold:
            action = "COMMIT"
        elif self.confidence >= self.hold_floor:
            action = "HOLD"
        else:
            action = "ABANDON"

        conclusion = {
            "action": action,
            "confidence": round(self.confidence, 4),
            "chain_length": len(self.chain),
            "chain": list(self.chain),
            "duration_s": round(time.time() - self._chain_start_time, 2),
            "reasoning_plan": self._extract_plan(),
        }

        # Terminal transition (reward filled by train_terminal).
        self.buffer.record(state=policy_input, action=_CONCLUDE_IDX,
                           reward=0.0, next_state=policy_input, done=True)
        self._chain_transitions.append((policy_input, _CONCLUDE_IDX))

        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        self._last_action = action

        # Reset chain state. _chain_transitions is preserved until train_terminal
        # (or the next start_chain) so the OuterMeta EVALUATE reward can credit
        # the chain just built. strategy_bias clears on chain end (parity inner).
        self.is_active = False
        self.chain = []
        self.chain_results = []
        self._last_result = None
        self._strategy_bias = None

        logger.debug("[OuterReasoning] %s conf=%.3f chain=%d",
                     action, conclusion["confidence"], conclusion["chain_length"])
        return conclusion

    def run_chain(self, problem_obs=None, strategy_bias=None,
                  temperature: float = 1.0, max_steps: Optional[int] = None) -> dict:
        """Run a full chain to conclusion in one call — the surface the OuterMeta
        DELEGATE invokes. Returns the COMMIT/HOLD/ABANDON conclusion dict."""
        if problem_obs is not None:
            self.set_problem(problem_obs)
        self.start_chain()
        if strategy_bias is not None:
            self.set_strategy_bias(strategy_bias)

        cap = max_steps if max_steps is not None else self.max_chain_length + 2
        for _ in range(cap):
            out = self.step(temperature)
            if out.get("action") != "CONTINUE":
                return out
        # Safety net — force a conclusion if the cap is hit mid-chain.
        return self._conclude(self._build_policy_input())

    def train_terminal(self, reward: float, baseline: float = 0.3) -> dict:
        """Train the fresh outer ReasoningPolicyNet toward the OuterMeta
        EVALUATE reward, crediting every action taken in the chain just
        concluded (policy-gradient with a fixed baseline). Also back-fills the
        chain's buffer transitions with the terminal reward for later
        dream-consolidation."""
        transitions = self._chain_transitions
        if not transitions:
            return {"trained": 0, "reason": "no_transitions"}
        advantage = float(reward) - float(baseline)
        trained = 0
        for state, action_idx in transitions:
            self.policy.train_step(state, int(action_idx), advantage)
            trained += 1
        # Back-fill the terminal transition's reward in the buffer.
        self.buffer.update_last_reward(float(reward))
        self._chain_transitions = []
        return {"trained": trained, "reward": float(reward),
                "advantage": round(advantage, 4),
                "total_updates": self.policy.total_updates}

    # ── Persistence ────────────────────────────────────────────────────
    def save_all(self) -> None:
        if not self.save_dir:
            return
        self.policy.save(os.path.join(self.save_dir, "outer_reasoning_policy.json"))
        self.buffer.save(os.path.join(self.save_dir, "outer_reasoning_buffer.json"))

    def get_stats(self) -> dict:
        return {
            "total_chains": self._total_chains,
            "action_counts": dict(self._action_counts),
            "last_action": self._last_action,
            "policy_updates": int(self.policy.total_updates),
            "buffer_size": self.buffer.size(),
            "input_dim": OUTER_REASON_INPUT_DIM,
        }


__all__ = [
    "OuterReasoningEngine",
    "OUTER_OBS_DIM",
    "OUTER_REASON_INPUT_DIM",
]
