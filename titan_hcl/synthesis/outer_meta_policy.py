"""titan_hcl.synthesis.outer_meta_policy — the learned OUTER decision operator.

RFP_synthesis_self_learning_meta_reasoning Phase 1 (§7.A). Titan learns — per
prompt — *which action to take* (answer directly / fire a tool / delegate a
learned skill / research / honestly say "I don't know"), trained from OUTCOMES
(oracle verdict in Phase 1 / RLVR), instead of today's static
`regex + fixed-threshold grounded_router`.

INV-OML-1: the policy is RL **scaffolding only** — it picks an *action index*,
never a grounding/understanding score (that stays the symbolic, tx-verifiable
`time_cost`; BRAIN owns it, INV-OML-2). The LLM is suggester + oracle, never the
router.

This clones the numpy `MetaPolicy` (logic/meta_reasoning.py) — a small ReLU MLP
trained by REINFORCE-with-baseline. No torch (the IQL gatekeeper was retired,
SPEC v0.27.0; §3 scope-fence). Weights publish to SHM as a fixed float32 vector
(`OUTER_META_POLICY_STATE_SPEC`) so the agno DECIDE path reads them O(1) with no
sync RPC (INV-OML-7 / G19/G20); the self_learning_worker is the single writer
(INV-OML-8 / G21).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from titan_hcl.core.state_registry import RegistrySpec
from titan_hcl.logic.sage.grounded_router import (
    MODE_RESEARCH,
    MODE_SHADOW,
    MODE_SKILL_DELEGATE,
    MODE_SOVEREIGN,
    MODE_TOOL_ORACLE,
)

# ── Action space ─────────────────────────────────────────────────────
# Order is the policy's output index → MUST stay stable (SHM layout + save/load
# migration depend on it). Append-only if ever extended.
OUTER_ACTIONS: tuple[str, ...] = (
    "direct",          # answer from substrate + light LLM narration → MODE_SOVEREIGN
    "tool",            # fire the deterministic coding_sandbox / oracle → MODE_TOOL_ORACLE
    "skill_delegate",  # delegate a verified learned skill → MODE_SKILL_DELEGATE
    "research",        # "I don't know — look it up" → MODE_RESEARCH
    "IDK",             # honest "I don't know" (no affordable path) → MODE_SHADOW
)
NUM_OUTER_ACTIONS = len(OUTER_ACTIONS)

# action index → grounded_router MODE dispatch string (consumed on the agno path)
_ACTION_TO_MODE: tuple[str, ...] = (
    MODE_SOVEREIGN,
    MODE_TOOL_ORACLE,
    MODE_SKILL_DELEGATE,
    MODE_RESEARCH,
    MODE_SHADOW,
)


def action_index_to_mode(idx: int) -> str:
    """Map a policy action index → the grounded_router MODE_* dispatch string."""
    if 0 <= idx < NUM_OUTER_ACTIONS:
        return _ACTION_TO_MODE[idx]
    return MODE_SOVEREIGN  # safe default (never observed; defensive)


def action_index_to_name(idx: int) -> str:
    if 0 <= idx < NUM_OUTER_ACTIONS:
        return OUTER_ACTIONS[idx]
    return "direct"


# ── Phase-1 feature schema (curated LOCAL subset) ────────────────────
# Phase 1 uses ONLY what is locally available on the agno hot path — recall /
# skill / engram match signals + the published MSL identity scalars. The FULL
# MSL `context[20]` vector (msl.py infer()) is NOT on the agno path today; its
# consumption is Phase 3 (Q4). Keep this layout stable (it is the policy input
# dim). Slot 0 is a constant bias term.
OUTER_FEATURE_NAMES: tuple[str, ...] = (
    "bias",                    # 0  constant 1.0
    "recall_top_cosine",       # 1  [0,1] top recall similarity (D3)
    "recall_count_norm",       # 2  [0,1] min(n_recalled,10)/10
    "skill_utility",           # 3  [0,1] top skill_cell time_cost proficiency (D5)
    "skill_matched",           # 4  {0,1} a skill cell matched the goal_class
    "engram_ground",           # 5  [0,1] matched Engram groundedness (D5)
    "requires_tool",           # 6  {0,1} tool-intent regex (D5)
    "has_code_signal",         # 7  {0,1} computational shape (intent.code present)
    "msl_i_confidence",        # 8  [0,1] published MSL identity confidence
    "msl_attention_entropy",   # 9  [0,1] published MSL attention entropy (normalized)
    "msl_concept_confidence",  # 10 [0,1] mean published MSL concept confidences
)
OUTER_POLICY_INPUT_DIM = len(OUTER_FEATURE_NAMES)  # 11

# Hidden layer widths (small input → small net; cf. MetaPolicy 80→40→20→6).
_H1 = 16
_H2 = 8

# Flat SHM layout: all weights/biases (row-major) then 2 metadata scalars
# (total_updates, reward_baseline). Reconstructed by from_flat() using the
# constant dims below — fixed size, so a fixed float32 RegistrySpec fits.
_W1_N = OUTER_POLICY_INPUT_DIM * _H1
_W2_N = _H1 * _H2
_W3_N = _H2 * NUM_OUTER_ACTIONS
OUTER_POLICY_FLAT_DIM = (_W1_N + _H1 + _W2_N + _H2 + _W3_N + NUM_OUTER_ACTIONS) + 2

OUTER_META_POLICY_STATE_SLOT = "outer_meta_policy_state"
OUTER_META_POLICY_STATE_SCHEMA_VERSION = 1
OUTER_META_POLICY_STATE_SPEC = RegistrySpec(
    name=OUTER_META_POLICY_STATE_SLOT,
    dtype=np.dtype("float32"),
    shape=(OUTER_POLICY_FLAT_DIM,),
    feature_flag="",  # enable-gating is at the worker (publish) + agno (read) level
    schema_version=OUTER_META_POLICY_STATE_SCHEMA_VERSION,
    variable_size=False,
)


def _clip01(v: float) -> float:
    if v != v:  # NaN guard
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


@dataclass
class OuterFeatures:
    """The Phase-1 decision feature bundle, built on the agno path (D2-D5).

    Every field is sourced from what is ALREADY computed locally per turn —
    no new sync RPC (INV-OML-7). Missing signals default to 0 / False (the
    cold-start, which the policy learns to weight)."""

    recall_top_cosine: float = 0.0
    recall_count: int = 0
    skill_utility: float = 0.0
    skill_matched: bool = False
    engram_ground: float = 0.0
    requires_tool: bool = False
    has_code_signal: bool = False
    msl_i_confidence: float = 0.0
    msl_attention_entropy: float = 0.0
    msl_concept_confidence: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                1.0,  # bias
                _clip01(self.recall_top_cosine),
                min(max(int(self.recall_count), 0), 10) / 10.0,
                _clip01(self.skill_utility),
                1.0 if self.skill_matched else 0.0,
                _clip01(self.engram_ground),
                1.0 if self.requires_tool else 0.0,
                1.0 if self.has_code_signal else 0.0,
                _clip01(self.msl_i_confidence),
                _clip01(self.msl_attention_entropy),
                _clip01(self.msl_concept_confidence),
            ],
            dtype=np.float32,
        )


class OuterMetaPolicy:
    """Numpy ReLU MLP (11 → 16 → 8 → 5) trained by REINFORCE-with-baseline.

    Mirrors `MetaPolicy` (logic/meta_reasoning.py): forward / select_action /
    train_step / save / load — plus the EMA reward baseline, SHM flat
    (de)serialization, and a grounded-route prior seed for cold-start.
    """

    def __init__(self, input_dim: int = OUTER_POLICY_INPUT_DIM, lr: float = 0.01):
        self.input_dim = input_dim
        self.lr = lr
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / _H1)
        s3 = math.sqrt(2.0 / _H2)
        self.w1 = np.random.randn(input_dim, _H1).astype(np.float32) * s1
        self.b1 = np.zeros(_H1, dtype=np.float32)
        self.w2 = np.random.randn(_H1, _H2).astype(np.float32) * s2
        self.b2 = np.zeros(_H2, dtype=np.float32)
        self.w3 = np.random.randn(_H2, NUM_OUTER_ACTIONS).astype(np.float32) * s3
        self.b3 = np.zeros(NUM_OUTER_ACTIONS, dtype=np.float32)
        self.total_updates = 0
        self.reward_baseline = 0.0  # EMA over observed rewards (REINFORCE baseline)
        self._cache: dict = {}

    # -- inference ---------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return z3

    def action_probs(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        scores = self.forward(np.asarray(x, dtype=np.float32))
        t = max(0.1, temperature)
        exp_s = np.exp((scores - scores.max()) / t)
        return exp_s / (exp_s.sum() + 1e-8)

    def select_action(self, x, temperature: float = 1.0) -> int:
        """Boltzmann-sampled action (EXPLORE; idle loop only, INV-OML-9)."""
        probs = self.action_probs(x, temperature)
        return int(np.random.choice(NUM_OUTER_ACTIONS, p=probs))

    def exploit_action(self, x) -> int:
        """Greedy argmax (EXPLOIT; the ONLY mode on a live user turn — no
        experiments on the user, INV-OML-9)."""
        return int(np.argmax(self.forward(np.asarray(x, dtype=np.float32))))

    # -- learning (off hot path) -------------------------------------
    def train_step(self, x, action: int, advantage: float) -> float:
        scores = self.forward(np.asarray(x, dtype=np.float32))
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / (exp_s.sum() + 1e-8)
        target = np.zeros(NUM_OUTER_ACTIONS)
        target[action] = 1.0
        d_z3 = (probs - target) * abs(advantage)
        if advantage < 0:
            d_z3 = -d_z3
        d_w3 = self._cache["h2"].reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (self._cache["z2"] > 0)
        d_w2 = self._cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (self._cache["z1"] > 0)
        d_w1 = self._cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        for g in (d_w1, d_w2, d_w3):
            np.clip(g, -5.0, 5.0, out=g)
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_z1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_z2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_z3
        self.total_updates += 1
        return float(-np.log(probs[action] + 1e-8) * abs(advantage))

    def learn(self, x, action: int, reward: float, baseline_alpha: float = 0.05) -> float:
        """One REINFORCE-with-baseline update: advantage uses the CURRENT EMA
        baseline, then the baseline moves toward the observed reward."""
        advantage = float(reward) - self.reward_baseline
        loss = self.train_step(x, action, advantage)
        self.reward_baseline += baseline_alpha * (float(reward) - self.reward_baseline)
        return loss

    def seed_prior(self, action: int, strength: float = 1.0) -> None:
        """Warm-start: bias the output toward the grounded-route action (RFP §5
        cold-start mitigation). Additive on the output bias — leaves the learned
        gradient free to override it once real rewards arrive."""
        if 0 <= action < NUM_OUTER_ACTIONS:
            self.b3[action] += float(strength)

    # -- persistence (JSON, for the worker's durable artifact) -------
    def save(self, path: str) -> None:
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "total_updates": self.total_updates,
            "reward_baseline": self.reward_baseline,
            "input_dim": self.input_dim,
            "num_actions": NUM_OUTER_ACTIONS,
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
                d = json.load(f)
            if d.get("input_dim", self.input_dim) != self.input_dim:
                return False
            if d.get("num_actions", NUM_OUTER_ACTIONS) != NUM_OUTER_ACTIONS:
                return False
            self.w1 = np.array(d["w1"], dtype=np.float32)
            self.b1 = np.array(d["b1"], dtype=np.float32)
            self.w2 = np.array(d["w2"], dtype=np.float32)
            self.b2 = np.array(d["b2"], dtype=np.float32)
            self.w3 = np.array(d["w3"], dtype=np.float32)
            self.b3 = np.array(d["b3"], dtype=np.float32)
            self.total_updates = int(d.get("total_updates", 0))
            self.reward_baseline = float(d.get("reward_baseline", 0.0))
            return True
        except Exception:
            return False

    # -- SHM flat (de)serialization ----------------------------------
    def to_flat(self) -> np.ndarray:
        """Pack all weights + metadata into the fixed OUTER_POLICY_FLAT_DIM
        float32 vector for the SHM publish (D6 reader reconstructs via
        from_flat). Layout is fixed by the module dims."""
        flat = np.concatenate([
            self.w1.ravel(), self.b1.ravel(),
            self.w2.ravel(), self.b2.ravel(),
            self.w3.ravel(), self.b3.ravel(),
            np.array([float(self.total_updates), float(self.reward_baseline)],
                     dtype=np.float32),
        ]).astype(np.float32)
        if flat.shape[0] != OUTER_POLICY_FLAT_DIM:  # invariant guard
            raise ValueError(
                f"OuterMetaPolicy flat dim {flat.shape[0]} != {OUTER_POLICY_FLAT_DIM}")
        return flat

    @classmethod
    def from_flat(cls, flat: np.ndarray) -> "OuterMetaPolicy":
        """Reconstruct a read-only-usable policy from the SHM flat vector."""
        if flat is None or flat.shape[0] != OUTER_POLICY_FLAT_DIM:
            raise ValueError("OuterMetaPolicy.from_flat: bad flat vector")
        p = cls()
        i = 0
        def take(n: int) -> np.ndarray:
            nonlocal i
            seg = flat[i:i + n]
            i += n
            return seg
        p.w1 = take(_W1_N).reshape(OUTER_POLICY_INPUT_DIM, _H1).astype(np.float32)
        p.b1 = take(_H1).astype(np.float32)
        p.w2 = take(_W2_N).reshape(_H1, _H2).astype(np.float32)
        p.b2 = take(_H2).astype(np.float32)
        p.w3 = take(_W3_N).reshape(_H2, NUM_OUTER_ACTIONS).astype(np.float32)
        p.b3 = take(NUM_OUTER_ACTIONS).astype(np.float32)
        p.total_updates = int(round(float(take(1)[0])))
        p.reward_baseline = float(take(1)[0])
        return p


__all__ = (
    "OUTER_ACTIONS", "NUM_OUTER_ACTIONS", "OUTER_FEATURE_NAMES",
    "OUTER_POLICY_INPUT_DIM", "OUTER_POLICY_FLAT_DIM",
    "OUTER_META_POLICY_STATE_SLOT", "OUTER_META_POLICY_STATE_SPEC",
    "OUTER_META_POLICY_STATE_SCHEMA_VERSION",
    "OuterFeatures", "OuterMetaPolicy",
    "action_index_to_mode", "action_index_to_name",
)
