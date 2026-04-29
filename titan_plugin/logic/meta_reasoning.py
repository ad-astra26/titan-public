"""
titan_plugin/logic/meta_reasoning.py — Meta-reasoning: thinking about thinking.

Crown jewel of Titan's cognitive architecture. Runs at epoch rate (one step per
consciousness epoch, ~3-5s). Chains span 5-100 epochs (resource-aware budget).

8 meta-primitives with 28 sub-modes (M1-M10):
  FORMULATE   — define/refine problems from 132D anomalies
  RECALL      — multi-source memory query
  HYPOTHESIZE — generate/refine/compare strategy hypotheses
  DELEGATE    — inject strategy bias into main reasoning
  SYNTHESIZE  — integrate insights, save proven strategies
  EVALUATE    — assess meta-strategy quality
  BREAK       — backtrack mid-chain (M7: rewind, checkpoint, restart)
  SPIRIT_SELF — felt-state nudge via neuromod adjustment (M8: maturity-gated)

Events (not primitives):
  EUREKA      — DA burst + wisdom crystallization on high-conf SYNTHESIZE (M9)

Orchestration:
  PARALLEL    — multi-chain scheduler with resource-aware budget (M10)

System 1 learning: per-step NN policy gradient (fast, ~0.1ms).
Used by: spirit_worker epoch loop (_t3_should_fire block).
Depends on: M1 chain_archive, M2 meta_wisdom, M3 meta_autoencoder.
"""

import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from titan_plugin.utils.silent_swallow import swallow_warn
from titan_plugin import bus

logger = logging.getLogger("titan.meta_reasoning")

# ── Constants ─────────────────────────────────────────────────────

META_PRIMITIVES = [
    "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
    "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF",
    "INTROSPECT",
]
NUM_META_ACTIONS = 9

SUB_MODES = {
    # F-phase (rFP §6 / Upgrade C): compositional sub-modes added to
    # FORMULATE, RECALL, HYPOTHESIZE. Other primitives keep current lists —
    # expand later from usage data. New entries route via the Recruitment
    # Layer when resolvers land (Session 2); Session 1 falls back to the
    # nearest existing sub-mode's behavior + compositional metadata tag.
    "FORMULATE":   ["define", "refine", "load_wisdom",
                    "compose_intersection", "compose_union",
                    "compose_difference", "narrow_to_subset",
                    "generalize_from_instance"],
    "RECALL":      ["chain_archive", "experience", "entity", "wisdom",
                    "episodic_specific", "semantic_neighbors",
                    "procedural_matching", "autobiographical_relevant",
                    "topic"],
    "HYPOTHESIZE": ["generate", "refine", "compare",
                    "analogize_from", "contrast_with",
                    "propose_by_inversion", "extend_pattern"],
    "DELEGATE":    ["full_chain", "quick_chain", "biased_chain",
                    "gap_fill"],
    "SYNTHESIZE":  ["combine", "abstract", "rank", "distill_save"],
    "EVALUATE":    ["check_progress", "check_strategy", "check_resources",
                    "peer_cgn"],
    "BREAK":       ["rewind_last", "rewind_to_checkpoint", "restart_fresh"],
    "SPIRIT_SELF": ["boost_curiosity", "boost_focus", "boost_calm",
                    "boost_energy", "release_tension"],
    "INTROSPECT":  ["state_audit", "prediction", "coherence_check",
                    "vocabulary_probe", "architecture_query",
                    "maker_alignment"],
}

# M8: SPIRIT_SELF neuromod nudge map
SPIRIT_SELF_NUDGE_MAP = {
    "boost_curiosity":  {"NE": 0.10, "DA": 0.08},
    "boost_focus":      {"ACh": 0.12, "NE": -0.05},
    "boost_calm":       {"5HT": 0.10},        # GABA excluded by safety
    "boost_energy":     {"DA": 0.10, "Endorphin": 0.08},
    "release_tension":  {"NE": 0.05},          # GABA excluded by safety
}

# Per-step intermediate rewards
STEP_REWARDS = {
    "FORMULATE.define": 0.05,
    "FORMULATE.refine": 0.03,
    "FORMULATE.load_wisdom": 0.10,
    "RECALL.chain_archive": 0.02,
    "RECALL.experience": 0.02,
    "RECALL.entity": 0.01,
    "RECALL.wisdom": 0.02,
    "HYPOTHESIZE.generate": 0.05,
    "HYPOTHESIZE.refine": 0.03,
    "HYPOTHESIZE.compare": 0.03,
    "DELEGATE.full_chain": 0.0,
    "DELEGATE.quick_chain": 0.0,
    "DELEGATE.biased_chain": 0.0,
    "SYNTHESIZE.combine": 0.03,
    "SYNTHESIZE.abstract": 0.03,
    "SYNTHESIZE.rank": 0.02,
    "SYNTHESIZE.distill_save": 0.10,
    "EVALUATE.check_progress": 0.02,
    "EVALUATE.check_strategy": 0.02,
    "EVALUATE.check_resources": 0.01,
    "BREAK.rewind_last": -0.08,
    "BREAK.rewind_to_checkpoint": -0.10,
    "BREAK.restart_fresh": -0.15,
    "SPIRIT_SELF.boost_curiosity": 0.0,
    "SPIRIT_SELF.boost_focus": 0.0,
    "SPIRIT_SELF.boost_calm": 0.0,
    "SPIRIT_SELF.boost_energy": 0.0,
    "SPIRIT_SELF.release_tension": 0.0,
    "INTROSPECT.state_audit": 0.04,
    "INTROSPECT.prediction": 0.06,
    "INTROSPECT.coherence_check": 0.05,
    "INTROSPECT.vocabulary_probe": 0.03,
    "INTROSPECT.architecture_query": 0.04,
    "INTROSPECT.maker_alignment": 0.04,
    # F-phase (rFP §6 / Upgrade C): compositional sub-modes. Values set to
    # each primitive's existing average so these aren't preferred or
    # penalized vs. current modes until emergent rewards ramp in (rFP §7).
    "FORMULATE.compose_intersection": 0.06,
    "FORMULATE.compose_union": 0.06,
    "FORMULATE.compose_difference": 0.06,
    "FORMULATE.narrow_to_subset": 0.06,
    "FORMULATE.generalize_from_instance": 0.06,
    "RECALL.episodic_specific": 0.02,
    "RECALL.semantic_neighbors": 0.02,
    "RECALL.procedural_matching": 0.02,
    "RECALL.autobiographical_relevant": 0.02,
    "HYPOTHESIZE.analogize_from": 0.04,
    "HYPOTHESIZE.contrast_with": 0.04,
    "HYPOTHESIZE.propose_by_inversion": 0.04,
    "HYPOTHESIZE.extend_pattern": 0.04,
}

META_POLICY_INPUT_DIM = 80
SUB_MODE_INPUT_DIM = 30


# ── Meta-Chain State ──────────────────────────────────────────────

@dataclass
class MetaChainState:
    """State of an active meta-reasoning chain. Spans multiple epochs."""
    is_active: bool = False
    chain: list = field(default_factory=list)
    chain_results: list = field(default_factory=list)
    confidence: float = 0.5
    formulate_output: dict = field(default_factory=dict)
    recalled_data: dict = field(default_factory=dict)
    hypotheses: list = field(default_factory=list)
    delegate_results: list = field(default_factory=list)
    synthesized: dict = field(default_factory=dict)
    start_epoch: int = 0
    start_time: float = 0.0
    break_count: int = 0
    max_steps: int = 20
    trigger_reason: str = ""
    awaiting_delegate: bool = False
    delegate_start_chains: int = 0
    pre_state_132d: list = field(default_factory=list)
    # M7: BREAK checkpoints
    checkpoints: list = field(default_factory=list)
    max_breaks: int = 3
    # M8: SPIRIT_SELF state
    spirit_self_cooldown: int = 0
    last_spirit_self_step: int = -1
    pre_nudge_confidence: float = 0.0
    # INTROSPECT state (max 1 per chain)
    introspect_used: bool = False
    # ── TUNING-012 v2: Sub-phase A — Compound reward context ──
    # Each primitive's compound reward needs context. These fields are
    # populated during chain execution and consumed by reward helpers in
    # meta_reasoning_rewards.py. See rFP §7.A step 1.
    recall_history: list = field(default_factory=list)        # source names per RECALL
    pre_eval_confidence: float = 0.0                          # captured before each EVALUATE
    pre_break_avg_reward: float = 0.0                         # captured before each BREAK
    eureka_after_break: bool = False                          # set True if EUREKA fires after BREAK
    # P7: EUREKA tracking for META-CGN accelerator (5× trigger, 3× support)
    eureka_fired: bool = False                                # any EUREKA during this chain
    eureka_trigger: str = ""                                  # primitive that fired the EUREKA
    chain_succeeded: float = 0.0                              # set at conclusion (0.0-1.0)
    # Phase D.1: chain_id for external reward correlation.
    # Assigned monotonically at _start_chain, round-trips through
    # META_LANGUAGE_RESULT → language_worker → META_LANGUAGE_REWARD →
    # meta_engine.add_external_reward → chain_iql.apply_external_reward.
    chain_id: int = -1
    # Phase D.2: SOAR impasse tracking — consecutive declining step rewards
    step_rewards: list = field(default_factory=list)
    impasse_detected: bool = False
    impasse_topic: str = ""
    knowledge_injected: bool = False
    # ── rFP_titan_meta_outer_layer — Bridges 1/2/3 ──────────────────────
    # entity_refs maps symbolic names ("primary_person", "current_topic",
    # "current_event") to concrete IDs. Set by FORMULATE when the intent
    # references a known entity. Read by RECALL.entity / .topic,
    # HYPOTHESIZE / EVALUATE / SYNTHESIZE — making chains carry specific
    # entity references end-to-end (not just abstract context).
    entity_refs: dict = field(default_factory=dict)
    # needs_outer is the hint FORMULATE emits to trigger async composed
    # recall in spirit_worker. When non-empty, spirit_worker dispatches
    # OuterContextReader.compose_recall_query() and stashes the future
    # on the chain state for later primitives to read.
    needs_outer: dict = field(default_factory=dict)
    # outer_context is populated by spirit_worker once the composed
    # recall future resolves (at first primitive that needs it, or at
    # budget deadline). See rFP §5 for shape.
    outer_context: dict = field(default_factory=dict)
    # outer_context_used flips True on first primitive that reads a
    # non-empty outer_context. Drives META_OUTER_REWARD at conclude.
    outer_context_used: bool = False


# ── Policy Networks ───────────────────────────────────────────────

class MetaPolicy:
    """Selects which meta-primitive to run next. 80D → 40 → 20 → 6."""

    def __init__(self, input_dim=META_POLICY_INPUT_DIM, h1=40, h2=20, lr=0.001):
        self.input_dim = input_dim
        self.lr = lr
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / h1)
        s3 = math.sqrt(2.0 / h2)
        self.w1 = np.random.randn(input_dim, h1).astype(np.float32) * s1
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.w2 = np.random.randn(h1, h2).astype(np.float32) * s2
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.w3 = np.random.randn(h2, NUM_META_ACTIONS).astype(np.float32) * s3
        self.b3 = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
        self.total_updates = 0
        self._cache = {}

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        z3 = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return z3

    def select_action(self, x, temperature=1.0):
        scores = self.forward(np.array(x, dtype=np.float32))
        t = max(0.1, temperature)
        exp_s = np.exp((scores - scores.max()) / t)
        probs = exp_s / (exp_s.sum() + 1e-8)
        return int(np.random.choice(NUM_META_ACTIONS, p=probs))

    def train_step(self, x, action, advantage):
        scores = self.forward(np.array(x, dtype=np.float32))
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / (exp_s.sum() + 1e-8)
        target = np.zeros(NUM_META_ACTIONS)
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
        for g in [d_w1, d_w2, d_w3]:
            np.clip(g, -5.0, 5.0, out=g)
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_z1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_z2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_z3
        self.total_updates += 1
        return float(-np.log(probs[action] + 1e-8) * abs(advantage))

    def save(self, path):
        data = {"w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist(),
                "w3": self.w3.tolist(), "b3": self.b3.tolist(),
                "total_updates": self.total_updates, "input_dim": self.input_dim}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("input_dim", self.input_dim) != self.input_dim:
                return False
            self.w1 = np.array(d["w1"], dtype=np.float32)
            self.b1 = np.array(d["b1"], dtype=np.float32)
            self.w2 = np.array(d["w2"], dtype=np.float32)
            self.b2 = np.array(d["b2"], dtype=np.float32)
            w3 = np.array(d["w3"], dtype=np.float32)
            b3 = np.array(d["b3"], dtype=np.float32)
            # M7-M8 migration: grow output dimension if saved < current
            if w3.shape[1] < NUM_META_ACTIONS:
                pad = NUM_META_ACTIONS - w3.shape[1]
                w3 = np.hstack([w3, np.random.randn(w3.shape[0], pad).astype(np.float32) * 0.01])
                b3 = np.concatenate([b3, np.zeros(pad, dtype=np.float32)])
                logger.info("[META] Policy migrated: %d → %d actions (learned weights preserved)",
                            w3.shape[1] - pad, NUM_META_ACTIONS)
            elif w3.shape[1] > NUM_META_ACTIONS:
                w3 = w3[:, :NUM_META_ACTIONS]
                b3 = b3[:NUM_META_ACTIONS]
            self.w3 = w3
            self.b3 = b3
            self.total_updates = d.get("total_updates", 0)
            return True
        except Exception:
            return False


class SubModePolicy:
    """Selects sub-mode within a primitive. 30D → 12 → 6 → N."""

    def __init__(self, n_modes, input_dim=SUB_MODE_INPUT_DIM, h1=12, h2=6, lr=0.001):
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.lr = lr
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / h1)
        s3 = math.sqrt(2.0 / h2)
        self.w1 = np.random.randn(input_dim, h1).astype(np.float32) * s1
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.w2 = np.random.randn(h1, h2).astype(np.float32) * s2
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.w3 = np.random.randn(h2, n_modes).astype(np.float32) * s3
        self.b3 = np.zeros(n_modes, dtype=np.float32)
        self.total_updates = 0
        self._cache = {}

    def select_action(self, x, temperature=1.0):
        x = np.array(x[:self.input_dim], dtype=np.float32)
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        scores = h2 @ self.w3 + self.b3
        self._cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        t = max(0.1, temperature)
        exp_s = np.exp((scores - scores.max()) / t)
        probs = exp_s / (exp_s.sum() + 1e-8)
        return int(np.random.choice(self.n_modes, p=probs))

    def train_step(self, x, action, advantage):
        x = np.array(x[:self.input_dim], dtype=np.float32)
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        scores = h2 @ self.w3 + self.b3
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / (exp_s.sum() + 1e-8)
        target = np.zeros(self.n_modes)
        target[action] = 1.0
        d_z3 = (probs - target) * abs(advantage)
        if advantage < 0:
            d_z3 = -d_z3
        d_w3 = h2.reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_h2 = d_z3 @ self.w3.T
        d_z2 = d_h2 * (z2 > 0)
        d_w2 = h1.reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (z1 > 0)
        d_w1 = x.reshape(-1, 1) @ d_z1.reshape(1, -1)
        for g in [d_w1, d_w2, d_w3]:
            np.clip(g, -5.0, 5.0, out=g)
        self.w1 -= self.lr * d_w1; self.b1 -= self.lr * d_z1
        self.w2 -= self.lr * d_w2; self.b2 -= self.lr * d_z2
        self.w3 -= self.lr * d_w3; self.b3 -= self.lr * d_z3
        self.total_updates += 1
        return float(-np.log(probs[action] + 1e-8) * abs(advantage))

    def to_dict(self):
        return {"w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist(),
                "w3": self.w3.tolist(), "b3": self.b3.tolist(),
                "n_modes": self.n_modes, "total_updates": self.total_updates}

    def from_dict(self, d):
        self.w1 = np.array(d["w1"], dtype=np.float32)
        self.b1 = np.array(d["b1"], dtype=np.float32)
        self.w2 = np.array(d["w2"], dtype=np.float32)
        self.b2 = np.array(d["b2"], dtype=np.float32)
        w3 = np.array(d["w3"], dtype=np.float32)
        b3 = np.array(d["b3"], dtype=np.float32)
        # Shape migration: SUB_MODES dict can grow when a new sub-mode is added
        # (e.g. INTROSPECT gained `maker_alignment` after weights were saved).
        # Without migration, w3/b3 shape mismatches self.n_modes and crashes
        # np.random.choice with "'a' and 'p' must have same size" on every
        # invocation. Mirrors MetaPolicy.load migration logic.
        if w3.shape[1] < self.n_modes:
            pad = self.n_modes - w3.shape[1]
            w3 = np.hstack([w3, np.random.randn(w3.shape[0], pad).astype(np.float32) * 0.01])
            b3 = np.concatenate([b3, np.zeros(pad, dtype=np.float32)])
            logger.info("[META] SubModePolicy migrated: %d → %d modes (learned weights preserved)",
                        w3.shape[1] - pad, self.n_modes)
        elif w3.shape[1] > self.n_modes:
            w3 = w3[:, :self.n_modes]
            b3 = b3[:self.n_modes]
            logger.info("[META] SubModePolicy truncated: %d → %d modes",
                        w3.shape[1] + (w3.shape[1] - self.n_modes), self.n_modes)
        self.w3 = w3
        self.b3 = b3
        # Hardening: post-load shape invariant — if this fails, migration above is wrong.
        assert self.w3.shape[1] == self.n_modes, (
            f"SubModePolicy shape invariant violated: w3 cols={self.w3.shape[1]} != n_modes={self.n_modes}"
        )
        assert self.b3.shape[0] == self.n_modes, (
            f"SubModePolicy shape invariant violated: b3 len={self.b3.shape[0]} != n_modes={self.n_modes}"
        )
        self.total_updates = d.get("total_updates", 0)


# ── Transition Buffer ─────────────────────────────────────────────

class MetaTransitionBuffer:
    """Stores meta-reasoning transitions for System 1 learning."""

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._states, self._actions, self._rewards = [], [], []
        self._sub_actions, self._dones = [], []

    def record(self, state, action, sub_action, reward, done=False):
        self._states.append(list(state))
        self._actions.append(action)
        self._sub_actions.append(sub_action)
        self._rewards.append(reward)
        self._dones.append(done)
        if len(self._states) > self.max_size:
            self._states.pop(0); self._actions.pop(0)
            self._sub_actions.pop(0); self._rewards.pop(0)
            self._dones.pop(0)

    def update_last_reward(self, reward):
        if self._rewards:
            self._rewards[-1] = reward

    def sample(self, batch_size=16):
        n = len(self._states)
        if n < batch_size:
            return None
        idxs = random.sample(range(n), batch_size)
        return (
            [self._states[i] for i in idxs],
            [self._actions[i] for i in idxs],
            [self._sub_actions[i] for i in idxs],
            [self._rewards[i] for i in idxs],
            [self._dones[i] for i in idxs],
        )

    def size(self):
        return len(self._states)

    def save(self, path):
        data = {"states": self._states[-500:], "actions": self._actions[-500:],
                "sub_actions": self._sub_actions[-500:],
                "rewards": self._rewards[-500:], "dones": self._dones[-500:]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                d = json.load(f)
            self._states = d.get("states", [])
            self._actions = d.get("actions", [])
            self._sub_actions = d.get("sub_actions", [])
            self._rewards = d.get("rewards", [])
            self._dones = d.get("dones", [])
        except Exception as _swallow_exc:
            swallow_warn('[logic.meta_reasoning] MetaTransitionBuffer.load: with open(path) as f: d = json.load(f)', _swallow_exc,
                         key='logic.meta_reasoning.MetaTransitionBuffer.load.line496', throttle=100)


# ── Trigger Evaluation ────────────────────────────────────────────

def should_trigger_meta(reasoning_engine, neuromods: dict,
                        chain_archive, config: dict) -> tuple:
    """Check if meta-reasoning should start. Returns (should_trigger, reason)."""
    if not reasoning_engine:
        return False, ""

    total_chains = reasoning_engine._total_chains
    total_commits = reasoning_engine._total_conclusions

    # 1. Low commit rate
    threshold = config.get("trigger_commit_rate_threshold", 0.30)
    if total_chains >= 10:
        rate = total_commits / total_chains
        if rate < threshold:
            return True, f"low_commit_rate({rate:.2f})"

    # 2. High REFLECTION
    refl_threshold = config.get("trigger_reflection_threshold", 0.75)
    refl = 0
    if isinstance(neuromods, dict):
        for key in ("REFLECTION", "reflection"):
            if key in neuromods:
                v = neuromods[key]
                refl = v.get("level", v) if isinstance(v, dict) else float(v)
                break
    if refl > refl_threshold:
        return True, f"high_reflection({refl:.2f})"

    # 3. Periodic
    interval = config.get("trigger_periodic_interval", 50)
    if total_chains > 0 and total_chains % interval == 0:
        return True, f"periodic({total_chains})"

    # 4. Experience pressure
    pressure = config.get("trigger_experience_pressure", 20)
    if chain_archive:
        stats = chain_archive.get_stats()
        if stats.get("unconsolidated", 0) >= pressure:
            return True, f"experience_pressure({stats['unconsolidated']})"

    # 5. Social event trigger — significant social perception creates
    # reasoning about the interaction (e.g., challenging conversation,
    # high-arousal engagement, identity-probing exchange)
    if hasattr(reasoning_engine, '_social_trigger') and reasoning_engine._social_trigger:
        reason = reasoning_engine._social_trigger
        reasoning_engine._social_trigger = None  # consume once
        return True, f"social_event({reason})"

    return False, ""


# ── Resource Detection (M10) ─────────────────────────────────────

def _detect_resource_budget():
    """Auto-detect chain budget and max parallel chains from hardware."""
    cpu = os.cpu_count() or 4
    try:
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except (ValueError, OSError, AttributeError):
        ram_gb = 8
    budget = 20
    if ram_gb >= 16: budget += 10
    if ram_gb >= 32: budget += 20
    if ram_gb >= 64: budget += 20
    if cpu >= 8:     budget += 5
    if cpu >= 16:    budget += 5
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram >= 8:
                budget = int(budget * 1.5)
    except Exception as _swallow_exc:
        swallow_warn('[logic.meta_reasoning] _detect_resource_budget: import torch', _swallow_exc,
                     key='logic.meta_reasoning._detect_resource_budget.line574', throttle=100)
    budget = min(budget, 100)
    max_parallel = 1
    if ram_gb >= 8 and cpu >= 4:   max_parallel = 2
    if ram_gb >= 16 and cpu >= 8:  max_parallel = 3
    return budget, max_parallel, ram_gb, cpu


# ── Multi-Chain Scheduler (M10) ─────────────────────────────────

class MultiChainScheduler:
    """Manages 1-3 concurrent meta-chains sharing a step budget."""

    def __init__(self, max_chains=3, total_budget=20, config=None):
        self.max_chains = max_chains
        self.total_budget = total_budget
        self.chains: list = []
        self.active_index: int = 0
        self.total_steps_used: int = 0
        self.schedule_mode = (config or {}).get("parallel_schedule_mode", "round_robin")

    def spawn_chain(self, reason, state_132d, max_steps) -> MetaChainState:
        chain = MetaChainState()
        chain.is_active = True
        chain.trigger_reason = reason
        chain.start_time = time.time()
        chain.pre_state_132d = list(state_132d[:132])
        chain.max_steps = max_steps
        self.chains.append(chain)
        return chain

    def get_active_chain(self):
        if not self.chains:
            return None
        self.active_index = min(self.active_index, len(self.chains) - 1)
        return self.chains[self.active_index]

    def advance(self):
        if len(self.chains) <= 1:
            return
        if self.schedule_mode == "round_robin":
            self.active_index = (self.active_index + 1) % len(self.chains)
        elif self.schedule_mode == "priority":
            best_idx = max(range(len(self.chains)),
                           key=lambda i: self.chains[i].confidence)
            self.active_index = best_idx

    def should_spawn(self, nm) -> bool:
        if len(self.chains) >= self.max_chains:
            return False
        if self.budget_remaining() < 5:
            return False
        # Personality-emergent: high NE+DA = parallel, high 5HT+GABA = serial
        ne = nm.get("NE", 0.5)
        da = nm.get("DA", 0.5)
        sht = nm.get("5HT", 0.5)
        gaba = nm.get("GABA", 0.5)
        tendency = (ne * 0.4 + da * 0.3) - (sht * 0.3 + gaba * 0.2)
        return tendency > 0.15

    def should_merge(self):
        for i in range(len(self.chains)):
            for j in range(i + 1, len(self.chains)):
                d_i = self.chains[i].formulate_output.get("domain", "")
                d_j = self.chains[j].formulate_output.get("domain", "")
                if d_i and d_j and d_i == d_j:
                    return (i, j)
        return None

    def merge_chains(self, idx_a, idx_b):
        a, b = self.chains[idx_a], self.chains[idx_b]
        a.hypotheses.extend(b.hypotheses)
        a.delegate_results.extend(b.delegate_results)
        if b.synthesized:
            a.synthesized.update(b.synthesized)
        a.confidence = max(a.confidence, b.confidence)
        logger.info("[META] Merged chain %d into %d (domain=%s)",
                    idx_b, idx_a, a.formulate_output.get("domain", "?"))
        self.chains.pop(idx_b)
        if self.active_index >= len(self.chains):
            self.active_index = 0

    def budget_remaining(self) -> int:
        return max(0, self.total_budget - self.total_steps_used)

    def all_done(self) -> bool:
        return not self.chains or self.budget_remaining() <= 0

    def remove_chain(self, idx):
        if 0 <= idx < len(self.chains):
            self.chains.pop(idx)
            if self.active_index >= len(self.chains):
                self.active_index = 0

    def reset(self):
        self.chains.clear()
        self.active_index = 0
        self.total_steps_used = 0


# ── Meta-Reasoning Engine ─────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: MetaReasoningEngine._chain_iql + _meta_cgn are
# sub-component object references that own their own persistence paths
# (chain_iql state + primitive_grounding.json). _subsystem_cache_pending is
# a transient async-fetch flag. None are state the MetaReasoningEngine
# should restore itself.
class MetaReasoningEngine:
    """Meta-reasoning: thinking about thinking.

    Runs at epoch rate. Resource-aware chain budget (20-100 steps).
    System 1 learning per step (NN policy gradient).
    """

    def __init__(self, config: dict = None, send_queue=None):
        cfg = config or {}
        # send_queue is the DivineBus send handle (set by spirit_worker). Used
        # by META-CGN to register as the 7th CGN consumer and emit transitions.
        # None in standalone/test contexts — META-CGN falls back to local-only
        # grounding accumulation with no CGN worker interaction.
        self._send_queue = send_queue
        # ── TUNING-012 v2: DNA (per-primitive compound reward coefficients) ──
        # Sourced from [meta_reasoning_dna] in titan_params.toml, merged with
        # per-Titan overrides ([meta_reasoning_dna.T1/T2/T3]) by spirit_worker.
        # Empty dict means compound rewards are disabled (legacy flat reward path).
        self._dna: dict = cfg.get("dna", {}) or {}
        self._titan_id: str = cfg.get("titan_id", "T1")
        # TUNING-012 v2 Phase D pre-flight: blend ratio between legacy and
        # compound reward paths is now DNA-tunable (was hardcoded 0.5).
        # Per-Titan overrides in [meta_reasoning_dna.T1/T2/T3]:
        #   T1=0.4 (cautious), T2=0.6 (aggressive escape), T3=0.5 (balanced).
        self._compound_blend_alpha: float = float(
            self._dna.get("compound_legacy_blend_alpha", 0.5))
        # Phase D.1 — external reward blend (META_LANGUAGE closed loop).
        # Weight applied to language_worker-measured grounding delta when
        # blending into chain_iql buffer entries via add_external_reward.
        self._external_reward_blend_alpha: float = float(
            self._dna.get("external_reward_blend_alpha", 0.5))
        # Phase D.1 — monotonic chain_id counter. Assigned at _start_chain,
        # round-trips through bus messages for external reward correlation.
        self._next_chain_id: int = 1
        # TUNING-012 v2 Sub-phase C: cognitive contracts DNA (R7 — per-Titan
        # contract thresholds, R5 — per-primitive eureka thresholds, R3 —
        # diversity pressure magnitudes). Loaded by spirit_worker DNA loader,
        # passed in cfg["contracts_dna"]. Empty dict means contracts/handlers
        # use code defaults (graceful degradation).
        self._contracts_dna: dict = cfg.get("contracts_dna", {}) or {}
        # Subsystem cache (refreshed at chain start, used per primitive).
        # A-finish (2026-04-09): TTL is now DNA-tunable, not hardcoded 30s.
        # Refresh flow: spirit_worker checks is_subsystem_cache_stale() each
        # epoch tick, sends TIMECHAIN_QUERY + CONTRACT_LIST async, response
        # handlers (TIMECHAIN_QUERY_RESP / CONTRACT_LIST_RESP) call
        # update_subsystem_cache() to populate the 10 wireable signals.
        # 2 INTROSPECT signals (self_prediction_accuracy, self_profile_divergence)
        # remain stubbed until Sub-phase E.
        self._subsystem_cache: dict = {}
        self._subsystem_cache_ts: float = 0.0
        self._subsystem_cache_ttl: float = float(
            self._dna.get("subsystem_cache_ttl_seconds", 30.0))
        # "Pending" flag — set when queries fire, cleared when responses
        # arrive. Prevents re-firing identical queries while in-flight.
        self._subsystem_cache_pending: bool = False
        self._subsystem_cache_pending_ts: float = 0.0
        # TUNING-012 v2 Sub-phase C (R3): directed diversity pressure on a
        # dominant primitive, applied via apply_diversity_pressure() from the
        # META_DIVERSITY_PRESSURE handler in spirit_worker. Decays per chain.
        self._primitive_bias = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
        self._diversity_pressure_remaining = 0
        self._diversity_pressure_target = ""
        self._diversity_pressure_initial_magnitude = 0.0
        self._diversity_pressure_initial_decay = 0
        self._diversity_pressure_total_applied = 0

        # ── Task 4 P1: in-engine monoculture control loop config ──
        # Architectural principle (per discussion 2026-04-12): contracts
        # declare invariants and fire on genesis seals (~4-5x/day); control
        # loops live with the controlled subsystem and run at the cadence of
        # the controlled signal (per chain). The contract framework's
        # genesis-seal cadence is for state checkpointing, not high-frequency
        # control. monoculture_detector contract still fires as state-invariant
        # safety net; this in-engine check does the actual responsive control.
        # Permanent (NOT transitional) — separation of concerns is the
        # right architecture, independent of META-CGN.
        self._inengine_mono_threshold: float = float(
            self._dna.get("inengine_mono_dominance_threshold", 0.80))
        self._inengine_mono_min_actions: int = int(
            self._dna.get("inengine_mono_min_actions", 100))
        self._inengine_mono_pressure_magnitude: float = float(
            self._dna.get("inengine_mono_pressure_magnitude", 0.75))
        self._inengine_mono_pressure_decay: int = int(
            self._dna.get("inengine_mono_pressure_decay", 100))
        self._inengine_mono_last_fire_chain: int = -1
        self._inengine_mono_total_fires: int = 0

        # ── Consecutive-repeat decay (TUNING-017) ──
        # Penalizes selecting the same primitive N times in a row within a chain.
        # Real thinking has natural flow: FORMULATE→EVALUATE→RECALL, not
        # FORMULATE→FORMULATE→FORMULATE. Each consecutive repeat gets a
        # multiplicative logit penalty: -decay_per_repeat * consecutive_count.
        # 1st repeat: small penalty. 3rd+: nearly impossible.
        self._repeat_decay_per_step: float = float(
            self._dna.get("repeat_decay_per_step", 1.5))
        self._repeat_decay_max: float = float(
            self._dna.get("repeat_decay_max", 5.0))

        # ── Audit telemetry (Task 3 observability — feeds /v4/meta-reasoning/audit) ──
        # Last 10 diversity pressure fires with timestamps for cadence analysis.
        self._diversity_pressure_fire_history: deque = deque(maxlen=10)
        # INTROSPECT health: picks vs successful executions vs rerouted-at-gate.
        # Divergence between (executions + rerouted) and picks indicates the
        # SubModePolicy shape-migration bug returned (or a new INTROSPECT bug).
        # Reroute reasons are by-design (max 1 per chain, gate not met, cooldown).
        self._introspect_picks_lifetime: int = 0
        self._introspect_executions_lifetime: int = 0
        self._introspect_rerouted_lifetime: int = 0
        # Monoculture-adjustment penalty tracking: counts fires + cumulative
        # penalty applied. If mono_adj is firing every chain but dominance isn't
        # decreasing, the penalty isn't strong enough.
        self._mono_adj_fires_count: int = 0
        self._mono_adj_cumulative: float = 0.0
        self._mono_adj_history: deque = deque(maxlen=100)
        # Per-primitive reward component breakdowns (last 100 occurrences each).
        # Lets us see whether minority primitives stay at base rewards vs.
        # FORMULATE pulling all the structural-fit components.
        self._recent_primitive_breakdowns: dict = {
            prim: deque(maxlen=100) for prim in META_PRIMITIVES
        }
        # Last 100 chain rewards for trend analysis.
        self._recent_chain_rewards: deque = deque(maxlen=100)

        # ── TUNING-012 v2 Sub-phase D.2: SOAR impasse → knowledge loop ──
        # Tracks impasse events and knowledge requests for gating.
        self._soar_config = {
            "threshold_consec": int(self._dna.get("soar_impasse_threshold_consec", 3)),
            "max_internal_per_hour": int(self._dna.get("soar_max_internal_recalls_per_hour", 20)),
            "max_external_per_hour": int(self._dna.get("soar_max_external_research_per_hour", 8)),
            "concept_cooldown_s": float(self._dna.get("soar_concept_cooldown_seconds", 900)),
            "urgency_plateau": float(self._dna.get("soar_urgency_plateau", 0.4)),
            "urgency_declining": float(self._dna.get("soar_urgency_declining", 0.7)),
            "urgency_stuck": float(self._dna.get("soar_urgency_stuck", 0.9)),
            "curiosity_bonus": int(self._dna.get("soar_curiosity_momentum_bonus", 1)),
            "curiosity_duration_s": float(self._dna.get("soar_curiosity_momentum_duration_s", 1800)),
            "relevance_threshold": float(self._dna.get("soar_knowledge_relevance_threshold", 0.2)),
        }
        self._soar_requests_this_hour: list = []    # timestamps of requests
        self._soar_concept_cooldowns: dict = {}     # concept → last_request_ts
        self._soar_curiosity_until: float = 0.0     # curiosity momentum expiry
        self._soar_last_successful_topic: str = ""  # for curiosity momentum

        # Resource-aware budget
        mcl = cfg.get("max_chain_length", "auto")
        if mcl == "auto" or not isinstance(mcl, (int, float)):
            auto_budget, auto_parallel, ram, cpu = _detect_resource_budget()
            self.max_chain_length = auto_budget
            self._auto_max_parallel = auto_parallel
            logger.info("[META] Resource budget: steps=%d parallel=%d (RAM=%.1fGB, CPU=%d)",
                        auto_budget, auto_parallel, ram, cpu)
        else:
            self.max_chain_length = int(mcl)
            self._auto_max_parallel = 2
        self.min_chain_length = cfg.get("min_chain_length", 3)
        self.save_dir = cfg.get("save_dir", "./data/reasoning")
        self._delegate_bias_strength = cfg.get("delegate_bias_strength", 0.5)
        self._delegate_max_bias = cfg.get("delegate_max_bias", 3.0)
        self._config = cfg

        # Maturity gates
        self._gates = {
            "vocabulary": cfg.get("gate_vocabulary", 20),
            "dream_cycles": cfg.get("gate_dream_cycles", 1),
            "ns_train_steps": cfg.get("gate_ns_train_steps", 10000),
            "reasoning_chains": cfg.get("gate_reasoning_chains", 50),
            "pi_clusters": cfg.get("gate_pi_clusters", 5),
        }

        # Policy networks
        self.meta_policy = MetaPolicy(
            input_dim=cfg.get("policy_input_dim", META_POLICY_INPUT_DIM),
            h1=cfg.get("policy_h1", 40), h2=cfg.get("policy_h2", 20),
            lr=cfg.get("learning_rate", 0.001))
        self.sub_mode_policies = {
            prim: SubModePolicy(n_modes=len(modes),
                                input_dim=cfg.get("sub_mode_input_dim", SUB_MODE_INPUT_DIM),
                                lr=cfg.get("learning_rate", 0.001))
            for prim, modes in SUB_MODES.items()
        }

        # M7: BREAK config
        self._max_breaks = cfg.get("max_breaks_per_chain", 3)
        self._break_base_cost = cfg.get("break_base_cost", 0.08)

        # M8: SPIRIT_SELF config
        self._spirit_self_gate = cfg.get("gate_spirit_self_chains", 50)
        self._spirit_self_cooldown_max = cfg.get("spirit_self_cooldown", 5)

        # M9: EUREKA config — per-primitive thresholds
        # Different cognitive styles produce insights through different primitives:
        # T1 (Hypothesizer) → SYNTHESIZE, T2 (Delegator) → via RECALL→connection,
        # T3 (Formulator) → FORMULATE. EUREKA must reward all paths to insight.
        # TUNING-012 v2 Sub-phase C (R5): rare primitives (RECALL, INTROSPECT,
        # BREAK) get a LOWER bar so EUREKA rewards diversity-of-insight, not
        # just insight-magnitude on whichever primitive currently dominates.
        # Per-primitive thresholds come from cognitive_contracts_dna when
        # present (titan_params.toml [cognitive_contracts_dna]).
        _default_eureka_thresholds = {
            "SYNTHESIZE": 0.70,     # Combining ideas — original EUREKA gate
            "FORMULATE": 0.72,      # Articulation insight (T3's strength)
            "HYPOTHESIZE": 0.75,    # Speculative — higher bar
            "EVALUATE": 0.78,       # Critique must be decisive — highest bar
            "SPIRIT_SELF": 0.65,    # Self-insight is rare and valuable — lower bar
        }
        # R5: pull from contracts_dna if present (per-primitive thresholds
        # spelled like eureka_threshold_recall, eureka_threshold_formulate, etc.)
        _cc_dna = self._contracts_dna
        _r5_thresholds = {}
        for _ek_prim in ("SYNTHESIZE", "FORMULATE", "HYPOTHESIZE", "EVALUATE",
                         "SPIRIT_SELF", "RECALL", "INTROSPECT", "BREAK", "DELEGATE"):
            _ek_key = f"eureka_threshold_{_ek_prim.lower()}"
            if _ek_key in _cc_dna:
                _r5_thresholds[_ek_prim] = float(_cc_dna[_ek_key])
        if _r5_thresholds:
            # Merge: R5 thresholds override defaults but defaults fill any gaps
            _merged = dict(_default_eureka_thresholds)
            _merged.update(_r5_thresholds)
            self._eureka_thresholds = cfg.get("eureka_thresholds", _merged)
        else:
            self._eureka_thresholds = cfg.get("eureka_thresholds", _default_eureka_thresholds)
        # Legacy single threshold as fallback for primitives not in map
        self._eureka_threshold = cfg.get("eureka_threshold", 0.70)
        self._eureka_cooldown_max = cfg.get("eureka_cooldown", 10)
        self._eureka_cooldown_steps = 0
        self._eureka_da_base = cfg.get("eureka_da_burst_base", 0.05)
        self._eureka_da_novelty = cfg.get("eureka_da_burst_novelty_scale", 0.10)

        # M10: PARALLEL config
        self._parallel_enabled = cfg.get("parallel_enabled", False)
        _mpc = cfg.get("max_parallel_chains", "auto")
        self._max_parallel = self._auto_max_parallel if _mpc == "auto" else int(_mpc)
        self.scheduler = MultiChainScheduler(
            max_chains=self._max_parallel,
            total_budget=self.max_chain_length,
            config=cfg,
        )

        # Buffer + state
        self.buffer = MetaTransitionBuffer(max_size=cfg.get("buffer_size", 1000))
        self.state = MetaChainState()

        # ── TUNING-012 v2 Sub-phase B: Chain-level IQL hierarchy ──
        # The chain Q-net learns "given THIS task, which CHAIN TEMPLATE
        # works best?". Records outcomes per chain, trains during dream
        # consolidation, biases primitive selection at chain start.
        # Lazy-imported to keep meta_reasoning module-load fast.
        self._chain_iql = None
        if self._dna.get("chain_iql_enabled", False):
            try:
                from titan_plugin.logic.chain_iql import ChainIQL
                self._chain_iql = ChainIQL(dna=self._dna, save_dir=self.save_dir)
                logger.info(
                    "[META] Chain IQL enabled: templates=%d, buffer=%d, blend_α=%.2f",
                    len(self._chain_iql.template_registry),
                    len(self._chain_iql.buffer),
                    self._chain_iql.blend_alpha,
                )
            except Exception as _ciql_err:
                logger.warning("[META] Chain IQL init failed: %s", _ciql_err)
                self._chain_iql = None
        # ── META-CGN Phase 1 (rFP_meta_cgn_v2.md) ──
        # Registers primitives as 7th CGN consumer via /dev/shm spine.
        # Shadow mode only — observes but doesn't influence selection yet.
        # Failsafe: if init fails, meta-reasoning continues via chain_iql alone.
        self._meta_cgn = None
        if self._dna.get("meta_cgn_enabled", True):
            try:
                from titan_plugin.logic.meta_cgn import MetaCGNConsumer
                self._meta_cgn = MetaCGNConsumer(
                    send_queue=self._send_queue,
                    titan_id=self._titan_id,
                    save_dir=cfg.get("meta_cgn_save_dir", "data/meta_cgn"),
                    module_name="spirit",
                    # Microkernel v2 §A.2 part 2 (S4): pass cgn_config so
                    # CGNConsumerClient.ShmWeightReader picks correct mode.
                    cgn_config=cfg,
                )
                _stats = self._meta_cgn.get_stats()
                logger.info(
                    "[META] META-CGN enabled (shadow mode): primitives=%d, "
                    "grounded=%d, transitions=%d",
                    _stats["primitives_total"], _stats["primitives_grounded"],
                    _stats["transitions_sent"],
                )
            except Exception as _mcgn_err:
                logger.warning("[META] META-CGN init failed: %s "
                               "(continuing without — chain_iql only)",
                               _mcgn_err)
                self._meta_cgn = None

        # ── EMOT-CGN — standalone worker (Phase 1.6h cutover 2026-04-20) ──
        # EmotCGNConsumer now lives in `titan_plugin/modules/emot_cgn_worker.py`
        # (standalone Guardian-supervised L2 subprocess) per rFP_emot_cgn_v2
        # §10 ADR. Meta-reasoning emits EMOT_CHAIN_EVIDENCE +
        # FELT_CLUSTER_UPDATE bus messages; downstream consumers read
        # state via ShmEmotReader (`/dev/shm/titan/emot_state.bin`) per
        # rFP_microkernel_v2 §State Registries.
        #
        # `self._emot_cgn = None` kept as sentinel for code paths that
        # still check truthiness during cutover window; removed entirely
        # in Phase 1.7+ once all call-sites converted to shm reads.
        self._emot_cgn = None

        # Per-chain task context (set in _start_chain, used in _conclude_chain)
        self._chain_task_emb = None
        self._chain_task_domain = "general"
        # Currently-suggested template (string, e.g. "FORMULATE→RECALL→EVALUATE")
        # Used for primitive selection bias during the chain.
        self._suggested_template = None
        self._suggested_template_q = 0.0

        # ── rFP_titan_meta_outer_layer — Bridges 1-5 ───────────────────
        # OuterContextReader is wired by spirit_worker at its own init time
        # via set_outer_reader(). Engine stays constructible in standalone
        # test contexts (no reader → outer-layer paths return early).
        self._outer_reader = None
        # In-flight composed-recall future per active chain. Written by
        # _start_chain/FORMULATE when needs_outer fires; consumed by the
        # first primitive that reads outer_context (e.g. RECALL.entity).
        self._outer_future = None

        # EMA tracking
        self._baseline_confidence = 0.5
        self._strategy_history = np.zeros(12, dtype=np.float32)
        self._ema_state = np.zeros(132, dtype=np.float32)
        self._ema_alpha = 0.02

        # Stats
        self._total_meta_chains = 0
        self._total_meta_steps = 0
        self._total_wisdom_saved = 0
        self._total_eurekas = 0
        # META-CGN Producer #4 — pending eureka events queue.
        # _fire_eureka appends here (decoupled from bus/send_queue); spirit_worker
        # drains after each meta_engine.tick() and emits via emit_meta_cgn_signal.
        # Keeps this logic module ignorant of the bus while preserving full event
        # context (novelty, trigger_primitive, chain, etc.).
        self._pending_cgn_events: list[dict] = []
        # META-CGN Producer #13 — pending reflection_depth events queue.
        # _prim_introspect appends (sub_mode, confidence) after each result.
        # spirit_worker drains + applies EdgeDetector.observe_new_max per sub_mode
        # so only NEW maxima per introspection sub_mode emit signals.
        self._pending_cgn_reflection_events: list[dict] = []
        # META-CGN Producer #14 — pending coherence_gain events queue.
        # _prim_introspect appends (chi_coh, ts) when sub=="coherence_check".
        # spirit_worker drains + applies EdgeDetector.observe(key, value, threshold)
        # at 4 thresholds [0.3, 0.5, 0.7, 0.9] so each threshold fires once per
        # crossing (up to 4 lifetime emissions per Titan plus re-crossings).
        self._pending_cgn_coherence_events: list[dict] = []
        # META-CGN Producer #15 — pending crystallized wisdom events queue.
        # _conclude_chain appends on high-reward chain completion. spirit_worker
        # drains + applies EdgeDetector.observe_first_time on chain signature so
        # only NEW (domain, chain_sequence) tuples fire signals (repeat-signature
        # high-conf crystallizations also fire per rFP § 12 row 15 secondary gate).
        self._pending_cgn_wisdom_events: list[dict] = []

        # Adaptive epsilon-greedy state — escapes policy collapse when softmax
        # temperature alone is insufficient. Tracks recent chain unique-prim count.
        self._recent_chain_uniques: deque = deque(maxlen=50)
        self._adaptive_epsilon = 0.0
        # Social event trigger — set externally by spirit_worker when
        # significant social perception arrives. Consumed once by should_trigger_meta.
        self._social_trigger: str | None = None

        # Self-reasoning engine — set via set_self_reasoning() from spirit_worker
        self._self_reasoning = None

        os.makedirs(self.save_dir, exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────────

    def tick(self, state_132d, neuromods, reasoning_engine,
             chain_archive, meta_wisdom, ex_mem, meta_autoencoder) -> dict:
        """One meta-reasoning step per epoch."""

        sv = list(state_132d) if state_132d else []
        if len(sv) < 132:
            sv = sv + [0.5] * (132 - len(sv))

        # Update EMA baseline
        sv_arr = np.array(sv[:132], dtype=np.float32)
        self._ema_state = self._ema_alpha * sv_arr + (1 - self._ema_alpha) * self._ema_state

        nm = self._normalize_neuromods(neuromods)
        # Stash for _conclude_chain → _emot_ctx producer wiring (rFP §23.6):
        # EMOT-CGN's bundle needs real 6D neuromod state, not the 0.5-default
        # stubs from _subsystem_cache. `nm` is the canonical per-tick reading.
        self._last_neuromods_dict = nm

        # If no active chain, check trigger
        if not self.state.is_active:
            should, reason = should_trigger_meta(
                reasoning_engine, neuromods, chain_archive, self._config)
            if not should:
                return {"action": "IDLE"}
            self._start_chain(reason, sv)

        # If waiting for DELEGATE result
        if self.state.awaiting_delegate:
            return self._check_delegate(reasoning_engine)

        # Build input + select primitive
        meta_input = self._build_meta_input(sv, nm, chain_archive, meta_autoencoder)
        temperature = self._get_temperature(nm)

        # ── TUNING-017: consecutive-repeat decay ──
        # Count how many times the last primitive repeated at the tail of the
        # current chain. Apply a logit penalty proportional to the repeat count.
        # FORMULATE→FORMULATE→FORMULATE is penalized; FORMULATE→EVALUATE→FORMULATE is not.
        _repeat_bias = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
        if self.state.chain:
            _last = self.state.chain[-1]
            _consec = 0
            for _p in reversed(self.state.chain):
                if _p == _last:
                    _consec += 1
                else:
                    break
            if _consec >= 1:
                # Chain entries are compound "PRIMITIVE.subtype" names
                # (FORMULATE.define, RECALL.episodic_specific, etc.) but
                # META_PRIMITIVES only holds the 9 base primitives — strip
                # the sub-mode before lookup. Without this strip, every
                # compound-chain repeat penalty silently failed (1000+
                # ValueErrors logged once Pattern C migration surfaced
                # them 2026-04-25).
                _last_base = _last.split(".", 1)[0]
                try:
                    _last_idx = META_PRIMITIVES.index(_last_base)
                    _penalty = min(self._repeat_decay_per_step * _consec,
                                   self._repeat_decay_max)
                    _repeat_bias[_last_idx] = -_penalty
                except ValueError as _swallow_exc:
                    swallow_warn('[logic.meta_reasoning] MetaReasoningEngine.tick: META_PRIMITIVES.index(_last_base)', _swallow_exc,
                                 key='logic.meta_reasoning.MetaReasoningEngine.tick.line1107', throttle=100)

        # TUNING-012 v2 Sub-phase C (R3): apply diversity pressure if active.
        # When monoculture_detector contract has fired and the handler called
        # apply_diversity_pressure(), we re-roll the primitive selection with
        # the directed negative bias added to the policy logits. This is the
        # active escape pressure that turns Phase C into a closed control loop.
        _has_bias = (self._diversity_pressure_remaining > 0 and np.any(self._primitive_bias)) or np.any(_repeat_bias)
        if _has_bias:
            try:
                _scores = self.meta_policy.forward(np.array(meta_input, dtype=np.float32))
                _total_bias = _repeat_bias.copy()
                if self._diversity_pressure_remaining > 0 and np.any(self._primitive_bias):
                    _total_bias += self._primitive_bias
                _biased = _scores + _total_bias
                _t = max(0.1, temperature)
                _exp = np.exp((_biased - _biased.max()) / _t)
                _probs = _exp / (_exp.sum() + 1e-8)
                prim_idx = int(np.random.choice(NUM_META_ACTIONS, p=_probs))
            except Exception as _dp_err:
                logger.warning("[META] Diversity/repeat bias failed: %s", _dp_err)
                prim_idx = self.meta_policy.select_action(meta_input, temperature)
        else:
            prim_idx = self.meta_policy.select_action(meta_input, temperature)

        # Adaptive epsilon-greedy: force exploration when policy is collapsed.
        # Self-decays as diversity recovers (no manual disable needed).
        self._adaptive_epsilon = self._compute_adaptive_epsilon()
        _epsilon_picked_introspect = False
        if self._adaptive_epsilon > 0 and np.random.random() < self._adaptive_epsilon:
            prim_idx = int(np.random.randint(0, NUM_META_ACTIONS))
            logger.info("[META] ε-greedy override: ε=%.2f, action=%s",
                        self._adaptive_epsilon, META_PRIMITIVES[prim_idx])
            # Audit telemetry: track INTROSPECT picks for crash diagnosis.
            # If template-bias override below steals the pick, we increment
            # rerouted_lifetime to preserve invariant picks == exec + rerouted.
            if META_PRIMITIVES[prim_idx] == "INTROSPECT":
                self._introspect_picks_lifetime += 1
                _epsilon_picked_introspect = True

        prim_name = META_PRIMITIVES[prim_idx]

        # ── TUNING-012 v2 Sub-phase B: chain template soft bias ──
        # If we have a high-Q suggested template AND we're at a step where
        # the template specifies a different primitive, override with
        # probability proportional to (Q - 0.5) * blend_alpha. This lets
        # the chain Q-net steer behavior without fully overriding the policy.
        # 2026-04-19: explicit `is not None` + `> 0` guards instead of
        # truthy checks — avoids `ValueError: truth value ambiguous` if
        # _suggested_template ever becomes a numpy array (same class
        # of bug as BUG-META-STATS-PERSISTENCE; see
        # memory/feedback_numpy_truthy_persistence.md).
        if (self._chain_iql is not None
                and self._suggested_template is not None
                and float(self._suggested_template_q or 0.0) > 0.5):
            try:
                from titan_plugin.logic.task_embedding import template_to_primitive_list
                tmpl_prims = template_to_primitive_list(self._suggested_template)
                step_idx = len(self.state.chain)
                if step_idx < len(tmpl_prims):
                    suggested = tmpl_prims[step_idx]
                    if suggested != prim_name and suggested in META_PRIMITIVES:
                        # Override probability scales with Q above 0.5
                        override_prob = (self._suggested_template_q - 0.5) * self._chain_iql.blend_alpha * 2.0
                        override_prob = max(0.0, min(1.0, override_prob))
                        if np.random.random() < override_prob:
                            prim_idx = META_PRIMITIVES.index(suggested)
                            prim_name = suggested
                            logger.info(
                                "[META] Template bias override → %s (Q=%.3f, p=%.2f)",
                                suggested, self._suggested_template_q, override_prob,
                            )
            except Exception as _tb_err:
                # Soft fail — template bias is a non-critical optimization
                swallow_warn('[logic.meta_reasoning] MetaReasoningEngine.tick: from titan_plugin.logic.task_embedding import template_to...', _tb_err,
                             key='logic.meta_reasoning.MetaReasoningEngine.tick.line1181', throttle=100)

        # Audit telemetry: if ε-greedy picked INTROSPECT but template-bias
        # just rerouted to something else, count as rerouted. Preserves the
        # invariant picks == exec + rerouted. Placed HERE (before gates) so
        # that a subsequent INTROSPECT gate reroute doesn't double-count.
        # 2026-04-12 fix: previously template bias silently stole picks.
        if _epsilon_picked_introspect and prim_name != "INTROSPECT":
            self._introspect_rerouted_lifetime += 1

        # M8: SPIRIT_SELF gate (50 chains) + cooldown enforcement
        if prim_name == "SPIRIT_SELF":
            if self._total_meta_chains < self._spirit_self_gate:
                self._reroute_count = getattr(self, '_reroute_count', 0) + 1
                if self._reroute_count <= 3 or self._reroute_count % 100 == 0:
                    logger.warning("[META] SPIRIT_SELF→EVALUATE reroute #%d: gate not met (%d < %d)",
                                   self._reroute_count, self._total_meta_chains, self._spirit_self_gate)
                prim_name = "EVALUATE"
                prim_idx = META_PRIMITIVES.index("EVALUATE")
            elif self.state.spirit_self_cooldown > 0:
                self._reroute_count = getattr(self, '_reroute_count', 0) + 1
                if self._reroute_count <= 3 or self._reroute_count % 100 == 0:
                    logger.warning("[META] SPIRIT_SELF→EVALUATE reroute #%d: cooldown=%d",
                                   self._reroute_count, self.state.spirit_self_cooldown)
                prim_name = "EVALUATE"
                prim_idx = META_PRIMITIVES.index("EVALUATE")

        # INTROSPECT gate: maturity-gated + cooldown (max 1 per chain)
        if prim_name == "INTROSPECT":
            _sr = getattr(self, '_self_reasoning', None)
            _introspect_gate = self._config.get("gate_introspect_chains", 20)
            _reason = None
            if self._total_meta_chains < _introspect_gate:
                _reason = f"gate({self._total_meta_chains}<{_introspect_gate})"
            elif _sr and not _sr.can_introspect:
                _reason = "cooldown(can_introspect=False)"
            elif self.state.introspect_used:
                _reason = "already_used_in_chain"
            if _reason is not None:
                # Audit telemetry: track gate-reroutes so picks/executions
                # invariant is `picks == executions + rerouted` (by-design).
                self._introspect_rerouted_lifetime += 1
                self._reroute_count = getattr(self, '_reroute_count', 0) + 1
                if self._reroute_count <= 3 or self._reroute_count % 100 == 0:
                    logger.warning("[META] INTROSPECT→EVALUATE reroute #%d: %s",
                                   self._reroute_count, _reason)
                prim_name = "EVALUATE"
                prim_idx = META_PRIMITIVES.index("EVALUATE")

        # M7: BREAK cap enforcement
        if prim_name == "BREAK" and self.state.break_count >= self.state.max_breaks:
            self._reroute_count = getattr(self, '_reroute_count', 0) + 1
            if self._reroute_count <= 3 or self._reroute_count % 100 == 0:
                logger.warning("[META] BREAK→EVALUATE reroute #%d: cap reached (%d/%d)",
                               self._reroute_count, self.state.break_count, self.state.max_breaks)
            prim_name = "EVALUATE"
            prim_idx = META_PRIMITIVES.index("EVALUATE")

        # Decrement M8 cooldown
        self.state.spirit_self_cooldown = max(0, self.state.spirit_self_cooldown - 1)

        # Select sub-mode
        sub_input = self._build_sub_mode_input(sv, nm, prim_name)
        sub_idx = self.sub_mode_policies[prim_name].select_action(sub_input, temperature)
        sub_name = SUB_MODES[prim_name][sub_idx]

        # ── TUNING-012 v2: Capture pre-state for compound rewards ──
        # EVALUATE needs to know the confidence BEFORE the eval happens
        # so the info_gain component can measure delta.
        if prim_name == "EVALUATE":
            self.state.pre_eval_confidence = float(self.state.confidence)
        # BREAK needs to know the average reward BEFORE the break so the
        # recovery component can measure post-break improvement.
        if prim_name == "BREAK":
            recent_rewards = [r for r in self.buffer._rewards[-10:] if r is not None]
            self.state.pre_break_avg_reward = (
                sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            )

        # Execute primitive
        result = self._execute(prim_name, sub_name, sv, nm,
                               reasoning_engine, chain_archive, meta_wisdom,
                               ex_mem, meta_autoencoder)

        # Audit telemetry: count INTROSPECT successful executions. Divergence
        # vs _introspect_picks_lifetime indicates a re-emergent crash.
        if prim_name == "INTROSPECT":
            self._introspect_executions_lifetime += 1

        # ── TUNING-012 v2: Capture post-state for compound rewards ──
        # RECALL: track which source we recalled from for entropy calculation
        if prim_name == "RECALL":
            self.state.recall_history.append({
                "source": sub_name,
                "count": result.get("count", 0),
            })

        # Record transition
        step_key = f"{prim_name}.{sub_name}"
        step_reward = STEP_REWARDS.get(step_key, 0.0)
        if result.get("count", 0) > 0:
            step_reward += min(result["count"] * 0.02, 0.10)
        if result.get("wisdom_found"):
            step_reward += 0.05

        self.buffer.record(meta_input, prim_idx, sub_idx, step_reward)

        # Update chain state
        self.state.chain.append(step_key)
        self.state.chain_results.append(result)
        self.state.step_rewards.append(step_reward)
        self.state.confidence = result.get("confidence", self.state.confidence)
        self._total_meta_steps += 1

        # M7: Auto-checkpoint at FORMULATE/SYNTHESIZE
        if prim_name in ("FORMULATE", "SYNTHESIZE"):
            self._save_checkpoint()

        # Update strategy history EMA (expanded to 8 primitives)
        sh_len = min(NUM_META_ACTIONS, 6)  # strategy_history is 12D: [0:6] EMA + [6:12] recency
        one_hot = np.zeros(sh_len, dtype=np.float32)
        if prim_idx < sh_len:
            one_hot[prim_idx] = 1.0
        self._strategy_history[:sh_len] = 0.9 * self._strategy_history[:sh_len] + 0.1 * one_hot
        self._strategy_history[sh_len:] = 0.95 * self._strategy_history[sh_len:]
        if prim_idx < len(self._strategy_history) - sh_len:
            self._strategy_history[sh_len + prim_idx] = 1.0

        # M9: EUREKA detection (insight-capable primitives)
        eureka_event = None
        if prim_name in self._eureka_thresholds and self._eureka_cooldown_steps <= 0:
            step_conf = result.get("confidence", 0)
            threshold = self._eureka_thresholds[prim_name]
            if step_conf > threshold:
                eureka_event = self._fire_eureka(
                    step_conf, sv, meta_wisdom, meta_autoencoder,
                    trigger_primitive=prim_name)
                # TUNING-012 v2: BREAK→EUREKA recovery signal for reward_break
                if self.state.break_count > 0:
                    self.state.eureka_after_break = True
                # P7: carry EUREKA tags through chain state to META-CGN hook.
                # Last fire wins on trigger (rare in one chain, but bounded).
                self.state.eureka_fired = True
                self.state.eureka_trigger = prim_name
        if self._eureka_cooldown_steps > 0:
            self._eureka_cooldown_steps -= 1

        # Check termination
        if self._should_conclude():
            concluded = self._conclude_chain(sv, chain_archive, meta_wisdom, meta_autoencoder)
            if eureka_event:
                concluded["eureka"] = eureka_event
            return concluded

        result_dict = {
            "action": "CONTINUE", "primitive": prim_name, "sub_mode": sub_name,
            "result": result, "chain_length": len(self.state.chain),
            "max_steps": self.state.max_steps, "confidence": self.state.confidence,
        }
        if eureka_event:
            result_dict["eureka"] = eureka_event
        if result.get("nudge_request"):
            result_dict["nudge_request"] = result["nudge_request"]
        return result_dict

    def gates_passed(self, pi_monitor, reasoning_engine, coordinator) -> bool:
        """Check developmental maturity gates."""
        if not reasoning_engine:
            return False
        if reasoning_engine._total_chains < self._gates["reasoning_chains"]:
            return False
        if pi_monitor and pi_monitor.developmental_age < self._gates["pi_clusters"]:
            return False
        if coordinator and hasattr(coordinator, 'inner'):
            if coordinator.inner and coordinator.inner.cycle_count < self._gates["dream_cycles"]:
                return False
        return True

    def consolidate_training(self, boost_factor=3.0) -> dict:
        """Dream-time System 1 training."""
        batch = self.buffer.sample(batch_size=min(64, self.buffer.size()))
        if not batch:
            return {"trained": False}
        states, actions, sub_actions, rewards, dones = batch
        original_lr = self.meta_policy.lr
        self.meta_policy.lr = min(original_lr * boost_factor, original_lr * 5.0)
        total_loss = 0.0
        for s, a, sa, r, d in zip(states, actions, sub_actions, rewards, dones):
            advantage = r - 0.05  # baseline
            loss = self.meta_policy.train_step(np.array(s), a, advantage)
            total_loss += loss
            # Train sub-mode policy too
            prim_name = META_PRIMITIVES[min(a, len(META_PRIMITIVES) - 1)]
            sub_input = s[:SUB_MODE_INPUT_DIM]
            self.sub_mode_policies[prim_name].train_step(sub_input, sa, advantage)
        self.meta_policy.lr = original_lr
        n = len(states)

        # ── TUNING-012 v2 Sub-phase B: Chain-level IQL training ──
        # Train the chain template Q-net on the buffer of chain outcomes
        # collected since the last dream. Runs alongside the per-step
        # MetaPolicy training.
        chain_iql_stats = {"trained": False}
        if self._chain_iql and self._chain_iql.enabled:
            try:
                chain_iql_stats = self._chain_iql.consolidate_during_dream(batch_size=64)
                if chain_iql_stats.get("trained"):
                    logger.info(
                        "[META] Chain IQL dream training: %d samples, loss=%.4f, "
                        "templates=%d, updates=%d",
                        chain_iql_stats["samples"], chain_iql_stats["avg_loss"],
                        chain_iql_stats["template_count"], chain_iql_stats["total_updates"],
                    )
            except Exception as _ci_err:
                logger.warning("[META] Chain IQL consolidation failed: %s", _ci_err)

        return {"trained": True, "samples": n,
                "avg_loss": round(total_loss / max(n, 1), 6),
                "buffer_size": self.buffer.size(),
                "total_updates": self.meta_policy.total_updates,
                "chain_iql": chain_iql_stats}

    def save_all(self):
        self.meta_policy.save(os.path.join(self.save_dir, "meta_policy.json"))
        sub_data = {k: v.to_dict() for k, v in self.sub_mode_policies.items()}
        path = os.path.join(self.save_dir, "meta_sub_modes.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sub_data, f)
        os.replace(tmp, path)
        self.buffer.save(os.path.join(self.save_dir, "meta_buffer.json"))
        # TUNING-012 v2 Sub-phase B: persist chain IQL state
        if self._chain_iql:
            try:
                self._chain_iql.save()
            except Exception as _ce_err:
                logger.warning("[META] Chain IQL save failed: %s", _ce_err)
        # META-CGN Phase 1: persist primitive grounding state
        if self._meta_cgn is not None:
            try:
                self._meta_cgn.save_state()
            except Exception as _mcgn_save_err:
                logger.warning("[META] META-CGN save failed: %s",
                               _mcgn_save_err)
        # Save stats — bulletproof pre-flight write.
        # 2026-04-19 CRITICAL FIX: prior atomic-write pattern (open(w) +
        # os.replace) was fundamentally fragile — open("w") truncates
        # tmp to 0 bytes BEFORE any serialization check, so any
        # json.dump exception left a 0-byte tmp that could corrupt the
        # path file on os.replace. All 3 Titans lost meta-reasoning
        # counter persistence since 2026-04-17 18:40-ish because a
        # non-serializable attribute (likely numpy/torch leak into
        # _diversity_pressure_target / _suggested_template) made
        # json.dump raise on every dream cycle.
        #
        # Bulletproof fix:
        #   1. Pre-flight: json.dumps() to an in-memory STRING. No file
        #      touched. Uses default=_jsonable_default — a catch-all
        #      that NEVER raises (falls back to repr() of the value).
        #      Result: json.dumps always succeeds, even on exotic types.
        #   2. Atomic write with PRE-SERIALIZED string. We only open(w)
        #      tmp after we know the content is valid, so the tmp file
        #      can't be 0 bytes.
        #   3. os.replace only on successful write (same as before).
        #   4. Outer try/except still wraps everything with WARN + tmp
        #      cleanup as defense-in-depth against disk-full / permission
        #      / other I/O failures.
        stats_path = os.path.join(self.save_dir, "meta_stats.json")
        tmp_stats = stats_path + ".tmp"

        def _jsonable_default(x):
            """json.dumps default= handler. NEVER raises.
            Coerces any unknown type to a JSON-safe value with progressively
            more lossy fallbacks — tolist → item → to_dict → repr → placeholder.
            """
            try:
                if hasattr(x, "tolist"):
                    return x.tolist()
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._jsonable_default: if hasattr(x, 'tolist'): return x.tolist()", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._jsonable_default.line1460', throttle=100)
            try:
                if hasattr(x, "item"):
                    return x.item()
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._jsonable_default: if hasattr(x, 'item'): return x.item()", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._jsonable_default.line1465', throttle=100)
            try:
                if hasattr(x, "to_dict"):
                    return x.to_dict()
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._jsonable_default: if hasattr(x, 'to_dict'): return x.to_dict()", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._jsonable_default.line1470', throttle=100)
            try:
                return repr(x)[:200]
            except Exception:
                return "<unrepresentable>"

        # Wrap EVERYTHING (payload construction + serialize + write) in
        # one try/except. The prior fix only wrapped json.dumps+write;
        # payload construction could raise (e.g., numpy truth-value
        # ambiguous error when an attribute unexpectedly became an
        # array — confirmed as the actual failure mode from brain log
        # analysis: "[Coordinator] Meta-reasoning consolidation error:
        # The truth value of an array with more than one element is
        # ambiguous. Use a.any() or a.all()" firing every consolidation
        # since 16:34 on 2026-04-19).
        #
        # Also: replace `and X` truthy checks with explicit isinstance
        # type guards so a numpy-array leak can't trigger the truth-value
        # error. Bool conversions go through `_safe_bool_nonempty()`
        # which handles dicts, sequences, and arrays correctly.
        def _safe_nonempty_dict(attr_name: str) -> dict:
            """Return {k: float(v)} for a dict-attribute, else {}.
            Guards against the attribute being a non-dict (e.g., numpy
            array leak) by explicit isinstance check."""
            x = getattr(self, attr_name, None)
            if isinstance(x, dict) and len(x) > 0:
                try:
                    return {str(k): float(v) for k, v in x.items()}
                except Exception:
                    return {}
            return {}

        def _safe_int(attr_name: str, default: int = 0) -> int:
            x = getattr(self, attr_name, default)
            try:
                return int(x) if x is not None else default
            except Exception:
                return default

        def _safe_float(attr_name: str, default: float = 0.0) -> float:
            x = getattr(self, attr_name, default)
            try:
                return float(x) if x is not None else default
            except Exception:
                return default

        try:
            stats_payload = {
                "total_chains": _safe_int("_total_meta_chains"),
                "total_steps": _safe_int("_total_meta_steps"),
                "total_wisdom_saved": _safe_int("_total_wisdom_saved"),
                "total_eurekas": _safe_int("_total_eurekas"),
                "baseline_confidence": _safe_float(
                    "_baseline_confidence", 0.5),
                "strategy_history": self._strategy_history.tolist()
                if hasattr(self, "_strategy_history")
                and hasattr(self._strategy_history, "tolist") else [],
                "ema_state": self._ema_state.tolist()
                if hasattr(self, "_ema_state")
                and hasattr(self._ema_state, "tolist") else [],
                "recent_chain_uniques": (
                    list(self._recent_chain_uniques)
                    if hasattr(self, "_recent_chain_uniques") else []),
                "primitive_bias": _safe_nonempty_dict("_primitive_bias"),
                "diversity_pressure_target": _jsonable_default(
                    getattr(self, "_diversity_pressure_target", None)),
                "reroute_count": _safe_int("_reroute_count"),
                "suggested_template": _jsonable_default(
                    getattr(self, "_suggested_template", None)),
                "suggested_template_q": _safe_float(
                    "_suggested_template_q"),
                "subsystem_cache_pending": bool(
                    getattr(self, "_subsystem_cache_pending", False)),
                # Cognitive-contract handler counters. Before 2026-04-21 these
                # were in-memory only, so every restart reset them to 0 —
                # dashboard showed fires=0 even when the CONTRACTS fired
                # correctly (orchestrator log proves firings, handler log
                # proves handlers ran). Root cause of BUG-CONTRACT-GATE-
                # STARVATION's "fires=0" symptom. Counters now persist so
                # lifetime accounting survives the frequent restarts caused
                # by other bugs being fixed. Plain lists/dicts are passed
                # through (json.dumps handles them); _jsonable_default would
                # repr() them into strings.
                "cc_strategy_drift_fires": _safe_int("_cc_strategy_drift_fires"),
                "cc_strategy_drift_last_top": list(
                    getattr(self, "_cc_strategy_drift_last_top", []) or []),
                "cc_pattern_emerged_fires": _safe_int("_cc_pattern_emerged_fires"),
                "cc_pattern_emerged_last": list(
                    getattr(self, "_cc_pattern_emerged_last", []) or []),
                "cc_monoculture_fires": _safe_int("_cc_monoculture_fires"),
                "cc_monoculture_last": dict(
                    getattr(self, "_cc_monoculture_last", {}) or {}),
                # --- Audit-driven additions (2026-04-21) ---
                # Systematic sweep of MetaReasoningEngine lifetime/stateful
                # attributes not previously persisted. Surfaced during the
                # BUG-CONTRACT-GATE-STARVATION investigation — `save_all`
                # had grown incrementally with 20 keys while __init__
                # carries 30+ stateful attributes, most of them exposed
                # in get_stats() as "*_lifetime" or mid-decay state. Every
                # restart silently reset them to 0. Fix parallels cc_*
                # counters: persist the ones clearly lifetime/stateful,
                # skip the ones that are recomputable or short-lived.
                "inengine_mono_total_fires":
                    _safe_int("_inengine_mono_total_fires"),
                "inengine_mono_last_fire_chain":
                    _safe_int("_inengine_mono_last_fire_chain", -1),
                "mono_adj_fires_count": _safe_int("_mono_adj_fires_count"),
                "mono_adj_cumulative": _safe_float("_mono_adj_cumulative"),
                "diversity_pressure_total_applied":
                    _safe_int("_diversity_pressure_total_applied"),
                "diversity_pressure_remaining":
                    _safe_int("_diversity_pressure_remaining"),
                "diversity_pressure_initial_magnitude":
                    _safe_float("_diversity_pressure_initial_magnitude"),
                "diversity_pressure_initial_decay":
                    _safe_int("_diversity_pressure_initial_decay"),
                "introspect_executions_lifetime":
                    _safe_int("_introspect_executions_lifetime"),
                "introspect_picks_lifetime":
                    _safe_int("_introspect_picks_lifetime"),
                "introspect_rerouted_lifetime":
                    _safe_int("_introspect_rerouted_lifetime"),
                "next_chain_id": _safe_int("_next_chain_id", 1),
                "last_concluded_chain_id":
                    _safe_int("_last_concluded_chain_id"),
                "repeat_impasse_count": _safe_int("_repeat_impasse_count"),
                "repeat_impasse_primitives":
                    _safe_nonempty_dict("_repeat_impasse_primitives"),
                "soar_last_successful_topic": str(
                    getattr(self, "_soar_last_successful_topic", "") or ""),
            }
            # Step 1 — pre-flight serialize to string (in-memory, no
            # file touched). default= handles any exotic type.
            payload_str = json.dumps(stats_payload,
                                     default=_jsonable_default)
            # Step 2 — atomic write of pre-serialized content. tmp can
            # only reach this block with valid content.
            with open(tmp_stats, "w") as f:
                f.write(payload_str)
            os.replace(tmp_stats, stats_path)
        except Exception as _ss_err:
            # Catches ANYTHING — payload construction, serialization, I/O.
            # Before this, numpy truth-value errors in payload construction
            # escaped all wrappers and propagated to the coordinator level
            # ("Meta-reasoning consolidation error" logged there every
            # dream cycle, but meta_stats save silently skipped).
            logger.warning(
                "[META] meta_stats.json save FAILED: %s — previous file "
                "preserved (no overwrite). Attr types: "
                "primitive_bias=%s reroute_count=%s suggested_template=%s "
                "ema_state=%s",
                _ss_err,
                type(getattr(self, "_primitive_bias", None)).__name__,
                type(getattr(self, "_reroute_count", None)).__name__,
                type(getattr(self, "_suggested_template", None)).__name__,
                type(getattr(self, "_ema_state", None)).__name__,
            )
            try:
                if os.path.exists(tmp_stats):
                    os.remove(tmp_stats)
            except Exception as _swallow_exc:
                swallow_warn('[logic.meta_reasoning] MetaReasoningEngine.save_all: if os.path.exists(tmp_stats): os.remove(tmp_stats)', _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine.save_all.line1631', throttle=100)

    def get_stats(self) -> dict:
        prim_counts = {}
        for a in self.buffer._actions:
            name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
            prim_counts[name] = prim_counts.get(name, 0) + 1
        rewards = [r for r, d in zip(self.buffer._rewards, self.buffer._dones) if d]
        avg_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        stats = {
            "total_chains": self._total_meta_chains,
            "total_steps": self._total_meta_steps,
            "total_wisdom_saved": self._total_wisdom_saved,
            "total_eurekas": self._total_eurekas,
            # commit telemetry — additively exposed for self_reasoning Layer A
            # novelty-gap detection (rFP_coding_explorer_activation.md §4.1)
            # 2026-04-21 hotfix: `_total_conclusions` lives on reasoning_engine
            # (see L457), not on MetaReasoningEngine. Use getattr fallback so
            # get_stats() doesn't AttributeError out — was breaking the entire
            # meta-reasoning dashboard (META CHAINS/WISDOM/AVG REWARD all
            # showed 0/empty because every tick raised AttributeError and the
            # except handler returned {}). 1310 errors on T1 before this fix.
            # Proper fix: wire reasoning_engine._total_conclusions through to
            # here, or rename this metric — follow-up, not urgent.
            "total_commits": getattr(self, "_total_conclusions", 0),
            "commit_rate": (
                getattr(self, "_total_conclusions", 0) / self._total_meta_chains
                if self._total_meta_chains > 0 else 0.0),
            "baseline_confidence": round(self._baseline_confidence, 4),
            "buffer_size": self.buffer.size(),
            "policy_updates": self.meta_policy.total_updates,
            "is_active": self.state.is_active,
            "chain_length": len(self.state.chain) if self.state.is_active else 0,
            "primitive_counts": prim_counts,
            "avg_reward": avg_reward,
            "spirit_self_unlocked": self._total_meta_chains >= self._spirit_self_gate,
            "parallel_enabled": self._parallel_enabled,
            "resource_budget": self.max_chain_length,
            "introspect_unlocked": self._total_meta_chains >= self._config.get("gate_introspect_chains", 20),
            # TUNING-012 v2 Sub-phase A: compound rewards status
            "compound_rewards_enabled": self._compound_rewards_enabled(),
            "dna_param_count": len(self._dna),
        }
        # TUNING-012 v2 Sub-phase B: chain IQL stats
        if self._chain_iql:
            stats["chain_iql"] = self._chain_iql.get_stats()
        # META-CGN Phase 1 — primitive grounding telemetry
        if self._meta_cgn is not None:
            try:
                stats["meta_cgn"] = self._meta_cgn.get_stats()
            except Exception as _mcgn_stats_err:
                stats["meta_cgn"] = {"error": str(_mcgn_stats_err)}
        # TUNING-012 v2 Sub-phase C: cognitive contracts + diversity pressure
        stats["cognitive_contracts"] = {
            "dna_param_count": len(self._contracts_dna),
            "eureka_thresholds": dict(self._eureka_thresholds),
            "diversity_pressure": {
                "active": bool(self._diversity_pressure_remaining > 0),
                "target": self._diversity_pressure_target,
                "remaining_chains": int(self._diversity_pressure_remaining),
                "current_bias": float(
                    -self._primitive_bias.min() if np.any(self._primitive_bias) else 0.0),
                "initial_magnitude": float(self._diversity_pressure_initial_magnitude),
                "initial_decay_chains": int(self._diversity_pressure_initial_decay),
                "total_applied": int(self._diversity_pressure_total_applied),
            },
            # R2 cosmetic fix: handler fire counts + last-fire data, set by
            # spirit_worker on each contract trigger. These attributes live on
            # meta_engine (same sub-process as spirit_worker), so they flow
            # through the cached coordinator naturally without cross-process
            # bus plumbing.
            "handlers": {
                "strategy_drift": {
                    "fires": int(getattr(self, "_cc_strategy_drift_fires", 0)),
                    "last_top_templates": list(
                        getattr(self, "_cc_strategy_drift_last_top", []) or []),
                },
                "pattern_emerged": {
                    "fires": int(getattr(self, "_cc_pattern_emerged_fires", 0)),
                    "last_emerging": list(
                        getattr(self, "_cc_pattern_emerged_last", []) or []),
                },
                "monoculture": {
                    "fires": int(getattr(self, "_cc_monoculture_fires", 0)),
                    "last": dict(getattr(self, "_cc_monoculture_last", {}) or {}),
                },
            },
        }
        return stats

    def get_audit_stats(self) -> dict:
        """Observability snapshot for /v4/meta-reasoning/audit (Task 3).

        Surface for diagnosing meta-reasoning healing dynamics:
        diversity, monoculture pressure, per-primitive reward components,
        contract fire history, INTROSPECT health. The `meta_cgn` block is
        intentionally a stub here; it will be populated when META-CGN
        (Task 7) lands and chain templates become CGN concepts.
        """
        # ── Diversity ────────────────────────────────────────────────
        recent_uniques = list(self._recent_chain_uniques)
        unique_ema = (
            sum(recent_uniques) / len(recent_uniques)
            if recent_uniques else 0.0
        )

        # ── Monoculture (current state from last 500 actions) ───────
        recent_actions = self.buffer._actions[-500:] if hasattr(self.buffer, '_actions') else []
        prim_counts: dict = {}
        for a in recent_actions:
            name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
            prim_counts[name] = prim_counts.get(name, 0) + 1
        total_a = sum(prim_counts.values())
        dominant_name = ""
        dom_share = 0.0
        if total_a > 0:
            dominant_name, dominant_n = max(prim_counts.items(), key=lambda x: x[1])
            dom_share = dominant_n / total_a

        # ── Per-primitive reward components (averaged over last 100) ─
        rewards_per_prim: dict = {}
        for prim, breakdowns in self._recent_primitive_breakdowns.items():
            if not breakdowns:
                continue
            n = len(breakdowns)
            # Sum each component across the breakdown deque, then divide.
            comp_sums: dict = {}
            total_sum = 0.0
            for bd in breakdowns:
                for k, v in bd.items():
                    if k == "total":
                        total_sum += float(v)
                        continue
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
            rewards_per_prim[prim] = {
                "avg_total": round(total_sum / n, 4),
                "n": n,
                "components": {k: round(v / n, 4) for k, v in comp_sums.items()},
            }

        # ── Contracts (handler fire metadata already in get_stats) ───
        contract_handlers = {
            "monoculture_detector": {
                "fires_lifetime": int(getattr(self, "_cc_monoculture_fires", 0)),
                "last": dict(getattr(self, "_cc_monoculture_last", {}) or {}),
            },
            "strategy_evolution": {
                "fires_lifetime": int(getattr(self, "_cc_strategy_drift_fires", 0)),
                "last_top_templates": list(
                    getattr(self, "_cc_strategy_drift_last_top", []) or []),
            },
            "abstract_pattern_extraction": {
                "fires_lifetime": int(getattr(self, "_cc_pattern_emerged_fires", 0)),
                "last_emerging": list(
                    getattr(self, "_cc_pattern_emerged_last", []) or []),
            },
        }

        return {
            "diversity": {
                "unique_prims_ema_50chains": round(float(unique_ema), 3),
                "unique_prims_per_chain_recent": recent_uniques[-20:],
                "current_epsilon": round(float(self._adaptive_epsilon), 3),
            },
            "monoculture": {
                "dominant_primitive": dominant_name,
                "dominant_share_500": round(float(dom_share), 3),
                "primitive_counts_500": dict(prim_counts),
                "mono_adj_fires_lifetime": int(self._mono_adj_fires_count),
                "mono_adj_cumulative": round(float(self._mono_adj_cumulative), 3),
                "mono_adj_recent": list(self._mono_adj_history)[-10:],
            },
            "diversity_pressure": {
                "active": bool(self._diversity_pressure_remaining > 0),
                "target": self._diversity_pressure_target,
                "remaining_chains": int(self._diversity_pressure_remaining),
                "current_bias": float(
                    -self._primitive_bias.min() if np.any(self._primitive_bias) else 0.0),
                "initial_magnitude": float(self._diversity_pressure_initial_magnitude),
                "initial_decay_chains": int(self._diversity_pressure_initial_decay),
                "total_fires_lifetime": int(self._diversity_pressure_total_applied),
                "fire_history": list(self._diversity_pressure_fire_history),
                # Task 4 P1: in-engine control loop telemetry
                "inengine_check": {
                    "threshold": float(self._inengine_mono_threshold),
                    "magnitude": float(self._inengine_mono_pressure_magnitude),
                    "decay_chains": int(self._inengine_mono_pressure_decay),
                    "fires_lifetime": int(self._inengine_mono_total_fires),
                    "last_fire_chain": int(self._inengine_mono_last_fire_chain),
                    "chains_since_last_fire": (
                        int(self._total_meta_chains - self._inengine_mono_last_fire_chain)
                        if self._inengine_mono_last_fire_chain >= 0 else None
                    ),
                },
            },
            "rewards_per_primitive": rewards_per_prim,
            "recent_chain_rewards": list(self._recent_chain_rewards)[-20:],
            "contracts": contract_handlers,
            "introspect_health": {
                "picks_lifetime": int(self._introspect_picks_lifetime),
                "executions_lifetime": int(self._introspect_executions_lifetime),
                "rerouted_lifetime": int(self._introspect_rerouted_lifetime),
                # Invariant when fix is healthy: picks <= executions + rerouted.
                # (≤ because meta-policy forward pass can also pick INTROSPECT
                # without going through ε-greedy; those bypass the picks counter.)
                "fix_healthy": int(self._introspect_picks_lifetime) <= (
                    int(self._introspect_executions_lifetime) +
                    int(self._introspect_rerouted_lifetime)
                ),
                "introspect_unlocked": self._total_meta_chains >= self._config.get(
                    "gate_introspect_chains", 20),
            },
            "subsystem_signals_status": self._compute_subsystem_signals_status(),
            "meta_cgn": {
                # META-CGN-ready stub — populated when Task 7 (META-CGN) lands.
                "templates_grounded": 0,
                "haov_hypotheses_tested": 0,
                "cross_consumer_signals": {},
                "status": "pending_implementation",
            },
        }

    def _compute_subsystem_signals_status(self) -> dict:
        """P3 (Task 4): which compound-reward subsystem signals are alive vs
        stubbed at default. The compound reward design has 14 signals; if
        most are 0/default, minority primitives (RECALL, INTROSPECT, BREAK)
        can never earn comparable rewards to FORMULATE no matter what
        diversity machinery we apply. This block makes the dead-signal
        problem visible in the audit so we can reason about whether to
        accelerate Sub-phase D wiring vs wait for META-CGN to replace it.
        """
        try:
            from titan_plugin.logic.meta_reasoning_rewards import (
                empty_subsystem_signals,
            )
            stub = empty_subsystem_signals()
            current = self._subsystem_cache or {}
            live = []
            dead = []
            for k, default_v in stub.items():
                v = current.get(k, default_v)
                if v != default_v:
                    live.append({"key": k, "value": round(float(v), 3)})
                else:
                    dead.append(k)
            return {
                "total_signals": len(stub),
                "live_count": len(live),
                "dead_count": len(dead),
                "live": live,
                "dead": dead,
                "cache_age_seconds": (
                    round(time.time() - self._subsystem_cache_ts, 1)
                    if self._subsystem_cache_ts > 0 else None
                ),
                "note": (
                    "Dead signals default to 0 → minority primitives stay at "
                    "base reward → structural bias toward FORMULATE. "
                    "Resolution path: META-CGN (Task 7) supersedes compound "
                    "rewards with grounded V(s)."
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def set_self_reasoning(self, engine):
        """Attach SelfReasoningEngine for INTROSPECT primitive.

        Called by spirit_worker after both engines are initialized.
        """
        self._self_reasoning = engine

    # ── Private: Chain Lifecycle ──────────────────────────────────

    def _start_chain(self, reason, state_132d):
        self.state = MetaChainState()
        self.state.is_active = True
        self.state.trigger_reason = reason
        self.state.start_time = time.time()
        self.state.pre_state_132d = list(state_132d[:132])
        # Phase D.1 — assign monotonic chain_id for external reward correlation.
        self.state.chain_id = self._next_chain_id
        self._next_chain_id += 1
        self._total_meta_chains += 1
        # rFP_titan_meta_outer_layer — clear in-flight outer future from prior chain.
        self._reset_outer_state()

        # EMOT-CGN: emit FELT_CLUSTER_UPDATE to worker (Phase 1.6h cutover).
        # Worker's clusterer assigns primitive + updates shm. Meta-reasoning
        # reads dominant-at-start via ShmEmotReader so H5 drift hypothesis
        # still works without an in-process EmotCGNConsumer reference.
        # Failsafe: emit failures don't break chain flow.
        try:
            from titan_plugin.bus import emit_felt_cluster_update
            if self._send_queue is not None and len(state_132d) >= 130:
                emit_felt_cluster_update(
                    self._send_queue, src="spirit",
                    felt_tensor_130d=list(state_132d[:130]))
        except Exception as _swallow_exc:
            swallow_warn('[logic.meta_reasoning] MetaReasoningEngine._start_chain: from titan_plugin.bus import emit_felt_cluster_update', _swallow_exc,
                         key='logic.meta_reasoning.MetaReasoningEngine._start_chain.line1930', throttle=100)
        # Snapshot dominant emotion at chain start from shm (worker-written).
        # Used by H5 drift hypothesis in EMOT_CHAIN_EVIDENCE ctx at conclude.
        try:
            from titan_plugin.logic.emot_shm_protocol import ShmEmotReader
            from titan_plugin.logic.emotion_cluster import EMOT_PRIMITIVES
            _st = ShmEmotReader().read_state()
            self._emot_dom_at_chain_start = (
                EMOT_PRIMITIVES[_st["dominant_idx"]]
                if _st is not None else "FLOW")
        except Exception:
            self._emot_dom_at_chain_start = "FLOW"

        # ── TUNING-012 v2 Sub-phase B: encode task + query best template ──
        # The task domain is unknown until FORMULATE.define runs (it sets
        # state.formulate_output["domain"]). At chain START we use the
        # trigger reason as a proxy and refine later. The task embedding
        # gets updated in _conclude_chain with the final domain.
        self._suggested_template = None
        self._suggested_template_q = 0.0
        if self._chain_iql and self._chain_iql.enabled:
            try:
                from titan_plugin.logic.task_embedding import encode_task
                task_emb = encode_task(
                    domain="pending",  # refined at conclude
                    trigger_reason=reason,
                    state_vector=state_132d,
                    dim=self._chain_iql._task_dim,
                )
                self._chain_task_emb = task_emb
                self._chain_task_domain = "pending"
                # P4: β-aware template selection. In shadow mode this is
                # identical to chain_iql's top pick. In graduating/active,
                # β reranks the top-3 candidates by blended score.
                # min_q=0 for β path — β needs full visibility to score; the
                # min_q=0.4 gate is re-applied via fallback below when β
                # declines/errors.
                best = None
                q = 0.0
                if self._meta_cgn is not None and hasattr(
                        self._chain_iql, "query_top_k_templates"):
                    try:
                        candidates = self._chain_iql.query_top_k_templates(
                            task_emb, k=3, min_q=0.0)
                        if candidates:
                            self._resolve_dominant_primitive()
                            # P6 I3: expose current domain so β composition can
                            # use per-domain V when available (falls back to
                            # pooled otherwise). Best guess at selection time:
                            # formulate_output.domain if FORMULATE already ran,
                            # else "general".
                            _dom_guess = "general"
                            if isinstance(self.state.formulate_output, dict):
                                _dom_guess = str(self.state.formulate_output.get(
                                    "domain", "general"))
                            shadow_ctx = {
                                "chain_len": 0,
                                "chain_id": self.state.chain_id,
                                "domain": _dom_guess,
                                "monoculture_share": float(
                                    getattr(self, "_last_dominance_share", 0.0)),
                                "is_in_soar_impasse": bool(
                                    getattr(self, "_in_soar_impasse", False)),
                                "confidence_avg_20": float(
                                    self._baseline_confidence),
                            }
                            best, final_score, info = (
                                self._meta_cgn.rerank_templates(
                                    candidates, shadow_ctx))
                            # Use chain_iql Q (not blended) for _suggested_template_q
                            # so existing bias mechanism continues to compute
                            # consistent override probabilities.
                            q = next((cq for ct, cq in candidates
                                      if ct == best), candidates[0][1])
                            if info.get("mode") != "shadow" and \
                                    best != candidates[0][0]:
                                logger.info(
                                    "[META] Chain #%d β-rerank: chose %s "
                                    "(β_score=%.3f) over chain_iql top %s "
                                    "(Q=%.3f, ramp=%.2f)",
                                    self._total_meta_chains, best, final_score,
                                    candidates[0][0], candidates[0][1],
                                    info.get("ramp", 0))
                    except Exception as _sc_err:
                        logger.warning("[META] β-aware template select "
                                       "failed: %s — falling back to α only",
                                       _sc_err)
                # Fallback: pure chain_iql query_best_template
                if best is None:
                    best, q = self._chain_iql.query_best_template(
                        task_emb, min_q=0.4)
                # Preserve legacy min_q=0.4 gate for `_suggested_template`
                # bias (template bias override mechanism requires trusted Q).
                # β still saw and scored low-Q candidates regardless.
                if best and q >= 0.4:
                    self._suggested_template = best
                    self._suggested_template_q = q
                    logger.info(
                        "[META] Chain #%d template bias: %s (Q=%.3f)",
                        self._total_meta_chains, best, q,
                    )
            except Exception as _se_err:
                logger.warning("[META] Chain IQL task encoding failed: %s", _se_err)

        # TUNING-012 v2 Phase D pre-flight: log subsystem signal population.
        # Today (Phase A) all signals are stubbed at 0/0.5 by empty_subsystem_signals().
        # When D.1 wires live bus queries to TimeChain + Contracts, this same
        # log will show the live count, making it trivial to spot which
        # subsystem queries returned data vs stayed empty. Critical for D.1
        # debugging.
        try:
            from titan_plugin.logic.meta_reasoning_rewards import (
                empty_subsystem_signals,
            )
            _stub_signals = empty_subsystem_signals()
            _current_signals = self._subsystem_cache or _stub_signals
            _live_keys = [
                k for k, v in _current_signals.items()
                if k in _stub_signals and v != _stub_signals[k]
            ]
            _total = len(_stub_signals)
            if not _live_keys:
                logger.info(
                    "[META] Subsystem signals: 0/%d live — all stubbed "
                    "(Phase A; D.1 will populate)", _total,
                )
            else:
                _summary = " ".join(
                    f"{k}={_current_signals[k]:.2f}" for k in _live_keys[:6]
                )
                _suffix = f" +{len(_live_keys) - 6}more" if len(_live_keys) > 6 else ""
                logger.info(
                    "[META] Subsystem signals: %d/%d live — %s%s",
                    len(_live_keys), _total, _summary, _suffix,
                )
        except Exception as _ss_err:
            logger.warning("[META] Subsystem signal log failed: %s", _ss_err)

        logger.info("[META] Chain #%d started — trigger=%s", self._total_meta_chains, reason)

    def _should_conclude(self) -> bool:
        n = len(self.state.chain)
        if n >= self.state.max_steps:
            return True
        if n >= self.min_chain_length and self.state.confidence > 0.8:
            return True
        # EVALUATE suggested conclude
        if self.state.chain_results and self.state.chain_results[-1].get("recommendation") == "conclude":
            if n >= self.min_chain_length:
                return True
        return False

    # ── Subsystem signal cache (A-finish per rFP §7.A) ──────────────

    def is_subsystem_cache_stale(self, now: Optional[float] = None) -> bool:
        """True if the subsystem cache is past its TTL OR has never been
        populated. Used by spirit_worker to decide whether to fire fresh
        TIMECHAIN_QUERY + CONTRACT_LIST messages.
        """
        if now is None:
            now = time.time()
        # Never populated → stale
        if self._subsystem_cache_ts == 0.0:
            return True
        return (now - self._subsystem_cache_ts) >= self._subsystem_cache_ttl

    def is_subsystem_cache_pending(self, now: Optional[float] = None) -> bool:
        """True if a refresh is currently in flight (queries sent, responses
        not yet arrived). Used to prevent re-firing while waiting. Pending
        flag auto-expires after 2x TTL as a safety net against lost responses.
        """
        if not self._subsystem_cache_pending:
            return False
        if now is None:
            now = time.time()
        # Safety net: if pending for too long (lost response?), clear it so
        # the next stale check re-fires.
        if (now - self._subsystem_cache_pending_ts) >= (self._subsystem_cache_ttl * 2):
            self._subsystem_cache_pending = False
            return False
        return True

    def mark_subsystem_cache_pending(self, now: Optional[float] = None) -> None:
        """Mark that a refresh has been initiated. Spirit_worker calls this
        immediately after sending TIMECHAIN_QUERY + CONTRACT_LIST so we don't
        re-fire on the next epoch tick before the responses arrive.
        """
        if now is None:
            now = time.time()
        self._subsystem_cache_pending = True
        self._subsystem_cache_pending_ts = now

    def update_subsystem_cache(
        self,
        timechain_results: Optional[list] = None,
        contract_results: Optional[list] = None,
        inner_relevance: Optional[float] = None,
        kuzu_centrality: Optional[float] = None,
        self_prediction_accuracy: Optional[float] = None,
        self_profile_divergence: Optional[float] = None,
        now: Optional[float] = None,
    ) -> dict:
        """Populate the subsystem cache from bus query responses.

        Called by spirit_worker's TIMECHAIN_QUERY_RESP and CONTRACT_LIST_RESP
        handlers (and any future Inner Memory query response). Each call
        merges into the existing cache, so partial responses don't wipe
        previously-populated signals.

        Mappings (heuristic, intentionally simple — to be tuned with telemetry):

        TimeChain (5 signals derived from query_blocks results):
          - timechain_depth        ← count(thought_type="recall") / 10, clipped
          - timechain_novelty      ← 1 - count(thought_type="formulate") / 10, clipped
          - timechain_eval_consistency ← avg significance of thought_type="evaluate"
          - timechain_self_continuity  ← avg significance of self-observation blocks
          - timechain_break_pattern    ← avg significance of thought_type="break"

        Contracts (5 signals derived from list_contracts results):
          - contract_ratified         ← active genesis count, normalized
          - contract_priority         ← active filter count, normalized
          - contract_compliance       ← same as contract_priority (proxy)
          - contract_identity_alignment ← genesis count tagged "identity"
          - contract_break_trigger    ← 1.0 if homeostatic_alert active, else 0.0

        Inner Memory (2 signals; left as 0 in this pass — Phase D follow-up):
          - inner_relevance, kuzu_centrality

        Returns the updated cache dict for observability.
        """
        from titan_plugin.logic.meta_reasoning_rewards import (
            empty_subsystem_signals,
        )
        if now is None:
            now = time.time()

        # Initialize cache from defaults if empty
        if not self._subsystem_cache:
            self._subsystem_cache = empty_subsystem_signals()

        cache = self._subsystem_cache

        if timechain_results is not None:
            try:
                blocks = list(timechain_results) if timechain_results else []
                # Group by thought_type.
                # TimeChain uses: declarative|procedural|episodic|meta|genesis
                # Map to meta-reasoning signal categories:
                _TC_TYPE_MAP = {
                    "declarative": "recall",      # stored knowledge retrieval
                    "procedural": "formulate",     # plans, procedures, how-to
                    "meta": "evaluate",            # meta-level assessment
                    "episodic": "introspect",      # experiential / self-observation
                    "genesis": "break",            # chain genesis = break/restart
                }
                by_type: dict = {}
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    tt = (b.get("thought_type") or b.get("t") or "").lower()
                    # Map TimeChain types to meta-reasoning categories
                    mapped = _TC_TYPE_MAP.get(tt, tt)
                    by_type.setdefault(mapped, []).append(b)

                def _avg_sig(items):
                    sigs = [float(it.get("significance", 0.0) or 0.0) for it in items]
                    return sum(sigs) / len(sigs) if sigs else 0.0

                def _count_norm(name, divisor=10.0):
                    return min(len(by_type.get(name, [])) / divisor, 1.0)

                cache["timechain_depth"] = _count_norm("recall")
                cache["timechain_novelty"] = max(0.0, 1.0 - _count_norm("formulate"))
                cache["timechain_eval_consistency"] = _avg_sig(by_type.get("evaluate", []))
                self_obs = by_type.get("introspect", []) + by_type.get("self_observation", [])
                cache["timechain_self_continuity"] = _avg_sig(self_obs)
                cache["timechain_break_pattern"] = _avg_sig(by_type.get("break", []))
            except Exception as _tc_err:
                logger.warning("[META] update_subsystem_cache TimeChain parse: %s", _tc_err)

        if contract_results is not None:
            try:
                contracts = list(contract_results) if contract_results else []
                active = [
                    c for c in contracts
                    if isinstance(c, dict) and (c.get("status") or "").lower() == "active"
                ]
                # Count by type
                genesis = [c for c in active if (c.get("contract_type") or "").lower() == "genesis"]
                filters = [c for c in active if (c.get("contract_type") or "").lower() == "filter"]

                # Normalize counts: 5 contracts = full signal
                cache["contract_ratified"] = min(len(genesis) / 5.0, 1.0)
                cache["contract_priority"] = min(len(filters) / 5.0, 1.0)
                cache["contract_compliance"] = cache["contract_priority"]

                # Identity alignment: genesis contracts whose id references identity/self.
                # Field is contract_id (Contract.to_dict), not name.
                identity_count = sum(
                    1 for c in genesis
                    if "identity" in (c.get("contract_id") or "").lower()
                    or "self" in (c.get("contract_id") or "").lower()
                )
                cache["contract_identity_alignment"] = min(identity_count / 2.0, 1.0)

                # Break trigger: 1.0 if a homeostatic_alert / cognitive_stall is active
                break_trigger = any(
                    "homeostatic" in (c.get("contract_id") or "").lower()
                    or "alert" in (c.get("contract_id") or "").lower()
                    or "stall" in (c.get("contract_id") or "").lower()
                    for c in active
                )
                cache["contract_break_trigger"] = 1.0 if break_trigger else 0.0
            except Exception as _cc_err:
                logger.warning("[META] update_subsystem_cache Contracts parse: %s", _cc_err)

        if inner_relevance is not None:
            cache["inner_relevance"] = float(max(0.0, min(1.0, inner_relevance)))

        if kuzu_centrality is not None:
            cache["kuzu_centrality"] = float(max(0.0, min(1.0, kuzu_centrality)))

        if self_prediction_accuracy is not None:
            cache["self_prediction_accuracy"] = float(max(0.0, min(1.0, self_prediction_accuracy)))

        if self_profile_divergence is not None:
            cache["self_profile_divergence"] = float(max(0.0, min(1.0, self_profile_divergence)))

        # Bump cache_ts on ANY response. TIMECHAIN_QUERY_RESP and
        # CONTRACT_LIST_RESP always arrive as separate update_subsystem_cache
        # calls, never with both args set. The prior "bump only when both"
        # branch never fired past the first response, so is_subsystem_cache_stale
        # stayed True forever and the audit displayed 24h-old timestamps.
        if timechain_results is not None or contract_results is not None:
            self._subsystem_cache_ts = now
        # Clear pending only when BOTH have arrived in the same call, OR when
        # the ttl*2 safety-net auto-clear fires in is_subsystem_cache_pending.
        if timechain_results is not None and contract_results is not None:
            self._subsystem_cache_pending = False

        return dict(cache)

    def apply_diversity_pressure(
        self,
        primitive_name: str,
        magnitude: float,
        decay_chains: int,
    ) -> bool:
        """TUNING-012 v2 Sub-phase C (R3): apply directed negative bias.

        Called by spirit_worker's META_DIVERSITY_PRESSURE handler when the
        monoculture_detector contract has identified primitive monoculture.
        Adds a NEGATIVE logit penalty on the dominant primitive that decays
        linearly over `decay_chains` chains. This is the active control loop
        that pushes meta-reasoning out of collapse without random ε-greedy
        noise (which is undirected).

        Args:
            primitive_name: e.g. "FORMULATE", "RECALL" — must be in META_PRIMITIVES.
            magnitude: positive float; applied as -magnitude to that primitive's logit.
            decay_chains: number of chains over which the bias persists.

        Returns:
            True if applied, False if primitive_name unknown or magnitude<=0.
        """
        if primitive_name not in META_PRIMITIVES:
            logger.warning(
                "[META] apply_diversity_pressure: unknown primitive '%s'",
                primitive_name)
            return False
        if magnitude <= 0 or decay_chains <= 0:
            return False
        idx = META_PRIMITIVES.index(primitive_name)
        # Replace any prior pressure (latest call wins)
        self._primitive_bias = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
        self._primitive_bias[idx] = -float(magnitude)
        self._diversity_pressure_remaining = int(decay_chains)
        self._diversity_pressure_target = primitive_name
        self._diversity_pressure_initial_magnitude = float(magnitude)
        self._diversity_pressure_initial_decay = int(decay_chains)
        self._diversity_pressure_total_applied += 1
        # Audit telemetry: record fire so we can analyse cadence over time.
        self._diversity_pressure_fire_history.append({
            "ts": time.time(),
            "chain": int(self._total_meta_chains),
            "target": primitive_name,
            "magnitude": float(magnitude),
            "decay_chains": int(decay_chains),
        })
        logger.info(
            "[META] Diversity pressure applied: %s logit -%.2f for %d chains "
            "(total applies: %d)",
            primitive_name, magnitude, decay_chains,
            self._diversity_pressure_total_applied,
        )
        return True

    def _inengine_monoculture_check(self) -> None:
        """Task 4 P1: per-chain monoculture control loop.

        Runs every chain conclusion. Fires diversity_pressure when:
          - Buffer has at least min_actions actions (signal stable)
          - Dominant primitive share >= threshold (monoculture present)
          - No active diversity_pressure (don't stack)
          - Chains since last in-engine fire >= pressure_decay_chains (cooldown
            == decay so coverage is continuous when monoculture persists, no
            stacking when monoculture is fading)

        Architectural rationale: contracts evaluate on genesis seals
        (~4-5x/day, dream-cycle-bound) which is too sparse for meta-reasoning
        control. Genesis seals serve state-checkpoint purpose; control loops
        belong with the controlled subsystem. The monoculture_detector
        contract still fires as a safety net at slower cadence.
        """
        actions = self.buffer._actions[-500:] if hasattr(self.buffer, '_actions') else []
        if len(actions) < self._inengine_mono_min_actions:
            return
        if self._diversity_pressure_remaining > 0:
            return  # already pressured
        # Cooldown: don't re-fire until previous pressure has fully decayed
        if self._inengine_mono_last_fire_chain >= 0:
            chains_since = self._total_meta_chains - self._inengine_mono_last_fire_chain
            if chains_since < self._inengine_mono_pressure_decay:
                return
        # Compute current dominance
        prim_counts: dict = {}
        for a in actions:
            name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
            prim_counts[name] = prim_counts.get(name, 0) + 1
        total = sum(prim_counts.values())
        if total == 0:
            return
        dominant_name, dominant_n = max(prim_counts.items(), key=lambda x: x[1])
        dom_share = dominant_n / total
        if dom_share < self._inengine_mono_threshold:
            return
        # Fire
        applied = self.apply_diversity_pressure(
            primitive_name=dominant_name,
            magnitude=self._inengine_mono_pressure_magnitude,
            decay_chains=self._inengine_mono_pressure_decay,
        )
        if applied:
            self._inengine_mono_last_fire_chain = self._total_meta_chains
            self._inengine_mono_total_fires += 1
            logger.info(
                "[META] In-engine monoculture pressure: %s@%.0f%% → -%.2f for %d chains "
                "(in-engine fires: %d)",
                dominant_name, dom_share * 100,
                self._inengine_mono_pressure_magnitude,
                self._inengine_mono_pressure_decay,
                self._inengine_mono_total_fires,
            )

    # ── rFP_titan_meta_outer_layer wiring ─────────────────────────

    def set_outer_reader(self, reader) -> None:
        """Wire the OuterContextReader. Called by spirit_worker at boot.

        reader may be None — engine continues with outer paths inert.
        """
        self._outer_reader = reader

    def _outer_enabled(self) -> bool:
        """True iff reader is wired AND runtime flag is set."""
        if self._outer_reader is None:
            return False
        try:
            return bool(self._outer_reader.is_active())
        except Exception:
            return False

    def _dispatch_outer_fetch(self) -> None:
        """Submit compose_recall_query if needs_outer is populated + flag on.

        Called at chain start (after FORMULATE populates needs_outer).
        No-op if already dispatched or gate off.
        """
        if self._outer_future is not None:
            return
        if not self._outer_enabled():
            return
        if not self.state.needs_outer:
            return
        try:
            self._outer_future = self._outer_reader.compose_recall_query(
                dict(self.state.entity_refs))
        except Exception as e:
            swallow_warn('[MetaOuter] dispatch err', e,
                         key="logic.meta_reasoning.dispatch_err", throttle=100)
            self._outer_future = None

    def _await_outer_context(self, timeout_s: float = 0.2) -> Optional[dict]:
        """Block up to timeout_s on the in-flight composed-recall future.

        Returns the composed outer_context dict (also stashed onto
        self.state.outer_context), or None if unavailable.
        """
        if not self._outer_enabled():
            return None
        if self.state.outer_context:
            return self.state.outer_context
        if self._outer_future is None:
            # Lazy dispatch if a primitive asked for outer context but
            # FORMULATE didn't emit needs_outer. Use whatever entity_refs
            # we have; safe no-op if empty.
            if self.state.entity_refs:
                self.state.needs_outer = dict(self.state.entity_refs)
                self._dispatch_outer_fetch()
            if self._outer_future is None:
                return None
        try:
            ctx = self._outer_future.result(timeout=float(timeout_s))
        except Exception:
            return None
        if isinstance(ctx, dict):
            self.state.outer_context = ctx
            return ctx
        return None

    def _reset_outer_state(self) -> None:
        """Called at chain start/conclude to drop in-flight future."""
        self._outer_future = None

    def _post_formulate_detect_entities(self) -> None:
        """After FORMULATE, detect entity/topic references in the formulated
        intent. Populate state.entity_refs + state.needs_outer; dispatch
        composed-recall future if gate on.

        Sources scanned (most-specific first):
          - state.formulate_output["participant_person_id"] (direct set by
            upstream formulate paths that already know the person)
          - state.impasse_topic (set by SOAR impasse detection at §D.2)
          - state.trigger_reason (free-text; contains @handle or topic hints)
          - state.formulate_output["problem_template"] (fallback)

        Conservative: only populate entity_refs if a pattern confidently
        matches (starts with '@' → person; explicit "topic:" prefix → topic).
        No false positives on pure anomaly-dim formulations.
        """
        if not self._outer_enabled():
            return
        if self.state.entity_refs or self.state.needs_outer:
            return  # already set — don't clobber
        refs: dict = {}
        # 1. Direct participant (if upstream set it)
        fo = self.state.formulate_output or {}
        pid = fo.get("participant_person_id")
        if isinstance(pid, str) and pid:
            refs["primary_person"] = pid
        # 2. Impasse topic
        topic = self.state.impasse_topic or fo.get("impasse_topic")
        if isinstance(topic, str) and topic:
            refs["current_topic"] = topic
        # 3. Trigger-reason scan — look for '@handle' or 'person:XYZ' or 'topic:XYZ'
        trig = self.state.trigger_reason or ""
        if isinstance(trig, str) and "primary_person" not in refs:
            for tok in trig.split():
                if tok.startswith("@") and len(tok) > 1:
                    refs["primary_person"] = tok
                    break
                if tok.startswith("person:") and len(tok) > 7:
                    refs["primary_person"] = tok[7:]
                    break
        if isinstance(trig, str) and "current_topic" not in refs:
            low = trig.lower()
            marker = "topic:"
            if marker in low:
                i = low.index(marker) + len(marker)
                tail = trig[i:].strip().split()[0] if i < len(trig) else ""
                if tail:
                    refs["current_topic"] = tail
        # No entity/topic detected → skip dispatch (cheap path for inner chains)
        if not refs:
            return
        self.state.entity_refs = refs
        self.state.needs_outer = dict(refs)
        self._dispatch_outer_fetch()

    def _emit_meta_outer_reward(self) -> None:
        """Emit META_OUTER_REWARD bus msg at chain conclude if outer was used.

        Reward shape: +config.reward_weight on successful use, -0.5*weight
        on hint-emitted-but-timeout (debit for wasted fetch). Wrapped in
        try/except — reward emission never breaks chain conclude.
        """
        try:
            if self._send_queue is None:
                return
            if not self._outer_enabled():
                return
            cid = self.state.chain_id
            if cid is None or cid < 0:
                return
            used = bool(self.state.outer_context_used)
            had_hint = bool(self.state.needs_outer)
            if not used and not had_hint:
                return  # no signal to emit
            cfg_w = 0.0
            try:
                cfg_w = float(self._outer_reader.config.reward_weight)
            except Exception as _swallow_exc:
                swallow_warn('[logic.meta_reasoning] MetaReasoningEngine._emit_meta_outer_reward: cfg_w = float(self._outer_reader.config.reward_weight)', _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._emit_meta_outer_reward.line2532', throttle=100)
            if cfg_w <= 0.0:
                return  # observe-only mode
            delta = cfg_w if used else (-0.5 * cfg_w)
            reason = "outer_used" if used else "outer_hint_no_data"
            from titan_plugin.bus import make_msg
            self._send_queue.put(make_msg(
                bus.META_OUTER_REWARD, "spirit", "chain_iql",
                {"chain_id": cid, "reward_delta": delta, "reason": reason}
            ))
        except Exception as e:
            swallow_warn('[MetaOuter] reward emit err', e,
                         key="logic.meta_reasoning.reward_emit_err", throttle=100)

    def _conclude_chain(self, state_132d, chain_archive, meta_wisdom, autoencoder):
        """Conclude meta-chain: compute reward, archive, possibly save wisdom."""
        # Track unique-prim count for adaptive epsilon-greedy escape mechanism
        prims_in_chain = [str(step).split(".")[0] for step in self.state.chain]
        unique_in_chain = len(set(prims_in_chain)) if prims_in_chain else 0
        self._recent_chain_uniques.append(unique_in_chain)

        # ── Task 4 P1: in-engine monoculture control loop ──
        # Per discussion 2026-04-12: contracts declare invariants and fire on
        # genesis seals (~4-5x/day, way too sparse for meta-reasoning control);
        # control loops live with the controlled subsystem and run at the
        # cadence of the controlled signal (per chain). This check fires
        # diversity_pressure responsively when monoculture persists, with a
        # cooldown = decay duration so we never stack pressure. The contract
        # `monoculture_detector` keeps firing on genesis seals as a state-
        # invariant safety net — both coexist permanently.
        try:
            self._inengine_monoculture_check()
        except Exception as _ime_err:
            logger.warning("[META] In-engine monoculture check failed: %s", _ime_err)

        # TUNING-012 v2 Sub-phase C (R3): decay diversity pressure per chain.
        # Linear decay: each chain reduces magnitude by initial/decay_chains
        # so by the end of decay_chains the bias is fully cleared.
        if self._diversity_pressure_remaining > 0:
            self._diversity_pressure_remaining -= 1
            if self._diversity_pressure_remaining == 0:
                self._primitive_bias = np.zeros(NUM_META_ACTIONS, dtype=np.float32)
                logger.info(
                    "[META] Diversity pressure on %s decayed to zero",
                    self._diversity_pressure_target,
                )
                self._diversity_pressure_target = ""
            elif self._diversity_pressure_initial_decay > 0:
                # Linear decay of magnitude
                _frac = (self._diversity_pressure_remaining
                         / self._diversity_pressure_initial_decay)
                _idx = META_PRIMITIVES.index(self._diversity_pressure_target) \
                    if self._diversity_pressure_target in META_PRIMITIVES else -1
                if _idx >= 0:
                    self._primitive_bias[_idx] = (
                        -self._diversity_pressure_initial_magnitude * _frac)

        terminal_reward = self._compute_meta_reward()
        # ── P9: Q-i reward blending — layer r_grounded on top of existing
        # legacy+compound blend. Failsafe-first: if META-CGN is disabled or
        # primitives empty, w_grounded = 0 → terminal_reward unchanged.
        if self._meta_cgn is not None and prims_in_chain:
            try:
                final_domain_p9 = (
                    self.state.formulate_output.get("domain", "general")
                    if isinstance(self.state.formulate_output, dict)
                    else "general"
                )
                # High-disagreement flag: pending cache from rerank time
                disagreement_for_chain = (
                    self._meta_cgn._pending_disagreement_by_chain.get(
                        self.state.chain_id, 0.0))
                high_disagreement = float(disagreement_for_chain) > 0.3
                r_grounded, _p9_rows = self._meta_cgn.compute_grounded_reward(
                    primitives=list(prims_in_chain),
                    domain=final_domain_p9,
                    high_disagreement_this_chain=high_disagreement,
                )
                w_leg, w_comp, w_grd = self._meta_cgn.compute_blend_weights(
                    current_domain=final_domain_p9)
                if w_grd > 0.001:
                    # Scale existing terminal_reward (which embodies
                    # legacy+compound already) by (w_leg + w_comp), add
                    # w_grd * r_grounded. This preserves the internal
                    # legacy/compound ratio from self._compound_blend_alpha
                    # while introducing r_grounded as a third signal.
                    legacy_plus_compound_weight = max(0.001, w_leg + w_comp)
                    blended_terminal = (
                        legacy_plus_compound_weight * terminal_reward +
                        w_grd * r_grounded
                    )
                    # E5: audit trail
                    self._meta_cgn.log_blend_weights(
                        chain_id=self.state.chain_id,
                        domain=final_domain_p9,
                        w_leg=w_leg, w_comp=w_comp, w_grd=w_grd,
                        r_leg=terminal_reward, r_comp=terminal_reward,
                        r_grd=r_grounded,
                        terminal=blended_terminal,
                    )
                    terminal_reward = max(0.0, min(1.0, blended_terminal))
            except Exception as _p9_err:
                logger.warning("[META] P9 grounded-reward blending failed: "
                               "%s (chain continues, reward unchanged)",
                               _p9_err)
        self.buffer.update_last_reward(terminal_reward)
        self.buffer.record(
            [0.0] * META_POLICY_INPUT_DIM, 0, 0, terminal_reward, done=True)

        # ── TUNING-012 v2 Sub-phase B: record chain outcome for IQL training ──
        # Refine the task embedding with the actual final domain (was "pending"
        # at chain start since FORMULATE.define hadn't run yet).
        if self._chain_iql and self._chain_iql.enabled:
            try:
                from titan_plugin.logic.task_embedding import encode_task
                final_domain = (
                    self.state.formulate_output.get("domain", "general")
                    if isinstance(self.state.formulate_output, dict)
                    else "general"
                )
                refined_task_emb = encode_task(
                    domain=final_domain,
                    trigger_reason=self.state.trigger_reason,
                    state_vector=state_132d,
                    dim=self._chain_iql._task_dim,
                )
                # Normalize terminal reward to [0, 1] for task_success
                task_success = max(0.0, min(1.0, float(terminal_reward)))
                self._chain_iql.record_chain_outcome(
                    task_emb=refined_task_emb,
                    chain=list(self.state.chain),
                    task_success=task_success,
                    primitives=prims_in_chain,
                    domain=final_domain,
                    chain_id=self.state.chain_id,
                )
            except Exception as _ro_err:
                logger.warning("[META] Chain IQL record_chain_outcome failed: %s", _ro_err)

        # ── META-CGN Phase 1+2: transitions + grounding + HAOV evidence ──
        # Shadow-mode: observe chain outcomes, feed CGN worker's SharedValueNet
        # training, accumulate local primitive V(s) via EMA, push HAOV evidence
        # per-chain. Never influences chain selection in Phase 1/2 — enables
        # Phase 4+ graduation based on HAOV-confirmed rules.
        # Reward attribution: uniform across primitives in chain (Phase 1 choice;
        # position-weighted comes in Phase 6 per rFP §16).
        if self._meta_cgn is not None and prims_in_chain:
            try:
                task_success_mcgn = max(0.0, min(1.0, float(terminal_reward)))
                final_domain_mcgn = (
                    self.state.formulate_output.get("domain", "general")
                    if isinstance(self.state.formulate_output, dict)
                    else "general"
                )
                # Snapshot per-primitive V BEFORE this chain's update — lets
                # HAOV compare "what V was" vs "what reward arrived".
                per_prim_V_snapshot = {
                    p: self._meta_cgn._primitives[p].V
                    for p in self._meta_cgn._primitives
                }
                pop_avg_V = (sum(per_prim_V_snapshot.values()) /
                             max(1, len(per_prim_V_snapshot)))
                # Resolve dominance FIRST — populates _last_dominance_share
                # which _build_meta_cgn_ctx reads into the 30D state vector.
                dominant_primitive = self._resolve_dominant_primitive()
                # Build one context dict reused across primitives in this chain.
                mcgn_ctx = self._build_meta_cgn_ctx(
                    final_domain=final_domain_mcgn,
                    terminal_reward=terminal_reward,
                )
                # P7: EUREKA accelerator — asymmetric weights
                # trigger primitive: 5×, other in-chain primitives: 3×, else: 1×
                _eureka_fired = bool(self.state.eureka_fired)
                _eureka_trigger = self.state.eureka_trigger or ""
                for prim_id in prims_in_chain:
                    if prim_id not in META_PRIMITIVES:
                        continue
                    state_vec = self._meta_cgn.encode_state(prim_id, mcgn_ctx)
                    if _eureka_fired:
                        p_weight = 5.0 if prim_id == _eureka_trigger else 3.0
                    else:
                        p_weight = 1.0
                    self._meta_cgn.update_primitive_V(
                        primitive_id=prim_id,
                        quality=task_success_mcgn,
                        chain_id=self.state.chain_id,
                        weight=p_weight,
                        domain=final_domain_mcgn,   # P6 I3 per-domain V
                    )
                    self._meta_cgn.send_transition(
                        primitive_id=prim_id,
                        state_vec=state_vec,
                        reward=task_success_mcgn,
                        chain_id=self.state.chain_id,
                        metadata={
                            "chain_id": self.state.chain_id,
                            "domain": final_domain_mcgn,
                            "epoch": self.state.start_epoch,
                            "chain_length": len(prims_in_chain),
                        },
                    )
                    # Upgrade III peer publishing (audit 2026-04-23 Q2) —
                    # broadcast meta chain-outcome so emot_cgn + other peer
                    # consumers can learn from it. Rate-gated + informative
                    # filter inside emit_cross_insight.
                    if getattr(self._meta_cgn, "_cgn_client", None):
                        try:
                            self._meta_cgn._cgn_client.emit_cross_insight(
                                terminal_reward=task_success_mcgn,
                                ctx={
                                    "domain": final_domain_mcgn,
                                    "chain_length": len(prims_in_chain),
                                    "primitive": prim_id,
                                },
                            )
                        except Exception as _ci_err:
                            swallow_warn('[MetaReasoning] meta cross-insight emit failed', _ci_err,
                                         key="logic.meta_reasoning.meta_cross_insight_emit_failed", throttle=100)
                # Phase 2 — HAOV evidence per chain (one record, not per-prim)
                self._meta_cgn.observe_chain_evidence({
                    "ts": time.time(),
                    "chain_id": self.state.chain_id,
                    "primitives": list(prims_in_chain),
                    "quality": task_success_mcgn,
                    "domain": final_domain_mcgn,
                    "monoculture_share": mcgn_ctx.get("monoculture_share", 0.0),
                    "is_in_soar_impasse": mcgn_ctx.get("is_in_soar_impasse",
                                                        False),
                    "dominant_primitive": dominant_primitive,
                    "pop_avg_V": pop_avg_V,
                    "per_primitive_V": per_prim_V_snapshot,
                    # P7: EUREKA tags — unlock EUREKA-weighted HAOV observations
                    "eureka_fired": _eureka_fired,
                    "eureka_trigger": _eureka_trigger,
                })
                # Phase 4+5 per-chain hooks — graduation ramp, rollback check,
                # impasse detection, failsafe cooldown decrement.
                # Pass RAW reward — rollback detector is scale-invariant
                # (compares post-grad mean to baseline_mean − k·σ).
                self._meta_cgn.record_chain_outcome(task_success_mcgn)
                # Track last-concluded chain_id so late-arriving cross-system
                # reward signals (persona session quality, events teacher
                # outcomes, etc.) can correlate to the most recent chain via
                # get_last_chain_id().
                self._last_concluded_chain_id = int(self.state.chain_id)
                # COMPLETE-9 HAOV telemetry — log chain outcome for
                # signal↔chain correlation analysis (next-session
                # META-CGN-V2-HAOV-REFINEMENT will learn SIGNAL_TO_PRIMITIVE
                # quality_nudge values from this data).
                self._meta_cgn.log_haov_chain(
                    chain_id=self.state.chain_id,
                    primitives=list(prims_in_chain),
                    dominant=dominant_primitive,
                    terminal_reward=task_success_mcgn,
                    domain=final_domain_mcgn,
                )
                self._meta_cgn.evaluate_graduation()
                self._meta_cgn.check_impasse()
                self._meta_cgn.maybe_exit_impasse()
                # P8: close expired knowledge-request windows → inject winners
                try:
                    winners = self._meta_cgn.finalize_expired_requests()
                    for winner in winners:
                        rel = float(winner.get("_rank_score", 0.0))
                        topic = str(winner.get("topic", ""))
                        injected = self.inject_knowledge(
                            topic, winner, relevance=rel)
                        if injected:
                            self._meta_cgn.mark_helpful(
                                str(winner.get("source", "unknown")))
                except Exception as _p8_err:
                    swallow_warn('[META] P8 knowledge finalize error', _p8_err,
                                 key="logic.meta_reasoning.p8_knowledge_finalize_error", throttle=100)
            except Exception as _mcgn_err:
                logger.warning("[META] META-CGN hook failed: %s "
                               "(chain continues unaffected)", _mcgn_err)

        # ── EMOT-CGN chain evidence (Phase 1.6h cutover) ──
        # Emit EMOT_CHAIN_EVIDENCE to worker — it owns β-posterior update,
        # neuromod EMA, and HAOV accumulation. Worker reads dom_end from
        # its own clusterer state (we don't need to query it). dom_start
        # snapshot was taken in _start_chain from shm.
        # Failsafe: emit errors don't break chain flow.
        try:
            def _share(chain_prims, target):
                if not chain_prims:
                    return 0.0
                hits = sum(1 for p in chain_prims if p == target)
                return float(hits) / float(len(chain_prims))

            # Read worker's current dominant emotion from shm for the
            # dom_end snapshot (used by H5 strategy_shift flag).
            dom_end = getattr(self, "_emot_dom_at_chain_start", "FLOW")
            try:
                from titan_plugin.logic.emot_shm_protocol import ShmEmotReader
                from titan_plugin.logic.emotion_cluster import EMOT_PRIMITIVES
                _st = ShmEmotReader().read_state()
                if _st is not None:
                    dom_end = EMOT_PRIMITIVES[_st["dominant_idx"]]
            except Exception as _swallow_exc:
                swallow_warn('[logic.meta_reasoning] MetaReasoningEngine._conclude_chain: from titan_plugin.logic.emot_shm_protocol import ShmEmotR...', _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._conclude_chain.line2832', throttle=100)

            # Neuromods: prefer `self._last_neuromods_dict` (real per-tick
            # reading stashed in tick()) over mcgn_ctx (which pulls from
            # _subsystem_cache — currently stubbed to 0.5 for all 6). Falls
            # back to mcgn_ctx / 0.5 if the stash is somehow absent.
            _nm_live = getattr(self, "_last_neuromods_dict", None) or {}
            def _nmval(k):
                return float(_nm_live.get(k, mcgn_ctx.get(k, 0.5)))
            # Trajectory 2D: per consciousness.py:46, state_132d = Inner 65D +
            # Outer 65D + curvature + density. [130:132] are the two global
            # meta-scalars — curvature (rate-of-change of inner-space volume)
            # and density (compactness). These ARE Titan's phase-space
            # trajectory signal; feeding real values to HDBSCAN.
            _traj_2d = (
                [float(state_132d[130]), float(state_132d[131])]
                if state_132d is not None and len(state_132d) >= 132
                else [0.0, 0.0]
            )
            _emot_ctx = {
                # Neuromod levels — worker's update_neuromod_ema is called
                # internally on EMOT_CHAIN_EVIDENCE payload.
                "DA": _nmval("DA"),
                "5HT": _nmval("5HT"),
                "NE": _nmval("NE"),
                "ACh": _nmval("ACh"),
                "Endorphin": _nmval("Endorphin"),
                "GABA": _nmval("GABA"),
                # Native trajectory (rFP §19.2 trajectory_2d slot).
                "trajectory_2d": _traj_2d,
                # Live 30D digital-body space topology (rFP §19.2
                # space_topology_30d slot) — outer_lower topology 10D +
                # inner_lower topology 10D + whole (unified spirit) 10D.
                # Assembled by spirit_worker._attach_emot_producer_ctx just
                # before each meta_engine.tick() call; falls back to zeros
                # if the attr is absent (e.g. tests or boot before first
                # body tick).
                "space_topology_30d": (
                    list(getattr(self, "_last_topology_30d", None) or [])
                    or [0.0] * 30
                )[:30],
                # Live 11D per-NS-program urgencies (rFP §19.2 ns_urgencies
                # slot) — ordered REFLEX, FOCUS, INTUITION, IMPULSE,
                # METABOLISM, CREATIVITY, CURIOSITY, EMPATHY, REFLECTION,
                # INSPIRATION, VIGILANCE per emot_bundle_protocol.NS_PROGRAMS.
                # Same convention: stashed via _attach_emot_producer_ctx,
                # zero-fallback if the hook hasn't run yet.
                "ns_urgencies_11d": (
                    list(getattr(self, "_last_ns_urgencies_11d", None) or [])
                    or [0.0] * 11
                )[:11],
                # Live 6D sphere-clock phases (rFP §19.2 pi_phase slot,
                # schema v2 2026-04-21). Order: inner_body, outer_body,
                # inner_mind, outer_mind, inner_spirit, outer_spirit.
                # Trinity × Inner/Outer symmetry preserved.
                "pi_phase_6d": (
                    list(getattr(self, "_last_pi_phase_6d", None) or [])
                    or [0.0] * 6
                )[:6],
                "monoculture_share": mcgn_ctx.get("monoculture_share", 0.0),
                "incomplete": float(terminal_reward) < 0.25,
                "spirit_self_share": _share(prims_in_chain, "SPIRIT_SELF"),
                "introspect_share": _share(prims_in_chain, "INTROSPECT"),
                "strategy_shift": (
                    getattr(self, "_emot_dom_at_chain_start", dom_end)
                    != dom_end),
                "knowledge_acq_rate": mcgn_ctx.get("knowledge_acq_rate", 0.0),
            }
            from titan_plugin.bus import emit_emot_chain_evidence
            if self._send_queue is not None:
                emit_emot_chain_evidence(
                    self._send_queue, src="spirit",
                    chain_id=self.state.chain_id,
                    dominant_at_start=getattr(
                        self, "_emot_dom_at_chain_start", dom_end),
                    dominant_at_end=dom_end,
                    terminal_reward=float(terminal_reward),
                    ctx=_emot_ctx)
        except Exception as _emot_err:
            swallow_warn('[META] EMOT_CHAIN_EVIDENCE emit error', _emot_err,
                         key="logic.meta_reasoning.emot_chain_evidence_emit_error", throttle=100)

        # Update baseline EMA
        self._baseline_confidence = (
            0.9 * self._baseline_confidence + 0.1 * self.state.confidence)

        # Archive the meta-chain
        if chain_archive:
            chain_archive.record_meta_chain(
                chain_sequence=self.state.chain,
                confidence=self.state.confidence,
                outcome_score=terminal_reward,
                problem_type=self.state.formulate_output.get("domain", "general"),
                strategy_label=self.state.trigger_reason,
                observation_snapshot=list(state_132d[:132]),
                epoch_id=self.state.start_epoch,
            )

        # Auto distill_save if high quality
        if terminal_reward > 0.5 and meta_wisdom and self.state.formulate_output:
            prob_emb = None
            if autoencoder and autoencoder.is_trained:
                prob_emb = autoencoder.encode(state_132d[:132])
            meta_wisdom.store_wisdom(
                problem_pattern=self.state.formulate_output.get("problem_template",
                                                                self.state.trigger_reason),
                strategy_sequence=self.state.chain,
                outcome_score=terminal_reward,
                problem_embedding=prob_emb,
            )
            self._total_wisdom_saved += 1
            logger.info("[META] Wisdom saved: reward=%.3f, chain=%s",
                        terminal_reward, self.state.chain[:5])
            # META-CGN Producer #15: meta_wisdom.crystallized (rFP v3 § 12 row 15).
            # Queue for spirit_worker drain. Fires on NEW chain signature OR
            # repeated signature with conf≥0.9 (per rFP secondary gate — keeps
            # the HIGH-confidence repeats flowing even after signature is seen).
            try:
                _p15_domain = self.state.formulate_output.get("domain", "general")
                _p15_sig = f"{_p15_domain}:{hash(tuple(self.state.chain))}"
                self._pending_cgn_wisdom_events.append({
                    "signature": _p15_sig,
                    "confidence": float(self.state.confidence),
                    "reward": float(terminal_reward),
                    "domain": _p15_domain,
                })
            except Exception as _p15_q_err:
                logger.warning(
                    "[META-CGN] Failed to queue crystallized event for producer #15 drain: %s "
                    "(chain=%s domain=%s) — this emission will be missed",
                    _p15_q_err, self.state.chain[:3], self.state.formulate_output.get("domain"))

        duration = time.time() - self.state.start_time
        logger.info("[META] Chain #%d concluded: reward=%.3f, steps=%d, duration=%.1fs, trigger=%s",
                    self._total_meta_chains, terminal_reward, len(self.state.chain),
                    duration, self.state.trigger_reason)

        # Close wisdom reuse loop: if this chain loaded prior wisdom via
        # FORMULATE.load_wisdom, feed the chain outcome back so confidence
        # and crystallization can update. Without this, wisdom accumulates
        # with times_reused=0 forever (dead-wiring bug found 2026-04-16).
        if meta_wisdom and self.state.formulate_output:
            prior = self.state.formulate_output.get("prior_strategy")
            if prior and isinstance(prior, dict) and "id" in prior:
                try:
                    meta_wisdom.record_reuse(
                        prior["id"],
                        success=(terminal_reward > 0.3))
                except Exception:
                    pass  # Non-fatal — reuse tracking is observability

        # Strategy collapse detection: every 100 chains, audit distribution
        if self._total_meta_chains > 0 and self._total_meta_chains % 100 == 0:
            prim_counts: dict = {}
            for a in self.buffer._actions[-500:]:
                name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
                prim_counts[name] = prim_counts.get(name, 0) + 1
            total = sum(prim_counts.values())
            if total > 0:
                dominant_name, dominant_n = max(prim_counts.items(), key=lambda x: x[1])
                dom_pct = dominant_n / total * 100
                dist_str = " ".join(
                    f"{k}={v/total*100:.0f}%"
                    for k, v in sorted(prim_counts.items(), key=lambda x: -x[1])
                )
                if dom_pct > 75:
                    logger.warning(
                        "[META] STRATEGY COLLAPSE @chain%d: %s dominates %.1f%% of last %d steps | %s",
                        self._total_meta_chains, dominant_name, dom_pct, total, dist_str,
                    )
                else:
                    logger.info(
                        "[META] strategy dist @chain%d (last %d steps): %s",
                        self._total_meta_chains, total, dist_str,
                    )

        # ── TUNING-012 v2 Sub-phase C (R1): chain outcome fields for contracts ──
        # The cognitive contracts (strategy_evolution, monoculture_detector,
        # abstract_pattern_extraction) AGGREGATE these fields from meta-fork
        # blocks. Without them, the contracts have nothing to read.
        chain_template = "→".join(prims_in_chain) if prims_in_chain else ""
        # Normalize terminal reward → task_success [0, 1]
        chain_task_success = max(0.0, min(1.0, float(terminal_reward)))
        chain_domain = (
            self.state.formulate_output.get("domain", "general")
            if isinstance(self.state.formulate_output, dict)
            else "general"
        )

        result = {
            "action": "CONCLUDE", "reward": round(terminal_reward, 4),
            "chain_id": self.state.chain_id,
            "chain_length": len(self.state.chain),
            "confidence": self.state.confidence,
            "duration_s": round(duration, 1),
            # Sub-phase C contract data
            "chain_template": chain_template,
            "task_success": round(chain_task_success, 4),
            "primitives_used": prims_in_chain,
            "domain": chain_domain,
            "unique_primitives": unique_in_chain,
            "knowledge_injected": self.state.knowledge_injected,
            "impasse_detected": self.state.impasse_detected,
        }

        # D.2: curiosity momentum — successful chains with knowledge injection
        # lower the impasse threshold for the next window (emergent curiosity).
        self.record_curiosity_success(terminal_reward)

        # ── Meta-Reasoning Teacher: META_CHAIN_COMPLETE emission ──
        # Per rFP_titan_meta_reasoning_teacher §3.3. Emitted unconditionally
        # after all existing chain-conclude hooks — teacher worker decides
        # whether to critique (sampling + rate cap). dst="all". No subscriber
        # when teacher is disabled → silent drop per documented behavior.
        #
        # rFP_meta_teacher_v2 Phase A (2026-04-24): outer_summary +
        # step_arguments now ride along so the teacher sees what each step
        # was reasoning ABOUT, not just which primitives were used. Both
        # optional in the payload; graceful fallback when absent.
        try:
            from titan_plugin.bus import emit_meta_chain_complete
            from titan_plugin.logic.meta_teacher_content import (
                build_teacher_outer_summary, build_step_arguments,
            )
            if self._send_queue is not None and prims_in_chain:
                transitions = [
                    (prims_in_chain[i], prims_in_chain[i + 1])
                    for i in range(len(prims_in_chain) - 1)
                ]
                haov_hid = None
                try:
                    if self._meta_cgn is not None and hasattr(
                            self._meta_cgn, "_last_haov_hypothesis_id"):
                        haov_hid = getattr(
                            self._meta_cgn, "_last_haov_hypothesis_id", None)
                except Exception as _swallow_exc:
                    swallow_warn('[logic.meta_reasoning] MetaReasoningEngine._conclude_chain: if self._meta_cgn is not None and hasattr(self._meta_cgn,...', _swallow_exc,
                                 key='logic.meta_reasoning.MetaReasoningEngine._conclude_chain.line3068', throttle=100)
                _mcgn_ctx_local = locals().get("mcgn_ctx") or {}
                # Phase A: build content helpers — defensive try/except so
                # a helper bug never blocks teacher emission.
                try:
                    outer_summary = build_teacher_outer_summary(self.state)
                except Exception as _os_err:
                    swallow_warn('[META] build_teacher_outer_summary failed', _os_err,
                                 key="logic.meta_reasoning.build_teacher_outer_summary_failed", throttle=100)
                    outer_summary = None
                try:
                    step_arguments = build_step_arguments(self.state)
                except Exception as _sa_err:
                    swallow_warn('[META] build_step_arguments failed', _sa_err,
                                 key="logic.meta_reasoning.build_step_arguments_failed", throttle=100)
                    step_arguments = []
                emit_meta_chain_complete(
                    self._send_queue, src="spirit",
                    chain_id=self.state.chain_id,
                    primitives_used=list(prims_in_chain),
                    primitive_transitions=transitions,
                    chain_length=len(self.state.chain),
                    domain=chain_domain,
                    task_success=chain_task_success,
                    chain_iql_confidence=float(self.state.confidence),
                    start_epoch=int(self.state.start_epoch),
                    conclude_epoch=int(time.time()),
                    context_summary={
                        "dominant_emotion": str(
                            getattr(self, "_emot_dom_at_chain_start", "FLOW")),
                        "chi_remaining": float(
                            _mcgn_ctx_local.get("chi_remaining", 0.0)),
                        "impasse_state": (
                            "detected" if self.state.impasse_detected else "none"),
                        "trigger_reason": str(self.state.trigger_reason or ""),
                        "knowledge_injected": bool(self.state.knowledge_injected),
                    },
                    haov_hypothesis_id=haov_hid,
                    final_observation={
                        "chain_template": chain_template,
                        "unique_primitives": unique_in_chain,
                    },
                    outer_summary=outer_summary,
                    step_arguments=step_arguments,
                )
        except Exception as _mtc_err:
            swallow_warn('[META] META_CHAIN_COMPLETE emit failed', _mtc_err,
                         key="logic.meta_reasoning.meta_chain_complete_emit_failed", throttle=100)

        # rFP_titan_meta_outer_layer — emit META_OUTER_REWARD before reset
        # so chain_id + outer_context_used are still on self.state.
        self._emit_meta_outer_reward()
        # Observability — include outer-layer stats in the returned result
        # so spirit_worker's brain-log line surfaces them to the user.
        result["outer_context_used"] = bool(self.state.outer_context_used)
        if self.state.needs_outer:
            result["outer_hint_entities"] = list(self.state.entity_refs.keys())

        self.state = MetaChainState()  # Reset
        self._reset_outer_state()
        return result

    # ── Phase D.1 — External reward injection (META_LANGUAGE loop) ────
    def get_last_chain_id(self) -> int:
        """Most-recently-concluded chain_id, for correlating late-arriving
        external reward signals (e.g. persona session quality events that
        fire after a chain already concluded). Returns -1 if no chain
        concluded yet or chain_id unavailable."""
        try:
            return int(getattr(self, "_last_concluded_chain_id", -1))
        except Exception:
            return -1

    def add_external_reward(
        self, chain_id: int, external_reward: float
    ) -> bool:
        """Apply an external reward to a previously-concluded chain.

        Called by spirit_worker's META_LANGUAGE_REWARD handler after
        language_worker measures vocab/grounding delta from a chain's
        downstream effects. Routes to chain_iql.apply_external_reward
        using the DNA-tuned blend alpha.

        Args:
            chain_id: the chain to correlate (from the result dict).
            external_reward: reward in [0, 1] (will be clipped by IQL).

        Returns:
            True if the chain was still in the buffer and got updated,
            False if it was already evicted (late-drop; counted in
            chain_iql stats).
        """
        if not (self._chain_iql and self._chain_iql.enabled):
            return False
        applied = self._chain_iql.apply_external_reward(
            chain_id=int(chain_id),
            external_reward=float(external_reward),
            alpha=self._external_reward_blend_alpha,
        )
        if applied:
            logger.info(
                "[META] External reward applied: chain_id=%d reward=%.3f alpha=%.2f",
                chain_id, external_reward, self._external_reward_blend_alpha,
            )
        else:
            logger.debug(
                "[META] External reward late-drop: chain_id=%d (already evicted)",
                chain_id,
            )
        return applied

    # ── Phase D.2 — SOAR Impasse Detection + Knowledge Research Loop ────

    def detect_chain_impasse(self) -> dict | None:
        """Detect impasse within the current chain (declining step rewards).

        Returns impasse dict with topic + urgency if detected, None otherwise.
        Called by spirit_worker after each meta step to check if the chain
        is stuck and needs external knowledge.

        Gating:
        - Requires soar_impasse_threshold_consec+ declining steps
        - Respects per-hour rate limits (internal + external separately)
        - Respects per-concept cooldown
        - Curiosity momentum lowers threshold after successful research
        """
        if not self.state.chain or self.state.impasse_detected:
            return None

        # ── Check 1: Primitive-repeat impasse (TUNING-017 / SOAR rFP) ──
        # Detects F→F→F as "stuck in a loop" — independent of reward trends.
        # A chain can have stable rewards from passive signals (spirit drift)
        # while being cognitively stuck in the same primitive.
        repeat_thresh = int(self._soar_config.get("repeat_threshold", 3))
        if len(self.state.chain) >= repeat_thresh:
            _tail = self.state.chain[-repeat_thresh:]
            if len(set(_tail)) == 1:
                # All same primitive — repeat_stuck impasse
                _repeated = _tail[0]
                _repeat_count = 0
                for _p in reversed(self.state.chain):
                    if _p == _repeated:
                        _repeat_count += 1
                    else:
                        break
                _urgency = float(self._soar_config.get("repeat_urgency", 0.8))
                _topic = f"cognitive diversity breaking {_repeated.lower()} repetition thinking strategies"

                self.state.impasse_detected = True
                self.state.impasse_topic = _topic
                # Track repeat impasses for META-CGN learning
                if not hasattr(self, "_repeat_impasse_count"):
                    self._repeat_impasse_count = 0
                self._repeat_impasse_count += 1
                if not hasattr(self, "_repeat_impasse_primitives"):
                    self._repeat_impasse_primitives = {}
                self._repeat_impasse_primitives[_repeated] = (
                    self._repeat_impasse_primitives.get(_repeated, 0) + 1)

                logger.info(
                    "[META] SOAR repeat impasse: primitive=%s count=%d "
                    "urgency=%.2f chain_step=%d (lifetime_repeat_impasses=%d)",
                    _repeated, _repeat_count, _urgency,
                    len(self.state.chain), self._repeat_impasse_count)

                return {
                    "type": "repeat_stuck",
                    "topic": _topic,
                    "urgency": _urgency,
                    "repeated_primitive": _repeated,
                    "repeat_count": _repeat_count,
                    "chain_id": getattr(self.state, "chain_id", 0),
                    "chain_step": len(self.state.chain),
                }

        # ── Check 2: Declining-reward impasse (original D.2) ──
        rewards = self.state.step_rewards
        threshold = self._soar_config["threshold_consec"]
        # Curiosity momentum: lower threshold by bonus after successful research
        if time.time() < self._soar_curiosity_until:
            threshold = max(2, threshold - self._soar_config["curiosity_bonus"])

        if len(rewards) < threshold:
            return None

        # Check for consecutive declining rewards
        recent = rewards[-threshold:]
        declining = all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))
        if not declining:
            return None

        # Rate limit: prune old requests (>1 hour ago)
        now = time.time()
        self._soar_requests_this_hour = [
            ts for ts in self._soar_requests_this_hour if now - ts < 3600]
        total_requests = len(self._soar_requests_this_hour)
        max_total = (self._soar_config["max_internal_per_hour"]
                     + self._soar_config["max_external_per_hour"])
        if total_requests >= max_total:
            return None

        # Extract topic from chain context. Must produce a semantically-meaningful
        # conceptual query that SearXNG can actually return results for.
        # Prior bug (fixed 2026-04-12): fell back to primitive.submode names like
        # "FORMULATE.load_wisdom" which SearXNG correctly returns 0 results for
        # — they're Titan-internal API names, not world topics. 60+ knowledge
        # requests were silently producing "no content" during verification.
        # Fix: 3-tier fallback to conceptual queries, returning None if none applies.
        topic = ""

        # Tier 1 — FORMULATE.define output (best signal when available).
        # Note: formulate_output["domain"] is always a Titan-internal label
        # (inner_spirit / outer_perception / outer_spirit) from _prim_formulate,
        # so we skip it here and let Tier 1b map it to a conceptual query.
        if isinstance(self.state.formulate_output, dict):
            topic = self.state.formulate_output.get("topic",
                    self.state.formulate_output.get("problem", ""))

        # Tier 1b — map internal domain label to conceptual query.
        # Avoids shipping internal names like "outer_spirit" to SearXNG.
        if not topic and isinstance(self.state.formulate_output, dict):
            domain_label = self.state.formulate_output.get("domain", "")
            DOMAIN_TO_QUERY = {
                "inner_spirit": "emotional regulation self-awareness cognitive strategies",
                "outer_perception": "sensory perception cognitive processing strategies",
                "outer_spirit": "social emotional interaction cognitive strategies",
            }
            topic = DOMAIN_TO_QUERY.get(domain_label, "")

        # Tier 2 — map trigger_reason to conceptual query
        if not topic:
            trigger = (self.state.trigger_reason or "").lower()
            TRIGGER_TO_QUERY = {
                "low_commit_rate": "cognitive reasoning improving decision confidence",
                "high_reflection": "metacognition self-reflection strategies",
                "high_acetylcholine": "cognitive focus attention strategies",
                "stuck_pattern": "breaking cognitive impasse techniques",
                "social_feedback": "social cognition interaction strategies",
                "chi_surplus": "mental energy regulation strategies",
                "novelty_surge": "novelty detection cognitive adaptation",
            }
            for key, query in TRIGGER_TO_QUERY.items():
                if key in trigger:
                    topic = query
                    break

        # Tier 3 — map dominant primitive to conceptual query
        # This avoids "FORMULATE.load_wisdom" → "problem formulation cognitive strategies"
        if not topic and self.state.chain:
            last_step = str(self.state.chain[-1])
            primitive = last_step.split(".")[0] if "." in last_step else last_step
            PRIMITIVE_TO_QUERY = {
                "FORMULATE": "problem formulation cognitive strategies",
                "RECALL": "memory retrieval strategies",
                "HYPOTHESIZE": "hypothesis generation critical thinking",
                "DELEGATE": "task decomposition delegation strategies",
                "SYNTHESIZE": "information synthesis combining knowledge",
                "EVALUATE": "evaluation decision quality strategies",
                "BREAK": "cognitive recovery impasse strategies",
                "SPIRIT_SELF": "self-regulation mindfulness cognition",
                "INTROSPECT": "metacognition self-reflection strategies",
            }
            topic = PRIMITIVE_TO_QUERY.get(primitive, "")

        # Safety: no valid topic — don't send garbage query to SearXNG
        if not topic:
            return None

        # Per-concept cooldown
        cooldown_s = self._soar_config["concept_cooldown_s"]
        last_req = self._soar_concept_cooldowns.get(topic, 0)
        if topic and now - last_req < cooldown_s:
            return None

        # Determine urgency based on impasse type
        avg_recent = sum(recent) / len(recent) if recent else 0
        if avg_recent <= 0.01:
            urgency = self._soar_config["urgency_stuck"]
            impasse_type = "stuck"
        elif declining and recent[-1] < recent[0] * 0.5:
            urgency = self._soar_config["urgency_declining"]
            impasse_type = "declining"
        else:
            urgency = self._soar_config["urgency_plateau"]
            impasse_type = "plateau"

        self.state.impasse_detected = True
        self.state.impasse_topic = topic
        self._soar_requests_this_hour.append(now)
        self._soar_concept_cooldowns[topic] = now

        logger.info(
            "[META] SOAR impasse detected: type=%s topic='%s' urgency=%.2f "
            "declining_steps=%d threshold=%d chain_step=%d",
            impasse_type, (topic or "?")[:40], urgency,
            threshold, self._soar_config["threshold_consec"],
            len(self.state.chain),
        )

        return {
            "type": impasse_type,
            "topic": topic,
            "urgency": urgency,
            "chain_id": self.state.chain_id,
            "chain_step": len(self.state.chain),
            "recent_rewards": recent,
        }

    def inject_knowledge(self, topic: str, knowledge: dict,
                         relevance: float = 0.5) -> bool:
        """Inject externally-acquired knowledge into the chain state.

        Called by spirit_worker when CGN_KNOWLEDGE_RESP arrives for an
        impasse we triggered. The knowledge enriches the formulate_output
        so subsequent chain steps can use it.

        Returns True if injected (chain still active), False if chain
        already concluded.
        """
        if not self.state.chain or self.state.knowledge_injected:
            return False

        threshold = self._soar_config["relevance_threshold"]
        if relevance < threshold:
            logger.info("[META] Knowledge relevance too low (%.3f < %.3f), skipping injection",
                        relevance, threshold)
            return False

        # Enrich formulate_output with knowledge context
        if not isinstance(self.state.formulate_output, dict):
            self.state.formulate_output = {}
        self.state.formulate_output["injected_knowledge"] = {
            "topic": topic,
            "summary": knowledge.get("summary", ""),
            "confidence": knowledge.get("confidence", 0),
            "source": knowledge.get("source", "unknown"),
        }
        self.state.knowledge_injected = True

        # Curiosity momentum: if we get here, track for potential reward
        self._soar_last_successful_topic = topic

        logger.info(
            "[META] Knowledge injected: topic='%s' relevance=%.3f source=%s",
            (topic or "?")[:40], relevance, knowledge.get("source", "?"),
        )
        return True

    def record_curiosity_success(self, chain_reward: float):
        """Called at chain conclusion if knowledge was injected.

        If the chain that received knowledge had above-average reward,
        activate curiosity momentum (lower impasse threshold for a window).
        """
        if not self.state.knowledge_injected:
            return
        # Compare to recent average
        avg = sum(self.state.step_rewards) / max(len(self.state.step_rewards), 1)
        if chain_reward > avg * 1.1:  # 10% above chain average
            self._soar_curiosity_until = (
                time.time() + self._soar_config["curiosity_duration_s"])
            logger.info(
                "[META] Curiosity momentum activated: topic='%s' "
                "reward=%.3f > avg=%.3f, momentum for %ds",
                self._soar_last_successful_topic[:30],
                chain_reward, avg,
                int(self._soar_config["curiosity_duration_s"]),
            )

    # ── Private: Primitive Execution ──────────────────────────────

    def _execute(self, prim, sub, sv, nm, reasoning_engine,
                 chain_archive, meta_wisdom, ex_mem, autoencoder):
        if prim == "FORMULATE":
            result = self._prim_formulate(sub, sv, nm, meta_wisdom, autoencoder)
            # rFP_titan_meta_outer_layer Bridge 2 — detect entity/topic refs
            # from the formulated intent; populate entity_refs + needs_outer;
            # dispatch composed-recall future so later primitives can read it.
            self._post_formulate_detect_entities()
            return result
        elif prim == "RECALL":
            return self._prim_recall(sub, chain_archive, meta_wisdom, ex_mem)
        elif prim == "HYPOTHESIZE":
            return self._prim_hypothesize(sub, nm)
        elif prim == "DELEGATE":
            return self._prim_delegate(sub, reasoning_engine)
        elif prim == "SYNTHESIZE":
            return self._prim_synthesize(sub, meta_wisdom, autoencoder, sv)
        elif prim == "EVALUATE":
            return self._prim_evaluate(sub, nm)
        elif prim == "BREAK":
            return self._prim_break(sub, sv, reasoning_engine)
        elif prim == "SPIRIT_SELF":
            return self._prim_spirit_self(sub, nm)
        elif prim == "INTROSPECT":
            return self._prim_introspect(sub, sv, nm)
        return {"primitive": prim, "error": "unknown"}

    def _prim_formulate(self, sub, sv, nm, meta_wisdom, autoencoder):
        """Define or refine the problem being investigated."""
        sv_arr = np.array(sv[:132], dtype=np.float32)

        if sub == "define":
            # Find top anomalous dimensions (deviation from EMA)
            deviation = np.abs(sv_arr - self._ema_state)
            top_dims = np.argsort(deviation)[-5:][::-1].tolist()
            # Classify domain from anomalous dimension ranges
            domain = "general"
            avg_dim = np.mean(top_dims)
            if avg_dim < 20:
                domain = "body_mind"
            elif avg_dim < 65:
                domain = "inner_spirit"
            elif avg_dim < 85:
                domain = "outer_perception"
            else:
                domain = "outer_spirit"

            difficulty = float(np.mean(deviation[top_dims]))
            template = f"{domain} anomaly: dims {top_dims}, deviation={difficulty:.3f}"

            self.state.formulate_output = {
                "problem_template": template,
                "domain": domain,
                "anomalous_dims": top_dims,
                "difficulty": difficulty,
                "trigger": self.state.trigger_reason,
            }
            return {"primitive": "FORMULATE", "sub_mode": "define",
                    "domain": domain, "difficulty": round(difficulty, 4),
                    "anomalous_dims": top_dims, "confidence": 0.5}

        elif sub == "refine":
            # Narrow problem based on recall data
            if self.state.recalled_data:
                best = self.state.recalled_data.get("best_match")
                if best:
                    self.state.formulate_output["prior_strategy"] = best
                    self.state.formulate_output["refined"] = True
            return {"primitive": "FORMULATE", "sub_mode": "refine",
                    "refined": bool(self.state.recalled_data),
                    "confidence": min(0.6, self.state.confidence + 0.05)}

        elif sub == "load_wisdom":
            wisdom_found = False
            if meta_wisdom and self.state.formulate_output:
                template = self.state.formulate_output.get("problem_template", "")
                results = meta_wisdom.query_by_pattern(template, min_confidence=0.4)
                if not results and autoencoder and autoencoder.is_trained:
                    emb = autoencoder.encode(sv[:132])
                    results = meta_wisdom.query_by_embedding(emb, min_confidence=0.4)
                if results:
                    self.state.formulate_output["prior_strategy"] = results[0]
                    wisdom_found = True
            return {"primitive": "FORMULATE", "sub_mode": "load_wisdom",
                    "wisdom_found": wisdom_found, "confidence": 0.6 if wisdom_found else 0.4}

        # ── F-phase compositional sub-modes (rFP §6) ────────────────────
        # Session 1: each new sub-mode extends `define` (anomaly-based problem
        # formulation) with a compositional operator tag. Session 2 dispatches
        # these through the Recruitment Layer to reasoning primitives
        # (DECOMPOSE / CONTRAST / GENERALIZE) + pattern_primitives.merge/abstract.
        elif sub in ("compose_intersection", "compose_union",
                     "compose_difference", "narrow_to_subset",
                     "generalize_from_instance"):
            deviation = np.abs(sv_arr - self._ema_state)
            top_k = 3 if sub == "narrow_to_subset" else 5
            top_dims = np.argsort(deviation)[-top_k:][::-1].tolist()
            avg_dim = np.mean(top_dims)
            if avg_dim < 20:
                domain = "body_mind"
            elif avg_dim < 65:
                domain = "inner_spirit"
            elif avg_dim < 85:
                domain = "outer_perception"
            else:
                domain = "outer_spirit"
            difficulty = float(np.mean(deviation[top_dims]))
            compose_op = {
                "compose_intersection": "intersection",
                "compose_union": "union",
                "compose_difference": "difference",
                "narrow_to_subset": "narrow",
                "generalize_from_instance": "generalize",
            }[sub]
            template = (f"{domain} anomaly [{compose_op}]: dims {top_dims}, "
                        f"deviation={difficulty:.3f}")
            self.state.formulate_output = {
                "problem_template": template,
                "domain": domain,
                "anomalous_dims": top_dims,
                "difficulty": difficulty,
                "compose_op": compose_op,
                "trigger": self.state.trigger_reason,
            }
            return {"primitive": "FORMULATE", "sub_mode": sub,
                    "domain": domain, "difficulty": round(difficulty, 4),
                    "anomalous_dims": top_dims, "compose_op": compose_op,
                    "confidence": 0.5, "session_1_stub": True,
                    "recruitment_resolved": False}

    def _prim_recall(self, sub, chain_archive, meta_wisdom, ex_mem):
        """Query memory sources."""
        results = []
        best_match = None

        if sub == "chain_archive" and chain_archive:
            domain = self.state.formulate_output.get("domain", "general")
            results = chain_archive.query_by_domain(domain, min_outcome=0.3, limit=10)
            if not results:
                results = chain_archive.query_high_scoring(min_outcome=0.5, limit=10)
            if results:
                best_match = results[0]

        elif sub == "experience" and ex_mem:
            try:
                domain = self.state.formulate_output.get("domain", "general")
                results = ex_mem.recall_similar(domain, top_k=5)
                if results:
                    best_match = results[0] if isinstance(results[0], dict) else {"score": results[0]}
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._prim_recall: domain = self.state.formulate_output.get('domain', 'gener...", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._prim_recall.line3589', throttle=100)

        elif sub == "wisdom" and meta_wisdom:
            template = self.state.formulate_output.get("problem_template", "")
            results = meta_wisdom.query_by_pattern(template, min_confidence=0.3)
            if results:
                best_match = results[0]

        elif sub == "entity":
            # rFP_titan_meta_outer_layer Bridge 1+2 — composed RECALL across
            # heterogeneous stores keyed by entity_refs["primary_person"].
            # Blocks up to 200ms on the async composed-recall future that
            # FORMULATE dispatched. Falls back to inner-only gracefully if
            # outer_reader not wired or is_active() is False.
            outer = self._await_outer_context(timeout_s=0.2)
            if outer:
                person = (outer.get("person") or {})
                felt = outer.get("felt_history") or []
                events = outer.get("recent_events") or []
                if person or felt or events:
                    results = []
                    if person:
                        results.append({"kind": "person", "data": person})
                    for f in felt[:5]:
                        results.append({"kind": "felt", "data": f})
                    for e in events[:3]:
                        results.append({"kind": "event", "data": e})
                    if results:
                        best_match = results[0]
                        self.state.outer_context_used = True

        elif sub == "topic":
            # rFP_titan_meta_outer_layer Bridge 1 — topic-scoped composed recall.
            # Pulls concept (via knowledge_worker bus-RPC) + felt_experiences
            # mentioning the topic + inner narrative snippets.
            outer = self._await_outer_context(timeout_s=0.2)
            if outer:
                concept = outer.get("topic")
                felt = outer.get("felt_history") or []
                inner = outer.get("inner_narrative") or []
                if concept or felt or inner:
                    results = []
                    if concept:
                        results.append({"kind": "concept", "data": concept})
                    for f in felt[:5]:
                        results.append({"kind": "felt", "data": f})
                    for n in inner[:5]:
                        results.append({"kind": "inner", "data": n})
                    if results:
                        best_match = results[0]
                        self.state.outer_context_used = True

        # ── F-phase compositional sub-modes (rFP §6) ────────────────────
        # Session 1: each new sub-mode falls back to the closest existing
        # retrieval path. Session 2 dispatches through Recruitment Layer
        # to episodic_memory / semantic_graph / chain_archive / timechain.
        elif sub == "episodic_specific" and ex_mem:
            try:
                domain = self.state.formulate_output.get("domain", "general")
                results = ex_mem.recall_similar(domain, top_k=5)
                if results:
                    best_match = (results[0] if isinstance(results[0], dict)
                                  else {"score": results[0]})
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._prim_recall: domain = self.state.formulate_output.get('domain', 'gener...", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._prim_recall.line3653', throttle=100)

        elif sub == "semantic_neighbors":
            pass  # Session 2: semantic_graph.neighbors resolver

        elif sub == "procedural_matching" and chain_archive:
            domain = self.state.formulate_output.get("domain", "general")
            results = chain_archive.query_by_domain(domain, min_outcome=0.3,
                                                     limit=10)
            if results:
                best_match = results[0]

        elif sub == "autobiographical_relevant" and ex_mem:
            try:
                domain = self.state.formulate_output.get("domain", "general")
                results = ex_mem.recall_similar(domain, top_k=10)
                if results:
                    best_match = (results[0] if isinstance(results[0], dict)
                                  else {"score": results[0]})
            except Exception as _swallow_exc:
                swallow_warn("[logic.meta_reasoning] MetaReasoningEngine._prim_recall: domain = self.state.formulate_output.get('domain', 'gener...", _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._prim_recall.line3673', throttle=100)

        is_new_mode = sub in ("episodic_specific", "semantic_neighbors",
                               "procedural_matching",
                               "autobiographical_relevant")
        self.state.recalled_data = {
            "source": sub, "results": results,
            "count": len(results), "best_match": best_match,
        }
        out = {"primitive": "RECALL", "sub_mode": sub,
               "count": len(results), "best_match": best_match is not None,
               "confidence": min(0.7, 0.4 + len(results) * 0.03)}
        if is_new_mode:
            out["session_1_stub"] = True
            out["recruitment_resolved"] = False
        return out

    def _prim_hypothesize(self, sub, nm):
        """Generate or refine strategy hypotheses."""
        if sub == "generate":
            # Build hypothesis from formulation + recall
            strategy = []
            # Default strategy template based on domain
            domain = self.state.formulate_output.get("domain", "general")
            if domain in ("body_mind", "outer_perception"):
                strategy = ["DECOMPOSE", "COMPARE", "IF_THEN"]
            elif domain in ("inner_spirit", "outer_spirit"):
                strategy = ["ASSOCIATE", "SEQUENCE", "COMPARE"]
            else:
                strategy = ["COMPARE", "DECOMPOSE", "IF_THEN"]

            # Enhance from recalled best match
            if self.state.recalled_data.get("best_match"):
                recalled = self.state.recalled_data["best_match"]
                if isinstance(recalled, dict) and recalled.get("chain_sequence"):
                    strategy = recalled["chain_sequence"][:5]

            predicted = 0.5 + random.random() * 0.3
            hypothesis = {
                "strategy": strategy,
                "predicted_confidence": predicted,
                "domain": domain,
                "reasoning": f"Based on {self.state.recalled_data.get('count', 0)} recalled chains",
            }
            self.state.hypotheses.append(hypothesis)
            return {"primitive": "HYPOTHESIZE", "sub_mode": "generate",
                    "hypothesis": hypothesis, "confidence": predicted}

        elif sub == "refine":
            if self.state.hypotheses and self.state.delegate_results:
                last_result = self.state.delegate_results[-1]
                best_hyp = self.state.hypotheses[-1]
                actual = last_result.get("confidence", 0.5)
                if actual < best_hyp["predicted_confidence"] * 0.7:
                    # Strategy underperformed — try different primitives
                    alt_prims = ["ASSOCIATE", "NEGATE", "LOOP", "SEQUENCE"]
                    best_hyp["strategy"] = random.sample(alt_prims, min(3, len(alt_prims)))
                    best_hyp["predicted_confidence"] *= 0.8
                    best_hyp["refined"] = True
            return {"primitive": "HYPOTHESIZE", "sub_mode": "refine",
                    "refined": bool(self.state.delegate_results),
                    "confidence": self.state.confidence}

        elif sub == "compare":
            if len(self.state.hypotheses) >= 2:
                ranked = sorted(self.state.hypotheses,
                                key=lambda h: h.get("predicted_confidence", 0), reverse=True)
                spread = ranked[0]["predicted_confidence"] - ranked[-1]["predicted_confidence"]
                return {"primitive": "HYPOTHESIZE", "sub_mode": "compare",
                        "best": ranked[0], "spread": round(spread, 3),
                        "count": len(ranked), "confidence": ranked[0]["predicted_confidence"]}
            return {"primitive": "HYPOTHESIZE", "sub_mode": "compare",
                    "count": len(self.state.hypotheses), "confidence": self.state.confidence}

        # ── F-phase compositional sub-modes (rFP §6) ────────────────────
        # Session 1: fall back to closest existing mode + tag the hypothesis
        # with the compositional operator. Session 2 routes via Recruitment
        # Layer to reasoning.ANALOGIZE / CONTRAST / IF_THEN + CREATIVITY
        # for richer downstream strategies.
        elif sub in ("analogize_from", "contrast_with",
                     "propose_by_inversion", "extend_pattern"):
            domain = self.state.formulate_output.get("domain", "general")
            # Strategy template biased by compositional intent
            if sub == "analogize_from":
                strategy = ["ASSOCIATE", "COMPARE", "SEQUENCE"]
            elif sub == "contrast_with":
                strategy = ["COMPARE", "NEGATE", "IF_THEN"]
            elif sub == "propose_by_inversion":
                strategy = ["NEGATE", "IF_THEN", "COMPARE"]
            else:  # extend_pattern
                strategy = ["SEQUENCE", "ASSOCIATE", "IF_THEN"]
            if self.state.recalled_data.get("best_match"):
                recalled = self.state.recalled_data["best_match"]
                if isinstance(recalled, dict) and recalled.get("chain_sequence"):
                    strategy = recalled["chain_sequence"][:5]
            predicted = 0.5 + random.random() * 0.3
            hypothesis = {
                "strategy": strategy,
                "predicted_confidence": predicted,
                "domain": domain,
                "compose_op": sub,
                "reasoning": (f"[{sub}] "
                              f"Based on {self.state.recalled_data.get('count', 0)} "
                              f"recalled chains"),
            }
            self.state.hypotheses.append(hypothesis)
            return {"primitive": "HYPOTHESIZE", "sub_mode": sub,
                    "hypothesis": hypothesis, "confidence": predicted,
                    "session_1_stub": True, "recruitment_resolved": False}

    def _prim_delegate(self, sub, reasoning_engine):
        """Inject strategy bias into main reasoning."""
        # rFP_titan_meta_outer_layer Bridge 3 — active search / gap-fill.
        # When chain hit impasse OR composed recall returned thin, DELEGATE
        # can invoke external fetchers: knowledge_search, X timeline,
        # events_window_poll. Result flows into outer_context.gap_fill.
        if sub == "gap_fill":
            return self._prim_delegate_gap_fill()

        if not reasoning_engine or not self.state.hypotheses:
            return {"primitive": "DELEGATE", "sub_mode": sub,
                    "delegated": False, "reason": "no_hypothesis"}

        best_hyp = max(self.state.hypotheses,
                       key=lambda h: h.get("predicted_confidence", 0))
        strategy = best_hyp.get("strategy", [])

        # Build 8D bias vector for main reasoning primitives
        MAIN_PRIMITIVES = ["COMPARE", "IF_THEN", "SEQUENCE", "ASSOCIATE",
                           "DECOMPOSE", "LOOP", "NEGATE", "CONCLUDE"]
        bias = np.zeros(8, dtype=np.float32)
        for prim in strategy:
            if prim in MAIN_PRIMITIVES:
                idx = MAIN_PRIMITIVES.index(prim)
                bias[idx] += self._delegate_bias_strength
        bias = np.clip(bias, -self._delegate_max_bias, self._delegate_max_bias)

        # Set bias on reasoning engine
        if hasattr(reasoning_engine, 'set_strategy_bias'):
            reasoning_engine.set_strategy_bias(bias)

        # Track delegation
        self.state.awaiting_delegate = True
        self.state.delegate_start_chains = reasoning_engine._total_chains

        return {"primitive": "DELEGATE", "sub_mode": sub,
                "delegated": True, "bias": bias.tolist(),
                "strategy": strategy, "confidence": self.state.confidence}

    def _prim_delegate_gap_fill(self):
        """DELEGATE.gap_fill — pull fresh external data when chain is thin.

        Invokes knowledge_search, X timeline search, or events window poll
        based on what the chain seems to need. Config-gated per source:
          - active_search_knowledge (default True)
          - active_search_x (default False)
          - active_search_events (default False)

        Result stored in outer_context["gap_fill"] for subsequent primitives.
        """
        if not self._outer_enabled():
            return {"primitive": "DELEGATE", "sub_mode": "gap_fill",
                    "gap_filled": False, "reason": "outer_inactive"}
        reader = self._outer_reader
        topic = self.state.entity_refs.get("current_topic") or ""
        person = self.state.entity_refs.get("primary_person") or ""
        if not topic and not person and not self.state.impasse_topic:
            return {"primitive": "DELEGATE", "sub_mode": "gap_fill",
                    "gap_filled": False, "reason": "no_handle"}
        gap: dict = {"sources": []}
        try:
            if topic or self.state.impasse_topic:
                t = topic or self.state.impasse_topic
                kn = reader.knowledge_search(t, max_results=5)
                if kn:
                    gap["knowledge"] = kn
                    gap["sources"].append("knowledge")
                if reader.config.active_search_x:
                    x_q = (f"@{person[1:]} {t}".strip()
                           if person.startswith("@") else t)
                    x_hits = reader.x_timeline_search(x_q, count=10)
                    if x_hits:
                        gap["x_timeline"] = x_hits
                        gap["sources"].append("x_timeline")
            if reader.config.active_search_events:
                ev = reader.events_window_poll()
                if ev:
                    gap["events_window"] = ev
                    gap["sources"].append("events_window")
        except Exception as e:
            swallow_warn('[MetaOuter] gap_fill err', e,
                         key="logic.meta_reasoning.gap_fill_err", throttle=100)
        if not gap["sources"]:
            return {"primitive": "DELEGATE", "sub_mode": "gap_fill",
                    "gap_filled": False, "reason": "all_sources_empty"}
        # Stash onto outer_context so downstream primitives can read
        self.state.outer_context.setdefault("gap_fill", {})
        self.state.outer_context["gap_fill"] = gap
        self.state.outer_context_used = True
        return {"primitive": "DELEGATE", "sub_mode": "gap_fill",
                "gap_filled": True, "sources": gap["sources"],
                "confidence": self.state.confidence}

    def _check_delegate(self, reasoning_engine):
        """Check if delegated main reasoning has completed."""
        if not reasoning_engine:
            self.state.awaiting_delegate = False
            return {"action": "CONTINUE", "primitive": "DELEGATE", "waiting": False}

        chains_since = reasoning_engine._total_chains - self.state.delegate_start_chains
        if chains_since < 1:
            return {"action": "WAITING", "primitive": "DELEGATE",
                    "chains_since": chains_since}

        # Collect result
        conf = reasoning_engine.confidence
        gut = reasoning_engine.gut_agreement
        self.state.delegate_results.append({
            "confidence": conf, "gut_agreement": gut,
            "chains_completed": chains_since,
        })
        self.state.awaiting_delegate = False

        # Clear bias
        if hasattr(reasoning_engine, 'clear_strategy_bias'):
            reasoning_engine.clear_strategy_bias()

        # Update meta confidence
        self.state.confidence = 0.4 * self.state.confidence + 0.6 * conf

        return {"action": "CONTINUE", "primitive": "DELEGATE",
                "delegate_done": True, "result_confidence": conf,
                "confidence": self.state.confidence}

    def _prim_synthesize(self, sub, meta_wisdom, autoencoder, sv):
        """Integrate insights from the meta-chain."""
        if sub == "combine":
            combined = {
                "formulation": self.state.formulate_output.get("problem_template", ""),
                "recalled_count": self.state.recalled_data.get("count", 0),
                "hypotheses_count": len(self.state.hypotheses),
                "delegate_count": len(self.state.delegate_results),
            }
            if self.state.delegate_results:
                avg_conf = np.mean([r["confidence"] for r in self.state.delegate_results])
                combined["avg_delegate_confidence"] = round(float(avg_conf), 4)
            self.state.synthesized = combined
            return {"primitive": "SYNTHESIZE", "sub_mode": "combine",
                    "combined": combined, "confidence": self.state.confidence}

        elif sub == "abstract":
            insight = f"Domain {self.state.formulate_output.get('domain', '?')}: "
            if self.state.delegate_results:
                best_del = max(self.state.delegate_results, key=lambda r: r["confidence"])
                insight += f"best strategy achieved {best_del['confidence']:.2f} confidence"
            else:
                insight += "no delegation results to abstract from"
            self.state.synthesized["insight"] = insight
            return {"primitive": "SYNTHESIZE", "sub_mode": "abstract",
                    "insight": insight, "confidence": self.state.confidence}

        elif sub == "rank":
            if self.state.hypotheses:
                ranked = sorted(self.state.hypotheses,
                                key=lambda h: h.get("predicted_confidence", 0), reverse=True)
                self.state.synthesized["ranked_strategies"] = [
                    {"strategy": h["strategy"], "predicted": h["predicted_confidence"]}
                    for h in ranked[:3]
                ]
            return {"primitive": "SYNTHESIZE", "sub_mode": "rank",
                    "count": len(self.state.hypotheses), "confidence": self.state.confidence}

        elif sub == "distill_save":
            saved = False
            if meta_wisdom and self.state.confidence > 0.4 and self.state.formulate_output:
                prob_emb = None
                if autoencoder and autoencoder.is_trained:
                    prob_emb = autoencoder.encode(sv[:132])
                meta_wisdom.store_wisdom(
                    problem_pattern=self.state.formulate_output.get("problem_template",
                                                                    self.state.trigger_reason),
                    strategy_sequence=self.state.chain,
                    outcome_score=self.state.confidence,
                    problem_embedding=prob_emb,
                )
                saved = True
                self._total_wisdom_saved += 1
            return {"primitive": "SYNTHESIZE", "sub_mode": "distill_save",
                    "saved": saved, "confidence": self.state.confidence}

    def _prim_evaluate(self, sub, nm):
        """Assess meta-chain quality."""
        n = len(self.state.chain)

        if sub == "check_progress":
            conf_trend = 0.0
            if len(self.state.chain_results) >= 3:
                recent = [r.get("confidence", 0.5) for r in self.state.chain_results[-5:]]
                conf_trend = (recent[-1] - recent[0]) / max(len(recent), 1)

            should_continue = n < self.state.max_steps and (conf_trend >= -0.1 or n < 5)
            rec = "continue" if should_continue else "conclude"

            return {"primitive": "EVALUATE", "sub_mode": "check_progress",
                    "should_continue": should_continue,
                    "confidence_trend": round(conf_trend, 4),
                    "steps_remaining": self.state.max_steps - n,
                    "recommendation": rec,
                    "confidence": self.state.confidence}

        elif sub == "check_strategy":
            improvement = 0.0
            if self.state.delegate_results:
                avg = np.mean([r["confidence"] for r in self.state.delegate_results])
                improvement = float(avg - self._baseline_confidence)
            return {"primitive": "EVALUATE", "sub_mode": "check_strategy",
                    "improvement": round(improvement, 4),
                    "baseline": round(self._baseline_confidence, 4),
                    "confidence": self.state.confidence}

        elif sub == "check_resources":
            fatigue_signal = nm.get("GABA", 0.5)
            has_energy = fatigue_signal < 0.7
            return {"primitive": "EVALUATE", "sub_mode": "check_resources",
                    "has_energy": has_energy, "gaba": round(fatigue_signal, 3),
                    "recommendation": "continue" if has_energy else "conclude",
                    "confidence": self.state.confidence}

        elif sub == "peer_cgn":
            # rFP_titan_meta_outer_layer Bridge 4 — weight meta confidence
            # by peer CGN consumers' β-posterior on current topic. When
            # peers are well-grounded on the topic, meta should be more
            # confident; when peers are uncertain, meta tempers down.
            if not self._outer_enabled():
                return {"primitive": "EVALUATE", "sub_mode": "peer_cgn",
                        "grounded": False, "reason": "outer_inactive",
                        "confidence": self.state.confidence}
            reader = self._outer_reader
            topic = (self.state.entity_refs.get("current_topic")
                     or self.state.impasse_topic or "")
            if not topic:
                return {"primitive": "EVALUATE", "sub_mode": "peer_cgn",
                        "grounded": False, "reason": "no_topic",
                        "confidence": self.state.confidence}
            betas = []
            for consumer in ("knowledge", "language", "social", "reasoning"):
                b = reader.peer_cgn_beta(consumer, topic)
                if b is not None:
                    betas.append((consumer, b))
            if not betas:
                return {"primitive": "EVALUATE", "sub_mode": "peer_cgn",
                        "grounded": False, "reason": "no_peer_data",
                        "peers_queried": ["knowledge", "language", "social",
                                           "reasoning"],
                        "confidence": self.state.confidence}
            avg_beta = sum(b for _, b in betas) / len(betas)
            # Soft-modulate confidence (±10% max) — peer β is informative
            # but cannot override meta's own estimate.
            weight_nudge = (avg_beta - 0.5) * 0.2
            new_conf = float(np.clip(self.state.confidence + weight_nudge,
                                      0.0, 1.0))
            self.state.confidence = new_conf
            self.state.outer_context_used = True
            return {"primitive": "EVALUATE", "sub_mode": "peer_cgn",
                    "grounded": True, "peers": dict(betas),
                    "avg_beta": round(avg_beta, 4),
                    "confidence_nudge": round(weight_nudge, 4),
                    "confidence": round(new_conf, 4)}

    # ── M7: BREAK Primitive ────────────────────────────────────────

    def _save_checkpoint(self):
        """Auto-save checkpoint at FORMULATE/SYNTHESIZE steps."""
        import copy
        self.state.checkpoints.append({
            "step_index": len(self.state.chain),
            "chain_snapshot": list(self.state.chain),
            "results_snapshot": list(self.state.chain_results),
            "confidence": self.state.confidence,
            "formulate_output": copy.deepcopy(self.state.formulate_output),
            "recalled_data": copy.deepcopy(self.state.recalled_data),
            "hypotheses": copy.deepcopy(self.state.hypotheses),
            "delegate_results": copy.deepcopy(self.state.delegate_results),
        })

    def _prim_break(self, sub, sv, reasoning_engine):
        """Backtrack the meta-chain."""
        self.state.break_count += 1

        if sub == "rewind_last":
            if self.state.chain:
                removed = self.state.chain.pop()
                if self.state.chain_results:
                    self.state.chain_results.pop()
                # Recalculate confidence from remaining results
                if self.state.chain_results:
                    self.state.confidence = self.state.chain_results[-1].get(
                        "confidence", self.state.confidence)
                else:
                    self.state.confidence = 0.5
                # Clear bias if we had DELEGATE
                if reasoning_engine and hasattr(reasoning_engine, 'clear_strategy_bias'):
                    reasoning_engine.clear_strategy_bias()
                return {"primitive": "BREAK", "sub_mode": "rewind_last",
                        "removed": removed, "confidence": self.state.confidence}
            return {"primitive": "BREAK", "sub_mode": "rewind_last",
                    "removed": None, "confidence": self.state.confidence}

        elif sub == "rewind_to_checkpoint":
            if self.state.checkpoints:
                cp = self.state.checkpoints.pop()
                self.state.chain = cp["chain_snapshot"]
                self.state.chain_results = cp["results_snapshot"]
                self.state.confidence = cp["confidence"]
                self.state.formulate_output = cp["formulate_output"]
                self.state.recalled_data = cp["recalled_data"]
                self.state.hypotheses = cp["hypotheses"]
                self.state.delegate_results = cp["delegate_results"]
                if reasoning_engine and hasattr(reasoning_engine, 'clear_strategy_bias'):
                    reasoning_engine.clear_strategy_bias()
                logger.info("[META] BREAK rewind to checkpoint (step %d)", cp["step_index"])
                return {"primitive": "BREAK", "sub_mode": "rewind_to_checkpoint",
                        "rewound_to": cp["step_index"], "confidence": cp["confidence"]}
            return {"primitive": "BREAK", "sub_mode": "rewind_to_checkpoint",
                    "rewound_to": None, "confidence": self.state.confidence}

        elif sub == "restart_fresh":
            # Remember what failed for negative bias
            failed_strategy = list(self.state.chain[-5:]) if self.state.chain else []
            self.state.chain.clear()
            self.state.chain_results.clear()
            self.state.checkpoints.clear()
            self.state.hypotheses.clear()
            self.state.delegate_results.clear()
            self.state.synthesized.clear()
            self.state.confidence = 0.5
            self.state.awaiting_delegate = False
            if reasoning_engine and hasattr(reasoning_engine, 'clear_strategy_bias'):
                reasoning_engine.clear_strategy_bias()
            logger.info("[META] BREAK restart_fresh (failed: %s)", failed_strategy[:3])
            return {"primitive": "BREAK", "sub_mode": "restart_fresh",
                    "failed_strategy": failed_strategy, "confidence": 0.5}

        return {"primitive": "BREAK", "sub_mode": sub, "confidence": self.state.confidence}

    # ── M8: SPIRIT_SELF Primitive ────────────────────────────────

    def _prim_spirit_self(self, sub, nm):
        """Nudge own neuromodulators to shift emotional context."""
        nudges = SPIRIT_SELF_NUDGE_MAP.get(sub, {})
        self.state.pre_nudge_confidence = self.state.confidence
        self.state.last_spirit_self_step = len(self.state.chain)
        self.state.spirit_self_cooldown = self._spirit_self_cooldown_max
        logger.info("[META] SPIRIT_SELF.%s — nudges=%s", sub, nudges)
        return {"primitive": "SPIRIT_SELF", "sub_mode": sub,
                "nudge_request": {"sub_mode": sub, "nudges": nudges},
                "confidence": self.state.confidence}

    # ── INTROSPECT Primitive ───────────────────────────────────

    def _prim_introspect(self, sub, sv, nm):
        """Self-reasoning: observe and model own cognitive state.

        Delegates to SelfReasoningEngine for actual introspection.
        If no engine attached, returns a minimal self-observation from
        the 132D state vector and neuromod levels.
        """
        self.state.introspect_used = True  # Max 1 per chain

        # Tier 3: maker_alignment queries the Maker-Titan bond state
        if sub == "maker_alignment":
            try:
                from titan_plugin.maker import get_titan_maker
                _tm = get_titan_maker()
                if _tm:
                    alignment = _tm.get_maker_alignment_score()
                    bond_health = _tm.get_bond_health()
                    dialogue_summary = _tm.get_dialogue_for_introspect(n=5)
                    result = {
                        "primitive": "INTROSPECT", "sub_mode": "maker_alignment",
                        "alignment_score": round(alignment, 4),
                        "bond_health": bond_health,
                        "dialogue_summary": dialogue_summary,
                        "confidence": min(0.9, 0.3 + alignment * 0.6),
                        "note": "Maker-Titan bond state via TitanMaker Tier 3",
                    }
                    logger.info(
                        "[META] INTROSPECT.maker_alignment — score=%.3f "
                        "interactions=%d trajectory=%.3f",
                        alignment, bond_health.get("interaction_count", 0),
                        bond_health.get("agreement_trajectory", 0))
                    return result
            except Exception as _ma_err:
                swallow_warn('[META] maker_alignment error', _ma_err,
                             key="logic.meta_reasoning.maker_alignment_error", throttle=100)
            return {
                "primitive": "INTROSPECT", "sub_mode": "maker_alignment",
                "alignment_score": 0.5, "confidence": 0.2,
                "note": "TitanMaker not available",
            }

        sr = self._self_reasoning
        if sr is None:
            # Minimal fallback — observe from raw state vector
            sv_arr = np.array(sv[:132], dtype=np.float32)
            inner_avg = float(np.mean(sv_arr[:65]))
            outer_avg = float(np.mean(sv_arr[65:])) if len(sv_arr) > 65 else 0.5
            result = {
                "primitive": "INTROSPECT", "sub_mode": sub,
                "inner_avg": round(inner_avg, 4),
                "outer_avg": round(outer_avg, 4),
                "neuromods": {k: round(v, 3) for k, v in nm.items()},
                "confidence": 0.3,
                "note": "SelfReasoningEngine not attached — minimal fallback",
            }
            logger.info("[META] INTROSPECT.%s (fallback) — inner=%.3f outer=%.3f",
                        sub, inner_avg, outer_avg)
            return result

        # Neuromod-coupled mode override: if sub_mode is "state_audit"
        # and neuromods suggest a more specific mode, prefer it
        effective_sub = sub
        if sub == "state_audit":
            suggested = sr.select_introspection_mode(nm)
            if suggested != "state_audit":
                effective_sub = suggested

        # Gather available data for the engine
        # (spirit_worker injects these via _introspect_context dict)
        ctx = getattr(self, '_introspect_context', {})

        result = sr.introspect(
            sub_mode=effective_sub,
            epoch=ctx.get("epoch", 0),
            neuromods=nm,
            msl_data=ctx.get("msl_data"),
            reasoning_stats=ctx.get("reasoning_stats"),
            language_stats=ctx.get("language_stats"),
            coordinator_data=ctx.get("coordinator_data"),
            state_132d=sv,
        )

        logger.info("[META] INTROSPECT.%s (via %s) — conf=%.3f trigger=%s",
                    effective_sub, sub, result.get("confidence", 0),
                    result.get("mode_trigger", "default"))
        # META-CGN Producer #13: self_reasoning.reflection_depth (rFP v3 § 12 row 13)
        # Queue for spirit_worker drain. Edge-detected per sub_mode so only NEW
        # personal maxima fire. Highest healing-impact producer: INTROSPECT +
        # SPIRIT_SELF primitives currently at 2-3% across all 3 Titans — this
        # signal pushes them up (quality > 0.5 reinforces per META-CGN semantics).
        try:
            _rd_conf = result.get("confidence") if isinstance(result, dict) else None
            if _rd_conf is not None:
                self._pending_cgn_reflection_events.append({
                    "sub_mode": effective_sub,
                    "confidence": float(_rd_conf),
                })
        except Exception as _rd_q_err:
            logger.warning(
                "[META-CGN] Failed to queue reflection_depth event for producer #13 drain: %s "
                "(sub_mode=%s) — this emission will be missed",
                _rd_q_err, effective_sub)
        # META-CGN Producer #14: self_reasoning.coherence_gain (rFP v3 § 12 row 14)
        # Fires only on coherence_check sub-mode. Passes chi_coherence value
        # for 4-threshold detection (0.3/0.5/0.7/0.9) in spirit_worker drain.
        try:
            if effective_sub == "coherence_check":
                _cg_chi = ctx.get("msl_data", {}).get("chi_coherence")
                if _cg_chi is not None:
                    self._pending_cgn_coherence_events.append({
                        "chi_coh": float(_cg_chi),
                    })
        except Exception as _cg_q_err:
            logger.warning(
                "[META-CGN] Failed to queue coherence_gain event for producer #14 drain: %s "
                "— this emission will be missed", _cg_q_err)
        return result

    # ── M9: EUREKA Detection ────────────────────────────────────

    def _fire_eureka(self, confidence, sv, meta_wisdom, autoencoder,
                     trigger_primitive: str = "SYNTHESIZE"):
        """Fire EUREKA pulse: DA burst + wisdom crystallization.

        trigger_primitive: which primitive triggered the EUREKA (SYNTHESIZE,
        FORMULATE, HYPOTHESIZE, EVALUATE, SPIRIT_SELF). Different cognitive
        styles produce insights through different primitives — T1 via SYNTHESIZE,
        T2 via recall-connection, T3 via FORMULATE articulation.
        """
        # Compute novelty — two complementary signals:
        # 1. Wisdom-embedding novelty: cosine distance to nearest stored wisdom
        # 2. Prediction novelty: from the Prediction engine's running EMA
        # The max of both is used so that even when the wisdom store is
        # saturated (novelty→0), the prediction novelty can still drive DA.
        #
        # A1 stop-bleed (2026-04-21 audit §A1, shipped afternoon):
        # Pre-audit every eureka fired with novelty=1.000 because
        # `query_by_embedding` returned empty (for reasons narrowed to 4
        # candidates — autoencoder drift, missing problem_embedding on old
        # rows, JSON encoding, wrong shape — needs live SQL inspection)
        # AND the permissive `else: wisdom_novelty = 1.0` fallback treated
        # "no match" as "fully novel." Combined with `pred_novelty=0.0`
        # (self_prediction_accuracy defaults to 1.0 → surprise=0) and the
        # permissive `max()` combiner, every eureka got novelty=1.0.
        # Stop-bleed: change the fallback to 0.5 (neutral — "we don't know
        # how novel this is") and count the empty-return events so we can
        # distinguish "wisdom store truly empty" from "query pathway
        # broken" during investigation. Proper fix to `query_by_embedding`
        # empty-return needs separate live-SQL work (A1.4).
        wisdom_novelty = 0.5
        if meta_wisdom and autoencoder and autoencoder.is_trained:
            try:
                emb = autoencoder.encode(sv[:132])
                similar = meta_wisdom.query_by_embedding(
                    emb, min_confidence=0.3, top_k=1)
                if similar:
                    wisdom_novelty = 1.0 - similar[0].get("similarity", 0.5)
                else:
                    wisdom_novelty = 0.5  # A1 stop-bleed: was 1.0 permissive
                    self._wisdom_query_empty_returns = getattr(
                        self, "_wisdom_query_empty_returns", 0) + 1
            except Exception as _wn_err:
                # Defensive — embedding encode or DB query can fail under
                # autoencoder-not-trained or DB-locked conditions. Stay at
                # 0.5 neutral + log for investigation.
                wisdom_novelty = 0.5
                logger.debug("[META] wisdom_novelty query failed: %s", _wn_err)
        # Prediction-based novelty: 1 - prediction_accuracy from subsystem cache
        # (populated by spirit_worker from prediction_engine._novelty_ema)
        pred_novelty = 0.0
        if self._subsystem_cache:
            pred_acc = self._subsystem_cache.get("self_prediction_accuracy", 1.0)
            pred_novelty = max(0.0, 1.0 - pred_acc)  # surprise = lack of prediction
        novelty = max(wisdom_novelty, pred_novelty)
        # Immediate wisdom crystallization
        wisdom_id = -1
        if meta_wisdom and self.state.formulate_output:
            prob_emb = None
            if autoencoder and autoencoder.is_trained:
                prob_emb = autoencoder.encode(sv[:132])
            wisdom_id = meta_wisdom.store_wisdom(
                problem_pattern=self.state.formulate_output.get("problem_template", "eureka"),
                strategy_sequence=self.state.chain,
                outcome_score=confidence,
                problem_embedding=prob_emb,
            )
            if wisdom_id > 0:
                meta_wisdom.force_crystallize(wisdom_id)
                self._total_wisdom_saved += 1
        # DA burst magnitude
        da_burst = self._eureka_da_base + novelty * self._eureka_da_novelty
        self._total_eurekas += 1
        self._eureka_cooldown_steps = self._eureka_cooldown_max
        logger.info("[META] *** EUREKA *** via %s conf=%.3f novelty=%.3f DA_burst=%.3f chain=%s",
                    trigger_primitive, confidence, novelty, da_burst, self.state.chain[:5])
        event = {
            "type": "eureka",
            "trigger_primitive": trigger_primitive,
            "confidence": round(confidence, 4),
            "novelty": round(novelty, 4),
            "da_burst_magnitude": round(da_burst, 4),
            "chain_length": len(self.state.chain),
            "domain": self.state.formulate_output.get("domain", "general"),
            "wisdom_id": wisdom_id,
        }
        # META-CGN Producer #4: queue for spirit_worker drain (rFP v3 § 12 row 4).
        # Decoupled queue pattern — this logic module stays ignorant of the bus;
        # spirit_worker drains after each tick() and emits via emit_meta_cgn_signal.
        try:
            self._pending_cgn_events.append(event)
        except Exception as _cgn_queue_err:
            # Best-effort: eureka detection itself must not fail on queue issues,
            # but silent failure would hide a regression (e.g. list replaced by
            # non-append-able type). Log at WARNING per directive_error_visibility.
            logger.warning(
                "[META-CGN] Failed to queue eureka event for producer #4 drain: %s "
                "(event_type=%s trigger=%s) — this emission will be missed",
                _cgn_queue_err, event.get("type"), event.get("trigger_primitive"))
        return event

    # ── Private: Input Construction ───────────────────────────────

    def _build_meta_input(self, sv, nm, chain_archive, meta_autoencoder=None) -> list:
        """Build 80D meta-policy input."""
        inp = []

        # [0:20] State summary (132D → 20D via structured pooling)
        sv_arr = np.array(sv[:132], dtype=np.float32)
        # Means per trinity section (6D)
        inp.extend([
            float(np.mean(sv_arr[0:5])),    # Inner Body
            float(np.mean(sv_arr[5:20])),   # Inner Mind
            float(np.mean(sv_arr[20:65])),  # Inner Spirit
            float(np.mean(sv_arr[65:70])),  # Outer Body
            float(np.mean(sv_arr[70:85])),  # Outer Mind
            float(np.mean(sv_arr[85:130])), # Outer Spirit
        ])
        # Variances (6D)
        inp.extend([
            float(np.std(sv_arr[0:5])),
            float(np.std(sv_arr[5:20])),
            float(np.std(sv_arr[20:65])),
            float(np.std(sv_arr[65:70])),
            float(np.std(sv_arr[70:85])),
            float(np.std(sv_arr[85:130])),
        ])
        # Cross-coherence: inner-outer correlation (3D)
        for i_start, i_end, o_start, o_end in [(0,5,65,70), (5,20,70,85), (20,65,85,130)]:
            inner = sv_arr[i_start:i_end]
            outer = sv_arr[o_start:o_end]
            min_len = min(len(inner), len(outer))
            if min_len > 0:
                corr = float(np.corrcoef(inner[:min_len], outer[:min_len])[0,1])
                inp.append(0.0 if np.isnan(corr) else corr)
            else:
                inp.append(0.0)
        # Top anomalous dims (5D, normalized 0-1)
        deviation = np.abs(sv_arr - self._ema_state)
        top5 = np.argsort(deviation)[-5:][::-1]
        inp.extend([float(d / 132.0) for d in top5])
        # Total: 20D

        # [20:36] Problem embedding (16D).
        # nn_iql_rl audit C1 fix 2026-04-20: prefer trained meta_autoencoder
        # (Component 14 — 7,968+ training steps on disk) which maps 132D felt
        # state → 16D contrastive embedding. Fall back to hash stub only if
        # the autoencoder is unavailable or not yet trained (< 100 steps).
        # AE output is tanh-bounded [-1, 1]; rescale to [0, 1] to match the
        # range expected by the rest of the input vector.
        emb_wired = False
        if meta_autoencoder is not None and getattr(meta_autoencoder, "is_trained", False):
            try:
                ae_emb = meta_autoencoder.encode(sv_arr.tolist())
                if ae_emb and len(ae_emb) >= 16:
                    inp.extend([float((v + 1.0) * 0.5) for v in ae_emb[:16]])
                    emb_wired = True
            except Exception:
                emb_wired = False
        if not emb_wired:
            if self.state.formulate_output:
                # Hash-based fallback until autoencoder graduates
                template = self.state.formulate_output.get("problem_template", "")
                emb = [(hash((template, i)) % 1000) / 1000.0 for i in range(16)]
                inp.extend(emb)
            else:
                inp.extend([0.5] * 16)

        # [36:48] Strategy history EMA (12D)
        inp.extend(self._strategy_history.tolist())

        # [48:56] Delegate outcomes (8D)
        del_vec = [0.0] * 8
        for i, dr in enumerate(self.state.delegate_results[-4:]):
            del_vec[i*2] = dr.get("confidence", 0.5)
            del_vec[i*2+1] = dr.get("gut_agreement", 0.5)
        inp.extend(del_vec)

        # [56:64] Archive stats (8D)
        if chain_archive:
            stats = chain_archive.get_stats()
            total = max(stats.get("total", 1), 1)
            by_src = stats.get("by_source", {})
            inp.extend([
                min(total / 500.0, 1.0),                    # total normalized
                by_src.get("main", 0) / max(total, 1),      # main fraction
                by_src.get("meta", 0) / max(total, 1),      # meta fraction
                stats.get("avg_outcome", 0.5),               # avg quality
                min(stats.get("unconsolidated", 0) / 50.0, 1.0),
                0.0, 0.0, 0.0,                              # reserved
            ])
        else:
            inp.extend([0.0] * 8)

        # [64:68] Meta-chain state (4D)
        n = len(self.state.chain)
        inp.extend([
            n / max(self.state.max_steps, 1),  # length normalized
            self.state.confidence,
            float(self._strategy_history[:6].sum()),  # diversity
            1.0 if self.state.awaiting_delegate else 0.0,
        ])

        # [68:74] Neuromods (6D)
        inp.extend([
            nm.get("DA", 0.5), nm.get("5HT", 0.5), nm.get("NE", 0.5),
            nm.get("ACh", 0.5), nm.get("Endorphin", 0.5), nm.get("GABA", 0.5),
        ])

        # [74:80] M7/M8/M9 state signals (6D)
        inp.extend([
            float(self.state.break_count) / max(self.state.max_breaks, 1),
            len(self.state.checkpoints) / 5.0,
            float(self.state.spirit_self_cooldown) / max(self._spirit_self_cooldown_max, 1),
            1.0 if self.state.last_spirit_self_step >= 0 else 0.0,
            float(self._total_eurekas) / max(self._total_meta_chains, 1),
            1.0 if self._total_meta_chains >= self._spirit_self_gate else 0.0,
        ])

        # Pad/truncate to 80D
        inp = inp[:META_POLICY_INPUT_DIM]
        while len(inp) < META_POLICY_INPUT_DIM:
            inp.append(0.0)
        return inp

    def _build_sub_mode_input(self, sv, nm, primitive) -> list:
        """Build 30D sub-mode policy input."""
        inp = []
        # [0:20] Same state summary as meta-policy
        sv_arr = np.array(sv[:132], dtype=np.float32)
        inp.extend([
            float(np.mean(sv_arr[0:5])), float(np.mean(sv_arr[5:20])),
            float(np.mean(sv_arr[20:65])), float(np.mean(sv_arr[65:70])),
            float(np.mean(sv_arr[70:85])), float(np.mean(sv_arr[85:130])),
        ])
        # Deviations (6D)
        deviation = np.abs(sv_arr - self._ema_state)
        inp.extend([
            float(np.mean(deviation[0:5])), float(np.mean(deviation[5:20])),
            float(np.mean(deviation[20:65])), float(np.mean(deviation[65:70])),
            float(np.mean(deviation[70:85])), float(np.mean(deviation[85:130])),
        ])
        # [12:16] Primitive-specific context (4D)
        n = len(self.state.chain)
        has_formulate = 1.0 if self.state.formulate_output else 0.0
        has_recall = 1.0 if self.state.recalled_data else 0.0
        has_hyp = min(len(self.state.hypotheses) / 3.0, 1.0)
        has_delegate = min(len(self.state.delegate_results) / 3.0, 1.0)
        inp.extend([has_formulate, has_recall, has_hyp, has_delegate])
        # [16:20] Chain position (4D)
        inp.extend([
            n / max(self.state.max_steps, 1),
            self.state.confidence,
            float(self.state.break_count) / 3.0,
            1.0 if self.state.awaiting_delegate else 0.0,
        ])
        # [20:26] Neuromods (6D)
        inp.extend([
            nm.get("DA", 0.5), nm.get("5HT", 0.5), nm.get("NE", 0.5),
            nm.get("ACh", 0.5), nm.get("Endorphin", 0.5), nm.get("GABA", 0.5),
        ])
        # [26:30] M7/M8 state
        inp.extend([
            float(self.state.break_count) / max(self.state.max_breaks, 1),
            len(self.state.checkpoints) / 5.0,
            float(self.state.spirit_self_cooldown) / max(self._spirit_self_cooldown_max, 1),
            1.0 if self.state.last_spirit_self_step >= 0 else 0.0,
        ])

        inp = inp[:SUB_MODE_INPUT_DIM]
        while len(inp) < SUB_MODE_INPUT_DIM:
            inp.append(0.0)
        return inp

    def _get_temperature(self, nm) -> float:
        """Neuromod-derived temperature for meta-action selection."""
        da = nm.get("DA", 0.5)
        ne = nm.get("NE", 0.5)
        # High DA → lower temp (exploit), High NE → higher temp (explore)
        return max(0.3, 1.0 - da * 0.5 + ne * 0.3)

    def _compute_adaptive_epsilon(self) -> float:
        """Forced exploration when meta-policy is collapsed.

        Task 4 P2 (2026-04-12): switched from unique-prims-count ramp to
        dominance-share ramp. Reason: empirical baseline showed T1 with
        unique_prims_ema=3.12 (looks healthy) but dominant_share=90%
        simultaneously — chains with FORMULATE×9 + RECALL×1 average
        "3 unique" while still being 90% one primitive. Unique-count
        rewards "pick something different occasionally" rather than
        "actually break out of dominance". Dominance is the honest signal.

        Returns 0.0 when dominance is healthy (≤0.50 — no monoculture),
        ramping linearly up to 0.40 when fully collapsed (≥0.90 dominance).

        Architecturally permanent — replaces the flawed unique-count metric.
        """
        actions = self.buffer._actions[-500:] if hasattr(self.buffer, '_actions') else []
        if len(actions) < 50:
            return 0.0
        prim_counts: dict = {}
        for a in actions:
            name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
            prim_counts[name] = prim_counts.get(name, 0) + 1
        total = sum(prim_counts.values())
        if total == 0:
            return 0.0
        dominant_share = max(prim_counts.values()) / total
        # Linear ramp: dominance 0.50 → ε=0.0, dominance 0.90 → ε=0.40
        if dominant_share <= 0.50:
            return 0.0
        if dominant_share >= 0.90:
            return 0.40
        return 0.40 * (dominant_share - 0.50) / 0.40

    def _normalize_neuromods(self, neuromods) -> dict:
        """Normalize neuromod dict to simple {name: float} format."""
        nm = {}
        if not isinstance(neuromods, dict):
            return {"DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5,
                    "Endorphin": 0.5, "GABA": 0.5}
        for key in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
            v = neuromods.get(key, 0.5)
            if isinstance(v, dict):
                nm[key] = v.get("level", 0.5)
            else:
                nm[key] = float(v) if v is not None else 0.5
        return nm

    # ── Private: Meta-Reward ──────────────────────────────────────

    def _compute_legacy_meta_reward(self) -> float:
        """Pre-TUNING-012 reward computation. Used as fallback when DNA is
        disabled (all toggles off) and as a safety net component when DNA
        is enabled. The legacy reward provides high-level structure rewards
        (strategy_improvement, insight_novelty) that the per-primitive
        compound rewards do not cover."""
        reward = 0.0

        # strategy_improvement (0.30)
        if self.state.delegate_results:
            avg = np.mean([r["confidence"] for r in self.state.delegate_results])
            improvement = (float(avg) - self._baseline_confidence) / max(0.1, self._baseline_confidence)
            reward += min(1.0, max(-0.5, improvement)) * 0.30

        # insight_novelty (0.20)
        if self.state.synthesized:
            reward += 0.10  # Base novelty for any synthesis
            if self.state.synthesized.get("insight"):
                reward += 0.10  # Additional for abstract insight

        # prediction_accuracy (0.20)
        if self.state.hypotheses and self.state.delegate_results:
            best_hyp = max(self.state.hypotheses,
                           key=lambda h: h.get("predicted_confidence", 0))
            predicted = best_hyp.get("predicted_confidence", 0.5)
            actual = self.state.delegate_results[-1].get("confidence", 0.5)
            accuracy = 1.0 - abs(predicted - actual)
            reward += accuracy * 0.20

        # efficiency (0.15)
        n = len(self.state.chain)
        efficiency = 1.0 - (n / max(self.state.max_steps, 1))
        reward += max(0, efficiency) * 0.15

        # patience (0.15) — 5-HT gated
        if n >= 5:
            reward += min(n / 10.0, 1.0) * 0.15

        # diversity (0.15) — emergent self-exploration via primitive variety
        if n >= 2:
            unique_prims = len(set(step.split(".")[0] for step in self.state.chain))
            reward += min((unique_prims - 1) * 0.05, 0.15)

        # 2026-04-13 (Phase 3 of foundational healing rFP): legacy M7
        # BREAK penalty REMOVED. Combined with the cost_weight in the
        # compound reward (also zeroed in same commit), this gave BREAK
        # a baseline of ~-0.13 per use that trained the policy to never
        # select BREAK. BREAK is a meta-cognitive escape tool; it should
        # be rewarded for breaking patterns, not punished for trying.
        # Per-primitive reward symmetry is restored (all primitives now
        # clamped at 0 baseline).
        # OLD: if self.state.break_count > 0:
        #          reward -= self._break_base_cost * self.state.break_count

        # M8: SPIRIT_SELF effectiveness (retroactive)
        if self.state.last_spirit_self_step >= 0:
            improvement = self.state.confidence - self.state.pre_nudge_confidence
            reward += max(-0.05, min(0.05, improvement * 0.5))

        return reward

    def _compute_compound_per_primitive_reward(self) -> tuple[float, dict]:
        """TUNING-012 v2 Sub-phase A: per-primitive compound rewards.

        Replays each primitive in the completed chain through its compound
        reward helper. Sources signals from Inner Memory + TimeChain +
        Smart Contracts (currently from cached subsystem signals — full bus
        wiring deferred to Sub-phase D next session).

        Returns (compound_reward, per_primitive_breakdown).
        """
        from titan_plugin.logic.meta_reasoning_rewards import (
            compute_primitive_reward,
            empty_subsystem_signals,
        )

        # Subsystem signals: this session uses cached defaults. Sub-phase D
        # next session will replace this with live bus queries to TimeChain
        # and Smart Contracts. The reward functions degrade gracefully when
        # signals are 0 — the base + state-dependent components still fire.
        if self._subsystem_cache and self._subsystem_cache_ts > 0:
            signals = self._subsystem_cache
        else:
            signals = empty_subsystem_signals()

        # Tag chain success for retroactive components (RECALL.outcome,
        # INTROSPECT.calibration, BREAK.recovery)
        legacy = self._compute_legacy_meta_reward()
        # Normalize legacy reward [0, 1.5] → success signal [0, 1]
        self.state.chain_succeeded = max(0.0, min(1.0, legacy / 1.0))

        compound_total = 0.0
        breakdown_per_step: dict = {}

        for step_idx, step_key in enumerate(self.state.chain):
            prim = step_key.split(".")[0]
            step_output = (
                self.state.chain_results[step_idx]
                if step_idx < len(self.state.chain_results)
                else {}
            )
            r, bd = compute_primitive_reward(
                prim, self.state, step_output, self._dna, signals
            )
            if not bd:
                # Primitive has no compound helper (HYPOTHESIZE/DELEGATE/
                # SYNTHESIZE/SPIRIT_SELF) — leave to legacy path.
                continue
            compound_total += r
            # Track first occurrence of each primitive's breakdown
            if prim not in breakdown_per_step:
                breakdown_per_step[prim] = bd
            # Audit telemetry: keep per-primitive breakdown history (last 100).
            if prim in self._recent_primitive_breakdowns:
                self._recent_primitive_breakdowns[prim].append(dict(bd))

        # Average across the number of compound-rewarded steps so chain length
        # doesn't dominate. Each primitive contributes its compound reward
        # ONCE per chain (averaged across occurrences).
        compound_steps = sum(
            1 for k in self.state.chain
            if k.split(".")[0] in {"RECALL", "FORMULATE", "EVALUATE", "INTROSPECT", "BREAK"}
        )
        if compound_steps > 0:
            compound_total = compound_total / compound_steps

        return compound_total, breakdown_per_step

    def _compound_rewards_enabled(self) -> bool:
        """Compound rewards fire when DNA is loaded AND at least one
        subsystem signal source toggle is true."""
        if not self._dna:
            return False
        return bool(
            self._dna.get("inner_memory_signals_enabled", False)
            or self._dna.get("timechain_signals_enabled", False)
            or self._dna.get("contract_signals_enabled", False)
        )

    def _parse_template_primitives(self, template_id: str) -> list:
        """Extract primitive names from a chain_iql template string.

        Templates are stored as arrow-joined strings like
        "FORMULATE→RECALL→SYNTHESIZE→EVALUATE". Returns ordered primitives,
        filtering out anything not in META_PRIMITIVES.
        """
        if not template_id:
            return []
        # Support both unicode arrow and ASCII fallback
        parts = template_id.replace("→", "->").split("->")
        return [p.strip() for p in parts if p.strip() in META_PRIMITIVES]

    def _resolve_dominant_primitive(self) -> str:
        """Return the primitive with highest action count in the buffer.

        Used by META-CGN HAOV evidence + context encoding. Mirrors the
        dominance calculation at meta_reasoning.py:1376-1384 (buffer slice).
        Returns "" if no actions recorded yet.
        """
        try:
            recent_actions = list(self.buffer._actions) if self.buffer else []
            if not recent_actions:
                return ""
            counts: dict = {}
            for a in recent_actions:
                name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else ""
                if name:
                    counts[name] = counts.get(name, 0) + 1
            if not counts:
                return ""
            dominant, n = max(counts.items(), key=lambda x: x[1])
            total = sum(counts.values())
            # Side-effect: cache dominance share for encode_state context
            self._last_dominance_share = n / total if total else 0.0
            return dominant
        except Exception:
            return ""

    def _build_meta_cgn_ctx(self, final_domain: str,
                            terminal_reward: float) -> dict:
        """Build the 30D encoding context for META-CGN (rFP §3 state schema).

        Populates the dict consumed by MetaCGNConsumer.encode_state. Missing
        fields fall back to safe defaults inside the encoder; we best-effort
        extract available signals from engine + spirit state.
        """
        # At _conclude_chain, chain has just completed — chain_step equals
        # chain_len (MetaChainState doesn't track them separately).
        _chain_len = len(self.state.chain)
        # state.gut doesn't exist on MetaChainState — gut-confidence gap is a
        # proxy signal; omit (defaults to 0) until meta tracks gut readout.
        ctx: dict = {
            "chain_len": _chain_len,
            "chain_step": _chain_len,
            "terminal_reward_ema": max(0.0, min(1.0, float(terminal_reward))),
            "confidence_avg_20": float(self._baseline_confidence),
            "gut_conf_gap": 0.0,
            "monoculture_share": float(
                getattr(self, "_last_dominance_share", 0.0)),
            "is_in_soar_impasse": bool(
                getattr(self, "_in_soar_impasse", False)),
        }
        # Neuromod + chi from subsystem cache (TUNING-012 Phase D signals)
        sc = self._subsystem_cache or {}
        ctx["DA"] = sc.get("neuromod_DA", 0.5)
        ctx["5HT"] = sc.get("neuromod_5HT", 0.5)
        ctx["NE"] = sc.get("neuromod_NE", 0.5)
        ctx["GABA"] = sc.get("neuromod_GABA", 0.5)
        ctx["chi_total"] = sc.get("chi_total", 0.5)
        # Cross-consumer recency (best-effort from cache if populated)
        ctx["language_grounding_rate"] = sc.get("language_grounding_rate", 0.0)
        ctx["knowledge_acq_rate"] = sc.get("knowledge_acq_rate", 0.0)
        ctx["social_quality_delta"] = sc.get("social_quality_delta", 0.5)
        # Spirit context
        ctx["sleep_drive"] = sc.get("sleep_drive", 0.5)
        ctx["wake_drive"] = sc.get("wake_drive", 0.5)
        ctx["is_dreaming"] = sc.get("is_dreaming", False)
        # Recent chain success rate from existing recent_uniques deque (rough proxy)
        if self._recent_chain_uniques:
            # proxy: uniqueness >= 3 = healthy chain
            _hits = sum(1 for u in self._recent_chain_uniques if u >= 3)
            ctx["success_rate_20"] = _hits / len(self._recent_chain_uniques)
        # Chains since eureka (if tracked)
        ctx["chains_since_eureka"] = float(
            getattr(self, "_chains_since_last_eureka", 100))
        ctx["chains_since_impasse"] = float(
            getattr(self, "_chains_since_last_impasse", 50))
        # SOAR repeat-impasse awareness: lets META-CGN learn that
        # repetitive chains → low V(s) → steer toward diversity
        ctx["repeat_impasse_lifetime"] = float(
            getattr(self, "_repeat_impasse_count", 0))
        # Diversity score: Shannon entropy of primitive distribution (0=monoculture, 1=uniform)
        try:
            _actions = list(self.buffer._actions) if self.buffer else []
            if _actions:
                _counts = {}
                for _a in _actions[-200:]:  # last 200 actions
                    _counts[_a] = _counts.get(_a, 0) + 1
                _total = sum(_counts.values())
                import math as _m
                _entropy = -sum((c/_total) * _m.log2(c/_total) for c in _counts.values() if c > 0)
                _max_entropy = _m.log2(max(len(_counts), 1)) or 1.0
                ctx["diversity_score"] = _entropy / _max_entropy  # normalized 0-1
            else:
                ctx["diversity_score"] = 0.5
        except Exception:
            ctx["diversity_score"] = 0.5
        return ctx

    def _compute_meta_reward(self) -> float:
        """Compute terminal reward for completed meta-chain.

        TUNING-012 v2: When DNA is loaded with subsystem toggles enabled,
        BLEND legacy structural reward with per-primitive compound rewards.
        Without DNA (or all toggles off), falls back to pure legacy path.
        """
        legacy = self._compute_legacy_meta_reward()

        if not self._compound_rewards_enabled():
            return max(0.0, round(legacy, 4))

        compound, breakdown = self._compute_compound_per_primitive_reward()

        # Blend: legacy provides chain-level structure (delegate improvement,
        # synthesis novelty, efficiency, patience). Compound provides
        # per-primitive context-dependent shaping. Blend ratio is DNA-tunable
        # (compound_legacy_blend_alpha in [meta_reasoning_dna], per-Titan
        # overrides for T1/T2/T3 cognitive personalities).
        blend_alpha = self._compound_blend_alpha
        blended = (1 - blend_alpha) * legacy + blend_alpha * compound

        # Per-component log: one line per primitive with its breakdown
        if breakdown:
            for prim, bd in breakdown.items():
                comps = " ".join(
                    f"{k}={v:+.3f}" for k, v in bd.items()
                    if k not in ("base", "total")
                )
                logger.info(
                    "[META] %s reward: base=%.3f %s total=%.3f",
                    prim, bd.get("base", 0.0), comps, bd.get("total", 0.0),
                )

        # ── Monoculture reward correction (Task 4 TRANSITIONAL) ─────
        # P13: gated behind `sunset_task4` config flag. When META-CGN's
        # structural monoculture prevention (P6 F anti-monoculture + UCB
        # composition + per-domain V) proves sufficient, Maker flips the
        # flag and mono_adj hardcoded penalty is skipped entirely. Default
        # False preserves current (working) behavior; Maker opts in post-
        # P6 soak when diversity healing is reliable.
        sunset_task4 = bool(self._dna.get("sunset_task4", False)) if hasattr(
            self, "_dna") else False
        mono_adj = 0.0
        recent_actions = (self.buffer._actions[-500:]
                          if hasattr(self.buffer, '_actions') else [])
        if not sunset_task4 and len(recent_actions) >= 100:
            prim_counts: dict = {}
            for a in recent_actions:
                name = META_PRIMITIVES[a] if 0 <= a < NUM_META_ACTIONS else "?"
                prim_counts[name] = prim_counts.get(name, 0) + 1
            total_a = sum(prim_counts.values())
            if total_a > 0:
                dominant_name, dominant_n = max(prim_counts.items(), key=lambda x: x[1])
                dom_share = dominant_n / total_a
                # Primitives used in THIS chain
                chain_prims = [str(s).split(".")[0] for s in self.state.chain]
                chain_dom_frac = chain_prims.count(dominant_name) / max(len(chain_prims), 1)
                if dom_share > 0.80:
                    # Penalty: chains dominated by the monoculture primitive get punished
                    # proportional to how much of the chain was that primitive.
                    # 2026-04-11: raised -0.08→-0.18 — previous value left dominant
                    # with highest net reward even after penalty (FORMULATE 0.239-0.064=0.175
                    # > RECALL 0.142). New value inverts the ranking.
                    # Task 4 T1 (2026-04-12): bumped -0.18 → -0.30. Baseline showed
                    # -0.18 gave avg penalty ~-0.15 vs FORMULATE 0.22, RECALL 0.09 — net
                    # FORMULATE 0.07 vs RECALL 0.09, ranking just barely flipped. -0.30
                    # gives clear inversion (FORMULATE -0.05 vs RECALL +0.09) so policy
                    # gradient has a real signal to learn from.
                    # TRANSITIONAL 2026-04-12: sunset when META-CGN lands — META-CGN's
                    # grounded V(s) replaces hardcoded reward-shaping coefficients.
                    mono_adj = -0.30 * chain_dom_frac
                    # Bonus: chains with minority primitives (<5% share) get rewarded
                    minority_count = sum(
                        1 for p in chain_prims
                        if prim_counts.get(p, 0) / total_a < 0.05 and p != dominant_name
                    )
                    if minority_count > 0:
                        mono_adj += 0.06 * minority_count / max(len(chain_prims), 1)
                    if abs(mono_adj) > 0.001:
                        logger.info(
                            "[META] Monoculture reward adj: %.3f (dom=%s@%.0f%%, chain_dom_frac=%.0f%%, minority_steps=%d)",
                            mono_adj, dominant_name, dom_share * 100, chain_dom_frac * 100, minority_count,
                        )
                        # Audit telemetry: track mono_adj fires + cumulative penalty.
                        self._mono_adj_fires_count += 1
                        self._mono_adj_cumulative += float(mono_adj)
                        self._mono_adj_history.append({
                            "ts": time.time(),
                            "chain": int(self._total_meta_chains),
                            "adj": round(float(mono_adj), 4),
                            "dominant": dominant_name,
                            "dom_share": round(float(dom_share), 3),
                            "chain_dom_frac": round(float(chain_dom_frac), 3),
                        })
        blended += mono_adj
        # Audit telemetry: rolling chain reward history.
        self._recent_chain_rewards.append(round(float(blended), 4))

        if breakdown:
            logger.info(
                "[META] Chain reward blend: legacy=%.3f compound=%.3f mono_adj=%+.3f → blended=%.3f",
                legacy, compound, mono_adj, blended,
            )

        return max(0.0, round(blended, 4))

    # ── Private: Persistence ──────────────────────────────────────

    def _load(self):
        self.meta_policy.load(os.path.join(self.save_dir, "meta_policy.json"))
        # Sub-mode policies
        sub_path = os.path.join(self.save_dir, "meta_sub_modes.json")
        if os.path.exists(sub_path):
            try:
                with open(sub_path) as f:
                    sub_data = json.load(f)
                for prim, data in sub_data.items():
                    if prim in self.sub_mode_policies:
                        self.sub_mode_policies[prim].from_dict(data)
            except Exception as _swallow_exc:
                swallow_warn('[logic.meta_reasoning] MetaReasoningEngine._load: with open(sub_path) as f: sub_data = json.load(f)', _swallow_exc,
                             key='logic.meta_reasoning.MetaReasoningEngine._load.line4953', throttle=100)
        self.buffer.load(os.path.join(self.save_dir, "meta_buffer.json"))
        # Stats
        stats_path = os.path.join(self.save_dir, "meta_stats.json")
        if os.path.exists(stats_path):
            # 2026-04-19: surface empty/corrupt-file case at WARN level
            # (was silent except pass, which hid T1's 0-byte file for
            # 2 days — counters restarted from 0 every boot). See
            # save_all defensive type coercion fix for the write side.
            file_size = os.path.getsize(stats_path)
            if file_size == 0:
                logger.warning(
                    "[META] meta_stats.json is 0 bytes at %s — counters "
                    "will start from defaults. Previous save likely failed "
                    "due to non-serializable type (see save_all try/except "
                    "WARN in brain log). DATA LOSS unless backup exists.",
                    stats_path)
            try:
                with open(stats_path) as f:
                    s = json.load(f)
                self._total_meta_chains = s.get("total_chains", 0)
                self._total_meta_steps = s.get("total_steps", 0)
                self._total_wisdom_saved = s.get("total_wisdom_saved", 0)
                self._total_eurekas = s.get("total_eurekas", 0)
                self._baseline_confidence = s.get("baseline_confidence", 0.5)
                if s.get("strategy_history"):
                    self._strategy_history = np.array(s["strategy_history"], dtype=np.float32)
                if s.get("ema_state"):
                    self._ema_state = np.array(s["ema_state"], dtype=np.float32)
                # Restore recent_chain_uniques deque (adaptive epsilon-greedy)
                saved_uniques = s.get("recent_chain_uniques", [])
                if saved_uniques:
                    self._recent_chain_uniques = deque(
                        (int(u) for u in saved_uniques), maxlen=50
                    )
                # v4 persistence gap fixes (2026-04-17)
                saved_bias = s.get("primitive_bias", {})
                if saved_bias and hasattr(self, '_primitive_bias'):
                    self._primitive_bias = {k: float(v) for k, v in saved_bias.items()}
                if s.get("diversity_pressure_target") is not None and hasattr(self, '_diversity_pressure_target'):
                    self._diversity_pressure_target = s["diversity_pressure_target"]
                if s.get("reroute_count") and hasattr(self, '_reroute_count'):
                    self._reroute_count = int(s["reroute_count"])
                if s.get("suggested_template") is not None and hasattr(self, '_suggested_template'):
                    self._suggested_template = s["suggested_template"]
                if s.get("suggested_template_q") and hasattr(self, '_suggested_template_q'):
                    self._suggested_template_q = float(s["suggested_template_q"])
                # Cognitive-contract counters (2026-04-21 fix for
                # BUG-CONTRACT-GATE-STARVATION — handler fires restored
                # across restarts so dashboard lifetime accounting is honest)
                if s.get("cc_strategy_drift_fires"):
                    self._cc_strategy_drift_fires = int(s["cc_strategy_drift_fires"])
                _saved_sd_top = s.get("cc_strategy_drift_last_top")
                if isinstance(_saved_sd_top, list):
                    self._cc_strategy_drift_last_top = _saved_sd_top
                if s.get("cc_pattern_emerged_fires"):
                    self._cc_pattern_emerged_fires = int(s["cc_pattern_emerged_fires"])
                _saved_pe_last = s.get("cc_pattern_emerged_last")
                if isinstance(_saved_pe_last, list):
                    self._cc_pattern_emerged_last = _saved_pe_last
                if s.get("cc_monoculture_fires"):
                    self._cc_monoculture_fires = int(s["cc_monoculture_fires"])
                _saved_mono_last = s.get("cc_monoculture_last")
                if isinstance(_saved_mono_last, dict):
                    self._cc_monoculture_last = _saved_mono_last
                # --- Audit-driven additions (2026-04-21): lifetime counters
                # + mid-decay state that were silently reset on every restart.
                # Pattern mirrors cc_* restoration — only assign if key
                # present, with type guards so old meta_stats.json without
                # these fields loads cleanly (backward compat).
                if s.get("inengine_mono_total_fires"):
                    self._inengine_mono_total_fires = int(
                        s["inengine_mono_total_fires"])
                if s.get("inengine_mono_last_fire_chain") is not None:
                    self._inengine_mono_last_fire_chain = int(
                        s["inengine_mono_last_fire_chain"])
                if s.get("mono_adj_fires_count"):
                    self._mono_adj_fires_count = int(s["mono_adj_fires_count"])
                if s.get("mono_adj_cumulative"):
                    self._mono_adj_cumulative = float(s["mono_adj_cumulative"])
                if s.get("diversity_pressure_total_applied"):
                    self._diversity_pressure_total_applied = int(
                        s["diversity_pressure_total_applied"])
                if s.get("diversity_pressure_remaining"):
                    self._diversity_pressure_remaining = int(
                        s["diversity_pressure_remaining"])
                if s.get("diversity_pressure_initial_magnitude"):
                    self._diversity_pressure_initial_magnitude = float(
                        s["diversity_pressure_initial_magnitude"])
                if s.get("diversity_pressure_initial_decay"):
                    self._diversity_pressure_initial_decay = int(
                        s["diversity_pressure_initial_decay"])
                if s.get("introspect_executions_lifetime"):
                    self._introspect_executions_lifetime = int(
                        s["introspect_executions_lifetime"])
                if s.get("introspect_picks_lifetime"):
                    self._introspect_picks_lifetime = int(
                        s["introspect_picks_lifetime"])
                if s.get("introspect_rerouted_lifetime"):
                    self._introspect_rerouted_lifetime = int(
                        s["introspect_rerouted_lifetime"])
                if s.get("next_chain_id"):
                    self._next_chain_id = int(s["next_chain_id"])
                if s.get("last_concluded_chain_id"):
                    self._last_concluded_chain_id = int(
                        s["last_concluded_chain_id"])
                if s.get("repeat_impasse_count"):
                    self._repeat_impasse_count = int(s["repeat_impasse_count"])
                _saved_rip = s.get("repeat_impasse_primitives")
                if isinstance(_saved_rip, dict):
                    self._repeat_impasse_primitives = {
                        str(k): int(v) for k, v in _saved_rip.items()}
                _saved_slt = s.get("soar_last_successful_topic")
                if isinstance(_saved_slt, str) and _saved_slt:
                    self._soar_last_successful_topic = _saved_slt
            except Exception as _ld_err:
                # 2026-04-19: surface the parse failure at WARN instead of
                # silent pass. Prior behavior hid T1's 0-byte file for 2
                # days. Legitimate reasons to reach here: empty file (handled
                # above with 0-byte WARN), genuinely corrupt JSON,
                # schema-mismatch after a save format change.
                logger.warning(
                    "[META] meta_stats.json load FAILED: %s — counters "
                    "start from defaults this boot. Investigate file size "
                    "and JSON validity.", _ld_err)
        logger.info("[META] Loaded: chains=%d, steps=%d, wisdom=%d, policy_updates=%d",
                    self._total_meta_chains, self._total_meta_steps,
                    self._total_wisdom_saved, self.meta_policy.total_updates)
