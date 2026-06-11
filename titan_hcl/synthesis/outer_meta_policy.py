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


# ── Phase-C feature schema (FULL MSL — Q4 RESOLVED 2026-06-11) ────────
# 8 local base features (recall/skill/engram/tool-intent) + the FULL 20D MSL
# `distilled_context` (Titan's OWN emergent inner signal — NOT a curated subset;
# distilling it would pre-impose our limits on Titan, Maker Ad-3 — instrumented
# via feature_contributions() so full-vs-curated can be assessed over time) + 2
# parametric retrieval-prior features (the matched Reasoning composite, D4/Q3).
# Slot 0 is a constant bias. The MSL block is [-1,1] (tanh); base/retrieval [0,1].
_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "bias",                    # 0  constant 1.0
    "recall_top_cosine",       # 1  [0,1] top recall similarity (D3)
    "recall_count_norm",       # 2  [0,1] min(n_recalled,10)/10
    "skill_utility",           # 3  [0,1] top skill_cell time_cost proficiency (D5)
    "skill_matched",           # 4  {0,1} a skill cell matched the goal_class
    "engram_ground",           # 5  [0,1] matched Engram groundedness (D5)
    "requires_tool",           # 6  {0,1} tool-intent regex (D5)
    "has_code_signal",         # 7  {0,1} computational shape (intent.code present)
)
MSL_CONTEXT_DIM = 20  # msl.infer()["distilled_context"] = tanh(raw[17:37]) ∈ [-1,1]
_MSL_FEATURE_NAMES: tuple[str, ...] = tuple(f"msl_ctx_{i}" for i in range(MSL_CONTEXT_DIM))
_RETRIEVAL_FEATURE_NAMES: tuple[str, ...] = (
    "composite_match_score",        # [0,1] top reasoning-composite SC-search cosine (D4)
    "composite_match_action_norm",  # [0,1] matched composite action idx / (NUM_OUTER_ACTIONS-1)
)
OUTER_FEATURE_NAMES: tuple[str, ...] = (
    _BASE_FEATURE_NAMES + _MSL_FEATURE_NAMES + _RETRIEVAL_FEATURE_NAMES)
OUTER_POLICY_INPUT_DIM = len(OUTER_FEATURE_NAMES)  # 30 (8 base + 20 MSL + 2 retrieval)

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
OUTER_META_POLICY_STATE_SCHEMA_VERSION = 2  # v2 (Phase C): 11→30 (full MSL context[20] + retrieval prior)
OUTER_META_POLICY_STATE_SPEC = RegistrySpec(
    name=OUTER_META_POLICY_STATE_SLOT,
    dtype=np.dtype("float32"),
    shape=(OUTER_POLICY_FLAT_DIM,),
    feature_flag="",  # enable-gating is at the worker (publish) + agno (read) level
    schema_version=OUTER_META_POLICY_STATE_SCHEMA_VERSION,
    variable_size=False,
)

# ── MSL distilled_context[20] SHM slot (Phase C piece 2 — Q4 full MSL) ─
# A DEDICATED fixed float32(20) slot so the agno DECIDE path reads the full
# `distilled_context` O(1) AT decision-time. The existing `msl_state.bin`
# (MSLStatePublisher) is a variable-size msgpack of identity/depth telemetry,
# read via the `_v5["msl"]` overlay fetched AFTER the decision (Q4) — this slot
# closes that timing gap with a minimal fixed-vector read.
# Producer = cognitive_worker (additive publish; G18/G20 single-writer).
OUTER_MSL_CONTEXT_STATE_SLOT = "outer_msl_context_state"
OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION = 1
OUTER_MSL_CONTEXT_STATE_SPEC = RegistrySpec(
    name=OUTER_MSL_CONTEXT_STATE_SLOT,
    dtype=np.dtype("float32"),
    shape=(MSL_CONTEXT_DIM,),
    feature_flag="",
    schema_version=OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION,
    variable_size=False,
)


def msl_context_to_fixed(ctx) -> np.ndarray:
    """Coerce a distilled_context (list/ndarray, normally 20D) → the fixed
    float32 `(MSL_CONTEXT_DIM,)` array the SHM slot requires — pad/trim/NaN-safe
    so a malformed/short context never breaks the publish."""
    arr = np.asarray(list(ctx) if ctx is not None else [], dtype=np.float32).ravel()
    fixed = np.zeros(MSL_CONTEXT_DIM, dtype=np.float32)
    n = min(MSL_CONTEXT_DIM, arr.shape[0])
    if n:
        fixed[:n] = np.nan_to_num(arr[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return fixed


def read_msl_context(shm_root=None) -> Optional[np.ndarray]:
    """Read the published MSL `distilled_context[20]` from SHM (O(1), G18/G20).
    Returns a length-`MSL_CONTEXT_DIM` float32 ndarray, or None if the slot is
    not yet published / unreadable (cold-start — caller treats as zeros)."""
    try:
        from titan_hcl.core.state_registry import (
            StateRegistryReader, resolve_shm_root)
        root = shm_root if shm_root is not None else resolve_shm_root()
        arr = StateRegistryReader(OUTER_MSL_CONTEXT_STATE_SPEC, root).read()
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float32).ravel()
    except Exception:
        return None


def _clip01(v: float) -> float:
    if v != v:  # NaN guard
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def _clip_pm1(v: float) -> float:
    """Clamp to [-1, 1] (the MSL distilled_context range; tanh-bounded)."""
    if v != v:  # NaN guard
        return 0.0
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))


@dataclass
class OuterFeatures:
    """The decision feature bundle, built on the agno path (D2-D5).

    Every field is sourced from what is ALREADY computed locally per turn —
    no new sync RPC (INV-OML-7). Missing signals default to 0 / empty (the
    cold-start, which the policy learns to weight)."""

    recall_top_cosine: float = 0.0
    recall_count: int = 0
    skill_utility: float = 0.0
    skill_matched: bool = False
    engram_ground: float = 0.0
    requires_tool: bool = False
    has_code_signal: bool = False
    # Q4 (full MSL, Phase C): the 20D `distilled_context` ∈ [-1,1]; empty →
    # zeros (cold-start / not-yet-published — wired in the prehook piece).
    msl_context: tuple = ()
    # D4 parametric retrieval prior — the top reasoning-composite SC-search hit.
    composite_match_score: float = 0.0
    composite_match_action_norm: float = 0.0
    # DEPRECATED — the 3 reserved MSL identity scalars; superseded by
    # `msl_context` (full MSL). Accepted-but-IGNORED so the agno caller does not
    # break until it is migrated to `msl_context` (Phase-C prehook piece).
    msl_i_confidence: float = 0.0
    msl_attention_entropy: float = 0.0
    msl_concept_confidence: float = 0.0

    def to_vector(self) -> np.ndarray:
        ctx = list(self.msl_context or ())
        if len(ctx) < MSL_CONTEXT_DIM:
            ctx = ctx + [0.0] * (MSL_CONTEXT_DIM - len(ctx))
        else:
            ctx = ctx[:MSL_CONTEXT_DIM]
        base = [
            1.0,  # bias
            _clip01(self.recall_top_cosine),
            min(max(int(self.recall_count), 0), 10) / 10.0,
            _clip01(self.skill_utility),
            1.0 if self.skill_matched else 0.0,
            _clip01(self.engram_ground),
            1.0 if self.requires_tool else 0.0,
            1.0 if self.has_code_signal else 0.0,
        ]
        msl = [_clip_pm1(c) for c in ctx]
        retrieval = [
            _clip01(self.composite_match_score),
            _clip01(self.composite_match_action_norm),
        ]
        return np.array(base + msl + retrieval, dtype=np.float32)


class OuterMetaPolicy:
    """Numpy ReLU MLP (30 → 16 → 8 → 5) trained by REINFORCE-with-baseline.

    Mirrors `MetaPolicy` (logic/meta_reasoning.py): forward / select_action /
    train_step / save / load — plus the EMA reward baseline, SHM flat
    (de)serialization, and a grounded-route prior seed for cold-start.
    """

    def __init__(self, input_dim: int = OUTER_POLICY_INPUT_DIM, lr: float = 0.01,
                 weight_decay: float = 0.001, max_weight_norm: float = 6.0):
        self.input_dim = input_dim
        self.lr = lr
        # Anti-runaway regularization (2026-06-11). The original train_step
        # clipped the per-step GRADIENT but had NO weight decay, so repeated
        # same-direction REINFORCE updates (the off-policy `tool` attribution
        # credits `tool` on EVERY verified tool-use) let `tool`'s weights grow
        # unbounded → scores ~1100 vs ~24 → argmax collapsed to always-`tool`,
        # feature-independent (verified live T3). weight_decay pulls weights
        # toward 0 each step (bounded equilibrium); max_weight_norm is a hard
        # per-matrix Frobenius cap (backstop). 0 disables either.
        self.weight_decay = float(weight_decay)
        self.max_weight_norm = float(max_weight_norm)
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

    # -- instrumentation (full-MSL-vs-curated assessment, Maker Ad-3) ----
    def feature_contributions(self, x) -> dict:
        """Per-feature influence proxy = |x_i| · ‖w1_row_i‖₁ (input magnitude ×
        first-layer fan-out) + the BASE / MSL / RETRIEVAL grouped shares. Logged
        periodically (NOT on the hot path) so we can assess over time whether the
        full 20D MSL block earns its keep vs a curated subset — and fall back to
        the curated subset only if the data says so (Q4 / Maker Ad-3)."""
        x = np.asarray(x, dtype=np.float32)
        w1_fanout = np.sum(np.abs(self.w1), axis=1)  # (input_dim,)
        contrib = np.abs(x) * w1_fanout
        total = float(contrib.sum()) + 1e-8
        n_base = len(_BASE_FEATURE_NAMES)
        n_msl = MSL_CONTEXT_DIM
        groups = {
            "base": float(contrib[:n_base].sum()) / total,
            "msl_context": float(contrib[n_base:n_base + n_msl].sum()) / total,
            "retrieval": float(contrib[n_base + n_msl:].sum()) / total,
        }
        per_feature = {
            name: float(contrib[i]) for i, name in enumerate(OUTER_FEATURE_NAMES)
        }
        return {"groups": groups, "per_feature": per_feature, "total": total}

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
        # Anti-runaway (2026-06-11): decoupled L2 weight decay (weights only,
        # not biases) + a hard per-matrix Frobenius cap. Without these the
        # off-policy `tool` attribution drove the weights to ~1100 → always-tool.
        if self.weight_decay:
            self.w1 *= (1.0 - self.weight_decay)
            self.w2 *= (1.0 - self.weight_decay)
            self.w3 *= (1.0 - self.weight_decay)
        self._clip_weight_norms()
        self.total_updates += 1
        return float(-np.log(probs[action] + 1e-8) * abs(advantage))

    def _clip_weight_norms(self) -> None:
        """Hard backstop against runaway: cap each weight matrix's Frobenius
        norm at max_weight_norm (rescale if exceeded). Biases are untouched."""
        cap = self.max_weight_norm
        if cap <= 0:
            return
        for attr in ("w1", "w2", "w3"):
            w = getattr(self, attr)
            n = float(np.linalg.norm(w))
            if n > cap:
                w *= (cap / n)

    def is_pathological(self, max_abs_score: float = 100.0) -> bool:
        """Detect a runaway/collapsed policy (the unregularized-REINFORCE
        failure mode): a sane policy produces bounded scores on a neutral
        input; a collapsed one (e.g. `tool` ~1100) blows past max_abs_score.
        Used at worker load to SELF-HEAL (re-init) a persisted runaway."""
        try:
            x = np.full(self.input_dim, 0.3, dtype=np.float32)
            x[0] = 1.0  # bias
            s = self.forward(x)
            return bool((not np.all(np.isfinite(s)))
                        or float(np.max(np.abs(s))) > float(max_abs_score))
        except Exception:
            return False

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
    "OUTER_POLICY_INPUT_DIM", "OUTER_POLICY_FLAT_DIM", "MSL_CONTEXT_DIM",
    "OUTER_META_POLICY_STATE_SLOT", "OUTER_META_POLICY_STATE_SPEC",
    "OUTER_META_POLICY_STATE_SCHEMA_VERSION",
    "OUTER_MSL_CONTEXT_STATE_SLOT", "OUTER_MSL_CONTEXT_STATE_SPEC",
    "OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION",
    "msl_context_to_fixed", "read_msl_context",
    "OuterFeatures", "OuterMetaPolicy",
    "action_index_to_mode", "action_index_to_name",
)
