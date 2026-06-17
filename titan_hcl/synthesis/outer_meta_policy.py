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
import time
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


# ── Structural verifier for idle-exploration reward (RLVR, §7.C tuning step 2) ─
# The verifiable structural ORACLE that scores a Boltzmann-explored action on an
# idle tick (off the live path — INV-OML-9). It is NOT the live router (that is
# the learned policy); it provides the exploration reward where no live verdict
# exists, so the under-used actions (esp. IDK, which has NO live reward stream)
# get tried + credited. The IDK/direct axis is the Maker's key insight: IDK-
# correctness is VERIFIABLE — "does a dereferenceable thought exist for this
# context?" is what the recall search over the timechain-anchored thought store
# answers (recall_top_cosine = that search's strength, captured per turn). The
# IDK-vs-research split is metabolic affordability (FC-6): research costs a
# request — go find out if affordable, else honestly say IDK. Mirrors
# grounded_route's ladder in OML feature-space (single semantic, exploration-
# reward-only — drift-guarded by the cited indices).
def structural_target_action(
        features, *, affordable: bool = True,
        know_threshold: float = 0.5, skill_floor: float = 0.3) -> int:
    """Return the structurally-appropriate action index for a feature vector.

    Reuse-first ladder (serves GB8 reuse-over-deferral + G5 honest-IDK):
      1. a proficient learned skill matches      → skill_delegate  (reuse, no re-run)
      2. computational shape (tool-intent/code)  → tool            (verify deterministically)
      3. dereferenceable knowledge present       → direct          (he KNOWS — recall hit)
      4. does NOT know, research affordable      → research        (find out, 1 request)
      5. does NOT know, not affordable           → IDK             (honest non-answer)
    """
    f = np.asarray(features, dtype=np.float32).ravel()
    if f.shape[0] < len(_BASE_FEATURE_NAMES):
        return OUTER_ACTIONS.index("IDK")
    recall_top = float(f[1])      # recall_top_cosine — the memory-search strength
    skill_util = float(f[3])      # skill_utility
    skill_matched = float(f[4]) >= 0.5
    requires_tool = float(f[6]) >= 0.5
    has_code = float(f[7]) >= 0.5
    if skill_matched and skill_util >= skill_floor:
        return OUTER_ACTIONS.index("skill_delegate")
    if requires_tool or has_code:
        return OUTER_ACTIONS.index("tool")
    if recall_top >= know_threshold:
        return OUTER_ACTIONS.index("direct")
    return OUTER_ACTIONS.index("research" if affordable else "IDK")


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

# ── Canonical-IQL learner dims (RFP_emergent_mastery_curriculum P2) ───
# A learned V(s) + Q(s,a) + Polyak Q-target, mirroring the CGN-IQL ALGORITHM
# (logic/cgn.py:_iql_update_consumer) BY HAND in numpy — never importing the
# torch classes (INV-MC-6 / INV-OML-8). The π net (w1/w2/w3) above is unchanged
# in shape; with IQL on it is updated by advantage-weighted-regression (AWR)
# instead of REINFORCE. These are TRAINING-only (self_learning_worker); the
# agno DECIDE path reads only the π net from SHM, so to_flat/from_flat + the
# SHM slot are byte-identical to legacy (the IQL nets persist separately).
# Fixed (not config) so the iql-flat layout never drifts.
_IQL_VH = 24   # V-net hidden width  (in → _IQL_VH → 1)
_IQL_QH = 24   # Q-net hidden width  (in → _IQL_QH → NUM_OUTER_ACTIONS)

# iql-flat layout (worker-only durable artifact, NOT the SHM slot):
#   V:  vw1[d,VH] vb1[VH] vw2[VH,1] vb2[1]
#   Q:  qw1[d,QH] qb1[QH] qw2[QH,A] qb2[A]
#   Qt: (same shape as Q)
#   meta: total_iql_updates (1)
_IQL_V_N = (OUTER_POLICY_INPUT_DIM * _IQL_VH) + _IQL_VH + (_IQL_VH * 1) + 1
_IQL_Q_N = (OUTER_POLICY_INPUT_DIM * _IQL_QH) + _IQL_QH + (_IQL_QH * NUM_OUTER_ACTIONS) + NUM_OUTER_ACTIONS
IQL_FLAT_DIM = _IQL_V_N + 2 * _IQL_Q_N + 1

# Flat SHM layout: all weights/biases (row-major) then the metadata tail
# (total_updates, reward_baseline, *reward_baseline_per_action). Reconstructed by
# from_flat() using the constant dims below — fixed size, so a fixed float32
# RegistrySpec fits. The PER-ACTION baseline tail (v3) is what dissolves the
# always-tool routing deadlock: with one GLOBAL baseline the dense off-policy
# `tool` +1 stream saturated the baseline to ~1.0, so a POSITIVE direct/research
# reward (e.g. +0.5) became a NEGATIVE advantage (0.5−1.0) and the policy
# actively suppressed non-tool actions (verified live T1: direct/research scores
# ~−0.15, baseline=1.0). A baseline PER ACTION measures each action's advantage
# against its OWN running mean → tool's +1 stream raises only tool's baseline
# (its advantage → 0, no monopoly) while direct's +0.5 stays a positive advantage
# vs direct's own (lower) baseline. (§24.3 residual fix; RFP §7.C tuning step 1.)
_W1_N = OUTER_POLICY_INPUT_DIM * _H1
_W2_N = _H1 * _H2
_W3_N = _H2 * NUM_OUTER_ACTIONS
# metadata tail = total_updates + reward_baseline (global EMA, telemetry/back-compat)
# + NUM_OUTER_ACTIONS per-action baselines.
_META_TAIL_N = 2 + NUM_OUTER_ACTIONS
OUTER_POLICY_FLAT_DIM = (_W1_N + _H1 + _W2_N + _H2 + _W3_N + NUM_OUTER_ACTIONS) + _META_TAIL_N

OUTER_META_POLICY_STATE_SLOT = "outer_meta_policy_state"
OUTER_META_POLICY_STATE_SCHEMA_VERSION = 3  # v3: per-action REINFORCE baseline (deadlock fix); v2 was 30-D MSL/retrieval
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


class OuterCompositeReader:
    """Phase-C piece 7b — the parametric retrieval prior (D4), lock-free.

    SC-searches the live prompt against the MACRO reasoning-composites and returns
    `(composite_match_score, composite_match_action_norm)` for the agno decide.

    Cross-process design (SPEC-canonical, ARCHITECTURE_synthesis_engine §0.11.0 FU-1
    / lesson 10): DuckDB 1.5+ holds the exclusive lock even for `read_only` opens, so
    the agno process CANNOT open `reasoning.duckdb`. The faiss FILE is read-only-safe;
    the `embedding_id→action` map rides the lock-free `reasoning_snapshot.json` (the
    `snapshot_export` "readable past the writer-lock" surface). Both cached + mtime/
    interval-refreshed; faiss is tiny (few macros). Reuses the PreHook's already-
    computed normalized `_prompt_vec` (no extra embed). Any miss → (0.0, 0.0)."""

    def __init__(self, faiss_path: str, snapshot_path: str, refresh_s: float = 60.0):
        self._faiss_path = str(faiss_path)
        self._snapshot_path = str(snapshot_path)
        self._refresh_s = float(refresh_s)
        self._index = None
        self._macros: dict = {}        # embedding_id -> action_idx
        self._weights: dict = {}       # embedding_id -> match weight ∈ [0,1] (§24.12 Track 2)
        self._meta: dict = {}          # §7.E embedding_id -> {goal_class, recipe_json, source, reasoning_id} (macro composites only)
        self._next_refresh = 0.0

    def _refresh(self, now: float) -> None:
        if self._index is not None and now < self._next_refresh:
            return
        self._next_refresh = now + self._refresh_s
        try:
            import faiss  # local — keep faiss out of agno's import-time RSS
            if os.path.exists(self._faiss_path):
                self._index = faiss.read_index(self._faiss_path)
        except Exception:
            self._index = None
        macros: dict = {}
        weights: dict = {}
        meta: dict = {}
        try:
            if os.path.exists(self._snapshot_path):
                with open(self._snapshot_path) as f:
                    snap = json.load(f)
                # §24.12 Track 2 — the EMERGENT prior: Titan's own oracle-VERIFIED
                # tool_use experience (reward>0 wins), reward-weighted. Loaded FIRST
                # so the rare hand-distilled `macro_strategy` composites (loaded
                # next, weight 1.0) OVERLAY them — the refined distillate wins on a
                # shared embedding_id, the verified experience fills the rest.
                for v in (snap.get("verified_priors") or []):
                    eid = int(v.get("embedding_id", -1))
                    act = str(v.get("action", "") or "")
                    if eid >= 0 and act in OUTER_ACTIONS:
                        macros[eid] = OUTER_ACTIONS.index(act)
                        weights[eid] = _clip01(float(v.get("reward", 0.0) or 0.0))
                for m in (snap.get("macros") or []):
                    eid = int(m.get("embedding_id", -1))
                    act = str(m.get("action", "") or "")
                    if eid >= 0 and act in OUTER_ACTIONS:
                        macros[eid] = OUTER_ACTIONS.index(act)
                        weights[eid] = 1.0
                        # §7.E — the replay recipe (E.1) / research source (E.3) /
                        # goal_class, so the agno fast path can deref a matched
                        # composite lock-free (no DuckDB open).
                        meta[eid] = {
                            "goal_class": str(m.get("goal_class", "") or ""),
                            "recipe_json": str(m.get("recipe_json", "") or ""),
                            "source": str(m.get("source", "") or ""),
                            "reasoning_id": str(m.get("reasoning_id", "") or ""),
                            "action": act,
                        }
        except Exception:
            macros = {}
            weights = {}
            meta = {}
        self._macros = macros
        self._weights = weights
        self._meta = meta

    def prior(self, prompt_vec, now: Optional[float] = None) -> tuple:
        """Return `(score, action_norm)` ∈ [0,1]² for the top matched macro
        composite, or `(0.0, 0.0)` on any miss (cold-start / no macros / unreadable)."""
        if now is None:
            now = time.time()
        self._refresh(now)
        if (self._index is None or not self._macros
                or getattr(self._index, "ntotal", 0) == 0 or prompt_vec is None):
            return (0.0, 0.0)
        try:
            v = np.asarray(prompt_vec, dtype=np.float32).reshape(1, -1)
            if v.shape[1] != self._index.d:
                return (0.0, 0.0)
            k = min(10, int(self._index.ntotal))
            dists, ids = self._index.search(v, k)
            # §24.12 Track 2 — among ALL matched neighbors in the top-k, pick the
            # one maximizing cos × weight (reward-weighted). For macro_strategy
            # (weight 1.0) this reduces to the nearest match = the prior behaviour
            # (faiss returns nearest-first), so macro-only snapshots are unchanged.
            best_score, best_action = -1.0, None
            for i in range(k):
                eid = int(ids[0][i])
                if eid in self._macros:
                    # IndexFlatL2 returns squared-L2; for a normalized embedder
                    # cos = 1 − L2²/2 (get_text_embedder normalizes — agno embed-once).
                    cos = 1.0 - float(dists[0][i]) / 2.0
                    score = _clip01(cos) * _clip01(self._weights.get(eid, 1.0))
                    if score > best_score:
                        best_score, best_action = score, self._macros[eid]
            if best_action is not None:
                a_norm = best_action / max(1, NUM_OUTER_ACTIONS - 1)
                return (_clip01(best_score), _clip01(a_norm))
        except Exception:
            return (0.0, 0.0)
        return (0.0, 0.0)

    def match(self, prompt_vec, now: Optional[float] = None) -> Optional[dict]:
        """§7.E (E1/E3) — the top matched MACRO COMPOSITE with its replay metadata,
        or None on any miss. Returns `{score, action, goal_class, recipe_json,
        source, reasoning_id}`. Used by the agno fast path to actually EXECUTE a
        matched composite (E.1 recipe replay / E.3 research-source direct call) —
        `prior()` stays the policy-feature surface (unchanged). Macro composites
        only (the `_meta` map); verified_priors have no recipe → not matchable here."""
        if now is None:
            now = time.time()
        self._refresh(now)
        if (self._index is None or not self._meta
                or getattr(self._index, "ntotal", 0) == 0 or prompt_vec is None):
            return None
        try:
            v = np.asarray(prompt_vec, dtype=np.float32).reshape(1, -1)
            if v.shape[1] != self._index.d:
                return None
            k = min(10, int(self._index.ntotal))
            dists, ids = self._index.search(v, k)
            best_score, best_eid = -1.0, None
            for i in range(k):
                eid = int(ids[0][i])
                if eid in self._meta:           # macro composites only
                    cos = 1.0 - float(dists[0][i]) / 2.0
                    score = _clip01(cos) * _clip01(self._weights.get(eid, 1.0))
                    if score > best_score:
                        best_score, best_eid = score, eid
            if best_eid is not None:
                m = self._meta[best_eid]
                return {"score": _clip01(best_score), "action": m["action"],
                        "goal_class": m["goal_class"], "recipe_json": m["recipe_json"],
                        "source": m["source"], "reasoning_id": m["reasoning_id"]}
        except Exception:
            return None
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
        # PER-ACTION REINFORCE baseline (v3, the deadlock fix). Each action's
        # advantage is measured against ITS OWN running mean — so the dense
        # off-policy `tool` +1 stream can no longer drive a positive non-tool
        # reward into negative advantage (the always-tool collapse, §24.3).
        self.reward_baseline_per_action = np.zeros(NUM_OUTER_ACTIONS, dtype=np.float32)
        # Global EMA over ALL rewards — retained for telemetry / back-compat
        # (the worker log line, the GB3 "baseline moves" gate, the diagnostic).
        self.reward_baseline = 0.0
        self._cache: dict = {}
        # ── Canonical-IQL learner state (P2) — lazily allocated by init_iql()
        # so a flag-off policy is byte-identical to legacy. V/Q/Q-target nets
        # + their own update counter. Training-only; never published to SHM.
        self._iql_inited: bool = False
        self.total_iql_updates: int = 0
        self._vw1 = self._vb1 = self._vw2 = self._vb2 = None
        self._qw1 = self._qb1 = self._qw2 = self._qb2 = None
        self._qtw1 = self._qtb1 = self._qtw2 = self._qtb2 = None

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
        """One REINFORCE-with-baseline update. The advantage uses the CURRENT
        baseline OF THIS ACTION (v3 — not the global one), then that action's
        baseline moves toward the observed reward. This is what breaks the
        always-tool deadlock: a positive direct/research reward stays a positive
        advantage even while `tool` is being flooded with +1 (its own baseline
        absorbs that, leaving direct's advantage untouched). The global EMA is
        still tracked for telemetry / the GB3 gate."""
        r = float(reward)
        a = int(action)
        if 0 <= a < NUM_OUTER_ACTIONS:
            advantage = r - float(self.reward_baseline_per_action[a])
        else:  # defensive — out-of-range action falls back to the global baseline
            advantage = r - self.reward_baseline
        loss = self.train_step(x, a, advantage)
        if 0 <= a < NUM_OUTER_ACTIONS:
            self.reward_baseline_per_action[a] += baseline_alpha * (
                r - float(self.reward_baseline_per_action[a]))
        self.reward_baseline += baseline_alpha * (r - self.reward_baseline)
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
            "reward_baseline_per_action": self.reward_baseline_per_action.tolist(),
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
            _bpa = d.get("reward_baseline_per_action")
            if _bpa is not None and len(_bpa) == NUM_OUTER_ACTIONS:
                self.reward_baseline_per_action = np.array(_bpa, dtype=np.float32)
            else:  # pre-v3 artifact — seed all per-action baselines from the global
                self.reward_baseline_per_action = np.full(
                    NUM_OUTER_ACTIONS, self.reward_baseline, dtype=np.float32)
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
            np.asarray(self.reward_baseline_per_action, dtype=np.float32).ravel(),
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
        p.reward_baseline_per_action = take(NUM_OUTER_ACTIONS).astype(np.float32).copy()
        return p

    # ── Canonical-IQL learner (RFP_emergent_mastery_curriculum P2) ───────
    # Mirrors the CGN-IQL ALGORITHM (logic/cgn.py:_iql_update_consumer) by hand
    # in numpy: expectile-V regression toward Q_target(s,a), Bellman-Q backup
    # with V(s') (never max_a Q → no OOD), AWR policy extraction, Polyak target.
    # FULL IQL (not bandit): transitions carry next_state/terminal. INV-MC-6.

    def init_iql(self) -> None:
        """Lazily allocate the V/Q/Q-target nets (He init). Idempotent — a no-op
        once allocated (so a load via iql_from_flat is not clobbered)."""
        if self._iql_inited:
            return
        d = self.input_dim
        sv = math.sqrt(2.0 / d)
        sq = math.sqrt(2.0 / d)
        svo = math.sqrt(2.0 / _IQL_VH)
        sqo = math.sqrt(2.0 / _IQL_QH)
        self._vw1 = (np.random.randn(d, _IQL_VH).astype(np.float32) * sv)
        self._vb1 = np.zeros(_IQL_VH, dtype=np.float32)
        self._vw2 = (np.random.randn(_IQL_VH, 1).astype(np.float32) * svo)
        self._vb2 = np.zeros(1, dtype=np.float32)
        self._qw1 = (np.random.randn(d, _IQL_QH).astype(np.float32) * sq)
        self._qb1 = np.zeros(_IQL_QH, dtype=np.float32)
        self._qw2 = (np.random.randn(_IQL_QH, NUM_OUTER_ACTIONS).astype(np.float32) * sqo)
        self._qb2 = np.zeros(NUM_OUTER_ACTIONS, dtype=np.float32)
        # Polyak target = exact copy of Q at init.
        self._qtw1 = self._qw1.copy()
        self._qtb1 = self._qb1.copy()
        self._qtw2 = self._qw2.copy()
        self._qtb2 = self._qb2.copy()
        self._iql_inited = True

    @staticmethod
    def _mlp2_forward(X, w1, b1, w2, b2):
        """Batched 2-layer ReLU MLP forward. Returns (out, cache)."""
        z1 = X @ w1 + b1
        h1 = np.maximum(0.0, z1)
        out = h1 @ w2 + b2
        return out, (X, z1, h1)

    @staticmethod
    def _mlp2_backward(d_out, w2, cache):
        """Batched backprop for _mlp2_forward. d_out is ∂L/∂out [B, out_dim]
        already scaled by 1/B. Returns (dw1, db1, dw2, db2)."""
        X, z1, h1 = cache
        dw2 = h1.T @ d_out
        db2 = d_out.sum(axis=0)
        dh1 = d_out @ w2.T
        dz1 = dh1 * (z1 > 0)
        dw1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return dw1, db1, dw2, db2

    @staticmethod
    def _clip_grad_norm(grads, max_norm: float) -> None:
        """Global-norm gradient clip across the listed arrays (mirrors CGN's
        torch.nn.utils.clip_grad_norm_). Rescales in place if exceeded."""
        if max_norm <= 0:
            return
        total = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if total > max_norm and total > 1e-12:
            scale = max_norm / total
            for g in grads:
                g *= scale

    def train_iql(self, transitions, *, tau: float = 0.7, beta: float = 3.0,
                  gamma: float = 0.99, polyak: float = 0.005,
                  adv_clip: float = 100.0, lr: float = 0.003,
                  steps: int = 20, batch_size: int = 32,
                  max_grad_norm: float = 1.0) -> dict:
        """One canonical-IQL consolidation pass over `transitions` (offline,
        idle-tick only). Each transition is a dict/tuple with keys/fields
        (state, action, reward, next_state, terminal). next_state may be None
        when terminal. Updates the π net (w1/w2/w3) by AWR. Returns loss stats.

        Mirrors cgn._iql_update_consumer exactly (numpy-by-hand):
          1. V ← expectile_τ regression toward Q_target(s, a_taken)
          2. Q ← Bellman backup r + γ(1−terminal)·V(s')   (never max_a Q)
          3. π ← AWR: weight=clamp(exp(β·(Q(s,a)−V(s))), max=adv_clip)
          4. Polyak: Q_target ← (1−polyak)Q_target + polyak·Q
        """
        self.init_iql()
        n = len(transitions)
        if n < 2:
            return {"skipped": True, "transitions": n}

        def _state(t):
            return np.asarray(t["state"], dtype=np.float32)

        def _next(t):
            ns = t.get("next_state")
            return (np.zeros(self.input_dim, dtype=np.float32)
                    if ns is None else np.asarray(ns, dtype=np.float32))

        bs = min(batch_size, n)
        tot_v = tot_q = tot_p = 0.0
        for _ in range(int(steps)):
            idx = np.random.choice(n, bs, replace=False)
            batch = [transitions[i] for i in idx]
            S = np.array([_state(t) for t in batch], dtype=np.float32)        # [B,in]
            S2 = np.array([_next(t) for t in batch], dtype=np.float32)         # [B,in]
            A = np.array([int(t["action"]) for t in batch], dtype=np.int64)    # [B]
            R = np.array([float(t["reward"]) for t in batch], dtype=np.float32)
            NT = np.array([0.0 if t.get("terminal") else 1.0 for t in batch],
                          dtype=np.float32)                                    # nonterminal
            B = S.shape[0]
            rows = np.arange(B)

            # 1. V update — expectile regression toward Q_target(s, a_taken).
            q_tgt_all, _ = self._mlp2_forward(S, self._qtw1, self._qtb1,
                                              self._qtw2, self._qtb2)          # [B,A]
            q_sa_tgt = q_tgt_all[rows, A]                                      # [B]
            v_pred, v_cache = self._mlp2_forward(S, self._vw1, self._vb1,
                                                 self._vw2, self._vb2)         # [B,1]
            v_pred = v_pred.reshape(B)
            diff = q_sa_tgt - v_pred                                           # target − pred
            w = np.where(diff < 0.0, 1.0 - tau, tau).astype(np.float32)
            v_loss = float(np.mean(w * diff * diff))
            d_v = (-2.0 * w * diff / B).reshape(B, 1).astype(np.float32)       # ∂L/∂V_pred
            dvw1, dvb1, dvw2, dvb2 = self._mlp2_backward(d_v, self._vw2, v_cache)
            self._clip_grad_norm([dvw1, dvb1, dvw2, dvb2], max_grad_norm)
            self._vw1 -= lr * dvw1; self._vb1 -= lr * dvb1
            self._vw2 -= lr * dvw2; self._vb2 -= lr * dvb2

            # 2. Q update — Bellman backup with V(s') (never max_a Q → no OOD).
            v_next, _ = self._mlp2_forward(S2, self._vw1, self._vb1,
                                           self._vw2, self._vb2)
            v_next = v_next.reshape(B) * NT
            q_backup = R + gamma * v_next                                      # [B]
            q_all, q_cache = self._mlp2_forward(S, self._qw1, self._qb1,
                                                self._qw2, self._qb2)          # [B,A]
            q_sa = q_all[rows, A]
            q_err = q_sa - q_backup
            q_loss = float(np.mean(q_err * q_err))
            d_q = np.zeros((B, NUM_OUTER_ACTIONS), dtype=np.float32)
            d_q[rows, A] = (2.0 * q_err / B).astype(np.float32)               # taken-action head only
            dqw1, dqb1, dqw2, dqb2 = self._mlp2_backward(d_q, self._qw2, q_cache)
            self._clip_grad_norm([dqw1, dqb1, dqw2, dqb2], max_grad_norm)
            self._qw1 -= lr * dqw1; self._qb1 -= lr * dqb1
            self._qw2 -= lr * dqw2; self._qb2 -= lr * dqb2

            # 3. Policy update — advantage-weighted regression (AWR) on the π net.
            q_all2, _ = self._mlp2_forward(S, self._qw1, self._qb1,
                                           self._qw2, self._qb2)
            v_pred2, _ = self._mlp2_forward(S, self._vw1, self._vb1,
                                            self._vw2, self._vb2)
            adv = q_all2[rows, A] - v_pred2.reshape(B)                         # [B]
            weight = np.minimum(np.exp(beta * adv), adv_clip).astype(np.float32)
            # π forward (batched) — own cache; does NOT touch self._cache (REINFORCE path).
            z1 = S @ self.w1 + self.b1; h1 = np.maximum(0.0, z1)
            z2 = h1 @ self.w2 + self.b2; h2 = np.maximum(0.0, z2)
            z3 = h2 @ self.w3 + self.b3                                        # logits [B,A]
            z3 = z3 - z3.max(axis=1, keepdims=True)
            exp_s = np.exp(z3)
            probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-8)
            target = np.zeros((B, NUM_OUTER_ACTIONS), dtype=np.float32)
            target[rows, A] = 1.0
            # ∂L/∂logits for L = mean(−weight·logπ(a)) = (1/B)·weight·(probs − onehot)
            d_z3 = (probs - target) * (weight.reshape(B, 1) / B)
            d_w3 = h2.T @ d_z3; d_b3 = d_z3.sum(axis=0)
            d_h2 = d_z3 @ self.w3.T; d_z2 = d_h2 * (z2 > 0)
            d_w2 = h1.T @ d_z2; d_b2 = d_z2.sum(axis=0)
            d_h1 = d_z2 @ self.w2.T; d_z1 = d_h1 * (z1 > 0)
            d_w1 = S.T @ d_z1; d_b1 = d_z1.sum(axis=0)
            self._clip_grad_norm([d_w1, d_b1, d_w2, d_b2, d_w3, d_b3], max_grad_norm)
            policy_loss = float(np.mean(-weight * np.log(probs[rows, A] + 1e-8)))
            self.w1 -= lr * d_w1; self.b1 -= lr * d_b1
            self.w2 -= lr * d_w2; self.b2 -= lr * d_b2
            self.w3 -= lr * d_w3; self.b3 -= lr * d_b3
            if self.weight_decay:
                self.w1 *= (1.0 - self.weight_decay)
                self.w2 *= (1.0 - self.weight_decay)
                self.w3 *= (1.0 - self.weight_decay)
            self._clip_weight_norms()

            # 4. Polyak target update: Q_target ← (1−polyak)Q_target + polyak·Q.
            self._qtw1 = (1.0 - polyak) * self._qtw1 + polyak * self._qw1
            self._qtb1 = (1.0 - polyak) * self._qtb1 + polyak * self._qb1
            self._qtw2 = (1.0 - polyak) * self._qtw2 + polyak * self._qw2
            self._qtb2 = (1.0 - polyak) * self._qtb2 + polyak * self._qb2

            self.total_updates += 1
            self.total_iql_updates += 1
            tot_v += v_loss; tot_q += q_loss; tot_p += policy_loss

        s = max(1, int(steps))
        return {"v_loss": tot_v / s, "q_loss": tot_q / s,
                "policy_loss": tot_p / s, "transitions": n,
                "iql_updates": self.total_iql_updates}

    def value(self, x) -> float:
        """V(s) — the learned expectile state-value (P3 MasteryLevel reads this).
        0.0 if IQL not yet initialised."""
        if not self._iql_inited:
            return 0.0
        out, _ = self._mlp2_forward(
            np.asarray(x, dtype=np.float32).reshape(1, -1),
            self._vw1, self._vb1, self._vw2, self._vb2)
        return float(out.reshape(-1)[0])

    def iql_to_flat(self) -> np.ndarray:
        """Pack the V/Q/Q-target nets + iql metadata into a flat float32 vector
        (worker-only durable artifact; NOT the SHM slot). init_iql() first so a
        never-trained policy still round-trips a valid (random-init) net."""
        self.init_iql()
        return np.concatenate([
            self._vw1.ravel(), self._vb1.ravel(), self._vw2.ravel(), self._vb2.ravel(),
            self._qw1.ravel(), self._qb1.ravel(), self._qw2.ravel(), self._qb2.ravel(),
            self._qtw1.ravel(), self._qtb1.ravel(), self._qtw2.ravel(), self._qtb2.ravel(),
            np.array([float(self.total_iql_updates)], dtype=np.float32),
        ]).astype(np.float32)

    def iql_from_flat(self, flat) -> bool:
        """Restore the IQL nets from iql_to_flat(). Returns False (and leaves a
        fresh init) on any size/shape mismatch (schema drift → relearn)."""
        try:
            flat = np.asarray(flat, dtype=np.float32).ravel()
            if flat.shape[0] != IQL_FLAT_DIM:
                return False
            d = self.input_dim
            i = 0
            def take(n):
                nonlocal i
                seg = flat[i:i + n]; i += n
                return seg
            self._vw1 = take(d * _IQL_VH).reshape(d, _IQL_VH).astype(np.float32)
            self._vb1 = take(_IQL_VH).astype(np.float32)
            self._vw2 = take(_IQL_VH * 1).reshape(_IQL_VH, 1).astype(np.float32)
            self._vb2 = take(1).astype(np.float32)
            self._qw1 = take(d * _IQL_QH).reshape(d, _IQL_QH).astype(np.float32)
            self._qb1 = take(_IQL_QH).astype(np.float32)
            self._qw2 = take(_IQL_QH * NUM_OUTER_ACTIONS).reshape(
                _IQL_QH, NUM_OUTER_ACTIONS).astype(np.float32)
            self._qb2 = take(NUM_OUTER_ACTIONS).astype(np.float32)
            self._qtw1 = take(d * _IQL_QH).reshape(d, _IQL_QH).astype(np.float32)
            self._qtb1 = take(_IQL_QH).astype(np.float32)
            self._qtw2 = take(_IQL_QH * NUM_OUTER_ACTIONS).reshape(
                _IQL_QH, NUM_OUTER_ACTIONS).astype(np.float32)
            self._qtb2 = take(NUM_OUTER_ACTIONS).astype(np.float32)
            self.total_iql_updates = int(round(float(take(1)[0])))
            self._iql_inited = True
            return True
        except Exception:
            self._iql_inited = False
            self.init_iql()
            return False


__all__ = (
    "OUTER_ACTIONS", "NUM_OUTER_ACTIONS", "OUTER_FEATURE_NAMES",
    "OUTER_POLICY_INPUT_DIM", "OUTER_POLICY_FLAT_DIM", "IQL_FLAT_DIM",
    "MSL_CONTEXT_DIM",
    "OUTER_META_POLICY_STATE_SLOT", "OUTER_META_POLICY_STATE_SPEC",
    "OUTER_META_POLICY_STATE_SCHEMA_VERSION",
    "OUTER_MSL_CONTEXT_STATE_SLOT", "OUTER_MSL_CONTEXT_STATE_SPEC",
    "OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION",
    "msl_context_to_fixed", "read_msl_context",
    "OuterFeatures", "OuterMetaPolicy", "OuterCompositeReader",
    "action_index_to_mode", "action_index_to_name",
    "structural_target_action",
)
