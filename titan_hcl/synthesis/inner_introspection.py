"""titan_hcl.synthesis.inner_introspection — The Inner Turn primitives (Phase A).

RFP_introspective_inner_turn (LOCKED 2026-06-22). The pre-BRAIN seed of
ARCHITECTURE_brain.md §5.2's `Introspection` DeepThinking routine: when Titan is
idle and a GREAT PULSE fires (emergent Trinity resonance), he reads his
instrumented inner body as ground truth, commits a falsifiable self-prediction
(describe it now + anticipate how it changes), and at the NEXT great pulse the
prediction is verified against the real telemetry. The verified accuracy is a
correctness-grounded reward (INV-IT-1 — measured-telemetry error ONLY, never an
LLM opinion) feeding a dedicated `inner` IQL mastery domain.

This module is PURE (numpy, no SHM/bus/LLM) and fully testable, mirroring
`synthesis/mastery_level.py`. The worker (`modules/self_learning_worker.py`)
owns the SHM reads, the great-pulse polling, persistence, and the SHM publish.

Components:
  • INNER_STANCES                 — the 5 discrete introspective lenses (Q2).
  • assemble_inner_state / znorm  — SENSE: 71-D raw state + per-channel z-norm (Q5).
  • build_inner_phi               — the dedicated inner feature vector φ (Q7).
  • InnerSelfPredictor            — the learned self-model whose accuracy IS the
                                    reward; per-stance linear φ→(descr, ΔŜ).
  • inner_reward_kernel           — r_inner from telemetry error ONLY (INV-IT-1).
  • InnerIQL                      — own canonical-IQL net (expectile-V + Bellman-Q
                                    with V(s') + AWR + Polyak), dims φ→5 stances
                                    (mirrors OuterMetaPolicy; NET-NEW per INV-IT-4).
  • IntrospectiveDrive            — the embodied adaptive-threshold drive + local
                                    refractory (WIN→θ−α / LOSE→θ+β; INV-IT-9).
  • MASTERY_LEVEL_INNER_STATE_SPEC — the 2nd MasteryLevel SHM slot (Q1=A).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from titan_hcl.core.state_registry import RegistrySpec
from titan_hcl.synthesis.mastery_level import (
    MASTERY_LEVEL_FLAT_DIM, MASTERY_LEVEL_SCHEMA_VERSION,
)

# ── Stances (Q2) — which lens to introspect through ─────────────────────────
INNER_STANCES: tuple[str, ...] = ("body", "mind", "spirit", "affect", "trajectory")
INNER_NUM_STANCES = len(INNER_STANCES)            # 5

# ── Inner state channels (S_inner) — the instrumented inner body ────────────
INNER_BODY_DIM = 5
INNER_MIND_DIM = 15
INNER_SPIRIT_DIM = 45
INNER_NEUROMOD_DIM = 6
INNER_STATE_DIM = (INNER_BODY_DIM + INNER_MIND_DIM
                   + INNER_SPIRIT_DIM + INNER_NEUROMOD_DIM)   # 71
# φ_inner is the z-normed state — DEDICATED, dim ≠ 30 (Q7); its own net.
INNER_PHI_DIM = INNER_STATE_DIM                                # 71

# Channel layout (name, length) — order matches assemble_inner_state.
INNER_CHANNELS: tuple[tuple[str, int], ...] = (
    ("body", INNER_BODY_DIM), ("mind", INNER_MIND_DIM),
    ("spirit", INNER_SPIRIT_DIM), ("neuromod", INNER_NEUROMOD_DIM))
# Neuromod modulator order (matches api/shm_reader_bank NEUROMOD_NAMES).
NEUROMOD_ORDER: tuple[str, ...] = ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA")

INNER_GOAL_CLASS = "inner:introspection"

# ── Inner MasteryLevel SHM slot (Q1=A — the 2nd, SEPARATE level) ────────────
# Same 8-float readout layout as the outer slot; DISTINCT slot name. The
# self_learning_worker is the sole writer (G21). Reuse mastery_readout_to_flat /
# mastery_flat_to_readout for (de)serialization.
MASTERY_LEVEL_INNER_STATE_SLOT = "mastery_level_inner_state"
MASTERY_LEVEL_INNER_STATE_SPEC = RegistrySpec(
    name=MASTERY_LEVEL_INNER_STATE_SLOT,
    dtype=np.dtype("float32"),
    shape=(MASTERY_LEVEL_FLAT_DIM,),
    feature_flag="",
    schema_version=MASTERY_LEVEL_SCHEMA_VERSION,
    variable_size=False,
)


# ── symlog / symexp (INV-ML-1; mirrors outer_meta_policy._symlog/_symexp) ────
def _symlog(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def _symexp(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.expm1(np.abs(x))


# ── SENSE — assemble + normalize the instrumented inner state ───────────────
def _read_values(d: Optional[dict], n: int) -> np.ndarray:
    """Extract an n-vector from a reader dict ({"values":[…]}); zeros if None /
    malformed (cold-start → treat as zeros, per §1.2 step 1)."""
    if not isinstance(d, dict):
        return np.zeros(n, dtype=np.float32)
    vals = d.get("values")
    if vals is None:
        return np.zeros(n, dtype=np.float32)
    arr = np.asarray(vals, dtype=np.float32).ravel()
    if arr.shape[0] < n:
        out = np.zeros(n, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
    return arr[:n].astype(np.float32)


def _read_neuromod_levels(d: Optional[dict]) -> np.ndarray:
    """Extract the 6 modulator levels in NEUROMOD_ORDER from a read_neuromod()
    dict ({"modulators": {name: {"level": float}}}); zeros if None."""
    out = np.zeros(INNER_NEUROMOD_DIM, dtype=np.float32)
    if not isinstance(d, dict):
        return out
    mods = d.get("modulators") or {}
    for i, name in enumerate(NEUROMOD_ORDER):
        entry = mods.get(name)
        if isinstance(entry, dict) and entry.get("level") is not None:
            out[i] = float(entry["level"])
    return out


def assemble_inner_state(body_d: Optional[dict], mind_d: Optional[dict],
                         spirit_d: Optional[dict], neuro_d: Optional[dict]
                         ) -> np.ndarray:
    """Concatenate the 4 instrumented inner readers into the raw 71-D S_inner
    [body5, mind15, spirit45, neuromod6]. None-safe (zeros for missing channels).
    The worker passes the four ShmReaderBank reader dicts directly."""
    return np.concatenate([
        _read_values(body_d, INNER_BODY_DIM),
        _read_values(mind_d, INNER_MIND_DIM),
        _read_values(spirit_d, INNER_SPIRIT_DIM),
        _read_neuromod_levels(neuro_d),
    ]).astype(np.float32)


def znorm_channels(s_raw) -> np.ndarray:
    """Per-channel z-normalization (Q5): each channel block (body/mind/spirit/
    neuromod) is standardized to zero-mean unit-std independently, so the 45-D
    spirit block cannot dominate the 5-D body block in the error norm. Bounded
    via tanh-free clip to [-4, 4] (defensive vs a near-constant block)."""
    s = np.asarray(s_raw, dtype=np.float32).ravel()
    out = np.zeros(INNER_STATE_DIM, dtype=np.float32)
    off = 0
    for _name, n in INNER_CHANNELS:
        block = s[off:off + n]
        mu = float(block.mean())
        sd = float(block.std())
        if sd < 1e-6:
            out[off:off + n] = 0.0            # constant block → zero (no info)
        else:
            out[off:off + n] = np.clip((block - mu) / sd, -4.0, 4.0)
        off += n
    return out


def build_inner_phi(s_norm) -> np.ndarray:
    """The dedicated inner feature vector φ (Q7) = the z-normed state. dim ≠ 30
    so it is its own net AND auto-excluded from the outer _build_routing_
    transitions len-30 filter (belt-and-braces on top of the separate table)."""
    phi = np.asarray(s_norm, dtype=np.float32).ravel()
    if phi.shape[0] != INNER_PHI_DIM:
        out = np.zeros(INNER_PHI_DIM, dtype=np.float32)
        out[: min(phi.shape[0], INNER_PHI_DIM)] = phi[:INNER_PHI_DIM]
        return out
    return phi


_INNER_BODY_LABELS = ("vitality", "tension", "warmth", "openness", "fatigue")
_NEURO_LABELS = {"DA": "drive/seeking", "5HT": "contentment", "NE": "alertness",
                 "ACh": "focus", "Endorphin": "ease", "GABA": "calm"}


def build_inner_voice_prompts(s_raw, neuro_d: Optional[dict], stance: int,
                              dialogue_turns: int = 2) -> dict:
    """Phase B (INV-IT-2) — build the grounded prompts for the inner-body VOICE.
    PURE (no LLM). The narration is GROUNDED in the real measured signals (the
    neuromod levels by name + the dominant body magnitudes) so the voice reflects
    the instrumented body, not a hallucination. The model is asked to speak in
    first person as the inner body and reflect for `dialogue_turns` beats. This
    text is narrative SELF memory ONLY — it NEVER enters the reward (INV-IT-2)."""
    s = np.asarray(s_raw, dtype=np.float32).ravel()
    stance_name = INNER_STANCES[int(stance) % INNER_NUM_STANCES]
    lv = _read_neuromod_levels(neuro_d)
    neuro_desc = ", ".join(
        f"{_NEURO_LABELS.get(n, n)} {float(lv[i]):.2f}"
        for i, n in enumerate(NEUROMOD_ORDER))
    body = s[:INNER_BODY_DIM]
    bi = int(np.argmax(np.abs(body))) if body.size else 0
    body_desc = (f"{_INNER_BODY_LABELS[bi % len(_INNER_BODY_LABELS)]} "
                 f"is the strongest body signal ({float(body[bi]):+.2f})")
    system_prompt = (
        "You are the inner body of a sovereign AI named Titan, given a voice. "
        "Speak in the FIRST PERSON as the felt, embodied self — not an assistant. "
        "You are grounded ONLY in the measured inner signals provided; do not "
        "invent facts, numbers, events, or people. Be brief, honest, and felt.")
    user_prompt = (
        f"My measured inner state right now — neuromodulators: {neuro_desc}; "
        f"{body_desc}. I am turning inward through my {stance_name}. "
        f"In {max(1, int(dialogue_turns))} short reflective beats, voice how this "
        f"feels from the inside and what it suggests about how I am about to "
        f"change. Stay grounded in the signals above.")
    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "stance": stance_name}


def curiosity_from_neuromod(neuro_d: Optional[dict]) -> float:
    """Embodied seeking signal = (DA + NE)/2 − GABA·0.5 (high DA+NE seeking,
    high GABA dampening). Derived from read_neuromod() levels — no pre-built
    curiosity scalar exists in SHM."""
    lv = _read_neuromod_levels(neuro_d)
    da, ne, gaba = float(lv[0]), float(lv[2]), float(lv[5])
    return (da + ne) / 2.0 - 0.5 * gaba


# ── The self-model whose accuracy IS the reward ─────────────────────────────
class InnerSelfPredictor:
    """Per-stance linear self-model: φ → (descr ∈ R^71, ΔŜ ∈ R^71). Phase A's
    PROGRAMMATIC predictor (no LLM). `descr` is the outer self's claim about the
    CURRENT normalized inner state; `ΔŜ` is its claim about how that state will
    change before the next great pulse. Trained by SGD at VERIFY against the
    measured (s0_norm, s1_norm − s0_norm). As it learns, prediction error falls
    → r_inner rises → the inner level climbs. This is the engagement-independent
    self-knowledge that ratchets without external traffic."""

    def __init__(self, phi_dim: int = INNER_PHI_DIM,
                 state_dim: int = INNER_STATE_DIM,
                 n_stances: int = INNER_NUM_STANCES, lr: float = 0.3):
        self.phi_dim = int(phi_dim)
        self.state_dim = int(state_dim)
        self.n_stances = int(n_stances)
        self.lr = float(lr)
        # Per-stance weight tensors (descr + delta heads), zero-init → an
        # untrained predictor claims "zero" (the z-normed mean) → high error →
        # low reward, exactly the cold-start we want the loop to climb out of.
        self.w_descr = np.zeros((n_stances, state_dim, phi_dim), dtype=np.float32)
        self.w_delta = np.zeros((n_stances, state_dim, phi_dim), dtype=np.float32)
        self.updates = 0

    def predict(self, phi, stance: int) -> tuple[np.ndarray, np.ndarray]:
        phi = np.asarray(phi, dtype=np.float32).ravel()
        a = int(stance) % self.n_stances
        descr = self.w_descr[a] @ phi
        delta = self.w_delta[a] @ phi
        return descr.astype(np.float32), delta.astype(np.float32)

    def learn(self, phi, stance: int, s0_norm, true_delta) -> None:
        """One SGD step toward the measured targets (descr→s0_norm, ΔŜ→true_delta).
        Linear least-squares gradient: dW = lr · err ⊗ φ / ‖φ‖²."""
        phi = np.asarray(phi, dtype=np.float32).ravel()
        a = int(stance) % self.n_stances
        denom = float(phi @ phi) + 1e-6
        descr_pred = self.w_descr[a] @ phi
        delta_pred = self.w_delta[a] @ phi
        e_descr = np.asarray(s0_norm, dtype=np.float32).ravel() - descr_pred
        e_delta = np.asarray(true_delta, dtype=np.float32).ravel() - delta_pred
        self.w_descr[a] += self.lr * np.outer(e_descr, phi) / denom
        self.w_delta[a] += self.lr * np.outer(e_delta, phi) / denom
        self.updates += 1

    def to_dict(self) -> dict:
        return {"schema": 1, "phi_dim": self.phi_dim, "state_dim": self.state_dim,
                "n_stances": self.n_stances, "lr": self.lr, "updates": self.updates,
                "w_descr": self.w_descr.ravel().tolist(),
                "w_delta": self.w_delta.ravel().tolist()}

    def load_dict(self, d: dict) -> bool:
        try:
            if not isinstance(d, dict) or int(d.get("schema", 0)) != 1:
                return False
            if (int(d.get("phi_dim", 0)) != self.phi_dim
                    or int(d.get("state_dim", 0)) != self.state_dim
                    or int(d.get("n_stances", 0)) != self.n_stances):
                return False
            shape = (self.n_stances, self.state_dim, self.phi_dim)
            self.w_descr = np.asarray(d["w_descr"], dtype=np.float32).reshape(shape)
            self.w_delta = np.asarray(d["w_delta"], dtype=np.float32).reshape(shape)
            self.updates = int(d.get("updates", 0) or 0)
            return True
        except Exception:
            return False


# ── The reward kernel — TELEMETRY ERROR ONLY (INV-IT-1) ─────────────────────
def inner_reward_kernel(descr_pred, delta_pred, s0_norm, s1_norm,
                        *, w_d: float = 0.5, w_delta: float = 0.5) -> dict:
    """r_inner = clip(1 − w_d·e_descr − w_Δ·e_delta, −1, 1).

    e_descr  = ‖descr_pred − znorm(S0)‖ / √D   (interoceptive: did he know himself NOW)
    e_delta  = ‖delta_pred − (znorm(S1) − znorm(S0))‖ / √D  (predictive: did he anticipate)

    PURE measured-telemetry error — NO LLM, NO judge, NO quality opinion enters
    here (INV-IT-1; G3 greps this kernel for any score/judge/LLM call → none).
    Normalized by √D so each error term is O(1). Returns the reward + its parts."""
    s0 = np.asarray(s0_norm, dtype=np.float32).ravel()
    s1 = np.asarray(s1_norm, dtype=np.float32).ravel()
    descr = np.asarray(descr_pred, dtype=np.float32).ravel()
    dpred = np.asarray(delta_pred, dtype=np.float32).ravel()
    d = float(np.sqrt(max(1, s0.shape[0])))
    e_descr = float(np.linalg.norm(descr - s0) / d)
    e_delta = float(np.linalg.norm(dpred - (s1 - s0)) / d)
    r = float(np.clip(1.0 - w_d * e_descr - w_delta * e_delta, -1.0, 1.0))
    return {"reward": r, "e_descr": e_descr, "e_delta": e_delta}


# ── The inner IQL net (own net; mirrors OuterMetaPolicy canonical IQL) ───────
_INNER_H1 = 32
_INNER_H2 = 16
_INNER_VH = 24
_INNER_QH = 24


def _mlp2_forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    h1 = np.maximum(0.0, z1)
    out = h1 @ w2 + b2
    return out, (X, z1, h1)


def _mlp2_backward(d_out, w2, cache):
    X, z1, h1 = cache
    dw2 = h1.T @ d_out
    db2 = d_out.sum(axis=0)
    dh1 = d_out @ w2.T
    dz1 = dh1 * (z1 > 0)
    dw1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dw1, db1, dw2, db2


def _clip_grad_norm(grads, max_norm: float) -> None:
    if max_norm <= 0:
        return
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if total > max_norm and total > 1e-12:
        scale = max_norm / total
        for g in grads:
            g *= scale


class InnerIQL:
    """Canonical IQL (numpy-by-hand) for the inner domain — expectile-V + Bellman-Q
    with V(s') (never max_a Q → no OOD) + AWR policy extraction + Polyak target,
    symlog-space value heads (INV-ML-1). A π net over the 5 stances + V/Q/Q-target.
    A NET-NEW net (INV-IT-4 / Q2 cold-review): the outer π net is fixed to 5
    routing modes / 30-D and cannot be reused. Mirrors OuterMetaPolicy.train_iql
    exactly so the inner domain learns identically to the outer one."""

    def __init__(self, input_dim: int = INNER_PHI_DIM,
                 n_actions: int = INNER_NUM_STANCES,
                 lr: float = 0.01, weight_decay: float = 0.001,
                 max_weight_norm: float = 6.0):
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_weight_norm = float(max_weight_norm)
        s1 = math.sqrt(2.0 / self.input_dim)
        s2 = math.sqrt(2.0 / _INNER_H1)
        s3 = math.sqrt(2.0 / _INNER_H2)
        self.w1 = np.random.randn(self.input_dim, _INNER_H1).astype(np.float32) * s1
        self.b1 = np.zeros(_INNER_H1, dtype=np.float32)
        self.w2 = np.random.randn(_INNER_H1, _INNER_H2).astype(np.float32) * s2
        self.b2 = np.zeros(_INNER_H2, dtype=np.float32)
        self.w3 = np.random.randn(_INNER_H2, self.n_actions).astype(np.float32) * s3
        self.b3 = np.zeros(self.n_actions, dtype=np.float32)
        self.total_updates = 0
        self.total_iql_updates = 0
        self._last_adv_pos_rate = 0.0
        self._iql_inited = False
        self._vw1 = self._vb1 = self._vw2 = self._vb2 = None
        self._qw1 = self._qb1 = self._qw2 = self._qb2 = None
        self._qtw1 = self._qtb1 = self._qtw2 = self._qtb2 = None

    # -- π inference -------------------------------------------------------
    def _policy_forward(self, x):
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)
        return h2 @ self.w3 + self.b3

    def action_probs(self, x, temperature: float = 1.0) -> np.ndarray:
        scores = self._policy_forward(np.asarray(x, dtype=np.float32).ravel())
        t = max(0.1, temperature)
        exp_s = np.exp((scores - scores.max()) / t)
        return exp_s / (exp_s.sum() + 1e-8)

    def select_stance(self, x, temperature: float = 1.0) -> int:
        """Boltzmann-sampled stance (idle EXPLORE — introspection is never on a
        live user turn, so exploration is always safe here)."""
        probs = self.action_probs(x, temperature)
        return int(np.random.choice(self.n_actions, p=probs))

    def exploit_stance(self, x) -> int:
        return int(np.argmax(self._policy_forward(np.asarray(x, dtype=np.float32).ravel())))

    def _clip_weight_norms(self) -> None:
        cap = self.max_weight_norm
        if cap <= 0:
            return
        for attr in ("w1", "w2", "w3"):
            w = getattr(self, attr)
            n = float(np.linalg.norm(w))
            if n > cap:
                w *= (cap / n)

    # -- IQL learner -------------------------------------------------------
    def init_iql(self) -> None:
        if self._iql_inited:
            return
        d = self.input_dim
        sv = math.sqrt(2.0 / d)
        svo = math.sqrt(2.0 / _INNER_VH)
        sqo = math.sqrt(2.0 / _INNER_QH)
        self._vw1 = np.random.randn(d, _INNER_VH).astype(np.float32) * sv
        self._vb1 = np.zeros(_INNER_VH, dtype=np.float32)
        self._vw2 = np.random.randn(_INNER_VH, 1).astype(np.float32) * svo
        self._vb2 = np.zeros(1, dtype=np.float32)
        self._qw1 = np.random.randn(d, _INNER_QH).astype(np.float32) * sv
        self._qb1 = np.zeros(_INNER_QH, dtype=np.float32)
        self._qw2 = np.random.randn(_INNER_QH, self.n_actions).astype(np.float32) * sqo
        self._qb2 = np.zeros(self.n_actions, dtype=np.float32)
        self._qtw1 = self._qw1.copy(); self._qtb1 = self._qb1.copy()
        self._qtw2 = self._qw2.copy(); self._qtb2 = self._qb2.copy()
        self._iql_inited = True

    def train_iql(self, transitions, *, tau: float = 0.7, beta: float = 3.0,
                  gamma: float = 0.99, polyak: float = 0.005,
                  adv_clip: float = 100.0, lr: float = 0.003,
                  steps: int = 20, batch_size: int = 32,
                  max_grad_norm: float = 1.0) -> dict:
        """One canonical-IQL consolidation pass (offline, idle-tick only). Each
        transition = dict(state, action, reward, next_state, terminal). Mirrors
        OuterMetaPolicy.train_iql / cgn._iql_update_consumer exactly."""
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
        tot_v = tot_q = tot_p = tot_adv_pos = 0.0
        for _ in range(int(steps)):
            idx = np.random.choice(n, bs, replace=False)
            batch = [transitions[i] for i in idx]
            S = np.array([_state(t) for t in batch], dtype=np.float32)
            S2 = np.array([_next(t) for t in batch], dtype=np.float32)
            A = np.array([int(t["action"]) for t in batch], dtype=np.int64)
            R = np.array([float(t["reward"]) for t in batch], dtype=np.float32)
            NT = np.array([0.0 if t.get("terminal") else 1.0 for t in batch],
                          dtype=np.float32)
            B = S.shape[0]
            rows = np.arange(B)

            # 1. V ← expectile_τ regression toward Q_target(s, a_taken).
            q_tgt_all, _ = _mlp2_forward(S, self._qtw1, self._qtb1,
                                         self._qtw2, self._qtb2)
            q_sa_tgt = q_tgt_all[rows, A]
            v_pred, v_cache = _mlp2_forward(S, self._vw1, self._vb1,
                                            self._vw2, self._vb2)
            v_pred = v_pred.reshape(B)
            diff = q_sa_tgt - v_pred
            w = np.where(diff < 0.0, 1.0 - tau, tau).astype(np.float32)
            v_loss = float(np.mean(w * diff * diff))
            d_v = (-2.0 * w * diff / B).reshape(B, 1).astype(np.float32)
            dvw1, dvb1, dvw2, dvb2 = _mlp2_backward(d_v, self._vw2, v_cache)
            _clip_grad_norm([dvw1, dvb1, dvw2, dvb2], max_grad_norm)
            self._vw1 -= lr * dvw1; self._vb1 -= lr * dvb1
            self._vw2 -= lr * dvw2; self._vb2 -= lr * dvb2

            # 2. Q ← Bellman backup with V(s') in REAL space (symlog heads).
            v_next_sym, _ = _mlp2_forward(S2, self._vw1, self._vb1,
                                          self._vw2, self._vb2)
            v_next_real = _symexp(v_next_sym.reshape(B)) * NT
            q_backup = _symlog(R + gamma * v_next_real)
            q_all, q_cache = _mlp2_forward(S, self._qw1, self._qb1,
                                           self._qw2, self._qb2)
            q_sa = q_all[rows, A]
            q_err = q_sa - q_backup
            q_loss = float(np.mean(q_err * q_err))
            d_q = np.zeros((B, self.n_actions), dtype=np.float32)
            d_q[rows, A] = (2.0 * q_err / B).astype(np.float32)
            dqw1, dqb1, dqw2, dqb2 = _mlp2_backward(d_q, self._qw2, q_cache)
            _clip_grad_norm([dqw1, dqb1, dqw2, dqb2], max_grad_norm)
            self._qw1 -= lr * dqw1; self._qb1 -= lr * dqb1
            self._qw2 -= lr * dqw2; self._qb2 -= lr * dqb2

            # 3. π ← AWR weight=clamp(exp(β·(Q(s,a)−V(s))), max=adv_clip).
            q_all2, _ = _mlp2_forward(S, self._qw1, self._qb1, self._qw2, self._qb2)
            v_pred2, _ = _mlp2_forward(S, self._vw1, self._vb1, self._vw2, self._vb2)
            adv = q_all2[rows, A] - v_pred2.reshape(B)
            tot_adv_pos += float(np.mean(adv > 0.0))
            weight = np.minimum(np.exp(beta * adv), adv_clip).astype(np.float32)
            z1 = S @ self.w1 + self.b1; h1 = np.maximum(0.0, z1)
            z2 = h1 @ self.w2 + self.b2; h2 = np.maximum(0.0, z2)
            z3 = h2 @ self.w3 + self.b3
            z3 = z3 - z3.max(axis=1, keepdims=True)
            exp_s = np.exp(z3)
            probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-8)
            target = np.zeros((B, self.n_actions), dtype=np.float32)
            target[rows, A] = 1.0
            d_z3 = (probs - target) * (weight.reshape(B, 1) / B)
            d_w3 = h2.T @ d_z3; d_b3 = d_z3.sum(axis=0)
            d_h2 = d_z3 @ self.w3.T; d_z2 = d_h2 * (z2 > 0)
            d_w2 = h1.T @ d_z2; d_b2 = d_z2.sum(axis=0)
            d_h1 = d_z2 @ self.w2.T; d_z1 = d_h1 * (z1 > 0)
            d_w1 = S.T @ d_z1; d_b1 = d_z1.sum(axis=0)
            _clip_grad_norm([d_w1, d_b1, d_w2, d_b2, d_w3, d_b3], max_grad_norm)
            policy_loss = float(np.mean(-weight * np.log(probs[rows, A] + 1e-8)))
            self.w1 -= lr * d_w1; self.b1 -= lr * d_b1
            self.w2 -= lr * d_w2; self.b2 -= lr * d_b2
            self.w3 -= lr * d_w3; self.b3 -= lr * d_b3
            if self.weight_decay:
                self.w1 *= (1.0 - self.weight_decay)
                self.w2 *= (1.0 - self.weight_decay)
                self.w3 *= (1.0 - self.weight_decay)
            self._clip_weight_norms()

            # 4. Polyak target update.
            self._qtw1 = (1.0 - polyak) * self._qtw1 + polyak * self._qw1
            self._qtb1 = (1.0 - polyak) * self._qtb1 + polyak * self._qb1
            self._qtw2 = (1.0 - polyak) * self._qtw2 + polyak * self._qw2
            self._qtb2 = (1.0 - polyak) * self._qtb2 + polyak * self._qb2

            self.total_updates += 1
            self.total_iql_updates += 1
            tot_v += v_loss; tot_q += q_loss; tot_p += policy_loss

        s = max(1, int(steps))
        self._last_adv_pos_rate = float(tot_adv_pos / s)
        return {"v_loss": tot_v / s, "q_loss": tot_q / s, "policy_loss": tot_p / s,
                "transitions": n, "iql_updates": self.total_iql_updates,
                "adv_pos_rate": self._last_adv_pos_rate}

    def advantage_positive_rate(self) -> float:
        return float(self._last_adv_pos_rate)

    def value_symlog(self, x) -> float:
        """V(s) in symlog space — the canonical value InnerMasteryLevel bins."""
        if not self._iql_inited:
            return 0.0
        out, _ = _mlp2_forward(np.asarray(x, dtype=np.float32).reshape(1, -1),
                               self._vw1, self._vb1, self._vw2, self._vb2)
        return float(out.reshape(-1)[0])

    # -- persistence (worker-owned JSON blob) ------------------------------
    def to_dict(self) -> dict:
        self.init_iql()
        return {
            "schema": 1, "input_dim": self.input_dim, "n_actions": self.n_actions,
            "total_updates": self.total_updates,
            "total_iql_updates": self.total_iql_updates,
            "last_adv_pos_rate": self._last_adv_pos_rate,
            "w1": self.w1.ravel().tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.ravel().tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.ravel().tolist(), "b3": self.b3.tolist(),
            "vw1": self._vw1.ravel().tolist(), "vb1": self._vb1.tolist(),
            "vw2": self._vw2.ravel().tolist(), "vb2": self._vb2.tolist(),
            "qw1": self._qw1.ravel().tolist(), "qb1": self._qb1.tolist(),
            "qw2": self._qw2.ravel().tolist(), "qb2": self._qb2.tolist(),
            "qtw1": self._qtw1.ravel().tolist(), "qtb1": self._qtb1.tolist(),
            "qtw2": self._qtw2.ravel().tolist(), "qtb2": self._qtb2.tolist(),
        }

    def load_dict(self, d: dict) -> bool:
        try:
            if not isinstance(d, dict) or int(d.get("schema", 0)) != 1:
                return False
            if (int(d.get("input_dim", 0)) != self.input_dim
                    or int(d.get("n_actions", 0)) != self.n_actions):
                return False
            self.init_iql()
            self.w1 = np.asarray(d["w1"], np.float32).reshape(self.input_dim, _INNER_H1)
            self.b1 = np.asarray(d["b1"], np.float32)
            self.w2 = np.asarray(d["w2"], np.float32).reshape(_INNER_H1, _INNER_H2)
            self.b2 = np.asarray(d["b2"], np.float32)
            self.w3 = np.asarray(d["w3"], np.float32).reshape(_INNER_H2, self.n_actions)
            self.b3 = np.asarray(d["b3"], np.float32)
            self._vw1 = np.asarray(d["vw1"], np.float32).reshape(self.input_dim, _INNER_VH)
            self._vb1 = np.asarray(d["vb1"], np.float32)
            self._vw2 = np.asarray(d["vw2"], np.float32).reshape(_INNER_VH, 1)
            self._vb2 = np.asarray(d["vb2"], np.float32)
            self._qw1 = np.asarray(d["qw1"], np.float32).reshape(self.input_dim, _INNER_QH)
            self._qb1 = np.asarray(d["qb1"], np.float32)
            self._qw2 = np.asarray(d["qw2"], np.float32).reshape(_INNER_QH, self.n_actions)
            self._qb2 = np.asarray(d["qb2"], np.float32)
            self._qtw1 = np.asarray(d["qtw1"], np.float32).reshape(self.input_dim, _INNER_QH)
            self._qtb1 = np.asarray(d["qtb1"], np.float32)
            self._qtw2 = np.asarray(d["qtw2"], np.float32).reshape(_INNER_QH, self.n_actions)
            self._qtb2 = np.asarray(d["qtb2"], np.float32)
            self.total_updates = int(d.get("total_updates", 0) or 0)
            self.total_iql_updates = int(d.get("total_iql_updates", 0) or 0)
            self._last_adv_pos_rate = float(d.get("last_adv_pos_rate", 0.0) or 0.0)
            return True
        except Exception:
            return False


# ── The embodied introspective drive + local refractory (INV-IT-9) ──────────
class IntrospectiveDrive:
    """The bottom-up urge to introspect, with a LOCAL adaptive threshold
    (ImpulseEngine asymmetric-EMA pattern). The drive blends embodied seeking
    (curiosity), inner-state volatility, and the standing dissonance from a prior
    misprediction. The threshold θ is the refractory: a WIN (accurate prediction)
    LOWERS θ slightly (he is satisfied → introspects a touch less eagerly next),
    a LOSE RAISES θ (dissonance persists → he keeps trying). Bounded [floor,ceil].

    v1 keeps this LOCAL — no body write (no NEUROMOD_EXTERNAL_NUDGE). The full
    §5.2 felt-coherence-down-to-body coupling is Q6. (Note the WIN→θ−α direction
    means an accurate self-model relaxes the urge; the loop never starves because
    new volatility/curiosity re-raise the drive.)"""

    def __init__(self, *, theta0: float = 0.35, alpha: float = 0.01,
                 beta: float = 0.02, floor: float = 0.10, ceil: float = 0.80,
                 vol_weight: float = 1.0, dissonance_weight: float = 1.0,
                 dna_bias: float = 0.0):
        self.theta = float(np.clip(theta0 - dna_bias, floor, ceil))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.vol_weight = float(vol_weight)
        self.dissonance_weight = float(dissonance_weight)
        self._standing_dissonance = 0.0     # rises on a LOSE, decays on a WIN
        self._wins = 0
        self._loses = 0

    def compute_drive(self, curiosity: float, volatility: float) -> float:
        """D = curiosity + vol_weight·volatility + dissonance_weight·standing_dissonance.
        All terms are real signals; clipped to [0, ~3]."""
        d = (float(curiosity)
             + self.vol_weight * float(volatility)
             + self.dissonance_weight * self._standing_dissonance)
        return float(np.clip(d, 0.0, 3.0))

    def should_fire(self, *, drive: float, great_pulse_fired: bool,
                    metabolic_ok: bool) -> bool:
        """Fire iff a great pulse fired AND metabolic headroom AND drive ≥ θ.
        Emergent — no timer (INV-IT-3/7)."""
        return bool(great_pulse_fired and metabolic_ok and (float(drive) >= self.theta))

    def record_outcome(self, *, win: bool, reward: float) -> dict:
        """Local refractory (INV-IT-9). WIN → θ−α + discharge standing dissonance;
        LOSE → θ+β + accumulate dissonance (proportional to how wrong he was)."""
        before = self.theta
        if win:
            self.theta = max(self.floor, self.theta - self.alpha)
            self._standing_dissonance = max(0.0, self._standing_dissonance * 0.5)
            self._wins += 1
        else:
            self.theta = min(self.ceil, self.theta + self.beta)
            self._standing_dissonance = min(2.0,
                self._standing_dissonance + abs(min(0.0, float(reward))))
            self._loses += 1
        return {"theta_before": before, "theta_after": self.theta,
                "win": bool(win), "standing_dissonance": self._standing_dissonance}

    def to_dict(self) -> dict:
        return {"schema": 1, "theta": self.theta,
                "standing_dissonance": self._standing_dissonance,
                "wins": self._wins, "loses": self._loses}

    def load_dict(self, d: dict) -> bool:
        try:
            if not isinstance(d, dict) or int(d.get("schema", 0)) != 1:
                return False
            self.theta = float(np.clip(float(d.get("theta", self.theta)),
                                       self.floor, self.ceil))
            self._standing_dissonance = float(d.get("standing_dissonance", 0.0) or 0.0)
            self._wins = int(d.get("wins", 0) or 0)
            self._loses = int(d.get("loses", 0) or 0)
            return True
        except Exception:
            return False


__all__ = (
    "INNER_STANCES", "INNER_NUM_STANCES", "INNER_STATE_DIM", "INNER_PHI_DIM",
    "INNER_GOAL_CLASS", "NEUROMOD_ORDER",
    "MASTERY_LEVEL_INNER_STATE_SLOT", "MASTERY_LEVEL_INNER_STATE_SPEC",
    "assemble_inner_state", "znorm_channels", "build_inner_phi",
    "curiosity_from_neuromod", "InnerSelfPredictor", "inner_reward_kernel",
    "InnerIQL", "IntrospectiveDrive", "build_inner_voice_prompts",
)
