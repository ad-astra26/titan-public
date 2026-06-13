"""Affective Grounding Loop — substrate-injection + emergent scoring.

RFP_affective_grounding_loop.md §7.A (Phase A) + §7.B (Phase B). Builds the
*missing channel* from a real event (here: synthesis skill-score outcomes) to a
gentle, decaying nudge on the neuromod substrate, so good/bad things that
actually happen to Titan move its inner felt-state. The 2026-06-10 experiment
proved this channel was ABSENT — 5 warm chat turns moved T1's emotion by exactly
zero.

PHASE A (the fixed-formula spine): ONE source (skill-score), ONE modulator
(`DA`, achievement/competence per RFP D4), a FIXED surprise formula
(`compute_skill_score_nudge`).

PHASE B (this file's emergent layer — `AffectiveNudgeNet` + `AffectiveNudgeRuntime`):
replaces the fixed magnitude formula with a small per-Titan LEARNED net (D2). It
learns each signal's MARGINAL affective value — how much REAL emot movement the
signal still produces — by training on the observed emot-delta that follows each
nudge. Habituation emerges by construction: a frequent signal whose felt-response
flattened → its training target → 0 → the net scores it down; a rare signal that
still moves the felt-state stays strong. Each Titan trains only on its own emot
trajectory ⇒ divergent emotional personalities (INV-AFF-SELF-SOVEREIGN).

  NUMPY-ONLY (BRAIN-INV-3): the net is pure-numpy forward AND pure-numpy SGD
  backprop — NEVER torch. This mirrors the OuterMetaPolicy learning operator
  (main SPEC §13 / BRAIN-INV-3 "RL scaffolding only … never torch"; the IQL/torch
  gatekeeper stays retired), NOT the heavy cross-process CGN value-net pattern.
  A ~300-param net needs no torch, and importing it at the dream boundary would
  risk the synthesis worker's RSS / the 40s-GIL dream window. State persists as a
  numpy `.npz` state_dict (atomic tmp+replace, like the EMA baseline json).

HONESTY / EMERGENCE (the RFP invariants this file must satisfy):
  - INV-AFF-EMERGENT  : magnitude = the net's learned output (Phase B) or, until
                        the net has trained, surprise × a config gain (Phase A
                        cold-start fallback). NEVER a hardcoded per-event magnitude.
                        Surprise = |rate − μ| / (σ + ε) against the signal's OWN
                        running baseline.
  - INV-AFF-HONEST    : valence is the SIGN of the real competence delta
                        (rate above baseline → +, below → −). Symmetric. No
                        forced positivity, no floor. (The net scores MAGNITUDE;
                        the valence sign is always the honest competence delta.)
  - habituation (G3)  : Phase A — a stable rate drives μ→rate→surprise→0. Phase B
                        — additionally, a flattened observed emot-delta trains the
                        net's predicted magnitude toward 0. Nothing hardcoded.

The OUTPUT is a *target-pull* instruction for `NeuromodulatorSystem
.apply_external_nudge` (logic/neuromodulator.py:372): a TARGET value in [0,1]
(NOT a delta — the agno_hooks.py:1239 landmine) plus a surprise-scaled
`max_delta` magnitude. apply_external_nudge composes it (GABA-excluded,
developmental-gated, clamped) — INV-AFF-COMPOSE is satisfied by that API, not
here.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("titan.affective_nudge")

# The modulator the skill-score (competence/achievement) signal maps to.
# RFP D4: achievement → DA ("sharp reward signals", neuromodulator.py:33).
SKILL_SCORE_MODULATOR = "DA"


# ── Config (titan_params.toml [affective]) ───────────────────────────────────
@dataclass(frozen=True)
class AffectiveConfig:
    """Defaults match titan_params.toml [affective]. Config GAINS are allowed
    (emergence-over-determinism: gains tune, they do not predetermine the felt
    outcome); there are NO per-tier/per-event magnitudes here."""

    enabled: bool = False          # master flag — default OFF until proven (RFP §5 rollback)
    k_surprise: float = 0.04       # gain: surprise → max_delta magnitude
    max_mag: float = 0.06          # ceiling on a single nudge's max_delta (gentle)
    ema_alpha: float = 0.15        # EMA learning rate for the per-signal baseline
    sigma_init: float = 0.25       # initial σ before enough samples (wide → low early surprise)
    eps: float = 1e-3              # σ floor in the surprise denominator
    min_samples: int = 2           # no nudge until the baseline has seen this many ticks
    # ── Phase B (the emergent AffectiveNudgeNet) ──────────────────────────────
    net_enabled: bool = True       # use the learned net once trained; else Phase-A formula
    net_lr: float = 0.05           # numpy-SGD learning rate (the net's only learning gain)
    net_l2: float = 1e-4           # weight decay (gentle regularization)
    net_hidden: int = 16           # hidden width (12→16→1, ~300 params)


def load_affective_config() -> AffectiveConfig:
    """Best-effort load of `[affective]` from titan_params.toml. Returns the
    in-code defaults on any error (file/section missing, parse error) so unit
    tests and minimal installs keep working. The synthesis_worker calls this
    once at boot."""
    try:
        try:
            import tomllib  # py3.11+
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → titan_hcl/
        path = os.path.join(here, "titan_params.toml")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        sub = data.get("affective", {}) or {}
        d = AffectiveConfig()
        return AffectiveConfig(
            enabled=bool(sub.get("enabled", d.enabled)),
            k_surprise=float(sub.get("k_surprise", d.k_surprise)),
            max_mag=float(sub.get("max_mag", d.max_mag)),
            ema_alpha=float(sub.get("ema_alpha", d.ema_alpha)),
            sigma_init=float(sub.get("sigma_init", d.sigma_init)),
            eps=float(sub.get("eps", d.eps)),
            min_samples=int(sub.get("min_samples", d.min_samples)),
            net_enabled=bool(sub.get("net_enabled", d.net_enabled)),
            net_lr=float(sub.get("net_lr", d.net_lr)),
            net_l2=float(sub.get("net_l2", d.net_l2)),
            net_hidden=int(sub.get("net_hidden", d.net_hidden)),
        )
    except Exception:
        return AffectiveConfig()


# ── Per-signal running baseline (the surprise + habituation substrate) ────────
@dataclass
class _SignalBaseline:
    mu: float = 0.0        # EMA mean of the signal's success_rate
    sigma: float = 0.0     # EMA std (sqrt of EMA variance) of success_rate
    n: int = 0             # observations folded in

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "_SignalBaseline":
        return cls(
            mu=float(d.get("mu", 0.0)),
            sigma=float(d.get("sigma", 0.0)),
            n=int(d.get("n", 0)),
        )


def _load_state(state_path: str) -> dict:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_state(state_path: str, state: dict) -> None:
    """Atomic best-effort persist (tmp + os.replace). A failed save loses only
    the baseline update for one tick — never raises into the drain loop."""
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(state_path), prefix=".affnudge_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp, state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as e:
        logger.debug("[affective_nudge] state save failed: %s", e)


# ── Result ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Nudge:
    """A target-pull instruction for NeuromodulatorSystem.apply_external_nudge.
    `target` ∈ {0.0, 1.0} = pull DA down / up; `magnitude` = the surprise-scaled
    max_delta. `magnitude == 0.0` means "no nudge this tick" (caller skips)."""

    magnitude: float
    target: float
    surprise: float
    valence: int       # +1 / -1 / 0
    rate: float
    mu_before: float
    n: int


def compute_skill_score_nudge(
    successes: int,
    failures: int,
    state_path: str,
    *,
    cfg: Optional[AffectiveConfig] = None,
    signal_type: str = "skill_score",
) -> Optional[Nudge]:
    """Fold this tick's skill-score outcomes into the per-signal EMA baseline,
    derive a surprise-scaled, honest-valence DA nudge, persist the updated
    baseline, and return the Nudge (or None when there is nothing to emit).

    Returns None when: no outcomes this tick · baseline still warming up
    (n < min_samples) · the outcome exactly matches baseline (surprise → 0,
    valence 0). All three are honest "no real movement" cases — Phase A never
    emits a nudge that isn't grounded in a real deviation.
    """
    cfg = cfg or AffectiveConfig()
    total = int(successes) + int(failures)
    if total <= 0:
        return None
    rate = float(successes) / float(total)

    state = _load_state(state_path)
    base = _SignalBaseline.from_dict(state.get(signal_type, {}))
    mu_before, n_before = base.mu, base.n

    # Fold the observation into the EMA baseline BEFORE deciding the nudge, so
    # the surprise is measured against the PRIOR baseline (mu_before) — the
    # honest "how unexpected was this, given what I knew" reading.
    if base.n == 0:
        # First-ever observation: seed the baseline, emit nothing (no prior →
        # no surprise is definable; claiming one would be a hardcoded magnitude).
        base.mu = rate
        base.sigma = cfg.sigma_init
        base.n = 1
        state[signal_type] = base.to_dict()
        _save_state(state_path, state)
        return None

    deviation = rate - mu_before
    sigma_for_surprise = max(base.sigma, cfg.eps) if base.n >= cfg.min_samples \
        else max(cfg.sigma_init, cfg.eps)
    surprise = abs(deviation) / (sigma_for_surprise + cfg.eps)

    # EMA update (mean + variance-EMA → σ). Welford-flavoured EMA: variance
    # tracks the squared deviation from the OLD mean, the standard EMA-var form.
    a = cfg.ema_alpha
    new_mu = (1.0 - a) * base.mu + a * rate
    new_var = (1.0 - a) * (base.sigma ** 2) + a * (deviation ** 2)
    base.mu = new_mu
    base.sigma = math.sqrt(max(0.0, new_var))
    base.n = base.n + 1
    state[signal_type] = base.to_dict()
    _save_state(state_path, state)

    # Still warming up → baseline not yet trustworthy → no nudge (honest).
    if n_before < cfg.min_samples:
        return None

    # Valence = sign of the real competence delta (INV-AFF-HONEST, symmetric).
    if deviation > 0:
        valence, target = 1, 1.0
    elif deviation < 0:
        valence, target = -1, 0.0
    else:
        return None  # exactly on baseline → no real movement

    magnitude = min(cfg.max_mag, cfg.k_surprise * surprise)
    if magnitude <= 0.0:
        return None

    return Nudge(
        magnitude=float(magnitude),
        target=float(target),
        surprise=float(surprise),
        valence=int(valence),
        rate=float(rate),
        mu_before=float(mu_before),
        n=int(n_before),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — the emergent AffectiveNudgeNet (D2)
# ─────────────────────────────────────────────────────────────────────────────

# Feature vector fed to the net (RFP §7.B):
#   [0] surprise          — |rate−μ|/(σ+ε), scaled (the deviation's unexpectedness)
#   [1] recent_freq       — log1p(n) proxy for how often this signal has fired
#   [2] |deviation|       — |rate−μ|, the raw competence-delta magnitude
#   [3] emot_V_blended    — current dominant blended valence (felt context)
#   [4..11] dominant 1-hot — which of the 8 emot primitives is dominant now
FEATURE_DIM = 12


def build_features(nudge: "Nudge",
                   emot_state: Optional[dict],
                   *, n_freq_scale: float = 8.0) -> np.ndarray:
    """Build the net input vector from a fired Nudge + the emot SHM snapshot.

    Pure + deterministic (testable). Robust to a missing emot read (None →
    felt-context features default to 0 / no dominant). Features are kept in
    roughly [0,1]/[-1,1] ranges so the small net needs no LayerNorm."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat[0] = float(np.clip(nudge.surprise / 5.0, 0.0, 3.0))   # surprise (scaled)
    feat[1] = float(min(1.0, math.log1p(max(0, nudge.n)) / n_freq_scale))
    feat[2] = float(np.clip(abs(nudge.rate - nudge.mu_before), 0.0, 1.0))
    if emot_state:
        feat[3] = float(np.clip(emot_state.get("V_blended", 0.0), -1.0, 1.0))
        idx = int(emot_state.get("dominant_idx", -1))
        if 0 <= idx < 8:
            feat[4 + idx] = 1.0
    return feat


def signed_delta_target(pre_v8: Optional[np.ndarray],
                        post_v8: Optional[np.ndarray]) -> Optional[float]:
    """The Hybrid attribution target (RFP §7.B, Maker-approved): the SIGNED
    magnitude of the emot drift across the wake cycle, measured on the 8-D
    grounding valence vector (`ShmEmotReader.read_grounding` → per-primitive V).

      target = sign(mean(post) − mean(pre)) · ‖post − pre‖₂

    The L2 norm captures how much the whole felt-space moved; the sign records
    whether the dominant valence rose or fell. The net learns to PREDICT this
    magnitude per signal → habituation emerges as the observed drift flattens.
    Returns None when either snapshot is unavailable (skip — never train on a
    fabricated delta, INV-AFF-HONEST)."""
    if pre_v8 is None or post_v8 is None:
        return None
    if pre_v8.shape != post_v8.shape or pre_v8.size == 0:
        return None
    diff = post_v8 - pre_v8
    mag = float(np.linalg.norm(diff))
    if mag == 0.0:
        return 0.0
    sign = 1.0 if float(post_v8.mean() - pre_v8.mean()) >= 0.0 else -1.0
    return sign * mag


class AffectiveNudgeNet:
    """A small per-Titan numpy MLP (12 → hidden → 1) that learns each signal's
    marginal affective value. Pure-numpy forward AND pure-numpy SGD backprop —
    NEVER torch (BRAIN-INV-3; mirrors OuterMetaPolicy). ReLU hidden, linear
    signed output. ~300 params at hidden=16.

    The net predicts the (signed) MAGNITUDE of emot drift a signal produces; the
    runtime uses |output| (clamped to max_mag) as the gentle `max_delta`, while
    valence/target stay the honest competence-delta sign (INV-AFF-HONEST)."""

    def __init__(self, hidden: int = 16, *, in_dim: int = FEATURE_DIM,
                 seed: Optional[int] = None):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        rng = np.random.default_rng(seed)
        # Small init — an untrained net is never used live (runtime cold-start
        # falls back to the Phase-A formula until trained_steps ≥ 1).
        self.W1 = rng.standard_normal((self.hidden, self.in_dim)) * 0.1
        self.b1 = np.zeros(self.hidden, dtype=np.float64)
        self.W2 = rng.standard_normal((1, self.hidden)) * 0.1
        self.b2 = np.zeros(1, dtype=np.float64)
        self.trained_steps = 0

    @property
    def is_trained(self) -> bool:
        return self.trained_steps >= 1

    def forward(self, x: np.ndarray) -> float:
        """x: (in_dim,) → scalar signed prediction."""
        h_pre = self.W1 @ x + self.b1
        h = np.maximum(0.0, h_pre)
        out = self.W2 @ h + self.b2          # (1,)
        return float(out.reshape(-1)[0])     # numpy-2 safe scalar extraction

    def train_step(self, X: np.ndarray, y: np.ndarray,
                   lr: float = 0.05, l2: float = 1e-4) -> float:
        """One batched SGD step over (X:(N,in_dim), y:(N,)). MSE loss, mean grad,
        L2 decay. Returns the pre-step mean loss. Pure numpy (no torch)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return 0.0
        # Forward (batched).
        h_pre = X @ self.W1.T + self.b1            # (N, hidden)
        h = np.maximum(0.0, h_pre)                  # (N, hidden)
        out = h @ self.W2.T + self.b2              # (N, 1)
        out = out.reshape(-1)                       # (N,)
        err = out - y                              # (N,)
        loss = float(np.mean(err ** 2))
        # Backward.
        d_out = (2.0 / n) * err                     # (N,)
        dW2 = d_out @ h                            # (hidden,)
        db2 = float(np.sum(d_out))
        dh = np.outer(d_out, self.W2.reshape(-1))  # (N, hidden)
        dh_pre = dh * (h_pre > 0.0)                 # ReLU grad
        dW1 = dh_pre.T @ X                         # (hidden, in_dim)
        db1 = np.sum(dh_pre, axis=0)               # (hidden,)
        # SGD + L2.
        self.W2 -= lr * (dW2.reshape(1, -1) + l2 * self.W2)
        self.b2 -= lr * db2
        self.W1 -= lr * (dW1 + l2 * self.W1)
        self.b1 -= lr * db1
        self.trained_steps += 1
        return loss

    def state_dict(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "trained_steps": np.array([self.trained_steps], dtype=np.int64)}

    def load_state_dict(self, sd: dict) -> None:
        self.W1 = np.asarray(sd["W1"], dtype=np.float64)
        self.b1 = np.asarray(sd["b1"], dtype=np.float64)
        self.W2 = np.asarray(sd["W2"], dtype=np.float64)
        self.b2 = np.asarray(sd["b2"], dtype=np.float64)
        self.hidden = int(self.W1.shape[0])
        self.in_dim = int(self.W1.shape[1])
        self.trained_steps = int(np.asarray(sd["trained_steps"]).reshape(-1)[0])

    def save_npz(self, path: str) -> None:
        """Atomic .npz persist (tmp + os.replace) — same discipline as the EMA
        baseline json. A failed save loses only this batch's training update."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # mkstemp with a .npz suffix → np.savez writes EXACTLY this path (it
            # only auto-appends .npz when the name lacks the suffix). So the data
            # lands in `tmp`, and we atomically replace into place.
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(path), prefix=".affnet_", suffix=".npz")
            os.close(fd)
            try:
                np.savez(tmp, **self.state_dict())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as e:
            logger.debug("[affective_nudge] net save failed: %s", e)

    @classmethod
    def load_npz(cls, path: str, hidden: int = 16) -> "AffectiveNudgeNet":
        net = cls(hidden=hidden)
        try:
            if os.path.exists(path):
                with np.load(path) as data:
                    net.load_state_dict({k: data[k] for k in data.files})
        except Exception as e:
            logger.debug("[affective_nudge] net load failed (fresh net): %s", e)
        return net


@dataclass
class _PendingNudge:
    """A fired nudge awaiting its dream-boundary emot-delta attribution."""
    features: np.ndarray
    pre_v8: Optional[np.ndarray]
    ts: float


class AffectiveNudgeRuntime:
    """Owns the per-Titan net + the wake-cycle pending buffer. Shared between the
    synthesis drain thread (forward + record pre-snapshot) and the single
    `synthesis-dream` thread (train on post-snapshot) — both in synthesis_worker,
    so no SHM round-trip. Thread-safe via one lock around the pending buffer + the
    net weights (forward/train never run concurrently in practice — drain is light,
    dream is the ordered sequence — but the lock keeps it correct).

    `emot_state_reader` / `emot_grounding_reader` are zero-arg callables returning
    the emot SHM `read_state()` dict and `read_grounding()` list (injected so the
    runtime stays decoupled from emot_shm_protocol — and unit-testable)."""

    def __init__(self, cfg: AffectiveConfig, state_path: str, net_path: str,
                 *, emot_state_reader=None, emot_grounding_reader=None,
                 max_pending: int = 256):
        self.cfg = cfg
        self.state_path = state_path
        self.net_path = net_path
        self._emot_state_reader = emot_state_reader
        self._emot_grounding_reader = emot_grounding_reader
        self._max_pending = int(max_pending)
        self._lock = threading.Lock()
        self._pending: List[_PendingNudge] = []
        self.net = AffectiveNudgeNet.load_npz(net_path, hidden=cfg.net_hidden)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _grounding_to_v8(grounding) -> Optional[np.ndarray]:
        """ShmEmotReader.read_grounding() → 8-vector of per-primitive V, or None."""
        if not grounding:
            return None
        try:
            v = np.array([float(e.get("V", 0.0)) for e in grounding],
                         dtype=np.float64)
            return v if v.size == 8 else None
        except Exception:
            return None

    def _read_emot_state(self) -> Optional[dict]:
        if self._emot_state_reader is None:
            return None
        try:
            return self._emot_state_reader()
        except Exception:
            return None

    def _read_v8(self) -> Optional[np.ndarray]:
        if self._emot_grounding_reader is None:
            return None
        try:
            return self._grounding_to_v8(self._emot_grounding_reader())
        except Exception:
            return None

    # ── drain-tick path (forward) ──────────────────────────────────────────────
    def observe_drain(self, successes: int, failures: int,
                      ts: float) -> Optional[Nudge]:
        """Phase-B forward at a drain tick. Folds the EMA baseline + derives the
        honest valence/surprise via the Phase-A helper, then OVERRIDES the
        magnitude with the learned net (once trained; else keeps the Phase-A
        formula magnitude = cold-start fallback). Records the pre-nudge emot
        snapshot so the dream boundary can attribute the resulting drift.

        Returns the Nudge to emit, or None (warming up / on-baseline / no events
        — same honest no-movement cases as Phase A)."""
        base_nudge = compute_skill_score_nudge(
            successes, failures, self.state_path, cfg=self.cfg)
        if base_nudge is None:
            return None

        emot_state = self._read_emot_state()
        features = build_features(base_nudge, emot_state)

        nudge = base_nudge
        if self.cfg.net_enabled:
            with self._lock:
                trained = self.net.is_trained
                pred = self.net.forward(features) if trained else None
            if trained and pred is not None:
                magnitude = float(min(self.cfg.max_mag, abs(pred)))
                if magnitude <= 0.0:
                    # Net learned this signal no longer moves the felt-state
                    # (full habituation). Honest no-nudge — but still record the
                    # baseline EMA (already persisted by the helper above).
                    return None
                nudge = Nudge(
                    magnitude=magnitude, target=base_nudge.target,
                    surprise=base_nudge.surprise, valence=base_nudge.valence,
                    rate=base_nudge.rate, mu_before=base_nudge.mu_before,
                    n=base_nudge.n)

        pre_v8 = self._read_v8()
        with self._lock:
            self._pending.append(_PendingNudge(features=features, pre_v8=pre_v8,
                                               ts=float(ts)))
            if len(self._pending) > self._max_pending:
                self._pending = self._pending[-self._max_pending:]
        return nudge

    # ── dream-boundary path (train) ────────────────────────────────────────────
    def train_on_dream(self) -> dict:
        """Phase-B training, run INLINE in the ordered `synthesis-dream` thread
        (INV-Syn-28) after `flush_companion_batches`. Reads the post-cycle emot
        snapshot, attributes each pending nudge's drift, runs ONE batched numpy
        SGD step, persists the net, clears the buffer. Returns a summary dict."""
        with self._lock:
            pending = self._pending
            self._pending = []
        if not pending:
            return {"trained": 0, "skipped": 0}

        post_v8 = self._read_v8()
        X_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        skipped = 0
        for p in pending:
            target = signed_delta_target(p.pre_v8, post_v8)
            if target is None:
                skipped += 1
                continue
            X_rows.append(p.features)
            y_rows.append(target)
        if not X_rows:
            return {"trained": 0, "skipped": skipped}

        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=np.float64)
        with self._lock:
            loss = self.net.train_step(X, y, lr=self.cfg.net_lr, l2=self.cfg.net_l2)
            self.net.save_npz(self.net_path)
            steps = self.net.trained_steps
        return {"trained": int(X.shape[0]), "skipped": int(skipped),
                "loss": float(loss), "trained_steps": int(steps),
                "target_abs_mean": float(np.mean(np.abs(y)))}
