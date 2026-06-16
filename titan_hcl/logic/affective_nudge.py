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

PHASE C (broaden the taps — §7.C): the SAME one net now learns from MANY signals,
not just skill_score. A `signal_type` one-hot is appended to the feature vector
(SIGNAL_TYPES / build_features) so one shared net learns each signal's marginal
value (D2). Phase C adds `sol_receipt` (balance-SHM delta, receipt + / spend −),
`maker_bond` (a receipt whose funding feePayer == the Maker), `x_engagement`
(ENGAGEMENT_SNAPSHOT_TAKEN delta) and `chain_reuse` (times_reused delta) — each a
real event with a real state-delta, all → DA in v1 (multi-modulator spread = C.2).
Event signals use `compute_event_nudge` (magnitude = log1p(|state-delta|) into the
per-signal EMA; valence = the intrinsic state-delta direction).

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

# ── Signal registry (RFP §7.C — broaden from one signal to many) ──────────────
# The ONE shared net (Phase B) learns each signal's marginal affective value via a
# `signal_type` one-hot appended to the feature vector. ORDER IS THE ONE-HOT INDEX:
# APPEND new signals, NEVER reorder/remove — a saved net's one-hot columns are
# positional, and `_SignalBaseline`s in affective_nudge_state.json are keyed by the
# name (so the EMA/surprise/habituation stay per-signal). Phase A/B = "skill_score";
# Phase C adds the next four (sol_receipt + maker_bond clean-in-process this build;
# x_engagement + chain_reuse = Tier 2 cross-process).
SIGNAL_TYPES: tuple = (
    "skill_score",     # Phase A/B — synthesis competence delta
    "sol_receipt",     # Phase C — balance-SHM delta (receipt + / spend −)
    "maker_bond",      # Phase C — a sol_receipt whose funding feePayer == the Maker
    "x_engagement",    # Phase C (Tier 2) — ENGAGEMENT_SNAPSHOT_TAKEN delta
    "chain_reuse",     # Phase C (Tier 2) — meta_wisdom times_reused delta
)
SIGNAL_INDEX = {name: i for i, name in enumerate(SIGNAL_TYPES)}
N_SIGNAL_TYPES = len(SIGNAL_TYPES)

# Modulator mapping (Maker-decided 2026-06-13): ALL signals → DA in Phase C v1
# (prove the multi-signal net on one channel first). Multi-modulator spread
# (sol/maker_bond→5HT/Endorphin, x→NE) = C.2 follow-up — kept as a map so C.2 can
# diverge per signal without touching the taps. RFP D4: achievement/reward → DA
# ("sharp reward signals", neuromodulator.py:33).
SIGNAL_MODULATOR = {name: "DA" for name in SIGNAL_TYPES}
SKILL_SCORE_MODULATOR = "DA"   # back-compat alias (existing synthesis_worker import)


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


def _fold_metric_nudge(
    metric: float,
    state_path: str,
    *,
    cfg: AffectiveConfig,
    signal_type: str,
    valence_override: Optional[int] = None,
) -> Optional[Nudge]:
    """The signal-agnostic EMA-fold + surprise + honest-valence core (RFP §7.A
    machinery, generalized for §7.C). `metric` = the signal's natural-scale scalar
    folded into its OWN per-signal EMA baseline (skill_score: success_rate; event
    signals: log1p(|state-delta|) — the event's MAGNITUDE).

    Valence:
      • `valence_override is None` (skill_score) → the honest above/below-baseline
        competence sign = sign(metric − μ). A LEVEL signal: doing better/worse
        than my own running rate.
      • `valence_override ∈ {+1,−1}` (event signals) → the intrinsic state-delta
        direction (a SOL spend is −, a receipt/engagement/reuse is +); the EMA
        gives the MAGNITUDE's surprise, not the direction. INV-AFF-HONEST holds:
        valence still derives from a real state-delta, never a forced positivity.

    Returns None on the honest "no real movement" cases: first-ever observation
    (seed only — no prior → no definable surprise), still warming up
    (n < min_samples), or surprise → 0 (magnitude collapses to 0 = habituated /
    on-baseline). Persists the updated per-signal baseline before returning."""
    state = _load_state(state_path)
    base = _SignalBaseline.from_dict(state.get(signal_type, {}))
    mu_before, n_before = base.mu, base.n

    # Fold the observation into the EMA baseline BEFORE deciding the nudge, so
    # the surprise is measured against the PRIOR baseline (mu_before) — the
    # honest "how unexpected was this, given what I knew" reading.
    if base.n == 0:
        # First-ever observation: seed the baseline, emit nothing (no prior →
        # no surprise is definable; claiming one would be a hardcoded magnitude).
        base.mu = metric
        base.sigma = cfg.sigma_init
        base.n = 1
        state[signal_type] = base.to_dict()
        _save_state(state_path, state)
        return None

    deviation = metric - mu_before
    sigma_for_surprise = max(base.sigma, cfg.eps) if base.n >= cfg.min_samples \
        else max(cfg.sigma_init, cfg.eps)
    surprise = abs(deviation) / (sigma_for_surprise + cfg.eps)

    # EMA update (mean + variance-EMA → σ). Welford-flavoured EMA: variance
    # tracks the squared deviation from the OLD mean, the standard EMA-var form.
    a = cfg.ema_alpha
    new_mu = (1.0 - a) * base.mu + a * metric
    new_var = (1.0 - a) * (base.sigma ** 2) + a * (deviation ** 2)
    base.mu = new_mu
    base.sigma = math.sqrt(max(0.0, new_var))
    base.n = base.n + 1
    state[signal_type] = base.to_dict()
    _save_state(state_path, state)

    # Still warming up → baseline not yet trustworthy → no nudge (honest).
    if n_before < cfg.min_samples:
        return None

    # Valence: intrinsic event direction (override) or the competence-delta sign.
    if valence_override is not None:
        if valence_override > 0:
            valence, target = 1, 1.0
        elif valence_override < 0:
            valence, target = -1, 0.0
        else:
            return None
    elif deviation > 0:
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
        rate=float(metric),
        mu_before=float(mu_before),
        n=int(n_before),
    )


def compute_skill_score_nudge(
    successes: int,
    failures: int,
    state_path: str,
    *,
    cfg: Optional[AffectiveConfig] = None,
    signal_type: str = "skill_score",
) -> Optional[Nudge]:
    """Phase A/B skill-score signal: fold this tick's success_rate into the
    per-signal EMA baseline and derive a surprise-scaled, honest-valence DA nudge
    (or None when there is nothing to emit — no outcomes / warming up / on
    baseline). Valence = the above/below-baseline competence sign. Byte-identical
    to the pre-§7.C behaviour (delegates to the shared core with no override)."""
    cfg = cfg or AffectiveConfig()
    total = int(successes) + int(failures)
    if total <= 0:
        return None
    rate = float(successes) / float(total)
    return _fold_metric_nudge(rate, state_path, cfg=cfg, signal_type=signal_type)


def compute_event_nudge(
    signed_delta: float,
    state_path: str,
    *,
    signal_type: str,
    cfg: Optional[AffectiveConfig] = None,
    intrinsic_positive: bool = False,
) -> Optional[Nudge]:
    """Phase C event signal (sol_receipt / maker_bond / x_engagement / chain_reuse).

    `signed_delta` = the raw state-delta on its NATURAL scale (SOL for sol/maker,
    engagement count for x, reuse count for chain). The EVENT'S MAGNITUDE
    `log1p(|signed_delta|)` folds into the signal's OWN per-signal EMA → surprise
    measures how unexpectedly large this event is vs the signal's running typical
    size, so habituation emerges per-signal (a stream of same-size events → μ
    matches → surprise → 0). Valence is the intrinsic state-delta direction:
      • `intrinsic_positive=True` (engagement / reuse / maker_bond — can't be
        negative) → always +.
      • else → sign(signed_delta) (sol_receipt: a receipt is +, a spend is −,
        symmetric per INV-AFF-HONEST).
    Returns None for a zero delta (no real event) and the same warming-up /
    habituated cases as the core."""
    cfg = cfg or AffectiveConfig()
    d = float(signed_delta)
    if d == 0.0:
        return None
    magnitude_metric = math.log1p(abs(d))   # natural-scale event magnitude
    valence_override = 1 if (intrinsic_positive or d > 0) else -1
    return _fold_metric_nudge(
        magnitude_metric, state_path, cfg=cfg, signal_type=signal_type,
        valence_override=valence_override)


# ── §7.C Tier 2 cross-process source readers (pure, read-only sqlite) ──────────
def read_engagement_delta(db_path: str,
                          last_cursor: Optional[float]) -> tuple:
    """Aggregate engagement delta from `events_teacher.db::engagement_snapshots`
    (the x_engagement source). Returns `(total_delta:int, new_cursor:float)`.

    `last_cursor` is the highest `checked_at` already counted. When it is None the
    cursor is *established* at the current MAX(checked_at) with delta 0 (so a fresh
    boot does NOT replay all historical engagement as one giant event). Otherwise
    sums `delta_likes+delta_replies+delta_quotes` over snapshots newer than the
    cursor and advances it. Read-only, soft: returns `(0, last_cursor or 0.0)` on a
    missing db / locked read / schema miss (never raises into the drain loop)."""
    import sqlite3
    fallback = (0, float(last_cursor or 0.0))
    if not db_path or not os.path.exists(db_path):
        return fallback
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        try:
            if last_cursor is None:
                row = con.execute(
                    "SELECT COALESCE(MAX(checked_at), 0) "
                    "FROM engagement_snapshots").fetchone()
                return (0, float((row or [0])[0] or 0.0))
            rows = con.execute(
                "SELECT delta_likes, delta_replies, delta_quotes, checked_at "
                "FROM engagement_snapshots WHERE checked_at > ? "
                "ORDER BY checked_at ASC LIMIT 200", (float(last_cursor),)
            ).fetchall()
            total = 0
            cursor = float(last_cursor)
            for r in rows:
                total += int(r[0] or 0) + int(r[1] or 0) + int(r[2] or 0)
                cursor = max(cursor, float(r[3] or 0.0))
            return (int(total), cursor)
        finally:
            con.close()
    except Exception as e:
        logger.debug("[affective_nudge] read_engagement_delta failed: %s", e)
        return fallback


def read_reuse_total(db_path: str) -> Optional[int]:
    """Return `SUM(times_reused)` over `inner_memory.db::meta_wisdom` (the
    chain_reuse running total), or None on a missing db / error. The caller diffs
    successive totals into a per-tick reuse delta. Read-only, soft."""
    import sqlite3
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        try:
            row = con.execute(
                "SELECT COALESCE(SUM(times_reused), 0) FROM meta_wisdom").fetchone()
            return int((row or [0])[0] or 0)
        finally:
            con.close()
    except Exception as e:
        logger.debug("[affective_nudge] read_reuse_total failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — the emergent AffectiveNudgeNet (D2)
# ─────────────────────────────────────────────────────────────────────────────

# Feature vector fed to the net (RFP §7.B base + §7.C signal one-hot):
#   [0] surprise          — |metric−μ|/(σ+ε), scaled (the deviation's unexpectedness)
#   [1] recent_freq       — log1p(n) proxy for how often this signal has fired
#   [2] |deviation|       — |metric−μ|, the raw state-delta magnitude
#   [3] emot_V_blended    — current dominant blended valence (felt context)
#   [4..11] dominant 1-hot — which of the 8 emot primitives is dominant now
#   [12..12+N) signal_type one-hot (§7.C) — lets the ONE shared net learn each
#              signal's marginal value (D2). Positional → SIGNAL_INDEX order.
_BASE_FEATURE_DIM = 12
FEATURE_DIM = _BASE_FEATURE_DIM + N_SIGNAL_TYPES   # 12 + 5 = 17


def build_features(nudge: "Nudge",
                   emot_state: Optional[dict],
                   *, signal_type: str = "skill_score",
                   n_freq_scale: float = 8.0) -> np.ndarray:
    """Build the net input vector from a fired Nudge + the emot SHM snapshot.

    Pure + deterministic (testable). Robust to a missing emot read (None →
    felt-context features default to 0 / no dominant). Features are kept in
    roughly [0,1]/[-1,1] ranges so the small net needs no LayerNorm. The
    `signal_type` one-hot (§7.C) tells the shared net which signal this is, so it
    can learn a per-signal marginal value while sharing the felt-context weights."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float64)
    feat[0] = float(np.clip(nudge.surprise / 5.0, 0.0, 3.0))   # surprise (scaled)
    feat[1] = float(min(1.0, math.log1p(max(0, nudge.n)) / n_freq_scale))
    feat[2] = float(np.clip(abs(nudge.rate - nudge.mu_before), 0.0, 1.0))
    if emot_state:
        feat[3] = float(np.clip(emot_state.get("V_blended", 0.0), -1.0, 1.0))
        idx = int(emot_state.get("dominant_idx", -1))
        if 0 <= idx < 8:
            feat[4 + idx] = 1.0
    sidx = SIGNAL_INDEX.get(signal_type)
    if sidx is not None:
        feat[_BASE_FEATURE_DIM + sidx] = 1.0
    return feat


def signed_delta_target(pre_v: Optional[float],
                        post_v: Optional[float]) -> Optional[float]:
    """The Hybrid attribution target (RFP §7.B): the SIGNED emot-valence drift
    across the wake cycle, measured on the dominant **blended valence**
    `V_blended` (`ShmEmotReader.read_state` → 'V_blended').

      target = post_V_blended − pre_V_blended

    LIVE-VERIFIED CHOICE (2026-06-13): the original design used the 8-D
    `read_grounding` per-primitive vector, but an online proof on T2 found the
    grounding SHM slot is NEVER written fleet-wide (`write_grounding` is defined
    but never called → header n=0 → `read_grounding()` returns None always). So
    the 8-D target would skip EVERY nudge → the net would never train. `V_blended`
    (the dominant blended valence) IS live-versioned in `emot_state.bin` and is the
    headline 'did the felt-state's valence move' signal — the faithful, available
    attribution scalar. Sign is intrinsic (rose → +, fell → −); the net learns to
    predict its magnitude per signal → habituation emerges as the drift flattens.
    Returns None when either snapshot is unavailable (skip — never train on a
    fabricated delta, INV-AFF-HONEST)."""
    if pre_v is None or post_v is None:
        return None
    return float(post_v) - float(pre_v)


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
                    sd = {k: data[k] for k in data.files}
                # §7.C feature-dim guard: a net saved under a DIFFERENT input dim
                # (e.g. a pre-Phase-C 12-D net before the signal_type one-hot) is
                # incompatible with the current FEATURE_DIM feature layout. Discard
                # it + start fresh (cold-start → Phase-A formula until re-trained)
                # rather than crash on a shape mismatch at the first forward().
                w1 = np.asarray(sd.get("W1"))
                if w1.ndim == 2 and int(w1.shape[1]) == FEATURE_DIM:
                    net.load_state_dict(sd)
                else:
                    logger.info(
                        "[affective_nudge] saved net in_dim=%s != FEATURE_DIM=%d "
                        "— starting fresh (Phase C feature-layout change)",
                        (int(w1.shape[1]) if w1.ndim == 2 else "?"), FEATURE_DIM)
        except Exception as e:
            logger.debug("[affective_nudge] net load failed (fresh net): %s", e)
        return net


@dataclass
class _PendingNudge:
    """A fired nudge awaiting its dream-boundary emot-delta attribution.
    `pre_v` = the dominant blended valence (`V_blended`) snapshotted at emit."""
    features: np.ndarray
    pre_v: Optional[float]
    ts: float


class AffectiveNudgeRuntime:
    """Owns the per-Titan net + the wake-cycle pending buffer. Shared between the
    synthesis drain thread (forward + record pre-snapshot) and the single
    `synthesis-dream` thread (train on post-snapshot) — both in synthesis_worker,
    so no SHM round-trip. Thread-safe via one lock around the pending buffer + the
    net weights (forward/train never run concurrently in practice — drain is light,
    dream is the ordered sequence — but the lock keeps it correct).

    `emot_state_reader` is a zero-arg callable returning the emot SHM
    `read_state()` dict — the source of BOTH the felt-context features
    (`V_blended`, `dominant_idx`) AND the attribution valence (`V_blended`). It is
    injected so the runtime stays decoupled from emot_shm_protocol (+ unit-testable).
    `emot_grounding_reader` is accepted for backward-compat but UNUSED — the 8-D
    grounding SHM slot is never written fleet-wide (`write_grounding` uncalled), so
    attribution rides the live `V_blended` from `read_state` (see signed_delta_target)."""

    def __init__(self, cfg: AffectiveConfig, state_path: str, net_path: str,
                 *, emot_state_reader=None, emot_grounding_reader=None,
                 max_pending: int = 256):
        self.cfg = cfg
        self.state_path = state_path
        self.net_path = net_path
        self._emot_state_reader = emot_state_reader
        self._emot_grounding_reader = emot_grounding_reader  # unused (see docstring)
        self._max_pending = int(max_pending)
        self._lock = threading.Lock()
        self._pending: List[_PendingNudge] = []
        self.net = AffectiveNudgeNet.load_npz(net_path, hidden=cfg.net_hidden)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _read_emot_state(self) -> Optional[dict]:
        if self._emot_state_reader is None:
            return None
        try:
            return self._emot_state_reader()
        except Exception:
            return None

    def _read_valence(self) -> Optional[float]:
        """The live attribution scalar = dominant blended valence (`V_blended`)
        from emot `read_state` (the grounding 8-vector is never written — see
        signed_delta_target). None if emot SHM is unavailable."""
        st = self._read_emot_state()
        if not st or "V_blended" not in st:
            return None
        try:
            return float(st["V_blended"])
        except Exception:
            return None

    # ── drain-tick path (forward) ──────────────────────────────────────────────
    def _observe(self, signal_type: str, base_nudge: Optional[Nudge],
                 ts: float) -> Optional[Nudge]:
        """Shared forward path for EVERY signal (RFP §7.C). Given the signal's
        already-folded base nudge (from compute_skill_score_nudge / compute_event
        _nudge), builds the feature vector WITH the signal_type one-hot, OVERRIDES
        the magnitude with the learned net (once trained; else keeps the Phase-A
        formula magnitude = cold-start fallback), and records the pre-nudge emot
        snapshot so the dream boundary can attribute the resulting drift.

        🔒 Single-consumer contract: all observe_* calls run on the ONE synthesis
        drain thread, so the per-signal EMA baseline file (written inside the
        compute_* helper) needs no extra lock; `self._lock` still guards the net
        weights + pending buffer (shared with the dream-thread trainer)."""
        if base_nudge is None:
            return None

        emot_state = self._read_emot_state()
        features = build_features(base_nudge, emot_state, signal_type=signal_type)

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

        pre_v = self._read_valence()
        with self._lock:
            self._pending.append(_PendingNudge(features=features, pre_v=pre_v,
                                               ts=float(ts)))
            if len(self._pending) > self._max_pending:
                self._pending = self._pending[-self._max_pending:]
        return nudge

    def observe_drain(self, successes: int, failures: int,
                      ts: float) -> Optional[Nudge]:
        """Phase-A/B skill_score forward at a drain tick. Returns the Nudge to
        emit, or None (warming up / on-baseline / no events)."""
        base_nudge = compute_skill_score_nudge(
            successes, failures, self.state_path, cfg=self.cfg)
        return self._observe("skill_score", base_nudge, ts)

    def observe_signal(self, signal_type: str, signed_delta: float, ts: float,
                       *, intrinsic_positive: bool = False) -> Optional[Nudge]:
        """Phase-C event-signal forward (sol_receipt / maker_bond / x_engagement /
        chain_reuse). `signed_delta` = the raw state-delta on its natural scale;
        `intrinsic_positive` forces a + valence for can't-be-negative signals
        (engagement / reuse / maker_bond). Returns the Nudge or None."""
        base_nudge = compute_event_nudge(
            signed_delta, self.state_path, signal_type=signal_type, cfg=self.cfg,
            intrinsic_positive=intrinsic_positive)
        return self._observe(signal_type, base_nudge, ts)

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

        post_v = self._read_valence()
        X_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        skipped = 0
        for p in pending:
            target = signed_delta_target(p.pre_v, post_v)
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
