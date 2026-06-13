"""Affective Grounding Loop — Phase A substrate-injection helper.

RFP_affective_grounding_loop.md §7.A. Builds the *missing channel* from a real
event (here: synthesis skill-score outcomes) to a gentle, decaying nudge on the
neuromod substrate, so good/bad things that actually happen to Titan move its
inner felt-state. The 2026-06-10 experiment proved this channel was ABSENT —
5 warm chat turns moved T1's emotion by exactly zero.

PHASE A SCOPE (this file): ONE source (skill-score), ONE modulator (`DA`,
achievement/competence per RFP D4), a FIXED surprise formula. The emergent
`AffectiveNudgeNet` (D2) replaces the formula in Phase B — but inherits the
per-signal EMA(μ,σ) baseline computed here (surprise + habituation seed).

HONESTY / EMERGENCE (the RFP invariants this file must satisfy):
  - INV-AFF-EMERGENT  : magnitude = surprise × a config gain, NOT a hardcoded
                        per-event magnitude. Surprise = |rate − μ| / (σ + ε)
                        against the signal's OWN running baseline.
  - INV-AFF-HONEST    : valence is the SIGN of the real competence delta
                        (rate above baseline → +, below → −). Symmetric. No
                        forced positivity, no floor.
  - habituation (G3)  : a frequent signal at a stable rate drives μ → rate →
                        surprise → 0 → magnitude → 0 BY CONSTRUCTION. A rare
                        deviation re-spikes surprise. Nothing hardcoded.

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
from dataclasses import dataclass
from typing import Optional

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
