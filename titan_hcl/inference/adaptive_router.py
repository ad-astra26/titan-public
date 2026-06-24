"""titan_hcl/inference/adaptive_router.py — Load-adaptive inference routing (Phase B).

RFP_load_adaptive_inference_routing §7.B. A contextual-bandit router that LEARNS to
offload chat from a saturated `gemma4:31b` to fast separate-pool fallback models under
load, and reverts to gemma as it cools — maximizing a config-weighted COMPOSITE reward
(responsiveness + quality + cost). The threshold heuristic is only the cold-start prior
until the bandit has data (the Affective-Loop pattern). Learned policy persists to
`data/inference_router_state.json` (save/resume).

SAFE-BY-DEFAULT: `choose()` is a pure passthrough to `provider.resolve_model_class`
when the router is disabled, the call is non-chat, or the class doesn't resolve to the
managed model — so wiring it in changes NOTHING until `router_enabled=true`. Every path
soft-fails to the plain resolution. Off the chat hot path (O(1) dict pick + EMA).
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STATE_SCHEMA = 1


def load_bucket(in_flight: int) -> str:
    """Coarse concurrency bucket (gemma sweet-spot ≈ ≤8 warm)."""
    if in_flight <= 2:
        return "low"
    if in_flight <= 5:
        return "mid"
    return "high"


class InferenceLatencyMonitor:
    """Per-model EMA of recent chat-completion latency (seconds)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = float(alpha)
        self._ema: dict[str, float] = {}

    def record(self, model: str, latency_s: float) -> None:
        a = self._alpha
        prev = self._ema.get(model)
        self._ema[model] = (latency_s if prev is None
                            else (1.0 - a) * prev + a * float(latency_s))

    def ema(self, model: str, default: float = 0.0) -> float:
        return self._ema.get(model, default)


class AdaptiveRouter:
    """Contextual-bandit model router (§7.B)."""

    def __init__(self, cfg: Optional[dict] = None,
                 managed_model: str = "gemma4:31b") -> None:
        self.cfg = cfg or {}
        self.managed = managed_model
        self.ladder = list(self.cfg.get(
            "model_ladder",
            [managed_model, "ministral-3:14b", "gemini-3-flash-preview"]))
        if self.managed not in self.ladder:
            self.ladder = [self.managed] + self.ladder
        self.monitor = InferenceLatencyMonitor()
        self.state_path = str(self.cfg.get(
            "router_state_path", "data/inference_router_state.json"))
        # reward tables: {bucket: {arm: {"r": ema_reward, "n": count}}}
        self._table: dict[str, dict[str, dict]] = {}
        # flap guardrail state
        self._current_arm = managed_model
        self._switched_at = 0.0
        self._last_logged: Optional[str] = None   # §8 G3 — log only on route change
        # quality / cost priors (B; B.2 will LEARN quality from the turn-judge).
        # gemma is the strongest + priciest; fallbacks descend.
        _q = [1.0, 0.72, 0.66, 0.6]
        _c = [0.3, 0.7, 0.82, 0.9]   # cheaper → higher cost-reward
        self._qprior = {m: _q[min(i, len(_q) - 1)] for i, m in enumerate(self.ladder)}
        self._cprior = {m: _c[min(i, len(_c) - 1)] for i, m in enumerate(self.ladder)}
        self._load()

    # ── config knobs (re-read live so config hot-reload tunes the router) ──
    def _f(self, k: str, d: float) -> float:
        try:
            return float(self.cfg.get(k, d))
        except Exception:
            return d

    def _enabled(self) -> bool:
        return bool(self.cfg.get("router_enabled", False))

    def update_cfg(self, cfg: dict) -> None:
        """Live config hot-reload — adopt new knobs without a restart."""
        if cfg:
            self.cfg = cfg

    def _bucket(self, in_flight: int, is_chat: bool) -> str:
        return f"{load_bucket(int(in_flight))}|{'chat' if is_chat else 'task'}"

    # ── the routing decision ──
    def choose(self, provider: Any, model_class: str,
               in_flight: int = 0, is_chat: bool = True) -> str:
        """Concrete model id for `model_class`. PASSTHROUGH (zero change) unless the
        router is enabled, the call is chat, AND the class resolves to the managed
        model. Soft-fails to the plain resolution."""
        base = provider.resolve_model_class(model_class)
        if not self._enabled() or not is_chat or base != self.managed:
            return base
        try:
            chosen = self._bandit_choose(int(in_flight), bool(is_chat))
            if chosen != self._last_logged:
                # §8 G3 observability — log every change of routing decision (offload
                # to a fallback, or revert to gemma). Low-frequency (only on change).
                logger.info(
                    "[AdaptiveRouter] route → %s (was %s, in_flight=%d gemma_ema=%.1fs)",
                    chosen, self._last_logged or self.managed, int(in_flight),
                    self.monitor.ema(self.managed))
                self._last_logged = chosen
            return chosen
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] choose soft-fail → passthrough: %s", e)
            return base

    def _bandit_choose(self, in_flight: int, is_chat: bool) -> str:
        bucket = self._bucket(in_flight, is_chat)
        arms = self._table.get(bucket, {})
        n_total = sum(int(a.get("n", 0)) for a in arms.values())
        if n_total < int(self._f("min_samples", 20)):
            chosen = self._heuristic(in_flight)            # cold-start prior
        elif random.random() < self._f("explore_eps", 0.08):
            chosen = random.choice(self.ladder)            # bounded exploration
        else:
            chosen = max(self.ladder,
                         key=lambda m: arms.get(m, {}).get("r", 0.0))
        return self._apply_dwell(chosen)

    def _heuristic(self, in_flight: int) -> str:
        """Cold-start threshold prior: offload when gemma is saturated/cold."""
        if (in_flight > int(self._f("in_flight_ceiling", 8))
                or self.monitor.ema(self.managed) > self._f("enter_latency_s", 12.0)):
            for m in self.ladder:
                if m != self.managed:
                    return m
        return self.managed

    def _apply_dwell(self, chosen: str) -> str:
        """Flap guardrail (§Ad-5): hold a fallback for min_dwell, and revert to gemma
        ONLY when it is confirmed-warm — reverting into a cold gemma re-triggers the
        ~60s spike."""
        now = time.time()
        if chosen == self.managed and self._current_arm != self.managed:
            if (now - self._switched_at) < self._f("min_dwell_s", 20.0):
                return self._current_arm
            if self.monitor.ema(self.managed) > self._f("exit_latency_s", 7.0):
                return self._current_arm   # gemma not confirmed-warm → keep fallback
        if chosen != self._current_arm:
            self._current_arm = chosen
            self._switched_at = now
        return chosen

    # ── the reward loop ──
    def feedback(self, model_id: str, latency_s: float,
                 in_flight: int = 0, is_chat: bool = True) -> None:
        """Fold an observed completion into the monitor + the bandit reward table."""
        if not self._enabled():
            return
        try:
            self.monitor.record(model_id, float(latency_s))
            bucket = self._bucket(int(in_flight), bool(is_chat))
            r = self._reward(model_id, float(latency_s), bool(is_chat))
            arms = self._table.setdefault(bucket, {})
            arm = arms.setdefault(model_id, {"r": r, "n": 0})
            a = 0.2
            arm["r"] = r if int(arm["n"]) == 0 else (1.0 - a) * arm["r"] + a * r
            arm["n"] = int(arm["n"]) + 1
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] feedback soft-fail: %s", e)

    def _reward(self, model_id: str, latency_s: float, is_chat: bool) -> float:
        sfx = "chat" if is_chat else "task"
        wl = self._f(f"w_latency_{sfx}", 0.6 if is_chat else 0.25)
        wq = self._f(f"w_quality_{sfx}", 0.3 if is_chat else 0.55)
        wc = self._f(f"w_cost_{sfx}", 0.1 if is_chat else 0.2)
        target = self._f(f"target_latency_{sfx}_s", 12.0 if is_chat else 40.0)
        # responsiveness: 1.0 at 0s, 0.5 at target, 0.0 at ≥2×target
        responsiveness = max(0.0, min(1.0, 1.0 - latency_s / (2.0 * max(1e-6, target))))
        q = self._qprior.get(model_id, 0.6)
        c = self._cprior.get(model_id, 0.5)
        return wl * responsiveness + wq * q + wc * c

    # ── persistence ──
    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                d = json.load(open(self.state_path))
                if int(d.get("schema", 0)) == _STATE_SCHEMA:
                    self._table = d.get("buckets", {}) or {}
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] state load failed (fresh start): %s", e)

    def save(self) -> None:
        try:
            d = os.path.dirname(self.state_path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"schema": _STATE_SCHEMA, "buckets": self._table}, f)
            os.replace(tmp, self.state_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] state save failed: %s", e)
