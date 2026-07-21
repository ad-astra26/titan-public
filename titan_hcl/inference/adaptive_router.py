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

_STATE_SCHEMA = 2   # v2 (B.2) — adds learned per-model quality EMA to the state


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
        # R2.3 (§R2 2026-07-08) — WARMTH signal, fed by keepalive pings, kept
        # SEPARATE from the chat-latency ema. A keepalive ping is tiny (8 tokens)
        # so its latency reflects model WARMTH (loaded vs cold), NOT full-turn
        # latency; mixing it into `_ema` would make gemma look fast + corrupt the
        # offload-enter signal + the bandit-adjacent latency. Used ONLY by the
        # revert-when-confirmed-warm gate (which asks "is gemma loaded?", not "how
        # slow is a real turn?"). Updates every keepalive tick → unfreezes the
        # revert decision while offloaded (the ema alone freezes with no real
        # gemma turns to feed it → offload was one-way/sticky).
        self._warmth: dict[str, float] = {}

    def record(self, model: str, latency_s: float) -> None:
        a = self._alpha
        prev = self._ema.get(model)
        self._ema[model] = (latency_s if prev is None
                            else (1.0 - a) * prev + a * float(latency_s))

    def ema(self, model: str, default: float = 0.0) -> float:
        return self._ema.get(model, default)

    def record_warmth(self, model: str, latency_s: float) -> None:
        """R2.3 — fold a keepalive-ping latency into the model's warmth EMA."""
        a = self._alpha
        prev = self._warmth.get(model)
        self._warmth[model] = (float(latency_s) if prev is None
                               else (1.0 - a) * prev + a * float(latency_s))

    def warmth(self, model: str, default: float = 999.0) -> float:
        """R2.3 — ping-based warmth (seconds). Default HIGH = 'not confirmed warm'
        (no ping seen yet → don't revert into a possibly-cold model)."""
        return self._warmth.get(model, default)


class AdaptiveRouter:
    """Contextual-bandit model router (§7.B)."""

    def __init__(self, cfg: Optional[dict] = None,
                 managed_model: str = "gemma4:31b") -> None:
        self.cfg = cfg or {}
        self.managed = managed_model
        # The offload ladder MUST be models the live provider serves AND whose
        # quality holds under load. ⚠ PROVIDER MODELS ROT: Ollama Cloud silently
        # retired BOTH prior offload arms on 2026-07-15 → every load-offload
        # failed with "model was retired" until
        # re-probed 2026-07-21. Re-probed live 2026-07-21 (warm latency + a
        # quality read against the LIVE provider model list):
        #   • gpt-oss:20b — 1.7s warm, fluent + coherent chat — the offload arm.
        #     (nemotron-3-nano:30b was fast but telegraphic; deepseek-v4-flash
        #     returned empty — both quality-disqualified as chat arms.)
        # ALWAYS validate a new arm against the live provider list before adding.
        # B.2 learns each arm's REAL in-situ quality from the turn-judge, so a poor
        # arm self-corrects. Override per-install via [inference.autoscale] model_ladder.
        self.ladder = list(self.cfg.get(
            "model_ladder",
            [managed_model, "gpt-oss:20b"]))
        if self.managed not in self.ladder:
            self.ladder = [self.managed] + self.ladder
        self.monitor = InferenceLatencyMonitor()
        # B.2 — LEARNED per-model quality, folded from the synthesis turn-judge reward
        # (MODEL_QUALITY_FEEDBACK). Replaces the static `_qprior` once a model has
        # `min_quality_samples` observations; prior is the fallback until then.
        self._quality_ema: dict[str, float] = {}
        self._quality_n: dict[str, int] = {}
        self.state_path = str(self.cfg.get(
            "router_state_path", "data/inference_router_state.json"))
        # reward tables: {bucket: {arm: {"r": ema_reward, "n": count}}}
        self._table: dict[str, dict[str, dict]] = {}
        # flap guardrail state
        self._current_arm = managed_model
        self._switched_at = 0.0
        self._last_logged: Optional[str] = None   # §8 G3 — log only on route change
        # quality / cost priors (B; B.2 will LEARN quality from the turn-judge).
        self._rebuild_priors()
        self._load()

    # ── config knobs (re-read live so config hot-reload tunes the router) ──
    def _f(self, k: str, d: float) -> float:
        try:
            return float(self.cfg.get(k, d))
        except Exception:
            return d

    def _enabled(self) -> bool:
        # Default ON (Maker 2026-06-24): the §7.B0 concurrency prereq has landed
        # + is verified, so the load-balancing the router provides is meaningful.
        # Per the all-flags-default-on rule, ship ON fleet-wide; router_enabled=
        # false is a kill-switch only. choose() is still a no-op for a single
        # warm request (gemma_ema < enter_latency, in_flight low) → it only
        # offloads when gemma is actually slow/saturated, so full-quality gemma
        # serves the common case and fast fallbacks absorb concurrent spikes.
        return bool(self.cfg.get("router_enabled", True))

    def _rebuild_priors(self) -> None:
        """Quality/cost priors indexed by ladder position — the managed head is the
        strongest + priciest; fallbacks descend. B.2 LEARNS the real quality from the
        turn-judge and supersedes these once a model has min_quality_samples obs.
        Single source of truth for the priors so __init__ and update_cfg agree."""
        _q = [1.0, 0.72, 0.66, 0.6]
        _c = [0.3, 0.7, 0.82, 0.9]   # cheaper → higher cost-reward
        self._qprior = {m: _q[min(i, len(_q) - 1)] for i, m in enumerate(self.ladder)}
        self._cprior = {m: _c[min(i, len(_c) - 1)] for i, m in enumerate(self.ladder)}

    def update_cfg(self, cfg: dict) -> None:
        """Live config hot-reload — adopt new knobs without a restart. Re-derives the
        offload ladder + priors so a live [inference.autoscale] model_ladder change
        actually takes effect (config_schema declares model_ladder reload='hot').
        Before 2026-07-21 this only swapped self.cfg and left self.ladder cached from
        __init__ — a FALSE hot-reload that stranded a retired-model ladder live until
        the next agno restart (the 2026-07-15 offload-arm retirement)."""
        if not cfg:
            return
        self.cfg = cfg
        new_ladder = list(cfg.get("model_ladder", self.ladder))
        if self.managed not in new_ladder:
            new_ladder = [self.managed] + new_ladder
        if new_ladder != self.ladder:
            self.ladder = new_ladder
            self._rebuild_priors()

    def _heaviness_bucket(self, heaviness: float) -> str:
        """Coarse session-depth dim (§7.C). 2-level keeps the contextual bandit's
        bucket space small → fast convergence; the reward-modulation carries the
        nuance (heavier ⇒ quality weighted up)."""
        return "heavy" if float(heaviness) >= self._f("heaviness_threshold", 0.5) else "light"

    def _bucket(self, in_flight: int, is_chat: bool, heaviness: float = 0.0) -> str:
        return (f"{load_bucket(int(in_flight))}|{self._heaviness_bucket(heaviness)}"
                f"|{'chat' if is_chat else 'task'}")

    # ── the routing decision ──
    def choose(self, provider: Any, model_class: str,
               in_flight: int = 0, is_chat: bool = True,
               heaviness: float = 0.0) -> str:
        """Concrete model id for `model_class`. PASSTHROUGH (zero change) unless the
        router is enabled, the call is chat, AND the class resolves to the managed
        model. `heaviness` ∈ [0,1] (§7.C) makes the bandit keep gemma for deep sessions
        (context dim + reward-modulation). Soft-fails to the plain resolution."""
        base = provider.resolve_model_class(model_class)
        if not self._enabled() or not is_chat or base != self.managed:
            return base
        try:
            chosen = self._bandit_choose(int(in_flight), bool(is_chat), float(heaviness))
            if chosen != self._last_logged:
                # §8 G3 observability — log every change of routing decision (offload
                # to a fallback, or revert to gemma). Low-frequency (only on change).
                logger.info(
                    "[AdaptiveRouter] route → %s (was %s, in_flight=%d gemma_ema=%.1fs "
                    "heaviness=%.2f)",
                    chosen, self._last_logged or self.managed, int(in_flight),
                    self.monitor.ema(self.managed), float(heaviness))
                self._last_logged = chosen
            return chosen
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] choose soft-fail → passthrough: %s", e)
            return base

    def _bandit_choose(self, in_flight: int, is_chat: bool,
                       heaviness: float = 0.0) -> str:
        bucket = self._bucket(in_flight, is_chat, heaviness)
        arms = self._table.get(bucket, {})
        n_total = sum(int(a.get("n", 0)) for a in arms.values())
        # INV-AR-EXPLORE-BOUNDED (§7.C extension) — never offload a DEEP session to a
        # fallback purely to explore when load is high; a wrong exploration there costs
        # the most (a long, invested conversation gets a weaker model). Clamp ε to 0.
        heavy = self._heaviness_bucket(heaviness) == "heavy"
        high_load = in_flight > int(self._f("in_flight_ceiling", 8))
        eff_eps = 0.0 if (heavy and high_load) else self._f("explore_eps", 0.08)
        if n_total < int(self._f("min_samples", 20)):
            chosen = self._heuristic(in_flight)            # cold-start prior
        elif random.random() < eff_eps:
            chosen = random.choice(self.ladder)            # bounded exploration
        else:
            chosen = max(self.ladder,
                         key=lambda m: arms.get(m, {}).get("r", 0.0))
        return self._apply_dwell(chosen)

    def _heuristic(self, in_flight: int) -> str:
        """Cold-start threshold prior: offload when gemma is saturated/cold."""
        # enter_latency stays the conservative 12s (Maker 2026-06-25): the pitch
        # route bypasses the router (fixed per-Titan models), so a lower threshold
        # would help nothing there AND would make the official /chat route offload
        # gemma4:31b→a smaller model more aggressively — an unwanted quality
        # downgrade for /chat, which is one-user-at-a-time and stays on gemma4:31b
        # (kept fast by the always-warm keepalive). 12s remains a rare safety valve.
        if (in_flight > int(self._f("in_flight_ceiling", 8))
                or self.monitor.ema(self.managed) > self._f("enter_latency_s", 12.0)):
            for m in self.ladder:
                if m != self.managed:
                    return m
        return self.managed

    def _apply_dwell(self, chosen: str) -> str:
        """Flap guardrail (§Ad-5 + R2.2). Three cases for a switch off `_current_arm`:
        (a) revert TO gemma — hold `min_dwell` AND revert only when gemma is
            confirmed-warm (< `exit_latency`); reverting into a cold gemma re-triggers
            the ~60s spike. (b) R2.2 (§R2 2026-07-08) fallback→DIFFERENT-fallback —
            hold `min_dwell` too: the old code guarded ONLY revert-to-gemma, so
            ministral↔gemma3 flapped every few seconds (observed) → repeated
            cold-starts. (c) gemma→fallback ESCALATION stays IMMEDIATE (offload the
            moment gemma is slow — responsiveness must not be dwell-blocked)."""
        now = time.time()
        if chosen == self._current_arm:
            return chosen
        held = (now - self._switched_at) < self._f("min_dwell_s", 20.0)
        if chosen == self.managed and self._current_arm != self.managed:
            # (a) revert to gemma — R2.3: gate on WARMTH (ping-fed, live during
            # offload), not the frozen chat-latency ema. warmth default is HIGH so
            # a never-pinged gemma is treated as not-confirmed-warm.
            if held or self.monitor.warmth(
                    self.managed, 999.0) > self._f("exit_latency_s", 7.0):
                return self._current_arm
        elif self._current_arm != self.managed and held:
            # (b) anti-flap between fallbacks — hold the current fallback
            return self._current_arm
        # (c) escalation (gemma→fallback) or a dwell-cleared switch — commit
        self._current_arm = chosen
        self._switched_at = now
        return chosen

    # ── the reward loop ──
    def feedback(self, model_id: str, latency_s: float,
                 in_flight: int = 0, is_chat: bool = True,
                 heaviness: float = 0.0) -> None:
        """Fold an observed completion into the monitor + the bandit reward table.
        `heaviness` must be the SAME value used at the routing decision for this turn
        (§7.C) so the reward lands in the right (load × heaviness) bucket and the
        quality-weighted composite matches the decision context."""
        if not self._enabled():
            return
        try:
            self.monitor.record(model_id, float(latency_s))
            bucket = self._bucket(int(in_flight), bool(is_chat), float(heaviness))
            r = self._reward(model_id, float(latency_s), bool(is_chat), float(heaviness))
            arms = self._table.setdefault(bucket, {})
            arm = arms.setdefault(model_id, {"r": r, "n": 0})
            a = 0.2
            arm["r"] = r if int(arm["n"]) == 0 else (1.0 - a) * arm["r"] + a * r
            arm["n"] = int(arm["n"]) + 1
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] feedback soft-fail: %s", e)

    # ── B.2 — learned quality (turn-judge) ──
    def feedback_quality(self, model_id: str, judge_reward: float) -> None:
        """Fold a synthesis turn-judge reward for `model_id` into its quality EMA.
        The judge reward is in [-1, +1] (good/ok/poor × confidence); normalize to the
        [0,1] quality scale (good→1.0, ok→0.5, poor→0.0) so it is comparable to the
        static `_qprior` band it displaces. Async, off the chat hot path, soft-fail."""
        if not self._enabled() or not model_id:
            return
        try:
            q_obs = max(0.0, min(1.0, (float(judge_reward) + 1.0) / 2.0))
            a = self._f("quality_alpha", 0.2)
            prev = self._quality_ema.get(model_id)
            self._quality_ema[model_id] = (q_obs if prev is None
                                           else (1.0 - a) * prev + a * q_obs)
            self._quality_n[model_id] = self._quality_n.get(model_id, 0) + 1
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] feedback_quality soft-fail: %s", e)

    def _q_for(self, model_id: str) -> float:
        """The quality term for the composite reward: the LEARNED EMA once the model
        has `min_quality_samples` judged turns, else the static prior (cold-start)."""
        if (self._quality_n.get(model_id, 0) >= int(self._f("min_quality_samples", 15))
                and model_id in self._quality_ema):
            return self._quality_ema[model_id]
        return self._qprior.get(model_id, 0.6)

    def _reward(self, model_id: str, latency_s: float, is_chat: bool,
                heaviness: float = 0.0) -> float:
        sfx = "chat" if is_chat else "task"
        wl = self._f(f"w_latency_{sfx}", 0.6 if is_chat else 0.25)
        wq = self._f(f"w_quality_{sfx}", 0.3 if is_chat else 0.55)
        wc = self._f(f"w_cost_{sfx}", 0.1 if is_chat else 0.2)
        target = self._f(f"target_latency_{sfx}_s", 12.0 if is_chat else 40.0)
        # §7.C reward-modulation (fully-emergent, no deterministic pin): a heavier
        # session weights QUALITY up and RELAXES the latency target, so — within the
        # heavy bucket — the bandit LEARNS to keep gemma (its quality dominates the
        # composite) while light sessions stay latency-tight and offload first. Both
        # scales are 0 at heaviness=0 → identical to the pre-C reward.
        h = max(0.0, min(1.0, float(heaviness)))
        wq = wq * (1.0 + h * self._f("heaviness_quality_boost", 1.0))
        target = target * (1.0 + h * self._f("heaviness_latency_relax", 1.5))
        # R2.4 (§R2, 2026-07-08) — responsiveness curve. The old form
        #   max(0, 1 - latency/(2·target))  CLAMPED to 0 at ≥2×target, so under
        # heavy load where every model exceeds 24s the latency signal FLATTENED
        # (30s and 90s both scored 0) → the bandit tipped to quality (gemma4, the
        # wrong pick under load). Replace with a monotonic, never-saturating curve:
        #   target/(target+latency)  → 1.0 @0s, 0.5 @target, 0.33 @2×target, →0 as
        # latency→∞ but ALWAYS strictly decreasing, so a faster model always
        # out-scores a slower one even when both are slow. Pure latency-signal fix;
        # quality/cost terms unchanged (INV-R2-NO-QUALITY).
        responsiveness = max(1e-6, target) / (max(1e-6, target) + max(0.0, latency_s))
        q = self._q_for(model_id)          # B.2 — learned quality, prior as fallback
        c = self._cprior.get(model_id, 0.5)
        return wl * responsiveness + wq * q + wc * c

    # ── persistence ──
    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                d = json.load(open(self.state_path))
                # Accept v1 AND v2 — a v1 file (bandit table only) upgrades in place;
                # its learned buckets are preserved, quality starts empty (re-learns).
                if int(d.get("schema", 0)) in (1, _STATE_SCHEMA):
                    self._table = d.get("buckets", {}) or {}
                    self._quality_ema = d.get("quality_ema", {}) or {}   # B.2 (absent in v1)
                    self._quality_n = d.get("quality_n", {}) or {}
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] state load failed (fresh start): %s", e)

    def save(self) -> None:
        try:
            d = os.path.dirname(self.state_path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"schema": _STATE_SCHEMA, "buckets": self._table,
                           "quality_ema": self._quality_ema,
                           "quality_n": self._quality_n}, f)
            os.replace(tmp, self.state_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("[AdaptiveRouter] state save failed: %s", e)
