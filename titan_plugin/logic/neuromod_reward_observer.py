"""
titan_plugin/logic/neuromod_reward_observer.py
NeuromodRewardObserver — rFP β Stage 2 Phase 2b.

Bridges Titan's biological-analog neuromodulator dynamics to NS program
training. The 6 neuromods (DA, 5-HT, NE, ACh, Endorphin, GABA) are
already Titan's natural reward currency — this observer reads them
every N ticks and emits per-program reward signals via
NeuralNervousSystem.record_outcome(reward, program=, source=).

The PRIMARY reward pathway for outer/personality programs (and
VIGILANCE), as designed in rFP β § 4a Option 4 + § 4h. Discrete event
hooks (Phase 2c) provide SECONDARY high-magnitude calibration signals.

Design intent (from rFP β session 2026-04-16 Q3):
"NE = vigilance neuromod in biology. Neuromods already ARE Titan's
biological reward currency — we just weren't listening to them for NS
training. Use neuromod dynamics as the PRIMARY reward signal for Layer 2
(outer/personality) programs, keep discrete event hooks as SECONDARY."

This observer does NOT directly modify the neural networks — it only
emits rewards via record_outcome. All training mechanics (z-normalize,
eligibility traces, soft-fire, stratified sampling) live in NeuralNervousSystem.

Key characteristics:
- Stateful: tracks per-neuromod EMAs to detect spikes/rises
- Stateless w.r.t. NS: never touches NN weights directly
- Rate-limited: emits at most once per program per tick_interval (default 10)
- Bounded magnitudes: rewards in [-1.0, +1.0], z-normalized downstream
- Symmetry-respecting: positive AND negative pathways for every program
"""
import logging
import math
import time
from collections import deque

logger = logging.getLogger(__name__)


# Per-program neuromod mapping (rFP β § 4a Option 4 + Q3 design)
# Each entry: (positive_signal, negative_signal) — functions of the
# current neuromod state. Functions take a NeuromodState and return
# raw reward in [-1, +1] before downstream z-normalization.

class NeuromodState:
    """Snapshot of all 6 neuromod levels + their recent EMA stats."""
    def __init__(self, levels: dict, ema_means: dict, ema_stds: dict):
        self.levels = dict(levels)            # current raw {DA, 5-HT, NE, ACh, Endorphin, GABA}
        self.ema_means = dict(ema_means)
        self.ema_stds = dict(ema_stds)

    def get(self, name: str) -> float:
        return float(self.levels.get(name, 0.0))

    def z(self, name: str) -> float:
        """Z-score of current level vs rolling EMA. Spikes return > 1.5."""
        mean = self.ema_means.get(name, 0.0)
        std = max(self.ema_stds.get(name, 0.01), 0.01)
        return (self.get(name) - mean) / std

    def is_spike(self, name: str, threshold: float = 1.5) -> bool:
        return self.z(name) > threshold

    def is_rise(self, name: str, threshold: float = 0.5) -> bool:
        return self.z(name) > threshold


# Reward-signal generators — each returns reward ∈ [-1, +1]


def _focus_reward(s: NeuromodState) -> float:
    """FOCUS: high sustained ACh = good attention maintenance.
    Positive when ACh rising; negative when ACh dropping during arousal."""
    ach_z = s.z("ACh")
    if ach_z > 0.5:
        return min(1.0, 0.4 + 0.3 * ach_z)
    elif ach_z < -1.0:
        return max(-1.0, -0.3)
    return 0.0


def _impulse_reward(s: NeuromodState) -> float:
    """IMPULSE: DA-ACh co-activation = ready-to-act state.
    Positive when both DA and ACh elevated; negative if action without
    follow-through (DA spike but ACh not engaged → low downstream effect)."""
    da_lvl = s.get("DA")
    ach_lvl = s.get("ACh")
    if da_lvl > 0.5 and ach_lvl > 0.5:
        return min(1.0, 0.5 * (da_lvl + ach_lvl) - 0.3)
    if da_lvl > 0.6 and ach_lvl < 0.3:
        return -0.2  # impulsive without attention
    return 0.0


def _intuition_reward(s: NeuromodState) -> float:
    """INTUITION: low GABA + moderate DA = following hunches.
    Positive when GABA disinhibits AND DA suggests reward signal present."""
    gaba = s.get("GABA")
    da = s.get("DA")
    if gaba < 0.3 and da > 0.4:
        return min(1.0, 0.4 + 0.3 * (1.0 - gaba))
    if gaba > 0.7:  # too much inhibition
        return -0.1
    return 0.0


def _metabolism_reward(s: NeuromodState) -> float:
    """METABOLISM: high GABA + low arousal = energy-conservation state.
    Distinct from spirit_worker's drain_delta hook (Phase 2c) which is
    additive — both pathways feed METABOLISM training."""
    gaba = s.get("GABA")
    ne = s.get("NE")
    if gaba > 0.5 and ne < 0.4:
        return min(1.0, 0.3 + 0.4 * gaba - 0.2 * ne)
    if gaba < 0.2 and ne > 0.7:  # depletion-state arousal
        return -0.4
    return 0.0


def _vigilance_reward_factory():
    """VIGILANCE: NE spike correlation. Returns a stateful reward fn that
    tracks recent VIGILANCE fires (provided by observer caller).

    rFP β § 4h: this is the canonical NE-tracker pathway. NO new anomaly
    detector needed — biology already solved this with NE itself.
    """
    state = {"recent_fires": deque(maxlen=20),  # (ts, fired) entries
             "last_ne_spike_ts": 0.0,
             "last_eval_ts": time.time()}

    def fn(s: NeuromodState, vigilance_fired_recently: bool = False,
           ne_window_s: float = 30.0) -> float:
        now = time.time()
        is_spike = s.is_spike("NE", threshold=1.5)
        if is_spike:
            state["last_ne_spike_ts"] = now

        # VIGILANCE fired AND NE spiked within window → correct detection
        if vigilance_fired_recently and is_spike:
            return min(1.0, 0.5 + 0.3 * s.z("NE"))

        # NE spike WITHOUT recent VIGILANCE fire → missed detection
        if is_spike and not vigilance_fired_recently:
            return -0.3

        # VIGILANCE fired but NE flat for 30s+ → false positive
        if vigilance_fired_recently and (now - state["last_ne_spike_ts"]) > ne_window_s:
            return -0.2

        return 0.0

    fn._state = state  # expose for inspection
    return fn


def _inspiration_reward(s: NeuromodState) -> float:
    """INSPIRATION: DA spike = reward signal (eureka biological correlate).
    Empirical Stage 0.5 finding: terminal eureka threshold is unreachable
    (rFP α scope), so we use neuromod DA spike as the PRIMARY signal.
    Phase 2c will add discrete eureka/significant event hook as secondary."""
    da_z = s.z("DA")
    if da_z > 1.5:
        return min(1.0, 0.6 + 0.2 * da_z)
    if da_z > 0.8:
        return min(1.0, 0.3 + 0.2 * da_z)
    return 0.0


def _creativity_reward(s: NeuromodState) -> float:
    """CREATIVITY: Endorphin + ACh = accomplishment + novelty attention.
    Positive when both elevated (creating-and-attending state)."""
    end = s.get("Endorphin")
    ach = s.get("ACh")
    if end > 0.5 and ach > 0.5:
        return min(1.0, 0.4 * (end + ach) - 0.2)
    return 0.0


def _curiosity_reward(s: NeuromodState) -> float:
    """CURIOSITY: DA rise + low GABA = seeking state.
    Stage 0.5 confirmed META-CGN concept_grounded events too sparse for
    primary signal — neuromod pathway carries the bulk."""
    da_z = s.z("DA")
    gaba = s.get("GABA")
    if da_z > 0.5 and gaba < 0.4:
        return min(1.0, 0.3 + 0.3 * da_z + 0.2 * (1.0 - gaba))
    if da_z < -1.0:  # DA depletion → loss of curiosity
        return -0.2
    return 0.0


def _empathy_reward(s: NeuromodState) -> float:
    """EMPATHY: 5-HT + Endorphin = post-social satisfaction.
    Positive when both elevated; negative on 5-HT depletion (social
    withdrawal signature)."""
    ht = s.get("5-HT")
    end = s.get("Endorphin")
    if ht > 0.5 and end > 0.4:
        return min(1.0, 0.4 * ht + 0.3 * end)
    if ht < 0.3:  # social-stress signature
        return -0.2
    return 0.0


def _reflection_reward(s: NeuromodState) -> float:
    """REFLECTION: low NE + high ACh = rest-with-attention (meditation).
    Stage 0.5 empirical: REFLECTION K=5, decay=0.725, window 2014s
    (matches dream-cycle scale). Combine with self_insights event hook
    in Phase 2c."""
    ne = s.get("NE")
    ach = s.get("ACh")
    if ne < 0.4 and ach > 0.5:
        return min(1.0, 0.3 + 0.3 * (1.0 - ne) + 0.3 * ach)
    return 0.0


def _reflex_reward(s: NeuromodState) -> float:
    """REFLEX: NE spike for fast response. Mostly handled via REFLEX_REWARD
    firehose at agno_hooks (existing Phase 2a path); neuromod adds
    secondary signal during pure-tonic states."""
    ne_z = s.z("NE")
    if ne_z > 1.0:
        return min(1.0, 0.3 + 0.2 * ne_z)
    return 0.0


# Master mapping
PROGRAM_REWARD_FUNCS = {
    "REFLEX": _reflex_reward,
    "FOCUS": _focus_reward,
    "INTUITION": _intuition_reward,
    "IMPULSE": _impulse_reward,
    "METABOLISM": _metabolism_reward,
    # VIGILANCE: stateful — created per-instance
    "INSPIRATION": _inspiration_reward,
    "CREATIVITY": _creativity_reward,
    "CURIOSITY": _curiosity_reward,
    "EMPATHY": _empathy_reward,
    "REFLECTION": _reflection_reward,
}


class NeuromodRewardObserver:
    """rFP β Stage 2 Phase 2b — emits per-program reward from neuromod state.

    Lifecycle:
      __init__(neural_nervous_system, neuromodulator_system, tick_interval=10)
      tick() — call once per spirit_worker tick; observer decides whether
               to emit (every N ticks, or on neuromod state change)

    The observer maintains rolling EMAs of each neuromod level and
    converts level dynamics into per-program reward signals via
    PROGRAM_REWARD_FUNCS. Every tick_interval ticks, it emits to
    record_outcome(reward, program=, source="neuromod.X") for each
    program with a non-zero reward.
    """

    NEUROMOD_NAMES = ("DA", "5-HT", "NE", "ACh", "Endorphin", "GABA")

    def __init__(self, neural_nervous_system, neuromodulator_system,
                 tick_interval: int = 10, ema_alpha: float = 0.05,
                 enabled: bool = True):
        self.nns = neural_nervous_system
        self.neuromods = neuromodulator_system
        self.tick_interval = max(1, tick_interval)
        self.ema_alpha = ema_alpha
        self.enabled = enabled
        # Per-neuromod rolling EMAs (mean + variance for z-scores)
        self._ema_means = {n: 0.5 for n in self.NEUROMOD_NAMES}  # neutral baseline
        self._ema_vars = {n: 0.01 for n in self.NEUROMOD_NAMES}
        self._tick_count = 0
        self._emissions_total = 0
        self._last_emission_per_program = {}
        # VIGILANCE special-case: stateful NE-tracker
        self._vigilance_fn = _vigilance_reward_factory()
        # Track recent VIGILANCE fires (for NE-correlation)
        self._vigilance_recent_fires = deque(maxlen=20)

    def _read_neuromod_levels(self) -> dict:
        """Extract current neuromod levels from the neuromodulator_system.
        Returns dict {name: level}. Resilient to missing attributes."""
        out = {}
        if not self.neuromods:
            return out
        modulators = getattr(self.neuromods, "modulators", None)
        if not modulators:
            return out
        for name in self.NEUROMOD_NAMES:
            mod = modulators.get(name) if hasattr(modulators, "get") else None
            if mod is None:
                continue
            level = getattr(mod, "level", None)
            if level is not None:
                out[name] = float(level)
        return out

    def _update_emas(self, levels: dict) -> None:
        """Update rolling EMA mean + variance for each neuromod."""
        a = self.ema_alpha
        for name, lvl in levels.items():
            prev_mean = self._ema_means.get(name, lvl)
            prev_var = self._ema_vars.get(name, 0.01)
            new_mean = prev_mean + a * (lvl - prev_mean)
            new_var = prev_var + a * ((lvl - prev_mean) ** 2 - prev_var)
            self._ema_means[name] = new_mean
            self._ema_vars[name] = max(new_var, 1e-6)

    def _build_state(self, levels: dict) -> NeuromodState:
        ema_stds = {n: math.sqrt(v) for n, v in self._ema_vars.items()}
        return NeuromodState(levels, self._ema_means, ema_stds)

    def record_vigilance_fire(self, fired: bool) -> None:
        """Called from spirit_worker after VIGILANCE program evaluation.
        Used by the NE-tracker to correlate fires with NE spikes."""
        self._vigilance_recent_fires.append((time.time(), fired))

    def tick(self) -> int:
        """Called every spirit_worker tick. Returns number of emissions
        this tick (0 if outside interval or disabled)."""
        if not self.enabled:
            return 0
        self._tick_count += 1
        # Always update EMAs (cheap, every tick)
        levels = self._read_neuromod_levels()
        if not levels:
            return 0
        self._update_emas(levels)
        # Only emit rewards every tick_interval ticks
        if self._tick_count % self.tick_interval != 0:
            return 0

        state = self._build_state(levels)
        emissions = 0

        # Standard 10 programs
        for prog_name, fn in PROGRAM_REWARD_FUNCS.items():
            try:
                reward = fn(state)
            except Exception as e:
                logger.debug("[NeuromodRewardObserver] %s reward fn error: %s",
                             prog_name, e)
                continue
            if abs(reward) < 0.01:  # below noise floor — don't emit
                continue
            try:
                self.nns.record_outcome(
                    reward=float(reward),
                    program=prog_name,
                    source=f"neuromod.{prog_name}",
                )
                emissions += 1
                self._last_emission_per_program[prog_name] = time.time()
            except Exception as e:
                logger.warning("[NeuromodRewardObserver] %s record_outcome failed: %s",
                               prog_name, e)

        # VIGILANCE — special NE-correlation logic
        try:
            now = time.time()
            recent_fired = any(
                f for ts, f in self._vigilance_recent_fires
                if now - ts < 5.0  # 5-second window
            )
            v_reward = self._vigilance_fn(state, vigilance_fired_recently=recent_fired)
            if abs(v_reward) >= 0.01:
                self.nns.record_outcome(
                    reward=float(v_reward),
                    program="VIGILANCE",
                    source="neuromod.NE_tracker",
                )
                emissions += 1
                self._last_emission_per_program["VIGILANCE"] = time.time()
        except Exception as e:
            logger.warning("[NeuromodRewardObserver] VIGILANCE NE-tracker failed: %s", e)

        self._emissions_total += emissions
        return emissions

    def get_stats(self) -> dict:
        """Snapshot for /v4/ns-health or arch_map ns-rewards."""
        return {
            "enabled": self.enabled,
            "tick_count": self._tick_count,
            "tick_interval": self.tick_interval,
            "emissions_total": self._emissions_total,
            "ema_means": {k: round(v, 4) for k, v in self._ema_means.items()},
            "ema_stds": {k: round(math.sqrt(v), 4) for k, v in self._ema_vars.items()},
            "last_emission_per_program": {
                k: round(time.time() - v, 1)
                for k, v in self._last_emission_per_program.items()
            },
        }
