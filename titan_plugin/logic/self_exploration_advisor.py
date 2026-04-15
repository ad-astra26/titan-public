"""
titan_plugin/logic/self_exploration_advisor.py — Self-governing exploration timing.

Controls WHEN self-exploration fires, governed entirely by Titan's own state.
ZERO hardcoded human time constants — only Schumann floor + DNA base values.

Refractory = DNA_BASE × GABA / CURIOSITY / chi_circulation
All timing derived from internal architecture: neuromodulators, Chi, hormones.

DNA_BASE values are starting parameters (like newborn reflexes from genesisNFT).
Over time, these can be learned via IQL/RL based on action outcomes.

Part of the Self-Exploration Outer Interface (ACTION→REACTION→OBSERVATION).
"""
import logging
import time

logger = logging.getLogger(__name__)

# ── Schumann-Derived Floor ──────────────────────────────────────
# SCHUMANN_MIND (1.15s) × 3 = absolute minimum between explorations
# This prevents CPU meltdown — the ONLY non-DNA timing constant.
SCHUMANN_MIND = 1.15
SCHUMANN_FLOOR = SCHUMANN_MIND * 3  # 3.45s


class SelfExplorationAdvisor:
    """Self-governing exploration rate — ZERO human time constants.

    All timing derived from Titan's internal state:
    - DNA_BASE: initial refractory per action type (from genesisNFT / birth params)
    - GABA: high inhibition → longer refractory → less exploration (calm = rest)
    - CURIOSITY: high curiosity → shorter refractory → more exploration
    - Chi circulation: stagnant Chi → longer refractory (energy conservation)
    """

    # DNA defaults — fallback if titan_params.toml not loaded.
    # Production values loaded from [self_exploration] section.
    _DEFAULT_DNA = {
        "art_generate": 30.0,
        "audio_generate": 45.0,
        "web_search": 60.0,
        "social_post": 120.0,
        "memo_inscribe": 90.0,
        "code_knowledge": 60.0,
        "self_express": 20.0,
        "infra_inspect": 45.0,
        "coding_sandbox": 90.0,
    }

    def __init__(self, dna_params: dict = None, params_config: dict = None):
        # Load from titan_params.toml [self_exploration] section if available
        self._base_refractory = dict(self._DEFAULT_DNA)
        if params_config:
            for action_type in self._DEFAULT_DNA:
                key = f"dna_base_{action_type}"
                if key in params_config:
                    self._base_refractory[action_type] = float(params_config[key])
        if dna_params:
            self._base_refractory.update(dna_params)

        self._last_action_time: dict[str, float] = {}
        self._action_outcomes: dict[str, list] = {}
        self._total_explorations = 0
        self._total_blocked = 0

    def should_explore(self, action_type: str,
                       gaba_level: float,
                       curiosity_level: float,
                       chi_circulation: float) -> bool:
        """Check if enough time has passed for this action type.

        Refractory formula (all from Titan's state):
            refractory = base × GABA / CURIOSITY / chi_circulation
            clamped to SCHUMANN_FLOOR minimum

        Args:
            action_type: helper name (art_generate, web_search, etc.)
            gaba_level: current GABA neuromodulator level (0-1)
            curiosity_level: CURIOSITY hormonal level (0-2+)
            chi_circulation: Chi circulation rate (0-3)
        """
        base = self._base_refractory.get(action_type, 60.0)

        # Self-governing refractory: DNA × inhibition / drive / energy-flow
        refractory = (
            base
            * max(0.1, gaba_level)          # High GABA → longer wait (calm = rest)
            / max(0.1, curiosity_level)     # High curiosity → shorter wait
            / max(0.1, chi_circulation)     # Flowing Chi → shorter wait
        )

        # Absolute floor: Schumann-derived (prevent CPU meltdown)
        refractory = max(SCHUMANN_FLOOR, refractory)

        last = self._last_action_time.get(action_type, 0)
        elapsed = time.time() - last
        allowed = elapsed >= refractory

        if not allowed:
            self._total_blocked += 1

        return allowed

    def record_action(self, action_type: str) -> None:
        """Record that an action was executed (resets refractory timer)."""
        self._last_action_time[action_type] = time.time()
        self._total_explorations += 1

    def record_outcome(self, action_type: str, reward: float) -> None:
        """Record action outcome for future IQL/RL-based refractory adaptation."""
        self._action_outcomes.setdefault(action_type, []).append({
            "reward": reward,
            "ts": time.time(),
        })
        # Keep last 100 outcomes per action type
        if len(self._action_outcomes[action_type]) > 100:
            self._action_outcomes[action_type] = \
                self._action_outcomes[action_type][-100:]

    def get_refractory(self, action_type: str,
                       gaba_level: float,
                       curiosity_level: float,
                       chi_circulation: float) -> float:
        """Compute current refractory for an action type (for observability)."""
        base = self._base_refractory.get(action_type, 60.0)
        refractory = (
            base
            * max(0.1, gaba_level)
            / max(0.1, curiosity_level)
            / max(0.1, chi_circulation)
        )
        return max(SCHUMANN_FLOOR, refractory)

    def get_stats(self) -> dict:
        return {
            "total_explorations": self._total_explorations,
            "total_blocked": self._total_blocked,
            "last_action_times": {
                k: round(time.time() - v, 1)
                for k, v in self._last_action_time.items()
            },
            "base_refractory": dict(self._base_refractory),
        }
