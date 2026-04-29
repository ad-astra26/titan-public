"""
core/metabolism.py
Monitors SOL balance for metabolic state and feature gating.
V2.0: Real growth metrics from memory graph data.
V3.0: 6-tier starvation table with graceful degradation (M7 Mainnet).

Starvation Table (configurable via titan_params.toml [metabolism.tiers.*]):
  > 1.0 SOL  → THRIVING    — Full capabilities, rate 100%
  0.3 - 1.0  → HEALTHY     — Normal operation, rate 100%
  0.15 - 0.3 → CONSERVING  — All services at 50% rate
  0.05 - 0.15→ SURVIVAL    — Consciousness + memory only, services stopped
  < 0.05     → EMERGENCY   — Contact Maker Protocol (M9), minimal loop
  < 0.01     → HIBERNATION — Save state, write testament, stop
"""
import logging
import math
import time
from collections import deque

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore
from pathlib import Path

logger = logging.getLogger(__name__)

# Ring buffer cap for gate decisions (exposed via /v4/metabolism/gate-status).
# ~500 decisions ≈ 10 min at 1 Hz aggregate emission across all call sites.
_GATE_DECISION_RING_SIZE = 500

# ── Default Metabolic Tiers ──────────────────────────────────────────
# Overridable via [metabolism] in titan_params.toml
_DEFAULT_TIERS = {
    "THRIVING":    {"min_sol": 1.0,  "chi_factor": 1.0, "rate_factor": 1.0,  "description": "Full capabilities"},
    "HEALTHY":     {"min_sol": 0.3,  "chi_factor": 1.0, "rate_factor": 1.0,  "description": "Normal operation"},
    "CONSERVING":  {"min_sol": 0.15, "chi_factor": 0.8, "rate_factor": 0.5,  "description": "Reduced rate, all services"},
    "SURVIVAL":    {"min_sol": 0.05, "chi_factor": 0.5, "rate_factor": 0.0,  "description": "Consciousness + memory only"},
    "EMERGENCY":   {"min_sol": 0.01, "chi_factor": 0.2, "rate_factor": 0.0,  "description": "Contact Maker Protocol"},
    "HIBERNATION": {"min_sol": 0.0,  "chi_factor": 0.0, "rate_factor": 0.0,  "description": "Save state and stop"},
}

# Feature availability per tier
_DEFAULT_FEATURES = {
    "THRIVING":    {"memos": True,  "nfts": True,  "expression": True,  "research": True,  "social": True},
    "HEALTHY":     {"memos": True,  "nfts": True,  "expression": True,  "research": True,  "social": True},
    "CONSERVING":  {"memos": True,  "nfts": True,  "expression": True,  "research": True,  "social": True},
    "SURVIVAL":    {"memos": False, "nfts": False, "expression": False, "research": False, "social": False},
    "EMERGENCY":   {"memos": False, "nfts": False, "expression": False, "research": False, "social": False},
    "HIBERNATION": {"memos": False, "nfts": False, "expression": False, "research": False, "social": False},
}


def _load_metabolic_config() -> tuple[dict, dict, bool]:
    """Load tier thresholds + gate kill-switch from titan_params.toml [metabolism]."""
    tiers = {k: dict(v) for k, v in _DEFAULT_TIERS.items()}
    features = {k: dict(v) for k, v in _DEFAULT_FEATURES.items()}
    gates_enforced = False

    params_path = Path(__file__).parent.parent / "titan_params.toml"
    if params_path.exists():
        try:
            with open(params_path, "rb") as f:
                params = tomllib.load(f)
            mcfg = params.get("metabolism", {})

            # Kill-switch: [metabolism] gates_enforced = true
            gates_enforced = bool(mcfg.get("gates_enforced", False))

            # Override tier thresholds: [metabolism.tiers.THRIVING] min_sol = 1.5
            for tier_name, overrides in mcfg.get("tiers", {}).items():
                tier_name = tier_name.upper()
                if tier_name in tiers:
                    for key in ("min_sol", "chi_factor", "rate_factor"):
                        if key in overrides:
                            tiers[tier_name][key] = float(overrides[key])

            # Override feature flags: [metabolism.features.CONSERVING] social = true
            for tier_name, feat_overrides in mcfg.get("features", {}).items():
                tier_name = tier_name.upper()
                if tier_name in features:
                    for feat, val in feat_overrides.items():
                        features[tier_name][feat] = bool(val)
        except Exception as e:
            logger.warning("[Metabolism] Failed to load titan_params.toml overrides: %s", e)

    return tiers, features, gates_enforced


# Module-level tables (loaded once, reloaded by controller on config change)
METABOLIC_TIERS, TIER_FEATURES, _GATES_ENFORCED = _load_metabolic_config()


class MetabolismController:
    """
    Manages the agent's metabolic state and SOL balance.
    Regulates energy levels, enforces governance reserves, and calculates
    Divine Growth metrics from actual memory graph data.
    """

    def __init__(self, soul, network, memory=None, config: dict = None, social_graph=None):
        """
        Args:
            soul: The SovereignSoul instance managing agent identity.
            network: The HybridNetworkClient for Solana balance checks.
            memory: The TieredMemoryGraph instance for growth calculations.
            config: Optional [growth_metrics] section from config.toml.
            social_graph: Optional SocialGraph instance for real interaction data.
        """
        self.soul = soul
        self.network = network
        self.memory = memory
        self._social_graph = social_graph

        config = config or {}
        # Saturation thresholds (configurable via config.toml [growth_metrics])
        self.node_saturation_24h = config.get("node_saturation_24h", 30)
        self.engagement_saturation_24h = config.get("engagement_saturation_24h", 50)

        # Metabolic thresholds (legacy — kept for backwards compat)
        self.RESERVE_FOR_GOVERNANCE = 0.05
        self.STARVATION_THRESHOLD = 0.1
        self.OPTIMAL_THRESHOLD = 0.5

        # M7: 6-tier metabolic state
        self._current_tier = "HEALTHY"
        self._tier_since = time.time()
        self._emergency_start = 0.0  # Track how long in EMERGENCY (for M9)

        # Cached balance for banner (updated by get_current_state)
        self._last_balance: float | None = None

        # Mainnet Lifecycle Wiring rFP (2026-04-20): gate kill-switch +
        # ring buffer of recent gate decisions for observability.
        self._gates_enforced = _GATES_ENFORCED
        self._gate_decisions: deque = deque(maxlen=_GATE_DECISION_RING_SIZE)
        self._gate_decision_counts: dict[str, int] = {}  # caller → total evaluations

    @property
    def _last_balance_pct(self) -> float:
        """Map current balance to a 0-100 life force percentage."""
        bal = self._last_balance
        if bal is None:
            return -1.0
        # 2.0 SOL = 100%, linear scale, floor at 1%
        return max(1.0, min(100.0, (bal / 2.0) * 100))

    async def get_current_state(self) -> str:
        """
        Check wallet balance and return metabolic state.

        Returns legacy states for backwards compatibility.
        Use get_metabolic_tier() for the 6-tier system.
        """
        balance = await self.network.get_balance()
        self._last_balance = balance

        # Update 6-tier state
        self._update_tier(balance)

        if balance > self.OPTIMAL_THRESHOLD:
            return "HIGH_ENERGY"
        elif balance > self.STARVATION_THRESHOLD:
            return "LOW_ENERGY"
        else:
            return "STARVATION"

    def _update_tier(self, balance: float) -> str:
        """Update the 6-tier metabolic state from SOL balance.

        Thresholds are read from METABOLIC_TIERS (configurable via titan_params.toml).
        """
        prev_tier = self._current_tier

        # Walk tiers top-down by min_sol threshold
        tier_order = ["THRIVING", "HEALTHY", "CONSERVING", "SURVIVAL", "EMERGENCY", "HIBERNATION"]
        self._current_tier = "HIBERNATION"
        for tier_name in tier_order:
            threshold = METABOLIC_TIERS.get(tier_name, {}).get("min_sol", 0)
            if balance > threshold:
                self._current_tier = tier_name
                break

        if self._current_tier != prev_tier:
            self._tier_since = time.time()
            logger.warning(
                "[Metabolism] Tier change: %s → %s (SOL=%.4f)",
                prev_tier, self._current_tier, balance)

            # Track emergency duration for Contact Maker Protocol (M9)
            if self._current_tier == "EMERGENCY":
                self._emergency_start = time.time()
            elif prev_tier == "EMERGENCY":
                self._emergency_start = 0.0

        return self._current_tier

    def get_metabolic_tier(self) -> str:
        """Get the current 6-tier metabolic state."""
        return self._current_tier

    def can_use_feature(self, feature: str) -> bool:
        """Check if a feature is available at the current metabolic tier.

        Features: memos, nfts, expression, research, social
        """
        tier_features = TIER_FEATURES.get(self._current_tier, {})
        return tier_features.get(feature, False)

    def get_service_gate(self, feature: str) -> tuple[bool, float, str]:
        """Check if a service should run and at what rate.

        Returns (allowed, rate_factor, reason) — a universal gate for any service.
        - allowed: True if the feature is enabled at this tier
        - rate_factor: 1.0 = every invocation, 0.5 = half, 0.0 = none
        - reason: human-readable explanation
        """
        tier = self._current_tier
        features = TIER_FEATURES.get(tier, {})
        tier_info = METABOLIC_TIERS.get(tier, {})
        rate = tier_info.get("rate_factor", 1.0)

        if not features.get(feature, False):
            return False, 0.0, f"tier_{tier.lower()}_{feature}_disabled"

        if rate <= 0.0:
            return False, 0.0, f"tier_{tier.lower()}_rate_zero"

        return True, rate, f"tier_{tier.lower()}_rate_{rate}"

    @property
    def gates_enforced(self) -> bool:
        """Kill-switch flag for Mainnet Lifecycle Wiring. False = observability-only."""
        return self._gates_enforced

    def evaluate_gate(self, feature: str, caller: str = "") -> tuple[bool, float]:
        """Universal call-site entry point for metabolism gates.

        Decides whether a service/call should proceed at the current tier, and
        records the decision in a ring buffer for observability.

        Semantics:
          - `gates_enforced = False` (observation-only): always returns (True, 1.0).
            The underlying decision is still logged + ringed so 24h soaks can
            confirm decisions match expectations before flipping enforcement.
          - `gates_enforced = True`: returns the real decision. If the gate is
            closed, the caller MUST skip the work. `rate_multiplier < 1.0` means
            the caller should probabilistically skip (random() > rate → skip).

        Args:
            feature: One of TIER_FEATURES keys (memos, nfts, expression, research, social).
            caller: Human-readable call-site name for log + ring buffer. Optional.

        Returns:
            (should_proceed, rate_multiplier)
        """
        allowed, rate, reason = self.get_service_gate(feature)
        tier = self._current_tier
        enforced = self._gates_enforced

        decision = {
            "ts": time.time(),
            "feature": feature,
            "caller": caller or feature,
            "tier": tier,
            "allowed": allowed,
            "rate": rate,
            "reason": reason,
            "enforced": enforced,
        }
        self._gate_decisions.append(decision)
        key = caller or feature
        self._gate_decision_counts[key] = self._gate_decision_counts.get(key, 0) + 1

        if not enforced:
            if not allowed:
                logger.info(
                    "[Metabolism-Gate] %s would-close: %s (observation-only, gates_enforced=False)",
                    caller or feature, reason)
            return (True, 1.0)

        if not allowed:
            logger.info("[Metabolism-Gate] %s CLOSED: %s", caller or feature, reason)
        return (allowed, rate)

    def get_gate_decision_summary(self) -> dict:
        """Aggregate summary of gate decisions for /v4/metabolism/gate-status."""
        decisions = list(self._gate_decisions)
        total = len(decisions)
        if total == 0:
            return {
                "gates_enforced": self._gates_enforced,
                "current_tier": self._current_tier,
                "total_evaluations": 0,
                "decisions_buffered": 0,
                "by_caller": {},
                "recent_closures": [],
            }

        now = time.time()
        window_10m = [d for d in decisions if now - d["ts"] <= 600]
        closures = [d for d in window_10m if not d["allowed"]]
        by_caller: dict[str, dict] = {}
        for d in window_10m:
            key = d["caller"]
            bucket = by_caller.setdefault(key, {"total": 0, "closed": 0, "feature": d["feature"]})
            bucket["total"] += 1
            if not d["allowed"]:
                bucket["closed"] += 1

        return {
            "gates_enforced": self._gates_enforced,
            "current_tier": self._current_tier,
            "sol_balance": self._last_balance,
            "total_evaluations": sum(self._gate_decision_counts.values()),
            "decisions_buffered": total,
            "window_10min_count": len(window_10m),
            "window_10min_closures": len(closures),
            "by_caller": by_caller,
            "recent_closures": closures[-20:],
        }

    def get_emergency_duration(self) -> float:
        """How long (seconds) Titan has been in EMERGENCY tier. 0 if not in emergency."""
        if self._current_tier == "EMERGENCY" and self._emergency_start > 0:
            return time.time() - self._emergency_start
        return 0.0

    def get_tier_info(self) -> dict:
        """Full tier info for API exposure."""
        tier_info = METABOLIC_TIERS.get(self._current_tier, {})
        features = TIER_FEATURES.get(self._current_tier, {})
        return {
            "tier": self._current_tier,
            "sol_balance": self._last_balance,
            "chi_factor": tier_info.get("chi_factor", 1.0),
            "rate_factor": tier_info.get("rate_factor", 1.0),
            "description": tier_info.get("description", ""),
            "tier_since": self._tier_since,
            "emergency_duration_s": self.get_emergency_duration(),
            "features": features,
        }

    async def can_afford(self, cost: float) -> bool:
        """Check if a transaction is affordable without breaching governance reserve."""
        balance = await self.network.get_balance()
        return (balance - cost) >= self.RESERVE_FOR_GOVERNANCE

    # -------------------------------------------------------------------------
    # Divine Growth Metrics (real data from memory graph)
    # -------------------------------------------------------------------------
    async def get_learning_velocity(self) -> float:
        """
        Learning Velocity from actual memory graph data.
        Score = min(1.0, ln(effective_nodes + 1) / ln(saturation + 1))

        Uses persistent node count weighted by average effective_weight
        (neuroplasticity-adjusted) from the last 24 hours.
        """
        if not self.memory:
            return 0.5

        # Count persistent nodes and calculate quality-weighted count
        effective_nodes = 0.0
        now = time.time()
        cutoff = now - 86400  # Last 24 hours

        for v in self.memory._node_store.values():
            if v.get("type") != "MemoryNode" or v.get("status") != "persistent":
                continue
            # Only count nodes created or accessed in the last 24h as "active learning"
            if v.get("created_at", 0) >= cutoff or v.get("last_accessed", 0) >= cutoff:
                weight = v.get("effective_weight", 1.0)
                effective_nodes += weight

        # Logarithmic saturation curve
        saturation = self.node_saturation_24h
        score = min(1.0, math.log(effective_nodes + 1) / math.log(saturation + 1))
        return score

    async def get_social_density(self) -> float:
        """
        Social Density based on unique interactors, median sentiment, and Social Gravity.

        Social Gravity captures two ratio-based signals:
          - Social Pressure:  mentions_received / max(1, replies_sent)
            High ratio = the world wants the Titan's attention (increases arousal).
          - Social Reward:    reply_likes / max(1, replies_sent)
            High ratio = the Titan's replies resonate (increases valence).

        Final Score = (Volume * 0.25) + (Sentiment * 0.35) + (Gravity * 0.40)
        """
        if not self.memory:
            return 0.5

        # 1. Volume: unique interactions in last 24h
        unique_users = await self.memory.get_unique_interactors(timespan_seconds=86400)
        vol_score = min(1.0, len(unique_users) / self.engagement_saturation_24h)

        # 2. Median sentiment from recent system pulse nodes
        sentiments = await self.memory.get_recent_sentiments(count=4)
        if not sentiments:
            avg_sentiment = 0.5
        else:
            sentiments.sort()
            mid = len(sentiments) // 2
            if len(sentiments) % 2 != 0:
                avg_sentiment = sentiments[mid]
            else:
                avg_sentiment = (sentiments[mid - 1] + sentiments[mid]) / 2.0

        # 3. Social Gravity from engagement metrics
        metrics = await self.memory.fetch_social_metrics()
        mentions = metrics.get("mentions_received", 0)
        replies = metrics.get("daily_replies", 0)
        reply_likes = metrics.get("reply_likes", 0)

        # Social Pressure: capped at 1.0 (normalized by a 5:1 mention/reply ratio)
        social_pressure = min(1.0, mentions / max(1, replies * 5))
        # Social Reward: capped at 1.0 (1+ likes per reply is "resonating")
        social_reward = min(1.0, reply_likes / max(1, replies))
        # Combined gravity: pressure contributes arousal, reward contributes valence
        gravity_score = (social_pressure * 0.4) + (social_reward * 0.6)

        base_score = (vol_score * 0.25) + (avg_sentiment * 0.35) + (gravity_score * 0.40)

        # Augment with SocialGraph real interaction data (if available)
        if self._social_graph:
            try:
                sg_stats = self._social_graph.get_stats()
                sg_users = sg_stats.get("users", 0)
                sg_edges = sg_stats.get("edges", 0)
                # Normalize: 20 users + 10 edges = full social density
                sg_density = min(1.0, (sg_users / 20.0) * 0.6 + (sg_edges / 10.0) * 0.4)
                # Blend 50/50 with existing score
                return (base_score + sg_density) / 2.0
            except Exception:
                pass

        return base_score

    async def get_metabolic_health(self) -> float:
        """SOL balance stability score: 1.0 (HIGH), 0.5 (LOW), 0.1 (STARVING)."""
        state = await self.get_current_state()
        if state == "HIGH_ENERGY":
            return 1.0
        elif state == "LOW_ENERGY":
            return 0.5
        return 0.1

    async def get_directive_alignment(self) -> float:
        """
        Directive alignment score based on gatekeeper routing history.
        Higher sovereign action ratio = better alignment with prime directives.

        Uses gatekeeper stats from memory if available, otherwise estimates
        from the ratio of high-weight persistent memories.
        """
        if not self.memory:
            return 0.5

        # Calculate alignment from persistent memory quality distribution
        total = 0
        high_quality = 0
        for v in self.memory._node_store.values():
            if v.get("type") != "MemoryNode" or v.get("status") != "persistent":
                continue
            total += 1
            if v.get("effective_weight", 1.0) >= 1.15:
                high_quality += 1

        if total == 0:
            return 0.5  # No data yet — neutral

        # Ratio of high-quality memories as a proxy for directive alignment
        return min(1.0, (high_quality / total) * 1.2)
