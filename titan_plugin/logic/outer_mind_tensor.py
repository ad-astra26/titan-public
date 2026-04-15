"""
titan_plugin/logic/outer_mind_tensor.py — 15D Outer Mind Tensor (OT1).

Expands Outer Mind from 5D to 15D following the Rosicrucian trinity:
  Thinking (5D) — What Outer Mind KNOWS from the world (practical cognition)
  Feeling  (5D) — What Outer Mind SENSES from the world (material perception)
  Willing  (5D) — What Outer Mind DOES in the world (material execution)

The Outer Mind is more MATERIAL than Inner Mind — it solves real-world
problems, executes with real tools, and senses the actual environment.
Inner Mind DRIVES (hormonal will). Outer Mind EXECUTES (practical action).

Density gradient: Outer Mind is denser than Inner Mind but lighter than
Outer Body (the densest layer in the entire architecture).
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

OUTER_MIND_DIM_NAMES = [
    # Thinking (0-4) — practical world-knowledge
    "research_effectiveness", "knowledge_retrieval", "situational_awareness",
    "problem_solving", "communication_clarity",
    # Feeling (5-9) — material world-sensing
    "social_temperature", "community_resonance", "market_awareness",
    "threat_sensing", "environmental_pressure",
    # Willing (10-14) — material world-acting
    "action_throughput", "social_initiative", "creative_output",
    "protective_response", "exploration_drive",
]


def collect_outer_mind_15d(
    current_5d: list,
    action_stats: Optional[dict] = None,
    creative_stats: Optional[dict] = None,
    guardian_stats: Optional[dict] = None,
    social_stats: Optional[dict] = None,
    research_stats: Optional[dict] = None,
    assessment_stats: Optional[dict] = None,
    body_state: Optional[dict] = None,
    twin_state: Optional[dict] = None,
    anchor_state: Optional[dict] = None,
    bus_stats: Optional[dict] = None,
) -> list:
    """
    Collect 15D Outer Mind tensor from world-facing sources.

    Args:
        current_5d: Existing 5D outer mind tensor
        action_stats: {total, success_count, success_rate, per_window}
        creative_stats: {total, art_count, audio_count, per_window}
        guardian_stats: {threats_detected, rejections, severity_avg}
        social_stats: {interactions, unique_users, sentiment_avg}
        research_stats: {queries, useful_results, usage_rate}
        assessment_stats: {mean_score, trend, count}
        body_state: Outer body values for infrastructure strain

    Returns:
        15D tensor [thinking(5) + feeling(5) + willing(5)]
    """
    acts = action_stats or {}
    crea = creative_stats or {}
    guard = guardian_stats or {}
    soc = social_stats or {}
    res = research_stats or {}
    assess = assessment_stats or {}
    body = body_state or {}

    # ── THINKING (5D) — practical world-knowledge ──
    thinking = [0.5] * 5

    # [0] Research effectiveness: did searches return useful data?
    if res.get("queries", 0) > 0:
        thinking[0] = _clamp(res.get("usage_rate", 0.5))
    else:
        thinking[0] = 0.3  # No research yet — low awareness

    # [1] Knowledge retrieval: memory recall relevance
    # Approximate from assessment quality (good assessments = good retrieval)
    thinking[1] = _clamp(assess.get("mean_score", 0.5))

    # [2] Situational awareness: world model freshness
    # Use research recency + social interaction recency
    last_research_age = res.get("seconds_since_last", 3600.0)
    thinking[2] = _clamp(1.0 / (1.0 + last_research_age / 1800.0))  # Half-life 30 min

    # [3] Problem-solving: tool execution success rate
    thinking[3] = _clamp(acts.get("success_rate", 0.5))

    # [4] Communication clarity: response assessment scores
    thinking[4] = _clamp(assess.get("mean_score", 0.5))

    # ── FEELING (5D) — material world-sensing ──
    feeling = [0.5] * 5

    # Shared social activity score (used by multiple feeling dims)
    twin = twin_state or {}
    _social_activity = _clamp(min(1.0, soc.get("interactions_per_window", 0) / 5.0) * 0.5
                              + soc.get("sentiment_avg", 0.5) * 0.5)

    # [5] Social temperature: interaction sentiment + conversation recency + activity rate
    _sentiment = soc.get("sentiment_avg", 0.5)
    _interaction_rate = min(1.0, soc.get("interactions_per_window", 0) / 8.0)
    feeling[0] = _clamp(0.5 * _sentiment + 0.3 * _interaction_rate + 0.2 * _social_activity)

    # [6] Social connection: twin resonance (bonus) + general social activity (fallback)
    # Twin enriches when reachable; general social always provides baseline.
    # Like a human: sibling nearby = strong connection; sibling away = still have friends.
    if twin.get("reachable"):
        _twin_da = twin.get("DA", 0.5)
        _twin_ne = twin.get("NE", 0.5)
        _twin_gaba = twin.get("GABA", 0.5)
        _twin_sim = 1.0 - (abs(_twin_da - 0.5) + abs(_twin_ne - 0.5) + abs(_twin_gaba - 0.5)) / 3.0
        feeling[1] = _clamp(0.6 * (0.3 + 0.5 * _twin_sim) + 0.4 * _social_activity)
    else:
        feeling[1] = _clamp(_social_activity)  # Full weight to general social

    # [7] Network weather: multi-endpoint latency variance from body entropy sensor
    # Real network conditions — always fluctuating. High variance = stormy network.
    body_vals = body.get("values", [0.5] * 5)
    feeling[2] = _clamp(1.0 - (body_vals[3] if len(body_vals) > 3 else 0.5))  # Inverted entropy = network clarity

    # [8] Environmental rhythm: blockchain activity + circadian phase + network oscillation
    # Composite world heartbeat from multiple real rhythmic sources
    _anc = anchor_state or {}
    _blockchain_active = 0.5
    if _anc.get("success") and _anc.get("last_anchor_time"):
        import time as _time
        _since = _time.time() - _anc.get("last_anchor_time", _time.time())
        _blockchain_active = max(0.1, 1.0 / (1.0 + _since / 300.0))  # Fresh = active
    _circadian = body_vals[4] if len(body_vals) > 4 else 0.5  # Thermal includes circadian
    # Network oscillation: entropy varies naturally → use as rhythm signal
    _net_oscillation = body_vals[3] if len(body_vals) > 3 else 0.5
    feeling[3] = _clamp(0.35 * _blockchain_active + 0.35 * _circadian + 0.30 * _net_oscillation)

    # [9] External information flow: bus diversity + social input + perturbation rate
    # Measures world PUSHING information toward Titan, not internal processing
    _bus = bus_stats or {}
    _bus_published = _bus.get("published", 0)
    _bus_diversity = min(1.0, _bus_published / 1000.0) if _bus_published > 0 else 0.1
    _social_input = min(1.0, soc.get("interactions_per_window", 0) / 10.0)
    # Bus message type variety as proxy for perturbation richness
    _bus_types = len(_bus.get("modules", set())) if isinstance(_bus.get("modules"), (set, list)) else 0
    _perturbation_richness = min(1.0, _bus_types / 8.0)  # 8 distinct bus modules = max richness
    feeling[4] = _clamp(0.4 * _social_input + 0.3 * _bus_diversity + 0.3 * _perturbation_richness)

    # ── WILLING (5D) — material world-acting ──
    willing = [0.5] * 5

    # [10] Action throughput: actions executed per window
    actions_per_window = acts.get("per_window", 0)
    willing[0] = _clamp(min(1.0, actions_per_window / 10.0))

    # [11] Social initiative: social outputs per window
    social_outputs = soc.get("outputs_per_window", 0)
    willing[1] = _clamp(min(1.0, social_outputs / 5.0))

    # [12] Creative output: art/audio per window
    creative_per_window = crea.get("per_window", 0)
    willing[2] = _clamp(min(1.0, creative_per_window / 5.0))

    # [13] Protective response: guardian interventions per window
    rejections = guard.get("rejections_per_window", 0)
    willing[3] = _clamp(min(1.0, rejections / 3.0))

    # [14] Exploration drive: research queries per window
    queries_per_window = res.get("queries_per_window", 0)
    willing[4] = _clamp(min(1.0, queries_per_window / 5.0))

    return thinking + feeling + willing


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, float(v) if v == v else 0.5))  # NaN check
