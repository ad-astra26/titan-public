"""
titan_plugin/logic/outer_spirit_tensor.py — 45D Outer Spirit Tensor (OT2).

Sat-Chit-Ananda consciousness architecture applied to MATERIAL engagement:
  SAT    (15D) — Material Being — Does Titan EXIST in the world?
  CHIT   (15D) — Material Awareness — Does Titan KNOW the world?
  ANANDA (15D) — Material Fulfillment — Is world-engagement FRUITFUL?

OBSERVER PRINCIPLE: Outer Spirit observes the WHOLE Outer Trinity
(itself + outer_mind + outer_body). Every dimension measures MATERIAL
engagement, not inner experience.

Density gradient: Outer Spirit is denser than Inner Spirit but lighter
than Outer Mind. It is the material witness — observing how the
world-facing self functions in concrete, measurable terms.

CHIT-25 (dream_recall) — REWIRED 2026-03-20:
  Now measures dream recall ratio (0.6 weight) blended with body coherence
  (0.4 weight). Dream recall = fraction of e_mem insights recalled during
  waking material actions. See rFP_dreaming_architecture_complete.md.
"""
import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

OUTER_SPIRIT_DIM_NAMES = [
    # SAT — Material Being (0-14)
    "world_recognition", "expressive_authenticity", "action_sovereignty",
    "boundary_enforcement", "operational_persistence", "origin_anchoring",
    "observable_growth", "world_footprint", "behavioral_consistency",
    "action_purity", "recovery_speed", "environmental_adaptation",
    "distinctive_voice", "transactional_integrity", "operational_vitality",
    # CHIT — Material Awareness (15-29)
    "world_model_depth", "signal_clarity", "threat_discernment",
    "cross_domain_integration", "witness_stability", "situation_recognition",
    "knowledge_growth", "information_quality", "engagement_depth",
    "outcome_reflection", "idle_awareness", "temporal_context",
    "network_awareness", "causal_attribution", "self_trajectory",
    # ANANDA — Material Fulfillment (30-44)
    "purpose_effectiveness", "interaction_depth", "creative_impact",
    "system_harmony", "aesthetic_quality", "information_accuracy",
    "community_connection", "capability_growth", "expression_reach",
    "discovery_value", "graceful_rest", "creative_tension",
    "surrender_capacity", "resource_appreciation", "flow_state",
]

_DEFAULT = 0.5


def collect_outer_spirit_45d(
    current_5d: list,
    outer_body: list,
    outer_mind: list,
    action_stats: Optional[dict] = None,
    creative_stats: Optional[dict] = None,
    guardian_stats: Optional[dict] = None,
    sovereignty_ratio: float = 0.0,
    uptime_ratio: float = 1.0,
    recovery_stats: Optional[dict] = None,
    social_stats: Optional[dict] = None,
    memory_stats: Optional[dict] = None,
    hormone_levels: Optional[dict] = None,
    solana_stats: Optional[dict] = None,
    assessment_stats: Optional[dict] = None,
    history: Optional[dict] = None,
) -> list:
    """
    Collect 45D Outer Spirit tensor: SAT(15) + CHIT(15) + ANANDA(15).

    Every dimension follows the Observer Principle:
    Outer Spirit observes the WHOLE Outer Trinity (itself + outer_mind + outer_body).
    """
    acts = action_stats or {}
    crea = creative_stats or {}
    guard = guardian_stats or {}
    recov = recovery_stats or {}
    soc = social_stats or {}
    mem = memory_stats or {}
    hlvl = hormone_levels or {}
    sol = solana_stats or {}
    assess = assessment_stats or {}
    hist = history or {}

    outer_body_coh = _mean(outer_body)
    outer_mind_coh = _mean(outer_mind)
    combined_coh = (outer_body_coh + outer_mind_coh) / 2.0

    sat = _collect_sat(current_5d, outer_body, outer_mind, acts, crea, guard,
                       sovereignty_ratio, uptime_ratio, recov, sol, assess,
                       hist, outer_body_coh, outer_mind_coh)
    chit = _collect_chit(current_5d, outer_body, outer_mind, acts, guard,
                         soc, mem, assess, hist,
                         outer_body_coh, outer_mind_coh, combined_coh)
    ananda = _collect_ananda(current_5d, outer_body, outer_mind, acts, crea,
                             guard, soc, mem, hlvl, assess, hist,
                             outer_body_coh, outer_mind_coh, combined_coh)

    return sat + chit + ananda


def _collect_sat(spirit, body, mind, acts, crea, guard, sovereignty,
                 uptime, recov, sol, assess, hist, body_coh, mind_coh) -> list:
    """SAT — Material Being (15D): Does Titan EXIST in the world?"""
    sat = [_DEFAULT] * 15

    # [0] World recognition: on-chain identity verifiable
    sat[0] = _clamp(sol.get("identity_verified", 0.5))

    # [1] Expressive authenticity: inner-outer action coherence
    # High when actions match hormonal drives
    sat[1] = _clamp(acts.get("inner_outer_coherence", 0.5))

    # [2] Action sovereignty: self-initiated vs externally triggered
    sat[2] = _clamp(sovereignty)

    # [3] Boundary enforcement: threat deflection success
    total_threats = guard.get("threats_detected", 0)
    blocked = guard.get("rejections", 0)
    sat[3] = _clamp(blocked / max(1, total_threats)) if total_threats > 0 else 0.8

    # [4] Operational persistence: uptime continuity
    sat[4] = _clamp(uptime)

    # [5] Origin anchoring: on-chain genesis NFT connection
    sat[5] = _clamp(sol.get("genesis_nft_exists", 0.5))

    # [6] Observable growth: assessment score improvement trend
    sat[6] = _clamp(0.5 + assess.get("trend", 0.0))  # Trend: positive=growing

    # [7] World footprint: total material outputs
    total_outputs = (acts.get("total", 0) + crea.get("total", 0) +
                     sol.get("tx_count", 0))
    sat[7] = _clamp(min(1.0, total_outputs / 200.0))

    # [8] Behavioral consistency: response style variance (low = consistent)
    sat[8] = _clamp(1.0 - assess.get("score_variance", 0.5))

    # [9] Action purity: right time × right intention × success
    # Pure action = high assessment score × high success rate
    purity = assess.get("mean_score", 0.5) * acts.get("success_rate", 0.5)
    sat[9] = _clamp(purity * 2.0)  # Scale: 0.5×0.5 = 0.5

    # [10] Recovery speed: inverse of mean recovery time
    mean_recovery_s = recov.get("mean_recovery_seconds", 60.0)
    sat[10] = _clamp(1.0 / (1.0 + mean_recovery_s / 30.0))  # 30s half-life

    # [11] Environmental adaptation: score stability under load
    sat[11] = _clamp(1.0 - assess.get("load_variance", 0.3))

    # [12] Distinctive voice: creative work diversity
    unique_types = crea.get("unique_types", 1)
    sat[12] = _clamp(min(1.0, unique_types / 5.0))

    # [13] Transactional integrity: on-chain tx success rate
    sat[13] = _clamp(sol.get("tx_success_rate", 0.5))

    # [14] Operational vitality: throughput × uptime
    actions_per_hour = acts.get("per_hour", 0)
    sat[14] = _clamp(min(1.0, actions_per_hour / 20.0) * uptime)

    return sat


def _collect_chit(spirit, body, mind, acts, guard, soc, mem, assess, hist,
                  body_coh, mind_coh, combined_coh) -> list:
    """CHIT — Material Awareness (15D): Does Titan KNOW the world?"""
    chit = [_DEFAULT] * 15

    # [0] World model depth: memory store depth × recency
    total_memories = mem.get("persistent_nodes", 0)
    chit[0] = _clamp(min(1.0, total_memories / 2000.0))

    # [1] Signal clarity: valid messages / total messages
    chit[1] = _clamp(combined_coh)  # Coherent Trinity = clear signal

    # [2] Threat discernment: guardian accuracy
    true_threats = guard.get("confirmed_threats", 0)
    total_flags = guard.get("threats_detected", 0)
    chit[2] = _clamp(true_threats / max(1, total_flags)) if total_flags > 0 else 0.8

    # [3] Cross-domain integration: multi-source success rate
    chit[3] = _clamp(acts.get("multi_source_success", 0.5))

    # [4] Witness stability: observation quality under load
    chit[4] = _clamp(body_coh * mind_coh * 2.0)

    # [5] Situation recognition: repeat-pattern detection
    chit[5] = _clamp(acts.get("pattern_reuse_rate", 0.5))

    # [6] Knowledge growth: memory node growth rate
    growth_rate = mem.get("growth_per_epoch", 0)
    chit[6] = _clamp(min(1.0, growth_rate / 10.0))

    # [7] Information quality: research result usage rate
    chit[7] = _clamp(mem.get("research_usage_rate", 0.5))

    # [8] Engagement depth: conversation quality
    chit[8] = _clamp(soc.get("mean_conversation_quality", 0.5))

    # [9] Outcome reflection: learning from outcomes (assessment trend)
    chit[9] = _clamp(0.5 + assess.get("trend", 0.0))

    # [10] Dream recall: ability to use dream-distilled insights in material actions
    # Rewired from idle_awareness to dream_recall per rFP_dreaming_architecture
    _recall_ratio = 0.0
    if hist and isinstance(hist, dict):
        _recall_ratio = hist.get("dream_recall_ratio", 0.0)
    chit[10] = _clamp(_recall_ratio * 0.6 + body_coh * 0.4)

    # [11] Temporal context: circadian cycle awareness
    # Use hour-of-day alignment with activity patterns
    chit[11] = _clamp(hist.get("circadian_alignment", 0.5))

    # [12] Network awareness: infrastructure topology understanding
    chit[12] = _clamp(body_coh)  # Outer body coherence = infra awareness

    # [13] Causal attribution: action→outcome correlation
    chit[13] = _clamp(assess.get("correlation_strength", 0.5))

    # [14] Self-trajectory: Outer Spirit observing its OWN change
    # Symmetric with Inner Spirit meta_cognition
    chit[14] = _clamp(hist.get("outer_spirit_trajectory", 0.5))

    return chit


def _collect_ananda(spirit, body, mind, acts, crea, guard, soc, mem, hlvl,
                    assess, hist, body_coh, mind_coh, combined_coh) -> list:
    """ANANDA — Material Fulfillment (15D): Is world-engagement FRUITFUL?"""
    ananda = [_DEFAULT] * 15

    # [0] Purpose effectiveness: actions achieving goals
    ananda[0] = _clamp(acts.get("success_rate", 0.5))

    # [1] Interaction depth: conversation quality
    ananda[1] = _clamp(soc.get("mean_conversation_quality", 0.5))

    # [2] Creative impact: quality of generated works
    ananda[2] = _clamp(crea.get("mean_assessment", 0.5))

    # [3] System harmony: all subsystems working together
    # Low cross-module errors = high harmony
    error_rate = acts.get("cross_module_error_rate", 0.1)
    ananda[3] = _clamp(1.0 - error_rate)

    # [4] Aesthetic quality: art/audio assessment scores
    ananda[4] = _clamp(crea.get("mean_assessment", 0.5))

    # [5] Information accuracy: research correctness
    ananda[5] = _clamp(mem.get("research_accuracy", 0.5))

    # [6] Community connection: social graph growth
    new_connections = soc.get("new_connections_per_window", 0)
    ananda[6] = _clamp(min(1.0, new_connections / 5.0))

    # [7] Capability growth: novel action types per window
    novel_actions = acts.get("novel_types_per_window", 0)
    ananda[7] = _clamp(min(1.0, novel_actions / 3.0))

    # [8] Expression reach: engagement with creative output
    ananda[8] = _clamp(soc.get("creative_engagement", 0.5))

    # [9] Discovery value: research-to-action conversion rate
    ananda[9] = _clamp(mem.get("research_to_action_rate", 0.5))

    # [10] Graceful rest: quality maintained during low-resource
    ananda[10] = _clamp(hist.get("rest_performance_floor", 0.5))

    # [11] Creative tension: building pressure before creation
    # CREATIVITY hormone level × time since last creation
    creativity_level = hlvl.get("CREATIVITY", 0.0)
    time_since_create = hist.get("seconds_since_last_create", 300.0)
    tension = creativity_level * min(1.0, time_since_create / 600.0)
    ananda[11] = _clamp(tension)

    # [12] Surrender capacity: acceptance — knowing when to stop pushing
    # LOW when: high retry rate, resource depletion, burst frequency
    # HIGH when: graceful acceptance of limits
    failed_retry = acts.get("failed_retry_rate", 0.0)
    resource_depletion = 1.0 - body_coh  # Low body health = depleted
    burst_freq = acts.get("burst_frequency", 0.0)
    surrender = 1.0 - _clamp((failed_retry + resource_depletion + burst_freq) / 3.0)
    ananda[12] = _clamp(surrender)

    # [13] Resource appreciation: efficient use of SOL, API calls, compute
    ananda[13] = _clamp(hist.get("resource_efficiency", 0.5))

    # [14] Flow state: outer_mind + outer_body simultaneous high coherence
    # DEPENDS on surrender (ANANDA-12) — must learn to let go before flow
    # Flow = min(body_coherence, mind_coherence) when both are high
    # Only achievable when surrender capacity is also high
    min_coherence = min(body_coh, mind_coh)
    error_factor = 1.0 - acts.get("error_rate", 0.1)
    assessment_factor = assess.get("mean_score", 0.5)
    surrender_gate = ananda[12]  # Must have learned surrender
    flow = min_coherence * error_factor * assessment_factor * surrender_gate
    ananda[14] = _clamp(flow)

    return ananda


# ── Utility Functions ───────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v == v else 0.5))  # NaN check


def _mean(tensor: list) -> float:
    """Mean of tensor values as coherence proxy."""
    if not tensor:
        return 0.5
    return sum(tensor) / len(tensor)
