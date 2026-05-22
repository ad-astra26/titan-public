"""
titan_hcl/logic/outer_mind_tensor.py — 15D Outer Mind Tensor (OT1).

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
    "social_temperature", "social_connection", "network_weather",
    "environmental_rhythm", "external_information_flow",
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
    # rFP_trinity_130d_awakening §12 / SPEC §23.8 — rich producer kwargs
    cgn_stats: Optional[dict] = None,
    meta_cgn_stats: Optional[dict] = None,
    language_stats: Optional[dict] = None,
    memory_growth_metrics: Optional[dict] = None,
    events_teacher_stats: Optional[dict] = None,
    knowledge_graph_stats: Optional[dict] = None,
    social_x_gateway_stats: Optional[dict] = None,
    uptime_seconds: float = 1.0,
) -> list:
    """
    Collect 15D Outer Mind tensor from world-facing sources.

    Args (legacy):
        current_5d: Existing 5D outer mind tensor
        action_stats: {total, success_count, success_rate, per_window}
        creative_stats: {total, art_count, audio_count, per_window}
        guardian_stats: {threats_detected, rejections, severity_avg}
        social_stats: {interactions, unique_users, sentiment_avg}
        research_stats: {queries, useful_results, usage_rate}
        assessment_stats: {mean_score, trend, count}
        body_state: Outer body values for infrastructure strain

    Args (Phase 1 rich producers — SPEC §23.1 numbered):
        cgn_stats (#7): {avg_reward, grounded_density (per_min), consolidations, ...}
        meta_cgn_stats (#8): {knowledge_helpful_by_source, knowledge_responses_received,
                              usage_gini, primitives_grounded, primitives_total,
                              eureka_accelerated_updates, knowledge_requests_emitted,
                              knowledge_requests_finalized, ...}
        language_stats (#10): {vocab_total, vocab_producible, avg_confidence,
                               teacher_sessions_last_hour, composition_level, ...}
        memory_growth_metrics (#6): {learning_velocity, directive_alignment, ...}
        events_teacher_stats (#9): {felt_experiences, windows_completed, ...}
        knowledge_graph_stats (#6): {node_count, edge_count, ...}
        social_x_gateway_stats (#12): {posts_last_hour, posts_last_day, ...}
        uptime_seconds: process uptime, used for per-hour rate normalization

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
    cgn_s = cgn_stats or {}
    mcgn = meta_cgn_stats or {}
    lang = language_stats or {}
    mem_growth = memory_growth_metrics or {}
    events = events_teacher_stats or {}
    sx = social_x_gateway_stats or {}

    # Helpers (Phase 1 — SPEC §23.1) — producer-published SPEC-named fields.
    # `knowledge_helpful_ratio` (meta_cgn.get_stats) + `avg_reward_norm`
    # (cgn.get_stats) consumed identically by Phase A+B Python L2 here and
    # Phase C Rust L1 outer-mind-rs (rFP_phase_c_130d_rust_l1_port §4.2).
    # Fallback path preserves correctness during producer-publish transition.
    def _knowledge_helpful_ratio() -> float:
        ratio = mcgn.get("knowledge_helpful_ratio")
        if ratio is not None:
            return _clamp(float(ratio))
        # Transition fallback: derive inline from underlying counters.
        helpful_by = mcgn.get("knowledge_helpful_by_source") or {}
        helpful = sum(helpful_by.values()) if isinstance(helpful_by, dict) else 0
        responses = mcgn.get("knowledge_responses_received", 0) or 0
        return _clamp(helpful / max(1, responses))

    def _cgn_avg_reward_norm() -> float:
        norm = cgn_s.get("avg_reward_norm")
        if norm is not None:
            return _clamp(float(norm))
        # Transition fallback: derive inline from raw avg_reward [-1, 1].
        r = float(cgn_s.get("avg_reward", 0.0) or 0.0)
        return _clamp((r + 1.0) / 2.0)

    # ── THINKING (5D) — practical world-knowledge ──
    thinking = [0.5] * 5

    # [0] Research effectiveness — REDESIGNED (SPEC §23.8 / rFP §12.1).
    # Old formula read res.usage_rate which no producer wrote. New: weighted
    # blend of meta-CGN knowledge helpfulness, CGN reward signal, and
    # memory directive alignment.
    thinking[0] = _clamp(
        0.4 * _knowledge_helpful_ratio()
        + 0.3 * _cgn_avg_reward_norm()
        + 0.3 * _clamp(mem_growth.get("directive_alignment", 0.5))
    )

    # [1] Knowledge retrieval — REDESIGNED. Retrieval QUALITY (not depth):
    # how good are we at finding the right thing when we look.
    vocab_avg_conf = _clamp(lang.get("avg_confidence", 0.5))
    usage_gini = _clamp(mcgn.get("usage_gini", 0.5))  # higher gini = more concentrated retrieval
    thinking[1] = _clamp(
        0.35 * _clamp(mem_growth.get("directive_alignment", 0.5))
        + 0.25 * _knowledge_helpful_ratio()
        + 0.20 * vocab_avg_conf
        + 0.20 * (1.0 - usage_gini)
    )

    # [2] Situational awareness — REDESIGNED. World-model freshness across
    # multiple awareness streams. NOTE: do NOT use `or` for default —
    # 0.0 is a valid (perfectly fresh) value, and `0.0 or 3600.0` would
    # silently demote it to "stale" via Python falsiness. Pinned by
    # test_situational_awareness_combines_event_freshness_and_velocity
    # (rFP_trinity_130d_awakening §3.7).
    _v_age = res.get("seconds_since_last")
    last_event_age = 3600.0 if _v_age is None else float(_v_age)
    _v_felt = events.get("felt_experiences")
    felt_experiences = 0.0 if _v_felt is None else float(_v_felt)
    thinking[2] = _clamp(
        0.5 * (1.0 / (1.0 + last_event_age / 1800.0))
        + 0.3 * min(1.0, felt_experiences / 20.0)
        + 0.2 * _clamp(mem_growth.get("learning_velocity", 0.5))
    )

    # [3] Problem-solving: tool execution success rate (schema bridge §3.1)
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

    # [11] Social initiative — REDESIGNED. Real X-platform outputs (posts +
    # replies in last hour) replace ghost soc.outputs_per_window.
    sx_per_hour = (
        float(sx.get("posts_last_hour", 0) or 0)
        + 0.0  # replies: gateway only tracks posts at hour granularity today
    )
    willing[1] = _clamp(min(1.0, sx_per_hour / 5.0))

    # [12] Creative output: art/audio per window (now from agency.creative_this_hour
    # via worker-side action_stats; saturation at 5/hour)
    creative_per_window = crea.get("per_window", 0)
    willing[2] = _clamp(min(1.0, creative_per_window / 5.0))

    # [13] Protective response: rejections-per-hour aggregated from real
    # producers (output_verifier, jailbreak_alerts.json, reflex boundary
    # blocks). Worker-side aggregation populates guard.rejections_per_window.
    # Saturation raised to /5.0 (was /3.0) since we now count more sources.
    rejections = guard.get("rejections_per_window", 0)
    willing[3] = _clamp(min(1.0, rejections / 5.0))

    # [14] Exploration drive — REDESIGNED. Three real exploration signals:
    # CGN grounding density (active concept exploration), language teacher
    # sessions (active learning), meta-CGN eureka acceleration (insight
    # bursts).
    grounded_density = float(cgn_s.get("grounded_density", 0.0) or 0.0)
    teacher_sessions_lh = float(lang.get("teacher_sessions_last_hour", 0) or 0)
    eureka_total = float(mcgn.get("eureka_accelerated_updates", 0) or 0)
    uptime_h = max(1.0, uptime_seconds / 3600.0)
    eureka_per_hour = eureka_total / uptime_h
    willing[4] = _clamp(
        0.40 * min(1.0, grounded_density / 2.0)
        + 0.30 * min(1.0, teacher_sessions_lh / 3.0)
        + 0.30 * min(1.0, eureka_per_hour / 5.0)
    )

    tensor_15d = thinking + feeling + willing
    # Phase 2.5.A — record firing for /v4/debug/dim-sources diagnostics.
    try:
        from titan_hcl.api.dim_registry import get_firing_tracker
        get_firing_tracker().record_block(
            "outer_mind",
            tensor_15d,
            {
                "meta_cgn_stats": meta_cgn_stats,
                "cgn_stats": cgn_stats,
                "memory_stats": memory_growth_metrics,
                "agency_stats": action_stats,
                "events_teacher_stats": events_teacher_stats,
                "social_x_gateway_stats": social_x_gateway_stats,
                "language_stats": language_stats,
                "vocab_stats": language_stats,  # vocab is part of language_stats
                "knowledge_graph_stats": knowledge_graph_stats,
                "verifier_stats": guardian_stats,
                "reflex_stats": guardian_stats,
                "jailbreak_stats": guardian_stats,
            },
        )
    except Exception:
        pass
    return tensor_15d


def collect_outer_mind_5d(
    art_count: int = 0,
    audio_count: int = 0,
    memory_status: Optional[dict] = None,
    uptime_seconds: float = 1.0,
) -> list:
    """
    Collect 5D Outer Mind tensor — creative/social levers.

    Pure function used by outer_mind_worker subprocess on each tick.
    Extracted from OuterTrinityCollector._collect_outer_mind.

    Args:
        art_count: recent art archive count (last 100)
        audio_count: recent audio archive count (last 100)
        memory_status: dict with persistent_count, total_nodes, research_nodes,
            unique_interactors
        uptime_seconds: process uptime in seconds

    Returns:
        [5 floats] normalized to [0.0, 1.0].
    """
    EXPECTED_ART_PER_DAY = 5
    EXPECTED_AUDIO_PER_DAY = 3
    EXPECTED_RESEARCH_PER_DAY = 5
    EXPECTED_INTERACTIONS_PER_DAY = 20

    mem = memory_status or {}
    uptime_days = max(0.01, max(1.0, uptime_seconds) / 86400.0)

    creative_output = _clamp(min(1.0, art_count / max(1.0, EXPECTED_ART_PER_DAY * uptime_days)))
    sonic_expression = _clamp(min(1.0, audio_count / max(1.0, EXPECTED_AUDIO_PER_DAY * uptime_days)))

    persistent = mem.get("persistent_count", 0)
    total_nodes = mem.get("total_nodes", 0)
    memory_quality = _clamp(persistent / total_nodes) if total_nodes > 0 else 0.5

    research_findings = mem.get("research_nodes", 0)
    research_depth = _clamp(min(1.0, research_findings / max(1.0, EXPECTED_RESEARCH_PER_DAY * uptime_days)))

    interactions = mem.get("unique_interactors", 0)
    social_engagement = _clamp(min(1.0, interactions / max(1.0, EXPECTED_INTERACTIONS_PER_DAY * uptime_days)))

    return [round(v, 4) for v in [
        creative_output, sonic_expression, memory_quality, research_depth, social_engagement
    ]]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, float(v) if v == v else 0.5))  # NaN check
