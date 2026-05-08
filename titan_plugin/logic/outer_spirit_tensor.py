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
    # SAT — Material Being (0-14). SPEC §23.9 names.
    "world_recognition", "expressive_authenticity", "action_sovereignty",
    "boundary_enforcement", "operational_persistence", "origin_anchoring",
    "observable_growth", "world_footprint", "behavioral_consistency",
    "action_purity", "recovery_speed", "environmental_adaptation",
    "distinctive_voice", "transactional_integrity", "operational_vitality",
    # CHIT — Material Awareness (15-29). SPEC §23.9 names.
    # Indices 25 + 26 renamed 2026-05-07 from `idle_awareness` /
    # `temporal_context` (pre-redesign 2025 labels) to `dream_recall` /
    # `circadian_alignment` to match the SPEC §23.9 lock 2026-05-06 +
    # the formula docstring above ("CHIT-25 (dream_recall) — REWIRED
    # 2026-03-20"). Surfaced by Phase 0 dim-live tooling 2026-05-07.
    "world_model_depth", "signal_clarity", "threat_discernment",
    "cross_domain_integration", "witness_stability", "situation_recognition",
    "knowledge_growth", "information_quality", "engagement_depth",
    "outcome_reflection", "dream_recall", "circadian_alignment",
    "network_awareness", "causal_attribution", "self_trajectory",
    # ANANDA — Material Fulfillment (30-44). SPEC §23.9 names.
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
    # rFP_trinity_130d_awakening §12 / SPEC §23.9 — rich producer kwargs
    anchor_state: Optional[dict] = None,
    bus_stats: Optional[dict] = None,
    cgn_stats: Optional[dict] = None,
    meta_cgn_stats: Optional[dict] = None,
    language_stats: Optional[dict] = None,
    memory_growth_metrics: Optional[dict] = None,
    knowledge_graph_stats: Optional[dict] = None,
    inner_memory_stats: Optional[dict] = None,
    jailbreak_alerts_stats: Optional[dict] = None,
    output_verifier_stats: Optional[dict] = None,
    world_footprint_inputs: Optional[dict] = None,
    deltas_24h: Optional[dict] = None,
    llm_calls_this_hour: int = 0,
) -> list:
    """
    Collect 45D Outer Spirit tensor: SAT(15) + CHIT(15) + ANANDA(15).

    Every dimension follows the Observer Principle:
    Outer Spirit observes the WHOLE Outer Trinity (itself + outer_mind + outer_body).

    Phase 1 (SPEC §23.9) — many dims redesigned to read rich producers
    (jailbreak_alerts, anchor_state, meta_cgn, KG, language, memory growth, etc.)
    instead of ghost agency keys.
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
    anc = anchor_state or {}
    bus_s = bus_stats or {}
    cgn_s = cgn_stats or {}
    mcgn = meta_cgn_stats or {}
    lang = language_stats or {}
    mem_growth = memory_growth_metrics or {}
    kg = knowledge_graph_stats or {}
    imem = inner_memory_stats or {}
    jb = jailbreak_alerts_stats or {}
    ov = output_verifier_stats or {}
    wf = world_footprint_inputs or {}
    dlt = deltas_24h or {}

    outer_body_coh = _mean(outer_body)
    outer_mind_coh = _mean(outer_mind)
    combined_coh = (outer_body_coh + outer_mind_coh) / 2.0

    sat = _collect_sat(current_5d, outer_body, outer_mind, acts, crea, guard,
                       sovereignty_ratio, uptime_ratio, recov, sol, assess,
                       hist, outer_body_coh, outer_mind_coh,
                       anc, jb, ov, wf, mcgn)
    chit = _collect_chit(current_5d, outer_body, outer_mind, acts, guard,
                         soc, mem, assess, hist,
                         outer_body_coh, outer_mind_coh, combined_coh,
                         jb, ov, mcgn, cgn_s, lang, mem_growth, kg, imem, dlt)
    ananda = _collect_ananda(current_5d, outer_body, outer_mind, acts, crea,
                             guard, soc, mem, hlvl, assess, hist,
                             outer_body_coh, outer_mind_coh, combined_coh,
                             bus_s, mcgn, cgn_s, lang, mem_growth, llm_calls_this_hour, dlt)

    return sat + chit + ananda


def _collect_sat(spirit, body, mind, acts, crea, guard, sovereignty,
                 uptime, recov, sol, assess, hist, body_coh, mind_coh,
                 anc, jb, ov, wf, mcgn) -> list:
    """SAT — Material Being (15D): Does Titan EXIST in the world?

    REDESIGNED 2026-05-06 (rFP_trinity_130d_awakening §12 / SPEC §23.9):
    [0,5]   wired to solana_local_stats (local-file checks, no RPC)
    [1]     wired to action_history hormone-coherence (uses trinity_before)
    [3]     boundary_enforcement reads jailbreak_alerts + output_verifier
    [7]     world_footprint REDESIGNED — weighted log-sum across all artifacts
    [10]    recovery_speed reads anchor_state.consecutive_failures
    [13]    transactional_integrity reads anchor_state directly
    """
    import math as _math

    sat = [_DEFAULT] * 15

    # [0] World recognition: identity-verified flag from solana_local_stats.
    # SPEC §23.9 SAT[0] — local check (soul keypair loaded AND not limbo).
    sat[0] = _clamp(sol.get("identity_verified", 0.5))

    # [1] Expressive authenticity — REDESIGNED. Ratio of recent actions
    # where the posture's target hormone WAS dominant in trinity_before.
    # Producer: agency._history dicts already carry trinity_before; the
    # worker computes the coherent-action ratio and passes it as
    # `acts.inner_outer_coherence`. Phase 1 falls back to 0.5 mid-point if
    # worker hasn't yet computed it (action_history not seeded).
    sat[1] = _clamp(acts.get("inner_outer_coherence", 0.5))

    # [2] Action sovereignty: ExpressionTranslator.sovereignty_ratio.
    sat[2] = _clamp(sovereignty)

    # [3] Boundary enforcement — REDESIGNED. Real producers:
    # threats = jailbreak alerts (24h) + output_verifier violations
    # blocked = jailbreak defended (score≥0.9) + verifier rejections
    threats_24h = (
        int(jb.get("threats_detected_24h", 0) or 0)
        + int(ov.get("rejected_count_24h_delta", 0) or 0)
    )
    blocked_24h = (
        int(jb.get("blocked_24h", 0) or 0)
        + int(ov.get("rejected_count_24h_delta", 0) or 0)
    )
    if threats_24h > 0:
        sat[3] = _clamp(blocked_24h / max(1, threats_24h))
    else:
        # No threats faced — formula's defined no-threats behavior.
        # NOT a producer fallback (the producer ran and reported zero threats).
        sat[3] = 0.8

    # [4] Operational persistence: uptime continuity
    sat[4] = _clamp(uptime)

    # [5] Origin anchoring: genesis NFT presence (local file).
    sat[5] = _clamp(sol.get("genesis_nft_exists", 0.5))

    # [6] Observable growth: assessment score improvement trend (linreg slope)
    sat[6] = _clamp(0.5 + assess.get("trend", 0.0))

    # [7] World footprint — REDESIGNED. Weighted log-sum across every
    # artifact stream Titan now produces (SPEC §23.3 + §23.9 SAT[7]).
    # `world_footprint_inputs` carries pre-computed counts from worker.
    if wf:
        try:
            target_log = float(wf.get("target_log", 0.0)) or 1.0
            score_sum = float(wf.get("score_sum", 0.0))
            sat[7] = _clamp(min(1.0, score_sum / target_log))
        except Exception:
            sat[7] = _clamp(min(1.0, (acts.get("total", 0) +
                                       crea.get("total", 0)) / 200.0))
    else:
        # Fallback to legacy formula if worker hasn't built world_footprint_inputs
        sat[7] = _clamp(min(1.0, (acts.get("total", 0) +
                                   crea.get("total", 0) +
                                   sol.get("tx_count", 0)) / 200.0))

    # [8] Behavioral consistency: response style variance (low = consistent)
    sat[8] = _clamp(1.0 - assess.get("score_variance", 0.5))

    # [9] Action purity: right time × right intention × success.
    purity = assess.get("mean_score", 0.5) * acts.get("success_rate", 0.5)
    sat[9] = _clamp(purity * 2.0)

    # [10] Recovery speed — REDESIGNED. Real anchor recovery signal.
    # 0 failures = fully recovered (1.0); 10+ = deep failure (0.0).
    consecutive_failures = float(anc.get("consecutive_failures", 0) or 0)
    sat[10] = _clamp(1.0 - min(1.0, consecutive_failures / 10.0))

    # [11] Environmental adaptation — SPEC §23.9 SAT[11].
    # Phase 2 producer: EnvironmentalAdaptationTracker reads assessment
    # scores during high-cpu_thermal windows (cpu_thermal > 0.6) and
    # exposes precomputed `environmental_adaptation = 1 - clamp(var*4)`
    # via outer_spirit_history_stats. Fallback to assessment-variance
    # proxy if the producer hasn't reported yet (cold-start).
    sat[11] = _clamp(hist.get(
        "environmental_adaptation",
        1.0 - assess.get("load_variance", assess.get("score_variance", 0.3)),
    ))

    # [12] Distinctive voice: creative work diversity.
    unique_types = crea.get("unique_types", 1)
    sat[12] = _clamp(min(1.0, unique_types / 5.0))

    # [13] Transactional integrity — REDESIGNED. Real anchor success rate.
    anchor_count = float(anc.get("anchor_count", 0) or 0)
    cf = float(anc.get("consecutive_failures", 0) or 0)
    if anchor_count == 0:
        sat[13] = 0.5  # No data — explicit empty-state mid-point per SPEC.
    else:
        sat[13] = _clamp(anchor_count / (anchor_count + 5.0 * cf))

    # [14] Operational vitality: throughput × uptime.
    actions_per_hour = acts.get("per_hour", 0)
    sat[14] = _clamp(min(1.0, actions_per_hour / 20.0) * uptime)

    return sat


def _collect_chit(spirit, body, mind, acts, guard, soc, mem, assess, hist,
                  body_coh, mind_coh, combined_coh,
                  jb, ov, mcgn, cgn_s, lang, mem_growth, kg, imem, dlt) -> list:
    """CHIT — Material Awareness (15D): Does Titan KNOW the world?

    REDESIGNED 2026-05-06 (rFP §12 / SPEC §23.9):
    [0,15] world_model_depth — KG nodes/edges + meta_cgn primitives + action_chains + vocab
    [2,17] threat_discernment — jailbreak.confirmed/total + verifier ratio
    [3,18] cross_domain_integration — meta_cgn knowledge_helpful diversity
    [5,20] situation_recognition — meta_cgn.usage_gini + cgn.consolidations + haov
    [6,21] knowledge_growth — memory.learning_velocity + vocab + meta_cgn deltas
    [7,22] information_quality — meta_cgn knowledge_helpful + memory.directive_alignment
    [13,28] causal_attribution — meta_cgn primitive_V_summary confidence + haov
    [14,29] self_trajectory — Phase 2 (deque history)
    """
    chit = [_DEFAULT] * 15

    # ── Helpers ────────────────────────────────────────────────────
    def _knowledge_helpful_ratio() -> float:
        helpful_by = mcgn.get("knowledge_helpful_by_source") or {}
        helpful = sum(helpful_by.values()) if isinstance(helpful_by, dict) else 0
        responses = mcgn.get("knowledge_responses_received", 0) or 0
        return _clamp(helpful / max(1, responses))

    # [0] World model depth — REDESIGNED. KG depth + meta_cgn primitives +
    # action_chains + vocab. Edges weighted slightly higher than nodes.
    kg_nodes = float(kg.get("node_count", 0) or 0)
    kg_edges = float(kg.get("edge_count", 0) or 0)
    primitives_total = max(1, mcgn.get("primitives_total", 1) or 1)
    primitives_grounded = float(mcgn.get("primitives_grounded", 0) or 0)
    action_chains = float(imem.get("action_chains", 0) or 0)
    vocab_total = float(lang.get("vocab_total", 0) or 0)
    chit[0] = _clamp(
        0.25 * min(1.0, kg_nodes / 5000.0)
        + 0.30 * min(1.0, kg_edges / 15000.0)
        + 0.20 * (primitives_grounded / primitives_total)
        + 0.15 * min(1.0, action_chains / 500.0)
        + 0.10 * min(1.0, vocab_total / 2000.0)
    )

    # [1] Signal clarity: Trinity coherence as signal-to-noise proxy.
    chit[1] = _clamp(combined_coh)

    # [2] Threat discernment — REDESIGNED. Real jailbreak signal.
    confirmed = int(jb.get("confirmed_threats_24h", 0) or 0)
    total_flags = int(jb.get("threats_detected_24h", 0) or 0)
    if total_flags > 0:
        chit[2] = _clamp(confirmed / max(1, total_flags))
    else:
        # No threats observed — defined no-threats behavior.
        chit[2] = 0.8

    # [3] Cross-domain integration — REDESIGNED. Diversity of helpful
    # knowledge sources (high = knowledge integrates across domains).
    helpful_by = mcgn.get("knowledge_helpful_by_source") or {}
    if isinstance(helpful_by, dict) and helpful_by:
        # Domain diversity: number of sources contributing helpful knowledge,
        # normalized by saturation at 6 sources.
        chit[3] = _clamp(min(1.0, len(helpful_by) / 6.0))
    else:
        chit[3] = _clamp(_knowledge_helpful_ratio())

    # [4] Witness stability: observation quality under load.
    chit[4] = _clamp(body_coh * mind_coh * 2.0)

    # [5] Situation recognition — REDESIGNED. Pattern-recall via meta_cgn
    # usage concentration + cgn consolidations + haov verified rules.
    usage_gini = float(mcgn.get("usage_gini", 0.5) or 0.5)
    cgn_consolidations = float(cgn_s.get("consolidations", 0) or 0)
    haov_verified = 0
    haov = mcgn.get("haov") or {}
    if isinstance(haov, dict):
        haov_verified = int(haov.get("verified_rules", 0) or 0)
    chit[5] = _clamp(
        0.5 * (1.0 - usage_gini)
        + 0.3 * min(1.0, cgn_consolidations / 50.0)
        + 0.2 * min(1.0, haov_verified / 10.0)
    )

    # [6] Knowledge growth — REDESIGNED. Multi-stream growth signal.
    learning_velocity = _clamp(mem_growth.get("learning_velocity", 0.5))
    vocab_producible_growth = float(dlt.get("vocab_producible_per_day", 0.0) or 0.0)
    primitives_grounded_delta = float(dlt.get("primitives_grounded_24h", 0.0) or 0.0)
    compositions_delta = float(dlt.get("compositions_computed_24h", 0.0) or 0.0)
    chit[6] = _clamp(
        0.35 * learning_velocity
        + 0.25 * min(1.0, vocab_producible_growth / 10.0)
        + 0.20 * min(1.0, primitives_grounded_delta / 3.0)
        + 0.20 * min(1.0, compositions_delta / 50.0)
    )

    # [7] Information quality — REDESIGNED.
    chit[7] = _clamp(
        0.5 * _knowledge_helpful_ratio()
        + 0.3 * _clamp(mem_growth.get("directive_alignment", 0.5))
        + 0.2 * _clamp(assess.get("mean_score", 0.5))
    )

    # [8] Engagement depth: conversation quality (already alive).
    chit[8] = _clamp(soc.get("mean_conversation_quality", 0.5))

    # [9] Outcome reflection: assessment trend.
    chit[9] = _clamp(0.5 + assess.get("trend", 0.0))

    # [10] Dream recall — Phase 2 (needs dreaming↔action match tracking).
    _recall_ratio = 0.0
    if hist and isinstance(hist, dict):
        _recall_ratio = hist.get("dream_recall_ratio", 0.0)
    chit[10] = _clamp(_recall_ratio * 0.6 + body_coh * 0.4)

    # [11] Temporal context — Phase 2 (circadian_alignment over action history)
    chit[11] = _clamp(hist.get("circadian_alignment", 0.5))

    # [12] Network awareness: infrastructure topology coherence.
    chit[12] = _clamp(body_coh)

    # [13] Causal attribution — REDESIGNED.
    primitive_V = mcgn.get("primitive_V_summary") or {}
    if isinstance(primitive_V, dict) and primitive_V:
        confidences = [
            float(p.get("confidence", 0.0) or 0.0)
            for p in primitive_V.values() if isinstance(p, dict)
        ]
        mean_conf = sum(confidences) / max(1, len(confidences))
    else:
        mean_conf = 0.0
    chit[13] = _clamp(
        0.6 * mean_conf
        + 0.4 * min(1.0, haov_verified / 10.0)
    )

    # [14] Self-trajectory — Phase 2 (snapshot history deque).
    chit[14] = _clamp(hist.get("outer_spirit_trajectory", 0.5))

    return chit


def _collect_ananda(spirit, body, mind, acts, crea, guard, soc, mem, hlvl,
                    assess, hist, body_coh, mind_coh, combined_coh,
                    bus_s, mcgn, cgn_s, lang, mem_growth, llm_calls_this_hour, dlt) -> list:
    """ANANDA — Material Fulfillment (15D): Is world-engagement FRUITFUL?

    REDESIGNED 2026-05-06 (rFP §12 / SPEC §23.9):
    [3,33]  system_harmony — bus.dropped/published ratio
    [5,35]  information_accuracy — meta_cgn helpful + memory + verified rules
    [7,37]  capability_growth — composition_level + meta_cgn + vocab + reflex deltas
    [9,39]  discovery_value — meta_cgn knowledge_requests + felt_experiences + cgn
    [13,43] resource_appreciation — outputs_per_hour / llm_calls_this_hour
    """
    ananda = [_DEFAULT] * 15

    def _knowledge_helpful_ratio() -> float:
        helpful_by = mcgn.get("knowledge_helpful_by_source") or {}
        helpful = sum(helpful_by.values()) if isinstance(helpful_by, dict) else 0
        responses = mcgn.get("knowledge_responses_received", 0) or 0
        return _clamp(helpful / max(1, responses))

    # [0] Purpose effectiveness: action success rate.
    ananda[0] = _clamp(acts.get("success_rate", 0.5))

    # [1] Interaction depth: conversation quality.
    ananda[1] = _clamp(soc.get("mean_conversation_quality", 0.5))

    # [2] Creative impact: assessment scores on creative outputs.
    ananda[2] = _clamp(crea.get("mean_assessment", 0.5))

    # [3] System harmony — REDESIGNED. Real bus drop ratio.
    bus_published = float(bus_s.get("published", 0) or 0)
    bus_dropped = float(bus_s.get("dropped", 0) or 0)
    if bus_published > 0:
        ananda[3] = _clamp(1.0 - bus_dropped / bus_published)
    else:
        ananda[3] = 1.0  # No traffic yet — perfect harmony (no dropped messages)

    # [4] Aesthetic quality: creative assessment scores.
    ananda[4] = _clamp(crea.get("mean_assessment", 0.5))

    # [5] Information accuracy — REDESIGNED.
    haov = mcgn.get("haov") or {}
    haov_verified = int(haov.get("verified_rules", 0) or 0) if isinstance(haov, dict) else 0
    # high-confidence verified rules (proxy via verified_rules — phase 2 may
    # split into high-conf vs all-conf when haov exposes confidence breakdown)
    high_conf_rules = haov_verified  # all verified are by definition >0.5 conf
    ananda[5] = _clamp(
        0.5 * _knowledge_helpful_ratio()
        + 0.3 * min(1.0, high_conf_rules / 10.0)
        + 0.2 * _clamp(mem_growth.get("directive_alignment", 0.5))
    )

    # [6] Community connection — Phase 2 (needs new x_handles delta tracking).
    new_connections = soc.get("new_connections_per_window", 0)
    ananda[6] = _clamp(min(1.0, new_connections / 5.0))

    # [7] Capability growth — REDESIGNED. Multi-stream skill expansion.
    composition_level = float(lang.get("composition_level", "L1")[1:]
                               if isinstance(lang.get("composition_level"), str)
                                  and lang.get("composition_level", "").startswith("L")
                               else 1)
    composition_growth = float(dlt.get("composition_level_24h", 0.0) or 0.0)
    primitives_grounded_24h = float(dlt.get("primitives_grounded_24h", 0.0) or 0.0)
    vocab_producible_24h = float(dlt.get("vocab_producible_24h", 0.0) or 0.0)
    reflex_distinct_24h = float(dlt.get("reflex_distinct_fired_24h", 0.0) or 0.0)
    ananda[7] = _clamp(
        0.30 * min(1.0, max(0.0, composition_growth))
        + 0.25 * min(1.0, primitives_grounded_24h / 3.0)
        + 0.25 * min(1.0, vocab_producible_24h / 10.0)
        + 0.20 * min(1.0, reflex_distinct_24h / 5.0)
    )

    # [8] Expression reach — Phase 2 (engagement aggregation).
    ananda[8] = _clamp(soc.get("creative_engagement", 0.5))

    # [9] Discovery value — REDESIGNED.
    knowledge_requests_emitted = float(mcgn.get("knowledge_requests_emitted", 0) or 0)
    knowledge_requests_finalized = float(mcgn.get("knowledge_requests_finalized", 0) or 0)
    if knowledge_requests_emitted > 0:
        kr_finalized_ratio = knowledge_requests_finalized / knowledge_requests_emitted
    else:
        kr_finalized_ratio = 0.5  # No requests yet — neutral
    felt_to_action = float(dlt.get("felt_experiences_to_action_rate", 0.5) or 0.5)
    cgn_consolidations = float(cgn_s.get("consolidations", 0) or 0)
    cgn_per_day_norm = min(1.0, cgn_consolidations / 5.0)
    ananda[9] = _clamp(
        0.5 * _clamp(kr_finalized_ratio)
        + 0.3 * _clamp(felt_to_action)
        + 0.2 * cgn_per_day_norm
    )

    # [10] Graceful rest — Phase 2.
    ananda[10] = _clamp(hist.get("rest_performance_floor", 0.5))

    # [11] Creative tension: CREATIVITY hormone × time since last creation.
    creativity_level = hlvl.get("CREATIVITY", 0.0)
    time_since_create = hist.get("seconds_since_last_create", 300.0)
    tension = creativity_level * min(1.0, time_since_create / 600.0)
    ananda[11] = _clamp(tension)

    # [12] Surrender capacity: needs failed_retry_rate + burst_frequency
    # from agency (Phase 2 §4.2). For Phase 1, partial: body_coh proxy only
    # (acts.failed_retry_rate / burst_frequency are 0 until §4.2 ships).
    failed_retry = acts.get("failed_retry_rate", 0.0)
    resource_depletion = 1.0 - body_coh
    burst_freq = acts.get("burst_frequency", 0.0)
    surrender = 1.0 - _clamp((failed_retry + resource_depletion + burst_freq) / 3.0)
    ananda[12] = _clamp(surrender)

    # [13] Resource appreciation — REDESIGNED. Outputs per LLM call.
    # Saturation at 1.0 = 1+ output per LLM call (efficient).
    outputs_per_hour = (
        float(acts.get("per_window", 0) or 0)
        + float(crea.get("per_window", 0) or 0)
    )
    llm_per_hour = max(1.0, float(llm_calls_this_hour or 1))
    ananda[13] = _clamp(min(1.0, outputs_per_hour / llm_per_hour))

    # [14] Flow state: gated by surrender capacity.
    min_coherence = min(body_coh, mind_coh)
    error_factor = 1.0 - acts.get("error_rate", 0.1)
    assessment_factor = assess.get("mean_score", 0.5)
    surrender_gate = ananda[12]
    flow = min_coherence * error_factor * assessment_factor * surrender_gate
    ananda[14] = _clamp(flow)

    return ananda


def collect_outer_spirit_5d(
    outer_body: list,
    outer_mind: list,
    soul_health: float = 0.5,
    total_impulses: int = 0,
    total_assessed: int = 0,
    avg_score: float = 0.5,
) -> list:
    """
    Collect 5D Outer Lower Spirit tensor — meta-awareness levers.

    Pure function used by outer_spirit_worker subprocess on each tick.
    Extracted from OuterTrinityCollector._collect_outer_spirit.

    Args:
        outer_body: 5D outer body tensor (from SHM or sources)
        outer_mind: 5D outer mind tensor (from SHM or sources)
        soul_health: soul alignment score [0.0-1.0]
        total_impulses: cumulative ImpulseEngine fires
        total_assessed: cumulative SelfAssessment assessments
        avg_score: SelfAssessment rolling average score

    Returns:
        [5 floats] normalized to [0.0, 1.0].
    """
    identity_coherence = _clamp(soul_health)

    if total_impulses > 0 and total_assessed > 0:
        purpose_clarity = _clamp(avg_score)
    else:
        purpose_clarity = 0.5

    action_quality = _clamp(avg_score)

    outer_body_scalar = sum(outer_body) / len(outer_body) if outer_body else 0.5
    outer_mind_scalar = sum(outer_mind) / len(outer_mind) if outer_mind else 0.5

    return [round(v, 4) for v in [
        identity_coherence, purpose_clarity, action_quality,
        outer_body_scalar, outer_mind_scalar,
    ]]


# ── Utility Functions ───────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v == v else 0.5))  # NaN check


def _mean(tensor: list) -> float:
    """Mean of tensor values as coherence proxy."""
    if not tensor:
        return 0.5
    return sum(tensor) / len(tensor)
