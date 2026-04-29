"""
F-phase Session 2 — 30D context-vector builders for the 7 remaining CGN
consumers (language, knowledge, reasoning, coding, self_model, emotional,
dreaming). Pattern mirrors social_narrator.build_social_meta_context_30d.

Each builder returns exactly 30 floats in [0, 1], layout per rFP §16.2-§16.8.
Missing inputs default to neutral 0.5 so the vector is always well-formed.
Session 3's active chain execution reads these vectors to condition meta
recruitment + sub-mode selection per consumer.

See rFP: titan-docs/rFP_meta_service_interface.md §16
"""
from __future__ import annotations

import math
from typing import Optional


def _f(d, k, default=0.5) -> float:
    """Clamp a dict-get to [0,1] with neutral default."""
    try:
        v = d.get(k, default) if isinstance(d, dict) else default
        if v is None:
            return float(default)
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return float(default)


def _pad30(vec: list) -> list:
    """Enforce exactly 30 floats clipped to [0,1]."""
    if len(vec) < 30:
        vec = vec + [0.5] * (30 - len(vec))
    elif len(vec) > 30:
        vec = vec[:30]
    return [max(0.0, min(1.0, float(x))) for x in vec]


# ═══════════════════════════════════════════════════════════════════════
# §16.2 language
# ═══════════════════════════════════════════════════════════════════════

def build_language_meta_context_30d(
    neuromods: Optional[dict] = None,
    vocab_stats: Optional[dict] = None,
    emotion: Optional[dict] = None,
    last_word_outcome: Optional[dict] = None,
    conversation: Optional[dict] = None,
    chi: Optional[dict] = None,
    temporal: Optional[dict] = None,
    metabolic: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.2. Layout:
        [0:4]   neuromods (DA, 5HT, NE, GABA)
        [4:9]   vocab stats (total/500, productive, L-level/9, teach rate, grounding velocity)
        [9:13]  dominant_emotion one-hot (FLOW/PEACE/WONDER/CURIOSITY)
        [13:17] last word outcome (grounded, teach_time_norm, user_positive, sim_to_vocab)
        [17:21] conversation (partner_trust, confusion_rate, user_vocab_size_norm, topic_familiarity)
        [21:24] pi_rate, chi_coherence, i_confidence
        [24:27] temporal (epochs_since_teach_norm, session_len_norm, dream_cycles_norm)
        [27:30] metabolic (SOL_tier, chi_total, dream_pressure)
    """
    nm = neuromods or {}
    vs = vocab_stats or {}
    em = emotion or {}
    lo = last_word_outcome or {}
    cv = conversation or {}
    ch = chi or {}
    tm = temporal or {}
    mb = metabolic or {}

    vec = [_f(nm, "DA"), _f(nm, "5HT"), _f(nm, "NE"), _f(nm, "GABA")]
    vocab_total = float(vs.get("vocab_size", 0) or 0)
    productive = float(vs.get("productive", 0) or 0)
    vec.append(min(1.0, vocab_total / 500.0))
    vec.append(min(1.0, productive / max(1.0, vocab_total)) if vocab_total > 0 else 0.0)
    vec.append(_f(vs, "level_norm", 0.5))
    vec.append(_f(vs, "teaching_recent_rate", 0.3))
    vec.append(_f(vs, "grounding_velocity", 0.3))
    # one-hot for top-4 emotions
    top = (em or {}).get("top", "") if isinstance(em, dict) else ""
    for label in ("FLOW", "PEACE", "WONDER", "CURIOSITY"):
        vec.append(1.0 if top == label else 0.0)
    vec.append(1.0 if bool(lo.get("grounded", False)) else 0.0)
    vec.append(math.tanh(float(lo.get("teaching_time_s", 0.0) or 0.0) / 60.0))
    vec.append(_f(lo, "user_reaction_positive", 0.5))
    vec.append(_f(lo, "concept_sim_to_vocab", 0.5))
    vec.append(_f(cv, "partner_trust", 0.5))
    vec.append(_f(cv, "recent_confusion_rate", 0.2))
    vec.append(_f(cv, "user_vocab_norm", 0.5))
    vec.append(_f(cv, "topic_familiarity", 0.5))
    vec.append(_f(ch, "pi_rate", 0.3))
    vec.append(_f(ch, "coherence", 0.5))
    vec.append(_f(vs, "i_confidence", 0.5))
    vec.append(_f(tm, "epochs_since_teach_norm", 0.3))
    vec.append(_f(tm, "session_len_norm", 0.5))
    vec.append(_f(tm, "dream_cycles_since_word_norm", 0.3))
    vec.append(_f(mb, "SOL_tier", 0.5))
    vec.append(_f(ch, "total", 0.5))
    vec.append(_f(mb, "dream_pressure", 0.3))
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.3 knowledge
# ═══════════════════════════════════════════════════════════════════════

def build_knowledge_meta_context_30d(
    topic: str = "",
    confidence: Optional[dict] = None,
    pipeline: Optional[dict] = None,
    urgency: float = 0.5,
    memory: Optional[dict] = None,
    neuromods: Optional[dict] = None,
    meta_state: Optional[dict] = None,
    bandwidth: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.3."""
    cf = confidence or {}
    pl = pipeline or {}
    mm = memory or {}
    nm = neuromods or {}
    ms = meta_state or {}
    bw = bandwidth or {}

    t = topic or ""
    vec = [
        min(1.0, len(t) / 80.0),
        1.0 if any(name in t for name in ("FORMULATE", "RECALL",
                                           "HYPOTHESIZE", "META", "CGN",
                                           "SPIRIT")) else 0.0,
        1.0 if "?" in t else 0.0,
    ]
    vec.append(_f(cf, "internal", 0.3))
    vec.append(_f(cf, "external", 0.3))
    vec.append(_f(cf, "last_research_age_norm", 0.5))
    vec.append(_f(cf, "total_research_count_norm", 0.3))
    vec.append(_f(pl, "searxng_up", 1.0))
    vec.append(_f(pl, "duckduckgo_up", 1.0))
    vec.append(_f(pl, "wiki_up", 1.0))
    vec.append(_f(pl, "budget_remaining_norm", 0.5))
    vec.append(max(0.0, min(1.0, float(urgency))))
    vec.append(_f(pl, "time_of_day_norm", 0.5))
    vec.append(_f(pl, "epochs_since_last_success_norm", 0.5))
    vec.append(_f(mm, "cache_hit_count_norm", 0.3))
    vec.append(_f(mm, "previous_attempts_norm", 0.3))
    vec.append(_f(mm, "last_outcome_score", 0.5))
    vec.append(_f(mm, "novelty_estimate", 0.5))
    vec.append(_f(nm, "DA", 0.5))
    vec.append(_f(nm, "5HT", 0.5))
    vec.append(_f(nm, "NE", 0.5))
    vec.append(_f(ms, "chain_commit_rate", 0.5))
    vec.append(1.0 if ms.get("impasse_active") else 0.0)
    vec.append(_f(ms, "confidence_trend", 0.5))
    vec.append(0.5)  # meta reserved
    vec.append(_f(bw, "bytes_today_ratio", 0.3))
    vec.append(_f(bw, "cost_estimate_ratio", 0.3))
    vec.append(_f(bw, "retry_count_norm", 0.2))
    vec.append(_f(bw, "error_rate_1h", 0.1))
    vec.append(_f(bw, "queue_depth_norm", 0.2))
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.4 reasoning (ARC)
# ═══════════════════════════════════════════════════════════════════════

def build_reasoning_meta_context_30d(
    policy_input: Optional[list] = None,
    chain_state: Optional[dict] = None,
    recent_primitives: Optional[list] = None,
    strategy_stats: Optional[dict] = None,
    confidence_state: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.4. Leverages existing ARC policy_input."""
    cs = chain_state or {}
    ss = strategy_stats or {}
    co = confidence_state or {}

    vec = []
    # [0:18] first 18 dims of policy_input (body+mind+spirit state)
    pi = list(policy_input) if policy_input else []
    for i in range(18):
        if i < len(pi):
            try:
                v = float(pi[i])
            except (TypeError, ValueError):
                v = 0.5
        else:
            v = 0.5
        vec.append(max(0.0, min(1.0, v)))

    # [18:20] chain length + max_chain_length_norm
    vec.append(min(1.0, float(cs.get("chain_length", 0) or 0) / 20.0))
    vec.append(min(1.0, float(cs.get("max_chain_length", 10) or 10) / 20.0))

    # [20:24] last 4 primitives as rank indices normalized 0-1
    prim_names = ["FORMULATE", "RECALL", "HYPOTHESIZE", "EVALUATE",
                  "SYNTHESIZE", "BREAK", "SPIRIT_SELF", "INTROSPECT",
                  "DELEGATE"]
    rp = list(recent_primitives or [])[-4:]
    while len(rp) < 4:
        rp.insert(0, "")
    for p in rp:
        try:
            idx = prim_names.index(str(p).upper())
            vec.append(idx / max(1, len(prim_names) - 1))
        except ValueError:
            vec.append(0.0)

    # [24:27] strategy history
    vec.append(_f(ss, "distribution_entropy", 0.5))
    vec.append(_f(ss, "dominance_share", 0.3))
    vec.append(_f(ss, "trend_direction", 0.5))

    # [27:30] confidence state
    vec.append(_f(co, "current_confidence", 0.5))
    vec.append(_f(co, "gut_agreement", 0.5))
    vec.append(_f(co, "delta_confidence", 0.5))

    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.5 coding
# ═══════════════════════════════════════════════════════════════════════

def build_coding_meta_context_30d(
    success_stats: Optional[dict] = None,
    trigger: Optional[dict] = None,
    recent_tests: Optional[dict] = None,
    code_metrics: Optional[dict] = None,
    neuromods: Optional[dict] = None,
    domain: str = "other",
    time_constraints: Optional[dict] = None,
    external: Optional[dict] = None,
    meta_state: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.5."""
    ss = success_stats or {}
    tr = trigger or {}
    rt = recent_tests or {}
    cm = code_metrics or {}
    nm = neuromods or {}
    tc = time_constraints or {}
    ex = external or {}
    ms = meta_state or {}

    vec = [
        _f(ss, "success_rate", 0.3),
        min(1.0, float(ss.get("total_successes", 0) or 0) / 100.0),
        min(1.0, float(ss.get("exercises_this_dream", 0) or 0) / 10.0),
    ]
    trig_type = str(tr.get("type", "")).lower()
    for label in ("novelty_gap", "cooldown_expired", "test_failed", "dream_wake"):
        vec.append(1.0 if trig_type == label else 0.0)
    vec.append(_f(rt, "pass_ratio", 0.5))
    vec.append(min(1.0, float(rt.get("consecutive_pass", 0) or 0) / 10.0))
    vec.append(min(1.0, float(rt.get("consecutive_fail", 0) or 0) / 10.0))
    vec.append(_f(cm, "lines_of_code_norm", 0.3))
    vec.append(_f(cm, "cyclomatic_norm", 0.3))
    vec.append(_f(cm, "imports_count_norm", 0.3))
    vec.append(_f(cm, "test_coverage_ratio", 0.5))
    vec.append(_f(nm, "DA", 0.5))
    vec.append(_f(nm, "NE", 0.5))
    vec.append(_f(nm, "CREATIVITY", 0.5))
    d = (domain or "").lower()
    for label in ("spatial", "logical", "sequential", "other"):
        vec.append(1.0 if d == label else 0.0)
    vec.append(_f(tc, "epochs_since_last_success_norm", 0.5))
    vec.append(_f(tc, "cooldown_remaining_norm", 0.3))
    vec.append(_f(tc, "dream_timer_norm", 0.3))
    vec.append(_f(ex, "metabolic_tier", 0.5))
    vec.append(_f(ex, "msl_coherence", 0.5))
    vec.append(_f(ex, "pi_rate", 0.3))
    vec.append(1.0 if ms.get("chain_active") else 0.0)
    vec.append(1.0 if ms.get("mono_flag") else 0.0)
    vec.append(_f(ms, "recent_monoculture_share", 0.5))
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.6 self_model
# ═══════════════════════════════════════════════════════════════════════

def build_self_model_meta_context_30d(
    self_state: Optional[dict] = None,
    recent_introspection: Optional[dict] = None,
    neuromods: Optional[dict] = None,
    meta_state: Optional[dict] = None,
    language_social: Optional[dict] = None,
    dreaming: Optional[dict] = None,
    recent_outcomes: Optional[dict] = None,
    metabolic_time: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.6."""
    ss = self_state or {}
    ri = recent_introspection or {}
    nm = neuromods or {}
    ms = meta_state or {}
    ls = language_social or {}
    dr = dreaming or {}
    ro = recent_outcomes or {}
    mt = metabolic_time or {}

    vec = [
        _f(ss, "chi_coherence", 0.5),
        _f(ss, "i_confidence", 0.5),
        _f(ss, "i_depth", 0.5),
        _f(ss, "self_prediction_accuracy", 0.5),
        _f(ss, "mismatch_rate_1h", 0.1),
        _f(ri, "last_check_coherence", 0.5),
        min(1.0, float(ri.get("consecutive_healthy_checks", 0) or 0) / 10.0),
        _f(ri, "last_mismatch_severity", 0.3),
        _f(ri, "prediction_volatility", 0.3),
        _f(nm, "DA", 0.5),
        _f(nm, "5HT", 0.5),
        _f(nm, "NE", 0.5),
        _f(nm, "GABA", 0.5),
        1.0 if ms.get("chain_active") else 0.0,
        1.0 if ms.get("impasse") else 0.0,
        _f(ms, "novelty_signal", 0.3),
        _f(ls, "vocab_confidence", 0.5),
        _f(ls, "persona_quality", 0.5),
        _f(ls, "engagement", 0.3),
        _f(ls, "teaching_success", 0.5),
        _f(dr, "epochs_since_dream_norm", 0.5),
        _f(dr, "distilled_count_norm", 0.3),
        _f(dr, "pre_dream_undistilled_norm", 0.3),
        _f(ro, "reward_mean_1h", 0.5),
        _f(ro, "reward_variance_1h", 0.3),
        _f(ro, "surprise_rate", 0.3),
        _f(ro, "consolidation_pressure", 0.3),
        _f(mt, "SOL_tier", 0.5),
        _f(mt, "epoch_mod_24h", 0.5),
        _f(mt, "day_cycle_phase", 0.5),
    ]
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.7 emotional (EMOT-CGN)
# ═══════════════════════════════════════════════════════════════════════

EMOT_ANCHORS = ("FLOW", "PEACE", "CURIOSITY", "GRIEF", "WONDER",
                "IMPASSE_TENSION", "RESOLUTION", "LOVE")


def build_emotional_meta_context_30d(
    anchors: Optional[dict] = None,     # {name: {V: float}}
    neuromods: Optional[dict] = None,
    cluster_stats: Optional[dict] = None,
    meta_state: Optional[dict] = None,
    body: Optional[dict] = None,
    temporal: Optional[dict] = None,
    external: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.7. 8-anchor V layout + neuromods + cluster."""
    an = anchors or {}
    nm = neuromods or {}
    cs = cluster_stats or {}
    ms = meta_state or {}
    bd = body or {}
    tm = temporal or {}
    ex = external or {}

    vec = []
    for name in EMOT_ANCHORS:
        blk = an.get(name, {}) if isinstance(an, dict) else {}
        v = blk.get("V", 0.3) if isinstance(blk, dict) else 0.3
        try:
            vec.append(max(0.0, min(1.0, float(v))))
        except (TypeError, ValueError):
            vec.append(0.3)
    vec.extend([
        _f(nm, "DA"), _f(nm, "5HT"), _f(nm, "NE"), _f(nm, "GABA"),
        _f(nm, "ACh", 0.5),
    ])
    vec.append(_f(cs, "n_samples_dominant_norm", 0.3))
    vec.append(_f(cs, "variance_1h", 0.3))
    vec.append(_f(cs, "recenter_count_norm", 0.3))
    vec.append(1.0 if cs.get("rollback_fired") else 0.0)
    vec.append(1.0 if ms.get("chain_active") else 0.0)
    vec.append(_f(ms, "recent_emotion_ref_count_norm", 0.3))
    vec.append(_f(ms, "impasse_share", 0.3))
    vec.append(_f(bd, "heart_rate_norm", 0.5))
    vec.append(_f(bd, "respiration_norm", 0.5))
    vec.append(_f(bd, "chi_circulation", 0.5))
    vec.append(_f(tm, "epochs_since_last_emergence_norm", 0.5))
    vec.append(_f(tm, "dream_cycles_since_anchor_norm", 0.3))
    vec.append(_f(tm, "sleep_drive", 0.3))
    vec.append(_f(ex, "persona_quality", 0.5))
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# §16.8 dreaming
# ═══════════════════════════════════════════════════════════════════════

def build_dreaming_meta_context_30d(
    dream_state: Optional[dict] = None,
    distill_stats: Optional[dict] = None,
    neuromods: Optional[dict] = None,
    backlog: Optional[dict] = None,
    circadian: Optional[dict] = None,
    recent_cycles: Optional[dict] = None,
    experiences: Optional[dict] = None,
    chi_coherence: Optional[dict] = None,
) -> list:
    """30D vector per rFP §16.8."""
    ds = dream_state or {}
    dl = distill_stats or {}
    nm = neuromods or {}
    bl = backlog or {}
    cr = circadian or {}
    rc = recent_cycles or {}
    ex = experiences or {}
    ch = chi_coherence or {}

    vec = [
        1.0 if ds.get("is_dreaming") else 0.0,
        _f(ds, "fatigue", 0.3),
        _f(ds, "distill_complete_frac", 0.5),
        _f(ds, "epochs_in_dream_norm", 0.3),
        _f(dl, "distilled_count_1d_norm", 0.3),
        _f(dl, "distill_attempts_1d_norm", 0.3),
        _f(dl, "distill_pass_rate", 0.5),
        _f(dl, "variance_samples_count_norm", 0.3),
        _f(nm, "GABA", 0.6),
        _f(nm, "DA", 0.4),
        _f(nm, "NE", 0.4),
        _f(bl, "total_undistilled_norm", 0.3),
        _f(bl, "pre_dream_undistilled_norm", 0.3),
        _f(bl, "oldest_age_epochs_norm", 0.3),
        _f(bl, "domain_diversity", 0.5),
        _f(cr, "epoch_of_day_norm", 0.5),
        _f(cr, "day_since_birth_norm", 0.3),
        _f(cr, "pi_rate", 0.3),
        _f(rc, "wake_efficiency", 0.5),
        _f(rc, "post_wake_creativity", 0.5),
        _f(rc, "memory_consolidation_success", 0.5),
        _f(rc, "next_cycle_variance_reduction", 0.5),
        _f(ex, "episodic_count_norm", 0.3),
        _f(ex, "declarative_count_norm", 0.3),
        _f(ex, "procedural_count_norm", 0.3),
        _f(ex, "total_sig_sum_norm", 0.3),
        _f(ch, "chi_total", 0.5),
        _f(ch, "chi_circulation", 0.5),
        _f(ch, "msl_coherence", 0.5),
        _f(ch, "pi_cluster_count_norm", 0.3),
    ]
    return _pad30(vec)


# ═══════════════════════════════════════════════════════════════════════
# Signed outcome computers (rFP §16.2-§16.8)
# ═══════════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def compute_outcome_language(teaching_result: dict) -> float:
    """§16.2 formula."""
    if not isinstance(teaching_result, dict):
        return 0.0
    if teaching_result.get("concept_regressed"):
        return -0.7
    if teaching_result.get("concept_grounded"):
        strength = max(0.0, min(1.0, float(
            teaching_result.get("grounding_strength", 0.3) or 0.3)))
        t = float(teaching_result.get("time_to_ground_s", 30) or 30)
        time_bonus = max(0.0, 1.0 - t / 60.0)
        return _clamp(0.7 * strength + 0.3 * time_bonus)
    return 0.0


def compute_outcome_knowledge(research_result: dict) -> float:
    """§16.3 formula."""
    if not isinstance(research_result, dict):
        return 0.0
    if research_result.get("contradicts_prior"):
        return -0.6
    gain = (float(research_result.get("post_confidence", 0) or 0)
            - float(research_result.get("pre_confidence", 0) or 0))
    bw_mb = float(research_result.get("bytes_used", 0) or 0) / 1e6
    if gain > 0.15:
        penalty = min(0.3, bw_mb / 50.0)
        return _clamp(1.0 - penalty)
    if gain < 0.02:
        return -0.2
    return 0.3


def compute_outcome_reasoning(chain_outcome: dict) -> float:
    """§16.4 formula — [0,1] → [-1,+1]."""
    if not isinstance(chain_outcome, dict):
        return 0.0
    tsuc = max(0.0, min(1.0, float(chain_outcome.get("task_success", 0) or 0)))
    cfn = max(0.0, min(1.0, float(chain_outcome.get("final_confidence", 0) or 0)))
    steps = float(chain_outcome.get("steps", 0) or 0)
    eff = 1.0 - min(1.0, steps / 15.0)
    score = 0.6 * tsuc + 0.3 * cfn + 0.1 * eff
    return _clamp(2.0 * score - 1.0)


def compute_outcome_coding(exercise_result: dict) -> float:
    """§16.5 formula."""
    if not isinstance(exercise_result, dict):
        return 0.0
    if exercise_result.get("regressed_from_baseline"):
        return -0.8
    total = max(1, int(exercise_result.get("tests_total", 1) or 1))
    passed = int(exercise_result.get("tests_passed", 0) or 0)
    pass_rate = passed / total
    nov = max(0.0, min(1.0, float(exercise_result.get("novelty_score", 0.3) or 0.3)))
    score = 0.7 * pass_rate + 0.3 * nov
    return _clamp(2.0 * score - 1.0)


def compute_outcome_self_model(pre: dict, post: dict) -> float:
    """§16.6 formula."""
    if not isinstance(pre, dict) or not isinstance(post, dict):
        return 0.0
    coh_d = (float(post.get("chi_coherence", 0.5) or 0.5)
             - float(pre.get("chi_coherence", 0.5) or 0.5))
    prd_d = (float(post.get("self_prediction_accuracy", 0.5) or 0.5)
             - float(pre.get("self_prediction_accuracy", 0.5) or 0.5))
    combined = 0.6 * coh_d + 0.4 * prd_d
    return _clamp(math.tanh(combined * 10.0))


def compute_outcome_emotional(pre: dict, post: dict, anchor: str = "FLOW") -> float:
    """§16.7 formula. anchor: primary anchor name at outcome time."""
    if not isinstance(pre, dict) or not isinstance(post, dict):
        return 0.0
    a_pre = (pre.get("anchors", {}) or {}).get(anchor, {})
    a_post = (post.get("anchors", {}) or {}).get(anchor, {})
    try:
        v_d = float(a_post.get("V", 0.3)) - float(a_pre.get("V", 0.3))
        var_d = float(a_pre.get("variance", 0.3)) - float(a_post.get("variance", 0.3))
    except (TypeError, ValueError):
        return 0.0
    if v_d > 0.02 and var_d > 0:
        return _clamp((v_d * 5) + (var_d * 5))
    if v_d < -0.02:
        return _clamp(-0.5 + v_d * 10)
    return 0.0


def compute_outcome_dreaming(pre_sleep: dict, post_wake: dict) -> float:
    """§16.8 formula."""
    if not isinstance(pre_sleep, dict) or not isinstance(post_wake, dict):
        return 0.0
    pr_d = (float(post_wake.get("distill_pass_rate", 0.5) or 0.5)
            - float(pre_sleep.get("distill_pass_rate", 0.5) or 0.5))
    var_d = (float(pre_sleep.get("variance_samples_count", 0) or 0)
             - float(post_wake.get("variance_samples_count", 0) or 0))
    cr_d = (float(post_wake.get("creativity_signal", 0.5) or 0.5)
            - float(pre_sleep.get("creativity_signal", 0.5) or 0.5))
    # Normalize variance component to roughly same scale
    var_d_norm = math.tanh(var_d / 100.0)
    score = 0.4 * pr_d + 0.3 * var_d_norm + 0.3 * cr_d
    return _clamp(math.tanh(score * 5.0))


# ═══════════════════════════════════════════════════════════════════════
# rFP_titan_meta_outer_layer — outer-summary signal helper
# ═══════════════════════════════════════════════════════════════════════
# Per rFP §9 Implementation Checklist — adds meta's situational awareness
# of its own outer-wiredness. Returns a 3-float summary that can be:
#   (a) supplied separately in META_REASON_REQUEST payload (no dim change
#       to the 30D context_vector — backward compatible with trained policies)
#   (b) spliced into a future 33D vector after coordinated policy retrain
#       (tracked in DEFERRED_ITEMS.md as rFP_meta_outer_33d_policy_retrain)
#
# Three fields (all in [0,1]):
#   [0] outer_fetched      — 0.0 if no fetch issued, 1.0 if fetch completed
#   [1] sources_completed  — fraction of sources that returned before budget
#   [2] fetch_ms_norm      — fetch latency normalized by budget (0=fast, 1=maxed)
#
# When is_active=False OR no outer_context was fetched: returns [0.0, 0.0, 0.0]
# (zero-padding semantics — preserves neutral meaning for inactive state).

def build_outer_summary_for_context_vec(outer_context: dict = None,
                                         budget_ms: float = 200.0) -> list:
    """3-float outer-context summary for meta-reasoning consumers.

    Safe additive signal — callers choose whether/how to integrate into
    their context payload. Returns zeros when no outer data was fetched.

    Args:
      outer_context: dict from OuterContextReader.compose_recall_query(),
                     or None if no fetch occurred for this request
      budget_ms:    fetch budget used (for latency normalization)

    Returns a list of 3 floats in [0,1].
    """
    if not outer_context or not isinstance(outer_context, dict):
        return [0.0, 0.0, 0.0]
    # [0] outer_fetched — binary
    outer_fetched = 1.0
    # [1] sources_completed — 1.0 means all queried returned; lower = partial
    queried = outer_context.get("sources_queried") or []
    timed_out = outer_context.get("sources_timed_out") or []
    failed = outer_context.get("sources_failed") or []
    n_q = len(queried)
    if n_q <= 0:
        sources_completed = 0.0
    else:
        completed = n_q - len(timed_out) - len(failed)
        sources_completed = max(0.0, min(1.0, completed / n_q))
    # [2] fetch_ms_norm
    fetch_ms = float(outer_context.get("fetch_ms", 0.0) or 0.0)
    budget = float(budget_ms) if budget_ms > 0 else 200.0
    fetch_ms_norm = max(0.0, min(1.0, fetch_ms / budget))
    return [round(outer_fetched, 4),
            round(sources_completed, 4),
            round(fetch_ms_norm, 4)]
