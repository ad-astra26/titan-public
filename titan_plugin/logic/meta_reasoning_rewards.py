"""
titan_plugin/logic/meta_reasoning_rewards.py — TUNING-012 v2 Sub-phase A.

Per-primitive compound reward helpers. Replaces flat ~0.15 reward with
context-dependent rewards from unique architectural subsystems.

Each helper takes:
  - state:           MetaChainState (current chain context)
  - step_output:     dict from the primitive execution (e.g. RECALL count, FORMULATE difficulty)
  - dna:             merged DNA dict (titan_params [meta_reasoning_dna] + per-Titan override)
  - subsystem_signals: dict of cached signals from Inner Memory + TimeChain + Contracts

Each helper returns: (reward: float, breakdown: dict)
  - reward is clipped to dna's defined range (per primitive)
  - breakdown is per-component for observability logging

DESIGN PRINCIPLES (from PLAN_emergent_neuromodulator_redesign.md success pattern):
1. Each primitive's components source from a UNIQUE architectural combination.
2. All coefficients are DNA — no hardcoded magic numbers.
3. Pure functions — no side effects, no I/O. Trivially unit-testable.
4. Graceful degradation — missing signals default to 0 (the base reward still fires).

See: titan-docs/rFP_tuning_012_compound_rewards_v2.md §5-§7.A
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable


# ── Utilities ─────────────────────────────────────────────────────


def _clip(x: float, lo: float, hi: float) -> float:
    """Hard clip helper."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Coerce value to float, fall back on default for None/non-numeric."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _shannon_entropy_of_history(history: Iterable[Any]) -> float:
    """Shannon entropy of recall sources (in nats), normalized to [0, 1].

    Used by RECALL to reward source diversity. A chain that recalls from
    only one source gets entropy 0; mixing 4+ sources approaches 1.0.
    """
    items = list(history) if history is not None else []
    if not items:
        return 0.0
    counts: dict = {}
    for item in items:
        # Extract a hashable key (source name or first dict field)
        if isinstance(item, dict):
            key = item.get("source") or item.get("sub") or "_"
        else:
            key = str(item)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    # Normalize: max entropy for N unique items is log(N).
    # Cap at log(4) so 4+ unique sources gives full reward.
    max_entropy = math.log(min(len(counts), 4)) if len(counts) > 1 else 1.0
    return _clip(entropy / max_entropy, 0.0, 1.0) if max_entropy > 0 else 0.0


# ── RECALL ────────────────────────────────────────────────────────


def reward_recall(
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """RECALL compound reward.

    Sources:
      - Inner Memory: FAISS similarity + Shannon entropy of source diversity
      - Inner Memory (Kuzu): graph centrality of recalled nodes
      - TimeChain: depth of historical similar thoughts
      - Smart Contracts: contracts that previously ratified recalled thoughts
      - Retroactive: outcome correlation (chain success)

    Range: 0.0 - 0.41 (DNA-defined).
    """
    base = _safe_float(dna.get("recall_base"), 0.08)

    # Inner memory components
    inner_rel = _safe_float(subsystem_signals.get("inner_relevance"), 0.0)
    sim = inner_rel * _safe_float(dna.get("recall_inner_relevance_weight"), 0.10)

    history = getattr(state, "recall_history", None)
    entropy_score = _shannon_entropy_of_history(history)
    entropy = entropy_score * _safe_float(dna.get("recall_inner_source_entropy_weight"), 0.05)

    centrality_score = _safe_float(subsystem_signals.get("kuzu_centrality"), 0.0)
    centrality = centrality_score * _safe_float(dna.get("recall_kuzu_centrality_weight"), 0.04)

    # TimeChain: how deep are similar past thoughts?
    depth_score = _safe_float(subsystem_signals.get("timechain_depth"), 0.0)
    depth = depth_score * _safe_float(dna.get("recall_timechain_depth_weight"), 0.07)

    # Smart contracts: did contracts previously ratify these thoughts?
    ratified_score = _safe_float(subsystem_signals.get("contract_ratified"), 0.0)
    ratified = ratified_score * _safe_float(dna.get("recall_contract_ratified_weight"), 0.05)

    # Retroactive: chain success (set by chain conclusion via state)
    chain_succeeded = _safe_float(getattr(state, "chain_succeeded", 0.0), 0.0)
    outcome = chain_succeeded * _safe_float(dna.get("recall_outcome_corr_weight"), 0.10)

    total = max(0.0, base + sim + entropy + centrality + depth + ratified + outcome)
    return total, {
        "base": round(base, 4),
        "sim": round(sim, 4),
        "entropy": round(entropy, 4),
        "centrality": round(centrality, 4),
        "depth": round(depth, 4),
        "ratified": round(ratified, 4),
        "outcome": round(outcome, 4),
        "total": round(total, 4),
    }


# ── FORMULATE ─────────────────────────────────────────────────────


def reward_formulate(
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """FORMULATE compound reward.

    Sources:
      - Spirit drift: anomaly magnitude (132D state divergence)
      - Spirit drift: dimensionality (how many anomalous dims)
      - Problem template specificity
      - TimeChain novelty: penalize repetitive formulations
      - Smart contracts: pattern match against contract priorities
      - Heuristic solvability

    Range: 0.0 - 0.47.
    """
    base = _safe_float(dna.get("formulate_base"), 0.08)

    # Spirit drift magnitude (from FORMULATE.define output)
    difficulty = _safe_float(step_output.get("difficulty"), 0.0)
    anomaly = difficulty * _safe_float(dna.get("formulate_anomaly_weight"), 0.10)

    # Dimensionality: how many distinct anomalous dims?
    anom_dims = step_output.get("anomalous_dims") or []
    if isinstance(anom_dims, list):
        dim_count = len(set(anom_dims))
    else:
        dim_count = 0
    # Normalize to [0, 1]: 0 dims = 0, 5+ dims = 1
    dim_score = _clip(dim_count / 5.0, 0.0, 1.0)
    anomaly_dim = dim_score * _safe_float(dna.get("formulate_anomaly_dim_weight"), 0.06)

    # Problem template specificity: a narrower template is more useful
    template = ""
    if hasattr(state, "formulate_output") and isinstance(state.formulate_output, dict):
        template = str(state.formulate_output.get("problem_template", "") or "")
    # Simple heuristic: length-based (longer template = more specific, capped)
    spec_score = _clip(len(template) / 80.0, 0.0, 1.0) if template else 0.0
    specificity = spec_score * _safe_float(dna.get("formulate_specificity_weight"), 0.06)

    # TimeChain novelty: 1.0 = totally novel, 0.0 = identical to prior.
    # 2026-04-13 (Phase 3 of foundational healing rFP): default lowered
    # 0.5 → 0.0. The 0.5 default gave FORMULATE a +0.04 free baseline
    # boost over RECALL/EVALUATE/etc. that have no such default-positive
    # component. Over 92K training updates this asymmetry baked
    # FORMULATE's policy bias to +1.146 vs all other primitives < +0.10.
    # If timechain signal is genuinely unknown, treat as "no novelty
    # information" (0.0), not "neutral novelty" (0.5).
    novelty_score = _safe_float(subsystem_signals.get("timechain_novelty"), 0.0)
    novelty = novelty_score * _safe_float(dna.get("formulate_timechain_novelty_weight"), 0.08)

    # Smart contracts: priority pattern match
    contract_score = _safe_float(subsystem_signals.get("contract_priority"), 0.0)
    contract = contract_score * _safe_float(dna.get("formulate_contract_priority_weight"), 0.05)

    # Solvability heuristic: if difficulty is mid-range (0.3-0.7), solvable; extremes are hard
    if 0.0 < difficulty <= 1.0:
        solvability_score = 1.0 - abs(difficulty - 0.5) * 2.0  # peak at 0.5
        solvability_score = _clip(solvability_score, 0.0, 1.0)
    else:
        solvability_score = 0.0
    solvability = solvability_score * _safe_float(dna.get("formulate_solvability_weight"), 0.04)

    total = max(0.0, base + anomaly + anomaly_dim + specificity + novelty + contract + solvability)
    return total, {
        "base": round(base, 4),
        "anomaly": round(anomaly, 4),
        "anomaly_dim": round(anomaly_dim, 4),
        "specificity": round(specificity, 4),
        "novelty": round(novelty, 4),
        "contract": round(contract, 4),
        "solvability": round(solvability, 4),
        "total": round(total, 4),
    }


# ── EVALUATE ──────────────────────────────────────────────────────


def reward_evaluate(
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """EVALUATE compound reward.

    Sources:
      - Reasoning state: information gain (|conf_post - conf_pre|)
      - Reasoning state: agreement with delegate result (consistency)
      - TimeChain: prior eval consistency
      - Smart contracts: contract pass/fail check
      - Timing: midpoint sweet spot bonus

    Range: 0.0 - 0.45.
    """
    base = _safe_float(dna.get("evaluate_base"), 0.08)

    # Information gain: how much did this EVALUATE move the meta-confidence?
    pre_conf = _safe_float(getattr(state, "pre_eval_confidence", 0.0), 0.0)
    post_conf = _safe_float(getattr(state, "confidence", 0.5), 0.5)
    info_gain_score = _clip(abs(post_conf - pre_conf) * 2.0, 0.0, 1.0)
    info_gain = info_gain_score * _safe_float(dna.get("evaluate_info_gain_weight"), 0.12)

    # Consistency: agreement with delegate result
    consistency_score = 0.0
    delegate_results = getattr(state, "delegate_results", []) or []
    if delegate_results:
        last_del_conf = _safe_float(delegate_results[-1].get("confidence"), 0.5)
        # If meta confidence agrees with delegate result, high consistency
        consistency_score = 1.0 - _clip(abs(post_conf - last_del_conf), 0.0, 1.0)
    consistency = consistency_score * _safe_float(dna.get("evaluate_consistency_weight"), 0.08)

    # TimeChain: prior evals consistency (cached signal)
    tc_consistency_score = _safe_float(subsystem_signals.get("timechain_eval_consistency"), 0.0)
    tc_consistency = tc_consistency_score * _safe_float(
        dna.get("evaluate_timechain_consistency_weight"), 0.07)

    # Smart contracts: did the eval pass active filter contracts?
    compliance_score = _safe_float(subsystem_signals.get("contract_compliance"), 0.0)
    compliance = compliance_score * _safe_float(
        dna.get("evaluate_contract_compliance_weight"), 0.05)

    # Timing: midpoint of chain is optimal time to eval
    chain_len = len(getattr(state, "chain", []) or [])
    max_steps = max(1, getattr(state, "max_steps", 20) or 20)
    if chain_len > 0:
        position = chain_len / max_steps
        # Peak at 0.5 (midpoint)
        timing_score = 1.0 - abs(position - 0.5) * 2.0
        timing_score = _clip(timing_score, 0.0, 1.0)
    else:
        timing_score = 0.0
    timing = timing_score * _safe_float(dna.get("evaluate_timing_weight"), 0.05)

    total = max(0.0, base + info_gain + consistency + tc_consistency + compliance + timing)
    return total, {
        "base": round(base, 4),
        "info_gain": round(info_gain, 4),
        "consistency": round(consistency, 4),
        "tc_consistency": round(tc_consistency, 4),
        "compliance": round(compliance, 4),
        "timing": round(timing, 4),
        "total": round(total, 4),
    }


# ── INTROSPECT ────────────────────────────────────────────────────


def reward_introspect(
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """INTROSPECT compound reward.

    Sources:
      - Self-Reasoning: prediction accuracy (stub→Sub-phase E)
      - Self-Reasoning: profile divergence/deepening (stub→Sub-phase E)
      - TimeChain: self-state continuity
      - Smart contracts: identity contract alignment
      - Calibration: meta-confidence vs success

    Range: 0.0 - 0.43.

    NOTE: accuracy + deepening signals are stub until Sub-phase E wires
    SELF_PROFILE dream consolidation. They'll read 0.0 in this session.
    """
    base = _safe_float(dna.get("introspect_base"), 0.08)

    # Self-prediction accuracy (stub until Sub-phase E)
    accuracy_score = _safe_float(subsystem_signals.get("self_prediction_accuracy"), 0.0)
    accuracy = accuracy_score * _safe_float(dna.get("introspect_accuracy_weight"), 0.10)

    # Profile deepening: how much did the introspection diverge from prior profile?
    deepening_score = _safe_float(subsystem_signals.get("self_profile_divergence"), 0.0)
    deepening = deepening_score * _safe_float(dna.get("introspect_deepening_weight"), 0.08)

    # TimeChain: continuity of self-state across recent observations
    continuity_score = _safe_float(subsystem_signals.get("timechain_self_continuity"), 0.0)
    continuity = continuity_score * _safe_float(
        dna.get("introspect_timechain_continuity_weight"), 0.07)

    # Identity contracts: alignment with genesis-type identity contracts
    identity_score = _safe_float(subsystem_signals.get("contract_identity_alignment"), 0.0)
    identity = identity_score * _safe_float(
        dna.get("introspect_identity_alignment_weight"), 0.05)

    # Calibration: how well did meta-confidence track actual success?
    confidence = _safe_float(getattr(state, "confidence", 0.5), 0.5)
    chain_succeeded = _safe_float(getattr(state, "chain_succeeded", 0.5), 0.5)
    calibration_score = 1.0 - _clip(abs(confidence - chain_succeeded), 0.0, 1.0)
    calibration = calibration_score * _safe_float(
        dna.get("introspect_calibration_weight"), 0.05)

    total = max(0.0, base + accuracy + deepening + continuity + identity + calibration)
    return total, {
        "base": round(base, 4),
        "accuracy": round(accuracy, 4),
        "deepening": round(deepening, 4),
        "continuity": round(continuity, 4),
        "identity": round(identity, 4),
        "calibration": round(calibration, 4),
        "total": round(total, 4),
    }


# ── BREAK ─────────────────────────────────────────────────────────


def reward_break(
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """BREAK compound reward.

    Sources:
      - Chain archive: post-break chain success vs pre-break average
      - Eureka: did BREAK enable an EUREKA?
      - TimeChain: historical break wisdom
      - Smart contracts: contract-driven break trigger (e.g. homeostatic_alert)
      - Always-on cost penalty

    Range: -0.05 - 0.36 (net positive — was net negative pre-fix).
    """
    base = _safe_float(dna.get("break_base"), 0.05)

    # Recovery: did the chain succeed AFTER the break?
    pre_avg = _safe_float(getattr(state, "pre_break_avg_reward", 0.0), 0.0)
    chain_succeeded = _safe_float(getattr(state, "chain_succeeded", 0.0), 0.0)
    if pre_avg > 0:
        recovery_score = _clip((chain_succeeded - pre_avg) / max(0.1, pre_avg), -1.0, 1.0)
        recovery_score = max(0.0, recovery_score)
    else:
        recovery_score = chain_succeeded
    recovery = recovery_score * _safe_float(dna.get("break_recovery_weight"), 0.12)

    # Eureka: BREAK enabled a EUREKA event in the same chain
    eureka_fired = bool(getattr(state, "eureka_after_break", False))
    eureka = (1.0 if eureka_fired else 0.0) * _safe_float(dna.get("break_eureka_weight"), 0.08)

    # TimeChain: historical break wisdom (success rate of similar past breaks)
    pattern_score = _safe_float(subsystem_signals.get("timechain_break_pattern"), 0.0)
    pattern = pattern_score * _safe_float(dna.get("break_timechain_pattern_weight"), 0.06)

    # Smart contracts: was the break triggered by a contract?
    trigger_score = _safe_float(subsystem_signals.get("contract_break_trigger"), 0.0)
    trigger = trigger_score * _safe_float(dna.get("break_contract_trigger_weight"), 0.05)

    # 2026-04-13 (Phase 3 of foundational healing rFP): cost weight
    # default raised -0.05 → 0.0. BREAK is a meta-cognitive escape tool,
    # not a failure to be punished. The previous -0.05 baseline plus the
    # legacy M7 penalty (-0.08 per use, removed in same commit) added up
    # to -0.13 baseline per BREAK use, training the policy to never
    # select BREAK regardless of context. Cost is now opt-in via DNA
    # config — leave at 0 unless tuned per-Titan to encode a real
    # situational cost.
    cost = _safe_float(dna.get("break_cost_weight"), 0.0)

    # 2026-04-13 — clamp at 0.0 like all other primitive helpers do
    # (FORMULATE/RECALL/EVALUATE/INTROSPECT all use max(0.0, ...)).
    # This restores BREAK to symmetry with peers — its rewards now grow
    # from base + recovery + eureka + pattern + trigger, never below 0.
    total = max(0.0, base + recovery + eureka + pattern + trigger + cost)
    return total, {
        "base": round(base, 4),
        "recovery": round(recovery, 4),
        "eureka": round(eureka, 4),
        "pattern": round(pattern, 4),
        "trigger": round(trigger, 4),
        "cost": round(cost, 4),
        "total": round(total, 4),
    }


# ── Subsystem Signal Collector (Sub-phase A bridge) ──────────────


def empty_subsystem_signals() -> dict:
    """Return a dict of all subsystem signals defaulted to 0.

    Used as the safe-default cache when subsystem queries fail or are
    disabled. Each compound reward function gracefully degrades to its
    base reward when signals are 0.
    """
    return {
        # RECALL signals
        "inner_relevance": 0.0,
        "kuzu_centrality": 0.0,
        "timechain_depth": 0.0,
        "contract_ratified": 0.0,
        # FORMULATE signals
        "timechain_novelty": 0.5,  # 0.5 = unknown novelty (neutral)
        "contract_priority": 0.0,
        # EVALUATE signals
        "timechain_eval_consistency": 0.0,
        "contract_compliance": 0.0,
        # INTROSPECT signals
        "self_prediction_accuracy": 0.0,
        "self_profile_divergence": 0.0,
        "timechain_self_continuity": 0.0,
        "contract_identity_alignment": 0.0,
        # BREAK signals
        "timechain_break_pattern": 0.0,
        "contract_break_trigger": 0.0,
    }


# ── Public dispatcher ────────────────────────────────────────────


PRIMITIVE_REWARD_HELPERS = {
    "RECALL": reward_recall,
    "FORMULATE": reward_formulate,
    "EVALUATE": reward_evaluate,
    "INTROSPECT": reward_introspect,
    "BREAK": reward_break,
}


def compute_primitive_reward(
    primitive: str,
    state: Any,
    step_output: dict,
    dna: dict,
    subsystem_signals: dict,
) -> tuple[float, dict]:
    """Dispatch to the appropriate compound reward helper.

    Returns (0.0, {}) for primitives without a compound reward (HYPOTHESIZE,
    DELEGATE, SYNTHESIZE, SPIRIT_SELF — these stay on the existing reward path).
    """
    helper = PRIMITIVE_REWARD_HELPERS.get(primitive)
    if helper is None:
        return 0.0, {}
    return helper(state, step_output, dna, subsystem_signals)
