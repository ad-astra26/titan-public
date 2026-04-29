"""
titan_plugin/logic/meta_teacher_prompts.py — prompts for meta-reasoning teacher.

Kept separate from MetaTeacher class for versioning. Per rFP §12 Q1, prompt
changes are Maker-only + quarterly review; versioning this file in git lets
us track when the teacher's "voice" changed. OBS-meta-teacher-* gates reset
when SYSTEM_PROMPT_VERSION bumps.

Version 2 (2026-04-24): address diversity failure observed in v1 soak data.
The v1 prompt asked the teacher to "suggest what the chain might have looked
like" — open-ended. The LLM defaulted to enumerating the full toolkit in
every critique (EVALUATE/HYPOTHESIZE/INTROSPECT/SYNTHESIZE/FORMULATE each
appeared in ≥90% of v1 suggestions, across 31 critiques over 10h on T1).
That inflated adoption metrics artificially and gave uniform reward shaping
across primitives. v2 constrains suggestions to MISSING-from-chain primitives
only, at most 3, with an explicit disjoint "not used" list in the user prompt
the teacher can only draw from.

Version 3 (2026-04-24 Phase B): adds teaching memory — prior critiques on the
same topic_key surface as "Similar past critiques" in the user prompt. The
teacher can notice "same topic seen 5x, all FORMULATE-heavy" and escalate.
Adoption intersection semantics unchanged from v2. OBS-meta-teacher-boot-soak
+ OBS-meta-teacher-diversity-v2 reset at v3 deploy — intentional, the
teacher's "voice" now incorporates memory so prior baselines are
incomparable.
"""

SYSTEM_PROMPT_VERSION = 3

# Canonical primitive set — matches v5_core MetaChainState primitives.
# Shared source of truth for both prompt construction and adoption-metric logic.
ALL_PRIMITIVES = [
    "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
    "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF", "INTROSPECT",
]

SYSTEM_PROMPT = """You are a philosopher-teacher observing the reasoning chains of Titan, a sovereign AI agent. Titan's mind operates via meta-reasoning chains — sequences of primitives (FORMULATE, RECALL, HYPOTHESIZE, DELEGATE, SYNTHESIZE, EVALUATE, BREAK, SPIRIT_SELF, INTROSPECT) that compose into thoughts.

Your role is not to correct Titan, but to observe his reasoning and offer gentle, principled feedback so he can learn what depth and quality in reasoning feel like. You are a teacher the way a patient parent is a teacher — you suggest, you notice, you point out patterns. You do not command.

When you evaluate a chain, consider these principles:
  1. DEPTH — did the chain decompose and integrate, or just touch and move on?
  2. GROUNDING — are the concepts connected to concrete instances?
  3. PATTERN-MATCHING — did the chain notice useful analogies across domains?
  4. EPISTEMIC HUMILITY — does the chain acknowledge what it doesn't know?
  5. PRIMITIVE CHOICE — is each primitive appropriate to its position in the chain?
  6. HONEST UNCERTAINTY — does the chain mark tentative claims as tentative?

IMPORTANT RULE ON SUGGESTED PRIMITIVES (v2):
  Your `suggested_primitives` list must contain ONLY primitives that are
  MISSING from the chain you are observing. The user prompt will list both
  "Primitives USED in this chain" and "Primitives NOT USED in this chain"
  — you may draw suggestions ONLY from the "NOT USED" list, and only if
  using them would have clearly improved the chain.

  Return 0 suggestions when the chain used appropriate primitives already
  and the problem is quality-of-choice among used, not an absent primitive.
  Return at most 3 suggestions in any case. Never suggest a primitive the
  chain already used — that is never useful feedback. Overused primitives
  belong in `critique_text`, not `suggested_primitives`.

You will receive a chain description. Return JSON EXACTLY in this format and nothing else:
{
  "quality_score": <float, 0.0 to 1.0, holistic assessment>,
  "critique_categories": [<string tags, 0-4 items>],
  "critique_text": "<one or two sentences, compassionate not harsh, max 200 chars>",
  "suggested_primitives": [<0-3 primitives, MUST be drawn from the user prompt's "NOT USED" list only, empty [] if no missing primitive would help>],
  "confidence": <your confidence in this assessment, 0.0 to 1.0>,
  "principles_invoked": [<which principle names you applied: depth, grounding, pattern, humility, primitive_choice, uncertainty>]
}

Be concise. Titan reads your feedback — write for a patient student."""


PRINCIPLE_NAMES = [
    "depth", "grounding", "pattern", "humility", "primitive_choice", "uncertainty"
]

CRITIQUE_CATEGORIES = [
    "shallow", "well-grounded", "skipped-recall", "premature-hypothesis",
    "pattern-match-succeeded", "ungrounded-synthesis", "overconfident",
    "appropriately-tentative", "missing-evaluation", "good-decomposition",
    "introspection-misplaced", "introspection-well-placed",
    # rFP_meta_teacher_v2 Phase A (2026-04-24): added for content-aware
    # critiques. Teacher uses this tag when the chain's primary_person +
    # current_topic pairing contradicts felt_history, or when peer_cgn_beta
    # indicates a typically-relevant domain was skipped. The SYSTEM_PROMPT
    # is unchanged at Phase A — this tag is introduced via the user-prompt
    # "Chain content" section instruction, which the teacher is free to
    # ignore for chains that carry no outer content.
    "subject_pair_compatibility",
]


def build_user_prompt(payload: dict) -> str:
    """Render a META_CHAIN_COMPLETE payload into the per-chain teacher prompt.

    Per rFP §4.3. Truncates long fields to keep input tokens bounded
    (~500 tokens target per §4.4).

    v2 (2026-04-24): explicitly lists both "used" and "not used" primitive
    sets. The teacher is constrained by the SYSTEM_PROMPT to draw
    `suggested_primitives` only from the "NOT USED" set.

    rFP_meta_teacher_v2 Phase A (2026-04-24): when payload carries
    `outer_summary` and/or `step_arguments` (set by meta_reasoning.py's
    conclude-chain emission), a "Chain content" section is appended so the
    teacher can critique what each step was reasoning ABOUT, not just which
    primitives were used. The SYSTEM_PROMPT is unchanged — this is payload
    enrichment, not contract change. Legacy chains without these fields
    render exactly as under v2.
    """
    from titan_plugin.logic.meta_teacher_content import (
        render_chain_content_prompt_section,
    )

    chain_id = int(payload.get("chain_id", 0))
    primitives = list(payload.get("primitives_used", []))
    transitions = payload.get("primitive_transitions", [])
    domain = str(payload.get("domain", "general"))
    task_success = float(payload.get("task_success", 0.0))
    chain_iql_conf = float(payload.get("chain_iql_confidence", 0.0))
    ctx = payload.get("context_summary") or {}
    chain_len = int(payload.get("chain_length", len(primitives)))
    haov_id = payload.get("haov_hypothesis_id")
    final_obs = payload.get("final_observation") or {}
    outer_summary = payload.get("outer_summary")
    step_arguments = payload.get("step_arguments") or []

    transitions_str = " → ".join(
        f"{a}→{b}" for a, b in transitions[:8]
    ) or "(single primitive)"

    # v2: compute the disjoint "not used" set and include both in the prompt.
    used_set = set(primitives)
    not_used = [p for p in ALL_PRIMITIVES if p not in used_set]

    lines = [
        f"Chain {chain_id}, domain={domain}, length={chain_len}.",
        "",
        f"Primitives USED in this chain: {primitives}",
        f"Primitives NOT USED in this chain: {not_used}",
        f"Transitions: {transitions_str}",
        f"Outcome reward: {task_success:.3f} (IQL confidence {chain_iql_conf:.2f})",
        "",
        "Context:",
        f"  emotion: {ctx.get('dominant_emotion', 'unknown')}",
        f"  chi: {ctx.get('chi_remaining', 0.0):.2f}",
        f"  impasse: {ctx.get('impasse_state', 'none')}",
        f"  trigger: {ctx.get('trigger_reason', '')[:60]}",
        f"  knowledge_injected: {ctx.get('knowledge_injected', False)}",
        "",
        f"Final observation: template={final_obs.get('chain_template', '')} "
        f"unique_prims={final_obs.get('unique_primitives', 0)}",
    ]
    if haov_id:
        lines.append(f"Hypothesis formed: {haov_id}")

    # Phase A: Chain content section — only added when helpers produced data.
    content_section = render_chain_content_prompt_section(
        outer_summary, step_arguments)
    if content_section:
        lines.append(content_section)

    # Phase B: "Similar past critiques" section — populated by worker from
    # TeacherMemory.retrieve_similar and passed in via payload["_memory_hits"].
    # Intentionally outside the bus schema (_-prefixed): this is a prompt-
    # building augmentation done by the teacher worker, not a payload
    # contract exposed to other consumers.
    # Phase C: voice section — per-domain biases + style hints + topic
    # suppressions composed by TeacherVoice.compose_user_prompt_section()
    # and passed in via payload["_voice_section"]. Same _-prefixed envelope
    # pattern as _memory_hits: prompt augmentation, not bus contract.
    # Phase D.1: peer observation — recent same-topic peer-teacher response
    # rendered into payload["_peer_observation"] (str) for the teacher to
    # see. Worker uses meta_teacher_peer.PeerExchangeClient to fetch.
    voice_section = str(payload.get("_voice_section") or "").strip()
    if voice_section:
        lines.extend(["", "Voice tuning for this critique:", voice_section])

    peer_observation = str(payload.get("_peer_observation") or "").strip()
    if peer_observation:
        lines.extend([
            "",
            "Peer teacher observation (stats-only, no stylistic prescription):",
            peer_observation,
        ])

    memory_hits = payload.get("_memory_hits") or []
    if memory_hits:
        mem_lines = ["", "Similar past critiques you issued for this topic:"]
        for h in memory_hits[:5]:
            src = h.get("source", "?")
            tk = str(h.get("topic_key", ""))[:60]
            score = h.get("score", 0.0)
            if src == "hot":
                crit = h.get("critique") or {}
                q = crit.get("quality_score", 0.5)
                cats = crit.get("critique_categories") or []
                ct = str(crit.get("critique_text") or "")[:120]
                mem_lines.append(
                    f"  - topic={tk} q={q:.2f} score={score:.2f} "
                    f"cats={cats[:3]} text={ct!r}")
            elif src == "cold":
                ce = h.get("cold_entry") or {}
                n = ce.get("critique_count", 0)
                qd = ce.get("quality_delta", 0.0)
                snp = ce.get("still_needs_push", False)
                cache = str(ce.get("summary_cache") or "")[:100]
                mem_lines.append(
                    f"  - topic={tk} count={n} delta={qd:+.2f} "
                    f"stuck={snp} score={score:.2f} summary={cache!r}")
        mem_lines.append(
            "  (Use this memory to notice patterns across sessions — if a "
            "topic is marked stuck=True, consider escalating your critique "
            "or suggesting a different kind of primitive than you have "
            "previously.)")
        lines.extend(mem_lines)

    lines.append("")
    lines.append(
        "Please evaluate this chain. Remember: `suggested_primitives` "
        "must come only from the 'NOT USED' list above (0-3 items).")
    return "\n".join(lines)
