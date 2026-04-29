"""
titan_plugin/logic/meta_teacher_content.py — Phase A of rFP_meta_teacher_v2.

Surfaces the content of a meta-reasoning chain to the teacher. Two helpers:

  build_teacher_outer_summary(chain_state) -> dict | None
      Distills MetaChainState.outer_context (composed by the outer-layer) into
      a teacher-safe ~300-token summary: entity labels, per-source status,
      top felt_summaries, peer_cgn_beta across consumers, sentiment/arousal/
      relevance averages. Returns None when outer_context is absent (graceful
      fallback — teacher reverts to syntactic critique).

  build_step_arguments(chain_state) -> list[dict]
      Lightweight per-step summary: primitive + sub_mode + short arg_summary
      + result shape. Works even when outer_context is empty — inner-state-only
      chains still get step-level content surfaced to the teacher.

Redaction philosophy (rFP §2, §3): no new redaction module. We apply truncation
caps + safe label handling only (handles kept for referent stability; raw
quoted text capped to 60 chars per felt_summary, overall dict bounded by
caller's outer_summary_max_tokens config). The OVG/Guardian trust boundary
for outbound LLM calls is enforced at the worker level — this module produces
content the worker can pass through that pipeline.

See rFP_meta_teacher_v2_content_awareness_memory.md §2 Phase A.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("titan.meta_teacher_content")


# ── Redaction / truncation caps ────────────────────────────────────────────
# Conservative defaults; caller overrides via config (outer_summary_max_tokens
# and step_arguments_max_tokens in [meta_teacher]). These caps are field-level
# belts on top of the token-level caps the caller applies.
_FELT_SUMMARY_MAX_CHARS = 60
_FELT_TOP_N = 2
_INNER_NARR_TOP_N = 2
_INNER_NARR_MAX_CHARS = 80
_TRIGGER_MAX_CHARS = 80
_ARG_SUMMARY_MAX_CHARS = 50
_PEER_CGN_CONSUMERS = (
    "meta", "emot", "language", "social", "knowledge", "coding", "reasoning",
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any, max_chars: int) -> str:
    if v is None:
        return ""
    s = str(v)
    return s[:max_chars]


def _source_status_map(outer: dict) -> dict[str, str]:
    """Condense sources_queried/failed/timed_out into per-source status.

    Values: "ok" (present + no failure), "timeout", "failed", "absent".
    Keys cover the label set from OuterContextReader._compose_sync().
    """
    queried = set(outer.get("sources_queried") or [])
    failed = set(outer.get("sources_failed") or [])
    timed_out = set(outer.get("sources_timed_out") or [])
    status: dict[str, str] = {}
    for label in queried:
        if label in timed_out:
            status[label] = "timeout"
        elif label in failed:
            status[label] = "failed"
        else:
            status[label] = "ok"
    return status


def _avg_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _extract_felt_averages(felt_history: list[dict]) -> dict[str, Optional[float]]:
    """Averages across felt_history rows for sentiment/arousal/relevance.

    Each row is a dict from FeltExperiencesReader.get_for_* (events_teacher.db
    felt_experiences columns). Missing values are skipped per-field; empty
    history yields all None.
    """
    sent, aro, rel = [], [], []
    for row in felt_history or []:
        if not isinstance(row, dict):
            continue
        for key, bucket in (
            ("sentiment", sent), ("arousal", aro), ("relevance", rel),
        ):
            v = row.get(key)
            if v is None:
                continue
            try:
                bucket.append(float(v))
            except (TypeError, ValueError):
                continue
    return {
        "sentiment_avg": _avg_or_none(sent),
        "arousal_avg": _avg_or_none(aro),
        "relevance_avg": _avg_or_none(rel),
    }


def _top_felt_summaries(felt_history: list[dict], n: int = _FELT_TOP_N) -> list[str]:
    """Pick top-N felt_summaries by relevance desc, cap each at 60 chars.

    Fallback to chronological (first N) if no relevance signal present.
    """
    if not felt_history:
        return []
    scored: list[tuple[float, str]] = []
    any_rel = False
    for row in felt_history:
        if not isinstance(row, dict):
            continue
        summary = _safe_str(row.get("felt_summary") or "", _FELT_SUMMARY_MAX_CHARS)
        if not summary:
            continue
        rel_raw = row.get("relevance")
        if rel_raw is None:
            rel = 0.0
        else:
            any_rel = True
            rel = _safe_float(rel_raw, 0.0)
        scored.append((rel, summary))
    if not scored:
        return []
    if any_rel:
        scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]


def _peer_cgn_beta_for_topic(
    peer_cgn: dict, topic: str,
) -> dict[str, Optional[float]]:
    """Extract β per consumer for the current topic, if present in peer_cgn.

    peer_cgn shape (from PeerCGNReader.peer_cgn_summary): {consumer: {...}}.
    When a consumer doesn't carry the topic, returns None for that slot (not
    the same as "present but zero"). Callers treat None as "not grounded".
    """
    if not isinstance(peer_cgn, dict) or not topic:
        return {c: None for c in _PEER_CGN_CONSUMERS}
    topic_key = str(topic).strip().lower()
    out: dict[str, Optional[float]] = {}
    for consumer in _PEER_CGN_CONSUMERS:
        state = peer_cgn.get(consumer)
        if not isinstance(state, dict):
            out[consumer] = None
            continue
        # Accept either flat {topic: beta} form or nested "concept_grounding"
        # form — PeerCGNReader.peer_cgn_summary layers differ per consumer
        # per its schema-tolerant extractor. Probe both.
        beta_val: Optional[float] = None
        # Direct lookup first (case-insensitive)
        for k, v in state.items():
            if isinstance(k, str) and k.lower() == topic_key and isinstance(v, (int, float)):
                beta_val = float(v)
                break
        if beta_val is None:
            # Nested "concept_grounding"/"_primitives"/"regions" probe
            for nested_key in ("concept_grounding", "_primitives", "regions"):
                nested = state.get(nested_key)
                if not isinstance(nested, dict):
                    continue
                for k, v in nested.items():
                    if isinstance(k, str) and k.lower() == topic_key:
                        if isinstance(v, (int, float)):
                            beta_val = float(v)
                        elif isinstance(v, dict):
                            candidate = v.get("beta") or v.get("beta_posterior")
                            if isinstance(candidate, (int, float)):
                                beta_val = float(candidate)
                        break
                if beta_val is not None:
                    break
        out[consumer] = None if beta_val is None else round(max(0.0, min(1.0, beta_val)), 3)
    return out


def _inner_narrative_snippets(outer: dict) -> list[str]:
    """Top-N inner_narrative snippets (if any), each capped to 80 chars.

    outer["inner_narrative"] items are dicts from InnerMemoryReader; each row
    carries a "text"/"snippet"/"summary" string depending on query variant.
    We probe common fields; empty list if nothing text-shaped.
    """
    items = outer.get("inner_narrative") or []
    if not items:
        return []
    out: list[str] = []
    for row in items[:_INNER_NARR_TOP_N]:
        if isinstance(row, str):
            out.append(_safe_str(row, _INNER_NARR_MAX_CHARS))
            continue
        if not isinstance(row, dict):
            continue
        for key in ("text", "snippet", "summary", "content"):
            v = row.get(key)
            if v:
                out.append(_safe_str(v, _INNER_NARR_MAX_CHARS))
                break
    return out


def _titan_self_pick(titan_self: dict) -> dict[str, Any]:
    """Pick a small set of teacher-relevant self-state fields.

    Used for context — NOT for "what should Titan do" critique. Helps the
    teacher ground tone (e.g., low chi + high impasse → be gentle).
    """
    if not isinstance(titan_self, dict):
        return {}
    picked: dict[str, Any] = {}
    for key in ("mood", "dominant_emotion", "chi_remaining", "impasse_state",
                 "pi_rate", "epoch"):
        if key in titan_self:
            picked[key] = titan_self[key]
    return picked


# ── Public API ─────────────────────────────────────────────────────────────

def build_teacher_outer_summary(chain_state) -> Optional[dict]:
    """Distill MetaChainState.outer_context for the teacher.

    Returns None when outer_context is empty / missing — the caller should
    then NOT include an outer_summary in the META_CHAIN_COMPLETE payload,
    and the teacher's user_prompt will omit the "Chain content" section
    gracefully.

    The returned dict is teacher-safe: string fields are length-capped,
    numeric fields bounded. Do NOT pass raw SQL rows through this function;
    it assumes outer_context was already composed by OuterContextReader
    and carries only its published schema.

    Shape (omits None/empty fields):
        {
          "primary_person":  <handle or None>,
          "current_topic":   <topic string or None>,
          "current_event":   <event id or None>,
          "sources_status":  {"person": "ok", "felt_person": "timeout", ...},
          "felt_summaries":  ["...", "..."],        # top 2 by relevance
          "felt_averages":   {"sentiment_avg": 0.42, "arousal_avg": ...},
          "peer_cgn_beta":   {"meta": 0.6, "emot": None, ...},
          "titan_self":      {"mood": ..., "chi_remaining": ...},
          "inner_narrative": ["snippet1", "snippet2"],
          "fetch_ms":        123.4,
        }
    """
    try:
        outer = getattr(chain_state, "outer_context", None) or {}
        refs = getattr(chain_state, "entity_refs", None) or {}
    except Exception:
        return None
    if not isinstance(outer, dict) or not outer:
        return None

    primary_person = refs.get("primary_person") if isinstance(refs, dict) else None
    current_topic = refs.get("current_topic") if isinstance(refs, dict) else None
    current_event = refs.get("current_event") if isinstance(refs, dict) else None

    felt_history = outer.get("felt_history") or []
    summary: dict[str, Any] = {}
    if primary_person:
        summary["primary_person"] = _safe_str(primary_person, 64)
    if current_topic:
        summary["current_topic"] = _safe_str(current_topic, 80)
    if current_event:
        summary["current_event"] = _safe_str(current_event, 64)

    sources_status = _source_status_map(outer)
    if sources_status:
        summary["sources_status"] = sources_status

    felt_summaries = _top_felt_summaries(felt_history)
    if felt_summaries:
        summary["felt_summaries"] = felt_summaries
    felt_avg = _extract_felt_averages(felt_history)
    # Only include non-None averages to keep payload tight.
    felt_avg_nonnull = {k: v for k, v in felt_avg.items() if v is not None}
    if felt_avg_nonnull:
        summary["felt_averages"] = felt_avg_nonnull

    peer_beta = _peer_cgn_beta_for_topic(
        outer.get("peer_cgn") or {}, current_topic or "")
    # Only include when current_topic is present AND we have ≥1 non-null beta.
    if current_topic and any(v is not None for v in peer_beta.values()):
        summary["peer_cgn_beta"] = peer_beta

    titan_self = _titan_self_pick(outer.get("titan_self_snapshot") or {})
    if titan_self:
        summary["titan_self"] = titan_self

    narr = _inner_narrative_snippets(outer)
    if narr:
        summary["inner_narrative"] = narr

    fetch_ms = outer.get("fetch_ms")
    if fetch_ms is not None:
        try:
            summary["fetch_ms"] = round(float(fetch_ms), 1)
        except (TypeError, ValueError):
            pass

    if not summary:
        return None
    return summary


def build_step_arguments(chain_state) -> list[dict]:
    """Per-step summary: primitive + sub_mode + arg_summary + result shape.

    Complementary to outer_summary — steps carry step-level intent regardless
    of whether outer-layer is active. Inner-state-only chains still produce
    meaningful step_arguments (e.g., FORMULATE.goal_setting with the goal
    label, HYPOTHESIZE.extend_pattern with the hypothesis seed).

    chain_state.chain is list[str] of "PRIMITIVE.sub_mode" keys — populated
    by MetaEngine._dispatch after each step. chain_state.chain_results is
    list[dict] aligned index-wise (one result per step).

    Returns a list one-per-step, same length as chain_state.chain. Each
    entry:
        {
          "step": <int, 0-based>,
          "primitive": <PRIMITIVE name>,
          "sub_mode": <sub-mode string>,
          "arg_summary": "<short, <=50 chars>",     # optional
          "result_shape": {"count": N, ...}          # optional; result fields
        }
    """
    try:
        chain = list(getattr(chain_state, "chain", None) or [])
        results = list(getattr(chain_state, "chain_results", None) or [])
        refs = getattr(chain_state, "entity_refs", None) or {}
    except Exception:
        return []
    if not chain:
        return []

    step_args: list[dict] = []
    for idx, step_key in enumerate(chain):
        step_key_str = str(step_key)
        if "." in step_key_str:
            prim, sub = step_key_str.split(".", 1)
        else:
            prim, sub = step_key_str, ""
        entry: dict[str, Any] = {
            "step": idx,
            "primitive": prim[:32],
            "sub_mode": sub[:40],
        }
        # Arg-summary: a short label of what this primitive was operating on.
        # Derive from entity_refs + result shape without inventing new state.
        arg_summary = _derive_arg_summary(prim, sub, refs, results, idx)
        if arg_summary:
            entry["arg_summary"] = _safe_str(arg_summary, _ARG_SUMMARY_MAX_CHARS)
        result = results[idx] if idx < len(results) else None
        if isinstance(result, dict):
            shape = _result_shape(result)
            if shape:
                entry["result_shape"] = shape
        step_args.append(entry)
    return step_args


def _derive_arg_summary(
    prim: str, sub: str, refs: dict, results: list, idx: int,
) -> Optional[str]:
    """Compose a short label for a step's "subject" for teacher visibility.

    Convention:
      - RECALL.entity / RECALL.topic → the referent entity / topic from refs
      - FORMULATE.* → result's goal_label if present
      - HYPOTHESIZE.* → result's hypothesis_seed if present
      - SYNTHESIZE.* → result's synthesis_label if present
      - DELEGATE.gap_fill → refs["current_topic"]
      - EVALUATE.* → "quality=<float>" from result
      - Otherwise: None (teacher sees primitive + sub_mode only)
    """
    refs = refs if isinstance(refs, dict) else {}
    primary_person = refs.get("primary_person")
    current_topic = refs.get("current_topic")
    current_event = refs.get("current_event")
    prim_upper = prim.upper()
    sub_lower = sub.lower()
    result = results[idx] if 0 <= idx < len(results) else None
    result = result if isinstance(result, dict) else {}

    if prim_upper == "RECALL":
        if sub_lower == "entity":
            return primary_person or None
        if sub_lower == "topic":
            return current_topic or None
        if sub_lower in ("experience", "wisdom"):
            return current_topic or primary_person or None
    if prim_upper == "FORMULATE":
        for key in ("goal_label", "formulate_goal", "intent"):
            v = result.get(key)
            if v:
                return str(v)
        return current_topic or None
    if prim_upper == "HYPOTHESIZE":
        for key in ("hypothesis_seed", "seed", "label"):
            v = result.get(key)
            if v:
                return str(v)
    if prim_upper == "SYNTHESIZE":
        for key in ("synthesis_label", "label", "summary"):
            v = result.get(key)
            if v:
                return str(v)
    if prim_upper == "DELEGATE" and sub_lower == "gap_fill":
        return current_topic or primary_person or None
    if prim_upper == "EVALUATE":
        quality = result.get("quality_score") or result.get("quality")
        if isinstance(quality, (int, float)):
            return f"quality={float(quality):.2f}"
    if prim_upper == "SPIRIT_SELF":
        return current_event or "self"
    if prim_upper == "INTROSPECT":
        return current_topic or "inner"
    if prim_upper == "BREAK":
        # "BREAK after N steps" is more useful than a label
        return f"after_step_{max(0, idx - 1)}"
    return None


def _result_shape(result: dict) -> dict:
    """Pick a minimal "shape" from a step's result dict.

    Keeps the payload bounded — caller's token cap is the ultimate authority.
    """
    shape: dict[str, Any] = {}
    for key in ("count", "found", "wisdom_found", "confidence",
                 "quality_score", "result_type"):
        if key in result:
            v = result[key]
            if isinstance(v, bool):
                shape[key] = v
            elif isinstance(v, (int, float)):
                try:
                    shape[key] = round(float(v), 3) if isinstance(v, float) else int(v)
                except (TypeError, ValueError):
                    pass
            elif isinstance(v, str):
                shape[key] = v[:24]
    return shape


def render_chain_content_prompt_section(
    outer_summary: Optional[dict], step_arguments: list[dict],
) -> str:
    """Render the "Chain content" prompt section for the teacher.

    Returns an empty string when neither outer_summary nor step_arguments
    carry meaningful content. Caller (build_user_prompt) inserts this into
    the per-chain user prompt.

    Format keeps lines short for LLM parsing reliability.
    """
    has_outer = bool(outer_summary)
    has_steps = bool(step_arguments)
    if not has_outer and not has_steps:
        return ""

    lines: list[str] = ["", "Chain content:"]
    if has_outer:
        os_ = outer_summary  # shorthand
        if os_.get("primary_person"):
            lines.append(f"  primary_person: {os_['primary_person']}")
        if os_.get("current_topic"):
            lines.append(f"  current_topic: {os_['current_topic']}")
        if os_.get("current_event"):
            lines.append(f"  current_event: {os_['current_event']}")
        fa = os_.get("felt_averages") or {}
        if fa:
            pieces = []
            for k in ("sentiment_avg", "arousal_avg", "relevance_avg"):
                if fa.get(k) is not None:
                    short = k.replace("_avg", "")
                    pieces.append(f"{short}={fa[k]}")
            if pieces:
                lines.append("  felt_averages: " + " ".join(pieces))
        felt = os_.get("felt_summaries") or []
        if felt:
            lines.append("  felt_summaries:")
            for s in felt:
                lines.append(f"    - {s}")
        pcb = os_.get("peer_cgn_beta") or {}
        if pcb:
            grounded = [f"{c}={v}" for c, v in pcb.items() if v is not None]
            if grounded:
                lines.append("  peer_cgn_beta: " + " ".join(grounded))
        ss = os_.get("sources_status") or {}
        if ss:
            bad = {k: v for k, v in ss.items() if v not in ("ok",)}
            if bad:
                lines.append("  sources_issues: " + str(bad))
        narr = os_.get("inner_narrative") or []
        if narr:
            lines.append("  inner_narrative:")
            for s in narr:
                lines.append(f"    - {s}")
    if has_steps:
        lines.append("  steps:")
        for sa in step_arguments:
            prim = sa.get("primitive", "")
            sub = sa.get("sub_mode", "")
            label = f"{prim}.{sub}" if sub else prim
            arg = sa.get("arg_summary", "")
            piece = f"    [{sa.get('step', 0)}] {label}"
            if arg:
                piece += f"  arg={arg}"
            shape = sa.get("result_shape") or {}
            if shape:
                shape_pieces = []
                for k in ("count", "found", "confidence", "quality_score"):
                    if k in shape:
                        shape_pieces.append(f"{k}={shape[k]}")
                if shape_pieces:
                    piece += "  " + " ".join(shape_pieces)
            lines.append(piece)

    lines.append("")
    lines.append(
        "If the primary_person + current_topic pairing is incoherent with "
        "the felt_averages or prior felt_summaries, or if the peer_cgn_beta "
        "suggests a domain was skipped that is typically relevant to this "
        "topic, you MAY add 'subject_pair_compatibility' to "
        "critique_categories and comment on it in critique_text.")
    return "\n".join(lines)
