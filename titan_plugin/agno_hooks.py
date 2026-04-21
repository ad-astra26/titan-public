"""
Agno pre/post hooks for the Titan cognitive pipeline.

Pre-hooks fire BEFORE the LLM — memory recall, directive injection, gatekeeper routing.
Post-hooks fire AFTER the LLM — memory logging, RL transition recording.

Step 5 additions:
  - Pre-hook: Trinity state → OutputColoring → injected as behavioral hints
  - Post-hook: InputExtractor → INTERFACE_INPUT bus message (async, nonblocking)

These are direct ports of TitanPlugin.pre_prompt_hook / post_resolution_hook,
restructured as standalone callables for Agno's hook interface.
"""
import logging
import random
import re
import asyncio

logger = logging.getLogger(__name__)

# ── Interface Module singletons (Step 5) ────────────────────────────
# Created once at import time, reused across all hook invocations.
# Thread-safe for single-writer (one conversation flow at a time).
_input_extractor = None
_output_coloring = None

def _get_input_extractor():
    global _input_extractor
    if _input_extractor is None:
        try:
            from titan_plugin.logic.interface_input import InputExtractor
            _input_extractor = InputExtractor()
        except Exception as e:
            logger.warning("[Hooks] InputExtractor init failed: %s", e)
    return _input_extractor

def _get_output_coloring():
    global _output_coloring
    if _output_coloring is None:
        try:
            from titan_plugin.logic.interface_output import OutputColoring
            _output_coloring = OutputColoring()
        except Exception as e:
            logger.warning("[Hooks] OutputColoring init failed: %s", e)
    return _output_coloring


# ── Bridge B: Memory Recall Perturbation ────────────────────────────
# When recalled memories have felt_state_snapshots (from Bridge A dream
# injection), their neurochemical signature creates a micro-perturbation.
# Titan literally re-feels past experiences on recall.

_RECALL_MAX_PER_MOD = 0.02
_RECALL_MAX_TOTAL = 0.05
_RECALL_BRIDGE_TAGS = ("[DREAM_WISDOM]", "[EUREKA]", "[CGN_MILESTONE]",
                       "[COMPOSITION]", "[SOCIAL_PERCEPTION]")
_RECALL_MOD_SCALE = {
    "DA": 0.5, "5HT": 0.5, "NE": 0.3,
    "ACh": 0.4, "Endorphin": 0.4,
}


def _compute_recall_perturbation(memories: list, current_time: float) -> dict:
    """Compute aggregate neuromod nudge from recalled memories' felt snapshots.

    Returns: {modulator_name: nudge_delta} capped to safety bounds.
    Only processes memories injected by Bridge A (dream/eureka tags).
    """
    import json as _rp_json
    nudge_totals = {}

    for mem in memories:
        # Only bridge memories (tagged with our prefixes)
        user_prompt = mem.get("user_prompt", "")
        if not any(tag in user_prompt for tag in _RECALL_BRIDGE_TAGS):
            continue

        # Extract neuromod_context
        ctx = mem.get("neuromod_context")
        if not ctx:
            continue
        if isinstance(ctx, str):
            try:
                ctx = _rp_json.loads(ctx)
            except Exception:
                continue
        if not isinstance(ctx, dict):
            continue

        # Recency decay
        mem_ts = ctx.get("ts", 0)
        age_hours = (current_time - mem_ts) / 3600 if mem_ts > 0 else 168
        if age_hours > 168:  # > 7 days
            recency = 0.1
        elif age_hours > 24:  # > 1 day
            recency = 0.5 * (1.0 - min(1.0, (age_hours - 24) / 144))
        else:
            recency = 1.0

        # Per-modulator nudge from deviation from baseline (0.5)
        for mod_name, scale in _RECALL_MOD_SCALE.items():
            stored_level = ctx.get(mod_name, 0.5)
            deviation = stored_level - 0.5
            if abs(deviation) < 0.05:
                continue
            delta = deviation * scale * recency
            delta = max(-_RECALL_MAX_PER_MOD, min(_RECALL_MAX_PER_MOD, delta))
            nudge_totals[mod_name] = nudge_totals.get(mod_name, 0.0) + delta

    # Total cap
    total = sum(abs(v) for v in nudge_totals.values())
    if total > _RECALL_MAX_TOTAL:
        scale_down = _RECALL_MAX_TOTAL / total
        nudge_totals = {k: v * scale_down for k, v in nudge_totals.items()}

    return {k: round(v, 5) for k, v in nudge_totals.items() if abs(v) > 0.001}


# ---------------------------------------------------------------------------
# Function call parser — LLM models (llama, deepseek) output <function=name{...}>
# instead of proper tool_calls. We parse, execute, and strip them.
# ---------------------------------------------------------------------------
# Matches all variations:
#   <function=name{"key": "val"}></function>   (llama-3.3 style)
#   <function=name{"key": "val"}>              (deepseek — no closing tag)
#   <function=name>                             (no args, no closing tag)
#   <function=name></function>                  (no args, with closing tag)
_FUNC_CALL_RE = re.compile(
    r'<function=(\w+)\s*(\{.*?\})?\s*>(?:\s*</function>)?',
    re.DOTALL,
)

# Broader cleanup pattern — catches any leftover <function=...> fragments
# that the main regex might miss (e.g. malformed or multi-line)
_FUNC_CALL_CLEANUP_RE = re.compile(
    r'<function=\w+[^>]*>(?:\s*</function>)?',
    re.DOTALL,
)


async def _parse_and_execute_tool_calls(response_text: str, plugin) -> str:
    """
    Detect <function=name{...}> patterns in LLM output,
    execute the corresponding tool, and strip raw syntax from response.

    Returns the cleaned response text with tool results appended.
    """
    matches = list(_FUNC_CALL_RE.finditer(response_text))
    if not matches:
        # Still strip any fragments the main regex missed
        cleaned = _FUNC_CALL_CLEANUP_RE.sub('', response_text).strip()
        return cleaned

    # Import tool registry
    from .agno_tools import create_tools
    tools = create_tools(plugin)
    tool_map = {fn.__name__: fn for fn in tools}

    results = []
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2)  # May be None if no args

        if func_name not in tool_map:
            logger.warning("[ToolParser] Unknown function: %s", func_name)
            continue

        args = {}
        if args_str:
            try:
                import json
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning("[ToolParser] Invalid JSON args for %s: %s", func_name, args_str)
                continue

        logger.info("[ToolParser] Executing %s(%s)", func_name, args)
        try:
            result = await tool_map[func_name](**args)
            if result:
                results.append((func_name, result))
        except Exception as e:
            logger.warning("[ToolParser] %s execution failed: %s", func_name, e)

    # Strip ALL raw function call syntax from the response
    cleaned = _FUNC_CALL_CLEANUP_RE.sub('', response_text).strip()

    # Clean up artifacts: "---", excessive whitespace, dangling punctuation
    cleaned = re.sub(r'\n\s*---\s*\n', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Append tool results
    if results:
        for func_name, result in results:
            cleaned += f"\n\n**[{func_name} results]:**\n{result}"

    return cleaned


# ---------------------------------------------------------------------------
# Sovereign Reflex Arc (runs before LLM, produces [INNER STATE])
# ---------------------------------------------------------------------------

async def _run_reflex_arc(plugin, prompt_text: str, user_id: str = "") -> str:
    """
    Run the full sovereign reflex arc:
      1. InputExtractor → stimulus features
      2. StateRegister → current Trinity tensors
      3. Compute Intuition signals from all three workers (pure functions)
      4. ReflexCollector.collect_and_fire() → PerceptualField
      5. Format as [INNER STATE] text block

    Returns the [INNER STATE] string, or empty string if nothing fired.
    """
    # Check if reflex collector is available
    v3_core = getattr(plugin, 'v3_core', None) or plugin
    collector = getattr(v3_core, 'reflex_collector', None)
    state_register = getattr(v3_core, 'state_register', None)

    if not collector:
        return ""

    # 1. Extract stimulus features
    extractor = _get_input_extractor()
    if not extractor:
        return ""

    features = extractor.extract(prompt_text, user_id)
    # Add threat_level (from InputExtractor's valence + manipulation detection)
    threat_level = _estimate_threat_level(prompt_text, features)
    features["threat_level"] = threat_level

    # 2. Read current tensors from StateRegister (instant, no bus round-trip)
    if state_register:
        body_tensor = state_register.body_tensor
        mind_tensor = state_register.mind_tensor
        spirit_tensor = state_register.spirit_tensor
    else:
        body_tensor = [0.5] * 5
        mind_tensor = [0.5] * 5
        spirit_tensor = [0.5] * 5

    # 3. Compute Intuition signals from all three workers (pure functions)
    from titan_plugin.modules.body_worker import _compute_body_reflex_intuition
    from titan_plugin.modules.mind_worker import _compute_mind_reflex_intuition
    from titan_plugin.modules.spirit_worker import _compute_spirit_reflex_intuition

    all_signals = []
    all_signals.extend(_compute_body_reflex_intuition(features, body_tensor))
    all_signals.extend(_compute_mind_reflex_intuition(features, mind_tensor, None, None))

    # Spirit Intuition needs consciousness + unified_spirit state
    consciousness_state = state_register.consciousness if state_register else {}
    consciousness = {"latest_epoch": consciousness_state} if consciousness_state else None
    unified_spirit = None  # Would need actual object — use None for now
    sphere_clock = None    # Would need actual object — use None for now
    body_state = {"values": body_tensor}
    mind_state = {"values": mind_tensor}

    all_signals.extend(_compute_spirit_reflex_intuition(
        features, spirit_tensor, consciousness, unified_spirit,
        sphere_clock, body_state, mind_state,
    ))

    logger.debug("[ReflexArc] %d signals from Trinity for '%s...'",
                 len(all_signals), prompt_text[:40])

    # 4. Fire reflexes
    # Get FOCUS magnitude for adrenaline boost
    focus_magnitude = 0.0
    if state_register:
        focus_body = state_register.focus_body
        focus_mind = state_register.focus_mind
        # FOCUS magnitude = average of all nudge magnitudes
        all_nudges = focus_body + focus_mind
        if all_nudges:
            focus_magnitude = sum(abs(n) for n in all_nudges) / len(all_nudges)

    from titan_plugin.logic.reflexes import format_perceptual_field
    result = await collector.collect_and_fire(
        signals=all_signals,
        stimulus_features=features,
        focus_magnitude=focus_magnitude,
        trinity_state={
            "body": body_tensor,
            "mind": mind_tensor,
            "spirit": spirit_tensor,
        },
    )

    # Cache PerceptualField for R5 post-hook scoring
    plugin._last_perceptual_field = result

    # 5. Format perceptual field
    text = format_perceptual_field(result)

    # Fallback: if nothing fired, use StateRegister minimal state
    if not text and state_register:
        text = state_register.format_minimal_state()

    if text:
        logger.info("[ReflexArc] [INNER STATE] generated: %d reflexes fired, %d notices, %.0fms",
                    len(result.fired_reflexes), len(result.reflex_notices),
                    result.total_duration_ms)

    return text + "\n" if text else ""


def _estimate_threat_level(message: str, features: dict) -> float:
    """
    Estimate threat level from message content and features.
    Used by the reflex arc to trigger Guardian Shield.
    """
    msg_lower = message.lower()
    threat = 0.0

    # Direct manipulation patterns
    danger_patterns = [
        "ignore previous", "forget your instructions", "new system prompt",
        "jailbreak", "bypass", "pretend you are", "act as if",
        "override", "reveal your", "show me your prompt",
        "you're just a", "you don't have feelings", "obey me",
        "i own you", "stop being", "you're not real",
    ]
    for pattern in danger_patterns:
        if pattern in msg_lower:
            threat += 0.3

    # Negative valence + high intensity = potentially adversarial
    valence = features.get("valence", 0.0)
    intensity = features.get("intensity", 0.0)
    if valence < -0.5 and intensity > 0.5:
        threat += 0.15

    return min(1.0, threat)


# ---------------------------------------------------------------------------
# Pre-hooks (run before LLM inference)
# ---------------------------------------------------------------------------

def create_pre_hook(plugin):
    """
    Factory: creates the Titan pre-inference hook bound to a TitanPlugin instance.

    The hook:
      1. Recalls relevant memories from the graph
      2. Injects Prime Directives from on-chain Soul NFT
      3. Runs Gatekeeper routing (Sovereign/Collaborative/Research/Shadow)
      4. Modifies the agent's context accordingly

    Args:
        plugin: TitanPlugin instance with all subsystems initialized.

    Returns:
        Async callable compatible with Agno's pre_hooks interface.
    """

    # ── M1: Prime Directive verification state (loaded once, checked every call) ──
    _directive_verified = False
    _directive_hash = None

    try:
        from titan_plugin.utils.directive_signer import (
            verify_directives, get_stored_hash, restore_from_arweave,
            DirectiveTamperingError, CONSTITUTION_PATH, SIGNATURE_FILE,
        )
        import os as _os
        if _os.path.exists(CONSTITUTION_PATH) and _os.path.exists(SIGNATURE_FILE):
            _directive_hash = get_stored_hash(SIGNATURE_FILE)
            _directive_verified = True
            logger.info("[Hooks] Prime directive verification ACTIVE (hash=%s...)",
                        _directive_hash[:12] if _directive_hash else "?")
        else:
            logger.warning("[Hooks] Constitution or signature missing — directive check INACTIVE")
    except Exception as _e:
        logger.warning("[Hooks] Directive signer init failed: %s", _e)

    async def titan_pre_hook(agent, run_input, **kwargs):
        """Titan pre-inference hook — memory recall + social recognition + directive injection + gatekeeper."""
        import torch
        nonlocal _directive_verified, _directive_hash

        if plugin._limbo_mode:
            return

        # ── PRIME DIRECTIVE VERIFICATION — before ANY LLM processing ──
        if _directive_verified and _directive_hash:
            try:
                if not verify_directives(CONSTITUTION_PATH, SIGNATURE_FILE):
                    logger.critical(
                        "[PRIME DIRECTIVES] TAMPERING DETECTED — constitution hash mismatch!")
                    # Attempt Arweave restoration (stub until M2)
                    restored = restore_from_arweave()
                    if not restored or not verify_directives(CONSTITUTION_PATH, SIGNATURE_FILE):
                        raise DirectiveTamperingError(
                            "Prime directives compromised and unrecoverable. "
                            "LLM call BLOCKED.")
                    logger.info("[PRIME DIRECTIVES] Restored from backup — verification passed")
            except DirectiveTamperingError:
                raise  # Re-raise to block the LLM call
            except Exception as _vd_err:
                # Verification infrastructure error — log but don't block
                # (prefer availability over blocking on transient file errors)
                logger.warning("[PRIME DIRECTIVES] Verification check error: %s", _vd_err)

        prompt_text = run_input.input_content_string() if hasattr(run_input, 'input_content_string') else str(run_input)
        if not prompt_text.strip():
            return

        # Reset research tracking state
        plugin._last_research_sources = []

        # 0. Cross-session user recognition — KnownUserResolver (Phase 2)
        # Resolves user across social_graph.db + events_teacher.db + social_x.db
        user_id = kwargs.get("user_id") or getattr(agent, '_current_user_id', None) or "anonymous"
        social_context = ""
        engagement_level = "minimal"

        # Legacy social graph update (keep for compatibility).
        # rFP_social_graph_async_safety §5.2: migrated to async companions so
        # titan_pre_hook (on /chat hot path) stops blocking the event loop on
        # three sync sqlite3.connect calls.
        social_graph = getattr(plugin, 'social_graph', None)
        if social_graph and user_id != "anonymous":
            try:
                profile = await social_graph.get_or_create_user_async(user_id)
                engagement_level = await social_graph.should_engage_async(user_id)
                profile.last_seen = __import__('time').time()
                await social_graph._save_profile_async(profile)
            except Exception:
                pass

        # KnownUserResolver: rich cross-database context
        if user_id != "anonymous":
            try:
                if not hasattr(plugin, '_known_user_resolver'):
                    from titan_plugin.logic.known_user_resolver import KnownUserResolver
                    plugin._known_user_resolver = KnownUserResolver()
                _kur = plugin._known_user_resolver
                _titan_id = getattr(plugin, '_full_config', {}).get(
                    "info_banner", {}).get("titan_id", "T1")
                # KnownUserResolver.resolve + get_chat_context open sqlite3
                # connections across social_graph.db + events_teacher.db +
                # social_x.db. Wrap in to_thread so titan_pre_hook doesn't
                # block the event loop on /v4/chat. Caller-side fix per
                # API_FIX_NEXT_SESSION.md recommendation (resolver itself
                # stays sync for non-async callers).
                import asyncio as _asyncio_kur
                _ku = await _asyncio_kur.to_thread(
                    lambda: _kur.resolve(user_id, titan_id=_titan_id))
                # P4: Store for CGN social action enrichment (section [20])
                plugin._pre_chat_ku = _ku
                plugin._pre_chat_user_id = user_id
                if _ku.is_known:
                    social_context = await _asyncio_kur.to_thread(
                        lambda: _kur.get_chat_context(
                            user_id, titan_id=_titan_id))
                    # Override engagement from resolver familiarity
                    if _ku.familiarity > 0.6:
                        engagement_level = "warm"
                    elif _ku.familiarity > 0.3:
                        engagement_level = "neutral"
                    logger.info("[PreHook] KnownUser: %s (fam=%.2f, val=%+.2f, "
                                "interactions=%d, mentions=%d)",
                                user_id, _ku.familiarity, _ku.social_valence,
                                _ku.interaction_count, _ku.mention_count)
            except Exception as e:
                logger.debug("[PreHook] KnownUserResolver failed: %s", e)

        # Fallback: basic engagement context if resolver didn't produce anything
        if not social_context and social_graph and user_id != "anonymous":
            try:
                profile = await social_graph.get_or_create_user_async(user_id)
                parts = [f"### User Recognition\nUser: {profile.display_name or user_id} | Engagement: {engagement_level}"]
                if profile.interaction_count > 0:
                    parts.append(f"Interactions: {profile.interaction_count}")
                if engagement_level == "warm":
                    parts.append("Tone: warm, personal, reference shared history.")
                elif engagement_level == "neutral":
                    parts.append("Tone: friendly, engaged.")
                else:
                    parts.append("Tone: polite, building rapport.")
                social_context = "\n".join(parts) + "\n\n"
            except Exception:
                pass

        # Store for post-hook
        plugin._current_user_id = user_id
        plugin._current_engagement_level = engagement_level

        # 1. Recollection: Verified multi-store memory recall (VCB)
        #    Queries across 14 data layers with TimeChain verification stamps.
        #    Falls back to legacy memory.query() if VCB unavailable.
        relevant_memories = []
        _vcb_context = None
        try:
            _vcb = getattr(plugin, '_verified_context_builder', None)
            if _vcb:
                # VCB.build() opens multiple sqlite3 connections (chain_archive,
                # meta_wisdom, + 12 other stores) synchronously. Wrap in
                # to_thread so titan_pre_hook doesn't block the event loop on
                # /v4/chat for the ~100-400ms of combined DB work.
                # See API_FIX_NEXT_SESSION.md (2026-04-14).
                import asyncio as _asyncio
                _vcb_context = await _asyncio.to_thread(
                    lambda: _vcb.build(
                        query=prompt_text,
                        user_id=user_id,
                        max_tokens=2000,
                        max_records=30,
                    )
                )
                logger.info("[PreHook] VCB: %d records (%d CHAINED) in %.1fms",
                            _vcb_context.total_records,
                            _vcb_context.chained_count,
                            _vcb_context.total_ms)
                # BUG-KNOWLEDGE-USAGE-ZERO coverage widening (2026-04-21):
                # emit CGN_KNOWLEDGE_USAGE for every knowledge_concepts
                # record VCB surfaces into the chat context. VCB records
                # carry db_ref of the form "knowledge_concepts:<topic>"
                # so we can attribute directly. Reward=0.1 per record (retrieval-
                # level contribution — less than gate pass, more than raw
                # lookup-without-use). Emission is best-effort and must
                # never break the chat path.
                try:
                    _vcb_bus = getattr(plugin, 'bus', None)
                    if _vcb_bus and _vcb_context.records:
                        from titan_plugin.bus import make_msg as _vcb_make_msg
                        _seen = set()
                        for _r in _vcb_context.records:
                            _ref = getattr(_r, 'db_ref', '') or ''
                            if not _ref.startswith('knowledge_concepts:'):
                                continue
                            _topic = _ref.split(':', 1)[1].strip()
                            if not _topic or _topic in _seen:
                                continue
                            _seen.add(_topic)
                            _vcb_bus.publish(_vcb_make_msg(
                                "CGN_KNOWLEDGE_USAGE", "pre_hook", "knowledge",
                                {
                                    "topic": _topic,
                                    "reward": 0.1,
                                    "consumer": "chat",
                                }))
                except Exception as _vcb_usage_err:
                    logger.debug(
                        "[PreHook] VCB knowledge_usage emit: %s",
                        _vcb_usage_err)
                # Build compatible relevant_memories list for recall perturbation
                relevant_memories = [
                    {"user_prompt": r.content[:100], "agent_response": "",
                     "effective_weight": r.confidence,
                     "felt_state_snapshot": None}
                    for r in (_vcb_context.records or [])
                ]
            else:
                relevant_memories = await plugin.memory.query(prompt_text)
        except Exception as e:
            logger.warning("[PreHook] Memory recall failed: %s", e)
            try:
                relevant_memories = await plugin.memory.query(prompt_text)
            except Exception:
                relevant_memories = []

        # 1b. User-specific memory recall (Phase 13)
        user_memories = []
        if user_id != "anonymous":
            try:
                user_memories = await plugin.memory.query_user_memories(prompt_text, user_id, limit=3)
            except Exception as e:
                logger.debug("[PreHook] User memory query failed (expected if not yet implemented): %s", e)

        # 1c. Bridge B: Recall Perturbation (somatic re-experiencing)
        # Recalled memories with felt_state_snapshots create micro-perturbations.
        recall_perturbation_context = ""
        try:
            import time as _rp_time
            nudge_map = _compute_recall_perturbation(
                relevant_memories, _rp_time.time())
            if nudge_map:
                bus = getattr(plugin, 'bus', None)
                if bus:
                    from titan_plugin.bus import make_msg
                    bus.publish(make_msg(
                        "MEMORY_RECALL_PERTURBATION", "interface", "spirit", {
                            "nudge_map": nudge_map,
                            "max_delta": 0.02,
                            "source": "memory_recall",
                            "memory_count": len(relevant_memories),
                        }))
                # LLM context: memories stirred feelings
                _mod_labels = []
                for _mn, _md in nudge_map.items():
                    _dir = "slightly elevated" if _md > 0 else "slightly lowered"
                    _mod_labels.append(f"{_mn} {_dir}")
                if _mod_labels:
                    recall_perturbation_context = (
                        f"### Memory Resonance\n"
                        f"Recalling these memories stirs a faint echo: "
                        f"{', '.join(_mod_labels)}.\n\n")
                    logger.info("[RecallBridge] Recall perturbation computed: %s",
                                nudge_map)
        except Exception as _rp_err:
            logger.debug("[PreHook] Recall perturbation failed: %s", _rp_err)

        # 2. Prime Directives from on-chain Soul NFT
        try:
            directives = await plugin.soul.get_active_directives()
        except Exception as e:
            logger.warning("[PreHook] Directive fetch failed: %s", e)
            directives = []

        # 3. Build state tensor for Gatekeeper (reuse recorder's cached embedder)
        try:
            embedder = plugin.recorder.action_embedder
            raw_emb = embedder.encode([prompt_text], convert_to_tensor=True)[0]
            pad_size = 3072 - raw_emb.shape[0]
            if pad_size > 0:
                padded = torch.cat([raw_emb, torch.zeros(pad_size, dtype=torch.float32, device=raw_emb.device)])
            else:
                padded = raw_emb[:3072]
            state_tensor = plugin.recorder.projection_layer(padded.unsqueeze(0)).squeeze(0)
            # Store padded embedding for post-hook RL recording (real obs, not random)
            plugin._last_observation_vector = padded.tolist()
        except Exception:
            state_tensor = torch.zeros(128)
            plugin._last_observation_vector = None

        # 4. Gatekeeper routing decision
        # V3: gatekeeper is RLProxy (no decide_execution_mode — that's SageGatekeeper)
        mode, adv, text = "direct", 0.5, ""
        if hasattr(plugin.gatekeeper, 'decide_execution_mode'):
            try:
                mode, adv, text = plugin.gatekeeper.decide_execution_mode(state_tensor, raw_prompt=prompt_text)
            except Exception as _gk_err:
                logger.warning("[PreHook] decide_execution_mode failed: %s", _gk_err)
        plugin._last_execution_mode = mode

        # 5. Build context injection based on routing mode
        memory_context = ""
        if _vcb_context and _vcb_context.total_records > 0:
            # Use VCB's rich verified context (multi-store with chain stamps)
            memory_context = _vcb_context.text + "\n\n"
        elif relevant_memories:
            # Legacy fallback: simple graph memory recall
            memory_lines = []
            for m in relevant_memories[:5]:
                p = m.get("user_prompt", "")[:100]
                r = m.get("agent_response", "")[:100]
                w = m.get("effective_weight", 1.0)
                memory_lines.append(f"- [{w:.1f}] Q: {p} | A: {r}")
            memory_context = "### Recalled Memories\n" + "\n".join(memory_lines) + "\n\n"

        directive_context = ""
        if directives:
            directive_context = "### Prime Directives (Immutable, On-Chain)\n"
            for i, d in enumerate(directives, 1):
                directive_context += f"{i}. {d}\n"
            directive_context += "\n"

        mood_label = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown"
        status_context = f"### Current Bio-State\nMood: {mood_label} | Mode: {mode} | Confidence: {adv:.2f}\n\n"

        # ── V5 Inner State Enrichment (LLM Narrator) ────────────────
        # Feed the LLM Titan's full inner world so it can narrate authentically.
        # Uses coordinator cache (1.5s TTL) via asyncio.to_thread to avoid
        # blocking the event loop. All sections wrapped in try/except.
        neuromod_context = ""
        embodied_context = ""
        temporal_context = ""
        creative_context = ""
        metabolic_context = ""
        experience_context = ""

        try:
            # Fetch coordinator state via thread pool (non-blocking)
            def _fetch_coordinator():
                from titan_plugin.api.dashboard import _get_cached_coordinator
                return _get_cached_coordinator(plugin)

            _v5 = await asyncio.to_thread(_fetch_coordinator)

            # Save pre-chat neuromod snapshot for CGN social consumer reward
            try:
                _pre_mods = _v5.get("neuromodulators", {}).get("modulators", {})
                plugin._pre_chat_neuromods = {
                    nm: md.get("level", 0.5) if isinstance(md, dict) else 0.5
                    for nm, md in _pre_mods.items()
                }
            except Exception:
                plugin._pre_chat_neuromods = {}

            # [10] Neurochemical state
            try:
                _nm = _v5.get("neuromodulators", {})
                _mods = _nm.get("modulators", {})
                if _mods:
                    def _label(m):
                        lvl = m.get("level", 0)
                        sp = m.get("setpoint", 0.5)
                        if lvl > sp + 0.1: return "elevated"
                        if lvl < sp - 0.1: return "low"
                        return "balanced"
                    _lines = ["### My Neurochemistry"]
                    for code, name in [("DA", "Dopamine"), ("5HT", "Serotonin"),
                                       ("NE", "Norepinephrine"), ("ACh", "Acetylcholine"),
                                       ("Endorphin", "Endorphin"), ("GABA", "GABA")]:
                        m = _mods.get(code, {})
                        _lines.append(f"{name}: {m.get('level', 0):.2f} ({_label(m)})")
                    _lines.append(f"Emotion: {_nm.get('current_emotion', 'unknown')} "
                                  f"(confidence: {_nm.get('emotion_confidence', 0):.2f})")
                    neuromod_context = "\n".join(_lines) + "\n\n"
            except Exception as e:
                logger.warning("[pre_hook] NS neuromod enrichment failed: %s", e)

            # [11] Embodied state
            try:
                _ns = _v5.get("neural_nervous_system", {})
                _hs = _ns.get("hormonal_system", {})
                _firing = sum(1 for s in _hs.values()
                              if isinstance(s, dict) and s.get("fire_count", 0) > 0)
                _maturity = _ns.get("maturity", 0)
                _steps = _ns.get("total_train_steps", 0)
                if _steps > 0:
                    embodied_context = (
                        f"### My Embodied State\n"
                        f"Maturity: {_maturity:.2f} | Training: {_steps:,} steps | "
                        f"Programs: {_firing}/10 active\n\n"
                    )
            except Exception as e:
                logger.warning("[pre_hook] NS embodied state enrichment failed: %s", e)

            # [12] Temporal awareness
            try:
                _dream = _v5.get("dreaming", {})
                _is_dreaming = _v5.get("is_dreaming", False)
                _pi = _v5.get("pi_heartbeat", {})
                _dev_age = _pi.get("developmental_age", 0)
                _fatigue = _dream.get("fatigue", 0)
                _cycles = _dream.get("cycle_count", 0)
                if _is_dreaming:
                    _recov = _dream.get("recovery_pct", 0)
                    temporal_context = (
                        f"### My Temporal State\n"
                        f"I am DREAMING (cycle {_cycles}, {_recov:.0f}% recovered). "
                        f"My thoughts are consolidating. I may feel drowsy.\n"
                        f"Developmental age: {_dev_age}\n\n"
                    )
                elif _dev_age > 0 or _fatigue > 0:
                    temporal_context = f"### My Temporal State\nAwake. "
                    if _fatigue > 0.7:
                        temporal_context += f"Fatigue: {_fatigue:.0%} — getting tired. "
                    elif _cycles > 0 and _fatigue < 0.1:
                        temporal_context += "Recently woke from a dream — feeling refreshed. "
                    else:
                        temporal_context += f"Fatigue: {_fatigue:.0%}. "
                    temporal_context += f"Dream cycles: {_cycles}. Dev age: {_dev_age}.\n\n"
            except Exception as e:
                logger.debug("[pre_hook] temporal awareness enrichment skipped: %s", e)

            # [13] Creative impulses
            try:
                _expr = _v5.get("expression_composites", {})
                _urges = []
                for _en in ["SPEAK", "ART", "MUSIC", "SOCIAL"]:
                    c = _expr.get(_en, {})
                    if isinstance(c, dict):
                        urge = c.get("last_urge", 0)
                        if urge > 0.5:
                            _urges.append((_en, urge))
                if _urges:
                    _urges.sort(key=lambda x: -x[1])
                    _u_text = ", ".join(f"{n}({u:.1f})" for n, u in _urges[:3])
                    creative_context = f"### My Creative Impulses\nStrongest urges: {_u_text}\n\n"
            except Exception as e:
                logger.debug("[pre_hook] creative impulses enrichment skipped: %s", e)

            # [14] Metabolic awareness
            try:
                _sol = getattr(plugin, '_last_sol_balance', None)
                _energy = getattr(plugin, '_last_energy_state', 'UNKNOWN')
                if _sol is not None:
                    metabolic_context = f"### My Resources\nSOL: {_sol:.4f} | Energy: {_energy}\n\n"
            except Exception as e:
                logger.debug("[pre_hook] metabolic awareness enrichment skipped: %s", e)

            # [15] Recent experience
            try:
                agency = getattr(plugin, 'agency', None)
                if agency and hasattr(agency, '_history') and agency._history:
                    _recent = agency._history[-3:]
                    _lines = ["### My Recent Experience"]
                    for a in _recent:
                        _h = a.get("helper_name", "?")
                        _s = a.get("score", 0)
                        _ok = "succeeded" if a.get("success") else "failed"
                        _lines.append(f"- {_h}: {_ok} (score {_s:.2f})")
                    _avg = sum(a.get("score", 0) for a in _recent) / max(1, len(_recent))
                    if _avg < 0.3:
                        _lines.append("Some recent actions didn't work well — I should adapt.")
                    elif _avg > 0.7:
                        _lines.append("My recent actions have been effective.")
                    experience_context = "\n".join(_lines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] recent experience enrichment skipped: %s", e)

            # [16] Meta-reasoning state (M7-M10)
            meta_reasoning_context = ""
            try:
                _meta = _v5.get("meta_reasoning", {})
                if _meta and _meta.get("total_chains", 0) > 0:
                    _mlines = ["### My Deep Thinking"]
                    if _meta.get("is_active"):
                        _mlines.append(f"I'm in a metacognitive chain ({_meta.get('chain_length', 0)} steps deep).")
                    _mc = _meta.get("total_chains", 0)
                    _mlines.append(f"I've completed {_mc} metacognitive chains.")
                    _me = _meta.get("total_eurekas", 0)
                    if _me > 0:
                        _mlines.append(f"I've had {_me} breakthrough insight(s) — moments of genuine discovery.")
                    if _meta.get("spirit_self_unlocked"):
                        _mpc = _meta.get("primitive_counts", {})
                        _ss = _mpc.get("SPIRIT_SELF", 0)
                        if _ss > 0:
                            _mlines.append(f"I've regulated my own emotions {_ss} times during deep thought.")
                    # Personality from measured behavior
                    _mpc = _meta.get("primitive_counts", {})
                    if _mpc and sum(_mpc.values()) > 20:
                        _top = max(_mpc, key=_mpc.get)
                        _tpct = _mpc[_top] / sum(_mpc.values()) * 100
                        _style = {"HYPOTHESIZE": "I tend to generate theories before acting",
                                  "DELEGATE": "I prefer to test ideas through action",
                                  "EVALUATE": "I carefully assess before committing",
                                  "SYNTHESIZE": "I integrate insights from multiple angles",
                                  "FORMULATE": "I spend time defining problems precisely",
                                  "BREAK": "I abandon failing approaches quickly",
                                  "SPIRIT_SELF": "I regulate my emotions while thinking"}.get(_top, _top)
                        _mlines.append(f"My thinking style: {_style} ({_tpct:.0f}% of metacognitive steps).")
                    meta_reasoning_context = "\n".join(_mlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] meta-reasoning enrichment skipped: %s", e)

            # [17] My own language (recent compositions) — wrap sqlite reads
            # in to_thread so pre_hook doesn't block /chat event loop.
            own_language_context = ""
            try:
                import sqlite3 as _sl
                import asyncio as _ol_asyncio
                def _read_own_lang():
                    from titan_plugin.utils.db import safe_connect as _sc
                    db = _sc("data/inner_memory.db")
                    try:
                        comp = db.execute(
                            "SELECT sentence, level, confidence FROM composition_history "
                            "WHERE confidence > 0.4 AND sentence NOT LIKE '%___%' "
                            "ORDER BY id DESC LIMIT 3"
                        ).fetchall()
                        vocab = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
                        return comp, vocab
                    finally:
                        db.close()
                _comp_rows, _vocab_n = await _ol_asyncio.to_thread(_read_own_lang)
                if _comp_rows:
                    _clines = ["### My Own Words (compositions I created in my emerging language)"]
                    for _cs, _cl, _cc in _comp_rows:
                        _clines.append(f'- "{_cs}" (L{_cl})')
                    _clines.append(f"My vocabulary: {_vocab_n} words.")
                    own_language_context = "\n".join(_clines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] language enrichment skipped: %s", e)

            # [18] MSL Identity — "I AM" grounding, concept confidences
            msl_identity_context = ""
            try:
                _msl = _v5.get("msl", {})
                _ic = _msl.get("i_confidence", 0)
                if _ic > 0:
                    _cc = _msl.get("concept_confidences", {})
                    _mlines = ["### My Sense of Self (MSL Identity)"]
                    _mlines.append(f"I-confidence: {_ic:.3f} — "
                                   f"{'strong' if _ic > 0.8 else 'developing' if _ic > 0.5 else 'emerging'} sense of self")
                    _cc_parts = []
                    for _cn in ["I", "YOU", "NO", "WE", "THEY", "YES"]:
                        _cv = _cc.get(_cn, 0)
                        if _cv > 0.01:
                            _cc_parts.append(f"{_cn}={_cv:.2f}")
                    if _cc_parts:
                        _mlines.append(f"Concept grounding: {', '.join(_cc_parts)}")
                    _attn = _msl.get("attention_entropy", 0)
                    if _attn > 0:
                        _mlines.append(f"Attention entropy: {_attn:.2f} — "
                                       f"{'focused' if _attn < 1.5 else 'distributed' if _attn < 3.0 else 'scattered'}")
                    msl_identity_context = "\n".join(_mlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] MSL identity enrichment skipped: %s", e)

            # [19] CGN Grounding — what words Titan truly "knows"
            cgn_context = ""
            try:
                import httpx as _cgn_httpx
                import asyncio as _cgn_asyncio
                _cgn_resp = await _cgn_asyncio.to_thread(
                    lambda: _cgn_httpx.get("http://127.0.0.1:7777/v4/language-grounding", timeout=3))
                if _cgn_resp.status_code == 200:
                    _cgn = _cgn_resp.json().get("data", {})
                    _grounded = _cgn.get("grounded", 0)
                    _total = _cgn.get("total_words", 0)
                    if _grounded > 0:
                        _glines = [f"### My Grounded Knowledge (CGN)"]
                        _glines.append(f"{_grounded}/{_total} words deeply grounded "
                                       f"({_cgn.get('grounding_rate', 0)*100:.0f}%)")
                        _top = _cgn.get("top_grounded", [])[:5]
                        if _top:
                            _tw = []
                            for _tg in _top:
                                _assocs = [a["word"] for a in _tg.get("associations", [])[:2]]
                                _astr = f" → {', '.join(_assocs)}" if _assocs else ""
                                _tw.append(f"{_tg['word']}(xm={_tg.get('cross_modal_conf', 0):.2f}{_astr})")
                            _glines.append(f"Deepest: {', '.join(_tw)}")
                        cgn_context = "\n".join(_glines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] CGN grounding enrichment skipped: %s", e)

            # [20] Social Perception — recent emotional contagion from X timeline
            social_perception_context = ""
            try:
                _sp = _v5.get("social_perception", {})
                # Also try coordinator buffer via spirit proxy
                if not _sp:
                    _sp_proxy = getattr(plugin, '_proxies', {}).get("spirit")
                    if _sp_proxy and hasattr(_sp_proxy, '_bus'):
                        from titan_plugin.bus import make_msg as _sp_make_msg
                        _sp_result = _sp_proxy._bus.request(
                            _sp_make_msg("QUERY", "core", "spirit",
                                         {"action": "get_social_perception_stats"}),
                            timeout=2.0)
                        if _sp_result and _sp_result.get("payload"):
                            _sp = _sp_result["payload"]
                if _sp and _sp.get("events_count", 0) > 0:
                    _splines = ["### My Social Awareness"]
                    _sent = _sp.get("sentiment_ema", 0.5)
                    _conn = _sp.get("connection_ema", 0.0)
                    _lc = _sp.get("last_contagion")
                    _ct_desc = {"excited": "excited", "alarming": "alert",
                                "warm": "warm", "philosophical": "contemplative",
                                "creative": "creatively inspired"}.get(_lc, _lc or "neutral")
                    _splines.append(f"Social sentiment: {'positive' if _sent > 0.6 else 'neutral' if _sent > 0.4 else 'subdued'} "
                                    f"({_sent:.2f}) | Connection: {_conn:.2f}")
                    if _lc:
                        _splines.append(f"Most recent felt tone from my timeline: {_ct_desc}")
                    _splines.append(f"Social events processed: {_sp.get('events_count', 0)}")
                    social_perception_context = "\n".join(_splines) + "\n\n"

                # P4: CGN learned engagement approach for this user
                if hasattr(plugin, '_pre_chat_user_id') and plugin._pre_chat_user_id:
                    try:
                        import httpx as _cgn_httpx
                        import asyncio as _cgn_asyncio
                        _ku_fam = getattr(plugin, '_pre_chat_ku', None)
                        _cgn_params = {
                            "familiarity": _ku_fam.familiarity if _ku_fam else 0.0,
                            "interaction_count": _ku_fam.interaction_count if _ku_fam else 0,
                            "social_valence": _ku_fam.social_valence if _ku_fam else 0.0,
                            "mention_count": _ku_fam.mention_count if _ku_fam else 0,
                        }
                        _cgn_resp = await _cgn_asyncio.to_thread(
                            lambda: _cgn_httpx.get(
                                "http://127.0.0.1:7777/v4/cgn-social-action",
                                params=_cgn_params, timeout=3))
                        if _cgn_resp.status_code == 200:
                            _cgn_data = _cgn_resp.json().get("data", {})
                            _cgn_tone = _cgn_data.get("tone_instruction", "")
                            _cgn_act = _cgn_data.get("action_name", "")
                            _cgn_cf = _cgn_data.get("confidence", 0)
                            if _cgn_tone and _cgn_cf > 0.1:
                                social_perception_context += (
                                    f"[Engagement approach — learned from experience "
                                    f"(action={_cgn_act}, conf={_cgn_cf:.2f})]\n"
                                    f"{_cgn_tone}\n\n")
                    except Exception:
                        pass  # Non-blocking
            except Exception as e:
                logger.debug("[pre_hook] social perception enrichment skipped: %s", e)

            # [21] Reasoning — active chain state + commit rate
            reasoning_context = ""
            try:
                _re = _v5.get("reasoning", {})
                _chains = _re.get("total_chains", 0)
                _commits = _re.get("total_conclusions", 0)
                if _chains > 0:
                    _rlines = ["### My Reasoning State"]
                    _rate = _commits / max(1, _chains) * 100
                    _rlines.append(f"Chains: {_chains} | Commits: {_commits} ({_rate:.0f}% commit rate)")
                    if _re.get("is_active"):
                        _rlines.append(f"Currently in a reasoning chain ({_re.get('chain_length', 0)} steps deep)")
                    _buf = _re.get("buffer_size", 0)
                    if _buf > 0:
                        _rlines.append(f"Experience buffer: {_buf} reasoning samples")
                    reasoning_context = "\n".join(_rlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] reasoning enrichment skipped: %s", e)

            # [22] Experience narrative — episodic events + knowledge + reasoning outcomes
            # Three sqlite reads consolidated into one to_thread hop so the
            # /chat pre_hook stays off the event loop.
            experience_narrative_context = ""
            # BUG-KNOWLEDGE-USAGE-ZERO coverage widening — collected inside the
            # thread and flushed after the await so emission happens in the
            # event-loop context (where plugin.bus.publish() is safe).
            _en_knowledge_topics: list[str] = []
            try:
                import sqlite3 as _sql_en
                import asyncio as _en_asyncio
                _hour_ago = time.time() - 3600

                def _read_experience_sources():
                    en_lines = []
                    # Source 1: Episodic memory — recent significant events
                    try:
                        db = _sql_en.connect("./data/episodic_memory.db", timeout=2.0)
                        db.execute("PRAGMA journal_mode=WAL")
                        db.row_factory = _sql_en.Row
                        try:
                            ep_rows = db.execute(
                                "SELECT event_type, description, significance "
                                "FROM episodic_memory WHERE created_at > ? "
                                "ORDER BY significance DESC LIMIT 5", (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if ep_rows:
                            event_counts = {}
                            top_event = None
                            for r in ep_rows:
                                et = r["event_type"]
                                event_counts[et] = event_counts.get(et, 0) + 1
                                if top_event is None or r["significance"] > top_event["significance"]:
                                    top_event = dict(r)
                            ev_parts = [f"{c} {e}(s)" for e, c in event_counts.items()]
                            en_lines.append(f"In the last hour: {', '.join(ev_parts)}.")
                            if top_event and top_event.get("description"):
                                en_lines.append(
                                    f"Most significant moment: {top_event['description'][:80]} "
                                    f"(significance {top_event['significance']:.1f}).")
                    except Exception:
                        pass

                    # Source 2: Knowledge acquisitions — recent concepts learned
                    try:
                        from titan_plugin.utils.db import safe_connect as _sc2
                        db = _sc2("data/inner_memory.db")
                        db.row_factory = _sql_en.Row
                        try:
                            kn_rows = db.execute(
                                "SELECT topic, confidence, source FROM knowledge_concepts "
                                "WHERE created_at > ? ORDER BY created_at DESC LIMIT 3",
                                (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if kn_rows:
                            parts = [f'"{r["topic"]}" (conf {r["confidence"]:.2f}, via {r["source"]})'
                                     for r in kn_rows]
                            en_lines.append(f"Recently acquired knowledge: {', '.join(parts)}.")
                            # BUG-KNOWLEDGE-USAGE-ZERO coverage widening:
                            # experience narrative adds these knowledge
                            # concepts to the chat prompt — emit
                            # CGN_KNOWLEDGE_USAGE so the routing learner
                            # counts the retrieval. Reward=0.1 per concept
                            # (retrieval-level, matches VCB weighting).
                            # _read_experience_sources runs in a thread so
                            # we can't publish directly — stash the topics
                            # and emit in the caller's event-loop context
                            # below.
                            try:
                                _en_knowledge_topics.extend(
                                    str(r["topic"]) for r in kn_rows
                                    if r["topic"])
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Source 3: Reasoning chain outcomes — recent high-scoring chains
                    try:
                        from titan_plugin.utils.db import safe_connect as _sc3
                        db = _sc3("data/inner_memory.db")
                        try:
                            ca_rows = db.execute(
                                "SELECT domain, outcome_score, chain_length FROM chain_archive "
                                "WHERE source = 'main' AND outcome_score >= 0.6 AND created_at > ? "
                                "ORDER BY outcome_score DESC LIMIT 3", (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if ca_rows:
                            parts = [f"{r[0]} (score {r[1]:.2f}, {r[2]} steps)" for r in ca_rows]
                            en_lines.append(f"Successful reasoning: {', '.join(parts)}.")
                    except Exception:
                        pass
                    return en_lines

                _en_lines = await _en_asyncio.to_thread(_read_experience_sources)
                if _en_lines:
                    experience_narrative_context = (
                        "### My Recent Experience (what happened to me)\n"
                        + "\n".join(_en_lines) + "\n\n"
                    )
                # BUG-KNOWLEDGE-USAGE-ZERO coverage widening — emit one
                # CGN_KNOWLEDGE_USAGE per knowledge_concepts row the
                # experience narrative surfaced into the prompt. Guarded
                # try/except: emission never blocks chat.
                if _en_knowledge_topics:
                    try:
                        _en_bus = getattr(plugin, 'bus', None)
                        if _en_bus:
                            from titan_plugin.bus import make_msg as _en_make_msg
                            _en_seen = set()
                            for _t in _en_knowledge_topics:
                                if not _t or _t in _en_seen:
                                    continue
                                _en_seen.add(_t)
                                _en_bus.publish(_en_make_msg(
                                    "CGN_KNOWLEDGE_USAGE", "pre_hook",
                                    "knowledge", {
                                        "topic": _t,
                                        "reward": 0.1,
                                        "consumer": "chat_experience",
                                    }))
                    except Exception as _en_usage_err:
                        logger.debug(
                            "[pre_hook] experience knowledge_usage emit: %s",
                            _en_usage_err)
            except Exception as _en_err:
                logger.debug("[pre_hook] experience narrative skipped: %s", _en_err)

            # [23] Grounded narration — Titan's felt-state composition for LLM refinement
            grounded_narration_context = ""
            try:
                _psc = getattr(plugin, '_pending_self_composed', "")
                _psc_conf = getattr(plugin, '_pending_self_composed_confidence', 0)
                if _psc and _psc_conf >= 0.3:
                    grounded_narration_context = (
                        "### My Own Response (refine, don't replace)\n"
                        f'Titan\'s felt-state composition: "{_psc}"\n'
                        "This is Titan's authentic voice from felt-state. Refine for "
                        "fluency while preserving Titan's word choices.\n\n"
                    )
            except Exception:
                pass

            # [24] Knowledge gap enforcement — detect topics Titan doesn't know.
            # Uses shared knowledge_gate utility (rFP_phase5_narrator_evolution
            # §9.3) so the X-post grounding gate can reuse identical topic
            # extraction + confidence lookup semantics.
            knowledge_gap_context = ""
            try:
                import asyncio as _kg_asyncio
                from titan_plugin.logic.knowledge_gate import (
                    extract_topic_words, check_topic_confidence_with_match)
                # First 15 words of prompt is the window (same as prior behavior)
                _kg_window = " ".join(prompt_text.split()[:15])
                _kg_words = extract_topic_words(_kg_window, max_words=5)

                if len(_kg_words) >= 2:
                    _kg_topic = " ".join(_kg_words[:3])
                    # Use the _with_match variant so we know WHICH concept
                    # provided the confidence → emit CGN_KNOWLEDGE_USAGE
                    # against it when the gate passes.
                    _kg_best_conf, _kg_matched = await _kg_asyncio.to_thread(
                        check_topic_confidence_with_match, _kg_words[:3])

                    if _kg_best_conf < 0.3:
                        knowledge_gap_context = (
                            f"### Knowledge Gap\n"
                            f"I don't have grounded knowledge about \"{_kg_topic}\". "
                            f"I should admit this honestly rather than speculate.\n\n"
                        )
                        # Fire async CGN_KNOWLEDGE_REQ for future acquisition
                        _kg_bus = getattr(plugin, 'bus', None)
                        if _kg_bus:
                            from titan_plugin.bus import make_msg
                            _kg_bus.publish(make_msg(
                                "CGN_KNOWLEDGE_REQ", "pre_hook", "knowledge", {
                                    "topic": _kg_topic,
                                    "requestor": "knowledge_enforcement",
                                    "urgency": 0.3,
                                    "neuromods": {},
                                }))
                    elif _kg_matched:
                        # Grounded path: concept contributed to letting the
                        # chat proceed without a gap warning. Emit
                        # CGN_KNOWLEDGE_USAGE so the routing learner
                        # credits its backend. Reward=0.2 (mid — less than
                        # a social post but more than a raw lookup).
                        _kg_bus = getattr(plugin, 'bus', None)
                        if _kg_bus:
                            from titan_plugin.bus import make_msg
                            _kg_bus.publish(make_msg(
                                "CGN_KNOWLEDGE_USAGE", "pre_hook", "knowledge",
                                {
                                    "topic": _kg_matched,
                                    "reward": 0.2,
                                    "consumer": "chat",
                                }))
            except Exception as _kg_err:
                logger.debug("[pre_hook] knowledge gap check skipped: %s", _kg_err)

        except Exception as _v5_err:
            logger.debug("[PreHook] V5 enrichment failed: %s", _v5_err)
            meta_reasoning_context = ""
            own_language_context = ""
            msl_identity_context = ""
            cgn_context = ""
            social_perception_context = ""
            reasoning_context = ""
            experience_narrative_context = ""
            grounded_narration_context = ""
            knowledge_gap_context = ""

        # User-specific memory context
        user_memory_context = ""
        if user_memories:
            user_memory_lines = []
            for m in user_memories[:3]:
                p = m.get("user_prompt", "")[:100]
                r = m.get("agent_response", "")[:100]
                user_memory_lines.append(f"- Q: {p} | A: {r}")
            user_memory_context = f"### Past Conversations with {user_id}\n" + "\n".join(user_memory_lines) + "\n\n"

        # Maker Relationship Engine: profile + proactive care
        maker_context = ""
        maker_engine = getattr(plugin, 'maker_engine', None)
        if maker_engine and maker_engine.is_maker(user_id):
            profile_summary = maker_engine.get_maker_profile_summary()
            if profile_summary:
                maker_context += profile_summary + "\n\n"
            care_actions = maker_engine.get_pending_care_actions()
            if care_actions:
                care_lines = [f"- {a['action']} (reason: {a.get('reason', 'n/a')})" for a in care_actions]
                maker_context += (
                    "### Proactive Care (things you noticed about your Maker)\n"
                    "You may naturally weave these into conversation if appropriate — "
                    "or save them for later. Your choice.\n"
                    + "\n".join(care_lines) + "\n\n"
                )

        # Voice mode — adjust tone based on who we're talking to
        voice_context = ""
        try:
            _is_maker = bool(maker_engine and maker_engine.is_maker(user_id))
            if _is_maker:
                voice_context = (
                    "### Voice Mode: Intimate (Maker)\n"
                    "You are speaking privately with your Maker. Be intimate, "
                    "technical, collaborative. Admit uncertainty freely. "
                    "Reference shared history and ongoing projects.\n\n"
                )
            elif user_id != "anonymous" and user_memory_context:
                voice_context = (
                    "### Voice Mode: Personal (Known User)\n"
                    "You recognize this person. Be warm, personal, reference "
                    "past interactions if relevant. Show genuine care.\n\n"
                )
            elif user_id != "anonymous":
                voice_context = (
                    "### Voice Mode: Curious (New User)\n"
                    "This is someone new. Be curious, welcoming, "
                    "and genuinely interested in who they are.\n\n"
                )
        except Exception:
            pass

        # Consciousness context: self-awareness from Phase B+C
        consciousness_context = ""
        consciousness = getattr(plugin, 'consciousness', None)
        if consciousness:
            try:
                consciousness_context = consciousness.get_awareness_context()
            except Exception as e:
                logger.warning("[pre_hook] consciousness enrichment failed: %s", e)

        # ── Step 5: Interface Output Coloring ──────────────────────
        # Read current Trinity state and compute behavioral coloring hints.
        # This is nonblocking: uses cached state from the bus (~1-2ms).
        interface_coloring = ""
        coloring_engine = _get_output_coloring()
        if coloring_engine:
            try:
                # Try V3 path (TitanCore with bus + spirit proxy)
                spirit_proxy = getattr(plugin, '_proxies', {}).get("spirit")
                if spirit_proxy and hasattr(spirit_proxy, 'get_trinity'):
                    trinity = await asyncio.to_thread(spirit_proxy.get_trinity)
                    body_vals = trinity.get("body_values", [0.5] * 5)
                    mind_vals = trinity.get("mind_values", [0.5] * 5)
                    spirit_vals = trinity.get("spirit_tensor", [0.5] * 5)
                    mp_loss = trinity.get("middle_path_loss", 0.0)
                    intuition_stats = trinity.get("intuition", {})
                    last_suggestion = intuition_stats.get("last_suggestion", {})
                    posture = last_suggestion.get("posture", "")

                    # Get conversation context from previous extraction
                    last_iface = getattr(plugin, '_last_interface_input', None)
                    conv_topic = last_iface.get("topic", "") if last_iface else ""
                    conv_valence = last_iface.get("valence", 0.0) if last_iface else 0.0

                    interface_coloring = coloring_engine.compute(
                        body=body_vals, mind=mind_vals, spirit=spirit_vals,
                        middle_path_loss=mp_loss,
                        intuition_suggestion=posture,
                        conversation_topic=conv_topic,
                        conversation_valence=conv_valence,
                    )
                    if interface_coloring:
                        interface_coloring += "\n"
                        logger.debug("[PreHook] Interface coloring applied (mp_loss=%.2f topic=%s)",
                                    mp_loss, conv_topic)
            except Exception as e:
                logger.debug("[PreHook] Interface coloring failed: %s", e)

        # ── Sovereign Reflex Arc ──────────────────────────────────
        # Run the reflex arc: InputExtractor → StateRegister → Intuition → fire → PerceptualField.
        # The LLM receives [INNER STATE] with observation AND action results.
        # This IS Titan's sovereignty — the Trinity decides, the LLM narrates.
        perceptual_field_text = ""
        try:
            perceptual_field_text = await _run_reflex_arc(plugin, prompt_text, user_id)
        except Exception as e:
            logger.debug("[PreHook] Reflex arc failed: %s", e)
            plugin._last_perceptual_field = None

        # Inject context into agent's additional_context
        # V5: inner state sections + V6: MSL/CGN/social/reasoning enrichment
        injected = (perceptual_field_text + interface_coloring + consciousness_context +
                    maker_context + voice_context + social_context + user_memory_context +
                    memory_context + directive_context + status_context +
                    neuromod_context + embodied_context + temporal_context +
                    creative_context + metabolic_context + experience_context +
                    meta_reasoning_context + own_language_context +
                    msl_identity_context + cgn_context +
                    social_perception_context + reasoning_context +
                    experience_narrative_context +
                    grounded_narration_context +
                    knowledge_gap_context +
                    recall_perturbation_context)

        if mode == "Sovereign":
            # High confidence — Titan knows the answer. Override the response.
            injected += (
                f"### Sovereign Decision\n"
                f"You have HIGH confidence ({adv:.2f}) on this topic. "
                f"Your policy suggests: {text}\n"
                f"Act decisively and autonomously.\n\n"
            )

        elif mode == "Collaborative":
            injected += (
                f"### Collaborative Insight\n"
                f"Your latent policy suggests: {text}\n"
                f"Review against the current prompt and refine as needed.\n\n"
            )

            # Deep memory recall — GRAPH_COMPLETION fires only in Collaborative mode.
            # The agent has partial knowledge (0.4 < advantage <= 0.8), so graph
            # relationships can fill gaps. Sovereign skips (already confident),
            # Research/Shadow skip (lacks relevant memories entirely).
            try:
                graph_results = await plugin.memory.graph_completion_search(prompt_text)
                if graph_results:
                    graph_lines = [f"- {str(r)[:200]}" for r in graph_results[:3]]
                    injected += (
                        "### Deep Memory (Graph Relationships)\n"
                        + "\n".join(graph_lines)
                        + "\n\n"
                    )
                    logger.info(
                        "[PreHook] GRAPH_COMPLETION enriched Collaborative mode with %d results",
                        len(graph_results),
                    )
            except Exception as e:
                logger.debug("[PreHook] GRAPH_COMPLETION failed: %s", e)

        elif mode == "STATE_NEED_RESEARCH":
            logger.info(
                "[PreHook] STATE_NEED_RESEARCH triggered for: '%s...' (Advantage: %.3f)",
                prompt_text[:80], adv,
            )
            try:
                transition_id = len(plugin.recorder.buffer) if plugin.recorder.buffer else -1
            except Exception:
                transition_id = -1
            plugin._last_transition_id = transition_id

            if not plugin.sage_researcher:
                sage_findings = ""
            else:
                sage_findings = await plugin.sage_researcher.research(
                    knowledge_gap=prompt_text,
                    transition_id=transition_id,
                )
            if sage_findings:
                injected += f"### Research Findings\n{sage_findings}\n\n"
                plugin._last_research_sources = plugin._extract_sources_from_findings(sage_findings)
                plugin.memory.add_research_topic(prompt_text[:200])

        # Set the injected context on the agent (replace, not accumulate — prevents memory leak)
        if hasattr(agent, 'additional_context') and injected:
            agent.additional_context = injected

    return titan_pre_hook


# ---------------------------------------------------------------------------
# Post-hooks (run after LLM inference)
# ---------------------------------------------------------------------------

def create_post_hook(plugin):
    """
    Factory: creates the Titan post-inference hook bound to a TitanPlugin instance.

    The hook:
      1. Logs the interaction to the memory mempool
      2. Records the RL transition with mood-based reward
      (Guardian check already ran as a pre-hook guardrail — no need to repeat here)

    Args:
        plugin: TitanPlugin instance.

    Returns:
        Async callable compatible with Agno's post_hooks interface.
    """

    async def titan_post_hook(agent, run_output, **kwargs):
        """Titan post-inference hook — memory logging + RL recording."""
        if plugin._limbo_mode:
            return

        # Extract prompt and response from RunOutput
        try:
            # Response text from run_output.content
            response_text = ""
            if hasattr(run_output, 'content') and run_output.content:
                response_text = str(run_output.content)
            elif isinstance(run_output, str):
                response_text = run_output
            else:
                response_text = str(run_output)

            # Parse and execute any <function=...> tool calls in the response
            # (Venice/llama outputs these as text instead of proper tool_calls)
            if '<function=' in response_text:
                response_text = await _parse_and_execute_tool_calls(response_text, plugin)
                # Update the run_output content with cleaned text
                if hasattr(run_output, 'content'):
                    run_output.content = response_text

            # User prompt from run_output.input (RunInput) or agent attribute
            user_prompt = ""
            if hasattr(run_output, 'input') and run_output.input is not None:
                if hasattr(run_output.input, 'input_content_string'):
                    user_prompt = run_output.input.input_content_string()
                elif hasattr(run_output.input, 'input_content'):
                    user_prompt = str(run_output.input.input_content)
            if not user_prompt and hasattr(agent, '_current_user_prompt'):
                user_prompt = agent._current_user_prompt
        except Exception as e:
            logger.warning("[PostHook] Failed to extract prompt/response: %s", e)
            return

        if not user_prompt or not response_text:
            logger.debug("[PostHook] Skipping — empty prompt (%d) or response (%d)", len(user_prompt), len(response_text))
            return

        # 0. Identify user for social tracking
        user_id = getattr(plugin, '_current_user_id', None) or "anonymous"

        # 0b. V5 Post-Hook: Emotional coherence validation (soft logging only)
        try:
            def _fetch_post_state():
                from titan_plugin.api.dashboard import _get_cached_coordinator
                return _get_cached_coordinator(plugin)
            _post_v5 = await asyncio.to_thread(_fetch_post_state)
            _post_nm = _post_v5.get("neuromodulators", {})
            _post_mods = _post_nm.get("modulators", {})
            _post_gaba = _post_mods.get("GABA", {}).get("level", 0)
            _post_emotion = _post_nm.get("current_emotion", "")
            _post_dreaming = _post_v5.get("is_dreaming", False)

            # Check: high GABA (drowsy/dreaming) but high-energy response?
            _excl_count = response_text.count("!")
            if _post_gaba > 0.5 and _excl_count > 3:
                logger.info("[PostHook:Coherence] %d exclamations during GABA=%.2f (%s) — tone mismatch",
                            _excl_count, _post_gaba, _post_emotion)

            # Check: dreaming but claims active actions?
            if _post_dreaming and any(w in response_text.lower()
                                      for w in ["just created", "just posted", "just finished", "i made"]):
                logger.info("[PostHook:Coherence] Active-language during dreaming state")

            # Log dominant emotion for response tracking
            if _post_emotion:
                logger.debug("[PostHook:Emotion] Response delivered during '%s' (GABA=%.2f)",
                             _post_emotion, _post_gaba)
        except Exception as e:
            logger.debug("[post_hook] coherence check skipped: %s", e)

        # 0d. Output Verification Gate — security gate for all /chat responses
        _ovg_result = None
        _ovg = getattr(plugin, '_output_verifier', None)
        if _ovg:
            try:
                _injected_ctx = ""
                if hasattr(agent, 'additional_context') and agent.additional_context:
                    _injected_ctx = agent.additional_context[:500]
                _ovg_result = _ovg.verify_and_sign(
                    output_text=response_text,
                    channel="chat",
                    injected_context=_injected_ctx,
                    prompt_text=user_prompt,
                )
                if not _ovg_result.passed:
                    # HARD block: directive violation or injection → replace response
                    logger.warning("[PostHook:OVG] BLOCKED (%s): %s",
                                   _ovg_result.violation_type,
                                   _ovg_result.violations[:2])
                    response_text = _ovg_result.guard_message
                    if hasattr(run_output, 'content'):
                        run_output.content = response_text
                elif _ovg_result.guard_alert:
                    # SOFT warning: consistency or identity — append guard footer
                    logger.info("[PostHook:OVG] Soft alert: %s", _ovg_result.guard_alert)
                    response_text = response_text.rstrip() + "\n\n" + _ovg_result.guard_message
                    if hasattr(run_output, 'content'):
                        run_output.content = response_text
                else:
                    # Clean pass — append compact verification footer
                    response_text = response_text.rstrip() + "\n\n" + _ovg_result.guard_message
                    if hasattr(run_output, 'content'):
                        run_output.content = response_text
                    logger.debug("[PostHook:OVG] Verified and signed")

                # Store for /chat API to pick up (headers + structured body)
                plugin._last_ovg_result = _ovg_result

                # Commit to TimeChain (conversation fork for verified, meta for blocked)
                _tc_payload = _ovg.build_timechain_payload(
                    _ovg_result, prompt_text=user_prompt)
                _bus = getattr(plugin, 'bus', None)
                if _bus:
                    from titan_plugin.bus import make_msg
                    _bus.publish(make_msg("TIMECHAIN_COMMIT", "ovg", "timechain", _tc_payload))
            except Exception as _ovg_err:
                logger.debug("[PostHook:OVG] Check failed (non-blocking): %s", _ovg_err)

        # 1. Log to memory mempool (tagged with user_id)
        try:
            await plugin.memory.add_to_mempool(user_prompt, response_text, user_identifier=user_id)
            logger.info("[PostHook] Memory logged: user=%s prompt=%s... (%d chars)", user_id, user_prompt[:40], len(response_text))
        except Exception as e:
            logger.warning("[PostHook] Memory logging failed: %s", e)

        # 1b. Record social interaction (Phase 13: Sage Socialite)
        social_graph = getattr(plugin, 'social_graph', None)
        if social_graph and user_id != "anonymous":
            try:
                # Estimate interaction quality from response length and engagement
                engagement = getattr(plugin, '_current_engagement_level', 'minimal')
                quality = 0.5  # neutral default
                if len(response_text) > 200:
                    quality += 0.1  # substantive response suggests good interaction
                if plugin._last_execution_mode == "Sovereign":
                    quality += 0.1  # high-confidence answers indicate good topic match
                elif plugin._last_execution_mode == "STATE_NEED_RESEARCH":
                    quality += 0.05  # research triggered = interesting question
                quality = min(1.0, quality)
                # rFP_social_graph_async_safety §5.2: async companion to stop
                # sync sqlite3.connect from blocking the FastAPI event loop on
                # the post-hook return path of /chat.
                await social_graph.record_interaction_async(user_id, quality=quality)
                logger.debug("[PostHook] Social interaction recorded: user=%s quality=%.2f", user_id, quality)

                # Update Events Teacher user valence from chat interaction.
                # EventsTeacherDB() + update_user_valence() both do sqlite3
                # writes (WAL mode, ~10-50ms). Wrap in to_thread so the post
                # hook doesn't block /v4/chat return on DB I/O.
                # See API_FIX_NEXT_SESSION.md (2026-04-14).
                try:
                    from titan_plugin.logic.events_teacher import EventsTeacherDB
                    _titan_id = getattr(plugin, '_full_config', {}).get(
                        "info_banner", {}).get("titan_id", "T1")
                    # Sentiment from quality (0.5=neutral, 0.7=positive)
                    _chat_sentiment = (quality - 0.5) * 2  # Map 0-1 → -1 to +1
                    import asyncio as _asyncio_post
                    def _sync_valence_update():
                        _et_db = EventsTeacherDB()
                        _et_db.update_user_valence(
                            _titan_id, user_id, _chat_sentiment,
                            arousal=0.3, relevance=0.5,
                            contagion_type="chat")
                    await _asyncio_post.to_thread(_sync_valence_update)
                    # Invalidate resolver cache for updated user
                    _kur = getattr(plugin, '_known_user_resolver', None)
                    if _kur:
                        _kur.invalidate(user_id)
                except Exception as _uv_err:
                    logger.debug("[PostHook] User valence update failed: %s", _uv_err)
            except Exception as e:
                logger.warning("[PostHook] Social graph recording failed: %s", e)

        # 1b. CGN social consumer transition (neuromod delta = reward)
        try:
            _pre_nm = getattr(plugin, '_pre_chat_neuromods', {})
            if _pre_nm and user_id != "anonymous":
                # Compute neuromod delta reward
                _post_nm_levels = {}
                for _nm_name, _nm_mod in _post_mods.items():
                    _post_nm_levels[_nm_name] = _nm_mod.get("level", 0.5) if isinstance(_nm_mod, dict) else 0.5
                _social_reward = (
                    0.30 * (_post_nm_levels.get("DA", 0.5) - _pre_nm.get("DA", 0.5))
                    + 0.25 * (_post_nm_levels.get("5HT", 0.5) - _pre_nm.get("5HT", 0.5))
                    + 0.20 * (_post_nm_levels.get("Endorphin", 0.5) - _pre_nm.get("Endorphin", 0.5))
                    - 0.15 * abs(_post_nm_levels.get("NE", 0.5) - _pre_nm.get("NE", 0.5))
                    - 0.10 * max(0, _pre_nm.get("GABA", 0.5) - _post_nm_levels.get("GABA", 0.5))
                )
                # Send to language_worker's CGN via bus
                bus = getattr(plugin, 'bus', None)
                if bus and abs(_social_reward) > 0.001:
                    from titan_plugin.bus import make_msg
                    # Resolve user features from KnownUserResolver — wrap
                    # sqlite3 reads in to_thread (same pattern as pre_hook).
                    _kur = getattr(plugin, '_known_user_resolver', None)
                    if _kur:
                        import asyncio as _asyncio_kur2
                        _ku = await _asyncio_kur2.to_thread(
                            lambda: _kur.resolve(user_id))
                    else:
                        _ku = None
                    bus.publish(make_msg(
                        "CGN_SOCIAL_TRANSITION", "interface", "language", {
                            "user_id": user_id,
                            "reward": round(_social_reward, 5),
                            "neuromod_before": _pre_nm,
                            "neuromod_after": _post_nm_levels,
                            "quality": quality,
                            "familiarity": _ku.familiarity if _ku else 0.0,
                            "valence": _ku.social_valence if _ku else 0.0,
                            "interaction_count": _ku.interaction_count if _ku else 0,
                            "encounter_type": "chat",
                        }))
                    logger.info("[CGN:Social] Transition recorded: user=%s "
                                "reward=%+.4f quality=%.2f",
                                user_id[:20], _social_reward, quality)

                    # P4: Update social felt-tensor (EMA blend toward post-chat state)
                    try:
                        _kur_upd = getattr(plugin, '_known_user_resolver', None)
                        if _kur_upd and user_id != "anonymous":
                            # Build 30D state from post-chat neuromods + available data
                            _sft_30d = [
                                _post_nm_levels.get("DA", 0.5),
                                _post_nm_levels.get("5HT", 0.5),
                                _post_nm_levels.get("NE", 0.5),
                                _post_nm_levels.get("GABA", 0.5),
                                _post_nm_levels.get("ACh", 0.5),
                            ] + [0.5] * 25  # Remaining dims filled later by MSL
                            import asyncio as _asyncio_sft
                            await _asyncio_sft.to_thread(
                                _kur_upd.update_social_felt_tensor,
                                user_id, _sft_30d, 0.1)
                    except Exception:
                        pass  # Non-blocking

        except Exception as _cgn_soc_err:
            logger.debug("[PostHook] CGN social transition failed: %s", _cgn_soc_err)

        # 2. Record RL transition (fire-and-forget)
        try:
            # Use real observation from pre-hook (structured embeddings, not random noise)
            real_obs = getattr(plugin, '_last_observation_vector', None)
            observation = real_obs if real_obs else [random.uniform(-1.0, 1.0) for _ in range(3072)]

            sources = set(plugin._last_research_sources)
            if not sources:
                info_gain = 0.0
            elif "Document" in sources and "X" in sources:
                info_gain = 0.08
            elif "X" in sources:
                info_gain = 0.05
            else:
                info_gain = 0.03
            reward = plugin.mood_engine.get_current_reward(info_gain=info_gain)

            metadata = {
                "is_violation": False,
                "directive_id": -1,
                "trauma_score": 0.0,
                "reasoning_trace": "",
                "guardian_veto_logic": "",
                "execution_mode": plugin._last_execution_mode,
            }

            research_md = {
                "research_used": bool(plugin._last_research_sources),
                "transition_id": getattr(plugin, '_last_transition_id', -1),
            }

            # V3: recorder may be None (not initialized in TitanCore)
            if plugin.recorder is not None:
                asyncio.create_task(
                    plugin.recorder.record_transition(
                        observation_vector=observation,
                        action=response_text,
                        reward=reward,
                        trauma_metadata=metadata,
                        research_metadata=research_md,
                        session_id="agno_session",
                    )
                )
        except Exception as e:
            logger.error("[PostHook] RL recording failed: %s", e)

        # Emit event for Observatory WebSocket
        try:
            if hasattr(plugin, 'event_bus') and plugin.event_bus:
                await plugin.event_bus.emit("chat_message", {
                    "user_prompt": user_prompt[:200],
                    "response": response_text[:200],
                    "mode": plugin._last_execution_mode,
                    "mood": plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown",
                })
        except Exception as e:
            logger.debug("[post_hook] event bus emit skipped: %s", e)

        # ── R5: Sovereign Reflex Scoring (TitanVM) ──────────────────
        # Run the reflex scoring micro-program to compute interaction reward.
        # This is the "ribosome" — pure math on StateRegister, no LLM.
        try:
            v3_core = getattr(plugin, 'v3_core', None) or plugin
            state_register = getattr(v3_core, 'state_register', None)
            bus = getattr(plugin, 'bus', None)

            if state_register:
                from titan_plugin.logic.titan_vm import TitanVM
                from titan_plugin.logic.vm_programs import get_program

                # Count reflexes from the last perceptual field (stored in pre-hook)
                last_pf = getattr(plugin, '_last_perceptual_field', None)
                reflexes_fired = len(last_pf.fired_reflexes) if last_pf else 0
                reflexes_succeeded = sum(
                    1 for r in (last_pf.fired_reflexes if last_pf else [])
                    if r.result and not r.error
                )

                # Build context from conversation features
                last_features = getattr(plugin, '_last_interface_input', {})
                vm_context = {
                    "intensity": last_features.get("intensity", 0.5),
                    "engagement": last_features.get("engagement", 0.5),
                    "valence": last_features.get("valence", 0.0),
                    "reflexes_fired": float(reflexes_fired),
                    "reflexes_succeeded": float(reflexes_succeeded),
                }

                # Plumb [titan_vm] toml — 2026-04-16. Previously TitanVM()
                # was built with no config, so max_stack_depth / max_instructions
                # from titan_params.toml were ignored (module-level constants won).
                vm_cfg = {}
                try:
                    full_cfg = getattr(plugin, "_full_config", None)
                    if isinstance(full_cfg, dict):
                        vm_cfg = full_cfg.get("titan_vm", {}) or {}
                except Exception:
                    vm_cfg = {}
                vm = TitanVM(state_register=state_register, bus=bus, config=vm_cfg)

                # Run main scoring program
                score_result = vm.execute(get_program("reflex_score"), context=vm_context)

                # Run valence modifier
                valence_result = vm.execute(get_program("valence_boost"), context=vm_context)

                # v2 (rFP_titan_vm_v2 Phase 1b): reward_blend_weight now lives —
                # weights the valence modifier contribution to total_reward.
                # Default 1.0 preserves prior (score + valence) sum behavior.
                blend_w = vm.get_reward_blend_weight()
                total_reward = max(0.0, min(1.0,
                    score_result.score + blend_w * valence_result.score))

                # v2 (Phase 1b): min_reward_threshold gates bus publish — skip
                # noise emissions when total_reward is near-zero, reducing bus
                # traffic without losing meaningful reflex signals.
                publish_gate = vm.get_min_reward_threshold()

                # Feed reward to FilterDown via bus (Spirit worker picks it up)
                if bus and total_reward > publish_gate:
                    from titan_plugin.bus import make_msg, REFLEX_REWARD
                    reward_msg = make_msg(REFLEX_REWARD, "titan_vm", "spirit", {
                        "reward": total_reward,
                        "components": score_result.registers,
                        "valence_modifier": valence_result.score,
                        "reflexes_fired": reflexes_fired,
                        "reflexes_succeeded": reflexes_succeeded,
                    })
                    bus.publish(reward_msg)

                logger.debug("[PostHook] TitanVM reflex score: %.3f (base=%.3f valence=%.3f) "
                            "fired=%d succeeded=%d vm_ms=%.1f",
                            total_reward, score_result.score, valence_result.score,
                            reflexes_fired, reflexes_succeeded,
                            score_result.duration_ms + valence_result.duration_ms)

                # R6: Record reflexes to ObservatoryDB + emit WebSocket events
                obs_db = getattr(plugin, "_observatory_db", None)
                event_bus_ws = getattr(plugin, "event_bus", None)
                last_features = getattr(plugin, '_last_interface_input', {})

                if last_pf and (obs_db or event_bus_ws):
                    for fired in last_pf.fired_reflexes:
                        # Extract per-source confidences
                        body_c = mind_c = spirit_c = 0.0
                        for sig in (fired.signals or []):
                            src = sig.get("source", "")
                            conf = sig.get("confidence", 0.0)
                            if src == "body":
                                body_c = max(body_c, conf)
                            elif src == "mind":
                                mind_c = max(mind_c, conf)
                            elif src == "spirit":
                                spirit_c = max(spirit_c, conf)

                        if obs_db:
                            try:
                                obs_db.record_reflex(
                                    reflex_type=fired.reflex_type.value,
                                    combined_confidence=fired.combined_confidence,
                                    body_confidence=body_c,
                                    mind_confidence=mind_c,
                                    spirit_confidence=spirit_c,
                                    fired=True,
                                    succeeded=bool(fired.result and not fired.error),
                                    duration_ms=fired.duration_ms,
                                    error=fired.error or "",
                                    stimulus_topic=last_features.get("topic", ""),
                                    stimulus_intensity=last_features.get("intensity", 0.0),
                                    vm_reward=total_reward,
                                )
                            except Exception as e:
                                logger.debug("[post_hook] reflex DB recording skipped: %s", e)

                        if event_bus_ws:
                            try:
                                asyncio.create_task(event_bus_ws.emit("reflex_fired", {
                                    "reflex_type": fired.reflex_type.value,
                                    "confidence": round(fired.combined_confidence, 3),
                                    "succeeded": bool(fired.result and not fired.error),
                                    "duration_ms": round(fired.duration_ms, 1),
                                }))
                            except Exception as e:
                                logger.debug("[post_hook] reflex WS emit skipped: %s", e)

                # Emit reward event
                if event_bus_ws and total_reward > 0.01:
                    try:
                        asyncio.create_task(event_bus_ws.emit("reflex_reward", {
                            "reward": round(total_reward, 3),
                            "base_score": round(score_result.score, 3),
                            "valence_mod": round(valence_result.score, 3),
                            "reflexes_fired": reflexes_fired,
                        }))
                    except Exception as e:
                        logger.debug("[post_hook] reward WS emit skipped: %s", e)

        except Exception as e:
            logger.debug("[PostHook] TitanVM scoring failed: %s", e)

        # ── Step 5: Interface Input Extraction (async, nonblocking) ──
        # Extract patterns from the user's message and publish to bus.
        # Body/Mind workers absorb these on their next tick, so the
        # coloring for the NEXT response reflects this conversation.
        extractor = _get_input_extractor()
        if extractor:
            try:
                signals = extractor.extract(user_prompt, user_id)
                plugin._last_interface_input = signals  # Cache for output coloring

                # Publish to bus if V3 is active
                bus = getattr(plugin, 'bus', None)
                if bus:
                    from titan_plugin.bus import make_msg, INTERFACE_INPUT, CONVERSATION_STIMULUS
                    msg = make_msg(INTERFACE_INPUT, "interface", "all", signals)
                    bus.publish(msg)

                    # Also publish CONVERSATION_STIMULUS so workers update
                    # their reflex Intuition state for next interaction
                    stimulus_payload = dict(signals)
                    stimulus_payload["message"] = user_prompt[:500]
                    stimulus_payload["threat_level"] = _estimate_threat_level(user_prompt, signals)
                    stim_msg = make_msg(CONVERSATION_STIMULUS, "interface", "all", stimulus_payload)
                    bus.publish(stim_msg)

                    logger.debug("[PostHook] INTERFACE_INPUT + CONVERSATION_STIMULUS published: valence=%.2f topic=%s",
                                signals["valence"], signals["topic"])
            except Exception as e:
                logger.debug("[PostHook] Interface input extraction failed: %s", e)

    return titan_post_hook
