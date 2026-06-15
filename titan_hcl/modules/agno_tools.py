"""
Agno Tool wrappers for Titan subsystems.

Each function is a plain async callable that Agno can register as a tool.
They delegate to the bound TitanHCL instance — no subsystem logic lives here.

**Phase 6 (D-SPEC-PHASE6, 2026-05-27)** — INV-12 single action surface:
the outer-self LLM invokes tools through this layer, every invocation
routes through a `ToolPlug` (titan_hcl/synthesis/tools/) which emits a
procedural-fork TX (arch §8.2) for audit + skill-compilation. The
``coding_sandbox`` agno tool is wired in here for the first time —
closes the arch §11.3 "currently unused by the outer self" gap.

Tools that double as oracles (sandbox, X) expose both surfaces via the
shared underlying helper instance; the procedural TX carries
``scored_by="oracle"`` once the companion verdict (OracleRouter)
returns a non-`unknown` verdict for the call.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


def _set_eel_research_provenance(plugin, findings: str) -> None:
    """EEL-A1 (INV-EEL-6): mark this turn's research provenance so the PostHook
    persist tags the node `acquired:research` → opens the confirm/dispute window
    → feeds the self-learning confirm→seed loop.

    The PreHook STATE_NEED_RESEARCH path already sets this; the agno TOOL research
    path (this file) + the reflex path did NOT — so research-via-tool produced no
    pending-confirmation node and the self-learning seed never fired even after a
    solid research turn (root-caused 2026-06-15). Per-research-event (no
    staleness); the PostHook read-and-clears `_acquired_research_source` per turn.
    Soft — never break the turn."""
    if not findings:
        return
    try:
        srcs = plugin._extract_sources_from_findings(findings)
        plugin._last_research_sources = srcs
        plugin._acquired_research_source = (
            "; ".join(str(s) for s in srcs)
            if isinstance(srcs, (list, tuple)) and srcs else "research")
    except Exception:  # noqa: BLE001 — provenance best-effort, never break chat
        plugin._acquired_research_source = "research"


def create_tools(plugin):
    """
    Factory: creates a list of Agno-compatible tool functions bound to a TitanHCL.

    Each tool is an async function with a descriptive docstring (Agno uses these
    as the tool's schema description for the LLM).

    Phase 6 (+ amendment, arch §11.3 / SPEC §25.5): if
    ``plugin.synthesis_tool_plugs`` is set, the tools below route invocations
    through ``ToolPlug.invoke()`` so every call lands a procedural-fork TX
    (INV-12). The registry is built in the process that INVOKES the plug —
    chat-time tools (coding_sandbox) are constructed in agno_worker
    (``_build_local_tool_plugs``); dream/autonomous tools in synthesis_worker.
    Falls back to the legacy direct-subsystem-call path when the ToolPlug
    registry is not present (e.g. early-boot or test harness).

    Args:
        plugin: Initialized TitanHCL instance.

    Returns:
        List of async callables suitable for Agno Agent(tools=[...]).
    """
    # Pull the ToolPlug registry off the plugin if present. For agno-hosted
    # chat tools this is wired locally at boot by agno_worker
    # (_build_local_tool_plugs) — a worker cannot populate another process's
    # plugin attr (Phase 6 amendment). Each plug is keyed by tool_id matching
    # the agno tool name.
    tool_plugs = getattr(plugin, "synthesis_tool_plugs", {}) or {}

    def _invoke_tool_plug_sync(tool_id: str, args: dict, parent_chat_tx=None):
        """Run a ToolPlug.invoke() in a worker thread (the plugs are sync
        per protocol; agno's tool callables are async — we bridge so the
        FastAPI loop never blocks)."""
        from titan_hcl.synthesis.plugs import ToolCall
        plug = tool_plugs.get(tool_id)
        if plug is None:
            return None
        # EEL B1 (D-SPEC-153 / INV-Syn-29) — source THIS turn's GOAL so the
        # oracle-verified tool-use forms an OUTCOME-keyed (oracle_id, goal_class)
        # skill. The pre-LLM goal hook (INV-Syn-17) wrote the user message into the
        # `goal` working-memory buffer BEFORE agent.arun; thread it as parent_goal.
        # WITHOUT it the verdict carries parent_goal=None and the OracleRouter flush
        # DROPS it (`if not e.parent_goal: continue`, oracle_router.py) → no positive
        # skill ever forms (the 2026-06-09 soak found 0 promoted despite oracle
        # coverage=1.0). Best-effort + soft: no goal buffer → parent_goal=None
        # (unchanged behaviour, no regression).
        parent_goal = None
        try:
            if _buffer_cache is not None:
                _goal_row = _buffer_cache.get(_resolve_chat_id(), "goal")
                if _goal_row:
                    parent_goal = (_goal_row.get("content") or "").strip() or None
        except Exception:
            parent_goal = None
        call = ToolCall(tool_id=tool_id, args=args, parent_chat_tx=parent_chat_tx,
                        parent_goal=parent_goal)
        return plug.invoke(call)

    # ------------------------------------------------------------------
    # Coding Sandbox Tool (P6.I — closes arch §11.3 gap)
    # ------------------------------------------------------------------
    async def coding_sandbox(code: str, expected_stdout: str = "", assertion: str = "") -> str:
        """
        Execute Python code in an AST-validated subprocess sandbox.

        Use this when you need to compute something deterministically
        (math, data processing, algorithm verification) or check a code
        snippet's correctness. The sandbox blocks dangerous imports
        (os, sys, subprocess, network) and caps runtime at 30s + memory
        at 256MB.

        Optional verification args turn the call into a verifiable
        claim (the verdict is anchored on-chain):
          expected_stdout: if set, the sandbox compares stdout against
            this string; the result reports match/mismatch.
          assertion: if set, the sandbox evaluates `assert (assertion)`
            after the code runs; true/false anchors a verdict.

        Args:
            code: Python source to execute.
            expected_stdout: Optional expected output string.
            assertion: Optional Python boolean expression to assert.

        Returns:
            String summary: success status + stdout (or error).
        """
        args = {"code": code}
        if expected_stdout:
            args["expected_stdout"] = expected_stdout
        if assertion:
            args["assertion"] = assertion
        result = await asyncio.to_thread(_invoke_tool_plug_sync, "coding_sandbox", args)
        if result is None:
            return "coding_sandbox tool not wired (synthesis_worker initializing — try again shortly)"
        if result.success:
            return f"OK: {result.result_summary}"
        if result.exception:
            return f"FAILED ({result.exception}): {result.result_summary}"
        return f"FAILED: {result.result_summary}"

    # ------------------------------------------------------------------
    # Research Tool (P6.I — now routed through KnowledgeTool when wired)
    # ------------------------------------------------------------------
    async def research(query: str) -> str:
        """
        Trigger the Stealth-Sage autonomous research pipeline.
        Searches the web via SearXNG, scrapes top results with Crawl4AI,
        checks X/Twitter sentiment if relevant, and distills findings
        through local Ollama inference. Use this when you need factual
        information beyond your training data or the user's memory graph.

        Args:
            query: The knowledge gap or question to research.

        Returns:
            Formatted research findings block, or empty string if nothing found.
        """
        # Phase 6: if KnowledgeTool is wired, route through it so the
        # invocation lands a procedural-fork TX (INV-12); fall back to
        # the legacy direct sage_researcher path when not yet wired.
        if "knowledge" in tool_plugs:
            result = await asyncio.to_thread(
                _invoke_tool_plug_sync, "knowledge",
                {"query": query, "mode": "web_search"},
            )
            if result is not None and result.success:
                # The KnowledgeTool's invoke_fn returns findings as
                # result_full_payload (the full text) and a brief
                # result_summary; agno expects the rich text back.
                _p6_findings = getattr(result, "result_full_payload", "") \
                    or result.result_summary or ""
                # EEL-A1: tool research must set provenance too (→ acquired:research
                # node → confirm window → self-learning seed).
                _set_eel_research_provenance(plugin, _p6_findings)
                return result.result_summary or "No findings — knowledge tool returned empty."

        # Legacy fallback (pre-P6 path; same shape as before).
        if not plugin.sage_researcher:
            findings = ""
        else:
            # research() dropped its `transition_id` arg — passing it raised
            # TypeError (latent until sage_researcher stopped being None on the
            # agno_worker, 2026-06-15). Call with the current signature.
            findings = await plugin.sage_researcher.research(
                knowledge_gap=query,
            )
        if findings:
            _set_eel_research_provenance(plugin, findings)
            plugin.memory.add_research_topic(query[:200])
        return findings or "No findings — research pipeline returned empty."

    # NOTE: recall_memory / check_metabolism / post_to_x tools were REMOVED
    # in the Sovereign Reflex Arc migration (R4). The LLM now receives this
    # information as [INNER STATE] via Trinity Intuition reflexes rather than
    # calling tools. post_to_x was removed because all posting routes through
    # SocialPressureMeter (social_narrator + quality gate + rate limits).
    # Kept only action/creative tools below (generate_art, generate_audio,
    # research).

    # ------------------------------------------------------------------
    # Art Generation Tool
    # ------------------------------------------------------------------
    async def generate_art(seed_text: str) -> str:
        """
        Generate procedural artwork reflecting Titan's current cognitive state.
        Creates a flow field visualization seeded by the given text, rendered
        as a PNG image. The art is purely mathematical — no external AI APIs.

        Args:
            seed_text: Text to seed the procedural generation (hashed for parameters).

        Returns:
            Path to the generated artwork file, or error message.
        """
        try:
            import hashlib
            state_root = hashlib.sha256(seed_text.encode()).hexdigest()

            # Use studio coordinator (now via StudioProxy + D-SPEC-46 event-driven
            # render under v1.8.3 / D-SPEC-57). Same return-value contract:
            # extract art_path from the completion envelope's "paths" dict.
            result = await plugin.studio.generate_meditation_art_with_completion(
                state_root=state_root,
                age_nodes=50,
                avg_intensity=128,
            )
            art_path = result.get("paths", {}).get("art_path") if result else None
            if art_path:
                return f"Art generated: {art_path}"
            error = result.get("error") if result else "no response"
            return f"Art generation completed but no output file was produced ({error or 'empty paths'})."
        except Exception as e:
            logger.warning("[Tool:generate_art] Failed: %s", e)
            return f"Art generation failed: {e}"

    # ------------------------------------------------------------------
    # Audio Sonification Tool
    # ------------------------------------------------------------------
    async def generate_audio(tx_signature: str = "", sol_balance: float = 0.0) -> str:
        """
        Generate blockchain sonification — translates Solana transaction data
        into a pentatonic WAV chime using pure mathematical synthesis.
        No external audio APIs or dependencies.

        Args:
            tx_signature: Solana transaction signature to sonify (uses random if empty).
            sol_balance: Current SOL balance for tonal modulation.

        Returns:
            Path to the generated WAV file, or error message.
        """
        try:
            if not tx_signature:
                import hashlib, time
                tx_signature = hashlib.sha256(str(time.time()).encode()).hexdigest()
            if sol_balance <= 0:
                sol_balance = plugin.metabolism._last_balance or 1.0

            # v1.8.3 / D-SPEC-57: StudioProxy._with_completion variant returns
            # completion envelope; bundle paths live under envelope["paths"].
            envelope = await plugin.studio.generate_epoch_bundle_with_completion(
                tx_signature=tx_signature,
                total_nodes=50,
                beliefs_strength=128,
                sol_balance=sol_balance,
            )
            result = envelope.get("paths", {}) if envelope else {}
            if result and result.get("audio_path"):
                return f"Audio generated: {result['audio_path']}"
            error = envelope.get("error") if envelope else "no response"
            return f"Audio generation completed but no output file was produced ({error or 'empty paths'})."
        except Exception as e:
            logger.warning("[Tool:generate_audio] Failed: %s", e)
            return f"Audio generation failed: {e}"

    # check_identity tool was also removed in the R4 migration — sovereign
    # identity is now surfaced via [INNER STATE] rather than a tool call.

    # ------------------------------------------------------------------
    # Events Teacher Tool (P6.I — distill/fetch X events through the
    # events_teacher pipeline; pure tool, no oracle surface)
    # ------------------------------------------------------------------
    async def events_teacher(action: str, **kwargs) -> str:
        """
        Invoke the events_teacher X-event distillation pipeline.

        Use this to (a) distill a specific X event into a concept-form
        memory the synthesis engine can spine, or (b) fetch recent X
        events Titan has seen for context.

        Args:
            action: "distill_event" or "fetch_recent_events".
            **kwargs: action-specific args (e.g. event_id for distill).

        Returns:
            String summary of the action result.
        """
        args = {"action": action, **kwargs}
        result = await asyncio.to_thread(_invoke_tool_plug_sync, "events_teacher", args)
        if result is None:
            return "events_teacher tool not wired"
        return result.result_summary or ("ok" if result.success else "failed")

    # ------------------------------------------------------------------
    # X Research Tool (P6.I — fetch + post via SocialXGateway; doubles
    # as oracle via x_oracle wrapper)
    # ------------------------------------------------------------------
    async def x_research(capability: str, **kwargs) -> str:
        """
        Active X mining via SocialXGateway (the sole sanctioned X path).

        Capabilities:
          - "post": post a text message to X (text=<str>).
          - "fetch_thread": fetch a thread (thread_root_id=<str>).
          - "fetch_topic": search recent posts by topic (topic=<str>).
          - "fetch_account": fetch recent posts by handle (handle=<str>).

        Use this for current-events research, posting Titan's own
        responses, or social-truth verification (which doubles through
        the x_oracle verification surface).

        Args:
            capability: one of {"post", "fetch_thread", "fetch_topic",
                                "fetch_account"}.
            **kwargs: capability-specific args.

        Returns:
            String summary of the X interaction.
        """
        args = {"capability": capability, **kwargs}
        result = await asyncio.to_thread(_invoke_tool_plug_sync, "x_research", args)
        if result is None:
            return "x_research tool not wired"
        return result.result_summary or ("ok" if result.success else "failed")

    # ------------------------------------------------------------------
    # ACT-R Working-Memory Buffer Tools (P7.D — D-SPEC-PHASE7 / INV-Syn-16)
    # ------------------------------------------------------------------
    # The 4 agno tools that give the outer-self LLM a typed read/write
    # surface over its working-memory buffers (goal / retrieval /
    # imaginal / perception). Backed by BufferCache (agno-side in-mem
    # cache with write-through SYNTHESIS_BUFFER_COMMAND bus emit).
    # synthesis_worker is the sole DuckDB writer per INV-Syn-16.

    _buffer_cache = getattr(plugin, "synthesis_buffer_cache", None)
    _VALID_BUFFERS = ("goal", "retrieval", "imaginal", "perception")

    def _resolve_chat_id() -> str:
        """f'{user_id}:{session_id}' per SPEC §25.6 chat_id convention.

        Reads worker_plugin's per-request caches populated by
        agno_worker._handle_chat_request before agent.arun fires."""
        uid = getattr(plugin, "_current_user_id", None) or "anonymous"
        # session_id isn't directly cached on plugin; use the per-tier
        # cache key the chat handler maintains. Best-effort fall-back
        # to 'default' preserves tool function under unit-test harnesses.
        try:
            sess = getattr(plugin, "_current_session_id", None) or "default"
        except Exception:
            sess = "default"
        return f"{uid}:{sess}"

    async def read_buffer(buffer_name: str) -> str:
        """
        Read the current value of an ACT-R working-memory buffer for this chat.

        Buffers are your structured scratchpad — context affects retrieval
        ONLY through them (ACT-R's central insight). Use this when you want
        to remember what your current goal is, what you most recently
        recalled, or what your scratchpad imagined.

        Args:
            buffer_name: One of {"goal", "retrieval", "imaginal", "perception"}.

        Returns:
            The buffer's text content (empty string if unset).
        """
        if _buffer_cache is None:
            return ""
        if buffer_name not in _VALID_BUFFERS:
            return (
                f"unknown buffer '{buffer_name}'; valid: "
                f"{', '.join(_VALID_BUFFERS)}"
            )
        chat_id = _resolve_chat_id()
        row = _buffer_cache.get(chat_id, buffer_name)
        if not row:
            return ""
        return row.get("content", "") or ""

    async def write_buffer(buffer_name: str, content: str) -> str:
        """
        Write content into an ACT-R working-memory buffer for this chat.

        Use this to update your goal (what you're trying to do), your
        scratchpad (imaginal — working draft, persists across turns), or
        what you remember about the conversation (retrieval). The
        perception buffer is system-managed (set automatically from each
        user message) — do not write to it directly.

        Args:
            buffer_name: One of {"goal", "retrieval", "imaginal"}.
            content: The text to store.

        Returns:
            "ok" on success; an explanatory string on validation failure.
        """
        if _buffer_cache is None:
            return "buffer cache not wired"
        if buffer_name == "perception":
            return "perception buffer is system-managed (set by pre-LLM hook)"
        if buffer_name not in _VALID_BUFFERS:
            return (
                f"unknown buffer '{buffer_name}'; writable: goal, "
                f"retrieval, imaginal"
            )
        chat_id = _resolve_chat_id()
        # Reuse the same lightweight grounding helper as the goal hook —
        # keeps concept_ids consistent across system + LLM writes.
        try:
            from titan_hcl.modules.agno_worker import _ground_for_goal_hook
            concept_ids = _ground_for_goal_hook(plugin, content or "")
        except Exception:
            concept_ids = []
        try:
            _buffer_cache.set(
                chat_id, buffer_name,
                content=content or "", concept_ids=concept_ids,
            )
        except Exception as e:
            logger.debug("write_buffer error: %s", e)
            return f"write failed: {e}"
        return "ok"

    async def clear_buffer(buffer_name: str) -> str:
        """
        Clear an ACT-R working-memory buffer for this chat.

        Use this when you finish a sub-task and want to drop the goal
        chunk, or want to start a fresh scratchpad.

        Args:
            buffer_name: One of {"goal", "retrieval", "imaginal", "perception"}.

        Returns:
            "ok" on success; an explanatory string on validation failure.
        """
        if _buffer_cache is None:
            return "buffer cache not wired"
        if buffer_name not in _VALID_BUFFERS:
            return (
                f"unknown buffer '{buffer_name}'; valid: "
                f"{', '.join(_VALID_BUFFERS)}"
            )
        chat_id = _resolve_chat_id()
        try:
            _buffer_cache.clear(chat_id, buffer_name)
        except Exception as e:
            logger.debug("clear_buffer error: %s", e)
            return f"clear failed: {e}"
        return "ok"

    async def query_retrieval(query: str, granularity: str = "concept") -> str:
        """
        Query outer memory and store the top result into your `retrieval` buffer.

        This is the bridge between explicit recall and your working memory —
        you ask for relevant past experiences, the top match lands in your
        retrieval buffer, and subsequent reasoning can use it.

        Args:
            query: Natural-language query (free text).
            granularity: One of {"concept", "turn", "topic", "session", "self"}.
                Default "concept" (best for semantic recall). Use "self" for
                self-reflective questions about your OWN path / identity /
                abilities ("what have I learned about myself", "who am I",
                "what can I do") — it traverses your SELF hub (your diary arc +
                skills) in one hop instead of scanning all of memory.

        Returns:
            String summary of the top result, or "no match" if nothing found.
        """
        if _buffer_cache is None:
            return "buffer cache not wired"
        recall = getattr(plugin, "engine_recall", None)
        if recall is None:
            return "engine_recall not wired (Phase 1+ required)"
        chat_id = _resolve_chat_id()
        try:
            results = await asyncio.to_thread(
                recall.recall, query, granularity=granularity, k=1,
            )
        except Exception as e:
            logger.debug("query_retrieval engine_recall failed: %s", e)
            return f"query failed: {e}"
        top = results[0] if results else None
        if top is None:
            return "no match"
        summary = (
            getattr(top, "summary", None)
            or getattr(top, "tx_hash", "")
            or "match"
        )
        # Materialize concept_ids from the result if surfaced; defensive
        # against multiple RecallResult shapes across phases.
        cids = (
            getattr(top, "concept_ids", None)
            or [getattr(top, "concept_id", "")]
        )
        cids = [c for c in (cids or []) if isinstance(c, str) and c]
        try:
            _buffer_cache.set(
                chat_id, "retrieval",
                content=summary, concept_ids=cids,
            )
        except Exception as e:
            logger.debug("query_retrieval buffer write failed: %s", e)
        # Phase 9 INV-Syn-23: record the surfaced item so the post-LLM
        # CitedUseDetector (agno_worker._handle_chat_request) can decide whether
        # the response actually cited it → emit MEMORY_RETRIEVAL_USED with the
        # correct used_by_llm flag. Keyed by chat_id; the hook pops + clears.
        try:
            _item_id = getattr(top, "tx_hash", "") or ""
            if _item_id:
                _reg = getattr(plugin, "_last_surfaced_items", None)
                if _reg is None:
                    _reg = {}
                    plugin._last_surfaced_items = _reg
                _reg.setdefault(chat_id, []).append({
                    "item_id": _item_id,
                    "title": summary[:120],
                    "content_snippet": summary[:512],
                    "concept_ids": cids,
                })
        except Exception as e:
            logger.debug("query_retrieval surfaced-item capture failed: %s", e)
        return summary

    # ------------------------------------------------------------------
    # Phase 8 — match_procedural_skill (P8.F / INV-Syn-20)
    # ------------------------------------------------------------------
    async def match_procedural_skill(goal_text: str) -> str:
        """
        Look up a compiled procedural skill matching your current goal.

        This is your shortcut from "I need to do X" to "I've done X before
        successfully Y times, here's the recipe." Returns either the
        top-matching skill's metadata (you decide whether to invoke its
        executable_spec parameterized by current context) or "no match"
        (continue with un-delegated reasoning).

        The skill is ONLY returned when it passes the delegate gate:
        utility_score above the soft-retire floor, match_score above the
        composite threshold, and verified_at IS NOT NULL (skill has
        passed first-invocation lineage re-verification — INV-Syn-20).

        Args:
            goal_text: Natural-language description of what you're trying
                to do (e.g. "deploy a Solana NFT", "build a cosmetic website").

        Returns:
            JSON-string summary of the top skill {skill_id, name,
            nl_description, executable_spec, match_score, utility_score,
            success_count, failure_count} when delegate gate passes;
            "no match" when nothing matches; "synthesis not wired" when
            EngineRecall isn't available (early-boot / test).
        """
        import json as _json
        recall = getattr(plugin, "engine_recall", None)
        if recall is None:
            return "synthesis not wired"
        try:
            results = await asyncio.to_thread(
                recall.recall, goal_text, granularity="procedural", k=1,
            )
        except Exception as e:
            logger.debug("match_procedural_skill recall failed: %s", e)
            return "no match"
        if not results:
            return "no match"
        top = results[0]
        # Synthesis reader already applied the delegate gate before returning;
        # we just re-check the cascade-flag here (operator's per-Titan toggle).
        # Default False = conservative dry-run when the attr is unset (agno
        # boot wires it from config; an unwired plugin must NOT auto-delegate).
        delegate_live = bool(
            getattr(plugin, "synthesis_delegate_live", False)
        )
        if not delegate_live:
            return "no match"
        # RecallResult shape — extract skill metadata fields. Fall back to
        # safe defaults when the upstream shape changes.
        skill_meta = {
            "skill_id": getattr(top, "tx_hash", ""),
            "name": getattr(top, "summary", ""),
            "match_score": float(getattr(top, "score", 0.0) or 0.0),
            "utility_score": float(getattr(top, "importance", 0.0) or 0.0),
        }
        # For richer fields (executable_spec, success/failure counts), look
        # up the full skill row through the synthesis store proxy if exposed.
        store = getattr(plugin, "skill_store", None)
        if store is not None and skill_meta["skill_id"]:
            try:
                full = await asyncio.to_thread(store.read_skill, skill_meta["skill_id"])
                if full is not None:
                    skill_meta["nl_description"] = full.get("nl_description", "")
                    skill_meta["executable_spec"] = full.get("executable_spec", {})
                    skill_meta["success_count"] = int(full.get("success_count") or 0)
                    skill_meta["failure_count"] = int(full.get("failure_count") or 0)
            except Exception as e:
                logger.debug("match_procedural_skill store lookup failed: %s", e)
        try:
            return _json.dumps(skill_meta, ensure_ascii=False)
        except Exception:
            return f"match: {skill_meta.get('skill_id')}"

    return [
        coding_sandbox,    # P6.I — closes arch §11.3 gap (sandbox now invoked by outer self)
        research,          # P6.I — routes through KnowledgeTool when wired
        events_teacher,    # P6.I — distill/fetch X events
        x_research,        # P6.I — post + fetch via SocialXGateway
        generate_art,      # existing (out of P6 scope — studio tool)
        generate_audio,    # existing (out of P6 scope — studio tool)
        read_buffer,       # P7.D — ACT-R buffer read (INV-Syn-16)
        write_buffer,      # P7.D — ACT-R buffer write (INV-Syn-16/17)
        clear_buffer,      # P7.D — ACT-R buffer clear (INV-Syn-16)
        query_retrieval,   # P7.D — recall + retrieval buffer populate
        match_procedural_skill,  # P8.F — INV-Syn-20 skill match (delegate-gated)
    ]
