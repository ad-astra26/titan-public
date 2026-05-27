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


def create_tools(plugin):
    """
    Factory: creates a list of Agno-compatible tool functions bound to a TitanHCL.

    Each tool is an async function with a descriptive docstring (Agno uses these
    as the tool's schema description for the LLM).

    Phase 6: if ``plugin.synthesis_tool_plugs`` is set (synthesis_worker has
    constructed the ToolPlug registry), the tools below route invocations
    through ``ToolPlug.invoke()`` so every call lands a procedural-fork TX
    (INV-12). Falls back to the legacy direct-subsystem-call path when the
    ToolPlug registry is not present (e.g. early-boot or test harness).

    Args:
        plugin: Initialized TitanHCL instance.

    Returns:
        List of async callables suitable for Agno Agent(tools=[...]).
    """
    # Pull the ToolPlug registry off the plugin if present (synthesis_worker
    # wires this at boot during P6.I integration). Each plug is keyed by
    # tool_id matching the agno tool name.
    tool_plugs = getattr(plugin, "synthesis_tool_plugs", {}) or {}

    def _invoke_tool_plug_sync(tool_id: str, args: dict, parent_chat_tx=None):
        """Run a ToolPlug.invoke() in a worker thread (the plugs are sync
        per protocol; agno's tool callables are async — we bridge so the
        FastAPI loop never blocks)."""
        from titan_hcl.synthesis.plugs import ToolCall
        plug = tool_plugs.get(tool_id)
        if plug is None:
            return None
        call = ToolCall(tool_id=tool_id, args=args, parent_chat_tx=parent_chat_tx)
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
                # Producers that need the source extraction still call
                # plugin._extract_sources_from_findings on the returned
                # string.
                return result.result_summary or "No findings — knowledge tool returned empty."

        # Legacy fallback (pre-P6 path; same shape as before).
        try:
            transition_id = len(plugin.recorder.buffer) if plugin.recorder.buffer else -1
        except Exception:
            transition_id = -1

        if not plugin.sage_researcher:
            findings = ""
        else:
            findings = await plugin.sage_researcher.research(
                knowledge_gap=query,
                transition_id=transition_id,
            )
        if findings:
            plugin._last_research_sources = plugin._extract_sources_from_findings(findings)
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

    return [
        coding_sandbox,    # P6.I — closes arch §11.3 gap (sandbox now invoked by outer self)
        research,          # P6.I — routes through KnowledgeTool when wired
        events_teacher,    # P6.I — distill/fetch X events
        x_research,        # P6.I — post + fetch via SocialXGateway
        generate_art,      # existing (out of P6 scope — studio tool)
        generate_audio,    # existing (out of P6 scope — studio tool)
    ]
