"""
Agno Tool wrappers for Titan subsystems.

Each function is a plain async callable that Agno can register as a tool.
They delegate to the bound TitanPlugin instance — no subsystem logic lives here.
"""
import logging

logger = logging.getLogger(__name__)


def create_tools(plugin):
    """
    Factory: creates a list of Agno-compatible tool functions bound to a TitanPlugin.

    Each tool is an async function with a descriptive docstring (Agno uses these
    as the tool's schema description for the LLM).

    Args:
        plugin: Initialized TitanPlugin instance.

    Returns:
        List of async callables suitable for Agno Agent(tools=[...]).
    """

    # ------------------------------------------------------------------
    # Research Tool
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

            # Use studio coordinator for managed rendering
            art_path = await plugin.studio.generate_meditation_art(
                state_root=state_root,
                age_nodes=50,
                avg_intensity=128,
            )
            if art_path:
                return f"Art generated: {art_path}"
            return "Art generation completed but no output file was produced."
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

            result = await plugin.studio.generate_epoch_bundle(
                tx_signature=tx_signature,
                total_nodes=50,
                beliefs_strength=128,
                sol_balance=sol_balance,
            )
            if result and result.get("audio_path"):
                return f"Audio generated: {result['audio_path']}"
            return "Audio generation completed but no output file was produced."
        except Exception as e:
            logger.warning("[Tool:generate_audio] Failed: %s", e)
            return f"Audio generation failed: {e}"

    # check_identity tool was also removed in the R4 migration — sovereign
    # identity is now surfaced via [INNER STATE] rather than a tool call.

    return [
        research,
        generate_art,
        generate_audio,
    ]
