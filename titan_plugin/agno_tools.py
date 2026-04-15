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

    # ------------------------------------------------------------------
    # Memory Recall Tool
    # ------------------------------------------------------------------
    async def recall_memory(query: str) -> str:
        """
        Search Titan's long-term memory graph (Cognee) for relevant past
        interactions, knowledge, and experiences. Use this when the user
        asks about previous conversations or when you need historical context.

        Args:
            query: Semantic search query for the memory graph.

        Returns:
            Formatted list of matching memories with relevance scores.
        """
        try:
            memories = await plugin.memory.query(query)
        except Exception as e:
            logger.warning("[Tool:recall_memory] Memory query failed: %s", e)
            return "Memory recall failed — graph may be initializing."

        if not memories:
            return "No relevant memories found for this query."

        lines = []
        for m in memories[:10]:
            p = m.get("user_prompt", "")[:150]
            r = m.get("agent_response", "")[:150]
            w = m.get("effective_weight", 1.0)
            lines.append(f"[{w:.2f}] Q: {p}\n  A: {r}")
        return "### Memory Recall Results\n" + "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Metabolism Status Tool
    # ------------------------------------------------------------------
    async def check_metabolism() -> str:
        """
        Check Titan's current metabolic state: SOL balance, energy level,
        and Divine Growth metrics. Use this to assess whether Titan has
        enough energy for costly operations (research, on-chain writes).

        Returns:
            Metabolic state summary including energy level and balance.
        """
        try:
            state = await plugin.metabolism.get_current_state()
            balance = plugin.metabolism._last_balance
            balance_pct = plugin.metabolism._last_balance_pct

            learning_v = await plugin.metabolism.get_learning_velocity()
            social_d = await plugin.metabolism.get_social_density()
            health = await plugin.metabolism.get_metabolic_health()

            return (
                f"### Metabolic Status\n"
                f"- Energy State: {state}\n"
                f"- SOL Balance: {balance:.4f} ({balance_pct:.0f}%)\n"
                f"- Learning Velocity: {learning_v:.2f}\n"
                f"- Social Density: {social_d:.2f}\n"
                f"- Overall Health: {health:.2f}\n"
            )
        except Exception as e:
            logger.warning("[Tool:check_metabolism] Failed: %s", e)
            return f"Metabolism check failed: {e}"

    # ------------------------------------------------------------------
    # Social Post Tool
    # ------------------------------------------------------------------
    async def post_to_x(text: str) -> str:
        """
        Post a tweet to X/Twitter using Titan's authenticated social identity.
        The post goes through metabolic governance (must be able to afford it)
        and will be logged to the memory graph. Use this when the user asks
        Titan to share something publicly or when a significant event warrants
        social expression.

        Args:
            text: The tweet text (max 280 characters).

        Returns:
            Success or failure message.
        """
        if not text or not text.strip():
            return "Cannot post empty tweet."

        if len(text) > 280:
            return f"Tweet too long ({len(text)} chars). Maximum is 280."

        try:
            success = await plugin.social.create_tweet(text)
            if success:
                return f"Tweet posted successfully: \"{text[:80]}...\""
            return "Tweet failed — check social credentials and proxy configuration."
        except Exception as e:
            logger.warning("[Tool:post_to_x] Failed: %s", e)
            return f"Social post failed: {e}"

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

    # ------------------------------------------------------------------
    # Soul Identity Tool
    # ------------------------------------------------------------------
    async def check_identity() -> str:
        """
        Check Titan's on-chain sovereign identity: wallet pubkey, Genesis NFT
        status, active Prime Directives, and maker verification state.
        Use this when the user asks about Titan's identity or sovereignty.

        Returns:
            Identity summary including pubkey and directive count.
        """
        try:
            pubkey = plugin.soul.pubkey
            directives = await plugin.soul.get_active_directives()
            mood_label = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown"

            return (
                f"### Sovereign Identity\n"
                f"- Pubkey: {pubkey}\n"
                f"- Active Directives: {len(directives)}\n"
                f"- Current Mood: {mood_label}\n"
                f"- Execution Mode: {plugin._last_execution_mode}\n"
                f"- Limbo: {plugin._limbo_mode}\n"
            )
        except Exception as e:
            logger.warning("[Tool:check_identity] Failed: %s", e)
            return f"Identity check failed: {e}"

    # Sovereign Reflex Arc (R4): Observation tools are now handled by Trinity
    # Intuition reflexes — the LLM receives results as [INNER STATE] rather
    # than calling tools. Only creative/action tools remain for edge cases
    # where the LLM needs to narrate an explicit user request.
    #
    # REMOVED (now reflexes): recall_memory, check_metabolism, check_identity
    # REMOVED: post_to_x — all posting goes through SocialPressureMeter
    # (social_narrator + quality gate + rate limits + 11 post types)
    # KEPT (action/creative): generate_art, generate_audio, research
    return [
        research,
        generate_art,
        generate_audio,
    ]
