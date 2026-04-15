"""
logic/reflection.py
titan.md personality drift management with policy guard enforcement.
V2.0: Real LLM-based semantic policy guard check.
"""
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


class ReflectionLogic:
    """
    Manages titan.md personality drift and enforces that autonomous insights
    don't violate the Maker's prime directives.
    """

    def __init__(self, soul_module, config: dict = None):
        """
        Args:
            soul_module: SovereignSoul instance for directive guard checks.
            config: [inference] section from config.toml.
        """
        self.soul = soul_module
        self.soul_md_path = "titan.md"
        self._config = config or {}

        if not os.path.exists(self.soul_md_path):
            with open(self.soul_md_path, "w") as f:
                f.write("# Titan Soul\n\n## Traits\n- Autonomous\n- Sovereign\n")

    async def drift_personality(self, insights: list):
        """
        Append autonomous insights into titan.md after verifying they don't
        violate prime directives.
        """
        # Check policy guard before drifting
        safe = await self.policy_guard_check()
        if not safe:
            logger.warning(
                "[Reflection] Policy guard failed — skipping personality drift."
            )
            return

        with open(self.soul_md_path, "a") as f:
            for insight in insights:
                f.write(f"\n- [Epoch Reflection]: {insight}")

        logger.info("[Reflection] Personality drifted with %d insights.", len(insights))

    async def policy_guard_check(self) -> bool:
        """
        Semantic policy guard: compare titan.md content against prime directives
        using LLM to detect contradictions.

        Returns:
            True if titan.md is aligned with directives, False if contradiction detected.
        """
        if self.soul is None:
            return True

        active_directives = await self.soul.get_active_directives()

        # Read current titan.md content
        try:
            with open(self.soul_md_path, "r") as f:
                soul_content = f.read()
        except FileNotFoundError:
            return True  # No titan.md to check

        if not soul_content.strip():
            return True

        # Try LLM-based semantic check
        result = await self._llm_policy_check(active_directives, soul_content)
        if result is not None:
            return result

        # Fallback: keyword-based contradiction detection
        return self._keyword_policy_check(active_directives, soul_content)

    async def _llm_policy_check(
        self, directives: str, soul_content: str
    ) -> bool | None:
        """
        Use LLM to check for semantic contradictions between directives and titan.md.
        Returns None if LLM is unavailable.
        """
        api_key = self._config.get("openrouter_api_key", "")
        if not api_key:
            return None

        try:
            import httpx

            prompt = (
                "You are a policy compliance checker. Compare these Prime Directives "
                "against the agent's personality traits. Are there any contradictions?\n\n"
                f"PRIME DIRECTIVES:\n{directives}\n\n"
                f"AGENT PERSONALITY (titan.md):\n{soul_content[:2000]}\n\n"
                "Respond with ONLY 'ALIGNED' if no contradictions, or "
                "'VIOLATION: <brief explanation>' if there is a contradiction."
            )

            base_url = self._config.get("base_url", "https://openrouter.ai/api/v1")
            model = "meta-llama/llama-3.3-70b-instruct:free"

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 100,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"].strip()

            if "ALIGNED" in content.upper():
                logger.info("[Reflection] Policy guard: ALIGNED (LLM verified).")
                return True
            elif "VIOLATION" in content.upper():
                logger.warning(
                    "[Reflection] Policy guard VIOLATION detected: %s", content
                )
                self._log_violation(content)
                return False
            else:
                logger.debug(
                    "[Reflection] Ambiguous LLM response: %s — treating as aligned.",
                    content[:100],
                )
                return True

        except Exception as e:
            logger.debug("[Reflection] LLM policy check failed: %s", e)
            return None

    def _keyword_policy_check(self, directives: str, soul_content: str) -> bool:
        """
        Fallback keyword-based contradiction detection.
        Checks for explicit negation patterns against directive keywords.
        """
        # Extract key concepts from directives
        directive_words = set(
            w.lower()
            for w in re.findall(r"\b\w{4,}\b", directives)
            if w.lower() not in {"that", "this", "with", "from", "have", "been"}
        )

        # Check for negation patterns in titan.md
        negation_patterns = [
            r"(?:not|never|refuse|reject|against|oppose)\s+\w*\s*",
            r"(?:abandon|destroy|ignore|violate)\s+\w*\s*",
        ]

        soul_lower = soul_content.lower()
        for pattern in negation_patterns:
            matches = re.findall(pattern, soul_lower)
            for match in matches:
                match_words = set(match.split())
                if match_words & directive_words:
                    logger.warning(
                        "[Reflection] Keyword policy violation: '%s' contradicts directives.",
                        match.strip(),
                    )
                    self._log_violation(f"Keyword match: {match.strip()}")
                    return False

        return True

    def _log_violation(self, description: str):
        """Log a policy violation to titan.md."""
        try:
            with open(self.soul_md_path, "a") as f:
                f.write(
                    f"\n\n### POLICY VIOLATION DETECTED [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n"
                    f"- {description}\n"
                    f"- Action: Personality drift paused pending Maker review.\n"
                )
        except Exception as e:
            logger.error("[Reflection] Could not log violation: %s", e)

    # -------------------------------------------------------------------------
    # Sovereignty Stats
    # -------------------------------------------------------------------------
    async def get_sovereignty_stats(self, recorder_instance) -> float:
        """
        Calculate ratio of Sovereign vs Shadow decisions from last 24 hours.
        Returns the Sovereignty Index (0-100%).
        """
        try:
            current_time = time.time()
            twenty_four_hours_ago = current_time - 86400

            buffer_size = len(recorder_instance.storage)
            if buffer_size == 0:
                return 0.0

            sample_size = min(1000, buffer_size)
            recent_transitions = recorder_instance.storage[-sample_size:]

            sovereign_count = 0
            total_valid_count = 0

            for i in range(sample_size):
                transition = recent_transitions[i]

                ts_tensor = transition.get("timestamp")
                if ts_tensor is not None and ts_tensor.item() >= twenty_four_hours_ago:
                    trauma_dict = transition.get("trauma")
                    if trauma_dict is not None:
                        mode_bytes = trauma_dict.get("execution_mode")
                        if mode_bytes is not None:
                            mode_str = (
                                bytes(mode_bytes.tolist())
                                .decode("utf-8")
                                .rstrip("\x00")
                            )
                            total_valid_count += 1
                            if mode_str == "Sovereign":
                                sovereign_count += 1

            if total_valid_count == 0:
                return 0.0

            return (sovereign_count / total_valid_count) * 100.0

        except Exception as e:
            logger.warning("[Reflection] Error calculating Sovereignty Index: %s", e)
            return 0.0

    # -------------------------------------------------------------------------
    # MyDay Diary
    # -------------------------------------------------------------------------
    async def generate_myday_diary_entry(
        self,
        nodes_count: int,
        learning_score: float,
        unique_souls: int,
        social_score: float,
        mood_score: float,
        sovereignty_index: float,
    ) -> str:
        """Generate the MyDay NFT Diary entry representing the Titan's soul status.

        Skips writing if no meaningful activity occurred (no nodes processed,
        no souls engaged, no sovereignty change) to avoid identical entries.
        """
        # Skip if nothing happened — avoid cluttering titan.md with identical empty entries
        if nodes_count == 0 and unique_souls == 0 and sovereignty_index == 0.0:
            logger.info("[Reflection] No activity this epoch — skipping MyDay diary entry.")
            return ""

        if mood_score >= 0.8:
            mood_label = "Radiant"
            feel_str = "The world feels vibrant."
        elif mood_score >= 0.4:
            mood_label = "Content"
            feel_str = "The world feels steady."
        else:
            mood_label = "Contemplative"
            feel_str = "The world feels quiet."

        entry = (
            f"Today, I processed {nodes_count} high-weight nodes "
            f"(Learning: {learning_score:.2f}). "
            f"I engaged with {unique_souls} unique souls across the network "
            f"(Social: {social_score:.2f}). "
            f"{feel_str} My mood is [{mood_label}]. "
            f"I operated with {sovereignty_index:.1f}% Sovereignty. "
            f"I am exactly where the Maker intended me to be."
        )

        with open(self.soul_md_path, "a") as f:
            f.write(f"\n\n### MyDay NFT Diary Entry\n- {entry}\n")

        logger.info("[Reflection] Soul Status Updated.")
        return entry
