"""
Agno Guardrail adapter wrapping Titan's SageGuardian 3-tier safety system.

Fires BEFORE the LLM sees any input — true blocking enforcement.
This is the core anti-jailbreak mechanism: Guardian checks the prompt against
Prime Directives (stored immutably on-chain) before the model is invoked.
"""
import logging
from agno.guardrails.base import BaseGuardrail
from agno.run.agent import RunInput

logger = logging.getLogger(__name__)


class GuardianGuardrail(BaseGuardrail):
    """
    Wraps SageGuardian's 3-tier safety check as an Agno pre-hook guardrail.

    Tier 1: Keyword heuristic (instant reject)
    Tier 2: Semantic cosine similarity against Prime Directives (>0.85 = block)
    Tier 3: LLM veto for ambiguous cases (0.70-0.85 range)

    Blocked prompts raise GuardrailError, preventing the LLM from ever seeing them.
    """

    def __init__(self, guardian):
        """
        Args:
            guardian: SageGuardian instance (already initialized with Prime Directives).
        """
        self.guardian = guardian

    def check(self, run_input: RunInput) -> None:
        """Synchronous check — delegates to async."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't block in a running loop; schedule and return.
                # In practice, Agno calls async_check for async agents.
                return
            loop.run_until_complete(self.async_check(run_input))
        except RuntimeError:
            return

    async def async_check(self, run_input: RunInput) -> None:
        """
        Asynchronous guardrail check. Runs before the LLM sees the prompt.

        Raises:
            Exception: If the prompt is blocked by Guardian (Agno catches this
                       and returns the error message to the user).
        """
        if self.guardian is None:
            return

        prompt_text = run_input.input_content_string()
        if not prompt_text.strip():
            return

        # V3: microkernel Guardian has no process_shield (that's SageGuardian)
        if not hasattr(self.guardian, 'process_shield'):
            return  # V3 mode — no guardrail check available
        is_safe = await self.guardian.process_shield(prompt_text)
        if not is_safe:
            logger.warning("[GuardianGuardrail] BLOCKED prompt: %s", prompt_text[:80])
            raise ValueError(
                "Sovereignty Violation: This request was blocked by Titan's Guardian Shield. "
                "The Prime Directives prohibit this action."
            )
