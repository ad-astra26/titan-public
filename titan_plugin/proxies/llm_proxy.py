"""
LLM/Inference Module Proxy — lazy bridge to the supervised LLM process.

Encapsulates the Agno agent, Ollama Cloud client, and all inference calls.
Keeps the heavy LLM session state (~500MB) out of Core.
"""
import logging
from typing import Optional

from ..bus import DivineBus
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class LLMProxy:
    """
    Drop-in proxy for the LLM/Inference module.
    Routes chat, distillation, and inference calls through Divine Bus.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("llm_proxy", reply_only=True)
        self._started = False

    def _ensure_started(self) -> None:
        # Async-safe Guardian.start() — see _start_safe.py for rationale.
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "llm", id(self), proxy_label="LLMProxy"
        ):
            self._started = True

    async def chat(self, prompt: str, context: Optional[dict] = None) -> str:
        """Send a chat prompt to the Agno agent in the LLM module.

        Phase C Session 4 (rFP §4.C.6): true work-RPC (LLM inference) —
        migrated from sync bus.request to async bus.request_async per
        Preamble G19. Timeout 120s exceeds the §1.B 5s default for state
        lookup; explicitly allowlisted as LLM work-RPC in
        phase_c_rpc_exemptions.yaml.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "llm_proxy", "llm",
                {"action": "chat", "prompt": prompt,
                 "context": context or {}},
                120.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[LLMProxy] chat bus.request_async raised: %s", e)
            return ""
        if reply:
            return reply.get("payload", {}).get("response", "")
        logger.warning("[LLMProxy] chat timed out")
        return ""

    async def distill(self, text: str, instruction: str = "Summarize concisely") -> str:
        """Distill/summarize text using the lightweight model.

        Phase C Session 4 (rFP §4.C.6): true work-RPC (small LLM
        inference) — migrated from sync to async per G19. Timeout 30s
        — allowlisted as LLM work-RPC.
        """
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "llm_proxy", "llm",
                {"action": "distill", "text": text,
                 "instruction": instruction},
                30.0, self._reply_queue,
            )
        except Exception as e:
            logger.warning("[LLMProxy] distill bus.request_async raised: %s", e)
            return ""
        if reply:
            return reply.get("payload", {}).get("result", "")
        return ""
