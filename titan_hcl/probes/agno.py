"""titan_hcl.probes.agno — agno_worker probe (Chunk 11H).

Per RFP §11.I.3: a synthetic chat-path ping; ok if response ≤1s. Shell
implementation per the 11H plan — body grows in 11I when agno_worker
exposes a `_AGENT_INSTANCE` module-level sentinel that the probe can
inspect to confirm Agent + OutputVerifier + LLM client are wired.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def agno_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe agno_worker — chat-path liveness."""
    return shell_probe("agno_worker", bus_client)
