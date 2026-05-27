"""titan_hcl.probes.social — social_worker probe (Chunk 11H).

Per RFP §11.I.3: SocialXGateway gate-status read ≤500ms. Shell implementation per the 11H plan — body
grows in 11I when social_worker exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def social_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe social_worker — SocialXGateway gate-status read ≤500ms."""
    return shell_probe("social_worker", bus_client)
