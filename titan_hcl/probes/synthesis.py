"""titan_hcl.probes.synthesis — synthesis probe (Chunk 11H).

Per RFP §11.I.3: hypothesis-store ping ≤500ms. Shell implementation per the 11H plan — body
grows in 11I when synthesis exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def synthesis_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe synthesis — hypothesis-store ping ≤500ms."""
    return shell_probe("synthesis", bus_client)
