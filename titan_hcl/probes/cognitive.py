"""titan_hcl.probes.cognitive — cognitive_worker probe (Chunk 11H).

Per RFP §11.I.3: epoch loop tick ≤500ms. Shell implementation per the 11H plan — body
grows in 11I when cognitive_worker exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def cognitive_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe cognitive_worker — epoch loop tick ≤500ms."""
    return shell_probe("cognitive_worker", bus_client)
