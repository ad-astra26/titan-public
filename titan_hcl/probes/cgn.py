"""titan_hcl.probes.cgn — cgn probe (Chunk 11H).

Per RFP §11.I.3: model.forward(zero_input) ok ≤1s. Shell implementation per the 11H plan — body
grows in 11I when cgn exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def cgn_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe cgn — model.forward(zero_input) ok ≤1s."""
    return shell_probe("cgn", bus_client)
