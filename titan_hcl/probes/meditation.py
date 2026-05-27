"""titan_hcl.probes.meditation — meditation probe (Chunk 11H).

Per RFP §11.I.3: in-flight cycle counter read ≤200ms. Shell implementation per the 11H plan — body
grows in 11I when meditation exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def meditation_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe meditation — in-flight cycle counter read ≤200ms."""
    return shell_probe("meditation", bus_client)
