"""titan_hcl.probes.observatory — observatory probe (Chunk 11H).

Per RFP §11.I.3: in-process snapshot read ≤200ms. Shell implementation per the 11H plan — body
grows in 11I when observatory exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def observatory_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe observatory — in-process snapshot read ≤200ms."""
    return shell_probe("observatory", bus_client)
