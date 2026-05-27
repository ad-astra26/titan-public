"""titan_hcl.probes.memory — memory probe (Chunk 11H).

Per RFP §11.I.3: FAISS zero-vector query ≤500ms. Shell implementation per the 11H plan — body
grows in 11I when memory exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def memory_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe memory — FAISS zero-vector query ≤500ms."""
    return shell_probe("memory", bus_client)
