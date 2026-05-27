"""titan_hcl.probes.output_verifier — output_verifier probe (Chunk 11H).

Per RFP §11.I.3: Ed25519 sign a 1-byte payload ≤200ms. Shell implementation per the 11H plan — body
grows in 11I when output_verifier exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def output_verifier_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe output_verifier — Ed25519 sign a 1-byte payload ≤200ms."""
    return shell_probe("output_verifier", bus_client)
