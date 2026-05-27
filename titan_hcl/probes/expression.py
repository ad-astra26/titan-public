"""titan_hcl.probes.expression — expression_worker probe (Chunk 11H).

Per RFP §11.I.3: ExpressionManager.evaluate_all dry-run ≤500ms. Shell implementation per the 11H plan — body
grows in 11I when expression_worker exposes the module-level sentinel(s)
the probe inspects.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.module_state import ProbeResult
from ._common import shell_probe


def expression_worker_probe(bus_client: Optional[Any] = None) -> ProbeResult:
    """Probe expression_worker — ExpressionManager.evaluate_all dry-run ≤500ms."""
    return shell_probe("expression_worker", bus_client)
