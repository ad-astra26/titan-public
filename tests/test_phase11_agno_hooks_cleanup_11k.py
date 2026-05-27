"""
Phase 11 §11.I.5 / Chunk 11K — agno_hooks import-graph cleanup.

Per RFP §3H.4 11K verification gate:
  > agno_hooks.py zero `from titan_hcl.modules.*_worker import`; agno probe passes

Folded from Phase 9 9C — `agno_hooks.py` previously imported reflex-intuition
helpers directly from `body_worker.py`, `mind_worker.py`, and `spirit_loop.py`,
which forced a boot-time transitive import of those worker bodies into
every process that loads agno_hooks (including the api subprocess + tests).
Phase 11 §11.I.5 surfaces this via the §11.B.4 process-boundary contract:
"workers communicate via bus + SHM, never by importing each other's module
bodies".

This test freezes the contract — adding any new
`from titan_hcl.modules.*_worker import` to agno_hooks.py fails the suite.
"""
from __future__ import annotations

import re
from pathlib import Path


_AGNO_HOOKS_PATH = "titan_hcl/modules/agno_hooks.py"
_WORKER_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+titan_hcl\.modules\.[A-Za-z_]+_worker\s+import\b",
    re.MULTILINE,
)


def test_agno_hooks_has_zero_worker_imports():
    src = Path(_AGNO_HOOKS_PATH).read_text()
    bad = _WORKER_IMPORT_PATTERN.findall(src)
    assert bad == [], (
        f"agno_hooks.py contains {len(bad)} `from titan_hcl.modules.*_worker "
        f"import` line(s) — Phase 11 §11.I.5 / 11K bans these. Found:\n"
        + "\n".join(bad)
    )


def test_agno_hooks_uses_logic_reflex_intuition():
    """Confirms the canonical import path is via `logic.reflex_intuition`
    (the post-11K surface), not via the legacy worker-direct path."""
    src = Path(_AGNO_HOOKS_PATH).read_text()
    assert re.search(
        r"from\s+titan_hcl\.logic\.reflex_intuition\s+import\s+\(",
        src,
    ), ("agno_hooks.py must import reflex-intuition helpers via "
        "`titan_hcl.logic.reflex_intuition` per Phase 11 §11.I.5 / 11K")


def test_logic_reflex_intuition_exports_all_three():
    """Roster guard — adding a fourth Trinity reflex source needs an
    explicit roster bump here so callers don't silently miss a signal."""
    from titan_hcl.logic.reflex_intuition import (
        compute_body_reflex_intuition,
        compute_mind_reflex_intuition,
        compute_spirit_reflex_intuition,
    )
    assert callable(compute_body_reflex_intuition)
    assert callable(compute_mind_reflex_intuition)
    assert callable(compute_spirit_reflex_intuition)


def test_underscore_aliases_are_same_callables():
    """Back-compat: the underscore-prefixed names re-exported alongside
    the public-by-convention names must be the SAME callable objects."""
    from titan_hcl.logic.reflex_intuition import (
        _compute_body_reflex_intuition,
        _compute_mind_reflex_intuition,
        _compute_spirit_reflex_intuition,
        compute_body_reflex_intuition,
        compute_mind_reflex_intuition,
        compute_spirit_reflex_intuition,
    )
    assert _compute_body_reflex_intuition is compute_body_reflex_intuition
    assert _compute_mind_reflex_intuition is compute_mind_reflex_intuition
    assert _compute_spirit_reflex_intuition is compute_spirit_reflex_intuition


def test_reflex_intuition_signals_via_logic_surface_are_lists():
    """End-to-end smoke: invoking via the new surface returns the
    expected list-of-dicts shape so downstream agno_hooks consumers
    aren't broken by the import-path move."""
    from titan_hcl.logic.reflex_intuition import (
        compute_body_reflex_intuition,
        compute_mind_reflex_intuition,
    )
    stimulus = {
        "threat_level": 0.5, "intensity": 0.5, "topics": ["technical"],
        "topic": "technical", "engagement": 0.5, "message": "what is X?",
        "user_id": "u1", "valence": 0.0,
    }
    tensor_5d = [0.4, 0.4, 0.4, 0.4, 0.4]
    body_signals = compute_body_reflex_intuition(stimulus, tensor_5d)
    mind_signals = compute_mind_reflex_intuition(stimulus, tensor_5d, None, None)
    assert isinstance(body_signals, list)
    assert isinstance(mind_signals, list)
    # All emitted signals MUST carry the canonical schema.
    for sig in body_signals + mind_signals:
        assert {"reflex", "source", "confidence", "reason"} <= set(sig.keys())
