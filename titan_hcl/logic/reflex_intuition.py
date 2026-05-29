"""titan_hcl/logic/reflex_intuition.py — pure-compute reflex intuition signals.

Phase 11 §11.I.5 / Chunk 11K (folded from Phase 9 9C) — the three
`_compute_*_reflex_intuition` helpers are PURE compute (no bus calls,
no I/O, no state mutation). Under SPEC §11.B.4 + the Phase 11 §3H.2
contract for worker isolation, `agno_hooks.py` MUST NOT import from
`titan_hcl.modules.*_worker` modules — that would re-introduce the
boot-time transitive-import cost the Phase 11 orchestrator is trying
to eliminate.

This module is the canonical IMPORT SURFACE for agno_hooks (and anyone
else who needs the reflex intuition signals). The function BODIES still
live in their respective worker files because the workers themselves
call them internally — `body_worker.py` / `mind_worker.py` /
`spirit_loop.py`. Moving the bodies would force a second move when the
workers shed the helpers; the re-export here is the load-bearing
contract change.

Pre-Phase-11 (the audit'd pattern):

    # agno_hooks.py — VIOLATES SPEC §11.B.4 boundary
    from titan_hcl.modules.body_worker  import _compute_body_reflex_intuition
    from titan_hcl.modules.mind_worker  import _compute_mind_reflex_intuition
    from titan_hcl.modules.spirit_loop  import _compute_spirit_reflex_intuition

Post-Phase-11 11K (this file):

    # agno_hooks.py — clean
    from titan_hcl.logic.reflex_intuition import (
        compute_body_reflex_intuition,
        compute_mind_reflex_intuition,
        compute_spirit_reflex_intuition,
    )

The `compute_*` names drop the underscore prefix since logic/ exports
are public-by-convention; the underscore-prefixed names are also
re-exported for back-compat with body/mind/spirit's own callsites until
a future no-shim sweep retires them.
"""
from __future__ import annotations

from titan_hcl.modules.body_worker import _compute_body_reflex_intuition
from titan_hcl.modules.mind_worker import _compute_mind_reflex_intuition
from titan_hcl.modules.spirit_loop import _compute_spirit_reflex_intuition

# Public-by-convention re-exports.
compute_body_reflex_intuition = _compute_body_reflex_intuition
compute_mind_reflex_intuition = _compute_mind_reflex_intuition
compute_spirit_reflex_intuition = _compute_spirit_reflex_intuition


__all__ = [
    "compute_body_reflex_intuition",
    "compute_mind_reflex_intuition",
    "compute_spirit_reflex_intuition",
    # Back-compat aliases — same objects, underscore names.
    "_compute_body_reflex_intuition",
    "_compute_mind_reflex_intuition",
    "_compute_spirit_reflex_intuition",
]
