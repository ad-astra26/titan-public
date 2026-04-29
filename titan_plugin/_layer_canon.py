"""
Canonical layer assignment for every Guardian-registered module.

Microkernel v2 Phase A §A.5 — authoritative mapping used by Guardian,
arch_map, dashboard endpoints, and tests. Single source of truth so
layer tags cannot drift between call sites.

Layer definitions (see titan-docs/rFP_microkernel_v2_shadow_core.md
§L0-L3 + titan-docs/PLAN_microkernel_phase_a.md §2):

    L0 — Microkernel (bus, guardian, state_registry framework, clocks,
         pi_heartbeat, soul, disk_health, bus_health, persistence).
         Runs in-process in the main kernel. No Guardian-supervised
         modules are L0 in Phase A.

    L1 — Trinity daemons + L1 persistence service.
         body, mind, spirit (Trinity 5DT); imw (writer-service for
         inner_memory.db which is L1's DB per rFP Q2).

    L2 — Higher cognition + cognitive substrates + L2 state registry.
         memory (FAISS+Kuzu+DuckDB), rl (IQL), cgn (concept grounding
         registry — see project_cgn_as_higher_state_registry.md),
         emot_cgn, language, meta_teacher, timechain (episodic/
         declarative substrate, not a Trinity daemon).

    L3 — Pluggable modules + L3 persistence service.
         llm (Agno inference), media (expression: speech/art/music),
         backup (on-chain anchoring + 3-2-1), knowledge (rFP L3
         "Knowledge search"), observatory_writer (writer-service for
         observatory.db which is L3's DB per rFP Q2).

References:
    - titan-docs/rFP_microkernel_v2_shadow_core.md §L0-L3
    - titan-docs/PLAN_microkernel_phase_a.md §2 (canonical table)
    - memory/project_cgn_as_higher_state_registry.md (CGN@L2 invariant)
    - HIGHLIGHTS_20260417_12_commit_session_... line 6444 (Maker's
      original 4-layer design quote)
"""
from typing import Final

VALID_LAYERS: Final[frozenset[str]] = frozenset({"L0", "L1", "L2", "L3"})

LAYER_CANON: Final[dict[str, str]] = {
    # L1 — Trinity daemons + L1 persistence service
    "body": "L1",
    "mind": "L1",
    "spirit": "L1",
    "imw": "L1",
    "outer_trinity": "L1",  # outer-trinity sister to body/mind/spirit (A.8.4)
    # L2 — Higher cognition + cognitive substrates + L2 state registry (cgn)
    "memory": "L2",
    "rl": "L2",
    "cgn": "L2",
    "emot_cgn": "L2",
    "language": "L2",
    "meta_teacher": "L2",
    "timechain": "L2",
    "output_verifier": "L2",  # security/verification gate (A.8.3 subprocess)
    # L3 — Pluggable modules + L3 persistence service
    "llm": "L3",
    "media": "L3",
    "backup": "L3",
    "knowledge": "L3",
    "observatory_writer": "L3",
    "warning_monitor": "L3",  # observability service (silent-swallow runtime visibility)
    "reflex": "L3",           # reflex aggregation subprocess (A.8.5)
    "agency_worker": "L3",    # impulse decoder + helper execution (A.8.6 subprocess)
    "consciousness_writer": "L3",   # IMW writer (writer-side lives at L3 with sibling writers)
    "social_graph_writer": "L3",    # IMW writer for L3 social_graph.db
    "events_teacher_writer": "L3",  # IMW writer for L3 events_teacher.db
}


def validate_layer(layer: str) -> str:
    """Return layer if valid, else raise ValueError."""
    if layer not in VALID_LAYERS:
        raise ValueError(
            f"Invalid layer {layer!r}; must be one of {sorted(VALID_LAYERS)}")
    return layer


def expected_layer(module_name: str) -> str | None:
    """Return the canonical layer for a module name, or None if not in canon."""
    return LAYER_CANON.get(module_name)
