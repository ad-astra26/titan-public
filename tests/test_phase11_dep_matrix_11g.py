"""
Phase 11 §11.I.8 / D-SPEC-141 — §3H.10 dep matrix population (Chunk 11G).

Verifies that `titan_hcl.module_catalog.build_catalog` registers each
module with the boot_priority + MODULE-kind dependencies declared in
RFP §3H.10. The matrix is the canonical source of truth — this test
freezes it in code so any future drift is loud.
"""
from __future__ import annotations

import pytest

from titan_hcl.bus import DivineBus
from titan_hcl.orchestrator import ModuleSpec, Orchestrator
from titan_hcl.supervision import (
    DependencyAction,
    DependencyKind,
    DependencySeverity,
)


# §3H.10 matrix — keep in sync with
# scripts/_phase11_11g_apply_dep_matrix.py:MATRIX. Each value is
# (boot_priority, [module_dep_names]).
EXPECTED_MATRIX: dict[str, tuple[str, list[str]]] = {
    # MANDATORY
    "imw":               ("mandatory", []),
    "body":              ("mandatory", []),
    "mind":              ("mandatory", []),
    "output_verifier":   ("mandatory", ["timechain"]),
    "warning_monitor":   ("mandatory", []),
    "health_monitor":    ("mandatory", []),
    "memory":            ("mandatory", []),
    "agno_worker":       ("mandatory",
                          ["memory", "output_verifier", "timechain"]),
    "cognitive_worker":  ("mandatory", ["memory"]),
    "observatory":       ("mandatory", []),
    "sovereignty":       ("mandatory", ["timechain"]),
    "timechain":         ("mandatory", []),
    "api":               ("mandatory", ["agno_worker", "memory"]),
    # OPTIONAL_POST_BOOT
    "reflex":            ("post_boot", []),
    "agency_worker":     ("post_boot", []),
    "observatory_writer":("post_boot", []),
    "recorder":          ("post_boot", []),
    "llm":               ("post_boot", []),
    "expression_worker": ("post_boot", ["cognitive_worker"]),
    "outer_interface_worker": ("post_boot", []),
    "self_reflection_worker": ("post_boot", ["cognitive_worker"]),
    "social_worker":     ("post_boot", ["social_graph"]),
    "social_graph":      ("post_boot", []),
    "metabolism":        ("post_boot", []),
    "journey_persistence": ("post_boot", []),
    "corrective_events_persistence": ("post_boot", []),
    "life_force":        ("post_boot", []),
    "studio":            ("post_boot", []),
    "dream_state":       ("post_boot", []),
    "synthesis":         ("post_boot", ["timechain", "memory"]),
    "meditation":        ("post_boot", ["memory", "timechain"]),
    "interface_advisor": ("post_boot", []),
    "ns_module":         ("post_boot", []),
    "neuromod_module":   ("post_boot", []),
    "hormonal_module":   ("post_boot", []),
    "media":             ("post_boot", []),
    "language":          ("post_boot", ["llm"]),
    "meta_teacher":      ("post_boot", ["llm"]),
    "cgn":               ("post_boot", []),
    "knowledge":         ("post_boot", []),
    "backup":            ("post_boot", ["timechain"]),
    "emot_cgn":          ("post_boot", []),
}


def _build_catalog_with_all_flags_on() -> Orchestrator:
    """Build a catalog with every flag-gated worker enabled so the
    matrix is exhaustively populated.

    Mirrors a real Titan boot config — `_a8_*_subprocess_enabled = True`
    + nested `outer_interface_worker_enabled` + `microkernel.*` flags
    are turned on. Returns the orchestrator with all 42 registrations.
    """
    bus = DivineBus()
    orch = Orchestrator(bus)
    cfg = {
        "microkernel": {
            "a8_output_verifier_subprocess_enabled": True,
            "a8_reflex_subprocess_enabled": True,
            "a8_agency_subprocess_enabled": True,
            "a8_sage_scholar_gatekeeper_subprocess_enabled": True,
            "outer_interface_worker_enabled": True,
            "spawn_graduated_workers_enabled": False,
            # Phase 11 §3H.10 — l0_rust_enabled gates cognitive/expression/
            # outer_interface/self_reflection registration in this catalog
            # (the l0_rust=false path was retired per `feedback_l0_rust_flag_is_authoritative`).
            "l0_rust_enabled": True,
            "social_worker_enabled": True,
        },
        "persistence": {"enabled": True},
        "memory_and_storage": {"data_dir": "./data"},
        "inference": {},
        "stealth_sage": {},
        "expressive": {},
        "studio": {},
        "info_banner": {},
        "outer_interface": {},
        "self_exploration": {},
        "action_decoder": {},
        "action_narrator": {},
        "kin": {},
    }
    # build_catalog is a top-level function — invoke it directly. Since C.6
    # (RFP_config_as_shm_state §7.C) it reads each section from SHM via
    # get_params (no boot config dict); in this no-daemon test we patch
    # get_params to serve the all-flags-on cfg above.
    from titan_hcl.module_catalog import build_catalog
    import titan_hcl.params as _params
    _orig = _params.get_params
    _params.get_params = lambda section: dict(cfg.get(section, {}))
    try:
        build_catalog(bus, orch, titan_id="test")
    finally:
        _params.get_params = _orig
    return orch


def test_every_matrix_entry_present_in_catalog():
    """Every module in the §3H.10 matrix must be registered in the
    catalog."""
    orch = _build_catalog_with_all_flags_on()
    missing = [n for n in EXPECTED_MATRIX if n not in orch._modules]
    assert missing == [], (
        f"Modules in §3H.10 matrix but not registered in catalog: {missing}")


def test_every_module_has_boot_priority_set():
    """No module should default-inherit boot_priority; every registration
    should explicitly declare it (matrix coverage gate)."""
    orch = _build_catalog_with_all_flags_on()
    # Allow modules not in the matrix to still default; flag only those
    # we EXPECT to be in the matrix.
    for name, expected in EXPECTED_MATRIX.items():
        if name not in orch._modules:
            continue
        spec = orch._modules[name].spec
        assert spec.boot_priority == expected[0], (
            f"Module '{name}': expected boot_priority={expected[0]!r}, "
            f"got {spec.boot_priority!r}")


def test_mandatory_module_module_deps_match_matrix():
    """Each module's `dependencies` field must contain (at least) the
    MODULE-kind ENSURE_RUNNING entries from the §3H.10 matrix.

    Allows additional deps beyond the matrix (e.g., EXTERNAL_SVC SOFT
    deps for Ollama/Solana) — extra deps are not regressions.
    """
    orch = _build_catalog_with_all_flags_on()
    for name, (_, expected_deps) in EXPECTED_MATRIX.items():
        if name not in orch._modules:
            continue
        spec = orch._modules[name].spec
        declared = [
            d.name for d in spec.dependencies
            if d.kind == DependencyKind.MODULE
            and d.action == DependencyAction.ENSURE_RUNNING
        ]
        missing = [d for d in expected_deps if d not in declared]
        assert not missing, (
            f"Module '{name}': matrix expects MODULE deps {expected_deps}; "
            f"declared {declared}; missing {missing}")


def test_mandatory_modules_partition_correctly():
    """Phase A bucket (MANDATORY) must include every module whose
    matrix entry says MANDATORY."""
    orch = _build_catalog_with_all_flags_on()
    expected_mandatory = {
        n for n, (bp, _) in EXPECTED_MATRIX.items() if bp == "mandatory"
    }
    autostart_names = [
        n for n, info in orch._modules.items() if info.spec.autostart
    ]
    mandatory_actual, _, _ = orch._partition_autostart_by_boot_priority(
        autostart_names)
    # Every expected MANDATORY autostart module is in mandatory_actual.
    expected_mandatory_autostart = {
        n for n in expected_mandatory
        if n in orch._modules and orch._modules[n].spec.autostart
    }
    missing = expected_mandatory_autostart - set(mandatory_actual)
    assert not missing, (
        f"Expected MANDATORY autostart modules missing from Phase A: "
        f"{sorted(missing)}")


def test_memory_promoted_to_autostart_and_eager():
    """§3H.10 promotion: memory was autostart=False + lazy=True; under
    Phase 11 it MUST eagerly start so fleet_ready can latch with it
    already RUNNING."""
    orch = _build_catalog_with_all_flags_on()
    assert "memory" in orch._modules
    memory_spec = orch._modules["memory"].spec
    assert memory_spec.autostart is True, (
        "memory must autostart=True under Phase 11 §3H.10 promotion")
    assert memory_spec.lazy is False, (
        "memory must lazy=False under Phase 11 §3H.10 promotion")
    assert memory_spec.boot_priority == "mandatory"


def test_boot_order_topological_with_matrix_deps():
    """End-to-end: with the §3H.10 deps populated, _compute_boot_order
    places each child AFTER its parent. Spot-checks the canonical
    chains: timechain → output_verifier → agno_worker → api;
    memory → agno_worker → api; memory → cognitive_worker."""
    orch = _build_catalog_with_all_flags_on()
    autostart_names = [
        n for n, info in orch._modules.items() if info.spec.autostart
    ]
    mandatory, _, _ = orch._partition_autostart_by_boot_priority(
        autostart_names)
    order = orch._compute_boot_order(mandatory)

    def idx(name):
        return order.index(name) if name in order else -1

    # Chain 1: timechain → output_verifier → agno_worker → api
    if all(n in order for n in ["timechain", "output_verifier", "agno_worker", "api"]):
        assert idx("timechain") < idx("output_verifier") < idx("agno_worker") < idx("api"), (
            f"timechain → output_verifier → agno_worker → api chain violated "
            f"in {order}")
    # Chain 2: memory → agno_worker, memory → api
    if all(n in order for n in ["memory", "agno_worker"]):
        assert idx("memory") < idx("agno_worker"), (
            f"memory must precede agno_worker; got {order}")
    if all(n in order for n in ["memory", "cognitive_worker"]):
        assert idx("memory") < idx("cognitive_worker"), (
            f"memory must precede cognitive_worker; got {order}")
    # Chain 3: timechain → sovereignty
    if all(n in order for n in ["timechain", "sovereignty"]):
        assert idx("timechain") < idx("sovereignty"), (
            f"timechain must precede sovereignty; got {order}")
