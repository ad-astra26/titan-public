"""Phase B.2.1 chunk C3 — spawn-mode graduation tests.

Verifies the 9 import-stable workers (warning_monitor, llm, body, mind,
media, language, meta_teacher, cgn, knowledge) flip from fork → spawn when
microkernel.spawn_graduated_workers_enabled is true. Backup runs under its
own pre-existing flag (S6 reference). Workers that stay fork-mode under
the new flag (spirit, memory, timechain, rl, emot_cgn) MUST also be
verified — graduating them later requires individual import-stability work.

Two layers:
  1. AST coverage  — every graduating worker's ModuleSpec call uses the
     `_spawn_grad` ternary; non-graduating workers do NOT.
  2. Runtime behaviour — registering modules with the flag flipped on/off
     yields the expected start_method per worker (queries Guardian after
     register).
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest


GRADUATING = {
    # First batch (C3, 2026-04-27 PM):
    "warning_monitor", "llm", "body", "mind", "media",
    "language", "meta_teacher", "cgn", "knowledge",
    # Second batch (Path B' extension, 2026-04-27 PM):
    # Graduate to "outlive the swap" rather than respawn — keeps
    # FAISS/DuckDB/timechain-state/region-buffer state in-process across
    # kernel swap. Architectural fix vs the "bump SHADOW_BOOT_TIMEOUT"
    # smell when workers re-load heavy state in shadow.
    "memory", "timechain", "emot_cgn", "rl",
}

# Workers that MUST stay fork-mode under spawn_graduated_workers_enabled.
# Spirit kept fork-mode for now — consciousness loop is tightly coupled
# to kernel state and needs separate analysis before graduation.
# imw + observatory_writer + the 3 new IMW-pattern writers (social_graph_writer,
# events_teacher_writer, consciousness_writer) are autonomous writer
# daemons with their own start path — not part of the GRADUATING flag.
# (Backup uses its own _spawn_ref flag and is excluded from this check.)
NON_GRADUATING = {
    "spirit",
    "imw", "observatory_writer",
    "social_graph_writer", "events_teacher_writer", "consciousness_writer",
}

PLUGIN_FILES = [
    Path(__file__).parent.parent / "titan_plugin" / "core" / "plugin.py",
    Path(__file__).parent.parent / "titan_plugin" / "legacy_core.py",
]


def _spec_calls_with_name(tree: ast.AST):
    """Yield (name, ModuleSpec ast.Call) for every <obj>.register(ModuleSpec(name=..., ...))."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "register"):
            continue
        if not node.args:
            continue
        a0 = node.args[0]
        if not (isinstance(a0, ast.Call)
                and isinstance(a0.func, ast.Name)
                and a0.func.id == "ModuleSpec"):
            continue
        for kw in a0.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                yield kw.value.value, a0
                break


def _start_method_kwarg(spec_call: ast.Call):
    for kw in spec_call.keywords:
        if kw.arg == "start_method":
            return kw.value
    return None


@pytest.mark.parametrize("plugin_file", PLUGIN_FILES,
                         ids=lambda p: p.name)
@pytest.mark.parametrize("worker_name", sorted(GRADUATING))
def test_graduating_workers_use_spawn_grad_ternary(plugin_file: Path,
                                                   worker_name: str):
    """Each graduating worker's ModuleSpec MUST set start_method via the
    _spawn_grad ternary. Catches anyone hardcoding "spawn" or forgetting
    the gate.
    """
    src = plugin_file.read_text()
    tree = ast.parse(src)
    found = False
    for name, spec_call in _spec_calls_with_name(tree):
        if name != worker_name:
            continue
        sm = _start_method_kwarg(spec_call)
        assert sm is not None, (
            f"{plugin_file.name}: ModuleSpec(name='{worker_name}') "
            f"is missing start_method kwarg"
        )
        # Must be `"spawn" if _spawn_grad else "fork"` (an IfExp using _spawn_grad)
        assert isinstance(sm, ast.IfExp), (
            f"{plugin_file.name}: ModuleSpec(name='{worker_name}') "
            f"start_method must be a ternary, got {ast.dump(sm)}"
        )
        cond = sm.test
        assert isinstance(cond, ast.Name) and cond.id == "_spawn_grad", (
            f"{plugin_file.name}: ModuleSpec(name='{worker_name}') "
            f"start_method ternary must gate on _spawn_grad"
        )
        found = True
        break
    assert found, (
        f"{plugin_file.name}: no ModuleSpec(name='{worker_name}') found"
    )


@pytest.mark.parametrize("plugin_file", PLUGIN_FILES,
                         ids=lambda p: p.name)
@pytest.mark.parametrize("worker_name", sorted(NON_GRADUATING))
def test_non_graduating_workers_unchanged(plugin_file: Path, worker_name: str):
    """Non-graduating workers must NOT use the _spawn_grad ternary.

    Either they have no start_method (default 'fork') or they have a
    different fixed value. Graduating them later requires a separate
    PR documenting the import-stability evidence per worker.
    """
    src = plugin_file.read_text()
    tree = ast.parse(src)
    for name, spec_call in _spec_calls_with_name(tree):
        if name != worker_name:
            continue
        sm = _start_method_kwarg(spec_call)
        if sm is None:
            return  # default fork — fine
        if isinstance(sm, ast.Constant):
            return  # hardcoded value — fine (backup uses _spawn_ref pattern)
        # If it's an IfExp, it must NOT gate on _spawn_grad
        if isinstance(sm, ast.IfExp):
            cond = sm.test
            assert not (isinstance(cond, ast.Name) and cond.id == "_spawn_grad"), (
                f"{plugin_file.name}: non-graduating worker '{worker_name}' "
                f"is using the _spawn_grad gate — that means it'll flip with "
                f"the rest. If you intend to graduate it, add to GRADUATING; "
                f"otherwise back this out."
            )


@pytest.mark.parametrize("plugin_file", PLUGIN_FILES, ids=lambda p: p.name)
def test_spawn_grad_read_present(plugin_file: Path):
    """The flag must be read from config in _register_modules.

    Asserts a literal `_spawn_grad = ` assignment + reads
    'spawn_graduated_workers_enabled' from microkernel section.
    """
    src = plugin_file.read_text()
    assert "_spawn_grad" in src, f"{plugin_file.name}: _spawn_grad not assigned"
    assert "spawn_graduated_workers_enabled" in src, (
        f"{plugin_file.name}: flag name 'spawn_graduated_workers_enabled' missing"
    )


def test_titan_params_flag_documented_default_off():
    """The flag MUST be documented as Default OFF in its comment block.

    The actual runtime value may be flipped to true on a Titan during
    active E2E testing (codified per session log) — but the documentation
    of the default must always say OFF, so any new deploy starts safe.
    """
    p = Path(__file__).parent.parent / "titan_plugin" / "titan_params.toml"
    txt = p.read_text()
    # Find the relevant comment block
    idx = txt.find("spawn_graduated_workers_enabled")
    assert idx >= 0, "spawn_graduated_workers_enabled flag not found"
    # The comment block lives in the ~30 lines BEFORE the assignment
    block_start = txt.rfind("\n\n", 0, idx) + 2
    comment_block = txt[block_start:idx]
    assert "Default OFF" in comment_block, (
        "Comment block for spawn_graduated_workers_enabled MUST document "
        "'Default OFF' so any new deploy starts safe regardless of the "
        "current testing-phase value."
    )


# ── Runtime: end-to-end via Guardian.register ──────────────────────────


def test_runtime_flag_off_keeps_workers_in_fork_mode():
    """When _spawn_grad=False, all 9 graduating workers register as fork."""
    from titan_plugin import bus as bus_mod
    from titan_plugin.guardian import Guardian, ModuleSpec

    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    # Direct simulation of the registration loop with flag off
    _spawn_grad = False
    for name in GRADUATING:
        g.register(ModuleSpec(
            name=name, layer="L3", entry_fn=lambda *a, **k: None,
            start_method="spawn" if _spawn_grad else "fork",
        ))
    for name in GRADUATING:
        assert g._modules[name].spec.start_method == "fork", (
            f"with flag off, '{name}' must be fork-mode"
        )


def test_runtime_flag_on_flips_workers_to_spawn():
    """When _spawn_grad=True, all 9 graduating workers register as spawn."""
    from titan_plugin import bus as bus_mod
    from titan_plugin.guardian import Guardian, ModuleSpec

    div = bus_mod.DivineBus(maxsize=100)
    g = Guardian(div)
    _spawn_grad = True
    for name in GRADUATING:
        g.register(ModuleSpec(
            name=name, layer="L3", entry_fn=lambda *a, **k: None,
            start_method="spawn" if _spawn_grad else "fork",
        ))
    for name in GRADUATING:
        assert g._modules[name].spec.start_method == "spawn", (
            f"with flag on, '{name}' must be spawn-mode"
        )
