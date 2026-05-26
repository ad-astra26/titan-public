#!/usr/bin/env python3
"""
Phase 6 chunk 6C — mechanical carve of titan_hcl/guardian.py into a package
titan_hcl/guardian_hcl/ with submodules per RFP §3C.3 + SPEC §11.B.4 (D-SPEC-135).

Why a script vs hand-edits: 2,813 LOC, ~40 methods, 25+ caller import sites.
Strict body-preserving extraction via line-range slicing (ast for boundaries,
text slicing for content — preserves comments/formatting/blank lines exactly).

Strategy:
1. Parse titan_hcl/guardian.py with ast → find line ranges of (a) module-level
   imports + constants, (b) ModuleState/ModuleSpec/ModuleInfo/ReloadState
   dataclasses, (c) Guardian methods, (d) trailing free functions
   (_module_wrapper, _append_meta_cgn_emission_log).
2. Write 4 destination files via verbatim line slicing:
   - module_registry.py: imports + dataclasses + _append_meta_cgn_emission_log
   - reload.py: GuardianReloadMixin class with reload_module +
     _reload_module_sync + _spawn_for_reload + _rollback_reload +
     _emit_reload_ack + _reload_result + _dispatch_reload_request +
     _reason_string_to_canonical (D-SPEC-50 surface)
   - dep_activation.py: GuardianDepActivationMixin with _activate_dependencies
     + _check_critical_dependencies (D-SPEC-90 §11.G.2.5)
   - core.py: class Guardian(GuardianReloadMixin, GuardianDepActivationMixin)
     with all remaining methods + _module_wrapper helper
3. Write __init__.py re-exporting Guardian, ModuleState, ModuleSpec, ModuleInfo,
   ReloadState, _module_wrapper (public surface; callers see the same names).
4. Rewrite all 'from titan_hcl.guardian' / 'titan_hcl.guardian.' import sites
   across the codebase → 'titan_hcl.guardian_hcl' / 'titan_hcl.guardian_hcl.'.
5. Delete titan_hcl/guardian.py.

Idempotent: re-running on already-carved tree is a no-op (detects sentinel).
Dry-run: --dry-run prints planned ops without writing.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Optional


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "titan_hcl" / "guardian.py"
PKG = REPO / "titan_hcl" / "guardian_hcl"

# Method routing — names on the LHS get moved to the listed file
RELOAD_METHODS = {
    "reload_module",
    "_reload_module_sync",
    "_spawn_for_reload",
    "_rollback_reload",
    "_emit_reload_ack",
    "_reload_result",
    "_dispatch_reload_request",
    "_reason_string_to_canonical",
}
DEP_ACTIVATION_METHODS = {
    "_activate_dependencies",
    "_check_critical_dependencies",
}
REGISTRY_CLASSES = {"ModuleState", "ModuleSpec", "ModuleInfo", "ReloadState"}
REGISTRY_FREE_FUNCS = {"_append_meta_cgn_emission_log"}
CORE_FREE_FUNCS = {"_module_wrapper"}


def parse_source(path: Path) -> tuple[str, ast.Module, list[str]]:
    text = path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)  # 0-indexed; ast lineno is 1-indexed
    return text, tree, lines


def node_lines(node: ast.AST, lines: list[str]) -> str:
    """Return source for a node by inclusive line-range slicing.

    Walks node.body if present to capture trailing comments INSIDE the
    block (ast end_lineno already covers them); for decorated nodes the
    first decorator's lineno is taken as the start to include decorators.
    """
    start = getattr(node, "lineno", 1)
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno
    return "".join(lines[start - 1:end])


def find_module_level_header(text: str, tree: ast.Module, lines: list[str]) -> str:
    """All lines BEFORE the first class/function definition: shebang,
    module docstring, imports, module-level constants. Preserves verbatim
    BUT rewrites `from .bus` → `from titan_hcl.bus` (the source was 1 level
    above the carve destination, so relative imports must be re-anchored)."""
    first_def = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first_def = node
            break
    if first_def is None:
        return text
    end = first_def.lineno
    if getattr(first_def, "decorator_list", None):
        end = min(d.lineno for d in first_def.decorator_list)
    header = "".join(lines[: end - 1])
    # Re-anchor 1-dot relative imports to absolute titan_hcl.* (new files live
    # one level deeper in titan_hcl/guardian_hcl/). 2+ dot relatives would also
    # break but guardian.py uses only single-dot relatives (verified manually).
    header = re.sub(
        r"^from \.(\w+) import",
        r"from titan_hcl.\1 import",
        header,
        flags=re.MULTILINE,
    )
    return header


def find_class(tree: ast.Module, name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_func(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def build_registry_module(text: str, tree: ast.Module, lines: list[str]) -> str:
    """Module containing the dataclasses + _append_meta_cgn_emission_log."""
    header = find_module_level_header(text, tree, lines)
    out = []
    out.append('"""\n')
    out.append("titan_hcl.guardian_hcl.module_registry — dataclasses + supervision-log helper.\n\n")
    out.append("Carved from titan_hcl/guardian.py by scripts/_phase6_carve_guardian.py per\n")
    out.append("SPEC §11.B.4 / D-SPEC-135 / v1.62.0. Dataclasses + free functions only —\n")
    out.append("no Guardian class behaviour here. Imports preserved verbatim from the\n")
    out.append("source so the dataclasses' field defaults / annotations resolve identically.\n")
    out.append('"""\n')
    out.append(header)
    for name in ("ModuleState", "ModuleSpec", "ModuleInfo", "ReloadState"):
        node = find_class(tree, name)
        if node is None:
            raise RuntimeError(f"missing class {name} in guardian.py")
        out.append("\n\n")
        out.append(node_lines(node, lines))
    for name in REGISTRY_FREE_FUNCS:
        node = find_func(tree, name)
        if node is not None:
            out.append("\n\n")
            out.append(node_lines(node, lines))
    return "".join(out)


def build_mixin_module(
    *,
    text: str,
    tree: ast.Module,
    lines: list[str],
    mixin_name: str,
    method_names: set[str],
    docstring_purpose: str,
    spec_refs: str,
) -> str:
    """Build a mixin class containing the named methods, body-verbatim."""
    guardian = find_class(tree, "Guardian")
    if guardian is None:
        raise RuntimeError("class Guardian not found")
    header = find_module_level_header(text, tree, lines)
    out = []
    out.append('"""\n')
    out.append(f"titan_hcl.guardian_hcl.{mixin_name.lower().replace('mixin', '')} — {docstring_purpose}.\n\n")
    out.append("Carved from titan_hcl/guardian.py by scripts/_phase6_carve_guardian.py per\n")
    out.append(f"SPEC §11.B.4 / D-SPEC-135 / v1.62.0. {spec_refs}\n\n")
    out.append("Mixed into class Guardian(GuardianReloadMixin, GuardianDepActivationMixin)\n")
    out.append("in core.py — `self` attributes (.bus, ._modules, ._reload_lock, etc.) come\n")
    out.append("from Guardian.__init__. Method bodies move verbatim — no logic change.\n")
    out.append('"""\n')
    out.append(header)
    # Cross-imports — the mixin methods carry type annotations referencing
    # the dataclasses defined in module_registry.py. Without this, parse-time
    # annotation evaluation in the carved file fails with NameError.
    out.append("\n")
    out.append("from titan_hcl.guardian_hcl.module_registry import (\n")
    out.append("    ModuleInfo,\n    ModuleSpec,\n    ModuleState,\n    ReloadState,\n)\n")
    out.append("\n")
    out.append(f"class {mixin_name}:\n")
    out.append(f'    """Mixin providing {docstring_purpose} — see {spec_refs}."""\n')

    # Re-indent method bodies under the mixin class
    found = []
    for child in guardian.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in method_names:
            method_src = node_lines(child, lines)
            # method_src is already indented at 4 spaces (inside Guardian class).
            # The new mixin class also indents at 4 spaces. So we keep the leading
            # indentation as-is.
            out.append("\n")
            out.append(method_src)
            found.append(child.name)

    missing = method_names - set(found)
    if missing:
        raise RuntimeError(f"methods not found in Guardian: {missing}")
    return "".join(out)


def build_core_module(
    *,
    text: str,
    tree: ast.Module,
    lines: list[str],
    extracted_methods: set[str],
) -> str:
    """Core module: class Guardian with all NON-extracted methods +
    _module_wrapper free function."""
    guardian = find_class(tree, "Guardian")
    if guardian is None:
        raise RuntimeError("class Guardian not found")
    header = find_module_level_header(text, tree, lines)
    out = []
    out.append('"""\n')
    out.append("titan_hcl.guardian_hcl.core — Guardian L1 supervisor class.\n\n")
    out.append("Carved from titan_hcl/guardian.py by scripts/_phase6_carve_guardian.py per\n")
    out.append("SPEC §11.B.4 / D-SPEC-135 / v1.62.0. Guardian = GuardianReloadMixin +\n")
    out.append("GuardianDepActivationMixin + remaining lifecycle methods. Dataclasses\n")
    out.append("(ModuleState/ModuleSpec/ModuleInfo/ReloadState) live in module_registry.\n")
    out.append("Public API surface frozen as bus messages per RFP §3C.3 6C.\n")
    out.append('"""\n')
    out.append(header)
    out.append("\n")
    out.append("from titan_hcl.guardian_hcl.module_registry import (\n")
    out.append("    ModuleState,\n    ModuleSpec,\n    ModuleInfo,\n    ReloadState,\n")
    out.append("    _append_meta_cgn_emission_log,\n)\n")
    out.append("from titan_hcl.guardian_hcl.reload import GuardianReloadMixin\n")
    out.append("from titan_hcl.guardian_hcl.dep_activation import GuardianDepActivationMixin\n")
    out.append("\n\n")

    # Build Guardian class with non-extracted methods only
    # Splice the class header + decorators + class-level docstring/assignments
    # then iterate body filtering out extracted methods.
    cls_start_line = guardian.lineno
    if guardian.decorator_list:
        cls_start_line = min(d.lineno for d in guardian.decorator_list)
    # Find the line where the body actually starts (after class header colon)
    body_start_line = guardian.body[0].lineno if guardian.body else guardian.end_lineno
    # Class header lines (decorators + class line + any blank/comments before body)
    class_header = "".join(lines[cls_start_line - 1:body_start_line - 1])
    # Replace `class Guardian:` with mixin-inherited form
    class_header = re.sub(
        r"^class Guardian\b[^:]*:",
        "class Guardian(GuardianReloadMixin, GuardianDepActivationMixin):",
        class_header,
        count=1,
        flags=re.MULTILINE,
    )
    out.append(class_header)

    # Body — emit each child node verbatim UNLESS its name is in extracted_methods
    last_end = body_start_line - 1
    for child in guardian.body:
        child_start = child.lineno
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.decorator_list:
            child_start = min(d.lineno for d in child.decorator_list)
        # Capture any inter-method content (blank lines, comments) between last_end and child_start
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in extracted_methods:
            # Preserve preceding inter-line content too? No — drop it cleanly.
            # The preceding blank/comments between previous-included-method and this
            # extracted method might belong to either; safest is to drop them when
            # the method itself is dropped, since they're typically section-divider
            # comments belonging to the extracted method.
            last_end = child.end_lineno
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            # Include inter-line content from last_end to (just before) this child
            if child_start - 1 > last_end:
                out.append("".join(lines[last_end:child_start - 1]))
            out.append(node_lines(child, lines))
            last_end = child.end_lineno
        else:
            # Other node types — include verbatim
            if child_start - 1 > last_end:
                out.append("".join(lines[last_end:child_start - 1]))
            out.append(node_lines(child, lines))
            last_end = child.end_lineno

    # Trailing content after class Guardian — module-level free functions
    cls_end = guardian.end_lineno
    # _module_wrapper + _append_meta_cgn_emission_log are at module level
    out.append("\n")
    for name in CORE_FREE_FUNCS:
        node = find_func(tree, name)
        if node is not None:
            out.append("\n")
            out.append(node_lines(node, lines))
    return "".join(out)


def build_init() -> str:
    return '''"""
titan_hcl.guardian_hcl — Guardian L1 supervisor package.

Carved from titan_hcl/guardian.py per SPEC §11.B.4 / D-SPEC-135 / v1.62.0.
Public surface re-exports preserved so existing call sites keep working:
    from titan_hcl.guardian_hcl import Guardian, ModuleSpec, ModuleInfo, ...

Internal organization:
    core.py             — class Guardian (lifecycle, monitor_tick, supervision)
    reload.py           — GuardianReloadMixin (D-SPEC-50 reload_module + 7-step seq)
    dep_activation.py   — GuardianDepActivationMixin (D-SPEC-90 §11.G.2.5)
    module_registry.py  — ModuleState / ModuleSpec / ModuleInfo / ReloadState

Public API per RFP §3C.3 6C is bus messages (MODULE_RELOAD_REQUEST/ACK,
MODULE_RESTART_REQUEST, SUPERVISION_*), not Python imports. The imports
below remain for legacy in-process callers until the standalone process
cutover (chunk 6F) replaces them with thin bus clients.
"""
from titan_hcl.guardian_hcl.core import Guardian, _module_wrapper
from titan_hcl.guardian_hcl.module_registry import (
    ModuleState,
    ModuleSpec,
    ModuleInfo,
    ReloadState,
    _append_meta_cgn_emission_log,
)
from titan_hcl.guardian_hcl.reload import GuardianReloadMixin
from titan_hcl.guardian_hcl.dep_activation import GuardianDepActivationMixin

__all__ = [
    "Guardian",
    "GuardianReloadMixin",
    "GuardianDepActivationMixin",
    "ModuleState",
    "ModuleSpec",
    "ModuleInfo",
    "ReloadState",
    "_module_wrapper",
    "_append_meta_cgn_emission_log",
]
'''


def rewrite_callers(dry_run: bool = False) -> list[Path]:
    """Sweep all *.py for 'titan_hcl.guardian' references and rewrite them to
    'titan_hcl.guardian_hcl' (preserves submodule paths like
    titan_hcl.guardian.os.kill → titan_hcl.guardian_hcl.os.kill)."""
    touched = []
    # Exclude paths
    excludes = {"__pycache__", ".git", "test_env", "node_modules"}
    # Also skip the carved files themselves (already use new path)
    skip_files = {
        REPO / "scripts" / "_phase6_carve_guardian.py",
        REPO / "titan_hcl" / "guardian_hcl" / "__init__.py",
        REPO / "titan_hcl" / "guardian_hcl" / "core.py",
        REPO / "titan_hcl" / "guardian_hcl" / "reload.py",
        REPO / "titan_hcl" / "guardian_hcl" / "dep_activation.py",
        REPO / "titan_hcl" / "guardian_hcl" / "module_registry.py",
        REPO / "titan_hcl" / "guardian.py",
    }
    # Match `titan_hcl.guardian` (word boundary on the right end so we don't
    # touch already-rewritten or unrelated tokens). Specifically need to NOT
    # match `titan_hcl.guardian_hcl`, `titan_hcl.guardian_state_publisher`,
    # `titan_hcl.guardian_state_specs`. Use negative lookahead on `_`.
    pattern = re.compile(r"\btitan_hcl\.guardian\b(?!_)")
    for py in REPO.rglob("*.py"):
        if any(part in excludes for part in py.parts):
            continue
        if py in skip_files:
            continue
        try:
            content = py.read_text()
        except UnicodeDecodeError:
            continue
        if "titan_hcl.guardian" not in content:
            continue
        new = pattern.sub("titan_hcl.guardian_hcl", content)
        if new != content:
            touched.append(py)
            if not dry_run:
                py.write_text(new)
    return touched


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv

    if PKG.exists() and any(PKG.iterdir()):
        print(f"package {PKG} already populated — refusing to overwrite (idempotency guard)")
        return 2

    if not SRC.exists():
        print(f"{SRC} not found — already carved?")
        return 2

    text, tree, lines = parse_source(SRC)

    extracted = RELOAD_METHODS | DEP_ACTIVATION_METHODS

    registry_src = build_registry_module(text, tree, lines)
    reload_src = build_mixin_module(
        text=text, tree=tree, lines=lines,
        mixin_name="GuardianReloadMixin",
        method_names=RELOAD_METHODS,
        docstring_purpose="D-SPEC-50 §8.3 per-module hot-reload (7-step sequence + pid-targeting + rollback)",
        spec_refs="See SPEC §8.3 + §11.B.3 + §11.B.3.1 + D-SPEC-50 / D-SPEC-93.",
    )
    dep_src = build_mixin_module(
        text=text, tree=tree, lines=lines,
        mixin_name="GuardianDepActivationMixin",
        method_names=DEP_ACTIVATION_METHODS,
        docstring_purpose="D-SPEC-90 §11.G.2.5 dep-activation + critical-deps gating",
        spec_refs="See SPEC §11.G + D-SPEC-90.",
    )
    core_src = build_core_module(text=text, tree=tree, lines=lines, extracted_methods=extracted)
    init_src = build_init()

    if dry_run:
        print(f"DRY RUN — would create:")
        print(f"  {PKG}/__init__.py            ({len(init_src.splitlines())} lines)")
        print(f"  {PKG}/module_registry.py     ({len(registry_src.splitlines())} lines)")
        print(f"  {PKG}/reload.py              ({len(reload_src.splitlines())} lines)")
        print(f"  {PKG}/dep_activation.py      ({len(dep_src.splitlines())} lines)")
        print(f"  {PKG}/core.py                ({len(core_src.splitlines())} lines)")
        print(f"  delete: {SRC}")
        touched = rewrite_callers(dry_run=True)
        print(f"  rewrite callers ({len(touched)}):")
        for p in touched:
            print(f"    {p.relative_to(REPO)}")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "__init__.py").write_text(init_src)
    (PKG / "module_registry.py").write_text(registry_src)
    (PKG / "reload.py").write_text(reload_src)
    (PKG / "dep_activation.py").write_text(dep_src)
    (PKG / "core.py").write_text(core_src)

    touched = rewrite_callers(dry_run=False)
    print(f"created {PKG}/ with 5 files")
    print(f"rewrote {len(touched)} caller import sites")

    SRC.unlink()
    print(f"deleted {SRC}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
