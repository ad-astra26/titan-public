#!/usr/bin/env python3
"""scripts/_phase11_11g_apply_dep_matrix.py — Phase 11 §11.I.8 / Chunk 11G.

One-shot tool that applies the §3H.10 dep matrix to
`titan_hcl/module_catalog.py` by inserting `boot_priority=` and
`dependencies=` kwargs into each ModuleSpec block per the canonical mapping.

Idempotent: re-running on an already-migrated catalog is a no-op (the
inserter detects an existing `boot_priority=` / `dependencies=` kwarg and
preserves it).

NOT a runtime dependency. Kept under scripts/ so the matrix application
is auditable + reproducible per RFP §3H.6 (Maker greenlight gate for
11G merge).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Per RFP §3H.10. None ⇒ default ("mandatory") + no MODULE deps to add.
# Boot-priority is "mandatory" / "post_boot" / "lazy"; dep list is a list
# of module names (each becomes _mod_dep(name) in the generated kwarg).
MATRIX: dict[str, tuple[str | None, list[str]]] = {
    # ── MANDATORY (Phase A — gates fleet_ready) ─────────────────────
    "imw":               ("mandatory", []),
    # body / mind are L1 trinity Rust daemons per §3H.10.A but the
    # current Python catalog still registers Python wrappers for them
    # (kernel-rs already spawns the Rust binary). Mark MANDATORY so
    # the wrapper joins Phase A (matches today's autostart behaviour).
    "body":              ("mandatory", []),
    "mind":              ("mandatory", []),
    "output_verifier":   ("mandatory", ["timechain"]),
    "warning_monitor":   ("mandatory", []),
    "health_monitor":    ("mandatory", []),
    "memory":            ("mandatory", []),       # promoted from lazy=True
    "agno_worker":       ("mandatory", ["memory", "output_verifier", "timechain"]),
    "cognitive_worker":  ("mandatory", ["memory"]),
    "observatory":       ("mandatory", []),
    "sovereignty":       ("mandatory", ["timechain"]),
    "timechain":         ("mandatory", []),
    "api":               ("mandatory", ["agno_worker", "memory"]),
    # ── OPTIONAL_POST_BOOT (Phase B — background) ────────────────────
    "reflex":            ("post_boot", []),
    "agency_worker":     ("post_boot", []),
    "observatory_writer":("post_boot", []),
    "recorder":          ("post_boot", []),       # Phase 12 candidate (Rust L1)
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
    "cgn":               ("post_boot", []),       # Phase 12 candidate (Rust L1)
    "knowledge":         ("post_boot", []),
    "backup":            ("post_boot", ["timechain"]),
    "emot_cgn":          ("post_boot", []),       # Phase 12 candidate (Rust L1)
}


def _format_dependencies_kwarg(names: list[str], indent: str) -> str:
    """Return the `dependencies=[_mod_dep(...), ...]` kwarg string.

    Empty list ⇒ skip the kwarg entirely (ModuleSpec default).
    `indent` is the leading whitespace of each kwarg line inside the
    enclosing ModuleSpec block (4 spaces deeper than the `guardian.register`
    line itself).
    """
    if not names:
        return ""
    lines = [
        f"{indent}# Phase 11 §11.I.8 / Chunk 11G — §3H.10 dep matrix:",
        f"{indent}dependencies=[",
    ]
    for n in names:
        lines.append(f"{indent}    _mod_dep({n!r}),")
    lines.append(f"{indent}],")
    return "\n".join(lines)


def _apply_block(block_text: str, module_name: str, kwarg_indent: str) -> tuple[str, bool]:
    """Insert boot_priority + dependencies into one ModuleSpec block.

    Returns (new_block_text, changed_flag).
    """
    if module_name not in MATRIX:
        return block_text, False
    boot_priority, deps = MATRIX[module_name]
    if boot_priority is None and not deps:
        return block_text, False

    # Skip if already migrated (re-run idempotency).
    if "boot_priority=" in block_text:
        return block_text, False

    insert_lines: list[str] = []
    if boot_priority is not None:
        insert_lines.append(
            f'{kwarg_indent}# Phase 11 §11.I.8 / Chunk 11G — §3H.10 boot priority.'
        )
        insert_lines.append(
            f'{kwarg_indent}boot_priority="{boot_priority}",'
        )
    if deps:
        if "dependencies=" in block_text:
            print(
                f"  [SKIP DEPS] {module_name}: dependencies= already present; "
                f"matrix specifies {deps} — preserving existing block",
                file=sys.stderr,
            )
        else:
            insert_lines.append(_format_dependencies_kwarg(deps, kwarg_indent))

    # The block ends with `<base_indent>))` on its own line; insert new
    # kwargs right before that closing line so trailing-comma rules stay
    # clean. base_indent is one nesting level shallower than kwarg_indent.
    base_indent = kwarg_indent[:-4] if len(kwarg_indent) >= 4 else ""
    close_pattern = re.compile(
        rf"(\n)({re.escape(base_indent)}\)\)\s*)$", re.MULTILINE)
    new_block, n = close_pattern.subn(
        lambda m: "\n" + "\n".join(insert_lines) + "\n" + m.group(2),
        block_text, count=1,
    )
    if n == 0:
        return block_text, False
    return new_block, True


def _migrate_file(src_text: str) -> tuple[str, int]:
    """Walk every `guardian.register(ModuleSpec(...))` block (at any
    indent level) and apply the §3H.10 matrix entry. Returns
    (new_text, changed_count)."""
    # Each block opens with `<indent>guardian.register(ModuleSpec(` and
    # closes with `<indent>))` at the SAME indent (the body kwargs are
    # indented +4 deeper). Capture the leading indent so nested
    # conditional blocks (8 / 12 spaces) are handled correctly.
    pattern = re.compile(
        r"(?P<indent>[ \t]*)guardian\.register\(ModuleSpec\(\n"
        r"(?:.*?\n)+?"
        r"(?P=indent)\)\)",
        re.MULTILINE,
    )
    name_pattern = re.compile(r'^\s+name=["\'](\w+)["\']', re.MULTILINE)
    total_changed = 0
    spans: list[tuple[int, int, str]] = []
    for m in pattern.finditer(src_text):
        block = m.group(0)
        indent = m.group("indent")
        kwarg_indent = indent + "    "
        nm = name_pattern.search(block)
        if not nm:
            continue
        module_name = nm.group(1)
        new_block, changed = _apply_block(block, module_name, kwarg_indent)
        if changed:
            spans.append((m.start(), m.end(), new_block))
            total_changed += 1
    out = src_text
    for start, end, new_block in reversed(spans):
        out = out[:start] + new_block + out[end:]
    return out, total_changed


def main() -> int:
    path = Path("titan_hcl/module_catalog.py")
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    original = path.read_text()
    migrated, changed = _migrate_file(original)
    if changed == 0:
        print("No changes — catalog is already migrated.")
        return 0
    path.write_text(migrated)
    print(f"Migrated {changed} ModuleSpec blocks per §3H.10 matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
