#!/usr/bin/env python3
"""scripts/_phase11_11i_apply_error_envelope.py — Phase 11 §11.I.4 / Chunk 11I.

One-shot tool that adds `@with_error_envelope(module_name=..., subsystem="entry")`
to every worker-entry function under `titan_hcl/modules/*.py` plus
`titan_hcl/persistence_entry.py`. Idempotent — re-running on an
already-migrated file is a no-op (skips files that already import
with_error_envelope OR already have the decorator on the entry fn).

Per RFP §3H.2 11I:
  Each entry function + each significant helper gets the decorator.
  ~10 LOC × 44 + per-module migration test. Decorator is opt-in per
  function; migration incremental; one test per migrated worker.

Kept under scripts/ for auditability per RFP §3H.6 (Maker greenlight
gate). NOT a runtime dependency.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Map worker entry-function file → (entry_fn_name, module_catalog_name).
# module_catalog_name MUST match `name=` in the ModuleSpec registration
# in titan_hcl/module_catalog.py — that's the value that becomes
# `ModuleError.module_name` in every typed-error envelope.
ENTRY_MAP: dict[str, tuple[str, str]] = {
    "titan_hcl/modules/agency_worker.py":                    ("agency_worker_main", "agency_worker"),
    "titan_hcl/modules/agno_worker.py":                      ("agno_worker_main", "agno_worker"),
    "titan_hcl/modules/backup_worker.py":                    ("backup_worker_main", "backup"),
    "titan_hcl/modules/body_worker.py":                      ("body_worker_main", "body"),
    "titan_hcl/modules/cgn_worker.py":                       ("cgn_worker_main", "cgn"),
    "titan_hcl/modules/cognitive_worker.py":                 ("cognitive_worker_main", "cognitive_worker"),
    "titan_hcl/modules/corrective_events_persistence_worker.py":
        ("corrective_events_persistence_worker_main", "corrective_events_persistence"),
    "titan_hcl/modules/dream_state_worker.py":               ("dream_state_worker_main", "dream_state"),
    "titan_hcl/modules/emot_cgn_worker.py":                  ("emot_cgn_worker_main", "emot_cgn"),
    "titan_hcl/modules/expression_worker.py":                ("expression_worker_main", "expression_worker"),
    "titan_hcl/modules/health_monitor_worker.py":            ("health_monitor_worker_main", "health_monitor"),
    "titan_hcl/modules/hormonal_worker.py":                  ("hormonal_worker_main", "hormonal_module"),
    "titan_hcl/modules/interface_advisor_worker.py":         ("interface_advisor_worker_main", "interface_advisor"),
    "titan_hcl/modules/journey_persistence_worker.py":       ("journey_persistence_worker_main", "journey_persistence"),
    "titan_hcl/modules/knowledge_worker.py":                 ("knowledge_worker_main", "knowledge"),
    "titan_hcl/modules/language_worker.py":                  ("language_worker_main", "language"),
    "titan_hcl/modules/life_force_worker.py":                ("life_force_worker_main", "life_force"),
    "titan_hcl/modules/llm_worker.py":                       ("llm_worker_main", "llm"),
    "titan_hcl/modules/media_worker.py":                     ("media_worker_main", "media"),
    "titan_hcl/modules/meditation_worker.py":                ("meditation_worker_main", "meditation"),
    "titan_hcl/modules/memory_worker.py":                    ("memory_worker_main", "memory"),
    "titan_hcl/modules/meta_teacher_worker.py":              ("meta_teacher_worker_main", "meta_teacher"),
    "titan_hcl/modules/metabolism_worker.py":                ("metabolism_worker_main", "metabolism"),
    "titan_hcl/modules/mind_worker.py":                      ("mind_worker_main", "mind"),
    "titan_hcl/modules/neuromod_worker.py":                  ("neuromod_worker_main", "neuromod_module"),
    "titan_hcl/modules/ns_worker.py":                        ("ns_worker_main", "ns_module"),
    "titan_hcl/modules/observatory_worker.py":               ("observatory_worker_main", "observatory"),
    "titan_hcl/modules/outer_interface_worker.py":           ("outer_interface_worker_main", "outer_interface_worker"),
    "titan_hcl/modules/output_verifier_worker.py":           ("output_verifier_worker_main", "output_verifier"),
    "titan_hcl/modules/recorder_worker.py":                  ("recorder_worker_main", "recorder"),
    "titan_hcl/modules/reflex_worker.py":                    ("reflex_worker_main", "reflex"),
    "titan_hcl/modules/self_reflection_worker.py":           ("self_reflection_worker_main", "self_reflection_worker"),
    "titan_hcl/modules/social_graph_worker.py":              ("social_graph_worker_main", "social_graph"),
    "titan_hcl/modules/social_worker.py":                    ("social_worker_main", "social_worker"),
    "titan_hcl/modules/sovereignty_worker.py":               ("sovereignty_worker_main", "sovereignty"),
    "titan_hcl/modules/studio_worker.py":                    ("studio_worker_main", "studio"),
    "titan_hcl/modules/synthesis_worker.py":                 ("synthesis_worker_main", "synthesis"),
    "titan_hcl/modules/timechain_worker.py":                 ("timechain_worker_main", "timechain"),
    "titan_hcl/modules/warning_monitor_worker.py":           ("warning_monitor_worker_main", "warning_monitor"),
    "titan_hcl/persistence_entry.py":                        ("imw_main", "imw"),
}


IMPORT_LINE = (
    "from titan_hcl.core.module_error_handler import with_error_envelope"
)
# Severity is imported under a unique alias to avoid any clash with
# pre-existing identifiers (e.g. body_worker has a local `class Severity`
# for its own substrate-severity enum). Decorator below references the
# alias by its unique name.
SEVERITY_IMPORT_LINE = (
    "from titan_hcl.errors import Severity as _phase11_sev"
)
DECORATOR_SEVERITY_TOKEN = "_phase11_sev"


def _migrate_one(path: Path, entry_fn: str, module_name: str) -> tuple[bool, str]:
    """Add the imports + decorator to `path`. Returns (changed, reason).

    Uses `ast` to find a safe insertion point for the new imports (avoids
    landing inside a multi-line `from X import (` block). The decorator is
    inserted above the TOP-LEVEL `def <entry_fn>` only (skips nested defs
    by checking ast.Module.body directly).
    """
    import ast as _ast

    if not path.exists():
        return False, "file_missing"
    src = path.read_text()

    if "with_error_envelope" in src:
        return False, "already_decorated_or_imported"

    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return False, "source_unparseable"

    # 1. Find the LAST top-level Import / ImportFrom and use its
    #    end_lineno (1-based, inclusive) as the insertion point for our
    #    two new imports. This is multi-line-import-safe.
    last_import_end_line = 0
    for node in tree.body:
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if end > last_import_end_line:
                last_import_end_line = end
    # 2. Find the TOP-LEVEL FunctionDef / AsyncFunctionDef matching entry_fn.
    entry_node: _ast.AST | None = None
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == entry_fn:
                entry_node = node
                break
    if entry_node is None:
        return False, "entry_fn_not_found"

    lines = src.splitlines(keepends=True)

    # Insert two unique-name imports right after the last existing import.
    # If no imports were found, insert at line 0.
    inserts_at = last_import_end_line  # 1-based line END; insert AFTER this line
    new_import_lines = [
        IMPORT_LINE + "\n",
        SEVERITY_IMPORT_LINE + "\n",
    ]
    lines[inserts_at:inserts_at] = new_import_lines

    # The entry_fn's `def` line moves down by len(new_import_lines).
    decorator_line_idx = entry_node.lineno - 1 + len(new_import_lines)
    # Skip up past any existing decorators (entry_node.decorator_list).
    if entry_node.decorator_list:
        first_decorator_line = min(d.lineno for d in entry_node.decorator_list)
        decorator_line_idx = first_decorator_line - 1 + len(new_import_lines)
    indent = ""  # entry fns are always module-top-level (col_offset=0)
    decorator_line = (
        f'{indent}@with_error_envelope(module_name="{module_name}", '
        f'subsystem="entry", severity={DECORATOR_SEVERITY_TOKEN}.FATAL)\n'
    )
    lines.insert(decorator_line_idx, decorator_line)

    path.write_text("".join(lines))
    return True, "migrated"


def main() -> int:
    changed_count = 0
    skipped_count = 0
    missing: list[str] = []
    for path_str, (entry_fn, module_name) in ENTRY_MAP.items():
        path = Path(path_str)
        changed, reason = _migrate_one(path, entry_fn, module_name)
        if changed:
            changed_count += 1
            print(f"  [MIGRATED] {path_str} ({entry_fn} → module_name={module_name!r})")
        else:
            skipped_count += 1
            if reason == "file_missing":
                missing.append(path_str)
            else:
                print(f"  [SKIP {reason}] {path_str}", file=sys.stderr)
    print(f"Migrated {changed_count} entry-fns; skipped {skipped_count}.")
    if missing:
        print(f"MISSING FILES (need ENTRY_MAP correction): {missing}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
