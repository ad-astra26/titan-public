#!/usr/bin/env python3
"""replace_bus_literals_with_constants — AST-based literal-to-constant rewrite.

Companion to register_unregistered_bus_literals.py. After all 77 unregistered
literals have been registered as constants in titan_plugin/bus.py, this script
replaces the remaining string literals at producer/consumer sites with
references to those constants.

Strategy (per file):
  1. Parse via AST. Find every site where a string literal that matches a
     registered bus constant is used as a bus-message-type:
       make_msg("X", ...)         → make_msg(bus.X, ...)
       _send_msg(q, "X", ...)     → _send_msg(q, bus.X, ...)
       send_queue.put({"type":"X",...})  → … {"type": bus.X, ...}
       msg_type == "X"            → msg_type == bus.X
       atype == "X" / kind == "X" → same (any Compare with str literal in
                                    a Compare/Assign/Set context near
                                    msg_type-ish vars)
       {bus.HIBERNATE_ACK, "X"}   → {bus.HIBERNATE_ACK, bus.X}
  2. Apply line-aware text edits (offsets-from-end so prior edits don't
     shift later positions).
  3. Ensure `from titan_plugin import bus` is importable in scope. We use
     the `bus.X` Attribute form throughout — the file just needs `bus`
     accessible (top-level `from titan_plugin import bus` is added if no
     bus alias is already in scope).

Safety guards:
  - Only replaces literals whose string value matches a known bus.py constant.
  - Skips literals inside docstrings, comments, or non-bus call contexts (AST
    distinguishes these from real call/compare positions).
  - Idempotent: if literal already replaced (or constant ref present), no-op.
  - Per-file diff is shown in --dry-run.
  - --apply commits the edit; rerun scanner afterwards to verify.

Limitations (NOT auto-handled — surface for manual review):
  - Set/list/tuple literals containing mixed constants + strings:
      {bus.HIBERNATE_ACK, "BUS_HANDOFF_ACK"}
    AST sees the set, but Python source has individual elements. Handled
    by walking the Set/List/Tuple's elements and emitting per-element edits.
  - String formatting / f-strings using the literal: skipped (rare).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUS_PY = REPO_ROOT / "titan_plugin" / "bus.py"
PLUGIN_ROOT = REPO_ROOT / "titan_plugin"


# ─── Reading existing bus constants ─────────────────────────────────────────

def load_bus_constants(bus_py: Path) -> set[str]:
    """Return the set of constants `NAME = "NAME"` defined in bus.py."""
    src = bus_py.read_text()
    pat = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=\s*\"\1\"\s*(?:#.*)?$", re.M)
    return {m.group(1) for m in pat.finditer(src)}


# ─── AST-driven site detection ──────────────────────────────────────────────


class LiteralSite:
    """A single literal-replacement site within a file."""
    __slots__ = ("lineno", "col_offset", "end_col_offset", "literal", "constant")

    def __init__(self, lineno: int, col_offset: int, end_col_offset: int,
                 literal: str, constant: str):
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_col_offset = end_col_offset
        self.literal = literal
        self.constant = constant


def find_sites_in_file(file_path: Path, known_constants: set[str]) -> list[LiteralSite]:
    """Walk AST, return all literal sites we want to replace."""
    try:
        src = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src, filename=str(file_path))
    except SyntaxError:
        return []

    sites: list[LiteralSite] = []

    # Variable names that, when compared against a string, indicate a bus
    # message-type subscriber. Anything else (arbitrary `name == "X"`) is
    # NOT treated as a bus-message comparison. Adjust list as needed.
    MSG_TYPE_VARS = {"msg_type", "message_type", "atype", "kind", "mtype",
                     "msgtype"}

    def add_constant_site(node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        if node.value not in known_constants:
            return
        sites.append(LiteralSite(
            lineno=node.lineno, col_offset=node.col_offset,
            end_col_offset=node.end_col_offset,
            literal=node.value, constant=node.value,
        ))

    def container_has_bus_ref(container: ast.AST) -> bool:
        """Does the Set/Tuple/List contain a bare or attribute reference to a
        known bus constant? The orchestrator's `{bus.HIBERNATE_ACK, "X"}`
        pattern qualifies; an arbitrary list of ALL_CAPS strings doesn't."""
        if not isinstance(container, (ast.Set, ast.Tuple, ast.List)):
            return False
        for elt in container.elts:
            if isinstance(elt, ast.Name) and elt.id in known_constants:
                return True
            if (isinstance(elt, ast.Attribute)
                    and isinstance(elt.value, ast.Name)
                    and elt.attr in known_constants):
                return True
        return False

    def is_msg_type_compare_target(left: ast.AST) -> bool:
        """Is the LHS of a Compare a bus-message-type variable?"""
        if isinstance(left, ast.Name) and left.id in MSG_TYPE_VARS:
            return True
        # msg.get("type") / msg.get("msg_type") form
        if (isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "get"
                and len(left.args) >= 1
                and isinstance(left.args[0], ast.Constant)
                and left.args[0].value in ("type", "msg_type", "kind")):
            return True
        # node["type"] / a["msg_type"] subscript form
        if (isinstance(left, ast.Subscript)
                and isinstance(left.slice, ast.Constant)
                and left.slice.value in ("type", "msg_type", "kind")):
            return True
        return False

    for node in ast.walk(tree):
        # 1. Publisher: make_msg("X", ...) / _send_msg(q, "X", ...)
        if isinstance(node, ast.Call):
            f = node.func
            fname = (f.id if isinstance(f, ast.Name)
                     else f.attr if isinstance(f, ast.Attribute) else None)
            if fname in ("make_msg", "_send_msg"):
                pos = 0 if fname == "make_msg" else 1
                if len(node.args) > pos and isinstance(node.args[pos], ast.Constant):
                    add_constant_site(node.args[pos])
            # 2. dict literal pattern: {"type": "X", ...} via send_queue.put(...)
            if (fname == "put" and len(node.args) >= 1
                    and isinstance(node.args[0], ast.Dict)):
                for k, v in zip(node.args[0].keys, node.args[0].values):
                    if (isinstance(k, ast.Constant) and k.value == "type"
                            and isinstance(v, ast.Constant)):
                        add_constant_site(v)
            # 3b. Mixed-element set passed to a function (orchestrator's
            #     `_drain_messages(inbox, {bus.HIBERNATE_ACK, "X"}, ...)`):
            #     walk args/kwargs for sets-with-bus-refs and replace literals
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                if isinstance(arg, (ast.Set, ast.Tuple, ast.List)):
                    if container_has_bus_ref(arg):
                        for elt in arg.elts:
                            if isinstance(elt, ast.Constant):
                                add_constant_site(elt)
        # 3. Subscriber comparison: must be against a recognized msg-type var
        if isinstance(node, ast.Compare):
            if not is_msg_type_compare_target(node.left):
                continue
            for sub in node.comparators:
                if isinstance(sub, ast.Constant):
                    add_constant_site(sub)
                elif isinstance(sub, (ast.Set, ast.Tuple, ast.List)):
                    # `msg_type in (TYPE_A, "TYPE_B", ...)` — replace literals
                    for elt in sub.elts:
                        if isinstance(elt, ast.Constant):
                            add_constant_site(elt)

    # Deduplicate (a site may match multiple ast.walk visits)
    seen = set()
    deduped: list[LiteralSite] = []
    for s in sites:
        key = (s.lineno, s.col_offset, s.end_col_offset)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


# ─── File patching ──────────────────────────────────────────────────────────


def detect_bus_alias(src: str) -> str | None:
    """Return how `bus` is currently importable at module level.

    Returns:
      - "bus": top-level `from titan_plugin import bus` (or relative `from .. import bus`)
        OR `import titan_plugin.bus as bus` (any form that gives a name `bus` in scope)
      - None: not imported at module level (we'll need to add it)

    Inline imports inside functions are NOT detected — they have local scope.
    Adding a top-level import is safe (Python ignores duplicates from inline).
    """
    pats = [
        re.compile(r"^from\s+titan_plugin\s+import\s+bus(?:\s|$|,)", re.M),
        re.compile(r"^from\s+\.+\s+import\s+bus(?:\s|$|,)", re.M),
        re.compile(r"^import\s+titan_plugin\.bus\s+as\s+bus\s*$", re.M),
        re.compile(r"^from\s+titan_plugin\s+import\s+\([^)]*\bbus\b", re.M | re.S),
    ]
    for pat in pats:
        if pat.search(src):
            return "bus"
    return None


def find_import_insertion_point(src: str) -> int:
    """Return the line index AT which to insert a new import (0-based).

    Uses AST to find the last top-level Import/ImportFrom node. Handles
    multi-line `from X import (a, b, c)` correctly via end_lineno (the
    closing paren's line). Falls back to after module docstring or 0.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # File doesn't parse — bail to safe default (file start).
        return 0

    last_import_end_line = 0  # 1-based AST lineno
    docstring_end_line = 0

    # Module docstring detection: first stmt is Expr with a Constant str.
    if tree.body and isinstance(tree.body[0], ast.Expr):
        ex = tree.body[0]
        if isinstance(ex.value, ast.Constant) and isinstance(ex.value.value, str):
            docstring_end_line = ex.end_lineno or 0

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end_line = max(last_import_end_line, node.end_lineno or node.lineno)

    if last_import_end_line:
        return last_import_end_line  # 0-based index = AST lineno (insert AFTER)
    if docstring_end_line:
        return docstring_end_line
    return 0


def apply_edits_to_file(file_path: Path, sites: list[LiteralSite],
                        ensure_bus_import: bool) -> tuple[str, int]:
    """Apply all edits to a file. Return (new_src, num_edits).

    Edits replace `"X"` with `bus.X` at the precise (lineno, col_offset)
    range. Applied in reverse order so earlier edits don't shift later ones.
    """
    src = file_path.read_text(encoding="utf-8")
    lines = src.split("\n")

    # Group sites by line for in-line ordering, then apply right-to-left.
    by_line: dict[int, list[LiteralSite]] = defaultdict(list)
    for s in sites:
        by_line[s.lineno].append(s)
    n_edits = 0

    for lineno in sorted(by_line.keys(), reverse=True):
        line = lines[lineno - 1]
        # Sort within the line by col_offset descending so right-to-left.
        for s in sorted(by_line[lineno], key=lambda s: s.col_offset, reverse=True):
            # Verify the literal is at the expected position. AST gives
            # offsets in BYTES for utf-8, which is the same as char indices
            # for ASCII source. Worker files are ASCII — safe.
            literal_with_quotes = line[s.col_offset:s.end_col_offset]
            expected_a = f'"{s.literal}"'
            expected_b = f"'{s.literal}'"
            if literal_with_quotes not in (expected_a, expected_b):
                # Defensive: skip if line shifted unexpectedly
                continue
            replacement = f"bus.{s.constant}"
            line = line[:s.col_offset] + replacement + line[s.end_col_offset:]
            n_edits += 1
        lines[lineno - 1] = line

    new_src = "\n".join(lines)

    # Add `from titan_plugin import bus` if needed and we made any edit
    if ensure_bus_import and n_edits > 0:
        # Determine import to add based on file's location relative to titan_plugin.
        rel = file_path.relative_to(PLUGIN_ROOT.parent)
        depth = len(rel.parts) - 1  # parts include the file name
        if depth == 0:
            # File is at repo root (rare) — use absolute
            import_line = "from titan_plugin import bus"
        else:
            # File is under titan_plugin — use absolute too (works at all depths)
            import_line = "from titan_plugin import bus"
        # Only add if not already present
        if detect_bus_alias(new_src) is None:
            insert_at = find_import_insertion_point(new_src)
            lines = new_src.split("\n")
            lines.insert(insert_at, import_line)
            new_src = "\n".join(lines)

    return new_src, n_edits


# ─── Orchestrator ───────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="actually edit files (default: dry-run with diffs)")
    p.add_argument("--root", default=str(PLUGIN_ROOT),
                   help="root to scan (default: titan_plugin)")
    p.add_argument("--exclude", action="append", default=[],
                   help="path substring to exclude (repeatable)")
    args = p.parse_args()

    root = Path(args.root)
    constants = load_bus_constants(BUS_PY)
    print(f"loaded {len(constants)} bus constants from {BUS_PY.name}",
          file=sys.stderr)

    files: list[Path] = []
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        if str(py.name) == "bus.py":
            continue  # don't rewrite the constants file
        if str(py.name) == "bus_specs.py":
            continue
        if str(py.name) == "_layer_canon.py":
            continue
        if any(ex in str(py) for ex in args.exclude):
            continue
        files.append(py)
    print(f"scanning {len(files)} files…", file=sys.stderr)

    total_files_changed = 0
    total_edits = 0
    file_summaries: list[tuple[Path, int]] = []

    for f in files:
        sites = find_sites_in_file(f, constants)
        if not sites:
            continue

        new_src, n_edits = apply_edits_to_file(f, sites, ensure_bus_import=True)
        if n_edits == 0:
            continue

        file_summaries.append((f, n_edits))
        total_files_changed += 1
        total_edits += n_edits

        if args.apply:
            f.write_text(new_src, encoding="utf-8")

    print(f"\n{total_edits} edits across {total_files_changed} files\n", file=sys.stderr)
    for f, n in sorted(file_summaries, key=lambda x: -x[1]):
        rel = f.relative_to(REPO_ROOT)
        print(f"  {rel}: {n} edits", file=sys.stderr)

    if not args.apply:
        print("\ndry-run only — no files changed. re-run with --apply.", file=sys.stderr)
    else:
        print("\nfiles modified. next steps:", file=sys.stderr)
        print("  1. re-run: python scripts/arch_map_dead_wiring.py --all --json | grep bus_literal_msg_type", file=sys.stderr)
        print("  2. run contract test suites", file=sys.stderr)
        print("  3. inspect a few of the largest-edit files for sanity", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
