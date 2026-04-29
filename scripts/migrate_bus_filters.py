#!/usr/bin/env python3
"""Mechanical migration helper for `bus.subscribe(name) → subscribe(name, types=[...])`.

Per BUG-BUS-DST-ALL-FLOODS-EVERY-SUBSCRIBER-20260429: a subscriber's queue
receives every dst="all" broadcast even if the subscriber only handles a
small whitelist. The mechanism for declared filters landed in commit
`6520a34d`; 3 hot subscribers were migrated in `ef7c3ba7`. This script
extracts CANDIDATE filter sets for the remaining 40 subscribers by static
analysis.

What it does:
  1. Find every `bus.subscribe(<name>)` / `self.bus.subscribe(<name>)`
     call site that does NOT already pass `types=`.
  2. For each, identify the queue variable it returns (or the attribute
     it's assigned to, e.g. `self._foo_queue`).
  3. Trace the queue var's drain site by scanning the same file for
     `<queue>.get_nowait()` / `<queue>.get()` / `bus.drain(<queue>, ...)`.
  4. In the drain block, extract the msg_type whitelist via three patterns:
       - `if msg.get("type") == X:` / `elif msg.get("type") == Y:`
       - `if msg_type == X:` / `elif msg_type == Y:`
       - `if msg_type in {X, Y, Z}:` (set-literal style, e.g. V4_EVENT_TYPES)
       - `if msg_type not in V4_EVENT_TYPES: continue` (existing whitelist
         identifier — script names it for human review)
  5. Output a per-subscriber CANDIDATE report: name, file:line, queue var,
     drain site, extracted whitelist (or "UNKNOWN — needs human review").

What it does NOT do:
  - Auto-edit source files. Human review of each candidate is required
    before landing — false negatives (missing types) silently filter
    real msgs; false positives (extra types) just preserve current
    behavior. Both are detectable in the migration regression test
    pattern from `tests/test_bus_filter_migrations.py`.
  - Trace queue vars across function boundaries (e.g. `self._q` set in
    __init__ and drained in a method 500 lines later). The simplest
    same-file/same-class trace is implemented; cross-class traces
    flagged for manual review.

Usage:
    python scripts/migrate_bus_filters.py [--json]

Output: human-readable table of CANDIDATE filters by default, JSON with
--json for machine-parseable input to a follow-up auto-edit script.

Exit code: 0 (always — this is a reporting tool, not a CI gate).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ──────────────────────────────────────────────────────────


def find_subscribe_call_sites(root: Path) -> list[dict]:
    """Find every bus.subscribe / self.bus.subscribe call WITHOUT types=.

    Returns list of {file, line, name, has_types, has_reply_only,
    assigned_to, parent_class}.
    """
    out = []
    for py in root.rglob("*.py"):
        if "test_env" in py.parts or ".git" in py.parts:
            continue
        if py.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(py.read_text(), filename=str(py))
        except Exception:
            continue
        # Track current-class context for accurate `self._x = ...` resolution
        class_stack: list[str] = []

        class V(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef):
                class_stack.append(node.name)
                self.generic_visit(node)
                class_stack.pop()

            def visit_Call(self, node: ast.Call):
                self.generic_visit(node)
                # Match bus.subscribe / self.bus.subscribe / <something>.bus.subscribe
                if not (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "subscribe"):
                    return
                # Verify the target attribute chain ends in `bus`
                target = node.func.value
                target_str = ast.unparse(target)
                if not (target_str == "bus" or target_str.endswith(".bus")):
                    return
                # First positional arg is the name
                if not node.args:
                    return
                name_node = node.args[0]
                if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                    name = name_node.value
                else:
                    name = "<dynamic>"
                # Parse kwargs
                has_types = any(kw.arg == "types" for kw in node.keywords)
                reply_only = next(
                    (kw for kw in node.keywords if kw.arg == "reply_only"),
                    None,
                )
                reply_only_value = (
                    ast.unparse(reply_only.value) if reply_only else "False"
                )
                out.append({
                    "file": str(py.relative_to(REPO_ROOT)),
                    "line": node.lineno,
                    "name": name,
                    "has_types": has_types,
                    "reply_only_expr": reply_only_value,
                    "parent_class": class_stack[-1] if class_stack else None,
                })

        V().visit(tree)
    return out


# Patterns inside drain blocks
_RE_TYPE_EQ = re.compile(
    r'(?:if|elif)\s+(?:msg\.get\("type"\)|msg_type|m\.get\("type"\))\s*'
    r'==\s*(?:bus\.)?([A-Z_][A-Z0-9_]*)'
)
_RE_TYPE_IN_SET = re.compile(
    r'(?:if|elif)\s+(?:msg\.get\("type"\)|msg_type|m\.get\("type"\))\s+'
    r'(?:not\s+)?in\s+(?:\{|\[)([^}\]]+)(?:\}|\])'
)
_RE_TYPE_IN_NAMED = re.compile(
    r'(?:if|elif)\s+(?:msg\.get\("type"\)|msg_type|m\.get\("type"\))\s+'
    r'(?:not\s+)?in\s+([A-Z_][A-Z0-9_]*)'
)


def _extract_function_source_at_line(file_path: Path, lineno: int) -> str | None:
    """Find the AST function/method enclosing `lineno` and return its source.

    Returns None if no enclosing function found (module-level subscribe).
    Per-function scoping prevents the whole-file regex from picking up
    drain patterns from sibling functions for the same file.
    """
    try:
        src = file_path.read_text()
        tree = ast.parse(src, filename=str(file_path))
    except Exception:
        return None
    src_lines = src.splitlines(keepends=True)
    # Find the innermost function that contains lineno
    candidates: list[tuple[ast.AST, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if start <= lineno <= end:
                candidates.append((node, start, end))
    if not candidates:
        return None
    # Pick the innermost (largest start = closest to lineno)
    candidates.sort(key=lambda t: t[1])
    inner = candidates[-1]
    start, end = inner[1], inner[2]
    return "".join(src_lines[start - 1:end])


def _extract_class_source_at_line(file_path: Path, lineno: int) -> str | None:
    """Return the source of the AST ClassDef enclosing `lineno`.

    Used as a fallback when `self._foo_queue = bus.subscribe(...)` is seen
    in __init__ or start() but the actual `if msg_type ==` patterns live
    in a separate method (e.g. `_process_bus_message`) called from a
    listener loop. Scanning the whole class catches the cross-method
    handler pattern without bleeding into other classes in the file.
    """
    try:
        src = file_path.read_text()
        tree = ast.parse(src, filename=str(file_path))
    except Exception:
        return None
    src_lines = src.splitlines(keepends=True)
    candidates: list[tuple[ast.ClassDef, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if start <= lineno <= end:
                candidates.append((node, start, end))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1])
    inner = candidates[-1]
    return "".join(src_lines[inner[1] - 1:inner[2]])


def extract_whitelist_from_source(file_path: Path,
                                  subscribe_lineno: int | None = None) -> dict:
    """Extract the msg_type whitelist from a file's drain code.

    If subscribe_lineno is given, scope the search to the AST function
    enclosing that line — prevents over-broadening when the same file
    contains many drain loops for different queues (e.g. core/plugin.py
    has agency, rl_stats, sovereignty, v4_bridge, chat_handler all in
    different methods).

    If a queue variable is assigned to `self._foo_queue` in the
    enclosing function, additionally scan the whole file for the
    drain site that uses `self._foo_queue` — covers the common pattern
    where subscribe happens in __init__ but drain is in a separate
    method.

    Returns dict with:
      - exact_types: list[str] of literal symbols found in == comparisons
      - set_literals: list[str] of types found in `in {A, B, C}` patterns
      - named_sets: list[str] of identifier names referenced (e.g.
        V4_EVENT_TYPES) — human must inspect these
      - confident: bool — True if all extractions are literal symbols
    """
    src_full = file_path.read_text()
    if subscribe_lineno is not None:
        scoped = _extract_function_source_at_line(file_path, subscribe_lineno)
        # Try to find a `self._<x>_queue = bus.subscribe(...)` assignment
        # in the scoped function — if present, extend search to file-wide
        # references of that attribute name (drain may be in another method).
        if scoped:
            attr_match = re.search(
                r"(self\._\w*queue\w*)\s*=\s*(?:[\w\.]*)?bus\.subscribe",
                scoped,
            )
            attr_name = attr_match.group(1) if attr_match else None
            sources_to_scan: list[str] = [scoped]
            if attr_name:
                # Find every method that references self._foo_queue and
                # include its source.
                tree = ast.parse(src_full, filename=str(file_path))
                src_lines = src_full.splitlines(keepends=True)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                        block = "".join(
                            src_lines[node.lineno - 1:
                                      getattr(node, "end_lineno", node.lineno)]
                        )
                        if attr_name in block and node.lineno != subscribe_lineno:
                            sources_to_scan.append(block)
                # Even after attr-tracing, the actual `if msg_type ==`
                # patterns may live in a method called BY the listener
                # loop (e.g. _process_bus_message). Always also scan the
                # enclosing class body — it catches the cross-method
                # handler pattern without bleeding to other classes.
                klass = _extract_class_source_at_line(
                    file_path, subscribe_lineno)
                if klass:
                    sources_to_scan.append(klass)
            else:
                # No self.attr — likely a local var like `bridge_queue =
                # self.bus.subscribe(...)`. Drain may still be in another
                # method of the same class. Add class scope as fallback.
                klass = _extract_class_source_at_line(
                    file_path, subscribe_lineno)
                if klass:
                    sources_to_scan.append(klass)
            src = "\n".join(sources_to_scan)
        else:
            src = src_full
    else:
        src = src_full
    types: set[str] = set()
    named_sets: set[str] = set()
    for m in _RE_TYPE_EQ.finditer(src):
        types.add(m.group(1))
    for m in _RE_TYPE_IN_SET.finditer(src):
        # Comma-split contents of {A, B, C}
        for tok in m.group(1).split(","):
            tok = tok.strip().lstrip("bus.").strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", tok):
                types.add(tok)
    for m in _RE_TYPE_IN_NAMED.finditer(src):
        named_sets.add(m.group(1))
    return {
        "exact_types": sorted(types),
        "named_sets": sorted(named_sets),
        "confident": bool(types) and not named_sets,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of table")
    args = ap.parse_args()

    sites = find_subscribe_call_sites(REPO_ROOT / "titan_plugin")
    # Narrow to the un-migrated subscribers
    todo = [s for s in sites if not s["has_types"]]

    # Group by name → multiple subscribe sites for the same logical
    # subscriber (legacy_core + core/plugin) are reported together.
    by_name: dict[str, list[dict]] = {}
    for s in todo:
        by_name.setdefault(s["name"], []).append(s)

    report = []
    for name in sorted(by_name):
        site_list = by_name[name]
        # All sites for this name — pick the file with the longest source
        # (likely the active drain location) for whitelist extraction.
        # Use per-function scoping per site to avoid bleeding patterns
        # from sibling functions.
        wl = {}
        for s in site_list:
            f = REPO_ROOT / s["file"]
            ext = extract_whitelist_from_source(f, subscribe_lineno=s["line"])
            for k, v in ext.items():
                if k == "confident":
                    wl[k] = wl.get(k, True) and v
                else:
                    wl.setdefault(k, [])
                    wl[k].extend(x for x in v if x not in wl[k])
        wl["exact_types"].sort() if wl.get("exact_types") else None
        # All-sites reply_only check: if every site declares reply_only=True,
        # this subscriber is excluded from broadcasts at the bus level
        # already — types= migration is cosmetic, not load-bearing.
        all_reply_only = all(
            s["reply_only_expr"].strip() == "True" for s in site_list
        )
        report.append({
            "name": name,
            "sites": [
                {"file": s["file"], "line": s["line"],
                 "reply_only": s["reply_only_expr"]}
                for s in site_list
            ],
            "candidate_filter": wl.get("exact_types") or [],
            "named_set_references": wl.get("named_sets") or [],
            "confident": wl.get("confident", False),
            "all_reply_only": all_reply_only,
        })

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
        return

    # Pretty print — partition output into 3 buckets so reply_only=True
    # subscribers don't get flagged as work-to-do alongside genuine
    # candidates. Cosmetic-only entries are listed but visually separated.
    real_candidates = [r for r in report if not r["all_reply_only"]]
    cosmetic_only = [r for r in report if r["all_reply_only"]]

    print("=" * 80)
    print(f"Bus subscribe-filter migration candidates")
    print(f"  {len(real_candidates):>2} flood-receiving subscribers needing real filter migration")
    print(f"  {len(cosmetic_only):>2} reply_only=True subscribers (already excluded from broadcasts —")
    print(f"      types=[] is cosmetic only, listed below for completeness)")
    print("=" * 80)
    print()

    def _emit(r: dict, *, kind: str) -> None:
        if kind == "real":
            label = "✓ confident" if r["confident"] else "⚠ needs review"
        else:
            label = "○ reply_only=True (no broadcasts received — types=[] is cosmetic)"
        print(f"  {r['name']}  [{label}]")
        for s in r["sites"]:
            print(f"    {s['file']}:{s['line']}  reply_only={s['reply_only']}")
        if r["candidate_filter"]:
            print(f"    → CANDIDATE types=[{', '.join(r['candidate_filter'])}]")
        elif kind == "real":
            print(f"    → no msg_type literal comparisons found in source — manual trace required")
        if r["named_set_references"]:
            print(f"    → named-set refs: {', '.join(r['named_set_references'])} (resolve manually)")
        print()

    if real_candidates:
        print("--- Flood-receiving subscribers (real migration needed) ---")
        print()
        for r in real_candidates:
            _emit(r, kind="real")

    if cosmetic_only:
        print("--- reply_only=True subscribers (no broadcasts; cosmetic types=[] only) ---")
        print()
        for r in cosmetic_only:
            _emit(r, kind="cosmetic")

    print("=" * 80)
    print("HOW TO USE:")
    print(" 1. Pick a subscriber from the FIRST section above (start with confident).")
    print(" 2. Open the file:line, change `bus.subscribe(name)` to")
    print("    `bus.subscribe(name, types=[<CANDIDATE>])` using the suggested filter.")
    print(" 3. Add an integration test pinning the filter (mirror")
    print("    tests/test_bus_filter_migrations.py).")
    print(" 4. Verify with `python -m pytest tests/test_bus_filter_migrations.py`.")
    print(" 5. Land in batches of ~5 subscribers per commit so reverts are easy.")
    print()
    print("Subscribers in the SECOND section are already excluded from broadcasts via")
    print("reply_only=True. Adding types=[] is documentation only; skip unless you want")
    print("the contract self-documented for arch_map.")
    print()


if __name__ == "__main__":
    main()
