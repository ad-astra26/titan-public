#!/usr/bin/env python3
"""arch_map dead-wiring — static analysis for silent-swallow bugs.

Detects the class of bugs where infrastructure is defined, deployed, but
never actually invoked at the needed moment. See titan-docs/DEFERRED_ITEMS.md
ARCH-MAP-DEAD-WIRING for the full design + 5+ motivating cases from the
2026-04-16 session.

v1 capabilities (this module):
  - Orphan method detection (defined but zero callers)
  - Pair-closure detection (query_X called but record_X never called)
  - Config section tracing (toml sections defined but never .get()'d)
  - rFP verification (extract class/method names from an rFP markdown,
    verify each exists in the codebase)

Deferred for v2/v3:
  - Bus-flow audit (publish/subscribe cross-reference)
  - Dim/shape drift detector (numpy constant mismatches)
  - DB CRUD-balance (INSERT vs UPDATE vs SELECT per table)
  - Cross-Titan endpoint schema diff
  - Runtime call overlay (brain log cross-check)
  - Git-context surfacing (introduction + last-mod + paired-site commits)
  - Architecture-invariant CI checks

Usage:
    python scripts/arch_map_dead_wiring.py
    python scripts/arch_map_dead_wiring.py --root titan_plugin --verbose
    python scripts/arch_map_dead_wiring.py --rfp titan-docs/rFP_foo.md
    python scripts/arch_map_dead_wiring.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


DUNDER = re.compile(r"^__.+__$")


# Pair patterns — (query-side regex, record-side regex-template).
# Stem ($1 from query) is substituted into record pattern.
PAIR_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"^query_(.+)$"),     ["record_{stem}", "mark_{stem}_used",
                                        "update_{stem}", "ack_{stem}",
                                        "commit_{stem}"]),
    (re.compile(r"^retrieve_(.+)$"),  ["ack_{stem}", "commit_{stem}",
                                        "record_{stem}", "consume_{stem}"]),
    (re.compile(r"^get_(.+)$"),       ["set_{stem}", "update_{stem}", "put_{stem}"]),
    (re.compile(r"^emit_(.+)$"),      ["consume_{stem}", "handle_{stem}"]),
    (re.compile(r"^register_(.+)$"),  ["unregister_{stem}"]),
    (re.compile(r"^lock_(.+)$"),      ["unlock_{stem}", "release_{stem}"]),
    (re.compile(r"^open_(.+)$"),      ["close_{stem}"]),
    (re.compile(r"^start_(.+)$"),     ["stop_{stem}", "end_{stem}"]),
]


# Decorators that mark a method as externally-invoked even without literal
# Python call sites in the codebase (FastAPI routes, bus handlers, CLI, etc.).
IGNORE_DECORATOR_SUBSTRINGS = frozenset({
    # FastAPI / web framework routes
    "app.get", "app.post", "app.put", "app.delete", "app.patch",
    "router.get", "router.post", "router.put", "router.delete",
    "router.websocket", "app.websocket", "app.route",
    "app.lifespan", "lifespan",
    # Bus / event handlers
    "on_message", "on_event", "subscribe", "register",
    # Python built-in method decorators
    "property", "staticmethod", "classmethod",
    "cache", "cached_property", "lru_cache",
    # Abstract base class methods — overrides are called polymorphically
    "abstractmethod", "abc.abstractmethod", "abstractproperty",
    # Task / background / async framework patterns
    "task", "background_task", "celery.task",
    # Test framework
    "pytest.fixture", "fixture", "pytest.mark",
})


# Method-name prefixes that conventionally indicate externally-driven entry
# points (bus dispatchers, event handlers, test helpers, lifecycle hooks).
IGNORE_NAME_PREFIXES = frozenset({
    "test_", "on_", "handle_", "_handle_", "_on_",
    "run_", "main_", "cmd_",
})


# Method-name equals — exact names always considered reachable.
IGNORE_NAME_EXACT = frozenset({
    "main", "run", "execute", "dispatch",
    "__init__", "__enter__", "__exit__", "__repr__", "__str__",
})


@dataclass
class MethodDef:
    name: str
    class_name: str | None
    file: str
    line: int
    is_private: bool
    decorators: list[str]


@dataclass
class CallSite:
    target: str   # method name being called
    file: str
    line: int
    in_class: str | None


@dataclass
class Finding:
    kind: str     # "orphan" | "pair_gap" | "unused_config" | "rfp_missing"
    severity: str  # "high" | "medium" | "info"
    title: str
    detail: str
    file: str = ""
    line: int = 0
    extra: dict = field(default_factory=dict)


# ─── Static scan ───────────────────────────────────────────────────────

def _dec_name(d: ast.expr) -> str:
    if isinstance(d, ast.Name):
        return d.id
    if isinstance(d, ast.Attribute):
        return f"{_dec_name(d.value)}.{d.attr}"
    if isinstance(d, ast.Call):
        return _dec_name(d.func)
    return "<?>"


def _call_target(call: ast.Call) -> str | None:
    """Extract the tail name of a Call target (method attr or function)."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def scan_file(path: Path) -> tuple[list[MethodDef], list[CallSite], list[CallSite]]:
    """Walk one .py file: collect method defs + call sites + bare references.

    Returns (defs, calls, refs). Bare references include:
      - Attribute access without Call: `obj.method` passed as argument
        (e.g., `asyncio.to_thread(obj.method, ...)`, `loop.run_in_executor(...)`)
      - Dict-literal values: `{"RECALL": reward_recall, ...}` — dispatch
        registries where methods are stored by string key
      - Keyword argument values: `entry_fn=worker_main`, `target=fn` —
        multiprocessing/threading entry points
      - Decorator use of another identifier: `@Depends(verify_auth)`
    These cases are "references without direct call" — the method IS
    reachable but the caller invokes it indirectly. Without this pass the
    scanner produces a high false-positive rate (~55-65% of orphans).
    """
    defs: list[MethodDef] = []
    calls: list[CallSite] = []
    refs: list[CallSite] = []
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return defs, calls, refs

    class_stack: list[str] = []

    # Names that commonly take a callable as kwarg value. Identifiers passed
    # to these should count as references (the function is invoked later).
    CALLABLE_KWARGS = {"entry_fn", "target", "fn", "func", "callback",
                       "handler", "on_success", "on_error", "on_complete",
                       "key", "lifespan", "done_callback", "dependency"}

    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node):
            cls = class_stack[-1] if class_stack else None
            decs = [_dec_name(d) for d in node.decorator_list]
            defs.append(MethodDef(
                name=node.name,
                class_name=cls,
                file=str(path),
                line=node.lineno,
                is_private=node.name.startswith("_") and not DUNDER.match(node.name),
                decorators=decs,
            ))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            target = _call_target(node)
            if target:
                cls = class_stack[-1] if class_stack else None
                calls.append(CallSite(
                    target=target,
                    file=str(path),
                    line=node.lineno,
                    in_class=cls,
                ))
            # Inspect positional args for bare callable references, e.g.
            # asyncio.to_thread(obj.method, ...) where obj.method is not a Call.
            for arg in node.args:
                self._record_bare_ref(arg, node.lineno)
            # Keyword args: entry_fn=worker_main, target=fn
            for kw in node.keywords:
                if kw.arg in CALLABLE_KWARGS:
                    self._record_bare_ref(kw.value, node.lineno)
            self.generic_visit(node)

        def visit_Dict(self, node):
            # Dispatch registries: {"RECALL": reward_recall, ...} — each
            # value that's a Name or Attribute is a bare reference.
            for v in node.values:
                self._record_bare_ref(v, node.lineno)
            self.generic_visit(node)

        def visit_Return(self, node):
            # Factory-return patterns — common in Agno tool/hook factories
            # and elsewhere that build a callable registry and return it.
            #   return titan_pre_hook           → bare Name reference
            #   return [fn1, fn2, fn3]          → list-literal of callables
            #   return (fn1, fn2)               → tuple-literal of callables
            #   return {"alias": fn1, ...}      → dict of callables (caught by
            #                                      visit_Dict already)
            v = node.value
            if v is None:
                return self.generic_visit(node)
            if isinstance(v, (ast.Name, ast.Attribute)):
                self._record_bare_ref(v, node.lineno)
            elif isinstance(v, (ast.List, ast.Tuple, ast.Set)):
                for elt in v.elts:
                    self._record_bare_ref(elt, node.lineno)
            self.generic_visit(node)

        def _record_bare_ref(self, expr, lineno):
            """Capture `obj.method` or `function_name` as a reference."""
            name: str | None = None
            if isinstance(expr, ast.Attribute):
                name = expr.attr
            elif isinstance(expr, ast.Name):
                name = expr.id
            if name and not DUNDER.match(name):
                cls = class_stack[-1] if class_stack else None
                refs.append(CallSite(
                    target=name,
                    file=str(path),
                    line=lineno,
                    in_class=cls,
                ))

    V().visit(tree)
    return defs, calls, refs


def scan_tree(roots: list[Path]) -> tuple[list[MethodDef], list[CallSite],
                                            list[CallSite], int]:
    """Scan every .py under root(s), excluding __pycache__. Returns
    (defs, calls, refs, file_count).

    Multiple roots supported so callers can scan titan_plugin/ + scripts/
    + tests/ together (v1.1 expansion: config + reference consumers live
    outside the primary module tree).
    """
    all_defs: list[MethodDef] = []
    all_calls: list[CallSite] = []
    all_refs: list[CallSite] = []
    n_files = 0
    for root in roots:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            n_files += 1
            d, c, r = scan_file(py)
            all_defs.extend(d)
            all_calls.extend(c)
            all_refs.extend(r)
    return all_defs, all_calls, all_refs, n_files


# ─── Detection: orphan methods ─────────────────────────────────────────

# Annotations that explicitly opt a method out of the orphan check. Placed as
# a comment within ±5 lines of the `def`. Used for methods that are
# intentionally defined-but-uncalled (deprecated-with-replacement, public API
# surface reserved for future callers, etc.) where wiring or deleting would
# be worse than leaving + annotating.
_ORPHAN_OPT_OUT_MARKERS = ("DEPRECATED:", "UNUSED_PUBLIC_API:",
                            "SCANNER_SKIP_ORPHAN", "RESERVED_PUBLIC_API:")

# Class-level opt-out: placed near `class X:` declaration (±12 lines). When
# present, ALL public methods of the class are skipped from orphan detection.
# Used for entire classes whose wiring is deferred to a dedicated rFP (e.g.
# SovereigntyTracker → SOVEREIGNTY-TRACKER-WIRING, MetabolismController gate
# methods → METABOLISM-GATE-WIRING, SocialGraph kin API → SOCIAL-GRAPH-WIRING).
_CLASS_OPT_OUT_MARKERS = ("DEFERRED_CLASS_WIRING:", "SCANNER_SKIP_CLASS_ORPHANS")

# Cache of (file, class_name) → has_class_marker, populated on first query
# per run so we don't re-read files for every method of the same class.
_CLASS_MARKER_CACHE: dict = {}


def _def_has_opt_out_marker(file_path: str, line_no: int,
                             markers: tuple[str, ...] = _ORPHAN_OPT_OUT_MARKERS) -> bool:
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    lo = max(0, line_no - 6)
    hi = min(len(lines), line_no + 3)
    context = " ".join(lines[lo:hi]).upper()
    return any(m.upper() in context for m in markers)


def _method_class_has_opt_out_marker(file_path: str,
                                      class_name: str | None) -> bool:
    """Check whether the class containing a method has a class-level
    orphan-suppression marker. Caches per (file, class) for the run.
    """
    if not class_name:
        return False
    key = (file_path, class_name)
    if key in _CLASS_MARKER_CACHE:
        return _CLASS_MARKER_CACHE[key]
    try:
        src = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        _CLASS_MARKER_CACHE[key] = False
        return False
    # Find `class <class_name>` declaration line
    lines = src.splitlines()
    class_line = None
    for i, ln in enumerate(lines):
        if re.match(rf'^\s*class\s+{re.escape(class_name)}\b', ln):
            class_line = i  # 0-indexed
            break
    if class_line is None:
        _CLASS_MARKER_CACHE[key] = False
        return False
    lo = max(0, class_line - 12)
    hi = min(len(lines), class_line + 2)
    ctx = " ".join(lines[lo:hi]).upper()
    found = any(m in ctx for m in _CLASS_OPT_OUT_MARKERS)
    _CLASS_MARKER_CACHE[key] = found
    return found


def find_orphans(defs: list[MethodDef], calls: list[CallSite],
                 refs: list[CallSite],
                 allowlist_names: set[str]) -> list[Finding]:
    """Methods defined but never called NOR referenced anywhere in scanned tree.

    "Referenced" includes bare attribute access (e.g., passed as callable to
    asyncio.to_thread), dict-literal values (dispatch registries), and
    keyword args like entry_fn=/target= (multiprocessing entry points).

    Filters out: dunders, _private, known external-entry decorators,
    known external-entry name prefixes, allowlisted names, and methods with
    an explicit # DEPRECATED: or # UNUSED_PUBLIC_API: marker in source.
    """
    call_counts: dict[str, int] = defaultdict(int)
    for c in calls:
        call_counts[c.target] += 1
    # References count the same as calls for reachability purposes
    for r in refs:
        call_counts[r.target] += 1

    findings: list[Finding] = []
    for d in defs:
        if DUNDER.match(d.name):
            continue
        if d.is_private:
            continue
        if d.name in IGNORE_NAME_EXACT or d.name in allowlist_names:
            continue
        if any(d.name.startswith(p) for p in IGNORE_NAME_PREFIXES):
            continue
        if any(any(s in dec for s in IGNORE_DECORATOR_SUBSTRINGS)
               for dec in d.decorators):
            continue
        if call_counts[d.name] > 0:
            continue
        # Explicit opt-out marker near def: skip.
        if _def_has_opt_out_marker(d.file, d.line):
            continue
        # Class-level opt-out: skip ALL public methods of the class.
        if _method_class_has_opt_out_marker(d.file, d.class_name):
            continue

        # Flag as orphan
        findings.append(Finding(
            kind="orphan",
            severity="high" if d.class_name else "medium",
            title=f"{d.class_name + '.' if d.class_name else ''}{d.name}",
            detail="Defined but zero callers in scanned tree",
            file=d.file,
            line=d.line,
            extra={"class_name": d.class_name, "method_name": d.name, "decorators": d.decorators},
        ))
    return findings


# ─── Detection: pair-closure gaps ──────────────────────────────────────

def find_pair_gaps(defs: list[MethodDef],
                   calls: list[CallSite],
                   refs: list[CallSite] | None = None) -> list[Finding]:
    """Pair patterns where the query side has callers but the record side
    (same class, matching stem) has zero callers — broken feedback loops."""
    call_counts: dict[str, int] = defaultdict(int)
    for c in calls:
        call_counts[c.target] += 1
    for r in (refs or []):
        call_counts[r.target] += 1

    # Index defs by (class, name) for O(1) lookup
    def_index: dict[tuple[str | None, str], MethodDef] = {}
    for d in defs:
        def_index[(d.class_name, d.name)] = d

    seen_pairs: set[tuple[str, str]] = set()
    findings: list[Finding] = []

    for d in defs:
        if call_counts[d.name] == 0:
            continue  # query side itself not called; not a closure-gap signal

        for q_pat, r_templates in PAIR_PATTERNS:
            m = q_pat.match(d.name)
            if not m:
                continue
            stem = m.group(1)
            for tmpl in r_templates:
                r_name = tmpl.format(stem=stem)
                r_def = def_index.get((d.class_name, r_name))
                if r_def is None:
                    continue
                pair_key = (d.file + ":" + d.name, r_def.file + ":" + r_name)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if call_counts[r_name] == 0:
                    findings.append(Finding(
                        kind="pair_gap",
                        severity="high",
                        title=f"{d.class_name}.{d.name} → {r_name}",
                        detail=(f"Query side called {call_counts[d.name]}x, "
                                f"record side {r_name} called 0x — closure loop broken"),
                        file=r_def.file,
                        line=r_def.line,
                        extra={
                            "query_name": d.name,
                            "query_file": d.file,
                            "query_line": d.line,
                            "query_calls": call_counts[d.name],
                            "record_name": r_name,
                        },
                    ))
                break  # matched a pair; don't try other templates for this stem

    return findings


# ─── Detection: unused config sections ─────────────────────────────────

def find_unused_config_sections(toml_paths: list[Path],
                                code_root: Path) -> list[Finding]:
    """Top-level TOML sections that no code reads via .get('<section>', ...)
    or subscript access."""
    sections: dict[str, Path] = {}
    for tp in toml_paths:
        if not tp.exists():
            continue
        try:
            with open(tp, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        for key in data.keys():
            # If a section exists in multiple toml files, first-seen path wins
            # (informational only — the real usage check is file-agnostic).
            sections.setdefault(key, tp)

    if not sections:
        return []

    # Build a single regex matching any of the known section names.
    # Recognized patterns (v1.1 expanded):
    #   .get("section", ...)           — dict access via get()
    #   ["section"]                    — subscript access
    #   get_params("section")          — Titan project-specific bespoke API
    #   load_titan_params()["section"] — alternate bespoke API
    #   load_titan_config().get("section", ...)  — already matched via .get
    escaped = "|".join(re.escape(s) for s in sections)
    pat = re.compile(
        rf'''(?:\.get\(\s*|\[\s*|get_params\(\s*|load_titan_params\(\)\s*\[\s*)["']({escaped})["']''',
    )

    usage: dict[str, int] = defaultdict(int)
    # Extended scan: titan_params/config users live in scripts/, tests/, and
    # occasionally other project siblings. Scan project root for .py files.
    project_root = code_root.parent if code_root.is_dir() else code_root.parent.parent
    scan_dirs = [code_root]
    for extra in ("scripts", "tests"):
        extra_dir = project_root / extra
        if extra_dir.exists():
            scan_dirs.append(extra_dir)

    for sd in scan_dirs:
        for py in sd.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                src = py.read_text(encoding="utf-8")
            except Exception:
                continue
            for m in pat.finditer(src):
                usage[m.group(1)] += 1

    findings: list[Finding] = []
    for section, tp in sections.items():
        if usage[section] == 0:
            findings.append(Finding(
                kind="unused_config",
                severity="medium",
                title=f"[{section}]",
                detail=f"Defined in {tp.name} but no .get('{section}') / ['{section}'] in code",
                file=str(tp),
                line=0,
                extra={"toml_file": str(tp)},
            ))
    return findings


# ─── rFP verification ──────────────────────────────────────────────────

RFP_CLASS_RE = re.compile(r"\b(class\s+)?([A-Z][A-Za-z0-9_]*Engine|[A-Z][A-Za-z0-9_]*Store|"
                          r"[A-Z][A-Za-z0-9_]*Tracker|[A-Z][A-Za-z0-9_]*Worker|"
                          r"[A-Z][A-Za-z0-9_]*Handler|[A-Z][A-Za-z0-9_]*Monitor|"
                          r"[A-Z][A-Za-z0-9_]*Net)\b")
RFP_METHOD_RE = re.compile(r"\b([a-z_][a-z0-9_]{3,})\(\)")
RFP_BUS_MSG_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,}_[A-Z][A-Z0-9_]+)\b")
RFP_CONFIG_SECTION_RE = re.compile(r"\[([a-z_][a-z_0-9]*)\]")
RFP_FILE_REF_RE = re.compile(r"\b([a-z_][a-z_0-9]*\.py)(?::(\d+))?")


def extract_rfp_entities(rfp_path: Path) -> dict[str, set[str]]:
    """Pull candidate implementation entities from an rFP markdown."""
    text = rfp_path.read_text(encoding="utf-8")
    return {
        "classes": {m.group(2) for m in RFP_CLASS_RE.finditer(text)},
        "methods": {m.group(1) for m in RFP_METHOD_RE.finditer(text)
                    if m.group(1) not in {"list", "dict", "str", "int", "float",
                                          "bool", "tuple", "set", "print", "len",
                                          "type", "open", "range", "len"}},
        "bus_msgs": {m.group(1) for m in RFP_BUS_MSG_RE.finditer(text)},
        "config_sections": {m.group(1) for m in RFP_CONFIG_SECTION_RE.finditer(text)},
        "files": {m.group(1) for m in RFP_FILE_REF_RE.finditer(text)},
    }


def verify_rfp(rfp_path: Path, defs: list[MethodDef], code_root: Path,
               toml_paths: list[Path]) -> list[Finding]:
    """For each entity extracted from the rFP, check it exists."""
    ents = extract_rfp_entities(rfp_path)
    findings: list[Finding] = []

    class_names = {d.class_name for d in defs if d.class_name}
    method_names = {d.name for d in defs}

    # Classes
    for c in ents["classes"]:
        if c not in class_names:
            findings.append(Finding(
                kind="rfp_missing",
                severity="high",
                title=f"class {c}",
                detail=f"Referenced in {rfp_path.name} but class definition not found in {code_root}",
                file=str(rfp_path),
            ))

    # Methods — looser signal (many natural English words match)
    # Only flag methods that look distinctly engineering (snake_case with 2+ words)
    for m in ents["methods"]:
        if m in method_names:
            continue
        if "_" not in m:
            continue  # single-word methods are unreliable signals
        findings.append(Finding(
            kind="rfp_missing",
            severity="medium",
            title=f"{m}()",
            detail=f"Referenced in {rfp_path.name} but no method with this name found",
            file=str(rfp_path),
        ))

    # Bus messages — grep for any mention in code
    for b in ents["bus_msgs"]:
        # Find any .py file referencing this constant
        found = False
        for py in code_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                src = py.read_text(encoding="utf-8")
            except Exception:
                continue
            if b in src:
                found = True
                break
        if not found:
            findings.append(Finding(
                kind="rfp_missing",
                severity="medium",
                title=f"bus msg {b}",
                detail=f"Referenced in {rfp_path.name} but not found in code",
                file=str(rfp_path),
            ))

    # Config sections — check both toml files
    toml_sections: set[str] = set()
    for tp in toml_paths:
        if not tp.exists():
            continue
        try:
            with open(tp, "rb") as f:
                data = tomllib.load(f)
            toml_sections.update(data.keys())
        except Exception:
            pass

    for s in ents["config_sections"]:
        # Filter out things that look like table of contents markers
        if len(s) < 3:
            continue
        if s in toml_sections:
            continue
        findings.append(Finding(
            kind="rfp_missing",
            severity="medium",
            title=f"[{s}]",
            detail=f"Config section referenced in {rfp_path.name} but not in any toml",
            file=str(rfp_path),
        ))

    return findings


# ─── Bus-flow audit ────────────────────────────────────────────────────

def extract_bus_message_types(bus_py: Path) -> set[str]:
    """Pull every `MSG_TYPE = "..."` constant definition from bus.py.

    Returns the set of known bus message type strings. These are the
    canonical vocabulary of DivineBus messages — any literal not in this
    set used as a message type is a code smell.
    """
    if not bus_py.exists():
        return set()
    try:
        src = bus_py.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(bus_py))
    except Exception:
        return set()

    msg_types: set[str] = set()
    for node in ast.walk(tree):
        # Match: NAME = "string_literal" at module level
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if (isinstance(target, ast.Name)
                    and target.id.isupper()
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                # Only capture constants whose value equals or resembles the name
                # (filters out unrelated string constants like version strings).
                val = value.value
                if val == target.id or val.upper() == target.id:
                    msg_types.add(val)
    return msg_types


def extract_helper_publishers(
        root: Path, known_types: set[str]
) -> dict[str, str]:
    """Detect helper functions that publish bus messages.

    Returns {helper_function_name: msg_type}. These are functions whose
    body contains a `make_msg(MSG_TYPE, ...)` or `_send_msg(q, "MSG_TYPE", ...)`
    call — meaning a caller invoking the helper is effectively publishing.

    Example: bus.py defines `emit_meta_cgn_signal()` as the enforced way
    to publish META_CGN_SIGNAL. Static scan naively counts 0 publishers
    because no `make_msg(META_CGN_SIGNAL, ...)` exists at a call site;
    this function resolves that indirection by treating emit_meta_cgn_signal
    as a synonym publisher for META_CGN_SIGNAL.
    """
    helpers: dict[str, str] = {}
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip very long functions (>200 lines) — unlikely to be helpers
            end_line = max(
                (getattr(n, "lineno", node.lineno) for n in ast.walk(node)),
                default=node.lineno)
            if end_line - node.lineno > 200:
                continue
            # Search body for publisher indicators. Match ANY of:
            #   - make_msg(MSG_TYPE, ...)
            #   - _send_msg(queue, "MSG_TYPE", ...)
            #   - dict literal {"type": "MSG_TYPE", ...}  (emit_meta_cgn_signal
            #     builds the message manually rather than via make_msg)
            found = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn_name = None
                    if isinstance(sub.func, ast.Name):
                        fn_name = sub.func.id
                    elif isinstance(sub.func, ast.Attribute):
                        fn_name = sub.func.attr
                    if fn_name in ("make_msg", "_send_msg"):
                        pos = 0 if fn_name == "make_msg" else 1
                        if len(sub.args) > pos:
                            arg = sub.args[pos]
                            msg_type = None
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                msg_type = arg.value
                            elif isinstance(arg, ast.Name):
                                msg_type = arg.id
                            if msg_type and msg_type in known_types:
                                helpers.setdefault(node.name, msg_type)
                                found = True
                                break
                if isinstance(sub, ast.Dict):
                    # Look for {"type": "MSG_TYPE", ...} or {"type": MSG_TYPE_CONST, ...}
                    for k, v in zip(sub.keys, sub.values):
                        if (isinstance(k, ast.Constant)
                                and isinstance(k.value, str)
                                and k.value == "type"):
                            msg_type = None
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                msg_type = v.value
                            elif isinstance(v, ast.Name):
                                msg_type = v.id
                            if msg_type and msg_type in known_types:
                                helpers.setdefault(node.name, msg_type)
                                found = True
                                break
                    if found:
                        break
    # Filter out the canonical helpers themselves (make_msg, _send_msg)
    helpers.pop("make_msg", None)
    helpers.pop("_send_msg", None)
    return helpers


def scan_bus_publishers_and_subscribers(
        root: Path, known_types: set[str],
        helper_publishers: dict[str, str] | None = None,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[tuple[str, int]]]]:
    """AST + source-level scan for bus publishers and subscribers.

    Publisher patterns:
      make_msg(TYPE, ...)            — Name reference
      make_msg("TYPE", ...)          — string literal
      _send_msg(queue, "TYPE", ...)  — string literal in position 1

    Subscriber patterns:
      msg_type == TYPE / == "TYPE"
      msg.get("type") == TYPE / == "TYPE"
      if msg_type in (TYPE_A, TYPE_B)
      match msg_type: case "TYPE": ...
    """
    publishers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    subscribers: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # Build a compiled regex union for fast prefilter (avoid AST on files
    # that definitely don't touch the bus).
    bus_marker = re.compile(r"\b(make_msg|_send_msg|msg_type|msg\.get\(|message_type)\b")

    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if not bus_marker.search(src):
            continue
        try:
            tree = ast.parse(src, filename=str(py))
        except Exception:
            continue

        for node in ast.walk(tree):
            # Publisher: Call where func is make_msg or _send_msg
            if isinstance(node, ast.Call):
                fname = None
                f = node.func
                if isinstance(f, ast.Name):
                    fname = f.id
                elif isinstance(f, ast.Attribute):
                    fname = f.attr

                if fname in ("make_msg", "_send_msg"):
                    # Message type is arg 0 for make_msg, arg 1 for _send_msg
                    pos = 0 if fname == "make_msg" else 1
                    if len(node.args) > pos:
                        arg = node.args[pos]
                        msg_type = None
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            msg_type = arg.value
                        elif isinstance(arg, ast.Name):
                            msg_type = arg.id
                        if msg_type and msg_type in known_types:
                            publishers[msg_type].append((str(py), node.lineno))
                elif fname == "put" and len(node.args) >= 1 and isinstance(node.args[0], ast.Dict):
                    # Raw send_queue.put({"type": "MSG_TYPE", ...}) pattern —
                    # used by spirit_worker for OBSERVABLES_SNAPSHOT and similar
                    # internal messages. Extract msg_type from the dict literal.
                    d = node.args[0]
                    for k, v in zip(d.keys, d.values):
                        if (isinstance(k, ast.Constant) and k.value == "type"
                                and isinstance(v, ast.Constant)
                                and isinstance(v.value, str)
                                and v.value in known_types):
                            publishers[v.value].append((str(py), node.lineno))
                            break
                elif helper_publishers and fname in helper_publishers:
                    # v2.1: call to a helper function that internally publishes.
                    # emit_meta_cgn_signal(...) publishes META_CGN_SIGNAL via the
                    # helper's enforced-invariants path (edge-detect + rate gate).
                    msg_type = helper_publishers[fname]
                    publishers[msg_type].append((str(py), node.lineno))

            # Subscriber: Compare with msg_type / msg.get("type")
            if isinstance(node, ast.Compare):
                left_is_type_ref = False
                if isinstance(node.left, ast.Name) and node.left.id in (
                        "msg_type", "message_type"):
                    left_is_type_ref = True
                elif (isinstance(node.left, ast.Call)
                      and isinstance(node.left.func, ast.Attribute)
                      and node.left.func.attr == "get"
                      and len(node.left.args) >= 1
                      and isinstance(node.left.args[0], ast.Constant)
                      and node.left.args[0].value in ("type", "msg_type")):
                    left_is_type_ref = True

                if left_is_type_ref:
                    for comp_val in node.comparators:
                        msg_type = None
                        if isinstance(comp_val, ast.Constant) and isinstance(comp_val.value, str):
                            msg_type = comp_val.value
                        elif isinstance(comp_val, ast.Name):
                            msg_type = comp_val.id
                        elif isinstance(comp_val, (ast.Tuple, ast.List, ast.Set)):
                            for elt in comp_val.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    if elt.value in known_types:
                                        subscribers[elt.value].append((str(py), node.lineno))
                                elif isinstance(elt, ast.Name) and elt.id in known_types:
                                    subscribers[elt.id].append((str(py), node.lineno))
                            continue
                        if msg_type and msg_type in known_types:
                            subscribers[msg_type].append((str(py), node.lineno))

    return publishers, subscribers


def _publisher_site_has_intentional_marker(file_path: str, line_no: int) -> bool:
    """Check if a publish site is marked INTENTIONAL_BROADCAST / INTENTIONAL_SELF_ROUTE
    in a ±10-line window. Same contract as arch_map.py::_has_intentional_marker.
    """
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    lo = max(0, line_no - 11)
    hi = min(len(lines), line_no + 4)
    context = " ".join(lines[lo:hi]).upper()
    return any(m in context for m in
               ("INTENTIONAL_BROADCAST", "INTENTIONAL_SELF_ROUTE",
                "INTENTIONAL BROADCAST", "INTENTIONAL SELF ROUTE"))


def find_bus_flow_imbalances(
        publishers: dict[str, list[tuple[str, int]]],
        subscribers: dict[str, list[tuple[str, int]]],
        known_types: set[str]
) -> list[Finding]:
    """Cross-reference publishers and subscribers. Flag:
    - Types with publishers but no subscribers (dead messages)
    - Types with subscribers but no publishers (dead handlers)
    - Types defined in bus.py but neither published nor subscribed
    """
    findings: list[Finding] = []

    # Common message types we know are handled by infra not statically visible
    # (QUERY/RESPONSE are routed by request-id inside DivineBus internals).
    INFRA_MSG_TYPES = frozenset({"QUERY", "RESPONSE", "DREAM_WAKE_REQUEST"})

    for msg_type in sorted(known_types):
        if msg_type in INFRA_MSG_TYPES:
            continue
        pubs = publishers.get(msg_type, [])
        subs = subscribers.get(msg_type, [])

        if pubs and not subs:
            # Skip if ANY publish site is marked as intentional broadcast
            # (e.g. TITAN_SELF_STATE per rFP #2 Phase 4 — consumed inline).
            # The marker is semantically attached to the msg_type, not each
            # publish site: helper-function callers inherit the intent.
            if any(_publisher_site_has_intentional_marker(p[0], p[1]) for p in pubs):
                continue
            findings.append(Finding(
                kind="bus_dead_msg",
                severity="high",
                title=f"msg {msg_type} — {len(pubs)} publishers, 0 subscribers",
                detail=f"Published at {len(pubs)} site(s) but no handler dispatches on it",
                file=pubs[0][0],
                line=pubs[0][1],
                extra={"pub_count": len(pubs), "sub_count": 0,
                       "first_publisher": f"{pubs[0][0]}:{pubs[0][1]}"},
            ))
        elif subs and not pubs:
            findings.append(Finding(
                kind="bus_dead_handler",
                severity="high",
                title=f"msg {msg_type} — 0 publishers, {len(subs)} subscribers",
                detail=f"Handler dispatches on it at {len(subs)} site(s) but nothing publishes",
                file=subs[0][0],
                line=subs[0][1],
                extra={"pub_count": 0, "sub_count": len(subs),
                       "first_subscriber": f"{subs[0][0]}:{subs[0][1]}"},
            ))
        elif not pubs and not subs:
            # Skip if the constant has a RESERVED:/LEGACY: marker in bus.py
            # comment (±3 lines around the constant definition). Used for
            # types that exist on purpose (reserved for future wiring or
            # legacy compat) rather than being stale.
            if _bus_const_has_reserved_marker(msg_type):
                continue
            findings.append(Finding(
                kind="bus_unused_type",
                severity="medium",
                title=f"msg {msg_type} — defined in bus.py, never used",
                detail="Constant exists but neither published nor subscribed anywhere",
                file="titan_plugin/bus.py",
                line=0,
                extra={"pub_count": 0, "sub_count": 0},
            ))

    return findings


# Cache for bus.py RESERVED markers — computed once per scan.
_BUS_RESERVED_CACHE: dict | None = None


def _bus_const_has_reserved_marker(msg_type: str,
                                    bus_path: str = "titan_plugin/bus.py") -> bool:
    """Check if a bus.py constant has a RESERVED:/LEGACY: marker within ±3
    lines. These markers flag constants that are intentionally unused (for
    future wiring or legacy back-compat), silencing the bus_unused_type
    finding without deleting the constant.
    """
    global _BUS_RESERVED_CACHE
    if _BUS_RESERVED_CACHE is None:
        _BUS_RESERVED_CACHE = {}
        try:
            lines = Path(bus_path).read_text(encoding="utf-8",
                                              errors="replace").splitlines()
        except Exception:
            return False
        # Match `MSG_NAME = "MSG_NAME"` on each line
        for i, ln in enumerate(lines):
            m = re.match(r'^([A-Z][A-Z0-9_]+)\s*=\s*"[A-Z][A-Z0-9_]+"', ln)
            if not m:
                continue
            name = m.group(1)
            lo = max(0, i - 3)
            hi = min(len(lines), i + 4)
            ctx = " ".join(lines[lo:hi]).upper()
            if any(marker in ctx for marker in
                   ("RESERVED:", "LEGACY:", "SCANNER_SKIP_UNUSED")):
                _BUS_RESERVED_CACHE[name] = True
    return _BUS_RESERVED_CACHE.get(msg_type, False)


# ─── SQL CRUD-balance audit ────────────────────────────────────────────

_SQL_INSERT = re.compile(r'\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)', re.IGNORECASE)
_SQL_UPDATE = re.compile(r'\bUPDATE\s+(\w+)(?!\s*\()', re.IGNORECASE)
_SQL_DELETE = re.compile(r'\bDELETE\s+FROM\s+(\w+)', re.IGNORECASE)
_SQL_SELECT_FROM = re.compile(r'\bFROM\s+(\w+)', re.IGNORECASE)
_SQL_SELECT_KW = re.compile(r'\bSELECT\b', re.IGNORECASE)
_SQL_KW_PREFILTER = re.compile(r'\b(INSERT|UPDATE|DELETE|SELECT|CREATE)\s+(INTO|FROM|TABLE|\w+)\b', re.IGNORECASE)

_SQL_RESERVED = frozenset({
    "where", "select", "from", "join", "inner", "left", "right", "outer",
    "on", "as", "and", "or", "not", "in", "is", "null", "order", "by",
    "group", "having", "limit", "offset", "union", "all", "case", "when",
    "then", "else", "end", "set", "values", "into", "distinct",
})


def find_crud_imbalances(root: Path, min_count: int = 3) -> list[Finding]:
    """Scan for write-only / read-only SQL tables.

    v2.3: AST-aware — finds SQL inside multi-line triple-quoted strings,
    not just per-line matches.

    Flags:
      - Write-only: INSERTs >= min_count, SELECTs == 0 (meta_wisdom pattern)
      - Read-only: SELECTs >= 10, INSERTs == 0

    Skips test files, SQL reserved words, sqlite_* tables.
    """
    ops: dict[str, dict[str, set[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(set))

    project_root = root.parent if root.is_dir() else root.parent.parent
    scan_dirs = [root]
    for extra in ("scripts",):
        extra_dir = project_root / extra
        if extra_dir.exists():
            scan_dirs.append(extra_dir)

    def _match(text: str, file: str, line: int):
        for m in _SQL_INSERT.finditer(text):
            t = m.group(1).lower()
            if t not in _SQL_RESERVED and not t.startswith("sqlite_"):
                ops[t]["INSERT"].add((file, line))
        for m in _SQL_UPDATE.finditer(text):
            t = m.group(1).lower()
            if t not in _SQL_RESERVED and not t.startswith("sqlite_"):
                ops[t]["UPDATE"].add((file, line))
        for m in _SQL_DELETE.finditer(text):
            t = m.group(1).lower()
            if t not in _SQL_RESERVED and not t.startswith("sqlite_"):
                ops[t]["DELETE"].add((file, line))
        if _SQL_SELECT_KW.search(text):
            for m in _SQL_SELECT_FROM.finditer(text):
                t = m.group(1).lower()
                if t not in _SQL_RESERVED and not t.startswith("sqlite_"):
                    ops[t]["SELECT"].add((file, line))

    for sd in scan_dirs:
        for py in sd.rglob("*.py"):
            if "__pycache__" in py.parts or py.name.startswith("test_"):
                continue
            try:
                src = py.read_text(encoding="utf-8")
            except Exception:
                continue
            # AST pass — captures multi-line triple-quoted SQL
            try:
                tree = ast.parse(src, filename=str(py))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                            and _SQL_KW_PREFILTER.search(node.value)):
                        _match(node.value, str(py), node.lineno)
            except Exception:
                pass
            # Per-line pass — catches f-strings that AST stringification misses
            for line_idx, line in enumerate(src.splitlines(), start=1):
                if _SQL_KW_PREFILTER.search(line):
                    _match(line, str(py), line_idx)

    findings: list[Finding] = []
    for table, table_ops in sorted(ops.items()):
        n_insert = len(table_ops.get("INSERT", set()))
        n_update = len(table_ops.get("UPDATE", set()))
        n_select = len(table_ops.get("SELECT", set()))
        n_delete = len(table_ops.get("DELETE", set()))

        if n_insert + n_update + n_select + n_delete < min_count:
            continue

        if n_insert >= min_count and n_select == 0:
            first = next(iter(table_ops["INSERT"]))
            findings.append(Finding(
                kind="crud_write_only", severity="high",
                title=f"table `{table}` — {n_insert} INSERTs, 0 SELECTs",
                detail=(f"Write-only pattern: {n_insert} insert sites but nothing "
                        f"reads. UPDATE={n_update} DELETE={n_delete}"),
                file=first[0], line=first[1],
                extra={"n_insert": n_insert, "n_update": n_update,
                       "n_select": n_select, "n_delete": n_delete},
            ))
        elif n_select >= 10 and n_insert == 0:
            first = next(iter(table_ops["SELECT"]))
            findings.append(Finding(
                kind="crud_read_only", severity="medium",
                title=f"table `{table}` — 0 INSERTs, {n_select} SELECTs",
                detail=(f"Read-only pattern: {n_select} reads but no writes. "
                        f"UPDATE={n_update} DELETE={n_delete}"),
                file=first[0], line=first[1],
                extra={"n_insert": 0, "n_update": n_update,
                       "n_select": n_select, "n_delete": n_delete},
            ))
    return findings


# ─── Persistence gap detection ──────────────────────────────────────────

def _extract_class_fields_and_persistence(path: Path) -> list[dict]:
    """For each class in a file, find fields assigned in __init__/methods
    and cross-reference against save/load/persist methods.

    Returns list of dicts: {class, file, fields_written: {name: [lines]},
                            fields_saved: {name: [lines]}, fields_loaded: {name: [lines]},
                            shm_sourced: bool}

    Classes with a `# SHM_SOURCED` marker within ±3 lines of the class def
    are flagged — their field values come from /dev/shm (or another shared
    memory region), not from on-disk state, so persistence asymmetries are
    expected and should be suppressed.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return []
    src_lines = src.splitlines()

    def _class_has_shm_marker(class_line: int) -> bool:
        # Widened to 10 lines above because the marker is usually in a
        # multi-line block comment that sits above `class X:` — not on an
        # adjacent line. Scanner is 1-indexed AST vs 0-indexed src_lines,
        # so account for both offsets.
        #
        # Two markers suppress persistence-asymmetry findings for the class:
        #   SHM_SOURCED — state comes from shared memory (/dev/shm), not disk
        #   PERSISTENCE_BY_DESIGN — derived / observability / circuit-breaker
        #       state that is intentionally recomputed-on-boot rather than
        #       persisted (e.g. XSessionManager, Neuromodulator peak/trough)
        lo = max(0, class_line - 11)
        hi = min(len(src_lines), class_line + 2)
        ctx = " ".join(src_lines[lo:hi]).upper()
        return "SHM_SOURCED" in ctx or "PERSISTENCE_BY_DESIGN" in ctx

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        cls_name = node.name
        shm_sourced = _class_has_shm_marker(node.lineno)
        fields_written: dict[str, list[int]] = defaultdict(list)   # self._field = ...
        fields_saved: dict[str, list[int]] = defaultdict(list)     # in save/persist/to_dict methods
        fields_loaded: dict[str, list[int]] = defaultdict(list)    # in load/restore/from_dict methods
        # Fields that receive a falsy constant assignment anywhere (None, False,
        # 0, 0.0, "", [], {}). A field that is both set and reset to a falsy
        # constant is a transient flag — not persistable state.
        fields_falsy_written: set[str] = set()

        def _is_falsy_constant(n) -> bool:
            if isinstance(n, ast.Constant):
                return n.value in (None, False, 0, 0.0, "", b"")
            if isinstance(n, ast.List) and not n.elts:
                return True
            if isinstance(n, ast.Dict) and not n.keys:
                return True
            if isinstance(n, ast.Tuple) and not n.elts:
                return True
            if isinstance(n, ast.Set) and not n.elts:
                return True
            return False

        for item in ast.walk(node):
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn_name = item.name.lower()
            is_save = any(kw in fn_name for kw in ("save", "persist", "to_dict",
                                                     "serialize", "_dump", "checkpoint",
                                                     "get_state"))
            is_load = any(kw in fn_name for kw in ("load", "restore", "from_dict",
                                                     "deserialize", "_load", "recover",
                                                     "set_state", "restore_state"))
            is_init = fn_name == "__init__"

            for sub in ast.walk(item):
                # Look for self._field assignments and reads. Both plain
                # assignments (x = y) and annotated assignments (x: T = y —
                # common for typed __init__ fields) are tracked.
                assign_like: list[tuple[ast.expr, ast.expr | None, int]] = []
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        assign_like.append((t, sub.value, sub.lineno))
                elif isinstance(sub, ast.AnnAssign) and sub.target is not None:
                    assign_like.append((sub.target, sub.value, sub.lineno))
                for t, rhs, lineno in assign_like:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self" and t.attr.startswith("_")):
                        field_name = t.attr
                        if is_save:
                            fields_saved[field_name].append(lineno)
                        elif is_init or (not is_load):
                            fields_written[field_name].append(lineno)
                        if rhs is not None and _is_falsy_constant(rhs):
                            fields_falsy_written.add(field_name)
                        # If the RHS is a call to self._load_X() — that's
                        # an in-place load idiom (init-time bootstrap via
                        # a dedicated loader method). Credit as loaded.
                        if (isinstance(rhs, ast.Call)
                                and isinstance(rhs.func, ast.Attribute)
                                and isinstance(rhs.func.value, ast.Name)
                                and rhs.func.value.id == "self"
                                and ("load" in rhs.func.attr.lower()
                                     or "restore" in rhs.func.attr.lower()
                                     or "read" in rhs.func.attr.lower())):
                            fields_loaded[field_name].append(lineno)
                # self._field on the right side of an assignment in save/to_dict
                # (reading for serialization: data["key"] = self._field)
                if is_save and isinstance(sub, ast.Attribute):
                    if (isinstance(sub.value, ast.Name) and sub.value.id == "self"
                            and sub.attr.startswith("_") and not isinstance(sub.ctx, ast.Store)):
                        fields_saved[sub.attr].append(getattr(sub, "lineno", 0))
                # self._field on the left side of an assignment in load/from_dict
                if is_load and isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.value.id == "self" and t.attr.startswith("_")):
                            fields_loaded[t.attr].append(sub.lineno)
                # In-place restoration. The scanner otherwise flags fields as
                # "saved but never loaded" when the load idiom is a method call
                # rather than an assignment. Three patterns covered:
                #   self._field.load_state_dict(...)  (PyTorch)
                #   self._field.load(...) / .restore(...) / .from_dict(...)
                #   setattr(self, "_field", ...)  (dynamic restore in a loop)
                #
                # Recognized in load-named methods AND in __init__ (common
                # pattern: engine boot calls self._net.load(weights_path)
                # inline rather than via a separate _load_state method).
                if (is_load or is_init) and isinstance(sub, ast.Call):
                    func = sub.func
                    if isinstance(func, ast.Attribute) and func.attr in (
                            "load_state_dict", "load", "restore",
                            "from_dict", "deserialize", "read",
                            "load_weights", "load_from"):
                        inner = func.value
                        if (isinstance(inner, ast.Attribute)
                                and isinstance(inner.value, ast.Name)
                                and inner.value.id == "self"
                                and inner.attr.startswith("_")):
                            fields_loaded[inner.attr].append(sub.lineno)
                    elif (isinstance(func, ast.Name) and func.id == "setattr"
                          and len(sub.args) >= 2
                          and isinstance(sub.args[0], ast.Name)
                          and sub.args[0].id == "self"
                          and isinstance(sub.args[1], ast.Constant)
                          and isinstance(sub.args[1].value, str)
                          and sub.args[1].value.startswith("_")):
                        fields_loaded[sub.args[1].value].append(sub.lineno)
                # Also recognize setattr(self, attr, ...) where the attribute
                # name is a Name (parameter) rather than a Constant — common
                # inside helper closures (e.g. V5's _restore(name, attr, len)
                # driven by loop). When we can't resolve the literal name, we
                # fall back to a "class-wide setattr-used" flag so any
                # later-written field in the same load method gets credited.
                if is_load and isinstance(sub, ast.Call):
                    f2 = sub.func
                    if (isinstance(f2, ast.Name) and f2.id == "setattr"
                            and len(sub.args) >= 2
                            and isinstance(sub.args[0], ast.Name)
                            and sub.args[0].id == "self"
                            and isinstance(sub.args[1], ast.Name)):
                        # Dynamic attr name — credit all fields written
                        # elsewhere in this class that share a prefix/suffix
                        # pattern referenced literally in the same method
                        # (e.g. _restore("inner_body", "_ib_mults", 5) ...
                        # _restore(...) calls enumerate the attr names).
                        # Walk sibling string literals in the method body.
                        for sub2 in ast.walk(item):
                            if (isinstance(sub2, ast.Constant)
                                    and isinstance(sub2.value, str)
                                    and sub2.value.startswith("_")):
                                fields_loaded[sub2.value].append(sub.lineno)

        if fields_written:
            results.append({
                "class": cls_name,
                "file": str(path),
                "fields_written": dict(fields_written),
                "fields_saved": dict(fields_saved),
                "fields_loaded": dict(fields_loaded),
                "fields_falsy_written": fields_falsy_written,
                "shm_sourced": shm_sourced,
            })
    return results


def find_persistence_gaps(root: Path) -> list[Finding]:
    """Detect fields that are mutated at runtime but never persisted.

    Two sub-patterns:
      A) Written in __init__ or methods, but absent from ALL save/persist paths
         → field is lost on every restart (today's _total_updates_applied bug)
      B) Saved but not loaded (or loaded but not saved) → asymmetric persistence
         (today's watchdog save/load mismatch bug)
    """
    findings: list[Finding] = []
    # Only scan classes that HAVE persistence methods — we don't flag
    # classes that were never designed to persist state.
    # Skip fields that are runtime infrastructure (not state to persist)
    SKIP_FIELDS = frozenset({
        "_db_path", "_save_dir", "_save_path", "_log_path", "_lock",
        "_send_queue", "_recv_queue", "_bus", "_config", "_dna",
        "_cache", "_module_name", "_start_ts", "_cgn_client",
        "_registered", "_shm_path",
    })
    # Skip fields matching these PATTERNS — infrastructure, not state
    SKIP_SUFFIXES = ("_path", "_dir", "_file", "_conn", "_db", "_lock",
                     "_queue", "_app", "_client", "_engine", "_worker",
                     "_thread", "_process", "_pool", "_loop", "_event",
                     "_semaphore", "_condition", "_timer", "_logger",
                     "_handler", "_callback", "_hook", "_router",
                     "_translator", "_advisor", "_verifier", "_collector",
                     "_observer", "_monitor", "_proxy", "_bridge",
                     "_factory", "_builder", "_manager", "_scheduler")
    SKIP_PREFIXES = ("_is_", "_has_", "_should_", "_can_", "_needs_",
                     # Transient-cache / transient-flag prefixes. Fields named
                     # _last_*/_pending_*/_recovery_* are by convention
                     # session-scoped caches or one-shot boot flags that are
                     # expected to reset on restart.
                     "_last_", "_pending_", "_recovery_")
    # Transient-flag suffixes — boolean flags that are set then cleared each
    # tick (e.g. _inner_fresh, _outer_fresh).
    SKIP_SUFFIXES_FLAGS = ("_fresh", "_stimulus", "_flag", "_ready",
                           "_active", "_dirty")

    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        class_data = _extract_class_fields_and_persistence(py)
        for cd in class_data:
            has_save = bool(cd["fields_saved"])
            has_load = bool(cd["fields_loaded"])
            if not has_save and not has_load:
                continue  # Class has no persistence — skip

            # Pattern A: written but never saved
            # Suppress entirely if class is marked PERSISTENCE_BY_DESIGN /
            # SHM_SOURCED — same suppression as Pattern B below.
            if cd.get("shm_sourced"):
                continue
            falsy_written = cd.get("fields_falsy_written", set())
            for field_name, write_lines in cd["fields_written"].items():
                if field_name in SKIP_FIELDS:
                    continue
                if any(field_name.endswith(s) for s in SKIP_SUFFIXES):
                    continue
                if any(field_name.endswith(s) for s in SKIP_SUFFIXES_FLAGS):
                    continue
                if any(field_name.startswith(p) for p in SKIP_PREFIXES):
                    continue
                if field_name in cd["fields_saved"]:
                    continue
                # Transient flag: the field is assigned a falsy constant in at
                # least one place. Real persistable state is almost never
                # reset to None/False/0/"". Flag-reset pattern (set true, do
                # work, clear to false) lives only in memory by design.
                if field_name in falsy_written:
                    continue
                # Check if it's a trivial field (set once in __init__ to a constant)
                if len(write_lines) <= 1:
                    continue  # Only set in __init__, never mutated — not a gap
                # Skip if only written in 2 places and one is __init__
                # (common pattern: init + one conditional setter, not a counter)
                if len(write_lines) == 2:
                    continue  # Likely init + one setter, low risk
                findings.append(Finding(
                    kind="persistence_gap",
                    severity="high",
                    title=f"{cd['class']}.{field_name} — written {len(write_lines)}x, never persisted",
                    detail=(f"Field mutated at lines {write_lines[:5]} but absent from "
                            f"save/persist/to_dict methods. Lost on every restart."),
                    file=cd["file"],
                    line=write_lines[0],
                    extra={"class": cd["class"], "field": field_name,
                           "pattern": "written_not_saved",
                           "write_count": len(write_lines)},
                ))

            # Pattern B: save/load asymmetry
            # Skip entirely if class is marked # SHM_SOURCED — its field values
            # come from shared memory, not on-disk state files.
            if cd.get("shm_sourced"):
                continue
            all_skip = SKIP_FIELDS | {f for f in cd["fields_saved"]
                                       if any(f.endswith(s) for s in SKIP_SUFFIXES)}
            all_skip |= {f for f in cd["fields_loaded"]
                          if any(f.endswith(s) for s in SKIP_SUFFIXES)}
            saved_only = set(cd["fields_saved"]) - set(cd["fields_loaded"]) - all_skip
            loaded_only = set(cd["fields_loaded"]) - set(cd["fields_saved"]) - all_skip
            # Only flag asymmetry for state-like fields (counters, statuses,
            # EMAs, accumulators) — not model weights/biases/embeddings/buffers
            _WEIGHT_SUBSTRINGS = ("weight", "_w1", "_w2", "_w3", "_b1", "_b2", "_b3",
                                  "bias", "emb", "_lr", "momentum", "running_",
                                  "buffer", "replay", "_rng", "template_",
                                  "_alpha", "_beta", "_gamma", "_dim", "_hidden",
                                  "_index", "_node_", "_edge_", "_vector")
            def _is_state_field(name: str) -> bool:
                if any(s in name for s in _WEIGHT_SUBSTRINGS):
                    return False
                return True

            for field_name in saved_only:
                if not _is_state_field(field_name):
                    continue
                if field_name in cd["fields_written"]:
                    findings.append(Finding(
                        kind="persistence_asymmetry",
                        severity="medium",
                        title=f"{cd['class']}.{field_name} — saved but never loaded",
                        detail="Field persisted to disk but not restored on load. "
                               "Persistence is write-only — value resets to default on restart.",
                        file=cd["file"],
                        line=cd["fields_saved"][field_name][0],
                        extra={"class": cd["class"], "field": field_name,
                               "pattern": "saved_not_loaded"},
                    ))
            for field_name in loaded_only:
                if not _is_state_field(field_name):
                    continue
                findings.append(Finding(
                    kind="persistence_asymmetry",
                    severity="medium",
                    title=f"{cd['class']}.{field_name} — loaded but never saved",
                    detail="Field restored from disk but never written back. "
                           "Load succeeds only from a manually-created or legacy state file.",
                    file=cd["file"],
                    line=cd["fields_loaded"][field_name][0],
                    extra={"class": cd["class"], "field": field_name,
                           "pattern": "loaded_not_saved"},
                ))
    return findings


# ─── State transition completeness ──────────────────────────────────────

_LIFECYCLE_PAIRS: list[tuple[str, list[str]]] = [
    ("begin_",    ["end_", "finish_", "complete_"]),
    ("start_",    ["stop_", "end_", "finish_"]),
    ("enter_",    ["exit_", "leave_"]),
    ("open_",     ["close_"]),
    ("enable_",   ["disable_"]),
    ("acquire_",  ["release_"]),
    ("init_",     ["cleanup_", "teardown_", "shutdown_"]),
    ("_begin_",   ["_end_", "_finish_", "_complete_"]),
    ("_start_",   ["_stop_", "_end_", "_finish_"]),
    ("_enter_",   ["_exit_", "_leave_"]),
    ("_enable_",  ["_disable_"]),
]


def find_lifecycle_gaps(defs: list[MethodDef], calls: list[CallSite],
                        refs: list[CallSite]) -> list[Finding]:
    """Detect lifecycle method pairs where one side is called but the other isn't.

    E.g., _enter_graduating() called 5 times, _exit_graduating() or
    _leave_graduating() never called → possible resource/state leak.
    """
    findings: list[Finding] = []

    # Build lookup: (class_name, method_name) -> MethodDef
    def_lookup: dict[tuple[str | None, str], MethodDef] = {}
    for d in defs:
        def_lookup[(d.class_name, d.name)] = d

    # Count calls per method name (regardless of class for simplicity)
    call_counts: dict[str, int] = defaultdict(int)
    for c in calls:
        call_counts[c.target] += 1
    for r in refs:
        call_counts[r.target] += 1

    # Check each defined method against lifecycle patterns
    checked: set[str] = set()
    for d in defs:
        name = d.name
        for prefix, closers in _LIFECYCLE_PAIRS:
            if not name.startswith(prefix):
                continue
            stem = name[len(prefix):]
            if not stem:
                continue
            opener_calls = call_counts.get(name, 0)
            if opener_calls == 0:
                continue  # Opener never called — not a leak

            # Check if any closer exists AND is called
            closer_found = False
            closer_defined = False
            for closer_prefix in closers:
                closer_name = closer_prefix + stem
                if (d.class_name, closer_name) in def_lookup or \
                   (None, closer_name) in def_lookup:
                    closer_defined = True
                    if call_counts.get(closer_name, 0) > 0:
                        closer_found = True
                        break

            key = f"{d.class_name}.{name}" if d.class_name else name
            if key in checked:
                continue
            checked.add(key)

            if closer_defined and not closer_found:
                closer_names = [cp + stem for cp in closers]
                findings.append(Finding(
                    kind="lifecycle_gap",
                    severity="medium",
                    title=f"{d.class_name or ''}.{name}() called {opener_calls}x "
                          f"but closer never called",
                    detail=f"Opener `{name}` has matching closers defined "
                           f"({', '.join(closer_names)}) but none are ever invoked. "
                           f"Possible state/resource leak.",
                    file=d.file, line=d.line,
                    extra={"opener": name, "closer_candidates": closer_names,
                           "opener_calls": opener_calls,
                           "class": d.class_name},
                ))
    return findings


# ─── Config-to-constructor plumbing audit ────────────────────────────────

def _extract_class_init_params(root: Path) -> dict[str, dict]:
    """For each class with __init__, extract whether it accepts a config/cfg param.

    Returns {ClassName: {file, line, has_config_param: bool, param_name: str|None}}
    """
    result: dict[str, dict] = {}
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    # Recognize several idioms that effectively plumb TOML section
                    # values through — scanner otherwise false-positives on param
                    # names other than config/cfg. Added: dna, dna_params,
                    # params_config, config_path.
                    config_names = {"config", "cfg", "params", "settings", "options", "conf",
                                    "dna", "dna_params", "params_config", "config_path"}
                    param_names = [a.arg for a in item.args.args]
                    has_config = bool(config_names & set(param_names))
                    config_param = None
                    for pn in param_names:
                        if pn in config_names:
                            config_param = pn
                            break
                    # Also count __init__ as config-plumbed if its body calls a
                    # self-method whose name loads config (e.g. MoodEngine calls
                    # self._load_config() which internally reads titan_params).
                    if not has_config:
                        for sub in ast.walk(item):
                            if isinstance(sub, ast.Call):
                                f = sub.func
                                if (isinstance(f, ast.Attribute)
                                        and isinstance(f.value, ast.Name)
                                        and f.value.id == "self"
                                        and ("load_config" in f.attr
                                             or "load_toml" in f.attr
                                             or "load_params" in f.attr)):
                                    has_config = True
                                    config_param = f"self.{f.attr}()"
                                    break
                    result[node.name] = {
                        "file": str(py),
                        "line": item.lineno,
                        "has_config_param": has_config,
                        "param_name": config_param,
                        "all_params": param_names,
                    }
                    break
    return result


def _extract_toml_class_mappings(toml_paths: list[Path], root: Path) -> list[dict]:
    """Heuristically map TOML section names to class names.

    Strategy: for each TOML section, search the codebase for:
      1. get_params("section_name") calls
      2. Class names that match the section (e.g., [impulse] -> ImpulseEngine)
      3. Constructor calls that pass config sections
    """
    mappings: list[dict] = []

    # Collect all TOML sections
    sections: dict[str, str] = {}  # section_name -> toml_file
    for tp in toml_paths:
        if not tp.exists():
            continue
        try:
            with open(tp, "rb") as f:
                data = tomllib.load(f)
            for key in data:
                if isinstance(data[key], dict):
                    sections[key] = str(tp)
        except Exception:
            continue

    # For each section, try to find the class it should configure
    # Search for: ClassName() with no config, or get_params("section") not passed to ctor
    for section_name, toml_file in sections.items():
        # Heuristic: section_name → CamelCase class name
        # e.g., "impulse" → "ImpulseEngine", "titan_vm" → "TitanVM", "guardian" → "Guardian"
        # Also try: "meta_reasoning" → "MetaReasoningEngine"
        candidate_classes: list[str] = []
        parts = section_name.split("_")
        # Basic CamelCase
        camel = "".join(p.capitalize() for p in parts)
        candidate_classes.append(camel)
        # With common suffixes
        for suffix in ("Engine", "Worker", "Store", "Manager", "Client", "Monitor"):
            candidate_classes.append(camel + suffix)
        # Also try the raw section name as-is (e.g., "Guardian")
        candidate_classes.append(section_name.capitalize())
        # Special cases
        if section_name == "titan_vm":
            candidate_classes.append("TitanVM")

        mappings.append({
            "section": section_name,
            "toml_file": toml_file,
            "candidate_classes": candidate_classes,
        })
    return mappings


def find_config_plumbing_gaps(toml_paths: list[Path], root: Path) -> list[Finding]:
    """Detect TOML config sections that are defined but not plumbed into constructors.

    Cross-references:
      1. TOML sections in titan_params.toml / config.toml
      2. Class __init__ methods — do they accept config?
      3. get_params("section") calls — is the section actually read?
      4. Constructor call sites — is config passed?
    """
    findings: list[Finding] = []
    class_info = _extract_class_init_params(root)
    mappings = _extract_toml_class_mappings(toml_paths, root)

    # Scan for get_params("section") calls in codebase
    sections_consumed: set[str] = set()
    get_params_pattern = re.compile(r'get_params\(\s*["\']([^"\']+)["\']\s*\)')
    config_get_pattern = re.compile(
        r'(?:config|cfg|_full_config|_config|spirit_config|_meta_cfg)\s*'
        r'(?:\.\s*get\s*\(\s*["\']([^"\']+)["\']\s*\)|'  # .get("section")
        r'\[\s*["\']([^"\']+)["\']\s*\])')                 # ["section"]

    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in get_params_pattern.finditer(src):
            sections_consumed.add(m.group(1))
        for m in config_get_pattern.finditer(src):
            consumed = m.group(1) or m.group(2)
            if consumed:
                sections_consumed.add(consumed)

    # Also check for direct tomllib.load reads of titan_params.toml
    # These bypass the config plumbing but DO consume the section
    direct_toml_pattern = re.compile(
        r'tomllib\.load.*titan_params|tomli\.load.*titan_params|'
        r'open.*titan_params.*tomllib')
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if direct_toml_pattern.search(src):
            # File reads titan_params directly — extract which sections it accesses
            for m in re.finditer(r'(?:data|params|_params|toml_data|td)\s*'
                                  r'(?:\.\s*get\s*\(\s*["\']([^"\']+)["\']\s*\)|'
                                  r'\[\s*["\']([^"\']+)["\']\s*\])', src):
                consumed = m.group(1) or m.group(2)
                if consumed:
                    sections_consumed.add(consumed)

    # Skip sections that are consumed via ANY path
    # Skip known non-class sections (metadata, per-Titan overrides, etc.)
    SKIP_SECTIONS = frozenset({
        "meta_reasoning_dna", "cognitive_contracts_dna",
        "kin", "kin.peers", "kin.peers.T1", "kin.peers.T2", "kin.peers.T3",
        "api", "network", "info_banner", "privacy", "frontend",
        "observatory", "endurance", "inference",
    })

    for mapping in mappings:
        section = mapping["section"]
        if section in SKIP_SECTIONS:
            continue
        if section in sections_consumed:
            continue

        # Check if any candidate class exists and lacks config param.
        # Prefer a candidate that DOES have config over one that doesn't —
        # otherwise false-positive on helper/child classes (e.g. [sphere_clock]
        # → SphereClock individual clock has no config, but SphereClockEngine
        # parent does plumb). Only flag if NO matching candidate has config.
        matched_with_config = None
        matched_without_config = None
        for candidate in mapping["candidate_classes"]:
            if candidate not in class_info:
                continue
            if class_info[candidate]["has_config_param"]:
                matched_with_config = candidate
                break
            elif matched_without_config is None:
                matched_without_config = candidate

        matched_class = matched_with_config or matched_without_config

        if matched_class:
            ci = class_info[matched_class]
            if not ci["has_config_param"]:
                findings.append(Finding(
                    kind="config_unplumbed",
                    severity="high",
                    title=f"[{section}] → {matched_class}.__init__() takes no config param",
                    detail=(f"TOML section [{section}] defined in {mapping['toml_file']} "
                            f"but {matched_class}.__init__ at {ci['file']}:{ci['line']} "
                            f"has no config/cfg parameter. Section values are ignored."),
                    file=ci["file"], line=ci["line"],
                    extra={"section": section, "class": matched_class,
                           "pattern": "section_no_config_param",
                           "toml_file": mapping["toml_file"]},
                ))
            else:
                # Class accepts config but section isn't consumed via get_params
                # This might mean the section IS passed but via a different path
                # Lower severity — it could be plumbed through spirit_config etc.
                pass
        elif section not in sections_consumed:
            # Section exists in TOML but no matching class found and no code reads it
            # This is already caught by find_unused_config_sections — skip to avoid duplication
            pass

    return findings


# ─── Runtime cross-check (live Titan API) ──────────────────────────────

def fetch_runtime_evidence(api_base: str = "http://127.0.0.1:7777",
                           timeout: float = 3.0) -> dict:
    """Query Titan's live API for runtime evidence of subsystem activity.

    Returns dict with:
      - active_bus_types: set of message types with observed emissions
      - healthy_services: set of service names responding
      - available: whether API was reachable
    """
    import urllib.request, urllib.error, glob

    evidence = {
        "active_bus_types": set(),
        "healthy_services": set(),
        "available": False,
        "source": api_base,
    }

    def _get(path):
        try:
            with urllib.request.urlopen(api_base + path, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None

    if not _get("/health"):
        return evidence
    evidence["available"] = True

    # bus-health — META_CGN_SIGNAL telemetry
    bus_h = _get("/v4/bus-health")
    if bus_h:
        data = bus_h.get("data", bus_h) if isinstance(bus_h, dict) else {}
        if isinstance(data, dict):
            producers = data.get("producers")
            if isinstance(producers, list) and any(
                    p.get("total_emissions", 0) > 0 for p in producers
                    if isinstance(p, dict)):
                evidence["active_bus_types"].add("META_CGN_SIGNAL")
            if data.get("total_emission_rate_1min_hz", 0) > 0:
                evidence["active_bus_types"].add("META_CGN_SIGNAL")

    # Brain log fallback for bus types not exposed via dedicated endpoints
    for log_path in glob.glob("/tmp/titan*brain.log")[:1]:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 3_000_000))
                tail = f.read()
        except Exception:
            continue
        for m in re.finditer(
                r'\b(TITAN_SELF_STATE|FILTER_DOWN_V5|STATE_SNAPSHOT|'
                r'OBSERVABLES_SNAPSHOT|SPHERE_PULSE|IMPULSE|BIG_PULSE|'
                r'GREAT_PULSE|BODY_STATE|MIND_STATE|EPOCH_TICK|SPEAK_REQUEST|'
                r'SPEAK_RESULT|CGN_TRANSITION|CGN_WEIGHTS_MAJOR|CGN_IMPASSE|'
                r'NEUROMOD_UPDATE|HORMONE_FIRED|EXPRESSION_FIRED|'
                r'META_CGN_SIGNAL|MODULE_HEARTBEAT|ACTION_RESULT|'
                r'OUTER_OBSERVATION|REFLEX_SIGNAL|TEACHER_SIGNALS)\b', tail):
            evidence["active_bus_types"].add(m.group(1))

    # Services diagnostics
    svc = _get("/v4/services") or _get("/v4/inner-trinity")
    if svc and isinstance(svc, dict):
        data = svc.get("data", svc)
        if isinstance(data, dict):
            for name, info in data.items():
                if isinstance(info, dict) and any(
                        info.get(k) for k in
                        ("active", "running", "healthy", "last_update",
                         "count", "sessions", "vocab")):
                    evidence["healthy_services"].add(name.lower())

    return evidence


def cross_check_with_runtime(findings: list[Finding],
                              evidence: dict) -> list[Finding]:
    """Promote/demote findings based on runtime evidence. Emits new
    'static_runtime_divergence' findings when static says dead but
    runtime shows activity — highest-value signal for static gaps."""
    if not evidence.get("available"):
        for f in findings:
            f.extra["runtime_evidence"] = "api_unavailable"
        return findings

    out: list[Finding] = []
    active_bus = evidence.get("active_bus_types", set())

    for f in findings:
        if f.kind in ("bus_dead_msg", "bus_dead_handler", "bus_unused_type"):
            m = re.search(r"msg\s+(\w+)", f.title)
            if m and m.group(1) in active_bus:
                msg_type = m.group(1)
                f.extra["runtime_evidence"] = "active_at_runtime"
                f.severity = "info"
                out.append(Finding(
                    kind="static_runtime_divergence", severity="high",
                    title=f"static:dead, runtime:alive — {msg_type}",
                    detail=(f"{f.kind} flagged {msg_type} as dead but runtime "
                            f"evidence shows it's active. Static scanner "
                            f"missed a dynamic dispatch — refine detection."),
                    file=f.file, line=f.line,
                    extra={"msg_type": msg_type, "original_kind": f.kind},
                ))
            else:
                f.extra["runtime_evidence"] = "not_observed"
        out.append(f)

    return out


# ─── Runtime overlay ───────────────────────────────────────────────────

def overlay_runtime_calls(findings: list[Finding], log_paths: list[Path],
                          since: str = "") -> None:
    """Annotate static findings with runtime observation from brain logs.

    For each orphan finding, grep the log for the method name. If it
    appears as `[...method_name...]` or ` method_name(` or similar, mark
    the finding with `extra["runtime_seen"] = True`. This demotes orphans
    that are actually called dynamically.

    Args:
      findings: list to annotate in place
      log_paths: brain log files to scan
      since: optional "HH:MM:SS" cutoff — only consider lines after this time
    """
    if not log_paths:
        return

    # Build a single compiled regex for all orphan method names.
    orphan_names: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.kind != "orphan":
            continue
        # Extract bare method name (title is 'ClassName.method' or 'method')
        method = f.title.split(".")[-1]
        orphan_names[method].append(f)

    if not orphan_names:
        return

    # Compile a fast-scan regex. Limit to word-boundary matches to reduce
    # false positives (e.g., "save" as substring of "saved_ts").
    name_pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in orphan_names.keys()) + r")\b"
    )

    seen_names: set[str] = set()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        try:
            # Stream-read — logs can be huge.
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if since and line[:8] < since:
                        continue
                    for m in name_pattern.finditer(line):
                        seen_names.add(m.group(1))
                        if len(seen_names) == len(orphan_names):
                            break
                    if len(seen_names) == len(orphan_names):
                        break
        except Exception:
            continue

    # Annotate + promote/demote severity
    for name, items in orphan_names.items():
        seen = name in seen_names
        for f in items:
            f.extra["runtime_seen"] = seen
            if seen and f.severity == "high":
                f.severity = "info"   # demoted: was flagged as orphan but runtime shows activity


# ─── Allowlist ─────────────────────────────────────────────────────────

def load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line)
    return names


# ─── Report ────────────────────────────────────────────────────────────

def format_path(p: str, root: str) -> str:
    try:
        rel = Path(p).relative_to(Path(root).parent if Path(root).is_file()
                                  else Path(root))
        return str(rel)
    except ValueError:
        return p


def print_report(findings: list[Finding], n_files: int, n_defs: int,
                 n_calls: int, root: str, head_limit: int = 40) -> None:
    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_kind[f.kind].append(f)

    print(f"\narch_map dead-wiring — scan of {root}")
    print(f"  {n_files} files / {n_defs} defs / {n_calls} calls\n")

    titles = {
        "orphan": "ORPHAN METHODS (defined, zero callers)",
        "pair_gap": "PAIR-CLOSURE GAPS (query called, record never called)",
        "unused_config": "UNUSED CONFIG SECTIONS (toml-defined, code never reads)",
        "rfp_missing": "rFP ENTITIES MISSING IN CODE",
        "bus_dead_msg": "BUS DEAD MESSAGES (published, no subscribers)",
        "bus_dead_handler": "BUS DEAD HANDLERS (subscribed, no publishers)",
        "bus_unused_type": "BUS UNUSED TYPES (defined in bus.py, never touched)",
        "crud_write_only": "DB WRITE-ONLY TABLES (INSERTs accumulate, no SELECT)",
        "crud_read_only": "DB READ-ONLY TABLES (SELECTs only, no INSERT)",
        "static_runtime_divergence": "⚡ STATIC-RUNTIME DIVERGENCE (static:dead, runtime:alive)",
        "new_orphan": "🔺 NEW ORPHANS (appeared since last scan)",
        "resolved_orphan": "✅ RESOLVED ORPHANS (wired since last scan)",
        "persistence_gap": "🔥 PERSISTENCE GAPS (field mutated but never saved to disk)",
        "persistence_asymmetry": "⚠ PERSISTENCE ASYMMETRY (save/load field mismatch)",
        "lifecycle_gap": "🔄 LIFECYCLE GAPS (opener called, closer never called)",
        "config_unplumbed": "🔧 CONFIG UNPLUMBED (TOML section, constructor takes no config)",
    }

    for kind in ["new_orphan", "persistence_gap", "config_unplumbed",
                 "persistence_asymmetry", "lifecycle_gap",
                 "static_runtime_divergence", "bus_dead_msg", "bus_dead_handler",
                 "crud_write_only", "pair_gap", "orphan", "bus_unused_type",
                 "crud_read_only", "unused_config", "rfp_missing", "resolved_orphan"]:
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"─── {titles[kind]} ({len(items)}) ───")
        for f in items[:head_limit]:
            loc = f"{format_path(f.file, root)}:{f.line}" if f.line else format_path(f.file, root)
            sev = {"high": "🔴", "medium": "🟡", "info": "·"}.get(f.severity, " ")
            print(f"  {sev} {loc}")
            print(f"     {f.title}")
            print(f"     {f.detail}")
            # Git context (if annotated)
            git = f.extra.get("git")
            if git:
                intro = git.get("introduced")
                if intro:
                    print(f"     ├ introduced: {intro['hash']} ({intro['date']}) {intro['subject'][:60]}")
                mt = git.get("method_last_touch")
                if mt:
                    print(f"     ├ method last touched: {mt['hash']} ({mt['date']}) {mt['subject'][:60]}")
                drift = git.get("paired_site_drift")
                if drift:
                    print(f"     └ ⚠ {drift['msg']}")
        if len(items) > head_limit:
            print(f"  ... and {len(items) - head_limit} more")
        print()

    # Summary
    print("═══ SUMMARY ═══")
    for kind in ["persistence_gap", "config_unplumbed", "persistence_asymmetry",
                 "lifecycle_gap", "orphan", "pair_gap", "unused_config",
                 "rfp_missing", "bus_dead_msg", "bus_dead_handler", "bus_unused_type"]:
        n = len(by_kind.get(kind, []))
        if n > 0:
            # Count runtime-seen-demoted orphans separately (if overlay ran)
            if kind == "orphan":
                demoted = sum(1 for f in by_kind[kind] if f.extra.get("runtime_seen"))
                if demoted > 0:
                    print(f"  {titles[kind].split(' (')[0]:50s} {n}  ({demoted} seen at runtime — demoted)")
                    continue
            print(f"  {titles[kind].split(' (')[0]:50s} {n}")


# ─── Temporal degradation detection ─────────────────────────────────────

BASELINE_PATH = Path("data/dead_wiring_baseline.json")


def _load_baseline() -> dict:
    """Load previous scan baseline (orphan names + counts)."""
    if BASELINE_PATH.exists():
        try:
            with open(BASELINE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_baseline(findings: list[Finding]) -> None:
    """Save current orphan set as baseline for next comparison."""
    import time
    orphans = set()
    for f in findings:
        if f.kind == "orphan":
            orphans.add(f.title)
    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "orphan_count": len(orphans),
        "orphans": sorted(orphans),
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)


def detect_temporal_changes(findings: list[Finding]) -> list[Finding]:
    """Compare current orphans against previous baseline.

    Produces NEW_ORPHAN findings for methods that weren't orphans last scan
    (introduced since last session) and RESOLVED findings for methods that
    were orphans but are no longer (someone wired them). Saves current state
    as the new baseline.
    """
    import time
    baseline = _load_baseline()
    if not baseline:
        _save_baseline(findings)
        return []

    prev_orphans = set(baseline.get("orphans", []))
    curr_orphans = {f.title for f in findings if f.kind == "orphan"}
    prev_ts = baseline.get("timestamp", "unknown")

    new_findings: list[Finding] = []

    # New orphans (appeared since last scan)
    new_orphans = curr_orphans - prev_orphans
    for name in sorted(new_orphans):
        new_findings.append(Finding(
            kind="new_orphan",
            severity="high",
            title=name,
            detail=f"NEW since last scan ({prev_ts}) — method became unreachable",
        ))

    # Resolved orphans (were orphans, now have callers)
    resolved = prev_orphans - curr_orphans
    for name in sorted(resolved):
        new_findings.append(Finding(
            kind="resolved_orphan",
            severity="info",
            title=name,
            detail=f"RESOLVED since last scan ({prev_ts}) — now has callers",
        ))

    if new_orphans or resolved:
        print(f"\n  Temporal change: +{len(new_orphans)} new orphans, "
              f"-{len(resolved)} resolved (vs {prev_ts})")

    # Save current as new baseline
    _save_baseline(findings)

    return new_findings


# ─── Git context annotation ────────────────────────────────────────────

def _git_cmd(args: list[str], cwd: str | None = None, timeout: int = 5) -> str:
    """Run a git command, return stdout or empty string on failure."""
    import subprocess
    try:
        r = subprocess.run(
            ["git"] + args, capture_output=True, text=True,
            timeout=timeout, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def annotate_git_context(findings: list[Finding], max_findings: int = 30) -> None:
    """Enrich orphan/pair_gap findings with git history.

    For each finding (up to *max_findings* to avoid long runtimes):
      - **introduced**: commit + date when the method name first appeared
      - **last_modified**: most recent commit touching the file
      - **paired_site_drift** (pair_gap only): commits that touched the
        query-side method but NOT the record-side in the same commit

    Results are added to ``finding.extra["git"]``.
    """
    # Only annotate actionable finding types
    eligible = [f for f in findings
                if f.kind in ("orphan", "pair_gap") and f.file]
    if not eligible:
        return

    annotated = 0
    for f in eligible[:max_findings]:
        git_info: dict = {}
        file_path = f.file
        method_name = f.extra.get("method_name", "") or f.title.split(".")[-1].split("(")[0].strip()

        # 1. Introduction: when was `def <method>` first added?
        if method_name:
            intro = _git_cmd([
                "log", "--all", "--diff-filter=A",
                "-S", f"def {method_name}(",
                "--format=%h %aI %s",
                "--", file_path,
            ])
            if intro:
                # Take the oldest (last line)
                lines = intro.strip().split("\n")
                oldest = lines[-1] if lines else ""
                if oldest:
                    parts = oldest.split(" ", 2)
                    git_info["introduced"] = {
                        "hash": parts[0],
                        "date": parts[1][:10] if len(parts) > 1 else "?",
                        "subject": parts[2] if len(parts) > 2 else "?",
                    }

        # 2. Last modification: most recent commit touching this file
        last_mod = _git_cmd([
            "log", "-1",
            "--format=%h %aI %s",
            "--", file_path,
        ])
        if last_mod:
            parts = last_mod.split(" ", 2)
            git_info["last_modified"] = {
                "hash": parts[0],
                "date": parts[1][:10] if len(parts) > 1 else "?",
                "subject": parts[2] if len(parts) > 2 else "?",
            }

        # 3. Method-level last touch (git log -S for the specific method)
        if method_name:
            method_touch = _git_cmd([
                "log", "-1",
                "-S", f"def {method_name}(",
                "--format=%h %aI %s",
                "--", file_path,
            ])
            if method_touch:
                parts = method_touch.split(" ", 2)
                git_info["method_last_touch"] = {
                    "hash": parts[0],
                    "date": parts[1][:10] if len(parts) > 1 else "?",
                    "subject": parts[2] if len(parts) > 2 else "?",
                }

        # 4. For pair_gap findings — check if query-side was modified
        #    without updating record-side
        if f.kind == "pair_gap":
            query_method = f.extra.get("query_method", "")
            record_method = f.extra.get("record_method", method_name)
            if query_method:
                # Commits touching the query side
                q_commits = _git_cmd([
                    "log", "--format=%h",
                    "-S", f"def {query_method}(",
                    "--", file_path,
                ])
                # Commits touching the record side
                r_commits = _git_cmd([
                    "log", "--format=%h",
                    "-S", f"def {record_method}(",
                    "--", file_path,
                ])
                q_set = set(q_commits.split("\n")) if q_commits else set()
                r_set = set(r_commits.split("\n")) if r_commits else set()
                drift = q_set - r_set - {""}
                if drift:
                    git_info["paired_site_drift"] = {
                        "query_only_commits": list(drift)[:5],
                        "msg": f"{len(drift)} commit(s) touched {query_method} but not {record_method}",
                    }

        if git_info:
            f.extra["git"] = git_info
            annotated += 1

    if annotated:
        print(f"\n  Git context annotated: {annotated} finding(s)")


# ─── Entry point ───────────────────────────────────────────────────────

def run(root: str = "titan_plugin",
        tomls: list[str] | None = None,
        allowlist_path: str | None = None,
        rfp_path: str | None = None,
        json_output: bool = False,
        head_limit: int = 40,
        bus_flow: bool = False,
        runtime_overlay_log: list[str] | None = None,
        with_runtime: bool = False,
        runtime_api_base: str = "http://127.0.0.1:7777",
        git_context: bool = False) -> dict:
    """Run the full scan. Returns structured result dict."""
    root_path = Path(root)
    if not root_path.exists():
        print(f"ERROR: scan root {root} does not exist", file=sys.stderr)
        return {"error": "root_missing"}

    allowlist = set()
    if allowlist_path:
        allowlist = load_allowlist(Path(allowlist_path))

    # v1.1: scan additional directories for reachability (scripts/, tests/)
    # so dispatch-dict + entry_fn= refs outside titan_plugin/ register.
    project_root = root_path.parent if root_path.is_dir() else root_path.parent.parent
    scan_roots = [root_path]
    for extra in ("scripts", "tests"):
        extra_dir = project_root / extra
        if extra_dir.exists():
            scan_roots.append(extra_dir)

    defs, calls, refs, n_files = scan_tree(scan_roots)

    # Filter defs to only those INSIDE the primary root — scripts/+tests/ are
    # scanned for their call/reference content, not their orphan candidates.
    defs = [d for d in defs if Path(d.file).resolve().is_relative_to(
        root_path.resolve())]

    toml_paths = [Path(t) for t in (tomls or [
        "titan_plugin/titan_params.toml",
        "titan_plugin/config.toml",
    ])]

    findings: list[Finding] = []
    findings.extend(find_orphans(defs, calls, refs, allowlist))
    findings.extend(find_pair_gaps(defs, calls, refs))
    findings.extend(find_unused_config_sections(toml_paths, root_path))

    if rfp_path:
        findings.extend(verify_rfp(Path(rfp_path), defs, root_path, toml_paths))

    # v4: persistence gap detection + state transition completeness + config plumbing
    findings.extend(find_persistence_gaps(root_path))
    findings.extend(find_lifecycle_gaps(defs, calls, refs))
    findings.extend(find_config_plumbing_gaps(toml_paths, root_path))

    if bus_flow:
        known_types = extract_bus_message_types(root_path / "bus.py")
        # v2.1: detect helper functions that publish bus messages (e.g.
        # emit_meta_cgn_signal -> META_CGN_SIGNAL). Calls to these count
        # as publishers even though they don't invoke make_msg directly.
        helper_pubs = extract_helper_publishers(root_path, known_types)
        publishers, subscribers = scan_bus_publishers_and_subscribers(
            root_path, known_types, helper_publishers=helper_pubs)
        findings.extend(find_bus_flow_imbalances(publishers, subscribers, known_types))

        # v2.3: SQL CRUD-balance — write-only / read-only table patterns
        findings.extend(find_crud_imbalances(root_path))

    if runtime_overlay_log:
        overlay_runtime_calls(findings, [Path(p) for p in runtime_overlay_log])

    # v2.3: cross-check static findings with live Titan API evidence.
    # Demotes "dead" findings when runtime confirms activity + promotes
    # STATIC-RUNTIME DIVERGENCE findings (our static scanner missed a path).
    runtime_evidence = None
    if with_runtime:
        runtime_evidence = fetch_runtime_evidence(api_base=runtime_api_base)
        findings = cross_check_with_runtime(findings, runtime_evidence)

    # v3: annotate findings with git history (introduction, last mod, drift)
    if git_context:
        annotate_git_context(findings)

    # v3: temporal degradation — compare against previous baseline
    temporal_findings = detect_temporal_changes(findings)
    findings.extend(temporal_findings)

    result = {
        "root": root,
        "n_files": n_files,
        "n_defs": len(defs),
        "n_calls": len(calls),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "orphan": sum(1 for f in findings if f.kind == "orphan"),
            "orphan_runtime_seen": sum(1 for f in findings
                                        if f.kind == "orphan" and f.extra.get("runtime_seen")),
            "pair_gap": sum(1 for f in findings if f.kind == "pair_gap"),
            "unused_config": sum(1 for f in findings if f.kind == "unused_config"),
            "rfp_missing": sum(1 for f in findings if f.kind == "rfp_missing"),
            "bus_dead_msg": sum(1 for f in findings if f.kind == "bus_dead_msg"),
            "bus_dead_handler": sum(1 for f in findings if f.kind == "bus_dead_handler"),
            "bus_unused_type": sum(1 for f in findings if f.kind == "bus_unused_type"),
            "crud_write_only": sum(1 for f in findings if f.kind == "crud_write_only"),
            "crud_read_only": sum(1 for f in findings if f.kind == "crud_read_only"),
            "static_runtime_divergence": sum(1 for f in findings
                                              if f.kind == "static_runtime_divergence"),
            "new_orphan": sum(1 for f in findings if f.kind == "new_orphan"),
            "resolved_orphan": sum(1 for f in findings if f.kind == "resolved_orphan"),
            "persistence_gap": sum(1 for f in findings if f.kind == "persistence_gap"),
            "persistence_asymmetry": sum(1 for f in findings if f.kind == "persistence_asymmetry"),
            "lifecycle_gap": sum(1 for f in findings if f.kind == "lifecycle_gap"),
            "config_unplumbed": sum(1 for f in findings if f.kind == "config_unplumbed"),
        },
        "runtime_evidence": {
            "available": runtime_evidence["available"] if runtime_evidence else False,
            "active_bus_types_count": len(runtime_evidence["active_bus_types"]) if runtime_evidence else 0,
            "healthy_services_count": len(runtime_evidence["healthy_services"]) if runtime_evidence else 0,
        } if with_runtime else None,
    }

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print_report(findings, n_files, len(defs), len(calls), root, head_limit)

    return result


def main():
    p = argparse.ArgumentParser(description="Dead-wiring static analysis for Titan codebase.")
    p.add_argument("--root", default="titan_plugin",
                   help="Directory to scan (default: titan_plugin)")
    p.add_argument("--tomls", nargs="*",
                   default=["titan_plugin/titan_params.toml",
                            "titan_plugin/config.toml"],
                   help="TOML files to trace section usage")
    p.add_argument("--allowlist", default="scripts/arch_map_dead_wiring_allowlist.txt",
                   help="Path to allowlist (one method name per line)")
    p.add_argument("--rfp", default=None,
                   help="Verify implementation of an rFP markdown file")
    p.add_argument("--bus-flow", action="store_true",
                   help="Enable bus-flow audit (publisher/subscriber cross-reference)")
    p.add_argument("--runtime-overlay", nargs="*", default=None,
                   help="One or more brain log paths to cross-check orphan findings "
                        "against runtime observation. Orphans mentioned in the log "
                        "are demoted to 'info' severity.")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of human report")
    p.add_argument("--head", type=int, default=40,
                   help="Max findings to show per category (default: 40)")
    p.add_argument("--with-runtime", action="store_true",
                   help="Cross-check findings against Titan's live API "
                        "(/v4/bus-health, /v4/services). Findings that static "
                        "calls dead but runtime confirms active become "
                        "DIVERGENCE findings (surfaces static-scanner gaps).")
    p.add_argument("--runtime-api-base", default="http://127.0.0.1:7777",
                   help="Titan API base URL for --with-runtime (default T1)")
    p.add_argument("--git-context", action="store_true",
                   help="Annotate orphan/pair_gap findings with git history "
                        "(introduction commit, last mod, paired-site drift)")
    p.add_argument("--all", action="store_true",
                   help="Enable all lenses (bus-flow + CRUD + runtime overlay "
                        "+ --with-runtime + --git-context)")
    args = p.parse_args()

    bus_flow = args.bus_flow or args.all
    runtime_overlay = args.runtime_overlay
    if args.all and runtime_overlay is None:
        runtime_overlay = ["/tmp/titan_brain.log"]
    with_runtime = args.with_runtime or args.all
    git_context = args.git_context or args.all

    run(root=args.root, tomls=args.tomls, allowlist_path=args.allowlist,
        rfp_path=args.rfp, json_output=args.json, head_limit=args.head,
        bus_flow=bus_flow, runtime_overlay_log=runtime_overlay,
        with_runtime=with_runtime, runtime_api_base=args.runtime_api_base,
        git_context=git_context)


if __name__ == "__main__":
    main()
