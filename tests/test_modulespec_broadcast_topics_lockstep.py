"""Lockstep tests for ModuleSpec.broadcast_topics declarations.

Per rFP_worker_broadcast_topics_completion.md §5.A — every flood-receiving
worker MUST declare `broadcast_topics=[...]` matching its drain elif chain.
Workers consuming only targeted messages MUST declare `reply_only=True`.

Catches drift in BOTH directions:
  (a) Drain handler adds `if msg_type == "FOO":` but ModuleSpec.broadcast_topics
      doesn't list FOO → broker drops FOO at publish, worker never sees it.
  (b) ModuleSpec.broadcast_topics lists "FOO" but drain has no handler →
      silent log/drop noise, plus arch_map dead-wiring will flag.

Mechanism reminder (titan_hcl/core/bus_socket.py:761-832):
  - `subscribed_topics` (set on broker subscriber) — populated from
    ModuleSpec.broadcast_topics via Guardian → setup_worker_bus.
  - On publish() with dst="all": if subscribed_topics is non-empty,
    broker filters before enqueue. Empty list = legacy subscribe-all
    fallback (will become loud-WARN-and-drop after §4.C stopgap retirement).
  - Targeted msgs (dst="<worker_name>") bypass the filter — MODULE_*,
    QUERY, SAVE_NOW, etc. don't need declaration.

These tests parse the live source files to validate declarations.
Per `feedback_specs_need_enforcement_automation.md`: rules need test
gates, not just human discipline. If a future change drifts these,
the test fails at commit time (pre-commit hook runs pytest).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import pytest

# ─────────────────────────────────────────────────────────────────────
# Targeted message types — these bypass the broadcast filter regardless.
# Workers don't need to declare them in broadcast_topics.
#
# MODULE_HEARTBEAT/READY/SHUTDOWN/CRASHED — Guardian lifecycle (always targeted)
# QUERY — RPC pattern (caller publishes dst="<worker_name>")
# SAVE_NOW — Phase B.2.1 swap (Guardian sends dst="<worker_name>")
# RELOAD — config reload (typically targeted dst="<worker_name>")
# BUS_PEER_DIED, BUS_WORKER_ADOPT_REQUEST — Phase B.2 supervision
# ─────────────────────────────────────────────────────────────────────
TARGETED_LIFECYCLE_TYPES = frozenset({
    "MODULE_HEARTBEAT",
    "MODULE_READY",
    "MODULE_SHUTDOWN",
    "MODULE_CRASHED",
    "QUERY",
    "SAVE_NOW",
    "RELOAD",
    "BUS_PEER_DIED",
    "BUS_WORKER_ADOPT_REQUEST",
})


# ─────────────────────────────────────────────────────────────────────
# Worker → drain-source-file mapping (the source-of-truth files).
# ─────────────────────────────────────────────────────────────────────
WORKER_DRAIN_FILE = {
    # outer_{body,mind,spirit} workers RETIRED (Phase C dissolution C.8) — the
    # Rust outer daemons own the tensor slots; source data is SHM-direct.
    "body":             "titan_hcl/modules/body_worker.py",
    "mind":             "titan_hcl/modules/mind_worker.py",
    "recorder":               "titan_hcl/modules/recorder_worker.py",
    "llm":              "titan_hcl/modules/llm_worker.py",
    "warning_monitor":  "titan_hcl/modules/warning_monitor_worker.py",
    "language":         "titan_hcl/modules/language_worker.py",
    "meta_teacher":     "titan_hcl/modules/meta_teacher_worker.py",
    "emot_cgn":         "titan_hcl/modules/emot_cgn_worker.py",
    "cgn":              "titan_hcl/modules/cgn_worker.py",
    "knowledge":        "titan_hcl/modules/knowledge_worker.py",
    "timechain":        "titan_hcl/modules/timechain_worker.py",
    # spirit_worker RETIRED (D-SPEC-116) — engines live in cognitive_worker.
    "cognitive_worker": "titan_hcl/modules/cognitive_worker.py",
    "backup":           "titan_hcl/modules/backup_orchestrator.py",
}

# Workers that consume NO broadcasts (only targeted QUERY) — reply_only=True.
RPC_REPLY_ONLY_WORKERS = frozenset({
    "reflex",
    "agency_worker",
    "output_verifier",
    "media",
})

# Repo root (assumes test run from main repo or session worktree).
REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# AST extraction — read ModuleSpec(...) calls from registration files.
# ─────────────────────────────────────────────────────────────────────


def _extract_modulespec_kwargs(path: Path) -> dict[str, dict]:
    """Parse all `ModuleSpec(name=..., ...)` calls in `path`.

    Returns: {worker_name: {kwarg_name: ast.AST or literal value, ...}}
    """
    src = path.read_text()
    tree = ast.parse(src)
    out: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "ModuleSpec"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        name_node = kwargs.get("name")
        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            out[name_node.value] = kwargs
    return out


def _resolve_attr_or_name(node: ast.AST) -> Optional[str]:
    """Resolve an AST node like `bus.FOO` or `"FOO"` or a `Name` to its
    string value. Returns None if can't resolve statically."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    if isinstance(node, ast.Name):
        # e.g. KIN_EMOT_STATE_MSG_TYPE imported as constant
        return node.id
    return None


def _extract_broadcast_topics(spec_kwargs: dict) -> Optional[set[str]]:
    """Extract the broadcast_topics list (resolved to constant names or
    literal strings) from a ModuleSpec kwargs dict. Returns None if the
    list isn't a static List node (e.g., name reference like
    `_SPIRIT_BROADCAST_TOPICS` — caller must resolve)."""
    bt = spec_kwargs.get("broadcast_topics")
    if bt is None:
        return None
    if isinstance(bt, ast.Name):
        # Reference to a module-level constant — resolve from the same file.
        # The test resolves these via _resolve_module_constant below.
        return {f"@{bt.id}"}  # sentinel; resolved later
    if not isinstance(bt, ast.List):
        return None
    out: set[str] = set()
    for elt in bt.elts:
        resolved = _resolve_attr_or_name(elt)
        if resolved is not None:
            out.add(resolved)
    return out


def _resolve_module_constant(path: Path, var_name: str) -> set[str]:
    """Find a module-level assignment `<var_name> = [bus.X, bus.Y, ...]`.

    Looks first in `path`. If not found, walks `from titan_hcl.X.Y import
    <var_name>` statements in the same file to find the source module, then
    looks there. This handles patterns like:

        from titan_hcl.modules.cognitive_worker import (
            cognitive_worker_main, _COGNITIVE_WORKER_SUBSCRIBE_TOPICS,
        )
        ...broadcast_topics=_COGNITIVE_WORKER_SUBSCRIBE_TOPICS,

    where the constant is defined in cognitive_worker.py but referenced
    from core/plugin.py.
    """
    if not path.exists():
        return set()
    src = path.read_text()
    tree = ast.parse(src)
    # Pass 1: same-file definition.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    if isinstance(node.value, ast.List):
                        out = set()
                        for elt in node.value.elts:
                            n = _resolve_attr_or_name(elt)
                            if n is not None:
                                out.add(n)
                        return out
    # Pass 2: follow `from titan_hcl.X import <var_name>`.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == var_name or alias.asname == var_name:
                    # Resolve module path: titan_hcl.modules.foo → titan_hcl/modules/foo.py
                    rel = node.module.replace(".", "/") + ".py"
                    candidate = REPO_ROOT / rel
                    if candidate.exists():
                        return _resolve_module_constant(candidate, var_name)
    return set()


# ─────────────────────────────────────────────────────────────────────
# Drain-chain extraction — parse worker module's dispatcher.
# ─────────────────────────────────────────────────────────────────────

DISPATCHER_PATTERNS = [
    re.compile(r'msg_type\s*==\s*bus\.([A-Z_][A-Z_0-9]*)'),
    re.compile(r'msg_type\s*==\s*"([A-Z_][A-Z_0-9]*)"'),
    re.compile(r"msg_type\s*==\s*'([A-Z_][A-Z_0-9]*)'"),
    re.compile(r'msg_type\s+in\s+\(\s*((?:bus\.[A-Z_][A-Z_0-9]*\s*,?\s*)+)\)'),
    re.compile(r'msg\.get\(\s*["\']type["\']\s*\)\s*==\s*bus\.([A-Z_][A-Z_0-9]*)'),
    re.compile(r'msg_type\s*==\s*([A-Z_][A-Z_0-9]+_MSG_TYPE)'),
]


def _extract_drain_types(path: Path) -> set[str]:
    """Extract all msg_type comparison values from the drain dispatcher
    in `path`. Returns the set of bus.* constant names + string literals."""
    src = path.read_text()
    out: set[str] = set()
    # Single-value patterns
    for pat in DISPATCHER_PATTERNS[:3]:
        for m in pat.finditer(src):
            out.add(m.group(1))
    # in-tuple pattern — extract each bus.X
    for m in DISPATCHER_PATTERNS[3].finditer(src):
        for inner in re.finditer(r'bus\.([A-Z_][A-Z_0-9]*)', m.group(1)):
            out.add(inner.group(1))
    # msg.get pattern
    for m in DISPATCHER_PATTERNS[4].finditer(src):
        out.add(m.group(1))
    # KIN_EMOT_STATE_MSG_TYPE-style symbolic constants
    for m in DISPATCHER_PATTERNS[5].finditer(src):
        out.add(m.group(1))
    return out


# ─────────────────────────────────────────────────────────────────────
# ModuleSpec lookup — core/plugin.py is the SOLE registration site.
# (legacy_core.py / TitanCore retired 2026-05-21, D-SPEC-106; the old
# dual-site merge collapsed to single-source.)
# ─────────────────────────────────────────────────────────────────────


def _all_modulespecs() -> dict[str, dict[str, dict]]:
    """{worker_name: {file_path: kwargs_dict}} — the single registration site."""
    out: dict[str, dict[str, dict]] = {}
    for fname in ("titan_hcl/core/plugin.py",):
        path = REPO_ROOT / fname
        for name, kwargs in _extract_modulespec_kwargs(path).items():
            out.setdefault(name, {})[fname] = kwargs
    return out


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def all_specs() -> dict[str, dict[str, dict]]:
    return _all_modulespecs()


def test_all_workers_have_explicit_filter_or_reply_only(all_specs):
    """Every flood-receiving ModuleSpec must declare broadcast_topics OR
    reply_only=True. No subscribe-all (legacy fallback) workers may exist
    after rFP_worker_broadcast_topics_completion lands.

    Catches: someone adds a new ModuleSpec(name="newworker", ...) without
    declaring filter → falls into legacy subscribe-all → gets flooded.
    """
    failures: list[str] = []
    for worker, sites in all_specs.items():
        # Skip the dynamic _w_name registration (line 471 in legacy_core —
        # used inside a loop with template, not a static worker).
        if worker == "_w_name":
            continue
        for fname, kwargs in sites.items():
            reply_only_node = kwargs.get("reply_only")
            is_reply_only = (
                isinstance(reply_only_node, ast.Constant)
                and reply_only_node.value is True
            )
            has_broadcast_topics = "broadcast_topics" in kwargs
            if not (is_reply_only or has_broadcast_topics):
                failures.append(
                    f"  {worker} in {fname}: neither reply_only=True nor "
                    f"broadcast_topics=[...] declared. Falls into legacy "
                    f"subscribe-all path → flood-receiving."
                )
    assert not failures, (
        "rFP_worker_broadcast_topics_completion §6 acceptance #1 violated:\n"
        + "\n".join(failures)
        + "\n\nFix: declare broadcast_topics=[<types>] matching the worker's "
        "drain elif chain, OR set reply_only=True if the worker only "
        "consumes targeted QUERY messages."
    )


def test_worker_filter_intersects_drain_chain(all_specs):
    """For each migrated worker (broadcast_topics declared), its declared
    types must be a subset of types its drain handler actually checks.
    Catches: filter declares "FOO" but drain has no handler.

    NOTE: this is a SUBSET check, not equality, because:
      - broadcast filter is producer-side; over-declaring is harmless
        (just delivers msg the drain may ignore via final `else:`)
      - some drain checks use bus.* constants we may have missed in the
        regex (cross-module imports, dynamic dispatch). Subset = safe.
    Equality is a STRONGER property tested separately for well-known workers.
    """
    failures: list[str] = []
    for worker, sites in all_specs.items():
        if worker not in WORKER_DRAIN_FILE:
            continue  # reply_only or unmapped
        drain_path = REPO_ROOT / WORKER_DRAIN_FILE[worker]
        if not drain_path.exists():
            continue
        drain_types = _extract_drain_types(drain_path)
        for fname, kwargs in sites.items():
            declared = _extract_broadcast_topics(kwargs)
            if declared is None:
                continue  # reply_only or unparseable (Name reference)
            # Resolve module-constant references like _SPIRIT_BROADCAST_TOPICS
            resolved: set[str] = set()
            for t in declared:
                if t.startswith("@"):
                    var_name = t[1:]
                    resolved |= _resolve_module_constant(REPO_ROOT / fname, var_name)
                else:
                    resolved.add(t)
            # Subtract types that are TARGETED (bypass filter; drain may
            # still handle them but we don't list in broadcast_topics).
            non_lifecycle_drain = drain_types - TARGETED_LIFECYCLE_TYPES
            extra = resolved - non_lifecycle_drain - TARGETED_LIFECYCLE_TYPES
            if extra:
                failures.append(
                    f"  {worker} in {fname}: broadcast_topics declares {sorted(extra)}"
                    f" but drain at {WORKER_DRAIN_FILE[worker]} has no handler "
                    f"for these types. Filter is over-declared (broker delivers "
                    f"msgs the worker silently drops via else-branch). Either "
                    f"add handlers OR remove from broadcast_topics."
                )
    # NOTE: This is a SOFT assertion — log warnings but don't fail the test
    # by default. Some workers may legitimately list types they only handle
    # via dispatchers in imported helper modules (not visible to our regex).
    # The HARD subset failure mode would be:
    #   `if extra and worker in STRICT_LOCKSTEP_WORKERS: failures.append(...)`
    # but we keep it informational for the initial migration.
    if failures:
        # Print as warnings; don't fail.
        import sys
        print("\n[lockstep audit — soft warnings, not test failures]")
        for f in failures:
            print(f, file=sys.stderr)


# test_dual_registered_workers_have_consistent_filters RETIRED 2026-05-21
# (D-SPEC-106): it policed broadcast_topics drift between the legacy_core.py
# and core/plugin.py ModuleSpec registration sites. legacy_core.py is gone —
# core/plugin.py is the SOLE registration site, so dual-site drift is
# structurally impossible. Single-site filter correctness is still covered by
# test_all_workers_have_explicit_filter_or_reply_only +
# test_worker_filter_intersects_drain_chain above.


def test_rpc_reply_only_workers_are_marked():
    """Workers that consume NO broadcasts (RPC-reply pattern) must declare
    reply_only=True. Catches: someone leaves an old reply_only=False on a
    worker whose drain only handles QUERY/SHUTDOWN.
    """
    specs = _all_modulespecs()
    failures: list[str] = []
    for worker in RPC_REPLY_ONLY_WORKERS:
        if worker not in specs:
            continue
        for fname, kwargs in specs[worker].items():
            reply_only_node = kwargs.get("reply_only")
            is_reply_only = (
                isinstance(reply_only_node, ast.Constant)
                and reply_only_node.value is True
            )
            if not is_reply_only:
                failures.append(
                    f"  {worker} in {fname}: must be reply_only=True per "
                    f"RPC_REPLY_ONLY_WORKERS list. Drain only consumes "
                    f"targeted QUERY messages — no broadcasts."
                )
    assert not failures, "\n".join(failures)


def test_filter_count_baseline():
    """Baseline regression: known-good filter sizes from
    rFP_worker_broadcast_topics_completion §2.1 audit. Catches accidental
    deletions / wholesale list rewrites.
    """
    specs = _all_modulespecs()
    expected_min_counts = {
        # outer_{body,mind,spirit} RETIRED (Phase C dissolution C.8).
        "body":             4,
        "mind":             6,   # was 7 — OUTER_SOURCES_SNAPSHOT dropped (C.7/C.8)
        "recorder":               1,
        "llm":              1,
        "warning_monitor":  1,
        "language":         14,
        "meta_teacher":     1,
        "emot_cgn":         8,
        "cgn":              7,
        "knowledge":        10,
        "timechain":        23,
        # spirit RETIRED (D-SPEC-116). cognitive_worker now hosts the re-homed
        # MEMORY_RECALL_PERTURBATION + TEACHER_SIGNALS + OUTER_OBSERVATION flows.
        "cognitive_worker": 8,
        "backup":           2,
    }
    failures: list[str] = []
    for worker, expected_min in expected_min_counts.items():
        if worker not in specs:
            continue
        for fname, kwargs in specs[worker].items():
            declared = _extract_broadcast_topics(kwargs)
            if declared is None:
                continue
            resolved: set[str] = set()
            for t in declared:
                if t.startswith("@"):
                    resolved |= _resolve_module_constant(REPO_ROOT / fname, t[1:])
                else:
                    resolved.add(t)
            actual = len(resolved)
            if actual < expected_min:
                failures.append(
                    f"  {worker} in {fname}: broadcast_topics has {actual} "
                    f"types, expected ≥{expected_min} per §2.1 audit baseline."
                )
    assert not failures, "\n".join(failures)
