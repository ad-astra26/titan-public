#!/usr/bin/env python
"""
scripts/s5_callsite_audit.py — categorize all `plugin.X.Y` callsites in
titan_plugin/api/ for the S5 amendment codemod.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25), Phase 2.

Walks every .py under titan_plugin/api/ and finds attribute access
chains rooted at `plugin` (any of the names: plugin, _plugin, agent.plugin,
request.app.state.titan_plugin, etc.). Each chain is classified:

  A — pure state read (e.g. plugin.network.pubkey, plugin.soul.maker_pubkey)
      ⇒ codemod target: state.<sub>.<attr>

  B — async cross-process call (e.g. await plugin.network.get_balance())
      ⇒ codemod target: state.<sub>.<attr>  (now a cached property)

  C — side-effect / complex / unmappable
      (e.g. plugin.guardian.start("module"), plugin._proxies.get(name).method(),
      conditional checks, dynamic getattr, etc.)
      ⇒ manual handling: state.commands.<verb>(...) or kept-RPC

Outputs CSV at titan-docs/s5_callsite_inventory.csv with columns:
  file, line, col, current_pattern, classification, suggested_target

Run:
  python scripts/s5_callsite_audit.py
  python scripts/s5_callsite_audit.py --paths titan_plugin/api/dashboard.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Roots that resolve to the plugin object in endpoint code.
PLUGIN_ROOT_NAMES = {"plugin", "_plugin", "request_plugin"}
# Attribute paths that mean "current plugin" via attribute chain
PLUGIN_ATTR_PREFIXES = [
    ("request", "app", "state", "titan_plugin"),
    ("app", "state", "titan_plugin"),
    ("agent", "plugin"),
]

# Known sub-accessor names — used to suggest targets. Most plugin
# attributes are state reads; only side-effect methods + special-case
# attributes (bus, _proxies dynamic) are Category C.
KNOWN_SUB_ACCESSORS = {
    # Direct sub-accessors on TitanStateAccessor
    "network", "trinity", "neuromods", "epoch", "spirit", "body", "mind",
    "identity", "soul", "cgn", "reasoning", "dreaming", "guardian",
    "agency", "language", "meta_teacher", "social", "config",
    # Plugin attributes that map cleanly to cache or sub-accessors
    "memory", "mood_engine", "metabolism", "backup", "recorder",
    "studio", "gatekeeper", "event_bus", "sovereignty", "params",
    "config_loader", "persistence", "maker", "interface_advisor",
    "agency_assessment", "_agency", "_agency_assessment",
    "_interface_advisor",
    # Static-config + private-state attributes (cache-resident)
    "_full_config", "_limbo_mode", "_dream_inbox", "_current_user_id",
    "_pending_self_composed", "_pending_self_composed_confidence",
    "_last_execution_mode", "_last_commit_signature",
    "_last_research_sources", "_start_time", "_is_meditating",
    "_gather_current_state", "_get_state_narrator",
    # Other top-level plugin attributes encountered
    "get_v3_status", "get",
    # Special handling
    "_proxies",      # → state.<proxy_name> (or C if dynamic)
    "bus",           # → state.commands.publish (raw, side-effect)
}

# Methods that are inherently side-effect — Category C
SIDE_EFFECT_METHODS = {
    "start", "stop", "restart", "kill", "publish", "request",
    "reload_api", "evolve_soul", "force_dream", "inject_memory",
    "register", "subscribe", "unsubscribe", "emit",
    "set_titan_maker", "set_somatic_channel", "set_narrative_channel",
}

# Methods that are async by convention (await fires kernel-side coroutine).
# After amendment: become cached property reads (no await needed).
ASYNC_BUS_CACHED_METHODS = {
    "get_balance", "get_raw_account_data", "get_topology",
    "get_top_memories", "get_neuromod_state", "get_ns_state",
    "get_reasoning_state", "get_memory_status", "get_persistent_count",
    "fetch_mempool", "fetch_social_metrics", "get_knowledge_graph",
    "get_coordinator", "get_v4_state", "get_sphere_clocks",
    "get_trinity", "get_body_tensor", "get_mind_tensor",
}


@dataclass
class CallsiteRecord:
    file: str
    line: int
    col: int
    current_pattern: str
    classification: str  # A / B / C
    suggested_target: str
    reason: str  # why this classification

    def as_csv_row(self) -> list[str]:
        return [
            self.file, str(self.line), str(self.col),
            self.current_pattern, self.classification,
            self.suggested_target, self.reason,
        ]


# ── Path-extraction helpers ───────────────────────────────────────────


def _flatten_attr_chain(node: ast.AST) -> list[str] | None:
    """Walk an Attribute/Name/Subscript/Call chain and return the dotted
    path as a list of strings. Returns None if the chain root is unknown.
    """
    parts: list[str] = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Subscript):
            # plugin._proxies["spirit"] or similar
            slc = cur.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                parts.append(f"[{slc.value!r}]")
            else:
                parts.append("[<dynamic>]")
            cur = cur.value
        elif isinstance(cur, ast.Call):
            # plugin._proxies.get("spirit") — record args but treat as call
            arg_strs = []
            for a in cur.args:
                if isinstance(a, ast.Constant):
                    arg_strs.append(repr(a.value))
                else:
                    arg_strs.append("<expr>")
            parts.append(f"({','.join(arg_strs)})")
            cur = cur.func
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            break
        else:
            return None
    parts.reverse()
    return parts


def _is_plugin_root(parts: list[str]) -> tuple[bool, list[str]]:
    """Return (is_plugin_chain, normalized_chain) — strips the prefix
    that resolves to plugin (request.app.state.titan_plugin etc.) and
    returns the remainder starting from the first plugin attribute.
    """
    if not parts:
        return False, []
    if parts[0] in PLUGIN_ROOT_NAMES:
        return True, parts[1:]
    # Try prefix match
    for prefix in PLUGIN_ATTR_PREFIXES:
        if len(parts) >= len(prefix) and tuple(parts[:len(prefix)]) == prefix:
            return True, parts[len(prefix):]
    return False, []


# ── Classifier ────────────────────────────────────────────────────────


def classify(parts: list[str], is_awaited: bool, is_called: bool) -> tuple[str, str, str]:
    """Return (classification, suggested_target, reason).

    parts is the chain AFTER the plugin root, e.g. ["network", "pubkey"]
    or ["_proxies", "(spirit)", "get_trinity", "()"].
    """
    if not parts:
        return ("C", "?", "empty chain")

    head = parts[0]

    # Special: plugin._proxies.get("X")[.method()...] or plugin._proxies["X"]...
    # AST chain forms after _proxies:
    #   ["get", "('spirit')", "method", "()"]    ← .get("X").method()
    #   ["get", "('spirit')"]                    ← .get("X")
    #   ["[ 'spirit' ]", "method", "()"]         ← ["X"].method()
    if head == "_proxies":
        # Find the proxy-name argument (in parts[1] if it's a string literal call,
        # or parts[2] if it's after .get("X"))
        proxy_name = None
        rest_idx = 0
        if len(parts) >= 2 and parts[1].startswith("[") and parts[1].endswith("]"):
            # Subscript form ["spirit"]
            inner = parts[1][1:-1].strip("'\"")
            if inner and not inner.startswith("<"):
                proxy_name = inner
                rest_idx = 2
        elif len(parts) >= 3 and parts[1] == "get" and parts[2].startswith("("):
            # .get("spirit") form
            inner = parts[2].strip("()").strip("'\"")
            if inner and not inner.startswith("<"):
                proxy_name = inner
                rest_idx = 3
        if proxy_name is None:
            return ("C", "?", "_proxies dynamic access")
        # Skip past trailing ()
        while rest_idx < len(parts) and parts[rest_idx].startswith("("):
            rest_idx += 1
        if proxy_name not in KNOWN_SUB_ACCESSORS:
            return ("C", f"state.{proxy_name}", f"unknown _proxies key: {proxy_name}")
        if rest_idx >= len(parts):
            return ("A", f"state.{proxy_name}",
                    f"_proxies attribute access: {proxy_name}")
        method = parts[rest_idx]
        if method in SIDE_EFFECT_METHODS:
            return ("C", f"state.commands.{method}(...)",
                    f"_proxies side-effect method {method}")
        if method in ASYNC_BUS_CACHED_METHODS or is_awaited:
            return ("B", f"state.{proxy_name}.{method.replace('get_', '')}",
                    f"_proxies async-cached method {method}")
        return ("A", f"state.{proxy_name}.{method.replace('get_', '')}",
                f"_proxies state read method {method}")

    # plugin.bus.publish/request/subscribe → commands
    if head == "bus":
        if len(parts) >= 2:
            verb = parts[1]
            if verb in {"publish", "request", "subscribe", "unsubscribe"}:
                return ("C", f"state.commands.publish(...)" if verb == "publish" else "?",
                        f"bus.{verb} side-effect")
        return ("C", "?", "bus access")

    # plugin._full_config → state.config
    if head == "_full_config":
        rest = ".".join(parts[1:]) if len(parts) > 1 else ""
        return ("A", f"state.config.{rest}" if rest else "state.config",
                "_full_config → config")

    # plugin.<sub>.<...>
    if head in KNOWN_SUB_ACCESSORS:
        if len(parts) == 1:
            return ("A", f"state.{head}", f"sub-accessor root: {head}")
        method_or_attr = parts[1]
        if method_or_attr in SIDE_EFFECT_METHODS:
            return ("C", f"state.commands.{method_or_attr}(...)",
                    f"side-effect method {method_or_attr}")
        if method_or_attr in ASYNC_BUS_CACHED_METHODS or is_awaited:
            return ("B", f"state.{head}.{method_or_attr.replace('get_', '')}",
                    f"async/cached: {method_or_attr}")
        # Plain attribute or simple getter
        return ("A", f"state.{head}.{method_or_attr.replace('get_', '')}",
                f"state read: {method_or_attr}")

    # Default: treat as state read on a sub-accessor we'll add.
    # If the attribute looks like a side-effect method by name, mark C.
    if head in SIDE_EFFECT_METHODS:
        return ("C", f"state.commands.{head}(...)",
                f"top-level side-effect method {head}")
    return ("A", f"state.{head}", f"state read (auto-mapped): {head}")


# ── AST visitor ────────────────────────────────────────────────────────


class CallsiteCollector(ast.NodeVisitor):
    """Walks a module, finds plugin.* chains, classifies each."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.records: list[CallsiteRecord] = []
        self._await_depth = 0

    def visit_Await(self, node: ast.Await) -> None:
        self._await_depth += 1
        self.generic_visit(node)
        self._await_depth -= 1

    def _check_node(self, node: ast.AST, is_called: bool) -> None:
        parts = _flatten_attr_chain(node)
        if parts is None:
            return
        is_plugin, chain = _is_plugin_root(parts)
        if not is_plugin or not chain:
            return
        cls, target, reason = classify(
            chain, is_awaited=self._await_depth > 0, is_called=is_called)
        # Build a readable current_pattern
        current = "plugin." + ".".join(chain).replace(".(", "(")
        self.records.append(CallsiteRecord(
            file=self.file_path,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            current_pattern=current,
            classification=cls,
            suggested_target=target,
            reason=reason,
        ))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Only check at the "outermost" Attribute — nested ones are subpaths
        # We handle this by checking: if our parent is also Attribute/Call/Subscript,
        # it'll catch us. So we only emit a record if we're a "leaf" or terminal.
        # Approach: visit all, but we dedupe by (file, line, col, current_pattern) at
        # write time. Simpler than tracking parent.
        self._check_node(node, is_called=False)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Pass the whole Call node so _flatten_attr_chain captures the args
        # (needed to identify _proxies.get("spirit") string literal).
        self._check_node(node, is_called=True)
        self.generic_visit(node)


# ── Driver ─────────────────────────────────────────────────────────────


def scan_file(path: Path) -> list[CallsiteRecord]:
    try:
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
    except Exception as e:
        print(f"  [skip] {path}: {e}", file=sys.stderr)
        return []
    rel = str(path.relative_to(PROJECT_ROOT))
    coll = CallsiteCollector(rel)
    coll.visit(tree)
    return coll.records


def scan_paths(paths: list[Path]) -> list[CallsiteRecord]:
    all_records: list[CallsiteRecord] = []
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.is_file() and p.suffix == ".py":
            files.append(p)
    for f in files:
        all_records.extend(scan_file(f))
    return all_records


def dedupe(records: list[CallsiteRecord]) -> list[CallsiteRecord]:
    """Records may include both the leaf attribute (plugin.X.Y) AND the
    parent chain (plugin.X). We keep only the longest chain per
    (file, line, col)."""
    by_loc: dict[tuple, CallsiteRecord] = {}
    for r in records:
        key = (r.file, r.line, r.col)
        if key not in by_loc:
            by_loc[key] = r
        else:
            existing = by_loc[key]
            if len(r.current_pattern) > len(existing.current_pattern):
                by_loc[key] = r
    return list(by_loc.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", nargs="*",
        default=["titan_plugin/api"],
        help="files or dirs to scan (default: titan_plugin/api)",
    )
    parser.add_argument(
        "--out", default="titan-docs/s5_callsite_inventory.csv",
        help="output CSV path",
    )
    parser.add_argument("--summary", action="store_true",
                        help="print classification summary instead of CSV path")
    args = parser.parse_args()

    paths = [PROJECT_ROOT / p for p in args.paths]
    records = scan_paths(paths)
    records = dedupe(records)
    records.sort(key=lambda r: (r.file, r.line, r.col))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "col", "current_pattern",
                    "classification", "suggested_target", "reason"])
        for r in records:
            w.writerow(r.as_csv_row())

    # Summary
    by_cls = {"A": 0, "B": 0, "C": 0}
    for r in records:
        by_cls[r.classification] = by_cls.get(r.classification, 0) + 1

    by_file: dict[str, dict[str, int]] = {}
    for r in records:
        by_file.setdefault(r.file, {"A": 0, "B": 0, "C": 0})
        by_file[r.file][r.classification] += 1

    print(f"\n[s5-callsite-audit] {len(records)} callsites in {len(by_file)} files")
    print(f"  A (state read):       {by_cls['A']}")
    print(f"  B (async cross-proc): {by_cls['B']}")
    print(f"  C (manual handle):    {by_cls['C']}")
    print(f"\n  CSV → {out_path}")

    if args.summary:
        print("\nPer-file:")
        for f, cls in sorted(by_file.items(), key=lambda x: -sum(x[1].values())):
            total = sum(cls.values())
            print(f"  {total:4d}  {f}  (A={cls['A']} B={cls['B']} C={cls['C']})")


if __name__ == "__main__":
    main()
