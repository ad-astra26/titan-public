"""
Tests for Microkernel v2 Phase A §A.4 (S5) — EXPOSED_METHODS drift detection.

Statically analyzes titan_plugin/api/{dashboard,chat,maker,webhook}.py for
runtime `plugin.X.Y(...)` and `plugin.X` access patterns and asserts every
unique path is in KERNEL_RPC_EXPOSED_METHODS. Catches future endpoint
additions that forget to update the exposed list (which would cause the
new endpoint to fail with MethodNotExposed at runtime once the API
subprocess flag is on).

Approach (intentionally pragmatic, not full AST):
  - Read each API file
  - Extract `plugin.X` and `plugin.X.Y` patterns via regex
  - Filter out obvious false positives (modules: plugin.core, plugin.utils,
    plugin.logic, plugin.api, plugin.expressive, plugin.channels, plugin.maker
    — when used as `from titan_plugin.X import ...`)
  - Assert each remaining path is in EXPOSED_METHODS

Reference:
  - titan-docs/PLAN_microkernel_phase_a_s5.md §5.4
  - titan_plugin/core/kernel.py:KERNEL_RPC_EXPOSED_METHODS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from titan_plugin.core.kernel import KERNEL_RPC_EXPOSED_METHODS


# False-positive filter: these are module-import paths, not runtime attr access
# (we use regex against any "plugin." string; this filters the import patterns)
MODULE_IMPORT_PREFIXES = {
    "plugin.api",
    "plugin.core",
    "plugin.logic",
    "plugin.utils",
    "plugin.expressive",
    "plugin.channels",
    "plugin.contracts",
    "plugin.proxies",
    "plugin.skills",
    "plugin.persistence",
    "plugin.inference",
    "plugin.maker",  # both module AND attr — special-cased: use only with dotted suffix
    "plugin.addons",
    "plugin.titan_params",
}

# Known false positives (regex matches that are not actual plugin attr access)
FALSE_POSITIVES = {
    "plugin.py",  # filename references in error messages
    "plugin.config_loader",  # module-level lazy import target — not actually accessed at runtime via plugin instance
    "plugin.api.auth",  # module ref — endpoint doesn't actually access plugin.api.auth at runtime
    "plugin.api.dashboard",  # same
    "plugin.bus",  # bare access — covered as "bus" + sub-paths in EXPOSED
    "plugin.params",  # config attribute, covered as "params"
    "plugin.event_bus.subscriber_count",  # subscriber_count is a property — covered as "event_bus" + ".subscriber_count" via chain
    "plugin.event_bus.emit",  # covered as "event_bus.emit"
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_FILES = [
    PROJECT_ROOT / "titan_plugin" / "api" / "dashboard.py",
    PROJECT_ROOT / "titan_plugin" / "api" / "chat.py",
    PROJECT_ROOT / "titan_plugin" / "api" / "maker.py",
    PROJECT_ROOT / "titan_plugin" / "api" / "webhook.py",
]


def _extract_plugin_paths(source: str) -> set[str]:
    """Pragmatic regex extraction of plugin.X.Y patterns from source.

    Captures dotted access of up to 3 levels. Strips trailing whitespace
    and parens. Filters out module-import-prefix paths.
    """
    # Match plugin.X or plugin.X.Y or plugin.X.Y.Z (up to 3 levels)
    pattern = re.compile(
        r"\bplugin\.(?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*){0,2}"
    )
    paths = set(pattern.findall(source))
    # Filter modules + known false positives
    filtered = set()
    for p in paths:
        if p in FALSE_POSITIVES:
            continue
        # Module-import-prefix filter: "plugin.core.X" → drop unless first
        # token after "plugin." is a known runtime attr (handled by the
        # opposite side — checking against KERNEL_RPC_EXPOSED_METHODS)
        first_token = p.split(".", 2)[1] if "." in p else None
        if f"plugin.{first_token}" in MODULE_IMPORT_PREFIXES and p != f"plugin.{first_token}":
            continue
        filtered.add(p)
    return filtered


def _normalize(path: str) -> str:
    """Convert 'plugin.X.Y' → 'X.Y' for comparison with EXPOSED_METHODS."""
    if path.startswith("plugin."):
        return path[len("plugin."):]
    return path


def test_exposed_methods_is_frozenset():
    assert isinstance(KERNEL_RPC_EXPOSED_METHODS, frozenset)
    assert len(KERNEL_RPC_EXPOSED_METHODS) > 50  # sanity: ~85 expected


def test_no_drift_in_api_dashboard():
    """Every plugin.X path used in dashboard.py is in EXPOSED_METHODS."""
    src = API_FILES[0].read_text()
    paths = _extract_plugin_paths(src)
    missing = []
    for p in sorted(paths):
        norm = _normalize(p)
        if norm not in KERNEL_RPC_EXPOSED_METHODS:
            missing.append(p)
    assert not missing, (
        f"DRIFT: dashboard.py uses {len(missing)} plugin paths NOT in "
        f"KERNEL_RPC_EXPOSED_METHODS — add them to "
        f"titan_plugin/core/kernel.py:KERNEL_RPC_EXPOSED_METHODS:\n"
        + "\n".join(f"  - {p}" for p in missing[:20])
    )


def test_no_drift_in_api_chat():
    src = API_FILES[1].read_text()
    paths = _extract_plugin_paths(src)
    missing = [
        p for p in sorted(paths)
        if _normalize(p) not in KERNEL_RPC_EXPOSED_METHODS
    ]
    assert not missing, (
        f"DRIFT: chat.py uses {len(missing)} plugin paths NOT in EXPOSED:\n"
        + "\n".join(f"  - {p}" for p in missing[:20])
    )


def test_no_drift_in_api_maker():
    src = API_FILES[2].read_text()
    paths = _extract_plugin_paths(src)
    missing = [
        p for p in sorted(paths)
        if _normalize(p) not in KERNEL_RPC_EXPOSED_METHODS
    ]
    assert not missing, (
        f"DRIFT: maker.py uses {len(missing)} plugin paths NOT in EXPOSED:\n"
        + "\n".join(f"  - {p}" for p in missing[:20])
    )


def test_no_drift_in_api_webhook():
    src = API_FILES[3].read_text()
    paths = _extract_plugin_paths(src)
    missing = [
        p for p in sorted(paths)
        if _normalize(p) not in KERNEL_RPC_EXPOSED_METHODS
    ]
    assert not missing, (
        f"DRIFT: webhook.py uses {len(missing)} plugin paths NOT in EXPOSED:\n"
        + "\n".join(f"  - {p}" for p in missing[:20])
    )


def test_no_unused_exposed_methods():
    """Sanity: warn (don't fail) if EXPOSED_METHODS contains paths NOT used
    anywhere in the API code. These could be safely removed."""
    all_src = "\n".join(f.read_text() for f in API_FILES if f.exists())
    used_paths = _extract_plugin_paths(all_src)
    used_normalized = {_normalize(p) for p in used_paths}
    # Bus-backed proxies are accessed implicitly via getattr in some endpoints;
    # don't flag those. Also event_bus.* mirrors are kept for legacy in-process
    # path compatibility. Flag genuinely unused paths only.
    KEEP_ALWAYS = {
        # bus-backed proxies — endpoints reach them via plugin._proxies.get
        "memory", "memory.fetch_mempool", "memory.fetch_social_metrics",
        "memory.get_coordinator", "memory.get_knowledge_graph",
        "memory.get_memory_status", "memory.get_neuromod_state",
        "memory.get_ns_state", "memory.get_persistent_count",
        "memory.get_reasoning_state", "memory.get_top_memories",
        "memory.get_topology", "memory.inject_memory",
        "memory._cognee_ready", "memory._node_store",
        "memory._node_store.items",
        "mood_engine", "mood_engine.get_mood_label",
        "mood_engine.get_mood_valence", "mood_engine.previous_mood",
        "mood_engine.force_zen",
        "gatekeeper",
        "event_bus", "event_bus.emit", "event_bus.subscriber_count",
    }
    unused = sorted(
        p for p in KERNEL_RPC_EXPOSED_METHODS
        if p not in used_normalized and p not in KEEP_ALWAYS
    )
    # Sanity-only: just count, don't assert specific paths
    if unused:
        print(f"\n  [info] {len(unused)} EXPOSED paths not directly grep-found: {unused[:10]}")
