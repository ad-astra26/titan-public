"""Tests for Phase B (rFP §3.4.1) §B6 G19 timeout audit.

Every `bus.request_async` (or `_work_rpc_sync`) call site in
`titan_hcl/proxies/memory_proxy.py` must have a timeout argument
≤5s OR be explicitly allowlisted in
`titan-docs/phase_c_rpc_exemptions.yaml` (run_meditation 300s).

This is a static-analysis test: parse the proxy source, find every
work-RPC call site + its timeout literal, and assert the ceiling.

Failure here means a future edit reintroduced a >5s work-RPC without
updating the exemption file — which is a SPEC violation per Preamble G19.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


PROXY_PATH = (
    Path(__file__).resolve().parent.parent
    / "titan_hcl/proxies/memory_proxy.py"
)
EXEMPTIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "titan-docs/phase_c_rpc_exemptions.yaml"
)

# Methods that are explicitly allowlisted with timeouts >5s. Each tuple is
# (method_name_substring, max_allowed_timeout). If a method needs to exceed
# 5s, add it here AND to phase_c_rpc_exemptions.yaml with a documented
# rationale. Default cap is 5.0s (Preamble G19).
ALLOWLISTED_METHODS = {
    "run_meditation": 300.0,        # Long-running cycle (LLM scoring + Solana TX)
    "run_meditation_async": 300.0,  # Same
}
DEFAULT_G19_CAP_S = 5.0


def _extract_request_async_timeouts(source: str) -> list[tuple[int, str, float]]:
    """Walk the AST + return list of (lineno, enclosing_func_name, timeout_value).

    Targets calls of the form:
      await self._bus.request_async("memory_proxy", "memory", payload, TIMEOUT, ...)
      self._work_rpc_sync(payload, TIMEOUT)

    The timeout is the 4th positional arg of request_async, or the 2nd
    positional of _work_rpc_sync. Returns the timeout's float value when
    it's a literal `Constant(int|float)`. Skips dynamic-value timeouts
    with a logged warning (those would need manual review).
    """
    tree = ast.parse(source)
    out: list[tuple[int, str, float]] = []

    # Build a map: lineno → enclosing function name (walk top-down).
    func_at: dict[int, str] = {}

    def _visit_func(node, name):
        end = getattr(node, "end_lineno", node.lineno)
        for ln in range(node.lineno, end + 1):
            if ln not in func_at:
                func_at[ln] = name
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef)):
                _visit_func(child, child.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            _visit_func(node, node.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # request_async(...): self._bus.request_async(src, dst, payload, TIMEOUT, reply_q)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "request_async"
                and len(node.args) >= 4
                and isinstance(node.args[3], ast.Constant)
                and isinstance(node.args[3].value, (int, float))):
            out.append((node.lineno, func_at.get(node.lineno, "?"),
                        float(node.args[3].value)))
        # _work_rpc_sync(payload, TIMEOUT)
        elif (isinstance(node.func, ast.Attribute)
                and node.func.attr == "_work_rpc_sync"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, (int, float))):
            out.append((node.lineno, func_at.get(node.lineno, "?"),
                        float(node.args[1].value)))
    return out


def test_phase_c_rpc_exemptions_yaml_exists():
    assert EXEMPTIONS_PATH.exists(), (
        f"phase_c_rpc_exemptions.yaml missing at {EXEMPTIONS_PATH} — "
        "the G19 allowlist must exist for this test to mean anything")


def test_every_memory_proxy_timeout_under_g19_cap_or_allowlisted():
    """Static audit: every work-RPC call site in memory_proxy.py is
    ≤5s OR allowlisted (run_meditation 300s)."""
    source = PROXY_PATH.read_text()
    timeouts = _extract_request_async_timeouts(source)
    # We expect to find at least the 8 method call sites (after Phase B B6
    # tightening: query, fetch_mempool, get_top_memories,
    # get_top_memories_for_observatory, fetch_mempool_for_observatory,
    # get_topology, get_knowledge_graph, run_meditation_async + the legacy
    # sync run_meditation has 2 _work_rpc_sync sites in the if/else
    # branches). Don't hard-code count — just enforce the ceiling.
    assert timeouts, "no work-RPC sites found — parser bug?"
    violations: list[str] = []
    for lineno, fname, timeout in timeouts:
        cap = DEFAULT_G19_CAP_S
        for method, allow_cap in ALLOWLISTED_METHODS.items():
            if method in fname:
                cap = allow_cap
                break
        if timeout > cap:
            violations.append(
                f"  memory_proxy.py:{lineno} (in {fname}): "
                f"timeout={timeout}s > cap={cap}s"
            )
    if violations:
        pytest.fail(
            "Phase B G19 violation — work-RPC timeouts exceed cap. "
            "Either tighten the timeout to ≤5s OR add the method to "
            "ALLOWLISTED_METHODS in this test + phase_c_rpc_exemptions.yaml.\n"
            + "\n".join(violations)
        )


def test_add_memory_no_longer_a_work_rpc():
    """Phase B B4 retired the `add_memory` work-RPC in favor of the
    one-way `MEMORY_INGEST_REQUEST` event. The proxy method must not
    contain an actual `bus.request_async(...)` call (docstring mentions
    of the historical pattern are fine)."""
    source = PROXY_PATH.read_text()
    tree = ast.parse(source)
    add_memory_func = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.AsyncFunctionDef)
                and node.name == "add_memory"):
            add_memory_func = node
            break
    assert add_memory_func is not None, "add_memory method not found"

    # Walk the function body for actual Call nodes to request_async or _work_rpc_sync.
    work_rpc_calls = []
    publish_calls = []
    for node in ast.walk(add_memory_func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("request_async", "_work_rpc_sync"):
                work_rpc_calls.append(node.lineno)
            elif node.func.attr == "publish":
                publish_calls.append(node.lineno)
    assert not work_rpc_calls, (
        f"add_memory still contains work-RPC call(s) at line(s) "
        f"{work_rpc_calls} — Phase B B4 migration not complete")
    assert publish_calls, (
        "add_memory should use bus.publish for the one-way "
        "MEMORY_INGEST_REQUEST event")


def test_phase_c_rpc_exemptions_yaml_lists_remaining_memory_proxy_sites():
    """The exemptions YAML must list every memory_proxy work-RPC site
    that still exists. Drift between code + YAML breaks G19 audit visibility."""
    yaml_text = EXEMPTIONS_PATH.read_text()
    # We expect explicit entries for the 7 still-present work-RPC methods.
    expected_methods = {
        "memory_proxy.py:query",
        "memory_proxy.py:fetch_mempool",
        "memory_proxy.py:get_top_memories",
        "memory_proxy.py:get_top_memories_for_observatory",
        "memory_proxy.py:fetch_mempool_for_observatory",
        "memory_proxy.py:run_meditation",
        "memory_proxy.py:get_topology",
        "memory_proxy.py:get_knowledge_graph",
    }
    missing = {m for m in expected_methods if m not in yaml_text}
    assert not missing, (
        f"phase_c_rpc_exemptions.yaml missing entries for: {missing}")
    # And `add_memory` should NOT appear (it was retired).
    assert "memory_proxy.py:add_memory" not in yaml_text, (
        "phase_c_rpc_exemptions.yaml still lists memory_proxy.add_memory "
        "as a work-RPC — Phase B B4 retired it (now one-way publish)")
