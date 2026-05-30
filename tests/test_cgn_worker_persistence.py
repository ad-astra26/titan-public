"""Regression guard for the CRITICAL cgn persistence bug.

cgn_state.pt was observed frozen for 1.5 days fleet-wide because:
  1. cgn_worker had NO `SAVE_NOW` handler, so the orchestrator's graceful
     `stop(save_first=True)` checkpoint (SAVE_NOW → SAVE_DONE) was a silent
     no-op for cgn (every other persistence worker handles SAVE_NOW).
  2. cgn's only disk-save paths were dream consolidation + MODULE_SHUTDOWN;
     the periodic online-consolidation path wrote SHM only. Under shm_pid_dead
     restart churn the worker died before any of those fired → all learning
     since the last dream was lost on every restart.

The fix adds (a) a SAVE_NOW handler that calls `cgn._save_state()` + emits
SAVE_DONE, and (b) a periodic time-gated `_maybe_checkpoint_state()` invoked on
BOTH the idle (Empty) and message paths so persistence is independent of a
graceful shutdown. These AST guards pin both so they can't silently regress.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parent.parent
    / "titan_hcl" / "modules" / "cgn_worker.py"
)


def _tree() -> ast.Module:
    return ast.parse(_SRC.read_text())


def _calls_attr(node: ast.AST, attr: str) -> bool:
    """True if any `<something>.<attr>(...)` call appears under node."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr):
            return True
    return False


def test_cgn_worker_handles_save_now_and_saves_state():
    """A `msg_type == bus.SAVE_NOW` branch must exist and call _save_state."""
    src = _SRC.read_text()
    assert "bus.SAVE_NOW" in src, "cgn_worker must reference bus.SAVE_NOW"

    tree = _tree()
    # Find the SAVE_NOW comparison branch and assert it (or its body) saves state.
    save_now_branch_saves = False
    save_done_emitted = "bus.SAVE_DONE" in src
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        # match: <x> == bus.SAVE_NOW
        rhs = node.comparators[0] if node.comparators else None
        if (isinstance(rhs, ast.Attribute) and rhs.attr == "SAVE_NOW"):
            # Walk up isn't trivial in ast; instead check the enclosing If body
            # by re-walking If nodes whose test contains this compare.
            for if_node in ast.walk(tree):
                if isinstance(if_node, ast.If) and _contains(if_node.test, node):
                    if any(_calls_attr(s, "_save_state") for s in if_node.body):
                        save_now_branch_saves = True
    assert save_now_branch_saves, (
        "the SAVE_NOW branch must call cgn._save_state() — without it the "
        "Guardian's graceful save_first is a no-op for cgn (the original bug)")
    assert save_done_emitted, (
        "SAVE_NOW handler should emit bus.SAVE_DONE so the orchestrator's "
        "stop() doesn't block the full save_timeout")


def test_cgn_worker_has_periodic_state_checkpoint():
    """A periodic `_maybe_checkpoint_state` must exist (calls _save_state) and
    be invoked at least twice (idle path + message path) so persistence does
    not depend on a graceful shutdown."""
    src = _SRC.read_text()
    assert "_maybe_checkpoint_state" in src

    tree = _tree()
    # The function definition must call cgn._save_state().
    fn_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_maybe_checkpoint_state"
    ]
    assert fn_defs, "_maybe_checkpoint_state must be defined"
    assert _calls_attr(fn_defs[0], "_save_state"), (
        "_maybe_checkpoint_state must call cgn._save_state()")

    # Counted invocations (excludes the def line) must be >= 2 (both loop paths).
    invocations = src.count("_maybe_checkpoint_state()")
    assert invocations >= 2, (
        f"_maybe_checkpoint_state must be invoked on BOTH the idle and message "
        f"paths (avoids the except-Empty trap); found {invocations} call sites")


def _contains(parent: ast.AST, target: ast.AST) -> bool:
    return any(n is target for n in ast.walk(parent))
