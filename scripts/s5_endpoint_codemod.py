#!/usr/bin/env python
"""
scripts/s5_endpoint_codemod.py — text-level + libcst codemod for the S5
amendment.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25), Phase 4.

Two passes:
  1. text_pass(): longest-first string replacements for the canonical
     patterns from titan_plugin.api.state_mapping. Safe because patterns
     are anchored on `plugin.<attr>` which is unique to endpoint code
     accessing the plugin object.
  2. libcst_pass(): structured edits for `await asyncio.wait_for(...)`
     wrappers that wrap what's now a sync attr access. Unwraps to plain
     attr read.

Run:
  python scripts/s5_endpoint_codemod.py --dry-run titan_plugin/api/
  python scripts/s5_endpoint_codemod.py --apply titan_plugin/api/

After --apply, re-run scripts/s5_callsite_audit.py to verify the
remaining unmatched callsites (Category C, manual handling).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import libcst as cst
import libcst.matchers as m

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the mapping table
sys.path.insert(0, str(PROJECT_ROOT))
from titan_plugin.api.state_mapping import build_simple_replacements  # noqa: E402


# ── Pass 1: text-level replacements ───────────────────────────────────


def text_pass(source: str) -> tuple[str, int]:
    """Apply longest-first text replacements with word-boundary safety.

    Two safety rules:
      1. Word boundary at the end — don't match `plugin._foo` when the
         source has `plugin._foo_bar` (the longer one runs first; if it's
         skipped due to LHS protection, we don't fall through to a
         partial match on the shorter form).
      2. LHS protection for replacements that produce non-lvalue
         expressions (e.g. `cache.get(...)`) — must NOT match LHS of
         assignments, since `cache.get(...) = x` is invalid Python.

    All replacements use regex now (uniform handling). Patterns where
    LHS protection skips the match leave the source alone; manual
    Category C pass handles those writes via state.commands.X(...).
    """
    repl = build_simple_replacements()
    # Sort by source length descending so longer matches run before shorter
    # (prevents shorter-as-substring-of-longer false positives).
    items = sorted(repl.items(), key=lambda kv: -len(kv[0]))
    count = 0
    new_source = source
    for src, dst in items:
        # Word-boundary at the end: \b doesn't work after `(` or `)`, so
        # we use a negative lookahead that excludes word chars right after
        # the match. This prevents `plugin._foo` matching inside
        # `plugin._foo_bar`. For sources ending in `)` (callable patterns),
        # \b would also fail; the `[A-Za-z0-9_]` lookahead covers both.
        end_anchor = r"(?![A-Za-z0-9_])"

        is_function_call_target = "cache.get(" in dst or "config.get(" in dst
        if is_function_call_target:
            # LHS protection: skip if followed by an assignment operator.
            pattern = re.compile(
                re.escape(src) + end_anchor +
                r"(?!\s*[+\-*/%|&^]?=(?!=))"
            )
        else:
            pattern = re.compile(re.escape(src) + end_anchor)

        new_source, n = pattern.subn(dst, new_source)
        count += n
    return new_source, count


# ── Pass 2: libcst — unwrap async wait_for + await on sync ────────────


class AsyncUnwrapTransformer(cst.CSTTransformer):
    """Unwrap legacy async patterns that S5 amendment makes synchronous.

    Patterns:

    1. `await asyncio.wait_for(titan_state.network.balance, timeout=X)`
       (after text_pass converted .get_balance() to .balance)
       → `titan_state.network.balance`

       The wait_for argument must be a `titan_state.X.Y` attribute chain
       (no parentheses) for the unwrap to fire. If it's still a Call
       (e.g. unmatched method), we leave it alone.

    2. `await titan_state.X.Y` where titan_state.X.Y is a sync attribute
       → `titan_state.X.Y`

       Detected: Await whose target is an Attribute chain rooted at
       `titan_state`.

    3. `asyncio.to_thread(titan_state.X.Y)` — function reference, not
       called. Unwrap to direct attribute read.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unwrap_count = 0

    @staticmethod
    def _is_titan_state_attr(node: cst.CSTNode) -> bool:
        """True if node is an Attribute chain rooted at titan_state (no calls)."""
        cur: cst.CSTNode = node
        while isinstance(cur, cst.Attribute):
            cur = cur.value
        return isinstance(cur, cst.Name) and cur.value == "titan_state"

    def leave_Await(
        self, original_node: cst.Await, updated_node: cst.Await
    ) -> cst.BaseExpression:
        """Unwrap `await titan_state.X.Y` → `titan_state.X.Y` when target is
        a plain attribute (no call). The await keyword has no semantic
        effect on a non-coroutine; Python actually raises TypeError for
        `await <non-awaitable>`, so we MUST remove it."""
        target = updated_node.expression
        # Pattern: await <Attribute on titan_state>
        if isinstance(target, cst.Attribute) and self._is_titan_state_attr(target):
            self.unwrap_count += 1
            return target
        # Pattern: await asyncio.wait_for(<titan_state.X>, ...)
        if (
            isinstance(target, cst.Call)
            and m.matches(
                target.func,
                m.Attribute(value=m.Name("asyncio"),
                            attr=m.Name("wait_for"))
            )
        ):
            # First positional arg is the awaitable
            if target.args and isinstance(target.args[0].value, cst.Attribute):
                inner = target.args[0].value
                if self._is_titan_state_attr(inner):
                    self.unwrap_count += 1
                    return inner
            # If it's a Call (e.g. titan_state.X.method()) — also valid sync
            if target.args and isinstance(target.args[0].value, cst.Call):
                inner_call = target.args[0].value
                if isinstance(inner_call.func, cst.Attribute) and \
                        self._is_titan_state_attr(inner_call.func):
                    self.unwrap_count += 1
                    return inner_call
        return updated_node


def libcst_pass(source: str) -> tuple[str, int]:
    """Apply structured edits via libcst. Returns (new_source, n_unwraps)."""
    try:
        tree = cst.parse_module(source)
    except Exception as e:
        print(f"  [libcst-pass] parse error: {e}", file=sys.stderr)
        return source, 0
    transformer = AsyncUnwrapTransformer()
    new_tree = tree.visit(transformer)
    return new_tree.code, transformer.unwrap_count


# ── Driver ─────────────────────────────────────────────────────────────


def process_file(path: Path, apply: bool) -> dict:
    """Returns stats dict."""
    src = path.read_text()
    new_src = src
    new_src, text_count = text_pass(new_src)
    new_src, libcst_count = libcst_pass(new_src)
    changed = new_src != src
    if apply and changed:
        path.write_text(new_src)
    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "text_replacements": text_count,
        "libcst_unwraps": libcst_count,
        "changed": changed,
        "diff_lines": _count_diff(src, new_src) if changed else 0,
    }


def _count_diff(a: str, b: str) -> int:
    """Cheap line-diff count for stats."""
    return sum(1 for la, lb in zip(a.splitlines(), b.splitlines()) if la != lb) \
        + abs(len(a.splitlines()) - len(b.splitlines()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["titan_plugin/api"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing files")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes to disk")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply", file=sys.stderr)
        sys.exit(1)

    # Files to EXCLUDE from codemod — these are S5 infrastructure that
    # already uses titan_state semantics (no plugin.X patterns to rewrite).
    EXCLUDE_NAMES = {
        "state_accessor.py",
        "state_mapping.py",
        "shm_reader_bank.py",
        "bus_subscriber.py",
        "cached_state.py",
        "command_sender.py",
        # api_subprocess.py is the construction site for TitanStateAccessor;
        # we modify it manually post-codemod (Phase 6).
        "api_subprocess.py",
    }

    paths = []
    for p in args.paths:
        path = PROJECT_ROOT / p
        if path.is_dir():
            paths.extend(
                p for p in sorted(path.rglob("*.py"))
                if p.name not in EXCLUDE_NAMES
            )
        elif path.is_file() and path.suffix == ".py":
            if path.name not in EXCLUDE_NAMES:
                paths.append(path)

    print(f"\n[s5-codemod] mode={'apply' if args.apply else 'dry-run'} "
          f"files={len(paths)}")
    print("=" * 70)

    total_text = 0
    total_libcst = 0
    files_changed = 0
    for p in paths:
        stats = process_file(p, apply=args.apply)
        if stats["text_replacements"] or stats["libcst_unwraps"]:
            print(f"  {stats['file']}: "
                  f"text={stats['text_replacements']} "
                  f"libcst={stats['libcst_unwraps']} "
                  f"diff_lines={stats['diff_lines']}")
            total_text += stats["text_replacements"]
            total_libcst += stats["libcst_unwraps"]
            if stats["changed"]:
                files_changed += 1

    print("=" * 70)
    print(f"  TOTAL: text={total_text} libcst={total_libcst} "
          f"files_changed={files_changed}")
    if not args.apply:
        print("\n  (dry-run — no files written; use --apply to write)")


if __name__ == "__main__":
    main()
