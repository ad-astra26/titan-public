"""scripts/migrate_bare_pass_swallow.py

Sister script to migrate_silent_swallow_pattern_c.py — handles the
remaining anti-pattern where the silent-swallow audit reports level=PASS
(bare `pass` body inside an except). Example:

    try:
        os.chmod(self._cfg.socket_path, 0o660)
    except OSError:
        pass

becomes:

    try:
        os.chmod(self._cfg.socket_path, 0o660)
    except OSError as e:
        swallow_warn(
            "[persistence.writer_service] _bind_socket: os.chmod", e,
            key="persistence.writer_service._bind_socket.line101",
            throttle=100,
        )

Differences vs the Pattern C migrator:
  - Original handler often has NO exception variable (bare `except X:`
    or `except:`). We add `as e` (or use existing bound name if present).
  - Original body is a literal `pass` — we replace it with a swallow_warn
    call and a comment that hints "previously bare pass — surfaced via
    Pattern C 2026-04-25".
  - Prefix is synthesized from <module-tag>.<func-name>.<first-try-stmt-summary>
    since there's no log-string to extract. Stable + grep-able.
  - Bare `except:` is upgraded to `except Exception as e:` (Pattern C
    is incompatible with truly-everything bare-except; if the original
    catches BaseException-class exits like SystemExit/KeyboardInterrupt,
    the rewrite would change semantics — we explicitly skip those).

usage:
  python scripts/migrate_bare_pass_swallow.py             # dry-run, T2+T3+T4
  python scripts/migrate_bare_pass_swallow.py --apply
  python scripts/migrate_bare_pass_swallow.py --tier T2 --apply
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the find_silent_swallows scanner + helpers from the sister script
from arch_map_dead_wiring import find_silent_swallows  # type: ignore  # noqa: E402
from migrate_silent_swallow_pattern_c import (  # type: ignore  # noqa: E402
    IMPORT_LINE,
    _file_module_key,
    _has_top_level_swallow_warn_import,
    _insert_import,
)


@dataclass
class Site:
    file: Path
    line: int        # 1-based line of the `pass` statement
    tier: str
    snippet: str


@dataclass
class Rewrite:
    site: Site
    handler_lineno: int
    handler_end_lineno: int
    pass_lineno: int      # line number of the `pass` body statement
    indent: str
    new_lines: list[str]
    new_handler_line: str | None  # if non-None, replaces the `except X:` line
    skipped: str | None
    key: str


SAFE_EXCEPTION_CLASSES = {
    # Heuristic — we'll let `except <these>` rewrite cleanly. Anything
    # broader (BaseException, GeneratorExit, KeyboardInterrupt,
    # SystemExit) gets skipped to preserve existing flow-control intent.
    "Exception",
    "OSError", "IOError", "FileNotFoundError", "PermissionError",
    "ValueError", "TypeError", "KeyError", "AttributeError",
    "RuntimeError", "ImportError", "ModuleNotFoundError",
    "json.JSONDecodeError", "JSONDecodeError",
    "asyncio.CancelledError", "CancelledError",
    "sqlite3.Error", "sqlite3.OperationalError", "sqlite3.IntegrityError",
    "ConnectionError", "ConnectionResetError", "BrokenPipeError",
    "TimeoutError", "asyncio.TimeoutError",
    "UnicodeDecodeError", "UnicodeEncodeError",
}

UNSAFE_EXCEPTION_CLASSES = {
    "BaseException", "GeneratorExit", "KeyboardInterrupt", "SystemExit",
}


def _exception_class_name(node) -> str:
    """ast.unparse for an exception spec — handles plain Names + Attributes
    (e.g. asyncio.CancelledError) + Tuples."""
    return ast.unparse(node)


def _find_enclosing_func_name(tree: ast.AST, line: int) -> str:
    """Return name of the FunctionDef/AsyncFunctionDef enclosing `line`.
    For methods, prefix with the class name (`MyClass.foo`).
    Returns "<module>" if not inside any function."""
    inner_func: tuple[str, int] | None = None
    for cls in ast.walk(tree):
        if isinstance(cls, (ast.ClassDef,)):
            cls_name = cls.name
            for n in ast.walk(cls):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = n.lineno
                    end = getattr(n, "end_lineno", n.lineno) or n.lineno
                    if start <= line <= end:
                        if inner_func is None or start > inner_func[1]:
                            inner_func = (f"{cls_name}.{n.name}", start)
        elif isinstance(cls, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = cls.lineno
            end = getattr(cls, "end_lineno", cls.lineno) or cls.lineno
            if start <= line <= end:
                if inner_func is None or start > inner_func[1]:
                    inner_func = (cls.name, start)
    return inner_func[0] if inner_func else "<module>"


def _summarize_first_try_stmt(handler: ast.ExceptHandler, tree: ast.AST,
                                source_lines: list[str]) -> str:
    """Find the Try whose handlers include `handler`, return a short
    string summarizing its first body statement."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            if handler in node.handlers and node.body:
                stmt = node.body[0]
                try:
                    s = ast.unparse(stmt)
                except Exception:
                    return "<unparseable>"
                # Truncate + normalize whitespace
                s = re.sub(r"\s+", " ", s).strip()
                if len(s) > 60:
                    s = s[:57] + "..."
                return s
    return "<unknown>"


def _find_handler_for_pass_line(tree: ast.AST, pass_line: int
                                  ) -> ast.ExceptHandler | None:
    """Locate the ExceptHandler whose body is exactly one `pass` at
    pass_line."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if len(node.body) != 1:
            continue
        stmt = node.body[0]
        if isinstance(stmt, ast.Pass) and stmt.lineno == pass_line:
            return node
    return None


def _classify_handler_safety(handler: ast.ExceptHandler) -> tuple[bool, str]:
    """Returns (safe_to_rewrite, reason). Skips handlers that catch
    BaseException-tier exceptions or have unusual shapes."""
    spec = handler.type
    if spec is None:
        # Bare `except:` — broad. We'll narrow to `Exception as e` since
        # in an "this is okay to silently swallow" context there's no
        # legitimate use for catching SystemExit/KeyboardInterrupt
        # silently, and if there is, the audit's PASS classification
        # would surface that for review separately.
        return True, "bare-except → Exception"
    spec_str = _exception_class_name(spec)
    # Tuple of types
    if isinstance(spec, ast.Tuple):
        names = [ast.unparse(elt) for elt in spec.elts]
    else:
        names = [spec_str]
    for n in names:
        # Strip parentheses if present
        n = n.strip("() ")
        if any(unsafe in n for unsafe in UNSAFE_EXCEPTION_CLASSES):
            return False, f"catches BaseException-tier ({n})"
    return True, "ok"


def _build_rewrite(handler: ast.ExceptHandler, tree: ast.AST,
                    source_lines: list[str], file: Path) -> Rewrite | None:
    safe, reason = _classify_handler_safety(handler)
    if not safe:
        return Rewrite(
            site=Site(file, handler.body[0].lineno, "?", ""),
            handler_lineno=handler.lineno,
            handler_end_lineno=handler.end_lineno or handler.lineno,
            pass_lineno=handler.body[0].lineno,
            indent="", new_lines=[], new_handler_line=None,
            skipped=reason, key="",
        )
    pass_stmt = handler.body[0]
    pass_lineno = pass_stmt.lineno
    indent_match = source_lines[pass_lineno - 1]
    indent = indent_match[:len(indent_match) - len(indent_match.lstrip())]

    func_name = _find_enclosing_func_name(tree, pass_lineno)
    try_summary = _summarize_first_try_stmt(handler, tree, source_lines)
    module_tag = _file_module_key(file)

    # Choose / introduce exception bound name
    if handler.name:
        exc_name = handler.name
        new_handler_line = None  # don't rewrite the except line
    else:
        exc_name = "_swallow_exc"
        # Build the new `except <class> as _swallow_exc:` line preserving
        # leading indent
        old_handler_line = source_lines[handler.lineno - 1]
        h_indent = old_handler_line[:len(old_handler_line) -
                                     len(old_handler_line.lstrip())]
        if handler.type is None:
            type_text = "Exception"
        else:
            type_text = ast.unparse(handler.type)
        new_handler_line = f"{h_indent}except {type_text} as {exc_name}:"

    prefix_str = f"[{module_tag}] {func_name}: {try_summary}"
    key_suffix = f"{func_name}.line{pass_lineno}"
    key_str = f"{module_tag}.{key_suffix}".replace("<module>.", "")

    # Two-line swallow_warn for readability
    new_lines = [
        f"{indent}swallow_warn({prefix_str!r}, {exc_name},",
        f"{indent}             key={key_str!r}, throttle=100)",
    ]

    return Rewrite(
        site=Site(file, pass_lineno, "?", ""),
        handler_lineno=handler.lineno,
        handler_end_lineno=handler.end_lineno or handler.lineno,
        pass_lineno=pass_lineno,
        indent=indent,
        new_lines=new_lines,
        new_handler_line=new_handler_line,
        skipped=None,
        key=key_str,
    )


def collect_rewrites(sites_by_file: dict[Path, list[Site]]) -> list[Rewrite]:
    rewrites: list[Rewrite] = []
    for file, sites in sites_by_file.items():
        try:
            text = file.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception as e:
            for s in sites:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=0, handler_end_lineno=0,
                    pass_lineno=s.line, indent="", new_lines=[],
                    new_handler_line=None,
                    skipped=f"parse error: {e}", key="",
                ))
            continue
        source_lines = text.splitlines()
        for s in sites:
            handler = _find_handler_for_pass_line(tree, s.line)
            if handler is None:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=0, handler_end_lineno=0,
                    pass_lineno=s.line, indent="", new_lines=[],
                    new_handler_line=None,
                    skipped="no enclosing except-with-bare-pass at this line",
                    key="",
                ))
                continue
            r = _build_rewrite(handler, tree, source_lines, file)
            if r is None:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=handler.lineno,
                    handler_end_lineno=handler.end_lineno or handler.lineno,
                    pass_lineno=s.line, indent="", new_lines=[],
                    new_handler_line=None,
                    skipped="build_rewrite returned None", key="",
                ))
                continue
            r.site = s
            rewrites.append(r)
    return rewrites


def apply_rewrites_to_file(file: Path, rewrites: list[Rewrite]
                            ) -> tuple[str, str, int]:
    text = file.read_text(encoding="utf-8")
    lines = text.splitlines()
    approved = [r for r in rewrites if r.skipped is None and r.new_lines]
    # Apply bottom-up so line indices stay stable. Sort by pass_lineno
    # descending; within the same handler, we touch BOTH the handler
    # line and the pass line, so sort by max of the two.
    approved.sort(key=lambda r: r.pass_lineno, reverse=True)
    applied = 0
    for r in approved:
        # Replace pass with new lines
        idx = r.pass_lineno - 1
        if idx >= len(lines):
            continue
        lines[idx:idx + 1] = r.new_lines
        # If we needed to rewrite the except line, do that AFTER the
        # pass replacement (the handler line is BEFORE pass — earlier
        # index — so still safe even though we're working bottom-up).
        if r.new_handler_line is not None:
            h_idx = r.handler_lineno - 1
            if 0 <= h_idx < len(lines):
                lines[h_idx] = r.new_handler_line
        applied += 1
    new_text = "\n".join(lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    if applied > 0 and not _has_top_level_swallow_warn_import(new_text):
        new_text = _insert_import(new_text)
    try:
        ast.parse(new_text)
    except SyntaxError:
        return text, text, 0
    return text, new_text, applied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tier", action="append",
                    choices=["T2", "T3", "T4"],
                    help="Tiers to migrate. Default: T2 + T3 + T4.")
    ap.add_argument("--root", default="titan_plugin", type=Path)
    args = ap.parse_args()

    tiers = set(args.tier) if args.tier else {"T2", "T3", "T4"}
    findings = find_silent_swallows(args.root)
    sites_by_file: dict[Path, list[Site]] = {}
    debug_skipped = 0
    for f in findings:
        if f["tier"] not in tiers:
            continue
        if f["level"] != "PASS":
            debug_skipped += 1
            continue
        path = REPO_ROOT / f["file"]
        sites_by_file.setdefault(path, []).append(Site(
            file=path, line=f["line"], tier=f["tier"], snippet=f["snippet"],
        ))

    print(f"Migration scope: tiers {sorted(tiers)} | "
          f"files: {len(sites_by_file)} | "
          f"sites (PASS): {sum(len(v) for v in sites_by_file.values())} | "
          f"sites (DEBUG, already migrated): {debug_skipped}")
    print()

    rewrites = collect_rewrites(sites_by_file)
    rewrites_by_file: dict[Path, list[Rewrite]] = {}
    for r in rewrites:
        rewrites_by_file.setdefault(r.site.file, []).append(r)

    total_apply = sum(1 for r in rewrites if r.skipped is None)
    total_skip = sum(1 for r in rewrites if r.skipped is not None)
    print(f"Plan: {total_apply} rewrites ready, {total_skip} skipped (manual).")
    print()
    if total_skip:
        print("── Sites NOT mechanically rewritten (manual review) ──")
        for r in rewrites:
            if r.skipped:
                print(f"  {r.site.file.relative_to(REPO_ROOT)}:{r.site.line}  "
                      f"[{r.site.tier}] {r.skipped}")
        print()

    files_planned = files_changed = 0
    sites_planned = sites_applied = 0
    for file, rs in sorted(rewrites_by_file.items()):
        rel = file.relative_to(REPO_ROOT)
        old, new, applied = apply_rewrites_to_file(file, rs)
        if applied == 0 or old == new:
            continue
        diff = list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=1,
        ))
        print(f"── {rel} — {applied} rewrites ──")
        for ln in diff[:50]:
            sys.stdout.write(ln)
        if len(diff) > 50:
            print(f"  ... ({len(diff) - 50} more diff lines)")
        files_planned += 1
        sites_planned += applied
        if args.apply:
            file.write_text(new, encoding="utf-8")
            files_changed += 1
            sites_applied += applied
        print()

    if args.apply:
        print(f"summary: {sites_applied} sites rewritten across "
              f"{files_changed} files (APPLIED)")
    else:
        print(f"summary: {sites_planned} sites would be rewritten across "
              f"{files_planned} files (DRY-RUN). Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
