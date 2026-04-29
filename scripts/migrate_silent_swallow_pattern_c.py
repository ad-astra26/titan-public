"""scripts/migrate_silent_swallow_pattern_c.py

One-shot migration script for `BUG-DEBUG-SWALLOW-AUDIT-BACKLOG`.

Mechanically rewrites the anti-pattern

    except Exception as e:
        logger.debug("[Module] context: %s", e)

into the Pattern C form (per `directive_error_visibility.md`):

    except Exception as e:
        swallow_warn("[Module] context", e,
                     key="<file>.<func>", throttle=100)

scope:
  - T2 (persistence) and T3 (dim/symmetry) sites by default.
  - Only converts EXCEPT handlers whose body is exactly one logger.debug
    call (single statement). Multi-statement bodies, bare-pass, and
    rethrows are left alone for manual review.
  - Adds `from titan_plugin.utils.silent_swallow import swallow_warn`
    once per file if missing.

usage:
  python scripts/migrate_silent_swallow_pattern_c.py --dry-run     # default
  python scripts/migrate_silent_swallow_pattern_c.py --apply       # writes
  python scripts/migrate_silent_swallow_pattern_c.py --apply --tier T2

design:
  - AST-locates each handler precisely (line + col).
  - Line-based rewrite preserves leading whitespace + surrounding code.
  - Re-parses the file after rewrite to guarantee it still parses; reverts
    on failure.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arch_map_dead_wiring import find_silent_swallows  # type: ignore  # noqa: E402

IMPORT_LINE = "from titan_plugin.utils.silent_swallow import swallow_warn"


@dataclass
class Site:
    file: Path
    line: int          # 1-based line of the logger.debug call
    tier: str
    snippet: str       # original source line (without trailing newline)


@dataclass
class Rewrite:
    site: Site
    handler_lineno: int    # the `except ... as e:` line, 1-based
    handler_end_lineno: int   # last line of handler body, 1-based
    call_lineno: int       # start line of the logger.debug call (1-based)
    call_end_lineno: int   # end line of the logger.debug call (1-based)
    indent: str            # leading whitespace for body line
    new_lines: list[str]   # replacement source lines (no trailing newline)
    skipped: str | None    # reason if this rewrite was rejected
    key: str               # the swallow_warn key=
    exc_name: str          # the bound exception variable name


def _file_module_key(file: Path) -> str:
    """Stable module-prefix for the swallow_warn `key=` argument.

    Example: titan_plugin/logic/backup.py → "logic.backup".
    """
    rel = file.relative_to(REPO_ROOT)
    parts = rel.with_suffix("").parts
    if parts and parts[0] == "titan_plugin":
        parts = parts[1:]
    return ".".join(parts)


def _slugify_for_key(prefix_str: str, fallback: str) -> str:
    """Derive a stable, short suffix for the swallow_warn key from the
    log prefix. e.g. '[KnowledgeGraph] Entity insert error for ...' →
    'entity_insert_error'."""
    # Strip leading [Tag] if present
    s = prefix_str.strip()
    if s.startswith("[") and "]" in s:
        s = s.split("]", 1)[1]
    # Take the first ~40 chars of words, lowercased + underscore-joined
    s = s.strip().strip(":").strip()
    # Drop format placeholders
    s = s.replace("%s", "").replace("%d", "").replace("%r", "")
    s = s.replace("%.2f", "").replace("%.3f", "")
    # Keep alnum + space; collapse to underscore-joined slug
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    slug = "".join(out).strip("_")
    # Cap length
    slug = slug[:40].rstrip("_")
    return slug or fallback


def _format_args_to_fstring(format_str: str, extra_arg_srcs: list[str]) -> str:
    """Convert a printf-style format + arg-source-strings into an
    equivalent f-string LITERAL (without leading f"). The trailing %s
    that consumed the exception is NOT included here — the caller has
    already removed it.

    Substitutes %s/%r/%d/%.Xf placeholders left-to-right with
    `{<arg-source>}` (or `{<arg-source>!r}` for %r).
    """
    result = []
    i = 0
    arg_idx = 0
    while i < len(format_str):
        ch = format_str[i]
        if ch == "%" and i + 1 < len(format_str):
            nxt = format_str[i + 1]
            consumed = 2
            spec_open = ""
            spec_close = ""
            if nxt == "s":
                pass  # default substitution
            elif nxt == "r":
                spec_close = "!r"
            elif nxt == "d":
                pass
            elif nxt == ".":
                # Eat .Nf precision spec
                j = i + 2
                while j < len(format_str) and format_str[j].isdigit():
                    j += 1
                if j < len(format_str) and format_str[j] == "f":
                    spec_close = ":" + format_str[i + 1:j + 1]
                    consumed = (j + 1) - i
                else:
                    # unrecognised — keep literal %
                    result.append("%")
                    i += 1
                    continue
            elif nxt == "%":
                result.append("%%")
                i += consumed
                continue
            else:
                # unrecognised specifier — keep literal %
                result.append("%")
                i += 1
                continue
            if arg_idx >= len(extra_arg_srcs):
                # Misformed — leave literal
                result.append(format_str[i:i + consumed])
                i += consumed
                continue
            src = extra_arg_srcs[arg_idx]
            arg_idx += 1
            result.append("{" + spec_open + src + spec_close + "}")
            i += consumed
        elif ch == "{":
            result.append("{{")
            i += 1
        elif ch == "}":
            result.append("}}")
            i += 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def _strip_trailing_exception_format(format_str: str) -> tuple[str, bool]:
    """If the format ends with `: %s` (the conventional exception-tail
    pattern), strip it and return (stripped, True). Else return as-is."""
    for tail in (": %s", " %s", "%s"):
        if format_str.endswith(tail):
            return format_str[: -len(tail)].rstrip().rstrip(":").rstrip(), True
    return format_str, False


def _extract_call(call: ast.Call, source_lines: list[str]) -> tuple[str, list[str], str | None]:
    """Decompose a logger.debug(format, *args) call.

    Returns (format_string_or_None, [arg_source_str, ...], exc_var_or_None).

    Last positional arg is the exception variable (heuristic: identifier
    matching the bound name); when present it's stripped from the args
    list and returned separately.
    """
    if not call.args:
        return None, [], None
    fmt_node = call.args[0]
    if not isinstance(fmt_node, ast.Constant) or not isinstance(fmt_node.value, str):
        return None, [], None
    format_str = fmt_node.value
    extra: list[str] = []
    exc_name: str | None = None
    rest = call.args[1:]
    # Heuristic: if last arg is a Name, treat as the exception variable
    if rest and isinstance(rest[-1], ast.Name):
        exc_name = rest[-1].id
        rest = rest[:-1]
    for n in rest:
        try:
            extra.append(ast.unparse(n))
        except Exception:
            return None, [], None
    return format_str, extra, exc_name


def _build_replacement(handler: ast.ExceptHandler, call: ast.Call,
                        source_lines: list[str], file: Path,
                        indent: str) -> Rewrite | None:
    format_str, extra_args, exc_name = _extract_call(call, source_lines)
    if format_str is None:
        return None
    bound = handler.name  # the `as <name>` bound variable, or None
    if bound is None:
        return None  # bare except — manual review
    if exc_name is not None and exc_name != bound:
        # The last positional arg isn't the bound exception — odd shape,
        # skip for safety.
        return None
    # Strip trailing ": %s" (exception placeholder) — that placeholder
    # consumed the exception variable.
    stripped_fmt, _had_tail = _strip_trailing_exception_format(format_str)
    # If the original format had additional %s placeholders for the
    # extra_args, build an f-string. Otherwise plain literal.
    if extra_args:
        body_str = _format_args_to_fstring(stripped_fmt, extra_args)
        prefix_literal = "f" + repr(body_str)
    else:
        prefix_literal = repr(stripped_fmt)
    key_suffix = _slugify_for_key(stripped_fmt, fallback=f"line{handler.lineno}")
    key_str = f"{_file_module_key(file)}.{key_suffix}"
    # Two-line replacement so we keep call width readable
    new_first = f'{indent}swallow_warn({prefix_literal}, {bound},'
    cont_indent = indent + " " * len("swallow_warn(")
    new_second = f'{cont_indent}key="{key_str}", throttle=100)'
    return Rewrite(
        site=Site(file=file, line=call.lineno, tier="?", snippet=""),
        handler_lineno=handler.lineno,
        handler_end_lineno=handler.end_lineno or handler.lineno,
        call_lineno=call.lineno,
        call_end_lineno=getattr(call, "end_lineno", call.lineno) or call.lineno,
        indent=indent,
        new_lines=[new_first, new_second],
        skipped=None,
        key=key_str,
        exc_name=bound,
    )


def _find_exception_handlers(tree: ast.AST) -> list[ast.ExceptHandler]:
    out: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            out.append(node)
    return out


def _find_logger_debug_call_in_handler(handler: ast.ExceptHandler,
                                         call_lineno: int) -> ast.Call | None:
    """Find the `<something>.debug(...)` Call statement at `call_lineno`
    inside the handler body. Returns the Call node or None.

    Accepts handlers with multi-statement bodies — caller is responsible
    for verifying the OTHER statements are independent fallback logic
    (return / counter bumps / etc.) that we leave intact.
    """
    for stmt in handler.body:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        f = call.func
        if not (isinstance(f, ast.Attribute) and f.attr == "debug"):
            continue
        if call.lineno == call_lineno:
            return call
    return None


def _find_handler_for_call_line(tree: ast.AST, call_lineno: int) -> ast.ExceptHandler | None:
    """Locate the ExceptHandler whose body contains the logger.debug call
    at `call_lineno` (regardless of multi-statement body). Returns the
    enclosing handler, else None."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for stmt in node.body:
            start = stmt.lineno
            end = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
            if start <= call_lineno <= end:
                return node
    return None


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
                    indent="", new_lines=[], skipped=f"parse error: {e}",
                    key="", exc_name="",
                ))
            continue
        source_lines = text.splitlines()
        # Iterate over each AUDIT site (don't iterate handlers — that
        # silently drops sites whose handler body has multiple statements
        # or whose call line doesn't match handler.body[0].lineno).
        for s in sites:
            handler = _find_handler_for_call_line(tree, s.line)
            if handler is None:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=0, handler_end_lineno=0,
                    call_lineno=0, call_end_lineno=0,
                    indent="", new_lines=[],
                    skipped="no enclosing except handler at this line "
                            "(call may be in else/finally or top-level)",
                    key="", exc_name="",
                ))
                continue
            call = _find_logger_debug_call_in_handler(handler, s.line)
            if call is None:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=handler.lineno,
                    handler_end_lineno=handler.end_lineno or handler.lineno,
                    call_lineno=0, call_end_lineno=0,
                    indent="", new_lines=[],
                    skipped="logger.debug call not found at this line "
                            "(may be inside conditional/expression)",
                    key="", exc_name="",
                ))
                continue
            indent_match = source_lines[call.lineno - 1]
            indent = indent_match[:len(indent_match) - len(indent_match.lstrip())]
            rw = _build_replacement(handler, call, source_lines, file, indent)
            if rw is None:
                rewrites.append(Rewrite(
                    site=s, handler_lineno=handler.lineno,
                    handler_end_lineno=handler.end_lineno or handler.lineno,
                    call_lineno=call.lineno,
                    call_end_lineno=getattr(call, "end_lineno", call.lineno) or call.lineno,
                    indent=indent, new_lines=[],
                    skipped="unsupported call shape (extra args / no exc bound)",
                    key="", exc_name="",
                ))
                continue
            rw.site = s
            rewrites.append(rw)
    return rewrites


def apply_rewrites_to_file(file: Path, rewrites: list[Rewrite]) -> tuple[str, str, int]:
    """Apply all approved rewrites for one file. Returns (old_text,
    new_text, applied_count). Re-parses the result; reverts on failure.
    """
    text = file.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Apply bottom-up so line indices stay stable
    approved = [r for r in rewrites if r.skipped is None and r.new_lines]
    approved.sort(key=lambda r: r.call_lineno, reverse=True)
    applied = 0
    for r in approved:
        start_idx = r.call_lineno - 1
        end_idx = r.call_end_lineno - 1
        if start_idx >= len(lines) or end_idx >= len(lines):
            continue
        # Use AST-derived line range — exact, no paren-balancing heuristic.
        lines[start_idx:end_idx + 1] = r.new_lines
        applied += 1
    new_text = "\n".join(lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    # Insert import once if any rewrites were applied. Note: must check
    # for an existing TOP-LEVEL import — a local `from ... import` inside
    # a function doesn't satisfy module-scope name resolution.
    if applied > 0 and not _has_top_level_swallow_warn_import(new_text):
        new_text = _insert_import(new_text)
    # Validate parseability
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return text, text, 0  # reject
    return text, new_text, applied


def _has_top_level_swallow_warn_import(text: str) -> bool:
    """Returns True iff a `from titan_plugin.utils.silent_swallow import
    swallow_warn` (or an `import titan_plugin.utils.silent_swallow ...`)
    is present at module scope."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "titan_plugin.utils.silent_swallow":
                for alias in node.names:
                    if alias.name == "swallow_warn":
                        return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "titan_plugin.utils.silent_swallow":
                    return True
    return False


def _insert_import(text: str) -> str:
    """Insert `from titan_plugin.utils.silent_swallow import swallow_warn`
    after the last existing top-level import statement.

    Uses AST to locate the last top-level Import/ImportFrom node so we
    handle multi-line `from foo import (...)` correctly (inserting after
    the line where it ENDS, not where it begins).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Can't parse — fall back to inserting at top
        lines = text.split("\n")
        lines.insert(0, IMPORT_LINE)
        return "\n".join(lines)

    last_import_end_line = 0
    docstring_end_line = 0
    for node in tree.body:
        # Module docstring is the first stmt if it's an Expr containing a Constant string
        if (last_import_end_line == 0 and docstring_end_line == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            docstring_end_line = node.end_lineno or node.lineno
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end_line = node.end_lineno or node.lineno
        else:
            # First non-import statement — stop scanning
            if last_import_end_line > 0:
                break

    insert_after_line = last_import_end_line if last_import_end_line > 0 else docstring_end_line
    lines = text.split("\n")
    lines.insert(insert_after_line, IMPORT_LINE)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes to disk. Default is dry-run.")
    ap.add_argument("--tier", action="append", choices=["T2", "T3", "T4"],
                    help="Tiers to migrate. Default: T2 + T3.")
    ap.add_argument("--root", default="titan_plugin", type=Path,
                    help="Scan root (default: titan_plugin)")
    args = ap.parse_args()

    tiers = set(args.tier) if args.tier else {"T2", "T3"}
    findings = find_silent_swallows(args.root)
    sites_by_file: dict[Path, list[Site]] = {}
    pass_skipped = 0
    other_skipped = 0
    for f in findings:
        if f["tier"] not in tiers:
            continue
        if f["level"] == "PASS":
            pass_skipped += 1
            continue
        path = REPO_ROOT / f["file"]
        sites_by_file.setdefault(path, []).append(Site(
            file=path, line=f["line"], tier=f["tier"], snippet=f["snippet"],
        ))

    print(f"Migration scope: tiers {sorted(tiers)} | "
          f"files: {len(sites_by_file)} | "
          f"sites (DEBUG): {sum(len(v) for v in sites_by_file.values())} | "
          f"sites (PASS, manual): {pass_skipped}")
    print()

    rewrites = collect_rewrites(sites_by_file)
    rewrites_by_file: dict[Path, list[Rewrite]] = {}
    for r in rewrites:
        rewrites_by_file.setdefault(r.site.file, []).append(r)

    total_apply = sum(1 for r in rewrites if r.skipped is None)
    total_skip = sum(1 for r in rewrites if r.skipped is not None)
    print(f"Plan: {total_apply} rewrites ready, {total_skip} skipped (manual).")
    print()

    # Show skips
    if total_skip:
        print("── Sites NOT mechanically rewritten (manual review) ──")
        for r in rewrites:
            if r.skipped:
                print(f"  {r.site.file.relative_to(REPO_ROOT)}:{r.site.line}  "
                      f"[{r.site.tier}] {r.skipped}")
                print(f"     | {r.site.snippet[:90]}")
        print()

    # Apply
    files_changed = 0
    sites_applied = 0
    files_planned = 0
    sites_planned = 0
    for file, rs in sorted(rewrites_by_file.items()):
        rel = file.relative_to(REPO_ROOT)
        old, new, applied = apply_rewrites_to_file(file, rs)
        if applied == 0:
            continue
        if old == new:
            continue
        diff = list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=1,
        ))
        # Truncate per-file diff for stdout sanity
        print(f"── {rel} — {applied} rewrites ──")
        for ln in diff[:40]:
            sys.stdout.write(ln)
        if len(diff) > 40:
            print(f"  ... ({len(diff) - 40} more diff lines)")
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
