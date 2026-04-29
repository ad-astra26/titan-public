#!/usr/bin/env python3
"""Mechanical refactor — replace `with sqlite3.connect(...) as conn:
conn.execute(<write>); conn.commit()` blocks with `self._route_write(...)`.

Conservative: only refactors blocks whose body is exactly
  conn.execute(<write-sql>, <params>)
  conn.commit()
(or with the optional cursor-rowcount return idiom).

Skipped (left for manual review):
  - blocks with SELECT (mixed read+write)
  - blocks with conditional / loop / multiple writes
  - blocks where return value depends on cursor (rowcount, lastrowid)
  - non-write SQL (won't match table-detection regex)

Usage:
    python scripts/refactor_to_route_write.py --dry-run path/to/file.py
    python scripts/refactor_to_route_write.py --apply path/to/file.py
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

TABLE_RE = re.compile(
    r"^\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|REPLACE\s+INTO|"
    r"INSERT\s+OR\s+(?:IGNORE|REPLACE)\s+INTO)\s+['\"`]?(\w+)['\"`]?",
    re.IGNORECASE,
)


def detect_table(sql: str) -> Optional[str]:
    """Extract the primary write target from a SQL statement (1st line)."""
    stripped = sql.strip().lstrip("'\"")
    for line in stripped.splitlines():
        m = TABLE_RE.match(line)
        if m:
            return m.group(1)
    return None


def is_sqlite_connect_with(node: ast.With) -> bool:
    """Check if a `with` is `with sqlite3.connect(self._db_path, ...) as conn:`
    or `with self._connect() as conn:` (events_teacher pattern)."""
    if len(node.items) != 1:
        return False
    item = node.items[0]
    ctx = item.context_expr
    target = item.optional_vars
    # Target must be `conn`
    if not isinstance(target, ast.Name) or target.id != "conn":
        return False
    # Context: sqlite3.connect(...) or self._connect()
    if isinstance(ctx, ast.Call):
        func = ctx.func
        if isinstance(func, ast.Attribute):
            if (isinstance(func.value, ast.Name) and func.value.id == "sqlite3"
                    and func.attr == "connect"):
                return True
            if (isinstance(func.value, ast.Name) and func.value.id == "self"
                    and func.attr == "_connect"):
                return True
    return False


def is_simple_write_block(node: ast.With) -> Optional[Tuple[ast.Call, str]]:
    """Return (execute_call, sql_str) if this With body is exactly:
        conn.execute(<sql>, <params>)
        conn.commit()
    where <sql> is a static string and resolves to a write target.
    Else None.
    """
    body = node.body
    if len(body) != 2:
        return None

    # First stmt: Expr(Call(conn.execute(...)))
    s1 = body[0]
    if not isinstance(s1, ast.Expr):
        return None
    if not isinstance(s1.value, ast.Call):
        return None
    call = s1.value
    if not (isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "conn"
            and call.func.attr == "execute"):
        return None
    if len(call.args) < 1:
        return None

    # Extract SQL string literal (may be concatenated string-of-strings)
    sql_node = call.args[0]
    sql = _extract_string(sql_node)
    if sql is None:
        return None

    table = detect_table(sql)
    if table is None:
        return None  # not a write SQL

    # Second stmt: Expr(Call(conn.commit()))
    s2 = body[1]
    if not isinstance(s2, ast.Expr):
        return None
    if not isinstance(s2.value, ast.Call):
        return None
    call2 = s2.value
    if not (isinstance(call2.func, ast.Attribute)
            and isinstance(call2.func.value, ast.Name)
            and call2.func.value.id == "conn"
            and call2.func.attr == "commit"):
        return None

    return call, table


def _extract_string(node: ast.AST) -> Optional[str]:
    """Resolve a string literal (incl. implicit concatenation, """""" blocks)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # f-string — not safe to mechanically refactor (params embedded)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _extract_string(node.left)
        right = _extract_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def find_refactor_sites(source: str) -> List[Tuple[ast.With, ast.Call, str]]:
    """Walk AST, find all simple write-block With sites."""
    tree = ast.parse(source)
    sites: List[Tuple[ast.With, ast.Call, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and is_sqlite_connect_with(node):
            simple = is_simple_write_block(node)
            if simple:
                call, table = simple
                sites.append((node, call, table))
    return sites


def find_tryfinally_sites(
    source: str,
) -> List[Tuple[ast.FunctionDef, ast.Assign, ast.Try, ast.Call, str, bool]]:
    """Find functions matching events_teacher's try/finally pattern:

        def fn(self, ...):
            <preamble lines>
            conn = self._connect()
            try:
                <maybe: cur =>conn.execute(<sql>, params)
                conn.commit()
                <maybe: return cur.lastrowid|cur.rowcount>
            finally:
                conn.close()

    Returns list of (FunctionDef, Assign(conn=...), Try, exec_call, table,
    returns_lastrowid).
    """
    tree = ast.parse(source)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Walk function body to find: Assign(conn=self._connect()), then Try
        # adjacent. We support arbitrary preamble before the Assign.
        body = node.body
        for i, stmt in enumerate(body):
            if not isinstance(stmt, ast.Assign):
                continue
            if (len(stmt.targets) != 1
                    or not isinstance(stmt.targets[0], ast.Name)
                    or stmt.targets[0].id != "conn"):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            call_func = stmt.value.func
            if not (isinstance(call_func, ast.Attribute)
                    and isinstance(call_func.value, ast.Name)
                    and call_func.value.id == "self"
                    and call_func.attr == "_connect"):
                continue
            # Next stmt should be Try
            if i + 1 >= len(body) or not isinstance(body[i + 1], ast.Try):
                continue
            try_node = body[i + 1]
            # Try.body must be: [Expr|Assign(conn.execute), Expr(conn.commit)]
            # or [Expr|Assign(conn.execute), Expr(conn.commit), Return cur.x]
            tb = try_node.body
            if len(tb) not in (2, 3):
                continue

            # First stmt: conn.execute or cur = conn.execute
            execute_stmt = tb[0]
            cursor_var = None
            exec_call = None
            if isinstance(execute_stmt, ast.Expr) and isinstance(execute_stmt.value, ast.Call):
                exec_call = execute_stmt.value
            elif isinstance(execute_stmt, ast.Assign):
                if (len(execute_stmt.targets) == 1
                        and isinstance(execute_stmt.targets[0], ast.Name)
                        and isinstance(execute_stmt.value, ast.Call)):
                    cursor_var = execute_stmt.targets[0].id
                    exec_call = execute_stmt.value
            if exec_call is None:
                continue
            if not (isinstance(exec_call.func, ast.Attribute)
                    and isinstance(exec_call.func.value, ast.Name)
                    and exec_call.func.value.id == "conn"
                    and exec_call.func.attr == "execute"):
                continue

            sql_node = exec_call.args[0] if exec_call.args else None
            if sql_node is None:
                continue
            sql = _extract_string(sql_node)
            if sql is None:
                continue
            table = detect_table(sql)
            if table is None:
                continue

            # Second stmt: conn.commit()
            commit_stmt = tb[1]
            if not (isinstance(commit_stmt, ast.Expr)
                    and isinstance(commit_stmt.value, ast.Call)
                    and isinstance(commit_stmt.value.func, ast.Attribute)
                    and isinstance(commit_stmt.value.func.value, ast.Name)
                    and commit_stmt.value.func.value.id == "conn"
                    and commit_stmt.value.func.attr == "commit"):
                continue

            # Optional 3rd stmt: Return cur.lastrowid (or cur.rowcount)
            returns_lastrowid = False
            if len(tb) == 3:
                ret_stmt = tb[2]
                if not (isinstance(ret_stmt, ast.Return)
                        and isinstance(ret_stmt.value, ast.Attribute)
                        and isinstance(ret_stmt.value.value, ast.Name)
                        and cursor_var
                        and ret_stmt.value.value.id == cursor_var
                        and ret_stmt.value.attr in ("lastrowid", "rowcount")):
                    continue  # 3rd stmt is something else — not our pattern
                returns_lastrowid = (ret_stmt.value.attr == "lastrowid")

            # finalbody must be exactly [Expr(conn.close())]
            fb = try_node.finalbody
            if len(fb) != 1:
                continue
            close_stmt = fb[0]
            if not (isinstance(close_stmt, ast.Expr)
                    and isinstance(close_stmt.value, ast.Call)
                    and isinstance(close_stmt.value.func, ast.Attribute)
                    and isinstance(close_stmt.value.func.value, ast.Name)
                    and close_stmt.value.func.value.id == "conn"
                    and close_stmt.value.func.attr == "close"):
                continue

            # All good — record the site.
            sites.append((node, stmt, try_node, exec_call, table, returns_lastrowid))
    return sites


def refactor_file(path: Path, apply: bool = False) -> Tuple[int, int]:
    """Returns (refactored_count, skipped_count). Writes file iff apply=True."""
    source = path.read_text(encoding="utf-8")
    sites = find_refactor_sites(source)
    tryfinally_sites = find_tryfinally_sites(source)

    # Also count all sqlite-connect Withs in the file for "skipped" reporting
    tree = ast.parse(source)
    total_with_blocks = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.With) and is_sqlite_connect_with(n)
    )

    total_sites = total_with_blocks + len(tryfinally_sites)
    refactorable_count = len(sites) + len(tryfinally_sites)

    if refactorable_count == 0:
        print(f"  [{path.name}] 0 refactorable sites "
              f"({total_sites} candidate blocks total — all need manual review)")
        return (0, total_sites)

    # Process in REVERSE source order so byte offsets stay valid as we replace.
    # Combine With-pattern and try/finally-pattern sites, mark which kind.
    combined: List[Tuple[int, str, object]] = []
    for s in sites:
        combined.append((s[0].lineno, "with", s))
    for s in tryfinally_sites:
        # try/finally site spans from the Assign(conn=...) to the Try.end_lineno
        _fn_node, assign_node, try_node, _exec_call, _table, _ret_last = s
        combined.append((assign_node.lineno, "tryfinally", s))
    sites_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    new_source = source
    refactored = 0

    for _lineno, kind, site_data in sites_sorted:
      if kind == "tryfinally":
        _fn_node, assign_node, try_node, exec_call, table, returns_lastrowid = site_data
        # The block to replace spans from assign_node.lineno to try_node.end_lineno
        start_line = assign_node.lineno
        end_line = try_node.end_lineno
        lines = new_source.splitlines(keepends=True)
        with_line_text = lines[start_line - 1]
        indent = with_line_text[:len(with_line_text) - len(with_line_text.lstrip())]
        # Extract sql + params text from source
        sql_arg = exec_call.args[0]
        sql_text = _slice_source(lines, sql_arg.lineno, sql_arg.col_offset,
                                  sql_arg.end_lineno, sql_arg.end_col_offset)
        params_text = "()"
        if len(exec_call.args) > 1:
            p_first = exec_call.args[1]
            p_last = exec_call.args[-1]
            params_text = _slice_source(
                lines, p_first.lineno, p_first.col_offset,
                p_last.end_lineno, p_last.end_col_offset)
        # Replacement
        prefix = "return self._route_write" if returns_lastrowid else "self._route_write"
        if "\n" in sql_text or "\n" in params_text:
            sql_indent = indent + "    "
            replacement = (
                f"{indent}{prefix}(\n"
                f"{sql_indent}{sql_text.lstrip()},\n"
                f"{sql_indent}{params_text.lstrip()},\n"
                f"{sql_indent}table=\"{table}\",\n"
                f"{indent})"
            )
        else:
            replacement = (
                f"{indent}{prefix}({sql_text}, {params_text}, "
                f"table=\"{table}\")"
            )
        before = "".join(lines[:start_line - 1])
        after = "".join(lines[end_line:])
        new_source = before + replacement + ("\n" if not replacement.endswith("\n") else "") + after
        lines = new_source.splitlines(keepends=True)
        refactored += 1
        continue

      # else: with-pattern
      with_node, exec_call, table = site_data
      start_line = with_node.lineno
      end_line = with_node.end_lineno
      lines = new_source.splitlines(keepends=True)
      with_line_text = lines[start_line - 1]
      indent = with_line_text[:len(with_line_text) - len(with_line_text.lstrip())]
      sql_arg = exec_call.args[0]
      sql_text = _slice_source(lines, sql_arg.lineno, sql_arg.col_offset,
                                sql_arg.end_lineno, sql_arg.end_col_offset)
      params_text = "()"
      if len(exec_call.args) > 1:
          p_first = exec_call.args[1]
          p_last = exec_call.args[-1]
          params_text = _slice_source(
              lines, p_first.lineno, p_first.col_offset,
              p_last.end_lineno, p_last.end_col_offset)
      if "\n" in sql_text or "\n" in params_text:
          sql_indent = indent + "    "
          replacement = (
              f"{indent}self._route_write(\n"
              f"{sql_indent}{sql_text.lstrip()},\n"
              f"{sql_indent}{params_text.lstrip()},\n"
              f"{sql_indent}table=\"{table}\",\n"
              f"{indent})"
          )
      else:
          replacement = (
              f"{indent}self._route_write({sql_text}, {params_text}, "
              f"table=\"{table}\")"
          )
      before = "".join(lines[:start_line - 1])
      after = "".join(lines[end_line:])
      new_source = before + replacement + ("\n" if not replacement.endswith("\n") else "") + after
      lines = new_source.splitlines(keepends=True)
      refactored += 1

    skipped = total_sites - refactored
    print(f"  [{path.name}] refactored {refactored} sites, "
          f"skipped {skipped} (need manual review)")

    if apply:
        path.write_text(new_source, encoding="utf-8")
        print(f"  [{path.name}] WRITTEN")
    else:
        # Show diff preview
        print(f"  [{path.name}] DRY RUN — pass --apply to write changes")

    return (refactored, skipped)


def _slice_source(lines: List[str], start_line: int, start_col: int,
                   end_line: int, end_col: int) -> str:
    """Slice text from source-lines using ast 1-based line + 0-based col."""
    if start_line == end_line:
        return lines[start_line - 1][start_col:end_col]
    # Multi-line slice
    pieces = [lines[start_line - 1][start_col:]]
    for ln in range(start_line, end_line - 1):
        pieces.append(lines[ln])
    pieces.append(lines[end_line - 1][:end_col])
    return "".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true",
                          help="write changes (default: dry run)")
    args = parser.parse_args()

    total_ref = 0
    total_skip = 0
    for p in args.paths:
        if not p.exists():
            print(f"  [{p}] not found")
            continue
        ref, skip = refactor_file(p, apply=args.apply)
        total_ref += ref
        total_skip += skip

    print()
    print(f"TOTAL: refactored={total_ref}  manual-review={total_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
