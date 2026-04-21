"""AST scan: forbid hardcoded `[T1:` / `[T2:` / `[T3:` log-prefix literals.

Background. On 2026-04-21 we discovered spirit_worker.py had three
logger.info() calls with hardcoded `[T2:EXPRESSION...]`, `[T2:SELF_EXPLORE]`,
and `[T1:SPEAK-CHECK]` prefixes — introduced 2026-03-22 / 2026-03-29 when
multi-Titan logic was rolled out, undetected for 30 days. T1 had been
emitting 8000+ `[T2:...]` mislabeled log lines per ~2 hours.

Canonical titan-id source is `data/titan_identity.json`; in worker code the
loaded value is bound to a local like `_TID` or `_titan_identity["titan_id"]`.
Hardcoded `[T1:...]/[T2:...]/[T3:...]` prefixes will be wrong on at least
two of the three Titans.

This test scans every `.py` file under `titan_plugin/` and `scripts/`,
parses with AST, and fails if any logger call has a string-literal first
argument starting with `[T1:`, `[T2:`, or `[T3:`.

Allow-list: a call may be excluded by appending `# noqa: log-prefix` on
the same source line as the literal — use only for genuinely titan-specific
diagnostic logs (rare).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = [PROJECT_ROOT / "titan_plugin", PROJECT_ROOT / "scripts"]
LOGGER_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}
FORBIDDEN_PREFIXES = ("[T1:", "[T2:", "[T3:")
NOQA_MARKER = "noqa: log-prefix"


def _iter_py_files():
    for root in SCAN_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            # Skip vendored / generated / cache
            if any(part in {"__pycache__", "test_env", ".venv"} for part in p.parts):
                continue
            yield p


def _is_logger_call(node: ast.Call) -> bool:
    """True if node looks like `<something>.info(...)` etc."""
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr in LOGGER_METHODS


def _first_string_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    arg0 = node.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value
    return None


def _line_has_noqa(file_lines: list[str], lineno: int) -> bool:
    if lineno <= 0 or lineno > len(file_lines):
        return False
    return NOQA_MARKER in file_lines[lineno - 1]


def _scan_file(path: pathlib.Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    file_lines = source.splitlines()
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_logger_call(node):
            continue
        msg = _first_string_arg(node)
        if msg is None or not msg.startswith(FORBIDDEN_PREFIXES):
            continue
        if _line_has_noqa(file_lines, node.lineno):
            continue
        prefix = msg.split("]")[0] + "]" if "]" in msg else msg[:8]
        rel = path.relative_to(PROJECT_ROOT)
        findings.append(f"{rel}:{node.lineno}: hardcoded log prefix {prefix!r}")
    return findings


def test_no_hardcoded_titan_log_prefixes() -> None:
    all_findings: list[str] = []
    for py in _iter_py_files():
        all_findings.extend(_scan_file(py))
    if all_findings:
        msg = (
            "Found logger calls with hardcoded `[TX:...]` titan-id prefixes.\n"
            "Replace with the loaded titan-id (e.g. `_TID = "
            "_titan_identity.get('titan_id', 'T1')` then `[%s:...]`, _TID, ...).\n"
            "If genuinely intentional, append `# noqa: log-prefix` on the same line.\n\n"
            + "\n".join(all_findings)
        )
        pytest.fail(msg)


if __name__ == "__main__":
    findings: list[str] = []
    for py in _iter_py_files():
        findings.extend(_scan_file(py))
    if findings:
        print("VIOLATIONS:")
        for f in findings:
            print("  " + f)
        raise SystemExit(1)
    print("OK: no hardcoded [TX:...] titan log prefixes")
