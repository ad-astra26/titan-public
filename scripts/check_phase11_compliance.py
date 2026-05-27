#!/usr/bin/env python3
"""Phase 11 §11.I.2 compliance guard — locked D2 enforcement.

Per SPEC D-SPEC-141 / v1.65.0 §11.I.2 + locked D2 + locked-D2-derived
`feedback_no_shim_old_path_must_be_deleted`:

  1. Every worker module in `titan_hcl/modules/*_worker.py` must construct
     a `ModuleStateWriter` and call `write_state(...)` at least once
     (the SHM-slot publication contract that replaces MODULE_READY).

  2. No worker module may emit `MODULE_READY` on the bus. Per locked D2
     ("this needs to have only one proper state that all modules will
     respect. not two states") — the old contract is DELETED, not
     dual-pathed.

This script is a build-time / CI-time guard. Exit status:
  0  — fleet 100% compliant
  1  — one or more workers violate one of the rules above

Usage:
  python scripts/check_phase11_compliance.py
  python scripts/check_phase11_compliance.py --strict   (also flag stragglers)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "titan_hcl" / "modules"

# Function-call patterns that count as a "live MODULE_READY emit".
# Comment / docstring text matches like "MODULE_READY retired" are ignored
# (those are documentation of the retirement, which is acceptable).
EMIT_PATTERNS = [
    re.compile(r"send_queue.*MODULE_READY"),
    re.compile(r"_send\s*\(.*MODULE_READY"),
    re.compile(r"_send_msg\s*\(.*MODULE_READY"),
    re.compile(r"\.publish\s*\(.*MODULE_READY"),
    re.compile(r"put\s*\(.*MODULE_READY"),
    re.compile(r"put_nowait\s*\(.*MODULE_READY"),
    re.compile(r"make_msg\s*\(.*MODULE_READY"),
    re.compile(r"bus\.MODULE_READY.*=|=.*bus\.MODULE_READY"),
]

# Acceptable forms for the SHM-slot writer:
# - `ModuleStateWriter(...)` constructor anywhere
# - `state_writer = ModuleStateWriter(`  or similar
WRITER_PATTERN = re.compile(r"ModuleStateWriter\s*\(")
STATE_PATTERN = re.compile(r"\.write_state\s*\(")


def line_is_code(line: str) -> bool:
    """True if `line` is plausibly executable code, not a comment or
    docstring-internal text. Conservative: a leading `#`, or being
    inside a triple-quoted block, marks the line as non-code. Since the
    docstring detection here is single-pass and naive, we only flag
    lines that ALSO look like a function call to avoid false positives
    from inline comments next to executable code."""
    stripped = line.strip()
    if stripped.startswith("#"):
        return False
    # Inside a docstring block at this line? We don't track that here;
    # we rely on the EMIT_PATTERNS being specific enough that docstring
    # text rarely matches (the patterns include `(` and other punct).
    return True


def find_live_module_ready_emits(file_path: Path) -> list[tuple[int, str]]:
    """Return [(line_number, line_text), ...] of suspected MODULE_READY emits."""
    matches: list[tuple[int, str]] = []
    in_docstring = False
    quote_char: str | None = None
    with file_path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            # Track triple-quoted docstring blocks (single-line patterns
            # like `"""docstring"""` open and close on same line).
            for q in ('"""', "'''"):
                if q in stripped:
                    occurrences = stripped.count(q)
                    if occurrences % 2 == 1:
                        if not in_docstring:
                            in_docstring = True
                            quote_char = q
                        elif quote_char == q:
                            in_docstring = False
                            quote_char = None
            if in_docstring:
                continue
            if not line_is_code(line):
                continue
            for pat in EMIT_PATTERNS:
                if pat.search(line):
                    matches.append((lineno, line))
                    break
    return matches


def has_module_state_writer(file_path: Path) -> bool:
    with file_path.open() as f:
        text = f.read()
    return bool(WRITER_PATTERN.search(text)) and bool(STATE_PATTERN.search(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true",
        help="Also flag minor stragglers (e.g. import lines that reference "
             "MODULE_READY without emitting).",
    )
    args = parser.parse_args()

    worker_files = sorted(WORKER_DIR.glob("*_worker.py"))
    if not worker_files:
        print(f"!! no worker files under {WORKER_DIR}", file=sys.stderr)
        return 1

    violations: list[str] = []
    missing_writer: list[str] = []

    for wf in worker_files:
        rel = wf.relative_to(REPO_ROOT)

        # Rule 1: ModuleStateWriter coverage
        if not has_module_state_writer(wf):
            missing_writer.append(str(rel))

        # Rule 2: zero live MODULE_READY emit
        emits = find_live_module_ready_emits(wf)
        if emits:
            for (lineno, line) in emits:
                violations.append(f"{rel}:{lineno}  {line.strip()}")

    # Report
    print("# Phase 11 §11.I.2 compliance audit")
    print(f"#   scanned: {len(worker_files)} worker files in {WORKER_DIR.relative_to(REPO_ROOT)}")
    print()

    if missing_writer:
        print(f"!! Rule 1 violations — {len(missing_writer)} worker(s) missing ModuleStateWriter:")
        for path in missing_writer:
            print(f"   {path}")
        print()

    if violations:
        print(f"!! Rule 2 violations — {len(violations)} live MODULE_READY emit(s) found:")
        for v in violations:
            print(f"   {v}")
        print()

    if not missing_writer and not violations:
        print("OK — fleet 100% Phase 11 compliant.")
        return 0

    print(f"FAIL — {len(missing_writer)} writer-missing + {len(violations)} module-ready-emit "
          f"violation(s). Per locked D2 + feedback_no_shim_old_path_must_be_deleted, "
          f"every worker MUST own its SHM slot and MUST NOT emit MODULE_READY.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
