#!/usr/bin/env python3
"""roadmap_collapse_old_status.py — keep top N Current Status blocks; convert older to History pointers.

Per session_close_protocol.md Step 2a + 2026-05-17 optimization #5.

Problem: ROADMAP.md accumulates one ~40-line "## Current Status — <date>"
block per session, stacked chronologically (newest first). After 3+ sessions
the older blocks are redundant with the bottom-of-file History entries (each
session writes both).

Fix: keep top KEEP_N (default 3) Current Status blocks intact. Convert older
ones to a single-line pointer:

    ## Current Status — 2026-05-12 night-late (✅ archived — see [History entry](#history))

This cuts ROADMAP.md from ~500 lines to ~200 without losing any info (the
detailed History entry at bottom of file is the canonical record).

Idempotent: re-running won't re-collapse already-collapsed blocks.

Usage:
    python scripts/roadmap_collapse_old_status.py            # dry-run + diff
    python scripts/roadmap_collapse_old_status.py --apply    # write changes
    python scripts/roadmap_collapse_old_status.py --keep 5   # keep top 5 (default 3)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROADMAP = Path(__file__).resolve().parent.parent / "titan-docs" / "ROADMAP.md"

CURRENT_STATUS_RE = re.compile(r"^## Current Status — (.+?)$", re.MULTILINE)
ALREADY_COLLAPSED_RE = re.compile(
    r"^## Current Status — .+? \(✅ archived — see \[History entry\]\(#history\)\)\s*$"
)


def collapse(text: str, keep_n: int = 3) -> tuple[str, int, int]:
    """Returns (new_text, num_kept, num_collapsed)."""
    lines = text.split("\n")

    # Find indices of all Current Status block headers.
    status_starts: list[int] = []
    for i, line in enumerate(lines):
        if CURRENT_STATUS_RE.match(line):
            status_starts.append(i)

    if len(status_starts) <= keep_n:
        return text, len(status_starts), 0

    # Indices of blocks to collapse (everything past the first keep_n).
    to_collapse = status_starts[keep_n:]

    # For each block-to-collapse, find its end (next ## header OR EOF).
    out_lines: list[str] = []
    skip_until: int | None = None
    collapsed_count = 0

    block_end_idx: dict[int, int] = {}
    all_h2 = [i for i, l in enumerate(lines) if l.startswith("## ")]
    for start in to_collapse:
        # Next ## after `start`.
        next_h2 = next((j for j in all_h2 if j > start), len(lines))
        block_end_idx[start] = next_h2

    for i, line in enumerate(lines):
        if skip_until is not None and i < skip_until:
            continue
        skip_until = None

        if i in to_collapse:
            # Already collapsed? Keep as-is.
            if ALREADY_COLLAPSED_RE.match(line):
                out_lines.append(line)
                continue
            # Extract the date suffix from the header.
            m = CURRENT_STATUS_RE.match(line)
            if not m:
                out_lines.append(line)
                continue
            date_suffix = m.group(1)
            # Strip the "(✅ ...)" parenthetical if present — keep just
            # the date for the collapsed pointer.
            date_only = re.sub(r"\s*\(.+\)\s*$", "", date_suffix).strip()
            out_lines.append(
                f"## Current Status — {date_only} (✅ archived — see [History entry](#history))"
            )
            out_lines.append("")
            collapsed_count += 1
            skip_until = block_end_idx[i]
        else:
            out_lines.append(line)

    return "\n".join(out_lines), keep_n, collapsed_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default: dry-run + print diff summary)")
    ap.add_argument("--keep", type=int, default=3,
                    help="Number of top Current Status blocks to keep (default: 3)")
    ap.add_argument("--file", default=str(ROADMAP),
                    help="Path to ROADMAP.md (default: titan-docs/ROADMAP.md)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    text = path.read_text()
    new_text, kept, collapsed = collapse(text, keep_n=args.keep)

    if collapsed == 0:
        print(f"  No collapse needed — {kept} Current Status block(s) found (≤ keep={args.keep}).")
        return 0

    old_lines = len(text.split("\n"))
    new_lines = len(new_text.split("\n"))
    delta = old_lines - new_lines

    print(f"  Found {kept + collapsed} Current Status blocks:")
    print(f"    Kept (newest): {kept}")
    print(f"    Collapsed to one-line pointer: {collapsed}")
    print(f"  Line delta: {old_lines} → {new_lines} (−{delta} lines)")

    if args.apply:
        path.write_text(new_text)
        print(f"  ✓ Wrote {path}")
    else:
        print(f"  Dry-run — pass --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
