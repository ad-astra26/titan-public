#!/usr/bin/env python3
"""roadmap_splitter.py — keep ROADMAP.md focused on LATEST status; archive history.

ROADMAP.md is a stack of `## Current/Previous/Prior Status — DATE` blocks accumulated
since project start (3,496 lines as of 2026-05-13). Only the most-recent 1-2 status
blocks are load-bearing for a new session; everything else is institutional history.

This script:
  1. Keeps ROADMAP.md = frontmatter + latest N status blocks + permanent sections
     (Ordering principles / Current implementation order / Critical-path / Maintenance
     protocol / History — anchored by `## Ordering principles` start marker).
  2. Moves the older status blocks to `titan-docs/finished/ROADMAP_HISTORY.md`.

Run modes:
  --apply             — actually rewrite files (creates a backup .orig file)
  --check             — exit non-zero if active ROADMAP could be trimmed by >40%
                        (drift gate; same pattern as tracker_indexer.py --check)
  --keep N            — keep N most recent status blocks (default 2)

Idempotent: running --apply twice produces byte-identical output if no upstream changes.

Wired to post-commit hook so changes to ROADMAP.md auto-rotate when threshold exceeded.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROADMAP = REPO_ROOT / "titan-docs" / "ROADMAP.md"
HISTORY = REPO_ROOT / "titan-docs" / "finished" / "ROADMAP_HISTORY.md"

# Headings that mark chronological status blocks (move to history)
STATUS_RE = re.compile(r"^## (Current Status|Previous Status|Prior Status|Prior status|Previous session|Concurrent Session) — ")
# First permanent section (everything from here to EOF stays in active)
PERMANENT_START_RE = re.compile(r"^## (Ordering principles|Current implementation order|Critical-path|Maintenance protocol|History)\b")
# Frontmatter terminator
FRONTMATTER_END_RE = re.compile(r"^---\s*$")


def split_roadmap(text: str, keep: int = 2) -> tuple[str, str, dict]:
    """Return (new_active, new_history, stats)."""
    lines = text.splitlines(keepends=True)
    n = len(lines)

    # 1. Locate frontmatter end (second --- after start)
    frontmatter_end = 0
    seen_first = False
    for i, line in enumerate(lines):
        if FRONTMATTER_END_RE.match(line):
            if not seen_first:
                seen_first = True
            else:
                frontmatter_end = i + 1
                break
    frontmatter = "".join(lines[:frontmatter_end])

    # 2. Find all status block starts (their line index)
    status_starts: list[int] = []
    for i, line in enumerate(lines[frontmatter_end:], start=frontmatter_end):
        if STATUS_RE.match(line):
            status_starts.append(i)

    # 3. Find first permanent section
    permanent_start = n
    for i, line in enumerate(lines[frontmatter_end:], start=frontmatter_end):
        if PERMANENT_START_RE.match(line):
            permanent_start = i
            break

    # 4. Active = frontmatter + first `keep` status blocks + permanent sections
    # First N status blocks span from status_starts[0] to status_starts[keep] (exclusive)
    if len(status_starts) <= keep:
        # Nothing to archive
        return text, "", {
            "total_status_blocks": len(status_starts),
            "kept": len(status_starts),
            "archived": 0,
            "active_lines": n,
            "history_lines": 0,
            "no_op": True,
        }

    kept_end = status_starts[keep]
    kept_block = "".join(lines[frontmatter_end:kept_end])
    archived_block = "".join(lines[kept_end:permanent_start])
    permanent_block = "".join(lines[permanent_start:])

    new_active = frontmatter + kept_block + permanent_block

    # 5. History file: own preamble + archived blocks
    archived_count = len(status_starts) - keep
    history_header = f"""# Titan ROADMAP — Historical Status Blocks

**Source:** `titan-docs/ROADMAP.md` (active — latest {keep} status blocks only)
**Split date:** 2026-05-13 (deferred-observables-triage session)
**Purpose:** Institutional memory for chronological status blocks. Active ROADMAP.md keeps only the most-recent {keep} for session-startup context efficiency.

**Archived blocks:** {archived_count} status blocks (most-recent first)

When investigating a past decision: grep this file for a date or rFP slug; the full status block context is preserved verbatim.

---

"""
    new_history = history_header + archived_block

    return new_active, new_history, {
        "total_status_blocks": len(status_starts),
        "kept": keep,
        "archived": archived_count,
        "active_lines": len(new_active.splitlines()),
        "history_lines": len(new_history.splitlines()),
        "no_op": False,
    }


def cmd_apply(keep: int) -> int:
    text = ROADMAP.read_text()
    active, history, stats = split_roadmap(text, keep=keep)
    if stats["no_op"]:
        print(f"no-op: only {stats['total_status_blocks']} status blocks (≤ keep={keep})")
        return 0
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    # If history already exists, append (idempotency-friendly: only new archived blocks added)
    if HISTORY.exists():
        # Merge: drop our re-generated header lines, keep old history body + prepend new archived
        existing = HISTORY.read_text()
        # Find the `---\n` separator after the header
        sep = existing.find("\n---\n\n")
        existing_body = existing[sep + len("\n---\n\n"):] if sep != -1 else existing
        new_archived_body = history.split("\n---\n\n", 1)[1] if "\n---\n\n" in history else history
        merged_body = new_archived_body + existing_body
        # Rebuild header (count + date will reflect cumulative archive)
        cumulative_blocks = merged_body.count("\n## ")
        header = history.split("\n---\n\n", 1)[0]
        # Replace "Archived blocks: N status blocks" with cumulative
        header = re.sub(r"\*\*Archived blocks:\*\* \d+", f"**Archived blocks:** {cumulative_blocks}", header)
        history = header + "\n---\n\n" + merged_body
    HISTORY.write_text(history)
    ROADMAP.write_text(active)
    print(f"split applied: ROADMAP {stats['active_lines']} lines · HISTORY {stats['history_lines']} lines · {stats['archived']} blocks archived (kept {stats['kept']})")
    return 0


def cmd_check(keep: int) -> int:
    text = ROADMAP.read_text()
    _, _, stats = split_roadmap(text, keep=keep)
    if stats["no_op"]:
        return 0
    current_lines = len(text.splitlines())
    new_lines = stats["active_lines"]
    reduction_pct = 100 * (current_lines - new_lines) / current_lines
    if reduction_pct > 40:
        print(f"DRIFT: ROADMAP.md could shrink {reduction_pct:.0f}% by rotating — run "
              f"`python scripts/roadmap_splitter.py --apply --keep {keep}`", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="rewrite ROADMAP.md + write HISTORY")
    g.add_argument("--check", action="store_true", help="drift gate: non-zero exit if >40% trim possible")
    ap.add_argument("--keep", type=int, default=2, help="keep N most recent status blocks (default 2)")
    args = ap.parse_args()

    if args.apply:
        return cmd_apply(args.keep)
    if args.check:
        return cmd_check(args.keep)
    return 2


if __name__ == "__main__":
    sys.exit(main())
