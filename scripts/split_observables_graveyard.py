#!/usr/bin/env python3
"""Split OBSERVABLES.md into active + graveyard.

Reads `titan-docs/OBSERVABLES.md`, identifies closed entries (table rows with
`~~OBS-{slug}~~` strikethrough OR detail-block headers tagged with closure
keywords), moves them to `titan-docs/OBSERVABLES_graveyard.md`, and leaves
only active entries in OBSERVABLES.md.

Closed entries (table row + detail body) are preserved verbatim in the
graveyard file so historical context isn't lost. The graveyard's table is
sorted by closure status, then by original entry order.

Usage:
    python scripts/split_observables_graveyard.py            # do the split
    python scripts/split_observables_graveyard.py --dry-run  # report only

Idempotent: running twice is a no-op (graveyard already extracted).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "titan-docs" / "OBSERVABLES.md"
GRAVEYARD = REPO_ROOT / "titan-docs" / "OBSERVABLES_graveyard.md"

# Match table rows where slug is strikethrough'd  → closed
CLOSED_ROW_RE = re.compile(r"^\| ~~OBS-([A-Za-z0-9_\-]+)~~ \|")
# Match active rows
ACTIVE_ROW_RE = re.compile(r"^\| \[OBS-([A-Za-z0-9_\-]+)\]\(#")
# Header for detail block
HEADER_RE = re.compile(r"^### OBS-([A-Za-z0-9_\-]+)(?:\s+—|\s*$)")
# Header-level closure markers (orphan detail blocks without table rows)
CLOSED_HEADER_MARKERS = (
    "✅ PASS", "✅ PASSED", "✅ CLOSED",
    "🔁 SUPERSEDED", "❌ SUPERSEDED", "SUPERSEDED",
    "WONTFIX", "ARCHIVED",
)


def parse(text: str):
    lines = text.splitlines(keepends=False)
    closed_slugs: set[str] = set()
    active_slugs: set[str] = set()
    table_closed_rows: list[int] = []  # line indices
    for i, ln in enumerate(lines):
        m = CLOSED_ROW_RE.match(ln)
        if m:
            closed_slugs.add(m.group(1))
            table_closed_rows.append(i)
            continue
        m = ACTIVE_ROW_RE.match(ln)
        if m:
            active_slugs.add(m.group(1))

    # Find detail-block ranges: from ### header to line BEFORE next ### or ## or end
    detail_ranges: dict[str, tuple[int, int]] = {}  # slug -> (start, end_exclusive)
    headers: list[tuple[int, str]] = []  # (line_idx, slug)
    for i, ln in enumerate(lines):
        m = HEADER_RE.match(ln)
        if m:
            headers.append((i, m.group(1)))

    # Add sentinel at end of file
    headers.append((len(lines), "_END_"))
    for idx in range(len(headers) - 1):
        start, slug = headers[idx]
        # End range = line before next section header (## or ###)
        end = headers[idx + 1][0]
        # Trim trailing blank lines from the block (keep one separator)
        while end > start + 1 and lines[end - 1].strip() == "":
            end -= 1
        detail_ranges[slug] = (start, end)

    return lines, closed_slugs, active_slugs, table_closed_rows, detail_ranges


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--check", action="store_true",
        help="Pre-commit mode: exit 1 if closed entries still live in "
             "OBSERVABLES.md (split is stale). Tells user to run the "
             "script manually + git add the result."
    )
    args = p.parse_args()

    text = SOURCE.read_text()
    lines, closed_slugs, active_slugs, closed_row_lines, detail_ranges = parse(text)

    # Sanity: report any slug that is in detail blocks but not in table (orphan)
    orphan_details = sorted(set(detail_ranges.keys()) - closed_slugs - active_slugs - {"_END_"})
    # Move orphan detail blocks whose header contains a closure marker
    orphan_closed: list[str] = []
    for slug in orphan_details:
        rng = detail_ranges.get(slug)
        if not rng:
            continue
        header = lines[rng[0]]
        if any(marker in header for marker in CLOSED_HEADER_MARKERS):
            closed_slugs.add(slug)
            orphan_closed.append(slug)
    if orphan_closed:
        print(f"orphan CLOSED detail blocks (no table row, header marker): {orphan_closed}", file=sys.stderr)
    orphan_remaining = sorted(set(orphan_details) - set(orphan_closed))
    if orphan_remaining:
        print(f"orphan detail blocks (no marker — STAY in source): {orphan_remaining}", file=sys.stderr)

    # Lines to remove from source (1-based after rebuild becomes 0-based deletes)
    closed_detail_lines: set[int] = set()
    for slug in closed_slugs:
        rng = detail_ranges.get(slug)
        if rng:
            for i in range(rng[0], rng[1]):
                closed_detail_lines.add(i)
    # Also remove closed table rows
    rows_to_remove: set[int] = set(closed_row_lines)

    # Build new source (active-only)
    new_source_lines = [ln for i, ln in enumerate(lines) if i not in closed_detail_lines and i not in rows_to_remove]

    # Build graveyard: header + summary table of closed (extracted from removed table rows) + detail bodies
    closed_table_rows = [lines[i] for i in closed_row_lines]
    closed_detail_blocks: list[tuple[str, list[str]]] = []
    for slug in sorted(closed_slugs):
        rng = detail_ranges.get(slug)
        if rng:
            block = lines[rng[0]:rng[1]]
            closed_detail_blocks.append((slug, block))

    grave_lines: list[str] = []
    grave_lines.append("# Titan OBSERVABLES Graveyard")
    grave_lines.append("")
    grave_lines.append("> Closed observables archive. Source-of-truth for active entries is `titan-docs/OBSERVABLES.md`.")
    grave_lines.append(f"> **Closed entries:** {len(closed_slugs)}  ·  Generated by `scripts/split_observables_graveyard.py`.")
    grave_lines.append("")
    grave_lines.append("This file preserves full historical context (entry bodies + closure notes).")
    grave_lines.append("Future sessions can grep here for prior PASSED / SUPERSEDED disposals.")
    grave_lines.append("")
    grave_lines.append("---")
    grave_lines.append("")
    grave_lines.append("## 🪦 Closed entries — index")
    grave_lines.append("")
    grave_lines.append("| ID | Priority | Status | Closure Note | Age (d) | Closed | Related |")
    grave_lines.append("|---|---|---|---|---|---|---|")
    grave_lines.extend(closed_table_rows)
    grave_lines.append("")
    grave_lines.append("---")
    grave_lines.append("")
    grave_lines.append("## 🪦 Closed entries — full bodies")
    grave_lines.append("")
    for slug, block in closed_detail_blocks:
        grave_lines.extend(block)
        grave_lines.append("")
        grave_lines.append("---")
        grave_lines.append("")

    # Idempotent merge with existing graveyard, if present.
    # If the existing graveyard has bodies for slugs that the current source
    # no longer contains (already moved in a prior run), preserve them.
    if GRAVEYARD.exists():
        prev = GRAVEYARD.read_text().splitlines()
        # Parse prior graveyard: extract slug -> (table_row, detail_block) — best effort
        prev_table: dict[str, str] = {}
        prev_bodies: dict[str, list[str]] = {}
        # Scan table rows
        for ln in prev:
            m = re.match(r"^\| ~~OBS-([A-Za-z0-9_\-]+)~~ \|", ln)
            if m:
                prev_table[m.group(1)] = ln
        # Scan detail-block bodies
        prev_headers: list[tuple[int, str]] = []
        for i, ln in enumerate(prev):
            # Match BOTH `### OBS-X —` and `### ~~OBS-X~~ —` (graveyard
            # bodies are stored with strikethrough'd slugs in header
            # for some entries).
            m = re.match(r"^### ~?~?OBS-([A-Za-z0-9_\-]+)~?~?(?:\s+—|\s*$)", ln)
            if m:
                prev_headers.append((i, m.group(1)))
        prev_headers.append((len(prev), "_END_"))
        for j in range(len(prev_headers) - 1):
            start, slug = prev_headers[j]
            end = prev_headers[j + 1][0]
            # Trim trailing blanks
            while end > start + 1 and prev[end - 1].strip() == "":
                end -= 1
            prev_bodies[slug] = prev[start:end]

        # Merge: prefer current-run content, but fill in slugs we've lost track of
        current_slugs = set(closed_slugs)
        added_rows: list[str] = []
        added_bodies: list[tuple[str, list[str]]] = []
        for slug, row in prev_table.items():
            if slug not in current_slugs:
                added_rows.append(row)
        for slug, body in prev_bodies.items():
            if slug not in current_slugs:
                added_bodies.append((slug, body))
        if added_rows or added_bodies:
            print(f"merging {len(added_rows)} prior table rows + {len(added_bodies)} prior bodies from existing graveyard", file=sys.stderr)
            # Insert added rows before "## 🪦 Closed entries — full bodies"
            insert_table_at = grave_lines.index("## 🪦 Closed entries — full bodies") - 2  # before the trailing blank+---
            for row in added_rows:
                grave_lines.insert(insert_table_at, row)
                insert_table_at += 1
            # Append added bodies at end
            for slug, body in added_bodies:
                grave_lines.extend(body)
                grave_lines.append("")
                grave_lines.append("---")
                grave_lines.append("")
            # Update count line
            # Union of table-row slugs + body-block slugs + current run.
            # Some closed entries have only a body (no table row) and
            # vice versa, so total-count = unique union of all sources.
            all_closed_slugs = set(closed_slugs) | set(prev_table.keys()) | set(prev_bodies.keys())
            for i, ln in enumerate(grave_lines):
                if ln.startswith("> **Closed entries:**"):
                    grave_lines[i] = f"> **Closed entries:** {len(all_closed_slugs)}  ·  Generated by `scripts/split_observables_graveyard.py`."
                    break

    if args.check:
        if len(closed_slugs) > 0:
            print(
                f"OBSERVABLES_graveyard split is STALE: {len(closed_slugs)} closed entries "
                f"still in OBSERVABLES.md (should be in OBSERVABLES_graveyard.md).",
                file=sys.stderr,
            )
            print(
                f"Fix: python scripts/split_observables_graveyard.py "
                f"&& git add titan-docs/OBSERVABLES.md titan-docs/OBSERVABLES_graveyard.md",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.dry_run:
        print(f"closed slugs: {len(closed_slugs)}")
        print(f"active slugs: {len(active_slugs)}")
        print(f"closed detail blocks: {len(closed_detail_blocks)}")
        print(f"lines to remove from source: {len(closed_detail_lines) + len(rows_to_remove)}")
        print(f"new source line count: {len(new_source_lines)} (was {len(lines)})")
        print(f"graveyard line count: {len(grave_lines)}")
        return 0

    SOURCE.write_text("\n".join(new_source_lines) + "\n")
    GRAVEYARD.write_text("\n".join(grave_lines) + "\n")
    print(f"wrote {SOURCE} ({len(new_source_lines)} lines)")
    print(f"wrote {GRAVEYARD} ({len(grave_lines)} lines)")
    print(f"moved {len(closed_slugs)} closed entries to graveyard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
