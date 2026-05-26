#!/usr/bin/env python3
"""Split BUGS.md into active + graveyard (physical move).

Mirror of `split_observables_graveyard.py` for BUGS.md.

Reads `titan-docs/BUGS.md`, identifies closed entries (table rows with
`~~BUG-{slug}~~` strikethrough OR detail-block headers tagged with closure
keywords), moves them to `titan-docs/BUGS_graveyard.md`, and leaves
only active entries in BUGS.md.

Note: this DIFFERS from the existing `emit_bugs_graveyard.py` script —
that one emits a SUMMARY INDEX (slugs only) of closed bugs, keeping the
full bodies in BUGS.md. This script PHYSICALLY MOVES the bodies out so
BUGS.md becomes a lean view of active bugs only.

Usage:
    python scripts/split_bugs_graveyard.py            # do the split
    python scripts/split_bugs_graveyard.py --dry-run  # report only

Idempotent: merges with existing graveyard, preserves prior bodies.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "titan-docs" / "BUGS.md"
GRAVEYARD = REPO_ROOT / "titan-docs" / "BUGS_graveyard.md"

CLOSED_ROW_RE = re.compile(r"^\| ~~BUG-([A-Za-z0-9_\-]+)~~ \|")
ACTIVE_ROW_RE = re.compile(r"^\| (?:\[BUG-|BUG-)([A-Za-z0-9_\-]+)")
# Detail-block header: ### BUG-X — ... OR ### ~~BUG-X~~ — ...
HEADER_RE = re.compile(r"^### ~?~?BUG-([A-Za-z0-9_\-]+)~?~?(?:\s+—|\s*$)")
CLOSED_HEADER_MARKERS = (
    "✅ FIXED", "✅ PASS", "✅ PASSED", "✅ CLOSED", "✅ RESOLVED",
    "🔁 SUPERSEDED", "❌ SUPERSEDED", "SUPERSEDED",
    "WONTFIX", "ARCHIVED",
)


def parse(text: str):
    lines = text.splitlines(keepends=False)
    closed_slugs: set[str] = set()
    active_slugs: set[str] = set()
    table_closed_rows: list[int] = []
    for i, ln in enumerate(lines):
        m = CLOSED_ROW_RE.match(ln)
        if m:
            closed_slugs.add(m.group(1))
            table_closed_rows.append(i)
            continue
        m = ACTIVE_ROW_RE.match(ln)
        if m:
            # Skip the schema example row (### BUG-slug — placeholder)
            if "slug" in m.group(1).lower() and "—" not in ln:
                continue
            active_slugs.add(m.group(1))

    detail_ranges: dict[str, tuple[int, int]] = {}
    headers: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        m = HEADER_RE.match(ln)
        if m:
            slug = m.group(1)
            # Skip schema placeholder
            if slug == "slug":
                continue
            headers.append((i, slug))

    headers.append((len(lines), "_END_"))
    for idx in range(len(headers) - 1):
        start, slug = headers[idx]
        end = headers[idx + 1][0]
        while end > start + 1 and lines[end - 1].strip() == "":
            end -= 1
        detail_ranges[slug] = (start, end)

    return lines, closed_slugs, active_slugs, table_closed_rows, detail_ranges


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    text = SOURCE.read_text()
    lines, closed_slugs, active_slugs, closed_row_lines, detail_ranges = parse(text)

    # Status patterns within body metadata block (first ~12 lines after header)
    BODY_CLOSED_STATUS_RE = re.compile(
        r"\*\*Status:\*\*\s*(?:🟠|🟡|🟢|🔴|⚪|✅|❌|🔁)?\s*"
        r"(FIXED|PASSED|RESOLVED|CLOSED|SUPERSEDED|WONTFIX|ARCHIVED|RETIRED)"
    )

    orphan_details = sorted(set(detail_ranges.keys()) - closed_slugs - active_slugs - {"_END_"})
    orphan_closed: list[str] = []
    for slug in orphan_details:
        rng = detail_ranges.get(slug)
        if not rng:
            continue
        header = lines[rng[0]]
        # 1. Treat as closed if header itself contains slug in strikethrough form
        # (### ~~BUG-X~~ ...) OR if a closure marker word appears in header.
        is_closed = (f"~~BUG-{slug}~~" in header) or any(marker in header for marker in CLOSED_HEADER_MARKERS)
        # 2. If header didn't reveal closure, scan the first 12 body lines for
        # `**Status:** FIXED/PASSED/...` — many entries have closure only in body.
        if not is_closed:
            body_head = "\n".join(lines[rng[0] + 1: rng[0] + 13])
            if BODY_CLOSED_STATUS_RE.search(body_head):
                is_closed = True
        if is_closed:
            closed_slugs.add(slug)
            orphan_closed.append(slug)
    if orphan_closed:
        print(f"orphan CLOSED detail blocks (no table row, header marker): {len(orphan_closed)} entries", file=sys.stderr)
    orphan_remaining = sorted(set(orphan_details) - set(orphan_closed))
    if orphan_remaining:
        print(f"orphan detail blocks (no marker — STAY in source): {len(orphan_remaining)} entries", file=sys.stderr)

    closed_detail_lines: set[int] = set()
    for slug in closed_slugs:
        rng = detail_ranges.get(slug)
        if rng:
            for i in range(rng[0], rng[1]):
                closed_detail_lines.add(i)
    rows_to_remove: set[int] = set(closed_row_lines)

    new_source_lines = [ln for i, ln in enumerate(lines) if i not in closed_detail_lines and i not in rows_to_remove]

    closed_table_rows = [lines[i] for i in closed_row_lines]
    closed_detail_blocks: list[tuple[str, list[str]]] = []
    for slug in sorted(closed_slugs):
        rng = detail_ranges.get(slug)
        if rng:
            block = lines[rng[0]:rng[1]]
            closed_detail_blocks.append((slug, block))

    grave_lines: list[str] = []
    grave_lines.append("# Titan BUGS Graveyard")
    grave_lines.append("")
    grave_lines.append("> Closed bugs archive. Source-of-truth for active bugs is `titan-docs/BUGS.md`.")
    grave_lines.append(f"> **Closed entries:** {len(closed_slugs)}  ·  Generated by `scripts/split_bugs_graveyard.py`.")
    grave_lines.append("")
    grave_lines.append("This file preserves full historical context (entry bodies + closure notes).")
    grave_lines.append("Future sessions can grep here for prior fixes / patterns without scrolling the active list.")
    grave_lines.append("")
    grave_lines.append("---")
    grave_lines.append("")
    grave_lines.append("## 🪦 Closed entries — index")
    grave_lines.append("")
    grave_lines.append("| Slug | Severity | Status | Closure Note | Age (d) | Source |")
    grave_lines.append("|---|---|---|---|---|---|")
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

    # Idempotent merge with existing graveyard
    if GRAVEYARD.exists():
        prev = GRAVEYARD.read_text().splitlines()
        prev_table: dict[str, str] = {}
        prev_bodies: dict[str, list[str]] = {}
        for ln in prev:
            m = re.match(r"^\| ~~BUG-([A-Za-z0-9_\-]+)~~ \|", ln)
            if m:
                prev_table[m.group(1)] = ln
        prev_headers: list[tuple[int, str]] = []
        for i, ln in enumerate(prev):
            m = re.match(r"^### BUG-([A-Za-z0-9_\-]+)(?:\s+—|\s*$)", ln)
            if m and m.group(1) != "slug":
                prev_headers.append((i, m.group(1)))
        prev_headers.append((len(prev), "_END_"))
        for j in range(len(prev_headers) - 1):
            start, slug = prev_headers[j]
            end = prev_headers[j + 1][0]
            while end > start + 1 and prev[end - 1].strip() == "":
                end -= 1
            prev_bodies[slug] = prev[start:end]

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
            insert_table_at = grave_lines.index("## 🪦 Closed entries — full bodies") - 2
            for row in added_rows:
                grave_lines.insert(insert_table_at, row)
                insert_table_at += 1
            for slug, body in added_bodies:
                grave_lines.extend(body)
                grave_lines.append("")
                grave_lines.append("---")
                grave_lines.append("")
            for i, ln in enumerate(grave_lines):
                if ln.startswith("> **Closed entries:**"):
                    grave_lines[i] = f"> **Closed entries:** {len(closed_slugs) + len(added_rows)}  ·  Generated by `scripts/split_bugs_graveyard.py`."
                    break

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
