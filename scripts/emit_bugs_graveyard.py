#!/usr/bin/env python3
"""DEPRECATED 2026-05-26: prefer `split_bugs_graveyard.py` (physically moves
closed bodies out of BUGS.md). This script kept for the summary-index-only
workflow; running it will OVERWRITE the physical-split output with a
summary-only graveyard. Do not run unless you want that behavior.

Emit `titan-docs/BUGS_graveyard.md` — searchable archive of all closed bugs.

Reuses `tracker_indexer.parse_tracker` so the graveyard stays consistent with
the canonical BUGS.md (the source of truth — full entry bodies live there).
The graveyard is a SUMMARY INDEX of closed bugs sorted by closure date
(most recent first), with anchor links back into BUGS.md for full detail.

Usage:
    python scripts/emit_bugs_graveyard.py                # writes BUGS_graveyard.md
    python scripts/emit_bugs_graveyard.py --check        # exit 1 if stale

Designed to live alongside `tracker_indexer.py --emit-index BUGS.md`; both can
run in CI / pre-commit. The graveyard is purely additive (BUGS.md remains
the canonical document — graveyard just gives a search-friendly closed-bug view).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tracker_indexer import find_config, parse_tracker  # noqa: E402

CLOSED_STATUSES = {
    "FIXED", "PASSED", "RESOLVED", "CLOSED", "SHIPPED",
    "SUPERSEDED", "WONTFIX", "ARCHIVED",
    "MIGRATED", "MIGRATED→BUGS", "MIGRATED→OBS",
    "REALIZED", "RESOLVED-UNNECESSARY",
}

GRAVEYARD_PATH = REPO_ROOT / "titan-docs" / "BUGS_graveyard.md"
BUGS_SOURCE = REPO_ROOT / "titan-docs" / "BUGS.md"


def _summary_for(entry, body_lines: list[str]) -> str:
    """One-line summary for the graveyard row. Prefers the title from the
    `### ` header; falls back to a trimmed first body paragraph."""
    title = (entry.title or "").strip()
    if title and len(title) > 8:
        return title[:240].replace("|", "\\|")
    # Fallback: first non-blank line after the header
    for i in range(entry.line, min(entry.line + 8, len(body_lines))):
        ln = body_lines[i].strip()
        if ln and not ln.startswith("###") and not ln.startswith(">"):
            return ln.lstrip("-* ").lstrip()[:240].replace("|", "\\|")
    return "(no summary)"


def render_graveyard(entries: list, body_lines: list[str]) -> str:
    closed = [e for e in entries if e.status in CLOSED_STATUSES]
    closed.sort(key=lambda e: (e.date or "0000-00-00", e.line), reverse=True)

    sev_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for e in closed:
        sev_counts[e.severity or "(none)"] = sev_counts.get(e.severity or "(none)", 0) + 1
        status_counts[e.status] = status_counts.get(e.status, 0) + 1

    out: list[str] = []
    out.append("# Titan BUGS Graveyard (auto-generated)")
    out.append("")
    out.append("> **Auto-generated** from `titan-docs/BUGS.md` by `scripts/emit_bugs_graveyard.py`.")
    out.append("> **DO NOT EDIT** — regenerate via `python scripts/emit_bugs_graveyard.py`.")
    out.append("")
    out.append("Searchable archive of all closed bugs. The canonical entry body for each row")
    out.append("lives in `titan-docs/BUGS.md` (follow the line-link); this file is a summary")
    out.append("index sorted by closure date (most recent first) so future sessions can search")
    out.append("for prior resolutions / patterns without scrolling through the full registry.")
    out.append("")
    out.append(f"**Closed bugs:** {len(closed)}  ·  **Source:** [`BUGS.md`](BUGS.md)")
    out.append("")
    if sev_counts:
        out.append("**By severity:** " + " · ".join(
            f"{k}: {v}" for k, v in sorted(sev_counts.items(),
                                            key=lambda kv: (-kv[1], kv[0]))))
        out.append("")
    if status_counts:
        out.append("**By closure type:** " + " · ".join(
            f"{k}: {v}" for k, v in sorted(status_counts.items(),
                                            key=lambda kv: (-kv[1], kv[0]))))
        out.append("")
    out.append("---")
    out.append("")
    out.append("## 🪦 Closed bugs — sorted by closure date (newest first)")
    out.append("")
    out.append("| Slug | Severity | Status | Closed | Line | Summary |")
    out.append("|---|---|---|---|---|---|")
    for e in closed:
        sev = e.severity or "—"
        date = e.date or "—"
        summary = _summary_for(e, body_lines)
        out.append(
            f"| `{e.slug}` | {sev} | {e.status} | {date} | "
            f"[L{e.line}](BUGS.md#L{e.line}) | {summary} |"
        )
    out.append("")
    out.append("---")
    out.append("")
    out.append("_Search tip: grep this file by slug substring, severity, or closure date band._")
    out.append("_Full entry body (root cause, fix, evidence) lives in `BUGS.md` at the linked line._")
    out.append("")
    out.append(f"_Regenerate: `python scripts/emit_bugs_graveyard.py`._")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if BUGS_graveyard.md is stale vs BUGS.md")
    args = ap.parse_args()

    cfg = find_config(str(BUGS_SOURCE))
    entries = parse_tracker(cfg)
    body_lines = BUGS_SOURCE.read_text().splitlines()
    new_content = render_graveyard(entries, body_lines)

    if args.check:
        existing = GRAVEYARD_PATH.read_text() if GRAVEYARD_PATH.exists() else ""
        if existing != new_content:
            print(f"DRIFT: {GRAVEYARD_PATH.relative_to(REPO_ROOT)} is stale. "
                  f"Regenerate: python scripts/emit_bugs_graveyard.py",
                  file=sys.stderr)
            return 1
        print(f"OK: {GRAVEYARD_PATH.relative_to(REPO_ROOT)} up to date")
        return 0

    GRAVEYARD_PATH.write_text(new_content)
    closed_count = sum(1 for e in entries if e.status in CLOSED_STATUSES)
    print(f"wrote {GRAVEYARD_PATH.relative_to(REPO_ROOT)} ({closed_count} closed bugs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
