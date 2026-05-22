#!/usr/bin/env python3
"""tracker_indexer.py — auto-generate compact index files for trackers.

Pattern: per `scripts/spec_index.py`, the full tracker files (BUGS.md ~3.2K
lines, OBSERVABLES.md ~4.2K lines, DEFERRED_ITEMS.md ~850 lines after split)
burn ~80-100K tokens at session startup. This script emits a ~200-line index
per tracker that captures slug + status + severity/priority + open-date +
line number, so future sessions can grep the index then `Read offset=N` into
the body only when needed.

Modes:
  --emit-index <FILE>        — scan a tracker file, write its index file
  --check <FILE>             — exit non-zero if index file is stale
  --emit-all                 — regenerate every registered tracker index
  --check-all                — drift gate for all registered trackers

Wire to post-commit hook for auto-regen; wire to pre-commit for drift gate.

Output is deterministic + idempotent: same input → byte-identical output.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── Per-tracker config ────────────────────────────────────────────────────
# Slug prefix patterns + status markers vary slightly per tracker;
# keep config explicit rather than auto-detect to avoid silent misclassification.

@dataclass(frozen=True)
class TrackerConfig:
    name: str
    body_path: Path
    index_path: Path
    # Regex matching entry headers (### …)
    # We require the slug to be the first identifier on the line, before ' — '.
    slug_re: re.Pattern
    # Display title for the index file
    title: str

TRACKERS: dict[str, TrackerConfig] = {
    "bugs": TrackerConfig(
        name="bugs",
        body_path=REPO_ROOT / "titan-docs" / "BUGS.md",
        index_path=REPO_ROOT / "titan-docs" / "BUGS_index.md",
        slug_re=re.compile(r"^### (~~)?(?P<slug>(?:BUG-)?[A-Z][A-Z0-9_\-]+(?:-20\d{6})?)(~~)?\s*(?:—|–|-) "),
        title="Titan BUGS Index",
    ),
    "observables": TrackerConfig(
        name="observables",
        body_path=REPO_ROOT / "titan-docs" / "OBSERVABLES.md",
        index_path=REPO_ROOT / "titan-docs" / "OBSERVABLES_index.md",
        slug_re=re.compile(r"^### (~~)?(?P<slug>OBS-[a-zA-Z0-9_\-]+)(~~)?\s*(?:—|–|-) "),
        title="Titan OBSERVABLES Index",
    ),
    "deferred": TrackerConfig(
        name="deferred",
        body_path=REPO_ROOT / "titan-docs" / "DEFERRED_ITEMS.md",
        index_path=REPO_ROOT / "titan-docs" / "DEFERRED_index.md",
        slug_re=re.compile(r"^### (~~)?(?P<slug>[A-Z][A-Z0-9a-z _/\-]+?)(~~)?\s*(?:—|–|-) "),
        title="Titan DEFERRED Index",
    ),
}

# ─── Parsing ───────────────────────────────────────────────────────────────

@dataclass
class Entry:
    slug: str
    line: int
    header: str   # full header line (raw)
    status: str   # OPEN / FIXED / RESOLVED / PASSED / SUPERSEDED / DEFERRED / ...
    severity: str # CRITICAL / HIGH / MEDIUM / LOW / COSMETIC / "" (not all trackers)
    date: str     # YYYY-MM-DD if found, else ""
    title: str    # short title portion after the slug em-dash

# Status detection priority order
STATUS_PATTERNS = [
    ("✅ FIXED", "FIXED"),
    ("✅ PASSED", "PASSED"),
    ("✅ RESOLVED", "RESOLVED"),
    ("✅ CLOSED", "CLOSED"),
    ("✅ SHIPPED", "SHIPPED"),
    ("RESOLVED-UNNECESSARY", "RESOLVED-UNNECESSARY"),
    ("🔄 SUPERSEDED", "SUPERSEDED"),
    ("SUPERSEDED", "SUPERSEDED"),
    ("WONTFIX", "WONTFIX"),
    ("ARCHIVED", "ARCHIVED"),
    ("SHIPPED", "SHIPPED"),
    ("REALIZED", "REALIZED"),
    ("MIGRATED to BUGS", "MIGRATED→BUGS"),
    ("MIGRATED to OBSERVABLES", "MIGRATED→OBS"),
    ("MIGRATED", "MIGRATED"),
    ("RESOLVED", "RESOLVED"),
    ("CLOSED", "CLOSED"),
    ("MITIGATED", "MITIGATED"),
    ("DEFERRED", "DEFERRED"),
    ("INVESTIGATING", "INVESTIGATING"),
    ("AWAITING-OBS", "AWAITING-OBS"),
    ("FIX SHIPPED", "FIX-SHIPPED"),
    ("REVIEW-NEEDED", "REVIEW-NEEDED"),
    ("OPEN", "OPEN"),
]

SEVERITY_WORD_RE = re.compile(r"\b(CRITICAL|CRIT|HIGH|MEDIUM|MED|LOW|COSMETIC)\b")
SEVERITY_EMOJI_RE = re.compile(r"(🔴|🟠|🟡|🟢|⚪)")
SEV_ALIASES = {"CRIT": "CRITICAL", "MED": "MEDIUM"}
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
SLUG_DATE_RE = re.compile(r"-(20\d{6})$")  # YYYYMMDD suffix on slug

EMOJI_TO_SEV = {"🔴": "CRITICAL", "🟠": "HIGH", "🟡": "MEDIUM", "🟢": "LOW", "⚪": "COSMETIC"}


def classify(header: str) -> tuple[str, str, str]:
    """Return (status, severity, date)."""
    status = "OPEN"
    # Strikethrough = closed (default RESOLVED unless something more specific matched)
    if "~~" in header:
        status = "CLOSED"
    for pat, label in STATUS_PATTERNS:
        if pat in header:
            status = label
            break
    sev = ""
    m = SEVERITY_WORD_RE.search(header)
    if m:
        token = m.group(1)
        sev = SEV_ALIASES.get(token, token)
    else:
        m2 = SEVERITY_EMOJI_RE.search(header)
        if m2:
            sev = EMOJI_TO_SEV.get(m2.group(1), "")
    date = ""
    m = DATE_RE.search(header)
    if m:
        date = m.group(1)
    return status, sev, date


def parse_tracker(cfg: TrackerConfig) -> list[Entry]:
    """Parse a tracker file using table-row-first semantics:

    For files WITH a top index table (BUGS.md, OBSERVABLES.md): the table is
    the canonical truth for status. Body `### Foo` sections are just per-entry
    detail; the indexer collects them only for line-number navigation links.

    For files WITHOUT a top index table (DEFERRED_ITEMS.md): fall back to
    body-header parsing with `**Status:**` line + strikethrough/closure-keyword
    detection.

    This inversion eliminates the drift class where a body section's header
    isn't strikethrough'd despite the table row marking the entry FIXED.
    The table is single source of truth.
    """
    if not cfg.body_path.exists():
        print(f"error: {cfg.body_path} not found", file=sys.stderr)
        sys.exit(1)
    text = cfg.body_path.read_text()
    lines = text.splitlines()

    # 1. Build slug → body-line# map from `### ` headers (used only for navigation links)
    body_line_by_slug: dict[str, int] = {}
    for i, line in enumerate(lines, start=1):
        if not line.startswith("### "):
            continue
        m = cfg.slug_re.match(line)
        if not m:
            continue
        slug = m.group("slug").strip()
        # Normalize for joining with table (drop BUG-/OBS- prefix + -date suffix)
        canon = re.sub(r"^(BUG-|OBS-)", "", slug)
        canon = re.sub(r"-20\d{6}$", "", canon)
        body_line_by_slug[slug] = i
        body_line_by_slug.setdefault(canon, i)

    # 2. Walk the top index table (if any) — primary source of truth
    table_entries: list[Entry] = []
    in_table = False
    for ln_idx, ln in enumerate(lines[:300], start=1):
        if ln.startswith("| ID |") or ln.startswith("| Slug |"):
            in_table = True
            continue
        if in_table:
            if not ln.startswith("|"):
                in_table = False
                continue
            cells = [c.strip() for c in ln.split("|")[1:-1]]
            if len(cells) < 3:
                continue
            id_cell, sev_cell, status_cell = cells[0], cells[1], cells[2]
            title_cell = cells[3] if len(cells) >= 4 else ""

            # Parse slug + anchor from id_cell
            #   `[NAME](#anchor)` → slug=NAME, anchor=anchor
            #   `~~NAME~~`        → slug=NAME, no anchor
            #   `[NAME](#anc) 🏛️`  → also OK, extra emojis after
            slug = ""
            anchor = ""
            m_link = re.match(r"\s*\[(~~)?([^\]]+?)(~~)?\]\(#([a-z0-9\-_]+)\)", id_cell)
            if m_link:
                slug = m_link.group(2).strip()
                anchor = m_link.group(4)
            else:
                m_strike = re.match(r"\s*~~([A-Za-z0-9_/\-]+)~~", id_cell)
                if m_strike:
                    slug = m_strike.group(1).strip()
                else:
                    m_bare = re.match(r"\s*([A-Za-z0-9_/\-]+)", id_cell)
                    if m_bare:
                        slug = m_bare.group(1).strip()

            if not slug:
                continue

            # Status classification — table cell is authoritative
            status, _, date = classify(status_cell)
            # Strikethrough in id_cell forces closure if status wasn't already closed
            if "~~" in id_cell and status == "OPEN":
                status = "CLOSED"

            # Severity from sev_cell
            sev = ""
            m_sev = SEVERITY_WORD_RE.search(sev_cell)
            if m_sev:
                token = m_sev.group(1)
                sev = SEV_ALIASES.get(token, token)
            else:
                m_emoji = SEVERITY_EMOJI_RE.search(sev_cell)
                if m_emoji:
                    sev = EMOJI_TO_SEV.get(m_emoji.group(1), "")

            # Date — prefer status cell date, fallback to id_cell date suffix
            if not date:
                m_date = DATE_RE.search(status_cell) or DATE_RE.search(id_cell)
                if m_date:
                    date = m_date.group(1)
            if not date:
                m_yyyymmdd = SLUG_DATE_RE.search(slug)
                if m_yyyymmdd:
                    raw = m_yyyymmdd.group(1)
                    date = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

            # Title — from title cell, truncated
            title = title_cell.replace("\\|", "|").strip()
            if len(title) > 120:
                title = title[:117] + "..."

            # Line link — try anchor match (anchor → body line# via slug lookup), else slug
            line_no = 0
            # Try direct slug match
            if slug in body_line_by_slug:
                line_no = body_line_by_slug[slug]
            else:
                # Try canon form (drop prefix/suffix)
                canon = re.sub(r"^(BUG-|OBS-)", "", slug)
                canon = re.sub(r"-20\d{6}$", "", canon)
                if canon in body_line_by_slug:
                    line_no = body_line_by_slug[canon]
                else:
                    # Try full prefixed: if slug doesn't start with BUG-/OBS-, prepend
                    if cfg.name in ("bugs",) and not slug.startswith("BUG-"):
                        prefixed = f"BUG-{slug}"
                        if prefixed in body_line_by_slug:
                            line_no = body_line_by_slug[prefixed]
                        else:
                            # Match by anchor (body header → anchor)
                            for body_slug, body_ln in body_line_by_slug.items():
                                if anchor and anchor.startswith(body_slug.lower().replace("_", "-")):
                                    line_no = body_ln
                                    break

            table_entries.append(Entry(slug=slug, line=line_no or ln_idx,
                                       header=ln, status=status,
                                       severity=sev, date=date, title=title))

    # If we got table entries, that IS the canonical entry list. Return.
    if table_entries:
        return table_entries

    # 3. No top table (e.g. DEFERRED) → fall back to body-header parsing
    entries: list[Entry] = []
    for i, line in enumerate(lines, start=1):
        if not line.startswith("### "):
            continue
        m = cfg.slug_re.match(line)
        if not m:
            continue
        slug = m.group("slug").strip()
        title = ""
        if "—" in line:
            title = line.split("—", 1)[1].strip()
            title = re.split(r"\s*\[(?:HIGH|MEDIUM|LOW|CRITICAL|COSMETIC|🔴|🟠|🟡|🟢|⚪|✅|🔄|🔍)", title)[0].strip()
            if len(title) > 120:
                title = title[:117] + "..."
        status, sev, date = classify(line)
        # Look at **Status:** line if present
        for j in range(i, min(i + 15, len(lines))):
            ln = lines[j]
            if ln.startswith("### "):
                break
            sm = re.match(r"\s*[-*]?\s*\*\*Status:\*\*\s*(.+)$", ln)
            if sm:
                s, _, d = classify(sm.group(1).strip())
                if s != "OPEN":
                    status = s
                if d and not date:
                    date = d
                break
        if "~~" in line and status == "OPEN":
            status = "CLOSED"
        entries.append(Entry(slug=slug, line=i, header=line, status=status,
                             severity=sev, date=date, title=title))
    return entries


# ─── Rendering ─────────────────────────────────────────────────────────────

def render_index(cfg: TrackerConfig, entries: list[Entry]) -> str:
    open_e = [e for e in entries if e.status in {"OPEN", "INVESTIGATING", "MITIGATED",
                                                  "DEFERRED", "AWAITING-OBS",
                                                  "FIX-SHIPPED", "REVIEW-NEEDED"}]
    closed_e = [e for e in entries if e not in open_e]
    total_lines = cfg.body_path.read_text().count("\n") + 1

    # Severity ranking for sort
    sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "COSMETIC": 4, "": 5}
    open_e.sort(key=lambda e: (sev_rank.get(e.severity, 5), e.slug))

    out: list[str] = []
    out.append(f"# {cfg.title} (auto-generated)")
    out.append("")
    out.append(f"> **Auto-generated** from `{cfg.body_path.relative_to(REPO_ROOT)}` by `scripts/tracker_indexer.py`. **DO NOT EDIT** — regenerate via `python scripts/tracker_indexer.py --emit-index {cfg.body_path.relative_to(REPO_ROOT)}`.")
    out.append("")
    out.append(f"Source body: **{total_lines:,} lines** · This index: ~{len(entries)*2 + 30} lines.")
    out.append("")
    out.append(f"**Total entries:** {len(entries)} ({len(open_e)} active, {len(closed_e)} closed)")
    out.append("")
    out.append("**Active count by severity:**")
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "COSMETIC": 0, "": 0}
    for e in open_e:
        sev_counts[e.severity if e.severity in sev_counts else ""] += 1
    out.append(f"- 🔴 CRITICAL: {sev_counts['CRITICAL']}  ·  🟠 HIGH: {sev_counts['HIGH']}  ·  🟡 MEDIUM: {sev_counts['MEDIUM']}  ·  🟢 LOW: {sev_counts['LOW']}  ·  ⚪ COSMETIC: {sev_counts['COSMETIC']}  ·  (untagged: {sev_counts['']})")
    out.append("")
    out.append("---")
    out.append("")

    # Active entries table
    out.append("## 🔴 Active entries")
    out.append("")
    if not open_e:
        out.append("_(none)_")
        out.append("")
    else:
        out.append("| Slug | Severity | Status | Date | Line | Title |")
        out.append("|---|---|---|---|---|---|")
        for e in open_e:
            title = e.title.replace("|", "\\|")
            out.append(f"| `{e.slug}` | {e.severity or '—'} | {e.status} | {e.date or '—'} | [L{e.line}]({cfg.body_path.name}#L{e.line}) | {title} |")
        out.append("")

    # Closed entries (collapsed)
    out.append(f"## 🪦 Closed entries ({len(closed_e)})")
    out.append("")
    out.append("<details><summary>expand</summary>")
    out.append("")
    if closed_e:
        out.append("| Slug | Severity | Status | Date | Line |")
        out.append("|---|---|---|---|---|")
        for e in closed_e:
            out.append(f"| `{e.slug}` | {e.severity or '—'} | {e.status} | {e.date or '—'} | [L{e.line}]({cfg.body_path.name}#L{e.line}) |")
    out.append("")
    out.append("</details>")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"_Regenerate: `python scripts/tracker_indexer.py --emit-index {cfg.body_path.relative_to(REPO_ROOT)}`. Drift-check: `python scripts/tracker_indexer.py --check {cfg.body_path.relative_to(REPO_ROOT)}` (non-zero exit on mismatch — wired into pre-commit)._")
    out.append("")
    return "\n".join(out)


def find_config(path_arg: str) -> TrackerConfig:
    p = Path(path_arg).resolve()
    for cfg in TRACKERS.values():
        if cfg.body_path == p:
            return cfg
    # Fall back to name match
    for cfg in TRACKERS.values():
        if cfg.body_path.name == p.name:
            return cfg
    print(f"error: unknown tracker {path_arg}; known: {list(TRACKERS)}", file=sys.stderr)
    sys.exit(2)


def cmd_emit_one(cfg: TrackerConfig) -> int:
    entries = parse_tracker(cfg)
    text = render_index(cfg, entries)
    cfg.index_path.write_text(text)
    open_n = sum(1 for e in entries if e.status not in {"FIXED", "PASSED", "RESOLVED",
        "CLOSED", "SHIPPED", "SUPERSEDED", "WONTFIX", "ARCHIVED", "REALIZED",
        "MIGRATED", "MIGRATED→BUGS", "MIGRATED→OBS", "RESOLVED-UNNECESSARY"})
    print(f"wrote {cfg.index_path.relative_to(REPO_ROOT)} ({len(entries)} entries, {open_n} active)")
    return 0


def cmd_check_one(cfg: TrackerConfig) -> int:
    entries = parse_tracker(cfg)
    fresh = render_index(cfg, entries)
    if not cfg.index_path.exists():
        print(f"DRIFT: {cfg.index_path.relative_to(REPO_ROOT)} does not exist", file=sys.stderr)
        return 1
    on_disk = cfg.index_path.read_text()
    if fresh != on_disk:
        print(f"DRIFT: {cfg.index_path.relative_to(REPO_ROOT)} is stale — regenerate via "
              f"`python scripts/tracker_indexer.py --emit-index {cfg.body_path.relative_to(REPO_ROOT)}`",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit-index", metavar="FILE", help="regenerate index for one tracker")
    g.add_argument("--check", metavar="FILE", help="exit non-zero if index file is stale")
    g.add_argument("--emit-all", action="store_true", help="regenerate every registered tracker index")
    g.add_argument("--check-all", action="store_true", help="drift gate for all registered trackers")
    args = ap.parse_args()

    if args.emit_index:
        return cmd_emit_one(find_config(args.emit_index))
    if args.check:
        return cmd_check_one(find_config(args.check))
    if args.emit_all:
        rc = 0
        for cfg in TRACKERS.values():
            rc |= cmd_emit_one(cfg)
        return rc
    if args.check_all:
        rc = 0
        for cfg in TRACKERS.values():
            rc |= cmd_check_one(cfg)
        return rc
    return 2


if __name__ == "__main__":
    sys.exit(main())
