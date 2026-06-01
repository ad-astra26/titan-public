#!/usr/bin/env python3
"""ARCHITECTURE_backup_restore_index.md generator — auto-built TOC + §A invariants ledger.

Mirrors `scripts/architecture_cgn_family_index.py` discipline for the
Backup/Restore/Resurrection SPEC family module:

  1. Lists every `## §X — Title` / `### §X.Y` / `#### §X.Y.Z` with line numbers
     so future sessions can `Read offset=N limit=L` directly into the relevant
     section without scanning the whole doc.
  2. Extracts the §A Invariants ledger (INV-BR-N table rows) into a quick-
     reference table with lock date — pairs with the Changelog at the top of
     ARCHITECTURE_backup_restore.md.

Output: titan-docs/specs/ARCHITECTURE_backup_restore_index.md (overwrites; never hand-edit).

Regenerate after every body edit. Wired into session_close_protocol when
`version` is bumped, same discipline as SPEC versioning. The pre-commit drift
gate runs `--check` and FAILS a staged body edit without a regenerated index.

Deterministic + idempotent: same input always produces byte-identical output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_PATH = REPO_ROOT / "titan-docs" / "specs" / "ARCHITECTURE_backup_restore.md"
INDEX_PATH = REPO_ROOT / "titan-docs" / "specs" / "ARCHITECTURE_backup_restore_index.md"


SECTION_RE = re.compile(r"^(#{2,4})\s+(§[\w\.]+(?:[A-Z])?)\s*[—-]\s*(.+?)\s*$")
# §A invariants are a markdown table: | INV-BR-N | invariant text | YYYY-MM-DD |
INVARIANT_ROW_RE = re.compile(
    r"^\|\s*(INV-BR-\d+)\s*\|\s*(.+?)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*$"
)
FRONTMATTER_VERSION_RE = re.compile(r"^version:\s*(.+?)\s*$")
FRONTMATTER_UPDATED_RE = re.compile(r"^last_updated:\s*(.+?)\s*$")


def parse_arch(arch_text: str) -> dict:
    """Walk doc line-by-line + extract sections and §A INV-BR invariants."""
    sections: list[dict] = []
    invariants: list[dict] = []
    version = "unknown"
    last_updated = "unknown"

    lines = arch_text.splitlines()
    for i, raw in enumerate(lines, start=1):
        m_v = FRONTMATTER_VERSION_RE.match(raw)
        if m_v and version == "unknown":
            version = m_v.group(1)
        m_u = FRONTMATTER_UPDATED_RE.match(raw)
        if m_u and last_updated == "unknown":
            last_updated = m_u.group(1)

        m_s = SECTION_RE.match(raw)
        if m_s:
            sections.append(
                {
                    "level": len(m_s.group(1)),
                    "num": m_s.group(2),
                    "title": m_s.group(3).strip(),
                    "line": i,
                }
            )

        m_inv = INVARIANT_ROW_RE.match(raw)
        if m_inv:
            invariants.append(
                {
                    "id": m_inv.group(1),
                    "title": m_inv.group(2).strip(),
                    "lock_date": m_inv.group(3),
                    "line": i,
                }
            )

    return {
        "version": version,
        "last_updated": last_updated,
        "sections": sections,
        "invariants": invariants,
        "total_lines": len(lines),
    }


def render(parsed: dict) -> str:
    """Compose the ARCHITECTURE_backup_restore_index.md content."""
    out: list[str] = []
    out.append("# ARCHITECTURE — Backup/Restore Family Index (auto-generated)")
    out.append("")
    out.append(
        "> **Auto-generated** from `titan-docs/specs/ARCHITECTURE_backup_restore.md` "
        "by `scripts/architecture_backup_restore_index.py`. **DO NOT EDIT** — "
        "regenerate via `python scripts/architecture_backup_restore_index.py` after "
        "every body edit. Wired into `session_close_protocol.md`: any `version` bump "
        "on the architecture doc MUST regenerate this index in the same commit."
    )
    out.append("")
    out.append(
        f"**Architecture doc version:** {parsed['version']} "
        f"(last_updated: {parsed['last_updated']}; "
        f"{parsed['total_lines']:,} lines)"
    )
    out.append("")
    out.append(
        "> **Read this file FIRST when working on backup / restore / resurrection / "
        "sovereign chain / Arweave / ZK Vault topics.** This is the canonical SPEC "
        "family module for §24; the main SPEC carries a §24 stub→ref. `§24.x` IDs "
        "are stable contract identifiers."
    )
    out.append("")

    # ── Table of Contents ────────────────────────────────────────────────
    out.append("## Table of Contents — §X sections (use line numbers to Read offset)")
    out.append("")
    out.append("| § | Title | Line | Level |")
    out.append("|---|---|---|---|")
    for s in parsed["sections"]:
        if s["level"] >= 5:
            continue
        indent = "&nbsp;&nbsp;" * (s["level"] - 2)
        marker = {2: "§", 3: "§§", 4: "§§§"}.get(s["level"], "§")
        out.append(
            f"| {indent}{s['num']} | {indent}{s['title']} | {s['line']} | {marker} |"
        )
    out.append("")

    # ── §A Invariants ledger ─────────────────────────────────────────────
    if parsed["invariants"]:
        out.append("## §A Invariants — Maker-locked rules (load-bearing)")
        out.append("")
        out.append(
            "> Pair with the SPEC Preamble Rules (G18-G22) — the L0/L1 analogs of "
            "these L2 architectural invariants. Removal or violation of any INV-BR "
            "requires Maker greenlight (per "
            "`feedback_spec_changes_need_maker_greenlight_first.md`)."
        )
        out.append("")
        out.append("| ID | Invariant | Lock Date | Line |")
        out.append("|---|---|---|---|")
        for inv in parsed["invariants"]:
            out.append(
                f"| {inv['id']} | {inv['title']} | {inv['lock_date']} | {inv['line']} |"
            )
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "_Regenerate with `python scripts/architecture_backup_restore_index.py` "
        "after every doc body edit. Output is deterministic + idempotent — same "
        "input always produces byte-identical output._"
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if index file is stale (drift gate for pre-commit)",
    )
    args = ap.parse_args()

    if not ARCH_PATH.exists():
        print(f"error: architecture doc not found at {ARCH_PATH}", file=sys.stderr)
        return 1
    arch_text = ARCH_PATH.read_text()
    parsed = parse_arch(arch_text)
    index_text = render(parsed)

    if args.check:
        existing = INDEX_PATH.read_text() if INDEX_PATH.exists() else ""
        if existing != index_text:
            print(
                f"drift: {INDEX_PATH.relative_to(REPO_ROOT)} is stale relative to "
                f"{ARCH_PATH.relative_to(REPO_ROOT)}",
                file=sys.stderr,
            )
            print(
                f"  fix: python {Path(__file__).relative_to(REPO_ROOT)}",
                file=sys.stderr,
            )
            return 1
        print(f"clean: {INDEX_PATH.relative_to(REPO_ROOT)} matches body")
        return 0

    INDEX_PATH.write_text(index_text)
    print(
        f"wrote {INDEX_PATH.relative_to(REPO_ROOT)} "
        f"({len(parsed['sections'])} sections, "
        f"{len(parsed['invariants'])} invariants; "
        f"v{parsed['version']}, {parsed['total_lines']:,} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
