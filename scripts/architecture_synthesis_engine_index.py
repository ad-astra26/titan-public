#!/usr/bin/env python3
"""ARCHITECTURE_synthesis_engine_index.md generator — auto-built TOC + §19 invariants ledger + §20 considered-rejected ledger.

Mirrors `scripts/architecture_cgn_family_index.py` discipline for the
Synthesis-Engine architecture narrative doc, adapted to its structure:

  1. Lists every `## §X — Title` and `### §X.Y — Subtitle` with line numbers
     (and the section's maturity tag) so future sessions can
     `Read offset=N limit=L` directly into the relevant section.
  2. Extracts the §19 Invariants TABLE (| INV-N | ... | status |) into a
     quick-reference ledger.
  3. Extracts the §20 Considered-and-rejected bullet list into a
     "design alternatives" ledger.

Output: titan-docs/ARCHITECTURE_synthesis_engine_index.md (overwrites; never hand-edit).

Regenerate after every body edit. Wired into the pre-commit drift gate and the
git merge driver. Deterministic + idempotent: same input → byte-identical output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_PATH = REPO_ROOT / "titan-docs" / "ARCHITECTURE_synthesis_engine.md"
INDEX_PATH = REPO_ROOT / "titan-docs" / "ARCHITECTURE_synthesis_engine_index.md"


SECTION_RE = re.compile(r"^(#{2,4})\s+(§[\w\.]+)\s*[—-]\s*(.+?)\s*$")
# §19 invariants table row: | INV-3 | <text> | <status> |
INVARIANT_ROW_RE = re.compile(r"^\|\s*(INV-\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
# §20 rejected bullet: - **<pattern>** <rest — reason>
REJECTED_RE = re.compile(r"^-\s+\*\*(.+?)\*\*\s*(.+?)\s*$")
# maturity tag suffix on a header title:  `[RATIFIED]` / `[TARGET]` / `[meta]` / `[ledger]` ...
MATURITY_RE = re.compile(r"`\[([A-Za-z][^\]]*)\]`\s*$")
FRONTMATTER_VERSION_RE = re.compile(r"^version:\s*(.+?)\s*$")
FRONTMATTER_UPDATED_RE = re.compile(r"^last_updated:\s*(.+?)\s*$")


def _split_maturity(title: str) -> tuple[str, str]:
    """Return (clean_title, maturity_tag) — strip a trailing `[TAG]` if present."""
    m = MATURITY_RE.search(title)
    if not m:
        return title.strip(), ""
    clean = MATURITY_RE.sub("", title).strip()
    return clean, m.group(1).strip()


def parse_arch(arch_text: str) -> dict:
    sections: list[dict] = []
    invariants: list[dict] = []
    rejected: list[dict] = []
    version = "unknown"
    last_updated = "unknown"

    in_invariants = False
    in_rejected = False

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
            level = len(m_s.group(1))
            sect_num = m_s.group(2)
            title, maturity = _split_maturity(m_s.group(3).strip())
            sections.append(
                {"level": level, "num": sect_num, "title": title,
                 "maturity": maturity, "line": i}
            )
            top_level = sect_num.split(".")[0]
            in_invariants = top_level == "§19"
            in_rejected = top_level == "§20"
            continue

        if in_invariants:
            m_inv = INVARIANT_ROW_RE.match(raw)
            if m_inv and m_inv.group(1).startswith("INV-"):
                invariants.append(
                    {"id": m_inv.group(1),
                     "text": m_inv.group(2).strip()[:90],
                     "status": m_inv.group(3).strip()[:40],
                     "line": i}
                )

        if in_rejected:
            m_rej = REJECTED_RE.match(raw)
            if m_rej:
                rejected.append(
                    {"pattern": m_rej.group(1).strip(),
                     "note": m_rej.group(2).strip()[:90],
                     "line": i}
                )

    return {
        "version": version,
        "last_updated": last_updated,
        "sections": sections,
        "invariants": invariants,
        "rejected": rejected,
        "total_lines": len(lines),
    }


def render(parsed: dict) -> str:
    out: list[str] = []
    out.append("# ARCHITECTURE — Synthesis Engine Index (auto-generated)")
    out.append("")
    out.append(
        "> **Auto-generated** from `titan-docs/ARCHITECTURE_synthesis_engine.md` by "
        "`scripts/architecture_synthesis_engine_index.py`. **DO NOT EDIT** — "
        "regenerate via `python scripts/architecture_synthesis_engine_index.py` "
        "after every body edit. Any `version` bump on the architecture doc MUST "
        "regenerate this index in the same commit (pre-commit drift gate enforces it)."
    )
    out.append("")
    out.append(
        f"**Architecture doc version:** {parsed['version']} "
        f"(last_updated: {parsed['last_updated']}; {parsed['total_lines']:,} lines)"
    )
    out.append("")
    out.append(
        "> **Read this file FIRST when working on the Synthesis Engine / outer "
        "memory / ACT-R / hypothesis forks / oracle+proof middleware / inner-outer "
        "integration.** Pair with the §0 Changelog at the top of "
        "`ARCHITECTURE_synthesis_engine.md`. Implementation proposals + phasing live "
        "in `rFP_outer_memory_enhancement.md`. Proto-SPEC-module: absorbs into a "
        "future `SPEC_synthesis_engine.md` at the ~5k-line SPEC modularization threshold."
    )
    out.append("")

    # ── Table of Contents ────────────────────────────────────────────────
    out.append("## Table of Contents — §X sections (use line numbers to Read offset)")
    out.append("")
    out.append("| § | Title | Maturity | Line | Level |")
    out.append("|---|---|---|---|---|")
    for s in parsed["sections"]:
        if s["level"] >= 4:
            continue
        indent = "&nbsp;&nbsp;" * (s["level"] - 2)
        maturity = s["maturity"] or "—"
        out.append(
            f"| {indent}{s['num']} | {indent}{s['title']} | {maturity} | "
            f"{s['line']} | {'§' if s['level'] == 2 else '§§'} |"
        )
    out.append("")

    # ── §19 Invariants ledger ─────────────────────────────────────────
    if parsed["invariants"]:
        out.append("## §19 Invariants — Maker-locked rules (load-bearing)")
        out.append("")
        out.append(
            "> Removal or violation of any invariant requires Maker greenlight "
            "(per `feedback_spec_changes_need_maker_greenlight_first.md`). "
            "INV-3 + INV-9 additionally gate folding into the main SPEC."
        )
        out.append("")
        out.append("| ID | Invariant | Status | Line |")
        out.append("|---|---|---|---|")
        for inv in parsed["invariants"]:
            out.append(
                f"| {inv['id']} | {inv['text']} | {inv['status']} | {inv['line']} |"
            )
        out.append("")

    # ── §20 Considered-and-rejected ledger ──────────────────────────
    if parsed["rejected"]:
        out.append("## §20 Considered and Rejected — design alternatives")
        out.append("")
        out.append(
            "> Before proposing a new pattern, check here for "
            "\"have we already considered X?\""
        )
        out.append("")
        out.append("| Rejected Pattern | Note | Line |")
        out.append("|---|---|---|")
        for rej in parsed["rejected"]:
            out.append(f"| {rej['pattern']} | {rej['note']} | {rej['line']} |")
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "_Regenerate with `python scripts/architecture_synthesis_engine_index.py` "
        "after every doc body edit. Output is deterministic + idempotent._"
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
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
            print(f"  fix: python {Path(__file__).relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        print(f"clean: {INDEX_PATH.relative_to(REPO_ROOT)} matches body")
        return 0

    INDEX_PATH.write_text(index_text)
    print(
        f"wrote {INDEX_PATH.relative_to(REPO_ROOT)} "
        f"({len(parsed['sections'])} sections, {len(parsed['invariants'])} invariants, "
        f"{len(parsed['rejected'])} rejected-design entries; "
        f"v{parsed['version']}, {parsed['total_lines']:,} lines)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
