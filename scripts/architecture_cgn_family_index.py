#!/usr/bin/env python3
"""ARCHITECTURE_cgn_family_index.md generator — auto-built TOC + §11 invariants ledger + §12 considered-rejected ledger.

Mirrors `scripts/spec_index.py` discipline for the CGN-family architecture
narrative doc:

  1. Lists every `## §X — Title` and `### §X.Y — Subtitle` with line numbers
     so future sessions can `Read offset=N limit=L` directly into the relevant
     section without scanning the whole 2000+ line doc.
  2. Extracts §11 Invariants entries (§11.1..§11.N) into a quick-reference
     table with lock date — pairs with the Changelog at the top of
     ARCHITECTURE_cgn_family.md.
  3. Extracts §12 Considered-and-rejected entries into a "design alternatives"
     table — fast lookup for "have we considered X already?"

Output: titan-docs/ARCHITECTURE_cgn_family_index.md (overwrites; never hand-edit).

Regenerate after every body edit. Wired into session_close_protocol when
`version` field is bumped per same discipline as SPEC versioning.

Deterministic + idempotent: same input always produces byte-identical output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_PATH = REPO_ROOT / "titan-docs" / "ARCHITECTURE_cgn_family.md"
INDEX_PATH = REPO_ROOT / "titan-docs" / "ARCHITECTURE_cgn_family_index.md"


SECTION_RE = re.compile(r"^(#{2,4})\s+(§[\w\.]+(?:[A-Z])?)\s*[—-]\s*(.+?)\s*$")
INVARIANT_RE = re.compile(r"^###\s+§11\.(\d+)\s*[—-]\s*(.+?)\s*$")
REJECTED_RE = re.compile(r"^###\s+§12\.(\d+)\s*[—-]\s*❌?\s*(.+?)\s*$")
FRONTMATTER_VERSION_RE = re.compile(r"^version:\s*(.+?)\s*$")
FRONTMATTER_UPDATED_RE = re.compile(r"^last_updated:\s*(.+?)\s*$")
LOCK_DATE_RE = re.compile(r"\*\*Lock:\*\*\s*(\d{4}-\d{2}-\d{2})")
STATUS_RE = re.compile(r"\*\*Status:\*\*\s*(.+?)(?:\.|\n|$)")


def parse_arch(arch_text: str) -> dict:
    """Walk doc line-by-line + extract sections, §11 invariants, §12 rejected."""
    sections: list[dict] = []
    invariants: list[dict] = []
    rejected: list[dict] = []
    version = "unknown"
    last_updated = "unknown"

    in_invariants = False
    in_rejected = False
    current_invariant: dict | None = None
    current_rejected: dict | None = None

    lines = arch_text.splitlines()
    for i, raw in enumerate(lines, start=1):
        # Frontmatter scan
        m_v = FRONTMATTER_VERSION_RE.match(raw)
        if m_v and version == "unknown":
            version = m_v.group(1)
        m_u = FRONTMATTER_UPDATED_RE.match(raw)
        if m_u and last_updated == "unknown":
            last_updated = m_u.group(1)

        # Top-level section headers (## §X) + subsections (### §X.Y)
        m_s = SECTION_RE.match(raw)
        if m_s:
            level = len(m_s.group(1))
            sect_num = m_s.group(2)
            title = m_s.group(3).strip()
            sections.append(
                {"level": level, "num": sect_num, "title": title, "line": i}
            )
            # Flag scope of §11 (Invariants) + §12 (Considered-and-rejected)
            top_level = sect_num.split(".")[0]
            in_invariants = top_level == "§11"
            in_rejected = top_level == "§12"

            # Capture §11.X invariants
            if in_invariants:
                m_inv = INVARIANT_RE.match(raw)
                if m_inv:
                    current_invariant = {
                        "num": int(m_inv.group(1)),
                        "title": m_inv.group(2).strip(),
                        "line": i,
                        "lock_date": "?",
                        "summary": "",
                    }
                    invariants.append(current_invariant)
                    current_rejected = None

            # Capture §12.X rejected
            if in_rejected:
                m_rej = REJECTED_RE.match(raw)
                if m_rej:
                    current_rejected = {
                        "num": int(m_rej.group(1)),
                        "title": m_rej.group(2).strip(),
                        "line": i,
                        "status": "?",
                    }
                    rejected.append(current_rejected)
                    current_invariant = None

        # Within an active §11.X invariant, look for **Lock:** date
        if current_invariant is not None:
            m_l = LOCK_DATE_RE.search(raw)
            if m_l and current_invariant["lock_date"] == "?":
                current_invariant["lock_date"] = m_l.group(1)

        # Within an active §12.X rejected, look for **Status:** line
        if current_rejected is not None:
            m_st = STATUS_RE.search(raw)
            if m_st and current_rejected["status"] == "?":
                # Trim long status texts to ≤80 chars for the table
                current_rejected["status"] = m_st.group(1).strip()[:80]

    return {
        "version": version,
        "last_updated": last_updated,
        "sections": sections,
        "invariants": invariants,
        "rejected": rejected,
        "total_lines": len(lines),
    }


def render(parsed: dict) -> str:
    """Compose the ARCHITECTURE_cgn_family_index.md content."""
    out: list[str] = []
    out.append("# ARCHITECTURE — CGN Family Index (auto-generated)")
    out.append("")
    out.append(
        "> **Auto-generated** from `titan-docs/ARCHITECTURE_cgn_family.md` by "
        "`scripts/architecture_cgn_family_index.py`. **DO NOT EDIT** — regenerate "
        "via `python scripts/architecture_cgn_family_index.py` after every body "
        "edit. Wired into `session_close_protocol.md` Step 1b: any `version` "
        "bump on the architecture doc MUST regenerate this index in the same commit."
    )
    out.append("")
    out.append(
        f"**Architecture doc version:** {parsed['version']} "
        f"(last_updated: {parsed['last_updated']}; "
        f"{parsed['total_lines']:,} lines)"
    )
    out.append("")
    out.append(
        "> **Read this file FIRST when working on CGN / META-CGN / EMOT-CGN / "
        "meta-reasoning / meta-teacher / HAOV / Sigma.** Pair with the Changelog "
        "at the top of `ARCHITECTURE_cgn_family.md` (recent changes) to get fast "
        "architectural awareness before any rFP draft or code change. The full "
        "doc is the proto-SPEC-module that will absorb into a future "
        "`SPEC_cgn_family.md` when the monolithic SPEC modularizes (~5k line "
        "threshold per Maker direction)."
    )
    out.append("")

    # ── Table of Contents ────────────────────────────────────────────────
    out.append("## Table of Contents — §X sections (use line numbers to Read offset)")
    out.append("")
    out.append("| § | Title | Line | Level |")
    out.append("|---|---|---|---|")
    for s in parsed["sections"]:
        if s["level"] >= 4:
            continue
        indent = "&nbsp;&nbsp;" * (s["level"] - 2)
        out.append(
            f"| {indent}{s['num']} | {indent}{s['title']} | {s['line']} | "
            f"{'§' if s['level'] == 2 else '§§'} |"
        )
    out.append("")

    # ── §11 Invariants ledger ─────────────────────────────────────────
    if parsed["invariants"]:
        out.append("## §11 Invariants — Maker-locked rules (load-bearing)")
        out.append("")
        out.append(
            "> Pair with the SPEC Preamble Rules (G18-G22) which are the L0/L1 "
            "analogs of these L2 architectural invariants. Removal or violation "
            "of any §11 invariant requires Maker greenlight (per "
            "`feedback_spec_changes_need_maker_greenlight_first.md`)."
        )
        out.append("")
        out.append("| §11.N | Invariant | Lock Date | Line |")
        out.append("|---|---|---|---|")
        for inv in sorted(parsed["invariants"], key=lambda x: x["num"]):
            out.append(
                f"| §11.{inv['num']} | {inv['title']} | {inv['lock_date']} | "
                f"{inv['line']} |"
            )
        out.append("")

    # ── §12 Considered-and-rejected ledger ──────────────────────────
    if parsed["rejected"]:
        out.append("## §12 Considered and Rejected — design alternatives")
        out.append("")
        out.append(
            "> Institutional memory of past design decisions. Before proposing "
            "a new design pattern in this family, check here for "
            "\"have we already considered X?\" Each entry has provenance."
        )
        out.append("")
        out.append("| §12.N | Rejected Pattern | Status | Line |")
        out.append("|---|---|---|---|")
        for rej in sorted(parsed["rejected"], key=lambda x: x["num"]):
            status_short = rej["status"] if rej["status"] != "?" else "—"
            out.append(
                f"| §12.{rej['num']} | {rej['title']} | {status_short} | "
                f"{rej['line']} |"
            )
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "_Regenerate with `python scripts/architecture_cgn_family_index.py` "
        "after every doc body edit. Output is deterministic + idempotent — "
        "same input always produces byte-identical output._"
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
        # Drift check — compare current index to what we would write.
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
        f"{len(parsed['invariants'])} invariants, "
        f"{len(parsed['rejected'])} rejected-design entries; "
        f"v{parsed['version']}, {parsed['total_lines']:,} lines)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
