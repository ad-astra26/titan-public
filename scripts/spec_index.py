#!/usr/bin/env python3
"""SPEC_index.md generator — auto-built TOC + Decision Log index + §9.B module map.

Per `feedback_spec_changelog_mandatory.md` + session_startup_protocol Step 1aa:
the SPEC has grown past ~3000 lines and reading the full file at session start
costs ~30K tokens. This script produces a compact (~200-line) index that:

  1. Lists every `## §X — Title` and `### §X.Y — Subtitle` with line numbers
     so future sessions can `Read offset=N limit=L` directly into the relevant
     section without scanning.
  2. Extracts the `## §21 — Decision Log` D-SPEC-NN entries into a
     reverse-chronological table — pairs with the SPEC Changelog
     (`SPEC_changelog.md`). Since Tier 1b SPEC-leaning (2026-06-01) the §21
     Decision Log lives in the companion `SPEC_decision_log.md`; the main SPEC
     keeps only a pointer stub. This generator parses D-SPEC entries from the
     companion when present (falling back to the main SPEC's §21 if not).
  3. Extracts `## §9.B — Python tree` `#### module_name` sub-blocks into a
     worker → section line map for fast "which §9.B has worker X" lookup.
     Since Tier 2 SPEC-leaning (2026-06-01) the §9 matrix lives in the companion
     `SPEC_subscriber_hierarchy_matrix.md`; the main SPEC keeps only a wiring-
     contract stub. This generator parses the §9.B module map from the companion
     when present (falling back to the main SPEC's §9.B if not).

Output: titan-docs/specs/SPEC_index.md (overwrites; never hand-edit).

Regenerate after every SPEC body edit. Wired into session_close_protocol Step 1b
when `spec_version` is bumped per §2.6.

Deterministic + idempotent: same input always produces byte-identical output
(stable ordering, no timestamps in output body — only `last_updated` from the
SPEC frontmatter).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
INDEX_PATH = REPO_ROOT / "titan-docs" / "specs" / "SPEC_index.md"
# §21 Decision Log was extracted to this companion (Tier 1b SPEC-leaning,
# 2026-06-01). It is the canonical source for D-SPEC-NN entries; the main SPEC
# §21 is now a pointer stub. `parse_spec` already keys D-SPEC extraction on the
# `## §21` header, so running it over the companion yields the D-SPEC index.
DECISION_LOG_PATH = REPO_ROOT / "titan-docs" / "specs" / "SPEC_decision_log.md"
# §9 Subscriber/Consumer/Hierarchy Matrix was extracted to this companion
# (Tier 2 SPEC-leaning, 2026-06-01). It is the canonical source for the §9.B
# Python-tree module→line map; the main SPEC §9 is now a wiring-contract stub.
# `parse_spec` keys module extraction on the `§9.B` header, so running it over
# the companion yields the module map.
SUBSCRIBER_MATRIX_PATH = (
    REPO_ROOT / "titan-docs" / "specs" / "SPEC_subscriber_hierarchy_matrix.md"
)


SECTION_RE = re.compile(r"^(#{2,4})\s+(§[\w\.]+(?:[A-Z])?)\s*[—-]\s*(.+?)\s*$")
DSPEC_RE = re.compile(
    r"^- \*\*D-SPEC-(\d+)\s*\(([^)]+)\)\*\*\s*:\s*(\*\*)?(.+?)(?:\.|$)"
)
MODULE_RE = re.compile(r"^####\s+`?([a-zA-Z_][a-zA-Z_0-9]*)`?(?:\s*\(.*?\))?\s*$")
FRONTMATTER_VERSION_RE = re.compile(r"^spec_version:\s*(.+?)\s*$")
FRONTMATTER_UPDATED_RE = re.compile(r"^last_updated:\s*(.+?)\s*$")


def parse_spec(spec_text: str) -> dict:
    """Walk SPEC line-by-line + extract sections, D-SPEC entries, §9.B modules."""
    sections: list[dict] = []
    dspecs: list[dict] = []
    modules: list[dict] = []
    spec_version = "unknown"
    last_updated = "unknown"

    in_decision_log = False
    in_python_tree = False  # tracks §9.B
    current_dspec_date = None  # most recent "**YYYY-MM-DD ...**" date header
    current_section: dict | None = None

    lines = spec_text.splitlines()
    for i, raw in enumerate(lines, start=1):
        # Frontmatter scan (top of file)
        m_v = FRONTMATTER_VERSION_RE.match(raw)
        if m_v and spec_version == "unknown":
            spec_version = m_v.group(1)
        m_u = FRONTMATTER_UPDATED_RE.match(raw)
        if m_u and last_updated == "unknown":
            last_updated = m_u.group(1)

        # Top-level section headers (## §X) and subsections (### §X.Y)
        m_s = SECTION_RE.match(raw)
        if m_s:
            level = len(m_s.group(1))
            sect_num = m_s.group(2)
            title = m_s.group(3).strip()
            current_section = {
                "level": level,
                "num": sect_num,
                "title": title,
                "line": i,
            }
            sections.append(current_section)
            # Flag scope of §21 (Decision Log) + §9.B (Python tree)
            in_decision_log = sect_num == "§21"
            # §9 has subsections §9.A, §9.B etc.; only §9.B = Python tree
            in_python_tree = sect_num == "§9.B"

        # Date-bucket headers within Decision Log:
        # **YYYY-MM-DD (... — SPEC vX.Y.Z → vA.B.C ...):**
        if in_decision_log and raw.startswith("**20") and raw.rstrip().endswith(":**"):
            current_dspec_date = raw[2:12]  # YYYY-MM-DD

        # D-SPEC-NN entries
        m_d = DSPEC_RE.match(raw)
        if m_d and in_decision_log:
            num = int(m_d.group(1))
            version_jump = m_d.group(2).strip()
            summary_raw = m_d.group(4).strip()
            # First sentence as summary (≤200 chars).
            summary = summary_raw.split(".")[0].strip()
            summary = re.sub(r"\*\*", "", summary)[:200]
            dspecs.append(
                {
                    "num": num,
                    "version": version_jump,
                    "date": current_dspec_date or "?",
                    "line": i,
                    "summary": summary,
                }
            )

        # §9.B module sub-blocks: lines like "#### `cognitive_worker`" or
        # "#### ns_module".
        if in_python_tree:
            m_m = MODULE_RE.match(raw)
            if m_m:
                modules.append({"name": m_m.group(1), "line": i})

    return {
        "spec_version": spec_version,
        "last_updated": last_updated,
        "sections": sections,
        "dspecs": dspecs,
        "modules": modules,
        "total_lines": len(lines),
    }


def render(parsed: dict) -> str:
    """Compose the SPEC_index.md content from parsed structures."""
    out: list[str] = []
    out.append("# SPEC Architecture Index (auto-generated)")
    out.append("")
    out.append(
        "> **Auto-generated** from `titan-docs/specs/SPEC_titan_architecture.md` by "
        "`scripts/spec_index.py`. **DO NOT EDIT** — regenerate via "
        "`python scripts/spec_index.py` after every SPEC body edit. Wired into "
        "`session_close_protocol.md` Step 1b: any `spec_version` bump per §2.6 "
        "MUST regenerate this index in the same commit."
    )
    out.append("")
    out.append(
        f"**SPEC version:** {parsed['spec_version']} "
        f"(last_updated: {parsed['last_updated']}; "
        f"{parsed['total_lines']:,} lines)"
    )
    out.append("")
    out.append(
        "> **Read this file FIRST at session start** — per "
        "`memory/session_startup_protocol.md` Step 1aa. It is ~200 lines vs "
        "the SPEC's ~3000, and pairs with the SPEC Changelog at the top of "
        "the SPEC itself (recent changes) to give you fast architectural "
        "awareness before any rFP draft or code change."
    )
    out.append("")

    # ── Table of Contents ────────────────────────────────────────────────
    out.append("## Table of Contents — §X sections (use line numbers to Read offset)")
    out.append("")
    out.append("| § | Title | Line | Level |")
    out.append("|---|---|---|---|")
    for s in parsed["sections"]:
        # Skip subsubsections (level 4 = ####) from TOC; they appear in the
        # §9.B module index below if applicable.
        if s["level"] >= 4:
            continue
        indent = "&nbsp;&nbsp;" * (s["level"] - 2)
        out.append(
            f"| {indent}{s['num']} | {indent}{s['title']} | {s['line']} | "
            f"{'§' if s['level'] == 2 else '§§'} |"
        )
    out.append("")

    # ── Decision Log index (reverse-chronological by D-SPEC number) ──
    out.append("## Decision Log — D-SPEC-NN index (most-recent first)")
    out.append("")
    out.append(
        "> Pair with the **SPEC Changelog** (`SPEC_changelog.md`) for recent "
        "version bumps. The Decision Log holds the full architectural rationale "
        "per D-SPEC; the Changelog holds the one-line summary + version mapping. "
        "Read both at session start. **The `Line` column below indexes into "
        "`SPEC_decision_log.md`** (the §21 Decision Log was extracted there in "
        "Tier 1b SPEC-leaning, 2026-06-01) — `Read offset=N` that file to "
        "deep-dive a decision's rationale."
    )
    out.append("")
    out.append("| D-SPEC | Version Bump | Date | Line (SPEC_decision_log.md) | Summary |")
    out.append("|---|---|---|---|---|")
    for d in sorted(parsed["dspecs"], key=lambda x: -x["num"]):
        out.append(
            f"| D-SPEC-{d['num']} | {d['version']} | {d['date']} | "
            f"{d['line']} | {d['summary']} |"
        )
    out.append("")

    # ── §9.B Python module index ───────────────────────────────────────
    if parsed["modules"]:
        out.append("## §9.B Python tree — module → line map")
        out.append("")
        out.append(
            "> Direct jump to a worker's `Owns` / bus subs / pubs / SHM "
            "reads-writes / persisted state block. For Rust L1 daemons see "
            "§9.A. **The `Line` column below indexes into "
            "`SPEC_subscriber_hierarchy_matrix.md`** (the §9 matrix was extracted "
            "there in Tier 2 SPEC-leaning, 2026-06-01) — `Read offset=N` that file."
        )
        out.append("")
        out.append("| Module | Line (SPEC_subscriber_hierarchy_matrix.md) |")
        out.append("|---|---|")
        for m in sorted(parsed["modules"], key=lambda x: x["name"]):
            out.append(f"| `{m['name']}` | {m['line']} |")
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "_Regenerate with `python scripts/spec_index.py` after every SPEC "
        "body edit. Output is deterministic + idempotent — same input "
        "always produces byte-identical output._"
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    if not SPEC_PATH.exists():
        print(f"error: SPEC not found at {SPEC_PATH}", file=sys.stderr)
        return 1
    spec_text = SPEC_PATH.read_text()
    parsed = parse_spec(spec_text)

    # D-SPEC entries live in the companion since Tier 1b (the main SPEC §21 is a
    # pointer stub). Re-parse the companion for D-SPEC entries; fall back to the
    # main SPEC's §21 only if the companion is absent (pre-extraction state).
    if DECISION_LOG_PATH.exists():
        dlog_parsed = parse_spec(DECISION_LOG_PATH.read_text())
        if dlog_parsed["dspecs"]:
            parsed["dspecs"] = dlog_parsed["dspecs"]

    # §9.B module map lives in the companion since Tier 2 (the main SPEC §9 is a
    # wiring-contract stub). Re-parse the companion for the module→line map; fall
    # back to the main SPEC's §9.B only if the companion is absent.
    if SUBSCRIBER_MATRIX_PATH.exists():
        matrix_parsed = parse_spec(SUBSCRIBER_MATRIX_PATH.read_text())
        if matrix_parsed["modules"]:
            parsed["modules"] = matrix_parsed["modules"]

    index_text = render(parsed)
    INDEX_PATH.write_text(index_text)
    print(
        f"wrote {INDEX_PATH.relative_to(REPO_ROOT)} "
        f"({len(parsed['sections'])} sections, "
        f"{len(parsed['dspecs'])} D-SPEC entries, "
        f"{len(parsed['modules'])} §9.B modules; "
        f"SPEC v{parsed['spec_version']}, {parsed['total_lines']:,} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
