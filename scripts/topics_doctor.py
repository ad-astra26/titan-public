#!/usr/bin/env python3
"""topics-doctor — automatic anti-drift check for the 5 canonical memory topic files.

The 5 `topic_*.md` files (infra / workflow_map / runtime_architecture / misc_rules /
claude_working_notes) name concrete paths + scripts. They rot silently when a script
is retired or a path moves (exactly what made the old session protocols dangerous).

This link-checks every repo path each topic names against the filesystem, and flags
any `verified:` frontmatter date older than the freshness window. Run at session close
(CLAUDE.md close step 5b) or any time.

  python scripts/topics_doctor.py            # check, human report
  python scripts/topics_doctor.py --strict   # exit 1 if any repo path is missing

Exit 0 = clean (or only warnings); 1 = a named repo path no longer exists (--strict).
"""
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

MEM_DIR = Path(os.path.expanduser(
    "~/.claude/projects/-home-antigravity-projects-titan/memory"))
REPO = Path(__file__).resolve().parent.parent  # repo root (scripts/..)
FRESH_DAYS = 30
TOPICS = ["topic_infra_stack", "topic_workflow_map", "topic_runtime_architecture",
          "topic_misc_rules", "topic_claude_working_notes"]

# A backtick token is a checkable REPO path if it contains '/' and either starts
# with a known repo dir or ends with a known source extension. Home/system paths
# (~/.titan, /etc) and not-yet-built BRAIN crates are skipped (planned/runtime).
REPO_PREFIXES = ("scripts/", "titan-docs/", "titan_hcl/", "titan-rust/", "data/")
SRC_EXT = (".py", ".sh", ".md", ".toml", ".json", ".rs")
SKIP = ("titan-rust/crates/brain",)  # planned, not built yet (BRAIN v1)


def _verified_age_days(text: str):
    m = re.search(r"verified:\s*(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    try:
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        return (date.today() - d).days
    except ValueError:
        return None


def _candidate_paths(text: str):
    """Backtick tokens that look like checkable repo file paths."""
    out = set()
    for tok in re.findall(r"`([^`]+)`", text):
        tok = tok.strip().split()[0].rstrip(".,;:)")  # first word, strip trailing punct
        if "/" not in tok or any(tok.startswith(s) for s in SKIP):
            continue
        if any(c in tok for c in "*{%"):  # globs / brace-patterns / printf-templates
            continue
        if tok.endswith("/"):  # directories
            continue
        if tok.startswith(("~", "/")) or tok.startswith(REPO_PREFIXES) or tok.endswith(SRC_EXT):
            out.add(tok)
    return sorted(out)


def _resolve(p: str) -> Path:
    """Resolve a token to an absolute path (home/abs expand; else repo-relative)."""
    if p.startswith("~"):
        return Path(os.path.expanduser(p))
    if p.startswith("/"):
        return Path(p)
    return REPO / p


def main():
    strict = "--strict" in sys.argv
    missing_total = 0
    stale_total = 0
    print(f"topics-doctor — checking {len(TOPICS)} topic files (repo={REPO})\n")
    for name in TOPICS:
        f = MEM_DIR / f"{name}.md"
        if not f.exists():
            print(f"  ✗ {name}.md  — TOPIC FILE MISSING")
            missing_total += 1
            continue
        text = f.read_text()
        age = _verified_age_days(text)
        age_str = "no verified: date" if age is None else f"verified {age}d ago"
        stale = age is not None and age > FRESH_DAYS
        if stale:
            stale_total += 1
        flag = "⚠ STALE" if stale else "ok"
        paths = _candidate_paths(text)
        missing = [p for p in paths if not _resolve(p).exists()]
        missing_total += len(missing)
        print(f"  {'⚠' if (missing or stale) else '✓'} {name}.md  "
              f"({len(paths)} paths checked, {age_str}) [{flag}]")
        for p in missing:
            print(f"      ✗ missing: {p}")
    print()
    if missing_total == 0 and stale_total == 0:
        print("  ✓ all clean — every named path exists, all verified dates fresh")
    else:
        print(f"  {missing_total} missing path(s), {stale_total} stale topic(s) "
              f"(>{FRESH_DAYS}d) — update the topic file + bump its verified: date")
    sys.exit(1 if (strict and missing_total) else 0)


if __name__ == "__main__":
    main()
