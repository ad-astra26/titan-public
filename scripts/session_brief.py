#!/usr/bin/env python3
"""session_brief.py — emit auto-fill header for SESSION_*.md files.

The session-close protocol's `arch_map session-close` command produces a
SESSION_*.md template with TODO blocks. Claude fills the narrative sections
(work-done summary, architectural decisions, next-priorities) — but the
mechanical fields (commits, diff stats, files touched) are perfectly
auto-fillable from git on the session worktree branch.

This script emits a "Session metrics" block to prepend to (or paste into)
the SESSION_*.md file:

  - Branch + base merge-base
  - Time span (first → last commit timestamps + duration)
  - N commits with subject lines
  - Diffstat (+lines / -lines / files changed)
  - Files touched (grouped by directory)
  - Empty narrative sections marked TODO for Claude to fill

Usage:
  python scripts/session_brief.py                       # current branch vs titan-v6
  python scripts/session_brief.py --branch FOO          # explicit branch
  python scripts/session_brief.py --base main           # explicit base
  python scripts/session_brief.py --output FILE.md      # write to file (else stdout)
  python scripts/session_brief.py --prepend FILE.md     # prepend to existing file

Wired into session_close_protocol Step 4 (SESSION_*.md generation) — run
alongside `arch_map session-close` so the SESSION file lands with all
mechanical fields filled.

Idempotent for a given branch+base pair (deterministic output).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True,
                          cwd=REPO_ROOT, check=False).stdout.strip()


def current_branch() -> str:
    return sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "HEAD"


def find_session_base(branch: str, base: str) -> str:
    """Find the base-SHA that the session branched FROM.

    Robust against the post-merge case where the session has already been
    merged into base (the trivial `git merge-base` would then return the
    latest merge point, hiding the session's commits). Order of preference:

      1. `git rev-list branch --not base | tail -1`'s parent — works
         pre-merge by listing commits exclusive to the branch.
      2. `git reflog show branch | tail -1` — first reflog entry is
         typically the create-branch event; its commit-after-action is
         the base.
      3. `git merge-base --fork-point base branch` — relies on reflog
         too, may miss if reflog rotated.
      4. plain `git merge-base branch base` — fallback (may give wrong
         answer post-merge).
    """
    # 1: commits exclusive to branch
    exclusive = sh("git", "rev-list", branch, f"^{base}")
    if exclusive:
        oldest = exclusive.splitlines()[-1]
        parent = sh("git", "rev-parse", f"{oldest}^")
        if parent:
            return parent
    # 2: reflog (the original create-branch entry)
    reflog = sh("git", "reflog", "show", branch, "--format=%H %gs")
    for line in reflog.splitlines():
        if "branch: Created from" in line or "Created from" in line:
            sha = line.split(" ", 1)[0]
            if sha:
                return sha
    # 3: --fork-point
    fp = sh("git", "merge-base", "--fork-point", base, branch)
    if fp:
        return fp
    # 4: plain merge-base
    return sh("git", "merge-base", branch, base)


def merge_base(branch: str, base: str) -> str:
    """Backward-compat wrapper; new code uses find_session_base."""
    return find_session_base(branch, base)


def commits_since(base_sha: str, branch: str) -> list[dict]:
    """Return list of {sha, subject, author_iso, files_changed} for branch since base_sha."""
    raw = sh("git", "log", f"{base_sha}..{branch}",
             "--pretty=format:%H|%s|%aI", "--no-merges")
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        sha, subject, iso = parts
        commits.append({"sha": sha[:8], "subject": subject, "iso": iso})
    return commits


def diff_stats(base_sha: str, branch: str) -> dict:
    """Return {insertions, deletions, files_changed, files: [list]}."""
    raw = sh("git", "diff", "--shortstat", f"{base_sha}..{branch}")
    files_raw = sh("git", "diff", "--name-only", f"{base_sha}..{branch}")
    insertions = deletions = files_changed = 0
    # e.g. " 23 files changed, 1058 insertions(+), 832 deletions(-)"
    import re
    m = re.search(r"(\d+)\s+files? changed", raw)
    if m:
        files_changed = int(m.group(1))
    m = re.search(r"(\d+)\s+insertions?\(\+\)", raw)
    if m:
        insertions = int(m.group(1))
    m = re.search(r"(\d+)\s+deletions?\(-\)", raw)
    if m:
        deletions = int(m.group(1))
    files = [f for f in files_raw.splitlines() if f.strip()]
    return {"insertions": insertions, "deletions": deletions,
            "files_changed": files_changed, "files": files}


def group_files_by_dir(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parts = f.split("/", 1)
        if len(parts) == 1:
            groups["(root)"].append(f)
        else:
            top = parts[0]
            groups[top].append(f)
    return dict(sorted(groups.items()))


def fmt_duration(first_iso: str, last_iso: str) -> str:
    """Return human-readable duration between two ISO timestamps."""
    try:
        a = datetime.fromisoformat(first_iso)
        b = datetime.fromisoformat(last_iso)
        delta = abs(b - a)
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"
    except Exception:
        return "?"


def render(branch: str, base: str, override_base_sha: str | None = None) -> str:
    base_sha = override_base_sha or find_session_base(branch, base)
    if not base_sha:
        print(f"error: could not find session-base between {branch} and {base}", file=sys.stderr)
        sys.exit(1)
    base_sha_short = base_sha[:8]
    base_desc = sh("git", "log", "-1", "--pretty=format:%s", base_sha) or "?"
    base_iso = sh("git", "log", "-1", "--pretty=format:%aI", base_sha) or ""

    commits = commits_since(base_sha, branch)
    stats = diff_stats(base_sha, branch)

    if not commits:
        return f"# Session brief — no commits on {branch} since {base} merge-base {base_sha_short}\n"

    first_iso = commits[-1]["iso"]  # git log is newest-first; oldest commit is last in list
    last_iso = commits[0]["iso"]
    duration = fmt_duration(first_iso, last_iso)

    date_label = first_iso[:10]
    out: list[str] = []
    out.append(f"# SESSION {date_label} — {branch.split('/', 1)[-1] if '/' in branch else branch}")
    out.append("")
    out.append(f"**Auto-filled by `scripts/session_brief.py` from git on {branch}.** Narrative sections below are Claude's to fill.")
    out.append("")
    out.append("## Session metrics (auto)")
    out.append("")
    out.append(f"- **Branch:** `{branch}`")
    out.append(f"- **Base merge-point:** `{base}` @ `{base_sha_short}` ({base_iso[:10]}) — _{base_desc}_")
    out.append(f"- **Time span:** {first_iso[:16].replace('T', ' ')} UTC → {last_iso[:16].replace('T', ' ')} UTC ({duration} active)")
    out.append(f"- **Commits:** {len(commits)}")
    out.append(f"- **Diffstat:** +{stats['insertions']:,} / −{stats['deletions']:,} across {stats['files_changed']} files")
    out.append("")
    out.append("### Commits (newest → oldest)")
    out.append("")
    for c in commits:
        out.append(f"- `{c['sha']}` — {c['subject']}")
    out.append("")
    out.append("### Files touched (grouped)")
    out.append("")
    groups = group_files_by_dir(stats["files"])
    for top, files in groups.items():
        out.append(f"**`{top}/`** ({len(files)} files)")
        for f in files[:20]:
            out.append(f"- `{f}`")
        if len(files) > 20:
            out.append(f"- _… and {len(files) - 20} more_")
        out.append("")

    # Per session_close_protocol opt #6 (2026-05-17): TODO sections are
    # owned by the lower template that `arch_map.py session-close` writes.
    # session_brief.py provides ONLY mechanical/auto-derived metrics here
    # to avoid duplication. The lower template's `## Architectural
    # Decisions`, `## rFPs / Tasks Touched`, `## Work Done`, `## Next
    # Session Priorities` are the canonical hand-fill targets.
    out.append("---")
    out.append("")
    out.append("_Generated by `scripts/session_brief.py` (mechanical metrics only). "
               "Narrative sections — Summary, Architectural Decisions (with "
               "auto-extracted AskUserQuestion answers), rFPs touched, Work "
               "Done, Next Priorities — are below in the lower template "
               "written by `arch_map.py session-close`._")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default=None, help="session branch (default: current)")
    ap.add_argument("--base", default="titan-v6", help="merge-base reference (default: titan-v6)")
    ap.add_argument("--base-sha", default=None,
                    help="explicit base SHA (overrides auto-detect; useful after the session has been merged into base)")
    ap.add_argument("--output", default=None, help="write to FILE (else stdout)")
    ap.add_argument("--prepend", default=None, help="prepend output to existing FILE")
    args = ap.parse_args()

    branch = args.branch or current_branch()
    text = render(branch, args.base, override_base_sha=args.base_sha)

    if args.prepend:
        p = Path(args.prepend)
        existing = p.read_text() if p.exists() else ""
        p.write_text(text + "\n" + existing)
        print(f"prepended to {p} ({len(text)} chars)")
        return 0
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output} ({len(text)} chars)")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
