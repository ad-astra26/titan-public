"""titan_hcl/core/self_inspect.py — scaffolding self-inspection (RFP §7.P5 / INV-SD-9).

Read-only, bounded to Titan's OWN tree. Produces self-observations from his real
substrate so the daily soul-diary is grounded in it and so the observations are
promoted as `domain="self"` self-Engrams he can recall (seeding BRAIN_DOMAIN_SELF;
this module does NOT build the BRAIN consumer).

Two BROAD sources (drafting §6.1):
  (a) journal errors  — recent ERROR/WARN/Traceback from his kernel journal
                        (`journalctl -u titan-<id>.service`);
  (b) error→code      — for an error pointing at `file.py:NN`, the code around it,
                        read read-only + traversal-guarded to the repo root;
  + a bounded structural glance at his own source tree.

Everything soft-fails to empty — self-inspection never blocks the diary cascade
(INV-SD-13). Nothing here writes; all reads are bounded to `repo_root`.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

# Bounded so a noisy journal / huge file can never blow the diary budget.
_DEFAULT_JOURNAL_LINES = 400
_MAX_ERROR_OBSERVATIONS = 8
_MAX_CORRELATIONS = 3
_CODE_CONTEXT_LINES = 6
_MAX_SOURCE_BYTES = 4000
_JOURNAL_TIMEOUT_S = 5.0

# ERROR/WARN/Traceback markers (case-insensitive) — the lines worth reflecting on.
_ERROR_MARKERS = ("error", "warning", "warn", "critical", "traceback",
                  "exception", "failed", "fatal")
# `path/to/file.py:123` (optionally `, line 123`) — the code pointer in a trace.
_FILE_LINE_RE = re.compile(r'([\w./\-]+\.py)["\']?,?\s*(?:line\s+)?:?\s*(\d+)')


def _within_repo(path: str, repo_root: str) -> bool:
    """True iff `path` resolves to a location inside `repo_root` (symlink-safe)."""
    try:
        rp = os.path.realpath(path)
        rr = os.path.realpath(repo_root)
        return rp == rr or rp.startswith(rr + os.sep)
    except Exception:  # noqa: BLE001
        return False


def read_journal_lines(titan_id: str, *, n: int = _DEFAULT_JOURNAL_LINES,
                       runner=None) -> list[str]:
    """Return the last `n` lines of Titan's kernel journal
    (`journalctl -u titan-<titan_id>.service`). `runner` is an injectable
    `(argv) -> str` (default: subprocess) so tests need no real journal. Soft-fails
    to [] (no journalctl / not a systemd box / timeout / nonzero)."""
    service = f"titan-{titan_id}.service"
    argv = ["journalctl", "-u", service, "-n", str(int(n)), "--no-pager",
            "--output", "cat"]
    try:
        if runner is not None:
            raw = runner(argv) or ""
        else:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                argv, capture_output=True, text=True,
                timeout=_JOURNAL_TIMEOUT_S, check=False)
            if proc.returncode != 0:
                logger.info("[self_inspect] journalctl rc=%s for %s — no journal",
                            proc.returncode, service)
                return []
            raw = proc.stdout or ""
    except FileNotFoundError:
        logger.info("[self_inspect] journalctl unavailable — skipping journal read")
        return []
    except Exception as e:  # noqa: BLE001
        logger.info("[self_inspect] journal read failed: %s", e)
        return []
    return [ln for ln in raw.splitlines() if ln.strip()]


def extract_error_observations(journal_lines: list[str]) -> list[str]:
    """The recent error/warning lines worth reflecting on (newest-last preserved,
    deduped, bounded). Lower-cased marker match; original line text kept."""
    out: list[str] = []
    seen: set[str] = set()
    for ln in journal_lines:
        low = ln.lower()
        if any(m in low for m in _ERROR_MARKERS):
            key = ln.strip()[:160]
            if key and key not in seen:
                seen.add(key)
                out.append(ln.strip())
    # Keep the most recent ones (journal is chronological).
    return out[-_MAX_ERROR_OBSERVATIONS:]


def correlate_to_code(error_lines: list[str], *, repo_root: str,
                      max_correlations: int = _MAX_CORRELATIONS,
                      context_lines: int = _CODE_CONTEXT_LINES) -> list[dict]:
    """For errors that name `file.py:NN`, read the code window around line NN —
    read-only + bounded to `repo_root` (path-traversal guarded). Returns
    `[{file, line, snippet}]`. Per-ref soft-fail; dedups by (file, line)."""
    correlations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for ln in error_lines:
        for m in _FILE_LINE_RE.finditer(ln):
            rel_or_abs, line_s = m.group(1), m.group(2)
            try:
                line_no = int(line_s)
            except ValueError:
                continue
            # Resolve: absolute-in-repo, or repo-relative; else try basename match
            # is intentionally NOT done (keep it bounded + literal).
            cand = (rel_or_abs if os.path.isabs(rel_or_abs)
                    else os.path.join(repo_root, rel_or_abs))
            if not _within_repo(cand, repo_root) or not os.path.isfile(cand):
                continue
            key = (os.path.realpath(cand), line_no)
            if key in seen:
                continue
            seen.add(key)
            snippet = _read_code_window(cand, line_no, context_lines)
            if snippet:
                correlations.append({
                    "file": os.path.relpath(os.path.realpath(cand),
                                            os.path.realpath(repo_root)),
                    "line": line_no, "snippet": snippet})
            if len(correlations) >= max_correlations:
                return correlations
    return correlations


def _read_code_window(path: str, line_no: int, context: int) -> str:
    """Read `context` lines either side of `line_no` (1-based). Soft-fail → ""."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_inspect] code read failed for %s: %s", path, e)
        return ""
    lo = max(0, line_no - 1 - context)
    hi = min(len(lines), line_no + context)
    return "".join(lines[lo:hi]).strip()


def read_self_source(rel_path: str, *, repo_root: str,
                     max_bytes: int = _MAX_SOURCE_BYTES) -> str:
    """Read one of his own source files (read-only, bounded to `repo_root`,
    traversal-guarded, byte-capped). Soft-fail → ""."""
    cand = os.path.join(repo_root, rel_path)
    if not _within_repo(cand, repo_root) or not os.path.isfile(cand):
        return ""
    try:
        with open(cand, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_inspect] self-source read failed for %s: %s", cand, e)
        return ""


def glance_self_structure(repo_root: str, *, pkg: str = "titan_hcl") -> dict:
    """A bounded, read-only structural glance at his own codebase — top-level
    package dirs + a .py file count. Cheap; gives the diary a sense of his own
    shape without reading content. Soft-fail → {}."""
    base = os.path.join(repo_root, pkg)
    if not _within_repo(base, repo_root) or not os.path.isdir(base):
        return {}
    try:
        subdirs = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and not d.startswith("__"))
        py_count = 0
        for _root, _dirs, files in os.walk(base):
            py_count += sum(1 for f in files if f.endswith(".py"))
        return {"package": pkg, "subsystems": subdirs[:24], "py_files": py_count}
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_inspect] structure glance failed: %s", e)
        return {}


def gather_self_observations(titan_id: str, *, repo_root: str,
                             runner=None) -> dict:
    """Run the BROAD self-inspection (drafting §6.1): journal errors +
    error→code correlation + a structural glance. Read-only, bounded, soft-fail.
    Returns {"journal_errors": [...], "correlations": [...], "structure": {...}}."""
    journal = read_journal_lines(titan_id, runner=runner)
    errors = extract_error_observations(journal)
    correlations = correlate_to_code(errors, repo_root=repo_root)
    structure = glance_self_structure(repo_root)
    return {
        "journal_errors": errors,
        "correlations": correlations,
        "structure": structure,
    }


def summarize_observations(obs: dict, *, max_chars: int = 600) -> str:
    """A compact first-person-ready summary of the self-inspection — fed into the
    diary GATHER bundle (grounding) AND promoted as the `domain="self"`
    self-inspect thought. Empty string when nothing was observed."""
    errors = obs.get("journal_errors") or []
    corr = obs.get("correlations") or []
    if not errors and not corr:
        return ""
    parts: list[str] = []
    if errors:
        # One representative line + a count (the prose, not a log dump).
        head = errors[-1]
        parts.append(f"{len(errors)} error/warning(s) passed through my substrate"
                     f" (latest: {head[:140]})")
    if corr:
        files = ", ".join(f"{c['file']}:{c['line']}" for c in corr)
        parts.append(f"I looked at the code it pointed to ({files})")
    return ". ".join(parts)[:max_chars]
