#!/usr/bin/env python3
"""One-time backfill: add audit fields to every active OBS gate.

Codified 2026-04-22 per memory/feedback_observable_persistence_audit.md.
Parses titan-docs/OBSERVABLES.md, inserts a 3-line audit stanza
(evidence_source / retention_horizon / persistence_verification) right
after each active gate's Status line.

Heuristics:
- evidence_source: extracted from the first `curl`/`grep`/`stat`/`cat`/
  `ls`/`python3` invocation in the gate's Check-command code fence.
- retention_horizon: computed from Hard deadline - Opened (falls back to
  "soak window per gate title" when dates are unparseable).
- persistence_verification: classified by evidence_source:
    - /tmp/titan*_brain.log → `acceptable_loss=legacy_pre_rule`
      (marks as retrofit-on-touch — NEXT edit must ship a writer)
    - curl /v4/... → `persisted_by=endpoint_snapshot` (point-in-time state)
    - data/* or /home/... actual file → `persisted_by=<path>`
    - ssh grep → `acceptable_loss=legacy_pre_rule` (remote log)
    - /v4/meta-cgn, meta_stats.json → `persisted_by=meta_stats.json`

Re-run is idempotent: existing **Evidence source**, **Retention horizon**,
**Persistence verification** lines in a gate body cause that gate to be
skipped.

This script is executed ONCE for the backfill commit, then kept under
scripts/ for archaeology. `arch_map observables --audit` is the enduring
enforcement tool.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
OBS = ROOT / "titan-docs" / "OBSERVABLES.md"


# ── Classifiers ──────────────────────────────────────────────────────

BRAIN_LOG_RX = re.compile(r"/tmp/titan\w*_?brain\.log|/tmp/titan_agent\.log")
SSH_RX = re.compile(r"ssh\s+root@")
CURL_RX = re.compile(r"curl\s+[^\s]*://[^\s/]+(/v4/[\w\-]+)")
# Data files only — skip scripts/, test_env/, nginx conf paths (noise)
DATA_FILE_RX = re.compile(
    r"(data/[\w/_\-\.]+|/home/antigravity/projects/titan/data/[\w/_\-\.]+)"
)
# arch_map subcommands are a common evidence source — capture the subcommand
ARCHMAP_RX = re.compile(r"arch_map\.py\s+([\w\-]+)")


def classify_evidence(check_block: str) -> tuple[str, str]:
    """Return (evidence_source, persistence_verification) from Check-command block.

    Preference order:
      1. Explicit data/ file path (truly persistent)
      2. curl /v4/... endpoint (point-in-time state)
      3. arch_map <subcommand> (wraps an endpoint; classify as snapshot)
      4. Local brain-log (rotation risk → legacy-pre-rule)
      5. Remote brain-log via ssh (same)
      6. Fallback unclassified
    """
    if not check_block:
        return (
            "(no Check command — fill during retrofit)",
            "acceptable_loss=legacy_pre_rule — retrofit-on-touch",
        )

    # 1. Explicit data file path (real persistence)
    fm = DATA_FILE_RX.search(check_block)
    if fm:
        path = fm.group(1)
        return (path, f"persisted_by={path}")

    # 2. Endpoint snapshot
    cm = CURL_RX.search(check_block)
    if cm:
        endpoint = cm.group(1)
        return (
            f"endpoint: {endpoint}",
            f"persisted_by=endpoint_snapshot ({endpoint}; retrofit to TSV "
            "if time-series needed)",
        )

    # 3. arch_map subcommand (wraps an endpoint)
    am = ARCHMAP_RX.search(check_block)
    if am:
        sub = am.group(1)
        return (
            f"arch_map {sub} (wraps live endpoint query)",
            "persisted_by=endpoint_snapshot via arch_map — retrofit to TSV "
            "if time-series needed",
        )

    # 4. Local brain-log
    if BRAIN_LOG_RX.search(check_block):
        return (
            "/tmp/titan*_brain.log (log rotation risk per "
            "feedback_observable_persistence_audit.md)",
            "acceptable_loss=legacy_pre_rule — retrofit-on-touch to ship "
            "per-gate TSV writer",
        )

    # 5. Remote brain-log via ssh
    if SSH_RX.search(check_block):
        return (
            "ssh remote brain-log grep",
            "acceptable_loss=legacy_pre_rule — retrofit-on-touch",
        )

    # 6. Fallback
    return (
        "(unclassified — fill during retrofit)",
        "acceptable_loss=legacy_pre_rule — retrofit-on-touch",
    )


# ── Retention horizon derivation ─────────────────────────────────────

DATE_RX = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}):(\d{2})(?:\s*UTC)?)?"
)


def parse_dt(s: str) -> datetime | None:
    m = DATE_RX.search(s)
    if not m:
        return None
    try:
        if m.group(2):
            return datetime(
                int(m.group(1)[:4]),
                int(m.group(1)[5:7]),
                int(m.group(1)[8:10]),
                int(m.group(2)),
                int(m.group(3)),
                tzinfo=timezone.utc,
            )
        return datetime(
            int(m.group(1)[:4]),
            int(m.group(1)[5:7]),
            int(m.group(1)[8:10]),
            tzinfo=timezone.utc,
        )
    except (ValueError, TypeError):
        return None


def derive_retention(body: str) -> str:
    opened_m = re.search(r"\*\*Opened:\*\*\s*([^\n]+)", body)
    deadline_m = re.search(r"\*\*Hard deadline:\*\*\s*([^\n]+)", body)
    earliest_m = re.search(r"\*\*Earliest decision:\*\*\s*([^\n]+)", body)

    opened = parse_dt(opened_m.group(1)) if opened_m else None
    deadline = parse_dt(deadline_m.group(1)) if deadline_m else None
    earliest = parse_dt(earliest_m.group(1)) if earliest_m else None

    if opened and deadline:
        delta = deadline - opened
        hours = int(delta.total_seconds() // 3600)
        if hours >= 168:
            return f"{hours // 168}d (hard deadline — opened)"
        if hours >= 48:
            return f"{hours // 24}d (hard deadline — opened)"
        return f"{hours}h (hard deadline — opened)"
    if opened and earliest:
        delta = earliest - opened
        hours = int(delta.total_seconds() // 3600)
        return f"{hours}h (earliest decision — opened)"
    # Fallback: look for "24h" / "7d" / "30d" tokens in title or body
    dur_m = re.search(r"(\d+)\s*(h|d|day|hour)", body)
    if dur_m:
        return dur_m.group(0)
    return "(unset — fill during retrofit)"


# ── Rewrite ──────────────────────────────────────────────────────────

GATE_HEADER_RX = re.compile(r"^### (OBS-[\w\-]+)\s+—\s+(.+?)$", re.MULTILINE)
CHECK_FENCE_RX = re.compile(
    r"\*\*Check command:\*\*\s*\n\s*```(?:bash)?\s*\n(.+?)\n\s*```",
    re.DOTALL,
)
STATUS_LINE_RX = re.compile(r"^- \*\*Status:\*\*[^\n]*\n", re.MULTILINE)
CLOSED_STATUS_RX = re.compile(
    r"(✅ PASSED|⚠ SUPERSEDED|\*\*SUPERSEDED\*\*|Status:\*\*\s*✅|Status:\*\*\s*⚠)",
    re.IGNORECASE,
)
ALREADY_BACKFILLED_RX = re.compile(
    r"\*\*Evidence source\*\*", re.IGNORECASE
)


def process_gate_body(body: str) -> str:
    # Skip closed gates (first 1500 chars = status region)
    if CLOSED_STATUS_RX.search(body[:1500]):
        return body
    # Skip already-backfilled gates
    if ALREADY_BACKFILLED_RX.search(body):
        return body

    check_m = CHECK_FENCE_RX.search(body)
    check_block = check_m.group(1) if check_m else ""
    evidence, persistence = classify_evidence(check_block)
    retention = derive_retention(body)

    audit_stanza = (
        f"- **Evidence source:** {evidence}\n"
        f"- **Retention horizon:** {retention}\n"
        f"- **Persistence verification:** {persistence}\n"
    )

    # Insert right after the Status line
    m = STATUS_LINE_RX.search(body)
    if not m:
        # No status line — prepend at start of body
        return audit_stanza + body

    return body[: m.end()] + audit_stanza + body[m.end() :]


def main() -> int:
    if not OBS.exists():
        print(f"not found: {OBS}", file=sys.stderr)
        return 2
    text = OBS.read_text()

    # Split into pre-first-gate + gate sections + tail.
    parts: list[str] = []
    last = 0
    matches = list(GATE_HEADER_RX.finditer(text))
    if not matches:
        print("no gates found", file=sys.stderr)
        return 1

    parts.append(text[: matches[0].start()])
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        gate_section = text[start:end]
        parts.append(process_gate_body(gate_section))

    OBS.write_text("".join(parts))
    print(f"backfill applied to {OBS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
