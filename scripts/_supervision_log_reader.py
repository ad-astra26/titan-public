"""_supervision_log_reader — Python reader for `data/supervision.jsonl`.

Used by `arch_map phase-c supervision-log` + `arch_map phase-c watch-escalations`
per SPEC §11.E + §11.G.4 + §20.4 +
PLAN_microkernel_phase_c_s2_kernel.md §12.8.

The titan-kernel-rs binary writes JSONL events with rotating archives
`supervision.jsonl`, `supervision.jsonl.1`, ..., `supervision.jsonl.10`.
Each line is a single JSON object whose `kind` field is one of:
  CHILD_STARTED, CHILD_EXITED, CHILD_RESTART, CHILD_DEPENDENCY_BLOCKED,
  CHILD_DEPENDENCY_UNBLOCKED, SUPERVISION_ESCALATION,
  SUPERVISION_ESCALATION_RESPONSE.

This reader is read-only — never writes the file. The kernel-rs binary is
the sole writer per SPEC §11.E.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional


DEFAULT_LOG_PATH = Path("data/supervision.jsonl")
DEFAULT_TAIL_POLL_INTERVAL_S = 0.5
ESCALATION_KINDS = frozenset(
    {"SUPERVISION_ESCALATION", "SUPERVISION_ESCALATION_RESPONSE"}
)


@dataclass(frozen=True)
class SupervisionFilter:
    """Filter criteria for `iter_supervision_log`."""

    child: Optional[str] = None
    reason: Optional[str] = None
    supervisor: Optional[str] = None
    kind: Optional[str] = None
    since: Optional[timedelta] = None

    def matches(self, event: dict) -> bool:
        if self.kind is not None and event.get("kind") != self.kind:
            return False
        if self.child is not None and event.get("child") != self.child:
            return False
        if self.reason is not None and event.get("reason") != self.reason:
            return False
        if self.supervisor is not None and event.get("supervisor") != self.supervisor:
            return False
        if self.since is not None:
            ts_str = event.get("ts")
            if ts_str is None:
                return False
            try:
                # ISO 8601 (e.g., "2026-04-29T13:42:01.123456789Z")
                ts = _parse_iso8601(ts_str)
            except ValueError:
                return False
            if datetime.now(tz=timezone.utc) - ts > self.since:
                return False
        return True


def _parse_iso8601(ts_str: str) -> datetime:
    """Parse the ISO 8601 timestamp written by the Rust supervision logger.

    Rust writes nanosecond precision with trailing 'Z'. Python's fromisoformat
    only handles up to microseconds, so trim nanoseconds → microseconds.
    """
    s = ts_str.rstrip("Z")
    # Trim fractional seconds beyond 6 digits (microsecond precision)
    if "." in s:
        head, frac = s.rsplit(".", 1)
        # Keep only digits in frac
        digits = "".join(c for c in frac if c.isdigit())
        s = f"{head}.{digits[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _archive_paths(log_path: Path) -> Iterable[Path]:
    """Yield archive paths in chronological order (oldest first), skipping
    ones that don't exist. The active log is yielded last.
    """
    parent = log_path.parent
    stem = log_path.name  # e.g., "supervision.jsonl"
    # Archives are .jsonl.10 (oldest) → .jsonl.1 (newest archive) → .jsonl (active)
    archives = []
    for i in range(10, 0, -1):
        p = parent / f"{stem}.{i}"
        if p.exists():
            archives.append(p)
    archives.append(log_path)
    return archives


def iter_supervision_log(
    log_path: Path = DEFAULT_LOG_PATH,
    *,
    child: Optional[str] = None,
    reason: Optional[str] = None,
    supervisor: Optional[str] = None,
    kind: Optional[str] = None,
    since: Optional[timedelta] = None,
    include_archives: bool = True,
) -> Iterator[dict]:
    """Read supervision.jsonl (+ rotated archives) and yield matching events.

    Args:
        log_path: path to the active supervision.jsonl. Archives are
            inferred as `<log_path>.1` ... `<log_path>.10`.
        child: only yield events whose `child` field matches.
        reason: only yield events whose `reason` field matches.
        supervisor: only yield events whose `supervisor` field matches.
        kind: only yield events whose `kind` field matches.
        since: only yield events whose `ts` is within this duration of now.
        include_archives: when True (default) walks `<log_path>.<N>` archives
            in chronological order (oldest first). When False, only the
            active log is read.

    Lines that fail to parse as JSON are skipped silently. The reader never
    writes the file — kernel-rs is the sole writer per SPEC §11.E.
    """
    flt = SupervisionFilter(
        child=child, reason=reason, supervisor=supervisor, kind=kind, since=since
    )
    paths = _archive_paths(log_path) if include_archives else [log_path]
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if flt.matches(event):
                        yield event
        except OSError:
            # File rotated mid-read or unreadable — skip silently
            continue


def watch_escalations(
    log_path: Path = DEFAULT_LOG_PATH,
    *,
    poll_interval_s: float = DEFAULT_TAIL_POLL_INTERVAL_S,
) -> Iterator[dict]:
    """Live-tail the supervision log and yield escalation events.

    Yields events whose `kind` is `SUPERVISION_ESCALATION` or
    `SUPERVISION_ESCALATION_RESPONSE` per SPEC §11.B.1 + §11.G.4. Loops
    forever — caller is expected to break on signal or external timeout.

    On rotation (file shrinks or inode changes), seeks back to the start.
    """
    last_inode: Optional[int] = None
    last_pos: int = 0
    while True:
        try:
            stat = log_path.stat()
        except FileNotFoundError:
            time.sleep(poll_interval_s)
            continue
        inode = stat.st_ino
        size = stat.st_size
        if last_inode is not None and (inode != last_inode or size < last_pos):
            # Rotated — start over
            last_pos = 0
        last_inode = inode
        try:
            with log_path.open("r", encoding="utf-8") as fh:
                fh.seek(last_pos)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("kind") in ESCALATION_KINDS:
                        yield event
                last_pos = fh.tell()
        except OSError:
            pass
        time.sleep(poll_interval_s)
