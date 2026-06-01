"""SPEC §24.3 — Unified backup manifest (Arweave plane).

Per-Titan append-only event index at `data/backup_unified_manifest_{titan_id}.json`.
This file IS the index for crash recovery; losing it costs months of backup
history. Indefinite retention. Atomic-write per SPEC §11.H.2 with 2-generation
.bak retention.

Coexists with the L5 LOCAL plane manifest (`data/backups/local_diff_manifest_
{titan_id}.json`, shipped 2026-05-14): one BackupWorker owns both planes; two
manifests; two restore walk paths. See SPEC §24.1.

Schema (§24.3):
    {
      "titan_id": "T1",
      "schema_version": 1,
      "current_baseline_event_id": "uuid-of-most-recent-baseline",
      "current_baseline_date": "YYYY-MM-DD",
      "events": [
        {
          "event_id": "uuid4",
          "ts_unix": float,
          "type": "baseline" | "incremental",
          "baseline_trigger": "month_boundary" | "depth_cap" | null,
          "prev_event_id": "uuid4 | null",
          "personality": {tx_id, merkle_root, size_bytes, diff_mode,
                          diff_against_event_id, skipped_files, encryption_tier},
          "timechain":   {tx_id, merkle_root, size_bytes, diff_mode,
                          block_range, prev_offset_bytes},
          "soul":        {tx_id, merkle_root, size_bytes, diff_mode,
                          diff_against_event_id} | null,
          "zk_commit_tx": str,
          "zk_memo_prev_short": str
        }
      ]
    }

Rebase trigger (SPEC §24.2 — Maker decision Q3 2026-05-15): FIRST-WINS of
(1st of UTC month) OR (chain depth ≥ BACKUP_INCREMENTAL_MAX_CHAIN_DEPTH=30).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator, Optional

from titan_hcl._phase_c_constants import (
    BACKUP_INCREMENTAL_MAX_CHAIN_DEPTH,
    DATA_BACKUP_RETENTION_GENERATIONS,
)

logger = logging.getLogger(__name__)


# SPEC §24.3 — current schema version. Bump only when manifest event shape
# changes incompatibly (older readers can't decode). Additive fields are
# fine without bumping.
UNIFIED_MANIFEST_SCHEMA_VERSION = 1


# ── atomic write per SPEC §11.H.2 ─────────────────────────────────────────


def _atomic_write_json(path: str, data: dict,
                       keep_backups: int = DATA_BACKUP_RETENTION_GENERATIONS) -> None:
    """Write `data` (JSON-serialized) to `path` atomically per SPEC §11.H.2.

    Steps:
      1. Write `<path>.tmp` with data + fsync
      2. Rotate: `<path>.bak.prev` → DELETE; `<path>.bak` → `<path>.bak.prev`;
                 `<path>` → `<path>.bak`
      3. `rename(<path>.tmp, <path>)` (POSIX atomic on Linux)
      4. fsync(parent_dir_fd)

    A crash at any step leaves the previous valid `<path>` or its `.bak`
    intact — restore walks back per §11.H.4 boot integrity check.
    """
    payload = json.dumps(data, indent=2, sort_keys=False).encode("utf-8")
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"

    # 1. Write tmp + fsync
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Cleanup on failure — leave the existing path untouched
        with _suppress(FileNotFoundError):
            os.unlink(tmp)
        raise

    # 2. Rotate backups (only if target already exists)
    if keep_backups > 0 and os.path.exists(path):
        bak = path + ".bak"
        bak_prev = path + ".bak.prev"
        if keep_backups >= 2 and os.path.exists(bak):
            with _suppress(FileNotFoundError):
                os.replace(bak, bak_prev)
        with _suppress(FileNotFoundError):
            os.replace(path, bak)

    # 3. Atomic rename
    os.replace(tmp, path)

    # 4. fsync parent dir so the rename is durable across power loss
    try:
        dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Some filesystems (e.g. in-memory test fs) don't support dir fsync —
        # the rename itself is still atomic per POSIX; the dir fsync is a
        # durability hardening, not a correctness requirement.
        pass


class _suppress:
    """Inline context manager (avoid pulling contextlib for one use)."""
    def __init__(self, *exc_types):
        self.exc_types = exc_types

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return exc_type is not None and issubclass(exc_type, self.exc_types)


# ── manifest object ───────────────────────────────────────────────────────


class UnifiedManifest:
    """SPEC §24.3 unified backup manifest (Arweave plane) — per-Titan.

    Append-only events[] indexed for restore walk. Atomic-write per §11.H.2.
    """

    def __init__(self, titan_id: str, base_dir: str = "data"):
        self.titan_id = titan_id
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, f"backup_unified_manifest_{titan_id}.json")
        self._data: dict = self._empty_skeleton()

    # ── construction / IO ────────────────────────────────────────────────

    def _empty_skeleton(self) -> dict:
        return {
            "titan_id": self.titan_id,
            "schema_version": UNIFIED_MANIFEST_SCHEMA_VERSION,
            "current_baseline_event_id": None,
            "current_baseline_date": None,
            "events": [],
        }

    @classmethod
    def load(cls, titan_id: str, base_dir: str = "data") -> "UnifiedManifest":
        """Load manifest from disk (or create empty skeleton if absent).

        On JSON decode error, attempts to fall through to `.bak`, then
        `.bak.prev` per SPEC §11.H.4 boot integrity check. If all 3 fail
        to decode, raises — manifest corruption is load-bearing and must
        surface to the supervisor, not be silently re-initialized
        (initialization would orphan the on-chain ZK Vault hash chain).

        Schema-validation errors (titan_id mismatch / unknown schema version
        / non-list events) propagate IMMEDIATELY — these are not transient
        corruption recoverable from a .bak file but architectural drift /
        cross-titan contamination that must halt the supervisor.
        """
        obj = cls(titan_id=titan_id, base_dir=base_dir)
        primary_exists = os.path.exists(obj.path)
        for candidate in (obj.path, obj.path + ".bak", obj.path + ".bak.prev"):
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate) as f:
                    obj._data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(
                    "[UnifiedManifest] JSON decode error on %s: %s — trying fallback",
                    candidate, e)
                continue
            # JSON decoded — validate schema. Schema errors propagate as
            # hard ValueError (no .bak fallback for architectural drift).
            obj._validate_schema()
            if candidate != obj.path:
                logger.warning(
                    "[UnifiedManifest] Loaded from fallback %s (primary missing/corrupt)",
                    candidate)
            return obj
        # No file exists OR all 3 candidates failed JSON decode
        if primary_exists:
            raise ValueError(
                f"UnifiedManifest at {obj.path} is corrupted on all 3 generations "
                f"(primary + .bak + .bak.prev). Manual recovery required."
            )
        # File doesn't exist — fresh-start skeleton
        return obj

    def save(self) -> None:
        """Atomic-write the manifest to disk per SPEC §11.H.2."""
        self._validate_schema()
        _atomic_write_json(self.path, self._data)

    def _validate_schema(self) -> None:
        """Defensive schema check — catches malformed entries before they
        rot in the on-chain hash chain."""
        d = self._data
        if not isinstance(d, dict):
            raise ValueError("UnifiedManifest data must be a dict")
        if d.get("titan_id") != self.titan_id:
            raise ValueError(
                f"UnifiedManifest titan_id mismatch: file has {d.get('titan_id')!r}, "
                f"loader expected {self.titan_id!r} (cross-titan contamination)"
            )
        if d.get("schema_version") not in (None, UNIFIED_MANIFEST_SCHEMA_VERSION):
            raise ValueError(
                f"UnifiedManifest schema_version {d.get('schema_version')!r} unknown "
                f"(supported: {UNIFIED_MANIFEST_SCHEMA_VERSION})"
            )
        if not isinstance(d.get("events", []), list):
            raise ValueError("UnifiedManifest events must be a list")

    # ── read accessors ───────────────────────────────────────────────────

    @property
    def events(self) -> list:
        return self._data["events"]

    @property
    def current_baseline_event_id(self) -> Optional[str]:
        return self._data.get("current_baseline_event_id")

    @property
    def current_baseline_date(self) -> Optional[str]:
        return self._data.get("current_baseline_date")

    def get_latest_event(self) -> Optional[dict]:
        """Return the most recently appended event (or None if empty)."""
        if not self.events:
            return None
        return self.events[-1]

    def get_event(self, event_id: str) -> Optional[dict]:
        """Return the event with matching event_id (or None)."""
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def walk_chain(self, from_event_id: Optional[str] = None) -> Iterator[dict]:
        """Walk events newest → oldest via prev_event_id chain.

        If `from_event_id` is None, starts at the latest event. Yields each
        event in chain order. Stops when prev_event_id is None or the
        pointed-to event isn't found (chain break — restore halts).
        """
        cursor = self.get_event(from_event_id) if from_event_id else self.get_latest_event()
        while cursor is not None:
            yield cursor
            prev = cursor.get("prev_event_id")
            cursor = self.get_event(prev) if prev else None

    def get_baseline_for_event(self, event_id: str) -> Optional[dict]:
        """Walk backward from `event_id` to the most recent baseline event
        in its chain. Returns None if no baseline reachable (chain break)."""
        for ev in self.walk_chain(from_event_id=event_id):
            if ev.get("type") == "baseline":
                return ev
        return None

    def incrementals_since_baseline(self, baseline_event_id: Optional[str] = None) -> list:
        """Return list of incremental events between the given baseline
        (or current baseline if None) and the latest event, in chronological
        order. Used to compute chain depth for the rebase trigger."""
        target_baseline = baseline_event_id or self.current_baseline_event_id
        if not target_baseline:
            return []
        incs = []
        # walk_chain yields newest-first; collect incrementals until we
        # hit the baseline, then reverse for chronological order
        for ev in self.walk_chain():
            if ev.get("event_id") == target_baseline:
                break
            if ev.get("type") == "incremental":
                incs.append(ev)
        incs.reverse()
        return incs

    # ── append / rebase ──────────────────────────────────────────────────

    def append_event(self, event: dict) -> None:
        """Append a new event to the manifest in-memory. Does NOT persist —
        caller must call .save() once the on-chain ZK Vault commit lands.

        Validates: event_id present, type ∈ {baseline, incremental},
        prev_event_id matches the current latest event (or None if first).
        Sets current_baseline_event_id + current_baseline_date if baseline.
        """
        if "event_id" not in event:
            raise ValueError("event must have event_id")
        if event.get("type") not in ("baseline", "incremental"):
            raise ValueError(
                f"event type must be 'baseline' or 'incremental', "
                f"got {event.get('type')!r}"
            )
        latest = self.get_latest_event()
        expected_prev = latest["event_id"] if latest else None
        if event.get("prev_event_id") != expected_prev:
            raise ValueError(
                f"prev_event_id chain break: event prev_event_id={event.get('prev_event_id')!r} "
                f"but latest event_id={expected_prev!r}"
            )
        if event["type"] == "baseline":
            # Baselines reset the chain pointer + record the rebase date
            self._data["current_baseline_event_id"] = event["event_id"]
            ts = event.get("ts_unix", time.time())
            self._data["current_baseline_date"] = datetime.fromtimestamp(
                ts, tz=timezone.utc).strftime("%Y-%m-%d")
        self._data["events"].append(event)

    def should_rebase(self, now: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
        """Per SPEC §24.2 FIRST-WINS of (1st of UTC month) OR
        (chain depth ≥ BACKUP_INCREMENTAL_MAX_CHAIN_DEPTH).

        Returns (should_rebase: bool, reason: str | None). Reason is one of
        "month_boundary" / "depth_cap" / None.

        First-ever event (no current baseline) → (True, "first_event").
        """
        if not self.current_baseline_event_id:
            return (True, "first_event")

        now = now or datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        # Month-boundary check: today is the 1st AND we haven't already
        # rebased today
        if now.day == 1 and self.current_baseline_date != today_str:
            return (True, "month_boundary")

        # Depth-cap check
        depth = len(self.incrementals_since_baseline())
        if depth >= BACKUP_INCREMENTAL_MAX_CHAIN_DEPTH:
            return (True, "depth_cap")

        return (False, None)


# ── event builders ────────────────────────────────────────────────────────


def new_event_id() -> str:
    """Generate a new event_id (uuid4 hex)."""
    return uuid.uuid4().hex


def make_event(
    *,
    event_id: str,
    event_type: str,
    prev_event_id: Optional[str],
    baseline_trigger: Optional[str],
    personality: dict,
    timechain: dict,
    soul: Optional[dict] = None,
    zk_commit_tx: Optional[str] = None,
    zk_memo_prev_short: Optional[str] = None,
    ts_unix: Optional[float] = None,
) -> dict:
    """Construct a SPEC §24.3 event dict with required fields validated.

    `personality`, `timechain`: dicts with tx_id, merkle_root, size_bytes,
    diff_mode, plus subtype-specific fields per §24.3.
    `soul`: weekly events only — None otherwise.
    """
    if event_type not in ("baseline", "incremental"):
        raise ValueError(f"event_type must be 'baseline' or 'incremental', got {event_type!r}")
    if event_type == "baseline" and baseline_trigger not in (
        "month_boundary", "depth_cap", "first_event",
    ):
        raise ValueError(
            f"baseline event must have baseline_trigger ∈ "
            f"{{month_boundary, depth_cap, first_event}}, got {baseline_trigger!r}"
        )
    if event_type == "incremental" and baseline_trigger is not None:
        raise ValueError("incremental events must have baseline_trigger=None")

    for label, sub in (("personality", personality), ("timechain", timechain)):
        if not isinstance(sub, dict):
            raise ValueError(f"event.{label} must be a dict")
        for k in ("tx_id", "merkle_root", "size_bytes", "diff_mode"):
            if k not in sub:
                raise ValueError(f"event.{label} missing required field {k!r}")

    return {
        "event_id": event_id,
        "ts_unix": ts_unix if ts_unix is not None else time.time(),
        "type": event_type,
        "baseline_trigger": baseline_trigger,
        "prev_event_id": prev_event_id,
        "personality": personality,
        "timechain": timechain,
        "soul": soul,
        "zk_commit_tx": zk_commit_tx,
        "zk_memo_prev_short": zk_memo_prev_short,
    }
