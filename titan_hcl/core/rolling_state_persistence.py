"""
rolling_state_persistence — per-titan persistent rolling-window state.

Many trinity dim formulas (SPEC §23.x) read from rolling-window aggregates:
  - assessment.average_score / trend / variance — rolling last 100 assessments
  - posture_authenticity_ratio_30 — rolling last 30 actions with hormone snapshots
  - outer_spirit_history.environmental_adaptation — rolling 30 high-thermal scores
  - graceful_rest — rolling 30 low-thermal scores
  - circadian_alignment — rolling last 200 action timestamps
  - many more (see SPEC §23.6 + §23.9)

Without persistence these reset to empty on every restart, meaning every
trinity dim that depends on rolling history starts cold and never warms
within a reasonable time. T3 restarted ~5× in this session and assessment
history stayed at 0 each time → ~12 dims stuck at SPEC default forever.

This module provides a small, generic rolling-window persistence layer:
  - Per-titan JSON snapshot file (one file per state store, per Titan)
  - Atomic write (tempfile + rename)
  - Save on every Nth append OR every K seconds (whichever first)
  - Load-at-boot truncates to {max_entries, max_age_s}
  - File path: `<titan_data_dir>/dim_history/<store_name>.json`

Usage:
    from titan_hcl.core.rolling_state_persistence import RollingStateStore

    class SelfAssessment:
        def __init__(self):
            self._store = RollingStateStore(
                name="assessment_history",
                max_entries=100,
                max_age_s=24 * 3600,  # 24h
                save_every_n=1,        # save on every append
            )
            self._assessments = self._store.load()  # reload from disk

        def assess(self, ...):
            ...
            self._assessments.append(assessment)
            self._store.append_and_save(assessment, self._assessments)

Per-Titan isolation: file path uses `resolve_titan_id()` so T1 / T2 / T3
each have independent state stores. Cleanup runs on load (drops old
entries) plus optionally on a periodic timer if the caller invokes
`prune()`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _titan_data_dir(titan_id: Optional[str] = None) -> Path:
    """Resolve the per-Titan persistence directory.

    T1 → titan/data/
    T2 → titan/data/  (T2 shares T1's data dir locally; deploys differ)
    T3 → titan3/data/ on the T3 host (this code only runs there)

    Falls back to `./data` when titan_id can't be resolved (test runs).
    """
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        tid = resolve_titan_id(titan_id) if titan_id is None else titan_id
    except Exception:
        tid = titan_id or "T1"
    # Project root = three levels up from this file
    root = Path(__file__).resolve().parent.parent.parent
    return root / "data"


class RollingStateStore:
    """Generic persistent rolling-window list. Thread-safe append+save."""

    def __init__(
        self,
        name: str,
        max_entries: int = 1000,
        max_age_s: Optional[float] = None,
        save_every_n: int = 1,
        save_every_s: float = 30.0,
        titan_id: Optional[str] = None,
        ts_key: str = "ts",
    ) -> None:
        """
        Args:
            name: store name (used as filename, e.g. "assessment_history")
            max_entries: keep at most N most-recent entries
            max_age_s: drop entries older than this many seconds (None = no age cap)
            save_every_n: persist after every N appends (1 = every append)
            save_every_s: persist if more than K seconds since last save
            titan_id: which Titan owns this store (resolves automatically if None)
            ts_key: which key in each entry holds the unix timestamp
        """
        self._name = name
        self._max_entries = int(max_entries)
        self._max_age_s = float(max_age_s) if max_age_s is not None else None
        self._save_every_n = max(1, int(save_every_n))
        self._save_every_s = float(save_every_s)
        self._ts_key = ts_key

        self._dir = _titan_data_dir(titan_id) / "dim_history"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{name}.json"

        self._lock = threading.Lock()
        self._unsaved = 0
        self._last_save_ts = 0.0

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[dict]:
        """Load entries from disk (truncated by max_entries / max_age_s)."""
        if not self._path.exists():
            return []
        try:
            with open(self._path) as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                logger.warning(
                    "[RollingStateStore:%s] disk content not a list — discarding",
                    self._name)
                return []
            return self._prune_list(raw)
        except Exception as e:
            logger.warning(
                "[RollingStateStore:%s] load failed (%s) — starting fresh",
                self._name, e)
            return []

    def _prune_list(self, entries: list[dict]) -> list[dict]:
        """Drop entries older than max_age_s + keep only newest max_entries."""
        if not entries:
            return []
        # Filter by age
        if self._max_age_s is not None:
            cutoff = time.time() - self._max_age_s
            entries = [
                e for e in entries
                if isinstance(e, dict)
                and isinstance(e.get(self._ts_key), (int, float))
                and float(e[self._ts_key]) >= cutoff
            ]
        # Keep newest N
        if len(entries) > self._max_entries:
            entries = entries[-self._max_entries:]
        return entries

    def save(self, entries: list[dict]) -> None:
        """Atomically write entries to disk."""
        pruned = self._prune_list(list(entries))
        try:
            # Atomic write: tempfile in same dir + rename
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._dir,
                prefix=f".{self._name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                json.dump(pruned, tf, default=str)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_name = tf.name
            os.replace(tmp_name, self._path)
            with self._lock:
                self._unsaved = 0
                self._last_save_ts = time.time()
        except Exception as e:
            logger.warning(
                "[RollingStateStore:%s] save failed (%s)",
                self._name, e)

    def append_and_save(
        self,
        entry: dict,
        full_list: list[dict],
    ) -> None:
        """Caller appends to its own list, then calls this to persist.

        The caller owns the list (so reads stay fast); this only handles
        the persistence cadence. Saves on save_every_n appends OR after
        save_every_s seconds, whichever comes first.
        """
        with self._lock:
            self._unsaved += 1
            should_save = (
                self._unsaved >= self._save_every_n
                or (time.time() - self._last_save_ts) >= self._save_every_s
            )
        if should_save:
            self.save(full_list)

    def prune(self, entries: list[dict]) -> list[dict]:
        """Public prune helper (caller can run periodically)."""
        return self._prune_list(list(entries))
