"""
meditation_state_publisher — owns meditation_state.bin SHM writer.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.D`.

Invoked from meditation_worker on dual-trigger cadence per Maker Q1/Q3
greenlight (2026-05-15): on every `KERNEL_EPOCH_TICK` (1.0 Hz adaptive floor)
AND on every transition (in_meditation flip, phase change, watchdog alert,
completion). Single-threaded (G21 single-writer-per-slot).

Hot-path reads from consumers (dashboard `/v4/meditation/health`, daily_nft
trigger, soul-NFT mint cron) bypass the bus entirely via this slot per
G18+G20.

Cold-boot safe — first publish before any state change uses defaults
(in_meditation=False, current_phase="idle", count=restored-from-disk or 0).
Readers see the cold values and proceed.

Failure modes mirror DreamStatePublisher + MetabolismStatePublisher
precedents:
  - encode/oversize/write fails handled per-tick; first WARN with exc_info,
    subsequent throttled to every 60 ticks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.meditation_state_specs import (
    MEDITATION_STATE_SLOT,
    MEDITATION_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)

# Valid `current_phase` enum values per SPEC §7.1 meditation_state.bin schema.
_VALID_PHASES = ("idle", "entering", "deep", "exiting")


class MeditationStatePublisher:
    """Owns meditation_state.bin SHM writer (G21 single-writer).

    Stateful: caches the tracker dict + watchdog snapshot + last_alert +
    last_completion across publish() calls so the KERNEL_EPOCH_TICK
    republish path doesn't lose them.
    """

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0

        # Tracker fields (mirrors the legacy spirit_worker.py:2207-2213 dict
        # shape one-for-one — meditation_worker restores from
        # data/meditation_state.json on boot, then mutates in-place via
        # update_tracker_field / record_completion / set_phase).
        self._tracker: dict[str, Any] = {
            "last_epoch": 0,
            "count": 0,
            "count_since_nft": 0,
            "last_ts": 0.0,
            "in_meditation": False,
            "current_phase": "idle",
        }

        # Watchdog snapshot (refreshed on every update_watchdog_snapshot call
        # — the worker calls health_snapshot() after every watchdog.check()
        # tick and passes it in).
        self._watchdog: dict[str, Any] = {
            "last_check_ts": 0.0,
            "gap_samples": 0,
            "expected_interval_hours": 0.0,
            "in_meditation_since_ts": 0.0,
            "consecutive_zero_promoted": 0,
            "selftest_done": False,
            "selftest_pass": False,
        }

        self._last_alert: Optional[dict[str, Any]] = None
        self._last_completion: Optional[dict[str, Any]] = None

        logger.info(
            "[MeditationStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.8.3 / Preamble G18 + G21 / D-SPEC-57)",
            titan_id, self._shm_root, MEDITATION_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(MEDITATION_STATE_SPEC, self._shm_root)
        logger.info(
            "[MeditationStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            MEDITATION_STATE_SLOT, MEDITATION_STATE_SPEC.payload_bytes,
            MEDITATION_STATE_SPEC.schema_version,
            self._shm_root / f"{MEDITATION_STATE_SLOT}.bin")
        return self._writer

    # ── Tracker mutators (called from worker main loop) ─────────────

    def restore_tracker(self, tracker: dict[str, Any]) -> None:
        """Restore tracker fields from data/meditation_state.json on boot.

        Tolerant of partial restore — only known keys are copied; unknown
        keys ignored; missing keys retain defaults.
        """
        for k in ("last_epoch", "count", "count_since_nft", "last_ts"):
            if k in tracker:
                try:
                    self._tracker[k] = type(self._tracker[k])(tracker[k])
                except (TypeError, ValueError):
                    pass
        # in_meditation always restored as False on boot (a meditation can
        # never have been mid-cycle when the worker exited cleanly; if it
        # crashed mid-cycle, restarting fresh is safer than restoring True
        # and confusing the M3 driver).
        self._tracker["in_meditation"] = False
        self._tracker["current_phase"] = "idle"

    def set_in_meditation(self, in_med: bool) -> None:
        self._tracker["in_meditation"] = bool(in_med)

    def set_phase(self, phase: str) -> None:
        if phase not in _VALID_PHASES:
            logger.warning(
                "[MeditationStatePublisher] invalid phase=%s — coercing to 'idle'",
                phase)
            phase = "idle"
        self._tracker["current_phase"] = phase

    def record_completion(self, *, epoch_id: int, completion: dict[str, Any]) -> None:
        """Apply a clean MEDITATION_COMPLETE event to the tracker."""
        self._tracker["in_meditation"] = False
        self._tracker["last_epoch"] = int(epoch_id or 0)
        self._tracker["last_ts"] = time.time()
        self._tracker["count"] = int(self._tracker.get("count", 0)) + 1
        self._tracker["count_since_nft"] = int(
            self._tracker.get("count_since_nft", 0)) + 1
        self._tracker["current_phase"] = "idle"
        self._last_completion = dict(completion) if isinstance(completion, dict) else None

    def reset_count_since_nft(self) -> None:
        """Called externally (daily_nft mint trigger) when an NFT is minted."""
        self._tracker["count_since_nft"] = 0

    def update_watchdog_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Replace the watchdog snapshot section. Called after every
        watchdog.check() tick.
        """
        if not isinstance(snapshot, dict):
            return
        # Only copy known keys to keep the schema strict.
        for k in self._watchdog.keys():
            if k in snapshot:
                try:
                    self._watchdog[k] = type(self._watchdog[k])(snapshot[k])
                except (TypeError, ValueError):
                    pass

    def record_alert(self, alert: dict[str, Any]) -> None:
        """Record the most-recent watchdog alert (overwrites prior)."""
        if not isinstance(alert, dict):
            return
        # Defensive: only copy declared shape (severity, failure_mode, detail, ts).
        self._last_alert = {
            "severity": str(alert.get("severity", "")),
            "failure_mode": str(alert.get("failure_mode", "")),
            "detail": str(alert.get("detail", ""))[:512],
            "ts": float(alert.get("ts", time.time())),
        }

    # ── Read-side helpers ───────────────────────────────────────────

    def get_tracker(self) -> dict[str, Any]:
        """Return a shallow copy of the current tracker dict (for callers
        outside the worker — e.g., persistence-to-disk on SAVE_NOW)."""
        return dict(self._tracker)

    def get_count(self) -> int:
        return int(self._tracker.get("count", 0))

    def get_count_since_nft(self) -> int:
        return int(self._tracker.get("count_since_nft", 0))

    def is_in_meditation(self) -> bool:
        return bool(self._tracker.get("in_meditation", False))

    # ── Publish ─────────────────────────────────────────────────────

    def publish(self) -> None:
        """Encode current state + write SHM. Called from worker on
        KERNEL_EPOCH_TICK + immediately after every mutator that the worker
        wants to surface to readers (transitions, completions, watchdog
        snapshot refresh).
        """
        self._publish_count += 1
        payload = self._compute_payload()
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[MeditationStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "in_meditation=%s phase=%s count=%d",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                self._tracker.get("in_meditation"),
                self._tracker.get("current_phase"),
                self._tracker.get("count"))

    def snapshot(self) -> dict[str, Any]:
        """Build a payload suitable for emitting on bus / for tests."""
        return self._compute_payload()

    def _compute_payload(self) -> dict[str, Any]:
        now_ts = time.time()
        return {
            "tracker": dict(self._tracker),
            "watchdog": dict(self._watchdog),
            "last_alert": dict(self._last_alert) if self._last_alert else None,
            "last_completion": dict(self._last_completion) if self._last_completion else None,
            "schema_version": MEDITATION_STATE_SPEC.schema_version,
            "ts": now_ts,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MeditationStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > MEDITATION_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[MeditationStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), MEDITATION_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[MeditationStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d in_meditation=%s phase=%s "
                    "(consumers can now read; closes spirit_supplemental "
                    "meditation_health G21 violation per D-SPEC-57)",
                    MEDITATION_STATE_SLOT, len(encoded),
                    payload.get("tracker", {}).get("in_meditation"),
                    payload.get("tracker", {}).get("current_phase"))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[MeditationStatePublisher] writer.write_variable failed (#%d): %s",
                    self._write_fails, e, exc_info=True)
