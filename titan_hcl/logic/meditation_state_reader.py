"""
meditation_state_reader — sub-µs G18 SHM-direct reader for meditation_state.bin.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.D`.

Used by dashboard `/v4/meditation/health` + daily_nft trigger + soul-NFT mint
cron + meditation_proxy.get_tracker() — all of these bypass the bus per G18+G20
and read meditation_worker's authoritative state directly via shared memory.

Cold-boot safe — if the slot hasn't been written yet (meditation_worker not
booted, or boot-race), returns None and callers fall back to defaults.

Failure mode mirrors DreamStateReader: any decode failure returns None.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
)
from titan_hcl.logic.meditation_state_specs import (
    MEDITATION_STATE_SLOT,
    MEDITATION_STATE_SPEC,
)

logger = logging.getLogger(__name__)


class MeditationStateReader:
    """Sub-µs SHM reader for meditation_state.bin.

    Reusable across many call sites — the underlying mmap survives across
    publisher rewrites (triple-buffered atomic swap per G15 +
    StateRegistryReader contract).
    """

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._reader: Optional[StateRegistryReader] = None
        self._decode_fails = 0

    def _reader_attach(self) -> Optional[StateRegistryReader]:
        if self._reader is not None:
            return self._reader
        try:
            self._reader = StateRegistryReader(MEDITATION_STATE_SPEC, self._shm_root)
            return self._reader
        except Exception as e:
            # Slot file may not exist yet (worker not booted, or pre-first-write).
            # This is expected during cold boot — log once at INFO, callers see None.
            logger.info(
                "[MeditationStateReader] cannot attach yet — slot=%s err=%s "
                "(expected during cold boot; will retry on next read)",
                MEDITATION_STATE_SLOT, e)
            return None

    def read(self) -> Optional[dict[str, Any]]:
        """Return decoded payload dict or None if slot is empty/unreadable."""
        reader = self._reader_attach()
        if reader is None:
            return None
        try:
            raw = reader.read_variable()
        except Exception as e:
            self._decode_fails += 1
            if self._decode_fails == 1 or self._decode_fails % 100 == 0:
                logger.warning(
                    "[MeditationStateReader] read_variable failed (#%d): %s",
                    self._decode_fails, e)
            return None
        if not raw:
            return None
        try:
            payload = msgpack.unpackb(raw, raw=False)
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception as e:
            self._decode_fails += 1
            if self._decode_fails == 1 or self._decode_fails % 100 == 0:
                logger.warning(
                    "[MeditationStateReader] msgpack decode failed (#%d): %s",
                    self._decode_fails, e)
            return None

    def get_tracker(self) -> Optional[dict[str, Any]]:
        """Convenience: extract just the tracker section."""
        payload = self.read()
        if payload is None:
            return None
        tracker = payload.get("tracker")
        return tracker if isinstance(tracker, dict) else None

    def get_watchdog(self) -> Optional[dict[str, Any]]:
        """Convenience: extract just the watchdog section."""
        payload = self.read()
        if payload is None:
            return None
        watchdog = payload.get("watchdog")
        return watchdog if isinstance(watchdog, dict) else None

    def is_in_meditation(self) -> bool:
        """Fast hot-path read — defaults to False on read failure."""
        tracker = self.get_tracker()
        if tracker is None:
            return False
        return bool(tracker.get("in_meditation", False))

    def get_count(self) -> int:
        """Fast hot-path read — defaults to 0 on read failure."""
        tracker = self.get_tracker()
        if tracker is None:
            return 0
        try:
            return int(tracker.get("count", 0))
        except (TypeError, ValueError):
            return 0
