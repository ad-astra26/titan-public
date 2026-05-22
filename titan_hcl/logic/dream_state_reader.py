"""
dream_state_reader — sub-µs G18 SHM-direct reader for dream_state.bin.

Phase C v1.8.2 (D-SPEC-56) per `rFP_titan_hcl_l2_separation_strategy.md §4.I`.

Used by all in-process consumers that need `is_dreaming` or full dream state
on a hot path (no bus RPC, no kernel_rpc, just mmap read):

  - plugin chat handler (`_post_chat`) — chat-during-dream buffer decision
  - api_subprocess `chat_pipeline.py` — same shape as plugin chat handler
  - spirit_worker `_read_is_dreaming_from_shm()` helper (replacing the
    deleted `_shared_is_dreaming` module-level flag + 20+ readers across
    reasoning_engine / msl / observatory / meditation / etc.)
  - expression_worker tick-gate cache (DREAM_STATE_CHANGED subscriber STAYS
    for transition edge notifications, but for "should I tick now" the SHM
    read is the fast path)
  - timechain_worker dream-hook (DREAM_STATE_CHANGED subscriber stays)
  - dashboard / api state_accessor for `/v4/dreaming`

100ms cache to absorb hot-path call volume (reasoning engine + msl + observatory
each tick at sub-second cadence; cache prevents thrashing mmap reads at >Hz rate).

Cold-boot safe: returns defaults (is_dreaming=False, state="awake",
recovery_pct=0.0) if the slot hasn't been written yet by dream_state_worker.
Logs a one-shot DEBUG when first successful read happens.

Staleness detection: reader exposes `is_stale()` for callers that want to
distinguish "definitely awake" from "worker hung". Based on the
`last_transition_ts` freshness probe + `DREAM_STATE_REPUBLISH_CADENCE_S × 5`
threshold per Maker Q6 greenlight.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl._phase_c_constants import DREAM_STATE_REPUBLISH_CADENCE_S
from titan_hcl.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic.dream_state_specs import (
    DREAM_STATE_SLOT,
    DREAM_STATE_SPEC,
)

logger = logging.getLogger(__name__)


# Cache TTL for hot-path readers — see module docstring.
_READ_CACHE_TTL_S = 0.1

# Staleness threshold: 5× the dual-trigger republish cadence.
# Above this, callers should treat dream_state_worker as potentially hung.
_STALENESS_THRESHOLD_S = DREAM_STATE_REPUBLISH_CADENCE_S * 5.0

# Default returned when slot is empty / decode fails / stale beyond recovery.
_DEFAULT_PAYLOAD: dict[str, Any] = {
    "is_dreaming": False,
    "state": "awake",
    "recovery_pct": 0.0,
    "remaining_epochs": 0,
    "wake_transition": False,
    "just_woke": False,
    "wake_ts": 0.0,
    "dream_started_ts": 0.0,
    "last_transition_ts": 0.0,
    "schema_version": DREAM_STATE_SPEC.schema_version,
    "ts": 0.0,
}


class DreamStateReader:
    """Sub-µs G18 SHM reader for dream_state.bin.

    Per-process singleton — instantiate once at boot, call `read()` /
    `is_dreaming()` from hot paths. Internally caches the last decoded
    payload for `_READ_CACHE_TTL_S` to absorb tick-rate call volume.
    """

    def __init__(self, titan_id: Optional[str] = None):
        self._titan_id = titan_id or resolve_titan_id()
        self._shm_root = ensure_shm_root(self._titan_id)
        self._reader: Optional[StateRegistryReader] = None
        self._cached_payload: Optional[dict[str, Any]] = None
        self._cached_at: float = 0.0
        self._read_count = 0
        self._decode_fails = 0
        self._first_read_logged = False

    def _reader_attach(self) -> StateRegistryReader:
        if self._reader is not None:
            return self._reader
        self._reader = StateRegistryReader(DREAM_STATE_SPEC, self._shm_root)
        return self._reader

    def read(self) -> dict[str, Any]:
        """Read the full dream state payload. Cached for `_READ_CACHE_TTL_S`.

        Returns a defensive copy so callers can mutate without affecting the
        cache (defensive against accidental aliasing — copy is sub-µs for
        the small fixed-shape payload).
        """
        now = time.time()
        if (self._cached_payload is not None
                and (now - self._cached_at) < _READ_CACHE_TTL_S):
            return dict(self._cached_payload)

        self._read_count += 1
        try:
            reader = self._reader_attach()
            raw = reader.read_variable()
        except Exception as e:
            if self._read_count <= 3 or self._read_count % 600 == 0:
                logger.debug(
                    "[DreamStateReader] SHM read failed (#%d): %s — "
                    "returning defaults (is_dreaming=False)",
                    self._read_count, e)
            return dict(_DEFAULT_PAYLOAD)

        if raw is None or len(raw) == 0:
            # Slot exists but never written — dream_state_worker not booted yet.
            return dict(_DEFAULT_PAYLOAD)

        try:
            payload = msgpack.unpackb(raw, raw=False)
            if not isinstance(payload, dict):
                raise TypeError(f"expected dict, got {type(payload).__name__}")
        except Exception as e:
            self._decode_fails += 1
            if self._decode_fails == 1 or self._decode_fails % 60 == 0:
                logger.warning(
                    "[DreamStateReader] msgpack decode failed (#%d): %s — "
                    "returning defaults",
                    self._decode_fails, e)
            return dict(_DEFAULT_PAYLOAD)

        self._cached_payload = payload
        self._cached_at = now

        if not self._first_read_logged:
            self._first_read_logged = True
            logger.debug(
                "[DreamStateReader] FIRST SUCCESSFUL READ — is_dreaming=%s "
                "state=%s recovery_pct=%.1f (slot=%s)",
                payload.get("is_dreaming"), payload.get("state"),
                payload.get("recovery_pct", 0.0), DREAM_STATE_SLOT)

        return dict(payload)

    def is_dreaming(self) -> bool:
        """Hot-path shortcut — return only the boolean. Cached same as read()."""
        return bool(self.read().get("is_dreaming", False))

    def is_stale(self) -> bool:
        """Return True if last_transition_ts is older than the staleness
        threshold (worker may be hung). Use sparingly — most callers should
        just trust `is_dreaming()` and not worry about freshness.
        """
        payload = self.read()
        last_ts = float(payload.get("last_transition_ts", 0.0) or 0.0)
        if last_ts <= 0.0:
            # Never written → cold boot, not "stale".
            return False
        return (time.time() - last_ts) > _STALENESS_THRESHOLD_S
