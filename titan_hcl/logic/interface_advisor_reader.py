"""
interface_advisor_reader — sub-µs G18 SHM-direct reader for
interface_advisor_state.bin.

Phase C v1.8.5 (D-SPEC-59) per `rFP_titan_hcl_l2_separation_strategy.md §4.H`.
Maker greenlit 2026-05-15 inline (SHM-rate-oracle pattern).

Used by parent `_handle_impulse` (and any future caller wanting rate-state
visibility) on hot paths — no bus RPC, no kernel_rpc, just mmap read.

Replaces the in-proc `self._interface_advisor.check()` synchronous path
that was retired in v1.8.5 §4.H. The pre-carve check was atomic (record +
return feedback in one in-proc µs-scale call); the post-carve replacement
is split:
  - Hot caller (parent): read SHM rates → if rate >= limit, skip + emit
    RATE_LIMIT feedback locally; else emit IMPULSE_RECEIVED + proceed.
  - Worker (subprocess): receives IMPULSE_RECEIVED → records timestamp in
    deque (advisor.check) → republishes SHM (rate-throttled).

100ms cache to absorb hot-path call volume (parent _handle_impulse may fire
many times per second under impulse storms; cache prevents thrashing mmap
reads at >Hz rate).

Cold-boot safe: returns defaults (empty rates dict, INITIAL_LIMITS, zero
rate_limit_count) if the slot hasn't been written yet. Readers see the cold
values and effectively pass all rate checks (no current_rate ⇒ rate=0 ⇒
within limit) — eventually-consistent behavior matches what would happen
if the InterfaceAdvisor were freshly instantiated.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic.interface_advisor import (
    DEFAULT_WINDOW,
    INITIAL_LIMITS,
)
from titan_hcl.logic.interface_advisor_specs import (
    INTERFACE_ADVISOR_STATE_SLOT,
    INTERFACE_ADVISOR_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_READ_CACHE_TTL_S = 0.1


_DEFAULT_PAYLOAD: dict[str, Any] = {
    "rates": {},
    "limits": dict(INITIAL_LIMITS),
    "window_s": float(DEFAULT_WINDOW),
    "rate_limit_count": 0,
    "schema_version": INTERFACE_ADVISOR_STATE_SPEC.schema_version,
    "ts": 0.0,
}


class InterfaceAdvisorStateReader:
    """Sub-µs G18 SHM reader for interface_advisor_state.bin.

    Per-process singleton — instantiate once at boot, call `read()` /
    `get_current_rate(msg_type)` / `check(msg_type, source)` from hot paths.
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
        self._reader = StateRegistryReader(
            INTERFACE_ADVISOR_STATE_SPEC, self._shm_root)
        return self._reader

    def read(self) -> dict[str, Any]:
        """Read the full interface_advisor state payload. Cached for
        `_READ_CACHE_TTL_S`. Returns defensive copy."""
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
                    "[InterfaceAdvisorStateReader] SHM read failed (#%d): "
                    "%s — returning defaults (empty rates)",
                    self._read_count, e)
            return dict(_DEFAULT_PAYLOAD)

        if raw is None or len(raw) == 0:
            return dict(_DEFAULT_PAYLOAD)

        try:
            payload = msgpack.unpackb(raw, raw=False)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"expected dict, got {type(payload).__name__}")
        except Exception as e:
            self._decode_fails += 1
            if self._decode_fails == 1 or self._decode_fails % 60 == 0:
                logger.warning(
                    "[InterfaceAdvisorStateReader] msgpack decode failed "
                    "(#%d): %s — returning defaults",
                    self._decode_fails, e)
            return dict(_DEFAULT_PAYLOAD)

        self._cached_payload = payload
        self._cached_at = now

        if not self._first_read_logged:
            self._first_read_logged = True
            logger.debug(
                "[InterfaceAdvisorStateReader] FIRST SUCCESSFUL READ — "
                "active_types=%d rate_limit_count=%d (slot=%s)",
                len(payload.get("rates", {})),
                int(payload.get("rate_limit_count", 0)),
                INTERFACE_ADVISOR_STATE_SLOT)

        return dict(payload)

    def get_current_rate(self, msg_type: str) -> int:
        """Hot-path shortcut — return the current rate count in the active
        window for the given message type. 0 if not yet recorded."""
        payload = self.read()
        rates = payload.get("rates", {}) or {}
        try:
            return int(rates.get(msg_type, 0))
        except (TypeError, ValueError):
            return 0

    def check(self, msg_type: str, source: str = "") -> Optional[dict]:
        """Parent-side rate check using SHM-read rates. Mirrors the
        InterfaceAdvisor.check() return contract: None if within limits,
        or a RATE_LIMIT feedback dict if exceeded.

        Semantics shift from in-proc advisor.check():
          - Pre-carve: atomic record + check in one in-proc call.
          - Post-carve: this is a READ-ONLY check using the SHM snapshot.
            The recording happens at the worker after the caller emits
            IMPULSE_RECEIVED. Eventually-consistent up to bus latency
            (~10-50ms).

        Returns:
            None if within limits — caller should emit IMPULSE_RECEIVED
            and proceed. RATE_LIMIT feedback dict if exceeded — caller
            should emit RATE_LIMIT bus event locally + skip the action.
        """
        payload = self.read()
        limits = payload.get("limits") or dict(INITIAL_LIMITS)
        rates = payload.get("rates", {}) or {}
        window_s = float(payload.get("window_s", DEFAULT_WINDOW))

        try:
            limit = limits.get(msg_type)
        except (TypeError, AttributeError):
            limit = None
        if limit is None:
            return None

        try:
            current_rate = int(rates.get(msg_type, 0))
        except (TypeError, ValueError):
            current_rate = 0

        # Caller is ABOUT to add 1 to the rate by emitting IMPULSE_RECEIVED.
        # Use >= limit so the about-to-be-emitted impulse is captured (matches
        # InterfaceAdvisor.check semantics where `current_rate > limit` AFTER
        # append; here we mimic by comparing current_rate + 1 > limit).
        if current_rate + 1 > int(limit):
            suggested_backoff = window_s / max(1, int(limit))
            return {
                "message_type": msg_type,
                "current_rate": current_rate,
                "limit": int(limit),
                "window_seconds": window_s,
                "suggested_backoff": round(suggested_backoff, 2),
                "source": source,
                "ts": time.time(),
            }
        return None

    def get_stats(self) -> dict[str, Any]:
        """Compat shim for callers that previously held the in-proc
        InterfaceAdvisor reference and called `.get_stats()` — returns the
        same shape (limits + current_rates + window_seconds + rate_limit_count)
        from SHM."""
        payload = self.read()
        return {
            "limits": payload.get("limits", {}) or {},
            "current_rates": payload.get("rates", {}) or {},
            "window_seconds": float(payload.get("window_s", DEFAULT_WINDOW)),
            "rate_limit_count": int(payload.get("rate_limit_count", 0)),
        }
