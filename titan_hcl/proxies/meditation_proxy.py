"""
Meditation Proxy — bridge to the supervised meditation_worker subprocess.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.D`.
Replaces the legacy `state_refs["meditation_tracker"]` cross-process dict
reference + `spirit_supplemental_state.bin` `meditation_health` section
indirection (G21 violation — meditation_worker would otherwise become 2nd
writer to a spirit-owned slot).

Classification per SPEC Preamble G18-G22:

  • get_tracker / get_watchdog_health   → SHM read of meditation_state.bin
                                          (G18 — state via SHM, never bus).
  • force_end                           → bus publish MEDITATION_FORCE_END
                                          (one-way, no reply expected — the
                                          worker emits MEDITATION_INTERRUPTED
                                          on completion which any observer
                                          can subscribe to).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..bus import (
    MEDITATION_FORCE_END,
    DivineBus,
    make_msg,
)
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian_hcl import Guardian
from ..logic.meditation_state_reader import MeditationStateReader
from ..logic.meditation_state_specs import (
    MEDITATION_STATE_SLOT,
    MEDITATION_STATE_SPEC,
)

logger = logging.getLogger(__name__)


# Default tracker payload returned when SHM slot is unreadable. Matches the
# legacy `_meditation_tracker` shape from spirit_worker.py:2207-2213 + adds
# the new `current_phase` field per D-SPEC-57 §3.1 schema.
_DEFAULT_TRACKER = {
    "last_epoch": 0,
    "count": 0,
    "count_since_nft": 0,
    "last_ts": 0.0,
    "in_meditation": False,
    "current_phase": "idle",
}

_DEFAULT_WATCHDOG = {
    "last_check_ts": 0.0,
    "gap_samples": 0,
    "expected_interval_hours": 0.0,
    "in_meditation_since_ts": 0.0,
    "consecutive_zero_promoted": 0,
    "selftest_done": False,
    "selftest_pass": False,
}


class MeditationProxy:
    """Drop-in proxy for the meditation_worker subprocess.

    Surfaces:
      - get_tracker():          SHM read of meditation_state.bin tracker section
      - get_watchdog_health():  SHM read of watchdog section
      - get_state():            SHM read of full payload (tracker + watchdog + alerts)
      - force_end(reason):      bus publish MEDITATION_FORCE_END (fire-and-forget)
      - is_in_meditation():     fast tracker.in_meditation read

    Identical pattern to MetabolismProxy / SocialGraphProxy / DreamStateProxy
    (D-SPEC-50 + D-SPEC-51 + D-SPEC-56 predecessors).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._started = False

        # SHM-direct reader (G18 path).
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_state = MeditationStateReader(self._titan_id)
        self._fallback_count = 0

        logger.info(
            "[MeditationProxy] initialized SHM-direct reader — "
            "titan_id=%s shm_root=%s (slot=%s per Preamble G18 + "
            "rFP_titan_hcl_l2_separation_strategy §4.D)",
            self._titan_id, self._shm_root, MEDITATION_STATE_SLOT)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        """Lazy-spawn meditation_worker if registered as lazy=True. For the
        default autostart=True registration this is a no-op."""
        if self._started:
            return
        try:
            from ._start_safe import ensure_started_async_safe
            ready = ensure_started_async_safe(
                self._guardian, "meditation", id(self),
                proxy_label="MeditationProxy",
            )
            if ready:
                self._started = True
        except Exception as e:
            logger.debug(
                "[MeditationProxy] _ensure_started raised "
                "(autostart fallback): %s", e)

    # ── SHM reads (G18 path) ────────────────────────────────────────

    def _read_state(self) -> Optional[dict]:
        payload = self._r_state.read()
        if payload is None:
            self._fallback_count += 1
            if self._fallback_count == 1:
                logger.info(
                    "[MeditationProxy] FIRST FALLBACK slot=%s — using cold "
                    "defaults until meditation_worker first publish",
                    MEDITATION_STATE_SLOT)
            return None
        return payload

    def get_state(self) -> dict[str, Any]:
        """Return full meditation_state.bin payload — tracker + watchdog +
        last_alert + last_completion. Cold defaults on read failure.
        """
        decoded = self._read_state()
        if decoded is None:
            return {
                "tracker": dict(_DEFAULT_TRACKER),
                "watchdog": dict(_DEFAULT_WATCHDOG),
                "last_alert": None,
                "last_completion": None,
                "schema_version": MEDITATION_STATE_SPEC.schema_version,
                "ts": 0.0,
            }
        return decoded

    def get_tracker(self) -> dict[str, Any]:
        """Return meditation tracker dict — drops in for direct dict access
        replacing `state_refs["meditation_tracker"]` reference. Mirrors
        spirit_worker.py:2207-2213 shape + adds `current_phase` per D-SPEC-57.
        """
        state = self._read_state()
        if state is None:
            return dict(_DEFAULT_TRACKER)
        tracker = state.get("tracker")
        if not isinstance(tracker, dict):
            return dict(_DEFAULT_TRACKER)
        # Defensive: fill missing keys with defaults so callers can do
        # `tracker.get("count_since_nft")` without surprise None.
        merged = dict(_DEFAULT_TRACKER)
        merged.update(tracker)
        return merged

    def get_watchdog_health(self) -> dict[str, Any]:
        """Return watchdog snapshot dict — mirrors
        `MeditationWatchdog.health_snapshot()` shape so dashboard
        `/v4/meditation/health` keeps the same response contract.
        """
        state = self._read_state()
        if state is None:
            return dict(_DEFAULT_WATCHDOG)
        watchdog = state.get("watchdog")
        if not isinstance(watchdog, dict):
            return dict(_DEFAULT_WATCHDOG)
        merged = dict(_DEFAULT_WATCHDOG)
        merged.update(watchdog)
        return merged

    def is_in_meditation(self) -> bool:
        """Fast hot-path read — defaults False on read failure."""
        return self._r_state.is_in_meditation()

    def get_count(self) -> int:
        """Fast hot-path count read — defaults 0 on read failure."""
        return self._r_state.get_count()

    # ── Force-end (fire-and-forget bus event) ───────────────────────

    def force_end(self, reason: str = "maker_dashboard",
                  source: str = "maker_dashboard") -> bool:
        """Publish MEDITATION_FORCE_END to the meditation worker.

        Fire-and-forget: the worker will reset its in_meditation flag and
        emit MEDITATION_INTERRUPTED on completion. The in-flight 300s
        run_meditation work-RPC is NOT aborted (it has its own timeout);
        the next cycle attempt waits for that to resolve.

        Returns True if the publish was queued successfully, False on
        publish failure (rare — surfacing for /v4/meditation/force-end
        endpoint).
        """
        import time as _time
        try:
            msg = make_msg(
                MEDITATION_FORCE_END, "meditation_proxy", "meditation",
                {"reason": reason, "source": source, "client_ts": _time.time()},
            )
            self._bus.publish(msg)
            logger.info(
                "[MeditationProxy] MEDITATION_FORCE_END published — "
                "reason=%s source=%s", reason, source)
            return True
        except Exception as e:
            logger.warning(
                "[MeditationProxy] MEDITATION_FORCE_END publish failed: %s", e)
            return False
