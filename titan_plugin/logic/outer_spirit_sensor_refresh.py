"""
outer_spirit_sensor_refresh — Phase C C-S6 Python sensor sidecar (outer spirit).

SPEC §9.D Sensor refresh sidecars + SPEC §18.1 stale handling +
SPEC §7.1 sensor_cache_<level>_<sense>.bin layout.

Hosted in ``titan_HCL`` as an asyncio task supervised by ``guardian_HCL``
via in-process exception handler (SPEC §11.B line 1236). Aggregates
the pre-aggregated outer-state ``sources`` dict that
``titan-outer-spirit-rs`` needs to compute its 45D Sat-Chit-Ananda
Material tensor (per ``outer_spirit_tensor.py:collect_outer_spirit_45d``),
msgpack-encodes it, and writes the result to
``/dev/shm/titan_<id>/sensor_cache_outer_spirit.bin`` via SeqLock.

Design parallels ``outer_body_sensor_refresh.py`` and
``outer_mind_sensor_refresh.py`` — see those modules' docstrings for
the full Cadence / Slot byte layout / Stale handling / Failure
semantics / Flag-gating sections. This module differs only in:

  - REFRESH_PERIOD_S = OUTER_SPIRIT_TICK_BASE_S (30.0 s) — natural
    cadence per SPEC §9.D line 1031 ("outer_spirit: observatory
    narrative ~30s, presence ~60s")
  - SLOT_NAME = "sensor_cache_outer_spirit"
  - MAX_PAYLOAD_BYTES = OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES (8192)
  - SOURCE_KEYS = canonical msgpack-friendly key set consumed by
    ``outer_spirit_tensor.collect_outer_spirit_45d``: action_stats /
    creative_stats / guardian_stats / sovereignty_ratio / uptime_ratio /
    recovery_stats / social_stats / memory_stats / hormone_levels /
    solana_stats / assessment_stats / history.

Per SPEC §9.A line 944, ``titan-outer-spirit-rs`` reads ONLY this cache
+ ``outer_body_5d.bin`` + ``outer_mind_15d.bin`` (Observer Principle).
It NEVER does direct DB/RPC — sidecar is the sole upstream provider.

──────────────────────────────────────────────────────────────────────
SAT-CHIT-ANANDA Material (per outer_spirit_tensor.py:9 + SPEC §7.1
row 7 + G8 Observer Principle)
──────────────────────────────────────────────────────────────────────
Rust daemon decodes msgpack, runs the Sat-Chit-Ananda Material
formula on inputs:

  SAT[0:15]    Material Being   — does Titan EXIST in the world?
  CHIT[15:30]  Material Awareness — does Titan KNOW the world?
  ANANDA[30:45] Material Fulfillment — is engagement FRUITFUL?

Observer dims `[0:5]` (TITAN_SELF absolute `[85:90]` per G8) are
present in the slot but **MASKED at filter_down emit only** — the slot
itself contains all 45D for downstream consumers. Observer mask
applied by the Rust daemon when composing OUTER_SPIRIT_FILTER_DOWN
per SPEC §10.F step 3.

Per SPEC §3.0 + master plan §12.1: sidecar runs unconditionally.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any, Optional

import msgpack
import numpy as np

from titan_plugin._phase_c_constants import (
    OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES,
    OUTER_SPIRIT_TICK_BASE_S,
)
from titan_plugin.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

REFRESH_PERIOD_S: float = OUTER_SPIRIT_TICK_BASE_S  # 30.0
MAX_PAYLOAD_BYTES: int = OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES  # 8192
SLOT_NAME: str = "sensor_cache_outer_spirit"
SCHEMA_VERSION: int = 1
_WARN_THROTTLE_EVERY: int = 60
_RESTART_BACKOFF_S: float = 5.0

#: Canonical source-dict key set for outer_spirit. Consumed by
#: ``outer_spirit_tensor.collect_outer_spirit_45d`` keyword args
#: (lines 53-69). All keys are msgpack-friendly plain types.
SOURCE_KEYS: tuple[str, ...] = (
    "action_stats",        # SAT/CHIT/ANANDA — derived from agency_stats
    "creative_stats",      # SAT/ANANDA — derived from art/audio counts
    "guardian_stats",      # SAT — derived from helper_statuses + bus_stats
    "sovereignty_ratio",   # SAT [2] — float (parent-derived from agency)
    "uptime_ratio",        # SAT — float (parent-derived from uptime_seconds)
    "recovery_stats",      # SAT — derived from network_monitor / restarts
    "social_stats",        # CHIT/ANANDA — derived from social_perception
    "memory_stats",        # CHIT — derived from memory_status
    "hormone_levels",      # ANANDA — neuromod / hormonal cascade state
    "solana_stats",        # ANANDA — derived from sol_balance + anchor_state
    "assessment_stats",    # CHIT — assessment self-evaluation
    "history",             # ANANDA — rolling outer_spirit recent state
)


SourcesProvider = Callable[[], dict[str, Any]]


# ── Sidecar class ───────────────────────────────────────────────────


class OuterSpiritSensorRefresh:
    """
    SPEC §9.D outer-spirit sensor refresh sidecar.

    Polls the in-process ``sources_provider`` callable at
    ``REFRESH_PERIOD_S``, msgpack-encodes the canonical pre-aggregated
    outer-state dict projected onto ``SOURCE_KEYS``, and writes the
    result to ``sensor_cache_outer_spirit.bin`` via SeqLock.

    See ``OuterBodySensorRefresh`` docstring for the full lifecycle +
    failure-semantics contract — this class is a structural twin.
    """

    def __init__(
        self,
        sources_provider: SourcesProvider,
        titan_id: Optional[str] = None,
        stop_event: Optional[asyncio.Event] = None,
        refresh_period_s: float = REFRESH_PERIOD_S,
    ) -> None:
        self._sources_provider = sources_provider
        self._titan_id = titan_id
        self._stop = stop_event if stop_event is not None else asyncio.Event()
        self._refresh_period_s = refresh_period_s

        self._writer: Optional[StateRegistryWriter] = None
        self._tick_count: int = 0
        self._provider_failure_count: int = 0
        self._oversize_failure_count: int = 0
        self._write_failure_count: int = 0
        self._last_payload_bytes: int = 0

    # -- public lifecycle -------------------------------------------

    async def run(self) -> None:
        logger.info(
            "outer_spirit_sensor_refresh starting "
            "(period=%.1fs, slot=%s, max_bytes=%d, schema_v=%d)",
            self._refresh_period_s, SLOT_NAME,
            MAX_PAYLOAD_BYTES, SCHEMA_VERSION,
        )
        try:
            self._writer = self._build_writer()
        except Exception:
            logger.critical(
                "outer_spirit_sensor_refresh: failed to build SeqLock "
                "writer for %s — sidecar will retry every %.1fs:\n%s",
                SLOT_NAME, _RESTART_BACKOFF_S, traceback.format_exc(),
            )

        while not self._stop.is_set():
            try:
                await self._refresh_loop()
            except asyncio.CancelledError:
                logger.info("outer_spirit_sensor_refresh cancelled — exiting")
                raise
            except Exception:
                logger.critical(
                    "outer_spirit_sensor_refresh _refresh_loop crashed; "
                    "restarting after %.1fs backoff:\n%s",
                    _RESTART_BACKOFF_S, traceback.format_exc(),
                )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=_RESTART_BACKOFF_S,
                    )
                except asyncio.TimeoutError:
                    pass

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        logger.info(
            "outer_spirit_sensor_refresh stopped after %d ticks "
            "(provider_failures=%d, oversize_failures=%d, write_failures=%d)",
            self._tick_count, self._provider_failure_count,
            self._oversize_failure_count, self._write_failure_count,
        )

    async def stop(self) -> None:
        self._stop.set()

    # -- introspection ----------------------------------------------

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def last_payload_bytes(self) -> int:
        return self._last_payload_bytes

    @property
    def provider_failure_count(self) -> int:
        return self._provider_failure_count

    @property
    def oversize_failure_count(self) -> int:
        return self._oversize_failure_count

    @property
    def write_failure_count(self) -> int:
        return self._write_failure_count

    # -- private -----------------------------------------------------

    def _build_writer(self) -> StateRegistryWriter:
        spec = RegistrySpec(
            name=SLOT_NAME,
            dtype=np.dtype("uint8"),
            shape=(MAX_PAYLOAD_BYTES,),
            schema_version=SCHEMA_VERSION,
            variable_size=True,
        )
        shm_root = ensure_shm_root(self._titan_id)
        return StateRegistryWriter(spec, shm_root)

    async def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            tick_start = time.monotonic()
            self._refresh_and_write()
            self._tick_count += 1

            elapsed = time.monotonic() - tick_start
            sleep_for = max(0.0, self._refresh_period_s - elapsed)
            if sleep_for > 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=sleep_for,
                    )
                except asyncio.TimeoutError:
                    pass

    def _refresh_and_write(self) -> None:
        try:
            raw = self._sources_provider()
        except Exception:
            self._provider_failure_count += 1
            if (
                self._provider_failure_count == 1
                or self._provider_failure_count % _WARN_THROTTLE_EVERY == 0
            ):
                logger.warning(
                    "outer_spirit_sensor_refresh sources_provider raised "
                    "(failure_count=%d, tick=%d):\n%s",
                    self._provider_failure_count, self._tick_count,
                    traceback.format_exc(),
                )
            return

        sources = self._normalize_sources(raw)

        try:
            payload = msgpack.packb(sources, use_bin_type=True)
        except (TypeError, ValueError) as exc:
            self._oversize_failure_count += 1
            logger.error(
                "outer_spirit_sensor_refresh msgpack encode failed "
                "(tick=%d): %s — keys=%s",
                self._tick_count, exc, sorted(sources.keys()),
            )
            return

        if len(payload) > MAX_PAYLOAD_BYTES:
            self._oversize_failure_count += 1
            logger.critical(
                "outer_spirit_sensor_refresh payload %dB > MAX %dB "
                "(tick=%d) — slot retains last-known. Investigate "
                "upstream source-shape drift; do NOT silently truncate.",
                len(payload), MAX_PAYLOAD_BYTES, self._tick_count,
            )
            return

        if self._writer is None:
            try:
                self._writer = self._build_writer()
            except Exception:
                self._write_failure_count += 1
                if self._write_failure_count == 1 or self._write_failure_count % _WARN_THROTTLE_EVERY == 0:
                    logger.warning(
                        "outer_spirit_sensor_refresh writer build failed "
                        "(failure_count=%d, tick=%d):\n%s",
                        self._write_failure_count, self._tick_count,
                        traceback.format_exc(),
                    )
                return

        try:
            self._writer.write_variable(payload)
            self._last_payload_bytes = len(payload)
        except Exception:
            self._write_failure_count += 1
            if (
                self._write_failure_count == 1
                or self._write_failure_count % _WARN_THROTTLE_EVERY == 0
            ):
                logger.warning(
                    "outer_spirit_sensor_refresh write_variable failed "
                    "(failure_count=%d, tick=%d):\n%s",
                    self._write_failure_count, self._tick_count,
                    traceback.format_exc(),
                )

    @staticmethod
    def _normalize_sources(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Project raw sources dict onto the canonical SOURCE_KEYS.
        Missing keys → None. Extra keys dropped.
        """
        return {key: raw.get(key) for key in SOURCE_KEYS}
