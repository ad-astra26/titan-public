"""
outer_mind_sensor_refresh — Phase C C-S6 Python sensor sidecar (outer mind).

SPEC §9.D Sensor refresh sidecars + SPEC §18.1 stale handling +
SPEC §7.1 sensor_cache_<level>_<sense>.bin layout.

Hosted in ``titan_HCL`` as an asyncio task supervised by ``guardian_HCL``
via in-process exception handler (SPEC §11.B line 1236). Aggregates
the ``sources`` dict that ``titan-outer-mind-rs`` needs to compute its
15D extended Outer Mind tensor (Thinking[0:5] + Feeling[5:10] +
Willing[10:15]; per ``outer_trinity.py:_collect_outer_mind`` 5D base
+ ``outer_mind_tensor.collect_outer_mind_15d`` extended), msgpack-
encodes it, and writes the result to
``/dev/shm/titan_<id>/sensor_cache_outer_mind.bin`` via SeqLock.

Design parallels ``outer_body_sensor_refresh.py`` — see that module's
docstring for the full Cadence / Slot byte layout / Stale handling /
Failure semantics / Flag-gating sections. This module differs only in:

  - REFRESH_PERIOD_S = OUTER_MIND_TICK_BASE_S (15.0 s) — G13 1:3:9
    middle cadence per D-SPEC-100 (spirit 5s / mind 15s / body 45s),
    mirroring OUTER_MIND_BUS_PUBLISH_INTERVAL_S. The Rust daemon ticks
    at full Schumann mind (23.49 Hz); this constant only sets the
    sidecar source-refresh cadence + 3× stale threshold (45s).
  - SLOT_NAME = "sensor_cache_outer_mind"
  - MAX_PAYLOAD_BYTES = OUTER_SENSOR_CACHE_MIND_MAX_BYTES (8192)
  - SOURCE_KEYS = the 16 canonical msgpack-friendly keys consumed by
    ``_collect_outer_mind`` (5D base) + ``collect_outer_mind_15d``
    (15D extended, both keyed off the parent's
    ``_gather_outer_trinity_sources`` plain-type dict).

Rust-daemon-side ground_up applies to **willing dims [10:15] ONLY**
per SPEC G10 (``ground_up_skip_mind_thinking=0:5``,
``ground_up_skip_mind_feeling=5:10``); filter_down multipliers
clamped to G7 ``[0.3, 3.0]``.
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

from titan_hcl._phase_c_constants import (
    OUTER_MIND_TICK_BASE_S,
    OUTER_SENSOR_CACHE_MIND_MAX_BYTES,
)
from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

#: Refresh period (s). Sourced from generated TOML constant; matches
#: titan-outer-mind-rs daemon's natural cadence per SPEC §18.1.
REFRESH_PERIOD_S: float = OUTER_MIND_TICK_BASE_S  # 15.0 (D-SPEC-100)

#: Max msgpack payload size (bytes). Matches slot_specs.rs C-S2 cap.
MAX_PAYLOAD_BYTES: int = OUTER_SENSOR_CACHE_MIND_MAX_BYTES  # 8192

#: Slot basename under shm root (per SPEC §7.1 + slot_specs.rs).
SLOT_NAME: str = "sensor_cache_outer_mind"

#: Schema version for sensor_cache_outer_mind.bin payload shape.
#: Bump when the source-dict key set or msgpack envelope changes.
SCHEMA_VERSION: int = 1

#: Throttle period for repeated WARN logs (avoid log flood).
_WARN_THROTTLE_EVERY: int = 60

#: Backoff after run-loop crash before restart (per SPEC §11.B
#: line 1236 in-process exception handler).
_RESTART_BACKOFF_S: float = 5.0

#: Canonical source-dict key set for outer_mind. Mirrors the keys
#: ``OuterTrinityCollector._collect_outer_mind`` (5D base) and
#: ``outer_mind_tensor.collect_outer_mind_15d`` (15D extended)
#: consume from the parent's ``_gather_outer_trinity_sources()`` dict.
#:
#: Excludes non-msgpackable instances (e.g. ``observatory_db``);
#: instead uses the parent's pre-extracted scalar counts
#: ``art_count_100`` / ``art_count_500`` / ``audio_count_*`` per the
#: A.8.4 "parent flattens" pattern (see plugin.py:3110-3124).
SOURCE_KEYS: tuple[str, ...] = (
    "uptime_seconds",
    "art_count_100",          # 5D base
    "audio_count_100",        # 5D base
    "art_count_500",          # 15D extended
    "audio_count_500",        # 15D extended
    "memory_status",          # 5D + 15D
    "assessment_stats",       # 5D + 15D
    "impulse_stats",          # 5D
    "soul_health",            # 5D
    "agency_stats",           # 15D (action_stats derives from this)
    "social_perception_stats",# 15D (social_stats derives)
    "twin_state",             # 15D
    "anchor_state",           # 15D
    "bus_stats",              # 15D (guardian_stats derives)
    "helper_statuses",        # 15D supplementary
    "llm_avg_latency",        # 15D research_stats input
    # Step 3 §3.1 additions for §4.2 P2 outer_mind redesigned dims
    # (Phase 1 §23.8 ports — 4 redesigned thinking dims + wiring fixes).
    "meta_cgn_stats",         # thinking[0,1,14] (knowledge_helpful_ratio, usage_gini, eureka_accelerated_per_hour)
    "cgn_stats",              # thinking[0,14] (avg_reward_norm, grounded_density)
    "memory_stats",           # thinking[0,1,2] (directive_alignment, learning_velocity)
    "language_stats",         # willing[14] (teacher_sessions_last_hour)
    "events_teacher_stats",   # thinking[2] (felt_experiences_24h)
    "social_x_gateway_stats", # willing[11] (posts_last_hour, replies_last_hour, t_since_last_event)
    "output_verifier_stats",  # willing[13] (rejected_per_hour)
    "jailbreak_alerts_stats", # willing[13] (defended_per_hour)
    # Phase C dissolution (2026-05-22) — D-SPEC-101 willing[80-84] ~90s breath;
    # titan-outer-mind-rs reads willing_window (was only on the deleted
    # OUTER_SOURCES_SNAPSHOT broadcast). Re-homed into the mind sidecar provider.
    "willing_window",
)


SourcesProvider = Callable[[], dict[str, Any]]


# ── Sidecar class ───────────────────────────────────────────────────


class OuterMindSensorRefresh:
    """
    SPEC §9.D outer-mind sensor refresh sidecar.

    Polls the in-process ``sources_provider`` callable at
    ``REFRESH_PERIOD_S``, msgpack-encodes the canonical source-dict
    projected onto ``SOURCE_KEYS``, and writes the result to
    ``sensor_cache_outer_mind.bin`` via SeqLock.

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
        """Top-level entry point with in-process restart loop."""
        logger.info(
            "outer_mind_sensor_refresh starting "
            "(period=%.1fs, slot=%s, max_bytes=%d, schema_v=%d)",
            self._refresh_period_s, SLOT_NAME,
            MAX_PAYLOAD_BYTES, SCHEMA_VERSION,
        )
        try:
            self._writer = self._build_writer()
        except Exception:
            logger.critical(
                "outer_mind_sensor_refresh: failed to build SeqLock writer "
                "for %s — sidecar will retry every %.1fs:\n%s",
                SLOT_NAME, _RESTART_BACKOFF_S, traceback.format_exc(),
            )

        while not self._stop.is_set():
            try:
                await self._refresh_loop()
            except asyncio.CancelledError:
                logger.info("outer_mind_sensor_refresh cancelled — exiting")
                raise
            except Exception:
                logger.critical(
                    "outer_mind_sensor_refresh _refresh_loop crashed; "
                    "restarting after %.1fs backoff:\n%s",
                    _RESTART_BACKOFF_S, traceback.format_exc(),
                )
                # Plain asyncio.sleep — see _refresh_loop comment.
                await asyncio.sleep(_RESTART_BACKOFF_S)

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        logger.info(
            "outer_mind_sensor_refresh stopped after %d ticks "
            "(provider_failures=%d, oversize_failures=%d, write_failures=%d)",
            self._tick_count, self._provider_failure_count,
            self._oversize_failure_count, self._write_failure_count,
        )

    async def stop(self) -> None:
        """Signal the sidecar to exit. Idempotent."""
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
        # Plain asyncio.sleep instead of wait_for(stop.wait()): self._stop
        # is an asyncio.Event constructed in main thread context but awaited
        # in this sidecar's dedicated daemon thread (own asyncio.run loop)
        # — silent cross-event-loop binding hangs the await forever
        # (verified 2026-05-10 T3 py-spy). is_set() is cross-thread-safe.
        while not self._stop.is_set():
            tick_start = time.monotonic()
            self._refresh_and_write()
            self._tick_count += 1

            elapsed = time.monotonic() - tick_start
            sleep_for = max(0.0, self._refresh_period_s - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def _refresh_and_write(self) -> None:
        # Stage 1: snapshot
        try:
            raw = self._sources_provider()
        except Exception:
            self._provider_failure_count += 1
            if (
                self._provider_failure_count == 1
                or self._provider_failure_count % _WARN_THROTTLE_EVERY == 0
            ):
                logger.warning(
                    "outer_mind_sensor_refresh sources_provider raised "
                    "(failure_count=%d, tick=%d):\n%s",
                    self._provider_failure_count, self._tick_count,
                    traceback.format_exc(),
                )
            return

        # Stage 2: project onto canonical key set
        sources = self._normalize_sources(raw)

        # Stage 3: encode + write
        try:
            payload = msgpack.packb(sources, use_bin_type=True)
        except (TypeError, ValueError) as exc:
            self._oversize_failure_count += 1
            logger.error(
                "outer_mind_sensor_refresh msgpack encode failed "
                "(tick=%d): %s — keys=%s",
                self._tick_count, exc, sorted(sources.keys()),
            )
            return

        if len(payload) > MAX_PAYLOAD_BYTES:
            self._oversize_failure_count += 1
            # 2026-05-13: surface per-key byte sizes once every
            # _WARN_THROTTLE_EVERY oversize ticks so we can identify
            # which upstream source is bloating without re-instrumenting
            # via a separate probe. Sorted descending; only logs the
            # top 5 to keep log volume bounded.
            top_offenders = ""
            if (self._oversize_failure_count == 1
                    or self._oversize_failure_count % _WARN_THROTTLE_EVERY == 0):
                try:
                    sizes = sorted(
                        ((len(msgpack.packb({k: v}, use_bin_type=True)), k)
                         for k, v in sources.items()),
                        reverse=True,
                    )[:5]
                    top_offenders = " top_5_keys=[" + ", ".join(
                        f"{k}:{sz}B" for sz, k in sizes) + "]"
                except Exception:
                    top_offenders = " (per-key sizing failed)"
            logger.critical(
                "outer_mind_sensor_refresh payload %dB > MAX %dB "
                "(tick=%d) — slot retains last-known. Investigate "
                "upstream source-shape drift; do NOT silently truncate.%s",
                len(payload), MAX_PAYLOAD_BYTES, self._tick_count,
                top_offenders,
            )
            return

        if self._writer is None:
            try:
                self._writer = self._build_writer()
            except Exception:
                self._write_failure_count += 1
                if self._write_failure_count == 1 or self._write_failure_count % _WARN_THROTTLE_EVERY == 0:
                    logger.warning(
                        "outer_mind_sensor_refresh writer build failed "
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
                    "outer_mind_sensor_refresh write_variable failed "
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
