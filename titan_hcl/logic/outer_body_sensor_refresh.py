"""
outer_body_sensor_refresh — Phase C C-S6 Python sensor sidecar (outer body).

SPEC §9.D Sensor refresh sidecars + SPEC §18.1 stale handling +
SPEC §7.1 sensor_cache_<level>_<sense>.bin layout.

Hosted in ``titan_HCL`` (today's ``titan_hcl`` parent) as an asyncio
task supervised by ``guardian_HCL`` via in-process exception handler
(SPEC §11.B line 1236). Aggregates the ``sources`` dict that
``titan-outer-body-rs`` needs to compute its 5DT V6 body-felt tensor
(per ``outer_trinity.py:_collect_outer_body``), msgpack-encodes it,
and writes the result to ``/dev/shm/titan_<id>/sensor_cache_outer_body.bin``
via SeqLock.

──────────────────────────────────────────────────────────────────────
Cadence (SPEC §9.D line 1029)
──────────────────────────────────────────────────────────────────────
Refresh period = ``OUTER_BODY_TICK_BASE_S`` (45.0 s — G13 body-slowest
per D-SPEC-100; strict 1:3:9 with mind 15s / spirit 5s, mirroring
OUTER_BODY_BUS_PUBLISH_INTERVAL_S). The Rust daemon ticks at full
Schumann body (7.83 Hz); this constant only sets the sidecar
source-refresh cadence + 3× stale threshold (135s).

Source proxies inside the in-process registry already poll their own
RPC/file/db endpoints at native rates — e.g. SOL RPC at
``SOLANA_RPC_BALANCE_POLL_INTERVAL_S``. The sidecar snapshots the
latest-known values at every refresh tick. **Zero new RPC traffic** —
sidecar reads from in-process registry only (per O4 lock + O6 lock
in ``PLAN_microkernel_phase_c_s6_outer_trinity.md`` §12.1).

──────────────────────────────────────────────────────────────────────
Slot byte layout (SPEC §7.1 row "sensor_cache_<level>_<sense>.bin")
──────────────────────────────────────────────────────────────────────
  [0:24]      24-byte SeqLock header (HEADER_STRUCT="<IIQII":
              seq + schema_version + wall_ns + payload_bytes + crc32)
  [24:24+N]   msgpack-encoded source dict (N ≤
              ``OUTER_SENSOR_CACHE_BODY_MAX_BYTES`` = 8192)

──────────────────────────────────────────────────────────────────────
Source-dict shape (msgpack-encoded)
──────────────────────────────────────────────────────────────────────
Mirrors today's ``sources`` dict consumed by
``OuterTrinityCollector._collect_outer_body``:

  {
    "agency_stats":         dict | None,
    "helper_statuses":      dict | None,
    "bus_stats":            dict | None,
    "system_sensor_stats":  dict | None,
    "network_monitor_stats":dict | None,
    "tx_latency_stats":     dict | None,
    "block_delta_stats":    dict | None,
    "anchor_state":         dict | None,
    "sol_balance":          float | int | None,
  }

The Rust daemon decodes msgpack, runs the V6 body-felt formula
(SPEC G10 ground_up applied to all 5 dims; G7 filter_down clamps
[0.3, 3.0]; per-dim source weighting from outer_trinity.py:165-200),
and writes 5 × float32 LE to ``outer_body_5d.bin``.

──────────────────────────────────────────────────────────────────────
Stale handling (SPEC §18.1 line 1867)
──────────────────────────────────────────────────────────────────────
This sidecar always bumps ``wall_ns`` on every tick (success OR
sources-provider-raises), so consumers see cache freshness while the
slot's payload bytes remain last-known. The Rust daemon's stale
threshold is ``wall_ns < now − OUTER_CACHE_STALE_CADENCE_MULTIPLIER ×
cadence`` (= 30s for outer_body); the sidecar's 10s tick keeps cache
well within freshness even under transient source-provider failure.

──────────────────────────────────────────────────────────────────────
Failure semantics
──────────────────────────────────────────────────────────────────────
- ``sources_provider()`` raises → caught, WARN log throttled (first +
  every Nth subsequent), last successfully-encoded payload retained.
  Wall-clock advances; next tick retries.
- msgpack encode > MAX_PAYLOAD_BYTES → CRITICAL log; skip write
  (slot stays last-known). Indicates upstream source-shape drift —
  investigate; never silently truncate (truncation would corrupt
  msgpack bytes the daemon decodes).
- shm write OSError → WARN log; retry next tick.
- Asyncio task itself crashes → ``run()``'s outer try/except catches,
  logs CRITICAL with traceback, sleeps ``_RESTART_BACKOFF_S``,
  re-enters ``_refresh_loop`` (in-process exception handler per
  SPEC §11.B line 1236).

──────────────────────────────────────────────────────────────────────
Flag-gating (SPEC §3.0 + master plan §12.1 + PLAN §1.1 item 9)
──────────────────────────────────────────────────────────────────────
Sidecar runs **unconditionally** (NOT flag-gated). Writing to a slot
the Rust daemon doesn't yet read is zero-cost; this gives a 7-day
soak window for sidecar logic before C-S7 first flag-flip activates
``titan-outer-body-rs``.
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
    OUTER_BODY_TICK_BASE_S,
    OUTER_SENSOR_CACHE_BODY_MAX_BYTES,
)
from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

#: Refresh period (s). Sourced from generated TOML constant; matches
#: titan-outer-body-rs daemon's natural cadence per SPEC §18.1.
REFRESH_PERIOD_S: float = OUTER_BODY_TICK_BASE_S  # 45.0 (D-SPEC-100)

#: Max msgpack payload size (bytes). Matches slot_specs.rs C-S2 cap.
MAX_PAYLOAD_BYTES: int = OUTER_SENSOR_CACHE_BODY_MAX_BYTES  # 8192

#: Slot basename under shm root (per SPEC §7.1 + slot_specs.rs).
SLOT_NAME: str = "sensor_cache_outer_body"

#: Schema version for sensor_cache_outer_body.bin payload shape.
#: Bump when the source-dict key set or msgpack envelope changes.
SCHEMA_VERSION: int = 1

#: Throttle period for repeated WARN logs (avoid log flood).
_WARN_THROTTLE_EVERY: int = 60  # log first + every 60th occurrence

#: Backoff after run-loop crash before restart (per SPEC §11.B
#: line 1236 in-process exception handler).
_RESTART_BACKOFF_S: float = 5.0

#: Canonical source-dict key set (msgpack stability — Rust daemon's
#: decoder must recognize all keys; missing keys are treated as None).
SOURCE_KEYS: tuple[str, ...] = (
    "agency_stats",
    "helper_statuses",
    "bus_stats",
    "system_sensor_stats",
    "network_monitor_stats",
    "tx_latency_stats",
    "block_delta_stats",
    "anchor_state",
    "sol_balance",
    # Step 3 §3.1 additions for §4.1 P1 outer_body thermal port:
    # SPEC §23.7 dim 4 thermal REDESIGNED → 0.40 * hormonal_heat where
    # hormonal_heat = mean(IMPULSE, VIGILANCE) per spirit_proxy hormones.
    "hormone_levels",
    # Phase C dissolution (2026-05-22) — D-SPEC-101 breath signals now reach
    # titan-outer-body-rs (it already reads these; previously they were only on
    # the deleted OUTER_SOURCES_SNAPSHOT broadcast → absent from the slot → the
    # daemon fell back to defaults). Re-homed into the body sidecar provider.
    "outer_body_change",   # entropy[68] + thermal[69] rate-of-change breath
    "pi_heartbeat_hrv",    # interoception[65] = π-heartbeat 24h HRV
)


SourcesProvider = Callable[[], dict[str, Any]]


# ── Sidecar class ───────────────────────────────────────────────────


class OuterBodySensorRefresh:
    """
    SPEC §9.D outer-body sensor refresh sidecar.

    Polls the in-process ``sources_provider`` callable at
    ``REFRESH_PERIOD_S``, msgpack-encodes the canonical source-dict,
    and writes the result to ``sensor_cache_outer_body.bin`` via
    SeqLock.

    Lifecycle::

        sidecar = OuterBodySensorRefresh(sources_provider, titan_id="T1")
        task = asyncio.create_task(sidecar.run())
        ...
        await sidecar.stop()  # sets stop_event; awaits task drain

    Single-instance assumption (SPEC §9.D + state_registry.py:199 —
    "one writer per registry"). Booting two instances against the
    same shm path will produce SeqLock counter conflicts on read.
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
        """
        Top-level entry point. Wraps ``_refresh_loop`` in an
        in-process restart loop per SPEC §11.B line 1236.

        Returns when ``stop_event`` is set; never raises (unrecoverable
        errors are logged CRITICAL and trigger backoff-then-restart).
        """
        logger.info(
            "outer_body_sensor_refresh starting "
            "(period=%.1fs, slot=%s, max_bytes=%d, schema_v=%d)",
            self._refresh_period_s, SLOT_NAME,
            MAX_PAYLOAD_BYTES, SCHEMA_VERSION,
        )
        try:
            self._writer = self._build_writer()
        except Exception:
            logger.critical(
                "outer_body_sensor_refresh: failed to build SeqLock writer "
                "for %s — sidecar will retry every %.1fs:\n%s",
                SLOT_NAME, _RESTART_BACKOFF_S, traceback.format_exc(),
            )
            # Even writer-build can fail (mmap permission, disk full).
            # Restart loop handles it.

        while not self._stop.is_set():
            try:
                await self._refresh_loop()
            except asyncio.CancelledError:
                logger.info("outer_body_sensor_refresh cancelled — exiting")
                raise
            except Exception:
                logger.critical(
                    "outer_body_sensor_refresh _refresh_loop crashed; "
                    "restarting after %.1fs backoff:\n%s",
                    _RESTART_BACKOFF_S, traceback.format_exc(),
                )
                # Plain asyncio.sleep — see _refresh_loop docstring re:
                # cross-event-loop Event binding hazard.
                await asyncio.sleep(_RESTART_BACKOFF_S)

        # Graceful exit
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass  # best-effort on shutdown
            self._writer = None
        logger.info(
            "outer_body_sensor_refresh stopped after %d ticks "
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
        """Construct the variable-size SeqLock writer for the slot."""
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
        """Run refresh ticks until ``stop_event`` is set.

        Uses plain ``asyncio.sleep`` instead of ``wait_for(stop.wait())``
        to avoid asyncio.Event cross-event-loop binding hazard: this
        sidecar runs in a dedicated daemon thread with its own
        ``asyncio.run()`` event loop, but ``self._stop`` was constructed
        in ``__init__`` from the main thread's context. Awaiting that
        Event from the new loop silently never wakes (verified 2026-05-10
        T3 py-spy: each sidecar ticked exactly once then hung in select
        forever). ``stop.is_set()`` is a cross-thread-safe atomic flag
        read; checking it at loop top is sufficient for stop responsiveness
        (worst case: 1 period of latency on shutdown).
        """
        while not self._stop.is_set():
            tick_start = time.monotonic()

            self._refresh_and_write()
            self._tick_count += 1

            elapsed = time.monotonic() - tick_start
            sleep_for = max(0.0, self._refresh_period_s - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def _refresh_and_write(self) -> None:
        """
        One refresh-and-write iteration. Catches sources-provider
        failure + oversize + write failure independently. Always
        attempts a write (success ⇒ fresh wall_ns; failure ⇒
        observed in counters).
        """
        # Stage 1: snapshot sources (catches provider raise)
        try:
            raw = self._sources_provider()
        except Exception:
            self._provider_failure_count += 1
            if (
                self._provider_failure_count == 1
                or self._provider_failure_count % _WARN_THROTTLE_EVERY == 0
            ):
                logger.warning(
                    "outer_body_sensor_refresh sources_provider raised "
                    "(failure_count=%d, tick=%d):\n%s",
                    self._provider_failure_count, self._tick_count,
                    traceback.format_exc(),
                )
            return  # next tick retries; slot retains last successful payload

        # Stage 2: normalize to canonical source-dict shape
        sources = self._normalize_sources(raw)

        # Stage 3: encode + write
        try:
            payload = msgpack.packb(sources, use_bin_type=True)
        except (TypeError, ValueError) as exc:
            self._oversize_failure_count += 1
            logger.error(
                "outer_body_sensor_refresh msgpack encode failed "
                "(tick=%d): %s — keys=%s",
                self._tick_count, exc, sorted(sources.keys()),
            )
            return

        if len(payload) > MAX_PAYLOAD_BYTES:
            self._oversize_failure_count += 1
            logger.critical(
                "outer_body_sensor_refresh payload %dB > MAX %dB "
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
                        "outer_body_sensor_refresh writer build failed "
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
                    "outer_body_sensor_refresh write_variable failed "
                    "(failure_count=%d, tick=%d):\n%s",
                    self._write_failure_count, self._tick_count,
                    traceback.format_exc(),
                )

    @staticmethod
    def _normalize_sources(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Project raw sources dict onto the canonical SOURCE_KEYS.
        Missing keys → None. Extra keys dropped (keeps msgpack envelope
        shape stable for the Rust daemon's decoder).
        """
        return {key: raw.get(key) for key in SOURCE_KEYS}
