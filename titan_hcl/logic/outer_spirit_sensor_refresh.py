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

  - REFRESH_PERIOD_S = OUTER_SPIRIT_TICK_BASE_S (5.0 s) — G13
    spirit-fastest per D-SPEC-100 (strict 1:3:9: spirit 5s / mind 15s /
    body 45s), mirroring OUTER_SPIRIT_BUS_PUBLISH_INTERVAL_S + the 45D
    dim count. The Rust daemon ticks at full Schumann spirit (70.47 Hz);
    this constant only sets the sidecar source-refresh cadence + 3×
    stale threshold (15s). Load-safe: _gather_outer_sources reads
    in-process + bus-cached stats; _heavy_stats_refresher owns DB/RPC.
  - SLOT_NAME = "sensor_cache_outer_spirit"
  - MAX_PAYLOAD_BYTES = OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES (8192)
  - SOURCE_KEYS = canonical RAW upstream key set provided by
    ``TitanHCL._gather_outer_sources``. The Rust outer-spirit daemon
    runs the ``_collect_extended`` preprocessing internally (port at
    ``titan-outer-spirit-rs/src/tick_loop.rs::project_outer_spirit_45d``
    closes rFP D1 / GAP-CS6-001), so the sidecar writes raw upstream
    stats (mirroring ``outer_mind_sensor_refresh.py`` and
    ``outer_body_sensor_refresh.py``). Keys: agency_stats /
    assessment_stats / memory_status / social_perception_stats /
    art_count_500 / audio_count_500 / uptime_seconds / hormone_levels /
    solana_stats / recovery_stats / history / soul_health.

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

from titan_hcl._phase_c_constants import (
    OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES,
    OUTER_SPIRIT_TICK_BASE_S,
)
from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

REFRESH_PERIOD_S: float = OUTER_SPIRIT_TICK_BASE_S  # 5.0 (D-SPEC-100)
MAX_PAYLOAD_BYTES: int = OUTER_SENSOR_CACHE_SPIRIT_MAX_BYTES  # 8192
SLOT_NAME: str = "sensor_cache_outer_spirit"
SCHEMA_VERSION: int = 1
_WARN_THROTTLE_EVERY: int = 60
_RESTART_BACKOFF_S: float = 5.0

#: Canonical RAW upstream source-dict key set provided by
#: ``TitanHCL._gather_outer_sources``. The Rust daemon
#: (``project_outer_spirit_45d``) does the ``_collect_extended``
#: preprocessing internally — sidecar writes RAW stats only.
#: All keys are msgpack-friendly plain types.
SOURCE_KEYS: tuple[str, ...] = (
    "agency_stats",            # → action_stats / guardian_stats / sovereignty_ratio
    "assessment_stats",        # → assessment_ext (mean_score / trend / score_variance)
    "memory_status",           # → memory_stats / social_stats.interactions
    "social_perception_stats", # → social_stats.sentiment / connection
    "art_count_500",           # → creative_stats.art_count
    "audio_count_500",         # → creative_stats.audio_count
    "uptime_seconds",          # → uptime_ratio
    "hormone_levels",          # → ANANDA[11] creative_tension
    "solana_stats",            # → SAT[0,5,7,13]
    "recovery_stats",          # → SAT[10]
    "history",                 # → CHIT[10,11,14] + ANANDA[10,11,13]
    "soul_health",             # → outer_lower_spirit 5D (sidecar pass-through)
    # Step 3 §3.1 additions for §4.3 P3 outer_spirit redesigned dims
    # (Phase 1/2 §23.9 ports — ~32 stale dims need new inputs).
    "meta_cgn_stats",          # CHIT[0,3,5,6,7,13] + ANANDA[5,7,9]
    "cgn_stats",               # CHIT[5,6] + ANANDA[9]
    "memory_stats",            # CHIT[6] + ANANDA[5]
    "knowledge_graph_stats",   # CHIT[0] world_model_depth (KG node/edge counts)
    "events_teacher_stats",    # ANANDA[9] discovery_value (felt_experiences_to_action_rate)
    "jailbreak_alerts_stats",  # SAT[3] boundary_enforcement + CHIT[2] threat_discernment
    "output_verifier_stats",   # SAT[3] + CHIT[2] (rejected_24h, violation_events_24h, high_severity)
    "anchor_state",            # SAT[10,13] (consecutive_failures, anchor_count)
    "bus_stats",               # ANANDA[3] system_harmony (1 - dropped/published)
    "expression_translator_stats", # SAT[2] action_sovereignty + CHIT[28] causal_understanding
    "outer_spirit_history_stats",  # SAT[11] env_adapt + CHIT[10/14] + ANANDA[10/11/13]
    "community_engagement_stats",  # ANANDA[6] community_connection + ANANDA[8] expression_reach (Phase 2.5.E per-Titan)
    "inner_memory_stats",      # for inner_spirit cross-block (deferred to Step 9 but kept here for symmetry)
    # rFP §9 closure batch 2026-05-12 — outer_spirit deferred dims (SAT[1,5,7] + ANANDA[7]):
    "genesis_record_exists",   # SAT[5] origin_anchoring (boot-once filesystem check, bool)
    "world_footprint_extra_counts",  # SAT[7] world_footprint per SPEC §23.3 (arweave_inscriptions + meditation_memos)
    "vocab_stats",             # ANANDA[7] capability_growth (producible_delta_24h)
    "reflex_stats",            # ANANDA[7] capability_growth (distinct_fired_24h)
    # Phase C dissolution (2026-05-22) — D-SPEC-101 outer-spirit breath signals;
    # titan-outer-spirit-rs reads expr_window (SAT[2]/ANANDA[4] cluster) +
    # outer_spirit_self_change (SAT[0] world_recognition). Were only on the
    # deleted OUTER_SOURCES_SNAPSHOT broadcast. Re-homed into the spirit provider.
    "expr_window",
    "outer_spirit_self_change",
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
                # Plain asyncio.sleep — see _refresh_loop comment.
                await asyncio.sleep(_RESTART_BACKOFF_S)

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
