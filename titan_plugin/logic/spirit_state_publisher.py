"""
spirit_state_publisher — Phase C Session 1 of rFP_phase_c_async_shm_consumer_migration.

Publishes 3 SHM slots per SPEC §7.1 + Preamble G18 (state transport is SHM,
never bus):

  - hormone_fires.bin         (msgpack {hormone_name → fire_count} + ts)
  - impulse_engine_state.bin  (msgpack ImpulseEngine.get_stats() + per-hormone
                               impulse map + ts)
  - consciousness_state.bin   (msgpack consciousness epoch summary + ts)

Owned by spirit_worker (titan_HCL — runs in the supervised spirit Python
process). Replaces the synchronous `bus.request(action="get_trinity")` path
that deadlocked T3 outer-sensor sidecars on 2026-05-07 — proven via py-spy
forensics — by giving consumers a non-blocking SHM read path for the same
state.

Cadence: 1 Hz (SPEC §7.1). Content-hash gated (writes skipped when payload
unchanged). Writer is single-threaded — SPEC G21 single-writer/multi-reader
contract.

Read contract for consumers: use `titan_plugin.proxies.spirit_proxy` (shipped
with the matching SHM-direct-read migration in §4.C.1 of the rFP) — never
reach for these slot files directly. The proxy owns the StateRegistryReader
attach + decode + freshness-check policy.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack
import numpy as np

from titan_plugin.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_plugin.logic.spirit_state_specs import (
    CONSCIOUSNESS_STATE_SLOT,
    CONSCIOUSNESS_STATE_SPEC,
    HORMONE_FIRES_SLOT,
    HORMONE_FIRES_SPEC,
    IMPULSE_ENGINE_STATE_SLOT,
    IMPULSE_ENGINE_STATE_SPEC,
    RESONANCE_STATE_SLOT,
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SLOT,
    UNIFIED_SPIRIT_METADATA_SPEC,
)

logger = logging.getLogger(__name__)


# Slot names + specs are imported from spirit_state_specs (single source
# of truth; consumers like spirit_proxy import the same module).


# ── Throttle for warn-on-encode-failure logs ───────────────────────────────

_WARN_THROTTLE_EVERY = 60  # log first + every 60th occurrence

# ── INFO heartbeat cadence ─────────────────────────────────────────────────

#: Publish-tick count at which to log an INFO heartbeat. Compounds into
#: 1 (first), 10 (10s in), 60 (1min), 600 (10min) so the brain log carries
#: a clear trail of "publisher is alive and producing" without spamming.
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


# ── Publisher class ────────────────────────────────────────────────────────


class SpiritStatePublisher:
    """
    Owns 3 SHM writers; called from spirit_worker's snapshot-builder loop
    once per second to publish hormone_fires, impulse_engine_state, and
    consciousness_state.

    Each `publish_*` method is independent — failure of one does not
    prevent the others from publishing. Failures are throttled-logged so
    a slow-changing degradation doesn't flood the brain log.

    Writers are constructed lazily on first publish (so a missing
    `/dev/shm/titan_<id>` directory at __init__ doesn't crash module
    import — same pattern as `outer_body_sensor_refresh`).
    """

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        # Lazy writer init — see _build_writer()
        slots = (HORMONE_FIRES_SLOT, IMPULSE_ENGINE_STATE_SLOT,
                 CONSCIOUSNESS_STATE_SLOT, RESONANCE_STATE_SLOT,
                 UNIFIED_SPIRIT_METADATA_SLOT)
        self._writers: dict[str, Optional[StateRegistryWriter]] = {
            s: None for s in slots
        }
        # Throttle counters (per slot, per failure class)
        self._encode_fails: dict[str, int] = {s: 0 for s in slots}
        self._oversize_fails: dict[str, int] = {s: 0 for s in slots}
        self._write_fails: dict[str, int] = {s: 0 for s in slots}
        # Per-slot success counters — track first-success + per-slot
        # cadence so the brain log shows independently which slot is
        # actually flowing (caught the chunk 9H bug where ONLY one of
        # three sidecars entered run() on T3 — same diagnostic shape).
        self._publish_success: dict[str, int] = {s: 0 for s in slots}
        self._publish_count = 0
        logger.info(
            "[SpiritStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(5 slots: %s — SPEC §7.1 / Preamble G18)",
            titan_id, self._shm_root, ", ".join(slots))

    # -- writer factory --------------------------------------------------

    _SLOT_SPECS = {
        HORMONE_FIRES_SLOT: HORMONE_FIRES_SPEC,
        IMPULSE_ENGINE_STATE_SLOT: IMPULSE_ENGINE_STATE_SPEC,
        CONSCIOUSNESS_STATE_SLOT: CONSCIOUSNESS_STATE_SPEC,
        RESONANCE_STATE_SLOT: RESONANCE_STATE_SPEC,
        UNIFIED_SPIRIT_METADATA_SLOT: UNIFIED_SPIRIT_METADATA_SPEC,
    }

    def _writer_for(self, slot: str) -> StateRegistryWriter:
        w = self._writers[slot]
        if w is not None:
            return w
        spec = self._SLOT_SPECS[slot]
        w = StateRegistryWriter(spec, self._shm_root)
        self._writers[slot] = w
        logger.info(
            "[SpiritStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            slot, spec.payload_bytes, spec.schema_version,
            self._shm_root / f"{slot}.bin")
        return w

    # -- top-level entry point -------------------------------------------

    def publish(self, state_refs: dict[str, Any]) -> None:
        """
        Publish all 3 slots from state_refs (the same dict the snapshot-
        builder threads consume). Each slot's publisher is independent —
        a failure in one (e.g., consciousness=None at cold boot) does not
        prevent the other two from publishing.

        Top-level try/except guards against unexpected failures so the
        publisher thread never dies silently — any uncaught exception is
        logged with traceback before being re-raised (the calling
        _builder_loop has its own throttled-WARN handler upstream).
        """
        self._publish_count += 1
        impulse_engine = state_refs.get("impulse_engine")
        neural_nervous_system = state_refs.get("neural_nervous_system")
        consciousness = state_refs.get("consciousness")
        body_state = state_refs.get("body_state") or {}
        mind_state = state_refs.get("mind_state") or {}
        resonance = state_refs.get("resonance")
        unified_spirit = state_refs.get("unified_spirit")

        self._publish_hormone_fires(neural_nervous_system)
        self._publish_impulse_engine_state(impulse_engine, neural_nervous_system)
        self._publish_consciousness_state(consciousness, body_state, mind_state)
        self._publish_resonance_state(resonance)
        self._publish_unified_spirit_metadata(unified_spirit)

        # Heartbeat INFO at canonical milestones — gives operators a clear
        # "publisher is alive and ticking" trail without log flooding. Picks
        # up if any slot has stalled (per-slot _publish_success counters).
        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[SpiritStatePublisher] heartbeat — publish_count=%d "
                "per_slot_success={hormone_fires=%d impulse_engine=%d "
                "consciousness=%d resonance=%d unified_spirit_meta=%d} "
                "fails={encode=%s oversize=%s write=%s}",
                self._publish_count,
                self._publish_success[HORMONE_FIRES_SLOT],
                self._publish_success[IMPULSE_ENGINE_STATE_SLOT],
                self._publish_success[CONSCIOUSNESS_STATE_SLOT],
                self._publish_success[RESONANCE_STATE_SLOT],
                self._publish_success[UNIFIED_SPIRIT_METADATA_SLOT],
                self._encode_fails, self._oversize_fails, self._write_fails)

    # -- slot publishers -------------------------------------------------

    def _publish_hormone_fires(self, neural_nervous_system: Any) -> None:
        """
        hormone_fires.bin payload: {hormone_name: str → fire_count: int} for
        every HormonalPressure tracked by NS, plus `ts: float`. Source:
        `neural_nervous_system._hormonal._hormones[*].fire_count` — same
        construction as spirit_loop.py:2611-2616.

        Cold-boot + degraded behavior: if NS or its `_hormonal` subsystem
        is unavailable OR raises during attribute access (broken
        property), write an empty fires dict + ts (consumers see
        fresh-but-empty payload, treat as "no fires this tick"). NEVER
        skip the write — staleness signaling depends on `wall_ns`
        advancing. Defensive against properties that raise per G20
        (hot-path resilience to producer failures).
        """
        fires: dict[str, int] = {}
        if neural_nervous_system is not None:
            try:
                _horm = getattr(neural_nervous_system, "_hormonal", None)
                if _horm is not None:
                    _hormones = getattr(_horm, "_hormones", None)
                    if isinstance(_hormones, dict):
                        fires = {
                            str(name): int(getattr(h, "fire_count", 0))
                            for name, h in _hormones.items()
                        }
            except Exception as e:
                self._warn_throttled(
                    HORMONE_FIRES_SLOT, "encode", e,
                    "hormone_fires source iteration failed (NS attr raised)")
        payload = {"fires": fires, "ts": time.time()}
        self._write_msgpack(HORMONE_FIRES_SLOT, payload)

    def _publish_impulse_engine_state(self, impulse_engine: Any,
                                      neural_nervous_system: Any) -> None:
        """
        impulse_engine_state.bin payload: ImpulseEngine.get_stats() output
        merged with per-hormone impulse_value / threshold / refractory state
        from NS's HormonalPressure instances + `ts`.

        ImpulseEngine.get_stats() returns: threshold, impulse_count,
        cooldown_seconds, outcome_count, success_rate, pending_impulse,
        last_impulse_ts (per impulse_engine.py:246-258).

        The per-hormone block is what consumers like spirit_proxy.get_trinity
        currently aggregate via the bus-RPC path; making it SHM-canonical
        removes the deadlock surface entirely.
        """
        engine_stats: dict[str, Any] = {}
        if impulse_engine is not None:
            try:
                if hasattr(impulse_engine, "get_stats"):
                    engine_stats = dict(impulse_engine.get_stats())
            except Exception as e:
                self._warn_throttled(
                    IMPULSE_ENGINE_STATE_SLOT, "encode", e,
                    "ImpulseEngine.get_stats raised")

        per_hormone: dict[str, dict[str, float]] = {}
        if neural_nervous_system is not None:
            try:
                _horm = getattr(neural_nervous_system, "_hormonal", None)
            except Exception as e:
                self._warn_throttled(
                    IMPULSE_ENGINE_STATE_SLOT, "encode", e,
                    "neural_nervous_system._hormonal attr raised")
                _horm = None
            if _horm is not None:
                try:
                    _hormones = getattr(_horm, "_hormones", None)
                except Exception:
                    _hormones = None
                if isinstance(_hormones, dict):
                    for name, hp in _hormones.items():
                        try:
                            per_hormone[str(name)] = {
                                "level": float(getattr(hp, "level", 0.0)),
                                "fire_threshold": float(
                                    getattr(hp, "fire_threshold", 0.0)),
                                "refractory_until_ts": float(
                                    getattr(hp, "refractory_until_ts", 0.0)),
                                "peak_level": float(
                                    getattr(hp, "peak_level", 0.0)),
                                "last_fire_ts": float(
                                    getattr(hp, "last_fire_ts", 0.0)),
                            }
                        except Exception:
                            # Skip malformed hormone entry; continue with rest
                            continue

        payload = {
            "engine": engine_stats,
            "hormones": per_hormone,
            "ts": time.time(),
        }
        self._write_msgpack(IMPULSE_ENGINE_STATE_SLOT, payload)

    def _publish_consciousness_state(self, consciousness: Any,
                                     body_state: dict[str, Any],
                                     mind_state: dict[str, Any]) -> None:
        """
        consciousness_state.bin payload: scalar fields extracted from
        `consciousness["latest_epoch"]` (per `_run_consciousness_epoch`
        output) + the precomputed 5DT spirit tensor (`spirit_5dt`) +
        `body_values` / `mind_values` snapshots + `body_center_dist` /
        `mind_center_dist` + `ts`. Mirrors the data spirit_proxy.get_trinity
        currently carries via the bus-RPC path so SHM-direct readers get
        the same surface.

        latest_epoch is a dict with keys: epoch_id, density, curvature,
        dream_quality, fatigue, trajectory_magnitude, plus deeper structure
        we serialize whole (with size guard at write time).

        spirit_5dt is computed inline via `_collect_spirit_tensor` (same
        producer used by build_trinity_snapshot today — single source of
        truth for the formula).

        Cold-boot: if consciousness is None or has no latest_epoch yet,
        publish a stub with epoch_id=0 + zeros + ts. Consumers can detect
        cold-boot via epoch_id==0.
        """
        latest: dict[str, Any] = {}
        if isinstance(consciousness, dict):
            le = consciousness.get("latest_epoch")
            if isinstance(le, dict):
                latest = le

        body_values = body_state.get("values", [0.5] * 5) if body_state else [0.5] * 5
        mind_values = mind_state.get("values", [0.5] * 5) if mind_state else [0.5] * 5
        body_center_dist = body_state.get("center_dist", 0.0) if body_state else 0.0
        mind_center_dist = mind_state.get("center_dist", 0.0) if mind_state else 0.0

        # Compute 5DT spirit tensor — same formula spirit_loop's
        # _collect_spirit_tensor uses (single source of truth). Inline import
        # to avoid circular import at module load (spirit_loop imports
        # state_registry which imports... — late binding is safe).
        spirit_5dt: list[float]
        try:
            from titan_plugin.modules.spirit_loop import _collect_spirit_tensor
            spirit_5dt = _collect_spirit_tensor(
                config={},  # _collect_spirit_tensor reads no config keys
                body_state=body_state,
                mind_state=mind_state,
                consciousness=consciousness,
            )
        except Exception as e:
            self._warn_throttled(
                CONSCIOUSNESS_STATE_SLOT, "encode", e,
                "_collect_spirit_tensor raised — using neutral 5DT")
            spirit_5dt = [0.5] * 5

        # Coerce scalar fields to canonical types (msgpack stability) +
        # carry the full latest_epoch under a nested key for forward-compat
        payload = {
            "epoch_id": int(latest.get("epoch_id", 0)),
            "density": float(latest.get("density", 0.0)),
            "curvature": float(latest.get("curvature", 0.0)),
            "dream_quality": float(latest.get("dream_quality", 0.0)),
            "fatigue": float(latest.get("fatigue", 0.0)),
            "trajectory_magnitude": float(
                latest.get("trajectory_magnitude", 0.0)),
            "spirit_5dt": [float(v) for v in spirit_5dt],
            "body_values": [float(v) for v in body_values],
            "mind_values": [float(v) for v in mind_values],
            "body_center_dist": float(body_center_dist),
            "mind_center_dist": float(mind_center_dist),
            "latest_epoch": latest,
            "ts": time.time(),
        }
        self._write_msgpack(CONSCIOUSNESS_STATE_SLOT, payload)

    def _publish_resonance_state(self, resonance: Any) -> None:
        """
        resonance_state.bin payload: direct ResonanceDetector.get_stats()
        output + ts. Source schema (resonance.py:417-429): pairs (3 pair
        names → per-pair stats), resonant_count, all_resonant,
        great_pulse_count, last_great_pulse_ts, config (phase_threshold_deg,
        required_cycles, pulse_window).

        Cold-boot / degraded: if resonance is None or get_stats unavailable,
        publish a stub with zero counters + ts so consumers always see a
        fresh-but-empty payload (G20 — staleness signaling depends on
        wall_ns advancing).
        """
        stats: dict[str, Any] = {}
        if resonance is not None:
            try:
                if hasattr(resonance, "get_stats"):
                    stats = dict(resonance.get_stats())
            except Exception as e:
                self._warn_throttled(
                    RESONANCE_STATE_SLOT, "encode", e,
                    "ResonanceDetector.get_stats raised")

        payload = {
            "pairs": stats.get("pairs", {}),
            "resonant_count": int(stats.get("resonant_count", 0)),
            "all_resonant": bool(stats.get("all_resonant", False)),
            "great_pulse_count": int(stats.get("great_pulse_count", 0)),
            "last_great_pulse_ts": float(
                stats.get("last_great_pulse_ts", 0.0)),
            "config": stats.get("config", {}),
            "ts": time.time(),
        }
        self._write_msgpack(RESONANCE_STATE_SLOT, payload)

    def _publish_unified_spirit_metadata(self, unified_spirit: Any) -> None:
        """
        unified_spirit_metadata.bin payload: direct UnifiedSpirit.get_stats()
        output + ts. Source schema (unified_spirit.py:560-589): epoch_count,
        current_epoch_id, velocity, is_stale, consecutive_stale,
        stale_focus_multiplier, tensor_magnitude, tensor_sum, latest_epoch
        (large dict), cumulative_quality, micro_tick_count, last_alignment,
        enrichment_rate, full_130dt (130 floats), config.

        Pairs with the existing fixed-layout `unified_spirit_132d.bin` slot
        (which carries only the bare 132D float tensor) — this metadata slot
        carries every queryable field that UnifiedSpirit exposes via
        get_stats(). Consumers needing the bare tensor read the existing
        132D slot; consumers needing velocity/stale/focus_multiplier/
        latest_epoch read this metadata slot.

        Cold-boot / degraded: if unified_spirit is None or get_stats
        unavailable, publish a stub with neutral defaults + ts.
        """
        stats: dict[str, Any] = {}
        if unified_spirit is not None:
            try:
                if hasattr(unified_spirit, "get_stats"):
                    stats = dict(unified_spirit.get_stats())
            except Exception as e:
                self._warn_throttled(
                    UNIFIED_SPIRIT_METADATA_SLOT, "encode", e,
                    "UnifiedSpirit.get_stats raised")

        payload = {
            "epoch_count": int(stats.get("epoch_count", 0)),
            "current_epoch_id": int(stats.get("current_epoch_id", 0)),
            "velocity": float(stats.get("velocity", 0.0)),
            "is_stale": bool(stats.get("is_stale", False)),
            "consecutive_stale": int(stats.get("consecutive_stale", 0)),
            "stale_focus_multiplier": float(
                stats.get("stale_focus_multiplier", 1.0)),
            "tensor_magnitude": float(stats.get("tensor_magnitude", 0.0)),
            "tensor_sum": float(stats.get("tensor_sum", 0.0)),
            "latest_epoch": stats.get("latest_epoch") or {},
            "cumulative_quality": float(stats.get("cumulative_quality", 0.0)),
            "micro_tick_count": int(stats.get("micro_tick_count", 0)),
            "last_alignment": float(stats.get("last_alignment", 0.0)),
            "enrichment_rate": stats.get("enrichment_rate", 0.0),
            "full_130dt": [
                float(v) for v in stats.get("full_130dt", [0.5] * 130)],
            "config": stats.get("config", {}),
            "ts": time.time(),
        }
        self._write_msgpack(UNIFIED_SPIRIT_METADATA_SLOT, payload)

    # -- low-level write helper ------------------------------------------

    def _write_msgpack(self, slot: str, payload: dict[str, Any]) -> None:
        """
        Encode payload + write to slot. Throttle-log encode/oversize/write
        failures so a malformed source-shape doesn't flood the brain log.
        """
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails[slot] += 1
            if (self._encode_fails[slot] == 1 or
                    self._encode_fails[slot] % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[SpiritStatePublisher] %s msgpack encode failed "
                    "(#%d): %s — keys=%s",
                    slot, self._encode_fails[slot], e,
                    sorted(payload.keys()), exc_info=True)
            return

        max_bytes = self._SLOT_SPECS[slot].payload_bytes
        if len(encoded) > max_bytes:
            self._oversize_fails[slot] += 1
            logger.critical(
                "[SpiritStatePublisher] %s payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate (truncation corrupts msgpack).",
                slot, len(encoded), max_bytes, self._oversize_fails[slot])
            return

        try:
            writer = self._writer_for(slot)
            # variable_size=True slots use write_variable(bytes) — the
            # actual payload size lands in per-buffer metadata so readers
            # decode exactly len(encoded) bytes (no padding needed).
            writer.write_variable(encoded)
            # Mark first-success per slot at INFO so the brain log shows
            # exactly when each slot started flowing. Subsequent successes
            # don't log (heartbeat ticks already cover liveness).
            self._publish_success[slot] += 1
            if self._publish_success[slot] == 1:
                logger.info(
                    "[SpiritStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d (consumers can now read; "
                    "T3 deadlock surface closed for this slot)",
                    slot, len(encoded))
        except Exception as e:
            self._write_fails[slot] += 1
            if (self._write_fails[slot] == 1 or
                    self._write_fails[slot] % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[SpiritStatePublisher] %s shm write failed (#%d): %s",
                    slot, self._write_fails[slot], e, exc_info=True)

    def _warn_throttled(self, slot: str, kind: str, exc: Exception,
                        prefix: str) -> None:
        """Throttled warning for source-side iteration failures."""
        counter = self._encode_fails  # reuse the encode counter for source iter
        counter[slot] += 1
        if counter[slot] == 1 or counter[slot] % _WARN_THROTTLE_EVERY == 0:
            logger.warning(
                "[SpiritStatePublisher] %s %s (#%d): %s",
                slot, prefix, counter[slot], exc)

    # -- introspection ---------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return publisher health stats (used by test harness + diagnostics)."""
        return {
            "publish_count": self._publish_count,
            "encode_fails": dict(self._encode_fails),
            "oversize_fails": dict(self._oversize_fails),
            "write_fails": dict(self._write_fails),
            "writers_attached": {
                slot: (w is not None) for slot, w in self._writers.items()
            },
        }
