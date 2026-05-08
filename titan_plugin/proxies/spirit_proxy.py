"""
Spirit Module Proxy — SHM-direct read bridge to the supervised Spirit process.

Phase C Session 1 of rFP_phase_c_async_shm_consumer_migration §4.C.1
(LANDED 2026-05-07). Migrates all 5 sync `bus.request` state-lookup
methods to non-blocking SHM reads via `StateRegistryReader`, closing the
deadlock surface uncovered on T3 2026-05-07 (py-spy live-confirmed all
3 outer-sensor sidecars stuck inside `sock.sendall` because the bus
broker was back-pressured and `bus.request`'s `timeout` only governs
the wait-for-reply, never the send).

This is the canonical Phase C consumer pattern per Preamble G18+G19:
  - producers publish to SHM at their own cadence (spirit_worker via
    `titan_plugin.logic.spirit_state_publisher.SpiritStatePublisher`,
    1 Hz cadence + content-hash gating)
  - consumers (this proxy + sidecars + dashboards + gather loops) read
    SHM via `StateRegistryReader` — bounded latency, lock-free, decoupled
    from producer process state
  - bus carries events/commands/notifications, never state

All 5 spirit_state slot specs come from
`titan_plugin.logic.spirit_state_specs` (single source of truth shared
with the producer per G21 single-writer/multi-reader contract).

Methods that DON'T migrate this session (Session 2 scope per rFP §8):
  - get_filter_down_status, get_meditation_health, get_coordinator,
    get_nervous_system — these have no SHM slot yet; their producers
    ship in Session 2 alongside the matching reader migrations.
  - get_v4_state — composite over get_trinity (transparently
    SHM-direct now via the migrated get_trinity).

Cold-boot / staleness handling:
  - Each `read_variable()` call returns ``None`` on
    uninitialized/missing/torn slots — proxy falls back to a
    schema-compatible default dict so callers get a stable shape (same
    as the legacy bus.request fallback).
  - First-fallback is INFO-logged once per slot per process lifetime so
    operators see the cold-boot transition; subsequent fallbacks throttle.
  - Fallback default ≠ silent error — `_fallback_count` per slot is
    surfaced via `get_diagnostics()` for health checks.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np

from titan_plugin.bus import DivineBus
from titan_plugin.core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from titan_plugin.guardian import Guardian
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


# ── Existing-slot specs (already published before this rFP) ────────────────
# Imported from canonical state_registry singletons used elsewhere
# (api/shm_reader_bank.py uses the same specs — single source of truth).
from titan_plugin.core.state_registry import (
    INNER_BODY_5D,
    INNER_MIND_15D,
    INNER_SPIRIT_45D,
    SPHERE_CLOCKS_STATE,
    TOPOLOGY_30D,
    HORMONAL_STATE,
)
# Note: the bare 132D unified-spirit tensor lives at unified_spirit_132d.bin
# (Rust-owned by titan-unified-spirit-rs) and does NOT yet have a Python
# RegistrySpec — the new unified_spirit_metadata.bin slot (Session 1)
# carries the full UnifiedSpirit.get_stats() including the full_130dt array,
# so the proxy reads metadata not the bare tensor. A Python RegistrySpec
# for unified_spirit_132d.bin can land in Session 4 §4.B.6/7 when other
# slot Python specs are added.


# Sphere clock canonical names (matches SPEC §7.1 sphere_clocks.bin layout
# 6 clocks × 7 fields = 168 bytes; same labels as
# api/shm_reader_bank.py:SPHERE_CLOCK_NAMES).
SPHERE_CLOCK_NAMES = (
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
)
SPHERE_CLOCK_FIELDS = (
    "radius", "scalar_position", "phase", "contraction_velocity",
    "pulse_count", "consecutive_balanced", "last_pulse_age_s",
)

# 11-hormone canonical NS_PROGRAMS order (SPEC §7.1 hormonal_state.bin)
HORMONE_NAMES = (
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",
)
HORMONE_FIELDS = ("level", "threshold", "refractory", "peak_level")


# Throttle for cold-boot fallback INFO logs (one per slot per process)
_FALLBACK_LOG_THRESHOLD_FIRST = True


class SpiritProxy:
    """
    Phase C SHM-direct proxy for the Spirit module.

    Constructor signature preserved for backward compatibility (callers
    still pass `bus` + `guardian`) — the bus reference is retained for
    the not-yet-migrated methods (Session 2 scope) and for control-plane
    bus operations. State-lookup methods (5 of 9) are SHM-direct and
    NEVER touch the bus.

    Each method:
      1. Reads from the relevant SHM slot(s) via StateRegistryReader.
      2. msgpack-decodes the payload (variable-size slots) or numpy-
         decodes structured layouts (fixed-size slots).
      3. Returns a dict whose schema matches the legacy bus.request
         response shape — callers see zero behavior change.
      4. On cold-boot/missing-slot, returns a schema-compatible default
         (same fallback shape as the legacy bus.request else-branch).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        # Reply queue retained for the 4 methods that haven't migrated
        # yet (filter_down_status, meditation_health, coordinator,
        # nervous_system — Session 2 scope).
        self._reply_queue = bus.subscribe("spirit_proxy", reply_only=True)

        # SHM root + lazy readers for the 10 slots this proxy reads.
        # Readers attach lazily on first read (matches ShmReaderBank
        # pattern in api/). Reader construction is cheap; mmap attach is
        # deferred to first hot-path call.
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)

        # SpiritStatePublisher slots (Session 1 §4.B.1 — produced by
        # spirit_worker)
        self._r_hormone_fires = StateRegistryReader(
            HORMONE_FIRES_SPEC, self._shm_root)
        self._r_impulse_engine = StateRegistryReader(
            IMPULSE_ENGINE_STATE_SPEC, self._shm_root)
        self._r_consciousness = StateRegistryReader(
            CONSCIOUSNESS_STATE_SPEC, self._shm_root)
        self._r_resonance = StateRegistryReader(
            RESONANCE_STATE_SPEC, self._shm_root)
        self._r_unified_spirit_meta = StateRegistryReader(
            UNIFIED_SPIRIT_METADATA_SPEC, self._shm_root)

        # Pre-existing slots (other producers; we just read)
        self._r_inner_body = StateRegistryReader(INNER_BODY_5D, self._shm_root)
        self._r_inner_mind = StateRegistryReader(INNER_MIND_15D, self._shm_root)
        self._r_inner_spirit = StateRegistryReader(INNER_SPIRIT_45D, self._shm_root)
        self._r_topology = StateRegistryReader(TOPOLOGY_30D, self._shm_root)
        self._r_sphere_clocks = StateRegistryReader(
            SPHERE_CLOCKS_STATE, self._shm_root)
        self._r_hormonal = StateRegistryReader(HORMONAL_STATE, self._shm_root)

        # Per-slot fallback counter (cold-boot / missing-slot diagnostics).
        self._fallback_counts: dict[str, int] = {}

        logger.info(
            "[SpiritProxy] initialized SHM-direct readers — titan_id=%s "
            "shm_root=%s (5 spirit_state slots + 7 pre-existing trinity "
            "slots — Phase C SHM-canonical per Preamble G18)",
            self._titan_id, self._shm_root)

    # ── Helpers ──────────────────────────────────────────────────────

    def _read_msgpack(self, reader: StateRegistryReader,
                      slot_name: str) -> Optional[dict]:
        """
        Read variable-size SHM slot, msgpack-decode, return dict.
        Returns None on cold-boot / missing / torn / decode-failure;
        caller substitutes schema-compatible default.

        First fallback per slot logs INFO so cold-boot transitions are
        visible; subsequent fallbacks throttle to DEBUG.
        """
        try:
            raw = reader.read_variable()
        except Exception as e:
            self._track_fallback(slot_name, f"read_raised:{type(e).__name__}")
            logger.warning(
                "[SpiritProxy] %s SHM read raised: %s",
                slot_name, e, exc_info=True)
            return None
        if raw is None:
            self._track_fallback(slot_name, "shm_unavailable")
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback(slot_name, f"decode_raised:{type(e).__name__}")
            logger.warning(
                "[SpiritProxy] %s msgpack decode failed (raw_bytes=%d): %s",
                slot_name, len(raw), e, exc_info=True)
            return None
        if not isinstance(decoded, dict):
            self._track_fallback(slot_name, f"decode_wrong_type:{type(decoded).__name__}")
            return None
        return decoded

    def _read_floats(self, reader: StateRegistryReader,
                     slot_name: str) -> Optional[np.ndarray]:
        """
        Read fixed-size SHM slot, return ndarray or None on fallback.
        Used for inner_body_5d / inner_mind_15d / etc. Numeric tensors.
        """
        try:
            arr = reader.read()
        except Exception as e:
            self._track_fallback(slot_name, f"read_raised:{type(e).__name__}")
            logger.warning(
                "[SpiritProxy] %s SHM read raised: %s",
                slot_name, e, exc_info=True)
            return None
        if arr is None:
            self._track_fallback(slot_name, "shm_unavailable")
            return None
        return arr

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        """Bump per-slot fallback counter; log INFO on first occurrence."""
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[SpiritProxy] FIRST FALLBACK for slot=%s reason=%s — "
                "consumer will use schema-compatible default; subsequent "
                "fallbacks throttled (likely cold-boot before producer "
                "first publish; should clear within ~1s)",
                slot_name, reason)

    # ── Migrated methods (5 of 9 — Session 1 §4.C.1) ──────────────────

    def get_spirit_tensor(self) -> list:
        """Get the 5DT Spirit state tensor (3DT+2). SHM-direct via
        consciousness_state.bin (publisher computes via the canonical
        _collect_spirit_tensor formula and includes it in payload)."""
        decoded = self._read_msgpack(
            self._r_consciousness, CONSCIOUSNESS_STATE_SLOT)
        if decoded is None:
            return [0.5] * 5
        spirit_5dt = decoded.get("spirit_5dt")
        if not isinstance(spirit_5dt, list) or len(spirit_5dt) != 5:
            self._track_fallback(
                CONSCIOUSNESS_STATE_SLOT, "spirit_5dt_missing_or_bad_shape")
            return [0.5] * 5
        return [float(v) for v in spirit_5dt]

    def get_trinity(self) -> dict:
        """
        Get unified Trinity state: Body, Mind, Spirit tensors + V4 data.

        Composes the trinity dict from 9 SHM slots:
          - inner_body_5d.bin → body_values
          - inner_mind_15d.bin → mind_values (first 5 — keeping legacy schema)
          - consciousness_state.bin → spirit_5dt + body_center_dist +
            mind_center_dist + consciousness epoch + middle_path_loss inputs
          - hormonal_state.bin → hormone_levels
          - hormone_fires.bin → hormone_fires
          - impulse_engine_state.bin → impulse_engine
          - sphere_clocks.bin → sphere_clock
          - resonance_state.bin → resonance
          - unified_spirit_metadata.bin → unified_spirit (full get_stats() shape)

        This is the deadlock-causing method on T3 (sidecars stuck on
        sock.sendall here) — SHM-direct read closes the deadlock by
        construction (no socket call, no broker round-trip, bounded
        latency).
        """
        # Consciousness payload carries the most: spirit_5dt + body/mind
        # values + center_dists + latest_epoch.
        consciousness_pl = self._read_msgpack(
            self._r_consciousness, CONSCIOUSNESS_STATE_SLOT)
        if consciousness_pl is None:
            consciousness_pl = {}

        spirit_tensor = consciousness_pl.get("spirit_5dt") or [0.5] * 5
        body_values = consciousness_pl.get("body_values") or [0.5] * 5
        mind_values = consciousness_pl.get("mind_values") or [0.5] * 5

        response: dict[str, Any] = {
            "spirit_tensor": [float(v) for v in spirit_tensor],
            "body_values": [float(v) for v in body_values],
            "mind_values": [float(v) for v in mind_values],
            "body_center_dist": float(
                consciousness_pl.get("body_center_dist", 0.0)),
            "mind_center_dist": float(
                consciousness_pl.get("mind_center_dist", 0.0)),
        }

        latest_epoch = consciousness_pl.get("latest_epoch")
        if isinstance(latest_epoch, dict) and latest_epoch:
            response["consciousness"] = latest_epoch

        # middle_path_loss is computed from the canonical body+mind+spirit
        # tensors (single source of truth — same formula spirit_loop's
        # build_trinity_snapshot uses).
        try:
            from titan_plugin.logic.middle_path import middle_path_loss
            response["middle_path_loss"] = round(
                middle_path_loss(body_values, mind_values, spirit_tensor), 4)
        except Exception as e:
            logger.warning(
                "[SpiritProxy] middle_path_loss compute failed: %s",
                e, exc_info=True)

        # Hormonal state — read pre-existing fixed-layout slot
        hormonal_arr = self._read_floats(self._r_hormonal, "hormonal_state")
        if hormonal_arr is not None:
            # 11 hormones × 4 fields = 44 floats packed; level is field [0]
            try:
                vals = hormonal_arr.view(np.float32).reshape(
                    len(HORMONE_NAMES), len(HORMONE_FIELDS))
                response["hormone_levels"] = {
                    HORMONE_NAMES[i]: float(vals[i][0])  # level field
                    for i in range(len(HORMONE_NAMES))
                }
            except Exception as e:
                logger.warning(
                    "[SpiritProxy] hormonal_state decode failed: %s",
                    e, exc_info=True)

        # Hormone fires — variable-size msgpack slot (Session 1 producer)
        fires_pl = self._read_msgpack(self._r_hormone_fires, HORMONE_FIRES_SLOT)
        if fires_pl is not None:
            fires = fires_pl.get("fires")
            if isinstance(fires, dict):
                response["hormone_fires"] = {
                    str(k): int(v) for k, v in fires.items()}

        # Impulse engine — variable-size msgpack slot (Session 1 producer)
        impulse_pl = self._read_msgpack(
            self._r_impulse_engine, IMPULSE_ENGINE_STATE_SLOT)
        if impulse_pl is not None:
            response["impulse_engine"] = impulse_pl.get("engine") or {}
            # Per-hormone impulse breakdown (publisher includes this for
            # consumers that need pressure-by-hormone state)
            if "hormones" in impulse_pl:
                response["impulse_engine_hormones"] = impulse_pl["hormones"]

        # Sphere clocks — read fixed-layout slot + decode to per-clock dict
        sc_dict = self._read_sphere_clocks_dict()
        if sc_dict is not None:
            response["sphere_clock"] = sc_dict

        # Resonance — variable-size msgpack slot (Session 1 producer)
        resonance_pl = self._read_msgpack(
            self._r_resonance, RESONANCE_STATE_SLOT)
        if resonance_pl is not None:
            # Strip our `ts` field — caller doesn't need it (it's already
            # implicit in age_seconds via read_meta).
            response["resonance"] = {
                k: v for k, v in resonance_pl.items() if k != "ts"}

        # Unified spirit metadata — variable-size msgpack slot (Session 1)
        us_pl = self._read_msgpack(
            self._r_unified_spirit_meta, UNIFIED_SPIRIT_METADATA_SLOT)
        if us_pl is not None:
            response["unified_spirit"] = {
                k: v for k, v in us_pl.items() if k != "ts"}

        return response

    def get_sphere_clocks(self) -> dict:
        """Get V4 SphereClockEngine state. SHM-direct via sphere_clocks.bin
        (already published by titan-trinity-rs / spirit_worker)."""
        sc = self._read_sphere_clocks_dict()
        if sc is None:
            return {"error": "SphereClocks not available"}
        return sc

    def _read_sphere_clocks_dict(self) -> Optional[dict]:
        """Decode 6 × 7 float32 sphere_clocks.bin into per-clock dict."""
        arr = self._read_floats(self._r_sphere_clocks, "sphere_clocks")
        if arr is None:
            return None
        try:
            vals = arr.view(np.float32).reshape(
                len(SPHERE_CLOCK_NAMES), len(SPHERE_CLOCK_FIELDS))
            return {
                SPHERE_CLOCK_NAMES[i]: {
                    field: float(vals[i][j])
                    for j, field in enumerate(SPHERE_CLOCK_FIELDS)
                }
                for i in range(len(SPHERE_CLOCK_NAMES))
            }
        except Exception as e:
            logger.warning(
                "[SpiritProxy] sphere_clocks decode failed: %s",
                e, exc_info=True)
            return None

    def get_resonance(self) -> dict:
        """Get V4 ResonanceDetector state. SHM-direct via
        resonance_state.bin (Session 1 §4.B.1 producer)."""
        decoded = self._read_msgpack(self._r_resonance, RESONANCE_STATE_SLOT)
        if decoded is None:
            return {"error": "Resonance not available"}
        # Strip ts (caller doesn't need it; freshness handled at meta layer)
        return {k: v for k, v in decoded.items() if k != "ts"}

    def get_unified_spirit(self) -> dict:
        """Get V4 UnifiedSpirit state: 130D tensor + velocity + stale +
        focus_multiplier + GreatEpoch state. SHM-direct via
        unified_spirit_metadata.bin (Session 1 §4.B.1 producer)."""
        decoded = self._read_msgpack(
            self._r_unified_spirit_meta, UNIFIED_SPIRIT_METADATA_SLOT)
        if decoded is None:
            return {"error": "UnifiedSpirit not available"}
        return {k: v for k, v in decoded.items() if k != "ts"}

    # ── Composite (transparently SHM-direct via migrated get_trinity) ─

    # ── Phase C Session 4 (rFP §4.C.1 expansion) — 4 spirit_supplemental
    # methods migrated from sync bus.request to SHM-direct read of
    # spirit_supplemental_state.bin (Session 4 producer in spirit_loop).
    # These were the 4 methods Session 1 explicitly retained sync; Maker
    # greenlit closing them in Session 4 for full G19 closure of
    # spirit_proxy.

    def _read_supplemental_section(self, section: str) -> Optional[dict]:
        """Read one section of spirit_supplemental_state.bin."""
        # Lazy attach the supplemental reader (defer to first call so
        # __init__ doesn't have to import session4_state_specs unconditionally
        # for byte-identical compat with non-session4 callers).
        if not hasattr(self, "_r_spirit_supplemental"):
            from titan_plugin.logic.session4_state_specs import (
                SPIRIT_SUPPLEMENTAL_STATE_SPEC)
            self._r_spirit_supplemental = StateRegistryReader(
                SPIRIT_SUPPLEMENTAL_STATE_SPEC, self._shm_root)
        decoded = self._read_msgpack(
            self._r_spirit_supplemental, "spirit_supplemental_state")
        if decoded is None:
            return None
        sec = decoded.get(section)
        return sec if isinstance(sec, dict) else None

    def get_v4_state(self) -> dict:
        """Get complete V4 Time Awareness state in a single call. Composite
        over get_trinity (which is SHM-direct as of Session 1)."""
        trinity = self.get_trinity()
        return {
            "sphere_clock": trinity.get("sphere_clock", {}),
            "resonance": trinity.get("resonance", {}),
            "unified_spirit": trinity.get("unified_spirit", {}),
            "impulse_engine": trinity.get("impulse_engine", {}),
            "filter_down": trinity.get("filter_down", {}),
            "intuition": trinity.get("intuition", {}),
            "consciousness": trinity.get("consciousness", {}),
            "middle_path_loss": trinity.get("middle_path_loss"),
        }

    # ── Phase C Session 4 (rFP §4.C.1 expansion) — MIGRATED ─────────
    # The 4 methods below migrated from sync bus.request to SHM-direct
    # read of spirit_supplemental_state.bin (producer:
    # SpiritSupplementalStatePublisher in spirit_loop, alongside the
    # 5 Session 1 + 2 Session 3 publishers in the same publish tick).
    # Closes spirit_proxy fully — zero sync bus.request remaining.

    def get_filter_down_status(self) -> dict:
        """V5 filter-down state. SHM-direct via spirit_supplemental_state.bin."""
        section = self._read_supplemental_section("filter_down_status")
        if section is None:
            return {"error": "FilterDown status not available"}
        return section

    def get_meditation_health(self) -> dict:
        """Meditation watchdog state. SHM-direct via spirit_supplemental_state.bin."""
        section = self._read_supplemental_section("meditation_health")
        if section is None:
            return {"error": "Meditation health not available"}
        return section

    def get_coordinator(self) -> dict:
        """T3 InnerTrinityCoordinator state. SHM-direct via spirit_supplemental_state.bin."""
        section = self._read_supplemental_section("coordinator")
        if section is None:
            return {"error": "Coordinator not available"}
        return section

    def get_nervous_system(self) -> dict:
        """V5 Neural NervousSystem state. SHM-direct via spirit_supplemental_state.bin."""
        section = self._read_supplemental_section("nervous_system")
        if section is None:
            return {"error": "NervousSystem not available"}
        return section

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        """Per-slot fallback counts + reader attach state. Used by
        health-check endpoints + arch_map verify."""
        return {
            "titan_id": self._titan_id,
            "shm_root": str(self._shm_root),
            "fallback_counts": dict(self._fallback_counts),
            "session1_migrated_methods": [
                "get_spirit_tensor",
                "get_trinity",
                "get_sphere_clocks",
                "get_resonance",
                "get_unified_spirit",
                "get_v4_state",  # composite over get_trinity
            ],
            "session2_pending_methods": [
                "get_filter_down_status",
                "get_meditation_health",
                "get_coordinator",
                "get_nervous_system",
            ],
        }
