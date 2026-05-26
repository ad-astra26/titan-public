"""
LifeForce Proxy — bridge to the supervised life_force_worker subprocess.

Phase C v1.8.3 (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy §4.G.
Replaces the in-process `LifeForceEngine` reference at
`cognitive_worker.py:1558` chunk 8M.6 (Track 1 drift retired).

Classification per SPEC Preamble G18-G22:

  • get_chi_total / get_metabolic_drain /        → SHM read of
    get_chi_state (full payload) / get_state /     life_force_state.bin
    get_developmental_phase / get_circulation /    (G18 sub-µs — hot
    is_dreaming                                    path for cognitive_worker
                                                   MSL static_context,
                                                   reasoning body_state,
                                                   hormonal_pressure inputs,
                                                   ground_up enricher
                                                   chi_overlay, NN
                                                   modulation cap)

  • get_stats / get_chi_history /                → bus.request_async
    get_contemplation_status                       work-RPC ≤5s per
                                                   G19 strict. Allow-listed
                                                   under `life_force_proxy:`
                                                   in `phase_c_rpc_exemptions.yaml`.

The proxy mirrors `MetabolismShmReader` + `MetabolismProxy` ergonomics
exactly (matches §4.J D-SPEC-51 pattern). cognitive_worker instantiates
`LifeForceShmReader` directly at the 5 hot-path read sites; plugin
wires `LifeForceProxy` for full-surface access.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import msgpack

from ..bus import DivineBus
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian import Guardian
from ..logic.life_force_state_specs import (
    LIFE_FORCE_STATE_SLOT,
    LIFE_FORCE_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_COLD_DEFAULT_TOTAL = 0.5
_COLD_DEFAULT_DRAIN = 0.0
_COLD_DEFAULT_STATE = "BOOTSTRAP"
_COLD_DEFAULT_PHASE = "BIRTH"


# ── LifeForceShmReader (no bus dependency, sub-µs hot-path) ───────────


class LifeForceShmReader:
    """Sub-µs SHM-direct reader for life_force_state.bin hot-path fields.

    Used by cognitive_worker at 5 sites previously reading
    `life_force_engine._latest_chi.get("total")` and `_metabolic_drain`
    directly (MSL static_context build, reasoning body_state,
    hormonal_pressure inputs, ground_up enricher chi_overlay, NN
    modulation cap). Also embedded in LifeForceProxy for the full
    surface.

    No bus dependency — only a per-Titan SHM root path is needed.

    Cold-boot tolerant: returns sensible defaults if the slot hasn't
    been written yet (worker not yet up, or first run after restart).
    """

    def __init__(self, titan_id: Optional[str] = None,
                 shm_root: Optional[Path] = None):
        self._titan_id = titan_id or resolve_titan_id()
        self._shm_root: Path = shm_root or ensure_shm_root(self._titan_id)
        self._r_state = StateRegistryReader(
            LIFE_FORCE_STATE_SPEC, self._shm_root)
        self._fallback_count = 0
        logger.debug(
            "[LifeForceShmReader] initialized — titan_id=%s shm_root=%s "
            "slot=%s",
            self._titan_id, self._shm_root, LIFE_FORCE_STATE_SLOT)

    def _read_state(self) -> Optional[dict]:
        try:
            raw = self._r_state.read_variable()
        except Exception as e:
            self._fallback_count += 1
            if self._fallback_count == 1:
                logger.info(
                    "[LifeForceShmReader] FIRST FALLBACK: read raised "
                    "%s — using cold defaults until producer first publish",
                    type(e).__name__)
            return None
        if raw is None:
            self._fallback_count += 1
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._fallback_count += 1
            logger.warning(
                "[LifeForceShmReader] msgpack decode raised: %s", e)
            return None
        return decoded if isinstance(decoded, dict) else None

    # ── Hot-path reads (sub-µs; the 5 cognitive_worker sites) ─────────

    def get_chi_total(self) -> float:
        """Composite chi value ∈ [0,1]. Default 0.5 on cold-boot."""
        decoded = self._read_state()
        if decoded is None:
            return _COLD_DEFAULT_TOTAL
        return float(decoded.get("total", _COLD_DEFAULT_TOTAL) or _COLD_DEFAULT_TOTAL)

    def get_metabolic_drain(self) -> float:
        """Adenosine-like drain accumulator ∈ [0,0.8]. Default 0.0 on cold-boot."""
        decoded = self._read_state()
        if decoded is None:
            return _COLD_DEFAULT_DRAIN
        return float(decoded.get("metabolic_drain", _COLD_DEFAULT_DRAIN) or _COLD_DEFAULT_DRAIN)

    # ── Diagnostic reads (full payload) ───────────────────────────────

    def get_chi_state(self) -> dict[str, Any]:
        """Full life_force_state.bin payload (msgpack-decoded). Empty dict
        on cold-boot."""
        decoded = self._read_state()
        return decoded if isinstance(decoded, dict) else {}

    def get_state(self) -> str:
        """Behavioral state classifier
        (FLOURISHING/HEALTHY/CONSERVING/SURVIVAL/STARVATION/BOOTSTRAP)."""
        decoded = self._read_state()
        if decoded is None:
            return _COLD_DEFAULT_STATE
        return str(decoded.get("state", _COLD_DEFAULT_STATE) or _COLD_DEFAULT_STATE)

    def get_developmental_phase(self) -> str:
        """BIRTH / YOUTH / MATURE."""
        decoded = self._read_state()
        if decoded is None:
            return _COLD_DEFAULT_PHASE
        return str(decoded.get("developmental_phase", _COLD_DEFAULT_PHASE) or _COLD_DEFAULT_PHASE)

    def get_circulation(self) -> float:
        """Chi flow rate ∈ [0,1+] (d_spirit + d_mind + d_body)."""
        decoded = self._read_state()
        if decoded is None:
            return 0.0
        return float(decoded.get("circulation", 0.0) or 0.0)

    def is_dreaming(self) -> bool:
        """Cached from DREAM_STATE_CHANGED via life_force_worker."""
        decoded = self._read_state()
        if decoded is None:
            return False
        return bool(decoded.get("is_dreaming", False))


# ── LifeForceProxy (full surface; bus + SHM) ──────────────────────────


class LifeForceProxy:
    """Drop-in proxy for the life_force_worker subprocess.

    Replaces the in-process `LifeForceEngine` reference per rFP §4.G +
    D-SPEC-57. Exposes:

      - Hot reads (SHM, sub-µs): get_chi_total, get_metabolic_drain,
        get_chi_state, get_state, get_developmental_phase, get_circulation,
        is_dreaming (delegated to embedded LifeForceShmReader).
      - Diagnostic queries (work-RPC ≤5s): get_stats, get_chi_history,
        get_contemplation_status.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe(
            "life_force_proxy", reply_only=True)
        self._started = False

        # Embedded SHM reader (hot-path).
        self._reader = LifeForceShmReader()

        logger.info(
            "[LifeForceProxy] initialized — SHM-direct reads for chi_total "
            "+ metabolic_drain + state; bus work-RPC for get_stats + "
            "chi_history + contemplation_status (rFP §4.G + D-SPEC-57)")

    # ── Lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "life_force", id(self),
            proxy_label="LifeForceProxy",
        )
        if ready:
            self._started = True

    # ── SHM-direct hot reads (delegate to reader) ────────────────────

    def get_chi_total(self) -> float:
        return self._reader.get_chi_total()

    def get_metabolic_drain(self) -> float:
        return self._reader.get_metabolic_drain()

    def get_chi_state(self) -> dict[str, Any]:
        return self._reader.get_chi_state()

    def get_state(self) -> str:
        return self._reader.get_state()

    def get_developmental_phase(self) -> str:
        return self._reader.get_developmental_phase()

    def get_circulation(self) -> float:
        return self._reader.get_circulation()

    def is_dreaming(self) -> bool:
        return self._reader.is_dreaming()

    # ── Work-RPC primitives (G19 ≤5s) ────────────────────────────────

    async def _work_rpc_async(self, action: str, extra: dict | None = None,
                              timeout: float = 5.0) -> dict:
        self._ensure_started()
        payload = {"action": action}
        if extra:
            payload.update(extra)
        try:
            reply = await self._bus.request_async(
                "life_force_proxy", "life_force", payload,
                timeout=timeout, reply_queue=self._reply_queue,
            )
            return reply.get("payload", {}) if reply else {}
        except Exception as e:
            logger.warning(
                "[LifeForceProxy] %s async work-RPC raised "
                "(timeout=%.1fs): %s", action, timeout, e)
            return {}

    def _work_rpc_sync(self, action: str, extra: dict | None = None,
                       timeout: float = 5.0) -> dict:
        self._ensure_started()
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            try:
                return asyncio.run(
                    self._work_rpc_async(action, extra, timeout=timeout))
            except Exception as e:
                logger.warning(
                    "[LifeForceProxy] %s asyncio.run failed: %s — "
                    "falling back to bounded sync bus.request",
                    action, e)

        payload = {"action": action}
        if extra:
            payload.update(extra)
        reply = self._bus.request(
            "life_force_proxy", "life_force", payload,
            timeout=timeout, reply_queue=self._reply_queue,
        )
        return reply.get("payload", {}) if reply else {}

    # ── Diagnostic queries (work-RPC) ────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """LifeForceEngine.get_stats() proxy — total_evaluations, current_state,
        last_chi, contemplation_phase, conviction_counter, chi_trend,
        metabolic_drain, total_neuromod_cost, total_somatic_cost, is_dreaming."""
        result = await self._work_rpc_async("get_stats")
        if isinstance(result, dict):
            inner = result.get("result")
            if isinstance(inner, dict):
                return inner
        return {}

    async def get_chi_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Recent chi_history entries — list of {ts, total, spirit, mind, body, circulation}."""
        result = await self._work_rpc_async(
            "get_chi_history", {"limit": int(limit)})
        if isinstance(result, dict):
            inner = result.get("result")
            if isinstance(inner, list):
                return inner
        return []

    async def get_contemplation_status(self) -> dict[str, Any]:
        """Contemplation/GRAND-CYCLE state — {active, phase, phase_name,
        conviction, conviction_threshold, mature_enough}."""
        result = await self._work_rpc_async("get_contemplation_status")
        if isinstance(result, dict):
            inner = result.get("result")
            if isinstance(inner, dict):
                return inner
        return {}
