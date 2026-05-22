"""
Metabolism Proxy — bridge to the supervised metabolism_worker subprocess.

Phase C v1.7.2 (D-SPEC-51) per rFP_titan_hcl_l2_separation_strategy §4.J.
Replaces the in-process `MetabolismController` reference at plugin.py:1612
(`self._proxies["metabolism"] = metabolism`).

Classification per SPEC Preamble G18-G22 + Maker-locked 2026-05-14
design calls:

  • get_metabolic_tier / get_gates_enforced /         → SHM read of
    gates_enforced (property) / get_tier_info /          metabolism_state.bin
    get_last_gate_decision_reason / get_balance_pct     (G18 sub-ms — hot path
    / can_use_feature (LOCAL via TIER_FEATURES)         for Soul NFT mint
                                                        gate + dashboard
                                                        /status + kernel
                                                        metabolism.* proxy)

  • All other surface (evaluate_gate, can_afford,     → bus.request_async
    get_current_state, get_metabolic_health,            work-RPC ≤5s per
    get_learning_velocity, get_directive_alignment,     G19 strict (Maker-
    get_social_density, get_service_gate,               locked 2026-05-14).
    get_gate_decision_summary, get_emergency_duration,  Allow-listed under
    get_gate_ring, get_tier_history)                    `metabolism_proxy:`
                                                        in `phase_c_rpc_exemptions.yaml`.

Soul migration (G16/G18 path per SPEC §9.B v1.7.2 + Maker-locked
"Replace with SHM read from soul"): a standalone `MetabolismShmReader`
class exposes the sub-ms hot-path reads without requiring a bus client
or a guardian reference. Soul instantiates one of these directly so its
`allow_mint` NFT-gate stays in-kernel without any cross-process roundtrip.
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
from ..logic.metabolism_state_specs import (
    METABOLISM_STATE_SLOT,
    METABOLISM_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_COLD_DEFAULT_TIER = "HEALTHY"


# ── MetabolismShmReader (Soul-friendly, no bus dependency) ────────────


class MetabolismShmReader:
    """Sub-ms SHM-direct reader for the hot-path metabolism state fields.

    Used by kernel-level Soul (replaces the legacy
    `self.soul.set_metabolism(metabolism)` reverse-injection at
    plugin.py:1670 per Maker-locked 2026-05-14 "Replace with SHM read
    from soul"). Also embedded in MetabolismProxy for consistency.

    No bus dependency — only a per-Titan SHM root path is needed.

    Cold-boot tolerant: returns sensible defaults if the slot hasn't
    been written yet (worker not yet up, or first run after restart).
    """

    def __init__(self, titan_id: Optional[str] = None,
                 shm_root: Optional[Path] = None):
        self._titan_id = titan_id or resolve_titan_id()
        self._shm_root: Path = shm_root or ensure_shm_root(self._titan_id)
        self._r_state = StateRegistryReader(
            METABOLISM_STATE_SPEC, self._shm_root)
        self._fallback_count = 0
        logger.debug(
            "[MetabolismShmReader] initialized — titan_id=%s shm_root=%s "
            "slot=%s",
            self._titan_id, self._shm_root, METABOLISM_STATE_SLOT)

    def _read_state(self) -> Optional[dict]:
        try:
            raw = self._r_state.read_variable()
        except Exception as e:
            self._fallback_count += 1
            if self._fallback_count == 1:
                logger.info(
                    "[MetabolismShmReader] FIRST FALLBACK: read raised "
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
                "[MetabolismShmReader] msgpack decode raised: %s", e)
            return None
        return decoded if isinstance(decoded, dict) else None

    # Public surface (hot-path; sub-ms).

    def get_metabolic_tier(self) -> str:
        decoded = self._read_state()
        if decoded is None:
            return _COLD_DEFAULT_TIER
        tier = decoded.get("tier")
        return str(tier) if isinstance(tier, str) else _COLD_DEFAULT_TIER

    def get_gates_enforced(self) -> bool:
        decoded = self._read_state()
        if decoded is None:
            return False
        return bool(decoded.get("gates_enforced", False))

    @property
    def gates_enforced(self) -> bool:
        return self.get_gates_enforced()

    def get_tier_info(self) -> dict:
        decoded = self._read_state()
        if decoded is None:
            return {}
        info = decoded.get("tier_info")
        return dict(info) if isinstance(info, dict) else {}

    def get_last_gate_decision_reason(self) -> str:
        decoded = self._read_state()
        if decoded is None:
            return ""
        return str(decoded.get("last_gate_decision_reason", "") or "")

    def get_balance_pct(self) -> float:
        decoded = self._read_state()
        if decoded is None:
            return 1.0
        return float(decoded.get("balance_pct", 1.0) or 1.0)

    def can_use_feature(self, feature: str) -> bool:
        """Local feature gate based on TIER_FEATURES lookup table.

        Single source of truth for the lookup table is
        `titan_hcl.core.metabolism.TIER_FEATURES`; this method reads
        the current tier from SHM and dispatches against the same dict
        used by the controller. Soul's `allow_mint` uses this for sub-ms
        in-kernel gate decisions without bus roundtrip per Maker-locked
        2026-05-14 "Replace with SHM read from soul".

        For the authoritative ring-buffer-tracked decision (with bus
        broadcast for audit), call `MetabolismProxy.evaluate_gate`
        instead — that goes via work-RPC per G19.
        """
        try:
            from titan_hcl.core.metabolism import TIER_FEATURES
        except Exception:
            return False
        tier = self.get_metabolic_tier()
        tier_features = TIER_FEATURES.get(tier, {})
        return bool(tier_features.get(feature, False))

    # ── Duck-typed MetabolismController fallback (subprocess contexts) ─
    #
    # MemoInscribeHelper today calls:
    #   - sync: `metabolism.evaluate_gate(feature, caller)` → (bool, rate)
    #   - async: `metabolism.can_afford(cost)`              → bool
    #
    # Pre-extraction these went to MetabolismController directly. In
    # titan_HCL post-extraction they route through MetabolismProxy
    # (work-RPC, authoritative). In agency_worker subprocess they have
    # no bus client to host the proxy — so we expose duck-typed methods
    # on the reader that compute the same decision locally from SHM
    # tier + TIER_FEATURES + tier-derived rate.
    #
    # The decision is LOCAL (no ring-buffer write, no bus broadcast).
    # This matches the pre-extraction agency_worker behavior where
    # MemoInscribeHelper(metabolism=None) bypassed the gate entirely —
    # we're now AT LEAST applying the SHM tier check, which is a
    # functional improvement, not a regression.

    def evaluate_gate(self, feature: str, caller: str = "") -> tuple[bool, float]:
        """Local (SHM-cached) gate decision — duck-typed for MemoInscribe.

        Returns a (should_proceed, rate_multiplier) tuple matching the
        controller's surface contract. Per the controller semantics
        (titan_hcl/core/metabolism.py:283-292):
          - gates_enforced=False (observation): always (True, 1.0).
          - gates_enforced=True: (can_use_feature(feature), tier rate_factor).

        Reads tier from SHM + TIER_FEATURES + METABOLIC_TIERS locally —
        sub-ms, no bus. Authoritative gate decisions (with ring buffer
        write) go via MetabolismProxy.evaluate_gate which work-RPCs to
        the worker.
        """
        try:
            from titan_hcl.core.metabolism import METABOLIC_TIERS, TIER_FEATURES
        except Exception:
            return False, 0.0
        tier = self.get_metabolic_tier()
        if not self.gates_enforced:
            return True, 1.0
        allowed = bool(TIER_FEATURES.get(tier, {}).get(feature, False))
        rate = float(METABOLIC_TIERS.get(tier, {}).get("rate_factor", 0.0))
        return allowed, rate

    async def can_afford(self, cost: float) -> bool:
        """Local (SHM-cached) affordability check — duck-typed for MemoInscribe.

        Approximates the controller's `can_afford` (which reads live SOL
        balance via network) by checking the tier-derived state:

          - HEALTHY / THRIVING / CONSERVING → True (above starvation floor)
          - SURVIVAL / EMERGENCY / HIBERNATION → False (governance reserve
            guard — write actions refuse)

        This is a STRICTER check than the controller's exact
        (balance ≥ governance_reserve + cost) — at SURVIVAL+ tiers the
        balance is by definition near or below the governance reserve
        floor, so refusing all writes there is the correct posture.
        Async signature matches the controller for await-compatibility.
        """
        try:
            from titan_hcl.core.metabolism import METABOLIC_TIERS
        except Exception:
            return False
        tier = self.get_metabolic_tier()
        rate = float(METABOLIC_TIERS.get(tier, {}).get("rate_factor", 0.0))
        # rate_factor > 0 ⇒ tier permits writes (HEALTHY=1.0, CONSERVING=0.5,
        # THRIVING=1.0 per titan_params.toml [metabolism.tiers]).
        return rate > 0.0


# ── MetabolismProxy (full surface; bus + SHM) ─────────────────────────


class MetabolismProxy:
    """Drop-in proxy for the metabolism_worker subprocess.

    Replaces the in-process `MetabolismController` reference per rFP §4.J
    + D-SPEC-51. Exposes the full MetabolismController public API:

      - Hot reads (SHM, sub-ms): get_metabolic_tier, get_gates_enforced,
        gates_enforced property, get_tier_info,
        get_last_gate_decision_reason, can_use_feature (local lookup).
      - Authoritative gate decision (work-RPC ≤5s): evaluate_gate
        (sync + async). Ring buffer + bus event written by worker.
      - All other state queries (work-RPC ≤5s): get_current_state,
        can_afford, get_metabolic_health, get_learning_velocity,
        get_directive_alignment, get_social_density, get_service_gate,
        get_gate_decision_summary, get_emergency_duration.
      - Diagnostic reads (work-RPC ≤5s): get_gate_ring, get_tier_history.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        # Reply queue retained for every work-RPC method (~13 actions).
        self._reply_queue = bus.subscribe(
            "metabolism_proxy", reply_only=True)
        self._started = False

        # Embedded SHM reader (hot-path).
        self._reader = MetabolismShmReader()

        logger.info(
            "[MetabolismProxy] initialized — SHM-direct reads for tier + "
            "gates + tier_info; bus work-RPC for evaluate_gate + async "
            "state (rFP §4.J + D-SPEC-51)")

    # ── Lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "metabolism", id(self),
            proxy_label="MetabolismProxy",
        )
        if ready:
            self._started = True

    # ── SHM-direct hot reads (delegate to reader) ────────────────────

    def get_metabolic_tier(self) -> str:
        return self._reader.get_metabolic_tier()

    def get_gates_enforced(self) -> bool:
        return self._reader.get_gates_enforced()

    @property
    def gates_enforced(self) -> bool:
        return self._reader.gates_enforced

    def get_tier_info(self) -> dict:
        return self._reader.get_tier_info()

    def get_last_gate_decision_reason(self) -> str:
        return self._reader.get_last_gate_decision_reason()

    def can_use_feature(self, feature: str) -> bool:
        """Local feature lookup via SHM tier; no bus roundtrip."""
        return self._reader.can_use_feature(feature)

    # ── Work-RPC primitives (G19 ≤5s) ────────────────────────────────

    async def _work_rpc_async(self, action: str, extra: dict | None = None,
                              timeout: float = 5.0) -> dict:
        self._ensure_started()
        payload = {"action": action}
        if extra:
            payload.update(extra)
        try:
            reply = await self._bus.request_async(
                "metabolism_proxy", "metabolism", payload,
                timeout=timeout, reply_queue=self._reply_queue,
            )
            return reply.get("payload", {}) if reply else {}
        except Exception as e:
            logger.warning(
                "[MetabolismProxy] %s async work-RPC raised "
                "(timeout=%.1fs): %s", action, timeout, e)
            return {}

    def _work_rpc_sync(self, action: str, extra: dict | None = None,
                       timeout: float = 5.0) -> dict:
        """Sync wrapper. Mirrors SocialGraphProxy._work_rpc_sync pattern:
        in-loop callers fall back to legacy sync bus.request with bounded
        timeout (allow-listed in phase_c_rpc_exemptions.yaml)."""
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
                    "[MetabolismProxy] %s asyncio.run failed: %s — "
                    "falling back to bounded sync bus.request",
                    action, e)

        payload = {"action": action}
        if extra:
            payload.update(extra)
        reply = self._bus.request(
            "metabolism_proxy", "metabolism", payload,
            timeout=timeout, reply_queue=self._reply_queue,
        )
        return reply.get("payload", {}) if reply else {}

    # ── Gate decision (authoritative; work-RPC per G19-strict) ──────

    def evaluate_gate(self, feature: str, caller: str = "") -> tuple[bool, float]:
        """Authoritative gate decision. Ring buffer + bus event side-
        effects executed inside worker. Sync sibling for legacy callers
        like Soul.allow_mint (replaced by SHM read; this remains as
        belt-and-braces) + memo_inscribe.send_memo.
        """
        result = self._work_rpc_sync("evaluate_gate", {
            "feature": feature, "caller": caller,
        })
        if isinstance(result, dict) and "result" in result:
            res = result["result"]
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                return bool(res[0]), float(res[1])
        # Fallback: conservative refuse on RPC failure.
        return False, 0.0

    async def evaluate_gate_async(self, feature: str,
                                  caller: str = "") -> tuple[bool, float]:
        result = await self._work_rpc_async("evaluate_gate", {
            "feature": feature, "caller": caller,
        })
        if isinstance(result, dict) and "result" in result:
            res = result["result"]
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                return bool(res[0]), float(res[1])
        return False, 0.0

    # ── Async state queries (mirror MetabolismController async surface) ─

    async def get_current_state(self) -> str:
        result = await self._work_rpc_async("get_current_state")
        if isinstance(result, dict):
            return str(result.get("result", _COLD_DEFAULT_TIER))
        return _COLD_DEFAULT_TIER

    async def can_afford(self, cost: float) -> bool:
        result = await self._work_rpc_async("can_afford", {"cost": cost})
        if isinstance(result, dict):
            return bool(result.get("result", False))
        return False

    async def get_metabolic_health(self) -> float:
        result = await self._work_rpc_async("get_metabolic_health")
        if isinstance(result, dict):
            return float(result.get("result", 0.5) or 0.5)
        return 0.5

    async def get_learning_velocity(self) -> float:
        result = await self._work_rpc_async("get_learning_velocity")
        if isinstance(result, dict):
            return float(result.get("result", 0.5) or 0.5)
        return 0.5

    async def get_directive_alignment(self) -> float:
        result = await self._work_rpc_async("get_directive_alignment")
        if isinstance(result, dict):
            return float(result.get("result", 0.5) or 0.5)
        return 0.5

    async def get_social_density(self) -> float:
        result = await self._work_rpc_async("get_social_density")
        if isinstance(result, dict):
            return float(result.get("result", 0.5) or 0.5)
        return 0.5

    # ── Sync gate-detail queries (work-RPC; infrequent) ─────────────

    def get_service_gate(self, feature: str) -> tuple[bool, float, str]:
        result = self._work_rpc_sync("get_service_gate", {"feature": feature})
        if isinstance(result, dict) and "result" in result:
            res = result["result"]
            if isinstance(res, (list, tuple)) and len(res) >= 3:
                return bool(res[0]), float(res[1]), str(res[2])
        return False, 0.0, "unknown"

    def get_gate_decision_summary(self) -> dict:
        result = self._work_rpc_sync("get_gate_decision_summary")
        if isinstance(result, dict):
            inner = result.get("result")
            if isinstance(inner, dict):
                return inner
        return {}

    def get_emergency_duration(self) -> float:
        result = self._work_rpc_sync("get_emergency_duration")
        if isinstance(result, dict):
            return float(result.get("result", 0.0) or 0.0)
        return 0.0

    # ── Diagnostic reads ────────────────────────────────────────────

    def get_gate_ring(self, limit: int = 64) -> list:
        result = self._work_rpc_sync("get_gate_ring", {"limit": limit})
        if isinstance(result, dict):
            data = result.get("result")
            if isinstance(data, list):
                return data
        return []

    def get_tier_history(self) -> list:
        result = self._work_rpc_sync("get_tier_history")
        if isinstance(result, dict):
            data = result.get("result")
            if isinstance(data, list):
                return data
        return []
