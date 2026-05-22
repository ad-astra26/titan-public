"""
dream_state_publisher — owns dream_state.bin SHM writer.

Phase C v1.8.2 (D-SPEC-56) per `rFP_titan_hcl_l2_separation_strategy.md §4.I`.

Invoked from dream_state_worker on dual-trigger cadence per Maker Q6 greenlight
(2026-05-15): on every `KERNEL_EPOCH_TICK` (1.0 Hz adaptive,
DREAM_STATE_REPUBLISH_CADENCE_S) AND on every `DREAMING_STATE_UPDATED` arrival
(immediate republish on transition). Single-threaded (G21 single-writer-per-slot).

Hot-path `is_dreaming` reads from many consumers (plugin chat-during-dream
buffer decision, api_subprocess chat-bridge buffer decision, spirit_worker
`_read_is_dreaming_from_shm()` helper replacing the deleted
`_shared_is_dreaming` module-level flag + 20+ readers, expression_worker
tick-gate cache, timechain_worker dream-hook) bypass the bus entirely via
this slot per G18+G20.

Cold-boot safe — first publish before any DREAMING_STATE_UPDATED arrives uses
defaults (is_dreaming=False, state="awake", recovery_pct=0.0, last_transition_ts=
boot_ts). Readers see the cold values and proceed.

Failure modes mirror MemoryStatePublisher + SocialGraphStatePublisher +
MetabolismStatePublisher precedents:
  - encode/oversize/write fails handled per-tick; first WARN with exc_info,
    subsequent throttled to every 60 ticks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.dream_state_specs import (
    DREAM_STATE_SLOT,
    DREAM_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)

# Valid `state` enum values per SPEC §7.1 dream_state.bin schema.
_VALID_STATES = ("awake", "dreaming", "dream_start", "dream_end")


class DreamStatePublisher:
    """Owns dream_state.bin SHM writer (G21 single-writer).

    Stateful: tracks (is_dreaming, dream_started_ts, wake_ts,
    last_transition_ts) across calls so transition timestamps are accurate
    even when the publisher is invoked on KERNEL_EPOCH_TICK without a fresh
    DREAMING_STATE_UPDATED arrival.
    """

    def __init__(self, titan_id: str):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        # Sticky state across publishes (DREAMING_STATE_UPDATED carries the
        # authoritative source; we cache the last-observed fields here so the
        # 1Hz KERNEL_EPOCH_TICK republish path doesn't lose them).
        self._last_is_dreaming: bool = False
        self._last_state: str = "awake"
        self._last_recovery_pct: float = 0.0
        self._last_remaining_epochs: int = 0
        self._wake_transition: bool = False
        self._just_woke: bool = False
        self._wake_ts: float = 0.0
        self._dream_started_ts: float = 0.0
        self._last_transition_ts: float = time.time()  # boot ts as initial freshness anchor
        # Additive lifetime/circadian telemetry — the DREAMING_STATE_UPDATED
        # payload carries the full Dreaming.get_stats() output (cycle_count +
        # last_fatigue) plus developmental_age from PiHeartbeat. The
        # Observatory DreamingIndicator + CircadianClock consume these; cache
        # them sticky so the 1Hz republish path doesn't lose them.
        self._last_cycle_count: int = 0
        self._last_fatigue: float = 0.0
        self._last_developmental_age: int = 0
        self._last_epochs_since_dream: int = 0
        # Distillation telemetry (rFP_experience_distillation_phase_c Bug B) —
        # the DREAMING_STATE_UPDATED payload already carries the full
        # Dreaming.get_stats() output incl. these counters; surface them in the
        # published slot so the health check reads real distill state instead of
        # defaulting to 0 (false "DISTILLATION DISCONNECTED"). Sticky.
        self._last_distill_attempts: int = 0
        self._last_distill_passed: int = 0
        self._last_distilled_count: int = 0
        self._last_distill_threshold: float = 0.0
        self._last_experience_buffer_size: int = 0
        logger.info(
            "[DreamStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.8.2 / Preamble G18 + G21 / D-SPEC-56)",
            titan_id, self._shm_root, DREAM_STATE_SLOT)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(DREAM_STATE_SPEC, self._shm_root)
        logger.info(
            "[DreamStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            DREAM_STATE_SLOT, DREAM_STATE_SPEC.payload_bytes,
            DREAM_STATE_SPEC.schema_version,
            self._shm_root / f"{DREAM_STATE_SLOT}.bin")
        return self._writer

    def update_from_dreaming_state(self, payload: dict[str, Any]) -> bool:
        """Ingest a fresh `DREAMING_STATE_UPDATED` bus payload and return True
        if a transition was detected (caller can decide to also emit
        DREAM_STATE_CHANGED + flush dream_inbox + emit DREAM_WAKE_FORWARD).

        Payload shape per spirit_loop._publish_coord_subdomains (lines 2101-2146):
          {is_dreaming: bool, state: str (∈ _VALID_STATES),
           developmental_age: float, dream_profile: dict|None,
           ...everything from coordinator.dreaming.get_stats()}
        """
        # Defensive: reject non-dict payloads upfront — bus events should
        # always carry dicts, but a malformed producer shouldn't crash the
        # worker.
        if not isinstance(payload, dict):
            logger.warning(
                "[DreamStatePublisher] malformed DREAMING_STATE_UPDATED payload "
                "(not a dict): type=%s", type(payload).__name__)
            return False
        try:
            new_is_dreaming = bool(payload.get("is_dreaming", False))
            new_state = payload.get("state", "awake")
            if new_state not in _VALID_STATES:
                # Tolerate non-canonical state strings — fall back to {awake,
                # dreaming} per is_dreaming flag rather than reject.
                new_state = "dreaming" if new_is_dreaming else "awake"
        except (TypeError, ValueError) as e:
            logger.warning(
                "[DreamStatePublisher] malformed DREAMING_STATE_UPDATED payload: %s",
                e, exc_info=True)
            return False

        # Edge detection — compare against last-observed sticky state.
        transitioned = (
            new_is_dreaming != self._last_is_dreaming
            or new_state in ("dream_start", "dream_end")
        )

        if transitioned:
            now_ts = time.time()
            self._last_transition_ts = now_ts
            if new_is_dreaming and not self._last_is_dreaming:
                # Entering dream.
                self._dream_started_ts = now_ts
                self._just_woke = False
                self._wake_transition = False
            elif self._last_is_dreaming and not new_is_dreaming:
                # Exiting dream.
                self._wake_ts = now_ts
                self._just_woke = True
                self._wake_transition = True

        self._last_is_dreaming = new_is_dreaming
        self._last_state = new_state

        # Capture additive circadian telemetry from the payload (full
        # Dreaming.get_stats() + developmental_age). Sticky so the 1Hz
        # republish path retains them between transitions.
        if "cycle_count" in payload:
            try:
                self._last_cycle_count = int(payload.get("cycle_count", 0) or 0)
            except (TypeError, ValueError):
                pass
        # Fatigue surfaces as `last_fatigue` in Dreaming.get_stats(); some
        # producers also send a flat `fatigue`.
        _fat = payload.get("last_fatigue", payload.get("fatigue"))
        if _fat is not None:
            try:
                self._last_fatigue = float(_fat)
            except (TypeError, ValueError):
                pass
        if "developmental_age" in payload:
            try:
                self._last_developmental_age = int(
                    payload.get("developmental_age", 0) or 0)
            except (TypeError, ValueError):
                pass
        if "epochs_since_dream" in payload:
            try:
                self._last_epochs_since_dream = int(
                    payload.get("epochs_since_dream", 0) or 0)
            except (TypeError, ValueError):
                pass
        # Distillation telemetry (rFP_experience_distillation_phase_c Bug B) —
        # sticky-capture from the get_stats() payload when present.
        for _dk, _attr, _cast in (
            ("distill_attempts", "_last_distill_attempts", int),
            ("distill_passed", "_last_distill_passed", int),
            ("distilled_count", "_last_distilled_count", int),
            ("distill_threshold", "_last_distill_threshold", float),
            ("experience_buffer_size", "_last_experience_buffer_size", int),
        ):
            if _dk in payload and payload.get(_dk) is not None:
                try:
                    setattr(self, _attr, _cast(payload.get(_dk)))
                except (TypeError, ValueError):
                    pass

        # Cache derived stats from the payload (recovery_pct is computed by
        # DreamingEngine.get_stats() under different names — use the same
        # heuristic as plugin.py:3413/3418 which set 0.0 on entry, 100.0 on
        # exit). If the payload carries `remaining_epochs` (legacy field used
        # by spirit_worker.py:3411-3412 expected_dream_epochs), surface it.
        # Otherwise estimate from epochs_since_dream if present.
        if new_is_dreaming:
            self._last_recovery_pct = 0.0
            self._last_remaining_epochs = int(
                payload.get("expected_dream_epochs",
                            payload.get("remaining_epochs", 0)) or 0
            )
        else:
            # On exit / awake, recovery_pct rolls toward 100% over the wake_transition window.
            # Simple model: 100% immediately post-wake, then sticky.
            self._last_recovery_pct = 100.0
            self._last_remaining_epochs = 0

        return transitioned

    def publish(self) -> None:
        """Encode current sticky state + write SHM. Called from worker on
        KERNEL_EPOCH_TICK + immediately after every successful
        `update_from_dreaming_state(...)`.
        """
        self._publish_count += 1
        payload = self._compute_payload()
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[DreamStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "is_dreaming=%s state=%s last_transition_age_s=%.1f",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                self._last_is_dreaming, self._last_state,
                time.time() - self._last_transition_ts)

    def snapshot_for_emit(self) -> dict[str, Any]:
        """Build a payload suitable for emitting on DREAM_STATE_CHANGED bus.
        Mirrors the SHM payload shape (callers shouldn't have to read SHM
        right after a transition emit).
        """
        return self._compute_payload()

    def _compute_payload(self) -> dict[str, Any]:
        now_ts = time.time()
        return {
            "is_dreaming": self._last_is_dreaming,
            "state": self._last_state,
            "recovery_pct": float(self._last_recovery_pct),
            "remaining_epochs": int(self._last_remaining_epochs),
            "wake_transition": bool(self._wake_transition),
            "just_woke": bool(self._just_woke),
            "wake_ts": float(self._wake_ts),
            "dream_started_ts": float(self._dream_started_ts),
            "last_transition_ts": float(self._last_transition_ts),
            # Additive circadian telemetry for DreamingIndicator + CircadianClock
            # + DreamingTab (Epochs Awake = epochs_since_dream, Dev Age).
            "cycle_count": int(self._last_cycle_count),
            "fatigue": round(float(self._last_fatigue), 4),
            "developmental_age": int(self._last_developmental_age),
            "epochs_since_dream": int(self._last_epochs_since_dream),
            # Distillation telemetry (rFP_experience_distillation_phase_c Bug B) —
            # surfaces real distill state to the health check via /v4/inner-trinity
            # so it no longer false-positives "DISTILLATION DISCONNECTED".
            "distill_attempts": int(self._last_distill_attempts),
            "distill_passed": int(self._last_distill_passed),
            "distilled_count": int(self._last_distilled_count),
            "distill_threshold": round(float(self._last_distill_threshold), 6),
            "experience_buffer_size": int(self._last_experience_buffer_size),
            "schema_version": DREAM_STATE_SPEC.schema_version,
            "ts": now_ts,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[DreamStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > DREAM_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[DreamStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), DREAM_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[DreamStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d is_dreaming=%s state=%s "
                    "(consumers can now read; closes Phase C DREAM_STATE_CHANGED "
                    "silent-emit bug per D-SPEC-56)",
                    DREAM_STATE_SLOT, len(encoded),
                    payload.get("is_dreaming"), payload.get("state"))
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[DreamStatePublisher] writer.write_variable failed (#%d): %s",
                    self._write_fails, e, exc_info=True)
