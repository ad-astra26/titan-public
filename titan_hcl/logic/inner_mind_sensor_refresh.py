"""
inner_mind_sensor_refresh — Phase C C-S5 Python sensor cache writer for inner mind.

SPEC §G1 (Inner-Mind 15D) + §23.5 (formulas) + §9.A line 1034 (sensor cache).

Writes 15 × float32 LE = 60 bytes raw inner mind tensor (Thinking[0:5]
+ Feeling[5:10] + Willing[10:15]) to
``/dev/shm/titan_<id>/sensor_cache_inner_mind.bin`` at Schumann mind
rate (23.49 Hz = Schumann body × 3).

The Rust daemon ``titan-inner-mind-rs`` reads this slot every Schumann
mind tick, applies UNIFIED + LOCAL filter_down + GROUND_UP nudge to
willing[10:15] only (per SPEC §G10), and writes ``inner_mind_15d.bin``.

──────────────────────────────────────────────────────────────────────
Why this module exists (C-S5 closure 2026-05-08)
──────────────────────────────────────────────────────────────────────
See ``inner_body_sensor_refresh.py`` docstring for the full C-S5 gap
context. This is the mind sibling — same architectural role.

──────────────────────────────────────────────────────────────────────
Cadence + Slot byte layout
──────────────────────────────────────────────────────────────────────
- Period: 1 / SCHUMANN_MIND_HZ ≈ 42.6 ms (locked by SPEC §G13 biology)
- Payload: 15 × float32 LE = 60 bytes
  - [0:5]   Thinking — memory_depth, social_cognition, perceptual_thinking,
            emotional_thinking, conceptual_thinking (per §23.5)
  - [5:10]  Feeling — inner_hearing, inner_touch, inner_sight,
            inner_taste, inner_smell
  - [10:15] Willing — action_drive, social_will, creative_will,
            protective_will, growth_will
- Slot variable_size up to 4096B per Rust spec.rs:243

──────────────────────────────────────────────────────────────────────
Threading model
──────────────────────────────────────────────────────────────────────
Hosted INSIDE the mind_worker subprocess. Tensor compute is
``mind_tensor.collect_mind_15d`` (logic/mind_tensor.py); inputs come
from mind_worker's in-memory cache + plugin_cache (already populated
by mind_worker's existing refresh threads). Wired under
``microkernel.l0_rust_enabled=true`` ONLY.

Failure semantics, drift recovery, throttled WARN logs — same pattern
as ``inner_body_sensor_refresh``. See that module for prose.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from typing import Optional

import numpy as np

from titan_hcl._phase_c_constants import SCHUMANN_MIND_HZ
from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

SLOT_NAME: str = "sensor_cache_inner_mind"
SCHEMA_VERSION: int = 1
DIMS: int = 15
PAYLOAD_BYTES: int = DIMS * 4  # 60
MAX_PAYLOAD_BYTES: int = 4096
PERIOD_S: float = 1.0 / SCHUMANN_MIND_HZ  # ≈ 0.04258
_WARN_THROTTLE_EVERY: int = 100
_DRIFT_RECOVERY_PERIODS: int = 5

# Min seconds between drift-recovery WARNs — see inner_body_sensor_refresh
# for rationale (sustained drift on CPU-contended hosts otherwise floods
# logs at sensor rate).
_DRIFT_WARN_THROTTLE_S: float = 60.0


TensorProvider = Callable[[], Optional[np.ndarray]]


class InnerMindSensorRefresh:
    """
    SPEC §9.A inner-mind sensor cache writer. See module docstring +
    ``InnerBodySensorRefresh`` for the architectural rationale and
    failure semantics.
    """

    def __init__(
        self,
        tensor_provider: TensorProvider,
        titan_id: Optional[str] = None,
        period_s: float = PERIOD_S,
    ) -> None:
        self._tensor_provider = tensor_provider
        self._titan_id = titan_id
        self._period_s = period_s

        self._writer: Optional[StateRegistryWriter] = None
        self._tick_count: int = 0
        self._provider_failure_count: int = 0
        self._shape_failure_count: int = 0
        self._write_failure_count: int = 0

        # Throttled drift-recovery WARN state.
        self._drift_recovery_count: int = 0
        self._drift_recovery_max_behind_s: float = 0.0
        self._drift_recovery_last_warn_ts: float = 0.0

    def start_thread(self, stop_event: threading.Event) -> threading.Thread:
        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="inner_mind_sensor_refresh",
            daemon=True,
        )
        thread.start()
        logger.info(
            "inner_mind_sensor_refresh started (period=%.4fs ≈ %.2fHz, slot=%s, "
            "dims=%d, max_bytes=%d)",
            self._period_s, 1.0 / self._period_s, SLOT_NAME, DIMS,
            MAX_PAYLOAD_BYTES,
        )
        return thread

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def provider_failure_count(self) -> int:
        return self._provider_failure_count

    @property
    def shape_failure_count(self) -> int:
        return self._shape_failure_count

    @property
    def write_failure_count(self) -> int:
        return self._write_failure_count

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

    def _run(self, stop_event: threading.Event) -> None:
        try:
            self._writer = self._build_writer()
        except Exception:
            logger.critical(
                "inner_mind_sensor_refresh: failed to build SeqLock writer "
                "for %s — thread exiting:\n%s",
                SLOT_NAME, traceback.format_exc(),
            )
            return

        next_tick = time.monotonic()
        while not stop_event.is_set():
            self._tick_once()
            self._tick_count += 1

            next_tick += self._period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if stop_event.wait(timeout=sleep_for):
                    break
            elif sleep_for < -self._period_s * _DRIFT_RECOVERY_PERIODS:
                # Throttle WARN — see inner_body_sensor_refresh for rationale.
                behind_s = -sleep_for
                now_mono = time.monotonic()
                self._drift_recovery_count += 1
                if behind_s > self._drift_recovery_max_behind_s:
                    self._drift_recovery_max_behind_s = behind_s
                if (self._drift_recovery_last_warn_ts == 0.0
                        or now_mono - self._drift_recovery_last_warn_ts
                        >= _DRIFT_WARN_THROTTLE_S):
                    if self._drift_recovery_last_warn_ts == 0.0:
                        logger.warning(
                            "inner_mind_sensor_refresh drift recovery: "
                            "behind by %.2fs (>%d periods); resetting "
                            "cadence (will throttle further WARNs to 1 / "
                            "%.0fs)",
                            behind_s, _DRIFT_RECOVERY_PERIODS,
                            _DRIFT_WARN_THROTTLE_S,
                        )
                    else:
                        logger.warning(
                            "inner_mind_sensor_refresh drift recovery: "
                            "%d sustained drift events in last %.0fs "
                            "(max behind=%.2fs); resetting cadence",
                            self._drift_recovery_count,
                            now_mono - self._drift_recovery_last_warn_ts,
                            self._drift_recovery_max_behind_s,
                        )
                    self._drift_recovery_last_warn_ts = now_mono
                    self._drift_recovery_count = 0
                    self._drift_recovery_max_behind_s = 0.0
                next_tick = now_mono

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        logger.info(
            "inner_mind_sensor_refresh stopped after %d ticks "
            "(provider_failures=%d, shape_failures=%d, write_failures=%d)",
            self._tick_count, self._provider_failure_count,
            self._shape_failure_count, self._write_failure_count,
        )

    def _tick_once(self) -> None:
        # Step 8 §4.5 schema migration v1→v2: provider may return msgpack
        # bytes (Phase C source-dict pattern) OR np.ndarray (legacy float32 LE).
        try:
            tensor = self._tensor_provider()
        except Exception:
            self._provider_failure_count += 1
            self._maybe_log_failure("tensor_provider", self._provider_failure_count)
            return

        if tensor is None:
            return

        if isinstance(tensor, (bytes, bytearray)):
            payload = bytes(tensor)
        else:
            try:
                arr = np.asarray(tensor, dtype="<f4")
            except (TypeError, ValueError):
                self._shape_failure_count += 1
                self._maybe_log_failure("asarray-cast", self._shape_failure_count)
                return
            if arr.shape != (DIMS,):
                self._shape_failure_count += 1
                self._maybe_log_failure(
                    "shape-mismatch", self._shape_failure_count,
                    extra=f"expected ({DIMS},), got {arr.shape}",
                )
                return
            payload = arr.tobytes()

        if self._writer is None:
            return

        try:
            self._writer.write_variable(payload)
        except Exception:
            self._write_failure_count += 1
            self._maybe_log_failure("write_variable", self._write_failure_count)

    def _maybe_log_failure(self, stage: str, count: int, extra: str = "") -> None:
        if count <= 5 or count % _WARN_THROTTLE_EVERY == 0:
            logger.warning(
                "inner_mind_sensor_refresh %s failure (count=%d, tick=%d) %s\n%s",
                stage, count, self._tick_count, extra,
                traceback.format_exc(),
            )
