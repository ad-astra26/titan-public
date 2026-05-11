"""
inner_spirit_sensor_refresh — Phase C C-S5 Python sensor cache writer for inner spirit.

SPEC §G1 (Inner-Spirit 45D) + §23.6 (formulas) + §9.A line 1055.

Writes 45 × float32 LE = 180 bytes raw inner spirit tensor (SAT[0:15]
+ CHIT[15:30] + ANANDA[30:45]) to
``/dev/shm/titan_<id>/sensor_cache_inner_spirit.bin`` at Schumann
spirit rate (70.47 Hz = Schumann body × 9).

The Rust daemon ``titan-inner-spirit-rs`` reads this slot every Schumann
spirit tick, applies UNIFIED filter_down to content[5:45] (observer
[0:5] NEVER filtered per SPEC §G8), and writes ``inner_spirit_45d.bin``.

──────────────────────────────────────────────────────────────────────
Why 45D, not 5D
──────────────────────────────────────────────────────────────────────
Per SPEC §G1 (Preamble, LOAD-BEARING ground truth) the entire
inner-spirit dimension is 45D. Architectural geometry doesn't change
between Phase A+B and Phase C; what changes is WHO writes which slot,
not the dim count. Earlier C-S5 Rust implementation
(``inner-spirit-rs/tick_loop.rs::read_spirit_cache``) read only 5
floats — that is a bug to be fixed alongside this module landing
(see §17 task: ``read_spirit_cache`` widen to ``[f32; 45]``).

──────────────────────────────────────────────────────────────────────
Producer + reuse of Phase A+B Python compute
──────────────────────────────────────────────────────────────────────
Tensor compute is ``spirit_tensor.collect_spirit_45d`` (logic/spirit_tensor.py).
This is the canonical 130D-Trinity Phase 2.5 implementation just
shipped on T1+T2. Reusing it directly here means the Rust port (when
it lands) stays byte-identical with the Python ground truth — no
formula drift between languages.

──────────────────────────────────────────────────────────────────────
Cadence + Slot byte layout
──────────────────────────────────────────────────────────────────────
- Period: 1 / SCHUMANN_SPIRIT_HZ ≈ 14.2 ms (locked by SPEC §G13)
- Payload: 45 × float32 LE = 180 bytes
  - [0:15]   SAT — self_recognition, authenticity, sovereignty, ...
  - [15:30]  CHIT — self_awareness_depth, observation_clarity, ...
  - [30:45]  ANANDA — purpose_alignment, meaning_depth, creative_joy, ...
  (Full names + formulas per SPEC §23.6)
- Slot variable_size up to 4096B per Rust spec.rs.

──────────────────────────────────────────────────────────────────────
Threading model
──────────────────────────────────────────────────────────────────────
Hosted INSIDE the spirit_worker subprocess. Under l0_rust_enabled=true
spirit_worker enters "spirit-cache mode" (replaces heartbeat-stub):
keeps the spirit_tensor compute path live but writes to
``sensor_cache_inner_spirit.bin`` instead of ``inner_spirit_45d.bin``.
This is the cleanest architectural fit because spirit_worker already
has all the inputs spirit_tensor.collect_spirit_45d needs (body, mind,
consciousness, topology, hormones, hormone_fires, sphere_clocks,
memory stats, expression stats, unified_spirit_stats — all populated
by spirit_loop's snapshot-builder threads).

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

from titan_plugin._phase_c_constants import SCHUMANN_SPIRIT_HZ
from titan_plugin.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

SLOT_NAME: str = "sensor_cache_inner_spirit"
SCHEMA_VERSION: int = 1
DIMS: int = 45  # SPEC §G1 — Inner-Spirit 45D
PAYLOAD_BYTES: int = DIMS * 4  # 180
MAX_PAYLOAD_BYTES: int = 4096
PERIOD_S: float = 1.0 / SCHUMANN_SPIRIT_HZ  # ≈ 0.01419
_WARN_THROTTLE_EVERY: int = 100
_DRIFT_RECOVERY_PERIODS: int = 5


TensorProvider = Callable[[], Optional[np.ndarray]]


class InnerSpiritSensorRefresh:
    """
    SPEC §9.A inner-spirit sensor cache writer. Hosts the canonical
    spirit_tensor.collect_spirit_45d compute under l0_rust=true mode
    (replaces direct inner_spirit_45d.bin write that legacy
    spirit_worker performed under l0_rust=false).
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

    def start_thread(self, stop_event: threading.Event) -> threading.Thread:
        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="inner_spirit_sensor_refresh",
            daemon=True,
        )
        thread.start()
        logger.info(
            "inner_spirit_sensor_refresh started (period=%.4fs ≈ %.2fHz, slot=%s, "
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
                "inner_spirit_sensor_refresh: failed to build SeqLock writer "
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
                logger.warning(
                    "inner_spirit_sensor_refresh drift recovery: behind by %.2fs",
                    -sleep_for,
                )
                next_tick = time.monotonic()

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        logger.info(
            "inner_spirit_sensor_refresh stopped after %d ticks "
            "(provider_failures=%d, shape_failures=%d, write_failures=%d)",
            self._tick_count, self._provider_failure_count,
            self._shape_failure_count, self._write_failure_count,
        )

    def _tick_once(self) -> None:
        # Step 9 §4.6 schema migration v1→v2: provider may return msgpack
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
                "inner_spirit_sensor_refresh %s failure (count=%d, tick=%d) %s\n%s",
                stage, count, self._tick_count, extra,
                traceback.format_exc(),
            )
