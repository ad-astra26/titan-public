"""
inner_body_sensor_refresh — Phase C C-S5 Python sensor cache writer for inner body.

SPEC §G1 (Inner-Body 5D) + §23.4 (formulas) + §9.A line 1014 (sensor cache layout).

Writes 5 × float32 LE = 20 bytes raw inner body tensor (interoception,
proprioception, somatosensation, entropy, thermal) to
``/dev/shm/titan_<id>/sensor_cache_inner_body.bin`` at Schumann body rate
(7.83 Hz).

The Rust daemon ``titan-inner-body-rs`` reads this slot every Schumann
tick, applies UNIFIED + LOCAL filter_down multipliers + GROUND_UP
nudge, and writes the result to ``inner_body_5d.bin``.

──────────────────────────────────────────────────────────────────────
Why this module exists (C-S5 closure 2026-05-08)
──────────────────────────────────────────────────────────────────────
Phase C C-S5 shipped the Rust inner trinity daemons
(``titan-inner-{body,mind,spirit}-rs``) but never created the Python
sensor refresh writers their tick loops require as input. Result on
T3 after l0_rust_enabled flag flip: inner trinity Rust binaries STUCK
at version=2 because ``sensor_cache_inner_*.bin`` slots don't exist →
``Slot::open`` returns None → ``read_sensor_cache`` falls back to
``[0.0; 5]`` → ContentGate suppresses every write because content
unchanged.

This module + its mind / spirit siblings (``inner_mind_sensor_refresh``,
``inner_spirit_sensor_refresh``) close the gap. Reuses the canonical
Python compute (``body_worker._collect_body_tensor``) so the future
Rust port stays byte-identical with the Phase A+B implementation.

──────────────────────────────────────────────────────────────────────
Cadence (SPEC §G13 — locked by biology, NOT tunable)
──────────────────────────────────────────────────────────────────────
Schumann body fundamental = 7.83 Hz, period 127.7 ms. Matches the
Rust daemon's tick rate so the daemon sees fresh sensor input every
tick.

──────────────────────────────────────────────────────────────────────
Slot byte layout
──────────────────────────────────────────────────────────────────────
  [0:16]   16-byte slot header (universal triple-buffer per SPEC §7.0)
  [16:20]  buffer 0: 16-byte buffer-meta (wall_ns + payload_bytes + crc32)
                      followed by N-byte payload (here N = 20)
  [...]    buffer 1, buffer 2 (same shape — triple-buffer)

Payload = 5 × float32 LE = 20 bytes:
  [0:4]   interoception
  [4:8]   proprioception
  [8:12]  somatosensation
  [12:16] entropy
  [16:20] thermal

Slot is variable_size up to 4096B per Rust ``spec.rs:235`` —
``StateRegistryWriter.write_variable`` writes the actual 20-byte
payload; the Rust daemon's ``read_sensor_cache`` reads first 5 floats.

──────────────────────────────────────────────────────────────────────
Failure semantics
──────────────────────────────────────────────────────────────────────
- ``tensor_provider()`` raises → counter incremented, throttled WARN
  log, slot retains last-known payload. Next tick retries.
- ``tensor_provider()`` returns wrong shape → counter incremented,
  WARN log, skip write. Next tick retries.
- ``write_variable`` raises → counter incremented, throttled WARN log.
  Next tick retries.
- Thread itself crashes → caught, logged CRITICAL, exits. Caller
  responsible for restart (in_process exception handler per SPEC §11.B
  line 1236; same pattern as outer sensor refresh).

──────────────────────────────────────────────────────────────────────
Threading model
──────────────────────────────────────────────────────────────────────
Hosted INSIDE the body_worker subprocess (``titan_hcl.modules.body_worker``)
because the sense functions ``_sense_interoception/proprioception/etc.``
are subprocess-local. Runs as a daemon thread alongside the existing
sensor refresh threads. Wired under ``microkernel.l0_rust_enabled=true``
ONLY — under l0_rust=false the legacy ``body_5d_writer`` writes
``inner_body_5d.bin`` directly + this sensor cache writer is NOT
instantiated.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from typing import Optional

import numpy as np

from titan_hcl._phase_c_constants import SCHUMANN_BODY_HZ
from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


# ── Constants (SPEC-bound) ──────────────────────────────────────────

#: Slot basename per SPEC §7.1 + spec.rs (Rust). No ".bin" suffix —
#: StateRegistryWriter appends it.
SLOT_NAME: str = "sensor_cache_inner_body"

#: Schema version for sensor_cache_inner_body.bin payload shape.
#: Bump when the dim count or float ordering changes.
SCHEMA_VERSION: int = 1

#: Tensor dim count per SPEC §G1 + §23.4. NEVER change without SPEC
#: amendment + Rust ``read_sensor_cache`` update.
DIMS: int = 5

#: Actual payload size in bytes (5 × float32 LE).
PAYLOAD_BYTES: int = DIMS * 4  # 20

#: Slot capacity per spec.rs:235 — variable size up to 4096B.
MAX_PAYLOAD_BYTES: int = 4096

#: Refresh period (s). Matches Schumann body fundamental per SPEC §G13.
PERIOD_S: float = 1.0 / SCHUMANN_BODY_HZ  # ≈ 0.1277

#: Throttle period for repeated WARN logs (avoid log flood at 7.83 Hz).
_WARN_THROTTLE_EVERY: int = 100

#: Drift recovery threshold — if behind schedule by more than 5 periods,
#: skip ahead instead of trying to catch up.
_DRIFT_RECOVERY_PERIODS: int = 5

#: Minimum seconds between consecutive drift-recovery WARN logs. The drift
#: condition can persist for thousands of consecutive ticks on a CPU-
#: contended host (shared-VPS Titan pair). Without throttling, the WARN
#: floods journalctl + warning_monitor at sensor rate (7.83 Hz ≈ 28K WARN/h).
#: First drift fires immediately; subsequent within this window are
#: aggregated and emitted once every _DRIFT_WARN_THROTTLE_S with the
#: accumulated count + max-behind delta.
_DRIFT_WARN_THROTTLE_S: float = 60.0


TensorProvider = Callable[[], Optional[np.ndarray]]


class InnerBodySensorRefresh:
    """
    SPEC §9.A inner-body sensor cache writer.

    Polls ``tensor_provider`` at Schumann body rate (7.83 Hz),
    encodes 5 × float32 LE, writes to ``sensor_cache_inner_body.bin``
    via SeqLock variable-size write.

    Lifecycle::

        sidecar = InnerBodySensorRefresh(
            tensor_provider=lambda: np.array(_collect_body_tensor(...)[0]),
            titan_id="T3",
        )
        thread = sidecar.start_thread(stop_event)
        ...
        stop_event.set()  # signals graceful exit
        thread.join(timeout=2.0)

    Single-instance assumption (G21 — one writer per slot).
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

        # Throttled drift-recovery WARN state (see _DRIFT_WARN_THROTTLE_S).
        self._drift_recovery_count: int = 0
        self._drift_recovery_max_behind_s: float = 0.0
        self._drift_recovery_last_warn_ts: float = 0.0

    # -- public lifecycle -------------------------------------------

    def start_thread(self, stop_event: threading.Event) -> threading.Thread:
        """
        Spawn the writer daemon thread. Returns the thread handle so
        caller can join() at shutdown.
        """
        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="inner_body_sensor_refresh",
            daemon=True,
        )
        thread.start()
        logger.info(
            "inner_body_sensor_refresh started (period=%.4fs ≈ %.2fHz, slot=%s, "
            "dims=%d, max_bytes=%d)",
            self._period_s, 1.0 / self._period_s, SLOT_NAME, DIMS,
            MAX_PAYLOAD_BYTES,
        )
        return thread

    # -- introspection ----------------------------------------------

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

    def _run(self, stop_event: threading.Event) -> None:
        """
        Top-level thread entry point. Builds writer + drives refresh
        ticks until ``stop_event`` is set. Catches all per-tick errors
        so a transient failure (provider raise, shape mismatch, write
        OSError) doesn't kill the thread.
        """
        try:
            self._writer = self._build_writer()
        except Exception:
            logger.critical(
                "inner_body_sensor_refresh: failed to build SeqLock writer "
                "for %s — thread exiting:\n%s",
                SLOT_NAME, traceback.format_exc(),
            )
            return

        next_tick = time.monotonic()
        while not stop_event.is_set():
            self._tick_once()
            self._tick_count += 1

            # Schedule next tick — accumulate against absolute time so
            # transient stalls don't drift the cadence.
            next_tick += self._period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if stop_event.wait(timeout=sleep_for):
                    break  # stop signaled during sleep
            elif sleep_for < -self._period_s * _DRIFT_RECOVERY_PERIODS:
                # Behind by > 5 periods — skip ahead. Don't try to
                # catch up at full rate. Throttle the WARN: drift can
                # persist for thousands of consecutive ticks on a CPU-
                # contended host; one WARN per cycle floods logs at
                # sensor rate (~28K WARN/h on a 7.83 Hz cadence).
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
                            "inner_body_sensor_refresh drift recovery: "
                            "behind by %.2fs (>%d periods); resetting "
                            "cadence (will throttle further WARNs to 1 / "
                            "%.0fs)",
                            behind_s, _DRIFT_RECOVERY_PERIODS,
                            _DRIFT_WARN_THROTTLE_S,
                        )
                    else:
                        logger.warning(
                            "inner_body_sensor_refresh drift recovery: "
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

        # Graceful shutdown
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass  # best-effort on shutdown
            self._writer = None
        logger.info(
            "inner_body_sensor_refresh stopped after %d ticks "
            "(provider_failures=%d, shape_failures=%d, write_failures=%d)",
            self._tick_count, self._provider_failure_count,
            self._shape_failure_count, self._write_failure_count,
        )

    def _tick_once(self) -> None:
        """One refresh-and-write iteration. Catches per-tick errors.

        Step 7 schema migration v1→v2 (rFP §4.4): provider may now return
        msgpack-encoded source-dict bytes (Phase C inner trinity Rust L1
        port — Rust daemon decodes msgpack + executes per-dim formulas).
        Backward-compatible with legacy np.ndarray providers (Phase A+B
        path still uses the float32 LE writer in body_worker directly).
        """
        # Stage 1: compute tensor / source dict
        try:
            tensor = self._tensor_provider()
        except Exception:
            self._provider_failure_count += 1
            self._maybe_log_failure(
                "tensor_provider", self._provider_failure_count,
            )
            return

        if tensor is None:
            return  # provider returned None — silent skip (cold start)

        # Stage 2: choose payload format based on provider return type.
        # bytes → already-encoded msgpack source dict (schema v2, Phase C).
        # ndarray-like → float32 LE tensor (schema v1, legacy fallback).
        if isinstance(tensor, (bytes, bytearray)):
            payload = bytes(tensor)
        else:
            try:
                arr = np.asarray(tensor, dtype="<f4")
            except (TypeError, ValueError):
                self._shape_failure_count += 1
                self._maybe_log_failure(
                    "asarray-cast", self._shape_failure_count,
                )
                return
            if arr.shape != (DIMS,):
                self._shape_failure_count += 1
                self._maybe_log_failure(
                    "shape-mismatch", self._shape_failure_count,
                    extra=f"expected ({DIMS},), got {arr.shape}",
                )
                return
            payload = arr.tobytes()

        # Stage 3: write
        if self._writer is None:
            return  # writer build failed — _run loop will exit

        try:
            self._writer.write_variable(payload)
        except Exception:
            self._write_failure_count += 1
            self._maybe_log_failure(
                "write_variable", self._write_failure_count,
            )

    def _maybe_log_failure(
        self,
        stage: str,
        count: int,
        extra: str = "",
    ) -> None:
        """Throttled WARN log — first occurrence + every Nth subsequent."""
        if count <= 5 or count % _WARN_THROTTLE_EVERY == 0:
            logger.warning(
                "inner_body_sensor_refresh %s failure (count=%d, tick=%d) %s\n%s",
                stage, count, self._tick_count, extra,
                traceback.format_exc(),
            )
