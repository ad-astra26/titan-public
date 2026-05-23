"""
focus_pid_publisher — write the §G12 FOCUS cascade nudge sidecar.

Per SPEC §G5.2 item 2 + §G12 (D-SPEC-112) + PLAN_trinity_homeostasis_p0 §1.4 +
§6: this publisher runs the canonical [`FocusPID`] controllers against each of
the 6 trinity tensors (inner/outer × body/mind/spirit) and writes a fixed-layout
SHM sidecar ``focus_input.bin`` (528 bytes) under the per-Titan ``shm_dir``.

Each Rust daemon retry-opens the slot at boot and reads its part's slice on
every Schumann tick, composing the nudge into ``enrichment_force`` for the
§G5.2 stateful-update kernel (see
``titan-trinity-daemon::focus_input::read_nudge``).

Single-writer per G21/INV-4: NO other process writes ``focus_input.bin``.

## Byte layout (528 bytes, all f32 little-endian)

    [0..4]    ts                       — Python time.time()
    [4..8]    stale_focus_multiplier   — ≥ 1.0 (cascade amplifier)
    [8..28]   inner_body 5×f32         — 20 bytes
    [28..88]  inner_mind 15×f32        — 60 bytes
    [88..268] inner_spirit 45×f32      — 180 bytes
    [268..288] outer_body 5×f32
    [288..348] outer_mind 15×f32
    [348..528] outer_spirit 45×f32

## §G12 stale_focus_multiplier (PLAN §1.4)

The PLAN says the multiplier "folds into the same slot (the cascade
SPIRIT→Lower-Spirit→Mind→Body amplifies the nudge to the imbalanced part)".
At P0 we compute a conservative scalar from the inner-spirit tensor's L2
distance from the 0.5 Divine Centre — when spirit drifts off-centre the
multiplier rises (≥ 1.0, clamped ≤ 8.0). This is the simplest unified-spirit
"is_stale" proxy that respects the SPEC's "amplify the cascade when STALE"
intent without a full UnifiedSpirit-stale wire-up at P0. The unified-spirit
detector (§G11) ships separately; once it publishes a stale flag, this
publisher can read it directly — for P0 the L2-from-centre proxy is the
intentionally-loose surrogate.
"""
from __future__ import annotations

import logging
import math
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

from titan_hcl._phase_c_constants import SCHUMANN_BODY_HZ
from titan_hcl.core.state_registry import ensure_shm_root
from titan_hcl.logic.focus_pid import FocusPID

logger = logging.getLogger(__name__)

# Fixed-layout sidecar — MUST match titan-trinity-daemon::focus_input.
FOCUS_INPUT_SIDECAR = "focus_input.bin"
FOCUS_INPUT_PAYLOAD_BYTES = 528

# Byte offsets (mirror Rust titan-trinity-daemon::focus_input::OFFSET_*).
OFFSET_TS = 0
OFFSET_STALE_FOCUS_MULT = 4
OFFSET_INNER_BODY = 8
OFFSET_INNER_MIND = 28
OFFSET_INNER_SPIRIT = 88
OFFSET_OUTER_BODY = 268
OFFSET_OUTER_MIND = 288
OFFSET_OUTER_SPIRIT = 348

# Cadence: write at the Schumann-body fundamental (≈ 7.83 Hz) so daemons reading
# the slot at ~1 Hz retry cadence always see a fresh nudge. Body's the slowest
# Schumann harmonic so 1/7.83 is a safe upper bound on staleness.
PERIOD_S = 1.0 / SCHUMANN_BODY_HZ  # ≈ 0.1277 s

# Cascade amplifier bounds — matches Rust clamp in
# titan-trinity-daemon::focus_input::read_nudge.
STALE_FOCUS_MULT_MIN = 1.0
STALE_FOCUS_MULT_MAX = 8.0


def _l2_from_centre(tensor: Sequence[float]) -> float:
    """L2 distance from the 0.5 Divine Centre — `[0, sqrt(N·0.25)]`."""
    return math.sqrt(sum((float(v) - 0.5) ** 2 for v in tensor))


def _compute_stale_focus_multiplier(inner_spirit_45d: Optional[Sequence[float]]) -> float:
    """Map inner-spirit's L2-from-centre to the cascade amplifier ∈ [1.0, 8.0].

    Conservative proxy at P0: when spirit drifts far from centre we want
    SPIRIT→Lower-Spirit→Mind→Body amplification. Theoretical max L2 for a 45D
    tensor in [0,1] = sqrt(45 · 0.25) ≈ 3.354. Map [0, max] → [1.0, 8.0].
    Returns 1.0 (no amplification) when inner-spirit isn't yet readable —
    substrate continues; ≥ 1.0 is enforced by the Rust clamp anyway.
    """
    if not inner_spirit_45d:
        return STALE_FOCUS_MULT_MIN
    l2 = _l2_from_centre(inner_spirit_45d)
    max_l2 = math.sqrt(45.0 * 0.25)  # ≈ 3.354
    if max_l2 <= 0.0:
        return STALE_FOCUS_MULT_MIN
    frac = max(0.0, min(1.0, l2 / max_l2))
    mult = STALE_FOCUS_MULT_MIN + (STALE_FOCUS_MULT_MAX - STALE_FOCUS_MULT_MIN) * frac
    return max(STALE_FOCUS_MULT_MIN, min(STALE_FOCUS_MULT_MAX, mult))


def _pack_payload(
    ts: float,
    stale_focus_multiplier: float,
    inner_body: Sequence[float],
    inner_mind: Sequence[float],
    inner_spirit: Sequence[float],
    outer_body: Sequence[float],
    outer_mind: Sequence[float],
    outer_spirit: Sequence[float],
) -> bytes:
    """Pack the 528-byte focus_input.bin payload."""
    def _vec(v: Sequence[float], n: int) -> bytes:
        # Pad / truncate to exactly n floats, clamping NaN/Inf to 0.0.
        out = []
        for i in range(n):
            if i < len(v):
                x = float(v[i])
                if not math.isfinite(x):
                    x = 0.0
            else:
                x = 0.0
            out.append(x)
        return struct.pack(f"<{n}f", *out)

    parts = [
        struct.pack("<f", float(ts)),
        struct.pack("<f", float(stale_focus_multiplier)),
        _vec(inner_body, 5),
        _vec(inner_mind, 15),
        _vec(inner_spirit, 45),
        _vec(outer_body, 5),
        _vec(outer_mind, 15),
        _vec(outer_spirit, 45),
    ]
    payload = b"".join(parts)
    assert (
        len(payload) == FOCUS_INPUT_PAYLOAD_BYTES
    ), f"payload size {len(payload)} != {FOCUS_INPUT_PAYLOAD_BYTES}"
    return payload


class FocusPIDPublisher:
    """Publishes ``focus_input.bin`` at ~7.83 Hz from the 6 canonical tensors.

    Construct with a ``trinity_reader`` callable returning the latest tensor
    snapshot (``{inner_body: [5], inner_mind: [15], inner_spirit: [45],
    outer_body: [5], outer_mind: [15], outer_spirit: [45]}`` — None entries
    mean "not yet readable"). The publisher runs the canonical FocusPID per
    part, computes the §G12 cascade amplifier, and writes the sidecar.

    Atomic write via tmp + rename so daemons never read a half-written payload.
    Failure mode: log + continue (substrate continues per §11.B).

    Lifecycle::

        publisher = FocusPIDPublisher(trinity_reader)
        thread = publisher.start_thread(stop_event)
        ...
        stop_event.set()
        thread.join(timeout=2.0)

    Single-instance per Titan (G21 / INV-4 — one writer for focus_input.bin).
    """

    def __init__(
        self,
        trinity_reader,  # callable() -> dict[str, list[float] | None]
        titan_id: Optional[str] = None,
        period_s: float = PERIOD_S,
    ) -> None:
        self._trinity_reader = trinity_reader
        self._titan_id = titan_id
        self._period_s = max(0.05, float(period_s))
        # 6 per-part PIDs — names match focus_input part keys.
        self._pids: dict[str, FocusPID] = {
            "inner_body": FocusPID("inner_body", 5),
            "inner_mind": FocusPID("inner_mind", 15),
            "inner_spirit": FocusPID("inner_spirit", 45),
            "outer_body": FocusPID("outer_body", 5),
            "outer_mind": FocusPID("outer_mind", 15),
            "outer_spirit": FocusPID("outer_spirit", 45),
        }
        self._tick_count = 0
        self._write_failure_count = 0

    # -- public lifecycle ----------------------------------------------

    def start_thread(self, stop_event: threading.Event) -> threading.Thread:
        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="focus_pid_publisher",
            daemon=True,
        )
        thread.start()
        return thread

    def _run(self, stop_event: threading.Event) -> None:
        try:
            shm_dir = ensure_shm_root(self._titan_id)
            target = shm_dir / FOCUS_INPUT_SIDECAR
            logger.info(
                "[focus_pid_publisher] starting: target=%s period=%.4fs",
                target,
                self._period_s,
            )
            next_t = time.monotonic()
            while not stop_event.is_set():
                self._tick(shm_dir, target)
                next_t += self._period_s
                wait = max(0.0, next_t - time.monotonic())
                if wait > 0.0:
                    stop_event.wait(wait)
        except Exception:
            # Surface — daemon thread crashes are caught by the worker host.
            logger.exception("[focus_pid_publisher] thread crashed; exiting")

    # -- internals -----------------------------------------------------

    def _tick(self, shm_dir: Path, target: Path) -> None:
        try:
            snap = self._trinity_reader() or {}
        except Exception:
            logger.exception(
                "[focus_pid_publisher] trinity_reader failed (#%d); skipping tick",
                self._tick_count,
            )
            self._tick_count += 1
            return

        # Each part: if tensor present, run its PID. If absent, emit zeros
        # for that part (no nudge — substrate keeps running on default).
        nudges: dict[str, list[float]] = {}
        for name, pid in self._pids.items():
            tensor = snap.get(name)
            if tensor is None:
                nudges[name] = [0.0] * pid.dims
                continue
            try:
                nudges[name] = pid.update(tensor)
            except Exception:
                logger.exception(
                    "[focus_pid_publisher] FocusPID(%s).update failed; zero nudge",
                    name,
                )
                nudges[name] = [0.0] * pid.dims

        mult = _compute_stale_focus_multiplier(snap.get("inner_spirit"))
        payload = _pack_payload(
            ts=time.time(),
            stale_focus_multiplier=mult,
            inner_body=nudges["inner_body"],
            inner_mind=nudges["inner_mind"],
            inner_spirit=nudges["inner_spirit"],
            outer_body=nudges["outer_body"],
            outer_mind=nudges["outer_mind"],
            outer_spirit=nudges["outer_spirit"],
        )

        # Atomic write: write to a same-FS tmp file + os.replace.
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".focus_input.", dir=str(shm_dir))
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                os.replace(tmp_name, str(target))
            except Exception:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
                raise
        except Exception:
            self._write_failure_count += 1
            if self._write_failure_count <= 3 or self._write_failure_count % 100 == 0:
                logger.exception(
                    "[focus_pid_publisher] write %s failed (#%d)",
                    target,
                    self._write_failure_count,
                )

        self._tick_count += 1
