"""
trinity_restoring_publisher — write the §G5.2 restoring-force gain sidecar.

Per SPEC §G5.2 item 5 + PLAN_trinity_homeostasis_p0 §1.5 + D-SPEC-112: the gain
coefficients for the §G5.2 stateful traveling-tensor integrator live in
``titan_hcl/titan_params.toml`` ``[trinity_restoring]`` (per-Titan tunable, NOT
hardcoded in Rust). Rather than parse TOML inside the Rust daemons (adds a
runtime dep + path-discovery), this module reads the section via the canonical
``titan_hcl.params.get_params`` 4-layer merge and writes a fixed-layout SHM
sidecar ``trinity_restoring.bin`` (8 × float32 LE = 32 bytes) under the per-Titan
``shm_dir``. Each of the 6 trinity daemons (inner/outer × body/mind/spirit)
retry-loads it at boot and refreshes at ~1 s cadence — see Rust
``titan_trinity_daemon::restoring_cfg::load_for_layer``.

Layer gradient (INV-9 body 0.7/0.3, mind 0.5/0.5, spirit 0.3/0.7) is STRUCTURAL
— fixed in the Rust kernel, NOT a TOML key. Neuromod gain is LIVE — read from
``neuromod_state.bin`` per tick, NOT a TOML key.

Field order (matches Rust ``restoring_cfg.rs``):

  [0] k_drive        [4] k_dir
  [1] k_restore      [5] a_mag
  [2] k_damp         [6] a_drift
  [3] k_mom          [7] a_dmag

Usage::

    from titan_hcl.logic.trinity_restoring_publisher import publish_trinity_restoring_cfg
    publish_trinity_restoring_cfg()  # call once at Python startup + on CONFIG_RELOAD

Atomic write via tmp + rename so daemons never read a half-written payload.
"""
from __future__ import annotations

import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional

from titan_hcl.core.state_registry import ensure_shm_root
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)

# Fixed-layout sidecar (must match titan-trinity-daemon::restoring_cfg).
TRINITY_RESTORING_SIDECAR = "trinity_restoring.bin"
TRINITY_RESTORING_PAYLOAD_BYTES = 32  # 8 × float32 LE

# Crate-default fallbacks (must match titan-trinity-daemon::homeostasis
# DEFAULT_* constants). Used ONLY when titan_params.toml omits a key — the
# publisher still writes a complete sidecar so daemons get the full payload.
_DEFAULTS: dict[str, float] = {
    "k_drive": 0.30,
    "k_restore": 0.05,
    "k_damp": 0.05,
    "k_mom": 0.10,
    "k_dir": 0.05,
    "a_mag": 0.50,
    "a_drift": 1.00,
    "a_dmag": 0.50,
}

# Field order — matches the Rust loader byte-for-byte.
_FIELD_ORDER = (
    "k_drive",
    "k_restore",
    "k_damp",
    "k_mom",
    "k_dir",
    "a_mag",
    "a_drift",
    "a_dmag",
)


def _resolve_gains() -> dict[str, float]:
    """Merge ``[trinity_restoring]`` over crate-defaults, returning 8 named floats."""
    cfg = get_params("trinity_restoring") or {}
    gains: dict[str, float] = {}
    for key, default in _DEFAULTS.items():
        raw = cfg.get(key, default)
        try:
            gains[key] = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "[trinity_restoring_publisher] non-numeric %s=%r — falling back to default %.3f",
                key,
                raw,
                default,
            )
            gains[key] = default
    return gains


def publish_trinity_restoring_cfg(titan_id: Optional[str] = None) -> Path:
    """Resolve gains from ``titan_params.toml`` and write the sidecar atomically.

    Returns the absolute path of the written file. Raises on I/O failure
    (caller surfaces — diagnosability over silence per
    ``directive_error_visibility``).
    """
    gains = _resolve_gains()
    payload = b"".join(struct.pack("<f", gains[k]) for k in _FIELD_ORDER)
    if len(payload) != TRINITY_RESTORING_PAYLOAD_BYTES:
        # Construction error — never silently ship a wrong-sized sidecar.
        raise RuntimeError(
            f"trinity_restoring sidecar payload size mismatch: "
            f"got {len(payload)}, expected {TRINITY_RESTORING_PAYLOAD_BYTES}"
        )

    shm_dir = ensure_shm_root(titan_id)
    target = shm_dir / TRINITY_RESTORING_SIDECAR

    # Atomic write: write to tmp in the same directory + os.replace().
    # Same-FS rename = atomic; daemons reading the path never see a partial file.
    fd, tmp_name = tempfile.mkstemp(prefix=".trinity_restoring.", dir=str(shm_dir))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(target))
    except Exception:
        # Cleanup the tmp file on any error path.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    logger.info(
        "[trinity_restoring_publisher] wrote %s (%d bytes): "
        "k_drive=%.3f k_restore=%.3f k_damp=%.3f k_mom=%.3f k_dir=%.3f "
        "a_mag=%.3f a_drift=%.3f a_dmag=%.3f",
        target,
        TRINITY_RESTORING_PAYLOAD_BYTES,
        gains["k_drive"],
        gains["k_restore"],
        gains["k_damp"],
        gains["k_mom"],
        gains["k_dir"],
        gains["a_mag"],
        gains["a_drift"],
        gains["a_dmag"],
    )
    return target
