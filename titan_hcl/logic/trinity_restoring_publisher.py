"""
trinity_restoring_publisher — write the §G5.2 restoring-force gain sidecar.

Per SPEC §G5.2 item 5 + PLAN_trinity_homeostasis_p0 §1.5 / §6.6.2 + D-SPEC-112
+ D-SPEC-129 (per-layer extension, v1.59.2, 2026-05-25): the gain coefficients
for the §G5.2 stateful traveling-tensor integrator live in
``titan_hcl/titan_params.toml`` ``[trinity_restoring.{body,mind,spirit}]``
(per-Titan, per-layer tunable). Inner+outer of a layer share one tuning unit
(matches Rust `Layer` enum {Body, Mind, Spirit}).

Rather than parse TOML inside the Rust daemons (adds a runtime dep +
path-discovery), this module reads the three subsections via the canonical
``titan_hcl.params.get_params`` 4-layer merge and writes a fixed-layout SHM
sidecar ``trinity_restoring.bin`` (3 layers × 8 floats × 4 bytes LE = 96
bytes) under the per-Titan ``shm_dir``. Each of the 6 trinity daemons
(inner/outer × body/mind/spirit) retry-loads it at boot and refreshes at ~1 s
cadence — see Rust ``titan_trinity_daemon::restoring_cfg::load_for_layer``.

Layer gradient (INV-9 body 0.7/0.3, mind 0.5/0.5, spirit 0.3/0.7) is STRUCTURAL
— fixed in the Rust kernel, NOT a TOML key. Neuromod gain is LIVE — read from
``neuromod_state.bin`` per tick, NOT a TOML key.

Layout (must match Rust ``restoring_cfg.rs``):

  bytes [ 0:32) — body  (8 × f32 LE)
  bytes [32:64) — mind  (8 × f32 LE)
  bytes [64:96) — spirit(8 × f32 LE)

Per-layer field order:

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
TRINITY_RESTORING_PER_LAYER_BYTES = 32   # 8 × float32 LE per layer
TRINITY_RESTORING_PAYLOAD_BYTES = 96     # 3 layers × 32 bytes

# Layer order in the sidecar — body first, mind, spirit. Matches Rust
# `Layer` enum discriminant order and `load_for_layer` offset table.
_LAYERS = ("body", "mind", "spirit")

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


def _resolve_layer_gains(layer: str) -> dict[str, float]:
    """Merge ``[trinity_restoring.<layer>]`` over crate-defaults, 8 named floats.

    Missing whole-subsection → all defaults (warned at info; substrate
    continues). Missing single key → that one defaults (warned at info).
    Non-numeric value → that one defaults (warned at warn — surface bad data).
    """
    cfg = get_params(f"trinity_restoring.{layer}") or {}
    gains: dict[str, float] = {}
    for key, default in _DEFAULTS.items():
        raw = cfg.get(key, default)
        try:
            gains[key] = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "[trinity_restoring_publisher] %s.%s=%r non-numeric — falling back to default %.3f",
                layer, key, raw, default,
            )
            gains[key] = default
    return gains


def publish_trinity_restoring_cfg(titan_id: Optional[str] = None) -> Path:
    """Resolve per-layer gains from ``titan_params.toml`` and write the sidecar.

    Returns the absolute path of the written file. Raises on I/O failure
    (caller surfaces — diagnosability over silence per
    ``directive_error_visibility``).
    """
    per_layer = {layer: _resolve_layer_gains(layer) for layer in _LAYERS}

    # Pack body → mind → spirit, each 8 × f32 LE = 32 bytes.
    payload = b"".join(
        struct.pack("<f", per_layer[layer][k])
        for layer in _LAYERS
        for k in _FIELD_ORDER
    )
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
        "[trinity_restoring_publisher] wrote %s (%d bytes) — per-layer k_drive: "
        "body=%.3f mind=%.3f spirit=%.3f (k_restore: body=%.3f mind=%.3f spirit=%.3f)",
        target,
        TRINITY_RESTORING_PAYLOAD_BYTES,
        per_layer["body"]["k_drive"],
        per_layer["mind"]["k_drive"],
        per_layer["spirit"]["k_drive"],
        per_layer["body"]["k_restore"],
        per_layer["mind"]["k_restore"],
        per_layer["spirit"]["k_restore"],
    )
    return target
