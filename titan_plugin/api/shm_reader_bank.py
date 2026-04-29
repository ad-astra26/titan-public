"""
titan_plugin/api/shm_reader_bank.py — typed shm-registry readers for the
api_subprocess.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

Wraps `titan_plugin.core.state_registry.StateRegistryReader` with per-
registry typed accessors. Each method returns a structured dict (or None
if the registry is unavailable/disabled), including `age_seconds` for
freshness checks.

The api_subprocess uses these readers to serve endpoint state without
issuing any RPC to the kernel. The mmap handles are owned per-reader
and never closed; reads are zero-copy + defensive-copy + SeqLock-
validated.

Design notes:
- All methods are sync (no async). Reads are sub-microsecond.
- Missing/disabled registries return None — endpoint code should fall
  back to bus-cached values (set by BusSubscriber).
- The bank is constructed once at api_subprocess boot; readers attach
  lazily on first read.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from titan_plugin.core.state_registry import (
    CHI_STATE,
    EPOCH_COUNTER,
    IDENTITY,
    INNER_SPIRIT_45D,
    NEUROMOD_STATE,
    SPHERE_CLOCKS_STATE,
    StateRegistryReader,
    TITANVM_REGISTERS,
    TRINITY_STATE,
    resolve_shm_root,
    resolve_titan_id,
)

logger = logging.getLogger(__name__)


# ── Schema labels ─────────────────────────────────────────────────────
# Index → name lookup so Python-side reads return labelled dicts instead
# of raw arrays. Endpoint code stays readable (no magic offsets).

NEUROMOD_NAMES = ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]

CHI_FIELD_NAMES = ["total", "spirit", "mind", "body", "coherence", "urgency"]

SPHERE_CLOCK_NAMES = [
    "inner_body", "inner_mind", "inner_spirit",
    "outer_body", "outer_mind", "outer_spirit",
]
SPHERE_CLOCK_FIELDS = [
    "radius", "scalar_position", "phase", "contraction_velocity",
    "pulse_count", "consecutive_balanced", "last_pulse_age_s",
]

NS_PROGRAM_NAMES = [
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",
]
NS_PROGRAM_FIELDS = ["urgency", "fire_count", "total_updates", "last_loss"]

# 45D inner spirit slot labels (per spirit_tensor.collect_spirit_45d).
INNER_SPIRIT_GROUPS = {
    "SAT":    slice(0, 15),
    "CHIT":   slice(15, 30),
    "ANANDA": slice(30, 45),
}

# 162D Trinity layout (per state_registry.py:606-609).
TRINITY_GROUPS = {
    "full_130dt":         slice(0, 130),
    "full_30d_topology":  slice(130, 160),
    "journey":            slice(160, 162),  # (curvature, density)
}


# ── Helper: read metadata + payload, return structured dict ───────────

def _payload_with_meta(
    reader: StateRegistryReader,
    name: str,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """Returns (payload, age_seconds, seq) — None tuple on read failure."""
    payload = reader.read()
    if payload is None:
        return (None, None, None)
    meta = reader.read_meta()
    if meta is None:
        # Should not happen if read() succeeded, but defensive.
        return (payload, None, None)
    return (payload, meta.get("age_seconds"), meta.get("seq"))


# ── Bank ──────────────────────────────────────────────────────────────

class ShmReaderBank:
    """Aggregate owner of per-registry readers.

    Constructed once at api_subprocess boot. Each registry has its own
    reader; readers attach lazily on first read.

    All methods return a structured dict with `age_seconds` and `seq`,
    or None if the registry is missing/disabled/torn.
    """

    __slots__ = (
        "titan_id", "shm_root",
        "_trinity", "_neuromod", "_epoch", "_inner_spirit",
        "_sphere_clocks", "_chi", "_titanvm", "_identity",
    )

    def __init__(self, titan_id: str | None = None) -> None:
        self.titan_id = resolve_titan_id(titan_id)
        self.shm_root: Path = resolve_shm_root(self.titan_id)
        # Lazy readers — attach on first read.
        self._trinity = StateRegistryReader(TRINITY_STATE, self.shm_root)
        self._neuromod = StateRegistryReader(NEUROMOD_STATE, self.shm_root)
        self._epoch = StateRegistryReader(EPOCH_COUNTER, self.shm_root)
        self._inner_spirit = StateRegistryReader(INNER_SPIRIT_45D, self.shm_root)
        self._sphere_clocks = StateRegistryReader(SPHERE_CLOCKS_STATE, self.shm_root)
        self._chi = StateRegistryReader(CHI_STATE, self.shm_root)
        self._titanvm = StateRegistryReader(TITANVM_REGISTERS, self.shm_root)
        self._identity = StateRegistryReader(IDENTITY, self.shm_root)
        logger.info(
            "[ShmReaderBank] initialized for titan_id=%s root=%s",
            self.titan_id, self.shm_root,
        )

    # -- Trinity (162D) ------------------------------------------------

    def read_trinity(self) -> dict[str, Any] | None:
        """Return Trinity state with subgroups + metadata, or None."""
        payload, age, seq = _payload_with_meta(self._trinity, "trinity_state")
        if payload is None:
            return None
        return {
            "full_130dt": payload[TRINITY_GROUPS["full_130dt"]].tolist(),
            "full_30d_topology": payload[TRINITY_GROUPS["full_30d_topology"]].tolist(),
            "journey": {
                "curvature": float(payload[160]),
                "density": float(payload[161]),
            },
            "age_seconds": age,
            "seq": seq,
        }

    # -- Neuromodulators (6) -------------------------------------------

    def read_neuromod(self) -> dict[str, Any] | None:
        """Return neuromodulator levels by name + metadata, or None."""
        payload, age, seq = _payload_with_meta(self._neuromod, "neuromod_state")
        if payload is None:
            return None
        return {
            "modulators": {
                name: {"level": float(payload[i])}
                for i, name in enumerate(NEUROMOD_NAMES)
            },
            "age_seconds": age,
            "seq": seq,
        }

    # -- Epoch counter -------------------------------------------------

    def read_epoch(self) -> dict[str, Any] | None:
        """Return current consciousness epoch counter + metadata, or None."""
        payload, age, seq = _payload_with_meta(self._epoch, "epoch_counter")
        if payload is None:
            return None
        return {
            "epoch": int(payload[0]),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Inner Spirit 45D ---------------------------------------------

    def read_inner_spirit_45d(self) -> dict[str, Any] | None:
        """Return SAT/CHIT/ANANDA groups + metadata, or None.

        S3b shm-spirit-fast — written at 70.47 Hz when flag enabled.
        """
        payload, age, seq = _payload_with_meta(
            self._inner_spirit, "inner_spirit_45d")
        if payload is None:
            return None
        return {
            "SAT": payload[INNER_SPIRIT_GROUPS["SAT"]].tolist(),
            "CHIT": payload[INNER_SPIRIT_GROUPS["CHIT"]].tolist(),
            "ANANDA": payload[INNER_SPIRIT_GROUPS["ANANDA"]].tolist(),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Sphere clocks (S4) -------------------------------------------

    def read_sphere_clocks(self) -> dict[str, Any] | None:
        """Return per-clock state by name + metadata, or None.

        S4 — flag-gated by `microkernel.shm_sphere_clocks_enabled`.
        """
        payload, age, seq = _payload_with_meta(
            self._sphere_clocks, "sphere_clocks")
        if payload is None:
            return None
        clocks = {}
        for i, name in enumerate(SPHERE_CLOCK_NAMES):
            row = payload[i]
            clocks[name] = {
                field: float(row[j])
                for j, field in enumerate(SPHERE_CLOCK_FIELDS)
            }
        return {
            "clocks": clocks,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Chi state (S4) ------------------------------------------------

    def read_chi(self) -> dict[str, Any] | None:
        """Return chi circulation fields by name + metadata, or None.

        S4 — flag-gated by `microkernel.shm_chi_enabled`.
        """
        payload, age, seq = _payload_with_meta(self._chi, "chi_state")
        if payload is None:
            return None
        return {
            **{name: float(payload[i]) for i, name in enumerate(CHI_FIELD_NAMES)},
            "age_seconds": age,
            "seq": seq,
        }

    # -- TitanVM registers (S4) ---------------------------------------

    def read_titanvm_registers(self) -> dict[str, Any] | None:
        """Return per-NS-program register state + metadata, or None.

        S4 — flag-gated by `microkernel.shm_titanvm_enabled`.
        """
        payload, age, seq = _payload_with_meta(self._titanvm, "titanvm_registers")
        if payload is None:
            return None
        programs = {}
        for i, name in enumerate(NS_PROGRAM_NAMES):
            row = payload[i]
            programs[name] = {
                field: float(row[j])
                for j, field in enumerate(NS_PROGRAM_FIELDS)
            }
        return {
            "programs": programs,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Identity (S4) -------------------------------------------------

    def read_identity(self) -> dict[str, Any] | None:
        """Return titan_id + maker_pubkey + kernel_instance_nonce, or None.

        S4 — flag-gated by `microkernel.shm_identity_enabled`. Until
        flag flipped, falls back to None and endpoint code reads from
        bus-cached "identity.*" keys.
        """
        payload, age, seq = _payload_with_meta(self._identity, "identity")
        if payload is None:
            return None
        # Decode per IDENTITY layout (state_registry.py:740-744):
        #   [0:32]   titan_id (UTF-8, NUL-padded)
        #   [32:64]  maker_pubkey (raw 32-byte Ed25519, zero if absent)
        #   [64:96]  kernel_instance_nonce (random per boot)
        titan_id_bytes = bytes(payload[0:32]).rstrip(b"\x00")
        maker_pubkey_bytes = bytes(payload[32:64])
        nonce_bytes = bytes(payload[64:96])
        return {
            "titan_id": titan_id_bytes.decode("utf-8", errors="replace"),
            "maker_pubkey": maker_pubkey_bytes.hex() if any(maker_pubkey_bytes) else "",
            "kernel_instance_nonce": nonce_bytes.hex(),
            "age_seconds": age,
            "seq": seq,
        }

    # -- Diagnostic ----------------------------------------------------

    def availability_report(self) -> dict[str, bool]:
        """Per-registry attached/readable status — used by /v4/api-status."""
        report = {}
        for name, reader in [
            ("trinity", self._trinity),
            ("neuromod", self._neuromod),
            ("epoch", self._epoch),
            ("inner_spirit_45d", self._inner_spirit),
            ("sphere_clocks", self._sphere_clocks),
            ("chi", self._chi),
            ("titanvm_registers", self._titanvm),
            ("identity", self._identity),
        ]:
            meta = reader.read_meta()
            report[name] = meta is not None
        return report
