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
    HORMONAL_STATE,
    IDENTITY,
    INNER_BODY_5D,
    INNER_MIND_15D,
    INNER_SPIRIT_45D,
    NEUROMOD_STATE,
    OUTER_BODY_5D,
    OUTER_MIND_15D,
    OUTER_SPIRIT_45D,
    SPHERE_CLOCKS_STATE,
    StateRegistryReader,
    TITANVM_REGISTERS,
    TOPOLOGY_30D,
    TRINITY_STATE,
    resolve_shm_root,
    resolve_titan_id,
)
from titan_plugin.logic.session4_state_specs import (
    SPIRIT_SUPPLEMENTAL_STATE_SPEC,
)
from titan_plugin.logic.spirit_state_specs import (
    RESONANCE_STATE_SPEC,
    UNIFIED_SPIRIT_METADATA_SPEC,
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

# 11-hormone schema (HormonalSystem per C-S5). Each row = (level, target,
# acceleration, decay) per hormone. Names mirror HormonalSystem.hormone_names.
HORMONE_NAMES = [
    "cortisol", "adrenaline", "oxytocin", "dopamine", "serotonin",
    "endorphin", "melatonin", "growth_hormone", "insulin",
    "thyroid", "estrogen",
]
HORMONE_FIELDS = ["level", "target", "acceleration", "decay"]

# 30D space-topology layout (per state_register.py docstring §rFP #1).
# Currently published as a flat 30D vector by the writer; consumers can
# reshape (6 parts × 5 observables) post-read if they need labelled access.
TOPOLOGY_PART_NAMES = [
    "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
]
TOPOLOGY_FIELDS = ["coherence", "magnitude", "velocity", "direction", "polarity"]

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
        # chunk 8M.4 (2026-05-05): cognitive_worker shm-direct-read bank
        # extended to all SPEC §1096 slots so the cognitive epoch driver
        # can populate /v4/inner-trinity / /v4/sphere-clocks / etc.
        "_topology_30d", "_hormonal",
        "_inner_body", "_inner_mind",
        "_outer_body", "_outer_mind", "_outer_spirit",
        # rFP_worker_broadcast_topics_completion §4.D abstraction-completion
        # (2026-05-10): variable-size msgpack-encoded composite slots that
        # SpiritAccessor's get_nervous_system / get_resonance /
        # get_unified_spirit need for SHM-first migration (matching the
        # get_sphere_clocks pattern at state_accessor.py:242-247).
        "_spirit_supplemental", "_resonance_state", "_unified_spirit_metadata",
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
        # chunk 8M.4 — additional SPEC §1096 cognitive_worker readers.
        self._topology_30d = StateRegistryReader(TOPOLOGY_30D, self.shm_root)
        self._hormonal = StateRegistryReader(HORMONAL_STATE, self.shm_root)
        self._inner_body = StateRegistryReader(INNER_BODY_5D, self.shm_root)
        self._inner_mind = StateRegistryReader(INNER_MIND_15D, self.shm_root)
        self._outer_body = StateRegistryReader(OUTER_BODY_5D, self.shm_root)
        self._outer_mind = StateRegistryReader(OUTER_MIND_15D, self.shm_root)
        self._outer_spirit = StateRegistryReader(OUTER_SPIRIT_45D, self.shm_root)
        # rFP_worker_broadcast_topics_completion §4.D — variable-size
        # msgpack-encoded composite slots (SHM-first migration for
        # SpiritAccessor.get_nervous_system / get_resonance /
        # get_unified_spirit, completing the abstraction get_sphere_clocks
        # has used since chunk 8M.4).
        self._spirit_supplemental = StateRegistryReader(
            SPIRIT_SUPPLEMENTAL_STATE_SPEC, self.shm_root)
        self._resonance_state = StateRegistryReader(
            RESONANCE_STATE_SPEC, self.shm_root)
        self._unified_spirit_metadata = StateRegistryReader(
            UNIFIED_SPIRIT_METADATA_SPEC, self.shm_root)
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

    # -- Topology 30D (chunk 8M.4) ------------------------------------

    def read_topology_30d(self) -> dict[str, Any] | None:
        """Return 30D space-topology vector + structured 6×5 view + meta.

        Written by titan-trinity-rs per SPEC §9.A. Cognitive_worker reads
        per epoch and injects into coordinator.topology snapshot.
        """
        payload, age, seq = _payload_with_meta(self._topology_30d, "topology_30d")
        if payload is None:
            return None
        flat = payload.tolist()
        parts: dict[str, dict[str, float]] = {}
        for i, name in enumerate(TOPOLOGY_PART_NAMES):
            base = i * 5
            parts[name] = {
                field: float(flat[base + j])
                for j, field in enumerate(TOPOLOGY_FIELDS)
            }
        return {
            "values": flat,
            "parts": parts,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Hormonal (11×4) (chunk 8M.4) ----------------------------------

    def read_hormonal(self) -> dict[str, Any] | None:
        """Return per-hormone 4-field state by name + metadata, or None.

        Written by hormonal_worker (registered in chunk 8M.1) via
        HORMONAL_STATE shm slot. Read by cognitive_worker per epoch +
        injected into coordinator.hormonal snapshot.
        """
        payload, age, seq = _payload_with_meta(self._hormonal, "hormonal_state")
        if payload is None:
            return None
        hormones = {}
        for i, name in enumerate(HORMONE_NAMES):
            row = payload[i]
            hormones[name] = {
                field: float(row[j])
                for j, field in enumerate(HORMONE_FIELDS)
            }
        return {
            "hormones": hormones,
            "age_seconds": age,
            "seq": seq,
        }

    # -- Inner / Outer trinity tensors (5/15/45) (chunk 8M.4) ---------

    def read_inner_body_5d(self) -> dict[str, Any] | None:
        """5D inner-body tensor — written by titan-inner-body-rs."""
        payload, age, seq = _payload_with_meta(self._inner_body, "inner_body_5d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_inner_mind_15d(self) -> dict[str, Any] | None:
        """15D inner-mind tensor — written by titan-inner-mind-rs."""
        payload, age, seq = _payload_with_meta(self._inner_mind, "inner_mind_15d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_body_5d(self) -> dict[str, Any] | None:
        """5D outer-body tensor — written by titan-outer-body-rs."""
        payload, age, seq = _payload_with_meta(self._outer_body, "outer_body_5d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_mind_15d(self) -> dict[str, Any] | None:
        """15D outer-mind tensor — written by titan-outer-mind-rs."""
        payload, age, seq = _payload_with_meta(self._outer_mind, "outer_mind_15d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    def read_outer_spirit_45d(self) -> dict[str, Any] | None:
        """45D outer-spirit tensor — written by titan-outer-spirit-rs."""
        payload, age, seq = _payload_with_meta(
            self._outer_spirit, "outer_spirit_45d")
        if payload is None:
            return None
        return {"values": payload.tolist(), "age_seconds": age, "seq": seq}

    # -- Variable-size composite slots (rFP_worker_broadcast_topics §4.D) ---
    #
    # spirit_supplemental_state.bin / resonance_state.bin /
    # unified_spirit_metadata.bin are msgpack-encoded variable-size slots
    # produced by SpiritStatePublisher + SpiritSupplementalStatePublisher
    # in spirit_loop's snapshot-builder threads. Read via reader.read_variable()
    # (NOT read() which is fixed-size), then msgpack-decode.
    #
    # Closes the bug surfaced 2026-05-10: SpiritAccessor.get_nervous_system /
    # get_resonance / get_unified_spirit returned `{}` because they read
    # bus-cache only (`self._cache.get("spirit.X", {}) or {}`) — and on T3
    # under l0_rust_enabled=true, spirit_worker is heartbeat-only so the
    # cache key is never populated. The publisher writes SHM correctly
    # (verified: spirit_supplemental_state.bin has 53,934-byte payload with
    # 4 sections including nervous_system with 10 keys); the api_subprocess
    # accessor just wasn't reading SHM. This completes the abstraction
    # get_sphere_clocks has used since chunk 8M.4 (state_accessor.py:242-247).

    def _read_msgpack_variable(
        self,
        reader: StateRegistryReader,
        slot_name: str,
    ) -> dict[str, Any] | None:
        """Read variable-size SHM slot, msgpack-decode, return dict.

        Returns None on cold-boot / missing / torn / decode-failure.
        Mirrors the proven pattern in titan_plugin/proxies/spirit_proxy.py:
        _read_msgpack — same byte-format contract, same failure modes.
        """
        try:
            raw = reader.read_variable()
        except Exception as e:
            logger.warning(
                "[ShmReaderBank] %s read_variable raised: %s",
                slot_name, e, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            import msgpack
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            logger.warning(
                "[ShmReaderBank] %s msgpack decode failed: %s",
                slot_name, e)
            return None
        if not isinstance(decoded, dict):
            logger.warning(
                "[ShmReaderBank] %s decoded to non-dict: %s",
                slot_name, type(decoded).__name__)
            return None
        return decoded

    def read_spirit_supplemental(self) -> dict[str, Any] | None:
        """Return full spirit_supplemental_state dict (4 sections:
        filter_down_status / meditation_health / coordinator / nervous_system
        + ts), or None if SHM unavailable. Per Session 4 §4.C.1 expansion
        producer in spirit_loop snapshot-builder.

        Caller extracts the section it needs (e.g., for /v4/nervous-system,
        SpiritAccessor.get_nervous_system reads `dict["nervous_system"]`).
        """
        return self._read_msgpack_variable(
            self._spirit_supplemental, "spirit_supplemental_state")

    def read_resonance_state(self) -> dict[str, Any] | None:
        """Return ResonanceDetector.get_stats() output as dict, or None
        if SHM unavailable. Producer: SpiritStatePublisher._publish_resonance_state."""
        return self._read_msgpack_variable(
            self._resonance_state, "resonance_state")

    def read_unified_spirit_metadata(self) -> dict[str, Any] | None:
        """Return UnifiedSpirit.get_stats() output as dict (velocity,
        stale, focus_multiplier, etc.), or None if SHM unavailable.
        Producer: SpiritStatePublisher._publish_unified_spirit_metadata."""
        return self._read_msgpack_variable(
            self._unified_spirit_metadata, "unified_spirit_metadata")

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
            # chunk 8M.4 additions
            ("topology_30d", self._topology_30d),
            ("hormonal", self._hormonal),
            ("inner_body_5d", self._inner_body),
            ("inner_mind_15d", self._inner_mind),
            ("outer_body_5d", self._outer_body),
            ("outer_mind_15d", self._outer_mind),
            ("outer_spirit_45d", self._outer_spirit),
        ]:
            meta = reader.read_meta()
            report[name] = meta is not None
        return report
