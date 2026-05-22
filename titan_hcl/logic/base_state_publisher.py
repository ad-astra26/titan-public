"""
base_state_publisher — shared scaffolding for the per-worker SHM state
publishers introduced in Phase C Sessions 1-3 of
rFP_phase_c_async_shm_consumer_migration.

A subclass implements:
  - ``_compute_payload(*args, **kwargs) -> dict[str, Any]`` — produce the
    msgpack-encodable payload for THIS slot. Always returns a dict (cold-
    boot stub if state is unavailable — never raise; let the caller decide
    if a write should happen).

The base class provides:
  - lazy writer attach + first-attach INFO log (slot path + max_bytes +
    schema_version)
  - encode → oversize-guard → write_variable pipeline
  - per-slot fail counters (encode / oversize / write) with throttled WARN +
    exc_info=True for first-occurrence diagnostics
  - first-publish-success INFO log (proves consumer-side deadlock-surface
    closure for that slot)
  - heartbeat INFO logs at canonical tick milestones (1, 10, 60, 600, 3600)
  - ``get_stats()`` introspection for test harnesses + diagnostics

This file deliberately mirrors the pattern that landed in Sessions 1+2
(SpiritStatePublisher + MemoryStatePublisher) — it does NOT replace those
existing publishers (they keep their fully-inlined shape because they
were written first and live-verified on T3 already; lifting them onto
this base class is a cleanup that can land in Session 5 §4.F.2 if we
choose).

Per Preamble G21 single-writer: each publisher instance is the SOLE
producer of its slot. Multi-instance is a SPEC violation and would
corrupt the SeqLock state.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    RegistrySpec,
    StateRegistryWriter,
    ensure_shm_root,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)


class BaseStatePublisher:
    """Single-slot SHM publisher base class.

    Subclasses MUST set:
      - ``slot_name`` (str)         — class attribute matching the SPEC §7.1 row
      - ``slot_spec`` (RegistrySpec) — single-source-of-truth from a *_state_specs
                                       module shared with the consuming proxy

    Subclasses MUST override:
      - ``_compute_payload(*args, **kwargs) -> dict[str, Any]``

    Subclasses MAY override:
      - ``_payload_for_logger() -> str`` — short identifying suffix used in
        the first-publish-success INFO log (defaults to slot_name)
    """

    # Subclass-set
    slot_name: str = ""
    slot_spec: Optional[RegistrySpec] = None

    def __init__(self, titan_id: str):
        if not self.slot_name or self.slot_spec is None:
            raise ValueError(
                f"{self.__class__.__name__} must set slot_name + slot_spec "
                f"class attributes before instantiation")
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0
        logger.info(
            "[%s] initialized — titan_id=%s shm_root=%s slot=%s "
            "(SPEC §7.1 / Preamble G18)",
            self.__class__.__name__, titan_id, self._shm_root,
            self.slot_name)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(self.slot_spec, self._shm_root)
        logger.info(
            "[%s] writer attached — slot=%s max_bytes=%d schema_version=%d "
            "path=%s",
            self.__class__.__name__, self.slot_name,
            self.slot_spec.payload_bytes, self.slot_spec.schema_version,
            self._shm_root / f"{self.slot_name}.bin")
        return self._writer

    # -- subclass hook ---------------------------------------------------

    def _compute_payload(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            "subclass must override _compute_payload")

    # -- top-level entry point ------------------------------------------

    def publish(self, *args, **kwargs) -> None:
        """Compute payload + write to SHM. Defensive against subclass-side
        compute failure — top-level try/except guards the publisher thread
        from dying on a producer bug. Never raises."""
        self._publish_count += 1
        try:
            payload = self._compute_payload(*args, **kwargs)
        except Exception as e:
            self._encode_fails += 1
            if (self._encode_fails == 1 or
                    self._encode_fails % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[%s] _compute_payload raised (#%d): %s",
                    self.__class__.__name__, self._encode_fails, e,
                    exc_info=True)
            payload = None

        if isinstance(payload, dict):
            self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[%s] heartbeat — slot=%s publish_count=%d success=%d "
                "fails={encode=%d oversize=%d write=%d}",
                self.__class__.__name__, self.slot_name,
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails)

    # -- low-level write helper -----------------------------------------

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if (self._encode_fails == 1 or
                    self._encode_fails % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[%s] msgpack encode failed (#%d): %s — keys=%s",
                    self.__class__.__name__, self._encode_fails, e,
                    sorted(payload.keys()), exc_info=True)
            return

        if len(encoded) > self.slot_spec.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[%s] payload %dB > MAX %dB (#%d) — slot retains last-known. "
                "Investigate upstream shape drift; do NOT silently truncate.",
                self.__class__.__name__, len(encoded),
                self.slot_spec.payload_bytes, self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[%s] FIRST PUBLISH SUCCESS — slot=%s payload_bytes=%d "
                    "(consumers can now read; deadlock surface closed for "
                    "this slot)",
                    self.__class__.__name__, self.slot_name, len(encoded))
        except Exception as e:
            self._write_fails += 1
            if (self._write_fails == 1 or
                    self._write_fails % _WARN_THROTTLE_EVERY == 0):
                logger.warning(
                    "[%s] shm write failed (#%d): %s",
                    self.__class__.__name__, self._write_fails, e,
                    exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "slot": self.slot_name,
            "publish_count": self._publish_count,
            "publish_success": self._publish_success,
            "encode_fails": self._encode_fails,
            "oversize_fails": self._oversize_fails,
            "write_fails": self._write_fails,
            "writer_attached": self._writer is not None,
        }


class MultiSlotStatePublisher:
    """Composes multiple BaseStatePublisher instances behind a single
    publish() entry point — used when one worker owns multiple slots
    (e.g., agency_worker owns both agency_state.bin and
    assessment_state.bin since SelfAssessment is held by AgencyModule).

    Subclasses or instantiations supply a list of (publisher, label)
    pairs; publish() invokes each with the same args/kwargs (each
    publisher's _compute_payload extracts what it needs from the args).

    Failure of one slot's publish does NOT affect the others (each is
    guarded by BaseStatePublisher's own try/except).
    """

    def __init__(self, publishers: list["BaseStatePublisher"]):
        self._publishers = list(publishers)
        logger.info(
            "[MultiSlotStatePublisher] composed %d publishers: %s",
            len(self._publishers),
            [p.slot_name for p in self._publishers])

    def publish(self, *args, **kwargs) -> None:
        for pub in self._publishers:
            pub.publish(*args, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        return {p.slot_name: p.get_stats() for p in self._publishers}
