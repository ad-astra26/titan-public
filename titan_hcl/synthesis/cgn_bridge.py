"""CGN spine-concept registration bridge (Phase 4 / §P4.C).

The synthesis-engine spine concept namespace (e.g. "metaplex_nft_minting",
"cosmetic_business_website") is **distinct from** CGN's per-word concept
namespace. Per the 2026-05-20 design conversation and arch §3.2 + INV-1:

  * CGN owns the **meaning oracle** — felt-state / grounding / per-word
    concept lifecycle. It's the sole grounding authority.
  * The spine namespace is owned by `synthesis_worker` and lives in this
    module's registry. Spine concepts AGGREGATE multiple CGN word-concepts
    via the spine's composition graph + the MeaningOraclePlug interface.

This module is the thin bridge that:

1. Registers a spine concept_id when ConceptStore materializes it (P4.B).
   Registration writes to `data/synthesis_spine_concepts.json` (atomic
   tmp+rename) — a small registry distinct from CGN's vocabulary, so
   neither namespace pollutes the other.
2. Provides `ensure_grounded()` — a stub in P4 that returns None. The
   inner-outer bridge phase (Phase 7+) deepens this into a real
   call into CGN's meaning_of() / ground() surface.

P4 ships the **interface**; the deep CGN integration lands incrementally
per the "interface-complete, implementation-incremental" §3.3 discipline.

Soft-fail semantics: any persistence error logs WARN but never raises so
ConceptStore.create_concept can still anchor its TX (the spine row is a
derived index, not the canonical record — INV-2).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "data/synthesis_spine_concepts.json"


@dataclass(frozen=True)
class Grounding:
    """Minimal P4 stub of arch §3.5 Grounding dataclass. The full shape
    (felt-state binding, CGN node refs, confidence) lands when the bridge
    phase deepens this surface."""

    concept_id: str
    version: int
    grounded: bool
    note: str = ""


class CGNRegistrationBridge:
    """Thin bridge from spine concept materialization → registry persistence.

    INV-1: CGN remains sole grounding authority; this only ASKS the registry
    to take ownership of a new spine concept_id (idempotent if already
    registered) and offers `ensure_grounded()` as the future CGN grounding
    hook. P4 implementation does NOT mutate the CGN vocabulary table or
    create CGN groundings.
    """

    def __init__(
        self,
        registry_path: str = DEFAULT_REGISTRY_PATH,
        *,
        cgn_handle: Any = None,  # reserved for Phase 7+ deep wiring
        clock: Any = time.time,
    ):
        self._registry_path = registry_path
        self._cgn = cgn_handle
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, dict] = {}  # concept_id → record
        self._loaded = False

    # ── Registry persistence ─────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load the JSON registry into memory. Soft-fails to an empty
        registry if the file is missing or corrupt."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if os.path.exists(self._registry_path):
                    with open(self._registry_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self._cache = {
                            k: v for k, v in data.items()
                            if isinstance(v, dict)
                        }
            except Exception as e:
                logger.warning(
                    "[CGNBridge] registry load failed (%s) — starting empty",
                    e,
                )
                self._cache = {}
            self._loaded = True

    def _persist(self) -> bool:
        """Atomic write of the in-memory cache to disk. Returns True on
        success, False on persistence error (caller decides whether to
        continue with in-memory-only state)."""
        try:
            os.makedirs(
                os.path.dirname(self._registry_path) or ".", exist_ok=True,
            )
            # Atomic tmp+rename. The tmp file lives in the same directory
            # so os.replace stays inode-stable.
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self._registry_path) or ".",
                prefix=".spine_registry_", suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._cache, f, indent=2, sort_keys=True)
                os.replace(tmp, self._registry_path)
                return True
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("[CGNBridge] registry persist failed: %s", e)
            return False

    # ── Public surface ──────────────────────────────────────────

    def register_spine_concept(
        self,
        concept_id: str,
        name: str,
        seed_consumer: str = "synthesis_engine",
    ) -> bool:
        """Register a new spine concept_id in the registry. Idempotent.

        Returns:
            True  — newly registered (first sighting).
            False — already registered (no-op).

        Soft-fails on persistence error: in-memory cache still updates so
        the in-process synthesis_worker stays consistent even when the disk
        write fails; the next successful tick will re-persist.
        """
        if not concept_id:
            logger.warning(
                "[CGNBridge] register_spine_concept: empty concept_id"
            )
            return False

        self._load()
        with self._lock:
            if concept_id in self._cache:
                return False
            self._cache[concept_id] = {
                "concept_id": concept_id,
                "name": name,
                "seed_consumer": seed_consumer,
                "registered_at": float(self._clock()),
            }
            persisted = self._persist()
            if not persisted:
                logger.warning(
                    "[CGNBridge] register_spine_concept(%s) — in-memory "
                    "only; persistence retry on next op",
                    concept_id,
                )
            return True

    def is_registered(self, concept_id: str) -> bool:
        """Check whether a spine concept has been registered."""
        self._load()
        with self._lock:
            return concept_id in self._cache

    def list_registered(self) -> list[dict]:
        """Snapshot of every registered spine concept. Used by the
        observatory endpoint (§P4.I) and the fleet E2E test (§P4.J)."""
        self._load()
        with self._lock:
            return [dict(v) for v in self._cache.values()]

    def ensure_grounded(
        self, concept_id: str, version: int,
    ) -> Optional[Grounding]:
        """Ask CGN to ground the spine concept in the current felt context.

        **P4 behavior:** returns None for any concept the registry doesn't
        know about; otherwise returns a stub `Grounding(grounded=False,
        note="phase4_stub")`. The inner-outer bridge phase deepens this
        into a real CGN.meaning_of() / ensure_grounded() call cycle.

        Caller (ConceptStore.recompute_groundedness via P4.G consolidation)
        treats None as "no felt strand yet" → felt_coverage=0.0 in the
        groundedness formula (§P4.E). Same numeric result as P4's stub,
        but the structural seam is in place for the bridge phase.
        """
        self._load()
        with self._lock:
            if concept_id not in self._cache:
                return None
            return Grounding(
                concept_id=concept_id,
                version=version,
                grounded=False,
                note="phase4_stub",
            )

    # ── Reset (test-only helper) ─────────────────────────────────

    def _reset_for_tests(self) -> None:
        """Clear the in-memory cache + delete the on-disk registry file.
        ONLY for use in test fixtures — synthesis_worker must NOT call this."""
        with self._lock:
            self._cache = {}
            self._loaded = True
            try:
                if os.path.exists(self._registry_path):
                    os.unlink(self._registry_path)
            except OSError as e:
                logger.debug(
                    "[CGNBridge] _reset_for_tests unlink failed: %s", e,
                )


__all__ = (
    "CGNRegistrationBridge",
    "Grounding",
    "DEFAULT_REGISTRY_PATH",
)
