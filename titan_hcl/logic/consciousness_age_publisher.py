"""
consciousness_age_publisher — ConsciousnessAgePublisher writes
consciousness_age.bin SHM slot.

Producer for the consciousness_age slot per SPEC §7.1 (D-SPEC-85 v1.25.0).
G21 single-writer contract: only cognitive_worker publishes here
(Consciousness object lives in spirit_loop under cognitive_worker per
SPEC §1 glossary).

Closes the post_dispatch gap surfaced 2026-05-18: the social_worker
subprocess (which builds PostContext for X posts) cannot reach
consciousness.db per G18 (state transport is SHM, never DB). Without
this slot the footer's "age" field had no canonical source — the
unified_spirit_metadata.epoch_count GreatEpoch counter (~1,611 on T1)
was being used as proxy, which confused readers because Titan's
actual self-observation count is ~1M+ (the fast cognitive epoch tick
counter at ~10s per tick lifetime).
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.consciousness_age_state_specs import (
    CONSCIOUSNESS_AGE_SLOT,
    CONSCIOUSNESS_AGE_SPEC,
)
from titan_hcl._phase_c_constants import CONSCIOUSNESS_AGE_SCHEMA_VERSION


class ConsciousnessAgePublisher(BaseStatePublisher):
    slot_name = CONSCIOUSNESS_AGE_SLOT
    slot_spec = CONSCIOUSNESS_AGE_SPEC

    def _compute_payload(self, consciousness: Any) -> dict[str, Any]:
        """Read lifetime epoch count from the consciousness DB.

        cognitive_worker stores `consciousness` in state_refs as a DICT
        (per spirit_loop._init_consciousness:1200-1207) carrying
        ``{"db": ConsciousnessDB, "topology": JourneyTopology, ...}``.
        The actual epoch counter lives on ``consciousness["db"]`` via
        ``get_epoch_count()`` (sqlite row count on ``epochs`` table).

        Live evidence on T2 (2026-05-18): `consciousness.db` table holds
        859,183 rows; calling `db.get_epoch_count()` returns that count.
        Defensive: tolerates either the dict-shape (canonical) or a
        Consciousness-like object that exposes ``get_epoch_count`` directly.
        """
        if consciousness is None:
            return self._stub()
        age_epochs = 0
        try:
            # Canonical dict-shape from _init_consciousness.
            if isinstance(consciousness, dict):
                db = consciousness.get("db")
                if db is not None:
                    getter = getattr(db, "get_epoch_count", None)
                    if callable(getter):
                        age_epochs = int(getter() or 0)
            else:
                # Defensive fallback: object that itself exposes get_epoch_count.
                getter = getattr(consciousness, "get_epoch_count", None)
                if callable(getter):
                    age_epochs = int(getter() or 0)
        except Exception:
            age_epochs = 0
        return {
            "age_epochs": age_epochs,
            "schema_version": CONSCIOUSNESS_AGE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "age_epochs": 0,
            "schema_version": CONSCIOUSNESS_AGE_SCHEMA_VERSION,
            "ts": time.time(),
        }
