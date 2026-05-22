"""
language_state_publisher — Phase C Session 4 §4.B.7.

Publishes language_state.bin from language_worker's
update_language_stats() output. Mirrors the LANGUAGE_STATS_UPDATE bus
event payload so consumers can read the same data via SHM (Preamble G18 —
state transport is SHM, never bus).

Schema per SPEC §7.1 language_state.bin:
    {
        vocab_total: int,
        vocab_producible: int,
        vocab_contextual: int,
        avg_confidence: float,
        max_confidence: float,
        recent_words: list[str],
        teacher_sessions_last_hour: int,
        teacher_sessions_last_day: int,
        composition_level: str,
        teacher_compositions_since: int,
        teacher_last_fire_time: float,
        ts: float,
    }
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.session4_state_specs import (
    LANGUAGE_STATE_SLOT,
    LANGUAGE_STATE_SPEC,
)


class LanguageStatePublisher(BaseStatePublisher):
    slot_name = LANGUAGE_STATE_SLOT
    slot_spec = LANGUAGE_STATE_SPEC

    def _compute_payload(
        self,
        language_stats: Any,
        teacher_compositions_since: int = 0,
        teacher_last_fire_time: float = 0.0,
    ) -> dict[str, Any]:
        """
        Args:
          language_stats — dict from language_pipeline.update_language_stats()
          teacher_compositions_since — counter from language_worker scope
          teacher_last_fire_time — last teacher trigger timestamp
        """
        if not isinstance(language_stats, dict):
            language_stats = {}

        return {
            "vocab_total": int(language_stats.get("vocab_total", 0) or 0),
            "vocab_producible": int(
                language_stats.get("vocab_producible", 0) or 0),
            "vocab_contextual": int(
                language_stats.get("vocab_contextual", 0) or 0),
            "avg_confidence": float(
                language_stats.get("avg_confidence", 0.0) or 0.0),
            "max_confidence": float(
                language_stats.get("max_confidence", 0.0) or 0.0),
            "recent_words": list(
                language_stats.get("recent_words", []) or [])[:50],
            "teacher_sessions_last_hour": int(
                language_stats.get("teacher_sessions_last_hour", 0) or 0),
            "teacher_sessions_last_day": int(
                language_stats.get("teacher_sessions_last_day", 0) or 0),
            "composition_level": str(
                language_stats.get("composition_level", "L1") or "L1"),
            "teacher_compositions_since": int(
                teacher_compositions_since or 0),
            "teacher_last_fire_time": float(teacher_last_fire_time or 0.0),
            "ts": time.time(),
        }
