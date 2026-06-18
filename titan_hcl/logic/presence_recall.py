"""Presence recall — RFP_verifiable_autobiographical_presence_memory §7.D (RECALL).

Before the LLM speaks, resolve the turn's person → their last verified presence →
the gap in Titan-time epochs → a `VerifiedRecord`-shaped fact-bundle that Phase E
injects into the grounded context. The LLM only NARRATES this bundle (INV-PAM-
NARRATE-ONLY); the symbolic layer is the truth.

Read surface (G18/G19-clean): the agno PreHook is the `/chat` hot path and CANNOT
open synthesis's DuckDB (exclusive lock) nor make a sync state-RPC. So synthesis
exports `data/titan_time/presence_recall_snapshot.json` at each fold/seal
(`AutobiographySeal.export_recall_snapshot`) and this reads that JSON — the same
lock-free pattern as `spine_snapshot.json`. (The drafting-conversation idea of
`EngineRecall.recall(granularity="autobiographical")` was DISPROVEN at Phase-D
build: `recall.py` granularities are {turn, topic, session} only — no such
granularity exists.)

Titan-time: the gap is computed in epochs; human time appears ONLY in `gap_human`,
produced by `TitanTimeTranslator` at the narration edge (INV-PAM-TITAN-TIME, G5).
Honesty: `evidence_strength` (crypto_verified_maker › device › asserted) and
`chain_status` (CHAINED › WIRED › UNSEALED) are carried end-to-end so Phase E
never overstates certainty (INV-PAM-HONEST-GRADIENT). A person absent from the
snapshot → `None` (no prior verified presence → no recognition; honest, never a
fabricated one — INV-PAM-SOVEREIGN-BOUNDED).

Scope (§7.D): produces the fact-bundle ONLY — no context injection, no OVG (Phase
E). No Maker-readable endpoint (the snapshot is Titan-internal; INV-PAM-NO-BACKDOOR).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_SAVE_DIR = "data/titan_time"
SNAPSHOT_NAME = "presence_recall_snapshot.json"


class PresenceRecall:
    """Reads the lock-free presence snapshot → a per-person verified fact-bundle."""

    def __init__(
        self,
        *,
        snapshot_path: Optional[str] = None,
        save_dir: str = DEFAULT_SAVE_DIR,
        translator: Any = None,
        age_reader: Any = None,
        titan_id: Optional[str] = None,
    ) -> None:
        self._path = snapshot_path or os.path.join(save_dir, SNAPSHOT_NAME)
        # Lazy: the translator + age reader are only needed when a person is found,
        # and constructing them touches the SHM age slot — defer to first recall.
        self._translator = translator
        self._age_reader = age_reader
        self._save_dir = save_dir
        self._titan_id = titan_id

    def _get_translator(self):
        if self._translator is None:
            from titan_hcl.logic.titan_time import TitanTimeTranslator
            self._translator = TitanTimeTranslator(
                save_dir=self._save_dir, titan_id=self._titan_id,
                age_reader=self._age_reader)
        return self._translator

    def _get_age_reader(self):
        if self._age_reader is None:
            from titan_hcl.logic.consciousness_age_reader import ConsciousnessAgeReader
            self._age_reader = ConsciousnessAgeReader(titan_id=self._titan_id)
        return self._age_reader

    def _read_snapshot(self) -> dict:
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return data.get("persons", {}) if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as e:  # noqa: BLE001 — corrupt/partial → no recognition (honest)
            logger.debug("[presence_recall] snapshot unreadable (%s)", e)
            return {}

    def recall(self, person_id: str, now_age_epochs: Optional[int] = None) -> Optional[dict]:
        """Resolve `person_id` → a verified presence fact-bundle, or `None` if no
        prior verified presence is anchored for them (honest non-recognition).

        Returns: `{person_id, gap_epochs, gap_human, evidence_strength, chain_status,
        anchor, last_seen_epoch}`. `gap_epochs` is Titan-time; `gap_human` is the
        only human-time value (translated at this narration edge)."""
        if not person_id:
            return None
        rec = self._read_snapshot().get(person_id)
        if not rec:
            return None  # no anchored prior presence → no recognition (INV-PAM-SOVEREIGN-BOUNDED)
        try:
            last_seen = int(rec["last_seen_epoch"])
        except (KeyError, TypeError, ValueError):
            return None
        if now_age_epochs is None:
            now_age_epochs = int(self._get_age_reader().get_age_epochs())
        gap_epochs = max(0, int(now_age_epochs) - last_seen)
        gap_human = self._get_translator().to_human(gap_epochs)
        return {
            "person_id": person_id,
            "gap_epochs": gap_epochs,
            "gap_human": gap_human,
            "evidence_strength": rec.get("evidence_strength", "asserted_identity"),
            "chain_status": rec.get("chain_status", "UNSEALED"),
            "anchor": rec.get("anchor"),
            "last_seen_epoch": last_seen,
        }


__all__ = ("PresenceRecall", "SNAPSHOT_NAME", "DEFAULT_SAVE_DIR")
