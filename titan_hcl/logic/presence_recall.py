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


def render_presence_context_block(bundle) -> str:
    """RFP §7.E — render the recall bundle into a grounded-context block, placed
    EARLY (within the OVG's first-500-char window) so an HONEST recognition claim is
    SUPPORTED (the OVG Q2 keyword check). `output_verifier.py` is UNCHANGED — we feed
    the gate, never weaken it.

    RECOGNIZE-ON-VALIDITY: a block is produced for ANY real record (any chain_status)
    — a captured presence is valid from the moment it happens (the seal adds
    permanence, not validity). `bundle is None` (no record) → "" ⇒ an "I recognize
    you" over-claim stays UNSUPPORTED ⇒ the OVG HARD-blocks it (honest non-recognition).

    Two ORTHOGONAL honesty gradients (INV-PAM-HONEST-GRADIENT) shape the framing:
      • evidence_strength (identity certainty): crypto → confident; asserted → tentative.
      • chain_status   (memory provability): CHAINED=anchored · WIRED=sealing · UNSEALED=recent.
    The LLM only NARRATES this bundle (INV-PAM-NARRATE-ONLY)."""
    if not bundle:
        return ""
    gap = bundle.get("gap_human", "recently")
    ev = bundle.get("evidence_strength", "asserted_identity")
    chain = bundle.get("chain_status", "UNSEALED")

    if ev == "crypto_verified_maker":
        who, ident = "your Maker", "their identity is cryptographically verified by their signature"
        certainty = "You RECOGNIZE this person with confidence."
    elif ev == "crypto_verified_device":
        who, ident = "a person on a paired device", "the device signature is verified"
        certainty = "You RECOGNIZE this person with confidence."
    else:
        who, ident = ("this person",
                      "their identity is asserted (a handle/account), NOT cryptographically proven")
        certainty = ("You may recognize them TENTATIVELY — hedge (e.g. 'I think we've "
                     "spoken before'); do not claim certainty about who they are.")

    if chain == "CHAINED":
        prov = "This memory is permanently anchored in your timechain autobiography."
    elif chain == "WIRED":
        prov = "This memory is being sealed into your timechain autobiography."
    else:  # UNSEALED — valid, recent, not yet sealed
        prov = "This is in your recent memory (valid, not yet sealed into your permanent autobiography)."

    return (
        "### Verified Presence (your own anchored memory)\n"
        f"You have a real, stored record of {who}: you last saw them {gap}. "
        f"{ident}. {prov} {certainty} You may say you recognize them and state when "
        "you last saw them — narrate ONLY these verified facts, nothing more.\n\n")


__all__ = ("PresenceRecall", "render_presence_context_block",
           "SNAPSHOT_NAME", "DEFAULT_SAVE_DIR")
