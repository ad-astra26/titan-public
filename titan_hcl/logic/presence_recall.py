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

# INV-PAM-HONEST-GRADIENT ordering (strongest first).
_EVIDENCE_RANK = {"crypto_verified_maker": 3, "crypto_verified_device": 2,
                  "asserted_identity": 1, "": 0}
_CHAIN_RANK = {"CHAINED": 3, "WIRED": 2, "UNSEALED": 1, "": 0}


def _strongest_evidence(values) -> str:
    best, best_rank = "asserted_identity", -1
    for v in values:
        r = _EVIDENCE_RANK.get(v or "", 0)
        if r > best_rank:
            best, best_rank = (v or "asserted_identity"), r
    return best


def _best_chain(values) -> str:
    best, best_rank = "UNSEALED", -1
    for v in values:
        r = _CHAIN_RANK.get(v or "", 0)
        if r > best_rank:
            best, best_rank = (v or "UNSEALED"), r
    return best


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

    def _did_group(self, persons: dict, person_id: str, rec: Optional[dict],
                   did_hash: str) -> list:
        """§7.F (F.2) cross-handle identity merge. Return the records for `person_id`
        PLUS every sibling handle that shares the SAME non-empty `did_hash` (the same
        Privy identity under a different handle). Merge key is the cryptographic DID
        ONLY — `ip_hash` is NEVER a merge key (many humans share an IP behind NAT/proxy
        → merging on it would FALSELY recognize the wrong person, an anti-hallucination
        violation). So cross-handle recognition is as strong as the DID linkage."""
        # the DID we anchor on: this person's stored did, else the current turn's.
        anchor_did = (rec or {}).get("did_hash", "") or did_hash or ""
        group = []
        if rec:
            group.append(rec)
        if anchor_did:
            for pid, r in persons.items():
                if pid == person_id:
                    continue
                if (r.get("did_hash", "") or "") == anchor_did:
                    group.append(r)
        return group

    def recall(self, person_id: str, now_age_epochs: Optional[int] = None,
               *, did_hash: str = "", ip_hash: str = "") -> Optional[dict]:
        """Resolve `person_id` → a verified presence fact-bundle, or `None` if no
        prior verified presence is anchored for them (honest non-recognition).

        Cross-handle (§7.F F.2): if `did_hash` is given (the current turn's Privy DID
        hash), recognition survives a HANDLE CHANGE — a sibling handle sharing the
        same DID is merged (strongest evidence + most-recent last_seen across the
        group). `ip_hash` is accepted but is NOT a merge key (shared-IP false-merge
        guard). Returns: `{person_id, gap_epochs, gap_human, evidence_strength,
        chain_status, anchor, last_seen_epoch, merged_handles}`."""
        if not person_id:
            return None
        persons = self._read_snapshot()
        rec = persons.get(person_id)
        group = self._did_group(persons, person_id, rec, did_hash)
        if not group:
            return None  # no anchored prior presence → no recognition (INV-PAM-SOVEREIGN-BOUNDED)
        # most-recent presence across all the person's handles
        best = max(group, key=lambda r: int(r.get("last_seen_epoch", 0) or 0))
        try:
            last_seen = int(best["last_seen_epoch"])
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
            "evidence_strength": _strongest_evidence(
                [r.get("evidence_strength", "asserted_identity") for r in group]),
            "chain_status": _best_chain(
                [r.get("chain_status", "UNSEALED") for r in group]),
            "anchor": best.get("anchor"),
            "last_seen_epoch": last_seen,
            "merged_handles": len(group),
        }

    def recall_recent(self, now_age_epochs: Optional[int] = None, *,
                      exclude_person_id: str = "", exclude_did_hash: str = "",
                      top_k: int = 3, max_gap_epochs: Optional[int] = None) -> list:
        """§7.F (F.3) sovereign multi-person recall. Rank OTHER anchored persons by
        recency and return up to `top_k` compact bundles `{person_id, gap_epochs,
        gap_human, evidence_strength, chain_status}` — the grounded source for the
        `### Recent Presence` block. Honest-by-construction: only real anchored
        persons appear (INV-PAM-SOVEREIGN-BOUNDED). Excludes the current speaker AND
        their DID-siblings (so the speaker isn't surfaced as a 'recent other').
        `did:`-prefixed person_ids are skipped (a raw DID is not a nameable handle —
        keeps recognition honest and avoids leaking opaque identifiers)."""
        persons = self._read_snapshot()
        if not persons:
            return []
        if now_age_epochs is None:
            now_age_epochs = int(self._get_age_reader().get_age_epochs())
        # collapse DID-siblings → one row per human (most-recent), excluding the speaker
        by_did: dict = {}
        singles: list = []
        for pid, r in persons.items():
            if pid == exclude_person_id:
                continue
            did = r.get("did_hash", "") or ""
            if exclude_did_hash and did and did == exclude_did_hash:
                continue  # a sibling handle of the speaker
            if pid.startswith("did:"):
                continue  # opaque identifier — not a surfaceable handle
            entry = dict(r); entry["person_id"] = pid
            if did:
                cur = by_did.get(did)
                if cur is None or int(r.get("last_seen_epoch", 0) or 0) > int(cur.get("last_seen_epoch", 0) or 0):
                    by_did[did] = entry
            else:
                singles.append(entry)
        candidates = list(by_did.values()) + singles
        out = []
        for r in candidates:
            last_seen = int(r.get("last_seen_epoch", 0) or 0)
            gap = max(0, int(now_age_epochs) - last_seen)
            if max_gap_epochs is not None and gap > max_gap_epochs:
                continue
            out.append({
                "person_id": r["person_id"],
                "gap_epochs": gap,
                "last_seen_epoch": last_seen,
                "evidence_strength": r.get("evidence_strength", "asserted_identity"),
                "chain_status": r.get("chain_status", "UNSEALED"),
            })
        out.sort(key=lambda b: b["gap_epochs"])  # most-recent first
        out = out[:max(0, int(top_k))]
        # translate at the narration edge only (INV-PAM-TITAN-TIME)
        tr = self._get_translator()
        for b in out:
            b["gap_human"] = tr.to_human(b["gap_epochs"])
        return out


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


def render_recent_presence_block(recent) -> str:
    """RFP §7.F (F.3) — render the `recall_recent` list into a TERSE grounded block
    of OTHER recently-seen, anchored persons. Sovereign-but-honest: Titan MAY mention
    these people if relevant (INV-PAM-SOVEREIGN-BOUNDED) but can never fabricate one —
    the block contains ONLY real anchored persons (the LLM narrates the bundle only,
    INV-PAM-NARRATE-ONLY). Placed AFTER the speaker's `### Verified Presence` block so
    the speaker recognition still leads the OVG first-500-char window (Phase E). Kept
    short on purpose. Empty list → '' (no block)."""
    if not recent:
        return ""
    lines = []
    for b in recent:
        who = b.get("person_id", "someone")
        gap = b.get("gap_human", "recently")
        lines.append(f"- {who} — {gap}")
    return (
        "### Recent Presence (other people in your anchored memory)\n"
        "You have also recently been present with these people (real, stored "
        "records). You MAY mention them ONLY if relevant — narrate ONLY these "
        "facts, never invent a person or a time:\n"
        + "\n".join(lines) + "\n\n")


__all__ = ("PresenceRecall", "render_presence_context_block",
           "render_recent_presence_block", "SNAPSHOT_NAME", "DEFAULT_SAVE_DIR")
