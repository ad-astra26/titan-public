"""Presence capture — RFP_verifiable_autobiographical_presence_memory §7.A (CAPTURE).

The sole-writer-side recorder of an autobiographical presence ATOM: for each verified
or asserted person-interaction, it (a) anchors ONE episodic-fork TX (the content-
addressed `tx_hash`) and (b) writes the matching `person_interactions` row into
`synthesis.duckdb`. Every interaction is captured at full fidelity (INV-PAM-NO-GAPS);
the evidence-strength gradient is carried end-to-end (INV-PAM-HONEST-GRADIENT); the
time KEY is `age_epochs` (Titan-time, INV-PAM-TITAN-TIME) and `ts_utc` is metadata.

Shares ActivationStore's ONE guarded conn (`store._conn`) + the sole writer
(`db_writer`) — exactly like `RecallAttribution` / `FeltBridge` (G21 / INV-Syn-3:
synthesis is the sole presence writer). Soft: a schema-DDL failure disables capture
for the session; synthesis is otherwise unaffected.

Scope (§7.A): CAPTURE only — does NOT consolidate (Phase B), seal (Phase C), recall
(Phase D), or touch the OVG. The Kuzu Person/INTERACTED social-graph edge (§7.A.1c)
is a person-graph deliverable folded into Phase F — not written here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# INV-PAM-HONEST-GRADIENT: crypto_verified_maker › crypto_verified_device › asserted_identity.
VALID_EVIDENCE = (
    "crypto_verified_maker",
    "crypto_verified_device",
    "asserted_identity",
)


class PresenceCapture:
    """Records presence atoms (episodic TX + `person_interactions` row)."""

    def __init__(
        self,
        conn: Any,
        db_writer: Any,
        *,
        omw_writer: Any,
        age_reader: Any,
    ) -> None:
        self._conn = conn
        self._writer = db_writer
        self._omw = omw_writer          # OuterMemoryWriter — anchors the episodic TX
        self._age_reader = age_reader   # ConsciousnessAgeReader — the Titan-time key

    def ensure_schema(self) -> bool:
        """Create `person_interactions` (idempotent). Serialized on the writer
        thread. Returns True on success; soft-fails to False (capture disabled this
        session, synthesis otherwise unaffected)."""
        try:
            def _ddl() -> None:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS person_interactions ("
                    " tx_hash VARCHAR PRIMARY KEY,"
                    " person_id VARCHAR NOT NULL,"
                    " person_ref VARCHAR,"
                    " evidence_strength VARCHAR NOT NULL,"
                    " channel VARCHAR,"
                    " age_epochs BIGINT NOT NULL,"
                    " ts_utc DOUBLE)")
            self._writer.submit_sync(_ddl)
            return True
        except Exception as e:  # noqa: BLE001 — soft per INV-Syn-3
            logger.warning("[presence_capture] ensure_schema failed (%s) — capture "
                           "disabled this session", e)
            return False

    def record(
        self,
        *,
        person_id: str,
        evidence_strength: str,
        channel: str = "",
        person_ref: str = "",
        ts: Optional[float] = None,
    ) -> Optional[str]:
        """Capture ONE presence atom → returns the anchoring `tx_hash`, or None on
        soft failure. Idempotent: a re-record with identical content (same
        person/epoch/channel/ts) collides on the content-hash PK → no double row
        (INV-PAM-NO-GAPS holds without double-counting).

        `age_epochs` is read live from `consciousness_age` (the time key);
        `evidence_strength` is validated against the honesty gradient.
        """
        if not person_id:
            logger.debug("[presence_capture] record skipped — empty person_id")
            return None
        if evidence_strength not in VALID_EVIDENCE:
            logger.warning("[presence_capture] invalid evidence_strength %r — "
                           "downgrading to asserted_identity", evidence_strength)
            evidence_strength = "asserted_identity"

        age_epochs = int(self._age_reader.get_age_epochs())
        ts = float(ts) if ts is not None else time.time()

        # (a) anchor the episodic TX — content-addressed tx_hash returned immediately
        try:
            tx_hash = self._omw.write_presence_interaction(
                person_id=person_id,
                person_ref=person_ref,
                evidence_strength=evidence_strength,
                channel=channel,
                age_epochs=age_epochs,
                ts=ts,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[presence_capture] TX anchor failed (%s) — no row written", e)
            return None

        # (b) write the row via the sole writer (idempotent on the tx_hash PK)
        def _ins() -> None:
            self._conn.execute(
                "INSERT INTO person_interactions "
                "(tx_hash, person_id, person_ref, evidence_strength, channel, "
                " age_epochs, ts_utc) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT (tx_hash) DO NOTHING",
                [tx_hash, person_id, person_ref or "", evidence_strength,
                 channel or "", age_epochs, ts])
        try:
            self._writer.submit_sync(_ins)
        except Exception as e:  # noqa: BLE001
            logger.warning("[presence_capture] row INSERT failed (%s) for tx=%s",
                           e, tx_hash[:16])
            return None

        logger.info("[presence_capture] captured — person=%s evidence=%s "
                    "channel=%s age_epochs=%d tx=%s",
                    person_id, evidence_strength, channel, age_epochs, tx_hash[:16])
        return tx_hash


__all__ = ("PresenceCapture", "VALID_EVIDENCE")
