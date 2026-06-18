"""Presence rollup — RFP_verifiable_autobiographical_presence_memory §7.B (CONSOLIDATE).

At each dream-boundary consolidation, fold the CURRENT cycle's raw `person_interactions`
atoms into a per-person index `presence_cycle_rollup(cycle_id, person_id, first_seen_epoch,
last_seen_epoch, count, evidence_strength)`. The rollup is a DERIVED, recomputable index —
the raw atoms are only read, never dropped (INV-PAM-NO-GAPS). Keyed by `cycle_id` +
`age_epoch` (Titan-time, INV-PAM-TITAN-TIME) — never wall-clock. Carries the STRONGEST
evidence each person showed this cycle (INV-PAM-HONEST-GRADIENT).

Shares ActivationStore's ONE guarded conn (`store._conn`) + the sole writer (`db_writer`)
— like PresenceCapture / RecallAttribution (G21 / INV-Syn-3). Reads run directly on the
conn; writes route through the sole writer. Idempotent: folding the same cycle twice is a
no-op-equivalent UPSERT.

Scope (§7.B): the rollup index ONLY — does NOT seal (Phase C) or recall (Phase D). The
current OPEN cycle is folded (no end-bound); Phase C does the closing-cycle final fold.
"""
from __future__ import annotations

import logging
from typing import Any

from titan_hcl.logic.titan_time import read_current_cycle

logger = logging.getLogger(__name__)

# INV-PAM-HONEST-GRADIENT ordinal (strongest wins): crypto_verified_maker › device › asserted.
_EVIDENCE_BY_ORDINAL = {3: "crypto_verified_maker", 2: "crypto_verified_device", 1: "asserted_identity"}
_EVIDENCE_CASE_SQL = (
    "CASE evidence_strength "
    "WHEN 'crypto_verified_maker' THEN 3 "
    "WHEN 'crypto_verified_device' THEN 2 "
    "ELSE 1 END")


class PresenceRollup:
    """Folds per-cycle presence atoms into the queryable rollup index."""

    def __init__(self, conn: Any, db_writer: Any, *, save_dir: str = "data/titan_time") -> None:
        self._conn = conn
        self._writer = db_writer
        self._save_dir = save_dir   # where the Titan-time cycle counter persists

    def ensure_schema(self) -> bool:
        """Create `presence_cycle_rollup` (idempotent). Serialized on the writer
        thread. Soft-fails to False (rollup disabled this session; capture + the raw
        atoms are unaffected)."""
        try:
            def _ddl() -> None:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS presence_cycle_rollup ("
                    " cycle_id BIGINT NOT NULL,"
                    " person_id VARCHAR NOT NULL,"
                    " first_seen_epoch BIGINT NOT NULL,"
                    " last_seen_epoch BIGINT NOT NULL,"
                    " count BIGINT NOT NULL,"
                    " evidence_strength VARCHAR NOT NULL,"
                    " PRIMARY KEY (cycle_id, person_id))")
            self._writer.submit_sync(_ddl)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[presence_rollup] ensure_schema failed (%s) — rollup "
                           "disabled this session", e)
            return False

    def fold(self) -> int:
        """Recompute the CURRENT open cycle's rollup from its raw atoms; UPSERT one
        row per person. Returns the number of person-rows folded (0 on empty/soft
        failure). Idempotent — safe to call every dream."""
        cycle_id, cycle_start_epoch = read_current_cycle(self._save_dir)
        # READ (direct on the shared conn): per-person aggregates over THIS cycle's
        # atoms (age_epochs >= the cycle's start; Titan-time partition).
        try:
            rows = self._conn.execute(
                "SELECT person_id, MIN(age_epochs), MAX(age_epochs), COUNT(*), "
                f"MAX({_EVIDENCE_CASE_SQL}) "
                "FROM person_interactions WHERE age_epochs >= ? "
                "GROUP BY person_id",
                [int(cycle_start_epoch)]).fetchall()
        except Exception as e:  # noqa: BLE001
            logger.warning("[presence_rollup] fold read failed (%s)", e)
            return 0
        if not rows:
            return 0

        folded = [
            (int(cycle_id), str(pid), int(first), int(last), int(cnt),
             _EVIDENCE_BY_ORDINAL.get(int(ev_ord), "asserted_identity"))
            for (pid, first, last, cnt, ev_ord) in rows
        ]

        def _upsert() -> None:
            for r in folded:
                self._conn.execute(
                    "INSERT INTO presence_cycle_rollup "
                    "(cycle_id, person_id, first_seen_epoch, last_seen_epoch, "
                    " count, evidence_strength) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT (cycle_id, person_id) DO UPDATE SET "
                    "first_seen_epoch=excluded.first_seen_epoch, "
                    "last_seen_epoch=excluded.last_seen_epoch, "
                    "count=excluded.count, "
                    "evidence_strength=excluded.evidence_strength", list(r))
        try:
            self._writer.submit_sync(_upsert)
        except Exception as e:  # noqa: BLE001
            logger.warning("[presence_rollup] fold upsert failed (%s)", e)
            return 0

        logger.info("[presence_rollup] folded cycle=%d persons=%d (start_epoch=%d)",
                    cycle_id, len(folded), cycle_start_epoch)
        return len(folded)


__all__ = ("PresenceRollup",)
