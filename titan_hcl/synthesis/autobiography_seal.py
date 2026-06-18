"""Autobiography seal — RFP_verifiable_autobiographical_presence_memory §7.C (SEAL).

At the circadian-trough `CYCLE_CLOSED` (published by soul_diary_worker when its
`CircadianCycleCounter.latch_if_trough()` fires), seal the just-closed cycle's
presence set onto FORK_MAIN (Titan's autobiographical spine):

  1. final-fold the closed cycle `[start_epoch, end_epoch)` → `presence_cycle_rollup`
     (end-bounded; catches atoms between the last dream fold and the latch).
  2. Merkle-root the cycle's interaction `tx_hash`es (empty cycle → SHA-256(b"")
     deterministic root — a "met no one" cycle STILL seals; INV-PAM-NO-GAPS).
  3. emit ONE `presence_seal` fork-main TX (Titan's own autobiographical act —
     separate from the Soul Diary's diary TX; the timechain seals both on its own
     cadence — Q2 two-separate-TXs, Maker-confirmed 2026-06-18). A LOCAL immutable
     block (Option A — no per-cycle Solana memo).
  4. idempotent: a small JSON ledger (`presence_seal_ledger.json`) persists
     `last_sealed_cycle_id`; re-sealing a cycle ≤ that is a no-op (INV-PAM-SEAL-IDEMPOTENT).
  5. CHAINED tracking (feeds Phase D recall): the seal is recorded `WIRED` at emit;
     on the first `TIMECHAIN_SEALED{fork=main, ts > emit_ts}` it flips `CHAINED`
     (seal_fork seals ALL pending fork-main TXs, so our TX is guaranteed in that
     block) + records the anchoring `block_height`/`block_hash`.

Shares the synthesis_worker's ONE guarded conn (`store._conn`) + sole writer
(`db_writer`) — EVERY `.execute()` (read OR write) routes through `submit_sync`
(G21 / INV-Syn-3). The seal TX is emitted via the worker's `OuterMemoryWriter`.

Scope (§7.C): SEAL only — no recall (Phase D), no OVG (Phase E). The ledger's
`chain_status(cycle_id)` read helper is the Phase-D recall hook.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from titan_hcl.synthesis.merkle import merkle_root_hex

logger = logging.getLogger(__name__)

_LEDGER_NAME = "presence_seal_ledger.json"


class AutobiographySeal:
    """Seals a closed circadian cycle's presence set onto FORK_MAIN (idempotent)."""

    def __init__(
        self,
        conn: Any,
        db_writer: Any,
        rollup: Any,
        omw: Any,
        *,
        save_dir: str = "data/titan_time",
    ) -> None:
        self._conn = conn
        self._writer = db_writer
        self._rollup = rollup          # PresenceRollup (for fold_closed_cycle)
        self._omw = omw                # OuterMemoryWriter (fork="main" emit)
        self._save_dir = save_dir
        self._ledger_path = os.path.join(save_dir, _LEDGER_NAME)

    # ── ledger (atomic JSON; the single source of seal idempotency + chain_status) ──
    def _load_ledger(self) -> dict:
        try:
            with open(self._ledger_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("ledger root not a dict")
            data.setdefault("last_sealed_cycle_id", -1)
            data.setdefault("cycles", {})
            return data
        except FileNotFoundError:
            return {"last_sealed_cycle_id": -1, "cycles": {}}
        except Exception as e:  # noqa: BLE001 — corrupt → start fresh (raw atoms intact)
            logger.warning("[autobiography_seal] ledger unreadable (%s) — re-init", e)
            return {"last_sealed_cycle_id": -1, "cycles": {}}

    def _persist_ledger(self, data: dict) -> None:
        os.makedirs(self._save_dir, exist_ok=True)
        tmp = self._ledger_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._ledger_path)  # atomic (G16)

    # ── read helper (Phase D recall hook) ───────────────────────────────────────
    def chain_status(self, cycle_id: int) -> Optional[str]:
        """`CHAINED` | `WIRED` for a sealed cycle, else `None` (never sealed)."""
        entry = self._load_ledger()["cycles"].get(str(int(cycle_id)))
        return entry.get("chain_status") if entry else None

    # ── §7.C step 1-4: seal the closed cycle ─────────────────────────────────────
    def seal_closed_cycle(
        self,
        cycle_id: int,
        start_epoch: int,
        end_epoch: int,
        *,
        ts_utc_range: Optional[list] = None,
    ) -> Optional[dict]:
        """Final-fold + Merkle-root + emit the presence_seal fork-main TX for the
        CLOSED cycle. Idempotent (no-op + return None for an already-sealed cycle).
        Returns the new ledger entry on a fresh seal."""
        cycle_id = int(cycle_id)
        ledger = self._load_ledger()
        last_sealed = int(ledger.get("last_sealed_cycle_id", -1))
        if cycle_id <= last_sealed:
            logger.info("[autobiography_seal] cycle=%d already sealed "
                        "(last_sealed=%d) — no-op (INV-PAM-SEAL-IDEMPOTENT)",
                        cycle_id, last_sealed)
            return None
        # No-gaps observability: the edge-triggered latch increments by 1/trough, so
        # a contiguous cycle_id is expected. A jump means a CYCLE_CLOSED was missed
        # (e.g. synthesis down when soul_diary published) — the intervening cycles'
        # atoms are still in person_interactions (raw, never dropped) but are not
        # individually sealed. Log loudly (no silent gap); Phase-C bound.
        if last_sealed >= 0 and cycle_id > last_sealed + 1:
            logger.warning("[autobiography_seal] cycle gap — sealing cycle=%d but "
                           "last_sealed=%d (cycles %d..%d were not individually "
                           "sealed; raw atoms retained)", cycle_id, last_sealed,
                           last_sealed + 1, cycle_id - 1)

        start_epoch, end_epoch = int(start_epoch), int(end_epoch)

        # 1) final-fold the closed cycle (end-bounded) into presence_cycle_rollup.
        try:
            self._rollup.fold_closed_cycle(cycle_id, start_epoch, end_epoch)
        except Exception as e:  # noqa: BLE001 — fold soft-fails; still seal what's there
            logger.warning("[autobiography_seal] final-fold failed for cycle=%d "
                           "(%s) — sealing from existing rollup", cycle_id, e)

        # 2) collect the cycle's interaction tx_hashes (ORDER BY for a deterministic
        #    Merkle root) + the per-person rollups — BOTH on the writer thread (G21).
        def _read_tx_hashes():
            return self._conn.execute(
                "SELECT tx_hash FROM person_interactions "
                "WHERE age_epochs >= ? AND age_epochs < ? ORDER BY tx_hash",
                [start_epoch, end_epoch]).fetchall()

        def _read_rollups():
            return self._conn.execute(
                "SELECT person_id, first_seen_epoch, last_seen_epoch, count, "
                "evidence_strength FROM presence_cycle_rollup "
                "WHERE cycle_id = ? ORDER BY person_id", [cycle_id]).fetchall()

        try:
            tx_rows = self._writer.submit_sync(_read_tx_hashes)
            roll_rows = self._writer.submit_sync(_read_rollups)
        except Exception as e:  # noqa: BLE001
            logger.warning("[autobiography_seal] seal read failed for cycle=%d (%s) "
                           "— deferring (will retry next CYCLE_CLOSED)", cycle_id, e)
            return None

        tx_hashes = [str(r[0]) for r in (tx_rows or [])]
        merkle_root = merkle_root_hex(tx_hashes)  # empty → SHA-256(b"") (no-gaps)
        person_rollups = [
            {"person_id": str(r[0]), "first_seen_epoch": int(r[1]),
             "last_seen_epoch": int(r[2]), "count": int(r[3]),
             "evidence_strength": str(r[4])}
            for r in (roll_rows or [])
        ]

        # 3) emit ONE presence_seal fork-main TX (Titan's own act; rides the cadence).
        emit_ts = time.time()
        try:
            anchor_tx = self._omw.write_presence_seal(
                cycle_id=cycle_id,
                age_epoch_range=[start_epoch, end_epoch],
                merkle_root=merkle_root,
                person_rollups=person_rollups,
                interaction_count=len(tx_hashes),
                ts_utc_range=ts_utc_range,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[autobiography_seal] presence_seal emit failed for "
                           "cycle=%d (%s) — not recording (will retry)", cycle_id, e)
            return None

        # 4) record WIRED in the ledger (CHAINED flips on TIMECHAIN_SEALED{fork=main}).
        entry = {
            "anchor_tx": anchor_tx,
            "merkle_root": merkle_root,
            "age_epoch_range": [start_epoch, end_epoch],
            "interaction_count": len(tx_hashes),
            "person_count": len(person_rollups),
            "emit_ts": emit_ts,
            "chain_status": "WIRED",
            "block_height": None,
            "block_hash": None,
        }
        ledger["cycles"][str(cycle_id)] = entry
        ledger["last_sealed_cycle_id"] = cycle_id
        self._persist_ledger(ledger)
        logger.info("[autobiography_seal] SEALED cycle=%d persons=%d interactions=%d "
                    "merkle=%s (WIRED — awaiting fork-main block)", cycle_id,
                    len(person_rollups), len(tx_hashes), merkle_root[:12])
        return entry

    # ── §7.D: the lock-free recall surface (PreHook reads this, never the DuckDB) ──
    def export_recall_snapshot(self, snapshot_path: Optional[str] = None) -> int:
        """RFP §7.D — export the per-person presence-recall surface as an atomic JSON
        (mirrors `engram_store.export_snapshot` → `spine_snapshot.json`). The agno
        PreHook reads THIS (cross-process; never opens synthesis.duckdb — G18). Per
        person: their most-recent verified presence (`last_seen_epoch`), evidence,
        and the `chain_status` of the cycle that holds it (CHAINED › WIRED ›
        UNSEALED). Returns the number of persons written. Reads on the writer
        thread (G21). Soft-fails to 0 (recall degrades to "no recognition" — honest).

        Default path = `<save_dir>/presence_recall_snapshot.json` — the same
        relative `data/titan_time/` dir the whole titan_time family uses (both the
        synthesis and agno processes share cwd), matching `presence_recall.py`'s
        read default (`PresenceRecall.SNAPSHOT_NAME`)."""
        if snapshot_path is None:
            # NOTE: filename kept in sync with logic/presence_recall.py SNAPSHOT_NAME
            snapshot_path = os.path.join(self._save_dir, "presence_recall_snapshot.json")
        ledger = self._load_ledger()

        # Read last-seen + strongest-evidence per person DIRECTLY from the durable
        # `person_interactions` atoms (NOT the dream-folded rollup) — so a person met
        # moments ago is recognizable IMMEDIATELY (recognize-on-validity: a captured
        # presence is valid the instant it is written, like a mempool memory; the seal
        # adds permanence, not validity). The atom is written at capture + this export
        # fires in the capture handler, so there is no fold-lag window.
        _ev_case = ("MAX(CASE evidence_strength "
                    " WHEN 'crypto_verified_maker' THEN 3 "
                    " WHEN 'crypto_verified_device' THEN 2 ELSE 1 END) AS ev_ord")

        def _read():
            # §7.F (F.2) — representative identity hashes per person (internal-only;
            # used ONLY to merge the same human across handles at recall). Resilient
            # to a person_interactions table created before the Phase-F migration
            # (a box mid-deploy): fall back to the pre-F shape with empty hashes.
            try:
                return self._conn.execute(
                    "SELECT person_id, MAX(age_epochs) AS last_seen, "
                    + _ev_case + ", MAX(did_hash) AS did_h, MAX(ip_hash) AS ip_h "
                    "FROM person_interactions GROUP BY person_id").fetchall()
            except Exception:
                rows = self._conn.execute(
                    "SELECT person_id, MAX(age_epochs) AS last_seen, "
                    + _ev_case + " FROM person_interactions "
                    "GROUP BY person_id").fetchall()
                return [(r[0], r[1], r[2], "", "") for r in (rows or [])]
        try:
            rows = self._writer.submit_sync(_read)
        except Exception as e:  # noqa: BLE001
            logger.warning("[autobiography_seal] recall snapshot read failed (%s)", e)
            return 0

        _EV = {3: "crypto_verified_maker", 2: "crypto_verified_device", 1: "asserted_identity"}
        cycles = ledger.get("cycles", {})
        persons = {}
        for pid, last_seen, ev_ord, did_h, ip_h in (rows or []):
            last_seen = int(last_seen)
            # chain_status = provability of the cycle whose age_epoch_range contains
            # last_seen (CHAINED › WIRED). No sealed cycle contains it ⇒ it is in the
            # current OPEN cycle ⇒ UNSEALED (valid + recent, not yet sealed — still
            # fully recognizable; the seal is the permanence layer).
            chain_status, anchor, cyc_id = "UNSEALED", None, None
            for cid, cyc in cycles.items():
                rng = cyc.get("age_epoch_range")
                if rng and int(rng[0]) <= last_seen < int(rng[1]):
                    chain_status = cyc.get("chain_status", "WIRED")
                    anchor = cyc.get("block_hash") or cyc.get("merkle_root")
                    cyc_id = int(cid)
                    break
            persons[str(pid)] = {
                "last_seen_epoch": last_seen,
                "evidence_strength": _EV.get(int(ev_ord), "asserted_identity"),
                "chain_status": chain_status,
                "cycle_id": cyc_id,
                "anchor": anchor,
                # §7.F (F.2) — internal-only identity helping-signals (NEVER
                # rendered into context/output); recall merges handles by these.
                "did_hash": did_h or "",
                "ip_hash": ip_h or "",
            }

        data = {"persons": persons, "updated_ts": time.time()}
        try:
            os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
            tmp = snapshot_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, snapshot_path)  # atomic (G16)
        except Exception as e:  # noqa: BLE001
            logger.warning("[autobiography_seal] recall snapshot write failed (%s)", e)
            return 0
        return len(persons)

    # ── §7.C step 5: CHAINED on the next fork-main seal ──────────────────────────
    def note_fork_main_sealed(self, block_height: int, block_hash: str,
                              sealed_ts: float) -> int:
        """A FORK_MAIN block sealed at `sealed_ts`. `seal_fork` seals ALL pending
        fork-main TXs, so every WIRED seal emitted at/before `sealed_ts` is now in a
        block → flip CHAINED + record the anchoring block. Returns the count
        upgraded. Idempotent (already-CHAINED entries are skipped)."""
        ledger = self._load_ledger()
        upgraded = 0
        for cid, entry in ledger["cycles"].items():
            if (entry.get("chain_status") == "WIRED"
                    and float(entry.get("emit_ts", 0.0)) <= float(sealed_ts)):
                entry["chain_status"] = "CHAINED"
                entry["block_height"] = int(block_height)
                entry["block_hash"] = str(block_hash)
                upgraded += 1
        if upgraded:
            self._persist_ledger(ledger)
            logger.info("[autobiography_seal] %d cycle seal(s) → CHAINED "
                        "(fork-main block #%d)", upgraded, block_height)
        return upgraded


__all__ = ("AutobiographySeal",)
