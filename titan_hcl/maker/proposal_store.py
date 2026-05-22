"""ProposalStore — sqlite layer for the TitanMaker substrate.

Storage for Maker-Titan dialogic exchanges. Separate from inner_memory.db
because governance state and cognitive state have different invariants
and different backup cadences (this DB is wired into the Arweave tarball
via timechain_backup.AUXILIARY_BACKUP_PATHS).

Thread-safety: every method opens its own sqlite3.connect (short-lived,
context-managed) — no shared connection, so safe across threads. The
RLock guards the create-or-find idempotency window.
"""
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Optional

from .schemas import (
    MIN_DESCRIPTION_LEN, MIN_REASON_LEN, MIN_TITLE_LEN,
    ProposalRecord, ProposalStatus, ProposalType, validate_reason,
)

logger = logging.getLogger("ProposalStore")

SCHEMA_VERSION = 1
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    proposal_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    created_epoch INTEGER NOT NULL,
    requires_signature INTEGER NOT NULL,
    status TEXT NOT NULL,
    expires_at REAL,
    approved_at REAL,
    approved_signature TEXT,
    approved_signer_pubkey TEXT,
    approval_reason TEXT,
    declined_at REAL,
    decline_reason TEXT,
    titan_low_response_json TEXT,
    titan_high_response_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_type ON proposals(proposal_type);
CREATE INDEX IF NOT EXISTS idx_proposals_payload_hash ON proposals(payload_hash);
"""


class ProposalStore:
    def __init__(self, db_path: str = "data/maker_proposals.db"):
        self._db_path = db_path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
                (str(SCHEMA_VERSION),))
            conn.commit()

    # ── Public API ──────────────────────────────────────────────

    def create(self, *, proposal_type: ProposalType, title: str,
               description: str, payload: dict, requires_signature: bool = False,
               expires_at: Optional[float] = None,
               created_epoch: int = 0) -> ProposalRecord:
        """Create a proposal. Idempotent by (type, payload_hash) for pending status."""
        if not title or len(title.strip()) < MIN_TITLE_LEN:
            raise ValueError(f"title must be ≥ {MIN_TITLE_LEN} chars")
        if not description or len(description.strip()) < MIN_DESCRIPTION_LEN:
            raise ValueError(f"description must be ≥ {MIN_DESCRIPTION_LEN} chars")
        # Canonical JSON for deterministic hashing
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(canonical.encode()).hexdigest()
        with self._lock:
            existing = self.find_pending_by_hash(payload_hash, proposal_type)
            if existing:
                logger.info(
                    "[ProposalStore] Idempotent re-create — returning existing "
                    "pending proposal id=%s type=%s",
                    existing.proposal_id[:8], proposal_type.value)
                return existing
            record = ProposalRecord(
                proposal_id=uuid.uuid4().hex,
                proposal_type=proposal_type,
                title=title.strip(),
                description=description.strip(),
                payload_json=canonical,
                payload_hash=payload_hash,
                created_at=time.time(),
                created_epoch=int(created_epoch),
                requires_signature=requires_signature,
                status=ProposalStatus.PENDING,
                expires_at=expires_at,
            )
            self._insert(record)
            return record

    def get(self, proposal_id: str) -> Optional[ProposalRecord]:
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def list_pending(self) -> list[ProposalRecord]:
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM proposals WHERE status = ? ORDER BY created_at",
                (ProposalStatus.PENDING.value,)
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_by_type(self, proposal_type: ProposalType,
                     limit: int = 50) -> list[ProposalRecord]:
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM proposals WHERE proposal_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (proposal_type.value, limit)
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_recent_responses(self, limit: int = 10) -> list[ProposalRecord]:
        """All-types union of recent approve/decline responses, newest first."""
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM proposals WHERE status IN (?, ?) "
                "ORDER BY COALESCE(approved_at, declined_at, 0) DESC LIMIT ?",
                (ProposalStatus.APPROVED.value, ProposalStatus.DECLINED.value, limit)
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def find_pending_by_hash(self, payload_hash: str,
                             proposal_type: ProposalType) -> Optional[ProposalRecord]:
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM proposals WHERE payload_hash = ? AND proposal_type = ? "
                "AND status = ? LIMIT 1",
                (payload_hash, proposal_type.value, ProposalStatus.PENDING.value)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def mark_approved(self, proposal_id: str, *, reason: str,
                      signature: Optional[str], signer_pubkey: Optional[str]) -> bool:
        reason = validate_reason(reason)
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            cur = conn.execute(
                "UPDATE proposals SET status = ?, approved_at = ?, "
                "approval_reason = ?, approved_signature = ?, approved_signer_pubkey = ? "
                "WHERE proposal_id = ? AND status = ?",
                (ProposalStatus.APPROVED.value, time.time(), reason,
                 signature, signer_pubkey, proposal_id, ProposalStatus.PENDING.value)
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_declined(self, proposal_id: str, reason: str) -> bool:
        reason = validate_reason(reason)
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            cur = conn.execute(
                "UPDATE proposals SET status = ?, declined_at = ?, decline_reason = ? "
                "WHERE proposal_id = ? AND status = ?",
                (ProposalStatus.DECLINED.value, time.time(), reason,
                 proposal_id, ProposalStatus.PENDING.value)
            )
            conn.commit()
            return cur.rowcount > 0

    def expire_old(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            cur = conn.execute(
                "UPDATE proposals SET status = ? "
                "WHERE status = ? AND expires_at IS NOT NULL AND expires_at < ?",
                (ProposalStatus.EXPIRED.value, ProposalStatus.PENDING.value, now)
            )
            conn.commit()
            return cur.rowcount

    def write_low_response(self, proposal_id: str, low_response_json: str) -> None:
        """Tier 2: spirit_worker writes its somatic processing back to the record."""
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.execute(
                "UPDATE proposals SET titan_low_response_json = ? WHERE proposal_id = ?",
                (low_response_json, proposal_id))
            conn.commit()

    def write_high_response(self, proposal_id: str, narration: str) -> None:
        """Tier 3: language_worker writes LLM-narrated reflection."""
        with self._lock, sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.execute(
                "UPDATE proposals SET titan_high_response_text = ? WHERE proposal_id = ?",
                (narration, proposal_id))
            conn.commit()

    # ── Private ──────────────────────────────────────────────────

    def _insert(self, r: ProposalRecord) -> None:
        with sqlite3.connect(self._db_path, timeout=5.0) as conn:
            conn.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.proposal_id, r.proposal_type.value, r.title, r.description,
                 r.payload_json, r.payload_hash, r.created_at, r.created_epoch,
                 int(r.requires_signature), r.status.value, r.expires_at,
                 r.approved_at, r.approved_signature, r.approved_signer_pubkey,
                 r.approval_reason, r.declined_at, r.decline_reason,
                 r.titan_low_response_json, r.titan_high_response_text))
            conn.commit()

    def _row_to_record(self, row) -> ProposalRecord:
        return ProposalRecord(
            proposal_id=row["proposal_id"],
            proposal_type=ProposalType(row["proposal_type"]),
            title=row["title"],
            description=row["description"],
            payload_json=row["payload_json"],
            payload_hash=row["payload_hash"],
            created_at=row["created_at"],
            created_epoch=row["created_epoch"],
            requires_signature=bool(row["requires_signature"]),
            status=ProposalStatus(row["status"]),
            expires_at=row["expires_at"],
            approved_at=row["approved_at"],
            approved_signature=row["approved_signature"],
            approved_signer_pubkey=row["approved_signer_pubkey"],
            approval_reason=row["approval_reason"],
            declined_at=row["declined_at"],
            decline_reason=row["decline_reason"],
            titan_low_response_json=row["titan_low_response_json"],
            titan_high_response_text=row["titan_high_response_text"],
        )
