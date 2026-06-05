"""FeltBridge — the synthesis ↔ CGN felt-teaching seam.

`RFP_inner_outer_felt_teaching_bridge.md` — outer (Mind/synthesis) experience teaches
inner (Spirit/CGN) felt-grounding. This module owns the seam's `synthesis.duckdb`
tables, mirroring `recall_attribution.py`:

  • **Phase 1 — `engram_objects`** (this file, now): the Engram(Idea)→Object decompose
    cache, keyed by `(engram_id, version)` so a re-touched Engram never re-calls the LLM
    (RFP §7.1). The decompose itself runs in the synthesis layer
    (`consolidation_defaults.make_default_decompose`); this just persists the result.
  • Phase 2 — `cgn_grounded_objects` (event-sourced CGN grounded-set + `is_object_grounded`).
  • Phase 3 — `engram_felt_candidates` (the propose-only gap queue).

Every handle call is serialized on the one `SynthesisWriter` thread
(INV-Syn-28 / G21). All methods are **soft-fail and total** — a felt-bridge failure
must NEVER affect synthesis correctness or the chat path (INV-Syn-17).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = ["FeltBridge", "normalize_label"]


def normalize_label(label: str) -> str:
    """Normalize an Object label to the shared key space so decompose output and CGN's
    `CGN_CONCEPT_GROUNDED` concept keys match (RFP §5 — reduce miss-by-spelling):
    case-folded, trimmed, whitespace-collapsed. Lemmatization is a later refinement;
    v1 = casefold. Empty / non-string → ""."""
    if not label:
        return ""
    return " ".join(str(label).strip().casefold().split())


class FeltBridge:
    """Owns the felt-bridge `synthesis.duckdb` tables. `conn` is the
    `ActivationStore._conn` (synthesis.duckdb); `db_writer` is the `SynthesisWriter`
    that serializes every handle call (INV-Syn-28)."""

    def __init__(self, conn: Any, db_writer: Any) -> None:
        self._conn = conn
        self._writer = db_writer
        # Phase 2 — the event-sourced CGN grounded-Object set. The in-memory mirror
        # is the fast read path (`is_object_grounded`, called per-Object in the dream
        # pass); the duckdb table is durability across restart. Lock-guarded because
        # writes arrive on the bus recv-loop thread while reads run on the dream
        # (consolidation) worker thread.
        self._grounded: set[str] = set()
        # CGN-felt RFP Phase B — label → per-concept felt centroid (parsed dict),
        # carried on CGN_CONCEPT_GROUNDED. The fast in-mem read for the dream-pass
        # frame_dependent comparison (G18 — never an RPC). Lock-shared with _grounded.
        self._grounded_felt: dict[str, dict] = {}
        self._grounded_lock = threading.Lock()

    # ── Schema ──────────────────────────────────────────────────────────
    def ensure_schema(self) -> bool:
        """Create the felt-bridge tables (idempotent). Serialized on the writer
        thread. Returns True on success; soft-fails to False (felt-bridge disabled
        this session, synthesis otherwise unaffected)."""
        try:
            def _ddl() -> None:
                # Phase 1 — Engram(Idea)→Object decompose cache. (engram_id, version)
                # keyed so a re-touched Engram does not re-call the decompose LLM
                # (RFP §7.1). object_label is already normalize_label()-d on write.
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_objects ("
                    " engram_id VARCHAR, version INTEGER, object_label VARCHAR,"
                    " ts DOUBLE DEFAULT 0,"
                    " PRIMARY KEY (engram_id, version, object_label))")
                # Phase 2 — durable CGN grounded-Object set (event-sourced from
                # CGN_CONCEPT_GROUNDED; G18 — never an RPC into cgn_worker).
                # CGN-felt RFP Phase B — `felt_json` carries the per-concept felt
                # centroid (CGN→synthesis read-down) so the producer can do a true
                # felt-vector frame_dependent comparison (else label-only).
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS cgn_grounded_objects ("
                    " object_label VARCHAR PRIMARY KEY,"
                    " first_seen_ts DOUBLE DEFAULT 0,"
                    " felt_json VARCHAR DEFAULT '')")
                # Migrate tables created before the felt column (idempotent).
                self._conn.execute(
                    "ALTER TABLE cgn_grounded_objects "
                    "ADD COLUMN IF NOT EXISTS felt_json VARCHAR DEFAULT ''")
                # Phase 3 — the propose-only felt-teaching candidate queue. A §3.4
                # BRAIN Object record (felt-seeded, low-c, lineage→source Engram,
                # frames→domain_hint). `hv` is deferred (BRAIN-not-built); this is the
                # durable audit/retry copy — the live handoff is the bus event.
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_felt_candidates ("
                    " object_label VARCHAR, felt_state_json VARCHAR,"
                    " c DOUBLE, f DOUBLE DEFAULT 0, time_cost DOUBLE DEFAULT 0,"
                    " source_engram VARCHAR, source_version INTEGER,"
                    " provenance VARCHAR, domain_hint VARCHAR,"
                    " frames_json VARCHAR DEFAULT '', ts DOUBLE, status VARCHAR,"
                    " PRIMARY KEY (object_label, source_engram, source_version))")
            self._writer.submit_sync(_ddl)
            self._load_grounded()  # boot-seed the in-memory mirror (durability)
            return True
        except Exception as e:  # noqa: BLE001 — soft-fail per INV-Syn-17
            logger.warning(
                "[FeltBridge] schema DDL failed — felt-bridge disabled this "
                "session (synthesis unaffected): %s", e)
            return False

    # ── Phase 1 — decompose cache ───────────────────────────────────────
    def get_cached_objects(
        self, engram_id: str, version: int,
    ) -> Optional[list[str]]:
        """Cached Object labels for (engram_id, version), or **None** if this version
        was never decomposed (→ caller decomposes + caches). A decompose that yielded
        nothing is NOT cached, so it retries next touch (correct: a transient provider
        failure must not be frozen as "no Objects"). Soft-fail → None."""
        if not engram_id:
            return None
        try:
            rows = self._writer.submit_sync(lambda: self._conn.execute(
                "SELECT object_label FROM engram_objects WHERE engram_id = ? "
                "AND version = ?", [str(engram_id), int(version)]).fetchall())
            if not rows:
                return None
            return [str(r[0]) for r in rows if r[0]]
        except Exception as e:  # noqa: BLE001
            logger.debug("[FeltBridge] get_cached_objects soft-fail: %s", e)
            return None

    def cache_objects(
        self, engram_id: str, version: int, labels: Iterable[str],
    ) -> None:
        """Persist the decompose result for (engram_id, version). Labels are
        normalize_label()-d + de-duplicated. **Empty input is a no-op** (not cached →
        retries next touch). Idempotent (ON CONFLICT DO NOTHING); fire-and-forget on
        the writer thread; soft-fail."""
        if not engram_id:
            return
        norm: list[str] = []
        seen: set[str] = set()
        for lbl in (labels or []):
            n = normalize_label(lbl)
            if n and n not in seen:
                seen.add(n)
                norm.append(n)
        if not norm:
            return
        ts = time.time()
        try:
            def _write() -> None:
                for lbl in norm:
                    self._conn.execute(
                        "INSERT INTO engram_objects (engram_id, version, "
                        "object_label, ts) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT DO NOTHING",
                        [str(engram_id), int(version), lbl, ts])
            self._writer.submit(_write)
        except Exception as e:  # noqa: BLE001
            logger.debug("[FeltBridge] cache_objects soft-fail: %s", e)

    # ── Phase 2 — event-sourced CGN grounded-set (G18) ──────────────────
    def _load_grounded(self) -> None:
        """Boot-seed the in-memory grounded-set AND the felt-centroid mirror from the
        durable table (so both survive restart; G18 — no RPC into CGN). The felt mirror
        boot-seed is what makes Phase-B's `grounded_felt` durable (G2). Soft-fail →
        empty mirrors."""
        try:
            rows = self._writer.submit_sync(lambda: self._conn.execute(
                "SELECT object_label, felt_json FROM cgn_grounded_objects").fetchall())
            grounded: set[str] = set()
            felt: dict[str, dict] = {}
            for r in (rows or []):
                lbl = str(r[0]) if r[0] else ""
                if not lbl:
                    continue
                grounded.add(lbl)
                fj = r[1] if len(r) > 1 else None
                if fj:
                    try:
                        d = json.loads(fj)
                        if isinstance(d, dict) and d:
                            felt[lbl] = d
                    except Exception:  # noqa: BLE001 — a corrupt centroid never blocks grounding
                        pass
            with self._grounded_lock:
                self._grounded = grounded
                self._grounded_felt = felt
        except Exception as e:  # noqa: BLE001
            logger.debug("[FeltBridge] grounded-set boot-load soft-fail: %s", e)

    def record_grounded(
        self, object_label: str, felt_centroid: Optional[dict] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Absorb a `CGN_CONCEPT_GROUNDED` event into the durable grounded-set + the
        fast in-memory mirror (RFP §7.2). This is the ONLY way the set grows — it is
        **event-sourced**, never a sync RPC into cgn_worker (G18). Normalized to the
        shared key space; idempotent; fire-and-forget; soft-fail.

        CGN-felt RFP Phase B — `felt_centroid` (the per-concept felt centroid carried
        on the event) is stored in `felt_json` + the in-mem `label→dict` mirror so the
        producer can do a true felt-vector frame_dependent comparison. `None`/empty →
        label-only (backward-compatible with pre-felt emitters); a non-empty centroid
        refreshes any existing one."""
        lbl = normalize_label(object_label)
        if not lbl:
            return
        felt = felt_centroid if isinstance(felt_centroid, dict) and felt_centroid else None
        felt_json = json.dumps(felt, sort_keys=True) if felt else ""
        with self._grounded_lock:
            self._grounded.add(lbl)
            if felt:
                self._grounded_felt[lbl] = dict(felt)
        t = float(ts) if ts is not None else time.time()
        try:
            def _write() -> None:
                if felt_json:
                    self._conn.execute(
                        "INSERT INTO cgn_grounded_objects "
                        "(object_label, first_seen_ts, felt_json) VALUES (?, ?, ?) "
                        "ON CONFLICT (object_label) DO UPDATE SET "
                        "felt_json = excluded.felt_json", [lbl, t, felt_json])
                else:
                    self._conn.execute(
                        "INSERT INTO cgn_grounded_objects (object_label, first_seen_ts) "
                        "VALUES (?, ?) ON CONFLICT DO NOTHING", [lbl, t])
            self._writer.submit(_write)
        except Exception as e:  # noqa: BLE001
            logger.debug("[FeltBridge] record_grounded soft-fail: %s", e)

    def is_object_grounded(self, label: str) -> bool:
        """True iff CGN has emitted `CGN_CONCEPT_GROUNDED` for this Object (RFP §7.2,
        the seam that replaces the `cgn_bridge.ensure_grounded` stub for the
        Object-level grounding query). Reads the in-memory event-sourced mirror
        ONLY — **no sync RPC into cgn_worker** (G18). Normalized to the shared key
        space."""
        lbl = normalize_label(label)
        if not lbl:
            return False
        with self._grounded_lock:
            return lbl in self._grounded

    def grounded_felt(self, label: str) -> Optional[dict]:
        """The per-concept felt centroid CGN grounded this Object under, or `None` if
        the Object is ungrounded OR matured before the felt-centroid channel existed
        (backward-compat). Reads the in-memory event-sourced mirror ONLY — no sync RPC
        (G18). Used by the producer (`_queue_felt_gaps`) for the true felt-vector
        frame_dependent comparison (CGN-felt RFP Phase C). Normalized key space."""
        lbl = normalize_label(label)
        if not lbl:
            return None
        with self._grounded_lock:
            fc = self._grounded_felt.get(lbl)
            return dict(fc) if fc else None

    # ── Phase 3 — propose-only candidate queue (INV-Syn-ENG-4) ──────────
    def queue_candidate(
        self,
        *,
        object_label: str,
        felt_state_json: str,
        c: float,
        source_engram: str,
        source_version: int,
        provenance: str = "engram_felt_gap",
        domain_hint: str = "",
        frames_json: str = "",
        status: str = "candidate",
    ) -> bool:
        """Queue a §3.4-Object-shaped felt candidate to `engram_felt_candidates`
        (RFP §7.3). PROPOSE-ONLY — this writes to synthesis.duckdb ONLY, NEVER into
        any CGN store (INV-Syn-ENG-4). Idempotent per (object_label, source_engram,
        source_version). Returns **True iff a NEW row was inserted** (so the producer
        emits the bus handoff once, not every dream). Soft-fail → False."""
        lbl = normalize_label(object_label)
        if not lbl or not source_engram:
            return False
        ts = time.time()
        try:
            def _ins() -> bool:
                exists = self._conn.execute(
                    "SELECT 1 FROM engram_felt_candidates WHERE object_label = ? "
                    "AND source_engram = ? AND source_version = ?",
                    [lbl, str(source_engram), int(source_version)]).fetchone()
                if exists is not None:
                    return False
                self._conn.execute(
                    "INSERT INTO engram_felt_candidates (object_label, "
                    "felt_state_json, c, f, time_cost, source_engram, "
                    "source_version, provenance, domain_hint, frames_json, ts, "
                    "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [lbl, str(felt_state_json or "{}"), float(c), 0.0, 0.0,
                     str(source_engram), int(source_version), str(provenance),
                     str(domain_hint or ""), str(frames_json or ""), ts,
                     str(status)])
                return True
            return bool(self._writer.submit_sync(_ins))
        except Exception as e:  # noqa: BLE001
            logger.debug("[FeltBridge] queue_candidate soft-fail: %s", e)
            return False
