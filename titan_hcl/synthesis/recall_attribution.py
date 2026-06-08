"""RecallAttribution — per-Engram citation attribution (RFP §7.E.0).

The missing source for two things at once:
  • the live `fluent` grounding axis (`engram_store.compute_axes`, today hardcoded
    `0.0` — "attribution deferred"), and
  • §7.E's offline learned-reduction reward: `(axes_at_recall, cited?)` tuples per
    Engram.

The mechanic (all OFF the chat hot path — every write/read runs on the one
`SynthesisWriter` thread, INV-Syn-28 / G21):

  1. **Membership reverse-index** (`engram_members`): consolidation already passes
     each new/bumped Engram its member promoted-thought tx_hashes
     (`derivation_evidence = [m.tx_hash for m in cluster.members]`). We persist that
     as a queryable `member_tx_hash → (engram_id, version)` index (today it only
     lands on the chain TX). Rebuildable from chain TXs (INV-2).
  2. **Live recall record** (`engram_recall_stats` + `engram_recall_events`): the
     per-turn `KNOWLEDGE_MOMENT` carries the surfaced + cited tx_hash sets (the
     tx-spine recall surfaces `item_id = tx_hash`; `cited_use` already detects the
     cited subset). We resolve each tx → its **latest** Engram version (Maker:
     latest-version-only credit) and bump `surfaced_count` / `cited_count`; every
     SURFACED Engram appends an event row (cited=true for cited, cited=false for
     surfaced-not-cited) carrying the axes snapshot at recall — both classes, so
     §7.E's combiner has a real decision boundary, not an all-positive label.
  3. **`fluent` feed**: `fluent = cited_count / (surfaced_count + k)` (NARS-smoothed,
     same form as the `verified` axis), read at the dream-boundary population
     recompute → the axis now varies → it joins the §7.D percentile-blend.

All methods are soft-fail and total — a recall-attribution failure must NEVER affect
synthesis correctness or the chat path (INV-Syn-17).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = ["RecallAttribution"]


class RecallAttribution:
    """Owns the 3 `synthesis.duckdb` attribution tables. `conn` is the
    `ActivationStore._conn` (synthesis.duckdb); `db_writer` is the
    `SynthesisWriter` that serializes every handle call (INV-Syn-28)."""

    def __init__(self, conn: Any, db_writer: Any) -> None:
        self._conn = conn
        self._writer = db_writer

    # ── Schema ──────────────────────────────────────────────────────────
    def ensure_schema(self) -> bool:
        """Create the 3 tables (idempotent). Serialized on the writer thread.
        Returns True on success; soft-fails to False (attribution disabled this
        session, synthesis otherwise unaffected)."""
        try:
            def _ddl() -> None:
                # member_tx_hash → which Engram (versioned) it composes.
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_members ("
                    " engram_id VARCHAR, version INTEGER, member_tx_hash VARCHAR,"
                    " PRIMARY KEY (engram_id, version, member_tx_hash))")
                # Per-Engram running counters + a denormalized axes cache (refreshed
                # each dream recompute) so event rows can snapshot axes_at_recall
                # without a cross-thread Kuzu read.
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_recall_stats ("
                    " engram_id VARCHAR, version INTEGER,"
                    " surfaced_count BIGINT DEFAULT 0, cited_count BIGINT DEFAULT 0,"
                    " axis_used DOUBLE DEFAULT 0, axis_verified DOUBLE DEFAULT 0,"
                    " axis_felt DOUBLE DEFAULT 0, axis_fluent DOUBLE DEFAULT 0,"
                    " updated_ts DOUBLE DEFAULT 0,"
                    " PRIMARY KEY (engram_id, version))")
                # Append-only training log for §7.E (axes_at_recall, cited?).
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_recall_events ("
                    " ts DOUBLE, engram_id VARCHAR, version INTEGER,"
                    " axis_used DOUBLE, axis_verified DOUBLE, axis_felt DOUBLE,"
                    " axis_fluent DOUBLE, cited BOOLEAN)")
            self._writer.submit_sync(_ddl)
            return True
        except Exception as e:  # noqa: BLE001 — soft-fail per INV-Syn-17
            logger.warning(
                "[RecallAttribution] schema DDL failed — attribution disabled "
                "this session (synthesis unaffected): %s", e)
            return False

    # ── Membership (written at consolidation create / version-bump) ─────
    def record_membership(
        self, engram_id: str, version: int, member_tx_hashes: Iterable[str],
    ) -> None:
        """Persist the (engram_id, version) → member tx_hashes index. Idempotent;
        fire-and-forget on the writer thread. Soft-fail."""
        txs = [str(t) for t in (member_tx_hashes or []) if t]
        if not engram_id or not txs:
            return
        try:
            def _write() -> None:
                for tx in txs:
                    self._conn.execute(
                        "INSERT INTO engram_members (engram_id, version, "
                        "member_tx_hash) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                        [str(engram_id), int(version), tx])
            self._writer.submit(_write)
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] record_membership soft-fail: %s", e)

    # ── Live recall record (off the chat hot path) ──────────────────────
    def record_recall(
        self,
        surfaced_tx_hashes: Iterable[str],
        cited_tx_hashes: Iterable[str],
        ts: float,
    ) -> None:
        """Resolve the surfaced + cited tx_hashes → their latest Engram(s) and
        record. Per-turn: each distinct Engram counts ONCE surfaced / once cited
        (not per member tx). All resolve+write happens in one writer-thread unit
        (race-free). Soft-fail."""
        surfaced = [str(t) for t in (surfaced_tx_hashes or []) if t]
        cited = [str(t) for t in (cited_tx_hashes or []) if t]
        if not surfaced and not cited:
            return
        try:
            self._writer.submit(
                lambda: self._record_recall_impl(surfaced, cited, float(ts)))
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] record_recall soft-fail: %s", e)

    def _record_recall_impl(
        self, surfaced: list[str], cited: list[str], ts: float,
    ) -> None:
        # Runs on the SynthesisWriter thread — reads + writes are serialized.
        surf_engrams = self._resolve_set(surfaced)
        cite_engrams = self._resolve_set(cited)
        # Surfaced set is the UNION (a cited Engram was necessarily surfaced).
        surfaced_all = surf_engrams | cite_engrams
        for (eid, ver) in surfaced_all:
            self._bump(eid, ver, surfaced=1, cited=0, ts=ts)
        for (eid, ver) in cite_engrams:
            self._bump(eid, ver, surfaced=0, cited=1, ts=ts)
        # Write ONE (axes_at_recall, cited?) event per SURFACED Engram — BOTH
        # classes: cited=True for the cited Engrams, cited=False for surfaced-
        # but-not-cited ones. The negatives are §7.E's decision-boundary signal:
        # without them the events table is all-positives, the citation label is
        # constant, and the learned combiner has nothing to discriminate (it
        # self-gates off forever). One row per Engram per turn (per-member dedup
        # already done by `_resolve_set`).
        for (eid, ver) in surfaced_all:
            was_cited = (eid, ver) in cite_engrams
            axes = self._cached_axes(eid, ver)
            try:
                self._conn.execute(
                    "INSERT INTO engram_recall_events (ts, engram_id, version, "
                    "axis_used, axis_verified, axis_felt, axis_fluent, cited) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [ts, eid, ver, axes[0], axes[1], axes[2], axes[3], was_cited])
            except Exception as e:  # noqa: BLE001
                logger.debug("[RecallAttribution] event insert soft-fail: %s", e)

    def _resolve_set(self, tx_hashes: list[str]) -> set[tuple[str, int]]:
        """tx_hashes → set of (engram_id, latest_version) — the latest version of
        EACH Engram any tx is a member of (Maker: latest-version-only credit)."""
        out: set[tuple[str, int]] = set()
        if not tx_hashes:
            return out
        try:
            placeholders = ", ".join("?" for _ in tx_hashes)
            qr = self._conn.execute(
                "SELECT engram_id, MAX(version) FROM engram_members "
                f"WHERE member_tx_hash IN ({placeholders}) GROUP BY engram_id",
                list(tx_hashes))
            for row in qr.fetchall():
                if row[0] is not None and row[1] is not None:
                    out.add((str(row[0]), int(row[1])))
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] resolve soft-fail: %s", e)
        return out

    def _bump(
        self, engram_id: str, version: int, *, surfaced: int, cited: int, ts: float,
    ) -> None:
        # Read-modify-write on the writer thread (serialized → race-free), so we
        # avoid duckdb ON CONFLICT DO UPDATE quirks across versions.
        try:
            qr = self._conn.execute(
                "SELECT surfaced_count, cited_count FROM engram_recall_stats "
                "WHERE engram_id = ? AND version = ?", [engram_id, version])
            row = qr.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO engram_recall_stats (engram_id, version, "
                    "surfaced_count, cited_count, updated_ts) VALUES (?, ?, ?, ?, ?)",
                    [engram_id, version, int(surfaced), int(cited), ts])
            else:
                self._conn.execute(
                    "UPDATE engram_recall_stats SET surfaced_count = ?, "
                    "cited_count = ?, updated_ts = ? WHERE engram_id = ? "
                    "AND version = ?",
                    [int(row[0]) + int(surfaced), int(row[1]) + int(cited), ts,
                     engram_id, version])
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] bump soft-fail: %s", e)

    def _cached_axes(self, engram_id: str, version: int) -> tuple[float, float, float, float]:
        """The denormalized axes snapshot for this Engram (refreshed at the dream
        recompute). (0,0,0,0) until the first post-record dream — fine, E's reward
        accumulates forward."""
        try:
            qr = self._conn.execute(
                "SELECT axis_used, axis_verified, axis_felt, axis_fluent FROM "
                "engram_recall_stats WHERE engram_id = ? AND version = ?",
                [engram_id, version])
            row = qr.fetchone()
            if row is not None:
                return (float(row[0] or 0.0), float(row[1] or 0.0),
                        float(row[2] or 0.0), float(row[3] or 0.0))
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] cached_axes soft-fail: %s", e)
        return (0.0, 0.0, 0.0, 0.0)

    # ── Dream-boundary integration ──────────────────────────────────────
    def fluent_map(self, k: float = 1.0) -> dict[tuple[str, int], float]:
        """{(engram_id, version): cited/(surfaced+k)} for every Engram with recall
        history. The §7.E.0 `fluent` axis feed for the population recompute. Empty
        dict on failure (the blend simply keeps fluent at its stored value)."""
        out: dict[tuple[str, int], float] = {}
        kf = max(0.0, float(k))
        try:
            qr = self._writer.submit_sync(lambda: self._conn.execute(
                "SELECT engram_id, version, surfaced_count, cited_count "
                "FROM engram_recall_stats").fetchall())
            for row in (qr or []):
                eid, ver = str(row[0]), int(row[1])
                surfaced, cited = int(row[2] or 0), int(row[3] or 0)
                denom = surfaced + kf
                out[(eid, ver)] = (cited / denom) if denom > 0 else 0.0
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] fluent_map soft-fail: %s", e)
        return out

    def read_training_events(
        self, limit: int = 50000,
    ) -> list[tuple[tuple[float, float, float, float], bool]]:
        """§7.E — the (axes_at_recall, cited?) training tuples from
        `engram_recall_events` (newest `limit` rows). Returns
        [((used, verified, felt, fluent), cited)]; soft-fail → []. Both classes
        are present once the §7.E.0 cited=false events accrue (else all-positive
        → the combiner's guard keeps it inactive)."""
        out: list[tuple[tuple[float, float, float, float], bool]] = []
        try:
            qr = self._writer.submit_sync(lambda: self._conn.execute(
                "SELECT axis_used, axis_verified, axis_felt, axis_fluent, cited "
                "FROM engram_recall_events ORDER BY ts DESC LIMIT ?",
                [int(limit)]).fetchall())
            for row in (qr or []):
                out.append((
                    (float(row[0] or 0.0), float(row[1] or 0.0),
                     float(row[2] or 0.0), float(row[3] or 0.0)),
                    bool(row[4])))
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] read_training_events soft-fail: %s", e)
        return out

    def update_axes_cache(self, axes_rows: Iterable[dict]) -> None:
        """Denormalize the freshly-recomputed axes into `engram_recall_stats` so
        event rows can snapshot axes_at_recall without a Kuzu read. `axes_rows`:
        [{concept_id, version, used, verified, felt, fluent}]. Soft-fail; rows with
        no recall history yet are upserted (counts default 0) so the cache is whole."""
        rows = list(axes_rows or [])
        if not rows:
            return
        try:
            def _write() -> None:
                for r in rows:
                    eid = str(r.get("concept_id") or "")
                    if not eid:
                        continue
                    ver = int(r.get("version") or 0)
                    au = float(r.get("used") or 0.0)
                    av = float(r.get("verified") or 0.0)
                    af = float(r.get("felt") or 0.0)
                    afl = float(r.get("fluent") or 0.0)
                    exists = self._conn.execute(
                        "SELECT 1 FROM engram_recall_stats WHERE engram_id = ? "
                        "AND version = ?", [eid, ver]).fetchone()
                    if exists is None:
                        self._conn.execute(
                            "INSERT INTO engram_recall_stats (engram_id, version, "
                            "axis_used, axis_verified, axis_felt, axis_fluent) "
                            "VALUES (?, ?, ?, ?, ?, ?)", [eid, ver, au, av, af, afl])
                    else:
                        self._conn.execute(
                            "UPDATE engram_recall_stats SET axis_used = ?, "
                            "axis_verified = ?, axis_felt = ?, axis_fluent = ? "
                            "WHERE engram_id = ? AND version = ?",
                            [au, av, af, afl, eid, ver])
            self._writer.submit(_write)
        except Exception as e:  # noqa: BLE001
            logger.debug("[RecallAttribution] update_axes_cache soft-fail: %s", e)
