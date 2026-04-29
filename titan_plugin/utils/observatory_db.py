"""
Observatory SQLite Database — long-term metrics, expressive archive, and event history.

Lightweight persistent storage for the Observatory Dashboard.
All writes happen via background calls; reads are lazy-loaded on user request.
Zero new dependencies — uses Python stdlib sqlite3.

Per-process singleton (`get_observatory_db()`): collapses any in-process
duplicate constructions to one instance. Cross-process coherence is handled
by the canonical-mode writer daemon (see rFP_universal_sqlite_writer.md);
the singleton is defense-in-depth against accidental N-instance bugs in the
same process.
"""
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 5

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vital_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    sovereignty_pct REAL,
    life_force_pct REAL,
    sol_balance REAL,
    energy_state TEXT,
    mood_label TEXT,
    mood_score REAL,
    persistent_count INTEGER,
    mempool_size INTEGER,
    epoch_counter INTEGER
);

CREATE TABLE IF NOT EXISTS expressive_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    type TEXT NOT NULL,
    title TEXT,
    content TEXT,
    media_path TEXT,
    media_hash TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS guardian_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    tier TEXT NOT NULL,
    action TEXT,
    category TEXT
);

CREATE TABLE IF NOT EXISTS trinity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    body_tensor TEXT NOT NULL,
    mind_tensor TEXT NOT NULL,
    spirit_tensor TEXT NOT NULL,
    middle_path_loss REAL,
    body_center_dist REAL,
    mind_center_dist REAL
);

CREATE TABLE IF NOT EXISTS growth_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    learning_velocity REAL,
    social_density REAL,
    metabolic_health REAL,
    directive_alignment REAL
);

CREATE TABLE IF NOT EXISTS v4_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    sphere_clocks TEXT,
    resonance TEXT,
    unified_spirit TEXT,
    consciousness TEXT,
    impulse_engine TEXT,
    filter_down TEXT,
    middle_path_loss REAL,
    great_pulse_count INTEGER DEFAULT 0,
    big_pulse_count INTEGER DEFAULT 0,
    spirit_velocity REAL,
    spirit_stale INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reflex_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    reflex_type TEXT NOT NULL,
    combined_confidence REAL,
    body_confidence REAL DEFAULT 0.0,
    mind_confidence REAL DEFAULT 0.0,
    spirit_confidence REAL DEFAULT 0.0,
    fired INTEGER DEFAULT 1,
    succeeded INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0.0,
    error TEXT,
    stimulus_topic TEXT,
    stimulus_intensity REAL DEFAULT 0.0,
    vm_reward REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS neuromod_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    da REAL, sht REAL, ne REAL, ach REAL, endorphin REAL, gaba REAL,
    emotion TEXT, confidence REAL,
    da_sensitivity REAL, sht_sensitivity REAL
);

CREATE TABLE IF NOT EXISTS hormonal_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    program TEXT NOT NULL,
    level REAL, fire_count INTEGER, refractory REAL, threshold REAL
);

CREATE TABLE IF NOT EXISTS expression_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    composite TEXT NOT NULL,
    urge REAL, fire_count INTEGER
);

CREATE TABLE IF NOT EXISTS dreaming_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    is_dreaming INTEGER, fatigue REAL,
    cycle_count INTEGER, recovery_pct REAL,
    trigger TEXT, duration_epochs INTEGER
);

CREATE TABLE IF NOT EXISTS training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    transitions INTEGER, train_steps INTEGER, maturity REAL,
    memory_count INTEGER, mempool_count INTEGER,
    experience_count INTEGER, vocabulary_count INTEGER
);

CREATE TABLE IF NOT EXISTS clock_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    clock TEXT NOT NULL,
    radius REAL, pulse_count INTEGER, streak INTEGER, phase REAL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vital_ts ON vital_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_expressive_ts ON expressive_archive(ts);
CREATE INDEX IF NOT EXISTS idx_expressive_type ON expressive_archive(type);
CREATE INDEX IF NOT EXISTS idx_event_ts ON event_log(ts);
CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_guardian_ts ON guardian_log(ts);
CREATE INDEX IF NOT EXISTS idx_trinity_ts ON trinity_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_growth_ts ON growth_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_v4_ts ON v4_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_reflex_ts ON reflex_log(ts);
CREATE INDEX IF NOT EXISTS idx_reflex_type ON reflex_log(reflex_type);
CREATE INDEX IF NOT EXISTS idx_neuromod_ts ON neuromod_history(ts);
CREATE INDEX IF NOT EXISTS idx_hormonal_ts ON hormonal_history(ts);
CREATE INDEX IF NOT EXISTS idx_expression_ts ON expression_history(ts);
CREATE INDEX IF NOT EXISTS idx_dreaming_ts ON dreaming_history(ts);
CREATE INDEX IF NOT EXISTS idx_training_ts ON training_history(ts);
CREATE INDEX IF NOT EXISTS idx_clock_ts ON clock_history(ts);
"""


class ObservatoryDB:
    """Thread-safe SQLite wrapper for Observatory long-term storage."""

    def __init__(self, db_path: str = None, writer_client=None):
        # Track whether the caller passed an explicit db_path. Production
        # callers leave it None; test callers pass a tmp file. Only the
        # explicit case needs the path-isolation guard against silently
        # diverging from the writer's configured DB. (rFP_universal_sqlite_
        # writer 2026-04-27 hot-fix: the previous guard fired in production
        # too because cfg.db_path is relative `data/observatory.db` while
        # self._db_path resolves to absolute via __file__, so the comparison
        # always failed → writer client silently skipped → all writes went
        # direct-to-disk → multi-process lock contention. This is what
        # caused BUG-TRINITY-SNAPSHOT-DB-LOCKED to keep firing post-Phase-2.)
        explicit_db_path = db_path is not None
        if db_path is None:
            base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            os.makedirs(base, exist_ok=True)
            db_path = os.path.join(base, "observatory.db")
        self._db_path = os.path.normpath(db_path)
        self._lock = threading.Lock()
        self._init_db()
        # rFP_observatory_writer_service Phase 1: optional writer client.
        # If [persistence_observatory].enabled = true, we route writes through
        # a second IMW-pattern daemon to relieve multi-process lock contention.
        # The client handles all mode logic (disabled/shadow/dual/canonical/
        # hybrid + per-table cutover via tables_canonical); ObservatoryDB
        # just calls _route_write() and lets the client decide direct vs writer.
        # Caller may pass an explicit client; otherwise we auto-construct from
        # [persistence_observatory] section so existing instantiation sites
        # don't need code changes.
        self._writer = writer_client
        if self._writer is None:
            try:
                from titan_plugin.persistence.config import IMWConfig
                from titan_plugin.persistence.writer_client import (
                    InnerMemoryWriterClient,
                )
                cfg = IMWConfig.from_titan_config_section("persistence_observatory")
                # Path-isolation safety — only checked when caller passed an
                # explicit db_path. The check uses realpath to handle
                # absolute-vs-relative + symlinks; tested against the
                # writer's configured DB (also realpath'd from cwd). Two
                # paths that resolve to the same file are equivalent — only
                # genuinely different files (e.g., tmp test files) trip the
                # guard. Production case (db_path=None) skips the guard
                # entirely because the constructor's default IS the writer's
                # configured DB by construction.
                if explicit_db_path and cfg.db_path:
                    self_real = os.path.realpath(self._db_path)
                    cfg_real = os.path.realpath(cfg.db_path)
                    if self_real != cfg_real:
                        logger.info(
                            "[ObservatoryDB] db_path %s != configured "
                            "writer path %s — writer client skipped "
                            "(using direct writes for path isolation)",
                            self._db_path, cfg.db_path)
                        return
                if cfg.enabled and cfg.mode != "disabled":
                    self._writer = InnerMemoryWriterClient(
                        cfg, caller_name="observatory_db")
                    logger.info(
                        "[ObservatoryDB] Routed via observatory_writer "
                        "(mode=%s, canonical=%s)",
                        cfg.mode, cfg.tables_canonical or "<none>")
            except Exception as e:
                # Defensive: never break ObservatoryDB if writer setup fails.
                # Direct path remains as fallback.
                logger.warning(
                    "[ObservatoryDB] writer client unavailable, "
                    "using direct writes: %s", e)
                self._writer = None

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            try:
                conn.executescript(_SCHEMA_SQL)
                # Check/set schema version
                cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cur.fetchone()
                if row is None:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
                elif row[0] < _SCHEMA_VERSION:
                    conn.execute("UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,))
                conn.commit()
            finally:
                conn.close()
        logger.info("[ObservatoryDB] Initialized at %s (schema v%d)", self._db_path, _SCHEMA_VERSION)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _route_write(self, sql: str, params: tuple, *, table: str) -> None:
        """Single write helper.

        Phase 4 sunset (2026-04-21, post 30-min canonical soak with zero errors
        across 25,107 writes): when writer is enabled, it is the SOLE write
        path. Errors propagate to caller — no silent direct-path fallback.
        Loud failure beats silent data divergence.

        Direct path remains in place for the writer-disabled deployment case
        (e.g. someone running ObservatoryDB without the writer service).
        """
        if self._writer is not None:
            self._writer.write(sql, params, table=table)
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(sql, params)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Vital Snapshots
    # ------------------------------------------------------------------

    def record_vital_snapshot(
        self,
        sovereignty_pct: float = 0.0,
        life_force_pct: float = -1.0,
        sol_balance: float = 0.0,
        energy_state: str = "UNKNOWN",
        mood_label: str = "Unknown",
        mood_score: float = 0.0,
        persistent_count: int = 0,
        mempool_size: int = 0,
        epoch_counter: int = 0,
    ):
        """Record a point-in-time vital snapshot. Called every 15 minutes."""
        ts = int(time.time())
        self._route_write(
            "INSERT INTO vital_snapshots "
            "(ts, sovereignty_pct, life_force_pct, sol_balance, energy_state, "
            "mood_label, mood_score, persistent_count, mempool_size, epoch_counter) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, sovereignty_pct, life_force_pct, sol_balance, energy_state,
             mood_label, mood_score, persistent_count, mempool_size, epoch_counter),
            table="vital_snapshots",
        )

    def get_vital_history(self, days: int = 7, metrics: list = None) -> list[dict]:
        """
        Fetch vital snapshot history.

        Args:
            days: How many days back to query.
            metrics: Optional list of column names to return. None = all.

        Returns:
            List of dicts with ts + requested metrics, ordered by time ascending.
        """
        cutoff = int(time.time()) - (days * 86400)
        all_cols = [
            "ts", "sovereignty_pct", "life_force_pct", "sol_balance", "energy_state",
            "mood_label", "mood_score", "persistent_count", "mempool_size", "epoch_counter",
        ]
        if metrics:
            cols = ["ts"] + [m for m in metrics if m in all_cols and m != "ts"]
        else:
            cols = all_cols

        col_str = ", ".join(cols)
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    f"SELECT {col_str} FROM vital_snapshots WHERE ts >= ? ORDER BY ts ASC",
                    (cutoff,),
                )
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Trinity Snapshots (V3 Body/Mind/Spirit tensor time-series)
    # ------------------------------------------------------------------

    def record_trinity_snapshot(
        self,
        body_tensor: list,
        mind_tensor: list,
        spirit_tensor: list,
        middle_path_loss: float = 0.0,
        body_center_dist: float = 0.0,
        mind_center_dist: float = 0.0,
    ):
        """Record a Trinity tensor snapshot. Called every trinity_snapshot_interval seconds."""
        ts = int(time.time())
        self._route_write(
            "INSERT INTO trinity_snapshots "
            "(ts, body_tensor, mind_tensor, spirit_tensor, "
            "middle_path_loss, body_center_dist, mind_center_dist) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, json.dumps(body_tensor), json.dumps(mind_tensor),
             json.dumps(spirit_tensor), middle_path_loss,
             body_center_dist, mind_center_dist),
            table="trinity_snapshots",
        )

    def get_trinity_history(self, hours: int = 24) -> list[dict]:
        """
        Fetch Trinity tensor time-series.

        Args:
            hours: How many hours back to query.

        Returns:
            List of dicts with ts, body/mind/spirit tensors, loss, distances.
        """
        cutoff = int(time.time()) - (hours * 3600)
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    "SELECT * FROM trinity_snapshots WHERE ts >= ? ORDER BY ts ASC",
                    (cutoff,),
                )
                rows = []
                for row in cur.fetchall():
                    d = dict(row)
                    for key in ("body_tensor", "mind_tensor", "spirit_tensor"):
                        if d.get(key) and isinstance(d[key], str):
                            try:
                                d[key] = json.loads(d[key])
                            except (json.JSONDecodeError, TypeError) as _swallow_exc:
                                swallow_warn('[utils.observatory_db] ObservatoryDB.get_trinity_history: d[key] = json.loads(d[key])', _swallow_exc,
                                             key='utils.observatory_db.ObservatoryDB.get_trinity_history.line390', throttle=100)
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Growth Snapshots (learning velocity, social density, etc.)
    # ------------------------------------------------------------------

    def record_growth_snapshot(
        self,
        learning_velocity: float = 0.0,
        social_density: float = 0.0,
        metabolic_health: float = 0.0,
        directive_alignment: float = 0.0,
    ):
        """Record growth metrics. Called alongside Trinity snapshots."""
        ts = int(time.time())
        self._route_write(
            "INSERT INTO growth_snapshots "
            "(ts, learning_velocity, social_density, metabolic_health, directive_alignment) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, learning_velocity, social_density, metabolic_health, directive_alignment),
            table="growth_snapshots",
        )

    def get_growth_history(self, days: int = 7) -> list[dict]:
        """Fetch growth metrics history, ordered chronologically."""
        cutoff = int(time.time()) - (days * 86400)
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    "SELECT * FROM growth_snapshots WHERE ts >= ? ORDER BY ts ASC",
                    (cutoff,),
                )
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Expressive Archive
    # ------------------------------------------------------------------

    def record_expressive(
        self,
        type_: str,
        title: str = "",
        content: str = "",
        media_path: str = "",
        media_hash: str = "",
        metadata: dict = None,
    ):
        """Archive an expressive output (art, haiku, audio, x_post)."""
        ts = int(time.time())
        meta_json = json.dumps(metadata) if metadata else "{}"
        self._route_write(
            "INSERT INTO expressive_archive "
            "(ts, type, title, content, media_path, media_hash, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, type_, title, content, media_path, media_hash, meta_json),
            table="expressive_archive",
        )

    def get_expressive_archive(
        self, type_: str = None, limit: int = 20, offset: int = 0,
    ) -> list[dict]:
        """Fetch expressive archive entries, newest first."""
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                if type_:
                    cur = conn.execute(
                        "SELECT * FROM expressive_archive WHERE type = ? "
                        "ORDER BY ts DESC LIMIT ? OFFSET ?",
                        (type_, limit, offset),
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM expressive_archive ORDER BY ts DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                rows = []
                for row in cur.fetchall():
                    d = dict(row)
                    # Parse metadata JSON
                    if d.get("metadata"):
                        try:
                            d["metadata"] = json.loads(d["metadata"])
                        except (json.JSONDecodeError, TypeError) as _swallow_exc:
                            swallow_warn("[utils.observatory_db] ObservatoryDB.get_expressive_archive: d['metadata'] = json.loads(d['metadata'])", _swallow_exc,
                                         key='utils.observatory_db.ObservatoryDB.get_expressive_archive.line483', throttle=100)
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Event Log
    # ------------------------------------------------------------------

    def record_event(self, event_type: str, summary: str = "", details: dict = None):
        """Log a significant event (meditation, guardian_block, memory_commit, etc.).

        rFP_observatory_writer_service: event_log is the **first canary table**
        for Phase 2 canonical cutover (highest contention → highest payoff).
        """
        ts = int(time.time())
        details_json = json.dumps(details) if details else "{}"
        self._route_write(
            "INSERT INTO event_log (ts, event_type, summary, details) VALUES (?, ?, ?, ?)",
            (ts, event_type, summary, details_json),
            table="event_log",
        )

    def get_events(
        self, event_type: str = None, limit: int = 50, offset: int = 0,
    ) -> list[dict]:
        """Fetch event log entries, newest first."""
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                if event_type:
                    cur = conn.execute(
                        "SELECT * FROM event_log WHERE event_type = ? "
                        "ORDER BY ts DESC LIMIT ? OFFSET ?",
                        (event_type, limit, offset),
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM event_log ORDER BY ts DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                rows = []
                for row in cur.fetchall():
                    d = dict(row)
                    if d.get("details"):
                        try:
                            d["details"] = json.loads(d["details"])
                        except (json.JSONDecodeError, TypeError) as _swallow_exc:
                            swallow_warn("[utils.observatory_db] ObservatoryDB.get_events: d['details'] = json.loads(d['details'])", _swallow_exc,
                                         key='utils.observatory_db.ObservatoryDB.get_events.line533', throttle=100)
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Guardian Log
    # ------------------------------------------------------------------

    def record_guardian_action(self, tier: str, action: str, category: str = ""):
        """Log a guardian safety intervention (never stores the blocked content)."""
        ts = int(time.time())
        self._route_write(
            "INSERT INTO guardian_log (ts, tier, action, category) VALUES (?, ?, ?, ?)",
            (ts, tier, action, category),
            table="guardian_log",
        )

    def get_guardian_log(self, limit: int = 50) -> list[dict]:
        """Fetch recent guardian actions."""
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    "SELECT * FROM guardian_log ORDER BY ts DESC LIMIT ?", (limit,)
                )
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Reflex Log (R6: Sovereign Reflex Observability)
    # ------------------------------------------------------------------

    def record_reflex(
        self,
        reflex_type: str,
        combined_confidence: float = 0.0,
        body_confidence: float = 0.0,
        mind_confidence: float = 0.0,
        spirit_confidence: float = 0.0,
        fired: bool = True,
        succeeded: bool = False,
        duration_ms: float = 0.0,
        error: str = "",
        stimulus_topic: str = "",
        stimulus_intensity: float = 0.0,
        vm_reward: float = 0.0,
    ):
        """Record a reflex firing event for observability."""
        ts = int(time.time())
        self._route_write(
            "INSERT INTO reflex_log "
            "(ts, reflex_type, combined_confidence, body_confidence, mind_confidence, "
            "spirit_confidence, fired, succeeded, duration_ms, error, "
            "stimulus_topic, stimulus_intensity, vm_reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, reflex_type, combined_confidence, body_confidence, mind_confidence,
             spirit_confidence, 1 if fired else 0, 1 if succeeded else 0,
             duration_ms, error or None, stimulus_topic, stimulus_intensity, vm_reward),
            table="reflex_log",
        )

    def get_reflex_history(self, hours: int = 24, reflex_type: str = None, limit: int = 200) -> list[dict]:
        """
        Fetch reflex firing history.

        Args:
            hours: How many hours back to query.
            reflex_type: Optional filter by reflex type.
            limit: Max rows to return.

        Returns:
            List of reflex log dicts, newest first.
        """
        cutoff = int(time.time()) - (hours * 3600)
        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                if reflex_type:
                    cur = conn.execute(
                        "SELECT * FROM reflex_log WHERE ts >= ? AND reflex_type = ? "
                        "ORDER BY ts DESC LIMIT ?",
                        (cutoff, reflex_type, limit),
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM reflex_log WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                        (cutoff, limit),
                    )
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    def get_reflex_stats(self, hours: int = 24) -> dict:
        """
        Aggregate reflex statistics: fire counts, success rates, avg confidence per type.
        """
        cutoff = int(time.time()) - (hours * 3600)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT reflex_type, COUNT(*) as fires, "
                    "SUM(succeeded) as successes, "
                    "AVG(combined_confidence) as avg_confidence, "
                    "AVG(duration_ms) as avg_duration_ms, "
                    "AVG(vm_reward) as avg_reward "
                    "FROM reflex_log WHERE ts >= ? "
                    "GROUP BY reflex_type ORDER BY fires DESC",
                    (cutoff,),
                )
                stats = {}
                for row in cur.fetchall():
                    stats[row[0]] = {
                        "fires": row[1],
                        "successes": row[2],
                        "success_rate": round(row[2] / row[1], 3) if row[1] > 0 else 0.0,
                        "avg_confidence": round(row[3], 4) if row[3] else 0.0,
                        "avg_duration_ms": round(row[4], 1) if row[4] else 0.0,
                        "avg_reward": round(row[5], 4) if row[5] else 0.0,
                    }
                # Total
                cur2 = conn.execute(
                    "SELECT COUNT(*), SUM(succeeded), AVG(vm_reward) "
                    "FROM reflex_log WHERE ts >= ?",
                    (cutoff,),
                )
                total = cur2.fetchone()
                return {
                    "per_type": stats,
                    "total_fires": total[0] or 0,
                    "total_successes": total[1] or 0,
                    "avg_reward": round(total[2], 4) if total[2] else 0.0,
                    "hours": hours,
                }
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # V4 Time Awareness Snapshots
    # ------------------------------------------------------------------

    def record_v4_snapshot(
        self,
        sphere_clocks: dict = None,
        resonance: dict = None,
        unified_spirit: dict = None,
        consciousness: dict = None,
        impulse_engine: dict = None,
        filter_down: dict = None,
        middle_path_loss: float = 0.0,
    ):
        """Record a V4 Time Awareness snapshot. Called alongside Trinity snapshots."""
        ts = int(time.time())

        # Extract key scalars for efficient querying
        great_pulse_count = 0
        big_pulse_count = 0
        if resonance:
            great_pulse_count = resonance.get("great_pulse_count", 0)
            for pair in resonance.get("pairs", {}).values():
                big_pulse_count += pair.get("big_pulse_count", 0)

        spirit_velocity = 1.0
        spirit_stale = 0
        if unified_spirit:
            spirit_velocity = unified_spirit.get("velocity", 1.0)
            spirit_stale = 1 if unified_spirit.get("is_stale", False) else 0

        self._route_write(
            "INSERT INTO v4_snapshots "
            "(ts, sphere_clocks, resonance, unified_spirit, consciousness, "
            "impulse_engine, filter_down, middle_path_loss, "
            "great_pulse_count, big_pulse_count, spirit_velocity, spirit_stale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts,
             json.dumps(sphere_clocks) if sphere_clocks else None,
             json.dumps(resonance) if resonance else None,
             json.dumps(unified_spirit) if unified_spirit else None,
             json.dumps(consciousness) if consciousness else None,
             json.dumps(impulse_engine) if impulse_engine else None,
             json.dumps(filter_down) if filter_down else None,
             middle_path_loss,
             great_pulse_count, big_pulse_count,
             spirit_velocity, spirit_stale),
            table="v4_snapshots",
        )

    def get_v4_history(self, hours: int = 24, scalars_only: bool = False) -> list[dict]:
        """
        Fetch V4 Time Awareness time-series.

        Args:
            hours: How many hours back to query.
            scalars_only: If True, return only ts + scalar columns (faster for graphs).

        Returns:
            List of dicts ordered by time ascending.
        """
        cutoff = int(time.time()) - (hours * 3600)

        if scalars_only:
            cols = "ts, middle_path_loss, great_pulse_count, big_pulse_count, spirit_velocity, spirit_stale"
        else:
            cols = "*"

        with self._lock:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    f"SELECT {cols} FROM v4_snapshots WHERE ts >= ? ORDER BY ts ASC",
                    (cutoff,),
                )
                rows = []
                for row in cur.fetchall():
                    d = dict(row)
                    if not scalars_only:
                        # Parse JSON fields
                        for key in ("sphere_clocks", "resonance", "unified_spirit",
                                    "consciousness", "impulse_engine", "filter_down"):
                            if d.get(key) and isinstance(d[key], str):
                                try:
                                    d[key] = json.loads(d[key])
                                except (json.JSONDecodeError, TypeError) as _swallow_exc:
                                    swallow_warn('[utils.observatory_db] ObservatoryDB.get_v4_history: d[key] = json.loads(d[key])', _swallow_exc,
                                                 key='utils.observatory_db.ObservatoryDB.get_v4_history.line762', throttle=100)
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old_data(self, max_days: int = 90):
        """Remove data older than max_days to keep the DB lean.

        Two failure modes this guards against (2026-04-21 fix):
        1. **VACUUM-blocks-writer**: The original implementation ran a
           DB-wide VACUUM at the end of prune. On a 1.88 GB observatory.db,
           VACUUM holds an exclusive lock for minutes — fatal for the new
           observatory_writer service. We skip VACUUM when the writer is
           enabled (writer service can do maintenance separately).
        2. **DELETE-races-writer**: With writer active, direct DELETEs
           contend for the lock. We route DELETEs through the writer so
           it serializes them with INSERTs.

        Pre-existing observation: spirit_worker calls this method after
        EVERY dream cycle (~22 min on T1/T2/T3) — far more frequent than
        necessary. A separate follow-up should cap to once per day.
        """
        cutoff = int(time.time()) - (max_days * 86400)
        tables = ["vital_snapshots", "event_log", "guardian_log",
                  "trinity_snapshots", "growth_snapshots", "v4_snapshots",
                  "reflex_log"]
        if self._writer is not None:
            # Route DELETEs through writer so they serialize with INSERTs.
            # Skip VACUUM — writer-side maintenance handles it (and a 1.88GB
            # VACUUM under load is exactly what triggered the T3 degradation
            # incident on 2026-04-21).
            for table in tables:
                self._route_write(
                    f"DELETE FROM {table} WHERE ts < ?",
                    (cutoff,),
                    table=table,
                )
            logger.info(
                "[ObservatoryDB] Pruned data older than %d days (via writer; VACUUM skipped).",
                max_days)
            return
        # Direct path (writer disabled) — original behavior including VACUUM
        with self._lock:
            conn = self._connect()
            try:
                for table in tables:
                    conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
                conn.commit()
                conn.execute("VACUUM")
            finally:
                conn.close()
        logger.info("[ObservatoryDB] Pruned data older than %d days.", max_days)


# ──────────────────────────────────────────────────────────────────────
# Per-process singleton accessor — rFP_universal_sqlite_writer Phase 2
# ──────────────────────────────────────────────────────────────────────

_singleton_lock = threading.Lock()
_singleton_instance: Optional["ObservatoryDB"] = None


def get_observatory_db(db_path: Optional[str] = None) -> "ObservatoryDB":
    """Return the per-process ObservatoryDB singleton.

    Multiple call sites historically constructed `ObservatoryDB()` directly,
    producing N parallel SQLite connections + N writer-clients per process.
    Closing `BUG-TRINITY-SNAPSHOT-DB-LOCKED` requires a single instance per
    process (cross-process coherence is then provided by the canonical-mode
    writer daemon — see [persistence_observatory] in config.toml).

    `db_path`: if specified AND differs from the existing singleton's path,
    a new instance is constructed and returned WITHOUT replacing the
    singleton (test isolation: temp files stay isolated from production).
    Pass None in production code.
    """
    global _singleton_instance
    if db_path is not None:
        # Test/explicit-path branch: never poison the singleton.
        # If the existing singleton happens to be at the same path, reuse it.
        existing = _singleton_instance
        if existing is not None:
            try:
                if os.path.normpath(db_path) == existing._db_path:
                    return existing
            except AttributeError as _swallow_exc:
                swallow_warn(
                    "[utils.observatory_db] get_observatory_db: existing._db_path",
                    _swallow_exc,
                    key="utils.observatory_db.get_observatory_db.existing_db_path",
                    throttle=100,
                )
        return ObservatoryDB(db_path=db_path)
    # Production path: lazy-init the per-process singleton.
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ObservatoryDB()
    return _singleton_instance


def reset_observatory_db_singleton_for_tests() -> None:
    """Reset the singleton so a test can get a fresh instance.

    DO NOT call from production code — only from pytest fixtures that need
    to verify the singleton's lazy init or swap in a new db_path.
    """
    global _singleton_instance
    with _singleton_lock:
        _singleton_instance = None
