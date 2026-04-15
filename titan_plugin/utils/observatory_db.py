"""
Observatory SQLite Database — long-term metrics, expressive archive, and event history.

Lightweight persistent storage for the Observatory Dashboard.
All writes happen via background calls; reads are lazy-loaded on user request.
Zero new dependencies — uses Python stdlib sqlite3.
"""
import json
import logging
import os
import sqlite3
import threading
import time

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

    def __init__(self, db_path: str = None):
        if db_path is None:
            base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            os.makedirs(base, exist_ok=True)
            db_path = os.path.join(base, "observatory.db")
        self._db_path = os.path.normpath(db_path)
        self._lock = threading.Lock()
        self._init_db()

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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO vital_snapshots "
                    "(ts, sovereignty_pct, life_force_pct, sol_balance, energy_state, "
                    "mood_label, mood_score, persistent_count, mempool_size, epoch_counter) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, sovereignty_pct, life_force_pct, sol_balance, energy_state,
                     mood_label, mood_score, persistent_count, mempool_size, epoch_counter),
                )
                conn.commit()
            finally:
                conn.close()

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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO trinity_snapshots "
                    "(ts, body_tensor, mind_tensor, spirit_tensor, "
                    "middle_path_loss, body_center_dist, mind_center_dist) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, json.dumps(body_tensor), json.dumps(mind_tensor),
                     json.dumps(spirit_tensor), middle_path_loss,
                     body_center_dist, mind_center_dist),
                )
                conn.commit()
            finally:
                conn.close()

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
                            except (json.JSONDecodeError, TypeError):
                                pass
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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO growth_snapshots "
                    "(ts, learning_velocity, social_density, metabolic_health, directive_alignment) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ts, learning_velocity, social_density, metabolic_health, directive_alignment),
                )
                conn.commit()
            finally:
                conn.close()

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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO expressive_archive "
                    "(ts, type, title, content, media_path, media_hash, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, type_, title, content, media_path, media_hash, meta_json),
                )
                conn.commit()
            finally:
                conn.close()

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
                        except (json.JSONDecodeError, TypeError):
                            pass
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Event Log
    # ------------------------------------------------------------------

    def record_event(self, event_type: str, summary: str = "", details: dict = None):
        """Log a significant event (meditation, guardian_block, memory_commit, etc.)."""
        ts = int(time.time())
        details_json = json.dumps(details) if details else "{}"
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO event_log (ts, event_type, summary, details) VALUES (?, ?, ?, ?)",
                    (ts, event_type, summary, details_json),
                )
                conn.commit()
            finally:
                conn.close()

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
                        except (json.JSONDecodeError, TypeError):
                            pass
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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO guardian_log (ts, tier, action, category) VALUES (?, ?, ?, ?)",
                    (ts, tier, action, category),
                )
                conn.commit()
            finally:
                conn.close()

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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO reflex_log "
                    "(ts, reflex_type, combined_confidence, body_confidence, mind_confidence, "
                    "spirit_confidence, fired, succeeded, duration_ms, error, "
                    "stimulus_topic, stimulus_intensity, vm_reward) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, reflex_type, combined_confidence, body_confidence, mind_confidence,
                     spirit_confidence, 1 if fired else 0, 1 if succeeded else 0,
                     duration_ms, error or None, stimulus_topic, stimulus_intensity, vm_reward),
                )
                conn.commit()
            finally:
                conn.close()

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

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
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
                )
                conn.commit()
            finally:
                conn.close()

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
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    rows.append(d)
                return rows
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old_data(self, max_days: int = 90):
        """Remove data older than max_days to keep the DB lean."""
        cutoff = int(time.time()) - (max_days * 86400)
        with self._lock:
            conn = self._connect()
            try:
                for table in ["vital_snapshots", "event_log", "guardian_log",
                              "trinity_snapshots", "growth_snapshots", "v4_snapshots",
                              "reflex_log"]:
                    conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
                conn.commit()
                conn.execute("VACUUM")
            finally:
                conn.close()
        logger.info("[ObservatoryDB] Pruned data older than %d days.", max_days)
