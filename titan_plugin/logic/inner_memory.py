"""
titan_plugin/logic/inner_memory.py — Dual-Layer Inner Memory System.

Records Titan's INTERNAL experiences for self-learning:
- Inner Layer: hormone snapshots, program fires, cross-talk cascades, topology
- Outer Layer: action chains, creative works, expression quality

Provides temporal queries critical for hormonal pressure system:
- time_since_last("explore") → feeds CURIOSITY boredom buildup
- get_action_patterns() → learn which actions produce best outcomes
- get_recent_fires() → understand program firing patterns

This is the foundation for genuine self-learning: Titan can observe
patterns in his own development and use them to refine behavior.
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default time returned when no event found (1 hour in seconds)
DEFAULT_TIME_SINCE = 3600.0


class InnerMemoryStore:
    """Titan's inner experience memory — mirrors Inner/Outer Trinity structure."""

    def __init__(self, db_path: str = "./data/inner_memory.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers + writer
        self._conn.execute("PRAGMA busy_timeout=5000")  # wait 5s on lock instead of failing
        self._conn.execute("PRAGMA cache_size = -16000")   # 16MB cap (was unbounded on 362MB DB)
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()
        count = self._count_all()
        if count > 0:
            logger.info("[InnerMemory] Loaded %d records from %s", count, db_path)
        else:
            logger.info("[InnerMemory] Initialized empty store at %s", db_path)

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn.cursor()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS hormone_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    epoch_id INTEGER,
                    levels TEXT NOT NULL,
                    thresholds TEXT NOT NULL,
                    refractory TEXT NOT NULL,
                    fired_programs TEXT,
                    stimuli TEXT
                );

                CREATE TABLE IF NOT EXISTS program_fires (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    program TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    pressure_at_fire REAL NOT NULL,
                    threshold_at_fire REAL NOT NULL,
                    stimulus_value REAL,
                    cross_talk_snapshot TEXT,
                    trinity_body TEXT,
                    trinity_mind TEXT,
                    trinity_spirit TEXT
                );

                CREATE TABLE IF NOT EXISTS action_chains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    impulse_id INTEGER,
                    triggering_program TEXT,
                    posture TEXT NOT NULL,
                    helper TEXT NOT NULL,
                    params TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    score REAL,
                    reasoning TEXT,
                    trinity_before TEXT,
                    trinity_after TEXT,
                    trinity_delta REAL,
                    epoch_id INTEGER
                );

                CREATE TABLE IF NOT EXISTS creative_works (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    work_type TEXT NOT NULL,
                    file_path TEXT,
                    triggering_program TEXT,
                    posture TEXT,
                    assessment_score REAL,
                    hormone_level_at_creation REAL,
                    trinity_snapshot TEXT
                );

                CREATE TABLE IF NOT EXISTS event_markers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    program TEXT,
                    details TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_hormone_ts
                    ON hormone_snapshots(timestamp);
                CREATE INDEX IF NOT EXISTS idx_fire_program
                    ON program_fires(program, timestamp);
                CREATE INDEX IF NOT EXISTS idx_action_helper
                    ON action_chains(helper, timestamp);
                CREATE INDEX IF NOT EXISTS idx_event_type
                    ON event_markers(event_type, timestamp);
                CREATE INDEX IF NOT EXISTS idx_creative_type
                    ON creative_works(work_type, timestamp);

                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL UNIQUE,
                    word_type TEXT NOT NULL,
                    stage INTEGER NOT NULL DEFAULT 1,
                    felt_tensor TEXT,
                    hormone_pattern TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    times_encountered INTEGER NOT NULL DEFAULT 0,
                    times_produced INTEGER NOT NULL DEFAULT 0,
                    last_encountered REAL,
                    learning_phase TEXT NOT NULL DEFAULT 'unlearned',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vocab_word
                    ON vocabulary(word);
                CREATE INDEX IF NOT EXISTS idx_vocab_phase
                    ON vocabulary(learning_phase);
                CREATE INDEX IF NOT EXISTS idx_vocab_stage
                    ON vocabulary(stage);

                CREATE TABLE IF NOT EXISTS composition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    epoch_id INTEGER,
                    level INTEGER,
                    template TEXT,
                    sentence TEXT,
                    words_used TEXT,
                    confidence REAL,
                    slots_filled INTEGER,
                    slots_total INTEGER,
                    intent TEXT,
                    stage TEXT,
                    state_resonance REAL,
                    pre_state TEXT,
                    post_state TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_comp_timestamp
                    ON composition_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_comp_stage
                    ON composition_history(stage);

                CREATE TABLE IF NOT EXISTS kin_encounters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    kin_pubkey TEXT NOT NULL,
                    resonance REAL,
                    my_emotion TEXT,
                    kin_emotion TEXT,
                    exchange_type TEXT,
                    great_kin_pulse INTEGER DEFAULT 0,
                    epoch_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS kin_profiles (
                    pubkey TEXT PRIMARY KEY,
                    name TEXT,
                    first_encounter_ts REAL,
                    last_encounter_ts REAL,
                    encounter_count INTEGER DEFAULT 0,
                    avg_resonance REAL DEFAULT 0.0,
                    great_kin_pulses INTEGER DEFAULT 0,
                    relationship_label TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_kin_enc_pubkey
                    ON kin_encounters(kin_pubkey);
                CREATE INDEX IF NOT EXISTS idx_kin_enc_ts
                    ON kin_encounters(timestamp);

                CREATE TABLE IF NOT EXISTS visual_autobiography (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    epoch_id INTEGER,
                    journey_5d TEXT NOT NULL,
                    resonance_5d TEXT NOT NULL,
                    semantic_summary TEXT,
                    source TEXT,
                    filename TEXT,
                    inner_state_hash TEXT,
                    emotional_context TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_visual_auto_ts
                    ON visual_autobiography(timestamp);
                CREATE INDEX IF NOT EXISTS idx_visual_auto_epoch
                    ON visual_autobiography(epoch_id);
            """)
            self._conn.commit()

    def _count_all(self) -> int:
        with self._lock:
            c = self._conn.cursor()
            total = 0
            for table in ("hormone_snapshots", "program_fires", "action_chains",
                          "creative_works", "event_markers", "vocabulary"):
                try:
                    c.execute(f"SELECT COUNT(*) FROM {table}")
                    total += c.fetchone()[0]
                except Exception:
                    pass
            return total

    # ── Inner Layer: Write API ───────────────────────────────────────

    def record_hormone_snapshot(
        self,
        epoch_id: int,
        levels: dict,
        thresholds: dict,
        refractory: dict,
        fired: list,
        stimuli: dict,
    ) -> None:
        """Record hormone state at a consciousness epoch."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO hormone_snapshots "
                "(timestamp, epoch_id, levels, thresholds, refractory, "
                "fired_programs, stimuli) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, epoch_id,
                 json.dumps(levels), json.dumps(thresholds),
                 json.dumps(refractory), json.dumps(fired),
                 json.dumps(stimuli)))
            self._conn.commit()

    def record_program_fire(
        self,
        program: str,
        layer: str,
        intensity: float,
        pressure: float,
        threshold: float,
        stimulus: float = 0.0,
        cross_talk: Optional[dict] = None,
        body: Optional[list] = None,
        mind: Optional[list] = None,
        spirit: Optional[list] = None,
    ) -> None:
        """Record a neural program firing with full context."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO program_fires "
                "(timestamp, program, layer, intensity, pressure_at_fire, "
                "threshold_at_fire, stimulus_value, cross_talk_snapshot, "
                "trinity_body, trinity_mind, trinity_spirit) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, program, layer, intensity, pressure, threshold,
                 stimulus,
                 json.dumps(cross_talk) if cross_talk else None,
                 json.dumps(body) if body else None,
                 json.dumps(mind) if mind else None,
                 json.dumps(spirit) if spirit else None))
            self._conn.commit()

    # ── Outer Layer: Write API ───────────────────────────────────────

    def record_action_chain(
        self,
        impulse_id: int,
        triggering_program: str,
        posture: str,
        helper: str,
        success: bool,
        score: float = 0.0,
        reasoning: str = "",
        params: Optional[dict] = None,
        trinity_before: Optional[dict] = None,
        trinity_after: Optional[dict] = None,
        trinity_delta: float = 0.0,
        epoch_id: int = 0,
    ) -> None:
        """Record a full impulse→intent→action→outcome chain."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO action_chains "
                "(timestamp, impulse_id, triggering_program, posture, helper, "
                "params, success, score, reasoning, trinity_before, "
                "trinity_after, trinity_delta, epoch_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, impulse_id, triggering_program, posture, helper,
                 json.dumps(params) if params else None,
                 1 if success else 0, score, reasoning,
                 json.dumps(trinity_before) if trinity_before else None,
                 json.dumps(trinity_after) if trinity_after else None,
                 trinity_delta, epoch_id))
            self._conn.commit()

    def record_creative_work(
        self,
        work_type: str,
        file_path: str = "",
        triggering_program: str = "",
        posture: str = "",
        assessment_score: float = 0.0,
        hormone_level: float = 0.0,
        trinity_snapshot: Optional[dict] = None,
    ) -> None:
        """Record a creative work with its generative context."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO creative_works "
                "(timestamp, work_type, file_path, triggering_program, "
                "posture, assessment_score, hormone_level_at_creation, "
                "trinity_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now, work_type, file_path, triggering_program, posture,
                 assessment_score, hormone_level,
                 json.dumps(trinity_snapshot) if trinity_snapshot else None))
            self._conn.commit()

    def record_event(
        self,
        event_type: str,
        program: str = "",
        details: Optional[dict] = None,
    ) -> None:
        """Record a time-based event marker for temporal queries."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO event_markers "
                "(timestamp, event_type, program, details) "
                "VALUES (?, ?, ?, ?)",
                (now, event_type, program,
                 json.dumps(details) if details else None))
            self._conn.commit()

    # ── Visual Autobiography ─────────────────────────────────────────

    def record_visual_journey(
        self,
        journey_5d: list,
        resonance_5d: list,
        semantic_summary: dict = None,
        source: str = "self",
        filename: str = None,
        epoch_id: int = 0,
        inner_state_hash: str = "",
        emotional_context: dict = None,
    ) -> None:
        """Record a visual perception snapshot for long-term autobiography."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO visual_autobiography "
                "(timestamp, epoch_id, journey_5d, resonance_5d, "
                "semantic_summary, source, filename, inner_state_hash, "
                "emotional_context) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, epoch_id,
                 json.dumps(journey_5d),
                 json.dumps(resonance_5d),
                 json.dumps(semantic_summary) if semantic_summary else None,
                 source, filename, inner_state_hash,
                 json.dumps(emotional_context) if emotional_context else None))
            self._conn.commit()

    # ── Temporal Query API (Critical for Hormones) ───────────────────

    def time_since_last(self, event_type: str) -> float:
        """
        Seconds since last event of this type.
        Returns DEFAULT_TIME_SINCE (3600) if no event found.
        Critical for hormone boredom/hunger stimulus calculation.
        """
        with self._lock:
            c = self._conn.execute(
                "SELECT MAX(timestamp) FROM event_markers WHERE event_type = ?",
                (event_type,))
            row = c.fetchone()
            if row and row[0] is not None:
                return time.time() - row[0]
            return DEFAULT_TIME_SINCE

    def get_recent_fires(
        self,
        program: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent program fire events, optionally filtered by program."""
        with self._lock:
            if program:
                c = self._conn.execute(
                    "SELECT * FROM program_fires WHERE program = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (program, limit))
            else:
                c = self._conn.execute(
                    "SELECT * FROM program_fires "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,))
            return [dict(row) for row in c.fetchall()]

    def get_action_patterns(
        self,
        helper: Optional[str] = None,
        program: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get action chain records for pattern analysis."""
        with self._lock:
            conditions = []
            params = []
            if helper:
                conditions.append("helper = ?")
                params.append(helper)
            if program:
                conditions.append("triggering_program = ?")
                params.append(program)

            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            params.append(limit)

            c = self._conn.execute(
                f"SELECT * FROM action_chains{where} "
                "ORDER BY timestamp DESC LIMIT ?",
                params)
            return [dict(row) for row in c.fetchall()]

    def get_hormone_history(
        self,
        program: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get hormone level history for a specific program."""
        with self._lock:
            c = self._conn.execute(
                "SELECT timestamp, epoch_id, levels, fired_programs "
                "FROM hormone_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,))
            results = []
            for row in c.fetchall():
                levels = json.loads(row[2])
                fired = json.loads(row[3]) if row[3] else []
                results.append({
                    "timestamp": row[0],
                    "epoch_id": row[1],
                    "level": levels.get(program, 0.0),
                    "fired": program in fired,
                })
            return results

    def get_best_actions(
        self,
        program: Optional[str] = None,
        min_score: float = 0.7,
        limit: int = 10,
    ) -> list[dict]:
        """Get highest-scoring actions for learning which expressions work."""
        with self._lock:
            if program:
                c = self._conn.execute(
                    "SELECT * FROM action_chains "
                    "WHERE score >= ? AND triggering_program = ? "
                    "ORDER BY score DESC LIMIT ?",
                    (min_score, program, limit))
            else:
                c = self._conn.execute(
                    "SELECT * FROM action_chains "
                    "WHERE score >= ? ORDER BY score DESC LIMIT ?",
                    (min_score, limit))
            return [dict(row) for row in c.fetchall()]

    def get_creative_works(
        self,
        work_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get creative works (art, audio, text), newest first."""
        with self._lock:
            if work_type:
                c = self._conn.execute(
                    "SELECT * FROM creative_works WHERE work_type = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (work_type, limit))
            else:
                c = self._conn.execute(
                    "SELECT * FROM creative_works "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,))
            return [dict(row) for row in c.fetchall()]

    # ── Vocabulary API (Language Learning) ────────────────────────────

    def store_word(
        self,
        word: str,
        word_type: str,
        stage: int = 1,
        felt_tensor: Optional[list] = None,
        hormone_pattern: Optional[dict] = None,
    ) -> None:
        """Store or update a word in vocabulary with its felt association."""
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM vocabulary WHERE word = ?", (word,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE vocabulary SET felt_tensor = ?, hormone_pattern = ?, "
                    "last_encountered = ? WHERE word = ?",
                    (json.dumps(felt_tensor) if felt_tensor else None,
                     json.dumps(hormone_pattern) if hormone_pattern else None,
                     now, word))
            else:
                self._conn.execute(
                    "INSERT INTO vocabulary "
                    "(word, word_type, stage, felt_tensor, hormone_pattern, "
                    "confidence, times_encountered, times_produced, "
                    "learning_phase, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 0.0, 0, 0, 'unlearned', ?)",
                    (word, word_type, stage,
                     json.dumps(felt_tensor) if felt_tensor else None,
                     json.dumps(hormone_pattern) if hormone_pattern else None,
                     now))
            self._conn.commit()

    def update_word_learning(
        self,
        word: str,
        phase: str,
        felt_tensor: Optional[list] = None,
        confidence_delta: float = 0.0,
        encountered: bool = False,
        produced: bool = False,
    ) -> None:
        """Update a word's learning progress after a training pass."""
        now = time.time()
        with self._lock:
            updates = ["learning_phase = ?", "last_encountered = ?"]
            params = [phase, now]
            if felt_tensor is not None:
                updates.append("felt_tensor = ?")
                params.append(json.dumps(felt_tensor))
            if confidence_delta != 0.0:
                updates.append(
                    "confidence = MIN(1.0, MAX(0.0, confidence + ?))")
                params.append(float(confidence_delta))
            if encountered:
                updates.append("times_encountered = times_encountered + 1")
            if produced:
                updates.append("times_produced = times_produced + 1")
            params.append(word)
            self._conn.execute(
                f"UPDATE vocabulary SET {', '.join(updates)} WHERE word = ?",
                params)
            self._conn.commit()

    def get_vocabulary(
        self,
        stage: Optional[int] = None,
        phase: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """Get vocabulary words, optionally filtered."""
        with self._lock:
            conditions = []
            params = []
            if stage is not None:
                conditions.append("stage = ?")
                params.append(stage)
            if phase is not None:
                conditions.append("learning_phase = ?")
                params.append(phase)
            if min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(min_confidence)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            c = self._conn.execute(
                f"SELECT * FROM vocabulary{where} ORDER BY stage, word",
                params)
            results = []
            for row in c.fetchall():
                d = dict(row)
                if d.get("felt_tensor"):
                    d["felt_tensor"] = json.loads(d["felt_tensor"])
                if d.get("hormone_pattern"):
                    d["hormone_pattern"] = json.loads(d["hormone_pattern"])
                results.append(d)
            return results

    def get_word(self, word: str) -> Optional[dict]:
        """Get a single word's vocabulary entry."""
        with self._lock:
            c = self._conn.execute(
                "SELECT * FROM vocabulary WHERE word = ?", (word,))
            row = c.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("felt_tensor"):
                d["felt_tensor"] = json.loads(d["felt_tensor"])
            if d.get("hormone_pattern"):
                d["hormone_pattern"] = json.loads(d["hormone_pattern"])
            return d

    def find_similar_words(
        self,
        felt_state: list,
        top_k: int = 5,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Find words whose felt_tensor is closest to given state (cosine sim)."""
        vocab = self.get_vocabulary(min_confidence=min_confidence)
        if not vocab or not felt_state:
            return []
        import math
        results = []
        mag_a = math.sqrt(sum(x * x for x in felt_state)) or 1e-10
        for entry in vocab:
            ft = entry.get("felt_tensor")
            if not ft:
                continue
            # Align lengths
            max_len = max(len(felt_state), len(ft))
            a = felt_state + [0.0] * (max_len - len(felt_state))
            b = ft + [0.0] * (max_len - len(ft))
            dot = sum(x * y for x, y in zip(a, b))
            mag_b = math.sqrt(sum(x * x for x in b)) or 1e-10
            sim = dot / (mag_a * mag_b)
            results.append((entry["word"], sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_vocab_stats(self) -> dict:
        """Vocabulary learning statistics."""
        with self._lock:
            c = self._conn.execute("SELECT COUNT(*) FROM vocabulary")
            total = c.fetchone()[0]
            c = self._conn.execute(
                "SELECT learning_phase, COUNT(*) FROM vocabulary GROUP BY learning_phase")
            phases = {row[0]: row[1] for row in c.fetchall()}
            c = self._conn.execute("SELECT AVG(confidence) FROM vocabulary")
            avg_conf = c.fetchone()[0] or 0.0
            c = self._conn.execute(
                "SELECT SUM(times_encountered), SUM(times_produced) FROM vocabulary")
            row = c.fetchone()
            return {
                "total_words": total,
                "phases": phases,
                "avg_confidence": round(avg_conf, 3),
                "total_encounters": row[0] or 0,
                "total_productions": row[1] or 0,
            }

    def get_stats(self) -> dict:
        """Summary stats for API."""
        with self._lock:
            stats = {}
            for table in ("hormone_snapshots", "program_fires", "action_chains",
                          "creative_works", "event_markers", "vocabulary"):
                try:
                    c = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = c.fetchone()[0]
                except Exception:
                    stats[table] = 0
            stats["vocab"] = self.get_vocab_stats()
            return stats

    def close(self):
        """Checkpoint WAL and close the database connection."""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
