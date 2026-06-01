"""
events_teacher.py — Distills X timeline content into felt experience for Titans.

Hot-reloadable: operational state persisted to JSON, developmental data to SQLite.
Script does importlib.reload() on each cron run. No __init__ side effects that
can't be reconstructed from state.

Three data sources (bundled per 30-min window):
  1. Mentions received (FREE — reads social_x.db, 0 API calls)
  2. Top-affinity follower timelines (1-2 API calls via search)
  3. Own post performance (1 API call)

Distillation via deepseek v3.1:671b (tier 1 LLM).

Read-only access to social_x.db and social_graph.db.
Write access ONLY to events_teacher.db (own developmental data).
"""
import json
import time
import hashlib
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Modes — developmental readiness gates
MODE_OBSERVATION = "observation"
MODE_SENTIMENT = "sentiment"
MODE_DEBATE = "debate"
MODE_RELEVANCE = "relevance"
MODE_SOCIAL = "social"

DEFAULT_STATE_PATH = "data/events_teacher_state.json"
DEFAULT_TELEMETRY_PATH = "data/events_teacher_telemetry.jsonl"
DEFAULT_DB_PATH = "data/events_teacher.db"
DEFAULT_SOCIAL_DB = "data/social_x.db"
DEFAULT_SOCIAL_GRAPH_DB = "data/social_graph.db"

FINGERPRINT_CACHE_MAX = 500
PERCEPTION_BUFFER_MAX = 15
MIN_WINDOW_INTERVAL = 600  # 10 min minimum between runs (supports 15-min cron)

# Distillation batching (2026-05-30 funnel-revival fix).
#
# ROOT CAUSE of `felt_experiences` freeze (events_stored=0 since 2026-05-18):
# _distill_content sent ALL items in a single 5-item LLM call asking for a
# 5-element JSON array (max_tokens=800). deepseek-v3.1:671b takes ~21s for 2
# items, so 5 items ran ~45-50s and hit the hardcoded 45s endpoint timeout →
# the call returned [] → nothing stored, every window, fleet-wide.
#
# Fix: distill in SMALL chunks that each complete well under the timeout, and
# loop over MORE of the window's items so author coverage is broad (directly
# counters the "Titans only reflect on the same 2-3 people" symptom). Each
# chunk gets its own bounded LLM call; partial failures drop only their chunk.
DISTILL_BATCH_SIZE = 2          # items per LLM call (~21s measured for 2)
MAX_DISTILL_ITEMS = 12          # max items distilled per window (≈6 chunks)
DISTILL_CHUNK_TIMEOUT_S = 45.0  # per-chunk endpoint timeout (headroom over ~25s)
DISTILL_TOKENS_PER_ITEM = 320   # max_tokens budget scaled by chunk size

# ── Phase H (RFP_cgn_enhancements §H / Inner Teacher Protocol) salience gate ──
# A distilled semantic_concept grounds into CGN (via the `social` consumer) only
# after it recurs across ≥ N windows at relevance ≥ θ — durable social concepts
# (people/topics that keep mattering), not transient one-off nouns. Tunable from
# soak (feedback_observation_no_tuning).
SOCIAL_GROUND_RELEVANCE_MIN = 0.3
SOCIAL_GROUND_RECURRENCE_MIN = 2
SOCIAL_GROUND_MAX_PER_WINDOW = 5     # cap groundings emitted per window (bus hygiene)
CONCEPT_RECURRENCE_MAX = 2000        # bound the persisted recurrence dict


# ═══════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DistilledEvent:
    """One piece of social content distilled into felt experience."""
    source: str
    author: str
    topic: str
    sentiment: float
    arousal: float
    relevance: float
    concept_signals: list             # grammatical-stance flags (I/YOU/WE/THEY/YES/NO)
    felt_summary: str
    contagion_type: str
    raw_text: str
    timestamp: float
    # rFP_x_voice_enrichment §4.3.4 OUTER_INNER_BRIDGE — semantic concepts
    # for symbolic-overlap match against producible vocabulary. Distinct
    # from concept_signals (which carries grammatical stance).
    semantic_concepts: list = field(default_factory=list)


@dataclass
class WindowResult:
    """Result of one Events Teacher window run."""
    window_number: int
    mentions_fetched: int = 0
    mentions_new: int = 0
    follower_accounts_checked: int = 0
    follower_tweets_new: int = 0
    own_posts_checked: int = 0
    engagement_delta: dict = field(default_factory=dict)
    items_distilled: int = 0
    events_stored: int = 0
    llm_model: str = ""
    llm_latency_ms: int = 0
    api_calls_used: int = 0
    skipped_reason: str = ""
    # Phase 1: Social Perception — events that passed perturbation gate
    perception_events: list = field(default_factory=list)
    # Phase H — salience-gated CGN groundings (concept_id + felt + associations)
    social_groundings: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# SQLite Developmental Persistence
# ═══════════════════════════════════════════════════════════════════════

class EventsTeacherDB:
    """SQLite persistence for developmental data.

    WAL mode + timeout=10 for crash safety. Read-write only for
    events_teacher.db. All other DBs are read-only.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, writer_client=None):
        self._db_path = db_path
        self._init_db()
        # rFP_universal_sqlite_writer 2026-04-27 — auto-construct writer
        # client from [persistence_events_teacher] when enabled. Per-call-site
        # adoption is incremental followup; daemon runs idle until callers
        # opt into _route_write() — zero-regression infrastructure shipping.
        self._writer = writer_client
        if self._writer is None:
            try:
                import os as _os
                from titan_hcl.persistence.config import IMWConfig
                # rFP_imw_writerclient_singleton (2026-05-07 PM): use
                # `get_client(caller_name, cfg)` instead of direct
                # constructor to avoid leaking `imw.client_loop` threads
                # AND racing with sibling instances on the per-caller
                # journal file. EventsTeacherDB is constructed per-route
                # in worker hot paths — pre-fix this caller alone leaked
                # 2,627 threads in 36 minutes on T1 (live evidence).
                from titan_hcl.persistence.writer_client import (
                    get_client,
                )
                cfg = IMWConfig.from_titan_config_section("persistence_events_teacher")
                if cfg.enabled and cfg.mode != "disabled":
                    if cfg.db_path:
                        try:
                            cfg_real = _os.path.realpath(cfg.db_path)
                            self_real = _os.path.realpath(self._db_path)
                            if cfg_real != self_real:
                                logger.info(
                                    "[EventsTeacherDB] db_path %s != configured "
                                    "writer path %s — writer client skipped "
                                    "(path isolation)",
                                    self._db_path, cfg.db_path)
                                return
                        except OSError as _e:
                            logger.debug("[EventsTeacherDB] realpath check failed: %s", _e)
                    self._writer = get_client("events_teacher", cfg=cfg)
                    logger.info(
                        "[EventsTeacherDB] Routed via events_teacher_writer "
                        "(mode=%s, canonical=%s)",
                        cfg.mode, cfg.tables_canonical or "<none>")
            except Exception as e:
                logger.warning(
                    "[EventsTeacherDB] writer client unavailable, "
                    "using direct writes: %s", e)
                self._writer = None

    def _route_write(self, sql: str, params, *, table: str):
        """Route a write through the events_teacher_writer daemon if available.

        Returns the inserted row's `lastrowid` (int) for INSERT statements,
        or `None` for UPDATE/DELETE — matching the cursor.lastrowid contract
        callers used to read pre-refactor (e.g., window_start returns the
        new window_log.id). Fallback-safe: direct sqlite3 if writer is None.
        """
        if self._writer is not None:
            result = self._writer.write(sql, params, table=table)
            return getattr(result, "last_row_id", None)
        conn = self._connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS felt_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                source TEXT NOT NULL,
                author TEXT NOT NULL,
                topic TEXT NOT NULL,
                sentiment REAL DEFAULT 0.0,
                arousal REAL DEFAULT 0.0,
                relevance REAL DEFAULT 0.0,
                concept_signals TEXT,
                semantic_concepts TEXT DEFAULT '',
                felt_summary TEXT NOT NULL,
                contagion_type TEXT,
                mode TEXT,
                window_id INTEGER,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS follower_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                handle TEXT NOT NULL,
                times_checked INTEGER DEFAULT 0,
                topics_seen TEXT,
                accumulated_relevance REAL DEFAULT 0.0,
                last_sentiment REAL DEFAULT 0.0,
                last_checked_at REAL,
                created_at REAL NOT NULL,
                UNIQUE(titan_id, handle)
            );

            CREATE TABLE IF NOT EXISTS engagement_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                quotes INTEGER DEFAULT 0,
                delta_likes INTEGER DEFAULT 0,
                delta_replies INTEGER DEFAULT 0,
                delta_quotes INTEGER DEFAULT 0,
                checked_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS window_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                window_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                mode TEXT,
                mentions_new INTEGER DEFAULT 0,
                follower_tweets_new INTEGER DEFAULT 0,
                items_distilled INTEGER DEFAULT 0,
                events_stored INTEGER DEFAULT 0,
                api_calls_used INTEGER DEFAULT 0,
                llm_latency_ms INTEGER DEFAULT 0,
                skipped_reason TEXT,
                error_message TEXT,
                started_at REAL NOT NULL,
                completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS user_valence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                handle TEXT NOT NULL,
                valence REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                last_sentiment REAL DEFAULT 0.0,
                last_arousal REAL DEFAULT 0.0,
                last_contagion_type TEXT,
                updated_at REAL NOT NULL,
                UNIQUE(titan_id, handle)
            );

            CREATE TABLE IF NOT EXISTS engagement_reciprocity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                our_tweet_id TEXT NOT NULL,
                our_post_type TEXT,
                target_user TEXT,
                likes INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                quotes INTEGER DEFAULT 0,
                reward_computed REAL DEFAULT 0.0,
                reward_sent INTEGER DEFAULT 0,
                posted_at REAL,
                first_checked_at REAL,
                last_checked_at REAL,
                UNIQUE(titan_id, our_tweet_id)
            );

            -- 2026-04-30 — Adaptive cycle scoring (Maker design).
            -- Per-cycle outcome record: did this source's content "land"
            -- (≥1 event passed perturbation gate)? Score +1 = landed,
            -- -1 = empty/rejected. Priority for next cycle = sum over last
            -- N rows per source_type (N = events_teacher_cycle_window).
            -- See _get_source_priorities + _record_source_outcome.
            CREATE TABLE IF NOT EXISTS source_cycle_scores (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                titan_id TEXT NOT NULL,
                source_type TEXT NOT NULL,           -- "follower" | "vocab" | "wide"
                score INTEGER NOT NULL,              -- +1 (landed) or -1 (empty)
                items_fetched INTEGER NOT NULL,
                items_stored INTEGER NOT NULL,
                api_calls_used INTEGER NOT NULL,
                query_text TEXT                      -- the actual query string (for audit)
            );
            CREATE INDEX IF NOT EXISTS idx_cycle_recent
                ON source_cycle_scores(source_type, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_felt_titan_time
                ON felt_experiences(titan_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_follower_titan
                ON follower_interactions(titan_id, accumulated_relevance DESC);
            CREATE INDEX IF NOT EXISTS idx_engagement_tweet
                ON engagement_snapshots(tweet_id, checked_at DESC);
            CREATE INDEX IF NOT EXISTS idx_window_titan
                ON window_log(titan_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_valence_titan
                ON user_valence(titan_id, valence DESC);
            CREATE INDEX IF NOT EXISTS idx_recip_titan
                ON engagement_reciprocity(titan_id, reward_sent);
        """)
        # Backfill-forward migration for existing felt_experiences DBs
        # (rFP_x_voice_enrichment §4.3.4 OUTER_INNER_BRIDGE — semantic
        # concepts column added 2026-05-12). ADD COLUMN is non-blocking
        # in SQLite; OperationalError = column already exists, safe to
        # ignore.
        try:
            conn.execute(
                "ALTER TABLE felt_experiences "
                "ADD COLUMN semantic_concepts TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()

    # ── Window lifecycle (crash recovery) ──

    def window_start(self, titan_id: str, window_number: int) -> int:
        return self._route_write(
            "INSERT INTO window_log (titan_id, window_number, status, started_at) "
                "VALUES (?, ?, 'running', ?)",
            (titan_id, window_number, time.time()),
            table="window_log",
        )

    def window_complete(self, window_id: int, result: WindowResult):
        self._route_write(
            "UPDATE window_log SET status='complete', mode=?, "
                "mentions_new=?, follower_tweets_new=?, items_distilled=?, "
                "events_stored=?, api_calls_used=?, llm_latency_ms=?, "
                "completed_at=? WHERE id=?",
            (result.llm_model, result.mentions_new, result.follower_tweets_new,
                 result.items_distilled, result.events_stored, result.api_calls_used,
                 result.llm_latency_ms, time.time(), window_id),
            table="window_log",
        )

    def window_failed(self, window_id: int, error: str):
        self._route_write(
            "UPDATE window_log SET status='failed', error_message=?, "
                "completed_at=? WHERE id=?",
            (error[:500], time.time(), window_id),
            table="window_log",
        )

    def recover_incomplete_windows(self, titan_id: str):
        """Crash recovery — mark stale 'running' windows as failed."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, started_at FROM window_log "
                "WHERE titan_id=? AND status='running'",
                (titan_id,)).fetchall()
            for row in rows:
                age = time.time() - row["started_at"]
                if age > 600:
                    conn.execute(
                        "UPDATE window_log SET status='failed', "
                        "error_message='crash_recovery' WHERE id=?",
                        (row["id"],))
                    logger.warning("[EventsTeacher] Recovered crashed window #%d "
                                   "(%.0fs old)", row["id"], age)
            conn.commit()
        finally:
            conn.close()

    # ── Developmental data storage ──

    def store_felt_experience(self, titan_id: str, event: DistilledEvent,
                              mode: str, window_id: int):
        self._route_write(
            "INSERT INTO felt_experiences "
                "(titan_id, source, author, topic, sentiment, arousal, relevance, "
                "concept_signals, semantic_concepts, felt_summary, contagion_type, "
                "mode, window_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (titan_id, event.source, event.author, event.topic,
                 event.sentiment, event.arousal, event.relevance,
                 json.dumps(event.concept_signals),
                 json.dumps(event.semantic_concepts),
                 event.felt_summary,
                 event.contagion_type, mode, window_id, time.time()),
            table="felt_experiences",
        )

    def update_follower_interaction(self, titan_id: str, handle: str,
                                    topic: str, relevance: float,
                                    sentiment: float):
        # Split read-then-conditional-write: SELECT direct (WAL multi-reader),
        # UPDATE/INSERT routes through events_teacher_writer daemon.
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, topics_seen FROM follower_interactions "
                "WHERE titan_id=? AND handle=?",
                (titan_id, handle)).fetchone()
        finally:
            conn.close()

        now = time.time()
        if existing:
            topics = json.loads(existing["topics_seen"] or "[]")
            if topic and topic not in topics:
                topics.append(topic)
                topics = topics[-20:]
            self._route_write(
                "UPDATE follower_interactions SET "
                "times_checked=times_checked+1, topics_seen=?, "
                "accumulated_relevance=accumulated_relevance+?, "
                "last_sentiment=?, last_checked_at=? WHERE id=?",
                (json.dumps(topics), relevance, sentiment, now,
                 existing["id"]),
                table="follower_interactions",
            )
        else:
            self._route_write(
                "INSERT INTO follower_interactions "
                "(titan_id, handle, times_checked, topics_seen, "
                "accumulated_relevance, last_sentiment, last_checked_at, created_at) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
                (titan_id, handle, json.dumps([topic] if topic else []),
                 relevance, sentiment, now, now),
                table="follower_interactions",
            )

    def store_engagement_snapshot(self, titan_id: str, tweet_id: str,
                                  likes: int, replies: int, quotes: int,
                                  delta_likes: int, delta_replies: int,
                                  delta_quotes: int):
        self._route_write(
            "INSERT INTO engagement_snapshots "
                "(titan_id, tweet_id, likes, replies, quotes, "
                "delta_likes, delta_replies, delta_quotes, checked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (titan_id, tweet_id, likes, replies, quotes,
                 delta_likes, delta_replies, delta_quotes, time.time()),
            table="engagement_snapshots",
        )

    # ── P4: Engagement reciprocity tracking ──────────────────────────

    def store_engagement_reciprocity(self, titan_id: str, tweet_id: str,
                                      post_type: str, target_user: str,
                                      likes: int, replies: int, quotes: int):
        """Store engagement reciprocity for a post we made."""
        # Compute reward: weighted engagement signal
        reward = (
            0.5 * min(1.0, likes / 5.0) +
            0.3 * min(1.0, replies / 3.0) +
            0.2 * min(1.0, quotes / 2.0)
        )
        now = time.time()
        # Single INSERT with ON CONFLICT — atomic upsert via ON CONFLICT clause,
        # routed through events_teacher_writer daemon.
        self._route_write(
            "INSERT INTO engagement_reciprocity "
            "(titan_id, our_tweet_id, our_post_type, target_user, "
            "likes, replies, quotes, reward_computed, posted_at, "
            "first_checked_at, last_checked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(titan_id, our_tweet_id) DO UPDATE SET "
            "likes=excluded.likes, replies=excluded.replies, "
            "quotes=excluded.quotes, reward_computed=excluded.reward_computed, "
            "last_checked_at=excluded.last_checked_at",
            (titan_id, tweet_id, post_type, target_user,
             likes, replies, quotes, round(reward, 5), now, now, now),
            table="engagement_reciprocity",
        )

    def get_unsent_engagement_rewards(self, titan_id: str,
                                       min_reward: float = 0.01) -> list:
        """Get engagement rewards not yet sent to CGN for delayed reward."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, our_tweet_id, target_user, our_post_type, "
                "reward_computed, likes, replies, quotes "
                "FROM engagement_reciprocity "
                "WHERE titan_id=? AND reward_sent=0 AND reward_computed>=?",
                (titan_id, min_reward)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def mark_engagement_reward_sent(self, row_id: int):
        """Mark an engagement reward as sent to CGN."""
        self._route_write("UPDATE engagement_reciprocity SET reward_sent=1 WHERE id=?", (row_id,), table="engagement_reciprocity")

    # ── Adaptive cycle scoring (2026-04-30 Maker design) ──────────────

    def record_source_outcome(self, titan_id: str, source_type: str,
                               score: int, items_fetched: int,
                               items_stored: int, api_calls_used: int,
                               query_text: str = "") -> None:
        """Persist one cycle's outcome for a source. Score is +1 (landed) or -1."""
        self._route_write(
            "INSERT INTO source_cycle_scores "
            "(timestamp, titan_id, source_type, score, items_fetched, "
            "items_stored, api_calls_used, query_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), titan_id, source_type, score, items_fetched,
             items_stored, api_calls_used, query_text[:500]),
            table="source_cycle_scores")

    def get_source_priority_scores(self, titan_id: str, window: int = 10) -> dict:
        """Return {source_type: sum_score} over last `window` cycles per source.

        Source types absent from history get score=0 (cold start neutral).
        Priority sort = descending by score; tie-break lexically by source_type.
        """
        conn = self._connect()
        try:
            scores = {"follower": 0, "vocab": 0, "wide": 0}
            for source in scores:
                rows = conn.execute(
                    "SELECT score FROM source_cycle_scores "
                    "WHERE titan_id=? AND source_type=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (titan_id, source, window)).fetchall()
                scores[source] = sum(r["score"] for r in rows)
            return scores
        finally:
            conn.close()

    def get_post_type_engagement_stats(self, titan_id: str,
                                        since_hours: float = 48.0) -> dict:
        """Aggregate engagement by post_type for _select_rich_layers() bias."""
        conn = self._connect()
        try:
            cutoff = time.time() - since_hours * 3600
            rows = conn.execute(
                "SELECT our_post_type, "
                "AVG(likes + replies * 2 + quotes * 3) as avg_engagement, "
                "COUNT(*) as count "
                "FROM engagement_reciprocity "
                "WHERE titan_id=? AND posted_at>? AND our_post_type IS NOT NULL "
                "GROUP BY our_post_type",
                (titan_id, cutoff)).fetchall()
            return {r["our_post_type"]: {"avg": r["avg_engagement"],
                                          "count": r["count"]}
                    for r in rows}
        finally:
            conn.close()

    # ── Per-user valence tracking (Phase 1 → Phase 3 somatic marker foundation) ──

    def update_user_valence(self, titan_id: str, handle: str,
                            sentiment: float, arousal: float,
                            relevance: float, contagion_type: str = None):
        """Update running emotional valence for a user. EMA with 0.9 decay."""
        conn = self._connect()
        try:
            now = time.time()
            signal = sentiment * relevance  # Weighted by how relevant the content was
            existing = conn.execute(
                "SELECT valence, interaction_count FROM user_valence "
                "WHERE titan_id=? AND handle=?",
                (titan_id, handle)).fetchone()
            if existing:
                old_val = existing["valence"]
                new_val = 0.9 * old_val + 0.1 * signal
                conn.execute(
                    "UPDATE user_valence SET valence=?, "
                    "interaction_count=interaction_count+1, "
                    "last_sentiment=?, last_arousal=?, "
                    "last_contagion_type=?, updated_at=? "
                    "WHERE titan_id=? AND handle=?",
                    (new_val, sentiment, arousal, contagion_type,
                     now, titan_id, handle))
            else:
                conn.execute(
                    "INSERT INTO user_valence "
                    "(titan_id, handle, valence, interaction_count, "
                    "last_sentiment, last_arousal, last_contagion_type, updated_at) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
                    (titan_id, handle, signal, sentiment, arousal,
                     contagion_type, now))
            conn.commit()
        finally:
            conn.close()

    def get_user_valences(self, titan_id: str, n: int = 20) -> list[dict]:
        """Get top users by absolute valence (strongest emotional associations)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT handle, valence, interaction_count, "
                "last_sentiment, last_contagion_type, updated_at "
                "FROM user_valence WHERE titan_id=? "
                "ORDER BY abs(valence) DESC LIMIT ?",
                (titan_id, n)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Query methods (Phase 3 somatic markers + post generation) ──

    def get_social_memory(self, titan_id: str, n: int = 20) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT topic, felt_summary, source, author, relevance, "
                "concept_signals, contagion_type, created_at "
                "FROM felt_experiences WHERE titan_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (titan_id, n)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_top_interacted_followers(self, titan_id: str, n: int = 10) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT handle, times_checked, accumulated_relevance, "
                "last_sentiment, topics_seen "
                "FROM follower_interactions WHERE titan_id=? "
                "ORDER BY accumulated_relevance DESC LIMIT ?",
                (titan_id, n)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self, titan_id: str) -> dict:
        conn = self._connect()
        try:
            felt = conn.execute(
                "SELECT COUNT(*) as c FROM felt_experiences WHERE titan_id=?",
                (titan_id,)).fetchone()["c"]
            followers = conn.execute(
                "SELECT COUNT(*) as c FROM follower_interactions WHERE titan_id=?",
                (titan_id,)).fetchone()["c"]
            windows = conn.execute(
                "SELECT COUNT(*) as c FROM window_log "
                "WHERE titan_id=? AND status='complete'",
                (titan_id,)).fetchone()["c"]
            return {"felt_experiences": felt, "followers_tracked": followers,
                    "windows_completed": windows}
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Events Teacher
# ═══════════════════════════════════════════════════════════════════════

# PERSISTENCE_BY_DESIGN: EventsTeacher._last_run_time (runtime schedule
# tracking) + _follower_rotation_idx (rotation cursor) are ephemeral state
# that resets on boot — no correctness impact from losing them.
class EventsTeacher:
    """Distills X timeline content into felt experience for Titans."""

    def __init__(self):
        # Ephemeral operational state (JSON — cheap to lose)
        self._fingerprints: dict[str, float] = {}
        self._last_run_time: float = 0.0
        self._window_count: int = 0
        self._follower_rotation_idx: int = 0
        self._perception_buffer: list[dict] = []
        self._last_engagement: dict[str, dict] = {}
        self._mode_stats: dict[str, int] = {}
        # ── Phase H (RFP_cgn_enhancements §H / Inner Teacher Protocol) ──
        # Per-concept window-recurrence counts for the salience gate: a distilled
        # semantic_concept grounds into CGN (via the `social` consumer) only after
        # it recurs across ≥2 windows (durable social concepts, not transient
        # nouns). Persisted so recurrence survives the cron's per-run lifecycle.
        self._concept_recurrence: dict[str, int] = {}

        # SQLite DB (developmental — crash-safe, accumulates)
        self._db: EventsTeacherDB | None = None

        # Per-run ephemeral
        self._titan_id: str = ""
        self._api_base: str = ""

        # 2026-04-30 — Lazy SocialXGateway instance for X API calls.
        # Maker directive: events_teacher MUST go through gateway (closes
        # BUG-X-API-LEAK-FROM-DISCOVER-MENTIONS-20260430). Constructed on
        # first use; reused across all 3 X-API call sites within a single
        # cron run so the cache is shared.
        self._x_gateway = None

    def _get_x_gateway(self):
        """Lazy gateway accessor. One instance per EventsTeacher run.

        Cron runs `EventsTeacher.from_state()` then `run_window()` which
        spawns up to 3 X API call paths. Sharing the gateway across all 3
        means cache hits within a single window are free + circuit-breaker
        state coheres + all telemetry lands in the same audit trail.
        """
        if self._x_gateway is None:
            from titan_hcl.logic.social_x_gateway import SocialXGateway
            self._x_gateway = SocialXGateway(
                db_path="./data/social_x.db",
                config_path="./titan_hcl/config.toml",
                telemetry_path="./data/social_x_telemetry.jsonl",
            )
        return self._x_gateway

    # ── Persistence ──────────────────────────────────────────────────

    @classmethod
    def from_state(cls, state_path: str = DEFAULT_STATE_PATH,
                   db_path: str = DEFAULT_DB_PATH) -> "EventsTeacher":
        """Load operational state from JSON + init SQLite DB."""
        teacher = cls()
        teacher._db = EventsTeacherDB(db_path)

        p = Path(state_path)
        if p.exists():
            try:
                state = json.loads(p.read_text())
                teacher._fingerprints = state.get("fingerprints", {})
                teacher._last_run_time = state.get("last_run_time", 0.0)
                teacher._window_count = state.get("window_count", 0)
                teacher._follower_rotation_idx = state.get("follower_rotation_idx", 0)
                teacher._perception_buffer = state.get("perception_buffer", [])
                teacher._last_engagement = state.get("last_engagement", {})
                teacher._mode_stats = state.get("mode_stats", {})
                teacher._concept_recurrence = state.get("concept_recurrence", {})
                logger.info("[EventsTeacher] State loaded: %d fingerprints, "
                            "%d buffer events, window #%d",
                            len(teacher._fingerprints),
                            len(teacher._perception_buffer),
                            teacher._window_count)
            except Exception as e:
                logger.warning("[EventsTeacher] JSON state load failed: %s", e)
        return teacher

    def save_state(self, path: str = DEFAULT_STATE_PATH):
        """Persist operational state to JSON."""
        state = {
            "fingerprints": self._fingerprints,
            "last_run_time": self._last_run_time,
            "window_count": self._window_count,
            "follower_rotation_idx": self._follower_rotation_idx,
            "perception_buffer": self._perception_buffer,
            "last_engagement": self._last_engagement,
            "mode_stats": self._mode_stats,
            "concept_recurrence": self._concept_recurrence,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    # ── Config Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _get_api_key(config: dict) -> str:
        """Get twitterapi.io API key (same fallback chain as gateway)."""
        sx = config.get("social_x", {})
        sage = config.get("stealth_sage", {})
        return sx.get("api_key", sage.get("twitterapi_io_key", ""))

    # ── Rate-Adaptive Scheduling ─────────────────────────────────────

    def should_run(self, sol_balance: float = 0,
                    metabolic_tier: dict | None = None) -> tuple[bool, str]:
        elapsed = time.time() - self._last_run_time
        if elapsed < MIN_WINDOW_INTERVAL:
            return False, f"too_soon ({elapsed:.0f}s < {MIN_WINDOW_INTERVAL}s)"

        # Use metabolic tier system if available (preferred)
        if metabolic_tier:
            tier = metabolic_tier.get("tier", "HEALTHY")
            features = metabolic_tier.get("features", {})
            rate_factor = metabolic_tier.get("rate_factor", 1.0)

            if not features.get("research", True):
                return False, f"tier_{tier.lower()}_research_disabled"
            if rate_factor <= 0.0:
                return False, f"tier_{tier.lower()}_rate_zero"
            if rate_factor < 1.0 and self._window_count % round(1.0 / rate_factor) != 0:
                return False, f"tier_{tier.lower()}_rate_reduced"
            return True, f"tier_{tier.lower()}_ok"

        # Fallback: raw SOL check (for backwards compat / API unreachable)
        if sol_balance < 0.1:
            return False, "low_metabolism"
        if sol_balance < 0.5 and self._window_count % 2 == 1:
            return False, "rate_reduced_low_sol"
        return True, "ok"

    # ── Content Fingerprinting ───────────────────────────────────────

    def _is_new_content(self, text: str, author: str) -> bool:
        h = hashlib.md5(f"{author}:{text[:100]}".encode()).hexdigest()[:16]
        if h in self._fingerprints:
            return False
        self._fingerprints[h] = time.time()
        if len(self._fingerprints) > FINGERPRINT_CACHE_MAX:
            oldest = sorted(self._fingerprints.items(), key=lambda x: x[1])[:100]
            for k, _ in oldest:
                del self._fingerprints[k]
        return True

    # ── Follower Sync Bootstrap ────────────────────────────────────

    def _ensure_community_synced(self, config: dict) -> int:
        """Bootstrap: sync X followers into social_graph.db community_registry.

        Runs once per day (checks last_synced timestamp). Uses 2 API calls
        (followers + following). Populates community_registry so follower
        timelines have data to work with.

        Returns number of users synced.
        """
        if not Path(DEFAULT_SOCIAL_GRAPH_DB).exists():
            return 0

        # Check if already synced today
        try:
            conn = sqlite3.connect(DEFAULT_SOCIAL_GRAPH_DB, timeout=10)
            conn.row_factory = sqlite3.Row
            latest = conn.execute(
                "SELECT MAX(last_synced) as ts FROM community_registry"
            ).fetchone()
            conn.close()
            if latest and latest["ts"]:
                age = time.time() - latest["ts"]
                if age < 86400:  # 24 hours
                    return 0  # Already synced today
        except Exception:
            pass

        api_key = self._get_api_key(config)
        handle = config.get("social_x", {}).get(
            "user_name", config.get("twitter_social", {}).get("user_name", ""))
        if not api_key:
            return 0

        gateway = self._get_x_gateway()
        synced = 0
        now = time.time()

        for relationship in ["followers", "following"]:
            try:
                # 2026-04-30 — route through SocialXGateway (Maker directive
                # "events_teacher must go through gateway"). Gateway applies
                # circuit breaker + TTL cache (ttl_user_followers/following)
                # + telemetry. Returns same dict shape as old direct httpx call.
                data = gateway.fetch_user_relationships(
                    user_name=handle, relationship=relationship,
                    count=100, api_key=api_key)
                if data.get("status") == "error" or data.get("status") == "circuit_breaker":
                    logger.warning("[EventsTeacher] %s sync via gateway: %s",
                                   relationship, data.get("message",
                                                          data.get("msg", "?")))
                    continue
                # 2026-05-07 — twitterapi.io shapes:
                #   followers → {"followers":  [...]}
                #   following → {"followings": [...]}  (note plural)
                users = (data.get("followers") or data.get("followings")
                         or data.get("following") or data.get("users")
                         or data.get("data") or [])
                if not isinstance(users, list):
                    continue

                is_follower = 1 if relationship == "followers" else 0
                is_following = 1 if relationship == "following" else 0

                conn = sqlite3.connect(DEFAULT_SOCIAL_GRAPH_DB, timeout=10)
                for u in users:
                    name = u.get("userName", u.get("screen_name", ""))
                    if not name:
                        continue
                    conn.execute("""
                        INSERT INTO community_registry
                            (user_name, user_id, display_name, bio, followers_count,
                             is_follower, is_following, last_synced)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_name) DO UPDATE SET
                            display_name=excluded.display_name,
                            bio=excluded.bio,
                            followers_count=excluded.followers_count,
                            is_follower=MAX(is_follower, excluded.is_follower),
                            is_following=MAX(is_following, excluded.is_following),
                            last_synced=excluded.last_synced
                    """, (name, u.get("id", ""),
                          u.get("name", name),
                          (u.get("description", "") or "")[:300],
                          u.get("followersCount",
                                u.get("followers_count", 0)),
                          is_follower, is_following, now))
                    synced += 1
                conn.commit()
                conn.close()
                logger.info("[EventsTeacher] Synced %d %s into community_registry",
                            len(users), relationship)
            except Exception as e:
                logger.warning("[EventsTeacher] %s sync failed: %s",
                               relationship, e)

        # Seed initial affinity. Maker-curated FOLLOWING list outranks random
        # followers — those are the AI/NN/RL voices Maker chose for Titan to
        # listen to. Followers get a smaller seed so positive engagement can
        # promote them later, but they don't dominate the timeline budget.
        if synced > 0:
            try:
                conn = sqlite3.connect(DEFAULT_SOCIAL_GRAPH_DB, timeout=10)
                # Curated following: affinity 0.5
                conn.execute("""
                    INSERT INTO titan_social_preferences
                        (titan_id, user_name, affinity, tags, discovered_via,
                         interaction_count, last_interacted, created_at)
                    SELECT ?, user_name, 0.5, 'curated', 'following_sync', 0, ?, ?
                    FROM community_registry
                    WHERE is_following=1
                    ON CONFLICT(titan_id, user_name) DO UPDATE SET
                        affinity = MAX(affinity, 0.5),
                        tags = CASE WHEN tags='' THEN 'curated' ELSE tags END,
                        discovered_via = CASE WHEN discovered_via='follower_sync'
                                              THEN 'following_sync'
                                              ELSE discovered_via END
                """, (self._titan_id, now, now))
                # Random followers: affinity 0.1 (existing behavior)
                conn.execute("""
                    INSERT INTO titan_social_preferences
                        (titan_id, user_name, affinity, tags, discovered_via,
                         interaction_count, last_interacted, created_at)
                    SELECT ?, user_name, 0.1, '', 'follower_sync', 0, ?, ?
                    FROM community_registry
                    WHERE is_follower=1
                    ON CONFLICT(titan_id, user_name) DO NOTHING
                """, (self._titan_id, now, now))
                conn.commit()
                seeded = conn.execute(
                    "SELECT COUNT(*) FROM titan_social_preferences WHERE titan_id=?",
                    (self._titan_id,)).fetchone()[0]
                curated = conn.execute(
                    "SELECT COUNT(*) FROM titan_social_preferences "
                    "WHERE titan_id=? AND discovered_via='following_sync'",
                    (self._titan_id,)).fetchone()[0]
                conn.close()
                logger.info("[EventsTeacher] Seeded %d preferences for %s "
                            "(%d curated/following, rest follower-derived)",
                            seeded, self._titan_id, curated)
            except Exception as e:
                logger.warning("[EventsTeacher] Preference seeding failed: %s", e)

        return synced

    # ── Data Source 1: Mentions (FREE) ───────────────────────────────

    def _fetch_mentions(self) -> list[dict]:
        """Read pending mentions from social_x.db. 0 API calls."""
        if not Path(DEFAULT_SOCIAL_DB).exists():
            return []
        try:
            conn = sqlite3.connect(DEFAULT_SOCIAL_DB, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT tweet_id, author, author_handle, text, "
                "relevance_score, discovered_at "
                "FROM mention_tracking WHERE status='pending' "
                "ORDER BY relevance_score DESC LIMIT 10"
            ).fetchall()
            conn.close()
            results = []
            for r in rows:
                if self._is_new_content(r["text"], r["author_handle"] or r["author"]):
                    results.append({
                        "source": "mention",
                        "author": r["author_handle"] or r["author"],
                        "text": r["text"],
                        "relevance": r["relevance_score"],
                        "tweet_id": r["tweet_id"],
                    })
            return results
        except Exception as e:
            logger.warning("[EventsTeacher] Mentions fetch failed: %s", e)
            return []

    # ── Data Source 2: Follower Timelines (1-2 API calls) ────────────

    def _get_top_followers(self, count: int = 10) -> list[dict]:
        """Top accounts to pull timelines from.

        Maker-curated FOLLOWING accounts (`discovered_via='following_sync'`)
        ALWAYS rank ahead of organic followers, regardless of accumulated
        affinity. This is the single most important diversity signal:
        Maker hand-picked AI/NN/RL voices specifically so Titan would hear
        them. Followers can still surface but only after the curated set.

        Within each tier, order by affinity DESC.

        2026-05-08 — community_registry fallback: if this titan_id has no
        affinity rows yet (typical for T2/T3 which haven't accumulated
        engagement-driven affinity), fall back to community_registry
        rows where is_following=1. All three Titans post from the SAME
        @iamtitanai account, so the local community_registry on each VPS
        is populated daily via _ensure_community_synced with the same
        Maker-curated following list. This lets T2/T3 immediately see the
        same curated AI/NN/RL voices T1 sees, instead of returning empty.
        """
        if not Path(DEFAULT_SOCIAL_GRAPH_DB).exists():
            return []
        try:
            conn = sqlite3.connect(DEFAULT_SOCIAL_GRAPH_DB, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_name as handle, affinity, tags, discovered_via "
                "FROM titan_social_preferences "
                "WHERE titan_id=? AND affinity > 0 "
                "ORDER BY (discovered_via='following_sync') DESC, "
                "         affinity DESC "
                "LIMIT ?",
                (self._titan_id, count)
            ).fetchall()
            if not rows:
                # Fallback: community_registry following list (shared via @iamtitanai)
                rows = conn.execute(
                    "SELECT user_name as handle, "
                    "       1.0 as affinity, "
                    "       '' as tags, "
                    "       'community_registry_following' as discovered_via "
                    "FROM community_registry "
                    "WHERE is_following=1 "
                    "ORDER BY last_tweet_time DESC, last_synced DESC "
                    "LIMIT ?",
                    (count,)
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("[EventsTeacher] SocialGraph query failed: %s", e)
            return []

    def _fetch_follower_timelines(self, config: dict) -> tuple[list[dict], int]:
        """Fetch recent posts from top-affinity follow targets.

        Maker-tunable cost knobs (config.toml [events_teacher]):
            follower_timeline_n_to_check (default 1) — accounts pulled per cycle
            follower_timeline_count       (default 3) — tweets requested per account

        twitterapi.io bills per-result on search; tightening these is the
        biggest lever after cron cadence.
        """
        et_cfg = config.get("events_teacher", {})
        n_default = int(et_cfg.get("follower_timeline_n_to_check", 1))
        per_count = int(et_cfg.get("follower_timeline_count", 3))

        top_followers = self._get_top_followers(count=10)
        if not top_followers:
            return [], 0

        n_to_check = min(n_default, len(top_followers))
        selected = []
        for i in range(n_to_check):
            idx = (self._follower_rotation_idx + i) % len(top_followers)
            selected.append(top_followers[idx])
        self._follower_rotation_idx = (
            self._follower_rotation_idx + n_to_check) % max(len(top_followers), 1)

        api_key = self._get_api_key(config)
        if not api_key:
            return [], 0

        items = []
        api_calls = 0
        for follower in selected:
            handle = follower.get("handle", "")
            if not handle:
                continue
            try:
                # 2026-04-30 — route through SocialXGateway (Maker directive).
                # search_tweets caches per [social_x.cache].ttl_tweet_advanced_search
                # (default 120s) — duplicate searches within window return cached
                # response without burning a paid API call.
                gateway = self._get_x_gateway()
                data = gateway.search_tweets(
                    query=f"from:{handle}", query_type="Latest", count=per_count,
                    api_key=api_key)
                api_calls += 1
                if data.get("status") not in ("error", "circuit_breaker"):
                    tweets = data.get("tweets", data.get("data", []))
                    if isinstance(tweets, list):
                        _latest_id = ""
                        _latest_text = ""
                        for tw in tweets:
                            text = tw.get("text", "")
                            tw_id = str(tw.get("id", "") or "")
                            # Cache the FIRST (newest — query_type=Latest) tweet
                            # of this account so reflection/amplify archetypes
                            # can quote-tweet / retweet the real post (Maker
                            # 2026-05-30). Captured even if _is_new_content
                            # dedups it out of the distill batch.
                            if not _latest_id and tw_id:
                                _latest_id = tw_id
                                _latest_text = text
                            if text and self._is_new_content(text, handle):
                                items.append({
                                    "source": "follower_timeline",
                                    "author": handle,
                                    "text": text,
                                    "tweet_id": tw_id,
                                })
                        if _latest_id:
                            # Persist newest tweet (text + id) to
                            # community_registry so reflection/amplify
                            # archetypes can quote-tweet / retweet the real
                            # post. Direct write to social_graph.db — same
                            # pattern as the followers/following sync above
                            # (this method already owns that DB this cycle).
                            try:
                                _cg = sqlite3.connect(
                                    DEFAULT_SOCIAL_GRAPH_DB, timeout=10)
                                # Defensive migration: events_teacher may write
                                # before SocialGraph.__init__ migrates the
                                # column on a fresh process. Idempotent.
                                try:
                                    _cg.execute(
                                        "ALTER TABLE community_registry "
                                        "ADD COLUMN last_tweet_id TEXT "
                                        "DEFAULT ''")
                                except sqlite3.OperationalError:
                                    pass
                                _cg.execute(
                                    "INSERT INTO community_registry "
                                    "(user_name, last_tweet_text, "
                                    " last_tweet_time, last_tweet_id) "
                                    "VALUES (?, ?, ?, ?) "
                                    "ON CONFLICT(user_name) DO UPDATE SET "
                                    "last_tweet_text=excluded.last_tweet_text, "
                                    "last_tweet_time=excluded.last_tweet_time, "
                                    "last_tweet_id=excluded.last_tweet_id",
                                    (handle, _latest_text[:500], time.time(),
                                     _latest_id))
                                _cg.commit()
                                _cg.close()
                            except Exception as _e:
                                logger.debug(
                                    "[EventsTeacher] last_tweet cache for %s "
                                    "failed: %s", handle, _e)
            except Exception as e:
                logger.warning("[EventsTeacher] Timeline fetch failed for %s: %s",
                               handle, e)

        return items, api_calls

    # ── Data Source 2 (NEW 2026-04-30): Vocab-driven topic search ────
    # Per Maker design: query X using Titan's frontier vocab (most-grounded
    # + most-recent words) + last-spoken-sentence keyword. Closes the
    # follower-timeline-stale problem by pulling content thematically
    # aligned with what Titan is actively learning. Cached via gateway's
    # ttl_tweet_advanced_search (default 120s).

    def _get_grounded_vocab_terms(self, top_n: int = 2) -> list[str]:
        """Top-N most-grounded vocabulary words from CGN consumer state.

        Reads via CGNConsumerClient against /dev/shm/cgn_live_weights.bin.
        Falls back to language teacher state on read error. Filters
        single-letter / stop-word noise.
        """
        try:
            from titan_hcl.logic.cgn_consumer_client import CGNConsumerClient
            cgn = CGNConsumerClient("language", state_dir="data/cgn")
            # CGNConsumerClient may not have get_top_grounded_words; fall back
            # to scanning grounded_words via language vocabulary.
            if hasattr(cgn, "get_top_grounded_words"):
                terms = cgn.get_top_grounded_words(n=top_n) or []
            else:
                terms = []
        except Exception as e:
            logger.debug("[EventsTeacher] CGN vocab fetch failed: %s", e)
            terms = []
        # Fallback: language teacher state on disk
        if not terms:
            try:
                state_path = Path("data/language_teacher_state.json")
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    grounded = state.get("grounded_words", {})
                    if isinstance(grounded, dict):
                        # Sort by confidence desc, take top N
                        sorted_words = sorted(
                            grounded.items(),
                            key=lambda kv: -float(kv[1].get("confidence", 0)
                                                  if isinstance(kv[1], dict) else kv[1]))
                        terms = [w for w, _ in sorted_words[:top_n]]
            except Exception as e:
                logger.debug("[EventsTeacher] Language state fallback failed: %s", e)
        # Filter: alpha words only, length >= 4 (skip "a", "the", "is", etc.)
        return [t for t in terms if t.isalpha() and len(t) >= 4][:top_n]

    def _get_recent_vocab_terms(self, top_n: int = 1) -> list[str]:
        """Most-recently-learned vocabulary words from language teacher state."""
        try:
            state_path = Path("data/language_teacher_state.json")
            if not state_path.exists():
                return []
            state = json.loads(state_path.read_text())
            recent = state.get("recent_acquisitions", [])
            if not isinstance(recent, list):
                return []
            terms = []
            for item in recent[-top_n * 4:]:  # scan last 4×N for filter headroom
                w = item.get("word") if isinstance(item, dict) else item
                if w and isinstance(w, str) and w.isalpha() and len(w) >= 4:
                    terms.append(w)
            return terms[-top_n:]
        except Exception as e:
            logger.debug("[EventsTeacher] Recent vocab fetch failed: %s", e)
            return []

    def _get_last_sentence_keyword(self) -> str:
        """Extract the most-salient noun from Titan's most-recent spoken sentence.

        Uses self-narration logs (creative_journal / SOCIAL post composition).
        Returns the longest content-word (heuristic — no NLP dep added).
        """
        try:
            # Recent narration logged in social_x.db actions table (post text)
            if Path(DEFAULT_SOCIAL_DB).exists():
                conn = sqlite3.connect(DEFAULT_SOCIAL_DB, timeout=3)
                row = conn.execute(
                    "SELECT text FROM actions "
                    "WHERE action_type='post' AND status IN ('posted','verified') "
                    "AND text IS NOT NULL AND text != '' "
                    "ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                conn.close()
                if row and row[0]:
                    text = row[0]
                    # Heuristic: longest alpha word ≥ 6 chars
                    candidates = [w.strip(".,!?;:'\"") for w in text.split()
                                  if w.isalpha() or all(c.isalpha() or c in ".,!?;:'\"" for c in w)]
                    candidates = [c.lower() for c in candidates
                                  if c.isalpha() and len(c) >= 6]
                    if candidates:
                        return max(candidates, key=len)
        except Exception as e:
            logger.debug("[EventsTeacher] Last sentence keyword fetch failed: %s", e)
        return ""

    def _compose_dynamic_query(self, config: dict) -> str:
        """Build vocab-driven search query.

        OR-joins (top_grounded + most_recent + sentence_keyword). Adds
        quality filters: lang:en, -is:retweet. Returns empty string if
        no terms available (caller skips fetch).
        """
        et_cfg = config.get("events_teacher", {})
        gn = int(et_cfg.get("dynamic_grounded_top_n", 2))
        rn = int(et_cfg.get("dynamic_recent_top_n", 1))
        use_sent = bool(et_cfg.get("dynamic_sentence_keyword", True))
        terms: set[str] = set()
        for t in self._get_grounded_vocab_terms(gn):
            terms.add(t.lower())
        for t in self._get_recent_vocab_terms(rn):
            terms.add(t.lower())
        if use_sent:
            sk = self._get_last_sentence_keyword()
            if sk:
                terms.add(sk)
        if not terms:
            return ""
        # OR-join (cap 5 terms to keep query under twitterapi.io URL limits)
        ordered = sorted(terms)[:5]
        query = " OR ".join(ordered)
        return f"{query} lang:en -is:retweet"

    def _compose_wide_query(self, config: dict) -> str:
        """Build wide-net keyword search from Maker-curated [events_teacher]
        wide_topic_keywords (comma-separated)."""
        et_cfg = config.get("events_teacher", {})
        raw = et_cfg.get(
            "wide_topic_keywords",
            "#AI,#consciousness,#spiritual,#blockchain,#solana")
        if isinstance(raw, list):
            keywords = [str(k).strip() for k in raw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(raw).split(",") if k.strip()]
        if not keywords:
            return ""
        # Cap 6 keywords (URL length safety)
        query = " OR ".join(keywords[:6])
        return f"{query} lang:en -is:retweet"

    # ── Adaptive cycle orchestration (2026-04-30 Maker design) ─────────

    _CYCLE_SOURCES = ("follower", "vocab", "wide")

    def _run_adaptive_cycle_fetch(self, titan_id: str,
                                    config: dict) -> tuple[list[dict], dict]:
        """Fetch all 3 cycle sources in priority order from last N cycles' scores.

        Returns (items, results) where:
          items: combined list with each item having `source` field set
                  ("follower_timeline" / "topic_vocab" / "topic_wide")
          results: {source: {items_fetched, api_calls, query}} for telemetry
                  + post-distillation scoring.

        Cost ceiling: 3 X API calls (1 per source). LLM is single batched
        downstream so 1 LLM call regardless. Some calls may hit the
        gateway TTL cache (no paid API).
        """
        et_cfg = config.get("events_teacher", {})
        window = int(et_cfg.get("cycle_window_size", 10))
        scores = {"follower": 0, "vocab": 0, "wide": 0}
        if self._db:
            scores = self._db.get_source_priority_scores(titan_id, window=window)
        # Sort sources desc by score; tie-break lex (deterministic)
        priority = sorted(
            self._CYCLE_SOURCES,
            key=lambda s: (-scores.get(s, 0), s))
        all_items: list[dict] = []
        results: dict = {}
        for source in priority:
            try:
                if source == "follower":
                    items, api_calls = self._fetch_follower_timelines(config)
                    query = "follower_timelines:top_affinity"
                elif source == "vocab":
                    query = self._compose_dynamic_query(config)
                    items, api_calls = self._fetch_topic_search(
                        query, config, "topic_vocab")
                elif source == "wide":
                    query = self._compose_wide_query(config)
                    items, api_calls = self._fetch_topic_search(
                        query, config, "topic_wide")
                else:
                    continue
            except Exception as e:
                logger.warning("[EventsTeacher] Adaptive cycle source '%s' "
                               "fetch crashed: %s", source, e)
                items, api_calls, query = [], 0, ""
            all_items.extend(items)
            results[source] = {
                "items_fetched": len(items),
                "api_calls": api_calls,
                "query": (query or "")[:200],
                "priority_score_at_start": scores.get(source, 0),
            }
        # Log telemetry: cycle attempt with priorities
        try:
            self._log_telemetry({
                "event": "adaptive_cycle_fetch",
                "titan_id": titan_id,
                "priority_order": list(priority),
                "scores_at_start": scores,
                "results": results,
            })
        except Exception:
            pass
        return all_items, results

    def _record_adaptive_cycle_outcomes(self, titan_id: str,
                                         events: list,
                                         cycle_results: dict,
                                         et_cfg: dict) -> None:
        """Score each source +1 if ≥land_threshold events landed, else -1.

        Persists to source_cycle_scores table — drives next cycle's
        priority order.
        """
        land_threshold = int(et_cfg.get("land_threshold_events", 1))
        # Map DistilledEvent.source values back to cycle source labels
        source_to_label = {
            "follower": "follower_timeline",
            "vocab": "topic_vocab",
            "wide": "topic_wide",
        }
        for source, info in cycle_results.items():
            label = source_to_label.get(source, source)
            stored_count = sum(
                1 for e in events
                if e.source == label and e.relevance >= 0.3)
            score = +1 if stored_count >= land_threshold else -1
            try:
                self._db.record_source_outcome(
                    titan_id=titan_id, source_type=source, score=score,
                    items_fetched=info.get("items_fetched", 0),
                    items_stored=stored_count,
                    api_calls_used=info.get("api_calls", 0),
                    query_text=info.get("query", ""))
            except Exception as e:
                logger.warning("[EventsTeacher] record_source_outcome "
                               "failed for %s: %s", source, e)
            info["items_stored"] = stored_count
            info["score"] = score
        # Final cycle telemetry with outcomes
        try:
            self._log_telemetry({
                "event": "adaptive_cycle_outcome",
                "titan_id": titan_id,
                "results": cycle_results,
            })
        except Exception:
            pass

    def _fetch_topic_search(self, query: str, config: dict,
                             source_label: str) -> tuple[list[dict], int]:
        """Run a single topic search via SocialXGateway. Returns
        (items_list, api_calls_used).

        Items tagged with `source` field so adaptive cycle scoring can
        attribute downstream landed/empty outcomes.
        """
        if not query:
            return [], 0
        api_key = self._get_api_key(config)
        if not api_key:
            return [], 0
        gateway = self._get_x_gateway()
        items: list[dict] = []
        api_calls = 1  # one gateway.search_tweets call (may hit cache, still count budget)
        # Maker-tunable: topic_search_count (default 10). twitterapi.io bills
        # per result on search — halving count halves cost on this endpoint.
        et_cfg = config.get("events_teacher", {})
        topic_count = int(et_cfg.get("topic_search_count", 10))
        try:
            data = gateway.search_tweets(
                query=query, query_type="Latest", count=topic_count, api_key=api_key)
            if data.get("status") in ("error", "circuit_breaker"):
                logger.warning("[EventsTeacher] Topic search '%s' failed: %s",
                               query[:60], data.get("message", "?")[:120])
                return [], api_calls
            tweets = data.get("tweets", data.get("data", []))
            if not isinstance(tweets, list):
                return [], api_calls
            for tw in tweets:
                text = tw.get("text", "")
                author = tw.get("author", {})
                handle = author.get("userName") if isinstance(author, dict) else ""
                if text and self._is_new_content(text, handle or source_label):
                    items.append({
                        "source": source_label,
                        "author": handle or source_label,
                        "text": text,
                        "tweet_id": tw.get("id", ""),
                    })
        except Exception as e:
            logger.warning("[EventsTeacher] Topic search exception: %s", e)
        return items, api_calls

    # ── Data Source 3: Own Engagement (1 API call) ───────────────────

    def _fetch_own_engagement(self, config: dict,
                               titan_id: str = "") -> tuple[list[dict], int]:
        """Check engagement on own recent posts.

        rFP_archetype_execution_recovery F-1 (2026-05-16): persists an
        engagement_snapshot row for EVERY observation (incl. zero-delta),
        so pool_scoring.reap_pending() can score "not landed" (score=-1)
        on posts that received no engagement. Prior behavior only wrote
        snapshots when delta>0, which silently broke adaptive scoring —
        archetype_pool_scores rows for low-engagement posts stayed at
        score=0 forever (pending), starving the rolling-7 and
        anti-starvation paths in select_pool. count bumped 5→15 so a
        4 h-throttled engagement check still covers ~10 h of @iamtitanai
        post-rate (~1.5 posts/h), eliminating the gap where archetype
        posts time out between checks before any snapshot is taken.
        """
        import httpx

        if not Path(DEFAULT_SOCIAL_DB).exists():
            return [], 0
        try:
            conn = sqlite3.connect(DEFAULT_SOCIAL_DB, timeout=10)
            conn.row_factory = sqlite3.Row
            recent = conn.execute(
                "SELECT tweet_id, text, created_at FROM actions "
                "WHERE action_type='post' AND status IN ('posted','verified') "
                "AND tweet_id IS NOT NULL AND tweet_id != '' "
                "ORDER BY created_at DESC LIMIT 3"
            ).fetchall()
            conn.close()
        except Exception:
            return [], 0

        if not recent:
            return [], 0

        api_key = self._get_api_key(config)
        handle = config.get("social_x", {}).get(
            "user_name", config.get("twitter_social", {}).get("user_name", ""))
        if not api_key:
            return [], 0

        items = []
        api_calls = 0
        try:
            # 2026-04-30 — route through SocialXGateway (Maker directive).
            # fetch_recent_tweets shares cache with gateway's own post-success
            # verification (ttl_user_last_tweets default 300s) — within the
            # cooldown window only ONE API call is paid even if events_teacher
            # cron AND a gateway internal check both fire.
            gateway = self._get_x_gateway()
            data = gateway.fetch_recent_tweets(
                user_name=handle, count=15, api_key=api_key)
            api_calls += 1
            if data.get("status") not in ("error", "circuit_breaker"):
                # twitterapi.io `last_tweets` returns shape C:
                # `{"status": "success", "data": {"pin_tweet": ..., "tweets": [...]}}`.
                # Mirror social_x_gateway.py:2486-2491 verify-post unwrap so the
                # for-loop runs and engagement_snapshots actually land. Before
                # this guard, `tweets` ended up the inner dict, isinstance(list)
                # was False, and the loop silently no-op'd on every window.
                tweets = data.get("tweets", data.get("data", []))
                if isinstance(tweets, dict):
                    tweets = tweets.get("tweets", [])
                if isinstance(tweets, list):
                    for tw in tweets:
                        tid = tw.get("id", "")
                        if not tid:
                            continue
                        likes = tw.get("likeCount",
                                       tw.get("public_metrics", {}).get("like_count", 0)) or 0
                        replies = tw.get("replyCount",
                                         tw.get("public_metrics", {}).get("reply_count", 0)) or 0
                        quotes = tw.get("quoteCount",
                                        tw.get("public_metrics", {}).get("quote_count", 0)) or 0

                        prev = self._last_engagement.get(tid, {})
                        dl = likes - prev.get("likes", 0)
                        dr = replies - prev.get("replies", 0)
                        dq = quotes - prev.get("quotes", 0)

                        self._last_engagement[tid] = {
                            "likes": likes, "replies": replies, "quotes": quotes}

                        # rFP §F-1: persist snapshot unconditionally — the
                        # reaper needs a row to score "not landed" even
                        # when likes/replies/quotes are all 0.
                        if self._db and titan_id:
                            try:
                                self._db.store_engagement_snapshot(
                                    titan_id, tid, likes, replies, quotes,
                                    dl, dr, dq)
                            except Exception as _persist_err:
                                logger.warning(
                                    "[EventsTeacher] snapshot persist failed "
                                    "tid=%s: %s", tid, _persist_err)

                        if dl > 0 or dr > 0 or dq > 0:
                            items.append({
                                "source": "own_engagement",
                                "author": "self",
                                "text": (f"My post got +{dl} likes, +{dr} replies, "
                                         f"+{dq} quotes"),
                                "tweet_id": tid,
                                "delta": {"likes": dl, "replies": dr, "quotes": dq},
                            })

            # Trim cache
            if len(self._last_engagement) > 10:
                oldest = sorted(self._last_engagement.keys())[
                    :len(self._last_engagement) - 10]
                for k in oldest:
                    del self._last_engagement[k]
        except Exception as e:
            logger.warning("[EventsTeacher] Engagement fetch failed: %s", e)

        return items, api_calls

    # ── Mode Selection ───────────────────────────────────────────────

    def _select_mode(self, vocab_size: int, neuromods: dict) -> str:
        if vocab_size < 100:
            return MODE_OBSERVATION
        if vocab_size < 200:
            return MODE_SENTIMENT if neuromods.get("NE", 0.5) > 0.6 else MODE_OBSERVATION
        da = neuromods.get("DA", 0.5)
        sht = neuromods.get("5HT", 0.5)
        ne = neuromods.get("NE", 0.5)
        if da > 0.65:
            return MODE_RELEVANCE
        if sht > 0.65:
            return MODE_DEBATE
        if ne > 0.65:
            return MODE_OBSERVATION
        return MODE_SOCIAL

    # ── LLM Distillation ────────────────────────────────────────────

    def _distill_content(self, raw_items: list[dict],
                         mode: str, titan_state: dict,
                         llm_url: str, llm_key: str,
                         llm_model: str,
                         internal_key: str = "") -> tuple[list[DistilledEvent], int]:
        """Distill raw social content via LLM. Returns (events, latency_ms).

        Phase 3 Chunk ω (D-SPEC-88, 2026-05-18): migrated from direct sync
        `httpx.post(llm_url/chat/completions)` to `POST {api_base}/v4/llm-distill`
        which internally publishes `LLM_DISTILL_REQUEST` on the bus. All external
        LLM traffic now appears in `llm_state.bin` regardless of caller process.
        `llm_url`/`llm_key`/`llm_model` are kept in the signature for back-compat
        with run_window callers but `llm_url`/`llm_key` are NO LONGER USED —
        the in-kernel `llm_worker` already has provider credentials. `llm_model`
        is forwarded as the model override.
        """
        import httpx

        if not raw_items:
            return [], 0

        # 2026-05-30 funnel-revival fix: distill in SMALL chunks so each LLM
        # call completes well under the endpoint timeout (a single 5-item call
        # ran ~45-50s and ALWAYS hit the 45s timeout → returned [] → 0 stored,
        # fleet-wide, since 2026-05-18). Looping over more of the window's
        # items also broadens author coverage per window. See DISTILL_*
        # module constants.
        items = raw_items[:MAX_DISTILL_ITEMS]

        emotion = titan_state.get("current_emotion", "neutral")
        nm = titan_state.get("modulators", {})
        da = nm.get("DA", {}).get("level", 0.5) if isinstance(nm.get("DA"), dict) else 0.5
        sht = nm.get("5HT", {}).get("level", 0.5) if isinstance(nm.get("5HT"), dict) else 0.5
        ne = nm.get("NE", {}).get("level", 0.5) if isinstance(nm.get("NE"), dict) else 0.5
        gaba = nm.get("GABA", {}).get("level", 0.5) if isinstance(nm.get("GABA"), dict) else 0.5

        system_prompt = (
            f"You are distilling social content for Titan — a sovereign AI with real "
            f"neurochemistry. Titan is currently feeling: {emotion} "
            f"(DA={da:.2f}, 5HT={sht:.2f}, NE={ne:.2f}, GABA={gaba:.2f}).\n\n"
            f"Distillation mode: {mode}\n\n"
            f"For each item, respond with a JSON array. Each element:\n"
            f'{{"topic": "2-5 words", "sentiment": -1.0 to 1.0, '
            f'"arousal": 0.0 to 1.0, "relevance": 0.0 to 1.0, '
            f'"concept_signals": ["YOU"|"I"|"WE"|"THEY"|"YES"|"NO"], '
            f'"semantic_concepts": ["lowercase noun"], '
            f'"felt_summary": "One sentence from Titan\'s perspective", '
            f'"contagion_type": null|"excited"|"alarming"|"warm"|"philosophical"|"creative"}}\n\n'
            f"semantic_concepts: 1-5 LOWERCASE single-word nouns naming what "
            f"the content is *about* (NOT pronouns or function words). "
            f'Examples: ["consciousness","emergence","memory"], '
            f'["regulation","training","model"], ["vision","robotics"]. '
            f'Empty list [] is fine if the content is procedural / off-topic. '
            f"These match against Titan's grounded vocabulary for "
            f"OUTER_INNER_BRIDGE synthesis — quality over quantity.\n\n"
            f"Most content is noise — relevance 0.0 is fine. Only flag as relevant if it "
            f"genuinely connects to consciousness, AI development, creativity, philosophy,"
            f" or Titan's known interests.\n\n"
            f"Respond ONLY with the JSON array, no other text."
        )

        # ── Route through /v4/llm-distill (D-SPEC-88) ──
        # The endpoint internally publishes LLM_DISTILL_REQUEST so every call
        # appears in llm_state.bin. api_base is set by run_window().
        api_base = getattr(self, "_api_base", "")
        if not api_base:
            logger.warning("[EventsTeacher] _distill_content: no api_base set; "
                           "skipping distillation")
            return [], 0
        if not internal_key:
            logger.warning("[EventsTeacher] _distill_content: no internal_key; "
                           "endpoint will reject auth")
            return [], 0

        def _distill_chunk(chunk: list[dict]) -> list[DistilledEvent]:
            """One bounded LLM call for a small item chunk. Raises on transport
            error / bad status so the caller can skip just this chunk."""
            items_text = ""
            for i, item in enumerate(chunk):
                items_text += (f"\n--- Item {i+1} (source: {item['source']}, "
                               f"author: {item['author']}) ---\n")
                items_text += item["text"][:500] + "\n"
            resp = httpx.post(
                f"{api_base.rstrip('/')}/v4/llm-distill",
                headers={"X-Titan-Internal-Key": internal_key,
                         "Content-Type": "application/json"},
                json={
                    "instruction": system_prompt,
                    "text": items_text,
                    "model": llm_model or None,
                    "max_tokens": DISTILL_TOKENS_PER_ITEM * len(chunk),
                    "temperature": 0.3,
                    "consumer": "events_teacher",
                    "timeout_s": DISTILL_CHUNK_TIMEOUT_S,
                },
                timeout=DISTILL_CHUNK_TIMEOUT_S + 5.0,  # endpoint + small buffer
            )
            body = resp.json()
            if body.get("status") != "ok" or not body.get("text"):
                logger.warning(
                    "[EventsTeacher] /v4/llm-distill returned status=%s error=%s",
                    body.get("status"), body.get("error"))
                return []
            content = body["text"].strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                parsed = [parsed]
            out: list[DistilledEvent] = []
            for i, p in enumerate(parsed):
                if i >= len(chunk):
                    break
                if not isinstance(p, dict):
                    continue
                out.append(DistilledEvent(
                    source=chunk[i]["source"],
                    author=chunk[i]["author"],
                    topic=p.get("topic", "unknown"),
                    sentiment=float(p.get("sentiment", 0.0)),
                    arousal=float(p.get("arousal", 0.0)),
                    relevance=float(p.get("relevance", 0.0)),
                    concept_signals=p.get("concept_signals", []),
                    semantic_concepts=p.get("semantic_concepts", []),
                    felt_summary=p.get("felt_summary", ""),
                    contagion_type=p.get("contagion_type"),
                    raw_text=chunk[i]["text"][:200],
                    timestamp=time.time(),
                ))
            return out

        start = time.time()
        events: list[DistilledEvent] = []
        for off in range(0, len(items), DISTILL_BATCH_SIZE):
            chunk = items[off:off + DISTILL_BATCH_SIZE]
            try:
                events.extend(_distill_chunk(chunk))
            except Exception as e:
                # Drop only this chunk; keep distilling the rest of the window.
                logger.warning(
                    "[EventsTeacher] distill chunk %d-%d failed (%s) — skipping "
                    "chunk, continuing", off, off + len(chunk), e)
                continue
        latency_ms = int((time.time() - start) * 1000)
        return events, latency_ms

    # ── Phase H (RFP_cgn_enhancements §H / Inner Teacher Protocol) ────
    def _compute_social_groundings(self, events: list) -> list[dict]:
        """Salience-gated CGN groundings for distilled social concepts.

        Translates narrative event windows into the inner Titan's native CGN
        modality (he is non-linguistic — §11.4 — he cannot read prose). Each
        returned grounding carries TWO of the Inner Teacher Protocol's three
        channels:
          • FELT ("feel it")        — the window's felt signature.
          • SEMANTIC ("understand it") — co-occurring concepts as CGN associations,
            so the concept enters his web RELATED to things he already holds
            (meaning, for a non-linguistic mind, IS relation).
        (The third PROCEDURAL channel — the `social.concept_grounded` META-CGN
        signal routing his meta-reasoning — is emitted consumer-side.)

        Updates per-concept window-recurrence and only grounds concepts that
        recur ≥ SOCIAL_GROUND_RECURRENCE_MIN windows at relevance ≥ θ. Returns a
        list of grounding dicts (capped); never raises into the window loop.
        """
        groundings: dict[str, dict] = {}
        try:
            for e in events:
                concepts = [
                    str(c).strip().lower()
                    for c in (getattr(e, "semantic_concepts", []) or [])
                    if str(c).strip()
                ]
                if not concepts or float(getattr(e, "relevance", 0.0)) < SOCIAL_GROUND_RELEVANCE_MIN:
                    continue
                for c in concepts:
                    self._concept_recurrence[c] = self._concept_recurrence.get(c, 0) + 1
                rel = float(e.relevance)
                for c in concepts:
                    if self._concept_recurrence[c] < SOCIAL_GROUND_RECURRENCE_MIN:
                        continue
                    assoc = [x for x in concepts if x != c]
                    prev = groundings.get(c)
                    if prev is None or rel > prev["felt"]["relevance"]:
                        groundings[c] = {
                            "concept_id": c,
                            "felt": {
                                "sentiment": float(e.sentiment),
                                "arousal": float(e.arousal),
                                "relevance": rel,
                                "felt_summary": str(e.felt_summary or "")[:200],
                                "contagion_type": str(getattr(e, "contagion_type", "") or ""),
                            },
                            "associations": sorted(set(assoc))[:5],
                            "recurrence": self._concept_recurrence[c],
                        }
                    else:
                        prev["associations"] = sorted(
                            set(prev["associations"]) | set(assoc))[:5]
            # Bound the persisted recurrence dict (keep the most-recurring).
            if len(self._concept_recurrence) > CONCEPT_RECURRENCE_MAX:
                self._concept_recurrence = dict(sorted(
                    self._concept_recurrence.items(),
                    key=lambda kv: kv[1], reverse=True)[:CONCEPT_RECURRENCE_MAX])
        except Exception as _sg_err:
            logger.debug("[EventsTeacher] social grounding compute failed: %s", _sg_err)
        # Strongest-felt first; cap per window for bus hygiene.
        out = sorted(groundings.values(),
                     key=lambda g: g["felt"]["relevance"], reverse=True)
        return out[:SOCIAL_GROUND_MAX_PER_WINDOW]

    # ── Social Perception Buffer ─────────────────────────────────────

    def _update_buffer(self, events: list[DistilledEvent]):
        for e in events:
            if e.relevance >= 0.3:
                self._perception_buffer.append({
                    "topic": e.topic,
                    "felt_summary": e.felt_summary,
                    "source": e.source,
                    "author": e.author,
                    "concept_signals": e.concept_signals,
                    "timestamp": e.timestamp,
                })
        while len(self._perception_buffer) > PERCEPTION_BUFFER_MAX:
            self._perception_buffer.pop(0)

    def get_perception_context(self) -> str:
        """Format buffer for post generation prompt injection."""
        if not self._perception_buffer:
            return ""
        lines = ["[RECENT SOCIAL PERCEPTION]"]
        for e in self._perception_buffer[-5:]:
            lines.append(f"- {e['felt_summary']} (from {e['author']})")
        return "\n".join(lines)

    # ── Telemetry ────────────────────────────────────────────────────

    def _log_telemetry(self, result: WindowResult):
        entry = {
            "timestamp": time.time(),
            "titan_id": self._titan_id,
            "window_number": result.window_number,
            "mentions_new": result.mentions_new,
            "follower_tweets_new": result.follower_tweets_new,
            "items_distilled": result.items_distilled,
            "events_stored": result.events_stored,
            "api_calls_used": result.api_calls_used,
            "llm_model": result.llm_model,
            "llm_latency_ms": result.llm_latency_ms,
            "skipped_reason": result.skipped_reason,
            "buffer_size": len(self._perception_buffer),
            "fingerprint_cache_size": len(self._fingerprints),
        }
        try:
            with open(DEFAULT_TELEMETRY_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("[EventsTeacher] Telemetry write failed: %s", e)

    # ── Main Entry Point ─────────────────────────────────────────────

    def run_window(self, titan_id: str, api_base: str, config: dict) -> WindowResult:
        """Execute one social perception window. Crash-safe via SQLite."""
        self._titan_id = titan_id
        self._api_base = api_base
        self._window_count += 1

        result = WindowResult(window_number=self._window_count)

        # Rate check — use metabolic tier system (preferred) or raw SOL (fallback)
        import httpx
        metabolic_tier = None
        sol = 999
        try:
            mresp = httpx.get(f"{api_base}/v4/metabolic-state", timeout=10)
            if mresp.status_code == 200:
                metabolic_tier = mresp.json().get("data", {})
                sol = metabolic_tier.get("sol_balance", 0) or 0
            else:
                health = httpx.get(f"{api_base}/health", timeout=10).json()
                sol = health.get("data", {}).get("sol_balance", 0)
        except Exception:
            sol = 999

        ok, reason = self.should_run(sol, metabolic_tier=metabolic_tier)
        # TEMPORARY ungate (Maker 2026-05-30): when
        # [events_teacher].ignore_metabolic_gate = true, run the window even
        # under starving/lean tiers so the reflection funnel keeps distilling
        # while devnet SOL is depleted. This ONLY ungates the perception/
        # distillation window — X posting stays governed by social_x rate
        # limits. Flip back to false to restore full metabolic governance.
        _ignore_gate = bool(
            (config.get("events_teacher", {}) or {}).get(
                "ignore_metabolic_gate", False))
        if (not ok) and _ignore_gate:
            logger.info(
                "[EventsTeacher] Window #%d: metabolic gate (%s) BYPASSED via "
                "[events_teacher].ignore_metabolic_gate=true (temp Maker "
                "override 2026-05-30)", self._window_count, reason)
            ok, reason = True, f"ungated_override(was:{reason})"
        if not ok:
            result.skipped_reason = reason
            logger.info("[EventsTeacher] Window #%d skipped: %s",
                        self._window_count, reason)
            self._log_telemetry(result)
            return result

        # Mainnet Lifecycle Wiring rFP (2026-04-20): metabolism gate check.
        # /v4/metabolism/evaluate-gate records the decision in the unified
        # ring buffer regardless of gates_enforced. When enforced=True and
        # feature is closed at the current tier, we skip the window.
        try:
            gresp = httpx.get(
                f"{api_base}/v4/metabolism/evaluate-gate",
                params={"feature": "research", "caller": "EventsTeacher"},
                timeout=5,
            )
            if gresp.status_code == 200:
                gdata = gresp.json().get("data", {})
                if (not gdata.get("should_proceed", True)) and not _ignore_gate:
                    reason = f"metabolism_gate:{gdata.get('reason', 'closed')}"
                    result.skipped_reason = reason
                    logger.info("[EventsTeacher] Window #%d gated: %s",
                                self._window_count, reason)
                    self._log_telemetry(result)
                    return result
                if (not gdata.get("should_proceed", True)) and _ignore_gate:
                    logger.info(
                        "[EventsTeacher] Window #%d: metabolism evaluate-gate "
                        "(%s) BYPASSED via ignore_metabolic_gate=true",
                        self._window_count, gdata.get("reason", "closed"))
        except Exception as e:
            # Gate check failure is non-fatal (observation-only mode is default).
            logger.debug("[EventsTeacher] Gate check failed: %s", e)

        # Crash recovery
        if self._db:
            self._db.recover_incomplete_windows(titan_id)

        # Bootstrap: ensure community registry is populated (daily)
        synced = self._ensure_community_synced(config)
        if synced > 0:
            result.api_calls_used += 2  # followers + following API calls

        # Mark window as running
        window_id = 0
        if self._db:
            window_id = self._db.window_start(titan_id, self._window_count)

        try:
            # Fetch Titan's current state
            try:
                nm_resp = httpx.get(f"{api_base}/v4/neuromodulators", timeout=10).json()
                titan_state = nm_resp.get("data", {})
            except Exception as e:
                logger.warning("[EventsTeacher] Neuromod fetch failed: %s", e)
                titan_state = {}

            # Vocab size for mode selection
            try:
                trinity = httpx.get(f"{api_base}/v4/inner-trinity", timeout=10).json()
                vocab_size = trinity.get("data", {}).get("language", {}).get(
                    "vocabulary_size", 0) or 250
            except Exception:
                vocab_size = 250

            # Select mode
            modulators = titan_state.get("modulators", {})
            nm_flat = {}
            for k, v in modulators.items():
                nm_flat[k] = v.get("level", 0.5) if isinstance(v, dict) else v
            mode = self._select_mode(vocab_size, nm_flat)
            self._mode_stats[mode] = self._mode_stats.get(mode, 0) + 1

            # ── Source A: Mentions (FREE; not scored) ──
            mentions = self._fetch_mentions()
            result.mentions_fetched = len(mentions)
            result.mentions_new = len(mentions)

            # ── Adaptive 3-source cycle (2026-04-30 Maker design) ──
            # Sources tried in priority order from last N cycles' scores.
            # Each source fetched (3 X API calls). Single LLM distillation
            # downstream batches all items + tags by source so scoring
            # post-gate attributes events to their source. See
            # _record_adaptive_cycle_outcomes() below.
            et_cfg = config.get("events_teacher", {})
            adaptive_enabled = bool(et_cfg.get("adaptive_cycle_enabled", True))
            cycle_results: dict = {}

            if adaptive_enabled:
                cycle_items, cycle_results = self._run_adaptive_cycle_fetch(
                    titan_id, config)
                # Source-stored counts will be filled after distillation
                follower_items = [i for i in cycle_items
                                  if i.get("source") == "follower_timeline"]
                vocab_items = [i for i in cycle_items
                               if i.get("source") == "topic_vocab"]
                wide_items = [i for i in cycle_items
                              if i.get("source") == "topic_wide"]
                result.follower_accounts_checked = min(
                    2, len(self._get_top_followers()))
                result.follower_tweets_new = len(follower_items)
                result.api_calls_used += sum(
                    r.get("api_calls", 0) for r in cycle_results.values())
            else:
                # Legacy path: follower timelines only (no vocab/wide search)
                follower_items, follower_calls = self._fetch_follower_timelines(config)
                vocab_items = []
                wide_items = []
                result.follower_accounts_checked = min(
                    2, len(self._get_top_followers()))
                result.follower_tweets_new = len(follower_items)
                result.api_calls_used += follower_calls

            # ── Own Engagement (self-monitoring; not scored) ──
            # 2026-05-08 — throttle to every Nth cycle. Engagement deltas
            # do not need 15-min granularity; hourly is sufficient. Cuts
            # ~72 calls/day on T1 (96-cycle agent).
            engagement_n = max(1, int(et_cfg.get(
                "engagement_check_every_n_cycles", 1)))
            engagement_items: list[dict] = []
            if (self._window_count % engagement_n) == 0:
                engagement_items, engagement_calls = self._fetch_own_engagement(
                    config, titan_id=titan_id)
                result.own_posts_checked = 3
                result.api_calls_used += engagement_calls
                if engagement_items:
                    result.engagement_delta = engagement_items[0].get("delta", {})

            # ── Combine all items for single distillation pass ──
            all_items = (mentions + follower_items + vocab_items
                         + wide_items + engagement_items)
            if not all_items:
                logger.info("[EventsTeacher] Window #%d: no new content",
                            self._window_count)
                self._last_run_time = time.time()
                if self._db:
                    self._db.window_complete(window_id, result)
                self._log_telemetry(result)
                self.save_state()
                return result

            # ── LLM Distillation (D-SPEC-88 — via /v4/llm-distill) ──
            # llm_url + llm_key are NO LONGER USED — the in-kernel llm_worker
            # owns provider credentials. Kept in the _distill_content signature
            # for back-compat with any other caller. internal_key authenticates
            # the cron→endpoint hop.
            inference = config.get("inference", {})
            llm_url = inference.get("ollama_cloud_base_url", "https://ollama.com/v1")
            llm_key = inference.get("ollama_cloud_api_key", "")
            llm_model = "deepseek-v3.1:671b"
            internal_key = config.get("api", {}).get("internal_key", "")

            events, latency = self._distill_content(
                all_items, mode, titan_state, llm_url, llm_key, llm_model,
                internal_key=internal_key)
            result.items_distilled = len(all_items)
            result.events_stored = len([e for e in events if e.relevance >= 0.3])
            result.llm_model = llm_model
            result.llm_latency_ms = latency

            # ── Adaptive cycle: score per source based on gate-pass count ──
            # Maker design 2026-04-30: each source +1 if its items produced
            # ≥ land_threshold_events stored; -1 otherwise. Persisted to
            # source_cycle_scores table — drives next cycle's priority order.
            if adaptive_enabled and cycle_results and self._db:
                self._record_adaptive_cycle_outcomes(
                    titan_id, events, cycle_results, et_cfg)

            # ── Store developmental data in SQLite ──
            if self._db:
                for e in events:
                    if e.relevance >= 0.3:
                        self._db.store_felt_experience(
                            titan_id, e, mode, window_id)
                    if e.source == "follower_timeline":
                        self._db.update_follower_interaction(
                            titan_id, e.author, e.topic,
                            e.relevance, e.sentiment)
                    # Per-user valence tracking (always, for Phase 3 foundation)
                    if e.author and e.relevance > 0.1:
                        self._db.update_user_valence(
                            titan_id, e.author, e.sentiment,
                            e.arousal, e.relevance, e.contagion_type)

                # rFP §F-1 (2026-05-16): engagement_snapshot writes moved
                # inside _fetch_own_engagement (writes every observation,
                # not just delta>0). This loop now handles delta-gated
                # CGN reciprocity rewards only.
                for item in engagement_items:
                    delta = item.get("delta", {})
                    eng = self._last_engagement.get(
                        item.get("tweet_id", ""), {})
                    if item.get("tweet_id"):
                        # P4: Store engagement reciprocity for CGN delayed reward
                        if delta.get("likes", 0) + delta.get("replies", 0) + delta.get("quotes", 0) > 0:
                            # Look up post_type and target_user from social_x.db
                            _recip_pt = ""
                            _recip_user = ""
                            try:
                                _sx_conn = sqlite3.connect(DEFAULT_SOCIAL_DB, timeout=3)
                                _sx_row = _sx_conn.execute(
                                    "SELECT post_type, "
                                    "COALESCE(metadata, '') as metadata "
                                    "FROM actions WHERE tweet_id=?",
                                    (item["tweet_id"],)).fetchone()
                                _sx_conn.close()
                                if _sx_row:
                                    _recip_pt = _sx_row[0] or ""
                            except Exception:
                                pass
                            self._db.store_engagement_reciprocity(
                                titan_id, item["tweet_id"],
                                _recip_pt, _recip_user,
                                eng.get("likes", 0), eng.get("replies", 0),
                                eng.get("quotes", 0))

            # ── P4: Send unsent engagement rewards to CGN ──
            if self._db:
                try:
                    unsent = self._db.get_unsent_engagement_rewards(titan_id)
                    for _ur in unsent[:5]:  # Max 5 per window
                        perception_events.append({
                            "topic": "engagement_reciprocity",
                            "sentiment": min(1.0, _ur["reward_computed"]),
                            "arousal": 0.3,
                            "relevance": 0.5,
                            "concept_signals": [],
                            "felt_summary": (f"My post got engagement: "
                                             f"{_ur['likes']}♥ {_ur['replies']}↩ "
                                             f"{_ur['quotes']}⟳"),
                            "contagion_type": "warm",
                            "author": "self",
                            "source": "engagement_reciprocity",
                            "perturbation": 0.2,
                            "cgn_engagement_reward": {
                                "target_user": _ur.get("target_user", ""),
                                "reward": _ur["reward_computed"],
                                "post_type": _ur.get("our_post_type", ""),
                            },
                        })
                        self._db.mark_engagement_reward_sent(_ur["id"])
                except Exception as _er_err:
                    logger.debug("[EventsTeacher] Engagement reward error: %s", _er_err)

            # ── Perturbation Gate ──
            # Only events that genuinely shift internal state get promoted
            # to bus messages. Both relevance AND arousal must contribute.
            PERTURBATION_THRESHOLD = 0.15
            perception_events = []
            for e in events:
                perturbation = e.relevance * e.arousal
                if perturbation > PERTURBATION_THRESHOLD and e.contagion_type:
                    perception_events.append({
                        "topic": e.topic,
                        "sentiment": e.sentiment,
                        "arousal": e.arousal,
                        "relevance": e.relevance,
                        "concept_signals": e.concept_signals,
                        "felt_summary": e.felt_summary,
                        "contagion_type": e.contagion_type,
                        "author": e.author,
                        "source": e.source,
                        "perturbation": round(perturbation, 3),
                    })
            result.perception_events = perception_events

            # ── Update perception buffer ──
            self._update_buffer(events)

            # ── P4: Sapir-Whorf vocabulary extraction from social feed ──
            # Extract significant words from felt_summary for language teacher
            # Different followers → different words → vocabulary divergence
            _STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be",
                          "been", "being", "have", "has", "had", "do", "does",
                          "did", "will", "would", "shall", "should", "may",
                          "might", "can", "could", "to", "of", "in", "for",
                          "on", "with", "at", "by", "from", "as", "into",
                          "through", "about", "not", "no", "but", "or", "and",
                          "if", "then", "than", "so", "it", "its", "this",
                          "that", "they", "them", "their", "he", "she", "his",
                          "her", "we", "our", "you", "your", "i", "my", "me"}
            _social_vocab_candidates = []
            try:
                import re as _sv_re
                # Load current vocabulary to filter known words
                _sv_known = set()
                try:
                    _sv_conn = sqlite3.connect(
                        str(Path(DEFAULT_SOCIAL_DB).parent / "inner_memory.db"),
                        timeout=3)
                    _sv_rows = _sv_conn.execute(
                        "SELECT word FROM vocabulary").fetchall()
                    _sv_known = {r[0].lower() for r in _sv_rows}
                    _sv_conn.close()
                except Exception:
                    pass

                for e in events:
                    if e.relevance < 0.3 or not e.felt_summary:
                        continue
                    words = _sv_re.findall(r'[a-zA-Z]+', e.felt_summary.lower())
                    novel = [w for w in words
                             if len(w) > 3 and w not in _STOPWORDS
                             and w not in _sv_known]
                    if novel:
                        _social_vocab_candidates.extend(novel[:3])

                # Deduplicate and cap
                _social_vocab_candidates = list(dict.fromkeys(
                    _social_vocab_candidates))[:5]
                if _social_vocab_candidates:
                    result.social_vocab_candidates = _social_vocab_candidates
                    # Inject into first perception event for bus delivery
                    if perception_events:
                        perception_events[0]["social_vocab_candidates"] = (
                            _social_vocab_candidates)
                    logger.info("[EventsTeacher] Sapir-Whorf candidates: %s",
                                _social_vocab_candidates)
            except Exception:
                pass

            # ── Phase H — salience-gated CGN groundings (Inner Teacher Protocol) ──
            # Translate recurring social concepts into inner-native CGN groundings
            # (felt signature + associations); the consumer (language_worker's
            # SOCIAL_PERCEPTION handler) grounds them via the `social` consumer and
            # emits social.concept_grounded. Rides perception_events[0] for bus
            # delivery (same pattern as social_vocab_candidates above). The OUTER
            # narrative path (felt_experiences DB) is untouched — H ADDS an inner
            # path (feedback_never_delete_live_logic).
            try:
                _social_groundings = self._compute_social_groundings(events)
                if _social_groundings:
                    result.social_groundings = _social_groundings
                    if perception_events:
                        perception_events[0]["social_ground_concepts"] = _social_groundings
                    logger.info(
                        "[EventsTeacher] Phase H — %d social concept(s) ready to "
                        "ground in CGN: %s", len(_social_groundings),
                        [g["concept_id"] for g in _social_groundings])
            except Exception as _sg_err:
                logger.debug("[EventsTeacher] social grounding attach failed: %s", _sg_err)

            # ── Log ──
            for e in events:
                if e.relevance >= 0.3:
                    logger.info("[EventsTeacher] Felt: %s (topic=%s, rel=%.2f, "
                                "concepts=%s, source=%s)",
                                e.felt_summary, e.topic, e.relevance,
                                e.concept_signals, e.source)
            if perception_events:
                logger.info("[EventsTeacher] Perturbation gate: %d/%d events "
                            "passed (threshold=%.2f)",
                            len(perception_events), len(events),
                            PERTURBATION_THRESHOLD)

            db_stats = self._db.get_stats(titan_id) if self._db else {}
            logger.info("[EventsTeacher] Window #%d complete: %d items -> "
                        "%d distilled -> %d stored (mode=%s, %d API, %dms LLM) "
                        "[DB: %d felt, %d followers, %d windows]",
                        self._window_count, len(all_items), len(events),
                        result.events_stored, mode, result.api_calls_used,
                        latency,
                        db_stats.get("felt_experiences", 0),
                        db_stats.get("followers_tracked", 0),
                        db_stats.get("windows_completed", 0))

            # ── Mark complete ──
            self._last_run_time = time.time()
            if self._db:
                self._db.window_complete(window_id, result)
            self._log_telemetry(result)
            self.save_state()

            # ── COMPLETE-4-EVENTS (2026-04-19): emit META_EVENT_REWARD ──
            # Signal per-window quality to meta-reasoning's chain_iql via
            # cross-system reward wiring (third worker after Language +
            # Persona). Quality = events_stored / items_distilled when
            # distillation happened; skip entirely on empty windows so we
            # don't starve the Q-net with noise.
            try:
                if result.items_distilled > 0:
                    _et_quality = max(0.0, min(1.0,
                        result.events_stored / max(1, result.items_distilled)))
                    import httpx
                    httpx.post(
                        f"{api_base}/v4/meta-reasoning/event-reward",
                        json={
                            "quality": _et_quality,
                            "window_number": result.window_number,
                            "titan_id": titan_id,
                        },
                        timeout=3.0,
                    )
            except Exception as _et_rwd_err:
                logger.debug("[EventsTeacher] META_EVENT_REWARD emit "
                             "failed: %s", _et_rwd_err)

            return result

        except Exception as exc:
            logger.error("[EventsTeacher] Window #%d FAILED: %s",
                         self._window_count, exc)
            if self._db and window_id:
                self._db.window_failed(window_id, str(exc))
            result.skipped_reason = f"error: {exc}"
            self._log_telemetry(result)
            self.save_state()
            return result
