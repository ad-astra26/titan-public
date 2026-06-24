"""Tests for the EventsTeacherReader read-only Observatory surface.

Covers rFP Observatory §5.2 — /v6/events-teacher/{feed,followers,impact}.
Validates: real read shapes, JSON-column parsing, window filtering,
valence merge, and the cold-start (missing DB) empty-state path.

Run isolated:
    python -m pytest tests/test_events_teacher_api.py -v -p no:anchorpy --tb=short
"""
import json
import sqlite3
import time

import pytest

from titan_hcl.logic.events_teacher import EventsTeacherReader


def _seed(path: str) -> None:
    """Build a minimal events_teacher.db with the real schema + a few rows."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE felt_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, titan_id TEXT NOT NULL,
            source TEXT NOT NULL, author TEXT NOT NULL, topic TEXT NOT NULL,
            sentiment REAL DEFAULT 0.0, arousal REAL DEFAULT 0.0,
            relevance REAL DEFAULT 0.0, concept_signals TEXT,
            felt_summary TEXT NOT NULL, contagion_type TEXT, mode TEXT,
            window_id INTEGER, created_at REAL NOT NULL,
            semantic_concepts TEXT DEFAULT '', author_id TEXT DEFAULT '',
            author_followers INTEGER DEFAULT 0);
        CREATE TABLE follower_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, titan_id TEXT NOT NULL,
            handle TEXT NOT NULL, times_checked INTEGER DEFAULT 0,
            topics_seen TEXT, accumulated_relevance REAL DEFAULT 0.0,
            last_sentiment REAL DEFAULT 0.0, last_checked_at REAL,
            created_at REAL NOT NULL, UNIQUE(titan_id, handle));
        CREATE TABLE user_valence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, titan_id TEXT NOT NULL,
            handle TEXT NOT NULL, valence REAL DEFAULT 0.0,
            interaction_count INTEGER DEFAULT 0, last_sentiment REAL DEFAULT 0.0,
            last_arousal REAL DEFAULT 0.0, last_contagion_type TEXT,
            updated_at REAL NOT NULL, UNIQUE(titan_id, handle));
        CREATE TABLE window_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, titan_id TEXT NOT NULL,
            window_number INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'running',
            mode TEXT, mentions_new INTEGER DEFAULT 0,
            follower_tweets_new INTEGER DEFAULT 0, items_distilled INTEGER DEFAULT 0,
            events_stored INTEGER DEFAULT 0, api_calls_used INTEGER DEFAULT 0,
            llm_latency_ms INTEGER DEFAULT 0, skipped_reason TEXT,
            error_message TEXT, started_at REAL NOT NULL, completed_at REAL);
        CREATE TABLE source_cycle_scores (
            cycle_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
            titan_id TEXT NOT NULL, source_type TEXT NOT NULL, score INTEGER NOT NULL,
            items_fetched INTEGER NOT NULL, items_stored INTEGER NOT NULL,
            api_calls_used INTEGER NOT NULL, query_text TEXT, window_number INTEGER DEFAULT 0);
        """
    )
    now = time.time()
    conn.executemany(
        "INSERT INTO felt_experiences (titan_id, source, author, topic, sentiment, "
        "arousal, relevance, concept_signals, felt_summary, contagion_type, mode, "
        "window_id, created_at, semantic_concepts, author_followers) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("T1", "topic_wide", "alice", "AI", 0.6, 0.7, 0.9, json.dumps(["THEY"]),
             "felt hopeful", "philosophical", "social", 5, now - 100,
             json.dumps(["research", "truth"]), 4200),
            ("T1", "mention", "bob", "music", -0.3, 0.4, 0.5, json.dumps(["YOU"]),
             "felt challenged", "critical", "social", 6, now - 200,
             json.dumps(["music"]), 8),
            ("T1", "follower_timeline", "carol", "AI", 0.2, 0.2, 0.3, None,
             "felt neutral", "philosophical", "observation", 7, now - 5_000_000,
             "", 1),  # old row → outside 24h window
        ],
    )
    conn.executemany(
        "INSERT INTO follower_interactions (titan_id, handle, times_checked, topics_seen, "
        "accumulated_relevance, last_sentiment, last_checked_at, created_at) VALUES "
        "(?,?,?,?,?,?,?,?)",
        [
            ("T1", "alice", 10, json.dumps(["AI", "music"]), 8.5, 0.6, now, now - 1000),
            ("T1", "bob", 3, json.dumps(["music"]), 2.1, -0.3, now, now - 2000),
        ],
    )
    conn.executemany(
        "INSERT INTO user_valence (titan_id, handle, valence, interaction_count, "
        "last_sentiment, last_arousal, last_contagion_type, updated_at) VALUES "
        "(?,?,?,?,?,?,?,?)",
        [("T1", "alice", 0.42, 10, 0.6, 0.7, "philosophical", now)],
    )
    conn.executemany(
        "INSERT INTO window_log (titan_id, window_number, status, mentions_new, "
        "follower_tweets_new, items_distilled, events_stored, api_calls_used, started_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [("T1", 1, "complete", 4, 2, 6, 6, 12, now - 300),
         ("T1", 2, "complete", 0, 0, 0, 0, 3, now - 600)],
    )
    conn.executemany(
        "INSERT INTO source_cycle_scores (timestamp, titan_id, source_type, score, "
        "items_fetched, items_stored, api_calls_used) VALUES (?,?,?,?,?,?,?)",
        [(now - 300, "T1", "follower", 1, 5, 5, 2),
         (now - 350, "T1", "wide", -1, 0, 0, 2)],
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def reader(tmp_path):
    db = tmp_path / "events_teacher.db"
    _seed(str(db))
    return EventsTeacherReader(str(db))


def test_feed_shape_and_json_parsing(reader):
    rows = reader.feed("T1", n=10)
    assert len(rows) == 3
    top = rows[0]  # most recent first
    assert top["author"] == "alice"
    # JSON TEXT columns parsed into real objects
    assert top["concept_signals"] == ["THEY"]
    assert top["semantic_concepts"] == ["research", "truth"]
    # affect triple present (richer than get_social_memory)
    assert top["sentiment"] == 0.6 and top["arousal"] == 0.7 and top["relevance"] == 0.9
    assert top["author_followers"] == 4200


def test_feed_since_filter(reader):
    recent = reader.feed("T1", n=10, since=time.time() - 24 * 3600)
    assert len(recent) == 2  # the 5M-second-old carol row is excluded


def test_feed_empty_for_unknown_titan(reader):
    assert reader.feed("T2", n=10) == []


def test_followers_merges_valence(reader):
    rows = reader.followers("T1", n=10)
    assert len(rows) == 2
    alice = next(r for r in rows if r["handle"] == "alice")
    assert alice["accumulated_relevance"] == 8.5
    assert alice["topics_seen"] == ["AI", "music"]  # parsed
    assert alice["valence"] == 0.42  # merged from user_valence
    assert alice["interaction_count"] == 10
    # bob has no valence row → graceful None
    bob = next(r for r in rows if r["handle"] == "bob")
    assert bob["valence"] is None


def test_impact_aggregates(reader):
    data = reader.impact("T1", window_hours=24)
    assert data["totals"]["felt_experiences"] == 3
    assert data["totals"]["followers_tracked"] == 2
    assert data["totals"]["windows_completed"] == 2
    # affect window excludes the old carol row → 2 in-window
    assert data["affect"]["count"] == 2
    assert data["affect"]["avg_sentiment"] == pytest.approx((0.6 - 0.3) / 2, abs=1e-6)
    assert {r["contagion_type"] for r in data["affect"]["by_contagion"]} == {
        "philosophical", "critical"}
    # pipeline throughput summed over window
    assert data["pipeline"]["items_distilled"] == 6
    assert data["pipeline"]["windows"] == 2
    # source effectiveness: follower landed, wide empty
    by_src = {r["source_type"]: r for r in data["sources"]}
    assert by_src["follower"]["landed"] == 1
    assert by_src["wide"]["empty"] == 1


def test_missing_db_raises_operational_error(tmp_path):
    """The api handler relies on OperationalError to fall back to empty-state."""
    reader = EventsTeacherReader(str(tmp_path / "does_not_exist.db"))
    with pytest.raises(sqlite3.OperationalError):
        reader.feed("T1", n=5)
