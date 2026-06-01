"""Tests for EventsTeacher._fetch_own_engagement response-shape parsing.

Closes rFP_social_x_improvements §B.3.F-1 (RC-1) — engagement_snapshots
table was empty on T1+T2+T3 because the twitterapi.io `last_tweets`
endpoint returns shape C (`{"data": {"tweets": [...]}}`) but the parser
only unwrapped shapes A (`{"tweets": [...]}`) and B (`{"data": [...]}`).
Mirror gateway's verify-post unwrap at `social_x_gateway.py:2486-2491`.

These tests are the regression bar — if the response-shape unwrap
breaks again, this file fails before the bug reaches @example_handle.
"""
import sqlite3
import time
from unittest.mock import patch, MagicMock

import pytest

from titan_hcl.logic.events_teacher import EventsTeacher, EventsTeacherDB


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def teacher_with_db(tmp_path):
    """EventsTeacher wired to a temp EventsTeacherDB."""
    db_path = tmp_path / "events_teacher.db"
    db = EventsTeacherDB(str(db_path))
    t = EventsTeacher()
    t._db = db
    return t, db, tmp_path


def _seed_social_x_with_posts(social_x_path, tweet_ids):
    """Seed social_x.db with verified posts so _fetch_own_engagement
    finds candidates to score."""
    conn = sqlite3.connect(str(social_x_path), timeout=5)
    conn.execute(
        "CREATE TABLE actions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_type TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', "
        "tweet_id TEXT, text TEXT, "
        "created_at REAL NOT NULL)"
    )
    now = time.time()
    for i, tid in enumerate(tweet_ids):
        conn.execute(
            "INSERT INTO actions (action_type, status, tweet_id, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("post", "verified", tid, f"sample post {i}", now - i * 60),
        )
    conn.commit()
    conn.close()


def _shape_c_response(tweet_ids, *, likes=2, replies=1, quotes=0):
    """twitterapi.io `last_tweets` actual response shape (verified live
    2026-05-17 via direct probe with valid API key)."""
    return {
        "status": "success",
        "code": 0,
        "msg": "",
        "has_next_page": False,
        "next_cursor": "",
        "data": {
            "pin_tweet": None,
            "tweets": [
                {"id": tid, "likeCount": likes,
                 "replyCount": replies, "quoteCount": quotes,
                 "text": f"sample {tid}"}
                for tid in tweet_ids
            ],
        },
    }


# ── Core regression test: shape C is unwrapped ───────────────────────


def test_shape_c_response_writes_snapshots(teacher_with_db):
    """The bug: twitterapi.io returns `{data: {tweets: [...]}}` and the
    parser must unwrap the dict before iterating. Pre-fix, the inner
    for-loop never ran and engagement_snapshots stayed at 0."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["111", "222", "333"])

    mock_gateway = MagicMock()
    mock_gateway.fetch_recent_tweets.return_value = _shape_c_response(
        ["111", "222", "333"])

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        items, api_calls = teacher._fetch_own_engagement(
            config, titan_id="T1")

    # Gateway was called once
    assert api_calls == 1
    assert mock_gateway.fetch_recent_tweets.call_count == 1

    # All 3 tweets produced engagement_snapshot rows
    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    rows = conn.execute(
        "SELECT tweet_id, likes, replies, quotes FROM engagement_snapshots "
        "ORDER BY tweet_id").fetchall()
    conn.close()
    assert len(rows) == 3
    assert {r[0] for r in rows} == {"111", "222", "333"}
    # All have likes=2 / replies=1 / quotes=0 from the shape-C fixture
    for _tid, l, r, q in rows:
        assert (l, r, q) == (2, 1, 0)


def test_shape_a_response_still_works(teacher_with_db):
    """Backwards-compat: the older shape `{tweets: [...]}` must still
    write snapshots — we widened the parser, not replaced it."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["AAA"])

    mock_gateway = MagicMock()
    mock_gateway.fetch_recent_tweets.return_value = {
        "status": "success",
        "tweets": [{"id": "AAA", "likeCount": 5,
                    "replyCount": 0, "quoteCount": 1,
                    "text": "shape A"}],
    }

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        teacher._fetch_own_engagement(config, titan_id="T1")

    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    rows = conn.execute(
        "SELECT tweet_id, likes, quotes FROM engagement_snapshots").fetchall()
    conn.close()
    assert rows == [("AAA", 5, 1)]


def test_shape_b_response_still_works(teacher_with_db):
    """Backwards-compat: `{data: [...]}` (list-direct) must still
    write snapshots."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["BBB"])

    mock_gateway = MagicMock()
    mock_gateway.fetch_recent_tweets.return_value = {
        "status": "success",
        "data": [{"id": "BBB", "likeCount": 3,
                  "replyCount": 2, "quoteCount": 0,
                  "text": "shape B"}],
    }

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        teacher._fetch_own_engagement(config, titan_id="T1")

    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    rows = conn.execute(
        "SELECT tweet_id, likes, replies FROM engagement_snapshots").fetchall()
    conn.close()
    assert rows == [("BBB", 3, 2)]


def test_circuit_breaker_response_skips_writes(teacher_with_db):
    """A circuit-breaker / error response must NOT produce snapshot
    writes — verifies the outer `if status not in (error, circuit_breaker)`
    gate still works after the unwrap widen."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["CCC"])

    mock_gateway = MagicMock()
    mock_gateway.fetch_recent_tweets.return_value = {
        "status": "circuit_breaker",
        "message": "too many failures",
    }

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        teacher._fetch_own_engagement(config, titan_id="T1")

    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    count = conn.execute(
        "SELECT COUNT(*) FROM engagement_snapshots").fetchone()[0]
    conn.close()
    assert count == 0


def test_empty_tweets_in_shape_c_no_writes(teacher_with_db):
    """Shape C with empty inner tweets list must not write anything
    and must not crash."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["DDD"])

    mock_gateway = MagicMock()
    mock_gateway.fetch_recent_tweets.return_value = {
        "status": "success",
        "data": {"pin_tweet": None, "tweets": []},
    }

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        items, api_calls = teacher._fetch_own_engagement(
            config, titan_id="T1")

    assert api_calls == 1
    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    count = conn.execute(
        "SELECT COUNT(*) FROM engagement_snapshots").fetchone()[0]
    conn.close()
    assert count == 0


def test_delta_computation_with_prior_snapshot(teacher_with_db):
    """Second fetch with bumped counts produces non-zero deltas — the
    `_last_engagement` cache + delta math survives the unwrap fix."""
    teacher, db, tmp_path = teacher_with_db
    social_x = tmp_path / "social_x.db"
    _seed_social_x_with_posts(social_x, ["EEE"])

    mock_gateway = MagicMock()
    # First fetch: 2 likes, 0 replies, 0 quotes
    mock_gateway.fetch_recent_tweets.return_value = _shape_c_response(
        ["EEE"], likes=2, replies=0, quotes=0)

    with patch("titan_hcl.logic.events_teacher.DEFAULT_SOCIAL_DB",
               str(social_x)), \
         patch.object(teacher, "_get_x_gateway", return_value=mock_gateway), \
         patch.object(EventsTeacher, "_get_api_key",
                      return_value="test-key-non-empty"):
        config = {"social_x": {"user_name": "example_handle"}}
        teacher._fetch_own_engagement(config, titan_id="T1")

        # Second fetch: 5 likes, 1 reply, 0 quotes → delta (+3, +1, 0)
        mock_gateway.fetch_recent_tweets.return_value = _shape_c_response(
            ["EEE"], likes=5, replies=1, quotes=0)
        teacher._fetch_own_engagement(config, titan_id="T1")

    conn = sqlite3.connect(str(tmp_path / "events_teacher.db"), timeout=5)
    rows = conn.execute(
        "SELECT likes, replies, quotes, "
        "delta_likes, delta_replies, delta_quotes "
        "FROM engagement_snapshots ORDER BY id").fetchall()
    conn.close()
    assert len(rows) == 2
    # First row: deltas are absolute (prior=0)
    assert rows[0] == (2, 0, 0, 2, 0, 0)
    # Second row: deltas reflect the bump
    assert rows[1] == (5, 1, 0, 3, 1, 0)
