"""rFP X-post PART B — B3: high-relevance non-followed eligibility (2026-06-03).

INV-XENG-3: world_mirror engages a NON-followed author only when relevance
clears HIGH_RELEVANCE (0.8) — via a standalone @mention (no tweet_id /
quote-tweet; VQ2 confirmed felt_experiences carries no tweet_id and the gateway
posts standalone when quoted_tweet_id is empty). Followed authors keep their
full path (bio + quote-tweet). Low-relevance non-followed authors stay excluded.

Run: python -m pytest tests/test_x_world_mirror_b3_nonfollowed_20260603.py -v -p no:anchorpy
"""
from __future__ import annotations

import sqlite3
import time

import pytest


def _build(tmp_path, *, author, relevance, is_following):
    sx = str(tmp_path / "sx.db")
    et = str(tmp_path / "et.db")
    sg = str(tmp_path / "sg.db")
    now = time.time()
    # actions table (cited_set + authors_on_cooldown read it; empty here)
    c = sqlite3.connect(sx)
    c.execute("CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "action_type TEXT, status TEXT, tweet_id TEXT, titan_id TEXT, "
              "post_type TEXT, text TEXT, consumer TEXT, created_at REAL, "
              "posted_at REAL, metadata TEXT)")
    c.commit(); c.close()
    ce = sqlite3.connect(et)
    ce.execute("CREATE TABLE felt_experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "titan_id TEXT, author TEXT, topic TEXT, relevance REAL, "
               "felt_summary TEXT, created_at REAL)")
    ce.execute("INSERT INTO felt_experiences (titan_id, author, topic, relevance, "
               "felt_summary, created_at) VALUES ('T1', ?, 'ai sovereignty', ?, "
               "'a sharp take on self-custody', ?)", (author, relevance, now - 3600))
    ce.commit(); ce.close()
    cg = sqlite3.connect(sg)
    cg.execute("CREATE TABLE community_registry (user_name TEXT PRIMARY KEY, "
               "bio TEXT, is_following INTEGER, last_tweet_text TEXT, last_tweet_id TEXT)")
    if is_following:
        cg.execute("INSERT INTO community_registry (user_name, bio, is_following, "
                   "last_tweet_text, last_tweet_id) VALUES (?, 'crypto OG', 1, "
                   "'their actual tweet', 'TW999')", (author,))
    cg.commit(); cg.close()
    from titan_hcl.logic.social_x.archetypes.world_mirror import WorldMirrorArchetype
    return WorldMirrorArchetype(gateway=None, social_x_db_path=sx,
                                events_teacher_db=et, social_graph_db=sg)


def test_b3_engages_nonfollowed_high_relevance(tmp_path):
    """Non-followed author at relevance 0.9 (≥0.8) → engaged via standalone @mention."""
    wm = _build(tmp_path, author="kirkworkssllc", relevance=0.9, is_following=0)
    cand = wm._fetch_candidate(titan_id="T1", now=time.time())
    assert cand is not None
    assert cand["author"] == "kirkworkssllc"
    assert cand["tweet_id"] == ""                       # standalone, no quote
    assert cand["follow_reason"]                        # honest, not "curated following"
    assert "curated following" not in cand["follow_reason"]
    assert cand["content_excerpt"]                      # from felt_summary


def test_b3_excludes_nonfollowed_low_relevance(tmp_path):
    """Non-followed author below 0.8 (but above the 0.55 floor) → NOT engaged."""
    wm = _build(tmp_path, author="randomguy", relevance=0.6, is_following=0)
    assert wm._fetch_candidate(titan_id="T1", now=time.time()) is None


def test_b3_followed_author_still_quote_tweets(tmp_path):
    """Followed author keeps the full path: bio + quote-tweet (tweet_id present)."""
    wm = _build(tmp_path, author="lopp", relevance=0.7, is_following=1)
    cand = wm._fetch_candidate(titan_id="T1", now=time.time())
    assert cand is not None
    assert cand["author"] == "lopp"
    assert cand["tweet_id"] == "TW999"                  # quote-tweet preserved
    assert cand["content_excerpt"] == "their actual tweet"
