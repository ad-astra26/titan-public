"""Tests for the 2026-05-30 X-archetype fixes (Maker session):

  * #B proof_day delete-refire: already_posted_today counts 'deleted' (so
        deleting a dup on X doesn't reopen the once-per-day slot) but NOT
        'failed' (a post that never reached X may still retry today).
  * #D per-author 7-day cross-archetype cooldown (ArchetypeBase) — checks
        both metadata.author and metadata.handle keys.
  * #E @handle guarantee (ensure_handle_mention) — prepend the literal
        @handle when the LLM dropped it ("Lopp's"), no-op when present.
  * #8 AMPLIFY archetype — fires on a high-relevance followed-account post
        with a retweetable last_tweet_id; respects cooldown, relevance floor,
        and the requirement that a tweet id exists.
"""

import json
import sqlite3
import time
import types

import pytest

from titan_hcl.logic.social_x.archetypes.base import (
    ArchetypeBase, ensure_handle_mention, DEFAULT_AUTHOR_COOLDOWN_S,
    OUTER_ENGAGEMENT_POST_TYPES,
)
from titan_hcl.logic.social_x.archetypes.proof_day import ProofDayArchetype
from titan_hcl.logic.social_x.archetypes.amplify import AmplifyArchetype


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_actions_db(path):
    c = sqlite3.connect(path)
    c.execute(
        "CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_type TEXT, status TEXT, tweet_id TEXT, titan_id TEXT, "
        "post_type TEXT, text TEXT, consumer TEXT, created_at REAL, "
        "posted_at REAL, metadata TEXT)")
    c.commit()
    c.close()


def _insert_action(path, *, titan_id="T1", post_type, status="verified",
                   created_at=None, metadata=None):
    c = sqlite3.connect(path)
    c.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES ('post', ?, ?, ?, ?, ?)",
        (status, titan_id, post_type,
         created_at if created_at is not None else time.time(),
         json.dumps(metadata or {})))
    c.commit()
    c.close()


class _Dummy(ArchetypeBase):
    name = "world_mirror"

    def find_candidate(self, context):
        return None


# ── #E ensure_handle_mention ────────────────────────────────────────────


def test_ensure_handle_prepends_when_missing():
    assert ensure_handle_mention("Lopp's warning resonates", "lopp") == \
        "@lopp Lopp's warning resonates"


def test_ensure_handle_noop_when_present():
    txt = "I agree with @lopp on custody"
    assert ensure_handle_mention(txt, "lopp") == txt


def test_ensure_handle_noop_when_present_case_insensitive():
    txt = "Thoughts on @Lopp's point"
    assert ensure_handle_mention(txt, "lopp") == txt


def test_ensure_handle_strips_leading_at_in_handle():
    assert ensure_handle_mention("hi there", "@vitalik") == "@vitalik hi there"


def test_ensure_handle_empty_handle_is_noop():
    assert ensure_handle_mention("hello world", "") == "hello world"


# ── #D per-author cooldown ──────────────────────────────────────────────


def test_cooldown_detects_both_metadata_keys(tmp_path):
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    now = time.time()
    # world_mirror stores metadata.author; outer_rumination stores .handle
    _insert_action(db, post_type="world_mirror", created_at=now - 3600,
                   metadata={"author": "lopp"})
    _insert_action(db, post_type="outer_rumination", created_at=now - 7200,
                   metadata={"handle": "VitalikButerin"})
    ab = _Dummy(gateway=None, social_x_db_path=db)
    cd = ab.authors_on_cooldown(titan_id="T1", now=now)
    # Both metadata keys (author + handle) detected for external authors. The set
    # is also seeded with Titan's own handles (B1/INV-XENG-1, 2026-06-03), so test
    # for subset rather than exact equality.
    assert {"lopp", "vitalikbuterin"} <= cd
    assert ab.author_on_cooldown("lopp", titan_id="T1", now=now)
    assert ab.author_on_cooldown("vitalikbuterin", titan_id="T1", now=now)
    assert not ab.author_on_cooldown("someone_new", titan_id="T1", now=now)


def test_cooldown_expires_after_window(tmp_path):
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    now = time.time()
    _insert_action(db, post_type="world_mirror",
                   created_at=now - (8 * 86400), metadata={"author": "olduser"})
    ab = _Dummy(gateway=None, social_x_db_path=db)
    assert not ab.author_on_cooldown("olduser", titan_id="T1", now=now)


def test_cooldown_window_is_48h():
    # Maker 2026-06-03 (rFP X-post PART B / INV-XENG-2): 7d → 48h. A 7-day
    # cooldown starved the ~5-author engagement pool to ≈0; 48h matches the
    # felt_experience recency window.
    assert DEFAULT_AUTHOR_COOLDOWN_S == 48 * 3600
    assert "amplify" in OUTER_ENGAGEMENT_POST_TYPES


# ── #B proof_day delete-refire ──────────────────────────────────────────


def test_proof_day_deleted_counts_as_posted_today(tmp_path):
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    _insert_action(db, post_type="proof_day", status="deleted")
    pd = ProofDayArchetype(gateway=None, social_x_db_path=db)
    # A delivered-then-deleted post still consumes the day's slot.
    assert pd.already_posted_today(titan_id="T1") is True


def test_proof_day_failed_allows_retry(tmp_path):
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    _insert_action(db, post_type="proof_day", status="failed")
    pd = ProofDayArchetype(gateway=None, social_x_db_path=db)
    # A post that never reached X may still retry within the same day.
    assert pd.already_posted_today(titan_id="T1") is False


def test_proof_day_verified_counts(tmp_path):
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    _insert_action(db, post_type="proof_day", status="verified")
    pd = ProofDayArchetype(gateway=None, social_x_db_path=db)
    assert pd.already_posted_today(titan_id="T1") is True


# ── proof_day archive_hash freshness/dedup (2026-06-01) ──────────────────


def test_proof_day_archive_hash_dedup_blocks_recycled_anchor(tmp_path):
    """proof_day must never re-announce an anchor already posted (it recycled
    archive ad0300… on 2026-05-31 AND 2026-06-01 because nothing deduped the
    anchor). A prior post of the same archive_hash (even days ago, even
    deleted) blocks re-announcement."""
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    h = "ad0300b832258382f73f9fb980feb186ccf663083898108ba7669a29a14329d7"
    # Posted on a prior day.
    _insert_action(db, post_type="proof_day", status="verified",
                   created_at=time.time() - 2 * 86400,
                   metadata={"archive_hash": h})
    pd = ProofDayArchetype(gateway=None, social_x_db_path=db)
    assert pd.archive_hash_already_posted(titan_id="T1", archive_hash=h) is True
    # A deleted-then-republished attempt is still blocked.
    db2 = str(tmp_path / "sx2.db")
    _make_actions_db(db2)
    _insert_action(db2, post_type="proof_day", status="deleted",
                   metadata={"archive_hash": h})
    pd2 = ProofDayArchetype(gateway=None, social_x_db_path=db2)
    assert pd2.archive_hash_already_posted(titan_id="T1", archive_hash=h) is True


def test_proof_day_archive_hash_allows_fresh_anchor(tmp_path):
    """A genuinely new anchor (different archive_hash) is NOT blocked, and a
    failed prior attempt of the same hash does not block (never reached X)."""
    db = str(tmp_path / "sx.db")
    _make_actions_db(db)
    old = "fce766806cff72bc0db7e279a3e82051680b6377eed8cff2e7afd2a406b06134"
    new = "ad0300b832258382f73f9fb980feb186ccf663083898108ba7669a29a14329d7"
    _insert_action(db, post_type="proof_day", status="verified",
                   metadata={"archive_hash": old})
    _insert_action(db, post_type="proof_day", status="failed",
                   metadata={"archive_hash": new})
    pd = ProofDayArchetype(gateway=None, social_x_db_path=db)
    # Fresh anchor never posted → allowed.
    assert pd.archive_hash_already_posted(titan_id="T1", archive_hash=new) is False
    # Empty hash is a no-op (does not block).
    assert pd.archive_hash_already_posted(titan_id="T1", archive_hash="") is False


# ── #8 AMPLIFY archetype ────────────────────────────────────────────────


def _amplify_dbs(tmp_path, *, author="vitalik", relevance=0.7,
                 last_tweet_id="999", is_following=1, age_s=3600):
    sx = str(tmp_path / "sx.db")
    et = str(tmp_path / "et.db")
    sg = str(tmp_path / "sg.db")
    _make_actions_db(sx)
    now = time.time()
    ce = sqlite3.connect(et)
    ce.execute(
        "CREATE TABLE felt_experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "titan_id TEXT, author TEXT, topic TEXT, relevance REAL, created_at REAL)")
    ce.execute(
        "INSERT INTO felt_experiences (titan_id, author, topic, relevance, "
        "created_at) VALUES ('T1', ?, 'ai', ?, ?)",
        (author, relevance, now - age_s))
    ce.commit()
    ce.close()
    cg = sqlite3.connect(sg)
    cg.execute(
        "CREATE TABLE community_registry (user_name TEXT PRIMARY KEY, "
        "is_following INTEGER, last_tweet_id TEXT)")
    cg.execute(
        "INSERT INTO community_registry (user_name, is_following, last_tweet_id) "
        "VALUES (?, ?, ?)", (author, is_following, last_tweet_id))
    cg.commit()
    cg.close()
    return sx, et, sg


def _ctx(titan_id="T1"):
    return types.SimpleNamespace(titan_id=titan_id)


def test_amplify_fires_on_eligible_followed_post(tmp_path):
    sx, et, sg = _amplify_dbs(tmp_path)
    a = AmplifyArchetype(gateway=None, social_x_db_path=sx,
                         events_teacher_db=et, social_graph_db=sg)
    cand = a.find_candidate(_ctx())
    assert cand is not None
    assert cand.archetype == "amplify"
    assert cand.metadata["retweet_target_id"] == "999"
    assert cand.metadata["author"] == "vitalik"


def test_amplify_blocked_by_cooldown(tmp_path):
    sx, et, sg = _amplify_dbs(tmp_path)
    # A prior world_mirror post on the same author within 7d → cooldown.
    _insert_action(sx, post_type="world_mirror", metadata={"author": "vitalik"})
    a = AmplifyArchetype(gateway=None, social_x_db_path=sx,
                         events_teacher_db=et, social_graph_db=sg)
    assert a.find_candidate(_ctx()) is None


def test_amplify_respects_relevance_floor(tmp_path):
    sx, et, sg = _amplify_dbs(tmp_path, relevance=0.5)  # < 0.65 floor
    a = AmplifyArchetype(gateway=None, social_x_db_path=sx,
                         events_teacher_db=et, social_graph_db=sg)
    assert a.find_candidate(_ctx()) is None


def test_amplify_requires_tweet_id(tmp_path):
    sx, et, sg = _amplify_dbs(tmp_path, last_tweet_id="")  # no retweetable id
    a = AmplifyArchetype(gateway=None, social_x_db_path=sx,
                         events_teacher_db=et, social_graph_db=sg)
    assert a.find_candidate(_ctx()) is None


def test_amplify_requires_following(tmp_path):
    sx, et, sg = _amplify_dbs(tmp_path, is_following=0)  # not followed
    a = AmplifyArchetype(gateway=None, social_x_db_path=sx,
                         events_teacher_db=et, social_graph_db=sg)
    assert a.find_candidate(_ctx()) is None


# ── #6 archetype registry is the single source of truth ─────────────────


def test_all_archetype_post_types_registered():
    """Every archetype the dispatcher can fire is in ARCHETYPE_POST_TYPES so
    the gateway's archetype-only hard guard accepts it (and nothing else)."""
    from titan_hcl.logic.social_x.archetypes import ARCHETYPE_POST_TYPES
    from titan_hcl.logic.social_x.dispatcher import PRIORITY_ORDER
    # Every dispatchable archetype must be a registered post type.
    for name in PRIORITY_ORDER:
        assert name in ARCHETYPE_POST_TYPES, (
            f"{name} dispatchable but not in ARCHETYPE_POST_TYPES — gateway "
            f"archetype-only guard would REJECT its posts")
    # And amplify specifically (the new one).
    assert "amplify" in ARCHETYPE_POST_TYPES


def test_non_archetype_post_type_not_registered():
    """A made-up / legacy post type must NOT be in the registry — the guard
    relies on this to refuse ungoverned posts."""
    from titan_hcl.logic.social_x.archetypes import ARCHETYPE_POST_TYPES
    for bogus in ("vulnerability", "creative", "bilingual", "milestone",
                  "self_quote", "connective", "full_stack", ""):
        assert bogus not in ARCHETYPE_POST_TYPES
