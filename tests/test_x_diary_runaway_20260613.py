"""Regression tests for the 2026-06-13 soul_diary X-post hourly-runaway.

ROOT CAUSE (three compounding layers):
  L1 (trigger): the twitterapi.io session died → create_tweet_v2 returned
      HTTP 200 {"status":"error","message":"could not extract tweet_id from
      response"} — the soft-fail signature where the tweet very often DID land
      on the timeline but the id couldn't be parsed back.
  L2 (the bug): post() marked that soft-fail S_FAILED. The must-post daily
      latch (soul_diary/proof_day already_posted_today) counted only
      ('posted','verified','pending','deleted') — never 'failed' — so the latch
      never closed; with bypass_rate_limit=True nothing throttled it → it
      re-fired and spammed REAL duplicate tweets.
  L3 (cadence): the write circuit-breaker (trips at 6 soft-fails, 3600s
      cooldown) was the ONLY accidental throttle → one duplicate/hour.

THE FIX:
  * NEW status S_UNVERIFIED = "unverified" — soft-failed-but-likely-landed.
    post() now records it instead of S_FAILED.
  * S_UNVERIFIED counts for every dedup / daily-latch / budget / interval
    query exactly like a posted tweet (so a must-post slot does NOT re-fire),
    but is distinct from 'posted'/'verified' for tweet_id-dependent reads.
  * Defense-in-depth: MUST_POST_DAILY_HARD_CAP — even a bypass_rate_limit
    archetype can never exceed N attempts (ANY status) per (titan, post_type)
    per 24h.
"""
import datetime as _dt
import sqlite3
import time

import pytest

from titan_hcl.logic.social_x_gateway import (
    SocialXGateway, PostContext, PostDescriptor,
)
from titan_hcl.logic.social_x.archetypes.soul_diary import SoulDiaryArchetype
from titan_hcl.logic.social_x.archetypes.proof_day import ProofDayArchetype


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_social_x.db")


@pytest.fixture
def tmp_config(tmp_path):
    config_path = str(tmp_path / "config.toml")
    with open(config_path, "w") as f:
        f.write("""
[social_x]
enabled = true
max_posts_per_hour = 2
max_posts_per_day = 5
min_post_interval = 1800
max_post_length = 500
quality_gate = true
url_domain = "https://example.com"

[social_x.consumers]
spirit_worker = "post,reply,like,search"
test_consumer = "post,reply,like,search"
""")
    return config_path


@pytest.fixture
def tmp_telemetry(tmp_path):
    return str(tmp_path / "telemetry.jsonl")


@pytest.fixture
def gateway(tmp_db, tmp_config, tmp_telemetry):
    gw = SocialXGateway(db_path=tmp_db, config_path=tmp_config,
                        telemetry_path=tmp_telemetry)
    gw._boot_time = 0  # bypass boot grace
    return gw


def _insert(db_path, *, post_type, status, titan_id="T1",
            created_at=None, metadata=None, action_type="post"):
    """Insert one actions row directly (test helper)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO actions (action_type, status, titan_id, post_type, "
            "created_at, metadata) VALUES (?,?,?,?,?,?)",
            (action_type, status, titan_id, post_type,
             created_at if created_at is not None else time.time(),
             metadata or "{}"))
        conn.commit()
    finally:
        conn.close()


def _utc_midnight_ts(now=None):
    n = now if now is not None else time.time()
    d = _dt.datetime.fromtimestamp(n, _dt.timezone.utc).date()
    return _dt.datetime(d.year, d.month, d.day,
                        tzinfo=_dt.timezone.utc).timestamp()


# ── L2: post() records S_UNVERIFIED on soft-fail (not S_FAILED) ──────────

class TestPostSoftFailRecordsUnverified:
    def test_soft_fail_unverified_when_verify_inconclusive(
            self, gateway, tmp_db, monkeypatch):
        """create_tweet soft-fails ('could not extract tweet_id') AND the
        recency-guard can't confirm → row is S_UNVERIFIED (was S_FAILED),
        result.status == 'unverified', posted_at is stamped, no tweet_id."""
        def fake_call(endpoint, method="GET", payload=None, **kw):
            if endpoint == "twitter/create_tweet_v2":
                return {"status": "error",
                        "message": "could not extract tweet_id from response"}
            return {"status": "success", "tweets": []}

        monkeypatch.setattr(gateway, "_call_x_api", fake_call)
        monkeypatch.setattr(gateway, "_verify_post_on_x",
                            lambda *a, **k: (False, ""))
        monkeypatch.setattr(gateway, "_quality_gate", lambda *a, **k: (True, ""))
        monkeypatch.setattr(gateway, "_assemble_final_text",
                            lambda text, *a, **k: text)

        desc = PostDescriptor(
            post_type="soul_diary", catalyst={"type": "test"},
            system_prompt="s", user_prompt="u",
            max_tokens=200, temperature=0.8, voice_cfg={})
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          composed_text="Today I traced the fragile edges.")
        result = gateway.post(ctx, consumer="spirit_worker", descriptor=desc)

        assert result.status == "unverified"
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT status, tweet_id, posted_at FROM actions "
            "WHERE action_type='post' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] == "unverified", "soft-fail must be S_UNVERIFIED, not failed"
        assert not row[1], "no tweet_id on an unverified post"
        assert row[2] is not None, "posted_at stamped (it occupies a slot)"

    def test_real_transport_failure_still_fails(self, gateway, tmp_db,
                                                monkeypatch):
        """A genuine transport failure (407, never reached X) must STILL be
        S_FAILED — only the soft-fail-could-not-extract case is unverified."""
        def fake_call(endpoint, method="GET", payload=None, **kw):
            if endpoint == "twitter/create_tweet_v2":
                return {"status": "error",
                        "message": "API returned status 407"}
            return {"status": "success", "tweets": []}

        monkeypatch.setattr(gateway, "_call_x_api", fake_call)
        monkeypatch.setattr(gateway, "_verify_post_on_x",
                            lambda *a, **k: (False, ""))
        monkeypatch.setattr(gateway, "_quality_gate", lambda *a, **k: (True, ""))
        monkeypatch.setattr(gateway, "_assemble_final_text",
                            lambda text, *a, **k: text)
        desc = PostDescriptor(
            post_type="reflection", catalyst={"type": "test"},
            system_prompt="s", user_prompt="u",
            max_tokens=200, temperature=0.8, voice_cfg={})
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          composed_text="A grounded reflection.")
        result = gateway.post(ctx, consumer="spirit_worker", descriptor=desc)
        assert result.status == "api_failed"
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT status FROM actions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row[0] == "failed"


# ── L2: the daily latch closes on S_UNVERIFIED ───────────────────────────

class TestDailyLatchClosesOnUnverified:
    def test_soul_diary_latch_closes(self, gateway, tmp_db):
        arch = SoulDiaryArchetype(gateway=gateway, social_x_db_path=tmp_db)
        assert arch.already_posted_today(titan_id="T1") is False
        _insert(tmp_db, post_type="soul_diary", status="unverified",
                created_at=_utc_midnight_ts() + 60)
        assert arch.already_posted_today(titan_id="T1") is True, (
            "an unverified diary post today MUST close the latch — "
            "this is the runaway fix")

    def test_soul_diary_failed_does_not_close_latch(self, gateway, tmp_db):
        """A genuine 'failed' (never-reached-X) must still allow a retry."""
        arch = SoulDiaryArchetype(gateway=gateway, social_x_db_path=tmp_db)
        _insert(tmp_db, post_type="soul_diary", status="failed",
                created_at=_utc_midnight_ts() + 60)
        assert arch.already_posted_today(titan_id="T1") is False

    def test_proof_day_latch_and_hash_dedup_count_unverified(
            self, gateway, tmp_db):
        arch = ProofDayArchetype(gateway=gateway, social_x_db_path=tmp_db)
        h = "deadbeefcafe0001"
        _insert(tmp_db, post_type="proof_day", status="unverified",
                created_at=_utc_midnight_ts() + 60,
                metadata=f'{{"archive_hash":"{h}"}}')
        assert arch.already_posted_today(titan_id="T1") is True
        assert arch.archive_hash_already_posted(
            titan_id="T1", archive_hash=h) is True


# ── L2: budget + interval count S_UNVERIFIED ─────────────────────────────

class TestBudgetCountsUnverified:
    def test_hourly_budget_counts_unverified(self, gateway, tmp_db):
        """Two unverified posts hit max_posts_per_hour=2 → next post blocked.
        Before the fix these were 'failed' and evaded the budget entirely."""
        now = time.time()
        for _ in range(2):
            _insert(tmp_db, post_type="reflection", status="unverified",
                    created_at=now - 10)
        res = gateway._check_rate_limits("post", gateway._load_config(),
                                         titan_id="T1", post_type="reflection")
        assert res is not None and res.status == "hourly_limit"

    def test_min_interval_counts_unverified(self, gateway, tmp_db):
        """A recent unverified post blocks the next within min_post_interval."""
        cfg = gateway._load_config()
        cfg["max_posts_per_hour"] = 999
        cfg["max_posts_per_day"] = 999
        _insert(tmp_db, post_type="reflection", status="unverified",
                created_at=time.time() - 60)  # 60s < 1800s interval
        res = gateway._check_rate_limits("post", cfg, titan_id="T1",
                                         post_type="reflection")
        assert res is not None and res.status == "too_soon"


# ── L3 backstop: must-post hard cap counting ALL statuses ────────────────

class TestMustPostHardCap:
    def test_bypass_blocked_after_hard_cap(self, gateway, tmp_db):
        """Even bypass_caps=True is bounded: MUST_POST_DAILY_HARD_CAP attempts
        of ANY status (failed/unverified mix) → must_post_hard_cap. This is the
        runaway-killer if a future bug ever re-opens a daily latch."""
        cap = gateway.MUST_POST_DAILY_HARD_CAP
        now = time.time()
        # Mix of failed + unverified — the exact runaway shape.
        for i in range(cap):
            _insert(tmp_db, post_type="soul_diary",
                    status="failed" if i % 2 else "unverified",
                    created_at=now - 100 * i)
        res = gateway._check_rate_limits(
            "post", gateway._load_config(), titan_id="T1",
            bypass_caps=True, post_type="soul_diary")
        assert res is not None and res.status == "must_post_hard_cap"

    def test_bypass_allowed_under_hard_cap(self, gateway, tmp_db):
        """Below the cap, a must-post archetype still bypasses the budget
        (returns None → publishes), preserving the 2026-06-11 must-post design."""
        _insert(tmp_db, post_type="soul_diary", status="failed",
                created_at=time.time() - 100)
        # Also stuff the normal budget full to prove bypass still works.
        for _ in range(5):
            _insert(tmp_db, post_type="reflection", status="posted",
                    created_at=time.time() - 10)
        res = gateway._check_rate_limits(
            "post", gateway._load_config(), titan_id="T1",
            bypass_caps=True, post_type="soul_diary")
        assert res is None, "under the hard cap, must-post still bypasses budget"

    def test_hard_cap_is_per_post_type(self, gateway, tmp_db):
        """The cap is per (titan, post_type): proof_day attempts don't count
        against soul_diary's ceiling."""
        cap = gateway.MUST_POST_DAILY_HARD_CAP
        for _ in range(cap + 2):
            _insert(tmp_db, post_type="proof_day", status="unverified",
                    created_at=time.time() - 10)
        res = gateway._check_rate_limits(
            "post", gateway._load_config(), titan_id="T1",
            bypass_caps=True, post_type="soul_diary")
        assert res is None, "proof_day attempts must not bound soul_diary"
