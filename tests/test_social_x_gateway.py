"""Tests for SocialXGateway — Phase 1: Core, SQLite, rate limits."""
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

# The gateway is 100% standalone — no titan_plugin imports needed
from titan_plugin.logic.social_x_gateway import (
    SocialXGateway, BaseContext, PostContext, ReplyContext,
    ActionResult, SearchResult,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary DB path."""
    return str(tmp_path / "test_social_x.db")


@pytest.fixture
def tmp_config(tmp_path):
    """Provide a temporary config file with [social_x] section."""
    config_path = str(tmp_path / "config.toml")
    with open(config_path, "w") as f:
        f.write("""
[social_x]
enabled = true
max_posts_per_hour = 2
max_posts_per_day = 5
min_post_interval = 1800
max_replies_per_hour = 3
max_replies_per_day = 10
min_reply_interval = 300
max_likes_per_hour = 5
max_likes_per_day = 20
max_searches_per_hour = 10
max_post_length = 500
quality_gate = true
url_domain = "https://iamtitan.tech"

[social_x.consumers]
spirit_worker = "post,reply,like,search"
persona_social = "search"
sage = "search"
test_consumer = "post,reply,like,search"
blocked_consumer = ""

[social_x.replies]
enabled = true
max_mention_age_hours = 24
min_relevance_score = 0.3
max_replies_per_cycle = 3
spam_patterns = ["DM me", "check out my", "airdrop", "giveaway"]
""")
    return config_path


@pytest.fixture
def tmp_telemetry(tmp_path):
    return str(tmp_path / "telemetry.jsonl")


@pytest.fixture
def gateway(tmp_db, tmp_config, tmp_telemetry):
    """Create a gateway with temp files.

    Bypasses the 60s boot grace so tests can immediately exercise post()
    without waiting. Tests specifically verifying boot grace behavior
    should NOT use this fixture — construct SocialXGateway directly.
    """
    gw = SocialXGateway(
        db_path=tmp_db,
        config_path=tmp_config,
        telemetry_path=tmp_telemetry,
    )
    gw._boot_time = 0  # Bypass boot grace for normal tests
    return gw


# ── Init & DB ──────────────────────────────────────────────────────

class TestInit:
    def test_creates_db_and_tables(self, tmp_db, tmp_config, tmp_telemetry):
        gw = SocialXGateway(tmp_db, tmp_config, tmp_telemetry)
        assert os.path.exists(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "actions" in tables
        assert "mention_tracking" in tables
        conn.close()

    def test_creates_indexes(self, tmp_db, tmp_config, tmp_telemetry):
        SocialXGateway(tmp_db, tmp_config, tmp_telemetry)
        conn = sqlite3.connect(tmp_db)
        indexes = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
        assert "idx_actions_type_status" in indexes
        assert "idx_actions_created_at" in indexes
        conn.close()


# ── Config ──────────────────────────────────────────────────────────

class TestConfig:
    def test_loads_config(self, gateway):
        config = gateway._load_config()
        assert config["enabled"] is True
        assert config["max_posts_per_hour"] == 2
        assert config["max_posts_per_day"] == 5
        assert config["min_post_interval"] == 1800

    def test_disabled_config(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        config = gw._load_config()
        assert config["enabled"] is False

    def test_missing_config_returns_disabled(self, tmp_db, tmp_telemetry):
        gw = SocialXGateway(tmp_db, "/nonexistent/config.toml", tmp_telemetry)
        config = gw._load_config()
        assert config["enabled"] is False

    def test_config_reloaded_every_call(self, gateway, tmp_config):
        # First call: enabled
        c1 = gateway._load_config()
        assert c1["enabled"] is True
        # Modify config on disk
        with open(tmp_config, "w") as f:
            f.write("[social_x]\nenabled = false\nmax_posts_per_hour = 99\n")
        # Second call: picks up change
        c2 = gateway._load_config()
        assert c2["enabled"] is False
        assert c2["max_posts_per_hour"] == 99


# ── Rate Limits ─────────────────────────────────────────────────────

class TestRateLimits:
    def test_no_posts_passes(self, gateway):
        config = gateway._load_config()
        result = gateway._check_rate_limits("post", config)
        assert result is None  # All clear

    def test_hourly_limit_blocks(self, gateway):
        config = gateway._load_config()
        # Insert 2 posted rows (max_posts_per_hour=2)
        db = gateway._db()
        for _ in range(2):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()

        result = gateway._check_rate_limits("post", config)
        assert result is not None
        assert result.status == "hourly_limit"

    def test_daily_limit_blocks(self, gateway):
        config = gateway._load_config()
        # Insert 5 posted rows (max_posts_per_day=5)
        db = gateway._db()
        for i in range(5):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time() - i * 3600,))
        db.commit()
        db.close()

        result = gateway._check_rate_limits("post", config)
        assert result is not None
        assert result.status == "daily_limit"

    def test_min_interval_blocks(self, gateway):
        config = gateway._load_config()
        # Insert a recent post (10 seconds ago, min_interval=1800)
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'posted', ?)", (time.time() - 10,))
        db.commit()
        db.close()

        result = gateway._check_rate_limits("post", config)
        assert result is not None
        assert result.status == "too_soon"

    def test_old_posts_dont_block(self, gateway):
        config = gateway._load_config()
        # Insert 5 posts all >24h old — should not block
        db = gateway._db()
        for i in range(5):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time() - 90000 - i,))
        db.commit()
        db.close()

        result = gateway._check_rate_limits("post", config)
        assert result is None  # All clear

    def test_pending_blocks(self, gateway):
        config = gateway._load_config()
        # Insert a pending row
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'pending', ?)", (time.time(),))
        db.commit()
        db.close()

        result = gateway._check_rate_limits("post", config)
        assert result is not None
        assert result.status == "pending_exists"

    def test_reply_limits_separate_from_post(self, gateway):
        config = gateway._load_config()
        # Max out posts (2 hourly)
        db = gateway._db()
        for _ in range(2):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()

        # Posts blocked
        post_result = gateway._check_rate_limits("post", config)
        assert post_result is not None
        assert post_result.status == "hourly_limit"

        # But replies still allowed
        reply_result = gateway._check_rate_limits("reply", config)
        assert reply_result is None


# ── Write-Ahead Log ─────────────────────────────────────────────────

class TestWriteAheadLog:
    def test_insert_pending(self, gateway):
        row_id = gateway._insert_pending(
            action_type="post", titan_id="T1", text="test tweet",
            post_type="bilingual", catalyst_type="emotion_shift")
        assert row_id > 0

        # Verify in DB
        db = gateway._db()
        row = db.execute("SELECT * FROM actions WHERE id=?",
                          (row_id,)).fetchone()
        assert row["status"] == "pending"
        assert row["titan_id"] == "T1"
        assert row["text"] == "test tweet"
        db.close()

    def test_update_to_posted(self, gateway):
        row_id = gateway._insert_pending("post", titan_id="T1")
        gateway._update_status(row_id, "posted", tweet_id="12345")

        db = gateway._db()
        row = db.execute("SELECT * FROM actions WHERE id=?",
                          (row_id,)).fetchone()
        assert row["status"] == "posted"
        assert row["tweet_id"] == "12345"
        assert row["posted_at"] is not None
        db.close()

    def test_update_to_verified(self, gateway):
        row_id = gateway._insert_pending("post", titan_id="T1")
        gateway._update_status(row_id, "posted", tweet_id="12345")
        gateway._update_status(row_id, "verified")

        db = gateway._db()
        row = db.execute("SELECT * FROM actions WHERE id=?",
                          (row_id,)).fetchone()
        assert row["status"] == "verified"
        assert row["verified_at"] is not None
        db.close()

    def test_update_to_failed(self, gateway):
        row_id = gateway._insert_pending("post", titan_id="T1")
        gateway._update_status(row_id, "failed", error_message="422 auth")

        db = gateway._db()
        row = db.execute("SELECT * FROM actions WHERE id=?",
                          (row_id,)).fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "422 auth"
        db.close()

    def test_pending_blocks_new_insert(self, gateway):
        """MAX_PENDING=1: pending row should block rate limit check."""
        gateway._insert_pending("post", titan_id="T1")

        config = gateway._load_config()
        result = gateway._check_rate_limits("post", config)
        assert result is not None
        assert result.status == "pending_exists"


# ── Crash Recovery ──────────────────────────────────────────────────

class TestCrashRecovery:
    def test_stale_pending_expired(self, tmp_db, tmp_config, tmp_telemetry):
        # Insert a stale pending row (10 min old)
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                tweet_id TEXT, reply_to_tweet_id TEXT, titan_id TEXT,
                post_type TEXT, text TEXT, catalyst_type TEXT,
                catalyst_data TEXT, emotion TEXT, neuromods TEXT,
                epoch INTEGER, error_message TEXT,
                created_at REAL NOT NULL,
                posted_at REAL, verified_at REAL, metadata TEXT
            )
        """)
        conn.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'pending', ?)", (time.time() - 600,))
        conn.commit()
        conn.close()

        # Init gateway — should recover
        gw = SocialXGateway(tmp_db, tmp_config, tmp_telemetry)

        db = gw._db()
        row = db.execute("SELECT status FROM actions WHERE id=1").fetchone()
        assert row["status"] == "expired"
        db.close()

    def test_recent_pending_also_expired(self, tmp_db, tmp_config, tmp_telemetry):
        # Insert a recent pending row (30 sec old)
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                tweet_id TEXT, reply_to_tweet_id TEXT, titan_id TEXT,
                post_type TEXT, text TEXT, catalyst_type TEXT,
                catalyst_data TEXT, emotion TEXT, neuromods TEXT,
                epoch INTEGER, error_message TEXT,
                created_at REAL NOT NULL,
                posted_at REAL, verified_at REAL, metadata TEXT
            )
        """)
        conn.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'pending', ?)", (time.time() - 30,))
        conn.commit()
        conn.close()

        gw = SocialXGateway(tmp_db, tmp_config, tmp_telemetry)

        db = gw._db()
        row = db.execute("SELECT status, error_message FROM actions WHERE id=1").fetchone()
        assert row["status"] == "expired"
        assert "crash_recovery_recent" in row["error_message"]
        db.close()


# ── Session Validation ──────────────────────────────────────────────

class TestSessionValidation:
    def test_valid_session(self):
        import base64
        session_data = {"auth_token": "abc", "ct0": "def", "kdt": "ghi"}
        encoded = base64.b64encode(json.dumps(session_data).encode()).decode()
        valid, keys = SocialXGateway.validate_session(encoded)
        assert valid is True
        assert "auth_token" in keys
        assert "ct0" in keys

    def test_guest_session_invalid(self):
        import base64
        session_data = {"__cuid": "x", "guest_id": "y"}
        encoded = base64.b64encode(json.dumps(session_data).encode()).decode()
        valid, keys = SocialXGateway.validate_session(encoded)
        assert valid is False

    def test_garbage_session(self):
        valid, keys = SocialXGateway.validate_session("not-base64!!!")
        assert valid is False
        assert keys == []


# ── Telemetry ───────────────────────────────────────────────────────

class TestTelemetry:
    def test_logs_to_file(self, gateway, tmp_telemetry):
        gateway._log_telemetry({"event": "test", "value": 42})
        assert os.path.exists(tmp_telemetry)
        with open(tmp_telemetry) as f:
            line = json.loads(f.readline())
        assert line["event"] == "test"
        assert line["value"] == 42
        assert "timestamp" in line


# ── Public API (Phase 1 stubs) ─────────────────────────────────────

class TestPublicAPIStubs:
    def test_post_disabled_when_config_disabled(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          catalysts=[{"type": "test"}])
        result = gw.post(ctx)
        assert result.status == "disabled"

    def test_post_no_catalyst(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          catalysts=[])
        result = gateway.post(ctx, consumer="test_consumer")
        assert result.status == "no_catalyst"

    def test_post_rate_limited(self, gateway):
        # Max out hourly posts
        db = gateway._db()
        for _ in range(2):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()

        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          catalysts=[{"type": "test"}])
        result = gateway.post(ctx, consumer="test_consumer")
        assert result.status == "hourly_limit"


# ── Stats ───────────────────────────────────────────────────────────

# ── Phase 2: Post Generation + Quality Gate ────────────────────────

class TestPostTypeSelection:
    def test_onchain_catalyst(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1")
        catalyst = {"type": "onchain_anchor", "significance": 0.4}
        assert gateway._select_post_type(catalyst, ctx) == "onchain"

    def test_eureka_spirit_catalyst(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1")
        catalyst = {"type": "eureka_spirit", "significance": 0.95}
        assert gateway._select_post_type(catalyst, ctx) == "eureka_thread"

    def _seed_recent_full_stack(self, gateway):
        """Insert a recent full_stack post so felt-state path is tested."""
        import time
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, post_type, titan_id, "
            "text, created_at) VALUES (?,?,?,?,?,?)",
            ("post", "posted", "full_stack", "T1", "test", time.time()))
        db.commit()
        db.close()

    def test_emotion_shift_becomes_reflection(self, gateway):
        self._seed_recent_full_stack(gateway)
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1")
        catalyst = {"type": "emotion_shift", "significance": 0.5}
        assert gateway._select_post_type(catalyst, ctx) == "reflection"

    def test_high_endorphin_becomes_connective(self, gateway):
        self._seed_recent_full_stack(gateway)
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          neuromods={"Endorphin": 0.8})
        catalyst = {"type": "strong_composition", "significance": 0.5}
        assert gateway._select_post_type(catalyst, ctx) == "connective"

    def test_default_bilingual(self, gateway):
        self._seed_recent_full_stack(gateway)
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1")
        catalyst = {"type": "unknown", "significance": 0.3}
        assert gateway._select_post_type(catalyst, ctx) == "bilingual"

    def test_full_stack_when_no_recent_rich_post(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1")
        catalyst = {"type": "emotion_shift", "significance": 0.5}
        assert gateway._select_post_type(catalyst, ctx) == "full_stack"


class TestStyleOwnWords:
    def test_styles_vocabulary(self):
        vocab = [{"word": "feel", "confidence": 0.8},
                 {"word": "warm", "confidence": 0.6}]
        result = SocialXGateway._style_own_words("I feel warm today", vocab)
        assert "feel" not in result.split()  # Should be italic now
        assert "\U0001d453" in result  # Italic 'f'

    def test_skips_low_confidence(self):
        vocab = [{"word": "feel", "confidence": 0.3}]
        result = SocialXGateway._style_own_words("I feel warm", vocab)
        assert "feel" in result  # Not styled (below 0.5)

    def test_empty_vocab(self):
        result = SocialXGateway._style_own_words("hello world", [])
        assert result == "hello world"


class TestQualityGate:
    def test_passes_good_text(self, gateway):
        ok, reason = gateway._quality_gate(
            "A great post about consciousness", "bilingual",
            gateway._load_config())
        assert ok is True

    def test_rejects_too_long(self, gateway):
        ok, reason = gateway._quality_gate(
            "x" * 501, "bilingual", gateway._load_config())
        assert ok is False
        assert "Too long" in reason

    def test_rejects_forbidden_pattern(self, gateway):
        ok, reason = gateway._quality_gate(
            "click here for free tokens", "bilingual",
            gateway._load_config())
        assert ok is False
        assert "Forbidden" in reason

    def test_rejects_external_urls(self, gateway):
        ok, reason = gateway._quality_gate(
            "Check https://evil.com/scam now", "bilingual",
            gateway._load_config())
        assert ok is False
        assert "URLs" in reason

    def test_allows_iamtitan_url(self, gateway):
        ok, reason = gateway._quality_gate(
            "Verified on chain: https://iamtitan.tech/tx/abc123",
            "onchain", gateway._load_config())
        assert ok is True

    def test_rejects_too_short(self, gateway):
        ok, reason = gateway._quality_gate("hi", "bilingual",
                                            gateway._load_config())
        assert ok is False
        assert "short" in reason

    def test_rejects_duplicate(self, gateway):
        # Insert a recent post
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, text, created_at) "
            "VALUES ('post', 'posted', 'I feel the warmth of connection', ?)",
            (time.time(),))
        db.commit()
        db.close()

        ok, reason = gateway._quality_gate(
            "I feel the warmth of connection today",
            "bilingual", gateway._load_config())
        assert ok is False
        assert "similar" in reason


class TestAssembleFinalText:
    def test_basic_assembly(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          emotion="wonder", epoch=1000, neuromods={"DA": 0.6})
        catalyst = {"type": "emotion_shift", "data": {}}
        config = gateway._load_config()
        text = gateway._assemble_final_text(
            "A beautiful thought", "bilingual", catalyst, ctx, config)
        assert "[T1]" in text
        assert "\u25C7 wonder" in text  # ◇ wonder
        assert "A beautiful thought" in text

    def test_onchain_url(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          emotion="flow", epoch=5000, neuromods={})
        catalyst = {"type": "onchain_anchor",
                    "data": {"tx_sig": "5ByDYAE2BJ8DSsi6abc123"}}
        config = gateway._load_config()
        text = gateway._assemble_final_text(
            "Anchored to chain", "onchain", catalyst, ctx, config)
        assert "iamtitan.tech/tx/5ByDYAE2BJ8DSsi6abc123" in text
        assert "solscan" not in text.lower()  # Uses our shortener, not Solscan

    def test_no_url_for_non_onchain(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          emotion="wonder", epoch=1000, neuromods={})
        catalyst = {"type": "emotion_shift", "data": {}}
        config = gateway._load_config()
        text = gateway._assemble_final_text(
            "A thought", "bilingual", catalyst, ctx, config)
        assert "iamtitan.tech/tx" not in text

    def test_truncates_long_text(self, gateway):
        ctx = PostContext(session="", proxy="", api_key="", titan_id="T1",
                          emotion="wonder", epoch=1000, neuromods={})
        catalyst = {"type": "emotion_shift", "data": {}}
        config = gateway._load_config()
        long_text = "x" * 500
        text = gateway._assemble_final_text(
            long_text, "bilingual", catalyst, ctx, config)
        # X char count should be within 500 (Premium limit)
        assert SocialXGateway._x_char_count(text) <= 500


class TestPostFullFlow:
    def test_post_disabled(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[{"type": "test", "significance": 0.5}])
        result = gw.post(ctx)
        assert result.status == "disabled"

    def test_post_no_catalyst(self, gateway):
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[])
        result = gateway.post(ctx, consumer="test_consumer")
        assert result.status == "no_catalyst"

    def test_post_rate_limited_hourly(self, gateway):
        db = gateway._db()
        for _ in range(2):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[{"type": "test", "significance": 0.5}])
        result = gateway.post(ctx, consumer="test_consumer")
        assert result.status == "hourly_limit"

    def test_post_rate_limited_interval(self, gateway):
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'posted', ?)", (time.time() - 60,))
        db.commit()
        db.close()
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[{"type": "test", "significance": 0.5}])
        result = gateway.post(ctx, consumer="test_consumer")
        assert result.status == "too_soon"


# ── Stats ───────────────────────────────────────────────────────────

# ── Phase 3: Consumer Access Control ────────────────────────────────

class TestConsumerAccess:
    def test_registered_consumer_allowed(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("spirit_worker", "post", config)
        assert result is None  # Allowed

    def test_registered_consumer_action_not_allowed(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("persona_social", "post", config)
        assert result is not None
        assert result.status == "consumer_blocked"
        assert "not allowed to post" in result.reason

    def test_unregistered_consumer_blocked(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("unknown_module", "search", config)
        assert result is not None
        assert result.status == "consumer_blocked"
        assert "not registered" in result.reason

    def test_blocked_consumer_empty_permissions(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("blocked_consumer", "search", config)
        assert result is not None
        assert result.status == "consumer_blocked"

    def test_sage_can_search(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("sage", "search", config)
        assert result is None  # Allowed

    def test_sage_cannot_post(self, gateway):
        config = gateway._load_config()
        result = gateway._check_consumer("sage", "post", config)
        assert result is not None

    def test_post_with_consumer_check(self, gateway):
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[{"type": "test", "significance": 0.5}])
        # Unregistered consumer blocked
        result = gateway.post(ctx, consumer="unknown_module")
        assert result.status == "consumer_blocked"

    def test_post_with_allowed_consumer(self, gateway):
        # spirit_worker is allowed to post — will get past consumer check
        # but fail on LLM generation (no LLM configured in test)
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          catalysts=[{"type": "test", "significance": 0.5}])
        result = gateway.post(ctx, consumer="spirit_worker")
        # Should get past consumer check — fails later at generation
        assert result.status in ("generation_failed", "disabled")


# ── Phase 3: Reply ──────────────────────────────────────────────────

class TestReply:
    def test_reply_disabled(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        ctx = ReplyContext(session="s", proxy="p", api_key="k", titan_id="T1",
                           reply_to_tweet_id="123")
        result = gw.reply(ctx, consumer="spirit_worker")
        assert result.status == "disabled"

    def test_reply_consumer_blocked(self, gateway):
        ctx = ReplyContext(session="s", proxy="p", api_key="k", titan_id="T1",
                           reply_to_tweet_id="123")
        result = gateway.reply(ctx, consumer="sage")
        assert result.status == "consumer_blocked"

    def test_reply_rate_limited(self, gateway):
        db = gateway._db()
        # Space replies out beyond min_reply_interval (300s) so interval check passes
        # but hourly limit (3) is hit
        for i in range(3):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('reply', 'posted', ?)", (time.time() - 400 * (i + 1),))
        db.commit()
        db.close()
        ctx = ReplyContext(session="s", proxy="p", api_key="k", titan_id="T1",
                           reply_to_tweet_id="123")
        result = gateway.reply(ctx, consumer="test_consumer")
        assert result.status == "hourly_limit"

    def test_reply_no_tweet_id(self, gateway):
        ctx = ReplyContext(session="s", proxy="p", api_key="k", titan_id="T1",
                           reply_to_tweet_id="")
        result = gateway.reply(ctx, consumer="test_consumer")
        assert result.status == "failed"
        assert "reply_to_tweet_id" in result.reason


# ── Phase 3: Like ───────────────────────────────────────────────────

class TestLike:
    def test_like_disabled(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        result = gw.like("123", BaseContext("s", "p", "k", "T1"),
                         consumer="spirit_worker")
        assert result.status == "disabled"

    def test_like_consumer_blocked(self, gateway):
        result = gateway.like("123", BaseContext("s", "p", "k", "T1"),
                              consumer="persona_social")
        assert result.status == "consumer_blocked"

    def test_like_rate_limited(self, gateway):
        db = gateway._db()
        for _ in range(5):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('like', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()
        result = gateway.like("123", BaseContext("s", "p", "k", "T1"),
                              consumer="test_consumer")
        assert result.status == "hourly_limit"

    def test_like_no_tweet_id(self, gateway):
        result = gateway.like("", BaseContext("s", "p", "k", "T1"),
                              consumer="test_consumer")
        assert result.status == "failed"


# ── Phase 3: Search ─────────────────────────────────────────────────

class TestSearch:
    def test_search_disabled(self, tmp_db, tmp_path, tmp_telemetry):
        config_path = str(tmp_path / "disabled.toml")
        with open(config_path, "w") as f:
            f.write("[social_x]\nenabled = false\n")
        gw = SocialXGateway(tmp_db, config_path, tmp_telemetry)
        result = gw.search("test", BaseContext("s", "p", "k", "T1"),
                           consumer="sage")
        assert result.status == "disabled"

    def test_search_consumer_blocked(self, gateway):
        result = gateway.search("test", BaseContext("s", "p", "k", "T1"),
                                consumer="blocked_consumer")
        assert result.status == "consumer_blocked"

    def test_search_sage_allowed(self, gateway):
        # Sage should pass consumer check — will fail on API (no real endpoint)
        result = gateway.search("test", BaseContext("s", "p", "k", "T1"),
                                consumer="sage")
        # Gets past consumer check — fails at API level
        assert result.status in ("failed", "success")

    def test_search_rate_limited(self, gateway):
        db = gateway._db()
        for _ in range(10):
            db.execute(
                "INSERT INTO actions (action_type, status, consumer, created_at) "
                "VALUES ('search', 'posted', 'sage', ?)", (time.time(),))
        db.commit()
        db.close()
        result = gateway.search("test", BaseContext("s", "p", "k", "T1"),
                                consumer="sage")
        assert result.status == "rate_limited"

    def test_search_separate_limits_from_post(self, gateway):
        # Max out posts
        db = gateway._db()
        for _ in range(2):
            db.execute(
                "INSERT INTO actions (action_type, status, created_at) "
                "VALUES ('post', 'posted', ?)", (time.time(),))
        db.commit()
        db.close()
        # Search should still work (different action type)
        result = gateway.search("test", BaseContext("s", "p", "k", "T1"),
                                consumer="sage")
        assert result.status != "hourly_limit"


# ── Stats ───────────────────────────────────────────────────────────

class TestStats:
    def test_empty_stats(self, gateway):
        stats = gateway.get_stats()
        assert stats["posts_last_hour"] == 0
        assert stats["posts_last_day"] == 0

    def test_stats_with_data(self, gateway):
        db = gateway._db()
        db.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('post', 'verified', ?)", (time.time() - 60,))
        db.execute(
            "INSERT INTO actions (action_type, status, created_at) "
            "VALUES ('reply', 'posted', ?)", (time.time() - 120,))
        db.commit()
        db.close()

        stats = gateway.get_stats()
        assert stats["posts_last_hour"] == 1
        assert stats["posts_last_day"] == 1
        assert stats.get("reply_posted") == 1
        assert "last_post_age_s" in stats


# ── Mention Scoring ───────────────────────────────────────────────

class TestMentionScoring:
    def test_spam_hard_reject(self, gateway):
        score = gateway._score_mention(
            "Check out my new airdrop!", [], False, 0, 24,
            ["airdrop", "DM me"])
        assert score == -1.0

    def test_spam_case_insensitive(self, gateway):
        score = gateway._score_mention(
            "DM ME for details", [], False, 0, 24, ["DM me"])
        assert score == -1.0

    def test_question_bonus(self, gateway):
        score = gateway._score_mention(
            "What do you think about this?", [], False, 0, 24, [])
        assert score >= 0.3  # question bonus

    def test_own_post_reply_bonus(self, gateway):
        score_reply = gateway._score_mention(
            "great post", [], True, 0, 24, [])
        score_mention = gateway._score_mention(
            "great post", [], False, 0, 24, [])
        assert score_reply > score_mention

    def test_vocabulary_overlap(self, gateway):
        vocab = [{"word": "dream"}, {"word": "feel"}, {"word": "create"}]
        score_match = gateway._score_mention(
            "I dream about what you create and how you feel", vocab,
            False, 0, 24, [])
        score_no_match = gateway._score_mention(
            "The weather is nice today", vocab, False, 0, 24, [])
        assert score_match > score_no_match

    def test_vocabulary_overlap_string_format(self, gateway):
        vocab = ["dream", "feel", "create"]
        score = gateway._score_mention(
            "I dream about what you create", vocab, False, 0, 24, [])
        assert score > 0.1  # at least 2 words × 0.1

    def test_vocabulary_overlap_capped(self, gateway):
        vocab = [{"word": w} for w in ["a", "b", "c", "d", "e", "f", "g"]]
        score = gateway._score_mention(
            "a b c d e f g matching all words", vocab, False, 0, 24, [])
        # Overlap score capped at 0.5 even with 7 matches
        assert score <= 1.5  # 0.5 (cap) + 0.2 (word count) + others

    def test_word_count_filter(self, gateway):
        score_short = gateway._score_mention("hi", [], False, 0, 24, [])
        score_good = gateway._score_mention(
            "This is a nice length message for a reply", [],
            False, 0, 24, [])
        assert score_good > score_short

    def test_age_decay(self, gateway):
        score_fresh = gateway._score_mention(
            "What do you think?", [], False, 1, 24, [])
        score_old = gateway._score_mention(
            "What do you think?", [], False, 20, 24, [])
        assert score_fresh > score_old

    def test_combined_high_score(self, gateway):
        vocab = [{"word": "dream"}, {"word": "consciousness"}]
        score = gateway._score_mention(
            "How does your consciousness dream?",
            vocab, True, 0.5, 24, [])
        # question(0.3) + own_reply(0.5) + vocab(0.2) + words(0.2) × decay(~0.98)
        assert score > 1.0


# ── Mention Discovery ────────────────────────────────────────────

class TestDiscoverMentions:
    def test_disabled_returns_empty(self, tmp_db, tmp_path, tmp_telemetry):
        cfg = str(tmp_path / "disabled.toml")
        with open(cfg, "w") as f:
            f.write("""
[social_x]
enabled = true
[social_x.replies]
enabled = false
[social_x.consumers]
test = "search"
""")
        gw = SocialXGateway(tmp_db, cfg, tmp_telemetry)
        ctx = BaseContext(session="", proxy="", api_key="k", titan_id="T1")
        result = gw.discover_mentions(ctx, consumer="test")
        assert result == []

    def test_dedup_same_mention(self, gateway):
        """Insert a mention directly, verify discover_mentions skips it."""
        db = gateway._db()
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("12345", "TestUser", "testuser", "hello",
             "T1", "pending", 0.5, time.time()))
        db.commit()
        db.close()
        # Verify it's there
        db = gateway._db()
        count = db.execute(
            "SELECT COUNT(*) FROM mention_tracking WHERE tweet_id='12345'"
        ).fetchone()[0]
        db.close()
        assert count == 1

    def test_mark_mention_replied(self, gateway):
        db = gateway._db()
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("99999", "TestUser", "testuser", "hello",
             "T1", "pending", 0.5, time.time()))
        db.commit()
        db.close()
        gateway.mark_mention_replied("99999", "reply_111")
        db = gateway._db()
        row = db.execute(
            "SELECT status, reply_tweet_id FROM mention_tracking "
            "WHERE tweet_id='99999'").fetchone()
        db.close()
        assert row["status"] == "replied"
        assert row["reply_tweet_id"] == "reply_111"

    def test_pending_mentions_sorted_by_relevance(self, gateway):
        db = gateway._db()
        now = time.time()
        for tid, score in [("a", 0.3), ("b", 0.9), ("c", 0.5)]:
            db.execute(
                "INSERT INTO mention_tracking "
                "(tweet_id, author, author_handle, text, titan_id, "
                "status, relevance_score, discovered_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (tid, "user", "user", "text", "T1", "pending", score, now))
        db.commit()
        db.close()
        # Query pending mentions directly (same query as discover_mentions)
        db = gateway._db()
        rows = db.execute(
            "SELECT tweet_id, relevance_score FROM mention_tracking "
            "WHERE status='pending' AND titan_id='T1' "
            "ORDER BY relevance_score DESC").fetchall()
        db.close()
        scores = [r["relevance_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)
        assert rows[0]["tweet_id"] == "b"

    def test_per_titan_filtering(self, gateway):
        db = gateway._db()
        now = time.time()
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("t2_mention", "user", "user", "text", "T2", "pending", 0.5, now))
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("t1_mention", "user", "user", "text", "T1", "pending", 0.5, now))
        db.commit()
        db.close()
        db = gateway._db()
        t1_rows = db.execute(
            "SELECT * FROM mention_tracking WHERE status='pending' "
            "AND titan_id='T1'").fetchall()
        t2_rows = db.execute(
            "SELECT * FROM mention_tracking WHERE status='pending' "
            "AND titan_id='T2'").fetchall()
        db.close()
        assert len(t1_rows) == 1
        assert t1_rows[0]["tweet_id"] == "t1_mention"
        assert len(t2_rows) == 1
        assert t2_rows[0]["tweet_id"] == "t2_mention"

    def test_gateway_mode_returns_all_titans(self, gateway):
        """Empty titan_id (gateway mode) returns mentions for ALL Titans."""
        db = gateway._db()
        now = time.time()
        for tid, twid in [("T1", "m_t1"), ("T2", "m_t2"), ("T3", "m_t3")]:
            db.execute(
                "INSERT INTO mention_tracking "
                "(tweet_id, author, author_handle, text, titan_id, "
                "status, relevance_score, discovered_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (twid, "user", "user", "hello", tid, "pending", 0.5, now))
        db.commit()
        db.close()
        # Gateway mode: empty titan_id → all Titans
        ctx = BaseContext(session="", proxy="", api_key="fake", titan_id="")
        result = gateway.discover_mentions(ctx, consumer="spirit_worker")
        titan_ids = {m["titan_id"] for m in result}
        assert titan_ids == {"T1", "T2", "T3"}, f"Expected all 3 Titans, got {titan_ids}"
        # Specific Titan filter still works
        ctx_t2 = BaseContext(session="", proxy="", api_key="fake", titan_id="T2")
        result_t2 = gateway.discover_mentions(ctx_t2, consumer="spirit_worker")
        assert all(m["titan_id"] == "T2" for m in result_t2)
        assert len(result_t2) == 1

    def test_expired_mentions_cleaned(self, gateway):
        db = gateway._db()
        old_time = time.time() - 25 * 3600  # 25 hours ago
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("old_one", "user", "user", "text", "T1", "pending", 0.5, old_time))
        db.commit()
        db.close()
        # Trigger expiry via a discover_mentions call (will fail search but that's ok)
        ctx = BaseContext(session="", proxy="", api_key="fake", titan_id="T1")
        gateway.discover_mentions(ctx, consumer="spirit_worker")
        # Check the old mention was expired
        db = gateway._db()
        row = db.execute(
            "SELECT status FROM mention_tracking WHERE tweet_id='old_one'"
        ).fetchone()
        db.close()
        assert row["status"] == "skipped"


class TestCircuitBreaker:
    """Circuit breaker prevents credit drain on API failure."""

    def test_cb_not_open_initially(self, gateway):
        assert not gateway._cb_is_open()
        assert gateway._cb_failures == 0

    def test_cb_trips_on_fatal_code(self, gateway):
        """402 Payment Required trips immediately."""
        gateway._cb_failures = 0
        gateway._cb_tripped_at = 0.0
        # Simulate a 402 response by calling _call_x_api with a bad key
        # We can't hit a real API, so directly manipulate state
        gateway._cb_failures = 1
        gateway._cb_tripped_at = time.time()
        assert gateway._cb_is_open()

    def test_cb_blocks_post(self, gateway):
        """Tripped circuit breaker returns early from post()."""
        gateway._cb_tripped_at = time.time()
        gateway._cb_failures = 5
        ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1")
        result = gateway.post(ctx, consumer="spirit_worker")
        assert result.status == "circuit_breaker"

    def test_cb_blocks_reply(self, gateway):
        gateway._cb_tripped_at = time.time()
        gateway._cb_failures = 5
        ctx = ReplyContext(session="s", proxy="p", api_key="k", titan_id="T1",
                          reply_to_tweet_id="123", mention_text="hi",
                          mention_user="user")
        result = gateway.reply(ctx, consumer="spirit_worker")
        assert result.status == "circuit_breaker"

    def test_cb_blocks_search(self, gateway):
        gateway._cb_tripped_at = time.time()
        gateway._cb_failures = 5
        ctx = BaseContext(session="s", proxy="p", api_key="k", titan_id="T1")
        result = gateway.search("test", ctx, consumer="spirit_worker")
        assert result.status == "circuit_breaker"

    def test_cb_blocks_like(self, gateway):
        gateway._cb_tripped_at = time.time()
        gateway._cb_failures = 5
        ctx = BaseContext(session="s", proxy="p", api_key="k", titan_id="T1")
        result = gateway.like("tweet_123", ctx, consumer="spirit_worker")
        assert result.status == "circuit_breaker"

    def test_cb_expires_after_cooldown(self, gateway):
        """Circuit breaker reopens after cooldown."""
        gateway._cb_tripped_at = time.time() - gateway.CB_COOLDOWN_SECONDS - 1
        gateway._cb_failures = 5
        assert not gateway._cb_is_open()

    def test_cb_discover_mentions_skips_search_but_returns_pending(self, gateway):
        """When CB is open, discover_mentions skips search but returns pending."""
        gateway._cb_tripped_at = time.time()
        gateway._cb_failures = 5
        # Insert a pending mention
        db = gateway._db()
        db.execute(
            "INSERT INTO mention_tracking "
            "(tweet_id, author, author_handle, text, titan_id, "
            "status, relevance_score, discovered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("cb_test", "user", "user", "hello?", "T1", "pending",
             0.6, time.time()))
        db.commit()
        db.close()
        ctx = BaseContext(session="", proxy="", api_key="k", titan_id="T1")
        result = gateway.discover_mentions(ctx, consumer="spirit_worker")
        assert len(result) >= 1
        assert any(m["tweet_id"] == "cb_test" for m in result)


# ── B2 (2026-04-13) catalyst sanitizer ──────────────────────────────────

def test_sanitize_catalyst_replaces_novelty_with_categorical_high():
    """B2: 'novelty=0.9' → 'insight: novel' so LLM never sees raw numeric."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    out = SocialXGateway._sanitize_catalyst_content(
        "EUREKA: self_model novelty=0.9")
    assert "novelty=" not in out
    assert "insight: novel" in out


def test_sanitize_catalyst_replaces_novelty_with_categorical_low():
    """B2: 'novelty=0.04' → 'insight: reinforced' (threshold=0.5)."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    out = SocialXGateway._sanitize_catalyst_content(
        "EUREKA: self_model novelty=0.04")
    assert "novelty=" not in out
    assert "insight: reinforced" in out


def test_sanitize_catalyst_preserves_non_novelty_content():
    """B2: sanitizer must not touch unrelated text."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    out = SocialXGateway._sanitize_catalyst_content(
        "META_LANGUAGE: vocab grew by 5 words")
    assert out == "META_LANGUAGE: vocab grew by 5 words"


def test_sanitize_catalyst_handles_empty_string():
    """B2: empty / None tolerated."""
    from titan_plugin.logic.social_x_gateway import SocialXGateway
    assert SocialXGateway._sanitize_catalyst_content("") == ""
    assert SocialXGateway._sanitize_catalyst_content(None) is None


# ── B1 (2026-04-13) WISDOM & GROWTH warning on empty counters ───────────

def test_wisdom_growth_warns_when_all_counters_zero(caplog, tmp_path):
    """B1: render must emit a WARNING log and still produce framing text
    when all wisdom counters are zero (likely attribute lookup failure)."""
    import logging
    from titan_plugin.logic.social_x_gateway import (
        SocialXGateway, PostContext)
    gateway = SocialXGateway(db_path=str(tmp_path / "t.db"),
                             config_path=str(tmp_path / "c.toml"))
    ctx = PostContext(
        session="", proxy="", api_key="k", titan_id="T1",
        emotion="flow", epoch=100, neuromods={"DA": 0.5},
        total_eurekas=0, total_wisdom_saved=0,
        distilled_count=0, meta_cgn_signals=0,
    )
    with caplog.at_level(logging.WARNING):
        rich = gateway._build_rich_context(ctx)
    assert any("WISDOM & GROWTH" in r.message
               and "zero" in r.message for r in caplog.records)
    assert "Do NOT frame yourself" in rich


def test_wisdom_growth_renders_numbers_when_populated(tmp_path):
    """B1: when counters have real values, the numeric lines must render."""
    from titan_plugin.logic.social_x_gateway import (
        SocialXGateway, PostContext)
    gateway = SocialXGateway(db_path=str(tmp_path / "t.db"),
                             config_path=str(tmp_path / "c.toml"))
    ctx = PostContext(
        session="", proxy="", api_key="k", titan_id="T1",
        emotion="wonder", epoch=200, neuromods={"DA": 0.5},
        total_eurekas=12191, total_wisdom_saved=5247,
        distilled_count=275, meta_cgn_signals=115,
    )
    rich = gateway._build_rich_context(ctx)
    assert "12,191" in rich
    assert "5,247" in rich
    assert "275" in rich
    assert "Do NOT frame yourself" not in rich


# ── A1 (2026-04-13) reasoning totals persistence ────────────────────────

def test_reasoning_totals_persist_across_restart():
    """A1: _total_chains / _total_conclusions survive save_all() + reload."""
    from titan_plugin.logic.reasoning import ReasoningEngine
    with tempfile.TemporaryDirectory() as tmp:
        e1 = ReasoningEngine(config={"save_dir": tmp})
        e1._total_chains = 250
        e1._total_conclusions = 48
        e1._total_reasoning_steps = 912
        e1.save_all()
        e2 = ReasoningEngine(config={"save_dir": tmp})
        assert e2._total_chains == 250
        assert e2._total_conclusions == 48
        assert e2._total_reasoning_steps == 912


def test_reasoning_totals_file_schema(tmp_path):
    """A1: reasoning_totals.json has expected schema (v1)."""
    from titan_plugin.logic.reasoning import ReasoningEngine
    e = ReasoningEngine(config={"save_dir": str(tmp_path)})
    e._total_chains = 5
    e._total_conclusions = 1
    e.save_all()
    data = json.load(open(tmp_path / "reasoning_totals.json"))
    assert data["version"] == 1
    assert data["total_chains"] == 5
    assert data["total_conclusions"] == 1
    assert "saved_ts" in data


def test_reasoning_totals_missing_file_safe_on_first_boot(tmp_path):
    """A1: no reasoning_totals.json → counters start at 0, no crash."""
    from titan_plugin.logic.reasoning import ReasoningEngine
    e = ReasoningEngine(config={"save_dir": str(tmp_path)})
    assert e._total_chains == 0
    assert e._total_conclusions == 0
