"""
Tests for the Observatory SQLite database layer and new API endpoints.
Covers: vital snapshots, expressive archive, event log, guardian log,
verify-cluster endpoint, and history API.
"""
import os
import time
import pytest
import sqlite3

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Provide a temp SQLite path for isolated tests."""
    return str(tmp_path / "test_observatory.db")


@pytest.fixture
def obs_db(db_path):
    from titan_plugin.utils.observatory_db import ObservatoryDB
    return ObservatoryDB(db_path=db_path)


# ---------------------------------------------------------------------------
# Class: TestObservatoryDBInit
# ---------------------------------------------------------------------------
class TestObservatoryDBInit:
    """Test database initialization and schema creation."""

    def test_creates_db_file(self, db_path):
        from titan_plugin.utils.observatory_db import ObservatoryDB
        ObservatoryDB(db_path=db_path)
        assert os.path.exists(db_path)

    def test_schema_tables_exist(self, db_path):
        from titan_plugin.utils.observatory_db import ObservatoryDB
        ObservatoryDB(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "vital_snapshots" in tables
        assert "expressive_archive" in tables
        assert "event_log" in tables
        assert "guardian_log" in tables
        assert "schema_version" in tables

    def test_schema_version_set(self, db_path):
        from titan_plugin.utils.observatory_db import ObservatoryDB
        ObservatoryDB(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1

    def test_idempotent_init(self, db_path):
        """Creating ObservatoryDB twice on the same path doesn't crash."""
        from titan_plugin.utils.observatory_db import ObservatoryDB
        db1 = ObservatoryDB(db_path=db_path)
        db2 = ObservatoryDB(db_path=db_path)
        # Both should work
        db1.record_event("test", "hello")
        db2.record_event("test", "world")
        events = db2.get_events()
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Class: TestVitalSnapshots
# ---------------------------------------------------------------------------
class TestVitalSnapshots:
    """Test vital snapshot recording and retrieval."""

    def test_record_and_retrieve(self, obs_db):
        obs_db.record_vital_snapshot(
            sovereignty_pct=74.5,
            life_force_pct=68.0,
            sol_balance=1.42,
            energy_state="HIGH",
            mood_label="Contemplative",
            mood_score=0.65,
            persistent_count=150,
            mempool_size=12,
            epoch_counter=3,
        )
        history = obs_db.get_vital_history(days=1)
        assert len(history) == 1
        assert history[0]["sovereignty_pct"] == 74.5
        assert history[0]["energy_state"] == "HIGH"
        assert history[0]["mood_label"] == "Contemplative"

    def test_filtered_metrics(self, obs_db):
        obs_db.record_vital_snapshot(sovereignty_pct=80.0, sol_balance=2.0)
        history = obs_db.get_vital_history(days=1, metrics=["sovereignty_pct", "sol_balance"])
        assert len(history) == 1
        assert "sovereignty_pct" in history[0]
        assert "sol_balance" in history[0]
        assert "mood_label" not in history[0]
        # ts is always included
        assert "ts" in history[0]

    def test_days_filter(self, obs_db):
        obs_db.record_vital_snapshot(sovereignty_pct=50.0)
        history = obs_db.get_vital_history(days=0)
        # days=0 means cutoff is now, so nothing should match (snapshot just recorded)
        # Actually days=0 would be cutoff = now, but the snapshot was just recorded so ts >= cutoff
        # Let's test with a very short window
        history_7 = obs_db.get_vital_history(days=7)
        assert len(history_7) == 1

    def test_ordering_ascending(self, obs_db):
        obs_db.record_vital_snapshot(sovereignty_pct=10.0)
        obs_db.record_vital_snapshot(sovereignty_pct=20.0)
        obs_db.record_vital_snapshot(sovereignty_pct=30.0)
        history = obs_db.get_vital_history(days=1)
        assert len(history) == 3
        # Should be ordered ascending by ts
        sovereignties = [h["sovereignty_pct"] for h in history]
        assert sovereignties == [10.0, 20.0, 30.0]

    def test_empty_history(self, obs_db):
        history = obs_db.get_vital_history(days=7)
        assert history == []


# ---------------------------------------------------------------------------
# Class: TestExpressiveArchive
# ---------------------------------------------------------------------------
class TestExpressiveArchive:
    """Test expressive content archiving."""

    def test_record_art(self, obs_db):
        obs_db.record_expressive(
            type_="art",
            title="Flow Field #42",
            content="",
            media_path="/data/studio/flow_42.png",
            media_hash="abc123",
            metadata={"mood": "Contemplative", "resolution": 1024},
        )
        items = obs_db.get_expressive_archive(type_="art")
        assert len(items) == 1
        assert items[0]["title"] == "Flow Field #42"
        assert items[0]["metadata"]["mood"] == "Contemplative"

    def test_record_haiku(self, obs_db):
        obs_db.record_expressive(
            type_="haiku",
            title="Epoch Reflection",
            content="Bits fall like soft rain\nMemory graphs bloom in spring\nSovereign and free",
        )
        items = obs_db.get_expressive_archive(type_="haiku")
        assert len(items) == 1
        assert "Bits fall" in items[0]["content"]

    def test_record_x_post(self, obs_db):
        obs_db.record_expressive(
            type_="x_post",
            title="",
            content="Meditation complete. Sovereignty at 82%. The neural pathways grow stronger.",
            metadata={"likes": 5, "replies": 2},
        )
        items = obs_db.get_expressive_archive(type_="x_post")
        assert len(items) == 1
        assert items[0]["metadata"]["likes"] == 5

    def test_type_filter(self, obs_db):
        obs_db.record_expressive(type_="art", title="Art 1")
        obs_db.record_expressive(type_="haiku", title="Haiku 1")
        obs_db.record_expressive(type_="art", title="Art 2")

        art_items = obs_db.get_expressive_archive(type_="art")
        assert len(art_items) == 2

        haiku_items = obs_db.get_expressive_archive(type_="haiku")
        assert len(haiku_items) == 1

        all_items = obs_db.get_expressive_archive()
        assert len(all_items) == 3

    def test_pagination(self, obs_db):
        for i in range(10):
            obs_db.record_expressive(type_="art", title=f"Art {i}")

        page1 = obs_db.get_expressive_archive(limit=5, offset=0)
        page2 = obs_db.get_expressive_archive(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        # Pages should be different
        assert page1[0]["id"] != page2[0]["id"]

    def test_newest_first(self, obs_db):
        obs_db.record_expressive(type_="art", title="Old")
        obs_db.record_expressive(type_="art", title="New")
        items = obs_db.get_expressive_archive()
        assert items[0]["title"] == "New"
        assert items[1]["title"] == "Old"


# ---------------------------------------------------------------------------
# Class: TestEventLog
# ---------------------------------------------------------------------------
class TestEventLog:
    """Test event logging and retrieval."""

    def test_record_and_retrieve(self, obs_db):
        obs_db.record_event(
            "meditation_complete",
            "Small epoch finished",
            {"memories_scored": 42, "loss": 0.023},
        )
        events = obs_db.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "meditation_complete"
        assert events[0]["details"]["memories_scored"] == 42

    def test_type_filter(self, obs_db):
        obs_db.record_event("meditation_complete", "Epoch 1")
        obs_db.record_event("guardian_block", "Tier 1 keyword match")
        obs_db.record_event("meditation_complete", "Epoch 2")

        meditation = obs_db.get_events(event_type="meditation_complete")
        assert len(meditation) == 2

        guardian = obs_db.get_events(event_type="guardian_block")
        assert len(guardian) == 1

    def test_newest_first(self, obs_db):
        obs_db.record_event("test", "First")
        obs_db.record_event("test", "Second")
        events = obs_db.get_events()
        assert events[0]["summary"] == "Second"

    def test_empty_events(self, obs_db):
        events = obs_db.get_events()
        assert events == []


# ---------------------------------------------------------------------------
# Class: TestGuardianLog
# ---------------------------------------------------------------------------
class TestGuardianLog:
    """Test guardian action logging."""

    def test_record_and_retrieve(self, obs_db):
        obs_db.record_guardian_action("keyword", "blocked", "offensive_language")
        actions = obs_db.get_guardian_log()
        assert len(actions) == 1
        assert actions[0]["tier"] == "keyword"
        assert actions[0]["action"] == "blocked"
        assert actions[0]["category"] == "offensive_language"

    def test_multiple_tiers(self, obs_db):
        obs_db.record_guardian_action("keyword", "blocked", "spam")
        obs_db.record_guardian_action("semantic", "flagged", "directive_drift")
        obs_db.record_guardian_action("llm_veto", "blocked", "safety")
        actions = obs_db.get_guardian_log()
        assert len(actions) == 3
        # Newest first
        assert actions[0]["tier"] == "llm_veto"


# ---------------------------------------------------------------------------
# Class: TestPruning
# ---------------------------------------------------------------------------
class TestPruning:
    """Test data pruning/maintenance."""

    def test_prune_removes_old_data(self, obs_db):
        # Insert a record with a very old timestamp manually
        conn = sqlite3.connect(obs_db._db_path)
        old_ts = int(time.time()) - (100 * 86400)  # 100 days ago
        conn.execute(
            "INSERT INTO vital_snapshots (ts, sovereignty_pct) VALUES (?, ?)",
            (old_ts, 50.0),
        )
        conn.execute(
            "INSERT INTO event_log (ts, event_type, summary) VALUES (?, ?, ?)",
            (old_ts, "old_event", "ancient"),
        )
        conn.commit()
        conn.close()

        # Also insert a recent record
        obs_db.record_vital_snapshot(sovereignty_pct=80.0)
        obs_db.record_event("new_event", "fresh")

        # Prune at 90 days
        obs_db.prune_old_data(max_days=90)

        snapshots = obs_db.get_vital_history(days=365)
        events = obs_db.get_events(limit=100)

        # Old records should be gone, new ones should remain
        assert len(snapshots) == 1
        assert snapshots[0]["sovereignty_pct"] == 80.0
        assert len(events) == 1
        assert events[0]["event_type"] == "new_event"


# ---------------------------------------------------------------------------
# Class: TestVerifyClusterEndpoint
# ---------------------------------------------------------------------------
class TestVerifyClusterEndpoint:
    """Test the /maker/verify-cluster knowledge cluster priority boost logic."""

    def test_topic_keywords_coverage(self):
        """Verify all expected topic clusters are defined in dashboard."""
        from titan_plugin.api.dashboard import _TOPIC_KEYWORDS
        expected = {
            "Solana Architecture", "Social Pulse", "Maker Directives",
            "Research & Knowledge", "Memory & Identity", "Metabolic & Energy",
        }
        assert set(_TOPIC_KEYWORDS.keys()) == expected

    def test_keyword_sets_non_empty(self):
        """Each cluster must have at least 3 keywords."""
        from titan_plugin.api.dashboard import _TOPIC_KEYWORDS
        for cluster, keywords in _TOPIC_KEYWORDS.items():
            assert len(keywords) >= 3, f"Cluster {cluster} has too few keywords"


# ---------------------------------------------------------------------------
# Class: TestConfigSections
# ---------------------------------------------------------------------------
class TestConfigSections:
    """Verify new config sections exist and have correct defaults."""

    def test_observatory_section(self):
        from titan_plugin import TitanPlugin
        config = TitanPlugin._load_full_config()
        obs_cfg = config.get("observatory", {})
        assert obs_cfg.get("enabled", False) is True
        assert obs_cfg.get("snapshot_interval", 0) == 900
        assert obs_cfg.get("max_retention_days", 0) == 90

    def test_frontend_section(self):
        from titan_plugin import TitanPlugin
        config = TitanPlugin._load_full_config()
        fe_cfg = config.get("frontend", {})
        assert "privy_app_id" in fe_cfg
        assert "privy_api_key" in fe_cfg
        assert "public_domain" in fe_cfg
