"""rFP_universal_sqlite_writer 2026-04-27 expansion bundle — tests.

Covers the 3 new IMW writer instances (social_graph + events_teacher +
consciousness) plus the inner_memory full-canonical config flip:

  1. inner_memory.db config — all writer-using tables in tables_canonical
  2. social_graph.db / events_teacher.db / consciousness.db config sections
     loadable via IMWConfig.from_titan_config_section
  3. Each owner class auto-constructs writer client when enabled
  4. consciousness.db fully migrated — insert_epoch routes through writer
  5. social_graph + events_teacher infrastructure shipped (per-call-site
     refactor is followup work; for now `_route_write` exists + writer
     client is constructed but most call sites still use direct sqlite3 —
     verify the helper is fallback-safe)
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from titan_plugin.persistence.config import IMWConfig


# ── Configuration loading ───────────────────────────────────────────────


def test_inner_memory_full_canonical_config():
    """[persistence] section: mode=canonical + all 20 writer-using tables."""
    cfg = IMWConfig.from_titan_config_section("persistence")
    assert cfg.enabled is True
    assert cfg.mode == "canonical", (
        f"inner_memory should be 'canonical' post-fast-track; got {cfg.mode!r}"
    )
    assert cfg.fast_path_enabled is False, (
        "fast_path_enabled must be off post-2026-04-27"
    )
    expected = {
        "self_insights", "self_predictions", "hormone_snapshots", "program_fires",
        "action_chains", "action_chains_step", "creative_works", "event_markers",
        "visual_autobiography", "vocabulary", "chain_archive", "meta_wisdom",
        "composition_history", "teacher_sessions", "grammar_patterns",
        "kin_encounters", "kin_profiles", "creative_journal", "word_associations",
        "knowledge_concepts",
    }
    actual = set(cfg.tables_canonical)
    missing = expected - actual
    assert not missing, f"inner_memory canonical missing tables: {missing}"


@pytest.mark.parametrize("section,expected_db", [
    ("persistence_social_graph", "social_graph.db"),
    ("persistence_events_teacher", "events_teacher.db"),
    ("persistence_consciousness", "consciousness.db"),
])
def test_new_writer_sections_loadable_canonical_default(section, expected_db):
    """Each new [persistence_*] section loads cleanly + defaults to canonical."""
    cfg = IMWConfig.from_titan_config_section(section)
    assert cfg.enabled is True, f"{section} must be enabled by default"
    assert cfg.mode == "canonical", (
        f"{section}.mode should be 'canonical'; got {cfg.mode!r}"
    )
    assert cfg.fast_path_enabled is False, (
        f"{section}.fast_path_enabled must be off"
    )
    assert cfg.db_path.endswith(expected_db), (
        f"{section}.db_path should end with {expected_db}; got {cfg.db_path!r}"
    )
    assert len(cfg.tables_canonical) > 0, (
        f"{section}.tables_canonical must list at least one table"
    )


# ── Owner-class writer-client construction ──────────────────────────────


def test_consciousness_db_constructs_writer_client_in_production(
    monkeypatch, tmp_path,
):
    """ConsciousnessDB() in production constructs writer client when
    [persistence_consciousness].enabled = true."""
    from titan_plugin.persistence import config as cfg_mod
    from titan_plugin.persistence import writer_client as wc_mod
    from titan_plugin.logic.consciousness import ConsciousnessDB

    real_db = tmp_path / "consciousness.db"
    real_db.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    fake_cfg = cfg_mod.IMWConfig(
        enabled=True, mode="canonical",
        socket_path=str(tmp_path / "c.sock"),
        wal_path=str(tmp_path / "c.wal"),
        journal_dir=str(tmp_path / "j"),
        db_path="consciousness.db",  # relative — production reality
        shadow_db_path="consciousness_shadow.db",
        tables_canonical=["epochs"],
    )
    monkeypatch.setattr(
        cfg_mod.IMWConfig, "from_titan_config_section",
        classmethod(lambda cls, section_name="persistence": fake_cfg),
    )
    fake_client = MagicMock()
    fake_client_class = MagicMock(return_value=fake_client)
    monkeypatch.setattr(wc_mod, "InnerMemoryWriterClient", fake_client_class)

    db = ConsciousnessDB(db_path=str(real_db.resolve()))
    assert db._writer is fake_client, (
        "consciousness writer client must be constructed when "
        "[persistence_consciousness].enabled = true"
    )


def test_consciousness_insert_epoch_routes_through_writer():
    """ConsciousnessDB.insert_epoch routes through writer when client present."""
    from titan_plugin.logic.consciousness import ConsciousnessDB, EpochRecord

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        mock_writer = MagicMock()
        db = ConsciousnessDB(db_path=f.name, writer_client=mock_writer)

    rec = EpochRecord(
        epoch_id=1, timestamp=0.0, block_hash="",
        state_vector=[0.0], drift_vector=[0.0], trajectory_vector=[0.0],
        journey_point=(0.0, 0.0, 0.0), curvature=0.0, density=0.0,
        distillation="", anchored_tx="",
    )
    db.insert_epoch(rec)
    assert mock_writer.write.called, "insert_epoch must call writer.write"
    call = mock_writer.write.call_args
    assert "INSERT OR REPLACE INTO epochs" in call.args[0]
    assert call.kwargs.get("table") == "epochs"


def test_social_graph_constructs_writer_client(monkeypatch, tmp_path):
    """SocialGraph() constructs writer client when [persistence_social_graph]
    is enabled — infrastructure presence (per-site refactor is followup)."""
    from titan_plugin.persistence import config as cfg_mod
    from titan_plugin.persistence import writer_client as wc_mod
    from titan_plugin.core.social_graph import SocialGraph

    real_db = tmp_path / "social_graph.db"
    monkeypatch.chdir(tmp_path)

    fake_cfg = cfg_mod.IMWConfig(
        enabled=True, mode="canonical",
        socket_path=str(tmp_path / "sg.sock"),
        wal_path=str(tmp_path / "sg.wal"),
        journal_dir=str(tmp_path / "j"),
        db_path="social_graph.db",
        shadow_db_path="social_graph_shadow.db",
        tables_canonical=["user_profiles"],
    )
    monkeypatch.setattr(
        cfg_mod.IMWConfig, "from_titan_config_section",
        classmethod(lambda cls, section_name="persistence": fake_cfg),
    )
    fake_client = MagicMock()
    fake_client_class = MagicMock(return_value=fake_client)
    monkeypatch.setattr(wc_mod, "InnerMemoryWriterClient", fake_client_class)

    sg = SocialGraph(db_path=str(real_db.resolve()))
    assert sg._writer is fake_client, (
        "SocialGraph writer client must be constructed when enabled — "
        "without it, future per-site refactor cannot route through daemon"
    )


def test_events_teacher_db_constructs_writer_client(monkeypatch, tmp_path):
    """EventsTeacherDB() constructs writer client when section enabled."""
    from titan_plugin.persistence import config as cfg_mod
    from titan_plugin.persistence import writer_client as wc_mod
    from titan_plugin.logic.events_teacher import EventsTeacherDB

    real_db = tmp_path / "events_teacher.db"
    monkeypatch.chdir(tmp_path)

    fake_cfg = cfg_mod.IMWConfig(
        enabled=True, mode="canonical",
        socket_path=str(tmp_path / "et.sock"),
        wal_path=str(tmp_path / "et.wal"),
        journal_dir=str(tmp_path / "j"),
        db_path="events_teacher.db",
        shadow_db_path="events_teacher_shadow.db",
        tables_canonical=["felt_experiences"],
    )
    monkeypatch.setattr(
        cfg_mod.IMWConfig, "from_titan_config_section",
        classmethod(lambda cls, section_name="persistence": fake_cfg),
    )
    fake_client = MagicMock()
    fake_client_class = MagicMock(return_value=fake_client)
    monkeypatch.setattr(wc_mod, "InnerMemoryWriterClient", fake_client_class)

    et = EventsTeacherDB(db_path=str(real_db.resolve()))
    assert et._writer is fake_client


# ── _route_write helper fallback safety ─────────────────────────────────


def test_social_graph_route_write_direct_fallback_when_no_writer(tmp_path):
    """SocialGraph._route_write must fall back to direct sqlite3 when
    writer is None (current state for most call sites pre-refactor)."""
    from titan_plugin.core.social_graph import SocialGraph

    db_file = tmp_path / "social_graph.db"
    sg = SocialGraph(db_path=str(db_file), writer_client=None)
    # Force-disable writer by patching after construction (easier than
    # mocking config).
    sg._writer = None

    sg._route_write(
        "INSERT OR IGNORE INTO engagement_ledger (tweet_id, user_name, action, timestamp) VALUES (?, ?, ?, ?)",
        ("tweet_x", "alice", "reply", 0.0),
        table="engagement_ledger",
    )
    # Verify the row landed via direct sqlite3.
    conn = sqlite3.connect(str(db_file), timeout=2)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM engagement_ledger WHERE tweet_id=?",
            ("tweet_x",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1, "fallback direct write must land in DB"


def test_events_teacher_route_write_direct_fallback_when_no_writer(tmp_path):
    """EventsTeacherDB._route_write fallback safety."""
    from titan_plugin.logic.events_teacher import EventsTeacherDB

    db_file = tmp_path / "events_teacher.db"
    et = EventsTeacherDB(db_path=str(db_file), writer_client=None)
    et._writer = None

    et._route_write(
        "INSERT INTO felt_experiences "
        "(titan_id, source, author, topic, felt_summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("t1", "x", "alice", "test", "summary", 0.0),
        table="felt_experiences",
    )
    conn = sqlite3.connect(str(db_file), timeout=2)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM felt_experiences"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1
