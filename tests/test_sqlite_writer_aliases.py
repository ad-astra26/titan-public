"""rFP_universal_sqlite_writer Phase 4 — generic-name alias tests.

The IMW engine is path-agnostic and serves any SQLite DB. The original names
(`InnerMemoryWriterClient`, `IMWDaemon`) reflect the first DB it migrated;
`SqliteWriterClient` / `SqliteWriterDaemon` are the preferred generic names
going forward. This test pins the alias relationship so a refactor can't
silently break either name.
"""

from __future__ import annotations


def test_sqlite_writer_client_alias_resolves():
    """`SqliteWriterClient` must be importable from titan_plugin.persistence
    AND be the same class as `InnerMemoryWriterClient`."""
    from titan_plugin.persistence import (
        InnerMemoryWriterClient,
        SqliteWriterClient,
    )
    assert SqliteWriterClient is InnerMemoryWriterClient, (
        "SqliteWriterClient must alias to the same class as "
        "InnerMemoryWriterClient — anything else is silent ABI drift"
    )


def test_sqlite_writer_daemon_alias_resolves():
    """`SqliteWriterDaemon` must be the same class as `IMWDaemon`."""
    from titan_plugin.persistence.writer_service import (
        IMWDaemon,
        SqliteWriterDaemon,
    )
    assert SqliteWriterDaemon is IMWDaemon


def test_imwconfig_loads_arbitrary_section_name():
    """`IMWConfig.from_titan_config_section()` must accept any
    [persistence_*] section name — this is the foundation of the
    generic-pattern adoption recipe."""
    from titan_plugin.persistence.config import IMWConfig

    cfg_inner = IMWConfig.from_titan_config_section("persistence")
    cfg_obs = IMWConfig.from_titan_config_section("persistence_observatory")
    # Both should load without error and produce different db_paths.
    assert cfg_inner.db_path != cfg_obs.db_path, (
        "Per-DB sections must load distinct configs"
    )
    # Sanity: both should point at SQLite files we recognize.
    assert cfg_inner.db_path.endswith("inner_memory.db")
    assert cfg_obs.db_path.endswith("observatory.db")


def test_observatory_section_canonical_mode():
    """rFP_universal_sqlite_writer Phase 3: observatory.db must boot in
    canonical mode with the 14 hot tables listed. Locks the config-side
    of the bug fix."""
    from titan_plugin.persistence.config import IMWConfig

    cfg = IMWConfig.from_titan_config_section("persistence_observatory")
    assert cfg.enabled is True
    assert cfg.mode == "canonical", (
        f"observatory mode should be 'canonical' post 2026-04-27; got {cfg.mode!r}"
    )
    assert cfg.fast_path_enabled is False, (
        "fast_path_enabled must be off post-rFP — daemon is sole writer"
    )
    expected_tables = {
        "trinity_snapshots", "growth_snapshots", "vital_snapshots",
        "event_log", "expressive_archive", "guardian_log", "v4_snapshots",
        "reflex_log", "neuromod_history", "hormonal_history",
        "expression_history", "dreaming_history", "training_history",
        "clock_history",
    }
    actual = set(cfg.tables_canonical)
    missing = expected_tables - actual
    extra = actual - expected_tables
    assert not missing, f"observatory canonical tables missing: {missing}"
    # Extra tables are not a failure; future tables may be added.
    if extra:
        # Just informational — flag if someone added a non-existent table.
        pass
