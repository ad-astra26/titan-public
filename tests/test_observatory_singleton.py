"""rFP_universal_sqlite_writer Phase 2 — singleton accessor tests.

`get_observatory_db()` is the per-process singleton entry point that closes
BUG-TRINITY-SNAPSHOT-DB-LOCKED's structural cause: multiple in-process
ObservatoryDB constructions creating parallel SQLite connections + parallel
writer clients. Cross-process contention is handled by canonical mode in
the writer daemon; this test only covers the in-process invariant.
"""

from __future__ import annotations

import tempfile
import threading

import pytest

from titan_plugin.utils.observatory_db import (
    ObservatoryDB,
    get_observatory_db,
    reset_observatory_db_singleton_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts from a clean singleton slot."""
    reset_observatory_db_singleton_for_tests()
    yield
    reset_observatory_db_singleton_for_tests()


def test_singleton_returns_same_instance_across_calls():
    """Two calls with no args return the same object."""
    a = get_observatory_db()
    b = get_observatory_db()
    assert a is b, "singleton must return identical instance"


def test_singleton_under_concurrent_access():
    """100 threads racing get_observatory_db() must all see the same instance."""
    results: list[ObservatoryDB] = []
    barrier = threading.Barrier(100)

    def worker():
        barrier.wait()
        results.append(get_observatory_db())

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 100 must be the same object — lazy init is thread-safe.
    first = results[0]
    assert all(r is first for r in results), (
        f"singleton race: got {len(set(id(r) for r in results))} distinct "
        f"instances under 100-way contention"
    )


def test_explicit_db_path_does_not_poison_singleton():
    """Passing an explicit db_path returns a separate instance for test
    isolation — must NOT replace the production singleton."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        custom_path = f.name
    custom = get_observatory_db(db_path=custom_path)
    assert custom._db_path.endswith(".db")
    # Production singleton was never touched.
    prod = get_observatory_db()
    assert prod is not custom, (
        "explicit-path call must not poison the production singleton"
    )
    # And the production singleton is stable across calls.
    prod2 = get_observatory_db()
    assert prod2 is prod


def test_explicit_db_path_matching_existing_returns_singleton():
    """If the existing singleton happens to live at the requested path,
    reuse it rather than constructing a duplicate."""
    first = get_observatory_db()
    second = get_observatory_db(db_path=first._db_path)
    assert first is second, (
        "explicit db_path that matches the existing singleton's path "
        "must reuse the singleton (avoid duplicate construction)"
    )


def test_reset_helper_for_tests():
    """`reset_observatory_db_singleton_for_tests()` clears the slot."""
    a = get_observatory_db()
    reset_observatory_db_singleton_for_tests()
    b = get_observatory_db()
    assert a is not b, "reset must produce a new instance on next call"


def test_production_construct_does_not_skip_writer_due_to_relative_path(
    monkeypatch, tmp_path,
):
    """Hot-fix regression test (2026-04-27 ~16:00 UTC): production
    ObservatoryDB() must construct the writer client. The pre-fix bug was
    that cfg.db_path = "data/observatory.db" (relative) never equaled
    self._db_path = "/abs/path/data/observatory.db" (absolute, derived
    from __file__), so the path-isolation guard wrongly tripped in
    production → writer skipped → all writes went direct → BUG-TRINITY-
    SNAPSHOT-DB-LOCKED kept firing post-Phase 2 deploy.

    Reproducer: cwd-relative cfg.db_path that, via realpath, points at
    the SAME inode as the absolute self._db_path. The fix must accept
    them as equivalent and proceed with writer construction.
    """
    # Arrange: a fake cfg whose db_path is RELATIVE (production reality).
    from titan_plugin.persistence import config as cfg_mod
    from titan_plugin.persistence import writer_client as wc_mod
    from unittest.mock import MagicMock

    # Create the relative + absolute pair that resolves to the same file.
    real_db = tmp_path / "observatory.db"
    real_db.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    fake_cfg = cfg_mod.IMWConfig(
        enabled=True,
        mode="canonical",
        socket_path=str(tmp_path / "obs.sock"),
        wal_path=str(tmp_path / "obs.wal"),
        journal_dir=str(tmp_path / "journals"),
        db_path="observatory.db",  # ← RELATIVE — the production case
        shadow_db_path="observatory_shadow.db",
        tables_canonical=["trinity_snapshots"],
    )
    monkeypatch.setattr(
        cfg_mod.IMWConfig, "from_titan_config_section",
        classmethod(lambda cls, section_name="persistence": fake_cfg),
    )
    fake_client_class = MagicMock()
    fake_client_instance = MagicMock()
    fake_client_class.return_value = fake_client_instance
    monkeypatch.setattr(wc_mod, "InnerMemoryWriterClient", fake_client_class)

    # Act: production-style construct with explicit absolute path that
    # realpath-equals the relative cfg.db_path.
    db = ObservatoryDB(db_path=str(real_db.resolve()))

    # Assert: writer was constructed despite literal-string mismatch.
    assert db._writer is fake_client_instance, (
        "production path-isolation guard must accept "
        "realpath-equivalent paths (relative vs absolute pointing at "
        "the same file) — the pre-fix bug compared raw strings and "
        "wrongly skipped the writer in production"
    )
