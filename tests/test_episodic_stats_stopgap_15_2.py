"""
test_episodic_stats_stopgap_15_2 — §3L Phase 15 chunk 15.2 stopgap.

The frozen episodic_memory.db GROUP-BY bleed is mtime-gated in the coord-build:
get_stats is recomputed ONLY when the backing DB file's mtime changes (while
frozen → computed once then cached; recomputes automatically when the rehoming
RFP restores the write path). Verifies the cache behavior — no stale bug.
"""
import types

from titan_hcl.logic import snapshot_builders


class _FakeEpisodic:
    def __init__(self, db_path, stats):
        self._db_path = db_path
        self._stats = stats
        self.calls = 0

    def get_stats(self):
        self.calls += 1
        return dict(self._stats)


def _reset_cache():
    snapshot_builders._EPISODIC_STATS_CACHE["data"] = None
    snapshot_builders._EPISODIC_STATS_CACHE["mtime"] = None


def test_recomputes_once_then_caches_on_stable_mtime(tmp_path, monkeypatch):
    _reset_cache()
    db = tmp_path / "episodic_memory.db"
    db.write_text("x")
    em = _FakeEpisodic(str(db), {"total": 161554, "by_type": {"kin_exchange": 100}})

    # First call computes; subsequent calls with unchanged mtime serve cache.
    r1 = snapshot_builders._episodic_stats_mtime_gated(em)
    r2 = snapshot_builders._episodic_stats_mtime_gated(em)
    r3 = snapshot_builders._episodic_stats_mtime_gated(em)
    assert r1 == {"total": 161554, "by_type": {"kin_exchange": 100}}
    assert r2 == r1 and r3 == r1
    assert em.calls == 1, "frozen DB must be aggregated once, not every cycle"


def test_recomputes_when_mtime_advances(tmp_path, monkeypatch):
    _reset_cache()
    db = tmp_path / "episodic_memory.db"
    db.write_text("x")
    em = _FakeEpisodic(str(db), {"total": 1})

    snapshot_builders._episodic_stats_mtime_gated(em)
    assert em.calls == 1

    # Simulate the rehoming RFP restoring writes (mtime advances).
    import os
    st = os.stat(str(db))
    os.utime(str(db), (st.st_atime + 10, st.st_mtime + 10))
    em._stats = {"total": 2}
    r = snapshot_builders._episodic_stats_mtime_gated(em)
    assert em.calls == 2, "must recompute when DB mtime advances (writes resumed)"
    assert r == {"total": 2}


def test_missing_db_does_not_crash(tmp_path):
    _reset_cache()
    em = _FakeEpisodic(str(tmp_path / "nonexistent.db"), {"total": 0})
    r = snapshot_builders._episodic_stats_mtime_gated(em)
    assert r == {"total": 0}
    assert em.calls == 1
