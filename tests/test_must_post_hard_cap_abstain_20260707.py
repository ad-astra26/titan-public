"""FX.4 — a hard-capped must-post archetype ABSTAINS (RFP_social_x §5.FX.4).

When a must-post (soul_diary/proof_day) has hit MUST_POST_DAILY_HARD_CAP (3 attempts
in 24h, ANY status incl 'failed'), `must_post_hard_capped()` returns True so
find_candidate abstains — the dispatcher then falls through to other archetypes
instead of re-selecting a blocked must-post every tick (the 2026-07 outage that
silenced a whole Titan for a day when its diary X-post failed 3x).
"""
import sqlite3
import time
import types

from titan_hcl.logic.social_x.archetypes.base import ArchetypeBase


def _mkdb(rows):
    """In-memory actions table seeded with (post_type, titan_id, status, age_s)."""
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.execute("CREATE TABLE actions (action_type TEXT, post_type TEXT, "
              "titan_id TEXT, status TEXT, created_at REAL)")
    now = time.time()
    for pt, tid, st, age in rows:
        c.execute("INSERT INTO actions VALUES ('post',?,?,?,?)",
                  (pt, tid, st, now - age))
    c.commit()
    return c


class _Arch:
    name = "soul_diary"
    def __init__(self, conn, hard_cap=3):
        self._c = conn
        self.gateway = types.SimpleNamespace(
            _load_config=lambda: {"must_post_daily_hard_cap": hard_cap})
    def _conn(self):
        return self._c
    must_post_hard_capped = ArchetypeBase.must_post_hard_capped


def test_capped_when_3_attempts_any_status():
    # 3 soul_diary rows in 24h, all 'failed' → capped (any status counts)
    db = _mkdb([("soul_diary", "T3", "failed", 3600 * i) for i in (1, 2, 3)])
    assert _Arch(db).must_post_hard_capped(titan_id="T3") is True


def test_not_capped_under_3():
    db = _mkdb([("soul_diary", "T3", "failed", 3600), ("soul_diary", "T3", "posted", 7200)])
    assert _Arch(db).must_post_hard_capped(titan_id="T3") is False


def test_old_attempts_not_counted():
    # 4 attempts but all >24h old → not capped
    db = _mkdb([("soul_diary", "T3", "failed", 86400 + 3600 * i) for i in range(1, 5)])
    assert _Arch(db).must_post_hard_capped(titan_id="T3") is False


def test_scoped_to_this_titan_and_post_type():
    # 3 for T2 + 3 verified soul_diary don't count toward T3's soul_diary
    db = _mkdb([("soul_diary", "T2", "failed", 3600 * i) for i in (1, 2, 3)]
               + [("proof_day", "T3", "failed", 3600 * i) for i in (1, 2, 3)])
    assert _Arch(db).must_post_hard_capped(titan_id="T3") is False


def test_disabled_when_cap_zero():
    db = _mkdb([("soul_diary", "T3", "failed", 3600 * i) for i in (1, 2, 3, 4, 5)])
    assert _Arch(db, hard_cap=0).must_post_hard_capped(titan_id="T3") is False


def test_verified_and_unverified_also_count():
    # mix of statuses — all count toward the runaway backstop
    db = _mkdb([("soul_diary", "T3", "posted", 3600),
                ("soul_diary", "T3", "unverified", 7200),
                ("soul_diary", "T3", "verified", 10800)])
    assert _Arch(db).must_post_hard_capped(titan_id="T3") is True
