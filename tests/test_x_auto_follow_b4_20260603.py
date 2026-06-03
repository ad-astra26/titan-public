"""rFP X-post PART B4 — organic auto-follow (INV-XENG-4) tests (2026-06-03).

Covers SocialXGateway.follow() (the sanctioned follow-write path, gated +
daily-cap backstop) and AutoFollowPolicy (recurring high-relevance selection,
metered). All API calls mocked — no network.

Run: python -m pytest tests/test_x_auto_follow_b4_20260603.py -v -p no:anchorpy
"""
from __future__ import annotations

import sqlite3
import time
import types

import pytest


def _ctx(titan_id="T1"):
    return types.SimpleNamespace(session="s", proxy="p", api_key="k", titan_id=titan_id)


def _gateway(tmp_path, *, enabled=True, max_per_day=2):
    from titan_hcl.logic.social_x_gateway import SocialXGateway
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[social_x]\nenabled = true\nuser_name = "your_x_handle"\n'
        f"[social_x.auto_follow]\nenabled = {'true' if enabled else 'false'}\n"
        f"max_per_day = {max_per_day}\nmin_recurrence = 3\nmin_relevance = 0.8\n"
        "window_days = 7\n"
    )
    gw = SocialXGateway(db_path=str(tmp_path / "x.db"), config_path=str(cfg),
                        telemetry_path=str(tmp_path / "t.jsonl"))
    gw._boot_time = 0
    return gw


# ── gateway.follow() ─────────────────────────────────────────────────────

def test_follow_disabled_when_flag_off(tmp_path):
    gw = _gateway(tmp_path, enabled=False)
    assert gw.follow("123", _ctx(), handle="alice").status == "disabled"


def test_follow_posts_when_enabled(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, enabled=True)
    monkeypatch.setattr(gw, "_call_x_api", lambda *a, **k: {"status": "success"})
    res = gw.follow("999", _ctx(), handle="kirk")
    assert res.status == "posted"
    c = sqlite3.connect(gw._db_path)
    n = c.execute("SELECT count(*) FROM actions WHERE action_type='follow' "
                  "AND status='posted'").fetchone()[0]
    c.close()
    assert n == 1


def test_follow_daily_cap_backstop(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, enabled=True, max_per_day=1)
    monkeypatch.setattr(gw, "_call_x_api", lambda *a, **k: {"status": "success"})
    assert gw.follow("1", _ctx(), handle="a").status == "posted"
    assert gw.follow("2", _ctx(), handle="b").status == "rate_limited"


# ── AutoFollowPolicy (2026-06-03 redesign: candidates from captured author_id) ──

def _policy_dbs(tmp_path):
    sx = str(tmp_path / "sx.db"); et = str(tmp_path / "et.db"); sg = str(tmp_path / "sg.db")
    now = time.time()
    c = sqlite3.connect(sx)
    c.execute("CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "action_type TEXT, status TEXT, titan_id TEXT, metadata TEXT, created_at REAL)")
    c.commit(); c.close()
    ce = sqlite3.connect(et)
    ce.execute("CREATE TABLE felt_experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "titan_id TEXT, author TEXT, relevance REAL, created_at REAL, "
               "author_id TEXT DEFAULT '', author_followers INTEGER DEFAULT 0)")
    # kirk: 2× @0.9, uid captured → ELIGIBLE.
    # lowguy: 4× @0.6 → below relevance.
    # noid: 3× @0.9 but NO captured user_id → unfollowable (the dry-pool bug fix).
    # followedguy: 2× @0.9, uid captured, but already-following → excluded.
    for author, rel, cnt, aid, fol in [
            ("kirk", 0.9, 2, "999", 1500),
            ("lowguy", 0.6, 4, "888", 10),
            ("noid", 0.9, 3, "", 0),
            ("followedguy", 0.9, 2, "555", 900)]:
        for _ in range(cnt):
            ce.execute("INSERT INTO felt_experiences (titan_id, author, relevance, "
                       "created_at, author_id, author_followers) VALUES ('T1',?,?,?,?,?)",
                       (author, rel, now - 3600, aid, fol))
    ce.commit(); ce.close()
    cg = sqlite3.connect(sg)
    cg.execute("CREATE TABLE community_registry (user_name TEXT PRIMARY KEY, user_id TEXT, is_following INTEGER)")
    cg.execute("INSERT INTO community_registry VALUES ('followedguy', '555', 1)")  # already following
    cg.commit(); cg.close()
    return sx, et, sg


class _FakeGateway:
    def __init__(self, sx):
        self._db_path = sx
        self.calls = []

    def follow(self, user_id, context, consumer="", *, handle="", source_id=""):
        self.calls.append((user_id, handle))
        c = sqlite3.connect(self._db_path)
        c.execute("INSERT INTO actions (action_type, status, titan_id, metadata, created_at) "
                  "VALUES ('follow', 'posted', 'T1', ?, ?)",
                  (f'{{"handle":"{handle}"}}', time.time()))
        c.commit(); c.close()
        return types.SimpleNamespace(status="posted")


def _cfg(enabled=True, max_per_day=2, min_recurrence=1):
    return {"user_name": "your_x_handle", "self_handles": ["iamtitantech"],
            "auto_follow": {"enabled": enabled, "min_recurrence": min_recurrence,
                            "min_relevance": 0.8, "window_days": 7,
                            "max_per_day": max_per_day}}


def test_policy_follows_discovered_voice_with_captured_uid(tmp_path):
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    n = pol.run(titan_id="T1", context=_ctx(), config=_cfg())
    assert n == 1
    # kirk follows (rel + uid); lowguy (low rel), noid (no uid), followedguy
    # (already following) all excluded — uses the CAPTURED author_id, not registry.
    assert gw.calls == [("999", "kirk")]


def test_policy_skips_author_without_captured_uid(tmp_path):
    """The dry-pool fix: a high-relevance recurring author with NO captured
    user_id is never followed (old design's blocker)."""
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    pol.run(titan_id="T1", context=_ctx(), config=_cfg())
    assert "noid" not in [h for _, h in gw.calls]


def test_policy_excludes_already_following(tmp_path):
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    pol.run(titan_id="T1", context=_ctx(), config=_cfg(max_per_day=5))
    assert "followedguy" not in [h for _, h in gw.calls]


def test_policy_recurrence_one_admits_single_encounter(tmp_path):
    """recurrence≥1 (the new default) lets a single high-relevance encounter
    qualify — the wide discovery pool the old ≥3 gate excluded."""
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    ce = sqlite3.connect(et)
    ce.execute("INSERT INTO felt_experiences (titan_id, author, relevance, created_at, "
               "author_id, author_followers) VALUES ('T1','solo',0.9,?, '321', 200)",
               (time.time() - 3600,))
    ce.commit(); ce.close()
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    pol.run(titan_id="T1", context=_ctx(), config=_cfg(max_per_day=5, min_recurrence=1))
    assert "solo" in [h for _, h in gw.calls]   # seen once, still followable


def test_policy_noop_when_disabled(tmp_path):
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    assert pol.run(titan_id="T1", context=_ctx(), config=_cfg(enabled=False)) == 0
    assert gw.calls == []


def test_policy_respects_daily_cap(tmp_path):
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    # bump lowguy's relevance so 2 eligible candidates (kirk + lowguy) exist
    ce = sqlite3.connect(et)
    ce.execute("UPDATE felt_experiences SET relevance=0.9 WHERE author='lowguy'")
    ce.commit(); ce.close()
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    n = pol.run(titan_id="T1", context=_ctx(), config=_cfg(max_per_day=1))
    assert n == 1                          # capped at 1 even though 2 qualify
    assert len(gw.calls) == 1
