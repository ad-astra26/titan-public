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


# ── AutoFollowPolicy ─────────────────────────────────────────────────────

def _policy_dbs(tmp_path):
    sx = str(tmp_path / "sx.db"); et = str(tmp_path / "et.db"); sg = str(tmp_path / "sg.db")
    now = time.time()
    c = sqlite3.connect(sx)
    c.execute("CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "action_type TEXT, status TEXT, titan_id TEXT, metadata TEXT, created_at REAL)")
    c.commit(); c.close()
    ce = sqlite3.connect(et)
    ce.execute("CREATE TABLE felt_experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "titan_id TEXT, author TEXT, relevance REAL, created_at REAL)")
    # kirk: 4× @0.9 → eligible. lowguy: 4× @0.6 → below relevance. rareguy: 1× @0.9 → below recurrence.
    for author, rel, cnt in [("kirk", 0.9, 4), ("lowguy", 0.6, 4), ("rareguy", 0.9, 1)]:
        for _ in range(cnt):
            ce.execute("INSERT INTO felt_experiences (titan_id, author, relevance, created_at) "
                       "VALUES ('T1', ?, ?, ?)", (author, rel, now - 3600))
    ce.commit(); ce.close()
    cg = sqlite3.connect(sg)
    cg.execute("CREATE TABLE community_registry (user_name TEXT PRIMARY KEY, user_id TEXT, is_following INTEGER)")
    cg.execute("INSERT INTO community_registry VALUES ('kirk', '999', 0)")       # known, not followed
    cg.execute("INSERT INTO community_registry VALUES ('lowguy', '888', 0)")
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


def _cfg(enabled=True, max_per_day=2):
    return {"user_name": "your_x_handle", "self_handles": ["iamtitantech"],
            "auto_follow": {"enabled": enabled, "min_recurrence": 3,
                            "min_relevance": 0.8, "window_days": 7,
                            "max_per_day": max_per_day}}


def test_policy_follows_only_recurring_high_relevance(tmp_path):
    from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
    sx, et, sg = _policy_dbs(tmp_path)
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    n = pol.run(titan_id="T1", context=_ctx(), config=_cfg())
    assert n == 1                          # only kirk qualifies
    assert gw.calls == [("999", "kirk")]   # lowguy (low rel) + rareguy (low recurrence) excluded


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
    # make lowguy eligible too (bump its relevance) so 2 candidates exist
    ce = sqlite3.connect(et)
    ce.execute("UPDATE felt_experiences SET relevance=0.9 WHERE author='lowguy'")
    ce.commit(); ce.close()
    gw = _FakeGateway(sx)
    pol = AutoFollowPolicy(gateway=gw, social_x_db=sx, events_teacher_db=et, social_graph_db=sg)
    n = pol.run(titan_id="T1", context=_ctx(), config=_cfg(max_per_day=1))
    assert n == 1                          # capped at 1 even though 2 qualify
    assert len(gw.calls) == 1
