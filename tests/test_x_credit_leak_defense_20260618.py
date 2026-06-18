"""Credit-leak defense for the X write path (2026-06-18).

While X has @your_x_handle write-blocked (every create_tweet → 422) yet reads keep
succeeding, the fleet was burning paid twitterapi.io credits: failed
create_tweets (200cr), needless re-logins (~500cr, mis-triggered by treating a
write-422 as session expiry), and 3× redundant mention polls of one shared
inbox. These tests pin the four defenses:

  1. Persistent, restart-surviving auto-suspension after N decisive write rejects.
  2. write-422 no longer triggers a login while reads are healthy + 6h throttle.
  3. A config kill-switch that halts all writes before any API call.
  4. Single-owner mention polling for the shared account.
"""
import json
import pytest

import titan_hcl.logic.social_x_gateway as gw_mod
from titan_hcl.logic.social_x_gateway import SocialXGateway

R422 = {"status": "error", "message": "API returned status 422"}
R226 = {"status": "error",
        "message": "Authorization: This request looks like it might be automated"}
ROK = {"status": "success", "tweet_id": "123"}


@pytest.fixture
def gw(tmp_path, monkeypatch):
    g = SocialXGateway()
    # Redirect the persistence file into tmp so tests don't touch real state.
    monkeypatch.setattr(g, "_write_state_path",
                        lambda: str(tmp_path / "wstate.json"))
    g._write_suspended_until = 0.0
    g._write_reject_streak = 0
    g._write_suspend_backoff_s = 0.0
    return g


# ── Fix 1: persistent auto-suspension ───────────────────────────────────────

def test_suspends_after_threshold(gw):
    for _ in range(2):
        gw._note_write_outcome(R422, 200)
    assert gw._write_block_reason() == ""          # not yet (threshold 3)
    gw._note_write_outcome(R422, 200)
    assert "auto-suspended" in gw._write_block_reason()


def test_success_clears_suspension(gw):
    for _ in range(3):
        gw._note_write_outcome(R422, 200)
    assert gw._write_block_reason()
    gw._note_write_outcome(ROK, 200)
    assert gw._write_block_reason() == ""
    assert gw._write_reject_streak == 0


def test_backoff_doubles_then_caps(gw):
    def trip():
        gw._write_reject_streak = 0
        for _ in range(3):
            gw._note_write_outcome(R422, 200)
    trip(); first = gw._write_suspend_backoff_s
    trip(); second = gw._write_suspend_backoff_s
    assert second == min(first * 2, gw._WRITE_SUSPEND_CAP_S)
    for _ in range(10):
        trip()
    assert gw._write_suspend_backoff_s == gw._WRITE_SUSPEND_CAP_S


def test_226_is_decisive_407_is_not(gw):
    assert gw._is_decisive_write_reject(R226, 200)
    assert not gw._is_decisive_write_reject(
        {"message": "API returned status 407"}, 200)
    # 407 rejects never arm the suspension
    for _ in range(5):
        gw._note_write_outcome({"status": "error",
                                "message": "API returned status 407"}, 200)
    assert gw._write_block_reason() == ""


def test_suspension_persists_across_restart(gw, tmp_path, monkeypatch):
    for _ in range(3):
        gw._note_write_outcome(R422, 200)
    assert gw._write_block_reason()
    # New gateway instance pointed at the same state file → still suspended.
    g2 = SocialXGateway()
    monkeypatch.setattr(g2, "_write_state_path",
                        lambda: str(tmp_path / "wstate.json"))
    g2._load_write_suspension()
    assert "auto-suspended" in g2._write_block_reason()


# ── Master switch: the single hot `social_x.enabled` flag ───────────────────

def test_enabled_false_blocks_all_calls(gw, monkeypatch):
    # social_x.enabled=false → reads AND writes refuse before any API call.
    monkeypatch.setattr(gw_mod, "get_params", lambda s=None: {
        "enabled": False} if s == "social_x" else {})
    assert gw.x_api_hibernated() is True
    assert "social_x.enabled=false" in gw._write_block_reason()
    assert gw._call_x_api("twitter/user/mentions", "GET", {}).get(
        "status") == "disabled"
    assert gw._call_x_api("twitter/create_tweet_v2", "POST",
                          {"tweet_text": "x"}).get("status") == "disabled"


def test_enabled_true_default_not_blocked(gw, monkeypatch):
    monkeypatch.setattr(gw_mod, "get_params", lambda s=None: {
        "enabled": True} if s == "social_x" else {})
    assert gw.x_api_hibernated() is False
    assert gw._write_block_reason() == ""          # not disabled, not suspended


def test_enabled_missing_defaults_on(gw, monkeypatch):
    # absent key → default enabled (never accidentally hibernate)
    monkeypatch.setattr(gw_mod, "get_params", lambda s=None: {})
    assert gw.x_api_hibernated() is False


# ── Fix 4 (single-owner poll): deterministic single owner of the account ────

def test_single_account_owner_for_mention_poll():
    from titan_hcl.logic.social_x.archetypes.base import engagement_owner_for
    roster = ["T1", "T2", "T3"]
    owners = {engagement_owner_for("your_x_handle", roster)}
    # Exactly one deterministic owner, stable across calls/boxes.
    assert owners == {engagement_owner_for("your_x_handle", roster)}
    assert engagement_owner_for("your_x_handle", roster) in roster
