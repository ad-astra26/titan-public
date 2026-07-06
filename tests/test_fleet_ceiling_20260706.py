"""FX.1 — shared-timeline fleet post-budget ceiling (SPEC §23.15.2 / RFP_social_x §5.FX.1).

Coordination-free: each box counts account-wide tweets on the shared @your_x_handle
timeline before posting. Tests the pure gate logic via a lightweight fake (no DB /
no twitterapi.io) so the SOFT-ceiling counting + fail-open semantics are asserted.
"""
import time
from datetime import datetime, timezone

import titan_hcl.logic.social_x_gateway as sxg_mod
from titan_hcl.logic.social_x_gateway import SocialXGateway


def _created(age_s: float) -> str:
    """A twitterapi.io-format createdAt at now-age_s (e.g. 'Sun Jul 05 12:18:27 +0000 2026')."""
    dt = datetime.fromtimestamp(time.time() - age_s, tz=timezone.utc)
    return dt.strftime("%a %b %d %H:%M:%S +0000 %Y")


class _Ctx:
    api_key = "k"


class _FakeGW:
    """Minimal self carrying only what _check_fleet_ceiling touches."""
    def __init__(self, tweets, *, raise_read=False):
        self._tweets = tweets
        self._raise = raise_read

    def fetch_recent_tweets(self, *a, **k):
        if self._raise:
            raise RuntimeError("timeline read blip")
        return {"data": {"tweets": self._tweets}}

    # bind the real method under test
    _check_fleet_ceiling = SocialXGateway._check_fleet_ceiling


def _patch_handle(monkeypatch, handle="your_x_handle"):
    monkeypatch.setattr(
        sxg_mod, "get_params",
        lambda section: {"user_name": handle} if section == "twitter_social" else {})


def test_day_ceiling_blocks_at_cap(monkeypatch):
    _patch_handle(monkeypatch)
    # 12 tweets spread 1h..12h ago (all within 24h, only the 1h one within the hour)
    tweets = [{"createdAt": _created(3600 * i)} for i in range(1, 13)]
    gw = _FakeGW(tweets)
    r = gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 12, "fleet_max_posts_per_hour": 99}, _Ctx())
    assert r is not None and r.status == "fleet_ceiling"
    assert "12/12" in r.reason


def test_under_day_cap_passes(monkeypatch):
    _patch_handle(monkeypatch)
    tweets = [{"createdAt": _created(3600 * i)} for i in range(1, 12)]  # 11 in 24h
    gw = _FakeGW(tweets)
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 12, "fleet_max_posts_per_hour": 99}, _Ctx()) is None


def test_hour_ceiling_blocks(monkeypatch):
    _patch_handle(monkeypatch)
    # 3 tweets in the last hour (600/1200/1800s ago) → hour cap 3 blocks
    tweets = [{"createdAt": _created(s)} for s in (600, 1200, 1800)]
    gw = _FakeGW(tweets)
    r = gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 99, "fleet_max_posts_per_hour": 3}, _Ctx())
    assert r is not None and r.status == "fleet_ceiling" and "this hour" in r.reason


def test_old_tweets_not_counted(monkeypatch):
    _patch_handle(monkeypatch)
    # all tweets are >24h old → not counted → under any positive cap
    tweets = [{"createdAt": _created(86400 + 3600 * i)} for i in range(1, 15)]
    gw = _FakeGW(tweets)
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 1, "fleet_max_posts_per_hour": 1}, _Ctx()) is None


def test_bypass_caps_skips_ceiling(monkeypatch):
    _patch_handle(monkeypatch)
    tweets = [{"createdAt": _created(60 * i)} for i in range(1, 30)]  # way over
    gw = _FakeGW(tweets)
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 1}, _Ctx(), bypass_caps=True) is None


def test_disabled_when_caps_zero(monkeypatch):
    _patch_handle(monkeypatch)
    tweets = [{"createdAt": _created(60 * i)} for i in range(1, 30)]
    gw = _FakeGW(tweets)
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 0, "fleet_max_posts_per_hour": 0}, _Ctx()) is None


def test_fail_open_on_read_error(monkeypatch):
    _patch_handle(monkeypatch)
    gw = _FakeGW([], raise_read=True)  # timeline read raises
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 1}, _Ctx()) is None


def test_fail_open_when_no_handle(monkeypatch):
    _patch_handle(monkeypatch, handle="")  # no shared account handle resolvable
    tweets = [{"createdAt": _created(60 * i)} for i in range(1, 30)]
    gw = _FakeGW(tweets)
    assert gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 1}, _Ctx()) is None


def test_defaults_active_when_config_absent(monkeypatch):
    # NO fleet_* keys in config → defaults ON (12/day, 3/hr) per all-flags-on.
    _patch_handle(monkeypatch)
    tweets = [{"createdAt": _created(1800 * i)} for i in range(1, 13)]  # 12 in 24h
    gw = _FakeGW(tweets)
    r = gw._check_fleet_ceiling({}, _Ctx())  # empty config
    assert r is not None and r.status == "fleet_ceiling"


def test_unparseable_createdat_skipped_not_crashed(monkeypatch):
    _patch_handle(monkeypatch)
    tweets = [{"createdAt": "garbage"}, {"createdAt": ""}, {}] + \
             [{"createdAt": _created(3600 * i)} for i in range(1, 5)]  # 4 valid
    gw = _FakeGW(tweets)
    # 4 valid in-window, cap 4 → blocks (garbage rows skipped, no crash)
    r = gw._check_fleet_ceiling(
        {"fleet_max_posts_per_day": 4, "fleet_max_posts_per_hour": 99}, _Ctx())
    assert r is not None and r.status == "fleet_ceiling"
