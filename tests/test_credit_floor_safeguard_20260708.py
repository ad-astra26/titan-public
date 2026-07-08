"""FX.5.D — credit-floor safeguard (RFP_social_x §5.FX.5.D).

Below the hibernate floor: set the gateway's runtime _credit_hibernated flag (every
box → fleet stops spending) + the canonical poller alerts the Maker. Auto-un-hibernates
when credits recover. Prevents the silent run-to-zero that blacked out the fleet ~23h.
"""
import types

import titan_hcl.logic.social_worker_post_dispatch as pd
from titan_hcl.logic.social_worker_post_dispatch import PostDispatchOrchestrator


class _GW:
    def __init__(self):
        self._credit_hibernated = False


class _Self:
    _check_credit_floor = PostDispatchOrchestrator._check_credit_floor

    def __init__(self, canonical=True):
        self._gateway = _GW()
        self._is_canonical_poller = canonical
        self._last_credit_check_ts = 0.0
        self.alerts = []

    def _maker_alert(self, text, key, rate_limit_s):
        self.alerts.append(key)


def _patch(monkeypatch, balance, sx_extra=None):
    sx = {"credit_check_interval_seconds": 900,
          "credit_hibernate_floor": 2000, "credit_alert_floor": 20000}
    if sx_extra:
        sx.update(sx_extra)
    monkeypatch.setattr(pd, "get_params", lambda s: sx if s == "social_x"
                        else ({"twitterapi_io_key": "k"} if s == "stealth_sage" else {}))
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: types.SimpleNamespace(
        json=lambda: {"recharge_credits": balance}))


def test_below_hibernate_floor_hibernates_and_alerts(monkeypatch):
    _patch(monkeypatch, 1500)
    s = _Self(canonical=True)
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is True
    assert "social_x.credit_hibernate" in s.alerts


def test_low_but_above_hibernate_warns_only(monkeypatch):
    _patch(monkeypatch, 10_000)
    s = _Self(canonical=True)
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is False
    assert s.alerts == ["social_x.credit_low"]


def test_healthy_balance_no_action(monkeypatch):
    _patch(monkeypatch, 500_000)
    s = _Self(canonical=True)
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is False
    assert s.alerts == []


def test_recovery_unhibernates(monkeypatch):
    _patch(monkeypatch, 500_000)
    s = _Self(canonical=True)
    s._gateway._credit_hibernated = True  # was hibernated
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is False  # recovered → resumed


def test_non_canonical_hibernates_but_does_not_alert(monkeypatch):
    _patch(monkeypatch, 1500)
    s = _Self(canonical=False)
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is True  # every box stops spending
    assert s.alerts == []                         # only canonical alerts


def test_interval_throttle_skips_recheck(monkeypatch):
    _patch(monkeypatch, 1500)
    s = _Self(canonical=True)
    s._last_credit_check_ts = 9_950.0            # 50s ago < 900s interval
    s._check_credit_floor(10_000.0)
    assert s._gateway._credit_hibernated is False  # skipped, no check
    assert s.alerts == []


def test_balance_fetch_error_is_safe(monkeypatch):
    _patch(monkeypatch, 1500)
    import httpx
    def _boom(*a, **k):
        raise RuntimeError("network blip")
    monkeypatch.setattr(httpx, "get", _boom)
    s = _Self(canonical=True)
    s._check_credit_floor(10_000.0)  # must not raise
    assert s._gateway._credit_hibernated is False  # fail-safe: no false hibernate
