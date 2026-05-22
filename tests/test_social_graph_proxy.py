"""tests/test_social_graph_proxy.py — SocialGraphProxy contract tests.

Per PLAN_microkernel_phase_c_social_graph_worker_extraction.md §7.2 +
SPEC v1.7.1 §9.B social_graph_worker + D-SPEC-50.

The flagship test in this file is
test_proxy_exposes_record_interaction_async — the regression gate for
BUG-MINDPROXY-MISSING-RECORD-INTERACTION-ASYNC-20260514 (the production
AttributeError that motivated this extraction).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan_hcl.proxies.social_graph_proxy import (  # noqa: E402
    SocialGraphProxy,
    _DictProfile,
    _DONATION_TIERS,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _build_proxy_with_mocks():
    """Construct SocialGraphProxy with mocked bus + guardian + SHM reader."""
    bus = MagicMock()
    guardian = MagicMock()
    # bus.subscribe returns a reply_queue (mock)
    bus.subscribe.return_value = MagicMock()
    # Patch StateRegistryReader so __init__ doesn't try to open /dev/shm
    with patch(
        "titan_hcl.proxies.social_graph_proxy.StateRegistryReader"
    ) as mock_reader_cls:
        reader_mock = MagicMock()
        mock_reader_cls.return_value = reader_mock
        with patch(
            "titan_hcl.proxies.social_graph_proxy.ensure_shm_root"
        ):
            with patch(
                "titan_hcl.proxies.social_graph_proxy.resolve_titan_id",
                return_value="T1",
            ):
                proxy = SocialGraphProxy(bus, guardian)
                proxy._reader_mock = reader_mock  # noqa — test handle
                return proxy, bus, guardian


# ── THE REGRESSION GATE ───────────────────────────────────────────────


def test_proxy_exposes_record_interaction_async():
    """BUG-MINDPROXY-MISSING-RECORD-INTERACTION-ASYNC-20260514 regression gate.

    Pre-fix: `_proxies["social_graph"]` aliased to MindProxy, which had
    NO `record_interaction_async` method — every chat post-hook fired
    `AttributeError: 'MindProxy' object has no attribute
    'record_interaction_async'` fleet-wide, silently breaking
    KnownUserResolver familiarity scoring.

    Post-fix: SocialGraphProxy has `record_interaction_async`.
    """
    proxy, _, _ = _build_proxy_with_mocks()
    assert hasattr(proxy, "record_interaction_async"), (
        "BUG-MINDPROXY-MISSING-RECORD-INTERACTION-ASYNC-20260514 regressed — "
        "SocialGraphProxy MUST expose record_interaction_async per rFP §4.P + "
        "D-SPEC-50 (v1.7.1)")
    assert callable(getattr(proxy, "record_interaction_async"))
    # And asyncio.iscoroutinefunction check
    assert asyncio.iscoroutinefunction(proxy.record_interaction_async)


# ── Full async surface coverage ───────────────────────────────────────


@pytest.mark.parametrize("method_name", [
    "record_interaction_async",
    "get_or_create_user_async",
    "_save_profile_async",
    "should_engage_async",
    "record_edge_async",
    "record_donation_async",
    "record_inspiration_async",
    "get_stats_async",
    "get_top_users_async",
    "ledger_record_async",
    "ledger_has_tweet_async",
    "ledger_user_reply_count_async",
    "ledger_last_reply_to_user_async",
    "ledger_total_today_async",
    "ledger_cleanup_async",
])
def test_proxy_async_method_exists(method_name):
    """Every SocialGraph public *_async method has a proxy sibling.

    Closes the class of bug where the proxy is missing an async method
    that callers expect (the original failure class).
    """
    proxy, _, _ = _build_proxy_with_mocks()
    assert hasattr(proxy, method_name), (
        f"SocialGraphProxy missing required async method: {method_name}")
    assert asyncio.iscoroutinefunction(getattr(proxy, method_name)), (
        f"{method_name} must be a coroutine function")


# ── Pure compute path (G19-exempt) ────────────────────────────────────


def test_proxy_get_donation_mood_boost_is_pure_compute():
    """get_donation_mood_boost is pure compute — no IO, no bus."""
    proxy, bus, _ = _build_proxy_with_mocks()
    # 0.10 SOL → tier 0 (mood_delta=0.10, weight=5.0)
    mood, weight = proxy.get_donation_mood_boost(0.10)
    assert mood == 0.10 and weight == 5.0
    # 0.06 SOL → tier 1 (mood_delta=0.05, weight=3.0)
    mood, weight = proxy.get_donation_mood_boost(0.06)
    assert mood == 0.05 and weight == 3.0
    # 0.02 SOL → tier 2
    mood, weight = proxy.get_donation_mood_boost(0.02)
    assert mood == 0.02 and weight == 2.0
    # 0.00 SOL → tier 3
    mood, weight = proxy.get_donation_mood_boost(0.00)
    assert mood == 0.01 and weight == 1.5
    # Defensive: non-numeric input falls back to bottom tier
    mood, weight = proxy.get_donation_mood_boost("invalid")  # type: ignore
    assert mood == 0.01 and weight == 1.5
    # No bus interaction occurred
    bus.publish.assert_not_called()
    bus.request.assert_not_called()


def test_proxy_donation_tiers_match_social_graph_module():
    """Pure-compute tier table mirrors SocialGraph DONATION_TIERS."""
    from titan_hcl.core.social_graph import DONATION_TIERS
    assert _DONATION_TIERS == DONATION_TIERS


# ── SHM stats path (G18) ──────────────────────────────────────────────


def test_proxy_get_stats_reads_shm_not_bus():
    """get_stats reads social_graph_state.bin via SHM (G18); never bus."""
    proxy, bus, _ = _build_proxy_with_mocks()

    import msgpack
    fixture_payload = {
        "users": 42,
        "edges": 17,
        "donations": 3,
        "total_donated_sol": 0.25,
        "inspirations": 5,
        "engagement_ledger_today": 12,
        "schema_version": 1,
        "ts": 1234567890.0,
    }
    proxy._reader_mock.read_variable.return_value = msgpack.packb(
        fixture_payload, use_bin_type=True)

    stats = proxy.get_stats()
    assert stats["users"] == 42
    assert stats["edges"] == 17
    assert stats["donations"] == 3
    assert abs(stats["total_donated_sol"] - 0.25) < 1e-9
    assert stats["inspirations"] == 5
    # No bus interaction
    bus.publish.assert_not_called()
    bus.request.assert_not_called()


def test_proxy_get_stats_cold_boot_returns_zeros():
    """SHM slot empty → defaults, no exception."""
    proxy, _, _ = _build_proxy_with_mocks()
    proxy._reader_mock.read_variable.return_value = None
    stats = proxy.get_stats()
    assert stats == {
        "users": 0, "edges": 0, "donations": 0,
        "total_donated_sol": 0.0, "inspirations": 0,
    }


# ── Bus work-RPC routing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proxy_record_interaction_async_routes_to_worker():
    """record_interaction_async fires a bus.request_async with
    dst=social_graph and action=record_interaction (G19 work-RPC)."""
    proxy, bus, _ = _build_proxy_with_mocks()

    async def fake_request_async(src, dst, payload, **kw):
        # Capture for assertion
        fake_request_async.captured = (src, dst, payload, kw)
        return {"payload": {"ok": True}}

    fake_request_async.captured = None
    bus.request_async = fake_request_async

    # Patch _ensure_started to be a no-op
    proxy._ensure_started = MagicMock()

    await proxy.record_interaction_async("alice", quality=0.7)
    assert fake_request_async.captured is not None
    src, dst, payload, kw = fake_request_async.captured
    assert src == "social_graph_proxy"
    assert dst == "social_graph"
    assert payload["action"] == "record_interaction"
    assert payload["user_id"] == "alice"
    assert payload["quality"] == 0.7
    assert kw.get("timeout") == 5.0  # G19 ≤5s


def test_dict_profile_attribute_access():
    """_DictProfile exposes user fields as attributes + net_sentiment / is_donor."""
    p = _DictProfile({
        "user_id": "alice",
        "like_score": 0.6,
        "dislike_score": 0.2,
        "total_donated_sol": 0.05,
    })
    assert p.user_id == "alice"
    assert p.like_score == 0.6
    assert p.is_donor is True
    # net_sentiment = (0.6 - 0.2) / 0.8 = 0.5
    assert abs(p.net_sentiment - 0.5) < 1e-9


def test_dict_profile_empty_returns_none_attrs():
    p = _DictProfile({})
    assert p.user_id is None
    assert p.is_donor is False
    assert p.net_sentiment == 0.0


# ── Subscribe registration ────────────────────────────────────────────


def test_proxy_subscribes_reply_only():
    """Proxy registers under name 'social_graph_proxy' with reply_only=True
    so the broker silently skips it from dst='all' broadcasts."""
    _, bus, _ = _build_proxy_with_mocks()
    bus.subscribe.assert_called_once()
    args, kwargs = bus.subscribe.call_args
    assert args[0] == "social_graph_proxy"
    assert kwargs.get("reply_only") is True
