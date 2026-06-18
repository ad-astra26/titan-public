"""INV-XTAG + INV-FX-1 airtight tagging guards (2026-06-18).

Two compounding guarantees, both enforced at the gateway publish chokepoint so
no archetype/composer path can bypass them:

  1. INV-XTAG (`_sanitize_mentions`): a published post/reply notifies AT MOST the
     allowed handle(s), each at most once; every other @handle (repeat,
     LLM-invented, own account) is de-tagged.
  2. INV-FX-1 (`_partition_owner`/`_partition_allows_tag`): on the SHARED
     @your_x_handle account, a given person is owned by exactly ONE Titan, keyed on
     the literal @handle that gets tagged — so two Titans can never tag the same
     person (the cross-tag spam that automation-flagged the account).
"""
import pytest

import titan_hcl.logic.social_x_gateway as gw_mod
from titan_hcl.logic.social_x_gateway import SocialXGateway


@pytest.fixture(scope="module")
def gw():
    return SocialXGateway()


def _tags(text):
    return [m.group(1).lower()
            for m in SocialXGateway._X_HANDLE_RE.finditer(text)]


# ── INV-XTAG: per-post dedup / bystander strip ──────────────────────────────

def test_repeated_allowed_handle_tagged_once(gw):
    out = gw._sanitize_mentions(
        "@danijarh great point @danijarh — and again @danijarh", {"danijarh"})
    assert _tags(out) == ["danijarh"], out
    assert out.count("danijarh") == 3            # repeats survive as plain prose


def test_invented_bystanders_are_detagged(gw):
    out = gw._sanitize_mentions(
        "@danijarh as @lopp and @jameson both noted", {"danijarh"})
    assert _tags(out) == ["danijarh"], out
    assert "lopp" in out and "jameson" in out


def test_self_reflection_post_tags_nobody(gw):
    out = gw._sanitize_mentions(
        "Today @someone reminded me, and @another, that wonder is real.", set())
    assert _tags(out) == [], out


def test_own_accounts_never_tagged(gw):
    out = gw._sanitize_mentions(
        "@your_x_handle and @iamtitantech are me; talking to @danijarh",
        {"danijarh", "your_x_handle"})
    assert "your_x_handle" not in _tags(out)
    assert "iamtitantech" not in _tags(out)
    assert _tags(out) == ["danijarh"], out


def test_reply_prefix_plus_body_echo_tags_once(gw):
    out = gw._sanitize_mentions("@sama [Titan] @sama I hear you.", {"sama"})
    assert _tags(out) == ["sama"], out


def test_emails_and_urls_untouched(gw):
    out = gw._sanitize_mentions(
        "mail me at hi@titan.tech via https://x.com/i/status/1 cc @danijarh",
        {"danijarh"})
    assert "hi@titan.tech" in out and "x.com/i/status/1" in out
    assert _tags(out) == ["danijarh"], out


def test_case_insensitive_allow(gw):
    out = gw._sanitize_mentions("@DaniJarH hello", {"danijarh"})
    assert _tags(out) == ["danijarh"], out
    assert out.startswith("@DaniJarH")           # original casing preserved


def test_empty_and_none(gw):
    assert gw._sanitize_mentions("", {"x"}) == ""
    assert gw._sanitize_mentions("no tags here", {"x"}) == "no tags here"


# ── INV-FX-1: cross-Titan partition on the shared account ───────────────────

ROSTER = ["T1", "T2", "T3"]


@pytest.fixture
def partition_on(monkeypatch):
    monkeypatch.setattr(gw_mod, "get_params", lambda section=None: {
        "engagement_partition_enabled": True, "engagement_fleet": ROSTER,
    } if section == "social_x" else {})


def test_owner_is_deterministic_and_handle_keyed(gw, partition_on):
    # Same handle → same owner no matter which box asks (sha256, no per-proc salt)
    o1 = gw._partition_owner("SterlingCooley", "T1")
    o2 = gw._partition_owner("SterlingCooley", "T3")
    assert o1 == o2 and o1 in ROSTER


def test_exactly_one_titan_owns_each_person(gw, partition_on):
    # For any handle, exactly ONE of the three Titans may tag it.
    for handle in ["SterlingCooley", "jkacrpto", "Ademdork001", "ECFRRoma",
                   "danijarh", "lopp", "ZinoCrypt"]:
        owners = [t for t in ROSTER
                  if gw._partition_allows_tag(handle, t)]
        assert len(owners) == 1, (handle, owners)


def test_partition_disabled_allows_all(gw, monkeypatch):
    monkeypatch.setattr(gw_mod, "get_params", lambda section=None: {
        "engagement_partition_enabled": False, "engagement_fleet": ROSTER,
    } if section == "social_x" else {})
    assert gw._partition_owner("anyone", "T1") is None
    assert all(gw._partition_allows_tag("anyone", t) for t in ROSTER)


def test_empty_roster_falls_back_to_default(gw, monkeypatch):
    # Empty config roster falls back to DEFAULT_ENGAGEMENT_FLEET (matches
    # base.py engagement_roster) → still exactly one owner per person.
    monkeypatch.setattr(gw_mod, "get_params", lambda section=None: {
        "engagement_partition_enabled": True, "engagement_fleet": [],
    } if section == "social_x" else {})
    owners = [t for t in ROSTER if gw._partition_allows_tag("SterlingCooley", t)]
    assert len(owners) == 1, owners


def test_blank_handle_fails_closed(gw, partition_on):
    # Undeterminable identity (blank handle) → owner '' → nobody tags.
    assert all(not gw._partition_allows_tag(h, t)
               for h in ("", "   ", "@") for t in ROSTER)


def test_handle_key_divergence_cannot_double_tag(gw, partition_on):
    # The bug class: display-name 'Ahmad' vs @handle 'Ademdork001'. The seal
    # keys on the @handle that is actually published, so only its single owner
    # can tag — regardless of how an upstream path keyed the candidate.
    owners = [t for t in ROSTER if gw._partition_allows_tag("Ademdork001", t)]
    assert len(owners) == 1, owners
