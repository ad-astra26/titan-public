"""rFP X-post PART B — engagement conversion: B1 (self-handle exclusion) + B2
(cooldown 7d→48h) tests (2026-06-03).

B1/INV-XENG-1: Titan's OWN handles are permanently on the outer-engagement
cooldown set, so the shared `if author in cooldown` check in all four outer
archetypes (world_mirror / outer_inner_bridge / outer_rumination / amplify)
skips them — Titan never mirrors/replies/amplifies its own account.

B2/INV-XENG-2: the per-author cooldown is 48h (was 7 days; 7d starved the
~5-author engagement pool to ≈0).

Run: python -m pytest tests/test_x_engagement_selfhandle_cooldown_20260603.py -v -p no:anchorpy
"""
from __future__ import annotations

import pytest


@pytest.fixture
def base_archetype(tmp_path):
    from titan_hcl.logic.social_x_gateway import SocialXGateway
    from titan_hcl.logic.social_x.archetypes.base import ArchetypeBase
    from titan_hcl.logic.social_x.schema_migrations import apply_social_x_migrations

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[social_x]\nenabled = true\nuser_name = "your_x_handle"\n'
        'self_handles = ["iamtitantech"]\n'
    )
    db = str(tmp_path / "x.db")
    gw = SocialXGateway(db_path=db, config_path=str(cfg),
                        telemetry_path=str(tmp_path / "t.jsonl"))
    gw._boot_time = 0
    apply_social_x_migrations(db)  # ensure actions table exists (empty)
    return ArchetypeBase(gateway=gw, social_x_db_path=db)


def test_b2_cooldown_is_24h():
    # Owned-author re-engage floor was 7d → 48h → 24h (Maker 2026-06-13,
    # RFP_fleet_x_engagement_coordination INV-FX-2): under the deterministic
    # author-hash partition a person is engaged by ≤1 Titan, so 24h bounds the
    # shared @your_x_handle account fleet-wide.
    from titan_hcl.logic.social_x.archetypes.base import DEFAULT_AUTHOR_COOLDOWN_S
    assert DEFAULT_AUTHOR_COOLDOWN_S == 24 * 3600


def test_b1_self_handles_from_config(base_archetype):
    handles = base_archetype._self_handles()
    assert "your_x_handle" in handles      # from user_name
    assert "iamtitantech" in handles    # from self_handles list


def test_b1_self_handles_seed_cooldown_set(base_archetype):
    """With an empty actions table, the cooldown set is exactly the self-handles
    — so the outer archetypes' `if author in cooldown` always skips our own account."""
    cd = base_archetype.authors_on_cooldown(titan_id="T1")
    assert "your_x_handle" in cd
    assert "iamtitantech" in cd


def test_b1_author_on_cooldown_true_for_self(base_archetype):
    assert base_archetype.author_on_cooldown("iamtitantech", titan_id="T1") is True
    assert base_archetype.author_on_cooldown("your_x_handle", titan_id="T1") is True
    # a normal external account with no prior engagement is NOT on cooldown
    assert base_archetype.author_on_cooldown("some_random_user", titan_id="T1") is False
