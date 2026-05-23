"""Test the bug-2 fix: plugin._social_x_gateway_reader binding.

The /v4/community-engagement-stats endpoint + kernel_rpc EXPOSED_METHODS
both referenced `plugin._social_x_gateway_reader` (and its
.get_community_engagement_stats method) but the attribute was never bound
in TitanHCL.__init__. Result: every dashboard call returned
`AttributeError: '_social_x_gateway_reader.get_community_engagement_stats'
not resolvable on plugin`.

Fix: __init__ sets `_social_x_gateway_reader = None` (defensive default),
boot() constructs a SocialXGateway instance and binds it.
"""
from __future__ import annotations

import inspect

import pytest


def test_titan_hcl_init_binds_social_x_gateway_reader_attribute():
    """__init__ must set _social_x_gateway_reader to SOMETHING (even None)
    so that getattr returns None instead of raising AttributeError."""
    from titan_hcl.core.plugin import TitanHCL
    # Inspect the __init__ source — it must assign the attribute.
    src = inspect.getsource(TitanHCL.__init__)
    assert "_social_x_gateway_reader" in src, (
        "TitanHCL.__init__ must initialize self._social_x_gateway_reader "
        "(see bug-2 fix 2026-05-23)")


def test_boot_binds_social_x_gateway_reader_instance():
    """boot() must construct a SocialXGateway instance and bind it as
    plugin._social_x_gateway_reader so the /v4/community-engagement-stats
    endpoint returns real data on T1."""
    from titan_hcl.core.plugin import TitanHCL
    src = inspect.getsource(TitanHCL.boot)
    assert "SocialXGateway" in src, (
        "TitanHCL.boot must construct SocialXGateway and bind it as "
        "_social_x_gateway_reader")
    assert "_social_x_gateway_reader = SocialXGateway" in src or \
           "self._social_x_gateway_reader = SocialXGateway" in src, (
        "TitanHCL.boot must assign SocialXGateway() to "
        "self._social_x_gateway_reader")


def test_social_x_gateway_has_get_community_engagement_stats():
    """The actual method must exist on SocialXGateway so the kernel_rpc
    EXPOSED_METHODS allowlist + dashboard endpoint resolve correctly."""
    from titan_hcl.logic.social_x_gateway import SocialXGateway
    assert hasattr(SocialXGateway, "get_community_engagement_stats"), (
        "SocialXGateway must expose get_community_engagement_stats so the "
        "kernel_rpc EXPOSED_METHODS allowlist + dashboard endpoint resolve")


def test_kernel_rpc_exposed_methods_includes_social_x_reader():
    """The kernel_rpc allowlist must include the attr + method paths so
    the api subprocess can read them under api_process_separation."""
    import titan_hcl.core.kernel as kernel_mod
    # EXPOSED_METHODS is a frozenset/set defined at module level — read source.
    src = inspect.getsource(kernel_mod)
    assert '"_social_x_gateway_reader"' in src
    assert '"_social_x_gateway_reader.get_community_engagement_stats"' in src


def test_social_x_gateway_instantiates_without_args():
    """SocialXGateway() — bare construction — must succeed (it has defaults
    for db_path, config_path, telemetry_path). The boot wiring assumes this."""
    from titan_hcl.logic.social_x_gateway import SocialXGateway
    sxg = SocialXGateway()
    assert hasattr(sxg, "get_community_engagement_stats")
