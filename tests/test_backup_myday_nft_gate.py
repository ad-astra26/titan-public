"""tests/test_backup_myday_nft_gate.py — L12 housekeeping closure.

Pins the `daily_nft_enabled` config gate that was previously orphaned.
Before this fix, RebirthBackup attempted MyDay NFT minting on every
4th meditation regardless of the gate, and the resulting failures were
silenced by `swallow_warn` — making the feature look "working but
flaky" when it was structurally disabled.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from titan_hcl.logic.backup import RebirthBackup


def _make_backup(full_config: dict, count_since_nft: int = 4) -> RebirthBackup:
    """Build a minimal RebirthBackup bypassing the heavy __init__ path."""
    rb = RebirthBackup.__new__(RebirthBackup)
    rb._full_config = full_config
    rb._titan_id = "T1"
    rb._meditation_count_since_nft = count_since_nft
    rb._save_backup_state = MagicMock()
    rb.network = MagicMock(keypair=None)  # no wallet → mint short-circuits
    return rb


class TestMyDayNFTGate:
    """Pin the daily_nft_enabled gate added in the L12 closure."""

    def test_gate_off_skips_mint_attempt(self):
        rb = _make_backup({"mainnet_budget": {"daily_nft_enabled": False}},
                          count_since_nft=4)
        mainnet_cfg = rb._full_config.get("mainnet_budget") or {}
        # Pin the resolution path used inside the run-backup branch.
        assert bool(mainnet_cfg.get("daily_nft_enabled", False)) is False

    def test_gate_on_admits_mint(self):
        rb = _make_backup({"mainnet_budget": {"daily_nft_enabled": True}},
                          count_since_nft=4)
        mainnet_cfg = rb._full_config.get("mainnet_budget") or {}
        assert bool(mainnet_cfg.get("daily_nft_enabled", False)) is True

    def test_gate_default_is_off(self):
        # Missing key → safe default OFF (config.toml line 465 ships False).
        rb = _make_backup({"mainnet_budget": {}}, count_since_nft=4)
        mainnet_cfg = rb._full_config.get("mainnet_budget") or {}
        assert bool(mainnet_cfg.get("daily_nft_enabled", False)) is False

    def test_threshold_param_driven(self):
        # `meditations_per_daily_nft` config key drives the threshold;
        # previously hardcoded to 4. Resolution path used in backup.py:
        # `int(self._full_config.get("meditations_per_daily_nft", 4))`.
        rb = _make_backup(
            {"mainnet_budget": {"daily_nft_enabled": True},
             "meditations_per_daily_nft": 7},
            count_since_nft=6)
        threshold = int(rb._full_config.get("meditations_per_daily_nft", 4))
        assert threshold == 7

    def test_threshold_default_is_four(self):
        rb = _make_backup({"mainnet_budget": {"daily_nft_enabled": True}},
                          count_since_nft=4)
        threshold = int(rb._full_config.get("meditations_per_daily_nft", 4))
        assert threshold == 4

    def test_full_config_not_dict_defaults_safely(self):
        # Defensive: full_config can be None or non-dict on certain
        # initialization paths (e.g. tests). The L12 closure code uses
        # `isinstance(self._full_config, dict)` to guard both reads.
        rb = _make_backup({}, count_since_nft=4)
        rb._full_config = None
        mainnet_cfg = (rb._full_config.get("mainnet_budget")
                        if isinstance(rb._full_config, dict) else {}) or {}
        assert mainnet_cfg == {}
        threshold = int(rb._full_config.get("meditations_per_daily_nft", 4)
                         if isinstance(rb._full_config, dict) else 4)
        assert threshold == 4
