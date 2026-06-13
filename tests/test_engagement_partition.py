"""Fleet X-Engagement Coordination — deterministic author-partition tests.

RFP_fleet_x_engagement_coordination (2026-06-13). The three Titans share ONE X
account (@your_x_handle); a stable sha256 author-hash assigns each author to exactly
one owning Titan so the account engages any human ≤1x/24h by one Titan, with ZERO
cross-box coordination. These tests pin the load-bearing properties:
  * deterministic + STABLE across processes (sha256, NOT builtin hash())
  * every roster member can own (distribution)
  * is_my_engagement_partition true only for the owner; disabled = no-op;
    empty roster = fail-closed (safety)
"""
import subprocess
import sys

import pytest

from titan_hcl.logic.social_x.archetypes.base import (
    ArchetypeBase, engagement_owner_for, normalize_handle,
)


class _FakeGateway:
    def __init__(self, cfg):
        self._cfg = cfg
    def _load_config(self):
        return self._cfg


def _arch(cfg, db="/tmp/_part_test.db"):
    return ArchetypeBase(gateway=_FakeGateway(cfg), social_x_db_path=db)


ROSTER = ["T1", "T2", "T3"]


class TestPureOwnerFunction:
    def test_deterministic_and_stable(self):
        """Same author → same owner, every call (no per-process variance)."""
        for a in ["jkacrpto", "kirkworkssllc", "EliSan57364554", "OWarren37613"]:
            owners = {engagement_owner_for(a, ROSTER) for _ in range(50)}
            assert len(owners) == 1, f"{a} non-deterministic: {owners}"

    def test_normalization(self):
        """@Handle / spacing / case all map to the same owner."""
        base = engagement_owner_for("jkacrpto", ROSTER)
        for variant in ["@jkacrpto", "  jkacrpto ", "JKACRPTO", "@JkAcRpTo"]:
            assert engagement_owner_for(variant, ROSTER) == base
        assert normalize_handle("@Foo ") == "foo"

    def test_every_roster_member_can_own(self):
        """Over many authors, all 3 Titans appear as owners (no dead third)."""
        owners = {engagement_owner_for(f"user_{i}", ROSTER) for i in range(300)}
        assert owners == set(ROSTER), f"distribution gap: {owners}"

    def test_stable_across_processes_sha256_not_builtin_hash(self):
        """The owner MUST be identical in a FRESH process (different
        PYTHONHASHSEED). builtin hash() would differ here; sha256 does not.
        This is the cross-box-determinism guarantee (G8)."""
        code = (
            "from titan_hcl.logic.social_x.archetypes.base import "
            "engagement_owner_for as o;"
            "print(o('jkacrpto', ['T1','T2','T3']))"
        )
        out1 = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONHASHSEED": "0"}).stdout.strip()
        out2 = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONHASHSEED": "12345"}).stdout.strip()
        assert out1 and out1 == out2, f"non-stable across hashseed: {out1!r} {out2!r}"
        assert out1 == engagement_owner_for("jkacrpto", ROSTER)

    def test_empty_inputs(self):
        assert engagement_owner_for("", ROSTER) == ""
        assert engagement_owner_for("x", []) == ""


class TestIsMyPartition:
    def test_owner_true_others_false(self):
        cfg = {"engagement_fleet": ROSTER, "engagement_partition_enabled": True}
        arch = _arch(cfg)
        owner = engagement_owner_for("jkacrpto", ROSTER)
        assert arch.is_my_engagement_partition("jkacrpto", owner) is True
        for t in [t for t in ROSTER if t != owner]:
            assert arch.is_my_engagement_partition("jkacrpto", t) is False

    def test_disabled_is_noop_true(self):
        """Rollback flag: partitioning off → always eligible (legacy behavior)."""
        arch = _arch({"engagement_fleet": ROSTER,
                      "engagement_partition_enabled": False})
        # even a non-owner returns True when disabled
        for t in ROSTER:
            assert arch.is_my_engagement_partition("jkacrpto", t) is True

    def test_empty_roster_fail_closed(self):
        """Misconfigured (empty) roster → fail-CLOSED (don't let the whole
        fleet engage everything). Safety > availability."""
        arch = _arch({"engagement_fleet": [],
                      "engagement_partition_enabled": True})
        assert arch.is_my_engagement_partition("jkacrpto", "T1") is False

    def test_roster_from_config(self):
        """engagement_roster reads the config list."""
        arch = _arch({"engagement_fleet": ["A", "B"],
                      "engagement_partition_enabled": True})
        assert arch.engagement_roster() == ("A", "B")
        owner = engagement_owner_for("someone", ["A", "B"])
        assert arch.is_my_engagement_partition("someone", owner) is True
