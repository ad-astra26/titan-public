"""
test_life_force_state_shm_reader — coverage for LifeForceShmReader hot-path.

§4.G chunk G5. Validates the SHM-direct sub-µs reader used by cognitive_worker
at 5 hot-path sites (MSL static_context, reasoning body_state, hormonal_pressure
inputs, ground_up_enricher chi_overlay, NN modulation cap).

Pattern mirrors tests/test_metabolism_worker_extraction.py Category 1
(SHM reader). Uses synthetic titan_ids to avoid colliding with live SHM
on hosts where a real Titan is running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import msgpack
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Cold-boot fallbacks ───────────────────────────────────────────────


def test_cold_boot_get_chi_total_returns_default():
    """No producer running → 0.5 default (matches LifeForceEngine BOOTSTRAP)."""
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_1")
    assert r.get_chi_total() == 0.5


def test_cold_boot_get_metabolic_drain_returns_zero():
    """No producer running → drain 0.0 (fresh, no accumulated pressure)."""
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_2")
    assert r.get_metabolic_drain() == 0.0


def test_cold_boot_get_chi_state_returns_empty_dict():
    """No producer → empty dict (consumer can defensively check len)."""
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_3")
    assert r.get_chi_state() == {}


def test_cold_boot_get_state_returns_bootstrap():
    """No producer → BOOTSTRAP sentinel."""
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_4")
    assert r.get_state() == "BOOTSTRAP"


def test_cold_boot_get_developmental_phase_returns_birth():
    """No producer → BIRTH default (matches LifeForceEngine BIRTH<50 logic)."""
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_5")
    assert r.get_developmental_phase() == "BIRTH"


def test_cold_boot_get_circulation_returns_zero():
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_6")
    assert r.get_circulation() == 0.0


def test_cold_boot_is_dreaming_returns_false():
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
    r = LifeForceShmReader(titan_id="T_LIFE_FORCE_COLD_BOOT_7")
    assert r.is_dreaming() is False


# ── Healthy-path roundtrip via LifeForceStatePublisher ────────────────


def _publish_and_read(titan_id: str, chi_result: dict, is_dreaming: bool = False):
    """Publish a chi_result via LifeForceStatePublisher + immediately read
    via LifeForceShmReader on the same SHM root."""
    from titan_hcl.logic.life_force_state_publisher import (
        LifeForceStatePublisher,
    )
    from titan_hcl.proxies.life_force_proxy import LifeForceShmReader

    pub = LifeForceStatePublisher(titan_id=titan_id)
    pub.publish(life_force_engine=None, chi_result=chi_result, is_dreaming=is_dreaming)
    rdr = LifeForceShmReader(titan_id=titan_id)
    return rdr


def test_healthy_path_get_chi_total_reads_published_value():
    chi = {
        "total": 0.72,
        "spirit": {"raw": 0.7, "effective": 0.65, "weight": 0.4,
                   "thinking": 0.7, "feeling": 0.7, "willing": 0.7,
                   "components": {}},
        "mind": {"raw": 0.7, "effective": 0.7, "weight": 0.35,
                 "thinking": 0.7, "feeling": 0.7, "willing": 0.7,
                 "components": {}},
        "body": {"raw": 0.7, "effective": 0.65, "weight": 0.25,
                 "thinking": 0.7, "feeling": 0.7, "willing": 0.7,
                 "components": {}},
        "circulation": 0.1,
        "weights": {"spirit": 0.4, "mind": 0.35, "body": 0.25},
        "state": "HEALTHY",
        "developmental_phase": "YOUTH",
        "contemplation": {"active": False, "phase": 0, "conviction": 0,
                          "conviction_threshold": 300, "mature_enough": False},
    }
    r = _publish_and_read("T_LIFE_FORCE_HEALTHY_1", chi)
    assert r.get_chi_total() == pytest.approx(0.72)


def test_healthy_path_get_state_reads_published_value():
    chi = {
        "total": 0.45, "spirit": {}, "mind": {}, "body": {},
        "circulation": 0.0, "weights": {},
        "state": "CONSERVING", "developmental_phase": "MATURE",
        "contemplation": {},
    }
    r = _publish_and_read("T_LIFE_FORCE_HEALTHY_2", chi)
    assert r.get_state() == "CONSERVING"
    assert r.get_developmental_phase() == "MATURE"


def test_healthy_path_is_dreaming_reads_published_flag():
    chi = {
        "total": 0.5, "spirit": {}, "mind": {}, "body": {},
        "circulation": 0.0, "weights": {},
        "state": "HEALTHY", "developmental_phase": "YOUTH",
        "contemplation": {},
    }
    r = _publish_and_read("T_LIFE_FORCE_HEALTHY_3", chi, is_dreaming=True)
    assert r.is_dreaming() is True


def test_healthy_path_get_chi_state_returns_full_payload():
    chi = {
        "total": 0.6, "spirit": {"raw": 0.6}, "mind": {"raw": 0.6}, "body": {"raw": 0.6},
        "circulation": 0.05, "weights": {"spirit": 0.4, "mind": 0.35, "body": 0.25},
        "state": "HEALTHY", "developmental_phase": "YOUTH",
        "contemplation": {"active": False, "phase": 0},
    }
    r = _publish_and_read("T_LIFE_FORCE_HEALTHY_4", chi)
    full = r.get_chi_state()
    assert isinstance(full, dict)
    assert full.get("total") == pytest.approx(0.6)
    assert full.get("state") == "HEALTHY"
    # Verify schema_version + ts get attached by publisher
    assert full.get("schema_version") == 1
    assert isinstance(full.get("ts"), float) and full.get("ts") > 0
