"""tests/test_life_force_worker_spec_parity.py — SPEC v1.8.4 / D-SPEC-58 parity.

Per rFP_titan_hcl_l2_separation_strategy.md §4.G. Static assertions
that the SPEC documents what the code implements (and vice versa) for
the life_force_worker extraction.

Covers per §21 D-SPEC-58 acceptance criteria:
  1. LifeForceShmReader cold-boot defaults + healthy roundtrip (covered
     in test_life_force_state_shm_reader.py — 11 tests).
  2. LifeForceProxy surface parity vs LifeForceEngine read-only surface.
  3. SPEC document drift (Changelog / glossary / §7.1 / §8.7 / §9.B / D-SPEC-58).
  4. Wiring drift (plugin.py / legacy_core.py / kernel.py / phase_c_rpc_
     exemptions.yaml / bus.py / constants TOML / cache_key_registry.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


# ═════════════════════════════════════════════════════════════════════
# Category 1 — SPEC document drift
# ═════════════════════════════════════════════════════════════════════


def _spec_text() -> str:
    return (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()


def test_spec_version_bumped_to_1_8_3():
    """v1.8.4 was a milestone version when life_force_worker §4.G shipped.
    SPEC moves forward; assertion is that v1.8.4 row remains documented in
    the SPEC (Changelog rolling-cap may eventually retire it to the Archived
    section — at which point this test should be migrated to read the
    archive too)."""
    spec = _spec_text()
    assert "v1.8.4 (PATCH)" in spec, (
        "v1.8.4 Changelog row must remain present (life_force_worker §4.G)")


def test_spec_changelog_v1_8_3_row_present():
    spec = _spec_text()
    assert "v1.8.4 (PATCH)" in spec, "Changelog must have v1.8.4 row"
    assert "life_force_worker" in spec
    assert "§4.G" in spec
    assert "D-SPEC-58" in spec


def test_spec_section_1_glossary_row_present():
    spec = _spec_text()
    assert "| **life_force_worker**" in spec, (
        "§1 glossary must have life_force_worker row")


def test_spec_section_7_1_life_force_state_slot_row_present():
    spec = _spec_text()
    assert "| `life_force_state.bin`" in spec
    assert "LIFE_FORCE_STATE_SCHEMA_VERSION" in spec
    assert "LIFE_FORCE_STATE_MAX_BYTES = 4096" in spec


def test_spec_section_7_1_life_force_inputs_slot_row_present():
    spec = _spec_text()
    assert "| `life_force_inputs.bin`" in spec
    assert "LIFE_FORCE_INPUTS_SCHEMA_VERSION" in spec
    assert "LIFE_FORCE_INPUTS_MAX_BYTES = 1024" in spec


def test_spec_section_8_7_life_force_updated_event_row_present():
    spec = _spec_text()
    assert "| `LIFE_FORCE_UPDATED`" in spec


def test_spec_section_8_7_fatigue_event_row_present():
    spec = _spec_text()
    assert "| `FATIGUE_LEVEL_CRITICAL`" in spec


def test_spec_section_9_b_life_force_worker_block_present():
    spec = _spec_text()
    assert "#### life_force_worker (Python L2 module" in spec, (
        "§9.B must have life_force_worker block")


def test_spec_d_spec_57_entry_present():
    spec = _spec_text()
    assert "D-SPEC-58" in spec
    # Verify the full entry — not just a passing mention
    assert "Closes the Track 1 drift" in spec


# ═════════════════════════════════════════════════════════════════════
# Category 2 — Constants TOML drift
# ═════════════════════════════════════════════════════════════════════


def test_constants_toml_has_life_force_state_constants():
    toml = (ROOT / "titan-docs" /
            "SPEC_titan_architecture_constants.toml").read_text()
    assert "LIFE_FORCE_STATE_SCHEMA_VERSION" in toml
    assert "LIFE_FORCE_STATE_MAX_BYTES" in toml
    assert 'introduced_in = "1.8.4"' in toml


def test_constants_toml_has_life_force_inputs_constants():
    toml = (ROOT / "titan-docs" /
            "SPEC_titan_architecture_constants.toml").read_text()
    assert "LIFE_FORCE_INPUTS_SCHEMA_VERSION" in toml
    assert "LIFE_FORCE_INPUTS_MAX_BYTES" in toml


def test_constants_toml_has_fatigue_threshold_and_reset():
    toml = (ROOT / "titan-docs" /
            "SPEC_titan_architecture_constants.toml").read_text()
    assert "LIFE_FORCE_FATIGUE_THRESHOLD" in toml
    assert "LIFE_FORCE_FATIGUE_RESET" in toml
    assert "LIFE_FORCE_MEDITATION_RECOVERY_FACTOR" in toml


def test_constants_toml_spec_version_bumped():
    """Constants TOML spec_version moves forward; only assert that it is
    at or above 1.8.4 when life_force_worker §4.G constants landed."""
    toml = (ROOT / "titan-docs" /
            "SPEC_titan_architecture_constants.toml").read_text()
    import re
    m = re.search(r'^spec_version\s*=\s*"(\d+\.\d+\.\d+)"', toml, re.MULTILINE)
    assert m is not None, "constants TOML must declare spec_version"
    parts = tuple(int(x) for x in m.group(1).split("."))
    assert parts >= (1, 8, 4), (
        f"constants TOML spec_version {m.group(1)} must be ≥ 1.8.4 (§4.G)")


# ═════════════════════════════════════════════════════════════════════
# Category 3 — Wiring drift
# ═════════════════════════════════════════════════════════════════════


def test_kernel_proxy_aliases_includes_life_force_proxy():
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "life_force_proxy" in KERNEL_PROXY_ALIASES, (
        "life_force_proxy missing from KERNEL_PROXY_ALIASES — "
        "RESPONSE messages won't route back to the proxy reply queue.")


def test_bus_has_two_new_event_constants():
    from titan_hcl import bus
    assert bus.LIFE_FORCE_UPDATED == "LIFE_FORCE_UPDATED"
    assert bus.FATIGUE_LEVEL_CRITICAL == "FATIGUE_LEVEL_CRITICAL"


def test_plugin_has_wire_life_force_method():
    from titan_hcl.core.plugin import TitanHCL
    assert hasattr(TitanHCL, "_wire_life_force"), (
        "TitanHCL._wire_life_force missing — proxy install path broken")


def test_phase_c_rpc_exemptions_allowlists_life_force_proxy():
    yaml_text = (ROOT / "titan-docs" /
                 "phase_c_rpc_exemptions.yaml").read_text()
    assert "life_force_proxy.py:get_stats" in yaml_text
    assert "life_force_proxy.py:get_chi_history" in yaml_text
    assert "life_force_proxy.py:get_contemplation_status" in yaml_text


# NOTE: cache_key_registry was RETIRED in Phase D D-SPEC-80. chi.state is
# now read SHM-direct from life_force_state.bin (Phase A §4.C.3 +
# life_force_worker single-writer per G21) — the producer_module field
# no longer exists in a registry. Producer authority is asserted via
# constants TOML + §7.1 SPEC row.


# ═════════════════════════════════════════════════════════════════════
# Category 4 — LifeForceProxy surface
# ═════════════════════════════════════════════════════════════════════


def test_proxy_exposes_hot_path_reads():
    from titan_hcl.proxies.life_force_proxy import LifeForceProxy
    expected_hot_reads = {
        "get_chi_total", "get_metabolic_drain", "get_chi_state",
        "get_state", "get_developmental_phase", "get_circulation",
        "is_dreaming",
    }
    proxy_methods = {name for name in dir(LifeForceProxy) if not name.startswith("_")}
    missing = expected_hot_reads - proxy_methods
    assert not missing, f"LifeForceProxy missing hot-path methods: {missing}"


def test_proxy_exposes_work_rpc_methods():
    from titan_hcl.proxies.life_force_proxy import LifeForceProxy
    expected_rpc = {"get_stats", "get_chi_history", "get_contemplation_status"}
    proxy_methods = {name for name in dir(LifeForceProxy) if not name.startswith("_")}
    missing = expected_rpc - proxy_methods
    assert not missing, f"LifeForceProxy missing work-RPC methods: {missing}"
