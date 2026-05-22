"""
Tests for Phase C Session 5 (rFP §4.E) G-RPC enforcement gates.

Verifies the AST analyzer in scripts/arch_map.py phase-c verify
correctly detects:
  - G-RPC-1: sync bus.request outside phase_c_rpc_exemptions.yaml
  - G-RPC-2: bus.request* without explicit timeout
  - G-RPC-3: proxy `def get_*` not reading SHM (with helper-pattern
    recognition for self._r_*, self._read_*, self.get_* delegation)
  - G-RPC-4: orphan `if action == "get_*"` handler (caller graph)

Run: ``python -m pytest tests/test_g_rpc_gates.py -v -p no:anchorpy``
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH_MAP = REPO_ROOT / "scripts" / "arch_map.py"


def _run_verify(*extra_args) -> tuple[int, str, str]:
    """Run arch_map phase-c verify with given extra args."""
    result = subprocess.run(
        [sys.executable, str(ARCH_MAP), "phase-c", "verify", *extra_args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def test_phase_c_verify_clean_baseline():
    """Sessions 1-5 should leave the codebase G-RPC clean."""
    rc, out, err = _run_verify("--strict")
    assert rc == 0, f"phase-c verify --strict failed:\nSTDOUT={out}\nSTDERR={err}"
    assert "phase-c verify clean" in out


def test_g_rpc_1_only():
    """--gate=G-RPC-1 runs only the sync-bus.request gate."""
    rc, out, err = _run_verify("--gate=G-RPC-1", "--strict")
    assert rc == 0, f"G-RPC-1 should be clean post-Session-5:\n{out}\n{err}"


def test_g_rpc_2_only():
    """--gate=G-RPC-2 runs only the timeout-kwarg gate."""
    rc, out, err = _run_verify("--gate=G-RPC-2", "--strict")
    assert rc == 0, f"G-RPC-2 should be clean post-Session-5:\n{out}\n{err}"


def test_g_rpc_3_only():
    """--gate=G-RPC-3 runs only the proxy SHM-read gate."""
    rc, out, err = _run_verify("--gate=G-RPC-3", "--strict")
    assert rc == 0, f"G-RPC-3 should be clean post-Session-5:\n{out}\n{err}"


def test_g_rpc_4_only():
    """--gate=G-RPC-4 runs only the orphan-handler gate."""
    rc, out, err = _run_verify("--gate=G-RPC-4", "--strict")
    assert rc == 0, f"G-RPC-4 should be clean post-Session-5:\n{out}\n{err}"


def test_exemptions_yaml_parseable():
    """phase_c_rpc_exemptions.yaml must be valid YAML."""
    import yaml
    p = REPO_ROOT / "titan-docs" / "phase_c_rpc_exemptions.yaml"
    assert p.exists(), "exemptions YAML missing"
    with p.open() as f:
        data = yaml.safe_load(f)
    assert "work_rpc_sites" in data
    assert "boot_init_sites" in data
    assert "orphan_handler_allowlist" in data
    # Each work_rpc_sites entry must have file + rationale.
    for entry in data["work_rpc_sites"]:
        assert "file" in entry, f"entry missing 'file': {entry}"
        assert "rationale" in entry, f"entry missing 'rationale': {entry}"
    # Each orphan_handler_allowlist entry must have action + rationale.
    for entry in data["orphan_handler_allowlist"]:
        assert "action" in entry
        assert "rationale" in entry


def test_18_shm_slots_in_spec():
    """Verify all 18 Sessions 1-4 SHM slots appear in SPEC §7.1."""
    spec = REPO_ROOT / "titan-docs" / "SPEC_titan_architecture.md"
    text = spec.read_text(encoding="utf-8")
    expected_slots = [
        "hormone_fires.bin", "impulse_engine_state.bin",
        "consciousness_state.bin", "resonance_state.bin",
        "unified_spirit_metadata.bin", "memory_state.bin",
        "agency_state.bin", "assessment_state.bin",
        "output_verifier_state.bin", "reflex_state.bin",
        "recorder_state.bin", "social_perception_state.bin",
        "timechain_state.bin", "mind_state.bin", "body_state.bin",
        "language_state.bin", "events_teacher_state.bin",
        "spirit_supplemental_state.bin",
    ]
    for slot in expected_slots:
        assert f"`{slot}`" in text, f"slot {slot} missing from SPEC §7.1"


def test_constants_toml_has_36_session_constants():
    """36 SCHEMA_VERSION + MAX_BYTES constants for the 18 slots."""
    toml = REPO_ROOT / "titan-docs" / "SPEC_titan_architecture_constants.toml"
    text = toml.read_text(encoding="utf-8")
    slot_prefixes = [
        "HORMONE_FIRES", "IMPULSE_ENGINE_STATE", "CONSCIOUSNESS_STATE",
        "RESONANCE_STATE", "UNIFIED_SPIRIT_METADATA", "MEMORY_STATE",
        "AGENCY_STATE", "ASSESSMENT_STATE", "OUTPUT_VERIFIER_STATE",
        "REFLEX_STATE", "RL_STATE", "SOCIAL_PERCEPTION_STATE",
        "TIMECHAIN_STATE", "MIND_STATE", "BODY_STATE", "LANGUAGE_STATE",
        "EVENTS_TEACHER_STATE", "SPIRIT_SUPPLEMENTAL_STATE",
    ]
    for prefix in slot_prefixes:
        assert f"[constants.{prefix}_SCHEMA_VERSION]" in text, \
            f"missing {prefix}_SCHEMA_VERSION"
        assert f"[constants.{prefix}_MAX_BYTES]" in text, \
            f"missing {prefix}_MAX_BYTES"


def test_no_orphan_migrated_handlers():
    """Sessions 1-4 migrated handlers MUST NOT remain in their owning
    worker code. Handlers are scoped per-file because the same action
    name (e.g. `get_status`) may exist in multiple workers — only the
    Session 1-4 migrated ones are retired."""
    # Map of relative path → set of retired handler actions.
    retired_per_file: dict[str, set[str]] = {
        "titan_hcl/modules/spirit_loop.py": {
            "get_tensor", "get_trinity", "get_consciousness",
            "get_filter_down", "get_intuition", "get_impulse_engine",
            "get_sphere_clock", "get_resonance", "get_unified_spirit",
            "get_filter_down_status", "get_meditation_health",
            "get_coordinator", "get_nervous_system",
            "get_social_perception_stats",
        },
        "titan_hcl/modules/mind_worker.py": {
            "get_tensor", "get_mood", "get_valence", "get_current_reward",
        },
        "titan_hcl/modules/body_worker.py": {
            "get_tensor", "get_status", "get_details",
        },
        "titan_hcl/modules/memory_worker.py": {"growth_metrics"},
        "titan_hcl/modules/agency_worker.py": {
            "agency_stats", "assessment_stats",
        },
        "titan_hcl/modules/language_worker.py": {"get_language_stats"},
    }
    for rel, retired_handlers in retired_per_file.items():
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for action in retired_handlers:
            pattern = f'action == "{action}"'
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if pattern in line:
                    pytest.fail(
                        f"retired handler `{pattern}` still present at "
                        f"{rel}:{lineno} — Session 5 §4.D should have "
                        f"deleted it"
                    )
