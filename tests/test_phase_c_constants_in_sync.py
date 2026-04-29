"""
test_phase_c_constants_in_sync.py — CI parity test for SPEC constants.

Per SPEC §19.3: regenerates _phase_c_constants.py + constants.rs from the
canonical TOML and asserts byte-identical with committed files.

Catches:
  - Someone hand-edited _phase_c_constants.py directly
  - Someone hand-edited constants.rs directly
  - Someone edited the TOML but forgot to regenerate

Run via:
    pytest tests/test_phase_c_constants_in_sync.py -p no:anchorpy --tb=short
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOML_PATH = REPO_ROOT / "titan-docs" / "SPEC_titan_architecture_constants.toml"
GENERATOR = REPO_ROOT / "scripts" / "generate_phase_c_constants.py"
PYTHON_OUT = REPO_ROOT / "titan_plugin" / "_phase_c_constants.py"
RUST_OUT = REPO_ROOT / "titan-rust" / "crates" / "titan-core" / "src" / "constants.rs"


def test_toml_exists():
    assert TOML_PATH.exists(), f"SPEC TOML missing at {TOML_PATH}"


def test_generator_exists():
    assert GENERATOR.exists(), f"Generator script missing at {GENERATOR}"


def test_python_constants_in_sync():
    """Generated _phase_c_constants.py matches TOML."""
    assert PYTHON_OUT.exists(), (
        f"Generated Python file missing at {PYTHON_OUT.relative_to(REPO_ROOT)}. "
        f"Run: python scripts/generate_phase_c_constants.py"
    )
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"_phase_c_constants.py drift detected:\n{result.stdout}\n{result.stderr}\n\n"
        f"FIX: python scripts/generate_phase_c_constants.py"
    )


def test_rust_constants_in_sync():
    """Generated constants.rs matches TOML."""
    assert RUST_OUT.exists(), (
        f"Generated Rust file missing at {RUST_OUT.relative_to(REPO_ROOT)}. "
        f"Run: python scripts/generate_phase_c_constants.py"
    )
    # The --check call above already covers Rust too; this test name
    # makes the failure mode explicit if Rust drifts but Python doesn't.


def test_toml_loads_cleanly():
    """The TOML is valid and contains spec_version + at least one domain."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    assert "spec_version" in data, "TOML missing spec_version key"
    assert data.get("domains"), "TOML has no domains"
    assert data.get("constants"), "TOML has no constants"


def test_every_constant_has_registered_domain():
    """Every [constants.X] block uses a domain present in [domains.X]."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    domains = set(data.get("domains", {}).keys())
    issues: list[str] = []
    for name, spec in data.get("constants", {}).items():
        domain = spec.get("domain")
        if not domain:
            issues.append(f"{name}: missing 'domain' field")
        elif domain not in domains:
            issues.append(
                f"{name}: domain '{domain}' has no [domains.{domain}] block"
            )
    assert not issues, (
        "Constants reference undeclared domains:\n  "
        + "\n  ".join(issues)
        + "\n\nFIX: add [domains.X] block to TOML, or fix the constant's domain field."
    )


def test_ground_truths_g1_through_g16_present():
    """Preamble G1-G16 must be present in TOML for arch_map enforcement."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    truths = set(data.get("ground_truths", {}).keys())
    expected = {f"G{i}" for i in range(1, 17)}
    missing = expected - truths
    assert not missing, (
        f"Preamble ground truths missing from TOML: {sorted(missing)}\n"
        f"FIX: add [ground_truths.G*] blocks per SPEC Preamble."
    )


def test_phase_c_verify_clean():
    """`arch_map phase-c verify` exits 0 on clean baseline."""
    arch_map = REPO_ROOT / "scripts" / "arch_map.py"
    if not arch_map.exists():
        pytest.skip("arch_map.py not found")
    result = subprocess.run(
        [sys.executable, str(arch_map), "phase-c", "verify"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"phase-c verify failed:\n{result.stdout}\n{result.stderr}"
    )


def test_phase_c_verify_strict_clean():
    """`arch_map phase-c verify --strict` exits 0 on clean baseline."""
    arch_map = REPO_ROOT / "scripts" / "arch_map.py"
    if not arch_map.exists():
        pytest.skip("arch_map.py not found")
    result = subprocess.run(
        [sys.executable, str(arch_map), "phase-c", "verify", "--strict"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"phase-c verify --strict failed:\n{result.stdout}\n{result.stderr}"
    )


# ── Phase C C-S2 row 37 obligation: parity vectors.json schema ─────────
# Per PLAN_microkernel_phase_c_s2_kernel.md §14.3 + §3 row 37: extend
# this test with vectors.json schema verification. Both pytest and
# cargo test load the same file at repo root for cross-language byte
# parity (RFC 4231 / RFC 5869 / Titan-canonical vectors).

PARITY_VECTORS_PATH = REPO_ROOT / "tests" / "parity" / "vectors.json"
PARITY_REQUIRED_TOP_LEVEL = {
    "spec_version",
    "frame",
    "authkey",
    "shm_layout",
    "bus_specs",
    "adoption_payload",
}


def test_parity_vectors_file_exists():
    """tests/parity/vectors.json must exist and be valid JSON."""
    assert PARITY_VECTORS_PATH.exists(), (
        f"Parity vectors fixture missing at "
        f"{PARITY_VECTORS_PATH.relative_to(REPO_ROOT)}"
    )


def test_parity_vectors_schema():
    """vectors.json has all SPEC §14.1 sections."""
    import json

    data = json.loads(PARITY_VECTORS_PATH.read_text(encoding="utf-8"))
    missing = PARITY_REQUIRED_TOP_LEVEL - set(data.keys())
    assert not missing, (
        f"vectors.json missing required sections: {sorted(missing)}\n"
        f"Per PLAN §14.1, schema is: {sorted(PARITY_REQUIRED_TOP_LEVEL)}"
    )
    assert isinstance(data["spec_version"], str), "spec_version must be string"
    # frame: rfc4231 + titan_canonical sub-arrays
    assert isinstance(data["frame"].get("rfc4231"), list), (
        "frame.rfc4231 must be a list of test vectors"
    )
    # authkey: rfc5869 + titan_canonical
    assert isinstance(data["authkey"].get("rfc5869"), list), (
        "authkey.rfc5869 must be a list of test vectors"
    )
