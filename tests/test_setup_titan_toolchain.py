"""Phase A unit tests — scripts/setup_titan/toolchain.py.

Offline / $0 / no installs: exercises the PINS drift-guard, the flag-override
resolver (INV-PROV-3), and the version-aware presence probe (INV-PROV-1) with
the actual install commands mocked. Phase D's test_setup_titan_provision.py
covers the provision loop that consumes these.
"""
from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.setup_titan import toolchain as tc


# ── PINS drift-guard (these literals ARE gate G3) ────────────────────────────

def test_pins_are_the_t1_verified_set():
    assert tc.PINS == {
        "rust": "stable",
        "solana": "3.1.10",
        "anchor": "0.32.1",
        "node": "22",
    }


def test_executable_map_and_musl_constant():
    assert tc.EXECUTABLE == {
        "rust": "cargo", "solana": "solana", "anchor": "anchor", "node": "node",
    }
    assert tc.MUSL_TARGET == "x86_64-unknown-linux-musl"
    assert tc.RUST_COMPONENTS == ("rustfmt", "clippy")


# ── resolve_versions (INV-PROV-3) ────────────────────────────────────────────

def test_resolve_none_returns_default_copy():
    out = tc.resolve_versions(None)
    assert out == tc.PINS
    assert out is not tc.PINS          # a copy, never the module dict


def test_resolve_namespace_without_flags_is_defensive():
    # Before Phase B wires the flags, the Namespace has no *_version attrs.
    assert tc.resolve_versions(Namespace()) == tc.PINS


def test_resolve_applies_overrides():
    args = Namespace(solana_version="3.2.0", node_version="24",
                     rust_version=None, anchor_version="")
    out = tc.resolve_versions(args)
    assert out["solana"] == "3.2.0"    # overridden
    assert out["node"] == "24"         # overridden
    assert out["rust"] == "stable"     # None → untouched
    assert out["anchor"] == "0.32.1"   # "" (falsy) → untouched


def test_resolve_accepts_mapping_and_strips():
    out = tc.resolve_versions({"anchor_version": "  0.33.0  "})
    assert out["anchor"] == "0.33.0"


def test_summary_orders_and_joins():
    assert tc.summary(tc.PINS) == "rust=stable · solana=3.1.10 · anchor=0.32.1 · node=22"


# ── _extract_version (pure parsing) ──────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("solana-cli 3.1.10 (src:devbuild; feat:..., client:Agave)", "3.1.10"),
    ("anchor-cli 0.32.1", "0.32.1"),
    ("v22.22.2", "22.22.2"),
    ("rustc 1.95.0 (abc123 2026-01-01)", "1.95.0"),
    ("cargo 1.95.0 (abc 2026-01-01)", "1.95.0"),
    ("no version here", None),
    ("", None),
])
def test_extract_version(raw, expect):
    assert tc._extract_version(raw) == expect


# ── _satisfies (token-shape comparison) ──────────────────────────────────────

@pytest.mark.parametrize("pinned,found,ok", [
    ("3.1.10", "3.1.10", True),    # exact semver — match
    ("3.1.10", "3.1.11", False),   # exact semver — patch differs
    ("3.1.10", "3.1", False),      # found is shorter than the pin
    ("0.32.1", "0.32.1", True),
    ("22", "22.22.2", True),       # bare major — node, minor floats
    ("22", "20.1.0", False),       # bare major — mismatch
    ("3.1", "3.1.10", True),       # major.minor — prefix match
    ("3.1", "3.2.0", False),
    ("3.1.10", None, False),       # nothing detected
])
def test_satisfies(pinned, found, ok):
    assert tc._satisfies(pinned, found) is ok


# ── tool_present_at (idempotency probe, INV-PROV-1) ──────────────────────────

def _patch_probe(monkeypatch, on_path: set[str], versions: dict[str, str]):
    """Wire a fake host: `on_path` = executables that exist; `versions` maps an
    executable name → its `--version` raw output."""
    monkeypatch.setattr(tc.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name in on_path else None)
    monkeypatch.setattr(tc, "_probe_version",
                        lambda exe, version_arg="--version": versions.get(exe))


def test_present_solana_exact_match(monkeypatch):
    _patch_probe(monkeypatch, {"solana"}, {"solana": "solana-cli 3.1.10 (Agave)"})
    st = tc.tool_present_at("solana", "3.1.10")
    assert st.present and st.compatible and st.found == "3.1.10"
    assert st.needs_install is False


def test_present_solana_version_mismatch(monkeypatch):
    _patch_probe(monkeypatch, {"solana"}, {"solana": "solana-cli 3.1.9 (Agave)"})
    st = tc.tool_present_at("solana", "3.1.10")
    assert st.present and not st.compatible and st.needs_install


def test_absent_solana(monkeypatch):
    _patch_probe(monkeypatch, set(), {})
    st = tc.tool_present_at("solana", "3.1.10")
    assert not st.present and not st.compatible and st.found is None


def test_present_node_major_match(monkeypatch):
    _patch_probe(monkeypatch, {"node"}, {"node": "v22.22.2"})
    st = tc.tool_present_at("node", "22")
    assert st.compatible and st.found == "22.22.2"


def test_present_node_major_mismatch(monkeypatch):
    _patch_probe(monkeypatch, {"node"}, {"node": "v20.11.0"})
    assert not tc.tool_present_at("node", "22").compatible


def test_present_anchor_exact(monkeypatch):
    _patch_probe(monkeypatch, {"anchor"}, {"anchor": "anchor-cli 0.32.1"})
    assert tc.tool_present_at("anchor", "0.32.1").compatible


def test_rust_channel_needs_cargo_and_musl(monkeypatch):
    # cargo present, rustc reports a version, musl target installed → compatible.
    _patch_probe(monkeypatch, {"cargo", "rustc"}, {"rustc": "rustc 1.95.0 (x)"})
    monkeypatch.setattr(tc, "_musl_target_installed", lambda: True)
    st = tc.tool_present_at("rust", "stable")
    assert st.present and st.compatible and st.found == "1.95.0"


def test_rust_channel_without_musl_is_incompatible(monkeypatch):
    _patch_probe(monkeypatch, {"cargo", "rustc"}, {"rustc": "rustc 1.95.0 (x)"})
    monkeypatch.setattr(tc, "_musl_target_installed", lambda: False)
    st = tc.tool_present_at("rust", "stable")
    assert st.present and not st.compatible and "musl" in st.detail


def test_rust_channel_absent_cargo(monkeypatch):
    _patch_probe(monkeypatch, set(), {})
    monkeypatch.setattr(tc, "_musl_target_installed", lambda: None)
    st = tc.tool_present_at("rust", "stable")
    assert not st.present and not st.compatible


def test_rust_numeric_override_is_exact_semver(monkeypatch):
    # A numeric --rust-version override leaves channel mode → exact rustc compare.
    _patch_probe(monkeypatch, {"cargo", "rustc"}, {"rustc": "rustc 1.95.0 (x)"})
    assert tc.tool_present_at("rust", "1.95.0").compatible
    _patch_probe(monkeypatch, {"cargo", "rustc"}, {"rustc": "rustc 1.94.0 (x)"})
    assert not tc.tool_present_at("rust", "1.95.0").compatible


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        tc.tool_present_at("golang", "1.22")
