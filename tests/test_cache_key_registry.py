"""
tests/test_cache_key_registry.py — drift detection for the observatory
data contract registry (rFP_observatory_data_loading_v1 Phase 1).

Runs the same audit as `arch_map cache-keys --audit` but in pytest form.
A failure here means the registry has drifted from the source tree —
either a producer was renamed, a consumer was added without registering,
or a bus constant was renamed.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def registry():
    from titan_plugin.api import cache_key_registry as ckr
    return ckr


@pytest.fixture(scope="module")
def bus_module():
    return importlib.import_module("titan_plugin.bus")


# ── Structural invariants ──────────────────────────────────────────────


def test_registry_non_empty(registry) -> None:
    assert len(registry.REGISTRY) > 0
    # Sanity-check: at least the core observatory keys are present
    keys = {s.key for s in registry.REGISTRY}
    for required in ("chi.state", "memory.status", "guardian.status", "spirit.coordinator"):
        assert required in keys, f"missing core registry entry: {required!r}"


def test_no_duplicate_keys(registry) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for s in registry.REGISTRY:
        if s.key in seen:
            dupes.append(s.key)
        seen.add(s.key)
    assert not dupes, f"duplicate cache keys in REGISTRY: {dupes}"


def test_event_to_cache_key_derived(registry) -> None:
    """EVENT_TO_CACHE_KEY must be derived purely from REGISTRY."""
    expected = {
        s.producer_event: s.key
        for s in registry.REGISTRY
        if s.producer_event is not None and s.kind in ("bus_event", "hybrid")
    }
    assert registry.EVENT_TO_CACHE_KEY == expected


def test_bus_subscriber_uses_derived_map(registry) -> None:
    """bus_subscriber.EVENT_TO_CACHE_KEY must exactly equal the registry-derived map."""
    from titan_plugin.api.bus_subscriber import EVENT_TO_CACHE_KEY as BS_MAP
    assert BS_MAP == registry.EVENT_TO_CACHE_KEY, (
        "bus_subscriber.EVENT_TO_CACHE_KEY drifted from cache_key_registry "
        "— it must be derived, not hand-maintained."
    )


# ── Producer-side drift detection ──────────────────────────────────────


def test_bus_constants_exist(registry, bus_module) -> None:
    """Every producer_event in the registry must be a constant in titan_plugin.bus."""
    missing = [
        s.producer_event
        for s in registry.REGISTRY
        if s.producer_event is not None and not hasattr(bus_module, s.producer_event)
    ]
    assert not missing, (
        f"producer_event names declared in REGISTRY but missing from "
        f"titan_plugin.bus: {missing}"
    )


def _module_to_path(mod_dotted: str) -> Path | None:
    parts = mod_dotted.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = REPO_ROOT / Path(*parts[:cut]).with_suffix(".py")
        if candidate.exists():
            return candidate
    return None


def test_bus_event_producers_exist_in_source(registry) -> None:
    """For every kind=bus_event/hybrid entry, the producer module must contain
    a publish callsite referencing the producer_event constant."""
    failures: list[str] = []
    for s in registry.REGISTRY:
        if s.kind not in ("bus_event", "hybrid"):
            continue
        prod_path = _module_to_path(s.producer_module)
        if prod_path is None:
            failures.append(f"{s.key}: producer_module file not found ({s.producer_module})")
            continue
        src = prod_path.read_text()
        ev = s.producer_event
        patterns = [
            rf"_send_msg\([^)]*\b{re.escape(ev)}\b",
            rf'_send_msg\([^)]*"{re.escape(ev)}"',
            rf"make_msg\(\s*{re.escape(ev)}\b",
            rf'make_msg\(\s*"{re.escape(ev)}"',
        ]
        if not any(re.search(p, src, re.DOTALL) for p in patterns):
            failures.append(
                f"{s.key}: no _send_msg/make_msg({ev}, ...) call found in "
                f"{prod_path.relative_to(REPO_ROOT)}"
            )
    assert not failures, "producer drift detected:\n  " + "\n  ".join(failures)


def test_snapshot_keys_written_by_kernel(registry) -> None:
    """Every kind=snapshot entry must be written by kernel._build_state_snapshot."""
    failures: list[str] = []
    for s in registry.REGISTRY:
        if s.kind != "snapshot":
            continue
        prod_path = _module_to_path(s.producer_module)
        if prod_path is None:
            failures.append(f"{s.key}: snapshot producer module not found")
            continue
        src = prod_path.read_text()
        # Direct write: snapshot["key"] = ...
        patt_direct = re.compile(rf'snapshot\[\s*[\'"]{re.escape(s.key)}[\'"]\s*\]')
        # F-string write for plugin._* loop: `snapshot[f"plugin.{attr}"]`
        if patt_direct.search(src):
            continue
        if s.key.startswith("plugin."):
            attr = s.key.split(".", 1)[1]
            if f'"{attr}"' in src and "snapshot[f" in src:
                continue
        failures.append(
            f"{s.key}: snapshot[\"{s.key}\"] not found in "
            f"{prod_path.relative_to(REPO_ROOT)}"
        )
    assert not failures, "snapshot drift detected:\n  " + "\n  ".join(failures)


# ── Consumer-side drift detection ─────────────────────────────────────


def test_no_unregistered_cache_get_calls(registry) -> None:
    """Every cache.get('X') in titan_plugin/{api,modules,core} must resolve
    to a REGISTRY entry or be in the allowlist."""
    api_dir = REPO_ROOT / "titan_plugin" / "api"
    modules_dir = REPO_ROOT / "titan_plugin" / "modules"
    core_dir = REPO_ROOT / "titan_plugin" / "core"
    skip = {api_dir / "cache_key_registry.py"}

    pat = re.compile(r'cache\.get\(\s*[\'"]([^\'"]+)[\'"]')
    found: dict[str, list[str]] = {}
    for d in (api_dir, modules_dir, core_dir):
        for py in d.rglob("*.py"):
            if py in skip:
                continue
            try:
                lines = py.read_text().splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, start=1):
                for m in pat.finditer(line):
                    k = m.group(1)
                    found.setdefault(k, []).append(
                        f"{py.relative_to(REPO_ROOT)}:{lineno}"
                    )
    unregistered = [
        f"{k}  ←  {sites[0]}"
        for k, sites in found.items()
        if k not in registry.REGISTERED_KEYS and not registry.is_allowlisted(k)
    ]
    assert not unregistered, (
        "cache.get() callsites without a registry entry "
        "(add to REGISTRY or ALLOWLIST_KEYS):\n  " + "\n  ".join(unregistered)
    )


# ── Spec validation ───────────────────────────────────────────────────


def test_spec_kind_validity(registry) -> None:
    for s in registry.REGISTRY:
        assert s.kind in registry.PRODUCER_KIND


def test_bus_event_specs_have_event(registry) -> None:
    for s in registry.REGISTRY:
        if s.kind == "bus_event":
            assert s.producer_event, f"{s.key}: kind=bus_event requires producer_event"


def test_hybrid_specs_have_event(registry) -> None:
    for s in registry.REGISTRY:
        if s.kind == "hybrid":
            assert s.producer_event, f"{s.key}: kind=hybrid requires producer_event"


def test_missing_specs_have_no_module(registry) -> None:
    """`missing` entries must NOT carry a producer_module — they are by
    definition unwired. If a producer ships, flip kind to bus_event/hybrid/snapshot."""
    for s in registry.REGISTRY:
        if s.kind == "missing":
            assert not s.producer_module, (
                f"{s.key}: kind=missing but producer_module={s.producer_module!r} "
                f"(flip kind once producer ships)"
            )


def test_lookup_helpers(registry) -> None:
    """by_key / by_event / specs_by_kind round-trip."""
    spec = registry.by_key("chi.state")
    assert spec is not None and spec.key == "chi.state"
    assert spec.producer_event == "CHI_UPDATED"
    assert registry.by_event("CHI_UPDATED") is spec
    assert registry.by_key("nonexistent.key") is None
    assert registry.by_event("NONEXISTENT_EVENT") is None
    assert all(s.kind == "deprecated" for s in registry.specs_by_kind("deprecated"))


def test_allowlist_disjoint_from_registry(registry) -> None:
    """A key listed in ALLOWLIST_KEYS must not also have a REGISTRY entry."""
    overlap = registry.ALLOWLIST_KEYS & registry.REGISTERED_KEYS
    assert not overlap, (
        f"ALLOWLIST_KEYS overlap with REGISTERED_KEYS: {overlap} — "
        f"a key must be in EXACTLY one of allowlist or registry"
    )
