"""Phase B B-core tests: get_params() SHM read + fallback + reload callbacks
(RFP_config_as_shm_state §7.B).

Seeds a real per-section slot with StateRegistryWriter (same wire format the Rust
daemon writes), points params at a temp shm root, and exercises the SHM read path,
the legacy fallback, the kill-switch, and the heartbeat reload-callback registry.

No TitanHCL instance is created. The live end-to-end parity (get_params(SHM) ==
config_loader merge) is verified separately against production slots by
scripts/config_phaseb_parity_live.py.
"""
from pathlib import Path
from unittest import mock

import msgpack
import numpy as np
import pytest

import titan_hcl.params as params
from titan_hcl.core.state_registry import RegistrySpec, StateRegistryWriter

SLOT_BYTES = params._CONFIG_SLOT_BYTES


def _reset_params_state():
    params._shm_readers.clear()
    params._reload_callbacks.clear()
    params._last_versions.clear()
    params._shm_root = None
    params._shm_unavailable = False


@pytest.fixture
def shm(tmp_path, monkeypatch):
    """A temp shm root with a config/ subdir; params resolves to it."""
    root = tmp_path / "titan_T1"
    (root / "config").mkdir(parents=True)
    _reset_params_state()
    params._shm_root = root  # bypass resolve_shm_root → our temp
    monkeypatch.setenv("TITAN_CONFIG_SHM_READ", "1")
    yield root
    _reset_params_state()


def _write_slot(root: Path, section: str, value: dict) -> StateRegistryWriter:
    spec = RegistrySpec(
        name=f"config/{section}",
        dtype=np.dtype(np.uint8),
        shape=(SLOT_BYTES,),
        variable_size=True,
    )
    w = StateRegistryWriter(spec, root)
    w.write_variable(msgpack.packb(value, use_bin_type=True))
    return w


def test_get_params_reads_shm_slot(shm):
    _write_slot(shm, "social_x", {"post_dispatch_tick_interval_seconds": 15.0, "enabled": True})
    out = params.get_params("social_x")
    assert out == {"post_dispatch_tick_interval_seconds": 15.0, "enabled": True}


def test_absent_slot_falls_back_to_loader(shm):
    # no slot for "reflexes" → _read_config_slot returns None → legacy path
    assert params._read_config_slot("reflexes") is None
    with mock.patch.object(params, "load_titan_config", return_value={"reflexes": {"fire_threshold": 0.15}}):
        assert params.get_params("reflexes") == {"fire_threshold": 0.15}


def test_kill_switch_forces_legacy(shm, monkeypatch):
    _write_slot(shm, "social_x", {"x": 1})  # slot exists...
    monkeypatch.setenv("TITAN_CONFIG_SHM_READ", "0")  # ...but flag off
    with mock.patch.object(params, "load_titan_config", return_value={"social_x": {"x": 999}}):
        assert params.get_params("social_x") == {"x": 999}  # legacy value, not the slot


def test_get_params_returns_fresh_copy(shm):
    _write_slot(shm, "cgn", {"a": 1})
    a = params.get_params("cgn")
    a["a"] = 99
    assert params.get_params("cgn")["a"] == 1  # mutation didn't leak


def test_reload_callback_fires_on_version_bump(shm):
    w = _write_slot(shm, "growth_metrics", {"node_saturation_24h": 30})
    seen = []
    params.register_config_reload("growth_metrics", lambda d: seen.append(d["node_saturation_24h"]))

    # first poll establishes the baseline version (may or may not fire once)
    params.poll_config_reloads()
    seen.clear()

    # no change → no fire
    assert params.poll_config_reloads() == []
    assert seen == []

    # bump the slot → next poll fires the callback with the new value
    w.write_variable(msgpack.packb({"node_saturation_24h": 31}, use_bin_type=True))
    applied = params.poll_config_reloads()
    assert applied == ["growth_metrics"]
    assert seen == [31]


def test_poll_never_raises_without_slot(shm):
    params.register_config_reload("nonexistent", lambda d: None)
    assert params.poll_config_reloads() == []  # no slot → no fire, no raise


def test_trinity_restoring_rehome_fires_publisher_on_bump(shm):
    """C.1 (§7.C): the trinity-restoring sidecar republish — formerly triggered by the
    now-retired /v4/reload-config endpoint — is re-homed as a heartbeat config-watch
    callback. This is the EXACT wiring scripts/titan_hcl.py installs at boot: a
    trinity_restoring slot version bump must re-run the publisher (so the 6 Rust trinity
    daemons pick up edited [trinity_restoring] gains with no restart)."""
    w = _write_slot(shm, "trinity_restoring", {"body": {"k_drive": 0.5}})
    calls = []
    params.register_config_reload("trinity_restoring", lambda _new: calls.append(1))

    params.poll_config_reloads()  # establish baseline version
    calls.clear()
    assert params.poll_config_reloads() == []  # no edit → no republish
    assert calls == []

    # operator edits [trinity_restoring].body.k_drive → daemon bumps the slot
    w.write_variable(msgpack.packb({"body": {"k_drive": 0.9}}, use_bin_type=True))
    assert params.poll_config_reloads() == ["trinity_restoring"]
    assert calls == [1]  # sidecar republished
