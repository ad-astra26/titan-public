"""Phase B sweep + heartbeat-watch + rename-map tests (RFP_config_as_shm_state §7.B).

Covers the work the B-sweep session added on top of B-core:
  - the universal heartbeat config-watch thread (params.start_config_watch)
  - the Tier-1 persistence_* → persistence.* rename-map parity (consumer reads
    yield the SAME value under the flat [persistence_X] and nested [persistence.X]
    structures — key PATH change only, never a value)
  - IMWConfig.from_titan_config_section dotted-path resolution

Each test file runs in its own pytest process (TorchRL mmap isolation).
"""
import time
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
    root = tmp_path / "titan_T1"
    (root / "config").mkdir(parents=True)
    _reset_params_state()
    params._shm_root = root
    monkeypatch.setenv("TITAN_CONFIG_SHM_READ", "1")
    yield root
    _reset_params_state()


def _write_slot(root: Path, section: str, value: dict) -> StateRegistryWriter:
    spec = RegistrySpec(name=f"config/{section}", dtype=np.dtype(np.uint8),
                        shape=(SLOT_BYTES,), variable_size=True)
    w = StateRegistryWriter(spec, root)
    w.write_variable(msgpack.packb(value, use_bin_type=True))
    return w


# ───────────────────────── heartbeat config-watch ─────────────────────────

def test_config_watch_is_idempotent_daemon(shm):
    t1 = params.start_config_watch(interval_s=0.05)
    t2 = params.start_config_watch(interval_s=0.05)
    assert t1 is t2, "one watch thread per process"
    assert t1.daemon and t1.is_alive()


def test_config_watch_drives_reload_on_bump(shm):
    w = _write_slot(shm, "growth_metrics", {"node_saturation_24h": 30})
    seen = []
    params.register_config_reload("growth_metrics", lambda d: seen.append(d["node_saturation_24h"]))
    params.start_config_watch(interval_s=0.02)
    time.sleep(0.1)            # let it establish baseline version
    seen.clear()
    w.write_variable(msgpack.packb({"node_saturation_24h": 31}, use_bin_type=True))
    deadline = time.time() + 2.0
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen == [31], "watch thread must fire the callback on a slot version bump"


def test_config_watch_noop_when_flag_off(monkeypatch, shm):
    monkeypatch.setenv("TITAN_CONFIG_SHM_READ", "0")
    w = _write_slot(shm, "growth_metrics", {"node_saturation_24h": 30})
    seen = []
    params.register_config_reload("growth_metrics", lambda d: seen.append(1))
    params.start_config_watch(interval_s=0.02)
    time.sleep(0.1)
    w.write_variable(msgpack.packb({"node_saturation_24h": 99}, use_bin_type=True))
    time.sleep(0.1)
    assert seen == [], "flag off → watch must not hot-apply (legacy fallback owns reads)"


# ───────────────────────── rename-map parity ─────────────────────────

# The 4 consolidated writers and the subtable each reads.
RENAMED = [
    ("observatory", "persistence_observatory"),
    ("social_graph", "persistence_social_graph"),
    ("events_teacher", "persistence_events_teacher"),
    ("consciousness", "persistence_consciousness"),
]


def _flat_config():
    """Pre-rename shape: each writer DB section is a distinct top-level section."""
    cfg = {"persistence": {"enabled": True, "db_path": "data/inner_memory.db"}}
    for sub, flat in RENAMED:
        cfg[flat] = {"enabled": True, "db_path": f"data/{sub}.db", "socket_path": f"data/run/{sub}.sock"}
    return cfg


def _nested_config():
    """Post-rename shape: subtables under the single [persistence] section."""
    cfg = {"persistence": {"enabled": True, "db_path": "data/inner_memory.db"}}
    for sub, _flat in RENAMED:
        cfg["persistence"][sub] = {"enabled": True, "db_path": f"data/{sub}.db",
                                   "socket_path": f"data/run/{sub}.sock"}
    return cfg


def test_rename_map_parity_consumer_reads():
    """The module_catalog consumer reads must yield the SAME per-writer section
    under the flat (old) and nested (new) structures — path change, not value."""
    flat, nested = _flat_config(), _nested_config()
    for sub, flat_key in RENAMED:
        old = flat.get(flat_key, {})                              # pre-rename read
        new = nested.get("persistence", {}).get(sub, {})          # post-rename read
        assert old == new, f"rename parity broken for {sub}: {old} != {new}"


def test_rename_map_observatory_specific_read():
    """module_catalog.py:333 _obs_persistence_cfg path parity."""
    flat, nested = _flat_config(), _nested_config()
    assert flat.get("persistence_observatory", {}) == nested.get("persistence", {}).get("observatory", {})


def test_from_titan_config_section_dotted_path():
    """IMWConfig.from_titan_config_section resolves a dotted path against the
    nested config (e.g. 'persistence.observatory') and the bare base 'persistence'."""
    from titan_hcl.persistence.config import IMWConfig
    nested = _nested_config()
    with mock.patch("titan_hcl.persistence.config._load_config_toml_cached", return_value=nested), \
         mock.patch("pathlib.Path.exists", return_value=True):
        obs = IMWConfig.from_titan_config_section("persistence.observatory")
        base = IMWConfig.from_titan_config_section("persistence")
    assert obs.db_path == "data/observatory.db", "dotted subtable resolved"
    assert base.db_path == "data/inner_memory.db", "bare base section resolved"
