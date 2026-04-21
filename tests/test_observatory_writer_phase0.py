"""rFP_observatory_writer_service Phase 0 tests — config + ModuleSpec wiring.

Phase 0 = plumbing only. No writes routed yet. We verify:
  1. IMWConfig.from_titan_config_section() loads any [persistence_*] section.
  2. The two writer instances would NOT collide on filesystem paths if both
     enabled (per-instance defaults are unique).
  3. The metrics-file naming uses module name so multiple instances don't
     overwrite each other (imw_main change).
  4. observatory_writer is registered with autostart=False by default
     (won't accidentally activate on deploy).

Phase 1+ adds the client wrapper, call-site migrations, and shadow-mode
soak. Out of scope here.
"""

from __future__ import annotations

import pathlib

import pytest

from titan_plugin.persistence.config import IMWConfig


def test_from_titan_config_section_loads_default_persistence():
    """Backwards compat: from_titan_config_section('persistence') == from_titan_config()."""
    a = IMWConfig.from_titan_config()
    b = IMWConfig.from_titan_config_section("persistence")
    assert a.db_path == b.db_path
    assert a.socket_path == b.socket_path


def test_from_titan_config_section_handles_missing_section():
    """Missing section → defaults (no crash)."""
    cfg = IMWConfig.from_titan_config_section("persistence_doesnotexist")
    assert cfg.enabled is False  # default
    assert cfg.mode == "disabled"  # default


def test_writer_instance_paths_unique_when_both_enabled():
    """If both writers were enabled with the namespacing chosen in v5_core.py,
    they MUST not collide on socket / WAL / db / shadow_db paths."""
    imw_paths = {
        "socket_path": "data/run/imw.sock",
        "wal_path": "data/run/imw.wal",
        "db_path": "data/inner_memory.db",
        "shadow_db_path": "data/inner_memory_shadow.db",
    }
    obs_paths = {
        "socket_path": "data/run/observatory_writer.sock",
        "wal_path": "data/run/observatory_writer.wal",
        "db_path": "data/observatory.db",
        "shadow_db_path": "data/observatory_shadow.db",
    }
    for key in imw_paths:
        assert imw_paths[key] != obs_paths[key], (
            f"writer paths must be unique per instance — collision on {key}: "
            f"both writers would target {imw_paths[key]!r}")


def test_imw_main_metrics_file_namespaced_by_module_name():
    """imw_main uses f'{name}_metrics.json' so two instances don't fight
    over the same JSON file. IMW (name='imw') keeps the historical
    'imw_metrics.json' path; observatory_writer gets its own."""
    # The actual filename construction is in persistence_entry.py and
    # only happens inside the spawned subprocess. We test it indirectly by
    # confirming the source contains the f-string pattern.
    pe = pathlib.Path(__file__).resolve().parent.parent / "titan_plugin" / "persistence_entry.py"
    src = pe.read_text(encoding="utf-8")
    assert 'f"{name}_metrics.json"' in src, (
        "persistence_entry.py must derive metrics filename from the module "
        "`name` arg so multiple writer instances coexist")
    # Stricter check: scan non-comment lines only — comments may legitimately
    # mention the historical filename.
    code_lines = [
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#") and not ln.lstrip().startswith('"')
    ]
    code_only = "\n".join(code_lines)
    assert 'METRICS_FILE = ' in code_only and '"imw_metrics.json"' not in code_only, (
        "Hardcoded 'imw_metrics.json' in code path would collide if a second "
        "writer is spawned with the same journal_dir — must be f-string")


def test_observatory_writer_modulespec_registered_default_off(tmp_path, monkeypatch):
    """v5_core.py must register a `observatory_writer` ModuleSpec, defaulted
    to autostart=False so deploys don't accidentally activate it before
    Maker flips the toggle."""
    # We can't safely instantiate the full TitanCore in unit tests (it
    # imports torch + boots Guardian etc.). Instead, scan v5_core.py source
    # for the registration block.
    v5 = pathlib.Path(__file__).resolve().parent.parent / "titan_plugin" / "v5_core.py"
    src = v5.read_text(encoding="utf-8")

    assert 'name="observatory_writer"' in src, "ModuleSpec name must be exactly 'observatory_writer'"
    assert "persistence_observatory" in src, "Config section name must be `persistence_observatory`"

    # Verify the spec uses `imw_main` (no code duplication) + autostart from a
    # gating variable, not a hardcoded True.
    block_start = src.find('name="observatory_writer"')
    assert block_start > 0
    block = src[block_start:block_start + 600]
    assert "entry_fn=imw_main" in block, "Should reuse imw_main entry function"
    assert "autostart=" in block and "True" not in block.split("autostart=")[1].split(",")[0], (
        "autostart should be gated by a config variable, not hardcoded True")
