"""Settings R/W for the TC² Settings tab.

Reuses setup_titan.config_model (pure stdlib — no Titan runtime deps) so the
SPA gets, per key: current value, the inline-comment help (the "info panel"),
and whether it's safely editable. Edits go through set_by_dotted (comment- and
formatting-preserving).
"""
from __future__ import annotations

import sys
from pathlib import Path

_CONFIG_FILES = ("titan_hcl/config.toml", "titan_plugin/titan_params.toml")


def _config_model(install_root: Path):
    """Import setup_titan.config_model, putting <install_root>/scripts on path."""
    scripts_dir = str(install_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)   # append: don't shadow titan_hcl package
    from setup_titan import config_model  # noqa: E402
    return config_model


def _resolve_files(install_root: Path) -> list[Path]:
    return [install_root / rel for rel in _CONFIG_FILES
            if (install_root / rel).exists()]


def list_config(install_root: Path, *, section: str | None = None) -> dict:
    """All keys across config files: value + help + editable + source file."""
    cm = _config_model(install_root)
    entries = []
    for f in _resolve_files(install_root):
        for e in cm.parse_toml_with_comments(f):
            if section and e.section != section:
                continue
            entries.append({
                "file": f.name, "section": e.section, "key": e.key,
                "dotted": e.dotted, "value": e.raw_value, "help": e.help,
                "editable": e.editable, "lineno": e.lineno,
            })
    sections = sorted({e["section"] for e in entries})
    return {"sections": sections, "entries": entries}


def get_config(install_root: Path, dotted: str) -> dict:
    cm = _config_model(install_root)
    for f in _resolve_files(install_root):
        matches = cm.find_entry(cm.parse_toml_with_comments(f), dotted)
        if matches:
            e = matches[0]
            return {"found": True, "file": f.name, "dotted": e.dotted,
                    "value": e.raw_value, "help": e.help, "editable": e.editable}
    return {"found": False, "dotted": dotted}


def set_config(install_root: Path, dotted: str, value: str) -> dict:
    """Edit one key (comment-preserving). Returns {ok, file} or {ok:False,error}."""
    cm = _config_model(install_root)
    for f in _resolve_files(install_root):
        # Guard against editing a non-editable (multi-line array) key.
        matches = cm.find_entry(cm.parse_toml_with_comments(f), dotted)
        if matches and not matches[0].editable:
            return {"ok": False, "error": f"{dotted} is not safely editable "
                    "(multi-line array/table) — edit the file directly"}
        if matches and cm.set_by_dotted(f, dotted, value):
            return {"ok": True, "file": f.name, "dotted": dotted, "value": value}
    return {"ok": False, "error": f"key not found: {dotted}"}
