"""
Centralized parameter loader for Titan V4.

Reads titan_params.toml once, caches result. All subsystems pull their
section from the shared dict instead of hardcoding constants.

Usage:
    from titan_plugin.params import get_params
    cfg = get_params("reflexes")
    threshold = cfg["fire_threshold"]   # 0.15
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PARAMS_CACHE: Optional[dict] = None
_PARAMS_PATH = Path(__file__).parent / "titan_params.toml"


def _load_toml(path: Path) -> dict:
    """Load a TOML file, trying tomllib (3.11+) then tomli fallback."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.warning("[params] No TOML parser available, returning empty config")
            return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_titan_params(force_reload: bool = False) -> dict:
    """Load and cache all titan_params.toml sections."""
    global _PARAMS_CACHE
    if _PARAMS_CACHE is not None and not force_reload:
        return _PARAMS_CACHE

    if not _PARAMS_PATH.exists():
        logger.warning("[params] %s not found, using defaults", _PARAMS_PATH)
        _PARAMS_CACHE = {}
        return _PARAMS_CACHE

    try:
        _PARAMS_CACHE = _load_toml(_PARAMS_PATH)
        logger.info("[params] Loaded %d sections from titan_params.toml",
                     len(_PARAMS_CACHE))
    except Exception as e:
        logger.error("[params] Failed to parse titan_params.toml: %s", e)
        _PARAMS_CACHE = {}

    return _PARAMS_CACHE


def get_params(section: str) -> dict:
    """Get a specific section from titan_params.toml, or empty dict."""
    params = load_titan_params()
    return dict(params.get(section, {}))
