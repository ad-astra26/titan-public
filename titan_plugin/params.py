"""
Centralized parameter loader for Titan V4 — thin delegator over
``titan_plugin.config_loader.load_titan_config``.

Originally (pre-2026-04-29) this module read ``titan_params.toml`` directly
into its own cache, bypassing the 4-layer merge that ``config_loader``
performs (titan_params.toml < config.toml < ~/.titan/secrets.toml <
~/.titan/microkernel_<TID>.toml). That parallel system meant any
``[reflexes]`` / ``[meta_cgn]`` / ``[emot_cgn]`` / ``[prediction_engine]``
override placed in config.toml or the per-Titan override file was invisible
to ``get_params(section)`` callers — see BUG-CONFIG-LOADER-MERGE-TITAN-PARAMS.

This rewrite delegates to ``config_loader.load_titan_config`` so all callers
see the same merged view. The public API (``get_params``, ``load_titan_params``)
is preserved byte-for-byte.

Usage:
    from titan_plugin.params import get_params
    cfg = get_params("reflexes")
    threshold = cfg["fire_threshold"]   # 0.15
"""
from typing import Optional

from titan_plugin.config_loader import load_titan_config


def load_titan_params(force_reload: bool = False) -> dict:
    """Return the full merged Titan config (4-layer).

    Backwards-compatible wrapper around ``config_loader.load_titan_config``.
    The legacy name is preserved so existing callers keep working without
    edits; the returned dict now reflects the full merge instead of just
    Layer 1.
    """
    return load_titan_config(force_reload=force_reload)


def get_params(section: str) -> dict:
    """Get a section from the merged Titan config.

    Returns a fresh ``dict()`` copy so callers cannot mutate the cached
    config in place.
    """
    cfg = load_titan_config()
    return dict(cfg.get(section, {}))
