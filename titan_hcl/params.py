"""
Centralized parameter loader for Titan V4 — thin delegator over
``titan_hcl.config_loader.load_titan_config``.

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

Phase B (RFP_config_as_shm_state §7.B, 2026-06-17): ``get_params`` now reads the
per-section SHM slot the in-kernel config daemon seeds + maintains
(``/dev/shm/titan_<id>/config/<section>.bin``, msgpack) — config-as-SHM-state
(INV-CFG-7). It falls back to the legacy ``config_loader`` merge when the slot is
absent (a box without the Phase-A daemon, or boot before the seed) or when
``TITAN_CONFIG_SHM_READ=0``. The ~8 existing callsites change zero lines.
``register_config_reload`` + ``poll_config_reloads`` give a worker's heartbeat a
cheap version-poll so a cached/derived ``hot`` value can be re-applied on a slot
bump (the §1.2 "worker reads on heartbeat" model).

Usage:
    from titan_hcl.params import get_params
    cfg = get_params("reflexes")
    threshold = cfg["fire_threshold"]   # 0.15
"""
import logging
import os
from typing import Callable, Optional

from titan_hcl.config_loader import load_titan_config

_LOG = logging.getLogger("titan.params")

# Per-section config slot capacity — MUST match the Rust daemon's CONFIG_SLOT_BYTES
# (titan-kernel-rs/src/config_daemon.rs); total file = 16 + 3*(16 + 65536) = 196672 B.
_CONFIG_SLOT_BYTES = 65536

_shm_readers: dict = {}          # section -> StateRegistryReader (cached; mmap reused)
_shm_root = None                 # resolved once
_reload_callbacks: dict = {}     # section -> list[Callable[[dict], None]]
_last_versions: dict = {}        # section -> last slot version seen by poll
_shm_unavailable = False         # set once if the SHM stack can't be imported/resolved


def _config_shm_enabled() -> bool:
    """Whether ``get_params`` reads the SHM slot (vs the legacy ``config_loader`` merge).

    DEFAULT-OFF as a DELIBERATE Phase-B-completion gate (not a forgotten flag — see
    RFP_config_as_shm_state §7.B): B-core (this reimpl) is parity-verified on T1, but
    flipping the fleet to SHM reads is unsafe until the B-sweep lands AND T2/T3's
    `microkernel_<id>.toml` overrides are folded into config.toml (the daemon doesn't
    read overrides, so T2/T3 slots are override-incomplete). Set ``TITAN_CONFIG_SHM_READ=1``
    to enable (T1 is ready today). When B fully completes fleet-wide, flip this default ON."""
    return os.environ.get("TITAN_CONFIG_SHM_READ", "0").lower() in ("1", "true", "yes", "on")


def _reader_for(section: str):
    """Cached ``StateRegistryReader`` for ``config/<section>.bin``; ``None`` if SHM unavailable."""
    global _shm_root, _shm_unavailable
    if _shm_unavailable:
        return None
    reader = _shm_readers.get(section)
    if reader is not None:
        return reader
    try:
        import numpy as np
        from titan_hcl.core.state_registry import (
            RegistrySpec,
            StateRegistryReader,
            resolve_shm_root,
        )
        if _shm_root is None:
            _shm_root = resolve_shm_root()
        spec = RegistrySpec(
            name=f"config/{section}",
            dtype=np.dtype(np.uint8),
            shape=(_CONFIG_SLOT_BYTES,),
            variable_size=True,
        )
        reader = StateRegistryReader(spec, _shm_root)
        _shm_readers[section] = reader
        return reader
    except Exception as e:  # SHM stack missing (minimal test env) → permanent fallback
        _shm_unavailable = True
        _LOG.info("params: SHM config read unavailable (%s) — using config_loader fallback", e)
        return None


def _read_config_slot(section: str) -> Optional[dict]:
    """Read + msgpack-decode the section's SHM slot. ``None`` ⇒ caller falls back."""
    reader = _reader_for(section)
    if reader is None:
        return None
    try:
        import msgpack
        raw = reader.read_variable()
        if not raw:
            return None
        val = msgpack.unpackb(raw, raw=False)
        return val if isinstance(val, dict) else None
    except Exception as e:
        _LOG.debug("params: SHM read of %s failed (%s) — falling back", section, e)
        return None


def _legacy_get_params(section: str) -> dict:
    cfg = load_titan_config()
    return dict(cfg.get(section, {}))


def load_titan_params(force_reload: bool = False) -> dict:
    """Return the full merged Titan config (4-layer).

    Backwards-compatible wrapper around ``config_loader.load_titan_config``.
    The legacy name is preserved so existing callers keep working without
    edits; the returned dict now reflects the full merge instead of just
    Layer 1. (Whole-config reads stay file-backed; only per-section
    ``get_params`` moved to SHM in Phase B.)
    """
    return load_titan_config(force_reload=force_reload)


def get_params(section: str) -> dict:
    """Get a section's config dict — from the per-section SHM slot (Phase B), with
    a fallback to the ``config_loader`` merge when the slot is absent or SHM is off.

    Returns a fresh dict so callers cannot mutate cached state in place.
    """
    if _config_shm_enabled():
        d = _read_config_slot(section)
        if d is not None:
            return d
    return _legacy_get_params(section)


def register_config_reload(section: str, callback: Callable[[dict], None]) -> None:
    """Register ``callback(new_section_dict)`` to fire when ``section``'s slot version
    bumps (call ``poll_config_reloads`` from the worker's heartbeat). Only needed for
    values a worker DERIVES/caches at boot; per-call ``get_params`` readers see live
    values for free. ``restart_required`` sections are never hot-applied (INV-CFG-4)."""
    _reload_callbacks.setdefault(section, []).append(callback)


def poll_config_reloads() -> list:
    """Heartbeat hook: for each registered section, if its slot version bumped since the
    last poll, re-read it and fire the callbacks. Returns the list of sections re-applied.
    Cheap (one int version-compare per registered section); never raises."""
    applied = []
    for section, callbacks in list(_reload_callbacks.items()):
        reader = _reader_for(section)
        if reader is None:
            continue
        try:
            meta = reader.read_meta()
            if not meta:
                continue
            version = meta.get("version")
            if version is None or _last_versions.get(section) == version:
                continue
            _last_versions[section] = version
            new = get_params(section)
            for cb in callbacks:
                try:
                    cb(new)
                except Exception as e:
                    _LOG.warning("params: config-reload callback for %s failed: %s", section, e)
            applied.append(section)
        except Exception as e:
            _LOG.debug("params: poll_config_reloads(%s) error: %s", section, e)
    return applied
