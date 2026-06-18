"""
Centralized parameter loader for Titan — config-as-SHM-state (INV-CFG-7).

``get_params(section)`` reads the per-section SHM slot the in-kernel config
daemon seeds + maintains (``/dev/shm/titan_<id>/config/<section>.bin``, msgpack)
— config IS SHM-state. ``load_titan_params()`` assembles the whole config by
enumerating every ``config/*.bin`` slot. This is the ONLY worker config read
path (RFP_config_as_shm_state §7.C / Phase C, 2026-06-18).

When the SHM stack is unavailable — pytest with no daemon, or birth/installer
before the daemon seeds slots — both fall back to ``_bootstrap_merge`` (the
minimal ``titan_params.toml ⊎ config.toml`` + ``~/.titan/secrets.toml`` overlay,
mirroring the Rust daemon, NO microkernel layer). 🚫 ``_bootstrap_merge`` is NOT
a worker runtime path (INV-CFG-1): on a live box every worker reads SHM.

The legacy ``config_loader.load_titan_config`` 4-layer merge + the per-section
``_legacy_get_params`` fallback are RETIRED in Phase C (C.4) — the merged view a
worker sees comes from the daemon's slots, not a Python re-merge.

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
import threading
import time
from typing import Callable, Optional

_LOG = logging.getLogger("titan.params")

# The runtime secrets overlay file. Single source for the bootstrap READ
# (_bootstrap_merge) and the runtime WRITE (update_secret) so a test can
# redirect both by patching this one symbol.
_SECRETS_PATH = os.path.join(os.path.expanduser("~/.titan"), "secrets.toml")

# Per-section config slot capacity — MUST match the Rust daemon's CONFIG_SLOT_BYTES
# (titan-kernel-rs/src/config_daemon.rs); total file = 16 + 3*(16 + 65536) = 196672 B.
_CONFIG_SLOT_BYTES = 65536

_shm_readers: dict = {}          # section -> StateRegistryReader (cached; mmap reused)
_shm_root = None                 # resolved once
_reload_callbacks: dict = {}     # section -> list[Callable[[dict], None]]
_last_versions: dict = {}        # section -> last slot version seen by poll
_shm_unavailable = False         # set once if the SHM stack can't be imported/resolved
_watch_thread = None             # the universal config-watch daemon thread (one per process)
_watch_lock = threading.Lock()


def _config_shm_enabled() -> bool:
    """Whether ``get_params`` reads the SHM slot (vs the ``_bootstrap_merge`` fallback).

    DEFAULT-ON (RFP_config_as_shm_state §7.C / C.4, 2026-06-18): SHM is the canonical
    worker config read path fleet-wide (Phase B verified parity 100% on all 3 boxes;
    overrides folded). The ``TITAN_CONFIG_SHM_READ=0`` env kill-switch is retained for
    emergencies only — the per-box ``.titan_env`` gate that used to flip this ON is
    removed at deploy (SHM is the only path, no env dependency). When SHM is genuinely
    unavailable (pytest/birth/pre-daemon) the slot read returns None and the caller
    bootstraps; the flag does not need to be off for those paths to work."""
    return os.environ.get("TITAN_CONFIG_SHM_READ", "1").lower() in ("1", "true", "yes", "on")


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
        _LOG.info("params: SHM config read unavailable (%s) — using bootstrap-merge fallback", e)
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


def _deep_merge_into(base: dict, overlay: dict) -> None:
    """Table-deep, later-wins merge of ``overlay`` into ``base`` (in place) — mirrors the
    Rust config daemon's ``deep_merge`` and the retired ``config_loader._deep_merge``."""
    for k, v in overlay.items():
        if isinstance(base.get(k), dict) and isinstance(v, dict):
            _deep_merge_into(base[k], v)
        else:
            base[k] = v


def _bootstrap_merge(base_dir: Optional[str] = None) -> dict:
    """Minimal whole-config merge: ``titan_params.toml ⊎ config.toml`` + the
    ``~/.titan/secrets.toml`` overlay — table-deep, later-wins, NO microkernel layer
    (matching the Rust config daemon, which dropped the override layer in Phase 0).

    🚫 NOT a worker runtime path (INV-CFG-1). Used ONLY when the SHM config stack is
    unavailable: pytest with no daemon, or birth/installer before the daemon seeds slots.
    On a live box every worker reads SHM (``get_params`` / ``load_titan_params`` assembly).

    ``base_dir`` overrides where the two config TOMLs are read from (tests only; the
    secrets overlay path is unaffected). Production callers pass nothing → titan_hcl/."""
    import tomllib
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    merged: dict = {}
    for fname in ("titan_params.toml", "config.toml"):
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    _deep_merge_into(merged, tomllib.load(f))
            except Exception as e:
                _LOG.warning("params: bootstrap merge of %s failed: %s", fname, e)
    if os.path.exists(_SECRETS_PATH):
        try:
            with open(_SECRETS_PATH, "rb") as f:
                _deep_merge_into(merged, tomllib.load(f))
        except Exception as e:
            _LOG.warning("params: bootstrap secrets merge failed: %s", e)
    return merged


# ── Runtime secrets writer ──────────────────────────────────────────────────
# The ONLY runtime config WRITE path (config is otherwise read-only SHM-state).
# Relocated from the retired config_loader.py (RFP_config_as_shm_state §7.C/C.5).
def update_secret(section: str, key: str, value) -> bool:
    """Atomically update a single ``[section].key`` field in ``~/.titan/secrets.toml``.

    Creates ``~/.titan/`` with mode 700 and the file with mode 600 if absent.
    Preserves all other keys. Returns True on success.

    Used by code paths that rotate secrets at runtime (e.g. SocialXGateway
    refreshing an X session cookie). In config-as-SHM-state the in-kernel config
    daemon watches ``secrets.toml``'s mtime (CONFIG_WATCH_POLL_MS=1s) and re-seeds
    the affected section slots, so a subsequent ``get_params(section)`` sees the
    new value within ~1s — no in-process cache to clear (INV-CFG-7)."""
    import tomllib
    import tomli_w

    try:
        secrets_dir = os.path.dirname(_SECRETS_PATH)
        os.makedirs(secrets_dir, exist_ok=True)
        try:
            os.chmod(secrets_dir, 0o700)
        except Exception:
            pass

        existing: dict = {}
        if os.path.exists(_SECRETS_PATH):
            try:
                with open(_SECRETS_PATH, "rb") as f:
                    existing = tomllib.load(f)
            except Exception as e:
                _LOG.warning("params: update_secret can't parse existing %s: %s", _SECRETS_PATH, e)
                existing = {}

        if section not in existing or not isinstance(existing.get(section), dict):
            existing[section] = {}
        existing[section][key] = value

        tmp_path = _SECRETS_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            tomli_w.dump(existing, f)
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        os.replace(tmp_path, _SECRETS_PATH)
        return True
    except Exception as e:
        _LOG.warning("params: update_secret(%s.%s) failed: %s", section, key, e)
        return False


def _all_config_sections() -> Optional[list]:
    """Top-level config section names from the SHM ``config/`` dir (one ``<section>.bin``
    slot per section, seeded by the in-kernel daemon). ``None`` ⇒ SHM unavailable ⇒ the
    caller bootstraps."""
    global _shm_root, _shm_unavailable
    if _shm_unavailable:
        return None
    try:
        from titan_hcl.core.state_registry import resolve_shm_root
        if _shm_root is None:
            _shm_root = resolve_shm_root()
        cfg_dir = os.path.join(str(_shm_root), "config")
        if not os.path.isdir(cfg_dir):
            return None
        return sorted(n[:-4] for n in os.listdir(cfg_dir) if n.endswith(".bin"))
    except Exception as e:
        _LOG.debug("params: cannot list SHM config slots (%s)", e)
        return None


def load_titan_params(force_reload: bool = False) -> dict:
    """Return the full merged Titan config as ``{section: dict}``.

    SHM-assembled (Phase C / INV-CFG-7): reads every per-section config slot the in-kernel
    daemon seeds (``/dev/shm/titan_<id>/config/*.bin``) and assembles them — config IS
    SHM-state. Falls back to ``_bootstrap_merge`` (the minimal file merge) ONLY when SHM is
    unavailable (pytest with no daemon, or birth/installer before the daemon seeds slots).

    ``force_reload`` is accepted for backwards-compatibility but is a no-op: SHM reads are
    always fresh (no cache); the bootstrap path re-reads each call."""
    if _config_shm_enabled():
        sections = _all_config_sections()
        if sections:
            whole: dict = {}
            for sec in sections:
                d = _read_config_slot(sec)
                if d is not None:
                    whole[sec] = d
            if whole:
                return whole
    return _bootstrap_merge()


def get_params(section: str) -> dict:
    """Get a section's config dict — from the per-section SHM slot (config-as-state,
    INV-CFG-7). Falls back to ``_bootstrap_merge`` ONLY when the SHM slot is absent
    (pytest with no daemon, or birth/installer before the daemon seeds slots).

    Returns a fresh dict so callers cannot mutate cached state in place.
    """
    if _config_shm_enabled():
        d = _read_config_slot(section)
        if d is not None:
            return d
    return dict(_bootstrap_merge().get(section, {}))


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


def start_config_watch(interval_s: float = 3.0):
    """Start the UNIVERSAL heartbeat config-watch daemon thread (idempotent per
    process). Installed once per worker at the orchestrator's ``_module_wrapper``
    (RFP_config_as_shm_state §7.B): a daemon thread that, while SHM reads are
    enabled, calls ``poll_config_reloads`` every ``interval_s`` so a slot version
    bump fires the worker's registered ``hot`` reload callbacks (the §1.2 "worker
    reads on heartbeat" model). Per-call ``get_params`` readers already see live
    values for free — this thread exists ONLY to re-apply values a worker derived
    once at boot. ``restart_required`` sections are surfaced by the daemon
    (``CONFIG_RESTART_REQUIRED``) and carry no hot callback, so they are never
    hot-applied here (INV-CFG-4). The thread is a no-op (cheap version-poll) when
    no callback is registered, and sleeps harmlessly when the flag is off."""
    global _watch_thread
    with _watch_lock:
        if _watch_thread is not None and _watch_thread.is_alive():
            return _watch_thread

        def _loop():
            while True:
                try:
                    if _config_shm_enabled():
                        poll_config_reloads()
                except Exception:  # never let the watch thread die or crash the worker
                    pass
                time.sleep(interval_s)

        t = threading.Thread(target=_loop, name="titan-config-watch", daemon=True)
        t.start()
        _watch_thread = t
        return t
