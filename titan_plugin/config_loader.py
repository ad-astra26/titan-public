"""Deep-merge config loader for secrets-outside-repo pattern.

Loads ``titan_plugin/config.toml`` (repo-tracked, non-secret values) and
deep-merges ``~/.titan/secrets.toml`` (local, gitignored, secret values) on top.

Introduced 2026-04-16 after the 2026-04-15 public-repo leak audit to move
all credentials out of the repo tree. See memory/feedback_public_sync_pipeline.md
and memory/project_next_session_external_secrets.md for the incident and plan.

Usage::

    from titan_plugin.config_loader import load_titan_config
    cfg = load_titan_config()
    api_key = cfg.get("inference", {}).get("venice_api_key", "")

If ``~/.titan/secrets.toml`` is absent, the loader returns the base config
unchanged and logs a single WARNING — features whose secrets are missing
will run in disabled mode rather than crashing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover — Python 3.10 fallback
    import tomli as tomllib  # type: ignore

_LOG = logging.getLogger("titan.config_loader")

TITAN_PARAMS_PATH = Path(__file__).parent / "titan_params.toml"
BASE_CONFIG_PATH = Path(__file__).parent / "config.toml"
SECRETS_PATH = Path(os.path.expanduser("~/.titan/secrets.toml"))

_cache: dict | None = None
_warned_missing_secrets = False


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base``. Returns a new dict.

    For each key in overlay:
      - If both base[key] and overlay[key] are dicts: recurse.
      - Otherwise: overlay[key] wins.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_titan_config(force_reload: bool = False) -> dict:
    """Load the full Titan config, deep-merging 3 layers bottom-up:
      Layer 1 (base):    titan_params.toml  — engineering defaults, code-tracked
      Layer 2 (middle):  config.toml         — deployment-specific, user-editable,
                                                code NEVER writes this file
      Layer 3 (top):     ~/.titan/secrets.toml  — secrets, external, out-of-repo

    Later layers override earlier ones (deep-merge semantics).

    Args:
        force_reload: If True, bypass the in-process cache (for tests).

    Returns:
        Deep-merged config dict. Empty dict if the base config.toml is
        missing/unreadable. Missing titan_params.toml is non-fatal (treated
        as empty layer); missing secrets.toml is non-fatal (warns once).
    """
    global _cache, _warned_missing_secrets

    if _cache is not None and not force_reload:
        return _cache

    # Layer 1: titan_params.toml — engineering defaults. Non-fatal if missing
    # (e.g., in minimal test fixtures). Historically loaded directly by ~15-20
    # modules; this merge unifies access so all callers see consistent config.
    params: dict = {}
    if TITAN_PARAMS_PATH.exists():
        try:
            with open(TITAN_PARAMS_PATH, "rb") as f:
                params = tomllib.load(f)
        except Exception as e:
            _LOG.warning(
                "[config_loader] Failed to parse %s: %s — skipping Layer 1",
                TITAN_PARAMS_PATH, e,
            )
            params = {}

    # Layer 2: config.toml — deployment-specific overrides. Required.
    if not BASE_CONFIG_PATH.exists():
        _LOG.error("[config_loader] Base config not found at %s", BASE_CONFIG_PATH)
        _cache = {}
        return _cache

    try:
        with open(BASE_CONFIG_PATH, "rb") as f:
            base = tomllib.load(f)
    except Exception as e:
        _LOG.error("[config_loader] Failed to parse %s: %s", BASE_CONFIG_PATH, e)
        _cache = {}
        return _cache

    merged = _deep_merge(params, base)

    # Layer 3: ~/.titan/secrets.toml — external secrets overrides.
    if SECRETS_PATH.exists():
        try:
            with open(SECRETS_PATH, "rb") as f:
                secrets = tomllib.load(f)
            merged = _deep_merge(merged, secrets)
            _LOG.info(
                "[config_loader] Merged secrets from %s (sections: %s)",
                SECRETS_PATH,
                sorted(secrets.keys()),
            )
            _cache = merged
            return _cache
        except Exception as e:
            _LOG.warning(
                "[config_loader] Failed to merge %s: %s — using base config only",
                SECRETS_PATH,
                e,
            )
            _cache = merged
            return _cache

    if not _warned_missing_secrets:
        _LOG.warning(
            "[config_loader] %s not found — secret-dependent features will be disabled. "
            "Create it with: mkdir -p ~/.titan && chmod 700 ~/.titan && "
            "touch ~/.titan/secrets.toml && chmod 600 ~/.titan/secrets.toml",
            SECRETS_PATH,
        )
        _warned_missing_secrets = True
    _cache = merged
    return _cache


def clear_cache() -> None:
    """Clear the in-process cache. Tests only."""
    global _cache, _warned_missing_secrets
    _cache = None
    _warned_missing_secrets = False


def update_secret(section: str, key: str, value) -> bool:
    """Atomically update a single ``[section].key`` field in ``~/.titan/secrets.toml``.

    Creates ``~/.titan/`` with mode 700 and the file with mode 600 if absent.
    Preserves all other keys. Clears the in-process cache so the next
    ``load_titan_config()`` picks up the new value.

    Used by code paths that rotate secrets at runtime (e.g. SocialXGateway
    refreshing an X session cookie). Returns True on success.
    """
    import tomli_w

    try:
        SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(SECRETS_PATH.parent, 0o700)
        except Exception:
            pass

        existing: dict = {}
        if SECRETS_PATH.exists():
            try:
                with open(SECRETS_PATH, "rb") as f:
                    existing = tomllib.load(f)
            except Exception as e:
                _LOG.warning("[config_loader] update_secret: can't parse existing %s: %s", SECRETS_PATH, e)
                existing = {}

        if section not in existing or not isinstance(existing.get(section), dict):
            existing[section] = {}
        existing[section][key] = value

        tmp_path = SECRETS_PATH.with_suffix(".toml.tmp")
        with open(tmp_path, "wb") as f:
            tomli_w.dump(existing, f)
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        os.replace(tmp_path, SECRETS_PATH)
        clear_cache()
        return True
    except Exception as e:
        _LOG.warning("[config_loader] update_secret(%s.%s) failed: %s", section, key, e)
        return False
