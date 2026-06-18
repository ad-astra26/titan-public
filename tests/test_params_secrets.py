"""Tests for the params helpers relocated from the retired config_loader
(RFP_config_as_shm_state §7.C / C.5): ``update_secret`` (the one runtime config
WRITE path) and ``_bootstrap_merge`` (the SHM-absent params⊎config fallback).

These cover the functions that used to live in titan_hcl/config_loader.py and
now live in titan_hcl/params.py. The legacy 4-layer ``load_titan_config`` merge,
its cache, and the per-titan override layer are GONE (the in-kernel config
daemon owns the merge in SHM); only the bootstrap fallback survives.
"""
from __future__ import annotations

from unittest import mock


def _write_toml(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_update_secret_creates_file_and_writes_section(tmp_path):
    """update_secret should create secrets.toml (and the section) if absent."""
    secrets = tmp_path / "subdir" / "secrets.toml"  # parent doesn't exist yet
    _write_toml(tmp_path / "config.toml", "[api]\nport = 7777\n")

    from titan_hcl import params

    with mock.patch.object(params, "_SECRETS_PATH", str(secrets)):
        ok = params.update_secret("twitter_social", "auth_session", "new_sess_v1")
        assert ok is True
        assert secrets.exists()
        # The bootstrap merge (SHM-absent fallback) now reflects the secret.
        cfg = params._bootstrap_merge(base_dir=str(tmp_path))
        assert cfg["twitter_social"]["auth_session"] == "new_sess_v1"


def test_update_secret_preserves_existing_keys(tmp_path):
    """update_secret must preserve all other keys in secrets.toml."""
    secrets = tmp_path / "secrets.toml"
    _write_toml(tmp_path / "config.toml", "[api]\nport = 7777\n")
    _write_toml(
        secrets,
        """\
[inference]
venice_api_key = "vk_placeholder_preserved"
openrouter_api_key = "or_placeholder_preserved"

[twitter_social]
password = "placeholder_preserved_pw"
auth_session = "placeholder_old_sess"
""",
    )

    from titan_hcl import params

    with mock.patch.object(params, "_SECRETS_PATH", str(secrets)):
        ok = params.update_secret("twitter_social", "auth_session", "new_sess_v2")
        assert ok is True
        cfg = params._bootstrap_merge(base_dir=str(tmp_path))
        assert cfg["inference"]["venice_api_key"] == "vk_placeholder_preserved"
        assert cfg["inference"]["openrouter_api_key"] == "or_placeholder_preserved"
        assert cfg["twitter_social"]["password"] == "placeholder_preserved_pw"
        assert cfg["twitter_social"]["auth_session"] == "new_sess_v2"


def test_bootstrap_merge_overlays_config_over_params(tmp_path):
    """``_bootstrap_merge`` deep-merges config.toml (Layer 2) over
    titan_params.toml (Layer 1) — the SHM-absent equivalent of what the daemon
    seeds into the per-section slots (GB1 parity-verified)."""
    _write_toml(
        tmp_path / "titan_params.toml",
        "[reflexes]\nfire_threshold = 0.10\ncooldown_ms = 250\n",
    )
    _write_toml(tmp_path / "config.toml", "[reflexes]\nfire_threshold = 0.20\n")

    from titan_hcl import params

    # No secrets file → overlay is a no-op; isolate from the real ~/.titan.
    with mock.patch.object(params, "_SECRETS_PATH", str(tmp_path / "nope.toml")):
        merged = params._bootstrap_merge(base_dir=str(tmp_path))
    section = merged["reflexes"]
    assert section["fire_threshold"] == 0.20    # Layer 2 overrides Layer 1
    assert section["cooldown_ms"] == 250        # Layer 1 key still visible
