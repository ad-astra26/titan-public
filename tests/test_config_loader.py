"""Tests for titan_plugin.config_loader deep-merge behavior."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    """Ensure cache is clean between tests."""
    from titan_plugin import config_loader

    config_loader.clear_cache()
    yield
    config_loader.clear_cache()


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_deep_merge_leaf_overrides():
    from titan_plugin.config_loader import _deep_merge

    base = {"a": 1, "b": 2}
    overlay = {"b": 20, "c": 3}
    merged = _deep_merge(base, overlay)
    assert merged == {"a": 1, "b": 20, "c": 3}
    assert base == {"a": 1, "b": 2}  # unmutated


def test_deep_merge_nested_dicts():
    from titan_plugin.config_loader import _deep_merge

    base = {"inference": {"provider": "venice", "api_key": "", "timeout": 30}}
    overlay = {"inference": {"api_key": "secret-key-value"}}
    merged = _deep_merge(base, overlay)
    assert merged == {"inference": {"provider": "venice", "api_key": "secret-key-value", "timeout": 30}}


def test_deep_merge_new_section_from_overlay():
    from titan_plugin.config_loader import _deep_merge

    base = {"api": {"port": 7777}}
    overlay = {"arc_agi_3": {"api_key": "new-key"}}
    merged = _deep_merge(base, overlay)
    assert merged == {"api": {"port": 7777}, "arc_agi_3": {"api_key": "new-key"}}


def test_load_merges_secrets_over_base(tmp_path):
    base = tmp_path / "config.toml"
    secrets = tmp_path / "secrets.toml"
    _write_toml(
        base,
        """\
[inference]
provider = "venice"
venice_api_key = ""
openrouter_api_key = ""
timeout = 30

[api]
port = 7777
internal_key = ""
""",
    )
    _write_toml(
        secrets,
        """\
[inference]
venice_api_key = "vk_placeholder_test"

[api]
internal_key = "internal_placeholder_test"
""",
    )

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", secrets
    ):
        cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg["inference"]["provider"] == "venice"
    assert cfg["inference"]["venice_api_key"] == "vk_placeholder_test"
    assert cfg["inference"]["openrouter_api_key"] == ""
    assert cfg["inference"]["timeout"] == 30
    assert cfg["api"]["port"] == 7777
    assert cfg["api"]["internal_key"] == "internal_placeholder_test"


def test_load_without_secrets_file_uses_base_and_warns(tmp_path, caplog):
    base = tmp_path / "config.toml"
    missing_secrets = tmp_path / "does_not_exist.toml"
    _write_toml(base, '[api]\nport = 7777\ninternal_key = ""\n')

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", missing_secrets
    ):
        with caplog.at_level("WARNING", logger="titan.config_loader"):
            cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg["api"]["port"] == 7777
    assert cfg["api"]["internal_key"] == ""
    assert any("secrets.toml" in r.message for r in caplog.records)


def test_load_without_base_returns_empty(tmp_path, caplog):
    base = tmp_path / "missing_config.toml"
    secrets = tmp_path / "missing_secrets.toml"

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", secrets
    ):
        with caplog.at_level("ERROR", logger="titan.config_loader"):
            cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg == {}


def test_load_with_malformed_secrets_falls_back_to_base(tmp_path, caplog):
    base = tmp_path / "config.toml"
    secrets = tmp_path / "secrets.toml"
    _write_toml(base, '[api]\nport = 7777\ninternal_key = ""\n')
    _write_toml(secrets, "this is not = valid toml [[[")

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", secrets
    ):
        with caplog.at_level("WARNING", logger="titan.config_loader"):
            cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg["api"]["port"] == 7777
    assert any("Failed to merge" in r.message for r in caplog.records)


def test_cache_is_reused(tmp_path):
    base = tmp_path / "config.toml"
    secrets = tmp_path / "secrets.toml"
    _write_toml(base, "[api]\nport = 7777\n")
    _write_toml(secrets, '[api]\ninternal_key = "k1"\n')

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", secrets
    ):
        cfg1 = config_loader.load_titan_config(force_reload=True)
        # Mutate the file — cached load should NOT reflect it.
        _write_toml(secrets, '[api]\ninternal_key = "k2"\n')
        cfg2 = config_loader.load_titan_config()
        assert cfg2["api"]["internal_key"] == "k1"
        # force_reload picks up the change.
        cfg3 = config_loader.load_titan_config(force_reload=True)
        assert cfg3["api"]["internal_key"] == "k2"


def test_update_secret_creates_file_and_writes_section(tmp_path):
    """update_secret should create secrets.toml and the section if absent."""
    secrets = tmp_path / "subdir" / "secrets.toml"  # parent doesn't exist yet
    base = tmp_path / "config.toml"
    _write_toml(base, "[api]\nport = 7777\n")

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "SECRETS_PATH", secrets), mock.patch.object(
        config_loader, "BASE_CONFIG_PATH", base
    ):
        ok = config_loader.update_secret("twitter_social", "auth_session", "new_sess_v1")
        assert ok is True
        assert secrets.exists()

        # Merged view now reflects the secret.
        cfg = config_loader.load_titan_config(force_reload=True)
        assert cfg["twitter_social"]["auth_session"] == "new_sess_v1"


def test_update_secret_preserves_existing_keys(tmp_path):
    """update_secret must preserve all other keys in secrets.toml."""
    secrets = tmp_path / "secrets.toml"
    base = tmp_path / "config.toml"
    _write_toml(base, "[api]\nport = 7777\n")
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

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "SECRETS_PATH", secrets), mock.patch.object(
        config_loader, "BASE_CONFIG_PATH", base
    ):
        ok = config_loader.update_secret("twitter_social", "auth_session", "new_sess_v2")
        assert ok is True

        cfg = config_loader.load_titan_config(force_reload=True)
        assert cfg["inference"]["venice_api_key"] == "vk_placeholder_preserved"
        assert cfg["inference"]["openrouter_api_key"] == "or_placeholder_preserved"
        assert cfg["twitter_social"]["password"] == "placeholder_preserved_pw"
        assert cfg["twitter_social"]["auth_session"] == "new_sess_v2"


def test_overlay_preserves_other_sections(tmp_path):
    """Overlay on one section must not wipe out other sections."""
    base = tmp_path / "config.toml"
    secrets = tmp_path / "secrets.toml"
    _write_toml(
        base,
        """\
[inference]
provider = "venice"
api_key = ""

[growth_metrics]
node_saturation_24h = 30
edge_rate_max = 100

[api]
port = 7777
""",
    )
    _write_toml(secrets, '[inference]\napi_key = "real_key"\n')

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), mock.patch.object(
        config_loader, "SECRETS_PATH", secrets
    ):
        cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg["inference"]["api_key"] == "real_key"
    assert cfg["inference"]["provider"] == "venice"
    assert cfg["growth_metrics"]["node_saturation_24h"] == 30
    assert cfg["growth_metrics"]["edge_rate_max"] == 100
    assert cfg["api"]["port"] == 7777
