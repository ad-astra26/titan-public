"""Tests for titan_plugin.config_loader deep-merge behavior."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_loader_cache(tmp_path_factory):
    """Ensure cache is clean between tests + isolate titan_params layer.

    The real titan_params.toml would be loaded as Layer 1 and pollute test
    expectations. Patching TITAN_PARAMS_PATH to a non-existent path makes
    Layer 1 empty by default for all tests — individual tests can override
    with their own mock.patch.object if they need titan_params behavior.
    """
    from titan_plugin import config_loader

    config_loader.clear_cache()
    fake_params = tmp_path_factory.mktemp("_no_params") / "titan_params.toml"
    with mock.patch.object(config_loader, "TITAN_PARAMS_PATH", fake_params):
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


def test_titan_params_layer1_merged_below_config(tmp_path):
    """Layer 1 (titan_params.toml) provides engineering defaults; Layer 2
    (config.toml) overrides them; Layer 3 (secrets) overrides those."""
    params = tmp_path / "titan_params.toml"
    base = tmp_path / "config.toml"
    secrets = tmp_path / "secrets.toml"
    _write_toml(
        params,
        """\
[filter_down_v5]
publish_enabled = true
cold_start_floor_epochs = 2000
spirit_filter_strength_multiplier = 0.3

[titan_self]
weight_felt = 1.0
weight_journey = 0.5
weight_topology = 0.3

[inference]
provider = "venice"
timeout = 30
""",
    )
    _write_toml(
        base,
        """\
[inference]
provider = "venice"
venice_api_key = ""
timeout = 60
""",
    )
    _write_toml(secrets, '[inference]\nvenice_api_key = "vk_placeholder_l3"\n')

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "TITAN_PARAMS_PATH", params), \
         mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), \
         mock.patch.object(config_loader, "SECRETS_PATH", secrets):
        cfg = config_loader.load_titan_config(force_reload=True)

    # Layer 1 only (not in config or secrets) flows through
    assert cfg["filter_down_v5"]["publish_enabled"] is True
    assert cfg["filter_down_v5"]["cold_start_floor_epochs"] == 2000
    assert cfg["titan_self"]["weight_felt"] == 1.0
    # Layer 2 overrides Layer 1 (timeout 30 → 60)
    assert cfg["inference"]["timeout"] == 60
    # Layer 3 overrides Layer 2 (api_key empty → placeholder)
    assert cfg["inference"]["venice_api_key"] == "vk_placeholder_l3"
    # Layer 1+2 agreement preserved
    assert cfg["inference"]["provider"] == "venice"


def test_titan_params_missing_is_non_fatal(tmp_path):
    """If titan_params.toml doesn't exist, Layer 1 is empty; config works."""
    missing_params = tmp_path / "no_params.toml"
    base = tmp_path / "config.toml"
    secrets = tmp_path / "no_secrets.toml"
    _write_toml(base, '[api]\nport = 7777\n')

    from titan_plugin import config_loader

    with mock.patch.object(config_loader, "TITAN_PARAMS_PATH", missing_params), \
         mock.patch.object(config_loader, "BASE_CONFIG_PATH", base), \
         mock.patch.object(config_loader, "SECRETS_PATH", secrets):
        cfg = config_loader.load_titan_config(force_reload=True)

    assert cfg["api"]["port"] == 7777


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


# ---------------------------------------------------------------------------
# titan_plugin.params delegates to config_loader (BUG-CONFIG-LOADER-MERGE-TITAN-PARAMS)
# ---------------------------------------------------------------------------

def test_params_get_params_returns_merged_layers(tmp_path):
    """``params.get_params(section)`` must return the merged config view.

    Pre-fix, ``params.py`` read ``titan_params.toml`` directly and skipped
    Layers 2-4. After the fix it delegates to ``config_loader.load_titan_config``
    so a section value written in config.toml (Layer 2) overrides
    titan_params.toml (Layer 1) and is visible to ``get_params``.
    """
    titan_params = tmp_path / "titan_params.toml"
    base = tmp_path / "config.toml"
    _write_toml(
        titan_params,
        """\
[reflexes]
fire_threshold = 0.10
cooldown_ms = 250
""",
    )
    _write_toml(
        base,
        """\
[reflexes]
fire_threshold = 0.20
""",
    )

    from titan_plugin import config_loader, params as params_mod

    config_loader.clear_cache()
    with mock.patch.object(config_loader, "TITAN_PARAMS_PATH", titan_params), mock.patch.object(
        config_loader, "BASE_CONFIG_PATH", base
    ):
        section = params_mod.get_params("reflexes")

    # Layer 2 (config.toml) overrides Layer 1 (titan_params.toml) — fire_threshold
    assert section["fire_threshold"] == 0.20
    # Layer 1 keys not overridden in Layer 2 must still be visible
    assert section["cooldown_ms"] == 250
    # Returned dict is a copy — mutation does not affect cache
    section["fire_threshold"] = 99.0
    section_again = params_mod.get_params("reflexes")
    assert section_again["fire_threshold"] == 0.20


def test_params_get_params_unknown_section_returns_empty_dict(tmp_path):
    """Unknown section returns ``{}`` (preserves pre-fix behavior)."""
    titan_params = tmp_path / "titan_params.toml"
    base = tmp_path / "config.toml"
    _write_toml(titan_params, "[reflexes]\nfire_threshold = 0.10\n")
    _write_toml(base, "[api]\nport = 7777\n")

    from titan_plugin import config_loader, params as params_mod

    config_loader.clear_cache()
    with mock.patch.object(config_loader, "TITAN_PARAMS_PATH", titan_params), mock.patch.object(
        config_loader, "BASE_CONFIG_PATH", base
    ):
        section = params_mod.get_params("nonexistent_section")
    assert section == {}
