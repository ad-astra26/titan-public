"""Tests for the setup_titan install wizard (W1).

Pure-logic coverage — no live network, no systemd, no real config files
(config.toml carries live secrets; every fixture here is synthetic). Run:

    python -m pytest tests/test_setup_titan.py -v -p no:anchorpy
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

# The wizard is invoked as `python -m scripts.setup_titan`; put scripts/ on path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from setup_titan import (  # noqa: E402
    binaries,
    config_model as cm,
    config_seed,
    genesis_runner,
    inference,
    observatory,
    systemd_runner,
)
from setup_titan.__main__ import build_parser  # noqa: E402
from setup_titan.modes import Mode  # noqa: E402


# ── config_model ─────────────────────────────────────────────────────────────

SAMPLE_TOML = """\
# file banner
[mood_engine]
base_weight = 1.0
update_interval_seconds = 21600 # 6 hours (Meditation Cycle)

[addons]
# List filenames from the /addons folder
active = [
    "weather_vibe",
    "bonk_pulse"
]

[net]
# the host to bind
host = "http://localhost:8080"
enabled = true
"""


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(SAMPLE_TOML)
    return p


def test_parse_pairs_keys_with_their_own_comments(sample: Path):
    by = {e.dotted: e for e in cm.parse_toml_with_comments(sample)}
    # inline comment becomes help
    assert by["mood_engine.update_interval_seconds"].help == "6 hours (Meditation Cycle)"
    # preceding comment becomes help
    assert by["net.host"].help == "the host to bind"
    assert by["net.host"].raw_value == '"http://localhost:8080"'
    assert by["net.enabled"].raw_value == "true"


def test_multiline_array_is_readonly_and_body_not_leaked(sample: Path):
    by = {e.dotted: e for e in cm.parse_toml_with_comments(sample)}
    assert by["addons.active"].editable is False
    # the array elements must not be parsed as keys
    assert "weather_vibe" not in by
    assert not any(e.key == "weather_vibe" for e in cm.parse_toml_with_comments(sample))


def test_coerce_like_matches_old_shape():
    assert cm.coerce_like('"old"', "new") == '"new"'
    assert cm.coerce_like('"old"', '"already-quoted"') == '"already-quoted"'
    assert cm.coerce_like("true", "FALSE") == "false"
    assert cm.coerce_like("42", "100") == "100"
    with pytest.raises(ValueError):
        cm.coerce_like("true", "maybe")


def test_set_value_preserves_comment_and_key(sample: Path):
    by = {e.dotted: e for e in cm.parse_toml_with_comments(sample)}
    e = by["mood_engine.update_interval_seconds"]
    cm.set_value(sample, e.lineno, "43200")
    after = {x.dotted: x for x in cm.parse_toml_with_comments(sample)}
    assert after["mood_engine.update_interval_seconds"].raw_value == "43200"
    assert after["mood_engine.update_interval_seconds"].help == "6 hours (Meditation Cycle)"


def test_set_by_dotted_quotes_string_and_misses_unknown(sample: Path):
    assert cm.set_by_dotted(sample, "net.host", "http://example:9000") is True
    by = {e.dotted: e for e in cm.parse_toml_with_comments(sample)}
    assert by["net.host"].raw_value == '"http://example:9000"'
    assert by["net.host"].help == "the host to bind"          # comment survives
    assert cm.set_by_dotted(sample, "net.does_not_exist", "x") is False


# ── inference: section-aware secrets upsert + read (the #9 fix) ───────────────

import tomllib  # noqa: E402


def test_upsert_secret_creates_section_then_replaces_in_place(tmp_path: Path):
    sp = tmp_path / "secrets.toml"
    inference.upsert_secret("inference", "openrouter_api_key", "sk-or-aaa", path=sp)
    inference.upsert_secret("channels", "telegram_bot_token", "123:abc", path=sp)
    # replace an existing key within its section (must not duplicate)
    inference.upsert_secret("inference", "openrouter_api_key", "sk-or-bbb", path=sp)
    data = tomllib.load(open(sp, "rb"))
    assert data["inference"]["openrouter_api_key"] == "sk-or-bbb"
    assert data["channels"]["telegram_bot_token"] == "123:abc"
    assert sp.read_text().count("openrouter_api_key") == 1     # replaced, not appended
    assert oct(sp.stat().st_mode)[-3:] == "600"


def test_upsert_secret_appends_key_into_existing_section(tmp_path: Path):
    sp = tmp_path / "secrets.toml"
    inference.upsert_secret("inference", "openrouter_api_key", "sk-or-aaa", path=sp)
    inference.upsert_secret("inference", "venice_api_key", "vv", path=sp)
    data = tomllib.load(open(sp, "rb"))
    assert data["inference"] == {"openrouter_api_key": "sk-or-aaa", "venice_api_key": "vv"}


def test_upsert_secret_escapes_quotes(tmp_path: Path):
    sp = tmp_path / "secrets.toml"
    inference.upsert_secret("twitter_social", "webshare_static_url", 'http://u:p"x@h:1/', path=sp)
    data = tomllib.load(open(sp, "rb"))
    assert data["twitter_social"]["webshare_static_url"] == 'http://u:p"x@h:1/'


def test_read_secret_roundtrip_and_missing(tmp_path: Path):
    sp = tmp_path / "secrets.toml"
    assert inference.read_secret("api", "internal_key", path=sp) is None   # no file
    inference.upsert_secret("api", "internal_key", "tok", path=sp)
    assert inference.read_secret("api", "internal_key", path=sp) == "tok"
    assert inference.read_secret("api", "absent", path=sp) is None


@pytest.mark.parametrize("provider,secret_key", [
    ("ollama_cloud", "ollama_cloud_api_key"),
    ("openrouter", "openrouter_api_key"),
])
def test_wire_cloud_key_writes_sectioned_secret_and_sets_provider(tmp_path: Path, provider, secret_key):
    # mini repo with a seeded config.toml carrying an [inference] provider key
    (tmp_path / "titan_hcl").mkdir()
    (tmp_path / "titan_hcl" / "config.toml").write_text(
        "[inference]\ninference_provider = \"venice\"\n")
    sp = tmp_path / "secrets.toml"
    res = inference._wire_cloud_key(tmp_path, provider, "K" * 40, secrets_path=sp)
    assert all(r.severity == "ok" for r in res)
    import tomllib as _t
    assert _t.load(open(sp, "rb"))["inference"][secret_key] == "K" * 40
    cfg = {e.dotted: e.raw_value for e in cm.parse_toml_with_comments(tmp_path / "titan_hcl" / "config.toml")}
    assert cfg["inference.inference_provider"] == f'"{provider}"'


# ── config_seed: required config.toml + minted internal_key (the #8 fix) ──────


def _mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "titan_hcl").mkdir()
    (tmp_path / "titan_hcl" / "config.toml.example").write_text(
        "[inference]\ninference_provider = \"ollama_cloud\"\n[api]\ninternal_key = \"\"\n")
    return tmp_path


def test_config_seed_copies_example_and_mints_key(tmp_path: Path):
    root = _mini_repo(tmp_path)
    sp = tmp_path / "secrets.toml"
    res = config_seed.run_config_seed_phase(root, secrets_path=sp)
    assert all(r.severity == "ok" for r in res)
    assert config_seed.config_path(root).exists()                 # seeded
    key = inference.read_secret("api", "internal_key", path=sp)
    assert key and len(key) >= 32                                 # minted


def test_config_seed_is_idempotent(tmp_path: Path):
    root = _mini_repo(tmp_path)
    sp = tmp_path / "secrets.toml"
    config_seed.run_config_seed_phase(root, secrets_path=sp)
    first = inference.read_secret("api", "internal_key", path=sp)
    # second run must NOT clobber the existing config or regenerate the key
    config_seed.config_path(root).write_text("[api]\ninternal_key = \"\"\n# user edit\n")
    config_seed.run_config_seed_phase(root, secrets_path=sp)
    assert inference.read_secret("api", "internal_key", path=sp) == first
    assert "# user edit" in config_seed.config_path(root).read_text()


def test_config_seed_fails_without_example(tmp_path: Path):
    (tmp_path / "titan_hcl").mkdir()
    res = config_seed.run_config_seed_phase(tmp_path, secrets_path=tmp_path / "s.toml")
    assert res[0].severity == "fail"


# ── binaries ─────────────────────────────────────────────────────────────────


def test_nine_daemons_match_release_workflow():
    assert len(binaries.DAEMONS) == 9
    assert "titan-kernel-rs" in binaries.DAEMONS


def test_parse_sha256sums():
    parsed = binaries.parse_sha256sums("abc  titan-kernel-rs\nDEF  titan-trinity-rs\n\n")
    assert parsed == {"titan-kernel-rs": "abc", "titan-trinity-rs": "def"}


def test_fetch_refuses_untagged(tmp_path: Path):
    for ref in ("main", "HEAD", ""):
        res = binaries.fetch_release_binaries(tmp_path, ref)
        assert res[0].severity == "fail"
        assert "no release tag" in res[0].detail


def test_binaries_phase_skips_when_present(tmp_path: Path):
    bd = tmp_path / "bin"
    bd.mkdir()
    for d in binaries.DAEMONS:
        (bd / d).write_text("stub")
    res = binaries.run_binaries_phase(tmp_path, tag="main", build_rust=False)
    assert res[0].severity == "ok" and "already present" in res[0].detail


# ── genesis_runner: bootable-identity materialization (the #6 fix) ────────────


def test_materialize_bootable_identity_from_authority(tmp_path: Path):
    # genesis (kept plaintext) leaves authority.json + data/genesis_record.json
    (tmp_path / "authority.json").write_text("[1,2,3,4]")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "genesis_record.json").write_text('{"titan_pubkey": "ZeFUoD…"}')
    res = genesis_runner._materialize_bootable_identity(tmp_path)
    assert res[0].severity == "ok"
    kp = genesis_runner.keypair_path(tmp_path)
    assert kp.exists() and kp.read_text() == "[1,2,3,4]"
    assert oct(kp.stat().st_mode)[-3:] == "600"          # 0600 like T2/T3
    assert not (tmp_path / "authority.json").exists()     # stray root copy wiped
    ident = json.loads(genesis_runner.identity_path(tmp_path).read_text())
    assert ident["titan_id"] == "T1" and ident["titan_pubkey"] == "ZeFUoD…"


def test_materialize_fails_without_keypair(tmp_path: Path):
    res = genesis_runner._materialize_bootable_identity(tmp_path)
    assert res[0].severity == "fail" and "no plaintext keypair" in res[0].detail


# ── systemd_runner ───────────────────────────────────────────────────────────


def test_render_unit_uses_install_root_and_id(tmp_path: Path):
    unit = systemd_runner.render_unit(install_root=Path("/srv/titan"), titan_id="T1", user="bob")
    assert "ExecStart=/srv/titan/bin/titan-kernel-rs --titan-id T1" in unit
    assert "ConditionPathExists=/srv/titan/data/titan_identity_keypair.json" in unit
    assert "User=bob" in unit
    assert "Environment=TITAN_ID=T1" in unit
    assert "WantedBy=multi-user.target" in unit


def test_resolve_titan_id_falls_back_to_T1(tmp_path: Path):
    assert systemd_runner.resolve_install_titan_id(tmp_path) == "T1"
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "titan_identity.json").write_text('{"titan_id": "T7"}')
    assert systemd_runner.resolve_install_titan_id(tmp_path) == "T7"


# ── observatory (the #15 prebuilt-bundle phase) ──────────────────────────────


def test_observatory_bundle_name_and_fetch_refuses_untagged(tmp_path: Path):
    assert observatory.bundle_name("v0.0.3") == "titan-observatory-v0.0.3.tar.gz"
    for ref in ("main", "HEAD", ""):
        res = observatory.fetch_observatory_bundle(tmp_path, ref)
        assert res[0].severity == "fail" and "no release tag" in res[0].detail


def test_observatory_phase_skips_when_not_enabled(tmp_path: Path):
    res = observatory.run_observatory_phase({}, tmp_path, tag="v0.0.3", user="bob")
    assert res[0].severity == "ok" and "skipped" in res[0].detail
    res2 = observatory.run_observatory_phase({"observatory_enabled": False}, tmp_path,
                                             tag="v0.0.3", user="bob")
    assert res2[0].severity == "ok" and "skipped" in res2[0].detail


def test_observatory_unit_binds_localhost_and_runs_server(tmp_path: Path):
    unit = observatory.render_observatory_unit(app_path=Path("/srv/obs"), user="bob", port=3000)
    assert "server.js" in unit and "WorkingDirectory=/srv/obs" in unit
    assert "Environment=HOSTNAME=127.0.0.1" in unit       # localhost-only by default
    assert "Environment=PORT=3000" in unit
    assert "User=bob" in unit
    assert "After=titan.service" in unit                  # starts after the brain


# ── CLI surface ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [
    ["install", "--mode", "local", "--dry-run"],
    ["install", "--default", "--tag", "v0.0.1", "--build-rust"],
    ["config", "--list"],
    ["config", "--get", "net.host"],
    ["config", "--set", "net.host=x"],
    ["diagnostic"],
    ["upgrade", "--tag", "v0.0.2"],
    ["repair"],
    ["uninstall", "--purge", "--yes"],
])
def test_all_subcommands_parse(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_mode_specs_consistent():
    # local needs no chain toolchain; mainnet/devnet do
    from setup_titan.modes import spec_for
    assert spec_for(Mode.LOCAL).genesis_on_chain is False
    assert spec_for(Mode.LOCAL).needs_rust is False
    assert spec_for(Mode.MAINNET).genesis_on_chain is True
    assert spec_for(Mode.DEVNET).needs_anchor is True
