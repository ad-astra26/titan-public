"""Tests for the setup_titan install wizard (W1).

Pure-logic coverage — no live network, no systemd, no real config files
(config.toml carries live secrets; every fixture here is synthetic). Run:

    python -m pytest tests/test_setup_titan.py -v -p no:anchorpy
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The wizard is invoked as `python -m scripts.setup_titan`; put scripts/ on path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from setup_titan import binaries, config_model as cm, systemd_runner  # noqa: E402
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
