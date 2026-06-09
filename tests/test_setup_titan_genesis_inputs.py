"""Offline tests for the genesis identity-collection phase (RFP_genesis_ceremony
Phase A). Non-interactive paths only (pytest stdin is not a TTY → _interactive is
False), so no $EDITOR / prompts fire — we verify the config writes + the gating.
"""
import os
import sys

import pytest

# Append (NOT insert-at-0): scripts/ contains a titan_hcl.py that would shadow
# the real titan_hcl package (conftest's autouse fixture imports it).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from setup_titan import genesis_inputs as gi   # noqa: E402
from setup_titan.modes import Mode             # noqa: E402
from setup_titan.prompts import StdinPrompter  # noqa: E402
from setup_titan.config_model import parse_toml_with_comments  # noqa: E402

DEPLOYER = "YOUR_DEPLOYER_PUBKEY"  # valid base58 pubkey


def _make_install_root(tmp_path):
    cfg_dir = tmp_path / "titan_hcl"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(
        '[network]\nmaker_pubkey = "YOUR_MAKER_PUBKEY"\n\n'
        '[genesis]\ntitan_name = "YOUR_TITAN_NAME"\n')
    return tmp_path


def _cfg_value(install_root, dotted):
    for e in parse_toml_with_comments(install_root / "titan_hcl" / "config.toml"):
        if e.dotted == dotted:
            return e.raw_value.strip().strip('"')
    return None


# ── validation helpers ────────────────────────────────────────────────────────

def test_valid_pubkey():
    assert gi._valid_pubkey(DEPLOYER) is True
    assert gi._valid_pubkey("too-short") is False
    assert gi._valid_pubkey("0OIl" + "1" * 40) is False  # 0/O/I/l not in base58
    assert gi._valid_pubkey("") is False


def test_has_directive_content(tmp_path):
    f = tmp_path / "c.md"
    f.write_text("# only comments\n#\n   \n")
    assert gi._has_directive_content(f) is False
    f.write_text("# header\n1. Sovereign Growth.\n")
    assert gi._has_directive_content(f) is True
    assert gi._has_directive_content(tmp_path / "missing.md") is False


# ── phase behaviour ───────────────────────────────────────────────────────────

def test_local_mode_skips(tmp_path):
    root = _make_install_root(tmp_path)
    res = gi.run_genesis_inputs_phase(
        root, Mode.LOCAL, {}, prompter=StdinPrompter(), default=True)
    assert len(res) == 1 and res[0].severity == "ok"
    # untouched
    assert _cfg_value(root, "network.maker_pubkey") == "YOUR_MAKER_PUBKEY"


def test_devnet_persists_maker_and_name(tmp_path):
    root = _make_install_root(tmp_path)
    state = {"maker_wallet": DEPLOYER, "titan_id": "T4"}
    res = gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, state, prompter=StdinPrompter(), default=True)
    by = {r.name: r for r in res}
    assert by["maker_pubkey"].severity == "ok"
    assert _cfg_value(root, "network.maker_pubkey") == DEPLOYER
    # name defaults to titan_id (T4) in non-interactive mode
    assert by["genesis_name"].severity == "ok"
    assert _cfg_value(root, "genesis.titan_name") == "T4"
    # no constitution → directives warn (mandatory, not defaulted)
    assert by["directives"].severity == "warn"


def test_devnet_warns_without_maker_wallet(tmp_path):
    root = _make_install_root(tmp_path)
    res = gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, {}, prompter=StdinPrompter(), default=True)
    by = {r.name: r for r in res}
    assert by["maker_pubkey"].severity == "warn"
    assert _cfg_value(root, "network.maker_pubkey") == "YOUR_MAKER_PUBKEY"


def test_devnet_directives_ok_when_constitution_present(tmp_path):
    root = _make_install_root(tmp_path)
    (root / "titan_constitution.md").write_text(
        "# directives\n1. Sovereign Growth.\n2. Truthfulness.\n")
    res = gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, {"maker_wallet": DEPLOYER}, prompter=StdinPrompter(),
        default=True)
    by = {r.name: r for r in res}
    assert by["directives"].severity == "ok"


def test_missing_config_fails(tmp_path):
    res = gi.run_genesis_inputs_phase(
        tmp_path, Mode.DEVNET, {"maker_wallet": DEPLOYER},
        prompter=StdinPrompter(), default=True)
    assert len(res) == 1 and res[0].severity == "fail"


# ── headless directive supply: --directives-file / TITAN_DIRECTIVES[_FILE] ─────
# Precedence (highest first): --directives-file (flag, path) ›
# TITAN_DIRECTIVES_FILE (env, path) › TITAN_DIRECTIVES (env, literal text).

REAL = "# Prime Directives\n1. Sovereign Growth.\n2. Truthfulness.\n"


def _constitution(root):
    return root / "titan_constitution.md"


def _run(root, *, monkeypatch=None, **kw):
    # Default to a clean env so a stray runner var can't leak into the supply path.
    if monkeypatch is not None:
        monkeypatch.delenv("TITAN_DIRECTIVES_FILE", raising=False)
        monkeypatch.delenv("TITAN_DIRECTIVES", raising=False)
    return {r.name: r for r in gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, {"maker_wallet": DEPLOYER},
        prompter=StdinPrompter(), default=True, **kw)}


def test_directives_file_flag_writes_and_oks(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    src = tmp_path / "dir.md"
    src.write_text(REAL)
    by = _run(root, monkeypatch=monkeypatch, directives_file=str(src))
    assert by["directives"].severity == "ok"
    assert _constitution(root).read_text() == REAL  # written verbatim
    assert "--directives-file" in by["directives"].detail


def test_directives_file_comment_only_warns_no_fabrication(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    src = tmp_path / "dir.md"
    src.write_text("# only a comment\n#\n   \n")
    by = _run(root, monkeypatch=monkeypatch, directives_file=str(src))
    assert by["directives"].severity == "warn"
    assert "comments" in by["directives"].detail.lower()


def test_directives_file_missing_path_warns_and_writes_nothing(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    by = _run(root, monkeypatch=monkeypatch, directives_file=str(tmp_path / "nope.md"))
    assert by["directives"].severity == "warn"
    assert "not found" in by["directives"].detail
    assert not _constitution(root).exists()


def test_env_file_path_oks(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    src = tmp_path / "envdir.md"
    src.write_text(REAL)
    monkeypatch.setenv("TITAN_DIRECTIVES_FILE", str(src))
    by = {r.name: r for r in gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, {"maker_wallet": DEPLOYER},
        prompter=StdinPrompter(), default=True)}
    assert by["directives"].severity == "ok"
    assert "TITAN_DIRECTIVES_FILE" in by["directives"].detail
    assert _constitution(root).read_text() == REAL


def test_env_literal_text_oks_and_newline_normalised(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    monkeypatch.delenv("TITAN_DIRECTIVES_FILE", raising=False)
    monkeypatch.setenv("TITAN_DIRECTIVES", "1. Sovereign Growth.\n2. Truthfulness.")
    by = {r.name: r for r in gi.run_genesis_inputs_phase(
        root, Mode.DEVNET, {"maker_wallet": DEPLOYER},
        prompter=StdinPrompter(), default=True)}
    assert by["directives"].severity == "ok"
    assert "TITAN_DIRECTIVES" in by["directives"].detail
    text = _constitution(root).read_text()
    assert text.endswith("\n")  # trailing newline added
    assert gi._has_directive_content(_constitution(root))


def test_precedence_flag_beats_env(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    flag_src = tmp_path / "flag.md"
    flag_src.write_text("1. FLAG directive.\n")
    envfile = tmp_path / "envfile.md"
    envfile.write_text("1. ENVFILE directive.\n")
    monkeypatch.setenv("TITAN_DIRECTIVES_FILE", str(envfile))
    monkeypatch.setenv("TITAN_DIRECTIVES", "1. ENVTEXT directive.")
    _run(root, directives_file=str(flag_src))
    assert "FLAG directive" in _constitution(root).read_text()


def test_precedence_envfile_beats_envtext(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    envfile = tmp_path / "envfile.md"
    envfile.write_text("1. ENVFILE directive.\n")
    monkeypatch.setenv("TITAN_DIRECTIVES_FILE", str(envfile))
    monkeypatch.setenv("TITAN_DIRECTIVES", "1. ENVTEXT directive.")
    _run(root)
    assert "ENVFILE directive" in _constitution(root).read_text()


def test_explicit_supply_overwrites_existing(tmp_path, monkeypatch):
    root = _make_install_root(tmp_path)
    _constitution(root).write_text("# old\n1. OLD directive.\n")
    src = tmp_path / "new.md"
    src.write_text("1. NEW directive.\n")
    _run(root, monkeypatch=monkeypatch, directives_file=str(src))
    out = _constitution(root).read_text()
    assert "NEW directive" in out and "OLD" not in out


def test_supplied_directives_resolver_unit(tmp_path, monkeypatch):
    monkeypatch.delenv("TITAN_DIRECTIVES_FILE", raising=False)
    monkeypatch.delenv("TITAN_DIRECTIVES", raising=False)
    assert gi._supplied_directives(None) == (None, "", None)   # nothing → fall through
    monkeypatch.setenv("TITAN_DIRECTIVES", "1. X.")
    text, label, problem = gi._supplied_directives(None)
    assert label == "TITAN_DIRECTIVES" and problem is None and "X" in text
