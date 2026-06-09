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
