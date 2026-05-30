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
# Append (not insert-at-0): scripts/ contains a titan_hcl.py that would
# otherwise shadow the real titan_hcl *package* (conftest's autouse fixture
# imports titan_hcl.persistence), breaking this suite when run in isolation.
_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO / "scripts"))

from setup_titan import (  # noqa: E402
    binaries,
    config_model as cm,
    config_seed,
    console as console_phase,
    genesis_runner,
    inference,
    observatory,
    restore as restore_mod,
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
    # genesis (kept plaintext) leaves authority.json (a real 64-int Ed25519
    # keypair array) + data/genesis_record.json
    key_arr = list(range(64))
    (tmp_path / "authority.json").write_text(json.dumps(key_arr))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "genesis_record.json").write_text('{"titan_pubkey": "ZeFUoD…"}')
    res = genesis_runner._materialize_bootable_identity(tmp_path)
    assert res[0].severity == "ok"
    kp = genesis_runner.keypair_path(tmp_path)
    assert kp.exists() and json.loads(kp.read_text()) == key_arr   # 64-int array
    assert oct(kp.stat().st_mode)[-3:] == "600"          # 0600 like T2/T3
    assert not (tmp_path / "authority.json").exists()     # stray root copy wiped
    ident = json.loads(genesis_runner.identity_path(tmp_path).read_text())
    assert ident["titan_id"] == "T1" and ident["titan_pubkey"] == "ZeFUoD…"


def test_materialize_fails_without_keypair(tmp_path: Path):
    res = genesis_runner._materialize_bootable_identity(tmp_path)
    assert res[0].severity == "fail" and "no plaintext keypair" in res[0].detail


# ── setup_titan restore — the W1.5 resurrection wrapper ──────────────────────


def test_restore_subcommand_parses():
    from setup_titan.__main__ import build_parser, cmd_restore
    ns = build_parser().parse_args(
        ["restore", "--shard1", "deadbeef", "--manifest", "/tmp/m.json",
         "--verify-only", "--titan-id", "T1"])
    assert ns.func is cmd_restore
    assert ns.shard1 == "deadbeef" and ns.manifest == "/tmp/m.json"
    assert ns.verify_only is True and ns.titan_id == "T1"
    assert ns.network == "mainnet" and ns.verify_zk is False


def test_run_restore_aborts_on_empty_shard(tmp_path, monkeypatch):
    # no shard arg + getpass returns empty → clean abort, no resurrection call.
    monkeypatch.setattr(restore_mod.getpass, "getpass", lambda *a, **k: "")
    called = {"load": False}
    monkeypatch.setattr(restore_mod, "_load_resurrection",
                        lambda: called.__setitem__("load", True))
    rc = restore_mod.run_restore(tmp_path)
    assert rc == 1 and called["load"] is False


def test_run_restore_threads_through_to_resurrection_phases(tmp_path, monkeypatch):
    """The wrapper must call phase_1 → phase_2_3 → phase_4 with the resolved
    identity, never logging the shard."""
    calls = []

    class _FakeResurrection:
        @staticmethod
        def phase_1_identity(args, install_root):
            calls.append(("p1", args.shard1, install_root))
            return (b"\x01" * 64, "PUBKEY123", None, "T1")

        @staticmethod
        def phase_2_3_restore(key_bytes, pubkey, titan_id, **kw):
            calls.append(("p23", pubkey, titan_id, kw["manifest_path"],
                          kw["network"], kw["verify_zk"]))

        @staticmethod
        def phase_4_first_breath(key_bytes, pubkey, titan_id, **kw):
            calls.append(("p4", pubkey, kw["verify_only"]))

    monkeypatch.setattr(restore_mod, "_load_resurrection", lambda: _FakeResurrection)
    rc = restore_mod.run_restore(
        tmp_path, shard1="ABCD", manifest="/tmp/m.json",
        network="mainnet", verify_only=True)
    assert rc == 0
    assert calls[0] == ("p1", "ABCD", str(tmp_path))
    assert calls[1] == ("p23", "PUBKEY123", "T1", "/tmp/m.json", "mainnet", False)
    assert calls[2] == ("p4", "PUBKEY123", True)


def test_run_restore_propagates_phase_failure(tmp_path, monkeypatch):
    class _Failing:
        @staticmethod
        def phase_1_identity(args, install_root):
            raise SystemExit(1)

    monkeypatch.setattr(restore_mod, "_load_resurrection", lambda: _Failing)
    rc = restore_mod.run_restore(tmp_path, shard1="ABCD", manifest="/tmp/m.json")
    assert rc == 1


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


# ── W1.b.2 — Prompter seam (shared by CLI stdin + Textual TUI) ──────────────

from setup_titan import comms, prompts  # noqa: E402
from setup_titan.prompts import Prompter, ScriptedPrompter, StdinPrompter  # noqa: E402

_TG = "12345678:" + "A" * 30           # valid BotFather-shaped token
_UUID = "12345678-1234-1234-1234-123456789abc"
_WEBSHARE = "http://user:pass@1.2.3.4:8080/"


def test_both_prompters_satisfy_protocol():
    assert isinstance(StdinPrompter(), Prompter)
    assert isinstance(ScriptedPrompter({}), Prompter)


def test_stdin_prompter_line_uses_default_when_blank(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p: "")
    assert StdinPrompter().line("k", "RPC", default="https://x") == "https://x"
    monkeypatch.setattr("builtins.input", lambda _p: "  https://y  ")
    assert StdinPrompter().line("k", "RPC", default="https://x") == "https://y"


def test_stdin_prompter_confirm_default_and_explicit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p: "")
    assert StdinPrompter().confirm("k", "Use it?", default_yes=True) is True
    assert StdinPrompter().confirm("k", "Use it?", default_yes=False) is False
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert StdinPrompter().confirm("k", "Use it?", default_yes=True) is False


def test_stdin_prompter_until_loops_then_accepts(monkeypatch):
    seq = iter(["bad", "alsobad", "sk-or-12345678"])
    monkeypatch.setattr("builtins.input", lambda _p: next(seq))
    out = StdinPrompter().until("k", "Key", validate=lambda s: s.startswith("sk-or-"),
                                hint="nope")
    assert out == "sk-or-12345678"


def test_stdin_prompter_eof_raises_systemexit(monkeypatch):
    def _raise(_p):
        raise EOFError
    monkeypatch.setattr("builtins.input", _raise)
    with pytest.raises(SystemExit):
        StdinPrompter().line("k", "x")


def test_scripted_prompter_returns_and_validates():
    sp = ScriptedPrompter({"a": "v", "b": True, "key": "sk-or-abcdefgh"})
    assert sp.line("a", "?") == "v"
    assert sp.confirm("b", "?", default_yes=False) is True
    assert sp.choice("a", "?", options=["v", "w"], default="w") == "v"
    assert sp.until("key", "?", validate=lambda s: s.startswith("sk-or-"), hint="h") == "sk-or-abcdefgh"


def test_scripted_prompter_missing_key_raises():
    with pytest.raises(KeyError):
        ScriptedPrompter({}).line("absent", "?")


def test_scripted_prompter_until_rejects_invalid_value():
    with pytest.raises(ValueError):
        ScriptedPrompter({"k": "garbage"}).until("k", "?", validate=lambda s: False, hint="bad")


def _capture_secrets(monkeypatch):
    """Redirect comms.upsert_secret into a recorder (never touch ~/.titan)."""
    rec: list[tuple] = []
    monkeypatch.setattr(comms, "upsert_secret", lambda *a, **k: rec.append(a))
    return rec


def test_comms_phase_via_scripted_telegram_only(monkeypatch):
    rec = _capture_secrets(monkeypatch)
    state: dict = {}
    sp = ScriptedPrompter({"telegram_bot_token": _TG, "enable_x": False,
                           "enable_observatory": False})
    results = comms.run_comms_phase(default=False, state=state, prompter=sp)
    names = {r.name: r for r in results}
    assert names["telegram"].severity == "ok"
    assert names["x_social"].detail.startswith("skipped")
    assert state["observatory_enabled"] is False
    assert ("channels", "telegram_bot_token", _TG) in rec


def test_comms_phase_via_scripted_full_optins(monkeypatch):
    rec = _capture_secrets(monkeypatch)
    state: dict = {}
    sp = ScriptedPrompter({"telegram_bot_token": _TG, "enable_x": True,
                           "twitterapi_key": _UUID, "webshare_url": _WEBSHARE,
                           "enable_observatory": True})
    results = comms.run_comms_phase(default=False, state=state, prompter=sp)
    names = {r.name: r for r in results}
    assert names["x_social"].severity == "ok"
    assert state["observatory_enabled"] is True
    assert ("stealth_sage", "twitterapi_io_key", _UUID) in rec
    assert ("twitter_social", "webshare_static_url", _WEBSHARE) in rec


def test_install_accepts_no_tui_flag():
    args = build_parser().parse_args(["install", "--no-tui", "--mode", "local"])
    assert args.no_tui is True


# ── W1.b.2 — Textual wizard headless pilot (collect-then-execute front-end) ──

import asyncio  # noqa: E402


def _drive_wizard(mode_id: str, fills: dict):
    """Mount the wizard headless, select a mode, fill inputs, press Begin.

    Returns the App.return_value: (mode, answers, state_seed) or None if the
    submit was blocked by validation. Robust to whether a local Ollama is up
    (inf_key is filled with a valid hosted-key shape but ignored when hidden).
    """
    from textual.widgets import Input, RadioButton, Button
    from setup_titan.tui import InstallWizard

    async def _run():
        app = InstallWizard()
        async with app.run_test(size=(120, 50)) as pilot:
            app.query_one("#" + mode_id, RadioButton).value = True
            await pilot.pause()
            for wid, val in fills.items():
                app.query_one("#" + wid, Input).value = val
            await pilot.pause()
            app.query_one("#begin", Button).scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#begin")
            await pilot.pause()
        return app.return_value

    return asyncio.run(_run())


def test_wizard_local_submit_collects_answers():
    res = _drive_wizard("mode_local",
                        {"telegram": _TG, "inf_key": "x" * 25})
    assert res is not None, "local-mode submit should succeed"
    mode, answers, seed = res
    assert mode is Mode.LOCAL
    assert answers["telegram_bot_token"] == _TG
    assert seed == {}                      # local mode collects no on-chain creds
    assert answers["enable_x"] is False
    assert answers["enable_observatory"] is False


def test_wizard_devnet_seeds_wallet_and_rpc():
    res = _drive_wizard("mode_devnet",
                        {"wallet": "4Nd1mYn" + "A" * 30,
                         "rpc": "https://api.devnet.solana.com",
                         "telegram": _TG, "inf_key": "y" * 25})
    assert res is not None
    _mode, _answers, seed = res
    assert seed["maker_wallet"].startswith("4Nd1mYn")
    assert seed["solana_rpc"] == "https://api.devnet.solana.com"


def test_wizard_devnet_blocks_without_wallet():
    # on-chain mode with no wallet → validation refuses to submit (returns None)
    res = _drive_wizard("mode_devnet", {"telegram": _TG, "inf_key": "y" * 25})
    assert res is None


# ── console phase (W8 — TC² install) ────────────────────────────────────────


def test_console_render_unit_fills_all_placeholders(tmp_path: Path):
    u = console_phase.render_unit(
        install_root=tmp_path, user="bob", venv_python="/usr/bin/python3",
        port=7799, api_base="http://127.0.0.1:7777", bind_host="127.0.0.1")
    assert "{{" not in u and "}}" not in u
    assert "bob" in u and "7799" in u and str(tmp_path) in u
    assert "-m titan_console" in u
    # decoupling: not Guardian/Titan-coupled — no After=titan.service dependency
    assert "Wants=titan" not in u


def test_console_ensure_bundle_detects_prebuilt(tmp_path: Path):
    spa = tmp_path / "titan-console" / "dist"
    spa.mkdir(parents=True)
    (spa / "index.html").write_text("<!doctype html>")
    r = console_phase.ensure_bundle(tmp_path)
    assert r.severity == "ok" and "present" in r.detail


def test_console_ensure_token_idempotent(tmp_path: Path, monkeypatch):
    tok = tmp_path / "console_token"
    monkeypatch.setattr(console_phase, "TOKEN_PATH", tok)
    r1 = console_phase.ensure_token()
    assert r1.severity == "ok" and tok.exists()
    first = tok.read_text()
    r2 = console_phase.ensure_token()  # second call must not regenerate
    assert tok.read_text() == first and "present" in r2.detail
