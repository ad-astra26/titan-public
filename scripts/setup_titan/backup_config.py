"""Phase — mainnet backup posture: encryption + config-in-backup (W1.5 / §24.4.B).

A mainnet Titan backs up to the sovereign Arweave chain automatically. Two posture
choices the operator must make consciously at install time:

1. **Encryption** (`[backup].encryption_enabled`) — STRONGLY recommended. On → every
   tarball (incl. config.toml when included) ships to Arweave as ciphertext (Mode B),
   recoverable only via the soul keypair (2-of-3 Shamir). Off (Mode A) → the DATA is
   plaintext on permanent, public Arweave.

2. **config-in-backup** (`[backup].backup_config_toml`, SPEC §24.4.B / D-SPEC-147) —
   include `titan_hcl/config.toml` in the sovereign backup so a resurrection restores a
   FULLY-configured Titan that boots ready (no re-prompt). config.toml carries the
   operator's credentials, so this is opt-in + warned: enabling it WITHOUT encryption
   publishes those credentials in the clear. Declining means the operator keeps their
   own copy of config.toml and supplies it (alongside Shard-1) at resurrection.

Sovereign Arweave backup + resurrection are MAINNET-only (locked decisions #15/#16),
so devnet/local installs skip this phase.
"""
from __future__ import annotations

from pathlib import Path

from .config_model import set_by_dotted
from .config_seed import config_path
from .modes import Mode
from .preflight import Result
from .prompts import Prompter
from .ui import cprint, section


def run_backup_config_phase(install_root: Path, mode: Mode, *,
                            prompter: Prompter, default: bool = False) -> list[Result]:
    """Set [backup].encryption_enabled + backup_config_toml for a mainnet install."""
    if mode != Mode.MAINNET:
        return [Result("backup_config", "ok",
                       f"{mode.value}: sovereign Arweave backup + resurrection are "
                       "mainnet-only — config-backup posture N/A")]

    cfg = config_path(install_root)
    if not cfg.exists():
        return [Result("backup_config", "fail", f"config.toml missing at {cfg}",
                       "The config-seed phase must run before this one.")]

    results: list[Result] = []
    section("Sovereign backup posture (mainnet)")
    cprint("  Your Titan backs up to the Arweave sovereign chain automatically.",
           role="text_muted")

    # 1. Encryption — strongly recommended (curated ON under --default).
    encrypt = True if default else prompter.confirm(
        "encrypt_backups",
        "Encrypt everything sent to Arweave? STRONGLY recommended — without it your "
        "backed-up data is plaintext on permanent public storage",
        default_yes=True)
    if not set_by_dotted(cfg, "backup.encryption_enabled", "true" if encrypt else "false"):
        return [Result("backup_config", "fail",
                       "could not set [backup].encryption_enabled in config.toml",
                       "config.toml may predate this key — re-seed from config.toml.example.")]
    results.append(Result(
        "encryption", "ok" if encrypt else "warn",
        f"encryption_enabled = {str(encrypt).lower()}",
        None if encrypt else "Plaintext Arweave — only safe if your data carries no secrets."))

    # 2. config-in-backup (§24.4.B) — opt-in + warned.
    cprint("\n  config.toml holds ALL your settings — and your credentials. Including it "
           "in the backup lets a resurrection restore a ready-to-run Titan, but it then "
           "lives on Arweave.", role="text_strong")
    if not encrypt:
        cprint("  ⚠ Encryption is OFF — if you include config.toml, it (and every "
               "credential in it) is PLAINTEXT on permanent public Arweave.",
               role="warning", bold=True)
    include = (True if default else prompter.confirm(
        "backup_config_toml",
        "Include config.toml in your sovereign backup (so resurrection needs no "
        "re-configuration)?",
        default_yes=encrypt))
    if not set_by_dotted(cfg, "backup.backup_config_toml", "true" if include else "false"):
        return [Result("backup_config", "fail",
                       "could not set [backup].backup_config_toml in config.toml",
                       "config.toml may predate this key — re-seed from config.toml.example.")]
    if include:
        results.append(Result("config_backup", "ok",
                              "backup_config_toml = true — config restored on resurrection"))
    else:
        cprint("  → You chose NOT to back up config.toml. Keep your own copy safe: you "
               "must supply it (with your Shard-1) when running the resurrection "
               "protocol.", role="text_muted")
        results.append(Result("config_backup", "ok",
                              "backup_config_toml = false — you self-supply config.toml at resurrection"))
    return results
