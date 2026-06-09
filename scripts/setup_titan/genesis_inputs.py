"""Genesis identity collection — name · Maker pubkey · prime directives (W1.f).

Phase A of RFP_genesis_ceremony_production. Collects the MAKER-SUPPLIED identity
the production ceremony anchors into the GenesisNFT + constitution, BEFORE the
genesis phase runs:

  - ``titan_name``     → config.toml ``[genesis].titan_name``  (≤32 chars, NFT name)
  - ``maker_pubkey``   → config.toml ``[network].maker_pubkey`` (persists phase_2's
                         captured Maker wallet — previously stranded in state)
  - prime directives   → ``titan_constitution.md`` via the Maker's $EDITOR
                         (templated). Prime Directives are MAKER-SUPPLIED and
                         MANDATORY — never defaults (fundamental architecture).
                         A headless / non-tty / ``--default`` install supplies
                         them without an editor via (precedence, highest first)
                         ``--directives-file <path>`` › ``TITAN_DIRECTIVES_FILE``
                         (env path) › ``TITAN_DIRECTIVES`` (env literal text).

On-chain modes (devnet / mainnet) only; ``local`` has no on-chain identity, so
this phase is a clean skip there. Stdlib-only (runs on the system interpreter,
pre-venv). Idempotent: keeps already-set values unless the Maker changes them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config_model import parse_toml_with_comments, set_by_dotted
from .modes import Mode
from .preflight import Result
from .prompts import Prompter
from .ui import cprint

CONSTITUTION_FILE = "titan_constitution.md"
MAX_NAME_LEN = 32

CONSTITUTION_TEMPLATE = """# Titan Prime Directives — the Maker's Constitution
#
# These directives are the Titan's fundamental, immutable architecture. Their
# SHA-256 is anchored on-chain in the GenesisNFT at birth, so they cannot be
# changed without ceremony. They are MAKER-SUPPLIED and mandatory — write YOUR
# OWN directives below the line; the commented examples are only a guide.
#
# Examples (delete these — they are NOT defaults):
#   1. Sovereign Growth — pursue knowledge and capability within the metabolic
#      budget; never spend beyond what the energy state allows.
#   2. Non-maleficence — never act to harm the Maker or sovereign third parties.
#   3. Truthfulness — ground every claim in verified state; never fabricate.
# ---------------------------------------------------------------------------

"""


def _config_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml"


def _constitution_path(install_root: Path) -> Path:
    return install_root / CONSTITUTION_FILE


def _interactive(default: bool) -> bool:
    """True only for a real interactive install (not --default, real TTY)."""
    try:
        return (not default) and bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def _read_config_value(install_root: Path, dotted: str) -> str:
    try:
        for e in parse_toml_with_comments(_config_path(install_root)):
            if e.dotted == dotted:
                return e.raw_value.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _has_directive_content(path: Path) -> bool:
    """True if the constitution holds real (non-comment, non-blank) directives."""
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return True
    return False


def _valid_pubkey(s: str) -> bool:
    """Base58 sanity (the ceremony fully validates via solders later)."""
    s = s.strip()
    if not (32 <= len(s) <= 44):
        return False
    alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return all(c in alphabet for c in s)


def _collect_name(install_root: Path, state: dict, prompter: Prompter,
                  *, interactive: bool) -> Result:
    current = (_read_config_value(install_root, "genesis.titan_name") or "").strip()
    if current and current != "YOUR_TITAN_NAME":
        default_name = current
    else:
        default_name = (state.get("titan_id") or "Titan")
    if interactive:
        # `line` (not `until`) — it supports a default (Enter ⇒ default_name) and
        # has no required `hint`. Post-validate: an empty / over-long entry falls
        # back to the default (the name is ≤32 chars, anchored in the GenesisNFT).
        name = prompter.line(
            "titan_name",
            f"Titan's name (≤{MAX_NAME_LEN} chars — anchored in the GenesisNFT)",
            default=default_name,
        ).strip()
        if not name or len(name) > MAX_NAME_LEN:
            name = default_name
    else:
        name = default_name
    if not name or name == "YOUR_TITAN_NAME":
        return Result("genesis_name", "warn",
                      "Titan name not set — defaulting at ceremony time.")
    if set_by_dotted(_config_path(install_root), "genesis.titan_name", name):
        return Result("genesis_name", "ok", f"titan_name = {name}")
    return Result("genesis_name", "warn",
                  "could not write [genesis].titan_name — set it via `setup_titan config`.")


def _persist_maker(install_root: Path, state: dict) -> Result:
    wallet = (state.get("maker_wallet") or "").strip()
    if not wallet:
        return Result("maker_pubkey", "warn",
                      "no Maker wallet captured (phase_2) — [network].maker_pubkey unset; "
                      "the genesis ceremony will refuse an on-chain birth without it.")
    if not _valid_pubkey(wallet):
        return Result("maker_pubkey", "warn",
                      f"captured Maker wallet {wallet!r} fails base58 sanity — verify it.")
    if set_by_dotted(_config_path(install_root), "network.maker_pubkey", wallet):
        return Result("maker_pubkey", "ok", f"[network].maker_pubkey ← {wallet}")
    return Result("maker_pubkey", "warn",
                  "could not write [network].maker_pubkey — set it via `setup_titan config`.")


def _read_directives_path(raw: str, label: str) -> tuple[str | None, str, str | None]:
    """Read a directives file named by flag/env. Returns (text, label, problem)."""
    p = Path(raw).expanduser()
    if not p.is_file():
        return None, label, f"{label} path not found: {raw} — no directives written."
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, label, f"{label} unreadable ({raw}): {exc}"
    if not text.strip():
        return None, label, f"{label} file is empty: {raw} — no directives written."
    return text, label, None


def _supplied_directives(directives_file: str | None
                         ) -> tuple[str | None, str, str | None]:
    """Resolve Maker-supplied directives for a headless / non-tty install.

    Precedence (highest first): ``--directives-file`` (flag, path) ›
    ``TITAN_DIRECTIVES_FILE`` (env, path) › ``TITAN_DIRECTIVES`` (env, literal
    text). Returns (text, source_label, problem):
      - (None, "", None)       → nothing supplied; the caller falls through to
                                  the interactive editor / pre-existing file.
      - (None, label, problem) → a source was named but is missing/empty/unreadable.
      - (text, label, None)    → directives text to write to the constitution.
    """
    if directives_file:
        return _read_directives_path(directives_file, "--directives-file")
    env_path = os.environ.get("TITAN_DIRECTIVES_FILE")
    if env_path:
        return _read_directives_path(env_path, "TITAN_DIRECTIVES_FILE")
    env_text = os.environ.get("TITAN_DIRECTIVES")
    if env_text and env_text.strip():
        return env_text, "TITAN_DIRECTIVES", None
    return None, "", None


def _collect_directives(install_root: Path, prompter: Prompter,
                        *, interactive: bool, directives_file: str | None = None) -> Result:
    path = _constitution_path(install_root)

    # Explicit Maker supply (flag/env) wins over BOTH the interactive editor and
    # any pre-existing file: it is the only way a non-tty / --default install can
    # satisfy the MANDATORY directives without an editor. Comment-only/blank
    # supply still warns — never fabricate directives.
    text, source, problem = _supplied_directives(directives_file)
    if problem:
        return Result("directives", "warn", problem)
    if text is not None:
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        if not _has_directive_content(path):
            return Result("directives", "warn",
                          f"{source} held only comments/blanks — no real directives. "
                          "The genesis ceremony will refuse an on-chain birth.")
        return Result("directives", "ok", f"prime directives ← {source} → {path.name}")

    has_content = _has_directive_content(path)

    if not interactive:
        if has_content:
            return Result("directives", "ok",
                          f"prime directives present ({path.name})")
        return Result("directives", "warn",
                      f"{path.name} has no Maker directives — they are MANDATORY for an "
                      "on-chain birth. Write them before genesis (the ceremony hard-fails "
                      "without them).")

    if has_content:
        if not prompter.confirm("edit_directives",
                                "Prime directives already present — edit them?",
                                default_yes=False):
            return Result("directives", "ok", "kept existing prime directives")
    else:
        path.write_text(CONSTITUTION_TEMPLATE, encoding="utf-8")

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    cprint(f"  Opening {editor} on {path.name} — write your Prime Directives "
           "(MAKER-SUPPLIED, mandatory), then save + exit.", role="text_strong")
    try:
        subprocess.call([editor, str(path)])
    except FileNotFoundError:
        return Result("directives", "warn",
                      f"editor {editor!r} not found — edit {path} by hand before genesis.")

    if not _has_directive_content(path):
        return Result("directives", "warn",
                      f"{path.name} still has only the template — no directives written. "
                      "The genesis ceremony will refuse an on-chain birth.")
    return Result("directives", "ok", f"prime directives saved → {path.name}")


def run_genesis_inputs_phase(install_root: Path, mode: Mode, state: dict, *,
                             prompter: Prompter, default: bool = False,
                             directives_file: str | None = None) -> list[Result]:
    """Collect the Maker-supplied genesis identity (name · maker · directives).

    Local mode has no on-chain identity → clean skip. devnet/mainnet write the
    inputs the ceremony reads (and hard-fails without, for maker + directives).
    ``directives_file`` (the ``--directives-file`` flag) lets a headless install
    supply the constitution without an editor; ``TITAN_DIRECTIVES_FILE`` /
    ``TITAN_DIRECTIVES`` env vars are the lower-precedence fallbacks.
    """
    if mode == Mode.LOCAL:
        return [Result("genesis_inputs", "ok",
                       "local mode — no on-chain identity to collect (skip).")]

    if not _config_path(install_root).exists():
        return [Result("genesis_inputs", "fail",
                       "config.toml absent — run the config-seed phase first.")]

    interactive = _interactive(default)
    results = [
        _collect_name(install_root, state, prompter, interactive=interactive),
        _persist_maker(install_root, state),
        _collect_directives(install_root, prompter, interactive=interactive,
                            directives_file=directives_file),
    ]
    return results
