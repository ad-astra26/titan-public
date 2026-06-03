"""Phase B of the zero-to-sovereign auto-provisioner (rFP_setup_titan_auto_provisioner.md §1.2/§7).

The ACTOR that actively installs the pinned toolchain on a bare box. Runs as a
phase BEFORE venv/binaries/genesis (INV-PROV-5), gated by mode + action
(INV-PROV-2), idempotent per-tool (INV-PROV-1 — a component already present at a
compatible version is never reinstalled), pins by default + override by flag
(INV-PROV-3, via toolchain.resolve_versions upstream).

Separation of concerns (scope fence §3): `toolchain.py` DECLARES + DETECTS;
`preflight.py` DETECTS only; THIS module is the only one that INSTALLS. A failed
install HARD-BLOCKS (returns a 'fail' Result → the walker halts) with exact
remediation; `--resume` re-enters and skips the components that now pass.

Privilege (INV-PROV-7): rust/solana/anchor install into the user's HOME (no
sudo). Only Node uses sudo — and that sudo is an explicit, streamed command the
user sees; we never assume passwordless root.

Sovereignty (INV-PROV-6): provisioning mutates only the host toolchain + the
user's PATH/shell-profile — never identity, keypair, shard, or config secrets.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from . import toolchain
from .modes import Mode, spec_for
from .preflight import Result
from .prompts import Prompter
from .toolchain import MUSL_TARGET
from .ui import cprint

# Canonical install order — rust first (anchor's avm is a cargo install; anchor
# needs rustc), then solana, then anchor, then node. required_tools() filters
# this list, preserving the order.
_INSTALL_ORDER = ("rust", "solana", "anchor", "node")

# Where each installer drops its binaries — prepended to PATH for the rest of
# this run AND written to the shell profile for the user's next login (§5).
def _toolchain_bin_dirs() -> list[Path]:
    home = Path.home()
    return [
        home / ".cargo" / "bin",                                                  # rustup, cargo, avm
        home / ".local" / "share" / "solana" / "install" / "active_release" / "bin",  # solana (Agave)
        home / ".avm" / "bin",                                                    # anchor (avm-managed)
    ]


_PROFILE_MARKER_BEGIN = "# >>> titan toolchain (setup_titan) >>>"
_PROFILE_MARKER_END = "# <<< titan toolchain (setup_titan) <<<"


class ProvisionError(Exception):
    """A toolchain installer failed. Carries a user-facing remediation hint."""

    def __init__(self, detail: str, remediation: str):
        super().__init__(detail)
        self.detail = detail
        self.remediation = remediation


# ── mode / action gating (INV-PROV-2) ────────────────────────────────────────

def required_tools(mode: Mode | str, *, resurrect: bool = False) -> list[str]:
    """The logical tools to ensure-present for this (mode, action), in install order.

    - LOCAL: none (no chain → `genesis_on_chain=false`).
    - devnet / mainnet install: the full set (rust+solana+anchor+node).
    - --resurrect: rust+solana+node (Arweave/Irys) but NOT anchor — resurrection
      restores from chain; it never deploys a program.
    """
    if resurrect:
        wanted = {"rust", "solana", "node"}
    else:
        spec = spec_for(mode)
        wanted = set()
        if spec.needs_rust:
            wanted.add("rust")
        if spec.needs_solana_cli:
            wanted.add("solana")
        if spec.needs_anchor:
            wanted.add("anchor")
        if spec.needs_node:
            wanted.add("node")
    return [t for t in _INSTALL_ORDER if t in wanted]


# ── low-level command runner (streamed, never capture-and-hang) ──────────────

def _run(cmd, *, shell: bool = False, env: Optional[dict] = None) -> None:
    """Run a provisioning command, STREAMING its output (the avm anchor compile
    is ~15–30 min — capturing it would look hung). Raises CalledProcessError on a
    non-zero exit. Factored out so tests can mock every install with one patch."""
    subprocess.run(cmd, shell=shell, env=env, check=True)


# ── PATH wiring (export for the run + persist for next login, §5) ─────────────

def ensure_path_for_run() -> None:
    """Prepend the toolchain bin dirs to THIS process's PATH so the re-verify +
    later phases (and avm, which needs cargo) resolve freshly-installed tools.
    Idempotent — a dir already on PATH is not duplicated."""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for d in reversed(_toolchain_bin_dirs()):   # reversed → final order matches _toolchain_bin_dirs
        s = str(d)
        if s not in parts:
            parts.insert(0, s)
    os.environ["PATH"] = os.pathsep.join(parts)


def _profile_path() -> Path:
    return Path.home() / ".profile"


def write_profile_lines(profile: Optional[Path] = None) -> bool:
    """Append a marker-guarded PATH export to ~/.profile for the user's next
    login. Idempotent: a no-op if the marker block already exists. Returns True
    if it wrote the block, False if it was already present."""
    profile = profile or _profile_path()
    existing = profile.read_text() if profile.exists() else ""
    if _PROFILE_MARKER_BEGIN in existing:
        return False
    export = ":".join(f"$HOME/{d.relative_to(Path.home())}" for d in _toolchain_bin_dirs())
    block = (f"\n{_PROFILE_MARKER_BEGIN}\n"
             f'export PATH="{export}:$PATH"\n'
             f"{_PROFILE_MARKER_END}\n")
    with profile.open("a") as fh:
        fh.write(block)
    return True


# ── installers (one per tool; raise ProvisionError on failure) ───────────────

RUSTUP_URL = "https://sh.rustup.rs"
ANZA_INSTALL_URL = "https://release.anza.xyz/v{ver}/install"
ANCHOR_REPO = "https://github.com/coral-xyz/anchor"
NODESOURCE_URL = "https://deb.nodesource.com/setup_{major}.x"


def _install_rust(version: str) -> None:
    """rustup → the pinned channel (minimal profile) + the musl cross-target +
    rustfmt/clippy. User-space (~/.cargo) — no sudo."""
    try:
        _run(f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_URL} | "
             f"sh -s -- -y --default-toolchain {version} --profile minimal",
             shell=True)
        ensure_path_for_run()          # rustup/cargo now on PATH for the next calls
        _run(["rustup", "component", "add", "rustfmt", "clippy"])
        _run(["rustup", "target", "add", MUSL_TARGET])
    except subprocess.CalledProcessError as e:
        raise ProvisionError(
            f"rustup install exited {e.returncode}",
            "Inspect the rustup output above (common: missing curl/build-essential). "
            f"Manual: curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_URL} | sh -s -- -y "
            f"--default-toolchain {version}; rustup target add {MUSL_TARGET}.")


def _install_solana(version: str) -> None:
    """Agave (Anza) Solana CLI at the pinned version. User-space — no sudo."""
    url = ANZA_INSTALL_URL.format(ver=version)
    try:
        _run(f'sh -c "$(curl -sSfL {url})"', shell=True)
        ensure_path_for_run()
    except subprocess.CalledProcessError as e:
        raise ProvisionError(
            f"solana (Agave {version}) install exited {e.returncode}",
            f"Inspect the output above. Manual: sh -c \"$(curl -sSfL {url})\".")


def _install_anchor(version: str) -> None:
    """avm (Anchor Version Manager) via cargo, then the pinned anchor CLI. Needs
    cargo on PATH (rust installs first). User-space — no sudo."""
    try:
        _run(["cargo", "install", "--git", ANCHOR_REPO, "avm", "--locked"])
        ensure_path_for_run()
        _run(["avm", "install", version])
        _run(["avm", "use", version])
    except subprocess.CalledProcessError as e:
        raise ProvisionError(
            f"anchor/avm install exited {e.returncode}",
            "Inspect the avm build output above (the anchor compile is slow but "
            f"verbose). Manual: cargo install --git {ANCHOR_REPO} avm --locked; "
            f"avm install {version}; avm use {version}.")


def _install_node(version: str) -> None:
    """Node.js at the pinned major via NodeSource. Needs sudo (apt) — the only
    privileged installer; the sudo is explicit + streamed (INV-PROV-7)."""
    major = version.split(".")[0]
    url = NODESOURCE_URL.format(major=major)
    try:
        _run(f"curl -fsSL {url} | sudo -E bash -", shell=True)
        _run(["sudo", "apt-get", "install", "-y", "nodejs"])
    except subprocess.CalledProcessError as e:
        raise ProvisionError(
            f"node {major}.x (NodeSource) install exited {e.returncode}",
            f"Inspect the output above (needs sudo). Manual: curl -fsSL {url} | "
            "sudo -E bash -; sudo apt-get install -y nodejs.")


_INSTALLERS = {
    "rust": _install_rust,
    "solana": _install_solana,
    "anchor": _install_anchor,
    "node": _install_node,
}


# ── phase entry point ─────────────────────────────────────────────────────────

def run_provision_phase(install_root: Path, mode: Mode | str, pins: dict[str, str],
                        *, resurrect: bool = False,
                        prompter: Optional[Prompter] = None,
                        default: bool = False) -> list[Result]:
    """Install the pinned toolchain for (mode, action). Returns the phase
    Results; the FIRST 'fail' halts the walker (INV-PROV-5). Idempotent: tools
    already present at a compatible version are skipped (INV-PROV-1).

    `pins` is the resolved {tool: version} map (toolchain.resolve_versions);
    `resurrect` drops anchor from the set; `prompter`/`default` are accepted for
    a uniform phase signature (provisioning needs no interactive input today).
    """
    wanted = required_tools(mode, resurrect=resurrect)
    mode_label = mode.value if isinstance(mode, Mode) else str(mode)
    if not wanted:
        return [Result("provision", "ok",
                       f"{mode_label}: no toolchain required (local — on-chain anchor skipped)")]

    # Make any already-installed (or about-to-be-installed) tools visible to the
    # version probe + the rest of the run.
    ensure_path_for_run()
    cprint(f"  Provisioning toolchain for {mode_label}: {toolchain.summary(pins)}",
           role="text_strong")

    results: list[Result] = []
    for tool in wanted:
        version = pins.get(tool, toolchain.PINS[tool])
        status = toolchain.tool_present_at(tool, version)
        if status.compatible:
            results.append(Result(tool, "ok", f"already present — {status.detail}"))
            continue

        cprint(f"  Installing {tool} {version} ({status.detail})…", role="text_strong")
        try:
            _INSTALLERS[tool](version)
        except ProvisionError as e:
            results.append(Result(tool, "fail", e.detail,
                                  e.remediation + " Then re-run with --resume."))
            return results          # HARD-BLOCK — the walker halts here

        ensure_path_for_run()
        post = toolchain.tool_present_at(tool, version)
        if not post.compatible:
            results.append(Result(tool, "fail",
                                  f"{tool} install completed but verification failed — {post.detail}",
                                  "Inspect the install output above; re-run with --resume."))
            return results
        results.append(Result(tool, "ok", f"installed — {post.detail}"))

    if write_profile_lines():
        results.append(Result("path", "ok",
                              f"PATH export written to {_profile_path()} (for your next login)"))
    results.append(Result("provision", "ok",
                          f"toolchain ready — {toolchain.summary(pins)}"))
    return results
