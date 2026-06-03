"""Phase A of the zero-to-sovereign auto-provisioner (rFP_setup_titan_auto_provisioner.md §7).

The single source of truth for the toolchain the provisioner installs:
  • PINS              — the T1-build-verified pinned versions (§6, INV-PROV-3)
  • resolve_versions  — apply any --<tool>-version override → resolved pins
  • tool_present_at   — "is this tool already on PATH at a compatible version?",
                        the idempotency probe (INV-PROV-1)

PURE / OFFLINE / NO INSTALLS. `provision.py` (Phase B) is the ACTOR that installs
the toolchain; this module only declares WHAT is pinned and DETECTS whether the
pin is already satisfied. (Scope fence §3: detect vs. act stay separate —
preflight stays detect-only too; this is the canonical *version-aware* presence
check the provisioner consults before installing anything.)

Pins (§6 — verified 2026-06-03 against T1's live build env; the rust channel +
targets + components mirror titan-rust/rust-toolchain.toml):
    rust    "stable" channel + musl target x86_64-unknown-linux-musl + rustfmt/clippy
            (rustc 1.95.0 today; the repo's rust-toolchain.toml governs per-build)
    solana  Agave 3.1.10   (release.anza.xyz/v3.1.10/install)
    anchor  0.32.1         (avm install 0.32.1)
    node    22             (NodeSource setup_22.x; T1 runs 22.22.2)
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# Canonical musl cross-target. binaries.MUSL_TARGET holds the same string for the
# build path; Phase B may collapse the two once provision.py owns the rust install.
MUSL_TARGET = "x86_64-unknown-linux-musl"
RUST_COMPONENTS = ("rustfmt", "clippy")
_RUST_CHANNELS = ("stable", "beta", "nightly")

# Logical tool → pinned version token. The token *shape* encodes how it is
# compared (see _satisfies): a bare channel name (rust), a bare major (node),
# or a full semver (solana/anchor).
PINS: dict[str, str] = {
    "rust":   "stable",
    "solana": "3.1.10",
    "anchor": "0.32.1",
    "node":   "22",
}

# Logical tool → the executable probed on PATH.
EXECUTABLE: dict[str, str] = {
    "rust":   "cargo",
    "solana": "solana",
    "anchor": "anchor",
    "node":   "node",
}

# argparse dest (Phase B's --<tool>-version flags) → logical tool. Read
# defensively in resolve_versions so this works before the flags are wired.
_FLAG_TO_TOOL: dict[str, str] = {
    "rust_version":   "rust",
    "solana_version": "solana",
    "anchor_version": "anchor",
    "node_version":   "node",
}


@dataclass(frozen=True)
class ToolStatus:
    """Result of tool_present_at — `compatible` drives the skip-vs-install call."""
    tool: str                  # logical tool key (rust/solana/anchor/node)
    pinned: str                # the version we want
    executable: str            # the probed binary name
    present: bool              # binary on PATH at all
    found: Optional[str]       # detected version (None if absent/unparseable)
    compatible: bool           # present AND satisfies the pin → skip install
    detail: str                # human one-liner for logs

    @property
    def needs_install(self) -> bool:
        """True → the provisioner must install/upgrade this component."""
        return not self.compatible


# ── version resolution (INV-PROV-3: pinned by default, override by flag) ──────

def resolve_versions(args=None) -> dict[str, str]:
    """Return the resolved {tool: version} pins, applying any --<tool>-version
    override carried on `args` (an argparse.Namespace, a mapping, or None).

    Read defensively so this works BEFORE Phase B wires the flags into argparse:
    a missing / empty / None attribute leaves the default pin untouched.
    """
    resolved = dict(PINS)
    if args is None:
        return resolved
    get = args.get if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
    for dest, tool in _FLAG_TO_TOOL.items():
        override = get(dest, None)
        if override:
            resolved[tool] = str(override).strip()
    return resolved


def summary(resolved: dict[str, str]) -> str:
    """One-line human summary of the resolved pins (for the provision log)."""
    order = ("rust", "solana", "anchor", "node")
    return " · ".join(f"{t}={resolved[t]}" for t in order if t in resolved)


# ── version parsing + comparison (pure) ──────────────────────────────────────

# A dotted version token: major + 1 or 2 more components (2.2, 3.1.10, 1.95.0).
_VERSION_RE = re.compile(r"(\d+(?:\.\d+){1,2})")


def _extract_version(text: str) -> Optional[str]:
    """Pull the first dotted version token out of a `--version` line.

    Handles 'solana-cli 3.1.10 (src:…)', 'anchor-cli 0.32.1', 'v22.22.2',
    'rustc 1.95.0 (…)'. Returns None when no version-looking token is present.
    """
    if not text:
        return None
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


def _satisfies(pinned: str, found: Optional[str]) -> bool:
    """Does the detected `found` version satisfy the `pinned` token?

    Comparison is by the *pinned token shape* (chosen per-tool in RFP §6):
      • a bare major ("22")       → major must match            (node)
      • a major.minor ("3.1")     → major + minor must match
      • a full semver ("3.1.10")  → exact match                 (solana, anchor)

    A prefix-equality of the dotted components realizes all three: the pin lists
    exactly the components that must match; the found version may carry more
    (e.g. node "22" is satisfied by "22.22.2"). Rust's channel pin is handled in
    tool_present_at (channel + musl target), not here.
    """
    if not found:
        return False
    want = pinned.split(".")
    have = found.split(".")
    if len(have) < len(want):
        return False
    return have[: len(want)] == want


# ── presence probes (best-effort, non-raising) ───────────────────────────────

def _probe_version(executable: str, version_arg: str = "--version") -> Optional[str]:
    """Run `<executable> --version` → its raw output, or None if the binary is
    absent / errors / times out. Detection never raises (best-effort)."""
    if shutil.which(executable) is None:
        return None
    try:
        proc = subprocess.run([executable, version_arg],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return ((proc.stdout or "") + (proc.stderr or "")).strip() or None


def _musl_target_installed() -> Optional[bool]:
    """True / False whether the musl cross-target is installed via rustup; None
    if rustup is absent (cannot tell → caller treats as not-satisfied)."""
    if shutil.which("rustup") is None:
        return None
    try:
        proc = subprocess.run(["rustup", "target", "list", "--installed"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return MUSL_TARGET in proc.stdout.split()


def tool_present_at(tool: str, version: str) -> ToolStatus:
    """Is `tool` already on PATH at a version compatible with `version`?

    The provisioner's idempotency gate (INV-PROV-1): `compatible=True` means
    "already satisfied — install nothing." Best-effort + non-raising on probe
    failures (an unparseable/absent tool simply reports not-compatible).
    """
    if tool not in EXECUTABLE:
        raise KeyError(f"unknown toolchain component {tool!r} "
                       f"(known: {sorted(EXECUTABLE)})")
    exe = EXECUTABLE[tool]

    # Rust pinned to a channel ("stable"): satisfied = cargo present AND the
    # musl cross-target installed. The exact rustc number floats with the
    # channel (rust-toolchain.toml governs per-build) → informational only.
    if tool == "rust" and version in _RUST_CHANNELS:
        rustc_ver = _extract_version(_probe_version("rustc") or "")
        if shutil.which(exe) is None:
            return ToolStatus(tool, version, exe, present=False, found=rustc_ver,
                              compatible=False, detail="cargo not on PATH")
        musl = _musl_target_installed()
        if musl is True:
            return ToolStatus(tool, version, exe, present=True, found=rustc_ver,
                              compatible=True,
                              detail=f"rustc {rustc_ver or '?'} ({version}) + {MUSL_TARGET}")
        why = ("musl target not installed" if musl is False
               else "rustup absent — musl target unverifiable")
        return ToolStatus(tool, version, exe, present=True, found=rustc_ver,
                          compatible=False,
                          detail=f"rustc {rustc_ver or '?'} present but {why}")

    # Everything else (incl. a numeric rust override): compare the probed
    # version against the pin by token shape. Rust's version comes from rustc;
    # presence is still gated on cargo.
    probe_exe = "rustc" if tool == "rust" else exe
    found = _extract_version(_probe_version(probe_exe) or "")
    if shutil.which(exe) is None:
        return ToolStatus(tool, version, exe, present=False, found=found,
                          compatible=False, detail=f"{exe} not on PATH")
    ok = _satisfies(version, found)
    detail = (f"{exe} {found} satisfies pin {version}" if ok
              else f"{exe} {found or '?'} != pin {version}")
    return ToolStatus(tool, version, exe, present=True, found=found,
                      compatible=ok, detail=detail)
