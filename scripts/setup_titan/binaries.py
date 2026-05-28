"""Rust daemon binaries — fetch (verified) or build locally.

A fresh public clone has no `bin/` (the 9 musl daemons are excluded from the
public repo and shipped as GitHub Release assets — see .github/workflows/
release.yml + Workstream 6). The systemd phase (W1.e) needs them present, so
the install walker runs this phase first.

Two acquisition paths:
  - fetch  (default): download the 9 daemons + SHA256SUMS from the GitHub
    Release for the chosen tag, VERIFY every binary against SHA256SUMS, install
    to bin/ with +x. Refuses to install anything that fails the checksum.
  - build  (--build-rust): the fully-sovereign path — `cargo build --release`
    the musl target from titan-rust/ source, then stage into bin/.

Stdlib-only for fetch (urllib + hashlib). Build shells out to cargo.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .preflight import Result
from .ui import cprint

# The 9 daemons published by release.yml (kept in sync with that workflow).
DAEMONS = (
    "titan-inner-body-rs",
    "titan-inner-mind-rs",
    "titan-inner-spirit-rs",
    "titan-kernel-rs",
    "titan-outer-body-rs",
    "titan-outer-mind-rs",
    "titan-outer-spirit-rs",
    "titan-trinity-rs",
    "titan-unified-spirit-rs",
)
SHA256SUMS = "SHA256SUMS"
RELEASE_BASE = "https://github.com/ad-astra26/titan-public/releases/download"
MUSL_TARGET = "x86_64-unknown-linux-musl"


def bin_dir(install_root: Path) -> Path:
    return install_root / "bin"


def _have_all(install_root: Path) -> bool:
    return all((bin_dir(install_root) / d).exists() for d in DAEMONS)


# ── checksum helpers ─────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse `<hash>  <name>` lines (the format `sha256sum … > SHA256SUMS` emits)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1]] = parts[0].lower()
    return out


def _download(url: str, dest: Path, timeout_s: float = 120.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout_s) as r, dest.open("wb") as fh:
        shutil.copyfileobj(r, fh)


# ── fetch path ───────────────────────────────────────────────────────────────


def fetch_release_binaries(install_root: Path, tag: str) -> list[Result]:
    """Download + verify + install the 9 daemons for `tag` into bin/."""
    if not tag or tag in ("main", "HEAD"):
        return [Result("binaries", "fail",
                       f"no release tag to fetch from (got {tag!r})",
                       "Pass --tag vX.Y.Z (a cut release), or use --build-rust to compile "
                       "from titan-rust/ source.")]

    bd = bin_dir(install_root)
    bd.mkdir(parents=True, exist_ok=True)
    base = f"{RELEASE_BASE}/{tag}"

    # 1. fetch the manifest of checksums
    try:
        with urllib.request.urlopen(f"{base}/{SHA256SUMS}", timeout=30) as r:
            sums = parse_sha256sums(r.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return [Result("binaries", "fail", f"could not fetch {SHA256SUMS} for {tag}: {e}",
                       f"Verify the release exists at {RELEASE_BASE}/{tag}, or use --build-rust.")]

    missing = [d for d in DAEMONS if d not in sums]
    if missing:
        return [Result("binaries", "fail",
                       f"{SHA256SUMS} is missing entries: {', '.join(missing)}",
                       "The release is incomplete — re-cut it, or use --build-rust.")]

    # 2. download + verify each daemon, install only on a checksum match
    cprint(f"  Fetching {len(DAEMONS)} Rust daemons for {tag} (verified against {SHA256SUMS})…",
           role="text_strong")
    for name in DAEMONS:
        tmp = bd / f".{name}.partial"
        try:
            _download(f"{base}/{name}", tmp)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            tmp.unlink(missing_ok=True)
            return [Result("binaries", "fail", f"download failed for {name}: {e}",
                           "Network issue or missing asset — retry, or use --build-rust.")]
        got = _sha256(tmp)
        if got != sums[name]:
            tmp.unlink(missing_ok=True)
            return [Result("binaries", "fail",
                           f"CHECKSUM MISMATCH for {name}: got {got[:12]}…, expected {sums[name][:12]}…",
                           "Refusing to install an unverified binary. Re-download or use --build-rust.")]
        final = bd / name
        tmp.replace(final)
        final.chmod(0o755)
        cprint(f"    ✓ {name}", role="success")

    return [Result("binaries", "ok", f"{len(DAEMONS)} daemons fetched + verified into {bd}")]


# ── build path (sovereign) ───────────────────────────────────────────────────


def build_release_binaries(install_root: Path) -> list[Result]:
    """`cargo build --release` the musl target from titan-rust/, stage into bin/."""
    rust_dir = install_root / "titan-rust"
    if not (rust_dir / "Cargo.toml").exists():
        return [Result("binaries", "fail", f"no Cargo project at {rust_dir}",
                       "titan-rust/ is excluded from some checkouts; re-clone in full.")]
    if shutil.which("cargo") is None:
        return [Result("binaries", "fail", "cargo not found",
                       "Install the Rust toolchain (https://rustup.rs) + the musl target: "
                       f"rustup target add {MUSL_TARGET}")]

    cprint(f"  Building 9 musl daemons from {rust_dir} (this can take several minutes)…",
           role="text_strong")
    try:
        subprocess.check_call(
            ["cargo", "build", "--release", "--bins", "--target", MUSL_TARGET],
            cwd=str(rust_dir),
        )
    except subprocess.CalledProcessError as e:
        return [Result("binaries", "fail", f"cargo build exited {e.returncode}",
                       "Inspect the build output above (common: missing musl-tools — "
                       "sudo apt install -y musl-tools).")]

    src = rust_dir / "target" / MUSL_TARGET / "release"
    bd = bin_dir(install_root)
    bd.mkdir(parents=True, exist_ok=True)
    staged = 0
    for name in DAEMONS:
        built = src / name
        if not built.exists():
            return [Result("binaries", "fail", f"built binary missing: {built}",
                           "The build did not produce all 9 daemons — check Cargo.toml [[bin]] targets.")]
        dest = bd / name
        shutil.copy2(built, dest)
        dest.chmod(0o755)
        staged += 1
    return [Result("binaries", "ok", f"{staged} daemons built + staged into {bd}")]


# ── install-phase entry point ────────────────────────────────────────────────


def run_binaries_phase(install_root: Path, *, tag: str, build_rust: bool) -> list[Result]:
    """Phase body — acquire the 9 daemons unless already present."""
    if _have_all(install_root) and not build_rust:
        return [Result("binaries", "ok", f"all {len(DAEMONS)} daemons already present in "
                                          f"{bin_dir(install_root)}")]
    if build_rust:
        return build_release_binaries(install_root)
    return fetch_release_binaries(install_root, tag)
