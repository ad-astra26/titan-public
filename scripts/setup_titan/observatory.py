"""Observatory phase — fetch + run the prebuilt Next.js standalone bundle (#15).

Opt-in web UI. The bundle is built in CI (release.yml `build-observatory` job) and
attached to the GitHub Release as `titan-observatory-<tag>.tar.gz`. The installer
downloads + SHA256-verifies + extracts it, installs a `titan-observatory.service`
systemd unit that runs `node server.js` on :3000, and starts it — NO on-box build,
no full node_modules (which is why a 4GB box can host it: it only RUNS the
~150MB bundle, it never builds it).

Data path (works out of the box): the Titan backend writes `data/observatory.db`;
the `/v6` API serves it; the bundle's `next.config` rewrites proxy `/v6` →
`127.0.0.1:7777` (baked at build time), so the frontend reads the LOCAL Titan
automatically — no per-user URL wiring needed for the localhost case. For remote
access the user puts their own reverse proxy / TLS in front of :3000.

Requires Node (>=18) on PATH; the phase apt-installs it if missing. Stdlib-only
(urllib + tarfile + subprocess) — runs on the system interpreter.
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

RELEASE_BASE = "https://github.com/ad-astra26/titan-public/releases/download"
APP_DIRNAME = "titan-observatory-app"      # extracted standalone bundle lives here
UNIT_NAME = "titan-observatory.service"
UNIT_DEST = Path("/etc/systemd/system") / UNIT_NAME
DEFAULT_PORT = 3000


def bundle_name(tag: str) -> str:
    return f"titan-observatory-{tag}.tar.gz"


def app_dir(install_root: Path) -> Path:
    return install_root / APP_DIRNAME


# ── fetch + verify + extract ─────────────────────────────────────────────────


def _download(url: str, dest: Path, timeout_s: int = 120) -> None:
    with urllib.request.urlopen(url, timeout=timeout_s) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_observatory_bundle(install_root: Path, tag: str) -> list[Result]:
    """Download titan-observatory-<tag>.tar.gz + verify SHA256 + extract."""
    if not tag or tag in ("main", "HEAD"):
        return [Result("observatory", "fail", f"no release tag (got {tag!r})",
                       "Observatory ships as a release asset; install with --tag vX.Y.Z.")]
    tarball = bundle_name(tag)
    base = f"{RELEASE_BASE}/{tag}/{tarball}"
    staging = install_root / "data" / ".obs_dl"
    staging.mkdir(parents=True, exist_ok=True)
    tgz, sha = staging / tarball, staging / f"{tarball}.sha256"
    try:
        _download(base, tgz)
        _download(f"{base}.sha256", sha)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return [Result("observatory", "fail", f"download failed: {e}",
                       f"Confirm the release {tag} has {tarball} attached "
                       "(the build-observatory CI job may still be running).")]

    expected = sha.read_text().split()[0]
    actual = _sha256(tgz)
    if expected != actual:
        tgz.unlink(missing_ok=True)
        return [Result("observatory", "fail", "SHA256 mismatch on Observatory bundle",
                       "Refusing to extract an unverified bundle. Re-download / re-cut the release.")]

    dest = app_dir(install_root)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    import tarfile
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(dest)            # bundle authored by our own CI; trusted post-SHA256
    tgz.unlink(missing_ok=True)
    sha.unlink(missing_ok=True)
    if not (dest / "server.js").exists():
        return [Result("observatory", "fail", "bundle missing server.js after extract",
                       "The tarball layout is unexpected; check the build-observatory job.")]
    return [Result("observatory", "ok", f"bundle {tag} fetched + verified → {dest}")]


# ── node prerequisite ────────────────────────────────────────────────────────


def ensure_node() -> Result:
    if shutil.which("node"):
        ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        return Result("node", "ok", f"node present ({ver})")
    cprint("  Installing Node.js (Observatory runtime)…", role="text_strong")
    rc = subprocess.run(["sudo", "apt-get", "install", "-y", "nodejs"]).returncode
    if rc != 0 or not shutil.which("node"):
        return Result("node", "fail", "could not install nodejs",
                      "Install Node >=18 manually, then `setup_titan repair`.")
    return Result("node", "ok", "nodejs installed")


# ── systemd unit ─────────────────────────────────────────────────────────────


def render_observatory_unit(*, app_path: Path, user: str, port: int = DEFAULT_PORT) -> str:
    """Unit for the standalone server. Binds 127.0.0.1 only — the user fronts it
    with their own reverse proxy / TLS for remote access."""
    node = shutil.which("node") or "/usr/bin/node"
    return f"""\
# {UNIT_NAME} — Titan Observatory (Next.js standalone bundle). Generated by setup_titan.
[Unit]
Description=Titan Observatory web UI
After=titan.service network.target
Wants=titan.service

[Service]
Type=simple
User={user}
WorkingDirectory={app_path}
Environment=NODE_ENV=production
Environment=PORT={port}
Environment=HOSTNAME=127.0.0.1
ExecStart={node} {app_path}/server.js
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


def _sudo(cmd: list[str], *, explain: str) -> int:
    cprint(f"  [sudo] {explain}", role="warning")
    cprint(f"        $ sudo {' '.join(cmd)}", role="text_muted")
    return subprocess.run(["sudo", *cmd]).returncode


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


# ── phase body ───────────────────────────────────────────────────────────────


def run_observatory_phase(state: dict, install_root: Path, *,
                          tag: str, user: str, port: int = DEFAULT_PORT) -> list[Result]:
    """Install + start the Observatory IF the user opted in during comms."""
    if not state.get("observatory_enabled"):
        return [Result("observatory", "ok", "skipped (opt-in; not enabled)")]

    results: list[Result] = []
    node_res = ensure_node()
    results.append(node_res)
    if node_res.severity == "fail":
        return results

    fetched = fetch_observatory_bundle(install_root, tag)
    results.extend(fetched)
    if any(r.severity == "fail" for r in fetched):
        return results

    unit = render_observatory_unit(app_path=app_dir(install_root), user=user, port=port)
    tmp = install_root / "data" / f".{UNIT_NAME}.staged"
    tmp.write_text(unit)
    steps = [
        (["cp", str(tmp), str(UNIT_DEST)], f"install {UNIT_NAME}"),
        (["chmod", "644", str(UNIT_DEST)], "set unit perms 0644"),
        (["systemctl", "daemon-reload"], "reload systemd"),
        (["systemctl", "enable", "--now", UNIT_NAME], "enable + start Observatory"),
    ]
    for cmd, explain in steps:
        if _sudo(cmd, explain=explain) != 0:
            tmp.unlink(missing_ok=True)
            return results + [Result("observatory", "fail", f"`sudo {' '.join(cmd)}` failed")]
    tmp.unlink(missing_ok=True)

    if _health_ok(port):
        results.append(Result("observatory", "ok",
                              f"Observatory live at http://127.0.0.1:{port} "
                              f"(reads the local Titan via /v6 → :7777)"))
    else:
        results.append(Result("observatory", "warn",
                              f"unit started but :{port} not 200 yet — it may still be warming",
                              f"Check: journalctl -u {UNIT_NAME} -n 50 --no-pager"))
    return results
