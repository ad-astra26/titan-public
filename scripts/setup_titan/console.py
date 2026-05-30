"""Console phase — install the TC² Console Agent + its prebuilt SPA (W8).

Unlike the opt-in Observatory (heavy Next.js showcase), the Console Agent is
the DEFAULT owner UI: stdlib-only Python, no node at RUNTIME, its own crash
domain. It installs `titan-console.service` (NOT Guardian-supervised) and
serves the prebuilt SPA bundle (committed at titan-console/dist) on :7799.

The SPA bundle ships prebuilt in the repo, so the target box never needs node.
If the bundle is somehow missing AND node is present, we build it on-box as a
fallback; otherwise the agent still runs (it serves a minimal placeholder page
and all the JSON/ops routes work).

Stdlib-only (subprocess + urllib) — runs on the system interpreter.
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .preflight import Result
from .ui import cprint

UNIT_NAME = "titan-console.service"
UNIT_DEST = Path("/etc/systemd/system") / UNIT_NAME
TEMPLATE = Path(__file__).resolve().parents[2] / "titan_console" / "titan-console.service.template"
SPA_DIR = "titan-console"
DEFAULT_PORT = 7799
TOKEN_PATH = Path(os.path.expanduser("~/.titan/console_token"))


def dist_dir(install_root: Path) -> Path:
    return install_root / SPA_DIR / "dist"


def ensure_token() -> Result:
    """Generate ~/.titan/console_token (0600) if absent — gates mutations."""
    try:
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(TOKEN_PATH.parent, 0o700)
    except OSError:
        pass
    if TOKEN_PATH.exists() and TOKEN_PATH.read_text().strip():
        return Result("console", "ok", "console token present")
    try:
        TOKEN_PATH.write_text(secrets.token_urlsafe(32))
        os.chmod(TOKEN_PATH, 0o600)
    except OSError as e:
        return Result("console", "warn", f"could not write console token: {e}",
                      "Mutations stay open on localhost; set ~/.titan/console_token to lock them.")
    return Result("console", "ok", f"console token generated → {TOKEN_PATH}")


def ensure_bundle(install_root: Path) -> Result:
    """Prefer the committed prebuilt dist; on-box build only as a fallback."""
    dist = dist_dir(install_root)
    if (dist / "index.html").exists():
        return Result("console", "ok", f"prebuilt SPA bundle present → {dist}")
    spa = install_root / SPA_DIR
    if not shutil.which("npm"):
        return Result("console", "warn", "SPA bundle missing and npm not installed",
                      "The agent runs fine (JSON/ops routes work); for the web UI, "
                      "install Node and run `npm ci && npm run build` in titan-console/.")
    cprint("  Building TC² SPA bundle (one-time, node present)…", role="text_strong")
    for cmd in (["npm", "ci", "--no-audit", "--no-fund"], ["npm", "run", "build"]):
        if subprocess.run(cmd, cwd=spa).returncode != 0:
            return Result("console", "warn", f"`{' '.join(cmd)}` failed in titan-console/",
                          "Agent still runs; build the SPA later for the web UI.")
    return Result("console", "ok", f"SPA bundle built → {dist}")


def render_unit(*, install_root: Path, user: str, venv_python: str,
                port: int, api_base: str, bind_host: str) -> str:
    tmpl = TEMPLATE.read_text()
    return (tmpl
            .replace("{{USER}}", user)
            .replace("{{INSTALL_ROOT}}", str(install_root))
            .replace("{{VENV_PYTHON}}", venv_python)
            .replace("{{BIND_HOST}}", bind_host)
            .replace("{{PORT}}", str(port))
            .replace("{{API_BASE}}", api_base)
            .replace("{{DIST_DIR}}", str(dist_dir(install_root))))


def _sudo(cmd: list[str], *, explain: str) -> int:
    cprint(f"  [sudo] {explain}", role="warning")
    cprint(f"        $ sudo {' '.join(cmd)}", role="text_muted")
    return subprocess.run(["sudo", *cmd]).returncode


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/console/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run_console_phase(state: dict, install_root: Path, *, user: str,
                      port: int = DEFAULT_PORT, api_base: str = "http://127.0.0.1:7777",
                      bind_host: str = "127.0.0.1") -> list[Result]:
    """Install + start the Console Agent. Default UI — always installed.

    Decoupling: runs on the SYSTEM python (stdlib-only) so a broken Titan venv
    can't take the console down with it.
    """
    results: list[Result] = [ensure_token(), ensure_bundle(install_root)]

    sys_python = shutil.which("python3") or "/usr/bin/python3"
    unit = render_unit(install_root=install_root, user=user, venv_python=sys_python,
                       port=port, api_base=api_base, bind_host=bind_host)
    tmp = install_root / "data" / f".{UNIT_NAME}.staged"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(unit)
    steps = [
        (["cp", str(tmp), str(UNIT_DEST)], f"install {UNIT_NAME}"),
        (["chmod", "644", str(UNIT_DEST)], "set unit perms 0644"),
        (["systemctl", "daemon-reload"], "reload systemd"),
        (["systemctl", "enable", "--now", UNIT_NAME], "enable + start Console Agent"),
    ]
    for cmd, explain in steps:
        if _sudo(cmd, explain=explain) != 0:
            tmp.unlink(missing_ok=True)
            return results + [Result("console", "fail", f"`sudo {' '.join(cmd)}` failed",
                                     f"Check: journalctl -u {UNIT_NAME} -n 50 --no-pager")]
    tmp.unlink(missing_ok=True)
    state["console_enabled"] = True

    if _health_ok(port):
        results.append(Result("console", "ok",
                              f"Console Agent live at http://{bind_host}:{port} "
                              f"(stays up even if the Titan is down)"))
    else:
        results.append(Result("console", "warn",
                              f"unit started but :{port}/console/health not 200 yet",
                              f"Check: journalctl -u {UNIT_NAME} -n 50 --no-pager"))
    return results
