"""Phase H — research runtime services: OCR system deps + the SearXNG search
service (rFP_setup_titan_auto_provisioner §1.2/§7; INV-PROV-8/9).

SearXNG is the search backbone for `knowledge_worker` + `SageRecorder` (queried
over base httpx; the heavier scrape/distill libs are the venv-phase `[research]`
extra). This phase installs the OCR apt deps `unstructured` needs for full
PDF/OCR + runs the SearXNG Docker container — idempotently, vCPU-safe,
mode-independent. `--minimal` skips it.

vCPU premise (INV-PROV-9): NO chromium browser, NO local Ollama — only the
container runtime + OCR libs + the SearXNG image. Privilege (INV-PROV-7): the
apt installs are explicit, streamed `sudo` commands the user sees.
"""
from __future__ import annotations

import secrets
import shutil
import subprocess
from pathlib import Path

from .preflight import Result
from .ui import cprint

SEARXNG_IMAGE = "searxng/searxng:latest"
SEARXNG_CONTAINER = "searxng"
SEARXNG_PORT = 8080
# unstructured's full PDF/OCR path (Maker: provision the fuller OCR stack, beyond
# T1's current libmagic1-only set).
OCR_APT_DEPS = ("libmagic1", "poppler-utils", "tesseract-ocr")
# docker.io = the container runtime for SearXNG.
RUNTIME_APT_DEPS = ("docker.io",)

_SEARXNG_SETTINGS_TMPL = """\
use_default_settings: true
server:
  secret_key: "{secret}"
  base_url: http://localhost:{port}/
search:
  safe_search: 0
  default_lang: en
  formats:
    - html
    - json
"""


def _run(cmd, *, shell: bool = False) -> None:
    """Streamed (never capture-and-hang); raises CalledProcessError on failure."""
    subprocess.run(cmd, shell=shell, check=True)


def _capture(cmd) -> str:
    """Run a read-only probe; return stdout, or '' on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _container_state(name: str) -> str:
    """'running' | 'stopped' | 'absent' for a docker container by name."""
    if name in _capture(["docker", "ps", "--format", "{{.Names}}"]).split():
        return "running"
    if name in _capture(["docker", "ps", "-a", "--format", "{{.Names}}"]).split():
        return "stopped"
    return "absent"


def _ensure_searxng_settings(install_root: Path) -> Path:
    """Write the SearXNG settings.yml (JSON format ON) once; keep the secret
    stable across re-runs (regenerating would needlessly churn CSRF tokens)."""
    cfg_dir = install_root / "searxng"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    settings = cfg_dir / "settings.yml"
    if not settings.exists():
        settings.write_text(_SEARXNG_SETTINGS_TMPL.format(
            secret=secrets.token_hex(32), port=SEARXNG_PORT))
    return cfg_dir


def run_services_phase(install_root: Path, *, minimal: bool = False) -> list[Result]:
    """Provision the research services. `--minimal` skips entirely (returns a
    warn, not a fail — the Titan still boots, web-search is just inert until set
    up). A real failure returns a 'fail' Result → the walker halts (resumable)."""
    if minimal:
        return [Result("services", "warn",
                       "--minimal: research services (SearXNG + OCR deps) skipped — "
                       "knowledge_worker web-search stays inert until provisioned.",
                       "Re-run without --minimal (or set up SearXNG + pip install -e .[research] by hand).")]

    results: list[Result] = []

    # 1. OCR system deps + the docker runtime (explicit sudo; INV-PROV-7).
    pkgs = (*OCR_APT_DEPS, *RUNTIME_APT_DEPS)
    cprint(f"  Installing OCR system deps + docker runtime (sudo apt-get): {', '.join(pkgs)}…",
           role="text_strong")
    try:
        _run(["sudo", "apt-get", "install", "-y", *pkgs])
    except subprocess.CalledProcessError as e:
        return [Result("services", "fail", f"apt install exited {e.returncode}",
                       f"Install manually: sudo apt-get install -y {' '.join(pkgs)}. Then re-run with --resume.")]
    results.append(Result("ocr+docker", "ok", f"installed {', '.join(pkgs)}"))

    if shutil.which("docker") is None:
        return results + [Result("searxng", "fail", "docker not on PATH after install",
                                 "Ensure docker.io installed + your user can run docker "
                                 "(sudo usermod -aG docker $USER; re-login). Then re-run with --resume.")]

    # 2. SearXNG settings (JSON format on) + idempotent container.
    cfg_dir = _ensure_searxng_settings(install_root)
    state = _container_state(SEARXNG_CONTAINER)
    if state == "running":
        results.append(Result("searxng", "ok",
                              f"already running ({SEARXNG_IMAGE}, :{SEARXNG_PORT})"))
        return results

    cprint(f"  Starting SearXNG ({SEARXNG_IMAGE}) on :{SEARXNG_PORT}…", role="text_strong")
    try:
        if state == "stopped":
            _run(["docker", "start", SEARXNG_CONTAINER])
        else:
            _run(["docker", "run", "-d", "--name", SEARXNG_CONTAINER,
                  "--restart=unless-stopped", "-p", f"{SEARXNG_PORT}:{SEARXNG_PORT}",
                  "-v", f"{cfg_dir}:/etc/searxng", SEARXNG_IMAGE])
    except subprocess.CalledProcessError as e:
        verb = "start" if state == "stopped" else "run"
        return results + [Result("searxng", "fail", f"docker {verb} exited {e.returncode}",
                                 f"Inspect docker output; check :{SEARXNG_PORT} is free + the image is "
                                 "pullable. Then re-run with --resume.")]
    results.append(Result("searxng", "ok",
                          f"SearXNG up ({SEARXNG_IMAGE}, :{SEARXNG_PORT}, JSON format on)"))
    return results
