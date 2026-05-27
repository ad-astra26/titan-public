"""Phase 4 — inference provider selection.

Per locked decision #3 (RFP_Titan_setup_release.md 2026-05-22):
    --default: auto-detect local Ollama at http://localhost:11434; if reachable,
    use it (most sovereign — no key). Otherwise prompt for an OpenRouter API key.

Stdlib-only:
- urllib for the Ollama liveness probe (HTTP GET /api/tags, 2s timeout)
- line-based upsert into ~/.titan/secrets.toml (the file is flat `key = "value"`;
  no nested tables; no need for a TOML library)
- secrets file lands at 0600 perms; parent dir 0700
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from .preflight import Result
from .ui import cprint

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OPENROUTER_KEY_PREFIX = "sk-or-"
SECRETS_PATH = Path.home() / ".titan" / "secrets.toml"


# ── Ollama liveness probe ───────────────────────────────────────────────────


def ollama_alive(host: str = OLLAMA_DEFAULT_HOST, timeout_s: float = 2.0) -> bool:
    """True iff `host/api/tags` responds with HTTP 200 within timeout_s.

    Uses urllib (stdlib) so this runs from the system Python interpreter that
    invokes setup_titan — before the venv exists.
    """
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# ── secrets.toml upsert ─────────────────────────────────────────────────────


def upsert_secret(key: str, value: str, path: Path = SECRETS_PATH) -> None:
    """Idempotent set of `key = "value"` in a flat TOML file.

    Existing line for `key` is replaced in place; new key is appended at file
    end. Creates the parent dir (0700) + the file (0600) if missing. The flat
    schema matches the existing ~/.titan/secrets.toml exactly — no library
    dependency, runs on system Python.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        path.touch(mode=0o600)
    lines = path.read_text().splitlines()
    new_line = f'{key} = "{value}"'
    replaced = False
    out: list[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
            out.append(new_line)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(new_line)
    path.write_text("\n".join(out) + "\n")
    path.chmod(0o600)


# ── OpenRouter key prompt ──────────────────────────────────────────────────


def prompt_openrouter_key(prompt_fn=input) -> str:
    """Prompt for an OpenRouter key; loop until it has the right shape.

    `prompt_fn` is injected for unit testing (the default is built-in `input`).
    Heuristic: starts with 'sk-or-' and is at least 14 chars total.
    """
    while True:
        try:
            ans = prompt_fn(f"  OpenRouter API key (starts with '{OPENROUTER_KEY_PREFIX}'): ").strip()
        except EOFError:
            raise SystemExit("setup_titan: stdin closed during OpenRouter key prompt")
        if ans.startswith(OPENROUTER_KEY_PREFIX) and len(ans) >= len(OPENROUTER_KEY_PREFIX) + 8:
            return ans
        print(f"    that doesn't look like an OpenRouter key (expected '{OPENROUTER_KEY_PREFIX}…' with ≥8 more chars)")


# ── Phase 4 body ───────────────────────────────────────────────────────────


def run_inference_phase(*, default: bool, secrets_path: Path = SECRETS_PATH) -> list[Result]:
    """Phase 4 — pick provider + persist credentials.

    Returns a Result list (matching the preflight pattern). `--default` skips
    interactive confirmation when local Ollama is reachable; interactive mode
    asks before falling through (user can force OpenRouter even with local
    Ollama present).
    """
    alive = ollama_alive()
    if alive:
        if default:
            cprint(f"  Local Ollama detected at {OLLAMA_DEFAULT_HOST} — using it (most sovereign).",
                   role="success")
            upsert_secret("inference_provider", "ollama_cloud", path=secrets_path)
            upsert_secret("ollama_host", OLLAMA_DEFAULT_HOST, path=secrets_path)
            return [Result("inference", "ok", f"local Ollama at {OLLAMA_DEFAULT_HOST}")]
        try:
            ans = input(f"  Local Ollama detected at {OLLAMA_DEFAULT_HOST}. Use it? [Y/n]: ").strip().lower()
        except EOFError:
            raise SystemExit("setup_titan: stdin closed during inference prompt")
        if ans in ("", "y", "yes"):
            upsert_secret("inference_provider", "ollama_cloud", path=secrets_path)
            upsert_secret("ollama_host", OLLAMA_DEFAULT_HOST, path=secrets_path)
            return [Result("inference", "ok", f"local Ollama at {OLLAMA_DEFAULT_HOST}")]

    cprint("  Falling back to OpenRouter (cloud-hosted; requires API key).", role="warning")
    key = prompt_openrouter_key()
    upsert_secret("openrouter_api_key", key, path=secrets_path)
    upsert_secret("inference_provider", "openrouter", path=secrets_path)
    return [Result("inference", "ok", f"OpenRouter key written to {secrets_path} (0600)")]
