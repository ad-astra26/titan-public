"""Phase 4 — inference provider selection.

Per locked decision #3 (RFP_Titan_setup_release.md 2026-05-22):
    --default: auto-detect local Ollama at http://localhost:11434; if reachable,
    use it (most sovereign — no key). Otherwise prompt for an OpenRouter API key.

Stdlib-only:
- urllib for the Ollama liveness/model probe (HTTP GET /api/tags, 2s timeout)
- SECTION-AWARE line-based upsert into ~/.titan/secrets.toml. config_loader
  deep-merges secrets into the config TREE, so each secret must live under the
  SAME [section] the runtime reads (e.g. [inference] openrouter_api_key,
  [channels] telegram_bot_token, [api] internal_key). A flat key never reaches
  the sectioned config and the feature stays silently unconfigured.
- secrets file lands at 0600 perms; parent dir 0700
"""
from __future__ import annotations

import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from .preflight import Result
from .ui import cprint

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OPENROUTER_KEY_PREFIX = "sk-or-"
SECRETS_PATH = Path.home() / ".titan" / "secrets.toml"


# ── Ollama liveness / model probe ───────────────────────────────────────────


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


def ollama_models(host: str = OLLAMA_DEFAULT_HOST, timeout_s: float = 2.0) -> list[str]:
    """Model names a local Ollama has pulled (GET /api/tags → models[].name).

    Empty list on any error. Used to wire a real, locally-available model into
    config.toml so the ollama_cloud provider (pointed at localhost) doesn't ask
    for a cloud-only model that isn't installed.
    """
    import json
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout_s) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return []


# ── secrets.toml: section-aware upsert + read ───────────────────────────────


def _toml_quote(value: str) -> str:
    """Render a string as a TOML basic string (escape backslash + double-quote)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def upsert_secret(section: str, key: str, value: str, path: Path = SECRETS_PATH) -> None:
    """Idempotent set of `key = "value"` UNDER `[section]` in a TOML file.

    Section-aware, stdlib-only (runs on system Python, before the venv). Matches
    the sectioned schema config_loader deep-merges into the config tree:
      - key present under [section]  → value replaced in place
      - [section] exists, key absent → key appended at the end of that block
      - [section] absent             → `[section]` + key appended at file end
    Creates the parent dir (0700) + file (0600). String values only (every
    secret is a string); the value is TOML-quoted.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        path.touch(mode=0o600)
    lines = path.read_text().splitlines()
    new_line = f"{key} = {_toml_quote(value)}"
    header = f"[{section}]"

    def _is_header(ln: str) -> bool:
        s = ln.strip()
        return s.startswith("[") and s.endswith("]")

    # Locate the [section] header.
    sec_idx = next((i for i, ln in enumerate(lines) if ln.strip() == header), None)

    if sec_idx is None:                       # section absent → append a new block
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header, new_line])
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o600)
        return

    # Scan the section body (until the next header or EOF).
    end = next((i for i in range(sec_idx + 1, len(lines)) if _is_header(lines[i])), len(lines))
    last_content = sec_idx                     # insertion point (after last non-blank line)
    for i in range(sec_idx + 1, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
            lines[i] = new_line               # replace in place
            path.write_text("\n".join(lines) + "\n")
            path.chmod(0o600)
            return
        if lines[i].strip():
            last_content = i
    lines.insert(last_content + 1, new_line)   # key absent in existing section
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def read_secret(section: str, key: str, path: Path = SECRETS_PATH) -> str | None:
    """Return the value of `[section].key` in the secrets file, or None.

    Used for idempotency checks (don't regenerate an internal_key that exists).
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        val = data.get(section, {}).get(key)
        return str(val) if val not in (None, "") else None
    except (OSError, tomllib.TOMLDecodeError):
        return None


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


def prompt_ollama_cloud_key(prompt_fn=input) -> str:
    """Prompt for an Ollama Cloud API key (https://ollama.com → API keys).

    Ollama Cloud keys are opaque tokens (no fixed prefix); accept any long,
    space-free string. `prompt_fn` injected for tests.
    """
    while True:
        try:
            ans = prompt_fn("  Ollama Cloud API key (from https://ollama.com → API keys): ").strip()
        except EOFError:
            raise SystemExit("setup_titan: stdin closed during Ollama Cloud key prompt")
        if len(ans) >= 20 and " " not in ans:
            return ans
        print("    that doesn't look like an Ollama Cloud key (expected a long token, no spaces)")


# ── Phase 4 body ───────────────────────────────────────────────────────────


def _config_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml"


def _set_inference_config(install_root: Path, **kv: str) -> list[str]:
    """Set [inference] keys in config.toml (provider/base_url/model — NOT secret).

    config.toml is seeded by the config-seed phase before this one runs, so the
    [inference] keys already exist (from config.toml.example). Returns the list
    of dotted keys that could not be set (missing in this config version).
    """
    from . import config_model as cm
    cfg = _config_path(install_root)
    misses: list[str] = []
    if not cfg.exists():
        return [f"inference.{k}" for k in kv]   # seed phase should have created it
    for k, v in kv.items():
        if not cm.set_by_dotted(cfg, f"inference.{k}", v):
            misses.append(f"inference.{k}")
    return misses


def _wire_local_ollama(install_root: Path) -> list[Result]:
    """Point the ollama_cloud provider at a local Ollama (base_url + real model)."""
    base_url = f"{OLLAMA_DEFAULT_HOST}/v1"
    models = ollama_models()
    settings = {"inference_provider": "ollama_cloud", "ollama_cloud_base_url": base_url}
    if models:
        settings["ollama_cloud_light_model"] = models[0]
        settings["ollama_cloud_heavy_model"] = models[0]
    misses = _set_inference_config(install_root, **settings)
    detail = (f"local Ollama at {OLLAMA_DEFAULT_HOST} (model: {models[0]})" if models
              else f"local Ollama at {OLLAMA_DEFAULT_HOST}")
    res = [Result("inference", "ok", detail)]
    if not models:
        res.append(Result("ollama_model", "warn",
                          "no local Ollama models found (GET /api/tags empty)",
                          "Pull one first, e.g. `ollama pull llama3.1`, then re-run "
                          "`setup_titan config --set inference.ollama_cloud_light_model=<model>`."))
    if misses:
        res.append(Result("inference_cfg", "warn", f"could not set in config.toml: {', '.join(misses)}",
                          "These keys may be absent in this config version; set them via `setup_titan config`."))
    return res


def _wire_cloud_key(install_root: Path, provider: str, key: str,
                    secrets_path: Path) -> list[Result]:
    """Persist a hosted-provider key (secrets.toml [inference]) + set the provider
    (config.toml [inference]). `provider` ∈ {'ollama_cloud','openrouter'}."""
    secret_key = "ollama_cloud_api_key" if provider == "ollama_cloud" else "openrouter_api_key"
    label = "Ollama Cloud" if provider == "ollama_cloud" else "OpenRouter"
    upsert_secret("inference", secret_key, key, path=secrets_path)
    misses = _set_inference_config(install_root, inference_provider=provider)
    res = [Result("inference", "ok",
                  f"{label} key → {secrets_path} [inference] (0600); provider → config.toml")]
    if misses:
        res.append(Result("inference_cfg", "warn", f"could not set in config.toml: {', '.join(misses)}",
                          f"Set via `setup_titan config --set inference.inference_provider={provider}`."))
    return res


def run_inference_phase(*, default: bool, install_root: Path,
                        secrets_path: Path = SECRETS_PATH) -> list[Result]:
    """Phase 4 — pick provider; secret keys → secrets.toml [inference], the
    (non-secret) provider/base_url/model selection → config.toml [inference].

    Order: (1) if a local Ollama is reachable, offer it (no key — most sovereign);
    (2) otherwise choose a HOSTED provider — Ollama Cloud (the fleet default) or
    OpenRouter. Both require the user's own API key (even OpenRouter's free tier
    needs one), so we collect it now — a Titan must be able to chat the moment it
    boots. `--default` auto-accepts a reachable local Ollama; the hosted choice is
    always interactive because a key is unavoidable.
    """
    alive = ollama_alive()
    if alive:
        use_local = default
        if not default:
            try:
                ans = input(f"  Local Ollama detected at {OLLAMA_DEFAULT_HOST}. Use it? [Y/n]: ").strip().lower()
            except EOFError:
                raise SystemExit("setup_titan: stdin closed during inference prompt")
            use_local = ans in ("", "y", "yes")
        if use_local:
            cprint(f"  Using local Ollama at {OLLAMA_DEFAULT_HOST} (most sovereign — no key).",
                   role="success")
            return _wire_local_ollama(install_root)

    # No usable local Ollama → choose a hosted provider (both need the user's key).
    cprint("  No local Ollama in use — choose a hosted inference provider:", role="text_strong")
    cprint("    [1] Ollama Cloud  (ollama.com — hosted, OpenAI-compatible; the fleet default)",
           role="text_muted")
    cprint("    [2] OpenRouter    (openrouter.ai — many models incl. a rate-limited free tier)",
           role="text_muted")
    try:
        choice = (input("  Provider [1/2, default 1]: ").strip() or "1")
    except EOFError:
        raise SystemExit("setup_titan: stdin closed during inference provider choice")
    if choice == "2":
        return _wire_cloud_key(install_root, "openrouter", prompt_openrouter_key(), secrets_path)
    return _wire_cloud_key(install_root, "ollama_cloud", prompt_ollama_cloud_key(), secrets_path)
