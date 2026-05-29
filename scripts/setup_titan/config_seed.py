"""Config-seed phase — make config.toml exist + mint the chat auth key (#8).

Two real-world install gaps this closes (surfaced on the 2026-05-29 fresh-box
test):

1. config.toml is config_loader's REQUIRED Layer 2 — if it's absent the whole
   config resolves to ``{}`` and every keyed feature (chat auth, inference
   provider, channels) silently stays unset. A fresh PUBLIC clone ships only
   ``config.toml.example`` (the real config.toml is public-sync-excluded because
   on dev it carries secrets), so the installer must seed it.

2. ``[api].internal_key`` (chat auth, X-Titan-Internal-Key) is empty out of the
   box. Generate one into ~/.titan/secrets.toml [api]; config_loader deep-merges
   it to ``config["api"]["internal_key"]`` at runtime.

Runs BEFORE the inference phase (which writes the non-secret provider selection
into the seeded config.toml). Stdlib-only — executes on the system interpreter.
"""
from __future__ import annotations

import secrets as _secrets
import shutil
from pathlib import Path

from .inference import SECRETS_PATH, read_secret, upsert_secret
from .preflight import Result
from .ui import cprint


def config_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml"


def example_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml.example"


def run_config_seed_phase(install_root: Path, *,
                          secrets_path: Path = SECRETS_PATH) -> list[Result]:
    """Seed config.toml from the example (if absent) + ensure [api].internal_key."""
    results: list[Result] = []
    cfg = config_path(install_root)
    example = example_path(install_root)

    if cfg.exists():
        results.append(Result("config", "ok", f"config.toml already present ({cfg})"))
    elif example.exists():
        shutil.copyfile(example, cfg)
        cprint(f"  Seeded config.toml from config.toml.example → {cfg}", role="success")
        results.append(Result("config", "ok", "seeded config.toml from example"))
    else:
        return [Result("config", "fail",
                       f"neither config.toml nor config.toml.example in {cfg.parent}",
                       "Incomplete checkout — re-clone the repo at the release tag.")]

    # [api].internal_key — generate once, idempotent on re-run.
    if read_secret("api", "internal_key", path=secrets_path):
        results.append(Result("internal_key", "ok", "already set in secrets.toml [api]"))
    else:
        key = _secrets.token_urlsafe(32)
        upsert_secret("api", "internal_key", key, path=secrets_path)
        cprint(f"  Generated chat auth key → {secrets_path} [api].internal_key (0600)",
               role="success")
        results.append(Result("internal_key", "ok",
                              f"generated + written to {secrets_path} [api] (0600)"))
    return results
