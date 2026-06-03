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

from .config_model import set_by_dotted
from .inference import SECRETS_PATH, read_secret, upsert_secret
from .modes import Mode
from .preflight import Result
from .ui import cprint

# Per-mode Solana network the installer writes into config.toml so the born
# Titan operates on the network the user chose. (Post-install bug: a devnet
# birth ran with the example's mainnet config → balance/vault read mainnet, the
# wrong chain.) (solana_network, public_rpc_url).
_NET_BY_MODE = {
    Mode.MAINNET: ("mainnet-beta", "https://api.mainnet-beta.solana.com"),
    Mode.DEVNET:  ("devnet",        "https://api.devnet.solana.com"),
    Mode.LOCAL:   ("devnet",        "https://api.devnet.solana.com"),  # no on-chain; harmless
}


def config_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml"


def _apply_network_config(install_root: Path, mode, state: dict | None) -> list[Result]:
    """Set [network].solana_network + the RPC per the chosen mode, so the Titan
    boots onto the right chain (every internal setting the user's selection
    implies, not the example's mainnet default)."""
    cfg = config_path(install_root)
    network, public = _NET_BY_MODE.get(mode, _NET_BY_MODE[Mode.MAINNET])
    out: list[Result] = []
    if set_by_dotted(cfg, "network.solana_network", network):
        out.append(Result("network", "ok", f"solana_network = {network} (mode {getattr(mode,'value',mode)})"))
    # The premium RPC captured in phase_2 (user's Helius/QuickNode) is the
    # EFFECTIVE endpoint; Helius resolves both nets via subdomain.
    rpc = ((state or {}).get("solana_rpc") or "").strip()
    if rpc and set_by_dotted(cfg, "network.premium_rpc_url", rpc):
        out.append(Result("network_rpc", "ok", "premium_rpc_url set from install"))
    # Public-fallback array → the per-mode public endpoint (the example ships
    # mainnet; repoint it so a devnet/local Titan never falls back to mainnet).
    try:
        txt = cfg.read_text()
        new = txt.replace("https://api.mainnet-beta.solana.com", public)
        if new != txt:
            cfg.write_text(new)
            out.append(Result("network_public", "ok", f"public_rpc_urls → {public}"))
    except OSError:
        pass
    return out


def example_path(install_root: Path) -> Path:
    return install_root / "titan_hcl" / "config.toml.example"


def run_config_seed_phase(install_root: Path, mode=None, state: dict | None = None, *,
                          secrets_path: Path = SECRETS_PATH) -> list[Result]:
    """Seed config.toml from the example (if absent) + ensure [api].internal_key
    + write the per-mode Solana network settings (so the Titan boots on the
    network the user chose)."""
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

    # Per-mode Solana network (devnet/mainnet/local) — only when the caller
    # passes a mode (the install flow does; `setup_titan config` re-seed does not).
    if mode is not None:
        results += _apply_network_config(install_root, mode, state)
    return results
