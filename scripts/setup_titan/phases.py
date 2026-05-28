"""7-phase install walker — orchestrates the wizard past preflight.

Phase 1 is `preflight.run_preflight` (already shipped in W1.a).
Phase 2-7 owners per RFP_Titan_setup_release.md §W1 sub-phase plan:

    Phase 2 — Mode confirm + Maker wallet + Solana RPC  (W1.b — this module)
    Phase 3 — Venv + `pip install -e .` + import check  (W1.b — this module)
    Phase 4 — Inference autodetect                      (W1.c — inference.py)
    Phase 5 — Comms (Telegram req'd, X/Obs opt-in)      (W1.d — stub)
    Phase 6 — Genesis ceremony                          (W1.b — genesis_runner.py)
    Phase 7 — Systemd install + first start + health    (W1.e — stub)

Each phase function returns list[Result] (matching the preflight pattern).
The walker (`run_phases`) renders them through ui.cprint, persists per-phase
status via state.mark_phase, and halts on any 'fail' Result.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import state as install_state
from .comms import run_comms_phase
from .inference import run_inference_phase
from .genesis_runner import run_genesis_phase
from .modes import Mode, spec_for
from .preflight import Result, summarize
from .ui import HAZE, ANSI, PULSE, GROWTH, METAL, DANGER, cprint, section


# ── helpers ────────────────────────────────────────────────────────────────


def render(results: list[Result]) -> int:
    """Render a phase's results in the brand palette. Returns fail count."""
    sev_color = {"ok": GROWTH, "warn": HAZE, "fail": DANGER}
    sev_glyph = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    for r in results:
        col = sev_color[r.severity]
        gly = sev_glyph[r.severity]
        print(f"  {col}{gly}{ANSI.RESET} {ANSI.BOLD}{r.name:<10}{ANSI.RESET} {r.detail}")
        if r.remediation and r.severity != "ok":
            print(f"      {METAL}→ {r.remediation}{ANSI.RESET}")
    ok, warn, fail = summarize(results)
    summary_role = "error" if fail else "warning" if warn else "success"
    cprint(f"  {ok} ok · {warn} warn · {fail} fail", role=summary_role, bold=True)
    return fail


def prompt_line(question: str, *, default: str | None = None) -> str:
    """Single-line stdin prompt; raises SystemExit if stdin is closed."""
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"  {question}{suffix}: ").strip()
    except EOFError:
        raise SystemExit(f"setup_titan: stdin closed during prompt: {question!r}")
    return ans or (default or "")


def looks_like_solana_pubkey(s: str) -> bool:
    """Heuristic base58 32-byte check (32 bytes encodes to ~32-44 base58 chars)."""
    if not (32 <= len(s) <= 44):
        return False
    alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return all(c in alphabet for c in s)


def venv_python(install_root: Path) -> Path:
    return install_root / "test_env" / "bin" / "python"


def venv_pip(install_root: Path) -> Path:
    return install_root / "test_env" / "bin" / "pip"


# ── Phase 2 — Mode confirm + Maker wallet + Solana RPC ─────────────────────


def run_mode_phase(state: dict, mode: Mode, *, default: bool) -> list[Result]:
    """Confirm mode + capture Maker wallet + Solana RPC (modes 1/2 only).

    The mode itself is chosen earlier (CLI flag, `--default` provisional, or
    the wizard's earlier mode-chooser screen). This phase persists it and
    captures the per-mode credentials.
    """
    results: list[Result] = []
    spec = spec_for(mode)
    state["mode"] = mode.value

    # Modes 1/2 need a Maker wallet + Solana RPC URL.
    if spec.genesis_on_chain:
        wallet = state.setdefault("maker_wallet", "")
        if not wallet:
            if default:
                results.append(Result("wallet", "fail",
                                      "--default cannot proceed for mainnet/devnet without a Maker wallet",
                                      "Re-run interactively or set state['maker_wallet'] manually."))
                return results
            while True:
                wallet = prompt_line("Maker wallet (Solana pubkey, base58)")
                if looks_like_solana_pubkey(wallet):
                    state["maker_wallet"] = wallet
                    break
                print("    that doesn't look like a Solana pubkey (32–44 base58 chars)")
        results.append(Result("wallet", "ok", wallet))

        rpc = state.setdefault("solana_rpc", "")
        if not rpc:
            default_rpc = ("https://api.mainnet-beta.solana.com" if mode == Mode.MAINNET
                           else "https://api.devnet.solana.com")
            rpc = prompt_line("Solana RPC URL", default=default_rpc) if not default else default_rpc
            if not rpc.startswith(("http://", "https://")):
                results.append(Result("rpc", "fail", f"invalid URL: {rpc!r}",
                                      "Must start with http:// or https://"))
                return results
            state["solana_rpc"] = rpc
        results.append(Result("rpc", "ok", rpc))

    install_state.save(state)
    return results


# ── Phase 3 — Venv + Python deps ───────────────────────────────────────────


def run_venv_phase(install_root: Path) -> list[Result]:
    """Create venv at install_root/test_env; pip install -e . + import check.

    Uses the system `python3` for the venv creation step (any 3.11+ works);
    after creation, all pip + import calls go through the venv interpreter.
    """
    venv_dir = install_root / "test_env"
    results: list[Result] = []

    if not venv_dir.exists():
        cprint(f"  Creating virtualenv at {venv_dir}…", role="text_strong")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except subprocess.CalledProcessError as e:
            return [Result("venv", "fail", f"venv create exited {e.returncode}",
                           "Ensure python3.11-venv is installed: sudo apt install python3.11-venv")]
        results.append(Result("venv", "ok", f"created at {venv_dir}"))
    else:
        results.append(Result("venv", "ok", f"already present at {venv_dir}"))

    pip = venv_pip(install_root)
    if not pip.exists():
        return [Result("pip", "fail", f"pip missing at {pip}",
                       "venv creation appears incomplete; delete test_env/ and re-run.")]

    cprint("  Upgrading pip + setuptools + wheel…", role="text_strong")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"])

    cprint("  Installing Titan core dependencies (pip install -e .) — this can take several minutes…",
           role="text_strong")
    try:
        subprocess.check_call([str(pip), "install", "-e", str(install_root)])
    except subprocess.CalledProcessError as e:
        return [Result("pip", "fail", f"pip install exited {e.returncode}",
                       "Inspect the pip output above; common causes: network, missing build tools (gcc).")]
    results.append(Result("pip", "ok", "core dependencies installed"))

    py = venv_python(install_root)
    check_src = "import titan_hcl; import duckdb; import faiss; import torch; print('OK')"
    r = subprocess.run([str(py), "-c", check_src], capture_output=True, text=True)
    if r.returncode != 0:
        return results + [Result("imports", "fail",
                                  f"core imports failed: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}",
                                  "Re-run pip install; check that titan-hcl is in pyproject.toml.")]
    results.append(Result("imports", "ok", "titan_hcl + duckdb + faiss + torch importable"))

    return results


# ── Phase walker ───────────────────────────────────────────────────────────


@dataclass
class PhaseDef:
    id: str           # "phase_2", "phase_3", … (used as install_state key)
    title: str
    sub_phase: str    # which W1.<letter> owns this body
    runner: Callable  # takes (ctx) -> list[Result]


def run_phases(*, state: dict, mode: Mode, install_root: Path, default: bool,
               minimal: bool, skip_genesis: bool) -> int:
    """Walk Phases 2→7 (Phase 1 already ran in preflight). Returns exit code."""
    phases: list[tuple[PhaseDef, Callable[[], list[Result]]]] = [
        (PhaseDef("phase_2", "Mode + Maker wallet", "W1.b", None),
         lambda: run_mode_phase(state, mode, default=default)),
        (PhaseDef("phase_3", "Venv + Python deps", "W1.b", None),
         lambda: run_venv_phase(install_root)),
        (PhaseDef("phase_4", "Inference autodetect", "W1.c", None),
         lambda: run_inference_phase(default=default)),
        (PhaseDef("phase_5", "Comms (Telegram / X / Observatory)", "W1.d", None),
         lambda: run_comms_phase(default=default)),
        (PhaseDef("phase_6", "Genesis ceremony", "W1.b", None),
         lambda: ([Result("genesis", "warn", "--skip-genesis requested.")] if skip_genesis
                  else run_genesis_phase(install_root, mode, venv_python=venv_python(install_root)))),
        (PhaseDef("phase_7", "Systemd install + first start + health", "W1.e", None),
         lambda: [Result("systemd", "warn", "Phase 7 owned by W1.e (not yet implemented).",
                         "Systemd unit install + first-start + health gate land in the next sub-phase.")]),
    ]

    for phase, run in phases:
        if install_state.phase_done(state, phase.id):
            cprint(f"  ✓ {phase.title} (already complete — resume)", role="success")
            continue
        section(f"{phase.title}  [{phase.sub_phase}]")
        results = run()
        fails = render(results)
        if fails:
            install_state.mark_phase(state, phase.id, "failed", action=phase.title)
            cprint(f"\nPhase '{phase.title}' FAILED — fix the issues above and re-run.",
                   role="error", bold=True)
            return 1
        # Phases owned by un-shipped sub-phases (W1.d/W1.e) emit a single 'warn' Result —
        # mark them 'skipped' so they re-run when their owning sub-phase ships.
        if any(r.severity == "warn" and "not yet implemented" in r.detail for r in results):
            install_state.mark_phase(state, phase.id, "skipped", action=phase.title)
            cprint(f"  ⤳ {phase.title} skipped (owned by {phase.sub_phase}).",
                   role="warning")
            continue
        install_state.mark_phase(state, phase.id, "done", action=phase.title)
        cprint(f"  ✓ {phase.title} complete.", role="success")

    cprint("\nAll implemented phases complete. Pending: Phase 5 (W1.d), Phase 7 (W1.e).",
           role="text_strong", bold=True)
    return 0
