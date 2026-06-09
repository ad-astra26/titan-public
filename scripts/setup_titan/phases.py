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

import getpass
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import state as install_state
from . import toolchain
from .backup_config import run_backup_config_phase
from .binaries import run_binaries_phase
from .comms import run_comms_phase
from .console import run_console_phase
from .config_seed import run_config_seed_phase
from .inference import run_inference_phase
from .genesis_inputs import run_genesis_inputs_phase
from .genesis_runner import run_genesis_phase
from .provision import run_provision_phase
from .resurrect import run_resurrect_phase
from .services import run_services_phase
from .systemd_runner import run_systemd_phase
from .modes import Mode, spec_for
from .preflight import Result, summarize
from .prompts import Prompter, StdinPrompter
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


def run_mode_phase(state: dict, mode: Mode, *, default: bool,
                   prompter: Prompter) -> list[Result]:
    """Confirm mode + capture Maker wallet + Solana RPC (modes 1/2 only).

    The mode itself is chosen earlier (CLI flag, `--default` provisional, or
    the wizard's earlier mode-chooser screen). This phase persists it and
    captures the per-mode credentials. Already-collected creds (CLI flags, a
    prior resume, or the TUI seeding ``state``) short-circuit the prompts.
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
            wallet = prompter.until(
                "maker_wallet", "Maker wallet (Solana pubkey, base58)",
                validate=looks_like_solana_pubkey,
                hint="that doesn't look like a Solana pubkey (32–44 base58 chars)")
            state["maker_wallet"] = wallet
        results.append(Result("wallet", "ok", wallet))

        rpc = state.setdefault("solana_rpc", "")
        if not rpc:
            default_rpc = ("https://api.mainnet-beta.solana.com" if mode == Mode.MAINNET
                           else "https://api.devnet.solana.com")
            rpc = (default_rpc if default
                   else prompter.line("solana_rpc", "Solana RPC URL", default=default_rpc))
            if not rpc.startswith(("http://", "https://")):
                results.append(Result("rpc", "fail", f"invalid URL: {rpc!r}",
                                      "Must start with http:// or https://"))
                return results
            state["solana_rpc"] = rpc
        results.append(Result("rpc", "ok", rpc))

    install_state.save(state)
    return results


# ── Phase 3 — Venv + Python deps ───────────────────────────────────────────


# ── Embedding model (GGUF) seed ──────────────────────────────────────────────
# Sovereign, offline-first text embedder (§3J.1): the f16 GGUF is vendored into
# the per-install model cache so the fleet NEVER does a runtime HF pull. Pinned by
# sha256 so a fresh install gets byte-identical vectors to the live fleet.
GGUF_REPO = "CompendiumLabs/bge-small-en-v1.5-gguf"
GGUF_FILENAME = "bge-small-en-v1.5-f16.gguf"
GGUF_SHA256 = "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999"
GGUF_URL = f"https://huggingface.co/{GGUF_REPO}/resolve/main/{GGUF_FILENAME}"

# CPU-only torch index (INV-PROV-9): on a vCPU server, PEP 440 ranks the `+cpu`
# local-version wheel ABOVE the PyPI default (CUDA) build, so torch resolves to
# the ~200MB CPU wheel instead of the ~5GB CUDA stack. Used for BOTH the base
# `-e .` install and the `[research]` extra (unstructured-inference can pull torch).
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_embedding_gguf(install_root: Path, py: Path) -> list[Result]:
    """Vendor the bge-small f16 GGUF into <install_root>/data/.gguf_cache and
    confirm the llama.cpp embedder embeds. Idempotent: skips the download when a
    sha-matching GGUF is already present (existing boxes seeded at P1)."""
    import urllib.request

    results: list[Result] = []
    cache_dir = install_root / "data" / ".gguf_cache"
    target = cache_dir / GGUF_FILENAME

    if target.exists() and _sha256(target) == GGUF_SHA256:
        results.append(Result("embed_gguf", "ok",
                              f"GGUF already vendored ({GGUF_FILENAME}, sha verified)"))
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cprint(f"  Vendoring embedding model {GGUF_FILENAME} (~65 MB) — offline-first…",
               role="text_strong")
        tmp = target.with_suffix(".gguf.partial")
        try:
            urllib.request.urlretrieve(GGUF_URL, tmp)  # noqa: S310 (pinned HF URL)
        except Exception as e:  # noqa: BLE001
            return [Result("embed_gguf", "fail", f"GGUF download failed: {e}",
                           f"Manually place {GGUF_FILENAME} into {cache_dir} "
                           f"(source: {GGUF_URL}).")]
        got = _sha256(tmp)
        if got != GGUF_SHA256:
            tmp.unlink(missing_ok=True)
            return [Result("embed_gguf", "fail",
                           f"GGUF sha256 mismatch (got {got[:12]}…, want {GGUF_SHA256[:12]}…)",
                           "Refusing to install an unverified embedding model.")]
        tmp.rename(target)
        results.append(Result("embed_gguf", "ok",
                              f"vendored {GGUF_FILENAME} (sha verified)"))

    # Strongest check: the §3J.1 boot self-test (constructs the singleton, embeds
    # a probe, asserts non-zero + semantically ordered) — fails LOUD, never silent.
    check = "from titan_hcl.utils.text_embedder import self_test; raise SystemExit(0 if self_test() else 1)"
    r = subprocess.run([str(py), "-c", check], capture_output=True, text=True,
                       cwd=str(install_root))
    if r.returncode != 0:
        results.append(Result("embed_selftest", "fail",
                              "llama.cpp embedder self-test failed",
                              (r.stderr.strip().splitlines()[-1] if r.stderr else "see log")))
    else:
        results.append(Result("embed_selftest", "ok",
                              "llama.cpp embedder self-test passed (non-zero, ordered)"))
    return results


def _pip_install_research(pip: Path, install_root: Path, *, minimal: bool) -> list[Result]:
    """Install the `[research]` extra (Crawl4AI · unstructured · OCR libs) into the
    venv with the CPU torch index (INV-PROV-9 — never CUDA). Skipped under
    `--minimal`. `knowledge_worker` + `SageRecorder` depend on this (INV-PROV-8)."""
    if minimal:
        return [Result("research", "warn",
                       "--minimal: research pipeline ([research] extra) skipped.",
                       "Re-run without --minimal for Crawl4AI + unstructured (knowledge/Sage).")]
    cprint("  Installing the research pipeline ([research] — Crawl4AI · unstructured · OCR libs), "
           "CPU wheels only…", role="text_strong")
    try:
        subprocess.check_call([str(pip), "install", "--extra-index-url", TORCH_CPU_INDEX,
                               "-e", f"{install_root}[research]"])
    except subprocess.CalledProcessError as e:
        return [Result("research", "fail", f"[research] install exited {e.returncode}",
                       "Inspect the pip output above; re-run (or use --minimal to skip the research stack).")]
    return [Result("research", "ok",
                   "research pipeline installed (Crawl4AI · unstructured · html2text · CPU-only)")]


def run_venv_phase(install_root: Path, *, minimal: bool = False) -> list[Result]:
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
    # --extra-index-url whl/cpu: on a CPU-only server, PEP 440 ranks the
    # `+cpu` local-version wheel ABOVE the PyPI default (CUDA) build, so torch
    # resolves to the ~200MB CPU wheel instead of dragging in the ~5GB CUDA
    # stack — which would blow a 4GB / small-disk min-tier box (RFP P3).
    # The CPU index is additive: every non-torch package still comes from PyPI.
    try:
        subprocess.check_call([str(pip), "install", "--extra-index-url", TORCH_CPU_INDEX,
                               "-e", str(install_root)])
    except subprocess.CalledProcessError as e:
        return [Result("pip", "fail", f"pip install exited {e.returncode}",
                       "Inspect the pip output above; common causes: network, missing build tools (gcc).")]
    results.append(Result("pip", "ok", "core dependencies installed (CPU torch wheel)"))

    py = venv_python(install_root)
    check_src = "import titan_hcl; import duckdb; import faiss; import torch; import llama_cpp; print('OK')"
    r = subprocess.run([str(py), "-c", check_src], capture_output=True, text=True)
    if r.returncode != 0:
        return results + [Result("imports", "fail",
                                  f"core imports failed: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}",
                                  "Re-run pip install; check that titan-hcl is in pyproject.toml.")]
    results.append(Result("imports", "ok", "titan_hcl + duckdb + faiss + torch + llama_cpp importable"))

    # Research pipeline ([research] extra — Crawl4AI/unstructured/html2text), CPU-only,
    # gated by --minimal. knowledge_worker + SageRecorder depend on it (INV-PROV-8/9).
    results.extend(_pip_install_research(pip, install_root, minimal=minimal))

    # Seed the sovereign embedding GGUF + run the §3J.1 self-test (a halting
    # 'fail' Result if the embedder can't produce non-zero vectors).
    results.extend(_seed_embedding_gguf(install_root, py))

    return results


# ── Phase walker ───────────────────────────────────────────────────────────


@dataclass
class PhaseDef:
    id: str           # "phase_2", "phase_3", … (used as install_state key)
    title: str
    sub_phase: str    # which W1.<letter> owns this body
    runner: Callable  # takes (ctx) -> list[Result]


def run_phases(*, state: dict, mode: Mode, install_root: Path, default: bool,
               minimal: bool, skip_genesis: bool, tag: str | None = None,
               build_rust: bool = False, prompter: Prompter | None = None,
               resurrect: bool = False, rpc_url: str | None = None,
               das_rpc_url: str | None = None,
               verify_only: bool = False, config_src: str | None = None,
               titan_pubkey: str | None = None,
               toolchain_pins: dict[str, str] | None = None,
               directives_file: str | None = None,
               inference_provider: str | None = None,
               inference_key: str | None = None,
               simulate: bool = False) -> int:
    """Walk the install phases (Phase 1 already ran in preflight). Returns exit code.

    ``prompter`` injects the input source: the default :class:`StdinPrompter`
    is the CLI flow; the Textual TUI passes a :class:`ScriptedPrompter` seeded
    with answers it collected in its branded question screens. The phase bodies
    are identical either way.

    When ``resurrect`` is set (D1, mainnet only) the walker builds the env
    (venv → Rust binaries) then SWAPS genesis/config/inference/comms for a single
    :func:`run_resurrect_phase` (config + settings come from the restored backup,
    not re-prompts), then systemd + console.
    """
    prompter = prompter or StdinPrompter()
    pins = toolchain_pins or toolchain.resolve_versions(None)

    if resurrect:
        phases: list[tuple[PhaseDef, Callable[[], list[Result]]]] = [
            (PhaseDef("phase_provision", "Toolchain provisioning (Rust · Solana · Node)", "RFP-provisioner", None),
             lambda: run_provision_phase(install_root, mode, pins, resurrect=True,
                                         prompter=prompter, default=default)),
            (PhaseDef("phase_3", "Venv + Python deps", "W1.b", None),
             lambda: run_venv_phase(install_root, minimal=minimal)),
            (PhaseDef("phase_services", "Research services (SearXNG + OCR)", "RFP-provisioner", None),
             lambda: run_services_phase(install_root, minimal=minimal)),
            (PhaseDef("phase_bin", "Rust daemon binaries", "W1.b", None),
             lambda: run_binaries_phase(install_root, tag=tag or "main", build_rust=build_rust)),
            (PhaseDef("phase_resurrect", "🜂 Sovereign Resurrection", "W1.5/D1", None),
             lambda: run_resurrect_phase(install_root, venv_python=venv_python(install_root),
                                         titan_id=state.get("titan_id"), rpc_url=rpc_url,
                                         das_rpc_url=das_rpc_url,
                                         verify_only=verify_only, config_src=config_src,
                                         titan_pubkey=titan_pubkey)),
            (PhaseDef("phase_7", "Systemd install + first start + health", "W1.e", None),
             lambda: run_systemd_phase(state, install_root, mode, default=default, prompter=prompter)),
            (PhaseDef("phase_console", "TC² Console Agent (owner UI)", "W8", None),
             lambda: run_console_phase(state, install_root, user=getpass.getuser())),
        ]
        return _walk(phases, state)

    phases: list[tuple[PhaseDef, Callable[[], list[Result]]]] = [
        (PhaseDef("phase_2", "Mode + Maker wallet", "W1.b", None),
         lambda: run_mode_phase(state, mode, default=default, prompter=prompter)),
        # Provision the toolchain right after the quick mode/wallet capture and
        # BEFORE venv/binaries/genesis (INV-PROV-5) — so a bad wallet fails fast
        # instead of after a multi-minute toolchain install.
        (PhaseDef("phase_provision", "Toolchain provisioning (Rust · Solana · Anchor · Node)", "RFP-provisioner", None),
         lambda: run_provision_phase(install_root, mode, pins, resurrect=False,
                                     prompter=prompter, default=default)),
        (PhaseDef("phase_3", "Venv + Python deps", "W1.b", None),
         lambda: run_venv_phase(install_root, minimal=minimal)),
        (PhaseDef("phase_services", "Research services (SearXNG + OCR)", "RFP-provisioner", None),
         lambda: run_services_phase(install_root, minimal=minimal)),
        (PhaseDef("phase_bin", "Rust daemon binaries", "W1.b", None),
         lambda: run_binaries_phase(install_root, tag=tag or "main", build_rust=build_rust)),
        (PhaseDef("phase_cfg", "Seed config.toml + chat auth key + per-mode network", "W1.f", None),
         lambda: run_config_seed_phase(install_root, mode, state)),
        (PhaseDef("phase_backup", "Sovereign backup posture (encryption + config-in-backup)", "W1.5/§24.4.B", None),
         lambda: run_backup_config_phase(install_root, mode, prompter=prompter, default=default)),
        (PhaseDef("phase_4", "Inference autodetect", "W1.c", None),
         lambda: run_inference_phase(default=default, install_root=install_root, prompter=prompter,
                                     provider=inference_provider, key=inference_key)),
        (PhaseDef("phase_5", "Comms (Telegram / X)", "W1.d", None),
         lambda: run_comms_phase(default=default, state=state, prompter=prompter)),
        (PhaseDef("phase_identity", "Genesis identity (name · Maker · directives)", "W1.f", None),
         lambda: run_genesis_inputs_phase(install_root, mode, state,
                                          prompter=prompter, default=default,
                                          directives_file=directives_file)),
        (PhaseDef("phase_6", "Genesis ceremony", "W1.b", None),
         lambda: ([Result("genesis", "warn", "--skip-genesis requested.")] if skip_genesis
                  else run_genesis_phase(install_root, mode, venv_python=venv_python(install_root),
                                         simulate=simulate))),
        (PhaseDef("phase_7", "Systemd install + first start + health", "W1.e", None),
         lambda: run_systemd_phase(state, install_root, mode, default=default, prompter=prompter)),
        (PhaseDef("phase_console", "TC² Console Agent (owner UI — the sole shipped front-end)", "W8", None),
         lambda: run_console_phase(state, install_root, user=getpass.getuser())),
    ]
    # --simulate is a MAINNET-readiness rehearsal: the genesis phase runs the
    # ceremony's --simulate (0 SOL, nothing minted), so there is no bootable
    # identity to install/boot. Drop systemd + console and end with a readiness
    # summary instead of standing up a (non-existent) live Titan.
    if simulate:
        phases = [p for p in phases if p[0].id not in ("phase_7", "phase_console")]
        phases.append(
            (PhaseDef("phase_simready", "Mainnet-readiness summary", "RFP §8/G7", None),
             lambda: [Result("readiness", "ok",
                             "Mainnet-readiness SIMULATION complete — toolchain, venv, binaries, "
                             "config + the full genesis ceremony all exercised; 0 SOL spent, "
                             "nothing minted. Re-run without --simulate to perform the real birth.")]))
    return _walk(phases, state)


def _walk(phases: list[tuple[PhaseDef, Callable[[], list[Result]]]], state: dict) -> int:
    """Render + persist each phase in order; halt on the first 'fail'. Returns exit code."""
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

    cprint("\nAll phases complete. Titan is installed, enabled, and (if genesis ran) running.",
           role="success", bold=True)
    return 0
