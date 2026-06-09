"""CLI router for `python -m scripts.setup_titan`.

Subcommands:
    install      Run the wizard.   --default / --mode={mainnet,devnet,local} / --resume / --minimal / --skip-genesis / --directives-file / --inference-provider / --inference-key / --simulate / --dry-run
    config       TUI to browse + edit config.toml + titan_params.toml  (stubbed in v0.1-alpha)
    diagnostic   User-friendly live health report  (stubbed in v0.1-alpha)
    repair       Idempotent re-run / fix detected problems  (stubbed in v0.1-alpha)
    uninstall    Clean removal  (stubbed in v0.1-alpha)

This v0.1-alpha ships the foundation: real preflight + mode constants + state
tracking. Each subcommand body will fill in over the W1 sub-phases per the RFP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import state as install_state
from . import toolchain
from .modes import Mode, spec_for
from .phases import run_phases
from .preflight import run_preflight, summarize
from .ui import ANSI, BRAND, HAZE, PULSE, GROWTH, METAL, DANGER, section, cprint


# ── ASCII banner (brand-coloured) ──────────────────────────────────────────
def banner() -> None:
    art = """\
    ████████╗██╗████████╗ █████╗ ███╗   ██╗
    ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
       ██║   ██║   ██║   ███████║██╔██╗ ██║
       ██║   ██║   ██║   ██╔══██║██║╚██╗██║
       ██║   ██║   ██║   ██║  ██║██║ ╚████║
       ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
"""
    print(HAZE + art + ANSI.RESET)
    cprint("       Sovereign AI Agent — Setup", role="text_strong", bold=True)
    cprint(f"       setup_titan {__version__}", role="text_muted")
    print()


# ── pretty-print a preflight report ────────────────────────────────────────
def _render_preflight(results) -> int:
    section("Preflight")
    sev_colors = {"ok": GROWTH, "warn": HAZE, "fail": DANGER}
    sev_glyph  = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    for r in results:
        col = sev_colors[r.severity]
        gly = sev_glyph[r.severity]
        print(f"  {col}{gly}{ANSI.RESET} {ANSI.BOLD}{r.name:<10}{ANSI.RESET} {r.detail}")
        if r.remediation and r.severity != "ok":
            print(f"      {METAL}→ {r.remediation}{ANSI.RESET}")
    ok, warn, fail = summarize(results)
    print()
    summary_role = "error" if fail else "warning" if warn else "success"
    cprint(f"  {ok} ok · {warn} warn · {fail} fail", role=summary_role, bold=True)
    return fail


# ── subcommand: install ────────────────────────────────────────────────────
def _cmd_resurrect(args: argparse.Namespace, repo_root: Path) -> int:
    """D1 — install in resurrection mode: recover a mainnet Titan from its on-chain
    sovereign backup chain (the walker swaps genesis for the resurrect phase)."""
    cprint("🜂 Sovereign Resurrection — recover a MAINNET Titan from its on-chain "
           "backup chain (your wallet + Shard-1 alone). Local/devnet keep a plaintext "
           "identity — just re-run install.", role="text_strong", bold=True)
    if args.verify_only:
        cprint("  --verify-only: RECOVERY observation mode (no on-chain writes / "
               "backups / X) — safe to run beside a living Titan.", role="warning")
    mode = Mode.MAINNET
    fails = _render_preflight(run_preflight(repo_root, mode))
    if fails:
        section("Result")
        cprint(f"Preflight FAILED ({fails} blocking issue(s) above). Fix and re-run.",
               role="error", bold=True)
        return 1
    if args.dry_run:
        section("Next")
        cprint("--dry-run: preflight only. Resurrection walker = venv → Rust binaries "
               "→ 🜂 resurrect → systemd → console (genesis/config/inference/comms are "
               "swapped — config comes from your restored backup).", role="text_muted")
        return 0
    state = install_state.load()
    state["setup_titan_version"] = __version__
    state["install_root"] = str(repo_root)
    if args.titan_id:
        state["titan_id"] = args.titan_id
    install_state.save(state)
    return run_phases(state=state, mode=mode, install_root=repo_root, default=args.default,
                      minimal=args.minimal, skip_genesis=False, tag=args.tag,
                      build_rust=args.build_rust, prompter=None,
                      resurrect=True, rpc_url=args.rpc_url, das_rpc_url=args.das_rpc_url,
                      verify_only=args.verify_only,
                      config_src=args.config, titan_pubkey=args.titan_pubkey,
                      toolchain_pins=toolchain.resolve_versions(args))


def cmd_install(args: argparse.Namespace) -> int:
    banner()
    repo_root = Path(__file__).resolve().parents[2]
    if getattr(args, "resurrect", False):
        return _cmd_resurrect(args, repo_root)
    mode = Mode(args.mode) if args.mode else None
    prompter = None          # None → run_phases uses StdinPrompter (the CLI flow)
    state_seed: dict = {}     # TUI pre-fills wallet/RPC so Phase 2 short-circuits

    # No mode + not --default → the guided path. Launch the branded Textual
    # wizard (W1.b.2) when we have a real terminal; otherwise (non-tty, or
    # --no-tui) fall back to the CLI "pick a mode" guidance.
    if mode is None and not args.default:
        use_tui = not args.no_tui and sys.stdin.isatty() and sys.stdout.isatty()
        if use_tui:
            # The Textual TUI is OPTIONAL — a fresh public box may not have textual
            # (it is not a base dep, and system `pip install` is PEP-668-locked on
            # modern Ubuntu). Degrade to the CLI mode-picker instead of crashing;
            # the --default / --mode paths never need the TUI.
            try:
                from .prompts import ScriptedPrompter
                from .tui import run_install_tui
            except ImportError as exc:
                cprint(f"  Textual TUI unavailable ({exc.name}) — falling back to the CLI. "
                       "Re-run with --default or --mode {mainnet,devnet,local}.", role="warning")
                use_tui = False
        if not use_tui:
            cprint("No mode selected. Re-run interactively for the guided wizard, or pick a mode:",
                   role="warning")
            cprint("  --mode {mainnet,devnet,local}  (or --default for the locked happy path)",
                   role="text_muted")
            section("The three modes")
            for m in Mode:
                spec = spec_for(m)
                print(f"  {PULSE}{spec.label}{ANSI.RESET}")
                print(f"    {spec.one_liner}")
                print(f"    {METAL}SOL: {spec.needs_sol}{ANSI.RESET}")
            return 2
        result = run_install_tui()   # imported above (in the use_tui try-block)
        if result is None:
            cprint("Setup cancelled — nothing was written.", role="warning")
            return 130
        mode, answers, state_seed = result
        prompter = ScriptedPrompter(answers)
        banner()  # the TUI took over the screen; re-print the banner for the install log

    if args.default and mode is None:
        # locked --default path routes to devnet (the realistic tester path).
        cprint("--default selected — using DEVNET (the realistic tester path).",
               role="text_strong")
        mode = Mode.DEVNET

    spec = spec_for(mode)
    section(f"Mode: {spec.label}")
    print(f"  {spec.one_liner}")
    if spec.notice:
        cprint(f"  {spec.notice}", role="warning" if "⚠" in spec.notice else "text_muted")
    if mode == Mode.MAINNET:
        cprint("  Mainnet genesis BURNS the plaintext key after a Shamir 2-of-3 split. "
               "Your only recovery from a lost box is `setup_titan restore` (resurrection "
               "from your offline Shard-1 + the on-chain shard). Record your Shard-1 and "
               "your Titan's PUBLIC address — that is ALL you need: the wallet + chain + "
               "Arweave supply everything else (NO off-site manifest, NO backup files). "
               "Resurrection is MAINNET-ONLY.",
               role="warning")

    if getattr(args, "simulate", False):
        if mode != Mode.MAINNET:
            section("Result")
            cprint("--simulate is a MAINNET-readiness rehearsal — re-run with --mode mainnet.",
                   role="error", bold=True)
            return 2
        cprint("  --simulate: a 0-SOL mainnet-readiness REHEARSAL — it provisions the box and "
               "walks the FULL birth ceremony, but mints nothing and spends nothing. Re-run "
               "without --simulate to perform the real birth.", role="text_strong")

    fails = _render_preflight(run_preflight(repo_root, mode))
    if fails:
        section("Result")
        cprint(f"Preflight FAILED ({fails} blocking issue(s) above). Fix and re-run.", role="error", bold=True)
        return 1

    if args.dry_run:
        section("Next")
        cprint("--dry-run: preflight only. Re-run without --dry-run to walk Phases 2-7.",
               role="text_muted")
        return 0

    # Walk Phases 2-7. The prompter (StdinPrompter for the CLI flow, or the
    # TUI's ScriptedPrompter) feeds every input; the phase bodies are identical.
    state = install_state.load()
    state.update(state_seed)
    state["setup_titan_version"] = __version__
    state["install_root"] = str(repo_root)
    install_state.save(state)
    return run_phases(state=state, mode=mode, install_root=repo_root,
                      default=args.default, minimal=args.minimal, skip_genesis=args.skip_genesis,
                      tag=args.tag, build_rust=args.build_rust, prompter=prompter,
                      toolchain_pins=toolchain.resolve_versions(args),
                      directives_file=args.directives_file,
                      inference_provider=args.inference_provider,
                      inference_key=args.inference_key,
                      simulate=getattr(args, "simulate", False))


# ── subcommands: stubs ─────────────────────────────────────────────────────
def cmd_restore(args: argparse.Namespace) -> int:
    banner()
    from .restore import run_restore
    repo_root = Path(args.install_root) if args.install_root else \
        Path(__file__).resolve().parents[2]
    return run_restore(
        repo_root, shard1=args.shard1, shard1_file=args.shard1_file,
        titan_pubkey=args.titan_pubkey, manifest=args.manifest,
        titan_id=args.titan_id, network=args.network,
        verify_zk=args.verify_zk, verify_only=args.verify_only, force=args.force,
        rpc_url=args.rpc_url, das_rpc_url=args.das_rpc_url)


def cmd_config(args: argparse.Namespace) -> int:
    banner()
    from .config import run_config
    repo_root = Path(__file__).resolve().parents[2]
    return run_config(repo_root, list_all=args.list, get=args.get, set_kv=args.set)


def cmd_diagnostic(args: argparse.Namespace) -> int:
    banner()
    from .manage import run_diagnostic
    return run_diagnostic()


def cmd_upgrade(args: argparse.Namespace) -> int:
    banner()
    from .manage import run_upgrade
    return run_upgrade(tag=args.tag, build_rust=args.build_rust)


def cmd_repair(args: argparse.Namespace) -> int:
    banner()
    from .manage import run_repair
    return run_repair()


def cmd_uninstall(args: argparse.Namespace) -> int:
    banner()
    from .manage import run_uninstall
    return run_uninstall(purge=args.purge, assume_yes=args.yes)


# ── argparse wiring ────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="setup_titan",
        description="Stand up a sovereign Titan on your own infra. See RFP_Titan_setup_release.md.",
    )
    p.add_argument("--version", action="version", version=f"setup_titan {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("install", help="Run the install wizard")
    pi.add_argument("--default", action="store_true",
                    help="Locked happy-path: only the 6 essential prompts.")
    pi.add_argument("--mode", choices=[m.value for m in Mode], default=None,
                    help="Setup mode (defaults to interactive selection).")
    pi.add_argument("--minimal", action="store_true",
                    help="Skip optional research stack (Crawl4AI, Unstructured, Playwright).")
    pi.add_argument("--skip-genesis", action="store_true",
                    help="Skip the Genesis Ceremony.")
    pi.add_argument("--directives-file", default=None,
                    help="Path to a file with the Maker's Prime Directives, for a "
                         "headless / non-tty / --default install (no $EDITOR). "
                         "Env fallbacks: TITAN_DIRECTIVES_FILE (path) and "
                         "TITAN_DIRECTIVES (literal text). The directives are "
                         "MANDATORY for an on-chain (devnet/mainnet) birth.")
    pi.add_argument("--inference-provider", default=None,
                    choices=["ollama_local", "ollama_cloud", "openrouter"],
                    help="Pick the inference provider non-interactively (headless / "
                         "non-tty install). ollama_local needs no key; ollama_cloud / "
                         "openrouter need --inference-key. Env fallback: "
                         "TITAN_INFERENCE_PROVIDER.")
    pi.add_argument("--inference-key", default=None,
                    help="API key for the hosted --inference-provider (ollama_cloud / "
                         "openrouter). Env fallback: TITAN_INFERENCE_KEY.")
    pi.add_argument("--simulate", action="store_true",
                    help="MAINNET-readiness rehearsal: provision the box and walk the FULL "
                         "birth ceremony, but mint nothing and spend 0 SOL (every on-chain "
                         "submit is stubbed). Requires --mode mainnet. Stops before boot.")
    pi.add_argument("--tag", default=None,
                    help="Release tag the binaries are fetched from (e.g. v0.0.1). "
                         "Forwarded by the bootstrap; needed unless --build-rust.")
    pi.add_argument("--build-rust", action="store_true",
                    help="Compile the 9 Rust daemons from titan-rust/ source instead of "
                         "downloading them (the fully-sovereign path; needs cargo + musl).")
    pi.add_argument("--resume", action="store_true",
                    help="Resume from the last completed phase (per ~/.titan/install_state.json).")
    pi.add_argument("--no-tui", action="store_true",
                    help="Skip the Textual wizard; use the plain CLI prompts "
                         "(implied automatically when stdin/stdout is not a terminal).")
    pi.add_argument("--dry-run", action="store_true",
                    help="Preflight only; no installs, no genesis, no writes.")
    # D1 — resurrection install-mode (mainnet): swaps genesis for the sovereign
    # on-chain recovery. config + settings come from the restored backup.
    pi.add_argument("--resurrect", action="store_true",
                    help="Recover a MAINNET Titan from its on-chain sovereign backup "
                         "chain instead of birthing a new one (needs your Shard-1).")
    pi.add_argument("--verify-only", action="store_true",
                    help="With --resurrect: boot in RECOVERY observation mode (no "
                         "on-chain writes / backups / X) — the live restore-test guard.")
    pi.add_argument("--rpc-url", default=None,
                    help="With --resurrect: Solana RPC for the chain walk "
                         "(default: public mainnet-beta). Prompted if omitted.")
    pi.add_argument("--das-rpc-url", default=None,
                    help="With --resurrect: DAS-capable RPC (Helius/Triton) for "
                         "GenesisNFT identity discovery. Defaults to --rpc-url when "
                         "one endpoint serves both.")
    pi.add_argument("--config", default=None,
                    help="With --resurrect: path to your own config.toml to stage "
                         "(for operators who opted config.toml OUT of their backup).")
    pi.add_argument("--titan-id", default=None,
                    help="With --resurrect: Titan id (default: resolved from your shard).")
    pi.add_argument("--titan-pubkey", default=None,
                    help="With --resurrect: your Titan's PUBLIC wallet address "
                         "(printed alongside Shard-1; not a secret). NO envelope/"
                         "manifest needed — the wallet discovers everything.")
    # Toolchain pin overrides (auto-provisioner — default to the T1-verified PINS
    # in toolchain.py; pass to freeze a specific version). See rFP_setup_titan_auto_provisioner.md §6.
    pi.add_argument("--rust-version", default=None,
                    help="Override the Rust toolchain pin (default: stable channel + musl target).")
    pi.add_argument("--solana-version", default=None,
                    help="Override the Solana CLI pin (default: Agave 3.1.10).")
    pi.add_argument("--anchor-version", default=None,
                    help="Override the Anchor CLI pin (default: 0.32.1, via avm).")
    pi.add_argument("--node-version", default=None,
                    help="Override the Node.js major pin (default: 22, via NodeSource).")
    pi.set_defaults(func=cmd_install)

    pr = sub.add_parser("restore",
                        help="Resurrect a mainnet-born Titan from your Maker shard + chain")
    pr.add_argument("--shard1", default=None,
                    help="Maker raw Shard-1 (hex). Omit to be prompted with no echo.")
    pr.add_argument("--shard1-file", default=None,
                    help="Path to a file holding the raw Maker Shard-1 hex.")
    pr.add_argument("--titan-pubkey", default=None,
                    help="Your Titan's PUBLIC wallet address (printed alongside "
                         "Shard-1; not a secret). Required on a fresh box — the "
                         "wallet discovers Shard-3 + the v=3 chain. NO manifest needed.")
    pr.add_argument("--manifest", default=None,
                    help="LEGACY/DEBUG ONLY: an off-site UnifiedManifest JSON. Omit "
                         "it — the sovereign v=3 chain restore needs no manifest.")
    pr.add_argument("--titan-id", default=None, help="Titan id (default: from record, else T1).")
    pr.add_argument("--rpc-url", default=None,
                    help="Mainnet RPC for the chain walk (default: config/public "
                         "mainnet-beta). Prompted if omitted.")
    pr.add_argument("--das-rpc-url", default=None,
                    help="DAS-capable RPC (Helius/Triton) for GenesisNFT identity "
                         "discovery. Defaults to --rpc-url when one endpoint serves both.")
    pr.add_argument("--install-root", default=None, help="Target install tree (default: this repo).")
    pr.add_argument("--network", choices=["mainnet", "devnet"], default="mainnet",
                    help="Arweave/Solana network (default: mainnet).")
    pr.add_argument("--verify-zk", action="store_true",
                    help="Also round-trip-verify each event against the on-chain ZK memo.")
    pr.add_argument("--verify-only", action="store_true",
                    help="Boot the resurrected Titan in observation mode (live restore test).")
    pr.add_argument("--force", action="store_true",
                    help="Permit in-place restore over a populated data/ tree.")
    pr.set_defaults(func=cmd_restore)

    pc = sub.add_parser("config", help="Browse/edit config.toml + DNA params (comment-driven)")
    pc.add_argument("--list", action="store_true", help="Dump every section.key = value + help.")
    pc.add_argument("--get", default=None, metavar="SEC.KEY", help="Print one key's value + help.")
    pc.add_argument("--set", default=None, metavar="SEC.KEY=VAL", help="Edit one key non-interactively.")
    pc.set_defaults(func=cmd_config)

    sub.add_parser("diagnostic", help="User-friendly live health report").set_defaults(func=cmd_diagnostic)

    pu = sub.add_parser("upgrade", help="In-place upgrade: git pull @ tag + refresh binaries + restart")
    pu.add_argument("--tag", default=None, help="Release tag to upgrade to (e.g. v0.0.2).")
    pu.add_argument("--build-rust", action="store_true",
                    help="Compile the Rust daemons from source instead of downloading.")
    pu.set_defaults(func=cmd_upgrade)

    sub.add_parser("repair",     help="Idempotent heal: regenerate unit + restart").set_defaults(func=cmd_repair)

    px = sub.add_parser("uninstall", help="Clean removal (keeps data/ unless --purge)")
    px.add_argument("--purge", action="store_true",
                    help="ALSO delete data/ + ~/.titan/ (identity loss — irreversible without your shard).")
    px.add_argument("--yes", action="store_true", help="Skip the purge confirmation prompt.")
    px.set_defaults(func=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
