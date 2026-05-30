"""CLI router for `python -m scripts.setup_titan`.

Subcommands:
    install      Run the wizard.   --default / --mode={mainnet,devnet,local} / --resume / --minimal / --skip-genesis / --dry-run
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
def cmd_install(args: argparse.Namespace) -> int:
    banner()
    repo_root = Path(__file__).resolve().parents[2]
    mode = Mode(args.mode) if args.mode else None

    if mode is None and not args.default:
        cprint("No mode selected. The interactive wizard (Textual TUI) will land in the next W1 sub-phase.",
               role="warning")
        cprint("For now, pick a mode explicitly: --mode {mainnet,devnet,local} (or --default for the locked happy path).",
               role="text_muted")
        section("The three modes")
        for m in Mode:
            spec = spec_for(m)
            print(f"  {PULSE}{spec.label}{ANSI.RESET}")
            print(f"    {spec.one_liner}")
            print(f"    {METAL}SOL: {spec.needs_sol}{ANSI.RESET}")
        return 2

    if args.default and mode is None:
        # locked --default path will route to devnet by default per RFP rationale
        # (the realistic tester path) — confirmed in the wizard, not silently.
        cprint("--default selected. The wizard will confirm the mode interactively (defaulting to DEVNET).",
               role="text_strong")
        mode = Mode.DEVNET   # provisional; TUI will let the user override

    spec = spec_for(mode)
    section(f"Mode: {spec.label}")
    print(f"  {spec.one_liner}")
    if spec.notice:
        cprint(f"  {spec.notice}", role="warning" if "⚠" in spec.notice else "text_muted")

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

    # Walk Phases 2-7 — Phase 4 (W1.c) + Phase 6 (W1.b) are real; Phase 5/7 are
    # owned by W1.d/W1.e and emit a single 'warn' Result (skipped, not failed).
    state = install_state.load()
    state["setup_titan_version"] = __version__
    state["install_root"] = str(repo_root)
    install_state.save(state)
    return run_phases(state=state, mode=mode, install_root=repo_root,
                      default=args.default, minimal=args.minimal, skip_genesis=args.skip_genesis,
                      tag=args.tag, build_rust=args.build_rust)


# ── subcommands: stubs ─────────────────────────────────────────────────────
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
    pi.add_argument("--tag", default=None,
                    help="Release tag the binaries are fetched from (e.g. v0.0.1). "
                         "Forwarded by the bootstrap; needed unless --build-rust.")
    pi.add_argument("--build-rust", action="store_true",
                    help="Compile the 9 Rust daemons from titan-rust/ source instead of "
                         "downloading them (the fully-sovereign path; needs cargo + musl).")
    pi.add_argument("--resume", action="store_true",
                    help="Resume from the last completed phase (per ~/.titan/install_state.json).")
    pi.add_argument("--dry-run", action="store_true",
                    help="Preflight only; no installs, no genesis, no writes.")
    pi.set_defaults(func=cmd_install)

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
