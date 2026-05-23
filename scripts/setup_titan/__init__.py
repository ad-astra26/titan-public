"""Titan setup CLI — install, configure, diagnose, repair, uninstall.

This package is the user-facing entry point for standing up a Titan on a
fresh Linux box. It is the implementation of Workstream 1 of
`titan-docs/RFP_Titan_setup_release.md`.

Subcommands (see `python -m scripts.setup_titan --help`):
    install        Guided install wizard (TUI), with --default / --mode={mainnet,devnet,local}
    config         Browse/edit config.toml (49 sections) + titan_params.toml (DNA)
    diagnostic     Tester-friendly health/diagnostic report
    repair         Idempotent re-run / fix detected problems
    uninstall      Clean removal

Entry point: `python -m scripts.setup_titan <subcommand> [args]`
(The thin bootstrap setup_titan.sh / install.sh invokes this for end users.)
"""

__version__ = "0.1.0-alpha"
