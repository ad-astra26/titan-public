"""The three setup modes, locked 2026-05-22 in RFP_Titan_setup_release.md.

mainnet  — real GenesisNFT + own ZK Vault program deploy on Solana mainnet.
           Requires real SOL. Full sovereign.
devnet   — same as mainnet on devnet (airdropped test SOL). The realistic
           tester path.
local    — NO chain. Real identity + soul.md + birth-certificate + SSS
           ceremony are still performed, but the on-chain anchor is skipped
           (`genesis_ceremony.py --skip-on-chain`) and backups are OFF.
           Lets a user *see* the whole birth ceremony with zero deps.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    MAINNET = "mainnet"
    DEVNET  = "devnet"
    LOCAL   = "local"


@dataclass(frozen=True)
class ModeSpec:
    """What a given mode requires of the host + the genesis flow."""
    label: str
    one_liner: str
    needs_solana_cli: bool
    needs_anchor: bool
    needs_rust: bool
    needs_node: bool        # NodeSource 22 — Irys/Arweave (mainnet backups) + the realistic devnet mirror
    needs_rpc: bool
    needs_sol: str          # human description of SOL requirement
    backups_on: bool
    genesis_on_chain: bool
    notice: str             # the headline string the wizard shows for this mode


SPECS: dict[Mode, ModeSpec] = {
    Mode.MAINNET: ModeSpec(
        label="MAINNET — full sovereign Titan",
        one_liner="real GenesisNFT + your own ZK Vault program on Solana mainnet",
        needs_solana_cli=True, needs_anchor=True, needs_rust=True, needs_node=True,
        needs_rpc=True,
        needs_sol="~1–2 SOL recommended (deploy program + mint + PDA init)",
        backups_on=True, genesis_on_chain=True,
        notice="⚠ Mainnet genesis MINTS A REAL NFT AND SPENDS REAL SOL. Irreversible.",
    ),
    Mode.DEVNET: ModeSpec(
        label="DEVNET — realistic tester path",
        one_liner="same flow on Solana devnet, airdropped test SOL — free and disposable",
        needs_solana_cli=True, needs_anchor=True, needs_rust=True, needs_node=True,
        needs_rpc=True,
        needs_sol="airdropped test SOL only — no real cost",
        backups_on=True, genesis_on_chain=True,
        notice="Recommended for first-time testers. Free, reversible, exercises the full ceremony.",
    ),
    Mode.LOCAL: ModeSpec(
        label="LOCAL — simulated, zero deps",
        one_liner="real identity + soul + SSS ceremony, on-chain anchor SKIPPED",
        needs_solana_cli=False, needs_anchor=False, needs_rust=False, needs_node=False,
        needs_rpc=False,
        needs_sol="none",
        backups_on=False, genesis_on_chain=False,
        notice="ⓘ SIMULATED — Titan boots with a real identity but is not anchored on-chain. Backups disabled.",
    ),
}


def spec_for(mode: Mode | str) -> ModeSpec:
    m = Mode(mode) if isinstance(mode, str) else mode
    return SPECS[m]
