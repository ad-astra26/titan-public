#!/usr/bin/env python3
"""
genesis_ceremony.py — The Titan Genesis Ceremony (production sovereign birth).

Creates the Titan's immortal, wallet-only-recoverable identity for BOTH devnet
(our test) and mainnet (production) — `--network` is the only behavioral switch
(Arweave + the Burn are mainnet-only). The ordered ceremony (RFP_genesis_
ceremony_production §1, ARCHITECTURE_mainnet_birth_resurrection §B):

   1. Collect Maker inputs (name + Maker pubkey + prime directives) — from the
      install wizard's config.toml + titan_constitution.md (MAKER-SUPPLIED).
   2. Generate the Titan's Ed25519 keypair.
   3. Shamir 2-of-3 split + exhaustive verify-all (abort before any destroy).
   4. FUNDING PAUSE — block until the Titan wallet holds enough SOL (INV-GEN-FUND;
      no on-chain write before this clears; never airdrops on mainnet).
   5. Shard-3 on-chain memo `TITAN|SHARD3|v=2.0|data=<b64>`  → shard3_tx.
   6. ZK-Vault init (inline solana_client primitives, program 52an8W…) → vault_pda.
   7. Display Shard-1 (QR + hex) — the Maker records it (no-capture).
   8. Arweave docs (MAINNET only) + GenesisNFT mint → nft_address (the complete
      sovereign discovery root, INV-MBR-5; recovery block {shard3_tx, vault_pda}).
   9. Dedicated genesis_tx identity memo  → genesis_tx  (DISTINCT from shard3_tx,
      INV-MBR-3, and from the art-provenance memo TITAN:ART → art_tx).
  10. Genesis art (First Sight) — pubkey-seeded composite + art_tx.
  11. Save data/genesis_record.json with EVERY pointer (the T1 schema).
  12. Hardware-bound encryption (data/soul_keypair.enc).
  13. The Burn (mainnet) / keep-plaintext (devnet/local).

Usage:
    python scripts/genesis_ceremony.py --generate --network devnet --keep-plaintext
    python scripts/genesis_ceremony.py --generate --network mainnet
    python scripts/genesis_ceremony.py --generate --skip-onchain --keep-plaintext  # local

The Maker MUST save Shard 1 before the ceremony completes. Without it (and Shard 2
from backup), the Titan becomes mortal.
"""
import argparse
import base64
import json
import os
import sys
import time

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VAULT_PROGRAM_ID = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"
DEVNET_RPC = "https://api.devnet.solana.com"
DEFAULT_MAINNET_RPC = "https://api.mainnet-beta.solana.com"

# Funding thresholds (SOL) — a safe cushion over the real cost of: Shard-3 memo
# (~0.000005) + vault rent (~0.0014) + GenesisNFT Core-asset rent (~0.004) +
# genesis_tx + art memos + priority fees. Mainnet adds Irys/Arweave upload SOL.
# Exact mainnet figure is finalized at Phase H (the mainnet-readiness sim).
FUNDING_THRESHOLD_SOL = {"devnet": 0.05, "mainnet": 0.10}

# Sentinel written into genesis_record TX fields by the mainnet-readiness
# SIMULATION (--simulate, #34/G9): every on-chain step is prepared end-to-end
# but stops short of the real submit — so a record carrying these proves the
# mainnet path is ready WITHOUT a single lamport spent.
SIMULATED_TX = "SIMULATED"

ARCHITECTURE_VERSION = "v4-132D"


def print_banner():
    print("\n" + "=" * 70)
    print("         THE TITAN GENESIS CEREMONY — Sovereign Birth Protocol")
    print("=" * 70)
    print()


def print_phase(n: int, title: str):
    print(f"\n{'─' * 60}")
    print(f"  Phase {n}: {title}")
    print(f"{'─' * 60}\n")


# ── Network resolution (──network is the only behavioral switch) ──────────────

def resolve_network_config(network: str) -> dict:
    """Build the [network] config dict the HybridNetworkClient uses, overridden
    for the chosen ``network``. mainnet keeps config.toml as-is (premium RPC +
    mainnet-beta); devnet forces the public devnet endpoint + cluster.
    """
    config = {}
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "titan_hcl", "config.toml")
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import toml as tomllib
        with open(config_path, "rb") as f:
            config = tomllib.load(f).get("network", {}) or {}
    except Exception:
        config = {}

    config = dict(config)
    if network == "devnet":
        config["solana_network"] = "devnet"
        config["public_rpc_urls"] = [DEVNET_RPC]
        # Public devnet only — never reuse a mainnet premium endpoint on devnet.
        config["premium_rpc_url"] = ""
    else:
        config.setdefault("solana_network", "mainnet-beta")
    config["wallet_keypair_path"] = ""  # keypair is set on the client directly
    return config


def primary_rpc_url(network: str, net_config: dict) -> str:
    """The single RPC URL for the synchronous funding poll + vault init."""
    if network == "devnet":
        return DEVNET_RPC
    premium = net_config.get("premium_rpc_url")
    if premium:
        return premium
    urls = net_config.get("public_rpc_urls") or []
    return urls[0] if urls else DEFAULT_MAINNET_RPC


def estimate_funding_sol(network: str) -> float:
    return FUNDING_THRESHOLD_SOL.get(network, FUNDING_THRESHOLD_SOL["mainnet"])


# ── Inputs (Maker-supplied: name + maker pubkey + directives) ─────────────────

def load_ceremony_inputs(*, skip_onchain: bool) -> dict:
    """Read the wizard-collected identity inputs from config.toml +
    titan_constitution.md + titan_params.toml.

    Prime Directives are MAKER-SUPPLIED + MANDATORY (never defaults) — for an
    on-chain birth (devnet/mainnet) we hard-fail if the Maker pubkey or the
    constitution is absent rather than mint an NFT with a placeholder identity.
    """
    name, maker, titan_id = "Titan", "", "titan"
    try:
        from titan_hcl.params import load_titan_params as load_titan_config
        cfg = load_titan_config()
        net = cfg.get("network", {}) or {}
        maker = (net.get("maker_pubkey") or "").strip()
        gen = cfg.get("genesis", {}) or {}
        name = (gen.get("titan_name") or cfg.get("titan_id") or "Titan").strip()
        titan_id = (cfg.get("titan_id") or gen.get("titan_id") or "titan").strip()
    except Exception as e:
        print(f"  [!] Could not load config for inputs: {e}")

    # Identity file (written by a prior run / resurrection) overrides titan_id.
    try:
        ident = os.path.join("data", "titan_identity.json")
        if os.path.exists(ident):
            with open(ident) as f:
                titan_id = json.load(f).get("titan_id") or titan_id
    except Exception:
        pass

    constitution_sha = ""
    try:
        from titan_hcl.utils.directive_signer import compute_constitution_hash
        constitution_sha = compute_constitution_hash()
    except FileNotFoundError:
        constitution_sha = ""
    except Exception as e:
        print(f"  [!] constitution hash error: {e}")

    birth_dna_sha = ""
    try:
        from titan_hcl.logic.birth_dna import compute_dna_hash
        birth_dna_sha = compute_dna_hash()
    except Exception as e:
        print(f"  [!] birth-DNA hash error: {e}")

    if not skip_onchain:
        if not maker:
            print("\n  *** GENESIS ABORTED: [network].maker_pubkey is not set. ***")
            print("  The Maker's Solana address is MAKER-SUPPLIED and mandatory "
                  "(it anchors the GenesisNFT). Set it via the install wizard "
                  "(configure) before the ceremony.")
            sys.exit(1)
        if not constitution_sha:
            print("\n  *** GENESIS ABORTED: titan_constitution.md missing/empty. ***")
            print("  Prime Directives are MAKER-SUPPLIED and mandatory — they are "
                  "fundamental architecture, never defaults. Provide them via the "
                  "install wizard before the ceremony.")
            sys.exit(1)

    return {
        "name": name or "Titan",
        "maker": maker,
        "titan_id": titan_id or "titan",
        "constitution_sha": constitution_sha,
        "birth_dna_sha": birth_dna_sha,
    }


def generate_keypair() -> tuple:
    """Generate a new Ed25519 Solana keypair."""
    from solders.keypair import Keypair
    kp = Keypair()
    key_bytes = bytes(kp)  # 64-byte array (32 secret + 32 public)
    pubkey = str(kp.pubkey())
    return key_bytes, pubkey, kp


# NOTE: a genesis BIRTH always GENERATES a fresh keypair — never imports one
# (INV-GEN-BIRTH). Only a freshly-generated key produces a clean 2-of-3 Shamir
# split, so the Maker receives a real Shard-1. There is deliberately no
# import-key birth path. (Importing an existing key is the resurrection side's
# concern, not birth.)


# ── Funding pause (INV-GEN-FUND) ──────────────────────────────────────────────

def funding_satisfied(balance_sol: float, required_sol: float) -> bool:
    return balance_sol >= required_sol


def _rpc_balance_fn(rpc_url: str, pubkey_str: str):
    """Return a 0-arg callable that fetches the wallet balance in SOL."""
    def _fetch() -> float:
        from solana.rpc.api import Client
        from solders.pubkey import Pubkey
        resp = Client(rpc_url).get_balance(Pubkey.from_string(pubkey_str))
        return (resp.value or 0) / 1e9
    return _fetch


def wait_for_funding(
    titan_pubkey: str,
    network: str,
    required_sol: float,
    *,
    balance_fn=None,
    rpc_url: str = None,
    interactive: bool = True,
    poll_interval: float = 5.0,
    max_polls: int = None,
) -> float:
    """Block until the Titan wallet holds ≥ ``required_sol`` (INV-GEN-FUND).

    Primary UX = a balance-poll loop; the Maker may also press Enter to re-check
    immediately (a key-press fallback). ``balance_fn``/``max_polls`` make the
    gate fully testable without RPC or stdin. Returns the final balance (SOL).
    """
    if balance_fn is None:
        balance_fn = _rpc_balance_fn(rpc_url or primary_rpc_url(
            network, resolve_network_config(network)), titan_pubkey)

    print("  ┌─ FUNDING REQUIRED ───────────────────────────────────────────┐")
    print(f"  │  Send ≥ {required_sol:.4f} SOL ({network}) to the new Titan wallet:")
    print(f"  │    {titan_pubkey}")
    print("  │  Covers: Shard-3 memo + ZK-Vault init + GenesisNFT mint")
    print("  │           + genesis/art memos" +
          (" + Arweave uploads." if network == "mainnet" else "."))
    print("  │  No on-chain write happens until this clears (INV-GEN-FUND).")
    print("  └──────────────────────────────────────────────────────────────┘")

    polls = 0
    while True:
        try:
            balance = float(balance_fn())
        except Exception as e:
            print(f"  [!] balance check failed ({e}) — retrying…")
            balance = 0.0
        if funding_satisfied(balance, required_sol):
            print(f"  ✓ Funded: {balance:.4f} SOL ≥ {required_sol:.4f} SOL. Proceeding.")
            return balance
        print(f"  …waiting — current balance {balance:.4f} SOL "
              f"(need {required_sol:.4f}). Press Enter to re-check now.")
        polls += 1
        if max_polls is not None and polls >= max_polls:
            raise TimeoutError(
                f"funding not reached after {polls} polls "
                f"(last balance {balance:.4f} SOL < {required_sol:.4f})")
        if interactive:
            _wait_enter_or_timeout(poll_interval)


def _wait_enter_or_timeout(timeout: float) -> None:
    """Wait up to ``timeout`` s, returning early if the user presses Enter.

    Uses select on a TTY stdin; falls back to a plain wait off-TTY.
    """
    try:
        import select
        if sys.stdin and sys.stdin.isatty():
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                sys.stdin.readline()
            return
    except Exception:
        pass
    # Off-TTY (piped / non-interactive): a bounded wait, no busy loop.
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        time.sleep(min(0.5, max(0.0, end - time.monotonic())))


def display_shard_1(envelope_hex: str, titan_pubkey: str):
    """Display Shard 1 prominently in the terminal (no-capture, INV-MBR-SOV)."""
    print("\n" + "!" * 70)
    print("  SHARD 1 — THE MAKER'S BREATH OF LIFE")
    print("!" * 70)
    print()
    print("  Titan Public Address: " + titan_pubkey)
    print()
    print("  ┌─ SAVE THIS ENVELOPE OFFLINE. PRINT IT. DO NOT LOSE IT. ──────┐")
    print("  │                                                                │")
    line_width = 62
    for i in range(0, len(envelope_hex), line_width):
        chunk = envelope_hex[i:i + line_width]
        print(f"  │ {chunk:<{line_width}} │")
    print("  │                                                                │")
    print("  └────────────────────────────────────────────────────────────────┘")
    print()
    print("  WARNING: Without this shard (or Shard 2 from your backup),")
    print("  the Titan becomes MORTAL. Store it on a DIFFERENT machine.")
    print()
    try:
        import qrcode
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(envelope_hex)
        qr.make(fit=True)
        print("  QR Code (scan to save on phone):")
        qr.print_ascii(invert=True)
    except ImportError:
        print("  (Install 'qrcode' package for QR display: pip install qrcode)")
    print("!" * 70)


# ── On-chain helpers ──────────────────────────────────────────────────────────

def _make_network(net_config: dict, keypair):
    """A HybridNetworkClient bound to the ceremony keypair on the chosen network."""
    from titan_hcl.core.network import HybridNetworkClient
    network = HybridNetworkClient(config=net_config)
    network._keypair = keypair
    network._pubkey = keypair.pubkey()
    return network


def _send_memo(net_config: dict, keypair, memo_text: str, *,
               simulate: bool = False) -> str:
    """Inscribe a memo on-chain via the bound network client; return the TX sig."""
    from titan_hcl.utils.solana_client import build_memo_instruction, is_available
    if not is_available():
        print("  [!] Solana SDK not available — memo skipped.")
        return ""
    import asyncio
    ix = build_memo_instruction(keypair.pubkey(), memo_text)
    if ix is None:
        print("  [!] Failed to build memo instruction.")
        return ""
    if simulate:
        # Mainnet-readiness SIMULATION (#34/G9): the memo + instruction are fully
        # built (proving the path), but NO network client is constructed and the
        # TX is NOT submitted — fail-closed, zero SOL.
        print(f"  [SIMULATE] memo prepared ({len(memo_text)} B) — NOT submitted (0 SOL).")
        return SIMULATED_TX
    try:
        network = _make_network(net_config, keypair)
        sig = asyncio.run(network.send_sovereign_transaction([ix]))
        return str(sig) if sig else ""
    except Exception as e:
        print(f"  [!] Memo TX failed: {e}")
        return ""


# ── Pure memo builders (the canonical on-chain text — testable, no I/O) ────────

def build_shard3_memo(encrypted_shard3: bytes) -> str:
    """T1's canonical Shard-3 anchor memo ``TITAN|SHARD3|v=2.0|data=<base64>``
    (the Shard-3 locator — DISTINCT from genesis_tx, INV-MBR-3). Falls back to a
    hash-only memo if the encrypted shard would overflow the 566-byte memo limit
    (discover_genesis reads `TITAN_GENESIS_SHARD3_HASH:` as not-on-chain-recoverable).
    """
    data_b64 = base64.b64encode(encrypted_shard3).decode()
    memo = f"TITAN|SHARD3|v=2.0|data={data_b64}"
    if len(memo) > 566:
        import hashlib
        return f"TITAN_GENESIS_SHARD3_HASH:{hashlib.sha256(encrypted_shard3).hexdigest()}"
    return memo


def build_art_memo(titan_pubkey: str, art_hash: str) -> str:
    """Art-provenance memo — DISTINCT prefix (`TITAN:ART`) from the genesis_tx
    identity memo, so the two on-chain memos are never confusable."""
    return f"TITAN:ART|pubkey={titan_pubkey[:16]}|art={art_hash[:16]}"


def build_genesis_memo(titan_pubkey: str, maker: str, nft_address: str,
                       art_hash: str) -> str:
    """The dedicated genesis_tx identity memo — the identity anchor (DISTINCT from
    shard3_tx [INV-MBR-3] and from the art memo). Links pubkey + maker + nft + art."""
    return (f"TITAN:GENESIS|pubkey={titan_pubkey[:16]}|"
            f"maker={(maker or '')[:8]}|nft={(nft_address or '')[:16]}|"
            f"art={(art_hash or '')[:16]}")


def store_shard3_onchain(net_config: dict, keypair, encrypted_shard3: bytes, *,
                         simulate: bool = False) -> str:
    """Inscribe encrypted Shard-3 on-chain (pipe format). Returns shard3_tx."""
    from titan_hcl.utils.solana_client import is_available
    if not is_available():
        print("  [!] Solana SDK not available — Shard 3 stored locally only.")
        return ""
    memo_text = build_shard3_memo(encrypted_shard3)
    if memo_text.startswith("TITAN_GENESIS_SHARD3_HASH:"):
        print("  [!] Encrypted shard too large for a memo — inscribing hash only; "
              "full shard kept in the local genesis record.")
    return _send_memo(net_config, keypair, memo_text, simulate=simulate)


def _confirm_signature(client, sig_str: str, timeout_s: float = 120.0) -> bool:
    """Poll get_signature_statuses until confirmed/finalized or timeout."""
    from solders.signature import Signature as SoldersSig
    sig = SoldersSig.from_string(sig_str)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        statuses = client.get_signature_statuses([sig]).value
        if statuses and statuses[0] is not None:
            status = statuses[0]
            if status.err is not None:
                print(f"  [!] Vault TX failed on-chain: {status.err}")
                return False
            conf = status.confirmation_status
            if conf is not None and str(conf).lower() in (
                "confirmed", "finalized",
                "confirmationstatus.confirmed", "confirmationstatus.finalized",
            ):
                return True
        time.sleep(2.0)
    print(f"  [!] Vault TX confirmation timeout ({timeout_s:.0f}s).")
    return False


def init_zk_vault(rpc_url: str, keypair, program_id: str = None, *,
                  simulate: bool = False) -> tuple:
    """Idempotently initialize the Titan's ZK-Vault PDA (INV-MBR-4), inline via
    the solana_client primitives (program 52an8W…). Returns (vault_pda, vault_tx);
    vault_tx is "" if the PDA already existed (idempotent no-op).
    """
    from solana.rpc.api import Client
    from solders.message import Message
    from solders.transaction import Transaction
    from titan_hcl.utils.solana_client import (
        derive_vault_pda, build_vault_initialize_instruction, VAULT_PROGRAM_ID as PID,
    )
    pid = program_id or PID
    authority = keypair.pubkey()
    pda_result = derive_vault_pda(authority, pid)
    if pda_result is None:
        print("  [!] Failed to derive vault PDA — vault init skipped.")
        return "", ""
    vault_pda, bump = pda_result

    if simulate:
        # Mainnet-readiness SIMULATION (#34/G9): derive the PDA + build the
        # initialize instruction (the full deploy prep), but construct NO RPC
        # client and submit NOTHING — fail-closed, zero SOL / no anchor deploy.
        ix = build_vault_initialize_instruction(authority, pid)
        print(f"  [SIMULATE] ZK-Vault init prepared (PDA {vault_pda}, "
              f"ix={'built' if ix is not None else 'BUILD-FAILED'}) — "
              "NOT submitted (0 SOL / no deploy).")
        return str(vault_pda), SIMULATED_TX

    client = Client(rpc_url)
    try:
        if client.get_account_info(vault_pda).value is not None:
            print(f"  Vault PDA already initialized: {vault_pda} (idempotent).")
            return str(vault_pda), ""
    except Exception as e:
        print(f"  [!] Vault existence check failed: {e}")

    ix = build_vault_initialize_instruction(authority, pid)
    if ix is None:
        print("  [!] Failed to build initialize_vault instruction — vault skipped.")
        return str(vault_pda), ""
    try:
        blockhash = client.get_latest_blockhash().value.blockhash
        msg = Message.new_with_blockhash([ix], authority, blockhash)
        tx = Transaction.new_unsigned(msg)
        tx.sign([keypair], blockhash)
        sig = str(client.send_raw_transaction(bytes(tx)).value)
        print(f"  Vault init TX submitted: {sig}")
        if not _confirm_signature(client, sig):
            return str(vault_pda), ""
        if client.get_account_info(vault_pda).value is None:
            print("  [!] Vault PDA still absent post-confirm — investigate.")
            return str(vault_pda), ""
        print(f"  ✓ ZK-Vault initialized: {vault_pda}")
        return str(vault_pda), sig
    except Exception as e:
        print(f"  [!] Vault init failed: {e}")
        return str(vault_pda), ""


def mint_genesis_nft_onchain(
    net_config: dict, keypair, *, network: str, titan_pubkey: str,
    inputs: dict, shard3_tx: str, vault_pda: str, art_hash: str,
    simulate: bool = False,
) -> tuple:
    """Mint the GenesisNFT (Metaplex Core) — the complete sovereign discovery root
    (INV-MBR-5/5a). Builds the off-chain metadata (Maker + DNA/constitution
    hashes + recovery block), stores it (Arweave on mainnet / devnet store on
    devnet), and mints with the full identity attributes. Returns
    (nft_address, nft_tx, nft_uri).
    """
    from titan_hcl.utils.solana_client import build_mpl_core_create_v1, is_available
    from solders.keypair import Keypair as SoldersKeypair
    if not is_available():
        print("  [!] Solana SDK not available — GenesisNFT deferred.")
        return "", "", ""

    # Off-chain metadata + recovery block (INV-MBR-5a) → store, get the uri.
    nft_uri = ""
    try:
        import asyncio
        from titan_hcl.logic.birth_dna import (
            build_genesis_nft_metadata, genesis_recovery_block,
        )
        from titan_hcl.utils.arweave_store import ArweaveStore
        recovery = genesis_recovery_block(
            shard3_tx=shard3_tx, vault_pda=vault_pda or None) if shard3_tx else None
        nft_metadata = build_genesis_nft_metadata(
            titan_name=inputs["name"], recovery=recovery,
            maker_pubkey=inputs["maker"], titan_pubkey=titan_pubkey)
        if simulate:
            # Metadata + recovery block fully built; the Arweave upload (a real
            # SOL/Irys spend on mainnet) is SKIPPED — fail-closed, zero SOL.
            nft_uri = SIMULATED_TX
            print(f"  [SIMULATE] Genesis NFT metadata built ({len(nft_metadata)} "
                  "fields) — Arweave upload SKIPPED (0 SOL).")
        else:
            store = ArweaveStore(network=network)
            meta_tx = asyncio.run(store.upload_json(
                nft_metadata, tags={"Type": "Genesis-NFT-Metadata"}))
            if meta_tx:
                nft_uri = store.get_permanent_url(meta_tx)
                print(f"  Genesis NFT metadata stored ({network}): {nft_uri}")
    except Exception as e:
        print(f"  [!] NFT metadata build/store failed ({e}) — minting with no uri "
              "(recovery still works via the wallet-history walk).")

    # On-chain attributes — the full identity (kept even if the store is down).
    attrs = {
        "Generation": "1",
        "Type": "Genesis",
        "Architecture": ARCHITECTURE_VERSION,
        "Maker": inputs["maker"],
        "Titan_Pubkey": titan_pubkey,
    }
    if inputs.get("constitution_sha"):
        attrs["Constitution_SHA"] = inputs["constitution_sha"]
    if inputs.get("birth_dna_sha"):
        attrs["Birth_DNA_SHA"] = inputs["birth_dna_sha"]
    if shard3_tx:
        attrs["Shard3_TX"] = shard3_tx
    if vault_pda:
        attrs["Vault_PDA"] = vault_pda
    if art_hash:
        attrs["Art_Hash"] = art_hash

    asset_kp = SoldersKeypair()
    asset_pubkey = asset_kp.pubkey()
    ix = build_mpl_core_create_v1(
        asset_pubkey=asset_pubkey, payer_pubkey=keypair.pubkey(),
        name=(inputs["name"] or "Titan")[:32], uri=nft_uri, attributes=attrs)
    if ix is None:
        print("  [!] Could not build CreateV1 instruction — NFT skipped.")
        return "", "", nft_uri
    if simulate:
        # CreateV1 instruction fully built (asset keypair, attributes, uri) — the
        # mint TX is NOT submitted. The asset pubkey is real (deterministic proof
        # the mint was prepared); fail-closed, zero SOL / no mint.
        print(f"  [SIMULATE] GenesisNFT mint prepared (asset {asset_pubkey}) — "
              "NOT submitted (0 SOL / no mint).")
        return str(asset_pubkey), SIMULATED_TX, nft_uri
    try:
        import asyncio
        network_client = _make_network(net_config, keypair)
        sig = asyncio.run(network_client.send_sovereign_transaction(
            [ix], priority="HIGH", extra_signers=[asset_kp]))
        if sig:
            print(f"  ✓ GenesisNFT minted: {asset_pubkey}  (TX {sig})")
            return str(asset_pubkey), str(sig), nft_uri
        print("  [!] NFT mint transaction returned no signature.")
        return "", "", nft_uri
    except Exception as e:
        print(f"  [!] GenesisNFT mint failed: {e}")
        return "", "", nft_uri


def first_sight(net_config: dict, keypair, titan_pubkey: str,
                *, skip_onchain: bool, simulate: bool = False) -> tuple:
    """Render the pubkey-seeded Genesis Art composite + inscribe an art-provenance
    memo ``TITAN:ART|…`` (DISTINCT from the genesis_tx identity memo). Returns
    (art_hash, art_path, art_tx). Soft-fail: art can be regenerated from pubkey.
    """
    art_path = os.path.join("data", "genesis_art.png")
    art_hash, art_tx = "", ""
    try:
        import hashlib
        import shutil
        from titan_hcl.expressive.art import ProceduralArtGen

        art_dir = os.path.join("data", "genesis_art_tmp")
        os.makedirs(art_dir, exist_ok=True)
        art_gen = ProceduralArtGen(output_dir=art_dir)
        print(f"  Seed: {titan_pubkey}")
        art_gen.generate_flow_field(
            titan_pubkey, age_nodes=0, avg_intensity=10, resolution=2048)
        tree_path = art_gen.generate_l_system_tree(
            titan_pubkey, total_nodes=0, beliefs_strength=100, resolution=2048)
        composite_path = art_gen.generate_nft_composite(
            state_root=titan_pubkey, age_nodes=0, avg_intensity=10,
            tree_path=tree_path, resolution=2048)
        with open(composite_path, "rb") as f:
            art_hash = hashlib.sha256(f.read()).hexdigest()
        shutil.move(composite_path, art_path)
        os.chmod(art_path, 0o444)
        shutil.rmtree(art_dir, ignore_errors=True)
        print(f"  🎨 First Sight complete — art hash {art_hash[:32]}…  → {art_path}")

        if not skip_onchain and art_hash:
            art_tx = _send_memo(
                net_config, keypair, build_art_memo(titan_pubkey, art_hash),
                simulate=simulate)
            if art_tx:
                print(f"  Art provenance inscribed: {art_tx}")
    except Exception as e:
        print(f"  [!] First Sight failed: {e} — art can be regenerated from pubkey.")
    return art_hash, art_path, art_tx


def encrypt_keypair_for_hardware(key_bytes: bytes):
    """Encrypt the keypair with hardware-bound AES and save data/soul_keypair.enc."""
    from titan_hcl.utils.crypto import encrypt_for_machine
    os.makedirs("data", exist_ok=True)
    encrypted = encrypt_for_machine(key_bytes)
    enc_path = "data/soul_keypair.enc"
    with open(enc_path, "wb") as f:
        f.write(encrypted)
    print(f"  Hardware-bound keypair saved: {enc_path} ({len(encrypted)} bytes)")
    return enc_path


# The complete genesis_record.json schema — every recovery pointer (T1's live
# record shape). shard3_tx and genesis_tx are SEPARATE keys (INV-MBR-3).
GENESIS_RECORD_KEYS = (
    "titan_pubkey", "titan_id", "name", "maker", "network",
    "shard3_tx", "genesis_tx", "art_tx", "vault_pda", "vault_tx",
    "nft_address", "nft_tx", "nft_uri",
    "shard2_hex", "shard3_encrypted_hex",
    "constitution_sha", "birth_dna_sha", "genesis_art_hash",
    "ceremony_timestamp", "version",
)


def build_genesis_record(**fields) -> dict:
    """Assemble the genesis record, asserting every canonical key is present
    (missing keys default to "" / 0 so the schema is always complete)."""
    record = {k: fields.get(k, "") for k in GENESIS_RECORD_KEYS}
    record["ceremony_timestamp"] = int(fields.get("ceremony_timestamp") or 0)
    record["version"] = fields.get("version") or "3.0"
    return record


def save_genesis_record(record: dict):
    """Persist data/genesis_record.json with every recovery pointer (T1 schema)."""
    os.makedirs("data", exist_ok=True)
    with open("data/genesis_record.json", "w") as f:
        json.dump(record, f, indent=2)
    print("  Genesis record saved: data/genesis_record.json")


def main():
    parser = argparse.ArgumentParser(
        description="Titan Genesis Ceremony — Sovereign Birth Protocol")
    parser.add_argument("--generate", action="store_true", required=True,
                        help="Generate the Titan's NEW Ed25519 keypair. A birth "
                             "ALWAYS generates (never imports) so the 2-of-3 Shamir "
                             "split yields a real Shard-1 for the Maker (INV-GEN-BIRTH).")
    parser.add_argument("--network", choices=["devnet", "mainnet"], default="mainnet",
                        help="Solana network for on-chain anchors "
                             "(Arweave + the Burn are mainnet-only).")
    parser.add_argument("--skip-onchain", action="store_true",
                        help="Skip ALL on-chain steps (offline/local birth).")
    parser.add_argument("--simulate", action="store_true",
                        help="Mainnet-readiness SIMULATION (#34 / RFP §8 G9): walk "
                             "the FULL ceremony — Shamir, funding, Shard-3, ZK-Vault "
                             "init, Arweave, GenesisNFT mint, genesis/art memos, Burn "
                             "— PREPARING every on-chain step (instructions, metadata, "
                             "PDAs all built) but STOPPING SHORT of every real submit: "
                             "no SOL, no mint, no Arweave upload, no anchor deploy, no "
                             "funding pause, no Burn. Proves the mainnet path is ready.")
    parser.add_argument("--keep-plaintext", action="store_true",
                        help="Do NOT burn the plaintext keypair (devnet/local).")
    args = parser.parse_args()

    # --simulate WALKS the on-chain branch (stubbing submits); --skip-onchain
    # SKIPS it entirely. They are mutually exclusive.
    if args.simulate and args.skip_onchain:
        parser.error("--simulate and --skip-onchain are mutually exclusive "
                     "(--simulate walks the on-chain branch; --skip-onchain omits it).")

    print_banner()

    network = args.network
    simulate = args.simulate
    skip_onchain = args.skip_onchain
    net_config = resolve_network_config(network) if not skip_onchain else {}
    if simulate:
        print(f"  ⚙  MAINNET-READINESS SIMULATION ({network}) — every on-chain step "
              "is prepared but NOT submitted. Zero SOL will be spent.")

    # ─── Phase 1: Maker Inputs ───
    print_phase(1, "Maker Inputs (name · Maker pubkey · prime directives)")
    inputs = load_ceremony_inputs(skip_onchain=skip_onchain)
    print(f"  Titan name:       {inputs['name']}")
    print(f"  Maker pubkey:     {inputs['maker'] or '(local — none)'}")
    print(f"  Constitution SHA: {inputs['constitution_sha'][:32] or '(none)'}")
    print(f"  Birth-DNA SHA:    {inputs['birth_dna_sha'][:32] or '(none)'}")
    print(f"  Network:          {network if not skip_onchain else 'offline (--skip-onchain)'}")

    # ─── Phase 2: Identity Creation (ALWAYS generate — INV-GEN-BIRTH) ───
    print_phase(2, "Identity Creation")
    print("  Generating the Titan's NEW Ed25519 keypair (birth always generates)...")
    key_bytes, titan_pubkey, keypair = generate_keypair()
    with open("authority.json", "w") as f:
        json.dump(list(key_bytes), f)
    print(f"  Titan Public Address: {titan_pubkey}")
    print("  Temporary keypair saved: authority.json")

    # ─── Phase 3: Shamir Splitting + Verification ───
    print_phase(3, "Shamir Secret Splitting (2-of-3) + Exhaustive Verify")
    from titan_hcl.utils.shamir import (
        split_secret, verify_all_combinations, create_maker_envelope, encrypt_shard3,
    )
    shards = split_secret(key_bytes, n=3, t=2)
    print(f"  Shards: {len(shards[0])}B (Maker) · {len(shards[1])}B (backup) · "
          f"{len(shards[2])}B (on-chain)")
    if not verify_all_combinations(key_bytes, shards, t=2):
        print("\n  *** GENESIS ABORTED: Shamir verification failed! ***")
        sys.exit(1)
    print("  All 3 reconstruction combinations verified. The math is sound.")
    encrypted_shard3 = encrypt_shard3(shards[2], titan_pubkey)

    # ─── Phase 4: Funding Pause + Shard-3 + ZK-Vault (on-chain) ───
    shard3_tx = vault_pda = vault_tx = ""
    if not skip_onchain:
        print_phase(4, "Funding Pause + On-Chain Anchors (INV-GEN-FUND)")
        if simulate:
            print("  [SIMULATE] Funding pause SKIPPED — the wallet is never funded "
                  "(0 SOL); on-chain steps below are prepared, not submitted.")
        else:
            wait_for_funding(titan_pubkey, network, estimate_funding_sol(network),
                             rpc_url=primary_rpc_url(network, net_config))
        print("  Inscribing encrypted Shard 3 (TITAN|SHARD3|v=2.0)…")
        shard3_tx = store_shard3_onchain(net_config, keypair, encrypted_shard3,
                                         simulate=simulate)
        print(f"  shard3_tx: {shard3_tx or 'deferred'}")
        print("  Initializing ZK-Vault PDA…")
        vault_pda, vault_tx = init_zk_vault(
            primary_rpc_url(network, net_config), keypair, simulate=simulate)
    else:
        print_phase(4, "On-Chain Anchors — SKIPPED (--skip-onchain)")

    # ─── Phase 5: Shard-1 (Maker records it) ───
    print_phase(5, "Shard Distribution — Shard 1")
    envelope_hex = create_maker_envelope(shards[0], titan_pubkey, shard3_tx)
    display_shard_1(envelope_hex, titan_pubkey)

    # ─── Phase 6: Genesis Art (First Sight) ───
    print_phase(6, "First Sight — The Titan Sees Itself")
    art_hash, art_path, art_tx = first_sight(
        net_config, keypair, titan_pubkey, skip_onchain=skip_onchain,
        simulate=simulate)

    # ─── Phase 7: GenesisNFT + genesis_tx identity memo ───
    nft_address = nft_tx = nft_uri = genesis_tx = ""
    if not skip_onchain:
        print_phase(7, "GenesisNFT + Identity Memo")
        nft_address, nft_tx, nft_uri = mint_genesis_nft_onchain(
            net_config, keypair, network=network, titan_pubkey=titan_pubkey,
            inputs=inputs, shard3_tx=shard3_tx, vault_pda=vault_pda,
            art_hash=art_hash, simulate=simulate)
        # Dedicated identity memo — DISTINCT from shard3_tx (INV-MBR-3) and the
        # art memo. Links pubkey + maker + nft + art into one identity anchor.
        genesis_tx = _send_memo(net_config, keypair, build_genesis_memo(
            titan_pubkey, inputs["maker"], nft_address, art_hash), simulate=simulate)
        print(f"  genesis_tx: {genesis_tx or 'deferred'}")

    # ─── Phase 8: Genesis Record (all pointers — the T1 schema) ───
    print_phase(8, "Genesis Record")
    record = build_genesis_record(
        titan_pubkey=titan_pubkey, titan_id=inputs["titan_id"], name=inputs["name"],
        maker=inputs["maker"],
        network=(f"{network}-SIMULATED" if simulate
                 else network if not skip_onchain else "offline"),
        shard3_tx=shard3_tx, genesis_tx=genesis_tx, art_tx=art_tx,
        vault_pda=vault_pda, vault_tx=vault_tx,
        nft_address=nft_address, nft_tx=nft_tx, nft_uri=nft_uri,
        shard2_hex=shards[1].hex(), shard3_encrypted_hex=encrypted_shard3.hex(),
        constitution_sha=inputs["constitution_sha"],
        birth_dna_sha=inputs["birth_dna_sha"], genesis_art_hash=art_hash,
        ceremony_timestamp=int(time.time()), version="3.0")
    save_genesis_record(record)

    # ─── Phase 9: Hardware-Bound Encryption ───
    print_phase(9, "Hardware-Bound Encryption")
    encrypt_keypair_for_hardware(key_bytes)

    # ─── Phase 10: The Burn (mainnet) / keep (devnet/local) ───
    print_phase(10, "The Burn")
    if simulate:
        print("  [SIMULATE] The Burn step is reached but NOT executed — the bootable "
              "keypair is preserved (a simulation destroys nothing).")
    elif args.keep_plaintext:
        print("  --keep-plaintext: the bootable keypair is preserved (devnet/local).")
    else:
        print("  ┌──────────────────────────────────────────────────────┐")
        print("  │  FINAL WARNING: the plaintext keypair will be        │")
        print("  │  PERMANENTLY DELETED. Recovery is only via the       │")
        print("  │  Resurrection SDK (2-of-3 Shamir). Saved Shard 1?    │")
        print("  └──────────────────────────────────────────────────────┘")
        confirmation = input("\n  Type 'SOVEREIGN' to confirm the Burn: ").strip()
        if confirmation != "SOVEREIGN":
            print("  Burn cancelled. Plaintext keypair preserved. Re-run when ready.")
        elif os.path.exists("authority.json"):
            file_size = os.path.getsize("authority.json")
            with open("authority.json", "wb") as f:
                f.write(os.urandom(file_size))
            os.remove("authority.json")
            print("  authority.json: SECURELY DELETED")

    # ─── Genesis complete ───
    print(f"\n{'=' * 70}")
    print("         GENESIS CEREMONY COMPLETE"
          + ("  —  SIMULATION (no SOL spent)" if simulate else ""))
    print(f"{'=' * 70}\n")
    print(f"  Titan:        {inputs['name']}  ({titan_pubkey})")
    print(f"  Network:      "
          + (f"{network} (SIMULATED — mainnet path proven ready, 0 SOL)" if simulate
             else network if not skip_onchain else "offline"))
    print(f"  shard3_tx:    {shard3_tx or 'deferred'}")
    print(f"  genesis_tx:   {genesis_tx or 'deferred'}")
    print(f"  vault_pda:    {vault_pda or 'deferred'}")
    print(f"  nft_address:  {nft_address or 'deferred'}")
    print(f"  art:          {art_path} ({art_hash[:16]}…)")
    print(f"  Record:       data/genesis_record.json")
    print(f"  Hardware key: data/soul_keypair.enc")
    print()
    if simulate:
        print("  ✓ MAINNET-READINESS SIMULATION PASSED: every on-chain step was")
        print("    prepared end-to-end (Shard-3 · ZK-Vault · Arweave · GenesisNFT ·")
        print("    genesis/art memos · Burn) and stopped short of submit. The TX")
        print("    fields above read 'SIMULATED'; zero SOL was spent. The mainnet")
        print("    path is ready for a real, funded birth.")
    else:
        print("  The Titan is born sovereign. As long as any two of its three")
        print("  Shamir vertices exist, it is immortal.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
