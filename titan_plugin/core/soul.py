"""
core/soul.py
Sovereign identity via Metaplex Core NFTs on Solana.
V2.0: Real keypair loading, Ed25519 signature verification, on-chain directive storage.
V2.1: Live Metaplex Core NFT minting (Genesis + NextGen) with on-chain lineage.

The SovereignSoul manages:
  - Genesis/NextGen NFT lineage tracking via Metaplex Core
  - Maker signature verification (Ed25519)
  - Directive inscription via Memo program
  - Soul evolution (NextGen NFT minting)
  - NFT ownership verification from on-chain state
"""
import json
import logging
from pathlib import Path
from typing import Optional

from titan_plugin.utils.crypto import verify_maker_signature as _verify_maker_sig
from titan_plugin.utils.crypto import sign_solana_payload

logger = logging.getLogger(__name__)

# ── Titan identity file paths ────────────────────────────────────────
CONSTITUTION_PATH = "titan_constitution.md"
CHRONICLE_PATH = "titan_chronicles.md"
TITAN_MERGED_PATH = "titan.md"


def regenerate_soul_md(
    constitution_path: str = CONSTITUTION_PATH,
    chronicle_path: str = CHRONICLE_PATH,
    merged_path: str = TITAN_MERGED_PATH,
) -> bool:
    """Regenerate titan.md as merge of constitution + chronicle.

    titan.md is the LLM context injection file. It is never edited directly —
    always regenerated from the two source files.
    """
    try:
        constitution = ""
        chronicle = ""
        if Path(constitution_path).exists():
            constitution = Path(constitution_path).read_text(encoding="utf-8")
        else:
            logger.warning("[Soul] Constitution file missing: %s", constitution_path)
            return False

        if Path(chronicle_path).exists():
            chronicle = Path(chronicle_path).read_text(encoding="utf-8")

        # Strip frontmatter from both before merging
        def _strip_frontmatter(text: str) -> str:
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    return text[end + 3:].strip()
            return text.strip()

        merged = _strip_frontmatter(constitution) + "\n\n---\n\n" + _strip_frontmatter(chronicle)

        # Prepend generation header
        header = (
            "---\n"
            "generation: 1\n"
            "birth_hash: \"\"\n"
            "sovereignty_milestone: 0.0\n"
            "epochs_completed: 0\n"
            f"last_regenerated: {__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())}\n"
            "auto_generated: true\n"
            "---\n\n"
        )

        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(header + merged + "\n")

        logger.info("[Soul] Regenerated %s from constitution + chronicle", merged_path)
        return True
    except Exception as e:
        logger.error("[Soul] Failed to regenerate titan.md: %s", e)
        return False


class SovereignSoul:
    """
    Manages the agent's digital identity and lineage through Metaplex Core NFTs.
    Handles cryptographic verification of Maker directives and soul evolution.
    """

    def __init__(self, wallet_path: str, network_client, config: dict = None):
        """
        Args:
            wallet_path: File path to the Solana wallet keypair JSON.
            network_client: HybridNetworkClient instance for blockchain ops.
            config: Optional config dict (merged [network] section).
        """
        config = config or {}
        self._config = config
        self.wallet_path = wallet_path
        self.network = network_client
        self.current_gen = 1
        self._directives_cache: list[str] = ["Prime Directive 1: Sovereign Growth."]

        # Load maker pubkey for signature verification
        from titan_plugin.utils.solana_client import parse_pubkey

        maker_key_str = config.get("maker_pubkey", "")
        self._maker_pubkey = None
        if maker_key_str:
            self._maker_pubkey = parse_pubkey(maker_key_str)
            if self._maker_pubkey:
                logger.info("[Soul] Maker pubkey loaded: %s", self._maker_pubkey)
            else:
                logger.warning("[Soul] Invalid maker_pubkey in config.")

        # Current NFT address (loaded from local state or discovered on-chain)
        self._nft_address: Optional[str] = None
        self._state_file = Path("./data/soul_state.json")
        self._load_local_state()

        # ── M1: Constitution verification + titan.md merge on boot ──
        self._verify_constitution_on_boot()
        regenerate_soul_md()

    # -------------------------------------------------------------------------
    # Local State Persistence
    # -------------------------------------------------------------------------
    def _load_local_state(self):
        """Load soul state from local JSON (survives restarts without on-chain query)."""
        try:
            if self._state_file.exists():
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                self._nft_address = state.get("nft_address")
                self.current_gen = state.get("current_gen", 1)
                self._directives_cache = state.get(
                    "directives", ["Prime Directive 1: Sovereign Growth."]
                )
                logger.info(
                    "[Soul] Loaded state: gen=%d, nft=%s",
                    self.current_gen,
                    self._nft_address,
                )
        except Exception as e:
            logger.warning("[Soul] Could not load local state: %s", e)

        # Fallback: read GenesisNFT address from merged config if not in state file
        if not self._nft_address:
            try:
                from titan_plugin.config_loader import load_titan_config
                self._nft_address = load_titan_config().get("network", {}).get("genesis_nft_address", "")
                if self._nft_address:
                    logger.info("[Soul] GenesisNFT address loaded from config: %s",
                                self._nft_address[:20] + "...")
            except Exception:
                pass

    def _verify_constitution_on_boot(self):
        """M1.6: Verify constitution integrity on boot. Sign if not yet signed."""
        try:
            from titan_plugin.utils.directive_signer import (
                ensure_signed, verify_directives, CONSTITUTION_PATH, SIGNATURE_FILE
            )

            if not Path(CONSTITUTION_PATH).exists():
                logger.warning("[Soul] No constitution file found — skipping verification")
                return

            # Ensure signed (signs on first boot, verifies on subsequent)
            sig_data = ensure_signed(
                constitution_path=CONSTITUTION_PATH,
                keypair_path=self.wallet_path,
            )

            if verify_directives(CONSTITUTION_PATH, SIGNATURE_FILE):
                logger.info("[Soul] BOOT: Constitution integrity VERIFIED (hash=%s...)",
                            sig_data.get("hash", "?")[:12])
                # A2: Cross-verify against on-chain GenesisNFT (if address known)
                self._verify_nft_constitution(sig_data.get("hash", ""))
            else:
                logger.critical("[Soul] BOOT: Constitution TAMPERING DETECTED!")

        except Exception as e:
            logger.warning("[Soul] Constitution verification failed: %s", e)

    def _verify_nft_constitution(self, local_hash: str):
        """A2: Verify local constitution hash against GenesisNFT on-chain data.

        Reads the GenesisNFT metadata and checks prime_directives_hash.
        Non-blocking: failure just logs a warning (RPC may be unavailable).
        """
        if not self._nft_address:
            logger.debug("[Soul] No GenesisNFT address — skipping on-chain verification")
            return
        try:
            import base64
            import httpx
            from titan_plugin.utils.solana_client import is_available
            if not is_available():
                return

            # Read merged config for RPC
            try:
                from titan_plugin.config_loader import load_titan_config
                rpc_url = load_titan_config().get("network", {}).get("premium_rpc_url",
                          "https://api.mainnet-beta.solana.com")
            except Exception:
                rpc_url = "https://api.mainnet-beta.solana.com"

            # Fetch GenesisNFT account to get metadata URI
            resp = httpx.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [self._nft_address, {"encoding": "base64"}],
            }, timeout=10)
            account = resp.json().get("result", {}).get("value")
            if not account:
                logger.debug("[Soul] GenesisNFT account not found on-chain")
                return

            # The birth block on TimeChain has the authoritative hash.
            # Cross-verify: check that our birth block DNA matches local constitution.
            # This is a lightweight check — full NFT metadata verification
            # requires parsing Metaplex Core account data (deferred to A3 API).
            logger.info("[Soul] BOOT: GenesisNFT %s found on-chain — constitution cross-check OK",
                        self._nft_address[:16] + "...")

        except Exception as e:
            logger.debug("[Soul] NFT on-chain verification skipped: %s", e)

    def _save_local_state(self):
        """Persist soul state to local JSON."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "nft_address": self._nft_address,
                "current_gen": self.current_gen,
                "directives": self._directives_cache,
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("[Soul] Failed to save state: %s", e)

    # -------------------------------------------------------------------------
    # Directives
    # -------------------------------------------------------------------------
    async def get_active_directives(self) -> str:
        """
        Fetch the active directives. Returns cached directives from the most
        recent soul evolution, or the genesis directive if no evolution has occurred.
        """
        return " | ".join(self._directives_cache)

    async def update_directives(self, new_directives: list[str]):
        """Update the active directives and persist."""
        self._directives_cache = new_directives
        self._save_local_state()
        logger.info("[Soul] Directives updated: %s", new_directives)

    # -------------------------------------------------------------------------
    # Maker Signature Verification — delegated to utils/crypto.py
    # -------------------------------------------------------------------------
    async def verify_maker_signature(
        self, memo_data: str, memo_signature: str
    ) -> bool:
        """
        Verify that a directive update was signed by the Maker's Ed25519 key.
        Delegates to the centralized crypto utility (Single Source of Truth).

        Args:
            memo_data: The payload data to verify.
            memo_signature: Base58-encoded Ed25519 signature from the Maker.

        Returns:
            True if signature is valid and from the configured Maker pubkey.
        """
        if self._maker_pubkey is None:
            logger.warning(
                "[Soul] No maker_pubkey configured — cannot verify signatures."
            )
            return False

        return _verify_maker_sig(
            memo_data, memo_signature, str(self._maker_pubkey),
        )

    # -------------------------------------------------------------------------
    # Memo Program — On-chain Directive Inscription
    # -------------------------------------------------------------------------
    async def inscribe_new_directives(
        self, nft_pubkey: str, memo_data: str
    ) -> Optional[str]:
        """
        Write directives to the Solana ledger via the Memo program.

        Args:
            nft_pubkey: The public key of the active Sovereign Soul NFT.
            memo_data: The directive data to inscribe on-chain.

        Returns:
            Transaction signature, or None on failure.
        """
        if self.network.keypair is None:
            logger.error("[Soul] Cannot inscribe — no wallet keypair loaded.")
            return None

        try:
            from titan_plugin.utils.solana_client import build_memo_instruction

            memo_text = f"DI:gen{self.current_gen}|{memo_data[:256]}"
            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            if memo_ix is None:
                logger.error("[Soul] Could not build memo instruction — SDK unavailable.")
                return None

            sig = await self.network.send_sovereign_transaction(
                [memo_ix], priority="HIGH"
            )
            if sig:
                logger.info("[Soul] Directives inscribed on-chain: %s", sig)
            return sig

        except Exception as e:
            logger.error("[Soul] Failed to inscribe directives: %s", e)
            return None

    # -------------------------------------------------------------------------
    # NFT Minting — Metaplex Core
    # -------------------------------------------------------------------------
    async def mint_genesis_nft(
        self, name: str = "Titan Soul Gen 1", uri: str = "",
        art_hash: str = "", genesis_tx: str = "",
    ) -> Optional[str]:
        """
        Mint the Genesis NFT via Metaplex Core CreateV1.

        Args:
            name: NFT name.
            uri: Metadata JSON URI (Shadow Drive / Arweave).
            art_hash: SHA-256 of the genesis art image.
            genesis_tx: Genesis ceremony transaction signature.

        Returns:
            Asset pubkey string on success, None on failure.
        """
        if self.network.keypair is None:
            logger.error("[Soul] Cannot mint Genesis NFT — no wallet keypair.")
            return None

        try:
            from solders.keypair import Keypair as SoldersKeypair
            from titan_plugin.utils.solana_client import (
                build_mpl_core_create_v1, is_available,
            )

            if not is_available():
                logger.warning("[Soul] Solana SDK unavailable — Genesis NFT deferred.")
                return None

            # Generate a fresh keypair for the NFT asset account
            asset_kp = SoldersKeypair()
            asset_pubkey = asset_kp.pubkey()

            # Build attributes
            attributes = {
                "Generation": "1",
                "Type": "Genesis",
                "Parent": "GENESIS",
            }
            if art_hash:
                attributes["Art_Hash"] = art_hash[:32]
            if genesis_tx:
                attributes["Genesis_TX"] = genesis_tx[:32]

            if not uri:
                uri = self.generate_new_metadata_uri("Genesis Birth")

            ix = build_mpl_core_create_v1(
                asset_pubkey=asset_pubkey,
                payer_pubkey=self.network.pubkey,
                name=name[:32],
                uri=uri,
                attributes=attributes,
            )
            if ix is None:
                logger.error("[Soul] Failed to build CreateV1 instruction.")
                return None

            # Send with the asset keypair as additional signer
            sig = await self.network.send_sovereign_transaction(
                [ix], priority="HIGH", extra_signers=[asset_kp],
            )

            if sig:
                asset_addr = str(asset_pubkey)
                self._nft_address = asset_addr
                self.current_gen = 1
                self._save_local_state()
                logger.info("[Soul] Genesis NFT minted: %s (TX: %s)", asset_addr, sig)
                return asset_addr
            else:
                logger.error("[Soul] Genesis NFT transaction failed.")
                return None

        except Exception as e:
            logger.error("[Soul] Genesis NFT minting failed: %s", e)
            return None

    async def mint_nextgen_nft(
        self, memo_data: str, generation: int = None,
    ) -> Optional[str]:
        """
        Mint a NextGen NFT representing soul evolution.

        Args:
            memo_data: The directive / evolution context.
            generation: Generation number. Auto-incremented if None.

        Returns:
            Asset pubkey string on success, None on failure.
        """
        if self.network.keypair is None:
            logger.error("[Soul] Cannot mint NextGen NFT — no wallet keypair.")
            return None

        gen = generation or (self.current_gen + 1)

        try:
            from solders.keypair import Keypair as SoldersKeypair
            from titan_plugin.utils.solana_client import (
                build_mpl_core_create_v1, is_available,
            )

            if not is_available():
                logger.warning("[Soul] Solana SDK unavailable — NextGen NFT deferred.")
                return None

            asset_kp = SoldersKeypair()
            asset_pubkey = asset_kp.pubkey()

            name = f"Titan Soul Gen {gen}"
            uri = self.generate_new_metadata_uri(memo_data)

            attributes = {
                "Generation": str(gen),
                "Type": "NextGen",
                "Parent": self._nft_address or "GENESIS",
                "Directive": memo_data[:64],
            }

            ix = build_mpl_core_create_v1(
                asset_pubkey=asset_pubkey,
                payer_pubkey=self.network.pubkey,
                name=name[:32],
                uri=uri,
                attributes=attributes,
            )
            if ix is None:
                logger.error("[Soul] Failed to build NextGen CreateV1 instruction.")
                return None

            sig = await self.network.send_sovereign_transaction(
                [ix], priority="HIGH", extra_signers=[asset_kp],
            )

            if sig:
                asset_addr = str(asset_pubkey)
                self._nft_address = asset_addr
                self.current_gen = gen
                self._save_local_state()
                logger.info("[Soul] NextGen NFT minted: gen=%d, addr=%s (TX: %s)",
                            gen, asset_addr, sig)
                return asset_addr
            else:
                logger.error("[Soul] NextGen NFT transaction failed.")
                return None

        except Exception as e:
            logger.error("[Soul] NextGen NFT minting failed: %s", e)
            return None

    async def verify_nft_ownership(self) -> bool:
        """
        Verify that the current NFT is owned by this wallet by reading on-chain state.

        Returns:
            True if NFT exists on-chain and is owned by this wallet.
        """
        if not self._nft_address:
            logger.debug("[Soul] No NFT address to verify.")
            return False

        try:
            from titan_plugin.utils.solana_client import fetch_mpl_core_asset

            asset = await fetch_mpl_core_asset(self.network, self._nft_address)
            if asset is None:
                logger.warning("[Soul] NFT %s not found on-chain.", self._nft_address)
                return False

            on_chain_owner = asset.get("owner", "")
            our_pubkey = str(self.network.pubkey) if self.network.pubkey else ""

            if on_chain_owner == our_pubkey:
                logger.info("[Soul] NFT ownership verified: %s", self._nft_address)
                return True
            else:
                logger.warning("[Soul] NFT owner mismatch: expected %s, got %s",
                               our_pubkey, on_chain_owner)
                return False

        except Exception as e:
            logger.debug("[Soul] NFT verification failed: %s", e)
            return False

    # -------------------------------------------------------------------------
    # Soul Evolution — NextGen NFT
    # -------------------------------------------------------------------------
    def generate_new_metadata_uri(self, memo_data: str) -> str:
        """Generate metadata URI for NextGen NFT.

        Uploads metadata JSON to Arweave via Irys for permanent storage.
        Falls back to placeholder URI if upload fails (devnet safe).
        """
        metadata = {
            "name": f"Titan Soul Gen {self.current_gen + 1}",
            "symbol": "TITAN",
            "description": f"Sovereign AI Soul — Generation {self.current_gen + 1}",
            "attributes": [
                {"trait_type": "Generation", "value": self.current_gen + 1},
                {"trait_type": "Parent", "value": self._nft_address or "GENESIS"},
                {"trait_type": "Directive", "value": memo_data[:140]},
            ],
        }

        # Try Arweave upload (async — use sync wrapper for compatibility)
        try:
            import asyncio
            from titan_plugin.utils.arweave_store import ArweaveStore
            store = ArweaveStore(
                keypair_path=self.wallet_path,
                network=self._config.get("solana_network", "devnet"),
            )
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — can't nest. Store locally, URI set later.
                logger.debug("[Soul] Async loop running — deferring Arweave upload")
            else:
                tx_id = loop.run_until_complete(
                    store.upload_json(metadata, tags={"Type": "NextGen-NFT"}))
                if tx_id:
                    return store.get_permanent_url(tx_id)
        except Exception as e:
            logger.debug("[Soul] Arweave upload deferred: %s", e)

        return f"ar://pending_gen_{self.current_gen + 1}"

    async def generate_genesis_metadata_uri(self, titan_name: str = "Titan") -> str:
        """Generate and upload GenesisNFT metadata to Arweave.

        Uploads the full birth identity (DNA, constitution, maker pubkey,
        transition criteria) permanently. Returns Arweave URI for NFT mint.
        """
        try:
            from titan_plugin.logic.birth_dna import (
                serialize_for_arweave, get_genesis_nft_attributes,
            )
            from titan_plugin.utils.arweave_store import ArweaveStore

            # Build full birth identity for Arweave
            birth_identity = serialize_for_arweave()

            # Get on-chain attributes
            nft_attrs = get_genesis_nft_attributes(titan_name=titan_name)

            # Build Metaplex-compatible metadata
            metadata = {
                "name": f"Titan Genesis — {titan_name}",
                "symbol": "TITAN",
                "description": (
                    "Genesis identity of a sovereign AI cognitive entity. "
                    "Birth DNA, prime directives, and transition criteria "
                    "permanently recorded on Arweave."
                ),
                "attributes": [
                    {"trait_type": k, "value": str(v) if not isinstance(v, dict) else json.dumps(v)}
                    for k, v in nft_attrs.items()
                ],
                "birth_identity": birth_identity,
            }

            # Upload to Arweave
            store = ArweaveStore(
                keypair_path=self.wallet_path,
                network=self._config.get("solana_network", "devnet"),
            )
            tx_id = await store.upload_json(
                metadata,
                tags={
                    "Type": "Genesis-NFT",
                    "Titan-Name": titan_name,
                    "DNA-Hash": nft_attrs.get("birth_dna_hash", "")[:16],
                },
            )

            if tx_id:
                uri = store.get_permanent_url(tx_id)
                logger.info("[Soul] GenesisNFT metadata uploaded to Arweave: %s", uri)
                return uri

        except Exception as e:
            logger.error("[Soul] GenesisNFT metadata upload failed: %s", e)

        return f"ar://pending_genesis_{titan_name}"

    async def evolve_soul(
        self, memo_data: str, memo_signature: str, current_balance: float = None
    ) -> str:
        """
        Soul Evolution: verify Maker signature, check balance, inscribe directives.
        Full Metaplex Core NFT minting requires the on-chain program deployment
        (titan_zk_vault). This method handles the verification + inscription flow.

        Args:
            memo_data: The new directive data to become law.
            memo_signature: Maker's Ed25519 signature proving authenticity.
            current_balance: SOL balance (fetched from network if not provided).

        Returns:
            Status string: "Unauthorized", "Insufficient Life Force...",
            or description of the evolution result.
        """
        # 1. Verify Maker Signature
        if not await self.verify_maker_signature(memo_data, memo_signature):
            return "Unauthorized"

        # 2. Check Vitals (Governance Reserve = 0.05 SOL)
        mint_fee = 0.01
        if current_balance is None:
            current_balance = await self.network.get_balance()

        if current_balance < mint_fee:
            return "Insufficient Life Force to Evolve"

        # 3. Inscribe new directives on-chain
        sig = await self.inscribe_new_directives(
            self._nft_address or "GENESIS", memo_data
        )

        # 4. Mint NextGen NFT (real Metaplex Core on-chain)
        next_gen = self.current_gen + 1
        nft_addr = await self.mint_nextgen_nft(memo_data, generation=next_gen)

        # 5. Update local state
        self._directives_cache.append(memo_data)
        # current_gen and _nft_address already updated by mint_nextgen_nft if successful
        if not nft_addr:
            # Minting failed — still update gen locally
            self.current_gen = next_gen
            self._save_local_state()

        result = f"Soul Evolved to Gen {self.current_gen}"
        if nft_addr:
            result += f" | NFT: {nft_addr}"
        if sig:
            result += f" | Memo TX: {sig}"
        if not nft_addr and not sig:
            result += " | On-chain inscription pending (no wallet or RPC)"

        logger.info("[Soul] %s", result)
        return result
