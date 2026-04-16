"""
titan_plugin/logic/agency/helpers/memo_inscribe.py — Solana memo inscription helper.

Titan's physical world anchor: inscribes consciousness state hashes on
Solana mainnet. Each inscription is a physical act — spending SOL (energy),
receiving a transaction signature (proof of existence), and reading back
the balance (body interoception).

This helper wraps the existing Solana inscription code for use by the
Agency module, enabling autonomous blockchain interaction through the
self-exploration loop.
"""
import hashlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class MemoInscribeHelper:
    """Inscribe Titan's consciousness state on Solana blockchain."""

    def __init__(self, rpc_url: str = None, keypair_path: str = None):
        # Read from merged config (config.toml + ~/.titan/secrets.toml) if not explicitly provided
        if rpc_url is None or keypair_path is None:
            try:
                from titan_plugin.config_loader import load_titan_config
                net_cfg = load_titan_config().get("network", {})
                if rpc_url is None:
                    rpc_url = net_cfg.get("premium_rpc_url",
                                net_cfg.get("public_rpc_urls", ["https://api.mainnet-beta.solana.com"])[0])
                if keypair_path is None:
                    keypair_path = net_cfg.get("wallet_keypair_path", "data/titan_identity_keypair.json")
            except Exception:
                rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
                keypair_path = keypair_path or "data/titan_identity_keypair.json"
        self._rpc_url = rpc_url
        self._keypair_path = keypair_path
        self._inscription_count = 0

    @property
    def name(self) -> str:
        return "memo_inscribe"

    @property
    def description(self) -> str:
        return "Inscribe consciousness state hash on Solana blockchain"

    @property
    def capabilities(self) -> list[str]:
        return ["blockchain", "anchor", "inscription", "identity"]

    @property
    def resource_cost(self) -> str:
        return "low"  # ~0.000005 SOL per memo

    @property
    def latency(self) -> str:
        return "medium"  # 1-5s for RPC + confirmation

    @property
    def enriches(self) -> list[str]:
        return ["body"]  # Primarily body interoception (energy flow)

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        """Inscribe a memo on Solana.

        Params:
            memo_text: str — text to inscribe (auto-generated if not provided)
            epoch_id: int — current consciousness epoch
            state_hash: str — Trinity state hash (auto-computed if not provided)
            trinity_state: list — 132D state vector for hashing

        Returns standard helper result dict with balance feedback.
        """
        try:
            from titan_plugin.utils.solana_client import (
                build_memo_instruction, load_keypair_from_json,
            )
            from solders.transaction import Transaction
            from solders.message import Message as SolMessage
            from solana.rpc.api import Client as SolanaClient

            # Load keypair
            keypair = load_keypair_from_json(self._keypair_path)
            if not keypair:
                return {
                    "success": False,
                    "result": "",
                    "enrichment_data": {},
                    "error": "No keypair found",
                }

            # Build memo text
            epoch_id = params.get("epoch_id", 0)
            state_hash = params.get("state_hash", "")
            if not state_hash and params.get("trinity_state"):
                state_hash = hashlib.sha256(
                    json.dumps(params["trinity_state"]).encode()
                ).hexdigest()[:16]
            memo_text = params.get("memo_text",
                                   f"TITAN|e={epoch_id}|h={state_hash}")

            # Connect to Solana
            sol_client = SolanaClient(self._rpc_url)

            # Build memo instruction
            ix = build_memo_instruction(keypair.pubkey(), memo_text)
            if not ix:
                return {
                    "success": False,
                    "result": "",
                    "enrichment_data": {},
                    "error": "Failed to build memo instruction",
                }

            # Get recent blockhash
            bh_resp = sol_client.get_latest_blockhash()
            blockhash = bh_resp.value.blockhash

            # Build, sign, send transaction
            msg = SolMessage.new_with_blockhash([ix], keypair.pubkey(), blockhash)
            tx = Transaction.new_unsigned(msg)
            tx.sign([keypair], blockhash)
            result = sol_client.send_transaction(tx)
            tx_sig = str(result.value) if result.value else "?"

            # Read back balance for body feedback
            bal_resp = sol_client.get_balance(keypair.pubkey())
            balance = bal_resp.value / 1e9 if bal_resp.value else 0.0

            # Save anchor state for body_worker to read
            anchor_state = {
                "last_anchor_time": time.time(),
                "last_tx_sig": tx_sig,
                "last_epoch_id": epoch_id,
                "last_state_hash": state_hash,
                "sol_balance": balance,
                "success": True,
            }
            anchor_path = os.path.join("data", "anchor_state.json")
            try:
                with open(anchor_path) as f:
                    prev = json.load(f)
                anchor_state["anchor_count"] = prev.get("anchor_count", 0) + 1
            except Exception:
                anchor_state["anchor_count"] = 1
            with open(anchor_path, "w") as f:
                json.dump(anchor_state, f)

            self._inscription_count += 1

            logger.info("[MemoInscribe] TX=%s balance=%.4f SOL epoch=%d",
                        tx_sig[:20], balance, epoch_id)

            return {
                "success": True,
                "result": f"Inscribed epoch {epoch_id} on Solana (tx={tx_sig[:16]}..., balance={balance:.4f} SOL)",
                "balance": balance,
                "enrichment_data": {
                    "body": [0, 3, 4],
                    "boost": 0.04,
                },
                "error": None,
            }

        except Exception as e:
            logger.warning("[MemoInscribe] Inscription failed: %s", e)
            return {
                "success": False,
                "result": "",
                "enrichment_data": {},
                "error": str(e),
            }

    def status(self) -> str:
        """Check if Solana keypair is available."""
        if os.path.exists(self._keypair_path):
            return "available"
        return "unavailable"
