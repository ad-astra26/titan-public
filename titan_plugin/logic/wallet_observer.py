"""
titan_plugin/logic/wallet_observer.py — Observe incoming wallet transactions.

Polls Titan's Solana wallet for incoming transactions with memo data.
Classifies as DI:/I:/DONATION and routes through neurochemistry via
EXTERNAL_INTENT bus message.

Polling interval: 30s (configurable)
RPC method: getSignaturesForAddress (recent transactions)

Architecture:
  WalletObserver.poll() → list[ParsedMemo]
  Spirit worker calls poll() in periodic loop → publishes EXTERNAL_INTENT
"""
import base64
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class WalletObserver:
    """Observe incoming Solana wallet transactions for DI:/I: memos."""

    def __init__(
        self,
        titan_pubkey: str,
        maker_pubkey: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        poll_interval: float = 30.0,
    ):
        self._titan_pubkey = titan_pubkey
        self._maker_pubkey = maker_pubkey
        self._rpc_url = rpc_url
        self._poll_interval = poll_interval

        # Track last seen signature to avoid reprocessing
        self._last_signature: Optional[str] = None
        self._last_poll: float = 0.0
        self._total_processed: int = 0
        self._total_di: int = 0
        self._total_i: int = 0
        self._total_donations: int = 0

        logger.info("[WalletObserver] Initialized for %s... (maker=%s..., poll=%ds)",
                    titan_pubkey[:12], maker_pubkey[:12], int(poll_interval))

    def should_poll(self) -> bool:
        """Check if enough time has passed since last poll."""
        return time.time() - self._last_poll >= self._poll_interval

    async def poll(self) -> list:
        """Poll for new incoming transactions.

        Returns list of ParsedMemo objects for new transactions since last poll.
        """
        import httpx
        from titan_plugin.logic.memo_parser import parse_memo

        self._last_poll = time.time()
        new_memos = []

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Get recent signatures for our wallet
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [
                        self._titan_pubkey,
                        {"limit": 10, "commitment": "confirmed"},
                    ],
                }

                resp = await client.post(self._rpc_url, json=payload)
                result = resp.json().get("result", [])

                if not result:
                    return []

                # Process signatures newest-first, stop at last_signature
                for sig_info in result:
                    sig = sig_info.get("signature", "")
                    if sig == self._last_signature:
                        break  # Already processed

                    # Get transaction details
                    memo_data, sender, sol_amount = await self._get_tx_details(
                        client, sig)

                    if sender and sender != self._titan_pubkey:
                        # Incoming transaction (not our own)
                        parsed = parse_memo(
                            memo_data=memo_data or "",
                            sender_pubkey=sender,
                            maker_pubkey=self._maker_pubkey,
                            sol_amount=sol_amount,
                            tx_signature=sig,
                        )

                        if parsed.memo_type == "DI" or parsed.memo_type == "DI_DIRECTIVE":
                            self._total_di += 1
                        elif parsed.memo_type == "I":
                            self._total_i += 1
                        elif parsed.memo_type == "DONATION":
                            self._total_donations += 1

                        self._total_processed += 1
                        new_memos.append(parsed)

                        logger.info(
                            "[WalletObserver] %s from %s... (%s): %s",
                            parsed.memo_type,
                            sender[:12],
                            "MAKER" if parsed.is_maker else "external",
                            parsed.content[:60] if parsed.content else f"{sol_amount:.4f} SOL")

                # Update last seen signature
                if result:
                    self._last_signature = result[0].get("signature")

        except Exception as e:
            logger.debug("[WalletObserver] Poll error: %s", e)

        return new_memos

    async def _get_tx_details(
        self, client, signature: str
    ) -> tuple[Optional[str], Optional[str], float]:
        """Get memo data, sender, and SOL amount from a transaction.

        Returns (memo_data, sender_pubkey, sol_amount).
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
                ],
            }
            resp = await client.post(self._rpc_url, json=payload)
            tx = resp.json().get("result")
            if not tx:
                return None, None, 0.0

            # Extract memo from log messages
            memo_data = None
            log_messages = tx.get("meta", {}).get("logMessages", [])
            for log in log_messages:
                if "Program log: Memo" in log or "Program data:" in log:
                    # Extract memo text after "Memo (len X): "
                    if "Memo (len" in log:
                        idx = log.index("): ") + 3
                        memo_data = log[idx:]
                        break

            # Extract sender (first signer that isn't us)
            account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
            sender = None
            for ak in account_keys:
                key = ak.get("pubkey", ak) if isinstance(ak, dict) else str(ak)
                signer = ak.get("signer", False) if isinstance(ak, dict) else False
                if signer and key != self._titan_pubkey:
                    sender = key
                    break

            # If no external signer found, check pre/post balances for SOL transfer
            sol_amount = 0.0
            pre_balances = tx.get("meta", {}).get("preBalances", [])
            post_balances = tx.get("meta", {}).get("postBalances", [])
            # Find our account index
            for i, ak in enumerate(account_keys):
                key = ak.get("pubkey", ak) if isinstance(ak, dict) else str(ak)
                if key == self._titan_pubkey and i < len(pre_balances) and i < len(post_balances):
                    diff = (post_balances[i] - pre_balances[i]) / 1e9  # lamports → SOL
                    if diff > 0:
                        sol_amount = diff
                    break

            return memo_data, sender, sol_amount

        except Exception as e:
            logger.debug("[WalletObserver] TX detail error for %s: %s", signature[:12], e)
            return None, None, 0.0

    def get_stats(self) -> dict:
        return {
            "titan_pubkey": self._titan_pubkey[:12] + "...",
            "maker_pubkey": self._maker_pubkey[:12] + "...",
            "total_processed": self._total_processed,
            "total_di": self._total_di,
            "total_i": self._total_i,
            "total_donations": self._total_donations,
            "last_poll": self._last_poll,
            "poll_interval": self._poll_interval,
        }
