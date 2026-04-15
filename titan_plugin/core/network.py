"""
core/network.py
Solana RPC client with fallback, priority fees, Jito bundles, and DI listener.
V2.0: Real Solana RPC via solana-py/solders. No more hardcoded balances.
"""
import asyncio
import json
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class HybridNetworkClient:
    """
    Client for Solana blockchain communications: balance queries, transaction
    sending with priority fees, Jito bundle atomicity, and Divine Inspiration
    WebSocket listener.
    """

    def __init__(self, mood_engine=None, config: dict = None):
        """
        Args:
            mood_engine: Optional MoodEngine reference for DI overrides.
            config: [network] section from config.toml.
        """
        config = config or {}
        self.rpc_urls = config.get(
            "public_rpc_urls",
            [
                "https://api.mainnet-beta.solana.com",
                "https://solana-api.projectserum.com",
            ],
        )
        self.premium_rpc = config.get("premium_rpc_url") or None
        self.economy_mode = False
        self.solana_network = config.get("solana_network", "mainnet-beta")

        # DI listener
        self.mood_engine = mood_engine
        self.MAKER_PUBKEY = config.get("maker_pubkey") or ""
        self._listener_task = None

        # Wallet keypair (loaded lazily)
        self._wallet_path = config.get("wallet_keypair_path", "./authority.json")
        self._keypair = None
        self._pubkey = None

        # Solana RPC client (lazy)
        self._rpc_client = None

        # Store full config for budget/vault access
        self._config = config

        # Mainnet budget enforcement (daily transaction caps)
        self._budget_tx_count = 0
        self._budget_sol_spent = 0.0
        self._budget_date = ""
        self._budget_config = {}  # Loaded on first use from parent config

    # -------------------------------------------------------------------------
    # Keypair Management
    # -------------------------------------------------------------------------
    def _load_keypair(self):
        """Load Solana keypair via the centralized solana_client facade."""
        if self._keypair is not None:
            return

        from titan_plugin.utils.solana_client import load_keypair_from_json

        kp = load_keypair_from_json(self._wallet_path)
        if kp is not None:
            self._keypair = kp
            self._pubkey = kp.pubkey()

    @property
    def pubkey(self):
        """Agent's Solana public key. Returns None if keypair not loaded."""
        self._load_keypair()
        return self._pubkey

    @property
    def keypair(self):
        """Agent's Solana keypair. Returns None if not loaded."""
        self._load_keypair()
        return self._keypair

    # -------------------------------------------------------------------------
    # RPC Client Management
    # -------------------------------------------------------------------------
    async def _get_rpc_client(self):
        """Get or create an async Solana RPC client with fallback logic."""
        if self._rpc_client is not None:
            return self._rpc_client

        from solana.rpc.async_api import AsyncClient

        # Try premium first, then public RPCs
        urls_to_try = []
        if self.premium_rpc:
            urls_to_try.append(self.premium_rpc)
        urls_to_try.extend(self.rpc_urls)

        for url in urls_to_try:
            try:
                client = AsyncClient(url)
                # Health check
                resp = await client.is_connected()
                if resp:
                    self._rpc_client = client
                    logger.info("[Network] Connected to RPC: %s", url)
                    return client
            except Exception as e:
                logger.debug("[Network] RPC %s unreachable: %s", url, e)
                continue

        logger.error("[Network] All RPC endpoints unreachable.")
        return None

    async def _rpc_call_with_fallback(self, method_name: str, *args, **kwargs):
        """Execute an RPC call with automatic fallback to next endpoint on failure."""
        from solana.rpc.async_api import AsyncClient

        urls_to_try = []
        if self.premium_rpc:
            urls_to_try.append(self.premium_rpc)
        urls_to_try.extend(self.rpc_urls)

        last_error = None
        for url in urls_to_try:
            try:
                client = AsyncClient(url)
                method = getattr(client, method_name)
                result = await method(*args, **kwargs)
                # Cache the working client
                self._rpc_client = client
                return result
            except Exception as e:
                last_error = e
                logger.debug("[Network] RPC %s failed for %s: %s", url, method_name, e)
                continue

        raise ConnectionError(
            f"All RPC endpoints failed for {method_name}: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Balance
    # -------------------------------------------------------------------------
    async def get_balance(self) -> float:
        """Fetch the current SOL balance of the agent's wallet from Solana RPC.

        Cached for 60s to avoid RPC rate limiting on devnet.
        """
        import time as _time
        _now = _time.time()
        _cached = getattr(self, '_balance_cache', None)
        _cached_ts = getattr(self, '_balance_cache_ts', 0.0)
        if _cached is not None and (_now - _cached_ts) < 60.0:
            return _cached

        self._load_keypair()
        if self._pubkey is None:
            logger.warning("[Network] No wallet loaded — returning 0 balance.")
            return 0.0

        try:
            resp = await self._rpc_call_with_fallback("get_balance", self._pubkey)
            # solana-py returns GetBalanceResp with .value in lamports
            lamports = resp.value
            sol = lamports / 1_000_000_000
            self._balance_cache = sol
            self._balance_cache_ts = _now
            return sol
        except Exception as e:
            logger.error("[Network] get_balance failed: %s", e)
            # Return stale cache if available, else 0
            if _cached is not None:
                return _cached
            return 0.0

    # -------------------------------------------------------------------------
    # Priority Fees
    # -------------------------------------------------------------------------
    async def get_priority_fee_estimate(self, accounts: list) -> float:
        """
        Get priority fee estimate. Uses getRecentPrioritizationFees RPC
        when available, otherwise returns conservative defaults.
        """
        if self.economy_mode:
            return 0.000001  # 1 microlamport — economy mode

        try:
            from solders.pubkey import Pubkey

            # Convert string accounts to Pubkey objects if needed
            pubkeys = []
            for acct in accounts:
                if isinstance(acct, str):
                    pubkeys.append(Pubkey.from_string(acct))
                else:
                    pubkeys.append(acct)

            resp = await self._rpc_call_with_fallback(
                "get_recent_prioritization_fees", pubkeys if pubkeys else None
            )

            if resp.value:
                # Take the median fee from recent slots
                fees = [f.prioritization_fee for f in resp.value if f.prioritization_fee > 0]
                if fees:
                    fees.sort()
                    median_fee = fees[len(fees) // 2]
                    # Convert from micro-lamports to SOL
                    return median_fee / 1_000_000_000
        except Exception as e:
            logger.debug("[Network] Priority fee estimate failed: %s — using default.", e)

        return 0.00005  # Default: 50k micro-lamports

    def set_economy_mode(self, enabled: bool):
        self.economy_mode = enabled

    # -------------------------------------------------------------------------
    # Transaction Sending
    # -------------------------------------------------------------------------
    async def send_sovereign_transaction(
        self, instructions: list, priority: str = "MEDIUM",
        extra_signers: list = None,
    ) -> Optional[str]:
        """
        Build, sign, and send a Solana transaction with priority fee and retry logic.

        Args:
            instructions: List of solders Instruction objects.
            priority: "HIGH", "MEDIUM", or "LOW" fee multiplier.
            extra_signers: Additional Keypair objects that must sign (e.g. NFT asset keypair).

        Returns:
            Transaction signature string, or None on failure.
        """
        self._load_keypair()
        if self._keypair is None:
            logger.error("[Network] Cannot send tx — no wallet loaded.")
            return None

        # ── Mainnet budget enforcement ──
        if self._check_budget_exceeded():
            logger.warning("[Network] Daily budget EXCEEDED — TX blocked. Reset at UTC midnight.")
            return None

        from solders.transaction import Transaction
        from solders.message import Message
        from solders.signature import Signature as SoldersSig
        from solana.rpc.commitment import Confirmed

        fee_estimate = await self.get_priority_fee_estimate([])
        multiplier = {"HIGH": 2.5, "MEDIUM": 1.0, "LOW": 0.5}.get(priority, 1.0)
        final_fee = fee_estimate * multiplier

        logger.info(
            "[Network] Sending tx, priority=%s, fee=%.9f SOL, ix_count=%d (budget: %d/%d TX today)",
            priority, final_fee, len(instructions),
            self._budget_tx_count, self._get_budget_max_tx(),
        )

        # Retry with exponential backoff
        for attempt in range(5):
            try:
                client = await self._get_rpc_client()
                if client is None:
                    raise ConnectionError("No RPC client available")

                # Get recent blockhash + last valid block height for confirmation
                blockhash_resp = await client.get_latest_blockhash()
                blockhash = blockhash_resp.value.blockhash
                last_valid_height = blockhash_resp.value.last_valid_block_height

                # Build and sign transaction
                msg = Message.new_with_blockhash(
                    instructions, self._pubkey, blockhash
                )
                tx = Transaction.new_unsigned(msg)
                signers = [self._keypair] + (extra_signers or [])
                tx.sign(signers, blockhash)

                # Send (skip preflight to avoid simulation mismatches with recent blockhash)
                from solana.rpc.types import TxOpts
                opts = TxOpts(skip_preflight=True, preflight_commitment=Confirmed)
                resp = await client.send_transaction(tx, opts=opts)
                sig_value = resp.value  # This is already a Signature object

                # Confirm — must pass a Signature object, not a string
                if isinstance(sig_value, str):
                    sig_obj = SoldersSig.from_string(sig_value)
                else:
                    sig_obj = sig_value

                await client.confirm_transaction(
                    sig_obj,
                    commitment=Confirmed,
                    last_valid_block_height=last_valid_height,
                )
                sig_str = str(sig_obj)
                logger.info(
                    "[Network] Tx confirmed on attempt %d: %s", attempt + 1, sig_str
                )
                # Track budget
                self._budget_tx_count += 1
                self._budget_sol_spent += final_fee
                return sig_str

            except Exception as e:
                logger.warning(
                    "[Network] Tx attempt %d failed: %s. Retrying...",
                    attempt + 1,
                    e,
                )
                self._rpc_client = None  # Force reconnect on retry
                await asyncio.sleep(min(2**attempt, 16))

        logger.error("[Network] Transaction failed after 5 attempts.")
        return None

    # ── Mainnet Budget Enforcement ─────────────────────────────────────
    def _check_budget_exceeded(self) -> bool:
        """Check if daily TX budget is exceeded. Resets at UTC midnight."""
        import time
        today = time.strftime("%Y-%m-%d", time.gmtime())

        # Reset counter at midnight UTC
        if today != self._budget_date:
            if self._budget_tx_count > 0:
                logger.info("[Network] Budget reset: %d TX, %.6f SOL spent yesterday",
                            self._budget_tx_count, self._budget_sol_spent)
            self._budget_tx_count = 0
            self._budget_sol_spent = 0.0
            self._budget_date = today

        # Load budget config (from parent config's [mainnet_budget] section)
        budget = self._get_budget_config()
        if not budget.get("enabled", False):
            return False  # Budget enforcement disabled

        max_tx = budget.get("max_daily_transactions", 40)
        max_sol = budget.get("max_daily_sol_spend", 0.01)

        if self._budget_tx_count >= max_tx:
            return True
        if self._budget_sol_spent >= max_sol:
            return True

        # Warn at 80%
        if self._budget_tx_count >= max_tx * 0.8:
            logger.warning("[Network] Budget warning: %d/%d TX used (80%%+)",
                           self._budget_tx_count, max_tx)

        return False

    def _get_budget_config(self) -> dict:
        """Get [mainnet_budget] config section."""
        if not self._budget_config and self._config:
            self._budget_config = self._config.get("mainnet_budget", {})
        return self._budget_config

    def _get_budget_max_tx(self) -> int:
        """Get max daily TX from budget config."""
        return self._get_budget_config().get("max_daily_transactions", 40)

    async def send_jito_bundle(self, transactions: list) -> bool:
        """
        Jito Bundle atomicity: submit multiple transactions as an atomic bundle.
        Requires a Jito-compatible RPC endpoint (e.g., block engine).
        """
        # Jito bundle endpoint (mainnet only)
        jito_url = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"

        try:
            import httpx

            # Serialize transactions to base64
            encoded_txs = []
            for tx in transactions:
                import base64

                raw = bytes(tx)
                encoded_txs.append(base64.b64encode(raw).decode("ascii"))

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [encoded_txs],
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(jito_url, json=payload)
                resp.raise_for_status()
                result = resp.json()

            if "result" in result:
                logger.info("[Network] Jito bundle landed: %s", result["result"])
                return True
            else:
                logger.error("[Network] Jito bundle rejected: %s", result.get("error"))
                return False

        except Exception as e:
            logger.error("[Network] Jito bundle failed: %s", e)
            return False

    # -------------------------------------------------------------------------
    # Account Info & Memo TX Retrieval (Resurrection SDK)
    # -------------------------------------------------------------------------
    async def get_account_info(self, pubkey_str: str) -> dict | None:
        """
        Fetch account info for a given public key via getAccountInfo RPC.
        Used by the Resurrection Protocol to query ZK-compressed state.

        Args:
            pubkey_str: Base58-encoded Solana public key.

        Returns:
            Decoded account data dict, or None on failure.
        """
        try:
            from titan_plugin.utils.solana_client import parse_pubkey, decode_zk_account_data

            pubkey = parse_pubkey(pubkey_str)
            if pubkey is None:
                logger.error("[Network] Invalid pubkey for account info: %s", pubkey_str)
                return None

            resp = await self._rpc_call_with_fallback("get_account_info", pubkey)

            if resp.value is None:
                logger.info("[Network] Account %s not found on-chain.", pubkey_str)
                return None

            # Decode raw account data via ZK schema
            raw_data = resp.value.data
            if isinstance(raw_data, (bytes, bytearray)):
                decoded = decode_zk_account_data(raw_data)
                return decoded

            logger.debug("[Network] Account data format not bytes: %s", type(raw_data))
            return {"raw": str(raw_data)}

        except Exception as e:
            logger.error("[Network] get_account_info failed: %s", e)
            return None

    async def get_raw_account_data(self, pubkey_str: str) -> bytes | None:
        """
        Fetch raw account bytes for a given public key via getAccountInfo RPC.
        Returns the raw binary data without any schema decoding — suitable for
        regular Anchor accounts (e.g. VaultState) that use binary struct layout.

        Args:
            pubkey_str: Base58-encoded Solana public key.

        Returns:
            Raw account bytes, or None if not found / on failure.
        """
        try:
            from titan_plugin.utils.solana_client import parse_pubkey

            pubkey = parse_pubkey(pubkey_str)
            if pubkey is None:
                logger.error("[Network] Invalid pubkey for raw account data: %s", pubkey_str)
                return None

            resp = await self._rpc_call_with_fallback("get_account_info", pubkey)

            if resp.value is None:
                logger.debug("[Network] Account %s not found on-chain.", pubkey_str)
                return None

            raw_data = resp.value.data
            if isinstance(raw_data, (bytes, bytearray)):
                return bytes(raw_data)

            logger.debug("[Network] Account data format not bytes: %s", type(raw_data))
            return None

        except Exception as e:
            logger.error("[Network] get_raw_account_data failed: %s", e)
            return None

    async def fetch_memo_transaction(self, tx_signature: str) -> str | None:
        """
        Fetch a confirmed transaction and extract memo data from it.
        Used to retrieve the Genesis Memo containing encrypted Shard 3.

        Args:
            tx_signature: Base58-encoded transaction signature.

        Returns:
            Memo text string, or None if not found.
        """
        try:
            from solders.signature import Signature

            sig = Signature.from_string(tx_signature)
            resp = await self._rpc_call_with_fallback(
                "get_transaction", sig, encoding="jsonParsed"
            )

            if resp.value is None:
                logger.warning("[Network] Transaction %s not found.", tx_signature)
                return None

            # Navigate the parsed transaction to find memo instructions
            tx_data = resp.value
            try:
                # solana-py returns parsed JSON; navigate to instructions
                meta = tx_data.transaction.meta
                inner = meta.inner_instructions if meta else []
                msg = tx_data.transaction.transaction.message

                for ix in msg.instructions:
                    program_id = str(ix.program_id) if hasattr(ix, "program_id") else ""
                    if "Memo" in program_id or "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr" in program_id:
                        # Memo data is in the instruction data field
                        if hasattr(ix, "data"):
                            memo_bytes = ix.data
                            if isinstance(memo_bytes, (bytes, bytearray)):
                                return memo_bytes.decode("utf-8", errors="replace")
                            return str(memo_bytes)
            except AttributeError:
                # Try alternate parsed format
                pass

            # Fallback: search log messages for memo content
            if hasattr(tx_data, "transaction") and hasattr(tx_data.transaction, "meta"):
                logs = tx_data.transaction.meta.log_messages or []
                for log in logs:
                    if "Memo" in log and "TITAN_GENESIS" in log:
                        # Extract memo content from log line
                        parts = log.split('"')
                        if len(parts) >= 2:
                            return parts[1]

            logger.warning("[Network] No memo found in transaction %s.", tx_signature)
            return None

        except Exception as e:
            logger.error("[Network] fetch_memo_transaction failed: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Divine Inspiration Background Listener
    # -------------------------------------------------------------------------
    async def start_divine_listener(self):
        """
        Spawn background listener for Divine Inspiration (Maker memo transactions).
        Uses Helius webhook if premium RPC is Helius, otherwise WSS fallback.
        """
        if self.premium_rpc and "helius" in self.premium_rpc.lower():
            logger.info("[Network] DI Listener: Helius Webhook strategy enabled.")
        else:
            logger.info("[Network] DI Listener: WSS fallback mode.")
            self._listener_task = asyncio.create_task(self._websocket_listener_loop())

    async def handle_webhook_payload(self, payload: dict):
        """Called externally if using Helius Webhook for DI detection."""
        try:
            memo = payload[0].get("description", "")
            if "DI:" in memo and payload[0].get("feePayer") == self.MAKER_PUBKEY:
                await self._trigger_override()
        except Exception as e:
            logger.error("[Network] Error parsing DI webhook: %s", e)

    async def _websocket_listener_loop(self):
        """Listen for 'DI:' memo signature in real-time via Solana WSS."""
        try:
            import websockets
        except ImportError:
            logger.warning("[Network] websockets not installed — DI listener disabled.")
            return

        # Derive WSS URL from RPC
        wss_url = "wss://api.mainnet-beta.solana.com"
        if self.premium_rpc:
            wss_url = self.premium_rpc.replace("https://", "wss://").replace(
                "http://", "ws://"
            )

        memo_program_id = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
        backoff = 1

        while True:
            try:
                logger.info("[Network] Connecting to WSS: %s for DI monitoring...", wss_url)
                async with websockets.connect(wss_url) as ws:
                    backoff = 1

                    sub_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [self.MAKER_PUBKEY, memo_program_id]},
                            {"commitment": "confirmed"},
                        ],
                    }
                    await ws.send(json.dumps(sub_payload))

                    while True:
                        msg_raw = await ws.recv()
                        msg = json.loads(msg_raw)

                        logs = (
                            msg.get("params", {})
                            .get("result", {})
                            .get("value", {})
                            .get("logs", [])
                        )
                        for log_line in logs:
                            if "DI:" in log_line:
                                await self._trigger_override()
                                break

            except Exception as e:
                logger.warning(
                    "[Network] WSS error: %s. Reconnecting in %ds...", e, backoff
                )

            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)

    async def _trigger_override(self):
        """Invoked when a Divine Inspiration transaction is confirmed."""
        logger.info("[Network] Divine Inspiration Received!")
        if self.mood_engine:
            self.mood_engine.force_zen()

        titan_path = "titan.md"
        log_entry = f"\n- [{time.strftime('%Y-%m-%d %H:%M:%S')}] DIVINE INSPIRATION ACCEPTED: State Shifted to ZEN."
        try:
            with open(titan_path, "a") as f:
                f.write(log_entry)
        except Exception as e:
            logger.error("[Network] Failed to write DI reflection to titan.md: %s", e)
