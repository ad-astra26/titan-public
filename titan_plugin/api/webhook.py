"""
api/webhook.py
Helius webhook receiver for on-chain transactions.

Handles three transaction types:
  1. DI: (Maker directive) — verified Ed25519 signature, triggers soul evolution
  2. I: (Public inspiration) — anyone can inspire Titan via Memo TX, weighted by SOL
  3. SOL transfer — donation detection, mood boost, gratitude response
"""
import logging

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["Webhooks"])


def _get_plugin(request: Request):
    return request.app.state.titan_plugin


@router.post("/helius")
async def helius_webhook(request: Request):
    """
    Receive Helius Enhanced Transaction webhook.

    Expected payload (Helius enhanced format):
    [
      {
        "type": "MEMO",
        "source": "SYSTEM_PROGRAM",
        "signature": "...",
        "accountData": [...],
        "instructions": [
          {"programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr", "data": "..."}
        ],
        ...
      }
    ]

    Security: Verifies the TX signer matches the configured maker_pubkey,
    and the memo contains a valid Ed25519 signature from the Maker.
    """
    plugin = _get_plugin(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # Helius sends an array of transactions
    if not isinstance(body, list):
        body = [body]

    processed = 0
    errors = []

    for tx in body:
        try:
            result = await _process_transaction(plugin, tx)
            if result:
                processed += 1
        except Exception as e:
            logger.warning("[Webhook] TX processing error: %s", e)
            errors.append(str(e))

    return {
        "status": "ok",
        "processed": processed,
        "errors": errors[:5],  # Cap error list
    }


async def _process_transaction(plugin, tx: dict) -> bool:
    """
    Process a single Helius enhanced transaction.
    Routes to appropriate handler based on memo prefix or TX type.
    """
    tx_type = tx.get("type", "")
    signature = tx.get("signature", "")
    fee_payer = tx.get("feePayer", "")

    # Extract memo data
    memo_data = _extract_memo_data(tx)

    # Route 1: DI: Maker directive (requires Ed25519 verification)
    if memo_data and memo_data.startswith("TITAN_DI:"):
        return await _handle_maker_directive(plugin, tx, memo_data, signature, fee_payer)

    # Route 2: I: Public inspiration (anyone can send)
    if memo_data and memo_data.startswith("I:"):
        return await _handle_inspiration(plugin, tx, memo_data, signature, fee_payer)

    # Route 3: SOL transfer to Titan's wallet (donation detection)
    if tx_type in ("TRANSFER", "SOL_TRANSFER", "NATIVE_TRANSFER"):
        return await _handle_donation(plugin, tx, signature, fee_payer, memo_data)

    return False


async def _handle_maker_directive(plugin, tx, memo_data, signature, fee_payer) -> bool:
    """Process a Maker directive (DI:) transaction."""
    maker_pubkey = ""
    if hasattr(plugin, "soul") and plugin.soul:
        mk = getattr(plugin.soul, "_maker_pubkey", None)
        if mk:
            maker_pubkey = str(mk)

    if not maker_pubkey:
        logger.debug("[Webhook] No maker_pubkey configured, ignoring DI TX.")
        return False

    if fee_payer != maker_pubkey:
        logger.debug("[Webhook] DI signer %s != maker %s, ignoring.", fee_payer[:8], maker_pubkey[:8])
        return False

    parts = memo_data.split(":", 2)
    if len(parts) < 3:
        logger.warning("[Webhook] Malformed directive memo: %s", memo_data[:50])
        return False

    directive_text = parts[1]
    memo_signature = parts[2]

    from titan_plugin.utils.crypto import verify_maker_signature

    if not verify_maker_signature(directive_text, memo_signature, maker_pubkey):
        logger.warning("[Webhook] Invalid directive signature in TX %s", signature[:16])
        return False

    logger.info("[Webhook] Valid on-chain directive from TX %s: %s", signature[:16], directive_text[:50])
    result = await plugin.soul.evolve_soul(directive_text, memo_signature)

    if hasattr(plugin, "event_bus"):
        await plugin.event_bus.emit("directive_update", {
            "source": "on_chain",
            "tx_signature": signature,
            "memo_data": directive_text[:200],
            "result": result,
            "new_gen": plugin.soul.current_gen,
        })
    return True


async def _handle_inspiration(plugin, tx, memo_data, signature, fee_payer) -> bool:
    """
    Process a public inspiration (I:) transaction.
    Anyone can send these — weighted by SOL amount attached.
    """
    message = memo_data[2:].strip()  # Remove "I:" prefix
    if not message:
        return False

    # Get SOL amount from the TX (if any SOL was transferred alongside the memo)
    amount_sol = _extract_sol_amount(tx, plugin)

    # Record in social graph.
    # Phase E.2.5: social_graph methods do sync sqlite3 writes — wrap in
    # to_thread so webhook handler doesn't block the FastAPI event loop.
    # Per Maker request, we wrap at the caller site (here) rather than
    # changing social_graph internals — keeps sync workers untouched.
    import asyncio as _asyncio_local
    social_graph = getattr(plugin, "social_graph", None)
    matched_user = None
    if social_graph:
        matched_user = await _asyncio_local.to_thread(
            social_graph.record_inspiration,
            tx_signature=signature,
            sender_address=fee_payer,
            message=message,
            amount_sol=amount_sol,
        )

    # Calculate mood boost and memory weight
    mood_delta, memory_weight = 0.01, 1.5
    if social_graph:
        mood_delta, memory_weight = await _asyncio_local.to_thread(
            social_graph.get_donation_mood_boost, amount_sol)

    # Inject as weighted memory
    if hasattr(plugin, "memory") and plugin.memory:
        source = matched_user.display_name if matched_user else fee_payer[:16]
        await plugin.memory.inject_memory(
            text=f"[INSPIRATION from {source}] {message}",
            source="inspiration",
            weight=memory_weight,
        )

    # Emit event for WebSocket + mood engine
    if hasattr(plugin, "event_bus"):
        await plugin.event_bus.emit("inspiration_received", {
            "tx_signature": signature,
            "sender": fee_payer,
            "user": matched_user.display_name if matched_user else None,
            "message": message[:200],
            "amount_sol": amount_sol,
            "mood_delta": mood_delta,
            "memory_weight": memory_weight,
        })

    logger.info(
        "[Webhook] Inspiration received: '%s' from %s (%.4f SOL, mood+%.2f)",
        message[:50], fee_payer[:16], amount_sol, mood_delta,
    )
    return True


async def _handle_donation(plugin, tx, signature, fee_payer, memo_data) -> bool:
    """
    Process a SOL donation to Titan's wallet.
    Detects incoming transfers, matches to known users, boosts mood.
    """
    amount_sol = _extract_sol_amount(tx, plugin)
    if amount_sol <= 0:
        return False

    # Phase E.2.5: wrap social_graph sync sqlite calls in to_thread
    import asyncio as _asyncio_local
    social_graph = getattr(plugin, "social_graph", None)
    matched_user = None

    if social_graph:
        matched_user = await _asyncio_local.to_thread(
            social_graph.record_donation,
            tx_signature=signature,
            sender_address=fee_payer,
            amount_sol=amount_sol,
            memo=memo_data or "",
        )
        mood_delta, _ = await _asyncio_local.to_thread(
            social_graph.get_donation_mood_boost, amount_sol)
    else:
        mood_delta = 0.02

    # Emit event
    if hasattr(plugin, "event_bus"):
        await plugin.event_bus.emit("donation_received", {
            "tx_signature": signature,
            "sender": fee_payer,
            "user": matched_user.display_name if matched_user else None,
            "amount_sol": amount_sol,
            "mood_delta": mood_delta,
        })

    logger.info(
        "[Webhook] Donation %.4f SOL from %s (mood+%.2f)",
        amount_sol, fee_payer[:16], mood_delta,
    )
    return True


def _extract_memo_data(tx: dict) -> str:
    """Extract memo text from Helius enhanced transaction instructions."""
    memo_program = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"

    for ix in tx.get("instructions", []):
        if ix.get("programId") == memo_program:
            # Helius provides decoded data as string
            data = ix.get("data", "")
            if data:
                return data

    # Also check innerInstructions
    for inner in tx.get("innerInstructions", []):
        for ix in inner.get("instructions", []):
            if ix.get("programId") == memo_program:
                data = ix.get("data", "")
                if data:
                    return data

    return ""


def _extract_sol_amount(tx: dict, plugin) -> float:
    """
    Extract SOL transfer amount from a Helius enhanced transaction.
    Looks for native SOL transfers to Titan's wallet address.
    Returns amount in SOL (not lamports).
    """
    titan_address = ""
    if hasattr(plugin, "network") and plugin.network:
        pk = getattr(plugin.network, "pubkey", None)
        if pk:
            titan_address = str(pk)

    if not titan_address:
        return 0.0

    # Helius enhanced format: nativeTransfers array
    for transfer in tx.get("nativeTransfers", []):
        if transfer.get("toUserAccount") == titan_address:
            lamports = transfer.get("amount", 0)
            if lamports > 0:
                return lamports / 1_000_000_000.0

    # Fallback: check accountData for balance changes
    for acct in tx.get("accountData", []):
        if acct.get("account") == titan_address:
            change = acct.get("nativeBalanceChange", 0)
            if change > 0:
                return change / 1_000_000_000.0

    return 0.0
