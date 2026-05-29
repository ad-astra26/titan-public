"""titan_hcl/logic/trinity_anchor.py — Trinity on-chain anchoring (Bridge to Solana).

rFP §3G Phase 10F — RESTORED dropped orchestration loop. The body was
``spirit_loop._maybe_anchor_trinity``; its invoker (the spirit_worker
consciousness-epoch loop) was deleted in the D8-3 spirit→cognitive migration
and never re-homed, so periodic on-chain trinity anchoring silently stopped
(zero invoker since; ``data/anchor_state.json`` going stale). Confirmed 100%
dropped (audit_phase10_relocation_liveness_findings) → restored per
``feedback_never_delete_live_logic_fix_dont_delete`` + Maker greenlight
(2026-05-28).

Homed under ``timechain_worker`` (owns on-chain ops + block counts) per Maker;
the worker calls ``maybe_anchor_trinity`` on a periodic SHM-fed tick. Logic is
UNCHANGED from the original (emergent curvature-EMA + TimeChain-block-delta
gating, circuit breaker, anchor_state.json persistence, tx-latency
instrumentation). The original ``send_queue``/``name`` params were unused and
have been dropped.

SOL-COST SAFETY: anchoring is gated by ``config["anchor_enabled"]`` (default
False) exactly as the original — restoring this loop does NOT spend SOL unless
anchoring is explicitly enabled in config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

from titan_hcl.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


def maybe_anchor_trinity(consciousness, config: dict,
                         body: list, mind: list, spirit: list) -> None:
    """
    Check if Trinity state should be anchored on-chain.

    Anchors when:
      - Curvature significantly exceeds its EMA trend (life-changing shift), AND
        enough new TimeChain blocks have sealed since the last anchor; OR
      - Very sparse state (density < 0.01) + enough new blocks (uncharted
        territory).

    Gated by ``config["anchor_enabled"]`` (default False) — no SOL is spent
    unless explicitly enabled.
    """
    if not consciousness or not consciousness.get("latest_epoch"):
        return

    if not config.get("anchor_enabled", False):
        return

    epoch = consciousness["latest_epoch"]
    epoch_id = epoch.get("epoch_id", 0)
    curvature = epoch.get("curvature", 0)
    density = epoch.get("density", 1.0)

    # ── Emergent anchoring: no hardcoded daily limits ──
    # Two signals must converge:
    #   1. Curvature must significantly exceed recent trend (adaptive EMA threshold)
    #   2. Enough new TimeChain blocks since last anchor (meaningful state change)
    # This lets Titan decide when to anchor based on his own internal dynamics.

    _anchor_path = os.path.join("data", "anchor_state.json")
    _prev_anchor = {}
    try:
        with open(_anchor_path) as _af:
            _prev_anchor = json.load(_af)
    except Exception as _swallow_exc:
        swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: with open(_anchor_path) as _af: _prev_anchor = json.load(...', _swallow_exc,
                     key='logic.trinity_anchor.maybe_anchor_trinity.line878', throttle=100)

    # Update curvature EMA (exponential moving average, α=0.02 for smooth tracking)
    _curvature_ema = _prev_anchor.get("curvature_ema", 2.0)
    _curvature_ema = 0.98 * _curvature_ema + 0.02 * curvature

    # Signal 1: Curvature significantly above trend (>30% above EMA = outlier moment)
    _curvature_significant = curvature > (_curvature_ema * 1.3)

    # Signal 2: Minimum TimeChain block delta since last anchor
    # ~20K blocks/day → 5000 blocks ≈ 6 hours → ~4 anchors/day naturally
    _min_tc_delta = config.get("mainnet_budget", {}).get("consciousness_anchor_min_tc_blocks", 5000)
    _last_anchor_tc = _prev_anchor.get("last_anchor_tc_blocks", 0)
    try:
        from titan_hcl.utils.db import safe_connect as _sc_tc
        _tc_db = _sc_tc("data/timechain/index.db")
        _current_tc_blocks = _tc_db.execute("SELECT COUNT(*) FROM block_index").fetchone()[0]
        _tc_db.close()
    except Exception:
        _current_tc_blocks = _last_anchor_tc  # Can't check — don't block on DB error
    _tc_delta = _current_tc_blocks - _last_anchor_tc
    _enough_new_state = _tc_delta >= _min_tc_delta

    # Persist EMA every 100 epochs (cheap write, keeps tracking accurate)
    if epoch_id % 100 == 0:
        try:
            _ema_state = _prev_anchor.copy()
            _ema_state["curvature_ema"] = _curvature_ema
            with open(_anchor_path, "w") as _af:
                json.dump(_ema_state, _af, indent=2)
        except Exception as _swallow_exc:
            swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: _ema_state = _prev_anchor.copy()', _swallow_exc,
                         key='logic.trinity_anchor.maybe_anchor_trinity.line909', throttle=100)

    should_anchor = False
    reason = ""

    if _curvature_significant and _enough_new_state:
        should_anchor = True
        reason = f"curvature={curvature:.3f}(ema={_curvature_ema:.3f})|tc_delta={_tc_delta}"
    elif _enough_new_state and density < 0.01 and epoch_id > 3:
        # Rare: very sparse state + enough blocks — exploring truly new territory
        should_anchor = True
        reason = f"sparse_exploration|density={density:.3f}|tc_delta={_tc_delta}"

    if should_anchor:
        # ── Circuit breaker: stop retrying after consecutive failures ──
        _consecutive_fails = _prev_anchor.get("consecutive_failures", 0)
        _last_fail_time = _prev_anchor.get("last_failure_time", 0)
        if _consecutive_fails >= 5 and (time.time() - _last_fail_time) < 3600:
            if epoch_id % 500 == 0:
                logger.info("[Anchor] Circuit breaker OPEN: %d consecutive failures, "
                            "cooldown %.0fm remaining",
                            _consecutive_fails, (3600 - (time.time() - _last_fail_time)) / 60)
            return

        # Build Trinity state hash
        trinity_state = body + mind + spirit
        state_hash = hashlib.sha256(json.dumps(trinity_state).encode()).hexdigest()[:16]

        logger.info("[TimeChain] ANCHOR: epoch=%d reason=%s hash=%s",
                     epoch_id, reason, state_hash)

        # Inscribe memo on Solana (mainnet) — bidirectional chain connection
        # Result feeds back to body senses via anchor_state.json
        try:
            from titan_hcl.utils.solana_client import build_memo_instruction, load_keypair_from_json

            _kp_path = config.get("wallet_keypair_path", "data/titan_identity_keypair.json")
            keypair = load_keypair_from_json(_kp_path)
            if keypair:
                # Include TimeChain Merkle root in memo (if available)
                _tc_merkle = ""
                _tc_height = 0
                try:
                    from titan_hcl.logic.timechain import TimeChain
                    # Read directly from index DB to avoid creating full instance
                    from titan_hcl.utils.db import safe_connect as _sc_tc2
                    _tc_idx = _sc_tc2("data/timechain/index.db")
                    _tc_cnt = _tc_idx.execute("SELECT COUNT(*) FROM block_index").fetchone()
                    _tc_height = _tc_cnt[0] if _tc_cnt else 0
                    # Compute merkle from genesis hash if available
                    _tc_gen_path = __import__("pathlib").Path("data/timechain/chain_main.bin")
                    if _tc_gen_path.exists() and _tc_gen_path.stat().st_size >= 128:
                        import hashlib as _tc_hl
                        with open(_tc_gen_path, "rb") as _tc_f:
                            _tc_merkle = _tc_hl.sha256(_tc_f.read(128)).hexdigest()[:16]
                    _tc_idx.close()
                except Exception as _swallow_exc:
                    swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: from titan_hcl.logic.timechain import TimeChain', _swallow_exc,
                                 key='logic.trinity_anchor.maybe_anchor_trinity.line966', throttle=100)
                memo_text = f"TITAN|e={epoch_id}|h={state_hash}|r={reason}"
                if _tc_merkle:
                    memo_text += f"|tc={_tc_merkle}|tb={_tc_height}"
                ix = build_memo_instruction(keypair.pubkey(), memo_text)
                if ix:
                    from solders.transaction import Transaction
                    from solders.message import Message as SolMessage
                    from solana.rpc.api import Client as SolanaClient

                    rpc_url = config.get("premium_rpc_url",
                              config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com"))
                    sol_client = SolanaClient(rpc_url)

                    # Get recent blockhash
                    bh_resp = sol_client.get_latest_blockhash()
                    blockhash = bh_resp.value.blockhash

                    # Build + sign + send transaction
                    msg = SolMessage.new_with_blockhash([ix], keypair.pubkey(), blockhash)
                    tx = Transaction.new_unsigned(msg)
                    tx.sign([keypair], blockhash)

                    # Phase 1 sensory wiring: instrument TX latency for
                    # outer_body[2] somatosensation composite. Try/except
                    # wrapper — instrumentation MUST NOT break anchor path.
                    _tx_t0 = time.monotonic()
                    result = sol_client.send_transaction(tx)
                    try:
                        from titan_hcl.logic.timechain_v2 import record_tx_latency
                        record_tx_latency(time.monotonic() - _tx_t0)
                    except Exception as _swallow_exc:
                        swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: from titan_hcl.logic.timechain_v2 import record_tx_lat...', _swallow_exc,
                                     key='logic.trinity_anchor.maybe_anchor_trinity.line998', throttle=100)
                    tx_sig = str(result.value) if result.value else "?"

                    # Read back balance for body feedback
                    _bal_resp = sol_client.get_balance(keypair.pubkey())
                    balance = _bal_resp.value / 1e9 if _bal_resp.value else 0.0
                    anchor_time = time.time()

                    # Save anchor state — includes emergent tracking (EMA, TC blocks)
                    _today_str = time.strftime("%Y-%m-%d")
                    _anchor_state = {
                        "last_anchor_time": anchor_time,
                        "last_tx_sig": tx_sig,
                        "last_epoch_id": epoch_id,
                        "last_state_hash": state_hash,
                        "sol_balance": balance,
                        "anchor_count": 0,
                        "success": True,
                        "consecutive_failures": 0,
                        "anchor_date": _today_str,
                        "today_count": 1,
                        "curvature_ema": _curvature_ema,
                        "last_anchor_tc_blocks": _current_tc_blocks,
                    }
                    # Read existing count + daily counter
                    try:
                        with open(_anchor_path) as _af:
                            _prev = json.load(_af)
                        _anchor_state["anchor_count"] = _prev.get("anchor_count", 0) + 1
                        if _prev.get("anchor_date") == _today_str:
                            _anchor_state["today_count"] = _prev.get("today_count", 0) + 1
                    except Exception:
                        _anchor_state["anchor_count"] = 1

                    with open(_anchor_path, "w") as _af:
                        json.dump(_anchor_state, _af, indent=2)

                    logger.info(
                        "[Anchor] Memo inscribed: tx=%s SOL=%.6f epoch=%d count=%d",
                        tx_sig[:16], balance, epoch_id, _anchor_state["anchor_count"])
        except ImportError:
            logger.debug("[Anchor] Solana SDK not available — skipping")
        except Exception as _ae:
            # Anchor failure must NOT crash the worker — track consecutive failures
            try:
                _fail_count = 0
                try:
                    with open(_anchor_path) as _af:
                        _prev = json.load(_af)
                    _fail_count = _prev.get("consecutive_failures", 0)
                except Exception as _swallow_exc:
                    swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: with open(_anchor_path) as _af: _prev = json.load(_af)', _swallow_exc,
                                 key='logic.trinity_anchor.maybe_anchor_trinity.line1049', throttle=100)
                _fail_count += 1
                _fail_state = {
                    "last_anchor_time": time.time(), "success": False,
                    "error": str(_ae), "last_epoch_id": epoch_id,
                    "consecutive_failures": _fail_count,
                    "last_failure_time": time.time(),
                }
                with open(_anchor_path, "w") as _af:
                    json.dump(_fail_state, _af, indent=2)
                if _fail_count <= 5:
                    logger.info("[Anchor] Inscription failed (%d/5): %s", _fail_count, _ae)
                elif _fail_count == 6:
                    logger.warning("[Anchor] Circuit breaker ENGAGED after 5 failures — "
                                   "pausing anchoring for 1 hour. Last error: %s", _ae)
            except Exception as _swallow_exc:
                swallow_warn('[logic.trinity_anchor] maybe_anchor_trinity: _fail_count = 0', _swallow_exc,
                             key='logic.trinity_anchor.maybe_anchor_trinity.line1065', throttle=100)
