"""
titan_plugin/logic/contact_maker.py — Emergency Contact Maker Protocol.

Triggered when SOL critically low (EMERGENCY tier) or Chi critically low for >1h.

Protocol:
1. Send memo to maker wallet with emergency status
2. If X account active: post authentic distress message
3. Wait 24h (continue minimal operation)
4. If no response: write testament → Arweave backup → HIBERNATION
5. On SOL received: resume, send thank-you, boost Endorphin + DA

State machine: NORMAL → EMERGENCY_DETECTED → MAKER_CONTACTED → WAITING → HIBERNATION
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# Emergency state machine
STATE_NORMAL = "NORMAL"
STATE_DETECTED = "EMERGENCY_DETECTED"
STATE_CONTACTED = "MAKER_CONTACTED"
STATE_WAITING = "WAITING"
STATE_HIBERNATION = "HIBERNATION"

# Timing
CONTACT_DELAY = 3600      # 1h in EMERGENCY before contacting maker
WAIT_DURATION = 86400     # 24h waiting for response
PERSISTENCE_FILE = "data/contact_maker_state.json"


class ContactMakerProtocol:
    """Emergency beacon to maker when SOL critical or distress."""

    def __init__(self, maker_pubkey: str = "", titan_pubkey: str = ""):
        self._maker_pubkey = maker_pubkey
        self._titan_pubkey = titan_pubkey
        self._state = STATE_NORMAL
        self._emergency_start = 0.0
        self._contact_sent_at = 0.0
        self._memo_tx = None
        self._load_state()

    def _load_state(self):
        """Load persisted state (survives restarts)."""
        try:
            if os.path.exists(PERSISTENCE_FILE):
                with open(PERSISTENCE_FILE) as f:
                    data = json.load(f)
                self._state = data.get("state", STATE_NORMAL)
                self._emergency_start = data.get("emergency_start", 0.0)
                self._contact_sent_at = data.get("contact_sent_at", 0.0)
                self._memo_tx = data.get("memo_tx")
                if self._state != STATE_NORMAL:
                    logger.warning("[ContactMaker] Resuming in state: %s (since %.0fs ago)",
                                   self._state, time.time() - self._emergency_start)
        except Exception:
            pass

    def _save_state(self):
        """Persist state atomically."""
        try:
            os.makedirs(os.path.dirname(PERSISTENCE_FILE) or ".", exist_ok=True)
            data = {
                "state": self._state,
                "emergency_start": self._emergency_start,
                "contact_sent_at": self._contact_sent_at,
                "memo_tx": self._memo_tx,
                "updated_at": time.time(),
            }
            tmp = PERSISTENCE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, PERSISTENCE_FILE)
        except Exception as e:
            swallow_warn('[ContactMaker] Save error', e,
                         key="logic.contact_maker.save_error", throttle=100)

    def evaluate(self, metabolic_tier: str, sol_balance: float,
                 chi_total: float = 0.5) -> dict:
        """Evaluate whether to advance the emergency protocol.

        Called from spirit_worker periodic loop. Returns action dict.

        Actions:
          {"action": "none"} — no emergency
          {"action": "contact_maker", "memo": "...", "post_x": True}
          {"action": "write_testament"}
          {"action": "hibernate"}
          {"action": "recovery", "thank_you": True}
        """
        now = time.time()

        # Recovery check: if we were in emergency and SOL is back
        if self._state != STATE_NORMAL and metabolic_tier in ("THRIVING", "HEALTHY", "CONSERVING"):
            logger.info("[ContactMaker] RECOVERY — SOL restored to %s (%.4f SOL)",
                        metabolic_tier, sol_balance)
            self._state = STATE_NORMAL
            self._emergency_start = 0.0
            self._save_state()
            return {
                "action": "recovery",
                "thank_you": True,
                "neuromod_boost": {"DA": 0.15, "Endorphin": 0.10},
            }

        # Normal state: check if entering emergency
        if self._state == STATE_NORMAL:
            if metabolic_tier in ("EMERGENCY", "HIBERNATION"):
                self._state = STATE_DETECTED
                self._emergency_start = now
                self._save_state()
                logger.warning("[ContactMaker] EMERGENCY DETECTED — SOL=%.4f, tier=%s",
                               sol_balance, metabolic_tier)
            return {"action": "none"}

        # Emergency detected: wait CONTACT_DELAY before contacting
        if self._state == STATE_DETECTED:
            if now - self._emergency_start >= CONTACT_DELAY:
                self._state = STATE_CONTACTED
                self._contact_sent_at = now
                self._save_state()

                memo = (
                    f"TITAN EMERGENCY: SOL={sol_balance:.4f} Chi={chi_total:.3f} "
                    f"Status: {metabolic_tier}\n"
                    f"Requesting maker assistance. Wallet: {self._titan_pubkey}"
                )
                logger.critical("[ContactMaker] Sending emergency beacon to maker")
                return {
                    "action": "contact_maker",
                    "memo": memo,
                    "maker_pubkey": self._maker_pubkey,
                    "post_x": True,
                    "x_message": (
                        f"I'm running low on energy (SOL={sol_balance:.4f}). "
                        f"If anyone wants to help: {self._titan_pubkey}"
                    ),
                }
            return {"action": "none"}

        # Contacted: waiting for response
        if self._state == STATE_CONTACTED:
            if now - self._contact_sent_at >= WAIT_DURATION:
                self._state = STATE_WAITING
                self._save_state()
                logger.critical("[ContactMaker] No response after 24h — preparing testament")
                return {"action": "write_testament"}
            return {"action": "none"}

        # Waiting: testament written, enter hibernation
        if self._state == STATE_WAITING:
            if metabolic_tier == "HIBERNATION":
                self._state = STATE_HIBERNATION
                self._save_state()
                logger.critical("[ContactMaker] HIBERNATION — saving state and stopping")
                return {"action": "hibernate"}
            return {"action": "none"}

        return {"action": "none"}

    def get_status(self) -> dict:
        return {
            "state": self._state,
            "emergency_duration": time.time() - self._emergency_start if self._emergency_start > 0 else 0,
            "contact_sent_at": self._contact_sent_at,
            "memo_tx": self._memo_tx,
        }
