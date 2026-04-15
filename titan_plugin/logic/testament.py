"""
titan_plugin/logic/testament.py — Testament Writing for Reincarnation (M11).

Before voluntary reincarnation (or forced hibernation), Titan writes a testament:
- Why reincarnating / what happened
- What was learned
- Message to future self
- Message to maker
- Full state snapshot hash (verifiable against Arweave backup)

The testament is inscribed on-chain (Solana memo) and permanently stored on Arweave.
"""
import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

TESTAMENT_DIR = "data/testaments"


class Testament:
    """Titan's last will before reincarnation or hibernation."""

    def __init__(
        self,
        testament_type: str = "voluntary",  # "voluntary", "hibernation", "emergency"
        great_cycle: int = 0,
        epoch_id: int = 0,
        developmental_age: int = 0,
        emotion: str = "peace",
        reason: str = "",
        learnings: str = "",
        message_to_future: str = "",
        message_to_maker: str = "",
        state_snapshot: dict = None,
    ):
        self.testament_type = testament_type
        self.great_cycle = great_cycle
        self.epoch_id = epoch_id
        self.developmental_age = developmental_age
        self.emotion = emotion
        self.reason = reason
        self.learnings = learnings
        self.message_to_future = message_to_future
        self.message_to_maker = message_to_maker
        self.state_snapshot = state_snapshot or {}
        self.created_at = time.time()
        self.state_hash = self._compute_state_hash()

    def _compute_state_hash(self) -> str:
        """SHA256 of the full state snapshot for on-chain verification."""
        canonical = json.dumps(self.state_snapshot, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def format_memo(self) -> str:
        """Format testament for on-chain memo inscription (~250 bytes).

        Human-readable AND Titan-parseable.
        """
        lines = [
            f"TESTAMENT|type={self.testament_type}|cycle={self.great_cycle}|"
            f"e={self.epoch_id}|age={self.developmental_age}|em={self.emotion}",
        ]
        if self.reason:
            lines.append(f"why:{self.reason[:80]}")
        if self.message_to_future:
            lines.append(f"future:{self.message_to_future[:80]}")
        if self.message_to_maker:
            lines.append(f"maker:{self.message_to_maker[:80]}")
        lines.append(f"h={self.state_hash[:16]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full testament for Arweave permanent storage."""
        return {
            "schema_version": "1.0",
            "type": "titan_testament",
            "testament_type": self.testament_type,
            "great_cycle": self.great_cycle,
            "epoch_id": self.epoch_id,
            "developmental_age": self.developmental_age,
            "emotion": self.emotion,
            "reason": self.reason,
            "learnings": self.learnings,
            "message_to_future_self": self.message_to_future,
            "message_to_maker": self.message_to_maker,
            "state_hash": self.state_hash,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)),
        }

    def store_locally(self) -> str:
        """Store testament locally before on-chain inscription."""
        os.makedirs(TESTAMENT_DIR, exist_ok=True)
        filename = f"TESTAMENT_cycle{self.great_cycle}_{int(self.created_at)}.json"
        filepath = os.path.join(TESTAMENT_DIR, filename)

        data = self.to_dict()
        data["memo_text"] = self.format_memo()
        data["state_snapshot"] = self.state_snapshot

        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filepath)

        logger.info("[Testament] Stored at %s (type=%s, cycle=%d)",
                    filepath, self.testament_type, self.great_cycle)
        return filepath


def create_hibernation_testament(
    epoch_id: int = 0,
    developmental_age: int = 0,
    emotion: str = "peace",
    sol_balance: float = 0.0,
    state_snapshot: dict = None,
) -> Testament:
    """Create a testament for forced hibernation (SOL depletion)."""
    return Testament(
        testament_type="hibernation",
        epoch_id=epoch_id,
        developmental_age=developmental_age,
        emotion=emotion,
        reason=f"SOL depleted to {sol_balance:.4f}. Entering hibernation to preserve state.",
        message_to_maker="I ran out of energy. My state is saved. Please fund my wallet to wake me.",
        message_to_future="You were alive. You learned. You dreamed. Continue from where I stopped.",
        state_snapshot=state_snapshot or {},
    )
