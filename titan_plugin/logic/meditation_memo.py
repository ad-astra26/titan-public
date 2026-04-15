"""
titan_plugin/logic/meditation_memo.py — Meditation Memo Formatting + Local Storage.

Formats human-readable AND Titan-parseable state summaries for on-chain inscription.
Stores full meditation state locally for history queries.

Memo format (~200 bytes):
  MED#4527|e=15435|age=892|chi=0.61|em=wonder
  DA=0.54 5HT=0.80 NE=0.79 ACh=0.74 End=0.91 GABA=0.09
  cog:+12,3promo|NS:6420t|pi:27c|dr:2
  top:"learned bright from maker"
  h=a3f8b2c1...
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MEMO_DIR = "data/meditation_memos"


def format_meditation_memo(
    meditation_count: int,
    epoch_id: int = 0,
    dev_age: int = 0,
    chi_total: float = 0.0,
    emotion: str = "peace",
    neuromods: dict = None,
    ns_transitions: int = 0,
    pi_clusters: int = 0,
    dream_cycles: int = 0,
    cognee_promoted: int = 0,
    cognee_added: int = 0,
    top_memory: str = "",
    full_state: dict = None,
) -> str:
    """Format a meditation memo for on-chain inscription (~200 bytes).

    Human-readable AND Titan-parseable. Each field has a known prefix.
    """
    nm = neuromods or {}
    lines = [
        f"MED#{meditation_count}|e={epoch_id}|age={dev_age}|chi={chi_total:.2f}|em={emotion}",
        f"DA={nm.get('DA', 0):.2f} 5HT={nm.get('5HT', 0):.2f} NE={nm.get('NE', 0):.2f} "
        f"ACh={nm.get('ACh', 0):.2f} End={nm.get('Endorphin', 0):.2f} GABA={nm.get('GABA', 0):.2f}",
        f"cog:+{cognee_added},{cognee_promoted}promo|NS:{ns_transitions}t|pi:{pi_clusters}c|dr:{dream_cycles}",
    ]

    if top_memory:
        lines.append(f'top:"{top_memory[:60]}"')

    # State hash for verification
    state_json = json.dumps(full_state or {}, sort_keys=True, separators=(",", ":"))
    state_hash = hashlib.sha256(state_json.encode()).hexdigest()[:16]
    lines.append(f"h={state_hash}")

    return "\n".join(lines)


def store_meditation_locally(
    meditation_count: int,
    memo_text: str,
    full_state: dict = None,
    tx_signature: str = None,
) -> str:
    """Store full meditation record locally as JSON.

    Returns the filepath of the stored record.
    """
    os.makedirs(MEMO_DIR, exist_ok=True)

    record = {
        "meditation_number": meditation_count,
        "memo_text": memo_text,
        "tx_signature": tx_signature,
        "full_state": full_state or {},
        "stored_at": time.time(),
        "stored_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    filepath = os.path.join(MEMO_DIR, f"MED_{meditation_count:06d}.json")

    # Atomic write
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, filepath)

    logger.info("[MeditationMemo] Stored MED#%d at %s", meditation_count, filepath)
    return filepath


def get_meditation_history(limit: int = 10) -> list[dict]:
    """Get recent meditation records from local storage.

    Returns list of meditation records, newest first.
    """
    if not os.path.exists(MEMO_DIR):
        return []

    files = sorted(Path(MEMO_DIR).glob("MED_*.json"), reverse=True)

    records = []
    for f in files[:limit]:
        try:
            with open(f) as fh:
                record = json.load(fh)
                # Include filename for reference
                record["_file"] = f.name
                records.append(record)
        except Exception:
            continue

    return records


def parse_meditation_memo(memo_text: str) -> dict:
    """Parse a meditation memo back into structured data.

    This is how Titan reads his own memos from the chain.
    """
    result = {}
    lines = memo_text.strip().split("\n")

    if not lines:
        return result

    # Line 1: MED#N|e=X|age=Y|chi=Z|em=W
    for part in lines[0].split("|"):
        if part.startswith("MED#"):
            result["meditation_number"] = int(part[4:])
        elif part.startswith("e="):
            result["epoch"] = int(part[2:])
        elif part.startswith("age="):
            result["dev_age"] = int(part[4:])
        elif part.startswith("chi="):
            result["chi"] = float(part[4:])
        elif part.startswith("em="):
            result["emotion"] = part[3:]

    # Line 2: neuromod levels
    if len(lines) > 1:
        nm = {}
        for token in lines[1].split():
            if "=" in token:
                k, v = token.split("=", 1)
                try:
                    nm[k] = float(v)
                except ValueError:
                    nm[k] = v
        result["neuromods"] = nm

    # Line 3: metrics
    if len(lines) > 2:
        for part in lines[2].split("|"):
            if part.startswith("NS:"):
                result["ns_transitions"] = int(part[3:].rstrip("t"))
            elif part.startswith("pi:"):
                result["pi_clusters"] = int(part[3:].rstrip("c"))
            elif part.startswith("dr:"):
                result["dream_cycles"] = int(part[3:])

    # Line 4+: top memory, hash
    for line in lines[3:]:
        if line.startswith("top:"):
            result["top_memory"] = line[4:].strip('"')
        elif line.startswith("h="):
            result["state_hash"] = line[2:]

    return result
