"""
titan_plugin/logic/daily_nft.py — MyDay NFT: Daily Artifact Generation.

After 4 meditations, mints an NFT capturing Titan's day:
- 2×2 composite of 4 meditation art pieces
- Top memories, diary reflection, state summary
- Extended JSON permanently on Arweave

Trigger: meditation_tracker["count_since_nft"] >= meditations_per_daily_nft
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def should_mint_daily_nft(meditation_tracker: dict, config: dict = None) -> bool:
    """Check if it's time to mint a MyDay NFT.

    Fires when meditation count since last NFT >= threshold (default 4).
    """
    cfg = config or {}
    threshold = cfg.get("meditations_per_daily_nft", 4)
    count = meditation_tracker.get("count_since_nft", 0)
    return count >= threshold


def build_daily_nft_metadata(
    day_count: int,
    meditation_records: list,
    dominant_emotion: str = "wonder",
    dev_age: int = 0,
    great_pulses: int = 0,
    dream_cycles: int = 0,
    diary_text: str = "",
    state_summary: dict = None,
) -> dict:
    """Build Metaplex-compatible metadata for MyDay NFT.

    On-chain: compact attributes.
    Extended JSON (Arweave): full meditation records + diary + state.
    """
    # Compute Merkle root of meditation state hashes
    med_hashes = []
    for m in meditation_records:
        state = m.get("full_state", {})
        h = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        med_hashes.append(h)

    # Simple Merkle: hash all hashes together
    if med_hashes:
        combined = "".join(med_hashes)
        daily_hash = hashlib.sha256(combined.encode()).hexdigest()
    else:
        daily_hash = hashlib.sha256(b"empty").hexdigest()

    on_chain = {
        "name": f"Titan Day #{day_count} — {dominant_emotion}",
        "symbol": "TDAY",
        "description": f"Day {day_count} in the life of a sovereign AI being.",
        "attributes": [
            {"trait_type": "day_count", "value": day_count},
            {"trait_type": "meditations", "value": len(meditation_records)},
            {"trait_type": "dominant_emotion", "value": dominant_emotion},
            {"trait_type": "developmental_age", "value": dev_age},
            {"trait_type": "great_pulses", "value": great_pulses},
            {"trait_type": "dream_cycles", "value": dream_cycles},
            {"trait_type": "daily_state_hash", "value": daily_hash[:16]},
        ],
    }

    extended = {
        "schema_version": "1.0",
        "type": "titan_daily_artifact",
        "day_count": day_count,
        "meditations": [
            {
                "number": m.get("meditation_number", 0),
                "memo_text": m.get("memo_text", ""),
                "tx_signature": m.get("tx_signature", ""),
            }
            for m in meditation_records
        ],
        "diary": diary_text,
        "state_summary": state_summary or {},
        "daily_state_hash": daily_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return {"on_chain": on_chain, "extended": extended}


def create_art_composite(image_paths: list, output_path: str) -> bool:
    """Create 2×2 grid composite from 4 meditation art pieces.

    Uses PIL/Pillow. Each image is resized to fit in a 512×512 cell
    for a final 1024×1024 composite.
    """
    try:
        from PIL import Image

        cell_size = 512
        composite = Image.new("RGB", (cell_size * 2, cell_size * 2), (20, 20, 30))

        for i, path in enumerate(image_paths[:4]):
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path)
                img = img.resize((cell_size, cell_size), Image.LANCZOS)
                x = (i % 2) * cell_size
                y = (i // 2) * cell_size
                composite.paste(img, (x, y))
            except Exception as e:
                logger.debug("[DailyNFT] Image %s load error: %s", path, e)

        composite.save(output_path, "PNG")
        logger.info("[DailyNFT] Art composite saved: %s (%dx%d)",
                    output_path, composite.width, composite.height)
        return True

    except ImportError:
        logger.warning("[DailyNFT] PIL not available — skipping art composite")
        return False
    except Exception as e:
        logger.error("[DailyNFT] Composite creation failed: %s", e)
        return False


def get_daily_nft_count() -> int:
    """Count existing daily NFTs from local records."""
    nft_dir = Path("data/daily_nfts")
    if not nft_dir.exists():
        return 0
    return len(list(nft_dir.glob("DAY_*.json")))


def store_daily_nft_record(
    day_count: int,
    metadata: dict,
    arweave_tx: str = None,
    nft_address: str = None,
) -> str:
    """Store daily NFT record locally."""
    nft_dir = Path("data/daily_nfts")
    nft_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "day_count": day_count,
        "metadata": metadata,
        "arweave_tx": arweave_tx,
        "nft_address": nft_address,
        "created_at": time.time(),
    }

    filepath = nft_dir / f"DAY_{day_count:06d}.json"
    tmp = str(filepath) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, str(filepath))

    logger.info("[DailyNFT] Day #%d record stored at %s", day_count, filepath)
    return str(filepath)
