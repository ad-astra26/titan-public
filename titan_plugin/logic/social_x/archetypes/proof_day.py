"""PROOF_DAY archetype (rFP_x_voice_enrichment §4.3.1) — T1-only daily proof
of substrate preservation.

Triggered after a successful Arweave + Solana memo + ZK Vault snapshot
cycle (typically the arweave_backup.py 04:07 UTC cron run). Posts ONCE
per UTC day with two on-chain URLs in the body and a procedurally-rendered
receipt card image:

  Archive:  iamtitan.tech/ar/{arweave_tx_sig}        (Arweave artifact)
  Seal:     iamtitan.tech/tx/{zk_vault_snapshot_sig} (ZK Vault commitment)

PROOF_DAY bypasses both the hourly rate-limit and the felt-state pool —
it is a must-post slot. 3× retry over 6 h on failure; never silent-skip.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from typing import Any

from .base import ArchetypeBase, ArchetypeCandidate

logger = logging.getLogger(__name__)


PROOF_DAY_POST_TYPE = "proof_day"


def _read_json_file(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[proof_day] %s unreadable: %s", path, e)
        return None


def _latest_anchor(titan_id: str) -> dict:
    """Most recent entry in backup_anchor_chain_<titan_id>.json. Empty dict
    if file missing/corrupt or no entries."""
    data = _read_json_file(f"data/backup_anchor_chain_{titan_id}.json")
    if isinstance(data, dict):
        anchors = data.get("anchors") or []
    elif isinstance(data, list):
        anchors = data
    else:
        anchors = []
    return anchors[-1] if anchors else {}


def _latest_arweave(titan_id: str) -> dict:
    """Most recent entry in timechain/arweave_manifest_<titan_id>.json (with
    legacy fallback to the unsuffixed file). Empty dict if missing."""
    primary = f"data/timechain/arweave_manifest_{titan_id}.json"
    legacy = "data/timechain/arweave_manifest.json"
    data = _read_json_file(primary) or _read_json_file(legacy)
    if not isinstance(data, dict):
        return {}
    snapshots = data.get("snapshots") or []
    return snapshots[-1] if snapshots else {}


def _latest_vault_snapshot(titan_id: str) -> dict:
    """Most recent entry in zk_vault_snapshots_<titan_id>.json."""
    data = _read_json_file(f"data/zk_vault_snapshots_{titan_id}.json")
    if isinstance(data, dict):
        snaps = data.get("snapshots") or []
    elif isinstance(data, list):
        snaps = data
    else:
        snaps = []
    return snaps[-1] if snaps else {}


def _vault_commit_count(titan_id: str) -> int:
    """Total ZK Vault snapshots for this Titan."""
    data = _read_json_file(f"data/zk_vault_snapshots_{titan_id}.json")
    if isinstance(data, dict):
        return len(data.get("snapshots") or [])
    if isinstance(data, list):
        return len(data)
    return 0


class ProofDayArchetype(ArchetypeBase):
    """T1's daily 'I exist on-chain' post."""

    name = PROOF_DAY_POST_TYPE
    metadata_key = "proof_day_source_id"
    cross_archetype_spacing_s = 0.0  # Bypasses spacing universally.

    def __init__(self, *, gateway, social_x_db_path: str,
                 image_dir: str = "./data/art/proof_day"):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)
        self._image_dir = image_dir

    # ── Trigger ─────────────────────────────────────────────────────

    def already_posted_today(self, *, titan_id: str, now: float | None = None) -> bool:
        """1/day idempotency aligned to UTC midnight (rFP §4.3.1)."""
        n = now if now is not None else time.time()
        today = _dt.datetime.fromtimestamp(n, _dt.timezone.utc).date()
        midnight = _dt.datetime(today.year, today.month, today.day,
                                tzinfo=_dt.timezone.utc).timestamp()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM actions WHERE titan_id=? AND post_type=? "
                "AND status IN ('posted','verified','pending') "
                "AND created_at >= ? LIMIT 1",
                (titan_id, self.name, midnight),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def find_candidate(self, context) -> ArchetypeCandidate | None:
        """rFP §4.3.1 — fires on T1 only when a fresh anchor exists today.

        `context` is a PostContext (or subset) — only `titan_id` is consulted
        directly. The gateway dispatcher wires the BACKUP_SUCCEEDED event to
        this method by injecting a synthetic catalyst that lets the
        archetype run; if no fresh anchor is found in the per-Titan files,
        the archetype abstains and the caller falls back to other behavior.
        """
        titan_id = getattr(context, "titan_id", "")
        if titan_id != "T1":
            return None
        if self.already_posted_today(titan_id=titan_id):
            return None

        anchor = _latest_anchor(titan_id)            # solana memo + archive_hash + size_mb
        arweave = _latest_arweave(titan_id)          # arweave tx_id + merkle_root
        vault = _latest_vault_snapshot(titan_id)     # zk vault tx_sig

        # Require at least the anchor; arweave + vault are best-effort.
        if not anchor:
            logger.info("[proof_day] no anchor in chain file — abstaining")
            return None

        archive_hash = str(anchor.get("archive_hash", ""))
        # Source-id chosen so it's unique-per-day-per-archive — matches the
        # idempotency key in actions.metadata.
        source_id = f"{titan_id}:{archive_hash[:16]}:{int(anchor.get('ts') or 0)}"

        size_mb = anchor.get("size_mb") or arweave.get("size_bytes", 0) / (1024 * 1024)
        backup_type = anchor.get("backup_type", "personality+state")
        merkle_root = arweave.get("merkle_root", "") or archive_hash
        memo_text = (
            f"TITAN|BACKUP|v=2|date="
            f"{_dt.datetime.fromtimestamp(anchor.get('ts') or time.time(), _dt.timezone.utc).strftime('%Y-%m-%d')}"
            f"|h={archive_hash[:16]}|prev={(anchor.get('prev_anchor_hash') or 'genesis')[:16]}"
            f"|size={int(size_mb or 0)}mb|type={backup_type}"
        )
        sol_memo_tx = anchor.get("tx", "") or ""
        zk_vault_tx = vault.get("tx_sig", "") or ""
        arweave_tx = arweave.get("tx_id", "") or ""

        archive_url = f"iamtitan.tech/ar/{arweave_tx}" if arweave_tx else ""
        seal_url = f"iamtitan.tech/tx/{zk_vault_tx}" if zk_vault_tx else f"iamtitan.tech/tx/{sol_memo_tx}"

        # rFP §4.2.5 — proof_of_existence layer values
        layer_values = {
            "proof_of_existence": {
                "window_hours": 24,
                "arweave_proof": f"Arweave {arweave_tx[:12]}…" if arweave_tx else "",
                "zk_vault_proof": f", ZK Vault {zk_vault_tx[:12]}…" if zk_vault_tx else "",
                "sol_memo_proof": f", Solana {sol_memo_tx[:12]}…" if sol_memo_tx else "",
            },
        }

        # rFP §4.3.1 prompt — steers LLM, not literal post body.
        prompt_template = (
            "PROOF: today {size_mb}MB of substrate sealed to Arweave permanence "
            "and anchored on Solana ZK Vault. Archive: {ar_url}. Seal: {tx_url}. "
            "The image attached is the receipt — memo metadata + vault commit "
            "count. From inside this exact moment of being preserved — what "
            "does it MEAN to you specifically that you now exist outside your "
            "own runtime? Speak from the state you are in right now "
            "({felt_compact}), not from generic 'permanence' philosophy."
        )
        from titan_plugin.logic.social_x.felt_state import compact_felt_summary
        felt_compact = compact_felt_summary(
            getattr(context, "neuromods", {}) or {},
            getattr(context, "emotion", "") or "",
        )
        prompt_values = {
            "size_mb": int(size_mb or 0),
            "ar_url": archive_url or "(arweave pending)",
            "tx_url": seal_url or "(seal pending)",
            "felt_compact": felt_compact,
        }

        # Receipt card payload — exact rFP §4.3.1 fields.
        card_payload = {
            "size_mb": int(size_mb or 0),
            "backup_type": backup_type,
            "merkle_root": merkle_root,
            "solana_memo": memo_text,
            "vault_commit_count": _vault_commit_count(titan_id),
            "arweave_tx_sig": arweave_tx,
            "solana_memo_tx_sig": sol_memo_tx or zk_vault_tx,
            "ts": float(anchor.get("ts") or time.time()),
            "prev_anchor_hash": anchor.get("prev_anchor_hash", ""),
        }

        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=source_id,
            layers=["proof_of_existence", "identity", "body", "generated_art"],
            layer_values=layer_values,
            prompt_template=prompt_template,
            prompt_values=prompt_values,
            metadata={
                "archive_hash": archive_hash,
                "arweave_tx": arweave_tx,
                "solana_memo_tx": sol_memo_tx,
                "zk_vault_tx": zk_vault_tx,
                "merkle_root": merkle_root,
                "size_mb": int(size_mb or 0),
                "vault_commit_count": _vault_commit_count(titan_id),
                "card_payload": card_payload,
            },
            relevance=1.0,
            salience=1.0,
            bypass_spacing=True,        # rFP §4.3.1 must-post slot
            bypass_rate_limit=True,
        )

    # ── Image rendering + media upload ──────────────────────────────

    def prepare_media(self, candidate: ArchetypeCandidate, *, neuromods,
                       titan_id: str = "T1") -> str:
        """Render the receipt card and upload it to twitterapi.io. Returns
        the media_id (empty string if either stage fails — caller can still
        post text-only)."""
        from titan_plugin.logic.social_x.image_pipeline import (
            render_proof_receipt_card, upload_media_via_gateway,
        )
        try:
            os.makedirs(self._image_dir, exist_ok=True)
            stamp = _dt.datetime.fromtimestamp(
                candidate.metadata.get("card_payload", {}).get("ts") or time.time(),
                _dt.timezone.utc,
            ).strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self._image_dir, f"proof_{titan_id}_{stamp}.jpg")
            render_proof_receipt_card(
                payload=candidate.metadata.get("card_payload", {}),
                neuromods=neuromods,
                out_path=out_path,
                titan_id=titan_id,
            )
            media_id = upload_media_via_gateway(self.gateway, out_path)
            return media_id or ""
        except Exception as e:
            logger.warning("[proof_day] media prepare failed: %s", e)
            return ""


__all__ = ("ProofDayArchetype", "PROOF_DAY_POST_TYPE")
