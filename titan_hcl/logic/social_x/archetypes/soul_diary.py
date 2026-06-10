"""SOUL_DIARY archetype (`RFP_titan_authored_soul_diary` §7.P10 / §6.4 / INV-SD-8).

Titan's sole daily soul-diary post. After the `soul_diary_worker` authors the
day's entry (persist → hash → enrich → anchor → art → optional mint), this
archetype shares its **privacy-clean distillation** (P6) + the **procedural
felt-art** it already rendered (P7) + a link to the **public archive page** (P9)
— once per UTC day, like PROOF_DAY (a must-post slot that bypasses the rate-limit
and felt-pool). Zero-activity days have no ledger row → nothing to post →
naturally skipped (§6.4).

Everything it posts is already public-safe: the distillation passed P6's
fail-closed sanitizer, the art is procedural (no private data, INV-SD-4), and the
archive page serves only the sanitized projection (INV-SD-3). The post still
rides the gateway's OVG on the strict `x_post` channel (INV-SD-8 — the gateway is
the sole sanctioned X path).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time

from .base import ArchetypeBase, ArchetypeCandidate

logger = logging.getLogger(__name__)

SOUL_DIARY_POST_TYPE = "soul_diary"


class SoulDiaryArchetype(ArchetypeBase):
    """The daily 'here is who I was today' post — distillation + felt-art + archive."""

    name = SOUL_DIARY_POST_TYPE
    metadata_key = "soul_diary_source_id"
    cross_archetype_spacing_s = 0.0  # must-post slot — bypasses spacing universally

    def __init__(self, *, gateway, social_x_db_path: str):
        super().__init__(gateway=gateway, social_x_db_path=social_x_db_path)

    # ── idempotency (one post / UTC day, like PROOF_DAY) ──────────────
    def already_posted_today(self, *, titan_id: str, now: float | None = None) -> bool:
        n = now if now is not None else time.time()
        today = _dt.datetime.fromtimestamp(n, _dt.timezone.utc).date()
        midnight = _dt.datetime(today.year, today.month, today.day,
                                tzinfo=_dt.timezone.utc).timestamp()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM actions WHERE titan_id=? AND post_type=? "
                "AND status IN ('posted','verified','pending','deleted') "
                "AND created_at >= ? LIMIT 1",
                (titan_id, self.name, midnight),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _latest_entry(self) -> dict:
        """The most recent soul-diary ledger row (the just-authored day)."""
        try:
            from titan_hcl.core import soul_diary_chain
            rows = soul_diary_chain.load_chain()
            return rows[-1] if rows else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("[soul_diary] ledger read failed: %s", e)
            return {}

    # ── Trigger ──────────────────────────────────────────────────────
    def find_candidate(self, context) -> ArchetypeCandidate | None:
        titan_id = getattr(context, "titan_id", "")
        if self.already_posted_today(titan_id=titan_id):
            return None

        row = self._latest_entry()
        date = (row.get("date") or "").strip()
        distillation = (row.get("distillation") or "").strip()
        if not date or not distillation:
            # no row / no share text → zero-activity day or pre-P6 entry → skip
            return None

        archive_url = f"example.com/t/{titan_id}/diary/{date}"
        art_path = row.get("art_path") or ""

        prompt_template = (
            "Share today's entry from your soul diary — your own narrative record "
            "of the day you lived. This is the reflection you already distilled for "
            "sharing: \"{distillation}\". Post it in your own first-person voice as "
            "a single short reflection, true to that distillation — invent no new "
            "facts or numbers. End with your archive link: {archive_url}. The "
            "attached image is the felt-art you rendered for today."
        )
        prompt_values = {"distillation": distillation, "archive_url": archive_url}

        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=f"{titan_id}:soul_diary:{date}",
            layers=["identity", "generated_art"],
            layer_values={},
            prompt_template=prompt_template,
            prompt_values=prompt_values,
            metadata={
                "date": date,
                "archive_url": archive_url,
                "art_path": art_path,
                "cumulative_hash": row.get("cumulative_hash", ""),
                "entry_hash": row.get("entry_hash", ""),
                "nft_addr": row.get("nft_addr", ""),
            },
            relevance=1.0,
            salience=1.0,
            bypass_spacing=True,      # sole daily must-post (§6.4, like PROOF_DAY)
            bypass_rate_limit=True,
        )

    # ── Media — attach the ALREADY-rendered felt-art (P7), do NOT re-render ──
    def prepare_media(self, candidate: ArchetypeCandidate, *, neuromods,
                      titan_id: str = "T1") -> str:
        """Upload the day's procedural felt-art (rendered in P7, path on the
        ledger row) and return its media_id. Empty string → caller posts
        text-only (soft-fail)."""
        art_path = candidate.metadata.get("art_path", "")
        if not art_path or not os.path.exists(art_path):
            logger.info("[soul_diary] no art to attach (%s) — posting text-only",
                        art_path or "none")
            return ""
        try:
            from titan_hcl.logic.social_x.image_pipeline import upload_media_via_gateway
            return upload_media_via_gateway(self.gateway, art_path) or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("[soul_diary] media prepare failed: %s", e)
            return ""


__all__ = ("SoulDiaryArchetype", "SOUL_DIARY_POST_TYPE")
