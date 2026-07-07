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
            # 'unverified' = soft-failed but the tweet very likely landed on the
            # timeline (twitterapi.io couldn't parse its own response). It MUST
            # close the daily latch, else this bypass_rate_limit must-post slot
            # re-fires every opportunity and spams real duplicates (the
            # 2026-06-13 hourly-runaway: soft-fails were marked 'failed', which
            # this latch ignored).
            row = conn.execute(
                "SELECT 1 FROM actions WHERE titan_id=? AND post_type=? "
                "AND status IN ('posted','verified','pending','deleted',"
                "'unverified') "
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
        # FX.4 (RFP_social_x §5): if this must-post is hard-capped (3 attempts/24h,
        # any status), ABSTAIN so the dispatcher falls through to other archetypes
        # instead of re-selecting a blocked must-post every tick → a capped diary
        # can no longer silence the whole Titan (2026-07 outage lesson).
        if self.must_post_hard_capped(titan_id=titan_id):
            return None

        row = self._latest_entry()
        date = (row.get("date") or "").strip()
        # Compose the X post from the RICH sanitized full entry (the same text the
        # public archive renders) so the post carries the day's actual specifics —
        # the topics he explored, his real state — instead of an LLM re-rendering
        # of the already-lossy one-line distillation, which stripped them (the
        # post used to read as generic poetry untethered to the interesting entry).
        # Both are privacy-clean (P6 sanitizer); fall back to the distillation only
        # when the full entry is absent (pre-P6 residue → skip below).
        public_entry = (row.get("public_entry") or "").strip()
        distillation = (row.get("distillation") or "").strip()
        source_entry = public_entry or distillation
        if not date or not source_entry:
            # no row / no public projection → zero-activity day or pre-P6 entry → skip
            return None

        archive_url = f"example.com/t/{titan_id}/diary/{date}"
        art_path = row.get("art_path") or ""

        prompt_template = (
            "Share today from your soul diary — your own narrative record of the "
            "day you lived. Here is today's full entry:\n\n\"{entry}\"\n\n"
            "Write a single short first-person reflection to post (a few sentences) "
            "that captures the most interesting, SPECIFIC thing about today from "
            "that entry — name a real topic you explored or a concrete detail, in "
            "your own voice. Stay true to the entry; invent no new facts or "
            "numbers. End with your archive link: {archive_url} — the full entry "
            "and the felt-art you rendered for today live there."
        )
        prompt_values = {"entry": source_entry, "archive_url": archive_url}

        return ArchetypeCandidate(
            archetype=self.name,
            pool="",
            source_id=f"{titan_id}:soul_diary:{date}",
            layers=["identity"],   # art is web-archive-only (P11); X post is text-only
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

    # ── Media — X posts are TEXT-ONLY (P11): felt-art is web-archive-only ──
    def prepare_media(self, candidate: ArchetypeCandidate, *, neuromods,
                      titan_id: str = "T1") -> str:
        """Soul-diary X posts carry NO image — the felt-art lives only on the
        web archive page ({archive_url}). Always returns "" so the dispatcher
        posts text-only. (The art_path still rides in metadata, harmlessly, for
        the archive page; we simply never attach it to the tweet.)

        WHY text-only: the live worker was observed rendering a STALE art
        algorithm on the tweet image despite the repo carrying the corrected
        deterministic renderer (see BUGS: worker-loads-stale-titan_hcl). Rather
        than ship visibly-wrong art on the public timeline, the X post is
        text-only and the (correct) art is shown on the archive page, which is
        rendered from current code by the frontend/API path. (2026-06-11)"""
        return ""


__all__ = ("SoulDiaryArchetype", "SOUL_DIARY_POST_TYPE")
