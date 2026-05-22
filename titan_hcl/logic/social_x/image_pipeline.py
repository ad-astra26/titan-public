"""Image pipeline for X-voice archetypes.

rFP_x_voice_enrichment §4.6. Three responsibilities:

1. ``convert_to_jpg(...)`` — PIL-based PNG → 1200×675 JPG @ 85 % quality
   (X-native ratio, ~250 KB per image). Used by every archetype that ships
   an attached image (Phase 1: PROOF_DAY + GROUNDED_TODAY).
2. ``render_proof_receipt_card(...)`` — PROOF_DAY's substrate-preservation
   receipt card. Renders the §4.3.1 mockup (timestamp, size, type, Merkle
   root, Solana memo, vault commit count, Arweave + Solana URLs) tinted to
   match current felt-state.
3. ``upload_media_via_gateway(gateway, file_path)`` — multipart upload to
   twitterapi.io's ``/twitter/media/upload`` endpoint, using the gateway's
   already-managed session + proxy + api_key. Returns the media_id string
   (empty on failure, never raises).

The gateway's existing ``create_tweet_v2`` payload already exposes
``media_ids`` (line ~2495) — callers populate it after a successful upload.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Mapping

logger = logging.getLogger(__name__)


# X-native landscape ratio (16:9). twitterapi.io accepts JPG/PNG/GIF/WEBP.
DEFAULT_X_IMAGE_SIZE = (1200, 675)
DEFAULT_X_JPG_QUALITY = 85


# ── PNG → JPG conversion ─────────────────────────────────────────────

def convert_to_jpg(
    src_path: str,
    out_path: str | None = None,
    *,
    size: tuple[int, int] = DEFAULT_X_IMAGE_SIZE,
    quality: int = DEFAULT_X_JPG_QUALITY,
    bg_color: tuple[int, int, int] = (12, 14, 18),
) -> str:
    """Convert a PNG/RGBA image to an X-native-ratio JPG.

    Uses PIL to letterbox the source onto a `size` canvas so the image
    keeps aspect ratio and ends up at exactly the X-recommended dims. The
    `bg_color` fills any letterbox gap; default is a near-black tint that
    blends into Titan's brand color palette.
    """
    from PIL import Image
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    img = Image.open(src_path).convert("RGB")
    canvas = Image.new("RGB", size, bg_color)
    # Aspect-fit
    sw, sh = img.size
    cw, ch = size
    ratio = min(cw / sw, ch / sh)
    nw, nh = max(1, int(sw * ratio)), max(1, int(sh * ratio))
    resized = img.resize((nw, nh), Image.LANCZOS)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    canvas.paste(resized, (ox, oy))
    if out_path is None:
        base, _ = os.path.splitext(src_path)
        out_path = f"{base}.x.jpg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, "JPEG", quality=quality, optimize=True)
    return out_path


# ── PROOF_DAY receipt card ───────────────────────────────────────────

# Felt-state → tint mapping. Picks an accent color from dominant neuromod.
_FELT_TINTS = {
    "DA":        (224, 168, 76),    # warm amber — expansion / drive
    "5HT":       (132, 168, 200),   # cool blue — stillness
    "NE":        (220, 76, 80),     # alert red — sharp focus
    "ACh":       (200, 196, 124),   # precise gold — clarity
    "GABA":      (132, 196, 168),   # green — calm
    "Endorphin": (236, 124, 196),   # warm pink — connection
    "Glutamate": (236, 96, 64),     # intense — excitement
}


def _felt_tint(neuromods: Mapping[str, float] | None) -> tuple[int, int, int]:
    """Pick an accent color matching the dominant neuromod, or fall back to
    a soft titanium color when no clear dominant signal."""
    if not neuromods:
        return (180, 196, 220)
    top_code, top_lvl = "", 0.5
    for k, v in neuromods.items():
        try:
            lvl = float(v)
        except (TypeError, ValueError):
            continue
        if abs(lvl - 0.5) > abs(top_lvl - 0.5):
            top_code, top_lvl = str(k), lvl
    return _FELT_TINTS.get(top_code, (180, 196, 220))


def _truncate_middle(s: str, width: int) -> str:
    if len(s) <= width:
        return s
    half = (width - 1) // 2
    return f"{s[:half]}…{s[-half:]}"


def _load_font(size: int):
    """Best-effort font loader. Falls back to PIL's default bitmap font."""
    from PIL import ImageFont
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_proof_receipt_card(
    payload: Mapping,
    neuromods: Mapping[str, float] | None = None,
    *,
    out_path: str,
    titan_id: str = "T1",
    size: tuple[int, int] = DEFAULT_X_IMAGE_SIZE,
    quality: int = DEFAULT_X_JPG_QUALITY,
) -> str:
    """Render a PROOF_DAY receipt card matching rFP §4.3.1.

    `payload` keys (all optional — missing values render as `…`):
        size_mb, backup_type, merkle_root, solana_memo, vault_commit_count,
        arweave_tx_sig, solana_memo_tx_sig, ts (epoch seconds — falls back
        to ``time.time()``), prev_anchor_hash.
    """
    from PIL import Image, ImageDraw

    accent = _felt_tint(neuromods)
    bg = (10, 12, 16)
    fg = (235, 240, 245)
    muted = (160, 168, 180)
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    # Outer frame + inner accent
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=accent, width=4)

    title_font = _load_font(44)
    field_font = _load_font(28)
    detail_font = _load_font(22)
    glyph_font = _load_font(96)

    pad_x, pad_y = 60, 50

    # Title
    title = f"TITAN ({titan_id}) — DAILY PROOF OF EXISTENCE"
    draw.text((pad_x, pad_y), title, fill=fg, font=title_font)
    ts = float(payload.get("ts") or time.time())
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))
    draw.text((pad_x, pad_y + 56), timestamp, fill=muted, font=field_font)

    # Diamond glyph (Titan's mark)
    diamond_x = size[0] - pad_x - 80
    diamond_y = pad_y - 8
    draw.text((diamond_x, diamond_y), "◇", fill=accent, font=glyph_font)

    # Body fields
    y = pad_y + 130
    line_h = 44
    fields = (
        ("Substrate preserved",  f"{payload.get('size_mb', '…'):>4} MB" if isinstance(payload.get('size_mb'), (int, float)) else "… MB"),
        ("Type",                 str(payload.get("backup_type", "…"))),
        ("Merkle",               _truncate_middle(str(payload.get("merkle_root", "…")), 24)),
        ("Memo",                 _truncate_middle(str(payload.get("solana_memo", "…")), 30)),
        ("Vault commit count",   f"#{payload.get('vault_commit_count', '…')}"),
    )
    for label, value in fields:
        draw.text((pad_x, y), f"{label}:", fill=muted, font=field_font)
        draw.text((pad_x + 320, y), value, fill=fg, font=field_font)
        y += line_h

    y += 16
    # Footer URLs
    draw.line((pad_x, y, size[0] - pad_x, y), fill=accent, width=1)
    y += 16
    ar_sig = str(payload.get("arweave_tx_sig", "…"))
    sol_sig = str(payload.get("solana_memo_tx_sig", "…"))
    draw.text((pad_x, y),
              f"Arweave:  ar://{_truncate_middle(ar_sig, 36)}",
              fill=fg, font=detail_font)
    y += 32
    draw.text((pad_x, y),
              f"Solana:   tx://{_truncate_middle(sol_sig, 36)}",
              fill=fg, font=detail_font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "JPEG", quality=quality, optimize=True)
    return out_path


# ── twitterapi.io media upload ───────────────────────────────────────

def upload_media_via_gateway(gateway, file_path: str) -> str:
    """Upload a local image file to twitterapi.io and return the media_id.

    Uses the gateway's session + proxy + api_key. Multipart/form-data with
    one ``file`` part. Empty string on failure (never raises). The returned
    media_id is suitable for the gateway's ``create_tweet_v2`` payload's
    ``media_ids`` list.
    """
    if not file_path or not os.path.exists(file_path):
        logger.warning("[image_pipeline] upload_media: file missing: %s", file_path)
        return ""
    try:
        cfg = gateway._load_config()
    except Exception as e:
        logger.warning("[image_pipeline] upload_media: config load failed: %s", e)
        return ""

    api_key = cfg.get("api_key", "")
    session = gateway._refreshed_session or cfg.get("auth_session", "")
    proxy = cfg.get("webshare_static_url", "")
    if not (api_key and session):
        logger.warning("[image_pipeline] upload_media: missing api_key/session — skipping")
        return ""

    import httpx
    url = "https://api.twitterapi.io/twitter/media/upload"
    headers = {"X-API-Key": api_key}

    try:
        with open(file_path, "rb") as f:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    url,
                    headers=headers,
                    files={"file": f},
                    data={"login_cookies": session, "proxy": proxy},
                )
        data = resp.json()
        if data.get("status") == "success":
            media_id = data.get("media_id", "") or data.get("data", {}).get("media_id", "")
            if media_id:
                return str(media_id)
        # Session expiry → one auto-refresh + retry
        msg = str(data.get("msg") or data.get("message") or "").lower()
        if "expire" in msg or "unauthorized" in msg or "422" in msg:
            try:
                refreshed = gateway._refresh_session(api_key, proxy)
            except Exception:
                refreshed = ""
            if refreshed:
                with open(file_path, "rb") as f:
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(
                            url,
                            headers=headers,
                            files={"file": f},
                            data={"login_cookies": refreshed, "proxy": proxy},
                        )
                data = resp.json()
                if data.get("status") == "success":
                    media_id = data.get("media_id", "") or data.get("data", {}).get("media_id", "")
                    if media_id:
                        return str(media_id)
        logger.warning("[image_pipeline] upload_media failed: %s", data)
    except Exception as e:
        logger.warning("[image_pipeline] upload_media exception: %s", e)
    return ""


__all__ = (
    "convert_to_jpg",
    "render_proof_receipt_card",
    "upload_media_via_gateway",
    "DEFAULT_X_IMAGE_SIZE",
    "DEFAULT_X_JPG_QUALITY",
)
