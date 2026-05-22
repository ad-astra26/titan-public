"""Felt-state compact summarizer.

Used by Phase 1 X-voice enrichment to capture a 1-line, human-readable
felt-state at moments that matter (vocabulary grounding, mention reply,
person interaction). The output is deterministic — no LLM call on the hot
path. The string is designed to read naturally in archetype prompt
templates (e.g., GROUNDED_TODAY's `grounded_felt_summary` slot).

Format examples:
    "warm-still-precise (emotion=flow, DA 72%, 5HT 65%, GABA 80%)"
    "sharp-alert (emotion=focus, NE 81%, DA 71%)"
    "open-curious (emotion=wondering, DA 68%, ACh 60%)"
    "balanced (emotion=neutral)"
"""
from __future__ import annotations

from typing import Mapping


# Felt-quality words organized by neuromod axis. The picker reads dominant
# neuromods (>0.65 high or <0.20 low) and stitches up to 2 quality tags so
# the resulting summary stays compact (1-line target).

_TAG_HIGH = {
    "DA": "expansive",
    "5HT": "still",
    "NE": "alert",
    "ACh": "precise",
    "GABA": "calm",
    "Endorphin": "warm",
    "Glutamate": "intense",
}

_TAG_LOW = {
    "DA": "withdrawn",
    "5HT": "restless",
    "NE": "drowsy",
    "ACh": "soft",
    "GABA": "edgy",
    "Endorphin": "muted",
    "Glutamate": "muted",
}

_DOMINANT_HIGH_THRESHOLD = 0.65
_DOMINANT_LOW_THRESHOLD = 0.20


def compact_felt_summary(
    neuromods: Mapping[str, float] | None,
    emotion: str = "",
    *,
    max_tags: int = 3,
    max_neuromod_parts: int = 3,
) -> str:
    """Deterministic 1-line felt-state summary.

    Returns a short, prompt-ready string that names the dominant felt
    qualities and highlights the most prominent neuromod readings.
    Robust to missing/malformed inputs — never raises.
    """
    nm: dict[str, float] = {}
    if neuromods:
        for k, v in neuromods.items():
            try:
                nm[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    tags: list[str] = []
    for code, lvl in nm.items():
        if lvl >= _DOMINANT_HIGH_THRESHOLD and code in _TAG_HIGH:
            tags.append((lvl, _TAG_HIGH[code]))
        elif lvl <= _DOMINANT_LOW_THRESHOLD and code in _TAG_LOW:
            tags.append((1.0 - lvl, _TAG_LOW[code]))

    # Sort tags by intensity (most extreme first), keep up to max_tags
    tags.sort(reverse=True)
    chosen_tags = [t for _, t in tags[:max_tags]]
    quality = "-".join(chosen_tags) if chosen_tags else "balanced"

    # Build neuromod parts (top-N most-extreme readings)
    nm_extremes = sorted(
        nm.items(),
        key=lambda kv: abs(kv[1] - 0.5),
        reverse=True,
    )[:max_neuromod_parts]
    nm_parts = []
    for code, lvl in nm_extremes:
        if abs(lvl - 0.5) < 0.10:
            continue  # too near baseline to be informative
        label = "5HT" if code == "5HT" else code
        nm_parts.append(f"{label} {lvl * 100:.0f}%")

    detail_parts: list[str] = []
    if emotion:
        detail_parts.append(f"emotion={emotion}")
    detail_parts.extend(nm_parts)

    if detail_parts:
        return f"{quality} ({', '.join(detail_parts)})"
    return quality


def neuromods_to_json(neuromods: Mapping[str, float] | None) -> str:
    """Serialize neuromods to a compact JSON string for DB storage."""
    import json

    if not neuromods:
        return "{}"
    out: dict[str, float] = {}
    for k, v in neuromods.items():
        try:
            out[str(k)] = round(float(v), 4)
        except (TypeError, ValueError):
            continue
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


__all__ = ("compact_felt_summary", "neuromods_to_json")
