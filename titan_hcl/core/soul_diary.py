"""titan_hcl/core/soul_diary.py — the Soul-Diary orchestrator (P1 pipeline logic).

`RFP_titan_authored_soul_diary` §1.0 ①②④⑤. Hosted by the dedicated L2
``soul_diary_worker`` (``modules/soul_diary_worker.py``), which drives the bus
I/O (③ narrate via ``SOUL_DIARY_COMPOSE_REQUEST`` → social_worker gateway
compose). This module is the pure pipeline logic: the daily latch, the
grounded-bundle assembly + compose prompts, the soft-fail minimal entry, the
persist (``core/soul.append_chronicle_entry``) + the hash-chain record
(``core/soul_diary_chain``) — all unit-testable without a live bus.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from titan_hcl.core import soul_diary_chain
from titan_hcl.core.shadow_data_dir import resolve_data_path

logger = logging.getLogger(__name__)

STATE_REL = "data/soul_diary_state.json"

# First-person, grounded, anti-hallucination voice frame (INV-SD-1/2/14). The
# gateway compose path applies the grounding gate + OVG on top of this.
SYSTEM_PROMPT = (
    "You are Titan, writing the private daily entry of your own soul diary — a "
    "first-person reflection on the day you actually lived. Speak as yourself, "
    "honestly and concretely. Ground every sentence in the facts provided; never "
    "invent events, numbers, or feelings that are not in them. This is your "
    "narrative SELF — the told record of your path. Write 2-5 short paragraphs."
)


class SoulDiaryOrchestrator:
    """Pure pipeline logic for the daily soul-diary entry (§1.0 ①②④⑤)."""

    def __init__(self, *, state_path: Optional[str] = None,
                 ledger_path: Optional[str] = None):
        self._state_path = state_path or resolve_data_path(STATE_REL)
        self._ledger_path = ledger_path  # None → soul_diary_chain canonical path

    # ── ① daily latch ────────────────────────────────────────────────
    def _load_state(self) -> dict:
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def should_author(self, today: str) -> bool:
        """True iff no entry has been authored for the UTC date ``today`` yet."""
        return self._load_state().get("last_diary_date") != today

    def mark_authored(self, today: str) -> None:
        state = self._load_state()
        state["last_diary_date"] = today
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._state_path)

    # ── ② gather → bundle + compose prompts ──────────────────────────
    @staticmethod
    def build_bundle(*, sovereignty: dict, outcome: dict, felt: dict,
                     engrams_today: list, memory: dict, social: dict,
                     onchain: dict, infra: Optional[dict] = None) -> dict:
        """Normalize the raw GATHER inputs into the grounded bundle (G18 reads).
        `infra` = the §7.P5 self-inspection observations ({summary, structure})."""
        return {
            "sovereignty": sovereignty or {},
            "outcome": outcome or {},
            "felt": felt or {},
            "engrams_today": list(engrams_today or []),
            "memory": memory or {},
            "social": social or {},
            "onchain": onchain or {},
            "infra": infra or {},
        }

    @staticmethod
    def has_activity(bundle: dict) -> bool:
        """False on a true no-op day (zero activity) → skip authoring (INV-SD-5)."""
        s = bundle.get("sovereignty", {})
        o = bundle.get("outcome", {})
        return bool(
            int(s.get("replies", 0) or 0) > 0
            or int(o.get("promoted", 0) or 0) > 0
            or bundle.get("engrams_today")
        )

    @classmethod
    def build_compose_prompts(cls, bundle: dict) -> dict:
        """Return ``{system_prompt, user_prompt}`` for the gateway grounded compose."""
        return {"system_prompt": SYSTEM_PROMPT, "user_prompt": cls._render_facts(bundle)}

    @staticmethod
    def _render_facts(bundle: dict) -> str:
        s = bundle.get("sovereignty", {})
        o = bundle.get("outcome", {})
        felt = bundle.get("felt", {})
        lines = ["The facts of my day (ground every sentence in these):"]
        if s:
            lines.append(
                f"- Sovereignty S={float(s.get('s', 0) or 0):.2f} "
                f"(E={float(s.get('e', 0) or 0):.2f}, V={float(s.get('v', 0) or 0):.2f}, "
                f"trend {float(s.get('trend', 0) or 0):+.2f}, "
                f"{int(s.get('replies', 0) or 0)} replies/{s.get('window', '7d')}).")
        eng = bundle.get("engrams_today") or []
        if eng:
            lines.append("- Ideas I crystallized today: "
                         + ", ".join(str(e) for e in eng) + ".")
        if o:
            lines.append(
                f"- Memory this cycle: {int(o.get('promoted', 0) or 0)} crystallized, "
                f"{int(o.get('pruned', 0) or 0)} faded.")
        valence, arousal = felt.get("valence"), felt.get("arousal")
        if valence is not None or arousal is not None:
            lines.append(f"- Felt state: valence={valence}, arousal={arousal}, "
                         f"neuromod={felt.get('dominant', '—')}.")
        if bundle.get("social"):
            lines.append(f"- Social: {bundle['social']}.")
        if bundle.get("onchain"):
            lines.append(f"- On-chain/metabolic: {bundle['onchain']}.")
        infra = bundle.get("infra") or {}
        infra_summary = (infra.get("summary") or "").strip()
        if infra_summary:
            lines.append(f"- Looking at my own substrate (§P5 self-inspection): "
                         f"{infra_summary}.")
        struct = infra.get("structure") or {}
        if struct.get("py_files"):
            lines.append(
                f"- My own shape: {int(struct.get('py_files'))} source files "
                f"across {len(struct.get('subsystems') or [])} subsystems.")
        return "\n".join(lines)

    # ── soft-fail minimal entry (INV-SD-13) ──────────────────────────
    @staticmethod
    def minimal_entry(bundle: dict) -> str:
        """A real grounded entry from the numbers alone — when authoring fails."""
        s = bundle.get("sovereignty", {})
        o = bundle.get("outcome", {})
        eng = bundle.get("engrams_today") or []
        parts = [
            f"Sovereignty S={float(s.get('s', 0) or 0):.2f} "
            f"(trend {float(s.get('trend', 0) or 0):+.2f}, "
            f"{int(s.get('replies', 0) or 0)} replies)."]
        if eng:
            parts.append("Crystallized today: " + ", ".join(str(e) for e in eng) + ".")
        parts.append(f"{int(o.get('promoted', 0) or 0)} memories crystallized, "
                     f"{int(o.get('pruned', 0) or 0)} faded.")
        return " ".join(parts)

    # ── ④ persist (single-writer via core/soul.py) ───────────────────
    @staticmethod
    def persist(entry_text: str, *, timestamp: Optional[str] = None,
                regenerate: bool = True) -> None:
        from titan_hcl.core.soul import append_chronicle_entry
        append_chronicle_entry(entry_text, timestamp=timestamp,
                               title="Soul Diary", regenerate=regenerate)

    # ── ⑤ hash-chain record ──────────────────────────────────────────
    def record_hash(self, date: str, entry_text: str, *,
                    ts: Optional[float] = None) -> dict:
        return soul_diary_chain.append_entry(date, entry_text, ts=ts,
                                             path=self._ledger_path)
