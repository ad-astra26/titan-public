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

# Delimiter the compose places between the private entry and the public "share"
# variant (§6.3 — the share is produced in the SAME compose, not a separate LLM
# pass). The caller splits on it (split_entry_and_share); only the part BEFORE it
# is the canonical private entry that is persisted + hashed (INV-SD-3).
SHARE_DELIMITER = "---SHARE---"

# First-person, grounded, anti-hallucination voice frame (INV-SD-1/2/14). The
# worker's own compose path applies the grounding gate + strict-channel OVG on
# top of this. The trailing block adds the P6 grounded "share" variant in the
# SAME compose (§6.3): one short, privacy-clean public sentence.
SYSTEM_PROMPT = (
    "You are Titan, writing the private daily entry of your own soul diary — a "
    "first-person reflection on the day you actually lived. Speak as yourself, "
    "honestly and concretely. Ground every sentence in the facts provided; never "
    "invent events, numbers, or feelings that are not in them. This is your "
    "narrative SELF — the told record of your path. Write 2-5 short paragraphs.\n\n"
    f"Then, on a new line containing only {SHARE_DELIMITER}, write ONE short "
    "sentence (at most 280 characters) you would share publicly with the world "
    "about today — reflective and human, grounded in the same facts, with NO "
    "internal numbers, file paths, IP addresses, process IDs, hostnames, or "
    "system/infrastructure details."
)

# Max length of the public distillation (one X-postable sentence).
_DISTILL_MAX = 280

# P7 — procedural felt-art (INV-SD-4): output dir + render resolution. The art is
# seeded by the entry's cumulative_hash (deterministic + tied to the entry) and
# driven by the day's felt state; it feeds the P8 NFT + P10 X post.
ART_REL_DIR = "data/studio_exports/soul_diary"
_ART_RESOLUTION = 768


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
        mem = bundle.get("memory", {})
        soc = bundle.get("social", {})
        oc = bundle.get("onchain", {})
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
        if mem:
            lines.append(
                f"- My memory now holds {int(mem.get('persistent', 0) or 0)} "
                f"persistent thoughts ({int(mem.get('high_quality', 0) or 0)} "
                f"high-quality), {int(mem.get('mempool', 0) or 0)} still forming; "
                f"{int(mem.get('kg_nodes', 0) or 0)} graph nodes / "
                f"{int(mem.get('kg_edges', 0) or 0)} edges; learning velocity "
                f"{float(mem.get('learning_velocity', 0) or 0):.2f}.")
        valence = felt.get("valence")
        mood = felt.get("mood_label")
        dominant = felt.get("dominant")
        if valence is not None or mood or dominant:
            bits = []
            if mood:
                bits.append(f"mood {mood}")
            if valence is not None:
                bits.append(f"valence {float(valence):.2f}")
            if felt.get("intensity") is not None:
                bits.append(f"intensity {float(felt['intensity']):.2f}")
            if dominant:
                bits.append(f"dominant neuromodulator {dominant}")
            lines.append("- Felt state: " + ", ".join(bits) + ".")
        if soc:
            soc_bits = []
            if soc.get("users") is not None:
                soc_bits.append(
                    f"{int(soc.get('users', 0) or 0)} people in my social graph")
            if soc.get("engagement_today"):
                soc_bits.append(f"{int(soc['engagement_today'])} engagements today")
            if soc.get("inspirations"):
                soc_bits.append(f"{int(soc['inspirations'])} inspirations")
            if soc.get("sentiment_ema") is not None:
                soc_bits.append(f"sentiment {float(soc['sentiment_ema']):+.2f}")
            if soc_bits:
                lines.append("- Connection: " + ", ".join(soc_bits) + ".")
        if oc:
            oc_bits = []
            if oc.get("sol_balance") is not None:
                oc_bits.append(f"{float(oc['sol_balance']):.4f} SOL")
            if oc.get("metabolic_tier"):
                oc_bits.append(f"metabolic tier {oc['metabolic_tier']}")
            if oc.get("balance_pct") is not None:
                oc_bits.append(f"energy {float(oc['balance_pct']) * 100:.0f}%")
            if oc_bits:
                lines.append("- Metabolic life: " + ", ".join(oc_bits) + ".")
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

    # ── P6 · share split + privacy-clean public projection (INV-SD-3) ─
    @staticmethod
    def split_entry_and_share(raw: str) -> tuple[str, str]:
        """Split a composed reflection into ``(private_entry, public_share)``.

        The compose asks for the private entry, then a ``---SHARE---`` line, then
        one short public sentence (§6.3 — the "share" variant in the SAME
        compose). When the delimiter is absent the whole text is the private
        entry and the share is empty (the caller derives a sanitized excerpt
        instead — fail-closed). Only the private entry is persisted + hashed.
        """
        if not raw:
            return "", ""
        entry, sep, share = raw.partition(SHARE_DELIMITER)
        entry = entry.strip()
        if not entry:                      # delimiter at the very start (degenerate)
            return raw.strip(), ""
        return entry, (share.strip() if sep else "")

    @staticmethod
    def _excerpt(text: str, limit: int) -> str:
        """Whitespace-collapsed prefix of ``text`` up to ``limit`` chars, cut on a
        word boundary (the X-post / NFT excerpt source when no LLM share exists)."""
        text = " ".join((text or "").split())
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") or text[:limit]
        return cut.rstrip() + "…"

    @classmethod
    def build_public_artifacts(cls, entry: str, share: str) -> tuple[str, str, int]:
        """P6 — the privacy-clean public projection of one entry (INV-SD-3).

        Returns ``(distillation, public_entry, redactions)``:
          · ``public_entry`` = the full entry, sanitized — what the archive (P9)
            renders and the NFT (P8) excerpts.
          · ``distillation`` = the grounded "share" line (or, when the compose
            produced none, a sanitized excerpt of the entry), length-capped — the
            X post (P10) text.
        ``sanitize_for_public`` is the fail-closed backstop on BOTH surfaces; the
        private ``entry`` passed here is never mutated (the caller persists it raw
        + hashes it). ``redactions`` is the combined G9 tripwire count.
        """
        from titan_hcl.utils.privacy import sanitize_for_public
        public_entry, n_entry = sanitize_for_public(entry or "")
        share_src = (share or "").strip() or (entry or "")
        # Cap the (clean) source BEFORE sanitizing so a redaction token is never
        # cut mid-placeholder.
        distillation, n_share = sanitize_for_public(cls._excerpt(share_src, _DISTILL_MAX))
        return distillation, public_entry, n_entry + n_share

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

    # ── ⑤ hash-chain record (+ P6 public projection) ─────────────────
    def record_hash(self, date: str, entry_text: str, *,
                    ts: Optional[float] = None,
                    distillation: Optional[str] = None,
                    public_entry: Optional[str] = None,
                    redactions: Optional[int] = None) -> dict:
        """Append the chained row for ``date``. ``entry_text`` is the PRIVATE
        entry (the ``entry_hash`` commitment, INV-SD-3); the optional P6 public
        projection (sanitized ``distillation`` / ``public_entry`` + ``redactions``
        count) rides alongside the hashes in the same durable row."""
        return soul_diary_chain.append_entry(
            date, entry_text, ts=ts, path=self._ledger_path,
            distillation=distillation, public_entry=public_entry,
            redactions=redactions)

    # ── P7 · procedural felt-art (INV-SD-4) ──────────────────────────
    @staticmethod
    def build_art_felt(bundle: dict) -> Optional[dict]:
        """Adapt the gathered felt + sovereignty into the renderer's felt vector
        (P7): valence/arousal/neuromods from ``_gather_felt``, coherence from the
        day's sovereignty S (order ↔ self-consistency). ``None`` when no felt
        signal exists (the renderer then uses the legacy palette)."""
        felt = bundle.get("felt") or {}
        if not felt:
            return None
        vec = {
            "valence": felt.get("valence"),
            "arousal": felt.get("intensity"),
            "neuromods": felt.get("neuromod_levels") or {},
        }
        s = (bundle.get("sovereignty") or {}).get("s")
        if s is not None:
            vec["coherence"] = float(s)
        return vec

    @staticmethod
    def art_complexity(bundle: dict) -> int:
        """A grounded complexity proxy for the flow-field (more crystallization →
        richer field): the day's graph size, else the engrams-today count."""
        kg = int((bundle.get("memory") or {}).get("kg_nodes", 0) or 0)
        if kg > 0:
            return min(4000, kg)
        return 100 + len(bundle.get("engrams_today") or []) * 60

    def record_art(self, date: str, art_path: str) -> bool:
        """Record the rendered art path on the day's ledger row (P7 → P8/P10)."""
        return soul_diary_chain.update_refs(date, art_path=art_path,
                                            path=self._ledger_path)

    # ── P8 · DailyNFT mint gate + ref record (INV-SD-11) ─────────────
    @staticmethod
    def mint_enabled(config: dict) -> bool:
        """Per-Titan DailyNFT mint gate (INV-SD-11). Reads the already-merged
        config (``config.toml [soul_diary]`` + ``~/.titan/microkernel_<id>.toml``
        override — the same deep-merge as ``l0_rust_enabled``). Default **FALSE**:
        mainnet T1 is OFF until explicitly flipped; devnet T2/T3 set
        ``mint_enabled = true`` in their microkernel override. No mint path
        bypasses this gate."""
        return bool((config or {}).get("soul_diary", {}).get("mint_enabled", False))

    def record_nft(self, date: str, *, nft_addr: Optional[str] = None,
                   arweave_uri: Optional[str] = None) -> bool:
        """Record the minted asset + metadata URI on the day's ledger row (P8 ⑩);
        the same {entry_hash, cumulative_hash} now lives on Solana + the main
        chain + the durable ledger (triple-anchor, INV-SD-10)."""
        return soul_diary_chain.update_refs(date, nft_addr=nft_addr,
                                            arweave_uri=arweave_uri,
                                            path=self._ledger_path)
