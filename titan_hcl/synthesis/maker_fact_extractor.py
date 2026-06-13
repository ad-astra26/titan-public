"""Maker-fact extractor — the gated post-turn producer of Maker facts
(RFP_missions_and_the_maker_model §7.1).

When the MAKER is speaking (is_maker — by default the app's paired device, or any
channel the maker_engine attributes to the Maker), a completed turn is a chance to
learn something durable about him. To keep it cheap, a two-stage gate:

  1. `looks_like_self_disclosure(text)` — a near-zero-cost heuristic pre-filter. Most
     turns (questions, commands, opinions-about-others) carry no self-disclosure and
     are skipped WITHOUT an LLM call.
  2. `extract_maker_facts(text, llm_fn)` — only for the survivors: ONE small LLM call
     that returns a JSON array of durable self-facts `[{category, value, confidence}]`.

The facts flow to `MakerStore.record_fact(provenance="maker-told", …)` — sovereign,
provenance-tagged, confidence<1.0. Pure + LLM-injected so the gate + parse are unit-
tested without a network; the daemon (`maker_fact_loop`) wires the real provider.

Identity note: attribution is the existing system's job — `maker_engine.is_maker`
(against the configured Maker platform ids) + the person-clustering already in the
graph. The app channel is the Maker by default (paired device); a non-Maker turn
(is_maker False — e.g. someone who introduced themselves) never reaches this extractor.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Cheap pre-filter: first-person self-reference + a stative/biographical cue. Tuned to
# fire on "I'm a software architect" / "my job is …" / "I grew up in …" and skip pure
# questions/commands. The LLM does the real discrimination + returns [] if nothing
# durable — this only spares the obvious non-disclosures an LLM call.
_SELF_DISCLOSURE = re.compile(
    r"\b(i'?m\b|i am\b|i was\b|i'?ve\b|i have\b|i had\b|i like\b|i love\b|i enjoy\b|"
    r"i hate\b|i prefer\b|i work\b|i worked\b|i live\b|i lived\b|i grew up\b|i study\b|"
    r"i studied\b|i learned\b|i can\b|i'?ll\b|i'?d\b|my name\b|my job\b|my work\b|"
    r"my favou?rite\b|my hometown\b|my partner\b|my wife\b|my husband\b|my kids?\b|"
    r"my family\b|i'?m from\b|i come from\b)",
    re.IGNORECASE)

_EXTRACT_SYSTEM = (
    "You extract durable personal facts a person states about THEMSELVES. "
    "Output ONLY a JSON array, nothing else.")


def looks_like_self_disclosure(text: str) -> bool:
    """Near-zero-cost gate: does this turn plausibly state a durable fact about the
    speaker? Skips the LLM on the overwhelming majority of turns that don't."""
    return bool(_SELF_DISCLOSURE.search(text or ""))


def build_extraction_prompt(text: str) -> str:
    return (
        "From the message below, extract durable facts the speaker states about "
        "THEMSELVES (occupation, location, hobby, preference, relationship, skill, "
        "background, etc.). IGNORE transient state, questions, requests, opinions "
        "about others, and anything not about the speaker. Use a short snake_case "
        "category. Output ONLY a JSON array of "
        '{"category": "...", "value": "...", "confidence": <0.0-1.0>} — '
        "an empty array [] if there are no durable self-facts.\n\nMessage:\n"
        + (text or "").strip())


def _parse_facts(raw: str) -> list[dict]:
    """Robustly pull the JSON array out of an LLM reply (tolerates code fences /
    surrounding prose). Returns only well-formed {category, value, confidence}."""
    s = (raw or "").strip()
    if "```" in s:
        s = re.sub(r"```(?:json)?", "", s).strip("` \n")
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    try:
        arr = json.loads(s[i:j + 1])
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    if isinstance(arr, list):
        for o in arr:
            if not isinstance(o, dict):
                continue
            cat = str(o.get("category", "") or "").strip()
            val = str(o.get("value", "") or "").strip()
            if not cat or not val:
                continue
            try:
                conf = float(o.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            out.append({"category": cat, "value": val,
                        "confidence": max(0.0, min(1.0, conf))})
    return out


def extract_maker_facts(text: str, llm_fn: Callable[[str], str]) -> list[dict]:
    """One LLM call → durable self-facts. `llm_fn(prompt) -> str`. Soft (returns []
    on any failure). Caller must have pre-filtered with `looks_like_self_disclosure`."""
    try:
        raw = llm_fn(build_extraction_prompt(text)) or ""
    except Exception as e:  # noqa: BLE001
        logger.debug("[MakerFactExtractor] llm call failed: %s", e)
        return []
    return _parse_facts(raw)


def maker_fact_loop(queue, maker_store, llm_fn, stop_event,
                    interval_s: float = 20.0, per_pass_cap: int = 8) -> None:
    """Off-hot-path daemon (sibling of the turn-judge): drain Maker turns, gate, and
    write extracted facts to the Maker model. LLM is network I/O on its own thread →
    no GIL/heartbeat starvation. Every step soft — a failure must never break chat."""
    while not stop_event.wait(interval_s):
        n = 0
        try:
            while queue and n < per_pass_cap:
                item = queue.popleft()
                n += 1
                text = str((item or {}).get("prompt", "") or "").strip()
                if not text or not looks_like_self_disclosure(text):
                    continue
                for f in extract_maker_facts(text, llm_fn):
                    try:
                        maker_store.record_fact(
                            category=f["category"], value=f["value"],
                            provenance="maker-told", confidence=f["confidence"],
                            source_turn=text[:280])
                    except Exception as e:  # noqa: BLE001
                        logger.debug("[MakerFactExtractor] record_fact soft-fail: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.debug("[MakerFactExtractor] loop pass soft-fail: %s", e)
