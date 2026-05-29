"""
titan_hcl.llm_pipeline.state_gather — SHM/DB-direct readers for the
inputs needed by compose_pre() and verify_post().

Designed to be callable from ANY context: agno_worker subprocess, llm_worker,
social_x_gateway (in-parent), api_subprocess, autonomous_language_pipeline
script. Reads from canonical sources via async helpers so no in-process
plugin reference is required.

Sources:
  - felt_state         → consciousness.db `epochs.state_vector` (most recent row)
  - vocabulary         → inner_memory.db `vocabulary` (top 128 by confidence)
  - chain_state.neuromods → caller-provided OR neuromod_state.bin SHM (D-SPEC-54)
  - chain_state.vocab_size + composition_level → caller-provided OR language SHM
  - chain_state.i_confidence → caller-provided OR msl SHM

Fallback semantics: every gather function returns SENSIBLE DEFAULTS on
read failure (empty list / empty dict / OVG default values). Callers do
NOT need to wrap gather calls in try/except — the pipeline degrades to
"no felt-state sentence prepended; OVG verifies with default chain state"
rather than blocking the chat path.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Default chain_state used when neither caller-provided state nor SHM
# reads succeed. Values match the legacy chat.py:303-355 inline fallbacks
# (0.5 per neuromod, vocab_size=300, composition_level=8, i_confidence=0.9).
_DEFAULT_NEUROMODS = {
    "DA": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5,
    "Endorphin": 0.5, "GABA": 0.5,
}
_DEFAULT_VOCAB_SIZE = 300
_DEFAULT_COMPOSITION_LEVEL = 8
_DEFAULT_I_CONFIDENCE = 0.9


async def gather_felt_state_and_vocab(
    consciousness_db_path: str = "./data/consciousness.db",
    inner_memory_db_path: str = "./data/inner_memory.db",
    vocab_limit: int = 128,
    min_confidence: float = 0.1,
) -> tuple[list, list]:
    """Read latest 130D-132D felt_state + top-N vocabulary rows.

    Mirrors `titan_hcl.api.dashboard._get_dialogue_state()` so call
    sites migrating to this library see byte-identical behavior. Both
    DBs are file-backed — readable from any process; no plugin instance
    needed.

    Returns:
        (felt_state, vocabulary). Empty lists if either DB read fails.
    """
    felt_state: list = []
    vocabulary: list[dict] = []

    try:
        from titan_hcl.core import sqlite_async
    except Exception as e:
        logger.debug("[llm_pipeline.state_gather] sqlite_async import failed: %s", e)
        return felt_state, vocabulary

    # 1. felt-state from consciousness DB
    try:
        row = await sqlite_async.query(
            consciousness_db_path,
            "SELECT state_vector FROM epochs ORDER BY epoch_id DESC LIMIT 1",
            fetch="one",
        )
        if row and row[0]:
            # SPEC §11.H.1.bis dual-read: BLOB f32-LE (new) or TEXT-JSON (legacy)
            from titan_hcl.logic.consciousness import unpack_vector
            felt_state = unpack_vector(row[0])
    except Exception as e:
        logger.debug(
            "[llm_pipeline.state_gather] felt_state read failed: %s", e
        )

    # 2. vocabulary from inner_memory DB
    try:
        rows = await sqlite_async.query(
            inner_memory_db_path,
            "SELECT word, word_type, confidence, felt_tensor FROM vocabulary "
            "WHERE confidence > ? ORDER BY confidence DESC LIMIT ?",
            (min_confidence, vocab_limit),
        )
        for vr in rows or []:
            ft = None
            if vr[3]:
                try:
                    ft = json.loads(vr[3]) if isinstance(vr[3], str) else vr[3]
                except Exception:
                    pass
            vocabulary.append({
                "word": vr[0],
                "word_type": vr[1],
                "confidence": vr[2],
                "felt_tensor": ft,
            })
    except Exception as e:
        logger.debug(
            "[llm_pipeline.state_gather] vocabulary read failed: %s", e
        )

    return felt_state, vocabulary


def build_hormone_shifts(input_signal: dict) -> dict[str, float]:
    """Map InputExtractor signal → DialogueComposer hormone_shifts.

    Verbatim port of the inline block in api/chat.py:362-376 + sibling sites.
    Used to drive intent detection (empathize / ask_question / share_insight
    / respond_feeling) inside DialogueComposer.

    Args:
        input_signal: dict from InputExtractor.extract(message, user_id),
                      containing at least 'valence' and 'engagement' keys.

    Returns:
        Dict of {hormone_name: shift_delta}.
    """
    valence = float(input_signal.get("valence", 0.0))
    engagement = float(input_signal.get("engagement", 0.0))
    return {
        "EMPATHY": max(0.0, valence) * 0.2,
        "CURIOSITY": engagement * 0.2,
        "CREATIVITY": 0.0,
        "REFLECTION": max(0.0, -valence) * 0.1,
    }


def gather_chain_state(
    coordinator_snapshot: Optional[dict] = None,
    override: Optional[dict] = None,
) -> dict[str, Any]:
    """Assemble OVG chain_state (Proof of Qualia) dict.

    Two-tier source:
      1. If `override` is supplied (caller has SHM-direct readings), use it.
      2. Else if `coordinator_snapshot` supplied (parent process or
         dashboard cache), extract neuromods + vocab + msl from it.
      3. Else return defaults — OVG still verifies with reasonable values,
         signature still attaches.

    Verbatim port of the inline block in api/chat.py:303-330 + agno_hooks
    PostHook + chat_pipeline.run_chat. Centralizing here means OVG calls
    fleet-wide get the SAME shape — no drift between sites.

    Args:
        coordinator_snapshot: dict from `dashboard._get_cached_coordinator(plugin)`
                              (in-parent contexts). May contain nested keys:
                              `neuromodulators.modulators.{mod}.level`,
                              `language.vocab_total`, `language.composition_level`,
                              `msl.i_confidence`.
        override: Pre-gathered chain_state dict (worker contexts that
                  read SHM directly). If supplied, returned as-is.

    Returns:
        chain_state dict with keys: neuromods (dict), vocab_size (int),
        composition_level (int), i_confidence (float).
    """
    if override is not None:
        return dict(override)

    chain_state: dict[str, Any] = {
        "neuromods": dict(_DEFAULT_NEUROMODS),
        "vocab_size": _DEFAULT_VOCAB_SIZE,
        "composition_level": _DEFAULT_COMPOSITION_LEVEL,
        "i_confidence": _DEFAULT_I_CONFIDENCE,
    }

    if not coordinator_snapshot:
        return chain_state

    try:
        nm_dict = (
            coordinator_snapshot.get("neuromodulators", {})
            .get("modulators", {})
        )
        if isinstance(nm_dict, dict) and nm_dict:
            chain_state["neuromods"] = {
                k: float(v.get("level", 0.5))
                for k, v in nm_dict.items()
                if isinstance(v, dict)
            }
    except Exception:
        pass

    try:
        lang = coordinator_snapshot.get("language", {})
        if isinstance(lang, dict):
            chain_state["vocab_size"] = int(
                lang.get("vocab_total", _DEFAULT_VOCAB_SIZE)
            )
            chain_state["composition_level"] = int(
                lang.get("composition_level", _DEFAULT_COMPOSITION_LEVEL)
            )
    except Exception:
        pass

    try:
        msl = coordinator_snapshot.get("msl", {})
        if isinstance(msl, dict):
            chain_state["i_confidence"] = float(
                msl.get("i_confidence", _DEFAULT_I_CONFIDENCE)
            )
    except Exception:
        pass

    return chain_state


def default_chain_state() -> dict[str, Any]:
    """Return the documented OVG chain_state default dict.

    Useful for tests and for callers that explicitly want "no neuromod
    influence" verification.
    """
    return {
        "neuromods": dict(_DEFAULT_NEUROMODS),
        "vocab_size": _DEFAULT_VOCAB_SIZE,
        "composition_level": _DEFAULT_COMPOSITION_LEVEL,
        "i_confidence": _DEFAULT_I_CONFIDENCE,
    }
