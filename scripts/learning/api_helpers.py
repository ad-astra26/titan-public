"""
scripts/learning/api_helpers.py — Shared HTTP API helpers for Learning TestSuite.

All communication with Titan happens via HTTP API endpoints.
The TestSuite is a standalone process — never touches bus or spirit_worker internals.
"""
import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("testsuite.api")

INTERNAL_KEY = os.getenv(
    "TITAN_INTERNAL_KEY", "")

# Layer dimensions for perturbation padding
LAYER_DIMS = {
    "inner_body": 5, "inner_mind": 15, "inner_spirit": 45,
    "outer_body": 5, "outer_mind": 15, "outer_spirit": 45,
}
LAYER_ORDER = list(LAYER_DIMS.keys())


def _headers() -> dict:
    return {
        "X-Titan-Internal-Key": INTERNAL_KEY,
        "X-Titan-User-Id": "testsuite",
        "Content-Type": "application/json",
    }


async def is_titan_alive(client: httpx.AsyncClient, api: str) -> bool:
    """Quick health check."""
    try:
        r = await client.get(f"{api}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


async def get_titan_state(client: httpx.AsyncClient, api: str) -> dict:
    """Full state snapshot: neuromods, Chi, self-exploration, pi-heartbeat, vocab."""
    state = {}
    try:
        r = await client.get(f"{api}/v4/inner-trinity", headers=_headers(), timeout=10)
        raw = r.json()
        coord = raw.get("data", raw)  # Unwrap {"status": "ok", "data": {...}}

        # Neuromodulators
        nm = coord.get("neuromodulators", {})
        mods = nm.get("modulators", {})
        state["emotion"] = nm.get("current_emotion", "peace")
        state["emotion_confidence"] = nm.get("emotion_confidence", 0)
        for name in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA"):
            m = mods.get(name, {})
            state[name.lower()] = m.get("level", 0.5) if isinstance(m, dict) else 0.5

        # Pi-heartbeat
        pi = coord.get("pi_heartbeat", {})
        state["dev_age"] = pi.get("developmental_age", 0)
        state["in_cluster"] = pi.get("in_cluster", False)
        state["pi_ratio"] = pi.get("heartbeat_ratio", 0)

        # Dreaming
        dr = coord.get("dreaming", {})
        state["is_dreaming"] = dr.get("is_dreaming", False)

        # Expression composites
        ec = coord.get("expression_composites", {})
        state["expression_fires"] = sum(
            c.get("fire_count", 0)
            for c in ec.get("composites", {}).values()
            if isinstance(c, dict)
        )

        # Phase events (for reactive complement)
        pe = coord.get("phase_events", {})
        state["current_phase"] = pe.get("current_phase", "idle")
        state["recent_events"] = pe.get("recent_events", [])

        # Experience memory (action history)
        ex = coord.get("experience_memory", {})
        state["total_actions"] = ex.get("total", 0)

        # Epoch
        state["tick_count"] = coord.get("tick_count", 0)

    except Exception as e:
        logger.debug("get_titan_state coordinator error: %s", e)

    # Chi
    try:
        r = await client.get(f"{api}/v4/chi", headers=_headers(), timeout=10)
        chi = r.json().get("data", {})
        state["chi_total"] = chi.get("total", 0.5)
        state["chi_circulation"] = chi.get("circulation", 0)
        state["chi_state"] = chi.get("state", "?")
    except Exception as e:
        logger.debug("get_titan_state chi error: %s", e)

    # Self-exploration
    try:
        r = await client.get(f"{api}/v4/self-exploration", headers=_headers(), timeout=10)
        se = r.json().get("data", {})
        state["se_mode"] = se.get("mode", "SELF_EXPLORE")
        state["se_actions"] = se.get("total_actions_processed", 0)
    except Exception as e:
        logger.debug("get_titan_state self-exploration error: %s", e)

    # Vocabulary
    try:
        r = await client.get(f"{api}/v4/vocabulary", headers=_headers(), timeout=10)
        data = r.json()
        words = data.get("data", data)
        if isinstance(words, dict):
            wlist = words.get("words", [])
            state["vocab_size"] = len(wlist) if isinstance(wlist, list) else 0
            state["vocab_words"] = wlist if isinstance(wlist, list) else []
        elif isinstance(words, list):
            state["vocab_size"] = len(words)
            state["vocab_words"] = words
        else:
            state["vocab_size"] = 0
            state["vocab_words"] = []
    except Exception as e:
        logger.debug("get_titan_state vocab error: %s", e)
        state["vocab_size"] = 0
        state["vocab_words"] = []

    return state


async def get_epoch_id(client: httpx.AsyncClient, api: str) -> int:
    """Get current consciousness epoch ID."""
    try:
        r = await client.get(
            f"{api}/status/consciousness/history",
            params={"limit": 1}, headers=_headers(), timeout=15)
        data = r.json()
        epochs = data.get("data", data)
        if isinstance(epochs, list) and epochs:
            return epochs[0].get("epoch_id", 0)
        if isinstance(epochs, dict):
            eps = epochs.get("epochs", [])
            if eps:
                return eps[0].get("epoch_id", 0)
    except Exception:
        pass
    return 0


async def wait_for_new_epoch(client: httpx.AsyncClient, api: str,
                              current_epoch_id: int,
                              timeout_s: float = 300) -> int:
    """Wait until consciousness epoch advances. Returns new epoch_id."""
    start = time.time()
    while time.time() - start < timeout_s:
        new_id = await get_epoch_id(client, api)
        if new_id > current_epoch_id:
            return new_id
        await asyncio.sleep(10)
    return await get_epoch_id(client, api)


async def inject_word_stimulus(client: httpx.AsyncClient, api: str,
                                word: str, perturbation: dict,
                                pass_type: str = "feel",
                                hormone_stimuli: dict = None) -> dict:
    """Inject word learning stimulus via /v4/experience-stimulus."""
    # Pad perturbation to correct dimensions
    padded = {}
    for layer, dim in LAYER_DIMS.items():
        vals = perturbation.get(layer, [])
        if isinstance(vals, list):
            padded[layer] = (vals + [0.0] * dim)[:dim]
        else:
            padded[layer] = [0.0] * dim

    payload = {
        "word": word,
        "pass_type": pass_type,
        "perturbation": padded,
    }
    if hormone_stimuli:
        payload["hormone_stimuli"] = hormone_stimuli

    try:
        r = await client.post(
            f"{api}/v4/experience-stimulus",
            json=payload,
            headers=_headers(),
            timeout=30,
        )
        return r.json()
    except Exception as e:
        logger.warning("inject_word_stimulus error: %s", e)
        return {"error": str(e)}


async def send_chat(client: httpx.AsyncClient, api: str,
                     message: str, user_id: str = "testsuite") -> str:
    """Send chat message via /chat with internal key auth."""
    try:
        headers = _headers()
        headers["X-Titan-User-Id"] = user_id
        r = await client.post(
            f"{api}/chat",
            json={"message": message, "user_id": user_id},
            headers=headers,
            timeout=30,
        )
        data = r.json()
        return data.get("response", data.get("data", {}).get("response", ""))
    except Exception as e:
        logger.warning("send_chat error: %s", e)
        return ""


async def get_vocabulary_list(client: httpx.AsyncClient, api: str) -> list[dict]:
    """Get vocabulary words with confidence and learning phase."""
    try:
        r = await client.get(f"{api}/v4/vocabulary", headers=_headers(), timeout=10)
        data = r.json()
        words = data.get("data", data)
        if isinstance(words, dict):
            return words.get("words", [])
        if isinstance(words, list):
            return words
    except Exception:
        pass
    return []


async def update_word_learning(
    client: httpx.AsyncClient, api: str,
    word: str, word_type: str, pass_type: str,
    score: float, stage: int = 1,
    felt_tensor: list = None,
    hormone_pattern: dict = None,
) -> dict:
    """Update vocabulary learning progress after a teaching/reinforcement pass."""
    payload = {
        "word": word,
        "word_type": word_type,
        "pass_type": pass_type,
        "score": score,
        "stage": stage,
    }
    if felt_tensor:
        payload["felt_tensor"] = felt_tensor
    if hormone_pattern:
        payload["hormone_pattern"] = hormone_pattern

    try:
        r = await client.post(
            f"{api}/v4/vocabulary/update-learning",
            json=payload,
            headers=_headers(),
            timeout=10,
        )
        return r.json()
    except Exception as e:
        logger.debug("update_word_learning error for '%s': %s", word, e)
        return {"error": str(e)}


def load_word_recipes() -> dict:
    """Load all word recipes from word_resonance JSON files."""
    recipes = {}
    for fname in ("data/word_resonance.json",
                   "data/word_resonance_phase2.json",
                   "data/word_resonance_phase3.json"):
        try:
            with open(fname) as f:
                data = json.load(f)
            for key, val in data.items():
                if not key.startswith("_"):
                    recipes[key.lower()] = val
                    recipes[key.lower()]["word"] = key.lower()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("load_word_recipes error for %s: %s", fname, e)
    return recipes


def scale_perturbation(perturbation: dict, scale: float) -> dict:
    """Scale perturbation values by a factor (e.g., 0.3 for recognize pass)."""
    result = {}
    for layer, dim in LAYER_DIMS.items():
        vals = perturbation.get(layer, [])
        if isinstance(vals, list):
            scaled = [v * scale for v in vals[:dim]]
            scaled += [0.0] * (dim - len(scaled))
            result[layer] = scaled
        else:
            result[layer] = [0.0] * dim
    return result
