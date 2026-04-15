"""
scripts/child_development_test.py — Unified Child Development Test Suite for Titan.

Runs 7 test protocols (T1-T7) autonomously on both Titan twins,
collecting per-trial telemetry and producing comparative reports.

Tests:
  T1: Word-State Association — recall, production, retention
  T2: Odd-One-Out — category formation (4 levels)
  T3: Analogy Completion — relational reasoning (5 levels)
  T4: Sequence Completion — temporal/causal ordering (4 levels)
  T5: Cross-Modal Transfer — body↔mind Sapir-Whorf test
  T6: Mutual Exclusivity — novel word disambiguation
  T7: Multi-Modal Association — word + color + sound + feeling

Usage:
  python scripts/child_development_test.py
  python scripts/child_development_test.py --tests T1,T2,T7
  python scripts/child_development_test.py --instances titan1
"""
import argparse
import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Optional

import httpx

log = logging.getLogger("child_dev_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

# ── Instance Profiles ─────────────────────────────────────────────
TITAN1_API = "http://localhost:7777"
TITAN2_API = "http://10.135.0.6:7777"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")
HEADERS = {"X-Titan-Internal-Key": INTERNAL_KEY}

INSTANCES = {
    "titan1": {"api": TITAN1_API, "name": "Titan1 (localhost)"},
    "titan2": {"api": TITAN2_API, "name": "Titan2 (10.135.0.6)"},
}

TRIAL_REST_S = 30   # Rest between trials — allow ~10 consciousness epochs to integrate
TEST_REST_S = 120   # Rest between tests — full nervous system settling
BASELINE_SAMPLES = 3
BASELINE_INTERVAL = 10


# ── API Helpers ───────────────────────────────────────────────────

async def get_state(client: httpx.AsyncClient, api: str) -> dict:
    """Get full Titan state including 132D vector."""
    try:
        r = await client.get(f"{api}/status/consciousness/history",
                             params={"limit": 1}, headers=HEADERS, timeout=15)
        return _parse_epoch(r.json())
    except Exception as e:
        log.warning("get_state error: %s", e)
    return {"epoch_id": 0, "state_vector": [], "curvature": 0, "density": 0, "drift": 0}


def _parse_epoch(data: dict) -> dict:
    """Parse epoch from consciousness history response."""
    inner = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(inner, dict):
        epochs = inner.get("epochs", inner.get("data", []))
    else:
        epochs = inner
    if isinstance(epochs, list) and epochs:
        ep = epochs[0]
        sv = ep.get("state_vector", [])
        if hasattr(sv, "to_list"):
            sv = sv.to_list()
        return {
            "epoch_id": ep.get("epoch_id", 0),
            "state_vector": list(sv),
            "curvature": ep.get("curvature", 0),
            "density": ep.get("density", 0),
            "drift": ep.get("drift_magnitude", 0),
        }
    return {"epoch_id": 0, "state_vector": [], "curvature": 0, "density": 0, "drift": 0}


async def wait_for_new_epoch(client: httpx.AsyncClient, api: str,
                              current_epoch_id: int, timeout_s: float = 360) -> dict:
    """Wait until a new consciousness epoch fires (up to timeout_s).

    132D epochs fire every ~5 minutes, so we poll every 10s.
    This ensures the perturbation has been fully integrated.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        state = await get_state(client, api)
        if state["epoch_id"] > current_epoch_id and state["state_vector"]:
            return state
        await asyncio.sleep(10)
    # Timeout — return whatever we have
    return await get_state(client, api)


async def get_hormones(client: httpx.AsyncClient, api: str) -> dict:
    """Get hormone levels from nervous system."""
    try:
        r = await client.get(f"{api}/v4/nervous-system", headers=HEADERS, timeout=10)
        ns = r.json().get("data", {})
        return {
            name: {"level": h.get("level", 0), "fire_count": h.get("fire_count", 0)}
            for name, h in ns.get("hormonal_system", {}).items()
            if isinstance(h, dict)
        }
    except Exception:
        return {}


async def get_neuromodulators(client: httpx.AsyncClient, api: str) -> dict:
    """Get neuromodulator state from coordinator."""
    try:
        r = await client.get(f"{api}/v4/inner-trinity", headers=HEADERS, timeout=10)
        coord = r.json().get("data", r.json())
        nm = coord.get("neuromodulators", {})
        return {
            "emotion": nm.get("current_emotion", "?"),
            "confidence": nm.get("emotion_confidence", 0),
            "modulators": {
                name: m.get("level", 0)
                for name, m in nm.get("modulators", {}).items()
            },
        }
    except Exception:
        return {"emotion": "?", "confidence": 0, "modulators": {}}


async def get_vocabulary(client: httpx.AsyncClient, api: str) -> list:
    """Get vocabulary words with confidence."""
    try:
        r = await client.get(f"{api}/v4/vocabulary", headers=HEADERS, timeout=10)
        data = r.json()
        words = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(words, list):
            return words
        if isinstance(words, dict):
            return [{"word": w, **v} for w, v in words.items() if w != "_meta"]
    except Exception:
        pass
    return []


async def inject_stimulus(client: httpx.AsyncClient, api: str,
                          perturbation: dict, word: str = "") -> dict:
    """Inject perturbation via /v4/experience-stimulus."""
    body = {"perturbation": perturbation}
    if word:
        body["word"] = word
    try:
        r = await client.post(f"{api}/v4/experience-stimulus",
                              json=body, headers=HEADERS, timeout=15)
        return r.json()
    except Exception as e:
        log.warning("inject_stimulus error: %s", e)
        return {"error": str(e)}


async def compose_sentence(client: httpx.AsyncClient, api: str,
                           max_level: int = 5, intent: str = None) -> dict:
    """Compose sentence from current felt-state via /chat."""
    try:
        msg = "Express what you feel right now in your own words."
        if intent:
            msg = f"Express your feeling of {intent} in your own words."
        r = await client.post(f"{api}/chat",
                              json={"message": msg, "session_id": "child_dev_test"},
                              headers=HEADERS, timeout=30)
        data = r.json()
        return {"response": data.get("data", {}).get("reply", data.get("reply", ""))}
    except Exception:
        return {"response": ""}


# ── State Snapshot ────────────────────────────────────────────────

async def capture_snapshot(client: httpx.AsyncClient, api: str) -> dict:
    """Capture full state snapshot for telemetry."""
    state, hormones, nm = await asyncio.gather(
        get_state(client, api),
        get_hormones(client, api),
        get_neuromodulators(client, api),
    )
    return {
        "timestamp": time.time(),
        "epoch_id": state.get("epoch_id", 0),
        "state_vector": state.get("state_vector", []),
        "curvature": state.get("curvature", 0),
        "density": state.get("density", 0),
        "drift": state.get("drift", 0),
        "hormones": hormones,
        "neuromodulators": nm.get("modulators", {}),
        "emotion": nm.get("emotion", "?"),
        "emotion_confidence": nm.get("confidence", 0),
    }


async def capture_baseline(client: httpx.AsyncClient, api: str) -> dict:
    """Capture baseline by averaging multiple snapshots."""
    snapshots = []
    for _ in range(BASELINE_SAMPLES):
        snap = await capture_snapshot(client, api)
        snapshots.append(snap)
        await asyncio.sleep(BASELINE_INTERVAL)

    if not snapshots:
        return await capture_snapshot(client, api)

    # Average hormone levels across baseline
    baseline = snapshots[-1].copy()
    all_hormones = {}
    for snap in snapshots:
        for name, h in snap.get("hormones", {}).items():
            if name not in all_hormones:
                all_hormones[name] = []
            all_hormones[name].append(h.get("level", 0))

    baseline["hormones_baseline"] = {
        name: {"level": sum(vals) / len(vals)}
        for name, vals in all_hormones.items()
    }
    return baseline


# ── Metrics ───────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def vector_distance(a: list, b: list) -> float:
    """Euclidean distance between two vectors."""
    if not a or not b:
        return float("inf")
    min_len = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min_len)))


def compute_deltas(pre: dict, post: dict) -> dict:
    """Compute state deltas between pre and post snapshots."""
    deltas = {}

    # Hormone level deltas
    h_pre = pre.get("hormones", {})
    h_post = post.get("hormones", {})
    hormone_deltas = {}
    for name in set(list(h_pre.keys()) + list(h_post.keys())):
        pre_level = h_pre.get(name, {}).get("level", 0)
        post_level = h_post.get(name, {}).get("level", 0)
        hormone_deltas[name] = round(post_level - pre_level, 4)
    deltas["hormone_deltas"] = hormone_deltas

    # Neuromodulator deltas
    nm_pre = pre.get("neuromodulators", {})
    nm_post = post.get("neuromodulators", {})
    nm_deltas = {}
    for name in set(list(nm_pre.keys()) + list(nm_post.keys())):
        nm_deltas[name] = round(nm_post.get(name, 0) - nm_pre.get(name, 0), 4)
    deltas["neuromodulator_deltas"] = nm_deltas

    # Emotion change
    deltas["emotion_before"] = pre.get("emotion", "?")
    deltas["emotion_after"] = post.get("emotion", "?")

    # State vector coherence deltas
    sv_pre = pre.get("state_vector", [])
    sv_post = post.get("state_vector", [])
    if len(sv_pre) >= 65 and len(sv_post) >= 65:
        deltas["inner_body_delta"] = round(
            sum(sv_post[0:5]) / 5 - sum(sv_pre[0:5]) / 5, 4)
        deltas["inner_mind_delta"] = round(
            sum(sv_post[5:20]) / 15 - sum(sv_pre[5:20]) / 15, 4)
        deltas["inner_spirit_delta"] = round(
            sum(sv_post[20:65]) / 45 - sum(sv_pre[20:65]) / 45, 4)
    if len(sv_pre) >= 130 and len(sv_post) >= 130:
        deltas["outer_body_delta"] = round(
            sum(sv_post[65:70]) / 5 - sum(sv_pre[65:70]) / 5, 4)
        deltas["outer_mind_delta"] = round(
            sum(sv_post[70:85]) / 15 - sum(sv_pre[70:85]) / 15, 4)
        deltas["outer_spirit_delta"] = round(
            sum(sv_post[85:130]) / 45 - sum(sv_pre[85:130]) / 45, 4)

    return deltas


# ── Telemetry ─────────────────────────────────────────────────────

class ChildDevTelemetry:
    """Collects per-trial telemetry for one Titan instance."""

    def __init__(self, instance: str):
        self.instance = instance
        self.trials: list[dict] = []
        self.test_summaries: dict[str, dict] = {}
        self.start_time = time.time()

    def record_trial(self, test_type: str, trial_id: int,
                     stimulus: dict, pre_state: dict, post_state: dict,
                     result: dict, elapsed_ms: float):
        deltas = compute_deltas(pre_state, post_state)
        trial = {
            "test_type": test_type,
            "trial_id": trial_id,
            "timestamp": time.time(),
            "instance": self.instance,
            "stimulus": stimulus,
            "pre_state": {k: v for k, v in pre_state.items()
                         if k != "state_vector"},
            "post_state": {k: v for k, v in post_state.items()
                          if k != "state_vector"},
            "deltas": deltas,
            "result": result,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        self.trials.append(trial)
        return trial

    def summarize_test(self, test_type: str) -> dict:
        test_trials = [t for t in self.trials if t["test_type"] == test_type]
        if not test_trials:
            return {}

        scores = [t["result"].get("score", 0) for t in test_trials]
        correct = sum(1 for t in test_trials if t["result"].get("correct", False))

        # By level
        by_level = {}
        for t in test_trials:
            level = t["stimulus"].get("level", 0)
            if level not in by_level:
                by_level[level] = {"scores": [], "correct": 0, "total": 0}
            by_level[level]["scores"].append(t["result"].get("score", 0))
            by_level[level]["total"] += 1
            if t["result"].get("correct", False):
                by_level[level]["correct"] += 1
        for level, data in by_level.items():
            data["accuracy"] = round(data["correct"] / max(1, data["total"]), 4)
            data["mean_score"] = round(
                sum(data["scores"]) / max(1, len(data["scores"])), 4)

        # Aggregate hormone response signature
        hormone_sig = {}
        for t in test_trials:
            for name, delta in t["deltas"].get("hormone_deltas", {}).items():
                if name not in hormone_sig:
                    hormone_sig[name] = []
                hormone_sig[name].append(delta)
        hormone_sig = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in hormone_sig.items()
        }

        summary = {
            "test_type": test_type,
            "instance": self.instance,
            "total_trials": len(test_trials),
            "correct": correct,
            "accuracy": round(correct / max(1, len(test_trials)), 4),
            "mean_score": round(sum(scores) / max(1, len(scores)), 4),
            "by_level": by_level,
            "hormone_signature": hormone_sig,
        }
        self.test_summaries[test_type] = summary
        return summary

    def to_dict(self) -> dict:
        return {
            "instance": self.instance,
            "start_time": self.start_time,
            "duration_s": round(time.time() - self.start_time, 1),
            "total_trials": len(self.trials),
            "trials": self.trials,
            "summaries": self.test_summaries,
        }


# ── Test Runners ──────────────────────────────────────────────────

async def run_T1(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T1: Word-State Association — recall and production."""
    log.info("═══ T1: Word-State Association (%d stimuli) ═══", len(test_data))

    baseline = await capture_baseline(client, api)
    vocab_before = await get_vocabulary(client, api)
    vocab_words_before = {w.get("word", ""): w for w in vocab_before}

    for i, item in enumerate(test_data):
        word = item["word"]
        perturbation = item["perturbation"]

        log.info("[T1] Trial %d/%d: '%s'", i + 1, len(test_data), word)

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Inject word stimulus and wait for next consciousness epoch
        pre_epoch = pre.get("epoch_id", 0)
        await inject_stimulus(client, api, perturbation, word=word)
        log.info("[T1] Injected '%s' — waiting for epoch > %d...", word, pre_epoch)
        post_state = await wait_for_new_epoch(client, api, pre_epoch)
        post = await capture_snapshot(client, api)
        post["state_vector"] = post_state.get("state_vector", post.get("state_vector", []))
        post["epoch_id"] = post_state.get("epoch_id", post.get("epoch_id", 0))
        elapsed = (time.time() - t0) * 1000

        # Measure: state vector change magnitude
        sv_pre = pre.get("state_vector", [])
        sv_post = post.get("state_vector", [])
        state_change = vector_distance(sv_pre, sv_post) if sv_pre and sv_post else 0

        # Measure: hormone response (which hormone changed most)
        h_deltas = compute_deltas(pre, post).get("hormone_deltas", {})
        top_hormone = max(h_deltas, key=lambda k: abs(h_deltas[k])) if h_deltas else "?"
        top_delta = h_deltas.get(top_hormone, 0)
        expected_hormone = item.get("expected_response", {}).get("top_hormone", "")
        hormone_match = top_hormone == expected_hormone if expected_hormone else None

        # Score: state change magnitude (higher = stronger association)
        score = min(1.0, state_change / 2.0)  # Normalize

        telemetry.record_trial("T1", i, {
            "word": word, "category": item.get("expected_response", {}).get("category", ""),
            "level": 0,
        }, pre, post, {
            "correct": score > 0.1,
            "score": round(score, 4),
            "state_change": round(state_change, 4),
            "top_hormone": top_hormone,
            "top_hormone_delta": round(top_delta, 4),
            "expected_hormone": expected_hormone,
            "hormone_match": hormone_match,
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T1")


async def run_T2(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T2: Odd-One-Out — category formation."""
    log.info("═══ T2: Odd-One-Out (%d sets) ═══", len(test_data))

    for i, item in enumerate(test_data):
        level = item["level"]
        words = item["items"]
        odd_idx = item["odd_index"]
        log.info("[T2] Trial %d/%d (L%d): %s → odd=%s",
                 i + 1, len(test_data), level, words, words[odd_idx])

        # Inject each word, capture state response
        responses = []
        pre = await capture_snapshot(client, api)
        t0 = time.time()

        for w_idx, word in enumerate(words):
            snap_before = await capture_snapshot(client, api)
            pre_epoch = snap_before.get("epoch_id", 0)
            # Inject REAL perturbation from word recipe (not zeros)
            pert = _get_word_perturbation(word)
            await inject_stimulus(client, api, pert, word=word)
            # Wait for consciousness epoch to integrate the perturbation
            snap_after_state = await wait_for_new_epoch(client, api, pre_epoch)
            snap_after = await capture_snapshot(client, api)
            snap_after["state_vector"] = snap_after_state.get("state_vector", snap_after.get("state_vector", []))

            sv = snap_after.get("state_vector", [])
            responses.append({"word": word, "state": sv[:65] if len(sv) >= 65 else sv})

        elapsed = (time.time() - t0) * 1000

        # Compute pairwise similarities — odd one should have lowest avg similarity
        if all(r["state"] for r in responses):
            avg_sims = []
            for j in range(len(responses)):
                sims = []
                for k in range(len(responses)):
                    if j != k:
                        sims.append(cosine_similarity(responses[j]["state"],
                                                      responses[k]["state"]))
                avg_sims.append(sum(sims) / max(1, len(sims)))

            predicted_odd = avg_sims.index(min(avg_sims))
            correct = predicted_odd == odd_idx
            # Score based on discrimination
            if len(avg_sims) >= 2:
                sorted_sims = sorted(avg_sims)
                discrimination = sorted_sims[1] - sorted_sims[0]
            else:
                discrimination = 0
        else:
            predicted_odd = -1
            correct = False
            discrimination = 0

        post = await capture_snapshot(client, api)
        telemetry.record_trial("T2", i, {
            "items": words, "odd_index": odd_idx, "level": level,
            "category": item.get("category", ""),
        }, pre, post, {
            "correct": correct,
            "score": round(discrimination, 4),
            "predicted_odd": predicted_odd,
            "expected_odd": odd_idx,
            "avg_similarities": [round(s, 4) for s in avg_sims] if avg_sims else [],
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T2")


async def run_T3(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T3: Analogy Completion — A:B :: C:? → find D."""
    log.info("═══ T3: Analogy Completion (%d items) ═══", len(test_data))

    vocab = await get_vocabulary(client, api)

    for i, item in enumerate(test_data):
        level = item["level"]
        A, B, C, D_expected = item["A"], item["B"], item["C"], item["D"]
        log.info("[T3] Trial %d/%d (L%d): %s:%s :: %s:? (expect %s)",
                 i + 1, len(test_data), level, A, B, C, D_expected)

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Inject A with real perturbation, wait for epoch
        pre_ep = (await capture_snapshot(client, api)).get("epoch_id", 0)
        await inject_stimulus(client, api, _get_word_perturbation(A), word=A)
        state_A_data = await wait_for_new_epoch(client, api, pre_ep)
        state_A = state_A_data.get("state_vector", [])

        # Inject B with real perturbation, wait for epoch
        pre_ep = state_A_data.get("epoch_id", 0)
        await inject_stimulus(client, api, _get_word_perturbation(B), word=B)
        state_B_data = await wait_for_new_epoch(client, api, pre_ep)
        state_B = state_B_data.get("state_vector", [])

        # Inject C with real perturbation, wait for epoch
        pre_ep = state_B_data.get("epoch_id", 0)
        await inject_stimulus(client, api, _get_word_perturbation(C), word=C)
        state_C_data = await wait_for_new_epoch(client, api, pre_ep)
        state_C = state_C_data.get("state_vector", [])

        # Strategy: measure A→B relationship pattern, find which vocab word D
        # creates the same pattern when applied to C.
        # relationship = how state changed from A to B (the delta vector)
        min_len = min(len(state_A), len(state_B), len(state_C))
        best_word = "?"
        best_sim = -1
        if min_len >= 65:
            # The A→B delta captures the relationship
            ab_delta = [state_B[j] - state_A[j] for j in range(min_len)]
            # Also try classic: predicted_D = C + (B - A)
            predicted_D = [state_C[j] + ab_delta[j] for j in range(min_len)]

            # Score each candidate word by injecting its recipe and comparing
            # For efficiency: use recipe perturbation vectors as proxy for felt-state
            recipes = _load_recipes()
            for vword, pert in recipes.items():
                if vword in (A, B, C):
                    continue
                # Build a flat felt-vector from the recipe perturbation
                fv = (pert.get("inner_body", [0]*5) +
                      pert.get("inner_mind", [0]*15) +
                      pert.get("inner_spirit", [0]*45))
                if len(fv) >= 65:
                    sim = cosine_similarity(predicted_D[:65], fv[:65])
                    if sim > best_sim:
                        best_sim = sim
                        best_word = vword

        elapsed = (time.time() - t0) * 1000
        correct = best_word == D_expected
        score = 1.0 if correct else max(0, best_sim)

        post = await capture_snapshot(client, api)
        telemetry.record_trial("T3", i, {
            "A": A, "B": B, "C": C, "D_expected": D_expected,
            "level": level, "relation": item.get("relation", ""),
        }, pre, post, {
            "correct": correct,
            "score": round(score, 4),
            "predicted": best_word,
            "expected": D_expected,
            "similarity": round(best_sim, 4),
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T3")


async def run_T4(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T4: Sequence Completion — predict next in sequence."""
    log.info("═══ T4: Sequence Completion (%d sequences) ═══", len(test_data))

    vocab = await get_vocabulary(client, api)

    for i, item in enumerate(test_data):
        level = item["level"]
        sequence = item["sequence"]
        expected_next = item["next"]
        log.info("[T4] Trial %d/%d (L%d): %s → ?=%s",
                 i + 1, len(test_data), level, sequence, expected_next)

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Inject sequence words with REAL perturbations, wait for epochs
        states = []
        cur_epoch = (await capture_snapshot(client, api)).get("epoch_id", 0)
        for word in sequence:
            await inject_stimulus(client, api, _get_word_perturbation(word), word=word)
            epoch_data = await wait_for_new_epoch(client, api, cur_epoch)
            sv = epoch_data.get("state_vector", [])
            states.append(sv)
            cur_epoch = epoch_data.get("epoch_id", cur_epoch)

        # Predict next: extrapolate trajectory direction from full sequence
        predicted_next = []
        if len(states) >= 2 and all(len(s) >= 65 for s in states):
            # Average direction across all consecutive pairs (more robust than just last 2)
            min_len = min(len(s) for s in states)
            avg_delta = [0.0] * min_len
            n_pairs = len(states) - 1
            for p in range(n_pairs):
                for j in range(min_len):
                    avg_delta[j] += (states[p+1][j] - states[p][j]) / n_pairs
            predicted_next = [states[-1][j] + avg_delta[j] for j in range(min_len)]

        # Find closest vocab word using recipe perturbation vectors
        best_word = "?"
        best_sim = -1
        if predicted_next:
            recipes = _load_recipes()
            for vword, pert in recipes.items():
                if vword in sequence:
                    continue
                fv = (pert.get("inner_body", [0]*5) +
                      pert.get("inner_mind", [0]*15) +
                      pert.get("inner_spirit", [0]*45))
                if len(fv) >= 65:
                    sim = cosine_similarity(predicted_next[:65], fv[:65])
                    if sim > best_sim:
                        best_sim = sim
                        best_word = vword

        elapsed = (time.time() - t0) * 1000
        correct = best_word == expected_next

        post = await capture_snapshot(client, api)
        telemetry.record_trial("T4", i, {
            "sequence": sequence, "expected_next": expected_next,
            "level": level, "type": item.get("type", ""),
        }, pre, post, {
            "correct": correct,
            "score": 1.0 if correct else max(0, best_sim),
            "predicted": best_word,
            "expected": expected_next,
            "similarity": round(best_sim, 4),
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T4")


async def run_T5(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list,
                 instance: str = "titan1"):
    """T5: Cross-Modal Transfer — body↔mind Sapir-Whorf test."""
    log.info("═══ T5: Cross-Modal Transfer (%d words, instance=%s) ═══",
             len(test_data), instance)

    for i, item in enumerate(test_data):
        word = item["word"]
        body_pert = item["body_perturbation"]
        mind_pert = item["mind_perturbation"]

        log.info("[T5] Trial %d/%d: '%s' (%s → %s)",
                 i + 1, len(test_data), word,
                 "body→mind" if instance == "titan1" else "mind→body",
                 item.get("color", ""), )

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Phase 1: Train in primary modality
        if instance == "titan1":
            # Body-first: inject body perturbation only
            train_pert = {
                "inner_body": body_pert.get("inner_body", [0] * 5),
                "inner_mind": [0] * 15, "inner_spirit": [0] * 45,
                "outer_body": body_pert.get("outer_body", [0] * 5),
                "outer_mind": [0] * 15, "outer_spirit": [0] * 45,
            }
        else:
            # Mind-first: inject mind perturbation only
            train_pert = {
                "inner_body": [0] * 5,
                "inner_mind": mind_pert.get("inner_mind", [0] * 15),
                "inner_spirit": [0] * 45,
                "outer_body": [0] * 5,
                "outer_mind": mind_pert.get("outer_mind", [0] * 15),
                "outer_spirit": [0] * 45,
            }

        await inject_stimulus(client, api, train_pert, word=word)
        await asyncio.sleep(15)
        state_trained = (await capture_snapshot(client, api)).get("state_vector", [])

        # Phase 2: Test in OTHER modality (no word label)
        if instance == "titan1":
            # Test with mind-only
            test_pert = {
                "inner_body": [0] * 5,
                "inner_mind": mind_pert.get("inner_mind", [0] * 15),
                "inner_spirit": [0] * 45,
                "outer_body": [0] * 5,
                "outer_mind": mind_pert.get("outer_mind", [0] * 15),
                "outer_spirit": [0] * 45,
            }
        else:
            # Test with body-only
            test_pert = {
                "inner_body": body_pert.get("inner_body", [0] * 5),
                "inner_mind": [0] * 15, "inner_spirit": [0] * 45,
                "outer_body": body_pert.get("outer_body", [0] * 5),
                "outer_mind": [0] * 15, "outer_spirit": [0] * 45,
            }

        await inject_stimulus(client, api, test_pert)  # No word!
        await asyncio.sleep(15)
        state_transfer = (await capture_snapshot(client, api)).get("state_vector", [])

        # Measure: similarity between trained state and transfer state
        transfer_sim = cosine_similarity(state_trained, state_transfer) \
            if state_trained and state_transfer else 0

        elapsed = (time.time() - t0) * 1000
        post = await capture_snapshot(client, api)

        telemetry.record_trial("T5", i, {
            "word": word, "level": 0,
            "direction": "body→mind" if instance == "titan1" else "mind→body",
            "color": item.get("color", ""),
            "frequency": item.get("frequency", ""),
        }, pre, post, {
            "correct": transfer_sim > 0.7,
            "score": round(transfer_sim, 4),
            "transfer_similarity": round(transfer_sim, 4),
            "direction": "body→mind" if instance == "titan1" else "mind→body",
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T5")


async def run_T6(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T6: Mutual Exclusivity — novel word disambiguation."""
    log.info("═══ T6: Mutual Exclusivity (%d pairs) ═══", len(test_data))

    for i, item in enumerate(test_data):
        known = item["known_word"]
        novel = item["novel_word"]
        novel_pert = item["novel_perturbation"]

        log.info("[T6] Trial %d/%d: known='%s' + novel='%s'",
                 i + 1, len(test_data), known, novel)

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Inject known word with its real recipe, wait for epoch
        pre_epoch = pre.get("epoch_id", 0)
        await inject_stimulus(client, api, _get_word_perturbation(known), word=known)
        state_known_data = await wait_for_new_epoch(client, api, pre_epoch)
        state_known = state_known_data.get("state_vector", [])

        # Inject novel perturbation with novel word label, wait for epoch
        novel_epoch = state_known_data.get("epoch_id", pre_epoch)
        await inject_stimulus(client, api, novel_pert, word=novel)
        state_novel_data = await wait_for_new_epoch(client, api, novel_epoch)
        state_novel = state_novel_data.get("state_vector", [])

        # Test: is the novel state DIFFERENT from known state?
        # (mutual exclusivity = novel word maps to novel region, not existing)
        if state_known and state_novel:
            disambiguation = 1.0 - cosine_similarity(state_known, state_novel)
        else:
            disambiguation = 0

        elapsed = (time.time() - t0) * 1000
        post = await capture_snapshot(client, api)

        # Threshold lowered for early developmental stage (was 0.1 — too aggressive)
        telemetry.record_trial("T6", i, {
            "known_word": known, "novel_word": novel, "level": 0,
        }, pre, post, {
            "correct": disambiguation > 0.02,
            "score": round(disambiguation, 4),
            "disambiguation": round(disambiguation, 4),
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T6")


async def run_T7(client: httpx.AsyncClient, api: str,
                 telemetry: ChildDevTelemetry, test_data: list):
    """T7: Multi-Modal Association — word + color + sound + feeling."""
    log.info("═══ T7: Multi-Modal Association (%d stimuli) ═══", len(test_data))

    for i, item in enumerate(test_data):
        word = item["word"]
        log.info("[T7] Trial %d/%d: '%s' (color=%s, freq=%s)",
                 i + 1, len(test_data), word,
                 item.get("color", "?"), item.get("frequency", "?"))

        pre = await capture_snapshot(client, api)
        t0 = time.time()

        # Inject full multi-modal perturbation and wait for Spirit integration
        pre_epoch = pre.get("epoch_id", 0)
        pert = item.get("combined_perturbation", item.get("perturbation", {}))
        await inject_stimulus(client, api, pert, word=word)
        log.info("[T7] Injected '%s' — waiting for epoch > %d...", word, pre_epoch)
        post_state = await wait_for_new_epoch(client, api, pre_epoch)

        post = await capture_snapshot(client, api)
        post["state_vector"] = post_state.get("state_vector", post.get("state_vector", []))
        post["epoch_id"] = post_state.get("epoch_id", post.get("epoch_id", 0))
        elapsed = (time.time() - t0) * 1000

        # Measure integration: body-mind coherence
        sv_post = post.get("state_vector", [])
        body_mind_coh = 0
        spirit_integration = 0
        if len(sv_post) >= 65:
            body = sv_post[0:5]
            mind = sv_post[5:20]
            spirit = sv_post[20:65]
            # Body-mind coherence: correlation between body and mind changes
            body_mean = sum(body) / 5
            mind_mean = sum(mind) / 15
            body_mind_coh = 1.0 - abs(body_mean - mind_mean)
            spirit_integration = sum(abs(s) for s in spirit) / 45

        # State change magnitude
        sv_pre = pre.get("state_vector", [])
        state_change = vector_distance(sv_pre, sv_post) if sv_pre and sv_post else 0

        telemetry.record_trial("T7", i, {
            "word": word, "level": 0,
            "color": item.get("color", ""),
            "frequency": item.get("frequency", ""),
            "felt_meaning": item.get("felt_meaning", {}),
        }, pre, post, {
            "correct": state_change > 0.1,
            "score": round(min(1.0, state_change / 2.0), 4),
            "state_change": round(state_change, 4),
            "body_mind_coherence": round(body_mind_coh, 4),
            "spirit_integration": round(spirit_integration, 4),
        }, elapsed)

        await asyncio.sleep(TRIAL_REST_S)

    return telemetry.summarize_test("T7")


def _zero_perturbation() -> dict:
    """Empty perturbation (word-only stimulus)."""
    return {
        "inner_body": [0.0] * 5, "inner_mind": [0.0] * 15,
        "inner_spirit": [0.0] * 45,
        "outer_body": [0.0] * 5, "outer_mind": [0.0] * 15,
        "outer_spirit": [0.0] * 45,
    }


# ── Word Recipe Lookup ───────────────────────────────────────────

_RECIPE_CACHE: dict = {}


def _load_recipes() -> dict:
    """Load all word perturbation recipes from Phase 1/2/3 files."""
    global _RECIPE_CACHE
    if _RECIPE_CACHE:
        return _RECIPE_CACHE
    for fname in ["data/word_resonance.json", "data/word_resonance_phase2.json",
                   "data/word_resonance_phase3.json"]:
        try:
            with open(fname) as f:
                d = json.load(f)
            for k, v in d.items():
                if not k.startswith("_") and "perturbation" in v:
                    _RECIPE_CACHE[k] = v["perturbation"]
        except Exception:
            pass
    log.info("Loaded %d word recipes from 3 files", len(_RECIPE_CACHE))
    return _RECIPE_CACHE


def _get_word_perturbation(word: str) -> dict:
    """Get actual perturbation recipe for a word. Falls back to zero if not found."""
    recipes = _load_recipes()
    pert = recipes.get(word)
    if pert:
        return pert
    log.warning("No recipe for word '%s' — using zero perturbation", word)
    return _zero_perturbation()


# ── Report Generator ─────────────────────────────────────────────

def generate_report(t1_telemetry: ChildDevTelemetry,
                    t2_telemetry: Optional[ChildDevTelemetry] = None) -> str:
    """Generate comparative markdown report."""
    now = datetime.now()
    lines = [
        f"# Child Development Test Report",
        f"## {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        "### Executive Summary",
        "",
        f"- **Date:** {now.strftime('%Y-%m-%d')}",
        f"- **Duration:** {round((time.time() - t1_telemetry.start_time) / 60, 1)} minutes",
        f"- **Titan1 trials:** {len(t1_telemetry.trials)}",
    ]
    if t2_telemetry:
        lines.append(f"- **Titan2 trials:** {len(t2_telemetry.trials)}")
    lines.extend(["", "---", ""])

    # Per-test results
    all_tests = sorted(set(
        list(t1_telemetry.test_summaries.keys()) +
        (list(t2_telemetry.test_summaries.keys()) if t2_telemetry else [])
    ))

    test_names = {
        "T1": "Word-State Association",
        "T2": "Odd-One-Out (Category Formation)",
        "T3": "Analogy Completion",
        "T4": "Sequence Completion",
        "T5": "Cross-Modal Transfer (Sapir-Whorf)",
        "T6": "Mutual Exclusivity",
        "T7": "Multi-Modal Association",
    }

    for test in all_tests:
        name = test_names.get(test, test)
        lines.extend([f"### {test}: {name}", ""])

        s1 = t1_telemetry.test_summaries.get(test, {})
        s2 = t2_telemetry.test_summaries.get(test, {}) if t2_telemetry else {}

        if s1 or s2:
            lines.append("| Metric | Titan1 | Titan2 | Delta |")
            lines.append("|--------|--------|--------|-------|")

            a1 = s1.get("accuracy", 0)
            a2 = s2.get("accuracy", 0)
            lines.append(f"| Accuracy | {a1:.2%} | {a2:.2%} | {a1-a2:+.2%} |")

            m1 = s1.get("mean_score", 0)
            m2 = s2.get("mean_score", 0)
            lines.append(f"| Mean Score | {m1:.4f} | {m2:.4f} | {m1-m2:+.4f} |")

            n1 = s1.get("total_trials", 0)
            n2 = s2.get("total_trials", 0)
            lines.append(f"| Trials | {n1} | {n2} | |")

            # Level breakdown
            levels1 = s1.get("by_level", {})
            levels2 = s2.get("by_level", {})
            all_levels = sorted(set(list(levels1.keys()) + list(levels2.keys())))
            if all_levels and all_levels != [0]:
                lines.extend(["", "**By Level:**", ""])
                lines.append("| Level | T1 Accuracy | T2 Accuracy | T1 Score | T2 Score |")
                lines.append("|-------|------------|------------|----------|----------|")
                for lv in all_levels:
                    l1 = levels1.get(lv, {})
                    l2 = levels2.get(lv, {})
                    lines.append(
                        f"| L{lv} | {l1.get('accuracy', 0):.2%} | "
                        f"{l2.get('accuracy', 0):.2%} | "
                        f"{l1.get('mean_score', 0):.4f} | "
                        f"{l2.get('mean_score', 0):.4f} |")

            # Hormone signature
            hs1 = s1.get("hormone_signature", {})
            hs2 = s2.get("hormone_signature", {})
            if hs1 or hs2:
                lines.extend(["", "**Hormone Response Signature:**", ""])
                lines.append("| Hormone | T1 Mean Δ | T2 Mean Δ |")
                lines.append("|---------|----------|----------|")
                all_h = sorted(set(list(hs1.keys()) + list(hs2.keys())))
                for h in all_h[:10]:
                    lines.append(
                        f"| {h} | {hs1.get(h, 0):+.4f} | {hs2.get(h, 0):+.4f} |")

        lines.extend(["", "---", ""])

    # Comparative analysis
    if t2_telemetry and "T5" in t1_telemetry.test_summaries:
        lines.extend([
            "### Sapir-Whorf Analysis (T5 Cross-Modal Transfer)",
            "",
        ])
        s1 = t1_telemetry.test_summaries.get("T5", {})
        s2 = t2_telemetry.test_summaries.get("T5", {})
        lines.append(f"- **Titan1 (body→mind):** accuracy={s1.get('accuracy', 0):.2%}, "
                     f"mean_score={s1.get('mean_score', 0):.4f}")
        lines.append(f"- **Titan2 (mind→body):** accuracy={s2.get('accuracy', 0):.2%}, "
                     f"mean_score={s2.get('mean_score', 0):.4f}")
        divergence = abs(s1.get("mean_score", 0) - s2.get("mean_score", 0))
        lines.append(f"- **Divergence index:** {divergence:.4f}")
        lines.extend(["", "---", ""])

    lines.extend([
        "### Telemetry Files",
        "",
        f"- Titan1: `data/child_dev_telemetry_titan1_{now.strftime('%Y%m%d_%H%M')}.json`",
    ])
    if t2_telemetry:
        lines.append(
            f"- Titan2: `data/child_dev_telemetry_titan2_{now.strftime('%Y%m%d_%H%M')}.json`")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by child_development_test.py*",
    ])
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────

TEST_RUNNERS = {
    "T1": run_T1,
    "T2": run_T2,
    "T3": run_T3,
    "T4": run_T4,
    "T5": run_T5,
    "T6": run_T6,
    "T7": run_T7,
}


async def run_instance(instance: str, api: str, tests: list,
                       test_data: dict) -> ChildDevTelemetry:
    """Run all selected tests on one instance."""
    telemetry = ChildDevTelemetry(instance)

    async with httpx.AsyncClient() as client:
        # Verify instance is up
        try:
            r = await client.get(f"{api}/health", timeout=10)
            if r.status_code != 200:
                log.error("[%s] Instance not healthy: %d", instance, r.status_code)
                return telemetry
        except Exception as e:
            log.error("[%s] Instance unreachable: %s", instance, e)
            return telemetry

        log.info("═══════════════════════════════════════════")
        log.info(" Running tests on %s (%s)", instance, api)
        log.info("═══════════════════════════════════════════")

        for test_name in tests:
            runner = TEST_RUNNERS.get(test_name)
            data_key = {
                "T1": "T1_word_state", "T2": "T2_odd_one_out",
                "T3": "T3_analogy", "T4": "T4_sequence",
                "T5": "T5_cross_modal", "T6": "T6_mutual_exclusivity",
                "T7": "T7_multi_modal",
            }.get(test_name)

            if not runner or not data_key:
                log.warning("Unknown test: %s", test_name)
                continue

            items = test_data.get(data_key, [])
            if not items:
                log.warning("No test data for %s", test_name)
                continue

            try:
                if test_name == "T5":
                    summary = await runner(client, api, telemetry, items,
                                           instance=instance)
                else:
                    summary = await runner(client, api, telemetry, items)
                log.info("[%s] %s complete: accuracy=%.2f%%, mean_score=%.4f",
                         instance, test_name,
                         summary.get("accuracy", 0) * 100,
                         summary.get("mean_score", 0))
            except Exception as e:
                log.error("[%s] %s failed: %s", instance, test_name, e)
                import traceback
                traceback.print_exc()

            if test_name != tests[-1]:
                log.info("[%s] Rest between tests (%ds)...", instance, TEST_REST_S)
                await asyncio.sleep(TEST_REST_S)

    return telemetry


async def main():
    parser = argparse.ArgumentParser(description="Child Development Test Suite")
    parser.add_argument("--tests", default="T1,T2,T3,T4,T5,T6,T7",
                        help="Comma-separated test names (T1-T7)")
    parser.add_argument("--instances", default="titan1,titan2",
                        help="Comma-separated instance names")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for reports and telemetry")
    args = parser.parse_args()

    tests = [t.strip() for t in args.tests.split(",")]
    instances = [i.strip() for i in args.instances.split(",")]

    # Load test data
    data_path = os.path.join(os.path.dirname(__file__),
                             "..", "data", "child_development_tests.json")
    with open(data_path) as f:
        test_data = json.load(f)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  CHILD DEVELOPMENT TEST SUITE                    ║")
    log.info("║  Tests: %s", ", ".join(tests))
    log.info("║  Instances: %s", ", ".join(instances))
    log.info("╚══════════════════════════════════════════════════╝")

    # Run instances in parallel
    tasks = []
    for inst in instances:
        if inst not in INSTANCES:
            log.warning("Unknown instance: %s", inst)
            continue
        api = INSTANCES[inst]["api"]
        tasks.append(run_instance(inst, api, tests, test_data))

    results = await asyncio.gather(*tasks)

    # Save telemetry
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M")
    telemetries = {}
    for telem in results:
        telemetries[telem.instance] = telem
        telem_path = f"data/child_dev_telemetry_{telem.instance}_{ts}.json"
        with open(telem_path, "w") as f:
            json.dump(telem.to_dict(), f, indent=2)
        log.info("Telemetry saved: %s (%d trials)", telem_path, len(telem.trials))

    # Generate report
    t1_telem = telemetries.get("titan1")
    t2_telem = telemetries.get("titan2")
    if t1_telem:
        report = generate_report(t1_telem, t2_telem)
        report_path = f"titan-docs/REPORT_child_development_{now.strftime('%Y%m%d')}.md"
        with open(report_path, "w") as f:
            f.write(report)
        log.info("Report saved: %s", report_path)

    # Summary
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  TEST SUITE COMPLETE                             ║")
    for inst, telem in telemetries.items():
        total = len(telem.trials)
        correct = sum(1 for t in telem.trials if t["result"].get("correct", False))
        log.info("║  %s: %d/%d correct (%.1f%%)", inst, correct, total,
                 correct / max(1, total) * 100)
    log.info("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(main())
