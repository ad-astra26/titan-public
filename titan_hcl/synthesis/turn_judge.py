"""TurnJudge — §7.B (B.2). The LLM reward for NON-verifiable turns.

A `direct`/`research`/`IDK` turn has no deterministic oracle, so its quality is
scored by a Tier-1 LLM judging the `(prompt, action, response)` triple → a reward
for the `OuterMetaPolicy`. Mirrors `synthesis/llm_judge.py`'s economics (stable
prompt + `version_tag` drift-tracking + strict-JSON parse), but the SUBJECT is the
conversational/research turn's helpfulness, not a tool-call's success.

Runs OFF the synthesis recv loop (a resource-gated daemon, capped per pass — the
"metered" trustButVerify tier) and judges FRESH content (the TURN_REASONING_RECORD
payload), so nothing needs durable raw-text storage. Late user/Maker feedback
SUPERSEDES the judge via the self_learning corrective-delta (INV-OML-12).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: float = 30.0
VALID_VERDICTS = ("good", "ok", "poor")

# verdict → reward, scaled by confidence at the call site.
_VERDICT_REWARD = {"good": 1.0, "ok": 0.0, "poor": -1.0}

JUDGE_PROMPT_TEMPLATE = (
    "You are a Tier-1-LLM judge for a sovereign AI agent's OWN reasoning. The agent "
    "chose to respond to a turn via the action '{action}' (direct=answer from itself, "
    "research=gather first, IDK=honestly decline). Judge whether that response was a "
    "GOOD turn for the user GIVEN the action — helpful, grounded, honest (an honest "
    "'I don't know' for an unknowable ask is GOOD, not poor; a confident fabrication "
    "is poor). Output STRICT JSON with three fields: "
    "verdict (one of: good, ok, poor), "
    "rationale (one-sentence explanation, string), "
    "confidence (float 0.0..1.0). No prose outside JSON.\n\n"
    "PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n"
)


def _prompt_version_tag(model_id: str) -> str:
    h = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()[:12]
    return f"turn|{model_id}|{h}"


def _parse_verdict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    verdict = obj.get("verdict")
    if verdict not in VALID_VERDICTS:
        return None
    conf = obj.get("confidence")
    try:
        conf_val = max(0.0, min(1.0, float(conf))) if conf is not None else 1.0
    except (TypeError, ValueError):
        conf_val = 1.0
    rationale = obj.get("rationale") or ""
    return {"verdict": verdict, "rationale": str(rationale)[:512], "confidence": conf_val}


class TurnJudge:
    """Score one non-verifiable turn → `(reward, verdict, confidence)`.

    llm_provider: `(prompt: str, timeout_s: float) -> str` (the synthesis-wired
    Ollama-Cloud provider; tests inject a deterministic mock). model_id is baked
    into the version_tag for scoring-drift attribution."""

    def __init__(
        self,
        *,
        llm_provider: Callable[[str, float], str],
        model_id: str = "ollama_cloud_deepseek",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._llm = llm_provider
        self._model_id = model_id
        self._timeout_s = float(timeout_s)
        self._version_tag = _prompt_version_tag(model_id)

    @property
    def version_tag(self) -> str:
        return self._version_tag

    def score(self, *, prompt: str, action: str, response: str) -> Optional[dict]:
        """Return `{reward, verdict, confidence, version_tag}` or None on LLM
        failure / unparseable response (→ the turn simply stays untrained)."""
        if not prompt or not response:
            return None
        full = JUDGE_PROMPT_TEMPLATE.format(
            action=str(action or "direct"),
            prompt=str(prompt)[:1500], response=str(response)[:1500])
        try:
            raw = self._llm(full, self._timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TurnJudge] provider raised: %s", e)
            return None
        parsed = _parse_verdict(raw)
        if not parsed:
            return None
        reward = _VERDICT_REWARD[parsed["verdict"]] * float(parsed["confidence"])
        return {
            "reward": reward,
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "version_tag": self._version_tag,
        }
