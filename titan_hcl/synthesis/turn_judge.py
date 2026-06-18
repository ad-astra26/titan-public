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


def _rubric_line(level_norm: float) -> str:
    """P5 (b) — the level-conditioned rubric (`ARCHITECTURE_mastery_leveling.md` §4).

    A CONTINUOUS descriptor (no discrete hand-set bands — the bar scales smoothly
    with the agent's emergent proven ability `level_norm` ∈ [0,1]), PREPENDED to the
    base template so `JUDGE_PROMPT_TEMPLATE` stays byte-identical (drift-tag stable).
    Lenient at low ability → demanding at high ability: the SAME answer earns a worse
    verdict as the agent grows (the co-adaptive teacher never goes stale, INV-MC-3)."""
    pct = int(round(max(0.0, min(1.0, level_norm)) * 100))
    return (
        "GRADING STANDARD: this agent's own PROVEN mastery is at the "
        f"{pct}th-percentile of its ability ladder — hold it to a correspondingly "
        "higher bar. At LOW proven ability, reward any helpful, honest, grounded "
        "turn as 'good'. At HIGH proven ability, reserve 'good' for an OPTIMALLY-"
        "routed, well-grounded, concise turn and grade a merely-adequate turn as "
        "'ok' (not 'good'). Scale your strictness to that percentile.\n\n")


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
        relself_gain: float = 1.0,
        ema_alpha: float = 0.1,
    ):
        self._llm = llm_provider
        self._model_id = model_id
        self._timeout_s = float(timeout_s)
        self._version_tag = _prompt_version_tag(model_id)
        # P5 (a) relative-to-self: a streaming EMA of the agent's own recent RAW
        # rewards — repeating prior competence yields diminishing shaped reward
        # (INV-MC-3). Held on the instance (persists across the daemon loop).
        self._relself_gain = float(relself_gain)
        self._ema_alpha = float(ema_alpha)
        self._reward_ema = 0.0

    @property
    def version_tag(self) -> str:
        return self._version_tag

    @property
    def reward_ema(self) -> float:
        return self._reward_ema

    def score(self, *, prompt: str, action: str, response: str,
              level_norm: Optional[float] = None) -> Optional[dict]:
        """Return `{reward, raw_reward, verdict, confidence, version_tag, level_norm}`
        or None on LLM failure / unparseable response (→ the turn stays untrained).

        P5 co-adaptive teacher: when `level_norm` is given (∈ [0,1], the agent's
        emergent proven ability read from the `mastery_level_state` SHM slot), the
        judge (b) injects a level-conditioned rubric and (a) shapes the reward
        relative-to-self (`raw − level_norm·gain·EMA(recent raw)`). `level_norm=None`
        ⇒ the legacy path: base template, raw reward, no EMA update — BYTE-IDENTICAL
        to pre-P5 (the `teacher_coadaptive_enabled=false` rollback, INV-MC-7)."""
        if not prompt or not response:
            return None
        coadaptive = level_norm is not None
        ln = max(0.0, min(1.0, float(level_norm))) if coadaptive else 0.0
        base = JUDGE_PROMPT_TEMPLATE.format(
            action=str(action or "direct"),
            prompt=str(prompt)[:1500], response=str(response)[:1500])
        full = (_rubric_line(ln) + base) if coadaptive else base
        try:
            raw = self._llm(full, self._timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TurnJudge] provider raised: %s", e)
            return None
        parsed = _parse_verdict(raw)
        if not parsed:
            return None
        raw_reward = _VERDICT_REWARD[parsed["verdict"]] * float(parsed["confidence"])
        reward = raw_reward
        if coadaptive:
            # (a) relative-to-self: subtract a level-scaled fraction of recent self.
            reward = raw_reward - (ln * self._relself_gain) * self._reward_ema
            self._reward_ema = ((1.0 - self._ema_alpha) * self._reward_ema
                                + self._ema_alpha * raw_reward)
        return {
            "reward": reward,
            "raw_reward": raw_reward,
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "version_tag": self._version_tag,
            "level_norm": ln,
        }
