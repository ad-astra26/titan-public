"""TaskCompletionJudge — P8.2 (RFP_emergent_mastery_curriculum §7.P8.2; INV-MC-8).

The correctness judge for AUTONOMOUS (no-chat) tool/research outcomes. It judges
*"does this real execution/retrieval EVIDENCE solve the concrete PROBLEM?"* — against
the actual stdout/finding, iterative-until-solved. This is **explicitly NOT** the
response-QUALITY `TurnJudge` (`turn_judge.py`): a quality judge reinforces plausible-
but-wrong actions and collapsed the policy
(`reference_oml_reward_must_be_correctness_aware_not_quality_aware`). A deterministic
oracle dominates wherever one exists (INV-MC-8); this judge is the fallback for the
common self-intent case where the generated code/query has NO pre-known answer.

Returns `{solved: bool, correction: str, confidence: float, verifiable: bool}` —
`correction` feeds the solve-until-correct retry (regenerate code / refine query).
`verifiable` (added 2026-06-22, INV-MC-8) is True only when the TASK has an objective,
checkable success criterion (a definite right/wrong answer or a concrete deliverable).
It is False for OPEN-ENDED / exploratory self-intents ("explore X", "reflect on Y")
that have no definite 'solved' state. The caller must NOT feed a negative reward for a
non-verifiable outcome: an open-ended exploration is structurally never "solved", and
the old binary `not solved → -0.5` POISONED the OML routing policy (research lane →
action=0 collapse; P9 perpetual-abandon). See
`reference_oml_reward_must_be_correctness_aware_not_quality_aware`. Mirrors `TurnJudge`'s
economics (stable prompt, strict-JSON parse, None on LLM failure).
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: float = 30.0

JUDGE_PROMPT_TEMPLATE = (
    "You are a strict CORRECTNESS judge for a sovereign AI agent's OWN autonomous "
    "work (no human is watching). The agent set itself a task and took the action "
    "'{action}' (tool=ran code; research=looked it up). Judge ONLY whether the real "
    "EVIDENCE below actually SOLVES the concrete TASK — not whether it reads well, "
    "not whether it ran without error. Code that executes cleanly but does NOT solve "
    "the task is NOT solved. If it is not solved, say concretely WHAT is wrong and HOW "
    "to fix it (this correction will guide a retry). ALSO judge whether the TASK is "
    "even VERIFIABLE: verifiable=true ONLY if it has an objective, checkable success "
    "criterion (a definite right/wrong answer or a concrete deliverable); "
    "verifiable=false if it is OPEN-ENDED or exploratory with no definite solved state "
    "(e.g. 'explore X', 'reflect on Y', 'understand Z better'). Output STRICT JSON with "
    "four fields: solved (true/false), correction (one concrete sentence; empty if "
    "solved), confidence (float 0.0..1.0), verifiable (true/false). No prose outside "
    "JSON.\n\n"
    "TASK:\n{problem}\n\nEVIDENCE (real execution/retrieval output):\n{evidence}\n"
)


def _parse(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    solved = obj.get("solved")
    if not isinstance(solved, bool):
        # tolerate string "true"/"false"
        if isinstance(solved, str) and solved.strip().lower() in ("true", "false"):
            solved = solved.strip().lower() == "true"
        else:
            return None
    conf = obj.get("confidence")
    try:
        conf_val = max(0.0, min(1.0, float(conf))) if conf is not None else 1.0
    except (TypeError, ValueError):
        conf_val = 1.0
    correction = str(obj.get("correction") or "")[:512]
    # verifiable (INV-MC-8, 2026-06-22): default True when missing/unparseable —
    # CONSERVATIVE so a malformed verdict never accidentally neutralizes a genuine
    # verifiable failure; only an EXPLICIT false marks the outcome non-verifiable.
    verifiable = obj.get("verifiable")
    if not isinstance(verifiable, bool):
        if isinstance(verifiable, str) and verifiable.strip().lower() in ("true", "false"):
            verifiable = verifiable.strip().lower() == "true"
        else:
            verifiable = True
    return {"solved": bool(solved), "correction": correction,
            "confidence": conf_val, "verifiable": bool(verifiable)}


class TaskCompletionJudge:
    """Score one autonomous task outcome → `{solved, correction, confidence, verifiable}` or None
    on LLM failure / unparseable response. A DISTINCT module + prompt from `TurnJudge`
    (INV-MC-8): completion against real evidence, never response quality.

    llm_provider: `(prompt: str, timeout_s: float) -> str` (tests inject a mock)."""

    def __init__(
        self,
        *,
        llm_provider: Callable[[str, float], str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._llm = llm_provider
        self._timeout_s = float(timeout_s)

    def judge(self, *, problem: str, action: str, evidence: str) -> Optional[dict]:
        """Return `{solved, correction, confidence, verifiable}` or None (LLM miss → the caller
        treats it as not-yet-solved without a correction → no false positive reward)."""
        if not (problem or "").strip() or not (evidence or "").strip():
            return None
        full = JUDGE_PROMPT_TEMPLATE.format(
            action=str(action or "tool"),
            problem=str(problem)[:1500], evidence=str(evidence)[:1500])
        try:
            raw = self._llm(full, self._timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TaskCompletionJudge] provider raised: %s", e)
            return None
        return _parse(raw)
