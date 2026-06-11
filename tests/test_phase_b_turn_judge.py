"""Phase B (§7.B B.2) — TurnJudge scores a non-verifiable turn → policy reward."""
import json

from titan_hcl.synthesis.turn_judge import TurnJudge


def _provider(verdict, confidence=1.0):
    def _p(prompt, timeout_s):
        return json.dumps({"verdict": verdict, "rationale": "r", "confidence": confidence})
    return _p


def _judge(verdict, confidence=1.0):
    return TurnJudge(llm_provider=_provider(verdict, confidence), model_id="m")


def test_good_turn_positive_reward():
    out = _judge("good").score(prompt="What is sovereignty?", action="direct",
                               response="Sovereignty is self-governance...")
    assert out is not None and out["reward"] == 1.0 and out["verdict"] == "good"


def test_poor_turn_negative_reward():
    out = _judge("poor").score(prompt="What's 2+2?", action="direct",
                               response="purple")
    assert out["reward"] == -1.0


def test_ok_turn_zero_reward():
    out = _judge("ok").score(prompt="hi", action="direct", response="hello")
    assert out["reward"] == 0.0


def test_confidence_scales_reward():
    out = _judge("good", confidence=0.5).score(
        prompt="q", action="direct", response="a")
    assert out["reward"] == 0.5


def test_empty_prompt_or_response_is_none():
    assert _judge("good").score(prompt="", action="direct", response="a") is None
    assert _judge("good").score(prompt="q", action="direct", response="") is None


def test_malformed_llm_output_is_none():
    j = TurnJudge(llm_provider=lambda p, t: "not json", model_id="m")
    assert j.score(prompt="q", action="direct", response="a") is None


def test_provider_exception_is_none():
    def _boom(p, t):
        raise RuntimeError("provider down")
    j = TurnJudge(llm_provider=_boom, model_id="m")
    assert j.score(prompt="q", action="direct", response="a") is None


def test_version_tag_stable_and_tagged():
    assert _judge("good").version_tag.startswith("turn|m|")
