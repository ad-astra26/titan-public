"""Phase B (§7.B B.2) — the synthesis turn-judge DAEMON drains fresh non-verifiable
turns → SELF_LEARN_REWARD{source:llm_judge} (the join trains the policy). Runs the
real `_turn_judge_loop` thread with a fake judge + send-queue."""
import collections
import threading
import time

from titan_hcl.bus import SELF_LEARN_REWARD
from titan_hcl.modules.synthesis_worker import _turn_judge_loop


class _FakeQ:
    def __init__(self):
        self.items = []

    def put_nowait(self, msg):
        self.items.append(msg)


class _FakeJudge:
    def __init__(self, reward):
        self._r = reward

    def score(self, *, prompt, action, response, level_norm=None):
        # level_norm added by P5 (the co-adaptive teacher reads the SHM level);
        # the daemon always passes it (None when co-adaptive is off).
        if not prompt or not response:
            return None
        return {"reward": self._r, "verdict": "good", "confidence": 1.0}


def _drain(queue, judge):
    sq = _FakeQ()
    stop = threading.Event()
    th = threading.Thread(target=_turn_judge_loop,
                          args=(queue, judge, sq, "T", stop, 0.05), daemon=True)
    th.start()
    deadline = time.time() + 4.0
    while time.time() < deadline and len(queue) > 0:
        time.sleep(0.02)
    time.sleep(0.15)          # let the in-flight pass finish emitting
    stop.set()
    th.join(timeout=2.0)
    return sq


def test_daemon_scores_and_emits_reward():
    q = collections.deque(maxlen=512)
    q.append({"reasoning_id": "rid1", "prompt": "what is sovereignty?",
              "action": "direct", "response": "self-governance",
              "goal_class": "philosophy"})
    sq = _drain(q, _FakeJudge(1.0))
    rewards = [m for m in sq.items if m["type"] == SELF_LEARN_REWARD]
    assert len(rewards) == 1
    p = rewards[0]["payload"]
    assert p["parent_tool_call_tx"] == "rid1"
    assert p["reward"] == 1.0
    assert p["source"] == "llm_judge"
    assert p["goal_class"] == "philosophy"


def test_daemon_emits_negative_reward():
    q = collections.deque(maxlen=512)
    q.append({"reasoning_id": "rid2", "prompt": "q", "action": "direct",
              "response": "bad", "goal_class": "g"})
    sq = _drain(q, _FakeJudge(-1.0))
    rewards = [m for m in sq.items if m["type"] == SELF_LEARN_REWARD]
    assert len(rewards) == 1 and rewards[0]["payload"]["reward"] == -1.0


def test_daemon_skips_judge_miss():
    q = collections.deque(maxlen=512)
    q.append({"reasoning_id": "rid3", "prompt": "", "action": "direct",
              "response": "", "goal_class": "g"})  # empty → judge returns None
    sq = _drain(q, _FakeJudge(1.0))
    assert [m for m in sq.items if m["type"] == SELF_LEARN_REWARD] == []
