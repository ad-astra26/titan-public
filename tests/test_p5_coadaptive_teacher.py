"""P5 — level-adaptive co-adaptive teacher (RFP_emergent_mastery_curriculum §7.P5;
ARCHITECTURE_mastery_leveling.md §4). Three hooks, all driven by the EMERGENT
ratcheted level (Q2: no hand-drawn curve) — (a) relative-to-self, (b) level-rubric
in the TurnJudge; (c) the graduated source-mix in self_learning_worker. "self" is
NOT a source (Q4 → the Inner Turn). flag-off / unpublished level ⇒ byte-identical.
"""
import json

from titan_hcl.synthesis.turn_judge import TurnJudge, _rubric_line
from titan_hcl.modules.self_learning_worker import (
    _graduated_source_weight, _REWARD_SOURCE_RANK,
)
from titan_hcl.synthesis.mastery_level import (
    mastery_readout_to_flat, mastery_flat_to_readout,
)


def _capturing_provider(verdict, confidence=1.0, sink=None):
    def _p(prompt, timeout_s):
        if sink is not None:
            sink.append(prompt)
        return json.dumps({"verdict": verdict, "rationale": "r", "confidence": confidence})
    return _p


def _judge(verdict, confidence=1.0, sink=None, **kw):
    return TurnJudge(llm_provider=_capturing_provider(verdict, confidence, sink),
                     model_id="m", **kw)


# ── (a) relative-to-self ────────────────────────────────────────────────────
def test_relative_to_self_diminishes_repeated_competence():
    """At level>0, a stream of identical 'good' turns earns LESS each time as the
    self-EMA rises (INV-MC-3 — repeating prior competence can't keep paying full)."""
    j = _judge("good", relself_gain=1.0, ema_alpha=0.5)
    r1 = j.score(prompt="q", action="direct", response="a", level_norm=0.8)["reward"]
    r2 = j.score(prompt="q", action="direct", response="a", level_norm=0.8)["reward"]
    r3 = j.score(prompt="q", action="direct", response="a", level_norm=0.8)["reward"]
    assert r1 > r2 > r3          # diminishing
    assert r1 == 1.0             # first turn: EMA still 0 → raw
    assert j.reward_ema > 0.0    # EMA tracked the recent self


def test_relative_to_self_scales_with_level():
    """The relative-to-self subtraction is level-scaled: a HIGHER level demands more
    (subtracts more of the recent-self EMA) than a lower one for the same history."""
    lo = _judge("good", relself_gain=1.0, ema_alpha=0.5)
    hi = _judge("good", relself_gain=1.0, ema_alpha=0.5)
    lo.score(prompt="q", action="direct", response="a", level_norm=0.2)
    hi.score(prompt="q", action="direct", response="a", level_norm=0.9)
    r_lo = lo.score(prompt="q", action="direct", response="a", level_norm=0.2)["reward"]
    r_hi = hi.score(prompt="q", action="direct", response="a", level_norm=0.9)["reward"]
    assert r_hi < r_lo           # higher level → harsher relative-to-self


# ── (b) level-conditioned rubric ────────────────────────────────────────────
def test_rubric_injected_only_when_coadaptive():
    sink_on, sink_off = [], []
    _judge("good", sink=sink_on).score(
        prompt="q", action="direct", response="a", level_norm=0.5)
    _judge("good", sink=sink_off).score(
        prompt="q", action="direct", response="a", level_norm=None)
    assert "GRADING STANDARD" in sink_on[0]       # rubric present at a level
    assert "GRADING STANDARD" not in sink_off[0]  # legacy prompt unchanged (None)


def test_rubric_bar_rises_with_level():
    assert "0th-percentile" in _rubric_line(0.0)
    assert "100th-percentile" in _rubric_line(1.0)
    assert "50th-percentile" in _rubric_line(0.5)


# ── flag-off / None parity (INV-MC-7 — byte-identical pre-P5) ────────────────
def test_level_none_is_byte_identical_legacy():
    """level_norm=None ⇒ raw reward, NO EMA mutation, base prompt — the rollback."""
    sink = []
    j = _judge("good", confidence=0.5, sink=sink)
    out = j.score(prompt="q", action="direct", response="a", level_norm=None)
    assert out["reward"] == 0.5 and out["raw_reward"] == 0.5   # raw, unshaped
    assert j.reward_ema == 0.0                                 # EMA untouched
    assert "GRADING STANDARD" not in sink[0]                   # base template


def test_level_zero_equals_raw():
    """level_norm=0.0 ⇒ the subtraction term is 0 → reward == raw (lenient floor)."""
    j = _judge("good")
    out = j.score(prompt="q", action="direct", response="a", level_norm=0.0)
    assert out["reward"] == out["raw_reward"] == 1.0


# ── (c) graduated source-mix (self_learning_worker) ─────────────────────────
_CFG = {"teacher_coadaptive_enabled": True, "teacher_judge_weight_floor": 0.3,
        "teacher_authority_rise_gain": 1.0}


def test_llm_judge_weight_decays_to_floor():
    base = 1.0
    w_lo = _graduated_source_weight("llm_judge", base, 0.0, _CFG)
    w_mid = _graduated_source_weight("llm_judge", base, 0.5, _CFG)
    w_hi = _graduated_source_weight("llm_judge", base, 1.0, _CFG)
    assert w_lo == 1.0                       # level 0 → full base
    assert w_mid == 0.5                       # 1 − 0.5
    assert w_hi == 0.3                         # floored (max(0.3, 0))
    assert w_lo > w_mid > w_hi               # monotone decay


def test_oracle_and_maker_weights_rise():
    assert _graduated_source_weight("oracle", 1.0, 0.0, _CFG) == 1.0
    assert _graduated_source_weight("oracle", 1.0, 1.0, _CFG) == 2.0   # ×(1+1·1)
    assert _graduated_source_weight("maker", 2.0, 0.5, _CFG) == 3.0    # 2×(1+0.5)


def test_user_weight_unchanged_and_no_self_source():
    assert _graduated_source_weight("user", 1.0, 1.0, _CFG) == 1.0     # not graduated
    # "self" is NOT a reward source in P5 (Q4 → Inner Turn) — falls through to base.
    assert _graduated_source_weight("self", 1.0, 1.0, _CFG) == 1.0


def test_graduated_mix_flag_off_is_base():
    off = {"teacher_coadaptive_enabled": False}
    assert _graduated_source_weight("llm_judge", 1.0, 1.0, off) == 1.0
    assert _graduated_source_weight("oracle", 1.0, 1.0, off) == 1.0


def test_graduated_mix_unpublished_level_is_base():
    # level_norm<=0 (slot not yet published) ⇒ static base regardless of source.
    assert _graduated_source_weight("llm_judge", 1.0, 0.0, _CFG) == 1.0
    assert _graduated_source_weight("oracle", 1.0, 0.0, _CFG) == 1.0


def test_authority_rank_order_unchanged():
    # P5 graduates only MAGNITUDES; the authority order (corrective-delta) is fixed.
    assert _REWARD_SOURCE_RANK == {"llm_judge": 0, "user": 1, "maker": 2, "oracle": 3}


# ── SHM level → level_norm normalization (the synthesis reader's math) ───────
def test_shm_level_roundtrips_to_level_norm():
    readout = {"level": 4.0, "grade": 4, "ema_v_symlog": 2.5, "competence": 0.7,
               "n_chunks": 3}
    flat = mastery_readout_to_flat(readout)
    back = mastery_flat_to_readout(flat)
    n_grades = 10
    level_norm = max(0.0, min(1.0, back["level"] / float(n_grades)))
    assert abs(level_norm - 0.4) < 1e-6
