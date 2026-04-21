"""Tests for TitanVM v2 Phase 1 — new opcodes + VMRuntimeState + persistence.

Covers rFP_titan_vm_v2 Phase 1 acceptance (§3.6):
  - 5 new opcodes: LOAD_EMA, LOAD_DT, SIGMOID, SOFT_GT, SOFT_LT
  - VMRuntimeState persistence (atomic + corruption-guard)
  - execute() accepts program_key for state attribution
  - Per-program telemetry exposed via get_telemetry()
  - No regression in existing v1 opcode behavior (covered by test_titan_vm.py)
"""
import json
import math
import os

import pytest

from titan_plugin.logic.titan_vm import Op, TitanVM, VMRuntimeState, _sigmoid


class _FakeRegister:
    """Minimal state_register stub for opcode testing."""

    def __init__(self, values: dict):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __getattr__(self, name):
        # Allow getattr(self, "body_tensor") to return list value if present
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        raise AttributeError(name)


class TestVMRuntimeState:

    def test_ema_bootstrap_on_first_call(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        val = rs.get_ema("REFLEX", "body_tensor.0", 0.5, alpha=0.1)
        assert val == pytest.approx(0.5)

    def test_ema_updates_correctly(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        rs.get_ema("REFLEX", "body_tensor.0", 0.5, alpha=0.1)   # bootstrap
        v = rs.get_ema("REFLEX", "body_tensor.0", 1.0, alpha=0.1)
        # (1-0.1)*0.5 + 0.1*1.0 = 0.55
        assert v == pytest.approx(0.55)

    def test_ema_is_per_program(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        rs.get_ema("A", "x", 1.0, 0.1)  # A.x bootstraps at 1.0
        rs.get_ema("B", "x", 5.0, 0.1)  # B.x bootstraps at 5.0
        assert rs.ema_state["A"]["x"] == pytest.approx(1.0)
        assert rs.ema_state["B"]["x"] == pytest.approx(5.0)

    def test_ema_without_program_key_returns_current(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        val = rs.get_ema(None, "x", 0.7, 0.1)
        assert val == pytest.approx(0.7)
        # No tracking for None program_key
        assert rs.ema_state == {}

    def test_dt_first_call_is_zero(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        dt = rs.get_dt("REFLEX", "path", 1.5)
        assert dt == 0.0
        # Subsequent call gives delta
        dt2 = rs.get_dt("REFLEX", "path", 2.0)
        assert dt2 == pytest.approx(0.5)

    def test_dt_sign_preserved(self, tmp_path):
        rs = VMRuntimeState(data_dir=str(tmp_path), load=False)
        rs.get_dt("P", "x", 1.0)
        dt = rs.get_dt("P", "x", 0.4)
        assert dt == pytest.approx(-0.6)

    def test_persistence_roundtrip(self, tmp_path):
        rs1 = VMRuntimeState(data_dir=str(tmp_path), load=False)
        rs1.get_ema("REFLEX", "body_tensor.0", 0.5, 0.1)
        rs1.get_ema("REFLEX", "body_tensor.0", 1.0, 0.1)
        rs1.get_dt("CURIOSITY", "outer_mind.magnitude", 0.3)
        rs1.get_dt("CURIOSITY", "outer_mind.magnitude", 0.5)
        rs1.note_execution()
        rs1.save()

        state_path = os.path.join(str(tmp_path), "titan_vm_runtime.json")
        assert os.path.exists(state_path)

        rs2 = VMRuntimeState(data_dir=str(tmp_path), load=True)
        assert rs2.ema_state["REFLEX"]["body_tensor.0"] == pytest.approx(0.55)
        assert rs2.prev_values["CURIOSITY"]["outer_mind.magnitude"] == pytest.approx(0.5)
        assert rs2.total_executions == 1

    def test_persistence_corruption_guard(self, tmp_path, caplog):
        import logging
        state_path = os.path.join(str(tmp_path), "titan_vm_runtime.json")
        with open(state_path, "w") as f:
            f.write("{not valid json at all")

        with caplog.at_level(logging.WARNING, logger="titan_plugin.logic.titan_vm"):
            rs = VMRuntimeState(data_dir=str(tmp_path), load=True)

        # No crash, fresh defaults
        assert rs.ema_state == {}
        assert rs.prev_values == {}
        assert rs.total_executions == 0
        assert any("Load failed" in r.message for r in caplog.records)

    def test_schema_mismatch_rejected(self, tmp_path, caplog):
        import logging
        state_path = os.path.join(str(tmp_path), "titan_vm_runtime.json")
        with open(state_path, "w") as f:
            json.dump({"schema": 99, "ema_state": {"X": {"y": 0.5}}}, f)

        with caplog.at_level(logging.WARNING, logger="titan_plugin.logic.titan_vm"):
            rs = VMRuntimeState(data_dir=str(tmp_path), load=True)

        assert rs.ema_state == {}  # schema 99 rejected, defaults used
        assert any("Schema mismatch" in r.message for r in caplog.records)


class TestSigmoidHelper:

    def test_sigmoid_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_monotonic(self):
        assert _sigmoid(-1.0) < _sigmoid(0.0) < _sigmoid(1.0)

    def test_sigmoid_bounds(self):
        # Stable for large magnitudes
        assert 0.0 <= _sigmoid(1e6) <= 1.0
        assert 0.0 <= _sigmoid(-1e6) <= 1.0
        assert _sigmoid(1e6) == pytest.approx(1.0)
        assert _sigmoid(-1e6) == pytest.approx(0.0)


class TestLoadEmaOpcode:

    def test_first_call_bootstraps_to_current(self, tmp_path):
        reg = _FakeRegister({"body_tensor": [0.5, 0.3]})
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(state_register=reg, runtime_state=runtime)
        program = [
            (Op.LOAD_EMA, "body_tensor.0", 0.1),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(program, program_key="REFLEX")
        assert result.error is None
        assert result.score == pytest.approx(0.5)

    def test_converges_toward_target(self, tmp_path):
        reg_a = _FakeRegister({"body_tensor": [0.0]})
        reg_b = _FakeRegister({"body_tensor": [1.0]})
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)

        # First run: register has 0.0, EMA bootstraps to 0
        vm_a = TitanVM(state_register=reg_a, runtime_state=runtime)
        vm_a.execute(
            [(Op.LOAD_EMA, "body_tensor.0", 0.5), (Op.SCORE,), (Op.HALT,)],
            program_key="FOCUS",
        )

        # Second run: register has 1.0, EMA is (1-0.5)*0 + 0.5*1 = 0.5
        vm_b = TitanVM(state_register=reg_b, runtime_state=runtime)
        r = vm_b.execute(
            [(Op.LOAD_EMA, "body_tensor.0", 0.5), (Op.SCORE,), (Op.HALT,)],
            program_key="FOCUS",
        )
        assert r.score == pytest.approx(0.5)

    def test_isolation_between_programs(self, tmp_path):
        reg = _FakeRegister({"body_tensor": [0.9]})
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(state_register=reg, runtime_state=runtime)

        # Two programs with the same LOAD_EMA path — independent state
        p = [(Op.LOAD_EMA, "body_tensor.0", 0.5), (Op.SCORE,), (Op.HALT,)]
        a1 = vm.execute(p, program_key="A")  # A bootstraps → 0.9
        b1 = vm.execute(p, program_key="B")  # B bootstraps → 0.9

        # Change the observed value; each should update independently
        reg._values["body_tensor"] = [0.1]
        a2 = vm.execute(p, program_key="A")  # (1-0.5)*0.9 + 0.5*0.1 = 0.5
        b2 = vm.execute(p, program_key="B")  # same math → 0.5

        assert a2.score == pytest.approx(0.5)
        assert b2.score == pytest.approx(0.5)
        assert runtime.ema_state["A"]["body_tensor.0"] == pytest.approx(0.5)
        assert runtime.ema_state["B"]["body_tensor.0"] == pytest.approx(0.5)


class TestLoadDtOpcode:

    def test_first_call_returns_zero(self, tmp_path):
        reg = _FakeRegister({"body_tensor": [0.7]})
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(state_register=reg, runtime_state=runtime)
        r = vm.execute(
            [(Op.LOAD_DT, "body_tensor.0"), (Op.SCORE,), (Op.HALT,)],
            program_key="REFLEX",
        )
        assert r.score == pytest.approx(0.0)

    def test_delta_computed_correctly(self, tmp_path):
        reg = _FakeRegister({"body_tensor": [0.3]})
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(state_register=reg, runtime_state=runtime)
        program = [(Op.LOAD_DT, "body_tensor.0"), (Op.SCORE,), (Op.HALT,)]

        vm.execute(program, program_key="REFLEX")   # returns 0.0, records 0.3
        reg._values["body_tensor"] = [0.8]
        r = vm.execute(program, program_key="REFLEX")  # 0.8 - 0.3 = 0.5
        assert r.score == pytest.approx(0.5)

        reg._values["body_tensor"] = [0.2]
        r2 = vm.execute(program, program_key="REFLEX")  # 0.2 - 0.8 = -0.6
        assert r2.score == pytest.approx(-0.6)


class TestSmoothGates:

    def test_sigmoid_opcode(self, tmp_path):
        vm = TitanVM(runtime_state=VMRuntimeState(data_dir=str(tmp_path), load=False))
        # SIGMOID k=1, input=0 → 0.5
        r = vm.execute(
            [(Op.PUSH, 0.0), (Op.SIGMOID, 1.0), (Op.SCORE,), (Op.HALT,)],
        )
        assert r.score == pytest.approx(0.5)

    def test_soft_gt_smoothly_transitions(self, tmp_path):
        vm = TitanVM(runtime_state=VMRuntimeState(data_dir=str(tmp_path), load=False))
        # SOFT_GT threshold=0.5, k=10. At x=0.5, value should be ~0.5 (at threshold)
        p = lambda x: [(Op.PUSH, x), (Op.SOFT_GT, 0.5, 10.0), (Op.SCORE,), (Op.HALT,)]

        assert vm.execute(p(0.5)).score == pytest.approx(0.5)
        assert vm.execute(p(1.0)).score > 0.9   # well above threshold
        assert vm.execute(p(0.0)).score < 0.1   # well below
        # Monotonicity
        lo = vm.execute(p(0.3)).score
        mid = vm.execute(p(0.5)).score
        hi = vm.execute(p(0.7)).score
        assert lo < mid < hi

    def test_soft_lt_is_inverse_of_soft_gt(self, tmp_path):
        vm = TitanVM(runtime_state=VMRuntimeState(data_dir=str(tmp_path), load=False))
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r_gt = vm.execute(
                [(Op.PUSH, x), (Op.SOFT_GT, 0.5, 10.0), (Op.SCORE,), (Op.HALT,)]).score
            r_lt = vm.execute(
                [(Op.PUSH, x), (Op.SOFT_LT, 0.5, 10.0), (Op.SCORE,), (Op.HALT,)]).score
            # soft_gt(x, t, k) + soft_lt(x, t, k) should equal 1.0
            assert r_gt + r_lt == pytest.approx(1.0)

    def test_stack_output_stays_in_unit_interval(self, tmp_path):
        vm = TitanVM(runtime_state=VMRuntimeState(data_dir=str(tmp_path), load=False))
        # Extreme inputs must not escape [0, 1]
        for x in [-1e6, -10.0, 0.0, 10.0, 1e6]:
            r = vm.execute([(Op.PUSH, x), (Op.SIGMOID, 1.0), (Op.SCORE,), (Op.HALT,)])
            assert 0.0 <= r.score <= 1.0


class TestTelemetry:

    def test_per_program_fire_count(self, tmp_path):
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(runtime_state=runtime, config={"min_reward_threshold": 0.5})
        # Three high-scoring runs on REFLEX, two low on FOCUS
        hi = [(Op.PUSH, 0.8), (Op.SCORE,), (Op.HALT,)]
        lo = [(Op.PUSH, 0.1), (Op.SCORE,), (Op.HALT,)]
        for _ in range(3):
            vm.execute(hi, program_key="REFLEX")
        for _ in range(2):
            vm.execute(lo, program_key="FOCUS")

        tel = vm.get_telemetry()
        assert tel["REFLEX"]["fire_count"] == 3
        assert tel["REFLEX"]["scored_count"] == 3
        assert tel["REFLEX"]["avg_score"] == pytest.approx(0.8)
        assert tel["FOCUS"]["fire_count"] == 0   # below 0.5 threshold
        assert tel["FOCUS"]["scored_count"] == 2
        assert tel["FOCUS"]["avg_score"] == pytest.approx(0.1)

    def test_runtime_save_cadence(self, tmp_path):
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(runtime_state=runtime, config={"runtime_save_every": 3})
        p = [(Op.PUSH, 0.1), (Op.SCORE,), (Op.HALT,)]
        for _ in range(3):
            vm.execute(p, program_key="X")
        # After 3 executions, save should have fired
        state_path = os.path.join(str(tmp_path), "titan_vm_runtime.json")
        assert os.path.exists(state_path)


class TestBackwardCompat:

    def test_existing_bytecode_unchanged(self, tmp_path):
        """Classic v1 program still executes correctly — no regressions."""
        reg = _FakeRegister({"body_tensor": [0.7, 0.3]})
        vm = TitanVM(
            state_register=reg,
            runtime_state=VMRuntimeState(data_dir=str(tmp_path), load=False),
        )
        program = [
            (Op.LOAD, "body_tensor.0"),
            (Op.PUSH, 0.5),
            (Op.CMP_GT,),
            (Op.BRANCH_IF, "high"),
            (Op.PUSH, 0.0),
            (Op.JMP, "end"),
            ("high",),
            (Op.PUSH, 0.8),
            ("end",),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(program)  # no program_key — works fine
        assert result.score == pytest.approx(0.8)
        assert result.error is None

    def test_execute_without_program_key_still_works(self, tmp_path):
        """program_key is optional — ad-hoc evaluation should not fail."""
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        vm = TitanVM(runtime_state=runtime)
        r = vm.execute(
            [(Op.LOAD_EMA, "context.x", 0.1), (Op.SCORE,), (Op.HALT,)],
            context={"x": 0.7},
        )
        # Without program_key, LOAD_EMA returns the current value (no tracking)
        assert r.score == pytest.approx(0.7)
        assert runtime.ema_state == {}  # no state recorded


class TestEndpointPayloadShape:
    """Exercises the file-fallback payload construction logic of /v4/titan-vm.

    The endpoint reads data/neural_nervous_system/titan_vm_runtime.json when
    TitanVM lives in spirit_worker subprocess (not directly accessible from
    dashboard). This test writes a state file and verifies the derived
    per-program view matches what /v4/titan-vm would serialize.
    """

    def test_file_fallback_payload(self, tmp_path):
        runtime = VMRuntimeState(data_dir=str(tmp_path), load=False)
        # Seed plausible state: REFLEX has 2 EMA paths + 1 DT path; FOCUS has 1 each
        runtime.get_ema("REFLEX", "body_tensor.0", 0.6, 0.1)
        runtime.get_ema("REFLEX", "body_tensor.0", 0.8, 0.1)  # bootstrap 0.6 → 0.62
        runtime.get_ema("REFLEX", "consciousness.drift", 0.3, 0.1)
        runtime.get_dt("REFLEX", "body_tensor.0", 0.8)
        runtime.get_ema("FOCUS", "mind_tensor.0", 0.5, 0.1)
        runtime.get_dt("FOCUS", "mind_tensor.0", 0.5)
        runtime.note_execution()
        runtime.note_execution()
        runtime.save()

        # Replay the endpoint's file-fallback logic (independent read)
        import os, json, time
        state_path = os.path.join(str(tmp_path), "titan_vm_runtime.json")
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        total_executions = int(state.get("total_executions", 0))
        ema = state.get("ema_state", {})
        prev = state.get("prev_values", {})
        all_keys = set(ema.keys()) | set(prev.keys())
        programs = {}
        for key in all_keys:
            programs[key] = {
                "ema_paths": {p: round(float(v), 4) for p, v in (ema.get(key, {}) or {}).items()},
                "prev_paths": {p: round(float(v), 4) for p, v in (prev.get(key, {}) or {}).items()},
            }

        assert total_executions == 2
        assert "REFLEX" in programs
        assert "FOCUS" in programs
        assert set(programs["REFLEX"]["ema_paths"].keys()) == {"body_tensor.0", "consciousness.drift"}
        assert set(programs["REFLEX"]["prev_paths"].keys()) == {"body_tensor.0"}
        assert set(programs["FOCUS"]["ema_paths"].keys()) == {"mind_tensor.0"}
        # EMA values should be in reasonable range
        for key, data in programs.items():
            for p, v in data["ema_paths"].items():
                assert 0.0 <= v <= 1.0
