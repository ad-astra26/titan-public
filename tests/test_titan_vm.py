"""
Tests for TitanVM — stack-based micro-instruction set and R5 scoring programs.

Tests cover:
  - VM stack operations (PUSH, POP, DUP, SWAP, ROT)
  - Math operations (ADD, SUB, MUL, DIV, ABS, CLAMP)
  - Compare operations (CMP_GT, CMP_LT, CMP_EQ)
  - Flow control (JMP, BRANCH_IF, HALT, labels)
  - State access (LOAD from StateRegister, STORE/LOAD registers)
  - Special opcodes (SCORE, CLOCK, AGE, EMIT)
  - R5 scoring programs (reflex_score, valence_boost)
  - Integration: VM + StateRegister + scoring flow
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from titan_plugin.logic.titan_vm import TitanVM, Op, VMResult, MAX_STACK_DEPTH


# ── Helpers ────────────────────────────────────────────────────────

class MockStateRegister:
    """Minimal StateRegister mock for VM tests."""

    def __init__(self, state=None):
        self._state = state or {}
        self._start = time.monotonic()

    @property
    def body_tensor(self):
        return self._state.get("body_tensor", [0.5] * 5)

    @property
    def mind_tensor(self):
        return self._state.get("mind_tensor", [0.5] * 5)

    @property
    def spirit_tensor(self):
        return self._state.get("spirit_tensor", [0.5] * 5)

    @property
    def consciousness(self):
        return self._state.get("consciousness", {"drift": 0.1, "epoch_number": 5})

    @property
    def focus_body(self):
        return self._state.get("focus_body", [0.0] * 5)

    @property
    def focus_mind(self):
        return self._state.get("focus_mind", [0.0] * 5)

    @property
    def metabolic(self):
        return self._state.get("metabolic", {"energy_state": "ACTIVE", "sol_balance": 5.0})

    def get(self, key, default=None):
        return self._state.get(key, default)

    def age_seconds(self):
        return time.monotonic() - self._start


# ── Test: Stack Operations ─────────────────────────────────────────

class TestStackOps:
    def test_push_pop(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 42.0),
            (Op.PUSH, 7.0),
            (Op.POP,),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 42.0
        assert result.halted

    def test_dup(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 3.0),
            (Op.DUP,),
            (Op.ADD,),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 6.0

    def test_swap(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 10.0),
            (Op.PUSH, 3.0),
            (Op.SWAP,),
            (Op.SUB,),       # 3 - 10 = -7
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == -7.0

    def test_rot(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 1.0),   # a
            (Op.PUSH, 2.0),   # b
            (Op.PUSH, 3.0),   # c
            (Op.ROT,),        # → b c a → stack: [2, 3, 1]
            (Op.SCORE,),      # pops 1.0 (a, now on top)
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 1.0

    def test_stack_overflow(self):
        vm = TitanVM()
        prog = [(Op.PUSH, 1.0)] * (MAX_STACK_DEPTH + 1) + [(Op.HALT,)]
        result = vm.execute(prog)
        assert result.error == "stack overflow"


# ── Test: Math Operations ──────────────────────────────────────────

class TestMathOps:
    def test_add(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 2.5), (Op.PUSH, 3.5), (Op.ADD,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 6.0

    def test_sub(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 10.0), (Op.PUSH, 3.0), (Op.SUB,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 7.0

    def test_mul(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 4.0), (Op.PUSH, 5.0), (Op.MUL,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 20.0

    def test_div(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 10.0), (Op.PUSH, 4.0), (Op.DIV,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 2.5

    def test_div_by_zero(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 10.0), (Op.PUSH, 0.0), (Op.DIV,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 0.0  # safe div by zero

    def test_abs(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, -5.0), (Op.ABS,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 5.0

    def test_clamp(self):
        vm = TitanVM()
        # CLAMP pops: value, min, max
        result = vm.execute([
            (Op.PUSH, 10.0),   # value
            (Op.PUSH, 0.0),    # min
            (Op.PUSH, 5.0),    # max
            (Op.CLAMP,),       # clamp(10, 0, 5) = 5
            (Op.SCORE,),
            (Op.HALT,),
        ])
        assert result.score == 5.0

    def test_clamp_below(self):
        vm = TitanVM()
        result = vm.execute([
            (Op.PUSH, -3.0),
            (Op.PUSH, 0.0),
            (Op.PUSH, 1.0),
            (Op.CLAMP,),       # clamp(-3, 0, 1) = 0
            (Op.SCORE,),
            (Op.HALT,),
        ])
        assert result.score == 0.0


# ── Test: Compare Operations ──────────────────────────────────────

class TestCompareOps:
    def test_cmp_gt_true(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 5.0), (Op.PUSH, 3.0), (Op.CMP_GT,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 1.0

    def test_cmp_gt_false(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 3.0), (Op.PUSH, 5.0), (Op.CMP_GT,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 0.0

    def test_cmp_lt_true(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 3.0), (Op.PUSH, 5.0), (Op.CMP_LT,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 1.0

    def test_cmp_eq_true(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 5.0), (Op.PUSH, 5.0), (Op.CMP_EQ,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 1.0

    def test_cmp_eq_close(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 5.0), (Op.PUSH, 5.0005), (Op.CMP_EQ,), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 1.0  # within 0.001 tolerance


# ── Test: Flow Control ────────────────────────────────────────────

class TestFlowControl:
    def test_jmp(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 1.0),
            (Op.JMP, "skip"),
            (Op.PUSH, 99.0),     # should be skipped
            ("skip",),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 1.0

    def test_branch_if_taken(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 1.0),        # truthy
            (Op.BRANCH_IF, "yes"),
            (Op.PUSH, 0.0),
            (Op.JMP, "done"),
            ("yes",),
            (Op.PUSH, 42.0),
            ("done",),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 42.0

    def test_branch_if_not_taken(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 0.0),        # falsy
            (Op.BRANCH_IF, "yes"),
            (Op.PUSH, 10.0),
            (Op.JMP, "done"),
            ("yes",),
            (Op.PUSH, 42.0),
            ("done",),
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 10.0

    def test_unknown_label(self):
        vm = TitanVM()
        prog = [(Op.JMP, "nonexistent"), (Op.HALT,)]
        result = vm.execute(prog)
        assert "unknown label" in result.error

    def test_halt_stops(self):
        vm = TitanVM()
        prog = [(Op.HALT,), (Op.PUSH, 99.0)]
        result = vm.execute(prog)
        assert result.halted
        assert result.instructions_executed == 1

    def test_instruction_limit(self):
        vm = TitanVM()
        # Infinite loop
        prog = [
            ("loop",),
            (Op.PUSH, 1.0),
            (Op.POP,),
            (Op.JMP, "loop"),
        ]
        result = vm.execute(prog)
        assert result.error == "instruction limit exceeded"


# ── Test: State Access ────────────────────────────────────────────

class TestStateAccess:
    def test_load_body_tensor_index(self):
        reg = MockStateRegister({"body_tensor": [0.1, 0.2, 0.3, 0.4, 0.5]})
        vm = TitanVM(state_register=reg)
        result = vm.execute([(Op.LOAD, "body_tensor.2"), (Op.SCORE,), (Op.HALT,)])
        assert result.score == pytest.approx(0.3)

    def test_load_body_tensor_average(self):
        reg = MockStateRegister({"body_tensor": [0.2, 0.4, 0.6, 0.8, 1.0]})
        vm = TitanVM(state_register=reg)
        result = vm.execute([(Op.LOAD, "body_tensor"), (Op.SCORE,), (Op.HALT,)])
        assert result.score == pytest.approx(0.6)  # average

    def test_load_consciousness_drift(self):
        reg = MockStateRegister({"consciousness": {"drift": 0.42, "epoch_number": 10}})
        vm = TitanVM(state_register=reg)
        result = vm.execute([(Op.LOAD, "consciousness.drift"), (Op.SCORE,), (Op.HALT,)])
        assert result.score == pytest.approx(0.42)

    def test_load_context_value(self):
        vm = TitanVM()
        result = vm.execute(
            [(Op.LOAD, "context.intensity"), (Op.SCORE,), (Op.HALT,)],
            context={"intensity": 0.75}
        )
        assert result.score == pytest.approx(0.75)

    def test_load_missing_returns_zero(self):
        vm = TitanVM()  # no state register
        result = vm.execute([(Op.LOAD, "body_tensor.0"), (Op.SCORE,), (Op.HALT,)])
        assert result.score == 0.0

    def test_store_and_load_register(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 7.5),
            (Op.STORE, "my_var"),
            (Op.PUSH, 1.0),       # push something else
            (Op.POP,),            # discard
            (Op.LOAD, "my_var"),  # load stored value
            (Op.SCORE,),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert result.score == 7.5
        assert result.registers["my_var"] == 7.5

    def test_age(self):
        reg = MockStateRegister()
        vm = TitanVM(state_register=reg)
        result = vm.execute([(Op.AGE,), (Op.SCORE,), (Op.HALT,)])
        assert result.score >= 0.0  # age is always non-negative
        assert result.score < 1.0   # should be very recent


# ── Test: Special Opcodes ─────────────────────────────────────────

class TestSpecialOps:
    def test_score(self):
        vm = TitanVM()
        result = vm.execute([(Op.PUSH, 0.87), (Op.SCORE,), (Op.HALT,)])
        assert result.score == pytest.approx(0.87)

    def test_clock(self):
        vm = TitanVM()
        result = vm.execute([(Op.CLOCK,), (Op.SCORE,), (Op.HALT,)])
        assert result.score > 0  # monotonic time

    def test_emit(self):
        vm = TitanVM()
        prog = [
            (Op.PUSH, 0.5),
            (Op.STORE, "test_val"),
            (Op.PUSH, 0.42),
            (Op.EMIT, "TEST_MSG"),
            (Op.HALT,),
        ]
        result = vm.execute(prog)
        assert len(result.emissions) == 1
        assert result.emissions[0]["type"] == "TEST_MSG"
        assert result.emissions[0]["value"] == pytest.approx(0.42)
        assert result.emissions[0]["registers"]["test_val"] == 0.5

    def test_emit_with_bus(self):
        mock_bus = MagicMock()
        vm = TitanVM(bus=mock_bus)
        result = vm.execute([(Op.PUSH, 1.0), (Op.EMIT, "TEST"), (Op.HALT,)])
        assert mock_bus.publish.called


# ── Test: R5 Scoring Programs ────────────────────────────────────

class TestReflexScoreProgram:
    def test_program_loads(self):
        from titan_plugin.logic.vm_programs import get_program
        prog = get_program("reflex_score")
        assert len(prog) > 0
        assert any(isinstance(i, tuple) and i[0] == Op.SCORE for i in prog)

    def test_program_executes_with_good_state(self):
        """Good convergence + high engagement + fresh state → high score."""
        from titan_plugin.logic.vm_programs import get_program

        reg = MockStateRegister({
            "body_tensor": [0.6, 0.6, 0.6, 0.6, 0.6],    # avg 0.6
            "mind_tensor": [0.6, 0.6, 0.6, 0.6, 0.6],    # avg 0.6 (perfect convergence)
            "spirit_tensor": [0.6, 0.6, 0.6, 0.6, 0.6],  # avg 0.6
            "consciousness": {"drift": 0.1, "epoch_number": 10},
        })
        vm = TitanVM(state_register=reg)
        result = vm.execute(get_program("reflex_score"), context={
            "intensity": 0.8,
            "engagement": 0.9,
            "valence": 0.5,
            "reflexes_fired": 3.0,
            "reflexes_succeeded": 3.0,
        })
        assert result.halted
        assert result.error is None
        # Perfect convergence (0.3) + high engagement (~0.21) + perfect hit (0.2) + fresh (0.1) + stable (0.15) ≈ 0.96
        assert result.score > 0.7, f"Expected high score for good state, got {result.score}"

    def test_program_executes_with_poor_state(self):
        """Poor convergence + low engagement + high drift → low score."""
        from titan_plugin.logic.vm_programs import get_program

        reg = MockStateRegister({
            "body_tensor": [0.1, 0.1, 0.1, 0.1, 0.1],    # avg 0.1
            "mind_tensor": [0.9, 0.9, 0.9, 0.9, 0.9],    # avg 0.9 (huge divergence)
            "spirit_tensor": [0.5, 0.5, 0.5, 0.5, 0.5],  # avg 0.5
            "consciousness": {"drift": 0.8, "epoch_number": 10},
        })
        vm = TitanVM(state_register=reg)
        result = vm.execute(get_program("reflex_score"), context={
            "intensity": 0.1,
            "engagement": 0.1,
            "valence": -0.5,
            "reflexes_fired": 0.0,
            "reflexes_succeeded": 0.0,
        })
        assert result.halted
        assert result.error is None
        # Poor convergence + low engagement + no reflexes + high drift → low score
        assert result.score < 0.3, f"Expected low score for poor state, got {result.score}"

    def test_program_no_reflexes_still_scores(self):
        """Even with no reflexes fired, convergence/engagement/freshness contribute."""
        from titan_plugin.logic.vm_programs import get_program

        reg = MockStateRegister({
            "body_tensor": [0.5, 0.5, 0.5, 0.5, 0.5],
            "mind_tensor": [0.5, 0.5, 0.5, 0.5, 0.5],
            "spirit_tensor": [0.5, 0.5, 0.5, 0.5, 0.5],
            "consciousness": {"drift": 0.2},
        })
        vm = TitanVM(state_register=reg)
        result = vm.execute(get_program("reflex_score"), context={
            "intensity": 0.5,
            "engagement": 0.5,
            "valence": 0.0,
            "reflexes_fired": 0.0,
            "reflexes_succeeded": 0.0,
        })
        assert result.halted
        assert result.error is None
        assert result.score > 0.0  # convergence + engagement + freshness contribute

    def test_program_emits_reflex_reward(self):
        """Program should emit REFLEX_REWARD to bus."""
        from titan_plugin.logic.vm_programs import get_program

        reg = MockStateRegister()
        vm = TitanVM(state_register=reg)
        result = vm.execute(get_program("reflex_score"), context={
            "intensity": 0.5,
            "engagement": 0.5,
            "valence": 0.0,
            "reflexes_fired": 1.0,
            "reflexes_succeeded": 1.0,
        })
        assert len(result.emissions) == 1
        assert result.emissions[0]["type"] == "REFLEX_REWARD"


class TestValenceBoostProgram:
    def test_positive_valence_high_engagement(self):
        from titan_plugin.logic.vm_programs import get_program
        vm = TitanVM()
        result = vm.execute(get_program("valence_boost"), context={
            "valence": 0.8,
            "engagement": 0.9,
        })
        assert result.halted
        assert result.score > 0.0  # positive boost

    def test_negative_valence_penalty(self):
        from titan_plugin.logic.vm_programs import get_program
        vm = TitanVM()
        result = vm.execute(get_program("valence_boost"), context={
            "valence": -0.8,
            "engagement": 0.5,
        })
        assert result.halted
        assert result.score == pytest.approx(-0.05)

    def test_neutral_valence_no_modifier(self):
        from titan_plugin.logic.vm_programs import get_program
        vm = TitanVM()
        result = vm.execute(get_program("valence_boost"), context={
            "valence": 0.0,
            "engagement": 0.5,
        })
        assert result.halted
        assert result.score == pytest.approx(0.0)


class TestProgramRegistry:
    def test_get_program_exists(self):
        from titan_plugin.logic.vm_programs import get_program
        prog = get_program("reflex_score")
        assert len(prog) > 10

    def test_get_program_unknown(self):
        from titan_plugin.logic.vm_programs import get_program
        with pytest.raises(KeyError):
            get_program("nonexistent_program")

    def test_all_programs_execute(self):
        """Every registered program should execute without error."""
        from titan_plugin.logic.vm_programs import PROGRAMS
        vm = TitanVM(state_register=MockStateRegister())
        for name, factory in PROGRAMS.items():
            result = vm.execute(factory(), context={
                "intensity": 0.5,
                "engagement": 0.5,
                "valence": 0.0,
                "reflexes_fired": 1.0,
                "reflexes_succeeded": 1.0,
            })
            assert result.error is None, f"Program '{name}' failed: {result.error}"
            assert result.halted, f"Program '{name}' didn't halt"


# ── Test: VM Performance ──────────────────────────────────────────

class TestVMPerformance:
    def test_scoring_under_5ms(self):
        """R5 scoring should complete in under 5ms (fast enough for every interaction)."""
        from titan_plugin.logic.vm_programs import get_program

        reg = MockStateRegister()
        vm = TitanVM(state_register=reg)
        context = {
            "intensity": 0.5,
            "engagement": 0.5,
            "valence": 0.0,
            "reflexes_fired": 2.0,
            "reflexes_succeeded": 2.0,
        }

        # Warm up
        vm.execute(get_program("reflex_score"), context=context)

        # Measure
        start = time.monotonic()
        for _ in range(100):
            vm.execute(get_program("reflex_score"), context=context)
        elapsed_ms = (time.monotonic() - start) * 1000

        avg_ms = elapsed_ms / 100
        assert avg_ms < 5.0, f"Average VM execution: {avg_ms:.2f}ms (should be < 5ms)"
