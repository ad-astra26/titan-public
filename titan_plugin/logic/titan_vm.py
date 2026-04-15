"""
TitanVM — Stack-based micro-instruction set for sovereign internal computation.

A deterministic, inspectable computation substrate that operates directly on
StateRegister data. Like the autonomic nervous system — no LLM needed, pure
math on raw state values.

Architecture:
  - Stack machine (Forth/JVM-like), configurable stack depth
  - ~20 opcodes: stack, math, compare, flow, state, bus, special
  - Data source: StateRegister (read via LOAD), DivineBus (emit via EMIT)
  - Execution boundary: only Spirit and Interface should run programs

Programs are lists of (opcode, *args) tuples. Text parser deferred to R6.

Usage:
    from titan_plugin.logic.titan_vm import TitanVM, Op

    vm = TitanVM(state_register=register)
    program = [
        (Op.LOAD, "body_tensor.0"),   # push body[0] (interoception)
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "healthy"),
        (Op.PUSH, 0.0),
        (Op.JMP, "done"),
        ("healthy",),                  # label
        (Op.PUSH, 0.1),
        ("done",),                     # label
        (Op.SCORE,),
        (Op.HALT,),
    ]
    result = vm.execute(program)
    # result.score = 0.1 or 0.0, result.emissions = [...]
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_STACK_DEPTH = 64
MAX_INSTRUCTIONS = 500  # prevent infinite loops


class Op(Enum):
    """TitanVM opcodes."""
    # Stack manipulation
    PUSH = auto()       # PUSH <value> — push literal onto stack
    POP = auto()        # POP — discard top
    DUP = auto()        # DUP — duplicate top
    SWAP = auto()       # SWAP — swap top two
    ROT = auto()        # ROT — rotate top three (a b c → b c a)

    # Math
    ADD = auto()        # ADD — pop two, push sum
    SUB = auto()        # SUB — pop two, push (second - top)
    MUL = auto()        # MUL — pop two, push product
    DIV = auto()        # DIV — pop two, push (second / top), safe div-by-zero
    ABS = auto()        # ABS — pop one, push absolute value
    CLAMP = auto()      # CLAMP — pop three: value, min, max → push clamped

    # Compare (push 1.0 for true, 0.0 for false)
    CMP_GT = auto()     # CMP_GT — pop two, push 1.0 if second > top
    CMP_LT = auto()     # CMP_LT — pop two, push 1.0 if second < top
    CMP_EQ = auto()     # CMP_EQ — pop two, push 1.0 if |second - top| < 0.001

    # Flow control
    JMP = auto()        # JMP <label> — unconditional jump
    BRANCH_IF = auto()  # BRANCH_IF <label> — jump if top > 0.0 (pops top)
    HALT = auto()       # HALT — stop execution

    # State access
    LOAD = auto()       # LOAD <path> — read from StateRegister, push value
    STORE = auto()      # STORE <name> — pop top, store in named register

    # Bus interaction
    EMIT = auto()       # EMIT <msg_type> — pop top as payload value, emit to bus

    # Special
    SCORE = auto()      # SCORE — pop top → set as program's reward score
    CLOCK = auto()      # CLOCK — push current time (monotonic seconds)
    AGE = auto()        # AGE — push StateRegister age in seconds


@dataclass
class VMResult:
    """Result of a TitanVM program execution."""
    score: float = 0.0                        # from SCORE opcode
    emissions: list[dict] = field(default_factory=list)  # from EMIT opcode
    registers: dict[str, float] = field(default_factory=dict)  # from STORE
    instructions_executed: int = 0
    duration_ms: float = 0.0
    halted: bool = False
    error: Optional[str] = None


class TitanVM:
    """
    Stack-based virtual machine for sovereign internal computation.

    Operates on StateRegister data, produces reward scores and bus emissions.
    Deterministic, inspectable, fast.
    """

    def __init__(self, state_register=None, bus=None):
        self._state_register = state_register
        self._bus = bus

    def execute(self, program: list, context: dict = None) -> VMResult:
        """
        Execute a TitanVM program.

        Args:
            program: List of instruction tuples. Each is (Op, *args) or
                     (str,) for labels.
            context: Optional dict of extra values accessible via LOAD context.<key>
                     Also, values saved with STORE can be re-read with LOAD.

        Returns:
            VMResult with score, emissions, registers, and execution stats.
        """
        start = time.monotonic()
        result = VMResult()
        stack: list[float] = []
        registers: dict[str, float] = {}
        context = context or {}

        # Pre-process: resolve label positions
        labels: dict[str, int] = {}
        instructions: list[tuple] = []
        for item in program:
            if isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], str):
                # This is a label
                labels[item[0]] = len(instructions)
            else:
                instructions.append(item)

        pc = 0  # program counter
        count = 0

        try:
            while pc < len(instructions) and count < MAX_INSTRUCTIONS:
                instr = instructions[pc]
                op = instr[0]
                count += 1

                if op == Op.PUSH:
                    val = float(instr[1])
                    if len(stack) >= MAX_STACK_DEPTH:
                        result.error = "stack overflow"
                        break
                    stack.append(val)

                elif op == Op.POP:
                    if stack:
                        stack.pop()

                elif op == Op.DUP:
                    if stack:
                        if len(stack) >= MAX_STACK_DEPTH:
                            result.error = "stack overflow"
                            break
                        stack.append(stack[-1])

                elif op == Op.SWAP:
                    if len(stack) >= 2:
                        stack[-1], stack[-2] = stack[-2], stack[-1]

                elif op == Op.ROT:
                    if len(stack) >= 3:
                        a, b, c = stack[-3], stack[-2], stack[-1]
                        stack[-3], stack[-2], stack[-1] = b, c, a

                elif op == Op.ADD:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(a + b)

                elif op == Op.SUB:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(a - b)

                elif op == Op.MUL:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(a * b)

                elif op == Op.DIV:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(a / b if abs(b) > 1e-10 else 0.0)

                elif op == Op.ABS:
                    if stack:
                        stack[-1] = abs(stack[-1])

                elif op == Op.CLAMP:
                    if len(stack) >= 3:
                        hi = stack.pop()
                        lo = stack.pop()
                        val = stack.pop()
                        stack.append(max(lo, min(hi, val)))

                elif op == Op.CMP_GT:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(1.0 if a > b else 0.0)

                elif op == Op.CMP_LT:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(1.0 if a < b else 0.0)

                elif op == Op.CMP_EQ:
                    if len(stack) >= 2:
                        b, a = stack.pop(), stack.pop()
                        stack.append(1.0 if abs(a - b) < 0.001 else 0.0)

                elif op == Op.JMP:
                    label = instr[1]
                    if label in labels:
                        pc = labels[label]
                        continue
                    else:
                        result.error = f"unknown label: {label}"
                        break

                elif op == Op.BRANCH_IF:
                    label = instr[1]
                    if stack:
                        cond = stack.pop()
                        if cond > 0.0:
                            if label in labels:
                                pc = labels[label]
                                continue
                            else:
                                result.error = f"unknown label: {label}"
                                break

                elif op == Op.HALT:
                    result.halted = True
                    break

                elif op == Op.LOAD:
                    path = instr[1]
                    # Check named registers first (from STORE)
                    if path in registers:
                        val = registers[path]
                    else:
                        val = self._load_value(path, context)
                    if len(stack) >= MAX_STACK_DEPTH:
                        result.error = "stack overflow"
                        break
                    stack.append(val)

                elif op == Op.STORE:
                    name = instr[1]
                    if stack:
                        registers[name] = stack.pop()

                elif op == Op.EMIT:
                    msg_type = instr[1]
                    val = stack.pop() if stack else 0.0
                    emission = {"type": msg_type, "value": val, "registers": dict(registers)}
                    result.emissions.append(emission)
                    # Actually publish to bus if available
                    if self._bus:
                        try:
                            from titan_plugin.bus import make_msg
                            msg = make_msg(msg_type, "titan_vm", "all", {
                                "value": val,
                                "registers": dict(registers),
                            })
                            self._bus.publish(msg)
                        except Exception as e:
                            logger.debug("[TitanVM] Bus emit failed: %s", e)

                elif op == Op.SCORE:
                    if stack:
                        result.score = stack.pop()

                elif op == Op.CLOCK:
                    stack.append(time.monotonic())

                elif op == Op.AGE:
                    if self._state_register:
                        stack.append(self._state_register.age_seconds())
                    else:
                        stack.append(0.0)

                else:
                    result.error = f"unknown opcode: {op}"
                    break

                pc += 1

        except Exception as e:
            result.error = f"execution error: {e}"
            logger.warning("[TitanVM] Execution error at pc=%d: %s", pc, e)

        if count >= MAX_INSTRUCTIONS:
            result.error = "instruction limit exceeded"

        result.instructions_executed = count
        result.registers = registers
        result.duration_ms = (time.monotonic() - start) * 1000

        return result

    def _load_value(self, path: str, context: dict) -> float:
        """
        Resolve a dotted path to a float value.

        Paths:
          - "body_tensor.0" → state_register.body_tensor[0]
          - "mind_tensor.2" → state_register.mind_tensor[2]
          - "consciousness.drift" → state_register.consciousness["drift"]
          - "focus_body.1" → state_register.focus_body[1]
          - "metabolic.sol_balance" → state_register.metabolic["sol_balance"]
          - "context.intensity" → context["intensity"]
        """
        parts = path.split(".")
        if not parts:
            return 0.0

        root = parts[0]

        # Context values — "context.key" prefix OR direct context key match
        if root == "context":
            key = ".".join(parts[1:]) if len(parts) > 1 else ""
            val = context.get(key, 0.0)
            return float(val) if isinstance(val, (int, float)) else 0.0

        # Direct context lookup: paths like "all.velocity_avg" or "inner_body.coherence"
        # check full dotted path in context before falling through to StateRegister
        if path in context:
            val = context[path]
            return float(val) if isinstance(val, (int, float)) else 0.0

        # StateRegister values
        if not self._state_register:
            return 0.0

        # Tensor properties (list[float])
        tensor_keys = {
            "body_tensor", "mind_tensor", "spirit_tensor",
            "focus_body", "focus_mind",
            "filter_down_body", "filter_down_mind",
        }
        if root in tensor_keys:
            tensor = getattr(self._state_register, root, None)
            if tensor is None:
                tensor = self._state_register.get(root, [])
            if isinstance(tensor, list) and len(parts) > 1:
                try:
                    idx = int(parts[1])
                    return float(tensor[idx]) if 0 <= idx < len(tensor) else 0.0
                except (ValueError, IndexError):
                    return 0.0
            elif isinstance(tensor, list):
                # No index → return average
                return sum(tensor) / len(tensor) if tensor else 0.0
            return 0.0

        # Dict properties
        dict_keys = {"consciousness", "metabolic", "resonance", "unified_spirit", "sphere_clocks"}
        if root in dict_keys:
            d = getattr(self._state_register, root, None)
            if d is None:
                d = self._state_register.get(root, {})
            if isinstance(d, dict) and len(parts) > 1:
                val = d.get(parts[1], 0.0)
                return float(val) if isinstance(val, (int, float)) else 0.0
            return 0.0

        # Scalar properties
        scalar_keys = {"last_impulse_ts", "body_ts", "mind_ts", "spirit_ts", "last_update_ts"}
        if root in scalar_keys:
            val = self._state_register.get(root, 0.0)
            return float(val) if isinstance(val, (int, float)) else 0.0

        return 0.0
