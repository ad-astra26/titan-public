"""
TitanVM — Stack-based micro-instruction set for sovereign internal computation.

A deterministic, inspectable computation substrate that operates directly on
StateRegister data. Like the autonomic nervous system — no LLM needed, pure
math on raw state values.

Architecture:
  - Stack machine (Forth/JVM-like), configurable stack depth
  - ~25 opcodes: stack, math, compare, flow, state, bus, temporal (v2), smooth (v2), special
  - Data source: StateRegister (read via LOAD), DivineBus (emit via EMIT)
  - Execution boundary: only Spirit and Interface should run programs

Programs are lists of (opcode, *args) tuples. Text parser deferred to R6.

v2 (2026-04-21, rFP_titan_vm_v2 Phase 1):
  - LOAD_EMA <path> <alpha> — push EMA of path value (per-program state)
  - LOAD_DT  <path>         — push delta since last call at path (per-program state)
  - SIGMOID  <k>            — pop x, push 1/(1+exp(-k·x)) — smooth squash
  - SOFT_GT  <th> <k>       — pop x, push sigmoid(k·(x−th))  — smooth "greater than"
  - SOFT_LT  <th> <k>       — pop x, push sigmoid(k·(th−x))  — smooth "less than"
  - Per-program runtime state persisted at data/neural_nervous_system/titan_vm_runtime.json
  - execute() accepts program_key for EMA/DT state attribution

Usage:
    from titan_plugin.logic.titan_vm import TitanVM, Op

    vm = TitanVM(state_register=register)
    program = [
        (Op.LOAD, "body_tensor.0"),
        (Op.PUSH, 0.5),
        (Op.CMP_GT,),
        (Op.BRANCH_IF, "healthy"),
        (Op.PUSH, 0.0),
        (Op.JMP, "done"),
        ("healthy",),
        (Op.PUSH, 0.1),
        ("done",),
        (Op.SCORE,),
        (Op.HALT,),
    ]
    result = vm.execute(program, program_key="REFLEX")
"""
import json
import logging
import math
import os
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

    # v2 — Temporal observables (L5-friendly priors)
    LOAD_EMA = auto()   # LOAD_EMA <path> <alpha> — push EMA(path) with α; per-program
    LOAD_DT = auto()    # LOAD_DT <path>          — push (current - previous) at path; per-program

    # v2 — Smooth gates (differentiable priors — AIF composition-ready)
    SIGMOID = auto()    # SIGMOID <k>            — pop x, push 1/(1+exp(-k·x))
    SOFT_GT = auto()    # SOFT_GT <threshold> <k> — pop x, push sigmoid(k·(x-threshold))
    SOFT_LT = auto()    # SOFT_LT <threshold> <k> — pop x, push sigmoid(k·(threshold-x))


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


class VMRuntimeState:
    """Per-program, per-path runtime state for LOAD_EMA + LOAD_DT opcodes.

    Structure:
      ema_state[program_key][path] = last_ema_value
      prev_values[program_key][path] = last_observed_value

    Persistence: atomic tmp+rename to titan_vm_runtime.json. Tolerant loader
    (corrupt file → WARNING + fresh defaults, no crash). Schema v1.

    EMA recurrence: new_ema = (1 - alpha) · old_ema + alpha · current. First
    call at a (program, path) bootstraps to the current value (no prior state).
    """

    SCHEMA_VERSION = 1

    def __init__(self, data_dir: str = "data/neural_nervous_system", load: bool = True):
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "titan_vm_runtime.json")
        self.ema_state: dict[str, dict[str, float]] = {}
        self.prev_values: dict[str, dict[str, float]] = {}
        self.total_executions: int = 0
        self.last_saved_ts: float = 0.0
        if load:
            self._try_load()

    def get_ema(self, program_key: Optional[str], path: str, current: float,
                alpha: float) -> float:
        """Return EMA of path under given program. First call bootstraps to current."""
        if not program_key:
            return current  # no tracking without program attribution
        prog = self.ema_state.setdefault(program_key, {})
        prev = prog.get(path)
        if prev is None:
            prog[path] = current
            return current
        updated = (1.0 - alpha) * prev + alpha * current
        prog[path] = updated
        return updated

    def get_dt(self, program_key: Optional[str], path: str, current: float) -> float:
        """Return (current − previous) at path. First call returns 0.0."""
        if not program_key:
            return 0.0
        prog = self.prev_values.setdefault(program_key, {})
        prev = prog.get(path)
        prog[path] = current
        if prev is None:
            return 0.0
        return current - prev

    def note_execution(self) -> None:
        self.total_executions += 1

    def to_dict(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "ema_state": self.ema_state,
            "prev_values": self.prev_values,
            "total_executions": self.total_executions,
            "last_saved_ts": time.time(),
        }

    def save(self) -> None:
        """Atomic tmp+rename save with pre-flight json encode guard."""
        try:
            payload = json.dumps(self.to_dict())
            os.makedirs(self._data_dir, exist_ok=True)
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(payload)
            os.replace(tmp, self._state_path)
            self.last_saved_ts = time.time()
        except Exception as e:
            logger.warning("[TitanVM runtime state] Save failed (non-fatal): %s", e)

    def _try_load(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("schema") != self.SCHEMA_VERSION:
                logger.warning("[TitanVM runtime state] Schema mismatch; initializing fresh")
                return
            ema = data.get("ema_state", {})
            prev = data.get("prev_values", {})
            if isinstance(ema, dict):
                self.ema_state = {
                    str(k): {str(p): float(v) for p, v in d.items()
                             if isinstance(v, (int, float))}
                    for k, d in ema.items() if isinstance(d, dict)
                }
            if isinstance(prev, dict):
                self.prev_values = {
                    str(k): {str(p): float(v) for p, v in d.items()
                             if isinstance(v, (int, float))}
                    for k, d in prev.items() if isinstance(d, dict)
                }
            self.total_executions = int(data.get("total_executions", 0))
            self.last_saved_ts = float(data.get("last_saved_ts", 0.0))
            logger.info(
                "[TitanVM runtime state] Loaded: %d programs tracked, %d total executions",
                len(self.ema_state), self.total_executions,
            )
        except Exception as e:
            logger.warning("[TitanVM runtime state] Load failed, using defaults: %s", e)


class TitanVM:
    """
    Stack-based virtual machine for sovereign internal computation.

    Operates on StateRegister data, produces reward scores and bus emissions.
    Deterministic, inspectable, fast.
    """

    def __init__(self, state_register=None, bus=None, config: dict | None = None,
                 runtime_state: Optional[VMRuntimeState] = None,
                 runtime_data_dir: str = "data/neural_nervous_system"):
        self._state_register = state_register
        self._bus = bus
        # [titan_vm] toml plumbing — 2026-04-16. Previously the section was
        # defined in titan_params.toml but every call site constructed
        # TitanVM() with no config, so max_stack_depth/max_instructions were
        # forever pinned to the module-level MAX_STACK_DEPTH/MAX_INSTRUCTIONS
        # constants. Constants remain as defaults to preserve behavior when
        # config is absent.
        cfg = config or {}
        self._max_stack_depth = int(cfg.get("max_stack_depth", MAX_STACK_DEPTH))
        self._max_instructions = int(cfg.get("max_instructions", MAX_INSTRUCTIONS))
        # Downstream reward plumbing (agno_hooks REFLEX_REWARD publisher)
        self._publish_rewards = bool(cfg.get("publish_rewards", True))
        self._min_reward_threshold = float(cfg.get("min_reward_threshold", 0.01))
        self._reward_blend_weight = float(cfg.get("reward_blend_weight", 0.3))
        # v2: runtime-state save cadence (evaluations between periodic saves)
        self._runtime_save_every = int(cfg.get("runtime_save_every", 100))

        # Per-program runtime state for LOAD_EMA + LOAD_DT (shared across all execute calls).
        self._runtime = runtime_state if runtime_state is not None else \
            VMRuntimeState(data_dir=runtime_data_dir, load=True)

        # Per-program fire telemetry (for /v4/titan-vm)
        # program_key → {"fire_count": int, "last_score": float, "last_ts": float, "score_sum": float, "scored_count": int}
        self._telemetry: dict[str, dict[str, float]] = {}

    def execute(self, program: list, context: dict = None,
                program_key: Optional[str] = None) -> VMResult:
        """
        Execute a TitanVM program.

        Args:
            program: List of instruction tuples. Each is (Op, *args) or
                     (str,) for labels.
            context: Optional dict of extra values accessible via LOAD context.<key>
                     Also, values saved with STORE can be re-read with LOAD.
            program_key: Program identifier (e.g. "REFLEX", "METABOLISM") for
                         LOAD_EMA / LOAD_DT state attribution. Pass None for
                         ad-hoc evaluation without temporal tracking.

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
                labels[item[0]] = len(instructions)
            else:
                instructions.append(item)

        pc = 0
        count = 0

        max_stack = self._max_stack_depth
        max_instr = self._max_instructions
        try:
            while pc < len(instructions) and count < max_instr:
                instr = instructions[pc]
                op = instr[0]
                count += 1

                if op == Op.PUSH:
                    val = float(instr[1])
                    if len(stack) >= max_stack:
                        result.error = "stack overflow"
                        break
                    stack.append(val)

                elif op == Op.POP:
                    if stack:
                        stack.pop()

                elif op == Op.DUP:
                    if stack:
                        if len(stack) >= max_stack:
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
                    if path in registers:
                        val = registers[path]
                    else:
                        val = self._load_value(path, context)
                    if len(stack) >= max_stack:
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

                # ── v2: temporal observables ──

                elif op == Op.LOAD_EMA:
                    path = instr[1]
                    alpha = float(instr[2]) if len(instr) > 2 else 0.1
                    current = self._load_value(path, context)
                    val = self._runtime.get_ema(program_key, path, current, alpha)
                    if len(stack) >= max_stack:
                        result.error = "stack overflow"
                        break
                    stack.append(val)

                elif op == Op.LOAD_DT:
                    path = instr[1]
                    current = self._load_value(path, context)
                    val = self._runtime.get_dt(program_key, path, current)
                    if len(stack) >= max_stack:
                        result.error = "stack overflow"
                        break
                    stack.append(val)

                # ── v2: smooth gates ──

                elif op == Op.SIGMOID:
                    k = float(instr[1]) if len(instr) > 1 else 1.0
                    if stack:
                        x = stack.pop()
                        stack.append(_sigmoid(k * x))

                elif op == Op.SOFT_GT:
                    threshold = float(instr[1]) if len(instr) > 1 else 0.0
                    k = float(instr[2]) if len(instr) > 2 else 10.0
                    if stack:
                        x = stack.pop()
                        stack.append(_sigmoid(k * (x - threshold)))

                elif op == Op.SOFT_LT:
                    threshold = float(instr[1]) if len(instr) > 1 else 0.0
                    k = float(instr[2]) if len(instr) > 2 else 10.0
                    if stack:
                        x = stack.pop()
                        stack.append(_sigmoid(k * (threshold - x)))

                else:
                    result.error = f"unknown opcode: {op}"
                    break

                pc += 1

        except Exception as e:
            result.error = f"execution error: {e}"
            logger.warning("[TitanVM] Execution error at pc=%d: %s", pc, e)

        if count >= max_instr:
            result.error = "instruction limit exceeded"

        result.instructions_executed = count
        result.registers = registers
        result.duration_ms = (time.monotonic() - start) * 1000

        # Per-program telemetry + periodic runtime-state save
        if program_key:
            tel = self._telemetry.setdefault(program_key, {
                "fire_count": 0.0, "last_score": 0.0, "last_ts": 0.0,
                "score_sum": 0.0, "scored_count": 0.0,
            })
            tel["last_score"] = result.score
            tel["last_ts"] = time.time()
            tel["scored_count"] += 1
            tel["score_sum"] += result.score
            if result.score > self._min_reward_threshold:
                tel["fire_count"] += 1

        self._runtime.note_execution()
        if (self._runtime.total_executions > 0
                and self._runtime.total_executions % self._runtime_save_every == 0):
            self._runtime.save()

        return result

    def get_telemetry(self) -> dict:
        """Per-program execution telemetry for /v4/titan-vm diagnostic endpoint."""
        out = {}
        for key, t in self._telemetry.items():
            n = max(1.0, t.get("scored_count", 0.0))
            out[key] = {
                "fire_count": int(t.get("fire_count", 0)),
                "last_score": round(t.get("last_score", 0.0), 4),
                "last_ts": t.get("last_ts", 0.0),
                "scored_count": int(t.get("scored_count", 0)),
                "avg_score": round(t.get("score_sum", 0.0) / n, 4),
            }
        return out

    def get_runtime_state(self) -> VMRuntimeState:
        """Access runtime-state object (for tests + persistence tooling)."""
        return self._runtime

    def get_reward_blend_weight(self) -> float:
        """Expose reward blend weight for agno_hooks reward aggregation (v2 consumer)."""
        return self._reward_blend_weight

    def get_min_reward_threshold(self) -> float:
        """Expose min reward threshold for VM fire-gate (v2 consumer)."""
        return self._min_reward_threshold

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

        if root == "context":
            key = ".".join(parts[1:]) if len(parts) > 1 else ""
            val = context.get(key, 0.0)
            return float(val) if isinstance(val, (int, float)) else 0.0

        if path in context:
            val = context[path]
            return float(val) if isinstance(val, (int, float)) else 0.0

        if not self._state_register:
            return 0.0

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
                return sum(tensor) / len(tensor) if tensor else 0.0
            return 0.0

        dict_keys = {"consciousness", "metabolic", "resonance", "unified_spirit", "sphere_clocks"}
        if root in dict_keys:
            d = getattr(self._state_register, root, None)
            if d is None:
                d = self._state_register.get(root, {})
            if isinstance(d, dict) and len(parts) > 1:
                val = d.get(parts[1], 0.0)
                return float(val) if isinstance(val, (int, float)) else 0.0
            return 0.0

        scalar_keys = {"last_impulse_ts", "body_ts", "mind_ts", "spirit_ts", "last_update_ts"}
        if root in scalar_keys:
            val = self._state_register.get(root, 0.0)
            return float(val) if isinstance(val, (int, float)) else 0.0

        return 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow for extreme x
    if x >= 0:
        ex = math.exp(-min(x, 500.0))
        return 1.0 / (1.0 + ex)
    else:
        ex = math.exp(max(x, -500.0))
        return ex / (1.0 + ex)
