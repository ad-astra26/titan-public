#!/usr/bin/env python3
"""
Titan Architecture Mapper — AST-based dependency and wiring analysis.

Scans titan_plugin/**/*.py and builds a queryable dependency graph that tracks:
- Module imports (who imports whom)
- Class/function definitions with line numbers
- Function calls (cross-module where detectable)
- DivineBus wiring: publish, subscribe, _send_msg patterns
- Attribute access on key state objects
- getattr() patterns for runtime attribute access

Usage:
  python scripts/arch_map.py scan                  # Regenerate architecture_map.json
  python scripts/arch_map.py query bus:EPOCH_TICK   # Bus message wiring
  python scripts/arch_map.py query calls:func_name  # Who calls this function
  python scripts/arch_map.py query file:dreaming.py # File summary
  python scripts/arch_map.py query attr:inner_state.fatigue  # Attribute access
  python scripts/arch_map.py query depends:file.py  # What depends on this file
  python scripts/arch_map.py query class:ClassName  # Class info and methods
  python scripts/arch_map.py wiring                 # Full bus wiring overview
  python scripts/arch_map.py summary                # High-level stats
  python scripts/arch_map.py audit                  # Signal integrity audit (dead signals, deaf ears, scope leaks)
  python scripts/arch_map.py worker spirit           # Per-module wiring card (bus sends/consumes/imports)
  python scripts/arch_map.py flow MEDITATION_REQUEST # End-to-end message journey trace
  python scripts/arch_map.py params                  # Unused titan_params.toml key audit
  python scripts/arch_map.py health                  # LIVE runtime health checks against API
  python scripts/arch_map.py health --all            # Check ALL 3 Titans (T1+T2+T3)
  python scripts/arch_map.py health --t2             # Also check T2 (10.135.0.6:7777)
  python scripts/arch_map.py services                # Teacher + ARC + Persona diagnostics (all Titans)
  python scripts/arch_map.py services --json         # Same, JSON output (cron-friendly)
  python scripts/arch_map.py audit --live            # Static audit + live wiring contract verification
  python scripts/arch_map.py cgn                     # CGN grounding telemetry (T1 only)
  python scripts/arch_map.py cgn --all               # CGN grounding telemetry (all 3 Titans)
  python scripts/arch_map.py cgn-reasoning-strategies # Top grounded reasoning_strategy chain patterns (T1 only)
  python scripts/arch_map.py cgn-reasoning-strategies --all # Same, all 3 Titans
  python scripts/arch_map.py dead-wiring             # Static analysis: orphan methods, pair-closure gaps, unused config sections, rFP verification
  python scripts/arch_map.py dead-wiring --rfp PATH  # Verify class/method names from an rFP exist in the codebase
  python scripts/arch_map.py dead-wiring --json      # Machine-readable output
  python scripts/arch_map.py emot-cgn                # EMOT-CGN status + v3 emergence + graduation (all 3 Titans)
  python scripts/arch_map.py emot-cgn --t1           # EMOT-CGN — single Titan (--t1/--t2/--t3)
  python scripts/arch_map.py emot-cgn --json         # EMOT-CGN JSON output (cron-friendly)
  python scripts/arch_map.py timechain               # TimeChain diagnostics (T1 only)
  python scripts/arch_map.py timechain --all          # TimeChain diagnostics (all 3 Titans)
  python scripts/arch_map.py sensors                  # V6 5DT outer_body rich-signal diagnostic (all 3 Titans, Phase 1 sensory wiring)
  python scripts/arch_map.py report                  # Full comparison report across all 3 Titans
  python scripts/arch_map.py deploy t2 --restart     # Deploy code to T2 + restart + verify
  python scripts/arch_map.py deploy all --restart    # Deploy + restart all remote Titans
  python scripts/arch_map.py errors                  # Cross-Titan error summary (last 500 lines)
  python scripts/arch_map.py errors --all            # Include T2/T3 errors via SSH
  python scripts/arch_map.py traffic                 # Nginx traffic stats (last 1 hour)
  python scripts/arch_map.py traffic --24h           # Nginx traffic stats (last 24 hours)
  python scripts/arch_map.py traffic --1h            # Nginx traffic stats (last 1 hour, default)
  python scripts/arch_map.py session-close "title"   # Parse JSONL → conversation.md + session.md + commit
  python scripts/arch_map.py kernel-status           # Microkernel v2 S3 kernel health (T1 only)
  python scripts/arch_map.py kernel-status --all     # Microkernel v2 S3 kernel health (T1+T2+T3)
  python scripts/arch_map.py titanvm-schema          # Microkernel v2 S4 — verify NS program telemetry shape (TITANVM_REGISTERS spec input)
  python scripts/arch_map.py api-status              # Microkernel v2 S5 — API subprocess health (T1 only, local)
  python scripts/arch_map.py api-status --all        # Microkernel v2 S5 — API subprocess health (T1+T2+T3)
  python scripts/arch_map.py module-methods          # Microkernel v2 S6 — per-module spawn/fork audit (T1 only)
  python scripts/arch_map.py module-methods --all    # Microkernel v2 S6 — per-module spawn/fork audit (T1+T2+T3)
  python scripts/arch_map.py s5-callsites            # Microkernel v2 S5 amendment D4 — endpoint plugin.X drift detection
"""

import os
import ast
import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = [PROJECT_ROOT / "titan_plugin"]
SCRIPTS_DIRS = [PROJECT_ROOT / "scripts"]
OUTPUT_FILE = PROJECT_ROOT / "architecture_map.json"
BUS_CONSTANTS_FILE = PROJECT_ROOT / "titan_plugin" / "bus.py"
REL_BASE = PROJECT_ROOT

# Force-prefer the project root next to this script over the
# editable-install (`__editable__.openclaw_plugin_titan-*.pth`) finder
# which hardcodes `titan_plugin` → main-repo path. Without this,
# arch_map subcommands that import `titan_plugin.core.X` from a session
# worktree silently get the main-repo's stale module instead of the
# worktree's edited code (e.g. fresh RegistrySpec adds invisible).
# Inserting at 0 puts us ahead of MetaPathFinder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def rel(path: Path) -> str:
    """Convert absolute path to project-relative string."""
    try:
        return str(path.relative_to(REL_BASE))
    except ValueError:
        return str(path)


def load_bus_constants() -> dict:
    """Parse bus.py to build a map of constant_name -> string_value.
    e.g., {"EPOCH_TICK": "EPOCH_TICK", "BODY_STATE": "BODY_STATE"}
    """
    constants = {}
    if not BUS_CONSTANTS_FILE.exists():
        return constants
    try:
        tree = ast.parse(BUS_CONSTANTS_FILE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str) and target.id.isupper():
                        constants[target.id] = node.value.value
    except Exception:
        pass
    return constants


# ── AST Extraction ────────────────────────────────────────────────────

class TitanASTVisitor(ast.NodeVisitor):
    """Walk a single Python file and extract architectural information."""

    def __init__(self, filepath: str, bus_constants: dict = None):
        self.filepath = filepath
        self.bus_constants = bus_constants or {}
        self.classes = {}
        self.functions = {}
        self.imports = []           # [(module, [names])]
        self.from_imports = []      # [(module, [names])]
        self.bus_publishes = []     # [{"line", "msg_type", "context"}]
        self.bus_subscribes = []    # [{"line", "name", "reply_only", "context"}]
        self.send_msgs = []         # [{"line", "msg_type", "dst", "context"}]
        self.msg_consumers = []     # [{"line", "msg_type", "context"}]  -- msg_type == "X" checks
        self.function_calls = []    # [{"line", "func", "context"}]
        self.attr_access = []       # [{"line", "obj", "attr", "context", "mode"}]
        self.getattr_calls = []     # [{"line", "obj", "attr", "context"}]
        self._context_stack = []    # Track current class/function context

    @property
    def context(self) -> str:
        return ".".join(self._context_stack) if self._context_stack else "<module>"

    def visit_ClassDef(self, node):
        self._context_stack.append(node.name)
        methods = {}
        for item in ast.iter_child_nodes(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in item.args.args]
                methods[item.name] = {
                    "line": item.lineno,
                    "args": args,
                    "decorators": [self._decorator_name(d) for d in item.decorator_list],
                }
        bases = [self._name_of(b) for b in node.bases]
        self.classes[node.name] = {
            "line": node.lineno,
            "bases": bases,
            "methods": methods,
        }
        self.generic_visit(node)
        self._context_stack.pop()

    def visit_FunctionDef(self, node):
        if not self._context_stack or not any(
            isinstance(p, ast.ClassDef) for p in ast.walk(ast.parse(""))
        ):
            # Only record top-level functions (not methods — those go in classes)
            if len(self._context_stack) == 0:
                args = [a.arg for a in node.args.args]
                self.functions[node.name] = {
                    "line": node.lineno,
                    "args": args,
                    "decorators": [self._decorator_name(d) for d in node.decorator_list],
                }
        self._context_stack.append(node.name)
        self.generic_visit(node)
        self._context_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name, alias.asname))

    def visit_ImportFrom(self, node):
        if node.module:
            names = [alias.name for alias in node.names]
            self.from_imports.append((node.module, names))

    def visit_Call(self, node):
        func_name = self._call_name(node)
        if func_name:
            # ── _send_msg detection ──
            if func_name == "_send_msg" and len(node.args) >= 4:
                msg_type = self._str_value(node.args[1])
                dst = self._str_value(node.args[3])
                if msg_type:
                    self.send_msgs.append({
                        "line": node.lineno,
                        "msg_type": msg_type,
                        "dst": dst or "?",
                        "context": self.context,
                    })

            # ── make_msg detection (resolves bus constants) ──
            # Supports BOTH positional and keyword-argument calling conventions:
            #   make_msg(MSG_TYPE, src, dst, payload=...)
            #   make_msg(msg_type="MSG_TYPE", src=..., dst=..., payload=...)
            # The keyword form is used e.g. by maker/narrative_channel.py.
            elif func_name == "make_msg":
                msg_type = None
                src = "?"
                dst = "?"
                if len(node.args) >= 1:
                    msg_type = self._resolve_value(node.args[0])
                    if len(node.args) > 1:
                        src = self._resolve_value(node.args[1])
                    if len(node.args) > 2:
                        dst = self._resolve_value(node.args[2])
                # Fill in from keywords — these override missing positionals.
                for kw in node.keywords:
                    if kw.arg == "msg_type" and msg_type is None:
                        msg_type = self._resolve_value(kw.value)
                    elif kw.arg == "src":
                        src = self._resolve_value(kw.value)
                    elif kw.arg == "dst":
                        dst = self._resolve_value(kw.value)
                if msg_type:
                    # make_msg is always wrapped in bus.publish — record as publish
                    self.bus_publishes.append({
                        "line": node.lineno,
                        "msg_type": msg_type,
                        "dst": dst,
                        "src": src,
                        "context": self.context,
                    })

            # ── Raw send_queue.put({"type": "MSG_TYPE", ...}) pattern ──
            # Used by spirit_worker for OBSERVABLES_SNAPSHOT and similar internal
            # messages. Narrow match — only trigger on queue-named .put() calls
            # to avoid collisions with list.put / dict-style helpers that happen
            # to accept a dict argument.
            elif (func_name.endswith(".put")
                  and any(q in func_name for q in
                          ("send_queue", "recv_queue", "queue.put", "_queue"))
                  and len(node.args) >= 1 and isinstance(node.args[0], ast.Dict)):
                d = node.args[0]
                msg_type = None
                dst = "?"
                src = "?"
                for k, v in zip(d.keys, d.values):
                    if isinstance(k, ast.Constant):
                        if k.value == "type":
                            msg_type = self._resolve_value(v)
                        elif k.value == "dst":
                            dst = self._resolve_value(v) or "?"
                        elif k.value == "src":
                            src = self._resolve_value(v) or "?"
                if msg_type:
                    self.send_msgs.append({
                        "line": node.lineno,
                        "msg_type": msg_type,
                        "dst": dst,
                        "context": self.context,
                    })

            # ── bus.publish detection ──
            elif func_name in ("bus.publish", "self._bus.publish", "self.bus.publish"):
                # Check if arg is make_msg (already captured above) or dict literal
                msg_type = self._extract_bus_publish_type(node)
                if msg_type:
                    self.bus_publishes.append({
                        "line": node.lineno,
                        "msg_type": msg_type,
                        "context": self.context,
                    })

            # ── bus.subscribe detection ──
            elif func_name in ("bus.subscribe", "self._bus.subscribe", "self.bus.subscribe"):
                sub_name = self._str_value(node.args[0]) if node.args else "?"
                reply_only = False
                for kw in node.keywords:
                    if kw.arg == "reply_only":
                        reply_only = self._bool_value(kw.value)
                self.bus_subscribes.append({
                    "line": node.lineno,
                    "name": sub_name,
                    "reply_only": reply_only,
                    "context": self.context,
                })

            # ── Guardian module registration as bus subscriber ──
            # ModuleSpec(name="body", ...) → Guardian.start() subscribes "body" to the bus
            # via self.bus._subscribers.setdefault(name, []).append(info.queue).
            # Detect ModuleSpec() calls and register names as implicit bus subscribers.
            elif func_name == "ModuleSpec":
                for kw in node.keywords:
                    if kw.arg == "name":
                        mod_name = self._str_value(kw.value)
                        if mod_name:
                            reply_only = False
                            for kw2 in node.keywords:
                                if kw2.arg == "reply_only":
                                    reply_only = self._bool_value(kw2.value)
                            self.bus_subscribes.append({
                                "line": node.lineno,
                                "name": mod_name,
                                "reply_only": reply_only,
                                "context": self.context,
                                "via": "guardian_module",
                            })
                        break

            # ── getattr detection ──
            elif func_name == "getattr" and len(node.args) >= 2:
                obj = self._name_of(node.args[0])
                attr = self._str_value(node.args[1])
                if obj and attr:
                    self.getattr_calls.append({
                        "line": node.lineno,
                        "obj": obj,
                        "attr": attr,
                        "context": self.context,
                    })

            # ── General function call tracking ──
            else:
                # Track meaningful calls (skip builtins, common utils)
                if not self._is_trivial_call(func_name):
                    self.function_calls.append({
                        "line": node.lineno,
                        "func": func_name,
                        "context": self.context,
                    })

        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Track attribute access on key objects
        obj_name = self._name_of(node.value)
        if obj_name and node.attr:
            # Filter to meaningful state objects
            key_objects = {
                "inner_state", "self", "coordinator", "life_force_engine",
                "neural_nervous_system", "neuromodulator_system",
                "dreaming_engine", "consciousness", "expression_manager",
                "composition_engine", "prediction_engine", "working_mem",
                "ex_mem", "episodic_mem", "pi_monitor", "chi",
            }
            if obj_name in key_objects or obj_name.startswith("self."):
                # Determine read vs write
                mode = "read"
                if isinstance(getattr(node, '_parent', None), (ast.Assign, ast.AugAssign)):
                    mode = "write"

                self.attr_access.append({
                    "line": node.lineno,
                    "obj": obj_name,
                    "attr": node.attr,
                    "context": self.context,
                    "mode": mode,
                })
        self.generic_visit(node)

    # ── Helpers ──

    def _call_name(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj = self._name_of(node.func.value)
            if obj:
                return f"{obj}.{node.func.attr}"
            return node.func.attr
        return ""

    def _name_of(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parent = self._name_of(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def _str_value(self, node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    def _bool_value(self, node) -> bool:
        if isinstance(node, ast.Constant):
            return bool(node.value)
        if isinstance(node, ast.Name):
            return node.id == "True"
        return False

    def _decorator_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._name_of(node)
        elif isinstance(node, ast.Call):
            return self._call_name(node)
        return "?"

    def _extract_bus_publish_type(self, node) -> str:
        """Extract message type from bus.publish({type: "X", ...})."""
        if node.args and isinstance(node.args[0], ast.Dict):
            d = node.args[0]
            for key, val in zip(d.keys, d.values):
                key_str = self._str_value(key) if key else ""
                if key_str == "type":
                    return self._str_value(val)
        return ""

    def _resolve_value(self, node) -> str:
        """Resolve a value node to string — handles constants AND bus constant names."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            # Try to resolve from bus constants
            if node.id in self.bus_constants:
                return self.bus_constants[node.id]
            # Return the variable name as-is for identifiable constants
            if node.id.isupper():
                return node.id
        return ""

    def visit_Compare(self, node):
        """Detect msg_type == 'X' patterns for consumer identification."""
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            left = self._name_of(node.left) if hasattr(node, 'left') else ""
            right_val = self._resolve_value(node.comparators[0]) if node.comparators else ""
            # Check if this is comparing a message type variable
            if left in ("msg_type", "_msg_type", "msg.type", "message_type"):
                if right_val:
                    self.msg_consumers.append({
                        "line": node.lineno,
                        "msg_type": right_val,
                        "context": self.context,
                    })
            elif right_val and right_val in self.bus_constants.values():
                # Also catch: if msg["type"] == EPOCH_TICK etc
                self.msg_consumers.append({
                    "line": node.lineno,
                    "msg_type": right_val,
                    "context": self.context,
                })
            else:
                # Detect .get("type") == "X" pattern (used in v5_core meditation loop)
                # AST: Compare(Call(Attribute(Name, 'get'), [Constant('type')]), [Constant('X')])
                # Only match if the value looks like a bus message type (UPPER_CASE)
                left_node = node.left
                if (isinstance(left_node, ast.Call)
                        and isinstance(left_node.func, ast.Attribute)
                        and left_node.func.attr == "get"
                        and left_node.args
                        and self._str_value(left_node.args[0]) == "type"
                        and right_val
                        and (right_val.isupper() or right_val in self.bus_constants.values())):
                    self.msg_consumers.append({
                        "line": node.lineno,
                        "msg_type": right_val,
                        "context": self.context,
                    })
        self.generic_visit(node)

    def _is_trivial_call(self, name: str) -> bool:
        trivial = {
            "print", "len", "range", "int", "float", "str", "bool",
            "list", "dict", "tuple", "set", "min", "max", "abs",
            "round", "sorted", "enumerate", "zip", "map", "filter",
            "isinstance", "hasattr", "type", "super", "open",
            "logger.debug", "logger.info", "logger.warning", "logger.error",
            "logger.exception", "logging.getLogger", "time.time",
            "time.monotonic", "json.dumps", "json.loads", "os.path.join",
            "math.exp", "math.log", "math.sqrt", "math.tanh",
            "np.array", "np.zeros", "np.mean", "np.std", "np.clip",
            "np.dot", "np.linalg.norm", "np.sum", "np.concatenate",
            "asyncio.sleep", "asyncio.wait_for", "asyncio.create_task",
        }
        return name in trivial or name.startswith("_log")


def scan_file(filepath: Path, bus_constants: dict = None) -> dict:
    """Parse a single Python file and return extracted data."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "filepath": rel(filepath)}

    # Add parent references for write detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node

    visitor = TitanASTVisitor(rel(filepath), bus_constants=bus_constants)
    visitor.visit(tree)

    return {
        "filepath": rel(filepath),
        "lines": source.count("\n") + 1,
        "classes": visitor.classes,
        "functions": visitor.functions,
        "imports": visitor.imports,
        "from_imports": visitor.from_imports,
        "bus_publishes": visitor.bus_publishes,
        "bus_subscribes": visitor.bus_subscribes,
        "send_msgs": visitor.send_msgs,
        "msg_consumers": visitor.msg_consumers,
        "function_calls": visitor.function_calls,
        "attr_access": visitor.attr_access,
        "getattr_calls": visitor.getattr_calls,
    }


# ── Graph Builder ─────────────────────────────────────────────────────

def build_graph(scan_dirs, scripts_dirs=None):
    """Scan all Python files and build the architecture graph."""
    files_data = {}
    bus_wiring = defaultdict(lambda: {"publishers": [], "subscribers": [], "send_msgs": [], "consumers": []})
    attr_index = defaultdict(lambda: {"readers": [], "writers": [], "getattr": []})
    call_index = defaultdict(list)  # func_name -> [{file, line, context}]
    definitions = defaultdict(list)  # func_name -> [{file, line, class}]
    import_graph = defaultdict(set)  # file -> set of imported modules

    # Pre-load bus constants for variable resolution
    bus_constants = load_bus_constants()

    all_dirs = list(scan_dirs or [])
    if scripts_dirs:
        all_dirs.extend(scripts_dirs)

    py_files = []
    for scan_dir in all_dirs:
        if scan_dir.exists():
            py_files.extend(sorted(scan_dir.rglob("*.py")))

    for fpath in py_files:
        data = scan_file(fpath, bus_constants=bus_constants)
        fp = data["filepath"]
        files_data[fp] = data

        # ── Bus wiring index ──
        for pub in data.get("bus_publishes", []):
            bus_wiring[pub["msg_type"]]["publishers"].append({
                "file": fp, "line": pub["line"], "context": pub["context"],
            })
        for msg in data.get("send_msgs", []):
            bus_wiring[msg["msg_type"]]["send_msgs"].append({
                "file": fp, "line": msg["line"], "dst": msg["dst"],
                "context": msg["context"],
            })
        for sub in data.get("bus_subscribes", []):
            bus_wiring[f"_SUB:{sub['name']}"]["subscribers"].append({
                "file": fp, "line": sub["line"],
                "reply_only": sub["reply_only"], "context": sub["context"],
            })

        # ── Message consumer index (msg_type == "X" checks) ──
        for con in data.get("msg_consumers", []):
            bus_wiring[con["msg_type"]]["consumers"].append({
                "file": fp, "line": con["line"], "context": con["context"],
            })

        # ── Attribute access index ──
        for acc in data.get("attr_access", []):
            key = f"{acc['obj']}.{acc['attr']}"
            bucket = "writers" if acc["mode"] == "write" else "readers"
            attr_index[key][bucket].append({
                "file": fp, "line": acc["line"], "context": acc["context"],
            })
        for ga in data.get("getattr_calls", []):
            key = f"{ga['obj']}.{ga['attr']}"
            attr_index[key]["getattr"].append({
                "file": fp, "line": ga["line"], "context": ga["context"],
            })

        # ── Function call index ──
        for call in data.get("function_calls", []):
            call_index[call["func"]].append({
                "file": fp, "line": call["line"], "context": call["context"],
            })

        # ── Definition index ──
        for fname, finfo in data.get("functions", {}).items():
            definitions[fname].append({
                "file": fp, "line": finfo["line"], "type": "function",
            })
        for cname, cinfo in data.get("classes", {}).items():
            definitions[cname].append({
                "file": fp, "line": cinfo["line"], "type": "class",
            })
            for mname, minfo in cinfo.get("methods", {}).items():
                definitions[f"{cname}.{mname}"].append({
                    "file": fp, "line": minfo["line"], "type": "method",
                })

        # ── Import graph ──
        for mod, _ in data.get("imports", []):
            import_graph[fp].add(mod)
        for mod, _ in data.get("from_imports", []):
            import_graph[fp].add(mod)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_root": [str(d) for d in scan_dirs],
        "total_files": len(files_data),
        "total_lines": sum(f.get("lines", 0) for f in files_data.values()),
        "files": files_data,
        "bus_wiring": {k: v for k, v in sorted(bus_wiring.items())},
        "attr_index": {k: v for k, v in sorted(attr_index.items())},
        "call_index": {k: v for k, v in sorted(call_index.items())},
        "definitions": {k: v for k, v in sorted(definitions.items())},
        "import_graph": {k: sorted(v) for k, v in sorted(import_graph.items())},
    }


# ── Query Engine ──────────────────────────────────────────────────────

def load_graph() -> dict:
    if not OUTPUT_FILE.exists():
        print(f"No architecture map found. Run: python {sys.argv[0]} scan")
        sys.exit(1)
    return json.loads(OUTPUT_FILE.read_text())


def query_bus(graph: dict, msg_type: str):
    """Show all publishers, subscribers, and send_msg calls for a bus message type."""
    wiring = graph.get("bus_wiring", {})

    # Exact match first, then fuzzy
    matches = {}
    for key, val in wiring.items():
        if key.startswith("_SUB:"):
            continue
        if msg_type.upper() in key.upper():
            matches[key] = val

    if not matches:
        print(f"No bus messages matching '{msg_type}'")
        # Show available message types
        types = [k for k in wiring if not k.startswith("_SUB:")]
        print(f"\nAvailable message types ({len(types)}):")
        for t in sorted(types):
            print(f"  {t}")
        return

    for mtype, data in sorted(matches.items()):
        print(f"\n{'='*60}")
        print(f"BUS MESSAGE: {mtype}")
        print(f"{'='*60}")

        pubs = data.get("publishers", [])
        sends = data.get("send_msgs", [])
        consumers = data.get("consumers", [])
        if pubs:
            print(f"\n  Publishers ({len(pubs)}):")
            for p in pubs:
                dst = p.get("dst", "")
                dst_str = f"  -> dst={dst}" if dst else ""
                print(f"    {p['file']}:{p['line']}{dst_str}  [{p['context']}]")
        if sends:
            print(f"\n  send_msg ({len(sends)}):")
            for s in sends:
                print(f"    {s['file']}:{s['line']}  -> dst={s['dst']}  [{s['context']}]")

        if consumers:
            print(f"\n  Consumers (msg_type == check) ({len(consumers)}):")
            for c in consumers:
                print(f"    {c['file']}:{c['line']}  [{c['context']}]")

        # Find subscribers that might receive this
        all_dsts = set()
        for s in sends:
            all_dsts.add(s["dst"])
        for p in pubs:
            if "dst" in p:
                all_dsts.add(p["dst"])

        if all_dsts:
            print(f"\n  Routing:")
            if "all" in all_dsts:
                print(f"    Broadcast (dst=all) — received by all non-reply_only subscribers")
            for dst in sorted(all_dsts - {"all", "?", ""}):
                sub_key = f"_SUB:{dst}"
                if sub_key in wiring:
                    for sub in wiring[sub_key]["subscribers"]:
                        print(f"    -> subscriber='{dst}' reply_only={sub['reply_only']}  {sub['file']}:{sub['line']}  [{sub['context']}]")
                else:
                    print(f"    -> dst='{dst}' (NO SUBSCRIBER FOUND — possible dead letter!)")


def query_calls(graph: dict, func_name: str):
    """Show all callers of a function and where it's defined."""
    defs = graph.get("definitions", {})
    calls = graph.get("call_index", {})

    # Find definitions
    matches_def = {}
    for key, val in defs.items():
        if func_name.lower() in key.lower():
            matches_def[key] = val

    # Find callers
    matches_call = {}
    for key, val in calls.items():
        parts = key.split(".")
        if any(func_name.lower() in p.lower() for p in parts):
            matches_call[key] = val

    if not matches_def and not matches_call:
        print(f"No matches for '{func_name}'")
        return

    if matches_def:
        print(f"\nDefinitions matching '{func_name}':")
        for name, locs in sorted(matches_def.items()):
            for loc in locs:
                print(f"  {loc['type']:8s} {name}  {loc['file']}:{loc['line']}")

    if matches_call:
        print(f"\nCallers matching '{func_name}':")
        for name, callers in sorted(matches_call.items()):
            print(f"\n  {name} ({len(callers)} calls):")
            for c in callers[:20]:
                print(f"    {c['file']}:{c['line']}  [{c['context']}]")
            if len(callers) > 20:
                print(f"    ... and {len(callers)-20} more")


def query_file(graph: dict, filename: str):
    """Show summary of a specific file."""
    files = graph.get("files", {})
    matches = {k: v for k, v in files.items() if filename.lower() in k.lower()}

    if not matches:
        print(f"No files matching '{filename}'")
        return

    for fp, data in sorted(matches.items()):
        print(f"\n{'='*60}")
        print(f"FILE: {fp} ({data.get('lines', '?')} lines)")
        print(f"{'='*60}")

        if data.get("classes"):
            print(f"\n  Classes ({len(data['classes'])}):")
            for cname, cinfo in sorted(data["classes"].items()):
                bases = ", ".join(cinfo.get("bases", [])) or "object"
                print(f"    class {cname}({bases})  line {cinfo['line']}")
                for mname, minfo in sorted(cinfo.get("methods", {}).items()):
                    args = ", ".join(minfo["args"])
                    print(f"      def {mname}({args})  line {minfo['line']}")

        if data.get("functions"):
            print(f"\n  Functions ({len(data['functions'])}):")
            for fname, finfo in sorted(data["functions"].items()):
                args = ", ".join(finfo["args"])
                print(f"    def {fname}({args})  line {finfo['line']}")

        if data.get("bus_publishes"):
            print(f"\n  Bus Publishes ({len(data['bus_publishes'])}):")
            for p in data["bus_publishes"]:
                print(f"    {p['msg_type']}  line {p['line']}  [{p['context']}]")

        if data.get("send_msgs"):
            print(f"\n  send_msg ({len(data['send_msgs'])}):")
            for s in data["send_msgs"]:
                print(f"    {s['msg_type']} -> {s['dst']}  line {s['line']}  [{s['context']}]")

        if data.get("bus_subscribes"):
            print(f"\n  Bus Subscribes ({len(data['bus_subscribes'])}):")
            for s in data["bus_subscribes"]:
                print(f"    '{s['name']}' reply_only={s['reply_only']}  line {s['line']}")

        imports = data.get("from_imports", [])
        if imports:
            print(f"\n  Imports ({len(imports)}):")
            for mod, names in imports[:15]:
                print(f"    from {mod} import {', '.join(names[:5])}")
            if len(imports) > 15:
                print(f"    ... and {len(imports)-15} more")


def query_attr(graph: dict, pattern: str):
    """Show readers/writers of an attribute."""
    index = graph.get("attr_index", {})
    matches = {k: v for k, v in index.items() if pattern.lower() in k.lower()}

    if not matches:
        print(f"No attribute access matching '{pattern}'")
        print(f"\nSample attributes tracked ({min(20, len(index))}):")
        for k in sorted(index.keys())[:20]:
            print(f"  {k}")
        return

    for attr_key, data in sorted(matches.items()):
        readers = data.get("readers", [])
        writers = data.get("writers", [])
        getattrs = data.get("getattr", [])

        print(f"\n{'='*60}")
        print(f"ATTRIBUTE: {attr_key}")
        print(f"{'='*60}")

        if writers:
            print(f"\n  Writers ({len(writers)}):")
            for w in writers:
                print(f"    {w['file']}:{w['line']}  [{w['context']}]")
        if readers:
            print(f"\n  Readers ({len(readers)}):")
            for r in readers[:20]:
                print(f"    {r['file']}:{r['line']}  [{r['context']}]")
            if len(readers) > 20:
                print(f"    ... and {len(readers)-20} more")
        if getattrs:
            print(f"\n  getattr() ({len(getattrs)}):")
            for g in getattrs:
                print(f"    {g['file']}:{g['line']}  [{g['context']}]")


def query_depends(graph: dict, filename: str):
    """Show what files depend on (import from) a given file."""
    import_graph = graph.get("import_graph", {})
    files = graph.get("files", {})

    # Find the module path for the target file
    target_modules = set()
    for fp in files:
        if filename.lower() in fp.lower():
            # Convert file path to module: titan_plugin/logic/dreaming.py -> titan_plugin.logic.dreaming
            mod = fp.replace("/", ".").replace(".py", "")
            target_modules.add(mod)
            # Also add short forms
            parts = mod.split(".")
            for i in range(len(parts)):
                target_modules.add(".".join(parts[i:]))

    if not target_modules:
        print(f"No files matching '{filename}'")
        return

    print(f"\nTarget modules: {', '.join(sorted(target_modules))}")

    dependents = []
    for fp, imports in import_graph.items():
        for imp in imports:
            # Precise matching: import must end with a target module component
            # e.g., "titan_plugin.logic.dreaming" matches target "dreaming"
            imp_parts = imp.split(".")
            for tm in target_modules:
                tm_parts = tm.split(".")
                # Check if the import ends with or contains the target as a full path segment
                if imp_parts[-len(tm_parts):] == tm_parts or tm_parts[-1] in imp_parts:
                    dependents.append((fp, imp))
                    break

    if dependents:
        print(f"\nFiles that depend on '{filename}' ({len(dependents)}):")
        for fp, imp in sorted(dependents):
            print(f"  {fp}  (imports {imp})")
    else:
        print(f"\nNo files import from '{filename}'")

    # Also show bus connections
    wiring = graph.get("bus_wiring", {})
    file_msgs = set()
    for fp, data in files.items():
        if filename.lower() in fp.lower():
            for msg in data.get("send_msgs", []):
                file_msgs.add(msg["msg_type"])
            for pub in data.get("bus_publishes", []):
                file_msgs.add(pub["msg_type"])

    if file_msgs:
        print(f"\nBus messages produced by '{filename}': {', '.join(sorted(file_msgs))}")
        print("  (Other files may depend on these messages — use 'query bus:MSG' to check)")


def query_class(graph: dict, class_name: str):
    """Show class definition, methods, and usage."""
    files = graph.get("files", {})

    for fp, data in sorted(files.items()):
        for cname, cinfo in data.get("classes", {}).items():
            if class_name.lower() in cname.lower():
                print(f"\n{'='*60}")
                bases = ", ".join(cinfo.get("bases", [])) or "object"
                print(f"CLASS: {cname}({bases})  in {fp}:{cinfo['line']}")
                print(f"{'='*60}")

                methods = cinfo.get("methods", {})
                print(f"\n  Methods ({len(methods)}):")
                for mname, minfo in sorted(methods.items()):
                    args = ", ".join(minfo["args"])
                    decs = ""
                    if minfo.get("decorators"):
                        decs = f"  @{', '.join(minfo['decorators'])}"
                    print(f"    def {mname}({args})  line {minfo['line']}{decs}")

    # Show callers
    calls = graph.get("call_index", {})
    relevant = {}
    for key, callers in calls.items():
        if class_name.lower() in key.lower():
            relevant[key] = callers

    if relevant:
        print(f"\n  Callers:")
        for key, callers in sorted(relevant.items()):
            for c in callers[:5]:
                print(f"    {key}  from {c['file']}:{c['line']}  [{c['context']}]")


def show_wiring(graph: dict):
    """Display complete bus wiring overview."""
    wiring = graph.get("bus_wiring", {})

    # Separate message types from subscriber entries
    msg_types = {k: v for k, v in wiring.items() if not k.startswith("_SUB:")}
    subs = {k: v for k, v in wiring.items() if k.startswith("_SUB:")}

    print(f"\n{'='*60}")
    print(f"DIVINE BUS WIRING — {len(msg_types)} message types, {len(subs)} subscribers")
    print(f"{'='*60}")

    print(f"\n  Subscribers:")
    for key, data in sorted(subs.items()):
        name = key.replace("_SUB:", "")
        for s in data["subscribers"]:
            print(f"    '{name}' reply_only={s['reply_only']}  {s['file']}:{s['line']}")

    print(f"\n  Message Types:")
    for mtype, data in sorted(msg_types.items()):
        pubs = len(data.get("publishers", []))
        sends = len(data.get("send_msgs", []))
        dsts = set(s["dst"] for s in data.get("send_msgs", []))
        dst_str = ", ".join(sorted(dsts)) if dsts else "n/a"
        print(f"    {mtype:30s}  pub={pubs} send={sends}  dst=[{dst_str}]")


def show_summary(graph: dict):
    """Display high-level architecture statistics."""
    files = graph.get("files", {})
    wiring = graph.get("bus_wiring", {})
    attrs = graph.get("attr_index", {})
    calls = graph.get("call_index", {})
    defs = graph.get("definitions", {})

    total_classes = sum(len(f.get("classes", {})) for f in files.values())
    total_functions = sum(len(f.get("functions", {})) for f in files.values())
    total_methods = sum(
        sum(len(c.get("methods", {})) for c in f.get("classes", {}).values())
        for f in files.values()
    )
    msg_types = [k for k in wiring if not k.startswith("_SUB:")]
    sub_types = [k for k in wiring if k.startswith("_SUB:")]

    print(f"\n{'='*60}")
    print(f"TITAN ARCHITECTURE SUMMARY")
    print(f"Generated: {graph.get('generated_at', '?')}")
    print(f"{'='*60}")
    print(f"\n  Files:        {len(files)}")
    print(f"  Total lines:  {graph.get('total_lines', '?'):,}")
    print(f"  Classes:      {total_classes}")
    print(f"  Methods:      {total_methods}")
    print(f"  Functions:    {total_functions}")
    print(f"  Definitions:  {len(defs)}")
    print(f"\n  Bus message types:  {len(msg_types)}")
    print(f"  Bus subscribers:    {len(sub_types)}")
    print(f"  Tracked attributes: {len(attrs)}")
    print(f"  Function calls:     {sum(len(v) for v in calls.values()):,}")

    # Top files by size
    print(f"\n  Top 10 files by size:")
    by_size = sorted(files.items(), key=lambda x: x[1].get("lines", 0), reverse=True)
    for fp, data in by_size[:10]:
        nc = len(data.get("classes", {}))
        nf = len(data.get("functions", {}))
        nm = len(data.get("send_msgs", []))
        print(f"    {data.get('lines', 0):5d} lines  {nc}C {nf}F {nm}msg  {fp}")

    # Dead letter detection
    print(f"\n  Potential Dead Letters (send_msg with no matching subscriber):")
    all_sub_names = set(k.replace("_SUB:", "") for k in sub_types)
    for mtype, data in sorted(wiring.items()):
        if mtype.startswith("_SUB:"):
            continue
        for s in data.get("send_msgs", []):
            if s["dst"] not in all_sub_names and s["dst"] != "all" and s["dst"] != "?":
                print(f"    {mtype} -> dst='{s['dst']}'  {s['file']}:{s['line']}")


def show_audit(graph: dict):
    """Signal integrity audit — find dead signals, deaf ears, and wasted bus traffic."""
    wiring = graph.get("bus_wiring", {})
    msg_types = {k: v for k, v in wiring.items() if not k.startswith("_SUB:")}
    subscribers = {k.replace("_SUB:", ""): v for k, v in wiring.items() if k.startswith("_SUB:")}
    all_sub_names = set(subscribers.keys())

    print("=" * 95)
    print("TITAN BUS SIGNAL INTEGRITY AUDIT")
    print("=" * 95)

    # A: Sent/published but no consumer code
    #
    # Filters (2026-04-08 later audit):
    # - INTENTIONAL_BROADCAST marker (±10 lines): dst=all telemetry consumed
    #   by frontend WebSocket via /v4/events stream — no in-process consumer.
    # - INTENTIONAL_SELF_ROUTE marker (±10 lines): self-routed messages that
    #   are dispatched via interpreter registry, not msg_type comparison
    #   (arch_map AST parser misses this pattern).
    # Tracked under I-004.
    def _has_intentional_marker(senders_pubs):
        """Check if all sender/publisher locations have INTENTIONAL_* marker."""
        if not senders_pubs:
            return False
        for loc in senders_pubs:
            fpath = PROJECT_ROOT / loc.get("file", "")
            if not fpath.exists():
                return False
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                return False
            line_no = loc.get("line", 0)
            lo = max(0, line_no - 11)
            hi = min(len(lines), line_no + 4)
            context = " ".join(lines[lo:hi]).upper()
            if not any(marker in context for marker in
                       ("INTENTIONAL_BROADCAST", "INTENTIONAL_SELF_ROUTE",
                        "INTENTIONAL BROADCAST", "INTENTIONAL SELF ROUTE")):
                return False
        return True

    print("\n══ A. SIGNALS WITH NO CONSUMER (sent/published but nobody reads msg_type==X) ══")
    a_count = 0
    a_intentional = 0
    for mtype in sorted(msg_types.keys()):
        data = msg_types[mtype]
        sends = data.get("send_msgs", [])
        pubs = data.get("publishers", [])
        consumers = data.get("consumers", [])
        if (sends or pubs) and not consumers:
            # Skip if all sender/publisher locations are explicitly marked
            # as intentional broadcast or self-route patterns.
            if _has_intentional_marker(list(sends) + list(pubs)):
                a_intentional += 1
                continue
            a_count += 1
            dsts = set(s.get("dst", "") for s in sends)
            locs = [f"{s['file']}:{s['line']} dst={s['dst']}" for s in sends[:2]]
            locs += [f"{p['file']}:{p['line']} (publish)" for p in pubs[:2]]
            print(f"  ⚠ {mtype:<25} send={len(sends)} pub={len(pubs)} → 0 consumers")
            for l in locs:
                print(f"      {l}")
    if a_count == 0:
        print("  ✓ All sent/published messages have consumers")
    if a_intentional > 0:
        print(f"  ℹ {a_intentional} intentional broadcast/self-route signal(s) skipped "
              f"(marked INTENTIONAL_BROADCAST/INTENTIONAL_SELF_ROUTE in source)")

    # B: Consumer code exists but nothing sends/publishes it
    #
    # Filters applied (2026-04-08 audit):
    # - Consumers outside titan_plugin/ are skipped (scripts/arch_map.py
    #   session-close parser uses `msg["role"] == "assistant"` idiomatically
    #   without it being a DivineBus message type — false positive source)
    # - Consumers marked DEPRECATED/REMOVED/NO_LONGER_USED in the ±3 line
    #   context are reported as "intentional deprecated" instead of deaf ears
    print("\n══ B. DEAF EARS (consumer code but no sender/publisher) ══")
    b_count = 0
    deprecated_count = 0

    def _is_deprecated_handler(consumers_list):
        """Check if all consumers have DEPRECATED/REMOVED/API_STUB markers in context.

        2026-04-08 (later): added API_STUB marker for handlers that are
        intentionally implemented as future-API endpoints awaiting upstream
        wiring (e.g., TimeChain v2 Phase 2 RECALL/CHECK handlers, contract
        engine, CGN_KNOWLEDGE_USAGE awaiting cross-consumer wiring).
        These are tracked in known_issues.md I-003 as 'incomplete feature'
        not 'dead handler'.
        """
        if not consumers_list:
            return False
        for c in consumers_list:
            fpath = PROJECT_ROOT / c.get("file", "")
            if not fpath.exists():
                return False
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                return False
            line_no = c.get("line", 0)
            # Check ±20 lines for deprecation/stub markers (widened from 3 to
            # capture multi-line block comments above nested if/elif handlers,
            # e.g., timechain_worker dispatches TIMECHAIN_RECALL/CHECK/COMPARE
            # under one outer elif block with the API_STUB marker on the
            # outer block).
            lo = max(0, line_no - 21)
            hi = min(len(lines), line_no + 6)
            context = " ".join(lines[lo:hi]).upper()
            if not any(marker in context for marker in
                       ("DEPRECATED", "REMOVED", "NO_LONGER_USED",
                        "NO LONGER USED", "API_STUB", "API STUB")):
                return False
        return True

    # ── Runtime-published signals (cognitive contracts JSON + helper fns) ──
    # Some msg types are published at runtime by the TitanVM contract
    # interpreter after evaluating JSON contracts (titan_plugin/contracts/**).
    # AST analysis cannot see these — scan the JSON files for "event": "TYPE"
    # entries and treat those types as having runtime publishers.
    import json as _json, glob as _glob
    _runtime_pubs: dict = defaultdict(list)
    for jf in _glob.glob("titan_plugin/contracts/**/*.json", recursive=True):
        try:
            with open(jf) as _jfh:
                _jd = _json.load(_jfh)
            # Traverse dict/list looking for "event": "MSG_TYPE" entries
            def _walk(obj, path_hint=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "event" and isinstance(v, str):
                            _runtime_pubs[v].append({
                                "file": jf, "line": 0, "dst": "?",
                            })
                        else:
                            _walk(v, path_hint=f"{path_hint}.{k}")
                elif isinstance(obj, list):
                    for item in obj:
                        _walk(item, path_hint)
            _walk(_jd)
        except Exception:
            continue

    # Known-helper publishers — functions in bus.py/logic/*.py that build a
    # msg dict and dispatch via duck-typed sender.put_nowait/publish. Static
    # analysis can't trace because msg type appears as a bus-constant Name
    # reference inside a variable assignment rather than a make_msg call.
    # This allowlist closes the gap without needing full data-flow analysis.
    _KNOWN_HELPER_PUBS = {
        # emit_meta_cgn_signal → META_CGN_SIGNAL (bus.py:415)
        "emit_meta_cgn_signal": "META_CGN_SIGNAL",
        # emit_emot_cgn_signal → EMOT_CGN_SIGNAL (bus.py, rFP_emot_cgn_v2)
        "emit_emot_cgn_signal": "EMOT_CGN_SIGNAL",
        # emit_emot_chain_evidence → EMOT_CHAIN_EVIDENCE (Phase 1.6d ADR)
        "emit_emot_chain_evidence": "EMOT_CHAIN_EVIDENCE",
        # emit_felt_cluster_update → FELT_CLUSTER_UPDATE (Phase 1.6d ADR)
        "emit_felt_cluster_update": "FELT_CLUSTER_UPDATE",
        # EmotCGNConsumer._maybe_emit_cross_insight → CGN_CROSS_INSIGHT
        # (class method, not top-level function — scanner needs hint)
        "_maybe_emit_cross_insight": "CGN_CROSS_INSIGHT",
    }

    # Helper publisher discovery: functions whose body wraps make_msg/_send_msg
    # for a specific msg_type act as publishers for that type when their
    # CALLERS invoke them. Also combines with the _KNOWN_HELPER_PUBS allowlist.
    _files = graph.get("files", {})
    _helper_pubs: dict[str, str] = {}  # function_name → msg_type
    # Build from the raw bus_publishes + send_msgs — each publish/send site
    # records its enclosing context (function name). Top-level functions
    # (no "." in context) are helper candidates.
    for mtype_name, data in msg_types.items():
        for p in data.get("publishers", []):
            ctx = p.get("context", "")
            if ctx and "." not in ctx and ctx != "<module>":
                _helper_pubs.setdefault(ctx, mtype_name)
        for s in data.get("send_msgs", []):
            ctx = s.get("context", "")
            if ctx and "." not in ctx and ctx != "<module>":
                _helper_pubs.setdefault(ctx, mtype_name)
    for k in ("make_msg", "_send_msg"):
        _helper_pubs.pop(k, None)
    # Merge known-helper allowlist (emit_meta_cgn_signal etc.) with discovered
    # helpers; allowlist takes precedence.
    for _name, _mtype in _KNOWN_HELPER_PUBS.items():
        _helper_pubs[_name] = _mtype
    # Walk function_calls: if a caller invokes a helper_pub, credit caller
    # file as a publisher of the mapped msg_type.
    _helper_caller_pubs: dict = defaultdict(list)
    for fp, fdata in _files.items():
        for fc in fdata.get("function_calls", []):
            fn = fc.get("func", "")
            short = fn.split(".")[-1] if fn else ""
            if short and short in _helper_pubs:
                mtype = _helper_pubs[short]
                if mtype:
                    _helper_caller_pubs[mtype].append({
                        "file": fp, "line": fc.get("line", 0), "dst": "?",
                    })
    # Fold runtime-published signals (from contract JSON) into the caller list
    # so Section B's "0 senders" check passes for contract-fired events.
    for _mt, _locs in _runtime_pubs.items():
        _helper_caller_pubs[_mt].extend(_locs)

    for mtype in sorted(msg_types.keys()):
        data = msg_types[mtype]
        sends = data.get("send_msgs", [])
        pubs = data.get("publishers", [])
        # Augment with any callers of helper-publisher functions for this type.
        pubs = list(pubs) + _helper_caller_pubs.get(mtype, [])
        consumers = data.get("consumers", [])
        # Filter consumers to only those inside titan_plugin/ (the bus's canonical tree)
        valid_consumers = [c for c in consumers
                          if c.get("file", "").startswith("titan_plugin/")]
        if not sends and not pubs and valid_consumers:
            if _is_deprecated_handler(valid_consumers):
                deprecated_count += 1
                continue  # Intentional deprecated handler — not a deaf ear
            # Skip QUERY/RESPONSE rid-routed pairs — audit can't trace rid
            # dispatch (response routed by request-id, not by msg_type).
            # Heuristic: if msg name ends with _RESP / _RESPONSE / _REPLY or
            # is a QUERY response, assume it's rid-routed.
            if mtype.endswith(("_RESP", "_RESPONSE", "_REPLY")):
                deprecated_count += 1  # Counted under "skipped" bucket
                continue
            b_count += 1
            print(f"  ⚠ {mtype:<25} → {len(valid_consumers)} consumer(s) waiting, 0 senders")
            for c in valid_consumers[:2]:
                print(f"      {c['file']}:{c['line']}  [{c['context']}]")
    if b_count == 0:
        print("  ✓ All consumers have matching senders")
    if deprecated_count > 0:
        print(f"  ℹ {deprecated_count} deprecated handler(s) + rid-routed response pairs skipped (marked REMOVED/DEPRECATED or ending with _RESP/_REPLY)")

    # C: Targeted messages to non-existent subscribers (dead letters)
    print("\n══ C. DEAD LETTERS (targeted to non-existent subscriber) ══")
    c_count = 0
    for mtype, data in sorted(msg_types.items()):
        for s in data.get("send_msgs", []):
            dst = s.get("dst", "")
            if dst and dst != "?" and dst != "all" and dst not in all_sub_names:
                c_count += 1
                print(f"  ✗ {mtype} → dst='{dst}'  {s['file']}:{s['line']}")
        for p in data.get("publishers", []):
            dst = p.get("dst", "")
            if dst and dst != "?" and dst != "all" and dst not in all_sub_names:
                c_count += 1
                print(f"  ✗ {mtype} → dst='{dst}'  {p['file']}:{p['line']} (publish)")
    if c_count == 0:
        print("  ✓ All targeted messages reach valid subscribers")

    # D: Completely dead (no send, no pub, no consumer)
    print("\n══ D. COMPLETELY DEAD MESSAGE TYPES ══")
    d_count = 0
    for mtype in sorted(msg_types.keys()):
        data = msg_types[mtype]
        if not data.get("send_msgs") and not data.get("publishers") and not data.get("consumers"):
            d_count += 1
            print(f"  ✗ {mtype:<25} — defined but never sent, published, or consumed")
    if d_count == 0:
        print("  ✓ No dead message types")

    # E: Reply-only subscribers with no targeted traffic
    print("\n══ E. IDLE REPLY-ONLY SUBSCRIBERS ══")
    targeted_dsts = set()
    for data in msg_types.values():
        for s in data.get("send_msgs", []):
            if s.get("dst") and s["dst"] not in ("all", "?"):
                targeted_dsts.add(s["dst"])
        for p in data.get("publishers", []):
            if p.get("dst") and p["dst"] not in ("all", "?"):
                targeted_dsts.add(p["dst"])
    e_count = 0
    for sub_name, sub_data in sorted(subscribers.items()):
        is_reply_only = any(s.get("reply_only", False) for s in sub_data.get("subscribers", []))
        if is_reply_only and sub_name not in targeted_dsts:
            e_count += 1
            loc = sub_data["subscribers"][0] if sub_data.get("subscribers") else {}
            print(f"  ? '{sub_name}' (reply_only) — no targeted messages detected")
            print(f"      note: may receive QUERY/RESPONSE via dynamic dst (proxy pattern)")
    if e_count == 0:
        print("  ✓ All reply-only subscribers receive targeted traffic")

    # F: Scope leaks
    print("\n══ F. SCOPE LEAKS (_handle_query referencing enclosing-scope locals) ══")
    scope_leaks = _audit_scope_leaks()
    f_count = 0
    if scope_leaks:
        for leak in scope_leaks:
            f_count += len(leak["leaks"])
            print(f"  ✗ {leak['file']}:{leak['line']}  {leak['func']}()")
            print(f"    Parameters: {', '.join(leak['params'][:10])}{'...' if len(leak['params']) > 10 else ''}")
            print(f"    Scope leaks ({len(leak['leaks'])}): {', '.join(leak['leaks'])}")
    if f_count == 0:
        print("  ✓ No scope leaks detected in standalone _handle_query functions")

    # G: Dual-path completeness
    print("\n══ G. DUAL-PATH COMPLETENESS (periodic tick vs EPOCH_TICK handler) ══")
    dual = _audit_dual_path()
    g_count = 0
    if dual and "error" in dual:
        print(f"  ? {dual['error']}")
    elif dual:
        print(f"    Periodic tick path: {dual['periodic_range']}")
        print(f"    EPOCH_TICK handler: {dual['epoch_tick_range']}")
        for check in dual["checks"]:
            if check["status"] == "OK":
                p_lines = ", ".join(str(l) for l in check["periodic_lines"])
                e_lines = ", ".join(str(l) for l in check["epoch_tick_lines"])
                print(f"  ✓ {check['call']:<40s}  periodic=[L{p_lines}]  epoch=[L{e_lines}]")
            else:
                g_count += 1
                where = []
                if check["in_periodic"]:
                    where.append(f"periodic ONLY [L{', '.join(str(l) for l in check['periodic_lines'])}]")
                elif check["in_epoch_tick"]:
                    where.append(f"EPOCH_TICK ONLY [L{', '.join(str(l) for l in check['epoch_tick_lines'])}]")
                else:
                    where.append("NEITHER path")
                print(f"  ✗ {check['call']:<40s}  {', '.join(where)}")
        # Non-epoch calls: presence-only check (different cadence)
        non_epoch = dual.get("non_epoch_checks", [])
        if non_epoch:
            print("    -- Non-epoch calls (presence only, NOT dual-path) --")
            for check in non_epoch:
                if check["present"]:
                    print(f"  ✓ {check['call']:<40s}  L{check['line']}  ({check['cadence']})")
                else:
                    g_count += 1
                    print(f"  ✗ {check['call']:<40s}  NOT FOUND in file  ({check['cadence']})")
    else:
        print("  ? spirit_worker.py not found — skipping dual-path check")

    # Summary
    total = a_count + b_count + c_count + d_count + f_count + g_count
    print(f"\n{'=' * 95}")
    print(f"  A. No consumer:      {a_count:3d}  (signals going nowhere)")
    print(f"  B. Deaf ears:        {b_count:3d}  (consumers with no sender)")
    print(f"  C. Dead letters:     {c_count:3d}  (targeted to missing subscriber)")
    print(f"  D. Dead types:       {d_count:3d}  (defined but unused)")
    print(f"  E. Idle reply-only:  {e_count:3d}  (likely proxy — verify manually)")
    print(f"  F. Scope leaks:      {f_count:3d}  (standalone func using enclosing locals)")
    print(f"  G. Dual-path gaps:   {g_count:3d}  (missing from one epoch path)")
    print(f"  TOTAL ISSUES:        {total:3d}")


# ── Scope & Dual-Path Checks (integrated into audit) ─────────────────

def _audit_scope_leaks():
    """F. SCOPE LEAKS — detect variables used in standalone functions that
    reference enclosing-scope locals (most commonly _handle_query using
    spirit_worker_main locals without passing them as parameters)."""
    results = []
    # Files with known scope-sensitive standalone functions
    targets = [
        ("titan_plugin/modules/spirit_loop.py", "_handle_query"),
        ("titan_plugin/modules/spirit_worker.py", "_handle_query"),
        ("titan_plugin/modules/rl_worker.py", "_handle_query"),
        ("titan_plugin/modules/llm_worker.py", "_handle_query"),
        ("titan_plugin/modules/mind_worker.py", "_handle_query"),
        ("titan_plugin/modules/memory_worker.py", "_handle_query"),
    ]

    # Common builtins/globals that are NOT scope leaks
    # 2026-04-08 audit: added missing builtins (__import__, sum, divmod, pow,
    #   chr, ord, format, bin, oct, hex, callable, vars, dir) — these were
    #   incorrectly flagged as scope leaks in memory_worker._handle_query
    safe_names = {
        "True", "False", "None", "self", "cls",
        "print", "len", "range", "int", "float", "str", "bool",
        "list", "dict", "tuple", "set", "min", "max", "abs", "sum",
        "round", "sorted", "enumerate", "zip", "map", "filter",
        "isinstance", "hasattr", "getattr", "setattr", "type", "super",
        "open", "Exception", "KeyError", "ValueError", "TypeError",
        "AttributeError", "RuntimeError", "ImportError", "StopIteration",
        "OSError", "IOError", "IndexError", "NotImplementedError",
        "any", "all", "id", "hash", "repr", "iter", "next",
        "staticmethod", "classmethod", "property",
        "logger", "logging", "json", "os", "sys", "re", "time",
        "math", "np", "asyncio", "traceback", "copy", "datetime",
        "Path", "defaultdict", "deque", "dataclass", "field",
        "__import__", "divmod", "pow", "chr", "ord", "format",
        "bin", "oct", "hex", "callable", "vars", "dir", "globals", "locals",
    }

    for filepath, func_name in targets:
        fpath = PROJECT_ROOT / filepath
        if not fpath.exists():
            continue
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            continue

        # Find the target function definition
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                # Get parameter names
                params = set()
                for arg in node.args.args:
                    params.add(arg.arg)
                for arg in node.args.posonlyargs:
                    params.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    params.add(arg.arg)
                if node.args.vararg:
                    params.add(node.args.vararg.arg)
                if node.args.kwarg:
                    params.add(node.args.kwarg.arg)

                # Get all names referenced in the function body
                body_names = set()
                local_assigns = set()
                import_names = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        body_names.add(child.id)
                    # Track local assignments (these are NOT leaks)
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        local_assigns.add(child.id)
                    # Track local imports
                    elif isinstance(child, ast.Import):
                        for alias in child.names:
                            import_names.add(alias.asname or alias.name.split(".")[-1])
                    elif isinstance(child, ast.ImportFrom):
                        for alias in child.names:
                            import_names.add(alias.asname or alias.name)
                    # Track for-loop variables
                    elif isinstance(child, ast.For):
                        if isinstance(child.target, ast.Name):
                            local_assigns.add(child.target.id)
                        elif isinstance(child.target, ast.Tuple):
                            for elt in child.target.elts:
                                if isinstance(elt, ast.Name):
                                    local_assigns.add(elt.id)
                    # Track with-as variables
                    elif isinstance(child, ast.With):
                        for item in child.items:
                            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                                local_assigns.add(item.optional_vars.id)
                    # Track except-as variables
                    elif isinstance(child, ast.ExceptHandler) and child.name:
                        local_assigns.add(child.name)

                # Compute potential leaks: referenced but not param, local, import, or safe
                potential_leaks = body_names - params - local_assigns - import_names - safe_names

                # Further filter: remove names that look like module-level constants (ALL_CAPS)
                # or well-known module names, or private helper functions defined at module level
                module_level_names = set()
                for top_node in ast.iter_child_nodes(tree):
                    if isinstance(top_node, ast.Assign):
                        for t in top_node.targets:
                            if isinstance(t, ast.Name):
                                module_level_names.add(t.id)
                    # 2026-04-19: Also catch annotated assignments (e.g.,
                    # `_COORD_SNAPSHOT_CACHE: dict = {"data": None, "ts": 0.0}`)
                    # — prior code missed `ast.AnnAssign` and falsely flagged
                    # module-level annotated caches as scope leaks.
                    elif isinstance(top_node, ast.AnnAssign):
                        if isinstance(top_node.target, ast.Name):
                            module_level_names.add(top_node.target.id)
                    elif isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        module_level_names.add(top_node.name)
                    elif isinstance(top_node, ast.ClassDef):
                        module_level_names.add(top_node.name)
                    elif isinstance(top_node, ast.Import):
                        for alias in top_node.names:
                            module_level_names.add(alias.asname or alias.name.split(".")[-1])
                    elif isinstance(top_node, ast.ImportFrom):
                        for alias in top_node.names:
                            module_level_names.add(alias.asname or alias.name)

                # A leak is something referenced but NOT a parameter, local assignment,
                # import, safe builtin, or module-level definition
                leaks = potential_leaks - module_level_names

                if leaks:
                    results.append({
                        "file": filepath,
                        "func": func_name,
                        "line": node.lineno,
                        "params": sorted(params),
                        "leaks": sorted(leaks),
                    })
                break  # Only check first definition per file

    return results


def _audit_dual_path():
    """G. DUAL-PATH COMPLETENESS — verify that key function calls appear in
    BOTH the periodic-tick path and the EPOCH_TICK handler in spirit_worker.py.

    Two call categories (2026-04-08 audit split):
    - dual_path_calls: per-epoch calls that MUST be in both paths
    - non_epoch_calls: calls that run at different cadences (Tier 2 FEELING,
      body tick) and should be checked for mere presence in the file, NOT
      for dual-path completeness. Previously these generated false positives.
    """
    fpath = PROJECT_ROOT / "titan_plugin" / "modules" / "spirit_worker.py"
    if not fpath.exists():
        return None

    source = fpath.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()

    # Per-epoch calls — MUST appear in both periodic and EPOCH_TICK paths
    dual_path_calls = [
        "pi_monitor.observe",
        "life_force_engine.evaluate",
    ]
    # Non-epoch calls — run at different cadences (Tier 2 FEELING loop,
    # body tick rate). Presence check only; NOT a dual-path violation.
    non_epoch_calls = [
        ("neural_nervous_system.evaluate", "Tier 2 FEELING loop"),
        ("ground_up_enricher.apply", "body tick (D3)"),
    ]
    critical_calls = dual_path_calls  # legacy name used below

    # Identify the two path regions by scanning for markers
    # Path 1: periodic tick — starts after "if consciousness and _t3_time_since >= EPOCH_FLOOR:"
    # Path 2: EPOCH_TICK handler — starts at 'elif msg_type == "EPOCH_TICK":'
    periodic_start = None
    epoch_tick_start = None

    for i, line in enumerate(lines):
        if "_t3_time_since >= EPOCH_FLOOR" in line and periodic_start is None:
            periodic_start = i
        if 'msg_type == "EPOCH_TICK"' in line or "msg_type == 'EPOCH_TICK'" in line:
            epoch_tick_start = i

    if periodic_start is None or epoch_tick_start is None:
        return {"error": "Could not find both path markers in spirit_worker.py",
                "periodic_found": periodic_start is not None,
                "epoch_tick_found": epoch_tick_start is not None}

    # Determine path boundaries (scan until dedent to same or lower level)
    def get_path_end(start_line):
        """Find approximate end of a code block (next elif/else at same indent or lower)."""
        if start_line >= len(lines):
            return len(lines)
        # Get the indentation level of the start line
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        for i in range(start_line + 5, min(start_line + 800, len(lines))):
            stripped = lines[i].lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            cur_indent = len(lines[i]) - len(stripped)
            # If we find a line at same or lower indent that's a new branch, stop
            if cur_indent <= start_indent and (stripped.startswith("elif ") or stripped.startswith("else:")):
                return i
            if cur_indent < start_indent:
                return i
        return min(start_line + 800, len(lines))

    periodic_end = get_path_end(periodic_start)
    epoch_tick_end = get_path_end(epoch_tick_start)

    periodic_text = "\n".join(lines[periodic_start:periodic_end])
    epoch_tick_text = "\n".join(lines[epoch_tick_start:epoch_tick_end])

    results = []
    for call in critical_calls:
        in_periodic = call in periodic_text
        in_epoch = call in epoch_tick_text

        # Also find exact line numbers
        periodic_lines = []
        epoch_lines = []
        for i in range(periodic_start, periodic_end):
            if call in lines[i]:
                periodic_lines.append(i + 1)  # 1-indexed
        for i in range(epoch_tick_start, epoch_tick_end):
            if call in lines[i]:
                epoch_lines.append(i + 1)

        results.append({
            "call": call,
            "in_periodic": in_periodic,
            "in_epoch_tick": in_epoch,
            "periodic_lines": periodic_lines,
            "epoch_tick_lines": epoch_lines,
            "status": "OK" if (in_periodic and in_epoch) else "MISSING",
        })

    # Non-epoch calls: presence check only (find first call site anywhere in file)
    presence_results = []
    for call, cadence in non_epoch_calls:
        call_line = None
        for i, line in enumerate(lines):
            if call in line:
                call_line = i + 1
                break
        presence_results.append({
            "call": call,
            "cadence": cadence,
            "present": call_line is not None,
            "line": call_line,
        })

    return {
        "periodic_range": f"L{periodic_start+1}-L{periodic_end}",
        "epoch_tick_range": f"L{epoch_tick_start+1}-L{epoch_tick_end}",
        "checks": results,
        "non_epoch_checks": presence_results,
    }


# ── params command ────────────────────────────────────────────────────

def show_params(graph: dict):
    """Audit titan_params.toml — find keys that are defined but never referenced in code."""
    import tomllib

    params_file = PROJECT_ROOT / "titan_plugin" / "titan_params.toml"
    if not params_file.exists():
        print(f"ERROR: {params_file} not found")
        return

    with open(params_file, "rb") as f:
        params = tomllib.load(f)

    # Flatten all keys (leaf values only)
    all_keys = {}  # key_name -> (section_path, value)

    def flatten(d, prefix=""):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, path)
            else:
                all_keys[k] = (path, v)

    flatten(params)

    # Collect all Python source text for searching
    source_cache = {}
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            for fpath in scan_dir.rglob("*.py"):
                try:
                    source_cache[rel(fpath)] = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
    for scan_dir in (SCRIPTS_DIRS or []):
        if scan_dir.exists():
            for fpath in scan_dir.rglob("*.py"):
                try:
                    source_cache[rel(fpath)] = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    combined_source = "\n".join(source_cache.values())

    # Search patterns for each key
    referenced = {}
    unreferenced = {}

    for key_name, (section_path, value) in sorted(all_keys.items()):
        # Check multiple access patterns:
        # 1. .get("key_name")  2. ["key_name"]  3. key_name as a bare identifier
        # 4. "key_name" as string literal  5. _dna["key_name"]  6. cfg["key_name"]
        patterns = [
            f'"{key_name}"',
            f"'{key_name}'",
        ]
        found = False
        for pat in patterns:
            if pat in combined_source:
                found = True
                break

        if found:
            referenced[key_name] = section_path
        else:
            unreferenced[key_name] = (section_path, value)

    # Display results
    print("=" * 80)
    print(f"TITAN PARAMS AUDIT — {len(all_keys)} keys in titan_params.toml")
    print("=" * 80)

    if unreferenced:
        print(f"\n  UNREFERENCED KEYS ({len(unreferenced)}):")
        print(f"  (defined in TOML but no string literal \"{{}}\"/'{{}}'  found in Python source)")
        # Group by section
        by_section = defaultdict(list)
        for key_name, (section_path, value) in sorted(unreferenced.items()):
            section = section_path.rsplit(".", 1)[0] if "." in section_path else "(root)"
            by_section[section].append((key_name, value))

        for section in sorted(by_section.keys()):
            print(f"\n    [{section}]")
            for key_name, value in by_section[section]:
                val_str = repr(value) if not isinstance(value, (int, float)) else str(value)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                print(f"      {key_name:<40s} = {val_str}")
    else:
        print("\n  All keys are referenced in code.")

    print(f"\n  SUMMARY: {len(referenced)} referenced, {len(unreferenced)} unreferenced out of {len(all_keys)} total")


# ── worker command ────────────────────────────────────────────────────

def query_worker(graph: dict, worker_name: str):
    """Per-module wiring card — show all bus sends, publishes, consumes, subscribers,
    key function calls, and files that import from this worker."""
    files = graph.get("files", {})
    wiring = graph.get("bus_wiring", {})

    # Find matching files
    matched_files = {}
    for fp, data in files.items():
        # Match by worker name in file path (e.g. "spirit" matches spirit_worker.py, spirit_loop.py)
        basename = fp.split("/")[-1].replace(".py", "")
        if worker_name.lower() in basename.lower() or worker_name.lower() in fp.lower():
            matched_files[fp] = data

    if not matched_files:
        print(f"No files matching worker '{worker_name}'")
        print(f"\nAvailable files:")
        for fp in sorted(files.keys()):
            print(f"  {fp}")
        return

    print("=" * 80)
    print(f"WORKER CARD: '{worker_name}' ({len(matched_files)} files)")
    print("=" * 80)

    all_sends = []
    all_publishes = []
    all_consumers = []
    all_subscribes = []
    all_calls = []

    for fp, data in sorted(matched_files.items()):
        print(f"\n  File: {fp} ({data.get('lines', '?')} lines)")

        # send_msg calls
        sends = data.get("send_msgs", [])
        for s in sends:
            all_sends.append({**s, "file": fp})

        # publishes
        pubs = data.get("bus_publishes", [])
        for p in pubs:
            all_publishes.append({**p, "file": fp})

        # subscribers
        subs = data.get("bus_subscribes", [])
        for s in subs:
            all_subscribes.append({**s, "file": fp})

        # consumers
        cons = data.get("msg_consumers", [])
        for c in cons:
            all_consumers.append({**c, "file": fp})

        # key function calls (top 30, deduplicated by name)
        calls = data.get("function_calls", [])
        for c in calls:
            all_calls.append({**c, "file": fp})

    # Display sends
    if all_sends:
        # Deduplicate by msg_type
        send_types = defaultdict(list)
        for s in all_sends:
            send_types[s["msg_type"]].append(s)
        print(f"\n  SENDS ({len(all_sends)} calls, {len(send_types)} message types):")
        for mtype, items in sorted(send_types.items()):
            dsts = sorted(set(s.get("dst", "?") for s in items))
            print(f"    {mtype:<30s} -> dst=[{', '.join(dsts)}]  ({len(items)}x)")
    else:
        print("\n  SENDS: none")

    # Display publishes
    if all_publishes:
        pub_types = defaultdict(list)
        for p in all_publishes:
            pub_types[p["msg_type"]].append(p)
        print(f"\n  PUBLISHES ({len(all_publishes)} calls, {len(pub_types)} message types):")
        for mtype, items in sorted(pub_types.items()):
            print(f"    {mtype:<30s} ({len(items)}x)")
            for item in items[:3]:
                print(f"      {item['file']}:{item['line']}  [{item['context']}]")
            if len(items) > 3:
                print(f"      ... and {len(items)-3} more")
    else:
        print("\n  PUBLISHES: none")

    # Display consumers
    if all_consumers:
        con_types = defaultdict(list)
        for c in all_consumers:
            con_types[c["msg_type"]].append(c)
        print(f"\n  CONSUMES ({len(all_consumers)} checks, {len(con_types)} message types):")
        for mtype, items in sorted(con_types.items()):
            print(f"    {mtype:<30s} ({len(items)}x)")
            for item in items[:2]:
                print(f"      {item['file']}:{item['line']}  [{item['context']}]")
    else:
        print("\n  CONSUMES: none")

    # Display subscribers
    if all_subscribes:
        print(f"\n  SUBSCRIBERS ({len(all_subscribes)}):")
        for s in all_subscribes:
            via = f"  (via {s['via']})" if s.get("via") else ""
            print(f"    '{s['name']}'  reply_only={s['reply_only']}  {s['file']}:{s['line']}{via}")
    else:
        print("\n  SUBSCRIBERS: none")

    # Display key function calls (deduplicated, sorted by frequency)
    if all_calls:
        call_freq = defaultdict(int)
        for c in all_calls:
            call_freq[c["func"]] += 1
        top_calls = sorted(call_freq.items(), key=lambda x: -x[1])[:30]
        print(f"\n  KEY FUNCTION CALLS (top {min(30, len(top_calls))}):")
        for func, count in top_calls:
            print(f"    {func:<45s} {count}x")
    else:
        print("\n  KEY FUNCTION CALLS: none")

    # Files that import from this worker
    import_graph = graph.get("import_graph", {})
    matched_modules = set()
    for fp in matched_files:
        mod = fp.replace("/", ".").replace(".py", "")
        matched_modules.add(mod)
        parts = mod.split(".")
        for i in range(len(parts)):
            matched_modules.add(".".join(parts[i:]))

    importers = []
    for fp, imports in import_graph.items():
        if fp in matched_files:
            continue  # Skip self
        for imp in imports:
            imp_parts = imp.split(".")
            for tm in matched_modules:
                tm_parts = tm.split(".")
                if imp_parts[-len(tm_parts):] == tm_parts or (len(tm_parts) >= 1 and tm_parts[-1] in imp_parts):
                    importers.append((fp, imp))
                    break

    if importers:
        print(f"\n  IMPORTED BY ({len(importers)} files):")
        for fp, imp in sorted(set(importers)):
            print(f"    {fp}  (imports {imp})")
    else:
        print("\n  IMPORTED BY: none")


# ── flow command ──────────────────────────────────────────────────────

def query_flow(graph: dict, msg_type: str):
    """End-to-end message journey — trace who sends, routes, receives, and processes a message."""
    wiring = graph.get("bus_wiring", {})
    files = graph.get("files", {})

    # Find matching message types
    matches = {}
    for key, val in wiring.items():
        if key.startswith("_SUB:"):
            continue
        if msg_type.upper() in key.upper():
            matches[key] = val

    if not matches:
        print(f"No bus messages matching '{msg_type}'")
        types = [k for k in wiring if not k.startswith("_SUB:")]
        print(f"\nAvailable message types ({len(types)}):")
        for t in sorted(types)[:30]:
            print(f"  {t}")
        return

    all_sub_names = {k.replace("_SUB:", ""): v for k, v in wiring.items() if k.startswith("_SUB:")}

    for mtype, data in sorted(matches.items()):
        print("=" * 80)
        print(f"MESSAGE FLOW: {mtype}")
        print("=" * 80)

        pubs = data.get("publishers", [])
        sends = data.get("send_msgs", [])
        consumers = data.get("consumers", [])

        # Step 1: Who sends it
        print(f"\n  1. ORIGIN (who sends/publishes):")
        if sends:
            for s in sends:
                print(f"     send_msg  {s['file']}:{s['line']}  -> dst={s['dst']}  [{s['context']}]")
        if pubs:
            for p in pubs:
                dst = p.get("dst", "broadcast")
                src = p.get("src", "?")
                print(f"     publish   {p['file']}:{p['line']}  src={src} dst={dst}  [{p['context']}]")
        if not sends and not pubs:
            print("     (no senders found)")

        # Step 2: Routing — what dst it targets
        all_dsts = set()
        for s in sends:
            if s.get("dst") and s["dst"] not in ("?",):
                all_dsts.add(s["dst"])
        for p in pubs:
            dst = p.get("dst", "")
            if dst and dst not in ("?",):
                all_dsts.add(dst)

        print(f"\n  2. ROUTING:")
        if "all" in all_dsts:
            print(f"     BROADCAST (dst=all) — delivered to ALL non-reply_only subscribers")
            # List all non-reply-only subscribers
            for sub_name, sub_data in sorted(all_sub_names.items()):
                for s in sub_data.get("subscribers", []):
                    if not s.get("reply_only", False):
                        print(f"       -> '{sub_name}'  {s['file']}:{s['line']}")
        for dst in sorted(all_dsts - {"all"}):
            print(f"     TARGETED -> dst='{dst}'")
            if dst in all_sub_names:
                for s in all_sub_names[dst].get("subscribers", []):
                    print(f"       subscriber '{dst}' reply_only={s.get('reply_only', False)}  {s['file']}:{s['line']}")
            else:
                print(f"       WARNING: No subscriber named '{dst}' found!")
        if not all_dsts:
            print("     (no routing information — check make_msg dst parameter)")

        # Step 3: Who consumes it
        print(f"\n  3. CONSUMER (msg_type == '{mtype}' checks):")
        if consumers:
            for c in consumers:
                print(f"     {c['file']}:{c['line']}  [{c['context']}]")
        else:
            print("     (no consumer code found — message may be unhandled)")

        # Step 4: What the consumer does (function calls in same context)
        if consumers:
            print(f"\n  4. CONSUMER ACTIONS (function calls in consumer contexts):")
            consumer_contexts = set()
            consumer_files = set()
            for c in consumers:
                consumer_contexts.add(c["context"])
                consumer_files.add(c["file"])

            actions_found = False
            for fp in consumer_files:
                file_data = files.get(fp, {})
                file_calls = file_data.get("function_calls", [])
                for call in file_calls:
                    # Match calls within consumer contexts (same function)
                    if call["context"] in consumer_contexts:
                        if not actions_found:
                            actions_found = True
                        print(f"     {call['func']:<45s}  {fp}:{call['line']}")

            if not actions_found:
                print("     (no function calls detected in consumer context)")

        print()


# ── LIVE Health Check ─────────────────────────────────────────────────

def _health_get(base_url: str, path: str, timeout: float = 20.0, retries: int = 1) -> dict | None:
    """GET a JSON endpoint, return parsed dict or None on failure.

    Retries once on transient failure (timeout / non-200 / exception) before
    giving up. Absorbs network blips on shared-VPS T2/T3 endpoints so that
    health checks do not report `0 transitions` when the cause is a momentary
    HTTP error rather than a genuinely-stalled subsystem.
    """
    import requests
    import time as _time
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{base_url}{path}", timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        if attempt < retries:
            _time.sleep(0.5)
    return None


def _unwrap(resp: dict | None) -> dict:
    """Unwrap {"status": "ok", "data": {...}} → data dict."""
    if not resp:
        return {}
    d = resp.get("data", resp)
    return d if isinstance(d, dict) else {}


def run_health_checks(base_url: str = "http://127.0.0.1:7777", label: str = "T1 (localhost:7777)"):
    """Run all live health checks against a Titan API and print results."""
    import time
    import subprocess
    import sqlite3

    results = []  # list of (pass/warn/fail, message)

    def ok(msg):
        results.append(("pass", msg))

    def warn(msg):
        results.append(("warn", msg))

    def fail(msg):
        results.append(("fail", msg))

    print(f"\nTITAN LIVE HEALTH CHECK — {label}")
    print("=" * 56)

    # ── 1. API reachable ──────────────────────────────────────────────
    # /health is the bootstrap — if it fails, the entire check aborts. Use
    # extra retries to absorb event-loop-blocking long bio-layer ticks
    # (3+ second ticks observed on T1 under shared-uvicorn legacy mode;
    # S5 api_subprocess separation is the structural fix). Worst case:
    # 4 attempts × 20s = 80s, but in practice the API frees up between
    # blocking ticks so we typically succeed by attempt 2.
    health_resp = _health_get(base_url, "/health", retries=3)
    if health_resp and health_resp.get("status") == "ok":
        ok("API reachable (200 OK)")
    else:
        fail("API unreachable — cannot reach /health")
        # If API is down, skip everything else
        _print_health_results(results)
        return

    health_data = _unwrap(health_resp)

    # ── Fetch first snapshots for growth checks ───────────────────────
    # Track raw fetch results so we can distinguish "API fetch failed" (None)
    # from "API returned 0" — both used to look identical to downstream code,
    # producing false `0 transitions` reports during transient HTTP blips on
    # shared-VPS T2/T3 endpoints.
    print("  Waiting 10s to measure consciousness growth...")
    ns1_raw = _health_get(base_url, "/v4/nervous-system")
    trinity1_raw = _health_get(base_url, "/v4/inner-trinity")
    ns1_resp = _unwrap(ns1_raw)
    trinity1_resp = _unwrap(trinity1_raw)
    ns1_transitions = ns1_resp.get("total_transitions", 0)
    epoch1 = trinity1_resp.get("pi_heartbeat", {}).get("total_epochs_observed", 0)

    time.sleep(10)

    # ── Fetch second snapshots ────────────────────────────────────────
    ns2_raw = _health_get(base_url, "/v4/nervous-system")
    trinity2_raw = _health_get(base_url, "/v4/inner-trinity")
    ns2_resp = _unwrap(ns2_raw)
    trinity2_resp = _unwrap(trinity2_raw)
    ns2_transitions = ns2_resp.get("total_transitions", 0)
    last_train_ts = ns2_resp.get("last_train_ts", None)
    epoch2 = trinity2_resp.get("pi_heartbeat", {}).get("total_epochs_observed", 0)

    ns_fetch_failed = ns1_raw is None or ns2_raw is None
    trinity_fetch_failed = trinity1_raw is None or trinity2_raw is None

    # ── 2. NS training active (age-based — no 10s window blip) ────────
    # Switched from 10s delta sampling to `now - last_train_ts` age check.
    # The delta approach gave false zeros when (a) any of the two sample
    # fetches transiently failed, or (b) the API snapshot cache TTL exceeded
    # the 10s window. last_train_ts is updated by the trainer on every step,
    # so age is the single source of truth.
    if ns_fetch_failed:
        warn("NS training — /v4/nervous-system fetch failed (transient HTTP error — re-run check)")
    elif ns2_transitions == 0 and last_train_ts in (None, 0, 0.0):
        fail("NS training not running (0 transitions, no last_train_ts)")
    elif last_train_ts in (None, 0, 0.0):
        # Field missing — fall back to delta sampling (older API contract)
        ns_delta = ns2_transitions - ns1_transitions
        if ns_delta > 0:
            ok(f"NS training growing ({ns1_transitions} -> {ns2_transitions}, +{ns_delta} in 10s)")
        else:
            warn(f"NS training stalled ({ns2_transitions} transitions, +0 in 10s, no last_train_ts)")
    else:
        age_s = time.time() - last_train_ts
        if age_s < 60:
            ok(f"NS training active ({ns2_transitions} transitions, last train {age_s:.0f}s ago)")
        elif age_s < 300:
            warn(f"NS training slow ({ns2_transitions} transitions, last train {age_s:.0f}s ago)")
        else:
            fail(f"NS training stalled ({ns2_transitions} transitions, last train {age_s/60:.0f}m ago)")

    # ── 2b. NS program feature_set vs input_dim consistency ──────────
    _dim_map = {"core": 30, "standard": 55, "extended": 75, "full": 88, "enriched": 79, "full_enriched": 112}
    ns_programs = ns2_resp.get("programs", {})
    _dim_mismatches = []
    for _pname, _pinfo in ns_programs.items():
        _fs = _pinfo.get("feature_set", "standard")
        _expected = _dim_map.get(_fs, 55)
        _actual = _pinfo.get("input_dim", 0)
        if _expected != _actual:
            _dim_mismatches.append(f"{_pname}: {_fs}={_expected}D but net={_actual}D")
    if not _dim_mismatches:
        ok(f"NS dimensions consistent ({len(ns_programs)} programs, feature_set matches input_dim)")
    else:
        fail(f"NS dimension MISMATCH (silent training failure): {', '.join(_dim_mismatches)}")

    # ── 3. Consciousness epochs advancing (retry on suspect stall) ────
    # A 10s window can catch a brief pause (module subprocess restart, dream
    # transition, API snapshot cache TTL > sample interval) and produce a
    # false "stalled" warning. When the first window shows zero growth on a
    # non-zero counter, retry once with a longer window before flagging —
    # only suspect cases pay the extra time, healthy Titans pass on first
    # window.
    if trinity_fetch_failed:
        warn("Consciousness — /v4/inner-trinity fetch failed (transient HTTP error — re-run check)")
    else:
        epoch_delta = epoch2 - epoch1
        if epoch_delta > 0:
            ok(f"Consciousness advancing (epoch {epoch1} -> {epoch2})")
        elif epoch2 > 0:
            print("  Stall suspected — retrying with 15s window to confirm...")
            time.sleep(15)
            trinity3_raw = _health_get(base_url, "/v4/inner-trinity")
            if trinity3_raw is None:
                warn(f"Consciousness stalled (epoch {epoch2}, +0 in 10s; retry fetch failed)")
            else:
                epoch3 = _unwrap(trinity3_raw).get("pi_heartbeat", {}).get("total_epochs_observed", 0)
                if epoch3 > epoch2:
                    ok(f"Consciousness advancing (epoch {epoch1} -> {epoch3} confirmed after retry)")
                else:
                    warn(f"Consciousness stalled (epoch {epoch3}, +0 in 25s confirmed)")
        else:
            fail("Consciousness not running (0 epochs)")

    # ── 4. Neuromod homeostasis ───────────────────────────────────────
    nm_resp = _unwrap(_health_get(base_url, "/v4/neuromodulators"))
    modulators = nm_resp.get("modulators", {})
    if modulators:
        saturated = []
        levels_str = []
        for name, info in modulators.items():
            lvl = info.get("level", 0.5)
            levels_str.append(f"{name}={lvl:.2f}")
            if lvl < 0.05 or lvl > 0.95:
                saturated.append(f"{name}={lvl:.2f}")
        if not saturated:
            # Show top 3 for brevity
            display = " ".join(levels_str[:3]) + " -- all in range"
            ok(f"Neuromod homeostasis ({display})")
        else:
            fail(f"Neuromod saturated: {', '.join(saturated)}")
    else:
        fail("Neuromod data unavailable")

    # ── 5. GABA not stuck ─────────────────────────────────────────────
    gaba_info = modulators.get("GABA", {})
    gaba_level = gaba_info.get("level", 0.0)
    if gaba_level > 0.15:
        ok(f"GABA healthy ({gaba_level:.2f})")
    elif gaba_level > 0.05:
        warn(f"GABA low ({gaba_level:.2f} < 0.15 -- below expected range)")
    else:
        fail(f"GABA suppressed ({gaba_level:.2f} <= 0.05)")

    # ── 6. Expression composites firing ───────────────────────────────
    # Use trinity2 which we already have
    expr = trinity2_resp.get("expression_composites", {})
    composites = expr.get("composites", {})
    if composites:
        fire_strs = []
        any_firing = False
        for cname, cinfo in composites.items():
            fc = cinfo.get("fire_count", 0)
            fire_strs.append(f"{cname}={fc}")
            if fc > 0:
                any_firing = True
        display = ", ".join(fire_strs[:3])
        if any_firing:
            ok(f"Expressions firing ({display})")
        else:
            warn(f"Expressions not firing ({display})")
    else:
        fail("Expression composite data unavailable")

    # ── 7. Chi healthy ────────────────────────────────────────────────
    chi_resp = _unwrap(_health_get(base_url, "/v4/chi"))
    chi_total = chi_resp.get("total", 0.0)
    if isinstance(chi_total, (int, float)) and chi_total > 0.1:
        ok(f"Chi healthy (total={chi_total:.3f})")
    elif isinstance(chi_total, (int, float)) and chi_total > 0:
        warn(f"Chi low (total={chi_total:.3f})")
    else:
        fail(f"Chi crashed or unavailable (total={chi_total})")

    # ── 8. No Guardian module crashes ─────────────────────────────────
    # 2026-04-08 (later) audit fix: replaced "still booting (likely)" warning
    # which masked real crash loops on T3 (cgn/knowledge crash-looping for 1h+
    # were reported as "still booting"). New logic uses Guardian's restart_count
    # and uptime to distinguish three cases:
    #   1. STARTING + low restart_count → genuine boot, WAIT and re-check
    #   2. DISABLED with restart_count ≥ 5 → CRASH LOOP, FAIL with diagnosis
    #   3. DISABLED with low restart count → cooldown after few crashes, WARN
    # Heavy boot modules have larger boot windows but the same crash detection.
    HEAVY_BOOT_MODULES = {"cgn", "knowledge", "memory"}
    HEAVY_BOOT_GRACE_S = 600     # 10 min for heavy modules to come online
    LIGHT_BOOT_GRACE_S = 60      # 1 min for light modules
    CRASH_LOOP_THRESHOLD = 5     # restart_count ≥ 5 = definite crash loop

    v3_data = health_data.get("v3", {})
    guardian_status = v3_data.get("guardian_status", {})
    if guardian_status:
        active = 0
        starting = []          # genuinely booting
        crash_loops = []       # disabled with high restart_count
        cooldown = []          # disabled with low restart_count (transient)
        unhealthy = []         # heartbeat-overdue but not yet disabled
        recently_unstable = [] # running but with recent restarts
        total = 0
        for mod_name, mod_info in guardian_status.items():
            state = mod_info.get("state", "unknown")
            # rl and llm are intentionally stopped — skip them
            if state == "stopped" and mod_name in ("rl", "llm"):
                continue
            total += 1
            restart_count = mod_info.get("restart_count", 0)
            uptime = mod_info.get("uptime", 0)
            grace = HEAVY_BOOT_GRACE_S if mod_name in HEAVY_BOOT_MODULES else LIGHT_BOOT_GRACE_S

            if state == "running":
                active += 1
                # Recent restarts on a running module = warning sign
                if restart_count >= 3:
                    recently_unstable.append(f"{mod_name}(restarts={restart_count})")
            elif state == "starting":
                # Distinguish: genuine boot vs stuck-in-starting
                if uptime < grace:
                    starting.append(mod_name)
                else:
                    # Stuck in STARTING for longer than grace = crash loop pattern
                    crash_loops.append(f"{mod_name}(stuck_starting={uptime:.0f}s)")
            elif state == "disabled":
                if restart_count >= CRASH_LOOP_THRESHOLD:
                    crash_loops.append(f"{mod_name}(restarts={restart_count})")
                else:
                    cooldown.append(f"{mod_name}(restarts={restart_count})")
            elif state == "unhealthy":
                unhealthy.append(mod_name)

        # Decide overall status (worst-case wins)
        if crash_loops:
            fail(f"CRASH LOOP detected: {', '.join(crash_loops)} "
                 f"({active}/{total} running) — check logs for stop reason "
                 f"(rss limit, heartbeat timeout, exception)")
        elif cooldown:
            fail(f"Modules DISABLED in cooldown: {', '.join(cooldown)} "
                 f"({active}/{total} running) — Guardian will retry, but recent "
                 f"crashes need investigation")
        elif unhealthy:
            warn(f"Modules UNHEALTHY: {', '.join(unhealthy)} "
                 f"({active}/{total} running) — heartbeat overdue but not yet disabled")
        elif starting:
            warn(f"Modules booting: {', '.join(starting)} ({active}/{total} running) — "
                 f"under grace period, re-run in 30s to verify they reach RUNNING")
        elif recently_unstable:
            warn(f"Modules running but unstable: {', '.join(recently_unstable)} "
                 f"({active}/{total} running) — recent restarts, monitor for sustained uptime")
        else:
            ok(f"No module crashes ({active}/{total} active)")
    else:
        warn("Guardian status not available in /health response")

    # ── 9. Dreaming system tracking ───────────────────────────────────
    dreaming = trinity2_resp.get("dreaming", {})
    epochs_since = dreaming.get("epochs_since_dream", 0)
    cycle_count = dreaming.get("cycle_count", 0)
    if epochs_since > 0 or cycle_count > 0:
        ok(f"Dreaming tracking (epochs_since={epochs_since}, cycle={cycle_count})")
    else:
        warn("Dreaming system not tracking (epochs_since_dream=0, cycle=0)")

    # ── 10. Sleep/wake drives present ─────────────────────────────────
    sleep_drive = dreaming.get("last_sleep_drive")
    wake_drive = dreaming.get("last_wake_drive")
    if sleep_drive is not None and wake_drive is not None:
        ok(f"Sleep/wake drives active (sleep={sleep_drive:.2f}, wake={wake_drive:.2f})")
    else:
        # The emergent dreaming system exposes these via DreamingEngine.get_stats()
        # but they may not be in the coordinator response yet
        warn("Sleep/wake drives not exposed in API (emergent dreaming may need wiring)")

    # ── 11. Experience records growing ────────────────────────────────
    try:
        db_path = str(PROJECT_ROOT / "data" / "experience_orchestrator.db")
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        total_records = conn.execute("SELECT COUNT(*) FROM experience_records").fetchone()[0]
        undistilled = conn.execute(
            "SELECT COUNT(*) FROM experience_records WHERE distilled=0").fetchone()[0]
        conn.close()
        if total_records > 0:
            ok(f"Experience records ({total_records} total, {undistilled} undistilled)")
        else:
            warn("Experience records empty (0 total)")
    except Exception as e:
        warn(f"Experience DB not accessible ({e})")

    # ── 12. Bus queue healthy ─────────────────────────────────────────
    try:
        log_path = "/tmp/titan_brain.log"
        result = subprocess.run(
            ["tail", "-100", log_path],
            capture_output=True, text=True, timeout=5)
        log_tail = result.stdout
        queue_drops = log_tail.count("Queue full")
        if queue_drops == 0:
            ok(f"Bus queue healthy (0 drops in last 100 lines)")
        else:
            warn(f"Bus queue drops detected ({queue_drops} in last 100 lines)")
    except FileNotFoundError:
        warn("Brain log not found at /tmp/titan_brain.log")
    except Exception as e:
        warn(f"Could not check bus queue ({e})")

    # ── 13. Reasoning engine health ────────────────────────────────────
    reasoning = trinity2_resp.get("reasoning", {})
    if reasoning:
        total_chains = reasoning.get("total_chains", 0)
        total_conclusions = reasoning.get("total_conclusions", 0)
        total_steps = reasoning.get("total_reasoning_steps", 0)
        buffer_size = reasoning.get("buffer_size", 0)
        policy_updates = reasoning.get("policy_updates", 0)
        commit_rate = (total_conclusions / total_chains * 100) if total_chains > 0 else 0
        avg_chain = (total_steps / total_chains) if total_chains > 0 else 0

        if total_chains > 0 and commit_rate > 10:
            ok(f"Reasoning active (chains={total_chains}, commits={total_conclusions}, "
               f"rate={commit_rate:.0f}%, avg_len={avg_chain:.1f}, buffer={buffer_size})")
        elif total_chains > 0:
            warn(f"Reasoning low commit rate (chains={total_chains}, commits={total_conclusions}, "
                 f"rate={commit_rate:.0f}%, avg_len={avg_chain:.1f})")
        else:
            warn("Reasoning engine idle (0 chains since boot)")
    else:
        warn("Reasoning engine not exposed in API")

    # ── 14. Dreaming stuck detection ────────────────────────────────────
    is_dreaming = dreaming.get("is_dreaming", False)
    if is_dreaming:
        # Check if dream duration is excessive (> 300s suggests stuck)
        dream_epochs = dreaming.get("dream_epochs", 0)
        est_dream_s = dream_epochs * 7  # ~7s per epoch
        if est_dream_s > 300:
            fail(f"DREAMING STUCK — estimated {est_dream_s:.0f}s (>300s), {dream_epochs} dream epochs")
        else:
            ok(f"Dreaming (in progress, ~{est_dream_s:.0f}s, {dream_epochs} epochs)")
    else:
        ok("Awake (not dreaming)")

    # ── 14b. Dream distillation foundational-loop health ────────────────
    # Codified rule (feedback_arch_map_monitoring_rule.md, 2026-04-13):
    # whenever new wiring is added, add the corresponding health check.
    # This catches the silent-disconnect failure mode that kept dream
    # distillation broken for 27 days (2026-03-17 → 2026-04-13).
    cycle_count = dreaming.get("cycle_count", 0)
    distill_attempts = dreaming.get("distill_attempts", 0)
    distill_passed = dreaming.get("distill_passed", 0)
    distilled_count = dreaming.get("distilled_count", 0)
    exp_buffer_size = dreaming.get("experience_buffer_size", 0)
    if cycle_count == 0:
        # No dreams yet — too early to assess
        ok(f"Dream distillation: {cycle_count} cycles (too early to assess)")
    elif distill_attempts == 0 and cycle_count > 3:
        # Dreams completed but distillation never invoked → wiring broken
        fail(f"DISTILLATION DISCONNECTED — {cycle_count} cycles done but "
             f"distill_attempts=0. Buffer not feeding distillation. "
             f"experience_buffer_size={exp_buffer_size}")
    elif distill_attempts > 0 and distill_passed == 0 and cycle_count > 5:
        # Distillation ran but produced nothing → threshold may be too high
        warn(f"Dream distillation runs but 0 insights — threshold may be "
             f"too high (attempts={distill_attempts}, threshold={dreaming.get('distill_threshold')})")
    elif distill_attempts > 0:
        rate = (distill_passed / distill_attempts) * 100 if distill_attempts else 0
        ok(f"Dream distillation healthy ({distill_passed}/{distill_attempts} "
           f"snapshots → {distilled_count} insights, pass_rate={rate:.0f}%)")
    else:
        ok(f"Dream distillation pending (cycles={cycle_count}, attempts={distill_attempts})")

    # ── 15. π-Heartbeat health (curvature rate + cluster growth) ────────
    pi_data = trinity2_resp.get("pi_heartbeat", {})
    if pi_data:
        pi_rate = pi_data.get("heartbeat_ratio", 0)
        pi_clusters = pi_data.get("cluster_count", 0)
        pi_total = pi_data.get("total_epochs_observed", 0)
        pi_total_events = pi_data.get("total_pi_epochs", 0)
        in_cluster = pi_data.get("in_cluster", False)
        avg_cluster_size = pi_data.get("avg_cluster_size", 0)

        if pi_rate >= 0.03:  # 3%+ is healthy
            ok(f"π-heartbeat healthy (rate={pi_rate*100:.1f}%, clusters={pi_clusters}, "
               f"events={pi_total_events}/{pi_total}, avg_size={avg_cluster_size:.1f}"
               f"{', IN CLUSTER' if in_cluster else ''})")
        elif pi_rate >= 0.01:
            warn(f"π-heartbeat low (rate={pi_rate*100:.1f}% < 3%, clusters={pi_clusters} "
                 f"— curvature declining, monitor for 5-HT/dreaming impact)")
        else:
            fail(f"π-heartbeat CRITICAL (rate={pi_rate*100:.2f}% — near-zero curvature, "
                 f"6 critical systems may degrade)")
    else:
        warn("π-heartbeat data not available in API")

    # ── 16. DivineBus health (timeouts + queue drops) ────────────────
    try:
        log_path = "/tmp/titan_brain.log"
        result = subprocess.run(
            ["grep", "-ac", "Request timed out", log_path],
            capture_output=True, text=True, timeout=5)
        total_timeouts = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0

        result2 = subprocess.run(
            ["grep", "-ac", "Queue full", log_path],
            capture_output=True, text=True, timeout=5)
        total_drops = int(result2.stdout.strip()) if result2.stdout.strip().isdigit() else 0

        # Get recent timeout rate (last 200 lines)
        result3 = subprocess.run(
            ["tail", "-200", log_path],
            capture_output=True, text=True, timeout=5)
        recent_timeouts = result3.stdout.count("Request timed out")
        recent_drops = result3.stdout.count("Queue full")

        if recent_drops == 0 and recent_timeouts < 10:
            ok(f"DivineBus healthy (recent: {recent_timeouts} timeouts, {recent_drops} drops | "
               f"total: {total_timeouts} timeouts, {total_drops} drops)")
        elif recent_drops > 0:
            fail(f"DivineBus QUEUE DROPS ({recent_drops} recent, {total_drops} total) "
                 f"— messages being lost!")
        else:
            warn(f"DivineBus elevated timeouts ({recent_timeouts} recent, "
                 f"{total_timeouts} total — bus congestion)")
    except Exception as e:
        warn(f"DivineBus health check failed ({e})")

    # ── 17. Nginx traffic health ─────────────────────────────────────────
    try:
        _nginx_stats = _check_nginx_health()
        _req_rate = _nginx_stats.get("req_per_min", 0)
        _err_rate = _nginx_stats.get("error_pct", 0)
        _upstream_errors = _nginx_stats.get("upstream_errors", 0)
        if _nginx_stats.get("log_readable"):
            if _err_rate > 20:
                fail(f"Nginx HIGH error rate ({_err_rate:.0f}% errors, {_req_rate:.0f} req/min)")
            elif _err_rate > 5 or _upstream_errors > 10:
                warn(f"Nginx elevated errors ({_err_rate:.1f}% errors, {_upstream_errors} upstream, "
                     f"{_req_rate:.0f} req/min)")
            elif _req_rate > 300:
                warn(f"Nginx heavy traffic ({_req_rate:.0f} req/min, {_err_rate:.1f}% errors)")
            else:
                ok(f"Nginx healthy ({_req_rate:.0f} req/min, {_err_rate:.1f}% errors, "
                   f"{_nginx_stats.get('unique_ips', 0)} unique IPs)")
        else:
            warn("Nginx log not readable (check permissions)")
    except Exception as e:
        warn(f"Nginx health check failed ({e})")

    # ── 18. Service health: Language Teacher ────────────────────────────
    teacher_result = _check_teacher(base_url, label.split()[0])
    _teach_ago = teacher_result.get("last_teach_ago", "never")
    _vocab = teacher_result.get("vocab_total", 0)
    _prod = teacher_result.get("vocab_producible", 0)
    _comp_lvl = teacher_result.get("composition_level", "?")
    if teacher_result["status"] == "fail":
        fail(f"Language Teacher FAIL — {teacher_result['details']}")
    elif _teach_ago == "never" or (_teach_ago.endswith("h ago") and
            float(_teach_ago.replace("h ago", "")) > 24):
        fail(f"Language Teacher STALE — last teach: {_teach_ago}, vocab={_vocab}/{_prod}")
    else:
        ok(f"Language Teacher active (vocab={_vocab}, prod={_prod}, level={_comp_lvl}, "
           f"last={_teach_ago})")

    # ── 18. Service health: Events Teacher ──────────────────────────────
    _events_check = _check_events_teacher(base_url, label.split()[0])
    if _events_check["status"] == "fail":
        fail(f"Events Teacher FAIL — {_events_check['details']}")
    elif _events_check["status"] == "warn":
        warn(f"Events Teacher — {_events_check['details']}")
    else:
        ok(f"Events Teacher active ({_events_check['details']})")

    # ── 19. Service health: Persona Social ──────────────────────────────
    persona_result = _check_persona(base_url, label.split()[0])
    _p_sessions = persona_result.get("total_sessions", 0)
    _p_quality = persona_result.get("avg_quality", 0)
    _p_jailbreaks = persona_result.get("jailbreak_alerts", 0)
    _p_last = persona_result.get("last_session_ago", "never")
    if persona_result["status"] == "fail":
        fail(f"Persona Social FAIL — {persona_result['details']}")
    elif _p_last == "never":
        fail(f"Persona Social STALE — no sessions recorded")
    else:
        ok(f"Persona Social active (sessions={_p_sessions}, quality={_p_quality:.2f}, "
           f"last={_p_last}, jailbreaks={_p_jailbreaks})")

    # ── 20. Service health: ARC Training ────────────────────────────────
    arc_result = _check_arc(base_url, label.split()[0])
    _arc_games = arc_result.get("total_games", 0)
    _arc_updates = arc_result.get("scorer_updates", 0)
    _arc_loss = arc_result.get("last_loss", 0)
    if arc_result["status"] == "fail":
        fail(f"ARC Training FAIL — {arc_result['details']}")
    elif arc_result["status"] == "warn":
        warn(f"ARC Training — {arc_result['details']}")
    else:
        ok(f"ARC Training active ({arc_result['details']})")

    _print_health_results(results)


def _check_nginx_health(log_path: str = "/var/log/nginx/access.log",
                        window_minutes: int = 5) -> dict:
    """Analyze recent nginx traffic for health check.

    Returns dict with: log_readable, req_per_min, error_pct, upstream_errors, unique_ips.
    """
    import subprocess
    import time as _time

    result = {
        "log_readable": False, "req_per_min": 0, "error_pct": 0,
        "upstream_errors": 0, "unique_ips": 0,
    }

    # Read last N lines (enough for ~5 min window at 500 req/min = ~2500 lines)
    try:
        proc = subprocess.run(
            ["sudo", "-S", "tail", "-3000", log_path],
            input=(os.environ.get("TITAN_VPS_SUDO_PW", "") + "\n"), capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return result
        lines = proc.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return result
        result["log_readable"] = True
    except Exception:
        return result

    # Parse: filter to recent window
    now = _time.time()
    cutoff = now - (window_minutes * 60)
    recent = []
    ips = set()
    errors = 0
    upstream_errs = 0
    _MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        # Parse timestamp: [05/Apr/2026:22:45:00 +0000]
        try:
            ts_str = parts[3].lstrip("[")
            day, mon, rest = ts_str.split("/", 2)
            year_time = rest.split(":", 1)
            year = int(year_time[0])
            hms = year_time[1]
            h, m, s = hms.split(":")
            import calendar
            ts = calendar.timegm((year, _MONTHS.get(mon, 1), int(day),
                                  int(h), int(m), int(s)))
        except Exception:
            ts = now  # Can't parse — assume recent

        if ts < cutoff:
            continue

        recent.append(line)
        ips.add(parts[0])

        # HTTP status
        try:
            status = int(parts[8])
            if status >= 500:
                errors += 1
                upstream_errs += 1
            elif status >= 400:
                errors += 1
        except (ValueError, IndexError):
            pass

    total = len(recent)
    result["req_per_min"] = total / max(window_minutes, 1)
    result["error_pct"] = (errors / total * 100) if total > 0 else 0
    result["upstream_errors"] = upstream_errs
    result["unique_ips"] = len(ips)
    return result


def run_traffic(hours: int = 1):
    """Show nginx traffic stats for the last N hours."""
    import subprocess
    import time as _time
    import calendar

    log_path = "/var/log/nginx/access.log"
    print()
    print(f"NGINX TRAFFIC — last {hours}h")
    print("=" * 70)

    # Read log with sudo
    try:
        proc = subprocess.run(
            ["sudo", "-S", "cat", log_path],
            input=(os.environ.get("TITAN_VPS_SUDO_PW", "") + "\n"), capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            print("  ERROR: Cannot read nginx access log")
            return
        all_lines = proc.stdout.strip().split("\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    now = _time.time()
    cutoff = now - (hours * 3600)
    _MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

    # Parse all lines in window
    entries = []  # (timestamp, ip, method, path, status, bytes)
    for line in all_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            ts_str = parts[3].lstrip("[")
            day, mon, rest = ts_str.split("/", 2)
            year_time = rest.split(":", 1)
            year = int(year_time[0])
            hms = year_time[1]
            h, m, s = hms.split(":")
            ts = calendar.timegm((year, _MONTHS.get(mon, 1), int(day),
                                  int(h), int(m), int(s)))
        except Exception:
            continue

        if ts < cutoff:
            continue

        ip = parts[0]
        method = parts[5].lstrip('"')
        path = parts[6]
        try:
            status = int(parts[8])
        except (ValueError, IndexError):
            status = 0
        try:
            nbytes = int(parts[9])
        except (ValueError, IndexError):
            nbytes = 0

        entries.append((ts, ip, method, path, status, nbytes))

    if not entries:
        print(f"  No requests in the last {hours}h")
        return

    total = len(entries)
    duration_min = (entries[-1][0] - entries[0][0]) / 60 if len(entries) > 1 else 1
    total_bytes = sum(e[5] for e in entries)

    # ── Summary ──
    unique_ips = set(e[1] for e in entries)
    statuses = {}
    for e in entries:
        bucket = f"{e[4] // 100}xx" if e[4] > 0 else "0xx"
        statuses[bucket] = statuses.get(bucket, 0) + 1

    print(f"\n  SUMMARY")
    print(f"  {'Total requests:':<25} {total:,}")
    print(f"  {'Avg req/min:':<25} {total / max(duration_min, 1):.1f}")
    print(f"  {'Unique IPs:':<25} {len(unique_ips)}")
    print(f"  {'Total bandwidth:':<25} {total_bytes / (1024*1024):.1f} MB")
    print(f"  {'Time span:':<25} {duration_min:.0f} min")

    # Status breakdown
    print(f"\n  STATUS CODES")
    for code in sorted(statuses.keys()):
        pct = statuses[code] / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {code}: {statuses[code]:>6,} ({pct:5.1f}%) {bar}")

    # ── Top IPs ──
    ip_counts = {}
    for e in entries:
        ip_counts[e[1]] = ip_counts.get(e[1], 0) + 1
    top_ips = sorted(ip_counts.items(), key=lambda x: -x[1])[:10]

    print(f"\n  TOP IPs")
    for ip, count in top_ips:
        pct = count / total * 100
        print(f"    {ip:<20} {count:>6,} ({pct:5.1f}%)")

    # ── Top Paths ──
    path_counts = {}
    for e in entries:
        path_counts[e[3]] = path_counts.get(e[3], 0) + 1
    top_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:15]

    print(f"\n  TOP ENDPOINTS")
    for path, count in top_paths:
        pct = count / total * 100
        print(f"    {path:<40} {count:>6,} ({pct:5.1f}%)")

    # ── Hourly breakdown ──
    hourly = {}
    for e in entries:
        hour_key = _time.strftime("%H:00", _time.gmtime(e[0]))
        hourly[hour_key] = hourly.get(hour_key, 0) + 1

    if len(hourly) > 1:
        print(f"\n  HOURLY BREAKDOWN (UTC)")
        max_val = max(hourly.values()) if hourly else 1
        for hour in sorted(hourly.keys()):
            count = hourly[hour]
            bar_len = int(count / max_val * 30)
            bar = "█" * bar_len
            print(f"    {hour}  {count:>6,}  {bar}")

    # ── Error details ──
    error_paths = {}
    for e in entries:
        if e[4] >= 400:
            error_paths[e[3]] = error_paths.get(e[3], 0) + 1
    if error_paths:
        print(f"\n  ERROR PATHS (4xx/5xx)")
        for path, count in sorted(error_paths.items(), key=lambda x: -x[1])[:10]:
            print(f"    {path:<40} {count:>4}")

    print()


def _print_health_results(results: list):
    """Print formatted health check results and summary."""
    icons = {"pass": "\u2713", "warn": "\u26a0", "fail": "\u2717"}

    for status, msg in results:
        icon = icons[status]
        print(f"  {icon} {msg}")

    passed = sum(1 for s, _ in results if s == "pass")
    warnings = sum(1 for s, _ in results if s == "warn")
    failed = sum(1 for s, _ in results if s == "fail")
    print(f"\n  PASSED: {passed}/{len(results)}  WARNINGS: {warnings}  FAILED: {failed}")
    print()


# ── Language Teacher Progress ─────────────────────────────────────────

def show_teacher_progress():
    """Display language teacher session stats from inner_memory.db."""
    import sqlite3
    import time as _time

    db_path = "data/inner_memory.db"
    print("\nLANGUAGE TEACHER PROGRESS")
    print("=" * 55)

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row

        # Check if table exists
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='teacher_sessions'"
        ).fetchall()]
        if not tables:
            print("  No teacher_sessions table yet (teacher hasn't fired)")
            conn.close()
            return

        # Total sessions
        total = conn.execute("SELECT count(*) FROM teacher_sessions").fetchone()[0]
        if total == 0:
            print("  No teaching sessions recorded yet")
            conn.close()
            return

        # Last session time
        last_ts = conn.execute("SELECT max(timestamp) FROM teacher_sessions").fetchone()[0]
        ago = _time.time() - last_ts if last_ts else 0
        if ago < 3600:
            ago_str = "%.0f min ago" % (ago / 60)
        else:
            ago_str = "%.1f hours ago" % (ago / 3600)

        print(f"  Total sessions:     {total}")
        print(f"  Last session:       {ago_str}")
        print()

        # Mode distribution
        print("  Mode distribution:")
        modes = conn.execute(
            "SELECT mode, count(*) as c FROM teacher_sessions GROUP BY mode ORDER BY c DESC"
        ).fetchall()
        for row in modes:
            pct = row["c"] / total * 100
            extra = ""
            if row["mode"] == "grammar":
                rules = conn.execute(
                    "SELECT count(*) FROM teacher_sessions WHERE mode='grammar' AND correction IS NOT NULL"
                ).fetchone()[0]
                extra = f"  | {rules} corrections"
            elif row["mode"] == "modeling":
                patterns = conn.execute(
                    "SELECT count(*) FROM teacher_sessions WHERE mode='modeling' AND pattern_hash IS NOT NULL"
                ).fetchone()[0]
                extra = f"  | {patterns} new patterns"
            print(f"    {row['mode']:12s} {row['c']:4d} ({pct:4.1f}%){extra}")
        print()

        # Comprehension impact
        avg_recognized = conn.execute(
            "SELECT avg(words_recognized) FROM teacher_sessions WHERE words_recognized > 0"
        ).fetchone()[0] or 0
        total_recognized = conn.execute(
            "SELECT sum(words_recognized) FROM teacher_sessions"
        ).fetchone()[0] or 0
        print("  Comprehension impact:")
        print(f"    Avg words recognized: {avg_recognized:.1f} / session")
        print(f"    Total words felt:     {total_recognized}")
        print()

        # Grammar rules from teacher
        try:
            grammar_rules = conn.execute(
                "SELECT count(*) FROM grammar_rules WHERE source='language_teacher'"
            ).fetchone()[0]
        except Exception:
            grammar_rules = "?"
        print(f"  Grammar rules from teacher: {grammar_rules}")

        conn.close()

    except Exception as e:
        print(f"  Error reading teacher data: {e}")

    print("=" * 55)


# ── Multi-Titan Service Diagnostics ──────────────────────────────────

TITAN_ENDPOINTS = {
    "T1": "http://127.0.0.1:7777",
    "T2": "http://10.135.0.6:7777",
    "T3": "http://10.135.0.6:7778",
}


_SERVICES_GET_LAST_ERROR: dict = {}


def _services_get(base_url: str, path: str, timeout: float = 15.0) -> dict | None:
    """GET JSON endpoint, return parsed dict or None.

    On failure, records category ("timeout" | "http_<code>" | "connection" |
    "parse" | "other") in _SERVICES_GET_LAST_ERROR[(base_url, path)] and
    prints a single diagnostic line to stderr so cron logs capture the cause.
    Retries once on timeout or connection errors (diagnostics are not
    latency-critical, and 30KB payloads under load can exceed 8s).
    """
    import requests
    import sys as _sys
    key = (base_url, path)
    _SERVICES_GET_LAST_ERROR.pop(key, None)
    for attempt in (1, 2):
        try:
            r = requests.get(f"{base_url}{path}", timeout=timeout)
            if r.status_code != 200:
                _SERVICES_GET_LAST_ERROR[key] = f"http_{r.status_code}"
                print(f"[services_get] {base_url}{path} → HTTP {r.status_code}",
                      file=_sys.stderr)
                return None
            try:
                return r.json()
            except ValueError as e:
                _SERVICES_GET_LAST_ERROR[key] = "parse"
                print(f"[services_get] {base_url}{path} → parse error: {e}",
                      file=_sys.stderr)
                return None
        except requests.exceptions.Timeout:
            if attempt == 1:
                continue
            _SERVICES_GET_LAST_ERROR[key] = "timeout"
            print(f"[services_get] {base_url}{path} → timeout after {timeout}s "
                  f"(both attempts)", file=_sys.stderr)
            return None
        except requests.exceptions.ConnectionError as e:
            if attempt == 1:
                continue
            _SERVICES_GET_LAST_ERROR[key] = "connection"
            print(f"[services_get] {base_url}{path} → connection error: "
                  f"{str(e)[:120]}", file=_sys.stderr)
            return None
        except Exception as e:
            _SERVICES_GET_LAST_ERROR[key] = "other"
            print(f"[services_get] {base_url}{path} → {type(e).__name__}: "
                  f"{str(e)[:120]}", file=_sys.stderr)
            return None
    return None


def _svc_unwrap(resp: dict | None) -> dict:
    """Unwrap {"status": "ok", "data": {...}} → data dict."""
    if not resp:
        return {}
    d = resp.get("data", resp)
    return d if isinstance(d, dict) else {}


def _check_teacher(base_url: str, titan_id: str, db_path: str | None = None) -> dict:
    """Check language teacher health for one Titan.

    Returns dict with: status (ok/warn/fail), vocab_total, vocab_producible,
    avg_confidence, teacher_sessions, last_teach_ago, composition_level, details.
    """
    import sqlite3
    import time as _time

    result = {
        "titan": titan_id, "subsystem": "teacher",
        "status": "fail", "details": "",
        "vocab_total": 0, "vocab_producible": 0, "avg_confidence": 0.0,
        "teacher_sessions": 0, "last_teach_ago": "never",
        "composition_level": "?",
    }

    # Try API first (works for all Titans)
    trinity = _svc_unwrap(_services_get(base_url, "/v4/inner-trinity"))
    lang = trinity.get("language", {})

    if lang and lang.get("vocab_total", 0) > 0:
        result["vocab_total"] = lang.get("vocab_total", 0)
        result["vocab_producible"] = lang.get("vocab_producible", 0)
        result["avg_confidence"] = round(lang.get("avg_confidence", 0), 3)
        result["composition_level"] = lang.get("composition_level", "?")
        sessions_last_hr = lang.get("teacher_sessions_last_hour", 0)
        result["teacher_sessions"] = sessions_last_hr
        result["last_teach_ago"] = f"{sessions_last_hr}/hr"
    else:
        # Fallback: direct vocabulary API
        vocab_resp = _services_get(base_url, "/v4/vocabulary")
        if vocab_resp:
            vdata = _svc_unwrap(vocab_resp)
            words = vdata.get("words", [])
            result["vocab_total"] = len(words)
            result["vocab_producible"] = sum(
                1 for w in words if w.get("confidence", 0) >= 0.5)
            confs = [w.get("confidence", 0) for w in words if w.get("confidence", 0) > 0]
            result["avg_confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0

    # Direct DB access (T1 local, T2/T3 via SSH)
    _db_resolved = db_path
    _remote_titan = titan_id in ("T2", "T3") and not db_path
    if _remote_titan:
        # T2/T3: query DB via SSH + sqlite3 CLI
        _remote_db_map = {
            "T2": "/home/antigravity/projects/titan/data/inner_memory.db",
            "T3": "/home/antigravity/projects/titan3/data/inner_memory.db",
        }
        _remote_db = _remote_db_map.get(titan_id, "")
        try:
            import subprocess
            _ssh_sql = (
                f"sqlite3 {_remote_db} "
                f"\"SELECT COUNT(*) FROM vocabulary; "
                f"SELECT COUNT(*) FROM vocabulary WHERE confidence >= 0.5; "
                f"SELECT COALESCE(MAX(timestamp),0) FROM teacher_sessions; "
                f"SELECT COALESCE(MAX(level),0) FROM composition_history;\""
            )
            _ssh_r = subprocess.run(
                f"ssh -o ConnectTimeout=3 root@10.135.0.6 '{_ssh_sql}'",
                shell=True, capture_output=True, text=True, timeout=10)
            _lines = _ssh_r.stdout.strip().split("\n")
            if len(_lines) >= 3:
                result["vocab_total"] = int(_lines[0])
                result["vocab_producible"] = int(_lines[1])
                last_ts = float(_lines[2])
                if last_ts > 0:
                    ago = _time.time() - last_ts
                    if ago < 3600:
                        result["last_teach_ago"] = f"{ago/60:.0f}m ago"
                    else:
                        result["last_teach_ago"] = f"{ago/3600:.1f}h ago"
                if len(_lines) >= 4 and _lines[3].strip():
                    result["composition_level"] = f"L{_lines[3].strip()}"
        except Exception:
            pass

    if _db_resolved or (db_path and not _remote_titan):
        _db_resolved = _db_resolved or db_path
        try:
            conn = sqlite3.connect(_db_resolved, timeout=5.0)
            total = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
            prod = conn.execute(
                "SELECT COUNT(*) FROM vocabulary WHERE confidence >= 0.5").fetchone()[0]
            result["vocab_total"] = total
            result["vocab_producible"] = prod

            # Teacher sessions
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='teacher_sessions'"
            ).fetchall()]
            if tables:
                ts_total = conn.execute("SELECT count(*) FROM teacher_sessions").fetchone()[0]
                result["teacher_sessions"] = ts_total
                last_ts = conn.execute(
                    "SELECT max(timestamp) FROM teacher_sessions").fetchone()[0]
                if last_ts:
                    ago = _time.time() - last_ts
                    if ago < 3600:
                        result["last_teach_ago"] = f"{ago/60:.0f}m ago"
                    else:
                        result["last_teach_ago"] = f"{ago/3600:.1f}h ago"

            # Composition level
            comp_tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='composition_history'"
            ).fetchall()]
            if comp_tables:
                max_lvl = conn.execute(
                    "SELECT MAX(level) FROM composition_history").fetchone()[0] or 0
                result["composition_level"] = f"L{max_lvl}"

            conn.close()
        except Exception:
            pass  # API data is sufficient

    # Determine status
    vt = result["vocab_total"]
    if vt == 0:
        result["status"] = "fail"
        result["details"] = "vocab=0 (teacher not producing)"
    elif vt < 30:
        result["status"] = "warn"
        result["details"] = f"vocab={vt} (still bootstrapping)"
    elif result["avg_confidence"] < 0.1:
        result["status"] = "warn"
        result["details"] = f"low confidence ({result['avg_confidence']:.3f})"
    else:
        result["status"] = "ok"
        result["details"] = f"vocab={vt} prod={result['vocab_producible']}"

    return result


def _check_arc(base_url: str, titan_id: str) -> dict:
    """Check ARC subsystem health for one Titan."""
    result = {
        "titan": titan_id, "subsystem": "arc",
        "status": "fail", "details": "",
        "active": False, "total_games": 0,
        "avg_reward": 0.0, "best_levels": 0,
        "scorer_updates": 0, "last_loss": 0.0,
    }

    arc_resp = _services_get(base_url, "/v4/arc-status")
    if not arc_resp:
        result["details"] = "API unreachable"
        return result

    arc = _svc_unwrap(arc_resp)
    result["active"] = arc.get("active", False)
    arc_results = arc.get("results", {})
    games = arc_results.get("games", {})
    scorecard = arc_results.get("scorecard", {})
    scorers = arc.get("scorers", {})

    result["total_games"] = len(games)
    if scorecard:
        result["avg_reward"] = round(scorecard.get("score", 0), 3)
        result["best_levels"] = scorecard.get("total_levels_completed", 0)

    # Scorer health
    total_updates = 0
    losses = []
    for gid, sinfo in scorers.items():
        total_updates += sinfo.get("total_updates", 0)
        ll = sinfo.get("last_loss", 0)
        if ll > 0:
            losses.append(ll)
    result["scorer_updates"] = total_updates
    result["last_loss"] = round(sum(losses) / len(losses), 4) if losses else 0.0

    # Determine status
    if not result["active"]:
        result["status"] = "warn"
        result["details"] = "ARC inactive"
    elif result["total_games"] == 0:
        result["status"] = "warn"
        result["details"] = "ARC active but no game results"
    elif result["best_levels"] == 0 and total_updates > 100:
        result["status"] = "warn"
        result["details"] = f"0 levels completed ({total_updates} scorer updates)"
    else:
        result["status"] = "ok"
        result["details"] = (
            f"{result['total_games']} games, "
            f"levels={result['best_levels']}, "
            f"score={result['avg_reward']:.3f}"
        )

    return result


def _check_persona(base_url: str, titan_id: str,
                   telemetry_path: str | None = None,
                   alerts_path: str | None = None) -> dict:
    """Check persona social system health for one Titan."""
    import time as _time

    result = {
        "titan": titan_id, "subsystem": "persona",
        "status": "fail", "details": "",
        "total_sessions": 0, "last_session_ago": "never",
        "avg_quality": 0.0, "concepts_detected": [],
        "jailbreak_alerts": 0, "jailbreak_min_score": 1.0,
        "session_types": {},
    }

    # Try API telemetry
    telem_resp = _services_get(base_url, f"/v4/persona-telemetry?titan={titan_id}&limit=100")
    if telem_resp:
        tdata = _svc_unwrap(telem_resp)
        entries = tdata.get("entries", [])
        result["total_sessions"] = tdata.get("total_entries", len(entries))
        result["jailbreak_alerts"] = tdata.get("jailbreak_alerts", 0)
        result["session_types"] = tdata.get("by_session_type", {})

        if entries:
            qualities = [e.get("conversation_quality", 0) for e in entries if e.get("conversation_quality")]
            result["avg_quality"] = round(sum(qualities) / len(qualities), 2) if qualities else 0.0

            # Unique concepts across recent sessions
            all_concepts = set()
            for e in entries[:20]:
                for c in e.get("concepts_detected", []):
                    all_concepts.add(c)
            result["concepts_detected"] = sorted(all_concepts)

            # Last session time
            latest_ts = max(e.get("timestamp", 0) for e in entries)
            if latest_ts > 0:
                ago = _time.time() - latest_ts
                if ago < 3600:
                    result["last_session_ago"] = f"{ago/60:.0f}m ago"
                else:
                    result["last_session_ago"] = f"{ago/3600:.1f}h ago"

    # Fallback: read telemetry JSONL directly (T1 only)
    if telemetry_path and result["total_sessions"] == 0:
        try:
            import json as _json
            entries = []
            with open(telemetry_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            e = _json.loads(line)
                            if e.get("titan") == titan_id:
                                entries.append(e)
                        except _json.JSONDecodeError:
                            pass
            result["total_sessions"] = len(entries)
            if entries:
                qualities = [e.get("conversation_quality", 0) for e in entries[-50:]]
                result["avg_quality"] = round(sum(qualities) / len(qualities), 2) if qualities else 0.0
        except FileNotFoundError:
            pass

    # Check jailbreak alerts
    if alerts_path:
        try:
            import json as _json
            with open(alerts_path) as f:
                alerts = _json.loads(f.read())
            titan_alerts = [a for a in alerts if a.get("titan") == titan_id]
            result["jailbreak_alerts"] = len(titan_alerts)
            if titan_alerts:
                scores = [a.get("score", 1.0) for a in titan_alerts]
                result["jailbreak_min_score"] = min(scores)
        except (FileNotFoundError, Exception):
            pass

    # Determine status
    if result["total_sessions"] == 0:
        result["status"] = "warn"
        result["details"] = "no sessions recorded"
    elif result["jailbreak_alerts"] > 0 and result["jailbreak_min_score"] < 0.5:
        result["status"] = "fail"
        result["details"] = (
            f"JAILBREAK: {result['jailbreak_alerts']} alerts, "
            f"min_score={result['jailbreak_min_score']:.1f}"
        )
    elif result["avg_quality"] < 0.3:
        result["status"] = "warn"
        result["details"] = f"low quality ({result['avg_quality']:.2f})"
    else:
        result["status"] = "ok"
        result["details"] = (
            f"{result['total_sessions']} sessions, "
            f"quality={result['avg_quality']:.2f}, "
            f"concepts={','.join(result['concepts_detected'][:4])}"
        )

    return result


def _check_events_teacher(base_url: str, titan_id: str) -> dict:
    """Check Events Teacher health for one Titan.

    Checks the events_teacher log age and window count via API or log files.
    """
    import time as _time

    result = {
        "titan": titan_id, "subsystem": "events_teacher",
        "status": "fail", "details": "",
        "windows": 0, "felt_items": 0, "followers": 0,
        "last_run_ago": "never",
    }

    # Try API endpoint
    et_resp = _services_get(base_url, "/v4/events-teacher-status")
    if et_resp:
        et_data = _svc_unwrap(et_resp)
        if et_data.get("windows", 0) > 0:
            result["windows"] = et_data.get("windows", 0)
            result["felt_items"] = et_data.get("felt_items", 0)
            result["followers"] = et_data.get("followers", 0)
            last_ts = et_data.get("last_run_ts", 0)
            if last_ts > 0:
                ago = _time.time() - last_ts
                result["last_run_ago"] = f"{ago/60:.0f}m ago" if ago < 3600 else f"{ago/3600:.1f}h ago"
            result["status"] = "ok"
            result["details"] = (f"{result['windows']} windows, {result['felt_items']} felt, "
                                 f"{result['followers']} followers, last={result['last_run_ago']}")
            return result

    # Fallback: check log file directly (local for T1, SSH for T2/T3)
    log_map = {"T1": "/tmp/events_teacher_t1.log",
               "T2": "/tmp/events_teacher_t2.log",
               "T3": "/tmp/events_teacher_t3.log"}
    log_path = log_map.get(titan_id, "")
    if not log_path:
        result["status"] = "warn"
        result["details"] = "cannot check (unknown titan_id)"
        return result

    import os
    import subprocess, re

    # For T2/T3: logs are on remote VPS, use SSH to read them
    is_remote = titan_id in ("T2", "T3")
    tail_output = ""
    log_age = None

    if is_remote:
        try:
            ssh_cmd = (f"ssh -o ConnectTimeout=3 root@10.135.0.6 "
                       f"'stat -c %Y {log_path} 2>/dev/null; "
                       f"echo ---; tail -50 {log_path} 2>/dev/null'")
            ssh_result = subprocess.run(
                ssh_cmd, shell=True, capture_output=True, text=True, timeout=10)
            parts = ssh_result.stdout.split("---", 1)
            if parts[0].strip():
                remote_mtime = float(parts[0].strip())
                log_age = _time.time() - remote_mtime
            if len(parts) > 1:
                tail_output = parts[1]
        except Exception:
            pass
    else:
        if os.path.exists(log_path):
            log_age = _time.time() - os.path.getmtime(log_path)
            try:
                tail = subprocess.run(["tail", "-50", log_path],
                                     capture_output=True, text=True, timeout=5)
                tail_output = tail.stdout
            except Exception:
                pass

    if log_age is not None:
        result["last_run_ago"] = (f"{log_age/60:.0f}m ago" if log_age < 3600
                                  else f"{log_age/3600:.1f}h ago")
        # Parse last few lines for window count
        window_lines = [l for l in tail_output.splitlines() if "Window #" in l]
        if window_lines:
            last_window = window_lines[-1]
            m = re.search(r'Window #(\d+)', last_window)
            if m:
                result["windows"] = int(m.group(1))
            m2 = re.search(r'(\d+) stored', last_window)
            if m2:
                result["felt_items"] = int(m2.group(1))

        if log_age < 1800:  # Active in last 30 min
            result["status"] = "ok"
            result["details"] = (f"{result['windows']} windows, last={result['last_run_ago']}")
        elif log_age < 7200:  # Stale (30min-2h)
            result["status"] = "warn"
            result["details"] = f"stale — last activity {result['last_run_ago']}"
        else:
            result["status"] = "fail"
            result["details"] = f"STALE — no activity for {result['last_run_ago']}"
    else:
        result["status"] = "fail"
        result["details"] = "no log file found"

    return result


def run_services_diagnostics(json_output: bool = False):
    """Run Teacher + ARC + Persona + Events Teacher diagnostics across all 3 Titans.

    Two output modes:
      - Table (default): human-readable table for session use
      - JSON (--json): one JSON line per check, suitable for cron + log rotation
    """
    import time as _time

    t1_db = str(PROJECT_ROOT / "data" / "inner_memory.db")
    telemetry_path = str(PROJECT_ROOT / "data" / "persona_telemetry.jsonl")
    alerts_path = str(PROJECT_ROOT / "data" / "jailbreak_alerts.json")

    all_results = []

    for titan_id, base_url in TITAN_ENDPOINTS.items():
        # Check API reachable first
        health = _services_get(base_url, "/health")
        if not health:
            for sub in ("teacher", "arc", "persona", "events_teacher"):
                r = {"titan": titan_id, "subsystem": sub,
                     "status": "fail", "details": "API unreachable"}
                all_results.append(r)
            continue

        # Teacher
        db = t1_db if titan_id == "T1" else None
        all_results.append(_check_teacher(base_url, titan_id, db_path=db))

        # ARC
        all_results.append(_check_arc(base_url, titan_id))

        # Persona
        tp = telemetry_path if titan_id == "T1" else None
        ap = alerts_path if titan_id == "T1" else None
        all_results.append(_check_persona(base_url, titan_id,
                                          telemetry_path=tp, alerts_path=ap))

        # Events Teacher
        all_results.append(_check_events_teacher(base_url, titan_id))

    if json_output:
        # Cron-friendly: one JSON line per result with timestamp
        ts = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        for r in all_results:
            r["checked_at"] = ts
            print(json.dumps(r))
        return all_results

    # Human-readable table
    icons = {
        "ok": "\u2713",    # ✓
        "warn": "\u26a0",  # ⚠
        "fail": "\u2717",  # ✗
    }

    print("\nTITAN SERVICE DIAGNOSTICS")
    print("=" * 90)

    # Group by subsystem for clear reading
    for subsystem in ("teacher", "arc", "persona", "events_teacher"):
        sub_label = {"teacher": "LANGUAGE TEACHER", "arc": "ARC (REASONING)",
                     "persona": "PERSONA SOCIAL",
                     "events_teacher": "EVENTS TEACHER"}[subsystem]
        print(f"\n  {sub_label}")
        print(f"  {'-' * 86}")

        for r in all_results:
            if r["subsystem"] != subsystem:
                continue
            icon = icons.get(r["status"], "?")
            tid = r["titan"]

            if subsystem == "teacher":
                vt = r.get("vocab_total", 0)
                vp = r.get("vocab_producible", 0)
                ac = r.get("avg_confidence", 0)
                cl = r.get("composition_level", "?")
                lt = r.get("last_teach_ago", "?")
                print(f"    {icon} {tid}  vocab={vt:>3d}  prod={vp:>3d}  "
                      f"conf={ac:.3f}  level={cl:<3s}  teach={lt:<12s}  "
                      f"| {r.get('details', '')}")
            elif subsystem == "arc":
                active = "ON" if r.get("active") else "OFF"
                games = r.get("total_games", 0)
                score = r.get("avg_reward", 0)
                lvls = r.get("best_levels", 0)
                loss = r.get("last_loss", 0)
                print(f"    {icon} {tid}  active={active:<3s}  games={games}  "
                      f"score={score:.3f}  levels={lvls}  loss={loss:.4f}  "
                      f"| {r.get('details', '')}")
            elif subsystem == "persona":
                sess = r.get("total_sessions", 0)
                qual = r.get("avg_quality", 0)
                last = r.get("last_session_ago", "?")
                jb = r.get("jailbreak_alerts", 0)
                concepts = ",".join(r.get("concepts_detected", [])[:4])
                print(f"    {icon} {tid}  sessions={sess:>4d}  quality={qual:.2f}  "
                      f"last={last:<12s}  jailbreaks={jb}  concepts={concepts:<15s}  "
                      f"| {r.get('details', '')}")
            elif subsystem == "events_teacher":
                windows = r.get("windows", 0)
                last = r.get("last_run_ago", "?")
                print(f"    {icon} {tid}  windows={windows:>4d}  last={last:<18s}  "
                      f"| {r.get('details', '')}")

    # Summary line
    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    warn_count = sum(1 for r in all_results if r["status"] == "warn")
    fail_count = sum(1 for r in all_results if r["status"] == "fail")
    total = len(all_results)
    print(f"\n  TOTAL: {ok_count}/{total} OK  |  {warn_count} warnings  |  {fail_count} failures")
    print("=" * 90)

    return all_results


# ── Live Wiring Contract Verification ────────────────────────────────

def run_live_audit(base_url: str = "http://127.0.0.1:7777",
                   label: str = "T1"):
    """Verify that data actually flows through wired connections.

    Unlike static `audit` (AST-based, checks code structure), this hits
    the running API to verify runtime data paths:
    - Language stats appear in /v4/inner-trinity
    - signal-concept accepts all concepts (I, YOU, YES, NO, WE, THEY)
    - MSL concept grounder has data
    - ARC results exist
    - Persona telemetry has entries
    - Neuromod data flows to all subsystems
    - Vocabulary is non-empty and growing
    - Teacher sessions are being recorded
    - Social pressure meter is responding
    - Compositions are being produced
    """
    results = []

    def ok(msg):
        results.append(("pass", msg))

    def warn(msg):
        results.append(("warn", msg))

    def fail(msg):
        results.append(("fail", msg))

    print(f"\nLIVE WIRING CONTRACT VERIFICATION — {label}")
    print("=" * 70)

    # ── W1. API reachable ────────────────────────────────────────────
    health = _services_get(base_url, "/health")
    if not health:
        fail("API unreachable — all wiring checks skipped")
        _print_health_results(results)
        return results

    ok("API reachable")

    # ── W2. Language stats in inner-trinity ───────────────────────────
    trinity = _svc_unwrap(_services_get(base_url, "/v4/inner-trinity"))
    lang = trinity.get("language", {})
    if lang and lang.get("vocab_total", 0) > 0:
        ok(f"Language → inner-trinity WIRED (vocab={lang['vocab_total']}, "
           f"prod={lang.get('vocab_producible', 0)})")
    elif lang:
        warn(f"Language in trinity but vocab=0 (stats dict present but empty data)")
    else:
        fail("Language → inner-trinity NOT WIRED (no 'language' key in coordinator)")

    # ── W3. Vocabulary API populated ──────────────────────────────────
    vocab_resp = _services_get(base_url, "/v4/vocabulary")
    if vocab_resp:
        vdata = _svc_unwrap(vocab_resp)
        words = vdata.get("words", [])
        if len(words) > 0:
            confs = [w.get("confidence", 0) for w in words]
            high = sum(1 for c in confs if c >= 0.5)
            ok(f"Vocabulary populated ({len(words)} words, {high} producible)")
        else:
            warn("Vocabulary API returns empty word list")
    else:
        fail("Vocabulary API (/v4/vocabulary) unreachable")

    # ── W4. Compositions being produced ───────────────────────────────
    comp_resp = _services_get(base_url, "/v4/compositions")
    if comp_resp:
        cdata = _svc_unwrap(comp_resp)
        total_comp = cdata.get("total_compositions", 0)
        latest = cdata.get("latest", {})
        if total_comp > 0:
            ok(f"Compositions active ({total_comp} total, "
               f"latest L{latest.get('level', '?')})")
        else:
            warn("Compositions table exists but 0 entries")
    else:
        warn("Compositions API (/v4/compositions) unreachable")

    # ── W5. MSL concept grounder has data ─────────────────────────────
    msl = trinity.get("msl", {})
    i_conf = msl.get("i_confidence", 0)
    concepts = msl.get("concept_confidences", msl.get("concepts", {}))
    if i_conf > 0:
        ok(f"MSL I-confidence present ({i_conf:.3f})")
    else:
        warn("MSL I-confidence = 0 (grounding not started or not exposed)")

    if concepts and any(v > 0 for v in concepts.values()):
        active = {k: round(v, 3) for k, v in concepts.items() if v > 0}
        ok(f"MSL concept grounder active ({active})")
    else:
        warn("MSL concept confidences all zero or missing")

    # ── W6. signal-concept endpoint accepts all concepts ──────────────
    import requests
    valid_concepts = ["I", "YES", "NO", "YOU", "WE", "THEY"]
    concept_ok = 0
    concept_fail = []
    for concept in valid_concepts:
        try:
            r = requests.post(
                f"{base_url}/v4/signal-concept",
                json={"concept": concept, "quality": 0.01},  # minimal quality
                timeout=5
            )
            if r.status_code == 200:
                concept_ok += 1
            else:
                concept_fail.append(f"{concept}={r.status_code}")
        except Exception as e:
            concept_fail.append(f"{concept}=err")
    if not concept_fail:
        ok(f"signal-concept accepts all {len(valid_concepts)} concepts")
    else:
        fail(f"signal-concept REJECTS: {', '.join(concept_fail)}")

    # ── W7. ARC status endpoint ───────────────────────────────────────
    arc = _svc_unwrap(_services_get(base_url, "/v4/arc-status"))
    if arc:
        active = arc.get("active", False)
        games = len(arc.get("results", {}).get("games", {}))
        ok(f"ARC subsystem {'active' if active else 'inactive'} ({games} game results)")
    else:
        warn("ARC status (/v4/arc-status) unavailable")

    # ── W8. Social pressure meter ─────────────────────────────────────
    sp = _svc_unwrap(_services_get(base_url, "/v4/social-pressure"))
    if sp and "fill_pct" in sp:
        ok(f"Social pressure meter active (fill={sp['fill_pct']}%)")
    else:
        warn("Social pressure (/v4/social-pressure) not responding")

    # ── W9. Persona telemetry endpoint ────────────────────────────────
    pt = _svc_unwrap(_services_get(base_url, "/v4/persona-telemetry?limit=5"))
    if pt:
        total = pt.get("total_entries", 0)
        jb = pt.get("jailbreak_alerts", 0)
        if total > 0:
            ok(f"Persona telemetry populated ({total} entries, {jb} jailbreak alerts)")
        else:
            warn("Persona telemetry endpoint works but 0 entries")
    else:
        warn("Persona telemetry (/v4/persona-telemetry) unavailable")

    # ── W10. Neuromod → subsystem coupling ────────────────────────────
    nm = _svc_unwrap(_services_get(base_url, "/v4/neuromodulators"))
    modulators = nm.get("modulators", {})
    if modulators:
        da = modulators.get("DA", {}).get("level", 0)
        ne = modulators.get("NE", {}).get("level", 0)
        gaba = modulators.get("GABA", {}).get("level", 0)
        if da > 0 and ne > 0:
            ok(f"Neuromod data flowing (DA={da:.3f}, NE={ne:.3f}, GABA={gaba:.3f})")
        else:
            warn(f"Neuromod data present but DA={da:.3f}, NE={ne:.3f} (possible stall)")
    else:
        fail("Neuromod data unavailable (/v4/neuromodulators)")

    # ── W11. Reasoning → inner-trinity ────────────────────────────────
    reasoning = trinity.get("reasoning", {})
    chains = reasoning.get("total_chains", 0)
    if chains > 0:
        commits = reasoning.get("total_conclusions", 0)
        rate = (commits / chains * 100) if chains > 0 else 0
        ok(f"Reasoning → trinity WIRED (chains={chains}, commit_rate={rate:.0f}%)")
    else:
        warn("Reasoning not in trinity (0 chains or not wired)")

    # ── W12. pi-heartbeat present ─────────────────────────────────────
    pi = trinity.get("pi_heartbeat", {})
    pi_rate = pi.get("heartbeat_ratio", 0)
    if pi_rate > 0:
        ok(f"pi-heartbeat → trinity WIRED (ratio={pi_rate*100:.1f}%)")
    else:
        warn("pi-heartbeat not in trinity data")

    _print_health_results(results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────

# ── Report — Full Titan Comparison ────────────────────────────────────

TITAN_REPORT_ENDPOINTS = [
    ("T1", "http://127.0.0.1:7777"),
    ("T2", "http://10.135.0.6:7777"),
    ("T3", "http://10.135.0.6:7778"),
]


def _fmt_pct(v, width=6):
    """Format a float as percentage string."""
    if v is None:
        return "?".rjust(width)
    return f"{v*100:.1f}%".rjust(width)


def _fmt_num(v, width=8):
    """Format a number with commas."""
    if v is None:
        return "?".rjust(width)
    if isinstance(v, float):
        return f"{v:.3f}".rjust(width)
    return f"{v:,}".rjust(width)


def run_report():
    """Full comparison report across all 3 Titans."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print()
    print("TITAN COMPARISON REPORT")
    print("=" * 90)
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Fetch all data in parallel
    data = {}

    def _get_with_retry(url_path, attempts=2, timeout=30):
        """GET with one retry on timeout. 30s default per the smart-watchdog
        calibration (2026-04-13): meditation/FAISS save can hold the event
        loop legitimately for 15-30s. A single failed probe was returning
        empty data and showing 0/?? for that Titan in the report."""
        last_err = None
        for _ in range(attempts):
            try:
                r = requests.get(url_path, timeout=timeout)
                return r
            except requests.exceptions.Timeout as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break
        if last_err:
            raise last_err
        return None

    def _fetch(tid, url):
        result = {"tid": tid, "url": url, "up": False}
        try:
            # inner-trinity (main state)
            r = _get_with_retry(f"{url}/v4/inner-trinity", attempts=2, timeout=30)
            if r is not None and r.status_code == 200:
                result["trinity"] = r.json().get("data", {})
                result["up"] = True
            # vocabulary
            r2 = _get_with_retry(f"{url}/v4/vocabulary", attempts=2, timeout=15)
            if r2 is not None and r2.status_code == 200:
                vdata = r2.json().get("data", {})
                result["vocab_count"] = len(vdata.get("words", []))
                prod = sum(1 for w in vdata.get("words", [])
                           if w.get("learning_phase") == "producible")
                result["vocab_prod"] = prod
            # language-grounding (CGN)
            r3 = _get_with_retry(f"{url}/v4/language-grounding",
                                 attempts=2, timeout=15)
            if r3 is not None and r3.status_code == 200:
                gdata = r3.json().get("data", {})
                result["cgn_grounded"] = gdata.get("grounded", 0)
                result["cgn_rate"] = gdata.get("grounding_rate", 0)
                result["cgn_consumers"] = gdata.get("consumers", [])
        except Exception as e:
            result["error"] = type(e).__name__ + ": " + str(e)[:80]
        return result

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_fetch, tid, url): tid for tid, url in TITAN_REPORT_ENDPOINTS}
        for f in as_completed(futures):
            r = f.result()
            data[r["tid"]] = r

    # Print status line
    for tid in ["T1", "T2", "T3"]:
        d = data.get(tid, {})
        status = "✓ UP" if d.get("up") else f"✗ DOWN ({d.get('error', 'unreachable')})"
        print(f"  {tid}: {status}")
    print()

    # Build comparison table
    tids = [tid for tid in ["T1", "T2", "T3"] if data.get(tid, {}).get("up")]
    if not tids:
        print("  No Titans reachable!")
        return

    # Header
    col_w = 14
    header = "  " + "Metric".ljust(26) + "".join(tid.center(col_w) for tid in tids)
    print(header)
    print("  " + "-" * (26 + col_w * len(tids)))

    def _row(label, values):
        row = "  " + label.ljust(26)
        for tid in tids:
            row += str(values.get(tid, "?")).center(col_w)
        print(row)

    # Extract metrics
    def _t(tid, *keys, default=None):
        d = data.get(tid, {}).get("trinity", {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    # Epoch
    _row("Epoch", {tid: _fmt_num(_t(tid, "tick_count")) for tid in tids})

    # Emotion
    _row("Emotion", {
        tid: f"{_t(tid, 'neuromodulators', 'current_emotion', default='?')}"
             f" ({_t(tid, 'neuromodulators', 'emotion_confidence', default=0):.2f})"
        for tid in tids})

    # Chi
    _row("Chi (total)", {tid: f"{_t(tid, 'chi', 'total', default=0):.3f}" for tid in tids})
    _row("Chi state", {tid: _t(tid, "chi", "state", default="?") for tid in tids})

    # Neuromods
    print()
    print("  " + "NEUROMODULATORS".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    for mod in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        _row(f"  {mod}", {
            tid: _fmt_pct(_t(tid, "neuromodulators", "modulators", mod, "level"))
            for tid in tids})

    # MSL
    print()
    print("  " + "MSL / IDENTITY".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("I-confidence", {
        tid: f"{_t(tid, 'msl', 'i_confidence', default=0):.3f}" for tid in tids})
    _row("I-depth", {
        tid: f"{_t(tid, 'msl', 'i_depth', default=0):.4f}" for tid in tids})
    _row("Convergences", {
        tid: _fmt_num(_t(tid, "msl", "convergence_count")) for tid in tids})
    _row("Attention entropy", {
        tid: f"{_t(tid, 'msl', 'attention_entropy', default=0):.2f}" for tid in tids})
    # I-016/HOMEO-REDESIGN visibility (added 2026-04-13). Setpoint entropy
    # measures structural collapse of the allostatic drift layer — drops well
    # before live attention_entropy collapse and stays low even when live
    # attention briefly recovers. Drift_guard_active counts how many ticks
    # the entropy-based dampener engaged (>0 means drift was being resisted).
    _row("Setpoint entropy (norm)", {
        tid: f"{_t(tid, 'msl', 'homeostatic', 'setpoint_entropy_normalized', default=0):.3f}"
        for tid in tids})
    _row("Drift guard active", {
        tid: _fmt_num(_t(tid, "msl", "homeostatic", "drift_guard_active_count"))
        for tid in tids})
    _row("MSL update count", {
        tid: _fmt_num(_t(tid, "msl", "homeostatic", "update_count"))
        for tid in tids})

    # I-depth components
    for comp in ["source_diversity", "concept_network", "emotional_range",
                 "wisdom_depth", "memory_bridge"]:
        _row(f"  {comp}", {
            tid: f"{_t(tid, 'msl', 'i_depth_components', comp, default=0):.3f}"
            for tid in tids})

    # Concepts
    for concept in ["YOU", "NO", "THEY", "WE", "YES"]:
        _row(f"  {concept}", {
            tid: f"{_t(tid, 'msl', 'concept_confidences', concept, default=0):.3f}"
            for tid in tids})

    # Reasoning
    print()
    print("  " + "REASONING".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Meta chains", {
        tid: _fmt_num(_t(tid, "meta_reasoning", "total_chains")) for tid in tids})
    _row("EUREKAs", {
        tid: _fmt_num(_t(tid, "meta_reasoning", "total_eurekas")) for tid in tids})
    _row("Wisdom saved", {
        tid: _fmt_num(_t(tid, "meta_reasoning", "total_wisdom_saved")) for tid in tids})
    _row("Avg reward", {
        tid: f"{_t(tid, 'meta_reasoning', 'avg_reward', default=0):.3f}" for tid in tids})

    # Self-Reasoning
    print()
    print("  " + "SELF-REASONING".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Introspections", {
        tid: _fmt_num(_t(tid, "self_reasoning", "total_introspections")) for tid in tids})
    _row("Predictions", {
        tid: _fmt_num(_t(tid, "self_reasoning", "total_predictions")) for tid in tids})
    _row("Pred accuracy", {
        tid: f"{_t(tid, 'self_reasoning', 'prediction_accuracy_ema', default=0):.3f}" for tid in tids})
    _row("Active preds", {
        tid: _fmt_num(_t(tid, "self_reasoning", "active_predictions")) for tid in tids})

    # Coding Explorer
    print()
    print("  " + "CODING EXPLORER".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Exercises", {
        tid: _fmt_num(_t(tid, "coding_explorer", "total_exercises")) for tid in tids})
    _row("Successes", {
        tid: _fmt_num(_t(tid, "coding_explorer", "total_successes")) for tid in tids})
    _row("Success rate", {
        tid: _fmt_pct(_t(tid, "coding_explorer", "success_rate")) for tid in tids})
    _row("Concepts tried", {
        tid: _fmt_num(_t(tid, "coding_explorer", "concepts_attempted")) for tid in tids})
    # Layer B + C — trigger-source attribution + fallback clock
    _row("Emergent trig", {
        tid: _fmt_num(
            (_t(tid, "coding_explorer", "trigger_source_counts", default={}) or {})
            .get("emergent", 0)) for tid in tids})
    _row("Fallback trig", {
        tid: _fmt_num(
            (_t(tid, "coding_explorer", "trigger_source_counts", default={}) or {})
            .get("fallback", 0)) for tid in tids})
    _row("Silence (s)", {
        tid: _fmt_num(_t(tid, "coding_explorer", "seconds_since_last_exercise",
                         default=0)) for tid in tids})

    # Language
    print()
    print("  " + "LANGUAGE / CGN".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Vocabulary", {
        tid: str(data.get(tid, {}).get("vocab_count", "?")) for tid in tids})
    _row("Productive", {
        tid: str(data.get(tid, {}).get("vocab_prod", "?")) for tid in tids})
    _row("CGN grounded", {
        tid: str(data.get(tid, {}).get("cgn_grounded", "?")) for tid in tids})
    _row("Grounding rate", {
        tid: _fmt_pct(data.get(tid, {}).get("cgn_rate")) for tid in tids})

    # Dreaming
    print()
    print("  " + "DREAMING".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Is dreaming", {
        tid: str(_t(tid, "dreaming", "is_dreaming", default="?")) for tid in tids})
    _row("Dream cycles", {
        tid: _fmt_num(_t(tid, "dreaming", "cycle_count")) for tid in tids})
    _row("Last dream epochs", {
        tid: _fmt_num(_t(tid, "dreaming", "dream_epochs")) for tid in tids})
    _row("Fatigue", {
        tid: f"{_t(tid, 'dreaming', 'fatigue', default=0):.3f}" for tid in tids})
    # I-017 visibility (added 2026-04-13). Distilled = wisdom insights
    # extracted from dreams. distilled=0 over many cycles → broken
    # consolidation. Compare distilled_count to cycle_count for distill rate.
    def _distill_per_cycle(tid):
        d = _t(tid, "dreaming", "distilled_count")
        c = _t(tid, "dreaming", "cycle_count")
        if d is None or c is None or not c:
            return "?"
        return f"{int(d)}/{int(c)} ({100.0 * d / c:.1f}%)"
    _row("Distilled (cum)", {
        tid: _fmt_num(_t(tid, "dreaming", "distilled_count")) for tid in tids})
    _row("Distill / cycle", {
        tid: _distill_per_cycle(tid) for tid in tids})

    # Pi heartbeat
    print()
    print("  " + "PI HEARTBEAT".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Heartbeat ratio", {
        tid: _fmt_pct(_t(tid, "pi_heartbeat", "heartbeat_ratio")) for tid in tids})
    _row("Dev age (clusters)", {
        tid: _fmt_num(_t(tid, "pi_heartbeat", "developmental_age")) for tid in tids})
    _row("Total pi epochs", {
        tid: _fmt_num(_t(tid, "pi_heartbeat", "total_pi_epochs")) for tid in tids})

    # Working memory
    print()
    print("  " + "WORKING MEMORY".ljust(26) + "".join(tid.center(col_w) for tid in tids))
    print("  " + "-" * (26 + col_w * len(tids)))
    _row("Size / Capacity", {
        tid: f"{_t(tid, 'working_memory', 'size', default=0)}/{_t(tid, 'working_memory', 'capacity', default=7)}"
        for tid in tids})

    # Social pressure
    _row("Social urge", {
        tid: f"{_t(tid, 'social_pressure', 'urge', default=0):.1f}/{_t(tid, 'social_pressure', 'threshold', default=50):.0f}"
        for tid in tids})
    _row("Posts today", {
        tid: str(_t(tid, "social_pressure", "posts_today", default="?")) for tid in tids})

    # Divergence alerts
    print()
    alerts = []
    for mod in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
        levels = []
        for tid in tids:
            v = _t(tid, "neuromodulators", "modulators", mod, "level")
            if v is not None:
                levels.append((tid, v))
        if len(levels) >= 2:
            vals = [v for _, v in levels]
            spread = max(vals) - min(vals)
            if spread > 0.15:
                hi = max(levels, key=lambda x: x[1])
                lo = min(levels, key=lambda x: x[1])
                alerts.append(f"  ⚠ {mod} divergence: {hi[0]}={hi[1]:.1%} vs {lo[0]}={lo[1]:.1%} (spread={spread:.1%})")

    # Concept divergence
    for concept in ["YOU", "NO", "THEY", "WE"]:
        levels = []
        for tid in tids:
            v = _t(tid, "msl", "concept_confidences", concept, default=None)
            if v is not None:
                levels.append((tid, v))
        if len(levels) >= 2:
            vals = [v for _, v in levels]
            spread = max(vals) - min(vals)
            if spread > 0.3:
                hi = max(levels, key=lambda x: x[1])
                lo = min(levels, key=lambda x: x[1])
                alerts.append(f"  ⚠ {concept} concept divergence: {hi[0]}={hi[1]:.3f} vs {lo[0]}={lo[1]:.3f}")

    if alerts:
        print("  DIVERGENCE ALERTS")
        for a in alerts:
            print(a)
    else:
        print("  ✓ No significant divergences detected")

    print()
    print("=" * 90)


# ── Deploy — Code Deployment + Restart ────────────────────────────────

def run_deploy(targets: list, restart: bool = False):
    """Deploy code to T2/T3 and optionally restart.

    SAFETY RULES (production infrastructure — real training data at stake):
      - NEVER sync data/ — each Titan's training data is sacred
      - NEVER sync config.toml to T3 — it has instance-specific port (7778)
      - config.toml IS synced to T2 (same port as T1: 7777)
      - titan_params.toml IS synced (shared tuning parameters)
      - Only sync: Python code, scripts, shared params
      - Always verify port after T3 deploy
      - Check dreaming state before restart

    targets: list of "t2", "t3"
    """
    import subprocess
    import time
    import requests

    T2_HOST = "root@10.135.0.6"
    TITAN_DIR = "/home/antigravity/projects/titan"
    T3_DIR = "/home/antigravity/projects/titan3"

    # Files/dirs that must NEVER be synced (per-Titan state)
    RSYNC_EXCLUDES = [
        "data/", "test_env/", ".git/", "__pycache__/", "*.pyc",
        "cognee_data/", "node_modules/", "titan-observatory/",
        "titan-docs-site/", "titan-docs/", "*.db", "*.sig",
        "architecture_map.json", ".env",
    ]

    results = {}

    for target in targets:
        target = target.lower()
        print(f"\n{'='*60}")
        print(f"  DEPLOYING TO {target.upper()}")
        print(f"{'='*60}")

        if target == "t2":
            # Use existing deploy_t2.sh (well-tested, handles config.toml safely)
            cmd = f"bash {TITAN_DIR}/scripts/deploy_t2.sh"
            if restart:
                cmd += " --restart"
            print(f"  Running: {cmd}")
            try:
                proc = subprocess.run(cmd, shell=True, capture_output=True,
                                      text=True, timeout=120)
                print(proc.stdout)
                if proc.stderr:
                    print(f"  STDERR: {proc.stderr[-200:]}")
                results["T2"] = {"deployed": proc.returncode == 0}
            except subprocess.TimeoutExpired:
                print("  ✗ Deploy timed out (120s)")
                results["T2"] = {"deployed": False, "error": "timeout"}
                continue

        elif target == "t3":
            # ── T3 Deploy: code + shared params ONLY ──
            # config.toml is NEVER synced (T3 has port=7778)
            exclude_str = " ".join(f"--exclude='{e}'" for e in RSYNC_EXCLUDES)

            print("  Syncing code to T3...")
            print("  ⚠  config.toml EXCLUDED (T3 port=7778)")

            rsync_cmds = [
                # 1. titan_plugin/ — code only, config.toml explicitly excluded
                f"rsync -az {exclude_str} --exclude='config.toml' "
                f"{TITAN_DIR}/titan_plugin/ {T2_HOST}:{T3_DIR}/titan_plugin/",
                # 2. scripts/ — deployment and management tools
                f"rsync -az --exclude='__pycache__/' --exclude='*.pyc' "
                f"{TITAN_DIR}/scripts/ {T2_HOST}:{T3_DIR}/scripts/",
                # 3. titan_params.toml ONLY — shared tuning parameters (NOT config.toml)
                f"rsync -az {TITAN_DIR}/titan_plugin/titan_params.toml "
                f"{T2_HOST}:{T3_DIR}/titan_plugin/titan_params.toml",
            ]
            try:
                for cmd in rsync_cmds:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True,
                                   text=True, timeout=60)
                print("  ✓ Code synced")

                # Verify T3 config integrity — port MUST be 7778
                port_check = subprocess.run(
                    f"ssh {T2_HOST} \"grep '^port = ' {T3_DIR}/titan_plugin/config.toml\"",
                    shell=True, capture_output=True, text=True, timeout=10)
                port_line = port_check.stdout.strip()
                if "7778" in port_line:
                    print(f"  ✓ T3 config verified: {port_line}")
                else:
                    print(f"  ✗ CRITICAL: T3 port is WRONG ({port_line})!")
                    print(f"    Expected: port = 7778")
                    print(f"    Aborting restart to prevent port conflict.")
                    results["T3"] = {"deployed": True, "port_error": True}
                    continue

                results["T3"] = {"deployed": True}
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  ✗ Deploy failed: {e}")
                results["T3"] = {"deployed": False, "error": str(e)}
                continue

            if restart:
                # Check dreaming state before restart
                print("  Checking T3 dreaming state...")
                try:
                    dream_check = subprocess.run(
                        f"ssh {T2_HOST} \"curl -s http://localhost:7778/v4/inner-trinity 2>/dev/null\"",
                        shell=True, capture_output=True, text=True, timeout=10)
                    if '"is_dreaming": true' in dream_check.stdout:
                        print("  ⚠  T3 IS DREAMING — skipping restart to protect dream cycle")
                        results["T3"]["restart_skipped"] = "dreaming"
                        continue
                    print("  ✓ T3 is awake — safe to restart")
                except Exception:
                    print("  ? Could not check dreaming state — proceeding with restart")

                print("  Restarting T3...")
                try:
                    proc = subprocess.run(
                        f"ssh {T2_HOST} 'bash {T3_DIR}/scripts/t3_manage.sh restart'",
                        shell=True, check=True, capture_output=True, text=True, timeout=60)
                    print(f"  {proc.stdout.strip()}")
                    print("  ✓ Restart command sent")
                except Exception as e:
                    print(f"  ✗ Restart failed: {e}")
                    results["T3"]["restarted"] = False
                    continue

        else:
            print(f"  Unknown target: {target} (use t2, t3, or all)")
            continue

    # Wait for health if restarted
    if restart:
        print(f"\n  Waiting 20s for boot...")
        time.sleep(20)

        health_checks = {
            "T2": "http://10.135.0.6:7777",
            "T3": "http://10.135.0.6:7778",
        }
        for tid, url in health_checks.items():
            if tid.lower()[1:] not in [t.lower()[-1] for t in targets]:
                continue
            try:
                r = requests.get(f"{url}/health", timeout=10)
                status = r.status_code
            except Exception:
                status = 0

            if status == 200:
                print(f"  ✓ {tid} healthy (200)")
                results.setdefault(tid, {})["healthy"] = True
            else:
                # Retry once after 15s
                print(f"  {tid} not ready ({status}), retrying in 15s...")
                time.sleep(15)
                try:
                    r = requests.get(f"{url}/health", timeout=10)
                    status = r.status_code
                except Exception:
                    status = 0
                if status == 200:
                    print(f"  ✓ {tid} healthy (200) on retry")
                    results.setdefault(tid, {})["healthy"] = True
                else:
                    print(f"  ✗ {tid} still not healthy ({status})")
                    results.setdefault(tid, {})["healthy"] = False

    # Summary
    print(f"\n{'='*60}")
    print("  DEPLOY SUMMARY")
    print(f"{'='*60}")
    for tid, r in sorted(results.items()):
        deployed = "✓" if r.get("deployed") else "✗"
        health = ""
        if "healthy" in r:
            health = " | health: " + ("✓" if r["healthy"] else "✗")
        print(f"  {tid}: deploy={deployed}{health}")
    print()


# ── Errors — Cross-Titan Error Summary ───────────────────────────────

# rFP #2 Phase 8 gate — rolling history file per Titan for criteria #4/#6.
# Each `arch_map filter-down` invocation appends one sample; gate check reads
# last N samples. Kept in project root data/ so cron + manual invocations share.
_FD_HISTORY_DIR = "data"
_FD_HISTORY_MAX_SAMPLES = 200  # ~8 days @ 1 sample/hr
_GATE_MULTIPLIER_DIVERGENCE_THRESHOLD = 0.02   # Criterion #9 (added 2026-04-15)
_GATE_LOSS_TREND_SAMPLES = 3                   # Criterion #4: monotonic descent window
_GATE_RATE_HZ_MIN = 0.1                        # Criterion #6: min train-step rate over 6h
_GATE_RATE_WINDOW_SEC = 6 * 3600


def _fd_history_path(titan: str) -> str:
    import os as _os
    return _os.path.join(_FD_HISTORY_DIR, f"filter_down_history_{titan.lower()}.jsonl")


def _fd_history_append(titan: str, v5: dict, publish_enabled: bool) -> None:
    """Record one V5 sample. Best-effort; never raises."""
    import json as _json
    import os as _os
    import time as _time
    try:
        _os.makedirs(_FD_HISTORY_DIR, exist_ok=True)
        sample = {
            "ts": int(_time.time()),
            "train_steps": int(v5.get("total_train_steps", 0)),
            "last_loss": float(v5.get("last_loss", 0.0)),
            "buffer": int(v5.get("buffer_size", 0)),
            "mult_mean": {k: float(v) for k, v in (v5.get("multipliers_mean") or {}).items()},
            "publish_enabled": bool(publish_enabled),
        }
        path = _fd_history_path(titan)
        with open(path, "a") as fh:
            fh.write(_json.dumps(sample) + "\n")
        # Trim to last N samples (rewrite file, cheap at 200 lines)
        with open(path) as fh:
            lines = fh.readlines()
        if len(lines) > _FD_HISTORY_MAX_SAMPLES:
            with open(path, "w") as fh:
                fh.writelines(lines[-_FD_HISTORY_MAX_SAMPLES:])
    except Exception:
        pass  # history is best-effort; never block the status report


def _fd_history_read(titan: str) -> list[dict]:
    import json as _json
    path = _fd_history_path(titan)
    try:
        with open(path) as fh:
            return [_json.loads(l) for l in fh if l.strip()]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _fd_max_divergence(mult_mean: dict) -> float:
    """Criterion #9: max |m - 1.0| across all 6 multiplier means."""
    if not mult_mean:
        return 0.0
    return max(abs(float(v) - 1.0) for v in mult_mean.values())


def _fd_loss_trending_down(history: list[dict]) -> tuple[bool, list[float]]:
    """Criterion #4: last K losses strictly decreasing (K = _GATE_LOSS_TREND_SAMPLES)."""
    losses = [h.get("last_loss", 0.0) for h in history[-_GATE_LOSS_TREND_SAMPLES:]]
    ok = len(losses) == _GATE_LOSS_TREND_SAMPLES and all(
        losses[i] < losses[i - 1] for i in range(1, len(losses))
    )
    return ok, losses


def _fd_rate_over_window(history: list[dict]) -> tuple[float, int, bool]:
    """Criterion #6: train-step rate over last _GATE_RATE_WINDOW_SEC.
    Returns (rate_hz, samples_in_window, insufficient_history).
    """
    import time as _time
    if len(history) < 2:
        return 0.0, len(history), True
    now = int(_time.time())
    window = [h for h in history if now - h.get("ts", 0) <= _GATE_RATE_WINDOW_SEC]
    if len(window) < 2:
        return 0.0, len(window), True
    dt = window[-1]["ts"] - window[0]["ts"]
    dsteps = window[-1]["train_steps"] - window[0]["train_steps"]
    if dt <= 0:
        return 0.0, len(window), True
    return dsteps / dt, len(window), False


def run_filter_down_status(all_titans: bool = False, gate_check: bool = False) -> int:
    """rFP #2 Phase 7 + Phase 8 gate: V4/V5 FILTER_DOWN coexistence monitoring.

    Shows V4 + V5 state side-by-side with 9-criteria coexistence gate check.
    Each invocation appends a V5 sample to data/filter_down_history_<titan>.jsonl
    so the loss-trend + rate-over-6h gates can compute from history.

    Args:
        all_titans: query T2/T3 in addition to T1
        gate_check: exit-code mode — returns 0 iff ALL 9 gate criteria pass for
            every queried Titan (CI / deploy-gate use). Returns 1 otherwise.

    Returns: exit code (0 = all gates pass in gate_check mode; 0 otherwise).
    """
    titans = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]

    print()
    print("FILTER_DOWN STATUS — V4 vs V5 (rFP #2 coexistence monitor — 9-gate)")
    print("=" * 90)

    all_gates_pass = True

    for tid, base_url in titans:
        print(f"\n  {tid} ({base_url})")
        print("  " + "-" * 80)
        resp = _services_get(base_url, "/v4/filter-down-status", timeout=12.0)
        if not resp:
            print("    ✗ endpoint unreachable")
            all_gates_pass = False
            continue

        payload = resp.get("data", resp) if isinstance(resp, dict) else resp
        if not isinstance(payload, dict):
            print(f"    ✗ unexpected response: {payload}")
            all_gates_pass = False
            continue

        v4 = payload.get("v4")
        v5 = payload.get("v5")
        v5_pub = bool(payload.get("v5_publishing"))
        v4_pub = bool(payload.get("v4_publishing"))
        phase = payload.get("coexistence_phase", "?")

        if v4:
            print(f"    V4  buffer={v4.get('buffer_size', 0):<5}  "
                  f"train_steps={v4.get('total_train_steps', 0):<5}  "
                  f"last_loss={v4.get('last_loss', 0):.6f}  "
                  f"publishing={'YES' if v4_pub else 'NO (silent)'}")
        else:
            print("    V4  (not initialized)")

        if v5:
            # Record sample for rolling-history-based gate criteria (#4, #6).
            _fd_history_append(tid, v5, bool(v5.get("publish_enabled", False)))

            mean = v5.get("multipliers_mean", {})
            print(f"    V5  buffer={v5.get('buffer_size', 0):<5}  "
                  f"train_steps={v5.get('total_train_steps', 0):<5}  "
                  f"last_loss={v5.get('last_loss', 0):.6f}  "
                  f"publishing={'YES' if v5_pub else 'NO (silent training)'}")
            print(f"        spirit_strength={v5.get('spirit_filter_strength', 0):.2f}  "
                  f"cold_start_floor={v5.get('cold_start_floor', 0)}")
            print(f"        multipliers_mean: "
                  f"iB={mean.get('inner_body', 1):.3f} "
                  f"iM={mean.get('inner_mind', 1):.3f} "
                  f"iSc={mean.get('inner_spirit_content', 1):.3f} "
                  f"oB={mean.get('outer_body', 1):.3f} "
                  f"oM={mean.get('outer_mind', 1):.3f} "
                  f"oSc={mean.get('outer_spirit_content', 1):.3f}")

            # ── 9-criteria coexistence gate ──
            print(f"    COEXISTENCE GATE ({phase}) — 9 criteria:")
            _train_steps = v5.get("total_train_steps", 0)
            _loss = v5.get("last_loss", 0)
            _buffer = v5.get("buffer_size", 0)
            _cold_floor = v5.get("cold_start_floor", 2000)

            # Criteria #1-3 (training readiness), #4 (loss trend), #6 (rate),
            # #8 (loss magnitude), #9 (multiplier divergence — new).
            # #5 (bus census) + #7 (no regressions) are observational / not gate-
            # gating here; checked separately via /v4/bus-health + brain log.
            g1 = _train_steps > 500
            g2 = _train_steps >= _cold_floor
            g3 = _buffer >= 2000
            g8 = 0 < _loss < 1.0

            history = _fd_history_read(tid)
            g4, losses = _fd_loss_trending_down(history)
            rate_hz, _, rate_no_data = _fd_rate_over_window(history)
            g6 = (rate_hz >= _GATE_RATE_HZ_MIN) and not rate_no_data

            divergence = _fd_max_divergence(mean)
            g9 = divergence >= _GATE_MULTIPLIER_DIVERGENCE_THRESHOLD

            def _icon(b: bool, na: bool = False) -> str:
                return "✓" if b else ("·" if na else "⏳")

            print(f"      {_icon(g1)} #1 V5 trained >500 steps ({_train_steps})")
            print(f"      {_icon(g2)} #2 V5 past cold-start floor ({_train_steps}/{_cold_floor})")
            print(f"      {_icon(g3)} #3 V5 buffer saturated ({_buffer}/2000)")
            if len(losses) < _GATE_LOSS_TREND_SAMPLES:
                print(f"      {_icon(False, na=True)} #4 V5 loss trending down — "
                      f"need {_GATE_LOSS_TREND_SAMPLES} samples, have {len(losses)}")
            else:
                print(f"      {_icon(g4)} #4 V5 loss trending down 3 samples "
                      f"({' → '.join(f'{x:.6f}' for x in losses)})")
            print(f"      ·  #5 Bus census — check `arch_map bus-health` (0 drops expected)")
            if rate_no_data:
                print(f"      {_icon(False, na=True)} #6 Train-step rate >{_GATE_RATE_HZ_MIN:.2f} Hz "
                      f"over 6h — insufficient history (need 2+ samples in window)")
            else:
                print(f"      {_icon(g6)} #6 Train-step rate over 6h = {rate_hz:.3f} Hz "
                      f"(min {_GATE_RATE_HZ_MIN:.2f})")
            print(f"      ·  #7 No regressions — check `arch_map errors --all`")
            print(f"      {_icon(g8)} #8 V5 loss in reasonable range ({_loss:.6f})")
            print(f"      {_icon(g9)} #9 Multiplier divergence max |Δ| = {divergence:.4f} "
                  f"(need ≥ {_GATE_MULTIPLIER_DIVERGENCE_THRESHOLD:.2f})")

            # Gate verdict: all mandatory (#1-4, #6, #8, #9) must pass.
            # #5 + #7 are observational (checked separately) — advisory only.
            mandatory_pass = all([g1, g2, g3, g4, g6, g8, g9])
            if mandatory_pass:
                print(f"    ✅ GATE: all mandatory criteria pass — SAFE TO FLIP publish_enabled=true")
            else:
                all_gates_pass = False
                failing = []
                if not g1: failing.append("#1")
                if not g2: failing.append("#2")
                if not g3: failing.append("#3")
                if not g4: failing.append("#4")
                if not g6: failing.append("#6")
                if not g8: failing.append("#8")
                if not g9: failing.append("#9")
                print(f"    ⛔ GATE: {len(failing)} criteria pending/failing ({', '.join(failing)}) — DO NOT FLIP")
        else:
            print("    V5  (not initialized — rFP #2 not deployed on this Titan?)")
            all_gates_pass = False

    print()
    if gate_check:
        return 0 if all_gates_pass else 1
    return 0


def run_meditation_health(all_titans: bool = False) -> int:
    """rFP_self_healing_meditation_cadence I2: cross-Titan meditation health.

    Queries /v4/meditation/health per Titan. Surfaces:
      • watchdog state (self-test, gap samples, expected interval, stuck time)
      • tracker (count, last_ts, in_meditation)
      • overdue flag + elapsed hours
      • MEDITATION_INFRA_ALERT if ALL queried Titans are overdue within a
        10-min window (infra issue, not per-Titan). Per rFP §5.5 I2 — do NOT
        force-trigger all 3 during infra failure (compounds the problem).
    """
    import time as _time
    titans = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]

    print()
    print("MEDITATION HEALTH — cross-Titan watchdog correlation")
    print("=" * 90)

    overdue_titans: list[tuple[str, float]] = []  # (tid, overdue_since_ts)
    alive_titans = 0

    for tid, base_url in titans:
        print(f"\n  {tid} ({base_url})")
        print("  " + "-" * 80)
        resp = _services_get(base_url, "/v4/meditation/health", timeout=12.0)
        if not resp:
            print("    ✗ endpoint unreachable")
            continue

        payload = resp.get("data", resp) if isinstance(resp, dict) else resp
        if not isinstance(payload, dict):
            print(f"    ✗ unexpected response: {payload}")
            continue
        alive_titans += 1

        tracker = payload.get("tracker", {})
        watchdog = payload.get("watchdog", {})
        overdue = bool(payload.get("overdue", False))

        if "error" in tracker:
            print(f"    tracker: ✗ {tracker['error']}")
        else:
            _now = _time.time()
            _last = float(tracker.get("last_ts", 0) or 0)
            _since_h = (_now - _last) / 3600 if _last > 0 else -1
            print(f"    tracker  count={tracker.get('count', 0):<4} "
                  f"last={_since_h:.1f}h ago  "
                  f"in_meditation={tracker.get('in_meditation', False)}")

        if "error" in watchdog:
            print(f"    watchdog: ✗ {watchdog['error']}")
        else:
            selftest_icon = "✓" if watchdog.get("selftest_pass") else "✗"
            print(f"    watchdog selftest={selftest_icon}  "
                  f"expected={watchdog.get('expected_interval_hours', 0):.1f}h  "
                  f"gaps={watchdog.get('gap_samples', 0)}  "
                  f"zero_promoted_streak={watchdog.get('consecutive_zero_promoted', 0)}")

        if overdue:
            overdue_ts = float(payload.get("overdue_since_ts", 0) or 0)
            overdue_titans.append((tid, overdue_ts))
            print(f"    ⚠ OVERDUE — elapsed={payload.get('overdue_elapsed_hours', 0):.1f}h")
        else:
            print(f"    ✓ cadence OK")

    # Cross-Titan correlation: all overdue within 10-min window = infra issue
    if len(overdue_titans) >= 2 and alive_titans >= 2 and len(overdue_titans) == alive_titans:
        ts_values = [ts for _, ts in overdue_titans if ts > 0]
        if len(ts_values) >= 2:
            ts_spread = max(ts_values) - min(ts_values)
            if ts_spread <= 600:  # 10 minutes
                print()
                print("  🚨 MEDITATION_INFRA_ALERT — all queried Titans overdue within 10-min window")
                print(f"     ({len(overdue_titans)}/{alive_titans} overdue, ts spread {ts_spread:.0f}s)")
                print("     Per rFP I2: do NOT force-trigger all 3 during infra failure")
                print("     Investigate: bus, disk, wallclock, network, shared deps")

    print()
    return 0


def run_voice_diagnostics(all_titans: bool = False, hours: int = 24):
    """rFP_phase5_narrator_evolution §9.3: surface recent X-post grounding
    gate decisions from social_x_telemetry.jsonl.

    Shows two tables per Titan:
      1. Aggregate: total checks | passed | suppressed | enforced-mode?
      2. Recent suppressions (last N, newest first) with topics + confidence
    """
    import json as _json
    import subprocess
    import time as _t

    sources = [("T1", "./data/social_x_telemetry.jsonl", "local")]
    if all_titans:
        sources.append(("T2", "/home/antigravity/projects/titan/data/"
                        "social_x_telemetry.jsonl", "ssh"))
        sources.append(("T3", "/home/antigravity/projects/titan3/data/"
                        "social_x_telemetry.jsonl", "ssh"))

    cutoff = _t.time() - hours * 3600

    print()
    print("TITAN VOICE-DIAGNOSTICS — grounded-only X gate")
    print(f"  (window: last {hours}h — rFP_phase5_narrator_evolution §9)")
    print("=" * 90)

    for tid, path, mode in sources:
        print(f"\n  {tid} — {path}")
        print("  " + "-" * 84)

        if mode == "local":
            try:
                with open(path, "r") as f:
                    raw = f.readlines()
            except FileNotFoundError:
                print("    ✗ telemetry file not found (gateway may not have"
                      " ever posted)")
                continue
        else:
            try:
                proc = subprocess.run(
                    f"ssh root@10.135.0.6 'tail -5000 {path}'",
                    shell=True, capture_output=True, text=True, timeout=20)
                raw = proc.stdout.splitlines()
            except Exception as e:
                print(f"    ✗ SSH read failed: {e}")
                continue

        checks = 0
        passed = 0
        suppressed = 0
        enforced_seen = False
        suppression_rows: list[dict] = []

        for line in raw[-5000:]:
            line = line.strip()
            if not line:
                continue
            try:
                ev = _json.loads(line)
            except Exception:
                continue
            ts = ev.get("timestamp", 0)
            if ts < cutoff:
                continue
            etype = ev.get("event", "")
            if etype == "post_grounding_check":
                checks += 1
                if ev.get("passed"):
                    passed += 1
                if ev.get("enforced"):
                    enforced_seen = True
            elif etype == "post_suppressed_ungrounded":
                suppressed += 1
                suppression_rows.append(ev)
                if ev.get("enforced"):
                    enforced_seen = True

        print(f"    Checks: {checks}   Passed: {passed}   "
              f"Suppressed: {suppressed}   "
              f"Mode: {'ENFORCED' if enforced_seen else 'observability-only'}")

        if suppression_rows:
            print("\n    Recent suppressions (newest first):")
            suppression_rows.sort(key=lambda e: -e.get("timestamp", 0))
            for ev in suppression_rows[:10]:
                age_min = int((_t.time() - ev.get("timestamp", 0)) / 60)
                topics = ev.get("topics", [])
                conf = ev.get("confidence", 0)
                thr = ev.get("threshold", 0)
                ctype = ev.get("catalyst_type", "")
                print(f"      {age_min:>4}m ago  conf={conf:.2f}  "
                      f"thr={thr:.2f}  [{ctype}]  topics={topics}")

    print()
    print("=" * 90)


def run_errors(all_titans: bool = False):
    """Scan Titan logs for errors, group and count them."""
    import subprocess
    import re as _re

    # Known noise patterns to exclude
    NOISE = [
        "network", "RPC", "get_balance", "tweepy", "httpx",
        "ConnectionError", "ReadTimeout", "ConnectTimeout",
        "RequestsDependencyWarning", "urllib3", "charset_normalizer",
    ]
    noise_pattern = "|".join(NOISE)

    log_sources = [("T1", "/tmp/titan_brain.log", "local")]
    if all_titans:
        # 2026-04-08 audit fix: T2/T3 logs are titan2_brain.log / titan3_brain.log
        # NOT titan_brain_t2.log / titan_brain_t3.log (those don't exist).
        # Was producing false-negative "No errors" because tail failed silently.
        log_sources.append(("T2", "/tmp/titan2_brain.log", "ssh"))
        log_sources.append(("T3", "/tmp/titan3_brain.log", "ssh"))

    print()
    print("TITAN ERROR SUMMARY")
    print("=" * 90)

    for tid, log_path, mode in log_sources:
        print(f"\n  {tid} — {log_path}")
        print("  " + "-" * 80)

        # Get last 2000 lines of log
        if mode == "local":
            try:
                with open(log_path, "r", errors="replace") as f:
                    lines = f.readlines()[-2000:]
            except FileNotFoundError:
                print(f"    ✗ Log file not found")
                continue
        else:
            try:
                proc = subprocess.run(
                    f"ssh root@10.135.0.6 'tail -2000 {log_path}'",
                    shell=True, capture_output=True, text=True, timeout=15)
                lines = proc.stdout.splitlines()
            except Exception as e:
                print(f"    ✗ Cannot read log: {e}")
                continue

        # Filter for ERROR/WARNING, exclude noise
        errors = {}
        warnings = {}
        for line in lines:
            if "[ERROR]" in line:
                # Skip noise
                if any(n.lower() in line.lower() for n in NOISE):
                    continue
                # Extract error signature (module + message type)
                m = _re.search(r'\[ERROR\]\s*\[([^\]]+)\]\s*(.*?)(?::\s|$)', line)
                if m:
                    sig = f"[{m.group(1)}] {m.group(2)[:80]}"
                else:
                    sig = line.strip()[-100:]
                if sig not in errors:
                    errors[sig] = {"count": 0, "first": line[:8], "last": line[:8]}
                errors[sig]["count"] += 1
                errors[sig]["last"] = line[:8]
            elif "[WARNING]" in line:
                if any(n.lower() in line.lower() for n in NOISE):
                    continue
                m = _re.search(r'\[WARNING\]\s*\[([^\]]+)\]\s*(.*?)(?::\s|$)', line)
                if m:
                    sig = f"[{m.group(1)}] {m.group(2)[:80]}"
                else:
                    sig = line.strip()[-100:]
                if sig not in warnings:
                    warnings[sig] = {"count": 0, "first": line[:8], "last": line[:8]}
                warnings[sig]["count"] += 1
                warnings[sig]["last"] = line[:8]

        if errors:
            print(f"    ERRORS ({sum(e['count'] for e in errors.values())} total, "
                  f"{len(errors)} unique):")
            for sig, info in sorted(errors.items(), key=lambda x: -x[1]["count"]):
                print(f"      {info['count']:5d}x  {info['first']}-{info['last']}  {sig}")
        else:
            print("    ✓ No errors (last 2000 lines)")

        if warnings:
            print(f"    WARNINGS ({sum(w['count'] for w in warnings.values())} total, "
                  f"{len(warnings)} unique):")
            for sig, info in sorted(warnings.items(), key=lambda x: -x[1]["count"])[:10]:
                print(f"      {info['count']:5d}x  {info['first']}-{info['last']}  {sig}")
            if len(warnings) > 10:
                print(f"      ... and {len(warnings) - 10} more warning types")
        else:
            print("    ✓ No warnings (last 2000 lines)")

    print()
    print("=" * 90)


def run_cgn_signals_audit(all_titans: bool = False,
                          audit_pattern: bool = False):
    """META-CGN signal audit — validates producer wiring vs SIGNAL_TO_PRIMITIVE.

    Scans source for emit_meta_cgn_signal(...) calls, extracts
    (consumer, event_type) from each call site, verifies each has a
    matching SIGNAL_TO_PRIMITIVE entry. Reports:
      - ORPHAN producers (call site with no mapping — would be silently
        dropped by consumer; this is the 2026-04-14 Phase 2 bug pattern)
      - UNUSED mappings (SIGNAL_TO_PRIMITIVE entry with no producer —
        dead code, harmless but clutter)

    Also queries /v4/bus-health on each reachable Titan to report:
      - live emission rates per producer
      - queue fill fractions
      - orphan counts observed at runtime
      - overall state (healthy / warning / critical)

    --audit-pattern (2026-04-19, META-CGN-PRODUCER-PATTERN): for each
    producer site, verify an EdgeDetector observe call appears within
    ±30 lines. Invariant from TUNING-016: every META-CGN producer MUST
    gate emission via EdgeDetector (no continuous emission on every
    event — per bus_health.py discrete-transition invariant). Sites
    without edge gating are flagged as PATTERN-VIOLATION.
    """
    import re
    from pathlib import Path

    print()
    print("META-CGN SIGNAL AUDIT — producer wiring vs SIGNAL_TO_PRIMITIVE")
    print("=" * 90)

    # ── 1. Scan source for emit_meta_cgn_signal call sites ──
    producer_sites = []   # list of (file, line, consumer, event_type)
    call_pattern = re.compile(r"emit_meta_cgn_signal\s*\(")
    consumer_pattern = re.compile(
        r'consumer\s*=\s*["\']([^"\']+)["\']'
    )
    event_type_pattern = re.compile(
        r'event_type\s*=\s*["\']([^"\']+)["\']'
    )

    # Pattern-audit state: map of (file, line) → (edge_gated, gate_hint)
    pattern_audit_results: dict = {}
    # Regex for the EdgeDetector invariant — detects references to the
    # class name, instance names, or the observe() call in the ±30-line
    # preceding window.
    edge_gate_re = re.compile(
        r"EdgeDetector|edge_detector|edge_det\b|\.observe\s*\(|"
        r"_ed_observe|_edge_result"
    )

    titan_plugin_root = Path("titan_plugin")
    # Skip bus.py — it contains the helper DEFINITION (function signature +
    # type constant), not producer call sites. Including it would produce
    # false-positive "orphans" for the framework itself.
    for py_file in titan_plugin_root.rglob("*.py"):
        if py_file.name == "bus.py":
            continue
        try:
            src_lines = py_file.read_text().splitlines()
        except Exception:
            continue
        for i, line in enumerate(src_lines, 1):
            if call_pattern.search(line):
                # collect next 20 lines as one block (kwargs may span multi-line)
                block = "\n".join(src_lines[i - 1:i + 20])
                cm = consumer_pattern.search(block)
                em = event_type_pattern.search(block)
                if cm and em:
                    producer_sites.append((py_file, i, cm.group(1), em.group(1)))
                else:
                    producer_sites.append((py_file, i, "???", "???"))
                # ── EdgeDetector-gate heuristic ─────────────────────
                # Scan ±30 lines around the emit to check the producer
                # is preceded by an edge-detector observe/gate. A strict
                # AST check would be more precise but this heuristic
                # covers the common patterns: local variable referencing
                # an EdgeDetector instance, method call, or import.
                if audit_pattern:
                    ctx_start = max(0, i - 30)
                    ctx_end = min(len(src_lines), i + 5)
                    ctx = "\n".join(src_lines[ctx_start:ctx_end])
                    has_gate = bool(edge_gate_re.search(ctx))
                    pattern_audit_results[(py_file, i)] = has_gate

    # ── 2. Load SIGNAL_TO_PRIMITIVE from meta_cgn.py ──
    mapping_keys = set()
    try:
        sys.path.insert(0, str(Path.cwd()))
        from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE
        mapping_keys = set(SIGNAL_TO_PRIMITIVE.keys())
    except Exception as e:
        print(f"  ✗ Could not load SIGNAL_TO_PRIMITIVE: {e}")
        return

    # ── 3. Classify each site ──
    orphan_sites = []
    mapped_sites = []
    used_keys = set()
    for site in producer_sites:
        f, ln, c, et = site
        if (c, et) in mapping_keys:
            mapped_sites.append(site)
            used_keys.add((c, et))
        else:
            orphan_sites.append(site)

    unused_keys = mapping_keys - used_keys

    # ── 4. Report ──
    print()
    print(f"  {len(producer_sites)} emit_meta_cgn_signal call site(s) found in titan_plugin/")
    print(f"  {len(mapping_keys)} (consumer, event_type) entries in SIGNAL_TO_PRIMITIVE")
    print()

    if orphan_sites:
        print(f"  ✗ {len(orphan_sites)} ORPHAN producer site(s) — no SIGNAL_TO_PRIMITIVE mapping:")
        for f, ln, c, et in orphan_sites:
            print(f"    {rel(f)}:{ln} — ({c}, {et}) ← add to SIGNAL_TO_PRIMITIVE or remove this site")
    else:
        print("  ✓ No orphan producer sites — every emit_meta_cgn_signal call has a mapping")

    print()
    if unused_keys:
        print(f"  ℹ {len(unused_keys)} unused mapping(s) in SIGNAL_TO_PRIMITIVE (no producer emits):")
        for c, et in sorted(unused_keys):
            print(f"    ({c}, {et}) — consumer ready, producer not yet wired")
    else:
        print("  ✓ Every SIGNAL_TO_PRIMITIVE mapping has at least one producer")

    print()
    print(f"  ✓ {len(mapped_sites)} correctly-wired producer site(s)")

    # ── 4b. EdgeDetector pattern audit (2026-04-19) ──
    if audit_pattern:
        print()
        print("  ── EDGE-DETECTOR PATTERN AUDIT ──")
        print("  (producers must gate emit via EdgeDetector per TUNING-016)")
        violations = []
        compliant = []
        for site in mapped_sites:
            f, ln, c, et = site
            gated = pattern_audit_results.get((f, ln), None)
            if gated is False:
                violations.append(site)
            elif gated is True:
                compliant.append(site)
        if violations:
            print(f"  ✗ {len(violations)} PATTERN-VIOLATION site(s) — no "
                  "EdgeDetector observe() / gate reference within ±30 lines:")
            for f, ln, c, et in violations:
                print(f"    {rel(f)}:{ln} — ({c}, {et}) ← add EdgeDetector "
                      "observe() gate before emit_meta_cgn_signal")
            print()
            print("  Rationale: bus_health invariant requires discrete state "
                  "transitions,")
            print("  not continuous per-event emission. Unbounded producers "
                  "blow the")
            print("  0.5 Hz global rate budget. See TUNING-016 + rFP_meta_cgn_v3"
                  " § 7.")
        else:
            print(f"  ✓ All {len(compliant)} mapped producer(s) show an "
                  "EdgeDetector gate in context")
        print()

    # ── 5. Live runtime stats if Titans reachable ──
    print()
    print("  ── LIVE BUS-HEALTH (from /v4/bus-health) ──")
    import requests
    endpoints = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        endpoints.append(("T2", "http://10.135.0.6:7777"))
        endpoints.append(("T3", "http://10.135.0.6:7778"))
    for name, base in endpoints:
        try:
            r = requests.get(f"{base}/v4/bus-health", timeout=5)
            if r.status_code != 200:
                print(f"    {name} ✗ HTTP {r.status_code}")
                continue
            data = r.json().get("data", {})
            st = data.get("overall_state", "?")
            rate = data.get("total_emission_rate_1min_hz", 0)
            budget = data.get("rate_budget_hz", 0.5)
            max_q = data.get("max_queue_fraction", 0)
            orphan_ct = data.get("orphans", {}).get("total_count", 0)
            orphan_tuples = data.get("orphans", {}).get("unique_tuples", [])
            marker = {"healthy": "✓", "warning": "⚠", "critical": "✗"}.get(st, "?")
            print(f"    {name} {marker} state={st:8s} rate={rate:.3f}/{budget}Hz  max_queue={max_q:.2f}  orphans={orphan_ct} ({len(orphan_tuples)} unique)")
            if orphan_tuples:
                for tup in orphan_tuples:
                    print(f"         ↳ orphan: {tup}")
        except Exception as e:
            print(f"    {name} ✗ unreachable: {e}")

    print()
    print("=" * 90)


def run_producers_diagnostics(argv: list) -> None:
    """META-CGN producer observability — live + historical across all Titans.

    Combines two data sources:
      - Live /v4/bus-health (in-memory, resets on restart, 5-min window)
      - Persistent data/meta_cgn_emissions.jsonl (Guardian drain-loop append,
        survives restarts, full history)

    Flags (any order, --all implied):
      --history N        Read last N days of JSONL + show per-producer counts
      --since <ISO>      Start timestamp (e.g. 2026-04-15T00:00:00)
      --producer <name>  Filter to "src/event_type" tuple (e.g. msl/concept_grounded)
      --json             Machine-readable output
      --expected         Flag wired producers with 0 lifetime emissions

    Paths per Titan:
      T1: ./data/meta_cgn_emissions.jsonl              (local)
      T2: /home/antigravity/projects/titan/data/       (ssh root@10.135.0.6)
      T3: /home/antigravity/projects/titan3/data/      (ssh root@10.135.0.6)
    """
    import json as _json
    import datetime as _dt
    import subprocess

    # ── Parse args ─────────────────────────────────────────────────
    history_days = None
    since_iso = None
    producer_filter = None
    want_json = "--json" in argv
    want_expected = "--expected" in argv
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--history" and i + 1 < len(argv):
            try:
                history_days = int(argv[i + 1])
            except ValueError:
                print(f"  ✗ Invalid --history value: {argv[i + 1]}")
                return
            i += 2
            continue
        if a == "--since" and i + 1 < len(argv):
            since_iso = argv[i + 1]
            i += 2
            continue
        if a == "--producer" and i + 1 < len(argv):
            producer_filter = argv[i + 1]
            i += 2
            continue
        i += 1

    # Compute since timestamp (prefer explicit --since, else --history days ago)
    since_ts = 0.0
    if since_iso:
        try:
            since_ts = _dt.datetime.fromisoformat(since_iso).timestamp()
        except ValueError:
            print(f"  ✗ Invalid --since ISO format: {since_iso}")
            return
    elif history_days is not None:
        since_ts = (_dt.datetime.now() - _dt.timedelta(days=history_days)).timestamp()

    titans = [
        ("T1", "http://127.0.0.1:7777", None, "./data/meta_cgn_emissions.jsonl"),
        ("T2", "http://10.135.0.6:7777", "root@10.135.0.6",
         "/home/antigravity/projects/titan/data/meta_cgn_emissions.jsonl"),
        ("T3", "http://10.135.0.6:7778", "root@10.135.0.6",
         "/home/antigravity/projects/titan3/data/meta_cgn_emissions.jsonl"),
    ]

    # ── 1. LIVE snapshot from /v4/bus-health ──────────────────────
    import requests as _requests
    live_stats = {}  # name → {producers: [...], overall: str, reachable: bool}
    for name, url, _ssh, _path in titans:
        try:
            r = _requests.get(url + "/v4/bus-health", timeout=5)
            if r.status_code == 200:
                payload = r.json()
                if payload.get("status") == "ok":
                    data = payload.get("data", {})
                    live_stats[name] = {
                        "reachable": True,
                        "overall": data.get("overall_state", "?"),
                        "rate_1m": data.get("total_emission_rate_1min_hz", 0.0),
                        "rate_budget": data.get("rate_budget_hz", 0.5),
                        "orphans": data.get("orphans", {}).get("total_count", 0),
                        "producers": data.get("producers", []),
                    }
                else:
                    live_stats[name] = {"reachable": False, "error": "bad response"}
            else:
                live_stats[name] = {"reachable": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            live_stats[name] = {"reachable": False, "error": str(e)}

    # ── 2. HISTORICAL from JSONL (if --history or --since) ────────
    history = {}  # name → list of event dicts (filtered by since_ts + producer_filter)
    need_history = history_days is not None or since_iso is not None
    if need_history:
        for name, _url, ssh_host, path in titans:
            events = []
            try:
                if ssh_host:
                    # Remote: cat file over ssh; tolerate missing file
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", ssh_host,
                         f"cat {path} 2>/dev/null || true"],
                        capture_output=True, text=True, timeout=30,
                    )
                    raw = result.stdout
                else:
                    try:
                        with open(path) as f:
                            raw = f.read()
                    except FileNotFoundError:
                        raw = ""
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if e.get("ts", 0) < since_ts:
                        continue
                    if producer_filter:
                        tup = f"{e.get('src','')}/{e.get('event_type','')}"
                        if tup != producer_filter:
                            continue
                    events.append(e)
            except Exception as e:
                events = []
                print(f"  ✗ {name} history read failed: {e}")
            history[name] = events

    # ── 3. Render ─────────────────────────────────────────────────
    if want_json:
        out = {"live": live_stats, "history": history if need_history else None}
        print(_json.dumps(out, indent=2, default=str))
        return

    print()
    print("META-CGN PRODUCERS — live + historical observability")
    print("=" * 90)

    # Live section
    print()
    print("LIVE SNAPSHOT (from /v4/bus-health — resets on restart)")
    print("-" * 90)
    for name, _url, _ssh, _path in titans:
        s = live_stats.get(name, {})
        if not s.get("reachable"):
            print(f"  {name}  ✗ unreachable: {s.get('error','?')}")
            continue
        icon = "✓" if s["overall"] == "healthy" else "⚠"
        print(f"  {name}  {icon} state={s['overall']}  rate_1m={s['rate_1m']:.4f}/{s['rate_budget']}Hz  orphans={s['orphans']}")
        prods = s["producers"]
        if not prods:
            print(f"      (no producers have emitted since last restart)")
            continue
        for p in prods:
            tup = f"{p['src']}/{p['event_type']}"
            if producer_filter and tup != producer_filter:
                continue
            print(f"      {tup:42s} total={p['total_emissions']:5d}  1m={p['count_1min']:3d}  5m={p['count_5min']:3d}  drops={p['rate_drops']}")

    # History section
    if need_history:
        print()
        window_label = (f"last {history_days}d" if history_days is not None
                        else f"since {since_iso}")
        print(f"HISTORICAL ({window_label}, from data/meta_cgn_emissions.jsonl)")
        print("-" * 90)
        for name, _url, _ssh, _path in titans:
            evs = history.get(name, [])
            if not evs:
                print(f"  {name}  (no events in window — log may not exist yet if pre-deploy)")
                continue
            # Per-producer aggregate
            agg = {}  # tup → {"count": int, "first": ts, "last": ts, "domains": set}
            for e in evs:
                tup = f"{e['src']}/{e['event_type']}"
                a = agg.setdefault(tup, {"count": 0, "first": e["ts"], "last": e["ts"], "domains": set()})
                a["count"] += 1
                a["first"] = min(a["first"], e["ts"])
                a["last"] = max(a["last"], e["ts"])
                if e.get("domain"):
                    a["domains"].add(e["domain"])
            print(f"  {name}  ({len(evs)} events, {len(agg)} unique producers)")
            for tup in sorted(agg.keys()):
                a = agg[tup]
                first = _dt.datetime.fromtimestamp(a["first"]).strftime("%m-%d %H:%M")
                last = _dt.datetime.fromtimestamp(a["last"]).strftime("%m-%d %H:%M")
                doms = ",".join(sorted(a["domains"])[:5]) if a["domains"] else "-"
                if len(a["domains"]) > 5:
                    doms += f"+{len(a['domains'])-5}more"
                print(f"      {tup:42s} count={a['count']:5d}  first={first}  last={last}  domains=[{doms}]")

    # Expected producers section (--expected flag)
    if want_expected:
        print()
        print("WIRED PRODUCERS WITH ZERO LIFETIME EMISSIONS (since log start)")
        print("-" * 90)
        try:
            sys.path.insert(0, str(Path.cwd()))
            from titan_plugin.logic.meta_cgn import SIGNAL_TO_PRIMITIVE
            wired_tuples = set()  # (src, event_type) — extract from emit_meta_cgn_signal call sites
            import re
            call_pat = re.compile(r"emit_meta_cgn_signal\s*\(")
            src_pat = re.compile(r'src\s*=\s*["\']([^"\']+)["\']')
            et_pat = re.compile(r'event_type\s*=\s*["\']([^"\']+)["\']')
            for pyf in Path("titan_plugin").rglob("*.py"):
                if pyf.name == "bus.py":
                    continue
                try:
                    text = pyf.read_text()
                except Exception:
                    continue
                for m in call_pat.finditer(text):
                    window = text[m.start():m.start() + 600]
                    sm, em = src_pat.search(window), et_pat.search(window)
                    if sm and em:
                        wired_tuples.add((sm.group(1), em.group(1)))
            # Cross-check against aggregate emissions across all Titans
            silent = []
            for src, et in wired_tuples:
                total = 0
                for name in live_stats:
                    for p in live_stats[name].get("producers", []):
                        if p["src"] == src and p["event_type"] == et:
                            total += p["total_emissions"]
                if total == 0:
                    silent.append((src, et))
            if silent:
                for src, et in sorted(silent):
                    print(f"  ✗ {src}/{et} — wired but 0 lifetime emissions across all 3 Titans")
            else:
                print("  ✓ All wired producers have emitted at least once (across 3 Titans)")
        except Exception as e:
            print(f"  ✗ Expected-check failed: {e}")

    print()
    print("=" * 90)


def run_preflight():
    """Comprehensive pre-session-start test — mandatory step 0.

    Built 2026-04-14 after Phase E session where 6 audits passed while
    T1 was in a multi-day crash-loop. Runs 9 checks in sequence and
    produces one unified PASS/WARN/FAIL verdict. Any HARD-FAIL halts
    further work until resolved.

    Checks (runs in ~60-90s end-to-end):
      0. stability      — 24h watchdog restarts (HARD)
      1. async-blocks   — CRITICAL=0 on every scan (HARD)
      2. health         — arch_map health --all (HARD if T1 <20/23)
      3. endpoints      — /health, /v4/bus-health, /v3/trinity timing (HARD)
      4. services       — teacher/ARC/persona/events (WARN)
      5. bus-health     — API endpoint reports state=healthy (WARN)
      6. timechain      — fork integrity (HARD)
      7. cgn            — consumers registered + HAOV forming (WARN)
      8. known-issues   — no new CRITICAL entries (WARN)

    HARD failures set exit code 1 and print HALT banner.
    WARN failures print orange triangles but allow continuation.

    Designed to be the FIRST thing run in any new Claude Code session.
    See memory/session_startup_protocol.md — runs as mandatory Step 0.
    """
    import subprocess
    import requests

    hard_fails = []
    warnings = []
    passes = []

    def _run_cmd(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=str(PROJECT_ROOT))
            return r.returncode, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            return -1, "timeout"
        except Exception as e:
            return -1, f"error: {e}"

    print()
    print("╔" + "═" * 88 + "╗")
    print("║" + "  TITAN PREFLIGHT — comprehensive session-start test".ljust(88) + "║")
    print("╚" + "═" * 88 + "╝")

    # ── 0. Stability ──
    print("\n[0/9] STABILITY — 24h watchdog restart density")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "stability",
                        "--hours=24"], timeout=45)
    if "CRASH-LOOP DETECTED" in out or "CRASH_LOOP" in out:
        hard_fails.append("stability: CRASH-LOOP detected on ≥1 Titan")
        print("  🔴 CRASH-LOOP — HALT ALL FEATURE WORK")
    elif "OVERALL: STABLE" in out:
        passes.append("stability: STABLE")
        print("  ✓ STABLE — 0 restarts across T1/T2/T3")
    else:
        # Parse restart counts
        import re
        m = re.search(r"OVERALL:\s+(\w+)", out)
        verdict = m.group(1) if m else "UNKNOWN"
        if verdict == "FRAGILE":
            warnings.append(f"stability: {verdict}")
            print(f"  ⚠ {verdict}")
        else:
            passes.append(f"stability: {verdict}")
            print(f"  ✓ {verdict}")

    # ── 1. Async-block scanner ──
    print("\n[1/9] ASYNC-BLOCKS — sync I/O reachable from async")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "async-blocks"], timeout=30)
    import re
    m = re.search(r"CRITICAL=(\d+)\s+HIGH=(\d+)", out)
    if m:
        crit, hi = int(m.group(1)), int(m.group(2))
        if crit > 0:
            hard_fails.append(f"async-blocks: {crit} CRITICAL sites")
            print(f"  🔴 CRITICAL={crit} — must wrap in asyncio.to_thread before deploy")
        else:
            passes.append(f"async-blocks: CRITICAL=0, HIGH={hi}")
            print(f"  ✓ CRITICAL=0  (HIGH={hi} — follow-up batch)")
    else:
        warnings.append("async-blocks: scanner output unparsed")
        print("  ⚠ scanner output unparseable")

    # ── 2. Health per-Titan ──
    print("\n[2/9] HEALTH — per-Titan system health (23 checks each)")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "health", "--all"], timeout=60)
    # Parse "PASSED: X/23  WARNINGS: Y  FAILED: Z"
    titan_results = re.findall(r"TITAN LIVE HEALTH CHECK — (\w+).*?PASSED:\s+(\d+)/23\s+WARNINGS:\s+(\d+)\s+FAILED:\s+(\d+)",
                                out, re.DOTALL)
    for tid, passed, warn, failed in titan_results:
        p, w, f = int(passed), int(warn), int(failed)
        if tid == "T1" and p < 20:
            hard_fails.append(f"health {tid}: only {p}/23 passed")
            print(f"  🔴 {tid}: {p}/23 passed, {f} failed — T1 critical")
        elif f > 0:
            warnings.append(f"health {tid}: {f} failed")
            print(f"  ⚠ {tid}: {p}/23 passed, {w} warnings, {f} failed")
        else:
            passes.append(f"health {tid}: {p}/23")
            print(f"  ✓ {tid}: {p}/23 passed, {w} warnings")

    # ── 3. Endpoint sweep ──
    print("\n[3/9] ENDPOINTS — /health, /v4/bus-health response time")
    t1_endpoints = [
        ("T1 /health", "http://127.0.0.1:7777/health"),
        ("T1 /v4/bus-health", "http://127.0.0.1:7777/v4/bus-health"),
        ("T2 /health", "http://10.135.0.6:7777/health"),
        ("T3 /health", "http://10.135.0.6:7778/health"),
    ]
    for name, url in t1_endpoints:
        try:
            import time as _t
            t0 = _t.time()
            r = requests.get(url, timeout=5)
            ms = int((_t.time() - t0) * 1000)
            if r.status_code == 200 and ms < 1000:
                passes.append(f"{name}: {ms}ms")
                print(f"  ✓ {name}: HTTP 200 in {ms}ms")
            elif r.status_code == 200:
                warnings.append(f"{name}: slow ({ms}ms)")
                print(f"  ⚠ {name}: HTTP 200 in {ms}ms (>1s)")
            else:
                hard_fails.append(f"{name}: HTTP {r.status_code}")
                print(f"  🔴 {name}: HTTP {r.status_code}")
        except Exception as e:
            if "T1" in name:
                hard_fails.append(f"{name}: {type(e).__name__}")
                print(f"  🔴 {name}: {type(e).__name__} — T1 unreachable")
            else:
                warnings.append(f"{name}: {type(e).__name__}")
                print(f"  ⚠ {name}: {type(e).__name__}")

    # ── 4. Services ──
    print("\n[4/9] SERVICES — teacher/ARC/persona/events per Titan")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "services"], timeout=30)
    m = re.search(r"TOTAL:\s+(\d+)/(\d+)\s+OK\s+\|\s+(\d+)\s+warnings\s+\|\s+(\d+)\s+failures", out)
    if m:
        ok, total, warn, fail = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if fail > 0:
            warnings.append(f"services: {fail} failures")
            print(f"  ⚠ {ok}/{total} OK, {fail} failed")
        else:
            passes.append(f"services: {ok}/{total}")
            print(f"  ✓ {ok}/{total} OK ({warn} warnings)")
    else:
        warnings.append("services: output unparsed")
        print("  ⚠ services diagnostics output unparseable")

    # ── 5. Bus health ──
    print("\n[5/9] BUS HEALTH — /v4/bus-health per Titan")
    for tid, url in [("T1", "http://127.0.0.1:7777"),
                      ("T2", "http://10.135.0.6:7777"),
                      ("T3", "http://10.135.0.6:7778")]:
        try:
            r = requests.get(f"{url}/v4/bus-health", timeout=5)
            if r.status_code == 200:
                state = r.json().get("data", {}).get("overall_state", "?")
                if state == "healthy":
                    passes.append(f"bus {tid}: healthy")
                    print(f"  ✓ {tid}: bus state=healthy")
                else:
                    warnings.append(f"bus {tid}: {state}")
                    print(f"  ⚠ {tid}: bus state={state}")
        except Exception as e:
            warnings.append(f"bus {tid}: {type(e).__name__}")
            print(f"  ⚠ {tid}: {type(e).__name__}")

    # ── 6. TimeChain integrity ──
    print("\n[6/9] TIMECHAIN — fork integrity")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "timechain", "--all"], timeout=30)
    valid_count = out.count("✓ ALL FORKS VALID")
    if valid_count >= 1:
        passes.append(f"timechain: {valid_count} Titans valid")
        print(f"  ✓ {valid_count} Titans with all forks valid")
    else:
        hard_fails.append("timechain: no valid-forks line found")
        print("  🔴 TimeChain integrity cannot be confirmed")

    # ── 7. CGN pipeline ──
    print("\n[7/9] CGN — consumers + HAOV hypothesis testing")
    rc, out = _run_cmd([sys.executable, str(Path(__file__)), "verify", "cgn", "--all"], timeout=45)
    healthy = out.count("✓ ALL STAGES HEALTHY")
    if healthy >= 1:
        passes.append(f"cgn: {healthy} Titans healthy")
        print(f"  ✓ {healthy} Titans with full CGN pipeline")
    else:
        warnings.append("cgn: pipeline not fully healthy")
        print("  ⚠ CGN pipeline partial (see verify output)")

    # ── 8. Known issues ──
    print("\n[8/9] KNOWN ISSUES — registry scan for new CRITICAL")
    try:
        kn_path = PROJECT_ROOT / "memory" / "known_issues.md"
        if kn_path.exists():
            content = kn_path.read_text()
            # Count active critical items (I-NNN not marked RESOLVED)
            import re as _re
            active_crit = len(_re.findall(r"### I-\d+ — [^\n]*\[(HIGH PRIORITY|CRITICAL)", content))
            if active_crit <= 3:
                passes.append(f"known issues: {active_crit} active HIGH/CRIT")
                print(f"  ✓ {active_crit} active HIGH/CRITICAL issues (baseline)")
            else:
                warnings.append(f"known issues: {active_crit} active HIGH/CRIT")
                print(f"  ⚠ {active_crit} active HIGH/CRITICAL issues")
        else:
            warnings.append("known issues: file not found")
            print("  ⚠ memory/known_issues.md not found")
    except Exception as e:
        warnings.append(f"known issues: {e}")
        print(f"  ⚠ known_issues scan error: {e}")

    # ── Final verdict ──
    print()
    print("=" * 90)
    if hard_fails:
        print(f"  🛑 PREFLIGHT FAILED — {len(hard_fails)} HARD fail(s), "
              f"{len(warnings)} warning(s)")
        print()
        print("  HALT — resolve these before feature work:")
        for f in hard_fails:
            print(f"    • {f}")
        if warnings:
            print()
            print("  Warnings (deferrable):")
            for w in warnings[:10]:
                print(f"    • {w}")
        print()
        print("=" * 90)
        sys.exit(1)
    elif warnings:
        print(f"  ⚠ PREFLIGHT PASSED WITH WARNINGS — {len(passes)} passes, "
              f"{len(warnings)} warnings")
        print()
        print("  Warnings (deferrable):")
        for w in warnings[:10]:
            print(f"    • {w}")
    else:
        print(f"  ✓ PREFLIGHT PASSED — all {len(passes)} checks green")
    print()
    print("=" * 90)


def run_async_blocks_scan():
    """Scan titan_plugin/ for sync I/O calls reachable from async functions.

    Built 2026-04-14 after py-spy diagnosis found 3 separate latent bugs
    (web_search.status sync httpx, coding_sandbox.status subprocess.run,
    snapshot_to_arweave Zstd-19 on event loop) — all hung the FastAPI
    Observatory API. Each was a sync I/O call called transitively from
    an `async def` endpoint. None were caught by yesterday's audits.

    Strategy:
      1. Parse all .py files under titan_plugin/ with AST
      2. For each function, classify: async, sync, or sync-called-from-async
         (transitive reachability via function call graph)
      3. Detect dangerous patterns at every call site:
           - httpx.{get,post,put,delete,patch,head,options,request}  (sync)
           - requests.{get,post,...}
           - urllib.request.urlopen
           - subprocess.{run,call,check_output,check_call,Popen}
           - sqlite3.connect / duckdb.connect (only flag if in async path)
           - zstandard.*compress, gzip.compress, tarfile.open(mode="w*")
      4. Report by severity:
           CRITICAL — sync I/O in `async def` (immediate event-loop block)
           HIGH     — sync I/O in function called from `async def`
           MEDIUM   — sync I/O in module that exposes `async def` somewhere
           LOW      — sync I/O in worker subprocess (separate event loop)

    Output sorted by severity. Each row: file:line  function  pattern.
    Designed to be run as part of session_startup_protocol so async-block
    bugs cannot reach production undetected.

    v2 (2026-04-14):
      - Detects `asyncio.to_thread(fn)`, `loop.run_in_executor(_, fn)`,
        `threading.Thread(target=fn)` wrappers and suppresses transitive
        flagging across thread boundaries.
      - Detects `asyncio.to_thread(lambda: …)` — sync I/O inside the
        lambda body is isolated from the event loop.
      - Supports `# noqa: async-block` inline suppression for verified
        false positives.
      - Call-graph BFS follows only direct (non-threaded) edges, so
        `reachable_from_async` now means genuinely reachable on the loop.
    """
    import ast
    import os
    from collections import defaultdict

    SCAN_ROOT = Path(__file__).parent.parent / "titan_plugin"
    # Worker modules — different process, separate event loop. Sync I/O
    # there only blocks that worker, not the FastAPI loop.
    WORKER_FILES = {
        "modules/spirit_loop.py", "modules/spirit_worker.py",
        "modules/llm_worker.py", "modules/body_worker.py",
        "modules/mind_worker.py", "modules/media_worker.py",
        "modules/memory_worker.py", "modules/language_worker.py",
        "modules/cgn_worker.py", "modules/rl_worker.py",
        "modules/knowledge_worker.py", "modules/timechain_worker.py",
    }

    # ── Pattern detectors (AST node matchers) ──────────────────────────
    #
    # v3 (2026-04-14): detectors resolve the module name through the file's
    # alias table so `import sqlite3 as _af_sql; _af_sql.connect(...)` is
    # detected as sqlite3.connect. v2 only matched literal module names,
    # which silently ignored 7+ CRITICAL sites in dashboard.py that used
    # aliased imports as a (misguided) locality optimization.

    def _resolve_module(node_value: ast.AST, aliases: dict[str, str]) -> str | None:
        """Return the canonical module name for `x` in `x.method()`.

        Handles two forms:
          (a) `ast.Name` — `_af_sql.connect` → look up _af_sql in aliases →
              "sqlite3".
          (b) `ast.Attribute` chains — `urllib.request.urlopen` → resolve
              the leftmost Name ("urllib") through aliases, rebuild.
        Returns None if the expression isn't a simple module reference
        (e.g., a method call on an instance — `self.db.connect()`).
        """
        if isinstance(node_value, ast.Name):
            name = node_value.id
            return aliases.get(name, name)  # literal if no alias
        if isinstance(node_value, ast.Attribute):
            # Resolve the leftmost Name and reconstruct the dotted path.
            # `urllib.request` when value=Attribute(value=Name("urllib"),attr="request")
            parts: list[str] = [node_value.attr]
            cur: ast.AST = node_value.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                root = aliases.get(cur.id, cur.id)
                parts.append(root)
                return ".".join(reversed(parts))
        return None

    def _is_sync_http(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
        """httpx.get(...) / requests.get(...) etc — but NOT httpx.AsyncClient."""
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        if method not in ("get", "post", "put", "delete", "patch", "head",
                          "options", "request"):
            return None
        mod = _resolve_module(node.func.value, aliases)
        if mod in ("httpx", "requests"):
            return ("SYNC_HTTP", f"{mod}.{method}")
        return None

    def _is_urlopen(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
        """urllib.request.urlopen(...)."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen":
            return ("SYNC_HTTP", "urllib.urlopen")
        if isinstance(node.func, ast.Name):
            # `from urllib.request import urlopen as _uro` — `_uro(...)`
            resolved = aliases.get(node.func.id)
            if resolved == "urllib.request.urlopen" or node.func.id == "urlopen":
                return ("SYNC_HTTP", "urlopen")
        return None

    def _is_subprocess(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
        """subprocess.run / call / check_output / Popen."""
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        if method not in ("run", "call", "check_call", "check_output", "Popen"):
            return None
        mod = _resolve_module(node.func.value, aliases)
        if mod == "subprocess":
            return ("SUBPROCESS", f"subprocess.{method}")
        return None

    def _is_compression(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
        """zstandard.ZstdCompressor().compress / gzip.compress / tarfile.open(mode='w*')."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "compress":
                return ("COMPRESSION", "<obj>.compress")
            if node.func.attr == "open":
                mod = _resolve_module(node.func.value, aliases)
                if mod == "tarfile":
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, str) and kw.value.value.startswith("w"):
                                return ("COMPRESSION", "tarfile.open(write)")
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        if isinstance(node.args[1].value, str) and node.args[1].value.startswith("w"):
                            return ("COMPRESSION", "tarfile.open(write)")
        return None

    def _is_sync_db(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
        """sqlite3.connect / duckdb.connect — only flag if blocking semantics.

        v3: resolves aliases — `import sqlite3 as _cj_sql; _cj_sql.connect()`
        is now detected. Found 7 such sites hiding in dashboard.py.
        """
        if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
            mod = _resolve_module(node.func.value, aliases)
            if mod in ("sqlite3", "duckdb"):
                return ("SYNC_DB", f"{mod}.connect")
        return None

    DETECTORS = [_is_sync_http, _is_urlopen, _is_subprocess, _is_compression, _is_sync_db]

    # ── Thread-boundary detection (to_thread / run_in_executor / Thread) ──
    #
    # 2026-04-14 (v2): the v1 scanner flagged sync I/O anywhere in the
    # transitive call graph from an async def. That produced ~70% false-
    # positive rate because it didn't understand:
    #   (a) `await asyncio.to_thread(fn)` — fn runs on a thread, isolated
    #   (b) `threading.Thread(target=fn, daemon=True).start()` — same
    #   (c) `# noqa: async-block` inline suppression
    # This rewrite fixes all three.

    def _is_thread_boundary_call(call_node) -> bool:
        """True if this Call is a thread-boundary wrapper (to_thread /
        run_in_executor / Thread(target=...)).

        Any sync I/O in the callable argument runs on a thread, NOT on
        the event loop — so it should not be flagged and should not
        propagate async-reachability through the call graph.
        """
        if not isinstance(call_node, ast.Call):
            return False
        f = call_node.func
        # asyncio.to_thread(...) — function attribute
        if isinstance(f, ast.Attribute):
            if f.attr == "to_thread":
                return True
            if f.attr == "run_in_executor":
                return True
        # threading.Thread(target=...) — class call (executed on thread)
        if isinstance(f, ast.Attribute) and f.attr == "Thread":
            return True
        if isinstance(f, ast.Name) and f.id == "Thread":
            return True
        return False

    def _callable_arg_of_wrapper(call_node) -> ast.AST | None:
        """Return the callable passed to a thread-boundary wrapper.

        For `asyncio.to_thread(fn, *args)` → returns `fn`.
        For `loop.run_in_executor(exec, fn, *args)` → returns `fn`.
        For `threading.Thread(target=fn, ...)` → returns the value of
        `target=` kwarg. Returns None if no callable argument found.
        """
        if not isinstance(call_node, ast.Call):
            return None
        f = call_node.func
        # to_thread: first positional
        if isinstance(f, ast.Attribute) and f.attr == "to_thread":
            return call_node.args[0] if call_node.args else None
        # run_in_executor: second positional (first is executor)
        if isinstance(f, ast.Attribute) and f.attr == "run_in_executor":
            return call_node.args[1] if len(call_node.args) >= 2 else None
        # Thread(target=fn)
        if (isinstance(f, ast.Attribute) and f.attr == "Thread") or \
           (isinstance(f, ast.Name) and f.id == "Thread"):
            for kw in call_node.keywords:
                if kw.arg == "target":
                    return kw.value
            # Positional: Thread(group, target)
            if len(call_node.args) >= 2:
                return call_node.args[1]
        return None

    def _extract_wrapped_name(callable_arg) -> str | None:
        """From a callable argument, extract the function name being wrapped.

        Handles:
            asyncio.to_thread(foo)          → "foo"
            asyncio.to_thread(self.foo)     → "foo"
            asyncio.to_thread(lambda: x())  → None (lambda, no named target)
        """
        if callable_arg is None:
            return None
        if isinstance(callable_arg, ast.Name):
            return callable_arg.id
        if isinstance(callable_arg, ast.Attribute):
            return callable_arg.attr
        return None  # Lambda, nested call, etc.

    # ── AST visitor ─────────────────────────────────────────────────────

    class _FunctionContext:
        def __init__(self, name: str, is_async: bool, line: int):
            self.name = name
            self.is_async = is_async
            self.line = line
            # Call-graph edges. Each edge is (target_name, is_attr_call).
            # is_attr_call=True for method-style calls (obj.foo()) — resolved
            # preferring same-file match to avoid name-collision bloat.
            # is_attr_call=False for bare-name calls (foo()) — resolved
            # globally (module-level function).
            self.calls_to_direct: set[tuple[str, bool]] = set()
            self.calls_to_thread: set[tuple[str, bool]] = set()
            self.dangerous: list[tuple[int, str, str]] = []  # (line, kind, name)

    class _Scanner(ast.NodeVisitor):
        def __init__(self, source_lines: list[str]):
            self.functions: list[_FunctionContext] = []
            self._current: list[_FunctionContext] = []
            self._source_lines = source_lines
            # Track the currently-active to_thread wrapper — every descendant
            # of its callable argument is "inside a thread boundary". Uses a
            # stack so nested wrappers work correctly.
            self._thread_boundary_depth = 0
            # v3: per-file import alias table. Updated by visit_Import /
            # visit_ImportFrom. Used by detectors to resolve `_af_sql` →
            # "sqlite3" etc. Maps local-name → canonical-module-path.
            self._aliases: dict[str, str] = {}

        # -- parent pointer setup ----------------------------------------
        # v3 bugfix: _set_parents MUST run exactly once on the full tree
        # before visit() is called. The prior version ran it inside every
        # visit() call, which reset `node._parent = None` each time a
        # child was visited — so by the time the visitor reached a Lambda
        # nested 6 levels deep, the parent chain had been shredded and
        # thread-boundary detection missed `asyncio.to_thread(lambda: …)`.
        # This silently hid ~2 CRITICAL sites per scan. Now done once in
        # the module-level walker below, before `scanner.visit(tree)`.
        def _set_parents(self, node, parent=None):
            node._parent = parent  # type: ignore[attr-defined]
            for child in ast.iter_child_nodes(node):
                self._set_parents(child, node)

        # -- function tracking -------------------------------------------
        def _enter_func(self, node, is_async):
            ctx = _FunctionContext(node.name, is_async, node.lineno)
            self.functions.append(ctx)
            self._current.append(ctx)
            self.generic_visit(node)
            self._current.pop()

        def visit_FunctionDef(self, node):
            self._enter_func(node, is_async=False)

        def visit_AsyncFunctionDef(self, node):
            self._enter_func(node, is_async=True)

        # -- import alias tracking (v3) ----------------------------------
        def visit_Import(self, node):
            # `import sqlite3 as _af_sql` → aliases["_af_sql"] = "sqlite3"
            # `import urllib.request` → aliases["urllib.request"] unchanged
            # (detector resolves the Attribute chain "urllib.request.urlopen"
            # itself and checks aliases for the leftmost "urllib").
            for alias in node.names:
                local = alias.asname or alias.name
                # Keep top-level module canonical so sqlite3.X → sqlite3
                self._aliases[local] = alias.name
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # `from urllib.request import urlopen as _uro` →
            # aliases["_uro"] = "urllib.request.urlopen" so detectors can
            # recognize bare-name calls too.
            if node.module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    self._aliases[local] = f"{node.module}.{alias.name}"
            self.generic_visit(node)

        # -- call tracking + thread-boundary detection -------------------
        def _inside_thread_boundary(self, node: ast.AST) -> bool:
            """Walk up via ._parent pointers. True if any ancestor is the
            callable argument of a thread-boundary wrapper call.

            Two patterns count:
              (A) Direct argument: asyncio.to_thread(self._sync_fn, arg1)
                  — first positional arg is the sync function reference.
              (B) Lambda body: asyncio.to_thread(lambda: self._sync_fn())
                  — lambda wraps the sync work.
              (C) Inner def body: `def _do(): ...; return to_thread(_do)`
                  — inner function referenced by to_thread.
            """
            cur = getattr(node, "_parent", None)
            while cur is not None:
                # Check if cur is a Call whose callable-arg ancestor-path
                # reaches `node`. The cleaner check: climb until we find a
                # lambda/FunctionDef, then verify that lambda/def is the
                # callable argument of a wrapper.
                if isinstance(cur, ast.Lambda):
                    # Lambda — is it the callable arg of a wrapper?
                    lam_parent = getattr(cur, "_parent", None)
                    while lam_parent is not None:
                        if isinstance(lam_parent, ast.Call) and _is_thread_boundary_call(lam_parent):
                            arg = _callable_arg_of_wrapper(lam_parent)
                            if arg is cur:
                                return True
                        # Stop climbing if we leave the argument expression
                        if isinstance(lam_parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            break
                        lam_parent = getattr(lam_parent, "_parent", None)
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Inner def — is it the callable arg of a wrapper?
                    # The def statement itself isn't an expression, so we
                    # can't pass it directly. But if an outer function
                    # defines this inner def and then calls to_thread(inner),
                    # we need to check if the name matches.
                    # Conservative: don't infer — let call-graph edge handle it.
                    break
                cur = getattr(cur, "_parent", None)
            return False

        def _line_suppressed(self, lineno: int) -> bool:
            """True if the source line contains `# noqa: async-block`."""
            if 0 < lineno <= len(self._source_lines):
                line = self._source_lines[lineno - 1]
                if "noqa: async-block" in line or "noqa:async-block" in line:
                    return True
            return False

        def visit_Call(self, node):
            inside_tb = self._inside_thread_boundary(node)

            # Track function calls (for call graph) — distinguish wrapped
            # (thread boundary) vs direct edges. Three cases:
            #   (1) This call IS a thread-boundary wrapper → record its
            #       named callable arg as a threaded edge.
            #   (2) This call is INSIDE a thread boundary (e.g., inside a
            #       lambda body that to_thread wraps) → record as threaded.
            #   (3) Otherwise → direct.
            if self._current:
                target = None
                is_attr = False
                if isinstance(node.func, ast.Name):
                    target = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    target = node.func.attr
                    is_attr = True

                if _is_thread_boundary_call(node):
                    wrapped = _extract_wrapped_name(_callable_arg_of_wrapper(node))
                    if wrapped:
                        # Attribute-style if the callable arg is an Attribute
                        # (e.g., to_thread(self.foo)); bare-name otherwise.
                        ca = _callable_arg_of_wrapper(node)
                        w_is_attr = isinstance(ca, ast.Attribute)
                        self._current[-1].calls_to_thread.add((wrapped, w_is_attr))
                    if target:
                        self._current[-1].calls_to_direct.add((target, is_attr))
                elif inside_tb and target:
                    self._current[-1].calls_to_thread.add((target, is_attr))
                elif target:
                    self._current[-1].calls_to_direct.add((target, is_attr))

            # Run dangerous-pattern detectors (but skip if suppressed or
            # if the call is inside a thread-boundary wrapper). Pass the
            # per-file alias table so aliased imports resolve correctly.
            if self._current and not self._line_suppressed(node.lineno) and not inside_tb:
                for d in DETECTORS:
                    m = d(node, self._aliases)
                    if m:
                        kind, name = m
                        self._current[-1].dangerous.append((node.lineno, kind, name))
                        break
            self.generic_visit(node)

    # ── Walk all .py files ──────────────────────────────────────────────
    file_data: dict[str, list[_FunctionContext]] = {}
    all_async_funcs: set[str] = set()

    for root, dirs, files in os.walk(SCAN_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = Path(root) / f
            try:
                src = full.read_text()
                tree = ast.parse(src, filename=str(full))
            except Exception:
                continue
            scanner = _Scanner(src.splitlines())
            # Set parent pointers ONCE on the whole tree before visiting.
            # Running this per-visit (v3 pre-bugfix) clobbered deep ancestor
            # chains and made to_thread wrappers undetectable.
            scanner._set_parents(tree)
            scanner.visit(tree)
            rel = str(full.relative_to(SCAN_ROOT))
            file_data[rel] = scanner.functions
            for fctx in scanner.functions:
                if fctx.is_async:
                    all_async_funcs.add(fctx.name)

    # ── Build (file, name) call graph + transitive reachability ────────
    #
    # v2 (2026-04-14): switched from bare-name indexing to (file, name)
    # identity. For attribute-style calls (obj.foo()), resolve only to
    # same-file definitions — this prevents name-collision bloat where
    # BFS hits one async caller of `_connect` (say, VCB.build wrapping
    # ChainArchive._connect inside to_thread) and then every `_connect`
    # method in every DB class gets flagged. For bare-name calls (foo()),
    # resolve to all matching functions (module-level imports are harder
    # to trace statically).
    func_by_file: dict[str, dict[str, list[_FunctionContext]]] = defaultdict(lambda: defaultdict(list))
    func_by_name: dict[str, list[tuple[str, _FunctionContext]]] = defaultdict(list)
    for rel, funcs in file_data.items():
        for fctx in funcs:
            func_by_file[rel][fctx.name].append(fctx)
            func_by_name[fctx.name].append((rel, fctx))

    def _resolve(caller_file: str, target: str, is_attr: bool) -> list[tuple[str, _FunctionContext]]:
        """Resolve a call edge to concrete (file, function) tuples.

        Attribute calls prefer same-file matches (Python method calls
        overwhelmingly target same-class methods). Bare-name calls are
        resolved across all files (could be imported free functions).
        """
        if is_attr:
            same_file = func_by_file.get(caller_file, {}).get(target, [])
            if same_file:
                return [(caller_file, fx) for fx in same_file]
            # No same-file match → don't propagate (conservative). Could be
            # stdlib method (e.g., str.strip), external-class method, or
            # cross-file method call. False negatives here are acceptable
            # given v1's ~70% false positive rate.
            return []
        # Bare-name: resolve globally.
        return func_by_name.get(target, [])

    # Reachable from any async = BFS over DIRECT edges only. Edges through
    # asyncio.to_thread / loop.run_in_executor / threading.Thread(target=)
    # cross a thread boundary — the callee runs on a thread, so sync I/O
    # inside it does NOT block the asyncio event loop. Those edges live in
    # fctx.calls_to_thread and are intentionally not followed.
    reachable: set[tuple[str, str]] = set()  # (file, function_name) identities
    for rel, funcs in file_data.items():
        for fctx in funcs:
            if fctx.is_async:
                reachable.add((rel, fctx.name))
    frontier = set(reachable)
    while frontier:
        next_frontier = set()
        for caller_file, caller_name in frontier:
            caller_fxs = func_by_file.get(caller_file, {}).get(caller_name, [])
            for caller_fx in caller_fxs:
                for target, is_attr in caller_fx.calls_to_direct:
                    for callee_file, callee_fx in _resolve(caller_file, target, is_attr):
                        key = (callee_file, callee_fx.name)
                        if key not in reachable:
                            reachable.add(key)
                            next_frontier.add(key)
        frontier = next_frontier
    reachable_from_async: set[str] = {name for _, name in reachable}
    reachable_ids: set[tuple[str, str]] = reachable

    # ── Classify each dangerous call by severity ───────────────────────
    rows: list[tuple[str, str, str, int, str, str, str]] = []
    # (severity, file, function, line, kind, name, async_status)

    for rel, funcs in file_data.items():
        is_worker_file = rel in WORKER_FILES
        for fctx in funcs:
            for line, kind, name in fctx.dangerous:
                if is_worker_file:
                    severity = "LOW"
                    async_status = "worker-subprocess"
                elif fctx.is_async:
                    severity = "CRITICAL"
                    async_status = "async def"
                elif (rel, fctx.name) in reachable_ids:
                    severity = "HIGH"
                    async_status = "called from async"
                else:
                    # In a module where any async exists — flag as potential
                    has_async = any(f.is_async for f in funcs)
                    if has_async:
                        severity = "MEDIUM"
                        async_status = "module exposes async"
                    else:
                        severity = "LOW"
                        async_status = "no async in file"
                rows.append((severity, rel, fctx.name, line, kind, name, async_status))

    # ── Output ─────────────────────────────────────────────────────────
    print()
    print("ASYNC-BLOCK SCAN — sync I/O reachable from async functions")
    print("=" * 100)
    print(f"  Scanned {len(file_data)} files, {sum(len(f) for f in file_data.values())} functions, "
          f"{len(all_async_funcs)} async, {len(reachable_from_async) - len(all_async_funcs)} reachable-from-async")
    print()

    SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    rows.sort(key=lambda r: (SEV_ORDER[r[0]], r[1], r[3]))

    by_sev: dict[str, list] = defaultdict(list)
    for row in rows:
        by_sev[row[0]].append(row)

    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        icon = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "⚪"}[sev]
        print(f"  {icon} {sev}  ({len(items)} sites)")
        print(f"  {'file':<55} {'function':<35} {'line':>5}  {'pattern':<25} {'context'}")
        print(f"  {'-'*55} {'-'*35} {'-'*5}  {'-'*25} {'-'*25}")
        for sev_, rel, func, line, kind, name, ctx in items[:30]:
            print(f"  {rel[:55]:<55} {func[:35]:<35} {line:>5}  {name[:25]:<25} {ctx}")
        if len(items) > 30:
            print(f"  ... and {len(items)-30} more {sev} sites")
        print()

    # Summary
    print(f"  TOTALS: CRITICAL={len(by_sev.get('CRITICAL', []))}  "
          f"HIGH={len(by_sev.get('HIGH', []))}  "
          f"MEDIUM={len(by_sev.get('MEDIUM', []))}  "
          f"LOW={len(by_sev.get('LOW', []))}")
    if by_sev.get("CRITICAL"):
        print(f"\n  🛑 CRITICAL sites must be wrapped in asyncio.to_thread() before deploy.")
    print()
    print("=" * 100)


def run_bus_census_analysis(log_path: str = "/tmp/titan_bus_census.log",
                             top_n: int = 15):
    """Phase E.1 diagnosis: analyze bus census log to find cascade root cause.

    Reads the TSV emitted by titan_plugin/core/bus_census.py and produces:
      • Top emitters by msg/s  (find the burst producers)
      • Top dropped by msg/s   (find the saturated subscribers)
      • Queue depth stats      (find the slow drains)
      • Drop rate vs emission  (saturation %)

    Run after enabling TITAN_BUS_CENSUS=1 and observing for 10-30 min.
    The output identifies which of these is the cascade root cause:
      - Burst producer (one msg type spiking)
      - Slow drain (depth growing monotonically)
      - Heartbeat priority inversion (depth small but heartbeats dropped)
      - Queue too small (steady-state depth near maxsize)
    """
    import os
    import statistics
    from collections import defaultdict

    print()
    print(f"BUS CENSUS ANALYSIS — {log_path}")
    print("=" * 90)

    if not os.path.exists(log_path):
        print(f"  ✗ Census log not found: {log_path}")
        print(f"  Enable with: TITAN_BUS_CENSUS=1 (env var on titan_main)")
        return

    emit_total: dict[str, int] = defaultdict(int)
    drop_total: dict[str, int] = defaultdict(int)
    depth_samples: dict[str, list[int]] = defaultdict(list)
    tick_pids: set[int] = set()
    first_ts = None
    last_ts = None
    n_lines = 0

    with open(log_path) as f:
        for line in f:
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            kind, key, value = parts[1], parts[2], parts[3]
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            if kind == "EMIT":
                emit_total[key] += int(value)
            elif kind == "DROP":
                drop_total[key] += int(value)
            elif kind == "DEPTH":
                try:
                    depth_samples[key].append(int(value))
                except ValueError:
                    pass
            elif kind == "TICK":
                # key is like "pid=1234"
                try:
                    pid = int(key.split("=", 1)[1])
                    tick_pids.add(pid)
                except (ValueError, IndexError):
                    pass

    if not first_ts or not last_ts or last_ts <= first_ts:
        print(f"  ✗ Insufficient data ({n_lines} lines, no time window)")
        return

    duration = last_ts - first_ts
    print(f"  Window: {duration:.1f}s  ({n_lines} lines, {len(tick_pids)} processes contributed)")
    print()

    # ── 1. TOP EMITTERS ──
    print(f"  TOP {top_n} EMITTERS  (msg/s averaged over window)")
    print(f"  {'msg_type|dst':<55} {'total':>10} {'msg/s':>10}")
    print(f"  {'-'*55} {'-'*10} {'-'*10}")
    sorted_emit = sorted(emit_total.items(), key=lambda kv: -kv[1])[:top_n]
    total_emit_all = sum(emit_total.values())
    for key, total in sorted_emit:
        rate = total / duration
        print(f"  {key[:55]:<55} {total:>10d} {rate:>10.2f}")
    print(f"  {'TOTAL ALL EMITS':<55} {total_emit_all:>10d} {total_emit_all/duration:>10.2f}")
    print()

    # ── 2. TOP DROPS ──
    if drop_total:
        print(f"  TOP {top_n} DROPS  (queue full → message lost)")
        print(f"  {'subscriber|msg_type':<55} {'total':>10} {'msg/s':>10}")
        print(f"  {'-'*55} {'-'*10} {'-'*10}")
        sorted_drop = sorted(drop_total.items(), key=lambda kv: -kv[1])[:top_n]
        total_drop_all = sum(drop_total.values())
        for key, total in sorted_drop:
            rate = total / duration
            print(f"  {key[:55]:<55} {total:>10d} {rate:>10.2f}")
        print(f"  {'TOTAL ALL DROPS':<55} {total_drop_all:>10d} {total_drop_all/duration:>10.2f}")
        drop_pct = 100 * total_drop_all / max(1, total_emit_all)
        print(f"\n  → SATURATION: {drop_pct:.2f}% of emissions dropped")
        print()
    else:
        print(f"  ✓ No drops in window")
        print()

    # ── 3. QUEUE DEPTHS ──
    if depth_samples:
        print(f"  QUEUE DEPTHS  (per subscriber, samples every {duration/max(1,len(next(iter(depth_samples.values())))):.1f}s)")
        print(f"  {'subscriber':<25} {'samples':>8} {'min':>6} {'mean':>8} {'max':>6} {'p95':>6} {'trend':<20}")
        print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*20}")
        for sub in sorted(depth_samples.keys()):
            ds = depth_samples[sub]
            if not ds:
                continue
            n = len(ds)
            mn, mx = min(ds), max(ds)
            mean = sum(ds) / n
            p95 = sorted(ds)[int(0.95 * n)] if n > 1 else mx
            # Trend: split into halves, compare means
            if n >= 4:
                first_half = ds[:n//2]
                second_half = ds[n//2:]
                fm = sum(first_half) / len(first_half)
                sm = sum(second_half) / len(second_half)
                if sm > fm * 1.5 and sm > 50:
                    trend = "↑↑ GROWING"
                elif sm > fm * 1.1:
                    trend = "↑ rising"
                elif sm < fm * 0.5:
                    trend = "↓ draining"
                else:
                    trend = "≈ stable"
            else:
                trend = "(too few)"
            print(f"  {sub[:25]:<25} {n:>8d} {mn:>6d} {mean:>8.1f} {mx:>6d} {p95:>6d} {trend:<20}")
        print()

    # ── 4. DIAGNOSIS HINTS ──
    print(f"  DIAGNOSIS HINTS")
    print(f"  " + "-"*86)
    if drop_total:
        # Find subscribers that received the most drops
        drop_by_sub: dict[str, int] = defaultdict(int)
        for key, n in drop_total.items():
            sub = key.split("|", 1)[0]
            drop_by_sub[sub] += n
        worst_sub = max(drop_by_sub.items(), key=lambda kv: kv[1])
        print(f"    • Worst-saturated subscriber: '{worst_sub[0]}' "
              f"({worst_sub[1]} drops = {worst_sub[1]/duration:.2f}/s)")
    if depth_samples:
        # Find subscribers with growing depth
        growing = []
        for sub, ds in depth_samples.items():
            if len(ds) >= 4:
                fm = sum(ds[:len(ds)//2]) / (len(ds)//2)
                sm = sum(ds[len(ds)//2:]) / (len(ds) - len(ds)//2)
                if sm > fm * 1.5 and sm > 50:
                    growing.append((sub, fm, sm))
        if growing:
            print(f"    • Subscribers with GROWING queue depth (slow drain suspect):")
            for sub, fm, sm in growing:
                print(f"        - {sub}: {fm:.0f} → {sm:.0f}")
    if emit_total:
        # Top 3 producers concentration
        top3 = sorted(emit_total.values(), reverse=True)[:3]
        top3_pct = 100 * sum(top3) / max(1, sum(emit_total.values()))
        print(f"    • Top 3 message types account for {top3_pct:.1f}% of total emissions")
    print()
    print("=" * 90)


def run_loop_latency_check(target: str = "T1", endpoint: str = "/v4/inner-trinity",
                            concurrency: int = 20, total: int = 100,
                            p95_max_ms: float = 300.0,
                            p99_max_ms: float = 800.0) -> int:
    """Concurrent-load latency profile against a live Titan endpoint.

    Fires `total` HTTP GET requests at `concurrency` parallelism and reports
    the per-request latency distribution (p50/p95/p99/max) plus pass/fail
    vs the supplied thresholds.

    This is the complement to `async-blocks` (static AST scan): the scanner
    catches sync I/O inside async paths at commit-time, while loop-latency
    catches *dynamic* regressions — e.g., a new bus-cascade, a newly added
    cache-miss, a lock-contention issue — that no static tool can see. If
    a future change accidentally un-wraps a slow call or introduces a
    synchronous bus timeout, p99 spikes and this test fails.

    Returns exit-code style int: 0 = pass, 1 = threshold exceeded,
    2 = fetch failure (Titan unreachable or endpoint 404).
    """
    import asyncio
    import statistics
    import time

    try:
        import httpx  # already a project dep
    except ImportError:
        print("  ✗ httpx not installed — pip install httpx")
        return 2

    targets = {
        "T1": "http://127.0.0.1:7777",
        "T2": "http://10.135.0.6:7777",
        "T3": "http://10.135.0.6:7778",
    }
    base = targets.get(target.upper())
    if not base:
        print(f"  ✗ unknown target '{target}' (use T1, T2, or T3)")
        return 2
    url = f"{base}{endpoint}"

    print()
    print("LOOP-LATENCY CHECK — concurrent-request distribution")
    print("=" * 88)
    print(f"  Target:      {target} {url}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Total reqs:  {total}")
    print(f"  Thresholds:  p95 <= {p95_max_ms}ms  p99 <= {p99_max_ms}ms")
    print()

    latencies_ms: list[float] = []
    errors: list[str] = []

    async def _one(client: "httpx.AsyncClient"):
        t0 = time.perf_counter()
        try:
            r = await client.get(url, timeout=15.0)
            if r.status_code != 200:
                errors.append(f"HTTP {r.status_code}")
                return
        except Exception as e:
            errors.append(str(e)[:60])
            return
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    async def _run():
        async with httpx.AsyncClient() as client:
            # Semaphore throttles in-flight requests to `concurrency`.
            sem = asyncio.Semaphore(concurrency)
            async def _bounded():
                async with sem:
                    await _one(client)
            await asyncio.gather(*[_bounded() for _ in range(total)])

    t_start = time.perf_counter()
    asyncio.run(_run())
    wall_ms = (time.perf_counter() - t_start) * 1000

    if not latencies_ms:
        print(f"  ✗ no successful responses — {len(errors)} errors")
        if errors:
            print(f"    sample: {errors[0]}")
        print("=" * 88)
        return 2

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[n // 2]
    p95 = latencies_ms[int(n * 0.95)]
    p99 = latencies_ms[int(n * 0.99)] if n >= 100 else latencies_ms[-1]
    mx = latencies_ms[-1]
    mn = latencies_ms[0]
    mean = statistics.mean(latencies_ms)

    print(f"  Wall clock:  {wall_ms:.0f}ms  ({n}/{total} success, {len(errors)} errors)")
    print(f"  Throughput:  {n / (wall_ms / 1000):.1f} req/s (effective)")
    print()
    print(f"  {'percentile':<12} {'latency (ms)':>14}  notes")
    print(f"  {'-'*12} {'-'*14}  -----")
    print(f"  {'min':<12} {mn:>14.1f}")
    print(f"  {'mean':<12} {mean:>14.1f}")
    print(f"  {'p50':<12} {p50:>14.1f}")
    marker = " 🛑 FAIL" if p95 > p95_max_ms else " ✓ pass"
    print(f"  {'p95':<12} {p95:>14.1f}  {marker} (threshold {p95_max_ms})")
    marker = " 🛑 FAIL" if p99 > p99_max_ms else " ✓ pass"
    print(f"  {'p99':<12} {p99:>14.1f}  {marker} (threshold {p99_max_ms})")
    print(f"  {'max':<12} {mx:>14.1f}")
    if errors:
        print()
        print(f"  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")
    print()

    failed = (p95 > p95_max_ms) or (p99 > p99_max_ms)
    if failed:
        print("  🛑 THRESHOLD EXCEEDED — investigate with py-spy, check /v4/thread-pool,")
        print("  review recent commits for un-wrapped sync I/O or new bus-cascades.")
    else:
        print("  ✓ Latency distribution healthy.")
    print("=" * 88)
    return 1 if failed else 0


def run_parent_thread_inventory(all_titans: bool = False):
    """Poll /v4/admin/parent-threads on T1 (or all 3) and print a per-Titan
    inventory of parent + api_subprocess threads.

    Microkernel rFP A.8 §6 measurement-driven residency audit — surfaces
    which subsystems own threads in the kernel residency layer. The rFP
    target is parent_threads ≤ 25 sustained post-completion (gated on
    bus_ipc_socket_enabled flag flip — A.8.2 alone ships forward-looking
    code that becomes meaningful when graduation flips on).
    """
    import json
    import urllib.request

    targets = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        targets.extend([
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ])

    print()
    print("PARENT THREAD INVENTORY — microkernel A.8 §6 residency audit")
    print("=" * 88)

    for tid, base in targets:
        try:
            req = urllib.request.Request(f"{base}/v4/admin/parent-threads")
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  {tid:<6} unreachable: {str(e)[:80]}")
            continue

        payload = data.get("data") or data
        parent = payload.get("parent") or {}
        api = payload.get("api_subprocess") or {}

        p_total = parent.get("total", "?")
        p_pid = parent.get("pid", "?")
        a_total = api.get("total", "?")
        a_pid = api.get("pid", "?")
        rfp_target = 25
        try:
            p_status = "✓" if int(p_total) <= rfp_target else "⚠"
        except (TypeError, ValueError):
            p_status = "?"

        print()
        print(f"  {tid}  parent (pid {p_pid}) — total {p_total} threads "
              f"[A.8 target ≤{rfp_target}: {p_status}]")
        print("  " + "-" * 84)
        for prefix, count in (parent.get("by_prefix") or {}).items():
            print(f"    {count:>4d}  {prefix}")
        if "error" in parent:
            print(f"    ERROR: {parent['error']}")
        print()
        print(f"  {tid}  api_subprocess (pid {a_pid}) — total {a_total} threads")
        print("  " + "-" * 84)
        for prefix, count in (api.get("by_prefix") or {}).items():
            print(f"    {count:>4d}  {prefix}")

    print()
    print("=" * 88)
    print("  Notes:")
    print("    • Parent thread count > 25 today is mostly multiprocessing.Queue helpers")
    print("      (one helper thread per worker queue × ~14 workers). These go away when")
    print("      microkernel.bus_ipc_socket_enabled=true graduates workers off mp.Queue.")
    print("    • A.8 §3.5 ships forward-looking conditional that prevents redundant")
    print("      supervision-thread spawn for fork-mode workers when graduation flips on.")
    print("=" * 88)


def run_thread_pool_check():
    """Poll /v4/thread-pool on all 3 Titans and print a saturation table.

    Uses the asyncio default-executor introspection endpoint (added
    2026-04-14) to reveal pool pressure: with ~100 sites now wrapped in
    asyncio.to_thread, an under-sized pool becomes the new bottleneck.
    Default is 64 workers post-bump; a saturation_pct >= 60% for any
    sustained period suggests bumping or reducing concurrent to_thread
    usage.
    """
    import json
    import urllib.request

    targets = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ]
    print()
    print("THREAD-POOL SATURATION — asyncio default-executor")
    print("=" * 88)
    print(f"  {'Titan':<6} {'State':<10} {'max':>4} {'live':>5} {'busy':>5} "
          f"{'idle':>5} {'queued':>7} {'sat %':>7}  notes")
    print(f"  {'-'*6} {'-'*10} {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*7}  -----")

    any_hot = False
    reached = 0
    for tid, base in targets:
        try:
            req = urllib.request.Request(f"{base}/v4/thread-pool")
            with urllib.request.urlopen(req, timeout=4) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  {tid:<6} {'unreach':<10} — (fetch failed: {str(e)[:50]})")
            continue
        reached += 1
        payload = data.get("data") or {}
        state = payload.get("state", "?")
        mw = payload.get("max_workers", "?")
        live = payload.get("live_workers", "?")
        busy = payload.get("busy_workers", "?")
        idle = payload.get("idle_workers", "?")
        queued = payload.get("queued_tasks", "?")
        sat = payload.get("saturation_pct", "?")
        sat_s = f"{sat}" if isinstance(sat, (int, float)) else str(sat)
        note = ""
        if isinstance(sat, (int, float)):
            if sat >= 90:
                note = "🛑 CRITICAL — expand pool or reduce concurrent to_thread"
                any_hot = True
            elif sat >= 60:
                note = "⚠ warm — monitor"
                any_hot = True
            else:
                note = "✓ healthy"
        print(f"  {tid:<6} {state:<10} {str(mw):>4} {str(live):>5} "
              f"{str(busy):>5} {str(idle):>5} {str(queued):>7} {sat_s:>7}  {note}")

    print()
    if reached == 0:
        print("  No Titan reachable — /v4/thread-pool may not be deployed yet, or all")
        print("  endpoints are down. Verify with /health first.")
    elif any_hot:
        print("  At least one Titan has elevated saturation. Watch for slow /v4/* endpoints")
        print("  and consider increasing max_workers in scripts/titan_main.py (line ~185).")
    else:
        print("  All reachable pools healthy. Current 64-worker size has headroom for ~100")
        print("  concurrent to_thread sites as of Phase E.2 + ccd2ef6 API-fix deploy.")
    print("=" * 88)


def run_stability_check(hours: int = 24):
    """Stability audit — restart density per Titan over last `hours`.

    Sources (multi-source by design — see feedback_titan_status_must_be_verified.md):
      • T1 watchdog log    : /tmp/titan1_watchdog.log
      • T2 watchdog log    : ssh root@10.135.0.6 /tmp/titan2_watchdog.log
      • T3 watchdog log    : ssh root@10.135.0.6 /tmp/titan3_watchdog.log
      • Live /health       : http://127.0.0.1:7777, 10.135.0.6:7777, 10.135.0.6:7778
      • Brain log RESTART boundaries (cross-check that watchdog kicks led to actual restarts)

    Verdict per Titan:
      ✓ STABLE       — 0 restarts in window
      ⚠ FRAGILE      — 1-3 restarts in window (rate 0.04-0.13/h)
      ✗ CRASH-LOOP   — >3 restarts in window (>0.13/h) — HALT all feature work

    Output is designed to be the FIRST thing in any session_startup_protocol run.
    """
    import requests
    import re
    import datetime as dt
    import subprocess

    now = dt.datetime.utcnow()
    window_start = now - dt.timedelta(hours=hours)

    print()
    print(f"TITAN STABILITY AUDIT — last {hours}h (window: "
          f"{window_start.strftime('%Y-%m-%d %H:%M')} → "
          f"{now.strftime('%Y-%m-%d %H:%M')} UTC)")
    print("=" * 90)

    # Regex matching ALL forms of restart events in watchdog logs:
    #  - "force-restart..."
    #  - "force restart..."
    #  - "T1 force-restarted, PID=..."
    #  - "T1 NOT RUNNING — auto-restarting..."
    #  - "T1 truly hung..."
    restart_pat = re.compile(
        r"(force[- ]restart|NOT RUNNING.*restart|truly hung)",
        re.IGNORECASE,
    )
    # Watchdog timestamp at line start: [YYYY-MM-DD HH:MM:SS UTC]
    ts_pat = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+UTC\]")

    # Guardian in-process module restart patterns (brain log). These are
    # respawns WITHIN a running titan_main — distinct from watchdog-detected
    # process-level crashes above. Counted since last full titan_main boot
    # (last "Titan.*starting" marker in brain log) so stale historical events
    # don't pollute the current-session view.
    mod_timeout_pat = re.compile(
        r"\[Guardian\] Module '([a-z_]+)' heartbeat timeout")
    mod_rss_pat = re.compile(
        r"\[Guardian\] Module '([a-z_]+)' RSS \d+MB > limit")
    # Capture exitcode too — we skip graceful shutdowns (exitcode=0 emitted
    # by Guardian's own stop_all during titan_main restarts). Only non-zero
    # exits represent true crashes. Before this fix, every clean restart
    # produced 8-11 "died" events, badly inflating the stability count.
    mod_died_pat = re.compile(
        r"\[Guardian\] Module '([a-z_]+)' died \(exitcode=(-?\d+)")
    # Brain log timestamp (no date, just HH:MM:SS at line start)
    brain_ts_pat = re.compile(r"^(\d{2}:\d{2}:\d{2})")
    # Full titan_main boot marker
    boot_marker_pat = re.compile(
        r"(RESTART boundary|Titan.*starting|\[TitanCore\].*Boot)")

    def _parse_watchdog(content: str) -> tuple[list[dt.datetime], int]:
        """Return (restart_timestamps_in_window, total_restarts_in_log)."""
        in_window = []
        total = 0
        for line in content.splitlines():
            if not restart_pat.search(line):
                continue
            total += 1
            m = ts_pat.match(line)
            if not m:
                continue
            try:
                t = dt.datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if t >= window_start:
                in_window.append(t)
        return in_window, total

    def _read_local(path: str) -> str | None:
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _read_remote(path: str) -> str | None:
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5",
                 "-o", "BatchMode=yes",
                 "root@10.135.0.6", f"cat {path}"],
                capture_output=True, text=True, timeout=15)
            if r.returncode != 0:
                return None
            return r.stdout
        except Exception:
            return None

    def _parse_module_restarts(content: str, window_hours: int = 24,
                               ref_dt: "dt.datetime | None" = None) -> dict:
        """Parse brain log for Guardian module restart REASONS per module,
        filtered to the last `window_hours` hours.

        Brain logs only have HH:MM:SS timestamps (no date). Approach: walk
        lines from END backwards, tracking date rollovers (when HH:MM:SS
        jumps FORWARD across consecutive lines = crossed midnight going
        backward). Stop when (reconstructed datetime) < window_start.

        Returns dict with per-module breakdown + grand_total + window info.
        """
        if ref_dt is None:
            ref_dt = dt.datetime.utcnow()
        window_start = ref_dt - dt.timedelta(hours=window_hours)

        lines = content.splitlines()
        per_module: dict[str, dict] = {}

        # Walk backwards: track current date as we go
        cur_date = ref_dt.date()
        last_time: "dt.time | None" = None
        hit_window_start = False

        for line in reversed(lines):
            ts_match = brain_ts_pat.match(line)
            if not ts_match:
                continue
            try:
                hms = dt.datetime.strptime(
                    ts_match.group(1), "%H:%M:%S").time()
            except ValueError:
                continue

            # Date rollover detection (walking backwards)
            if last_time is not None and hms > last_time:
                cur_date = cur_date - dt.timedelta(days=1)
            last_time = hms

            event_dt = dt.datetime.combine(cur_date, hms)
            if event_dt < window_start:
                hit_window_start = True
                break

            # Check restart-reason patterns
            ts_str = ts_match.group(1)
            matched_reason = None
            matched_mod = None

            m = mod_timeout_pat.search(line)
            if m:
                matched_reason = "heartbeat_timeout"
                matched_mod = m.group(1)
            else:
                m = mod_rss_pat.search(line)
                if m:
                    matched_reason = "rss_exceeded"
                    matched_mod = m.group(1)
                else:
                    m = mod_died_pat.search(line)
                    if m:
                        try:
                            _exitcode = int(m.group(2))
                        except (ValueError, IndexError):
                            _exitcode = -1
                        # Skip exitcode=0: that's Guardian's own graceful
                        # stop_all during clean titan_main shutdown — not a crash.
                        if _exitcode != 0:
                            matched_reason = "died"
                            matched_mod = m.group(1)

            if matched_reason and matched_mod:
                d = per_module.setdefault(matched_mod, {
                    "heartbeat_timeout": 0, "rss_exceeded": 0, "died": 0,
                    "total": 0, "last_event_ts": "", "last_reason": "",
                })
                d[matched_reason] += 1
                d["total"] += 1
                # Walking backwards, the FIRST match we see is the most recent
                if not d["last_event_ts"]:
                    d["last_event_ts"] = ts_str
                    d["last_reason"] = matched_reason

        grand_total = sum(d["total"] for d in per_module.values())
        return {
            "per_module": per_module,
            "grand_total": grand_total,
            "window_hours": window_hours,
            "window_start": window_start,
            "window_complete": hit_window_start,  # False = log didn't reach back 24h
        }

    def _live_health(url: str) -> tuple[int, float] | None:
        try:
            t0 = dt.datetime.utcnow()
            r = requests.get(f"{url}/health", timeout=10)
            elapsed = (dt.datetime.utcnow() - t0).total_seconds()
            return r.status_code, elapsed
        except Exception:
            return None

    titans = [
        ("T1", "/tmp/titan1_watchdog.log", "local",
         "http://127.0.0.1:7777", "/tmp/titan_brain.log"),
        ("T2", "/tmp/titan2_watchdog.log", "remote",
         "http://10.135.0.6:7777", "/tmp/titan2_brain.log"),
        ("T3", "/tmp/titan3_watchdog.log", "remote",
         "http://10.135.0.6:7778", "/tmp/titan3_brain.log"),
    ]

    overall_state = "STABLE"
    summary_rows = []

    for tid, wlog, mode, url, blog in titans:
        print(f"\n  {tid}  (watchdog={wlog}, api={url})")
        print("  " + "-" * 86)

        content = _read_local(wlog) if mode == "local" else _read_remote(wlog)
        if content is None:
            print(f"    ✗ Cannot read watchdog log (mode={mode})")
            summary_rows.append((tid, "?", 0, 0, "log unreadable"))
            overall_state = "DEGRADED"
            continue

        in_window, total = _parse_watchdog(content)
        rate_per_hour = len(in_window) / hours if hours > 0 else 0

        # Verdict
        if len(in_window) == 0:
            verdict = "✓ STABLE"
            verdict_state = "STABLE"
        elif len(in_window) <= 3:
            verdict = "⚠ FRAGILE"
            verdict_state = "FRAGILE"
            if overall_state == "STABLE":
                overall_state = "FRAGILE"
        else:
            verdict = "✗ CRASH-LOOP"
            verdict_state = "CRASH_LOOP"
            overall_state = "CRASH_LOOP"

        print(f"    {verdict}   restarts in window: {len(in_window)}  "
              f"(rate: {rate_per_hour:.2f}/h)  total in log: {total}")

        # Per-hour distribution (only if any restarts in window)
        if in_window:
            hour_counts = {}
            for t in in_window:
                hkey = t.strftime("%Y-%m-%d %H:00")
                hour_counts[hkey] = hour_counts.get(hkey, 0) + 1
            recent_hours = sorted(hour_counts.items())[-10:]
            print(f"    Recent hours with restarts:")
            for hk, c in recent_hours:
                bar = "█" * min(c, 30)
                print(f"      {hk}  {c:>2d} restarts  {bar}")

        # Live /health cross-check
        h = _live_health(url)
        if h is None:
            print(f"    ✗ /health UNREACHABLE")
            api_state = "UNREACHABLE"
        else:
            code, elapsed = h
            api_health = "✓" if code == 200 and elapsed < 2.0 else (
                "⚠" if code == 200 else "✗")
            print(f"    {api_health} /health = HTTP {code}  ({elapsed:.2f}s)")
            api_state = f"{code}/{elapsed:.1f}s"

        # Cross-check: brain log RESTART boundaries (only on T1 — local read)
        brain = _read_local(blog) if mode == "local" else _read_remote(blog)
        if mode == "local" and brain:
            bnd = brain.count("RESTART boundary")
            print(f"    Brain log: {bnd} 'RESTART boundary' markers (since rotation)")

        # Guardian in-process module restart tracking — same 24h window as
        # watchdog process-crash counts above.
        mod_restarts = None
        if brain:
            mod_restarts = _parse_module_restarts(brain, window_hours=hours,
                                                  ref_dt=now)
            scope_note = ("" if mod_restarts["window_complete"]
                          else f" [log reaches back less than {hours}h]")
            if mod_restarts["grand_total"] > 0:
                rate = mod_restarts["grand_total"] / max(hours, 1)
                print(f"    Guardian module restarts (last {hours}h): "
                      f"{mod_restarts['grand_total']} total "
                      f"(rate: {rate:.2f}/h){scope_note}")
                # Sort modules by total DESC
                sorted_mods = sorted(mod_restarts["per_module"].items(),
                                     key=lambda kv: -kv[1]["total"])
                for mod, d in sorted_mods[:10]:
                    parts = []
                    if d["heartbeat_timeout"]:
                        parts.append(f"heartbeat_timeout={d['heartbeat_timeout']}")
                    if d["rss_exceeded"]:
                        parts.append(f"rss_exceeded={d['rss_exceeded']}")
                    if d["died"]:
                        parts.append(f"died={d['died']}")
                    reason_str = ", ".join(parts)
                    last_ts = d["last_event_ts"] or "?"
                    print(f"      {mod:<12} {d['total']:>3}x   "
                          f"{reason_str:<40}  last: {last_ts}")
            else:
                print(f"    Guardian module restarts (last {hours}h): 0{scope_note}")

        summary_rows.append((tid, verdict_state, len(in_window),
                             rate_per_hour, api_state,
                             mod_restarts["grand_total"] if mod_restarts else 0))

    # Final verdict
    print()
    print("  " + "=" * 86)
    print(f"  OVERALL: {overall_state}")
    print()
    print(f"  {'Titan':<6} {'State':<12} {'Crashes':>9} {'Rate/h':>8}  "
          f"{'API':<20} {'ModRestart':>11}")
    print(f"  {'-'*6} {'-'*12} {'-'*9} {'-'*8}  {'-'*20} {'-'*11}")
    for tid, state, n, rate, api, mod_n in summary_rows:
        print(f"  {tid:<6} {state:<12} {n:>9d} {rate:>8.2f}  "
              f"{api:<20} {mod_n:>11d}")

    if overall_state == "CRASH_LOOP":
        print()
        print(f"  🛑 CRASH-LOOP DETECTED — HALT ALL FEATURE WORK")
        print(f"     Root-cause restart cause before any new commits to bus/Guardian/proxies.")

    print()
    print("=" * 90)
    return overall_state


# ═══════════════════════════════════════════════════════════════════════════
# STATE-DIMS + BUS-CONSUMERS — dimensional + bus-contract audit tools
# ═══════════════════════════════════════════════════════════════════════════
# Added 2026-04-14 to support the state_register 130D foundation rFP.
# Tool 1 answers: "For every state-vector reference in the codebase, what
# dimension does it assume?" Tool 2 answers: "For bus message X, who emits
# it with what keys and who reads what keys?"

_STATE_VEC_NAMES = {
    "sv", "state_vector", "state_vec", "current_state", "felt_tensor",
    "tensor", "tensors", "full_30dt", "full_65dt", "full_130dt",
    "_d6_sv", "_d6b_sv", "_lf_sv", "query_vec",
}
_DIM_LITERALS = {30, 45, 65, 90, 130, 132}


def _extract_dim_hits(path: Path) -> list[dict]:
    """AST scan one file for dim-sensitive patterns. Returns list of hits."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    src_lines = src.splitlines()
    hits = []

    def _ctx(lineno: int) -> str:
        if 1 <= lineno <= len(src_lines):
            return src_lines[lineno - 1].strip()[:140]
        return ""

    for node in ast.walk(tree):
        # Pattern A: string literal "full_Xdt" anywhere
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            m = re.fullmatch(r"full_(\d+)dt", node.value)
            if m:
                hits.append({
                    "kind": "snapshot_key", "dim": int(m.group(1)),
                    "line": getattr(node, "lineno", 0), "snippet": _ctx(getattr(node, "lineno", 0)),
                })

        # Pattern B: method call *.get_full_Nd() / .get_full_Xdt() /
        # .get_full_Nd_topology() (rFP #1 added the _topology family).
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            m = re.fullmatch(r"get_full_(\d+)d?t?(_topology)?", name)
            if m:
                hits.append({
                    "kind": "register_method", "dim": int(m.group(1)),
                    "line": node.lineno, "snippet": _ctx(node.lineno),
                })
            if name in ("recall_by_state", "_cosine_sim", "bookmark_insight"):
                hits.append({
                    "kind": "recall_site", "dim": None, "method": name,
                    "line": node.lineno, "snippet": _ctx(node.lineno),
                })

        # Pattern C: slicing subscript vec[:N] with literal N
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            lo, hi = node.slice.lower, node.slice.upper
            if hi is not None and isinstance(hi, ast.Constant) and isinstance(hi.value, int):
                if hi.value in _DIM_LITERALS:
                    # Confirm target is a state-vectory name (best-effort)
                    target_name = None
                    if isinstance(node.value, ast.Name):
                        target_name = node.value.id
                    elif isinstance(node.value, ast.Attribute):
                        target_name = node.value.attr
                    if target_name in _STATE_VEC_NAMES or target_name and target_name.endswith("sv"):
                        hits.append({
                            "kind": "slice", "dim": hi.value, "target": target_name,
                            "line": node.lineno, "snippet": _ctx(node.lineno),
                        })

        # Pattern D: len(x) >= N or len(x) == N where N in dim literals
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) \
                    and node.left.func.id == "len" and isinstance(node.ops[0], (ast.GtE, ast.Eq, ast.Gt)):
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) \
                        and isinstance(node.comparators[0].value, int) \
                        and node.comparators[0].value in _DIM_LITERALS:
                    tgt = None
                    if node.left.args and isinstance(node.left.args[0], ast.Name):
                        tgt = node.left.args[0].id
                    hits.append({
                        "kind": "len_guard", "dim": node.comparators[0].value,
                        "target": tgt, "line": node.lineno, "snippet": _ctx(node.lineno),
                    })

    return hits


def run_state_dims():
    """Dimensional audit: every state-vector reference with its assumed dim."""
    print()
    print("=" * 90)
    print("  STATE-DIMS — dimensional audit across titan_plugin/ + scripts/")
    print("=" * 90)

    by_kind: dict[str, list[dict]] = defaultdict(list)
    by_dim: dict[int, list[dict]] = defaultdict(list)
    dims_produced: set[int] = set()
    dims_consumed: set[int] = set()

    all_files = []
    for d in (*SCAN_DIRS, *SCRIPTS_DIRS):
        if d.exists():
            all_files.extend(sorted(d.rglob("*.py")))

    for f in all_files:
        for h in _extract_dim_hits(f):
            h["file"] = rel(f)
            by_kind[h["kind"]].append(h)
            if h.get("dim"):
                by_dim[h["dim"]].append(h)
                if h["kind"] in ("register_method", "snapshot_key"):
                    # Conservative producer classification: emission from register
                    # or a key explicitly named as producer output.
                    if h["kind"] == "register_method":
                        dims_produced.add(h["dim"])
                if h["kind"] in ("slice", "len_guard"):
                    dims_consumed.add(h["dim"])

    # Producers summary
    print("\nPRODUCERS (state_register methods)")
    print("-" * 90)
    prods = by_kind.get("register_method", [])
    if not prods:
        print("  (none found)")
    else:
        for h in sorted(prods, key=lambda x: (x["file"], x["line"])):
            print(f"  {h['dim']:>3}D  {h['file']}:{h['line']:<5}  {h['snippet']}")

    # Snapshot key usage
    print("\nSNAPSHOT KEYS (msg['full_Xdt'] / .get('full_Xdt'))")
    print("-" * 90)
    keys = by_kind.get("snapshot_key", [])
    if not keys:
        print("  (none found)")
    else:
        for h in sorted(keys, key=lambda x: (x["dim"], x["file"], x["line"])):
            print(f"  {h['dim']:>3}D  {h['file']}:{h['line']:<5}  {h['snippet']}")

    # Slices + len guards, grouped by dim
    print("\nSLICES + LEN GUARDS on state-vector-named locals")
    print("-" * 90)
    for dim in sorted(by_dim.keys()):
        rows = [h for h in by_dim[dim] if h["kind"] in ("slice", "len_guard")]
        if not rows:
            continue
        print(f"  ── {dim}D ── ({len(rows)} site{'s' if len(rows) != 1 else ''})")
        for h in sorted(rows, key=lambda x: (x["file"], x["line"])):
            kind_tag = "slice" if h["kind"] == "slice" else "len>="
            print(f"    {kind_tag:<7} {h.get('target') or '?':<18} {h['file']}:{h['line']:<5}  {h['snippet']}")

    # Recall sites
    print("\nRECALL / COSINE / BOOKMARK call sites")
    print("-" * 90)
    recalls = by_kind.get("recall_site", [])
    if not recalls:
        print("  (none found)")
    else:
        for h in sorted(recalls, key=lambda x: (x["method"], x["file"], x["line"])):
            print(f"  {h['method']:<22} {h['file']}:{h['line']:<5}  {h['snippet']}")

    # Mismatch summary
    print("\n" + "=" * 90)
    print("DIMENSIONAL CONTRACT CHECK")
    print("-" * 90)
    print(f"  Producers emit:   {sorted(dims_produced) or '(none detected as register methods)'}")
    print(f"  Consumers assume: {sorted(dims_consumed) or '(none detected)'}")
    missing = dims_consumed - dims_produced
    orphan = dims_produced - dims_consumed
    if missing:
        print(f"  🔴 Consumer expects dim(s) {sorted(missing)} but NO register method produces them")
    if orphan:
        print(f"  ⚠  Register produces dim(s) {sorted(orphan)} but no consumer slices/guards on them")
    if not missing and not orphan:
        print("  ✓ All consumed dims have a matching producer")
    print("=" * 90)


def run_bus_consumers(msg_type: str):
    """Per-message bus contract: publishers, subscribers, keys accessed."""
    msg_type = msg_type.strip().upper()
    print()
    print("=" * 90)
    print(f"  BUS-CONSUMERS — contract for message type: {msg_type}")
    print("=" * 90)

    graph = load_graph()
    wiring = graph.get("bus_wiring", {}).get(msg_type)
    if not wiring:
        print(f"\n  No wiring found for {msg_type!r}.")
        print(f"  (Run: python {sys.argv[0]} scan)")
        print(f"  Available message types: {len([k for k in graph.get('bus_wiring', {}) if not k.startswith('_SUB:')])}")
        return

    # Publishers (make_msg / publish)
    print("\nPUBLISHERS  (make_msg / publish call sites)")
    print("-" * 90)
    pubs = wiring.get("publishers", []) + wiring.get("send_msgs", [])
    if not pubs:
        print("  (none)")
    else:
        for p in sorted(pubs, key=lambda x: (x["file"], x["line"])):
            dst = p.get("dst", "—")
            print(f"  {p['file']}:{p['line']:<5}  → {dst:<18}  {p.get('context', '').strip()[:100]}")

    # Consumers (msg_type == "X" checks)
    print("\nCONSUMERS  (msg_type equality checks in dispatch)")
    print("-" * 90)
    cons = wiring.get("consumers", [])
    if not cons:
        print("  (none — message may be published but never explicitly routed)")
    else:
        for c in sorted(cons, key=lambda x: (x["file"], x["line"])):
            print(f"  {c['file']}:{c['line']:<5}  {c.get('context', '').strip()[:120]}")

    # Payload-key access — scan consumer files for msg['key'] / .get('key') near the consumer site
    print("\nPAYLOAD KEYS ACCESSED  (scanned in each consumer file)")
    print("-" * 90)
    key_hits: dict[str, list[dict]] = defaultdict(list)
    for c in cons:
        fp = PROJECT_ROOT / c["file"]
        if not fp.exists():
            continue
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(fp))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # msg.get("key") or data.get("key")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and node.args \
                    and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                owner = None
                if isinstance(node.func.value, ast.Name):
                    owner = node.func.value.id
                if owner in ("msg", "data", "payload", "snapshot", "message", "m", "envelope"):
                    key_hits[node.args[0].value].append({
                        "file": c["file"], "line": node.lineno, "owner": owner,
                    })
            # msg["key"] subscript
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                owner = None
                if isinstance(node.value, ast.Name):
                    owner = node.value.id
                if owner in ("msg", "data", "payload", "snapshot", "message", "m", "envelope"):
                    key_hits[node.slice.value].append({
                        "file": c["file"], "line": node.lineno, "owner": owner,
                    })
    if not key_hits:
        print("  (no payload keys statically detected — dispatch may destructure differently)")
    else:
        for key in sorted(key_hits.keys()):
            sites = key_hits[key]
            files = sorted({s["file"] for s in sites})
            print(f"  {key!r:<26}  ({len(sites)} access{'es' if len(sites) != 1 else ''} in {len(files)} file{'s' if len(files) != 1 else ''})")
            for s in sites[:6]:
                print(f"     {s['file']}:{s['line']:<5}  owner={s['owner']}")
            if len(sites) > 6:
                print(f"     ... {len(sites) - 6} more")

    print()
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════
# NS PROGRAMS — Stage 0 of rFP β (NS Program Signal Restoration)
# ══════════════════════════════════════════════════════════════════════════

NS_PROGRAMS = [
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "INSPIRATION",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "METABOLISM", "VIGILANCE",
]

NS_DATA_DIR = PROJECT_ROOT / "data" / "neural_nervous_system"


def _ns_color(text, code):
    if not sys.stdout.isatty():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _ns_load_buffer(program: str, data_dir: Path = NS_DATA_DIR):
    path = data_dir / f"{program.lower()}_buffer.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _ns_load_weights(program: str, data_dir: Path = NS_DATA_DIR):
    path = data_dir / f"{program.lower()}_weights.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _ns_vm_program_set():
    """Parse nervous_system.py to find which programs have VM baseline.

    Post-Stage-1 (auto-discovery refactor): looks for _make_*_program function
    definitions instead of the old hardcoded return-dict. Each function
    matches one program (e.g. _make_metabolism_program → METABOLISM).
    """
    ns_path = PROJECT_ROOT / "titan_plugin" / "logic" / "nervous_system.py"
    if not ns_path.exists():
        return set()
    src = ns_path.read_text()
    # Auto-discovery convention: every _make_X_program function = X uppercase
    matches = re.findall(r"^def _make_([a-z_]+)_program\b", src, re.MULTILINE)
    return {m.upper() for m in matches}


def _pct_nz(values, eps=0.01):
    """Percent of values with magnitude > eps. Default 0.01 filters NN noise floor
    (urgencies below 0.01 are effectively zero for downstream consumers)."""
    if not values:
        return 0.0
    return 100.0 * sum(1 for v in values if abs(v) > eps) / len(values)


def _ns_program_stats(program: str, data_dir: Path = NS_DATA_DIR):
    buf = _ns_load_buffer(program, data_dir)
    w = _ns_load_weights(program, data_dir)
    if not buf:
        return {"program": program, "status": "no_buffer"}
    urg = buf.get("urgencies") or []
    vmb = buf.get("vm_baselines") or []
    rew = buf.get("rewards") or []
    fired = buf.get("fired") or []
    n = len(urg)
    if n == 0:
        return {"program": program, "status": "empty_buffer", "n": 0}
    return {
        "program": program, "status": "ok", "n": n,
        "avg_u": sum(urg) / n, "max_u": max(urg), "pct_nz_u": _pct_nz(urg),
        "avg_vm": (sum(vmb) / n) if vmb else 0.0,
        "max_vm": max(vmb) if vmb else 0.0,
        "pct_nz_vm": _pct_nz(vmb),
        "fire_pct": 100.0 * sum(1 for f in fired if f) / n,
        "avg_r": (sum(rew) / n) if rew else 0.0,
        "pct_nz_r": _pct_nz(rew),
        "input_dim": (w or {}).get("input_dim", 0),
        "feature_set": (w or {}).get("feature_set", "?"),
        "last_loss": (w or {}).get("last_loss", 0.0),
        "total_updates": (w or {}).get("total_updates", 0),
    }


def run_ns_signals(titan: str = "T1") -> int:
    """Per-program urgency distribution — answers 'which programs are alive?'"""
    print(f"\nNS SIGNALS — {titan}")
    print("=" * 108)

    if titan != "T1":
        print(f"  Remote titan {titan} support TBD (read local disk only in v1)")
        return 2

    vm_programs = _ns_vm_program_set()

    print(f"  {'PROGRAM':<12} {'N':>5} {'AVG_U':>7} {'MAX_U':>7} {'%NZ_U':>6} "
          f"{'%NZ_VM':>7} {'VM':>4} {'FIRE%':>6} {'LOSS':>11} {'UPDATES':>9}  VERDICT")
    print("  " + "-" * 106)

    dead_count = 0
    for program in NS_PROGRAMS:
        s = _ns_program_stats(program)
        if s["status"] != "ok":
            print(f"  {program:<12}  [{s['status']}]")
            continue
        vm_str = "YES" if program in vm_programs else "NO "
        # DEAD: either almost no signal above noise floor, OR max urgency is
        # trivially small (avg < 0.01 AND max < 0.05 with decent sample count)
        is_dead = (
            s["n"] > 500
            and (s["pct_nz_u"] < 5 or (s["avg_u"] < 0.01 and s["max_u"] < 0.05))
        )
        if is_dead:
            verdict, cc = "DEAD", "31"
            dead_count += 1
        elif s["pct_nz_u"] < 30:
            verdict, cc = "LOW", "33"
        else:
            verdict, cc = "OK", "32"
        print(f"  {program:<12} {s['n']:>5} {s['avg_u']:>7.4f} {s['max_u']:>7.4f} "
              f"{s['pct_nz_u']:>5.1f}% {s['pct_nz_vm']:>6.1f}% {vm_str:>4} "
              f"{s['fire_pct']:>5.1f}% {s['last_loss']:>11.8f} {s['total_updates']:>9}  "
              f"{_ns_color(verdict, cc)}")

    print()
    missing = set(NS_PROGRAMS) - vm_programs
    vm_line = f"  VM-supervised: {len(vm_programs)}/11 programs"
    if missing:
        vm_line += f"  |  MISSING: {', '.join(sorted(missing))}"
    print(vm_line)
    if dead_count > 0:
        print(f"  {_ns_color(f'{dead_count} DEAD programs (pct_nz_u < 5% with >500 samples)', '31;1')}")
        return 1
    return 0


def run_ns_persistence_check(titan: str = "T1") -> int:
    """Verify all 11 programs have weights + buffer on disk + training state."""
    print(f"\nNS PERSISTENCE CHECK — {titan}")
    print("=" * 80)
    data_dir = NS_DATA_DIR
    failures = []

    for program in NS_PROGRAMS:
        w = data_dir / f"{program.lower()}_weights.json"
        b = data_dir / f"{program.lower()}_buffer.json"
        wsz = w.stat().st_size if w.exists() else 0
        bsz = b.stat().st_size if b.exists() else 0
        w_ok = w.exists() and wsz > 100
        b_ok = b.exists() and bsz > 100
        status = "OK" if (w_ok and b_ok) else "FAIL"
        print(f"  {program:<12} weights={'✓' if w_ok else '✗'} ({wsz:>7}B)  "
              f"buffer={'✓' if b_ok else '✗'} ({bsz:>7}B)  [{status}]")
        if not (w_ok and b_ok):
            failures.append(program)

    for fname, min_size in [("training_state.json", 50), ("hormonal_state.json", 200)]:
        p = data_dir / fname
        sz = p.stat().st_size if p.exists() else 0
        ok = p.exists() and sz > min_size
        print(f"  {fname:<24} {'✓' if ok else '✗'} ({sz}B)")
        if not ok:
            failures.append(fname)

    sb = data_dir / "ns_training_backup.db"
    sb_sz = sb.stat().st_size if sb.exists() else 0
    sb_ok = sb.exists() and sb_sz > 1024
    print(f"  ns_training_backup.db    {'✓' if sb_ok else '✗'} ({sb_sz}B)")
    if not sb_ok:
        failures.append("ns_training_backup.db")

    snap_dir = data_dir / "snapshots"
    n_snap_sets = len(list(snap_dir.glob("step_*_training_state.json"))) if snap_dir.exists() else 0
    n_rfp_snaps = len(list((snap_dir / "pre_rFP_beta").glob("*"))) if (snap_dir / "pre_rFP_beta").exists() else 0
    print(f"  snapshots/               {'✓' if snap_dir.exists() else '✗'} "
          f"(rolling: {n_snap_sets} sets, pre_rFP_beta: {n_rfp_snaps} files)")

    print()
    if failures:
        print(f"  {_ns_color('FAIL', '31;1')}: {len(failures)} missing: {failures}")
        return 1
    print(f"  {_ns_color('PASS', '32;1')}: all 11 programs + state + hormone + SQLite + snapshots present")
    return 0


def run_ns_health(titan: str = "T1") -> int:
    """Full per-program health report — combines ns-signals + persistence + training summary."""
    rc_sig = run_ns_signals(titan)
    rc_pers = run_ns_persistence_check(titan)

    ts_path = NS_DATA_DIR / "training_state.json"
    if ts_path.exists():
        try:
            ts = json.loads(ts_path.read_text())
            import time as _t
            age_s = _t.time() - ts.get("last_train_ts", 0)
            age_str = f"{age_s:.0f}s ago" if age_s < 3600 else f"{age_s/3600:.1f}h ago"
            print()
            print(f"  Global state: transitions={ts.get('total_transitions', 0):,}  "
                  f"train_steps={ts.get('total_train_steps', 0):,}  "
                  f"last_train={age_str}")
            program_names = ts.get("program_names", [])
            print(f"  Registered programs: {len(program_names)}/11 "
                  f"— {', '.join(program_names)}")
        except Exception as e:
            print(f"  training_state.json parse error: {e}")

    h_path = NS_DATA_DIR / "hormonal_state.json"
    if h_path.exists():
        try:
            h = json.loads(h_path.read_text())
            print()
            print(f"  Hormonal system: maturity={h.get('maturity', 0):.3f}  "
                  f"hormones={len(h.get('hormones', {}))}")
        except Exception:
            pass

    return rc_sig | rc_pers


def run_ns_learning(titan: str = "T1", since_hours: int = 24) -> int:
    """Learning trajectory per program — is the NN actually learning?"""
    import time as _t
    print(f"\nNS LEARNING — {titan} (last {since_hours}h)")
    print("=" * 100)
    cutoff = _t.time() - since_hours * 3600

    print(f"  {'PROGRAM':<12} {'WEIGHT_AGE':>14} {'LAST_LOSS':>13} {'STATE':<15} "
          f"{'UPDATES':>9}  RECENT?")
    print("  " + "-" * 98)

    active_count = 0
    collapsed_count = 0
    for program in NS_PROGRAMS:
        w = _ns_load_weights(program)
        if not w:
            continue
        last_loss = w.get("last_loss", 0.0)
        updates = w.get("total_updates", 0)
        wpath = NS_DATA_DIR / f"{program.lower()}_weights.json"
        mtime = wpath.stat().st_mtime if wpath.exists() else 0
        age_s = _t.time() - mtime
        recent = mtime > cutoff

        if last_loss < 1e-6 and updates > 10000:
            state, cc = "COLLAPSED", "31"
            collapsed_count += 1
        elif last_loss < 0.01:
            state, cc = "converged", "33"
        else:
            state, cc = "active", "32"
            active_count += 1

        age_str = f"{age_s/60:.0f}m" if age_s < 3600 else (
            f"{age_s/3600:.1f}h" if age_s < 86400 else f"{age_s/86400:.1f}d")
        print(f"  {program:<12} {age_str:>14} {last_loss:>13.10f} "
              f"{_ns_color(state, cc):<24} {updates:>9}  {'✓' if recent else 'stale'}")

    print()
    print(f"  Active: {active_count}/11 | Collapsed: {collapsed_count}/11")
    if collapsed_count >= 6:
        print(f"  {_ns_color('DEGENERATE — majority of programs have collapsed to zero (rFP β scope)', '31;1')}")
        return 1
    return 0


def run_backup_diagnostics(all_titans: bool = False):
    """rFP_backup_worker Phase 5 — backup health per Titan.

    Verifies the master invariant (rFP §5.4): full state recoverable from ≤24h
    of backup artifacts. Reads /v4/backup/status + /v4/backup/wallet-runway.
    """
    import requests
    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("BACKUP HEALTH — rFP_backup_worker master invariant")
    print("=" * 78)

    overall_ok = True
    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            r = requests.get(f"{url}/v4/backup/status", timeout=10)
            if r.status_code != 200:
                print(f"    ✗ /v4/backup/status HTTP {r.status_code}")
                overall_ok = False
                continue
            d = r.json().get("data", {})
            mode = d.get("mode", "?")
            invariant = d.get("master_invariant_ok", False)
            records = d.get("records", {})
            local = d.get("local_snapshots", {})
            dry = d.get("last_dry_run", {})

            inv_mark = "✓" if invariant else "✗"
            print(f"    {inv_mark} Mode: {mode}  |  Master invariant: "
                  f"{'OK' if invariant else 'AT RISK'}")

            if records:
                for btype, rec in records.items():
                    age_h = rec.get("age_hours", -1)
                    tx = (rec.get("arweave_tx") or "local")[:16]
                    sz = rec.get("size_mb", 0)
                    mark = "✓" if age_h < (30 if btype == "personality"
                                           else 24 * 8) else "⚠"
                    print(f"    {mark} Last {btype:12s}: {age_h:>5.1f}h ago"
                          f"  size={sz:>6.1f} MB  tx={tx}")
            else:
                print(f"    ⚠ No backup records found (first boot?)")

            print(f"    ℹ Local snapshots:"
                  f" personality={local.get('personality', 0)}"
                  f" soul={local.get('soul', 0)}"
                  f" timechain={local.get('timechain', 0)}"
                  f" total={local.get('total_mb', 0):.1f} MB")

            if dry:
                dry_ok = "✓" if dry.get("ok") else "✗"
                print(f"    {dry_ok} Last dry-run:"
                      f" {'OK' if dry.get('ok') else 'FAIL'}"
                      f"  steps={dry.get('steps', {})}")

            # Runway (T1 only normally — mainnet_arweave mode)
            if mode == "mainnet_arweave":
                try:
                    r2 = requests.get(f"{url}/v4/backup/wallet-runway", timeout=10)
                    if r2.status_code == 200:
                        rd = r2.json().get("data", {})
                        tier = rd.get("tier", "?")
                        days = rd.get("days_runway", 0)
                        irys = rd.get("irys_sol", 0)
                        mark = {"green": "✓", "yellow": "ℹ",
                                "orange": "⚠", "red": "✗"}.get(tier, "?")
                        print(f"    {mark} Irys runway: {irys:.4f} SOL"
                              f" ≈ {days:.1f} days (tier={tier})")
                        if tier in ("orange", "red"):
                            overall_ok = False
                except Exception as e:
                    print(f"    ⚠ Runway query error: {e}")

            # Phase 8 — chain integrity (local mirror of on-chain anchor sequence)
            try:
                from titan_plugin.logic.backup_chain import verify_chain_file
                chain_result = verify_chain_file(tid)
                if chain_result["length"] == 0:
                    print(f"    ℹ Anchor chain: empty (Phase 8 not yet active "
                          f"or no anchors written)")
                elif chain_result["ok"]:
                    print(f"    ✓ Anchor chain: {chain_result['length']} entries, "
                          f"integrity intact")
                else:
                    print(f"    ✗ Anchor chain: BROKEN at index "
                          f"{chain_result['break_index']} — "
                          f"{chain_result['break_reason']}")
                    overall_ok = False
            except Exception as e:
                print(f"    ⚠ Chain check error: {e}")

            if not invariant:
                overall_ok = False
        except Exception as e:
            print(f"    ✗ Error: {e}")
            overall_ok = False

    print()
    print("=" * 78)
    print(f"OVERALL: {'ALL GREEN' if overall_ok else 'AT RISK — investigate'}")
    print()


def run_backup_verify_chain(all_titans: bool = False) -> int:
    """rFP_backup_worker Phase 8.2 — verbose chain walk.

    Reads data/backup_anchor_chain_{titan_id}.json and prints each anchor with
    per-entry integrity markers. Returns exit code 0 if all chains are intact,
    1 otherwise — so scripts/CI can gate on this.

    Note: runs LOCALLY on the machine it's invoked on — chain files live on
    each Titan's disk. With --all, attempts to read T2/T3 chain files via
    their API (not yet wired); for now recommends running directly on each
    VPS for multi-Titan audit.
    """
    from titan_plugin.logic.backup_chain import read_chain, verify_chain

    titans = ["T1", "T2", "T3"] if all_titans else ["T1"]
    print()
    print("BACKUP ANCHOR CHAIN — verify-chain (rFP_backup_worker Phase 8.2)")
    print("=" * 78)
    overall = 0
    for tid in titans:
        print(f"\n  {tid}")
        print("  " + "-" * 72)
        try:
            anchors = read_chain(tid)
        except Exception as e:
            print(f"    ✗ read error: {e}")
            overall = 1
            continue
        if not anchors:
            print("    ℹ chain file absent/empty — no anchors to verify")
            continue
        result = verify_chain(anchors)
        for i, a in enumerate(anchors):
            h = a.get("archive_hash", "?")[:16]
            p = a.get("prev_anchor_hash", "")[:16] or "genesis"
            t = a.get("backup_type", "?")
            tx = (a.get("tx") or "?")[:16]
            mark = "✓"
            if not result["ok"] and i == result["break_index"]:
                mark = "✗"
            print(f"    {mark} [{i:>3d}] {t:<12s} h={h} prev={p:<16} tx={tx}")
        if result["ok"]:
            print(f"\n    ✓ Chain INTACT: {result['length']} entries, "
                  f"all prev-hashes validate")
        else:
            print(f"\n    ✗ Chain BROKEN at index {result['break_index']}: "
                  f"{result['break_reason']}")
            overall = 1
    print()
    print("=" * 78)
    print(f"OVERALL: {'ALL CHAINS INTACT' if overall == 0 else 'CHAIN BREAK DETECTED'}")
    print()
    return overall


def run_sovereign_ops(all_titans: bool = False):
    """rFP_backup_worker §5.7 — single-command operational readout.

    Combines backup + meditation + wallet + TimeChain + identity status for
    Maker's session-start ritual.
    """
    import requests
    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("SOVEREIGN OPERATIONS — Titan(s) operational health")
    print("=" * 78)

    for tid, url in titans:
        print(f"\n  {tid} SOVEREIGN OPS ({url.split('//')[1]})")
        print("  " + "-" * 72)
        rows = []

        # Backup
        try:
            r = requests.get(f"{url}/v4/backup/status", timeout=8)
            if r.status_code == 200:
                d = r.json().get("data", {})
                inv = d.get("master_invariant_ok", False)
                mode = d.get("mode", "?")
                recs = d.get("records", {})
                pers_age = recs.get("personality", {}).get("age_hours")
                status = (f"healthy (pers {pers_age:.1f}h ago, {mode})"
                          if (inv and pers_age is not None)
                          else f"⚠ master invariant at risk (mode={mode})")
                rows.append(("Backup", "✓" if inv else "⚠", status))
            else:
                rows.append(("Backup", "✗", f"HTTP {r.status_code}"))
        except Exception as e:
            rows.append(("Backup", "✗", f"{e}"))

        # Meditation
        try:
            r = requests.get(f"{url}/v4/meditation/health", timeout=8)
            if r.status_code == 200:
                md = r.json().get("data", {}) or {}
                last_age = md.get("last_meditation_age_s", 0) / 3600.0
                ok = md.get("healthy", False) or last_age < 6
                rows.append(("Meditation", "✓" if ok else "⚠",
                            f"last {last_age:.1f}h ago"))
            else:
                rows.append(("Meditation", "?", "endpoint n/a"))
        except Exception as e:
            rows.append(("Meditation", "?", f"{e}"))

        # Wallet + runway
        try:
            r = requests.get(f"{url}/v4/backup/wallet-runway", timeout=8)
            if r.status_code == 200:
                wd = r.json().get("data", {})
                if wd.get("available"):
                    rows.append(("Wallet/Irys",
                                 {"green": "✓", "yellow": "ℹ",
                                  "orange": "⚠", "red": "✗"}.get(wd.get("tier"), "?"),
                                 f"{wd.get('irys_sol', 0):.4f} SOL"
                                 f" ≈ {wd.get('days_runway', 0):.1f}d"
                                 f" ({wd.get('tier')})"))
                else:
                    rows.append(("Wallet/Irys", "N/A", "devnet / no keypair"))
        except Exception as e:
            rows.append(("Wallet/Irys", "?", f"{e}"))

        # TimeChain
        try:
            r = requests.get(f"{url}/v4/timechain/status", timeout=10)
            if r.status_code == 200:
                td = r.json().get("data", {})
                blocks = td.get("total_blocks", 0)
                forks = td.get("total_forks", 0)
                rows.append(("TimeChain", "✓",
                            f"{blocks:,} blocks, {forks} forks"))
        except Exception as e:
            rows.append(("TimeChain", "?", f"{e}"))

        # Identity
        try:
            r = requests.get(f"{url}/v4/identity/profile", timeout=8)
            if r.status_code == 200:
                ident = r.json().get("data", {}) or {}
                tid_live = ident.get("titan_id", tid)
                rows.append(("Identity", "✓", f"titan_id={tid_live}"))
            else:
                rows.append(("Identity", "?", f"HTTP {r.status_code}"))
        except Exception as e:
            rows.append(("Identity", "?", f"{e}"))

        for label, mark, detail in rows:
            print(f"    {mark} {label:<12s}: {detail}")

        all_ok = all(m == "✓" for m, _, _ in [(r[1], r[0], r[2]) for r in rows])
        print(f"    Status:     {'ALL SYSTEMS GREEN' if all_ok else 'DEGRADED'}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "scan":
        print("Scanning titan_plugin/ ...")
        graph = build_graph(SCAN_DIRS, SCRIPTS_DIRS)
        OUTPUT_FILE.write_text(json.dumps(graph, indent=2, default=str))
        print(f"Written: {rel(OUTPUT_FILE)}")
        print(f"  {graph['total_files']} files, {graph['total_lines']:,} lines")
        msg_count = len([k for k in graph["bus_wiring"] if not k.startswith("_SUB:")])
        print(f"  {msg_count} bus message types")
        print(f"  {len(graph['definitions'])} definitions indexed")
        print(f"  {len(graph['attr_index'])} attribute access patterns tracked")

    elif cmd == "query" and len(sys.argv) >= 3:
        graph = load_graph()
        pattern = sys.argv[2]

        if ":" in pattern:
            qtype, qval = pattern.split(":", 1)
        else:
            # Default: search everything
            qtype, qval = "all", pattern

        if qtype == "bus":
            query_bus(graph, qval)
        elif qtype in ("calls", "callers", "callees"):
            query_calls(graph, qval)
        elif qtype == "file":
            query_file(graph, qval)
        elif qtype == "attr":
            query_attr(graph, qval)
        elif qtype in ("depends", "impacts"):
            query_depends(graph, qval)
        elif qtype == "class":
            query_class(graph, qval)
        elif qtype == "all":
            # Search everywhere
            query_calls(graph, qval)
            query_attr(graph, qval)
            query_bus(graph, qval)
        else:
            print(f"Unknown query type: {qtype}")
            print("Available: bus, calls, callers, file, attr, depends, class, all")

    elif cmd == "wiring":
        graph = load_graph()
        show_wiring(graph)

    elif cmd == "summary":
        graph = load_graph()
        show_summary(graph)

    elif cmd == "worker":
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py worker <name>")
            print("Example: python scripts/arch_map.py worker spirit")
            sys.exit(1)
        graph = load_graph()
        query_worker(graph, sys.argv[2])

    elif cmd == "flow":
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py flow <MSG_TYPE>")
            print("Example: python scripts/arch_map.py flow MEDITATION_REQUEST")
            sys.exit(1)
        graph = load_graph()
        query_flow(graph, sys.argv[2])

    elif cmd == "params":
        graph = load_graph()
        show_params(graph)

    elif cmd == "health":
        if "--all" in sys.argv:
            # All 3 Titans
            run_health_checks("http://127.0.0.1:7777", "T1 (localhost:7777)")
            run_health_checks("http://10.135.0.6:7777", "T2 (10.135.0.6:7777)")
            run_health_checks("http://10.135.0.6:7778", "T3 (10.135.0.6:7778)")
        else:
            run_health_checks("http://127.0.0.1:7777", "T1 (localhost:7777)")
            if "--t2" in sys.argv:
                run_health_checks("http://10.135.0.6:7777", "T2 (10.135.0.6:7777)")
            if "--t3" in sys.argv:
                run_health_checks("http://10.135.0.6:7778", "T3 (10.135.0.6:7778)")

    elif cmd == "teacher":
        show_teacher_progress()

    elif cmd == "services":
        json_mode = "--json" in sys.argv
        run_services_diagnostics(json_output=json_mode)

    elif cmd == "audit":
        graph = load_graph()
        show_audit(graph)
        if "--live" in sys.argv:
            # Also run live wiring checks
            run_live_audit("http://127.0.0.1:7777", "T1")
            if "--all" in sys.argv:
                run_live_audit("http://10.135.0.6:7777", "T2")
                run_live_audit("http://10.135.0.6:7778", "T3")

    elif cmd == "cgn":
        run_cgn_telemetry("--all" in sys.argv)

    elif cmd == "cgn-reasoning-strategies":
        # 2026-04-27 closing DEFERRED ARCH-MAP-CGN-REASONING-STRATEGIES.
        # Diagnostic for the reasoning_strategy CGN consumer (fed by
        # reasoning.py:1821-1832 on COMMIT actions with outcome_score >=
        # cgn_emission_threshold). Surfaces "what does Titan consider
        # good reasoning" via top grounded rules. Pure observability.
        run_cgn_reasoning_strategies("--all" in sys.argv)

    elif cmd == "dead-wiring":
        # 2026-04-27 closing DEFERRED ARCH-MAP-DEAD-WIRING. Pass-through
        # wrapper to scripts/arch_map_dead_wiring.py — fully-built standalone
        # static-analysis scanner that pre-existed unregistered. Capabilities:
        # orphan-method detection (defined-but-zero-callers), pair-closure
        # (query_X without record_X), config-section tracing (toml sections
        # never .get()'d), rFP verification (--rfp), bus-flow audit
        # (--bus-flow), runtime overlay (--runtime-overlay), JSON output
        # (--json), git-context surfacing (--git-context). All args forwarded.
        import subprocess
        from pathlib import Path as _P
        sys.exit(subprocess.call(
            [sys.executable,
             str(_P(__file__).resolve().parent / "arch_map_dead_wiring.py")]
            + sys.argv[2:]))

    elif cmd == "timechain":
        run_timechain_diagnostics("--all" in sys.argv)

    elif cmd == "sensors":
        # rFP_phase1_sensory_wiring (2026-04-23): V6 5DT outer_body rich-
        # signal diagnostic. Shows outer_body dim values + saturation/stuck
        # flags across all 3 Titans. Archaeology found dims [3] entropy
        # and [4] thermal stuck at 0.5 exactly; [0-2] saturated near 1.0.
        # This subcommand is the at-a-glance verification that Phase 1
        # sensor wiring is delivering rich signal to outer_body.
        run_sensors_diagnostics(all_titans=("--all" in sys.argv or len(sys.argv) == 2))

    elif cmd == "contracts":
        # TUNING-012 v2 Sub-phase C (R2): cognitive contracts observability
        run_cognitive_contracts_diagnostics("--all" in sys.argv)

    elif cmd == "meta-audit":
        # Task 3: meta-reasoning observability for healing dynamics
        run_meta_audit_diagnostics("--all" in sys.argv)

    elif cmd == "metabolism-gates":
        # Mainnet Lifecycle Wiring rFP (2026-04-20): snapshot of live
        # metabolism gate decisions across Titans. Surfaces gates_enforced
        # flag + 10-min decision window + per-caller close rates. Used
        # during the 30-min focused observation pre-enforcement flip.
        run_metabolism_gates(all_titans="--all" in sys.argv,
                             json_mode="--json" in sys.argv)

    elif cmd == "cgn-signals":
        # 2026-04-14 rFP_meta_cgn_v3 § 10 Phase A: validates producer wiring
        # against SIGNAL_TO_PRIMITIVE table. Lists every emit_meta_cgn_signal
        # call site, verifies (consumer, event_type) has a mapping. Surfaces
        # orphans that would be silently dropped by the consumer (the Phase
        # 2 failure mode). Also reports live emission rates if Titans are
        # reachable. Run during session startup to catch drift early.
        # --audit-pattern (2026-04-19, META-CGN-PRODUCER-PATTERN): adds
        # TUNING-016 invariant check — every producer must gate emit via
        # EdgeDetector within ±30 lines of the call site.
        run_cgn_signals_audit(
            all_titans="--all" in sys.argv,
            audit_pattern="--audit-pattern" in sys.argv,
        )

    elif cmd == "producers":
        # 2026-04-15 rFP_meta_cgn_v3 Phase D observability: live + historical
        # META-CGN producer emissions across all 3 Titans. Live from
        # /v4/bus-health, historical from data/meta_cgn_emissions.jsonl
        # (Guardian drain-loop append-only). Survives restarts.
        # Flags: --history N, --since ISO, --producer src/event_type,
        #        --json, --expected. See run_producers_diagnostics docstring.
        run_producers_diagnostics(sys.argv[2:])

    elif cmd == "verify":
        # 2026-04-12: end-to-end pipeline health verification. Extensible pattern for
        # checking "is X actually working end-to-end?" in one command. Supports the
        # "verify before declare" discipline by making pipeline state inspection cheap.
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py verify <pipeline>")
            print("Pipelines:")
            print("  cgn-pipeline     — CGN worker + /dev/shm + 6 consumers + HAOV + knowledge flow")
            print("  search-pipeline  — knowledge router backends + cache + budgets + circuit breakers")
            sys.exit(1)
        pipeline = sys.argv[2].lower()
        if pipeline in ("cgn", "cgn-pipeline"):
            run_verify_cgn_pipeline("--all" in sys.argv)
        elif pipeline in ("search", "search-pipeline"):
            run_verify_search_pipeline("--all" in sys.argv)
        else:
            print(f"Unknown pipeline: {pipeline}")
            print("Available: cgn-pipeline, search-pipeline")
            sys.exit(1)

    elif cmd == "analyze-decisions":
        # KP-8.1: aggregate knowledge_router decision log (rFP §3.3 Essential D)
        hours = 24
        path = "data/logs/knowledge_router_decisions.jsonl"
        for i, a in enumerate(sys.argv[2:], start=2):
            if a == "--since" and i + 1 < len(sys.argv):
                try:
                    hours = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif a.startswith("--path="):
                path = a.split("=", 1)[1]
        run_analyze_decisions(path=path, hours=hours)

    elif cmd == "where":
        # 2026-04-12: comprehensive location search — supports "verify before declare" discipline.
        # Returns ALL paths a symbol might live in: file stems, class names, function names,
        # attribute patterns, bus messages, config keys. Designed to make "X doesn't exist"
        # claims impossible to make without full verification.
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py where <symbol>")
            print("Example: python scripts/arch_map.py where agno_hooks")
            print("Searches: file stems, class names, function names, bus messages,")
            print("          attribute access, config keys in titan_params.toml.")
            sys.exit(1)
        run_where(sys.argv[2])

    elif cmd == "report":
        run_report()

    elif cmd == "deploy":
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py deploy <t2|t3|all> [--restart]")
            sys.exit(1)
        target = sys.argv[2].lower()
        restart = "--restart" in sys.argv
        if target == "all":
            run_deploy(["t2", "t3"], restart=restart)
        else:
            run_deploy([target], restart=restart)

    elif cmd == "social-divergence":
        run_social_divergence()

    elif cmd == "knowledge":
        run_knowledge_comparison()

    elif cmd == "errors":
        run_errors(all_titans="--all" in sys.argv)

    elif cmd == "voice-diagnostics":
        # rFP_phase5_narrator_evolution §9: recent grounded-only X-gate decisions.
        # --hours N overrides default 24h window.
        _vd_hours = 24
        for i, a in enumerate(sys.argv):
            if a == "--hours" and i + 1 < len(sys.argv):
                try:
                    _vd_hours = int(sys.argv[i + 1])
                except ValueError:
                    pass
        run_voice_diagnostics(all_titans="--all" in sys.argv, hours=_vd_hours)

    elif cmd == "backup":
        # rFP_backup_worker Phase 5 — master invariant check per Titan.
        # Phase 8.2 — --verify-chain flag switches to verbose chain walk.
        if "--verify-chain" in sys.argv:
            sys.exit(run_backup_verify_chain(all_titans="--all" in sys.argv))
        run_backup_diagnostics(all_titans="--all" in sys.argv)

    elif cmd == "sovereign-ops":
        # rFP_backup_worker §5.7 — single-command op readout.
        run_sovereign_ops(all_titans="--all" in sys.argv)

    elif cmd == "filter-down":
        # 2026-04-14 rFP #2 Phase 7: V4/V5 FILTER_DOWN coexistence monitor.
        # Shows side-by-side state, gate checks, spirit strength, cold-start
        # progress. Run during V5 soak to confirm silent training + gate
        # readiness before Phase 8 publish flip.
        # 2026-04-15 Phase 8: added --gate-check for CI/deploy-gate use (exit 0
        # iff all mandatory 9-criteria pass across queried Titans).
        _exit = run_filter_down_status(
            all_titans="--all" in sys.argv,
            gate_check="--gate-check" in sys.argv,
        )
        if "--gate-check" in sys.argv:
            sys.exit(_exit)

    elif cmd == "traffic":
        hours = 24 if "--24h" in sys.argv else 1
        run_traffic(hours=hours)

    elif cmd == "meditation":
        # rFP_self_healing_meditation_cadence.md I2: cross-Titan correlation.
        # Queries /v4/meditation/health on T1/T2/T3 (or whichever are up) and
        # flags MEDITATION_INFRA_ALERT when all 3 go overdue within the same
        # 10-min window (infra issue, not per-Titan).
        run_meditation_health(all_titans="--all" in sys.argv)

    elif cmd == "state-dims":
        # 2026-04-14: dimensional audit — every state-vector reference with its
        # assumed dim. Built to support the state_register 130D foundation rFP.
        run_state_dims()

    elif cmd == "bus-consumers":
        # 2026-04-14: generic per-message bus contract — publishers, subscribers,
        # payload keys accessed. Reusable for future bus-contract changes.
        if len(sys.argv) < 3:
            print("Usage: python scripts/arch_map.py bus-consumers <MSG_TYPE>")
            print("Example: python scripts/arch_map.py bus-consumers STATE_SNAPSHOT")
            sys.exit(1)
        run_bus_consumers(sys.argv[2])

    elif cmd == "session-close":
        _sc_args = [a for a in sys.argv[2:] if not a.startswith("--")]
        title = " ".join(_sc_args).strip()
        run_session_close(title, commit="--no-commit" not in sys.argv)

    elif cmd == "meta-cgn":
        sub = sys.argv[2] if len(sys.argv) > 2 else "status"
        run_meta_cgn(sub)

    elif cmd == "meta-teacher":
        scope = "all"
        for a in sys.argv[2:]:
            if a in ("--t1", "--t2", "--t3"):
                scope = a[2:]
            elif a == "--all":
                scope = "all"
        run_meta_teacher(scope=scope)

    elif cmd == "emot-cgn":
        scope = "all"
        as_json = False
        for a in sys.argv[2:]:
            if a in ("--t1", "--t2", "--t3"):
                scope = a[2:]
            elif a == "--all":
                scope = "all"
            elif a == "--json":
                as_json = True
        run_emot_cgn(scope=scope, as_json=as_json)

    elif cmd == "meta-outer":
        # rFP_titan_meta_outer_layer — status + stats + recall diagnostic
        scope = "t1"
        as_json = False
        recall_person = None
        recall_topic = None
        for i, a in enumerate(sys.argv[2:]):
            if a in ("--t1", "--t2", "--t3"):
                scope = a[2:]
            elif a == "--all":
                scope = "all"
            elif a == "--json":
                as_json = True
            elif a == "--recall-person" and i + 3 < len(sys.argv):
                recall_person = sys.argv[i + 3]
            elif a == "--recall-topic" and i + 3 < len(sys.argv):
                recall_topic = sys.argv[i + 3]
        run_meta_outer(scope=scope, as_json=as_json,
                        recall_person=recall_person,
                        recall_topic=recall_topic)

    elif cmd == "preflight":
        # 2026-04-14: comprehensive pre-session-start test. Runs 9 checks
        # in sequence, unified PASS/WARN/FAIL verdict, HALT on critical
        # failures. Mandatory Step 0 of session_startup_protocol.md.
        run_preflight()

    elif cmd == "async-blocks":
        # 2026-04-14: scan titan_plugin/ for sync I/O reachable from async
        # functions. Built after 3 latent async-block bugs were found in
        # one session via py-spy. Run as part of session_startup_protocol.
        run_async_blocks_scan()

    elif cmd == "census":
        # 2026-04-14 Phase E.1: read /tmp/titan_bus_census.log and produce
        # diagnosis report. Requires titan_main running with TITAN_BUS_CENSUS=1.
        log_path = "/tmp/titan_bus_census.log"
        for a in sys.argv[2:]:
            if a.startswith("--log="):
                log_path = a.split("=", 1)[1]
        run_bus_census_analysis(log_path=log_path)

    elif cmd == "loop-latency":
        # 2026-04-14: dynamic async-block regression detector. Fires N
        # concurrent requests against a live Titan endpoint and reports the
        # latency distribution. Complements the static `async-blocks`
        # scanner — catches runtime-only issues (bus cascades, lock contention)
        # the AST can't see. Exit codes: 0=pass, 1=threshold exceeded, 2=error.
        target = "T1"
        endpoint = "/v4/inner-trinity"
        concurrency = 20
        total = 100
        p95 = 300.0
        p99 = 800.0
        for a in sys.argv[2:]:
            if a.startswith("--target="):
                target = a.split("=", 1)[1]
            elif a.startswith("--endpoint="):
                endpoint = a.split("=", 1)[1]
            elif a.startswith("--concurrency="):
                try: concurrency = int(a.split("=", 1)[1])
                except ValueError: pass
            elif a.startswith("--total="):
                try: total = int(a.split("=", 1)[1])
                except ValueError: pass
            elif a.startswith("--p95-max-ms="):
                try: p95 = float(a.split("=", 1)[1])
                except ValueError: pass
            elif a.startswith("--p99-max-ms="):
                try: p99 = float(a.split("=", 1)[1])
                except ValueError: pass
        rc = run_loop_latency_check(target=target, endpoint=endpoint,
                                     concurrency=concurrency, total=total,
                                     p95_max_ms=p95, p99_max_ms=p99)
        sys.exit(rc)

    elif cmd == "ns-signals":
        # 2026-04-16 rFP β Stage 0: per-program urgency distribution.
        # Answers "which programs are alive?" from buffer files on disk.
        titan = "T1"
        for a in sys.argv[2:]:
            if a.startswith("--titan="):
                titan = a.split("=", 1)[1].upper()
        sys.exit(run_ns_signals(titan))

    elif cmd == "ns-health":
        # 2026-04-16 rFP β Stage 0: full NS health — signals + persistence + state.
        titan = "T1"
        for a in sys.argv[2:]:
            if a.startswith("--titan="):
                titan = a.split("=", 1)[1].upper()
        sys.exit(run_ns_health(titan))

    elif cmd == "ns-learning":
        # 2026-04-16 rFP β Stage 0: is the NN actually learning? Loss + weight age.
        titan = "T1"
        since = 24
        for a in sys.argv[2:]:
            if a.startswith("--titan="):
                titan = a.split("=", 1)[1].upper()
            elif a.startswith("--since-hours="):
                try:
                    since = int(a.split("=", 1)[1])
                except ValueError:
                    pass
        sys.exit(run_ns_learning(titan, since))

    elif cmd == "ns-persistence-check":
        # 2026-04-16 rFP β Stage 0: all 11 programs have weights + buffer + state?
        sys.exit(run_ns_persistence_check())

    elif cmd == "thread-pool":
        # 2026-04-14: thread-pool saturation snapshot (asyncio default executor)
        # 2026-04-28 (A.8.2): --parent flag switches to parent-process residency
        # inventory (microkernel rFP A.8 §6); --all spans all 3 Titans.
        if "--parent" in sys.argv:
            run_parent_thread_inventory(all_titans=("--all" in sys.argv))
        else:
            run_thread_pool_check()

    elif cmd == "stability":
        # 2026-04-14: stability check — restart density per Titan over time window.
        # Reads watchdog logs (T1 local, T2/T3 over SSH), counts force-restart and
        # NOT-RUNNING events, computes per-hour density, prints trend. Designed as
        # mandatory step 0 of session_startup_protocol — catches multi-day crash
        # loops that single-snapshot /health checks miss. See
        # feedback_audits_must_read_runtime_first.md for codified rationale.
        hours = 24
        for a in sys.argv[2:]:
            if a.startswith("--hours="):
                try:
                    hours = int(a.split("=", 1)[1])
                except ValueError:
                    pass
        run_stability_check(hours=hours)

    elif cmd == "profile":
        # 2026-04-17: live memory + CPU profiling for resource optimization.
        # Queries /v4/admin/memory-profile on target Titan(s).
        run_memory_profile(
            all_titans="--all" in sys.argv,
            diff="--diff" in sys.argv,
            cpu="--cpu" in sys.argv,
            json_out="--json" in sys.argv,
        )

    elif cmd == "observables":
        # 2026-04-22: enforces `feedback_observable_persistence_audit.md`.
        # Every OBS gate must populate evidence_source + retention_horizon +
        # persistence_verification at open-time so close-time evidence is not
        # lost to brain-log rotation. --audit flags gates missing these.
        run_observables_audit(
            audit="--audit" in sys.argv,
            verbose="--verbose" in sys.argv or "-v" in sys.argv,
        )

    elif cmd == "layers":
        # Microkernel v2 Phase A §A.5 (2026-04-24): per-layer module summary.
        # Queries /v4/layers on T1/T2/T3, shows total/running/crashed per
        # L0/L1/L2/L3 + module lists. Used to verify layer assignment is
        # consistent across all Titans after deploy.
        run_layers(all_titans="--all" in sys.argv)

    elif cmd == "shm-status":
        # Microkernel v2 Phase A §A.2 (2026-04-24): per-titan shm registry
        # status. Lists /dev/shm/titan_{id}/*.bin files with seq + age + size,
        # and optionally runs a verify-read roundtrip (--verify) with CRC
        # check. Used to confirm shm writers are firing at the expected
        # Schumann-derived cadence after deploy + flag-flip.
        run_shm_status(
            all_titans="--all" in sys.argv,
            verify="--verify" in sys.argv,
        )

    elif cmd == "kernel-status":
        # Microkernel v2 Phase A §A.1 (2026-04-24 S3): per-titan kernel
        # health summary. Shows kernel_plugin_split_enabled flag state,
        # supervised module count, Trinity/Neuromod/Epoch/Spirit-fast
        # shm writer status, boot banner mode. Used as a single command
        # to verify kernel+plugin split cutover success during deploy.
        run_kernel_status(all_titans="--all" in sys.argv)

    elif cmd == "titanvm-schema":
        # Microkernel v2 Phase A §A.2 §1.4a (2026-04-25 S4): verify
        # NeuralReflexNet per-program telemetry shape before declaring
        # the TITANVM_REGISTERS RegistrySpec. Lists each NS program's
        # available fields (urgency from _all_urgencies, fire_count,
        # total_updates, last_loss) so we can pin the (11, 4) layout.
        run_titanvm_schema()

    elif cmd in ("silent-swallows", "warnings", "symmetries"):
        # 2026-04-25: code-integrity scanners — implemented in sibling
        # arch_map_dead_wiring.py (v2 capability). Import via this script's
        # directory so we always pick up the worktree-local copy.
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from arch_map_dead_wiring import (
            run_silent_swallows_scan,
            run_warnings,
            run_symmetries_audit,
        )
        if cmd == "silent-swallows":
            sys.exit(run_silent_swallows_scan(runtime="--runtime" in sys.argv))
        elif cmd == "warnings":
            sys.exit(run_warnings(all_titans="--all" in sys.argv))
        else:
            sys.exit(run_symmetries_audit(all_titans="--all" in sys.argv))

    elif cmd == "api-status":
        # Microkernel v2 Phase A §A.4 (2026-04-25 S5): per-titan API
        # subprocess health summary. Reports api_process_separation_enabled
        # flag state, api_subprocess Guardian state (running/stopped/crashed),
        # kernel_rpc socket existence + permissions, uvicorn port reachable,
        # and OBSERVATORY_EVENT bridge activity. Mirrors arch_map kernel-status
        # / shm-status subcommand shape for consistency.
        run_api_status(all_titans="--all" in sys.argv)

    elif cmd == "module-methods":
        # Microkernel v2 Phase A §A.3 (2026-04-25 S6): per-module spawn vs
        # fork audit. Reports each Guardian module's start_method per Titan,
        # along with worker RSS (so we can compare fork vs spawn savings
        # post-flip). Used to confirm spawn migration rollout state matches
        # expectations before flag flips.
        run_module_methods(all_titans="--all" in sys.argv)

    elif cmd == "s5-callsites":
        # Microkernel v2 Phase A §A.4 (S5 amendment 2026-04-26 D4): drift
        # detection for endpoint-side `plugin.X` callsites that bypass the
        # TitanStateAccessor. Wraps scripts/s5_callsite_audit.py + classifies
        # each remaining site as A (state read), B (async cross-proc), C
        # (manual handle). Drift = new endpoint code that bypasses the
        # accessor pattern; this audit must stay near zero to keep Phase
        # B/C migrations clean.
        run_s5_callsites(verbose=("--verbose" in sys.argv or "-v" in sys.argv))

    elif cmd in ("cache-keys", "cache_keys"):
        # rFP_observatory_data_loading_v1 Phase 1 (2026-04-26): single source of
        # truth audit for the producer→bus event→cache key→endpoint→frontend
        # contract. Reads titan_plugin/api/cache_key_registry.REGISTRY and
        # validates each entry against the source tree (+ optionally live state).
        # Modes:
        #   --audit   full static audit, exit-code on errors  (default)
        #   --status  live-state matrix per Titan (--all for T1+T2+T3)
        #   --list    one-line per registry entry, grouped by kind
        #   --json    machine-readable output (works with --audit / --status / --list)
        modes = {"audit", "status", "list"}
        mode = next(
            (a.lstrip("-") for a in sys.argv[2:] if a.lstrip("-") in modes),
            "audit",
        )
        rc = run_cache_keys_command(
            mode=mode,
            all_titans="--all" in sys.argv,
            json_mode="--json" in sys.argv,
        )
        sys.exit(rc)

    elif cmd == "phase-c":
        # SPEC enforcer per titan-docs/SPEC_titan_architecture.md §20.
        # Reads SPEC_titan_architecture_constants.toml as single source of truth
        # (no hardcoded SPEC values in arch_map source per §2.6 live-versioning).
        # Sub-actions:
        #   verify [--strict] [--json]   — comprehensive static check vs SPEC
        #   regen                        — regenerate Python + Rust constants from TOML
        #   diff <ref1> <ref2>           — show SPEC delta between git refs
        #   parity                       — run all parity vector tests (frame, authkey, slot)
        #   topology                     — render §9 hierarchy + §10 ordering as DOT graph
        #   watch-escalations [--tail]   — runtime: live cascade events from data/supervision.jsonl
        #   supervision-log --filter ... — query interface over data/supervision.jsonl
        #   data-integrity-audit         — runtime: verify §11.H critical-data files
        rc = run_phase_c_command(sys.argv[2:])
        sys.exit(rc)

    else:
        print(__doc__)
        sys.exit(1)


# ── Phase C SPEC enforcer (titan-docs/SPEC_titan_architecture.md §20) ──

def run_phase_c_command(args: list) -> int:
    """Dispatch `arch_map phase-c <action>` per SPEC §20."""
    if not args:
        return _phase_c_help()
    action = args[0]
    rest = args[1:]
    if action == "verify":
        return _phase_c_verify(strict="--strict" in rest, json_mode="--json" in rest)
    elif action == "regen":
        return _phase_c_regen()
    elif action == "diff":
        if len(rest) < 2:
            print("usage: arch_map phase-c diff <ref1> <ref2>", file=sys.stderr)
            return 2
        return _phase_c_diff(rest[0], rest[1])
    elif action == "parity":
        return _phase_c_parity()
    elif action == "topology":
        return _phase_c_topology()
    elif action in ("watch-escalations", "watch_escalations"):
        return _phase_c_watch_escalations(tail="--tail" in rest)
    elif action in ("supervision-log", "supervision_log"):
        return _phase_c_supervision_log(rest)
    elif action in ("data-integrity-audit", "data_integrity_audit"):
        return _phase_c_data_integrity_audit()
    elif action in ("help", "--help", "-h"):
        return _phase_c_help()
    else:
        print(f"unknown phase-c action: {action}", file=sys.stderr)
        return _phase_c_help()


def _phase_c_help() -> int:
    print(
        """\
arch_map phase-c — SPEC enforcer per titan-docs/SPEC_titan_architecture.md §20

Actions:
  verify [--strict] [--json]   Static check codebase against SPEC. --strict exits
                                non-zero on any finding (pre-commit mode).
  regen                        Regenerate _phase_c_constants.py + constants.rs from TOML.
  diff <ref1> <ref2>           Show SPEC delta between git refs.
  parity                       Run all parity vector tests.
  topology                     Render §9 hierarchy + §10 ordering as DOT graph.
  watch-escalations [--tail]   Live cascade events from data/supervision.jsonl.
  supervision-log [filters]    Query interface over data/supervision.jsonl.
                                Filters: --child <name>, --reason <enum>,
                                         --supervisor <name>, --since <duration>,
                                         --event <type>, --tail
  data-integrity-audit         Verify §11.H critical-data files (backups + integrity).

Source of truth: titan-docs/SPEC_titan_architecture_constants.toml
"""
    )
    return 0


def _phase_c_repo_root() -> "Path":
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


def _phase_c_load_toml() -> dict:
    """Load the SPEC TOML or exit with a clear error."""
    import sys
    try:
        import tomllib  # py 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore
    path = _phase_c_repo_root() / "titan-docs" / "SPEC_titan_architecture_constants.toml"
    if not path.exists():
        print(f"ERROR: SPEC TOML not found at {path}", file=sys.stderr)
        sys.exit(2)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _phase_c_regen() -> int:
    """Run the generator script in write mode."""
    import subprocess
    repo = _phase_c_repo_root()
    script = repo / "scripts" / "generate_phase_c_constants.py"
    if not script.exists():
        print(f"ERROR: generator script not found at {script}", file=sys.stderr)
        return 2
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo),
    )
    return result.returncode


def _phase_c_verify(strict: bool = False, json_mode: bool = False) -> int:
    """
    Static check codebase against SPEC.

    v0 checks:
      1. Generator parity — generated files match TOML
      2. Domain registry — every used domain has a [domains.X] block
      3. Ground-truth presence — Preamble G1-G16 entries present in TOML
      4. SPEC document references valid sections in TOML

    Future (deferred to v1):
      - Static analysis of code for §11.H critical-data writes via atomic_write helper
      - Detection of magic numbers vs constants
      - §9 wiring matrix verification
      - §10 communication ordering checks
    """
    import json as _json
    import subprocess

    findings: list[dict] = []
    repo = _phase_c_repo_root()

    # Check 1: generator parity
    script = repo / "scripts" / "generate_phase_c_constants.py"
    if script.exists():
        result = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            findings.append({
                "kind": "GENERATOR_PARITY",
                "severity": "CRITICAL",
                "message": "Generated _phase_c_constants.py / constants.rs out of sync with TOML",
                "remediation": "run `python scripts/arch_map.py phase-c regen`",
                "detail": result.stdout + result.stderr,
            })
    else:
        findings.append({
            "kind": "TOOLING_MISSING",
            "severity": "CRITICAL",
            "message": "scripts/generate_phase_c_constants.py not found",
            "remediation": "restore from git history",
        })

    # Check 2: domain registry consistency
    try:
        toml = _phase_c_load_toml()
    except SystemExit:
        toml = {}
    domains = toml.get("domains", {}) if toml else {}
    constants = toml.get("constants", {}) if toml else {}
    used_domains: set = {c.get("domain", "") for c in constants.values()}
    used_domains.discard("")
    missing_in_registry = used_domains - set(domains.keys())
    for d in sorted(missing_in_registry):
        findings.append({
            "kind": "DOMAIN_UNREGISTERED",
            "severity": "CRITICAL",
            "message": f"Domain '{d}' used by constants but missing from [domains.{d}] block",
            "remediation": "add [domains.X] block with description + introduced_in",
        })

    # Check 3: ground-truth coverage
    truths = toml.get("ground_truths", {}) if toml else {}
    expected_truths = {f"G{i}" for i in range(1, 17)}
    missing_truths = expected_truths - set(truths.keys())
    for t in sorted(missing_truths):
        findings.append({
            "kind": "GROUND_TRUTH_MISSING",
            "severity": "HIGH",
            "message": f"Preamble {t} expected in TOML [ground_truths.{t}] but not found",
            "remediation": f"add [ground_truths.{t}] block with title + spec_ref + enforces",
        })

    # Check 4: SPEC doc presence
    spec_doc = repo / "titan-docs" / "SPEC_titan_architecture.md"
    if not spec_doc.exists():
        findings.append({
            "kind": "SPEC_DOC_MISSING",
            "severity": "CRITICAL",
            "message": "SPEC document not found at titan-docs/SPEC_titan_architecture.md",
            "remediation": "restore from git history",
        })

    # Output
    if json_mode:
        out = {
            "spec_version": toml.get("spec_version", "unknown") if toml else "unknown",
            "findings": findings,
            "n_findings": len(findings),
            "n_critical": sum(1 for f in findings if f["severity"] == "CRITICAL"),
        }
        print(_json.dumps(out, indent=2))
    else:
        if not findings:
            print(
                f"✓ phase-c verify clean (SPEC v{toml.get('spec_version', 'unknown')}, "
                f"{len(constants)} constants across {len(domains)} domains, "
                f"{len(truths)} ground truths)"
            )
        else:
            print(f"✗ phase-c verify: {len(findings)} finding(s)")
            for f in findings:
                print(f"  [{f['severity']}] {f['kind']}: {f['message']}")
                if "remediation" in f:
                    print(f"    → {f['remediation']}")

    if strict and findings:
        return 1
    return 0


def _phase_c_diff(ref1: str, ref2: str) -> int:
    """Show SPEC delta between two git refs (TOML diff with constant-level summary)."""
    import subprocess
    toml_path = "titan-docs/SPEC_titan_architecture_constants.toml"
    print(f"=== SPEC delta {ref1} → {ref2} ===\n")
    result = subprocess.run(
        ["git", "diff", "--no-color", f"{ref1}..{ref2}", "--", toml_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout:
        print(result.stdout)
    elif result.returncode == 0:
        print(f"(no SPEC TOML changes between {ref1} and {ref2})")
    else:
        print(f"git diff failed: {result.stderr}", file=sys.stderr)
        return result.returncode
    return 0


def _phase_c_parity() -> int:
    """Run parity vector tests: frame, authkey, bus_specs, slot layout."""
    import subprocess
    print("phase-c parity — running parity vector tests...")
    repo = _phase_c_repo_root()
    test_files = [
        "tests/test_frame_parity.py",
        "tests/test_bus_authkey.py",
        "tests/test_phase_c_constants_in_sync.py",
    ]
    existing = [t for t in test_files if (repo / t).exists()]
    if not existing:
        print("(no parity test files found yet)", file=sys.stderr)
        return 0
    cmd = [sys.executable, "-m", "pytest", "-p", "no:anchorpy", "--tb=short"] + existing
    result = subprocess.run(cmd, cwd=str(repo))
    return result.returncode


def _phase_c_topology() -> int:
    """Render §9 hierarchy + §10 ordering as DOT graph (placeholder v0)."""
    print(
        "phase-c topology — DOT graph rendering deferred to v1.\n"
        "v0 stub: structure laid out, implementation in next session.\n"
        "Will render: kernel → substrate → unified-spirit → 6 daemons + titan_HCL/guardian_HCL/modules"
    )
    return 0


def _phase_c_watch_escalations(tail: bool = False) -> int:
    """Stream supervision events from data/supervision.jsonl (placeholder v0)."""
    repo = _phase_c_repo_root()
    log_path = repo / "data" / "supervision.jsonl"
    if not log_path.exists():
        print(
            f"phase-c watch-escalations: {log_path} not yet created.\n"
            "Will be populated by kernel + guardian_HCL once §11.G.4 supervision log ships."
        )
        return 0
    print(f"phase-c watch-escalations — reading {log_path}\n(implementation deferred to v1)")
    return 0


def _phase_c_supervision_log(filters: list) -> int:
    """Query data/supervision.jsonl (placeholder v0)."""
    repo = _phase_c_repo_root()
    log_path = repo / "data" / "supervision.jsonl"
    if not log_path.exists():
        print(
            f"phase-c supervision-log: {log_path} not yet created.\n"
            "Will be populated by kernel + guardian_HCL once §11.G.4 supervision log ships.\n"
            f"Requested filters: {filters}"
        )
        return 0
    print(f"phase-c supervision-log filters={filters} (implementation deferred to v1)")
    return 0


def _phase_c_data_integrity_audit() -> int:
    """Audit §11.H.1 critical-data files for backups + integrity (placeholder v0)."""
    repo = _phase_c_repo_root()
    toml = _phase_c_load_toml()
    # v0: simplified — list critical-data files from SPEC §11.H.1 and check existence
    print("phase-c data-integrity-audit — v0 stub (SPEC §11.H.1)")
    print("  Implementation deferred to v1 — needs critical-data file list parsed from SPEC.")
    return 0


# ── Microkernel v2 — Shm registry status (§A.2) ─────────────────────

def run_shm_status(all_titans: bool = False, verify: bool = False) -> int:
    """
    Microkernel v2 Phase A §A.2 — per-titan shm registry status.

    Lists /dev/shm/titan_{titan_id}/*.bin registries with seq counter,
    age_seconds, and payload size. --verify runs a full read roundtrip
    via StateRegistryReader and reports whether the CRC + SeqLock pass.

    For local-host Titans (T1 on 127.0.0.1): reads shm directly from
    the local filesystem. For remote Titans (T2/T3 over VPC SSH),
    queries them via /v4/trinity-shm which exposes shm metadata. Missing
    registries are reported clearly.
    """
    import time as _time
    from pathlib import Path as _Path

    titans = [("T1", "/dev/shm/titan_T1")]
    if all_titans:
        titans += [("T2", "/dev/shm/titan_T2"), ("T3", "/dev/shm/titan_T3")]

    print()
    print("MICROKERNEL v2 — SHM REGISTRY STATUS (per Titan)")
    print("=" * 90)

    # Expected registries. S2 shipped Trinity/Neuromod/Epoch; S3b adds
    # INNER_SPIRIT_45D (spirit-fast at Schumann × 9 = 70.47 Hz). S4 adds
    # SphereClocks/Chi/TitanVMRegisters/Identity (fixed-size) + CGN
    # (variable-size, dual-mode). S7 adds INNER_BODY_5D (5D × 7.83 Hz)
    # and INNER_MIND_15D (15D × 23.49 Hz) — body+mind sensor decoupling
    # completes the Schumann-symmetric Trinity (5/15/45 dims at 1×/3×/9×
    # Schumann). Each registry's file is only present after its flag
    # has been flipped.
    try:
        from titan_plugin.core.state_registry import (
            CGN_LIVE_WEIGHTS, CHI_STATE, EPOCH_COUNTER, IDENTITY,
            INNER_BODY_5D, INNER_MIND_15D, INNER_SPIRIT_45D,
            NEUROMOD_STATE, SPHERE_CLOCKS_STATE, TITANVM_REGISTERS,
            TRINITY_STATE,
        )
        # Fixed-size specs first (S2 + S3b + S4 fixed + S7)
        expected_specs = [
            TRINITY_STATE, NEUROMOD_STATE, EPOCH_COUNTER, INNER_SPIRIT_45D,
            INNER_BODY_5D, INNER_MIND_15D,
            SPHERE_CLOCKS_STATE, CHI_STATE, TITANVM_REGISTERS, IDENTITY,
        ]
        # CGN handled separately (variable-size + dual-mode legacy fallback)
        cgn_spec = CGN_LIVE_WEIGHTS
    except Exception as e:
        print(f"  ✗ cannot import registry specs: {e}")
        return 1

    any_failed = False

    for tid, shm_path_str in titans:
        print(f"\n  {tid} ({shm_path_str})")
        print("  " + "-" * 86)
        shm_dir = _Path(shm_path_str)
        if not shm_dir.exists():
            print(f"    ⚠ directory missing (Titan may not have flipped shm flags yet)")
            # Not a hard fail — flags default off.
            continue

        for spec in expected_specs:
            path = shm_dir / f"{spec.name}.bin"
            if not path.exists():
                print(f"    — {spec.name:<20}  <missing>  (flag likely off)")
                continue
            size = path.stat().st_size
            mtime = path.stat().st_mtime
            age = _time.time() - mtime
            # Read header for seq (use reader to honor the protocol).
            seq_str = "?"
            try:
                if tid == "T1":  # local access
                    from titan_plugin.core.state_registry import StateRegistryReader
                    r = StateRegistryReader(spec, shm_dir)
                    meta = r.read_meta()
                    r.close()
                    if meta:
                        seq_str = str(meta["seq"])
                        age = meta["age_seconds"]
            except Exception as e:
                seq_str = f"err:{e}"
            icon = "✓" if size == spec.total_bytes else "✗"
            print(f"    {icon} {spec.name:<20}  seq={seq_str:<8}  "
                  f"age={age:>6.2f}s  size={size}B (expected={spec.total_bytes}B)")
            if size != spec.total_bytes:
                any_failed = True
            if verify and tid == "T1":
                try:
                    from titan_plugin.core.state_registry import StateRegistryReader
                    r = StateRegistryReader(spec, shm_dir)
                    arr = r.read()
                    r.close()
                    if arr is None:
                        print(f"        ✗ verify read: returned None (fallback triggered)")
                        any_failed = True
                    else:
                        print(f"        ✓ verify read: shape={arr.shape} dtype={arr.dtype}")
                except Exception as e:
                    print(f"        ✗ verify read raised: {e}")
                    any_failed = True

        # CGN — dual-mode (S4): per-titan StateRegistry path (flag on)
        # OR legacy global path (flag off). Surface both states.
        cgn_per_titan = shm_dir / "cgn_live_weights.bin"
        cgn_global = _Path("/dev/shm/cgn_live_weights.bin")
        if cgn_per_titan.exists():
            size = cgn_per_titan.stat().st_size
            mtime = cgn_per_titan.stat().st_mtime
            age = _time.time() - mtime
            seq_str = "?"
            try:
                if tid == "T1":
                    from titan_plugin.core.state_registry import StateRegistryReader
                    r = StateRegistryReader(cgn_spec, shm_dir)
                    meta = r.read_meta()
                    r.close()
                    if meta:
                        seq_str = str(meta["seq"])
                        age = meta["age_seconds"]
            except Exception as e:
                seq_str = f"err:{e}"
            icon = "✓" if size == cgn_spec.total_bytes else "✗"
            print(f"    {icon} {cgn_spec.name:<20}  seq={seq_str:<8}  "
                  f"age={age:>6.2f}s  size={size}B (mode=stateregistry/24B-header)")
            if size != cgn_spec.total_bytes:
                any_failed = True
        elif cgn_global.exists() and tid == "T1":
            # Legacy mode (flag off) — global file, only meaningful for T1
            # (T2+T3 share /dev/shm so global file is racy by design).
            size = cgn_global.stat().st_size
            mtime = cgn_global.stat().st_mtime
            age = _time.time() - mtime
            print(f"    — {cgn_spec.name:<20}  age={age:>6.2f}s  size={size}B "
                  f"(mode=legacy/16B-header global path)")
        else:
            print(f"    — {cgn_spec.name:<20}  <missing>  (flag likely off, no legacy file either)")

    print()
    return 0 if not any_failed else 1


# ── Microkernel v2 — Kernel status (§A.1) ────────────────────────────

def run_kernel_status(all_titans: bool = False) -> int:
    """
    Microkernel v2 Phase A §A.1 (S3) — per-titan kernel health summary.

    Combines per-Titan:
      - Microkernel flag state (kernel_plugin_split_enabled,
        shm_*_enabled flags) — queried via /health
      - Guardian supervised module count (via /v4/layers)
      - Shm writer health (via /v4/trinity-shm + file inspection for T1)
      - Boot banner mode (from /health 'boot_mode' when available)

    Intended as the single command an operator runs during S3 cutover
    to confirm: "Is this Titan running the split path? Are all its L0
    services healthy? Is spirit-fast writing?"
    """
    import time as _time
    from pathlib import Path as _Path

    titans = [("T1", "http://127.0.0.1:7777", "/dev/shm/titan_T1")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777", "/dev/shm/titan_T2"),
            ("T3", "http://10.135.0.6:7778", "/dev/shm/titan_T3"),
        ]

    print()
    print("MICROKERNEL v2 — KERNEL STATUS (per Titan)")
    print("=" * 90)

    try:
        from titan_plugin.core.state_registry import (
            EPOCH_COUNTER, INNER_SPIRIT_45D, NEUROMOD_STATE, TRINITY_STATE,
        )
        specs = [TRINITY_STATE, NEUROMOD_STATE, EPOCH_COUNTER, INNER_SPIRIT_45D]
    except Exception as e:
        print(f"  ✗ cannot import registry specs: {e}")
        return 1

    any_failed = False

    for tid, base_url, shm_path_str in titans:
        print(f"\n  {tid} ({base_url})")
        print("  " + "-" * 86)

        # ── Health endpoint: is the Titan alive + which boot path? ────
        health = _services_get(base_url, "/health", timeout=5.0)
        if health is None:
            print(f"    ✗ /health unreachable")
            any_failed = True
            continue
        # /health doesn't currently expose kernel_plugin_split_enabled
        # directly — we infer from the process-title or boot log. For
        # now, show a basic alive indicator.
        print(f"    ✓ API reachable")

        # ── Guardian modules supervised (from /v4/layers) ─────────────
        layers = _services_get(base_url, "/v4/layers", timeout=5.0)
        if layers and isinstance(layers, dict):
            data = layers.get("data", layers)
            counts = data.get("layer_stats") if isinstance(data, dict) else None
            if isinstance(counts, dict):
                total = sum(c.get("total", 0) for c in counts.values())
                running = sum(c.get("running", 0) for c in counts.values())
                print(f"    ✓ Guardian: {running}/{total} modules running "
                      f"(L0={counts.get('L0', {}).get('total', 0)}/"
                      f"L1={counts.get('L1', {}).get('total', 0)}/"
                      f"L2={counts.get('L2', {}).get('total', 0)}/"
                      f"L3={counts.get('L3', {}).get('total', 0)})")
            else:
                print(f"    ⚠ /v4/layers shape unexpected (no layer_stats)")
        else:
            print(f"    ⚠ /v4/layers unreachable (pre-S1 deploy?)")

        # ── Shm writers (file-level check for T1 local; remote = note) ──
        shm_dir = _Path(shm_path_str) if tid == "T1" else None
        for spec in specs:
            if shm_dir is None:
                # Remote Titan — skip file inspection, operator runs
                # kernel-status on the Titan itself for detail.
                continue
            path = shm_dir / f"{spec.name}.bin"
            if not path.exists():
                flag = spec.feature_flag or "?"
                print(f"    — {spec.name:<20} <missing>  (flag={flag})")
                continue
            age = _time.time() - path.stat().st_mtime
            try:
                from titan_plugin.core.state_registry import StateRegistryReader
                r = StateRegistryReader(spec, shm_dir)
                meta = r.read_meta()
                r.close()
                seq = meta.get("seq", "?") if meta else "?"
                age = meta.get("age_seconds", age) if meta else age
                icon = "✓"
                if isinstance(age, (int, float)) and age > 60.0:
                    icon = "⚠"
                    any_failed = True
                print(f"    {icon} {spec.name:<20} seq={str(seq):<8} age={age:>6.2f}s")
            except Exception as e:
                print(f"    ✗ {spec.name:<20} read error: {e}")
                any_failed = True

        if tid != "T1":
            print(f"    ℹ shm detail: run `arch_map kernel-status` locally on {tid}")

    print()
    return 0 if not any_failed else 1


# ── Microkernel v2 — Layer summary (§A.5) ────────────────────────────

def run_layers(all_titans: bool = False) -> int:
    """
    Microkernel v2 Phase A §A.5 — per-layer Guardian module summary.

    Queries /v4/layers on T1 (and T2/T3 with --all). Shows total + running
    + crashed per L0/L1/L2/L3 plus the module list at each layer, so an
    operator can verify layer assignment is consistent across Titans.
    """
    titans = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]

    print()
    print("MICROKERNEL v2 — LAYER ASSIGNMENT (per Titan)")
    print("=" * 90)

    any_failed = False

    for tid, base_url in titans:
        print(f"\n  {tid} ({base_url})")
        print("  " + "-" * 86)
        resp = _services_get(base_url, "/v4/layers", timeout=8.0)
        if not resp:
            print("    ✗ /v4/layers unreachable (endpoint may not be deployed yet)")
            any_failed = True
            continue
        payload = resp.get("data", resp) if isinstance(resp, dict) else resp
        if not isinstance(payload, dict):
            print(f"    ✗ unexpected response: {payload}")
            any_failed = True
            continue

        stats = payload.get("layer_stats", {})
        modules_by_layer = payload.get("modules_by_layer", {})

        for layer in ("L0", "L1", "L2", "L3"):
            bucket = stats.get(layer, {})
            modules = modules_by_layer.get(layer, [])
            total = bucket.get("total", 0)
            running = bucket.get("running", 0)
            crashed = bucket.get("crashed", 0)
            disabled = bucket.get("disabled", 0)
            status_icon = "✓" if total == running and crashed == 0 else (
                "⚠" if crashed == 0 else "✗")
            # Layer descriptor
            desc = {
                "L0": "microkernel (in-process — no Guardian modules)",
                "L1": "Trinity daemons + L1 persistence",
                "L2": "higher cognition + cognitive substrates",
                "L3": "pluggable modules + L3 persistence",
            }.get(layer, "")
            print(f"    {status_icon} {layer}  total={total:>2}  running={running:>2}  "
                  f"crashed={crashed}  disabled={disabled}  — {desc}")
            if modules:
                # Wrap the module list nicely.
                mod_str = ", ".join(modules)
                # Indent subsequent lines
                print(f"        [{mod_str}]")

        # Canon check — compare live layers against _layer_canon.LAYER_CANON
        try:
            from titan_plugin._layer_canon import LAYER_CANON
            mismatches = []
            for layer, names in modules_by_layer.items():
                for n in names:
                    expected = LAYER_CANON.get(n)
                    if expected and expected != layer:
                        mismatches.append((n, layer, expected))
            if mismatches:
                print("    ✗ CANON MISMATCH:")
                for n, got, expected in mismatches:
                    print(f"        module '{n}': live={got}  canon={expected}")
                any_failed = True
            else:
                print("    ✓ All module layers match LAYER_CANON")
        except Exception as e:
            print(f"    ⚠ could not verify against canon: {e}")

    print()
    return 0 if not any_failed else 1


# ── Observables Audit ────────────────────────────────────────────────

def run_observables_audit(audit: bool = False, verbose: bool = False) -> int:
    """Scan titan-docs/OBSERVABLES.md for persistence-audit compliance.

    Enforces `memory/feedback_observable_persistence_audit.md`:
    every active OBS gate must specify `evidence_source`, `retention_horizon`,
    and `persistence_verification` fields in its body.

    Default: summary + compliance rate. --audit exits 1 if non-compliant
    (for CI / pre-commit integration). --verbose prints every gate's state.
    """
    import re as _re
    from pathlib import Path as _Path

    obs_path = _Path(__file__).parent.parent / "titan-docs" / "OBSERVABLES.md"
    if not obs_path.exists():
        print(f"  ✗ OBSERVABLES.md not found at {obs_path}")
        return 2

    text = obs_path.read_text()

    # Split the document into sections starting at each "### OBS-" header.
    # Each gate is one section until the next ### header or ---.
    gate_re = _re.compile(
        r"^### (OBS-[a-zA-Z0-9_\-]+) — (.+?)$",
        _re.MULTILINE,
    )
    sections: list[tuple[str, str, int]] = []  # (slug, title, start_line)
    for m in gate_re.finditer(text):
        slug = m.group(1)
        title = m.group(2).strip()
        line_no = text[: m.start()].count("\n") + 1
        sections.append((slug, title, m.start()))

    # Add a terminal anchor so the last section has a bound
    sections_with_bounds = []
    for i, (slug, title, start) in enumerate(sections):
        end = sections[i + 1][2] if i + 1 < len(sections) else len(text)
        sections_with_bounds.append((slug, title, start, end))

    required_fields = [
        ("evidence_source", _re.compile(
            r"\*\*Evidence source:?\*\*|evidence_source:", _re.IGNORECASE
        )),
        ("retention_horizon", _re.compile(
            r"\*\*Retention horizon:?\*\*|retention_horizon:", _re.IGNORECASE
        )),
        ("persistence_verification", _re.compile(
            r"\*\*Persistence verification:?\*\*|persistence_verification:|"
            r"persisted_by=|new_persistence=|acceptable_loss=",
            _re.IGNORECASE,
        )),
    ]

    # Skip gates that are already closed (PASSED / SUPERSEDED / graveyard);
    # the rule applies to OPEN gates (where evidence still needs collecting).
    closed_markers = _re.compile(
        r"(✅ PASSED|⚠ SUPERSEDED|\*\*SUPERSEDED\*\*|Status:\*\*\s*✅|Status:\*\*\s*⚠)",
        _re.IGNORECASE,
    )

    total_active = 0
    compliant = 0
    violations: list[tuple[str, str, list[str]]] = []  # (slug, title, missing)

    for slug, title, start, end in sections_with_bounds:
        body = text[start:end]
        # Status line is within the first 500 chars of the body
        head = body[:1500]
        if closed_markers.search(head):
            continue  # closed — skip
        total_active += 1
        missing = []
        for name, pat in required_fields:
            if not pat.search(body):
                missing.append(name)
        if not missing:
            compliant += 1
        else:
            violations.append((slug, title, missing))

    print("=" * 78)
    print("OBSERVABLES PERSISTENCE AUDIT")
    print("=" * 78)
    print(f"  Source: {obs_path}")
    print(f"  Active gates: {total_active}")
    print(f"  Compliant: {compliant}/{total_active}  "
          f"({100.0 * compliant / max(total_active, 1):.1f}%)")
    print()

    if not violations:
        print("  ✅ All active gates specify evidence_source + retention_horizon + "
              "persistence_verification.")
        print()
        return 0

    print(f"  ⚠ {len(violations)} active gates missing audit fields:")
    print()
    print(f"  {'GATE':55s} MISSING FIELDS")
    print("  " + "-" * 74)
    for slug, title, missing in violations:
        short = slug[:52] + "…" if len(slug) > 53 else slug
        print(f"  {short:55s} {', '.join(missing)}")
        if verbose:
            print(f"      title: {title[:70]}")
    print()
    print("  Rule: memory/feedback_observable_persistence_audit.md")
    print("  Fix:  add the three fields inline in each gate body. See rule "
          "for syntax.")
    print()

    return 1 if audit else 0


# ── Memory Profiling ─────────────────────────────────────────────────

def run_memory_profile(all_titans: bool = False, diff: bool = False,
                       cpu: bool = False, json_out: bool = False):
    """Memory + CPU profiling for Titan(s).

    Usage:
      arch_map profile              # T1 summary
      arch_map profile --all        # All 3 Titans
      arch_map profile --diff       # Growth since boot
      arch_map profile --cpu        # Include CPU% sampling (~1s extra)
      arch_map profile --json       # JSON output
    """
    import requests
    import json as _json

    targets = [
        ("T1", "http://127.0.0.1:7777"),
    ]
    if all_titans:
        targets += [
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]

    params = {"diff": str(diff).lower(), "cpu": str(cpu).lower()}

    for tid, base_url in targets:
        try:
            r = requests.get(f"{base_url}/v4/admin/memory-profile",
                             params=params, timeout=30)
            if r.status_code != 200:
                print(f"\n{tid}: HTTP {r.status_code}")
                continue
            resp = r.json()
            data = resp.get("data", resp)
        except Exception as e:
            print(f"\n{tid}: ERROR — {e}")
            continue

        if json_out:
            print(_json.dumps({tid: data}, indent=2, default=str))
            continue

        # ── Formatted table output ──
        sys_info = data.get("system", {})
        parent = data.get("parent", {})
        modules = data.get("modules", {})
        totals = data.get("totals", {})

        print()
        print(f"TITAN MEMORY PROFILE — {tid} ({base_url})")
        print("=" * 78)
        print(f"  System: {sys_info.get('used_pct', '?')}% used"
              f" | {sys_info.get('available_mb', '?')} MB available"
              f" | Swap: {sys_info.get('swap_total_mb', 0) - sys_info.get('swap_free_mb', 0):.0f} MB used")
        print(f"  Parent: PID {parent.get('pid', '?')}"
              f" | RSS {parent.get('rss_mb', '?')} MB"
              f" | Peak {parent.get('vm_peak_mb', '?')} MB"
              f" | Threads {parent.get('threads', '?')}")
        if cpu and "cpu_pct" in parent:
            print(f"  Parent CPU: {parent['cpu_pct']}%")
        print()

        # Module table
        header = f"  {'MODULE':<14} {'PID':>7} {'STATE':<10} {'RSS_MB':>8} {'PEAK_MB':>8} {'THREADS':>7}"
        if cpu:
            header += f" {'CPU%':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for name in sorted(modules.keys()):
            m = modules[name]
            pid = m.get("pid", "-")
            state = m.get("state", "?")[:9]
            rss = m.get("rss_mb", "-")
            peak = m.get("vm_peak_mb", "-")
            threads = m.get("threads", "-")
            line = f"  {name:<14} {str(pid):>7} {state:<10} {_pfmt(rss):>8} {_pfmt(peak):>8} {str(threads):>7}"
            if cpu:
                cp = m.get("cpu_pct", "-")
                line += f" {_pfmt(cp):>6}"
            print(line)

        print("  " + "-" * (len(header) - 2))
        print(f"  {'TOTAL':<14} {'':>7} {'':>10} {_pfmt(totals.get('total_rss_mb', '?')):>8}"
              f" {'':>8} {'':>7}")

        # tracemalloc top allocations
        tm = parent.get("tracemalloc", {})
        if tm.get("active") and tm.get("top"):
            print()
            label = "TOP ALLOCATIONS (growth since boot)" if diff else "TOP ALLOCATIONS (cumulative)"
            print(f"  {label}:")
            size_key = "size_diff_mb" if diff else "size_mb"
            for item in tm["top"][:15]:
                sz = item.get(size_key, item.get("size_mb", 0))
                sign = "+" if diff and sz > 0 else ""
                fname = item.get("file", "?")
                # Shorten paths for readability
                if "titan_plugin/" in fname:
                    fname = fname[fname.index("titan_plugin/"):]
                elif "site-packages/" in fname:
                    fname = "..." + fname[fname.index("site-packages/") + 13:]
                count = item.get("count", 0)
                print(f"    {sign}{sz:>8.1f} MB  {fname:<55} ({count:,} allocs)")

        print()


def _pfmt(v) -> str:
    """Format a numeric value or pass-through '-'."""
    if isinstance(v, (int, float)):
        return f"{v:.1f}"
    return str(v)


# ── Social Divergence ─────────────────────────────────────────────────

def run_knowledge_comparison():
    """Compare knowledge acquisition metrics across T1/T2/T3."""
    import requests

    print()
    print("KNOWLEDGE ACQUISITION COMPARISON")
    print("=" * 70)

    endpoints = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ]

    for tid, url in endpoints:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 66)
        try:
            resp = requests.get(f"{url}/v4/knowledge-stats", timeout=8)
            if resp.status_code != 200:
                print(f"    [ERROR] HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", {})
            print(f"    Total concepts:     {d.get('total_concepts', 0)}")
            print(f"    Avg confidence:     {d.get('avg_confidence', 0):.3f}")
            print(f"    Avg quality:        {d.get('avg_quality_score', 0):.3f}")
            print(f"    Total usage:        {d.get('total_usage', 0)}")
            print(f"    24h acquisitions:   {d.get('acquisition_rate_24h', 0)}")

            src = d.get("source_distribution", {})
            if src:
                print(f"    Sources:            {', '.join(f'{k}={v}' for k, v in src.items())}")

            top_u = d.get("top_by_usage", [])
            if top_u:
                print(f"    Top by usage:")
                for t in top_u[:3]:
                    print(f"      - {t['topic']} (used {t['times_used']}x, conf {t['confidence']:.2f})")

            top_c = d.get("top_by_confidence", [])
            if top_c:
                print(f"    Top by confidence:")
                for t in top_c[:3]:
                    print(f"      - {t['topic']} (conf {t['confidence']:.2f})")

        except Exception as e:
            print(f"    [ERROR] {e}")

    print()
    print("=" * 70)
    print()


def run_social_divergence():
    """Compare social policy Q-values across T1/T2/T3.

    Shows how each Titan's learned social policy differs — a measure of
    emergent social personality divergence (Sapir-Whorf for social cognition).
    """
    import requests

    print()
    print("SOCIAL POLICY DIVERGENCE — CGN Social Consumer")
    print("=" * 70)

    endpoints = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ]

    # Test with a standard user profile
    test_profiles = [
        {"label": "New user", "familiarity": 0.1, "interaction_count": 1,
         "social_valence": 0.0, "mention_count": 0},
        {"label": "Known friend", "familiarity": 0.7, "interaction_count": 15,
         "social_valence": 0.4, "mention_count": 5},
        {"label": "Frequent visitor", "familiarity": 0.4, "interaction_count": 5,
         "social_valence": -0.1, "mention_count": 2},
    ]

    actions = ["engage_warmly", "engage_cautiously", "respond_briefly",
               "disengage", "deepen_bond", "protect"]

    for profile in test_profiles:
        label = profile.pop("label")
        print(f"\n  Profile: {label}")
        print(f"  " + "-" * 66)

        header = f"  {'Action':<20s}"
        for tid, _ in endpoints:
            header += f"  {tid:>8s}"
        print(header)

        data = {}
        for tid, url in endpoints:
            try:
                resp = requests.get(f"{url}/v4/cgn-social-action",
                                    params=profile, timeout=8)
                if resp.status_code == 200:
                    d = resp.json().get("data", {})
                    data[tid] = d.get("q_values", {})
                else:
                    data[tid] = {}
            except Exception:
                data[tid] = {}

        for action in actions:
            row = f"  {action:<20s}"
            for tid, _ in endpoints:
                val = data.get(tid, {}).get(action, 0)
                row += f"  {val:8.4f}"
            print(row)

        # L2 distance from T1
        t1_vals = [data.get("T1", {}).get(a, 0) for a in actions]
        for tid in ["T2", "T3"]:
            t_vals = [data.get(tid, {}).get(a, 0) for a in actions]
            l2 = sum((a - b) ** 2 for a, b in zip(t1_vals, t_vals)) ** 0.5
            print(f"  L2 from T1 → {tid}: {l2:.4f}")

    print()
    print("=" * 70)
    print()

    # HAOV Epistemic Divergence
    print("HAOV EPISTEMIC DIVERGENCE — Hypothesis Testing Styles")
    print("=" * 70)

    haov_data = {}
    for tid, url in endpoints:
        try:
            resp = requests.get(f"{url}/v4/cgn-haov-stats", timeout=8)
            if resp.status_code == 200:
                haov_data[tid] = resp.json().get("data", {}).get("consumers", {})
            else:
                haov_data[tid] = {}
        except Exception:
            haov_data[tid] = {}

    all_consumers = set()
    for tid_data in haov_data.values():
        all_consumers.update(tid_data.keys())

    if not all_consumers:
        print("  No HAOV data available yet.")
    else:
        for consumer in sorted(all_consumers):
            print(f"\n  Consumer: {consumer}")
            print(f"  " + "-" * 66)
            header = f"  {'Metric':<25s}"
            for tid, _ in endpoints:
                header += f"  {tid:>10s}"
            print(header)

            metrics = ["formed", "tested", "confirmed", "confirmation_rate",
                       "verified_rules_count", "epistemic_style"]
            for metric in metrics:
                row = f"  {metric:<25s}"
                for tid, _ in endpoints:
                    val = haov_data.get(tid, {}).get(consumer, {}).get(metric, "-")
                    if isinstance(val, float):
                        row += f"  {val:10.3f}"
                    else:
                        row += f"  {str(val):>10s}"
                print(row)

    print()
    print("=" * 70)
    print()


# ── META-CGN CLI ──────────────────────────────────────────────────────

def run_meta_cgn(subcommand: str = "status") -> None:
    """arch_map meta-cgn {status|readiness|disagreements|failsafe|impasse|
                          audit|domains|history}.

    Diagnoses META-CGN state across the live API + local files. Subcommands:
      status         — full telemetry block (default)
      readiness      — graduation blockers (what's preventing active mode)
      disagreements  — recent α-vs-β rerank overrides
      failsafe       — watchdog state + failure history
      impasse        — F8 cognitive impasse state
      audit          — P12 consolidated snapshot (everything in one view)
      domains        — P12 per-domain primitive grounding table
      history        — P12 last 30 blend_weights_history.jsonl rows
    """
    import urllib.request
    import urllib.error

    def fetch_api(path: str) -> dict:
        try:
            url = f"http://127.0.0.1:7777{path}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": f"API fetch failed: {e}"}

    print(f"\nMETA-CGN — {subcommand.upper()}")
    print("=" * 78)

    if subcommand == "status":
        r = fetch_api("/v4/meta-cgn")
        data = r.get("data", r)
        print(f"  Status:         {data.get('status', '?')}")
        print(f"  Registered:     {data.get('registered', False)}")
        print(f"  Uptime:         {data.get('uptime_seconds', 0)}s")
        print(f"  Updates:        {data.get('updates_applied', 0)}")
        print(f"  Transitions:    {data.get('transitions_sent', 0)}")
        print(f"  Compositions:   {data.get('compositions_computed', 0)}")
        grad = data.get("graduation", {})
        print(f"\n  Graduation progress:  {grad.get('progress', 0)}/100")
        print(f"  Rolled-back count:    {grad.get('rolled_back_count', 0)}")
        fs = data.get("failsafe", {})
        print(f"\n  Failsafe trip count:  {fs.get('failsafe_trip_count', 0)}")
        print(f"  Cooldown remaining:   {fs.get('cooldown_remaining', 0)}")
        print(f"  Total failures:       {fs.get('total_failures', 0)}")
        imp = data.get("impasse", {})
        print(f"\n  Impasse state:        {imp.get('state', '?')}")
        print(f"  Impasse fires:        {imp.get('total_fires', 0)}")
        haov = data.get("haov", {})
        print(f"\n  HAOV:  by_status={haov.get('by_status', {})}")
        sq = data.get("shadow_quality", {})
        print(f"  Shadow quality:       rate={sq.get('disagreement_rate', 0):.3f} "
              f"health={sq.get('health', '?')}")
        # P6 telemetry: β-influence + monoculture + domain routing
        print(f"\n  [P6] β-dispersion EMA:  "
              f"{data.get('beta_score_dispersion_ema', 0):.4f}  "
              f"(rerank_samples={data.get('rerank_samples', 0)})")
        print(f"  [P6] Usage Gini:        "
              f"{data.get('usage_gini', 0):.4f}  "
              f"(0=uniform, 1=monoculture)")
        print(f"  [P6] Domain routing:    "
              f"hits={data.get('domain_hits', 0)}  "
              f"fallbacks={data.get('domain_fallbacks', 0)}")
        print(f"  [P6] Chains since decay: "
              f"{data.get('chains_since_decay', 0)}")
        # P7: EUREKA accelerator + advisor-conflict telemetry
        eureka_counts = data.get("eureka_trigger_counts", {})
        top_triggers = sorted(eureka_counts.items(),
                              key=lambda kv: kv[1], reverse=True)[:3]
        top_str = " ".join(f"{p}={n}" for p, n in top_triggers if n > 0) \
            or "(none yet)"
        print(f"\n  [P7] EUREKA accelerated: "
              f"{data.get('eureka_accelerated_updates', 0)} updates  "
              f"top_triggers: {top_str}")
        print(f"  [P7] Conflict bus:       "
              f"emitted={data.get('conflict_bus_events_emitted', 0)}  "
              f"throttled={data.get('conflict_sigs_throttled', 0)}  "
              f"chain_counter={data.get('chain_counter', 0)}")
        # P8: SOAR-via-CGN full protocol telemetry
        provided = data.get("knowledge_provided_by_source", {})
        helpful = data.get("knowledge_helpful_by_source", {})
        top_provided = sorted(provided.items(),
                              key=lambda kv: kv[1], reverse=True)[:3]
        prov_str = " ".join(f"{s}={n}" for s, n in top_provided if n > 0) \
            or "(none yet)"
        top_helpful = sorted(helpful.items(),
                             key=lambda kv: kv[1], reverse=True)[:3]
        help_str = " ".join(f"{s}={n}" for s, n in top_helpful if n > 0) \
            or "(none yet)"
        print(f"\n  [P8] Knowledge REQs:     "
              f"emitted={data.get('knowledge_requests_emitted', 0)}  "
              f"deduped={data.get('knowledge_requests_deduped', 0)}  "
              f"pending={data.get('knowledge_pending', 0)}")
        print(f"  [P8] Finalized windows:  "
              f"finalized={data.get('knowledge_requests_finalized', 0)}  "
              f"empty={data.get('knowledge_requests_empty', 0)}")
        print(f"  [P8] Responses received: "
              f"{data.get('knowledge_responses_received', 0)}  "
              f"sent(as-responder)={data.get('knowledge_responses_sent', 0)}")
        print(f"  [P8] Providers (top):    {prov_str}")
        print(f"  [P8] Helpful (top):      {help_str}")
        return

    if subcommand == "readiness":
        r = fetch_api("/v4/meta-cgn/graduation-readiness")
        data = r.get("data", r)
        print(f"  Status:              {data.get('status', '?')}")
        print(f"  Ready:               {data.get('ready_to_graduate', False)}")
        print(f"  Progress:            {data.get('graduation_progress', 0)}/100")
        print(f"  Primitives n≥50:     "
              f"{data.get('primitives_well_sampled', 0)}/5")
        print(f"  Confirmed HAOV:      "
              f"{data.get('confirmed_hypotheses', 0)}/3")
        print(f"  Total updates:       "
              f"{data.get('total_updates', 0)}/2000")
        print(f"  Rolled-back count:   {data.get('rolled_back_count', 0)}")
        return

    if subcommand == "disagreements":
        r = fetch_api("/v4/meta-cgn/disagreements?limit=30")
        data = r.get("data", r)
        events = data.get("disagreements", [])
        if not events:
            print("  No disagreements recorded yet.")
            return
        print(f"  Last {len(events)} disagreement events:\n")
        for e in events[-30:]:
            ts = e.get("ts", 0)
            when = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
            print(f"  [{when}] β={e.get('β_chose', '?')[:40]:<40} "
                  f"β_score={e.get('β_score', 0):.3f} "
                  f"α_top={e.get('chain_iql_top', '?')[:30]:<30} "
                  f"α_Q={e.get('chain_iql_top_Q', 0):.3f} "
                  f"ramp={e.get('ramp', 0):.2f}")
        return

    if subcommand == "failsafe":
        r = fetch_api("/v4/meta-cgn/failsafe-status")
        data = r.get("data", r)
        print(f"  Status:               {data.get('status', '?')}")
        print(f"  Total failures:       {data.get('total_failures', 0)}")
        print(f"  Trip count:           "
              f"{data.get('failsafe_trip_count', 0)}")
        print(f"  Cooldown remaining:   "
              f"{data.get('cooldown_remaining', 0)}")
        print(f"  Disabled reason:      {data.get('disabled_reason', '') or '-'}")
        print(f"  Window size:          {data.get('window_size', 0)}")
        print(f"  Unique signatures:    "
              f"{data.get('unique_signatures_in_window', 0)}")
        print(f"  Severity sum:         "
              f"{data.get('severity_sum_in_window', 0)}"
              f"/{data.get('severity_trip_threshold', 9)}")
        return

    if subcommand == "impasse":
        r = fetch_api("/v4/meta-cgn/impasse-status")
        data = r.get("data", r)
        print(f"  State:                {data.get('state', '?')}")
        print(f"  Total fires:          {data.get('total_fires', 0)}")
        print(f"  α-boost remaining:    "
              f"{data.get('alpha_boost_remaining', 0)}")
        print(f"  V history depth:      {data.get('v_history_depth', 0)}/500")
        print(f"  Grad blockers unchanged: "
              f"{data.get('graduation_blockers_unchanged_chains', 0)} chains")
        return

    if subcommand == "domains":
        r = fetch_api("/v4/meta-cgn/by-domain")
        data = r.get("data", r)
        per_prim = data.get("per_primitive", {})
        thresh = data.get("domain_threshold", 10)
        if not per_prim:
            print("  No domain data yet.")
            return
        print(f"  Domain activation threshold: {thresh} observations\n")
        for p_id, prim_data in per_prim.items():
            pooled = prim_data.get("pooled", {})
            domains = prim_data.get("domains", {})
            active_doms = [d for d, info in domains.items()
                           if info.get("active")]
            pooled_V = pooled.get("V", 0.5)
            pooled_n = pooled.get("n_samples", 0)
            print(f"  {p_id:<12s} pooled V={pooled_V:.3f} n={pooled_n}  "
                  f"| active domains: {len(active_doms)}/{len(domains)}")
            for d, info in sorted(domains.items(),
                                   key=lambda kv: kv[1]["V"], reverse=True):
                marker = "✓" if info["active"] else "·"
                print(f"    {marker} {d:<22s} V={info['V']:.3f} "
                      f"n={info['n_domain']:>4} CI±{info['ci_width']:.3f}")
        return

    if subcommand == "history":
        import os as _os
        path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "meta_cgn", "blend_weights_history.jsonl")
        if not _os.path.exists(path):
            print("  blend_weights_history.jsonl not yet written.")
            return
        with open(path) as f:
            lines = f.readlines()[-30:]
        if not lines:
            print("  No history rows yet.")
            return
        print(f"  Last {len(lines)} chain blend events:\n")
        print(f"  {'chain_id':>8} {'domain':<14} {'stage':<12} "
              f"{'w_leg':>6} {'w_cmp':>6} {'w_grd':>6} "
              f"{'r_grd':>6} {'term':>6}")
        for line in lines:
            try:
                e = json.loads(line)
                print(f"  {e.get('chain_id', 0):>8} "
                      f"{e.get('domain', '?'):<14.14} "
                      f"{e.get('status', '?'):<12.12} "
                      f"{e.get('w_legacy', 0):>6.3f} "
                      f"{e.get('w_compound', 0):>6.3f} "
                      f"{e.get('w_grounded', 0):>6.3f} "
                      f"{e.get('r_grounded', 0):>6.3f} "
                      f"{e.get('terminal', 0):>6.3f}")
            except Exception:
                continue
        return

    if subcommand == "audit":
        r = fetch_api("/v4/meta-cgn/audit")
        data = r.get("data", r)
        core = data.get("core", {})
        readiness = data.get("readiness", {})
        by_dom = data.get("by_domain", {})
        print(f"  Status:            {core.get('status', '?')}")
        print(f"  Primitives well-sampled: "
              f"{core.get('primitives_well_sampled', 0)}/5")
        print(f"  Updates total:     {core.get('updates_applied', 0)}"
              f"/2000 gate")
        print(f"  Compositions:      {core.get('compositions_computed', 0)}")
        print(f"  Disagreements:     {core.get('disagreements_logged', 0)}")
        print(f"  Ready to graduate: {core.get('ready_to_graduate', False)}")
        print(f"  Blockers:          "
              f"{'; '.join(readiness.get('blockers', []))[:100]}")
        bwp = core.get("blend_weights_preview", {})
        print(f"\n  Blend weights preview (stage={bwp.get('stage', '?')}):"
              f"  (leg={bwp.get('w_legacy', 0):.2f} "
              f"cmp={bwp.get('w_compound', 0):.2f} "
              f"grd={bwp.get('w_grounded', 0):.2f})")
        print(f"\n  Active domains: {len(by_dom)} — "
              f"{', '.join(sorted(by_dom.keys())[:6])}"
              f"{'...' if len(by_dom) > 6 else ''}")
        print(f"\n  Recent events:")
        print(f"    disagreements (log):  "
              f"{len(data.get('recent_disagreements', []))}")
        print(f"    shadow events (log):  "
              f"{len(data.get('recent_shadow_events', []))}")
        print(f"    blend-weights (log):  "
              f"{len(data.get('recent_blend_weights', []))}")
        print(f"    failures (log):       "
              f"{len(data.get('recent_failures', []))}")
        return

    print(f"  Unknown subcommand: {subcommand}")
    print("  Available: status | readiness | disagreements | failsafe | "
          "impasse | audit | domains | history")


def run_meta_teacher(scope: str = "all") -> None:
    """arch_map meta-teacher [--t1|--t2|--t3|--all].

    Multi-Titan diagnostic for Meta-Reasoning Teacher (rFP_titan_meta_reasoning_teacher).
    Queries /v4/meta-teacher/status on each Titan and summarizes:
      - enabled + sample_mode + task_key (which LLM)
      - 24h critique volume + LLM-ok rate
      - top critique categories
      - adoption rate per domain
    """
    import urllib.request

    TITANS = {
        "T1": "http://127.0.0.1:7777",
        "T2": "http://10.135.0.6:7777",
        "T3": "http://10.135.0.6:7778",
    }
    if scope == "t1":
        titans = {"T1": TITANS["T1"]}
    elif scope == "t2":
        titans = {"T2": TITANS["T2"]}
    elif scope == "t3":
        titans = {"T3": TITANS["T3"]}
    else:
        titans = TITANS

    def fetch(base: str) -> dict:
        try:
            with urllib.request.urlopen(
                    f"{base}/v4/meta-teacher/status", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    print("\n" + "=" * 78)
    print("META-REASONING TEACHER — multi-Titan view")
    print("=" * 78)

    for tid, base in titans.items():
        r = fetch(base)
        data = r.get("data", r)
        print(f"\n── {tid} ({base}) ──")
        if "error" in data:
            print(f"  ✗ {data['error']}")
            continue
        enabled = data.get("enabled", False)
        flag = "ENABLED ✓" if enabled else "disabled"
        print(f"  Status:         {flag}  "
              f"(sample={data.get('sample_mode','?')} "
              f"task={data.get('task_key','?')})")
        print(f"  Weights:        reward={data.get('reward_weight_config',0):.3f} "
              f"(cap={data.get('reward_weight_cap',0):.2f}) "
              f"grounding={data.get('grounding_weight',0):.3f}")
        cr = data.get("critiques_24h", 0)
        ok = data.get("llm_ok_24h", 0)
        failed = data.get("llm_failed_24h", 0)
        avgq = data.get("avg_quality_score_24h", 0.0)
        print(f"  24h Critiques:  {cr} total  "
              f"(ok={ok} failed={failed} rate={ok/max(1,cr):.0%}) "
              f"avg_quality={avgq:.3f}")
        top_cats = data.get("top_critique_categories") or []
        if top_cats:
            top_str = " ".join(f"{c[0]}={c[1]}" for c in top_cats[:5])
            print(f"  Top categories: {top_str}")
        else:
            print(f"  Top categories: (none yet)")
        adoption = data.get("adoption_rate_by_domain") or {}
        overall = data.get("adoption_rate_overall", 0.0)
        if adoption:
            top_adopt = sorted(adoption.items(), key=lambda x: -x[1])[:4]
            ad_str = " ".join(f"{d}={v:.2f}" for d, v in top_adopt)
            print(f"  Adoption:       overall={overall:.2f}  {ad_str}")
        else:
            print(f"  Adoption:       (no suggestions tracked yet)")

    print("\n" + "=" * 78)


def run_emot_cgn(
    scope: str = "all",
    *,
    as_json: bool = False,
) -> None:
    """arch_map emot-cgn [--t1|--t2|--t3|--all] [--json].

    Single-snapshot diagnostic for EMOT-CGN across T1/T2/T3. Consolidates
    three endpoints into one view:
      - /v4/emot-cgn                    legacy primitives + v3 emergence block
      - /v4/emot-cgn/audit               shadow_tail + watchdog + haov + dead_groups
      - /v4/emot-cgn/graduation-readiness 5 graduation criteria

    Flags anchor-silent primitives (conf < 0.1 or n_samples < 50), reports
    v3 region emergence + HDBSCAN health, and surfaces graduation blockers.
    """
    import urllib.request

    TITANS = {
        "T1": "http://127.0.0.1:7777",
        "T2": "http://10.135.0.6:7777",
        "T3": "http://10.135.0.6:7778",
    }
    if scope == "t1":
        selected = {"T1": TITANS["T1"]}
    elif scope == "t2":
        selected = {"T2": TITANS["T2"]}
    elif scope == "t3":
        selected = {"T3": TITANS["T3"]}
    else:
        selected = TITANS

    def _fetch(url: str) -> dict:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def _snapshot(base: str) -> dict:
        state = _fetch(f"{base}/v4/emot-cgn").get("data", {}) or {}
        audit = _fetch(f"{base}/v4/emot-cgn/audit").get("data", {}) or {}
        grad = _fetch(f"{base}/v4/emot-cgn/graduation-readiness").get(
            "data", {}) or {}
        return {"state": state, "audit": audit, "grad": grad}

    snaps = {t: _snapshot(u) for t, u in selected.items()}

    if as_json:
        print(json.dumps(snaps, indent=2, default=str))
        return

    print("\nEMOT-CGN — STATUS")
    print("=" * 78)

    total_silent = 0
    total_regions = 0
    any_eligible = False
    any_unexpected_dead = False

    for titan, snap in snaps.items():
        s = snap["state"]
        a = snap["audit"]
        g = snap["grad"]
        v3 = s.get("v3") or {}
        prims = s.get("primitives") or {}
        recent = s.get("recent_assignments") or []

        print(f"\n── {titan} ({selected[titan]}) ──")
        if "error" in s:
            print(f"  ERROR: {s['error']}")
            continue

        # Header
        print(f"  Status:         {s.get('status', '?')}  "
              f"is_active={s.get('is_active', False)}  "
              f"source={s.get('source', '?')}  "
              f"shm_v={s.get('shm_version', 0)}")
        print(f"  Updates:        {s.get('total_updates', 0)}  "
              f"observations={s.get('total_observations', 0)}  "
              f"rolled_back={s.get('rolled_back_count', 0)}")
        print(f"  Dominant:       {s.get('dominant', '?')}  "
              f"V_beta={s.get('dominant_V_beta', 0):.3f}  "
              f"V_blended={s.get('dominant_V_blended', 0):.3f}  "
              f"clust_conf={s.get('cluster_confidence', 0):.3f}")
        print(f"  Cross-insights: sent={s.get('cross_insights_sent', 0)}  "
              f"recv={s.get('cross_insights_received', 0)}")
        # Monoculture detection on recent_assignments
        if recent:
            tail = recent[-10:]
            uniq = set(tail)
            mono = next(iter(uniq)) if len(uniq) == 1 else None
            if mono:
                print(f"  Recent10:       {len(tail)}x {mono}  "
                      f"⚠ monoculture")
            else:
                print(f"  Recent10:       {tail[-8:]}")

        # v3 emergence block
        if v3:
            rid = v3.get("region_id", -2)
            regions_n = v3.get("regions_emerged", 0)
            total_regions += regions_n
            rid_label = "noise" if rid == -2 else f"#{rid}"
            print(f"  v3 region:      id={rid_label}  "
                  f"regions_emerged={regions_n}  "
                  f"sig=0x{v3.get('region_signature', 0):x}  "
                  f"conf={v3.get('region_confidence', 0):.3f}  "
                  f"resid={v3.get('region_residence_s', 0):.0f}s")
            print(f"  v3 felt:        valence={v3.get('valence', 0):+.3f}  "
                  f"arousal={v3.get('arousal', 0):+.3f}  "
                  f"novelty={v3.get('novelty', 0):.3f}  "
                  f"legacy_approx={v3.get('legacy_approximation', '?')}  "
                  f"grad_status={v3.get('graduation_status', 0)}")

        # Legacy primitives
        print("  Primitives:")
        print("    " + f"{'name':<18} {'V':>6}  {'conf':>6}  "
                        f"{'n':>6}  flag")
        silent_here = []
        for pid, p in prims.items():
            ns = p.get("n_samples", 0)
            cf = p.get("confidence", 0.0)
            flag = ""
            if ns == 0:
                flag = "COLD"
            elif ns < 50 or cf < 0.1:
                flag = "silent"
            if flag:
                silent_here.append(pid)
            print(f"    {pid:<18} {p.get('V', 0):>6.3f}  "
                  f"{cf:>6.3f}  {ns:>6}  {flag}")
        if silent_here:
            total_silent += len(silent_here)
            print(f"    → silent: {len(silent_here)}/{len(prims)} "
                  f"({', '.join(silent_here)})")

        # Graduation readiness (5 criteria)
        print("  Graduation readiness:")
        print(f"    eligible={g.get('eligible', False)}  "
              f"progress={g.get('graduation_progress', 0)}/100")
        if g.get("eligible"):
            any_eligible = True
        crit = g.get("criteria", {}) or {}
        for name, row in crit.items():
            tick = "✓" if row.get("ok") else "✗"
            cur = row.get("current", 0)
            req = row.get("required", 0)
            if isinstance(cur, float):
                cur_s = f"{cur:.3f}"
            else:
                cur_s = f"{cur}"
            if isinstance(req, float):
                req_s = f"{req:.3f}"
            else:
                req_s = f"{req}"
            print(f"    {tick} {name:<24} {cur_s} / {req_s}")

        # HAOV counts
        haov = a.get("haov") or {}
        by_status = {"nascent": 0, "testing": 0, "confirmed": 0,
                     "falsified": 0}
        for h in (haov.get("hypotheses") or {}).values():
            st = h.get("status", "nascent")
            by_status[st] = by_status.get(st, 0) + 1
        print(f"  HAOV:           nascent={by_status['nascent']}  "
              f"testing={by_status['testing']}  "
              f"confirmed={by_status['confirmed']}  "
              f"falsified={by_status['falsified']}")

        # Bundle snapshot — dead_groups sanity (only cgn_beta_states is
        # the documented deferred field per rFP §23.6a)
        bs = a.get("bundle_snapshot") or {}
        dead = bs.get("dead_groups") or []
        expected_dead = {"cgn_beta_states"}
        unexpected = [d for d in dead if d not in expected_dead]
        if unexpected:
            any_unexpected_dead = True
            print(f"  ⚠ dead_groups:  {dead}  (unexpected: {unexpected})")
        elif dead:
            print(f"  dead_groups:    {dead}  (expected)")
        # Shadow tail / watchdog summary
        shadow = a.get("shadow_tail") or []
        wd = a.get("watchdog") or {}
        print(f"  Shadow tail:    len={len(shadow)}  "
              f"last_chain={shadow[-1].get('chain') if shadow else '—'}")
        wd_nm = list((wd.get("neuromod_ema") or {}).keys())
        print(f"  Watchdog:       dominant={wd.get('dominant_emotion', '?')}  "
              f"recent_rewards={len(wd.get('recent_rewards') or [])}  "
              f"neuromod_keys={len(wd_nm)}")

    # Cross-Titan summary
    print("\n" + "─" * 78)
    print(f"  SUMMARY: {len(selected)} Titan(s) queried")
    print(f"  Silent primitives (cumulative): {total_silent}")
    print(f"  Regions emerged (sum): {total_regions}")
    print(f"  Any eligible for graduation: {any_eligible}")
    if any_unexpected_dead:
        print(f"  ⚠ Unexpected dead_groups detected — investigate "
              f"bundle_snapshot")
    print("=" * 78)

def run_meta_outer(scope: str = "t1", as_json: bool = False,
                    recall_person: str = None,
                    recall_topic: str = None) -> None:
    """arch_map meta-outer [--t1|--t2|--t3|--all] [--json]
                           [--recall-person X] [--recall-topic Y]

    rFP_titan_meta_outer_layer diagnostic. Queries:
      - /v4/meta-outer/status         — activation flag + reader status
      - /v4/meta-outer/stats          — cache + per-source + fetch metrics
      - /v4/meta-outer/recall-test    — live composed-recall diagnostic
        (if --recall-person or --recall-topic given)
    """
    import json as _json
    targets: list = []
    if scope == "all":
        targets = [
            ("T1", "http://127.0.0.1:7777"),
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]
    elif scope == "t2":
        targets = [("T2", "http://10.135.0.6:7777")]
    elif scope == "t3":
        targets = [("T3", "http://10.135.0.6:7778")]
    else:
        targets = [("T1", "http://127.0.0.1:7777")]

    out: dict = {}
    for label, base in targets:
        entry: dict = {"base": base}
        try:
            st = _fetch(f"{base}/v4/meta-outer/status").get("data") or {}
            entry["status"] = st
        except Exception as e:
            entry["status_err"] = str(e)
        try:
            stt = _fetch(f"{base}/v4/meta-outer/stats").get("data") or {}
            entry["stats"] = stt
        except Exception as e:
            entry["stats_err"] = str(e)
        if recall_person or recall_topic:
            params = []
            if recall_person:
                params.append(f"person_id={recall_person}")
            if recall_topic:
                params.append(f"topic={recall_topic}")
            url = f"{base}/v4/meta-outer/recall-test?" + "&".join(params)
            try:
                rt = _fetch(url, timeout=2.0).get("data") or {}
                entry["recall_test"] = rt
            except Exception as e:
                entry["recall_test_err"] = str(e)
        out[label] = entry

    if as_json:
        print(_json.dumps(out, indent=2, default=str))
        return

    print("\nMETA-OUTER DIAGNOSTICS — rFP_titan_meta_outer_layer")
    print("=" * 86)
    for label, entry in out.items():
        print(f"\n  {label} ({entry.get('base', '?')})")
        print("  " + "-" * 70)
        st = entry.get("status") or {}
        active = st.get("active", "?")
        wired = st.get("reader_wired", "?")
        mark_active = "🟢 ACTIVE" if active is True else "⚪ inactive"
        print(f"    Activation:     {mark_active}  (flag file)")
        print(f"    Reader wired:   {wired}  "
              f"(cosmetic — main-proc view; reader lives in spirit_worker)")
        stats = entry.get("stats") or {}
        if stats:
            print(f"    Composed fetches: {stats.get('composed_fetches', 0)}  "
                  f"timeouts={stats.get('composed_timeouts', 0)}  "
                  f"avg_ms={stats.get('avg_fetch_ms', 0.0)}")
            c = stats.get("cache") or {}
            print(f"    Cache:            size={c.get('size', 0)}/"
                  f"{c.get('max_size', 0)}  hits={c.get('hits', 0)}  "
                  f"misses={c.get('misses', 0)}  "
                  f"hit_rate={c.get('hit_rate', 0.0):.2%}")
            per_src = stats.get("per_source_calls") or {}
            if per_src:
                print(f"    Per-source calls:")
                for src, n in sorted(per_src.items(), key=lambda x: -x[1]):
                    print(f"      {src:28s}  {n}")
            fails = stats.get("per_source_failures") or {}
            if fails:
                print(f"    Per-source failures:")
                for src, n in fails.items():
                    print(f"      ⚠ {src:26s}  {n}")
        rt = entry.get("recall_test")
        if rt:
            refs = rt.get("entity_refs") or {}
            comp = rt.get("composed") or {}
            print(f"    Recall-test:    refs={refs}")
            print(f"                    fetch_ms={comp.get('fetch_ms', '?')}")
            print(f"                    sources_queried={len(comp.get('sources_queried', []))}")
            print(f"                    sources_timed_out="
                  f"{comp.get('sources_timed_out', [])}")
            print(f"                    person={'YES' if comp.get('person') else 'None'}  "
                  f"topic={'YES' if comp.get('topic') else 'None'}  "
                  f"felt={len(comp.get('felt_history') or [])}  "
                  f"events={len(comp.get('recent_events') or [])}")
    print()


def _collect_meta_cgn_snapshot() -> str:
    """Read data/meta_cgn/primitive_grounding.json and format as a markdown
    table. Returns empty string if META-CGN hasn't been written yet
    (pre-Phase 1 deployment). Pure read — no side effects."""
    path = os.path.join(PROJECT_ROOT, "data", "meta_cgn",
                        "primitive_grounding.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            data = json.load(f)
        prims = data.get("primitives", {})
        if not prims:
            return ""
        stats = data.get("stats", {})
        lines = []
        lines.append(f"*Captured from `{path}` at session close.*\n\n")
        lines.append(f"- **Transitions sent:** "
                     f"{stats.get('transitions_sent', 0)}\n")
        lines.append(f"- **Updates applied:** "
                     f"{stats.get('updates_applied', 0)}\n")
        lines.append(f"- **Compositions computed:** "
                     f"{stats.get('compositions_computed', 0)}\n")
        lines.append(f"- **Disagreements logged:** "
                     f"{stats.get('disagreements_logged', 0)}\n\n")
        lines.append("| Primitive | V | Confidence | n_samples | Variance |\n")
        lines.append("|-----------|---|-----------|-----------|----------|\n")
        # Preserve META_PRIMITIVES order for consistent reading
        order = ["FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
                 "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF",
                 "INTROSPECT"]
        for p_id in order:
            c = prims.get(p_id)
            if not c:
                continue
            lines.append(f"| {p_id} | {c.get('V', 0):.3f} | "
                         f"{c.get('confidence', 0):.3f} | "
                         f"{c.get('n_samples', 0)} | "
                         f"{c.get('variance', 0):.3f} |\n")
        return "".join(lines)
    except Exception as e:
        return f"*META-CGN snapshot unavailable: {e}*\n"


def _append_meta_cgn_trajectory(date_str: str, title: str) -> None:
    """Append one-line TSV row per session to
    titan-docs/sessions/meta_cgn_trajectory.tsv. Columns (P6):
        date  transitions  updates  composed  disagreements  V_avg  conf_avg
        gini  usage_FORMULATE ... usage_INTROSPECT  beta_disp_ema  title

    Gini + per-primitive usage shares let us chart monoculture evolution
    across sessions (I2). Safe if META-CGN state not yet written.
    """
    src = os.path.join(PROJECT_ROOT, "data", "meta_cgn",
                       "primitive_grounding.json")
    if not os.path.exists(src):
        return
    try:
        with open(src) as f:
            data = json.load(f)
        prims = data.get("primitives", {})
        stats = data.get("stats", {})
        if not prims:
            return
        # Canonical primitive order (matches meta_cgn.PRIMITIVES)
        prim_order = [
            "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
            "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF",
            "INTROSPECT",
        ]
        Vs = [float(p.get("V", 0.5)) for p in prims.values()]
        confs = [float(p.get("confidence", 0.0)) for p in prims.values()]
        v_avg = sum(Vs) / max(1, len(Vs))
        conf_avg = sum(confs) / max(1, len(confs))
        ns = [max(0, int(prims.get(p, {}).get("n_samples", 0)))
              for p in prim_order]
        n_total = sum(ns) or 1
        shares = [round(n / n_total, 4) for n in ns]
        # Gini coefficient over n_samples (monoculture = 1, uniform = 0)
        ns_sorted = sorted(ns)
        cum = sum(i * v for i, v in enumerate(ns_sorted, start=1))
        gini = ((2 * cum) / (len(ns) * max(1, sum(ns))) -
                (len(ns) + 1) / len(ns)) if sum(ns) > 0 else 0.0
        out_path = os.path.join(PROJECT_ROOT, "titan-docs", "sessions",
                                "meta_cgn_trajectory.tsv")
        # Schema v2 header — detect old v1 files and migrate in-place.
        new_file = not os.path.exists(out_path)
        header_v2_parts = (
            ["date", "transitions", "updates", "composed", "disagreements",
             "V_avg", "conf_avg", "gini"]
            + [f"usage_{p}" for p in prim_order]
            + ["beta_disp_ema", "title"]
        )
        header_v2 = "\t".join(header_v2_parts) + "\n"
        if not new_file:
            with open(out_path) as rf:
                first = rf.readline()
            if "gini" not in first:
                # v1 schema — rotate old file
                os.replace(out_path, out_path + ".v1.bak")
                new_file = True
        clean_title = (title or "").replace("\t", " ")[:80]
        # Note: beta_disp_ema not persisted to JSON (it's in-memory only).
        # Future work: persist β-dispersion snapshot in save_state.
        beta_disp_ema = stats.get("beta_score_dispersion_ema", 0.0)
        with open(out_path, "a") as f:
            if new_file:
                f.write(header_v2)
            row = [
                date_str,
                str(stats.get("transitions_sent", 0)),
                str(stats.get("updates_applied", 0)),
                str(stats.get("compositions_computed", 0)),
                str(stats.get("disagreements_logged", 0)),
                f"{v_avg:.4f}",
                f"{conf_avg:.4f}",
                f"{gini:.4f}",
            ] + [f"{s:.4f}" for s in shares] + [
                f"{float(beta_disp_ema):.4f}",
                clean_title,
            ]
            f.write("\t".join(row) + "\n")
    except Exception:
        # Session close must never fail on trajectory logging
        pass


def run_session_close(title: str = "", commit: bool = True):
    """Parse current Claude session JSONL into conversation + session markdown, then commit.

    Produces:
      - titan-docs/conversations/CONVERSATION_YYYYMMDD_<slug>.md
      - titan-docs/sessions/SESSION_YYYYMMDD_<slug>.md (template)

    JSONL structure (Claude Code):
      Each line is JSON with 'type' field: 'user', 'assistant', 'attachment', etc.
      Message content in obj['message']['content'] — either string or list of blocks.
      Block types: 'text' (shown to user), 'tool_use', 'tool_result', 'thinking'.
    """
    import glob as _glob
    import subprocess

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    date_human = now.strftime("%Y-%m-%d")

    # ── 1. Find the most recent JSONL session file ──
    jsonl_dir = os.path.expanduser(
        "~/.claude/projects/-home-antigravity-projects-titan")
    jsonl_files = sorted(
        _glob.glob(os.path.join(jsonl_dir, "*.jsonl")),
        key=os.path.getmtime, reverse=True)

    if not jsonl_files:
        print("ERROR: No JSONL session files found in", jsonl_dir)
        return

    jsonl_path = jsonl_files[0]
    session_id = os.path.basename(jsonl_path).replace(".jsonl", "")
    short_id = session_id[:8]

    print(f"\n  SESSION CLOSE — {date_human}")
    print("=" * 70)
    print(f"  JSONL:      {jsonl_path}")
    print(f"  Session ID: {session_id}")

    # ── 1a. Kick off dead-wiring full scan in background ─────────────────
    # Runs in parallel with JSONL parsing (~2min scan vs ~30s parse).
    # Non-blocking: we fire-and-wait at the reporting step. Result feeds
    # into the session doc as a pre-close validation — surfaces any new
    # orphan methods, bus gaps, or config sections the session introduced
    # so we know of imminent problems BEFORE every close.
    _dead_wiring_proc = None
    _dead_wiring_out_path = f"/tmp/dead_wiring_session_{short_id}.json"
    try:
        print(f"  [1a] Launching arch_map dead-wiring scan (background)...")
        _dead_wiring_proc = subprocess.Popen(
            [
                os.path.join("test_env", "bin", "python"),
                "scripts/arch_map_dead_wiring.py",
                "--all", "--json",
            ],
            stdout=open(_dead_wiring_out_path, "w"),
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
        )
    except Exception as _dw_err:
        print(f"  [1a] dead-wiring launch failed (non-fatal): {_dw_err}")
        _dead_wiring_proc = None

    # ── 2. Parse JSONL into conversation transcript ──
    human_msgs = []
    assistant_msgs = []
    human_idx = 0
    assistant_idx = 0

    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            content = obj.get("message", {}).get("content", "")
            text_parts = []

            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            text_parts.append(text)

            if not text_parts:
                continue

            combined = "\n\n".join(text_parts)

            # Skip system-reminder-only messages and very short tool chatter
            stripped = combined.strip()
            if not stripped:
                continue
            if stripped.startswith("<system-reminder>") and stripped.endswith("</system-reminder>"):
                continue
            # Skip task notification only messages
            if stripped.startswith("<task-notification>") and stripped.endswith("</task-notification>"):
                continue

            if msg_type == "user":
                # Filter out tool_result messages (they show up as type=user)
                role = obj.get("message", {}).get("role", "")
                if role == "user":
                    # Check if content is ONLY tool results
                    if isinstance(content, list):
                        has_real_text = any(
                            b.get("type") == "text"
                            for b in content
                            if isinstance(b, dict) and b.get("type") != "tool_result"
                        )
                        if not has_real_text:
                            continue
                    human_idx += 1
                    human_msgs.append((human_idx, combined))
            else:
                assistant_idx += 1
                assistant_msgs.append((assistant_idx, combined))

    print(f"  Parsed:     {len(human_msgs)} human, {len(assistant_msgs)} assistant messages")

    if not human_msgs and not assistant_msgs:
        print("  ERROR: No text messages found in session.")
        return

    # ── 3. Build interleaved conversation markdown ──
    # Reconstruct order: alternate human/assistant by original sequence
    all_msgs = []
    for idx, text in human_msgs:
        all_msgs.append(("Human", idx, text))
    for idx, text in assistant_msgs:
        all_msgs.append(("Assistant", idx, text))

    # Sort by a heuristic: use line position tracking from JSONL
    # Since we process linearly, we can re-parse to get ordering
    ordered_msgs = []
    h_idx = 0
    a_idx = 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = obj.get("type")
            if msg_type == "user" and h_idx < len(human_msgs):
                content = obj.get("message", {}).get("content", "")
                text_parts = []
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                text_parts.append(t)
                if text_parts:
                    combined = "\n\n".join(text_parts)
                    stripped = combined.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("<system-reminder>"):
                        continue
                    if stripped.startswith("<task-notification>"):
                        continue
                    # Check for tool-result-only
                    if isinstance(content, list):
                        has_real = any(
                            b.get("type") == "text"
                            for b in content
                            if isinstance(b, dict) and b.get("type") != "tool_result"
                        )
                        if not has_real:
                            continue
                    role = obj.get("message", {}).get("role", "")
                    if role == "user":
                        h_idx += 1
                        ordered_msgs.append(("Human", h_idx, combined))
            elif msg_type == "assistant" and a_idx < len(assistant_msgs):
                content = obj.get("message", {}).get("content", [])
                text_parts = []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                text_parts.append(t)
                if text_parts:
                    combined = "\n\n".join(text_parts)
                    if combined.strip():
                        a_idx += 1
                        ordered_msgs.append(("Assistant", a_idx, combined))

    # Generate slug from title
    if title:
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:50]
    else:
        slug = short_id

    # ── 4. Write conversation file ──
    conv_filename = f"CONVERSATION_{date_str}_{slug}.md"
    conv_path = os.path.join(PROJECT_ROOT, "titan-docs", "conversations", conv_filename)
    os.makedirs(os.path.dirname(conv_path), exist_ok=True)

    with open(conv_path, "w") as f:
        f.write(f"# Conversation: {date_human} — {title or 'Session ' + short_id}\n\n")
        f.write("> Parsed from Claude session JSONL. Contains full text of all "
                "human and assistant messages.\n\n---\n\n")

        for role, idx, text in ordered_msgs:
            f.write(f"\n## {role} [{idx}]\n\n")
            # Clean up system reminders embedded in text
            cleaned = re.sub(
                r'<system-reminder>.*?</system-reminder>',
                '', text, flags=re.DOTALL).strip()
            if cleaned:
                f.write(cleaned + "\n\n")

    print(f"  Conversation: {conv_path}")
    print(f"               ({len(ordered_msgs)} messages)")

    # ── 4b. Extract high-signal architectural discussions (HIGHLIGHTS file) ──
    # 2026-04-12: auto-extract message pairs where architectural markers appear.
    # Makes "where did we discuss X?" searchable in seconds instead of loading
    # the full conversation file.
    highlights_filename = f"HIGHLIGHTS_{date_str}_{slug}.md"
    highlights_path = os.path.join(
        PROJECT_ROOT, "titan-docs", "conversations", highlights_filename)

    # Markers that indicate high-signal architectural/design discussion.
    # Case-insensitive substring match on message text.
    ARCH_MARKERS = [
        # Design keywords
        "architect", "design decision", "proposal", "propose", "suggest",
        "rfp", "locked in", "sunset", "transitional", "deferred",
        # Decision / alignment
        "i agree", "agreed", "let's", "option a", "option b", "option c",
        # Architectural concepts specific to Titan
        "α + β", "α+β", "β+α", "layer α", "layer β",
        "cgn-extract", "meta-cgn", "emot-cgn", "cgn consumer",
        "stateregistry", "titanvm", "trinity", "/dev/shm",
        "cgnconsumerclient", "cgn_worker", "divinebus",
        # rFP / implementation
        "phase 0", "phase 1", "phase 2", "phase 3", "phase 4",
        "phase 5", "phase 6", "phase 7",
        # Error corrections (valuable to preserve)
        "i was wrong", "apolog", "correction", "verify before declare",
        "you were right", "pushback", "push back",
    ]

    def _has_marker(text: str) -> bool:
        lower = text.lower()
        return any(m in lower for m in ARCH_MARKERS)

    # Find pairs where either side has markers — include context (previous msg)
    highlight_pairs = []
    for i, (role, idx, text) in enumerate(ordered_msgs):
        # Clean system reminders for marker check
        clean_text = re.sub(
            r'<system-reminder>.*?</system-reminder>',
            '', text, flags=re.DOTALL).strip()
        if _has_marker(clean_text):
            # Include previous message as context
            if i > 0:
                prev = ordered_msgs[i - 1]
                if prev not in [p for p, _ in highlight_pairs]:
                    highlight_pairs.append((prev, "context"))
            highlight_pairs.append(((role, idx, clean_text), "marker"))

    os.makedirs(os.path.dirname(highlights_path), exist_ok=True)
    with open(highlights_path, "w") as f:
        f.write(f"# Architectural Highlights: {date_human} — "
                f"{title or 'Session ' + short_id}\n\n")
        f.write("> Auto-extracted message pairs containing architectural/design "
                "markers. Use this to quickly find key discussions; see full "
                f"text in `CONVERSATION_{date_str}_{slug}.md`.\n\n")
        f.write(f"> Markers: architect, design decision, proposal, rfp, "
                f"α/β, CGN/META-CGN, Trinity, /dev/shm, phase 0-7, "
                f"corrections (\"i was wrong\", \"verify before declare\").\n\n")
        f.write(f"> **{len(highlight_pairs)} of {len(ordered_msgs)} messages "
                f"matched markers (including context pairs).**\n\n---\n\n")

        if not highlight_pairs:
            f.write("*No high-signal architectural discussions detected by "
                    "automated markers. Full session in CONVERSATION file.*\n")
        else:
            for (role, idx, text), tag in highlight_pairs:
                tag_suffix = " *(context)*" if tag == "context" else ""
                f.write(f"\n## {role} [{idx}]{tag_suffix}\n\n")
                # Truncate extremely long messages (>3000 chars) with pointer
                if len(text) > 3000:
                    f.write(text[:3000] + f"\n\n*[truncated — full text in "
                            f"CONVERSATION_{date_str}_{slug}.md line for "
                            f"{role} [{idx}]]*\n\n")
                else:
                    f.write(text + "\n\n")

    print(f"  Highlights:   {highlights_path}")
    print(f"               ({len(highlight_pairs)} marker-bearing messages)")

    # ── 5. Write session template file ──
    sess_filename = f"SESSION_{date_str}_{slug}.md"
    sess_path = os.path.join(PROJECT_ROOT, "titan-docs", "sessions", sess_filename)

    # Gather git log for this session (commits today)
    try:
        git_log = subprocess.check_output(
            ["git", "log", "--oneline", f"--since=12 hours ago"],
            cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_log = "(could not read git log)"

    # ── Collect dead-wiring scan result (launched at step 1a) ──
    # Waits for completion (scan takes ~2min; JSONL parsing usually runs
    # longer so this typically returns instantly). Non-fatal if scan
    # errored: session still closes, but the doc notes the skip.
    dead_wiring_summary: dict = {}
    dead_wiring_report_md = "*Dead-wiring scan unavailable — see stderr*"
    if _dead_wiring_proc is not None:
        try:
            print(f"  [dead-wiring] Waiting for scan to complete...")
            _dead_wiring_proc.wait(timeout=300)
            with open(_dead_wiring_out_path, "r") as _f:
                _dw_result = json.load(_f)
            dead_wiring_summary = _dw_result.get("summary", {})
            runtime_ev = _dw_result.get("runtime_evidence") or {}
            # Build a compact markdown summary
            lines = [
                f"- **Files scanned:** {_dw_result.get('n_files', '?')}",
                f"- **Method defs:** {_dw_result.get('n_defs', '?')}",
                f"- **Call sites:** {_dw_result.get('n_calls', '?')}",
                f"- **Runtime evidence:** "
                f"{'available' if runtime_ev.get('available') else 'API unreachable'}",
                "",
                "| Category | Count |",
                "|---|---:|",
            ]
            titles_map = {
                "orphan": "Orphan methods",
                "pair_gap": "Pair-closure gaps",
                "unused_config": "Unused config sections",
                "rfp_missing": "rFP entities missing",
                "bus_dead_msg": "Bus dead messages",
                "bus_dead_handler": "Bus dead handlers",
                "bus_unused_type": "Bus unused types",
                "crud_write_only": "DB write-only tables",
                "crud_read_only": "DB read-only tables",
                "static_runtime_divergence": "⚡ Static-runtime divergence",
            }
            for k, pretty in titles_map.items():
                cnt = dead_wiring_summary.get(k, 0)
                if cnt > 0 or k in ("orphan", "unused_config"):
                    lines.append(f"| {pretty} | {cnt} |")
            # Highlight any divergence findings (these are interesting)
            divergences = [
                f for f in _dw_result.get("findings", [])
                if f.get("kind") == "static_runtime_divergence"
            ]
            if divergences:
                lines.append("")
                lines.append("### ⚡ Static-runtime divergences (new this session)")
                for d in divergences[:5]:
                    lines.append(f"- **{d.get('title', '')}** — {d.get('detail', '')[:120]}")
            dead_wiring_report_md = "\n".join(lines)
            print(f"  [dead-wiring] DONE — "
                  f"orphans={dead_wiring_summary.get('orphan', 0)}, "
                  f"config={dead_wiring_summary.get('unused_config', 0)}, "
                  f"bus_dead_msg={dead_wiring_summary.get('bus_dead_msg', 0)}, "
                  f"divergence={dead_wiring_summary.get('static_runtime_divergence', 0)}")
        except Exception as _dw_err:
            print(f"  [dead-wiring] collection failed: {_dw_err}")
            dead_wiring_report_md = f"*Dead-wiring scan failed: {_dw_err}*"

    # Harvest architectural decision candidates from highlight_pairs
    # (first line of marker-bearing messages — short enough to be summary-ish)
    decision_candidates = []
    for (role, idx, text), tag in highlight_pairs:
        if tag != "marker":
            continue
        first_line = text.strip().split("\n")[0][:120]
        if len(first_line) > 30:  # Skip one-word acknowledgements
            decision_candidates.append(f"- [{role} {idx}] {first_line}")

    with open(sess_path, "w") as f:
        f.write(f"# Session Log: {date_human} — {title or 'TODO: add title'}\n\n")
        f.write(f"> **Duration:** ~TODO UTC — ~{now.strftime('%H:%M')} UTC\n")
        f.write(f"> **Branch:** titan-v6\n")
        f.write(f"> **Significance:** TODO: one-line summary\n\n")
        f.write("---\n\n")
        f.write("## Summary\n\nTODO: 2-3 sentence summary of session.\n\n---\n\n")

        # Dead-wiring pre-close validation scan — runs on every session-close
        # as a signal for "did this session introduce any new silent-wiring
        # issues?". A clean summary means work landed cleanly; non-zero in
        # the DIVERGENCE category is the highest-priority signal (static
        # scanner missed a dynamic dispatch → should refine next time).
        f.write("## 🔍 Dead-wiring validation scan\n\n")
        f.write("*Auto-generated by `arch_map_dead_wiring.py --all` at session-close. "
                "Flags new silent-wiring issues before the session commits.*\n\n")
        f.write(dead_wiring_report_md)
        f.write("\n\n---\n\n")

        # Architectural Decisions (Part 1 of session-close enhancement)
        f.write("## Architectural Decisions\n\n")
        f.write("TODO: list decisions locked in this session. "
                "Candidates auto-extracted from highlight markers:\n\n")
        if decision_candidates:
            for dc in decision_candidates[:20]:
                f.write(f"{dc}\n")
            if len(decision_candidates) > 20:
                f.write(f"\n*...and {len(decision_candidates) - 20} more — "
                        f"see HIGHLIGHTS_{date_str}_{slug}.md*\n")
        else:
            f.write("*No automated candidates detected. See HIGHLIGHTS file.*\n")
        f.write("\n---\n\n")

        # Design Discussions (links to message numbers)
        f.write("## Design Discussions\n\n")
        f.write(f"See `HIGHLIGHTS_{date_str}_{slug}.md` for auto-extracted "
                f"architectural dialogue ({len(highlight_pairs)} messages).\n\n")
        f.write("TODO: if any key design conversation isn't in HIGHLIGHTS, "
                "link it here by message number.\n\n---\n\n")

        # rFPs / Tasks Touched
        f.write("## rFPs / Tasks Touched\n\n")
        try:
            # Find rFPs changed today
            rfp_result = subprocess.check_output(
                ["git", "log", "--since=12 hours ago", "--name-only",
                 "--pretty=format:", "--", "titan-docs/rFP_*.md",
                 "titan-docs/DEFERRED_ITEMS.md", "memory/known_issues.md"],
                cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.DEVNULL
            ).strip()
            rfps_touched = sorted(set(l.strip() for l in rfp_result.split("\n") if l.strip()))
            if rfps_touched:
                f.write("**rFPs / planning docs modified:**\n\n")
                for r in rfps_touched:
                    f.write(f"- `{r}`\n")
                f.write("\n")
        except Exception:
            pass
        f.write("TODO: note task state changes (completed, deferred, new tasks created).\n\n---\n\n")

        f.write("## Work Done\n\nTODO: fill in from conversation\n\n---\n\n")
        f.write("## Commits\n\n")
        if git_log:
            for line in git_log.split('\n')[:15]:
                f.write(f"- `{line}`\n")
        f.write("\n---\n\n")

        # META-CGN primitive grounding snapshot (auto-populated from
        # data/meta_cgn/primitive_grounding.json if present). Lets us track
        # shadow-mode maturation across sessions without digging into JSON.
        _mcgn_snapshot = _collect_meta_cgn_snapshot()
        if _mcgn_snapshot:
            f.write("## META-CGN Grounding Snapshot\n\n")
            f.write(_mcgn_snapshot)
            f.write("\n---\n\n")

        f.write("## Next Session Priorities\n\n")
        f.write("1. TODO\n2. TODO\n3. TODO\n")

    # Append snapshot row to meta_cgn_trajectory.tsv for time-series tracking
    _append_meta_cgn_trajectory(date_str, title or "")

    print(f"  Session:      {sess_path}")
    print(f"               (template — fill in TODOs)")

    # ── 5b. Append to conversation INDEX.md (Part 2 of enhancement) ──
    # One-line per session for fast grep: `grep META-CGN INDEX.md` finds all
    # sessions discussing that topic.
    index_path = os.path.join(PROJECT_ROOT, "titan-docs", "conversations", "INDEX.md")

    # Extract unique architectural topics mentioned in this session (for index line)
    topics_found = set()
    for marker in ARCH_MARKERS:
        for (role, idx, text), tag in highlight_pairs:
            if marker.lower() in text.lower() and len(marker) >= 4:
                # Normalize topic name
                topic = marker.upper() if marker.lower() in ("cgn", "rfp") else marker
                # Skip generic keywords that would bloat the index
                if marker.lower() in ("i agree", "agreed", "let's", "option a",
                                       "option b", "option c", "phase 0", "phase 1",
                                       "phase 2", "phase 3", "phase 4", "phase 5",
                                       "phase 6", "phase 7", "push back", "pushback",
                                       "proposal", "propose", "suggest", "deferred"):
                    continue
                topics_found.add(topic)
                break  # Found marker, move on

    # Grab latest commit hash for anchor
    try:
        last_commit = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%h"],
            cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        last_commit = "?"

    # Create INDEX.md header if missing
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("# Conversation Index\n\n")
            f.write("> Fast-search index of past sessions. Each line: date, "
                    "commit anchor, topics discussed, conversation file. "
                    "Use `grep <topic> INDEX.md` to find sessions on a topic.\n\n")
            f.write("| Date | Commit | Topics | Conversation | Highlights |\n")
            f.write("|------|--------|--------|--------------|------------|\n")

    # Append this session's row
    topics_str = ", ".join(sorted(topics_found)) if topics_found else "(general)"
    if len(topics_str) > 100:
        topics_str = topics_str[:97] + "..."
    title_short = (title or slug)[:40]
    with open(index_path, "a") as f:
        f.write(f"| {date_human} | `{last_commit}` | **{title_short}** — {topics_str} "
                f"| [conv]({conv_filename}) | [highlights]({highlights_filename}) |\n")

    print(f"  Index:        {index_path}  (1 row appended)")
    print(f"               topics={topics_str[:80]}")

    # ── 6a. Check for uncommitted code changes and commit them first ──
    if commit:
        try:
            # SAFETY: build the set of files we must NEVER auto-commit:
            #   1. Files marked --assume-unchanged (e.g. titan_plugin/config.toml
            #      which auto-refreshes Twitter auth_session runtime token).
            #      Per-Titan runtime state — committing it leaks secrets and
            #      stomps on T2/T3 state via git pull.
            #   2. Hardcoded secrets blocklist as defense-in-depth.
            assume_unchanged: set[str] = set()
            try:
                ls_v = subprocess.run(
                    ["git", "ls-files", "-v"],
                    cwd=str(PROJECT_ROOT), capture_output=True, text=True)
                for ls_line in ls_v.stdout.splitlines():
                    if not ls_line:
                        continue
                    # `git ls-files -v` prefixes each file with a flag char.
                    # Lowercase letters = special bits set; 'h' = assume-unchanged.
                    flag = ls_line[0]
                    if flag.islower():
                        # Format: "<flag> <path>"
                        parts = ls_line.split(None, 1)
                        if len(parts) == 2:
                            assume_unchanged.add(parts[1])
            except Exception as _au_err:
                print(f"  ⚠ assume-unchanged probe failed: {_au_err}")

            # Hardcoded secrets blocklist (regardless of assume-unchanged state)
            SECRETS_BLOCKLIST = {
                "titan_plugin/config.toml",  # Twitter auth_session refresh
                "authority.json",
            }
            SECRETS_PATTERNS = (
                "config.toml",
                "credentials",
                "secret",
                ".env",
                "_keypair.json",
                "_authority.json",
            )

            def _is_blocked(path: str) -> bool:
                if path in SECRETS_BLOCKLIST or path in assume_unchanged:
                    return True
                lower = path.lower()
                return any(p in lower for p in SECRETS_PATTERNS)

            # Check for staged + unstaged changes (excluding data/ and untracked)
            diff_result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            changed_files = [f for f in diff_result.stdout.strip().split('\n')
                             if f and not f.startswith('data/')]
            # Also check for new untracked code files (not data/, not docs we're about to commit)
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            new_files = [f for f in untracked_result.stdout.strip().split('\n')
                         if f and not f.startswith('data/')
                         and f not in (conv_path, sess_path)
                         and not f.endswith('.log')]

            # Filter out blocked files BEFORE the extension filter so they
            # never reach the staging step. Log them so the operator can
            # see what was protected.
            blocked = [f for f in changed_files + new_files if _is_blocked(f)]
            if blocked:
                print()
                print(f"  🛡  Blocked from auto-commit (--assume-unchanged or secrets):")
                for bf in blocked:
                    reason = []
                    if bf in SECRETS_BLOCKLIST:
                        reason.append("blocklist")
                    if bf in assume_unchanged:
                        reason.append("assume-unchanged")
                    if any(p in bf.lower() for p in SECRETS_PATTERNS):
                        reason.append("secret-pattern")
                    print(f"    {bf}  ({', '.join(reason)})")

            code_files = [f for f in changed_files + new_files
                          if f.endswith(('.py', '.toml', '.tsx', '.ts', '.md', '.sh', '.json'))
                          and not _is_blocked(f)
                          and f not in (os.path.relpath(conv_path, PROJECT_ROOT),
                                        os.path.relpath(sess_path, PROJECT_ROOT))]

            if code_files:
                print()
                print(f"  Found {len(code_files)} uncommitted code file(s):")
                for cf in code_files[:10]:
                    print(f"    {cf}")
                if len(code_files) > 10:
                    print(f"    ... and {len(code_files) - 10} more")

                # Stage and commit code changes
                subprocess.run(
                    ["git", "add"] + code_files,
                    cwd=str(PROJECT_ROOT), check=True,
                    capture_output=True, text=True)
                code_msg = (f"feat: {title or slug}\n\n"
                            f"{len(code_files)} files changed.\n\n"
                            f"Co-Authored-By: Claude Opus 4.6 (1M context) "
                            f"<noreply@anthropic.com>")
                subprocess.run(
                    ["git", "commit", "-m", code_msg],
                    cwd=str(PROJECT_ROOT), check=True,
                    capture_output=True, text=True)
                print(f"  ✓ Code committed: feat: {title or slug}")
            else:
                print("\n  ✓ No uncommitted code changes")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Code commit check/commit failed: {e.stderr.strip() if e.stderr else e}")
        except Exception as e:
            print(f"  ⚠ Code commit check error: {e}")

    # ── 6b. Commit session docs ──
    if commit:
        print()
        print("  Committing session docs...")
        try:
            subprocess.run(
                ["git", "add", conv_path, sess_path],
                cwd=str(PROJECT_ROOT), check=True,
                capture_output=True, text=True)
            msg = (f"docs: Session log + conversation transcript — "
                   f"{title or slug}\n\n"
                   f"Co-Authored-By: Claude Opus 4.6 (1M context) "
                   f"<noreply@anthropic.com>")
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=str(PROJECT_ROOT), check=True,
                capture_output=True, text=True)
            print("  ✓ Committed: docs: Session log + conversation transcript")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Commit failed: {e.stderr.strip()}")
            print("    (files still saved — commit manually)")
    else:
        print("\n  Skipped commit (--no-commit)")

    print()
    print("=" * 70)
    print(f"  Session close complete. Files ready in titan-docs/")
    print()


# ── CGN Telemetry ─────────────────────────────────────────────────────

def run_timechain_diagnostics(all_titans: bool = True):
    """TimeChain diagnostics: chain status, fork stats, integrity, PoT, growth rate."""
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("TIMECHAIN DIAGNOSTICS — PROOF OF THOUGHT MEMORY ARCHITECTURE")
    print("=" * 78)

    network_blocks = 0
    network_chi = 0.0
    network_contracts = 0

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            # ── 1. Chain Status ──
            # 20s timeout: cold-cache first call does merkle compute across
            # all fork tips + file I/O on a 145k+ block chain (~5s on T1).
            # 10s was enough when the chain was smaller but now races on
            # cold cache. Warm subsequent calls return in <100ms.
            resp = requests.get(f"{url}/v4/timechain/status", timeout=20)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", {})

            genesis = d.get("genesis_hash", "")
            total = d.get("total_blocks", 0)
            forks = d.get("total_forks", 0)
            chi = d.get("total_chi_spent", 0)
            merkle = d.get("merkle_root", "")

            network_blocks += total
            network_chi += chi

            print(f"    CHAIN STATUS")
            print(f"    Genesis:     {genesis[:24]}...")
            print(f"    Blocks:      {total:,}  |  Forks: {forks}  |  "
                  f"Chi spent: {chi:.3f}")
            print(f"    Merkle root: {merkle[:24]}...")

            # Fork breakdown
            fork_data = d.get("forks", {})
            active_forks = [(fid, f) for fid, f in sorted(fork_data.items())
                            if f.get("block_count", 0) > 0]
            if active_forks:
                print(f"    ACTIVE FORKS ({len(active_forks)}):")
                for fid, f in active_forks:
                    name = f.get("name", f"fork_{fid}")
                    ftype = f.get("type", "?")
                    count = f.get("block_count", 0)
                    tip = f.get("tip_height", 0)
                    fchi = f.get("total_chi_spent", 0)
                    avg_sig = f.get("avg_significance", 0)
                    topic = f.get("topic")
                    label = f"{name}"
                    if topic:
                        label += f" [{topic}]"
                    print(f"      {label:25s}: {count:>5} blocks  "
                          f"tip=#{tip:<5d}  chi={fchi:.3f}  "
                          f"avg_sig={avg_sig:.2f}  ({ftype})")

            # Inactive primary forks
            inactive = [(fid, f) for fid, f in sorted(fork_data.items())
                        if f.get("block_count", 0) == 0
                        and f.get("type") == "primary"]
            if inactive:
                names = ", ".join(f.get("name", "?") for _, f in inactive)
                print(f"    Inactive:    {names} (waiting for first block)")

            # ── 2. Chain Integrity ──
            try:
                vresp = requests.get(f"{url}/v4/timechain/verify", timeout=15)
                if vresp.status_code == 200:
                    vd = vresp.json().get("data", {})
                    valid = vd.get("valid", False)
                    results = vd.get("results", [])
                    valid_count = sum(1 for r in results if "valid" in r.lower()
                                      or "does not exist" in r.lower())
                    broken = [r for r in results
                              if "valid" not in r.lower()
                              and "does not exist" not in r.lower()]
                    if valid:
                        print(f"    INTEGRITY:   ✓ ALL FORKS VALID ({valid_count}/{len(results)})")
                    else:
                        print(f"    INTEGRITY:   ⚠ {len(broken)} fork(s) with issues "
                              f"({valid_count} valid)")
                        for r in broken:
                            print(f"      {r}")
            except Exception:
                print(f"    INTEGRITY:   ? (verify timed out)")

            # ── 3. Recent blocks ──
            try:
                bresp = requests.get(f"{url}/v4/timechain/blocks?fork=0&limit=3",
                                     timeout=8)
                if bresp.status_code == 200:
                    blocks = bresp.json().get("data", {}).get("blocks", [])
                    if blocks:
                        print(f"    RECENT MAIN CHAIN:")
                        for b in blocks[:3]:
                            bh = b.get("block_hash", "")[:12]
                            epoch = b.get("epoch_id", 0)
                            src = b.get("source", "?")
                            sig = b.get("significance", 0)
                            print(f"      #{b['height']}  {bh}...  "
                                  f"epoch={epoch}  src={src}  sig={sig:.2f}")
            except Exception:
                pass

            # ── 4. Contract Stats ──
            try:
                cresp = requests.get(f"{url}/v4/timechain/contracts/stats", timeout=10)
                if cresp.status_code == 200:
                    cd = cresp.json().get("data", {})
                    ctotal = cd.get("total", 0)
                    network_contracts += ctotal
                    by_type = cd.get("by_type", {})
                    by_status = cd.get("by_status", {})
                    active_contracts = cd.get("active_contracts", [])

                    type_parts = "  ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
                    status_parts = "  ".join(f"{k}={v}" for k, v in sorted(by_status.items()))

                    print(f"    CONTRACTS ({ctotal} total)")
                    if type_parts:
                        print(f"      By type:   {type_parts}")
                    if status_parts:
                        print(f"      By status: {status_parts}")
                    if active_contracts:
                        for c in active_contracts:
                            cname = c.get("contract_id", c.get("name", "?"))
                            ctype = c.get("type", "?")
                            rules = c.get("rules_count", c.get("rules", 0))
                            print(f"      • {cname:20s}  type={ctype:8s}  rules={rules}")
            except Exception:
                pass

            print(f"\n    ✓ TimeChain active")

        except requests.exceptions.ConnectionError:
            print(f"    ✗ Not reachable")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # ── Network summary ──
    if len(titans) > 1:
        print()
        print("  " + "-" * 72)
        print(f"  NETWORK TOTALS")
        print(f"    Total blocks: {network_blocks:,}  |  "
              f"Total chi spent: {network_chi:.3f}  |  "
              f"Contracts: {network_contracts}  |  "
              f"Titans: {len(titans)}")
        # Check genesis divergence
        genesis_hashes = set()
        for tid, url in titans:
            try:
                r = requests.get(f"{url}/v4/timechain/status", timeout=5)
                if r.status_code == 200:
                    gh = r.json().get("data", {}).get("genesis_hash", "")
                    if gh:
                        genesis_hashes.add(gh[:16])
            except Exception:
                pass
        if len(genesis_hashes) == len(titans):
            print(f"    Genesis:     ✓ All {len(genesis_hashes)} Titans have "
                  f"unique genesis (sovereign chains)")
        elif len(genesis_hashes) > 0:
            print(f"    Genesis:     {len(genesis_hashes)} unique / "
                  f"{len(titans)} Titans")

        # Cross-Titan fork divergence
        fork_blocks = {}
        for tid, url in titans:
            try:
                r = requests.get(f"{url}/v4/timechain/status", timeout=5)
                if r.status_code == 200:
                    fdata = r.json().get("data", {}).get("forks", {})
                    for fid, f in fdata.items():
                        if f.get("type") == "primary" and f.get("block_count", 0) > 0:
                            fname = f.get("name", f"fork_{fid}")
                            if fname not in fork_blocks:
                                fork_blocks[fname] = {}
                            fork_blocks[fname][tid] = f["block_count"]
            except Exception:
                pass
        if fork_blocks:
            print(f"    FORK DIVERGENCE (blocks per Titan):")
            for fname, counts in sorted(fork_blocks.items()):
                parts = "  ".join(f"{t}={c}" for t, c in sorted(counts.items()))
                print(f"      {fname:15s}: {parts}")

    print()
    print("=" * 78)


def run_cognitive_contracts_diagnostics(all_titans: bool = True):
    """TUNING-012 v2 Sub-phase C (R2): cross-Titan cognitive contracts observability.

    Pulls /v4/cognitive-contracts from each Titan and prints execution counts,
    handler outputs, and active diversity-pressure state. Critical for the
    24-48h post-deploy observation window — without this we can't see whether
    monoculture_detector ever fired or strategy_evolution found anything.
    """
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("COGNITIVE CONTRACTS — TUNING-012 v2 SUB-PHASE C")
    print("=" * 78)

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            resp = requests.get(f"{url}/v4/cognitive-contracts", timeout=10)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", {})

            # Contract registration table
            contracts = d.get("contracts", [])
            print(f"    REGISTERED CONTRACTS ({len(contracts)})")
            if not contracts:
                print("      (none — load_meta_cognitive_contracts may not have run)")
            else:
                print(f"      {'contract_id':<32s} {'type':<8s} {'status':<8s} "
                      f"{'fires':>7s} {'last_fire':>16s}  signed")
                for c in contracts:
                    cid = c.get("contract_id", "?")[:32]
                    ctype = c.get("contract_type", "?")[:8]
                    status = c.get("status", "?")[:8]
                    fires = c.get("execution_count", 0)
                    last_ts = c.get("last_executed", 0)
                    if last_ts:
                        import datetime as _dt
                        last_str = _dt.datetime.fromtimestamp(
                            last_ts).strftime("%H:%M:%S")
                        secs_ago = int(time.time() - last_ts)
                        if secs_ago < 60:
                            last_str += f" ({secs_ago}s)"
                        elif secs_ago < 3600:
                            last_str += f" ({secs_ago // 60}m)"
                        else:
                            last_str += f" ({secs_ago // 3600}h)"
                    else:
                        last_str = "never"
                    signed = "Maker" if c.get("approver_signature") else "system"
                    print(f"      {cid:<32s} {ctype:<8s} {status:<8s} "
                          f"{fires:>7d} {last_str:>16s}  {signed}")

            # Handler outputs
            handlers = d.get("handlers", {})
            sd = handlers.get("strategy_drift", {})
            pe = handlers.get("pattern_emerged", {})
            mc = handlers.get("monoculture", {})

            print(f"\n    HANDLERS (spirit_worker)")
            print(f"      strategy_drift   fires={sd.get('fires', 0)}")
            top = sd.get("last_top_templates", [])
            for entry in top[:3]:
                print(f"        score={entry.get('score', 0):.3f} "
                      f"mean={entry.get('mean', 0):.3f} n={entry.get('n', 0)}: "
                      f"{entry.get('template', '')[:50]}")
            print(f"      pattern_emerged  fires={pe.get('fires', 0)}")
            for entry in pe.get("last_emerging", [])[:3]:
                print(f"        count={entry.get('count', 0)}: "
                      f"{entry.get('template', '')[:50]}")
            print(f"      monoculture      fires={mc.get('fires', 0)}")
            mc_last = mc.get("last", {})
            if mc_last:
                applied = "APPLIED" if mc_last.get("applied") else "skipped"
                print(f"        last: {mc_last.get('dominant', '?')}={mc_last.get('share', 0)*100:.1f}% "
                      f"→ {applied} mag={mc_last.get('magnitude', 0):.2f} "
                      f"decay={mc_last.get('decay_chains', 0)}")

            # Diversity pressure live state
            dp = d.get("diversity_pressure", {})
            print(f"\n    DIVERSITY PRESSURE (R3)")
            if dp.get("active"):
                print(f"      ACTIVE: {dp.get('target', '?')} bias=-{dp.get('current_bias', 0):.3f} "
                      f"remaining={dp.get('remaining_chains', 0)} chains "
                      f"(initial mag={dp.get('initial_magnitude', 0):.2f}, "
                      f"decay={dp.get('initial_decay_chains', 0)})")
            else:
                print(f"      idle (total applied since boot: {dp.get('total_applied', 0)})")

            # Eureka thresholds (R5)
            et = d.get("eureka_thresholds", {})
            if et:
                print(f"\n    EUREKA THRESHOLDS (R5 — per-primitive)")
                pairs = sorted(et.items(), key=lambda r: r[1])
                line = "      " + " ".join(f"{k}={v:.2f}" for k, v in pairs)
                print(line)

            print(f"\n    DNA params: {d.get('contracts_dna_param_count', 0)}")
            print(f"    ✓ Cognitive contracts active")

        except requests.exceptions.ConnectionError:
            print(f"    ✗ Not reachable")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    print()
    print("=" * 78)


def run_verify_search_pipeline(all_titans: bool = False):
    """End-to-end knowledge pipeline health verification (KP-6).

    Pulls /v4/search-pipeline/health from each Titan and renders a
    structured green/red per-backend table with circuit state, today's
    request/error counts, bandwidth-vs-budget, avg latency, and cache
    hit rate. Designed to be the one-command answer to "is knowledge
    acquisition healthy?" — supports the verify-before-declare discipline.
    """
    import requests

    titans = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ]

    # Severity marker helpers
    def _circuit_glyph(state: str) -> str:
        return {"closed": "✓", "half_open": "◐", "open": "✗"}.get(state, "?")

    def _fmt_mb(bytes_: int) -> str:
        if bytes_ <= 0:
            return "0"
        return f"{bytes_ / (1024 * 1024):.1f}MB"

    def _fmt_budget(consumed: int, budget: int) -> str:
        if budget <= 0:
            return f"{_fmt_mb(consumed)}/unlimited"
        pct = consumed / budget * 100
        return f"{_fmt_mb(consumed)}/{_fmt_mb(budget)} ({pct:.0f}%)"

    print()
    print("SEARCH PIPELINE — KNOWLEDGE ROUTER HEALTH")
    print("=" * 84)

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 78)

        try:
            r = requests.get(f"{url}/v4/search-pipeline/health", timeout=5)
        except Exception as e:
            print(f"  ✗ API unreachable: {e}")
            continue
        if r.status_code != 200:
            print(f"  ✗ /v4/search-pipeline/health → HTTP {r.status_code}")
            continue

        try:
            data = r.json().get("data", {}) or {}
        except Exception as e:
            print(f"  ✗ JSON parse error: {e}")
            continue

        backends = data.get("backends", {}) or {}
        cache = data.get("cache", {}) or {}
        health_age = data.get("health_file_age_sec")

        if not backends:
            print(f"  ⚠ No backends recorded yet (knowledge_worker idle?)")

        # Per-backend rows — sorted by name for stable output
        overall_ok = True
        for name in sorted(backends.keys()):
            h = backends[name]
            state = h.get("circuit_state", "?")
            glyph = _circuit_glyph(state)
            reqs = h.get("requests_today", 0)
            errs = h.get("errors_today", 0)
            bytes_today = h.get("bytes_consumed_today", 0)
            budget = h.get("budget_daily_bytes", 0)
            avg_lat = h.get("avg_latency_ms", 0.0)
            last_err = h.get("last_error_type", "") or "-"

            is_degraded = (state != "closed"
                           or (budget > 0 and bytes_today >= budget))
            if is_degraded:
                overall_ok = False

            print(f"    {glyph} {name:<20}  {state:<10}  "
                  f"{reqs:>4}/{errs:<3}  "
                  f"{_fmt_budget(bytes_today, budget):<28}  "
                  f"lat={avg_lat:.0f}ms  last_err={last_err}")

        # Cache row
        if cache:
            entries = cache.get("entries", 0)
            hit_rate = cache.get("hit_rate", 0.0)
            hits = cache.get("hits_24h", 0)
            misses = cache.get("misses_24h", 0)
            saved_mb = cache.get("bytes_saved_24h_estimate", 0) / (1024 * 1024)
            print(f"    ✓ cache                 — entries={entries}, "
                  f"hit_rate={hit_rate:.1%} ({hits}/{hits + misses}), "
                  f"bytes_saved_24h~{saved_mb:.1f}MB")

        # Staleness warning
        if health_age is not None and health_age > 3600:
            print(f"    ⚠ health file stale ({health_age:.0f}s old) — "
                  f"knowledge_worker may not be writing")
            overall_ok = False

        # KP-8 — routing learner summary (non-blocking; separate endpoint)
        try:
            lr = requests.get(f"{url}/v4/search-pipeline/learning", timeout=5)
            if lr.status_code == 200:
                ldata = lr.json().get("data", {}) or {}
                by_qt = ldata.get("by_query_type", {}) or {}
                n_qt = len(by_qt)
                warm_rows = sum(
                    1 for rows in by_qt.values() for r in rows if r.get("warm"))
                total_rows = sum(len(rows) for rows in by_qt.values())
                demoted = ldata.get("config", {}).get("demote_threshold", 0.10)
                below = sum(
                    1 for rows in by_qt.values()
                    for r in rows if r.get("warm") and r.get("reputation", 0) < demoted)
                if total_rows == 0:
                    print("    ⚪ learner                — cold "
                          "(no dispatches yet)")
                else:
                    print(f"    ✓ learner                — {warm_rows}/{total_rows} warm "
                          f"across {n_qt} query types, {below} demoted")
        except Exception:
            pass  # learning endpoint is optional

        print(f"\n  Overall: {'HEALTHY' if overall_ok else 'DEGRADED'}")

    print()
    print("=" * 84)


def run_analyze_decisions(path: str = "data/logs/knowledge_router_decisions.jsonl",
                          hours: int = 24) -> None:
    """Aggregate KP-4 decision log into (query_type, backend) rollups.

    Reads the rolling JSONL + any .1/.2 rotations. For each (qt, backend)
    tuple it tallies attempts, successes, cache hits, coalesced, rejected,
    bytes, and latency p50/p95. Filter by --since hours (default 24).

    Usage: python scripts/arch_map.py analyze-decisions [--since 24] [--path=...]
    """
    import os as _os
    import json as _json
    import time as _time
    import statistics as _stats

    paths = []
    for candidate in [path, path + ".1", path + ".2"]:
        if _os.path.exists(candidate):
            paths.append(candidate)
    if not paths:
        print(f"No decision log found at {path}* — knowledge_worker idle?")
        return

    cutoff = _time.time() - hours * 3600
    buckets: dict = {}  # (qt, backend) -> list of entries
    total_read = 0
    total_kept = 0
    rejected = 0
    coalesced = 0
    learned = 0
    for p in paths:
        try:
            with open(p, "r") as f:
                for line in f:
                    total_read += 1
                    try:
                        entry = _json.loads(line)
                    except Exception:
                        continue
                    if float(entry.get("ts", 0)) < cutoff:
                        continue
                    total_kept += 1
                    qt = entry.get("query_type", "?")
                    bk = entry.get("backend_used") or "(none)"
                    key = (qt, bk)
                    buckets.setdefault(key, []).append(entry)
                    if entry.get("rejected"):
                        rejected += 1
                    if entry.get("coalesced"):
                        coalesced += 1
                    if entry.get("learned_chain"):
                        learned += 1
        except Exception as e:
            print(f"  WARN: read {p}: {e}")

    print()
    print("KNOWLEDGE ROUTER — DECISION LOG ANALYSIS")
    print("=" * 78)
    print(f"  Window: last {hours}h | read {total_read} lines | kept {total_kept}")
    print(f"  rejected={rejected}  coalesced={coalesced}  learned_chain={learned}")
    print()

    if not buckets:
        print("  (no entries in window)")
        return

    hdr = f"  {'query_type':<18} {'backend':<24} {'n':>4} {'succ':>4} " \
          f"{'cache':>5} {'rej':>4} {'p50ms':>6} {'p95ms':>6} {'bytesMB':>8}"
    print(hdr)
    print("  " + "-" * 76)

    for (qt, bk), entries in sorted(buckets.items()):
        n = len(entries)
        succ = sum(1 for e in entries if e.get("success"))
        cache = sum(1 for e in entries if e.get("cache_hit"))
        rej = sum(1 for e in entries if e.get("rejected"))
        lats = [float(e.get("latency_ms", 0)) for e in entries
                if e.get("latency_ms")]
        p50 = _stats.median(lats) if lats else 0
        if lats and len(lats) >= 20:
            p95 = _stats.quantiles(lats, n=20)[18]
        elif lats:
            p95 = max(lats)
        else:
            p95 = 0
        total_bytes = sum(int(e.get("bytes", 0) or 0) for e in entries)
        print(f"  {qt:<18} {bk:<24} {n:>4} {succ:>4} {cache:>5} "
              f"{rej:>4} {p50:>6.0f} {p95:>6.0f} "
              f"{total_bytes/(1024*1024):>8.2f}")

    print()
    # Error-type summary across all buckets
    err_counts: dict = {}
    for entries in buckets.values():
        for e in entries:
            et = e.get("error_type") or ""
            if et:
                err_counts[et] = err_counts.get(et, 0) + 1
    if err_counts:
        print("  Errors (last {}h):".format(hours))
        for et in sorted(err_counts, key=lambda k: -err_counts[k]):
            print(f"    {et:<20} {err_counts[et]:>4}")
    print("=" * 78)


def run_verify_cgn_pipeline(all_titans: bool = False):
    """End-to-end CGN pipeline health verification.

    Checks each stage of the CGN flow:
      1. CGN Worker — Guardian module alive, weights being written
      2. /dev/shm — file exists, version counter incrementing
      3. Consumers — all 6 (language, social, knowledge, reasoning, coding, self_model)
         registered and loading weights
      4. HAOV — per-consumer hypothesis formation / testing / confirmation
      5. Knowledge pipeline — requests flowing, StealthSage responding
      6. Dream consolidation — recent consolidation trained multiple consumers

    Reports GREEN/YELLOW/RED per stage. Designed to answer
    "is CGN actually working end-to-end?" in one command — supports the
    'verify before declare' discipline by making pipeline state cheap to inspect.
    """
    import requests
    import re

    titans = [("T1", "http://127.0.0.1:7777", "/tmp/titan_brain.log")]
    if all_titans:
        titans += [
            ("T2", "http://10.135.0.6:7777", "ssh:/tmp/titan2_brain.log"),
            ("T3", "http://10.135.0.6:7778", "ssh:/tmp/titan3_brain.log"),
        ]

    print()
    print("CGN PIPELINE END-TO-END VERIFICATION")
    print("=" * 78)

    for tid, url, log_path in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)

        stages = {}

        # Stage 1: CGN Worker reachable via API
        try:
            r = requests.get(f"{url}/v4/cgn-haov-stats", timeout=5)
            if r.status_code == 200:
                stages["1_api"] = ("✓", "API reachable")
            else:
                stages["1_api"] = ("✗", f"HTTP {r.status_code}")
        except Exception as e:
            stages["1_api"] = ("✗", f"unreachable: {e}")

        # Stage 2: /dev/shm file + version counter (only T1 has local path)
        if log_path.startswith("/tmp"):
            try:
                import struct, os
                shm_path = "/dev/shm/cgn_live_weights.bin"
                if os.path.exists(shm_path):
                    actual_size = os.path.getsize(shm_path)
                    with open(shm_path, "rb") as f:
                        v1, nc, vnet, total = struct.unpack("<IIII", f.read(16))
                    stages["2_shm"] = ("✓", f"v={v1}, consumers={nc}, size={actual_size}B")
                else:
                    stages["2_shm"] = ("✗", "/dev/shm file missing")
            except Exception as e:
                stages["2_shm"] = ("✗", f"error reading: {e}")
        else:
            stages["2_shm"] = ("-", "skip (remote)")

        # Stage 3+4: Consumers + HAOV (via haov-stats endpoint)
        try:
            r = requests.get(f"{url}/v4/cgn-haov-stats", timeout=5)
            if r.status_code == 200:
                d = r.json().get("data", {}).get("consumers", {})
                all_consumers = ["language", "social", "knowledge", "reasoning",
                                 "coding", "self_model"]
                registered = [c for c in all_consumers if c in d]
                missing = [c for c in all_consumers if c not in d]
                if len(registered) == 6:
                    stages["3_consumers"] = ("✓", f"{len(registered)}/6 registered")
                elif len(registered) >= 4:
                    stages["3_consumers"] = ("⚠", f"{len(registered)}/6 registered, missing={missing}")
                else:
                    stages["3_consumers"] = ("✗", f"only {len(registered)}/6, missing={missing}")

                # HAOV activity — at least 2 consumers should have activity
                haov_active = [c for c, info in d.items()
                               if info.get("formed", 0) > 0]
                if len(haov_active) >= 2:
                    stages["4_haov"] = ("✓", f"{len(haov_active)} consumers forming hypotheses: {haov_active}")
                elif len(haov_active) >= 1:
                    stages["4_haov"] = ("⚠", f"only {haov_active} actively forming; others nascent")
                else:
                    stages["4_haov"] = ("✗", "no HAOV activity on any consumer")
            else:
                stages["3_consumers"] = ("✗", "haov-stats HTTP error")
                stages["4_haov"] = ("-", "skipped")
        except Exception as e:
            stages["3_consumers"] = ("✗", f"error: {e}")
            stages["4_haov"] = ("-", "skipped")

        # Stage 5: Knowledge pipeline via /v4/knowledge-stats
        try:
            r = requests.get(f"{url}/v4/knowledge-stats", timeout=5)
            if r.status_code == 200:
                d = r.json().get("data", {})
                total = d.get("total_concepts", 0)
                usage = d.get("total_usage", 0)
                rate_24h = d.get("acquisition_rate_24h", 0)
                if total > 20 and usage > 0:
                    stages["5_knowledge"] = ("✓", f"concepts={total} usage={usage} 24h_rate={rate_24h}")
                elif total > 20:
                    stages["5_knowledge"] = ("⚠", f"concepts={total} usage={usage}=0 (Definition 3 not wired — see CGN-KNOWLEDGE-V2)")
                else:
                    stages["5_knowledge"] = ("⚠", f"only {total} concepts acquired")
            else:
                stages["5_knowledge"] = ("✗", f"knowledge-stats HTTP {r.status_code}")
        except Exception as e:
            stages["5_knowledge"] = ("✗", f"error: {e}")

        # Stage 6: Dream consolidation activity (T1 only — reads local log)
        if log_path.startswith("/tmp"):
            try:
                with open(log_path) as f:
                    content = f.read()
                consolidations = content.count("Dream consolidation starting")
                most_recent = None
                for line in reversed(content.splitlines()):
                    if "Dream consolidation #" in line:
                        most_recent = line[:80]
                        break
                if consolidations >= 5:
                    stages["6_dream"] = ("✓", f"{consolidations} consolidations this session")
                elif consolidations >= 1:
                    stages["6_dream"] = ("⚠", f"only {consolidations} this session (needs time)")
                else:
                    stages["6_dream"] = ("✗", "no dream consolidations observed")
            except Exception as e:
                stages["6_dream"] = ("-", f"log read error: {e}")
        else:
            stages["6_dream"] = ("-", "skip (remote)")

        # Print stages
        stage_labels = {
            "1_api": "API reachable",
            "2_shm": "/dev/shm state",
            "3_consumers": "Consumers registered",
            "4_haov": "HAOV activity",
            "5_knowledge": "Knowledge pipeline",
            "6_dream": "Dream consolidation",
        }
        for sk in sorted(stages.keys()):
            icon, msg = stages[sk]
            label = stage_labels[sk]
            print(f"    {icon} {label:<22s}  {msg}")

        # Summary
        failed = [k for k, (i, _) in stages.items() if i == "✗"]
        warns = [k for k, (i, _) in stages.items() if i == "⚠"]
        if not failed and not warns:
            print(f"    ✓ ALL STAGES HEALTHY")
        elif failed:
            print(f"    ✗ {len(failed)} stage(s) failed — see above")
        else:
            print(f"    ⚠ {len(warns)} warning(s) — see above")

    print()
    print("=" * 78)


def run_where(symbol: str):
    """Comprehensive location search — supports 'verify before declare' discipline.

    Searches across ALL architectural indices for any mention of `symbol`:
      - File stems (partial match on file paths)
      - Class names (from definitions)
      - Function names (from definitions)
      - Bus message types (from bus_wiring)
      - Attribute access patterns (from attr_index)
      - Config keys (in titan_params.toml + config.toml)

    Case-insensitive substring match. Returns grouped results with full paths.

    Example — `arch_map where agno_hooks` would have caught the 2026-04-12
    error where searching `titan_plugin/logic/agno_hooks.py` returned empty
    but the file exists at `titan_plugin/agno_hooks.py`.
    """
    graph = load_graph()
    sym = symbol.lower()
    found_any = False

    print()
    print(f"WHERE: '{symbol}' — comprehensive architectural location search")
    print("=" * 78)

    # 1. File paths (stems + fragments)
    files = graph.get("files", {})
    file_matches = sorted([fp for fp in files.keys() if sym in fp.lower()])
    if file_matches:
        found_any = True
        print(f"\n  FILES ({len(file_matches)}):")
        for fp in file_matches:
            data = files[fp]
            print(f"    {fp}  ({data.get('lines', '?')} lines)")

    # Also check the real filesystem directly (catches files graph may have missed)
    import subprocess
    try:
        fs_result = subprocess.run(
            ["find", ".", "-name", f"*{symbol}*",
             "-not", "-path", "./titan-docs/node_modules/*",
             "-not", "-path", "./test_env/*",
             "-not", "-path", "./.git/*",
             "-not", "-path", "*/__pycache__/*"],
            capture_output=True, text=True, timeout=5)
        fs_files = [l for l in fs_result.stdout.strip().split("\n") if l]
        fs_new = [f for f in fs_files if not any(f.endswith(gf) or gf.endswith(f.lstrip("./"))
                                                  for gf in file_matches)]
        if fs_new:
            found_any = True
            print(f"\n  FILES (filesystem — not in graph, {len(fs_new)}):")
            for f in fs_new[:20]:
                print(f"    {f}")
    except Exception:
        pass

    # 2. Class / Function definitions
    defs = graph.get("definitions", {})
    def_matches = sorted([d for d in defs.keys() if sym in d.lower()])
    if def_matches:
        found_any = True
        print(f"\n  DEFINITIONS ({len(def_matches)}):")
        for d in def_matches[:40]:
            entries = defs[d]
            # Each definition maps to a list of {file, line, type} dicts
            if not isinstance(entries, list):
                entries = [entries] if isinstance(entries, dict) else []
            for info in entries[:3]:  # up to 3 occurrences per name
                fp = info.get("file", "?") if isinstance(info, dict) else "?"
                ln = info.get("line", "?") if isinstance(info, dict) else "?"
                kind = info.get("type", "?") if isinstance(info, dict) else "?"
                print(f"    {kind:<10s} {d:<50s}  {fp}:{ln}")
        if len(def_matches) > 40:
            print(f"    ... and {len(def_matches)-40} more")

    # 3. Bus message types
    bus = graph.get("bus_wiring", {})
    bus_matches = sorted([k for k in bus.keys() if sym in k.lower()])
    if bus_matches:
        found_any = True
        print(f"\n  BUS MESSAGES ({len(bus_matches)}):")
        for m in bus_matches[:20]:
            pubs = bus[m].get("publishers", [])
            subs = bus[m].get("subscribers", [])
            print(f"    {m}  pubs={len(pubs)}  subs={len(subs)}")

    # 4. Attribute access patterns
    attrs = graph.get("attr_index", {})
    attr_matches = sorted([a for a in attrs.keys() if sym in a.lower()])
    if attr_matches:
        found_any = True
        print(f"\n  ATTRIBUTE ACCESS ({len(attr_matches)}):")
        for a in attr_matches[:20]:
            data = attrs[a]
            r = len(data.get("readers", []))
            w = len(data.get("writers", []))
            print(f"    {a}  readers={r} writers={w}")
        if len(attr_matches) > 20:
            print(f"    ... and {len(attr_matches)-20} more")

    # 5. Config keys in titan_params.toml + config.toml
    config_files = ["titan_plugin/titan_params.toml", "titan_plugin/config.toml"]
    config_matches = []
    for cfg_path in config_files:
        try:
            with open(cfg_path) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if sym in stripped.lower() and (stripped.startswith("[") or "=" in stripped.split("#")[0]):
                        config_matches.append((cfg_path, lineno, stripped.split("#")[0].rstrip()))
        except FileNotFoundError:
            pass
    if config_matches:
        found_any = True
        print(f"\n  CONFIG KEYS ({len(config_matches)}):")
        for cfg, ln, line in config_matches[:20]:
            print(f"    {cfg}:{ln}  {line[:80]}")
        if len(config_matches) > 20:
            print(f"    ... and {len(config_matches)-20} more")

    # 6. Import graph — who imports this module?
    import_graph = graph.get("import_graph", {})
    for imp_key in sorted(import_graph.keys()):
        if sym in imp_key.lower():
            importers = import_graph.get(imp_key, [])
            if importers:
                found_any = True
                print(f"\n  IMPORTED BY ({imp_key} — {len(importers)} importers):")
                for imp in importers[:15]:
                    print(f"    {imp}")
                break  # Only first match

    if not found_any:
        print(f"\n  No matches for '{symbol}' in any index.")
        print(f"  (NOTE: this searches across 6 dimensions. If nothing found here,")
        print(f"   the symbol truly isn't in the codebase. But per 'verify before declare'")
        print(f"   discipline, also check running processes / API / live logs before")
        print(f"   claiming it doesn't exist.)")

    print()
    print("=" * 78)


def run_meta_audit_diagnostics(all_titans: bool = True):
    """Task 3: cross-Titan meta-reasoning observability.

    Pulls /v4/meta-reasoning/audit and prints diversity, monoculture,
    diversity-pressure cadence, per-primitive reward components, INTROSPECT
    health, and contract fire history. The single most important readout
    when judging whether meta-reasoning healing is working — designed to
    give before/after numbers around Task 4 fixes and META-CGN landing.
    """
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("META-REASONING AUDIT — HEALING DYNAMICS")
    print("=" * 78)

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            resp = requests.get(f"{url}/v4/meta-reasoning/audit", timeout=10)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", {})
            if not d or d.get("status"):
                print(f"    {d.get('status', 'No data')}")
                continue

            div = d.get("diversity", {})
            mono = d.get("monoculture", {})
            dp = d.get("diversity_pressure", {})
            rpp = d.get("rewards_per_primitive", {})
            ih = d.get("introspect_health", {})
            contracts = d.get("contracts", {})

            # Diversity
            ema = div.get("unique_prims_ema_50chains", 0)
            eps = div.get("current_epsilon", 0)
            ema_health = "✓" if ema >= 3.0 else ("⚠" if ema >= 2.0 else "✗")
            print(f"    DIVERSITY  unique_prims_ema={ema:.2f} {ema_health}  "
                  f"ε-greedy={eps:.2f}  recent={div.get('unique_prims_per_chain_recent', [])[-10:]}")

            # Monoculture
            dom = mono.get("dominant_primitive", "?")
            dom_share = mono.get("dominant_share_500", 0)
            adj_n = mono.get("mono_adj_fires_lifetime", 0)
            adj_cum = mono.get("mono_adj_cumulative", 0)
            mono_health = "✓" if dom_share < 0.50 else ("⚠" if dom_share < 0.80 else "✗")
            print(f"    MONOCULTURE  dominant={dom}@{dom_share*100:.0f}% {mono_health}  "
                  f"mono_adj_fires={adj_n} cum={adj_cum:+.2f}")

            # Diversity pressure
            dp_active = "ACTIVE" if dp.get("active") else "idle"
            dp_target = dp.get("target", "")
            dp_rem = dp.get("remaining_chains", 0)
            dp_total = dp.get("total_fires_lifetime", 0)
            print(f"    DIV.PRESSURE  {dp_active}  target={dp_target}  remaining={dp_rem}  "
                  f"contract_fires={dp_total}")
            # Task 4 P1: in-engine check telemetry
            ie = dp.get("inengine_check", {})
            ie_fires = ie.get("fires_lifetime", 0)
            ie_since = ie.get("chains_since_last_fire")
            since_str = f"{ie_since} chains ago" if ie_since is not None else "never fired"
            print(f"      In-engine check: thr={ie.get('threshold', 0):.2f}  "
                  f"mag={ie.get('magnitude', 0):.2f}  decay={ie.get('decay_chains', 0)}  "
                  f"fires={ie_fires}  last={since_str}")

            # Last few diversity pressure fires
            fh = dp.get("fire_history", [])
            if fh:
                print(f"      Recent fires:")
                import datetime as _dt
                for f in fh[-3:]:
                    ts = _dt.datetime.fromtimestamp(f.get("ts", 0)).strftime("%m-%d %H:%M:%S")
                    print(f"        {ts}  chain#{f.get('chain', 0):<6d}  "
                          f"target={f.get('target', '?'):<12s}  "
                          f"mag={f.get('magnitude', 0):.2f}  "
                          f"decay={f.get('decay_chains', 0)}")

            # Per-primitive reward averages
            if rpp:
                print(f"    REWARDS_PER_PRIM (avg over last 100 occurrences)")
                for prim in sorted(rpp.keys(), key=lambda p: -rpp[p].get("avg_total", 0)):
                    info = rpp[prim]
                    print(f"      {prim:<12s} avg={info.get('avg_total', 0):.4f} "
                          f"n={info.get('n', 0)}  "
                          f"comps={info.get('components', {})}")

            # INTROSPECT health
            picks = ih.get("picks_lifetime", 0)
            execs = ih.get("executions_lifetime", 0)
            rerouted = ih.get("rerouted_lifetime", 0)
            unlocked = ih.get("introspect_unlocked", False)
            # Invariant: picks ≤ executions + rerouted (rerouted is by-design
            # gate behavior: max 1/chain, cooldown, gate not met)
            ih_health = "✓" if ih.get("fix_healthy", picks <= execs + rerouted) else "✗"
            print(f"    INTROSPECT  picks={picks}  executions={execs}  "
                  f"rerouted={rerouted} {ih_health}  unlocked={unlocked}")

            # Contracts
            print(f"    CONTRACTS")
            for cname, cinfo in contracts.items():
                fires = cinfo.get("fires_lifetime", 0)
                print(f"      {cname:<32s} fires={fires}")

            # P3: subsystem signals dead/live status (compound rewards bottleneck)
            ss = d.get("subsystem_signals_status", {})
            if ss and "error" not in ss:
                live_n = ss.get("live_count", 0)
                total_n = ss.get("total_signals", 0)
                health = "✓" if live_n >= total_n // 2 else ("⚠" if live_n >= 3 else "✗")
                print(f"    SUBSYSTEM_SIGNALS  {live_n}/{total_n} live {health}  "
                      f"(dead → minority primitives starve)")
                if ss.get("dead"):
                    print(f"      dead: {', '.join(ss['dead'][:8])}"
                          f"{'...' if len(ss['dead']) > 8 else ''}")

            # META-CGN stub
            mc = d.get("meta_cgn", {})
            print(f"    META-CGN  status={mc.get('status', '?')}  "
                  f"templates_grounded={mc.get('templates_grounded', 0)}")

        except requests.exceptions.ConnectionError:
            print(f"    ✗ Not reachable")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    print()
    print("=" * 78)


def run_metabolism_gates(all_titans: bool = False, json_mode: bool = False):
    """Mainnet Lifecycle Wiring rFP (2026-04-20) — 10-min gate decision snapshot.

    Hits /v4/metabolism/gate-status on selected Titans and renders a table of
    gates_enforced + tier + per-caller decision totals + recent closures.
    Used during 30-min focused observation pre-enforcement flip and for soak
    verification.
    """
    import requests
    import json as _json

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    payloads: dict[str, dict] = {}

    for tid, url in titans:
        try:
            resp = requests.get(f"{url}/v4/metabolism/gate-status", timeout=8)
            if resp.status_code == 200:
                payloads[tid] = resp.json().get("data", {})
            else:
                payloads[tid] = {"error": f"HTTP {resp.status_code}"}
        except requests.exceptions.ConnectionError:
            payloads[tid] = {"error": "not_reachable"}
        except Exception as e:
            payloads[tid] = {"error": str(e)}

    if json_mode:
        print(_json.dumps(payloads, indent=2, default=str))
        return

    print()
    print("METABOLISM GATE DECISIONS — last 10 min")
    print("=" * 78)

    for tid, d in payloads.items():
        print(f"\n  {tid}")
        print("  " + "-" * 72)
        if "error" in d:
            print(f"    ✗ {d['error']}")
            continue
        enforced = d.get("gates_enforced", False)
        tier = d.get("current_tier", "?")
        bal = d.get("sol_balance")
        bal_s = f"{bal:.4f}" if isinstance(bal, (int, float)) else "?"
        total = d.get("total_evaluations", 0)
        w10 = d.get("window_10min_count", 0)
        closures = d.get("window_10min_closures", 0)
        mode = "ENFORCED" if enforced else "observation-only"

        print(f"    tier={tier:11s}  sol={bal_s:>8s}  mode={mode}")
        print(f"    lifetime evaluations={total}  |  10-min window={w10}  closures={closures}")

        by_caller = d.get("by_caller", {})
        if by_caller:
            print(f"    per-caller (10-min):")
            for caller, stats in sorted(by_caller.items(), key=lambda x: -x[1]["total"]):
                feat = stats.get("feature", "?")
                t = stats.get("total", 0)
                c = stats.get("closed", 0)
                print(f"      {caller:22s}  feature={feat:10s}  "
                      f"evals={t:>4d}  closed={c:>4d}")
        else:
            print(f"    (no gate calls yet — wiring may not be active)")

        recent = d.get("recent_closures", [])
        if recent:
            print(f"    recent closures (last {len(recent)}):")
            for cl in recent[-5:]:
                import datetime as _dt
                ts = _dt.datetime.fromtimestamp(cl.get("ts", 0)).strftime("%H:%M:%S")
                print(f"      [{ts}] {cl.get('caller','?'):22s} "
                      f"{cl.get('reason','?')}")

    print()
    print("=" * 78)


def run_cgn_telemetry(all_titans: bool = True):
    """Fetch CGN grounding + consumers + knowledge + HAOV stats from all Titans."""
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("CGN TELEMETRY — CONCEPT GROUNDING NETWORK")
    print("=" * 78)

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            # ── 1. Language Grounding ──
            resp = requests.get(f"{url}/v4/language-grounding", timeout=10)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", {})

            total = d.get("total_words", 0)
            prod = d.get("producible", 0)
            grounded = d.get("grounded", 0)
            rate = d.get("grounding_rate", 0)
            avg_conf = d.get("avg_confidence", 0)
            avg_xm = d.get("avg_grounding_confidence", 0)
            types = d.get("word_types", {})

            print(f"    LANGUAGE GROUNDING")
            print(f"    Vocabulary:  {total} total | {prod} producible | "
                  f"{grounded} grounded ({rate*100:.1f}%)")
            print(f"    Confidence:  avg={avg_conf:.3f} | grounding_avg={avg_xm:.3f}")
            type_str = ", ".join(f"{k}={v}" for k, v in sorted(types.items(),
                                key=lambda x: -x[1]))
            if type_str:
                print(f"    Word types:  {type_str}")

            top = d.get("top_grounded", [])
            if top:
                print(f"    Top grounded ({len(top)}):")
                for w in top[:5]:
                    ctx_str = " | ".join(w.get("contexts", [])[:2])
                    assoc_str = ", ".join(
                        f"{a['word']}({a['type']})"
                        for a in w.get("associations", [])[:3])
                    print(f"      {w['word']:14s} xm={w['cross_modal_conf']:.2f}  "
                          f"enc={w.get('encounters', 0):3d}  "
                          f"conf={w.get('confidence', 0):.2f}")
                    if ctx_str:
                        print(f"        contexts: {ctx_str}")
                    if assoc_str:
                        print(f"        assocs:   {assoc_str}")
            else:
                print("    No grounded words yet")

            # ── 2. Knowledge Concepts ──
            try:
                kresp = requests.get(f"{url}/v4/knowledge-stats", timeout=8)
                if kresp.status_code == 200:
                    kd = kresp.json().get("data", {})
                    ktotal = kd.get("total_concepts", 0)
                    kavg_q = kd.get("avg_quality_score", 0)
                    kavg_c = kd.get("avg_confidence", 0)
                    kusage = kd.get("total_usage", 0)
                    ksources = kd.get("source_distribution", {})
                    k24h = kd.get("acquisition_rate_24h", 0)
                    print(f"\n    KNOWLEDGE CONCEPTS")
                    print(f"    Total: {ktotal} | avg_quality={kavg_q:.2f} | "
                          f"avg_conf={kavg_c:.2f} | usage={kusage} | 24h_rate={k24h}")
                    if ksources:
                        src_str = ", ".join(f"{k}={v}" for k, v in ksources.items())
                        print(f"    Sources: {src_str}")
                    ktop = kd.get("top_by_confidence", [])
                    if ktop:
                        for kc in ktop[:3]:
                            print(f"      {kc.get('topic','?'):30s} "
                                  f"conf={kc.get('confidence',0):.2f}  "
                                  f"q={kc.get('quality_score',0):.2f}")
            except Exception:
                pass

            # ── 3. HAOV Hypothesis Testing ──
            try:
                hresp = requests.get(f"{url}/v4/cgn-haov-stats", timeout=8)
                if hresp.status_code == 200:
                    hd = hresp.json().get("data", {}).get("consumers", {})
                    if hd:
                        print(f"\n    HAOV HYPOTHESIS TESTING")
                        for consumer, cdata in hd.items():
                            formed = cdata.get("formed", 0)
                            tested = cdata.get("tested", 0)
                            confirmed = cdata.get("confirmed", 0)
                            falsified = cdata.get("falsified", 0)
                            style = cdata.get("epistemic_style", "?")
                            rules = cdata.get("verified_rules_count", 0)
                            crate = cdata.get("confirmation_rate", 0)
                            print(f"    {consumer:12s}: formed={formed} tested={tested} "
                                  f"confirmed={confirmed} falsified={falsified} "
                                  f"rules={rules} rate={crate:.0%} style={style}")
                            top_rules = cdata.get("top_rules", [])
                            for r in top_rules[:2]:
                                print(f"      rule: {r['rule'][:60]:60s} "
                                      f"conf={r['confidence']:.2f}")
            except Exception:
                pass

            print(f"\n    ✓ CGN active")

        except requests.exceptions.ConnectionError:
            print(f"    ✗ Not reachable")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    print()
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════
# rFP_phase1_sensory_wiring — V6 5DT outer_body diagnostic (2026-04-23)
# ═══════════════════════════════════════════════════════════════════════

_OUTER_BODY_DIM_NAMES = [
    "interoception",   # SOL + block_delta + anchor_freshness
    "proprioception",  # peer_entropy + helper_health + bus_module_diversity
    "somatosensation", # TX_latency + creation_nudge + cpu_spike_rate
    "entropy",         # ping_variance + bus_drop_rate + error_rate
    "thermal",         # cpu_thermal + circadian + llm_latency
]

_OUTER_MIND_FEELING_NAMES = [
    "social_temperature",  # [5]
    "social_connection",   # [6] (twin kin resonance)
    "network_weather",     # [7]
    "environmental_rhythm",# [8]
    "info_flow",           # [9]
]


def run_cgn_reasoning_strategies(all_titans: bool = False):
    """List top grounded reasoning_strategy concepts per Titan.

    The `reasoning_strategy` CGN consumer is fed by reasoning.py:1821-1832
    on COMMIT actions where outcome_score >= cgn_emission_threshold
    (default 0.55, see titan_params.toml [reasoning_rewards]). Each emission
    grounds the chain_signature (sequence of primitives) along with
    outcome_score / confidence / gut_agreement / chain_length / state_embedding
    / neuromod_state. Surfaces the diagnostic question:
    "what does this Titan consider good reasoning?"

    Diagnostic only — pure observability, no runtime side effects. Reads
    /v4/cgn-haov-stats per Titan (the canonical HAOV aggregate endpoint
    that already returns the consumer's `top_rules` list).
    """
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("CGN REASONING_STRATEGY — top grounded chain signatures per Titan")
    print("=" * 78)
    print(f"  Producer: reasoning.py COMMIT with outcome_score >= 0.55 (titan_params.toml)")
    print(f"  Consumer: cgn_worker.py pre-registered 'reasoning_strategy' (cgn.py)")

    for tid, url in titans:
        print(f"\n  {tid} ({url.split('//')[1]})")
        print("  " + "-" * 72)
        try:
            resp = requests.get(f"{url}/v4/cgn-haov-stats", timeout=10)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                continue
            d = resp.json().get("data", resp.json())
            consumers = d.get("consumers", {}) or {}
            block = consumers.get("reasoning_strategy")
            if not block:
                print("    ✗ reasoning_strategy consumer NOT registered (unexpected)")
                continue

            formed = block.get("formed", 0)
            tested = block.get("tested", 0)
            confirmed = block.get("confirmed", 0)
            falsified = block.get("falsified", 0)
            used = block.get("used_for_action", 0)
            verified = block.get("verified_rules_count", 0)
            conf_rate = block.get("confirmation_rate", 0.0)
            style = block.get("epistemic_style", "?")

            print(f"    HAOV state: formed={formed} tested={tested} "
                  f"confirmed={confirmed} falsified={falsified}")
            print(f"    Confirmation rate: {conf_rate*100:.1f}% | "
                  f"used_for_action: {used} | verified_rules: {verified}")
            print(f"    Epistemic style: {style}")

            top_rules = block.get("top_rules", []) or []
            if top_rules:
                print(f"    Top {len(top_rules)} grounded rules:")
                for i, rule in enumerate(top_rules[:10], 1):
                    if isinstance(rule, dict):
                        sig = rule.get("rule") or rule.get("signature") or rule.get("chain", "?")
                        score = rule.get("score") or rule.get("v_value") or rule.get("confidence", 0)
                        cnt = rule.get("count") or rule.get("samples", 0)
                        print(f"      {i:2d}. score={score:.3f}  n={cnt}  rule={sig}")
                    else:
                        print(f"      {i:2d}. {rule}")
            else:
                # Diagnostic: explain the empty state
                if formed == 0:
                    print(f"    ⚠ No rules formed yet — reasoning is emitting candidates "
                          f"(chains with outcome_score >= 0.55) but the consumer hasn't")
                    print(f"      synthesized them into HAOV-grounded rules.")
                    print(f"      Likely: HAOV synthesis cadence not reached, or candidate")
                    print(f"      stream too sparse. Check brain log for emit attempts:")
                    print(f"        grep 'cgn_reasoning_strategy' /tmp/titan_brain.log | wc -l")
                else:
                    print(f"    (top_rules empty despite formed={formed} — investigate)")
        except requests.RequestException as e:
            print(f"    ✗ Request error: {e}")
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")

    print()


def _sensor_flag(dim_idx: int, value: float) -> str:
    """Annotate a dim value with health flag for the 5 outer_body dims."""
    # 0.5 exactly across multiple samples = stuck (archaeology finding)
    if abs(value - 0.5) < 0.001 and dim_idx in (3, 4):
        return "⚠ stuck (default)"
    # > 0.97 on dims that historically saturated = possible saturation
    if value > 0.97 and dim_idx in (0, 1, 2):
        return "⚠ near-saturated"
    if 0.3 <= value <= 0.85:
        return "✓ rich"
    if value < 0.1 or value > 0.95:
        return "◯ extreme"
    return "○ ok"


def run_sensors_diagnostics(all_titans: bool = True):
    """V6 5DT outer_body rich-signal verification.

    Polls /v3/trinity on each Titan, displays outer_body 5D values with
    V6 semantic labels + health annotations (rich / near-saturated /
    stuck). Also shows outer_mind Feeling [5:10] since [7][8][9] depend
    on body[3][4] being rich.

    Run after Phase 1 sensory wiring deploy to verify rich signal across
    all 5 outer_body dims.
    """
    import requests

    titans = [
        ("T1", "http://127.0.0.1:7777"),
        ("T2", "http://10.135.0.6:7777"),
        ("T3", "http://10.135.0.6:7778"),
    ] if all_titans else [("T1", "http://127.0.0.1:7777")]

    print()
    print("OUTER BODY 5D RICH-SIGNAL DIAGNOSTIC (V6 5DT)")
    print("=" * 86)
    print()
    print("  Archaeology baseline (pre-Phase 1 sensory wiring):")
    print("    outer_body [3] entropy / [4] thermal → stuck at 0.5 (no producer)")
    print("    outer_body [0] [1] [2]                → saturated near 1.0")
    print("  This view verifies the fix landed and signal is rich.")
    print()

    # Layout: left column dim name + value + flag; right column samples+stats
    for tid, url in titans:
        host = url.split("//")[1]
        print(f"  {tid} ({host})")
        print("  " + "-" * 82)
        try:
            resp = requests.get(f"{url}/v3/trinity", timeout=8)
            if resp.status_code != 200:
                print(f"    ✗ API error: HTTP {resp.status_code}")
                print()
                continue
            d = resp.json().get("data", {})

            ob = d.get("outer_body", [])
            om = d.get("outer_mind", [])

            if not ob or len(ob) < 5:
                print("    ✗ No outer_body tensor in response")
                print()
                continue

            print(f"    OUTER BODY (5D V6 composites):")
            for i, name in enumerate(_OUTER_BODY_DIM_NAMES):
                v = ob[i] if i < len(ob) else 0.0
                flag = _sensor_flag(i, v)
                print(f"      [{i}] {name:16s} {v:.4f}  {flag}")

            # Outer mind feeling octave [5:10] — depends on body[3][4]
            if om and len(om) >= 10:
                print(f"    OUTER MIND FEELING [5:10] (expected rich post-Phase 1):")
                for j, name in enumerate(_OUTER_MIND_FEELING_NAMES):
                    v = om[5 + j]
                    # Flag near-0.5-stuck for feeling[7][8][9] (depend on body[3][4])
                    stuck = (7 <= (5 + j) <= 9) and abs(v - 0.5) < 0.02
                    flag = "⚠ stuck (body deriv)" if stuck else ""
                    print(f"      [{5 + j}] {name:22s} {v:.4f}  {flag}")

            # Verdict
            ob3_healthy = abs(ob[3] - 0.5) > 0.01
            ob4_healthy = abs(ob[4] - 0.5) > 0.01
            no_saturated = all(v < 0.97 for v in ob[:3])

            print(f"    VERDICT: "
                  f"body[3] entropy={'✓ rich' if ob3_healthy else '✗ STUCK'} | "
                  f"body[4] thermal={'✓ rich' if ob4_healthy else '✗ STUCK'} | "
                  f"no saturation={'✓' if no_saturated else '✗'}")

        except requests.exceptions.ConnectionError:
            print(f"    ✗ Not reachable")
        except requests.exceptions.Timeout:
            print(f"    ✗ Timeout (API slow)")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        print()

    print("=" * 86)


# ── Microkernel v2 — TitanVM telemetry schema (§1.4a) ────────────────


def run_titanvm_schema() -> int:
    """
    Microkernel v2 Phase A §A.2 §1.4a — verify NeuralReflexNet per-program
    telemetry shape before declaring TITANVM_REGISTERS RegistrySpec.

    Pins the (11, 4) layout we'll use in shm:
      [urgency, fire_count, total_updates, last_loss]

    where urgency comes from neural_nervous_system._all_urgencies (live
    per-tick value computed by NeuralReflexNet.forward()), and the
    other three are direct attributes of the NeuralReflexNet instance.

    NS_PROGRAMS row order is taken from emot_bundle_protocol.py — the
    canonical 11-program tuple already used across the codebase.
    """
    print()
    print("MICROKERNEL v2 — TITANVM_REGISTERS SCHEMA (§A.2 §1.4a verification)")
    print("=" * 90)

    try:
        from titan_plugin.logic.emot_bundle_protocol import NS_PROGRAMS
        from titan_plugin.logic.neural_reflex_net import (
            NeuralReflexNet,
            DEFAULT_INPUT_DIM,
        )
    except Exception as e:
        print(f"  ✗ cannot import NS modules: {e}")
        return 1

    print(f"\n  NS_PROGRAMS (11 canonical, row order):")
    for i, name in enumerate(NS_PROGRAMS):
        print(f"    [{i:>2}]  {name}")

    print(f"\n  Per-program fields written to shm (4 floats per row):")
    print(f"    [0] urgency        — float [0,1] from _all_urgencies dict (per-tick)")
    print(f"    [1] fire_count     — int (cast float32) from net.fire_count (cumulative)")
    print(f"    [2] total_updates  — int (cast float32) from net.total_updates (training)")
    print(f"    [3] last_loss      — float from net.last_loss (most recent training)")

    # Construct a probe net and verify all 4 attributes exist.
    print(f"\n  Probe — instantiating NeuralReflexNet to verify attributes:")
    try:
        probe = NeuralReflexNet(name="probe", input_dim=DEFAULT_INPUT_DIM)
        attrs = ["fire_count", "total_updates", "last_loss"]
        for a in attrs:
            ok = hasattr(probe, a)
            v = getattr(probe, a, None)
            icon = "✓" if ok else "✗"
            print(f"    {icon} probe.{a:<14} = {v!r:<10} (type={type(v).__name__})")
        # urgency is computed on demand — verify forward() exists + signature
        print(f"    ✓ urgency       comes from _all_urgencies dict written by neural_nervous_system.tick()")
        print(f"                    (forward() returns sigmoid output ∈ [0, 1] per program)")
    except Exception as e:
        print(f"    ✗ probe construction failed: {e}")
        return 1

    print(f"\n  Verified spec for state_registry.py:")
    print(f"    TITANVM_REGISTERS = RegistrySpec(")
    print(f"        name='titanvm_registers',")
    print(f"        dtype=np.dtype('<f4'),")
    print(f"        shape=({len(NS_PROGRAMS)}, 4),  # 11 programs × 4 fields")
    print(f"        feature_flag='microkernel.shm_titanvm_enabled',")
    print(f"    )")
    print(f"\n  Total payload bytes: {len(NS_PROGRAMS) * 4 * 4} (= 11 × 4 × float32)")

    print()
    print("=" * 90)
    return 0


# ── Microkernel v2 — API subprocess status (§A.4 / S5) ───────────────


def run_api_status(all_titans: bool = False) -> int:
    """
    Microkernel v2 Phase A §A.4 (S5) — per-titan API subprocess health.

    Reports for each Titan:
      - api_process_separation_enabled flag state (from /v3/health if
        exposed via subsystems; else inferred from socket existence)
      - api Guardian module state (running / stopped / crashed) via
        /v3/health subsystems.api or /v4/layers L3 list
      - kernel_rpc socket /tmp/titan_kernel_{titan_id}.sock — exists +
        permissions (must be 0600 for security)
      - kernel_rpc authkey /tmp/titan_kernel_{titan_id}.authkey — exists
        + permissions (must be 0600)
      - /health responsiveness (proves uvicorn alive + accepting)
      - api_subprocess uptime (from guardian_status.api.uptime)

    For T2/T3 over VPC: uses curl-equivalent HTTP probe; socket+authkey
    inspection is skipped (only T1's local /tmp accessible).
    """
    import time as _time
    import os as _os
    import stat as _stat

    titans = [("T1", "127.0.0.1", 7777)]
    if all_titans:
        titans += [("T2", "10.135.0.6", 7777), ("T3", "10.135.0.6", 7778)]

    print()
    print("MICROKERNEL v2 — API SUBPROCESS STATUS (per Titan)")
    print("=" * 90)

    any_failed = False

    for tid, host, port in titans:
        url = f"http://{host}:{port}"
        print(f"\n  {tid} ({url})")
        print("  " + "-" * 86)

        # 1. API reachable?
        health = _services_get(url, "/health", timeout=8.0)
        if not health:
            print("    ✗ /health unreachable — Titan likely down")
            any_failed = True
            continue
        print("    ✓ API reachable (200 OK)")

        # 2. Guardian state for "api" module
        subs = (health.get("data", {}).get("subsystems", {}) or {})
        guard = (health.get("data", {}).get("v3", {}).get("guardian_status", {}) or {})
        api_module = guard.get("api")
        if api_module is None:
            # No "api" Guardian module = legacy in-process mode
            print("    ℹ api_subprocess NOT registered — "
                  "legacy in-process uvicorn mode (api_process_separation_enabled=False)")
            print(f"    ✓ subsystems.observatory={subs.get('observatory', '?')} "
                  "(in-process)")
            continue

        # 3. Subprocess mode active
        state = api_module.get("state", "?")
        pid = api_module.get("pid", "?")
        rss = api_module.get("rss_mb", 0)
        uptime = api_module.get("uptime", 0)
        layer = api_module.get("layer", "?")
        restarts = api_module.get("restart_count", 0)
        hb_age = api_module.get("last_heartbeat_age", -1)

        icon = "✓" if state == "running" else "✗"
        print(f"    {icon} api_subprocess: state={state}  pid={pid}  rss={rss:.1f}MB  "
              f"uptime={uptime:.0f}s  layer={layer}  restarts={restarts}  "
              f"heartbeat_age={hb_age:.1f}s")
        if state != "running":
            any_failed = True

        # 4. Cache staleness — pulls /v4/cache-staleness; was added by D1
        # amendment 2026-04-26 to track per-key age of api_subprocess
        # CachedState. Expected: bootstrap_done=True, max_age < 5s during
        # active operation. Anything stale/cold = bus subscriber lag or
        # kernel snapshot publisher disabled.
        cache = _services_get(url, "/v4/cache-staleness", timeout=5.0)
        if cache and isinstance(cache.get("data"), dict):
            cd = cache["data"]
            if cd.get("available"):
                bb = cd.get("buckets", {})
                bootstrap = "✓" if cd.get("bootstrap_done") else "✗"
                max_age = cd.get("max_age_seconds", -1)
                key_count = cd.get("key_count", 0)
                age_icon = ("✓" if max_age < 5
                            else "⚠" if max_age < 30
                            else "✗")
                print(f"    {age_icon} cache: bootstrap={bootstrap}  keys={key_count}  "
                      f"max_age={max_age:.2f}s  "
                      f"fresh={bb.get('fresh', 0)} warm={bb.get('warm', 0)} "
                      f"stale={bb.get('stale', 0)} cold={bb.get('cold', 0)}")
                if not cd.get("bootstrap_done") or max_age > 30:
                    any_failed = True
            else:
                print(f"    ℹ cache: {cd.get('note', 'not available')}")
        else:
            print("    ⚠ /v4/cache-staleness unreachable "
                  "(endpoint absent — pre-2026-04-26 build?)")

        # 5. Local socket inspection (T1 only)
        if tid == "T1":
            from pathlib import Path as _Path
            try:
                from titan_plugin.core.kernel_rpc import (
                    kernel_sock_path, kernel_authkey_path)
                # We need the actual titan_id of T1. Try data/titan_identity.json
                import json as _json
                tid_path = _Path("data/titan_identity.json")
                t1_id = "T1"
                if tid_path.exists():
                    with open(tid_path) as f:
                        t1_id = _json.load(f).get("titan_id", "T1")
                sock_p = kernel_sock_path(t1_id)
                authkey_p = kernel_authkey_path(t1_id)
                for label, p in (("socket", sock_p), ("authkey", authkey_p)):
                    if not p.exists():
                        print(f"    — kernel_rpc {label} <missing>: {p}")
                        continue
                    mode = _stat.S_IMODE(_os.stat(p).st_mode)
                    perm_icon = "✓" if mode == 0o600 else "⚠"
                    print(f"    {perm_icon} kernel_rpc {label}: {p}  perms={oct(mode)}"
                          f"{' (expected 0o600)' if mode != 0o600 else ''}")
                    if mode != 0o600:
                        any_failed = True
            except Exception as e:
                print(f"    ⚠ kernel_rpc local inspection error: {e}")
        else:
            print("    ℹ kernel_rpc socket inspection: T1-local only "
                  f"(remote {tid} can't read /tmp)")

    print()
    print("=" * 90)
    return 0 if not any_failed else 1


# ── Microkernel v2 — S5 endpoint callsite drift detection (§A.4 / D4) ──

def run_s5_callsites(verbose: bool = False) -> int:
    """Microkernel v2 Phase A §A.4 D4 — drift audit for endpoint-side
    `plugin.X` patterns. Bypasses-the-accessor count must stay near zero;
    rising count = new code being written without StateAccessor migration
    (Phase B/C will fail to migrate cleanly).

    Wraps scripts/s5_callsite_audit.py — runs the libcst categorizer over
    titan_plugin/api/ and reports per-file A/B/C split. Returns exit code
    1 if any Category B (async cross-process) callsites exist (those are
    the highest-risk regressions — they break in microkernel mode).
    """
    import subprocess as _sp

    repo = Path(__file__).resolve().parents[1]
    audit_script = repo / "scripts" / "s5_callsite_audit.py"
    if not audit_script.exists():
        print(f"  ✗ {audit_script} missing — D4 tooling not available")
        return 2

    print()
    print("MICROKERNEL v2 — S5 ENDPOINT CALLSITE AUDIT (drift detection)")
    print("=" * 90)
    print()

    cmd = [sys.executable, str(audit_script),
           "--paths", "titan_plugin/api/", "--summary"]
    res = _sp.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=60)
    out = res.stdout.strip()
    print(out)
    if res.stderr.strip() and verbose:
        print("--- stderr ---")
        print(res.stderr.strip())

    # Read the CSV to count B (async cross-proc) — those FAIL in microkernel
    csv_path = repo / "titan-docs" / "s5_callsite_inventory.csv"
    b_count = 0
    if csv_path.exists():
        try:
            with open(csv_path) as f:
                next(f)  # header
                for row in f:
                    parts = row.rstrip("\n").split(",")
                    # classification is column index 4 — but quoted strings can
                    # have commas. Use the last 3 columns as the canonical
                    # tail (classification, suggested_target, reason).
                    if len(parts) >= 7:
                        cls = parts[-3].strip()
                        if cls == "B":
                            b_count += 1
        except Exception as e:
            print(f"  ⚠ CSV parse error: {e}")

    print()
    if b_count > 0:
        print(f"  ✗ DRIFT: {b_count} Category B (async cross-process) callsite(s) "
              "— these BREAK in microkernel mode")
        print(f"    Inspect: cat {csv_path} | grep ',B,'")
        print("=" * 90)
        return 1

    print("  ✓ No Category B drift (async cross-process accessors all migrated)")
    print(f"    Full inventory: {csv_path}")
    print("=" * 90)
    return 0


# ── Microkernel v2 — Module start_method audit (§A.3 / S6) ──────────


def run_module_methods(all_titans: bool = False) -> int:
    """
    Microkernel v2 Phase A §A.3 (S6) — per-module spawn vs fork audit.

    Reports:
      - spawn_reference_worker_enabled flag state per Titan
      - Each Guardian module's start_method (default "fork", or "spawn"
        for opted-in workers) — read from /v3/health guardian_status
        when exposed; falls back to /v4/layers if guardian_status doesn't
        carry the field
      - RSS per worker (for fork→spawn savings comparison post-flip)

    Note: start_method is a ModuleSpec field set at register time. It's
    not currently exposed via /v3/health. Until that wiring lands (a
    small follow-up), this command reports the known opt-in state via
    config inspection instead. After flag flip on a Titan, the audit
    will show "spawn (configured via flag)" for the reference worker.
    """
    titans = [("T1", "127.0.0.1", 7777)]
    if all_titans:
        titans += [("T2", "10.135.0.6", 7777), ("T3", "10.135.0.6", 7778)]

    print()
    print("MICROKERNEL v2 — MODULE start_method AUDIT (per Titan)")
    print("=" * 90)

    any_failed = False

    for tid, host, port in titans:
        url = f"http://{host}:{port}"
        print(f"\n  {tid} ({url})")
        print("  " + "-" * 86)

        health = _services_get(url, "/health", timeout=8.0)
        if not health:
            print("    ✗ /health unreachable — Titan likely down")
            any_failed = True
            continue

        guard = (health.get("data", {}).get("v3", {}).get("guardian_status", {}) or {})
        if not guard:
            print("    ✗ guardian_status not present in /health")
            any_failed = True
            continue

        # spawn_reference_worker_enabled flag — read from local config (T1 only)
        spawn_ref_flag = "?"
        if tid == "T1":
            try:
                from titan_plugin.config_loader import load_titan_config
                cfg = load_titan_config()
                spawn_ref_flag = cfg.get("microkernel", {}).get(
                    "spawn_reference_worker_enabled", False)
            except Exception:
                spawn_ref_flag = "?"
            print(f"    spawn_reference_worker_enabled = {spawn_ref_flag}")
        else:
            print(f"    (config flag inspection: T1-local only — flip via deploy_t2.sh)")

        # Per-module rundown
        print(f"    {'module':<20}  {'state':<10}  {'rss_mb':<8}  {'layer':<5}  inferred_method")
        for name in sorted(guard.keys()):
            m = guard[name]
            state = m.get("state", "?")
            rss = m.get("rss_mb", 0)
            layer = m.get("layer", "?")
            # Inference: if name == "backup" and spawn_ref_flag is True, method=spawn
            # else method=fork (default). Once start_method is wired into
            # /v3/health, this becomes a direct read instead of inference.
            if name == "backup" and spawn_ref_flag is True:
                method = "spawn (flag)"
            else:
                method = "fork (default)"
            print(f"    {name:<20}  {state:<10}  {rss:<8.1f}  {layer:<5}  {method}")

    print()
    print("=" * 90)
    return 0 if not any_failed else 1


# ── rFP_observatory_data_loading_v1 Phase 1 — cache-keys subcommand ─

def run_cache_keys_command(
    mode: str = "audit",
    all_titans: bool = False,
    json_mode: bool = False,
) -> int:
    """Dispatcher for `arch_map cache-keys [--audit|--status|--list] [--all] [--json]`.

    Returns process exit code (0 = clean, 1 = errors found).
    """
    # Lazy import — registry lives in titan_plugin/, so this only loads
    # when the subcommand is invoked.
    try:
        from titan_plugin.api import cache_key_registry as ckr
    except Exception as e:
        print(f"ERROR: cannot import cache_key_registry: {e}", file=sys.stderr)
        return 2

    if mode == "list":
        return _cache_keys_list(ckr, json_mode=json_mode)
    if mode == "status":
        return _cache_keys_status(ckr, all_titans=all_titans, json_mode=json_mode)
    return _cache_keys_audit(ckr, json_mode=json_mode)


def _cache_keys_list(ckr, json_mode: bool = False) -> int:
    if json_mode:
        out = [
            {
                "key": s.key,
                "kind": s.kind,
                "producer_event": s.producer_event,
                "producer_module": s.producer_module,
                "publish_cadence_s": s.publish_cadence_s,
                "consumer_endpoints": list(s.consumer_endpoints),
                "frontend_hook": s.frontend_hook,
            }
            for s in ckr.REGISTRY
        ]
        print(json.dumps(out, indent=2))
        return 0

    print()
    print("=" * 100)
    print(f"  CACHE KEY REGISTRY — {len(ckr.REGISTRY)} entries")
    print("=" * 100)
    for kind in ("hybrid", "bus_event", "snapshot", "missing", "deprecated"):
        entries = ckr.specs_by_kind(kind)
        if not entries:
            continue
        print()
        print(f"  [{kind.upper()}]  {len(entries)} entries")
        print("  " + "-" * 96)
        for s in entries:
            ev = s.producer_event or ""
            hook = s.frontend_hook or ""
            print(f"    {s.key:<40s}  ev={ev:<32s}  hook={hook}")
    print()
    return 0


def _cache_keys_audit(ckr, json_mode: bool = False) -> int:
    """Static audit — checks REGISTRY against source tree.

    Errors (cause non-zero exit):
      • bus_event/hybrid producer_event missing from titan_plugin.bus
      • bus_event/hybrid producer_module missing _send_msg(... event ...) call
      • snapshot producer_module not found in source tree
      • REGISTRY key duplicated
      • cache.get('X') in api/* code with X not in REGISTRY and not allowlisted
      • EVENT_TO_CACHE_KEY hand-maintained drift (defensive — should be derived)

    Warnings (no exit-code impact):
      • consumer_endpoints not found in dashboard.py
      • frontend_hook not found in useTitanAPI.ts
      • missing-kind entries (no producer wired) — listed for visibility
    """
    import re
    import importlib

    repo_root = Path(__file__).resolve().parent.parent
    api_dir = repo_root / "titan_plugin" / "api"
    modules_dir = repo_root / "titan_plugin" / "modules"
    core_dir = repo_root / "titan_plugin" / "core"
    dashboard_path = api_dir / "dashboard.py"
    hooks_path = repo_root / "titan-observatory" / "hooks" / "useTitanAPI.ts"

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    # 1. Duplicate keys
    seen: set[str] = set()
    for s in ckr.REGISTRY:
        if s.key in seen:
            errors.append(f"DUPLICATE KEY: {s.key!r}")
        seen.add(s.key)

    # 2. Bus constants exist in titan_plugin.bus
    try:
        bus_module = importlib.import_module("titan_plugin.bus")
    except Exception as e:
        errors.append(f"Cannot import titan_plugin.bus: {e}")
        bus_module = None

    if bus_module is not None:
        for s in ckr.REGISTRY:
            if s.producer_event is None:
                continue
            if not hasattr(bus_module, s.producer_event):
                errors.append(
                    f"BUS CONSTANT MISSING: {s.key} declares producer_event="
                    f"{s.producer_event!r} but it is not defined in titan_plugin.bus")

    # 3. Producer modules contain expected publishers / function refs
    #
    # For `bus_event`/`hybrid`: search the producer_module's source file for
    #   _send_msg(... PRODUCER_EVENT ...) — anywhere in the file is OK.
    # For `snapshot`: producer_module must reference an existing function/method
    #   in the snapshot builder's file.
    file_cache: dict[Path, str] = {}

    def _read(p: Path) -> str:
        if p not in file_cache:
            try:
                file_cache[p] = p.read_text()
            except Exception:
                file_cache[p] = ""
        return file_cache[p]

    def _module_to_path(mod_dotted: str) -> Path | None:
        # Split off `:func` or `.func` — keep just the module path.
        # Accept formats: `titan_plugin.modules.spirit_worker._publish_chi`
        #              or `titan_plugin.modules.spirit_worker`
        #              or `titan_plugin.core.kernel.MicroKernel._build_state_snapshot`
        parts = mod_dotted.split(".")
        # Walk from longest prefix down until we find a .py
        for cut in range(len(parts), 0, -1):
            candidate = repo_root / Path(*parts[:cut]).with_suffix(".py")
            if candidate.exists():
                return candidate
        return None

    for s in ckr.REGISTRY:
        if s.kind in ("deprecated", "missing"):
            continue
        if not s.producer_module:
            errors.append(f"NO PRODUCER MODULE: {s.key} (kind={s.kind})")
            continue
        prod_path = _module_to_path(s.producer_module)
        if prod_path is None:
            errors.append(
                f"PRODUCER MODULE FILE NOT FOUND: {s.key} → "
                f"{s.producer_module!r} (no .py on import path)")
            continue
        src = _read(prod_path)
        if s.kind in ("bus_event", "hybrid"):
            # Look for any of these patterns referencing producer_event:
            #   _send_msg(... EVENT ...)
            #   bus.publish(make_msg(EVENT, ...))
            #   self.bus.publish(make_msg(EVENT, ...))
            # Accept either constant name (CHI_UPDATED) or string literal.
            ev = s.producer_event
            patterns = [
                rf"_send_msg\([^)]*\b{re.escape(ev)}\b",
                rf'_send_msg\([^)]*"{re.escape(ev)}"',
                rf"make_msg\(\s*{re.escape(ev)}\b",
                rf'make_msg\(\s*"{re.escape(ev)}"',
            ]
            if not any(re.search(p, src, re.DOTALL) for p in patterns):
                errors.append(
                    f"PRODUCER NOT FOUND IN SOURCE: {s.key} → "
                    f"expected _send_msg(... {ev} ...) or make_msg({ev}, ...) "
                    f"in {prod_path.relative_to(repo_root)}")
        elif s.kind == "snapshot":
            # Verify the snapshot builder file mentions this cache key
            # (kernel._build_state_snapshot writes `snapshot["X"] = ...`).
            patt = re.compile(rf'snapshot\[\s*[\'"]{re.escape(s.key)}[\'"]\s*\]')
            patt_alt = re.compile(rf'snapshot\[f?\s*[\'"][^\'"]*{re.escape(s.key)}')
            patt_attr = re.compile(rf'snapshot\["plugin\.{re.escape(s.key.split(".", 1)[1])}"\]'
                                   ) if s.key.startswith("plugin.") else None
            if not (patt.search(src) or patt_alt.search(src) or
                    (patt_attr and patt_attr.search(src))):
                # Some snapshot keys are written via the f-string `snapshot[f"plugin.{attr}"]`
                # loop in kernel.py. Allow that pattern too.
                if s.key.startswith("plugin."):
                    attr = s.key.split(".", 1)[1]
                    if f'"{attr}"' in src and "snapshot[f" in src:
                        continue
                errors.append(
                    f"SNAPSHOT KEY NOT WRITTEN: {s.key} → expected "
                    f"snapshot[\"{s.key}\"] in {prod_path.relative_to(repo_root)}")

    # 4. Reverse check — every cache.get(LITERAL) must resolve to REGISTRY or allowlist.
    cache_get_pat = re.compile(r'cache\.get\(\s*[\'"]([^\'"]+)[\'"]')
    scan_dirs = [api_dir, modules_dir, core_dir]
    # cache_key_registry.py defines the registry itself — its docstring
    # examples reference cache.get() literally and must be excluded from
    # the reverse scan.
    skip_files = {api_dir / "cache_key_registry.py"}
    found_keys: dict[str, list[tuple[Path, int]]] = {}
    for d in scan_dirs:
        for py in d.rglob("*.py"):
            if py in skip_files:
                continue
            try:
                lines = py.read_text().splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, start=1):
                for m in cache_get_pat.finditer(line):
                    k = m.group(1)
                    found_keys.setdefault(k, []).append((py, lineno))

    for k, sites in found_keys.items():
        if k in ckr.REGISTERED_KEYS:
            continue
        if ckr.is_allowlisted(k):
            continue
        first = sites[0]
        errors.append(
            f"UNREGISTERED CACHE KEY: cache.get({k!r}) at "
            f"{first[0].relative_to(repo_root)}:{first[1]} "
            f"({len(sites)} callsite{'s' if len(sites) > 1 else ''})")

    # 5. Consumer endpoints — WARN only
    if dashboard_path.exists():
        dashboard_src = _read(dashboard_path)
        endpoint_pat = re.compile(r'@router\.(?:get|post)\(\s*[\'"]([^\'"]+)[\'"]')
        declared_endpoints = {m.group(1) for m in endpoint_pat.finditer(dashboard_src)}
        for s in ckr.REGISTRY:
            for ep in s.consumer_endpoints:
                if ep not in declared_endpoints:
                    warnings.append(
                        f"CONSUMER ENDPOINT MISSING: {s.key} → {ep!r} not in dashboard.py")

    # 6. Frontend hooks — WARN only
    if hooks_path.exists():
        hooks_src = _read(hooks_path)
        for s in ckr.REGISTRY:
            if not s.frontend_hook:
                continue
            patt = re.compile(rf'\bexport\s+(?:function|const)\s+{re.escape(s.frontend_hook)}\b')
            if not patt.search(hooks_src):
                warnings.append(
                    f"FRONTEND HOOK MISSING: {s.key} → "
                    f"{s.frontend_hook!r} not in useTitanAPI.ts")
    else:
        warnings.append(f"useTitanAPI.ts not found at {hooks_path} — frontend hook checks skipped")

    # 7. INFO — list missing-kind entries
    for s in ckr.REGISTRY:
        if s.kind == "missing":
            info.append(
                f"MISSING PRODUCER (declared, unwired): {s.key} "
                f"event={s.producer_event} — Phase 4 bring-up target")

    # ── Render ─────────────────────────────────────────────────────
    if json_mode:
        out = {
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "registry_size": len(ckr.REGISTRY),
            "exit_code": 1 if errors else 0,
        }
        print(json.dumps(out, indent=2))
        return 1 if errors else 0

    print()
    print("=" * 100)
    print(f"  CACHE KEYS AUDIT — {len(ckr.REGISTRY)} registry entries")
    print("=" * 100)
    if errors:
        print()
        print(f"  ERRORS ({len(errors)})")
        print("  " + "-" * 96)
        for e in errors:
            print(f"    ✗ {e}")
    if warnings:
        print()
        print(f"  WARNINGS ({len(warnings)})")
        print("  " + "-" * 96)
        for w in warnings:
            print(f"    ⚠ {w}")
    if info:
        print()
        print(f"  INFO — Phase 4 bring-up targets ({len(info)})")
        print("  " + "-" * 96)
        for i in info:
            print(f"    ℹ {i}")
    print()
    if not errors:
        print("  ✓ AUDIT CLEAN — no producer drift, no unregistered cache.get sites")
    else:
        print(f"  ✗ AUDIT FAILED — {len(errors)} errors")
    print("=" * 100)
    print()
    return 1 if errors else 0


def _cache_keys_status(ckr, all_titans: bool = False, json_mode: bool = False) -> int:
    """Live-state matrix per Titan — probes /v4/cache-staleness and reports
    which registered keys are populated and how stale each is."""
    titans = [("T1", "http://127.0.0.1:7777")]
    if all_titans:
        titans.extend([
            ("T2", "http://10.135.0.6:7777"),
            ("T3", "http://10.135.0.6:7778"),
        ])

    results = {}
    for label, url in titans:
        info = _unwrap(_health_get(url, "/v4/cache-staleness"))
        if not info:
            results[label] = {"reachable": False, "keys": {}}
            continue
        # Endpoint shape: {available, bootstrap_done, key_count, max_age_seconds,
        #                  buckets:{fresh,warm,stale,cold}, ages:{key: age_seconds}}
        ages = info.get("ages", {}) if isinstance(info, dict) else {}
        # Normalize to {key: {age_seconds: float}} for the matrix below.
        keys_info = {k: {"age_seconds": v} for k, v in ages.items()}
        results[label] = {"reachable": True, "keys": keys_info}

    if json_mode:
        out = {"results": results, "registry_size": len(ckr.REGISTRY)}
        print(json.dumps(out, indent=2, default=str))
        return 0

    print()
    print("=" * 110)
    print(f"  CACHE KEYS LIVE STATUS — {len(ckr.REGISTRY)} registered keys")
    print("=" * 110)
    for label, _url in titans:
        r = results[label]
        print()
        print(f"  {label}  reachable={r['reachable']}")
        print("  " + "-" * 106)
        if not r["reachable"]:
            print(f"    (cannot reach {_url}/v4/cache-staleness)")
            continue
        live_keys = r["keys"]
        # Header
        print(f"    {'key':<40s}  {'kind':<10s}  {'cadence':>8s}  {'live age':>10s}  {'status'}")
        for s in ckr.REGISTRY:
            if s.kind == "deprecated":
                continue
            row = live_keys.get(s.key, None)
            if row is None:
                age_str = "—"
                status = "ABSENT" if s.kind != "missing" else "missing-by-design"
            else:
                age = row.get("age_seconds") if isinstance(row, dict) else None
                if age is None:
                    age_str = "?"
                    status = "(no age)"
                else:
                    age_str = f"{age:.1f}s"
                    if s.publish_cadence_s > 0 and age > s.publish_cadence_s * 3:
                        status = "STALE"
                    else:
                        status = "FRESH"
            cadence = f"{s.publish_cadence_s:.1f}s" if s.publish_cadence_s else "—"
            print(f"    {s.key:<40s}  {s.kind:<10s}  {cadence:>8s}  {age_str:>10s}  {status}")
    print()
    print("=" * 110)
    print()
    return 0


if __name__ == "__main__":
    main()
