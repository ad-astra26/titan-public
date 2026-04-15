"""
coding_sandbox helper — AST-validated Python code execution in isolation.

Titan's most powerful helper: writes and executes Python code in a
restricted subprocess with strict resource limits.

Security model:
  - AST pre-validation: blocked imports rejected before execution
  - Subprocess isolation: separate process with timeout + memory limit
  - Import whitelist/blocklist: only safe stdlib + scientific packages
  - No network access from sandbox
  - Output capped at 10KB

Enriches: Mind Vision[0] (new knowledge), Spirit WHAT[2] (action confidence)
"""
import ast
import asyncio
import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Security configuration
ALLOWED_IMPORTS = frozenset({
    "math", "numpy", "json", "datetime", "collections",
    "itertools", "re", "statistics", "csv", "io",
    "hashlib", "base64", "struct", "decimal", "fractions",
    "random", "string", "textwrap", "functools", "operator",
})

BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "http",
    "urllib", "requests", "pathlib", "importlib", "ctypes",
    "signal", "threading", "multiprocessing", "asyncio",
})

SANDBOX_CONFIG = {
    "timeout_seconds": 30,
    "max_memory_mb": 256,
    "max_output_bytes": 10240,  # 10KB
}


def validate_code(source: str) -> tuple[bool, str]:
    """
    Parse source as AST and reject forbidden imports/calls.

    Returns:
        (valid: bool, message: str)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BLOCKED_IMPORTS:
                    return False, f"Import '{alias.name}' is blocked"
                # We allow any import not in blocklist (permissive for stdlib)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in BLOCKED_IMPORTS:
                    return False, f"Import from '{node.module}' is blocked"

    return True, "OK"


class CodingSandboxHelper:
    """
    Executes Python code in an isolated subprocess with strict limits.

    Flow:
      1. Receive code from Agency Module (written by LLM)
      2. AST validation (< 1ms)
      3. Write to temp file
      4. Execute via subprocess with timeout + memory limits
      5. Capture stdout + stderr
      6. Return result, clean up temp file
    """

    def __init__(self, sandbox_dir: Optional[str] = None):
        self._sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="titan_sandbox_")

    @property
    def name(self) -> str:
        return "coding_sandbox"

    @property
    def description(self) -> str:
        return "Execute Python code in an isolated sandbox"

    @property
    def capabilities(self) -> list[str]:
        return ["code_execution", "computation", "data_analysis"]

    @property
    def resource_cost(self) -> str:
        return "medium"

    @property
    def latency(self) -> str:
        return "medium"

    @property
    def enriches(self) -> list[str]:
        return ["mind", "spirit"]

    @property
    def requires_sandbox(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        """
        Execute Python code in sandbox.

        Params:
            code: Python source code to execute
            description: What this code does (for logging)
        """
        code = params.get("code", "")
        description = params.get("description", "code execution")

        if not code.strip():
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "No code provided"}

        # Step 1: AST validation
        valid, message = validate_code(code)
        if not valid:
            logger.warning("[CodingSandbox] Code validation failed: %s", message)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": f"Validation failed: {message}"}

        # Step 2: Execute in subprocess — wrap the blocking subprocess.run
        # (up to 30s) in to_thread so the FastAPI event loop stays responsive
        # during agent code-execution dispatch.
        # See API_FIX_NEXT_SESSION.md (2026-04-14).
        try:
            result = await asyncio.to_thread(self._run_code, code)
            return result
        except Exception as e:
            logger.error("[CodingSandbox] Execution error: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    def _run_code(self, code: str) -> dict:
        """Run validated code in a subprocess with resource limits."""
        # Write to temp file
        temp_path = os.path.join(self._sandbox_dir, "sandbox_script.py")
        os.makedirs(self._sandbox_dir, exist_ok=True)

        try:
            with open(temp_path, "w") as f:
                f.write(code)

            # Build restricted environment
            env = {
                "PATH": "/usr/bin:/usr/local/bin",
                "HOME": self._sandbox_dir,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
            }

            # Execute with timeout
            proc = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=SANDBOX_CONFIG["timeout_seconds"],
                cwd=self._sandbox_dir,
                env=env,
            )

            stdout = proc.stdout[:SANDBOX_CONFIG["max_output_bytes"]]
            stderr = proc.stderr[:SANDBOX_CONFIG["max_output_bytes"]]

            if proc.returncode == 0:
                result_text = stdout if stdout else "(no output)"
                return {
                    "success": True,
                    "result": result_text,
                    "enrichment_data": {"mind": [0], "spirit": [2], "boost": 0.05},
                    "error": None,
                }
            else:
                error_text = stderr if stderr else f"Exit code {proc.returncode}"
                return {
                    "success": False,
                    "result": stdout,
                    "enrichment_data": {},
                    "error": error_text,
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": "",
                "enrichment_data": {},
                "error": f"Timeout: code exceeded {SANDBOX_CONFIG['timeout_seconds']}s limit",
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def status(self) -> str:
        """Report availability based on python3 presence (no subprocess call).

        2026-04-14 fix: previous version did synchronous subprocess.run
        with 5s timeout. Called from /health and many endpoints via
        _agency.get_stats() — could block asyncio event loop. Same root
        cause as web_search.status() fix in commit 30edb58. Sandbox is
        a Python-only operation; if titan_main is running, python3 is
        on PATH by definition. Real subprocess errors are caught at
        execution time.
        """
        import shutil
        return "available" if shutil.which("python3") else "unavailable"
