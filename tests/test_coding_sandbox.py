"""Tests for Step 7.7 — CodingSandboxHelper (AST-validated code execution)."""
import pytest
from titan_plugin.logic.agency.helpers.coding_sandbox import (
    CodingSandboxHelper, validate_code, ALLOWED_IMPORTS, BLOCKED_IMPORTS,
)
from titan_plugin.logic.agency.registry import BaseHelper


class TestASTValidation:
    """AST pre-check: import blocking and syntax validation."""

    def test_clean_code_passes(self):
        valid, msg = validate_code("import math\nprint(math.sqrt(16))")
        assert valid is True
        assert msg == "OK"

    def test_blocked_import_os(self):
        valid, msg = validate_code("import os\nos.system('rm -rf /')")
        assert valid is False
        assert "os" in msg

    def test_blocked_import_subprocess(self):
        valid, msg = validate_code("import subprocess\nsubprocess.run(['ls'])")
        assert valid is False
        assert "subprocess" in msg

    def test_blocked_import_from(self):
        valid, msg = validate_code("from pathlib import Path")
        assert valid is False
        assert "pathlib" in msg

    def test_blocked_nested_import(self):
        valid, msg = validate_code("import os.path")
        assert valid is False
        assert "os" in msg

    def test_allowed_import_math(self):
        valid, _ = validate_code("import math\nprint(math.pi)")
        assert valid is True

    def test_allowed_import_json(self):
        valid, _ = validate_code("import json\nprint(json.dumps({'a': 1}))")
        assert valid is True

    def test_syntax_error_rejected(self):
        valid, msg = validate_code("def foo(\n  invalid syntax here")
        assert valid is False
        assert "Syntax error" in msg

    def test_empty_code_passes(self):
        valid, _ = validate_code("")
        assert valid is True

    def test_blocked_from_import_urllib(self):
        valid, msg = validate_code("from urllib.request import urlopen")
        assert valid is False
        assert "urllib" in msg


class TestCodingSandboxProtocol:
    """BaseHelper protocol compliance."""

    def test_implements_protocol(self):
        helper = CodingSandboxHelper()
        assert isinstance(helper, BaseHelper)
        assert helper.name == "coding_sandbox"
        assert "code_execution" in helper.capabilities
        assert "mind" in helper.enriches
        assert "spirit" in helper.enriches
        assert helper.requires_sandbox is True

    def test_status_available(self):
        helper = CodingSandboxHelper()
        assert helper.status() == "available"


class TestCodingSandboxExecution:
    """Code execution in sandbox subprocess."""

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        helper = CodingSandboxHelper()
        result = await helper.execute({"code": "print(2 + 2)"})
        assert result["success"] is True
        assert "4" in result["result"]
        assert result["enrichment_data"].get("mind") is not None

    @pytest.mark.asyncio
    async def test_math_computation(self):
        helper = CodingSandboxHelper()
        result = await helper.execute({
            "code": "import math\nprint(math.factorial(10))"
        })
        assert result["success"] is True
        assert "3628800" in result["result"]

    @pytest.mark.asyncio
    async def test_empty_code_rejected(self):
        helper = CodingSandboxHelper()
        result = await helper.execute({"code": ""})
        assert result["success"] is False
        assert "No code" in result["error"]

    @pytest.mark.asyncio
    async def test_blocked_import_rejected(self):
        helper = CodingSandboxHelper()
        result = await helper.execute({"code": "import os\nprint(os.getcwd())"})
        assert result["success"] is False
        assert "Validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self):
        helper = CodingSandboxHelper()
        result = await helper.execute({"code": "print(1/0)"})
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    @pytest.mark.asyncio
    async def test_output_captured(self):
        helper = CodingSandboxHelper()
        code = "for i in range(5): print(f'line {i}')"
        result = await helper.execute({"code": code})
        assert result["success"] is True
        assert "line 0" in result["result"]
        assert "line 4" in result["result"]

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        helper = CodingSandboxHelper()
        # Infinite loop — should timeout
        result = await helper.execute({"code": "while True: pass"})
        assert result["success"] is False
        assert "Timeout" in result["error"]


class TestHelperRegistration:
    """Verify CodingSandbox registers in the registry."""

    def test_register_in_registry(self):
        from titan_plugin.logic.agency.registry import HelperRegistry
        registry = HelperRegistry()
        helper = CodingSandboxHelper()
        registry.register(helper)
        assert "coding_sandbox" in registry.list_all_names()
        assert registry.get_status("coding_sandbox") == "available"
