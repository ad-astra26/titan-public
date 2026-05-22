"""
titan_hcl — Titan's Python package.

The SOLE Python entry point is `titan_HCL`, spawned by the Rust L0 kernel
(`scripts/titan_hcl.py` → `titan_hcl.core.plugin.TitanHCL(kernel)`).

The legacy in-process V2 parent (`titan_hcl/__init__.py:TitanHCL` +
`init_plugin()`, the OpenClaw plugin export) was RETIRED in SPEC v1.47.0
(D-SPEC-109): OpenClaw/MCP integration is discarded (Titans are standalone
agno agents). Its eager imports + the `_LAZY_IMPORTS`/`__getattr__` lazy-export
mechanism were used only by that class and went with it. This module is now a
minimal package init; submodules (`titan_hcl.bus`, `titan_hcl.guardian`,
`titan_hcl.core.plugin`, …) are imported directly where needed.
"""
